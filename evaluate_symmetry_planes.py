import os
import sys
import io
import json
import argparse
import contextlib
from collections import Counter

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add necessary paths
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(EXPERIMENT_DIR, 'detector'))
sys.path.append(os.path.join(EXPERIMENT_DIR, 'generativos', 'Symmetry_Measurement_Protocol'))

from implementation.utils import DINOWrapper
from implementation.compute import compute_features, compute_planes
from FarthestPointSampling import farthest_point_sampling
from ChamferDistance import chamfer_distance

CATEGORIES = {
    "plane": "02691156",
    "car": "02958343",
    "chair": "03001627",
}

# Strategies that depend on the detector (only defined if >=1 candidate was found)
DET_STRATEGIES = ["detector_top1", "closest_yz", "closest_yz_top3", "oracle"]
# Strategies that never depend on the detector
BASE_STRATEGIES = ["baseline_x0_orig", "baseline_x0"]


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def compute_diagonal(points):
    """Bounding box diagonal of the point cloud."""
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def reflect_points_across_plane(points, normal, midpoint):
    """Reflect (N, 3) points across the plane defined by normal and a point on it."""
    normal = normal / np.linalg.norm(normal)
    v = points - midpoint
    projection = np.dot(v, normal)
    return points - 2 * np.outer(projection, normal)


def ncsd_for_plane(points, normal, midpoint, diagonal):
    reflected = reflect_points_across_plane(points, normal, midpoint)
    cd = float(chamfer_distance(points, reflected))
    return cd / diagonal


def normalize_points(points, mode):
    """
    mode='none': raw coordinates.
    mode='unit': center at bbox center and scale so max ||p|| = 1.
    Returns (points_normalized, center, scale). Map raw -> normalized: (p - center) * scale.
    """
    if mode == "none":
        return points, np.zeros(3, dtype=np.float32), 1.0
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    centered = points - center
    scale = 1.0 / np.linalg.norm(centered, axis=1).max()
    return (centered * scale).astype(np.float32), center.astype(np.float32), float(scale)


def subsample(points, n, mode, rng):
    if points.shape[0] <= n:
        return points
    if mode == "fps":
        return farthest_point_sampling(points, n)
    idx = rng.choice(points.shape[0], n, replace=False)
    return points[idx]


# ----------------------------------------------------------------------------
# Candidate handling
# ----------------------------------------------------------------------------
def extract_candidates(normals, midpoints, scores, eval_points, diagonal):
    """
    Converts detector output into a list of dicts sorted by detector score
    (score = 1 - chamfer/threshold, higher is better).
    """
    N = normals.detach().cpu().numpy().reshape(-1, 3)
    M = midpoints.detach().cpu().numpy().reshape(-1, 3)
    S = scores.detach().cpu().numpy().reshape(-1)

    cands = []
    for n, m, s in zip(N, M, S):
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        cands.append({
            "normal": n.tolist(),
            "midpoint": m.tolist(),
            "score": float(s),
            "angle_to_x_deg": float(np.degrees(np.arccos(min(1.0, abs(n[0]))))),
            "dominant_axis": "xyz"[int(np.argmax(np.abs(n)))],
            "offset": float(abs(np.dot(m, n))),  # distance plane -> origin of the frame
            "ncsd": float(ncsd_for_plane(eval_points, n, m, diagonal)),
        })
    cands.sort(key=lambda c: -c["score"])
    for rank, c in enumerate(cands):
        c["rank"] = rank
    return cands


def select_strategies(cands):
    if not cands:
        return {}
    by_angle = lambda c: c["angle_to_x_deg"]
    return {
        "detector_top1": cands[0]["ncsd"],                    # detector's own best plane
        "closest_yz": min(cands, key=by_angle)["ncsd"],       # your previous criterion
        "closest_yz_top3": min(cands[:3], key=by_angle)["ncsd"],
        "oracle": min(c["ncsd"] for c in cands),              # lower bound among candidates
    }


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def _stats(vals):
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "n": len(vals)} if vals else None


def summarize(records):
    recs = list(records.values())
    found = [r for r in recs if r["found"]]
    summary = {
        "n_total": len(recs),
        "n_found": len(found),
        "common_subset": {},   # all strategies on the SAME objects (detector found a plane)
        "all_objects": {},
    }
    for k in BASE_STRATEGIES + DET_STRATEGIES:
        summary["common_subset"][k] = _stats([r["ncsd"][k] for r in found])
    for k in BASE_STRATEGIES:
        summary["all_objects"][k] = _stats([r["ncsd"][k] for r in recs])
    # detector top1 with fallback to the fixed plane when nothing is found
    summary["all_objects"]["detector_top1_fallback_x0"] = _stats(
        [r["ncsd"].get("detector_top1", r["ncsd"]["baseline_x0_orig"]) for r in recs]
    )
    summary["top1_dominant_axis_counts"] = dict(Counter(r["top1_axis"] for r in found))
    if found:
        summary["top1_within_10deg_of_x"] = float(np.mean([r["top1_angle_to_x_deg"] < 10 for r in found]))
    return summary


def print_summary(cat_name, summary):
    print(f"\n=== {cat_name}: {summary['n_found']}/{summary['n_total']} objects with >=1 plane ===")
    print(f"{'strategy (common subset)':32s} {'mean':>9s} {'median':>9s}")
    for k, v in summary["common_subset"].items():
        if v:
            print(f"{k:32s} {v['mean']:9.5f} {v['median']:9.5f}")
    fb = summary["all_objects"]["detector_top1_fallback_x0"]
    print(f"{'detector_top1_fallback_x0 (all)':32s} {fb['mean']:9.5f} {fb['median']:9.5f}")
    print(f"top1 dominant axis counts: {summary['top1_dominant_axis_counts']}")
    if "top1_within_10deg_of_x" in summary:
        print(f"top1 within 10 deg of x-axis: {summary['top1_within_10deg_of_x']:.2%}")


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate symmetry planes and compute NCSD")
    parser.add_argument("--dataset_root", type=str, required=True, help="e.g. data/ShapeNetCore.v2.PC15k")
    parser.add_argument("--output_json", type=str, default="ncsd_results.json")
    parser.add_argument("--max_objects", type=int, default=1000, help="Max objects per category")
    parser.add_argument("--category", type=str, default="all", choices=["all", "plane", "car", "chair"])
    parser.add_argument("--normalize", type=str, default="unit", choices=["unit", "none"],
                        help="'unit': center (bbox) + scale to unit sphere before detecting/evaluating. "
                             "'none': raw coordinates (reproduces the previous run).")
    parser.add_argument("--det_n", type=int, default=10000, help="Points given to the detector")
    parser.add_argument("--eval_n", type=int, default=2048, help="Points used to compute NCSD")
    parser.add_argument("--eval_sampling", type=str, default="fps", choices=["fps", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="Show detector prints")
    args = parser.parse_args()

    categories = CATEGORIES if args.category == "all" else {args.category: CATEGORIES[args.category]}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading DINO model on {device}...")
    model = DINOWrapper(device=device, small=True, reg=True)

    output = {"args": vars(args), "objects": {}, "summary": {}, "errors": {}}

    for cat_name, cat_id in categories.items():
        print(f"\nProcessing category: {cat_name} ({cat_id})")
        cat_path = Path(args.dataset_root) / cat_id / "train"
        if not cat_path.exists():
            print(f"Path not found: {cat_path}")
            continue

        # sorted() so the first N objects are the same on every run / machine
        files = sorted(cat_path.glob("*.npy"))[:args.max_objects]
        print(f"Found {len(files)} objects for {cat_name}")

        cat_records = {}
        cat_errors = {}

        for idx, file_path in enumerate(tqdm(files, desc=cat_name)):
            try:
                seed = args.seed + idx
                np.random.seed(seed)
                torch.manual_seed(seed)  # compute_planes uses torch.randperm
                rng = np.random.default_rng(seed)

                # 1. Load + normalize (everything below lives in the same frame)
                raw = np.load(file_path).astype(np.float32)
                if raw.ndim != 2 or raw.shape[1] != 3:
                    raise ValueError(f"unexpected shape {raw.shape}")
                points, center, scale = normalize_points(raw, args.normalize)
                # where the original x=0 plane ends up in this frame
                origin_in_frame = (-center * scale).astype(np.float64)

                # 2. Subsample: detector always uses FPS (as before)
                det_points = subsample(points, args.det_n, "fps", rng)
                eval_points = subsample(points, args.eval_n, args.eval_sampling, rng)
                diagonal = max(compute_diagonal(eval_points), 1e-6)

                # 3. Detector
                det_tensor = torch.tensor(det_points, dtype=torch.float32, device=device)
                ctx = contextlib.nullcontext() if args.verbose else contextlib.redirect_stdout(io.StringIO())
                with torch.no_grad(), ctx:
                    features = compute_features(det_tensor, model, device, view_quantity=114)
                    normals, midpoints, scores = compute_planes(det_tensor, features, device)
                del features

                # 4. Candidates + all strategies on the SAME eval points
                cands = extract_candidates(normals, midpoints, scores, eval_points, diagonal)
                x_axis = np.array([1.0, 0.0, 0.0])
                ncsd = {
                    # plane x=0 of the ORIGINAL coordinates (previous baseline)
                    "baseline_x0_orig": ncsd_for_plane(eval_points, x_axis, origin_in_frame, diagonal),
                    # plane x=0 of the normalized frame (bbox center if normalize='unit')
                    "baseline_x0": ncsd_for_plane(eval_points, x_axis, np.zeros(3), diagonal),
                }
                ncsd.update(select_strategies(cands))

                rec = {
                    "found": len(cands) > 0,
                    "n_candidates": len(cands),
                    "ncsd": {k: float(v) for k, v in ncsd.items()},
                    "diagonal": float(diagonal),
                    "center": center.tolist(),
                    "scale": float(scale),
                    "candidates": cands,
                }
                if cands:
                    rec["top1_axis"] = cands[0]["dominant_axis"]
                    rec["top1_angle_to_x_deg"] = cands[0]["angle_to_x_deg"]
                cat_records[file_path.name] = rec

            except Exception as e:
                cat_errors[file_path.name] = repr(e)
                print(f"Error processing {file_path.name}: {e}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

        if cat_records:
            summary = summarize(cat_records)
            summary["n_errors"] = len(cat_errors)
            print_summary(cat_name, summary)
            output["summary"][cat_name] = summary
        output["objects"].update({f"{cat_name}/{k}": {**v, "category": cat_name} for k, v in cat_records.items()})
        output["errors"][cat_name] = cat_errors

        # Save after every category so a crash doesn't lose finished work
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
