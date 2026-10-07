"""
Symmetry plane detection + NCSD using the FULL point cloud (15k points for ShapeNet PC15k).

- The detector receives all points (no FPS downsampling).
- NCSD is computed on all points ("full").
- Optionally the same planes are also evaluated on a 2048-point subsample ("n2048")
  so the numbers remain comparable with previous runs / the paper.

Usage:
    python evaluate_symmetry_planes_full.py --dataset_root data/ShapeNetCore.v2.PC15k --category plane --max_objects 50
    python evaluate_symmetry_planes_full.py --dataset_root ... --check_chamfer   # sanity check of the Chamfer implementation
"""
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
from ChamferDistance import chamfer_distance as protocol_chamfer

CATEGORIES = {
    "plane": "02691156",
    "car": "02958343",
    "chair": "03001627",
}

BASE_STRATEGIES = ["baseline_x0_orig", "baseline_x0"]
DET_STRATEGIES = ["detector_top1", "closest_yz", "closest_yz_top3", "oracle"]


# ----------------------------------------------------------------------------
# Chamfer
# ----------------------------------------------------------------------------
def _nn_dist(a, b, chunk=2048):
    """For each point in a, euclidean distance to its nearest neighbour in b (exact, chunked)."""
    out = []
    for i in range(0, a.shape[0], chunk):
        d = torch.cdist(a[i:i + chunk], b, compute_mode="donot_use_mm_for_euclid_dist")
        out.append(d.min(dim=1).values)
    return torch.cat(out)


def chamfer_torch(a_np, b_np, device, squared, combine):
    """Exact chunked GPU Chamfer. squared: bool. combine: 'sum' (a->b + b->a) or 'mean' (average of both)."""
    a = torch.as_tensor(a_np, dtype=torch.float32, device=device)
    b = torch.as_tensor(b_np, dtype=torch.float32, device=device)
    d_ab, d_ba = _nn_dist(a, b), _nn_dist(b, a)
    if squared:
        d_ab, d_ba = d_ab ** 2, d_ba ** 2
    s = d_ab.mean() + d_ba.mean()
    return float(s if combine == "sum" else s / 2)


def make_chamfer(args, device):
    if args.chamfer == "protocol":
        return lambda a, b: float(protocol_chamfer(a, b))
    return lambda a, b: chamfer_torch(a, b, device, args.cd_squared, args.cd_combine)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def compute_diagonal(points):
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def reflect_points_across_plane(points, normal, midpoint):
    normal = normal / np.linalg.norm(normal)
    v = points - midpoint
    projection = np.dot(v, normal)
    return points - 2 * np.outer(projection, normal)


def normalize_points(points, mode):
    """'unit': bbox-center + scale to unit sphere. 'none': raw. Returns (pts, center, scale)."""
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
    return points[rng.choice(points.shape[0], n, replace=False)]


# ----------------------------------------------------------------------------
# Candidates and strategies
# ----------------------------------------------------------------------------
def extract_candidates(normals, midpoints, scores):
    """Detector output -> list of dicts sorted by score (score = 1 - cd/threshold, higher is better)."""
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
            "offset": float(abs(np.dot(m, n))),
        })
    cands.sort(key=lambda c: -c["score"])
    for rank, c in enumerate(cands):
        c["rank"] = rank
    return cands


def select_strategies(cands, cand_ncsd):
    if not cands:
        return {}
    idx = list(range(len(cands)))
    by_angle = lambda i: cands[i]["angle_to_x_deg"]
    return {
        "detector_top1": cand_ncsd[0],
        "closest_yz": cand_ncsd[min(idx, key=by_angle)],
        "closest_yz_top3": cand_ncsd[min(idx[:3], key=by_angle)],
        "oracle": min(cand_ncsd),
    }


def evaluate_set(points, cands, origin_in_frame, cd):
    """NCSD of baselines and every candidate on one evaluation point set."""
    diagonal = max(compute_diagonal(points), 1e-6)

    def ncsd(normal, midpoint):
        normal = np.asarray(normal, dtype=np.float64)
        return float(cd(points, reflect_points_across_plane(points, normal, np.asarray(midpoint, dtype=np.float64))) / diagonal)

    x_axis = np.array([1.0, 0.0, 0.0])
    cand_ncsd = [ncsd(c["normal"], c["midpoint"]) for c in cands]
    strat = {
        "baseline_x0_orig": ncsd(x_axis, origin_in_frame),  # x=0 of the original coordinates
        "baseline_x0": ncsd(x_axis, np.zeros(3)),           # x=0 of the normalized frame
    }
    strat.update(select_strategies(cands, cand_ncsd))
    return {"n_points": int(points.shape[0]), "diagonal": float(diagonal),
            "candidate_ncsd": cand_ncsd, "ncsd": {k: float(v) for k, v in strat.items()}}


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def _stats(vals):
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "n": len(vals)} if vals else None


def summarize(records, set_name):
    recs = list(records.values())
    found = [r for r in recs if r["found"]]
    out = {"n_total": len(recs), "n_found": len(found), "common_subset": {}, "all_objects": {}}
    for k in BASE_STRATEGIES + DET_STRATEGIES:
        out["common_subset"][k] = _stats([r["eval"][set_name]["ncsd"][k] for r in found])
    for k in BASE_STRATEGIES:
        out["all_objects"][k] = _stats([r["eval"][set_name]["ncsd"][k] for r in recs])
    out["all_objects"]["detector_top1_fallback_x0"] = _stats(
        [r["eval"][set_name]["ncsd"].get("detector_top1", r["eval"][set_name]["ncsd"]["baseline_x0_orig"]) for r in recs])
    return out


def print_summary(cat_name, set_name, s):
    print(f"\n=== {cat_name} [{set_name}]: {s['n_found']}/{s['n_total']} objects with >=1 plane ===")
    print(f"{'strategy (common subset)':32s} {'mean':>9s} {'median':>9s}")
    for k, v in s["common_subset"].items():
        if v:
            print(f"{k:32s} {v['mean']:9.5f} {v['median']:9.5f}")
    fb = s["all_objects"]["detector_top1_fallback_x0"]
    if fb:
        print(f"{'detector_top1_fallback_x0 (all)':32s} {fb['mean']:9.5f} {fb['median']:9.5f}")


# ----------------------------------------------------------------------------
def check_chamfer(files, args, device):
    """Compare the protocol's Chamfer with the 4 torch variants on x=0 reflection of one object."""
    pts = np.load(files[0]).astype(np.float32)
    pts, _, _ = normalize_points(pts, args.normalize)
    refl = reflect_points_across_plane(pts, np.array([1.0, 0, 0]), np.zeros(3)).astype(np.float32)
    print(f"Object: {files[0].name}, {pts.shape[0]} points")
    try:
        print(f"protocol chamfer_distance : {float(protocol_chamfer(pts, refl)):.6f}")
    except Exception as e:
        print(f"protocol chamfer_distance failed on full cloud: {e!r}")
    for squared in (False, True):
        for combine in ("sum", "mean"):
            v = chamfer_torch(pts, refl, device, squared, combine)
            print(f"torch squared={squared!s:5s} combine={combine:4s}: {v:.6f}")
    print("Pick the torch variant matching the protocol value and pass it with --chamfer torch "
          "[--cd_squared] --cd_combine {sum,mean}.")


def main():
    parser = argparse.ArgumentParser(description="Symmetry planes + NCSD on the full point cloud")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="ncsd_results_full.json")
    parser.add_argument("--max_objects", type=int, default=1000)
    parser.add_argument("--category", type=str, default="all", choices=["all", "plane", "car", "chair"])
    parser.add_argument("--normalize", type=str, default="unit", choices=["unit", "none"])
    parser.add_argument("--max_points", type=int, default=0,
                        help="0 = use all points for detection/evaluation. >0 = random cap (debug / memory).")
    parser.add_argument("--extra_eval_n", type=int, default=2048,
                        help="Also evaluate the same planes on this many points, for comparison with previous runs. 0 = off.")
    parser.add_argument("--extra_eval_sampling", type=str, default="fps", choices=["fps", "random"])
    parser.add_argument("--chamfer", type=str, default="protocol", choices=["protocol", "torch"],
                        help="'protocol': ChamferDistance from the protocol repo. 'torch': exact chunked GPU version.")
    parser.add_argument("--cd_squared", action="store_true", help="(torch chamfer) use squared distances")
    parser.add_argument("--cd_combine", type=str, default="sum", choices=["sum", "mean"], help="(torch chamfer)")
    parser.add_argument("--check_chamfer", action="store_true", help="Compare Chamfer implementations on one object and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    categories = CATEGORIES if args.category == "all" else {args.category: CATEGORIES[args.category]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.check_chamfer:
        first_cat = next(iter(categories.values()))
        files = sorted((Path(args.dataset_root) / first_cat / "train").glob("*.npy"))
        check_chamfer(files, args, device)
        return

    cd = make_chamfer(args, device)
    print(f"Loading DINO model on {device}...")
    model = DINOWrapper(device=device, small=True, reg=True)

    output = {"args": vars(args), "objects": {}, "summary": {}, "errors": {}}

    for cat_name, cat_id in categories.items():
        print(f"\nProcessing category: {cat_name} ({cat_id})")
        cat_path = Path(args.dataset_root) / cat_id / "train"
        if not cat_path.exists():
            print(f"Path not found: {cat_path}")
            continue

        files = sorted(cat_path.glob("*.npy"))[:args.max_objects]
        print(f"Found {len(files)} objects for {cat_name}")

        cat_records, cat_errors = {}, {}

        for idx, file_path in enumerate(tqdm(files, desc=cat_name)):
            try:
                seed = args.seed + idx
                np.random.seed(seed)
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)

                # 1. Load + normalize. ALL points are used from here on.
                raw = np.load(file_path).astype(np.float32)
                if raw.ndim != 2 or raw.shape[1] != 3:
                    raise ValueError(f"unexpected shape {raw.shape}")
                if args.max_points > 0 and raw.shape[0] > args.max_points:
                    raw = raw[rng.choice(raw.shape[0], args.max_points, replace=False)]
                points, center, scale = normalize_points(raw, args.normalize)
                origin_in_frame = (-center * scale).astype(np.float64)

                # 2. Detector on the full cloud
                det_tensor = torch.tensor(points, dtype=torch.float32, device=device)
                ctx = contextlib.nullcontext() if args.verbose else contextlib.redirect_stdout(io.StringIO())
                with torch.no_grad(), ctx:
                    features = compute_features(det_tensor, model, device, view_quantity=114)
                    normals, midpoints, scores = compute_planes(det_tensor, features, device)
                del features, det_tensor
                cands = extract_candidates(normals, midpoints, scores)

                # 3. Evaluate on the full cloud (+ optional reference subsample), same planes
                eval_sets = {"full": points}
                if args.extra_eval_n > 0 and points.shape[0] > args.extra_eval_n:
                    eval_sets[f"n{args.extra_eval_n}"] = subsample(points, args.extra_eval_n, args.extra_eval_sampling, rng)
                evals = {name: evaluate_set(pts, cands, origin_in_frame, cd) for name, pts in eval_sets.items()}

                rec = {
                    "found": len(cands) > 0,
                    "n_candidates": len(cands),
                    "n_points": int(points.shape[0]),
                    "center": center.tolist(),
                    "scale": float(scale),
                    "candidates": cands,
                    "eval": evals,
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
            found = [r for r in cat_records.values() if r["found"]]
            output["summary"][cat_name] = {"n_errors": len(cat_errors)}
            for set_name in next(iter(cat_records.values()))["eval"].keys():
                s = summarize(cat_records, set_name)
                print_summary(cat_name, set_name, s)
                output["summary"][cat_name][set_name] = s
            axes = dict(Counter(r["top1_axis"] for r in found))
            output["summary"][cat_name]["top1_dominant_axis_counts"] = axes
            print(f"top1 dominant axis counts: {axes}")
        output["objects"].update({f"{cat_name}/{k}": {**v, "category": cat_name} for k, v in cat_records.items()})
        output["errors"][cat_name] = cat_errors

        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()