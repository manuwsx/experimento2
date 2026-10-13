#!/usr/bin/env python3
"""
Verificaciones del experimento 1 (sección 5 del enunciado), para correr ANTES de la corrida larga.

CPU (no necesitan GPU):
  reflection    nube sintética simétrica respecto a un plano oblicuo que no pasa por el origen:
                NSCD ≈ 0 con ese plano y > 0 con YZ; identidades de la reflexión (Householder, -n, involución,
                forma Hp + t del paper) y lo mismo pasando por el pipeline completo (centrado X + FPS 2048).
  invariance    NSCD invariante a traslación + escala uniforme (llevando el plano), incluido el marco
                normalizado del detector; con nubes sintéticas y reales (si hay datos).
  paper_script  nscd_yz de este código == el que da Symmetry_NormalizedPCs.py ORIGINAL (subproceso) sobre los
                mismos archivos (sintéticos siempre; reales si hay datos).
  paper_json    nscd_yz vs generativos/Replicability/computed_NSCD/shapenet_<cat>.json objeto por objeto.
  data_stats    forma, dtype, escala y centrado de los .npy (n por categoría).
GPU:
  batch         --planes_batch_size: mismas features, compute_planes con batch 5 (repo) vs otros -> ¿idéntico?
                + tiempos y VRAM por objeto (base para estimar la corrida completa).
  determinism   detector completo dos veces sobre el mismo objeto -> ¿idéntico?
  frames        NSCD con el plano en coords originales == NSCD en el marco normalizado del detector.

Ejemplo:  python verify.py --checks all
          python verify.py --checks cpu            # solo las de CPU
          python verify.py --checks batch --planes_batch_sizes 16 64 256
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

import sym_nscd as nscd
import sym_summary as summ
from sym_imports import (CATEGORIES, DEFAULT_DATA_ROOT, DEFAULT_DETECTOR_ROOT, DEFAULT_NSCD_ROOT,
                         DEFAULT_PAPER_JSON_DIR, import_detector, load_nscd_protocol, resolve_cli_path)

HERE = Path(__file__).resolve().parent
CPU_CHECKS = ["reflection", "invariance", "paper_script", "paper_json", "data_stats"]
GPU_CHECKS = ["batch", "determinism", "frames"]


def say(msg=""):
    print(msg, flush=True)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# ----------------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------------
def sorted_files(args, category, n):
    split_dir = args.data_root / CATEGORIES[category] / args.split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob("*.npy"), key=lambda p: p.name)[:n]


def synthetic_symmetric(n_half, normal, point, rng):
    """Mitad asimétrica (dos blobs anisótropos) + su reflejo respecto a (normal, point)."""
    half = np.concatenate([
        rng.normal(size=(n_half // 2, 3)) * [0.35, 0.12, 0.5] + [0.4, 0.1, 0.0],
        rng.normal(size=(n_half - n_half // 2, 3)) * [0.1, 0.3, 0.1] + [0.15, -0.2, 0.3],
    ])
    return np.concatenate([half, nscd.reflect_across_plane(half, normal, point)])


OBLIQUE_N = np.array([0.8, 0.5, -0.33]) / np.linalg.norm([0.8, 0.5, -0.33])
OBLIQUE_M = np.array([0.3, -0.2, 0.5])


# ----------------------------------------------------------------------------------------------
# CPU
# ----------------------------------------------------------------------------------------------
def check_reflection(protocol, args):
    rng = np.random.default_rng(0)
    n, m = OBLIQUE_N, OBLIQUE_M
    S = synthetic_symmetric(1000, n, m, rng)
    diag = nscd.aabb_diagonal(S)
    exact_det = nscd.nscd_of_reflection(S, nscd.reflect_across_plane(S, n, m), protocol.chamfer_distance, diag)["nscd"]
    exact_yz = nscd.nscd_of_reflection(S, protocol.householder_transformation(S), protocol.chamfer_distance, diag)["nscd"]
    H = np.eye(3) - 2 * np.outer(n, n)
    t = 2 * np.outer(n, n) @ m
    ident = {
        "reflect([1,0,0], 0) == householder_transformation (bit a bit)":
            bool(np.array_equal(nscd.reflect_across_plane(S, [1, 0, 0], [0, 0, 0]), protocol.householder_transformation(S))),
        "reflect(n) == reflect(-n)": bool(np.allclose(nscd.reflect_across_plane(S, n, m), nscd.reflect_across_plane(S, -n, m), atol=1e-12)),
        "involución R(R(p)) == p": bool(np.allclose(nscd.reflect_across_plane(nscd.reflect_across_plane(S, n, m), n, m), S, atol=1e-12)),
        "forma del paper Hp + t": bool(np.allclose(S @ H.T + t, nscd.reflect_across_plane(S, n, m), atol=1e-12)),
    }
    # Pipeline completo: 15k puntos, plano oblicuo (coords originales), centrado X + FPS 2048 del paper
    S15 = synthetic_symmetric(7500, n, m, rng)
    ev = nscd.evaluate_object(S15, protocol, n[None], m[None], top_k=1, nscd_points=args.nscd_points)
    # Piso de muestreo del FPS: objeto del mismo tipo simétrico respecto a x = 0.05, medido con YZ
    S_yz = synthetic_symmetric(7500, np.array([1.0, 0, 0]), np.array([0.05, 0, 0]), rng)
    floor = nscd.evaluate_object(S_yz, protocol, np.zeros((0, 3)), np.zeros((0, 3)), 0, args.nscd_points)["yz"]["nscd"]
    det_fps, yz_fps = ev["per_rank"][0]["nscd"], ev["yz"]["nscd"]
    ok = exact_det < 1e-12 and exact_yz > 1e-3 and all(ident.values()) and det_fps < 0.25 * yz_fps
    return ("PASS" if ok else "FAIL"), {
        "nube_exacta_2000pts": {"nscd_plano_oblicuo": exact_det, "nscd_yz": exact_yz},
        "identidades": ident,
        "pipeline_15k_fps": {"nscd_det_plano_verdadero": det_fps, "nscd_yz": yz_fps,
                             "piso_muestreo_fps (obj. simétrico en YZ)": floor, "x_shift": ev["x_shift"]},
    }


def _invariance_cases(protocol, args, S, n, m, label):
    base = nscd.evaluate_object(S, protocol, n[None], m[None], 1, args.nscd_points)
    c = S.mean(axis=0)
    s = np.linalg.norm(S - c, axis=1).max()
    cases = [(2.5, np.array([0.3, -1.2, 0.7])), (0.1, np.array([5.0, 0.0, -3.0])),
             (37.0, np.array([-0.002, 0.0004, 0.001])), (1.0 / s, -c / s)]  # el último = marco del detector
    out = []
    for a, t in cases:
        ev = nscd.evaluate_object(a * S + t, protocol, n[None], (a * m + t)[None], 1, args.nscd_points)
        out.append({"objeto": label, "escala": a, "traslacion": t.tolist(),
                    "rel_nscd_yz": rel(ev["yz"]["nscd"], base["yz"]["nscd"]),
                    "rel_nscd_det": rel(ev["per_rank"][0]["nscd"], base["per_rank"][0]["nscd"]),
                    "rel_nscd_sq_yz": rel(ev["yz"]["nscd_sq"], base["yz"]["nscd_sq"]),
                    "rel_nscd_sq_det": rel(ev["per_rank"][0]["nscd_sq"], base["per_rank"][0]["nscd_sq"])})
    return out


def check_invariance(protocol, args):
    rng = np.random.default_rng(1)
    S = synthetic_symmetric(7500, OBLIQUE_N, OBLIQUE_M, rng)
    S[:600] += [0.05, 0.02, 0.0]   # algo de asimetría para que el NSCD no sea ~0
    rows = _invariance_cases(protocol, args, S, OBLIQUE_N, OBLIQUE_M, "sintético")
    for category in args.categories:
        for f in sorted_files(args, category, 2):
            P = np.load(f).astype(np.float64)
            tilt = np.radians(3.0)
            rows += _invariance_cases(protocol, args, P, np.array([np.cos(tilt), np.sin(tilt), 0.0]),
                                      P.mean(axis=0), f"{category}/{f.stem}")
    worst = max(max(r[k] for k in r if k.startswith("rel_")) for r in rows)
    return ("PASS" if worst < 1e-6 else "FAIL"), {"max_rel_diff": worst, "casos": rows}


def _run_original_script(protocol, files, workdir: Path):
    data_dir = workdir / "npy"
    data_dir.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, data_dir / f.name)
    out_json = workdir / "original.json"
    proc = subprocess.run([sys.executable, str(protocol.original_script), str(data_dir), str(out_json), "npy"],
                          cwd=str(protocol.root), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Symmetry_NormalizedPCs.py falló:\n{proc.stderr[-2000:]}")
    with open(out_json, encoding="utf-8") as fh:
        return json.load(fh)


def _compare_with_original(protocol, args, files, label):
    with tempfile.TemporaryDirectory(prefix="verify_nscd_") as tmp:
        original = _run_original_script(protocol, files, Path(tmp))
    rows = []
    for f in files:
        mine = nscd.evaluate_object(np.load(f), protocol, np.zeros((0, 3)), np.zeros((0, 3)), 0, args.nscd_points)["yz"]["nscd"]
        theirs = original[f"{f.name} normalized cd"]
        rows.append({"grupo": label, "archivo": f.name, "nuestro": mine, "original": theirs,
                     "abs_diff": abs(mine - theirs), "identico": mine == theirs})
    return rows


def check_paper_script(protocol, args):
    rows = []
    with tempfile.TemporaryDirectory(prefix="verify_synth_") as tmp:
        rng = np.random.default_rng(2)
        files = []
        for i, x_off in enumerate([0.0, 0.0004, -0.03, 0.2, -0.5]):   # cubre |media_x| < 1e-3 y >= 1e-3
            P = synthetic_symmetric(7500, np.array([1.0, 0, 0]), np.zeros(3), rng)
            P[:900] += [0.04, 0.0, 0.01]
            P[:, 0] += x_off - P[:, 0].mean()
            path = Path(tmp) / f"synthetic_{i}.npy"
            np.save(path, P.astype(np.float32))
            files.append(path)
        rows += _compare_with_original(protocol, args, files, "sintético")
    for category in args.categories:
        files = sorted_files(args, category, args.n_paper)
        if files:
            rows += _compare_with_original(protocol, args, files, category)
    real = [r for r in rows if r["grupo"] != "sintético"]
    status = "PASS" if all(r["identico"] for r in rows) else "FAIL"
    return status, {"n_sinteticos": len(rows) - len(real), "n_reales": len(real),
                    "max_abs_diff": max(r["abs_diff"] for r in rows),
                    "nota": "" if real else "sin datos reales: solo se comparó con nubes sintéticas", "filas": rows}


def check_paper_json(protocol, args):
    out, worst, n_real = {}, 0.0, 0
    for category in args.categories:
        paper = summ.load_paper_nscd_json(args.paper_json_dir, category)
        if not paper:
            out[category] = "no encontré el JSON"
            continue
        values = np.array(list(paper.values()))
        info = {"n_json": len(paper), "media_json": float(values.mean()), "paper_reporta": nscd.PAPER_NSCD_MEAN[category]}
        split_dir = args.data_root / CATEGORIES[category] / args.split
        present = [i for i in sorted(paper) if (split_dir / f"{i}.npy").is_file()]
        info["ids_del_json_presentes_en_datos"] = len(present)
        rows = []
        for object_id in present[: args.n_paper]:
            raw = np.load(split_dir / f"{object_id}.npy")
            mine = nscd.evaluate_object(raw, protocol, np.zeros((0, 3)), np.zeros((0, 3)), 0, args.nscd_points)["yz"]["nscd"]
            rows.append({"id": object_id, "nuestro": mine, "json": paper[object_id], "abs_diff": abs(mine - paper[object_id])})
        if rows:
            info["max_abs_diff"] = max(r["abs_diff"] for r in rows)
            info["max_rel_diff"] = max(r["abs_diff"] / r["json"] for r in rows)
            info["n_identicos"] = sum(r["abs_diff"] == 0 for r in rows)
            worst = max(worst, info["max_rel_diff"])
            n_real += len(rows)
        info["filas"] = rows
        out[category] = info
    if n_real == 0:
        return "SKIP", {"motivo": "no hay datos para comparar objeto por objeto", **out}
    return ("PASS" if worst < 1e-6 else "FAIL"), {"max_rel_diff": worst, **out}


def check_data_stats(protocol, args):
    out = {}
    for category in args.categories:
        files = sorted_files(args, category, args.n_stats)
        paper = summ.load_paper_nscd_json(args.paper_json_dir, category)
        info = {}
        if files:
            per_file = []
            for f in files:
                P = np.load(f)
                c = P.astype(np.float64).mean(axis=0)
                lo, hi = P.min(axis=0).astype(np.float64), P.max(axis=0).astype(np.float64)
                per_file.append({"id": f.stem, "shape": list(P.shape), "dtype": str(P.dtype),
                                 "min": lo.tolist(), "max": hi.tolist(), "centroide": c.tolist(),
                                 "centro_caja": ((lo + hi) / 2).tolist(), "diag_aabb": float(np.linalg.norm(hi - lo)),
                                 "radio_max_desde_centroide": float(np.linalg.norm(P - c, axis=1).max()),
                                 "paper_centra_en_x": bool(abs(np.mean(P[:, 0])) >= nscd.X_CENTER_TOL)})
            agg = lambda k: [float(np.min([r[k] for r in per_file])), float(np.mean([r[k] for r in per_file])),
                             float(np.max([r[k] for r in per_file]))]
            info.update({"n_archivos": len(files), "shapes": sorted({str(r["shape"]) for r in per_file}),
                         "dtypes": sorted({r["dtype"] for r in per_file}),
                         "diag_aabb_min_media_max": agg("diag_aabb"),
                         "radio_max_min_media_max": agg("radio_max_desde_centroide"),
                         "norma_centroide_max": float(max(np.linalg.norm(r["centroide"]) for r in per_file)),
                         "abs_centroide_por_eje_max": np.max(np.abs([r["centroide"] for r in per_file]), axis=0).tolist(),
                         "frac_que_el_paper_centra_en_x": float(np.mean([r["paper_centra_en_x"] for r in per_file])),
                         "archivos": per_file})
        if paper:
            # "normal cd" / "normalized cd" del JSON = diagonal de la muestra de 2048 pts del paper
            path = args.paper_json_dir / f"shapenet_{category}.json"
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            diags = np.array([raw[f"{i}.npy normal cd"] / raw[f"{i}.npy normalized cd"] for i in paper])
            info["diag_muestra_paper_percentiles_0_5_50_95_100"] = np.percentile(diags, [0, 5, 50, 95, 100]).tolist()
        out[category] = info or "sin datos"
    return "INFO", out


# ----------------------------------------------------------------------------------------------
# GPU
# ----------------------------------------------------------------------------------------------
def run_gpu_checks(protocol, args, wanted):
    import torch
    import sym_detector as sdet

    results = {}
    sdet.configure_determinism(not args.no_deterministic)
    try:
        gpu = sdet.check_device(args.device)
    except Exception as exc:  # noqa: BLE001
        return {c: ("FAIL", {"error": str(exc)}) for c in wanted}
    say(f"  GPU: {gpu.get('gpu_name', args.device)} | VRAM {gpu.get('vram_total_gb', '?')} GB | "
        f"DINO en caché: {not sdet.dino_cache_status()['needs_internet']}")
    det = import_detector(args.detector_root)
    device = torch.device(args.device)
    model, _ = sdet.load_dino(det, device)

    objects = []
    for category in args.categories:
        objects += [(category, f) for f in sorted_files(args, category, 1)]
    objects = objects[: args.n_gpu_objects]
    if not objects:
        return {c: ("SKIP", {"motivo": f"no hay .npy en {args.data_root}"}) for c in wanted}

    sizes = [sdet.REPO_PLANES_BATCH_SIZE] + [b for b in args.planes_batch_sizes if b != sdet.REPO_PLANES_BATCH_SIZE]
    det_kw = dict(view_samp=args.view_samp, view_quantity=args.views, rot_aug=args.rot_aug)
    batch_rows, detections = [], []
    for category, f in objects:
        raw = np.load(f)
        pts_norm, center, scale = sdet.normalize_unit_sphere(raw)
        pts_gpu = torch.from_numpy(pts_norm).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        feats, _ = sdet.detector_features(det, model, pts_gpu, device, args.seed, **det_kw)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_feat = time.perf_counter() - t0
        ref = None
        sizes_to_run = sizes if "batch" in wanted else sizes[:1]
        for bs in sizes_to_run:
            t0 = time.perf_counter()
            normals, mids, scores, _ = sdet.detector_planes(det, pts_gpu, feats, device, args.seed, bs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_planes = time.perf_counter() - t0
            if ref is None:
                ref = (normals, mids, scores)
            same_shape = all(a.shape == b.shape for a, b in zip(ref, (normals, mids, scores)))
            batch_rows.append({
                "objeto": f"{category}/{f.stem}", "batch_size": bs, "n_planos": len(normals),
                "t_features_s": round(t_feat, 2), "t_planes_s": round(t_planes, 2),
                "identico_a_batch_5": same_shape and all(np.array_equal(a, b) for a, b in zip(ref, (normals, mids, scores))),
                "max_abs_diff": (max(float(np.abs(a - b).max(initial=0.0)) for a, b in zip(ref, (normals, mids, scores)))
                                 if same_shape else "distinto número de planos"),
                "vram_peak_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
            })
            say(f"    {batch_rows[-1]}")
        detections.append((category, f, raw, pts_norm, center, scale, ref))
        del feats, pts_gpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "batch" in wanted:
        others = [r for r in batch_rows if r["batch_size"] != sdet.REPO_PLANES_BATCH_SIZE]
        t5 = [r["t_features_s"] + r["t_planes_s"] for r in batch_rows if r["batch_size"] == sdet.REPO_PLANES_BATCH_SIZE]
        results["batch"] = ("PASS" if all(r["identico_a_batch_5"] for r in others) else "FAIL", {
            "filas": batch_rows,
            "t_detect_medio_batch5_s": float(np.mean(t5)),
            "estimacion_3x1000_h_batch5": float(np.mean(t5)) * 3000 / 3600,
        })

    if "determinism" in wanted:
        category, f, raw, *_ = detections[0]
        a = sdet.detect_object(det, model, raw, device, args.seed, batch_size=sdet.REPO_PLANES_BATCH_SIZE, **det_kw)
        b = sdet.detect_object(det, model, raw, device, args.seed, batch_size=sdet.REPO_PLANES_BATCH_SIZE, **det_kw)
        keys = ("normals_norm", "midpoints_norm", "scores")
        same = all(a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in keys)
        results["determinism"] = ("PASS" if same else "FAIL", {
            "objeto": f"{category}/{f.stem}", "identico": same, "n_planos": [len(a["scores"]), len(b["scores"])],
            "nota": "con --no_deterministic se espera que pueda fallar (index_put con índices repetidos en CUDA)"})

    if "frames" in wanted:
        rows = []
        for category, f, raw, pts_norm, center, scale, (normals_n, mids_n, _) in detections:
            if not len(normals_n):
                rows.append({"objeto": f"{category}/{f.stem}", "nota": "0 planos"})
                continue
            normals_o, points_o = nscd.planes_to_original(normals_n, mids_n, center, scale)
            ev = nscd.evaluate_object(raw, protocol, normals_o, points_o, args.top_k, args.nscd_points)
            # (a) misma muestra llevada al marco normalizado
            Sn = (ev["sample"] + [ev["x_shift"], 0.0, 0.0] - center) / scale
            diag_n = nscd.aabb_diagonal(Sn)
            same_sample = [rel(nscd.nscd_of_reflection(Sn, nscd.reflect_across_plane(Sn, normals_n[r], mids_n[r]),
                                                       protocol.chamfer_distance, diag_n)["nscd"], ev["per_rank"][r]["nscd"])
                           for r in range(len(ev["per_rank"]))]
            # (b) pipeline completo sobre la nube normalizada (float32, como la ve el detector)
            ev_n = nscd.evaluate_object(pts_norm, protocol, normals_n, mids_n, args.top_k, args.nscd_points)
            end_to_end = [rel(x["nscd"], y["nscd"]) for x, y in zip(ev_n["per_rank"], ev["per_rank"])]
            rows.append({"objeto": f"{category}/{f.stem}", "max_rel_misma_muestra": max(same_sample),
                         "max_rel_pipeline_completo": max(end_to_end)})
        worst = max((r.get("max_rel_misma_muestra", 0.0) for r in rows), default=0.0)
        results["frames"] = ("PASS" if worst < 1e-9 else "FAIL", {
            "filas": rows, "nota": "pipeline_completo puede diferir algo porque el FPS corre sobre coordenadas "
                                   "float32 normalizadas (empates del FPS); la versión 'misma muestra' es la exacta"})
    return results


# ----------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checks", nargs="+", default=["all"], choices=CPU_CHECKS + GPU_CHECKS + ["all", "cpu", "gpu"])
    p.add_argument("--categories", nargs="+", default=["all"], choices=[*CATEGORIES, "all"])
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--detector_root", type=Path, default=DEFAULT_DETECTOR_ROOT)
    p.add_argument("--nscd_root", type=Path, default=DEFAULT_NSCD_ROOT)
    p.add_argument("--paper_json_dir", type=Path, default=DEFAULT_PAPER_JSON_DIR)
    p.add_argument("--split", default="train")
    p.add_argument("--n_paper", type=int, default=20, help="Objetos por categoría para paper_script / paper_json")
    p.add_argument("--n_stats", type=int, default=20, help="Objetos por categoría para data_stats")
    p.add_argument("--n_gpu_objects", type=int, default=3, help="Objetos (uno por categoría) para batch/frames")
    p.add_argument("--planes_batch_sizes", nargs="+", type=int, default=[16, 64])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--views", type=int, default=6)
    p.add_argument("--rot_aug", type=int, default=3)
    p.add_argument("--view_samp", type=int, default=1)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--nscd_points", type=int, default=2048)
    p.add_argument("--no_deterministic", action="store_true")
    p.add_argument("--out_dir", type=Path, default=HERE / "verify")
    args = p.parse_args()
    args.categories = list(CATEGORIES) if "all" in args.categories else args.categories
    for key in ("data_root", "detector_root", "nscd_root", "paper_json_dir", "out_dir"):
        setattr(args, key, resolve_cli_path(getattr(args, key)))
    wanted = []
    for c in args.checks:
        wanted += {"all": CPU_CHECKS + GPU_CHECKS, "cpu": CPU_CHECKS, "gpu": GPU_CHECKS}.get(c, [c])
    wanted = list(dict.fromkeys(wanted))

    say(f"Python {platform.python_version()} | datos: {args.data_root} (existe: {args.data_root.is_dir()})")
    protocol = load_nscd_protocol(args.nscd_root)
    report = {}
    funcs = {"reflection": check_reflection, "invariance": check_invariance, "paper_script": check_paper_script,
             "paper_json": check_paper_json, "data_stats": check_data_stats}
    for name in [c for c in wanted if c in CPU_CHECKS]:
        say(f"\n--- {name}")
        try:
            report[name] = funcs[name](protocol, args)
        except Exception as exc:  # noqa: BLE001
            report[name] = ("FAIL", {"error": short_tb(exc)})
        say(f"[{report[name][0]}] {name}")
    gpu_wanted = [c for c in wanted if c in GPU_CHECKS]
    if gpu_wanted:
        say(f"\n--- {', '.join(gpu_wanted)} (GPU)")
        try:
            report.update(run_gpu_checks(protocol, args, gpu_wanted))
        except Exception as exc:  # noqa: BLE001
            report.update({c: ("FAIL", {"error": short_tb(exc)}) for c in gpu_wanted})
        for name in gpu_wanted:
            say(f"[{report[name][0]}] {name}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "verify_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({k: {"status": s, "detalle": d} for k, (s, d) in report.items()}, fh, indent=2,
                  ensure_ascii=False, default=str)
    say("\n==== Resumen ====")
    for name, (status, detail) in report.items():
        brief = {k: v for k, v in detail.items() if not isinstance(v, (list, dict))} if isinstance(detail, dict) else detail
        say(f"[{status:4}] {name}: {brief}")
    say(f"\nDetalle completo en {out}")
    return 1 if any(s == "FAIL" for s, _ in report.values()) else 0


def short_tb(exc):
    return f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"


if __name__ == "__main__":
    sys.exit(main())
