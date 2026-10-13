#!/usr/bin/env python3
"""
Experimento 1 — Detector de simetrías (tal cual) + NSCD sobre ShapeNetCore.v2.PC15k (airplane, car, chair).

Flujo por objeto:
  .npy (15k pts) -> normalización (centroide + esfera unitaria) -> detector con los 15k puntos
  (compute_features -> compute_planes) -> planos a coordenadas originales -> muestra del paper
  (centrado en X + FPS 2048) -> NSCD-YZ (paper) y NSCD de los top-K planos del detector sobre la
  MISMA muestra -> fila del CSV.

Etapas (--stage):
  all       detect + evaluate por objeto y al final summary (default)
  detect    solo detector; guarda los planos en cache/<categoria>/<id>.npz
  evaluate  solo NSCD a partir de la caché (barato: sirve para cambiar la selección/NSCD sin re-detectar)
  summary   summary.csv + resumen por consola + figuras
Ejemplos de uso en README.md.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # determinismo de cuBLAS; antes de importar torch

import argparse
import csv
import gc
import json
import logging
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import sym_detector as sdet
import sym_nscd as nscd
import sym_summary as summ
from sym_imports import (CATEGORIES, DEFAULT_DATA_ROOT, DEFAULT_DETECTOR_ROOT, DEFAULT_NSCD_ROOT,
                         DEFAULT_PAPER_JSON_DIR, PROJECT_ROOT, import_detector, load_nscd_protocol,
                         resolve_cli_path)

try:
    from tqdm import tqdm
except ImportError:  # sin tqdm se procesa igual, solo sin barra
    tqdm = None

HERE = Path(__file__).resolve().parent
EXPECTED_N_POINTS = 15000
RESUMABLE_STATUSES = {"ok", "no_planes"}  # filas que --resume da por terminadas

RESULT_FIELDS = [
    "category", "category_id", "object_id", "status", "error_msg", "n_points",
    "center_x", "center_y", "center_z", "scale",
    "n_planes_detected", "selection", "chosen_rank", "detector_score",
    "normal_x", "normal_y", "normal_z", "point_x", "point_y", "point_z",
    "angle_to_x_deg", "offset_norm",
    "nscd_yz", "nscd_det", "nscd_det_top1", "nscd_det_bestk", "bestk_rank", "nscd_det_topk",
    "delta_nscd", "delta_nscd_top1", "delta_nscd_bestk",
    "nscd_sq_yz", "nscd_sq_det", "nscd_sq_det_top1", "nscd_sq_det_bestk",
    "nscd_det_fb", "fallback_used",
    "angle_to_x_deg_top1", "angle_to_x_deg_bestk", "min_angle_to_x_topk_deg", "rank_closest_to_x",
    "cd_yz", "cd_det", "diag_nscd", "x_shift_nscd", "n_nscd_points",
    "nscd_yz_paper_json", "diff_yz_vs_paper_json",
    "time_features_s", "time_planes_s", "time_detect_s", "time_eval_s",
    "vram_peak_alloc_mb", "vram_peak_reserved_mb", "from_cache",
]
DETECT_LOG_FIELDS = ["category", "object_id", "status", "error_msg", "n_points", "n_planes_detected",
                     "time_features_s", "time_planes_s", "vram_peak_alloc_mb", "vram_peak_reserved_mb", "timestamp"]
_CACHE_SCALARS = ("scale", "n_points", "n_nonfinite", "time_features_s", "time_planes_s",
                  "vram_peak_alloc_mb", "vram_peak_reserved_mb")

log = logging.getLogger("exp01")


class CacheParamsMismatch(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# CLI y logging
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Detector de simetrías + NSCD en ShapeNetCore.v2.PC15k. Las rutas por defecto se anclan a "
                    "la raíz del proyecto (funciona desde cualquier directorio); las rutas relativas que pases "
                    "explícitamente se interpretan respecto al directorio actual.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["all", "detect", "evaluate", "summary"], default="all")
    p.add_argument("--categories", nargs="+", default=["all"], choices=[*CATEGORIES, "all"])
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--detector_root", type=Path, default=DEFAULT_DETECTOR_ROOT)
    p.add_argument("--nscd_root", type=Path, default=DEFAULT_NSCD_ROOT)
    p.add_argument("--paper_json_dir", type=Path, default=DEFAULT_PAPER_JSON_DIR,
                   help="Carpeta con shapenet_<cat>.json (NSCD por objeto calculados por el paper)")
    p.add_argument("--split", default="train")
    p.add_argument("--n_objects", type=int, default=1000, help="Objetos por categoría")
    p.add_argument("--id_source", choices=["sorted", "paper"], default="sorted",
                   help="sorted: los primeros N .npy en orden alfabético. paper: los objetos que usó el paper "
                        "(claves de shapenet_<cat>.json), en orden alfabético")
    p.add_argument("--top_k", type=int, default=5, help="Planos del detector evaluados con NSCD")
    p.add_argument("--selection", choices=["top1", "bestk"], default="top1",
                   help="Plano que va a nscd_det: top1 = mayor score del detector; bestk = mínimo NSCD entre top-K")
    p.add_argument("--nscd_points", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=HERE / "results")
    p.add_argument("--resume", action="store_true", help="Retomar: reutiliza la caché y salta objetos ya evaluados")
    p.add_argument("--views", type=int, default=sdet.DEFAULT_VIEW_QUANTITY,
                   help="view_quantity de compute_features (default de la función: 6; detector/main.py usa 114)")
    p.add_argument("--rot_aug", type=int, default=sdet.DEFAULT_ROT_AUG, choices=range(5))
    p.add_argument("--view_samp", type=int, default=sdet.DEFAULT_VIEW_SAMP, choices=[0, 1])
    p.add_argument("--planes_batch_size", type=int, default=sdet.REPO_PLANES_BATCH_SIZE,
                   help="batch_size de get_candidate_planes (repo: 5). Solo cambia el costo; verifícalo con verify.py")
    p.add_argument("--no_deterministic", action="store_true",
                   help="No activar torch.use_deterministic_algorithms")
    p.add_argument("--n_figures", type=int, default=5,
                   help="Figuras por categoría: los N con mayor delta_nscd y los N con mayor ángulo (0 = ninguna)")
    p.add_argument("--save_ply", action="store_true", help="Guardar también .ply de las nubes de las figuras")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = p.parse_args(argv)

    args.categories = list(CATEGORIES) if "all" in args.categories else list(dict.fromkeys(args.categories))
    for key in ("data_root", "detector_root", "nscd_root", "paper_json_dir", "out_dir"):
        setattr(args, key, resolve_cli_path(getattr(args, key)))
    if min(args.n_objects, args.top_k, args.nscd_points, args.planes_batch_size) < 1:
        p.error("--n_objects, --top_k, --nscd_points y --planes_batch_size deben ser >= 1")
    if args.view_samp == 1 and args.views < 2:
        p.error("--views debe ser >= 2 con muestreo Fibonacci")
    if args.view_samp == 0 and args.views not in (6, 14, 26, 42, 62, 86, 114):
        p.error("con --view_samp 0, --views debe ser uno de 6, 14, 26, 42, 62, 86, 114")
    return args


class _TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if tqdm is not None:
                tqdm.write(msg, file=sys.stderr)
            else:
                print(msg, file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001
            self.handleError(record)


def setup_logging(out_dir: Path, level: str) -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    console = _TqdmHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    log.setLevel(logging.DEBUG)
    log.addHandler(file_handler)
    log.addHandler(console)
    # Los warnings de Python (p. ej. de use_deterministic_algorithms) van solo al archivo
    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    warn_logger.addHandler(file_handler)
    warn_logger.propagate = False


def progress(items, desc):
    return tqdm(items, desc=desc, dynamic_ncols=True, mininterval=2.0) if tqdm is not None else items


def short(exc: BaseException, limit: int = 500) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:limit]


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _py(value):
    """Escalares numpy -> Python, para que el CSV quede con repr completo."""
    return value.item() if isinstance(value, np.generic) else value


# ----------------------------------------------------------------------------------------------
# Datos, IDs y parámetros
# ----------------------------------------------------------------------------------------------
def load_points(path: Path) -> np.ndarray:
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"forma inesperada {points.shape} (se esperaba (N, 3))")
    if not np.isfinite(points).all():
        raise ValueError("la nube tiene NaN/inf")
    if points.shape[0] != EXPECTED_N_POINTS:
        log.warning("%s tiene %d puntos (se esperaban %d)", path.name, points.shape[0], EXPECTED_N_POINTS)
    return points


def select_object_ids(args, category: str):
    split_dir = args.data_root / CATEGORIES[category] / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"No existe {split_dir}. Revisa --data_root/--split.")
    available = [p.stem for p in sorted(split_dir.glob("*.npy"), key=lambda p: p.name)]
    if args.id_source == "sorted":
        ids = available[: args.n_objects]
    else:
        paper = summ.load_paper_nscd_json(args.paper_json_dir, category)
        if not paper:
            raise SystemExit(f"--id_source paper: no encuentro shapenet_{category}.json en {args.paper_json_dir}")
        present = set(available)
        missing = [i for i in paper if i not in present]
        if missing:
            log.warning("%s: %d objetos del JSON del paper no están en %s", category, len(missing), split_dir)
        ids = sorted(i for i in paper if i in present)[: args.n_objects]
    if len(ids) < args.n_objects:
        log.warning("%s: solo hay %d objetos disponibles (pediste %d)", category, len(ids), args.n_objects)
    return ids, split_dir


def detector_params(args) -> dict:
    """Lo que determina los planos detectados. Se guarda en cada .npz y se exige igual al reutilizar la caché."""
    return {"normalization": sdet.NORMALIZATION, "view_samp": args.view_samp, "view_quantity": args.views,
            "rot_aug": args.rot_aug, "dino_small": sdet.DINO_SMALL, "dino_reg": sdet.DINO_REG,
            "seed": args.seed, "detector_points": "all"}


def evaluation_params(args) -> dict:
    return {"top_k": args.top_k, "selection": args.selection, "nscd_points": args.nscd_points}


# ----------------------------------------------------------------------------------------------
# Caché de planos
# ----------------------------------------------------------------------------------------------
def cache_path(out_dir: Path, category: str, object_id: str) -> Path:
    return out_dir / "cache" / category / f"{object_id}.npz"


def save_cache(path: Path, cache: dict, params: dict, batch_size: int) -> None:
    arrays = {k: np.asarray(cache[k]) for k in ("normals_norm", "midpoints_norm", "scores", "normals", "points", "center")}
    arrays.update({k: np.asarray(cache[k]) for k in _CACHE_SCALARS})
    arrays["detector_params"] = np.array(json.dumps(params, sort_keys=True))
    arrays["planes_batch_size"] = np.asarray(batch_size)
    arrays["created"] = np.array(datetime.now().isoformat(timespec="seconds"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:   # con file object np.savez no agrega ".npz"
        np.savez(f, **arrays)
    os.replace(tmp, path)        # escritura atómica: nunca queda un .npz a medias


def load_cache(path: Path, params: dict) -> dict:
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    stored = json.loads(str(data["detector_params"]))
    if stored != params:
        raise CacheParamsMismatch(
            f"{path} se generó con otros parámetros del detector:\n  caché : {stored}\n  ahora : {params}\n"
            "Usa los mismos --views/--rot_aug/--view_samp/--seed, o un --out_dir distinto.")
    cache = {k: data[k].reshape(-1, 3) for k in ("normals_norm", "midpoints_norm", "normals", "points")}
    cache["scores"] = data["scores"].reshape(-1)
    cache["center"] = data["center"].reshape(3)
    cache.update({k: float(data[k]) for k in _CACHE_SCALARS})
    return cache


def detect_and_cache(ctx, raw: np.ndarray, path: Path) -> dict:
    args = ctx.args
    res = sdet.detect_object(ctx.det, ctx.model, raw, ctx.device, args.seed, view_samp=args.view_samp,
                             view_quantity=args.views, rot_aug=args.rot_aug, batch_size=args.planes_batch_size)
    if res["detector_stdout"].strip():
        log.debug("salida del detector:\n%s", res["detector_stdout"].rstrip())
    if res["n_nonfinite"]:
        log.warning("el detector devolvió %d planos no finitos (descartados)", res["n_nonfinite"])
    normals, points = nscd.planes_to_original(res["normals_norm"], res["midpoints_norm"], res["center"], res["scale"])
    cache = {"normals_norm": res["normals_norm"], "midpoints_norm": res["midpoints_norm"], "scores": res["scores"],
             "normals": normals, "points": points, "center": res["center"], "scale": res["scale"],
             "n_points": raw.shape[0], "n_nonfinite": res["n_nonfinite"],
             "time_features_s": res["time_features_s"], "time_planes_s": res["time_planes_s"],
             "vram_peak_alloc_mb": res["vram_peak_alloc_mb"], "vram_peak_reserved_mb": res["vram_peak_reserved_mb"]}
    save_cache(path, cache, ctx.params, args.planes_batch_size)
    return cache


# ----------------------------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------------------------
def prepare_results_csv(path: Path, resume: bool) -> set:
    """Sin --resume: CSV nuevo. Con --resume: conserva las filas terminadas (ok / no_planes) y descarta
    las de error para reintentarlas. Devuelve los IDs ya terminados."""
    if resume and path.exists():
        rows = summ.read_csv_rows(path)
        keep = [r for r in rows if r.get("status") in RESUMABLE_STATUSES]
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if len(keep) != len(rows) or header != RESULT_FIELDS:
            summ.write_csv_atomic(path, keep, RESULT_FIELDS)
        return {r["object_id"] for r in keep}
    summ.write_csv_atomic(path, [], RESULT_FIELDS)
    return set()


def append_row(fh, writer, row: dict) -> None:
    writer.writerow({k: _py(v) for k, v in row.items()})
    fh.flush()
    os.fsync(fh.fileno())


# ----------------------------------------------------------------------------------------------
# Un objeto: detección (o caché) + evaluación
# ----------------------------------------------------------------------------------------------
def fill_eval_fields(row: dict, ev: dict, cache, args, paper_value) -> None:
    yz = ev["yz"]
    diag = ev["diag"]
    row.update(nscd_yz=yz["nscd"], nscd_sq_yz=yz["nscd_sq"], cd_yz=yz["cd"], diag_nscd=diag,
               x_shift_nscd=ev["x_shift"], n_nscd_points=len(ev["sample"]),
               nscd_det_fb=yz["nscd"], fallback_used=1)  # se sobreescribe si hay plano
    if paper_value is not None:
        row.update(nscd_yz_paper_json=paper_value, diff_yz_vs_paper_json=yz["nscd"] - paper_value)
    if cache is None:
        return
    center = cache["center"]
    row.update(center_x=center[0], center_y=center[1], center_z=center[2], scale=cache["scale"],
               n_planes_detected=len(cache["normals"]),
               time_features_s=cache["time_features_s"], time_planes_s=cache["time_planes_s"],
               time_detect_s=cache["time_features_s"] + cache["time_planes_s"],
               vram_peak_alloc_mb=cache["vram_peak_alloc_mb"], vram_peak_reserved_mb=cache["vram_peak_reserved_mb"])
    per = ev["per_rank"]
    if not per:
        return
    nscds = [p["nscd"] for p in per]
    best = int(np.argmin(nscds))
    chosen = 0 if args.selection == "top1" else best
    normals, points, scores = cache["normals"], cache["points"], cache["scores"]
    angles = [nscd.angle_to_x_deg(n) for n in normals[: len(per)]]
    n, m = normals[chosen], points[chosen]
    row.update(
        chosen_rank=chosen + 1, detector_score=scores[chosen],
        normal_x=n[0], normal_y=n[1], normal_z=n[2], point_x=m[0], point_y=m[1], point_z=m[2],
        angle_to_x_deg=angles[chosen], offset_norm=nscd.signed_offset(center, n, m) / diag,
        nscd_det=per[chosen]["nscd"], nscd_det_top1=per[0]["nscd"], nscd_det_bestk=per[best]["nscd"],
        bestk_rank=best + 1, nscd_det_topk=";".join(f"{v:.10g}" for v in nscds),
        delta_nscd=yz["nscd"] - per[chosen]["nscd"], delta_nscd_top1=yz["nscd"] - per[0]["nscd"],
        delta_nscd_bestk=yz["nscd"] - per[best]["nscd"],
        nscd_sq_det=per[chosen]["nscd_sq"], nscd_sq_det_top1=per[0]["nscd_sq"], nscd_sq_det_bestk=per[best]["nscd_sq"],
        cd_det=per[chosen]["cd"], nscd_det_fb=per[chosen]["nscd"], fallback_used=0,
        angle_to_x_deg_top1=angles[0], angle_to_x_deg_bestk=angles[best],
        min_angle_to_x_topk_deg=min(angles), rank_closest_to_x=int(np.argmin(angles)) + 1)


def process_object(ctx, category: str, object_id: str, path: Path, paper_value) -> dict:
    args = ctx.args
    row = dict.fromkeys(RESULT_FIELDS, "")
    row.update(category=category, category_id=CATEGORIES[category], object_id=object_id, selection=args.selection)
    try:
        raw = load_points(path)
    except Exception as exc:  # noqa: BLE001
        row.update(status="error", error_msg=f"carga: {short(exc)}")
        return row
    row["n_points"] = raw.shape[0]

    cache, status, msg = None, None, ""
    cpath = cache_path(args.out_dir, category, object_id)
    if cpath.exists():
        cache = load_cache(cpath, ctx.params)  # CacheParamsMismatch aborta la corrida a propósito
        row["from_cache"] = 1
    elif ctx.det is not None:
        oom = False
        try:
            cache = detect_and_cache(ctx, raw, cpath)
            row["from_cache"] = 0
        except torch.cuda.OutOfMemoryError as exc:
            status, msg, oom = "detect_oom", short(exc), True
        except Exception as exc:  # noqa: BLE001
            status, msg = "detect_error", short(exc)
            log.error("El detector falló en %s/%s:\n%s", category, object_id, traceback.format_exc())
        if oom:
            free_cuda()  # fuera del except, cuando el traceback ya no retiene tensores
    else:
        status, msg = "no_cache", "sin planos en caché (corre --stage detect o all)"

    t0 = time.perf_counter()
    try:
        normals = cache["normals"] if cache is not None else np.zeros((0, 3))
        points = cache["points"] if cache is not None else np.zeros((0, 3))
        ev = nscd.evaluate_object(raw, ctx.protocol, normals, points, args.top_k, args.nscd_points)
        fill_eval_fields(row, ev, cache, args, paper_value)
        if status is None:
            status = "ok" if len(normals) else "no_planes"
    except Exception as exc:  # noqa: BLE001
        status = "error"
        msg = (msg + " | " if msg else "") + f"evaluación: {short(exc)}"
        log.error("La evaluación falló en %s/%s:\n%s", category, object_id, traceback.format_exc())
    row.update(status=status, error_msg=msg, time_eval_s=time.perf_counter() - t0)
    return row


# ----------------------------------------------------------------------------------------------
# Etapas
# ----------------------------------------------------------------------------------------------
def run_detect(ctx) -> None:
    args = ctx.args
    for category in args.categories:
        log_path = args.out_dir / f"detect_log_{category}.csv"
        new_file = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=DETECT_LOG_FIELDS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            todo = [o for o in ctx.ids[category] if not cache_path(args.out_dir, category, o).exists()]
            log.info("%s: %d objetos, %d ya en caché, %d por detectar", category, len(ctx.ids[category]),
                     len(ctx.ids[category]) - len(todo), len(todo))
            for object_id in progress(todo, f"detect {category}"):
                row = {"category": category, "object_id": object_id,
                       "timestamp": datetime.now().isoformat(timespec="seconds")}
                oom = False
                try:
                    raw = load_points(ctx.split_dirs[category] / f"{object_id}.npy")
                    row["n_points"] = raw.shape[0]
                    cache = detect_and_cache(ctx, raw, cache_path(args.out_dir, category, object_id))
                    row.update(status="ok" if len(cache["normals"]) else "no_planes",
                               n_planes_detected=len(cache["normals"]),
                               **{k: cache[k] for k in ("time_features_s", "time_planes_s",
                                                        "vram_peak_alloc_mb", "vram_peak_reserved_mb")})
                except torch.cuda.OutOfMemoryError as exc:
                    row.update(status="detect_oom", error_msg=short(exc))
                    oom = True
                except Exception as exc:  # noqa: BLE001
                    row.update(status="detect_error", error_msg=short(exc))
                    log.error("El detector falló en %s/%s:\n%s", category, object_id, traceback.format_exc())
                if oom:
                    free_cuda()
                append_row(fh, writer, row)


def run_evaluate(ctx) -> None:
    """Etapas 'all' (detecta lo que falte) y 'evaluate' (solo caché)."""
    args = ctx.args
    for category in args.categories:
        paper_json = summ.load_paper_nscd_json(args.paper_json_dir, category)
        res_path = args.out_dir / f"results_per_object_{category}.csv"
        done = prepare_results_csv(res_path, args.resume)
        todo = [o for o in ctx.ids[category] if o not in done]
        log.info("%s: %d objetos, %d ya evaluados, %d por procesar", category, len(ctx.ids[category]),
                 len(ctx.ids[category]) - len(todo), len(todo))
        counts = {}
        with open(res_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            for object_id in progress(todo, f"{args.stage} {category}"):
                row = process_object(ctx, category, object_id, ctx.split_dirs[category] / f"{object_id}.npy",
                                     paper_json.get(object_id))
                append_row(fh, writer, row)
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                if row["status"] not in RESUMABLE_STATUSES:
                    log.warning("%s/%s -> %s: %s", category, object_id, row["status"], row["error_msg"])
                else:
                    log.debug("%s/%s -> %s nscd_yz=%s nscd_det=%s planos=%s t_det=%s", category, object_id,
                              row["status"], row["nscd_yz"], row["nscd_det"], row["n_planes_detected"],
                              row["time_detect_s"])
        log.info("%s terminado: %s", category, counts)


def run_summary(args, protocol) -> None:
    by_cat = {}
    for category in args.categories:
        path = args.out_dir / f"results_per_object_{category}.csv"
        if path.exists():
            by_cat[category] = summ.read_csv_rows(path)
    if not by_cat:
        log.warning("No hay results_per_object_*.csv en %s para resumir", args.out_dir)
        return
    summary_rows = [summ.summarize_category(c, rows) for c, rows in by_cat.items()]
    summ.write_csv_atomic(args.out_dir / "summary.csv", summary_rows, list(summary_rows[0].keys()))
    text = summ.format_summary(summary_rows, args.n_objects)
    (args.out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    log.info("Resumen (también en summary.csv / summary.txt):%s", text)
    if args.n_figures > 0:
        make_figures(args, protocol, by_cat)


def make_figures(args, protocol, by_cat: dict) -> None:
    params = detector_params(args)
    n_done = 0
    for category, rows in by_cat.items():
        split_dir = args.data_root / CATEGORIES[category] / args.split
        for tag, rank, row in summ.pick_objects_for_figures(rows, args.n_figures):
            object_id = row["object_id"]
            try:
                raw = load_points(split_dir / f"{object_id}.npy")
                cache = load_cache(cache_path(args.out_dir, category, object_id), params)
                ev = nscd.evaluate_object(raw, protocol, cache["normals"], cache["points"], args.top_k, args.nscd_points)
                k = int(float(row["chosen_rank"])) - 1
                shift = np.array([ev["x_shift"], 0.0, 0.0])
                plane_det = (cache["normals"][k], cache["points"][k] - shift)
                planes = [("YZ (paper)", np.array([1.0, 0.0, 0.0]), np.zeros(3), ev["yz"]["nscd"]),
                          (f"detector (rango {k + 1})", *plane_det, ev["per_rank"][k]["nscd"])]
                title = (f"{category} {object_id} | ángulo vs YZ {float(row['angle_to_x_deg']):.2f}° | "
                         f"offset {float(row['offset_norm']):+.4f} diag | delta {float(row['delta_nscd']):+.5f}")
                stem = f"{tag}_{rank:02d}_{object_id}"
                fig_dir = args.out_dir / "figures" / category
                summ.plot_object(fig_dir / f"{stem}.png", ev["sample"], planes, title)
                if args.save_ply:
                    sample = ev["sample"]
                    summ.save_ply(fig_dir / f"{stem}_orig.ply", sample, (0.12, 0.47, 0.71))
                    summ.save_ply(fig_dir / f"{stem}_refl_det.ply",
                                  nscd.reflect_across_plane(sample, *plane_det), (1.0, 0.5, 0.05))
                    summ.save_ply(fig_dir / f"{stem}_refl_yz.ply",
                                  protocol.householder_transformation(sample), (0.17, 0.63, 0.17))
                n_done += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("No se pudo generar la figura de %s/%s: %s", category, object_id, short(exc))
    log.info("%d figuras en %s", n_done, args.out_dir / "figures")


# ----------------------------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------------------------
def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(["git", "-C", str(PROJECT_ROOT), *cmd], capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    return {"commit": run("rev-parse", "HEAD"),
            "repos_modified": run("status", "--porcelain", "--", "detector", "generativos")}


def write_config(args, ids: dict, env: dict) -> None:
    cfg_path = args.out_dir / "config.json"
    if cfg_path.exists() and args.resume and args.stage in ("all", "evaluate"):
        with open(cfg_path, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("stage") in ("all", "evaluate") and prev.get("evaluation_params") != evaluation_params(args):
            raise SystemExit(
                f"--resume con parámetros de evaluación distintos a los de la corrida anterior "
                f"({prev.get('evaluation_params')} vs {evaluation_params(args)}). Para cambiarlos, corre "
                "--stage evaluate SIN --resume: regenera los CSV desde la caché sin re-detectar.")
    cfg = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "stage": args.stage,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "detector_params": detector_params(args),
        "evaluation_params": evaluation_params(args),
        "normalization": sdet.NORMALIZATION_DESC,
        "detector_fixed_in_repo": {
            "get_candidate_planes.threshold": "0.01 sobre pytorch3d.chamfer_distance (distancias al cuadrado, "
                                              "medias) en el marco normalizado",
            "angle_threshold_degrees": 5, "max_planes": 10,
            "intersection_radius": "5% de la norma máxima de 1000 puntos al azar",
            "repo_planes_batch_size": sdet.REPO_PLANES_BATCH_SIZE,
            "planes_batch_size_used": args.planes_batch_size,
        },
        "nscd_definition": {
            "nscd": "Symmetry_NormalizedPCs.py: centrado en X (si |media_x| >= 1e-3) + FPS Open3D a nscd_points + "
                    "ChamferDistance.chamfer_distance (KDTree, NO al cuadrado) / diagonal AABB",
            "nscd_sq": "ec. (1)-(2) del paper: CD con distancias al cuadrado / diagonal^2, misma muestra",
            "det_reflection": "p' = p - 2((p - m)·n) n, plano del detector en coords originales desplazado por el "
                              "centrado en X del paper",
            "selection": {"top1": "mayor score del detector", "bestk": "mínimo NSCD entre los top-K"}[args.selection],
            "no_planes": "nscd_det = NaN (status no_planes); nscd_det_fb = nscd_yz (fallback_used = 1)",
        },
        "paper_nscd_mean": nscd.PAPER_NSCD_MEAN,
        "environment": env,
        "git": git_info(),
        "object_ids": ids,
    }
    tmp = cfg_path.with_name("config.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, cfg_path)
    with open(args.out_dir / "config_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({k: cfg[k] for k in cfg if k != "object_ids"}, ensure_ascii=False, default=str) + "\n")
    for category, object_ids in ids.items():
        (args.out_dir / f"object_ids_{category}.txt").write_text("\n".join(object_ids) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out_dir, args.log_level)
    log.info("==== stage=%s | categorías=%s | out_dir=%s ====", args.stage, args.categories, args.out_dir)
    protocol = load_nscd_protocol(args.nscd_root)

    if args.stage == "summary":
        run_summary(args, protocol)
        return 0

    ids, split_dirs = {}, {}
    for category in args.categories:
        ids[category], split_dirs[category] = select_object_ids(args, category)

    ctx = SimpleNamespace(args=args, protocol=protocol, ids=ids, split_dirs=split_dirs,
                          params=detector_params(args), det=None, model=None, device=None)
    env = {"python": sys.version, "platform": platform.platform(), "host": socket.gethostname(),
           "versions": sdet.library_versions()}

    if args.stage in ("all", "detect"):
        if not args.resume:
            n_cached = sum(len(list((args.out_dir / "cache" / c).glob("*.npz"))) for c in args.categories)
            if n_cached:
                raise SystemExit(f"{args.out_dir} ya tiene {n_cached} objetos en caché. Usa --resume para continuar "
                                 "esa corrida, o un --out_dir distinto.")
        sdet.configure_determinism(not args.no_deterministic)
        env["gpu"] = sdet.check_device(args.device)
        if args.device.startswith("cuda"):
            g = env["gpu"]
            log.info("GPU %s: %s | VRAM %.1f GB | capability %s | CUDA_VISIBLE_DEVICES=%s", g["gpu_index"],
                     g["gpu_name"], g["vram_total_gb"], g["gpu_capability"], g["cuda_visible_devices"])
        else:
            log.warning(env["gpu"]["warning"])
        v = env["versions"]
        log.info("torch %s (CUDA %s, cuDNN %s) | pytorch3d %s | torch_cluster %s | open3d %s | Python %s",
                 v["torch"], v["torch_cuda"], v["cudnn"], v["pytorch3d"], v["torch_cluster"], v["open3d"],
                 platform.python_version())
        env["dino_cache"] = sdet.dino_cache_status()
        if env["dino_cache"]["needs_internet"]:
            log.warning("DINOv2 no está completo en la caché de torch.hub (%s): torch.hub lo descargará "
                        "(necesita internet a github.com y dl.fbaipublicfiles.com).", env["dino_cache"]["hub_dir"])
        else:
            log.info("DINOv2 en caché local (%s): no necesita internet", env["dino_cache"]["checkpoint"])
        ctx.det = import_detector(args.detector_root)
        ctx.device = torch.device(args.device)
        t0 = time.perf_counter()
        ctx.model, dino_out = sdet.load_dino(ctx.det, ctx.device)
        log.debug("DINOWrapper:\n%s", dino_out.rstrip())
        log.info("DINOv2 (small, reg) cargado en %.1f s", time.perf_counter() - t0)
        log.info("Detector: view_samp=%d views=%d rot_aug=%d | planes_batch_size=%d (repo: %d) | determinista=%s",
                 args.view_samp, args.views, args.rot_aug, args.planes_batch_size, sdet.REPO_PLANES_BATCH_SIZE,
                 not args.no_deterministic)
    write_config(args, ids, env)

    t_start = time.perf_counter()
    try:
        if args.stage == "detect":
            run_detect(ctx)
        else:
            run_evaluate(ctx)
            run_summary(args, protocol)
    except KeyboardInterrupt:
        log.warning("Interrumpido. El progreso está guardado: retoma con el mismo comando + --resume.")
        return 130
    except CacheParamsMismatch as exc:
        log.error(str(exc))
        return 2
    log.info("Listo en %.1f min", (time.perf_counter() - t_start) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
