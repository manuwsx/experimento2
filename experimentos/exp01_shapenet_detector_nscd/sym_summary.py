"""
Resumen por categoría (summary.csv), lectura de los NSCD del paper y figuras de inspección.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from sym_nscd import PAPER_NSCD_MEAN, reflect_across_plane

_PAPER_SUFFIX = ".npy normalized cd"
DETECT_FAIL_STATUSES = ("detect_error", "detect_oom", "no_cache")


# ----------------------------------------------------------------------------------------------
# Datos del paper
# ----------------------------------------------------------------------------------------------
def load_paper_nscd_json(paper_json_dir: Path, category: str) -> dict:
    """generativos/Replicability/computed_NSCD/shapenet_<cat>.json -> {object_id: NSCD}.
    Las claves son los 1000 objetos que usó el paper (en el orden de glob de su máquina)."""
    path = Path(paper_json_dir) / f"shapenet_{category}.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k[: -len(_PAPER_SUFFIX)]: float(v) for k, v in data.items() if k.endswith(_PAPER_SUFFIX)}


# ----------------------------------------------------------------------------------------------
# Lectura de resultados
# ----------------------------------------------------------------------------------------------
def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_atomic(path: Path, rows: list[dict], fields: list[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _num(value) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _col(rows, key) -> np.ndarray:
    return np.array([_num(r.get(key)) for r in rows], dtype=np.float64)


def _stats(values: np.ndarray, prefix: str) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {f"{prefix}_mean": math.nan, f"{prefix}_std": math.nan, f"{prefix}_median": math.nan, f"{prefix}_n": 0}
    return {f"{prefix}_mean": float(v.mean()), f"{prefix}_std": float(v.std(ddof=1)) if v.size > 1 else math.nan,
            f"{prefix}_median": float(np.median(v)), f"{prefix}_n": int(v.size)}


def _pct(mask: np.ndarray, denom_mask: np.ndarray) -> float:
    d = int(denom_mask.sum())
    return 100.0 * float((mask & denom_mask).sum()) / d if d else math.nan


# ----------------------------------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------------------------------
def summarize_category(category: str, rows: list[dict]) -> dict:
    status = np.array([r.get("status", "") for r in rows])
    ok = status == "ok"
    yz = _col(rows, "nscd_yz")
    det = _col(rows, "nscd_det")
    paired = ok & np.isfinite(yz) & np.isfinite(det)
    angle = _col(rows, "angle_to_x_deg")
    from_cache = _col(rows, "from_cache")
    timed = paired & (from_cache == 0)

    out = {"category": category, "n_objects": len(rows), "n_ok": int(ok.sum()),
           "n_no_planes": int((status == "no_planes").sum()),
           "n_detect_failed": int(np.isin(status, DETECT_FAIL_STATUSES).sum()),
           "n_eval_error": int((status == "error").sum())}

    # Baseline YZ vs paper
    out.update(_stats(yz, "nscd_yz"))
    paper_mean = PAPER_NSCD_MEAN.get(category, math.nan)
    out["paper_nscd_mean"] = paper_mean
    out["yz_minus_paper"] = out["nscd_yz_mean"] - paper_mean
    out["yz_rel_diff_pct"] = 100.0 * out["yz_minus_paper"] / paper_mean
    pj = _col(rows, "nscd_yz_paper_json")
    overlap = np.isfinite(pj) & np.isfinite(yz)
    out["paper_json_n_overlap"] = int(overlap.sum())
    out["paper_json_max_abs_diff"] = float(np.abs(yz - pj)[overlap].max()) if overlap.any() else math.nan

    # Detector: sin fallos (solo objetos con plano) y con fallback YZ (todos)
    out.update(_stats(det, "nscd_det"))
    out.update(_stats(np.where(paired, yz, np.nan), "nscd_yz_paired"))
    out.update(_stats(_col(rows, "nscd_det_fb"), "nscd_det_fb"))
    out.update(_stats(_col(rows, "nscd_det_top1"), "nscd_det_top1"))
    out.update(_stats(_col(rows, "nscd_det_bestk"), "nscd_det_bestk"))
    out.update(_stats(_col(rows, "nscd_sq_yz"), "nscd_sq_yz"))
    out.update(_stats(_col(rows, "nscd_sq_det"), "nscd_sq_det"))

    # Descomposición: delta = NSCD_YZ - NSCD_det (desviación del plano)
    out.update(_stats(_col(rows, "delta_nscd"), "delta_nscd"))
    out.update(_stats(_col(rows, "delta_nscd_top1"), "delta_nscd_top1"))
    out.update(_stats(_col(rows, "delta_nscd_bestk"), "delta_nscd_bestk"))
    fb = _col(rows, "nscd_det_fb")
    out["delta_nscd_fb_mean"] = float(np.nanmean(yz - fb)) if np.isfinite(yz - fb).any() else math.nan

    # Porcentajes (denominador: objetos con plano detectado y NSCD válido)
    out["pct_improved_of_ok"] = _pct(det < yz, paired)
    out["pct_improved_top1_of_ok"] = _pct(_col(rows, "nscd_det_top1") < yz, paired)
    out["pct_improved_bestk_of_ok"] = _pct(_col(rows, "nscd_det_bestk") < yz, paired)
    for thr in (1, 5, 10):
        out[f"pct_angle_lt{thr}_of_ok"] = _pct(angle < thr, paired)
    out["pct_x_plane_in_topk_lt5_of_ok"] = _pct(_col(rows, "min_angle_to_x_topk_deg") < 5, paired)
    out["pct_bestk_is_top1_of_ok"] = _pct(_col(rows, "bestk_rank") == 1, paired)

    # Costo (solo objetos detectados en esta corrida, no leídos de caché)
    t_det = _col(rows, "time_detect_s")[timed]
    vram = _col(rows, "vram_peak_alloc_mb")[timed]
    out["n_timed"] = int(timed.sum())
    out["time_detect_mean_s"] = float(np.nanmean(t_det)) if t_det.size else math.nan
    out["time_detect_median_s"] = float(np.nanmedian(t_det)) if t_det.size else math.nan
    out["time_eval_mean_s"] = float(np.nanmean(_col(rows, "time_eval_s"))) if rows else math.nan
    out["vram_peak_alloc_mean_mb"] = float(np.nanmean(vram)) if vram.size else math.nan
    out["vram_peak_alloc_max_mb"] = float(np.nanmax(vram)) if vram.size else math.nan
    return out


def _fmt(v, nd=5):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def format_summary(summary_rows: list[dict], n_objects_target: int) -> str:
    lines = []
    for s in summary_rows:
        c = s["category"]
        lines.append(f"\n=== {c}: {s['n_objects']} objetos | ok={s['n_ok']} sin_planos={s['n_no_planes']} "
                     f"fallo_detector={s['n_detect_failed']} error_eval={s['n_eval_error']}")
        lines.append(f"  NSCD-YZ (paper, todos)      : media {_fmt(s['nscd_yz_mean'])}  sd {_fmt(s['nscd_yz_std'])}  "
                     f"mediana {_fmt(s['nscd_yz_median'])}   [paper: {s['paper_nscd_mean']:.4f}, "
                     f"dif {_fmt(s['yz_rel_diff_pct'], 2)}%]")
        if s["paper_json_n_overlap"]:
            lines.append(f"  vs JSON del paper           : {s['paper_json_n_overlap']} objetos en común, "
                         f"max |dif| = {s['paper_json_max_abs_diff']:.3e}")
        lines.append(f"  NSCD-det (sin fallos)       : media {_fmt(s['nscd_det_mean'])}  sd {_fmt(s['nscd_det_std'])}  "
                     f"mediana {_fmt(s['nscd_det_median'])}   (YZ en esos mismos: {_fmt(s['nscd_yz_paired_mean'])})")
        lines.append(f"  NSCD-det (fallback YZ)      : media {_fmt(s['nscd_det_fb_mean'])}")
        lines.append(f"  top-1 / mejor-de-K          : {_fmt(s['nscd_det_top1_mean'])} / {_fmt(s['nscd_det_bestk_mean'])}"
                     f"   (mejor-de-K = top-1 en {_fmt(s['pct_bestk_is_top1_of_ok'], 1)}%)")
        lines.append(f"  NSCD_sq (fórmula paper)     : YZ {s['nscd_sq_yz_mean']:.3e}  det {s['nscd_sq_det_mean']:.3e}")
        lines.append(f"  delta = YZ - det            : media {_fmt(s['delta_nscd_mean'])}  mediana {_fmt(s['delta_nscd_median'])}"
                     f"   | mejora en {_fmt(s['pct_improved_of_ok'], 1)}% de los objetos")
        lines.append(f"  ángulo plano vs YZ          : <1° {_fmt(s['pct_angle_lt1_of_ok'], 1)}%  <5° "
                     f"{_fmt(s['pct_angle_lt5_of_ok'], 1)}%  <10° {_fmt(s['pct_angle_lt10_of_ok'], 1)}%"
                     f"   | algún top-K a <5° de YZ: {_fmt(s['pct_x_plane_in_topk_lt5_of_ok'], 1)}%")
        if s["n_timed"]:
            eta_h = s["time_detect_mean_s"] * n_objects_target / 3600.0
            lines.append(f"  tiempo detección            : media {s['time_detect_mean_s']:.1f} s/obj (mediana "
                         f"{s['time_detect_median_s']:.1f}) -> {n_objects_target} objetos ≈ {eta_h:.1f} h | "
                         f"VRAM pico media {s['vram_peak_alloc_mean_mb']:.0f} MB (máx {s['vram_peak_alloc_max_mb']:.0f})")
    timed = [s for s in summary_rows if s["n_timed"]]
    if timed:
        mean_t = float(np.mean([s["time_detect_mean_s"] for s in timed]))
        lines.append(f"\nEstimación total: {mean_t:.1f} s/objeto x {n_objects_target} x 3 categorías ≈ "
                     f"{mean_t * n_objects_target * 3 / 3600.0:.1f} h (una GPU)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------------------------
def _plane_trace(ax, normal, point, a, b):
    """Dibuja la intersección del plano con el corte (eje c) = m_c, proyectada en los ejes (a, b)."""
    na, nb = normal[a], normal[b]
    if max(abs(na), abs(nb)) < 1e-6:
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if abs(nb) >= abs(na):
        u = np.linspace(x0, x1, 2)
        v = point[b] - na * (u - point[a]) / nb
    else:
        v = np.linspace(y0, y1, 2)
        u = point[a] - nb * (v - point[b]) / na
    ax.plot(u, v, "k--", lw=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def plot_object(png_path: Path, sample: np.ndarray, planes: list[tuple], title: str) -> None:
    """planes: [(etiqueta, normal, punto, nscd), ...] en el marco de `sample`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Z", "Y", 2, 1)]
    fig, axes = plt.subplots(len(planes), 3, figsize=(15, 4.8 * len(planes)), squeeze=False)
    for row, (label, normal, point, value) in enumerate(planes):
        reflected = reflect_across_plane(sample, normal, point)
        for col, (la, lb, a, b) in enumerate(views):
            ax = axes[row, col]
            ax.scatter(sample[:, a], sample[:, b], s=1, c="tab:blue", alpha=0.45, label="original")
            ax.scatter(reflected[:, a], reflected[:, b], s=1, c="tab:orange", alpha=0.45, label="reflejada")
            ax.set_aspect("equal", adjustable="box")
            ax.autoscale_view()
            _plane_trace(ax, normal, point, a, b)
            ax.set_xlabel(la)
            ax.set_ylabel(lb)
            if col == 0:
                ax.set_title(f"{label}: NSCD = {value:.5f}", loc="left", fontsize=10)
        axes[row, 0].legend(loc="upper right", markerscale=6, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def save_ply(path: Path, points: np.ndarray, rgb) -> None:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(rgb, dtype=np.float64), (len(points), 1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def pick_objects_for_figures(rows: list[dict], n: int) -> list[tuple[str, int, dict]]:
    """Los n objetos con mayor delta_nscd y los n con mayor ángulo respecto a YZ: (tag, ranking, fila)."""
    ok = [r for r in rows if r.get("status") == "ok" and math.isfinite(_num(r.get("delta_nscd")))]
    picks = []
    for tag, key in (("delta", "delta_nscd"), ("angle", "angle_to_x_deg")):
        for rank, r in enumerate(sorted(ok, key=lambda r: _num(r[key]), reverse=True)[:n], start=1):
            picks.append((tag, rank, r))
    return picks
