"""
Ejecución del detector de simetrías (GPU) sobre una nube normalizada, y chequeos de CUDA/pytorch3d.

Notas sobre el código del detector que condicionan esta implementación
----------------------------------------------------------------------
* detector/implementation/clouds/render.py::render_and_map_pytorch3d hace, sobre el MISMO tensor
  que recibe compute_features (si ya está en la GPU, `.to(device)` no copia):
        verts = point_cloud; verts -= mean; verts /= max ||verts||
  O sea: el detector normaliza IN-PLACE a centroide + esfera unitaria. Esa es la normalización
  "del laboratorio" (main.py y evaluation/*.py dependen de ese efecto lateral). Aquí la nube se
  normaliza explícitamente igual ANTES de llamar al detector, y a compute_features se le pasa un
  `.clone()` para que el tensor que recibe compute_planes no cambie por debajo.
* compute_features fija la distancia de la cámara con la norma máxima de la nube ANTES de esa
  normalización interna: si la nube de entrada no tuviera radio 1, el objeto renderizado (radio 1)
  no calzaría con las cámaras. Otra razón para pre-normalizar.
* compute_planes usa torch.randperm (RNG de CPU) -> se re-siembra antes de cada etapa y de cada
  objeto, así el resultado no depende del orden de procesamiento ni de --resume.
* compute_planes llama a get_candidate_planes(..., batch_size=5, threshold=0.01, ...). Para cambiar
  SOLO el tamaño de lote (parámetro puramente computacional) se envuelve temporalmente esa función
  en el namespace de comp_planes; umbrales y lógica quedan intactos.
"""
from __future__ import annotations

import contextlib
import gc
import io
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

REPO_PLANES_BATCH_SIZE = 5            # valor fijo en comp_planes.compute_planes
# Parámetros por defecto de compute_features (firma de la función en el repo).
# Ojo: detector/main.py usa view_quantity=114; aquí el default es el de la función (6).
DEFAULT_VIEW_SAMP = 1
DEFAULT_VIEW_QUANTITY = 6
DEFAULT_ROT_AUG = 3
# DINOv2 como en detector/main.py: DINOWrapper(device, small=True, reg=True)
DINO_SMALL = True
DINO_REG = True
_DINO_CKPT = {
    (True, True): "dinov2_vits14_reg4_pretrain.pth",
    (True, False): "dinov2_vits14_pretrain.pth",
    (False, True): "dinov2_vitg14_reg4_pretrain.pth",
    (False, False): "dinov2_vitg14_pretrain.pth",
}

NORMALIZATION = "centroid_unit_sphere"
NORMALIZATION_DESC = (
    "x' = (x - c) / s, con c = centroide de los 15k puntos y s = max ||x - c|| (esfera unitaria). "
    "Es la misma normalización que aplica in-place render_and_map_pytorch3d dentro de compute_features."
)


# ----------------------------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    """Determinismo best-effort en GPU. Lo relevante para el detector: la asignación
    `features[mapping] = pixel_features` (index_put con índices repetidos) es no determinista en CUDA
    salvo con use_deterministic_algorithms. warn_only=True: las ops sin versión determinista solo
    avisan. CUBLAS_WORKSPACE_CONFIG se fija en run_experiment.py antes de importar torch."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = enabled
    torch.use_deterministic_algorithms(enabled, warn_only=True)


@contextlib.contextmanager
def capture_stdout():
    """El detector imprime con print(); se captura para mandarlo al log y no romper la barra tqdm."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


@contextlib.contextmanager
def planes_batch_size(comp_planes_module, batch_size: int):
    """Cambia SOLO batch_size en la llamada que compute_planes hace a get_candidate_planes."""
    if batch_size is None or batch_size == REPO_PLANES_BATCH_SIZE:
        yield
        return
    original = comp_planes_module.get_candidate_planes

    def _get_candidate_planes(*args, **kwargs):
        kwargs["batch_size"] = batch_size
        return original(*args, **kwargs)

    comp_planes_module.get_candidate_planes = _get_candidate_planes
    try:
        yield
    finally:
        comp_planes_module.get_candidate_planes = original


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ----------------------------------------------------------------------------------------------
# Chequeos de entorno
# ----------------------------------------------------------------------------------------------
def library_versions() -> dict:
    versions = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}
    for name in ["pytorch3d", "torchvision", "torch_geometric", "torch_cluster", "open3d", "scipy",
                 "numpy", "point_cloud_utils", "sklearn"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            versions[name] = f"NO DISPONIBLE ({exc.__class__.__name__}: {exc})"
    return versions


def check_device(device_str: str) -> dict:
    """Falla con un mensaje claro si se pidió CUDA y no está, o si pytorch3d / torch_cluster no
    corren en GPU. Nunca cae en silencio a CPU."""
    info = {"device": device_str}
    if device_str == "cpu":
        info["warning"] = "Corriendo el detector en CPU por pedido explícito (--device cpu): será MUY lento."
        return info
    if not device_str.startswith("cuda"):
        raise ValueError(f"--device debe ser 'cuda', 'cuda:N' o 'cpu' (recibí {device_str!r})")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible (torch.cuda.is_available() == False). Revisa el driver, "
            "CUDA_VISIBLE_DEVICES y que torch sea una build con CUDA (torch.version.cuda = "
            f"{torch.version.cuda}). Si de verdad quieres correr en CPU, pasa --device cpu."
        )
    device = torch.device(device_str)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"{device_str} no existe: solo hay {torch.cuda.device_count()} GPU(s) visibles")
    torch.cuda.set_device(index)
    props = torch.cuda.get_device_properties(index)
    info.update({
        "gpu_index": index,
        "gpu_name": props.name,
        "gpu_capability": f"{props.major}.{props.minor}",
        "vram_total_gb": round(props.total_memory / 1024**3, 2),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    dev = torch.device("cuda", index)

    # pytorch3d con CUDA: chamfer_distance y knn_points en GPU, comparados con CPU
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.ops import knn_points
    gen = torch.Generator().manual_seed(0)
    a = torch.rand(2, 257, 3, generator=gen)
    b = torch.rand(2, 311, 3, generator=gen)
    try:
        cd_gpu, _ = chamfer_distance(a.to(dev), b.to(dev), batch_reduction=None)
        knn = knn_points(a.to(dev), b.to(dev), K=1)
    except RuntimeError as exc:
        raise RuntimeError(f"pytorch3d falló en GPU ({exc}). ¿pytorch3d está compilado con CUDA y "
                           "para esta misma versión de torch?") from exc
    cd_cpu, _ = chamfer_distance(a, b, batch_reduction=None)
    if not (knn.dists.is_cuda and torch.allclose(cd_gpu.cpu(), cd_cpu, rtol=1e-4, atol=1e-6)):
        raise RuntimeError("pytorch3d.chamfer_distance en GPU no coincide con CPU")

    # Rasterizador de puntos de pytorch3d (lo usa compute_features para renderizar la nube)
    from pytorch3d.renderer import (FoVPerspectiveCameras, PointsRasterizationSettings, PointsRasterizer,
                                    look_at_view_transform)
    from pytorch3d.structures import Pointclouds
    R, T = look_at_view_transform(dist=3.0, elev=10.0, azim=30.0, device=dev)
    rasterizer = PointsRasterizer(
        cameras=FoVPerspectiveCameras(device=dev, R=R, T=T, znear=0.01),
        raster_settings=PointsRasterizationSettings(image_size=64, radius=0.02, points_per_pixel=4, bin_size=0),
    )
    fragments = rasterizer(Pointclouds(points=[(a[0] - 0.5).to(dev)]))
    if not (fragments.idx.is_cuda and bool((fragments.idx >= 0).any())):
        raise RuntimeError("El rasterizador de pytorch3d no produjo fragmentos en GPU")

    # torch_geometric.nn.knn (torch_cluster), usado por interpolate_point_cloud en compute_features
    try:
        from torch_geometric.nn import knn as pyg_knn
        pyg_knn(a[0].to(dev), a[0].to(dev), 2)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"torch_geometric.nn.knn falló en GPU ({exc}). Instala torch-cluster con CUDA "
                           "para tu versión de torch.") from exc
    info["pytorch3d_cuda_ok"] = True
    info["torch_cluster_cuda_ok"] = True
    return info


def dino_cache_status(small: bool = DINO_SMALL, reg: bool = DINO_REG) -> dict:
    hub_dir = Path(torch.hub.get_dir())
    repo_dirs = sorted(str(p) for p in hub_dir.glob("facebookresearch_dinov2_*"))
    checkpoint = hub_dir / "checkpoints" / _DINO_CKPT[(small, reg)]
    return {"hub_dir": str(hub_dir), "repo_cached": bool(repo_dirs), "repo_dirs": repo_dirs,
            "checkpoint": str(checkpoint), "checkpoint_cached": checkpoint.is_file(),
            "needs_internet": not (repo_dirs and checkpoint.is_file())}


def load_dino(det, device: torch.device, small: bool = DINO_SMALL, reg: bool = DINO_REG):
    """Carga DINOv2 UNA vez. DINOWrapper se traga la excepción de torch.hub y luego falla en
    self.model.eval() con AttributeError: se traduce a un error explicativo."""
    with capture_stdout() as buf:
        try:
            model = det.DINOWrapper(device=device, small=small, reg=reg)
        except AttributeError as exc:
            raise RuntimeError("No se pudo cargar DINOv2 desde torch.hub (¿sin internet y sin caché?). "
                               f"Salida de DINOWrapper:\n{buf.getvalue()}") from exc
    model.eval()
    return model, buf.getvalue()


# ----------------------------------------------------------------------------------------------
# Detección
# ----------------------------------------------------------------------------------------------
def normalize_unit_sphere(points: np.ndarray):
    p = np.asarray(points, dtype=np.float64)
    center = p.mean(axis=0)
    scale = float(np.linalg.norm(p - center, axis=1).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Nube degenerada (escala = {scale})")
    return ((p - center) / scale).astype(np.float32), center, scale


def detector_features(det, model, points_gpu: torch.Tensor, device: torch.device, seed: int,
                      view_samp: int, view_quantity: int, rot_aug: int):
    seed_everything(seed)
    with torch.no_grad(), capture_stdout() as buf:
        # .clone(): compute_features normaliza in-place el tensor que recibe
        features = det.compute_features(points_gpu.clone(), model, device,
                                        view_samp=view_samp, view_quantity=view_quantity, rot_aug=rot_aug)
    return features, buf.getvalue()


def detector_planes(det, points_gpu: torch.Tensor, features: torch.Tensor, device: torch.device, seed: int,
                    batch_size: int = REPO_PLANES_BATCH_SIZE):
    seed_everything(seed)
    with torch.no_grad(), capture_stdout() as buf, planes_batch_size(det.comp_planes_module, batch_size):
        normals, midpoints, scores = det.compute_planes(points_gpu, features, device)
    return (normals.detach().cpu().numpy().reshape(-1, 3).astype(np.float64),
            midpoints.detach().cpu().numpy().reshape(-1, 3).astype(np.float64),
            scores.detach().cpu().numpy().reshape(-1).astype(np.float64),
            buf.getvalue())


def detect_object(det, model, raw_points: np.ndarray, device: torch.device, seed: int,
                  view_samp: int = DEFAULT_VIEW_SAMP, view_quantity: int = DEFAULT_VIEW_QUANTITY,
                  rot_aug: int = DEFAULT_ROT_AUG, batch_size: int = REPO_PLANES_BATCH_SIZE) -> dict:
    """Normaliza, corre compute_features -> compute_planes con los 15k puntos y devuelve los planos
    en el MARCO NORMALIZADO (más c y s para llevarlos al original)."""
    points_norm, center, scale = normalize_unit_sphere(raw_points)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    points_gpu = torch.from_numpy(points_norm).to(device)
    features = None
    try:
        _sync(device)
        t0 = time.perf_counter()
        features, log_features = detector_features(det, model, points_gpu, device, seed,
                                                   view_samp, view_quantity, rot_aug)
        _sync(device)
        t1 = time.perf_counter()
        normals, midpoints, scores, log_planes = detector_planes(det, points_gpu, features, device, seed, batch_size)
        _sync(device)
        t2 = time.perf_counter()
    finally:
        del features, points_gpu
        if device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    finite = np.isfinite(normals).all(axis=1) & np.isfinite(midpoints).all(axis=1) & np.isfinite(scores)
    return {
        "center": center, "scale": scale,
        "normals_norm": normals[finite], "midpoints_norm": midpoints[finite], "scores": scores[finite],
        "n_nonfinite": int((~finite).sum()),
        "time_features_s": t1 - t0, "time_planes_s": t2 - t1,
        "vram_peak_alloc_mb": peak_alloc if device.type == "cuda" else float("nan"),
        "vram_peak_reserved_mb": peak_reserved if device.type == "cuda" else float("nan"),
        "detector_stdout": log_features + log_planes,
    }
