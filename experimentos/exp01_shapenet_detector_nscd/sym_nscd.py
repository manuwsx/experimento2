"""
NSCD "estilo paper" (CPU) generalizado a un plano arbitrario.

Replica EXACTAMENTE lo que hace generativos/Symmetry_Measurement_Protocol/Symmetry_NormalizedPCs.py
(rama .npy), que es lo que generó los números del paper:
  1. centrar SOLO en X restando la media de X, y solo si |media_x| >= 1e-3 (is_centered_x)
  2. FPS de Open3D a 2048 puntos (si hay más de 2048)
  3. reflejar (YZ: Householder_transform.householder_transformation)
  4. ChamferDistance.chamfer_distance  (KDTree, distancias NO al cuadrado, mean(a->b) + mean(b->a))
  5. dividir por la diagonal de la AABB de la muestra de 2048 puntos (sin elevar al cuadrado)
Las funciones del repo (FPS, Chamfer, Householder) se reciben ya cargadas (ver sym_imports.py);
aquí no se reimplementan. Solo se agrega:
  * la reflexión respecto a un plano arbitrario (n, m)
  * la variante `nscd_sq` de la fórmula del paper (ec. 1-2): CD con distancias al cuadrado / diag^2
  * diagnósticos del plano (ángulo con X, offset respecto al centroide)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

# Medias reportadas en el paper (sec. 4.1, 1000 objetos de train)
PAPER_NSCD_MEAN = {"airplane": 0.0101, "car": 0.0219, "chair": 0.0190}
# Tolerancia de is_centered_x() en Symmetry_NormalizedPCs.py
X_CENTER_TOL = 1e-3


# ----------------------------------------------------------------------------------------------
# Preprocesamiento del paper
# ----------------------------------------------------------------------------------------------
def paper_preprocess(points: np.ndarray, farthest_point_sampling, n_points: int = 2048):
    """Réplica literal de Symmetry_NormalizedPCs.compute_files (rama npy).

    Devuelve (muestra, x_shift): `x_shift` es lo que se restó a la coordenada X antes del FPS
    (0.0 si el objeto ya estaba centrado en X según la tolerancia del script original). Para llevar
    un plano en coordenadas originales al marco de la muestra basta restar [x_shift, 0, 0] a su punto.
    """
    points = np.array(points, copy=True)  # el script original modifica in-place el array cargado
    centroid_x = np.mean(points[:, 0])
    if abs(centroid_x) < X_CENTER_TOL:
        x_shift = 0.0
    else:
        points[:, 0] -= centroid_x
        x_shift = float(centroid_x)
    if points.shape[0] > n_points:
        sample = farthest_point_sampling(points, n_points)
    else:
        sample = points
    return sample, x_shift


def aabb_diagonal(points: np.ndarray) -> float:
    """Igual que compute_diagonal() de Symmetry_NormalizedPCs.py."""
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    return np.sqrt(dx**2 + dy**2 + dz**2)


# ----------------------------------------------------------------------------------------------
# Reflexión y Chamfer
# ----------------------------------------------------------------------------------------------
def reflect_across_plane(points: np.ndarray, normal, point_on_plane) -> np.ndarray:
    """p' = p - 2((p - m)·n) n, equivalente a R(p) = Hp + t con H = I - 2nnᵀ, t = 2nnᵀm (sec. 3.2).
    Generaliza Householder_transform.householder_transformation (n = [1,0,0], m = 0)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    m = np.asarray(point_on_plane, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    d = (p - m) @ n
    return p - 2.0 * d[:, None] * n[None, :]


def chamfer_distance_squared(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """CD de la ec. (1) del paper: mean(min ||x-y||²) en ambos sentidos (KDTree, como ChamferDistance.py)."""
    d1 = KDTree(pc2).query(pc1)[0]
    d2 = KDTree(pc1).query(pc2)[0]
    return float(np.mean(d1**2) + np.mean(d2**2))


def nscd_of_reflection(sample: np.ndarray, reflected: np.ndarray, chamfer_distance, diag: float) -> dict:
    cd = chamfer_distance(sample, reflected)          # función del repo generativos
    cd_sq = chamfer_distance_squared(sample, reflected)
    return {"cd": float(cd), "nscd": float(cd / diag), "cd_sq": cd_sq, "nscd_sq": float(cd_sq / diag**2)}


# ----------------------------------------------------------------------------------------------
# Planos: marco normalizado -> original, y diagnósticos
# ----------------------------------------------------------------------------------------------
def canonical_normal(normal) -> np.ndarray:
    """n y -n son el mismo plano; se fija el signo para que la componente de mayor |.| sea positiva
    (así el signo del offset es consistente entre objetos)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return -n if n[int(np.argmax(np.abs(n)))] < 0 else n


def planes_to_original(normals_norm: np.ndarray, points_norm: np.ndarray, center: np.ndarray, scale: float):
    """Con x' = (x - c)/s: la normal no cambia y m_orig = m_norm * s + c."""
    normals = np.array([canonical_normal(n) for n in normals_norm]).reshape(-1, 3)
    points = np.asarray(points_norm, dtype=np.float64).reshape(-1, 3) * scale + np.asarray(center, dtype=np.float64)
    return normals, points


def angle_to_x_deg(normal) -> float:
    n = np.asarray(normal, dtype=np.float64)
    return float(np.degrees(np.arccos(np.clip(abs(n[0]) / np.linalg.norm(n), 0.0, 1.0))))


def signed_offset(center, normal, point_on_plane) -> float:
    """(c - m)·n: distancia con signo del centroide al plano."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return float(np.dot(np.asarray(center, dtype=np.float64) - np.asarray(point_on_plane, dtype=np.float64), n))


# ----------------------------------------------------------------------------------------------
# Evaluación de un objeto
# ----------------------------------------------------------------------------------------------
def evaluate_object(raw_points: np.ndarray, protocol, normals: np.ndarray, points: np.ndarray,
                    top_k: int = 5, nscd_points: int = 2048) -> dict:
    """Calcula el NSCD-YZ (paper) y el NSCD de los top-K planos del detector SOBRE LA MISMA muestra.

    `normals`/`points`: planos del detector en coordenadas ORIGINALES del .npy, ordenados del mejor
    al peor (puede venir vacío). Devuelve un dict con la muestra y los resultados por rango.
    """
    sample, x_shift = paper_preprocess(raw_points, protocol.farthest_point_sampling, nscd_points)
    diag = aabb_diagonal(sample)
    yz = nscd_of_reflection(sample, protocol.householder_transformation(sample), protocol.chamfer_distance, diag)

    shift = np.array([x_shift, 0.0, 0.0])
    per_rank = []
    for r in range(min(top_k, len(normals))):
        reflected = reflect_across_plane(sample, normals[r], points[r] - shift)
        per_rank.append(nscd_of_reflection(sample, reflected, protocol.chamfer_distance, diag))
    return {"sample": sample, "x_shift": x_shift, "diag": float(diag), "yz": yz, "per_rank": per_rank}
