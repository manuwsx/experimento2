"""
Resolución de rutas e imports explícitos de los dos repos (sin modificarlos).

Colisiones de nombres y cómo se evitan
--------------------------------------
* detector/ usa imports ABSOLUTOS a un paquete top-level llamado `implementation`
  (p. ej. `from implementation.utils import DINOWrapper`). Por eso `detector/` tiene que
  estar en sys.path. Se inserta en sys.path[0] y luego se COMPRUEBA que
  `implementation.__file__` apunte a detector/implementation (si otro paquete con ese nombre
  estuviera instalado o ya importado, se aborta). Los sub-paquetes `implementation.utils`,
  `implementation.planes`, etc. quedan bajo el espacio de nombres `implementation`, así que no
  chocan con ningún `utils`/`planes` top-level.
* generativos/Symmetry_Measurement_Protocol tiene módulos "planos" con nombres genéricos
  (`ChamferDistance`, `FarthestPointSampling`, `Householder_transform`). NO se agregan a sys.path:
  se cargan por ruta de archivo con `importlib` bajo nombres privados
  (`_symmatters_ChamferDistance`, ...). Así no pueden chocar con nada (por ejemplo con
  `generativos/Replicability/utils.py` o `detector/evaluation/utils.py`).
* `Symmetry_NormalizedPCs.py` NO se importa: ejecuta argparse y procesa una carpeta al
  importarse. Su preprocesamiento se replica literalmente en sym_nscd.py y se verifica contra el
  script original ejecutándolo como subproceso (verify.py).
* Los módulos propios llevan prefijo `sym_` para no ensombrecer nada de los repos.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

# <raíz>/experimentos/exp01_shapenet_detector_nscd/sym_imports.py -> parents[2] = <raíz>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "ShapeNetCore.v2.PC15k"
DEFAULT_DETECTOR_ROOT = PROJECT_ROOT / "detector"
DEFAULT_NSCD_ROOT = PROJECT_ROOT / "generativos" / "Symmetry_Measurement_Protocol"
DEFAULT_PAPER_JSON_DIR = PROJECT_ROOT / "generativos" / "Replicability" / "computed_NSCD"

CATEGORIES = {"airplane": "02691156", "car": "02958343", "chair": "03001627"}


def resolve_cli_path(path: Path) -> Path:
    """Los defaults ya son absolutos (anclados a PROJECT_ROOT, así el script funciona desde cualquier
    directorio). Una ruta relativa pasada explícitamente por CLI se interpreta, como en cualquier comando,
    respecto al directorio actual."""
    return Path(path).expanduser().resolve()


def _load_module_from_file(private_name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}")
    if private_name in sys.modules:
        return sys.modules[private_name]
    spec = importlib.util.spec_from_file_location(private_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[private_name] = module
    spec.loader.exec_module(module)
    return module


def load_nscd_protocol(nscd_root: Path) -> SimpleNamespace:
    """Carga ChamferDistance.py, FarthestPointSampling.py y Householder_transform.py del repo
    generativos, tal cual, bajo nombres privados."""
    nscd_root = Path(nscd_root).resolve()
    chamfer = _load_module_from_file("_symmatters_ChamferDistance", nscd_root / "ChamferDistance.py")
    fps = _load_module_from_file("_symmatters_FarthestPointSampling", nscd_root / "FarthestPointSampling.py")
    householder = _load_module_from_file("_symmatters_Householder_transform", nscd_root / "Householder_transform.py")
    return SimpleNamespace(
        root=nscd_root,
        chamfer_distance=chamfer.chamfer_distance,
        farthest_point_sampling=fps.farthest_point_sampling,
        householder_transformation=householder.householder_transformation,
        original_script=nscd_root / "Symmetry_NormalizedPCs.py",
    )


def import_detector(detector_root: Path) -> SimpleNamespace:
    """Importa el paquete `implementation` del repo del detector (requiere Python >= 3.12 por los
    f-strings de comp_features.py)."""
    if sys.version_info < (3, 12):
        raise RuntimeError(
            f"Python {sys.version.split()[0]} detectado: detector/implementation/compute/comp_features.py "
            "usa f-strings con comillas anidadas (PEP 701) y requiere Python >= 3.12."
        )
    detector_root = Path(detector_root).resolve()
    pkg_dir = detector_root / "implementation"
    if not (pkg_dir / "__init__.py").is_file():
        raise FileNotFoundError(f"No encuentro el paquete del detector en {pkg_dir}")

    def _pkg_dir(module):
        file = getattr(module, "__file__", None)
        return Path(file).resolve().parent if file else None

    already = sys.modules.get("implementation")
    if already is not None and _pkg_dir(already) != pkg_dir:
        raise ImportError(f"Ya hay otro módulo 'implementation' importado ({already!r})")
    if str(detector_root) not in sys.path:
        sys.path.insert(0, str(detector_root))

    implementation = importlib.import_module("implementation")
    if _pkg_dir(implementation) != pkg_dir:
        raise ImportError(f"'implementation' se resolvió a {implementation!r}, no a {pkg_dir} (colisión de nombres)")
    utils = importlib.import_module("implementation.utils")
    comp_features = importlib.import_module("implementation.compute.comp_features")
    comp_planes = importlib.import_module("implementation.compute.comp_planes")
    planes = importlib.import_module("implementation.planes.planes")
    return SimpleNamespace(
        root=detector_root,
        DINOWrapper=utils.DINOWrapper,
        compute_features=comp_features.compute_features,
        compute_planes=comp_planes.compute_planes,
        comp_planes_module=comp_planes,   # para envolver get_candidate_planes (solo batch_size)
        planes_module=planes,
    )
