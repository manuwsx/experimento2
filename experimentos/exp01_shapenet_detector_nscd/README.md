# Experimento 1 — detector de simetrías + NSCD en ShapeNet (airplane, car, chair)

Control: el detector **tal cual** sobre objetos reales de ShapeNetCore.v2.PC15k (15k puntos). Se compara el
NSCD con el plano detectado contra el NSCD del paper "Symmetry matters", que asume el plano YZ (x = 0).

```
run_experiment.py   CLI: etapas detect / evaluate / all / summary
verify.py           verificaciones previas (reflexión, invariancia, equivalencia con el paper, batch, determinismo)
sym_imports.py      rutas e imports explícitos de detector/ y generativos/ (sin tocarlos)
sym_detector.py     normalización, chequeos CUDA/pytorch3d, carga de DINOv2, ejecución del detector
sym_nscd.py         NSCD idéntico a Symmetry_NormalizedPCs.py + reflexión respecto a un plano arbitrario
sym_summary.py      summary.csv, lectura de los JSON del paper, figuras
```

Requisitos: Python ≥ 3.12 (`comp_features.py` usa f-strings de PEP 701), torch + pytorch3d **con CUDA y de la
misma versión de torch**, torch_geometric + torch_cluster con CUDA, point_cloud_utils, scikit-learn, open3d, scipy,
numpy, tqdm y matplotlib (este último solo para las figuras). DINOv2 se baja con `torch.hub` la primera vez
(github.com + dl.fbaipublicfiles.com); el script avisa si no está en caché.

## Comandos (Linux, desde esta carpeta)

```bash
cd experimentos/exp01_shapenet_detector_nscd

# 0) Verificaciones (sección 5). Primero las de CPU, después las de GPU
python verify.py --checks cpu
python verify.py --checks gpu --planes_batch_sizes 16 64 256

# 1) Prueba rápida: 5 objetos por categoría (mide tiempo/objeto y estima el total)
python run_experiment.py --categories all --n_objects 5 --out_dir results_smoke

# 2) Corrida completa, todas las categorías
python run_experiment.py --categories all --out_dir results

#    una sola categoría / varias
python run_experiment.py --categories car --out_dir results
python run_experiment.py --categories car chair --out_dir results

# 3) Retomar una corrida interrumpida (mismo comando + --resume)
python run_experiment.py --categories all --out_dir results --resume

# 4) Re-evaluar desde la caché sin volver a correr el detector (p. ej. otra selección del plano)
python run_experiment.py --stage evaluate --selection bestk --out_dir results
python run_experiment.py --stage summary --out_dir results --n_figures 5 --save_ply
```

En segundo plano, sin que muera al cerrar la terminal:

```bash
nohup python run_experiment.py --categories all --out_dir results --resume > results_nohup.log 2>&1 &
tail -f results/run.log
```

o con tmux: `tmux new -s exp01`, correr el comando, desacoplar con `Ctrl-b d`, volver con `tmux attach -t exp01`.

Elegir GPU: `CUDA_VISIBLE_DEVICES=1 python run_experiment.py ...` (dentro del proceso esa GPU pasa a ser
`cuda:0`). Con varias GPUs se puede repartir por categoría en el **mismo** `--out_dir`, porque cada categoría
escribe sus propios archivos:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python run_experiment.py --categories airplane --out_dir results --resume > a.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python run_experiment.py --categories car      --out_dir results --resume > c.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python run_experiment.py --categories chair    --out_dir results --resume > h.log 2>&1 &
# al terminar todas:
python run_experiment.py --stage summary --out_dir results
```

## Salidas (`--out_dir`)

| Archivo | Contenido |
|---|---|
| `cache/<cat>/<id>.npz` | Todos los planos del detector (≤ 10): normales/puntos en el marco normalizado y en coords originales, scores, `c`, `s`, tiempos, VRAM y parámetros del detector |
| `results_per_object_<cat>.csv` | Una fila por objeto, escrita y sincronizada al disco objeto por objeto |
| `summary.csv`, `summary.txt` | Resumen por categoría (también se imprime por consola) |
| `detect_log_<cat>.csv` | Solo con `--stage detect` |
| `figures/<cat>/{delta,angle}_NN_<id>.png` (+ `.ply`) | Los N objetos con mayor `delta_nscd` y los N con mayor ángulo respecto a YZ |
| `config.json`, `config_history.jsonl`, `object_ids_<cat>.txt`, `run.log` | Parámetros, versiones, GPU, git, IDs procesados y log |

### Columnas de `results_per_object_<cat>.csv`

| Columna | Significado |
|---|---|
| `status` | `ok` · `no_planes` (el detector devolvió 0 planos) · `detect_error` / `detect_oom` · `no_cache` (etapa evaluate sin caché) · `error` (carga o evaluación). `error_msg` trae el detalle |
| `n_points` | Puntos del `.npy` (debería ser 15000) |
| `center_*`, `scale` | `c` (centroide) y `s` (radio máximo) de la normalización del detector: `x' = (x - c)/s` |
| `n_planes_detected` | Planos devueltos por `compute_planes` (0–10, ya deduplicados a 5°) |
| `selection`, `chosen_rank` | Estrategia (`top1`/`bestk`) y rango (1 = mejor score) del plano que va a `nscd_det` |
| `detector_score` | `1 - CD/0.01` del detector (más alto = mejor; **no** es una distancia) |
| `normal_*`, `point_*` | Plano elegido en coordenadas **originales** del `.npy`; la normal tiene el signo fijado (componente de mayor módulo > 0) |
| `angle_to_x_deg` | `arccos(|n·[1,0,0]|)`: 0° = plano paralelo a YZ |
| `offset_norm` | `(c - m)·n / diag_nscd`: distancia con signo del centroide al plano, normalizada |
| `nscd_yz` | NSCD del paper (YZ tras centrar en X), idéntico a `Symmetry_NormalizedPCs.py` |
| `nscd_det` | NSCD con el plano elegido, **sobre la misma muestra de 2048 puntos** |
| `nscd_det_top1`, `nscd_det_bestk`, `bestk_rank`, `nscd_det_topk` | NSCD del top-1, el mínimo entre los top-K y su rango, y los K valores (`;`) |
| `delta_nscd` (+ `_top1`, `_bestk`) | `nscd_yz - nscd_det`: brecha atribuible al plano (> 0: el detector mejora) |
| `nscd_sq_*` | Variante de la fórmula del paper: CD con distancias² / diag² |
| `nscd_det_fb`, `fallback_used` | `nscd_det` con fallback a `nscd_yz` cuando no hay plano (1 = se usó el fallback) |
| `angle_to_x_deg_top1`, `angle_to_x_deg_bestk`, `min_angle_to_x_topk_deg`, `rank_closest_to_x` | Ángulos de otros candidatos: ¿había un plano ≈ YZ entre los top-K y en qué rango? |
| `cd_yz`, `cd_det`, `diag_nscd`, `x_shift_nscd`, `n_nscd_points` | Ingredientes: Chamfer sin normalizar, diagonal AABB de la muestra, desplazamiento X aplicado por el protocolo del paper, tamaño de la muestra |
| `nscd_yz_paper_json`, `diff_yz_vs_paper_json` | NSCD del JSON del paper para ese objeto (si lo usó) y la diferencia con el nuestro |
| `time_features_s`, `time_planes_s`, `time_detect_s`, `time_eval_s` | Tiempos (detección en GPU; evaluación en CPU) |
| `vram_peak_alloc_mb`, `vram_peak_reserved_mb`, `from_cache` | VRAM pico del objeto; 1 si los planos venían de la caché |
