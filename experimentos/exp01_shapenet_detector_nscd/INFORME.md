# Experimento 1 — Informe de implementación

Detector de simetrías (tal cual) + NSCD sobre ShapeNetCore.v2.PC15k (airplane, car, chair).

> **Estado:** el código está escrito pero **no se ha ejecutado** (ni el detector, ni pruebas, ni las
> verificaciones de la sección 5). Las verificaciones quedan en `verify.py` para correrlas en la nube.

Archivos de esta carpeta: [run_experiment.py](run_experiment.py), [verify.py](verify.py), los módulos `sym_*.py`
y el [README.md](README.md) con todos los comandos y la tabla completa de columnas. No se tocaron `detector/` ni
`generativos/`.

## Antes de correr: lo que se encontró en el código

1. **El detector normaliza la nube in-place** (`detector/implementation/clouds/render.py:59-61`): centra en el
   centroide y escala a esfera unitaria. Si recibe un tensor que ya está en la GPU, lo modifica. Esa es la
   normalización del laboratorio: `main.py` depende de ese efecto. El script normaliza igual antes de llamar al
   detector y le pasa una copia (`.clone()`). Ojo: el script borrado `evaluate_symmetry_planes_full.py` centraba
   con el centro de la caja y le pasaba el tensor en GPU, así que sus planos probablemente quedaban desplazados
   al volver a coordenadas originales.
2. **Número de vistas:** `compute_features` trae `view_quantity=6` por defecto, pero `main.py` usa 114. Se usa 6
   (parámetros por defecto del repo); se cambia con `--views 114`.
3. **Los 1000 objetos del paper no son los primeros 1000 en orden alfabético.** El paper usó el orden de `glob`
   de su máquina, y la lista exacta está en las claves de `shapenet_<cat>.json`. El orden alfabético queda por
   defecto y `--id_source paper` usa exactamente sus objetos.
4. **El script del paper solo centra en X si |media_x| ≥ 1e-3.** Se replicó tal cual, para que el resultado sea
   idéntico.
5. **Entornos inconsistentes:** el `requirements.txt` de la raíz fija torch 2.4/cu121 con pytorch3d sin versión,
   mientras que `detector/requirements.txt` pide torch 2.8/cu129 con `pytorch3d==0.7.8+pt2.8.0cu129`. Si pytorch3d
   no calza con torch, el chequeo de arranque lo detecta y falla con un mensaje claro. Además,
   `detector/evaluation/planes_calculation.py` le pasa `vggt_dino=` a `DINOWrapper`, que no lo acepta, así que esos
   scripts de evaluación están desactualizados (no se usan).
6. **Medias de los JSON del paper:** 0.01013 / 0.02186 / 0.01902, que calzan con lo reportado. La diagonal de las
   muestras que usó el paper es ≈ 1 en la mayoría de los objetos. En car hay un objeto con diagonal 0.005,
   probablemente degenerado.
7. **Los archivos adjuntos (zip) son idénticos** a los de los repos.

## Comandos

Todos se corren desde la carpeta del experimento:

```bash
cd experimentos/exp01_shapenet_detector_nscd
```

Verificaciones, primero las de CPU y después las de GPU:

```bash
python verify.py --checks cpu
```

```bash
python verify.py --checks gpu --planes_batch_sizes 16 64 256
```

Prueba rápida, 5 objetos por categoría (mide el tiempo y estima el total):

```bash
python run_experiment.py --categories all --n_objects 5 --out_dir results_smoke
```

Una sola categoría:

```bash
python run_experiment.py --categories car --out_dir results
```

Corrida completa en segundo plano. Con `--resume` también sirve para la primera vez y para retomar:

```bash
nohup python run_experiment.py --categories all --out_dir results --resume > results_nohup.log 2>&1 &
```

Para elegir la GPU, por ejemplo la segunda:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python run_experiment.py --categories all --out_dir results --resume > results_nohup.log 2>&1 &
```

Con tmux: `tmux new -s exp01`, desacoplar con `Ctrl-b d` y volver con `tmux attach -t exp01`. Con varias GPUs se
puede lanzar un proceso por categoría sobre el mismo `--out_dir`; el README trae el ejemplo.

Para cambiar la estrategia de selección sin volver a correr el detector:

```bash
python run_experiment.py --stage evaluate --selection bestk --out_dir results
```

## Cómo funciona

**Flujo de un objeto:**

1. Carga el `.npy` y verifica que sea N×3 y finito; avisa si no tiene 15000 puntos.
2. Normaliza para el detector: `x' = (x−c)/s`, con `c` el centroide y `s` el radio máximo.
3. Corre `compute_features` y luego `compute_planes` con los 15k puntos, en GPU, bajo `no_grad`, con las semillas
   reiniciadas antes de cada objeto (así el resultado no depende del orden ni de `--resume`).
4. Lleva los planos a coordenadas originales: la normal no cambia y `m = m'·s + c`. El signo de la normal se fija
   para que su componente de mayor módulo sea positiva.
5. Guarda en caché **todos** los planos en `cache/<cat>/<id>.npz`, con escritura atómica.
6. Arma la muestra del paper: centrado en X y FPS a 2048 puntos.
7. Calcula sobre **esa misma muestra** el NSCD-YZ con el Householder del repo y el NSCD de los top-K planos,
   reflejando con `p' = p − 2((p−m)·n)n` (al punto del plano se le resta el mismo centrado en X).
8. Escribe una fila en el CSV con `flush` + `fsync`.

**Etapas (`--stage`):** `detect` (cara, en GPU), `evaluate` (barata, desde la caché), `all` (por defecto) y
`summary`. La evaluación en CPU es mínima comparada con la detección, así que no se superpone con la GPU: es más
simple y robusto.

**Columnas del CSV, agrupadas** (la tabla completa está en el README):

- **Estado:** `status` (`ok` / `no_planes` / `detect_error` / `detect_oom` / `no_cache` / `error`) y `error_msg`.
- **Normalización:** `center_*` y `scale`.
- **Plano elegido, en coordenadas originales:** `normal_*` y `point_*`, más `chosen_rank`, `detector_score`
  (es `1−CD/0.01`, no una distancia), `angle_to_x_deg` y `offset_norm`.
- **Métrica:**
  - `nscd_yz`: el NSCD del paper.
  - `nscd_det`: el NSCD con el plano elegido.
  - `nscd_det_top1`, `nscd_det_bestk` / `bestk_rank` y `nscd_det_topk` (los K valores).
  - `delta_nscd = nscd_yz − nscd_det`: la parte de la brecha que se debe al plano.
  - `nscd_sq_*`: la variante con la fórmula del paper.
  - `nscd_det_fb` / `fallback_used`: el NSCD con fallback a YZ cuando no hay plano.
- **Diagnóstico de los candidatos:** `min_angle_to_x_topk_deg` y `rank_closest_to_x`.
- **Comparación con el paper:** `nscd_yz_paper_json` y `diff_yz_vs_paper_json`.
- **Costo:** tiempos, VRAM pico y `from_cache`.

## Decisiones

- **Selección del plano: top-1 como métrica principal.** No usa la métrica de evaluación para elegir, así que
  `delta` puede salir negativo cuando el detector se equivoca, y eso es información. El mínimo entre los top-K
  queda como cota optimista, en columnas propias. El resumen reporta en qué porcentaje de objetos coinciden. YZ
  nunca entra como candidato.
- **NSCD:** el principal replica exactamente el código del paper (FPS de Open3D, `chamfer_distance` del repo sin
  elevar al cuadrado, división por la diagonal). `nscd_sq` sigue las ecuaciones (1) y (2) del paper; en el PDF se
  confirmó que ahí el CD va al cuadrado y se divide por diag².
- **Objetos sin planos:** `nscd_det` queda en NaN con `status=no_planes`, y `nscd_det_fb` toma el valor de YZ. El
  resumen da el promedio sin esos objetos y con fallback, y el conteo de fallos por categoría. `nscd_yz` se
  calcula siempre, aunque el detector falle.
- **Cambio computacional:** el tamaño de lote del detector (`--planes_batch_size`) queda por defecto en **5, el
  valor del repo**. Para usar otro, el script envuelve temporalmente `get_candidate_planes` y solo cambia
  `batch_size`; umbrales y lógica no se tocan. `verify.py --checks batch` compara bit a bit, con las mismas
  features, el lote 5 contra otros tamaños en 3 objetos. Usarlo solo si sale idéntico.
- **Determinismo:** se activa `torch.use_deterministic_algorithms` (con `warn_only`) porque la asignación
  `features[mapping] = ...` del detector, con índices repetidos, no es determinista en CUDA. Se desactiva con
  `--no_deterministic`.
- **Colisiones de nombres:** `detector/` va en `sys.path[0]` y el script comprueba que `implementation` se
  resuelva a esa carpeta. Los módulos de `generativos` se cargan con `importlib` bajo nombres privados
  (`_symmatters_*`), sin agregarlos a `sys.path`. `Symmetry_NormalizedPCs.py` no se importa porque corre argparse
  al importarse; se replica y se compara ejecutándolo como subproceso. Los módulos propios llevan el prefijo
  `sym_`.
- **Arranque:** el script falla con un mensaje claro si no hay CUDA (no cae en silencio a CPU), si Python es menor
  que 3.12, si pytorch3d no corre en GPU (prueba `chamfer_distance`, `knn_points` y el rasterizador) o si
  `torch_cluster` no tiene CUDA. También avisa si DINOv2 no está en la caché de `torch.hub` y necesita internet.
- **Protección de la caché:** `--resume` reintenta los objetos que fallaron y aborta si los parámetros del
  detector no coinciden con los de la caché. Sin `--resume`, se niega a escribir sobre una caché existente.

## Verificaciones (sección 5): pendientes

`verify.py` cubre los cinco puntos y deja un `verify_report.json`:

| Check | Qué hace |
|---|---|
| `reflection` | Nube sintética simétrica respecto a un plano oblicuo que no pasa por el origen: NSCD ≈ 0 con ese plano y > 0 con YZ. Verifica también que coincida con el Householder del paper, la involución, que n y −n den lo mismo, y la forma `Hp+t`. |
| `invariance` | Traslación y escala uniforme, incluido el marco normalizado del detector, con nubes sintéticas y reales. |
| `paper_script` | Corre el `Symmetry_NormalizedPCs.py` **original** como subproceso sobre 20 objetos por categoría y exige igualdad exacta. |
| `paper_json` | Compara objeto por objeto contra `shapenet_<cat>.json`. |
| `data_stats` | Escala, centrado y diagonal de 20 `.npy` por categoría. |
| `batch`, `determinism`, `frames` | En GPU: lote 5 contra otros tamaños, dos corridas del detector, y NSCD en marco normalizado contra original. Reportan tiempo y VRAM por objeto. |

**Tiempo:** no se midió. Cada lote del detector hace una Chamfer de 15k contra 15k puntos y hay miles de lotes
por objeto, así que se esperan decenas de segundos por objeto: para 3×1000 objetos, del orden de un día o más en
una GPU. Es una estimación gruesa sin verificar; la prueba de 5 objetos imprime el número real y el total
estimado.

## Limitaciones y riesgos

- **Sesgo de `bestk`:** elige usando la misma métrica que se reporta, así que es optimista. Por eso queda como cota
  y no como métrica principal.
- **Umbral absoluto del detector:** el filtro de 0.01 sobre la Chamfer al cuadrado depende de la escala, aunque
  con la esfera unitaria queda fijo. En nubes ralas y con ruido, más objetos van a quedar sin planos.
- **Objetos con varios planos casi simétricos** (en autos, el plano XY): el top-1 puede no ser YZ y aun así tener
  un NSCD bajo. `delta > 0` no implica que el detector encontró "el" plano correcto; revisar `angle_to_x_deg`,
  `rank_closest_to_x` y las figuras.
- **Paso a objetos generados (2048 puntos):** el detector va a ver menos puntos, el FPS a 2048 deja de
  submuestrear, los artefactos alteran las features de DINO y el umbral, y conviene volver a calibrar.
- **Reproducibilidad:** el determinismo es por GPU y versión de las librerías. Entre GPUs distintas puede haber
  diferencias mínimas en las features.
