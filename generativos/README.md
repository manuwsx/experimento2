# Symmetry Matters: Auditing and Symmetrizing 3D Generative Models

![Teaser](assets/Symmetrization%20of%203D%20Generative%20Models.jpg)

> **Symmetry Matters: Auditing and Symmetrizing 3D Generative Models**<br>
> Nicolas Caytuiro and Ivan Sipiran<br>
> University of Chile

[![Paper](https://img.shields.io/badge/arXiv-2512.18953-b31b1b.svg)](https://arxiv.org/abs/2512.18953)
[![Code](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/NicolasPCS/Symmetrization-of-3D-Generative-Models)

Abstract: _Symmetry is a strong prior present in many object categories, yet standard benchmarks for 3D generative models rarely report whether this prior is preserved. We study symmetry preservation in unconditional point cloud generation. We first audit the symmetry of generated shapes by several 3D generative models and compute a normalized symmetry score based on the Chamfer Distance (CD). We show that although current 3D generative models achieve competitive results under standard evaluation, they reveal a persistent symmetry gap when a symmetry-aware evaluation protocol is applied. To test whether this gap is merely inherited from the training data, we evaluate these models over a mirrored-objects dataset derived from ShapeNet and analyze symmetry dynamics during training. Mechanism-inspired diagnostic tests were conducted at the sampling and latent-representation levels to further show that reflection symmetry is not reliably encoded in the learned generative process. Finally, to address this gap, we propose a data-centric symmetry-based intervention: training generative models on a half-objects dataset and reconstructing full objects by reflection during sampling. Across multiple backbones, this intervention substantially improves geometric consistency and visual plausibility while remaining competitive under standard metrics. These findings suggest that symmetry-aware evaluation is needed alongside standard benchmarks, and future 3D generative models should incorporate this prior explicitly, either during training or sampling._

## Contents

- [Install](#install)
- [Reproducing the paper](#reproducing-the-paper)
- [Baseline forks](#baseline-forks)
- [Checkpoints and generated samples](#checkpoints-and-generated-samples)
- [Mirrored Dataset preparation](#mirrored-dataset-preparation)
- [Half-Object Dataset preparation](#ho-dataset-preparation)
- [Standard generative evaluation](#standard-generative-evaluation)
- [Symmetry measurement protocol](#symmetry-measurement-protocol)
- [Repository structure](#repository-structure)
- [Citation](#citation)

## Install

### Dependencies

- Conda 22.9.0 or newer.
- CUDA 11.6 for GPU-based baseline inference and evaluation environments.
- The replication notebooks and symmetry-measurement utilities can be run with the provided Conda environment file.

### Setup the environment

Install from the Conda file:

```bash
conda env create --file env.yml
conda activate symmetry-matters
```

Some model-specific forks may require building local CUDA/C++ extensions inside their own environments.

Full training or inference pipelines for each baseline should follow the installation instructions in the corresponding fork listed below.

## Reproducing the paper

The committed JSON files under `Replicability/` contain the precomputed NSCD and
standard evaluation results consumed by the paper notebooks. Run both notebooks
from the repository root:

```bash
Replicability/Tables.ipynb
Replicability/Figures.ipynb
```

The expected behavior is:

| Notebook | Reproduces | Output |
|---|---|---|
| [`Replicability/Tables.ipynb`](Replicability/Tables.ipynb) | Tables 1-4 from the main paper | Rendered tables in notebook cell outputs |
| [`Replicability/Figures.ipynb`](Replicability/Figures.ipynb) | Main-paper Figures 2, 3, 4, 5, and 7 | PDF files in `Replicability/generated_figures/` |

`Tables.ipynb` reads Table 3 values from the saved outputs of the notebooks in
`Replicability/table3_results/`. Recomputing those values from point clouds is
more expensive and requires the corresponding files under
`Replicability/samples_table3/`. The LION latent-space experiment has additional
setup instructions in
[`Replicability/table3_results/GenerateLatent_fromVAE.md`](Replicability/table3_results/GenerateLatent_fromVAE.md).

## Baseline forks

Our experiments use dedicated forks of the evaluated models. Each fork contains
the model-specific training and inference changes used in the paper.

| Model | Repository |
|---|---|
| PVD | [NicolasPCS/PVD_necs](https://github.com/NicolasPCS/PVD_necs) |
| DiT-3D | [NicolasPCS/DiT-3D_necs](https://github.com/NicolasPCS/DiT-3D_necs) |
| XCube | [NicolasPCS/XCube_necs](https://github.com/NicolasPCS/XCube_necs) |
| SPVD-S and SPVD-L | [NicolasPCS/SPVD_necs](https://github.com/NicolasPCS/SPVD_necs) |
| SLIDE-3D | [NicolasPCS/SLIDE_necs](https://github.com/NicolasPCS/SLIDE_necs) |
| DPM | [NicolasPCS/diffusion-point-cloud_necs](https://github.com/NicolasPCS/diffusion-point-cloud_necs) |
| LION | [NicolasPCS/LION_necs](https://github.com/NicolasPCS/LION_necs) |

To generate samples from a model, install its fork, download the corresponding
checkpoint, and follow the inference command documented in that fork. Because
these models use substantially different environments and output formats, this
repository provides their generated samples in a common layout for a simpler
evaluation workflow.

## Checkpoints and generated samples

### Checkpoints

Download the released checkpoints and place them in their model-specific
subdirectories under `Baselines_Checkpoints/`. The expected directory layout,
download link, and availability notes are documented in
[`Baselines_Checkpoints/README.md`](Baselines_Checkpoints/README.md).

### Generated samples

Pre-generated samples are provided so that the paper metrics can be recomputed
without installing and running every baseline. Download and extraction
instructions are available in [`Samples/README.md`](Samples/README.md).

The samples are organized by model and include outputs from the original,
mirrored-object, and proposed symmetry-aware experimental conditions, together
with the corresponding reference sets used for evaluation.

## Mirrored Dataset preparation

The experiments use the ShapeNetCore.v2.PC15k point-cloud dataset. It can be downloaded from the
[ShapeNetCore.v2.PC15k folder](https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j?usp=sharing).
We use the following ShapeNet categories:

| Category | Synset ID |
|---|---|
| Airplane | `02691156` |
| Car | `02958343` |
| Chair | `03001627` |

### Mirrored point-cloud dataset for PVD and LION

Create the output directory tree first, then run the NPY preparation script for
each category and split. For example:

```bash
mkdir -p data/Mirrored_ShapeNetCore.v2.PC15k/02691156/train

python Dataset_Preparation//home/isipiran/Symmetrization-of-3D-Generative-Models/Dataset_Preparation/MirrorPCsNPY.py \
  data/ShapeNetCore.v2.PC15k/02691156/train \
  data/Mirrored_ShapeNetCore.v2.PC15k/02691156/train
```

Repeat this command for `train`, `val`, and `test`, and for all three categories.

### SLIDE-3D dataset

SLIDE-3D uses per-shape `pointcloud.npz` files. Prepare its mirrored dataset with:

```bash
python Dataset_Preparation/MirrorPCsNPZ_DatasetForSLIDE3D.py \
  --input_path <SLIDE_INPUT_CATEGORY_DIRECTORY> \
  --output_path <SLIDE_MIRRORED_CATEGORY_DIRECTORY>
```

The script mirrors the geometry and recomputes point normals with Open3D.

### XCube dataset

XCube requires fVDB and serialized sparse grids. Install the XCube environment
as described in [NicolasPCS/XCube_necs](https://github.com/NicolasPCS/XCube_necs),
then convert the prepared NPY data with:

```bash
python Dataset_Preparation/CreatePKLFiles_DatasetForXCube.py \
  <NPY_CATEGORY_DIRECTORY> \
  <XCUBE_OUTPUT_DIRECTORY> \
  <IS_COARSE>
```

Create separate outputs for the coarse (`128`) and fine (`512`) levels, following
the hierarchy described at the top of the conversion script.

## Half-Object Dataset preparation

### Mirrored point-cloud dataset for PVD and LION

Create the output directory tree first, then run the NPY preparation script for
each category and split. For example:

```bash
mkdir -p data/HalfObject_ShapeNetCore.v2.PC15k/02691156/train

python Dataset_Preparation//home/isipiran/Symmetrization-of-3D-Generative-Models/Dataset_Preparation/MirroredPCsNPY.py \
  data/ShapeNetCore.v2.PC15k/02691156/train \
  data/HalfObject_ShapeNetCore.v2.PC15k/02691156/train
```

Repeat this command for `train`, `val`, and `test`, and for all three categories.

For SLIDE-3D and XCube, follow the same procedure as in the Mirrored Dataset preparation using the new half-objects dataset.

## Standard generative evaluation

We use the same 1-NNA and coverage evaluation protocol as
[LION](https://github.com/nv-tlabs/LION#evaluate-the-samples-with-the-1-nna-metrics).
Chamfer Distance and Earth Mover's Distance are used as the underlying distance
metrics. All results reported in this project were computed with
`norm_box=True`, matching LION's normalized ShapeNet evaluation setting.

`Evaluation/compute_scores.py` imports LION's `utils.eval_helper`, so make the
LION fork available on `PYTHONPATH`.

Run the command in the environment created for LION, since its evaluation code
depends on compiled Chamfer Distance and EMD extensions. Replace the sample and
reference paths with the desired model, category, and experimental condition.
The output JSON return the computed metrics.

## Symmetry measurement protocol

The normalized symmetry Chamfer distance (NSCD) compares each point cloud with
its reflection across the YZ plane. Before measuring Chamfer Distance, the
protocol centers the cloud along the x-axis and downsamples clouds with more
than 2,048 points using farthest-point sampling. The Chamfer Distance is then
normalized by the point cloud's bounding-box diagonal.

The implementation accepts directories containing `npy`, `pkl`, or `xyz` point
clouds. Run it from the repository root as follows:

```bash
python Symmetry_Measurement_Protocol/Symmetry_NormalizedPCs.py \
  <POINT_CLOUD_DIRECTORY> \
  Replicability/computed_NSCD/my_symmetry_results.json \
  npy
```

Replace `npy` with `pkl` or `xyz` when appropriate. The resulting JSON stores both raw and normalized Chamfer
Distance values for each shape. We perform our analysis using the normalized Chamfer Distance values.

## Repository structure

```text
Symmetrization-of-3D-Generative-Models/
|-- Baselines_Checkpoints/       # Model-specific checkpoints and instructions
|-- Dataset_Preparation/         # Scripts to create the different version of the datasets
|-- Evaluation/                  # LION-based 1-NNA and COV evaluation entry point
|-- Replicability/
|   |-- Tables.ipynb             # Reproduces Tables 1-4
|   |-- Figures.ipynb            # Reproduces the main quantitative figures
|   |-- computed_NSCD/           # Per-shape symmetry measurements
|   |-- computed_NSCD_PER_EPOCHS/
|   |-- metric_results/          # Standard evaluation results
|   |-- table3_results/          # Table 3 computation notebooks
|   `-- generated_figures/       # Figure outputs
|-- Samples/                     # Generated and reference point-cloud sets
`-- Symmetry_Measurement_Protocol/
    |-- Householder_transform.py
    |-- ChamferDistance.py
    `-- FarthestPointSampling.py
```

## Citation

If you use this repository or the symmetry evaluation protocol, please cite:

```bibtex
@article{caytuiro2025symmetry,
  title   = {Symmetry Matters: Auditing and Symmetrizing 3D Generative Models},
  author  = {Caytuiro, Nicolas and Sipiran, Ivan},
  journal = {arXiv preprint arXiv:2512.18953},
  year    = {2025}
}
```

## Acknowledgements

This project builds on the original implementations of PVD, DiT-3D, XCube,
SPVD, SLIDE-3D, DPM, and LION. We thank their authors for releasing their code,
models, datasets, and evaluation tools.
