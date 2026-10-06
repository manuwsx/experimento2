# Table 3 LION(E) Results

`GenerateLatent_fromVAE.ipynb` reproduces the LION encoder inspection row reported as `LION (E)` in Table 3. This notebook uses the original LION VAE implementation and compares latent-space distances between original point clouds and their mirrored counterparts.

## External Requirements

Install the original LION code and its environment following the upstream instructions:

- LION repository: https://github.com/nv-tlabs/LION
- LION project page: https://nv-tlabs.github.io/LION
- Forked LION's repository: https://github.com/NicolasPCS/LION_necs

The notebook imports the LION modules directly from this repository.

## Dataset

Download the original `ShapeNetCore.v2.PC15k` point-cloud dataset used by PointFlow/LION:

- PointFlow repository: https://github.com/stevenygd/PointFlow
- ShapeNetCore.v2.PC15k Google Drive folder: https://drive.google.com/drive/folders/1MMRp7mMvRj8-tORDaGTJvrAeCMYTWU2j?usp=sharing

Place it under:

```text
Replicability/datasets/ShapeNetCore.v2.PC15k/
```

The expected category folders are:

```text
Replicability/datasets/ShapeNetCore.v2.PC15k/02691156/val/  # airplane
Replicability/datasets/ShapeNetCore.v2.PC15k/02958343/val/  # car
Replicability/datasets/ShapeNetCore.v2.PC15k/03001627/val/  # chair
```

Create the mirrored dataset from the original `.npy` point clouds using:

```bash
python Dataset_Preparation/MirroredPCsNPY.py
```

The mirrored dataset should be placed under:

```text
Replicability/datasets/Mirrored_ShapeNetCore.v2.PC15k/
```

with the same category and split structure as the original dataset.

## Checkpoints

The notebook expects the original and mirrored-dataset LION VAE checkpoints under `Baselines_Checkpoints`:

```text
Baselines_Checkpoints/LION/
├── original/
│   ├── cfg_airplane.yml
│   ├── cfg_car.yml
│   ├── cfg_chair.yml
│   ├── vae_only_airplane.pt
│   ├── vae_only_car.pt
│   └── vae_only_chair.pt
└── mirrored/
    ├── cfg_airplane.yml
    ├── cfg_car.yml
    ├── cfg_chair.yml
    ├── vae_airplane_epoch_7999_iters_175999.pt
    ├── vae_car_epoch_7999_iters_151999.pt
    └── vae_chair_epoch_7999_iters_287999.pt
```

The `original/` checkpoints correspond to the original ShapeNet training data. The `mirrored/` checkpoints correspond to models trained with the mirrored-object dataset.

## Notebook Paths

`GenerateLatent_fromVAE.ipynb` uses paths relative to the project root:

```python
REPLICABILITY_DIR = PROJECT_ROOT / "Replicability"
BASELINES_CHECKPOINTS_DIR = PROJECT_ROOT / "Baselines_Checkpoints"
DATASETS_DIR = REPLICABILITY_DIR / "datasets"
LION_CHECKPOINTS_DIR = BASELINES_CHECKPOINTS_DIR / "LION"
LION_ORIGINAL_CHECKPOINTS_DIR = LION_CHECKPOINTS_DIR / "original"
LION_MIRRORED_CHECKPOINTS_DIR = LION_CHECKPOINTS_DIR / "mirrored"
ORIGINAL_SHAPENET_DIR = DATASETS_DIR / "ShapeNetCore.v2.PC15k"
MIRRORED_SHAPENET_DIR = DATASETS_DIR / "Mirrored_ShapeNetCore.v2.PC15k"
```

Run the notebook from either the project root, `Replicability/`, or `Replicability/table3_results/`; the path setup cell resolves `PROJECT_ROOT` for these locations.
