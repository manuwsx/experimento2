# Generated Samples

The generated and reference point clouds used in the paper will be uploaded to
the following Google Drive folder:

**[Download checkpoints and samples from Google Drive](https://drive.google.com/drive/folders/1ilCFuFbfBObzQprOLV-_uE5kP3FivZgD?usp=sharing)**

Download the sample archive, extract it, and place each model directory directly
under `Samples/` so that the repository has the following layout:

```text
Samples/
|-- DiT-3D/
|-- DPM/
|-- LION/
|-- PVD/
|-- SLIDE-3D/
|-- SPVD-L/
|-- SPVD-S/
`-- XCube/
```

Do not add an extra archive-level directory between `Samples/` and the model
folders. The replication notebooks and saved metric files use this layout.

## File naming

Each model directory contains samples for airplane, car, and chair:

| Pattern | Description |
|---|---|
| `generated_<model>_<category>.pth` | Samples from the original baseline |
| `generated_ours_<model>_<category>.pth` | Samples from the proposed symmetry-aware method |
| `ablation_<model>_<category>.pth` | Samples from the mirrored-object training condition |
| `reference_<model>_<category>.pth` | Reference set paired with the original or mirrored-object condition |
| `reference_ours_<model>_<category>.pth` | Reference set paired with the symmetry-aware condition |

These files allow the standard LION-based evaluation to be rerun without
installing every baseline or regenerating all samples. See the
[main README](../README.md#standard-generative-evaluation) for the exact
`Evaluation/compute_scores.py` command and the required `norm_box=True` setting.

For full inference from checkpoints, install the corresponding model fork and
follow its model-specific instructions listed in the
[main README](../README.md#baseline-forks).
