# Enhanced Back-Projection of Vision Features for 3D Symmetry Detection

[_WACV 2026_](https://wacv.thecvf.com/Conferences/2026)

Project page: [https://spulp.github.io/EnhancedBackProjection/](https://spulp.github.io/EnhancedBackProjection/)

# Important
## Environment Setup

- **Python Version:** 3.12  
- It is recommended to use a virtual environment for package management.
- Requires GPU with CUDA 12.9.

### Creating a Virtual Environment

#### Using `venv`
```bash
python -m venv p12
source p12/bin/activate
```

#### Using Miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```
```bash
conda create -n p12 python=3.12
conda install -n p12 pip
conda activate p12
```

### Installing Required Libraries

All required libraries are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```
Or install them individually:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.8.0cu129
pip install point_cloud_utils
pip install torch_geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
pip install trimesh pyvista scikit-learn
```

### References

This code is inspired by the following repositories:
- [Back-to-3D Few-Shot Keypoints](https://github.com/wimmerth/back-to-3d-few-shot-keypoints)
- [COPS](https://github.com/marco-garosi/COPS)

### Example Usage

To run the main script, use:
```bash
main.py <path_to_obj_file> <planes_or_axes> <mode>
```

- `<path_to_obj_file>`: Path to an `.obj` file (e.g., `objects/example_planes.obj`)
- `<planes_or_axes>`: Specify the planes or axes to use (e.g., `planes`)
- `<mode>`: Choose one of the following modes:
  - `FM`: Feature-Mesh sampling
  - `RM`: Raw-Mesh
  - `PC`: Point Cloud

**Example:**
```bash
main.py objects/example_planes.obj planes FM
main.py objects/example_axes.obj axes RM
```

### Notes on Code Structure and Usage

The code does not provide extensive explanations for each part, but with enough digging (e.g., looking at function names and structure), you can deduce the workflow. The entry point is the `main.py` script, which is straightforward to follow.

When executing, a visual interface pops up. The console will display commands for interacting with the viewer, summarized here:

> **Note:** There is a known bug when using the `PC` mode—the point clouds in the viewer appear larger than normal. However, the computed planes can still be visualized using the original mesh, even though the plane was computed from the point cloud.



---

**Interactive 3D Viewer Controls:**

- Use mouse to rotate/zoom the object
- Press `s` to save current camera view
- Press `r` to reset view
- Press `1` to show/hide point cloud
- Press `2` to show/hide features
- Press `3` to add plane
- Press `q` to quit

---

## Datasets

Datasets are included in the release. The `datasets` folder contains text files that list the object IDs for both the full dataset and the 10% subset.

## Evaluation Scripts

Python scripts for evaluating methods and datasets are located in the `evaluation` folder.

Run them from the repository root:
```bash
python evaluation/script.py <params>
```

### Parameters for Plane/Axis Computation Scripts

Use the following arguments when running the dataset scripts for plane/axis computation:

```python
(
  dataset_folder,
  output_folder,
  objects_file_path=None,
  gpu_device="cuda:0",
  ds_small=True,
  ds_reg=True,
  rot_aug=3,
  view_samp=1,
  fm_samp=10000,
  pc_samp=None,
  ...
)
```

| Parameter | Default | Description |
|---|---|---|
| `dataset_folder` | *(required)* | Path to the input dataset directory. |
| `output_folder` | *(required)* | Path where output files/results are saved. |
| `objects_file_path` | `None` | Optional path to a text file with object IDs to process. If `None`, all objects are processed. |
| `gpu_device` | `"cuda:0"` | Device used for computation (for example, the first CUDA GPU). |
| `ds_small` | `True` | Use the small DinoV2 model variant. |
| `ds_reg` | `True` | Use DinoV2 registers. |
| `rot_aug` | `3` | Number/type of rotation augmentations per object (see mapping below). |
| `view_samp` | `1` | View sampling strategy: `0` = standard, `1` = Fibonacci. |
| `fm_samp` | `10000` | Number of samples for Feature-Mesh mode. |
| `pc_samp` | `None` | Number of samples for Point-Cloud mode. |

**`rot_aug` mapping**

- `0`: `(0)`
- `1`: `(0, 180)`
- `2`: `(0, 90, 270)`
- `3`: `(0, 90, 180, 270)`
- `4`: `(0, 0, 0, 0)`

## Citation

If you find this useful or build on this work, please cite the paper below:

```bibtex
@InProceedings{Aguirre_2026_WACV,
  author    = {Aguirre, Isaac and Sipiran, Ivan},
  title     = {Enhanced Back-Projection of Vision Features for 3D Symmetry Detection},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month     = {March},
  year      = {2026},
  pages     = {66-76}
}
```