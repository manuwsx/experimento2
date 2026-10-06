"""
Fist, install fvdb. Please follow the instructions of the original XCube repository: https://github.com/NicolasPCS/XCube_necs/tree/main?tab=readme-ov-file
Use ShapeNetCore.v<CHANGE>.PC15k as input path and XCube_Dataset (New dataset) as output path.

The ouput directory must have the following directory structure:

XCube_Dataset
    - 128
        - 02691156
        - 02958343
        - 03001627
    - 512
        - 02691156
        - 02958343
        - 03001627
"""


import fvdb
import os
import torch
import numpy as np
import open3d as o3d
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Create PKL files from PC data")
parser.add_argument("input_path", type=str, help="Path to the input point cloud directory")
parser.add_argument("output_path", type=str, help="Path to the output directory")
parser.add_argument("is_coarse", type=bool, help="Whether the voxel grid is coarse-level (true/false)")

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
is_coarse = args.is_coarse

def compute_fpdb(input_xyz, input_normal, split_out):
    # Note that coarse-level voxel set to True and fine-level voxel set to False
    build_splatting = is_coarse
    voxel_size = 0.0025 # 0.01 for coarse-level, 0.0025 for fine-level

    # Note that input_xyz, input_normal are torch tensors of shape [N, 3], [N, 3]
    if build_splatting:
        target_grid = fvdb.sparse_grid_from_nearest_voxels_to_points(
            fvdb.JaggedTensor(input_xyz), voxel_sizes=voxel_size, origins=[voxel_size / 2.] * 3)
    else:
        target_grid = fvdb.sparse_grid_from_points(
            fvdb.JaggedTensor(input_xyz), voxel_sizes=voxel_size, origins=[voxel_size / 2.] * 3)
        
    # get target normal
    target_normal = target_grid.splat_trilinear(fvdb.JaggedTensor(input_xyz), fvdb.JaggedTensor(input_normal))
    target_normal.jdata /= (target_normal.jdata.norm(dim=1, keepdim=True) + 1e-6)

    save_dict = {
        "points": target_grid.to("cpu"),
        "normals": target_normal.cpu(),
    }

    torch.save(save_dict, split_out)

def compute_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)
    return normals

splits = ["train", "val", "test"]

for split in splits:
    print("Processing split:", split)

    split_in = os.path.join(input_path, split)
    split_out = output_path
    lst_path = os.path.join(output_path, f"{split}.lst")

    os.makedirs(split_out, exist_ok=True)

    names = []

    for fname in os.listdir(split_in):
        if fname.endswith(".npy"):
            npy_path = os.path.join(split_in, fname)
            names.append(fname[:-4])

            try:
                # Load point cloud data
                point_cloud = np.load(npy_path).astype(np.float32)
                point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

                # Compute normals
                normals_np = compute_normals(point_cloud.numpy())
                normals = torch.tensor(normals_np, dtype=torch.float32)

                # Create FVDB grid and splat normals
                out_file = os.path.join(split_out, fname.replace(".npy", ".pkl"))
                compute_fpdb(point_cloud, normals, out_file)

            except Exception as e:
                print(f"Error processing {npy_path}: {e}")
                continue

    # write .lst
    with open(lst_path, "w") as f:
        for name in names:
            f.write(f"{name}\n")

    print(f"Finished processing split: {split}")