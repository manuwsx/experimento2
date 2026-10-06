"""
Use shapenet_psr as input path and shapenet_psr_<variant> as output path.

The input and the ouput directories must have the following directory structure:

shapenet_psr_<variant>
    - 02691156
        ...
        test.lst
        train.lst
        val.lst
    - 02958343
        ...
        test.lst
        train.lst
        val.lst
    - 03001627
        ...
        test.lst
        train.lst
        val.lst
"""

import numpy as np
import open3d as o3d
import os
import argparse

def compute_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)
    return normals

# To rotate the PC to the left
R_left = np.array([
    [ 0.0, 0.0,  1.0],
    [ 0.0, 1.0,  0.0],
    [-1.0, 0.0,  0.0]
])

# Argument parser
parser = argparse.ArgumentParser(description="Create Dataset for SLIDE3D")
parser.add_argument("--input_path", type=str, help="Input path")
parser.add_argument("--output_path", type=str, help="Output path")

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

processed = 0
skipped = 0

for model_id in sorted(os.listdir(input_path)):
    model_input_dir = os.path.join(input_path, model_id)
    if not os.path.isdir(model_input_dir):
        skipped += 1
        continue

    input_npz = os.path.join(model_input_dir, "pointcloud.npz")
    if not os.path.isfile(input_npz):
        print(f"[WARN] No pointcloud.npz en {model_id}")
        skipped += 1
        continue

    print(f"Processing: {model_id}")

    # Load file
    data = np.load(input_npz)

    points = data['points']

    # Compute the centroid of the bounding
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_center = (bbox_max + bbox_min) / 2.0

    # Move cloud to x = 0
    normalized_pc = points - bbox_center

    normalized_pc = normalized_pc @ R_left.T

    # Filter points with coordinate x <= 0 (left half with respect to the plane YZ)
    half_cloud_left = normalized_pc[normalized_pc[:, 0] <= 0]
    half_cloud_right = normalized_pc[normalized_pc[:, 0] >= 0]

    half_cloud_left[:, 0] *= -1

    # Join with the original cloud
    full_points = np.concatenate([half_cloud_left, half_cloud_right], axis=0)

    R_right = R_left.T
    full_points = full_points @ R_right.T

    # Compute normals
    normals = compute_normals(full_points)

    # Save the resulting half
    model_output_dir = os.path.join(output_path, model_id)
    os.makedirs(model_output_dir, exist_ok=True)

    output_npz = os.path.join(model_output_dir, "pointcloud.npz")
    np.savez(output_npz, points=full_points, normals=normals)

print("Processing completed")
print(f"Processed: {processed}")
print(f"Ingnored: {skipped}")