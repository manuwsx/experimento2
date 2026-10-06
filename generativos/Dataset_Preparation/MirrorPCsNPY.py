"""
Use ShapeNetCore.v2.PC15k as input path and HalfObject_ShapeNetCore.v2.PC15k as output path.

The input and the ouput directories must have the following directory structure:

HalfObject_ShapeNetCore.v2.PC15k
    - 02691156
        - test
        - train
        - val
    - 02958343
        - test
        - train
        - val
    - 03001627
        - test
        - train
        - val
"""

import os
import json
import numpy as np
import open3d as o3d
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Create PKL files from PC data")
parser.add_argument("input_path", type=str, help="Path to the input point cloud directory")
parser.add_argument("output_path", type=str, help="Path to the output directory")

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

cont = 0

for filename in os.listdir(input_path):
    if filename.endswith(".npy"):
        file_path = os.path.join(input_path, filename)
        point_cloud = np.load(file_path)

        # Compute the centroid of the bounding
        bbox_min = np.min(point_cloud, axis=0)
        bbox_max = np.max(point_cloud, axis=0)
        bbox_center = (bbox_max + bbox_min) / 2.0

        # Move cloud to x = 0
        normalized_pc = point_cloud - bbox_center

        # Filter points with coordinate x <= 0 (left half with respect to the plane YZ)
        half_cloud_left = normalized_pc[normalized_pc[:, 0] <= 0]
        half_cloud_right = normalized_pc[normalized_pc[:, 0] >= 0]

        half_cloud_left[:, 0] *= -1

        # Join with the original cloud
        full_points = np.concatenate([half_cloud_left, half_cloud_right], axis=0)

        # Save the resulting half
        output_file = os.path.join(output_path, filename)
        np.save(output_file, full_points)

        print(f"Saved cloud {cont} with {full_points.shape[0]} points.")
        
        cont += 1