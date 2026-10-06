"""
Use ShapeNetCore.v2.PC15k as input path and Mirrored_ShapeNetCore.v2.PC15k as output path.

The input and the ouput directories must have the following directory structure:

Mirrored_ShapeNetCore.v2.PC15k
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

import sys
import os
import json
import torch
import numpy as np
import open3d as o3d
from Symmetry_Measurement_Protocol.Householder_transform import householder_transformation
from Symmetry_Measurement_Protocol.FarthestPointSampling import farthest_point_sampling

# Argument parser
parser = argparse.ArgumentParser(description="Create half-object point clouds")
parser.add_argument("input_path", type=str, help="Path to the input point cloud directory")
parser.add_argument("output_path", type=str, help="Path to the output directory")

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

files = sorted([f for f in os.listdir(input_path) if f.endswith(".npy")])
cont = 0
num_points = 15000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for filename in files:
    file_path = os.path.join(input_path, filename)
    point_cloud = np.load(file_path)

    point_cloud = torch.tensor(point_cloud).unsqueeze(0) # [1, N, 3]
    point_cloud = point_cloud.to(device)

    # Clonar la nube
    cloned_pc = point_cloud.clone()

    # Aplicar transformación de householder
    cloned_pc = cloned_pc.squeeze(0).cpu().numpy()
    cloned_pc = householder_transformation(cloned_pc)
    cloned_pc = torch.tensor(cloned_pc).unsqueeze(0)
    cloned_pc = cloned_pc.to(device)

    # Unir la nube original con T(O)
    full_pc = torch.cat((point_cloud, cloned_pc), dim=1) # [1, 2N, 3]

    full_pc = full_pc.squeeze(0).cpu().numpy()
    full_pc = farthest_point_sampling(full_pc, num_points)

    full_pc = torch.tensor(full_pc).unsqueeze(0)
    full_pc = full_pc.to(device)

    # Guardar la nueva nube de puntos
    out_name = os.path.splitext(filename)[0] + ".npy"
    out_path = os.path.join(output_path, out_name)
    np.save(out_path, full_pc.squeeze(0).cpu().numpy())

    cont += 1

print(f"Done {cont}!!!")