import os
import sys
import numpy as np
import json
import torch
from pathlib import Path
from tqdm import tqdm

# Add necessary paths
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(EXPERIMENT_DIR, 'detector'))
sys.path.append(os.path.join(EXPERIMENT_DIR, 'generativos', 'Symmetry_Measurement_Protocol'))

from implementation.utils import DINOWrapper
from implementation.compute import compute_features, compute_planes
from FarthestPointSampling import farthest_point_sampling
from ChamferDistance import chamfer_distance

def compute_diagonal(points):
    """
    Computes the bounding box diagonal of the point cloud.
    """
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    return np.sqrt(dx**2 + dy**2 + dz**2)

def reflect_points_across_plane(points, normal, midpoint):
    """
    Reflects points across a plane defined by normal and midpoint.
    points: (N, 3) numpy array
    normal: (3,) numpy array (must be normalized)
    midpoint: (3,) numpy array
    """
    normal = normal / np.linalg.norm(normal)
    v = points - midpoint
    projection = np.dot(v, normal)
    reflected_points = points - 2 * np.outer(projection, normal)
    return reflected_points

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Symmetry Planes and calculate NCSD")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to ShapeNet root (e.g., data/ShapeNetCore.v2.PC15k)")
    parser.add_argument("--output_json", type=str, default="ncsd_results.json", help="Output JSON path")
    parser.add_argument("--max_objects", type=int, default=1000, help="Max objects per category")
    parser.add_argument("--category", type=str, default="all", choices=["all", "plane", "car", "chair"], help="Category to process")
    args = parser.parse_args()

    # Dictionary mapping category name to synset ID
    categories = {
        "plane": "02691156",
        "car": "02958343",
        "chair": "03001627"
    }

    if args.category != "all":
        categories = {args.category: categories[args.category]}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading DINO model on {device}...")
    model = DINOWrapper(device=device, small=True, reg=True)
    
    results = {}

    for cat_name, cat_id in categories.items():
        print(f"\nProcessing category: {cat_name} ({cat_id})")
        cat_path = Path(args.dataset_root) / cat_id / "train"
        if not cat_path.exists():
            print(f"Path not found: {cat_path}")
            continue
        
        # Taking npy objects
        files = list(cat_path.glob("*.npy"))[:args.max_objects]
        print(f"Found {len(files)} objects for {cat_name}")
        
        cat_ncsds = []
        
        for file_path in tqdm(files, desc=f"{cat_name}"):
            try:
                # 1. Load object
                points = np.load(file_path)
                
                # Sample to 10000 points for the detector if necessary
                if points.shape[0] > 10000:
                    det_points_np = farthest_point_sampling(points, 10000)
                else:
                    det_points_np = points
                    
                det_points_tensor = torch.tensor(det_points_np, dtype=torch.float32, device=device)
                
                # 2. Compute features and symmetry planes
                features = compute_features(det_points_tensor, model, device, view_quantity=114)
                
                # Suppress printing for every iteration inside compute features/planes if they do it
                # with standard outputs
                normals, midpoints, distances = compute_planes(det_points_tensor, features, device)
                
                if len(normals) == 0:
                    print(f"No symmetry plane found for {file_path.name}")
                    continue
                
                # Choose the plane least deviated from YZ plane
                # YZ plane normal vector is [1, 0, 0] or [-1, 0, 0]
                # Deviation is minimized when dot product absolute value is maximized
                best_idx = -1
                max_abs_x = -1
                for i in range(len(normals)):
                    n = normals[i].cpu().numpy()
                    norm_val = np.linalg.norm(n)
                    if norm_val < 1e-6:
                        continue
                    n = n / norm_val
                    abs_x = abs(n[0])
                    if abs_x > max_abs_x:
                        max_abs_x = abs_x
                        best_idx = i
                        
                if best_idx == -1:
                    print(f"No valid normal found for {file_path.name}")
                    continue
                    
                best_normal = normals[best_idx].cpu().numpy()
                best_midpoint = midpoints[best_idx].cpu().numpy()

                # 3. Calculate NCSD using the chosen plane
                # Sample to 2048 for evaluation
                if points.shape[0] > 2048:
                    eval_points = farthest_point_sampling(points, 2048)
                else:
                    eval_points = points
                    
                # Reflect
                reflected_points = reflect_points_across_plane(eval_points, best_normal, best_midpoint)
                
                # Chamfer distance
                cd = chamfer_distance(eval_points, reflected_points)
                
                # Normalization
                diagonal = compute_diagonal(eval_points)
                if diagonal < 1e-6:
                    diagonal = 1e-6
                ncsd = cd / diagonal
                
                cat_ncsds.append(ncsd)
                results[str(file_path.name)] = {
                    "ncsd": ncsd,
                    "cd": cd,
                    "diagonal": diagonal,
                    "category": cat_name,
                    "best_normal": best_normal.tolist(),
                    "best_midpoint": best_midpoint.tolist()
                }

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
            
        if cat_ncsds:
            avg_ncsd = np.mean(cat_ncsds)
            print(f"Average NCSD for {cat_name}: {avg_ncsd:.5f}")
            results[f"avg_{cat_name}"] = avg_ncsd
            
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
