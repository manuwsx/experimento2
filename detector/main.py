import torch
import sys
import os
from implementation.utils import load_mesh, DINOWrapper, sampling_fun
from implementation.compute import compute_features, compute_planes, compute_axis
from pytorch3d.ops import sample_points_from_meshes
from implementation.visual import InteractiveRenderer

import numpy as np

if __name__ == "__main__":
    if len(sys.argv) > 3:
        mesh_path = sys.argv[1]
        symm_type = sys.argv[2].lower()
        feature_type = sys.argv[3]
        if symm_type not in ["planes", "axes"]:
            print("Please provide a valid symmetry type: 'planes' or 'axes'.")
            exit()
        if feature_type not in ["RM", "FM", "PC"]:
            print("Please provide a valid feature type: 'RM', 'FM', or 'PC'.")
            exit()
    else:
        print("Usage: python main.py <mesh_path> <symm_type: planes|axes> <feature_type: RM|FM|PC>")
        exit()


    if symm_type == "planes":
        device = torch.device("cuda:0")
        model = DINOWrapper(device=device, small=True, reg=True)
        name = os.path.splitext(os.path.basename(mesh_path))[0]
        mesh = load_mesh(mesh_path, device)

        renderer = InteractiveRenderer()

        if feature_type == "RM":
            print("Using Raw-Mesh features")
            features = compute_features(mesh, model, device, view_quantity=114)
            normals, points, _ = compute_planes(mesh, features, device)
            renderer.initialize_withmesh(mesh, name, "./img", 800, 600, features=features.cpu().numpy())
        elif feature_type == "PC":
            print("Using Point Cloud features")
            point_cloud = sample_points_from_meshes(mesh, 10000).squeeze(0) 
            features = compute_features(point_cloud, model, device, view_quantity=114)
            normals, points, _ = compute_planes(point_cloud, features, device)
            renderer.initialize_withmesh(mesh, name, "./img", 800, 600)
            renderer.add_point_cloud(point_cloud.cpu().numpy(), features.cpu().numpy())
        elif feature_type == "FM": # BEST RESULTS
            print("Using Feature-Mesh Sampling features")
            features = compute_features(mesh, model, device, view_quantity=114)
            pc, fe = sampling_fun(mesh, features, device, 10000)
            normals, points, _ = compute_planes(pc, fe, device)
            renderer.initialize_withmesh(mesh, name, "./img", 800, 600, features=features.cpu().numpy())
            renderer.add_point_cloud(pc.cpu().numpy(), fe.cpu().numpy())

        for idx in range(len(points)):
            point = points[idx].cpu().numpy()
            normal = normals[idx].cpu().numpy()
            renderer.add_pre_plane(point, normal, idx)
        renderer.start_interactive_session()

    elif symm_type == "axes":
        device = torch.device("cuda:0")
        model = DINOWrapper(device=device, small=True, reg=True)
        name = os.path.splitext(os.path.basename(mesh_path))[0]
        mesh = load_mesh(mesh_path, device)

        renderer = InteractiveRenderer()

        if feature_type == "RM": # BEST RESULTS
            print("Using Raw-Mesh features")
            features = compute_features(mesh, model, device, view_quantity=114)
            normals, points, _ = compute_axis(mesh, features, device, 50)
            renderer.initialize_withmesh(mesh, name, "./img", 800, 600, features=features.cpu().numpy())
        elif feature_type == "PC":
            print("Using Point Cloud features")
            point_cloud = sample_points_from_meshes(mesh, 10000).squeeze(0) 
            features = compute_features(point_cloud, model, device, view_quantity=114)
            normals, points, _ = compute_axis(point_cloud, features, device, 50)
            renderer.initialize_withmesh(mesh, name, "./img", 800, 600)
            renderer.add_point_cloud(point_cloud.cpu().numpy(), features.cpu().numpy())
        elif feature_type == "FM":
            print("Using Feature-Mesh Sampling features")
            features = compute_features(mesh, model, device, view_quantity=114)
            pc, fe = sampling_fun(mesh, features, device, 10000)
            normals, points, _ = compute_axis(pc, fe, device, 50)
            renderer.initialize_withmesh(mesh, name, "./img", 800, 600, features=features.cpu().numpy())
            renderer.add_point_cloud(pc.cpu().numpy(), fe.cpu().numpy())

        for idx in range(len(points)):
            point = points[idx].cpu().numpy()
            normal = normals[idx].cpu().numpy()
            renderer.add_pre_axis(point, normal, idx)
        renderer.start_interactive_session(look_for_axis=True)