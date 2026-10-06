import torch
import os
from .utils import get_given_dataset_files_list, get_dataset_objects_given_file
from implementation.utils import load_mesh, DINOWrapper, sampling_fun
from implementation.compute import compute_features
from implementation.compute import compute_planes
from implementation.compute import compute_axis
from pytorch3d.ops import sample_points_from_meshes
import time

def save_time_to_txt(result_path, times):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        for key, value in times.items():
            f.write(f"{key}: {value}\n")

def time_eval_planes(dataset_folder, output_folder, objects_file_path=None, gpu_device="cuda:0", ds_small=True, ds_reg=True, ds_vggt=False, rot_aug=3, view_samp=1, fm_samp=10000, pc_samp=None, view=None):
    """
    ### rot_aug=(0,1,2,3,4)  ---> (0) VS (0, 180) VS (0, 90, 270) VS (0, 90, 180, 270) VS (0, 0, 0, 0)
    ### view_samp=(0,1)  ---> (standard,fibonacci)
    ### fm_samp=(NONE,N) ---> raw mesh VS N feature-mesh sampling (after feature extraction)
    ### pc_samp=(NONE,N) ---> raw mesh VS N point cloud sampling (before feature extraction)
    ### views ---> List of views to evaluate (e.g. [6, 14, 26, 42, 62, 86, 114])
    ### random_pairing ---> random pairing OR feature invariance pairing
    """
    if objects_file_path is None:
        files_path_list = get_given_dataset_files_list(dataset_folder)
    else:
        files_path_list = get_dataset_objects_given_file(dataset_folder, objects_file_path)
    print("Files to evaluate:", len(files_path_list))
    #########################################
    device = torch.device(gpu_device)
    model = DINOWrapper(device, small=ds_small, reg=ds_reg, vggt_dino=ds_vggt)
    #########################################
    if pc_samp:
        fm_samp = None
    #########################################
    if ds_vggt:
        path_name = f"time_eval_planes_vggt_{rot_aug}_{view_samp}_{fm_samp}_{pc_samp}_{view}"
    else:
        path_name = f"time_eval_planes_{ds_small}_{ds_reg}_{rot_aug}_{view_samp}_{fm_samp}_{pc_samp}_{view}"
    result_path = os.path.join(output_folder, path_name + ".txt")
    #########################################
    start_time = time.time()
    times_list_features = []
    times_list_planes = []
    for idx, obj_file_path in enumerate(files_path_list):
        ######################################### loads mesh
        original_mesh = load_mesh(obj_file_path, device)
        mesh_or_cloud = original_mesh
        print(idx, "Vertices:", original_mesh.num_verts_per_mesh().item(), "File:", obj_file_path, flush=True)
        if pc_samp:
            mesh_or_cloud = sample_points_from_meshes(original_mesh, pc_samp).squeeze(0) # (N, 3)
            print("\tUsing Point-cloud sampling instead of Mesh", flush=True)
        elif fm_samp:
            print("\tUsing Feature-mesh sampling", fm_samp, flush=True)
        else:
            print("\tUsing Raw-mesh", flush=True)
        ######################################### computes features
        start_time_features = time.time()
        features = compute_features(mesh_or_cloud, model, device, view_samp=view_samp, view_quantity=view, rot_aug=rot_aug)
        if fm_samp:
            mesh_or_cloud, features = sampling_fun(original_mesh, features, device, num_points=fm_samp)
        elapsed_time_features = time.time() - start_time_features
        times_list_features.append(elapsed_time_features)
        ######################################### calcula los planos
        start_time_planes = time.time()
        planes = compute_planes(mesh_or_cloud, features, device)
        elapsed_time_planes = time.time() - start_time_planes
        times_list_planes.append(elapsed_time_planes)
        print(f"\tPlanes computed", flush=True)
        # empty cache
        del features
        del planes
        torch.cuda.empty_cache()
        # print free memory
        # print(f"\tIn use memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB", flush=True)
        # print(f"\tRemaining memory: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB", flush=True)
    #########################################
    total_time = time.time() - start_time
    avg_time_features = sum(times_list_features) / len(times_list_features)
    avg_time_planes = sum(times_list_planes) / len(times_list_planes)
    avg_total_time = total_time / len(files_path_list)
    times = {
        "Number of Objects": len(files_path_list),
        "Objects from": dataset_folder,
        "Subset file": objects_file_path,
        "Total Time (all objects)": total_time,
        "Average Time per Object": avg_total_time,
        "Average Feature Extraction Time per Object": avg_time_features,
        "Average Plane Computation Time per Object": avg_time_planes
    }
    save_time_to_txt(result_path, times)
    print("Results saved in:", result_path)
    print("Done!")

def time_eval_axes(dataset_folder, output_folder, objects_file_path=None, gpu_device="cuda:0", ds_small=True, ds_reg=True, rot_aug=3, view_samp=1, fm_samp=10000, pc_samp=None, view=None, index_num=None):
    """
    ### rot_aug=(0,1,2,3,4)  ---> (0) VS (0, 180) VS (0, 90, 270) VS (0, 90, 180, 270) VS (0, 0, 0, 0)
    ### view_samp=(0,1)  ---> (standard,fibonacci)
    ### fm_samp=(NONE,N) ---> raw mesh VS N feature-mesh sampling (after feature extraction)
    ### pc_samp=(NONE,N) ---> raw mesh VS N point cloud sampling (before feature extraction)
    ### views ---> List of views to evaluate (e.g. [6, 14, 26, 42, 62, 86, 114])
    ### index_num ---> size of group to evaluate (e.g. 10, 20, 30, 40, 50)
    """
    if objects_file_path is None:
        files_path_list = get_given_dataset_files_list(dataset_folder)
    else:
        files_path_list = get_dataset_objects_given_file(dataset_folder, objects_file_path)
    print("Files to evaluate:", len(files_path_list))
    #########################################
    device = torch.device(gpu_device)
    model = DINOWrapper(device, small=ds_small, reg=ds_reg)
    #########################################
    if pc_samp:
        fm_samp = None
    #########################################
    path_name = f"time_eval_axes_{ds_small}_{ds_reg}_{rot_aug}_{view_samp}_{fm_samp}_{pc_samp}_{view}_{index_num}"
    result_path = os.path.join(output_folder, path_name + ".txt")
    #########################################
    start_time = time.time()
    times_list_features = []
    times_list_axes = []
    for idx, obj_file_path in enumerate(files_path_list):
        ######################################### loads mesh
        original_mesh = load_mesh(obj_file_path, device)
        mesh_or_cloud = original_mesh
        print(idx, "Vertices:", original_mesh.num_verts_per_mesh().item(), "File:", obj_file_path, flush=True)
        if pc_samp:
            mesh_or_cloud = sample_points_from_meshes(original_mesh, pc_samp).squeeze(0) # (N, 3)
            print("\tUsing Point-cloud sampling instead of Mesh", flush=True)
        elif fm_samp:
            print("\tUsing Feature-mesh sampling", fm_samp, flush=True)
        else:
            print("\tUsing Raw-mesh", flush=True)
        ######################################### computes features
        start_time_features = time.time()
        features = compute_features(mesh_or_cloud, model, device, view_samp=view_samp, view_quantity=view, rot_aug=rot_aug)
        if fm_samp:
            mesh_or_cloud, features = sampling_fun(original_mesh, features, device, num_points=fm_samp)
        elapsed_time_features = time.time() - start_time_features
        times_list_features.append(elapsed_time_features)
        ######################################### calcula los ejes
        start_time_axes = time.time()
        axis = compute_axis(mesh_or_cloud, features, device, index_num)
        elapsed_time_axes = time.time() - start_time_axes
        times_list_axes.append(elapsed_time_axes)
        print(f"\tAxes computed", flush=True)
        # empty cache
        del axis
        del features
        torch.cuda.empty_cache()
        # print free memory
        # print(f"\tIn use memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB", flush=True)
        # print(f"\tRemaining memory: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB", flush=True)
    #########################################
    total_time = time.time() - start_time
    avg_time_features = sum(times_list_features) / len(times_list_features)
    avg_time_axes = sum(times_list_axes) / len(times_list_axes)
    avg_total_time = total_time / len(files_path_list)
    times = {
        "Number of Objects": len(files_path_list),
        "Objects from": dataset_folder,
        "Subset file": objects_file_path,
        "Total Time (all objects)": total_time,
        "Average Time per Object": avg_total_time,
        "Average Feature Extraction Time per Object": avg_time_features,
        "Average Axis Computation Time per Object": avg_time_axes
    }
    save_time_to_txt(result_path, times)
    print("Results saved in:", result_path)
    print("Done!")