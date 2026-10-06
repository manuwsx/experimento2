import torch
import os
from .utils import get_given_dataset_files_list, get_dataset_objects_given_file, save_planes_to_txt
from implementation.utils import load_mesh, DINOWrapper, sampling_fun
from implementation.compute import compute_features
from implementation.compute import compute_planes
from pytorch3d.ops import sample_points_from_meshes

def planes_calc(dataset_folder, output_folder, objects_file_path=None, gpu_device="cuda:0", ds_small=True, ds_reg=True, ds_vggt=False, rot_aug=3, view_samp=1, fm_samp=10000, pc_samp=None, view=None):
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
    for idx, obj_file_path in enumerate(files_path_list):
        res_file_name = os.path.basename(obj_file_path).replace(".obj", "_res.txt")
        if ds_vggt:
            path_name = f"planes_eval_vggt_{rot_aug}_{view_samp}_{fm_samp}_{pc_samp}_{view}"
        else:
            path_name = f"planes_eval_{ds_small}_{ds_reg}_{rot_aug}_{view_samp}_{fm_samp}_{pc_samp}_{view}"
        result_path = os.path.join(output_folder, path_name, res_file_name)
        if os.path.exists(result_path):
            print(f"File {result_path} already exists, skipping...")
            continue
        ######################################### loads mesh
        original_mesh = load_mesh(obj_file_path, device)
        object_points = original_mesh.verts_packed()
        print(idx, "Vertices:", object_points.shape[0], "File:", obj_file_path, flush=True)
        if pc_samp:
            object_points = sample_points_from_meshes(original_mesh, pc_samp).squeeze(0) # (N, 3)
            print("\tUsing Point-cloud sampling instead of Mesh:", object_points.shape[0], flush=True)
        else:
            if fm_samp:
                print("\tUsing Feature-mesh sampling", fm_samp, flush=True)
            else:
                print("\tUsing Raw-mesh", flush=True)
        ######################################### computes features
        if pc_samp:
            features = compute_features(object_points, model, device, view_samp=view_samp, view_quantity=view, rot_aug=rot_aug)
        else:
            features = compute_features(original_mesh, model, device, view_samp=view_samp, view_quantity=view, rot_aug=rot_aug)
        if fm_samp:
            object_points, features = sampling_fun(original_mesh, features, device, num_points=fm_samp)
        ######################################### calcula los planos
        planes = compute_planes(object_points, features, device)
        ######################################### save results
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        save_planes_to_txt(result_path, planes)
        print(f"\tPlanes computed and saved")
        # empty cache
        del features
        del planes
        torch.cuda.empty_cache()
    #########################################
    print("Results saved in:", result_path)
    print("Done!")