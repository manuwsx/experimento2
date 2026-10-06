import torch
import os
from .utils import get_given_dataset_files_list, get_dataset_objects_given_file, save_axes_to_txt
from implementation.utils import load_mesh, DINOWrapper, sampling_fun
from implementation.compute import compute_features
from implementation.compute import compute_axis
from pytorch3d.ops import sample_points_from_meshes

def axis_calc(dataset_folder, output_folder, objects_file_path=None, gpu_device="cuda:0", ds_small=True, ds_reg=True, rot_aug=3, view_samp=1, fm_samp=10000, pc_samp=None, view=None, index_num_list=None):
    """
    ### rot_aug=(0,1,2,3,4)  ---> (0) VS (0, 180) VS (0, 90, 270) VS (0, 90, 180, 270) VS (0, 0, 0, 0)
    ### view_samp=(0,1)  ---> (standard,fibonacci)
    ### fm_samp=(NONE,N) ---> raw mesh VS N feature-mesh sampling (after feature extraction)
    ### pc_samp=(NONE,N) ---> raw mesh VS N point cloud sampling (before feature extraction)
    ### views ---> List of views to evaluate (e.g. [6, 14, 26, 42, 62, 86, 114])
    ### index_num_list ---> List of index numbers to evaluate (e.g. [10, 20, 30, 40, 50])
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
    for idx, obj_file_path in enumerate(files_path_list):
        real_index_num_list = []
        for idx2, index_num in enumerate(index_num_list):
            res_file_name = os.path.basename(obj_file_path).replace(".obj", "_res.txt")
            path_name = f"axis_eval_{ds_small}_{ds_reg}_{rot_aug}_{view_samp}_{fm_samp}_{pc_samp}_{view}_{index_num}"
            result_path = os.path.join(output_folder, path_name, res_file_name)
            if not os.path.exists(result_path):
                real_index_num_list.append(index_num)
        if len(real_index_num_list) == 0:
            print(f"All axes for {obj_file_path} already computed, skipping...")
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
        torch.cuda.empty_cache()
        ######################################### calcula los planos
        for idx2, index_num in enumerate(real_index_num_list):
            if not pc_samp and not fm_samp: # raw mesh
                if object_points.shape[0] > 100000:
                    print(f"\t\t {idx2} Mesh too large, skipping axis calculation", flush=True)
                    axis = [[0, 0, 0],], [[0, 0, 0],], [0,]
                else:
                    print(f"\t\t {idx2} Computing axis {index_num}", flush=True)
                    axis = compute_axis(object_points, features, device, index_num)
            else:
                print(f"\t\t {idx2} Computing axis {index_num}", flush=True)
                axis = compute_axis(object_points, features, device, index_num)
            ######################################### save results
            res_file_name = os.path.basename(obj_file_path).replace(".obj", "_res.txt")
            path_name = f"axis_eval_{ds_small}_{ds_reg}_{rot_aug}_{view_samp}_{fm_samp}_{pc_samp}_{view}_{index_num}"
            result_path = os.path.join(output_folder, path_name, res_file_name)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            save_axes_to_txt(result_path, axis)
            del axis
            torch.cuda.empty_cache()
        del features
        torch.cuda.empty_cache()
        print(f"\tAxis computed and saved")
    print("Results saved in:", result_path)
    print("Done!")