import torch
import os
from .utils import get_given_dataset_files_list, get_dataset_objects_given_file, open_axis_txt
from implementation.utils import load_mesh, DINOWrapper, sampling_fun
from implementation.compute import compute_features
from pytorch3d.ops import sample_points_from_meshes

def get_top_L_similar_indices_two_sets(ORI, X, L, batch_size=1000):
    """
    Find the indices of the top L closest feature vectors for each vector in X using Manhattan distance.
    Uses double batching to handle large datasets efficiently and avoid memory issues.
    
    Args:
        X: Input tensor of shape (N, D) where N is number of vectors and D is dimensions
        L: Number of top similar vectors to return
        batch_size: Size of batches for processing
        
    Returns:
        Tensor of shape (N, L) containing indices of the L most similar vectors for each vector
    """
    N = ORI.shape[0]
    M = X.shape[0]
    R = torch.full((N, L), -1, dtype=torch.long, device=X.device)  # Initialize result tensor with -1
    
    # Process input in batches
    for i in range(0, N, batch_size):
        batch_query = ORI[i:i+batch_size]
        batch_size_actual = batch_query.size(0)
        
        # To store top L distances and indices for the current batch
        top_L_distances = torch.full((batch_size_actual, L), float('inf'), device=X.device)
        top_L_indices = torch.full((batch_size_actual, L), -1, dtype=torch.long, device=X.device)
        
        # Compare against X in batches too
        for j in range(0, M, batch_size):
            batch_ref = X[j:j+batch_size]
            
            # Compute distances between current batch pairs
            distances = torch.cdist(batch_query, batch_ref, p=2)  # euclidian distance
            
            # Mask self-similarities for overlapping batches
            if i <= j < i + batch_size_actual:
                mask_start = max(0, i - j)
                mask_end = min(batch_size_actual, batch_size - (j - i))
                row_indices = torch.arange(mask_start, mask_end, device=X.device)
                distances[row_indices, row_indices] = float('inf')
            
            # Concatenate current distances and indices
            batch_indices = torch.arange(j, j + batch_ref.size(0), device=X.device).repeat(batch_size_actual, 1)
            concatenated_distances = torch.cat([top_L_distances, distances], dim=1)
            concatenated_indices = torch.cat([top_L_indices, batch_indices], dim=1)
            
            # Select the top L smallest distances and corresponding indices
            top_L_distances, indices = torch.topk(concatenated_distances, L, dim=1, largest=False)
            top_L_indices = torch.gather(concatenated_indices, 1, indices)
        
        # Update result tensor
        R[i:i+batch_size_actual] = top_L_indices
    
    return R

def get_random_L_indices_two_sets(ORI, X, L):
    N = ORI.shape[0]
    M = X.shape[0]
    R = torch.randint(0, M, (N, L), dtype=torch.long, device=X.device)
    return R


def rotate_points_around_axis(points, axis_normal, axis_point, angle_degrees):
    """
    Rotate points around an arbitrary axis defined by a normal vector and a point on the axis.
    Uses Rodrigues' rotation formula.
    
    Args:
        points: torch.Tensor of shape (N, 3) - points to rotate
        axis_normal: torch.Tensor of shape (3,) - normalized axis direction vector
        axis_point: torch.Tensor of shape (3,) - point on the axis (axis passes through this point)
        angle_degrees: float - rotation angle in degrees
    
    Returns:
        torch.Tensor of shape (N, 3) - rotated points
    """
    # Convert angle to radians
    angle_rad = torch.deg2rad(torch.tensor(angle_degrees, device=points.device, dtype=points.dtype))
    
    # Ensure axis_normal is normalized
    axis_normal = axis_normal / torch.norm(axis_normal)
    
    # Translate points so that axis passes through origin
    points_translated = points - axis_point.unsqueeze(0)
    
    # Rodrigues' rotation formula components
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)
    
    # Cross product matrix for axis_normal
    K = torch.tensor([
        [0, -axis_normal[2], axis_normal[1]],
        [axis_normal[2], 0, -axis_normal[0]],
        [-axis_normal[1], axis_normal[0], 0]
    ], device=points.device, dtype=points.dtype)
    
    # Rodrigues' rotation matrix: R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, device=points.device, dtype=points.dtype)
    R = I + sin_theta * K + (1 - cos_theta) * torch.matmul(K, K)
    
    # Apply rotation
    points_rotated = torch.matmul(points_translated, R.T)
    
    # Translate back
    points_final = points_rotated + axis_point.unsqueeze(0)
    
    return points_final

def feature_distance_eval_axis(dataset_folder, output_folder, objects_file_path=None, gpu_device="cuda:0", ds_small=True, ds_reg=True, rot_aug=3, view_samp=1, fm_samp=10000, pc_samp=None, viewpoints=None, random_pairing=False):
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
    model = DINOWrapper(device, small=ds_small, reg=ds_reg)
    #########################################
    # ground truths
    gt_files_list = []
    for file in files_path_list:
        gt_files_list.append(file.replace(".obj", ".txt"))
    #########################################
    if viewpoints is None:
        viewpoints = [6, 14, 26, 42, 62, 86, 114]
    distances_dict = {}
    for view in viewpoints:
        distances_dict[view] = []
    if pc_samp:
        fm_samp = None
    #########################################
    for idx, obj_file_path in enumerate(files_path_list):
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
        ######################################### features
        features_dict = {}
        for view in viewpoints:
            if pc_samp:
                features = compute_features(object_points, model, device, view_samp=view_samp, view_quantity=view, rot_aug=rot_aug)
            else:
                features = compute_features(original_mesh, model, device, view_samp=view_samp, view_quantity=view, rot_aug=rot_aug)
            features_dict[view] = features
        ######################################### opens ground truth
        axes = open_axis_txt(gt_files_list[idx])
        normal, mid_point = axes[0] # only 1 anyways
        normal = torch.tensor(normal, device=device)  # (3,)
        mid_point = torch.tensor(mid_point, device=device)  # (3,)
        ######################################### pairing but only is using raw-mesh
        object_angle_distances = {view: [] for view in viewpoints}
        for angle in [45, 90, 135, 180]:
            # iterates over the plans of the objects and computes the mean to penalize bad axes
            if fm_samp is None:
                ######################################### take the mesh/points and rotates them given a nomarl and mid_point
                object_points_rotated = rotate_points_around_axis(object_points, normal, mid_point, angle)
                ######################################### encuentra pares de puntos más cercanos
                if not random_pairing:
                    closest_point_by_index = get_top_L_similar_indices_two_sets(object_points, object_points_rotated, 1)
                else:
                    closest_point_by_index = get_random_L_indices_two_sets(object_points, object_points_rotated, 1)
            ######################################### pair features and calculate distance
            for view in viewpoints:
                ######################################### pairing but only is using feature-mesh sampling
                if fm_samp:
                    #object_points, features = sampled_points_dict[view]
                    object_points, features = sampling_fun(original_mesh, features_dict[view], device, num_points=fm_samp)
                    # take the mesh/points and rotates them given a nomarl and mid_point
                    object_points_rotated = rotate_points_around_axis(object_points, normal, mid_point, angle)
                    # encuentra pares de puntos más cercanos
                    if not random_pairing:
                        closest_point_by_index = get_top_L_similar_indices_two_sets(object_points, object_points_rotated, 1)
                    else:
                        closest_point_by_index = get_random_L_indices_two_sets(object_points, object_points_rotated, 1)
                    # emparejar features
                    features_pair = torch.cat([features.unsqueeze(1), features[closest_point_by_index[:, :1]]], dim=1)
                else:
                    features_pair = torch.cat([features_dict[view].unsqueeze(1), features_dict[view][closest_point_by_index[:, :1]]], dim=1)
                # calcula la diustancia L1 (manhattan) entre los features
                distance = torch.mean(torch.sum(torch.abs(features_pair[:, 0] - features_pair[:, 1]), dim=1))
                object_angle_distances[view].append(distance.item())

        for view in viewpoints:
            distances_dict[view].append(torch.mean(torch.tensor(object_angle_distances[view])))
    #########################################
    # calcula el promedio de las distancias
    distances_means_dict = {}
    for view in viewpoints:
        distances_means_dict[view] = torch.mean(torch.tensor(distances_dict[view]))
    print("Average distances:", distances_means_dict)
    #########################################
    # guarda los resultados en un archivo, separado por tabulaciones
    # 6 14 26 42 62 86 114
    # file is like ds_small=True, ds_reg=True, rot_aug=3, view_samp=1, fm_samp=10000
    file_name = f"feature_distance_axis_{ds_small}_{ds_reg}_{rot_aug}_{view_samp}_{fm_samp}_{pc_samp}"
    if viewpoints != [6, 14, 26, 42, 62, 86, 114]:
        file_name += "_views_" + "_".join([str(view) for view in viewpoints])
    if random_pairing:
        file_name += "_RANDOM"
    result_path = os.path.join(output_folder, file_name + ".txt")
    with open(result_path, "w") as f:
        header_string = ""
        for view in viewpoints:
            header_string += f"{view}\t"
        f.write(header_string + "\n")
        row_string = ""
        for view in viewpoints:
            row_string +=f"{distances_means_dict[view]}\t"
        f.write(row_string + "\n")
    print("Results saved in:", result_path)
    print("Done!")
