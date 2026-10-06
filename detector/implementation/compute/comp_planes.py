import torch
# features utils
from pytorch3d.ops import sample_points_from_meshes
#types
from pytorch3d.structures import Meshes
from implementation.planes import get_planes_normalized, get_planes_normalized_3, get_top_L_similar_indices
from implementation.planes import get_candidate_planes

def compute_planes(mesh_or_cloud: Meshes|torch.Tensor, features: torch.Tensor, device: str):
    mesh_or_cloud  = mesh_or_cloud.to(device)
    features = features.to(device)
    ##################################################
    if isinstance(mesh_or_cloud, Meshes):
        dist_by_samples = sample_points_from_meshes(mesh_or_cloud, 1000).norm(dim=-1).max().item()
        mesh_points = mesh_or_cloud.verts_packed()
    else:
        num_samples = min(1000, mesh_or_cloud.shape[0])
        indices = torch.randperm(mesh_or_cloud.shape[0])[:num_samples]
        dist_by_samples = mesh_or_cloud[indices].norm(dim=-1).max().item()
        mesh_points = mesh_or_cloud
    ##################################################
    # Calculate indices of the closest features
    # closest_features_idx (N,)
    closest_point_by_index = get_top_L_similar_indices(features, 2)
    # tuples (N, 2, 3) and (N, 3, 3)
    # Pair the closest points in pairs and all together
    tuple_2_points_0 = torch.cat([mesh_points.unsqueeze(1), mesh_points[closest_point_by_index[:, :1]]], dim=1) # 0,1
    tuple_2_points_1 = torch.cat([mesh_points[closest_point_by_index[:, :1]], mesh_points[closest_point_by_index[:, 1:2]]], dim=1) # 1,2
    tuple_2_points_2 = torch.cat([mesh_points.unsqueeze(1), mesh_points[closest_point_by_index[:, 1:2]]], dim=1) # 0,2
    tuple_2_points = torch.cat([tuple_2_points_0, tuple_2_points_1, tuple_2_points_2], dim=0)
    tuple_3_points = torch.cat([mesh_points.unsqueeze(1), mesh_points[closest_point_by_index[:, :2]]], dim=1) # 0,1,2
    ##################################################
    # Calculate a plane between pairs of points, such that the plane is equidistant to both points
    planes_normals_2, planes_midpoints_2 = get_planes_normalized(tuple_2_points)
    # Calculate a plane formed by the 3 points
    planes_normals_3, planes_midpoints_3 = get_planes_normalized_3(tuple_3_points)
    ##################################################
    # Shape: (4N, 3)
    merged_planes_normals = torch.cat([planes_normals_2, planes_normals_3], dim=0) 
    merged_planes_midpoints = torch.cat([planes_midpoints_2, planes_midpoints_3], dim=0)
    ##################################################
    normals, midpoints, distances = get_candidate_planes(mesh_points, dist_by_samples, merged_planes_normals, merged_planes_midpoints, batch_size=5, threshold=0.01, angle_threshold_degrees=5)
    return normals, midpoints, distances