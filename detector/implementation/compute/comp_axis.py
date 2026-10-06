import torch
from pytorch3d.structures import Meshes
from implementation.axis import find_support_circles_batched, find_generator_axis_batched
from implementation.planes import get_top_L_similar_indices

def compute_axis(mesh_or_cloud: Meshes|torch.Tensor, features: torch.Tensor, device: str, L: int):
    mesh_or_cloud  = mesh_or_cloud.to(device)
    features = features.to(device)
    ##################################################
    if isinstance(mesh_or_cloud, Meshes):
        mesh_points = mesh_or_cloud.verts_packed()
    else:
        mesh_points = mesh_or_cloud
    N = mesh_points.shape[0]
    ##################################################
    # these are only indices for mesh_points
    closest_point_by_index = get_top_L_similar_indices(features, L) # (N, L)
    ##################################################
    # X[c] = [a, b] -> x[c] = [c,a,b]
    c = torch.arange(N, device=device).view(-1, 1)  # Shape: (N, 1)
    closest_point_by_index = torch.cat((c, closest_point_by_index), dim=1)  # Shape: (N, L+1) 
    ################################################## 
    circles = find_support_circles_batched(mesh_points, closest_point_by_index, device, num_iterations=500, threshold=0.0001, batch_size=2048)
    ##################################################
    final_normals, final_points, some_val = find_generator_axis_batched(circles, device, angle_r=0.015, angle_s=0.03, angle_k=10, axis_r=0.25, axis_s=0.5, axis_k=5)
    return final_normals, final_points, some_val