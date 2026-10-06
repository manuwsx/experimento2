## sampling utils
import torch
import point_cloud_utils as pcu
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from typing import Tuple

def sampling_fun(mesh: Meshes, features : torch.Tensor, device: str, num_points=10000) -> Tuple[torch.Tensor, torch.Tensor]:
    verts = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy().astype(np.int32)
    features = features.cpu().numpy()

    fid, bary = pcu.sample_mesh_random(verts, faces, num_points)
    fid = fid.astype(np.int32)

    points = pcu.interpolate_barycentric_coords(faces, fid, bary, verts)
    final_features = pcu.interpolate_barycentric_coords(faces, fid, bary, features)
    # send to device
    points = torch.tensor(points, device=device, dtype=torch.float32)
    final_features = torch.tensor(final_features, device=device, dtype=torch.float32)
    return points, final_features