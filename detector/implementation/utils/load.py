import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import load_objs_as_meshes

def load_mesh(mesh_path: str, device: str, texture: bool = True) -> Meshes:
    mesh = load_objs_as_meshes([mesh_path], load_textures=False, device=device)
    if texture:
        mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_packed()[None]) * 0.7)
    return mesh