import torch
# features utils
from pytorch3d.ops import sample_points_from_meshes
from implementation.utils import get_min_camera_distance, sample_fibonacci_viewpoints, sample_standard_viewpoints
from implementation.backprojection import features_backprojection
from implementation.utils import DINOWrapper
#types
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform
#cops
from implementation.clouds import FeatureExtractor

# (6, 14, 26, 42, 62, 86, 114) VS (1, 2, 3, 4, 5, 6, 7)
normal_q_to_standard = {6: 1, 14: 2, 26: 3, 42: 4, 62: 5, 86: 6, 114: 7}
num_to_rot_aug = {0: "(0)",1: "(0, 180)",2: "(0, 90, 270)",3: "(0, 90, 180, 270)",4: "(0, 0, 0, 0)",}

def compute_features(mesh_or_cloud: Meshes|torch.Tensor, model: DINOWrapper, device: str, view_samp=1, view_quantity=6, rot_aug=3) -> torch.Tensor:
    """
    Compute the features extracted by 'model' from rendered images with rotation augmentation.
    ### view_samp=(0,1) ---> (standard,fibonacci)
    ### rot_aug=(0,1,2,3,4)  ---> (0) VS (0, 180) VS (0, 90, 270) VS (0, 90, 180, 270) VS (0, 0, 0, 0)
    """
    mesh_or_cloud  = mesh_or_cloud.to(device)
    model = model.to(device)
    mesh:Meshes = None
    point_cloud:torch.Tensor = None
    if isinstance(mesh_or_cloud, Meshes):
        is_point_cloud = False
        mesh = mesh_or_cloud
    elif isinstance(mesh_or_cloud, torch.Tensor):
        is_point_cloud = True
        point_cloud = mesh_or_cloud
    else:
        raise ValueError("mesh_or_cloud must be a Meshes or a point cloud")
    #############################################################
    batch_size = 16
    res = 224
    fov = 60
    #############################################################
    # viewpoint sampling
    if is_point_cloud:
        dist_to_view = point_cloud.norm(dim=-1).max().item()
    else:
        dist_to_view = sample_points_from_meshes(mesh, 1000).norm(dim=-1).max().item()
    render_dist = get_min_camera_distance(dist_to_view, fov) * 1.2 # 1.1 para tener un margen
    if view_samp == 0:
        views = sample_standard_viewpoints(render_dist, normal_q_to_standard[view_quantity])
    elif view_samp == 1:
        views = sample_fibonacci_viewpoints(render_dist, view_quantity)
    else:
        raise ValueError("view_samp must be 0 or 1")
    #############################################################
    # compute features
    print_string = f"\tComputing features; view_sampling={"Fibonacci" if view_samp else "Standard"}, views={view_quantity} and rot_aug="
    print_string += num_to_rot_aug[rot_aug]
    print_string += f" (model={model.model_type} batch_size={batch_size})"
    if is_point_cloud:
        print_string += " (as point cloud)"
    else:
        print_string += " (as normal mesh)"
    print(print_string, flush=True)
    if not is_point_cloud:
        features = features_backprojection(model=model, mesh=mesh, views=views, batch_size=batch_size, rot_aug=rot_aug, res=res, fov=fov, device=device)
    else:
        rotations, translations = look_at_view_transform(eye=views, device=device)
        feature_extractor = FeatureExtractor(model, rotations, translations, device, canvas_height=res, canvas_width=res)
        features = feature_extractor(point_cloud, rot_aug=rot_aug, batch_size=batch_size)
    return features
