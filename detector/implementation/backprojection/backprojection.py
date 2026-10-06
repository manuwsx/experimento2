import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, MeshRendererWithFragments
from .other import check_visible_vertices_optimized
# classes
from implementation.utils import DINOWrapper

import torch
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer, 
    HardFlatShader,
    look_at_view_transform
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import  DirectionalLights

from implementation.rotation import rotate_and_interleave_images, rotate_and_interleave_coordinates

# rot_aug=(0,1,2,3,4)  ---> (0) VS (0, 180) VS (0, 90, 270) VS (0, 90, 180, 270) VS (0, 0, 0, 0)
rotation_maps = {
    0: [0],                    # No rotation (0 degrees only)
    1: [0, 2],                 # 0 and 180 degrees
    2: [0, 1, 3],              # 0, 90, 270 degrees
    3: [0, 1, 2, 3],           # All four rotations
    4: [0, 0, 0, 0],           # Four copies of the original
}

# rot_aug=(0,1,2,3,4)  ---> (0) VS (0, 180) VS (0, 90, 270) VS (0, 90, 180, 270) VS (0, 0, 0, 0)
def features_backprojection(model: DINOWrapper, mesh: Meshes, views, batch_size, rot_aug=0, res=224, fov=60, device="cpu") -> torch.Tensor:
    """
    Compute the features extracted by 'model' from rendered images with rotation augmentation.

    Args:
        model: 2D ViT pipeline (including pre- and post-processing) in: (N, H, W, C), out: (N, num_patches, emb_dim)
        mesh: pytorch3d Mesh
        views: viewpoints (e.g. from 'views_around_object')
        batch_size: batch size used in processing
        rot_aug: rotation augmentation level (0-4)
                 0: no rotation (0 degrees only)
                 1: 0 and 180 degrees
                 2: 0, 90, 270 degrees
                 3: all four rotations (0, 90, 180, 270 degrees)
                 4: four copies of the original (no rotation)
        res: image resolution
        fov: field of view
        device: computation device
    
    Returns:
        torch.Tensor point features (N, emb_dim)
    """
    mesh = mesh.to(device)
    # Get rotation indices based on augmentation level
    rotation_indices = rotation_maps.get(rot_aug, [0])
    
    R, T = look_at_view_transform(eye=views, device=device)
    
    # Configure the rasterizer and renderer
    raster_settings = RasterizationSettings(
        image_size=(res, res),
        faces_per_pixel=5,
        bin_size=0,
        cull_backfaces=True
    )
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=device,
        )
    )

    # (N, 3) - Get mesh vertices
    points = mesh.verts_packed()
    overall_visibility = torch.zeros(len(points))
    point_values_counts = torch.zeros(len(points), device=device)
    ret_array = None  # Initialize when we know the embedding size

    while len(views) > 0:
        batch_views = views[:batch_size]
        batch_R = R[:batch_size]
        batch_T = T[:batch_size]

        views = views[batch_size:]
        R = R[batch_size:]
        T = T[batch_size:]

        okay = False

        while not okay:
            try:
                # 1. Render the mesh
                cameras = FoVPerspectiveCameras(R=batch_R, T=batch_T, fov=fov, device=device)
                batch_views = batch_views / np.linalg.norm(batch_views, axis=1)[:, None]
                lights = DirectionalLights(direction=batch_views, device=device)

                with torch.no_grad():
                    # images (V, res_y, res_x, 4), pix_to_face (V, res_y, res_x, 5)
                    images, fragments = renderer(mesh.extend(len(batch_views)), cameras=cameras, lights=lights)
                    # images (V, res_y, res_x, 3)
                    images = images[..., :3]
                
                # (V, N, 3)
                pixel_coords_all_points = cameras.transform_points_screen(points, image_size=(res, res)).cpu()

                # Apply rotation augmentations based on selected indices
                images = rotate_and_interleave_images(images, rotation_indices)
                pixel_coords_all_points = rotate_and_interleave_coordinates(
                    pixel_coords_all_points, (res, res), rotation_indices
                )

                # 2. Visibility check
                # (V, N) boolean mask for visibility of each vertex in each image
                visible_points = check_visible_vertices_optimized(fragments.pix_to_face, mesh)
                
                # Repeat for each rotation
                visible_points = visible_points.repeat_interleave(len(rotation_indices), dim=0)
                overall_visibility += visible_points.cpu().sum(dim=0)

                # 3. Extract features
                with torch.no_grad():
                    # (V*len(rotation_indices), num_patches, emb_dim)
                    processed_images = model(images)

                # (V*len(rotation_indices), N, emb_dim)
                features_per_view = get_feature_for_pixel_location_optimized_2(
                    processed_images, pixel_coords_all_points, image_size=res,
                    patch_size=model.patch_size()
                )

                # 4. Aggregate features
                if ret_array is None:
                    # (N, emb_dim)
                    ret_array = torch.zeros(len(points), features_per_view.shape[-1], device=device)

                ret_array += torch.sum(features_per_view * visible_points[..., None], dim=0)
                point_values_counts += visible_points.sum(dim=0)

                okay = True

            except AssertionError as e:
                print(e)
                batch_T = batch_T * 1.1
                batch_views = batch_views * 1.1

    if torch.any(overall_visibility == 0):
        print(f"WARNING: {torch.sum(overall_visibility == 0)} points are not visible in any view! ")
    # Normalize by visibility count
    ret_array[point_values_counts > 0] /= point_values_counts[point_values_counts > 0][:, None]
    return ret_array

def get_feature_for_pixel_location_optimized_2(feature_map, pixel_locations, image_size=224, patch_size=14):
    """
    Maps image features to 3D vertices based on their projected pixel locations.
    
    Args:
        feature_map: Tensor of shape (V, (image_size/patch_size)^2, emb_dim)
        pixel_locations: Tensor of shape (V, N, 2) or (V, N, 3)
        image_size: Integer representing the size of the square image
        patch_size: Integer representing the size of each patch
    
    Returns:
        Tensor of shape (V, N, emb_dim) containing features mapped to vertices
    """
    V, _, emb_dim = feature_map.shape
    _, N, _ = pixel_locations.shape
    
    # Convert pixel coordinates to patch coordinates
    # Take only x, y coordinates if pixel_locations is 3D
    pixel_coords = pixel_locations[..., :2]  
    
    # Normalize pixel coordinates to the range [0, num_patches-1]
    num_patches = image_size // patch_size
    patch_coords = pixel_coords / patch_size

    # Round to the nearest patch indices
    patch_indices_x = torch.clamp(patch_coords[..., 0].floor(), 0, num_patches-1).long()
    patch_indices_y = torch.clamp(patch_coords[..., 1].floor(), 0, num_patches-1).long()

    # Convert 2D patch coordinates to 1D indices
    patch_indices = patch_indices_y * num_patches + patch_indices_x

    # Create indices for gather operation
    batch_indices = torch.arange(V, device=feature_map.device)[:, None].expand(-1, N)

    # Get features for each vertex using patch indices
    vertex_features = feature_map[batch_indices, patch_indices]
    
    return vertex_features

