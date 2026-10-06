import torch
import torch.nn
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    NormWeightedCompositor
)
from implementation.rotation import rotate_and_interleave_images, rotate_and_interleave_by_dimensions

# rot_aug=(0,1,2,3,4)  ---> (0) VS (0, 180) VS (0, 90, 270) VS (0, 90, 180, 270) VS (0, 0, 0, 0)
rotation_maps = {
    0: [0],                    # No rotation (0 degrees only)
    1: [0, 2],                 # 0 and 180 degrees
    2: [0, 1, 3],              # 0, 90, 270 degrees
    3: [0, 1, 2, 3],           # All four rotations
    4: [0, 0, 0, 0],           # Four copies of the original
}

class PointsRendererWithFragments(torch.nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        fragments = self.rasterizer(point_clouds, **kwargs)
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )
        images = images.permute(0, 2, 3, 1)
        return images, fragments

def render_and_map_pytorch3d(
        point_cloud,
        rotations, translations,
        canvas_width=600, canvas_height=600,
        point_size=0.007,
        points_per_pixel=10,
        perspective=True,
        device='cpu',
        rot_aug=0,
    ):

    verts = point_cloud
    verts -= point_cloud.mean(dim=0)
    verts /= verts.norm(dim=-1).max()
    rgb = point_cloud / 255

    point_cloud_object = Pointclouds(points=[verts], features=[rgb])
    point_cloud_stacked = point_cloud_object.extend(len(rotations))

    # Prepare the cameras
    if perspective:
        cameras = FoVPerspectiveCameras(device=device, R=rotations, T=translations, znear=0.01)
    else:
        cameras = FoVOrthographicCameras(device=device, R=rotations, T=translations, znear=0.01)

    # Prepare the rasterizer
    raster_settings = PointsRasterizationSettings(
        image_size=(canvas_width, canvas_height),
        radius=point_size,
        points_per_pixel=points_per_pixel,
        bin_size=0
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRendererWithFragments(
        rasterizer=rasterizer,
        compositor=NormWeightedCompositor(background_color=(1., 1., 1.))
    )

    # Get mappings and rendered images
    rendered_images, fragments = renderer(point_cloud_stacked)
    for idx in range(len(rotations)):
        fragments.idx[idx, fragments.idx[idx] != -1] -= (idx * len(verts))

    rotation_indices = rotation_maps.get(rot_aug, [0])
    return rotate_and_interleave_images(rendered_images[..., :3], rotation_indices), rotate_and_interleave_by_dimensions(fragments.zbuf[..., 0], rotation_indices), rotate_and_interleave_by_dimensions(fragments.idx[..., 0], rotation_indices)