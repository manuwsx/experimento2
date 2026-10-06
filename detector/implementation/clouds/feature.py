import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF
from torch_geometric.nn import knn

from .other import interpolate_and_aggregate_in_batches
from .render import render_and_map_pytorch3d

class FeatureExtractor(torch.nn.Module):
    def __init__(self,
                 model,
                 rotations, translations,
                 device,
                 use_colorized_renders=True, point_size=0.01,
                 canvas_width=224, canvas_height=224,
                 subsample_point_cloud=None,
                 early_subsample=True,
                 use_manual_projection=False,
                 improved_depth_maps=False,
                 perspective=True,
                 layer_features=None
                 ):
        super().__init__()

        if use_colorized_renders and use_manual_projection:
            raise Exception('Use manual projection is available only when not using colorized renders')

        self.model = model
        self.layer_features = layer_features
        self.rotations = rotations
        self.translations = translations
        self.device = device
        self.use_colorized_renders = use_colorized_renders
        self.improved_depth_maps = improved_depth_maps
        self.use_manual_projection = use_manual_projection
        self.point_size = point_size
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.subsample_point_cloud = subsample_point_cloud
        self.early_subsample = early_subsample
        self.perspective = perspective

    def forward(self, point_cloud, rot_aug=0, batch_size=8, return_outputs=False):
        device = self.device

        if self.early_subsample:
            point_cloud = self.subsample(point_cloud)

        rendered_images, depth_maps, mappings = render_and_map_pytorch3d(
            point_cloud, self.rotations, self.translations,
            canvas_width=self.canvas_width, canvas_height=self.canvas_height,
            point_size= self.point_size,
            points_per_pixel= 10,
            perspective= self.perspective,
            device= device,
            rot_aug=rot_aug,
        )

        if not self.early_subsample:
            point_cloud = self.subsample(point_cloud, mappings, False)#self.early_subsample)

            # Double mappings to get more precise mappings with a small point size
            _, _, mappings = render_and_map_pytorch3d(
                point_cloud, self.rotations, self.translations,
                canvas_width=self.canvas_width, canvas_height=self.canvas_height,
                point_size= self.point_size,
                points_per_pixel= 10,
                perspective= self.perspective,
                device= device,
                rot_aug=rot_aug,
            )

        if self.use_colorized_renders:
            inputs_images = rendered_images
        else:
            if self.improved_depth_maps:
                distances = depth_maps - depth_maps[depth_maps != -1].min()
                distances /= distances[depth_maps != -1].max()
                distances[depth_maps == -1] = 1.
                distances = TF.gaussian_blur(distances, 7)
                distances = TF.adjust_gamma(distances.unsqueeze(1), 1.5).squeeze(1)
                inputs_images = (torch.stack([distances] * 3).permute(1, 2, 3, 0) * 255).int()
            else:
                inputs = (depth_maps - depth_maps.view(len(self.rotations), -1).min(dim=-1).values.unsqueeze(-1).unsqueeze(-1))
                inputs /= inputs.view(len(self.rotations), -1).max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
                inputs_images = 255 - (torch.stack([inputs] * 3).permute(1, 2, 3, 0) * 255).int()
                del inputs

        inputs_images = inputs_images.float()
        with torch.no_grad():
            outputs = self.model(inputs_images)

        features_aggregated = interpolate_and_aggregate_in_batches(
            outputs, mappings, point_cloud, self.canvas_width, self.canvas_height, batch_size, device)
    
        if not return_outputs:
            del outputs
        return features_aggregated

    def subsample(self, point_cloud, mappings=None, early=True):
        np.random.seed(0)
        if self.subsample_point_cloud is not None and len(point_cloud) > self.subsample_point_cloud:
            subsampled_point_indices = torch.from_numpy(
                np.random.choice(np.arange(len(point_cloud)), self.subsample_point_cloud, replace=False))

            point_cloud_reduced = point_cloud[subsampled_point_indices]

            # Map each point to a superpoint
            if not early:
                point_to_superpoint_mapping = knn(
                    point_cloud_reduced[:, :3], point_cloud[:, :3], 1
                )[1].int()

                mappings[mappings != -1] = point_to_superpoint_mapping[mappings[mappings != -1]]
                del point_to_superpoint_mapping

            # Perform subsampling
            point_cloud = point_cloud_reduced
        return point_cloud
