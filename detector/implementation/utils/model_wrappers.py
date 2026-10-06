"""
Model wrappers for feature extraction from large pre-trained vision models.
"""
from abc import ABC
import numpy as np
import torch
import torchvision.transforms as T

class DINOWrapper(torch.nn.Module, ABC):
    def __init__(self, device='cpu', small=True, reg=True):
        super().__init__()
        print("Initializing DINOWrapper...")
        self._init_regular_dino(device, small, reg)
        self.model.eval()
        # ImageNet normalization
        self.image_transforms = T.Compose([
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _init_regular_dino(self, device, small, reg):
        """Initialize regular DINOv2 from torch hub."""
        self.model_type = "small" if small else "large"
        try:
            if not small:
                if reg:
                    print("Loading large model...")
                    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
                else:
                    print("Loading large model...")
                    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
            else:
                if reg:
                    print("Loading small model...")
                    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device)
                else:
                    print("Loading small model...")
                    self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    def forward(self, images, interpolation=False, raw=False):
        # Convert from [B, H, W, C] to [B, C, H, W] if needed
        if images.dim() == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        
        images = self.image_transforms(images)
        
        # Use forward_features method
        with torch.no_grad():
            out = self.model.forward_features(images)
        
        if raw:
            for key in out:
                print(f"Key: {key}, Shape: {out[key].shape}")
            return out
            
        if interpolation:
            features = out['x_norm_patchtokens']
            N, num_patches, C = features.shape
            features = torch.permute(features, (0, 2, 1))
            features = features.view(N, C, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
            features = torch.nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
            features = features.view(N, C, -1)
            features = torch.permute(features, (0, 2, 1))
            return features
        else:
            return out['x_norm_patchtokens']

    def patch_size(self):
        if hasattr(self, '_patch_size'):
            return self._patch_size
        return 14