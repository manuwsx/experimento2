import torch

def rotate_and_interleave_images(images, rotation_indices):
    """
    Selective version: returns interleaved rotations based on the rotation indices.
    
    Args:
        images: Tensor of shape (V, H, W, 3)
        rotation_indices: List of indices (0=0°, 1=90°, 2=180°, 3=270°)
    
    Returns:
        Tensor of shape (len(rotation_indices)*V, H, W, 3)
    """
    V = images.shape[0]
    
    # Create all rotations
    all_rotations = [
        images,  # 0 degrees
        torch.rot90(images, k=1, dims=(1, 2)),  # 90 degrees
        torch.rot90(images, k=2, dims=(1, 2)),  # 180 degrees
        torch.rot90(images, k=3, dims=(1, 2)),  # 270 degrees
    ]
    
    # Select only the requested rotations
    selected_rotations = [all_rotations[idx] for idx in rotation_indices]
    
    # Stack along a new dimension
    rotations = torch.stack(selected_rotations, dim=0)  # (len(rotation_indices), V, H, W, 3)
    
    # Interleave: move batch to first, then reshape
    rotations = rotations.permute(1, 0, 2, 3, 4)  # (V, len(rotation_indices), H, W, 3)
    return rotations.reshape(len(rotation_indices) * V, *images.shape[1:])  # (len(rotation_indices)*V, H, W, 3)

def rotate_and_interleave_coordinates(pixel_coords, image_size, rotation_indices):
    """
    Selective: Rotates pixel coordinates to match specified rotations
    
    Args:
        pixel_coords: Tensor of shape (V, N, 3) containing pixel coordinates
        image_size: Tuple (height, width)
        rotation_indices: List of indices (0=0°, 1=90°, 2=180°, 3=270°)
    
    Returns:
        Tensor of shape (len(rotation_indices)*V, N, 3), interleaved per image
    """
    V, N, _ = pixel_coords.shape
    H, W = image_size
    
    # Calculate all rotations
    coords_0 = pixel_coords
    
    coords_90 = pixel_coords.clone()
    coords_90[:, :, 0] = H - 1 - pixel_coords[:, :, 1]
    coords_90[:, :, 1] = pixel_coords[:, :, 0]
    
    coords_180 = pixel_coords.clone()
    coords_180[:, :, 0] = W - 1 - pixel_coords[:, :, 0]
    coords_180[:, :, 1] = H - 1 - pixel_coords[:, :, 1]
    
    coords_270 = pixel_coords.clone()
    coords_270[:, :, 0] = pixel_coords[:, :, 1]
    coords_270[:, :, 1] = W - 1 - pixel_coords[:, :, 0]
    
    all_rotations = [coords_0, coords_90, coords_180, coords_270]
    
    # Select only the requested rotations
    selected_rotations = [all_rotations[idx] for idx in rotation_indices]
    
    # Stack and interleave
    stacked = torch.stack(selected_rotations, dim=1)  # (V, len(rotation_indices), N, 3)
    return stacked.view(-1, N, 3)  # (len(rotation_indices)*V, N, 3)

def rotate_and_interleave_by_dimensions(tensor, rotation_indices):
    """
    Selective version: rotates and interleaves tensor based on specified rotation indices.
    
    Args:
        tensor: Tensor of shape (V, H, W)
        rotation_indices: List of indices (0=0°, 1=90°, 2=180°, 3=270°)
    
    Returns:
        Tensor: (len(rotation_indices)*V, H, W), interleaved per image
    """
    V = tensor.shape[0]
    
    # Create all possible rotations
    all_rotations = [
        tensor,  # 0 degrees
        torch.rot90(tensor, k=1, dims=(1, 2)),  # 90 degrees
        torch.rot90(tensor, k=2, dims=(1, 2)),  # 180 degrees
        torch.rot90(tensor, k=3, dims=(1, 2)),  # 270 degrees
    ]
    
    # Select only the requested rotations
    selected_rotations = [all_rotations[idx] for idx in rotation_indices]
    
    # Stack along a new dimension
    rotations = torch.stack(selected_rotations, dim=0)  # (len(rotation_indices), V, H, W)
    
    # Interleave: move batch to first, then reshape
    rotations = rotations.permute(1, 0, 2, 3)  # (V, len(rotation_indices), H, W)
    return rotations.reshape(len(rotation_indices) * V, *tensor.shape[1:])  # (len(rotation_indices)*V, H, W)
