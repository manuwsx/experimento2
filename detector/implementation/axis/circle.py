import torch
from typing import Tuple, List

def compute_circle_from_points_batched(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate circle parameters (center, radius, normal) from batched three points
    Input shape: (batch_size, 3, 3) where each batch contains 3 points of 3D coordinates
    Returns: centers (batch_size, 3), radii (batch_size), normals (batch_size, 3)
    """
    # Move computation to GPU if available
    device = points.device
    
    # Extract points
    a = points[:, 0]  # (batch_size, 3)
    b = points[:, 1]  # (batch_size, 3)
    c = points[:, 2]  # (batch_size, 3)
    
    # Compute vectors
    v1 = b - a  # (batch_size, 3)
    v2 = c - a  # (batch_size, 3)
    
    # Compute dot products
    v11 = torch.sum(v1 * v1, dim=1)  # (batch_size,)
    v12 = torch.sum(v1 * v2, dim=1)  # (batch_size,)
    v22 = torch.sum(v2 * v2, dim=1)  # (batch_size,)
    
    # Compute lambdas
    common = 2 * (v11 * v22 - v12**2)
    lambda_1 = v22 * (v11 - v12) / common
    lambda_2 = v11 * (v22 - v12) / common
    
    # Compute centers
    centers = a + lambda_1.unsqueeze(1) * v1 + lambda_2.unsqueeze(1) * v2
    
    # Compute radii
    radii = torch.norm(lambda_1.unsqueeze(1) * v1 + lambda_2.unsqueeze(1) * v2, dim=1)
    
    # Compute normals
    normals = torch.cross(v1, v2, dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    
    return centers, radii, normals

def distance_to_circle_batched(points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """
    Calculate distances from points to circles in batched form
    points: (N, 3) or (batch_size, N, 3)
    centers: (batch_size, 3)
    radii: (batch_size,)
    normals: (batch_size, 3)
    Returns: distances (batch_size, N)
    """
    if points.dim() == 2:
        points = points.unsqueeze(0).expand(centers.size(0), -1, -1)
    
    # Compute displacement vectors
    d = points - centers.unsqueeze(1)  # (batch_size, N, 3)
    
    # Compute normal projections
    nd = torch.sum(normals.unsqueeze(1) * d, dim=2)  # (batch_size, N)
    
    # Compute cross products and their norms
    cross_norms = torch.norm(
        torch.cross(normals.unsqueeze(1).expand(-1, points.size(1), -1), d, dim=2),
        dim=2
    )  # (batch_size, N)
    
    # Compute distances
    pq2 = nd ** 2
    kq2 = (cross_norms - radii.unsqueeze(1)) ** 2
    
    return torch.sqrt(pq2 + kq2)

def find_support_circles_batched(
    point_cloud: torch.Tensor,
    grouped_indexes: torch.Tensor,
    device: str,
    num_iterations: int = 500,
    threshold: float = 0.005,
    batch_size: int = 32
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    RANSAC to find support circles for each group using batched operations
    """
    point_cloud = point_cloud.to(device)
    grouped_indexes = grouped_indexes.to(device)
    
    N, L = grouped_indexes.shape
    circles = []
    
    for i in range(N):
        group_idx = grouped_indexes[i]
        group = point_cloud[group_idx]
        #print(i, group_idx)
        
        best_votes = 0
        best_circle = None
        
        # Process RANSAC iterations in batches
        for batch_start in range(0, num_iterations, batch_size):
            batch_end = min(batch_start + batch_size, num_iterations)
            current_batch_size = batch_end - batch_start
            
            # Sample points for the entire batch at once
            rand_idx = torch.randint(0, L, (current_batch_size, 3), device=device)
            batch_points = group[rand_idx]  # (batch_size, 3, 3)
            
            # Compute circles for the entire batch
            centers, radii, normals = compute_circle_from_points_batched(batch_points)
            
            # Compute distances for all points to all circles in the batch
            distances = distance_to_circle_batched(group, centers, radii, normals)
            # Count votes for each circle in the batch
            votes = torch.sum(distances < threshold, dim=1)
            # Update best circle if necessary
            max_votes, max_idx = torch.max(votes, dim=0)
            if max_votes > best_votes:
                best_votes = max_votes
                best_circle = (
                    centers[max_idx].cpu(),
                    radii[max_idx].cpu(),
                    normals[max_idx].cpu()
                )
        
        if best_circle is not None:
            circles.append(best_circle)
    
    return circles