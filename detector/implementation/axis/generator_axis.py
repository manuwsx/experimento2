import torch
from typing import Tuple, Callable

def adaptive_clustering_batched(
        n_points: int,
        device: str,
        distance_fn: Callable[[int, torch.Tensor], torch.Tensor],
        medoid_fn: Callable[[torch.Tensor], torch.Tensor],
        alpha: float, beta: float, min_elements: int, max_iterations: int = 10) -> list:
    """
    Perform adaptive clustering using on-the-fly distance computation.
    n_points: number of points to cluster
    distance_fn: function that computes distance between two indices (i, j)
    """
    marked = torch.zeros(n_points, dtype=torch.bool, device=device)
    clusters = []
    
    for _ in range(max_iterations):
        for i in range(n_points):
            if marked[i]:
                continue

            min_dist = float('inf')
            nearest_cluster_idx = None
            
            if clusters:
                cluster_medoids = torch.tensor([m for _, m in clusters], dtype=torch.long, device=device)
                cluster_dists = distance_fn(i, cluster_medoids).squeeze(0)
                min_dist, nearest_cluster_idx = torch.min(cluster_dists, dim=0)

            # Step 11: If distance >= β, create a new cluster (Step 12-13)
            if min_dist >= beta:
                clusters.append((torch.tensor([i], device=device), i))  # Create a new cluster with pi as medoid
                marked[i] = True

            # Step 14-16: If distance <= α, assign pi to nearest cluster and update medoid
            elif min_dist <= alpha:
                indices, medoid = clusters[nearest_cluster_idx]
                new_indices = torch.cat([indices, torch.tensor([i], device=device)])
                clusters[nearest_cluster_idx] = (new_indices, medoid)
                marked[i] = True

        # Step 19: Remove clusters with fewer than min_elements (K in your pseudocode)
        delete_mark = [0] * len(clusters)
        valid_clusters = []
        for idx, data in enumerate(clusters):
            indices, __ = data
            if len(indices) >= min_elements:
                medoid = medoid_fn(indices) # Update medoid
                valid_clusters.append((indices, medoid.item()))
            else:
                delete_mark[idx] = 1
        
        # Printing number of clusters for debugging purposes
        #print(f"Iter: {_} -> Num. clusters: {len(clusters)}, removing {delete_mark.count(1)} clusters")
        
        # Keep valid clusters
        clusters = valid_clusters
    
    return clusters

def find_generator_axis_batched(circles: list, device: str, angle_r: float = 0.015, angle_s: float = 0.03,
                                angle_k: int = 10, axis_r: float = 0.25, axis_s: float = 0.5,
                                axis_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find generator axis using on-the-fly distance computation.
    circles: list of (center, radius, normal) tuples
    """
    centers = torch.stack([c[0] for c in circles]).to(device)
    normals = torch.stack([c[2] for c in circles]).to(device)
    n_points = centers.size(0)

    def angular_distance_fn(i, cluster_medoids):
        # Vectorized computation of angular distance for all medoids
        dot_products = torch.matmul(normals[i].unsqueeze(0), normals[cluster_medoids].T)
        return 1 - torch.abs(torch.clamp(dot_products, -1.0, 1.0))

    def compute_angular_medoid(indices: torch.Tensor) -> torch.Tensor:
        cluster_normals = normals[indices]  # [n, 3]
        n = len(indices)
        device = cluster_normals.device
        sum_dist = torch.zeros(n, device=device)
        
        batch_size = 128  # Start with this, adjust based on your VRAM
        
        for i_start in range(0, n, batch_size):
            i_end = min(i_start + batch_size, n)
            i_batch = torch.arange(i_start, i_end, device=device)
            
            # Get ALL j indices for this i_batch
            j_all = torch.arange(n, device=device)
            
            # Compute full distance matrix chunk for this batch
            dots = torch.mm(cluster_normals[i_batch], cluster_normals.T)  # [batch_size, n]
            dist = 1 - torch.abs(torch.clamp(dots, -1.0, 1.0))  # [batch_size, n]
            
            # Create upper triangle mask for this batch
            mask = j_all.unsqueeze(0) > i_batch.unsqueeze(1)  # [batch_size, n]
            
            # Zero out lower triangle (including diagonal)
            masked_dist = dist * mask.float()  # [batch_size, n]
            
            # Symmetric accumulation: add to both i and j
            sum_dist[i_batch] += masked_dist.sum(dim=1)  # Add row sums
            sum_dist += masked_dist.sum(dim=0)  # Add column sums
        
        min_idx = torch.argmin(sum_dist)
        return indices[min_idx]

    orientation_clusters = adaptive_clustering_batched(
        n_points=n_points,
        device=device,
        distance_fn=angular_distance_fn,
        medoid_fn=compute_angular_medoid,
        alpha=angle_r,
        beta=angle_s,
        min_elements=angle_k
    )
    
    if not orientation_clusters:
        return [], [], []

    # Get largest orientation cluster
    largest_cluster = max(orientation_clusters, key=lambda x: len(x[0]))
    cluster_indices = largest_cluster[0]
    cluster_centers = centers[cluster_indices]
    cluster_normals = normals[cluster_indices]
    n_axial_points = cluster_centers.size(0)

    def axial_distance_fn(i, cluster_medoids):
        # Vectorized computation of axial distance for all medoids
        center_diff = cluster_centers[i] - cluster_centers[cluster_medoids]  # (M, D)
        cross_i = torch.cross(cluster_normals[i].unsqueeze(0), center_diff, dim=-1)  # (M, D)
        cross_j = torch.cross(cluster_normals[cluster_medoids], -center_diff, dim=-1)  # (M, D)
        return torch.norm(cross_i, dim=-1) + torch.norm(cross_j, dim=-1)

    def compute_axial_medoid(indices: torch.Tensor) -> torch.Tensor:
        cluster_centers_ = cluster_centers[indices]  # [n, 3]
        cluster_normals_ = cluster_normals[indices]  # [n, 3]
        n = len(indices)
        device = cluster_centers_.device
        sum_dist = torch.zeros(n, device=device)
        
        batch_size = 128  # Adjust based on VRAM (axial is more memory-intensive)
        
        # Process in batches of i indices
        for i_start in range(0, n, batch_size):
            i_end = min(i_start + batch_size, n)
            i_batch = torch.arange(i_start, i_end, device=device)
            centers_i = cluster_centers_[i_batch]  # [batch_size, 3]
            normals_i = cluster_normals_[i_batch]  # [batch_size, 3]
            
            # Compute all j > i for this batch
            j_batch = torch.arange(n, device=device)
            j_mask = j_batch.unsqueeze(0) > i_batch.unsqueeze(1)  # [batch_size, n]
            
            # Compute center differences for all j
            center_diff = cluster_centers_ - centers_i.unsqueeze(1)  # [batch_size, n, 3]
            
            # Cross products for i's normals and differences
            cross_i = torch.cross(
                normals_i.unsqueeze(1).expand(-1, n, -1),  # [batch_size, n, 3]
                -center_diff,
                dim=2
            )
            norm_cross_i = torch.norm(cross_i, dim=2)  # [batch_size, n]
            
            # Cross products for j's normals and differences
            cross_j = torch.cross(
                cluster_normals_.unsqueeze(0).expand(len(i_batch), -1, -1),  # [batch_size, n, 3]
                center_diff,
                dim=2
            )
            norm_cross_j = torch.norm(cross_j, dim=2)  # [batch_size, n]
            
            # Combine distances and apply mask
            total_dist = norm_cross_i + norm_cross_j  # [batch_size, n]
            masked_dist = total_dist * j_mask.float()
            
            # Accumulate sums for i and j
            sum_dist[i_batch] += masked_dist.sum(dim=1)
            sum_dist += masked_dist.sum(dim=0)
        
        min_idx = torch.argmin(sum_dist)
        return indices[min_idx]

    axis_clusters = adaptive_clustering_batched(
        n_points=n_axial_points,
        device=device,
        distance_fn=axial_distance_fn,
        medoid_fn=compute_axial_medoid,
        alpha=axis_r,
        beta=axis_s,
        min_elements=axis_k
    )

    if not axis_clusters:
        return [], [], []

    sorted_clusters = sorted(axis_clusters, key=lambda x: len(x[0]), reverse=True)
    final_normals = []
    final_points = []
    some_val = []

    for cluster in sorted_clusters:
        subset_indices = cluster[0]
        original_indices = cluster_indices[subset_indices.cpu()].to(device)
        axis_normal = torch.mean(normals[original_indices], dim=0)
        axis_normal = axis_normal / torch.norm(axis_normal)
        axis_point = torch.mean(centers[original_indices], dim=0)
        final_normals.append(axis_normal)
        final_points.append(axis_point)
        some_val.append(len(original_indices))
    
    sum_val = sum(some_val)
    if sum_val == 0:
        return [], [], []
    some_val = [v / sum_val for v in some_val]

    return final_normals, final_points, some_val