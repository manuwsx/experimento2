import torch
from pytorch3d.loss import chamfer_distance
######################################################################################################################
## get top L similar indices
def get_top_L_similar_indices(X, L, batch_size=1000):
    """
    Find the indices of the top L closest feature vectors for each vector in X using Manhattan distance.
    Uses double batching to handle large datasets efficiently and avoid memory issues.
    
    Args:
        X: Input tensor of shape (N, D) where N is number of vectors and D is dimensions
        L: Number of top similar vectors to return
        batch_size: Size of batches for processing
        
    Returns:
        Tensor of shape (N, L) containing indices of the L most similar vectors for each vector
    """
    N = X.shape[0]
    R = torch.full((N, L), -1, dtype=torch.long, device=X.device)  # Initialize result tensor with -1
    
    # Process input in batches
    for i in range(0, N, batch_size):
        batch_query = X[i:i+batch_size]
        batch_size_actual = batch_query.size(0)
        
        # To store top L distances and indices for the current batch
        top_L_distances = torch.full((batch_size_actual, L), float('inf'), device=X.device)
        top_L_indices = torch.full((batch_size_actual, L), -1, dtype=torch.long, device=X.device)
        
        # Compare against X in batches too
        for j in range(0, N, batch_size):
            batch_ref = X[j:j+batch_size]
            
            # Compute distances between current batch pairs
            distances = torch.cdist(batch_query, batch_ref, p=1)  # Manhattan distance
            
            # Mask self-similarities for overlapping batches
            if i <= j < i + batch_size_actual:
                mask_start = max(0, i - j)
                mask_end = min(batch_size_actual, batch_size - (j - i))
                row_indices = torch.arange(mask_start, mask_end, device=X.device)
                distances[row_indices, row_indices] = float('inf')
            
            # Concatenate current distances and indices
            batch_indices = torch.arange(j, j + batch_ref.size(0), device=X.device).repeat(batch_size_actual, 1)
            concatenated_distances = torch.cat([top_L_distances, distances], dim=1)
            concatenated_indices = torch.cat([top_L_indices, batch_indices], dim=1)
            
            # Select the top L smallest distances and corresponding indices
            top_L_distances, indices = torch.topk(concatenated_distances, L, dim=1, largest=False)
            top_L_indices = torch.gather(concatenated_indices, 1, indices)
        
        # Update result tensor
        R[i:i+batch_size_actual] = top_L_indices
    
    return R

######################################################################################################################
## get planes normalized

def get_planes_normalized(points):
    # 1. Calcular el punto medio para cada par de puntos
    midpoints = (points[:, 0, :] + points[:, 1, :]) / 2
    # 2. Calcular el vector normal para cada par de puntos
    normals = points[:, 1, :] - points[:, 0, :]
    # 3. Normalizar los vectores normales
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    # if torch.any(torch.isnan(normals)):
    #     print("normals contain NaN values")
    # cada plano es un par de puntos, normals y midpoints que son ambos (N, 3)
    # (N, 2, 3)
    #planes = torch.cat((normals.unsqueeze(1), midpoints.unsqueeze(1)), dim=1)
    return normals, midpoints

def get_planes_normalized_3(points):
    """
    Calculate the normals and midpoints of planes defined by tuples of 3 points.

    Args:
        points: Tensor of shape (N, 3, D), where N is the number of point sets,
                3 is the number of points defining each plane, and D is the dimensionality of the space.

    Returns:
        normals: Tensor of shape (N, D), normalized normal vectors of the planes.
        midpoints: Tensor of shape (N, D), midpoints (centroids) of the planes.
    """
    # 1. Compute the centroid (midpoint) for each set of 3 points
    midpoints = points.mean(dim=1)  # Shape: (N, D)

    # 2. Compute two vectors within the plane
    vec1 = points[:, 1, :] - points[:, 0, :]  # Shape: (N, D)
    vec2 = points[:, 2, :] - points[:, 0, :]  # Shape: (N, D)

    # 3. Compute the normal vector using the cross product
    normals = torch.cross(vec1, vec2, dim=1)  # Shape: (N, D)

    # 4. Normalize the normal vectors
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    # if torch.any(torch.isnan(normals)):
    #     print("normals contain NaN values")

    return normals, midpoints

######################################################################################################################
## are planes similar
def are_planes_similar_rotation(normal1: torch.Tensor, normals2: torch.Tensor, angle_threshold_degrees: float = 5.0) -> torch.Tensor:
    N = normals2.size(0)
    if N == 0:
        return torch.tensor(False, device=normal1.device)
    # Ensure normal1 is a 2D tensor (1, 3) for matrix multiplication
    normal1 = normal1.unsqueeze(0).expand(N, 3)  # (N, 3)
    # Compute the dot product between the single normal and each normal in normals2
    dot_product = torch.sum(normal1 * normals2, dim=1)  # (N,)
    # Compute the angle between the normal and each normal in normals2
    angle = torch.acos(torch.clamp(torch.abs(dot_product), -1.0, 1.0))  # (N,)
    # Convert angle to degrees
    angle_degrees = torch.rad2deg(angle)  # (N,)
    similar_mask = angle_degrees.squeeze() < angle_threshold_degrees # (N,)
    return similar_mask


######################################################################################################################
## some utils
def get_intersecting_planes(normals: torch.Tensor, midpoints: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Find planes that intersect with a sphere centered at origin with given radius.
    A plane intersects with a sphere if the distance from sphere center to plane
    is less than or equal to the sphere radius.
    
    Args:
        normals: Tensor of shape (N, 3) containing normal vectors for each plane
        midpoints: Tensor of shape (N, 3) containing points that lie on each plane
        radius: Radius of the sphere centered at origin
        
    Returns:
        Tensor of boolean masks indicating which planes intersect with sphere
    """
    # Normalize normal vectors
    #normals = normals / torch.norm(normals, dim=1, keepdim=True)
    
    # Distance from origin (sphere center) to plane is |ax0 + by0 + cz0 + d|/sqrt(a^2 + b^2 + c^2)
    # where (a,b,c) is normal vector and (x0,y0,z0) is point on plane
    # Since normals are normalized, denominator is 1
    # d = -(ax0 + by0 + cz0) where (x0,y0,z0) is midpoint
    d = -torch.sum(normals * midpoints, dim=1)
    
    # Distance from origin to plane is |d| since normals are normalized
    distances = torch.abs(d)
    
    # Plane intersects sphere if distance <= radius
    intersecting_mask = distances <= radius
    
    return intersecting_mask

######################################################################################################################

def get_candidate_planes(mesh_points: torch.Tensor, dist_by_samples: float, normals: torch.Tensor, midpoints: torch.Tensor, batch_size = 5, threshold=0.01, angle_threshold_degrees=5):
    # mesh_points (N, 3) - mesh points
    # normals (N, 3) - plane normals
    # midpoints (N, 3) - plane midpoints
    N, _ = mesh_points.shape
    intersecting_mask = get_intersecting_planes(normals, midpoints, dist_by_samples*0.05)
    normals = normals[intersecting_mask]
    midpoints = midpoints[intersecting_mask]

    M, _ = normals.shape

    result_normals = torch.empty((0,), dtype=torch.float32, device=mesh_points.device)
    result_midpoints = torch.empty((0,), dtype=torch.float32, device=mesh_points.device)
    result_distances = torch.empty((0,), dtype=torch.float32, device=mesh_points.device)
    
    # Process in batches
    for i in range(0, M, batch_size):
        # Get a batch of normals and midpoints
        batch_normals = normals[i:i + batch_size]  # (batch_size, 3)
        batch_midpoints = midpoints[i:i + batch_size]  # (batch_size, 3)
        batch_size_actual = batch_normals.size(0)
        
        # Expand dimensions for vectorized operations
        # mesh_points: (N, 3)
        # batch_normals, batch_midpoints: (batch_size, 3)
        
        # Expand mesh_points to match batch_size
        expanded_points = mesh_points.unsqueeze(1).expand(N, batch_size_actual, 3)  # (N, batch_size, 3)
        expanded_normals = batch_normals.unsqueeze(0).expand(N, batch_size_actual, 3)  # (N, batch_size, 3)
        expanded_midpoints = batch_midpoints.unsqueeze(0).expand(N, batch_size_actual, 3)  # (N, batch_size, 3)
        
        # Vector from the plane point to the mesh point
        vec_to_midpoint = expanded_points - expanded_midpoints  # (N, batch_size, 3)
        
        # Dot product between (P - M) and the normal N
        dot_product = torch.sum(vec_to_midpoint * expanded_normals, dim=-1, keepdim=True)  # (N, batch_size, 1)
        
        # Reflect the points with respect to the plane
        reflection = expanded_points - 2 * (dot_product / torch.sum(expanded_normals**2, dim=-1, keepdim=True)) * expanded_normals  # (N, batch_size, 3)

        reflection = reflection.permute(1, 0, 2)  # (batch_size, N, 3)
        expanded_points = expanded_points.permute(1, 0, 2)  # (batch_size, N, 3)
        # Compute Chamfer distance

        batch_distance, _ = chamfer_distance(expanded_points, reflection, batch_reduction=None) # (batch_size,)
        # Filter planes that meet the threshold
        mask = batch_distance < threshold  # (batch_size,)
        result_normals = torch.cat((result_normals, batch_normals[mask]), dim=0) # + (num_filtered, 3)
        result_midpoints = torch.cat((result_midpoints, batch_midpoints[mask]), dim=0) # + (num_filtered, 3)
        result_distances = torch.cat((result_distances, batch_distance[mask]), dim=0) # + (num_filtered, 3)

    # Sort planes by distance
    result_distances, indices = torch.sort(result_distances)
    result_normals = result_normals[indices]
    result_midpoints = result_midpoints[indices]

    # Filter similar planes
    final_normals = torch.empty((0, 3), dtype=torch.float32, device=mesh_points.device)
    final_midpoints = torch.empty((0, 3), dtype=torch.float32, device=mesh_points.device)
    final_distances = torch.empty((0,), dtype=torch.float32, device=mesh_points.device)
    count = []
    for i in range(result_normals.size(0)):
        normal = result_normals[i]
        midpoint = result_midpoints[i]
        distance = result_distances[i]
        new_distance = 1 - distance/threshold
        # using torch check if var has true in it
        bool_similar = are_planes_similar_rotation(normal, final_normals, angle_threshold_degrees)
        if not torch.any(bool_similar):
            final_normals = torch.cat((final_normals, normal.unsqueeze(0)), dim=0)
            final_midpoints = torch.cat((final_midpoints, midpoint.unsqueeze(0)), dim=0)
            final_distances = torch.cat((final_distances, new_distance.unsqueeze(0)), dim=0)
            count.append(1)
        else:
            # get index of repeated plane given the True value in boolean mask
            repeated_plane_idx = torch.where(bool_similar)[0][0].item()
            # increment count for repeated plane
            count[repeated_plane_idx] += 1
    num = min(10, len(final_normals))
    return final_normals[:num], final_midpoints[:num], final_distances[:num]
##########################################################################################################################