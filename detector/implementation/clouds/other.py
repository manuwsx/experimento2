import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import knn

def backproject(mapping, point_cloud, pixel_features, device='cpu'):
    """
    Back-project features to points in the point cloud using the given mapping.

    :param mapping: tensor with shape (CANVAS_HEIGHT, CANVAS_WIDTH)
    :param point_cloud: tensor, points in the point cloud, with shape (#points, 3)
    :param pixel_features: features extracted with a backbone model. Shape is
        (CANVAS_HEIGHT, CANVAS_WIDTH, feature_dimensionality)
    :param device: device on which to place tensors and perform operations.
        Default to `cpu`

    :return: a new tensor of shape (#points, feature_dimensionality) that associates
        to each point in the point cloud a feature vector. Feature vector is all 0
        if no feature is being associated to a point.
        It can be indexed as `point_cloud`, so the i-th feature vector is associated
        to the i-th point in `point_cloud`.
    """
    # Feature vector: (points, feature_dimension)
    # For example: for 10,000 points and embedding dimension of 768,
    # features will be a (10000, 768) tensor
    features = torch.zeros((len(point_cloud), pixel_features.shape[-1]), dtype=torch.float, device=device)
    # Get pixel coordinates of pixels on which the point cloud has been projected
    yx_coords_of_pcd = (mapping != -1).nonzero()

    # Map features to the points
    # Explanation: `mapping` is a (HEIGHT, WIDTH) map with the same dimensionality
    # of the render. Each entry is either `-1` (no point has been mapped to the corresponding pixel)
    # or the index of the point in `point_cloud` that was mapped/projected to the corresponding
    # pixel.
    # So: `mapping != -1` returns a boolean mask to tell if a pixel is "empty" or if something
    # was projected there. Then, `mapping[mapping != -1]` returns the indices of the points
    # in `point_cloud` that have been mapped to a point. They're returned from top-left to
    # bottom-right order. Since `features` has a "row" for each point in `point_cloud`,
    # it can be indexed via the same indices as `point_cloud`. Therefore, `features[mapping[mapping != -1]]`
    # accesses the features of all the points that have been rendered (are visible) in the rendering.
    # Lastly, the assignment simply assigns features to those points. Features come from an "image"
    # (where number of channels is arbitrary, depending on the backbone model).
    # Note that a point may be mapped to multiple pixels, especially if using a large enough `point_size`.
    # In this case, a point will be assigned just the "last" features: if `pixel_features` has
    # two distinct feature vectors (e.g. [1, 2] and [3, 4]) for the point (x, y), the point (x, y)
    # will be ultimately assigned features [3, 4]. While this may sound like a problem, it is actually
    # not in most practical applications: if a point is projected into multiple pixels, they are certainly
    # neighbouring pixels. Therefore, they most likely have very similar feature vectors: so overwriting
    # the features and just keeping the "last" that comes (usually the most bottom-right in the
    # `pixel_features` "image") is not an actual problem.
    features[mapping[mapping != -1]] = pixel_features[yx_coords_of_pcd[:, 0], yx_coords_of_pcd[:, 1]]

    return features

def interpolate_point_cloud(pcd, features, points_with_missing_features=None, neighbors=10, copy_features=False, zero_nan=True):
    """
    Interpolate features associate to each point for points that are missing them.
    A point misses a feature if all the feature vector associated to it is made up
    of NaN values.
    Features are therefore interpolated using neighboring points.

    Note that if all the closest neighbors of a point are missing features as well,
    then the feature vector will still be composed only of NaN values. The `zero_nan` parameter allows to control
    this behavior: setting it to True (which is the default value) will substitute all NaN values with 0.

    :param pcd: the point cloud tensor, with shape (#points, 3)
    :param features: the features. Each element is a tensor representing the features associated to the
        point with the same index in `pcd`
    :param points_with_missing_features: indices of points which are missing features. If None, it will be
        automatically determined by finding all the feature vectors which have all features (along feature
        dimension) set to 0.0. Default to None
    :param neighbors: how many neighbors to consider in the interpolation
    :param copy_features: whether to copy the feature tensor or modify directly the one being passed
    :param zero_nan: if a feature vector is still NaN after interpolation (because all its neighbors are NaN, too),
        set it to 0 anyway. True: set NaN to 0 after interpolation. Defaults to True
    :return: the interpolated tensor
    """
    if copy_features:
        features = copy.deepcopy(features)

    if points_with_missing_features is None:
        points_with_missing_features = torch.all(features == 0, dim=-1).nonzero().view(-1)

    if len(points_with_missing_features) == 0:
        return features

    # k-NN between all the points
    neighbors += 1
    knn_on_cluster_assignment = knn(pcd, pcd, neighbors)

    # Get the features only for the points that will have to be used to interpolate feature
    # vectors, and then compute the average between all the neighbors for each point
    knn_on_cluster_assignment = knn_on_cluster_assignment[1].view(len(pcd), neighbors)
    neighbors_features = features[knn_on_cluster_assignment[points_with_missing_features].view(-1)].view(
        len(points_with_missing_features), neighbors, -1
    )
    # Setting to NaN makes it possible to compute the average only of those points which actually
    # have features. If this was left to 0, they would influence the mean, while this makes
    # it possible not to take them into account
    neighbors_features[torch.all(neighbors_features == 0, dim=-1)] = float('nan')
    features[points_with_missing_features] = neighbors_features.nanmean(dim=1)

    # Adjust feature vectors for points whose neighbors are all NaN (no feature vector assigned to them)
    if zero_nan:
        features = torch.nan_to_num(features)

    return features


def interpolate_feature_map(features, width, height, mode='bicubic'):
    """
    Interpolate a patchy feature map to the specified size (width, height)

    :param features: tensor of shape (batch_size, #patches + 1 for [CLS], embedding_dimension)
    :param width: width of the output
    :param height: height of the output
    :param mode: interpolation method. Default to 'bicubic'
    :return: interpolated feature map
    """
    # R: renders
    # L: length
    R, L, _ = features.shape
    W = H = np.sqrt(L).astype(int)

    with torch.no_grad():
        interpolated_features = F.interpolate(
            features.view(R, W, H, -1).permute(3, 0, 1, 2),
            size=(width, height),
            mode=mode,
            align_corners=False if mode not in ['nearest', 'area'] else None,
        )
    interpolated_features = interpolated_features.permute(1, 2, 3, 0)

    return interpolated_features

def interpolate_and_aggregate_in_batches(outputs, mappings, point_cloud, canvas_width, canvas_height, batch_size=1, device='cpu'):
    """
    Interpolate and aggregate features in batches to avoid OOM.
    """
    num_views = outputs.shape[0]
    feature_dim = outputs.shape[-1]
    num_points = len(point_cloud)
    
    # Initialize aggregation tensors
    feature_pcd_aggregated = torch.zeros((num_points, feature_dim), device=device, dtype=torch.double)
    count = torch.zeros(num_points, device=device)
    
    for i in range(0, num_views, batch_size):
        batch_outputs = outputs[i:i+batch_size]  # (B, L, D)
        batch_mappings = mappings[i:i+batch_size]  # (B, H, W)
        
        # Interpolate each in the batch individually to avoid blowing up memory
        for j in range(len(batch_outputs)):
            interpolated = interpolate_feature_map(batch_outputs[j:j+1], canvas_width, canvas_height)  # (1, H, W, D)
            interpolated = interpolated.squeeze(0)  # (H, W, D)
            
            feature_pcd = backproject(batch_mappings[j], point_cloud, interpolated, device=device)
            nan_mask = ~torch.all(feature_pcd == 0.0, dim=-1)
            feature_pcd_aggregated[nan_mask] += feature_pcd[nan_mask]
            count[nan_mask] += 1
            
            # Optional: explicit cleanup for maximum memory efficiency
            del interpolated, feature_pcd
        
        # Optional: cleanup batch tensors
        del batch_outputs, batch_mappings
        if device != 'cpu':
            torch.cuda.empty_cache()
    
    count[count == 0] = 1
    feature_pcd_aggregated /= count.unsqueeze(-1)
    feature_pcd_aggregated = interpolate_point_cloud(point_cloud[:, :3], feature_pcd_aggregated, neighbors=20)
    
    return feature_pcd_aggregated

