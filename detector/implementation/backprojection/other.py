import torch
from collections import defaultdict

def find_adjacent_faces(faces):
    """
    Find adjacent faces of faces in a mesh.

    :param faces: Faces of mesh (m x 3)
    :return: torch tensor of size (m x 3) with indices of adjacent faces, -1 if not 3 adjacent faces
    """
    device = faces.device
    faces = faces.tolist()
    # Mapping from edges to faces
    edge_to_faces = defaultdict(list)

    # Populate the edge_to_faces map
    for i, face in enumerate(faces):
        for j in range(3):
            edge = tuple(sorted((face[j], face[(j + 1) % 3])))
            edge_to_faces[edge].append(i)

    # Find adjacent faces
    adjacent_faces = []
    max_adjacent_faces = 0
    for i, face in enumerate(faces):
        neighbors = set()
        for j in range(3):
            edge = tuple(sorted((face[j], face[(j + 1) % 3])))
            for neighbor_face in edge_to_faces[edge]:
                if neighbor_face != i:
                    neighbors.add(neighbor_face)
        adjacent_faces.append(list(neighbors))
        max_adjacent_faces = max(max_adjacent_faces, len(neighbors))

    ret = torch.zeros(len(faces), max_adjacent_faces, dtype=torch.long, device=device) - 1
    for i, neighbors in enumerate(adjacent_faces):
        ret[i, :len(neighbors)] = torch.tensor(neighbors, dtype=torch.long, device=device)
    return ret

########################################################################################################################
def check_visible_vertices_optimized(pix_to_face, mesh, adjacent_faces=True):
    """
    Find all visible vertices in the rendered images using the pix_to_face tensor returned in the fragments when running
    MeshRendererWithFragments.

    :param pix_to_face: pix_to_face tensor returned in the fragments when running MeshRendererWithFragments
    :param mesh: pytorch3d.structures.Meshes object
    :param adjacent_faces: If True, also mark adjacent faces as visible (helps in dealing with small triangles)
    :return: Boolean array of shape (V, num_vertices) where V is the number of rendered views
    """

    num_views = pix_to_face.shape[0]
    faces_packed = mesh.faces_packed()
    num_faces = len(faces_packed)
    verts_packed = mesh.verts_packed()
    num_verts = len(verts_packed)
    
    # Reshape pix_to_face and handle negative values
    visible_faces_per_view = pix_to_face.view(num_views, -1) % num_faces
    valid_mask = visible_faces_per_view >= 0
    visible_faces_per_view = visible_faces_per_view * valid_mask

    if adjacent_faces:
        adjacent_faces_ = find_adjacent_faces(faces_packed)
        visible_faces_per_view = torch.cat([visible_faces_per_view, adjacent_faces_[visible_faces_per_view].view(num_views, -1)], dim=1)
    
    visible_vertices = faces_packed[visible_faces_per_view.type(torch.long)].view(num_views, -1)
    
    # Create the output tensor
    visible_vertices_per_view = torch.zeros(num_views, num_verts, dtype=torch.bool, device=mesh.device)
    
    # Use scatter to mark visible vertices in a vectorized manner
    visible_vertices_per_view.scatter_(1, visible_vertices, True)
    
    return visible_vertices_per_view