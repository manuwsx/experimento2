import numpy as np

def sample_standard_viewpoints(radius, partition):
    """
    Views from top and bottom are given, partition determines level of even subdivision between these points
    """
    points = []

    phi = np.linspace(0, 2 * np.pi, (partition + 1) * 2, endpoint=False)
    theta = np.linspace(0, np.pi, (partition + 1), endpoint=False)

    for i, p in enumerate(phi):
        for t in theta:
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.cos(t)
            z = radius * np.sin(t) * np.sin(p)
            if t == 0:
                continue
            points.append([x, y, z])

    points.append([0, radius, 0])
    points.append([0, -radius, 0])
    rotated_points = rotate_points(points, 0.001, axis=[1, 0, 0])

    return np.array(rotated_points)

# calculates the minimum distance from a camera to a sphere of radius r such that the sphere is always within the camera's view
def get_min_camera_distance(r: float, f: float, aspect_ratio: float = 1) -> float:
    # Convert field of view to radians
    f_rad = np.deg2rad(f)
    # Calculate horizontal and vertical field of view
    fov_h = 2 * np.arctan(np.tan(f_rad/2) * np.sqrt(aspect_ratio**2 / (1 + aspect_ratio**2)))
    fov_v = 2 * np.arctan(np.tan(f_rad/2) * np.sqrt(1 / (1 + aspect_ratio**2)))
    # Use the smaller of the two fields of view
    fov_min = min(fov_h, fov_v)
    # Calculate the minimum distance
    d = r / np.tan(fov_min / 2)
    return d

def sample_fibonacci_viewpoints(radius, num_points):
    """
    Generates evenly distributed points on a sphere using the Fibonacci spiral method.
    
    Args:
        radius: The radius of the sphere
        num_points: The number of points to generate
    
    Returns:
        np.array: An array of points on the sphere surface
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        
        points.append([x * radius, y * radius, z * radius])
    
    rotated_points = rotate_points(points, 0.001, axis=[1, 0, 0])

    return np.array(rotated_points)

def rotate_points(points, angle, axis=None):
    """
    Rotate a set of points around an axis by a specified angle.
    """
    # Create rotation matrix around the axis
    if axis is None:
        axis = np.array([0, 1, 0])
    else:
        axis = np.array(axis)

    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    # Rotation matrix using Rodrigues' rotation formula
    rotation_matrix = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C    ]
    ])

    # Apply rotation to each point
    rotated_points = [rotation_matrix @ point for point in points]
    return rotated_points
