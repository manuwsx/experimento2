import os
import math

def get_given_dataset_files_list(dataset_folder) -> list[str]:
    file_list = []
    for file in os.listdir(dataset_folder):
        if file.endswith(".obj"):
            file_list.append(os.path.join(dataset_folder, file))
    return file_list

def get_dataset_objects_given_file(dataset_folder, objects_file_path) -> list[str]:
    final_list = []
    with open(objects_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            file = line.strip()
            final_list.append(os.path.join(dataset_folder, file + ".obj"))
    return final_list

def open_plane_txt(file_path):
    # Check if corresponding _res.txt file exists
    if os.path.exists(file_path):
        planes = []
        try:
            with open(file_path, 'r') as f:
                # Skip the first line (number of planes)
                f.readline()
                # Read plane data
                for line in f:
                    # Split line and convert to floats (skip 'plane' string)
                    parts = line.strip().split()
                    if len(parts) == 7 and parts[0] == 'plane':
                        normal = list(map(float, parts[1:4]))
                        midpoint = list(map(float, parts[4:7]))
                        if any(math.isnan(v) for v in normal + midpoint):
                            continue
                        planes.append([normal, midpoint])
            return planes
            
        except (IOError, ValueError) as e:
            return []
    return []

def open_axis_txt(file_path):
    # Check if corresponding _res.txt file exists
    if os.path.exists(file_path):
        axes = []
        try:
            with open(file_path, 'r') as f:
                # Skip the first line (number of axes)
                f.readline()
                # Read axis data
                for line in f:
                    # Split line and convert to floats (skip 'axis' string)
                    parts = line.strip().split()
                    if len(parts) == 7 and parts[0] == 'axis':
                        normal = list(map(float, parts[1:4]))
                        midpoint = list(map(float, parts[4:7]))
                        if any(math.isnan(v) for v in normal + midpoint):
                            continue
                        axes.append([normal, midpoint])
            return axes
            
        except (IOError, ValueError) as e:
            return []
    return []

def save_planes_to_txt(file_path, planes):
    normals, midpoints, distances = planes
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(str(len(normals)) + "\n")
        for idx, normal in enumerate(normals):
            string = f"planes {normal[0]} {normal[1]} {normal[2]} {midpoints[idx][0]} {midpoints[idx][1]} {midpoints[idx][2]} {distances[idx]}\n"
            f.write(string)

def save_axes_to_txt(file_path, axes):
    normals, midpoints, distances = axes
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(str(len(normals)) + "\n")
        for idx, normal in enumerate(normals):
            string = f"axis {normal[0]} {normal[1]} {normal[2]} {midpoints[idx][0]} {midpoints[idx][1]} {midpoints[idx][2]} {distances[idx]}\n"
            f.write(string)