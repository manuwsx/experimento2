import numpy as np
import os
import trimesh
import pyvista as pv
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
from sklearn.decomposition import PCA
from implementation.utils import sample_standard_viewpoints, sample_fibonacci_viewpoints
#import meshes
from pytorch3d.structures import Meshes

class InteractiveRenderer:
    def __init__(self):
        self.current_visible = 'mesh'  # 'mesh' or 'point_cloud'
        self.features_visible = True

    def _init_common(self, output_dir, img_width, img_height):
        self.output_dir = output_dir
        self.img_width = img_width
        self.img_height = img_height
        self.camera_positions = []
        self.planes = []
        self.pre_planes = []
        self.planes_actors = []
        self.axis = []
        self.pre_axis = []
        self.axis_actors = []
        os.makedirs(output_dir, exist_ok=True)
        self.plotter = pv.Plotter(window_size=(img_width, img_height))
        self.plotter.background_color = 'white'

    def initialize_withmesh(self, mesh: Meshes, name, output_dir="./renders", img_width=800, img_height=600, features=None):
        self._init_common(output_dir, img_width, img_height)
        self.base_filename = name
        self.vertices = mesh.verts_packed().cpu().numpy()
        self.faces = mesh.faces_packed().cpu().numpy()
        self.face_array = np.column_stack((np.full(len(self.faces), 3), self.faces)) if self.faces.ndim == 2 else self.faces
        self.pv_mesh = pv.PolyData(self.vertices, self.face_array)
        self._add_mesh_to_plotter(features)
        self.current_visible = 'mesh'
        self.features_visible = features is not None

    def initialize(self, obj_path, output_dir="./renders", img_width=800, img_height=600, features=None):
        self._init_common(output_dir, img_width, img_height)
        self.obj_path = obj_path
        self.base_filename = os.path.splitext(os.path.basename(obj_path))[0]
        self.mesh_obj = trimesh.load_mesh(obj_path)
        self.vertices = self.mesh_obj.vertices
        self.faces = self.mesh_obj.faces
        self.mesh_center = np.mean(self.vertices, axis=0)
        self.face_array = np.column_stack((np.full(len(self.faces), 3), self.faces)) if self.faces.ndim == 2 else self.faces
        self.pv_mesh = pv.PolyData(self.vertices, self.face_array)
        self._add_mesh_to_plotter(features)

    def initialize_empty(self, output_dir="./renders", img_width=800, img_height=600):
        self._init_common(output_dir, img_width, img_height)

    def _add_mesh_to_plotter(self, features):
        if features is not None:
            pca = PCA(n_components=3)
            features_rgb = pca.fit_transform(features)
            self.min_vals = features_rgb.min(axis=0)
            self.max_vals = features_rgb.max(axis=0)
            features_rgb_norm = (features_rgb - self.min_vals) / (self.max_vals - self.min_vals)
            self.pv_mesh.point_data['rgb'] = features_rgb_norm
            self.mesh_actor = self.plotter.add_mesh(
                self.pv_mesh,
                scalars='rgb',
                rgb=True,
                lighting=True,
                ambient=0.2,
                specular=0.3,
                specular_power=5,
                pickable=False
            )
            print("Mesh colored with features")
        else:
            self.mesh_actor = self.plotter.add_mesh(
                self.pv_mesh,
                color='gray',
                lighting=True,
                ambient=0.2,
                specular=0.3,
                specular_power=5,
                pickable=False
            )
            print("Mesh added to the scene")
   
    def start_interactive_session(self, look_for_axis=False):
        """Start interactive viewer session with key bindings."""
        # Add key bindings
        self.plotter.add_key_event('s', self.save_current_view)           # Save current camera view
        self.plotter.add_key_event('r', self.reset_view)                  # Reset view
        self.plotter.add_key_event('2', self.hide_features)               # Hide features
        self.plotter.add_key_event('1', self.show_point_cloud)            # Show point cloud

        if look_for_axis:
            self.plotter.add_key_event('3', self.add_current_axis)        # Add axis
        else:
            self.plotter.add_key_event('3', self.add_current_plane)       # Add plane

        print("\n--- Interactive 3D Viewer ---")
        print("Use mouse to rotate/zoom the object")
        print("Press 's' to save current camera view")
        print("Press 'r' to reset view")
        print("Press '1' to show/hide point cloud")
        print("Press '2' to show/hide features")
        if look_for_axis:
            print("Press '3' to add axis")
        else:
            print("Press '3' to add plane")
        print("Press 'q' to quit")
        
        # Show interactive viewer
        self.plotter.show()

    def add_point_cloud(self, point_cloud, point_cloud_features):
        self.point_cloud = point_cloud
        self.point_cloud_features = point_cloud_features
        # If mesh is not visible, show point cloud
        if self.current_visible == 'point_cloud':
            self._show_point_cloud_actor()

    def show_point_cloud(self):
        """Toggle between mesh and point cloud display."""
        if self.current_visible == 'mesh' and hasattr(self, 'point_cloud') and self.point_cloud is not None:
            # Hide mesh, show point cloud
            self._show_point_cloud_actor()
            self.current_visible = 'point_cloud'
        elif self.current_visible == 'point_cloud' and hasattr(self, 'pv_mesh') and self.pv_mesh is not None:
            # Hide point cloud, show mesh
            self._show_mesh_actor()
            self.current_visible = 'mesh'
        else:
            print("No mesh or point cloud to toggle.")

    def _show_point_cloud_actor(self):
        # Remove mesh actor if present
        if hasattr(self, 'mesh_actor') and self.mesh_actor is not None:
            self.plotter.remove_actor(self.mesh_actor)
        point_cloud = pv.PolyData(self.point_cloud)
        if self.features_visible and hasattr(self, 'point_cloud_features') and self.point_cloud_features is not None:
            pca = PCA(n_components=3)
            features_rgb = pca.fit_transform(self.point_cloud_features)
            min_vals = features_rgb.min(axis=0)
            max_vals = features_rgb.max(axis=0)
            features_rgb_norm = (features_rgb - min_vals) / (max_vals - min_vals)
            point_cloud.point_data['rgb'] = features_rgb_norm
            color_mode = {'scalars': 'rgb', 'rgb': True}
        else:
            color_mode = {'color': 'gray'}
        self.mesh_actor = self.plotter.add_points(
            point_cloud,
            point_size=5,
            **color_mode
        )
        print(f"Point cloud {'with features' if self.features_visible else 'in gray'} shown.")

    def _show_mesh_actor(self):
        # Remove mesh actor if present
        if hasattr(self, 'mesh_actor') and self.mesh_actor is not None:
            self.plotter.remove_actor(self.mesh_actor)
        if self.features_visible and hasattr(self, 'pv_mesh') and self.pv_mesh is not None and 'rgb' in self.pv_mesh.point_data:
            self.mesh_actor = self.plotter.add_mesh(
                self.pv_mesh,
                scalars='rgb',
                rgb=True,
                lighting=True,
                ambient=0.2,
                specular=0.3,
                specular_power=5,
                pickable=False
            )
            print("Mesh with features shown.")
        else:
            self.mesh_actor = self.plotter.add_mesh(
                self.pv_mesh,
                color='gray',
                lighting=True,
                ambient=0.2,
                specular=0.3,
                specular_power=5,
                pickable=False
            )
            print("Mesh in gray shown.")

    def hide_features(self):
        """Toggle feature coloring for currently visible object."""
        self.features_visible = not self.features_visible
        if self.current_visible == 'mesh':
            self._show_mesh_actor()
        elif self.current_visible == 'point_cloud':
            self._show_point_cloud_actor()
        else:
            print("No object visible to toggle features.")

    def save_current_view(self):
        """Save current camera position and screenshot."""
        # Get current camera position
        cam_pos = self.plotter.camera_position
        
        # Convert data to serializable format
        position = tuple(cam_pos[0])
        focal_point = tuple(cam_pos[1])
        up_vector = tuple(cam_pos[2])
        
        # Save camera position
        view_id = len(self.camera_positions)
        self.camera_positions.append({
            'position': position,
            'focal_point': focal_point,
            'up_vector': up_vector
        })
        
        # Take screenshot of current view
        screenshot_path = os.path.join(self.output_dir, f"{self.base_filename}_view_{view_id}.png")
        self.plotter.screenshot(screenshot_path, window_size=(2400, 1800))
        
        # Show confirmation
        print(f"View {view_id} saved: {screenshot_path}")
        
    def reset_view(self):
        """Reset camera to default view."""
        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()
        self.hide_all_planes()
        self.hide_all_axes()
        print("View reset to default isometric view, planes and axes hidden")

    def hide_all_planes(self):
        for actor in self.planes_actors:
            self.plotter.remove_actor(actor)
        self.planes_actors = []
        self.pre_planes.extend(self.planes)
        self.planes = []
        self.pre_planes.sort(key=lambda item: item['order'])

    def hide_all_axes(self):
        for actor in self.axis_actors:
            self.plotter.remove_actor(actor)
        self.axis_actors = []
        self.pre_axis.extend(self.axis)
        self.axis = []
        self.pre_axis.sort(key=lambda item: item['order'])
    
    def add_current_plane(self):
        """Add a plane to the current view."""
        if not self.pre_planes:
            print("No planes to add.")
            return
        plane = self.pre_planes[0]
        self.pre_planes = self.pre_planes[1:]
        self.add_plane(plane)
        
    def add_pre_plane(self, point, normal, order):
        # Save pre-plane data
        self.pre_planes.append({
            'order': order,
            'point': tuple(point),
            'normal': tuple(normal)
        })

    def add_plane(self, plane):
        """Add a plane to the scene."""
        order = plane['order']
        point = plane['point']
        normal = plane['normal']
        # Highlight the slice (plane intersection)
        sliced = self.pv_mesh.slice(normal=normal, origin=point)
        plane_actor = self.plotter.add_mesh(sliced, color='red', line_width=5, pickable=False)
        self.planes_actors.append(plane_actor)
        # Save plane data
        self.planes.append(plane)
        print(f"Plane added ({order}): point={point}, normal={normal}")

    def add_current_axis(self):
        """Add an axis to the current view."""
        if not self.pre_axis:
            print("No axes to add.")
            return
        axis = self.pre_axis[0]
        self.pre_axis = self.pre_axis[1:]
        self.add_axis(axis)

    def add_pre_axis(self, point, normal, order):
        # Save pre-axis data
        self.pre_axis.append({
            'order': order,
            'point': tuple(point),
            'normal': tuple(normal)
        })

    def add_axis(self, axis):
        """Add an axis to the scene."""
        order = axis['order']
        point = axis['point']
        normal = axis['normal']
        # Highlight the axis
        axis_length = 10.0
        point = np.array(point)
        normal = np.array(normal)
        start = point - normal * axis_length
        end = point + normal * axis_length 
        line = pv.Line(start, end)  

        axis_actor = self.plotter.add_mesh(line, color='blue', line_width=4)
        self.axis_actors.append(axis_actor)
        # Save axis data
        self.axis.append(axis)
        print(f"Axis added ({order}): point={point}, direction={normal}")