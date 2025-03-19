import os
import open3d as o3d
import numpy as np

def visualize_scene_with_objects(obj_folder_path):
    """
    Visualize multiple obj files along with a scene ply file.
    
    Args:
        obj_folder_path (str): Path to folder containing obj files
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    scene_ply_path = os.path.join(obj_folder_path, "scene_complete.ply")

    # Check if scene file exists
    if not os.path.exists(scene_ply_path):
        raise FileNotFoundError(f"Scene file not found at {scene_ply_path}")

    # Load scene as point cloud instead of mesh
    scene = o3d.io.read_point_cloud(scene_ply_path)
    if not scene.has_points():
        raise ValueError(f"No points found in scene file {scene_ply_path}")

    # Add scene point cloud
    vis.add_geometry(scene)

    # Set initial render options for point cloud
    opt = vis.get_render_option()
    opt.point_size = 1.0

    # Load and add all obj files
    for filename in os.listdir(obj_folder_path):
        if filename.endswith('.obj'):
            obj_path = os.path.join(obj_folder_path, filename)
            mesh = o3d.io.read_triangle_mesh(obj_path)
            mesh.compute_vertex_normals()
            vis.add_geometry(mesh)

    # Set up visualization options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
    opt.show_coordinate_frame = True

    # Set up camera
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    # Run visualization
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize scene with objects')
    parser.add_argument('--obj_folder', type=str, required=True,
                        help='Path to folder containing obj files')
    
    args = parser.parse_args()
    
    visualize_scene_with_objects(args.obj_folder)
