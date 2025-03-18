import numpy as np
import open3d as o3d
import cv2
import os
import argparse

def create_scene_pointcloud(rgb_path, depth_path, intrinsics_path):
    """
    Generate a colored point cloud from RGB-D data.
    
    Args:
        rgb_path (str): Path to RGB image
        depth_path (str): Path to depth image 
        intrinsics_path (str): Path to camera intrinsics file
    
    Returns:
        o3d.geometry.PointCloud: Colored point cloud of the scene
    """
    # Read RGB and depth images
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, -1)  # Read depth as-is
    
    # Load camera intrinsics
    K = np.loadtxt(intrinsics_path)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    
    # Create point cloud from depth
    rows, cols = depth.shape
    
    # Get x,y grid
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Back-project 2D points to 3D
    z = depth.astype(float) / 1000.0  # Convert to meters if depth is in mm
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    # Stack to 3D points
    xyz = np.stack([x, y, z], axis=-1)
    
    # Create open3d point cloud
    pcd = o3d.geometry.PointCloud()
    
    # Filter out invalid depth points
    valid_depth = z > 0
    xyz = xyz[valid_depth]
    rgb = rgb[valid_depth]
    
    # Set points and colors
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(float) / 255.0)
    
    # Estimate normals
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location()
    
    return pcd

def main():
    parser = argparse.ArgumentParser(description="Generate scene point cloud from RGB-D data")
    parser.add_argument('--folder_path', type=str, required=True, 
                        help='Path to folder containing rgb.png, depth.png and cam_K.txt')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the point cloud (optional)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the point cloud after generation')
    args = parser.parse_args()
    
    # Construct file paths
    rgb_path = os.path.join(args.folder_path, 'rgb.png')
    depth_path = os.path.join(args.folder_path, 'depth.png')
    intrinsics_path = os.path.join(args.folder_path, 'cam_K.txt')
    
    # Check if files exist
    for filepath in [rgb_path, depth_path, intrinsics_path]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")
    
    # Generate point cloud
    pcd = create_scene_pointcloud(rgb_path, depth_path, intrinsics_path)
    
    # Save point cloud if output path is provided
    if args.output_path:
        o3d.io.write_point_cloud(args.output_path, pcd)
        print(f"Point cloud saved to: {args.output_path}")
    
    # Visualize if requested
    if args.visualize:
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()