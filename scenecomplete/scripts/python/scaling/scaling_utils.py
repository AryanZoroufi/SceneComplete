"""
mesh_scaling_utils.py

Utilities for computing pixel-wise correspondences between an image and a 3D model
to estimate scaling (and optionally transform) a reconstructed mesh
so it aligns with a partial pointcloud of the object.
"""

import os
import json
import cv2
import open3d as o3d
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
import torch
from scipy.spatial.transform import Rotation

from scenecomplete.modules.dino_vit_features.correspondences import find_correspondences, draw_correspondences

def read_camera_intrinsics(intrinsic_json: str) -> dict:
    """
    Read intrinsics from a JSON file, returning them as a dictionary
    in the format: {'intrinsic_matrix': [...], 'width': .., 'height': .., ... }.
    """
    with open(intrinsic_json, 'r') as f:
        data = json.load(f)
    return data


def parse_intrinsics(intrinsics_file: str):
    """
    Extract fx, fy, cx, cy from the loaded intrinsics JSON.
    """
    data = read_camera_intrinsics(intrinsics_file)
    fx = data['intrinsic_matrix'][0]
    fy = data['intrinsic_matrix'][4]
    cx = data['intrinsic_matrix'][6]
    cy = data['intrinsic_matrix'][7]
    return cx, cy, fx, fy


def project_depth_to_pC(depth_pixels: np.ndarray, intrinsics_file: str) -> np.ndarray:
    """
    Project depth pixel coordinates into 3D camera space using a pinhole camera model.

    Args:
        depth_pixels (np.ndarray): shape (N, 3) or (3,) containing [u, v, Z].
        intrinsics_file (str): Path to JSON intrinsics file.

    Returns:
        pC (np.ndarray): shape (N, 3) array of 3D points in camera coordinates.
    """
    v = depth_pixels[:, 0]  # row index
    u = depth_pixels[:, 1]  # col index
    Z = depth_pixels[:, 2]

    cx, cy, fx, fy = parse_intrinsics(intrinsics_file)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pC = np.c_[X, Y, Z]
    return pC


def get_pointcloud(
    depth_path: str,
    image_path: str,
    intrinsics_file: str,
    remove_origin: bool=False
) -> o3d.geometry.PointCloud:
    """
    Construct an Open3D pointcloud from depth + color images.

    Args:
        depth_path (str): Path to a 16-bit or 32-bit depth image.
        image_path (str): Path to the corresponding color image (RGB).
        intrinsics_file (str): Path to a JSON file with camera intrinsics.
        remove_origin (bool): If True, removes points with zero (0,0,0) coords.

    Returns:
        pcd (o3d.geometry.PointCloud): PointCloud with 'points' and 'colors'.
    """
    color_image = (mpimg.imread(image_path) * 255).astype(np.uint8)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    # Build a list of per-pixel colors
    height, width, _ = color_image.shape
    color_points = []
    for j in range(height):
        for i in range(width):
            color_points.append(color_image[j, i] / 255.0)

    # Project the depth into 3D
    u_range = np.arange(depth_image.shape[0])
    v_range = np.arange(depth_image.shape[1])
    depth_v, depth_u = np.meshgrid(v_range, u_range)
    depth_points = np.dstack([depth_u, depth_v, depth_image / 1000.0])
    depth_points = depth_points.reshape((-1, 3))

    pC = project_depth_to_pC(depth_points, intrinsics_file)

    # Optionally remove zeros
    if remove_origin:
        pC_filtered = []
        color_filtered = []
        for idx, point in enumerate(pC):
            if np.linalg.norm(point) == 0:
                continue
            pC_filtered.append(point)
            color_filtered.append(color_points[idx])
        pC = np.array(pC_filtered)
        color_points = np.array(color_filtered)
    else:
        color_points = np.array(color_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pC)
    pcd.colors = o3d.utility.Vector3dVector(color_points)

    return pcd


def get_dino_correspondences(image_path1: str, image_path2: str, debug=False):
    """
    Use the DINO-based 'find_correspondences()' to get correspondences among images with different viewpoints.

    Args:
        image_path1 (str): Path to first image.
        image_path2 (str): Path to second image.
        debug (bool): If True, prints or plots debug info.

    Returns:
        pixel_points1, pixel_points2, image1_pil, image2_pil
    """
    # default parameters (you might expose some as arguments)
    num_pairs = 15
    load_size = 224
    layer = 9
    facet = 'key'
    bin = True
    thresh = 0.05
    model_type = 'dino_vits8'
    stride = 4

    with torch.no_grad():
        pixel_points1, pixel_points2, image1_pil, image2_pil = find_correspondences(
            image_path1, 
            image_path2, 
            num_pairs, 
            load_size, 
            layer,
            facet, 
            bin, 
            thresh, 
            model_type, 
            stride
        )

    if debug:
        fig1, ax1 = plt.subplots()
        ax1.imshow(image1_pil)
        ax1.axis("off")

        fig2, ax2 = plt.subplots()
        ax2.imshow(image2_pil)
        ax2.axis("off")

        # Visualize the correspondences
        draw_correspondences(pixel_points1, pixel_points2, image1_pil, image2_pil)
        plt.show()

    return pixel_points1, pixel_points2, image1_pil, image2_pil


def highlight_image(image_path, points):
    """
    Debugging helper to highlight certain pixel coordinates on an image (for plotting).
    """
    image = mpimg.imread(image_path)
    if not isinstance(points, list):
        points = [points]

    plt.imshow(image)
    for pt in points:
        plt.plot(pt[1], pt[0], 'og', markersize=10)
    plt.show()
    plt.close()


def highlight_points(pcd: o3d.geometry.PointCloud, pcd_indices: list, visualize=False):
    """
    Visualize (optionally) certain points on an Open3D pointcloud by coloring them black.

    Args:
        pcd (o3d.geometry.PointCloud): The base cloud.
        pcd_indices (list): Indices of points to highlight.
        visualize (bool): If True, open an O3D visualization window.
    """
    pcd_points = np.asarray(pcd.points)
    new_geoms = [pcd]

    # create a small cluster around each highlight
    for idx in pcd_indices:
        cluster_points = np.random.normal(loc=pcd_points[idx], scale=5e-3, size=(50, 3))
        cluster_colors = np.zeros((50, 3))  # black

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

        new_geoms.append(cluster_pcd)

    if visualize:
        o3d.visualization.draw_geometries(new_geoms)
    return new_geoms


def get_pixel_indices(image: np.ndarray, pcd: o3d.geometry.PointCloud, pixel_coords: list):
    """
    Given a set of 2D pixel coordinates, compute the corresponding indices in the flattened
    pointcloud (which is row-major).
    
    Args:
        image (np.ndarray): The color image, shape(H, W, 3).
        pcd (o3d.geometry.PointCloud): The pointcloud from get_pointcloud().
        pixel_coords (list): List of (row, col) pixel coordinates.

    Returns:
        pcd_indices (list): Indices in pcd.points.
        pcd_values (list): The actual 3D points.
    """
    height, width, _ = image.shape
    pcd_points = np.asarray(pcd.points)

    pcd_indices = []
    pcd_values = []

    for px in pixel_coords:
        row, col = px
        idx = row * width + col
        pcd_indices.append(idx)
        pcd_values.append(pcd_points[idx, :])

    return pcd_indices, pcd_values


def is_origin(point: np.ndarray) -> bool:
    """
    Check if a 3D point is effectively the origin (0,0,0).
    """
    return np.linalg.norm(point) == 0


def estimate_affine_transformation(source_pts: np.ndarray, target_pts: np.ndarray) -> np.ndarray:
    """
    Estimate a 4x4 affine transformation (including scale, shear, rotation, translation)
    via least squares from correspondences.

    Args:
        source_pts (np.ndarray): shape (N, 3)
        target_pts (np.ndarray): shape (N, 3)

    Returns:
        transformation_matrix (np.ndarray): 4x4 homogeneous transform.
    """
    ones_col = np.ones((source_pts.shape[0], 1))
    source_hom = np.hstack([source_pts, ones_col])   # shape (N, 4)

    # solve A * X = B => X ~ np.linalg.lstsq(A, B)
    X, residuals, rank, s = np.linalg.lstsq(source_hom, target_pts, rcond=None)
    # X is shape (4, 3) => we want a 4x4
    transform = np.vstack([X.T, [0, 0, 0, 1]])
    return transform


def estimate_similarity_transform(source_pts: np.ndarray, target_pts: np.ndarray):
    """
    Estimate a similarity transform (uniform scale + rotation + translation)
    that aligns source_pts to target_pts. (No shear)

    Args:
        source_pts (np.ndarray): shape (N, 3)
        target_pts (np.ndarray): shape (N, 3)

    Returns:
        transformation_matrix (np.ndarray): 4x4 homogeneous transform
        scale (float): The uniform scale factor
    """
    centroid_s = np.mean(source_pts, axis=0)
    centroid_t = np.mean(target_pts, axis=0)

    centered_s = source_pts - centroid_s
    centered_t = target_pts - centroid_t

    norm_s = np.linalg.norm(centered_s)
    norm_t = np.linalg.norm(centered_t)
    scale = norm_t / norm_s if norm_s != 0 else 1.0
    scaled_s = centered_s * scale

    cov = np.dot(scaled_s.T, centered_t)
    U, _, Vt = np.linalg.svd(cov)

    rotation = np.dot(U, Vt)
    # Ensure a proper rotation (no reflection)
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = np.dot(U, Vt)

    translation = centroid_t - np.dot(rotation, centroid_s * scale)

    transform = np.eye(4)
    transform[:3, :3] = rotation * scale
    transform[:3, 3] = translation

    return transform, scale


def run_icp_initialization(
    pcd_target: o3d.geometry.PointCloud,
    pcd_source: o3d.geometry.PointCloud,
    max_correspondence_dist: float=0.02,
    debug=False
) -> np.ndarray:
    """
    Use ICP to register pcd_source onto pcd_target
    (point-to-point).

    Args:
        pcd_target (o3d.geometry.PointCloud): The reference cloud
        pcd_source (o3d.geometry.PointCloud): The cloud to be transformed
        max_correspondence_dist (float): Maximum distance for ICP
        debug (bool): If True, does some extra visualization.

    Returns:
        transform (np.ndarray): 4x4 transform that maps pcd_source into pcd_target.
    """
    pcd_target.estimate_normals()
    pcd_source.estimate_normals()

    result = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, max_correspondence_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    if debug:
        # visualize before & after
        o3d.visualization.draw_geometries([pcd_target, pcd_source])
        src_transformed = deepcopy(pcd_source).transform(result.transformation)
        o3d.visualization.draw_geometries([pcd_target, src_transformed])

    # Typically, we want the transform that maps pcd_target -> pcd_source, so invert:
    return np.linalg.inv(result.transformation)
