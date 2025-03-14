"""
registration_utils.py

Utilities for registering a reconstructed mesh against a partial object point cloud.
"""

import os
import cv2
import numpy as np
import imageio
import open3d as o3d
import trimesh
from copy import deepcopy

def load_mask(mask_path: str, width: int, height: int) -> np.ndarray:
    """
    Load and resize a mask image to (height, width).
    If the file is multi-channel, picks the first non-empty channel.

    Args:
        mask_path (str): Path to the mask image on disk.
        width (int): Desired width.
        height (int): Desired height.

    Returns:
        np.ndarray: Binary mask of shape (height, width) in {0,1}.
    """
    mask = cv2.imread(mask_path, -1)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    # If multi-channel, pick the first non-empty channel
    if len(mask.shape) == 3:
        for c in range(3):
            if mask[..., c].sum() > 0:
                mask = mask[..., c]
                break

    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    # Binarize
    mask_resized = (mask_resized > 0).astype(np.uint8)
    return mask_resized


def load_color(color_path: str, width: int, height: int) -> np.ndarray:
    """
    Load and resize an RGB color image.

    Args:
        color_path (str): Path to an RGB image (PNG/JPG).
        width (int): Desired width.
        height (int): Desired height.

    Returns:
        np.ndarray: shape (height, width, 3) in [0..255].
    """
    color = imageio.imread(color_path)
    if color.ndim == 2:
        # Grayscale => expand dimension
        color = np.stack([color, color, color], axis=-1)
    elif color.shape[-1] == 4:
        color = color[..., :3]

    color = cv2.resize(color, (width, height), interpolation=cv2.INTER_NEAREST)
    return color


def load_depth(depth_path: str, width: int, height: int, zfar: float = np.inf) -> np.ndarray:
    """
    Load and resize a depth image (16-bit or 32-bit), converting to meters.

    Args:
        depth_path (str): Path to a depth image on disk.
        width (int): Desired width.
        height (int): Desired height.
        zfar (float): Far plane distance (anything beyond is set to 0).

    Returns:
        np.ndarray: shape (height, width) in meters.
    """
    depth = cv2.imread(depth_path, -1)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth: {depth_path}")

    # Convert to meters if the raw is in millimeters
    depth_m = depth.astype(np.float32) / 1e3
    depth_resized = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)

    # Zero out invalid or too-far depths
    depth_resized[(depth_resized < 0.001) | (depth_resized >= zfar)] = 0
    return depth_resized


def load_camera_intrinsics(k_path: str, downscale: float = 1.0) -> np.ndarray:
    """
    Load a 3x3 camera intrinsics from a .txt file, shape (3,3).

    Args:
        k_path (str): Path to a text file with 9 floats representing the intrinsic matrix row by row.
        downscale (float): If the image was resized by 'downscale', scale fx, fy, cx, cy accordingly.

    Returns:
        np.ndarray: shape (3,3) float intrinsics.
    """
    K = np.loadtxt(k_path).reshape(3, 3)
    K[:2] *= downscale
    return K


def transform_and_export_mesh(
    mesh_path: str,
    scale_factor: float,
    transform_pose: np.ndarray,
    out_obj_path: str,
    debug: bool = False
) -> None:
    """
    Load a mesh (.obj) from disk, apply scale and transformation,
    then export to a new .obj. Also handles rotating bounding boxes if desired.

    Args:
        mesh_path (str): Path to the .obj file.
        scale_factor (float): Uniform scale to apply to the mesh.
        transform_pose (np.ndarray): 4x4 matrix to transform the mesh in place.
        out_obj_path (str): Where to save the new .obj file.
        debug (bool): If True, print debug info.
    """
    mesh = trimesh.load(mesh_path)
    if debug:
        print(f"[DEBUG] Original mesh has {len(mesh.vertices)} vertices.")

    mesh.apply_scale(scale_factor)
    mesh.apply_transform(transform_pose)

    mesh.export(out_obj_path)
    if debug:
        print(f"[DEBUG] Transformed mesh exported to {out_obj_path}")


def rename_mtl_and_texture(
    obj_file_path: str,
    old_mtl_name: str = "material.mtl",
    old_tex_name: str = "material_0.png",
    new_mtl_prefix: str = "",
    new_tex_prefix: str = "",
    debug: bool = False
):
    """
    For the .obj file at 'obj_file_path', rename references to the .mtl and texture.
    Then physically rename the .mtl and texture files on disk.

    Args:
        obj_file_path (str): The .obj file path to read+edit lines.
        old_mtl_name (str): The old name of the MTL file (in the .obj).
        old_tex_name (str): The old name of the texture (in the MTL).
        new_mtl_prefix (str): e.g. "123_" => "123_material.mtl"
        new_tex_prefix (str): e.g. "123_" => "123_texture.png"
        debug (bool): If True, prints debug logs.
    """
    obj_dir = os.path.dirname(obj_file_path)
    base_obj_name = os.path.splitext(os.path.basename(obj_file_path))[0]

    new_mtl_name = f"{new_mtl_prefix}material.mtl" if new_mtl_prefix else "material.mtl"
    new_tex_name = f"{new_tex_prefix}texture.png" if new_tex_prefix else "texture.png"

    old_mtl_path = os.path.join(obj_dir, old_mtl_name)
    old_tex_path = os.path.join(obj_dir, old_tex_name)

    new_mtl_path = os.path.join(obj_dir, new_mtl_name)
    new_tex_path = os.path.join(obj_dir, new_tex_name)

    # Move old texture
    if os.path.isfile(old_tex_path):
        os.rename(old_tex_path, new_tex_path)
    else:
        if debug:
            print(f"[DEBUG] Could not find old texture: {old_tex_path}")

    # Update references in the MTL file
    if os.path.isfile(old_mtl_path):
        with open(old_mtl_path, 'r') as f:
            data = f.read()
        data = data.replace(old_tex_name, new_tex_name)
        with open(old_mtl_path, 'w') as f:
            f.write(data)
        # Rename the MTL
        os.rename(old_mtl_path, new_mtl_path)
    else:
        if debug:
            print(f"[DEBUG] Could not find old MTL: {old_mtl_path}")

    # Update .obj references
    with open(obj_file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith("mtllib") and old_mtl_name in line:
            lines[i] = line.replace(old_mtl_name, new_mtl_name)

    with open(obj_file_path, 'w') as f:
        f.writelines(lines)


def depth_to_xyz_map(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Converts a depth map to a per-pixel 3D coordinate map in camera space.

    Args:
        depth (np.ndarray): shape(H, W) in meters.
        K (np.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        xyz_map (np.ndarray): shape(H, W, 3) of 3D points.
    """
    H, W = depth.shape
    i_range = np.arange(W)
    j_range = np.arange(H)
    u, v = np.meshgrid(i_range, j_range)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    xyz_map = np.dstack((x, y, z))
    return xyz_map


def numpy_to_open3d_cloud(points: np.ndarray, colors: np.ndarray=None) -> o3d.geometry.PointCloud:
    """
    Convert Nx3 numpy arrays to Open3D point cloud. Optional Nx3 color array.

    Args:
        points (np.ndarray): shape(N, 3).
        colors (np.ndarray): shape(N, 3), optional.

    Returns:
        pc (o3d.geometry.PointCloud).
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(colors)
    return pc