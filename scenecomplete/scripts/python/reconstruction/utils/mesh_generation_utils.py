"""
zero123plus_utils.py

Utility functions for the Zero123Plus-based pipeline.
Contains camera creation, and rendering for color (RGB) and depth frames.
"""

import torch
import numpy as np
from tqdm import tqdm
from scenecomplete.modules.InstantMesh.src.utils.camera_util import (
    FOV_to_intrinsics,
    get_circular_camera_poses
)

def get_render_cameras(
    batch_size=1,
    num_views=120,
    radius=4.0,
    elevation=20.0,
    use_flexicubes=False,
    return_intrinsics=False,
    fov_degrees=30.0
):
    """
    Get the rendering camera parameters for circular views around an object. 

    Args:
        batch_size (int): Batch size
        num_views (int): Number of camera viewpoints in a circular path.
        radius (float): Distance from the center.
        elevation (float): Camera elevation angle in degrees.
        use_flexicubes (bool): If True, use FlexiCubes for geometric representation.
        return_intrinsics (bool): If True, return the intrinsics matrix.
        fov_degrees (float): Field of view in degrees to compute intrinsics.

    Returns:
        cameras (torch.Tensor): Camera parameters or extrinsics, shape depends on flexicubes usage.
        intrinsics (torch.Tensor, optional): If return_intrinsics=True, also returns the intrinsics.
    """
    # shape: (M, 4, 4)
    c2ws = get_circular_camera_poses(M=num_views, radius=radius, elevation=elevation, azi=False)

    # If using flexicubes, we keep cameras as invert of c2ws
    if use_flexicubes:
        cameras = torch.linalg.inv(c2ws).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        intrinsics = FOV_to_intrinsics(fov_degrees).unsqueeze(0).repeat(num_views, 1, 1).float().flatten(-2)
        if return_intrinsics:
            return cameras, intrinsics
        return cameras

    # Otherwise, flatten extrinsics + intrinsics
    extrinsics = c2ws.flatten(-2)
    intrinsics = FOV_to_intrinsics(fov_degrees).unsqueeze(0).repeat(num_views, 1, 1).float().flatten(-2)
    cameras = torch.cat([extrinsics, intrinsics], dim=-1)  # shape (M, 4+4=8) if flattening by 2 dims
    cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)  # shape (batch_size, M, 8)
    if return_intrinsics:
        return cameras, intrinsics
    return cameras


def render_rgb_frames(
    model,
    planes,
    cameras,
    render_size=512,
    chunk_size=1,
    use_flexicubes=False
):
    """
    Render color (RGB) frames from triplanes.

    Args:
        model: Reconstruction model.
        planes: Learned triplanes.
        cameras (torch.Tensor): Camera parameters or extrinsics to use for rendering.
        render_size (int): Resolution of the output rendering.
        chunk_size (int): Number of cameras to process at once.
        use_flexicubes (bool): If True, calls model.forward_geometry instead of forward_synthesizer.

    Returns:
        frames (torch.Tensor): Tensor of rendered frames, shape (M, C, H, W).
    """
    frames_list = []
    total_cams = cameras.shape[1]
    for i in tqdm(range(0, total_cams, chunk_size), desc="Rendering RGB frames"):
        chunk = cameras[:, i:i+chunk_size]
        if use_flexicubes:
            out = model.forward_geometry(
                planes,
                chunk,
                render_size=render_size,
            )['img']
        else:
            out = model.forward_synthesizer(
                planes,
                chunk,
                render_size=render_size,
            )['images_rgb']
        frames_list.append(out)

    frames = torch.cat(frames_list, dim=1)[0]  # shape (M, C, H, W) if batch_size=1
    return frames


def render_depth_frames(
    model,
    planes,
    cameras,
    render_size=512,
    chunk_size=1,
    use_flexicubes=False
):
    """
    Render depth frames from triplanes.

    Args:
        model: The reconstruction model.
        planes: The learned triplanes.
        cameras (torch.Tensor): Camera parameters or extrinsics to use for rendering.
        render_size (int): Resolution of the output rendering.
        chunk_size (int): Number of cameras to process at once.
        use_flexicubes (bool): If True, calls model.forward_geometry for 'depth' key.
    
    Returns:
        frames (torch.Tensor): Tensor of rendered depth frames, shape (M, 1, H, W) or similar.
    """
    frames_list = []
    total_cams = cameras.shape[1]
    for i in tqdm(range(0, total_cams, chunk_size), desc="Rendering depth frames"):
        chunk = cameras[:, i:i+chunk_size]
        if use_flexicubes:
            out = model.forward_geometry(
                planes,
                chunk,
                render_size=render_size,
            )['depth']
        else:
            out = model.forward_synthesizer(
                planes,
                chunk,
                render_size=render_size,
            )['images_depth']
        frames_list.append(out)

    frames = torch.cat(frames_list, dim=1)[0]  # shape (M, 1, H, W) if batch_size=1
    return frames