"""
compute_mesh_scaling.py

Script to compute a scaling factor between the reconstructed meshes and input partial pointclouds.
Computes the pixel-wise correspondences and project to point-correspondences to estimate scale factor.
"""

import os
import os.path as osp
import numpy as np
import matplotlib.image as mpimg
import open3d as o3d
from copy import deepcopy
from argparse import ArgumentParser

# Scaling utilities
from scenecomplete.scripts.python.scaling.scaling_utils import (
    get_dino_correspondences,
    get_pointcloud,
    get_pixel_indices,
    highlight_image,
    highlight_points,
    is_origin,
    estimate_similarity_transform,
)

DEBUG = False

def compute_scaling_for_object(
    instant_mesh_dirpath: str,
    segmentation_data_dirpath: str,
    obj_id: int,
    debug: bool=False,
    num_correspondences: int=10,
):
    """
    Compute the scaling factor for a single object based on DINO-ViT correspondences
    between the reconstructed mesh's rendered image and the segmented image of the object.

    Args:
        instant_mesh_dirpath (str): Path to the folder with inpainted images/videos, meshes, etc.
        segmentation_data_dirpath (str): Path to segmentation_data containing the segmented depth, RGB, etc.
        obj_id (int): The object ID to process (example: 0, 1, 2, etc.).
        debug (bool): If True, enable debug prints and visualizations.

    Returns:
        scale (float or None): The scale factor. If no correspondences found, returns None.
    """
    # Build paths
    image_path1 = f'{instant_mesh_dirpath}/videos/{obj_id}_rgba.png'
    image_path2 = f'{segmentation_data_dirpath}/{obj_id}_masked.png'

    depth_path1 = f'{instant_mesh_dirpath}/videos/{obj_id}_rgba_depth.png'
    depth_path2 = f'{segmentation_data_dirpath}/{obj_id}_depth.png'

    intrinsics_file1 = f'{instant_mesh_dirpath}/videos/{obj_id}_rgba.json'
    intrinsics_file2 = f'{segmentation_data_dirpath}/cam_K.json'

    # 1. Get correspondences via DINO
    try:
        px1, px2, im1_pil, im2_pil = get_dino_correspondences(image_path1, image_path2, debug=debug)
    except Exception as e:
        if debug:
            print(f"[DEBUG] No correspondences found or error: {e}")
        return None

    # 2. Scale the DINO pixel coords to the original image resolution
    color_image1 = (mpimg.imread(image_path1) * 255).astype(np.uint8)
    color_image2 = (mpimg.imread(image_path2) * 255).astype(np.uint8)
    
    height1, width1, _ = color_image1.shape
    resized_w1, resized_h1 = im1_pil.size  # PIL => (width, height)
    
    height2, width2, _ = color_image2.shape
    resized_w2, resized_h2 = im2_pil.size

    # Take the first {num_correspondences} correspondences for demonstration; you can vary
    pixel_indices = np.arange(num_correspondences)
    px1_scaled = [
        (int(pt[0]*height1/resized_h1), int(pt[1]*width1/resized_w1))
        for i, pt in enumerate(px1) if i in pixel_indices
    ]
    px2_scaled = [
        (int(pt[0]*height2/resized_h2), int(pt[1]*width2/resized_w2))
        for i, pt in enumerate(px2) if i in pixel_indices
    ]

    if debug:
        highlight_image(image_path1, px1_scaled)
        highlight_image(image_path2, px2_scaled)

    # 3. Build pointclouds
    pcd1 = get_pointcloud(depth_path1, image_path1, intrinsics_file1)
    pcd2 = get_pointcloud(depth_path2, image_path2, intrinsics_file2)

    # 4. Retrieve the 3D points that correspond to the scaled pixel coords
    pcd_indices1, pcd_values1 = get_pixel_indices(color_image1, pcd1, px1_scaled)
    pcd_indices2, pcd_values2 = get_pixel_indices(color_image2, pcd2, px2_scaled)

    # Filter out origin points
    keep_pcd1 = []
    keep_pcd2 = []
    keep_idx1 = []
    keep_idx2 = []
    for i, (v1, v2) in enumerate(zip(pcd_values1, pcd_values2)):
        if is_origin(v1) or is_origin(v2):
            continue
        keep_pcd1.append(v1)
        keep_pcd2.append(v2)
        keep_idx1.append(pcd_indices1[i])
        keep_idx2.append(pcd_indices2[i])

    if debug:
        highlight_points(pcd1, keep_idx1, visualize=False)
        highlight_points(pcd2, keep_idx2, visualize=False)

    if len(keep_pcd1) == 0:
        if debug:
            print("[DEBUG] No valid 3D correspondences after filtering origin points.")
        return None

    keep_pcd1 = np.asarray(keep_pcd1)
    keep_pcd2 = np.asarray(keep_pcd2)

    # 5. Estimate transform
    try:
        _, scale_factor = estimate_similarity_transform(keep_pcd1, keep_pcd2)
    except Exception as e:
        if debug:
            print(f"[DEBUG] Could not estimate transform. Error: {e}")
        return None

    return scale_factor


def main():
    parser = ArgumentParser(description="Compute scaling factor between reconstructed mesh and partial pointcloud.")
    parser.add_argument("--segmentation_dirpath", type=str, required=True,
                        help="Directory containing segmentation masks.")
    parser.add_argument("--imesh_outputs", type=str, default="imesh_outputs",
                        help="Directory containing reconstruction outputs.")
    parser.add_argument("--output_dirpath", type=str, required=True,
                        help="Directory to store scaled meshes.")
    parser.add_argument("--instant_mesh_model", type=str, default="instant-mesh-base",
                        help="Name of the Instant Mesh model.")
    parser.add_argument("--camera_name", type=str, default="realsense",
                        help="Name of the camera.")
    parser.add_argument("--debug", action="store_true",
                        help="If set, enables debug logs and extra visualizations.")

    args = parser.parse_args()
    DEBUG = args.debug

    os.makedirs(args.output_dirpath, exist_ok=True)

    # Build directories
    instant_mesh_model = args.instant_mesh_model
    instant_mesh_dirpath = osp.join(args.imesh_outputs, instant_mesh_model)
    segmentation_data_dirpath = args.segmentation_dirpath

    # Sort objects by numeric ID
    images_dir = osp.join(instant_mesh_dirpath, 'images')
    if not osp.isdir(images_dir):
        print(f"[ERROR] Missing directory: {images_dir}")
        return

    objects = os.listdir(images_dir)
    obj_ids = sorted(int(obj.split('_')[0]) for obj in objects)
    
    if DEBUG:
        obj_ids = obj_ids[args.debug_index:]

    obj_scale_mapping = {} # obj_id -> scale factor
    avg_scale = 0.0
    count = 0

    for obj_id in obj_ids:
        scale = compute_scaling_for_object(
            instant_mesh_dirpath=instant_mesh_dirpath,
            segmentation_data_dirpath=segmentation_data_dirpath,
            obj_id=obj_id,
            debug=DEBUG
        )
        obj_scale_mapping[obj_id] = scale
        if scale is not None:
            avg_scale = (avg_scale * count + scale) / (count + 1)
            count += 1

    # Fill in None values with average scale
    for k, v in obj_scale_mapping.items():
        if v is None:
            print(f"[WARNING] Filling None value for object {k} with average scale {avg_scale}")
            obj_scale_mapping[k] = avg_scale

    # Write results
    scale_txt_path = osp.join(args.output_dirpath, "obj_scale_mapping.txt")
    with open(scale_txt_path, "w") as f:
        for key, val in obj_scale_mapping.items():
            f.write(f"{key}:{val}\n")

    print(f"[INFO] Scale mapping saved to {scale_txt_path}")


if __name__ == "__main__":
    main()
