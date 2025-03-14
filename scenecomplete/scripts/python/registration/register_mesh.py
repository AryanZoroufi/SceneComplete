"""
register_mesh.py

Script to register the reconstructed meshes against their partial object pointclouds. 
"""

import os
import argparse
import logging
from glob import glob
import numpy as np
import cv2
import open3d as o3d
import trimesh
from copy import deepcopy

# local util imports
from scenecomplete.scripts.python.registration.utils.registration_utils import (
    load_mask,
    load_color,
    load_depth,
    load_camera_intrinsics,
    transform_and_export_mesh,
    rename_mtl_and_texture,
    depth_to_xyz_map,
    numpy_to_open3d_cloud,
)

import nvdiffrast.torch as dr
from scenecomplete.modules.FoundationPose.estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
from scenecomplete.modules.FoundationPose.datareader import set_logging_format, set_seed

def main():
    parser = argparse.ArgumentParser(description="Register a reconstructed mesh to partial object point cloud.")
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('--imesh_outputs', type=str, required=True,
                        help="Directory containing reconstructed .obj files")
    parser.add_argument('--segmentation_dirpath', type=str, required=True,
                        help="Directory containing the scene's full images and mask files, etc.")
    parser.add_argument('--obj_scale_mapping', type=str, required=True,
                        help="Path to the 'obj_idx:scale_value' mapping file.")
    parser.add_argument("--instant_mesh_model", type=str, default="instant-mesh-base",
                        help="Name of the Instant Mesh model.")
    parser.add_argument('--est_refine_iter', type=int, default=5, help="Pose refinement iterations.")
    parser.add_argument('--track_refine_iter', type=int, default=2, help="(Unused) Additional refine iterations.")
    parser.add_argument('--debug', type=int, default=1, help="If 1, run in debug mode.")
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug',
                        help="Where to store debug info.")
    parser.add_argument('--output_dirpath', type=str, required=True,
                        help="Directory to store the registered meshes")

    args = parser.parse_args()
    print(args)

    # Setup
    set_logging_format()
    set_seed(0)

    debug = (args.debug == 1)
    if debug:
        os.system(f'rm -rf {args.debug_dir}/* && mkdir -p {args.debug_dir}/track_vis {args.debug_dir}/ob_in_cam')

    # Initialize estimators
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    # Create output dir
    os.makedirs(args.output_dirpath, exist_ok=True)

    # Load scale mappings
    if not os.path.exists(args.obj_scale_mapping):
        logging.error(f"Could not find scale mapping file: {args.obj_scale_mapping}")
        return
    
    scales = {}
    with open(args.obj_scale_mapping, 'r') as f:
        for line in f:
            obj_idx_str, scale_str = line.strip().split(':')
            scales[int(obj_idx_str)] = float(scale_str)

    rgb_filename = os.path.join(args.segmentation_dirpath, "scene_full_image.png")
    depth_filename = os.path.join(args.segmentation_dirpath, "scene_full_depth.png")
    camk_filepath = os.path.join(args.segmentation_dirpath, "cam_K.txt")

    if not os.path.exists(rgb_filename):
        logging.error(f"Could not find scene image: {rgb_filename}")
        return

    # Register each mesh
    mesh_files = sorted(glob(os.path.join(args.imesh_outputs, args.instant_mesh_model, 'meshes/*_rgba.obj')))
    for mesh_file in mesh_files:
        filename = os.path.basename(mesh_file)
        obj_index = int(filename.split('_')[0])
        scale_factor = scales.get(obj_index, 1.0)

        color_bgr = cv2.imread(rgb_filename)
        if color_bgr is None:
            logging.error(f"Could not read scene image: {rgb_filename}")
            continue
        H, W = color_bgr.shape[:2]

        mask_filename = os.path.join(args.segmentation_dirpath, f"{obj_index}_mask.png")
        mask = load_mask(mask_filename, W, H).astype(bool)

        color = load_color(rgb_filename, W, H)
        depth = load_depth(depth_filename, W, H, zfar=np.inf)
        K = load_camera_intrinsics(camk_filepath, downscale=1.0)

        mesh = trimesh.load(mesh_file)
        mesh_vertices = mesh.vertices
        mesh_vertex_normals = mesh.vertex_normals

        est = FoundationPose(
            model_pts=mesh_vertices,
            model_normals=mesh_vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=args.debug_dir,
            debug=debug,
            glctx=glctx
        )
        logging.info("Estimator initialization done.")

        # Get the pose
        init_pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

        # Apply the scale and transformation to the mesh
        out_obj_path = os.path.join(args.output_dirpath, f"{obj_index}.obj")
        transform_and_export_mesh(
            mesh_file,
            scale_factor=scale_factor,
            transform_pose=init_pose,
            out_obj_path=out_obj_path,
            debug=debug
        )

        # Rename MTL & texture files
        #    The original code expects "material_0.png" -> "{obj_index}_texture.png"
        #    and "material.mtl" -> "{obj_index}_material.mtl"
        rename_mtl_and_texture(
            out_obj_path,
            old_mtl_name="material.mtl",
            old_tex_name="material_0.png",
            new_mtl_prefix=f"{obj_index}_",
            new_tex_prefix=f"{obj_index}_",
            debug=debug
        )

        logging.info(f"Registered mesh saved for object {obj_index}")

    # Save the partial scene pointcloud
    xyz_map = depth_to_xyz_map(depth, K)
    valid = (depth >= 0.001)
    points_3d = xyz_map[valid]    # shape (N, 3)
    colors_3d = (color[valid] / 255.0).astype(np.float32)

    pcd = numpy_to_open3d_cloud(points_3d, colors_3d)
    out_ply_path = os.path.join(args.output_dirpath, f"{obj_index}_scene_complete.ply")
    o3d.io.write_point_cloud(out_ply_path, pcd)

    logging.info(f"Scene pointcloud saved to {out_ply_path}")

if __name__ == "__main__":
    main()
