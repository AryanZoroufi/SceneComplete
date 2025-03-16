"""
prepare_3d_inputs.py

Preprocessing code to prepare inputs for image-to-3D reconstruction.
"""

import os
import os.path as osp
import numpy as np
import json
import shutil
import cv2
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="Prepare inputs for image-to-3D reconstruction.")
    parser.add_argument('--segmentation_dirpath', type=str, required=True,
                        help="Directory containing segmented object files (RGB, depth, mask).")
    parser.add_argument('--inpainting_dirpath', type=str, required=True,
                        help="Directory containing inpainted RGBA files (e.g., <idx>_rgba_inpainting.png).")
    parser.add_argument('--out_path', type=str, default='grasp_data',
                        help="Output directory where the final 3D input files will be written.")
    parser.add_argument('--scene_rgb_filepath', type=str, required=True,
                        help="Path to the full-scene RGB image.")
    parser.add_argument('--scene_depth_filepath', type=str, required=True,
                        help="Path to the full-scene depth image.")
    parser.add_argument('--intrinsics_path', type=str, required=True,
                        help="Path to the .txt file containing the camera intrinsics.")

    args = parser.parse_args()

    # Create the main output directory and a subdirectory for the RGBA inpainting files
    os.makedirs(args.out_path, exist_ok=True)
    imesh_input_dir = osp.join(args.out_path, 'imesh_inputs')
    os.makedirs(imesh_input_dir, exist_ok=True)

    # 1) Load intrinsics and write to JSON + TXT
    intrinsics = np.loadtxt(args.intrinsics_path)
    scene_img = cv2.imread(args.scene_rgb_filepath)
    if scene_img is None:
        raise ValueError(f"Could not read scene RGB image: {args.scene_rgb_filepath}")

    height, width = scene_img.shape[:2]
    intrinsics_data = {
        'width': width,
        'height': height,
        'intrinsic_matrix': intrinsics.T.reshape(-1).tolist()  # Flatten row-major after transpose
    }

    cam_json_path = osp.join(args.out_path, 'cam_K.json')
    with open(cam_json_path, 'w') as f:
        json.dump(intrinsics_data, f, indent=4)
    print(f"[INFO] Saved intrinsics JSON to: {cam_json_path}")

    # Also write out a plain-text matrix for convenience
    cam_txt_path = osp.join(args.out_path, "cam_K.txt")
    K_matrix = np.array(intrinsics_data["intrinsic_matrix"]).reshape(3, 3).T
    np.savetxt(cam_txt_path, K_matrix)
    print(f"[INFO] Saved intrinsics TXT to: {cam_txt_path}")

    # 2) Identify all object indices by scanning for *_segmented_object.png in segmentation_dirpath
    seg_obj_files = [f for f in os.listdir(args.segmentation_dirpath) if f.endswith('_segmented_object.png')]
    # Each filename is something like "<index>_segmented_object.png"
    indices = [int(osp.splitext(fn)[0].split('_')[0]) for fn in seg_obj_files]
    print(f"[INFO] Found object indices: {indices}")

    # 3) For each index, copy the corresponding files to out_path
    for idx in indices:
        rgb_path = osp.join(args.segmentation_dirpath, f'{idx}_segmented_object.png')
        depth_path = osp.join(args.segmentation_dirpath, f'{idx}_segmented_depth.png')
        mask_path = osp.join(args.segmentation_dirpath, f'{idx}_object_mask.png')
        inpainted_rgba_path = osp.join(args.inpainting_dirpath, f'{idx}_rgba_inpainting.png')

        masked_rgb_dest = osp.join(args.out_path, f"{idx}_masked.png")
        depth_dest = osp.join(args.out_path, f"{idx}_depth.png")
        mask_dest = osp.join(args.out_path, f"{idx}_mask.png")
        rgba_dest = osp.join(imesh_input_dir, f"{idx}_rgba.png")

        for src, dst in [
            (rgb_path,   masked_rgb_dest),
            (depth_path, depth_dest),
            (mask_path,  mask_dest),
            (inpainted_rgba_path, rgba_dest),
        ]:
            if not osp.isfile(src):
                print(f"[WARNING] Missing file: {src}")
                continue
            shutil.copy2(src, dst)
            print(f"[INFO] Copied {src} -> {dst}")

    # 4) Also copy the full scene RGB and depth
    scene_rgb_out = osp.join(args.out_path, 'scene_full_image.png')
    scene_depth_out = osp.join(args.out_path, 'scene_full_depth.png')

    shutil.copy2(args.scene_rgb_filepath, scene_rgb_out)
    shutil.copy2(args.scene_depth_filepath, scene_depth_out)
    print(f"[INFO] Copied full scene images to: {scene_rgb_out}, {scene_depth_out}")

    print("[INFO] Done preparing 3D inputs.")

if __name__ == '__main__':
    main()
