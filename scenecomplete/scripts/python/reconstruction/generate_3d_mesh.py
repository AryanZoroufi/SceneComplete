"""
generate_3d_mesh.py

Script runs InstantMesh-based image-to-3D reconstruction model. 
- Takes in input images.
- Optionally removes background (specified by `--no_rembg` flag).
- Uses a diffusion model for multi-view generation. 
- Reconstructs a 3D mesh using triplanes. 
- Exports texture map and or/saves video of the renders. 

Example usage:
    python generate_3d_mesh.py configs/instant-mesh-large.yaml ../imesh_inputs \
        --output_path ../outputs \
        --seed 42 \
        -- no_rembg \
        --export_texmap \
        ...
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from omegaconf import OmegaConf
from einops import rearrange
from importlib.resources import files
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
import rembg
import torch
from torchvision.transforms import v2
from pytorch_lightning import seed_everything

# Project utils
from scenecomplete.modules.InstantMesh.src.utils.train_util import instantiate_from_config
from scenecomplete.modules.InstantMesh.src.utils.camera_util import get_zero123plus_input_cameras
from scenecomplete.modules.InstantMesh.src.utils.mesh_util import save_obj, save_obj_with_mtl
from scenecomplete.modules.InstantMesh.src.utils.infer_util import remove_background, resize_foreground
from scenecomplete.scripts.python.reconstruction.utils.mesh_generation_utils import (
    get_render_cameras,
    render_rgb_frames,
    render_depth_frames
)

def initialize_configs(args):
    # Set random seed
    seed_everything(args.seed)

    # Load config
    config_dir = files('scenecomplete.modules.InstantMesh.configs')
    args.config = config_dir / args.config
    config = OmegaConf.load(args.config)
    config_name = os.path.basename(args.config).replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config

    # Determine if using InstantMesh
    IS_FLEXICUBES = config_name.startswith('instant-mesh')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return config, config_name, model_config, infer_config, IS_FLEXICUBES, device


def create_output_directories(base_output_path, config_name):
    output_dirs = {
        "image": os.path.join(base_output_path, config_name, 'images'),
        "input_image": os.path.join(base_output_path, config_name, 'input_images'),
        "triplane": os.path.join(base_output_path, config_name, 'triplanes'),
        "mesh": os.path.join(base_output_path, config_name, 'meshes'),
        "video": os.path.join(base_output_path, config_name, 'videos'),
    }

    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    return output_dirs

def main():
    parser = argparse.ArgumentParser(description="InstantMesh-based 3D reconstruction pipeline.")
    parser.add_argument('input_path', type=str, help='Path to the image directory.')
    parser.add_argument('--config', type=str, default='instant-mesh-large.yaml', help='Path to config file.')
    parser.add_argument('--output_path', type=str, default='imesh_outputs/', help='Output directory.')
    parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale of the generated object.')
    parser.add_argument('--distance', type=float, default=4.5, help='Camera distance for final video rendering.')
    parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views to consider.')
    parser.add_argument('--no_rembg', action='store_true', help='Skip background removal.')
    parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
    parser.add_argument('--vis_mesh', action='store_true', help='Visualize the final mesh (requires Open3D).')
    args = parser.parse_args()

    # Set random seed
    seed_everything(args.seed)

    # Load config
    _, config_name, model_config, infer_config, IS_FLEXICUBES, device = initialize_configs(args)

    # 1) Load diffusion pipeline
    print('[INFO] Loading diffusion model ...')
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline=str(files('scenecomplete.modules.InstantMesh.zero123plus') / 'pipeline.py'),
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )

    # 2) Load custom white-background UNet
    print('[INFO] Loading custom white-background unet ...')
    if os.path.exists(infer_config.unet_path):
        unet_ckpt_path = infer_config.unet_path
    else:
        unet_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="diffusion_pytorch_model.bin",
            repo_type="model"
        )
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)
    pipeline = pipeline.to(device)

    # 3) Load reconstruction model
    print('[INFO] Loading reconstruction model ...')
    model = instantiate_from_config(model_config)
    if os.path.exists(infer_config.model_path):
        model_ckpt_path = infer_config.model_path
    else:
        model_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename=f"{config_name.replace('-', '_')}.ckpt",
            repo_type="model"
        )
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model.eval()

    # 4) Make output directories
    output_dirs = create_output_directories(args.output_path, config_name)

    image_path = output_dirs["image"]
    input_image_path = output_dirs["input_image"]
    triplane_path = output_dirs["triplane"]
    mesh_path = output_dirs["mesh"]
    video_path = output_dirs["video"]

    # 5) Process input files
    if os.path.isdir(args.input_path):
        input_files = [
            os.path.join(args.input_path, f)
            for f in os.listdir(args.input_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
    else:
        input_files = [args.input_path]
    print(f'[INFO] Total number of input images: {len(input_files)}')

    rembg_session = None if args.no_rembg else rembg.new_session()

    # 6) Stage 1: Multi-view generation
    outputs = []
    for idx, image_file in enumerate(input_files):
        name = os.path.splitext(os.path.basename(image_file))[0]
        print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')

        # Remove background optionally
        input_image = Image.open(image_file).convert('RGBA')
        if not args.no_rembg:
            input_image = remove_background(input_image, rembg_session)
            input_image = resize_foreground(input_image, 0.85)

        # Save the (possibly new) input image
        input_image_savepath = os.path.join(input_image_path, f'{name}.png')
        input_image.save(input_image_savepath)
        print(f"[INFO] Input image saved to {input_image_savepath}")

        # Run diffusion pipeline
        output_image = pipeline(
            input_image,
            num_inference_steps=args.diffusion_steps,
        ).images[0]

        output_savepath = os.path.join(image_path, f'{name}.png')
        output_image.save(output_savepath)
        print(f"[INFO] Output image saved to {output_savepath}")

        # Convert to torch tensor
        images_np = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images_np).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

        outputs.append({'name': name, 'images': images})

    # Free up memory from pipeline
    del pipeline

    # 7) Stage 2: Reconstruction
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * args.scale).to(device)
    chunk_size = 20 if IS_FLEXICUBES else 1

    for idx, sample in enumerate(outputs):
        name = sample['name']
        images = sample['images'].unsqueeze(0).to(device)
        # Resize images to 320 x 320
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        if args.view == 4:
            indices = torch.tensor([0, 2, 4, 5]).long().to(device)
            images = images[:, indices]
            sub_cameras = input_cameras[:, indices]
        else:
            sub_cameras = input_cameras

        print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')
        with torch.no_grad():
            planes = model.forward_planes(images, sub_cameras)

            # save triplane
            npy_path = os.path.join(triplane_path, f'{name}.npy')
            np.save(npy_path, planes.cpu().numpy())
            print(f"[INFO] Triplane saved to {npy_path}")

            # extract mesh
            print(f"[INFO] Extracting mesh for {name} ...")
            mesh_obj_path = os.path.join(mesh_path, f'{name}.obj')
            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=args.export_texmap,
                **infer_config,
            )
            if args.export_texmap:
                verts, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                save_obj_with_mtl(
                    verts.data.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces.data.cpu().numpy(),
                    mesh_tex_idx.data.cpu().numpy(),
                    tex_map.permute(1, 2, 0).data.cpu().numpy(),
                    mesh_obj_path
                )
            else:
                verts, faces, vert_colors = mesh_out
                save_obj(verts, faces, vert_colors, mesh_obj_path)
            print(f"[INFO] Mesh saved to {mesh_obj_path}")

            # Optional mesh visualization
            if args.vis_mesh:
                import open3d as o3d
                mesh = o3d.io.read_triangle_mesh(mesh_obj_path)
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                o3d.visualization.draw_geometries([mesh, axis])

            # 8) Save rendered rgb and depth frames
            video_out_path = os.path.join(video_path, f'{name}.mp4')
            render_cameras, intrinsics = get_render_cameras(
                batch_size=1,
                num_views=1,  # or however many frames you want
                radius=args.distance,
                elevation=20.0,
                use_flexicubes=IS_FLEXICUBES,
                return_intrinsics=True,
                fov_degrees=30.0
            )
            render_cameras = render_cameras.to(device)
            render_res = 256
            intrinsics *= render_res
            intrinsics = intrinsics.reshape(3, 3).T.reshape(-1)
            intrinsics[-1] = 1.0

            # Save intrinsics JSON
            camK_json = {
                'width': render_res,
                'height': render_res,
                'intrinsic_matrix': intrinsics.cpu().numpy().tolist(),
            }
            json_path = video_out_path.replace('.mp4', '.json')
            with open(json_path, 'w') as f:
                json.dump(camK_json, f, indent=4)

            # Render frames
            frames = render_rgb_frames(
                model, planes,
                cameras=render_cameras,
                render_size=render_res,
                chunk_size=chunk_size,
                use_flexicubes=IS_FLEXICUBES
            )
            depths = render_depth_frames(
                model, planes,
                cameras=render_cameras,
                render_size=render_res,
                chunk_size=chunk_size,
                use_flexicubes=IS_FLEXICUBES
            )

            print(f"[DEBUG] frames shape: {frames.shape}, depths shape: {depths.shape}")

            # Save first frame as PNG
            first_frame = frames[0].cpu().numpy().transpose(1, 2, 0)
            first_frame_uint8 = (first_frame * 255).astype(np.uint8)
            Image.fromarray(first_frame_uint8).save(video_out_path.replace('.mp4', '.png'))

            plt.clf()
            first_depth = depths[0].cpu().numpy().transpose(1, 2, 0)[..., 0]
            first_depth_grad = np.gradient(first_depth)
            first_depth_grad = np.sqrt(first_depth_grad[0]**2 + first_depth_grad[1]**2)
            first_depth_grad = (first_depth_grad * 255 / first_depth_grad.max()).astype(np.uint8)
            mask1 = first_depth_grad > 128
            first_depth_grad[mask1] = 255

            Image.fromarray(first_depth_grad).save(video_out_path.replace('.mp4', '_depth_grad.png'))

            first_depth = (first_depth * 1000).astype(np.uint16)
            bg_val = first_depth[0, 0]
            first_depth[first_depth == bg_val] = 0
            first_depth[mask1] = 0
            plt.imshow(first_depth)
            plt.colorbar()
            plt.savefig(video_out_path.replace('.mp4', '_depth_plt.png'))
            Image.fromarray(first_depth).save(video_out_path.replace('.mp4', '_depth.png'))

    print("[INFO] Done!")

if __name__ == "__main__":
    main()
