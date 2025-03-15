"""
inpaint_objects.py

Main script to perform BrushNet-based inpainting on images.
"""

import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser

import torch
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
from peft import PeftModel

import importlib.resources as pkg_resources
import scenecomplete.modules.weights.inpainting_weights as weights
from scenecomplete.scripts.python.inpainting.utils.inpainting_utils import (
    set_random_seed,
    generate_brush_stroke_mask
)


def main():
    parser = ArgumentParser(description="BrushNet-based inpainting with LoRA adaptation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--prompt_filepath", type=str, required=True,
                        help="Path to .txt file containing lines: prompt, image_path, mask_path, _, _")
    parser.add_argument("--output_dirpath", type=str, required=True,
                        help="Directory to save final results and inpainting debug output.")

    # Model / checkpoint arguments
    base_model_path = pkg_resources.files(weights) / 'realisticVisionV60B1_v51VAE'
    brushnet_model_path = pkg_resources.files(weights) / 'random_mask_brushnet_ckpt'
    # lora_model_path = pkg_resources.files(weights) / 'lora_ckpt'
    parser.add_argument("--base_model_path", type=str, required=True, default=base_model_path,
                        help="Path to the base SD model checkpoint used by BrushNet.")
    parser.add_argument("--brushnet_model_path", type=str, required=True, default=brushnet_model_path,
                        help="Path to the pretrained BrushNet model.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA finetuned checkpoint (if not using pretrained).")
    parser.add_argument("--use_pretrained", action="store_true",
                        help="If set, uses the pretrained BrushNet rather than a LoRA finetuned version.")

    # Additional options
    parser.add_argument("--blended", action="store_true",
                        help="If set, blends the inpainted result with a blurred mask boundary.")
    parser.add_argument("--debug", action="store_true",
                        help="If set, saves intermediate inpainting debug outputs (e.g. init, mask).")
    parser.add_argument("--brush_radius", type=int, default=10,
                        help="Radius of the brush stroke for generating the brushed inpainting mask.")
    args = parser.parse_args()

    # 1) SEEDING
    set_random_seed(args.seed)

    # 2) Prepare output directories
    if args.debug:
        inpaint_debug_dir = osp.join(args.output_dirpath, "inpainting_debug")
        print(f"[INFO] Creating inpainting intermediate outputs: {inpaint_debug_dir}")
        os.makedirs(inpaint_debug_dir, exist_ok=True)

    os.makedirs(args.output_dirpath, exist_ok=True)

    # 3) Log if the pretrained model or the LoRA-adapted model is being used 
    if args.use_pretrained:
        print(f"[INFO] Using pretrained BrushNet model: {args.brushnet_model_path}")
    else:
        print(f"[INFO] Using LoRA-finetuned BrushNet checkpoint: {args.lora_path}")

    # 4) Read the prompts file
    #    Each line has: "caption, image_path, mask_path, _, _"
    prompts = []
    with open(args.prompt_filepath, 'r') as f:
        for line in f:
            caption, img_path, msk_path, _, _ = line.strip().split('\t')
            prompts.append((caption, img_path, msk_path))

    # 5) Load the BrushNet model
    print("[INFO] Initializing BrushNet model...")
    brushnet = BrushNetModel.from_pretrained(args.brushnet_model_path, torch_dtype=torch.float16)

    # If not using pretrained, load the LoRA weights
    if not args.use_pretrained and args.lora_path is not None:
        brushnet = PeftModel.from_pretrained(brushnet, args.lora_path)
    else:
        print("[WARNING] No LoRA checkpoint provided. Using pretrained BrushNet model.")

    # Build the pipeline
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        args.base_model_path,
        brushnet=brushnet,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()  # in case of limited VRAM

    # 6) Inference loop
    generator = torch.Generator("cuda").manual_seed(args.seed)
    brushnet_conditioning_scale = 1.0

    for idx, (caption, image_path, mask_path) in enumerate(prompts):
        print(f"[INFO] Processing {image_path} with mask {mask_path}")

        # 1. Read RGBA image
        init_bgra = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if init_bgra is None:
            print(f"[WARNING] Could not read image: {image_path}; skipping.")
            continue

        # Convert BGRA -> RGBA
        init_image = cv2.cvtColor(init_bgra, cv2.COLOR_BGRA2RGBA)
        mask_image = init_bgra[:, :, 3:] / 255.0  # alpha channel in [0,1]

        # 2. Resize to 512 if smaller
        if init_image.shape[0] < 512 or init_image.shape[1] < 512:
            init_image = cv2.resize(init_image, (512, 512))
            mask_image = cv2.resize(mask_image, (512, 512))[..., np.newaxis]

        # 3. Read random mask, resize, brush
        random_mask_bgra = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if random_mask_bgra is None:
            print(f"[WARNING] Could not read random mask: {mask_path}; skipping.")
            continue

        random_mask = random_mask_bgra[:, :, :1] / 255.0  # single channel
        random_mask = cv2.resize(
            random_mask,
            (init_image.shape[1], init_image.shape[0])
        )[:, :, np.newaxis]

        if args.debug:
            print(f"[DEBUG] init_image shape={init_image.shape}, random_mask shape={random_mask.shape}")

        # 4. Apply brush strokes
        # generate_brush_stroke_mask must return shape [1, H, W]
        random_mask = generate_brush_stroke_mask(init_image.shape[0], init_image.shape[1], random_mask)[0]  # (H, W)

        # 5. Composite init_image with alpha => object remains, background=255
        init_image = init_image * mask_image + 255.0 * (1 - mask_image)

        # 6. Zero out region to inpaint => multiply by (1-random_mask)
        init_image = init_image * (1 - random_mask)

        # 7. Convert to PIL images
        mask_image = (random_mask).astype(np.uint8)
        init_image_pil = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
        mask_image_pil = Image.fromarray(
            (mask_image.repeat(3, -1) * 255)
        ).convert("RGB")

        # Save debug init & mask if required 
        if args.debug:
            if inpaint_debug_dir is not None:
                debug_mask_path = osp.join(inpaint_debug_dir, f"mask_{idx}.png")
                debug_init_path = osp.join(inpaint_debug_dir, f"init_{idx}.png")
                init_image_pil.save(debug_init_path)
                mask_image_pil.save(debug_mask_path)
                print(f"[DEBUG] Saved debug init & mask to {inpaint_debug_dir}")

        # 7. Run inpainting
        result_image = pipe(
            caption,
            init_image_pil,
            mask_image_pil,
            num_inference_steps=50,
            generator=generator,
            brushnet_conditioning_scale=brushnet_conditioning_scale
        ).images[0]

        # 8. Blend the results if required
        if args.blended:
            print("[INFO] Blending final inpainted result with original image.")

            # Convert the pipeline output (PIL image) to a NumPy array
            image_np = np.array(result_image)

            # Read the original image in BGRA format
            init_image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if init_image_np is None:
                print(f"[WARNING] Could not read original image for blending: {image_path}")
            else:
                # Convert BGRA -> RGBA -> normalized [0..1]
                init_image_np = cv2.cvtColor(init_image_np, cv2.COLOR_BGRA2RGBA) / 255.0

                # (Optional) Previously, you used the alpha channel to combine foreground vs. background
                # init_image_np = init_image_np[..., :3] * init_image_np[..., 3:] + (1 - init_image_np[..., 3:])
                # Then scale back to [0..255]
                init_image_np = init_image_np[..., :3] * init_image_np[..., 3:] + (1.0 - init_image_np[..., 3:])
                init_image_np *= 255.0

                # Retrieve the random mask from the existing variable
                mask_np = random_mask

                # Resize the original image to match the mask shape
                init_image_np = cv2.resize(init_image_np, (mask_np.shape[1], mask_np.shape[0]))

                # Where mask_np = 1, set background to 255; keep original color where mask=0
                init_image_np = init_image_np[..., :3] * (1.0 - mask_np) + 255.0 * mask_np

                # Apply a blur to the mask boundary for smoother blending (you can tune kernel size)
                mask_blurred = cv2.GaussianBlur((mask_np * 255).astype(np.uint8), (21, 21), 0) / 255.0
                mask_blurred = mask_blurred[:, :, np.newaxis]
                # Soft transition: mask_np = 1 - (1 - mask_np)*(1 - mask_blurred)
                mask_np = 1.0 - (1.0 - mask_np) * (1.0 - mask_blurred)

                # Resize the final pipeline result to match mask size
                image_np_resized = cv2.resize(image_np, (mask_np.shape[1], mask_np.shape[0]))

                # Debug info on image ranges
                if args.debug:
                    print(
                        "[DEBUG] init_image_np range: min={}, max={}; "
                        "image_np_resized range: min={}, max={}".format(
                            init_image_np.min(), init_image_np.max(),
                            image_np_resized.min(), image_np_resized.max()
                        )
                    )

                # Blend the original and the inpainted result
                image_pasted = init_image_np * (1.0 - mask_np) + image_np_resized * mask_np
                image_pasted = image_pasted.astype(image_np.dtype)
                image = Image.fromarray(image_pasted)


        # 9) Save final output
        extension = osp.splitext(image_path)[1]
        output_filepath = osp.join(
            args.output_dirpath,
            osp.basename(image_path).split('.')[0] + '_inpainting' + extension
        )
        image.save(output_filepath)
        print(f"[INFO] Inpainting result saved to: {output_filepath}")

if __name__ == "__main__":
    main()