import argparse
import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image

from scenecomplete.scripts.python.segmentation.utils.segmentation_processor import get_individual_mask
from scenecomplete.scripts.python.segmentation.utils.segmentation_utils import (
    load_config,
    score_from_phrase,
    resize_foreground_without_mask
)

def get_batch_individual_masks(
        image_paths,
        prompts,
        config
):
    """
    For each prompt in prompts, calls 'get_individual_mask' to process the inpainted image.
    """
    gdino_cfg = config['groundingdino']
    sam_cfg   = config['sam']
    device    = config.get('device', 'cuda')

    # Call get_individual_mask from segmentation_processor
    prompt_sam_dict = get_individual_mask(
        image_paths=image_paths,
        text_prompts=prompts,
        config_file=gdino_cfg['config_file'],
        grounded_checkpoint=gdino_cfg['checkpoint'],
        sam_version=sam_cfg['version'],
        sam_checkpoint=sam_cfg['checkpoint'],
        device=device,
        box_threshold=gdino_cfg.get('box_threshold', 0.3),
        text_threshold=gdino_cfg.get('text_threshold', 0.25)
    )

    return prompt_sam_dict

def main():
    parser = argparse.ArgumentParser(description="Post-inpainting segmentation using GroundingDINO+SAM.")
    parser.add_argument('--input_dirpath', type=str, required=True,
                        help='Path to the directory containing inpainted images.')
    parser.add_argument('--prompt_mask_mapping_filepath', type=str, required=True,
                        help='Path to the prompt-to-mask mapping file produced by the prior step.')
    parser.add_argument('--save_dirpath', type=str, required=True,
                        help='Directory to save the new RGBA images after re-segmentation.')
    parser.add_argument('--resize_ratio', type=float, default=0.85,
                        help='Ratio for resizing the foreground RGBA image.')

    # GroundingDINO + SAM configuration
    parser.add_argument('--config_path', type=str, default='utils/segment_config.yaml',
                        help='Path to YAML config file for DINO/SAM parameters')

    args = parser.parse_args()

    # Load DINO + SAM configuration from YAML file
    config = load_config(args.config_path)

    os.makedirs(args.save_dirpath, exist_ok=True)

    # 1) Read the lines from prompt_mask_mapping_filepath
    prompts = []
    rgb_files = []
    with open(args.prompt_mask_mapping_filepath, 'r') as f:
        lines = [line.strip() for line in f]
        for line in lines:
            prompt, rgba_filename, _, _, _ = line.split('\t')
            prompts.append(prompt)

            # Convert something like "0_rgba.png" into "0_inpainting.png" inside input_dirpath
            extension = osp.splitext(rgba_filename)[1]
            base_name = osp.basename(rgba_filename).split('.')[0]
            inpainting_name = base_name + '_inpainting' + extension
            rgb_file = osp.join(args.input_dirpath, inpainting_name)
            rgb_files.append(rgb_file)

    # 2) Get segmentations on the inpainted images via the get_batch_individual_masks function
    print("[INFO] Calling get_batch_individual_masks to get post-inpainting masks...")
    prompt_sam_dict = get_batch_individual_masks(
        image_paths=rgb_files,
        prompts=prompts,
        config=config
    )

    # 3) For each prompt, pick the highest-scoring segmentation, build RGBA, resize, and save
    for prompt, (image_filepath, boxes_filt, masks, pred_phrases) in prompt_sam_dict.items():
        if boxes_filt is None or masks is None or pred_phrases is None:
            print(f"[WARNING] No segmentation found for prompt='{prompt}'. Skipping.")
            continue

        # Extract highest scoring mask
        scores = [score_from_phrase(pp) for pp in pred_phrases]
        max_idx = np.argmax(scores)
        chosen_mask = masks[max_idx].squeeze(0).detach().cpu().numpy()

        # Load the inpainted image as RGB
        image_bgr = cv2.imread(image_filepath)
        if image_bgr is None:
            print(f"[WARNING] Could not read inpainted image: {image_filepath}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Convert that to RGBA
        rgba_image = Image.fromarray(image_rgb).convert("RGBA")

        # Create alpha from the chosen mask
        mask_range = (chosen_mask * 255).astype(np.uint8)
        alpha_channel = Image.fromarray(mask_range, mode='L')
        rgba_image.putalpha(alpha_channel)

        # Resize the RGBA image (foreground only) with the specified ratio
        resized_rgba = resize_foreground_without_mask(rgba_image, args.resize_ratio)

        # Save the resulting RGBA
        out_name = osp.basename(image_filepath)  # e.g. 0_inpainting.png
        out_path = osp.join(args.save_dirpath, out_name)
        resized_rgba.save(out_path)
        print(f"[INFO] Saved post-inpainting RGBA for prompt='{prompt}' to: {out_path}")


if __name__ == '__main__':
    main()