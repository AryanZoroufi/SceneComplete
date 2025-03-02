import argparse
import os
import cv2
import numpy as np
from PIL import Image
import yaml

import sys
sys.path.append('utils')
from utils.segmentation_processor import get_masks_and_boxes
from utils.segmentation_utils import (
    Segmentation,
    score_from_phrase,
    calculate_iou,
    non_maximum_suppression,
    enlarge_coordinates,
    resize_foreground,
    resize_foreground_padded,
    get_inpainting_input,
)

def load_config(config_path):
    """
    Loads DINO/SAM-related config from a YAML file.
    """
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data

def get_segmentations(
    filename,
    prompts,
    config
):
    """
    For each prompt in `prompts`, calls GroundingDINO + SAM to get bounding boxes, masks,
    and predicted phrases. Returns a numpy array of Segmentation objects.
    """
    segmentations_list = np.array([])

    # Unpack GroundingDINO + SAM config
    gdino_cfg  = config['groundingdino']
    sam_cfg    = config['sam']
    device     = config.get('device', 'cuda')  # fallback 'cuda' if not in YAML

    for prompt in prompts:
        boxes, masks, pred_phrases = get_masks_and_boxes(
            image_path=filename,
            text_prompt=prompt,
            config_file=gdino_cfg['config_file'],
            grounded_checkpoint=gdino_cfg['checkpoint'],
            sam_version=sam_cfg['version'],
            sam_checkpoint=sam_cfg['checkpoint'],
            device=device,
            box_threshold=gdino_cfg.get('box_threshold', 0.3),
            text_threshold=gdino_cfg.get('text_threshold', 0.25),
            output_dir=None  # or specify if you want to save intermediate results
        )

        # If no detections, skip
        if boxes is None or masks is None or pred_phrases is None or len(boxes) == 0:
            continue

        # Loop over each detection
        for (box, mask, pred_phrase) in zip(boxes, masks, pred_phrases):
            scr = score_from_phrase(pred_phrase)
            # Convert Torch mask to numpy
            mask_np = mask.squeeze(0).detach().cpu().numpy()

            seg_obj = Segmentation(prompt, box, mask_np, scr)
            segmentations_list = np.append(segmentations_list, seg_obj)

    return segmentations_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment the objects in the image')

    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the RGB image')
    parser.add_argument('--depth_path', type=str, required=True,
                        help='Path to the depth image')
    parser.add_argument('--prompts_filepath', type=str, required=True,
                        help='Path to the file containing text prompts (one per line)')
    parser.add_argument('--prompt_mask_mapping_filepath', type=str, required=True,
                        help='Where to write the prompt-to-mask mapping output')
    parser.add_argument('--save_dirpath', type=str, required=True,
                        help='Directory to save output images/masks')
    
    # GroundingDINO + SAM configuration
    parser.add_argument('--config_path', type=str, default='utils/segment_config.yaml',
                        help='Path to YAML config file for DINO/SAM parameters')

    # NMS + resizing configuration
    parser.add_argument('--nms_threshold', type=float, default=0.7,
                        help='IOU threshold for non-maximum suppression')
    parser.add_argument('--resize_ratio', type=float, default=0.8,
                        help='Resize ratio for the foreground in inpainting')
    parser.add_argument('--enlargement', type=float, default=0.1,
                        help='Enlargement ratio for bounding boxes')

    args = parser.parse_args()

    # Load DINO + SAM configuration from YAML file
    config = load_config(args.config_path)

    os.makedirs(args.save_dirpath, exist_ok=True)

    # 0. Read prompts from file
    with open(args.prompts_filepath, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]

    # 1. Get segmentations from GroundingDINO + SAM
    segmentations_list = get_segmentations(
        filename=args.image_path,
        prompts=prompts,
        config=config,
    )
    print(f"Found {len(segmentations_list)} segmentations")

    # 2. Non-maximum suppression to remove overlapping objects with lower confidence
    segmentations_list_filtered = non_maximum_suppression(
        segmentations_list, threshold=args.nms_threshold
    )
    print(f"Found {len(segmentations_list_filtered)} segmentations after NMS")

    # 3. Load the depth
    depth = cv2.imread(args.depth_path, cv2.IMREAD_UNCHANGED)

    # 4. Generate inpainting inputs, RGBA objects, etc.
    inpaintings_dirpath = os.path.join(args.save_dirpath, 'sam_outputs')
    os.makedirs(inpaintings_dirpath, exist_ok=True)

    prompt_mask_mapping_lines = get_inpainting_input(
        rgb_filepath=args.image_path,
        segmentations_list_filtered=segmentations_list_filtered,
        save_dirpath=inpaintings_dirpath,
        resize_ratio=args.resize_ratio,
        enlargement=args.enlargement,
        depth=depth
    )

    # 5. Write the prompt-to-mask mapping info to file
    with open(args.prompt_mask_mapping_filepath, 'w') as f:
        f.write('\n'.join(prompt_mask_mapping_lines))