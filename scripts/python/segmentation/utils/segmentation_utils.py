import os
import numpy as np
from PIL import Image
import cv2
from copy import deepcopy
import re

class Segmentation:
    """
    A simple container for storing a segment’s prompt, bounding box, mask, and confidence score.
    """
    def __init__(self, prompt, box, mask, score):
        self.prompt = prompt
        self.box = box
        self.mask = mask
        self.score = score

def score_from_phrase(phrase):
    """
    Extracts numeric score from a string (e.g., 'some phrase(0.78)').
    Returns None if no floating number is found inside parentheses.
    """
    match = re.search(r'\(([\d.]+)\)', phrase)
    if match:
        return float(match.group(1))
    return None

def calculate_iou(mask1, mask2):
    """
    Computes IoU (Intersection over Union) between two binary masks (numpy arrays).
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0.0
    return np.sum(intersection) / np.sum(union)

def non_maximum_suppression(segmentations_list, threshold=0.7):
    """
    Performs simple mask-based NMS. Sorts segmentations by score, then
    removes any segmentation that has IoU > threshold with a higher-scored segmentation.
    """
    # Sort by descending confidence
    segs_sorted = sorted(segmentations_list, key=lambda x: x.score, reverse=True)
    
    for i in range(len(segs_sorted)):
        if segs_sorted[i] is None:
            continue
        for j in range(i+1, len(segs_sorted)):
            if segs_sorted[j] is None:
                continue
            iou_val = calculate_iou(segs_sorted[i].mask, segs_sorted[j].mask)
            if iou_val > threshold:
                # Remove the lower-scored segmentation
                segs_sorted[j] = None

    # Filter out any None entries
    segs_filtered = [s for s in segs_sorted if s is not None]
    return segs_filtered

def enlarge_coordinates(coordinates, ratio):
    """
    Enlarges (x1, y1, x2, y2) by a given ratio around its center.
    """
    x1, y1, x2, y2 = coordinates
    width, height = x2 - x1, y2 - y1
    x1 -= int(width * ratio)
    y1 -= int(height * ratio)
    x2 += int(width * ratio)
    y2 += int(height * ratio)
    return x1, y1, x2, y2

def resize_foreground(image, masked_image, ratio):
    """
    Resizes the cropped foreground (with alpha channel) and the corresponding inpainting mask.
    """
    DEBUG = False
    image = np.array(image)
    masked_image = np.array(masked_image)
    assert image.shape[-1] == 4

    alpha = np.where(image[..., 3] > 0)
    y1, y2 = alpha[0].min(), alpha[0].max()
    x1, x2 = alpha[1].min(), alpha[1].max()

    # Crop the foreground
    fg = image[y1:y2, x1:x2]
    fg_mask = masked_image[y1:y2, x1:x2]

    size = max(fg.shape[0], fg.shape[1])

    # Pad to square
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0

    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    new_mask = np.pad(
        fg_mask,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    new_size = int(new_image.shape[0] / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0

    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    new_mask = np.pad(
        new_mask,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    new_image = Image.fromarray(new_image)
    new_mask = Image.fromarray(new_mask)
    return new_image, new_mask

def resize_foreground_padded(image, masked_image, ratio):
    """
    Alternative version that tries to avoid certain cropping/padding artifacts.
    """
    DEBUG = False
    image = np.array(image)
    masked_image = np.array(masked_image)
    assert image.shape[-1] == 4

    alpha = np.where(image[..., 3] > 0)
    y1, y2 = alpha[0].min(), alpha[0].max()
    x1, x2 = alpha[1].min(), alpha[1].max()

    fg = image[y1:y2, x1:x2]
    size = max(fg.shape[0], fg.shape[1])

    ph0_1, pw0_1 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1_1, pw1_1 = size - fg.shape[0] - ph0_1, size - fg.shape[1] - pw0_1

    new_image = np.pad(
        fg,
        ((ph0_1, ph1_1), (pw0_1, pw1_1), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    new_size = int(new_image.shape[0] / ratio)
    ph0_2, pw0_2 = (new_size - size) // 2, (new_size - size) // 2
    ph1_2, pw1_2 = new_size - size - ph0_2, new_size - size - pw0_2

    new_image = np.pad(
        new_image,
        ((ph0_2, ph1_2), (pw0_2, pw1_2), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Recompute new_mask from original masked_image
    new_mask = masked_image[y1 - ph0_1 - ph0_2 : y2 + ph1_1 + ph1_2,
                            x1 - pw0_1 - pw0_2 : x2 + pw1_1 + pw1_2]

    new_image = Image.fromarray(new_image)
    new_mask = Image.fromarray(new_mask)
    return new_image, new_mask

def get_inpainting_input(
    rgb_filepath,
    segmentations_list_filtered,
    save_dirpath,
    resize_ratio,
    enlargement=0,
    depth=None
):
    """
    Creates and saves various images needed for inpainting or 3D tasks:
      - Masked object
      - Masked depth
      - RGBA object (with alpha = mask)
      - Inpainting mask
      - Cropped versions, etc.

    Returns a list of lines mapping prompt -> filepaths (for saving in a text file).
    """
    file_output = []
    image = cv2.cvtColor(cv2.imread(rgb_filepath), cv2.COLOR_BGR2RGB)

    for index, segmentation in enumerate(segmentations_list_filtered):
        coordinates = [int(segmentation.box[i]) for i in range(len(segmentation.box))]
        mask_np = segmentation.mask[:, :, np.newaxis]

        # 1. Save the segmented (masked) RGB
        masked_image = Image.fromarray(image * mask_np)
        segmented_object_filepath = os.path.join(save_dirpath, f"{index}_segmented_object.png")
        masked_image.save(segmented_object_filepath)

        # 2. Save the segmented depth
        if depth is not None:
            segmented_depth_array = np.where(segmentation.mask, depth, 0)
            segmented_depth = Image.fromarray(segmented_depth_array)
            segmented_depth_filepath = os.path.join(save_dirpath, f"{index}_segmented_depth.png")
            segmented_depth.save(segmented_depth_filepath)

        # 3. Create an RGBA image of the object
        mask_range = (segmentation.mask * 255).astype(np.uint8)
        alpha_channel = Image.fromarray(mask_range, mode='L')
        rgba_image = Image.fromarray(image).convert("RGBA")
        rgba_image.putalpha(alpha_channel)

        # 4. Create the inpainting mask
        #    (any other segmentation's region gets white)
        inpainting_array = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        for j, seg in enumerate(segmentations_list_filtered):
            if j == index:
                continue
            other_mask = seg.mask[:, :, np.newaxis]
            inpainting_mask = np.squeeze(other_mask, axis=2)
            inpainting_array[inpainting_mask] = [255, 255, 255]

        inpainting_masked_image = Image.fromarray(inpainting_array)

        # 5. Resize the RGBA object + inpainting mask
        rgba_image_resized, inpainting_masked_image_resized = resize_foreground(
            rgba_image, inpainting_masked_image, ratio=resize_ratio
        )
        # Alternatively, you could use `resize_foreground_padded` if desired:
        # rgba_image_resized, inpainting_masked_image_resized = resize_foreground_padded(
        #     rgba_image, inpainting_masked_image, resize_ratio
        # )

        rgba_object_filepath = os.path.join(save_dirpath, f"{index}_rgba.png")
        rgba_image_resized.save(rgba_object_filepath)

        inpainting_mask_filepath = os.path.join(save_dirpath, f"{index}_inpainting_mask.png")
        inpainting_masked_image_resized.save(inpainting_mask_filepath)

        # 6. Optionally enlarge bounding box and crop
        enlarged_coords = enlarge_coordinates(coordinates, enlargement)
        object_crop = Image.fromarray(image).crop(enlarged_coords)
        object_crop_filepath = os.path.join(save_dirpath, f"{index}_object_crop.png")
        object_crop.save(object_crop_filepath)

        inpainting_mask_crop = inpainting_masked_image.crop(enlarged_coords)
        inpainting_mask_crop_filepath = os.path.join(save_dirpath, f"{index}_inpainting_mask_crop.png")
        inpainting_mask_crop.save(inpainting_mask_crop_filepath)

        # 7. Save the target mask (only the current segmentation’s region)
        target_mask_array = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        target_mask_array[segmentation.mask] = [255, 255, 255]
        target_mask_filepath = os.path.join(save_dirpath, f"{index}_object_mask.png")
        target_mask_image = Image.fromarray(target_mask_array)
        target_mask_image.save(target_mask_filepath)

        # Collect results for writing to prompt-mask mapping file
        file_output.append('\t'.join([
            segmentation.prompt,
            rgba_object_filepath,
            inpainting_mask_filepath,
            object_crop_filepath,
            inpainting_mask_crop_filepath
        ]))

    return file_output
