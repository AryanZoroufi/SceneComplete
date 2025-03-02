"""
inpainting_utils.py

Utility functions for brushnet-based inpainting.
"""

import math
import numpy as np
import torch
from PIL import Image, ImageDraw
import cv2


def set_random_seed(seed: int):
    """
    Set random seeds for reproducibility across numpy, torch, and CUDA.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_brush_stroke(
    max_tries: int,
    h: int,
    w: int,
    min_num_vertex: int = 0,
    max_num_vertex: int = 8,
    mean_angle: float = 2 * math.pi / 5,
    angle_range: float = 2 * math.pi / 15,
    min_width: int = 128,
    max_width: int = 128
) -> np.ndarray:
    """
    Generates a random "brush stroke" mask using line/ellipse draws on a PIL image.
    Returns a binary mask of shape (H, W), where 1 = stroke, 0 = background.
    """
    average_radius = math.sqrt(h * h + w * w) / 8
    mask_pil = Image.new('L', (w, h), 0)

    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)

        # Vary the angle
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []

        # Build angle list
        for i in range(num_vertex):
            if i % 2 == 0:
                # big jump
                angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
            else:
                # small jump
                angles.append(np.random.uniform(angle_min, angle_max))

        # Generate the vertex list
        vertex = []
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0,
                2 * average_radius
            )
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        # Draw line
        draw = ImageDraw.Draw(mask_pil)
        stroke_width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=stroke_width)
        for v in vertex:
            draw.ellipse(
                (v[0] - stroke_width // 2,
                 v[1] - stroke_width // 2,
                 v[0] + stroke_width // 2,
                 v[1] + stroke_width // 2),
                fill=1
            )

        # Random flips
        if np.random.random() > 0.5:
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)

    mask = np.asarray(mask_pil, np.uint8)
    # Additional flips
    if np.random.random() > 0.5:
        mask = np.flip(mask, axis=0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, axis=1)

    return mask


def generate_random_mask(h: int, w: int) -> np.ndarray:
    """
    Creates a random mask of shape (1, H, W) by combining a random_brush_stroke.
    Returns a float32 array with 1 = keep, 0 = hole.
    """
    ones_mask = np.ones((h, w), dtype=np.uint8)
    stroke_mask = random_brush_stroke(4, h, w)
    combined = np.logical_and(ones_mask, 1 - stroke_mask)  # hole denoted as 0, reserved as 1
    return combined[np.newaxis, ...].astype(np.float32)


def brush_stroke_from_mask(h: int, w: int, mask: np.ndarray, radius: int) -> np.ndarray:
    """
    For each pixel that is set in `mask` (non-zero),
    create a circular brush stroke area around it with radius=10.
    """
    # coords_y, coords_x = np.where(mask > 0)
    coords = np.where(mask)
    coords_y, coords_x = coords[0], coords[1]
    for x0, y0 in zip(coords_x, coords_y):
        for x1 in range(x0 - radius, x0 + radius + 1):
            for y1 in range(y0 - radius, y0 + radius + 1):
                if (x1 - x0) ** 2 + (y1 - y0) ** 2 <= radius * radius:
                    if 0 <= y1 < h and 0 <= x1 < w:
                        mask[y1, x1] = 1
    return mask


def generate_brush_stroke_mask(h: int, w: int, mask: np.ndarray, radius: int = 10) -> np.ndarray:
    """
    Combines an existing binary `mask` with a brush stroke version of itself.
    Returns shape (1, H, W) in float32.
    """
    brushed = brush_stroke_from_mask(h, w, mask.copy(), radius)
    combined = np.logical_or(mask, brushed).astype(np.float32)
    return combined[np.newaxis, ...]