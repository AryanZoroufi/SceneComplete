import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# GroundingDINO imports
import scenecomplete.modules.GroundedSegmentAnything.GroundingDINO.groundingdino.datasets.transforms as T
from scenecomplete.modules.GroundedSegmentAnything.GroundingDINO.groundingdino.models import build_model
from scenecomplete.modules.GroundedSegmentAnything.GroundingDINO.groundingdino.util.slconfig import SLConfig
from scenecomplete.modules.GroundedSegmentAnything.GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# Segment Anything imports
from segment_anything import sam_model_registry, SamPredictor


def load_image(image_path):
    """
    Loads and preprocesses an image for GroundingDINO.
    """
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(config_file, model_checkpoint_path, device="cpu"):
    """
    Loads a GroundingDINO model from the specified config and checkpoint.
    """
    args = SLConfig.fromfile(config_file)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    _ = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(
    model,
    image,
    caption,
    box_threshold=0.3,
    text_threshold=0.25,
    device="cpu",
    with_logits=True
):
    """
    Given the loaded GroundingDINO model, the preprocessed image tensor,
    and a text prompt (caption), returns bounding boxes and phrases above thresholds.
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]              # (nq, 4)

    # Filter output by box threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # Get text phrases
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt.cpu(), pred_phrases


def show_mask(mask, ax, random_color=False):
    """
    Overlays a mask on the given matplotlib axis for visualization.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    """
    Draws a bounding box and label on the given matplotlib axis.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)
    )
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    """
    Example helper that saves a combined mask image and a JSON file of metadata.
    Not currently used in the main pipeline, but kept here if needed.
    """
    import json

    # Construct one mask image from all masks
    mask_img = torch.zeros(mask_list.shape[-2:])
    value = 0  # 0 for background
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'),
                bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    # Also save metadata in JSON
    json_data = [{"value": value, "label": "background"}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # remove trailing ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })

    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def get_masks_and_boxes(
    image_path,
    text_prompt,
    config_file,
    grounded_checkpoint,
    sam_version,
    sam_checkpoint,
    device="cuda",
    box_threshold=0.3,
    text_threshold=0.25,
    output_dir=None
):
    """
    Given an image path and a text prompt:
      1. Runs GroundingDINO to get bounding boxes + text phrases.
      2. Initializes the specified SAM model to get masks for those boxes.
      3. Returns (boxes, masks, predicted_phrases).
    """
    # Load and preprocess the image for GroundingDINO
    image_pil, image_tensor = load_image(image_path)

    # Load the GroundingDINO model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # Get bounding boxes and phrases from GroundingDINO
    boxes_filt, pred_phrases = get_grounding_output(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
        with_logits=True
    )

    # If no boxes found above threshold, return
    if boxes_filt is None or len(boxes_filt) == 0:
        return None, None, None

    # Initialize SAM
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    # Convert image to numpy for SAM
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # Convert boxes from cxcywh -> xyxy
    w, h = image_pil.size
    for i in range(len(boxes_filt)):
        cx, cy, bw, bh = boxes_filt[i]
        x1 = cx - bw/2
        y1 = cy - bh/2
        x2 = cx + bw/2
        y2 = cy + bh/2
        boxes_filt[i] = torch.tensor([x1*w, y1*h, x2*w, y2*h])

    # Transform boxes for SAM
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_filt, image_rgb.shape[:2]
    ).to(device)

    # Predict masks
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    # Optional: visualize or save the results if output_dir is given
    # (You can remove/modify these lines if you don’t need to save visualizations)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"),
                    bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close()

    return boxes_filt, masks, pred_phrases


def get_individual_mask(
    image_paths,
    text_prompts,
    config_file,
    grounded_checkpoint,
    sam_version,
    sam_checkpoint,
    device="cuda",
    box_threshold=0.3,
    text_threshold=0.25
):
    """
    Example function if you want to batch-process multiple image+prompt pairs.
    """
    model = load_model(config_file, grounded_checkpoint, device=device)
    prompt_sam_dict = {}

    for img_path, txt_prompt in zip(image_paths, text_prompts):
        image_pil, image_tensor = load_image(img_path)

        # Run GroundingDINO
        boxes_filt, pred_phrases = get_grounding_output(
            model=model,
            image=image_tensor,
            caption=txt_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
            with_logits=True
        )

        if boxes_filt is None or len(boxes_filt) == 0:
            prompt_sam_dict[txt_prompt] = [img_path, None, None, None]
            continue

        # Initialize SAM
        sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
        predictor = SamPredictor(sam)

        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        w, h = image_pil.size
        for i in range(len(boxes_filt)):
            cx, cy, bw, bh = boxes_filt[i]
            x1 = cx - bw/2
            y1 = cy - bh/2
            x2 = cx + bw/2
            y2 = cy + bh/2
            boxes_filt[i] = torch.tensor([x1*w, y1*h, x2*w, y2*h])

        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes_filt, image_rgb.shape[:2]
        ).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )

        prompt_sam_dict[txt_prompt] = [img_path, boxes_filt, masks, pred_phrases]

        # Optional: visualize each result if you want
        # ...

    return prompt_sam_dict
