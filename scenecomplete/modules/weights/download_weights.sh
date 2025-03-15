#!/bin/bash

echo "Downloading weights ..."

POSE_ESTIMATION_WEIGHTS_DIR="./pose_estimation_weights"
mkdir -p "$POSE_ESTIMATION_WEIGHTS_DIR"
# Add an __init__.py file to the pose_estimation_weights directory
touch "$POSE_ESTIMATION_WEIGHTS_DIR/__init__.py"

INPAINTING_WEIGHTS_DIR="./inpainting_weights"
BASE_MODEL_DIR=$INPAINTING_WEIGHTS_DIR/"realisticVisionV60B1_v51VAE"
BRUSHNET_MODEL_DIR=$INPAINTING_WEIGHTS_DIR/"random_mask_brushnet_ckpt"
mkdir -p "$INPAINTING_WEIGHTS_DIR"
# Add an __init__.py file to the inpainting_weights directory
touch "$INPAINTING_WEIGHTS_DIR/__init__.py"
mkdir -p "$BASE_MODEL_DIR"
mkdir -p "$BRUSHNET_MODEL_DIR"

# Grounded-SAM Weights
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM weights..."
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
    echo "SAM weights already downloaded."
fi

if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    echo "Downloading GroundingDINO weights..."
    wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "GroundingDINO weights already downloaded."
fi

# Download BrushNet weights
echo "Downloading BrushNet weights..."
gdown --folder 1hCYIjeRGx3Zk9WZtQf0s3nDGfeiwqTsN -O "$BRUSHNET_MODEL_DIR"
gdown --folder 1dQeSFqpQg_NSFLhd3ChuSJCZ0zCSquh8 -O "$BASE_MODEL_DIR"

echo "Google restricts file downloads. If you encounter an error while downloading the weights, access the files from your browser using the links below:"
echo "https://drive.google.com/drive/folders/1hCYIjeRGx3Zk9WZtQf0s3nDGfeiwqTsN and name the folder as 'random_mask_brushnet_ckpt'"
echo "https://drive.google.com/drive/folders/1dQeSFqpQg_NSFLhd3ChuSJCZ0zCSquh8 and name the folder as 'realisticVisionV60B1_v51VAE'"

# Download FoundationPose Weights
echo "Downloading FoundationPose weights..."
gdown --folder 1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O "$POSE_ESTIMATION_WEIGHTS_DIR"

echo "Google restricts file downloads. If you encounter an error while downloading the weights, access the files from your browser using the link below:"
echo "https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i and name the folder as 'pose_estimation_weights'"

echo "All weights downloaded successfully!"