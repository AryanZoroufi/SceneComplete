#!/usr/bin/env bash
# scenecomplete/scripts/install_all.sh
#

#    bash scenecomplete/scripts/install_all.sh

# Initialize conda in the script
echo "Initializing conda"
source ~/miniconda3/bin/activate # replace with anaconda if you are using anaconda

# 1) Create & activate conda env
conda create -n scenecomplete python=3.9 -y
conda activate scenecomplete

# 2) Install the top-level "scenecomplete" package in editable mode
pip install -e .

# 3) GPU stuff
echo "Installing CUDA 11.8.0"
conda install cuda -c nvidia/label/cuda-11.8.0 -y

# 4) PyTorch & Torchvision (for cu118)
echo "Installing PyTorch and related dependencies"
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 5) GroundedSegmentAnything dependencies
echo "Installing GroundedSegmentAnything dependencies"
cd scenecomplete/modules/GroundedSegmentAnything
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
cd ..

# 6) BrushNet dependencies
echo "Installing BrushNet dependencies"
cd BrushNet
pip install -e .
pip install huggingface_hub==0.23.2 peft==0.12.0
cd ..

# 7) InstantMesh + additional inpainting + 3D dependencies
echo "Installing InstantMesh dependencies"
pip install rembg==2.0.56 pytorch-lightning==2.0 omegaconf==2.3.0 einops==0.7.0 xatlas trimesh==4.3.1
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install open3d==0.18.0


echo "[INFO] All packages installed successfully!"
echo "[INFO] You can now 'conda activate scenecomplete' to use your environment."