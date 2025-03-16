# SceneComplete

### [🌐 Project Website](https://scenecomplete.github.io) | [📝 Paper](https://arxiv.org/pdf/2410.23643v1) | [🎥 Video](https://www.youtube.com/watch?v=Tuzhn4HWiL0)

**SceneComplete** is an *open-world 3D scene completion system*, that constructs a complete, segmented, 3D model of a scene from a single RGB-D image. SceneComplete is a framework for intelligently composing multiple large pre-trained models – vision-language, segmentation, inpainting, image-to-3D, correspondence-based scaling, and 6D pose estimation – to generate high-quality, fully completed 3D object meshes, each registered in the global scene coordinate frame. 

Please read the official paper for a detailed overview of our work. 
> **SceneComplete: Open-World 3D Scene Completion in Complex Real-World Environments for Robot Manipulation**  
> Aditya Agarwal, Gaurav Singh, Bipasha Sen, Tomás Lozano-Pérez, Leslie Pack Kaelbling (2024)  
> [arXiv:2410.23643v2]()

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
   - [Downloading Pretrained Weights](#download-pretrained-weights)
   - [Setting up Environment Variables](#setup-environment-variables)
   - [Testing Individual Modules](#testing-individual-modules)
   - [Running SceneComplete](#running-scenecomplete)
   - [Visualizing Output](#visualizing-output)
- [Limitations & Contributing to SceneComplete](#limitations-and-contributing)
- [Citations](#citations)


## Installation
#### 1. Setup conda environment
```bash
# We recommend using conda to manage your environments
conda create -n scenecomplete python=3.9
conda activate scenecomplete
```

#### 2. Clone and install SceneComplete
```bash
git clone https://github.com/skymanaditya1/SceneComplete.git
cd SceneComplete
git submodule update --init --recursive
```

#### 3. Install submodule dependencies
We provide a script to download and setup submodule dependencies automatically
```bash
pip install -e .
bash scenecomplete/scripts/install_all.sh
```

### 4. Install FoundationPose dependencies
```bash
# Create foundationpose conda environment
conda create -n foundationpose python=3.9
conda activate foundationpose

# Install cuda-toolkit in your conda environment
conda install cuda -c nvidia/label/cuda-11.8.0
# Install torch for CUDA 11.8
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Install GCC 11 (to the conda env)
conda install -c conda-forge gcc=11 gxx=11

# Add conda lib path to your LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/<username>/miniconda3/envs/foundationpose/lib:$LD_LIBRARY_PATH

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda" (e.g., "/home/<username>/miniconda3/envs/foundationpose/include/eigen3")

# Install dependencies
cd scenecomplete/modues/FoundationPose
python -m pip install -r requirements.txt

# Install NVDiffRast, Kaolin, and PyTorch3D
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions, ensure that your CONDA_PREFIX points to your miniconda3 setup (e.g., /home/<username>/miniconda3)
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

# Finally, install SceneComplete as a python package
cd ../../..
pip install -e .
```

## Usage
### Downloading Pretrained Weights
We provide a script to download the pretrained weights of individual submodules. 

```bash
cd scenecomplete/modules/weights
bash download_weights.sh
```

This will automatically download and place the checkpoints in their respective directories. Google restricts large file downloads via scripts. If you encounter issues while downloading pretrained checkpoints, follow the steps in the `download_weights.sh` file. Huggingface weights required by the individual modules are downloaded automatically when the code is run for the first time. 

We finetune BrushNet using LoRA to adapt its performance on tabletop objects. We anticipate releasing the LoRA weights in the next few days. In the meantime, we use the pretrained BrushNet model for inpainting. 

### Setting up Environment Variables
```bash
# Set your datasets path
export scdirpath="<your datasets path>"

# Create your OpenAI API Key (https://platform.openai.com/api-keys) and add the secret as an environment variable.
export OPENAI_API_KEY="<your key>"
```

### Testing Individual Modules
We provide examples to test each module individually. However, since each module depends on the output of the previous step, they must be executed in sequence.

#### Prompting
```bash
python scenecomplete/scripts/python/prompting/generate_scene_prompts.py \
   --image_path $scdirpath/rgb.png \
   --output_filepath $scdirpath/prompts.txt
```

#### Segmentation
```bash
python scenecomplete/scripts/python/segmentation/segment_objects.py \
   --image_path $scdirpath/rgb.png \
   --depth_path $scdirpath/depth.png \
   --prompts_filepath $scdirpath/prompts.txt \
   --prompt_mask_mapping_filepath $scdirpath/prompt_mask_mapping.txt \
   --save_dirpath $scdirpath/sam_outputs
```

#### Inpainting
```bash
python scenecomplete/scripts/python/inpainting/inpaint_objects.py \
   --seed 42 \
   --prompt_filepath $scdirpath/prompt_mask_mapping.txt \
   --output_dirpath $scdirpath/inpainting_outputs \
   --use_pretrained \
   --blended
```

#### Segmentation post Inpainting
```bash
python scenecomplete/scripts/python/segmentation/segment_objects_post_inpainting.py \
   --input_dirpath $scdirpath/inpainting_outputs \
   --prompt_mask_mapping_filepath $scdirpath/prompt_mask_mapping.txt \
   --save_dirpath $scdirpath/sam_post_processed
```

#### Preparing 3D inputs
```bash
python scenecomplete/scripts/python/reconstruction/utils/prepare_3d_inputs.py \
   --segmentation_dirpath $scdirpath/sam_outputs \
   --inpainting_dirpath $scdirpath/sam_post_processed \
   --out_path $scdirpath/grasp_data \
   --scene_rgb_filepath $scdirpath/rgb.png \
   --scene_depth_filepath $scdirpath/depth.png \
   --intrinsics_path $scdirpath/cam_K.txt
```

#### Reconstruction
```bash
python scenecomplete/scripts/python/reconstruction/generate_3d_mesh.py \
   $scdirpath/grasp_data/imesh_inputs \
   --config instant-mesh-base.yaml \
   --output_path $scdirpath/imesh_outputs \
   --seed 42 \
   --no_rembg \
   --export_texmap
```

#### Scaling
```bash
python scenecomplete/scripts/python/scaling/compute_mesh_scaling.py \
   --segmentation_dirpath $scdirpath/grasp_data \
   --imesh_outputs $scdirpath/imesh_outputs \
   --output_filepath $scdirpath/obj_scale_mapping.txt \
   --instant_mesh_model instant-mesh-base
   --camera_name realsense
```

#### Registration
```bash
python scenecomplete/scripts/python/registration/register_mesh.py \
   --imesh_outputs $scdirpath/imesh_outputs \
   --segmentation_dirpath $scdirpath/
   --obj_scale_mapping $scdirpath/obj_scale_mapping.txt \
   --output_dirpath $scdirpath/fpose_outputs
```

### Running SceneComplete
```bash
bash scenecomplete/scripts/bash/scenecomplete.sh 
```

### Visualizing Output
```

```

## Limitations & Contributing to SceneComplete

We encourage swapping modules for better performance:

Replace BrushNet with another inpainting approach.
Switch out InstantMesh for Zero123 or a new 2D→3D model.
Adjust pose registration if you have a robust alternative.
Better alternative for matching correspondences between images. 
PRs are Welcome: If you improve or replace modules with stronger versions, please open a pull request – we’d love to incorporate better approaches or domain adaptations.

Since SceneComplete provides an intelligent way of composing different models, this allows for flexibility in swapping different models or upgraded versions of existing models for better performance, as long as the input/output contract between different models is satisfied. 

## Citation
```
@inproceedings{agarwal2024scenecomplete,
  title={{SceneComplete}: Open-World 3D Scene Completion in Complex Real-World Environments for Robot Manipulation},
  author={Agarwal, Aditya and Singh, Gaurav and Sen, Bipasha and Lozano-P{\'e}rez, Tom{\'a}s and Kaelbling, Leslie Pack},
  year={2024},
  archivePrefix={arXiv},
  eprint={2410.23643v2}
}
```