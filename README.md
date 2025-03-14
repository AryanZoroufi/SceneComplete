# SceneComplete

**SceneComplete** is an *open-world 3D scene completion system*, that constructs a complete, segmented, 3D model of a scene from a single RGB-D image. SceneComplete provides a framework for intelligently composing multiple large pre-trained models – vision-language, segmentation, inpainting, image-to-3D, correspondence-based scaling, and 6D pose estimation – to generate high-quality, fully completed 3D object meshes, each registered in the global scene coordinate frame. 

Please read the official paper for a detailed overview of our work. 
> **SceneComplete: Open-World 3D Scene Completion in Complex Real-World Environments for Robot Manipulation**  
> Aditya Agarwal, Gaurav Singh, Bipasha Sen, Tomás Lozano-Pérez, Leslie Pack Kaelbling (2024)  
> [arXiv:2410.23643v2]()

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Setup & Installation](#setup--installation)
4. [Modules & Pipelines](#modules--pipelines)
   - [1. Vision-Language Model (VLM)](#1-vision-language-model-vlm)
   - [2. Segmentation](#2-segmentation)
   - [3. 2D Image Inpainting](#3-2d-image-inpainting)
   - [4. Image-to-3D](#4-image-to-3d)
   - [5. DINO Dense Correspondence for Scale Estimation](#5-dino-dense-correspondence-for-scale-estimation)
   - [6. Pose Registration (6D)](#6-pose-registration-6d)
5. [Submodules & Model Weights](#submodules--model-weights)
6. [OpenAI API Key for VLM](#openai-api-key-for-vlm)
7. [Sample Usage](#sample-usage)
8. [Updating Modules & Contributing](#updating-modules--contributing)
9. [Limitations](#limitations)
10. [Citations](#citations)

---

## Overview

SceneComplete addresses the following steps to produce a **high-fidelity** scene reconstruction from **one** RGB-D view:

1. **VLM-based Labeling**  
   - Uses a large Vision-Language Model (e.g., ChatGPT + image prompting) to identify objects and generate short text labels like “blue bowl,” “banana,” “grey mug,” etc.

2. **Segmentation**  
   - For each text label, a grounded-segmentation model (e.g., Grounded-DINO + SAM) extracts object masks from the input image.

3. **Inpainting**  
   - Each object mask is individually inpainted using a 2D diffusion/BrushNet pipeline to fill in occluded regions.

4. **Image-to-3D**  
   - The inpainted 2D object images are passed to a model (like InstantMesh or Zero123-like) to produce a **completed** 3D mesh for each object.

5. **Scale Estimation using DINO**  
   - The system uses **dense correspondences** (via dino-vit-features) to match the partial real scene points to the synthetic object model, solving for a uniform scale factor.

6. **Pose Registration**  
   - A final 6D pose for each object is found via a pre-trained predictor (e.g., FoundationPose), aligning each scaled mesh with the real partial scan in world coordinates.

**Result**: A fully-segmented, completed 3D mesh for each object in the scene, ready for downstream manipulation, planning, or collision avoidance.

**Reference**: For details, see the accompanying paper “SceneComplete: Open-World 3D Scene Completion in Complex Real-World Environments for Robot Manipulation”  :contentReference[oaicite:1]{index=1}.

---

## Repository Structure


## Setup & Installation

1. **Clone & Update Submodules**  
   ```bash
   git clone https://github.com/YourUsername/SceneComplete.git
   cd SceneComplete
   git submodule update --init --recursive
   pip install -e .

2. **Download Pretrained Weights**

3. **Setup OpenAI API Key for VLM**

## Modules & Pipelines


## Sample Usage
'''bash
python scenecomplete/segmentation/segment_objects.py \
    --image_path data/samples/scene_full_image.png \
    --depth_path data/samples/scene_full_depth.png \
    --prompts_filepath data/samples/prompts.txt \
    --prompt_mask_mapping_filepath data/outputs/prompt_mask.txt \
    --save_dirpath data/outputs/seg_sam \
    --config_path segmentation/utils/segment_config.yaml


## Contributing to SceneComplete
We encourage swapping modules for better performance:

Replace BrushNet with another inpainting approach.
Switch out InstantMesh for Zero123 or a new 2D→3D model.
Adjust pose registration if you have a robust alternative.
PRs are Welcome: If you improve or replace modules with stronger versions, please open a pull request – we’d love to incorporate better approaches or domain adaptations.


## Citation
@inproceedings{agarwal2024scenecomplete,
  title={{SceneComplete}: Open-World 3D Scene Completion in Complex Real-World Environments for Robot Manipulation},
  author={Agarwal, Aditya and Singh, Gaurav and Sen, Bipasha and Lozano-P{\'e}rez, Tom{\'a}s and Kaelbling, Leslie Pack},
  year={2024},
  archivePrefix={arXiv},
  eprint={2410.23643v2}
}
