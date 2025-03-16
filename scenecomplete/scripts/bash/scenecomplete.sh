#!/usr/bin/env bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scenecomplete

echo "Generating scene prompts"
python scenecomplete/scripts/python/prompting/generate_scene_prompts.py \
    --image_path $scdirpath/rgb.png \
    --output_filepath $scdirpath/prompts.txt


echo "Running segmentation"
python scenecomplete/scripts/python/segmentation/segment_objects.py \
   --image_path $scdirpath/rgb.png \
   --depth_path $scdirpath/depth.png \
   --prompts_filepath $scdirpath/prompts.txt \
   --prompt_mask_mapping_filepath $scdirpath/prompt_mask_mapping.txt \
   --save_dirpath $scdirpath/sam_outputs


echo "Running inpainting"
python scenecomplete/scripts/python/inpainting/inpaint_objects.py \
   --seed 42 \
   --prompt_filepath $scdirpath/prompt_mask_mapping.txt \
   --output_dirpath $scdirpath/inpainting_outputs \
   --use_pretrained \
   --blended


echo "Running segmentation post-inpainting"
python scenecomplete/scripts/python/segmentation/segment_objects_post_inpainting.py \
    --input_dirpath 
    --prompt_mask_mapping_filepath $scdirpath/prompt_mask_mapping.txt \
    --save_dirpath $scdirpath/sam_outputs_post_inpainting
    --resize_ratio 0.85


echo "Running 3D reconstruction"
