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
    --input_dirpath $scdirpath/inpainting_outputs \
    --prompt_mask_mapping_filepath $scdirpath/prompt_mask_mapping.txt \
    --save_dirpath $scdirpath/sam_post_processed
    --resize_ratio 0.85


echo "Preparing the input for reconstruction"
python scenecomplete/scripts/python/reconstruction/utils/prepare_3d_inputs.py \
    --segmentation_dirpath $scdirpath/sam_outputs \
    --inpainting_dirpath $scdirpath/sam_post_processed \
    --out_path $scdirpath/grasp_data \
    --scene_rgb_filepath $scdirpath/rgb.png \
    --scene_depth_filepath $scdirpath/depth.png \
    --intrinsics_path $scdirpath/cam_K.txt \


