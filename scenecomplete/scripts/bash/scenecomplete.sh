#!/usr/bin/env bash

# Generate experiment ID if not provided
if [ -z "$1" ]; then
    # Format: YYYYMMDD_HHMMSS
    experiment_id=$(date +"%Y%m%d_%H%M%S")
    echo "No experiment_id provided. Using auto-generated ID: $experiment_id"
else
    experiment_id=$1
fi

# Create experiment directory
experiment_dir="$scdirpath/$experiment_id"
mkdir -p "$experiment_dir"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scenecomplete

echo "Generating scene prompts"
python scenecomplete/scripts/python/prompting/generate_scene_prompts.py \
    --image_path $scdirpath/rgb.png \
    --output_filepath $experiment_dir/prompts.txt


echo "Running segmentation"
python scenecomplete/scripts/python/segmentation/segment_objects.py \
   --image_path $scdirpath/rgb.png \
   --depth_path $scdirpath/depth.png \
   --prompts_filepath $experiment_dir/prompts.txt \
   --prompt_mask_mapping_filepath $experiment_dir/prompt_mask_mapping.txt \
   --save_dirpath $experiment_dir/sam_outputs


echo "Running inpainting"
python scenecomplete/scripts/python/inpainting/inpaint_objects.py \
   --seed 42 \
   --prompt_filepath $experiment_dir/prompt_mask_mapping.txt \
   --output_dirpath $experiment_dir/inpainting_outputs \
   --use_pretrained \
   --blended


echo "Running segmentation post-inpainting"
python scenecomplete/scripts/python/segmentation/segment_objects_post_inpainting.py \
    --input_dirpath $experiment_dir/inpainting_outputs \
    --prompt_mask_mapping_filepath $experiment_dir/prompt_mask_mapping.txt \
    --save_dirpath $experiment_dir/sam_post_processed \
    --resize_ratio 0.85


echo "Preparing the input for reconstruction"
python scenecomplete/scripts/python/reconstruction/utils/prepare_3d_inputs.py \
    --segmentation_dirpath $experiment_dir/sam_outputs \
    --inpainting_dirpath $experiment_dir/sam_post_processed \
    --out_path $experiment_dir/grasp_data \
    --scene_rgb_filepath $scdirpath/rgb.png \
    --scene_depth_filepath $scdirpath/depth.png \
    --intrinsics_path $scdirpath/cam_K.txt \


echo "Running 3D reconstruction"
python scenecomplete/scripts/python/reconstruction/generate_3d_mesh.py \
    $experiment_dir/grasp_data/imesh_inputs \
    --config instant-mesh-base.yaml \
    --output_path $experiment_dir/imesh_outputs \
    --seed 42 \
    --no_rembg \
    --export_texmap


echo "Computing scaling by estimating correspondences"
python scenecomplete/scripts/python/scaling/compute_mesh_scaling.py \
    --segmentation_dirpath $experiment_dir/grasp_data \
    --imesh_outputs $experiment_dir/imesh_outputs \
    --output_filepath $experiment_dir/obj_scale_mapping.txt \
    --instant_mesh_model instant-mesh-base \
    --camera_name realsense


echo "Computing 6D poses to register the object meshes to the scene"
echo "Activating the foundationpose conda environment"
conda activate foundationpose
python scenecomplete/scripts/python/registration/register_mesh.py \
    --imesh_outputs $experiment_dir/imesh_outputs \
    --segmentation_dirpath $experiment_dir/grasp_data \
    --obj_scale_mapping $experiment_dir/obj_scale_mapping.txt \
    --instant_mesh_model instant-mesh-base \
    --output_dirpath $experiment_dir/registered_meshes


echo "SceneComplete run successfully"