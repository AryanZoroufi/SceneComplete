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

# Activate the conda environment
echo "Activating the scenecomplete environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scenecomplete

run_prompting=true
run_segmentation=true
run_inpainting=true
run_segmentation_post_inpainting=true
prepare_3d_inputs=true
run_3d_reconstruction=true
run_mesh_scaling=true
run_registration=true

perform_cleanup=false


if [ "$run_prompting" = true ]; then
    echo "Generating scene prompts"
    python scenecomplete/scripts/python/prompting/generate_scene_prompts.py \
        --image_path $scdirpath/inputs/rgb.png \
        --output_filepath $experiment_dir/prompts.txt
fi

if [ "$run_segmentation" = true ]; then
    echo "Running segmentation"
    python scenecomplete/scripts/python/segmentation/segment_objects.py \
    --image_path $scdirpath/inputs/rgb.png \
    --depth_path $scdirpath/inputs/depth.png \
    --prompts_filepath $experiment_dir/prompts.txt \
    --prompt_mask_mapping_filepath $experiment_dir/prompt_mask_mapping.txt \
    --save_dirpath $experiment_dir/sam_outputs
fi

if [ "$run_inpainting" = true ]; then
    echo "Running inpainting"
    python scenecomplete/scripts/python/inpainting/inpaint_objects.py \
    --seed 42 \
    --prompt_filepath $experiment_dir/prompt_mask_mapping.txt \
    --output_dirpath $experiment_dir/inpainting_outputs \
    --use_pretrained \
    --blended

fi

if [ "$run_segmentation_post_inpainting" = true ]; then
    echo "Running segmentation post-inpainting"
    python scenecomplete/scripts/python/segmentation/segment_objects_post_inpainting.py \
        --input_dirpath $experiment_dir/inpainting_outputs \
        --prompt_mask_mapping_filepath $experiment_dir/prompt_mask_mapping.txt \
        --save_dirpath $experiment_dir/sam_post_processed \
        --resize_ratio 0.85
fi

if [ "$prepare_3d_inputs" = true ]; then
    echo "Preparing the input for reconstruction"
    python scenecomplete/scripts/python/reconstruction/utils/prepare_3d_inputs.py \
        --segmentation_dirpath $experiment_dir/sam_outputs \
        --inpainting_dirpath $experiment_dir/sam_post_processed \
        --out_path $experiment_dir/grasp_data \
        --scene_rgb_filepath $scdirpath/inputs/rgb.png \
        --scene_depth_filepath $scdirpath/inputs/depth.png \
        --intrinsics_path $scdirpath/inputs/cam_K.txt
fi

if [ "$run_3d_reconstruction" = true ]; then
    echo "Running 3D reconstruction"
    python scenecomplete/scripts/python/reconstruction/generate_3d_mesh.py \
        $experiment_dir/grasp_data/imesh_inputs \
        --config instant-mesh-base.yaml \
        --output_path $experiment_dir/imesh_outputs \
        --seed 42 \
        --no_rembg \
        --export_texmap
fi

if [ "$run_mesh_scaling" = true ]; then
    echo "Computing scaling by estimating correspondences"
    python scenecomplete/scripts/python/scaling/compute_mesh_scaling.py \
        --segmentation_dirpath $experiment_dir/grasp_data \
        --imesh_outputs $experiment_dir/imesh_outputs \
        --output_filepath $experiment_dir/obj_scale_mapping.txt \
        --instant_mesh_model instant-mesh-base \
        --camera_name realsense
fi

if [ "$run_registration" = true ]; then
    echo "Computing 6D poses to register the object meshes to the scene"
    echo "Activating the foundationpose conda environment"
    conda activate foundationpose
    python scenecomplete/scripts/python/registration/register_mesh.py \
        --imesh_outputs $experiment_dir/imesh_outputs \
        --segmentation_dirpath $experiment_dir/grasp_data \
        --obj_scale_mapping $experiment_dir/obj_scale_mapping.txt \
        --instant_mesh_model instant-mesh-base \
        --output_dirpath $experiment_dir/registered_meshes
fi

if [ "$perform_cleanup" = true ]; then
    echo "Performing cleanup"
    rm -rf $experiment_dir/grasp_data
    rm -rf $experiment_dir/imesh_outputs
    rm -rf $experiment_dir/inpainting_outputs
    rm -rf $experiment_dir/sam_outputs
    rm -rf $experiment_dir/sam_post_processed
    rm $experiment_dir/obj_scale_mapping.txt
    rm $experiment_dir/prompt_mask_mapping.txt
    rm $experiment_dir/prompts.txt
fi

echo "SceneComplete run successfully"