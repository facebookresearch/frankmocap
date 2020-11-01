#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.


input_type=image
# input_type=video
# input_type=pkl

root_dir=/mnt/SSD/rongyu/data/3D/frank_mocap/demo

echo "Running Whole Body Demo (Image & EFT)"
CUDA_VISIBLE_DEVICES=0 python -m demo.demo_frankmocap \
    --input_path $root_dir/frame/adobe/video_02 \
    --out_dir  $root_dir/output/fm_eft/adobe/video_02 \
    --openpose_dir $root_dir/openpose_output/adobe/video_02 \
    --integrate_type opt \
    --renderer_type opendr \
    --no_display \
    --save_bbox_output \
    --save_pred_pkl \
    --single_person \
    --save_mesh 