#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

# type=hand
# type=hand_ego_centric
# type=body
# type=frank
type=frank_fast

input_type=image
# input_type=video
# input_type=pkl


if [ "$input_type" = "image" ]; then

    if [ "$type" = "hand" ]; then
        echo "Running Hand Demo (Image)"
        xvfb-run -a python -m demo.demo_handmocap \
            --out_dir ./sample_data/output/images/compose/hand \
            --input_path ./sample_data/images/compose \
            --renderer_type pytorch3d \
            --save_bbox_output \
            --save_pred_pkl \
            --save_mesh 

    elif [ "$type" = "body" ]; then
        echo "Running Body Demo (Image)"
        xvfb-run -a python -m demo.demo_bodymocap \
            --input_path ./sample_data/images/compose \
            --out_dir ./sample_data/output/images/compose/body \
            --renderer_type pytorch3d \
            --use_smplx \
            --save_bbox_output \
            --save_pred_pkl \
            --save_mesh 

    elif [ "$type" = "frank" ]; then
        echo "Running Whole Body Demo (Image & Slow Mode)"
        python -m demo.demo_frankmocap \
            --input_path ./sample_data/images/compose \
            --out_dir ./sample_data/output/images/compose/whole_slow \
            --renderer_type pytorch3d \
            --save_bbox_output \
            --save_pred_pkl \
            --save_mesh 

    elif [ "$type" = "frank_fast" ]; then
        echo "Running Whole Body Demo (Image & Fast Mode)"
        xvfb-run -a python -m demo.demo_frankmocap \
            --input_path ./sample_data/images/compose \
            --out_dir ./sample_data/output/images/compose/whole_fast \
            --renderer_type pytorch3d \
            --frankmocap_fast_mode \
            --save_bbox_output \
            --save_pred_pkl \
            --save_mesh 

    else
        exit
    fi


# video input
elif [ "$input_type" = "video" ]; then

     if [ "$type" = "hand" ]; then
        echo "Running Hand Demo (Video & Third View)"
        xvfb-run -a python -m demo.demo_handmocap \
            --input_path ./sample_data/videos/demo/rongyu_hand.mp4 \
            --out_dir ./sample_data/output/videos/rongyu_hand \
            --renderer_type pytorch3d 

    elif [ "$type" = "hand_ego_centric" ]; then
        echo "Running Hand Demo (Video & Ego Centric View)"
        xvfb-run -a python -m demo.demo_handmocap \
            --input_path ./sample_data/videos/demo/ego_centric.mp4 \
            --out_dir ./sample_data/output/videos/ego_centric \
            --view_type ego_centric \
            --renderer_type pytorch3d 
    
     elif [ "$type" = "body" ]; then
        echo "Running Body Demo (Video)"
        xvfb-run -a python -m demo.demo_bodymocap \
            --input_path ./sample_data/videos/demo/xiongyu_body.mp4 \
            --out_dir ./sample_data/output/videos/body \
            --renderer_type pytorch3d \
            --use_smplx 
    
    elif [ "$type" = "frank" ]; then
        echo "Running Whole Body Demo (Video & Slow Mode)"
        xvfb-run -a python -m demo.demo_frankmocap \
            --input_path ./sample_data/videos/demo/xiongyu_body.mp4 \
            --out_dir ./sample_data/output/videos/xiongyu_body \
            --renderer_type pytorch3d 
         
    
    elif [ "$type" = "frank_fast" ]; then
        echo "Running Whole Body Demo (Video & Fast Mode)"
        xvfb-run -a python -m demo.demo_frankmocap \
            --input_path ./sample_data/videos/demo/multi_person.mp4 \
            --out_dir ./sample_data/output/videos/multi_person_whole_fast \
            --frankmocap_fast_mode \
            --save_pred_pkl \
            --save_mesh 

    else
        exit
    fi


# pkl
else
    echo "Visualize Mesh from Saved PKL Files"
    CUDA_VISIBLE_DEVICES=9 xvfb-run -a python -m demo.demo_visualize_prediction \
        --pkl_dir ./sample_data/output/images/compose/whole_fast \
        --out_dir ./sample_data/output/pkl_vis/whole_fast \
        --renderer_type pytorch3d \
        --save_pred_pkl \
        --save_mesh
fi
