# Copyright (c) Facebook, Inc. and its affiliates.

import torch

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import demo.demo_common
from demo.demo_options import DemoOptions


def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer):
    for input_frame_and_metadata in demo.demo_common.input_frame_and_metadata_iterator(args):
        img_original_bgr = input_frame_and_metadata.img_original_bgr

        if not demo.demo_common.detect_hand_bbox_and_save_it_into_frame_and_metadata(
                args, input_frame_and_metadata, bbox_detector.detect_hand_bbox
        ):
            continue

        body_bbox_list = input_frame_and_metadata.body_bbox_list
        hand_bbox_list = input_frame_and_metadata.hand_bbox_list

        # Hand Pose Regression
        pred_output_list = hand_mocap.regress(
                img_original_bgr, hand_bbox_list, add_margin=True)
        assert len(hand_bbox_list) == len(body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        demo.demo_common.show_and_save_result(
            args, 'hand', input_frame_and_metadata, visualizer, pred_output_list
        )


def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    #Set Bbox detector
    bbox_detector =  HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    # run
    run_hand_mocap(args, bbox_detector, hand_mocap, visualizer)
   

if __name__ == '__main__':
    main()
