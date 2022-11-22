# Copyright (c) Facebook, Inc. and its affiliates.

import os
from re import M
import sys
import os.path as osp
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle
from datetime import datetime

from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
from bodymocap.body_bbox_detector_ort import LightweightedPoseDetector
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
from alfred.utils.timer import ATimer
import open3d as o3d
# from renderer.render import render
from realrender.render import render_human_mesh


def run_body_mocap(args, body_bbox_detector, body_mocap, visualizer):
    # Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)

    cur_frame = args.start_frame
    video_frame = 0
    timer = Timer()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window("Test", width=1280, height=900)

    while True:
        timer.tic()
        # load data
        load_bbox = False

        if input_type == 'image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames",
                                  f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)

        elif input_type == 'webcam':
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames",
                                  f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame += 1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break
        print("--------------------------------------")

        if load_bbox:
            body_pose_list = None
        else:
            with ATimer('pose'):
                keypoints_2d = body_bbox_detector.run_one_img(img_original_bgr)
                body_bbox_list = body_bbox_detector.get_enlarged_boxes_from_poses(
                    keypoints_2d, img_original_bgr.shape[0], img_original_bgr.shape[1])

        hand_bbox_list = [None, ] * len(body_bbox_list)

        # save the obtained body & hand bbox to json file
        if args.save_bbox_output:
            demo_utils.save_info_to_json(
                args, image_path, body_bbox_list, hand_bbox_list)

        if len(body_bbox_list) < 1:
            print(f"No body deteced: {image_path}")
            continue

        # Sort the bbox using bbox size
        # (to make the order as consistent as possible without tracking)
        bbox_size = [(x[2] * x[3]) for x in body_bbox_list]
        idx_big2small = np.argsort(bbox_size)[::-1]
        body_bbox_list = [body_bbox_list[i] for i in idx_big2small]
        if args.single_person and len(body_bbox_list) > 0:
            body_bbox_list = [body_bbox_list[0], ]

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        with ATimer('regress'):
            print('start regress.')
            # Body Pose Regression
            pred_output_list = body_mocap.regress(
                img_original_bgr, body_bbox_list)
            assert len(body_bbox_list) == len(pred_output_list)
            pred_mesh_list = demo_utils.extract_mesh_from_output(
                pred_output_list)

        # visualization
        with ATimer('vis'):
            # sim3dr
            pred_mesh_list = pred_mesh_list[0]
            vertices = pred_mesh_list['vertices']
            vertices[:, 2] = - vertices[:, 2]
            res_img = render_human_mesh(img_original_bgr, [
                                        vertices], pred_mesh_list['faces'], alpha=0.9, with_bg_flag=True)

        # vis.poll_events()
        # vis.update_renderer()

        cv2.imshow('original', img_original_bgr)
        cv2.imshow('res_img', res_img)
        cv2.waitKey(1)
        timer.toc(bPrint=True, title="Time")

    # vis.destroy_window()
    # save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    if input_type == 'webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()


def mesh_list_to_o3d_mesh(pred_mesh_list):
    res = []
    for i in pred_mesh_list:
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(i['vertices'])
        m.triangles = o3d.utility.Vector3iVector(i['faces'])
        res.append(m)
    return res


def main():
    args = DemoOptions().parse()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # assert torch.cuda.is_available(), "Current version only supports GPU"

    # Set bbox detector
    # body_bbox_detector = BodyPoseEstimator()
    body_bbox_detector = LightweightedPoseDetector(
        'extra_data/body_module/human-pose-estimation.onnx')

    # Set mocap regressor
    use_smplx = args.use_smplx
    checkpoint_path = args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    print("use_smplx", use_smplx)
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    run_body_mocap(args, body_bbox_detector, body_mocap, visualizer)


if __name__ == '__main__':
    main()
