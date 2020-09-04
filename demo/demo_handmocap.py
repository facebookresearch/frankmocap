# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize

from .demo_options import DemoOptions
import mocap_utils.general_utils as g_utils
import mocap_utils.demo_utils as demo_utils

# from renderer import viewer2D #, glViewer
# from renderer.visualizer import Visualizer
from handmocap.mocap_api import HandMocap
from demo.demo_bbox_detector import HandBboxDetector
import renderer.opendr_renderer as od_render
from mocap_utils.vis_utils import Visualizer


def run_mocap_video(args, bbox_detector, hand_mocap):
    """
    Not implemented Yet.
    """
    pass


def run_mocap_image(args, bbox_detector, hand_mocap):
    #Set up input data (images or webcam)
    image_list, _ = demo_utils.setup_input(args)
    visualizer = Visualizer()

    for f_id, img_name in enumerate(image_list):
        img_path = osp.join(args.input_image_dir, img_name)

        # read images
        img_original_bgr = cv2.imread(img_path)

        if args.crop_type == 'hand_crop':
            pred_output = hand_mocap.regress(img_original_bgr, None, 'rhand')

            if args.renderer_type == "opendr":
                cam = np.zeros(3,)
                cam[0] = pred_output['cam_scale']
                cam[1:] = pred_output['cam_trans']
                bbox_scale_ratio = pred_output['bbox_scale_ratio']
                bbox_top_left = pred_output['bbox_top_left']
                verts = pred_output['pred_vertices_origin']
                faces = pred_output['faces']
                img = pred_output['img_cropped']

                rend_img0 = od_render.render(cam, verts, faces, bg_img=img)
                cv2.imwrite("0.png", rend_img0)
                rend_img1 = od_render.render_to_origin_img(cam, verts, faces, 
                    bg_img=img_original_bgr, bbox_scale=bbox_scale_ratio, bbox_top_left=bbox_top_left)
                cv2.imwrite("1.png", rend_img1)
                sys.exit(0)
            elif args.renderer_type == "opengl_no_gui":
                pass
            else:
                continue
        else:            
            # Input images has other body part or hand not cropped.
            assert args.crop_type == 'no_crop'
            body_pose_list, hand_bbox_list, raw_hand_bboxes = bbox_detector.detect_hand_bbox(img_original_bgr.copy())

            vis_img = visualizer.visualize(
                input_img = img_original_bgr.copy(), 
                hand_bbox_list = hand_bbox_list,
                body_pose_list = body_pose_list,
                raw_hand_bboxes = raw_hand_bboxes
            )
            res_img_path = osp.join(args.render_out_dir, img_name)
            g_utils.make_subdir(res_img_path)
            cv2.imwrite(res_img_path, vis_img)
   

def main():
    args = DemoOptions().parse()
    print(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    bbox_detector =  HandBboxDetector(args.view_type, device)

    SMPL_MODEL_DIR = './data/smplx/'
    hand_mocap = HandMocap(args.checkpoint, SMPL_MODEL_DIR, device = device)

    if args.input_type == 'image':
        run_mocap_image(args, bbox_detector, hand_mocap)
    else:
        run_mocap_video(args, bbox_detector, hand_mocap)

if __name__ == '__main__':
    main()