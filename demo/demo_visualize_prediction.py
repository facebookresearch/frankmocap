# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import os.path as osp
import torch
import numpy as np
import cv2
import argparse
import json
import pickle
import smplx
from datetime import datetime

from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from bodymocap.models import SMPL, SMPLX
from handmocap.hand_modules.h3dw_model import extract_hand_output
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm


def __get_data_type(pkl_files):
    for pkl_file in pkl_files:
        saved_data = gnu.load_pkl(pkl_file)
        return saved_data['demo_type'], saved_data['smpl_type']


def __get_smpl_model(demo_type, smpl_type):
    smplx_model_path = './extra_data/smpl/SMPLX_NEUTRAL.pkl'
    smpl_model_path = './extra_data/smpl//basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'

    if demo_type == 'hand':
        # use original smpl-x
        smpl = smplx.create(
            smplx_model_path, 
            model_type = "smplx", 
            batch_size = 1,
            gender = 'neutral',
            num_betas = 10,
            use_pca = False,
            ext='pkl'
        )
    else:
        if smpl_type == 'smplx':
            # use modified smpl-x from body module
            smpl = SMPLX(
                smplx_model_path,
                batch_size=1,
                num_betas = 10,
                use_pca = False,
                create_transl=False)
        else:
            # use modified smpl from body module
            assert smpl_type == 'smpl'
            smpl = SMPL(
                smpl_model_path, 
                batch_size=1, 
                create_transl=False)
    return smpl
    

def __calc_hand_mesh(hand_type, pose_params, betas, smplx_model):
    hand_rotation = pose_params[:, :3]
    hand_pose = pose_params[:, 3:]
    body_pose = torch.zeros((1, 63)).float()

    assert hand_type in ['left_hand', 'right_hand']
    if hand_type == 'right_hand':
        body_pose[:, 60:] = hand_rotation # set right hand rotation
        right_hand_pose = hand_pose
        left_hand_pose = torch.zeros((1, 45), dtype=torch.float32)
    else:
        body_pose[:, 57:60] = hand_rotation # set right hand rotation
        left_hand_pose = hand_pose
        right_hand_pose = torch.zeros((1, 45), dtype=torch.float32)

    output = smplx_model(
        global_orient = torch.zeros((1,3)),
        body_pose = body_pose,
        betas = betas,
        left_hand_pose = left_hand_pose,
        right_hand_pose = right_hand_pose,
        return_verts = True)
    
    hand_info_file = "extra_data/hand_module/SMPLX_HAND_INFO.pkl"
    hand_info = gnu.load_pkl(hand_info_file)
    hand_output = extract_hand_output(
        output, 
        hand_type = hand_type.split("_")[0],
        hand_info = hand_info,
        top_finger_joints_type = 'ave',
        use_cuda = False)

    pred_verts = hand_output['hand_vertices_shift'].detach().numpy()
    faces = hand_info[f'{hand_type}_faces_local']
    return pred_verts[0], faces


def _calc_body_mesh(smpl_type, smpl_model, body_pose, betas, 
    left_hand_pose, right_hand_pose):
    if smpl_type == 'smpl':
        smpl_output = smpl_model(
            global_orient = body_pose[:, :3],
            body_pose = body_pose[:, 3:], 
            betas = betas, 
        )
    else:
        smpl_output = smpl_model(
            global_orient = body_pose[:, :3],
            body_pose = body_pose[:, 3:], 
            betas = betas, 
            left_hand_pose = left_hand_pose,
            right_hand_pose = right_hand_pose,
        )

    vertices = smpl_output.vertices.detach().cpu().numpy()[0]
    faces = smpl_model.faces
    return vertices, faces


def __calc_mesh(demo_type, smpl_type, smpl_model, img_shape, pred_output_list):
    for pred_output in pred_output_list:
        if pred_output is not None:
            # hand
            if demo_type == 'hand':
                assert 'left_hand' in pred_output and 'right_hand' in pred_output
                for hand_type in pred_output:
                    hand_pred = pred_output[hand_type]
                    if hand_pred is not None:
                        pose_params = torch.from_numpy(hand_pred['pred_hand_pose'])
                        betas = torch.from_numpy(hand_pred['pred_hand_betas'])
                        pred_verts, hand_faces = __calc_hand_mesh(hand_type, pose_params, betas, smpl_model)
                        hand_pred['pred_vertices_smpl'] = pred_verts

                        cam_scale = hand_pred['pred_camera'][0]
                        cam_trans = hand_pred['pred_camera'][1:]
                        vert_bboxcoord = convert_smpl_to_bbox(
                            pred_verts, cam_scale, cam_trans, bAppTransFirst=True) # SMPL space -> bbox space

                        bbox_scale_ratio = hand_pred['bbox_scale_ratio']
                        bbox_top_left = hand_pred['bbox_top_left']
                        vert_imgcoord = convert_bbox_to_oriIm(
                                vert_bboxcoord, bbox_scale_ratio, bbox_top_left,
                                img_shape[1], img_shape[0]) 
                        pred_output[hand_type]['pred_vertices_img'] = vert_imgcoord
            # body
            else:
                pose_params = torch.from_numpy(pred_output['pred_body_pose'])
                betas = torch.from_numpy(pred_output['pred_betas'])
                if 'pred_right_hand_pose' in pred_output:
                    pred_right_hand_pose = torch.from_numpy(pred_output['pred_right_hand_pose'])
                else:
                    pred_right_hand_pose = torch.zeros((1, 45), dtype=torch.float32)
                if 'pred_left_hand_pose' in pred_output:
                    pred_left_hand_pose = torch.from_numpy(pred_output['pred_left_hand_pose'])
                else:
                    pred_left_hand_pose = torch.zeros((1, 45), dtype=torch.float32)
                pred_verts, faces = _calc_body_mesh(
                    smpl_type, smpl_model, pose_params, betas, pred_left_hand_pose, pred_right_hand_pose)

                pred_output['pred_vertices_smpl'] = pred_verts
                pred_output['faces'] = faces

                cam_scale = pred_output['pred_camera'][0]
                cam_trans = pred_output['pred_camera'][1:]
                vert_bboxcoord = convert_smpl_to_bbox(
                    pred_verts, cam_scale, cam_trans, bAppTransFirst=False) # SMPL space -> bbox space

                bbox_scale_ratio = pred_output['bbox_scale_ratio']
                bbox_top_left = pred_output['bbox_top_left']
                vert_imgcoord = convert_bbox_to_oriIm(
                        vert_bboxcoord, bbox_scale_ratio, bbox_top_left,
                        img_shape[1], img_shape[0]) 
                pred_output['pred_vertices_img'] = vert_imgcoord


def visualize_prediction(args, demo_type, smpl_type, smpl_model, pkl_files, visualizer):
    for pkl_file in pkl_files:
        # load data
        saved_data = gnu.load_pkl(pkl_file)
        
        image_path = saved_data['image_path']
        img_original_bgr = cv2.imread(image_path)
        if img_original_bgr is None:
            print(f"{image_path} does not exists, skip")
        
        print("--------------------------------------")

        demo_type = saved_data['demo_type']
        assert saved_data['smpl_type'] == smpl_type

        hand_bbox_list = saved_data['hand_bbox_list']        
        body_bbox_list = saved_data['body_bbox_list']
        pred_output_list = saved_data['pred_output_list']

        if not saved_data['save_mesh']:
            __calc_mesh(
                demo_type, smpl_type, smpl_model, img_original_bgr.shape[:2], pred_output_list)
        else:
            pass

        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualization
        res_img = visualizer.visualize(
            img_original_bgr,
            pred_mesh_list = pred_mesh_list,
            body_bbox_list = body_bbox_list,
            hand_bbox_list = hand_bbox_list)

        # save result image
        demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # save predictions to pkl
        if args.save_pred_pkl:
            args.use_smplx = smpl_type == 'smplx'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)


def main():
    args = DemoOptions().parse()

    # load pkl files
    pkl_files = gnu.get_all_files(args.pkl_dir, ".pkl", "full")

    # get smpl type
    demo_type, smpl_type = __get_data_type(pkl_files)

    # get smpl model
    smpl_model = __get_smpl_model(demo_type, smpl_type)

   # Set Visualizer
    assert args.renderer_type in ['pytorch3d', 'opendr'], \
        f"{args.renderer_type} not implemented yet."
    from renderer.screen_free_visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    # load smpl model
    visualize_prediction(args, demo_type, smpl_type, smpl_model, pkl_files, visualizer)


if __name__ == '__main__':
    main()