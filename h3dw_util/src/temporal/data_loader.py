import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
from collections import defaultdict
import json
import smplx
import cv2
import ry_utils
import parallel_io as pio
import pdb
from temporal.temporal_model import TemporalModel
from temporal.sample import Sample
from utils import vis_utils
import utils.geometry_utils as gu
import time
import torch

def load_pred_body(in_dir):
    frame_dir = osp.join(in_dir, "frame")
    pred_res = pio.load_pkl_single(osp.join(in_dir, "smpl_pred.pkl"))

    body_info = dict()
    for img_name, value in pred_res.items():
        frame_path = osp.join(frame_dir, img_name)
        frame_path = ry_utils.update_extension(frame_path, '.jpg')
        body_info[img_name] = dict(
            frame_path = frame_path,
            pred_body_cam = value['pred_cam'],
            pred_body_pose = value['pred_pose_param'],
            pred_body_shape = value['pred_shape_param'],
        )
    return body_info


def _default_pred_hand():
    return dict(
        left_hand = (dict(pred_hand_cam=np.zeros(3,), pred_hand_pose=np.zeros(48,)), '', ''),
        right_hand = (dict(pred_hand_cam=np.zeros(3,), pred_hand_pose=np.zeros(48,)), '', ''),
    )

def load_pred_hand(in_dir):
    pred_res = pio.load_pkl_single(osp.join(in_dir, "pred_results_demo_body_capture.pkl"))
    crop_hand_dir = osp.join(in_dir, "image_hand")
    render_hand_dir = osp.join(in_dir, "image_render")

    hand_info = defaultdict(
        _default_pred_hand
    )
    for value in pred_res:
        img_name = value['img_name']

        # get crop hand and render hand image
        crop_hand_img = osp.join(crop_hand_dir, img_name)
        render_hand_img = osp.join(render_hand_dir, img_name)
        crop_hand_img = ry_utils.update_extension(crop_hand_img, '.jpg')
        render_hand_img = ry_utils.update_extension(render_hand_img, '.jpg')
        assert osp.exists(crop_hand_img)
        assert osp.exists(render_hand_img)

        # get hand_type
        record = img_name.split('/')
        if record[0] == 'left_hand':
            hand_type = 'left'
        else:
            assert record[0] == 'right_hand'
            hand_type = 'right'
        img_id = record[-1].replace(f'_{hand_type}_hand', '')[:-4]
        img_name = osp.join(record[1], img_id + '.png')

        # get predicted hand pose
        pred_pose = value['pred_pose_params']
        if hand_type == 'left':
            pred_pose = gu.flip_hand_pose(pred_pose)
        
        # save to res
        pred_hand_info = dict(
            pred_hand_cam = value['cam'],
            pred_hand_pose = pred_pose,
        )
        hand_info[img_name][f'{hand_type}_hand'] = (pred_hand_info, crop_hand_img, render_hand_img)

    return hand_info


def _default_score():
    return dict(
        left_hand = 0.0,
        right_hand = 0.0
    )

def load_openpose_score(openpose_dir, hand_info):
    openpose_score = defaultdict(
        _default_score
    )
    for img_name in hand_info:
        json_file = osp.join(openpose_dir, img_name.replace(".png", "_keypoints.json"))
        with open(json_file, 'r') as in_f:
            all_data = json.load(in_f)
            data = all_data['people']
            assert len(data)>0
            data = data[0]

            for hand_type in ['left', 'right']:
                key = f"hand_{hand_type}_keypoints_2d"
                hand_2d = data[key]
                keypoint = np.array(hand_2d.copy()).reshape(21, 3)

                score = np.average(keypoint[:, 2])
                openpose_score[img_name][f"{hand_type}_hand"] = score

    return openpose_score


# add local hand wrist
def update_hand_info(pred_body_info, hand_info):
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    smplx_model_file = osp.join(root_dir, "data/models/smplx/SMPLX_NEUTRAL.pkl")
    smplx_model = smplx.create(smplx_model_file, model_type="smplx")

    pred_body_pose = pred_body_info['pred_body_pose']
    for hand_type in ['left_hand', 'right_hand']:
        img_name = hand_info[hand_type][1]
        pred_hand_pose = hand_info[hand_type][0]['pred_hand_pose']
        wrist_id = 20 if hand_type == 'left_hand' else 21

        wrist_rot_local = pred_hand_pose[:3].copy()
        wrist_rot_body = pred_hand_pose[:3].copy()

        if len(img_name) == 0:
            wrist_rot_local = pred_body_pose[wrist_id*3 : (wrist_id+1)*3]
            # pred_hand_pose_local[:3] = pred_body_pose[wrist_id*3 : (wrist_id+1)*3]
        else:
            hand_wrist_global = torch.from_numpy(pred_hand_pose[:3]).view(1, 3).float()
            body_pose = torch.from_numpy(pred_body_pose).float()
            kinematic_map = vis_utils.get_kinematic_map(smplx_model, wrist_id)
            wrist_rot_local = vis_utils.get_local_hand_rot(
                body_pose, hand_wrist_global, kinematic_map).numpy()[0]

            wrist_from_body = pred_body_pose[wrist_id*3 : (wrist_id+1)*3]
            wrist_from_body = torch.from_numpy(wrist_from_body).float().view(1, 3)
            wrist_rot_body = vis_utils.get_global_hand_rot(
                body_pose, wrist_from_body, kinematic_map).numpy()[0]

        # pred_hand_pose: (48,), hand pose (global rotation)
        # wrist_rot_local (3,), local rotation of wrist predicted from hand model (in term of body pose)
        # wrist_rot_body_global, (3,), global rot of wrist predicted from body
        hand_info[hand_type][0]['wrist_rot_local'] = wrist_rot_local
        hand_info[hand_type][0]['wrist_rot_body_global'] = wrist_rot_body
            

def merge_data(body_info, hand_info, openpose_score):
    seq_info = defaultdict(list)
    for img_name in body_info:
        seq_name = img_name.split('/')[0]
        seq_info[seq_name].append(img_name)
    for seq_name in seq_info:
        seq_info[seq_name] = sorted(seq_info[seq_name]) # sort the data, important

    all_samples = defaultdict(list)
    for seq_name in seq_info:
        all_img_names = seq_info[seq_name]
        for sample_id, img_name in enumerate(all_img_names):
            frame_path = body_info[img_name]['frame_path']
            pred_body_info = body_info[img_name]

            left_hand_info = hand_info[img_name]['left_hand']
            right_hand_info = hand_info[img_name]['right_hand']

            update_hand_info(pred_body_info, hand_info[img_name])
            print(sample_id)

            hand_img_path = dict(
                left_hand = left_hand_info[1],
                right_hand = right_hand_info[1]
            )
            render_img_path = dict(
                left_hand = left_hand_info[2],
                right_hand = right_hand_info[2]
            )
            pred_hand_info = dict(
                left_hand = left_hand_info[0],
                right_hand = right_hand_info[0]
            )

            # update_hand_wrist_local(pred_hand_info, pred_body_info)

            sample = Sample(
                seq_name = seq_name,
                sample_id = sample_id,
                img_name = img_name,
                frame_path = frame_path,
                hand_img_path = hand_img_path,
                render_img_path = render_img_path,
                pred_body_info = pred_body_info,
                pred_hand_info = pred_hand_info,
                openpose_score = openpose_score[img_name],
            )

            all_samples[seq_name].append(sample)
    
    return all_samples


def load_all_samples(data_dir, exp_res_dir):
    # load predicted body pose
    pred_body_dir = osp.join(exp_res_dir, 'body')
    body_info = load_pred_body(pred_body_dir)

    # load predicted hand pose
    pred_hand_dir = osp.join(exp_res_dir, 'hand')
    hand_info = load_pred_hand(pred_hand_dir)

    # load openpose score
    # only consider samples with crop hand image / predicted hand pose available
    openpose_dir = osp.join(data_dir, "openpose_output_origin")
    openpose_score = load_openpose_score(openpose_dir, hand_info)

    # let's have this assertion first
    assert len(body_info) == len(hand_info)
    assert len(openpose_score) == len(hand_info)

    all_samples = merge_data(body_info, hand_info, openpose_score)
    return all_samples