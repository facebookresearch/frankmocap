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
from utils import vis_utils
import utils.geometry_utils as gu
import time
import torch

from demo.temporal_two_hands.temporal_model import TemporalModel
from demo.temporal_two_hands.sample import Sample


def load_bbox_info(bbox_info_dir):
    bbox_info = dict()
    bbox_info_files = ry_utils.get_all_files(bbox_info_dir, ".pkl", "name_only")
    for file in bbox_info_files:
        # if file.find("left")>=0 or file.find("right")>=0:
        bbox_file = osp.join(bbox_info_dir, file)
        one_hand_info = pio.load_pkl_single(bbox_file)
        bbox_info.update(one_hand_info)
    # print(list(bbox_info.keys())[0])
    return bbox_info


def load_all_frame(frame_dir):
    frame_info = dict()
    for subdir, dirs, files in os.walk(frame_dir):
        for file in files:
            if file.endswith(".png"):
                key = file.split('.')[0]
                full_path = osp.join(subdir, file)
                frame_info[key] = full_path
    # print(list(frame_info.keys())[0])
    return frame_info


def load_pred_info(pred_info_file):
    all_data = pio.load_pkl_single(pred_info_file)
    pred_info = dict()
    for data in all_data:
        img_name = data['img_name']
        img_key = img_name.split('/')[-1].split('.')[0]
        if img_key.find("left_hand")>=0:
            hand_type = 'left_hand'
        else:
            assert img_key.find("right_hand")
            hand_type = 'right_hand'
        pred_cam = data['cam']
        pred_pose = data['pred_pose_params']
        if hand_type == 'left_hand':
            pred_pose = gu.flip_hand_pose(pred_pose)
            pred_cam[1] *= -1 # flip x
        pred_info[img_key] = dict(
            pred_cam = pred_cam,
            pred_shape = data['pred_shape_params'],
            pred_pose = pred_pose)
    # print(list(pred_info.keys())[0])
    return pred_info


def load_openpose_info(openpose_dir):
    openpose_json_files = ry_utils.get_all_files(openpose_dir, ".json", "full")
    openpose_info = dict()

    for json_file in openpose_json_files:
        img_id = json_file.split('/')[-1]
        img_id = img_id.replace('_keypoints.json','')
        openpose_info[img_id] = json_file
    return openpose_info


def load_openpose_score(openpose_dir):
    openpose_json_files = ry_utils.get_all_files(openpose_dir, ".json", "full")
    openpose_score = dict()

    for json_file in openpose_json_files:
        hand_score = dict(left=0.0, right=0.0)
        img_id = json_file.split('/')[-1]
        img_id = img_id.replace('_keypoints.json','')

        with open(json_file, "r") as in_f:
            all_data = json.load(in_f)
            data = all_data['people']
            if len(data) > 0: 
                data = data[0]
                for hand_type in ['left', 'right']:
                    key = f"hand_{hand_type}_keypoints_2d"
                    hand_2d = np.array(data[key]).reshape(21, 3)
                    score = np.average(hand_2d[:, 2])
                    hand_score[hand_type+"_hand"] = score

        openpose_score[img_id] = hand_score
    return openpose_score


def load_all_data(root_dir):
    # load bbox
    bbox_info_dir = osp.join(root_dir, "bbox_info")
    bbox_info = load_bbox_info(bbox_info_dir)

    # load frame
    frame_dir = osp.join(root_dir, "frame")
    frame_info = load_all_frame(frame_dir)

    # load openpose_score
    openpose_dir = osp.join(root_dir, "openpose_output")
    openpose_score = load_openpose_score(openpose_dir)
    openpose_info = load_openpose_info(openpose_dir)

    # load openpose file

    # load predicted camera & pose
    pred_info_file = osp.join(root_dir, "prediction/pred_results.pkl")
    pred_info = load_pred_info(pred_info_file)


    return frame_info, bbox_info, openpose_score, openpose_info, pred_info


# def merge_data(body_info, hand_info, openpose_score):
def merge_data(frame_info, bbox_info, openpose_score, openpose_info, pred_info):
    seq_info = defaultdict(list)
    for img_name in frame_info:
        seq_name = img_name.split('/')[0]
        seq_name = '_'.join(img_name.split('_')[:-1])
        seq_info[seq_name].append(img_name)
    for seq_name in seq_info:
        seq_info[seq_name] = sorted(seq_info[seq_name]) # sort the data, important
    
    all_samples = defaultdict(list)
    for seq_name in seq_info:
        all_img_names = seq_info[seq_name]

        for sample_id, img_name in enumerate(all_img_names):
            res_pred_hand_info = dict(
                left_hand = dict(),
                right_hand = dict(),
            )
            frame_path = frame_info[img_name]
            img_name = frame_path.split('/')[-1].split('.')[0]

            for hand_type in ['left_hand', 'right_hand']:

                img_name_hand = f"{img_name}_{hand_type}"
                pred_hand_info = pred_info[img_name_hand]
                res_pred_hand_info[hand_type].update(pred_hand_info)

                res_pred_hand_info[hand_type]['openpose_score'] = \
                     openpose_score[img_name][hand_type]
                
                res_pred_hand_info[hand_type]['bbox'] = \
                    bbox_info[img_name_hand]
                
            sample = Sample(
                seq_name = seq_name,
                sample_id = sample_id,
                img_name = img_name,
                frame_path = frame_path,
                openpose_path = openpose_info[img_name],
                pred_hand_info = res_pred_hand_info,
            )
            all_samples[seq_name].append(sample)

    return all_samples


def load_all_samples(root_dir):
    frame_info, bbox_info, openpose_score, openpose_info, pred_info = load_all_data(root_dir)

    all_samples = merge_data(frame_info, bbox_info, openpose_score, openpose_info, pred_info)

    return all_samples