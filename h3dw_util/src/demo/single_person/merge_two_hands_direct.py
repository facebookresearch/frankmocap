import os, sys, shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('src/')
import os.path as osp
import argparse
import numpy as np
import torch
import smplx
from utils.render_utils import render
import cv2
import multiprocessing as mp
import utils.geometry_utils as gu
import parallel_io as pio
import ry_utils
import pdb
import time
from collections import defaultdict
from utils.vis_utils import render_hand, render_body


def load_bbox_info(bbox_info_dir):
    bbox_info = dict()
    bbox_info_files = ry_utils.get_all_files(bbox_info_dir, ".pkl", "name_only")
    for file in bbox_info_files:
        if file.find("left")>=0 or file.find("right")>=0:
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
            pred_pose = pred_pose)
    # print(list(pred_info.keys())[0])
    return pred_info


def load_all_data(root_dir):
    # load bbox
    bbox_info_dir = osp.join(root_dir, "bbox_info")
    bbox_info = load_bbox_info(bbox_info_dir)

    # load frame
    frame_dir = osp.join(root_dir, "frame")
    frame_info = load_all_frame(frame_dir)

    # load predicted camera & pose
    pred_info_file = osp.join(root_dir, "prediction/h3dw/pred_results_youtube.pkl")
    pred_info = load_pred_info(pred_info_file)

    return frame_info, bbox_info, pred_info


def get_pred_verts(smplx_model, hand_info_smplx, hand_type, pred_hand_pose):
    hand_type = hand_type.split('_')[0] # left_hand -> left
    if hand_type == 'left':
        left_hand_rot = torch.from_numpy(pred_hand_pose[:3].reshape(1, 3)).float()
        left_hand_pose = torch.from_numpy(pred_hand_pose[3:].reshape(1, 45)).float()
        right_hand_rot = torch.zeros((1, 3)).float()
        right_hand_pose = torch.zeros((1, 45)).float()
    else:
        assert hand_type == 'right'
        right_hand_rot = torch.from_numpy(pred_hand_pose[:3].reshape(1, 3)).float()
        right_hand_pose = torch.from_numpy(pred_hand_pose[3:].reshape(1, 45)).float()
        left_hand_rot = torch.zeros((1, 3)).float()
        left_hand_pose = torch.zeros((1, 45)).float()

    output = smplx_model(
        left_hand_rot = left_hand_rot,
        left_hand_pose_full = left_hand_pose,
        right_hand_rot = right_hand_rot,
        right_hand_pose_full = right_hand_pose)
    
    hand_output = smplx_model.get_hand_output(output, hand_type, hand_info_smplx, 'ave')
    verts_shift = hand_output.hand_vertices_shift.detach().cpu().numpy().squeeze()
    hand_faces = hand_info_smplx[f'{hand_type}_hand_faces_local']
    return verts_shift, hand_faces


def get_new_cam(pred_cam, inputSize, bbox):
    scale = pred_cam[0]
    # trans = pred_cam[1:]

    min_x, min_y, max_x, max_y = bbox
    bbox_size = max(max_x-min_x, max_y-min_y)

    scale *= (bbox_size / inputSize)

    t_x = (min_x / inputSize) * 2
    t_y = (min_y / inputSize) * 2

    pred_cam[1] += (2*min_x + bbox_size - inputSize) / (inputSize * scale)
    pred_cam[2] += (2*min_y + bbox_size - inputSize) / (inputSize * scale)
    # pred_cam[2] += t_y / pred_cam[0]

    # print(t_x / pred_cam[0])
    # print(t_y / pred_cam[0])
    # pred_cam[1] = 0
    # pred_cam[2] = 0

    pred_cam[0] = scale
    return pred_cam


def render_img(bbox, pred_cam, pred_verts, hand_faces, res_img):
    min_x, min_y, max_x, max_y = bbox
    assert max_x > min_x and max_y > min_y

    h_res, w_res = res_img.shape[:2]
    inputSize = max(h_res, w_res)
    pad_img = np.zeros((inputSize, inputSize, 3), dtype=np.uint8)
    pad_img[:h_res, :w_res] = res_img

    new_cam = get_new_cam(pred_cam, inputSize, bbox)
    pad_img = render(pred_verts, hand_faces, new_cam, inputSize, pad_img)
    return pad_img[:h_res, :w_res]


def render_two_hand(frame_info, bbox_info, pred_info, res_dir):
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data"
    smplx_model_file = osp.join(root_dir, "models/smplx/SMPLX_NEUTRAL.pkl")
    smplx_model = smplx.create(smplx_model_file, model_type="smplx")
    hand_info_file = osp.join(root_dir, "models/smplx/SMPLX_HAND_INFO.pkl")
    hand_info_smplx  = pio.load_pkl_single(hand_info_file)

    sorted_keys = sorted(list(frame_info.keys()))
    for key_id, img_key in enumerate(sorted_keys):
        frame_path = frame_info[img_key]
        # res_img_path = 
        record = frame_path.split('/')[-1].split('.')[0].split('_')
        seq_name = '_'.join(record[:-1])
        res_img_path = osp.join(res_dir, seq_name, frame_path.split('/')[-1])
        ry_utils.make_subdir(res_img_path)

        res_img = cv2.imread(frame_path)
        has_both_hand = True
        for hand_type in ['right_hand', 'left_hand']:
            img_key_hand = img_key + '_' + hand_type
            if img_key_hand in pred_info:
                pred = pred_info[img_key_hand]
                pred_cam = pred['pred_cam']
                pred_hand_pose =pred['pred_pose']
                pred_verts, hand_faces = get_pred_verts(smplx_model, hand_info_smplx, hand_type, pred_hand_pose)
                res_img = render_img(bbox_info[img_key_hand], pred_cam, pred_verts, hand_faces, res_img)
            else:
                has_both_hand = False
                continue
        if has_both_hand:
            cv2.imwrite(res_img_path, res_img)
    
        print(f"Processed:{key_id:04d}/{len(sorted_keys)}")
        

def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube"
    frame_info, bbox_info, pred_info = load_all_data(root_dir)

    res_dir = osp.join(root_dir, 'prediction/h3dw/origin_frame')
    ry_utils.renew_dir(res_dir)
    render_two_hand(frame_info, bbox_info, pred_info, res_dir)


if __name__ == '__main__':
    main()