# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os, sys, shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os.path as osp
import argparse
import numpy as np
import torch
import smplx
import ry_utils
import parallel_io as pio
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt
import random as rd

from utils.freihand_utils.fh_utils import *
from utils.freihand_utils.model import HandModel, recover_root, get_focal_pp, split_theta
from utils import data_utils
from utils import vis_utils


def load_all_data(data_root, res_img_dir):
    all_data = list()
    db_data_anno = load_db_annotation(data_root, 'training')
    db_data_anno = list(db_data_anno)

    num_data = db_size("training")
    # num_data = 20
    versions = sample_version.all_versions
    for version_id, version in enumerate(versions[:2]): # only consider first versions now
        for origin_data_id in range(num_data):
            # img path
            img_idx = sample_version.map_id(origin_data_id, version)
            img_name = f"{img_idx:08d}.jpg"
            img_path = osp.join(data_root, 'training', 'rgb', img_name)
            res_img_path = osp.join(res_img_dir, img_name)
            shutil.copy2(img_path, res_img_path)

            # mano parameters
            K, mano, xyz = db_data_anno[origin_data_id]
            K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
            poses, _, _, _ = split_theta(mano)

            # joints 2d
            frei_joints = projectPoints(xyz, K)
            smplx_hand_joints = data_utils.remap_joints(frei_joints, "freihand", "smplx", "hand")
            num_joint = smplx_hand_joints.shape[0]
            scores = np.ones((num_joint, 1), dtype=np.float32)
            smplx_hand_joints = np.concatenate((smplx_hand_joints, scores), axis=1)

            res_anno = dict(
                image_root = res_img_dir,
                image_name = img_name,
                mano_pose = poses[0, :],
                joints_2d = smplx_hand_joints,
                augmented = version_id > 0
            )
            all_data.append(res_anno)
            if len(all_data) % 1000 == 0:
                print(f"Processed:{version} {len(all_data)}")

    return all_data


def get_anno_image_single(single_data):
    img_path = osp.join(single_data['image_root'], 
        single_data['image_name'])
    poses = single_data['mano_pose']
    if poses.shape[0] > 45:
        poses = poses[3:]
    joints_2d = single_data['joints_2d']
    img = cv2.imread(img_path)

    slice = 2*np.pi / 32.0
    hand_global_rotation = np.array([
        [0, 0, 0],
        [23*slice, 0, 0],
        [0, 9*slice, 0],
    ])
    res_img = img.copy()
    h, w = img.shape[:2]
    for idx in range(len(hand_global_rotation)):
        global_rotation = hand_global_rotation[idx]
        render_img = vis_utils.render_hand(poses, global_rotation)
        render_img = cv2.resize(render_img, (w, h))
        res_img = np.concatenate( (res_img, render_img), axis=1)
    
    joint_img_full = vis_utils.draw_keypoints(img.copy(), joints_2d)
    joint_img_list = list()
    for joint_id in range(joints_2d.shape[0]):
        x, y = joints_2d[joint_id][:2].astype(np.int32)
        joint_img = cv2.circle(img.copy(), (x,y), radius=3, color=(0,0,255), thickness=-1)
        joint_img_list.append(joint_img)
    return res_img, joint_img_full, joint_img_list

def get_anno_images(all_data, res_vis_dir):
    ry_utils.renew_dir(res_vis_dir)
    for data_idx, single_data in enumerate(all_data):
        render_img, joint_img_list = get_anno_image_single(single_data)
        render_img_path = osp.join(res_vis_dir, f"mesh_{data_idx:02d}.png")
        cv2.imwrite(render_img_path, render_img)
        res_subdir = osp.join(res_vis_dir, f"joints_{data_idx:02d}")
        ry_utils.renew_dir(res_subdir)
        for joint_id, joint_img in enumerate(joint_img_list):
            joint_img_path = osp.join(res_subdir, f"joint_{joint_id:02d}.png")
            cv2.imwrite(joint_img_path, joint_img)


def split_data(all_data, res_anno_dir):
    train_ratio = 0.8
    val_ratio = 1.0 - train_ratio
    interval = int(1.0 / val_ratio)
    train_data, val_data = list(), list()
    for data_id, single_data in enumerate(all_data):
        if (data_id+1) % interval == 0 and not single_data['augmented']:
            val_data.append(single_data)
        else:
            train_data.append(single_data)

    pio.save_pkl_single(osp.join(res_anno_dir, 'train.pkl'), train_data)
    pio.save_pkl_single(osp.join(res_anno_dir, 'val.pkl'), val_data)


def process_frei(origin_data_dir, res_anno_dir, res_img_dir, res_vis_dir):
    # load data first
    all_data = load_all_data(origin_data_dir, res_img_dir)

    # visualize annotation
    # get_anno_images(all_data, res_vis_dir)

    # split train and test
    split_data(all_data, res_anno_dir)


def main():
    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/FreiHAND/data/"
    origin_data_dir = osp.join(data_root, "data_original")

    res_anno_dir = osp.join(data_root, "data_processed/annotation")
    res_img_dir = osp.join(data_root, "data_processed/image")
    res_vis_dir = osp.join(data_root, "data_processed/image_anno")
    ry_utils.renew_dir(res_anno_dir)
    ry_utils.renew_dir(res_img_dir)

    process_frei(origin_data_dir, res_anno_dir, res_img_dir, res_vis_dir)


if __name__ == '__main__':
    main()
