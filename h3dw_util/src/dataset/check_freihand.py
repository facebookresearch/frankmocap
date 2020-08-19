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
from render.render_utils import render
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt

from utils.freihand_utils.fh_utils import *
from utils.freihand_utils.model import HandModel, recover_root, get_focal_pp, split_theta
from utils import vis_utils
# import utils.geometry_utils as gu
import utils.rotate_utils as ru


def show_training_samples(data_root, res_dir=''):
    # load annotations
    db_data_anno = load_db_annotation(data_root, 'training')
    db_data_anno = list(db_data_anno)

    slice = 2*np.pi / 32.0
    hand_rotation = np.array([
        [0, 0, 0],
        [23*slice, 0, 0],
        [0, 9*slice, 0],
    ])

    # iterate over all samples
    for idx in range(db_size('training')):
        # load image and mask
        img = read_img(idx, data_root, 'training', 'gs')
        msk = read_msk(idx, data_root)

        # annotation for this frame
        K, mano, xyz = db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)

        # split mano parameters
        poses, shapes, uv_root, scale = split_theta(mano)

        res_img = img.copy()[:,:,::-1]
        res_img = cv2.resize(res_img, (512, 512))

        hand_rot = poses[0, :3]
        hand_pose = poses[0, 3:]
        render_img = vis_utils.render_hand(hand_pose, hand_rot)
        res_img = np.concatenate((res_img, render_img), axis=1)
        res_img_path = osp.join(res_dir, f"{idx:03d}.png")
        cv2.imwrite(res_img_path, res_img)
        if idx>4: break



def add_rotation(data_root, res_dir=''):
    # load annotations
    db_data_anno = load_db_annotation(data_root, 'training')
    db_data_anno = list(db_data_anno)

    # iterate over all samples
    for idx in range(db_size('training')):
        # load image and mask
        img = read_img(idx, data_root, 'training', 'gs')

        # annotation for this frame
        K, mano, xyz = db_data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        joints_2d = projectPoints(xyz, K)
        # split mano parameters
        poses, shapes, uv_root, scale = split_theta(mano)

        input_img = img.copy()[:,:,::-1]
        height, width = input_img.shape[:2]
        y_ratio, x_ratio = 512/height, 512/width
        input_img = cv2.resize(input_img, (512, 512))
        joints_2d[:, 0] *= x_ratio
        joints_2d[:, 1] *= y_ratio
        
        res_img_list = list()

        hand_rot = poses[0, :3]
        hand_pose = poses[0, 3:]
        render_img = vis_utils.render_hand(hand_pose, hand_rot)
        joint_img = vis_utils.draw_keypoints(input_img.copy(), joints_2d.copy(), color='red')
        res_img = np.concatenate((input_img, render_img, joint_img), axis=1)
        res_img_list.append(res_img)


        for angle in range(-180, 181, 30):
            rotate_img = ru.rotate_image(input_img.copy(), angle)
            rot_orient = ru.rotate_orient(hand_rot, angle)
            render_img = vis_utils.render_hand(hand_pose, rot_orient)
            origin = np.array((256, 256)).reshape(1, 2)
            joints_rot = ru.rotate_joints_2d(joints_2d.copy(), origin, angle)
            joint_img = vis_utils.draw_keypoints(rotate_img.copy(), joints_rot.copy(), color='red')
            res_img = np.concatenate((rotate_img, render_img, joint_img), axis=1)
            res_img_list.append(res_img)

        res_img = np.concatenate(res_img_list, axis=0)
        res_img_path = osp.join(res_dir, f"{idx:03d}.png")
        cv2.imwrite(res_img_path, res_img)
        if idx>4: break
 
    

def main():
    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/FreiHAND/data/data_original"

    res_dir = "visualization/freihand_anno"
    ry_utils.renew_dir(res_dir)
    # show_training_samples( data_root, res_dir=res_dir)

    add_rotation( data_root, res_dir=res_dir)


if __name__ == '__main__':
    main()