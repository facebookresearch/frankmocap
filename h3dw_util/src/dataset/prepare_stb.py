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
import torchgeometry as gu

# from utils.vis_utils import draw_keypoints, render_hand
# from utils.ho3d_utils.vis_utils import *
from utils import data_utils
import utils.vis_utils as vis_utils
from render.render_utils import project_joints
import check_stb as cs
from scipy.io import loadmat


def crop_image_single(img, hand_kps):
    kps = hand_kps.copy()
    ori_height, ori_width = img.shape[:2]
    min_x, min_y = np.min(kps, axis=0)
    max_x, max_y = np.max(kps, axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    if width > height:
        margin = (width-height) // 2
        min_y = max(min_y-margin, 0)
        max_y = min(max_y+margin, ori_height)
    else:
        margin = (height-width) // 2
        min_x = max(min_x-margin, 0)
        max_x = min(max_x+margin, ori_width)

    # add additoinal margin
    margin = int(0.3 * (max_y-min_y)) # if use loose crop, change 0.03 to 0.1
    min_y = max(min_y-margin, 0)
    max_y = min(max_y+margin, ori_height)
    min_x = max(min_x-margin, 0)
    max_x = min(max_x+margin, ori_width)

    # return results
    hand_kps = hand_kps - np.array([min_x, min_y]).reshape(1, 2)
    img = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
  
    return img, hand_kps


def process_stb(origin_data_dir, res_anno_dir, res_img_dir, res_vis_dir):
    cam_type = "BB"
    suffix = f"_{cam_type}.mat"
    label_dir = osp.join(origin_data_dir, "labels")
    all_files = sorted([file for file in os.listdir(label_dir) if file.endswith(suffix)])

    res_dict = dict(
        train = list(),
        val = list()
    )

    for file in all_files:
        # determine phase
        if file.startswith("B1"):
            phase = "val"
        else:
            phase = "train"
        
        # get joints for a sequence first
        joints_3d_all = loadmat(osp.join(label_dir, file))['handPara']
        joints_3d_all = cs.calc_wrist(joints_3d_all)
        joints_2d_left, _ = cs.calc_joints_2d(joints_3d_all, cam_type)
        joints_2d_left = data_utils.remap_joints(joints_2d_left, "stb", "smplx", "hand")
        # remap joints_3d is performed in normalize_joints(**)
        joints_3d_norm, scale_ratio = cs.normalize_joints(joints_3d_all)

        # left -> right
        joints_3d_norm = cs.flip_joints_3d(joints_3d_norm)

        seq_name = file.split('_')[0]
        img_dir = osp.join(origin_data_dir, f'image/{seq_name}')
        res_img_subdir = osp.join(res_img_dir, seq_name)
        res_vis_subdir = osp.join(res_vis_dir, seq_name)
        ry_utils.renew_dir(res_img_subdir)
        ry_utils.renew_dir(res_vis_subdir)

        for img_id in range(1500):
            img_name = f"BB_left_{img_id}.png"

            # write image
            img_path = osp.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img = np.fliplr(img) # left -> right

            joints_2d = joints_2d_left[img_id]
            joints_3d = joints_3d_norm[img_id]

            # left -> right
            joints_2d[:, 0] = img.shape[1]-1 - joints_2d[:, 0] 
            # center crop
            img_cropped, joints_2d_cropped = crop_image_single(img, joints_2d)

            res_img_path = osp.join(res_img_subdir, img_name)
            cv2.imwrite(res_img_path, img_cropped)

            vis_img = vis_utils.draw_keypoints(img_cropped.copy(), joints_2d_cropped)
            res_vis_img_path = osp.join(res_vis_subdir, img_name)
            cv2.imwrite(res_vis_img_path, vis_img)

            image_name = osp.join(seq_name, img_name)
            single_data = dict(
                image_root = res_img_dir,
                image_name = image_name,
                joints_2d = joints_2d_cropped,
                hand_joints_3d = joints_3d,
                scale_ratio = scale_ratio,
                augment = False,
            )

            res_dict[phase].append(single_data)

            if img_id%100 == 0:
                print(f"{seq_name} processed {img_id}")
    
    for phase in ['train', 'val']:
        res_anno_file = osp.join(res_anno_dir, f"{phase}.pkl")
        pio.save_pkl_single(res_anno_file, res_dict[phase])


def main():
    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/stb"
    origin_data_dir = osp.join(data_root, "data_original")

    res_anno_dir = osp.join(data_root, "data_processed/annotation")
    res_img_dir = osp.join(data_root, "data_processed/image")
    res_vis_dir = osp.join(data_root, "data_processed/image_anno")
    ry_utils.build_dir(res_anno_dir)
    ry_utils.renew_dir(res_img_dir)
    ry_utils.renew_dir(res_vis_dir)

    process_stb(origin_data_dir, res_anno_dir, res_img_dir, res_vis_dir)


if __name__ == '__main__':
    main()
