import sys
sys.path.append("src")
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import random
import pdb
import torch
import ry_utils
import parallel_io as pio
import utils.vis_utils as vu
import json
from scipy.io import loadmat
import multiprocessing as mp
import utils.normalize_joints_utils as nju
import utils.vis_utils as vis_utils
import utils.data_utils as data_utils


def load_raw_data(in_dir):
    raw_data = list()
    all_imgs = ry_utils.get_all_files(in_dir, '.jpg', 'full')
    for img_path in all_imgs:
        json_file = img_path.replace('.jpg', '.json')
        with open(json_file, 'r') as in_f:
            anno_data = json.load(in_f)
            is_mpii = anno_data['is_mpii']
            # if is_mpii:
            hand_joints_2d = np.array(anno_data['hand_pts'])
            img = cv2.imread(img_path)
            width = img.shape[1]

            is_left = anno_data['is_left']
            if is_left:
                img = np.fliplr(img).astype(np.uint8)
                width = img.shape[1]
                hand_joints_2d[:, 0] = width-1 - hand_joints_2d[:, 0]

            img_name = img_path.split('/')[-1]
            raw_data.append((img_name, img, hand_joints_2d, is_mpii))
    return raw_data


def anno_data(raw_data, res_vis_dir):
    for img_name, img, joints_2d in raw_data:
        res_subdir = osp.join(res_vis_dir, img_name.split('.')[0])
        ry_utils.build_dir(res_subdir)
        num_joints = joints_2d.shape[0]
        for j_id in range(num_joints):
            res_img_path = osp.join(res_subdir, f'{j_id:02d}.jpg')
            vis_img = vis_utils.draw_keypoints(img, joints_2d[j_id:j_id+1, :], radius=3)
            cv2.imwrite(res_img_path, vis_img)


def crop_hand_single(img, joints_2d):
    hand_vis = joints_2d[:, 2:3]
    hand_kps = joints_2d[:, :2]
    kps = hand_kps

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
    margin = int(0.4 * (max_y-min_y)) # if use loose crop, change 0.03 to 0.1
    min_y = max(min_y-margin, 0)
    max_y = min(max_y+margin, ori_height)
    min_x = max(min_x-margin, 0)
    max_x = min(max_x+margin, ori_width)

    # return results
    hand_kps = hand_kps - np.array([min_x, min_y]).reshape(1, 2)
    hand_kps = np.concatenate((hand_kps, hand_vis), axis=1)
    img = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
  
    return img, hand_kps



def prepare_panoptic_hand(root_dir, res_img_root, res_anno_dir, res_vis_root):
    # for phase in ['val']:
    for phase in ['train', 'val']:
        # load raw_data
        phase_dir = osp.join(root_dir, 'data_original', f'manual_{phase}')
        raw_data = load_raw_data(phase_dir)

        # make dirs
        res_img_dir = osp.join(res_img_root, phase)
        ry_utils.renew_dir(res_img_dir)
        res_vis_dir = osp.join(res_vis_root, phase)
        ry_utils.renew_dir(res_vis_dir)

        res_data = list()
        for img_name, img, joints_2d, is_mpii in raw_data:
            crop_img, crop_joints_2d = crop_hand_single(img, joints_2d)
            crop_joints_2d = data_utils.remap_joints_hand(crop_joints_2d, "pmhand", "smplx")

            res_img_path = osp.join(res_img_dir, img_name)
            cv2.imwrite(res_img_path, crop_img)

            vis_img_path = osp.join(res_vis_dir, img_name)
            vis_img = vis_utils.draw_keypoints(crop_img, crop_joints_2d)
            cv2.imwrite(vis_img_path, vis_img)

            res_data.append(dict(
                image_name = osp.join(phase, img_name),
                joints_2d = crop_joints_2d,
                is_mpii = is_mpii,
                augment = False
            ))
        
        res_anno_file = osp.join(res_anno_dir, f'{phase}.pkl')
        pio.save_pkl_single(res_anno_file, res_data)


def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/panoptic_hand/'
    res_img_dir = osp.join(root_dir, "data_processed/image")
    res_anno_dir = osp.join(root_dir, "data_processed/annotation")
    res_vis_dir = osp.join(root_dir, "data_processed/image_anno")
    ry_utils.renew_dir(res_img_dir)
    ry_utils.renew_dir(res_anno_dir)
    ry_utils.renew_dir(res_vis_dir)

    prepare_panoptic_hand(root_dir, res_img_dir, res_anno_dir, res_vis_dir)


if __name__ == '__main__':
    main()