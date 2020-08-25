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
    for img_id, img_path in enumerate(all_imgs):
        json_file = img_path.replace('.jpg', '.json')
        with open(json_file, 'r') as in_f:
            anno_data = json.load(in_f)
            is_mpii = anno_data['is_mpii']
            hand_joints_2d = np.array(anno_data['hand_pts'])
            body_joints_2d = np.array(anno_data['mpii_body_pts'])
            img = cv2.imread(img_path)
            width = img.shape[1]

            is_left = anno_data['is_left']
            if is_left:
                img = np.fliplr(img).astype(np.uint8)
                width = img.shape[1]
                hand_joints_2d[:, 0] = width-1 - hand_joints_2d[:, 0]
                body_joints_2d[:, 0] = width-1 - body_joints_2d[:, 0]

            img_name = img_path.split('/')[-1]
            raw_data.append((img_name, img, body_joints_2d, hand_joints_2d, is_mpii))

    return raw_data


'''
def anno_data(raw_data, res_vis_dir):
    for img_name, img, body_joints_2d, hand_joints_2d, joints_2d, _ in raw_data:
        res_subdir = osp.join(res_vis_dir, img_name.split('.')[0])
        ry_utils.build_dir(res_subdir)
        num_joints = joints_2d.shape[0]
        for j_id in range(num_joints):
            res_img_path = osp.join(res_subdir, f'{j_id:02d}.jpg')
            vis_img = vis_utils.draw_keypoints(img, joints_2d[j_id:j_id+1, :], radius=3)
            cv2.imwrite(res_img_path, vis_img)
'''


def crop_hand_single(img, body_joints_2d, hand_joints_2d):
    num_body_joints = body_joints_2d.shape[0]
    joints_2d = np.concatenate((body_joints_2d, hand_joints_2d), axis=0)

    kps_vis = joints_2d[:, 2:3]
    kps = joints_2d[:, :2]

    ori_height, ori_width = img.shape[:2]
    min_x, min_y = np.min(kps, axis=0)
    max_x, max_y = np.max(kps, axis=0)

    width = max_x - min_x
    height = max_y - min_y
    margin = int(max(height, width) * 0.3)

    min_y = max(min_y-margin, 0)
    max_y = min(max_y+margin, ori_height)
    min_x = max(min_x-margin, 0)
    max_x = min(max_x+margin, ori_width)

    kps = kps - np.array([min_x, min_y]).reshape(1, 2)
    kps = np.concatenate((kps, kps_vis), axis=1)
    body_kps = kps[:num_body_joints, :]
    hand_kps = kps[num_body_joints:, :]
    img = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
    bbox = np.array([min_x, min_y, max_x, max_y], dtype=np.int32)

    return img, body_kps, hand_kps, bbox


def prepare_panoptic_hand(root_dir, res_img_root, res_anno_dir, res_vis_root):
    for phase in ['val']:
    # for phase in ['train', 'val']:
        # load raw_data
        phase_dir = osp.join(root_dir, 'data_original', f'manual_{phase}')
        raw_data = load_raw_data(phase_dir)

        # make dirs
        res_img_dir = osp.join(res_img_root, phase)
        ry_utils.renew_dir(res_img_dir)
        res_vis_dir = osp.join(res_vis_root, phase)
        ry_utils.renew_dir(res_vis_dir)

        res_data = list()
        for img_name, img, body_joints_2d, hand_joints_2d, is_mpii in raw_data:
            crop_img, crop_body_joints_2d, crop_hand_joints_2d, bbox = \
                 crop_hand_single(img, body_joints_2d, hand_joints_2d)

            '''
            crop_img = img
            crop_body_joints_2d = body_joints_2d
            crop_hand_joints_2d = hand_joints_2d
            '''

            res_img_path = osp.join(res_img_dir, img_name)
            cv2.imwrite(res_img_path, crop_img)

            vis_img_path = osp.join(res_vis_dir, img_name)
            crop_body_joints_2d[2] = 1
            vis_img = vis_utils.draw_keypoints(crop_img, crop_hand_joints_2d, color=(255, 0, 0))
            vis_img = vis_utils.draw_keypoints(vis_img, crop_body_joints_2d, color=(0, 0, 255))
            # vis_img = cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=5)
            cv2.imwrite(vis_img_path, vis_img)

            res_data.append(dict(
                image_name = osp.join(phase, img_name),
                body_joints_2d = crop_body_joints_2d,
                hand_joints_2d = crop_hand_joints_2d,
                is_mpii = is_mpii,
                augment = False
            ))
        
        res_anno_file = osp.join(res_anno_dir, f'{phase}.pkl')
        pio.save_pkl_single(res_anno_file, res_data)


def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/panoptic_hand/'
    res_img_dir = osp.join(root_dir, "data_processed_body/image")
    res_anno_dir = osp.join(root_dir, "data_processed_body/annotation")
    res_vis_dir = osp.join(root_dir, "data_processed_body/image_anno")
    ry_utils.renew_dir(res_img_dir)
    ry_utils.renew_dir(res_anno_dir)
    ry_utils.renew_dir(res_vis_dir)

    prepare_panoptic_hand(root_dir, res_img_dir, res_anno_dir, res_vis_dir)


if __name__ == '__main__':
    main()