import sys
sys.path.append('src')
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
from scipy.io import loadmat
import multiprocessing as mp
import utils.normalize_joints_utils as nju
from utils.render_utils import project_joints
import utils.vis_utils as vis_utils
import utils.data_utils as data_utils
import check_rhd as cr

def crop_image_single(img, hand_kps):
    hand_vis = hand_kps[:, 2:3]
    hand_kps = hand_kps[:, :2]
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
    margin = int(0.3 * (max_y-min_y)) # if use loose crop, change 0.03 to 0.1
    min_y = max(min_y-margin, 0)
    max_y = min(max_y+margin, ori_height)
    min_x = max(min_x-margin, 0)
    max_x = min(max_x+margin, ori_width)

    # return results
    hand_kps = hand_kps - np.array([min_x, min_y]).reshape(1, 2)
    hand_kps = np.concatenate((hand_kps, hand_vis), axis=1)
    img = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
  
    return img, hand_kps


def process_rhd(root_dir, res_anno_dir, res_img_dir, res_vis_dir):
    
    for phase in ['training', 'evaluation']:
    # for phase in ['evaluation']:
        anno_file = osp.join(root_dir, phase, f"anno_{phase}.pickle")
        anno_all = pio.load_pkl_single(anno_file)

        extract_data_file = osp.join(root_dir, phase, f"extract_data_{phase}.pkl")
        extract_data = pio.load_pkl_single(extract_data_file)

        res_data = list()
        for sample_id, anno in anno_all.items():
            # load data
            img_path = osp.join(root_dir, phase, 'color', f'{sample_id:05d}.png')
            img = cv2.imread(img_path)

            # get info from annotation dictionary
            joints_2d = anno['uv_vis'][:, :2] # u, v coordinates of 42 hand keypoints, pixel
            joints_3d = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
            hand_side = extract_data[sample_id]['hand_side']
            joints_vis = extract_data[sample_id]['keypoint_vis']
            joints_vis = joints_vis.astype(np.int32)
            # joints_vis[joints_vis<1e-8] = -1

            print(joints_vis)

            continue
            
            if hand_side[0][0] == 1:
                start, end = 0, 21
            else:
                assert hand_side[0][1] == 1
                start, end = 21, 21*2
            joints_2d = joints_2d[start:end, :]
            joints_2d = np.concatenate([joints_2d, joints_vis.T], axis=1)
            joints_3d = joints_3d[start:end, :]
             # flip left to right
            if hand_side[0][0] == 1:
                img = np.fliplr(img)
                joints_2d[:, 0] = img.shape[1]-1 - joints_2d[:, 0]
                rot_mat = np.diag([-1, 1, 1])
                joints_3d = np.matmul(rot_mat, joints_3d.T).T

            joints_2d = data_utils.remap_joints(joints_2d, "rhd", "smplx", "hand")
            joints_3d = data_utils.remap_joints(joints_3d, "rhd", "smplx", "hand")
            joints_3d, scale_ratio = cr.normalize_joints(joints_3d)

            img_cropped, joints_2d_cropped = crop_image_single(img, joints_2d)

            res_subdir = osp.join(res_img_dir, phase)
            ry_utils.build_dir(res_subdir)
            img_name = osp.join(phase, f'{sample_id:05d}.png')
            res_img_path = osp.join(res_img_dir, img_name)
            cv2.imwrite(res_img_path, img_cropped)

            res_subdir = osp.join(res_vis_dir, phase)
            ry_utils.build_dir(res_subdir)
            res_vis_path = osp.join(res_vis_dir, img_name)
            vis_img = vis_utils.draw_keypoints(img_cropped.copy(), joints_2d_cropped)
            cv2.imwrite(res_vis_path, vis_img)

            res_data.append(dict(
                image_root = res_img_dir,
                image_name = img_name,
                joints_2d = joints_2d_cropped,
                hand_joints_3d = joints_3d,
                scale_ratio = scale_ratio,
                augment = False,
            ))

            if sample_id % 100 == 0:
                print(f"{phase} processed {sample_id}")

            '''
            cam = np.array([2.95, 0.1, 0.0])
            joints_2d_proj = project_joints(joints_3d, cam)
            joints_2d_proj = (joints_2d_proj+1.0) * 0.5 * np.min(img_cropped.shape[:2])
            vis_img = vis_utils.draw_keypoints(img_cropped.copy(), joints_2d_proj, color=(255, 0 ,0), radius=3)
            for i in range(joints_2d.shape[0]):
                res_img = vis_utils.draw_keypoints(vis_img.copy(), joints_2d_cropped[i:i+1, :])
                res_img = vis_utils.draw_keypoints(res_img.copy(), joints_2d_proj[i:i+1, :], color=(0,255,0))
                cv2.imwrite(f"{i:02d}.png", res_img)
            sys.exit(0)
            '''
        res_phase = 'val' if phase.find("evaluation")>=0 else "train"
        res_anno_file = osp.join(res_anno_dir, f"{res_phase}.pkl")
        pio.save_pkl_single(res_anno_file, res_data)
           

def main():
    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/rhd"
    origin_data_dir = osp.join(data_root, "data_original")

    res_anno_dir = osp.join(data_root, "data_processed/annotation")
    res_img_dir = osp.join(data_root, "data_processed/image")
    res_vis_dir = osp.join(data_root, "data_processed/image_anno")
    ry_utils.build_dir(res_anno_dir)
    ry_utils.build_dir(res_img_dir)
    ry_utils.build_dir(res_vis_dir)

    process_rhd(origin_data_dir, res_anno_dir, res_img_dir, res_vis_dir)


if __name__ == '__main__':
    main()