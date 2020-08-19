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


def normalize_joints(joints_3d):
    num_joints = joints_3d.shape[0]
    joints_batch = joints_3d.reshape(1, num_joints, 3)
    joints_3d_norm, scale_ratio = nju.normalize_joints_to_smplx(joints_batch)
    return joints_3d_norm[0], scale_ratio

# iterate samples of the set
def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/rhd/data_original'
    vis_dir = "visualization/rhd"
    ry_utils.renew_dir(vis_dir)

    # for data_type in ['evaluation', 'training']:
    # for data_type in ['evaluation']:
    for data_type in ['training']:
        anno_file = osp.join(root_dir, data_type, f"anno_{data_type}.pickle")
        anno_all = pio.load_pkl_single(anno_file)
        res_dir = osp.join(vis_dir, data_type)
        ry_utils.renew_dir(res_dir)

        extract_data_file = osp.join(root_dir, data_type, f"extract_data_{data_type}.pkl")
        extract_data = pio.load_pkl_single(extract_data_file)

        left_valid, right_valid = 0, 0
        num_data = 0

        for sample_id, anno in anno_all.items():
            # load data
            img_path = osp.join(root_dir, data_type, 'color', f'{sample_id:05d}.png')
            img = cv2.imread(img_path)

            # get info from annotation dictionary
            joints_2d = anno['uv_vis'][:, :2] # u, v coordinates of 42 hand keypoints, pixel
            joints_3d = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
            
            hand_side = extract_data[sample_id]['hand_side']
            joints_vis = extract_data[sample_id]['keypoint_vis']
            
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

            # normalize joints
            joints_3d_norm, scale_ratio = normalize_joints(joints_3d)

            joint_img = vu.draw_keypoints(img.copy(), joints_2d, radius=3)

            cam = np.array([2.95, 0.1, 0.0])
            joints_2d = project_joints(joints_3d_norm, cam)
            joints_2d = (joints_2d+1.0) * 0.5 * np.min(img.shape[:2])
            res_img = vis_utils.draw_keypoints(joint_img.copy(), joints_2d, color=(255, 0 ,0), radius=3)
            res_img_path = osp.join(res_dir, f"{sample_id:05d}.png")
            cv2.imwrite(res_img_path, res_img)

            if sample_id > 10:
                break


if __name__ == '__main__':
    main()