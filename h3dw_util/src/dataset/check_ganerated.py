import sys
assert sys.version_info > (3, 0)
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
from render.render_utils import project_joints
import utils.vis_utils as vis_utils
import utils.data_utils as data_utils


def normalize_joints(joints_3d):
    num_joints = joints_3d.shape[0]
    joints_batch = joints_3d.reshape(1, num_joints, 3)
    joints_3d_norm, scale_ratio = nju.normalize_joints_to_smplx(joints_batch)
    return joints_3d_norm[0], scale_ratio


def load_data(root_dir):
    all_data = list()
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png"):
                # get sample id
                sample_id = file.split('_')[0]
                img_path = osp.join(subdir, file)
                # 2D joints
                joints_2d_file = osp.join(subdir, f"{sample_id}_joint2D.txt")
                with open(joints_2d_file, 'r') as in_f:
                    joints_2d = np.array(list(map(float, in_f.readline().strip().split(',')))).reshape(-1, 2)
                # 3D joints
                joints_3d_file = osp.join(subdir, f"{sample_id}_joint_pos.txt")
                with open(joints_3d_file, 'r') as in_f:
                    joints_3d = np.array(list(map(float, in_f.readline().strip().split(',')))).reshape(-1, 3)
                all_data.append(
                    dict(
                        img_path = img_path,
                        joints_2d = joints_2d,
                        joints_3d = joints_3d
                    )
                )
    return all_data

def vis_joints_3d_norm(vis_img, joints_3d_norm):
    cam = np.array([4.95, 0.0, 0.0])
    joints_2d_norm = project_joints(joints_3d_norm, cam)
    joints_2d_norm = (joints_2d_norm+1.0) * 0.5 * np.min(vis_img.shape[:2])
    joint_img = vis_utils.draw_keypoints(vis_img.copy(), joints_2d_norm, color=(255, 0 ,0), radius=3)
    return joint_img


def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/ganerated/data_original/data'
    all_data = load_data(root_dir)

    vis_dir = "visualization/GANerated"
    ry_utils.renew_dir(vis_dir)

    for single_data in all_data:
        img_path = single_data['img_path']
        img = cv2.imread(img_path)

        joints_2d = single_data['joints_2d']
        joints_2d = data_utils.remap_joints(joints_2d, "ganerated", "smplx", "hand")

        joints_3d = single_data['joints_3d']
        joints_3d = data_utils.remap_joints(joints_3d, "ganerated", "smplx", "hand")
        
        # left -> right
        img = np.fliplr(img)
        joints_2d[:, 0] = img.shape[1]-1 - joints_2d[:, 0]
        rot_mat = np.diag([-1, 1, 1])
        joints_3d = np.matmul(rot_mat, joints_3d.T).T

        # normalize
        joints_3d_norm, scale_ratio = nju.normalize_joints_to_smplx(joints_3d.reshape(1, 21, 3))
        joints_3d_norm = joints_3d_norm[0]

        # visualize
        vis_img = vis_utils.draw_keypoints(img.copy(), joints_2d)
        joint_img = vis_joints_3d_norm(vis_img, joints_3d_norm)
        
        # save to image
        res_img_path = img_path.replace(root_dir, vis_dir)
        res_subdir = '/'.join(res_img_path.split('/')[:-1])
        ry_utils.build_dir(res_subdir)
        cv2.imwrite(res_img_path, joint_img)
        '''
        res_subdir = img_path.replace(root_dir, vis_dir)[:-4]
        ry_utils.renew_dir(res_subdir)
        for i in range(joints_2d.shape[0]):
            vis_img = vis_utils.draw_keypoints(img.copy(), joints_2d[i:i+1, :])
            res_img_path = osp.join(res_subdir, f"{i:02d}.png")
            cv2.imwrite(res_img_path, vis_img)
        '''


if __name__ == '__main__':
    main()