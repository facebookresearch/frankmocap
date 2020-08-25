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
from utils import data_utils
import utils.vis_utils as vis_utils
from render.render_utils import project_joints


def load_img(img_dir):
    all_imgs = dict()
    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(".png"):
                img_id = int(file.replace("image",'')[:-4])
                cam_id = int(subdir.split('/')[-1][3:])
                key = f"{cam_id:06d}_{img_id:04d}"
                img_path = osp.join(subdir, file)
                all_imgs[key] = img_path
    return all_imgs


def load_joints_3d(joints_dir, hand_type):
    all_joints = dict()
    for file in os.listdir(joints_dir):
        if file.endswith(".pts"):
            file_path = osp.join(joints_dir, file)
            img_id = int(file.replace("image",'')[:-4])
            key = f"{img_id:04d}"
            all_joints[key] = np.zeros((21, 3))
            with open(file_path, 'r') as in_f:
                valid_joint_num = 0
                for line in in_f:
                    record = line.strip().split()
                    # joint id
                    joint_id = int(record[0])
                    if hand_type == 'left':
                        if joint_id < 21: continue
                        joint_id = joint_id-21
                    else:
                        if joint_id >= 21:
                            continue
                    # joints 3d
                    joints_single = np.array(list(map(float, record[1:4])))
                    all_joints[key][joint_id] = joints_single
                    valid_joint_num += 1
                assert valid_joint_num == 21, "The number of joints should be 21"
    return all_joints


def load_cam_params(cam_file):
    all_cam_params = dict()
    with open(cam_file, 'r') as in_f:
        line = in_f.readline()
        while line != '':
            # cam id
            record = line.strip().split()
            cam_id = int(record[0])
            key = f"{cam_id:06d}"

            # intrinsic
            K = np.zeros((3,3))
            for i in range(3):
               K[i] = np.array(list(map(float, in_f.readline().strip().split()))) 
            in_f.readline() # skip the redundant line

            # extrinsic
            R = np.zeros((3,4))
            for i in range(3):
               R[i] = np.array(list(map(float, in_f.readline().strip().split()))) 

            all_cam_params[key] = dict(
                K = K,
                R = R,
            )
            # next param
            # skip space
            in_f.readline()
            line = in_f.readline()

    return all_cam_params


def load_data(root_dir):
    all_data = dict()
    for seq_name in os.listdir(root_dir):
        if seq_name == '.DS_Store': continue
        subdir = osp.join(root_dir, seq_name)

        # We only consider single hand now
        if not seq_name.startswith( ("Right", "Left") ): continue
        hand_type = "left" if seq_name.startswith("Left") else "right"

        img_dir = osp.join(subdir, "images")
        joints_dir = osp.join(subdir, "keypoints/3D_rot")
        camera_file = osp.join(subdir, "KRT2")

        all_imgs = load_img(img_dir)
        all_joints_3d = load_joints_3d(joints_dir, hand_type)
        all_cam_params = load_cam_params(camera_file)

        all_data[seq_name] = dict(
            imgs = all_imgs,
            joints_3d = all_joints_3d,
            cam_params = all_cam_params
        )
    return all_data


def calc_joints_2d(joints_3d, cam_params):
    K = cam_params['K']
    R = cam_params['R']

    # joints_3d: (21, 3)
    num_joint = joints_3d.shape[0]

    joints_3d = np.concatenate((joints_3d.T, np.ones((1, num_joint))), axis=0)
    joints_2d = np.matmul(np.matmul(K, R), joints_3d)
    joints_2d = np.true_divide(joints_2d, joints_2d[2, :].reshape(1, -1)).T
    return joints_2d[:, :2]


def normalize_joints(joints_3d, cam_params):
    # remap joints to smplx format
    joints_3d = data_utils.remap_joints(joints_3d, 'frl', 'smplx', 'hand')
    num_joint = joints_3d.shape[0]


    extrinsic = cam_params['R']
    R = extrinsic[:3, :3]
    T = extrinsic[:, 3]
    joints_3d_1 = np.dot(R, joints_3d.T) + T.reshape(3, 1)
    joints_3d_2 = np.matmul(R, joints_3d.T) + T.reshape(3, 1)
    joints_3d_3 = np.concatenate((joints_3d.T, np.ones((1, num_joint))), axis=0)
    joints_3d_3 = np.matmul(extrinsic, joints_3d_3)

    joints_3d = joints_3d_3.T

    joints_3d_batch = joints_3d.reshape(1, num_joint, 3)

    joints_3d_norm, scale_ratio = nju.normalize_joints_to_smplx(joints_3d_batch)
    return joints_3d_norm[0], scale_ratio


def anno_2d_joints(all_data, res_dir):
    for seq_name in all_data:
        res_subdir = osp.join(res_dir, seq_name)
        ry_utils.renew_dir(res_subdir)

        all_imgs = all_data[seq_name]['imgs']
        all_joints_3d = all_data[seq_name]['joints_3d']
        all_cam_params = all_data[seq_name]['cam_params']

        for key in all_imgs:
            record = key.split('_')
            cam_id = int(record[0])
            img_id = int(record[1])

            img_path = all_imgs[key]
            joints_3d = all_joints_3d[f"{img_id:04d}"]
            cam_params = all_cam_params[f"{cam_id:06d}"]
            joints_2d = calc_joints_2d(joints_3d, cam_params)
            joints_2d = data_utils.remap_joints(joints_2d, "frl", "smplx", "hand")

            img = cv2.imread(img_path)
            # joint_img = vis_utils.draw_keypoints(img.copy(), joints_2d, radius=30)

            joints_3d_norm, scale_ratio = normalize_joints(joints_3d, cam_params)
            cam = np.array([3.95, 0.1, 0.4])
            joints_2d_norm = project_joints(joints_3d_norm, cam)
            joints_2d_norm = (joints_2d_norm+1.0) * 0.5 * np.min(img.shape[:2])
            joint_img = vis_utils.draw_keypoints(img.copy(), joints_2d_norm, color=(255, 0 ,0), radius=30)

            res_cam_subdir = osp.join(res_subdir, f"cam{cam_id:06d}")
            res_img_subdir = osp.join(res_cam_subdir,  f"image{img_id:04d}")
            ry_utils.build_dir(res_img_subdir)
            for i in range(joints_2d.shape[0]):
                res_img = vis_utils.draw_keypoints(joint_img.copy(), joints_2d[i:i+1, :], radius=30)
                res_img = vis_utils.draw_keypoints(res_img.copy(), joints_2d_norm[i:i+1, :], color=(0,255,0), radius=20)
                res_img_path = osp.join(res_img_subdir, f"{i:02d}.png")
                cv2.imwrite(res_img_path, res_img)


def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/FRL_data/sample_data'
    all_data = load_data(root_dir)

    vis_dir = "visualization/FRL"
    ry_utils.renew_dir(vis_dir)
    anno_2d_joints(all_data, vis_dir)


if __name__ == '__main__':
    main()