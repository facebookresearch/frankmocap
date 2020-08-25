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

def get_cam_param(type = 'BB'):
    if type == 'BB':

        fx = 822.79041
        fy = 822.79041
        tx = 318.47345
        ty = 250.31296
        base = 120.054

        R_l = np.zeros((3, 4))
        R_l[0, 0] = 1
        R_l[1, 1] = 1
        R_l[2, 2] = 1
        R_r = R_l.copy()
        R_r[0, 3] -= base

    else:
        assert type == 'SK'
        fx = 607.92271
        fy = 607.88192
        tx = 314.78337
        ty = 236.42484

        rvec = np.array([0.00531, -0.01196, 0.00301])
        tvec = np.array([-24.0381, -0.4563, -1.2326]).reshape(3, 1)
        # tvec = np.zeros((3,1))
        rot = cv2.Rodrigues(rvec)[0]
        cam_ext = np.concatenate((rot, tvec), axis=1)
        R_l = cam_ext
        R_r = R_l.copy()
       
    K = np.diag([fx, fy, 1.0])
    K[0, 2] = tx
    K[1, 2] = ty
    return R_l, R_r, K


def calc_joints_2d(joints_3d_all, cam_type):
    R_l, R_r, K = get_cam_param(cam_type)

    num_data = joints_3d_all.shape[2]
    num_joint = joints_3d_all.shape[1]
    joints_2d_left = np.zeros((num_data, num_joint, 2))
    joints_2d_right = np.zeros((num_data, num_joint, 2))

    for i in range(num_data):
        joints_3d = joints_3d_all[:, : , i]
        joints_3d = np.concatenate((joints_3d, np.ones((1, num_joint))), axis=0)

        joints_2d = np.matmul(np.matmul(K, R_l), joints_3d)
        joints_2d = np.true_divide(joints_2d, joints_2d[2, :].reshape(1, -1)).T
        joints_2d_left[i] = joints_2d[:, :2]

        joints_2d = np.matmul(np.matmul(K, R_r), joints_3d)
        joints_2d = np.true_divide(joints_2d, joints_2d[2, :].reshape(1, -1)).T
        joints_2d_right[i] = joints_2d[:, :2]
    
    return joints_2d_left, joints_2d_right


def calc_wrist(joints_3d_all):
    palm = joints_3d_all[:, 0, :]
    index_root = joints_3d_all[:, 13, :]
    middle_root = joints_3d_all[:, 9, :]
    ring_root = joints_3d_all[:, 5, :]
    middle_root = ((index_root + ring_root)/2 + middle_root) / 2
    vec = palm - middle_root
    wrist = palm + vec * 1.1
    return np.concatenate((joints_3d_all, wrist.reshape(3, 1, 1500)), axis=1)


def normalize_joints(joints_3d_all):
    num_sample= joints_3d_all.shape[2]

    R_l, R_r, K = get_cam_param()
    joints_batch = list()
    for i in range(num_sample):
        joints_3d = joints_3d_all[:,:,i].T
        joints_3d = data_utils.remap_joints(joints_3d, 'stb', 'smplx', 'hand')
        joints_batch.append(joints_3d)
    joints_batch = np.array(joints_batch)
    joints_3d_norm, scale_ratio = nju.normalize_joints_to_smplx(joints_batch)
    return joints_3d_norm, scale_ratio


def flip_joints_3d(joints_3d_origin):
    joints_3d_flipped = np.zeros(joints_3d_origin.shape)
    for i in range(joints_3d_origin.shape[0]):
        joints_3d = joints_3d_origin[i]
        joints_3d_flipped[i] = np.matmul(np.diag([-1, 1, 1]), joints_3d.T).T
    return joints_3d_flipped


def anno_joints_2d(root_dir, label_dir, vis_dir, file, cam_type):
    assert cam_type == 'BB' # only support 'BB' right now

    joints_3d_all = loadmat(osp.join(label_dir, file))['handPara']
    # joints_3d_all = calc_wrist(joints_3d_all)
    joints_2d_left, joints_2d_right = calc_joints_2d(joints_3d_all, cam_type)

    joints_3d_norm = normalize_joints(joints_3d_all)
    # left -> right
    joints_3d_norm = flip_joints_3d(joints_3d_norm)

    seq_name = file.split('_')[0]
    img_dir = osp.join(root_dir, f'image/{seq_name}')
    res_dir = osp.join(vis_dir, seq_name)
    ry_utils.renew_dir(res_dir)

    for i in range(1500):
        # load image
        if cam_type == 'SK':
            img_name = f"SK_color_{i}.png"
        else:
            assert cam_type == "BB"
            img_name = f"BB_left_{i}.png"

        # write image
        img_path = osp.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = np.fliplr(img) # left -> right

        joints_2d = joints_2d_left[i]
        joints_2d[:, 0] = img.shape[1]-1 - joints_2d[:, 0] # left -> right
        res_img = vu.draw_keypoints(img.copy(), joints_2d)

        # check normalized joints
        joints_3d = joints_3d_norm[i]
        cam = np.array([2.95, 0.2, 0.0])
        joints_2d = project_joints(joints_3d, cam)
        joints_2d = (joints_2d+1.0) * 0.5 * np.min(img.shape[:2])
        res_img = vis_utils.draw_keypoints(res_img.copy(), joints_2d, color=(255, 0 ,0))
        cv2.imwrite(osp.join(res_dir, f"{i:04d}_left.png"), res_img)


def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/stb/data_original'

    vis_dir = osp.join("visualization/stb")
    ry_utils.renew_dir(vis_dir)

    label_dir = osp.join(root_dir, 'labels')
    all_files = sorted(os.listdir(label_dir))

    cam_type = 'BB'

    for file in all_files:
        if file.endswith(f"_{cam_type}.mat"):
            anno_joints_2d(root_dir, label_dir, vis_dir, file, cam_type)

    '''
    pool = mp.Pool(10)
    results = list()
    for file in all_files:
        if file.endswith(f"_{cam_type}.mat"):
            results.append(pool.apply_async(anno_joints_2d, args=(root_dir, label_dir, vis_dir, file, cam_type)))
    pool.close()
    pool.join()
    for result in results:
        result.get()
    '''
            

if __name__ == '__main__':
    main()