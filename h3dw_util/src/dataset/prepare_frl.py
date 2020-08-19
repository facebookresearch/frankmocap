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
from collections import OrderedDict
import multiprocessing as mp
import time

def load_img(img_dir):
    all_imgs = dict()
    for subdir, dirs, files in os.walk(img_dir, followlinks=True):
        for file in files:
            if file.endswith(".png"):
                img_id = int(file.replace("image",'')[:-4])
                cam_id = int(subdir.split('/')[-1][3:])
                key = f"{cam_id:06d}_{img_id:04d}"
                img_path = osp.join(subdir, file)
                all_imgs[key] = img_path
    all_imgs_sorted = OrderedDict()
    sorted_keys = sorted(list(all_imgs.keys()))
    for key in sorted_keys:
        all_imgs_sorted[key] = all_imgs[key]
    return all_imgs_sorted


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
    all_seq = os.listdir(root_dir)
    for seq_id, seq_name in enumerate(all_seq):
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
        print(f"Load {seq_name}, {seq_id+1}/{len(all_seq)}")
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


def crop_image_single(img, hand_kps, final_size):
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
    img = img[int(min_y):int(max_y), int(min_x):int(max_x), :]

    height, width = img.shape[:2]
    if height > width:
        new_height = final_size
        scale_ratio = new_height/height
        new_width = int(scale_ratio * width)
    else:
        new_width = final_size
        scale_ratio = new_width / width
        new_height = int(scale_ratio * height)
    img = cv2.resize(img, (new_height, new_width))
    hand_kps *= scale_ratio
  
    return img, hand_kps


def visualize_anno(res_vis_subdir, img_cropped, joints_2d_cropped, joints_3d_norm):
    res_cam_subdir = osp.join(res_vis_subdir, f"cam{cam_id:06d}")
    res_anno_subdir = osp.join(res_cam_subdir, f"image{img_id:04d}")
    ry_utils.build_dir(res_anno_subdir)

    cam = np.array([4.95, -0.1, -0.1])
    joints_2d_norm = project_joints(joints_3d_norm, cam)
    joints_2d_norm = (joints_2d_norm+1.0) * 0.5 * np.min(img_cropped.shape[:2])
    joint_img = vis_utils.draw_keypoints(img_cropped.copy(), joints_2d_norm, color=(255, 0 ,0), radius=3)
    for i in range(joints_2d_cropped.shape[0]):
        res_img = vis_utils.draw_keypoints(joint_img.copy(), joints_2d_cropped[i:i+1, :], radius=5)
        res_img = vis_utils.draw_keypoints(res_img.copy(), joints_2d_norm[i:i+1, :], color=(0,255,0), radius=2)
        res_img_path = osp.join(res_anno_subdir, f"{i:02d}.png")
        cv2.imwrite(res_img_path, res_img)


def process_frl_single(process_id, seq_name_list, all_data, res_anno_dir, res_img_dir, res_vis_dir):
    res_data = list()
    start_time = time.time()

    for seq_id, seq_name in enumerate(seq_name_list):

        res_img_subdir = osp.join(res_img_dir, seq_name)
        ry_utils.renew_dir(res_img_subdir)
        res_vis_subdir = osp.join(res_vis_dir, seq_name)
        ry_utils.renew_dir(res_vis_subdir)

        all_imgs = all_data[seq_name]['imgs']
        all_joints_3d = all_data[seq_name]['joints_3d']
        all_cam_params = all_data[seq_name]['cam_params']

        if seq_name.startswith("Left"):
            hand_type = "left"
        else:
            assert seq_name.startswith("Right")
            hand_type = "right"

        sample_ratio = 5
        for key_id, key in enumerate(all_imgs):
            if key_id % sample_ratio != 0: continue

            record = key.split('_')
            cam_id = int(record[0])
            img_id = int(record[1])

            # 3D joints and 2D joints
            joints_3d = all_joints_3d[f"{img_id:04d}"]
            cam_params = all_cam_params[f"{cam_id:06d}"]
            joints_2d = calc_joints_2d(joints_3d, cam_params)
            joints_2d = data_utils.remap_joints(joints_2d, "frl", "smplx", "hand")
            joints_3d_norm, scale_ratio = normalize_joints(joints_3d, cam_params)

            # image
            img_path = all_imgs[key]
            img = cv2.imread(img_path)

            # crop image
            final_size = 256
            img_cropped, joints_2d_cropped = crop_image_single(img, joints_2d, final_size)

            # flip left hand to right
            if hand_type == 'left':
                img_cropped = np.fliplr(img_cropped)
                joints_2d_cropped[:, 0] = img_cropped.shape[1]-1 - joints_2d_cropped[:, 0]
                rot_mat = np.diag([-1, 1, 1])
                joints_3d_norm = np.matmul(rot_mat, joints_3d_norm.T).T

            # save results
            res_cam_subdir = osp.join(res_img_subdir, f"cam{cam_id:06d}")
            ry_utils.build_dir(res_cam_subdir)
            res_img_path = osp.join(res_cam_subdir,  f"image{img_id:04d}.jpg")
            cv2.imwrite(res_img_path, img_cropped)

            img_name = res_img_path.replace(res_img_dir, '')[1:]
            res_data.append(dict(
                image_root = res_img_dir,
                image_name = img_name,
                joints_2d = joints_2d_cropped,
                hand_joints_3d = joints_3d_norm,
                scale_ratio = scale_ratio,
                augment = False,
            ))

        speed = (seq_id+1) / (time.time() - start_time)
        remain_time = (len(seq_name_list)-seq_id-1) / speed / 60
        print(f"Process-{process_id:03d}: {seq_name} completes, {seq_id+1}/{len(seq_name_list)},"
              f"remain requires {remain_time} mins or {remain_time/60.0} hours")

    res_anno_file = osp.join(res_anno_dir, f"{process_id:03d}.pkl")
    pio.save_pkl_single(res_anno_file, res_data)


def process_frl(all_data, res_anno_dir, res_img_dir, res_vis_dir):
    seq_name_list = list(all_data.keys())
    num_process = min(64, len(all_data))
    num_each = len(seq_name_list) // num_process
    p_list = list()
    for i in range(num_process):
        start = i * num_each
        end = (i+1)*num_each if i<num_process-1 else len(seq_name_list)
        p = mp.Process(target=process_frl_single, 
            args=(i, seq_name_list[start:end], all_data, res_anno_dir, res_img_dir, res_vis_dir))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()


def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/FRL_data'
    origin_data_dir = osp.join(root_dir, 'sample_data')
    all_data = load_data(origin_data_dir)

    res_anno_dir = osp.join(root_dir, "data_processed/annotation")
    res_img_dir = osp.join(root_dir, "data_processed/image")
    res_vis_dir = osp.join(root_dir, "data_processed/image_anno")
    ry_utils.build_dir(res_anno_dir)
    ry_utils.renew_dir(res_img_dir)
    # ry_utils.renew_dir(res_vis_dir)

    process_frl(all_data, res_anno_dir, res_img_dir, res_vis_dir)


if __name__ == '__main__':
    main()