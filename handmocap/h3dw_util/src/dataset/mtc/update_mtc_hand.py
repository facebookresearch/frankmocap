import sys
assert sys.version_info > (3, 0)
sys.path.append("src/")
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
import utils.rotate_utils as ru
from dataset.mtc.mtc_utils import project2D
from utils.data_utils import remap_joints_hand
import utils.normalize_joints_utils as nju
import utils.geometry_utils as gu
from utils.render_utils import project_joints
from dataset.mtc.prepare_mtc_hand import normalize_joints, crop_image_single


valid_seq_name = "171026_pose1" 
valid_frame_str = [f"{id:08d}" for id in (
    2975, 3310, 13225, 14025, 16785)]


def load_old_annos(anno_file):
    old_annos = dict()
    for data in pio.load_pkl_single(anno_file):
        # img_name = data['body_img_name']
        img_name = data['image_name']
        img_key = '/'.join(img_name.split('/')[-3:])
        old_annos[img_key] = data
    return old_annos


def process_mtc(all_data, all_cam, eft_data, root_dir, old_anno_dir, res_img_dir, res_anno_dir, res_vis_dir):

    for training_testing, mode_data in all_data.items():
        # determine the phase
        if training_testing.find("train")>=0:
            phase = 'train'
        else:
            phase = 'val'
        
        old_anno_file = osp.join(old_anno_dir, f"{phase}.pkl")
        if not osp.exists(old_anno_file): continue
        old_annos = load_old_annos(old_anno_file)

        new_annos = list()
        for i, sample in enumerate(mode_data):
            # load data info
            seqName = sample['seqName']
            frame_str = sample['frame_str']
            if seqName != valid_seq_name: continue
            if frame_str not in valid_frame_str: continue

            for hand_type in ['left', 'right']:
                key = f'{hand_type}_hand'
                if key in sample:

                    joints_3d = np.array(sample[key]['landmarks']).reshape(-1, 3)
                    frame_path = osp.join(root_dir, 'data_original/hdImgs', seqName, frame_str)
                    res_img_subdir = osp.join(res_img_dir, phase, seqName, frame_str)
                    ry_utils.build_dir(res_img_subdir)

                    for c in range(31):
                        # check the existence of image
                        img_name = osp.join(seqName, frame_str, f'00_{c:02d}_{frame_str}.jpg')
                        if img_name in eft_data:

                            if img_name in old_annos:
                                new_annos.append(old_annos[img_name])
                            
                            else:
                                img_name = osp.join(frame_path, f'00_{c:02d}_{frame_str}.jpg')
                                if not osp.exists(img_name): continue

                                calib_data = all_cam[seqName][c]

                                # normalize 3D joints
                                hand_joints_rot = (np.dot(calib_data['R'], joints_3d.T) + calib_data['t']).T
                                hand_joints_rot = remap_joints_hand(hand_joints_rot, "mtc", "smplx")

                                joints_3d_norm, scale_ratio = normalize_joints(hand_joints_rot)

                                joints_2d_mtc = project2D(joints_3d, calib_data, applyDistort=True)
                                joints_2d = remap_joints_hand(joints_2d_mtc, "mtc", "smplx")

                                img = cv2.imread(img_name)
                                hand_img, joints_2d_cropped = crop_image_single(img, joints_2d.copy())

                                if hand_type == "left":
                                    hand_img = np.fliplr(hand_img)
                                    joints_2d_cropped[:, 0] = hand_img.shape[1]-1 - joints_2d_cropped[:, 0]
                                    joints_3d_norm = gu.flip_hand_joints_3d(joints_3d_norm)

                                if hand_img.shape[0] <= 0 or hand_img.shape[1] <= 0:
                                    continue
                                if np.max(hand_img.shape[:2]) / np.min(hand_img.shape[:2]) > 2.0:
                                    continue

                                res_img_path = osp.join(res_img_subdir, f'00_{c:02d}_{frame_str}_{hand_type}.jpg')
                                img_name = osp.join(phase, seqName, frame_str, f'00_{c:02d}_{frame_str}_{hand_type}.jpg')

                                cv2.imwrite(res_img_path, hand_img)
                                res_anno = dict(
                                    image_root = res_img_dir,
                                    image_name = img_name,
                                    joints_2d = joints_2d_cropped,
                                    hand_joints_3d = joints_3d_norm,
                                    scale_ratio = scale_ratio,
                                    augmented = False,
                                )
                                new_annos.append(res_anno)

                                # annotate image
                                joint_img = vu.draw_keypoints(hand_img.copy(), joints_2d_cropped, radius=3)
                                cam = np.array([4.95, 0.0, 0.0])
                                joints_2d_norm = project_joints(joints_3d_norm, cam)
                                joints_2d_norm = (joints_2d_norm+1.0) * 0.5 * np.min(hand_img.shape[:2])
                                joint_img_norm = vu.draw_keypoints(hand_img.copy(), joints_2d_norm, color=(255, 0 ,0), radius=1)
                                res_img = np.concatenate((joint_img, joint_img_norm), axis=1)

                                res_vis_img_path = res_img_path.replace(res_img_dir, res_vis_dir)
                                res_vis_subdir = '/'.join(res_vis_img_path.split('/')[:-1])
                                ry_utils.build_dir(res_vis_subdir)
                                cv2.imwrite(res_vis_img_path, res_img)

            print(f"{phase}, processed:", seqName, frame_str)

        res_anno_file = osp.join(res_anno_dir, f'{phase}.pkl')
        pio.save_pkl_single(res_anno_file, new_annos)


def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/mtc'
    anno_file = osp.join(root_dir, 'data_original', 'annotation/annotation.pkl')
    all_data = pio.load_pkl_single(anno_file)
    cam_file = osp.join(root_dir, 'data_original', 'annotation/camera_data.pkl')
    all_cam = pio.load_pkl_single(cam_file)
    print("Load data complete")

    new_eft_fitting = osp.join(root_dir, "data_original/eft_fitting/eft_fitting_two_hands_selected.pkl")
    eft_data = pio.load_pkl_single(new_eft_fitting)

    res_img_dir = osp.join(root_dir, 'data_processed/image')
    old_anno_dir = osp.join(root_dir, 'data_processed/annotation')
    res_anno_dir = osp.join(root_dir, "data_processed/annotation_new")
    res_vis_dir = osp.join(root_dir, 'data_processed/image_anno')
    ry_utils.build_dir(res_img_dir)
    ry_utils.build_dir(res_anno_dir)
    ry_utils.build_dir(res_vis_dir)
    process_mtc(all_data, all_cam, eft_data, root_dir, old_anno_dir, res_img_dir, res_anno_dir, res_vis_dir)


if __name__ == '__main__':
    main()