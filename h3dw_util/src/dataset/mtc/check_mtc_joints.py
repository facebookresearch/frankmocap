import sys
# assert sys.version_info > (3, 0)
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
# import utils.geometry_utils as gu
import utils.rotate_utils as ru
from dataset.mtc.mtc_utils import project2D



def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/mtc/data_original/'
    anno_file = osp.join(root_dir, "annotation/annotation.pkl")
    all_data = pio.load_pkl_single(anno_file)
    cam_file = osp.join(root_dir, 'annotation/camera_data.pkl')
    all_cam = pio.load_pkl_single(cam_file)
    print("Load data complete")

    valid_seq_name = "171026_pose1" 
    valid_frame_str = [f"{id:08d}" for id in (
        2975, 3310, 13225, 14025, 16785)]

    valid_frame_str = [f"{id:08d}" for id in (
        2975,)]

    vis_root_dir = "visualization/mtc"
    ry_utils.renew_dir(vis_root_dir)

    for training_testing, mode_data in all_data.items():
        for i, sample in enumerate(mode_data):

            # load data info
            seqName = sample['seqName']
            frame_str = sample['frame_str']
            if seqName != valid_seq_name: continue
            if frame_str not in valid_frame_str: continue
            if ('left_hand' not in sample) and ('right_hand' not in sample):
                continue

            vis_subdir = osp.join(vis_root_dir, frame_str)
            ry_utils.build_dir(vis_subdir)

            frame_path = osp.join(root_dir, 'hdImgs', seqName, frame_str)

            body_landmark = np.array(sample['body']['landmarks']).reshape(-1, 3)
            left_hand_landmark = np.array(sample['left_hand']['landmarks']).reshape(-1, 3)
            right_hand_landmark = np.array(sample['right_hand']['landmarks']).reshape(-1, 3)
           
            # choose different camera
            for c in range(31):
                img_name = osp.join(frame_path, f'00_{c:02d}_{frame_str}.jpg')
                if not osp.exists(img_name): continue
                img = cv2.imread(img_name)

                calib_data = all_cam[seqName][c]
                body_2d = project2D(body_landmark, calib_data, applyDistort=True)

                left_hand_2d = project2D(left_hand_landmark, calib_data, applyDistort=True)
                right_hand_2d = project2D(right_hand_landmark, calib_data, applyDistort=True)

                vis_dir = vis_subdir
                joint_img_dir = osp.join(vis_dir, f"{c:02d}_joints")
                ry_utils.renew_dir(joint_img_dir)
                for j in range(body_2d.shape[0]):
                    joint_img_single = vu.draw_keypoints(img.copy(), body_2d[j:j+1, :], radius=10)
                    joint_img_path = osp.join(joint_img_dir, f"{j:02d}.png")
                    cv2.imwrite(joint_img_path, joint_img_single)

                for hand_type in ('left', 'right'):
                    if hand_type == 'left':
                        hand_2d = left_hand_2d
                    else:
                        hand_2d = right_hand_2d
                    vis_dir = osp.join(vis_subdir, hand_type)
                    ry_utils.build_dir(vis_dir)

                    joint_img = vu.draw_keypoints(img.copy(), hand_2d, color='red', radius=3)
                    res_img_path = osp.join(vis_dir, f"{c:02d}.png")
                    cv2.imwrite(res_img_path, joint_img)
                
                    joint_img_dir = osp.join(vis_dir, f"{c:02d}_joints")
                    ry_utils.renew_dir(joint_img_dir)
                    for j in range(hand_2d.shape[0]):
                        joint_img_single = vu.draw_keypoints(img.copy(), hand_2d[j:j+1, :], radius=3)
                        joint_img_path = osp.join(joint_img_dir, f"{j:02d}.png")
                        cv2.imwrite(joint_img_path, joint_img_single)

if __name__ == '__main__':
    main()