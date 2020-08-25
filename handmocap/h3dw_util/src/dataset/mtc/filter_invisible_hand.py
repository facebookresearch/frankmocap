# This code is used to filter out the images with hand occluded in MTC dataset

import os, sys, shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append("src/")
import os.path as osp
import argparse
import numpy as np
import torch
import smplx
import ry_utils
import parallel_io as pio
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt
import random as rd
import pdb
# from utils.freihand_utils.fh_utils import *
# from utils.freihand_utils.model import HandModel, recover_root, get_focal_pp, split_theta
from utils import data_utils
from utils import vis_utils
from utils.render_utils import project_joints, render
from check_mtc import project2D
import torch
import smplx
import utils.geometry_utils as gu





def render_prediction(root_dir, img, global_rot, smplx_pose):
    model_file = osp.join(root_dir, "models/smplx/SMPLX_NEUTRAL.pkl")
    model = smplx.create(model_file, model_type='smplx')

    body_pose = smplx_pose.contiguous().view(1, 63).float()
    left_hand_rot = body_pose[:, 19*3:20*3].float()
    right_hand_rot = body_pose[:, 20*3:21*3].float()
    zero_pose = torch.zeros((1, 45)).float()

    output = model(
        global_orient=global_rot,
        body_pose = body_pose,
        left_hand_pose_full=zero_pose,
        left_hand_rot=left_hand_rot,
        right_hand_pose_full=zero_pose,
        right_hand_rot=right_hand_rot,
        return_verts=True)

    verts = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces

    cam = np.array([0.3, 0.0, 0.0])
    inputSize = np.min(img.shape[:2])
    render_img, visible_faces = render(verts, faces, cam, inputSize, img, get_visible_faces=True)
    return render_img, visible_faces


def calc_visible_ratio(hand_faces, visible_faces):
    num_visible = 0
    for face_id in hand_faces:
        if face_id in visible_faces:
            num_visible += 1
    return num_visible / len(hand_faces)


def filter_out_occlusion(root_dir, anno_file, cam_file, fitting_results, hand_info):
    all_data = pio.load_pkl_single(anno_file)
    all_cam = pio.load_pkl_single(cam_file)
    # selected_fitting_results = dict()

    valid_seq_name = "171026_pose1" 
    valid_frame_str = [f"{id:08d}" for id in (
        2975, 3310, 13225, 14025, 16785)]
    
    res_dir = "visualization/mtc_eft_fitting"
    ry_utils.renew_dir(res_dir)

    img_dir = osp.join(root_dir, 'mtc/data_original', 'hdImgs')
    for training_testing, mode_data in all_data.items():
        for i, sample in enumerate(mode_data):

            # load data info
            seqName = sample['seqName']
            frame_str = sample['frame_str']
            if seqName != valid_seq_name: continue
            if frame_str not in valid_frame_str: continue

            # load hand info
            for c in range(31):
                for hand_type in ['left', 'right']:
                    key = f'{hand_type}_hand'
                    if key in sample:
                        img_name = osp.join(seqName, frame_str, f'00_{c:02d}_{frame_str}.jpg')
                        img_path = osp.join(img_dir, img_name)
                        if img_name in fitting_results:
                            img = cv2.imread(img_path)
                            global_rot, smplx_pose = fitting_results[img_name]
                            render_img, visible_faces = render_prediction(root_dir, img, global_rot, smplx_pose)
                            hand_faces = hand_info[f'{hand_type}_hand_faces_idx']
                            visible_ratio = calc_visible_ratio(hand_faces, visible_faces)
                            vis_key = f"visible_ratio_{c:02d}"
                            sample[key][vis_key] = visible_ratio
                            print(f'{frame_str}_{c:02d}.jpg', hand_type, visible_ratio)
                            res_img_path = osp.join(res_dir, f'{frame_str}_{c:02d}.jpg')
                            cv2.imwrite(res_img_path, render_img)
                            # selected_fitting_results[img_name] = fitting_results[img_name]
    pio.save_pkl_single(anno_file, all_data)
    # return selected_fitting_results


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/"
    '''
    eft_fitting_dir = osp.join(root_dir, 'mtc/11-05_panoptic_refit')
    mtc_eft_fittings = load_eft_fitting(eft_fitting_dir)
    fitting_file = osp.join(root_dir, 'mtc', 'eft_fitting.pkl')
    pio.save_pkl_single(fitting_file, mtc_eft_fittings)
    '''

    # fitting_file = osp.join(root_dir, 'mtc', 'eft_fitting.pkl')
    fitting_file = osp.join(root_dir, 'mtc/data_original', 'eft_fitting/eft_fitting_selected.pkl')
    mtc_eft_fittings = pio.load_pkl_single(fitting_file)
    mtc_anno_file = osp.join(root_dir, "mtc/data_original/annotation/annotation.pkl")
    mtc_cam_file = osp.join(root_dir, "mtc/data_original/annotation/camera_data.pkl")
    hand_info_file = osp.join(root_dir, "models/smplx/SMPLX_HAND_INFO.pkl")
    hand_info = pio.load_pkl_single(hand_info_file)
    filter_out_occlusion(root_dir, mtc_anno_file, mtc_cam_file, mtc_eft_fittings, hand_info)
    # fitting_file = osp.join(root_dir, 'mtc', 'eft_fitting_selected.pkl')
    # pio.save_pkl_single(fitting_file, selected_fitting_results)

if __name__ == '__main__':
    main()