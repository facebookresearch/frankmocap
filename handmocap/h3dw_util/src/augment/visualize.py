import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
from collections import defaultdict
import json
import smplx
import cv2
import ry_utils
import parallel_io as pio
import pdb
from augment.augment_model import AugmentModel
from augment.sample import Sample
from utils import vis_utils
import utils.geometry_utils as gu
import time
import torch

def visualize_hand(all_samples, res_dir, updated_only=False):
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    smplx_model_file = osp.join(root_dir, "data/models/smplx/SMPLX_NEUTRAL.pkl")
    smplx_model = smplx.create(smplx_model_file, model_type="smplx")

    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/"
    hand_info_file = osp.join(data_root, "models/smplx/SMPLX_HAND_INFO.pkl")
    smplx_hand_info  = pio.load_pkl_single(hand_info_file)

    ry_utils.renew_dir(res_dir)
    # test loaded samples
    for seq_name in all_samples:
        samples = all_samples[seq_name]
        for sample_id, sample in enumerate(samples):

            # check hand pose
            for hand_type in ['left', 'right']:
                key = f"{hand_type}_hand"
                hand_img_path = sample.hand_img_path[key]
                render_img_path = sample.render_img_path[key]
                pred_hand_info = sample.pred_hand_info[key]

                # skip those samples without hand images (due to the failure of openpose)
                if len(hand_img_path) == 0:
                    assert len(render_img_path) == 0
                    continue

                if updated_only:
                    if not sample.hand_updated[key]:
                        continue

                pred_hand_cam = pred_hand_info['pred_hand_cam']
                pred_hand_pose = pred_hand_info['pred_hand_pose']
                pred_hand_verts = pred_hand_info['pred_hand_verts']

                hand_img = cv2.imread(hand_img_path)
                render_img_origin = cv2.imread(render_img_path)

                if hand_type == 'left':
                    hand_img = np.fliplr(hand_img)
                    render_img_origin = np.fliplr(render_img_origin)
                
                render_img_new = vis_utils.render_hand(
                    smplx_model, smplx_hand_info, 
                    hand_type, pred_hand_pose,
                    pred_hand_cam, hand_img.copy(), return_verts=False)
                res_img = np.concatenate((hand_img, render_img_origin, render_img_new), axis=1)

                if updated_only:
                    select_render_img_path = sample.select_sample_render_path[key]
                    select_render_img = cv2.imread(select_render_img_path)
                    if hand_type == 'left':
                        select_render_img = np.fliplr(select_render_img)
                    res_img = np.concatenate((res_img, select_render_img), axis=1)


                res_subdir = osp.join(res_dir, key, seq_name)
                ry_utils.build_dir(res_subdir)
                img_name = hand_img_path.split('/')[-1]
                res_img_path = osp.join(res_subdir, img_name)
                cv2.imwrite(res_img_path, res_img)
            
            if (sample_id+1)%10 == 0:
                print(f"Visualize Hand, {seq_name}, process {sample_id}/{len(samples)}")


def visualize_body(all_samples, res_dir, updated_only=False, use_hand_rot=True, scale_ratio=1):
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    smplx_model_file = osp.join(root_dir, "data/models/smplx/SMPLX_NEUTRAL.pkl")
    smplx_model = smplx.create(smplx_model_file, model_type="smplx")

    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/"
    hand_info_file = osp.join(data_root, "models/smplx/SMPLX_HAND_INFO.pkl")
    smplx_hand_info  = pio.load_pkl_single(hand_info_file)

    ry_utils.renew_dir(res_dir)

    num_total = 0
    for samples in all_samples.values():
        for sample in samples:
            if updated_only:
                if sample.updated:
                    num_total += 1
            else:
                num_total += 1
    num_img = 0
    start_time = time.time()
    for seq_name in all_samples:
        samples = all_samples[seq_name]
        for sample in samples:

            pred_left_hand_pose = sample.pred_hand_info['left_hand']['pred_hand_pose']
            pred_right_hand_pose = sample.pred_hand_info['right_hand']['pred_hand_pose']
            pred_left_wrist_local = sample.pred_hand_info['left_hand']['wrist_rot_local']
            pred_right_wrist_local = sample.pred_hand_info['right_hand']['wrist_rot_local']
            pred_body_pose = sample.pred_body_info['pred_body_pose']
            pred_body_shape = sample.pred_body_info['pred_body_shape']
            pred_cam = sample.pred_body_info['pred_body_cam']
            frame_path = sample.frame_path

            if updated_only:
                if not sample.updated:
                    continue
            num_img += 1

            img = cv2.imread(frame_path)
            height, width = img.shape[:2]
            new_height, new_width = int(height * scale_ratio), int(width * scale_ratio)
            img = cv2.resize(img, (new_width, new_height))

            render_body_img = vis_utils.render_body(
                smplx_model,
                smplx_hand_info,
                pred_body_pose,
                pred_body_shape,
                pred_left_hand_pose, 
                pred_right_hand_pose,
                pred_left_wrist_local,
                pred_right_wrist_local,
                pred_cam,
                img,
                img_size = img.shape[0],
                render_separate_hand = False,
                use_hand_rot = use_hand_rot,
            )
            res_img = np.concatenate((img, render_body_img), axis=1)

            res_subdir = osp.join(res_dir, seq_name)
            ry_utils.build_dir(res_subdir)
            img_name = frame_path.split('/')[-1]
            res_img_path = osp.join(res_subdir, img_name)
            cv2.imwrite(res_img_path, res_img)

            if num_img % 10 == 0:
                speed = num_img / (time.time() - start_time)
                remain_time = (num_total-num_img) / speed / 60
                print(f"visualize body, {seq_name}, process {num_img}/{num_total}, remain requires {remain_time} mins")
            
