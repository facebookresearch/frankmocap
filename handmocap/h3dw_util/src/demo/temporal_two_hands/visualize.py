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
from temporal.temporal_model import TemporalModel
from temporal.sample import Sample
import utils.geometry_utils as gu
import time
import torch

from demo.single_person.merge_two_hands import get_pred_verts, render_img

def visualize_two_hands(all_samples, res_dir, updated_only=False):
    # root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    root_dir = "/checkpoint/rongyu/data/models"
    smplx_model_file = osp.join(root_dir, "smplx/SMPLX_NEUTRAL.pkl")
    smplx_model = smplx.create(smplx_model_file, model_type="smplx")

    # data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/"
    hand_info_file = osp.join(root_dir, "smplx/SMPLX_HAND_INFO.pkl")
    smplx_hand_info  = pio.load_pkl_single(hand_info_file)

    # test loaded samples
    start = time.time()
    for sample_id, sample in enumerate(all_samples):

        if updated_only and not sample.updated:
            continue

        frame_path = sample.frame_path
        record = frame_path.split('/')[-1].split('.')[0].split('_')
        seq_name = '_'.join(record[:-1])
        res_img_path = osp.join(res_dir, seq_name, frame_path.split('/')[-1]).replace(".png", ".jpg")

        res_img = cv2.imread(frame_path)
        for hand_type in ['left_hand', 'right_hand']:
            pred_hand_info = sample.pred_hand_info[hand_type]
            pred_cam = pred_hand_info['pred_cam']
            pred_shape = pred_hand_info['pred_shape']
            pred_pose = pred_hand_info['pred_pose']

            pred_verts, hand_faces = get_pred_verts(
                smplx_model, smplx_hand_info, hand_type, pred_pose, pred_shape)
            bbox = pred_hand_info['bbox']
            res_img = render_img(bbox, pred_cam, pred_verts, hand_faces, res_img)
        cv2.imwrite(res_img_path, res_img)

        # print(sample.frame_path)
        img_name = frame_path.split('/')[-1].split('.')[0]
        time_cost = time.time() - start
        speed = time_cost / (sample_id+1)
        remain_time = (len(all_samples)-(sample_id+1))/speed / 60
        print(f"PID:{os.getpid()}, Processed:{img_name} {sample_id:04d}/{len(all_samples)}, remain_time:{remain_time:.3f} mins")
        sys.stdout.flush()


def main():
    samples = pio.load_pkl_single(sys.argv[1])
    res_dir = sys.argv[2]
    updated_only = int(sys.argv[3])
    visualize_two_hands(samples, res_dir, updated_only)


if __name__ == '__main__':
    main()