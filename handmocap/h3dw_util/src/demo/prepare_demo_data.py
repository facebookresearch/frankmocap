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
import utils.vis_utils as vis_utils
# import utils.geometry_utils as gu
import utils.rotate_utils as ru
from dataset.mtc.mtc_utils import project2D, top_down_camera_ids
from collections import defaultdict
from utils.data_utils import remap_joints_hand, remap_joints_body
import utils.normalize_joints_utils as nju
import utils.normalize_body_joints_utils as nbju
import utils.geometry_utils as gu
from utils.render_utils import project_joints
import time


def prepare_demo(root_dir, body_img_dir, hand_img_dir, res_anno_file):
    all_data = list()
    num_hand_img = 0
    for subdir, dirs, files in os.walk(body_img_dir):
        for file in files:
            if file.endswith(".png"):
                body_img_path = osp.join(subdir, file)
                record = body_img_path.split('/')
                body_img_name = body_img_path.replace(root_dir, '')[1:]
                single_data = dict(
                    body_img_name = body_img_name,
                    uncommon_view = False,
                )
                seq_name = record[-2]
                img_name = record[-1]
                for hand_type in ['left_hand', 'right_hand']:
                    hand_img_name = img_name.replace(".png", f"_{hand_type}.png")
                    hand_img_path = osp.join(hand_img_dir, hand_type, seq_name, hand_img_name)
                    if osp.exists(hand_img_path):
                        hand_img_name = hand_img_path.replace(root_dir, '')[1:]
                    else:
                        hand_img_name = None
                    single_data[f'{hand_type}_img_name'] = hand_img_name
                    if hand_img_name is not None:
                        num_hand_img += 1
                all_data.append(single_data)
    pio.save_pkl_single(res_anno_file, all_data)


def main():
    # root_dir = '/checkpoint/rongyu/data/3d_hand/demo_data/body_capture'
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube"
    body_img_dir = osp.join(root_dir, "image_body")
    hand_img_dir = osp.join(root_dir, "image_hand")

    res_anno_file = osp.join(root_dir, "annotation/demo.pkl")
    ry_utils.make_subdir(res_anno_file)

    prepare_demo(root_dir, body_img_dir, hand_img_dir, res_anno_file)
    
if __name__ == '__main__':
    main()