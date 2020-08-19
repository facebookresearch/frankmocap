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
from utils.data_utils import remap_joints
import utils.normalize_joints_utils as nju
import utils.geometry_utils as gu
from utils.render_utils import project_joints



           


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/mtc/data_original/annotation"

    in_file = osp.join(root_dir, "annotation_all.pkl")
    res_file = osp.join(root_dir, "annotation.pkl")

    all_data = pio.load_pkl_single(in_file)

    res_data = dict()

    valid_seq_name = "171026_pose1" 
    valid_frame_str = [f"{id:08d}" for id in (
     2975, 3310, 13225, 14025, 16785)]

    for training_testing, mode_data in all_data.items():
        # determine the phase
        if training_testing.find("train")>=0:
            phase = 'train'
        else:
            phase = 'val'
        
        res_data[phase] = list()
        for i, sample in enumerate(mode_data):
            # load data info
            seqName = sample['seqName']
            frame_str = sample['frame_str']
            if seqName != valid_seq_name: continue
            if frame_str not in valid_frame_str: continue

            print(seqName, frame_str, phase)
            res_data[phase].append(sample)
    
    pio.save_pkl_single(res_file, res_data)


if __name__ == '__main__':
    main()