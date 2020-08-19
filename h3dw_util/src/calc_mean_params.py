# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os, sys, shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os.path as osp
import argparse
import numpy as np
import torch
import smplx
import ry_utils
import parallel_io as pio
from render.render_utils import render
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt
import random as rd

from utils.freihand_utils.fh_utils import *
from utils.freihand_utils.model import HandModel, recover_root, get_focal_pp, split_theta
from utils import data_utils


def get_all_data(anno_dir):
    all_data = list()
    for subdir, dirs, files in os.walk(anno_dir):
        for file in files:
            if file.endswith(".pkl"):
                anno_file_path = osp.join(subdir, file)
                all_data.extend(pio.load_pkl_single(anno_file_path))
    return all_data

def main():
    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/FreiHAND/data/"
    anno_dir = osp.join(data_root, "data_processed/annotation")

    all_data = get_all_data(anno_dir)
    all_mano_pose = np.array([data['mano_pose'] for data in all_data if not data['augmented']])
    mean_mano_pose = np.average(all_mano_pose, axis=0)

    res_file_path = "/Users/rongyu/Documents/research/FAIR/workplace/data/models/stat_results/mean_mano_params.pkl"
    res_dict = dict(
        mean_pose = mean_mano_pose,
    )
    pio.save_pkl_single(res_file_path, res_dict)


if __name__ == '__main__':
    main()