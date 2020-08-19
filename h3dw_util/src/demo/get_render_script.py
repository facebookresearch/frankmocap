import os, sys, shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('src/')
import os.path as osp
import argparse
import numpy as np
import torch
import smplx
from utils.render_utils import render
import cv2
import multiprocessing as mp
import utils.geometry_utils as gu
import parallel_io as pio
import ry_utils
import pdb
import time
from collections import defaultdict
from utils.vis_utils import render_hand, render_body
import multiprocessing as mp


def get_all_keys(frame_dir):
    all_keys = list()
    all_files = ry_utils.get_all_files(frame_dir, ".png", "name_only")
    for file in all_files:
        key = file.split('.')[0]
        all_keys.append(key)
    return all_keys


def main():
    # root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_processed_03"
    # root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube/augment_bbox"
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_multi/"

    frame_dir = osp.join(root_dir, "frame")
    all_keys = get_all_keys(frame_dir)

    res_dir = osp.join(root_dir, 'prediction/h3dw/origin_frame')
    ry_utils.renew_dir(res_dir)
    for key in all_keys:
        seq_name = '_'.join(key.split('_')[:-1])
        res_subdir = osp.join(res_dir, seq_name)
        ry_utils.build_dir(res_subdir)

    res_key_dir = "data/render_keys"
    ry_utils.renew_dir(res_key_dir)

    person_type = "multi"
    assert person_type in ["single", "multi"]

    num_process = 4
    num_data = len(all_keys)
    num_each = num_data // num_process
    res_sh_file = "run_render.sh"
    with open(res_sh_file, "w") as out_f:
        for i in range(num_process):
            start = i*num_each
            end = (i+1)*num_each if i<num_process-1 else num_data
            keys = all_keys[start:end]
            res_pkl_file = osp.join(res_key_dir, f"key_list_{i:02d}.pkl")
            pio.save_pkl_single(res_pkl_file, keys)

            python_script = f"src/demo/{person_type}_person/merge_two_hands.py"

            line = f"python {python_script} {root_dir} {res_dir} {res_pkl_file}" + \
                f" 2>&1 | tee data/render_log/{i:02d}.log & \n"
            out_f.write(line)


if __name__ == '__main__':
    main()