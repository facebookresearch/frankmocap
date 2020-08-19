import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
import copy
from temporal.sample import Sample
import torch
from utils import vis_utils
import smplx
from collections import defaultdict


def get_default_value():
    return dict(
        pred_hand_pose = list(),
        wrist_rot_local = list(),
    )
            

def get_frame_ids(sample_id, win_size, num_frame):
    assert sample_id>=0 and sample_id<num_frame
    assert win_size <= num_frame
    num_half = win_size // 2
    if sample_id-num_half < 0:
        start = 0
        end = win_size
    elif sample_id+num_half+1 > num_frame:
        start = num_frame-win_size
        end = num_frame
    else:
        start = sample_id - num_half
        end = sample_id + num_half + 1
    return list(range(start, end))


def average_frame(sample, sample_id, win_size, memory_bank):
    for hand_type in ['left_hand', 'right_hand']:
        for key in ['pred_hand_pose', 'wrist_rot_local']:
            data_list = memory_bank[hand_type][key]
            num_frame = len(data_list)
            frame_ids = get_frame_ids(sample_id, win_size, num_frame)
            select_data = np.array([data_list[frame_id] for frame_id in frame_ids])
            sample.pred_hand_info[hand_type][key] = np.average(select_data, axis=0)


def apply(t_model):
    memory_bank = defaultdict(get_default_value)
    for sample in t_model.samples_origin:
        for hand_type in ['left_hand', 'right_hand']:
            for key in ['pred_hand_pose', 'wrist_rot_local']:
                memory_bank[hand_type][key].append(sample.pred_hand_info[hand_type][key])
    
    samples_new = list()
    for sample_id, sample_origin in enumerate(t_model.samples_origin):
        sample = copy.deepcopy(sample_origin)
        # for hand_type in ['left_hand', 'right_hand']:
        average_frame(sample, sample_id, t_model.win_size, memory_bank)
        sample.updated = True

        samples_new.append(sample)
    
    return samples_new