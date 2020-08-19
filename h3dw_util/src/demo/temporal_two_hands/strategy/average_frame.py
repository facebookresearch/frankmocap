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


def select_frames(sample_id, win_size, num_frame):
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

        
def update_hand_info(sample, hand_type, data_list, select_ids):
    pred_hand_info = sample.pred_hand_info[hand_type]

    for key in ['pred_cam', 'pred_shape', 'pred_pose', 'bbox']:
        select_data = np.array([data_list[id][key] for id in select_ids])
        pred_hand_info[key] = np.average(select_data, axis=0)


def average_frame(sample, sample_id, win_size, memory_bank):
    for hand_type in ['left_hand', 'right_hand']:
        data_list = memory_bank[hand_type]
        num_frame = len(data_list)
        select_ids = select_frames(sample_id, win_size, num_frame)
        update_hand_info(sample, hand_type, data_list, select_ids)


def apply(t_model):
    memory_bank = dict(left_hand=list(), right_hand=list())
    for sample in t_model.samples_origin:
        for hand_type in ['left_hand', 'right_hand']:
            memory_bank[hand_type].append(sample.pred_hand_info[hand_type])
    
    samples_new = list()
    for sample_id, sample_origin in enumerate(t_model.samples_origin):
        sample = copy.deepcopy(sample_origin)
        # for hand_type in ['left_hand', 'right_hand']:
        average_frame(sample, sample_id, t_model.win_size, memory_bank)
        sample.updated = True
        samples_new.append(sample)
    
    return samples_new