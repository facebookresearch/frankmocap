import cv2
import numpy as np
import os.path as osp
import sys
import time
import ry_utils
import numpy as np
import pdb
from collections import defaultdict


def get_single_joints_error(joints_3d_1, joints_3d_2, joint_weights):
    num_joint = joint_weights.shape[0]
    pcks = list()
    for i in range(num_joint):
        if joint_weights[i][0] > 0:
            joints_1 = joints_3d_1[i]
            joints_2 = joints_3d_2[i]
            distance = np.linalg.norm( (joints_1-joints_2) )
            pcks.append(distance)
    return pcks


def get_single_joints_error_2d(joints_2d_1, joints_2d_2):
    pcks = list()
    num_joint = joints_2d_1.shape[0]
    for i in range(num_joint):
        joints_1 = joints_2d_1[i]
        joints_2 = joints_2d_2[i]
        distance = np.linalg.norm( (joints_1-joints_2) )
        pcks.append(distance)
    return pcks


def calc_auc_3d(all_pck):
    '''
    all_pck = np.array([-1,])
    for data in pred_results:
        pck = data['pck']
        all_pck = np.concatenate((all_pck, pck))
    all_pck = all_pck[1:]
    '''

    if len(all_pck) == 0:
        return -1

    start = 0.02
    end = 0.05
    xs = list()
    ratios = list()
    for thresh in np.linspace(start, end):
        ratio = np.mean(all_pck < thresh)
        x = (thresh-start) / (end-start)
        xs.append(x)
        ratios.append(ratio)

    auc = np.trapz(ratios, xs)
    return auc


def calc_auc_2d(all_pck):
    '''
    all_pck = np.array([-1,])
    for data in pred_results:
        pck = data['pck_2d']
        all_pck = np.concatenate((all_pck, pck))
    all_pck = all_pck[1:]
    '''

    if len(all_pck) == 0:
        return -1

    start = 0
    end = 30
    xs = list()
    ratios = list()
    for thresh in np.linspace(start, end):
        ratio = np.mean(all_pck < thresh)
        x = (thresh-start) / (end-start)
        xs.append(x)
        ratios.append(ratio)

    auc = np.trapz(ratios, xs)
    return auc


def get_sequence_joints(pred_results):
    # get sequence data first
    sequence_data = defaultdict(list)
    for single_data in pred_results:
        img_path = single_data['img_path']
        seq_name = img_path.split('/')[-2]
        sequence_data[seq_name].append( (img_path, single_data['pred_joints_3d']) )
    for seq_name in sequence_data:
        data = sequence_data[seq_name]
        sequence_data[seq_name] = sorted(data, key=lambda a:a[0])

    # prepare joints 3d in sequence 
    sequence_joints = defaultdict(list)
    for seq_name in sequence_data:
        data = sequence_data[seq_name]
        for img_path, pred_joints_3d in data:
            sequence_joints[seq_name].append(pred_joints_3d)
        sequence_joints[seq_name] = np.array(sequence_joints[seq_name])
        # print(seq_name, sequence_joints[seq_name].shape)
    return sequence_joints


def calc_seq_jitter(pred_results):
    sequence_joints = get_sequence_joints(pred_results)
    seq_results = defaultdict(list)
    win_size = 7

    for seq_name in sequence_joints:
        joints_all = sequence_joints[seq_name]
        num_sample = joints_all.shape[0]
        num_joints = joints_all.shape[1]
        for i in range(num_sample-1):
            start = max(0, i-(win_size-1)//2)
            end = min(num_sample, i+(win_size-1)//2)
            move_dis = 0.0
            for j in range(start, end):
                if j == i: continue
                joints1 = joints_all[i]
                joints2 = joints_all[j]
                move_dis += np.average(
                    np.linalg.norm((joints1-joints2), axis=1))
            seq_results[seq_name].append(move_dis/(end-start-1))

    all_results = list()
    for seq_name in seq_results:
        all_results += seq_results[seq_name]
    return np.average(all_results)