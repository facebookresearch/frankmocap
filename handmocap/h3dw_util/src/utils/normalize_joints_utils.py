import os, sys, shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
# from render.render_utils import project_joints
# from check_mtc import project2D


def calc_skeleton(joints, skeleton_idxs):
    assert len(joints.shape) == 3
    bs = joints.shape[0]
    num_skeleton = len(skeleton_idxs)
    skeleton_vecs = np.zeros((bs, num_skeleton, 3))
    for i in range(num_skeleton):
        idx1, idx2 = skeleton_idxs[i]
        skeleton_vecs[:, i] = joints[:, idx1] - joints[:, idx2]
    return skeleton_vecs


def calc_joint_lens(joints_3d, joint_type='hand'):
    # define skeleton, based on smpl-x
    skeleton_idxs = data_utils.get_skeleton_idxs(joint_type)
    # for mano joints, since the shape does not change, 
    # so the length of fingers nearly does not vary
    skeleton_vecs = calc_skeleton(joints_3d, skeleton_idxs)
    # calc length of each finger knuckle
    joint_lens = np.linalg.norm(skeleton_vecs, axis=2)
    # calculate each
    return joint_lens


def normalize_joints_single(joints_3d, joints_3d_old, parent_id, child_id, scale_ratio):
    # print(parent_id, child_id)
    parent_joints_old = joints_3d_old[:, parent_id, :]
    child_joints_old = joints_3d_old[:, child_id, :]
    parent_joints = joints_3d[:, parent_id, :]
    child_joints = joints_3d[:, child_id, :]
    bone = child_joints_old - parent_joints_old
    new_child_joints = parent_joints + bone * scale_ratio
    joints_3d[:, child_id, :] = new_child_joints


def normalize_joints_batch(joints_3d_src, joint_lens_dst):
    joint_lens_src = np.average(calc_joint_lens(joints_3d_src), axis=0)
    scale_ratio_all_skeleton = np.true_divide(joint_lens_dst, joint_lens_src)
    scale_ratio = scale_ratio_all_skeleton[5] # choose the ratio of middle skeleton of middle finger

    # move root
    joints_3d = joints_3d_src - joints_3d_src[:, 0:1, :]

    # update joints in the order defined in kinematics_map
    # using the width first search
    kinematics_map = data_utils.get_kinematics_map("hand")
    joint_id_queue = [ (0, 0) ]
    while len(joint_id_queue)>0 :
        cur_id, skeleton_id = joint_id_queue[0]
        if cur_id in kinematics_map:
            for child_id, child_skeleton_id in kinematics_map[cur_id]:
                normalize_joints_single(joints_3d, joints_3d_src, cur_id, child_id, scale_ratio)
                joint_id_queue.append( (child_id, child_skeleton_id) )
        joint_id_queue = joint_id_queue[1:]
    joint_lens = np.average(calc_joint_lens(joints_3d), axis=0)

    joints_3d = joints_3d - joints_3d[:, 1:2, :]
    return joints_3d
    

def normalize_joints_to_smplx(joints_3d_src):
    joint_lens_mano = pio.load_pkl_single("data/hand_joints/joint_lens_smplx.pkl")
    joint_lens_dst = joint_lens_mano

    joint_lens_src = np.average(calc_joint_lens(joints_3d_src), axis=0)
    scale_ratio_all_skeleton = np.true_divide(joint_lens_dst, joint_lens_src)
    scale_ratio = scale_ratio_all_skeleton[5] # choose the ratio of middle skeleton of middle finger

    # move root
    joints_3d = joints_3d_src - joints_3d_src[:, 0:1, :]

    # update joints in the order defined in kinematics_map
    # using the width first search
    kinematics_map = data_utils.get_kinematics_map("hand")
    joint_id_queue = [ (0, 0) ]
    while len(joint_id_queue)>0 :
        cur_id, skeleton_id = joint_id_queue[0]
        if cur_id in kinematics_map:
            for child_id, child_skeleton_id in kinematics_map[cur_id]:
                normalize_joints_single(joints_3d, joints_3d_src, cur_id, child_id, scale_ratio)
                joint_id_queue.append( (child_id, child_skeleton_id) )
        joint_id_queue = joint_id_queue[1:]
    joint_lens = np.average(calc_joint_lens(joints_3d), axis=0)

    joints_3d = joints_3d - joints_3d[:, 1:2, :]
    return joints_3d, scale_ratio


def normalize_joints_general(joints_3d_src, joint_lens_dst):

    joint_lens_src = np.average(calc_joint_lens(joints_3d_src), axis=0)
    scale_ratio_all_skeleton = np.true_divide(joint_lens_dst, joint_lens_src)
    scale_ratio = scale_ratio_all_skeleton[5] # choose the ratio of middle skeleton of middle finger

    # move root
    joints_3d = joints_3d_src - joints_3d_src[:, 0:1, :]

    # update joints in the order defined in kinematics_map
    # using the width first search
    kinematics_map = data_utils.get_kinematics_map("hand")
    joint_id_queue = [ (0, 0) ]
    while len(joint_id_queue)>0 :
        cur_id, skeleton_id = joint_id_queue[0]
        if cur_id in kinematics_map:
            for child_id, child_skeleton_id in kinematics_map[cur_id]:
                normalize_joints_single(joints_3d, joints_3d_src, cur_id, child_id, scale_ratio)
                joint_id_queue.append( (child_id, child_skeleton_id) )
        joint_id_queue = joint_id_queue[1:]
    # joint_lens = np.average(calc_joint_lens(joints_3d), axis=0)

    joints_3d = joints_3d - joints_3d[:, 1:2, :]
    return joints_3d, scale_ratio