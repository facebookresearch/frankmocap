import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import ry_utils
from collections import OrderedDict
from collections import defaultdict
import parallel_io as pio


def get_joint_map(src_name, dst_name, part):
    assert part in ['body', 'hand']

    # dst_info: joint_id -> joint_name
    dst_joint_file = f'src/data/{dst_name}_{part}_joint.map'
    dst_info = OrderedDict()
    with open(dst_joint_file, 'r') as in_f:
        for line in in_f:
            record = line.strip().split(':')
            joint_id = int(record[0])
            joint_name = record[1].strip()
            dst_info[joint_id] = joint_name

    # src_info: joint_name -> joint_id
    src_joint_file = f'src/data/{src_name}_{part}_joint.map'
    src_info = OrderedDict()
    with open(src_joint_file, 'r') as in_f:
        for line in in_f:
            record = line.strip().split(':')
            joint_id = int(record[0])
            joint_name = record[1].strip()
            src_info[joint_name] = joint_id

    map_info = OrderedDict()
    for joint_id in dst_info:
        dst_joint_name = dst_info[joint_id]
        if dst_joint_name in src_info:
            map_info[joint_id] = src_info[dst_joint_name]
        else:
            map_info[joint_id] = -1
    return map_info


def remap_joints_hand(src_joints, src_name, dst_name):
    part = 'hand'
    map_info = get_joint_map(src_name, dst_name, part)
    joint_shape = src_joints.shape
    num_dst_joint = len(map_info)

    if len(joint_shape) == 2:
        # dim_joint = len(src_joints[0])
        dim_joint = joint_shape[1]
        dst_joints = np.zeros((num_dst_joint, dim_joint), dtype=np.float32)
        for dst_joint_id in map_info.keys():
            src_joint_id = map_info[dst_joint_id]
            assert src_joint_id >= 0
            dst_joints[dst_joint_id] = src_joints[src_joint_id]

    elif len(joint_shape) == 3:
        bs = joint_shape[0]
        dim_joint = joint_shape[2]
        dst_joints = np.zeros((bs, num_dst_joint, dim_joint), dtype=np.float32)
        for dst_joint_id in map_info.keys():
            src_joint_id = map_info[dst_joint_id]
            assert src_joint_id >= 0
            dst_joints[:, dst_joint_id, :] = src_joints[:, src_joint_id, :]
    
    else:
        raise ValueError("Unsupported data format")

    return dst_joints

# remap_joints = remap_joints_hand


def remap_joints_body(src_joints, src_name, dst_name):
    part = 'body'

    map_info = get_joint_map(src_name, dst_name, part)
    # print(map_info)
    joint_shape = src_joints.shape
    num_dst_joint = len(map_info)

    if len(joint_shape) == 2:
        # dim_joint = len(src_joints[0])
        dim_joint = joint_shape[1]
        dst_joints = np.zeros((num_dst_joint, dim_joint), dtype=np.float32)
        dst_joints_exist = np.zeros((num_dst_joint,), dtype=np.float32)
        for dst_joint_id in map_info.keys():
            src_joint_id = map_info[dst_joint_id]
            if src_joint_id >= 0:
                dst_joints[dst_joint_id] = src_joints[src_joint_id]
                dst_joints_exist[dst_joint_id] = 1

    elif len(joint_shape) == 3:
        bs = joint_shape[0]
        dim_joint = joint_shape[2]
        dst_joints = np.zeros((bs, num_dst_joint, dim_joint), dtype=np.float32)
        dst_joints_exist = np.zeros((bs, num_dst_joint), dtype=np.float32)
        for dst_joint_id in map_info.keys():
            src_joint_id = map_info[dst_joint_id]
            if src_joint_id >= 0:
                dst_joints[:, dst_joint_id, :] = src_joints[:, src_joint_id, :]
                dst_joints_exist[:, dst_joint_id] = 1
    else:
        raise ValueError("Unsupported data format")

    return dst_joints, dst_joints_exist


def get_skeleton_idxs_old(part='hand'):
    map_file = f'src/data/smplx_{part}_skeleton.map'
    skeleton_idxs = list()
    with open(map_file, "r") as in_f:
        for line in in_f:
            idx1, idx2 = map(int, line.strip().split(','))
            skeleton_idxs.append((idx1, idx2))
    return skeleton_idxs


def get_skeleton_idxs_new(part='hand'):
    # map_file = f'src/data/smplx_{part}_skeleton.map'
    map_file = f'src/data/smplx_{part}_skeleton_new.map'
    skeleton_idxs = list()
    with open(map_file, "r") as in_f:
        for line_id, line in enumerate(in_f):
            # idx1, idx2 = map(int, line.strip().split(','))
            record = line.strip().split(':')
            skeleton_id = int(record[0])
            assert line_id == skeleton_id
            idx1, idx2 = map(int, record[1].split(','))
            skeleton_idxs.append((idx1, idx2))
    return skeleton_idxs
get_skeleton_idxs = get_skeleton_idxs_new


def get_kinematics_map(part='hand'):
    map_file = f'src/data/smplx_{part}_skeleton.map'
    kinematic_map = defaultdict(list)
    with open(map_file, "r") as in_f:
        for skeleton_id, line in enumerate(in_f):
            idx1, idx2 = map(int, line.strip().split(','))
            kinematic_map[idx1].append( (idx2, skeleton_id) )
    return kinematic_map