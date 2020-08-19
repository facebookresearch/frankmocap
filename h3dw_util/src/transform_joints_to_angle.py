import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
# from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import quat2mat, quat2axangle
import minimal_hand.config as config
from minimal_hand.hand_mesh import HandMesh
from minimal_hand.kinematics import mpii_to_mano
from minimal_hand.utils import OneEuroFilter, imresize
from minimal_hand.wrappers import ModelConverter
from minimal_hand.utils import *
import pdb
import ry_utils
import parallel_io as pio
from utils import data_utils
import utils.normalize_joints_utils as nju
import utils.geometry_utils as gu
import torch
import smplx
from render.render_utils import render, project_joints


def get_joint_lens_minimal():
    joints_3d_minimal = np.array(pio.load_pkl_single("data/joints_3d_minimal_hand.pkl"))
    # although joints_3d_minimal is for left hand, but lens of skeleton does affected by left/right hand
    joints_3d_minimal_smplx = data_utils.remap_joints(joints_3d_minimal, "mpii", "smplx", "hand")
    joint_lens_minimal = nju.calc_joint_lens(joints_3d_minimal_smplx)
    return np.average(joint_lens_minimal, axis=0)


def normalize_joints_to_minimal_hand(joints_3d_mano, joint_lens_minimal):
    # right -> left
    rot_mat = np.diag([-1, 1, 1])
    joints_3d_mano = np.matmul(rot_mat, joints_3d_mano.T).T
    # normalize scale
    joints_3d_norm, scale_ratio = nju.normalize_joints_general(joints_3d_mano.reshape(1, 21, 3), joint_lens_minimal)
    joints_3d_norm = joints_3d_norm.reshape(21, 3)
    # recorder and move root
    joints_3d_mpii = data_utils.remap_joints(joints_3d_norm, "smplx", "mpii", "hand")
    joints_3d_mpii -= joints_3d_mpii[9:10, :]
    return joints_3d_mpii


def quat_to_angle_axis_01(theta):
    angle_axis = list()
    for t in theta:
        res = quat2axangle(t)
        axis, theta = res
        angle_axis.append(axis * theta)
    return np.array(angle_axis)


def quat_to_angle_axis(theta):
    angle_axis = gu.quaternion_to_angle_axis(torch.from_numpy(theta))
    return angle_axis.numpy()


def get_smplx_model():
    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/"
    model_dir = osp.join(data_root, "models/smplx/SMPLX_NEUTRAL.pkl")
    hand_info_file = osp.join(data_root, "models/smplx/SMPLX_HAND_INFO.pkl")
    hand_info = pio.load_pkl_single(hand_info_file)
    model = smplx.create(model_dir, model_type='smplx')
    return model, hand_info


def render_hand_single(model, hand_info, mano_pose):
    MANO_MODEL_PATH = '/Users/rongyu/Documents/research/FAIR/workplace/data/models/smplh_origin/MANO_RIGHT.pkl'
    hands_mean = pio.load_pkl_single(MANO_MODEL_PATH)['hands_mean']
    hands_mean = torch.from_numpy(hands_mean.reshape(1,45)).float()
    # right_hand_pose_full = torch.zeros((1, 15*3), dtype=torch.float32) - hands_mean
    right_hand_pose_full = torch.from_numpy(mano_pose.reshape(1, 45)).float()
    hand_type = "right"

    global_orient = torch.zeros((1,3), dtype=torch.float32)
    global_orient[0, 0] = np.pi
    cam = np.array([6.240435, 0.0, 0.0])

    slice = 2*np.pi / 32
    hand_rotation = np.array([
        [0, 0, 0],
        [23*slice, 0, 0],
        [0, 9*slice, 0],
    ])

    img_list = list()
    for num_rot in range(len(hand_rotation)):
        global_rot = hand_rotation[num_rot]
        hand_rot = torch.from_numpy(global_rot[None, :]).float()

        output = model(global_orient=global_orient, 
                        left_hand_rot = hand_rot,
                        left_hand_pose_full = right_hand_pose_full,
                        right_hand_rot = hand_rot,
                        right_hand_pose_full = right_hand_pose_full,
                        return_verts=True)
        
        hand_output = model.get_hand_output(output, hand_type, hand_info, 'ave')
        verts_shift = hand_output.vertices_shift.detach().cpu().numpy().squeeze()
        hand_joints_shift = hand_output.hand_joints_shift.detach().cpu().numpy().squeeze()
        hand_faces = hand_info[f'{hand_type}_hand_faces_holistic']

        inputSize = 1024
        img = render(cam, verts_shift, hand_faces, inputSize)
        img_list.append(img)
    return np.concatenate(img_list, axis=1)


def main():
    assert False, "This code does not work"
    # calc joints length of minimal_hand
    joint_lens_minimal = get_joint_lens_minimal()

    # model for transforming 3D joints to angle
    model = ModelConverter()

    # smplx model
    smplx_model, smplx_hand_info = get_smplx_model()

    # load data
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/FRL_data/data_processed"
    anno_file = osp.join(root_dir, "annotation/all.pkl")
    all_data = pio.load_pkl_single(anno_file)

    for single_data in all_data:
        joints_3d_mano = single_data['hand_joints_3d']
        # transform joints 3d from mano to mpii format
        joints_3d_mpii = normalize_joints_to_minimal_hand(joints_3d_mano, joint_lens_minimal)
        # get theta from model (in the form of quaterion)
        theta_mpii = model.process(joints_3d_mpii)
        theta_mano = mpii_to_mano(theta_mpii)
        # transfer quaterion to angle_axis
        axis_angle = quat_to_angle_axis(theta_mano)
        axis_angle = axis_angle[1:16, :].reshape(15, 3)
        # left -> right
        axis_angle[:, 1] *= -1
        axis_angle[:, 2] *= -1
        # transfer
        render_hand = render_hand_single(smplx_model, smplx_hand_info, axis_angle)
        print(render_hand.shape)
        cv2.imwrite("0.png", render_hand)
        sys.exit(0)


if __name__ == '__main__':
    main()
