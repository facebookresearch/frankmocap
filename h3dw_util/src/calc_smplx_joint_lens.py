
import os, sys, shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('src/')
import os.path as osp
import argparse
import numpy as np
import torch
import smplx
import ry_utils
import parallel_io as pio
import cv2
import pdb
import multiprocessing as mp
from utils.render_utils import project_joints, render
from utils.data_utils import get_skeleton_idxs, remap_joints_body
import utils.normalize_body_joints_utils as nbju

def calc_smplx():
    # create model
    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/"
    model_dir = osp.join(data_root, "models/smplx/SMPLX_NEUTRAL.pkl")
    model = smplx.create(model_dir, model_type='smplx')

    # set default
    global_orient = torch.zeros((1,3), dtype=torch.float32)
    global_orient[0, 0] = np.pi
    hand_pose = torch.zeros((1,45), dtype=torch.float32)
    hand_rot = torch.zeros((1,3))

    output = model(global_orient=global_orient, 
                    left_hand_rot = hand_rot,
                    left_hand_pose_full = hand_pose,
                    right_hand_rot = hand_rot,
                    right_hand_pose_full = hand_pose,
                    return_verts=True)
    verts = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()
    faces = model.faces
    return verts, joints, faces


def visualize_smplx(verts, joints, faces):
    # render mesh
    inputSize = 1024
    cam = np.array([0.85, 0.0, 0.15])
    render_img = render(verts, faces, cam, inputSize)

    # draw skeleton, only consider joints contained in mtc
    joints_mtc, joints_mtc_exist = remap_joints_body(joints, "smplx", "mtc")
    joints_smplx_remap, joints_smplx_remap_exist = remap_joints_body(joints_mtc, "mtc", "smplx")

    '''
    for i in range(joints_smplx_remap.shape[0]):
        if joints_smplx_remap_exist[i]:
            print(joints[i] - joints_smplx_remap[i])
    '''

    # project 3d joints
    joints_2d = project_joints(joints_smplx_remap, cam)
    joints_2d = (joints_2d + 1.0) * 0.5 * inputSize

    skeleton_idxs = get_skeleton_idxs('body')
    for id1, id2 in skeleton_idxs:
        if joints_smplx_remap_exist[id1] and joints_smplx_remap_exist[id2]:
            joint1 = joints_2d[id1].astype(np.int32)
            joint2 = joints_2d[id2].astype(np.int32)
            render_img = cv2.arrowedLine(render_img.copy(), tuple(joint1), tuple(joint2), color=(0,0,255), thickness=2)

    cv2.imwrite("visualization/smplx_body.png", render_img)


def calc_smplx_body_lens(joints):

    joints_mtc, joints_mtc_exist = remap_joints_body(joints, "smplx", "mtc")
    joints_smplx_remap, joints_smplx_remap_exist = remap_joints_body(joints_mtc, "mtc", "smplx")
    joints_smplx_remap = np.concatenate((joints_smplx_remap, joints_smplx_remap_exist.reshape(-1, 1)), axis=1)
    # joints_mtc = np.concatenate((joints_smplx_remap, joints_smplx_remap_exist.reshape(-1, 1)), axis=1)

    joints = joints[:25]
    joints_smplx = np.concatenate((joints, np.ones((joints.shape[0],1))), axis=1)
    joints_smplx = joints_smplx.reshape(1, -1, 4)
    joint_lens_smplx = np.average(nbju.calc_joint_lens(joints_smplx, "body"), axis=0)
    # print(joint_lens_smplx.shape)

    joints_mtc = joints_smplx_remap.reshape(1, -1, 4)
    joints_mtc[:, :, :3] *= 2
    joints_mtc_norm, scale_ratio = nbju.normalize_joints(joints_mtc, joint_lens_smplx)

    joints_smplx[:, :, :3] -= joints_smplx[:, 0:1, :3]
    for i in range(joints_mtc_norm.shape[1]):
        if joints_mtc_norm[0, i, 3] > 0.0:
            print(i, joints_smplx[0, i, :3] - joints_mtc_norm[0, i, :3])
    
    pio.save_pkl_single("data/body_joints/joint_lens_smplx.pkl", joint_lens_smplx)


def main():
    verts, joints, faces = calc_smplx()
    visualize_smplx(verts, joints, faces)
    calc_smplx_body_lens(joints)


if __name__ == '__main__':
    main()