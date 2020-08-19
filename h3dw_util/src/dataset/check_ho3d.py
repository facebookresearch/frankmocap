"""
Visualize the projections in published HO-3D dataset
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from os.path import join
import os.path as osp
import sys
from utils.ho3d_utils.vis_utils import *
import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
# from mpl_toolkits.mplot3d import Axes3D
from utils.vis_utils import draw_keypoints, render_hand
from utils.freihand_utils.model import HandModel
from render.render_utils import render
# import utils.geometry_utils as gu
import torchgeometry as gu
import torch
import numpy as np
import parallel_io as pio
import ry_utils


MANO_MODEL_PATH = '/Users/rongyu/Documents/research/FAIR/workplace/data/models/smplh_origin/MANO_RIGHT.pkl'
jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
import pdb


def forwardKinematics(fullpose, trans, beta):
    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)
    m = load_model(MANO_MODEL_PATH, ncomps=45, flat_hand_mean=True)
    m.fullpose[:] = fullpose
    # m.pose[:] = fullpose
    m.trans[:] = trans
    m.betas[:] = beta
    hand_faces = m.f
    return m.J_transformed.r, m, hand_faces


pi = 3.14159265358979323846
def rotate_global_orient(orient):
    rot_t = torch.from_numpy(orient.reshape(1, 3))
    rot_x = torch.Tensor((pi, 0, 0)).view(1,3)
    rotmat_t = gu.angle_axis_to_rotation_matrix(rot_t)
    rotmat_x = gu.angle_axis_to_rotation_matrix(rot_x)
    rotmat_new = torch.matmul(rotmat_x, rotmat_t)
    rot_new = gu.rotation_matrix_to_angle_axis(rotmat_new[:, :3, :])
    rot = rot_new.numpy()[0]
    return rot

def transform_hand_pose(hand_pose):
    hands_components = pio.load_pkl_single(MANO_MODEL_PATH)['hands_components']
    hands_mean = pio.load_pkl_single(MANO_MODEL_PATH)['hands_mean']
    rot = rotate_global_orient(hand_pose[:3])
    pose = hand_pose[3:]-hands_mean
    return rot, pose

def freihand_forward(hand_rot, hand_pose):
    renderer = HandModel(use_mean_pca=False, use_mean_pose=True)
    mano_model = renderer.model

    full_hand_pose = np.concatenate((hand_rot, hand_pose))
    mano_model.pose[:] = full_hand_pose
    verts = mano_model.r
    return verts


def main():
    data_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/HO3D/data"
    split = 'train'
    seq_list = [file for file in os.listdir(osp.join(data_dir, "train")) if file[0]!='.']

    res_dir = "visualization/ho3d_anno"
    ry_utils.renew_dir(res_dir)
    for seq_name in seq_list:
        for idx in range(0, 300, 100):
            img_id = f"{idx:04d}"
            # read image, depths maps and annotations
            img = read_RGB_img(data_dir, seq_name, img_id, split)
            anno = read_annotation(data_dir, seq_name, img_id, split)

            # get object 3D corner locations for the current pose
            objCorners = anno['objCorners3DRest']
            objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

            handJoints3D, handMesh, handFaces = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])

            img_size = 512
            handMesh_numpy = handMesh.r.copy()
            handMesh_numpy[:, 1] *= -1
            handMesh_numpy[:, 2] *= -1
            cams = np.array([5.0, -0.1, 0])
            render_img_ho3d = render(cams, handMesh_numpy, handFaces, img_size)

            '''
            hand_mesh_frei = freihand_forward(hand_rot_t, hand_pose_t)
            cams = np.array([5.0, -0.2, 0])
            render_img_freihand = render(cams, hand_mesh_frei, handFaces, img_size)
            '''
            origin_img = cv2.resize(img, (img_size, img_size))
            res_img = np.concatenate((origin_img, render_img_ho3d), axis=1)

            hand_rot_t, hand_pose_t = transform_hand_pose(anno['handPose'])
            render_img = render_hand(hand_pose_t, hand_rot_t, img_size)
            res_img = np.concatenate((res_img, render_img), axis=1)
            cv2.imwrite(osp.join(res_dir, f"{seq_name}_{img_id}_mesh.png"), res_img)

            # project to 2D
            handKps = project_3D_points(anno['camMat'], handJoints3D, is_OpenGL_coords=True)
            objKps = project_3D_points(anno['camMat'], objCornersTrans, is_OpenGL_coords=True)
            res_img = draw_keypoints(img.copy(), objKps)
            res_img = draw_keypoints(res_img, handKps, color='blue')
            cv2.imwrite(osp.join(res_dir, f"{seq_name}_{img_id}_kps.png"), res_img)

            res_subdir = osp.join(res_dir, f"{seq_name}_{img_id}_kps_seperate")
            ry_utils.renew_dir(res_subdir)
            for joint_id in range(handKps.shape[0]):
                joint_img = draw_keypoints(img.copy(), handKps[joint_id:joint_id+1, :])
                cv2.imwrite(osp.join(res_subdir, f"{joint_id:02d}.png"), joint_img)

            print(f"{seq_name}_{img_id} complete")


if __name__ == '__main__':
    main()