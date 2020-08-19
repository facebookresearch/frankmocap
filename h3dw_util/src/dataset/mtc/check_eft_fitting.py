# This code is used to filter out the images with hand occluded in MTC dataset

import os, sys, shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append("src/")
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
from utils.render_utils import project_joints, render
# from check_mtc import project2D
import torch
import smplx
import utils.geometry_utils as gu



def render_prediction(smplx_model, smplx_pred, img):

    global_rot = smplx_pred['global_rot']
    smplx_pose = smplx_pred['smplx_pose']
    smplx_shape = smplx_pred['smplx_shape']

    body_pose = smplx_pose.contiguous().view(1, 63).float()
    left_hand_rot = body_pose[:, 19*3:20*3].float()
    right_hand_rot = body_pose[:, 20*3:21*3].float()
    zero_pose = torch.zeros((1, 45)).float()

    output = smplx_model(
        global_orient=global_rot,
        body_pose = body_pose,
        left_hand_pose_full=zero_pose,
        left_hand_rot=left_hand_rot,
        right_hand_pose_full=zero_pose,
        right_hand_rot=right_hand_rot,
        return_verts=True)

    verts = output.vertices.detach().cpu().numpy().squeeze()
    faces = smplx_model.faces

    cam = np.array([0.8, 0.0, 0.0])
    inputSize = np.min(img.shape[:2])
    render_img = render(verts, faces, cam, inputSize, img, get_visible_faces=False)
    return render_img


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/mtc/data_original"
    eft_fitting_file = osp.join(root_dir, "eft_fitting/eft_fitting_two_hands_selected.pkl")
    eft_data = pio.load_pkl_single(eft_fitting_file)

    smplx_model_file = "/Users/rongyu/Documents/research/FAIR/workplace/data/models/smplx/SMPLX_NEUTRAL.pkl"
    smplx_model = smplx.create(smplx_model_file, model_type='smplx')

    img_dir = osp.join(root_dir, 'hdImgs')
    res_img_dir = osp.join(root_dir, 'eft_fitting_vis')
    for img_name in eft_data:
        in_img_path = osp.join(img_dir, img_name)
        img = cv2.imread(in_img_path)
        res_img_path = osp.join(res_img_dir, img_name)
        ry_utils.make_subdir(res_img_path)
        smplx_pred = eft_data[img_name]
        render_img = render_prediction(smplx_model, smplx_pred, img)
        cv2.imwrite(res_img_path, render_img)


if __name__ == '__main__':
    main()