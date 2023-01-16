# Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# dct is the abbr. of Human Model recovery with Densepose supervision
import numpy as np
import torch
import os
import sys
import shutil
import os.path as osp
from collections import OrderedDict
import itertools
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import pdb
import cv2
from . import resnet
from handmocap.hand_modules.h3dw_networks_cpu import H3DWEncoder
import time
import mocap_utils.general_utils as gnu
import smplx
import pdb


def extract_hand_output(output, hand_type, hand_info, top_finger_joints_type='ave', use_cuda=True):
    assert hand_type in ['left', 'right']

    if hand_type == 'left':
        wrist_idx, hand_start_idx, middle_finger_idx = 20, 25, 28
    else:
        wrist_idx, hand_start_idx, middle_finger_idx = 21, 40, 43
    
    vertices = output.vertices
    joints = output.joints
    vertices_shift = vertices - joints[:, hand_start_idx:hand_start_idx+1, :]

    hand_verts_idx = torch.Tensor(hand_info[f'{hand_type}_hand_verts_idx']).long()
    if use_cuda:
        hand_verts_idx = hand_verts_idx.cpu()

    hand_verts = vertices[:, hand_verts_idx, :]
    hand_verts_shift = hand_verts - joints[:, hand_start_idx:hand_start_idx+1, :]

    hand_joints = torch.cat((joints[:, wrist_idx:wrist_idx+1, :], 
        joints[:, hand_start_idx:hand_start_idx+15, :] ), dim=1)

    # add top hand joints
    if len(top_finger_joints_type) > 0:
        if top_finger_joints_type in ['long', 'manual']:
            key = f'{hand_type}_top_finger_{top_finger_joints_type}_vert_idx'
            top_joint_vert_idx = hand_info[key]
            hand_joints = torch.cat((hand_joints, vertices[:, top_joint_vert_idx, :]), dim=1)
        else:
            assert top_finger_joints_type == 'ave'
            key1 = f'{hand_type}_top_finger_{top_finger_joints_type}_vert_idx'
            key2 = f'{hand_type}_top_finger_{top_finger_joints_type}_vert_weight'
            top_joint_vert_idxs = hand_info[key1]
            top_joint_vert_weight = hand_info[key2]
            bs = vertices.size(0)

            for top_joint_id, selected_verts in enumerate(top_joint_vert_idxs):
                top_finger_vert_idx = hand_verts_idx[np.array(selected_verts)]
                top_finger_verts = vertices[:, top_finger_vert_idx]
                # weights = torch.from_numpy(np.repeat(top_joint_vert_weight[top_joint_id]).reshape(1, -1, 1)
                weights = top_joint_vert_weight[top_joint_id].reshape(1, -1, 1)
                weights = np.repeat(weights, bs, axis=0)
                weights = torch.from_numpy(weights)
                if use_cuda:
                    weights = weights.cpu()
                top_joint = torch.sum((weights * top_finger_verts),dim=1).view(bs, 1, 3)
                hand_joints = torch.cat((hand_joints, top_joint), dim=1)

    hand_joints_shift = hand_joints - joints[:, hand_start_idx:hand_start_idx+1, :]

    output = dict(
        wrist_idx = wrist_idx,
        hand_start_idx = hand_start_idx,
        middle_finger_idx = middle_finger_idx,
        vertices_shift = vertices_shift,
        hand_vertices = hand_verts,
        hand_vertices_shift = hand_verts_shift,
        hand_joints = hand_joints,
        hand_joints_shift = hand_joints_shift
    )
    return output


class H3DWModel(object):
    @property
    def name(self):
        return 'H3DWModel'

    def __init__(self, opt):
        self.opt = opt
        self.Tensor = torch.FloatTensor

        # set params
        self.inputSize = opt.inputSize
        self.total_params_dim = opt.total_params_dim
        self.cam_params_dim = opt.cam_params_dim
        self.pose_params_dim = opt.pose_params_dim
        self.shape_params_dim = opt.shape_params_dim
        self.top_finger_joints_type = opt.top_finger_joints_type

        assert(self.total_params_dim ==
               self.cam_params_dim+self.pose_params_dim+self.shape_params_dim)

        if opt.dist:
            self.batch_size = opt.batchSize // torch.distributed.get_world_size()
        else:
            self.batch_size = opt.batchSize
        nb = self.batch_size

        # set input image and 2d keypoints
        self.input_img = self.Tensor(
            nb, opt.input_nc, self.inputSize, self.inputSize)
      
        # joints 2d
        self.keypoints = self.Tensor(nb, opt.num_joints, 2)
        self.keypoints_weights = self.Tensor(nb, opt.num_joints)

        # mano pose params
        self.gt_pose_params = self.Tensor(nb, opt.pose_params_dim)
        self.mano_params_weight = self.Tensor(nb, 1)

        # joints 3d
        self.joints_3d = self.Tensor(nb, opt.num_joints, 3)
        self.joints_3d_weight = self.Tensor(nb, opt.num_joints, 1)

        # load mean params, the mean params are from HMR
        self.mean_param_file = osp.join(
            opt.model_root, opt.mean_param_file)
        self.load_params()

        # set differential SMPL (implemented with pytorch) and smpl_renderer
        # smplx_model_path = osp.join(opt.model_root, opt.smplx_model_file)
        smplx_model_path = opt.smplx_model_file
        self.smplx = smplx.create(
            smplx_model_path, 
            model_type = "smplx", 
            batch_size = self.batch_size,
            gender = 'neutral',
            num_betas = 10,
            use_pca = False,
            ext='pkl')

        # set encoder and optimizer
        self.encoder = H3DWEncoder(opt, self.mean_params)
        if opt.dist:
            self.encoder = DistributedDataParallel(
                self.encoder, device_ids=[torch.cpu.current_device()])
        
        checkpoint_path = opt.checkpoint_path
        if not osp.exists(checkpoint_path): 
            print(f"Error: {checkpoint_path} does not exists")
            self.success_load = False
        else:
            if self.opt.dist:
                self.encoder.module.load_state_dict(torch.load(
                    checkpoint_path, map_location=lambda storage, loc: storage.cpu(torch.cpu.current_device())))
            else:
                saved_weights = torch.load(checkpoint_path)
                self.encoder.load_state_dict(saved_weights)
            self.success_load = True


    def load_params(self):
        # load mean params first
        mean_vals = gnu.load_pkl(self.mean_param_file)
        mean_params = np.zeros((1, self.total_params_dim))

        # set camera model first
        mean_params[0, 0] = 5.0

        # set pose (might be problematic)
        mean_pose = mean_vals['mean_pose'][3:]
        # set hand global rotation
        mean_pose = np.concatenate( (np.zeros((3,)), mean_pose) )
        mean_pose = mean_pose[None, :]

        # set shape
        mean_shape = np.zeros((1, 10))
        mean_params[0, 3:] = np.hstack((mean_pose, mean_shape))
        # concat them together
        mean_params = np.repeat(mean_params, self.batch_size, axis=0)
        self.mean_params = torch.from_numpy(mean_params).float()
        self.mean_params.requires_grad = False

        # define global rotation
        self.global_orient = torch.zeros((self.batch_size, 3), dtype=torch.float32)
        # self.global_orient[:, 0] = np.pi
        self.global_orient.requires_grad = False

        # load smplx-hand faces
        hand_info_file = osp.join(self.opt.model_root, self.opt.smplx_hand_info_file)

        self.hand_info = gnu.load_pkl(hand_info_file)
        self.right_hand_faces_holistic = self.hand_info['right_hand_faces_holistic']        
        self.right_hand_faces_local = self.hand_info['right_hand_faces_local']
        self.right_hand_verts_idx = np.array(self.hand_info['right_hand_verts_idx'], dtype=np.int32)


    def set_input_imgonly(self, input):
        # image
        input_img = input['img']
        self.input_img.resize_(input_img.size()).copy_(input_img)


    def get_smplx_output(self, pose_params, shape_params=None):
        hand_rotation = pose_params[:, :3]
        hand_pose = pose_params[:, 3:]
        body_pose = torch.zeros((self.batch_size, 63)).float().cpu() 
        body_pose[:, 60:] = hand_rotation # set right hand rotation

        output = self.smplx(
            global_orient = self.global_orient,
            body_pose = body_pose,
            right_hand_pose = hand_pose,
            betas = shape_params,
            return_verts = True)
        
        hand_output = extract_hand_output(
            output, 
            hand_type = 'right', 
            hand_info = self.hand_info,
            top_finger_joints_type = self.top_finger_joints_type, 
            use_cuda=True)

        pred_verts = hand_output['vertices_shift']
        pred_joints_3d = hand_output['hand_joints_shift']
        return pred_verts, pred_joints_3d


    def batch_orth_proj_idrot(self, X, camera):
        # camera is (batchSize, 1, 3)
        camera = camera.view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        res = camera[:, :, 0] * X_trans.view(X_trans.size(0), -1)
        return res.view(X_trans.size(0), X_trans.size(1), -1)


    def forward(self):
        # get predicted params first
        self.output = self.encoder(self.input_img)
        # print(self.output.mean())
        self.final_params = self.output
        # self.final_params = self.output + self.mean_params

        # get predicted params for cam, pose, shape
        cam_dim = self.cam_params_dim
        pose_dim = self.pose_params_dim
        shape_dim = self.shape_params_dim
        self.pred_cam_params = self.final_params[:, :cam_dim]
        self.pred_pose_params = self.final_params[:, cam_dim: (
            cam_dim + pose_dim)]
        self.pred_shape_params = self.final_params[:, (cam_dim + pose_dim):]

        #  get predicted smpl verts and joints,
        self.pred_verts, self.pred_joints_3d = self.get_smplx_output(
            self.pred_pose_params, self.pred_shape_params)

        # generate additional visualization of mesh, with constant camera params
        # self.generate_mesh_multi_view()

        # generate predicted joints 2d
        self.pred_joints_2d = self.batch_orth_proj_idrot(
            self.pred_joints_3d, self.pred_cam_params)
        

    def test(self):
        with torch.no_grad():
            self.forward()


    def get_pred_result(self):
        pred_result = OrderedDict(
            cams = self.pred_cam_params.cpu().numpy(),
            pred_shape_params = self.pred_shape_params.cpu().numpy(),
            pred_pose_params = self.pred_pose_params.cpu().numpy(),
            pred_verts = self.pred_verts.cpu().numpy()[:, self.right_hand_verts_idx, :],
            pred_joints_3d = self.pred_joints_3d.cpu().numpy(),
        )
        return pred_result


    def eval(self):
        self.encoder.eval()