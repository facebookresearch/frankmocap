
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import shutil
import os.path as osp
from collections import OrderedDict
import deepdish
import itertools
import torch.nn.functional as F
import torch.nn as nn
import torch
import pdb
import cv2
from .smpl import batch_rodrigues


class LossUtil(object):

    def __init__(self, opt):
        self.inputSize = opt.inputSize
        self.pose_params_dim = opt.pose_params_dim
        self.isTrain = opt.isTrain
        self.use_hand_rotation = opt.use_hand_rotation
        if opt.dist:
            self.batch_size = opt.batchSize // torch.distributed.get_world_size()
        else:
            self.batch_size = opt.batchSize


    def _keypoint_2d_loss(self, gt_keypoint, pred_keypoint, keypoint_weights):
        # print("keypoint_weights", keypoint_weights[0])
        abs_loss = torch.abs((gt_keypoint-pred_keypoint))
        weighted_loss = abs_loss * keypoint_weights
        if self.isTrain:
            loss = torch.mean(weighted_loss)
        else:
            loss = weighted_loss
        return loss


    def _mano_params_loss(self, mano_pose, pred_mano_pose, mano_params_weight):
        # change pose parameters to rodrigues matrix
        # pose_params shape (bs, 72), pose_rodrigues shape(bs, 24, 3, 3)
        pose_rodrigues = batch_rodrigues(mano_pose.contiguous().view(-1, 3)).view(self.batch_size, self.pose_params_dim//3, 3, 3)

        pred_pose_rodrigues = batch_rodrigues(pred_mano_pose.contiguous().view(-1, 3)).view(\
            self.batch_size, self.pose_params_dim//3, 3, 3)
        
        # neglect the global rotation, the first 9  elements of rodrigues matrix
        if not self.use_hand_rotation:
            pose_params = pose_rodrigues[:, 1:, :, :].view(self.batch_size, -1)
            pred_pose_params = pred_pose_rodrigues[:, 1:, :, :].view(self.batch_size, -1)
        else:
            pose_params = pose_rodrigues.view(self.batch_size, -1)
            pred_pose_params = pred_pose_rodrigues.view(self.batch_size, -1)

        # square loss
        params_diff = pose_params - pred_pose_params
        square_loss = torch.mul(params_diff, params_diff)
        square_loss = square_loss * mano_params_weight

        if self.isTrain:
            loss = torch.mean(square_loss)
        else:
            loss = square_loss
        return loss
    

    def _align_by_root(self, joints):
        root = joints[:, 0:1, :]
        return joints-root


    def _joints_3d_loss(self, gt_joints_3d, pred_joints_3d, joints_3d_weight):
        # align the root by default
        gt_joints_3d = self._align_by_root(gt_joints_3d)
        pred_joints_3d = self._align_by_root(pred_joints_3d)

        # calc squared loss
        joints_diff = gt_joints_3d - pred_joints_3d
        square_loss = torch.mul(joints_diff, joints_diff)
        # print("square_loss", square_loss.size())
        # print("joints_3d_weight", joints_3d_weight.size())
        # sys.exit(0)
        square_loss = square_loss * joints_3d_weight

        if self.isTrain:
            loss = torch.mean(square_loss)
        else:
            loss = square_loss
        return loss
    

    def _shape_reg_loss(self, shape_params):
        shape_params_square = shape_params * shape_params
        loss = torch.mean(shape_params_square)
        return loss