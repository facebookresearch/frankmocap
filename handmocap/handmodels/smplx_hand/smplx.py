# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from collections import namedtuple

import torch
import torch.nn as nn
import pdb

from .lbs import (
    lbs, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords)

from .vertex_ids import vertex_ids as VERTEX_IDS
from .utils import Struct, to_np, to_tensor
from .vertex_joint_selector import VertexJointSelector
from .smpl import SMPL
from .smplh import SMPLH


ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints', 'full_pose', 'betas',
                          'global_orient',
                          'body_pose', 'expression',
                          'left_hand_pose', 'right_hand_pose',
                          'left_hand_joints', 'right_hand_joints',
                          'jaw_pose'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)

HandOutput = namedtuple('HandOutput', [
                    "wrist_idx",
                    "hand_start_idx",
                    "middle_finger_idx",
                    "vertices_shift", 
                    "hand_vertices",
                    "hand_vertices_shift",
                    "hand_joints",
                    "hand_joints_shift"])
HandOutput.__new__.__defaults__ = (None,) * len(HandOutput._fields)


class SMPLX(SMPLH):
    '''
    SMPL-X (SMPL eXpressive) is a unified body model, with shape parameters
    trained jointly for the face, hands and body.
    SMPL-X uses standard vertex based linear blend skinning with learned
    corrective blend shapes, has N=10475 vertices and K=54 joints,
    which includes joints for the neck, jaw, eyeballs and fingers.
    '''

    NUM_BODY_JOINTS = SMPLH.NUM_BODY_JOINTS
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    NUM_EXPR_COEFFS = 10
    NECK_IDX = 12

    def __init__(self, model_path,
                 create_expression=True, expression=None,
                 create_jaw_pose=True, jaw_pose=None,
                 create_leye_pose=True, leye_pose=None,
                 create_reye_pose=True, reye_pose=None,
                 use_face_contour=False,
                 batch_size=1, gender='neutral',
                 dtype=torch.float32,
                 ext='npz',
                 **kwargs):

        ''' SMPLX model constructor

            Parameters
            ----------
            model_path: str
                The path to the folder or to the file where the model
                parameters are stored
            create_expression: bool, optional
                Flag for creating a member variable for the expression space
                (default = True).
            expression: torch.tensor, optional, Bx10
                The default value for the expression member variable.
                (default = None)
            create_jaw_pose: bool, optional
                Flag for creating a member variable for the jaw pose.
                (default = False)
            jaw_pose: torch.tensor, optional, Bx3
                The default value for the jaw pose variable.
                (default = None)
            create_leye_pose: bool, optional
                Flag for creating a member variable for the left eye pose.
                (default = False)
            leye_pose: torch.tensor, optional, Bx10
                The default value for the left eye pose variable.
                (default = None)
            create_reye_pose: bool, optional
                Flag for creating a member variable for the right eye pose.
                (default = False)
            reye_pose: torch.tensor, optional, Bx10
                The default value for the right eye pose variable.
                (default = None)
            use_face_contour: bool, optional
                Whether to compute the keypoints that form the facial contour
            batch_size: int, optional
                The batch size used for creating the member variables
            gender: str, optional
                Which gender to load
            dtype: torch.dtype
                The data type for the created variables
        '''

        # Load the model
        if osp.isdir(model_path):
            model_fn = 'SMPLX_{}.{ext}'.format(gender.upper(), ext=ext)
            smplx_path = os.path.join(model_path, model_fn)
        else:
            smplx_path = model_path
        # print("smplx_path", smplx_path)
        assert osp.exists(smplx_path), 'Path {} does not exist!'.format(
            smplx_path)

        if ext == 'pkl':
            with open(smplx_path, 'rb') as smplx_file:
                model_data = pickle.load(smplx_file, encoding='latin1')
        elif ext == 'npz':
            model_data = np.load(smplx_path, allow_pickle=True)
        else:
            raise ValueError('Unknown extension: {}'.format(ext))

        data_struct = Struct(**model_data)

        super(SMPLX, self).__init__(
            model_path=model_path,
            data_struct=data_struct,
            dtype=dtype,
            batch_size=batch_size,
            vertex_ids=VERTEX_IDS['smplx'],
            gender=gender, ext=ext,
            **kwargs)
        
        lmk_faces_idx = data_struct.lmk_faces_idx
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = data_struct.lmk_bary_coords
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=dtype))

        self.use_face_contour = use_face_contour
        if self.use_face_contour:
            dynamic_lmk_faces_idx = data_struct.dynamic_lmk_faces_idx
            dynamic_lmk_faces_idx = torch.tensor(
                dynamic_lmk_faces_idx,
                dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx',
                                 dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = data_struct.dynamic_lmk_bary_coords
            dynamic_lmk_bary_coords = torch.tensor(
                dynamic_lmk_bary_coords, dtype=dtype)
            self.register_buffer('dynamic_lmk_bary_coords',
                                 dynamic_lmk_bary_coords)

            neck_kin_chain = []
            curr_idx = torch.tensor(self.NECK_IDX, dtype=torch.long)
            while curr_idx != -1:
                neck_kin_chain.append(curr_idx)
                curr_idx = self.parents[curr_idx]
            self.register_buffer('neck_kin_chain',
                                 torch.stack(neck_kin_chain))

        if create_jaw_pose:
            if jaw_pose is None:
                default_jaw_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_jaw_pose = torch.tensor(jaw_pose, dtype=dtype)
            jaw_pose_param = nn.Parameter(default_jaw_pose,
                                          requires_grad=False)
            self.register_parameter('jaw_pose', jaw_pose_param)

        if create_leye_pose:
            if leye_pose is None:
                default_leye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_leye_pose = torch.tensor(leye_pose, dtype=dtype)
            leye_pose_param = nn.Parameter(default_leye_pose,
                                           requires_grad=False)
            self.register_parameter('leye_pose', leye_pose_param)

        if create_reye_pose:
            if reye_pose is None:
                default_reye_pose = torch.zeros([batch_size, 3], dtype=dtype)
            else:
                default_reye_pose = torch.tensor(reye_pose, dtype=dtype)
            reye_pose_param = nn.Parameter(default_reye_pose,
                                           requires_grad=False)
            self.register_parameter('reye_pose', reye_pose_param)

        if create_expression:
            if expression is None:
                default_expression = torch.zeros(
                    [batch_size, self.NUM_EXPR_COEFFS], dtype=dtype)
            else:
                default_expression = torch.tensor(expression, dtype=dtype)
            expression_param = nn.Parameter(default_expression,
                                            requires_grad=False)
            self.register_parameter('expression', expression_param)
        

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        # Create the array for the mean pose. If flat_hand is false, then use
        # the mean that is given by the data, rather than the flat open hand
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3],
                                     dtype=self.dtype)
        jaw_pose_mean = torch.zeros([3], dtype=self.dtype)
        leye_pose_mean = torch.zeros([3], dtype=self.dtype)
        reye_pose_mean = torch.zeros([3], dtype=self.dtype)

        pose_mean = np.concatenate([global_orient_mean, body_pose_mean,
                                    jaw_pose_mean,
                                    leye_pose_mean, reye_pose_mean,
                                    self.left_hand_mean, self.right_hand_mean],
                                   axis=0)

        return pose_mean

    def extra_repr(self):
        msg = super(SMPLX, self).extra_repr()
        msg += '\nGender: {}'.format(self.gender.title())
        msg += '\nExpression Coefficients: {}'.format(
            self.NUM_EXPR_COEFFS)
        msg += '\nUse face contour: {}'.format(self.use_face_contour)
        return msg

    def forward(self, betas=None, global_orient=None, body_pose=None,
                left_hand_pose=None, right_hand_pose=None, transl=None,
                left_hand_rot=None, right_hand_rot=None,
                left_hand_pose_full=None, right_hand_pose_full=None,
                expression=None, jaw_pose=None, leye_pose=None, reye_pose=None,
                return_verts=True, return_full_pose=False, pose2rot=True, **kwargs):
        '''
        Forward pass for the SMPLX model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            expression: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `expression` and use it
                instead. For example, it can used if expression parameters
                `expression` are predicted from some external model.
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            left_hand_pose: torch.tensor, optional, shape BxP
                If given, ignore the member variable `left_hand_pose` and
                use this instead. It should either contain PCA coefficients or
                joint rotations in axis-angle format.
            right_hand_pose: torch.tensor, optional, shape BxP
                If given, ignore the member variable `right_hand_pose` and
                use this instead. It should either contain PCA coefficients or
                joint rotations in axis-angle format.
            jaw_pose: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `jaw_pose` and
                use this instead. It should either joint rotations in
                axis-angle format.
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                          self.left_hand_pose)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                           self.right_hand_pose)
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
        leye_pose = leye_pose if leye_pose is not None else self.leye_pose
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose
        expression = expression if expression is not None else self.expression


        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        '''
        if self.use_pca:
            if left_hand_pose_full is None:
                left_hand_pose_full = torch.einsum(
                    'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            if right_hand_pose_full is None:
                right_hand_pose_full = torch.einsum(
                    'bi,ij->bj', [right_hand_pose, self.right_hand_components])
        else:
            left_hand_pose_full = left_hand_pose
            right_hand_pose_full = right_hand_pose
        '''
        assert left_hand_pose_full.size(1) == 45
        assert right_hand_pose_full.size(1) == 45

        full_pose = torch.cat([global_orient, body_pose[:, :19*3], 
                               left_hand_rot, right_hand_rot,
                               jaw_pose, leye_pose, reye_pose,
                               left_hand_pose_full,
                               right_hand_pose_full], dim=1)

        full_pose += self.pose_mean

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])
        # Concatenate the shape and expression coefficients
        scale = int(batch_size / betas.shape[0])
        if scale > 1:
            betas = betas.expand(scale, -1)
        shape_components = torch.cat([betas, expression], dim=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot,
                               dtype=self.dtype)
        
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            self.batch_size, 1, 1)
        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain, dtype=self.dtype)

            lmk_faces_idx = torch.cat([lmk_faces_idx,
                                       dyn_lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [lmk_bary_coords.expand(batch_size, -1, -1),
                 dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)


        # move joints 0 to the center of Left & Right Hip
        # move joints 12 to the center of Left & Right Shoulder
        joints[:, 0] = (joints[:, 1] + joints[:, 2]) / 2
        joints[:, 12] = (joints[:, 16] + joints[:, 17]) / 2

        output = ModelOutput(vertices=vertices if return_verts else None,
                             joints=joints,
                             betas=betas,
                             expression=expression,
                             global_orient=self.global_orient,
                             body_pose=body_pose,
                             jaw_pose=jaw_pose,
                             full_pose=full_pose if return_full_pose else None)
        return output
    

    def get_hand_output(self, output, hand_type, hand_info, top_finger_joints_type='', use_cuda=False):
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
            hand_verts_idx = hand_verts_idx.cuda()

        hand_verts = vertices[:, hand_verts_idx, :]
        hand_verts_shift = hand_verts - joints[:, hand_start_idx:hand_start_idx+1, :]

        hand_joints = torch.cat((joints[:, wrist_idx:wrist_idx+1, :], 
            joints[:, hand_start_idx:hand_start_idx+15, :] ), dim=1)

        # add top hand joints
        if len(top_finger_joints_type) > 0:
            if top_finger_joints_type in ['long', 'manual']:
                key = f'{hand_type}_top_finger_{top_finger_joints_type}_vert_idx'
                top_joint_vert_idx = hand_info[key]
                hand_joints = torch.cat((hand_joints, vertices[:, top_joint_vert_idx, :]), axis=1)
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
                        weights = weights.cuda()
                    top_joint = torch.sum((weights * top_finger_verts),dim=1).view(bs, 1, 3)
                    hand_joints = torch.cat((hand_joints, top_joint), axis=1)

        hand_joints_shift = hand_joints - joints[:, hand_start_idx:hand_start_idx+1, :]

        output = HandOutput(
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
