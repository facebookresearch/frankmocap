# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.12.13

import numpy as np

import torch
import torch.nn as nn

from smplx.lbs import lbs
# from human_body_prior.body_model.lbs import lbs


class BodyModel(nn.Module):

    def __init__(self,
                 bm_path,
                 params=None,
                 num_betas=10,
                 batch_size=1, v_template = None,
                 num_dmpls=None, path_dmpl=None,
                 num_expressions=10,
                 use_posedirs=True,
                 dtype=torch.float32):

        super(BodyModel, self).__init__()

        '''
        :param bm_path: path to a SMPL model as pkl file
        :param num_betas: number of shape parameters to include.
                if betas are provided in params, num_betas would be overloaded with number of thoes betas
        :param batch_size: number of smpl vertices to get
        :param device: default on gpu
        :param dtype: float precision of the compuations
        :return: verts, trans, pose, betas 
        '''
        # Todo: if params the batchsize should be read from one of the params

        self.dtype = dtype

        if params is None: params = {}

        # -- Load SMPL params --
        if '.npz' in bm_path:
            smpl_dict = np.load(bm_path, encoding='latin1')
        else:
            raise ValueError('bm_path should be either a .pkl nor .npz file')

        njoints = smpl_dict['posedirs'].shape[2] // 3
        self.model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano'}[njoints]

        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'mano'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano.')

        self.use_dmpl = False
        if num_dmpls is not None:
            if path_dmpl is not None:
                self.use_dmpl = True
            else:
                raise (ValueError('path_dmpl should be provided when using dmpls!'))

        if self.use_dmpl and self.model_type in ['smplx', 'mano']: raise (
            NotImplementedError('DMPLs only work with SMPL/SMPLH models for now.'))

        # Mean template vertices
        if v_template is None:
            v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)
        else:
            v_template = np.repeat(v_template[np.newaxis], batch_size, axis=0)

        self.register_buffer('v_template', torch.tensor(v_template, dtype=dtype))

        self.register_buffer('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32))

        if len(params):
            if 'betas' in params.keys():
                num_betas = params['betas'].shape[1]
            if 'dmpls' in params.keys():
                num_dmpls = params['dmpls'].shape[1]

        num_total_betas = smpl_dict['shapedirs'].shape[-1]
        if num_betas < 1:
            num_betas = num_total_betas

        shapedirs = smpl_dict['shapedirs'][:, :, :num_betas]
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=dtype))

        if self.model_type == 'smplx':
            begin_shape_id = 300 if smpl_dict['shapedirs'].shape[-1] > 300 else 10
            exprdirs = smpl_dict['shapedirs'][:, :, begin_shape_id:(begin_shape_id + num_expressions)]
            self.register_buffer('exprdirs', torch.tensor(exprdirs, dtype=dtype))

            expression = torch.tensor(np.zeros((batch_size, num_expressions)), dtype=dtype, requires_grad=True)
            self.register_parameter('expression', nn.Parameter(expression, requires_grad=True))

        if self.use_dmpl:
            dmpldirs = np.load(path_dmpl)['eigvec']

            dmpldirs = dmpldirs[:, :, :num_dmpls]
            self.register_buffer('dmpldirs', torch.tensor(dmpldirs, dtype=dtype))

        # Regressor for joint locations given shape - 6890 x 24
        self.register_buffer('J_regressor', torch.tensor(smpl_dict['J_regressor'], dtype=dtype))

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        if use_posedirs:
            posedirs = smpl_dict['posedirs']
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
            self.register_buffer('posedirs', torch.tensor(posedirs, dtype=dtype))
        else:
            self.posedirs = None

        # indices of parents for each joints
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.int32))

        # LBS weights
        # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
        weights = smpl_dict['weights']
        self.register_buffer('weights', torch.tensor(weights, dtype=dtype))

        if 'trans' in params.keys():
            trans = params['trans']
        else:
            trans = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        # root_orient
        # if self.model_type in ['smpl', 'smplh']:
        root_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('root_orient', nn.Parameter(root_orient, requires_grad=True))

        # pose_body
        if self.model_type in ['smpl', 'smplh', 'smplx']:
            pose_body = torch.tensor(np.zeros((batch_size, 63)), dtype=dtype, requires_grad=True)
            self.register_parameter('pose_body', nn.Parameter(pose_body, requires_grad=True))

        # pose_hand
        if 'pose_hand' in params.keys():
            pose_hand = params['pose_hand']
        else:
            if self.model_type in ['smpl']:
                pose_hand = torch.tensor(np.zeros((batch_size, 1 * 3 * 2)), dtype=dtype, requires_grad=True)
            elif self.model_type in ['smplh', 'smplx']:
                pose_hand = torch.tensor(np.zeros((batch_size, 15 * 3 * 2)), dtype=dtype, requires_grad=True)
            elif self.model_type in ['mano']:
                pose_hand = torch.tensor(np.zeros((batch_size, 15 * 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('pose_hand', nn.Parameter(pose_hand, requires_grad=True))

        # face poses
        if self.model_type == 'smplx':
            pose_jaw = torch.tensor(np.zeros((batch_size, 1 * 3)), dtype=dtype, requires_grad=True)
            self.register_parameter('pose_jaw', nn.Parameter(pose_jaw, requires_grad=True))
            pose_eye = torch.tensor(np.zeros((batch_size, 2 * 3)), dtype=dtype, requires_grad=True)
            self.register_parameter('pose_eye', nn.Parameter(pose_eye, requires_grad=True))

        if 'betas' in params.keys():
            betas = params['betas']
        else:
            betas = torch.tensor(np.zeros((batch_size, num_betas)), dtype=dtype, requires_grad=True)
        self.register_parameter('betas', nn.Parameter(betas, requires_grad=True))

        if self.use_dmpl:
            if 'dmpls' in params.keys():
                dmpls = params['dmpls']
            else:
                dmpls = torch.tensor(np.zeros((batch_size, num_dmpls)), dtype=dtype, requires_grad=True)
            self.register_parameter('dmpls', nn.Parameter(dmpls, requires_grad=True))
        self.batch_size = batch_size

    def r(self):
        from human_body_prior.tools.omni_tools import copy2cpu as c2c
        return c2c(self.forward().v)

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None,
                trans=None, dmpls=None, expression=None, return_dict=False, v_template =None, **kwargs):
        '''

        :param root_orient: Nx3
        :param pose_body:
        :param pose_hand:
        :param pose_jaw:
        :param pose_eye:
        :param kwargs:
        :return:
        '''
        assert not (v_template  is not None and betas  is not None), ValueError('vtemplate and betas could not be used jointly.')
        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'mano'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano')
        if root_orient is None:  root_orient = self.root_orient
        if self.model_type in ['smplh', 'smpl']:
            if pose_body is None:  pose_body = self.pose_body
            if pose_hand is None:  pose_hand = self.pose_hand
        elif self.model_type == 'smplx':
            if pose_body is None:  pose_body = self.pose_body
            if pose_hand is None:  pose_hand = self.pose_hand
            if pose_jaw is None:  pose_jaw = self.pose_jaw
            if pose_eye is None:  pose_eye = self.pose_eye
        elif self.model_type in ['mano', 'mano']:
            if pose_hand is None:  pose_hand = self.pose_hand
        
        if pose_hand is None:  pose_hand = self.pose_hand

        if trans is None: trans = self.trans
        if v_template is None: v_template = self.v_template
        if betas is None: betas = self.betas

        if self.model_type in ['smplh', 'smpl']:
            full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=1)
        elif self.model_type == 'smplx':
            full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand],
                                  dim=1)  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
        elif self.model_type in ['mano', 'mano']:
            full_pose = torch.cat([root_orient, pose_hand], dim=1)

        if self.use_dmpl:
            if dmpls is None: dmpls = self.dmpls
            shape_components = torch.cat([betas, dmpls], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.dmpldirs], dim=-1)
        elif self.model_type == 'smplx':
            if expression is None: expression = self.expression
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)
        else:
            shape_components = betas
            shapedirs = self.shapedirs

        verts, joints = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                            shapedirs=shapedirs, posedirs=self.posedirs,
                            J_regressor=self.J_regressor, parents=self.kintree_table[0].long(),
                            lbs_weights=self.weights,
                            dtype=self.dtype)

        Jtr = joints + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)

        res = {}
        res['v'] = verts
        res['f'] = self.f
        res['betas'] = self.betas
        res['Jtr'] = Jtr  # Todo: ik can be made with vposer

        if self.model_type == 'smpl':
            res['pose_body'] = pose_body
        elif self.model_type == 'smplh':
            res['pose_body'] = pose_body
            res['pose_hand'] = pose_hand
        elif self.model_type == 'smplx':
            res['pose_body'] = pose_body
            res['pose_hand'] = pose_hand
            res['pose_jaw'] = pose_jaw
            res['pose_eye'] = pose_eye
        elif self.model_type in ['mano', 'mano']:
            res['pose_hand'] = pose_hand
        res['full_pose'] = full_pose

        if not return_dict:
            class result_meta(object):
                pass

            res_class = result_meta()
            for k, v in res.items():
                res_class.__setattr__(k, v)
            res = res_class

        return res


