import os

import numpy as np
import torch
from torch import nn as nn

from human_body_prior.body_model.body_model import BodyModel


class BodyModelWithPoser(BodyModel):
    def __init__(self, poser_type='vposer', smpl_exp_dir='0020_06', mano_exp_dir=None, **kwargs):
        '''
        :param poser_type: vposer/gposer
        :param kwargs:
        '''
        super(BodyModelWithPoser, self).__init__(**kwargs)
        self.poser_type = poser_type

        self.use_hands = False if mano_exp_dir is None else True

        if self.poser_type == 'vposer':

            if self.model_type == 'smpl':
                from human_body_prior.tools.model_loader import load_vposer as poser_loader

                self.poser_body_pt, self.poser_body_ps = poser_loader(smpl_exp_dir)
                self.poser_body_pt.to(self.trans.device)

                poZ_body = torch.tensor(np.zeros([self.batch_size, self.poser_body_ps.latentD]), requires_grad=True,
                                        dtype=self.trans.dtype)
                self.register_parameter('poZ_body', nn.Parameter(poZ_body, requires_grad=True))
                self.pose_body.requires_grad = False

            elif self.model_type in ['smplh', 'smplx']:
                # from experiments.nima.body_prior.tools_pt.load_vposer import load_vposer as poser_loader

                from human_body_prior.tools.model_loader import load_vposer as poser_loader
                # body
                self.poser_body_pt, self.poser_body_ps = poser_loader(smpl_exp_dir)
                self.poser_body_pt.to(self.trans.device)

                poZ_body = self.pose_body.new(np.zeros([self.batch_size, self.poser_body_ps.latentD]))
                self.register_parameter('poZ_body', nn.Parameter(poZ_body, requires_grad=True))
                self.pose_body.requires_grad = False

                if self.use_hands:
                    # hand left
                    self.poser_handL_pt, self.poser_handL_ps = poser_loader(mano_exp_dir)
                    self.poser_handL_pt.to(self.trans.device)

                    poZ_handL = self.pose_hand.new(np.zeros([self.batch_size, self.poser_handL_ps.latentD]))
                    self.register_parameter('poZ_handL', nn.Parameter(poZ_handL, requires_grad=True))

                    # hand right
                    self.poser_handR_pt, self.poser_handR_ps = poser_loader(mano_exp_dir)
                    self.poser_handR_pt.to(self.trans.device)

                    poZ_handR = self.pose_hand.new(np.zeros([self.batch_size, self.poser_handR_ps.latentD]))
                    self.register_parameter('poZ_handR', nn.Parameter(poZ_handR, requires_grad=True))
                    self.pose_hand.requires_grad = False

            elif self.model_type in ['mano_left', 'mano_right']:
                if not self.use_hands: raise ('When using MANO only VPoser you have to provide mano_exp_dir')

                from human_body_prior.tools.model_loader import load_vposer as poser_loader

                self.poser_hand_pt, self.poser_hand_ps = poser_loader(mano_exp_dir)
                self.poser_hand_pt.to(self.trans.device)

                poZ_hand = self.pose_hand.new(np.zeros([self.batch_size, self.poser_hand_ps.latentD]))
                self.register_parameter('poZ_hand', nn.Parameter(poZ_hand, requires_grad=True))
                self.pose_hand.requires_grad = False

    def forward(self, poZ_body=None, **kwargs):

        if self.poser_type == 'vposer':
            if self.model_type in ['smpl', 'smplh', 'smplx']:
                if poZ_body is None:  poZ_body = self.poZ_body

                pose = self.poser_body_pt.decode(poZ_body, output_type='aa').view(self.batch_size, -1)

                if pose.shape[1] > 63:
                    pose_body = pose[:, 3:66]
                    root_orient = pose[:, :3]
                else:
                    pose_body = pose[:, :63]
                    root_orient = None

                if self.use_hands and self.model_type in['smplh', 'smplx']:
                    pose_handL = self.poser_handL_pt.decode(self.poZ_handL, output_type='aa').view(self.batch_size, -1)
                    pose_handR = self.poser_handR_pt.decode(self.poZ_handR, output_type='aa').view(self.batch_size, -1)
                    pose_hand = torch.cat([pose_handL, pose_handR], dim=1)
                else:
                    pose_hand = None

                new_body = super(BodyModelWithPoser, self).forward(pose_body=pose_body, root_orient=root_orient, pose_hand=pose_hand, **kwargs)
                new_body.poZ_body = poZ_body


            if self.model_type in ['mano_left', 'mano_right']:
                pose_hand = self.poser_hand_pt.decode(self.poZ_hand, output_type='aa').view(self.batch_size, -1)
                new_body = super(BodyModelWithPoser, self).forward(pose_hand=pose_hand, **kwargs)

        else:
            new_body = BodyModel.forward(self)

        return new_body