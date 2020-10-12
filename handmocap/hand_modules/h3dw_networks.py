# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np
from . import resnet

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def get_model(arch):
    if hasattr(resnet, arch):
        network = getattr(resnet, arch)
        return network(pretrained=True, num_classes=512)
    else:
        raise ValueError("Invalid Backbone Architecture")


class H3DWEncoder(nn.Module):
    def __init__(self, opt, mean_params):
        super(H3DWEncoder, self).__init__()
        self.two_branch = opt.two_branch
        self.mean_params = mean_params.clone().cuda()
        self.opt = opt

        relu = nn.ReLU(inplace=False)
        fc2  = nn.Linear(1024, 1024)
        regressor = nn.Linear(1024 + opt.total_params_dim, opt.total_params_dim)

        feat_encoder = [relu, fc2, relu]
        regressor = [regressor, ]
        self.feat_encoder = nn.Sequential(*feat_encoder)
        self.regressor = nn.Sequential(*regressor)

        self.main_encoder = get_model(opt.main_encoder)


    def forward(self, main_input):
        main_feat = self.main_encoder(main_input)
        feat = self.feat_encoder(main_feat)

        pred_params = self.mean_params
        for i in range(3):
            input_feat = torch.cat([feat, pred_params], dim=1)
            output = self.regressor(input_feat)
            pred_params = pred_params + output
        return pred_params
