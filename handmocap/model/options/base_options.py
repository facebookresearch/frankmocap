from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dist', action='store_true', help='whether to use distributed training')
        self.parser.add_argument('--local_rank', type=int, default=0)
        self.parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
        self.parser.add_argument('--inputSize', type=int, default=224, help='Input size of hand images for encoder (You do not need to prepose data to this size, code will automatically padding and resize any image to this size)')
        self.parser.add_argument('--input_nc', type=int, default=3, help='channel of input image channels')
        self.parser.add_argument('--name', type=str, default='h3dw', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=1, help='if positive, display all images in a single visdom web panel with certain number of images per row.')

        self.parser.add_argument('--data_root', type=str, default='', help='root dir for all the datasets')
        self.parser.add_argument('--freihand_anno_path', type=str, default='', help='annotation_path that stores the information of freihand dataset')
        self.parser.add_argument('--ho3d_anno_path', type=str, default='', help='annotation_path that stores the information of HO3D dataset')
        self.parser.add_argument('--mtc_anno_path', type=str, default='', help='annotation_path that stores the information of MTC (Panoptic 3D) dataset')
        self.parser.add_argument('--stb_anno_path', type=str, default='', help='annotation_path that stores the information of  STB dataset')
        self.parser.add_argument('--rhd_anno_path', type=str, default='', help='annotation_path that stores the information of RHD dataset')
        self.parser.add_argument('--frl_anno_path', type=str, default='', help='annotation_path that stores the information of FRL dataset')
        self.parser.add_argument('--ganerated_anno_path', type=str, default='', help='annotation_path that stores the information of GANerated dataset')
        self.parser.add_argument('--pmhand_anno_path', type=str, default='', help='annotation_path that stores the information of Panoptic Hand (MPII part) dataset')
        self.parser.add_argument('--demo_img_dir', type=str, default='', help='image root of demo dataset')

        self.parser.add_argument('--num_joints', type=int, default=21, help='number of keypoints')
        self.parser.add_argument('--total_params_dim', type=int, default=61, help='total dim of params to be estimated')
        self.parser.add_argument('--cam_params_dim', type=int, default=3, help='dim of camera params to be estimated')
        self.parser.add_argument('--pose_params_dim', type=int, default=48, help='dim of pose params to be estimated')
        self.parser.add_argument('--shape_params_dim', type=int, default=10, help='dim of shape params to be estimated')

        self.parser.add_argument('--model_root', type=str, default='checkpoint/rongyu/data/models/', help='root dir for all the pretrained weights and pre-defined models')
        self.parser.add_argument('--smplx_model_file', type=str, default='smplx/SMPLX_NEUTRAL.pkl', help='path of pretraind smpl model')
        self.parser.add_argument('--smplx_hand_info_file', type=str, default='smplx/SMPLX_HAND_INFO.pkl', help='path of smpl face')
        self.parser.add_argument('--mean_param_file', type=str, default='stat/mean_mano_params.pkl', help='path of smpl face')

        self.parser.add_argument('--single_branch', action='store_true', help='Please only use single_branch, this code only supports single_branch')
        self.parser.add_argument('--two_branch', action='store_true', help='Please ignore this, two_branch is not supported')
        self.parser.add_argument('--aux_as_main', action='store_true', help='use aux as input instead of image')
        self.parser.add_argument('--main_encoder', type=str, default='resnet50', help='selects model to use for major input, it is usually image')
        self.parser.add_argument('--aux_encoder', type=str, default='resnet18', help='selects model to use for auxiliary input, it could be IUV') 

        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--use_hand_rotation', action='store_true', help='if specified, use ground truth hand rotation in training')
        self.parser.add_argument('--top_finger_joints_type', type=str, default='ave', help="use which kind of top finger joints")
        self.initialized = True
        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        self.opt.single_branch = True
        self.opt.two_branch = False
    
        return self.opt