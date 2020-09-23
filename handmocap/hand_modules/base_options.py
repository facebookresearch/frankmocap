# Copyright (c) Facebook, Inc. and its affiliates.

# Part of the code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

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
        self.parser.add_argument('--inputSize', type=int, default=224, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='h3dw', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/home/hjoo/dropbox/hand_yu/checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=80, help='visdom port of the web display')

        self.parser.add_argument('--data_root', type=str, default='', help='root dir for all the datasets')
        self.parser.add_argument('--freihand_anno_path', type=str, default='', help='annotation_path that stores the information of freihand dataset')
        self.parser.add_argument('--ho3d_anno_path', type=str, default='', help='annotation_path that stores the information of HO3D dataset')
        self.parser.add_argument('--mtc_anno_path', type=str, default='', help='annotation_path that stores the information of MTC (Panoptic 3D) dataset')
        self.parser.add_argument('--stb_anno_path', type=str, default='', help='annotation_path that stores the information of  STB dataset')
        self.parser.add_argument('--rhd_anno_path', type=str, default='', help='annotation_path that stores the information of RHD dataset')
        self.parser.add_argument('--frl_anno_path', type=str, default='', help='annotation_path that stores the information of FRL dataset')
        self.parser.add_argument('--ganerated_anno_path', type=str, default='', help='annotation_path that stores the information of GANerated dataset')
        self.parser.add_argument('--demo_img_dir', type=str, default='', help='image root of demo dataset')
        self.parser.add_argument('--wild_img_dir', type=str, default='', help='image root of in-the-wild dataset (in-the-wild means without any annotation, only image)')

        self.parser.add_argument('--num_joints', type=int, default=21, help='number of keypoints')
        self.parser.add_argument('--total_params_dim', type=int, default=61, help='number of params to be estimated')
        self.parser.add_argument('--cam_params_dim', type=int, default=3, help='number of params to be estimated')
        self.parser.add_argument('--pose_params_dim', type=int, default=48, help='number of params to be estimated')
        self.parser.add_argument('--shape_params_dim', type=int, default=10, help='number of params to be estimated')

        self.parser.add_argument('--model_root', type=str, default='./extra_data', help='root dir for all the pretrained weights and pre-defined models')
        self.parser.add_argument('--smplx_model_file', type=str, default='./extra_data/smpl/SMPLX_NEUTRAL.pkl', help='path of pretraind smpl model')
        self.parser.add_argument('--smplx_hand_info_file', type=str, default='hand_module/SMPLX_HAND_INFO.pkl', help='path of smpl face')
        self.parser.add_argument('--mean_param_file', type=str, default='hand_module/mean_mano_params.pkl', help='path of smpl face')

        self.parser.add_argument('--single_branch', action='store_true', help='use only one branch, this branch could either be IUV or other format such as image')
        self.parser.add_argument('--two_branch', action='store_true', help='two branch input, image and another auxiliary branch, the auxiliary branch is IUV in default')
        self.parser.add_argument('--aux_as_main', action='store_true', help='use aux as input instead of image')
        self.parser.add_argument('--main_encoder', type=str, default='resnet50', help='selects model to use for major input, it is usually image')
        self.parser.add_argument('--aux_encoder', type=str, default='resnet18', help='selects model to use for auxiliary input, it could be IUV') 

        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--use_hand_rotation', action='store_true', help='if specified, use ground truth hand rotation in training')
        self.parser.add_argument('--top_finger_joints_type', type=str, default='ave', help="use which kind of top finger joints")
        self.initialized = True
        self.initialized = True


    def parse(self, args=None):
        if not self.initialized:
            self.initialize()

        if args is None:
            self.opt = self.parser.parse_args()
        else:
            self.opt = self.parser.parse_args(args)
        # self.opt, unknown = self.parser.parse_known_args()
        self.opt.isTrain = self.isTrain   # train or test

        return self.opt
