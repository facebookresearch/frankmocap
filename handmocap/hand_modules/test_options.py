# Copyright (c) Facebook, Inc. and its affiliates.

# Part of the code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='-1', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--visualize_eval', action='store_true')
        self.parser.add_argument('--test_dataset', type=str, choices=['freihand', 'ho3d', 'stb', 'rhd', 'mtc', 'wild', 'demo'], help="which dataset to test on")
        self.parser.add_argument("--checkpoint_path", type=str, default=None, help="path of checkpoints used in test")
        self.isTrain = False
