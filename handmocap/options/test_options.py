from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--visualize_eval', action='store_true')
        self.parser.add_argument('--test_dataset', type=str, choices=['freihand', 'ho3d', 'stb', 'rhd', 'mtc', 'wild', 'demo'], help="which dataset to test on")
        self.isTrain = False
