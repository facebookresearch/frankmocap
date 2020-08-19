from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=2048, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=2048, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--total_epoch', type=int, default=100, help='the number of epoch we need to train the model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load?')
        self.parser.add_argument('--lr_e', type=float, default=1e-5, help='initial learning rate for encoder') # In HMR, the paper says the lr is setting to 1e-5, but the code use 1e-3
        self.parser.add_argument('--lr_decay', action='store_true', help='decrease learning rate during training')
        self.parser.add_argument('--kp_loss_weight', type=float, default=10.0, help='loss weight for 2D keypoints loss')
        self.parser.add_argument('--loss_3d_weight', type=float, default=10.0, help='loss weight for 3d loss, which includes 3d joints, smpl params')
        self.parser.add_argument('--shape_reg_weight', type=float, default=0.1, help='loss weight for regularize shape params')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/web/')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--pretrained_weights', type=str, default=None, help='path to pretrained weights')

        self.parser.add_argument('--train_datasets', type=str, help="which datasets to use in training")
        self.parser.add_argument("--use_random_rescale", action='store_true', help='use random rescale in data augmentation')
        self.parser.add_argument("--use_random_position", action='store_true', help='use random position in data augmentation')
        self.parser.add_argument("--use_random_rotation", action='store_true', help='use random rotation in data augmentation')
        self.parser.add_argument("--use_color_jittering", action="store_true", help="use color jittering in data augmentation")
        self.parser.add_argument("--use_motion_blur", action="store_true", help="use motion blur augmentation")
        self.parser.add_argument("--blur_kernel_dir", type=str, default="path of directory that stores blur kernel")
        self.parser.add_argument("--motion_blur_prob", type=float, default=0.5, help="the probability of using motion blur")
        self.parser.add_argument("--sample_train_data", action="store_true", help="sample_train_data")
        self.parser.add_argument("--num_sample", type=int, default=70000,)
        self.parser.add_argument("--top_finger_joints_weight", type=float, default=1.0)
        self.isTrain = True
