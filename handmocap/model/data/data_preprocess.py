from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import os.path as osp
import random
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
import torch
import cv2
import pickle
from PIL import Image
import util.rotate_utils as ru
from data import data_utils


class DataProcessor(object):
    def __init__(self, opt):
        self.opt = opt
        '''
        self.rescale_range = [0.2, 1.0]
        self.angle_scale = [-180, 180]
        self.num_slice = 30
        '''
        self.rescale_range = [0.6, 1.0]
        self.angle_scale = [-90, 90]
        self.num_slice = 10
        self.color_transfomer = transforms.ColorJitter(
            brightness = (0.9, 1.3),
            contrast = (0.8, 1.3),
            saturation = (0.4, 1.6),
            hue = (-0.1, 0.1)
        )

        if opt.isTrain:
            self.blur_kernels = data_utils.load_blur_kernel(opt.blur_kernel_dir) 
            self.motion_blur_prob = opt.motion_blur_prob
    

    def padding_and_resize(self, img, kps):
        kps_weight = np.ones(kps.shape[0])
        kps_weight[kps[:, 2] < 1e-8] = 0
        visible_ky_num = np.count_nonzero(kps_weight)
        weight_scale = kps.shape[0]/visible_ky_num if visible_ky_num>0 else 0.0
        kps_weight *= weight_scale
        kps_weight = kps_weight.reshape(kps.shape[0], 1)

        final_size = self.opt.inputSize
        height, width = img.shape[:2]
        if height > width:
            ratio = final_size / height
            new_height = final_size
            new_width = int(ratio * width)
        else:
            ratio = final_size / width
            new_width = final_size
            new_height = int(ratio * height)
        new_img = np.zeros((final_size, final_size, 3), dtype=np.uint8)
        new_img[:new_height, :new_width, :] = cv2.resize(img, (new_width, new_height))

        kps = kps[:, :2]
        kps *= ratio
        return new_img, kps, kps_weight


    def random_rescale(self, img, kps, use_random_position=False):
        min_s, max_s = self.rescale_range
        final_size = self.opt.inputSize
        random_scale = random.random() * (max_s-min_s) + min_s
        new_size = int(final_size * random_scale)
        res_img = np.zeros((final_size, final_size, 3), dtype=np.uint8)

        y_pos, x_pos = 0, 0
        if use_random_position:
            height, width = img.shape[:2]
            assert height==width
            end = final_size-new_size-1
            x_pos = random.randint(0, end)
            y_pos = random.randint(0, end)

        new_img = cv2.resize(img, (new_size, new_size))
        res_img[y_pos:new_size+y_pos, x_pos:new_size+x_pos, :] = new_img

        kps *= random_scale
        kps[:, 0] += x_pos
        kps[:, 1] += y_pos

        return res_img, kps
    

    def random_rotate(self, img, kps, mano_pose, joints_3d):
        min_angle, max_angle = self.angle_scale
        num_slice = self.num_slice
        slice_id = random.randint(0, num_slice-1)
        angle = (max_angle-min_angle)/num_slice * slice_id + min_angle
        # image
        img = ru.rotate_image(img.copy(), angle)
        # orient of hand
        rot_orient = ru.rotate_orient(mano_pose[:3], angle)
        mano_pose[:3] = rot_orient
        # joints 2d
        origin = np.array((img.shape[1]/2, img.shape[0]/2)).reshape(1,2)
        kps = ru.rotate_joints_2d(kps, origin, angle)
        # joints 3d
        joints_3d = ru.rotate_joints_3d(joints_3d.T, angle)
        joints_3d = joints_3d.T
        return img, kps, mano_pose, joints_3d


    def color_jitter(self, img):
        pil_img = Image.fromarray(img)
        transformed_img = self.color_transfomer(pil_img)
        res_img = np.asarray(transformed_img)
        return res_img
    

    def add_motion_blur(self, img):
        if random.random() < self.motion_blur_prob:
            blur_kernel = random.choice(self.blur_kernels)
            img = cv2.filter2D(img, -1, blur_kernel)
        return img


    def normalize_keypoints(self, keypoints):
        final_size = self.opt.inputSize

        new_kps = np.copy(keypoints)
        new_kps[:, 0] = (keypoints[:, 0] / final_size) * 2.0 - 1.0
        new_kps[:, 1] = (keypoints[:, 1] / final_size) * 2.0 - 1.0

        return new_kps