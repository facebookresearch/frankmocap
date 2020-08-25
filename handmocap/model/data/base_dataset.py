from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, shutil
import os.path as osp
import random
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from PIL import Image
from scipy import misc
import cv2
import pickle
from data.data_preprocess import DataProcessor
from data import data_utils
from util.vis_util import draw_keypoints
import ry_utils
import parallel_io as pio

class BaseDataset(data.Dataset):

    def __init__(self, opt, dataset_info):
        super(BaseDataset, self).__init__()

        name, anno_path, subdir_name, check_size = dataset_info
        self.name = name
        self.anno_path = anno_path
        self.subdir_name = subdir_name
        self.check_size = check_size

        self.opt = opt
        self.isTrain = opt.isTrain
        self.data_root = opt.data_root
        self.data_processor = DataProcessor(opt)
        if self.isTrain:
            self.use_random_rescale = opt.use_random_rescale
            self.use_random_position = opt.use_random_position
            self.use_random_rotation = opt.use_random_rotation
            self.use_color_jittering = opt.use_color_jittering
            self.top_finger_joints_weight = opt.top_finger_joints_weight
            self.use_motion_blur = opt.use_motion_blur

        transform_list = [ transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
    

    def load_data(self):
        data_list = data_utils.load_annotation(self.data_root, self.anno_path)
        data_list = self.update_path(self.data_root, data_list)
        data_list = sorted(data_list, key=lambda a:a['image_path'])
        self.all_data_list = data_list

        if self.isTrain and not self.opt.sample_train_data:
            self.data_list = self.all_data_list

        # padding the evaluation data to make it divisble by the batch_size
        self.num_add = 0 # there is no necessary to pad data in training
        if not self.isTrain:
            bs = self.opt.batchSize
            self.num_add = bs - len(self.all_data_list) % bs
            self.data_list = self.all_data_list + self.all_data_list[0:1] * self.num_add
            self.update_eval_info()
    

    def preprocess_data(self, img, keypoints, mano_pose, joints_3d):
        # pad and resize, 
        img, kps, kps_weight = \
            self.data_processor.padding_and_resize(img, keypoints)
        
        if self.isTrain and self.use_random_rescale:
            img, kps = self.data_processor.random_rescale(img, kps, self.use_random_position)

        if self.isTrain and self.use_random_rotation: 
            img, kps, mano_pose, joints_3d = self.data_processor.random_rotate(img, kps, mano_pose, joints_3d)
        
        if self.isTrain and self.use_color_jittering:
            img = self.data_processor.color_jitter(img)
        
        if self.isTrain and self.use_motion_blur:
            img = self.data_processor.add_motion_blur(img)
        
        # normalize coords of keypoinst to [-1, 1]
        kps = self.data_processor.normalize_keypoints(kps)
        # return the results
        return img, kps, kps_weight, mano_pose, joints_3d


    def __getitem__(self, index):
        # load raw data
        single_data = self.data_list[index]

        # image
        img_path = single_data['image_path']
        img = cv2.imread(img_path)

        ori_img_size = np.max(img.shape[:2])

         # joints
        if self.isTrain:
            assert "joints_2d" in single_data, "Joints 2D must be provided by training data"
        if "joints_2d" in single_data:
            keypoints = single_data['joints_2d'].copy()
            # keypoints = single_data['joints_2d']
        else:
            keypoints = np.zeros((self.opt.num_joints, 3))
        if keypoints.shape[1] == 2:
            num_joints = keypoints.shape[0]
            score = np.ones((num_joints, 1), dtype=np.float32)
            keypoints = np.concatenate((keypoints, score), axis=1)
        
        # mano pose
        if "mano_pose" in single_data:
            mano_pose = single_data['mano_pose'].copy()
            mano_params_weight = np.ones((1,), dtype=np.float32)
        else:
            mano_pose = np.zeros((48,))
            mano_params_weight = np.zeros((1,), dtype=np.float32)
        # add zero padding for global rotation
        if mano_pose.shape[0] == 45:
            mano_pose = np.concatenate( (np.zeros((3,)), mano_pose) ) 
        
        # 3d joints
        if "hand_joints_3d" in single_data:
            joints_3d = single_data["hand_joints_3d"].copy()
            joints_3d_weight = np.ones((self.opt.num_joints, 1), dtype=np.float32)
        else:
            joints_3d = np.zeros((self.opt.num_joints, 3), dtype=np.float32)
            joints_3d_weight = np.zeros((self.opt.num_joints, 1), dtype=np.float32)
        # scale ratio of joints 3d
        if "scale_ratio" in single_data:
            scale_ratio = single_data['scale_ratio'].copy()
        else:
            scale_ratio = 1.0

        # special cases
        # stb is mm, change mm to meter
        if self.name == 'stb':
            scale_ratio *= 1000 
        if self.name == 'rhd':
            joints_3d_weight = keypoints[:, 2:3]
        

        # preprocess the images and the corresponding annotation
        img, kps, kps_weight, mano_pose, joints_3d = self.preprocess_data(img, keypoints, mano_pose, joints_3d)

        # transfer data from numpy to torch tensor
        img = self.transform(img).float()
        kps = torch.from_numpy(kps).float()
        kps_weight = torch.from_numpy(kps_weight).float()
        mano_pose = torch.from_numpy(mano_pose).float()
        mano_params_weight = torch.from_numpy(mano_params_weight).float()
        joints_3d = torch.from_numpy(joints_3d).float()
        joints_3d_weight = torch.from_numpy(joints_3d_weight).float()

        if self.opt.isTrain:
            kps_weight[16:, :] *= self.top_finger_joints_weight
            joints_3d_weight[16:, :] *= self.top_finger_joints_weight
        
        result = dict(
            img = img,
            keypoints = kps,
            keypoints_weights = kps_weight,
            mano_pose = mano_pose,
            mano_params_weight = mano_params_weight,
            joints_3d = joints_3d,
            joints_3d_weight = joints_3d_weight,
            scale_ratio = torch.tensor(scale_ratio),
            ori_img_size = torch.tensor(ori_img_size),
            index = torch.tensor(index),
        )

        return result


    def update_eval_info(self):
        single_data = self.all_data_list[0]

        self.has_mano_anno = False
        if 'mano_pose' in single_data:
            self.has_mano_anno = True
        
        self.has_joints_3d = False
        if 'hand_joints_3d' in single_data:
            self.has_joints_3d = True

        self.evaluate_demo_video = False
        if self.name == 'demo':
            self.evaluate_demo_video = True

        self.evaluate_joints_2d = False
        if self.name == 'pmhand':
            self.evaluate_joints_2d = True
        

    def update_path(self, data_root, data_list):
        res_data_list = list()
        for data in data_list:
            image_path = osp.join(
                self.data_root, self.subdir_name, data['image_name'])
            if self.check_size:
                if 'image_shape' in data:
                    img_shape = data['image_shape']
                else:
                    img = cv2.imread(image_path)
                    img_shape = img.shape[:2]
                if img_shape[0]>50 or img_shape[1]>50:
                    data['image_path'] = image_path
                    res_data_list.append(data)
            else:
                data['image_path'] = image_path
                res_data_list.append(data)
        return res_data_list


    def getitem(self, index):
        return self.__getitem__(index)

    
    def sample_train_data(self):
        assert self.opt.isTrain, "Sample train data can only be called in Train Phase."
        num_sample = min(len(self.all_data_list), self.opt.num_sample)
        self.data_list = random.sample(self.all_data_list, num_sample)
        '''
        tmp_sample_dir = ".data_samples"
        tmp_sample_path = osp.join(tmp_sample_dir, f"{self.name}_sample.pkl")

        if self.opt.process_rank <= 0:
            self.load_data()
            ry_utils.build_dir(tmp_sample_dir)
            num_sample = min(len(self.all_data_list), self.opt.num_sample)
            data_list = random.sample(self.all_data_list, num_sample)
            pio.save_pkl_single(tmp_sample_path, data_list)

        torch.distributed.barrier()
        self.data_list = pio.load_pkl_single(tmp_sample_path)
        '''


    def __len__(self):
        return len(self.data_list)
