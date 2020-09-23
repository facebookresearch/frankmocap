# Copyright (c) Facebook, Inc. and its affiliates.

import os
import os.path as osp
import sys
import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
# from PIL import Image

# Code from https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/demo.py

# 2D body pose estimator
pose2d_estimator_path = './detectors/body_pose_estimator'
sys.path.append(pose2d_estimator_path)
from detectors.body_pose_estimator.pose2d_models.with_mobilenet import PoseEstimationWithMobileNet
from detectors.body_pose_estimator.modules.load_state import load_state
from detectors.body_pose_estimator.val import normalize, pad_width
from detectors.body_pose_estimator.modules.pose import Pose, track_poses
from detectors.body_pose_estimator.modules.keypoints import extract_keypoints, group_keypoints


class BodyPoseEstimator(object):
    """
    Hand Detector for third-view input.
    It combines a body pose estimator (https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git)
    """
    def __init__(self):
        print("Loading Body Pose Estimator")
        self.__load_body_estimator()
    

    def __load_body_estimator(self):
        net = PoseEstimationWithMobileNet()
        pose2d_checkpoint = "./extra_data/body_module/body_pose_estimator/checkpoint_iter_370000.pth"
        checkpoint = torch.load(pose2d_checkpoint, map_location='cpu')
        load_state(net, checkpoint)
        net = net.eval()
        net = net.cuda()
        self.model = net
    

    #Code from https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/demo.py
    def __infer_fast(self, img, input_height_size, stride, upsample_ratio, 
        cpu=False, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
        height, width, _ = img.shape
        scale = input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [input_height_size, max(scaled_img.shape[1], input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not cpu:
            tensor_img = tensor_img.cuda()

        stages_output = self.model(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad
    
    def detect_body_pose(self, img):
        """
        Output:
            current_bbox: BBOX_XYWH
        """
        stride = 8
        upsample_ratio = 4
        orig_img = img.copy()

        # forward
        heatmaps, pafs, scale, pad = self.__infer_fast(img, 
            input_height_size=256, stride=stride, upsample_ratio=upsample_ratio)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        num_keypoints = Pose.num_kpts
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        
        '''
        # print(len(pose_entries))
        if len(pose_entries)>1:
            pose_entries = pose_entries[:1]
            print("We only support one person currently")
            # assert len(pose_entries) == 1, "We only support one person currently"
        '''

        current_poses, current_bbox = list(), list()
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18]) 
            current_poses.append(pose.keypoints)
            current_bbox.append(np.array(pose.bbox))

        # enlarge the bbox
        for i, bbox in enumerate(current_bbox):
            x, y, w, h = bbox
            margin = 0.05
            x_margin = int(w * margin)
            y_margin = int(h * margin)
            x0 = max(x-x_margin, 0)
            y0 = max(y-y_margin, 0)
            x1 = min(x+w+x_margin, orig_img.shape[1])
            y1 = min(y+h+y_margin, orig_img.shape[0])
            current_bbox[i] = np.array((x0, y0, x1-x0, y1-y0)).astype(np.int32)

        return current_poses, current_bbox