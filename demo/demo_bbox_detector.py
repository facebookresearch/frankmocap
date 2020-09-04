import os
import os.path as osp
import sys
import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
from PIL import Image

# 2D body pose estimator
pose2d_estimator_path = './detectors/body_pose_estimator'
sys.path.append(pose2d_estimator_path)
from detectors.body_pose_estimator.pose2d_models.with_mobilenet import PoseEstimationWithMobileNet
from detectors.body_pose_estimator.modules.load_state import load_state
from detectors.body_pose_estimator.val import normalize, pad_width
from detectors.body_pose_estimator.modules.pose import Pose, track_poses
from detectors.body_pose_estimator.modules.keypoints import extract_keypoints, group_keypoints


# Type agnostic hand detector
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances


class Body_Pose_Estimator(object):
    """
    Hand Detector for third-view input.
    It combines a body pose estimator (https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git)
    """
    def __init__(self):
        self.__load_body_estimator()
        # self.__load_hand_detector()
    

    def __load_body_estimator(self):
        net = PoseEstimationWithMobileNet()
        pose2d_checkpoint = "./data/weights/body_pose_estimator/checkpoint_iter_370000.pth"
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
        
        assert len(pose_entries) == 1, "We only support one person currently"
        current_poses = []
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
        return current_poses


class Third_View_Detector(Body_Pose_Estimator):
    """
    Hand Detector for third-view input.
    It combines a body pose estimator (https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git)
    with a type-agnostic hand detector (https://github.com/ddshan/hand_detector.d2)
    """
    def __init__(self):
        super(Third_View_Detector, self).__init__()
        self.__load_hand_detector()
    

    def __load_hand_detector(self):
         # load cfg and model
        cfg = get_cfg()
        cfg.merge_from_file("detectors/hand_only_detector/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
        cfg.MODEL.WEIGHTS = 'data/weights/hand_detector/model_0529999.pth' # add model weight here
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # 0.3 , use low thresh to increase recall
        self.hand_detector = DefaultPredictor(cfg)


    def __get_raw_hand_bbox(self, img):
        bbox_tensor = self.hand_detector(img)['instances'].pred_boxes
        bboxes = bbox_tensor.tensor.cpu().numpy()
        return bboxes


    def detect_hand_bbox(self, img):
        # get body pose
        body_pose_list = self.detect_body_pose(img)
        assert len(body_pose_list) == 1, "Current version only supports one person"

        # get raw hand bboxes
        raw_hand_bboxes = self.__get_raw_hand_bbox(img)
        hand_bbox_list = list()
        num_bbox = raw_hand_bboxes.shape[0]

        if num_bbox > 0:
            for body_pose in body_pose_list:
                # By default, we use distance to ankle to distinguish left/right, 
                # if ankle is unavailable, use elbow, then use shoulder. 
                # The joints used by two arms should exactly the same)
                dist_left_arm = np.ones((num_bbox,)) * float('inf')
                dist_right_arm = np.ones((num_bbox,)) * float('inf')
                hand_bboxes = dict(
                    left_hand = None,
                    righ_hand = None
                )
                # left arm
                if body_pose[7][0]>0 and body_pose[6][0]>0:
                    # distance between elbow and ankle
                    dist_wrist_elbow = np.linalg.norm(body_pose[7]-body_pose[6])
                    for i in range(num_bbox):
                        bbox = raw_hand_bboxes[i]
                        c_x = (bbox[0]+bbox[2])/2
                        c_y = (bbox[1]+bbox[3])/2
                        center = np.array([c_x, c_y])
                        dist_bbox_ankle = np.linalg.norm(center - body_pose[7])
                        if dist_bbox_ankle < dist_wrist_elbow*1.5:
                            dist_left_arm[i] = np.linalg.norm(center - body_pose[7])
                # right arm
                if body_pose[4][0]>0 and body_pose[3][0]>0:
                    # distance between elbow and ankle
                    dist_wrist_elbow = np.linalg.norm(body_pose[3]-body_pose[4])
                    for i in range(num_bbox):
                        bbox = raw_hand_bboxes[i]
                        c_x = (bbox[0]+bbox[2])/2
                        c_y = (bbox[1]+bbox[3])/2
                        center = np.array([c_x, c_y])
                        dist_bbox_ankle = np.linalg.norm(center - body_pose[4])
                        if dist_bbox_ankle < dist_wrist_elbow*1.5:
                            dist_right_arm[i] = np.linalg.norm(center - body_pose[4])

                # assign bboxes
                hand_bboxes = dict()
                left_id = np.argmin(dist_left_arm)
                right_id = np.argmin(dist_right_arm)

                if dist_left_arm[left_id] < float('inf'):
                    hand_bboxes['left_hand'] = raw_hand_bboxes[left_id]
                if dist_right_arm[right_id] < float('inf'):
                    hand_bboxes['right_hand'] = raw_hand_bboxes[right_id]
                hand_bbox_list.append(hand_bboxes)

        return body_pose_list, hand_bbox_list, raw_hand_bboxes
    

class Ego_Centric_Detector(object):
    """
    Hand Detector for ego-centric input.
    It uses type-aware hand detector:
    (https://github.com/ddshan/hand_object_detector)
    """
    def __init__(self):
        pass


class HandBboxDetector(object):
    def __init__(self, view_type, device):
        """
        args:
            view_type: third_view or ego_centric.
        """
        self.view_type = view_type

        if view_type == "ego_centric":
            print("Loading Ego Centric Hand Detector")
            self.model = Ego_Centric_Detector()
        elif view_type == "third_view":
            print("Loading Third View Hand Detector")
            self.model = Third_View_Detector()
        else :
            print("Invalid view_type")
            assert False

    def detect_hand_bbox(self, img_bgr):
        """
        args:
            img_bgr: Raw image with BGR order (cv2 default). Currently assumes BGR
        output:
            bbox_list: list of bboxes. Each bbox has XHWH form (min_, min_y, width, height)

        """
        return self.model.detect_hand_bbox(img_bgr)