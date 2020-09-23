# Copyright (c) Facebook, Inc. and its affiliates.

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
import os
import os.path as osp
import sys
import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
# from PIL import Image

from bodymocap.body_bbox_detector import BodyPoseEstimator

# Type agnostic hand detector
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.utils.visualizer import Visualizer

# from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
# from detectron2.modeling import GeneralizedRCNNWithTTA
# from detectron2.data.datasets import register_coco_instances

# Type-aware hand (hand-object) hand detector
hand_object_detector_path = './detectors/hand_object_detector'
sys.path.append(hand_object_detector_path)
from model.utils.config import cfg as cfgg

from detectors.hand_object_detector.lib.model.rpn.bbox_transform import clip_boxes
from detectors.hand_object_detector.lib.model.roi_layers import nms # might raise segmentation fault at the end of program
from detectors.hand_object_detector.lib.model.rpn.bbox_transform import bbox_transform_inv
from detectors.hand_object_detector.lib.model.utils.blob import im_list_to_blob
from detectors.hand_object_detector.lib.model.faster_rcnn.resnet import resnet as detector_resnet 


class Third_View_Detector(BodyPoseEstimator):
    """
    Hand Detector for third-view input.
    It combines a body pose estimator (https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git)
    with a type-agnostic hand detector (https://github.com/ddshan/hand_detector.d2)
    """
    def __init__(self):
        super(Third_View_Detector, self).__init__()
        print("Loading Third View Hand Detector")
        self.__load_hand_detector()
    

    def __load_hand_detector(self):
         # load cfg and model
        cfg = get_cfg()
        cfg.merge_from_file("detectors/hand_only_detector/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
        cfg.MODEL.WEIGHTS = 'extra_data/hand_module/hand_detector/model_0529999.pth' # add model weight here
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # 0.3 , use low thresh to increase recall
        self.hand_detector = DefaultPredictor(cfg)


    def __get_raw_hand_bbox(self, img):
        bbox_tensor = self.hand_detector(img)['instances'].pred_boxes
        bboxes = bbox_tensor.tensor.cpu().numpy()
        return bboxes


    def detect_hand_bbox(self, img):
        '''
            output: 
                body_bbox: [min_x, min_y, width, height]
                hand_bbox: [x0, y0, x1, y1]
            Note:
                len(body_bbox) == len(hand_bbox), where hand_bbox can be None if not valid
        '''
        # get body pose
        body_pose_list, body_bbox_list = self.detect_body_pose(img)
        # assert len(body_pose_list) == 1, "Current version only supports one person"

        # get raw hand bboxes
        raw_hand_bboxes = self.__get_raw_hand_bbox(img)
        hand_bbox_list = [None, ] * len(body_pose_list)
        num_bbox = raw_hand_bboxes.shape[0]

        if num_bbox > 0:
            for idx, body_pose in enumerate(body_pose_list):
                # By default, we use distance to ankle to distinguish left/right, 
                # if ankle is unavailable, use elbow, then use shoulder. 
                # The joints used by two arms should exactly the same)
                dist_left_arm = np.ones((num_bbox,)) * float('inf')
                dist_right_arm = np.ones((num_bbox,)) * float('inf')
                hand_bboxes = dict(
                    left_hand = None,
                    right_hand = None
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
                # hand_bboxes = dict()
                left_id = np.argmin(dist_left_arm)
                right_id = np.argmin(dist_right_arm)

                if dist_left_arm[left_id] < float('inf'):
                    hand_bboxes['left_hand'] = raw_hand_bboxes[left_id].copy()
                if dist_right_arm[right_id] < float('inf'):
                    hand_bboxes['right_hand'] = raw_hand_bboxes[right_id].copy()

                hand_bbox_list[idx] = hand_bboxes


        assert len(body_bbox_list) == len(hand_bbox_list)
        return body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes
    

class Ego_Centric_Detector(BodyPoseEstimator):
    """
    Hand Detector for ego-centric input.
    It uses type-aware hand detector:
    (https://github.com/ddshan/hand_object_detector)
    """
    def __init__(self):
        super(Ego_Centric_Detector, self).__init__()
        print("Loading Ego Centric Hand Detector")
        self.__load_hand_detector()
    

    # part of the code comes from https://github.com/ddshan/hand_object_detector
    def __load_hand_detector(self):
        classes = np.asarray(['__background__', 'targetobject', 'hand']) 
        fasterRCNN = detector_resnet(classes, 101, pretrained=False, class_agnostic=False)
        fasterRCNN.create_architecture()
        self.classes = classes

        checkpoint_path = "extra_data/hand_module/hand_detector/faster_rcnn_1_8_132028.pth"
        checkpoint = torch.load(checkpoint_path)
        assert osp.exists(checkpoint_path), "Hand checkpoint does not exist"
        fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfgg.POOLING_MODE = checkpoint['pooling_mode']
        
        fasterRCNN.cuda()
        fasterRCNN.eval()
        self.hand_detector = fasterRCNN
    

    # part of the code comes from https://github.com/ddshan/hand_object_detector/demo.py
    def __get_image_blob(self, im):
        im_orig = im.astype(np.float32, copy=True)
        pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        im_orig -= pixel_means

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = list()
        im_scale_factors = list()

        test_scales = [600,]
        for target_size in test_scales:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfgg.TEST.MAX_SIZE:
                im_scale = float(cfgg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)
        return blob, np.array(im_scale_factors)


    # part of the code comes from https://github.com/ddshan/hand_object_detector/demo.py
    def __get_raw_hand_bbox(self, img):
        with torch.no_grad():
            im_data = torch.FloatTensor(1).cuda()
            im_info = torch.FloatTensor(1).cuda()
            num_boxes = torch.LongTensor(1).cuda()
            gt_boxes = torch.FloatTensor(1).cuda()
            box_info = torch.FloatTensor(1).cuda()

            im_blob, im_scales = self.__get_image_blob(img)

            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_() 

            # forward
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, loss_list = self.hand_detector(im_data, im_info, gt_boxes, num_boxes, box_info) 

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # hand side info (left/right)
            lr_vector = loss_list[2][0].detach()
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()

            box_deltas = bbox_pred.data
            stds = [0.1, 0.1, 0.2, 0.2]
            means = [0.0, 0.0, 0.0, 0.0]
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(stds).cuda() \
                + torch.FloatTensor(means).cuda()
            box_deltas = box_deltas.view(1, -1, 4 * len(self.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

            pred_boxes /= im_scales[0]
            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            thresh_hand = 0.5
            j = 2
            inds = torch.nonzero(scores[:, j]>thresh_hand, as_tuple=False).view(-1)

            if inds.numel()>0:
                cls_boxes = pred_boxes[inds][:, j*4 : (j+1)*4]
                cls_scores = scores[:,j][inds]

                _, order = torch.sort(cls_scores, 0, True)

                lr = lr[inds][order]
                cls_boxes = cls_boxes[order]
                cls_scores = cls_scores[order]

                keep = nms(cls_boxes, cls_scores, cfgg.TEST.NMS) # cfgg.TEST.NMS : 0.3
                cls_boxes = cls_boxes[keep]
                lr_selected = lr[keep]

                cls_boxes_np = cls_boxes.detach().cpu().numpy()
                lr_np = lr_selected.detach().cpu().numpy()

                return cls_boxes_np, lr_np[:, 0]
                # return cls_boxes_np, lr_np
            else:
                return None, None
        

    def detect_hand_bbox(self, img):
        hand_bbox_list = list()
        hand_bbox_list.append(
            dict(
                left_hand = None,
                right_hand = None
            )
        )

        bboxes, hand_types = self.__get_raw_hand_bbox(img)

        if bboxes is not None:
            assert bboxes.shape[0] <= 2, "Ego centric version only supports one person per image"

            left_bbox = bboxes[hand_types==0]
            if len(left_bbox)>0:
                hand_bbox_list[0]['left_hand'] = left_bbox[0]

            right_bbox = bboxes[hand_types==1]
            if len(right_bbox)>0:
                hand_bbox_list[0]['right_hand'] = right_bbox[0]
            
        body_bbox_list = [None, ] * len(hand_bbox_list)
        return None, body_bbox_list, hand_bbox_list, None


class HandBboxDetector(object):
    def __init__(self, view_type, device):
        """
        args:
            view_type: third_view or ego_centric.
        """
        self.view_type = view_type

        if view_type == "ego_centric":
            self.model = Ego_Centric_Detector()
        elif view_type == "third_view":
            self.model = Third_View_Detector()
        else :
            print("Invalid view_type")
            assert False
    

    def detect_body_bbox(self, img_bgr):
        return self.model.detect_body_pose(img_bgr)
    

    def detect_hand_bbox(self, img_bgr):
        """
        args:
            img_bgr: Raw image with BGR order (cv2 default). Currently assumes BGR
        output:
            body_pose_list: body poses
            bbox_bbox_list: list of bboxes. Each bbox has XHWH form (min_x, min_y, width, height)
            hand_bbox_list: each element is 
            dict(
                left_hand = None / [min_x, min_y, width, height]
                right_hand = None / [min_x, min_y, width, height]
            )
            raw_hand_bboxes: list of raw hand detection, each element is [min_x, min_y, width, height]
        """
        output = self.model.detect_hand_bbox(img_bgr)
        body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = output

        # convert raw_hand_bboxes from (x0, y0, x1, y1) to (x0, y0, w, h)
        if raw_hand_bboxes is not None:
            for i in range(raw_hand_bboxes.shape[0]):
                bbox = raw_hand_bboxes[i]
                x0, y0, x1, y1 = bbox
                raw_hand_bboxes[i] = np.array([x0, y0, x1-x0, y1-y0])
        
        # convert hand_bbox_list from (x0, y0, x1, y1) to (x0, y0, w, h)
        for hand_bbox in hand_bbox_list:
            if hand_bbox is not None:
                for hand_type in hand_bbox:
                    bbox = hand_bbox[hand_type]
                    if bbox is not None:
                        x0, y0, x1, y1 = bbox
                        hand_bbox[hand_type] = np.array([x0, y0, x1-x0, y1-y0])

        return body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes
