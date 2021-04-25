# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import torch
import numpy as np
import cv2
from torchvision.transforms import transforms

from handmocap.hand_modules.test_options import TestOptions
from handmocap.hand_modules.h3dw_model import H3DWModel
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm


class HandMocap:
    def __init__(self, regressor_checkpoint, smpl_dir, device = torch.device('cuda') , use_smplx = False):
        #For image transform
        transform_list = [ transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.normalize_transform = transforms.Compose(transform_list)

        #Load Hand network 
        self.opt = TestOptions().parse([])

        #Default options
        self.opt.single_branch = True
        self.opt.main_encoder = "resnet50"
        # self.opt.data_root = "/home/hjoo/dropbox/hand_yu/data/"
        self.opt.model_root = "./extra_data"
        self.opt.smplx_model_file = os.path.join(smpl_dir,'SMPLX_NEUTRAL.pkl')

        self.opt.batchSize = 1
        self.opt.phase = "test"
        self.opt.nThreads = 0
        self.opt.which_epoch = -1
        self.opt.checkpoint_path = regressor_checkpoint

        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.process_rank = -1

        # self.opt.which_epoch = str(epoch)
        self.model_regressor = H3DWModel(self.opt)
        # if there is no specified checkpoint, then skip
        assert self.model_regressor.success_load, "Specificed checkpoints does not exists: {}".format(self.opt.checkpoint_path)
        self.model_regressor.eval()


    def __pad_and_resize(self, img, hand_bbox, add_margin, final_size=224):
        ori_height, ori_width = img.shape[:2]
        min_x, min_y = hand_bbox[:2].astype(np.int32)
        width, height = hand_bbox[2:].astype(np.int32)
        max_x = min_x + width
        max_y = min_y + height

        if width > height:
            margin = (width-height) // 2
            min_y = max(min_y-margin, 0)
            max_y = min(max_y+margin, ori_height)
        else:
            margin = (height-width) // 2
            min_x = max(min_x-margin, 0)
            max_x = min(max_x+margin, ori_width)
        
        # add additional margin
        if add_margin:
            margin = int(0.3 * (max_y-min_y)) # if use loose crop, change 0.3 to 1.0
            min_y = max(min_y-margin, 0)
            max_y = min(max_y+margin, ori_height)
            min_x = max(min_x-margin, 0)
            max_x = min(max_x+margin, ori_width)

        img_cropped = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
        new_size = max(max_x-min_x, max_y-min_y)
        new_img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
        # new_img = np.zeros((new_size, new_size, 3))
        new_img[:(max_y-min_y), :(max_x-min_x), :] = img_cropped
        bbox_processed = (min_x, min_y, max_x, max_y)

        # resize to 224 * 224
        new_img = cv2.resize(new_img, (final_size, final_size))

        ratio = final_size / new_size
        return new_img, ratio, (min_x, min_y, max_x-min_x, max_y-min_y)
  

    def __process_hand_bbox(self, raw_image, hand_bbox, hand_type, add_margin=True):
        """
        args: 
            original image, 
            bbox: (x0, y0, w, h)
            hand_type ("left_hand" or "right_hand")
            add_margin: If the input hand bbox is a tight bbox, then set this value to True, else False
        output:
            img_cropped: 224x224 cropped image (original colorvalues 0-255)
            norm_img: 224x224 cropped image (normalized color values)
            bbox_scale_ratio: scale factor to convert from original to cropped
            bbox_top_left_origin: top_left corner point in original image cooridate
        """
        # print("hand_type", hand_type)

        assert hand_type in ['left_hand', 'right_hand']
        img_cropped, bbox_scale_ratio, bbox_processed = \
            self.__pad_and_resize(raw_image, hand_bbox, add_margin)
        
        #horizontal Flip to make it as right hand
        if hand_type=='left_hand':
            img_cropped = np.ascontiguousarray(img_cropped[:, ::-1,:], img_cropped.dtype) 
        else:
            assert hand_type == 'right_hand'

        # img normalize
        norm_img = self.normalize_transform(img_cropped).float()
        # return
        return img_cropped, norm_img, bbox_scale_ratio, bbox_processed


    def regress(self, img_original, hand_bbox_list, add_margin=False):
        """
            args: 
                img_original: original raw image (BGR order by using cv2.imread)
                hand_bbox_list: [
                    dict(
                        left_hand = [x0, y0, w, h] or None
                        right_hand = [x0, y0, w, h] or None
                    )
                    ...
                ]
                add_margin: whether to do add_margin given the hand bbox
            outputs:
                To be filled
            Note: 
                Output element can be None. This is to keep the same output size with input bbox
        """
        pred_output_list = list()
        hand_bbox_list_processed = list()

        for hand_bboxes in hand_bbox_list:

            if hand_bboxes is None:     # Should keep the same size with bbox size
                pred_output_list.append(None)
                hand_bbox_list_processed.append(None)
                continue

            pred_output = dict(
                left_hand = None,
                right_hand = None
            )
            hand_bboxes_processed = dict(
                left_hand = None,
                right_hand = None
            )

            for hand_type in hand_bboxes:
                bbox = hand_bboxes[hand_type]
                
                if bbox is None: 
                    continue
                else:
                    img_cropped, norm_img, bbox_scale_ratio, bbox_processed = \
                        self.__process_hand_bbox(img_original, hand_bboxes[hand_type], hand_type, add_margin)
                    hand_bboxes_processed[hand_type] = bbox_processed

                    with torch.no_grad():
                        # pred_rotmat, pred_betas, pred_camera = self.model_regressor(norm_img.to(self.device))
                        self.model_regressor.set_input_imgonly({'img': norm_img.unsqueeze(0)})
                        self.model_regressor.test()
                        pred_res = self.model_regressor.get_pred_result()

                        ##Output
                        cam = pred_res['cams'][0, :]  #scale, tranX, tranY
                        pred_verts_origin = pred_res['pred_verts'][0]
                        faces = self.model_regressor.right_hand_faces_local
                        pred_pose = pred_res['pred_pose_params'].copy()
                        pred_joints = pred_res['pred_joints_3d'].copy()[0]

                        if hand_type == 'left_hand':
                            cam[1] *= -1
                            pred_verts_origin[:, 0] *= -1
                            faces = faces[:, ::-1]
                            pred_pose[:, 1::3] *= -1
                            pred_pose[:, 2::3] *= -1
                            pred_joints[:, 0] *= -1

                        pred_output[hand_type] = dict()
                        pred_output[hand_type]['pred_vertices_smpl'] = pred_verts_origin # SMPL-X hand vertex in bbox space
                        pred_output[hand_type]['pred_joints_smpl'] = pred_joints
                        pred_output[hand_type]['faces'] = faces

                        pred_output[hand_type]['bbox_scale_ratio'] = bbox_scale_ratio
                        pred_output[hand_type]['bbox_top_left'] = np.array(bbox_processed[:2])
                        pred_output[hand_type]['pred_camera'] = cam
                        pred_output[hand_type]['img_cropped'] = img_cropped

                        # pred hand pose & shape params & hand joints 3d
                        pred_output[hand_type]['pred_hand_pose'] = pred_pose # (1, 48): (1, 3) for hand rotation, (1, 45) for finger pose.
                        pred_output[hand_type]['pred_hand_betas'] = pred_res['pred_shape_params'] # (1, 10)

                        #Convert vertices into bbox & image space
                        cam_scale = cam[0]
                        cam_trans = cam[1:]
                        vert_smplcoord = pred_verts_origin.copy()
                        joints_smplcoord = pred_joints.copy()
                        
                        vert_bboxcoord = convert_smpl_to_bbox(
                            vert_smplcoord, cam_scale, cam_trans, bAppTransFirst=True) # SMPL space -> bbox space
                        joints_bboxcoord = convert_smpl_to_bbox(
                            joints_smplcoord, cam_scale, cam_trans, bAppTransFirst=True) # SMPL space -> bbox space

                        hand_boxScale_o2n = pred_output[hand_type]['bbox_scale_ratio']
                        hand_bboxTopLeft = pred_output[hand_type]['bbox_top_left']

                        vert_imgcoord = convert_bbox_to_oriIm(
                                vert_bboxcoord, hand_boxScale_o2n, hand_bboxTopLeft, 
                                img_original.shape[1], img_original.shape[0]) 
                        pred_output[hand_type]['pred_vertices_img'] = vert_imgcoord

                        joints_imgcoord = convert_bbox_to_oriIm(
                                joints_bboxcoord, hand_boxScale_o2n, hand_bboxTopLeft, 
                                img_original.shape[1], img_original.shape[0]) 
                        pred_output[hand_type]['pred_joints_img'] = joints_imgcoord

            pred_output_list.append(pred_output)
            hand_bbox_list_processed.append(hand_bboxes_processed)
        
        assert len(hand_bbox_list_processed) == len(hand_bbox_list)
        return pred_output_list
