# Copyright (c) Facebook, Inc. and its affiliates.
import os, sys, shutil
import os.path as osp
import torch
import numpy as np
import cv2
from torchvision.transforms import transforms
from handmocap.options.test_options import TestOptions
from handmocap.hand_modules.h3dw_model import H3DWModel


class HandMocap:
    def __init__(self, regressor_checkpoint, smpl_dir, device = torch.device('cuda') , bUseSMPLX = False):
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
        self.opt.model_root = "./data"
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
        assert self.model_regressor.success_load, "Specificed checkpoints does not exists"
        self.model_regressor.eval()

        #Save mesh faces
        self.rhand_mesh_face = self.model_regressor.right_hand_faces_local.copy()
        self.lhand_mesh_face = self.model_regressor.right_hand_faces_local.copy()[:,::-1]
    

    def __pad_and_resize(self, img, hand_bbox, add_margin, final_size=224):
        ori_height, ori_width = img.shape[:2]
        min_x, min_y = hand_bbox[:2].astype(np.int32)
        max_x, max_y = hand_bbox[2:].astype(np.int32)

        width = max_x - min_x
        height = max_y - min_y
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
            margin = int(0.3 * (max_y-min_y)) # if use loose crop, change 0.03 to 0.1
            min_y = max(min_y-margin, 0)
            max_y = min(max_y+margin, ori_height)
            min_x = max(min_x-margin, 0)
            max_x = min(max_x+margin, ori_width)

        img_cropped = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
        new_size = max(max_x-min_x, max_y-min_y)
        new_img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
        # new_img = np.zeros((new_size, new_size, 3))
        new_img[:(max_y-min_y), :(max_x-min_x), :] = img_cropped
        new_bbox = (min_x, min_y, max_x, max_y)

        # resize to 224 * 224
        new_img = cv2.resize(new_img, (final_size, final_size))

        ratio = final_size / new_size
        return new_img, ratio, (min_x, min_y, max_x, max_y)
  

    def __process_hand_bbox(self, raw_image, hand_bbox, hand_type, add_margin=True):
        """
        args: 
            original image, 
            bbox: (x0, y0, x1, y1)
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
        img_cropped, bbox_scale_ratio, new_bbox = \
            self.__pad_and_resize(raw_image, hand_bbox, add_margin)

        #horizontal Flip to make it as right hand
        if hand_type=='left_hand':
            img_cropped = np.ascontiguousarray(img_cropped[:, ::-1,:], img_cropped.dtype) 
        else:
            assert hand_type == 'right_hand'

        # img normalize
        norm_img = self.normalize_transform(img_cropped).float()
        # return
        return img_cropped, norm_img, bbox_scale_ratio, new_bbox


    def regress(self, img_original, hand_bbox_list, add_margin=False):
        """
            args: 
                img_original: original raw image (BGR order by using cv2.imread)
                hand_bbox_list: [
                    dict(
                        left_hand = [x0, y0, x1, y1] or None
                        right_hand = [x0, y0, x1, y1] or None
                    )
                    ...
                ]
                add_margin: whether to do add_margin given the hand bbox
            outputs:
                To be filled
        """
        pred_output_list = list()
        hand_bbox_list_new = list()

        for hand_bboxes in hand_bbox_list:
            pred_output = dict(
                left_hand = None,
                right_hand = None
            )
            hand_bboxes_new = dict(
                left_hand = None,
                right_hand = None
            )

            for hand_type in hand_bboxes:
                bbox = hand_bboxes[hand_type]
                
                if bbox is None: 
                    continue
                else:
                    img_cropped, norm_img, bbox_scale_ratio, new_bbox = \
                        self.__process_hand_bbox(img_original, hand_bboxes[hand_type], hand_type, add_margin)
                    hand_bboxes_new[hand_type] = new_bbox

                    with torch.no_grad():
                        # pred_rotmat, pred_betas, pred_camera = self.model_regressor(norm_img.to(self.device))
                        self.model_regressor.set_input_imgonly({'img': norm_img.unsqueeze(0)})
                        self.model_regressor.test()
                        pred_res = self.model_regressor.get_pred_result()

                        ##Output
                        cam = pred_res['cams'][0, :]
                        pred_verts_origin = pred_res['pred_verts'][0]

                        if hand_type == 'left_hand':
                            cam[1] *= -1
                            pred_verts_origin[:, 0] *= -1

                        pred_output[hand_type] = dict()
                        pred_output[hand_type]['pred_vertices_origin'] = pred_verts_origin # SMPL-X hand vertex in bbox space
                        pred_output[hand_type]['faces'] = self.model_regressor.right_hand_faces_local

                        pred_output[hand_type]['bbox_scale_ratio'] = bbox_scale_ratio
                        pred_output[hand_type]['bbox_top_left'] = new_bbox[:2]
                        pred_output[hand_type]['cam'] = cam
                        pred_output[hand_type]['img_cropped'] = img_cropped

            pred_output_list.append(pred_output)
            hand_bbox_list_new.append(hand_bboxes_new)
        
        return hand_bbox_list_new, pred_output_list