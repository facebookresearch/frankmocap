# Copyright (c) Facebook, Inc. and its affiliates.

# Original code from SPIN: https://github.com/nkolot/SPIN

"""
This file contains functions that are used to perform data augmentation.
"""
import sys
import torch
import numpy as np
import scipy.misc
import cv2
from torchvision.transforms import Normalize

# For converting coordinate between SMPL 3D coord <-> 2D bbox <-> original 2D image 
# data3D: (N,3), where N is number of 3D points in "smpl"'s 3D coordinate (vertex or skeleton)

def convert_smpl_to_bbox(data3D, scale, trans, bAppTransFirst=False):
    data3D = data3D.copy()
    resnet_input_size_half = 224 *0.5
    if bAppTransFirst:      # Hand model
        data3D[:,0:2] += trans
        data3D *= scale   # apply scaling
    else:
        data3D *= scale # apply scaling
        data3D[:,0:2] += trans
    
    data3D*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    # data3D[:,:2]*= resnet_input_size_half # 112 is originated from hrm's input size (224,24)
    return data3D


def convert_bbox_to_oriIm(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH):
    data3D = data3D.copy()
    resnet_input_size_half = 224 *0.5
    imgSize = np.array([imgSizeW,imgSizeH])

    data3D /= boxScale_o2n

    if not isinstance(bboxTopLeft, np.ndarray):
        assert isinstance(bboxTopLeft, tuple)
        assert len(bboxTopLeft) == 2
        bboxTopLeft = np.array(bboxTopLeft)

    data3D[:,:2] += (bboxTopLeft + resnet_input_size_half/boxScale_o2n)

    return data3D


def convert_smpl_to_bbox_perspective(data3D, scale_ori, trans_ori, focalLeng, scaleFactor=1.0):
    data3D = data3D.copy()
    resnet_input_size_half = 224 *0.5

    scale = scale_ori* resnet_input_size_half
    trans = trans_ori *resnet_input_size_half

    if False:   #Weak perspective
        data3D *= scale           #apply scaling
        data3D[:,0:2] += trans
    else:
        # delta = (trans - imageShape*0.5)/scale            
        # Current projection already consider camera center during the rendering. 
        # Thus no need to consider principle axis
        delta = (trans )/scale
        data3D[:,0:2] +=delta

        newZ = focalLeng/scale
        deltaZ =  newZ - np.mean(data3D[:,2])
        data3D[:,2] +=deltaZ
        # data3D[:,2] +=16.471718554146534        #debug

    if False:   #Scaling to be a certain dist from camera
        texture_plan_depth = 500
        ratio = texture_plan_depth /np.mean(data3D[:,2])
        data3D *=ratio  
    else:
        data3D *=scaleFactor

    return data3D


""" Extract bbox information """
def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        data = json.load(f)
        if 'people' not in data or len(data['people'])==0:
            return None, None
        # keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
        keypoints = data['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh

    valid_keypoints = keypoints[valid][:,:-1]           #(25,2)

    # min_pt = np.min(valid_keypoints, axis=0)
    # max_pt = np.max(valid_keypoints, axis=0)
    # bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale#, bbox


# keypoints: (Nx3)
def bbox_from_keypoint2d(keypoints, rescale=1.2, detection_thresh=0.2):
    """
        output:
            center: bbox center
            scale: scale_n2o: 224x224 -> original bbox size (max length if not a square bbox)
    """
    # """Get center and scale for bounding box from openpose detections."""

    if len(keypoints.shape)==2 and keypoints.shape[1]==2:       #(X,2)
        valid_keypoints = keypoints
    else:
        keypoints = np.reshape(np.array(keypoints), (-1,3))
        valid = keypoints[:,-1] > detection_thresh

        valid_keypoints = keypoints[valid][:,:-1]           #(25,2)

    # min_pt = np.min(valid_keypoints, axis=0)
    # max_pt = np.max(valid_keypoints, axis=0)
    # bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()


    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale#, bbox


def bbox_from_keypoints(keypoints, rescale=1.2, detection_thresh=0.2, imageHeight= None):
    """Get center and scale for bounding box from openpose detections."""
  
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh

    valid_keypoints = keypoints[valid][:,:-1]           #(25,2)

    if len(valid_keypoints)<2:
        return None, None, None


    if False:            #Should have all limbs and nose
        if np.sum(valid[ [ 2,3,4, 5,6,7, 9,10, 12,13,1,0] ]) <12:
            return None, None, None

    min_pt = np.min(valid_keypoints, axis=0)
    max_pt = np.max(valid_keypoints, axis=0)

    bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

    if imageHeight is not None:

        if valid[10]==False and valid[13] == False:  # No knees ub ioeb
            max_pt[1] = min(max_pt[1] + (max_pt[1]- min_pt[1]), imageHeight )
            bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]
            valid_keypoints = np.vstack( (valid_keypoints, np.array(max_pt)) )


        elif valid[11]==False and valid[14] == False: #No foot
            max_pt[1] = min(max_pt[1] + (max_pt[1]- min_pt[1])*0.2, imageHeight )
            bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

            valid_keypoints = np.vstack( (valid_keypoints, np.array(max_pt)) )


    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale, bbox


def bbox_from_bbr(bbox_XYWH, rescale=1.2, detection_thresh=0.2, imageHeight= None):
    #bbr: (minX, minY, width, height)
    """Get center and scale for bounding box from openpose detections."""

    center = bbox_XYWH[:2] + 0.5 * bbox_XYWH[2:]
    bbox_size = max(bbox_XYWH[2:])
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale#, bbox_XYWH


def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale