# Original code from SPIN: https://github.com/nkolot/SPIN

"""
This file contains functions that are used to perform data augmentation.
"""
import torch
import numpy as np
import scipy.misc
import cv2

# from bodymocap.core import constants
from torchvision.transforms import Normalize

## For converting coordinate between SMPL 3D coord <-> 2D bbox <-> original 2D image 
#data3D: (N,3), where N is number of 3D points in "smpl"'s 3D coordinate (vertex or skeleton)

# Note: camScal is based on normalized 2D keypoint (-1~1). 112 = 0.5 =224 (to come back to the original resolution)
# camScale and camTrans is for normalized coord.
# (camScale*(vert) + camTras )  ==> normalized coordinate  (-1 ~ 1)
# 112* ((camScale*(vert) + camTras )  + 1) == 112*camScale*vert +  112*camTrans + 112
def convert_smpl_to_bbox(data3D, scale, trans, bAppTransFirst=False):
    hmrIntputSize_half = 224 *0.5

    if bAppTransFirst:      #Hand model
        data3D[:,0:2] += trans
        data3D *= scale           #apply scaling
    else:
        data3D *= scale           #apply scaling
        data3D[:,0:2] += trans
    
    data3D*= hmrIntputSize_half         #112 is originated from hrm's input size (224,24)

    return data3D

def convert_bbox_to_oriIm(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH):
    hmrIntputSize_half = 224 *0.5

    # if type(imgSize) is tuple:
    #     imgSize = np.array(imgSize)
    imgSize = np.array([imgSizeW,imgSizeH])

    # pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape)
    data3D/=boxScale_o2n
    data3D[:,:2] += bboxTopLeft - imgSize*0.5 + hmrIntputSize_half/boxScale_o2n
    # data3D[:,1] += bboxTopLeft[1] - rawImg.shape[0]*0.5 + 112/boxScale_o2n
    return data3D



def convert_smpl_to_bbox_perspective(data3D, scale_ori, trans_ori, focalLeng, scaleFactor=1.0):
    hmrIntputSize_half = 224 *0.5

    scale = scale_ori* hmrIntputSize_half
    trans = trans_ori *hmrIntputSize_half

    if False:   #Weak perspective
        data3D *= scale           #apply scaling
        data3D[:,0:2] += trans
    # data3D*= hmrIntputSize_half         #112 is originated from hrm's input size (224,24)
    else:
        # delta = (trans - imageShape*0.5)/scale            #Current projection already consider camera center during the rendering. Thus no need to consider principle axis
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
        # height  = np.max(data3D[:,1]) - np.min(data3D[:,1])
        # # print(f"height: {height}")
        # # targetHeight = 380
        # ratio = targetHeight /height
        # data3D *=ratio  

        data3D *=scaleFactor

    return data3D


def convert_bbox_to_oriIm_perspective(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH, focalLeng):
    hmrIntputSize_half = 224 *0.5

    # if type(imgSize) is tuple:
    #     imgSize = np.array(imgSize)
    imgSize = np.array([imgSizeW,imgSizeH])

    # pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, boxScale_o2n, bboxTopLeft, rawImg.shape)
    if False:
        data3D/=boxScale_o2n
        data3D[:,:2] += bboxTopLeft - imgSize*0.5 + hmrIntputSize_half/boxScale_o2n
    else:
        scale = 1.0/boxScale_o2n
        # print(f"Scale: {scale}")
        # deltaZ =  focalLeng/scale - np.mean(data3D[:,2]) 
        deltaZ =  np.mean(data3D[:,2])/scale - np.mean(data3D[:,2]) 
        data3D[:,2] +=deltaZ
        # data3D[:,2] += 400

        trans = bboxTopLeft - imgSize*0.5 + hmrIntputSize_half/boxScale_o2n
        delta = np.mean(data3D[:,2]) /focalLeng *trans
        # delta = (trans )*boxScale_o2n
        data3D[:,:2] += delta

        # newZ = focalLeng/scale
        # deltaZ =  newZ - np.mean(data3D[:,2])
        # data3D[:,2] +=deltaZ


    # data3D[:,1] += bboxTopLeft[1] - rawImg.shape[0]*0.5 + 112/boxScale_o2n
    return data3D




## Conversion for Antrho


def anthro_crop_fromRaw(rawimage, bbox_XYXY):
    bbox_w = bbox_XYXY[2] - bbox_XYXY[0]
    bbox_h = bbox_XYXY[3] - bbox_XYXY[1]
    bbox_size = max(bbox_w, bbox_h)     #take the max
    bbox_center = (bbox_XYXY[:2] + bbox_XYXY[2:])*0.5
    pt_ul = (bbox_center - bbox_size*0.5).astype(np.int32)
    pt_br = (bbox_center + bbox_size*0.5).astype(np.int32)
    croppedImg = rawimage[pt_ul[1]:pt_br[1], pt_ul[0]:pt_br[0]]
    croppedImg = np.ascontiguousarray(croppedImg)
    return rawimage, pt_ul, pt_br

def anthro_convert_smpl_to_bbox(data3D, scale, trans, bbox_max_size):
    hmrIntputSize_half = bbox_max_size *0.5

    data3D *= scale           #apply scaling
    # data3D[:,0] += data3D[b,1]        #apply translation x
    # data3D[:,1] += data3D[b,2]        #apply translation y
    data3D[:,0:2] += trans
    data3D*= hmrIntputSize_half         #112 is originated from hrm's input size (224,24)

    return data3D

# def anthro_convert_bbox_to_oriIm(data3D, boxScale_o2n, bboxTopLeft, imgSizeW, imgSizeH):
def anthro_convert_bbox_to_oriIm(pred_vert_vis, rawImg_w, rawImg_h, bbox_pt_ul, bbox_max_size):
    pred_vert_vis[:,:2] +=  bbox_pt_ul - np.array((rawImg_w, rawImg_h))*0.5 +(bbox_max_size*0.5)  # + hmrIntputSize_half#+ hmrIntputSize_half 
    return pred_vert_vis



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




# def bbox_from_bboxXYXY(bboxXYXY, rescale=1.2):
#     """
#     bboxXYXY
#     """
    # pass

def bbox_from_keypoints(keypoints, rescale=1.2, detection_thresh=0.2, imageHeight= None):
    """Get center and scale for bounding box from openpose detections."""
    # with open(openpose_file, 'r') as f:
    #     data = json.load(f)
    #     if 'people' not in data or len(data['people'])==0:
    #         return None, None
    #     # keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    #     keypoints = data['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh

    # if g_debugUpperBodyOnly:    #Intentionally remove lower bodies
    #     valid[ [ 9,10,11,12,13,14, 22,23,24, 19,20,21] ] = False

    valid_keypoints = keypoints[valid][:,:-1]           #(25,2)

    if len(valid_keypoints)<2:
        return None, None, None


    if False:            #Should have all limbs and nose
        if np.sum(valid[ [ 2,3,4, 5,6,7, 9,10, 12,13,1,0] ]) <12:
            return None, None, None

    min_pt = np.min(valid_keypoints, axis=0)
    max_pt = np.max(valid_keypoints, axis=0)

    
    bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]



    # print(valid_keypoints)
    # print(valid)
    print(bbox)

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


# def process_image_keypoints(img, keypoints, input_res=224):
#     """Read image, do preprocessing and possibly crop it according to the bounding box.
#     If there are bounding box annotations, use them to crop the image.
#     If no bounding box is specified but openpose detections are available, use them to get the bounding box.
#     """
#     normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
#     img = img[:,:,::-1].copy() # PyTorch does not support negative stride at the moment

#     center, scale, bbox = bbox_from_keypoints(keypoints, imageHeight = img.shape[0])
#     if center is None:
#         return None, None, None, None, None

#     img, boxScale_o2n, bboxTopLeft = crop_bboxInfo(img, center, scale, (input_res, input_res))

#     # viewer2D.ImShow(img, name='cropped', waitTime=1)        #224,224,3


#     if img is None:
#         return None, None, None, None, None


#     # unCropped = uncrop(img, center, scale, (input_res, input_res))

#     # if True:
#     #     viewer2D.ImShow(img)
#     img = img.astype(np.float32) / 255.
#     img = torch.from_numpy(img).permute(2,0,1)
#     norm_img = normalize_img(img.clone())[None]
#     # return img, norm_img, img_original, boxScale_o2n, bboxTopLeft, bbox
#     bboxInfo ={"center": center, "scale": scale, "bboxXYWH":bbox}
#     return img, norm_img, boxScale_o2n, bboxTopLeft, bboxInfo


#bbr: (minX, minY, width, height)
def bbox_from_bbr(bbox_XYWH, rescale=1.2, detection_thresh=0.2, imageHeight= None):
    """Get center and scale for bounding box from openpose detections."""

    # bbox= bbr
    # if imageHeight is not None:

    #     if valid[10]==False and valid[13] == False:  # No knees ub ioeb
    #         max_pt[1] = min(max_pt[1] + (max_pt[1]- min_pt[1]), imageHeight )
    #         bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]
    #         valid_keypoints = np.vstack( (valid_keypoints, np.array(max_pt)) )


    #     elif valid[11]==False and valid[14] == False: #No foot
    #         max_pt[1] = min(max_pt[1] + (max_pt[1]- min_pt[1])*0.2, imageHeight )
    #         bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

    #         valid_keypoints = np.vstack( (valid_keypoints, np.array(max_pt)) )


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



# def process_image_bbox(img_original, bbox_XYWH, input_res=224):
#     """Read image, do preprocessing and possibly crop it according to the bounding box.
#     If there are bounding box annotations, use them to crop the image.
#     If no bounding box is specified but openpose detections are available, use them to get the bounding box.
#     """
#     normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
#     img_original = img_original[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
#     img = img_original.copy()

#     center, scale = bbox_from_bbr(bbox_XYWH, imageHeight = img.shape[0])
#     if center is None:
#         return None, None,  None, None, None

#     img, boxScale_o2n, bboxTopLeft = crop_bboxInfo(img, center, scale, (input_res, input_res))

#     # viewer2D.ImShow(img, name='cropped', waitTime=1)        #224,224,3


#     if img is None:
#         return None, None,  None, None, None


#     # unCropped = uncrop(img, center, scale, (input_res, input_res))

#     # if True:
#     #     viewer2D.ImShow(img)
#     img = img.astype(np.float32) / 255.convert_bbox_to_oriIm