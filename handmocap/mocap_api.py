# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np
from torchvision.transforms import transforms
import cv2
# from bodymocap.models import hmr, SMPL, SMPLX
# from bodymocap.core import config
# from bodymocap.utils.imutils import crop,crop_bboxInfo, process_image_bbox, process_image_keypoints, bbox_from_keypoints
# from bodymocap.utils.imutils import convert_smpl_to_bbox, convert_bbox_to_oriIm
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
from renderer import viewer2D, glViewer

from handmocap.options.test_options import TestOptions
from handmocap.handmodels.h3dw_model import H3DWModel

###  Bbox cropping + Image processing function. (TODO: this is way too complicated.. originated from SPIN's body part) ###
# keypoints: (Nx3)
def bbox_to_scale_center(bbox_XYWH, rescale=1.2):
    """
        output:
            center: bbox center
            scale: scale_n2o: scaling ratio to convert 224x224 -> original bbox size (max length if not a square bbox)
    """
    # """Get center and scale for bounding box from openpose detections."""


    # min_pt = np.min(valid_keypoints, axis=0)
    # max_pt = np.max(valid_keypoints, axis=0)
    # bbox= [ min_pt[0], min_pt[1], max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]]

    center = np.array([bbox_XYWH[0] + bbox_XYWH[2]*0.5, bbox_XYWH[1] + bbox_XYWH[3]*0.5] )       #Bbox center
    bbox_size = max(bbox_XYWH[2], bbox_XYWH[3])              #Bbox size (takes max if not a squre)


    # adjust bounding box tightness  (this is something used in SPIN. not important for hand)
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale#, bbox

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1


# Note: this is way too complicated than it should be. 
def crop_bboxInfo(img, center, scale, res =(224,224)):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,
                             res[1]+1], center, scale, res, invert=1))-1


    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    # new_img = np.zeros(new_shape)
    if new_shape[0] <1  or new_shape[1] <1:
        return None, None, None
    new_img = np.zeros(new_shape, dtype=np.uint8)

    if new_img.shape[0] ==0:
        return None, None, None

    #Compute bbox for Han's format
    bboxScale_o2n = res[0]/new_img.shape[0]             #224/ 531

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])

    if new_y[0] <0 or new_y[1]<0 or new_x[0] <0 or new_x[1]<0 :
        return None, None, None

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]

    # bboxTopLeft_inOriginal = (old_x[0], old_y[0] )
    bboxTopLeft_inOriginal = (ul[0], ul[1] )

    if new_img.shape[0] <20 or new_img.shape[1]<20:
        return None, None, None
    # print(bboxTopLeft_inOriginal)
    # from renderer import viewer2D
    # viewer2D.ImShow(new_img.astype(np.uint8),name='cropped')

    cropedImg224 = cv2.resize(new_img, res)

    # viewer2D.ImShow(new_img.astype(np.uint8),name='resized224',waitTime=0)

    return cropedImg224, bboxScale_o2n, np.array(bboxTopLeft_inOriginal)


#Padding and resizing
def process_img(croppedImg):
    padImgSize = max(croppedImg.shape)
    padImg = np.zeros((padImgSize,padImgSize,3), dtype=np.uint8)

    if croppedImg.shape[0]<padImgSize:
        start = padImgSize-croppedImg.shape[0]
        padImg[:-(padImgSize-croppedImg.shape[0]),:] = croppedImg
    elif croppedImg.shape[1]<padImgSize:
        start = padImgSize-croppedImg.shape[1]
        padImg[:,:-(padImgSize-croppedImg.shape[1])] = croppedImg
    else:
        padImg = croppedImg

    # if padImg.shape[0]!=224:
    bboxImg_224 = cv2.resize(padImg, (224,224))
    return bboxImg_224



class HandMocap:

    def __init__(self, regressor_checkpoint, smpl_dir, device = torch.device('cuda') , bUseSMPLX = False):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        self.opt.data_root = "/home/hjoo/dropbox/hand_yu/data/"
        self.opt.model_root = "/home/hjoo/dropbox/hand_yu/data/models/"
        # self.opt.demo_img_dir = "youtube/image_hand"
        self.opt.batchSize = 1
        self.opt.phase = "test"
        # self.opt.test_dataset = "demo"
        self.opt.nThreads = 0
        epoch = 190     #Need to load the encoder from 190 epoch
        self.opt.which_epoch = str(epoch)


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


    def _process_image_bbox(self, raw_image, bbox_XYWH, lr):
        """
        args: original raw image, bbox
        output:
            img_cropped: 224x224 cropped image (original colorvalues 0-255)
            norm_img: 224x224 cropped image (normalized color values)
            bboxScale_o2n: scale factor to convert from original to cropped
            bboxTopLeft_inOriginal: top_left corner point in original image cooridate
        """

        imgWidth = raw_image.shape[1]
        imgHeight = raw_image.shape[0]
        # imgCenter = np.array((imgWidth,imgHeight) ) * 0.5

        ##Get BBox for Hand
        center, scale = bbox_to_scale_center(bbox_XYWH)
        img_cropped, bboxScale_o2n, bboxTopLeft_inOriginal = crop_bboxInfo(raw_image, center, scale)        #Cropping image using bbox information

        if img_cropped is not None:
            # if lr=='rhand':
            #     viewer2D.ImShow(img_cropped, waitTime=0, name='croppedH')
            pass
        else:
            return None
        
        if lr=='lhand':
            img_cropped = np.ascontiguousarray(img_cropped[:, ::-1,:], img_cropped.dtype)        #horizontal Flip to make it as right hand

        norm_img = process_img(img_cropped)     #Crop and padding...(TODO: why do we need this again?)
        norm_img = self.normalize_transform(norm_img).float()

        return img_cropped, norm_img, bboxScale_o2n, bboxTopLeft_inOriginal
        # ##Crop and resize to 224
        # raw_image_vis = viewer2D.Vis_Bbox_minmaxPt(raw_image.copy(), lhand_pt2d_min, lhand_pt2d_max , color=(0,255,255))
        # raw_image_vis = viewer2D.Vis_Bbox_minmaxPt(raw_image_vis, rhand_pt2d_min, rhand_pt2d_max, color=(0,255,255))
        # viewer2D.ImShow(raw_image_vis,waitTime=0)

        # ##Crop and resize to 224


    def regress(self, img_original, bbox_XYWH, lr, bExport=True):
        """
            args: 
                img_original: original raw image (BGR order by using cv2.imread)
                bbox_XYWH: bounding box around the target: (minX,minY,width, height)
                lr: 'lhand' or 'rhand'
            outputs:
                Default output:
                    pred_vertices_img:
                    pred_joints_vis_img:
                if bExport==True
                    pred_rotmat
                    pred_betas
                    pred_camera
                    bbox: [bbr[0], bbr[1],bbr[0]+bbr[2], bbr[1]+bbr[3]])
                    bboxTopLeft:  bbox top left (redundant)
                    boxScale_o2n: bbox scaling factor (redundant) 
        """
        img, norm_img, boxScale_o2n, bboxTopLeft = self._process_image_bbox(img_original, bbox_XYWH, lr)
        norm_img =norm_img.unsqueeze(0)

        if img is None:
            return None

        with torch.no_grad():
            # pred_rotmat, pred_betas, pred_camera = self.model_regressor(norm_img.to(self.device))
            self.model_regressor.set_input_imgonly({'img': norm_img})
            self.model_regressor.test()
            pred_res = self.model_regressor.get_pred_result()

            #Visualize
            if False:
                i =0
                #Visualize Mesh
                camParam_scale = pred_res['cams'][i,0]
                camParam_trans = pred_res['cams'][i,1:]
                pred_vert_vis = pred_res['pred_verts'][i]        #778,3
                pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)
                mesh ={'ver':pred_vert_vis, "f":self.model_regressor.right_hand_faces_local }
                glViewer.setMeshData([mesh],bComputeNormal=True)

                #Visualize Skeleton
                pred_joints_vis = pred_res['pred_joints_3d'][i]        #21,3
                pred_joints_vis = convert_smpl_to_bbox(pred_joints_vis, camParam_scale, camParam_trans)
                pred_joints_vis = pred_joints_vis.ravel()[:,np.newaxis]
                glViewer.setSkeleton( [pred_joints_vis], jointType='hand_smplx')

                ################ Other 3D setup############### 
                # glViewer.setBackgroundTexture(croppedImg)
                # glViewer.setWindowSize(croppedImg.shape[1]*3, croppedImg.shape[0]*3)
                # glViewer.SetOrthoCamera(True)
                glViewer.show()

            ##Output
            predoutput ={}
            i=0
            camParam_scale = pred_res['cams'][i,0]
            camParam_trans = pred_res['cams'][i,1:]
            pred_vert_vis = pred_res['pred_verts'][i]
            hand_boxScale_o2n = boxScale_o2n
            hand_bboxTopLeft = bboxTopLeft

            pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans, bAppTransFirst=True)        #SMPL space -> bbox space
            if lr =="lhand":        #Flip
                pred_vert_vis[:,0] *= -1
            pred_vert_vis = convert_bbox_to_oriIm(pred_vert_vis, hand_boxScale_o2n, hand_bboxTopLeft, img_original.shape[1], img_original.shape[0]) 

            predoutput['pred_vertices_img'] =  pred_vert_vis #SMPL vertex in image space
            # predoutput['pred_joints_img'] = pred_joints_vis_img #SMPL joints in image space
            # if bExport:
            #     predoutput['pred_rotmat'] = pred_rotmat.detach().cpu().numpy()
            #     predoutput['pred_betas'] = pred_betas.detach().cpu().numpy()
            #     predoutput['pred_camera'] = pred_camera
            #     predoutput['bbox_xyxy'] = [bbox_XYWH[0], bbox_XYWH[1], bbox_XYWH[0]+bbox_XYWH[2], bbox_XYWH[1]+bbox_XYWH[3] ]
            #     predoutput['bboxTopLeft'] = bboxTopLeft
            #     predoutput['boxScale_o2n'] = boxScale_o2n
         
        return predoutput