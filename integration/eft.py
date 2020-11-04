# Copyright (c) Facebook, Inc. and its affiliates.

import sys
import numpy as np
import torch
import pdb
import mocap_utils.geometry_utils as gu
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
from bodymocap.utils.imutils import j2d_normalize, conv_bbox_xywh_to_center_scale
from bodymocap.wholebody_eft import Whole_Body_EFT
from integration.copy_and_paste import transfer_rotation


def integration_eft_optimization(
    body_module, pred_body_list, pred_hand_list, 
    body_bbox_list, openpose_kp_imgcoord, 
    img_original_bgr, is_debug_vis=False):
    """
    Peform EFT optimization given 2D keypoint
    """

    smplx_model = body_module.smpl
    is_vis = is_debug_vis
    image_shape = img_original_bgr.shape

    # get eft model
    eft = Whole_Body_EFT(body_module.model_regressor, smplx_model)

    # Convert Openpose image process to body bbox space
    bboxInfo = conv_bbox_xywh_to_center_scale(body_bbox_list[0], image_shape)
    openpose_bboxNormcoord={}       #BBox normalize coordinate (-1 to 1)

    # obtain normalized 
    # (1, 25, 3) for body, (1, 21, 3) for hand
    for key in ['pose_keypoints_2d', 'hand_right_keypoints_2d', 'hand_left_keypoints_2d']:
        openpose_bboxNormcoord[key] = j2d_normalize(openpose_kp_imgcoord[key], bboxInfo['center'], bboxInfo['scale'])[np.newaxis,:]
    
    if False:
        # img_vis = viewer2D.Vis_Skeleton_2D_general(smpl_output_imgspace['body_joints'][:,:2], image =raw_image, offsetXY = np.array( (raw_image.shape[1], raw_image.shape[0]) )*0.5)
        raw_image_vis = img_original_bgr.copy()
        raw_image_vis = viewer2D.Vis_Skeleton_2D_Openpose25(openpose_kp_imgcoord['pose_keypoints_2d'][:,:2], openpose_kp_imgcoord['pose_keypoints_2d'][:,2],image =raw_image_vis)
        raw_image_vis = viewer2D.Vis_Skeleton_2D_Openpose_hand(openpose_kp_imgcoord['hand_right_keypoints_2d'][:,:2], openpose_kp_imgcoord['hand_right_keypoints_2d'][:,2],image =raw_image_vis)
        raw_image_vis = viewer2D.Vis_Skeleton_2D_Openpose_hand(openpose_kp_imgcoord['hand_left_keypoints_2d'][:,:2], openpose_kp_imgcoord['hand_left_keypoints_2d'][:,2],image =raw_image_vis)
        viewer2D.ImShow(raw_image_vis, waitTime=0, name="raw_openpose")

    if is_vis:    
        import renderer.viewer2D as viewer2D 
        #Visualize Bbox and Openpose
        # eft.visualize_smpl(output,img_cropped)
        # eft.visualize_joints(smpl_output_bbox, img_cropped)
        # img_cropped = viewer2D.Vis_Skeleton_2D_general(smpl_output_bbox['body_joints'][:,:2], image =img_cropped, offsetXY = np.array(img_cropped.shape[:2])*0.5)
        # viewer2D.ImShow(img_cropped, waitTime=0)
        img_cropped_vis = pred_body_list[0]['img_cropped'].copy()#(img_cropped.permute(1,2,0).numpy()[:,:,::-1]*255).astype(np.uint8)
        img_cropped_vis = viewer2D.Vis_Skeleton_2D_Openpose25( (openpose_bboxNormcoord['pose_keypoints_2d'][0,:,:2]+1)*112, openpose_bboxNormcoord['pose_keypoints_2d'][0,:,2],image =img_cropped_vis)
        img_cropped_vis = viewer2D.Vis_Skeleton_2D_Openpose_hand((openpose_bboxNormcoord['hand_right_keypoints_2d'][0,:,:2]+1)*112, openpose_bboxNormcoord['hand_right_keypoints_2d'][0,:,2],image =img_cropped_vis)
        img_cropped_vis = viewer2D.Vis_Skeleton_2D_Openpose_hand((openpose_bboxNormcoord['hand_left_keypoints_2d'][0,:,:2]+1)*112, openpose_bboxNormcoord['hand_left_keypoints_2d'][0,:,2],image =img_cropped_vis)
        viewer2D.ImShow(img_cropped_vis, waitTime=1, scale=4.0, name="openpose_cropped")

    #Run EFT iterations
    integral_output_list = list()
    assert len(pred_body_list)==1 , print("optimization-based integration mode is only valid for a single person only images ") 
    for i in range(len(pred_body_list)):
        body_info = pred_body_list[i]
        hand_info = pred_hand_list[i]
        if body_info is None:
            integral_output_list.append(None)
            continue

        input_batch ={}
        input_batch['img'] = body_info['img_cropped_norm'] # input image [1,3,224,224]
        input_batch['init_rotmat'] = torch.from_numpy(body_info['pred_rotmat']).cuda() # input image     #[1,24,3,3]
        input_batch['body_joints_2d_bboxNormCoord'] = torch.from_numpy(openpose_bboxNormcoord['pose_keypoints_2d']).cuda()
        input_batch['rhand_joints_2d_bboxNormCoord'] =torch.from_numpy( openpose_bboxNormcoord['hand_right_keypoints_2d']).cuda()
        input_batch['lhand_joints_2d_bboxNormCoord'] = torch.from_numpy(openpose_bboxNormcoord['hand_left_keypoints_2d']).cuda()

        scale_ratio_image_to_bbox = pred_body_list[i]['bbox_scale_ratio']
        input_batch['rhand_pose'] = None
        input_batch['lhand_pose'] = None
        if hand_info is not None and hand_info['right_hand'] is not None:
            input_batch['rhand_pose'] = torch.from_numpy(hand_info['right_hand']['pred_hand_pose'][:, 3:]).cuda()
            input_batch['rhand_verts'] = torch.from_numpy( hand_info['right_hand']['pred_vertices_img']).cuda() * scale_ratio_image_to_bbox
            input_batch['rhand_joints'] = torch.from_numpy( hand_info['right_hand']['pred_joints_img']).cuda()  * scale_ratio_image_to_bbox
            input_batch['rhand_faces'] = hand_info['right_hand']['faces']
        else:
            input_batch['rhand_pose'] = None # torch.from_numpy(np.zeros( (1,45) , dtype= np.float32)).cuda()

        if hand_info is not None and hand_info['left_hand'] is not None:
            input_batch['lhand_pose'] = left_hand_pose = torch.from_numpy(hand_info['left_hand']['pred_hand_pose'][:, 3:]).cuda()
            input_batch['lhand_verts'] = torch.from_numpy( hand_info['left_hand']['pred_vertices_img']).cuda()  *scale_ratio_image_to_bbox
            input_batch['lhand_joints'] = torch.from_numpy( hand_info['left_hand']['pred_joints_img']).cuda()  *scale_ratio_image_to_bbox
            input_batch['lhand_faces'] = hand_info['left_hand']['faces']
        else:
            input_batch['lhand_pose'] = None # torch.from_numpy(np.zeros((1,45), dtype= np.float32)).cuda()

        # run eft 
        input_batch['img_cropped_rgb'] = body_info['img_cropped'] 
        pred_rotmat, pred_betas, pred_camera = eft.eft_run(input_batch, eftIterNum=20, is_vis=is_debug_vis)

        # Save output
        body_info['eft_pred_betas'] = pred_betas.detach().cpu().numpy()
        body_info['eft_pred_rotmat'] = pred_rotmat.detach().cpu().numpy()
        body_info['eft_pred_camera'] = pred_camera.detach().cpu().numpy()[0]

        # original prediction from Han
        pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
        pred_aa = pred_aa.view(pred_aa.shape[0],-1)
        smplx_output = smplx_model(
            betas = pred_betas, 
            body_pose = pred_aa[:,3:], 
            global_orient = pred_aa[:,:3],
            right_hand_pose = input_batch['rhand_pose'],#right_hand_pose, 
            left_hand_pose= input_batch['lhand_pose'], #left_hand_pose,
            pose2rot = True)

        pred_vertices = smplx_output.vertices
        pred_vertices = pred_vertices[0].detach().cpu().numpy()
        pred_joints_3d = smplx_output.joints
        pred_joints_3d = pred_joints_3d[0].detach().cpu().numpy()   
        pred_lhand_joints_3d = smplx_output.left_hand_joints
        pred_lhand_joints_3d = pred_lhand_joints_3d[0].detach().cpu().numpy()
        pred_rhand_joints_3d = smplx_output.right_hand_joints
        pred_rhand_joints_3d = pred_rhand_joints_3d[0].detach().cpu().numpy()

        # pred_camera
        camScale = body_info['eft_pred_camera'][0]
        camTrans = body_info['eft_pred_camera'][1:]
        bbox_top_left = body_info['bbox_top_left']
        bbox_scale_ratio = body_info['bbox_scale_ratio']

        integral_output = dict()
        integral_output['pred_vertices_smpl'] = pred_vertices
        integral_output['faces'] = smplx_model.faces
        integral_output['bbox_scale_ratio'] = bbox_scale_ratio
        integral_output['bbox_top_left'] = bbox_top_left

        # Save EFT output (not the initial one)
        integral_output['pred_camera'] = body_info['eft_pred_camera']        
        pred_aa_tensor = gu.rotation_matrix_to_angle_axis(pred_rotmat.detach().cpu()[0])
        integral_output['pred_body_pose'] = pred_aa_tensor.cpu().numpy().reshape(1, 72)
        integral_output['pred_betas'] = pred_betas.detach().cpu().numpy()

        # convert mesh to original image space (X,Y are aligned to image)
        pred_vertices_bbox = convert_smpl_to_bbox(
            pred_vertices, camScale, camTrans)
        pred_vertices_img = convert_bbox_to_oriIm(
            pred_vertices_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
        integral_output['pred_vertices_img'] = pred_vertices_img

        # convert predicted 3D body joints to image space (X, Y are aligned to image)
        pred_joints_bbox = convert_smpl_to_bbox(
            pred_joints_3d, camScale, camTrans)
        pred_joints_img = convert_bbox_to_oriIm(
            pred_joints_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
        integral_output['pred_joints_img'] = pred_joints_img

        # convert predicted 3D left hand joints to image space (X, Y are aligned to image)
        pred_lhand_joints_bbox = convert_smpl_to_bbox(
            pred_lhand_joints_3d, camScale, camTrans)
        pred_lhand_joints_img = convert_bbox_to_oriIm(
            pred_lhand_joints_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
        integral_output['pred_lhand_joints_img'] = pred_lhand_joints_img

        # convert predicted 3D left hand joints to image space (X, Y are aligned to image)
        pred_rhand_joints_bbox = convert_smpl_to_bbox(
            pred_rhand_joints_3d, camScale, camTrans)
        pred_rhand_joints_img = convert_bbox_to_oriIm(
            pred_rhand_joints_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
        integral_output['pred_rhand_joints_img'] = pred_rhand_joints_img


        # Hand related prediction from Yu
        # Obtain hand rotation / poses
        l_hand_local_orient_body = body_info['eft_pred_rotmat'][:, 20]
        l_hand_global_orient_body = transfer_rotation(
            smplx_model, pred_rotmat,
            torch.from_numpy(l_hand_local_orient_body).cuda(),
            20, 'l2g', 'aa').detach().cpu().numpy().reshape(1, 3)
        l_hand_local_orient_body = gu.rotation_matrix_to_angle_axis(l_hand_local_orient_body) # rot-mat

        r_hand_local_orient_body = body_info['eft_pred_rotmat'][:, 21] # rot-mat
        r_hand_global_orient_body = transfer_rotation(
            smplx_model, pred_rotmat,
            torch.from_numpy(r_hand_local_orient_body).cuda(),
            21, 'l2g', 'aa').detach().cpu().numpy().reshape(1, 3) # aa
        r_hand_local_orient_body = gu.rotation_matrix_to_angle_axis(r_hand_local_orient_body) # rot-mat -> aa

        l_hand_local_orient_hand = None
        l_hand_global_orient_hand = None
        if hand_info is not None and hand_info['left_hand'] is not None:
            left_hand_global_orient = torch.from_numpy(hand_info['left_hand']['pred_hand_pose'][:, :3]).cuda()
            left_hand_local_orient = transfer_rotation(
                smplx_model, pred_rotmat, left_hand_global_orient, 20)
            l_hand_local_orient_hand = gu.rotation_matrix_to_angle_axis(
                left_hand_local_orient).detach().cpu().numpy().reshape(1, 3)
            l_hand_global_orient_hand = left_hand_global_orient.detach().cpu().numpy().reshape(1, 3)
            pred_left_hand_pose = hand_info['left_hand']['pred_hand_pose'][:, 3:]
        else:
            pred_left_hand_pose = np.zeros((1, 45), dtype=np.float32)

        r_hand_local_orient_hand = None
        r_hand_global_orient_hand = None
        if hand_info is not None and hand_info['right_hand'] is not None:
            right_hand_global_orient = torch.from_numpy(hand_info['right_hand']['pred_hand_pose'][:,:3]).cuda()
            right_hand_local_orient = transfer_rotation(
                smplx_model, pred_rotmat, right_hand_global_orient, 21)
            r_hand_local_orient_hand = gu.rotation_matrix_to_angle_axis(
                right_hand_local_orient).detach().cpu().numpy().reshape(1, 3)
            r_hand_global_orient_hand = right_hand_global_orient.detach().cpu().numpy().reshape(1, 3)
            pred_right_hand_pose = hand_info['right_hand']['pred_hand_pose'][:, 3:]
        else:
            pred_right_hand_pose = np.zeros((1, 45), dtype=np.float32)

         # poses and rotations related to hands
        integral_output['pred_left_hand_pose'] = pred_left_hand_pose
        integral_output['pred_right_hand_pose'] = pred_right_hand_pose

        integral_output['left_hand_local_orient_body'] = l_hand_local_orient_body
        integral_output['left_hand_global_orient_body'] = l_hand_global_orient_body
        integral_output['right_hand_local_orient_body'] = r_hand_local_orient_body
        integral_output['right_hand_global_orient_body'] = r_hand_global_orient_body

        integral_output['left_hand_local_orient_hand'] = l_hand_local_orient_hand
        integral_output['left_hand_global_orient_hand'] = l_hand_global_orient_hand
        integral_output['right_hand_local_orient_hand'] = r_hand_local_orient_hand
        integral_output['right_hand_global_orient_hand'] = r_hand_global_orient_hand

        # predicted hand betas, cameras, top-left corner and center
        left_hand_betas = None
        left_hand_camera = None
        left_hand_bbox_scale = None
        left_hand_bbox_top_left = None
        if hand_info is not None and hand_info['left_hand'] is not None:
            left_hand_betas = hand_info['left_hand']['pred_hand_betas']
            left_hand_camera = hand_info['left_hand']['pred_camera']
            left_hand_bbox_scale = hand_info['left_hand']['bbox_scale_ratio']
            left_hand_bbox_top_left = hand_info['left_hand']['bbox_top_left']

        right_hand_betas = None
        right_hand_camera = None
        right_hand_bbox_scale = None
        right_hand_bbox_top_left = None
        if hand_info is not None and hand_info['right_hand'] is not None:
            right_hand_betas = hand_info['right_hand']['pred_hand_betas']
            right_hand_camera = hand_info['right_hand']['pred_camera']
            right_hand_bbox_scale = hand_info['right_hand']['bbox_scale_ratio']
            right_hand_bbox_top_left = hand_info['right_hand']['bbox_top_left']

        integral_output['pred_left_hand_betas'] = left_hand_betas
        integral_output['left_hand_camera'] = left_hand_camera
        integral_output['left_hand_bbox_scale_ratio'] = left_hand_bbox_scale
        integral_output['left_hand_bbox_top_left'] = left_hand_bbox_top_left

        integral_output['pred_right_hand_betas'] = right_hand_betas
        integral_output['right_hand_camera'] = right_hand_camera
        integral_output['right_hand_bbox_scale_ratio'] = right_hand_bbox_scale
        integral_output['right_hand_bbox_top_left'] = right_hand_bbox_top_left


        integral_output_list.append(integral_output)
    return integral_output_list


    

