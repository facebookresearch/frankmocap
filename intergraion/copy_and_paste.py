# Copyright (c) Facebook, Inc. and its affiliates.

import sys
import numpy as np
import torch
import pdb
import mocap_utils.geometry_utils as gu
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm


def get_kinematic_map(smplx_model, dst_idx):
    cur = dst_idx
    kine_map = dict()
    while cur>=0:
        parent = int(smplx_model.parents[cur])
        if cur != dst_idx: # skip the dst_idx itself
            kine_map[parent] = cur
        cur = parent
    return kine_map


def __transfer_hand_rot(body_pose_mat, hand_rot, kinematic_map, transfer_type):
    # get global rotmat
    parent_id = 0
    rotmat= body_pose_mat[0]    # global orientation of body
    while parent_id in kinematic_map:
        child_id = kinematic_map[parent_id]
        local_rotmat = body_pose_mat[child_id]
        rotmat = torch.matmul(rotmat, local_rotmat)
        parent_id = child_id

    if hand_rot.dim() == 2:
        # hand rot in angle-axis format
        assert hand_rot.size(0) == 1 and hand_rot.size(1) == 3
    else:
        # hand rot in rotation matrix format
        assert hand_rot.dim() == 3
        assert hand_rot.size(0) == 1 and hand_rot.size(1) == 3 and hand_rot.size(2) == 3

    if hand_rot.dim() == 2:
        # hand rot in angle-axis format
        hand_rotmat = gu.angle_axis_to_rotation_matrix(hand_rot)[0,:3,:3]
    else:
        # hand rot in rotation matrix format
        hand_rotmat = hand_rot[0,:3,:3]

    if transfer_type == 'g2l':
        hand_rot_new = torch.matmul(rotmat.T, hand_rotmat)
    else:
        assert transfer_type == 'l2g'
        hand_rot_new = torch.matmul(rotmat, hand_rotmat)

    return hand_rot_new


def transfer_hand_wrist(
    smplx_model, body_pose, hand_wrist, hand_type, 
    transfer_type="g2l", result_format="rotmat"):

    assert transfer_type in ["g2l", "l2g"]
    assert result_format in ['rotmat', 'aa']

    if hand_type == 'left_hand':
        kinematic_map = get_kinematic_map(smplx_model, 20)
    else:
        assert hand_type == 'right_hand'
        kinematic_map = get_kinematic_map(smplx_model, 21)

    if transfer_type == "l2g":      
        # local to global
        hand_wrist_local = hand_wrist.clone()
        hand_wrist_mat = __transfer_hand_rot(
            body_pose, hand_wrist_local, kinematic_map, transfer_type)
    else:
        # global to local
        assert transfer_type == "g2l"
        hand_wrist_global = hand_wrist.clone()
        hand_wrist_mat = __transfer_hand_rot(
            body_pose, hand_wrist_global, kinematic_map, transfer_type)

    if result_format == 'rotmat':    
        return hand_wrist_mat
    else:
        hand_wrist_aa = gu.rotation_matrix_to_angle_axis(hand_wrist_mat)
        return hand_wrist_aa


def intergration_copy_paste(pred_body_list, pred_hand_list, smplx_model, image_shape):
    integral_output_list = list()
    for i in range(len(pred_body_list)):
        body_info = pred_body_list[i]
        hand_info = pred_hand_list[i]
        if body_info is None:
            integral_output_list.append(None)
            continue
    
        # copy and paste 
        pred_betas = torch.from_numpy(body_info['pred_betas']).cuda()
        pred_rotmat = torch.from_numpy(body_info['pred_rotmat']).cuda()

        # integrate right hand pose
        hand_output = dict()
        if hand_info is not None and hand_info['right_hand'] is not None:
            right_hand_pose = torch.from_numpy(hand_info['right_hand']['pred_hand_pose'][:, 3:]).cuda()
            right_hand_global_orient = torch.from_numpy(hand_info['right_hand']['pred_hand_pose'][:,: 3]).cuda()
            right_hand_local_orient = transfer_hand_wrist(
                smplx_model, pred_rotmat[0], right_hand_global_orient, 'right_hand')
            pred_rotmat[0, 21] = right_hand_local_orient
        else:
            right_hand_pose = torch.from_numpy(np.zeros( (1,45) , dtype= np.float32)).cuda()
            right_hand_global_orient = None
            right_hand_local_orient = None

        # integrate left hand pose
        if hand_info is not None and hand_info['left_hand'] is not None:
            left_hand_pose = torch.from_numpy(hand_info['left_hand']['pred_hand_pose'][:, 3:]).cuda()
            left_hand_global_orient = torch.from_numpy(hand_info['left_hand']['pred_hand_pose'][:, :3]).cuda()
            left_hand_local_orient = transfer_hand_wrist(
                smplx_model, pred_rotmat[0], left_hand_global_orient, 'left_hand')
            pred_rotmat[0, 20] = left_hand_local_orient
        else:
            left_hand_pose = torch.from_numpy(np.zeros((1,45), dtype= np.float32)).cuda()
            left_hand_global_orient = None
            left_hand_local_orient = None

        pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
        pred_aa = pred_aa.reshape(pred_aa.shape[0], 72)
        smplx_output = smplx_model(
            betas = pred_betas, 
            body_pose = pred_aa[:,3:], 
            global_orient = pred_aa[:,:3],
            right_hand_pose = right_hand_pose, 
            left_hand_pose= left_hand_pose,
            pose2rot = True)

        pred_vertices = smplx_output.vertices
        pred_vertices = pred_vertices[0].detach().cpu().numpy()
        pred_joints_3d = smplx_output.joints
        pred_joints_3d = pred_joints_3d[0].detach().cpu().numpy()   

        camScale = body_info['pred_camera'][0]
        camTrans = body_info['pred_camera'][1:]
        bbox_top_left = body_info['bbox_top_left']
        bbox_scale_ratio = body_info['bbox_scale_ratio']

        integral_output = dict()
        integral_output['pred_vertices_smpl'] = pred_vertices
        integral_output['faces'] = smplx_model.faces
        integral_output['bbox_scale_ratio'] = bbox_scale_ratio
        integral_output['bbox_top_left'] = bbox_top_left
        integral_output['pred_camera'] = body_info['pred_camera']

        pred_aa_tensor = gu.rotation_matrix_to_angle_axis(pred_rotmat.detach().cpu()[0])
        integral_output['pred_body_pose'] = pred_aa_tensor.cpu().numpy().reshape(1, 72)
        integral_output['pred_betas'] = pred_betas.detach().cpu().numpy()

        # convert mesh to original image space (X,Y are aligned to image)
        pred_vertices_bbox = convert_smpl_to_bbox(
            pred_vertices, camScale, camTrans)
        pred_vertices_img = convert_bbox_to_oriIm(
            pred_vertices_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
        integral_output['pred_vertices_img'] = pred_vertices_img


        # keep hand info
        r_hand_local_orient_body = body_info['pred_rotmat'][:, 21] # rot-mat
        r_hand_global_orient_body = transfer_hand_wrist(
            smplx_model, pred_rotmat[0],
            torch.from_numpy(r_hand_local_orient_body).cuda(), 
            'right_hand', 'l2g', 'aa').numpy().reshape(1, 3) # aa
        r_hand_local_orient_body = gu.rotation_matrix_to_angle_axis(r_hand_local_orient_body) # rot-mat -> aa

        l_hand_local_orient_body = body_info['pred_rotmat'][:, 20]
        l_hand_global_orient_body = transfer_hand_wrist(
            smplx_model, pred_rotmat[0],
            torch.from_numpy(l_hand_local_orient_body).cuda(), 
            'left_hand', 'l2g', 'aa').numpy().reshape(1, 3)
        l_hand_local_orient_body = gu.rotation_matrix_to_angle_axis(l_hand_local_orient_body) # rot-mat -> aa

        r_hand_local_orient_hand = None
        r_hand_global_orient_hand = None
        if right_hand_local_orient is not None:
            r_hand_local_orient_hand = gu.rotation_matrix_to_angle_axis(
                right_hand_local_orient).detach().cpu().numpy().reshape(1, 3)
            r_hand_global_orient_hand = right_hand_global_orient.detach().cpu().numpy().reshape(1, 3)

        l_hand_local_orient_hand = None
        l_hand_global_orient_hand = None
        if left_hand_local_orient is not None:
            l_hand_local_orient_hand = gu.rotation_matrix_to_angle_axis(
                left_hand_local_orient).detach().cpu().numpy().reshape(1, 3)
            l_hand_global_orient_hand = left_hand_global_orient.detach().cpu().numpy().reshape(1, 3)

        # poses and rotations related to hands
        integral_output['left_hand_local_orient_body'] = l_hand_local_orient_body
        integral_output['left_hand_global_orient_body'] = l_hand_global_orient_body
        integral_output['right_hand_local_orient_body'] = r_hand_local_orient_body
        integral_output['right_hand_global_orient_body'] = r_hand_global_orient_body

        integral_output['left_hand_local_orient_hand'] = l_hand_local_orient_hand
        integral_output['left_hand_global_orient_hand'] = l_hand_global_orient_hand
        integral_output['right_hand_local_orient_hand'] = r_hand_local_orient_hand
        integral_output['right_hand_global_orient_hand'] = r_hand_global_orient_hand

        integral_output['pred_left_hand_pose'] = left_hand_pose.detach().cpu().numpy()
        integral_output['pred_right_hand_pose'] = right_hand_pose.detach().cpu().numpy()

        # hand betas
        integral_output['pred_left_hand_betas'] = hand_info['left_hand']['pred_hand_betas']
        integral_output['pred_right_hand_betas'] = hand_info['right_hand']['pred_hand_betas']

        # predicted hand cameras, top-left corner and center
        integral_output['pred_left_hand_camera'] = hand_info['left_hand']['pred_camera']
        integral_output['pred_right_hand_camera'] = hand_info['right_hand']['pred_camera']
        integral_output['left_hand_bbox_scale_ratio'] = hand_info['left_hand']['bbox_scale_ratio']
        integral_output['right_hand_bbox_scale_ratio'] = hand_info['right_hand']['bbox_scale_ratio']
        integral_output['left_hand_bbox_top_left'] = hand_info['left_hand']['bbox_top_left']
        integral_output['right_hand_bbox_top_left'] = hand_info['right_hand']['bbox_top_left']

        integral_output_list.append(integral_output)

    return integral_output_list