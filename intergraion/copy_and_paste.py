# Copyright (c) Facebook, Inc. and its affiliates.

import sys
import numpy as np
import torch
import mocap_utils.geometry_utils as gu
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
from mocap_utils.geometry_utils import rotmat3x3_to_angleaxis


def get_local_hand_rot(body_pose, hand_rot_global, kinematic_map):
    hand_rotmat_global = gu.angle_axis_to_rotation_matrix(hand_rot_global.view(1,3))
    body_pose = body_pose.reshape(-1, 3)
    # the shape is (1,4,4), torch matmul support 3 dimension
    rotmat = gu.angle_axis_to_rotation_matrix(body_pose[0].view(1, 3))
    parent_id = 0
    while parent_id in kinematic_map:
        child_id = kinematic_map[parent_id]
        local_rotmat = gu.angle_axis_to_rotation_matrix(body_pose[child_id].view(1,3))
        rotmat = torch.matmul(rotmat, local_rotmat)
        parent_id = child_id
    hand_rotmat_local = torch.matmul(rotmat.inverse(), hand_rotmat_global)
    # print("hand_rotmat_local", hand_rotmat_local.size())
    hand_rot_local = gu.rotation_matrix_to_angle_axis(hand_rotmat_local[:, :3, :])
    return hand_rot_local


def get_global_hand_rot_mat(body_pose_mat, hand_rot_local, kinematic_map):
    hand_rotmat_local = gu.angle_axis_to_rotation_matrix(hand_rot_local.view(1,3))[0,:3,:3]
    rotmat= body_pose_mat[0]    #global orientation of body
    parent_id = 0
    while parent_id in kinematic_map:
        child_id = kinematic_map[parent_id]
        local_rotmat = body_pose_mat[child_id]
        rotmat = torch.matmul(rotmat, local_rotmat)
        parent_id = child_id
    hand_rot_local_mat = torch.matmul(rotmat.T, hand_rotmat_local)
    return hand_rot_local_mat


def get_kinematic_map(smplx_model, dst_idx):
    cur = dst_idx
    kine_map = dict()
    while cur>=0:
        parent = int(smplx_model.parents[cur])
        if cur != dst_idx: # skip the dst_idx itself
            kine_map[parent] = cur
        cur = parent
    return kine_map


def transfer_hand_wrist(smplx_model, body_pose, hand_wrist, hand_type, transfer_type="l2g"):
    if hand_type == 'left_hand':
        kinematic_map = get_kinematic_map(smplx_model, 20)
    else:
        assert hand_type == 'right_hand'
        kinematic_map = get_kinematic_map(smplx_model, 21)

    if transfer_type == "l2g":      
        # local to global
        hand_wrist_local = hand_wrist.clone()
        # hand_wrist_global = vis_utils.get_global_hand_rot(
        hand_wrist_global_mat = get_global_hand_rot_mat(
            body_pose, hand_wrist_local, kinematic_map)
        return hand_wrist_global_mat
    else:
        # global to local
        assert transfer_type == "g2l"
        hand_wrist_global = hand_wrist.clone()
        hand_wrist_local = get_local_hand_rot(
            body_pose, hand_wrist_global, kinematic_map)
        return hand_wrist_local


def intergration_copy_paste(pred_body_list, pred_hand_list, smplx_model, image_shape):
    integral_output_list = list()
    for i in range(len(pred_body_list)):
        body_info = pred_body_list[i]
        hand_info = pred_hand_list[i]
        if body_info is None or hand_info is None:
            integral_output_list.append(None)
            continue
    
        # copy and paste 
        pred_betas = torch.from_numpy(body_info['pred_betas']).cuda()
        pred_rotmat = torch.from_numpy(body_info['pred_rotmat']).cuda()

        if hand_info['right_hand'] is not None:
            right_hand_pose = torch.from_numpy(hand_info['right_hand']['pred_hand_pose'][:, 3:]).cuda()
            right_hand_global_orient = torch.from_numpy(hand_info['right_hand']['pred_hand_pose'][:,: 3]).cuda()
            right_hand_local_orient = transfer_hand_wrist(
                smplx_model, pred_rotmat[0], right_hand_global_orient, 'right_hand', 'l2g')
            pred_rotmat[0, 21] = right_hand_local_orient
        else:
            right_hand_pose = torch.from_numpy(np.zeros( (1,45) , dtype= np.float32)).cuda()

        if hand_info['left_hand'] is not None:
            left_hand_pose = torch.from_numpy(hand_info['left_hand']['pred_hand_pose'][:, 3:]).cuda()
            left_hand_global_orient = torch.from_numpy(hand_info['left_hand']['pred_hand_pose'][:, :3]).cuda()
            left_hand_local_orient = transfer_hand_wrist(smplx_model, pred_rotmat[0], left_hand_global_orient, 'left_hand', 'l2g')
            pred_rotmat[0, 20] = left_hand_local_orient
        else:
            left_hand_pose = torch.from_numpy(np.zeros((1,45), dtype= np.float32)).cuda()

        # smplx_output = smplx_model(
        #     betas = pred_betas, 
        #     body_pose = pred_rotmat[:,1:], 
        #     global_orient = pred_rotmat[:,0].unsqueeze(1),
        #     right_hand_pose = right_hand_pose, 
        #     left_hand_pose= left_hand_pose,
        #     pose2rot = False)

        #Convert rot_mat to aa since hands are always in aa
        pred_aa = rotmat3x3_to_angleaxis(pred_rotmat)
        pred_aa = pred_aa.view(pred_aa.shape[0],-1)
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

        pred_rotmat_tensor = torch.zeros((1, 24, 3, 4), dtype=torch.float32)
        pred_rotmat_tensor[:, :, :, :3] = pred_rotmat.detach().cpu()
        pred_aa_tensor = gu.rotation_matrix_to_angle_axis(pred_rotmat_tensor.squeeze())
        integral_output['pred_body_pose'] = pred_aa_tensor.cpu().numpy().reshape(1, 72)

        integral_output['pred_betas'] = pred_betas.detach().cpu().numpy()
        integral_output['pred_left_hand_pose'] = left_hand_pose.detach().cpu().numpy()
        integral_output['pred_right_hand_pose'] = right_hand_pose.detach().cpu().numpy()

        # convert mesh to original image space (X,Y are aligned to image)
        pred_vertices_bbox = convert_smpl_to_bbox(
            pred_vertices, camScale, camTrans)
        pred_vertices_img = convert_bbox_to_oriIm(
            pred_vertices_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
        integral_output['pred_vertices_img'] = pred_vertices_img

        integral_output_list.append(integral_output)

    return integral_output_list