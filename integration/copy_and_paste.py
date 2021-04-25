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


def __transfer_rot(body_pose_rotmat, part_rotmat, kinematic_map, transfer_type):

    rotmat= body_pose_rotmat[0] 
    parent_id = 0
    while parent_id in kinematic_map:
        child_id = kinematic_map[parent_id]
        local_rotmat = body_pose_rotmat[child_id]
        rotmat = torch.matmul(rotmat, local_rotmat)
        parent_id = child_id

    if transfer_type == 'g2l':
        part_rot_new = torch.matmul(rotmat.T, part_rotmat)
    else:
        assert transfer_type == 'l2g'
        part_rot_new = torch.matmul(rotmat, part_rotmat)

    return part_rot_new


def transfer_rotation(
    smplx_model, body_pose, part_rot, part_idx, 
    transfer_type="g2l", result_format="rotmat"):

    assert transfer_type in ["g2l", "l2g"]
    assert result_format in ['rotmat', 'aa']

    assert type(body_pose) == type(part_rot)
    return_np = False

    if isinstance(body_pose, np.ndarray):
        body_pose = torch.from_numpy(body_pose)
        return_np = True
    
    if isinstance(part_rot, np.ndarray):
        part_rot = torch.from_numpy(part_rot)
        return_np = True

    if body_pose.dim() == 2:
        # aa
        assert body_pose.size(0) == 1 and body_pose.size(1) in [66, 72]
        body_pose_rotmat = gu.angle_axis_to_rotation_matrix(body_pose.view(22, 3)).clone()
    else:
        # rotmat
        assert body_pose.dim() == 4
        assert body_pose.size(0) == 1 and body_pose.size(1) in [22, 24]
        assert body_pose.size(2) == 3 and body_pose.size(3) == 3
        body_pose_rotmat = body_pose[0].clone()

    if part_rot.dim() == 2:
        # aa
        assert part_rot.size(0) == 1 and part_rot.size(1) == 3
        part_rotmat = gu.angle_axis_to_rotation_matrix(part_rot)[0,:3,:3].clone()
    else:
        # rotmat
        assert part_rot.dim() == 3
        assert part_rot.size(0) == 1 and part_rot.size(1) == 3 and part_rot.size(2) == 3
        part_rotmat = part_rot[0,:3,:3].clone()

    kinematic_map = get_kinematic_map(smplx_model, part_idx)
    part_rot_trans = __transfer_rot(
        body_pose_rotmat, part_rotmat, kinematic_map, transfer_type)

    if result_format == 'rotmat':    
        return_value = part_rot_trans
    else:
        part_rot_aa = gu.rotation_matrix_to_angle_axis(part_rot_trans)
        return_value = part_rot_aa
    if return_np:
        return_value = return_value.numpy()
    return return_value


def integration_copy_paste(pred_body_list, pred_hand_list, smplx_model, image_shape):
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
            right_hand_local_orient = transfer_rotation(
                smplx_model, pred_rotmat, right_hand_global_orient, 21)
            pred_rotmat[0, 21] = right_hand_local_orient
        else:
            right_hand_pose = torch.from_numpy(np.zeros( (1,45) , dtype= np.float32)).cuda()
            right_hand_global_orient = None
            right_hand_local_orient = None

        # integrate left hand pose
        if hand_info is not None and hand_info['left_hand'] is not None:
            left_hand_pose = torch.from_numpy(hand_info['left_hand']['pred_hand_pose'][:, 3:]).cuda()
            left_hand_global_orient = torch.from_numpy(hand_info['left_hand']['pred_hand_pose'][:, :3]).cuda()
            left_hand_local_orient = transfer_rotation(
                smplx_model, pred_rotmat, left_hand_global_orient, 20)
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
        pred_body_joints = smplx_output.joints
        pred_body_joints = pred_body_joints[0].detach().cpu().numpy()   
        pred_lhand_joints = smplx_output.left_hand_joints
        pred_lhand_joints = pred_lhand_joints[0].detach().cpu().numpy()
        pred_rhand_joints = smplx_output.right_hand_joints
        pred_rhand_joints = pred_rhand_joints[0].detach().cpu().numpy()

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

        # convert joints to original image space (X, Y are aligned to image)
        pred_body_joints_bbox = convert_smpl_to_bbox(
            pred_body_joints, camScale, camTrans)
        pred_body_joints_img = convert_bbox_to_oriIm(
            pred_body_joints_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
        integral_output['pred_body_joints_img'] = pred_body_joints_img

        # convert left /right joints to original image space (X, Y are aligned to image)
        pred_lhand_joints_bbox = convert_smpl_to_bbox(
            pred_lhand_joints, camScale, camTrans)
        pred_lhand_joints_img = convert_bbox_to_oriIm(
            pred_lhand_joints_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
        integral_output['pred_lhand_joints_img'] = pred_lhand_joints_img

        pred_rhand_joints_bbox = convert_smpl_to_bbox(
            pred_rhand_joints, camScale, camTrans)
        pred_rhand_joints_img = convert_bbox_to_oriIm(
            pred_rhand_joints_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
        integral_output['pred_rhand_joints_img'] = pred_rhand_joints_img

        # keep hand info
        r_hand_local_orient_body = body_info['pred_rotmat'][:, 21] # rot-mat
        r_hand_global_orient_body = transfer_rotation(
            smplx_model, pred_rotmat,
            torch.from_numpy(r_hand_local_orient_body).cuda(),
            21, 'l2g', 'aa').numpy().reshape(1, 3) # aa
        r_hand_local_orient_body = gu.rotation_matrix_to_angle_axis(r_hand_local_orient_body) # rot-mat -> aa

        l_hand_local_orient_body = body_info['pred_rotmat'][:, 20]
        l_hand_global_orient_body = transfer_rotation(
            smplx_model, pred_rotmat,
            torch.from_numpy(l_hand_local_orient_body).cuda(),
            20, 'l2g', 'aa').numpy().reshape(1, 3)
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