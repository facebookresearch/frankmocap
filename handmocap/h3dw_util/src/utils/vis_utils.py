"""
Renders mesh using OpenDr for visualization.
"""
import sys
sys.path.append("src")
import numpy as np
import cv2
import subprocess as sp
import torch
from utils.render_utils import render
import utils.geometry_utils as gu
import parallel_io as pio
import smplx


def img_pad_and_resize(img, final_size=224):
    height, width = img.shape[:2]
    if height > width:
        ratio = final_size / height
        new_height = final_size
        new_width = int(ratio * width)
    else:
        ratio = final_size / width
        new_width = final_size
        new_height = int(ratio * height)
    new_img = np.zeros((final_size, final_size, 3), dtype=np.uint8)
    new_img[:new_height, :new_width, :] = cv2.resize(img, (new_width, new_height))
    return new_img


def draw_keypoints(image, kps, color=(0,0,255), radius=5, consider_exist=False):
    # recover color 
    if color == 'red':
        color = (0, 0, 255)
    elif color == 'green':
        color = (0, 255, 0)
    elif color == 'blue':
        color = (255, 0, 0)
    else:
        assert isinstance(color, tuple) and len(color) == 3
    # draw keypoints
    for i in range(kps.shape[0]):
        x, y = kps[i][:2].astype(np.int32)
        if consider_exist:
            score = kps[i][2]
        else:
            score = 1.0
        # print(i, score)
        if score > 0.0:
            cv2.circle(image, (x,y), radius=radius, color=color, thickness=-1)
    return image.astype(np.uint8)


def draw_bbox(image, bbox, color=(0,0,255), thickness=3):
    x0, y0 = int(bbox[0]), int(bbox[1])
    x1, y1 = int(bbox[2]), int(bbox[3])
    res_img = cv2.rectangle(image.copy(), (x0,y0), (x1,y1), color=color, thickness=thickness)
    return res_img.astype(np.uint8)



def get_kinematic_map(smplx_model, dst_idx):
    cur = dst_idx
    kine_map = dict()
    while cur>=0:
        parent = int(smplx_model.parents[cur])
        if cur != dst_idx: # skip the dst_idx itself
            kine_map[parent] = cur
        cur = parent
    return kine_map


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


def get_global_hand_rot(body_pose, hand_rot_local, kinematic_map):
    hand_rotmat_local = gu.angle_axis_to_rotation_matrix(hand_rot_local.view(1,3))
    body_pose = body_pose.reshape(-1, 3)
    # the shape is (1,4,4), torch matmul support 3 dimension
    rotmat = gu.angle_axis_to_rotation_matrix(body_pose[0].view(1, 3))
    parent_id = 0
    while parent_id in kinematic_map:
        child_id = kinematic_map[parent_id]
        local_rotmat = gu.angle_axis_to_rotation_matrix(body_pose[child_id].view(1,3))
        rotmat = torch.matmul(rotmat, local_rotmat)
        parent_id = child_id
    hand_rotmat_local = torch.matmul(rotmat, hand_rotmat_local)
    hand_rot_local = gu.rotation_matrix_to_angle_axis(hand_rotmat_local[:, :3, :])
    return hand_rot_local


def render_hand(
    smplx_model, smplx_hand_info, 
    hand_type, hand_pose,
    cam=None, img=None, img_size=512, return_verts=False):

    if cam is None:
        cam = np.array([5.0, 0.0, 0.0])
    
    if img is None:
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    else:
        img_size = img.shape[0]

    if isinstance(hand_pose, np.ndarray):
        hand_pose = torch.from_numpy(hand_pose).float()
    else:
        assert isinstance(hand_pose, torch.Tensor)
    
    hand_rot = hand_pose[:3].view(1, 3)
    hand_pose = hand_pose[3:].view(1, 45)


    global_orient = torch.zeros((1,3)).float()
    output = smplx_model(global_orient=global_orient, 
                    right_hand_rot = hand_rot,
                    right_hand_pose_full = hand_pose,
                    left_hand_rot = hand_rot,
                    left_hand_pose_full = hand_pose,
                    return_verts=True)
    
    hand_output = smplx_model.get_hand_output(output, hand_type, smplx_hand_info, 'ave')
    verts_shift = hand_output.vertices_shift.detach().cpu().numpy().squeeze()
    hand_verts_shift = hand_output.hand_vertices_shift.detach().cpu().numpy().squeeze()
    hand_joints_shift = hand_output.hand_joints_shift.detach().cpu().numpy().squeeze()
    hand_faces_global = smplx_hand_info[f'{hand_type}_hand_faces_holistic']
    hand_faces_local = smplx_hand_info[f'{hand_type}_hand_faces_local']

    # render_img = render(verts_shift, hand_faces, cam, img_size, img)
    render_img = render(hand_verts_shift, hand_faces_local, cam, img_size, img)
    if return_verts:
        return render_img, hand_verts_shift
    else:
        return render_img


def render_body(
    smplx_model, 
    smplx_hand_info,
    body_pose, 
    body_shape=None,
    left_hand_pose=None, 
    right_hand_pose=None,
    left_hand_rot_local=None, 
    right_hand_rot_local=None,
    cam=None, img=None, img_size=512, 
    render_separate_hand=False,
    use_hand_rot = False):

    if body_shape is None:
        body_shape = torch.zeros((10,)).float()
    
    if left_hand_pose is None:
        left_hand_pose = torch.zeros((48,)).float()

    if right_hand_pose is None:
        right_hand_pose = torch.zeros((48,)).float()

    if cam is None:
        cam = np.array([5.0, 0.0, 0.0])
    
    if img is None:
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    else:
        img_size = img.shape[0]
    
    if isinstance(body_pose, np.ndarray):
        body_pose = torch.from_numpy(body_pose).float()
    else:
        assert isinstance(body_pose, torch.Tenosr)

    if isinstance(body_shape, np.ndarray):
        body_shape = torch.from_numpy(body_shape).float()
    else:
        assert isinstance(body_shape, torch.Tenosr)

    if isinstance(left_hand_pose, np.ndarray):
        left_hand_pose = torch.from_numpy(left_hand_pose).float()
    else:
        assert isinstance(left_hand_pose, torch.Tenosr)

    if isinstance(right_hand_pose, np.ndarray):
        right_hand_pose = torch.from_numpy(right_hand_pose).float()
    else:
        assert isinstance(right_hand_pose, torch.Tenosr)

    # body pose and shape
    assert body_pose.size(0) == 72
    global_orient = body_pose[:3].view(1, 3)
    smplx_body_pose = body_pose[3:22*3].view(1, 63)
    smplx_body_shape = body_shape.view(1, 10)

    # hand rotation (wrist rotation)
    if use_hand_rot:
        left_hand_rot = torch.from_numpy(left_hand_rot_local).view(1,3).float()
        right_hand_rot = torch.from_numpy(right_hand_rot_local).view(1,3).float()
    else:
        left_hand_rot = smplx_body_pose[:, 19*3:20*3]
        right_hand_rot = smplx_body_pose[:, 20*3:21*3]
    
    left_hand_pose = torch.cat((left_hand_rot[0], left_hand_pose[3:]), dim=0)
    right_hand_pose = torch.cat((right_hand_rot[0], right_hand_pose[3:]), dim=0)

    # hand pose
    left_hand_pose_full = left_hand_pose[3:].view(1, 45)
    right_hand_pose_full = right_hand_pose[3:].view(1, 45)

    output = smplx_model(
        global_orient = global_orient,
        body_pose = smplx_body_pose,
        betas = smplx_body_shape,
        left_hand_rot = left_hand_rot,
        left_hand_pose_full = left_hand_pose_full,
        right_hand_rot = right_hand_rot,
        right_hand_pose_full = right_hand_pose_full)

    verts = output.vertices.cpu().numpy()[0]
    faces = smplx_model.faces

    render_img = render(verts, faces, cam, img_size, img)
    res_img = render_img

    if render_separate_hand:
        cam = np.array([6.240435, 0.0, 0.0])

        # left_hand
        left_hand_render_img = render_hand(
            smplx_model, smplx_hand_info,
            "left", left_hand_pose,
            cam, None, img.shape[0])
        res_img = np.concatenate((res_img, left_hand_render_img), axis=1)

        # right_hand
        right_hand_render_img = render_hand(
            smplx_model, smplx_hand_info,
            "right", right_hand_pose,
            cam, None, img.shape[0])
        res_img = np.concatenate((res_img, right_hand_render_img), axis=1)

    return res_img