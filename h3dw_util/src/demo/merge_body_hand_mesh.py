import os, sys, shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('src/')
import os.path as osp
import argparse
import numpy as np
import torch
import smplx
from utils.render_utils import render
import cv2
import multiprocessing as mp
import utils.geometry_utils as gu
import parallel_io as pio
import ry_utils
import pdb
import time
from collections import defaultdict
from utils.vis_utils import render_hand, render_body

def load_pred_body(in_dir):
    frame_dir = osp.join(in_dir, "frame")
    pred_res = pio.load_pkl_single(osp.join(in_dir, "smpl_pred.pkl"))

    body_info = dict()
    for img_name, value in pred_res.items():
        frame_path = osp.join(frame_dir, img_name)
        body_info[img_name] = dict(
            img_path = frame_path,
            pred_cam = value['pred_cam'],
            pred_body_pose = value['pred_pose_param'],
            pred_body_shape = value['pred_shape_param'],
        )
    return body_info


def load_pred_hand(in_dir):
    pred_res = pio.load_pkl_single(osp.join(in_dir, "pred_results_demo_body_capture.pkl"))
    hand_info = defaultdict(
        lambda : dict(
            left_hand_pose = np.zeros((48,)),
            right_hand_pose = np.zeros((48,))
        )
    )
    for value in pred_res:
        img_name = value['img_name']
        # get hand_type
        record = img_name.split('/')
        if record[0] == 'left_hand':
            hand_type = 'left'
        else:
            assert record[0] == 'right_hand'
            hand_type = 'right'
        img_id = record[-1].replace(f'_{hand_type}_hand', '')[:-4]
        img_name = osp.join(record[1], img_id + '.png')

        pred_pose = value['pred_pose_params']
        if hand_type == 'left':
            pred_pose = pred_pose.reshape(16, 3)
            pred_pose[:, 1] *= -1
            pred_pose[:, 2] *= -1
            '''
            pred_pose[:, 0] *= -1
            '''
            pred_pose = pred_pose.reshape(48)
        hand_info[img_name][f'{hand_type}_hand_pose'] = pred_pose
    return hand_info


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
    # print("body_pose", body_pose.size())
    # print("hand_rot_global", hand_rot_global.size())
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


def render_single_body(model, body_pose, body_shape, left_hand_pose, right_hand_pose, wrist_rot_type):
    body_pose_torch = torch.from_numpy(body_pose).float()
    body_shape_torch = torch.from_numpy(body_shape).float().view(1, 10)
    left_hand_pose_torch = torch.from_numpy(left_hand_pose).float()
    right_hand_pose_torch = torch.from_numpy(right_hand_pose).float()

    global_orient = body_pose_torch[:3].view(1, 3)
    smplx_body_pose = body_pose_torch[3:22*3].view(1, 63)

    if wrist_rot_type == 'use_hand_rot':
        kinematic_map = get_kinematic_map(model, 20)
        left_hand_rot_global = left_hand_pose_torch[:3]
        left_hand_rot = get_local_hand_rot(body_pose_torch, left_hand_rot_global, kinematic_map)

        kinematic_map = get_kinematic_map(model, 21)
        right_hand_rot_global = right_hand_pose_torch[:3]
        right_hand_rot = get_local_hand_rot(body_pose_torch, right_hand_rot_global, kinematic_map)
    else:
        left_hand_rot = smplx_body_pose[:, 19*3:20*3]
        right_hand_rot = smplx_body_pose[:, 20*3:21*3]

    left_hand_pose = left_hand_pose_torch[3:].view(1, 45)
    right_hand_pose = right_hand_pose_torch[3:].view(1, 45)

    output = model(
        global_orient = global_orient,
        betas = body_shape_torch,
        body_pose = smplx_body_pose,
        left_hand_rot = left_hand_rot,
        left_hand_pose_full = left_hand_pose,
        right_hand_rot = right_hand_rot,
        right_hand_pose_full = right_hand_pose)

    pred_verts = output.vertices.cpu().numpy()[0]
    faces = model.faces
    return pred_verts, faces


def render_single_hand(model, hand_info, hand_type, hand_pose):
    global_orient = torch.zeros((1,3))
    hand_pose_torch = torch.from_numpy(hand_pose).float()
    hand_rot = hand_pose_torch[:3].view(1, 3)
    hand_pose = hand_pose_torch[3:].view(1, 45)
    output = model(global_orient=global_orient, 
                    left_hand_rot = hand_rot,
                    left_hand_pose_full = hand_pose,
                    right_hand_rot = hand_rot,
                    right_hand_pose_full = hand_pose,
                    return_verts=True)
        
    hand_output = model.get_hand_output(output, hand_type, hand_info, 'long')
    verts_shift = hand_output.vertices_shift.detach().cpu().numpy().squeeze()
    hand_joints_shift = hand_output.hand_joints_shift.detach().cpu().numpy().squeeze()
    hand_faces = hand_info[f'{hand_type}_hand_faces_holistic']
    return verts_shift, hand_faces


def merge_hand_body(smplx_model, body_info, hand_info, wrist_rot_type, res_dir):
    # build res_subdir first
    all_img_names = sorted(list(hand_info.keys()))
    for img_name in all_img_names:
        res_img_path = osp.join(res_dir, img_name)
        res_subdir = '/'.join(res_img_path.split('/')[:-1])
        if not osp.exists(res_subdir):
            ry_utils.build_dir(res_subdir)

    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/"
    hand_info_file = osp.join(data_root, "models/smplx/SMPLX_HAND_INFO.pkl")
    hand_info_smplx  = pio.load_pkl_single(hand_info_file)

    num_img = 0
    num_total = len(all_img_names)
    start_time = time.time()
    for img_name in all_img_names:
        assert img_name in body_info

        body_pose = body_info[img_name]['pred_body_pose']
        body_shape = body_info[img_name]['pred_body_shape']
        left_hand_pose = hand_info[img_name]['left_hand_pose']
        right_hand_pose = hand_info[img_name]['right_hand_pose']
        verts, faces = render_single_body(
            smplx_model, body_pose, body_shape, left_hand_pose, right_hand_pose, wrist_rot_type)

        frame_path = body_info[img_name]['img_path']
        frame = cv2.imread(body_info[img_name]['img_path'])
        cam = body_info[img_name]['pred_cam']
        inputSize = frame.shape[0]
        # print(cam.shape)
        render_img_body = render(verts, faces, cam, inputSize, frame)

        cam = np.array([6.240435, 0.0, 0.0])
        verts_hand, faces_hand = render_single_hand(smplx_model, hand_info_smplx, 'left', left_hand_pose) 
        render_img_left_hand = render(verts_hand, faces_hand, cam, inputSize)
        verts_hand, faces_hand = render_single_hand(smplx_model, hand_info_smplx, 'right', right_hand_pose) 
        render_img_right_hand = render(verts_hand, faces_hand, cam, inputSize)

        res_img = np.concatenate(
            (frame, render_img_body, render_img_left_hand, render_img_right_hand), axis=1)
        height, width = res_img.shape[:2]
        res_height, res_width = height//2, width//2
        res_img = cv2.resize(res_img, (width, height))
        
        res_img_path = osp.join(res_dir, img_name)
        cv2.imwrite(res_img_path, res_img)

        num_img += 1
        if num_img%10 == 0:
            speed = num_img / (time.time() - start_time)
            remain_time = (num_total-num_img) / speed / 60
            print(f"Processed: {num_img}/{num_total}, remain requires {remain_time} mins")


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    smplx_model_file = osp.join(root_dir, "data/models/smplx/SMPLX_NEUTRAL.pkl")
    smplx_model = smplx.create(smplx_model_file, model_type="smplx")

    pred_dir = osp.join(root_dir, "experiment/experiment_results/3d_hand/h3dw/demo_data/body_capture/prediction")
    pred_body_dir = osp.join(pred_dir, "body")
    body_info = load_pred_body(pred_body_dir)

    pred_hand_dir = osp.join(pred_dir, "hand")
    hand_info = load_pred_hand(pred_hand_dir)

    wrist_rot_type = "use_hand_rot"
    res_dir = f"visualization/body_hand_merge/body_capture_{wrist_rot_type}"
    ry_utils.build_dir(res_dir)
    merge_hand_body(smplx_model, body_info, hand_info, wrist_rot_type, res_dir)

if __name__ == '__main__':
    main()