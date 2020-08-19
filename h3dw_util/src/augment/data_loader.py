import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
from collections import defaultdict
import json
import smplx
import cv2
import ry_utils
import parallel_io as pio
import pdb
from augment.sample import Sample
from utils import vis_utils
import utils.geometry_utils as gu
import time
import torch


def rotmat_to_angle(pred_rotmat):
    pred_rotmat_t = torch.from_numpy(pred_rotmat.copy())
    num_pose = pred_rotmat_t.size(0)
    pred_rotmat_pad = torch.zeros((num_pose, 3, 4))
    pred_rotmat_pad[:, :, :3] = pred_rotmat_t
    pred_rotmat_pad[:, :, 3] = 1
    pred_pose_angle_t = gu.rotation_matrix_to_angle_axis(pred_rotmat_pad)
    pred_pose_angle = pred_pose_angle_t.numpy().reshape(72)
    return pred_pose_angle


def load_pred_body(in_dir):
    frame_dir = osp.join(in_dir, "frame")
    pred_res_dir = osp.join(in_dir, "eft_fitting")

    body_info = dict()
    num_data = 0
    for subdir, dirs, files in os.walk(pred_res_dir, followlinks=True):
        for file in files:
            if file.endswith(".pkl"):
                # get predicted body data first
                pkl_file = osp.join(subdir, file)
                body_data = pio.load_pkl_single(pkl_file)

                # get predicted pose (in angle-axis format)
                pred_pose_rotmat = body_data['pred_pose_rotmat'][0]
                pred_pose_angle = rotmat_to_angle(pred_pose_rotmat)

                # get pred cameras (in h3dw format)
                pred_cam = body_data['pred_camera'][0]
                pred_cam[1:] /= pred_cam[0]
                pred_cam[0] = 1

                # get img name and frame path
                anno_id = body_data['annotId'][0]
                img_name = '_'.join(pkl_file.split('/')[-1][:-4].split('_')[:-1])
                img_key = f"{img_name}_{anno_id}"
                phase = 'train' if img_name.find('train')>=0 else 'val'
                frame_path = osp.join(frame_dir, phase, img_name+'.jpg')
                assert osp.exists(frame_path)
                # img_key = img_name
                body_info[img_key] = dict(
                    frame_path = frame_path,
                    pred_body_cam = pred_cam,
                    pred_body_pose = pred_pose_angle,
                    pred_body_shape = body_data['pred_shape']
                )
                num_data += 1
                if num_data % 100 == 0:
                    print(f"Load body data: {num_data:05d}")
                # if num_data > 50: break
    return body_info


def _default_pred_hand_pose():
    res_dict = dict(
        pred_hand_cam=np.zeros(3,), 
        pred_hand_pose=np.zeros(48,), 
        pred_hand_verts=np.zeros((778,)))
    return res_dict

def _default_pred_hand():
    return dict(
        left_hand = (_default_pred_hand_pose(), '', ''),
        right_hand = (_default_pred_hand_pose(), '', ''),
    )


def load_pred_hand(in_dir):
    pred_res = pio.load_pkl_single(osp.join(in_dir, "pred_results_coco.pkl"))
    crop_hand_dir = osp.join(in_dir, "image_hand")
    render_hand_dir = osp.join(in_dir, "image_render")

    hand_info = defaultdict(
        _default_pred_hand
    )
    for value in pred_res:
        img_name = value['img_name'] # train2014/COCO_train2014_000000000036_left.jpg

        # get crop hand and render hand image
        crop_hand_img = osp.join(crop_hand_dir, img_name)
        render_hand_img = osp.join(render_hand_dir, img_name)
        assert osp.exists(crop_hand_img)
        assert osp.exists(render_hand_img)

        if img_name.find("left")>=0:
            hand_type = "left"
        else:
            assert img_name.find("right")>=0
            hand_type = "right"

        record = img_name.split('/')[-1]
        img_key = '_'.join(record.split('_')[:-1])

        # get predicted camera
        pred_cam = value['cam']
        pred_pose = value['pred_pose_params']
        pred_verts = value['pred_verts']
        if hand_type == 'left':
            pred_cam[1] *= -1 # flip x
            pred_pose = gu.flip_hand_pose(pred_pose)
            pred_verts = gu.flip_hand_joints_3d(pred_verts) # flip joints 3d and vertices are the same

        # save to res
        pred_hand_info = dict(
            pred_hand_cam = value['cam'],
            pred_hand_pose = pred_pose,
            pred_hand_verts = pred_verts,
        )
        hand_info[img_key][f'{hand_type}_hand'] = (pred_hand_info, crop_hand_img, render_hand_img)

    return hand_info


# add local hand wrist
def update_hand_info(pred_body_info, hand_info):
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    smplx_model_file = osp.join(root_dir, "data/models/smplx/SMPLX_NEUTRAL.pkl")
    smplx_model = smplx.create(smplx_model_file, model_type="smplx")

    pred_body_pose = pred_body_info['pred_body_pose']
    for hand_type in ['left_hand', 'right_hand']:
        img_name = hand_info[hand_type][1]
        pred_hand_pose = hand_info[hand_type][0]['pred_hand_pose']
        wrist_id = 20 if hand_type == 'left_hand' else 21

        wrist_rot_local = pred_hand_pose[:3].copy()
        wrist_rot_body = pred_hand_pose[:3].copy()

        if len(img_name) == 0:
            wrist_rot_local = pred_body_pose[wrist_id*3 : (wrist_id+1)*3]
            # pred_hand_pose_local[:3] = pred_body_pose[wrist_id*3 : (wrist_id+1)*3]
        else:
            hand_wrist_global = torch.from_numpy(pred_hand_pose[:3]).view(1, 3).float()
            body_pose = torch.from_numpy(pred_body_pose).float()
            kinematic_map = vis_utils.get_kinematic_map(smplx_model, wrist_id)
            wrist_rot_local = vis_utils.get_local_hand_rot(
                body_pose, hand_wrist_global, kinematic_map).numpy()[0]

            wrist_from_body = pred_body_pose[wrist_id*3 : (wrist_id+1)*3]
            wrist_from_body = torch.from_numpy(wrist_from_body).float().view(1, 3)
            wrist_rot_body = vis_utils.get_global_hand_rot(
                body_pose, wrist_from_body, kinematic_map).numpy()[0]

        # pred_hand_pose: (48,), hand pose (global rotation)
        # wrist_rot_local (3,), local rotation of wrist predicted from hand model (in term of body pose)
        # wrist_rot_body_global, (3,), global rot of wrist predicted from body
        hand_info[hand_type][0]['wrist_rot_local'] = wrist_rot_local
        hand_info[hand_type][0]['wrist_rot_body_global'] = wrist_rot_body
            

def merge_data(body_info, hand_info, res_hand_data_file=''):
    seq_info = defaultdict(list)
    for img_name in body_info:
        seq_name = 'train' if img_name.find("train")>=0 else "val"
        seq_info[seq_name].append(img_name)
    for seq_name in seq_info:
        seq_info[seq_name] = sorted(seq_info[seq_name]) # sort the data, important
    
    '''
    if len(res_hand_data_file) > 0:
        assert osp.exists(res_hand_data_file)
        res_hand_data = pio.load_pkl_single(res_hand_data_file)
    else:
        res_hand_data = None
    '''

    all_samples = defaultdict(list)
    for seq_name in seq_info:
        all_img_names = seq_info[seq_name]
        for sample_id, img_name in enumerate(all_img_names):

            # only considering samples with hand information
            if img_name not in hand_info:
                continue
                
            frame_path = body_info[img_name]['frame_path']
            pred_body_info = body_info[img_name]

            left_hand_info = hand_info[img_name]['left_hand']
            right_hand_info = hand_info[img_name]['right_hand']

            update_hand_info(pred_body_info, hand_info[img_name])

            hand_img_path = dict(
                left_hand = left_hand_info[1],
                right_hand = right_hand_info[1]
            )
            render_img_path = dict(
                left_hand = left_hand_info[2],
                right_hand = right_hand_info[2]
            )
            pred_hand_info = dict(
                left_hand = left_hand_info[0],
                right_hand = right_hand_info[0]
            )

            sample = Sample(
                seq_name = seq_name,
                sample_id = sample_id,
                img_name = img_name,
                frame_path = frame_path,
                hand_img_path = hand_img_path,
                render_img_path = render_img_path,
                pred_body_info = pred_body_info,
                pred_hand_info = pred_hand_info,
            )

            '''
            if res_hand_data is not None:
                img_key = sample.img_name
                if img_key in res_hand_data:
                    res_data = res_hand_data[sample.img_name]
                    for hand_type in ['left_hand', 'right_hand']:
                        if res_data[f'{hand_type}_exist']:
                            res_wrist = res_data[f'{hand_type}_wrist']
                            cur_wrist = sample.pred_hand_info[hand_type]['wrist_rot_local']
                            res_pose = res_data[f'{hand_type}_pose']
                            cur_pose = sample.pred_hand_info[hand_type]['pred_hand_pose'][3:]
                            print(np.average(np.abs(res_wrist - cur_wrist)))
                            print(np.average(np.abs(res_pose - cur_pose)))
            if sample_id > 10: break
            '''

            all_samples[seq_name].append(sample)
            if sample_id % 100 == 0:
                print(f"Merge data, {seq_name}, {sample_id:05d}")
    
    return all_samples


def load_all_samples(exp_res_dir, res_hand_data_file=''):
    # load predicted body pose
    pred_body_dir = osp.join(exp_res_dir, 'body')
    pred_body_info_file = osp.join(pred_body_dir, "body_info.pkl")
    '''
    body_info = load_pred_body(pred_body_dir)
    pio.save_pkl_single(pred_body_info_file, body_info)
    '''
    body_info = pio.load_pkl_single(pred_body_info_file)

    # load predicted hand pose
    pred_hand_dir = osp.join(exp_res_dir, 'hand')
    hand_info = load_pred_hand(pred_hand_dir)

    all_samples = merge_data(body_info, hand_info, res_hand_data_file)
    return all_samples