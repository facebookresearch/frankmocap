import sys
# assert sys.version_info > (3, 0)
sys.path.append("src/")
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import random
import pdb
import torch
import ry_utils
import parallel_io as pio
import utils.vis_utils as vis_utils
# import utils.geometry_utils as gu
import utils.rotate_utils as ru
from dataset.mtc.mtc_utils import project2D, top_down_camera_ids
from collections import defaultdict
from utils.data_utils import remap_joints_hand, remap_joints_body
import utils.normalize_joints_utils as nju
import utils.normalize_body_joints_utils as nbju
import utils.geometry_utils as gu
from utils.render_utils import project_joints
from dataset.mtc.prepare_mtc_body_hand import *
# from dataset.mtc.update_mtc_hand import load_old_annos


def load_old_annos(anno_file):
    old_annos = dict()
    for data in pio.load_pkl_single(anno_file):
        img_name = data['body_img_name']
        # img_name = data['image_name']
        img_key = '/'.join(img_name.split('/')[-3:])
        old_annos[img_key] = data
    return old_annos


def update_mtc(all_data_raw, eft_data, hand_anno_dir, origin_img_dir, old_anno_dir, res_anno_dir, res_img_dir, res_vis_dir):

    all_data = defaultdict(list)

    for phase in ['train', 'val']:
        hand_anno_file = osp.join(hand_anno_dir, f"{phase}.pkl")
        all_data_hand = load_hand_data(hand_anno_file)

        old_anno_file = osp.join(old_anno_dir, f"{phase}.pkl")
        if not osp.exists(old_anno_file):
            continue
        old_annos = load_old_annos(old_anno_file)

        all_data_body = all_data_raw[phase]
        res_data = all_data[phase]

        for body_data in all_data_body:
            img_path = body_data['img_path']
            img = cv2.imread(img_path)

            img_key = '_'.join(img_path[:-4].split('/')[-3:])

            new_img_key = '/'.join(img_path.split('/')[-3:])
            if new_img_key not in eft_data: continue

            if new_img_key in old_annos:
                left_hand_img_name = old_annos[new_img_key]['left_hand_img_name']
                right_hand_img_name = old_annos[new_img_key]['right_hand_img_name']
                if left_hand_img_name is not None and right_hand_img_name is not None:
                    res_data.append(old_annos[new_img_key])
                    continue

            for hand_type in ['left', 'right']:
                img_key_hand = img_key + f"_{hand_type}"
                if img_key_hand not in all_data_hand:
                    num_invalid += 1
                    continue

            calib_data = body_data['calib_data']

            # load hand data first
            res_hand_data = dict()
            for hand_type in ['left', 'right']:
                img_key_hand = img_key + f"_{hand_type}"
                if img_key_hand in all_data_hand:
                    hand_data = all_data_hand[img_key_hand]
                    hand_img_name = hand_data['hand_img_name']
                    hand_joints_3d = hand_data['hand_joints_3d']
                    hand_scale_ratio = hand_data['hand_scale_ratio']
                    # single hand data are all right hand
                    # but for full body data
                    if hand_type == 'left': 
                        hand_joints_3d = gu.flip_hand_joints_3d(hand_joints_3d)
                    
                    # load hand joints in original data
                    hand_joints_3d_origin = body_data[f'{hand_type}_hand_joints_3d']
                    assert np.average(np.abs(hand_joints_3d_origin)) > 1e-8
                    hand_joints_2d = project2D(hand_joints_3d_origin, calib_data, applyDistort=True)
                    hand_joints_2d = remap_joints_hand(hand_joints_2d, "mtc", "smplx")

                    # normalize again for double check
                    hand_joints_3d_norm, scale_ratio = normalize_hand_joints(hand_joints_3d_origin, calib_data)
                    assert np.average(np.abs(hand_joints_3d-hand_joints_3d_norm)<1e-8)
                    assert np.average(np.abs(scale_ratio-hand_scale_ratio))<1e-8

                    hand_joints_3d = np.concatenate((hand_joints_3d, np.ones((21, 1))), axis=1)
                    hand_joints_2d = np.concatenate((hand_joints_2d, np.ones((21, 1))), axis=1)
                else:
                    hand_img_name = None
                    hand_joints_3d = np.zeros((21, 4))
                    hand_joints_2d = np.zeros((21, 3))
                    hand_scale_ratio = 0.0

                res_hand_data[hand_type] = dict(
                    hand_img_name = hand_img_name,
                    hand_joints_3d = hand_joints_3d, # center at index finger
                    hand_joints_2d = hand_joints_2d,
                    hand_scale_ratio = hand_scale_ratio,
                )
            
            # load body joints
            body_joints_3d = body_data['body_joints_3d']
            body_inside_img = body_data['body_inside_img'].reshape(19, 1)
            body_joints_3d = np.concatenate( (body_joints_3d, body_inside_img), axis=1)
            body_joints_3d, joints_exist_01 = remap_joints_body(body_joints_3d, "mtc", "smplx")
            joints_exist = body_joints_3d[:, 3:4]

            # 2D joints
            body_joints_2d = project2D(body_joints_3d[:, :3], calib_data, applyDistort=True)
            body_joints_2d = np.concatenate((body_joints_2d, joints_exist), axis=1)

            # normalized 3D joints
            body_joints_3d_norm, body_scale_ratio = normalize_body_joints(body_joints_3d, calib_data)

            if body_scale_ratio <= 0.0:
                continue
            if np.sum(body_joints_2d[:, 2]) < 10:
                continue
            if body_joints_3d_norm[0, 3] < 1:
                continue
            if res_hand_data['left']['hand_img_name'] is not None:
                if body_joints_3d_norm[20, 3] < 1:
                    continue
            if res_hand_data['right']['hand_img_name'] is not None:
                # print(res_hand_data['right']['hand_img_name'])
                # print(body_joints_3d_norm[:, 3])
                if body_joints_3d_norm[21, 3] < 1:
                    continue

            # merge hand body joints
            all_joints_3d_norm, all_joints_2d = merge_body_hand_joints_raw(body_joints_3d_norm, body_joints_2d, res_hand_data)

            # crop image
            img_cropped, all_joints_2d_cropped = crop_img(img, all_joints_2d)
            body_joints_2d_cropped = all_joints_2d_cropped[0:25, :]
            left_hand_joints_2d_cropped = all_joints_2d_cropped[25:25+21, :]
            right_hand_joints_2d_cropped = all_joints_2d_cropped[25+21:, :]

            res_img_path = img_path.replace(origin_img_dir, res_img_dir)
            ry_utils.make_subdir(res_img_path)
            cv2.imwrite(res_img_path, img_cropped)

            camera_id = int(img_path.split('/')[-1].split('_')[1])
            if camera_id in top_down_camera_ids:
                uncommon_view = True
            else:
                uncommon_view = False

            global_rot = eft_data[new_img_key]['global_rot'].numpy()
            pose_param = eft_data[new_img_key]['smplx_pose'].numpy()[0]
            body_pose_param = np.concatenate((global_rot, pose_param), axis=0).ravel()
            body_shape_param = eft_data[new_img_key]['smplx_shape'].squeeze()

            img_name = img_path.replace(origin_img_dir, '')[1:]
            single_data = dict(
                body_img_name = img_name,
                uncommon_view = uncommon_view,
                left_hand_img_name = res_hand_data['left']['hand_img_name'],
                right_hand_img_name = res_hand_data['right']['hand_img_name'],
                body_joints_2d = body_joints_2d_cropped,
                left_hand_joints_2d = left_hand_joints_2d_cropped,
                right_hand_joints_2d = right_hand_joints_2d_cropped,
                body_joints_3d = body_joints_3d_norm,
                left_hand_joints_3d = res_hand_data['left']['hand_joints_3d'],
                right_hand_joints_3d = res_hand_data['right']['hand_joints_3d'],
                body_scale_ratio = body_scale_ratio,
                left_hand_scale_ratio = res_hand_data['left']['hand_scale_ratio'],
                right_hand_scale_ratio = res_hand_data['right']['hand_scale_ratio'],
                body_pose_param = body_pose_param,
                body_shape_param = body_shape_param,
            )

            all_joints_3d_norm, all_joints_2d = merge_body_hand_joints_processed(single_data)
            img_path = osp.join(res_img_dir, single_data['body_img_name'])
            img_cropped = cv2.imread(img_path)
            hand_img = get_hand_img(single_data, res_img_dir, img_cropped)
            img = np.concatenate((img_cropped, hand_img), axis=0)
            visualize_anno(img, res_img_path, res_img_dir, res_vis_dir, all_joints_2d, all_joints_3d_norm, separate=False)

            res_data.append(single_data)
            print(f"{img_key} complete")
    
    for phase in all_data:
        res_pkl_file = osp.join(res_anno_dir, f"{phase}.pkl")
        pio.save_pkl_single(res_pkl_file, all_data[phase])


def main_update():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/mtc'
    anno_file = osp.join(root_dir, "data_original/annotation/annotation.pkl")
    cam_file = osp.join(root_dir, 'data_original/annotation/camera_data.pkl')
    all_data_raw = load_raw_data(root_dir, anno_file, cam_file)
    pio.save_pkl_single("data/mtc_raw_data.pkl", all_data_raw)

    new_eft_fitting = osp.join(root_dir, "data_original/eft_fitting/eft_fitting_two_hands_selected.pkl")
    eft_data = pio.load_pkl_single(new_eft_fitting)

    hand_anno_dir = osp.join(root_dir, "data_processed/annotation_new")
    old_anno_dir = osp.join(root_dir, "data_processed_body/annotation")
    res_anno_dir = osp.join(root_dir, "data_processed_body/annotation_new")
    res_img_dir = osp.join(root_dir, "data_processed_body/image")
    res_vis_dir = osp.join(root_dir, "data_processed_body/image_anno")
    origin_img_dir = osp.join(root_dir, "data_original/hdImgs")
    ry_utils.build_dir(res_anno_dir)
    ry_utils.build_dir(res_img_dir)
    ry_utils.build_dir(res_vis_dir)
    update_mtc(all_data_raw, eft_data, hand_anno_dir, origin_img_dir, old_anno_dir, res_anno_dir, res_img_dir, res_vis_dir)
    
if __name__ == '__main__':
    main_update()