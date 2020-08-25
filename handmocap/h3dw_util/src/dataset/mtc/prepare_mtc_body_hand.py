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


def load_raw_data(root_dir, anno_file, cam_file):
    all_data = pio.load_pkl_single(anno_file)
    all_cam = pio.load_pkl_single(cam_file)

    valid_seq_name = "171026_pose1" 
    valid_frame_str = [f"{id:08d}" for id in (
        2975, 3310, 13225, 14025, 16785)]

    all_data_raw = defaultdict(list)
    for training_testing, mode_data in all_data.items():
        for i, sample in enumerate(mode_data):
            if training_testing.find("train")>=0:
                phase = 'train'
            else:
                phase = 'val'

            # load data info
            seqName = sample['seqName']
            frame_str = sample['frame_str']
            if seqName != valid_seq_name: continue
            if frame_str not in valid_frame_str: continue
            if ('left_hand' not in sample) and ('right_hand' not in sample):
                continue

            # image
            img_root_name = osp.join(root_dir, 'data_original/hdImgs', seqName, frame_str)

            # 3D joints of body & hand
            body_joints_3d = np.array(sample['body']['landmarks']).reshape(-1, 3)
            body_joints_2d_info = sample['body']['2D']
            if "left_hand" in sample:
                left_hand_joints_3d = np.array(sample['left_hand']['landmarks']).reshape(-1, 3)
            else:
                left_hand_joints_3d = np.zeros((21, 3))
            if "right_hand" in sample:
                right_hand_joints_3d = np.array(sample['right_hand']['landmarks']).reshape(-1, 3)
            else:
                right_hand_joints_3d = np.zeros((21, 3))
           
            # 2D joints of body & hand
            for c in range(31):
                img_path = osp.join(img_root_name, f'00_{c:02d}_{frame_str}.jpg')
                if not osp.exists(img_path):
                    continue
                
                calib_data = all_cam[seqName][c]

                inside_img = np.array(body_joints_2d_info[c]['insideImg'])
                occlued = np.array(body_joints_2d_info[c]['occluded'])
                visible = 1 - occlued

                single_data = dict(
                    img_path = img_path,
                    body_joints_3d = body_joints_3d,
                    left_hand_joints_3d = left_hand_joints_3d,
                    right_hand_joints_3d = right_hand_joints_3d,
                    calib_data = calib_data,
                    body_inside_img = inside_img,
                    body_visible = visible,
                )
                all_data_raw[phase].append(single_data)

    print("Load raw data complete")
    return all_data_raw


def load_hand_data(anno_file):
    hand_data = pio.load_pkl_single(anno_file)
    res_data = dict()
    for single_data in hand_data:
        img_name = single_data['image_name']
        # print(single_data.keys())
        record = img_name[:-4].split('/')
        img_key = '_'.join(record[-3:])
        res_data[img_key] = dict(
            hand_img_name = img_name,
            hand_joints_2d = single_data['joints_2d'],
            hand_joints_3d = single_data['hand_joints_3d'],
            hand_scale_ratio = single_data['scale_ratio'],
            hand_img_augmented = single_data['augmented']
        )
    return res_data


def normalize_hand_joints(hand_joints_3d, calib_data):
    hand_joints_rot = (np.dot(calib_data['R'], hand_joints_3d.T) + calib_data['t']).T
    hand_joints_rot = remap_joints_hand(hand_joints_rot, "mtc", "smplx")
    joints_3d_norm, scale_ratio = nju.normalize_joints_to_smplx(hand_joints_rot.reshape(1, 21, 3))
    return joints_3d_norm[0], scale_ratio


def normalize_body_joints(body_joints_3d, calib_data):
    body_joints_rot = (np.dot(calib_data['R'], body_joints_3d[:, :3].T) + calib_data['t']).T
    body_joints_rot = np.concatenate((body_joints_rot, body_joints_3d[:, 3:4]), axis=1)
    joint_lens_smplx = pio.load_pkl_single("data/body_joints/joint_lens_smplx.pkl")
    joints_3d_norm, scale_ratio = nbju.normalize_joints(body_joints_rot.reshape(1, -1, 4), joint_lens_smplx)
    return joints_3d_norm[0], scale_ratio


def merge_body_hand_joints_raw(body_joints_3d_norm, body_joints_2d, res_hand_data):
    all_joints_3d_norm = body_joints_3d_norm.copy()
    all_joints_2d = body_joints_2d.copy()
    for hand_type in ['left', 'right']:
        all_joints_2d = np.concatenate((all_joints_2d, res_hand_data[hand_type]['hand_joints_2d']), axis=0)
        hand_joints_3d = res_hand_data[hand_type]['hand_joints_3d']
        wrist_idx = 20 if hand_type == 'left' else 21
        hand_joints_3d[:, :3] -= hand_joints_3d[0:1, :3]
        hand_joints_3d[:, :3] += body_joints_3d_norm[wrist_idx:wrist_idx+1, :3]
        all_joints_3d_norm = np.concatenate((all_joints_3d_norm, hand_joints_3d), axis=0)
    return all_joints_3d_norm, all_joints_2d


def merge_body_hand_joints_processed(single_data):
    all_joints_2d = single_data['body_joints_2d'].copy()
    all_joints_3d = single_data['body_joints_3d'].copy()
    for hand_type in ['left', 'right']:
        hand_joints_2d = single_data[f"{hand_type}_hand_joints_2d"]
        hand_joints_3d = single_data[f"{hand_type}_hand_joints_3d"]
        all_joints_2d = np.concatenate((all_joints_2d, hand_joints_2d))
        all_joints_3d = np.concatenate((all_joints_3d, hand_joints_3d))
    return all_joints_3d, all_joints_2d


def crop_img(img, joints_2d):
    exists = joints_2d[:, 2]>0
    valid_joints_2d = joints_2d[exists, :2]
    kps = valid_joints_2d

    ori_height, ori_width = img.shape[:2]
    min_x, min_y = np.min(kps, axis=0)
    max_x, max_y = np.max(kps, axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    if width > height:
        margin = (width-height) // 2
        min_y = max(min_y-margin, 0)
        max_y = min(max_y+margin, ori_height)
    else:
        margin = (height-width) // 2
        min_x = max(min_x-margin, 0)
        max_x = min(max_x+margin, ori_width)
    
    margin = int(0.25 * (max_y-min_y)) # if use loose crop, change 0.03 to 0.1
    min_y = max(min_y-margin, 0)
    max_y = min(max_y+margin, ori_height)
    min_x = max(min_x-margin, 0)
    max_x = min(max_x+margin, ori_width)

    # print(min_x, max_x, min_y, max_y)
    joints_2d_cropped = joints_2d.copy()
    joints_2d_cropped[:, :2] -= np.array([min_x, min_y]).reshape(1, 2)
    img_cropped = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
    return img_cropped, joints_2d_cropped


def get_hand_img(single_data, body_img_dir, img_cropped):
    left_hand_img_name = single_data['left_hand_img_name']
    if left_hand_img_name is None:
        left_hand_img = np.ones((100, 100, 3))*255
    else:
        left_hand_img_path = osp.join(body_img_dir, left_hand_img_name).replace("data_processed_body", "data_processed")
        left_hand_img = cv2.imread(left_hand_img_path)
        left_hand_img = cv2.resize(left_hand_img, (100, 100))

    right_hand_img_name = single_data['right_hand_img_name']
    if right_hand_img_name is None:
        right_hand_img = np.ones((100, 100, 3))*255
    else:
        right_hand_img_path = osp.join(body_img_dir, right_hand_img_name).replace("data_processed_body", "data_processed")
        right_hand_img = cv2.imread(right_hand_img_path)
        right_hand_img = cv2.resize(right_hand_img, (100, 100))

    res_img = np.ones((100, img_cropped.shape[1], 3))*255
    res_img[:100, :100, :] = left_hand_img
    res_img[:100, 100:200, :] = right_hand_img
    return res_img


# def visualize_anno(img, img_path, all_joints_2d, all_joints_3d_norm, origin_img_dir, res_vis_dir):
def visualize_anno(img, img_path, origin_img_dir, res_vis_dir, all_joints_2d, all_joints_3d_norm, separate=False):

    body_joints_2d = all_joints_2d
    body_joints_3d_norm = all_joints_3d_norm

    # draw projected normalized 3D joints
    cam = np.array([0.7, -0.7, 0.1])
    joints_exist = body_joints_3d_norm[:, 3:4]
    body_joints_2d_norm = project_joints(body_joints_3d_norm, cam)
    body_joints_2d_norm = (body_joints_2d_norm+1.0) * 0.5 * np.min(img.shape[:2])
    body_joints_2d_norm = np.concatenate((body_joints_2d_norm, joints_exist), axis=1)
    joint_img = vis_utils.draw_keypoints(img.copy(), body_joints_2d_norm, color=(255, 0 ,0), radius=3, consider_exist=True)

    # draw original 2D joints
    joint_img = vis_utils.draw_keypoints(joint_img.copy(), body_joints_2d, color=(0,0,255), radius=3, consider_exist=True)

    if separate:
        # draw each joint separately
        res_vis_path = img_path.replace(origin_img_dir, res_vis_dir)
        res_vis_subdir = res_vis_path[:-4]
        ry_utils.renew_dir(res_vis_subdir)
        for i in range(body_joints_2d.shape[0]):
            # vis_img = vis_utils.draw_keypoints(joint_img.copy(), body_joints_2d[i:i+1, :], consider_exist=True)
            vis_img = vis_utils.draw_keypoints(joint_img.copy(), body_joints_2d[i:i+1, :], color=(0,255,0), radius=3, consider_exist=True)
            vis_img = vis_utils.draw_keypoints(vis_img.copy(), body_joints_2d_norm[i:i+1, :], color=(0,255,0), radius=3, consider_exist=True)
            res_vis_path = osp.join(res_vis_subdir, f"{i:02d}.jpg")
            cv2.imwrite(res_vis_path, vis_img)
    else:
        res_vis_path = img_path.replace(origin_img_dir, res_vis_dir)
        ry_utils.make_subdir(res_vis_path)
        vis_img = joint_img
        cv2.imwrite(res_vis_path, vis_img)


def prepare_mtc(all_data_raw, hand_anno_dir, origin_img_dir, res_anno_dir, res_img_dir, res_vis_dir):
    all_data = defaultdict(list)

    for phase in ['train',]:
        hand_anno_file = osp.join(hand_anno_dir, f"{phase}.pkl")
        all_data_hand = load_hand_data(hand_anno_file)
        all_data_body = all_data_raw[phase]
        res_data = all_data[phase]

        for body_data in all_data_body:
            img_path = body_data['img_path']
            img = cv2.imread(img_path)
            img_key = '_'.join(img_path[:-4].split('/')[-3:])
            calib_data = body_data['calib_data']

            # sys.exit(0)
            
            has_hand = False
            for hand_type in ['left', 'right']:
                img_key_hand = img_key + f"_{hand_type}"
                if img_key_hand in all_data_hand:
                    has_hand =True
            if not has_hand:
                continue

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


def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/mtc'
    '''
    anno_file = osp.join(root_dir, "data_original/annotation/annotation.pkl")
    cam_file = osp.join(root_dir, 'data_original/annotation/camera_data.pkl')
    all_data_raw = load_raw_data(root_dir, anno_file, cam_file)
    pio.save_pkl_single("data/mtc_raw_data.pkl", all_data_raw)
    '''
    all_data_raw = pio.load_pkl_single("data/mtc_raw_data.pkl")

    hand_anno_dir = osp.join(root_dir, "data_processed/annotation")
    res_anno_dir = osp.join(root_dir, "data_processed_body/annotation")
    res_img_dir = osp.join(root_dir, "data_processed_body/image")
    res_vis_dir = osp.join(root_dir, "data_processed_body/image_anno")
    origin_img_dir = osp.join(root_dir, "data_original/hdImgs")
    ry_utils.renew_dir(res_anno_dir)
    ry_utils.renew_dir(res_img_dir)
    ry_utils.renew_dir(res_vis_dir)
    prepare_mtc(all_data_raw, hand_anno_dir, origin_img_dir, res_anno_dir, res_img_dir, res_vis_dir)
    
if __name__ == '__main__':
    main()