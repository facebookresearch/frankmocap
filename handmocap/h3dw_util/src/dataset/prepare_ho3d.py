import os, sys, shutil
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os.path as osp
import argparse
import numpy as np
import torch
import smplx
import ry_utils
import parallel_io as pio
import cv2
import multiprocessing as mp
import matplotlib.pyplot as plt
import random as rd
import torchgeometry as gu

from utils.vis_utils import draw_keypoints, render_hand
from utils.ho3d_utils.vis_utils import *
from utils import data_utils
from utils import vis_utils
from check_ho3d import transform_hand_pose, forwardKinematics
from prepare_freihand import get_anno_image_single


def process_anno(anno_file):
    anno = pio.load_pkl_single(anno_file)
    # get mano pose first
    hand_rot_t, hand_pose_t = transform_hand_pose(anno['handPose'])
    hand_pose_full = np.concatenate((hand_rot_t, hand_pose_t))
    # 
    handJoints3D, handMesh, handFaces = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])
    objCorners = anno['objCorners3DRest']
    objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']
    hand_kps = project_3D_points(anno['camMat'], handJoints3D, is_OpenGL_coords=True)
    obj_kps = project_3D_points(anno['camMat'], objCornersTrans, is_OpenGL_coords=True)
    return hand_pose_full, hand_kps, obj_kps


def load_single_seq(data_dir, seq_name):
    img_dir = osp.join(data_dir, "train", seq_name, "rgb")
    img_list = sorted(os.listdir(img_dir))
    all_data = list()
    sample_ratio = 1
    for img_name in img_list:
        img_id = int(img_name[:-4])
        # only consider sampled samples
        if img_id % sample_ratio == 0:
            # load and process annotation 
            anno_file = osp.join(data_dir, "train", seq_name, "meta", f"{img_id:04d}.pkl")
            hand_pose, hand_kps, obj_kps = process_anno(anno_file)
            # pack result
            res_anno = dict(
                image_root = img_dir,
                image_name = img_name,
                mano_pose = hand_pose,
                hand_kps = hand_kps,
                obj_kps = obj_kps,
                augmented = False
            )
            all_data.append(res_anno)
    print(f"{seq_name} complete")
    return all_data


def load_all_data(data_dir):
    train_dir = osp.join(data_dir, 'train')
    seq_list = sorted([file for file in os.listdir(train_dir) if file[0]!='.'])
    # seq_list = seq_list[:3]
    process_num = min(len(seq_list), 16)
    pool = mp.Pool(process_num)
    params_list = [(data_dir, seq_name) for seq_name in seq_list]
    result_list = pool.starmap(load_single_seq, params_list)
    all_data = list()
    for sub_data in result_list:
        all_data.extend(sub_data)
    return all_data


def crop_image_single(img_path, hand_kps, obj_kps):
    img = cv2.imread(img_path)
    kps = np.concatenate((hand_kps, obj_kps), axis=0)
    ori_height, ori_width = img.shape[:2]
    min_x, min_y = np.min(kps, axis=0)
    max_x, max_y = np.max(kps, axis=0)
    # print(ori_height, ori_width)
    # print(min_x, max_x, min_y, max_y)
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

    # add additoinal margin
    margin = int(0.03 * (max_y-min_y)) # if use loose crop, change 0.03 to 0.1
    min_y = max(min_y-margin, 0)
    max_y = min(max_y+margin, ori_height)
    min_x = max(min_x-margin, 0)
    max_x = min(max_x+margin, ori_width)

    # return results
    hand_kps = hand_kps - np.array([min_x, min_y]).reshape(1, 2)
    img = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
  
    return img, hand_kps

def update_kps_weight(img, img_path, hand_kps):
    kps_weight = np.ones( hand_kps.shape[0] )
    kps_weight[hand_kps[:, 0]<=0] = 0
    kps_weight[hand_kps[:, 0]>=img.shape[1]] = 0
    kps_weight[hand_kps[:, 1]<=0] = 0
    kps_weight[hand_kps[:, 1]>=img.shape[0]] = 0
    return np.concatenate( (hand_kps, kps_weight.reshape(-1, 1)), axis=1)

def crop_image(all_data, res_img_dir):
    for data_id, single_data in enumerate(all_data):
        img_dir = single_data['image_root']
        img_name = single_data['image_name']
        img_path = osp.join(img_dir, img_name)

        #  crop image and joints_2d
        # img = cv2.imread(img_path)
        hand_kps = single_data['hand_kps']
        obj_kps = single_data['obj_kps']
        img, hand_kps = crop_image_single(img_path, hand_kps, obj_kps)
        hand_kps = data_utils.remap_joints(hand_kps, "ho3d", "smplx", "hand")
        hand_kps = update_kps_weight(img, img_path, hand_kps)

        # save cropped images
        subdir_name = img_dir.split('/')[-2]
        res_subdir = osp.join(res_img_dir, subdir_name)
        ry_utils.build_dir(res_subdir)
        res_img_path = osp.join(res_subdir, img_name)
        cv2.imwrite(res_img_path, img)

        # update annotation
        del single_data['hand_kps']
        del single_data['obj_kps']
        single_data['joints_2d'] = hand_kps
        single_data['image_root'] = res_img_dir
        single_data['image_name'] = osp.join(subdir_name, img_name)

        if data_id > 0 and data_id % 100 == 0:
            print(f"Cropped {data_id}")


def get_anno_images(all_data, res_vis_dir):
    for data_id, single_data in enumerate(all_data):
        subdir_name, img_name = single_data['image_name'].split('/')
        res_subdir = osp.join(res_vis_dir, subdir_name)
        ry_utils.build_dir(res_subdir)

        render_img, joint_img, joint_img_list = get_anno_image_single(single_data)
        render_img_path = osp.join(res_subdir, img_name.replace(".png", "_mesh.png"))
        cv2.imwrite(render_img_path, render_img)

        joint_img_path = osp.join(res_subdir, img_name.replace(".png", "_joint.png"))
        cv2.imwrite(joint_img_path, joint_img)
        joint_subdir = osp.join(res_subdir, f"joints_{img_name[:-4]}")

        ry_utils.renew_dir(joint_subdir)
        for joint_id, joint_img in enumerate(joint_img_list):
            joint_img_path = osp.join(joint_subdir, f"joint_{joint_id:02d}.png")
            cv2.imwrite(joint_img_path, joint_img)
        
        if data_id>0 and data_id%10==0:
            print(f"Processed {data_id}/{len(all_data)}")


def split_data(all_data, res_anno_dir):
    subject_name_set = set()
    for single_data in all_data:
        video_name = single_data['image_name'].split('/')[0]
        subject_name = ''.join([c for c in video_name if not c.isdigit()])
        subject_name_set.add(subject_name)
    subject_name_list = sorted(list(subject_name_set))
    num_subject = len(subject_name_list)

    ratio = 0.8
    pivot = int(num_subject * ratio)
    train_subject_name = subject_name_list[:pivot]
    val_subject_name = subject_name_list[pivot:]
    
    train_data, val_data = list(), list()
    for single_data in all_data:
        video_name = single_data['image_name'].split('/')[0]
        subject_name = ''.join([c for c in video_name if not c.isdigit()])
        if subject_name in train_subject_name:
            train_data.append(single_data)
        else:
            assert subject_name in val_subject_name
            val_data.append(single_data)

    pio.save_pkl_single(osp.join(res_anno_dir, 'train.pkl'), train_data)
    pio.save_pkl_single(osp.join(res_anno_dir, 'val.pkl'), val_data)


def process_ho3d(origin_data_dir, res_anno_dir, res_img_dir, res_vis_dir):
    # load data first
    anno_raw_file = osp.join(res_anno_dir, "all_data_raw.pkl")
    '''
    all_data = load_all_data(origin_data_dir)
    pio.save_pkl_single(anno_raw_file, all_data)
    '''

    all_data = pio.load_pkl_single(anno_raw_file)
    print("Total number of data:", len(all_data))
    crop_image(all_data, res_img_dir)

    '''
    # visualize annotation
    # all_data = all_data[:10]
    get_anno_images(all_data, res_vis_dir)
    sys.exit(0)
    '''

    # split train and test
    split_data(all_data, res_anno_dir)


def main():
    data_root = "/Users/rongyu/Documents/research/FAIR/workplace/data/HO3D/data/"
    origin_data_dir = osp.join(data_root, "data_original")

    res_anno_dir = osp.join(data_root, "data_processed/annotation")
    res_img_dir = osp.join(data_root, "data_processed/image")
    res_vis_dir = osp.join(data_root, "data_processed/image_anno")
    ry_utils.build_dir(res_anno_dir)
    ry_utils.renew_dir(res_img_dir)
    ry_utils.renew_dir(res_vis_dir)

    process_ho3d(origin_data_dir, res_anno_dir, res_img_dir, res_vis_dir)


if __name__ == '__main__':
    main()
