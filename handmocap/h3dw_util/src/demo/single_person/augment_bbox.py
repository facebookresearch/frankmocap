import os, sys, shutil
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
import multiprocessing as mp
from demo.single_person.merge_two_hands import load_bbox_info, load_all_frame


def load_all_data(root_dir):
    # load bbox
    bbox_info_dir = osp.join(root_dir, "bbox_info")
    bbox_info = load_bbox_info(bbox_info_dir)

    # load frame
    frame_dir = osp.join(root_dir, "frame")
    frame_info = load_all_frame(frame_dir)

    return frame_info, bbox_info


def get_seq_keys(all_keys):
    seq_keys = defaultdict(list)
    for key in all_keys:
        seq_name = '_'.join(key.split('_')[:-1])
        seq_keys[seq_name].append(key)
    return seq_keys


def check_bbox_exist(bbox_info, img_key_hand):
    if img_key_hand not in bbox_info:
        return False
    bbox = bbox_info[img_key_hand]
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    if x0 >= x1 or y0 >= y1:
        return False
    return True


def update_missing_bbox(bbox_info, img_key, hand_type):
    img_key_hand = img_key + '_' + hand_type
    record = img_key.split('_')
    seq_name = '_'.join(record[:-1])
    img_id = int(record[-1])
    
    img_key_hand = f'{seq_name}_{img_id:05d}_{hand_type}'
    prev_img_key_hand = f'{seq_name}_{img_id-1:05d}_{hand_type}'
    post_img_key_hand = f'{seq_name}_{img_id+1:05d}_{hand_type}'

    if check_bbox_exist(bbox_info, prev_img_key_hand):
        bbox_info[img_key_hand] = bbox_info[prev_img_key_hand]
        # new_bbox_info[img_key_hand] = bbox_info[prev_img_key_hand]
    else:
        assert check_bbox_exist(bbox_info, post_img_key_hand), img_key_hand
        bbox_info[img_key_hand] = bbox_info[post_img_key_hand]

    return bbox_info[img_key_hand]



def augment_bbox(frame_info, bbox_info):
    # organize original keys by sequence
    all_keys = sorted(list(frame_info.keys()))
    seq_keys = get_seq_keys(all_keys)

    updated_bbox_info = dict()
    for seq_name in seq_keys:
        sorted_keys = seq_keys[seq_name]

        for key_id, img_key in enumerate(sorted_keys):

            frame_path = frame_info[img_key]
            for hand_type in ['right_hand', 'left_hand']:
                img_key_hand = img_key + '_' + hand_type
                # check existence of bbox
                bbox_exist = check_bbox_exist(bbox_info, img_key_hand)
                if not bbox_exist:
                    new_bbox = update_missing_bbox(bbox_info, img_key, hand_type)
                    updated_bbox_info[img_key_hand] = new_bbox
    return updated_bbox_info
                

def main():
    # root_dir = "/checkpoint/rongyu/data/3d_hand/demo_data/youtube_example/"
    root_dir = sys.argv[1]

    frame_info, bbox_info = load_all_data(root_dir)

    # the original bbox_info also updated
    # updated_bbox_info contains only new added bbox
    updated_bbox_info = augment_bbox(frame_info, bbox_info)

    # save updated bbox_info
    res_bbox_info_file = osp.join(root_dir, "bbox_info/bbox_info.pkl")
    pio.save_pkl_single(res_bbox_info_file, bbox_info)

    for file in os.listdir(osp.join(root_dir, "bbox_info")):
        if file.endswith(".pkl") and (file.find("left")>0 or file.find("right")>0):
            os.remove(osp.join(root_dir, "bbox_info", file))

    hand_img_dir = osp.join(root_dir, "image_hand")

    # using new obtained bbox to crop hand
    for img_key_hand in updated_bbox_info:
        print(img_key_hand)
        record = img_key_hand.split('_')
        hand_type = '_'.join(record[-2:])
        img_id = record[-3]
        seq_name = '_'.join(record[:-3])

        # get frame first
        img_key = f"{seq_name}_{img_id}"
        frame_path = frame_info[img_key]
        frame = cv2.imread(frame_path)

        # new bbox
        bbox = updated_bbox_info[img_key_hand]
        x0, y0, x1, y1 = bbox
        hand_img = frame[y0:y1, x0:x1]
        if hand_type == 'left_hand':
            hand_img = np.fliplr(hand_img)

        res_img_path = osp.join(hand_img_dir, hand_type, seq_name, img_key_hand+'.png')
        ry_utils.make_subdir(res_img_path)
        cv2.imwrite(res_img_path, hand_img)


if __name__ == '__main__':
    main()