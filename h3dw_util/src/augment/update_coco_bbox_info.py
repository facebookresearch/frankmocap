import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import pickle
import pdb
import json
import ry_utils
import cv2
import numpy as np
import parallel_io as pio
import time
import multiprocessing as mp
from collections import defaultdict
import utils.coco_utils as cu
from pycocotools.coco import COCO
from utils import vis_utils


def get_bbox(img, kps, part_type):
    height, width = img.shape[:2]
    min_x, min_y = np.min(kps, axis=0).astype(np.int32)
    max_x, max_y = np.max(kps, axis=0).astype(np.int32)

    if part_type == 'body':
        margin_ratio = 0.1
    else:
        assert part_type == 'hand'
        margin_ratio = 0.7

    margin = int(max(max_y-min_y, max_x-min_x) * margin_ratio)
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    min_y = max(min_y, 0)
    max_y = min(max_y, height)
    min_x = max(min_x, 0)
    max_x = min(max_x, width)
    return (min_x, min_y, max_x, max_y)


def update_hand_info(
        merge_annos, hand_info, origin_img_dir, res_img_dir, img_prefix):

    for img_id in merge_annos:
        # load image first
        img_id_str = '0' *  (6-len(str(img_id))) + str(img_id)
        origin_img_path = osp.join(origin_img_dir, img_prefix+img_id_str+'.jpg')
        res_img_path = osp.join(res_img_dir, img_prefix+img_id_str+'.jpg')

        for anno_id, dp_anno, kp_anno in merge_annos[img_id]:

            img_key = f"{img_prefix}{img_id_str}_{anno_id}"
            if img_key not in hand_info: continue
            if not cu.check_dp_anno_valid(dp_anno): continue

            # print(origin_img_path)
            img = cv2.imread(origin_img_path)

            dp_x, dp_y, dp_I = cu.recover_densepose(dp_anno)
            dp_x = np.array(dp_x, dtype=np.float32)
            dp_y = np.array(dp_y, dtype=np.float32)
            dp_I = np.array(dp_I, dtype=np.int32)
            left_hand_kps, right_hand_kps = cu.get_hand_kps(dp_x, dp_y, dp_I)

            hand_info[img_key]['left_hand_bbox'] = np.zeros((4,))
            hand_info[img_key]['right_hand_bbox'] = np.zeros((4,))
            
            if right_hand_kps.shape[0] > 0:
                right_hand_bbox = get_bbox(img, right_hand_kps, 'hand')
                hand_info[img_key]['right_hand_bbox'][:] = np.array(right_hand_bbox)
                # img = vis_utils.draw_bbox(img.copy(), right_hand_bbox, color=(0,0,255), thickness=3)
            
            if left_hand_kps.shape[0] > 0:
                left_hand_bbox = get_bbox(img, left_hand_kps, 'hand')
                hand_info[img_key]['left_hand_bbox'][:] = np.array(left_hand_bbox)
                # img = vis_utils.draw_bbox(img.copy(), left_hand_bbox, color=(255,0,0), thickness=3)
            
            '''
            print(res_img_path)
            cv2.imwrite(res_img_path, img)
            '''


def main():
    coco_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/coco'
    merge_train_annos, merge_val_annos, kp_train_coco, kp_val_coco = \
        pio.load_pkl_single(osp.join(coco_dir, "annotation/merge_data.pkl"))
    print("Load annotations complete")

    hand_info_file = "/Users/rongyu/Documents/research/FAIR/workplace/data/coco/prediction/hand_info.pkl"
    hand_info = pio.load_pkl_single(hand_info_file)

    origin_img_dir = osp.join(coco_dir, 'image/train')
    res_img_dir = "visualization/coco"
    ry_utils.renew_dir(res_img_dir)

    img_prefix = 'COCO_train2014_000000'
    update_hand_info(
        merge_train_annos, hand_info, origin_img_dir, res_img_dir, img_prefix)

    pio.save_pkl_single(hand_info_file, hand_info)

if __name__ == '__main__':
    main()