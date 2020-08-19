import os, sys, shutil
import os.path as osp
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
from pycocotools.coco import COCO


def recover_densepose(dp_anno):
    dp_x = dp_anno['dp_x']
    dp_y = dp_anno['dp_y']
    dp_bbox = dp_anno['bbox']
    x_scale = dp_bbox[2]/255
    y_scale = dp_bbox[3]/255
    dp_x = np.array(dp_x)*x_scale + dp_bbox[0]
    dp_y = np.array(dp_y)*y_scale + dp_bbox[1]
    dp_I = dp_anno['dp_I']
    return dp_x, dp_y, dp_I


def get_hand_kps(dp_x, dp_y, dp_I):
    right_hand_x = dp_x[dp_I == 3]
    right_hand_y = dp_y[dp_I == 3]
    right_hand_kps = np.array(list(zip(right_hand_x, right_hand_y)))
    left_hand_x = dp_x[dp_I == 4]
    left_hand_y = dp_y[dp_I == 4]
    left_hand_kps = np.array(list(zip(left_hand_x, left_hand_y)))
    return left_hand_kps, right_hand_kps


def get_body_kps(dp_x, dp_y, dp_I):
    body_x = dp_x[dp_I>0]
    body_y = dp_y[dp_I>0]
    body_kps = np.array(list(zip(body_x, body_y)))
    return body_kps


def check_dp_anno_valid(dp_anno):
    anno_names = ['dp_x', 'dp_y', 'dp_U', 'dp_V', 'dp_I']
    if all( [(name in dp_anno) for name in anno_names] ):
        if all( [len(dp_anno[name])==len(dp_anno['dp_x'])\
                for name in anno_names] ):
            if len(dp_anno['dp_x'])>0:
                return True
    return False


def load_anno_from_json(anno_file):
    coco = COCO(anno_file)
    catIds = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=catIds)
    all_annos = dict()
    for img_id in img_ids:
        anno_id = coco.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)
        single_annos = coco.loadAnns(anno_id)
        one_image_anno = dict()
        for anno in single_annos:
            one_image_anno[anno['id']] = anno
        all_annos[img_id] = one_image_anno
    return coco, all_annos


def get_dp_annos(coco_dir):
    train_anno_file = osp.join(coco_dir, 'annotations_raw/densepose_coco_2014_train.json')
    _, train_annos = load_anno_from_json(train_anno_file)
    val_anno_file = osp.join(coco_dir, 'annotations_raw/densepose_coco_2014_minival.json')
    _, val_annos = load_anno_from_json(val_anno_file)
    valminus_anno_file = osp.join(coco_dir, 'annotations_raw/densepose_coco_2014_valminusminival.json')
    _, valminus_annos = load_anno_from_json(valminus_anno_file)
    for img_id in valminus_annos.keys():
        val_annos[img_id] = valminus_annos[img_id]
    return train_annos, val_annos


def get_kp_annos(coco_dir):
    train_anno_file = osp.join(coco_dir,'annotations/person_keypoints_train2014.json')
    train_coco, train_annos = load_anno_from_json(train_anno_file)
    val_anno_file = osp.join(coco_dir,'annotations/person_keypoints_val2014.json')
    val_coco, val_annos = load_anno_from_json(val_anno_file)
    return train_annos, val_annos, train_coco, val_coco


def merge_dp_kp_annos(dp_annos, kp_annos):
    merge_annos = defaultdict(list)
    dp_anno_keys= list(dp_annos.keys())
    for img_id in dp_anno_keys:
        if img_id in kp_annos:
            dp_anno = dp_annos[img_id]
            kp_anno = kp_annos[img_id]
            shared_anno_id = set(set(dp_anno.keys()) & set(kp_anno.keys()))
            for anno_id in shared_anno_id:
                merge_annos[img_id].append((anno_id, dp_anno[anno_id], kp_anno[anno_id]))
    return merge_annos