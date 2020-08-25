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


def crop_img_by_kps(img, kps, part_type):
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
    return img[min_y:max_y, min_x:max_x, :]


def crop_hand_images_single(
        coco, merge_annos, selected_keys, process_id,\
        in_dir, out_dir, img_prefix):

    origin_img_dir = in_dir
    res_img_dir = out_dir
    num_img = 0

    start_time = time.time()
    for img_id in selected_keys:

        # load image first
        img_id_str = '0' *  (6-len(str(img_id))) + str(img_id)
        origin_img_path = osp.join(origin_img_dir, img_prefix+img_id_str+'.jpg')
        if not osp.exists(origin_img_path): continue
        img = cv2.imread(origin_img_path)

        # analysis annotation
        for anno_id, dp_anno, kp_anno in merge_annos[img_id]:
            if not cu.check_dp_anno_valid(dp_anno): continue
            dp_x, dp_y, dp_I = cu.recover_densepose(dp_anno)
            dp_x = np.array(dp_x, dtype=np.float32)
            dp_y = np.array(dp_y, dtype=np.float32)
            dp_I = np.array(dp_I, dtype=np.int32)
            left_hand_kps, right_hand_kps = cu.get_hand_kps(dp_x, dp_y, dp_I)
            
            if right_hand_kps.shape[0] > 0:
                hand_img = crop_img_by_kps(img, right_hand_kps, 'hand')
                if hand_img.shape[0]>0 and hand_img.shape[1]>0:
                    # res_img_path = osp.join(res_img_dir, img_prefix+img_id_str+'_right.jpg')
                    res_img_path = osp.join(res_img_dir, f'{img_prefix}{img_id_str}_{anno_id}_right.jpg')
                    cv2.imwrite(res_img_path, hand_img)
            
            if left_hand_kps.shape[0] > 0:
                hand_img = crop_img_by_kps(img, left_hand_kps, 'hand')
                if hand_img.shape[0]>0 and hand_img.shape[1]>0:
                    hand_img = np.fliplr(hand_img)
                    # res_img_path = osp.join(res_img_dir, img_prefix+img_id_str+'_left.jpg')
                    res_img_path = osp.join(res_img_dir, f'{img_prefix}{img_id_str}_{anno_id}_left.jpg')
                    cv2.imwrite(res_img_path, hand_img)
            
        num_img += 1
        if num_img>0 and num_img%100 == 0:
            print(f"{os.getpid()}: {num_img}/{len(selected_keys)}")


def crop_body_images_single(
        coco, merge_annos, selected_keys, process_id,\
        in_dir, out_dir, img_prefix):
    origin_img_dir = in_dir
    res_img_dir = out_dir
    num_img = 0

    start_time = time.time()
    for img_id in selected_keys:

        # load image first
        img_id_str = '0' *  (6-len(str(img_id))) + str(img_id)
        origin_img_path = osp.join(origin_img_dir, img_prefix+img_id_str+'.jpg')
        if not osp.exists(origin_img_path): continue
        img = cv2.imread(origin_img_path)

        # analysis annotation
        for anno_id, dp_anno, kp_anno in merge_annos[img_id]:
            if not cu.check_dp_anno_valid(dp_anno): continue
            dp_x, dp_y, dp_I = cu.recover_densepose(dp_anno)
            dp_x = np.array(dp_x, dtype=np.float32)
            dp_y = np.array(dp_y, dtype=np.float32)
            dp_I = np.array(dp_I, dtype=np.int32)
            body_kps = cu.get_body_kps(dp_x, dp_y, dp_I)
            
            if body_kps.shape[0] > 0:
                body_img = crop_img_by_kps(img, body_kps, 'body')
                if body_img.shape[0]>0 and body_img.shape[1]>0:
                    res_img_path = osp.join(res_img_dir, img_prefix+img_id_str+'_body.jpg')
                    cv2.imwrite(res_img_path, body_img)
            
        num_img += 1
        if num_img>0 and num_img%100 == 0:
            print(f"{os.getpid()}: {num_img}/{len(selected_keys)}")


def crop_images(kp_coco, merge_annos, part_type, in_img_dir, out_img_dir, img_prefix):
    # assert False, "This code needs to be updated to include support for annotation"
    process_num = 4
    all_keys = list(merge_annos.keys())
    pivot = len(all_keys)//process_num
    process_list = list()
    target_func = crop_hand_images_single if part_type == 'hand' else crop_body_images_single
    for i in range(process_num):
        start = i * pivot
        end = (i+1)*pivot if i<process_num-1 else len(all_keys)
        selected_keys = all_keys[start:end]
        p = mp.Process(
                target=target_func,
                args=(kp_coco, merge_annos, selected_keys, i,
                    in_img_dir, out_img_dir, img_prefix))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()


def main():
    coco_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/coco'

    '''
    # load densepose annotations first
    dp_train_annos, dp_val_annos = cu.get_dp_annos(coco_dir)
    # load original coco annotations
    kp_train_annos, kp_val_annos, kp_train_coco, kp_val_coco = cu.get_kp_annos(coco_dir)
    # merge annotation from densepose and original coco dataset
    merge_train_annos = cu.merge_dp_kp_annos(dp_train_annos, kp_train_annos)
    merge_val_annos = cu.merge_dp_kp_annos(dp_val_annos, kp_val_annos)
    all_data = (merge_train_annos, merge_val_annos, kp_train_coco, kp_val_coco)
    pio.save_pkl_single(osp.join(coco_dir, "merge_data.pkl"), all_data)
    '''

    merge_train_annos, merge_val_annos, kp_train_coco, kp_val_coco = \
        pio.load_pkl_single(osp.join(coco_dir, "annotations/merge_data.pkl"))
    print("Load annotations complete")

    # for part_type in ['body', 'hand']:
    for part_type in ['hand']:
        res_img_dir = osp.join(coco_dir, "image_crop_dp", f'{part_type}')
        # ry_utils.renew_dir(res_img_dir)

        infos = [ ('train', (kp_train_coco, merge_train_annos)),
                ('val',   (kp_val_coco, merge_val_annos)) ]
        for phase, (kp_coco, merge_annos) in infos:
            origin_img_dir = osp.join(coco_dir, 'image/{}2014'.format(phase))
            res_img_subdir = osp.join(res_img_dir, '{}2014'.format(phase))
            ry_utils.build_dir(res_img_subdir)
            img_prefix = 'COCO_{}2014_000000'.format(phase)
            crop_images(
                kp_coco, merge_annos, part_type, origin_img_dir, res_img_subdir, img_prefix)

if __name__ == '__main__':
    main()