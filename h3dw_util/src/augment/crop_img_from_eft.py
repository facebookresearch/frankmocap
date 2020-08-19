import os, sys, shutil
import os.path as osp
from scipy.io import loadmat
import numpy as np
import cv2
import pdb
import h5py
import ry_utils
import parallel_io as pio


def crop_body_img(img, seg):
    height, width = img.shape[:2]
    # print(np.unique(seg))
    Y, X = np.where( (seg<1.0) & (seg>0.0) )
    if len(Y)==0 or len(X)==0:
        return None
    min_x, max_x = np.min(X), np.max(X)
    min_y, max_y = np.min(Y), np.max(Y)
    crop_img = img[min_y:max_y, min_x:max_x, :]
    bbox = (min_x, min_y, max_x, max_y)
    return crop_img, bbox


def crop_hand_img(img, seg, body_bbox):
    min_bx, min_by, max_bx, max_by = body_bbox
    seg = seg[min_by:max_by, min_bx:max_bx]

    # get initial bbox
    Y, X = np.where( seg<1.0 )
    Y += min_by
    X += min_bx
    if len(Y)==0 or len(X)==0:
        return None
    min_x, max_x = np.min(X), np.max(X)
    min_y, max_y = np.min(Y), np.max(Y)

    img_height, img_width = img.shape[:2]
    height, width = max_y-min_y, max_x-min_x
    if width > height:
        margin = (width - height) // 2
        min_y = max(min_y-margin, 0)
        max_y = min(max_y+margin, img_height)
    else:
        margin = (-width + height) // 2
        min_x = max(min_x-margin, 0)
        max_x = min(max_x+margin, img_width)
    
    # multi-crop stage I: multi size
    res_img_list = list()
    min_x0 = min_x
    min_y0 = min_y
    max_x0 = max_x
    max_y0 = max_y
    for scale_ratio in [50, 60, 70, 80, 90, 100]:
        ratio = scale_ratio / 100
        margin = int((max_x0-min_x0) * ratio)
        min_y = max(min_y0-margin, 0)
        max_y = min(max_y0+margin, img_height)
        min_x = max(min_x0-margin, 0)
        max_x = min(max_x0+margin, img_width)
        # crop_img = cv2.rectangle(img.copy(), (min_x,min_y), (max_x, max_y), thickness=2, color=(0,0,255))
        crop_img = img[min_y:max_y:, min_x:max_x, :]
        res_img_list.append(crop_img)

    # mutli-crop stage II: position translation
    for s_ratio in [80, 90]:
        ratio = s_ratio / 100
        margin = int((max_x0-min_x0) * ratio)
        min_y = max(min_y0-margin, 0)
        max_y = min(max_y0+margin, img_height)
        min_x = max(min_x0-margin, 0)
        max_x = min(max_x0+margin, img_width)
        height, width = max_y-min_y, max_x-min_x

        ratio = 0.05
        for x in [-1, 1]:
            for y in [-1, 1]:
                x_shift = int(x * width * ratio)
                y_shift = int(y * height * ratio)
                min_y = max(min_y+y_shift, 0)
                min_x = max(min_x+x_shift, 0)
                max_y = min(max_y+y_shift, img_height)
                max_x = min(max_x+x_shift, img_width)
                # crop_img = cv2.rectangle(img.copy(), (min_x,min_y), (max_x, max_y), thickness=2, color=(0,0,255))
                crop_img = img[min_y:max_y:, min_x:max_x, :]
                res_img_list.append(crop_img)

    return res_img_list


def crop_image(img_dir, eft_dp_dir, res_dir):
    for file in os.listdir(eft_dp_dir):
        if file.endswith(".pkl"):
            eft_file = file
            # load img & eft fitting
            eft_path = osp.join(eft_dp_dir, eft_file)
            img_name = '_'.join(eft_file.split('_')[:-1]) + '.jpg'
            # image
            img_path = osp.join(img_dir, img_name)
            assert osp.exists(img_path)
            img = cv2.imread(img_path)
            # dp output
            dp_output = pio.load_pkl_single(eft_path)
            body_seg = dp_output['seg']
            hand_seg = dict(
                left = dp_output['seg_left_hand'],
                right = dp_output['seg_right_hand'])

            # crop body by seg
            img_body, body_bbox = crop_body_img(img, body_seg)
            if img_body is not None:
                res_img_path = osp.join(res_dir, "body", eft_file.replace(".pkl", ".jpg"))
                ry_utils.make_subdir(res_img_path)
                cv2.imwrite(res_img_path, img_body)

            # crop hand by seg
            for hand_type in ['left', 'right']:
                img_hand_list = crop_hand_img(img, hand_seg[hand_type], body_bbox)
                if img_hand_list is not None:
                    for img_id, img_hand in enumerate(img_hand_list):
                        if hand_type == 'left':
                            img_hand = np.fliplr(img_hand)
                        hand_img_name = eft_file.replace(".pkl", f"_{hand_type}_{img_id:03d}.jpg")
                        res_img_path = osp.join(res_dir, "hand", hand_img_name)
                        ry_utils.make_subdir(res_img_path)
                        cv2.imwrite(res_img_path, img_hand)


def process_coco():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/coco"
    img_dir = osp.join(root_dir, "image/train2014")
    eft_dp_dir = osp.join(root_dir, "eft_fitting/11-08_coco_with8143_dpOut")
    res_dir = osp.join(root_dir, "image_crop_eft")
    ry_utils.renew_dir(res_dir)
    crop_image(img_dir, eft_dp_dir, res_dir)


def process_lsp():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/lsp"
    img_dir = osp.join(root_dir, "image")
    eft_dp_dir = osp.join(root_dir, "eft_fitting/11-08_lspet_with8143_dpOut")
    res_dir = osp.join(root_dir, "image_crop_eft")
    ry_utils.renew_dir(res_dir)
    crop_image(img_dir, eft_dp_dir, res_dir)


def process_mpii():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/mpii"
    img_dir = osp.join(root_dir, "image")
    eft_dp_dir = osp.join(root_dir, "eft_fitting/11-08_mpii_with8143_dpOut")
    res_dir = osp.join(root_dir, "image_crop_eft")
    ry_utils.renew_dir(res_dir)
    crop_image(img_dir, eft_dp_dir, res_dir)


def main():
    # process_coco()
    # process_lsp()
    process_mpii()

    
if __name__ == '__main__':
    main()