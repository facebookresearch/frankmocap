import os, sys, shutil
import os.path as osp
from scipy.io import loadmat
import numpy as np
import cv2
import pdb
import h5py
import ry_utils
import parallel_io as pio

def crop_img_by_kps(img, kps, margin_ratio):
    height, width = img.shape[:2]
    valid = kps[:, 2]>0
    valid_kps = kps[valid, :2]
    kps = valid_kps
    min_x, min_y = np.min(kps, axis=0).astype(np.int32)
    max_x, max_y = np.max(kps, axis=0).astype(np.int32)

    margin = int(max(max_y-min_y, max_x-min_x) * margin_ratio)
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    min_y = max(min_y, 0)
    max_y = min(max_y, height)
    min_x = max(min_x, 0)
    max_x = min(max_x, width)
    # print(min_y,max_y,min_x,max_x)
    # print(img.shape)
    return img[min_y:max_y, min_x:max_x, :]


def load_imgs(img_dir):
    all_imgs = list()
    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(".png"):
                all_imgs.append(osp.join(subdir, file))
    return sorted(all_imgs)


def process_lsp():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/lsp"
    img_dir = osp.join(root_dir, "images")
    all_imgs = load_imgs(img_dir)

    joint_file = osp.join(root_dir, "joints.mat")
    joints_all = loadmat(joint_file)['joints']
    assert len(all_imgs) == joints_all.shape[2]

    res_img_dir = osp.join(root_dir, "images_body")
    ry_utils.renew_dir(res_img_dir)
    for i, img_path in enumerate(all_imgs):
        joints = joints_all[:, :, i]
        img = cv2.imread(img_path)
        crop_img = crop_img_by_kps(img, joints, 0.3)
        res_img_path = img_path.replace(img_dir, res_img_dir).replace(".png", ".jpg")
        cv2.imwrite(res_img_path, crop_img)
        if i%100 == 0 and i>0:
            print(f"Processed: {i:05d}")


def process_mpii():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/mpii"
    img_dir = osp.join(root_dir, "images")
    all_imgs = load_imgs(img_dir)

    res_img_dir = osp.join(root_dir, "images_body")
    ry_utils.renew_dir(res_img_dir)
    anno_file = osp.join(root_dir, "train.h5")
    with h5py.File(anno_file, 'r') as f:
        centers, img_names, parts, scales = \
            f['center'], f['imgname'], f['part'], f['scale']
        
        img_names = sorted([img_name.decode('utf-8') for img_name in img_names])
        for i, img_name in enumerate(img_names):
            joints = parts[i]
            joints_sum = np.sum(joints, axis=1)
            score = np.ones((joints.shape[0], 1))
            score[joints_sum<1e-8] = 0.0
            joints = np.concatenate((joints, score), axis=1)
            print(img_name)
            # print(joints)
            img_path = osp.join(img_dir, img_name)
            img = cv2.imread(img_path)
            crop_img =crop_img_by_kps(img, joints, 0)
            res_img_path = img_path.replace(img_dir, res_img_dir).replace(".png", ".jpg")
            cv2.imwrite(res_img_path, crop_img)
            if i>10: break
            if i%100 == 0 and i>0:
                print(f"Processed: {i:05d}")

    # print(img_names[0])
    # for i, img_name in enumerate(img_names):
        # print(img_name.decode('utf-8'))
    '''
    joint_file = osp.join(root_dir, "annotation.mat")
    all_data = loadmat(joint_file)["RELEASE"]
    pio.save_pkl_single(osp.join(root_dir, "annotation.pkl"), all_data)
    pdb.set_trace()
    assert len(all_imgs) == joints_all.shape[2]

    res_img_dir = osp.join(root_dir, "images_body")
    ry_utils.renew_dir(res_img_dir)
    for i, img_path in enumerate(all_imgs):
        joints = joints_all[:, :, i]
        img = cv2.imread(img_path)
        crop_img = crop_img_by_kps(img, joints, 0.3)
        res_img_path = img_path.replace(img_dir, res_img_dir).replace(".png", ".jpg")
        cv2.imwrite(res_img_path, crop_img)
        if i%100 == 0 and i>0:
            print(f"Processed: {i:05d}")
    '''


def main():
    # process_lsp()
    process_mpii()


if __name__ == '__main__':
    main()