# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import cv2
import numpy as np
import general_utils as gnu


def main():
    in_dir = "./sample_data/images/single_person"
    out_dir = "./sample_data/images/multi_person"
    gnu.renew_dir(out_dir)

    all_imgs = gnu.get_all_files(in_dir, (".jpg", ".png", ".jpeg"), "full")
    num_img = len(all_imgs)

    for i in range(num_img):
        for j in range(num_img):
            img1 = cv2.imread(all_imgs[i])
            img2 = cv2.imread(all_imgs[j])
            img2 = cv2.resize(img2, img1.shape[:2][::-1])
            res_img = np.concatenate((img1, img2), axis=1)
            res_img_path = osp.join(out_dir, f"{i:02d}_{j:02d}.jpg")
            cv2.imwrite(res_img_path, res_img)


if __name__ == '__main__':
    main()