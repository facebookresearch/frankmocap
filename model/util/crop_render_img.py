import os, sys, shutil
import os.path as osp
import ry_utils
import cv2
import numpy as np


def main():
    in_dir = "evaluate_results/images"
    out_dir = "../../render_images"
    ry_utils.build_dir(out_dir)

    all_imgs = ry_utils.get_all_files(in_dir, ".jpg", "relative")
    for img_path in all_imgs:
        in_path = osp.join(in_dir, img_path)
        out_path = osp.join(out_dir, img_path)
        
        if osp.exists(out_path): continue

        img = cv2.imread(in_path)
        h, w = img.shape[:2]
        render_img = img[:, h:h*2, :]

        ry_utils.make_subdir(out_path)
        cv2.imwrite(out_path, render_img)

        print(img_path)


if __name__ == '__main__':
    main()