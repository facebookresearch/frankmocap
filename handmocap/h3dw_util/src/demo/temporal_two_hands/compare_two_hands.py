import os
import os.path as osp
import sys
import ry_utils
import numpy as np
import cv2
from collections import defaultdict

def get_all_names(all_dirs, select_seqs):
    all_names = list()
    for in_dir in all_dirs:
        # print(in_dir)
        # continue
        names = ry_utils.get_all_files(in_dir, (".png", ".jpg"), "relative")
        for name in names:
            seq_name = name.split('/')[-2]
            if seq_name in select_seqs:
                all_names.append(name)
    all_names = sorted(list(set(all_names)))
    return all_names


def concat_result(all_names, in_dirs, res_dir):
    frames = defaultdict(list)
    for name_id, name in enumerate(all_names):
        # load origin image and determine size

        num_exist = 0
        for in_dir in in_dirs:
            img_path = osp.join(in_dir, name)
            num_exist += osp.exists(img_path)
        if num_exist != len(in_dirs): continue
        
        all_imgs = list()
        for in_dir in in_dirs:
            img_path = osp.join(in_dir, name)
            all_imgs.append(cv2.imread(img_path))
        res_img = np.concatenate(all_imgs, axis=1)

        res_img_path = osp.join(res_dir, name)
        ry_utils.make_subdir(res_img_path)
        cv2.imwrite(res_img_path, res_img)

        print(f"{name_id:04d}/{len(all_names)}")


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube/"

    h3dw_dirs = [
        # "augment_bbox/prediction/h3dw/origin_frame",
        "temporal_refine/origin_frame",
        "temporal_refine/merge_frame/copy_and_paste",
        "temporal_refine/update_frame/average_frame"
    ]
    h3dw_full_dirs = [osp.join(root_dir, dir) for dir in h3dw_dirs]

    select_seqs = ['lecture_01_01', ]
    all_names = get_all_names(h3dw_full_dirs, select_seqs)
    print(len(all_names))

    res_dir = osp.join(root_dir, "temporal_refine/compare/average_frame")
    ry_utils.build_dir(res_dir)
    concat_result(all_names, h3dw_full_dirs, res_dir)
    
if __name__ == '__main__':
    main()
