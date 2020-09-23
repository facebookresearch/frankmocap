# Copyright (c) Facebook, Inc. and its affiliates.

"""
This code is used to visually compare the results
"""
import os, sys, shutil
import os.path as osp
import ry_utils
import cv2
import numpy as np

def check_keywords(subdir, keywords):
    if len(keywords) == 0:
        return True
    else:
        for keyword in keywords:
            if subdir.find(keyword)>=0:
                return True
    return False

def main():
    dir_list = [
        'samples/output/body/third_view_thresh_0.3_distance_2.0',
        'samples/output/body/third_view_thresh_0.5_distance_1.5',
        'samples/output/body/third_view_thresh_0.7_distance_1.0',
    ]
    dir1 = dir_list[0]

    keywords = ['cj_dance', 'body_capture']

    res_dir = "samples/output/body/third_view_compare"
    res_dir = osp.join(res_dir, '_&&_'.join(['_'.join(item.split('/')[-1:]) for item in dir_list]))

    for subdir in os.listdir(dir1):
        if osp.isdir(osp.join(dir1, subdir)):
            if check_keywords(subdir, keywords):
                dir_path1 = osp.join(dir1, subdir)
                for img_name in ry_utils.get_all_files(dir_path1, ('.jpg','.png'), 'name_only'):
                    img_list = list()
                    #print(img_name)
                    for dir in dir_list:
                        dir_path = dir_path1.replace(dir1, dir)
                        img_path = osp.join(dir_path, img_name)
                        img = cv2.imread(img_path)
                        img_list.append(img)
                        if img_path.find(dir1)>=0:
                            res_img_path = img_path.replace(dir1, res_dir)
                        #print(img_path, osp.exists(img_path))
                    if any([img is None for img in img_list]):
                        continue
                    res_img = np.concatenate(img_list, axis=0)
                    h, w = res_img.shape[:2]
                    res_img = cv2.resize(res_img, (int(w*0.7), int(h*0.7)))
                    res_img_path = res_img_path.replace('.png', '.jpg')
                    ry_utils.make_subdir(res_img_path)
                    cv2.imwrite(res_img_path, res_img)
                    print(res_img_path)



if __name__ == '__main__':
     main()
