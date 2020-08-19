import sys
assert sys.version_info > (3, 0)
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import random
import pdb
import torch
import ry_utils
import parallel_io as pio
import utils.vis_utils as vu
import utils.rotate_utils as ru
from dataset.mtc.mtc_utils.data_utils import project2D
from utils.data_utils import remap_joints


def get_center_scale(joints_2d):
    min_x = np.min(joints_2d[:, 0])
    max_x = np.max(joints_2d[:, 0])
    min_y = np.min(joints_2d[:, 1])
    max_y = np.max(joints_2d[:, 1])

    center = ((max_x+min_x)/2, (max_y+min_y)/2)
    bbox_size = max(max_y-min_y, max_x-min_x)

    rescale = 1.2
    scale = bbox_size / 200 
    scale *= rescale
    return center, scale


def process_h36m(root_dir, img_dir, anno_file, res_anno_file):
    raw_data = pio.load_pkl_single(anno_file)
    all_data = list()
    for data in raw_data:
        # image
        img_name = data['image_path']
        img_path = osp.join(img_dir, img_name)
        img = cv2.imread(img_path)
        # joints
        joints_2d = data['joints_2d']
        center, scale = get_center_scale(joints_2d)
        # res dict
        res_data = dict(
            image_name = img_name,
            center = center,
            scale = scale,
        )
        all_data.append(res_data)
    pio.save_pkl_single(res_anno_file, all_data)


def main():
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/h36m_pick'
    img_dir = osp.join(root_dir, 'image')
    anno_file = osp.join(root_dir, 'annotation', 'raw.pkl')
    res_anno_file = osp.join(root_dir, 'annotation', 'val.pkl')
    process_h36m(root_dir, img_dir, anno_file, res_anno_file)


if __name__ == '__main__':
    main()