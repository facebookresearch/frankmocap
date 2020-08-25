import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import ry_utils
import parallel_io as pio
from scipy.io import loadmat
import random as rd
import pdb
import multiprocessing as mp

def select_file(in_dir, extension, num_select):
    all_files = list()
    for subdir, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith(extension):
                all_files.append(osp.join(subdir, file))
    return rd.sample(all_files, num_select)


def load_kernel(kernel_dir):
    kernels = dict()
    for file in os.listdir(kernel_dir):
        if file == '.DS_Store': continue

        record = file[:-4].split('_')
        psf_size = int(record[0])
        anxiety_100 = int(record[1])
        exp_time_10 = int(record[2])

        key = f"{psf_size:02d}_{anxiety_100:03d}_{exp_time_10:02d}"
        kernels[key] = loadmat(osp.join(kernel_dir, file))['PSFs'][0][0]

    return kernels
    

def gen_blur_img(origin_img, res_dir, kernels):
    kernels = sorted(list(kernels.items()), key=lambda a:a[0])
    for key, kernel in kernels:
        img = cv2.filter2D(origin_img, -1, kernel)

        res_img = np.concatenate((origin_img, img), axis=1)
        res_img_path = osp.join(res_dir, f"{key}.png")
        cv2.imwrite(res_img_path, res_img)


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/"
    img_dir = osp.join(root_dir, "ho3d/data/data_processed/image_tight")
    kernel_dir = osp.join(root_dir, "blur_kernel/kernel")

    selected_imgs = select_file(img_dir, 'png', 20)
    kernels = load_kernel(kernel_dir)
    # selected_kernels = select_file(kernel_dir, 'mat', 50)

    res_dir = "visualization/motion_blur"
    ry_utils.renew_dir(res_dir)
    p_list = list()
    for img_id, img_file in enumerate(selected_imgs):
        res_subdir = osp.join(res_dir, f"img_{img_id:02d}")
        ry_utils.renew_dir(res_subdir)
        origin_img = cv2.imread(img_file)

        p = mp.Process(target=gen_blur_img, args=(origin_img, res_subdir, kernels))
        p.start()
    
    for p in p_list:
        p.join()


if __name__ == '__main__':
    main()