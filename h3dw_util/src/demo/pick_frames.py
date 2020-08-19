import os, sys, shutil
import os.path as osp
import json
import pdb
import cv2
import numpy as np
import ry_utils
import multiprocessing as mp


def pick_cello(dirs):
    for dir in dirs:
        dir_path = osp.join(dir, 'cello')
        all_files = ry_utils.get_all_files(dir_path, (".png", ".jpg"), "full")
        for file in all_files:
            if file.find("left_hand")>=0 or file.find("right_hand")>=0:
                frame_id = file.split('/')[-1].split('.')[0].split('_')[-3]
            else:
                frame_id = file.split('/')[-1].split('.')[0].split('_')[-1]
            frame_id = int(frame_id)
            if frame_id < 1000 or frame_id > 2000:
                os.remove(file)


def pick_oliver(dirs):
    for dir in dirs:
        dir_path = osp.join(dir, 'oliver_01')
        all_files = ry_utils.get_all_files(dir_path, (".png", ".jpg"), "full")
        for file in all_files:
            if file.find("left_hand")>=0 or file.find("right_hand")>=0:
                frame_id = file.split('/')[-1].split('.')[0].split('_')[-3]
            else:
                frame_id = file.split('/')[-1].split('.')[0].split('_')[-1]
            frame_id = int(frame_id)
            if frame_id < 12 or frame_id > 1200:
                os.remove(file)


def pick_lecture_04_02(dirs):
    for dir in dirs:
        dir_path = osp.join(dir, 'lecture_04_02')
        all_files = ry_utils.get_all_files(dir_path, (".png", ".jpg"), "full")
        for file in all_files:
            if file.find("left_hand")>=0 or file.find("right_hand")>=0:
                frame_id = file.split('/')[-1].split('.')[0].split('_')[-3]
            else:
                frame_id = file.split('/')[-1].split('.')[0].split('_')[-1]
            frame_id = int(frame_id)
            if frame_id > 590:
                os.remove(file)


def pick_lecture_01_01(dirs):
    for dir in dirs:
        dir_path = osp.join(dir, 'lecture_01_01')
        all_files = ry_utils.get_all_files(dir_path, (".png", ".jpg"), "full")
        for file in all_files:
            if file.find("left_hand")>=0 or file.find("right_hand")>=0:
                frame_id = file.split('/')[-1].split('.')[0].split('_')[-3]
            else:
                frame_id = file.split('/')[-1].split('.')[0].split('_')[-1]
            frame_id = int(frame_id)
            if frame_id < 615 or frame_id > 1195:
                os.remove(file)


def pick_legao(dirs):
    for dir in dirs:
        dir_path = osp.join(dir, 'legao_02_01')
        all_files = ry_utils.get_all_files(dir_path, (".png", ".jpg"), "full")
        for file in all_files:
            if file.find("left_hand")>=0 or file.find("right_hand")>=0:
                frame_id = file.split('/')[-1].split('.')[0].split('_')[-3]
            else:
                frame_id = file.split('/')[-1].split('.')[0].split('_')[-1]
            frame_id = int(frame_id)
            if frame_id < 1117 or frame_id > 2000:
                os.remove(file)


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube/origin"

    frame_dir = osp.join(root_dir, "frame")
    crop_left_hand_dir = osp.join(root_dir, "image_hand/left_hand")
    crop_right_hand_dir = osp.join(root_dir, "image_hand/right_hand")
    ov_dir = osp.join(root_dir, "openpose_visualization")
    render_two_hand_dir = osp.join(root_dir, "prediction/h3dw/origin_frame")
    render_single_hand_dir = osp.join(root_dir, "prediction/h3dw/224_size")

    # dirs = [frame_dir, render_dir, ov_dir]
    # dirs = [crop_left_hand_dir, crop_right_hand_dir]
    dirs = [frame_dir, crop_left_hand_dir, crop_right_hand_dir, ov_dir, render_two_hand_dir, render_single_hand_dir]

    pick_cello(dirs)
#  
    pick_oliver(dirs)

    pick_lecture_04_02(dirs)

    pick_lecture_01_01(dirs)

    pick_legao(dirs)

        
if __name__ == '__main__':
    main()