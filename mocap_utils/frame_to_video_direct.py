# Copyright (c) Facebook, Inc. and its affiliates.

import os
import os.path as osp
import sys
import ry_utils
import numpy as np
import cv2
from collections import defaultdict
# from  ..utils.vis_utils import img_pad_and_resize

def frame_to_video(all_frames, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    sample_img = all_frames[0]
    height, width = sample_img.shape[:2]
    video_out = cv2.VideoWriter(video_path, fourcc, 25, (width, height))
    for frame in all_frames:
        video_out.write(frame)
    video_out.release()


def get_all_frames(all_files):
    # all_files = all_files[:20]
    # all_frames = [cv2.imread(file) for file in all_files]
    all_frames = list()
    for file in all_files:
        frame = cv2.imread(file)
        h, w = frame.shape[:2]
        # if h>2000 or w>2000:
            # frame = cv2.resize(frame, (w//2, h//2))
        all_frames.append(frame)
    return all_frames


def main():
    in_dir = "sample_data/output/videos/rongyu_hand/output/"

    all_seq =list()
    for file in os.listdir(in_dir):
        if osp.isdir(osp.join(in_dir, file)):
            all_seq.append(file)
    
    res_dir = in_dir
    ry_utils.build_dir(res_dir)

    for seq_name in all_seq:
        
        print(f"{seq_name} starts")
        subdir = osp.join(in_dir, seq_name)
        all_files = ry_utils.get_all_files(subdir, (".png", ".jpg"), "full")
        all_frames = get_all_frames(all_files)
        video_path = osp.join(res_dir, f"{seq_name}.mp4")
        frame_to_video(all_frames, video_path)
        print(f"{seq_name} ends")

    
if __name__ == '__main__':
    main()
