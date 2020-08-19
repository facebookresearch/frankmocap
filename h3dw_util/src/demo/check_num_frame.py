import os, sys, shutil
import os.path as osp
import ry_utils
from collections import defaultdict


def get_num_frame(in_dir, extension):
    num_frame = defaultdict(int)
    all_files = ry_utils.get_all_files(in_dir, extension, "relative")
    for file in all_files:
        seq_name = file.split('/')[-2]
        num_frame[seq_name] += 1
    return num_frame


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube"

    openpose_dir = osp.join(root_dir, "openpose_output")
    openpose_num_frame = get_num_frame(openpose_dir, ".json")

    frame_dir = osp.join(root_dir, "frame")
    frame_num_frame = get_num_frame(frame_dir, ".jpg")

    for key in frame_num_frame:
        print(key, frame_num_frame[key], openpose_num_frame[key])


if __name__ == '__main__':
    main()