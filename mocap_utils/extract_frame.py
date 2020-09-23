# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import subprocess as sp
import general_utils as gnu


def extract_frame(video_dir, frame_dir):
    for file in os.listdir(video_dir):
        if file.endswith((".mov", ".mp4")):
            file_path = osp.join(video_dir, file)
            file_name = file[:-4]
            # if file_name != 'legao_02_01': continue
            res_dir = osp.join(frame_dir, file_name)
            gnu.build_dir(res_dir)
            command = f"ffmpeg -i {file_path} {res_dir}/{file_name}_%05d.png"
            command = command.split()
            sp.run(command)


def main():
    root_dir = "./sample_data/"

    video_dir = osp.join(root_dir, "videos")
    frame_dir = osp.join(root_dir, "frames")
    gnu.renew_dir(frame_dir)

    extract_frame(video_dir, frame_dir)

if __name__ == '__main__':
    main()