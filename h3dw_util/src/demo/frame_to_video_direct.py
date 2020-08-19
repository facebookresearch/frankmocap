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


def main():
    # in_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_processed_03/prediction/h3dw/origin_frame"
    # in_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_processed_03/prediction/h3dw/origin_frame"
    # in_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube/origin/prediction/h3dw/origin_frame"
    # in_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube/temporal_refine/compare/average_frame"
    # in_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_multi/prediction/h3dw/origin_frame"
    # in_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_shared_02/frame"
    in_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_shared_02/compare/compare_between_mtc"


    all_seq =list()
    for file in os.listdir(in_dir):
        if osp.isdir(osp.join(in_dir, file)):
            all_seq.append(file)
    
    res_dir = in_dir
    # res_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_shared_02/video"
    ry_utils.build_dir(res_dir)

    for seq_name in all_seq:
        print(f"{seq_name} starts")
        subdir = osp.join(in_dir, seq_name)
        # all_files = sorted([osp.join(subdir, file) for file in os.listdir(subdir) if file.endswith('.png')])
        all_files = ry_utils.get_all_files(subdir, (".png", ".jpg"), "full")
        all_frames = [cv2.imread(file) for file in all_files]
        video_path = osp.join(res_dir, f"{seq_name}.mp4")
        frame_to_video(all_frames, video_path)
        print(f"{seq_name} ends")

    
if __name__ == '__main__':
    main()
