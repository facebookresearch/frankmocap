import os
import os.path as osp
import sys
import ry_utils
import numpy as np
import cv2
from collections import defaultdict

def get_all_names(all_dirs):
    all_names = list()
    for in_dir in all_dirs:
        img_names = list()
        for subdir, dirs, files in os.walk(in_dir):
            for file in files:
                if file.endswith(".png"):
                    img_names.append(osp.join(subdir, file).replace(in_dir, '')[1:])
        all_names += img_names
    all_names = sorted(list(set(all_names)))
    return all_names


def frame_to_video(all_frames, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    sample_img = all_frames[0]
    height, width = sample_img.shape[:2]
    video_out = cv2.VideoWriter(video_path, fourcc, 25, (width, height))
    for frame in all_frames:
        video_out.write(frame)
    video_out.release()


def concat_result(all_names, all_dirs, res_dir):
    frames = defaultdict(list)
    for name_id, name in enumerate(all_names):

        print(name_id)
        if name_id < 3250: continue

        if name.find("cello")>=0: continue

        # load original images first
        frame_dir = all_dirs[0]
        img_path = osp.join(frame_dir, name).replace(".png", ".jpg")
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        img_list = [img,]

        # get h3dw output
        all_valid = True
        for in_dir in all_dirs[1:]:
            img_path = osp.join(in_dir, name)
            if not osp.exists(img_path): 
                all_valid = False
                break
            img_origin = cv2.imread(img_path)
            h, w = img_origin.shape[:2]
            if w / h > 1920 / 1080:
                img = cv2.imread(img_path)[:height, width:width*2, :]
            else:
                img = cv2.imread(img_path)[:height, :width, :]
            img_list.append(img)

        if not all_valid: continue


        res_img = np.concatenate(img_list, axis=1)

        # store single frames
        res_img_path = osp.join(res_dir, name)
        key = '_'.join(name.split('/')[:-1])
        frames[key].append(res_img)

        res_img_path = res_img_path.replace(".png", ".jpg")
        ry_utils.make_subdir(res_img_path)
        cv2.imwrite(res_img_path, res_img)

        print(f"{name_id:04d}/{len(all_names)}")

    # save results to images 
    for key in frames:
        video_path = osp.join(res_dir, f"{key}.mp4")
        frame_to_video(frames[key], video_path)
        print(f"{key} complete")


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_shared_02"

    frame_dir = osp.join(root_dir, "frame")
    mtc_res_dir = osp.join(root_dir, "mtc_output")
    fairmocap_res_dir = osp.join(root_dir, "fairmocap_output/frame")

    all_res_dirs = [frame_dir, mtc_res_dir, fairmocap_res_dir]

    all_names = get_all_names(all_res_dirs)

    res_dir = osp.join(root_dir, "compare/compare_between_mtc")
    ry_utils.build_dir(res_dir)
    concat_result(all_names, all_res_dirs, res_dir)
    
if __name__ == '__main__':
    main()
