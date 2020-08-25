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
                if file.endswith(".jpg"):
                    img_names.append(osp.join(subdir, file).replace(in_dir, '')[1:])
        all_names += img_names
    all_names = sorted(list(set(all_names)))
    return all_names


def pad_and_resize(img, final_size=224):
    height, width = img.shape[:2]
    if height > width:
        ratio = final_size / height
        new_height = final_size
        new_width = int(ratio * width)
    else:
        ratio = final_size / width
        new_width = final_size
        new_height = int(ratio * height)
    new_img = np.zeros((final_size, final_size, 3), dtype=np.uint8)
    new_img[:new_height, :new_width, :] = cv2.resize(img, (new_width, new_height))
    return new_img


def frame_to_video(all_frames, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    sample_img = all_frames[0]
    height, width = sample_img.shape[:2]
    video_out = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    for frame in all_frames:
        video_out.write(frame)
    video_out.release()


def concat_result(all_names, h3dw_full_dirs, res_dir):
    frames = defaultdict(list)
    for name_id, name in enumerate(all_names):
        # load origin image and determine size
        in_dir = h3dw_full_dirs[0]
        img_path = osp.join(in_dir, name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        origin_img = img[:, :height, :]
        img_list = [origin_img,]

        # get h3dw output
        for in_dir in h3dw_full_dirs:
            img_path = osp.join(in_dir, name)
            img = cv2.imread(img_path)
            img = img[:, height:height*2, :]
            img_list.append(img)
        res_img = np.concatenate(img_list, axis=1)

        # store single frames
        res_img_path = osp.join(res_dir, name)
        key = '_'.join(name.split('/')[:-1])
        frames[key].append(res_img)

        ry_utils.make_subdir(res_img_path)
        cv2.imwrite(res_img_path, res_img)

        print(f"{name_id:04d}/{len(all_names)}")

    # save results to images 
    for key in frames:
        video_path = osp.join(res_dir, f"{key}.mp4")
        frame_to_video(frames[key], video_path)
        print(f"{key} complete")


def main():
    '''
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    h3dw_root_dir = "experiment/experiment_results/3d_hand/h3dw/demo_data/body_capture/hand_prediction/h3dw"
    h3dw_dirs = [
        "use_motion_blur/1e-3_prob-0.5",
        "use_stb_rhd_data/prob-0.7_1e-3"
    ]
    h3dw_full_dirs = [osp.join(root_dir, h3dw_root_dir, dir) for dir in h3dw_dirs]
    '''

    root_dir = "visualization/temporal/render_body/"

    '''
    h3dw_dirs = [
        "origin/body_wrist_rot",
        "origin/hand_wrist_rot",
        "update_wrist/full"
    ]
    '''

    h3dw_dirs = [
        "origin/hand_wrist_rot",
        "update_wrist/full",
        "average/3_wrist_combine",
    ]
    h3dw_full_dirs = [osp.join(root_dir, dir) for dir in h3dw_dirs]

    all_names_h3dw = get_all_names(h3dw_full_dirs)
    all_names = sorted(list(set(all_names_h3dw)))
    

    res_dir = "visualization/compare/body_capture_between_h3dw/average"
    ry_utils.renew_dir(res_dir)
    concat_result(all_names, h3dw_full_dirs, res_dir)
    
if __name__ == '__main__':
    main()
