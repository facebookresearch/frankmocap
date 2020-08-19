import os
import os.path as osp
import sys
import ry_utils
import numpy as np
import cv2
from collections import defaultdict
# from  ..utils.vis_utils import img_pad_and_resize

def img_pad_and_resize(img, final_size=224):
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

def get_all_names(in_dir):
    img_names = list()
    for subdir, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith(".png"):
                img_names.append(osp.join(subdir, file).replace(in_dir, '')[1:])
    all_names = sorted(list(set(img_names)))
    return all_names


def frame_to_video(all_frames, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    sample_img = all_frames[0]
    height, width = sample_img.shape[:2]
    video_out = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    for frame in all_frames:
        video_out.write(frame)
    video_out.release()

def get_frames(all_names, h3dw_dir, frame_dir, openpose_anno_dir, res_frame_dir):
    all_res_frames = defaultdict(list)
    for name in all_names:
        h3dw_img_path = osp.join(h3dw_dir, name)
        op_anno_img_path = osp.join(openpose_anno_dir, name)
        frame_path = osp.join(frame_dir, name).replace("_right_hand.png", ".png")
        # print(frame_path)

        h3dw_img = cv2.imread(h3dw_img_path)
        anno_img = cv2.imread(op_anno_img_path)
        frame = cv2.imread(frame_path)

        h, w = h3dw_img.shape[:2]
        frame = img_pad_and_resize(frame, h*6)

        hand_img = h3dw_img[:, :h, :]
        anno_img = img_pad_and_resize(anno_img, h)
        res_img = np.concatenate((hand_img, anno_img, h3dw_img[:, h:, :]), axis=1)
        res_img = np.concatenate((frame, res_img), axis=0)

        seq_name = name.split('/')[0]
        all_res_frames[seq_name].append(res_img)

        res_img_path = osp.join(res_frame_dir, name)
        res_subdir = osp.join(res_frame_dir, seq_name)
        ry_utils.build_dir(res_subdir)
        cv2.imwrite(res_img_path, res_img)
    return all_res_frames
    

def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    h3dw_dir = osp.join(root_dir, 
        "experiment/experiment_results/3d_hand/h3dw/demo_data/body_capture/top_finger_ave_1e-3")
    all_names_h3dw = get_all_names(h3dw_dir)
    all_names = sorted(list(set(all_names_h3dw)))

    openpose_anno_dir = osp.join(root_dir, "data/demo_data/body_capture/image_hand_anno")
    frame_dir = osp.join(root_dir, "data/demo_data/body_capture/frame_origin")

    res_dir = osp.join("visualization/body_capture")
    ry_utils.renew_dir(res_dir)
    res_frame_dir = osp.join(res_dir, "frame")
    ry_utils.renew_dir(res_frame_dir)

    all_res_frames = get_frames(all_names, h3dw_dir, frame_dir, openpose_anno_dir, res_frame_dir)
   
    for seq_name in all_res_frames:
        frames = all_res_frames[seq_name]
        video_path = osp.join(res_dir, f"{seq_name}.mp4")
        frame_to_video(frames, video_path)

    
if __name__ == '__main__':
    main()
