import os, sys, shutil
import os.path as osp
import json
import pdb
import cv2
import numpy as np
import ry_utils
import multiprocessing as mp


def draw_keypoints(img, keypoint, color=(0, 0, 255)):
    num_keypoint = len(keypoint)//3
    res_img = img.copy()
    for i in range(num_keypoint):
        x, y, score = keypoint[i*3:(i+1)*3]
        if score > 0:
            cv2.circle(res_img, (int(x), int(y)), radius=5, color=color, thickness=-1)
    return res_img


def draw_bbox(img, keypoint, color=(0, 0, 255)):
    keypoint = np.array(keypoint).reshape(-1, 3)
    min_x = np.min(keypoint[:, 0]).astype(np.int32)
    max_x = np.max(keypoint[:, 0]).astype(np.int32)
    min_y = np.min(keypoint[:, 1]).astype(np.int32)
    max_y = np.max(keypoint[:, 1]).astype(np.int32)
    res_img =cv2.rectangle(img.copy(), (min_x,min_y), (max_x, max_y), color=color, thickness=5)
    return res_img


def read_single(frame_subdir, openpose_subdir, res_subdir):
    colors = [
       ( (0, 0, 255), (0, 0, 122) ),
       ( (0, 255, 0), (0, 122, 0) ),
       ( (255, 0, 0), (122, 0, 0) ),
       ( (0, 255, 255), (0, 122, 122) ),
       ( (255, 255, 0), (122, 122, 0) ),
       ( (255, 0, 255), (122, 0, 122) ),
    ]

    all_files = ry_utils.get_all_files(frame_subdir, (".png", ".jpg"), "name_only")
    for file in all_files:
        img_path = osp.join(frame_subdir, file)
        json_file = osp.join(openpose_subdir, file[:-4] + "_keypoints.json")

        with open(json_file) as in_f:
            all_data = json.load(in_f)['people']
            img_id = json_file.split('/')[-1]
            img_id = img_id.replace('_keypoints.json','')
            img = cv2.imread(img_path)
            res_img_path = osp.join(res_subdir, f"{img_id}.jpg")
            res_img = img

            for data_id, data in enumerate(all_data):
                pose_2d = data['pose_keypoints_2d']
                left_hand_2d = data['hand_left_keypoints_2d']
                right_hand_2d = data['hand_right_keypoints_2d']

                color_l, color_r = colors[data_id]
                res_img = draw_bbox(res_img, right_hand_2d.copy(), color=color_r)
                res_img = draw_keypoints(res_img, right_hand_2d.copy(), color=color_r)
                res_img = draw_bbox(res_img, left_hand_2d.copy(), color=color_l)
                res_img = draw_keypoints(res_img, left_hand_2d.copy(), color=color_l)
            cv2.imwrite(res_img_path, res_img)
    print(f"{frame_subdir} completes")


def main():
    # root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/demo_image/"
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_multi/"

    frame_dir = osp.join(root_dir, "frame")
    openpose_dir = osp.join(root_dir, "openpose_output")

    res_dir = osp.join(root_dir, "openpose_visualization")
    ry_utils.renew_dir(res_dir)

    process_list = list()
    for dir in os.listdir(openpose_dir):
        if dir == '.DS_Store': continue
        dir_path = osp.join(openpose_dir, dir)
        openpose_subdir = dir_path
        frame_subdir = osp.join(frame_dir, dir)
        res_subdir = osp.join(res_dir, dir)
        ry_utils.renew_dir(res_subdir)

        p = mp.Process(target=read_single, 
            args=(frame_subdir, openpose_subdir, res_subdir))
        process_list.append(p)
        p.start()
    
    for p in process_list:
        p.join()

        
if __name__ == '__main__':
    main()