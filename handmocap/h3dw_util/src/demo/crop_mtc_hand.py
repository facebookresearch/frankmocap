import os, sys, shutil
import os.path as osp
import json
import pdb
import cv2
import ry_utils
import multiprocessing as mp
import numpy as np


def crop_hand_single(img, keypoint):
    new_height = 1920
    new_width = int(new_height/img.shape[0] * img.shape[1])
    img = cv2.resize(img, (new_width, new_height))

    num_keypoint = len(keypoint)//3
    valid_kps = list()
    for i in range(num_keypoint):
        x, y, score = keypoint[i*3:(i+1)*3]
        if score > 0.001:
            valid_kps.append((x,y))
    
    if len(valid_kps)>5: 
        valid_kps = np.array(valid_kps).astype(np.int32)
        min_x = np.min(valid_kps[:, 0])
        max_x = np.max(valid_kps[:, 0])
        min_y = np.min(valid_kps[:, 1])
        max_y = np.max(valid_kps[:, 1])

        center = (min_x+max_x)//2, (min_y+max_y)//2
        size = max(max_x-min_x, max_y-min_y)//2
        min_x = center[0]-size
        max_x = center[0]+size
        min_y = center[1]-size
        max_y = center[1]+size

        margin = int(max(max_y-min_y, max_x-min_x) * 1.0)
        height, width = img.shape[:2]
        min_x = max(0, min_x-margin)
        max_x = min(width, max_x+margin)
        min_y = max(0, min_y-margin)
        max_y = min(height, max_y+margin)

        res_img = img[min_y:max_y, min_x:max_x, :]
        return res_img
    else:
        return None


def crop_hand(hand_type, frame_subdir, openpose_subdir, res_subdir):
    for file in sorted(os.listdir(frame_subdir)):
        img_path = osp.join(frame_subdir, file)
        img = cv2.imread(img_path)
        subdir_name = openpose_subdir.split('/')[-1]
        json_file = osp.join(openpose_subdir, f"{subdir_name}_{file[:-4]}_keypoints.json")

        with open(json_file) as in_f:
            all_data = json.load(in_f)
            data = all_data['people']
            if len(data) == 0: continue

            data = data[0]
            key = f"hand_{hand_type}_keypoints_2d"
            hand_2d = data[key]

            img_id = json_file.split('/')[-1]
            img_id = img_id.replace('_keypoints.json','')

            keypoint = hand_2d.copy()
            res_img_path = osp.join(res_subdir, f"{img_id}_{hand_type}_hand.jpg")
            res_img = crop_hand_single(img, keypoint)
            if res_img is None:
                continue
            else:
                if hand_type == 'left':
                    res_img = np.fliplr(res_img)
                cv2.imwrite(res_img_path, res_img)

    print(f"{frame_subdir} completes")


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    exp_res_dir = "experiment/experiment_results/3d_hand/h3dw/demo_data/body_capture/"
    openpose_dir = osp.join(root_dir, "data/demo_data/body_capture/openpose_output_origin")
    mtc_hand_dir = osp.join(root_dir, exp_res_dir, "hand_prediction/mtc_hand_result")
    frame_dir = osp.join(root_dir, "data/demo_data/body_capture/mtc_prediction")

    for hand_type in ['left', 'right']:
        res_dir = osp.join(mtc_hand_dir, f"{hand_type}_hand")
        ry_utils.renew_dir(res_dir)
        for dir in os.listdir(openpose_dir):
            if dir == '.DS_Store': continue
            dir_path = osp.join(openpose_dir, dir)
            openpose_subdir = dir_path
            frame_subdir = osp.join(frame_dir, dir)
            res_subdir = osp.join(res_dir, dir)
            ry_utils.renew_dir(res_subdir)
            crop_hand(hand_type, frame_subdir, openpose_subdir, res_subdir)
     

if __name__ == '__main__':
    main()