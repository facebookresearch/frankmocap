import os, sys, shutil
import os.path as osp
import json
import pdb
import cv2
import ry_utils
import multiprocessing as mp
import numpy as np
import parallel_io as pio


def crop_hand_single(img, keypoint):
    num_keypoint = len(keypoint)//3
    valid_kps = list()
    for i in range(num_keypoint):
        x, y, score = keypoint[i*3:(i+1)*3]
        if score > 0.0:
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

        margin = int(max(max_y-min_y, max_x-min_x) * 0.6)
        height, width = img.shape[:2]
        min_x = max(0, min_x-margin)
        max_x = min(width, max_x+margin)
        min_y = max(0, min_y-margin)
        max_y = min(height, max_y+margin)

        res_img = img[min_y:max_y, min_x:max_x, :]
        return res_img, np.array([min_x, min_y, max_x, max_y])
    else:
        return None, None


def get_all_joints(data):
    joints_2d = data['pose_keypoints_2d']
    for hand_type in ['left', 'right']:
        hand_joints_2d = data[f'hand_{hand_type}_keypoints_2d']
        joints_2d += hand_joints_2d
    return joints_2d

def crop_hand(frame_subdir, openpose_subdir, res_subdir, bbox_info_file):
    bbox_info = dict()
    for file in sorted(os.listdir(frame_subdir)):
        img_path = osp.join(frame_subdir, file)
        img = cv2.imread(img_path)
        json_file = osp.join(openpose_subdir, file[:-4] + "_keypoints.json")

        with open(json_file) as in_f:
            all_data = json.load(in_f)
            data = all_data['people']
            if len(data) == 0: continue

            data = data[0]
            joints_2d = get_all_joints(data) # include both hand and body joints

            img_id = json_file.split('/')[-1]
            img_id = img_id.replace('_keypoints.json','')

            keypoints = joints_2d.copy()
            res_img, bbox = crop_hand_single(img, keypoints)

            res_img_path = osp.join(res_subdir, f"{img_id}.png")
            img_key = img_id

            if res_img is None:
                bbox_info[img_key] = None
                continue
            else:
                bbox_info[img_key] = bbox
                if res_img.shape[0] == 0 or res_img.shape[1] == 0:
                    continue
                cv2.imwrite(res_img_path, res_img)
            
            pio.save_pkl_single(bbox_info_file, bbox_info)

    print(f"{frame_subdir} completes")


def main():
    # root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/body_capture"
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube"
    openpose_dir = osp.join(root_dir, "openpose_output")
    frame_dir = osp.join(root_dir, "frame")
    bbox_info_dir = osp.join(root_dir, "bbox_info")

    res_dir = osp.join(root_dir, f"image_body")
    ry_utils.build_dir(res_dir)
    bbox_info_file = osp.join(bbox_info_dir, f"body_bbox.pkl")

    for dir in os.listdir(openpose_dir):
        if dir == '.DS_Store': continue
        dir_path = osp.join(openpose_dir, dir)
        openpose_subdir = dir_path
        frame_subdir = osp.join(frame_dir, dir)
        res_subdir = osp.join(res_dir, dir)
        ry_utils.build_dir(res_subdir)
        crop_hand(frame_subdir, openpose_subdir, res_subdir, bbox_info_file)

        
if __name__ == '__main__':
    main()