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

        # margin = int(max(max_y-min_y, max_x-min_x) * 0.6)
        margin = int(max(max_y-min_y, max_x-min_x) * 0.3)
        height, width = img.shape[:2]
        min_x = max(0, min_x-margin)
        max_x = min(width, max_x+margin)
        min_y = max(0, min_y-margin)
        max_y = min(height, max_y+margin)

        res_img = img[min_y:max_y, min_x:max_x, :]
        return res_img, np.array([min_x, min_y, max_x, max_y])
    else:
        return None, None


def crop_hand(hand_type, frame_subdir, openpose_subdir, res_subdir, bbox_info_file):
    bbox_info = dict()
    # for file in sorted(os.listdir(frame_subdir)):
    for file in ry_utils.get_all_files(frame_subdir, (".png", "jpg"), "name_only"):
        img_path = osp.join(frame_subdir, file)
        img = cv2.imread(img_path)
        json_file = osp.join(openpose_subdir, file[:-4] + "_keypoints.json")
        img_id = json_file.split('/')[-1]
        img_id = img_id.replace('_keypoints.json','')

        with open(json_file) as in_f:
            all_data = json.load(in_f)['people']
            if len(all_data) == 0: continue

            for data_id, data in enumerate(all_data):

                key = f"hand_{hand_type}_keypoints_2d"
                hand_2d = data[key]
                keypoint = hand_2d.copy()
                res_img, bbox = crop_hand_single(img, keypoint)

                res_img_path = osp.join(res_subdir, f"{img_id}_{data_id:02d}_{hand_type}_hand.png")
                img_key = f"{img_id}_{data_id:02d}_{hand_type}_hand"

                if res_img is None:
                    bbox_info[img_key] = None
                    continue
                else:
                    bbox_info[img_key] = bbox
                    if res_img.shape[0] == 0 or res_img.shape[1] == 0:
                        continue
                    if hand_type == 'left':
                        res_img = np.fliplr(res_img)
                    cv2.imwrite(res_img_path, res_img)
            pio.save_pkl_single(bbox_info_file, bbox_info)

    print(f"{frame_subdir} completes")


def main():
    # root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/body_capture"
    # root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/demo_image"
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube_multi"

    openpose_dir = osp.join(root_dir, "openpose_output")
    frame_dir = osp.join(root_dir, "frame")
    bbox_info_dir = osp.join(root_dir, "bbox_info")

    process_list = list()
    for hand_type in ['left', 'right']:
        res_dir = osp.join(root_dir, f"image_hand/{hand_type}_hand")
        # ry_utils.renew_dir(res_dir)

        for dir in os.listdir(frame_dir):
            if dir == '.DS_Store': continue
            
            bbox_info_file = osp.join(bbox_info_dir, f"{dir}_{hand_type}_bbox.pkl")
            ry_utils.make_subdir(bbox_info_file)

            dir_path = osp.join(openpose_dir, dir)
            openpose_subdir = dir_path
            frame_subdir = osp.join(frame_dir, dir)
            res_subdir = osp.join(res_dir, dir)
            ry_utils.build_dir(res_subdir)
            p = mp.Process(target=crop_hand, 
                args=(hand_type, frame_subdir, openpose_subdir, res_subdir, bbox_info_file))
            process_list.append(p)
            p.start()
    
    for p in process_list:
        p.join()

        
if __name__ == '__main__':
    main()