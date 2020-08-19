import os, sys, shutil
import os.path as osp
import json
import pdb
import cv2
import ry_utils
import multiprocessing as mp
import numpy as np


def crop_hand_single(img, keypoint):
    num_keypoint = len(keypoint)//3
    valid_kps = list()
    for i in range(num_keypoint):
        x, y, score = keypoint[i*3:(i+1)*3]
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
        return res_img
    else:
        return None


def crop_hand(hand_type, frame_subdir, openpose_subdir, res_subdir):
    for file in sorted(os.listdir(frame_subdir)):
        img_path = osp.join(frame_subdir, file)
        img = cv2.imread(img_path)
        json_file = osp.join(openpose_subdir, file[:-4] + "_keypoints.json")

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
            res_img_path = osp.join(res_subdir, f"{img_id}_{hand_type}_hand.png")
            res_img = crop_hand_single(img, keypoint)
            if res_img is None:
                continue
            else:
                if hand_type == 'left':
                    res_img = np.fliplr(res_img)
                cv2.imwrite(res_img_path, res_img)

    print(f"{frame_subdir} completes")


def main():
    '''
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/experiment/" + \
        "experiment_results/3d_hand/h3dw/demo_data/body_capture/prediction/hand"
    '''
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/coco/prediction/hand"
    in_dir = osp.join(root_dir, "images")

    hand_img_dir = osp.join(root_dir, "image_hand")
    render_img_dir = osp.join(root_dir, "image_render")
    ry_utils.renew_dir(hand_img_dir)
    ry_utils.renew_dir(render_img_dir)

    for subdir, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith((".png", ".jpg")):
                img_path = osp.join(subdir, file)
                img = cv2.imread(img_path)

                hand_img_path = img_path.replace(in_dir, hand_img_dir)
                ry_utils.make_subdir(hand_img_path)
                render_img_path = img_path.replace(in_dir, render_img_dir)
                ry_utils.make_subdir(render_img_path)

                height, width = img.shape[:2]
                hand_img = img[:, :height, :]
                render_img = img[:, height:height*2, :]
                
                cv2.imwrite(hand_img_path, hand_img)
                cv2.imwrite(render_img_path, render_img)
        
if __name__ == '__main__':
    main()