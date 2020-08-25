import os, sys, shutil
import os.path as osp
import json
import pdb
import cv2
import ry_utils
import multiprocessing as mp
import numpy as np
from crop_hand import crop_hand_single


def paste_hand_single(hand_type, bbox, render_img_path, res_img):
    min_x, min_y, max_x, max_y = bbox

    if min_x < max_x and min_y < max_y:
        h = max_y - min_y
        w = max_x - min_x
        render_img_all = cv2.imread(render_img_path)
        r_size = render_img_all.shape[0]
        render_img = render_img_all[:, r_size:r_size*2, :]
        render_hand = render_img[:h, :w, :]
        if hand_type == 'left':
            render_hand = np.fliplr(render_hand)
        res_img[min_y:max_y, min_x:max_x, :] = render_hand


def paste_hand(frame_subdir, openpose_subdir, render_subdir, res_subdir):
    for file in sorted(os.listdir(frame_subdir)):
        frame_img_path = osp.join(frame_subdir, file)
        json_file = osp.join(openpose_subdir, file[:-4] + "_keypoints.json")
        img_id = json_file.split('/')[-1].replace('_keypoints.json','')
        res_img_path = osp.join(res_subdir, f"{img_id}.png")
        res_img = cv2.imread(frame_img_path)

        for hand_type in ["left", "right"]:
            render_dir = '/'.join(render_subdir.split('/')[:-1])
            seq_name = render_subdir.split('/')[-1]
            render_img_path = osp.join(render_dir, f'{hand_type}_hand', 
                seq_name, file.replace(".png", f"_{hand_type}_hand.jpg"))
            if not osp.exists(render_img_path):
                continue

            with open(json_file) as in_f:
                all_data = json.load(in_f)
                data = all_data['people']
                data = data[0]
                hand_2d = data[f'hand_{hand_type}_keypoints_2d']
                frame = cv2.imread(frame_img_path)

                bbox = crop_hand_single(frame, hand_2d.copy())[1]
                if bbox is None:
                    res_img = frame
                else:
                    paste_hand_single(hand_type, bbox, render_img_path, res_img)

        cv2.imwrite(res_img_path, res_img)
        print(res_img_path)

    print(f"{frame_subdir} completes")


def main():
    # root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/body_capture"
    root_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/youtube'
    frame_dir = osp.join(root_dir, "frame")
    render_img_dir = osp.join(root_dir, 'prediction/origin_size')
    openpose_dir = osp.join(root_dir, "openpose_output")

    res_dir = osp.join(root_dir, "prediction/origin_frame")
    ry_utils.renew_dir(res_dir)

    process_list = list()
    for dir in os.listdir(openpose_dir):
        if dir == '.DS_Store': continue
        dir_path = osp.join(openpose_dir, dir)
        openpose_subdir = dir_path
        frame_subdir = osp.join(frame_dir, dir)
        render_subdir = osp.join(render_img_dir, dir)
        res_subdir = osp.join(res_dir, dir)
        ry_utils.renew_dir(res_subdir)

        paste_hand(frame_subdir, openpose_subdir, render_subdir, res_subdir)
    '''
        p = mp.Process(target=paste_hand, 
            args=(frame_subdir, openpose_subdir, render_subdir, res_subdir))
        process_list.append(p)
        p.start()
    
    for p in process_list:
        p.join()
    '''

        
if __name__ == '__main__':
    main()