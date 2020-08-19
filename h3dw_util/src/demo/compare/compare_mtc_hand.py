import os
import os.path as osp
import sys
import ry_utils
import numpy as np
import cv2
from collections import defaultdict

def get_all_names(in_dir):
    img_names = list()
    for subdir, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith((".png", ".jpg")):
                img_names.append(osp.join(subdir, file).replace(in_dir, '')[1:])
    all_names = sorted(img_names)
    return all_names


def frame_to_video(all_frames, video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    sample_img = all_frames[0]
    height, width = sample_img.shape[:2]
    video_out = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
    for frame in all_frames:
        frame = np.uint8(frame)
        video_out.write(frame)
    video_out.release()


def concat_result(all_names, h3dw_dir, mtc_hand_dir, res_dir, save_to_disk = False):
    if save_to_disk:
        ry_utils.renew_dir(res_dir)

    frames = defaultdict(list)
    all_imgs = dict()
    for name_id, name in enumerate(all_names):
        '''
        if name.find("body_only")<0:
            continue
        '''

        # load origin image and determine size
        img_path = osp.join(h3dw_dir, name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        origin_img = img[:, :height, :]
        img_list = [origin_img,]

        h3dw_img = img[:, height:height*2, :]
        img_list.append(h3dw_img)

        mh_img = cv2.imread(osp.join(mtc_hand_dir, name))
        mh_render_img = cv2.resize(mh_img, (height, height))
        img_list.append(mh_render_img)

        res_img = np.concatenate(img_list, axis=1)

        record = name.split('/')
        ext = "." + record[-1].split('.')[-1]
        img_name = record[-1].replace(ext, ".jpg")
        seq_name = f"{record[-2]}_{record[-3]}"
        img_key = record[-1][:-4]

        frames[seq_name].append(res_img)
        all_imgs[img_key] = (res_img, img[:, height*2:, :])

        if save_to_disk:
            res_img_path = osp.join(res_dir, seq_name, img_name)
            res_subdir = '/'.join(res_img_path.split('/')[:-1])
            ry_utils.build_dir(res_subdir)
            cv2.imwrite(res_img_path, res_img)

        print(f"{name_id}/{len(all_names)}")


    # save results to images 
    if save_to_disk:
        for key in frames:
            video_path = osp.join(res_dir, f"{key}.mp4")
            frame_to_video(frames[key], video_path)
            print(f"{key} complete")
    
    return all_imgs


def get_hand_results(save_to_disk=False):
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    exp_res_dir = "experiment/experiment_results/3d_hand/h3dw/demo_data/body_capture/"
    h3dw_res_dir = osp.join(root_dir, exp_res_dir, "prediction/hand/images")
    mtc_hand_dir = osp.join(root_dir, exp_res_dir, "hand_prediction/mtc_hand_result")

    all_names = get_all_names(h3dw_res_dir)

    res_dir = "visualization/compare/body_capture_mtc_hand_only"
    all_imgs = concat_result(all_names, h3dw_res_dir, mtc_hand_dir, res_dir, save_to_disk)
    return all_imgs


def load_hand_results():
    res_dir = "visualization/compare/body_capture_mtc_hand_only"

    all_imgs = dict()
    num_img = 0
    for subdir, dirs, files in os.walk(res_dir):
        for file in files:
            if file.endswith((".jpg", ".png")):
                img_path = osp.join(subdir, file)
                img = cv2.imread(img_path)
                img_key = file.split('.')[0]
                all_imgs[img_key] = img
                num_img += 1
    return all_imgs


def main():
    # get_hand_results(save_to_disk = True)
    get_hand_results(save_to_disk = False)
    # load_hand_results()
    

if __name__ == '__main__':
    main()
