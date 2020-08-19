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
            if file.endswith((".png", "jpg")):
                img_names.append(osp.join(subdir, file).replace(in_dir, '')[1:])
    all_names = sorted(img_names)
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


def concat_result(all_names, h3dw_dir, minimal_hand_dir, res_dir):
    frames = defaultdict(list)
    for name in all_names:
        # load origin image and determine size
        img_path = osp.join(h3dw_dir, name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        origin_img = img[:, :height, :]
        img_list = [origin_img,]

        h3dw_img = img[:, height:height*2, :]
        img_list.append(h3dw_img)

        minimal_hand_img = osp.join(minimal_hand_dir, name)
        if not osp.exists(minimal_hand_img):
            continue
            
        mh_img = cv2.imread(osp.join(minimal_hand_dir, name))
        mh_height, mh_weight = mh_img.shape[:2]
        mh_render_img = cv2.resize(mh_img[:, mh_height:mh_height*2, :], (height, height))
        img_list.append(mh_render_img)

        res_img = np.concatenate(img_list, axis=1)

        
        seq_name = name.split('/')[0]
        res_img_path = osp.join(res_dir, name)
        ry_utils.make_subdir(res_img_path)
        cv2.imwrite(res_img_path, res_img)

        # store single frames
        seq_name = res_img_path.split('/')[-2]
        frames[seq_name].append(res_img)

    '''
    # save results to images 
    for key in frames:
        video_path = osp.join(res_dir, f"{key}.mp4")
        frame_to_video(frames[key], video_path)
        print(f"{key} complete")
    '''


def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    exp_res_dir = "experiment/experiment_results/3d_hand/h3dw/demo_data/body_capture/"
    h3dw_res_dir = osp.join(root_dir, exp_res_dir, "prediction/hand/images/right_hand")
    minimal_hand_dir = osp.join(root_dir, exp_res_dir, "hand_prediction/minimal_hand_result")

    '''
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    exp_res_dir = "data/coco/prediction/hand"
    h3dw_res_dir = osp.join(root_dir, exp_res_dir, "images/val")
    minimal_hand_dir = osp.join(root_dir, exp_res_dir, "minimal_hand/val")
    '''

    all_names = get_all_names(h3dw_res_dir)

    res_dir = "visualization/compare/coco_minimal_hand"
    ry_utils.renew_dir(res_dir)
    concat_result(all_names, h3dw_res_dir, minimal_hand_dir, res_dir)
    
if __name__ == '__main__':
    main()
