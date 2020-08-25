import os
import os.path as osp
import sys
sys.path.append("src/")
import ry_utils
import numpy as np
import cv2
from collections import defaultdict
import parallel_io as pio
from demo.compare.compare_mtc_hand import get_all_names, frame_to_video, get_hand_results, load_hand_results


def load_bbox_info(pkl_file):
    all_data = pio.load_pkl_single(pkl_file)
    bbox_info = dict()
    for img_name in all_data:
        bbox_info[img_name.replace(".png", ".jpg")] = all_data[img_name]['pred_bbox']
    return bbox_info


def crop_mtc_to_fairmocap(mtc_img, bbox, final_size):
    ratio = mtc_img.shape[0] / 1920
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    bbox = np.array([x1, y1, x2, y2]) * ratio
    x1, y1, x2, y2 = bbox.astype(np.int32)

    height, width = mtc_img.shape[:2]
    if x2-x1 > y2-y1:
        margin = ((x2-x1)-(y2-y1))//2
        y1 = max(0, y1-margin)
        y2 = min(height, y2+margin)
    else:
        margin = ((y2-y1)-(x2-x1))//2
        x1 = max(0, x1-margin)
        x2 = min(width, x2+margin)
    
    margin = int(max(y2-y1, x2-x1)*0.05)
    x1 = max(0, x1-margin)
    x2 = min(width, x2+margin)
    y1 = max(0, y1-margin)
    y2 = min(height, y2+margin)
    
    mtc_img = cv2.resize(mtc_img[y1:y2, x1:x2, :], (final_size, final_size))
    return mtc_img


def concat_result(all_names, h3dw_dir, column_ids, mtc_dir, bbox_info, res_dir, save_to_disk=False):
    if save_to_disk:
        ry_utils.renew_dir(res_dir)

    num_process = 0
    frames = defaultdict(list)
    all_imgs = dict()
    for name_id, name in enumerate(all_names):
        '''
        if name.find("body_only") < 0:
            continue
        num_process += 1
        if num_process > 10:
            break
        '''

        # load origin image and determine size
        img_path = osp.join(h3dw_dir, name)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        origin_img = img[:, :height, :]
        img_list = [origin_img,]

        for i in column_ids:
            h3dw_img = img[:, height*i:height*(i+1), :]
            img_list.append(h3dw_img)

        record = name.split('/')
        mtc_img_name = osp.join(record[0], record[1].split('_')[-1])
        mtc_img_path = osp.join(mtc_dir, mtc_img_name)
        mtc_img = cv2.imread(mtc_img_path)

        # crop mtc img
        final_size = h3dw_img.shape[0]
        bbox = bbox_info[name]
        mtc_img = crop_mtc_to_fairmocap(mtc_img, bbox, final_size)
        img_list.append(mtc_img)

        res_img = np.concatenate(img_list, axis=1)
        height, width = res_img.shape[:2]
        if height >= 1000:
            new_height, new_width = height//2, width//2
            res_img = cv2.resize(res_img, (new_width, new_height))

        seq_name = name.split('/')[0]
        frames[seq_name].append(res_img)

        img_key = name.split('/')[-1][:-4]
        all_imgs[img_key] = res_img

        if save_to_disk:
            res_subdir = osp.join(res_dir, seq_name)
            ry_utils.build_dir(res_subdir)
            ext = "." + name.split('.')[-1]
            res_img_path = osp.join(res_dir, name).replace(ext, ".jpg")
            cv2.imwrite(res_img_path, res_img)

        print(f"Processed {name_id}/{len(all_names)}")

    if save_to_disk:
        # save results to videos 
        for key in frames:
            video_path = osp.join(res_dir, f"{key}.mp4")
            frame_to_video(frames[key], video_path)
            print(f"{key} complete")
    
    return all_imgs


def get_body_results(save_to_disk = False):
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    exp_res_dir = "experiment/experiment_results/3d_hand/h3dw/demo_data/body_capture/"
    mtc_res_dir = osp.join(root_dir, exp_res_dir, "mtc_prediction")

    bbox_file = osp.join(root_dir, exp_res_dir, "prediction/body/smpl_pred.pkl")
    bbox_info = load_bbox_info(bbox_file)

    # h3dw_res_dir = "visualization/body_hand_merge/body_capture"
    h3dw_res_dir = "visualization/compare/body_capture_between_h3dw/average"
    column_ids = [1, 2, 3]
    all_names = get_all_names(h3dw_res_dir)

    res_dir = "visualization/compare/body_capture_mtc_body_only/single"
    all_imgs = concat_result(all_names, h3dw_res_dir, column_ids, mtc_res_dir, bbox_info, res_dir, save_to_disk)
    return all_imgs


def load_body_results():
    res_dir = "visualization/compare/body_capture_mtc_body_only/single"
    all_imgs = dict()
    for subdir, dirs, files in os.walk(res_dir):
        for file in files:
            if file.endswith((".jpg", ".png")):
                img_path = osp.join(subdir, file)
                img = cv2.imread(img_path)
                img_key = file.split('.')[0]
                all_imgs[img_key] = img
    return all_imgs


def flip_concat_img(input_img):
    res_img = np.zeros(input_img.shape)
    height, width = input_img.shape[:2]
    assert width % height == 0
    for i in range(width // height):
        res_img[:, i*height:(i+1)*height, :] = \
            np.fliplr(input_img[:, i*height:(i+1)*height, :])
    return res_img


def resize_img(body_img, hand_img):
    b_height, b_width = body_img.shape[:2]
    h_height, h_width = hand_img.shape[:2]
    h_height = int(b_width/h_width * h_height)
    h_width = b_width
    hand_img = cv2.resize(hand_img, (h_width, h_height))
    return hand_img


def merge_body_hand(hand_imgs, body_imgs, res_dir):
    frames = defaultdict(list)
    sorted_keys = sorted(body_imgs.keys())
    for img_id, img_key in enumerate(sorted_keys):
        body_img = body_imgs[img_key]

        # hand_img
        hand_img_list = list()
        render_hand_img_list = list()
        has_hand = True
        for hand_type in ["right_hand", "left_hand"]:
            hand_img_key = f"{img_key}_{hand_type}"
            if hand_img_key not in hand_imgs:
                has_hand = False
                break
            hand_img, render_hand_img = hand_imgs[hand_img_key]
            if hand_type == 'left_hand':
                hand_img = flip_concat_img(hand_img)
                render_hand_img = flip_concat_img(render_hand_img)
            hand_img_list.append(hand_img)
            render_hand_img_list.append(render_hand_img)

        if not has_hand:
            continue

        hand_img = np.concatenate(hand_img_list, axis=1)
        render_hand_img = np.concatenate(render_hand_img_list, axis=1)
        # resize hand img to body img
        hand_img = resize_img(body_img, hand_img)
        render_hand_img = resize_img(body_img, render_hand_img)

        res_img = np.concatenate((body_img, hand_img, render_hand_img), axis=0)
        seq_name = '_'.join(img_key.split('_')[:-1])
        res_img_path = osp.join(res_dir, seq_name, img_key+".jpg")
        ry_utils.make_subdir(res_img_path)
        cv2.imwrite(res_img_path, res_img)

        frames[seq_name].append(res_img)

        print(f"Processed {img_id}/{len(sorted_keys)}")

    for seq_name in frames:
        video_path = osp.join(res_dir, f"{seq_name}.mp4")
        frame_to_video(frames[seq_name], video_path)
        print(f"{seq_name} complete")


def main():
    hand_imgs = get_hand_results(save_to_disk = False)
    print("load hand images complete")

    body_imgs = get_body_results(save_to_disk = True)
    # body_imgs = load_body_results()
    print("load body images complete")

    res_dir = "visualization/compare/body_capture_mtc_body_hand/single"
    ry_utils.renew_dir(res_dir)
    merge_body_hand(hand_imgs, body_imgs, res_dir)
    
if __name__ == '__main__':
    main()
