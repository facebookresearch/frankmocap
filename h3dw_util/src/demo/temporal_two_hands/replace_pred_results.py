import os, sys, shutil
import os.path as osp
import ry_utils
import parallel_io as pio
import numpy as np

def main():
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/data/demo_data/"
    src_file = osp.join(root_dir, 
        "youtube/augment_bbox/prediction/h3dw/other_prediction/no_shape_prob-0.7_1e-3_epoch-140/pred_results_youtube.pkl")
    dst_file = osp.join(root_dir, "youtube/augment_bbox/prediction/h3dw/pred_results_youtube.pkl")

    dst_seq = "lecture_01_01"

    all_data = pio.load_pkl_single(src_file)
    src_data = dict()
    for data in all_data:
        img_name = data['img_name']
        seq_name = img_name.split('/')[-2]
        if seq_name == dst_seq:
            src_data[img_name] = dict(
                pred_shape_params = np.zeros(10,),
                pred_pose_params =  data['pred_pose_params']
            )
    
    dst_data = pio.load_pkl_single(dst_file)
    for data in dst_data:
        img_name = data['img_name']
        seq_name = img_name.split('/')[-2]
        if seq_name == dst_seq:
            data['pred_shape_params'] = src_data[img_name]['pred_shape_params']
            data['pred_pose_params'] = src_data[img_name]['pred_pose_params']
    
    pio.save_pkl_single(dst_file, dst_data)

if __name__ == '__main__':
    main()