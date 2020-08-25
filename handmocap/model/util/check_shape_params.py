import os, sys, shutil
import os.path as osp
import parallel_io as pio
import numpy as np


def main():
    pkl_file = "evaluate_results/pred_results_body_capture.pkl"
    pred_results = pio.load_pkl_single(pkl_file)
    for data_id, data in enumerate(pred_results):
        img_name = data['img_name']
        # shape_param = np.average(np.abs(data['pred_shape_params']))
        shape_param = np.linalg.norm(data['pred_shape_params'])
        print(img_name, shape_param)
        # print(img_name)
        # key = 'body_object_00027_left_hand'
        # if img_name.find(key)>=0:


if __name__ == '__main__':
    main()