import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
import ry_utils
import parallel_io as pio
import copy
import multiprocessing as mp

from augment.sample import Sample
from augment.data_loader import load_all_samples
from visualize import visualize_body, visualize_hand
import config
from augment.augment_model import AugmentModel


def main():
    data_file = "/Users/rongyu/Documents/research/FAIR/workplace/data/coco/prediction/hand_info.pkl"
    all_data = pio.load_pkl_single(data_file)
    valid_hand = 0
    for data in all_data.values():
        has_valid_hand = False
        for hand_type in ['left_hand', 'right_hand']:
            if data[f'{hand_type}_valid']:
                has_valid_hand = True
        if has_valid_hand:
            valid_hand += 1
    print(valid_hand, len(all_data))

    
if __name__ == '__main__':
    main()