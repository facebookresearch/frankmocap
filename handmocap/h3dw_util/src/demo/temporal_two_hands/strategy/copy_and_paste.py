import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
import copy
from temporal.sample import Sample


def get_best_sample(memory_bank, hand_type, select_thresh):
    sample_select = None
    max_score = -1
    for sample in memory_bank:
        # score = sample.openpose_score[hand_type]
        score = sample.pred_hand_info[hand_type]['openpose_score']
        if score > select_thresh and score > max_score:
            sample_select = sample
            max_score = score
    return sample_select, max_score


def updated_hand_pose(sample, sample_select, hand_type):
    sample.pred_hand_info[hand_type]['pred_shape'] = sample_select.pred_hand_info[hand_type]['pred_shape']
    sample.pred_hand_info[hand_type]['pred_pose'] = sample_select.pred_hand_info[hand_type]['pred_pose']


def apply(t_model):
    # in "coyp_and_paste" strategy, memory_bank_left == memory_bank_right
    t_model.memory_bank_right = t_model.memory_bank_left 
    memory_bank = t_model.memory_bank_left
    update_thresh = t_model.config.strategy_params['copy_and_paste']['update_thresh']
    select_thresh = t_model.config.strategy_params['copy_and_paste']['select_thresh']
    assert select_thresh >= update_thresh

    samples_new = list()
    for sample_id, sample_origin in enumerate(t_model.samples_origin):
        # update sample            
        sample = copy.deepcopy(sample_origin)
        for hand_type in ['left_hand', 'right_hand']:
            openpose_score = sample.pred_hand_info[hand_type]['openpose_score']
            # only update when openpose_score is lower than thresh
            if openpose_score < update_thresh and openpose_score > 0.0: # if openpose score == 0.0 then means it does not have openpose prediction
                sample_select, max_score = get_best_sample(memory_bank, hand_type, select_thresh)
                if max_score > 0:
                    updated_hand_pose(sample, sample_select, hand_type)
                    sample.update_hand(hand_type)
                    '''
                    print(hand_type, sample.img_name, sample_select.img_name)
                    print(openpose_score, max_score)
                    print(sample.updated)
                    print("========================================================")
                    '''
        samples_new.append(sample)
        # push current sample to memory_bank
        if len(memory_bank) < t_model.memory_size:
            memory_bank.append(sample_origin)
        else:
            memory_bank.pop(0)
            memory_bank.append(sample_origin)
    
    return samples_new