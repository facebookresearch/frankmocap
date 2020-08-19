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
        score = sample.openpose_score[hand_type]
        if score > select_thresh and score > max_score:
            sample_select = sample
            max_score = score
    return sample_select, max_score


def updated_hand_pose(sample, sample_select, hand_type):
    update_hand_info = sample.pred_hand_info[hand_type]
    select_hand_info = sample_select.pred_hand_info[hand_type]
    update_hand_info['pred_hand_pose'] = select_hand_info['pred_hand_pose']
    update_hand_info['wrist_rot_local'] = select_hand_info['wrist_rot_local']
    update_hand_info['wrist_rot_body_global'] = select_hand_info['wrist_rot_body_global']
    '''
    sample.pred_hand_info[hand_type]['pred_hand_pose'] = \
        sample_select.pred_hand_info[hand_type]['pred_hand_pose'].copy()
    '''


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
            openpose_score = sample.openpose_score[hand_type]
            # only update when openpose_score is lower than thresh
            if openpose_score < update_thresh:
                sample_select, max_score = get_best_sample(memory_bank, hand_type, select_thresh)
                if max_score > 0:
                    updated_hand_pose(sample, sample_select, hand_type)
                    sample.update_hand(hand_type, sample_select.render_img_path[hand_type])
                    sample.updated = True
                    samples_new.append(sample)
                    '''
                    print(hand_type, sample.img_name, sample_select.img_name)
                    print(openpose_score, max_score)
                    print("========================================================")
                    '''
            else:
                sample.updated = False

        # push current sample to memory_bank
        if len(memory_bank) < t_model.memory_size:
            memory_bank.append(sample_origin)
        else:
            memory_bank.pop(0)
            memory_bank.append(sample_origin)
    
    return samples_new