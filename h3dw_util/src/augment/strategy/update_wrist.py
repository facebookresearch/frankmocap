import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
import copy
from augment.sample import Sample
from utils.pose_prior import MaxMixturePrior
import torch
from utils import vis_utils
import smplx

def get_pose_prior_score(pose_prior_model, hand_wrist, hand_type='left'):
    if hand_type == 'left_hand':
        hand_wrist[:, 1] *= -1
        hand_wrist[:, 2] *= -1
    else:
        assert hand_type == 'right_hand'
    body_pose = torch.zeros((1, 69))
    body_pose[:, 20*3:21*3] = hand_wrist
    betas = torch.zeros((1, 10))
    pose_prior_score = pose_prior_model(body_pose, betas)
    return pose_prior_score.item()


def get_prior_model():
    pose_prior_model = MaxMixturePrior(
        prior_folder='data', num_gaussians=8, dtype=torch.float32)
    return pose_prior_model


def transfer_hand_wrist(body_pose, hand_wrist, hand_type, transfer_type="l2g"):
    root_dir = "/Users/rongyu/Documents/research/FAIR/workplace/"
    smplx_model_file = osp.join(root_dir, "data/models/smplx/SMPLX_NEUTRAL.pkl")
    smplx_model = smplx.create(smplx_model_file, model_type="smplx")
    
    if hand_type == 'left_hand':
        kinematic_map = vis_utils.get_kinematic_map(smplx_model, 20)
    else:
        assert hand_type == 'right_hand'
        kinematic_map = vis_utils.get_kinematic_map(smplx_model, 21)

    if transfer_type == "l2g":
        # local to global
        hand_wrist_local = hand_wrist.clone()
        hand_wrist_global = vis_utils.get_global_hand_rot(
            body_pose, hand_wrist_local, kinematic_map)
        return hand_wrist_global
    else:
        # global to local
        assert transfer_type == "g2l"
        hand_wrist_global = hand_wrist.clone()
        hand_wrist_local = vis_utils.get_local_hand_rot(
            body_pose, hand_wrist_global, kinematic_map)
        return hand_wrist_local
    

def apply(t_model):
    pose_prior_model = get_prior_model()

    for sample_id, sample in enumerate(t_model.all_samples):
        score = dict()
        for hand_type in ['left_hand', 'right_hand']:
            hand_wrist_h3dw_local = sample.pred_hand_info[hand_type]['wrist_rot_local'].copy()
            hand_wrist_h3dw_local = torch.from_numpy(hand_wrist_h3dw_local).float().view(1, 3)
            score[hand_type] = get_pose_prior_score(pose_prior_model, hand_wrist_h3dw_local, hand_type)
            sample.hand_valid[hand_type] = score[hand_type] < t_model.threshold
            sample.pose_prior_score[hand_type] = score[hand_type]