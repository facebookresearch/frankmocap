import sys, os
import os.path as osp
# sys.path.append("/Users/rongyu/Documents/research/FAIR/workplace/data/amass/code/human_body_prior")
import torch
import numpy as np
'''
from human_body_prior.tools.omni_tools import copy2cpu as c2c
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from human_body_prior.body_model.body_model import BodyModel
import trimesh
from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.mesh.sphere import points_to_spheres
'''
import cv2
from utils.pose_prior import MaxMixturePrior
import parallel_io as pio


def get_pose_prior_score(pose_prior_model, hand_wrist, hand_type='left'):
    if hand_type == 'left':
        hand_wrist[:, 1] *= -1
        hand_wrist[:, 2] *= -1
    else:
        assert hand_type == 'right'
    body_pose = torch.zeros((1, 69))
    body_pose[:, 20*3:21*3] = hand_wrist
    betas = torch.zeros((1, 10))
    pose_prior_score = pose_prior_model(body_pose, betas)
    return pose_prior_score


def main():
    pose_prior_model = MaxMixturePrior(prior_folder='data',
                                num_gaussians=8,
                                dtype=torch.float32)

    '''
    for i in range(32):
        hand_wrist = torch.zeros((1,3)).float()
        hand_wrist[0, 0] = 2*np.pi/32 * i
        score_01 = get_pose_prior_score(pose_prior_model, hand_wrist, "left").item()

        hand_wrist = torch.zeros((1,3)).float()
        hand_wrist[0, 1] = 2*np.pi/32 * i
        score_02 = get_pose_prior_score(pose_prior_model, hand_wrist, "left").item()
 
        hand_wrist = torch.zeros((1,3)).float()
        hand_wrist[0, 2] = 2*np.pi/32 * i
        score_03 = get_pose_prior_score(pose_prior_model, hand_wrist, "left").item()

        print(f"{i}, {score_01:.2f}, {score_02:.2f}, {score_03:.2f}")
    '''

    all_prior_score = list()
    amass_dir = '/Users/rongyu/Documents/research/FAIR/workplace/data/amass/data/SSMsynced/SSM_synced'
    for subdir, dirs, files in os.walk(amass_dir):
        for file in files:
            if file.endswith(".npz"):
                npz_data_path = osp.join(subdir, file)
                data = np.load(npz_data_path)
                fId = 0 # frame id of the mocap sequence
                num_data = data['poses'].shape[0]
                for i in range(num_data):
                    pose_body = torch.Tensor(data['poses'][i:i+1, 3:66])
                    left_wrist = pose_body[:, 19*3:20*3]
                    left_score = get_pose_prior_score(pose_prior_model, left_wrist, "left")
                    all_prior_score.append(left_score)
                    right_wrist = pose_body[:, 20*3:21*3]
                    right_score = get_pose_prior_score(pose_prior_model, right_wrist, "right")
                    all_prior_score.append(right_score)
    pio.save_pkl_single("data/amass_prior_score.pkl", all_prior_score)
    all_prior_score = pio.load_pkl_single("data/amass_prior_score.pkl") 
    print(np.mean(all_prior_score), np.min(all_prior_score), np.max(all_prior_score))


if __name__ == '__main__':
    main()