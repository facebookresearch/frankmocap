import sys
assert sys.version_info > (3, 0)
sys.path.append('src/')
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import random
import pdb
import torch
from utils.render_utils import project_joints


body_connMat = np.array([0, 1, 0, 3, 3, 4, 4, 5, 0, 9, 9, 10, 10, 11, 0, 2, 2, 6, 6, 7, 7, 8, 2, 12, 12, 13, 13, 14, 1, 15, 15, 16, 1, 17, 17, 18]).reshape(-1, 2)
hand_connMat = np.array([0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 0, 9, 9, 10, 10, 11, 11, 12, 0, 13, 13, 14, 14, 15, 15, 16, 0, 17, 17, 18, 18, 19, 19, 20]).reshape(-1, 2)

top_down_camera_ids = [1, 2, 4, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21, 26, 28]

def project2D(joints, calib, imgwh=None, applyDistort=True):
    """
    Input:
    joints: N * 3 numpy array.
    calib: a dict containing 'R', 'K', 't', 'distCoef' (numpy array)
    Output:
    pt: 2 * N numpy array
    inside_img: (N, ) numpy array (bool)
    """

    # print(joints.shape)

    x = np.dot(calib['R'], joints.T) + calib['t']
    xp = x[:2, :] / x[2, :]

    if applyDistort:
        X2 = xp[0, :] * xp[0, :]
        Y2 = xp[1, :] * xp[1, :]
        XY = X2 * Y2
        R2 = X2 + Y2
        R4 = R2 * R2
        R6 = R4 * R2

        dc = calib['distCoef']
        radial = 1.0 + dc[0] * R2 + dc[1] * R4 + dc[4] * R6
        tan_x = 2.0 * dc[2] * XY + dc[3] * (R2 + 2.0 * X2)
        tan_y = 2.0 * dc[3] * XY + dc[2] * (R2 + 2.0 * Y2)

        # xp = [radial;radial].*xp(1:2,:) + [tangential_x; tangential_y]
        xp[0, :] = radial * xp[0, :] + tan_x
        xp[1, :] = radial * xp[1, :] + tan_y

    # pt = bsxfun(@plus, cam.K(1:2,1:2)*xp, cam.K(1:2,3))';
    # calib['K'][1,1] = calib['K'][0,0]


    '''
    cam = np.array([0.7/120, 0.1, 0.1])
    body_joints_2d_norm = project_joints(x.T, cam)
    body_joints_2d_norm = (body_joints_2d_norm+1.0) * 0.5 * 1080
    return body_joints_2d_norm
    '''
    pt = np.dot(calib['K'][:2, :2], xp) + calib['K'][:2, 2].reshape((2, 1))
    return pt.T
    
 
def project2D_with_rotation(joints, calib, imgwh=None, applyDistort=True):
    """
    Input:
    joints: N * 3 numpy array.
    calib: a dict containing 'R', 'K', 't', 'distCoef' (numpy array)
    Output:
    pt: 2 * N numpy array
    inside_img: (N, ) numpy array (bool)
    """
    x = np.dot(calib['R'], joints.T) + calib['t']
    x_r = ru.rotate_joints_3d(x.copy(), 90)

    xp = x[:2, :] / x[2, :]
    xp_t = xp.copy()
    xp_r = x_r[:2, :] / x[2, :]

    if applyDistort:
        X2 = xp[0, :] * xp[0, :]
        Y2 = xp[1, :] * xp[1, :]
        XY = X2 * Y2
        R2 = X2 + Y2
        R4 = R2 * R2
        R6 = R4 * R2

        dc = calib['distCoef']
        radial = 1.0 + dc[0] * R2 + dc[1] * R4 + dc[4] * R6
        tan_x = 2.0 * dc[2] * XY + dc[3] * (R2 + 2.0 * X2)
        tan_y = 2.0 * dc[3] * XY + dc[2] * (R2 + 2.0 * Y2)

        # xp = [radial;radial].*xp(1:2,:) + [tangential_x; tangential_y]
        xp[0, :] = radial * xp[0, :] + tan_x
        xp[1, :] = radial * xp[1, :] + tan_y

    # pt = bsxfun(@plus, cam.K(1:2,1:2)*xp, cam.K(1:2,3))';
    pt = np.dot(calib['K'][:2, :2], xp) + calib['K'][:2, 2].reshape((2, 1))
    pt_t = np.dot(calib['K'][:2, :2], xp_t) + calib['K'][:2, 2].reshape((2, 1))
    pt_r = np.dot(calib['K'][:2, :2], xp_r) + calib['K'][:2, 2].reshape((2, 1))
    
    return pt.T, pt_r.T