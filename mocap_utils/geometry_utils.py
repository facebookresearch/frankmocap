# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
# sys.path.append("/")
import numpy as np
import torch
from torch.nn import functional as F
import cv2
import numpy.matlib as npm
import mocap_utils.geometry_utils_torch as gut


def flip_hand_pose(pose):
    pose = pose.copy()
    if len(pose.shape) == 1:
        pose = pose.reshape(-1, 3)
        pose[:, 1] *= -1
        pose[:, 2] *= -1
        return pose.reshape(-1,)
    else:
        assert len(pose.shape) == 2
        pose[:, 1] *= -1
        pose[:, 2] *= -1
        return pose


def flip_hand_joints_3d(joints_3d):
    assert joints_3d.shape[1] == 3
    assert len(joints_3d.shape) == 2
    rot_mat = np.diag([-1, 1, 1])
    return np.matmul(rot_mat, joints_3d.T).T


def __quaternion_to_angle_axis_torch(quat):
    quat = quat.clone()
    if quat.dim() == 1:
        assert quat.size(0) == 4
        quat = quat.view(1, 4)
        angle_axis = gut.quaternion_to_angle_axis(quat)[0]
    elif quat.dim() == 2:
        assert quat.size(1) == 4
        angle_axis = gut.quaternion_to_angle_axis(quat)
    else:
        assert quat.dim() == 3
        dim0 = quat.size(0)
        dim1 = quat.size(1)
        assert quat.size(2) == 4
        quat = quat.view(dim0*dim1, 4)
        angle_axis = gut.quaternion_to_angle_axis(quat)
        angle_axis = angle_axis.view(dim0, dim1, 3)
    return angle_axis


def quaternion_to_angle_axis(quaternion):
    quat = quaternion
    if isinstance(quat, torch.Tensor):
        return __quaternion_to_angle_axis_torch(quaternion)
    else:
        assert isinstance(quat, np.ndarray)
        quat_torch = torch.from_numpy(quat)
        angle_axis_torch = __quaternion_to_angle_axis_torch(quat_torch)
        return angle_axis_torch.numpy()


def __angle_axis_to_quaternion_torch(aa):
    aa = aa.clone()
    if aa.dim() == 1:
        assert aa.size(0) == 3 
        aa = aa.view(1, 3)
        quat = gut.angle_axis_to_quaternion(aa)[0]
    elif aa.dim() == 2:
        assert aa.size(1) == 3
        quat = gut.angle_axis_to_quaternion(aa)
    else:
        assert aa.dim() == 3
        dim0 = aa.size(0)
        dim1 = aa.size(1)
        assert aa.size(2) == 3
        aa = aa.view(dim0*dim1, 3)
        quat = gut.angle_axis_to_quaternion(aa)
        quat = quat.view(dim0, dim1, 4)
    return quat


def angle_axis_to_quaternion(angle_axis):
    aa = angle_axis
    if isinstance(aa, torch.Tensor):
        return __angle_axis_to_quaternion_torch(aa)
    else:
        assert isinstance(aa, np.ndarray)
        aa_torch = torch.from_numpy(aa)
        quat_torch = __angle_axis_to_quaternion_torch(aa_torch)
        return quat_torch.numpy()


def __angle_axis_to_rotation_matrix_torch(aa):
    aa = aa.clone()
    if aa.dim() == 1:
        assert aa.size(0) ==3 
        aa = aa.view(1, 3)
        rotmat = gut.angle_axis_to_rotation_matrix(aa)[0][:3, :3]
    elif aa.dim() == 2:
        assert aa.size(1) == 3
        rotmat = gut.angle_axis_to_rotation_matrix(aa)[:, :3, :3]
    else:
        assert aa.dim() == 3
        dim0 = aa.size(0)
        dim1 = aa.size(1)
        assert aa.size(2) == 3
        aa = aa.view(dim0*dim1, 3)
        rotmat = gut.angle_axis_to_rotation_matrix(aa)
        rotmat = rotmat.view(dim0, dim1, 4, 4)
        rotmat = rotmat[:, :, :3, :3]
    return rotmat


def angle_axis_to_rotation_matrix(angle_axis):
    aa = angle_axis
    if isinstance(aa, torch.Tensor):
        return __angle_axis_to_rotation_matrix_torch(aa)
    else:
        assert isinstance(aa, np.ndarray)
        aa_torch = torch.from_numpy(aa)
        rotmat_torch = __angle_axis_to_rotation_matrix_torch(aa_torch)
        return rotmat_torch.numpy()


def __rotation_matrix_to_angle_axis_torch(rotmat):
    rotmat = rotmat.clone()
    if rotmat.dim() == 2:
        assert rotmat.size(0) == 3
        assert rotmat.size(1) == 3
        rotmat0 = torch.zeros((1, 3, 4))
        rotmat0[0, :, :3] = rotmat
        rotmat0[:, 2, 3] = 1.0
        aa = gut.rotation_matrix_to_angle_axis(rotmat0)[0]
    elif rotmat.dim() == 3:
        dim0 = rotmat.size(0)
        assert rotmat.size(1) == 3
        assert rotmat.size(2) == 3
        rotmat0 = torch.zeros((dim0, 3, 4))
        rotmat0[:, :, :3] = rotmat
        rotmat0[:, 2, 3] = 1.0
        aa = gut.rotation_matrix_to_angle_axis(rotmat0)
    else:
        assert rotmat.dim() == 4
        dim0 = rotmat.size(0)
        dim1 = rotmat.size(1)
        assert rotmat.size(2) == 3
        assert rotmat.size(3) == 3
        rotmat0 = torch.zeros((dim0*dim1, 3, 4))
        rotmat0[:, :, :3] = rotmat.view(dim0*dim1, 3, 3)
        rotmat0[:, 2, 3] = 1.0
        aa = gut.rotation_matrix_to_angle_axis(rotmat0)
        aa = aa.view(dim0, dim1, 3)
    return aa


def rotation_matrix_to_angle_axis(rotmat):
    if isinstance(rotmat, torch.Tensor):
        return __rotation_matrix_to_angle_axis_torch(rotmat)
    else:
        assert isinstance(rotmat, np.ndarray)
        rotmat_torch = torch.from_numpy(rotmat)
        aa_torch = __rotation_matrix_to_angle_axis_torch(rotmat_torch)
        return aa_torch.numpy()
    

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    assert isinstance(x, torch.Tensor), "Current version only supports torch.tensor"

    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def angle_axis_to_rot6d(aa):
    assert aa.dim() == 2
    assert aa.size(1) == 3
    bs = aa.size(0)

    rotmat = angle_axis_to_rotation_matrix(aa)
    rot6d = rotmat[:, :3, :2]
    return rot6d