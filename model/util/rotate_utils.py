import torch
import numpy as np
import util.geometry_utils as gu
import cv2
import sys

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

pi = 3.14159265358979323846
def rotate_orient(orient, angle):
    rot_t = torch.from_numpy(orient.reshape(1, 3))
    rot_x = torch.Tensor((0, 0, -pi*angle/180)).view(1,3)
    rotmat_t = gu.angle_axis_to_rotation_matrix(rot_t).float()
    rotmat_x = gu.angle_axis_to_rotation_matrix(rot_x).float()
    rotmat_new = torch.matmul(rotmat_x, rotmat_t)
    rot_new = gu.rotation_matrix_to_angle_axis(rotmat_new[:, :3, :])
    rot = rot_new.numpy()[0]
    return rot


def rotate_joints_2d(joints, origin, angle):
    angle = -angle/180 * np.pi

    joints_x = joints[:, 0]
    joints_y = joints[:, 1]
    origin_x = origin[:, 0]
    origin_y = origin[:, 1]

    res_x = origin_x + np.cos(angle) * (joints_x-origin_x) - np.sin(angle)*(joints_y-origin_y)
    res_y = origin_y + np.sin(angle) * (joints_x-origin_x) + np.cos(angle)*(joints_y-origin_y)

    res_joints = np.concatenate( (res_x[:, None], res_y[:, None]), axis=1)
    return res_joints


def rotate_joints_3d(joints, angle):
    joints_t = torch.from_numpy(joints).float()
    rot_x = torch.Tensor((0, 0, -np.pi*angle/180)).view(1,3)
    rotmat_x = gu.angle_axis_to_rotation_matrix(rot_x).float()
    rotmat_x = rotmat_x[:, :3, :3]
    joints_rotate = torch.matmul(rotmat_x, joints_t)
    return joints_rotate.numpy()[0]