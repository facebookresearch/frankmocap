# Copyright (c) Facebook, Inc. and its affiliates.

import cv2
import numpy as np

def draw_keypoints(image, kps, color=(0,0,255), radius=5, check_exist=False):
    # recover color 
    if color == 'red':
        color = (0, 0, 255)
    elif color == 'green':
        color = (0, 255, 0)
    elif color == 'blue':
        color = (255, 0, 0)
    else:
        assert isinstance(color, tuple) and len(color) == 3

    # draw keypoints
    res_img = image.copy()
    for i in range(kps.shape[0]):
        x, y = kps[i][:2].astype(np.int32)
        if check_exist:
            score = kps[i][2]
        else:
            score = 1.0
        # print(i, score)
        if score > 0.0:
            cv2.circle(res_img, (x,y), radius=radius, color=color, thickness=-1)
    return res_img.astype(np.uint8)


def draw_bbox(image, bbox, color=(0,0,255), thickness=3):
    x0, y0 = int(bbox[0]), int(bbox[1])
    x1, y1 = int(bbox[2]), int(bbox[3])
    res_img = cv2.rectangle(image.copy(), (x0,y0), (x1,y1), color=color, thickness=thickness)
    return res_img.astype(np.uint8)



def draw_raw_bbox(img, bboxes):
    img = img.copy()
    for bbox in bboxes:
        x0, y0, w, h = bbox
        bbox_xyxy = (x0, y0, x0+w, y0+h)
        img = draw_bbox(img, bbox_xyxy)
    return img


def draw_body_bbox(img, body_bbox_list):
    img = img.copy()
    for body_bbox in body_bbox_list:
        if body_bbox is not None:
            x0, y0, w, h = body_bbox
            img = draw_bbox(img, (x0, y0, x0+w, y0+h))
    return img


def draw_arm_pose(img, body_pose_list):
    img = img.copy()
    for body_pose in body_pose_list:
        # left & right arm
        img = draw_keypoints(
            img, body_pose[6:8, :], radius=10, color=(255, 0, 0))
        img = draw_keypoints(
            img, body_pose[3:5, :], radius=10, color=(0, 0, 255))
    return img


def draw_hand_bbox(img, hand_bbox_list):
    img = img.copy()
    for hand_bboxes in hand_bbox_list:
        if hand_bboxes is not None:
            for key in hand_bboxes:
                bbox = hand_bboxes[key]
                if bbox is not None:
                    x0, y0, w, h = bbox
                    bbox_new = (x0, y0, x0+w, y0+h)
                    color = (255, 0, 0) if key == 'left_hand' else (0, 255, 0)
                    img = draw_bbox(img, bbox_new, color=color)
    return img
