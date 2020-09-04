import os
import os.path as osp
import sys
sys.path.append("mocap_utils")
import general_utils as g_utils
import renderer.opendr_renderer as od_render
import numpy as np
import cv2


class Visualizer(object):


    def __init__(self):
        pass

    
    def __draw_raw_bbox(self, img, bboxes):
        img = img.copy()
        for bbox in bboxes:
            img = od_render.draw_bbox(img, bbox, color=(0,255,0))
        return img


    def __draw_hand_bbox(self, img, hand_bbox_list):
        img = img.copy()
        for hand_bboxes in hand_bbox_list:
            for key in hand_bboxes:
                if key == 'left_hand':
                    img = od_render.draw_bbox(img, hand_bboxes[key], color=(255,0,0))
                else:
                    img = od_render.draw_bbox(img, hand_bboxes[key], color=(0,0,255))
        return img
    

    def __draw_arm_pose(self, img, body_pose_list):
        img = img.copy()
        for body_pose in body_pose_list:
            # left & right arm
            img = od_render.draw_keypoints(
                img, body_pose[5:8, :], radius=10, color=(255, 0, 0))
            img = od_render.draw_keypoints(
                img, body_pose[2:5, :], radius=10, color=(0, 0, 255))
        return img


    def visualize(self, 
        input_img, 
        hand_bbox_list = None, 
        body_pose_list = None,
        raw_hand_bboxes = None,
        cam = None,
        verts = None,
        faces = None,
    ):
        # init
        res_img = input_img.copy()

        # draw hand bbox
        if hand_bbox_list is not None:
            res_img = self.__draw_hand_bbox(res_img, hand_bbox_list)

        # draw 2D Pose
        if body_pose_list is not None:
            res_img = self.__draw_arm_pose(res_img, body_pose_list)

        # draw raw hand bboxes
        if raw_hand_bboxes is not None:
            raw_bbox_img = self.__draw_raw_bbox(input_img, raw_hand_bboxes)
            res_img = np.concatenate((res_img, raw_bbox_img), axis=1)
        
        return res_img