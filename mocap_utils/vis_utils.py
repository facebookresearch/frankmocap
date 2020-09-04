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
                img, body_pose[6:8, :], radius=10, color=(255, 0, 0))
            img = od_render.draw_keypoints(
                img, body_pose[3:5, :], radius=10, color=(0, 0, 255))
        return img
    

    def __render_pred_verts(self, img_original, pred_meshes_list):
        res_img = img_original.copy()
        for pred_output in pred_meshes_list:
            for hand_type in pred_output:
                if pred_output[hand_type] is not None:
                    pred = pred_output[hand_type]
                    verts = pred['pred_vertices_origin']
                    faces = pred['faces']
                    cam = pred['cam']
                    bbox_scale_ratio = pred['bbox_scale_ratio']
                    bbox_top_left = pred['bbox_top_left']
                    img_cropped = pred['img_cropped']
                    if hand_type == 'left_hand':
                        img_cropped = np.fliplr(img_cropped)

                    res_img = od_render.render_to_origin_img(
                        cam, verts, faces,
                        bg_img = res_img,
                        bbox_scale = bbox_scale_ratio, 
                        bbox_top_left = bbox_top_left)
        return res_img


    def visualize(self, 
        input_img, 
        hand_bbox_list = None, 
        body_pose_list = None,
        raw_hand_bboxes = None,
        pred_meshes_list = None,
        vis_raw_hand_bbox = True,
        vis_body_pose = True,
        vis_hand_bbox = True,
    ):
        # print("vis_raw_hand_bbox", vis_raw_hand_bbox)
        # print("vis_body_pose", vis_body_pose)
        # print("vis_hand_bbox", vis_hand_bbox)

        # init
        res_img = input_img.copy()

        # draw raw hand bboxes
        if raw_hand_bboxes is not None and vis_raw_hand_bbox:
            res_img = self.__draw_raw_bbox(input_img, raw_hand_bboxes)
            # res_img = np.concatenate((res_img, raw_bbox_img), axis=1)

        # draw 2D Pose
        if body_pose_list is not None and vis_body_pose:
            res_img = self.__draw_arm_pose(res_img, body_pose_list)

        # draw hand bbox
        if hand_bbox_list is not None and vis_hand_bbox:
            hand_bbox_img = self.__draw_hand_bbox(input_img, hand_bbox_list)
            if vis_raw_hand_bbox or vis_body_pose:
                res_img = np.concatenate((res_img, hand_bbox_img), axis=1)
            else:
                res_img = hand_bbox_img

        # render predicted hands
        if pred_meshes_list is not None:
            rend_img = self.__render_pred_verts(input_img, pred_meshes_list)
            res_img = np.concatenate((res_img, rend_img), axis=1)
        
        return res_img