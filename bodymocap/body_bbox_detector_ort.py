import argparse
import cv2
import numpy as np
from alfred.utils.timer import ATimer
from .utils.utils import ORTWrapper
import math
from operator import itemgetter
from alfred.vis.image.pose import vis_pose_result, vis_pose_by_joints
from alfred.utils.file_io import ImageSourceIter
from .utils.utils import normalize, extract_keypoints, connections_nms, group_keypoints


class LightweightedPoseDetector:

    def __init__(self, onnx_f='data/human-pose-estimation.onnx') -> None:
        self.onnx_model = ORTWrapper(onnx_f)

        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = 18
        self.img_mean = np.array([128, 128, 128]).astype(np.float32)
        self.img_scale = np.float32(1/256)

    def infer(self, img):
        height, width, _ = img.shape

        net_in_height = 256
        net_in_width = 288

        scale = min(net_in_height / height, net_in_width/width)

        scaled_img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # scaled_img = normalize(scaled_img, img_mean, img_scale)
        s_h, s_w, _ = scaled_img.shape
        in_img = np.ones([net_in_height, net_in_width, 3]
                         ).astype(np.uint8) * 128
        top = (net_in_height - s_h) // 2
        left = (net_in_width - s_w) // 2
        in_img[top: top + s_h, left: left + s_w] = scaled_img

        in_img = normalize(in_img, self.img_mean, self.img_scale, )
        inp_img = np.expand_dims(in_img.transpose((2, 0, 1)), axis=0)
        print(inp_img.shape)
        stages_output = self.onnx_model.infer(inp_img)
        # print(stages_output)
        heatmaps = stages_output['stage_1_output_1_heatmaps']
        pafs = stages_output['stage_1_output_0_pafs']

        heatmaps = heatmaps.squeeze(0)
        pafs = pafs.squeeze(0)
        return heatmaps, pafs, scale, [top, left]

    def run_one_img(self, img):
        heatmaps, pafs, scale, pad = self.infer(img)

        all_keypoints_by_type = []
        total_keypoints_num = 0
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(
            all_keypoints_by_type, pafs)

        # h, w
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale

        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones(
                (self.num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1])
            # print(pose_entries[n][18])
            current_poses.append(pose_keypoints)

        if len(current_poses) > 0:
            return np.stack(current_poses)
        return current_poses

    def get_enlarged_boxes_from_poses(self, poses, img_h, img_w):
        current_bbox = []
        for pose in poses:
            found_keypoints = np.zeros(
                (np.count_nonzero(pose[:, 0] != -1), 2), dtype=np.int32)
            found_kpt_id = 0
            for kpt_id in range(self.num_keypoints):
                if pose[kpt_id, 0] == -1:
                    continue
                found_keypoints[found_kpt_id] = pose[kpt_id]
                found_kpt_id += 1
            bb = cv2.boundingRect(found_keypoints)
            current_bbox.append(bb)
        for i, bbox in enumerate(current_bbox):
            x, y, w, h = bbox
            margin = 0.05
            x_margin = int(w * margin)
            y_margin = int(h * margin)
            x0 = max(x-x_margin, 0)
            y0 = max(y-y_margin, 0)
            x1 = min(x+w+x_margin, img_w)
            y1 = min(y+h+y_margin, img_h)
            current_bbox[i] = np.array((x0, y0, x1-x0, y1-y0)).astype(np.int32)
        return current_bbox
