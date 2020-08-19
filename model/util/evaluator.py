from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('.')
import shutil
import os.path as osp
import numpy as np
import pickle
import copy
import cv2
import time
import parallel_io as pio
import ry_utils as ry_utils
import util.eval_utils as eval_utils
import util.metric_utils as metric_utils
from util.vis_util import render_mesh_to_image
import multiprocessing as mp
import pdb


class Evaluator(object):

    def __init__(self, data_list, model):
        self.data_list = copy.deepcopy(data_list)
        self.right_hand_faces_holistic = model.right_hand_faces_holistic
        self.right_hand_faces_local = model.right_hand_faces_local
        self.inputSize = model.inputSize
        self.mean_params = model.mean_params[0].cpu().detach().numpy().squeeze()
        self.pred_results = list()

    def clear(self):
        self.pred_results = list()

    def update(self, data_idxs, scale_ratios, ori_img_sizes, pred_results):
        for i, data_idx in enumerate(data_idxs):
            single_data = dict(
                data_idx = data_idx,
                cam = pred_results['cams'][i],
                pred_shape_params = pred_results['pred_shape_params'][i],
                pred_pose_params = pred_results['pred_pose_params'][i],
                pred_verts = pred_results['pred_verts'][i].astype(np.float16),
                gt_verts = pred_results['gt_verts'][i].astype(np.float16),
                pred_verts_multi_view = [verts[i].astype(np.float16) \
                    for verts in pred_results['pred_verts_multi_view']],
                pred_joints_3d = pred_results['pred_joints_3d'][i],
                gt_joints_2d = pred_results['gt_joints_2d'][i],
                pred_joints_2d = pred_results['pred_joints_2d'][i],
                img_name = self.data_list[data_idx]['image_name'],
                img_path = self.data_list[data_idx]['image_path'],
                ori_img_size = ori_img_sizes[i]
            )

            # pve
            pred_verts = pred_results['pred_verts'][i]
            gt_verts = pred_results['gt_verts'][i]
            mano_weight = pred_results['mano_params_weight'][i]
            single_data['pve'] = np.average(
                np.linalg.norm( (pred_verts-gt_verts)*mano_weight, axis=1))

            # mpjpe for mano joints
            mano_joints1 = pred_results['gt_joints_3d_mano'][i]
            mano_joints2 = pred_results['pred_joints_3d'][i]
            single_data['mpjpe_mano'] = np.average(
                np.linalg.norm( (mano_joints1-mano_joints2)*mano_weight, axis=1))

            # mpjpe for 3d joints
            joints_3d_1 = pred_results['gt_joints_3d'][i]
            joints_3d_2 = mano_joints2
            joint_weights = pred_results['joints_3d_weight'][i]
            single_data['mpjpe_3d'] = np.average(
                np.linalg.norm( (joints_3d_1-joints_3d_2)*joint_weights, axis=1))
            
            pck = metric_utils.get_single_joints_error(joints_3d_1, joints_3d_2, joint_weights)
            scale_ratio = scale_ratios[i]
            pck = np.array(pck) / scale_ratio
            # pck = np.array(pck)
            single_data['pck_3d'] = pck

            joints_2d_1 = single_data['gt_joints_2d']
            joints_2d_2 = single_data['pred_joints_2d']
            pck_2d = metric_utils.get_single_joints_error_2d(joints_2d_1, joints_2d_2)
            scale_ratio = single_data['ori_img_size'] / 2
            '''
            if single_data['img_name'] == 'val/000648952_02_l.jpg':
                print(joints_2d_1)
                print(joints_2d_2)
            '''
            # print(np.array(pck_2d))
            # sys.exit(0)
            pck_2d = np.array(pck_2d) * scale_ratio
            single_data['pck_2d'] = pck_2d
            # print(pck_2d)
            # print('====================================')
            
            # update pred_results
            self.pred_results.append(single_data)


    def remove_redunc(self):
        new_pred_results = list()
        img_id_set = set()
        for data in self.pred_results:
            data_idx = data['data_idx']
            img_id = self.data_list[data_idx]['image_path']
            if img_id not in img_id_set:
                new_pred_results.append(data)
                img_id_set.add(img_id)
        self.pred_results = new_pred_results
        print("Number of test data:", len(self.pred_results))


    @property
    def pve(self):
        res = np.average([data['pve']
                          for data in self.pred_results])
        return res


    @property
    def mpjpe_mano(self):
        res = np.average([data['mpjpe_mano']
                          for data in self.pred_results])
        return res


    @property
    def mpjpe_3d(self):
        res = np.average([data['mpjpe_3d']
                          for data in self.pred_results])
        return res
    
    @property
    def auc_3d(self):
        all_pck = self.get_all_pck('pck_3d')
        auc_3d = metric_utils.calc_auc_3d(all_pck)
        return auc_3d
    
    @property
    def auc_2d(self):
        all_pck = self.get_all_pck('pck_2d')
        auc_2d = metric_utils.calc_auc_2d(all_pck)
        return auc_2d
    
    @property
    def seq_jitter(self):
        seq_jitter = metric_utils.calc_seq_jitter(self.pred_results)
        return seq_jitter
    

    def get_all_pck(self, key):
        assert key in ['pck_3d', 'pck_2d']
        all_pck = np.array([-1,])
        for data in self.pred_results:
            pck = data[key]
            all_pck = np.concatenate((all_pck, pck))
        all_pck = all_pck[1:]
        return all_pck


    def visualize_result_single(self, start, end, res_dir, use_origin_size, use_double_size):
        for i, result in enumerate(self.pred_results[start:end]):
            # get result subdir path and image file path
            img_path = self.data_list[result['data_idx']]['image_path']
            img_name = self.data_list[result['data_idx']]['image_name']
            res_img_path = osp.join(res_dir, img_name)
            # print("img_name", img_name)
            # print("res_img_path", res_img_path)
            # render predicted smpl to image
            ori_img = cv2.imread(img_path)
            if use_origin_size:
                final_size = np.max(ori_img.shape[:2])
            elif use_double_size:
                final_size = self.inputSize * 2
            else:
                final_size = self.inputSize
            img = eval_utils.pad_and_resize(ori_img, final_size)
            render_img_pred = render_mesh_to_image(
                final_size, img, result['cam'], result['pred_verts'], self.right_hand_faces_local)
            res_img = np.concatenate((img, render_img_pred), axis=1)
            # render predicted smpl in different views
            for vis_verts in result['pred_verts_multi_view']:
                bg_img = np.ones((final_size, final_size, 3), dtype=np.uint8) * 255
                cam = self.mean_params[:3]
                render_img = render_mesh_to_image(
                    final_size, bg_img, cam, vis_verts, self.right_hand_faces_local)
                res_img = np.concatenate((res_img, render_img), axis=1)
            # save results
            res_img_path = res_img_path.replace(".png", ".jpg")
            cv2.imwrite(res_img_path, res_img)
            if i%10 == 0:
                print("{} Processed:{}/{}".format(os.getpid(), i, end-start))
            

    def visualize_result(self, res_dir, use_origin_size, use_double_size):
        # start processing
        # self.pred_results = self.pred_results[:64]
        num_process = min(len(self.pred_results), 64)
        num_each = len(self.pred_results) // num_process
        process_list = list()
        eval_utils.build_dirs(self.pred_results, self.data_list, res_dir)
        for i in range(num_process):
            start = i*num_each
            end = (i+1)*num_each if i<num_process-1 else len(self.pred_results)
            p = mp.Process(target=self.visualize_result_single, args=(start, end, res_dir, use_origin_size, use_double_size))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()


def get_all_pred_files(root_dir, dataset):
    all_files = list()
    for file in os.listdir(root_dir):
        if file.startswith("estimator") and file.endswith(".pkl"):
            if file.find(dataset) >= 0:
                all_files.append(osp.join(root_dir, file))
    return all_files
    


def main():
    use_origin_size = False
    use_double_size = False
    if len(sys.argv) > 2:
        use_origin_size = (sys.argv[2] == 'use_origin_size')
        use_double_size = (sys.argv[2] == 'use_double_size')
    
    dataset = sys.argv[1]
    root_dir = "evaluate_results"
    pred_result_files = get_all_pred_files(root_dir, dataset)

    for pred_result_file in pred_result_files:
        pkl_path = pred_result_file
        name = pkl_path.split('/')[-1].replace("estimator_", '').replace(".pkl", "").replace(".pth", "")
        res_dir = osp.join(f"evaluate_results/images/{name}")
        ry_utils.build_dir(res_dir)

        evaluator = pio.load_pkl_single(pkl_path)
        evaluator.visualize_result(res_dir, use_origin_size, use_double_size)

        print(f"{name} complete")


if __name__ == '__main__':
    main()