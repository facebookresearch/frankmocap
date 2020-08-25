import os, sys, shutil
import os.path as osp
import time
from datetime import datetime
import torch
import numpy
import random
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.h3dw_model import H3DWModel
from util.visualizer import Visualizer
from util.evaluator import Evaluator
from util.eval_utils import ResultStat, Timer
import cv2
import numpy as np
import ry_utils as ry_utils
import parallel_io as pio
import pdb
from util import eval_utils

def main():
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    visualize_eval = opt.visualize_eval
    opt.process_rank = -1

    # train_script_file = "train.sh"
    # eval_utils.update_opt_from_train(train_script_file, opt)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_dataset()
    assert(len(dataset.dataset.all_datasets) == 1) #only test one dataset each time
    test_dataset = dataset.dataset.all_datasets[0]

    result_info = [
        ('pve', 'small', 1000), 
        ('mpjpe_mano', 'small', 1000),
        ('mpjpe_3d', 'small', 1000),
        ('auc_3d', 'large', 1),
        ('auc_2d', 'large', 1),
        ('seq_jitter', 'small', 1000)
    ]
    eval_type = eval_utils.get_evaluate_type(test_dataset)

    Stater = ResultStat(result_info, eval_type)

    test_res_dir = 'evaluate_results'
    ry_utils.build_dir(test_res_dir)


    checkpoint_dir = opt.test_checkpoint_dir
    all_checkpoints = eval_utils.load_checkpoints(checkpoint_dir)

    # for epoch in range(10, 201, 10):
    for checkpoint_path in all_checkpoints:
        print("\n===================================")
        # opt.which_epoch = str(epoch)
        opt.checkpoint_path = checkpoint_path
        model = H3DWModel(opt)
        # if there is no specified checkpoint, then skip
        if not model.success_load: continue
        model.eval()

        evaluator = Evaluator(test_dataset.data_list, model)
        evaluator.clear()

        timer = Timer(len(dataset))
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            pred_res = model.get_pred_result()
            data_idxs = data['index'].numpy()
            scale_ratios = data['scale_ratio'].numpy()
            ori_img_sizes = data['ori_img_size'].numpy()
            evaluator.update(data_idxs, scale_ratios, ori_img_sizes, pred_res)
            # timer.click(i)
        
        evaluator.remove_redunc()

        # all_metrics = ['pve', 'mpjpe_mano', 'mpjpe_3d', 'auc_3d', 'seq_jitter']
        checkpoint_name = checkpoint_path.split('/')[-1].replace(".pth", "")
        all_metrics = [info[0] for info in result_info]
        for metric in all_metrics:
            Stater.update(metric, checkpoint_name, getattr(evaluator, metric))
        Stater.print_current_result(checkpoint_name)

        # save predicted results to file
        if test_dataset.name != "demo":
            dataset_name = test_dataset.name
        else:
            record = opt.demo_img_dir.split('/')
            dataset_name = record[1] if record[0] == 'demo_data' else record[0]
            dataset_name = f"demo_{dataset_name}"

        # save prediction results
        res_eval_file = osp.join(test_res_dir, f"estimator_{dataset_name}_{checkpoint_name}.pkl")
        pio.save_pkl_single(res_eval_file, evaluator)
        res_eval_file = osp.join(test_res_dir, f"pred_results_{dataset_name}_{checkpoint_name}.pkl")
        pio.save_pkl_single(res_eval_file, evaluator.pred_results)

        # save all pcks for draw curve
        if test_dataset.name == 'pmhand':
            all_pck = evaluator.get_all_pck('pck_2d')
            res_pck_file = osp.join(test_res_dir, f"pck_{dataset_name}_{checkpoint_name}.pkl")
            pio.save_pkl_single(res_pck_file, all_pck)

       
    Stater.print_best_results()
    sys.stdout.flush()


if __name__ == '__main__':
    main()