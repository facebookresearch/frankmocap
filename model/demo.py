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


def get_checkpoint():
    epochs = list()
    for file in os.listdir("checkpoints"):
        if file.endswith("encoder.pth"):
            epoch = int(file.split('_')[0])
            epochs.append(epoch)
    if len(epochs) > 1:
        print("Please mannual set epoch to run demo")
        sys.exit(0)
    else:
        epoch = epochs[0]
        print(f"Use {epoch} epoch to run demo")
        return epochs[0]


def main():
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    visualize_eval = opt.visualize_eval
    opt.process_rank = -1


    # in demo, only use wild dataset
    assert opt.test_dataset == 'demo'

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_dataset()
    assert(len(dataset.dataset.all_datasets) == 1)
    test_dataset = dataset.dataset.all_datasets[0]

    test_res_dir = 'evaluate_results'
    ry_utils.build_dir(test_res_dir)

    checkpoint_dir = opt.test_checkpoint_dir
    all_checkpoints = eval_utils.load_checkpoints(checkpoint_dir)

    for checkpoint_path in all_checkpoints:
        print("\n===================================")
        # opt.which_epoch = str(epoch)
        opt.checkpoint_path = checkpoint_path
        model = H3DWModel(opt)
        assert model.success_load, "Specificed checkpoints does not exists"
        model.eval()

        print(f"Run demo images using {checkpoint_path}")

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
            timer.click(i)
        evaluator.remove_redunc()

        record = opt.demo_img_dir.split('/')
        dataset_name = record[1] if record[0] == 'demo_data' else record[0]
        dataset_name = f"demo_{dataset_name}"

        checkpoint_name = checkpoint_path.split('/')[-1].replace(".pth", "")
        res_eval_file = osp.join(test_res_dir, f"estimator_{dataset_name}_{checkpoint_name}.pkl")
        pio.save_pkl_single(res_eval_file, evaluator)

        res_eval_file = osp.join(test_res_dir, f"pred_results_{dataset_name}_{checkpoint_name}.pkl")
        pio.save_pkl_single(res_eval_file, evaluator.pred_results)

        print(f"Results saved to {res_eval_file}")

        sys.stdout.flush()


if __name__ == '__main__':
    main()
