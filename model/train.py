import os
import os.path as osp
import sys
import shutil
import time
from datetime import datetime
import torch
import numpy
import random
from util.train_utils import *
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from util.visualizer import Visualizer
from models.h3dw_model import H3DWModel
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2
import numpy as np
import pdb
import ry_utils


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def init_opt(opt):
    expr_dir = os.path.join(opt.checkpoints_dir)
    if not osp.exists(expr_dir):
        os.makedirs(expr_dir)
    args = vars(opt)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')


def save_vis_result(vis_dict, vis_dir, epoch, i):
    res_img = None
    for key, value in vis_dict.items():
        if res_img is None:
            res_img = value
        else:
            res_img = np.concatenate( (res_img, value), axis=1)
    res_path = osp.join(vis_dir, f"epoch_{epoch:03d}_iter_{i:04d}.png")
    cv2.imwrite(res_path, res_img[:,:,::-1])


def main():
    opt = TrainOptions().parse()

    # distributed learning initiate
    if opt.dist:
        init_dist()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = -1
    opt.process_rank = rank
    if rank <= 0:
        init_opt(opt)
    
    # set data loader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_dataset()
    dataset_size = len(data_loader)

    # init model
    model = H3DWModel(opt)

    # set auxiliary class
    time_stat = TimeStat(opt.total_epoch)
    if rank <= 0:
        visualizer = Visualizer(opt)
        total_steps = 0
        print_count = 0
        loss_stat = LossStat(len(data_loader))
    
    if rank <= 0:
        vis_dir = osp.join(opt.checkpoints_dir, "visualization")
        ry_utils.renew_dir(vis_dir)

    # start training
    for epoch in range(opt.epoch_count, opt.total_epoch+1):
        epoch_start_time = time.time()
        epoch_iter = 0
        # important, sample data each time
        torch.manual_seed(int(time.time()))
        numpy.random.seed(int(time.time()))
        random.seed(int(time.time()))
        data_loader.shuffle_data()
        if opt.sample_train_data:
            data_loader.sample_train_data()

        time_stat.epoch_init(epoch)
        if rank <= 0:
            loss_stat.set_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            model.set_input(data)
            time_stat.stat_data_time()
            model.forward()
            model.optimize_parameters()
            time_stat.stat_forward_time()

            if rank <= 0:
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                # get training losses
                errors = model.get_current_errors()
                loss_stat.update(errors)
                # get visualization
                if total_steps % opt.display_freq == 0:
                    vis_dict = model.get_current_visuals()
                    visualizer.display_current_results(
                        vis_dict, epoch)
                    save_vis_result(vis_dict, vis_dir, epoch, i)
                    visualizer.plot_current_errors(epoch, float(
                        epoch_iter)/dataset_size, opt, errors)
                # print loss
                if total_steps/opt.print_freq > print_count:
                    loss_stat.print_loss(epoch_iter)
                    print_count += 1
            # print training time
            time_stat.stat_visualize_time()

        if rank <= 0:
            if epoch % opt.save_epoch_freq == 0:
                print( f"saving the model at the end of epoch {epoch}, iters {total_steps}")
                model.save(epoch, epoch)
            time_stat.stat_epoch_time()
            time_stat.print_stat()

        # update learning rate, use cosine learning rate decay
        if opt.lr_decay:
            model.update_learning_rate(epoch)

if __name__ == '__main__':
    main()