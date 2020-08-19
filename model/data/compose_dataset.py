from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import os.path as osp
import random
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
import torch.utils.data as data
from PIL import Image
from scipy import misc
import cv2
import numpy as np
import pickle
'''
from data.freihand_dataset import FreiHandDataset
from data.ho3d_dataset import HO3DDataset
from data.wild_eval_dataset import WildEvalDataset
'''
# from data.detailed_dataset import DetailedDataset


class ComposeDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt


        # load all potential dataset first
        dataset_info = dict(
            freihand =  ("freihand", opt.freihand_anno_path, "freihand/image", False),
            ho3d = ("ho3d", opt.ho3d_anno_path, "ho3d/image_tight", False),
            mtc = ("mtc", opt.mtc_anno_path, "mtc/data_processed/image", True), 
            stb = ("stb", opt.stb_anno_path, "stb/image", False),
            rhd = ("rhd", opt.rhd_anno_path, "rhd/image", False),
            frl = ("frl", opt.frl_anno_path, "frl/image", False),
            ganerated = ('ganerated', opt.ganerated_anno_path, "ganerated/GANerated/data_processed/image", False),
            pmhand = ('pmhand', opt.pmhand_anno_path, "panoptic_hand/image", False),
            demo = ("demo", opt.demo_img_dir, opt.demo_img_dir, False),
        )

        '''
        dataset_info = dict(
            freihand =  ("freihand", opt.freihand_anno_path, "freihand/image", False),
        )
        '''

        all_potential_datasets = dict()
        for dataset_name in dataset_info:
            dataset = BaseDataset(opt, dataset_info[dataset_name])
            all_potential_datasets[dataset_name] = dataset
            # print("dataset_name", dataset_name)
            # sys.stdout.flush()
        
        candidate_datasets = list()
        if opt.isTrain:
            datasets_str = opt.train_datasets
            train_dataset_names = datasets_str.strip().split(',')
            for dataset_name in train_dataset_names:
                candidate_datasets.append(all_potential_datasets[dataset_name])
        else:
            candidate_datasets.append(all_potential_datasets[opt.test_dataset])
        
        assert(len(candidate_datasets)>0)
        self.all_datasets = candidate_datasets

        for dataset in candidate_datasets:
            dataset.load_data()

        if self.opt.isTrain and self.opt.sample_train_data:
            self.sample_train_data()

        if opt.process_rank <= 0: 
            for dataset in candidate_datasets:
                print('{} dataset has {} data'.format(dataset.name, len(dataset)-dataset.num_add))

        self.set_index_map()


    def set_index_map(self):
        total_data_num = self.__len__()
        index_map = list()
        for dataset_id, dataset in enumerate(self.all_datasets):
            dataset_len = len(dataset)
            index_map += [(dataset_id, idx) for idx in range(dataset_len)]
        self.index_map = index_map


    def __getitem__(self, index):
        dataset_id, dataset_index = self.index_map[index]
        dataset = self.all_datasets[dataset_id]
        data = dataset.getitem(dataset_index)
        return data


    def shuffle_data(self):
        for dataset in self.all_datasets:
            random.shuffle(dataset.data_list)
    

    def sample_train_data(self):
        for dataset in self.all_datasets:
            dataset.sample_train_data()


    def __len__(self):
        total_data_num = sum([len(dataset) for dataset in self.all_datasets])
        return total_data_num

    @property
    def name(self):
        'ComposeDataset'