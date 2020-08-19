
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.utils.data
#from data.base_data_loader import BaseDataLoader
from data.compose_dataset import ComposeDataset
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


class CustomDatasetDataLoader(object):
    @property
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
        self.dataset = ComposeDataset(opt)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            drop_last=opt.isTrain)

    def load_dataset(self):
        return self.dataloader
    
    def shuffle_data(self):
        self.dataset.shuffle_data()
    
    def sample_train_data(self):
        self.dataset.sample_train_data()

    def __len__(self):
        return len(self.dataset)
    

class DistributedDataLoader(object):
    def initialize(self, opt):
        # print("Use distributed dataloader")
        self.dataset = ComposeDataset(opt)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        self.train_sampler = DistributedSampler(self.dataset, world_size, rank)

        num_workers = opt.nThreads
        assert opt.batchSize % world_size == 0
        batch_size = opt.batchSize // world_size
        shuffle = False
        drop_last = opt.isTrain

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=self.train_sampler,
            drop_last=drop_last,
            pin_memory=False)

    def load_dataset(self):
        return self.data_loader
    
    def shuffle_data(self):
        self.dataset.shuffle_data()
    
    def sample_train_data(self):
        self.dataset.sample_train_data()

    def __len__(self):
        return len(self.dataset)


def CreateDataLoader(opt):
    if opt.dist:
        data_loader = DistributedDataLoader()
    else:
        data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader

