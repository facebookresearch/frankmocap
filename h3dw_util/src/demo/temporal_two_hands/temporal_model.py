import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
import copy
from demo.temporal_two_hands.strategy import copy_and_paste
from demo.temporal_two_hands.strategy import average_frame

class TemporalModel(object):

    def __init__(self,
        config,
        all_samples,
        strategy,
    ):
        self.config = config
        self.strategy = strategy
        self.samples_origin = copy.deepcopy(all_samples)

        # initialize model according to different startegy
        self._init_model()
    

    def _init_model(self):
        if self.strategy == "copy_and_paste":
            self.memory_size = self.config.strategy_params[self.strategy]['memory_size']
            self.memory_bank_left = list() # memory_bank for left hand
            self.memory_bank_right = list() # memory_bank for right hand
        
        elif self.strategy == "average_frame":
            self.win_size = self.config.strategy_params[self.strategy]['win_size']
        else:
            raise ValueError(f"Unsupported strategy:{self.config.strategy}")
    

    def update_sample(self):
        if self.strategy == 'copy_and_paste':
            samples_new = copy_and_paste.apply(self)
        
        elif self.strategy == 'average_frame':
            samples_new = average_frame.apply(self)

        else:
            raise ValueError(f"Unsupported strategy:{self.config.strategy}")

        return samples_new