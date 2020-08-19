import os, sys, shutil
import os.path as osp
sys.path.append('src/')
import numpy as np
import copy
from augment.sample import Sample
from augment.strategy import update_wrist

class AugmentModel(object):

    def __init__(self,
        config,
        all_samples,
        strategy,
    ):
        self.config = config
        self.strategy = strategy
        self.all_samples = all_samples

        # initialize model according to different startegy
        self._init_model()
    

    def _init_model(self):
        if self.strategy == "update_wrist":
            self.threshold = self.config.strategy_params[self.strategy]['threshold']
        else:
            raise ValueError(f"Unsupported strategy:{self.config.strategy}")
    

    def update_sample(self):
        if self.strategy == 'update_wrist':
            update_wrist.apply(self)
        else:
            raise ValueError(f"Unsupported strategy:{self.config.strategy}")