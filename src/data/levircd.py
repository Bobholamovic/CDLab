# H. Chen and Z. Shi, “A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection,” Remote Sensing, vol. 12, no. 10, p. 1662, 2020, doi: 10.3390/rs12101662.

import random
from glob import glob
from os.path import join

import numpy as np

from . import CDDataset


class LEVIRCDDataset(CDDataset):
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1,
        subset='val',
        aug_train=False
    ):
        super().__init__(root, phase, transforms, repeats, subset)
        self.aug_train = aug_train

    def _read_file_paths(self):
        t1_list = sorted(glob(join(self.root, self.subset, 'A', '**', '*.png'), recursive=True))
        t2_list = sorted(glob(join(self.root, self.subset, 'B', '**', '*.png'), recursive=True))
        tar_list = sorted(glob(join(self.root, self.subset, 'label', '**', '*.png'), recursive=True))
        assert len(t1_list) == len(t2_list) == len(tar_list)
        return t1_list, t2_list, tar_list

    def fetch_target(self, target_path):
        return (super().fetch_target(target_path)/255).astype(np.bool)

    def preprocess(self, t1, t2, tar):
        if self.phase == 'train' and self.aug_train:
            if random.random() < 0.2:
                # Time reversal
                t1, t2 = t2, t1
            if random.random() < 0.2:
                # Random identity
                t2 = t1
                tar.fill(0)
        return super().preprocess(t1, t2, tar)