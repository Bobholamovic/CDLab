# Q. Shi, M. Liu, S. Li, X. Liu, F. Wang, and L. Zhang, “A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection,” IEEE Trans. Geosci. Remote Sensing, pp. 1–16, 2021, doi: 10.1109/TGRS.2021.3085870.

import random
from glob import glob
from os.path import join

import numpy as np

from . import CDDataset


class SYSUCDDataset(CDDataset):
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1,
        subset='val',
        use_denoised=False
    ):
        self.use_denoised = use_denoised
        super().__init__(root, phase, transforms, repeats, subset)

    def _read_file_paths(self):
        t1_list = sorted(glob(join(self.root, self.subset, 'time1_d' if self.use_denoised else 'time1', '*.png')))
        t2_list = sorted(glob(join(self.root, self.subset, 'time2', '*.png')))
        tar_list = sorted(glob(join(self.root, self.subset, 'label', '*.png')))
        assert len(t1_list) == len(t2_list) == len(tar_list)
        return t1_list, t2_list, tar_list

    def fetch_target(self, target_path):
        return (super().fetch_target(target_path)/255).astype(np.bool)