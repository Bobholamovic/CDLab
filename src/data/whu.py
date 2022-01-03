# S. Ji, S. Wei, and M. Lu, “Fully Convolutional Networks for Multisource Building Extraction From an Open Aerial and Satellite Imagery Data Set,” IEEE Trans. Geosci. Remote Sensing, vol. 57, no. 1, pp. 574–586, 2019, doi: 10.1109/TGRS.2018.2858817.

import random
from glob import glob
from os.path import join

import numpy as np

from . import CDDataset


class WHUDataset(CDDataset):
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1,
        subset='val'
    ):
        super().__init__(root, phase, transforms, repeats, subset)

    def _read_file_paths(self):
        t1_list = sorted(glob(join(self.root, self.subset, 'A', '*.png')))
        t2_list = sorted(glob(join(self.root, self.subset, 'B', '*.png')))
        tar_list = sorted(glob(join(self.root, self.subset, 'label', '*.png')))
        assert len(t1_list) == len(t2_list) == len(tar_list)
        return t1_list, t2_list, tar_list

    def fetch_target(self, target_path):
        return (super().fetch_target(target_path)/255).astype(np.bool)