# H. Chen and Z. Shi, “A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection,” Remote Sensing, vol. 12, no. 10, p. 1662, 2020, doi: 10.3390/rs12101662.

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
        subset='val'
    ):
        super().__init__(root, phase, transforms, repeats, subset)

    def _read_file_paths(self):
        t1_list = sorted(glob(join(self.root, self.subset, 'A', '*', '*.png')))
        t2_list = sorted(glob(join(self.root, self.subset, 'B', '*', '*.png')))
        tar_list = sorted(glob(join(self.root, self.subset, 'label', '*', '*.png')))
        return t1_list, t2_list, tar_list

    def fetch_target(self, target_path):
        return (super().fetch_target(target_path)/255).astype(np.bool)