from os.path import join, basename
from functools import lru_cache

import numpy as np

from utils.data_utils.misc import default_loader
from utils.data_utils.augmentations import Crop
from . import CDDataset


class _AirChangeDataset(CDDataset):
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1,
        subset='val'
    ):
        super().__init__(root, phase, transforms, repeats, subset)
        self.cropper = Crop(bounds=(0, 0, 784, 448))

    @property
    def LOCATION(self):
        return ''

    @property
    def TEST_SAMPLE_IDS(self):
        return ()

    @property
    def N_PAIRS(self):
        return 0

    def _read_file_paths(self):
        if self.subset == 'train':
            sample_ids = [i for i in range(self.N_PAIRS) if i not in self.TEST_SAMPLE_IDS]
            t1_list = [join(self.root, self.LOCATION, str(i+1), 'im1') for i in sample_ids]
            t2_list = [join(self.root, self.LOCATION, str(i+1), 'im2') for i in sample_ids]
            tar_list = [join(self.root, self.LOCATION, str(i+1), 'gt') for i in sample_ids]
        else:
            # val and test subsets are equal
            t1_list = [join(self.root, self.LOCATION, str(i+1), 'im1') for i in self.TEST_SAMPLE_IDS]
            t2_list = [join(self.root, self.LOCATION, str(i+1), 'im2') for i in self.TEST_SAMPLE_IDS]
            tar_list = [join(self.root, self.LOCATION, str(i+1), 'gt') for i in self.TEST_SAMPLE_IDS]

        return t1_list, t2_list, tar_list

    # XXX: In a multi-process environment, there might be multiple caches in memory, each for one process.
    @lru_cache(maxsize=16)
    def fetch_image(self, image_name):
        image = self._bmp_loader(image_name)
        return image if self.phase == 'train' else self.cropper(image)

    @lru_cache(maxsize=8)
    def fetch_target(self, target_name):
        tar = self._bmp_loader(target_name)
        tar = (tar > 0).astype(np.bool)    # To 0,1
        return tar if self.phase == 'train' else self.cropper(tar)

    def get_name(self, index):
        return '{loc}-{id}-cm.bmp'.format(loc=self.LOCATION, id=index)

    @staticmethod
    def _bmp_loader(bmp_path_wo_ext):
        # Case-insensitive .bmp loader
        try:
            return default_loader(bmp_path_wo_ext+'.bmp')
        except FileNotFoundError:
            return default_loader(bmp_path_wo_ext+'.BMP')