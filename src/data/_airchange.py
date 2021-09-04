# C. Benedek and T. Sziranyi, "Change Detection in Optical Aerial Images by a Multilayer Conditional Mixed Markov Model," in IEEE Transactions on Geoscience and Remote Sensing, vol. 47, no. 10, pp. 3416-3430, Oct. 2009, doi: 10.1109/TGRS.2009.2022633.

from os.path import join
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
        if self.phase == 'eval':
            self._cropper = Crop(bounds=(0, 0, 784, 448))

    @property
    def LOCATION(self):
        return ''

    @property
    def TRAIN_SAMPLE_IDS(self):
        return ()

    @property
    def VAL_SAMPLE_IDS(self):
        return ()

    @property
    def TEST_SAMPLE_IDS(self):
        return ()

    def _read_file_paths(self):
        sample_ids = getattr(self, self.subset.upper()+'_SAMPLE_IDS')
        t1_list = [join(self.root, self.LOCATION, str(i), 'im1') for i in sample_ids]
        t2_list = [join(self.root, self.LOCATION, str(i), 'im2') for i in sample_ids]
        tar_list = [join(self.root, self.LOCATION, str(i), 'gt') for i in sample_ids]
        return t1_list, t2_list, tar_list

    # XXX: In a multi-process environment, there might be multiple caches in memory, each for one process.
    @lru_cache(maxsize=12)
    def fetch_image(self, image_name):
        image = self._bmp_loader(image_name)
        return image if self.phase == 'train' else self._cropper(image)

    @lru_cache(maxsize=8)
    def fetch_target(self, target_name):
        tar = self._bmp_loader(target_name)
        tar = (tar > 0).astype(np.bool)    # To 0,1
        return tar if self.phase == 'train' else self._cropper(tar)

    def get_name(self, index):
        return '{loc}-{id}-cm.png'.format(loc=self.LOCATION, id=index)

    @staticmethod
    def _bmp_loader(bmp_path_wo_ext):
        # Case-insensitive .bmp loader
        try:
            return default_loader(bmp_path_wo_ext+'.bmp')
        except FileNotFoundError:
            return default_loader(bmp_path_wo_ext+'.BMP')