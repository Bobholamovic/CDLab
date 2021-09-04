# R. C. Daudt, B. Le Saux, A. Boulch and Y. Gousseau, "Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks," IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, Valencia, 2018, pp. 2115-2118, doi: 10.1109/IGARSS.2018.8518015.

import os
from os.path import join, basename

import numpy as np

from utils.data_utils.misc import default_loader
from . import CDDataset


class OSCDDataset(CDDataset):
    _BAND_NAMES = (
        'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
        'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
    )
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1,
        subset='val',
        cache_level=1
    ):
        super().__init__(root, phase, transforms, repeats, subset)
        # cache_level=0 for no cache, 1 to cache labels, 2 and higher to cache all.
        self.cache_level = int(cache_level)
        if self.cache_level > 0:
            self._pool = dict()

    def _read_file_paths(self):
        image_dir = join(self.root, "Onera Satellite Change Detection dataset - Images")
        if self.subset in ('train', 'val'):
            target_dir = join(self.root, "Onera Satellite Change Detection dataset - Train Labels")
            txt_file = join(image_dir, "train.txt")
        elif self.subset == 'test':
            target_dir = join(self.root, "Onera Satellite Change Detection dataset - Test Labels")
            txt_file = join(image_dir, "test.txt")

        # Read cities
        with open(txt_file, 'r') as f:
            cities = [city.strip() for city in f.read().strip().split(',')]

        if self.subset == 'train':
            # For training, use the first 11 pairs
            cities = cities[:-3]
        elif self.subset == 'val':
            # For validation, use the remaining 3 pairs
            cities = cities[-3:]
            
        # Use resampled images
        t1_list = [[join(image_dir, city, "imgs_1_rect", band+'.tif') for band in self._BAND_NAMES] for city in cities]
        t2_list = [[join(image_dir, city, "imgs_2_rect", band+'.tif') for band in self._BAND_NAMES] for city in cities]
        tar_list = [join(target_dir, city, 'cm', city+'-cm.tif') for city in cities]

        return t1_list, t2_list, tar_list

    def fetch_image(self, image_paths):
        key = '-'.join(image_paths[0].split(os.sep)[-3:-1])
        if self.cache_level >= 2:
            image = self._pool.get(key, None)
            if image is not None:
                return image
        image = np.stack([default_loader(p) for p in image_paths], axis=-1).astype(np.float32)
        if self.cache_level >= 2:
            self._pool[key] = image
        return image

    def fetch_target(self, target_path):
        key = basename(target_path)
        if self.cache_level >= 1:
            tar = self._pool.get(key, None)
            if tar is not None:
                return tar
        # In the tif labels, 1 stands for NC and 2 for C,
        # thus a -1 offset is added.
        tar = (default_loader(target_path) - 1).astype(np.bool)
        if self.cache_level >= 1:
            self._pool[key] = tar
        return tar
