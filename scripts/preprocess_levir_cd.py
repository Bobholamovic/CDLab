#!/usr/bin/env bash

import sys
from glob import glob
from itertools import count
from os import makedirs
from os.path import join, basename, splitext, exists

from skimage.io import imread, imsave
from tqdm import tqdm


CROP_SIZE = 256
STRIDE = 256


if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    for subset in ('train', 'val', 'test'):
        for subdir in ('A','B','label'):
            for path in tqdm(glob(join(data_dir, subset, subdir, '*.png'))):
                name, ext = splitext(basename(path))
                img = imread(path)
                w, h = img.shape[:2]
                counter = count()
                out_subdir = join(out_dir, subset, subdir, name)
                if not exists(out_subdir):
                    makedirs(out_subdir)
                stride = STRIDE//2 if subset == 'train' else STRIDE
                for i in range(0, h-CROP_SIZE+1, stride):
                    for j in range(0, w-CROP_SIZE+1, stride):
                        imsave(join(out_subdir, '{}_{}{}'.format(name,next(counter),ext)), img[i:i+CROP_SIZE, j:j+CROP_SIZE])