#!/usr/bin/env bash

import sys
import math
from itertools import count
from os import makedirs
from os.path import join, exists

from skimage.io import imread, imsave


TRAIN_AND_VAL_RATIO = 0.7
TRAIN_RATIO = 0.9
CROP_SIZE = 256
STRIDE = 128
EXT = '.png'


if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    im1_path = join(data_dir, 'before', 'before.tif')
    im2_path = join(data_dir, 'after', 'after.tif')
    gt_path = join(data_dir, 'change label', 'change_label.tif')

    im1 = imread(im1_path)
    im2 = imread(im2_path)
    gt = (imread(gt_path)*255).astype('uint8')  # bool to uint8 (1->255)
    
    # train and val subsets
    h, w = gt.shape[:2]
    train_and_val_rows = h
    train_and_val_cols = int(w*TRAIN_AND_VAL_RATIO)
    train_rows = int(train_and_val_rows*math.sqrt(TRAIN_RATIO))
    train_cols = int(train_and_val_cols*math.sqrt(TRAIN_RATIO))
    
    train_counter = count()
    val_counter = count()
    for i in range(0, train_and_val_rows-CROP_SIZE+1, STRIDE):
        for j in range(0, train_and_val_cols-CROP_SIZE+1, STRIDE):
            if i < train_rows and j < train_cols:
                subset = 'train'
                counter = train_counter
            else:
                subset = 'val'
                counter = val_counter
            name = str(next(counter))+EXT
            for tag, im in (('A',im1),('B',im2),('label',gt)):
                print(tag, name)
                out_subdir = join(out_dir, subset, tag)
                if not exists(out_subdir):
                    makedirs(out_subdir)
                imsave(join(out_subdir, name), im[i:i+CROP_SIZE, j:j+CROP_SIZE])
    
    # test subset
    for tag, im in (('A',im1),('B',im2),('label',gt)):
        out_subdir = join(out_dir, 'test', tag)
        if not exists(out_subdir):
            makedirs(out_subdir)
        imsave(join(out_subdir, 'whole'+EXT), im[:,train_and_val_cols:])