#!/usr/bin/env bash

import sys
import math
import random
from itertools import count
from os import makedirs
from os.path import join, exists

import numpy as np
from skimage.io import imread, imsave


TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
CROP_SIZE = 256
STRIDE = 256
EXT = '.png'
SEED = 114514


if __name__ == '__main__':
    random.seed(SEED)

    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    im1_path = join(data_dir, 'before', 'before.tif')
    im2_path = join(data_dir, 'after', 'after.tif')
    gt_path = join(data_dir, 'change label', 'change_label.tif')

    im1 = imread(im1_path)
    im2 = imread(im2_path)
    gt = (imread(gt_path)*255).astype('uint8')  # bool to uint8 (1->255)
    
    h, w = gt.shape[:2]
    num_patches = math.ceil(h/CROP_SIZE)*math.ceil(w/CROP_SIZE)
    indices = list(range(num_patches))
    random.shuffle(indices)
    train_cnt = int(num_patches*TRAIN_RATIO)
    val_cnt = int(num_patches*VAL_RATIO)
    train_indices = set(indices[:train_cnt])
    val_indices = set(indices[train_cnt:train_cnt+val_cnt])
    
    global_counter = count()
    train_counter = count()
    val_counter = count()
    test_counter = count()

    for i in range(0, h, STRIDE):
        for j in range(0, w, STRIDE):
            idx = next(global_counter)
            if idx in train_indices:
                subset = 'train'
                counter = train_counter
            elif idx in val_indices:
                subset = 'val'
                counter = val_counter
            else:
                subset = 'test'
                counter = test_counter
            name = str(next(counter))+EXT
            for tag, im in (('A',im1),('B',im2),('label',gt)):
                print(subset, tag, name)
                out_subdir = join(out_dir, subset, tag)
                if not exists(out_subdir):
                    makedirs(out_subdir)
                patch = im[i:i+CROP_SIZE, j:j+CROP_SIZE]
                # Padding to the right and bottom
                if patch.shape[0] < CROP_SIZE or patch.shape[1] < CROP_SIZE:
                    pad_width = [(0,CROP_SIZE-patch.shape[0]), (0,CROP_SIZE-patch.shape[1])]
                    if len(patch.shape) == 3:
                        pad_width.append((0,0))
                    patch = np.pad(patch, pad_width=pad_width, mode='constant')
                imsave(join(out_subdir, name), patch)