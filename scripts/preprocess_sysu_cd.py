#!/usr/bin/env bash

import sys
import os
import os.path as osp
from glob import glob

import cv2
from tqdm import tqdm


if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    for subset in ('train', 'val', 'test'):
        dst_dir = osp.join(out_dir, subset, 'time1_d')
        if not osp.exists(dst_dir):
            os.makedirs(dst_dir)
        for p in tqdm(glob(osp.join(data_dir, subset, 'time1', '*.png'))):
            src = cv2.imread(p)
            dst = cv2.fastNlMeansDenoisingColored(src,None,5,5,7,21)
            cv2.imwrite(osp.join(dst_dir, osp.basename(p)), dst)