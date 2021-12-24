#!/bin/bash

for module in calc_assoc calc_pixel_feats calc_spixel_feats init_spixel_feats rel_to_abs_index smear
do
    cd $module
    python setup.py install
    cd ..
done