# Adapted from https://github.com/Bobholamovic/ESCNet/blob/main/ssn.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv3x3, MaxPool2x2, DoubleConv
from .lib import (CalcAssoc, CalcPixelFeats, CalcSpixelFeats, InitSpixelFeats, RelToAbsIndex, Smear)
from .utils import init_grid


class FeatureExtactor(nn.Module):
    def __init__(self, n_filters=64, in_ch=5, out_ch=20):
        super().__init__()
        self.conv1 = DoubleConv(in_ch, n_filters)
        self.pool1 = MaxPool2x2()
        self.conv2 = DoubleConv(n_filters, n_filters)
        self.pool2 = MaxPool2x2()
        self.conv3 = DoubleConv(n_filters, n_filters)
        self.conv4 = Conv3x3(3*n_filters+in_ch, out_ch-in_ch, act=True)
    
    def forward(self, x):
        f1 = self.conv1(x)
        p1 = self.pool1(f1)
        f2 = self.conv2(p1)
        p2 = self.pool2(f2)
        f3 = self.conv3(p2)

        # Resize feature maps
        f2_rsz = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3_rsz = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate multi-level features and fuse them
        f_cat = torch.cat([x, f1, f2_rsz, f3_rsz], dim=1)
        f_out = self.conv4(f_cat)

        y = torch.cat([x, f_out], dim=1)

        return y


class SSN(nn.Module):
    def __init__(
        self, 
        feat_cvrter,
        n_iters=10, 
        n_spixels=100,
        n_filters=64, in_ch=5, out_ch=20,
        cnn=True
    ):
        super().__init__()

        self.n_spixels = n_spixels
        self.n_iters = n_iters

        self.feat_cvrter = feat_cvrter

        self.cnn = cnn
        if cnn:
            # The pixel-wise feature extractor
            self.cnn_modules = FeatureExtactor(n_filters, in_ch, out_ch)
        else:
            self.cnn_modules = None

        self._cached = False
        self._ops = {}
        self._layout = (None, 1, 1)

    def forward(self, x):
        if self.training:
            # Training mode
            # Use cached objects
            ops, (_, nw_spixels, nh_spixels) = self.get_ops_and_layout(x, ofa=True)
        else:
            # Evaluation mode
            # Every time update the objects
            ops, (_, nw_spixels, nh_spixels) = self.get_ops_and_layout(x, ofa=False)

        x = self.feat_cvrter(x, nw_spixels, nh_spixels)

        # Forward
        pf = self.cnn_modules(x) if self.cnn else x

        spf = ops['init_spixels'](pf.detach())

        # Iterations
        for itr in range(self.n_iters):
            Q = self.nd2Q(ops['calc_neg_dist'](pf, spf))
            spf = ops['map_p2sp'](pf, Q)

        return Q, ops, x, spf, pf

    @staticmethod
    def nd2Q(neg_dist):
        # Use softmax to compute pixel-superpixel relative soft-associations (degree of membership)
        return F.softmax(neg_dist, dim=1)

    def get_ops_and_layout(self, x, ofa=False):
        if ofa and self._cached:
            return self._ops, self._layout
        
        b, _, h, w = x.size()   # Get size of the input

        # Initialize grid
        init_idx_map, n_spixels, nw_spixels, nh_spixels = init_grid(self.n_spixels, w, h)
        init_idx_map = torch.IntTensor(init_idx_map).expand(b, 1, h, w).to(x.device)

        # Contruct operation modules
        init_spixels = InitSpixelFeats(n_spixels, init_idx_map)
        map_p2sp = CalcSpixelFeats(nw_spixels, nh_spixels, init_idx_map)
        map_sp2p = CalcPixelFeats(nw_spixels, nh_spixels, init_idx_map)
        calc_neg_dist = CalcAssoc(nw_spixels, nh_spixels, init_idx_map)
        map_idx = RelToAbsIndex(nw_spixels, nh_spixels, init_idx_map)
        smear = Smear(n_spixels)

        ops = {
            'init_spixels': init_spixels,
            'map_p2sp': map_p2sp,
            'map_sp2p': map_sp2p,
            'calc_neg_dist': calc_neg_dist,
            'map_idx': map_idx,
            'smear': smear
        }

        if ofa:
            self._ops = ops
            self._layout = (init_idx_map, nw_spixels, nh_spixels)
            self._cached = True

        return ops, (init_idx_map, nw_spixels, nh_spixels)