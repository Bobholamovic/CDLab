# Adapted from https://github.com/Bobholamovic/ESCNet/blob/main/utils.py

import math

import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def init_grid(n_spixels_expc, w, h):
    # n_spixels >= n_spixels_expc
    nw_spixels = math.ceil(math.sqrt(w*n_spixels_expc/h))
    nh_spixels = math.ceil(math.sqrt(h*n_spixels_expc/w))

    n_spixels = nw_spixels*nh_spixels   # Actual number of spixels

    if n_spixels > w*h:
        raise ValueError("Superpixels must be fewer than pixels!")
        
    w_spixel, h_spixel = (w+nw_spixels-1) // nw_spixels, (h+nh_spixels-1) // nh_spixels
    rw, rh = w_spixel*nw_spixels-w, h_spixel*nh_spixels-h

    if (rh/2 + h_spixel) < 0 or (rw/2 + w_spixel) < 0 or (rh/2-h_spixel) > 0 or (rw/2-w_spixel) > 0:
        raise ValueError("The expected number of superpixels does not fit the image size!")

    y = np.array([-1, *np.arange((h_spixel-1)/2, h+rh, h_spixel), h+rh])-rh/2
    x = np.array([-1, *np.arange((w_spixel-1)/2, w+rw, w_spixel), w+rw])-rw/2

    s = np.arange(n_spixels).reshape(nh_spixels, nw_spixels).astype(np.int32)
    s = np.pad(s, ((1,1),(1,1)), 'edge')
    f = RegularGridInterpolator((y, x), s, method='nearest')

    pts = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    pts = np.stack(pts, axis=-1)
    init_idx_map = f(pts).astype(np.int32)
    
    return init_idx_map, n_spixels, nw_spixels, nh_spixels


def get_scale_factors(eta_pos, gamma_clr, nw_spixels, nh_spixels, w, h):
    return eta_pos*max(nw_spixels/w, nh_spixels/h), gamma_clr


class FeatureConverter:
    def __init__(self, eta_pos, gamma_clr, offsets):
        super().__init__()
        self.eta_pos = eta_pos
        self.gamma_clr = gamma_clr
        self.offsets = torch.Tensor(offsets).view(1,len(offsets),1,1)

    @torch.no_grad()
    def __call__(self, feats, nw_spixels, nh_spixels):
        # Do not require grad
        b, c, h, w = feats.size()
        scale_pos, scale_clr = get_scale_factors(self.eta_pos, self.gamma_clr, nw_spixels, nh_spixels, w, h)

        scales = torch.Tensor((scale_pos,)*2+(scale_clr,)*(c-2))
        scales.resize_(1,c,1,1)
        
        return feats * scales.type_as(feats) + self.offsets.type_as(feats)