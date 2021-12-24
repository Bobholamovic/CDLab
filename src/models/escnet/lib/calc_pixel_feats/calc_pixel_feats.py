# Python wrapper

import torch
from torch import nn
from torch.autograd import Function

import calc_pixel_feats_cuda


class CalcPixelFeatsCUDA(Function):
    @staticmethod
    def forward(ctx, spixel_feats, assoc_map, index_map, nw_spixels, nh_spixels):
        pixel_feats, weights = calc_pixel_feats_cuda.forward(spixel_feats, assoc_map, index_map, nw_spixels, nh_spixels)
        ctx.saved_sizes = (nw_spixels, nh_spixels)
        ctx.save_for_backward(spixel_feats, assoc_map, index_map, pixel_feats, weights)
        return pixel_feats

    @staticmethod
    def backward(ctx, grad_output):
        spixel_feats, assoc_map, index_map, pixel_feats, weights = ctx.saved_tensors
        nw_spixels, nh_spixels = ctx.saved_sizes
        grad_feat, grad_assoc = calc_pixel_feats_cuda.backward(
            grad_output.contiguous(), spixel_feats, assoc_map, index_map, 
            pixel_feats, weights, 
            nw_spixels, nh_spixels
        )
        return grad_feat, grad_assoc, None, None, None


class CalcPixelFeats(nn.Module):
    def __init__(self, nw_spixels, nh_spixels, index_map):
        super().__init__()
        self.nwh_spixels = (nw_spixels, nh_spixels)
        self.index_map = index_map
    
    def forward(self, spixel_feats, assoc_map):
        return CalcPixelFeatsCUDA.apply(spixel_feats, assoc_map, self.index_map, *self.nwh_spixels)