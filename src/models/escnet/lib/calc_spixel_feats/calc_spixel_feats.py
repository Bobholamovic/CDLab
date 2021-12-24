# Python wrapper

import torch
from torch import nn
from torch.autograd import Function

import calc_spixel_feats_cuda


class CalcSpixelFeatsCUDA(Function):
    @staticmethod
    def forward(ctx, pixel_feats, assoc_map, index_map, nw_spixels, nh_spixels):
        spixel_feats, weights = calc_spixel_feats_cuda.forward(pixel_feats, assoc_map, index_map, nw_spixels, nh_spixels)
        ctx.saved_sizes = (nw_spixels, nh_spixels)
        ctx.save_for_backward(pixel_feats, assoc_map, index_map, spixel_feats, weights)
        return spixel_feats

    @staticmethod
    def backward(ctx, grad_output):
        pixel_feats, assoc_map, index_map, spixel_feats, weights = ctx.saved_tensors
        nw_spixels, nh_spixels = ctx.saved_sizes
        grad_feat, grad_assoc = calc_spixel_feats_cuda.backward(
            grad_output.contiguous(), pixel_feats, assoc_map, index_map, 
            spixel_feats, weights, 
            nw_spixels, nh_spixels
        )
        return grad_feat, grad_assoc, None, None, None


class CalcSpixelFeats(nn.Module):
    def __init__(self, nw_spixels, nh_spixels, index_map):
        super().__init__()
        self.nwh_spixels = (nw_spixels, nh_spixels)
        self.index_map = index_map
    
    def forward(self, pixel_feats, assoc_map):
        return CalcSpixelFeatsCUDA.apply(pixel_feats, assoc_map, self.index_map, *self.nwh_spixels)