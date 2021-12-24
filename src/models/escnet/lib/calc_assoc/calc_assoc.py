# Python wrapper

import torch
from torch import nn
from torch.autograd import Function

import calc_assoc_cuda


class CalcAssocCUDA(Function):
    @staticmethod
    def forward(ctx, pixel_feats, spixel_feats, index_map, nw_spixels, nh_spixels):
        ctx.save_for_backward(pixel_feats, spixel_feats, index_map)
        ctx.saved_sizes = (nw_spixels, nh_spixels)
        return calc_assoc_cuda.forward(pixel_feats, spixel_feats, index_map, nw_spixels, nh_spixels)

    @staticmethod
    def backward(ctx, grad_output):
        pixel_feats, spixel_feats, index_map = ctx.saved_tensors
        nw_spixels, nh_spixels = ctx.saved_sizes
        grad_feat, grad_assoc = calc_assoc_cuda.backward(
            grad_output.contiguous(), 
            pixel_feats, spixel_feats, 
            index_map, 
            nw_spixels, nh_spixels
        )
        return grad_feat, grad_assoc, None, None, None


class CalcAssoc(nn.Module):
    def __init__(self, nw_spixels, nh_spixels, index_map):
        super().__init__()
        self.nwh_spixels = (nw_spixels, nh_spixels)
        self.index_map = index_map
    
    def forward(self, pixel_feats, spixel_feats):
        return CalcAssocCUDA.apply(pixel_feats, spixel_feats, self.index_map, *self.nwh_spixels)