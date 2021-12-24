# Python wrapper

import torch
from torch import nn
from torch.autograd import Function

import rel_to_abs_index_cuda


class RelToAbsIndexCUDA(Function):
    @staticmethod
    def forward(ctx, rel_idx_map, init_idx_map, nw_spixels, nh_spixels):
        spixel_feats = rel_to_abs_index_cuda.forward(rel_idx_map, init_idx_map, nw_spixels, nh_spixels)
        return spixel_feats

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class RelToAbsIndex(nn.Module):
    def __init__(self, nw_spixels, nh_spixels, init_idx_map):
        super().__init__()
        self.nwh_spixels = (nw_spixels, nh_spixels)
        self.init_idx_map = init_idx_map
    
    def forward(self, rel_idx_map):
        return RelToAbsIndexCUDA.apply(rel_idx_map, self.init_idx_map, *self.nwh_spixels)