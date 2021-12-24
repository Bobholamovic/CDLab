# Python wrapper

import torch
from torch import nn
from torch.autograd import Function

import init_spixel_feats_cuda


class InitSpixelFeatsCUDA(Function):
    @staticmethod
    def forward(ctx, pixel_feats, index_map, n_spixels):
        spixel_feats = init_spixel_feats_cuda.forward(pixel_feats, index_map, n_spixels)
        return spixel_feats

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class InitSpixelFeats(nn.Module):
    def __init__(self, n_spixels, index_map):
        super().__init__()
        self.n_spixels = n_spixels
        self.index_map = index_map
    
    def forward(self, pixel_feats):
        return InitSpixelFeatsCUDA.apply(pixel_feats, self.index_map, self.n_spixels)