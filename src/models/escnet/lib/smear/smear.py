# Python wrapper

import torch
from torch import nn
from torch.autograd import Function

import smear_cuda


class SmearCUDA(Function):
    @staticmethod
    def forward(ctx, spixel_feats, index_map, n_spixels):
        ctx.save_for_backward(index_map, spixel_feats)
        spixel_feats = smear_cuda.forward(spixel_feats, index_map, n_spixels)
        return spixel_feats

    @staticmethod
    def backward(ctx, grad_output):
        index_map, spixel_feats = ctx.saved_tensors
        spixel_feats = spixel_feats.size(-1)
        grad_feat = smear_cuda.backward(grad_output.contiguous(), index_map, spixel_feats)
        # no gradient for index_map
        return grad_feat, None, None


class Smear(nn.Module):
    def __init__(self, n_spixels):
        super().__init__()
        self.n_spixels = n_spixels
    
    def forward(self, spixel_feats, index_map):
        return SmearCUDA.apply(spixel_feats, index_map, self.n_spixels)