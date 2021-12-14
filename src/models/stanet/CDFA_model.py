# Adapted from https://github.com/justchenhao/STANet/blob/master/models/CDFA_model.py
# with massive modification

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from . import backbone


class CDFAModel(nn.Module):
    def __init__(self, in_c, f_c, out_c, arch, ds, SA_mode):
        super().__init__()
        self.ds = 1
        self.n_class =2
        self.netF = backbone.define_F(in_c=in_c, f_c=f_c, type=arch)
        self.netA = backbone.CDSA(in_c=f_c, ds=ds, mode=SA_mode)
        # Use BIT classification head to replace the metric learning module
        self.conv_out = nn.Sequential(
            nn.Conv2d(f_c, f_c, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(f_c),
            nn.ReLU(),
            nn.Conv2d(f_c, out_c, 3, padding=1, stride=1)
        )

    def forward(self, A, B):
        feat_A = self.netF(A)  # f(A)
        feat_B = self.netF(B)   # f(B)

        feat_A, feat_B = self.netA(feat_A, feat_B)
        feat_A = F.interpolate(feat_A, size=A.shape[2:], mode='bilinear', align_corners=True)
        feat_B = F.interpolate(feat_B, size=B.shape[2:], mode='bilinear', align_corners=True)

        return self.conv_out(torch.abs(feat_A-feat_B))