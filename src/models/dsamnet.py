# Implementation of
# Q. Shi, M. Liu, S. Li, X. Liu, F. Wang, and L. Zhang, “A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection,” IEEE Trans. Geosci. Remote Sensing, pp. 1–16, 2021, doi: 10.1109/TGRS.2021.3085870.

# The resnet implementation differs from the original work. See src/models/stanet.py for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._blocks import make_norm
from ._common import CBAM
from .stanet import Backbone, Decoder


class DSLayer(nn.Sequential):
    def __init__(self, in_ch, out_ch, itm_ch, **convd_kwargs):
        super().__init__(
            nn.ConvTranspose2d(in_ch, itm_ch, kernel_size=3, padding=1, **convd_kwargs),
            make_norm(itm_ch),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.ConvTranspose2d(itm_ch, out_ch, kernel_size=3, padding=1)
        )


class DSAMNet(nn.Module):
    def __init__(self, in_ch, out_ch, width=64, backbone='resnet18', ca_ratio=8, sa_kernel=7):
        super().__init__()

        self.backbone = Backbone(in_ch=in_ch, arch=backbone, strides=(1,1,2,2,1))
        self.decoder = Decoder(width)

        self.cbam1 = CBAM(64, ratio=ca_ratio, kernel_size=sa_kernel)
        self.cbam2 = CBAM(64, ratio=ca_ratio, kernel_size=sa_kernel)

        self.dsl2 = DSLayer(64, out_ch, 32, stride=2, output_padding=1)
        self.dsl3 = DSLayer(128, out_ch, 32, stride=4, output_padding=3)

        self.calc_dist = nn.PairwiseDistance(keepdim=True)

    def forward(self, t1, t2):
        f1 = self.backbone(t1)
        f2 = self.backbone(t2)

        y1 = self.decoder(f1)
        y2 = self.decoder(f2)

        y1 = self.cbam1(y1)
        y2 = self.cbam2(y2)

        dist = self.calc_dist(y1, y2)
        dist = F.interpolate(dist, size=t1.shape[2:], mode='bilinear', align_corners=True)

        ds2 = self.dsl2(torch.abs(f1[0]-f2[0]))
        ds3 = self.dsl3(torch.abs(f1[1]-f2[1]))

        return dist, ds2, ds3