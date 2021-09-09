import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

from .distill import Distill
# from .diffuse import Diffuse


class DnDNet(nn.Module):
    def __init__(self, in_ch, out_ch, itm_ch=128, k=32, p=8):
        super().__init__()
        self.backbone = smp.Unet(
            encoder_name='efficientnet-b3',
            in_channels=in_ch,
            classes=itm_ch,
        )
        self.distill = Distill(itm_ch, k, p)
        self.conv_out = nn.Conv2d(2*itm_ch, out_ch, 1, bias=False)

    def forward(self, t1, t2):
        x1 = self.backbone(t1)
        x2 = self.backbone(t2)

        y1, y2 = self.distill(x1, x2)

        pred = self.conv_out(torch.cat([y1, y2], dim=1))

        if self.training:
            return x1, x2, y1, y2, pred
        else:
            return pred