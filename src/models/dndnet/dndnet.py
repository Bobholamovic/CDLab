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
        self.conv_itm = nn.Conv2d(itm_ch+in_ch, itm_ch, 1)
        self.distill = Distill(itm_ch, k, p)
        self.conv_out = nn.Conv2d(2*itm_ch, out_ch, 1, bias=False)

    def forward(self, t1, t2):
        x1 = self.backbone(t1)
        x1 = self.conv_itm(torch.cat([x1, t1], dim=1)) + x1
        x2 = self.backbone(t2)
        x2 = self.conv_itm(torch.cat([x2, t2], dim=1)) + x2

        y1, y2 = self.distill(x1, x2)

        pred = self.conv_out(torch.cat([y1, y2], dim=1))

        if self.training:
            return x1, x2, y1, y2, pred
        else:
            return pred


class BaselineModel(DnDNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        class _Identity2(nn.Module):
            def forward(self, x1, x2):
                return x1, x2

        self.distill = _Identity2()


class DnDNet_align(DnDNet):
    def forward(self, t1, t2):
        feats1 = self.backbone.encoder(t1)
        feats2 = self.backbone.encoder(t2)

        # Align bi-temporal features
        feats1 = list(map(self.ada_in, feats1, feats2))
        
        x1 = self.backbone.segmentation_head(self.backbone.decoder(*feats1))
        x1 = self.conv_itm(torch.cat([x1, feats1[0]], dim=1)) + x1
        x2 = self.backbone.segmentation_head(self.backbone.decoder(*feats2))
        x2 = self.conv_itm(torch.cat([x2, feats2[0]], dim=1)) + x2

        y1, y2 = self.distill(x1, x2)

        pred = self.conv_out(torch.cat([y1, y2], dim=1))

        if self.training:
            return x1, x2, y1, y2, pred
        else:
            return pred

    def ada_in(self, f1, f2):
        mu1 = f1.mean(1, keepdims=True)
        sigma1 = f1.std(1, keepdims=True)
        mu2 = f2.mean(1, keepdims=True)
        sigma2 = f2.std(1, keepdims=True)
        return sigma2*(f1-mu1)/sigma1 + mu2


class BaselineModel_align(DnDNet_align):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        class _Identity2(nn.Module):
            def forward(self, x1, x2):
                return x1, x2

        self.distill = _Identity2()