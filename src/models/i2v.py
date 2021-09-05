import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision import models

from ._blocks import DecBlock, ResBlock, Conv3x3


class VideoDecoder(nn.Module):
    def __init__(self, enc_chs=(64,64,128,256,512), dec_chs=(256,128,64,32)):
        super().__init__()
        if not len(enc_chs) == len(dec_chs)+1:
            raise ValueError
        enc_chs = enc_chs[::-1]
        self.bottom = ResBlock(enc_chs[0], enc_chs[0])
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, out_ch)
            for in_ch1, in_ch2, out_ch in zip(enc_chs[1:], (enc_chs[0],)+dec_chs[:-1], dec_chs)
        ])
    
    def forward(self, *feats):
        feats = feats[::-1]
        x = self.bottom(feats[0].mean(2))
        for i, (blk, f) in enumerate(zip(self.blocks, feats[1:])):
            x = blk(f.mean(2), x)
        return x

class VideoSegModel(nn.Module):
    def __init__(self, in_ch, dec_chs):
        super().__init__()
        if in_ch != 3:
            raise ValueError
        # Encoder
        self.encoder = models.video.r3d_18(pretrained=True)
        self.encoder.fc = nn.Identity()
        # Decoder
        self.decoder = VideoDecoder(dec_chs=dec_chs)

    def forward(self, x):
        x = self.encoder.stem(x)
        feats = [F.interpolate(x, scale_factor=2, mode='nearest')]
        for i in range(4):
            layer = getattr(self.encoder, f'layer{i+1}')
            x = layer(x)
            feats.append(x)
        out = self.decoder(*feats)
        return out


class I2VNet(nn.Module):
    def __init__(self, in_ch, out_ch, itm_ch=32, video_len=8):
        super().__init__()
        self.video_len = video_len
        if self.video_len < 2:
            raise ValueError
        self.image_stage = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            in_channels=2*in_ch,
            classes=itm_ch
        )
        self.image_head = Conv3x3(itm_ch, out_ch)
        self.conv_factor = nn.Sequential(
            Conv3x3(itm_ch+2*in_ch, itm_ch, bn=True, act=True),
            Conv3x3(itm_ch, 1),
            nn.Sigmoid()
        )
        self.video_stage = VideoSegModel(in_ch, dec_chs=(256,128,64,32))
        self.video_head = Conv3x3(32, out_ch)
    
    def forward(self, t1, t2):
        out_i = self.image_stage(torch.cat([t1,t2], dim=1))
        factor = self.conv_factor(torch.cat([out_i,t1,t2], dim=1))
        pred_i = self.image_head(out_i)
        frames = self.image_to_video(t1, t2, factor)
        out_v = self.video_stage(frames.transpose(1,2))
        pred_v = self.video_head(out_v)
        return pred_i, pred_v

    def image_to_video(self, im1, im2, factor_map):
        delta = 1.0/(self.video_len-1)
        delta_map = factor_map * delta
        steps = torch.arange(self.video_len, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)
        frames = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
        return frames