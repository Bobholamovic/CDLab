import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ._blocks import (DecBlock, ResBlock, ResBlock2, Conv3x3, MaxPool2x2, BasicConv)


class JointDecoder(nn.Module):
    def __init__(self, enc_chs_p, enc_chs_v, dec_chs):
        super().__init__()
        
        enc_chs_p = enc_chs_p[::-1]
        enc_chs_v = enc_chs_v[::-1]
        enc_chs = [ch_p+ch_v for ch_p, ch_v in zip(enc_chs_p, enc_chs_v)]

        self.convs_video = nn.ModuleList(
            [
                BasicConv(2*ch, ch, 1, bn=True, act=True)
                for ch in enc_chs_v
            ]
        )
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, out_ch)
            for in_ch1, in_ch2, out_ch in zip(enc_chs[1:], (enc_chs[0],)+dec_chs[:-1], dec_chs)
        ])
        self.conv_out = BasicConv(dec_chs[-1], 1, 1, bn=False, act=False)
    
    def forward(self, feats_p, feats_v):
        feats_p = feats_p[::-1]
        feats_v = feats_v[::-1]
        
        x = self.combine(feats_p[0], feats_v[0], self.convs_video[0])
        
        for feat_p, feat_v, conv, blk in zip(feats_p[1:], feats_v[1:], self.convs_video[1:], self.blocks):
            feat = self.combine(feat_p, feat_v, conv)
            x = blk(feat, x)

        y = self.conv_out(x)

        return y

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)

    def combine(self, feat_p, feat_v, conv):
        feat_v = self.tem_aggr(feat_v)
        feat_v = conv(feat_v)
        feat_v = F.interpolate(feat_v, size=feat_p.shape[2:])
        feat = torch.cat([feat_p, feat_v], dim=1)
        return feat


class PairEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(16,32,64)):
        super().__init__()

        self.conv1 = ResBlock(2*in_ch, enc_chs[0])
        self.pool1 = MaxPool2x2()

        self.conv2 = ResBlock(enc_chs[0], enc_chs[1])
        self.pool2 = MaxPool2x2()

        self.conv3 = ResBlock2(enc_chs[1], enc_chs[2])
        self.pool3 = MaxPool2x2()

    def forward(self, x1, x2):
        x = torch.cat([x1,x2], dim=1)
        feats = [x]
        for i in range(3):
            conv = getattr(self, f'conv{i+1}')
            x = conv(x)
            pool = getattr(self, f'pool{i+1}')
            x = pool(x)
            feats.append(x)
        return feats


class VideoEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(64,128,256)):
        super().__init__()
        if in_ch != 3 or enc_chs != (64,128,256):
            raise NotImplementedError
        self.encoder = models.video.r3d_18(pretrained=True)
        self.encoder.layer4 = nn.Identity()
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        feats = [x]
        x = self.encoder.stem(x)
        for i in range(3):
            layer = getattr(self.encoder, f'layer{i+1}')
            x = layer(x)
            feats.append(x)
        return feats
        

class P2VNet(nn.Module):
    def __init__(self, in_ch, video_len=8, enc_chs_p=(64,128,256), enc_chs_v=(64,128,256), dec_chs=(128,64,32)):
        super().__init__()
        if video_len < 2:
            raise ValueError
        self.video_len = video_len
        self.encoder_p = PairEncoder(in_ch, enc_chs=enc_chs_p)
        self.encoder_v = VideoEncoder(in_ch, enc_chs=enc_chs_v)
        self.decoder = JointDecoder((2*in_ch,*enc_chs_p), (in_ch,*enc_chs_v), dec_chs)
    
    def forward(self, t1, t2):
        feats_p = self.encoder_p(t1, t2)
        frames = self.pair_to_video(F.interpolate(t1, scale_factor=0.5), F.interpolate(t2, scale_factor=0.5))
        feats_v = self.encoder_v(frames.transpose(1,2))
        pred = self.decoder(feats_p, feats_v)
        return pred

    def pair_to_video(self, im1, im2, rate_map=None):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0/(len-1)
            delta_map = rate_map * delta
            steps = torch.arange(len, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)
            interped = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:,0:1])
        frames = _interpolate(im1, im2, rate_map, self.video_len)
        return frames