import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ._blocks import DecBlock, ResBlock, Conv3x3, BasicConv


class VideoDecoder(nn.Module):
    def __init__(self, enc_chs, dec_chs):
        super().__init__()
        if not len(enc_chs) == len(dec_chs)+1:
            raise ValueError
        enc_chs = enc_chs[::-1]

        self.convs_video = nn.ModuleList(
            [
                BasicConv(2*ch, ch, 1, bn=True, act=True)
                for ch in enc_chs
            ]
        )
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, out_ch)
            for in_ch1, in_ch2, out_ch in zip(enc_chs[1:], (enc_chs[0],)+dec_chs[:-1], dec_chs)
        ])
        self.conv_out = BasicConv(dec_chs[-1], 1, 1, bn=False, act=False)
    
    def forward(self, *feats):
        feats = feats[::-1]
        x = self.convs_video[0](self.tem_aggr(feats[0]))
        for blk, conv, f in zip(self.blocks, self.convs_video[1:], feats[1:]):
            x = blk(conv(self.tem_aggr(f)), x)
        y = self.conv_out(x)
        return y

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)


class VideoSegModel(nn.Module):
    def __init__(self, in_ch, enc_chs, dec_chs):
        super().__init__()
        # Encoder
        if in_ch != 3 and enc_chs != (64,128,256):
            raise NotImplementedError
        self.encoder = models.video.r3d_18(pretrained=True)
        self.encoder.layer4 = nn.Identity()
        self.encoder.fc = nn.Identity()
        # Decoder
        self.decoder = VideoDecoder(
            enc_chs=(in_ch,enc_chs), 
            dec_chs=dec_chs
        )

    def forward(self, x):
        feats = [x]
        x = self.encoder.stem(x)
        for i in range(3):
            layer = getattr(self.encoder, f'layer{i+1}')
            x = layer(x)
            feats.append(x)
        out = self.decoder(*feats)
        return out
        

class P2VNet(nn.Module):
    def __init__(self, in_ch, video_len=8, enc_chs=(64,128,256), dec_chs=(128,64,32)):
        super().__init__()
        if video_len < 2:
            raise ValueError
        self.video_len = video_len
        self.seg_video = VideoSegModel(
            in_ch, 
            enc_chs=enc_chs,
            dec_chs=dec_chs
        )
    
    def forward(self, t1, t2):
        frames = self.pair_to_video(t1, t2)
        pred = self.seg_video(frames.transpose(1,2))
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