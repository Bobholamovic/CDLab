import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ._blocks import DecBlock, ResBlock, Conv3x3, BasicConv


class VideoDecoder(nn.Module):
    def __init__(self, enc_chs, dec_chs, tem_lens):
        super().__init__()
        if not len(enc_chs) == len(dec_chs)+1:
            raise ValueError
        enc_chs = enc_chs[::-1]
        tem_lens = tem_lens[::-1]

        self.convs_video = nn.ModuleList(
            [
                BasicConv(tem_len*ch, ch, 1, bn=True, act=True)
                for tem_len, ch in zip(tem_lens, enc_chs)
            ]
        )
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, out_ch)
            for in_ch1, in_ch2, out_ch in zip(enc_chs[1:], (enc_chs[0],)+dec_chs[:-1], dec_chs)
        ])
        self.convs_out = nn.ModuleList([
            nn.Sequential(
                BasicConv(dec_ch, 32, 1, bn=False, act=True),
                BasicConv(32, 1, 1, bn=False, act=False)
            )
            for dec_ch in dec_chs
        ])
    
    def forward(self, *feats, l):
        feats = feats[::-1]
        x = self.convs_video[0](self.flatten_video(feats[0]))
        for blk, conv, f in zip(self.blocks[:l], self.convs_video[1:l+1], feats[1:l+1]):
            x = blk(conv(self.flatten_video(f)), x)
        y = self.convs_out[l-1](x)
        return y

    def flatten_video(self, f):
        return f.flatten(1,2)


class VideoSegModel(nn.Module):
    def __init__(self, in_ch, dec_chs, video_len):
        super().__init__()
        if in_ch != 3:
            raise ValueError
        # Encoder
        self.encoder = models.video.r3d_18(pretrained=True)
        self.encoder.layer4 = nn.Identity()
        self.encoder.fc = nn.Identity()
        # Decoder
        enc_chs=(in_ch,64,128,256)
        tem_lens=tuple(int(video_len*s) for s in (1.0,1.0,0.5,0.25))
        self.decoder = VideoDecoder(
            enc_chs=enc_chs, 
            dec_chs=dec_chs, 
            tem_lens=tem_lens
        )

    def forward(self, x, l):
        feats = [x]
        x = self.encoder.stem(x)
        for i in range(3):
            layer = getattr(self.encoder, f'layer{i+1}')
            x = layer(x)
            feats.append(x)
        out = self.decoder(*feats, l=l)
        return out


class SigmoidBeta(nn.Sigmoid):
    def __init__(self, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def forward(self, x):
        return super().forward(self.beta*x)


class P2VNet(nn.Module):
    def __init__(self, in_ch, video_len=8, beta=0.5, k=3):
        super().__init__()
        self.video_len = video_len
        if self.video_len < 2:
            raise ValueError
        self.seg_video = VideoSegModel(
            in_ch, 
            dec_chs=(128,64,32), 
            video_len=video_len
        )
        self.act_rate = SigmoidBeta(beta)
        if k > 3 or k < 1 or video_len>>(k-1)<=1:
            raise ValueError
        self.k = k
    
    def forward(self, t1, t2):
        preds = []
        rate_map = None
        frames = None
        for iter in range(self.k):
            frames = self.pair_to_video(t1, t2, rate_map, frames, self.video_len>>(self.k-iter))
            pred = self.seg_video(frames.transpose(1,2), 4-self.k+iter)
            pred = F.interpolate(pred, size=t1.shape[2:])
            preds.append(pred)
            rate_map = self.act_rate(pred.detach())
        return preds

    def pair_to_video(self, im1, im2, rate_map, old_frames, shift):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0/(len-1)
            delta_map = rate_map * delta
            steps = torch.arange(len, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)
            interped = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:,0:1])
        if old_frames is None:
            frames = _interpolate(im1, im2, rate_map, self.video_len)
        else:
            interped = _interpolate(old_frames[:,shift], im2, rate_map, self.video_len-shift)
            frames = torch.cat((old_frames[:,:shift], interped), dim=1)
        return frames