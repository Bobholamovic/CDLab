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

        self.conv_video = nn.ModuleList(
            [
                BasicConv(tem_len*ch, ch, 1, bn=True, act=True)
                for tem_len, ch in zip(tem_lens, enc_chs)
            ]
        )
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, out_ch)
            for in_ch1, in_ch2, out_ch in zip(enc_chs[1:], (enc_chs[0],)+dec_chs[:-1], dec_chs)
        ])
        self.conv_out = BasicConv(dec_chs[-1], 1, 1)
    
    def forward(self, *feats):
        feats = feats[::-1]
        x = self.conv_video[0](self.flatten_video(feats[0]))
        for blk, conv, f in zip(self.blocks, self.conv_video[1:], feats[1:]):
            x = blk(conv(self.flatten_video(f)), x)
        y = self.conv_out(x)
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
        tem_lens=tuple(int(video_len*s) for s in (1,1,0.5,0.25))
        self.decoder = VideoDecoder(
            enc_chs=enc_chs, 
            dec_chs=dec_chs, 
            tem_lens=tem_lens
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


class SigmoidBeta(nn.Sigmoid):
    def __init__(self, beta, *args, **kwargs):
        self.beta = beta
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(self.beta*x)


class P2VNet(nn.Module):
    def __init__(self, in_ch, video_len=8, beta=0.5):
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
    
    def forward(self, t1, t2, k=1):
        preds = []
        for iter in range(k):
            if iter == 0:
                rate_map = torch.ones_like(t1[:,0:1])
            else:
                rate_map = self.act_rate(pred.detach())
            frames = self.pair_to_video(t1, t2, rate_map)
            pred = self.seg_video(frames.transpose(1,2))
            preds.append(pred)
        return preds

    def pair_to_video(self, im1, im2, rate_map):
        delta = 1.0/(self.video_len-1)
        delta_map = rate_map * delta
        steps = torch.arange(self.video_len, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)
        frames = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
        return frames