# Implementation of
# Papadomanolaki Maria, Vakalopoulou Maria, and Karantzalos Konstantinos, “A Deep Multitask Learning Framework Coupling Semantic Segmentation and Fully Convolutional LSTM Networks for Urban Change Detection,” IEEE Trans. Geosci. Remote Sensing, vol. 59, no. 9, pp. 7651–7668, 2021, doi: 10.1109/TGRS.2021.3055584.

# Adapted from https://github.com/mpapadomanolaki/multi-task-L-UNet/blob/main/code/network.py
# Different from the original paper, the multi-task branch was not implemented.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._blocks import Conv1x1, Conv3x3, MaxPool2x2


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            Conv3x3(in_ch, out_ch, norm=True, act=True),
            Conv3x3(out_ch, out_ch, norm=True, act=True)
        )


class Up(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Upsample(scale_factor=2),
            Conv3x3(in_ch, out_ch, norm=True, act=True),
        )


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.in_gate = Conv3x3(input_size+hidden_size, hidden_size)
        self.forget_gate = Conv3x3(input_size+hidden_size, hidden_size)
        self.out_gate = Conv3x3(input_size+hidden_size, hidden_size)
        self.cell_gate = Conv3x3(input_size+hidden_size, hidden_size)

    def forward(self, x, h_state, c_state):
        x = torch.cat([x, h_state], 1)

        in_gate = F.sigmoid(self.in_gate(x))
        forget_gate = F.sigmoid(self.forget_gate(x))
        out_gate = F.sigmoid(self.out_gate(x))
        cell_gate = F.tanh(self.cell_gate(x))

        c_state = (forget_gate * c_state) + (in_gate * cell_gate)
        h_state = out_gate * F.tanh(c_state)

        return h_state, c_state


class RNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, seq):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq = seq
        self.rnn_cell = RNNCell(input_size, hidden_size)

    def forward(self, x):
        n = x.shape[1]
        out_list = []
        h_state, c_state = None, None

        for t in range(n):
            input_t = self.seq(x[:,t])
            out_list.append(input_t)
            if h_state is None:
                b, _, h, w = input_t.shape
                h_state = torch.zeros((b,self.hidden_size,h,w)).to(input_t.device)
                c_state = torch.zeros((b,self.hidden_size,h,w)).to(input_t.device)
            h_state, c_state = self.rnn_cell(input_t, h_state, c_state)

        return h_state, torch.stack(out_list, dim=1)


class LUNet(nn.Module):
    def __init__(self, in_ch, out_ch, video_len=2):
        super().__init__()
        
        if video_len < 2:
            raise ValueError
        self.video_len = video_len

        self.block1 = RNNBlock(
            16, 16,
            ConvBlock(in_ch, 16)
        )
        self.block2 = RNNBlock(
            32, 32,
            nn.Sequential(MaxPool2x2(), ConvBlock(16, 32))
        )
        self.block3 = RNNBlock(
            64, 64,
            nn.Sequential(MaxPool2x2(), ConvBlock(32, 64))
        )
        self.block4 = RNNBlock(
            128, 128,
            nn.Sequential(MaxPool2x2(), ConvBlock(64, 128))
        )

        self.block5 = RNNBlock(
            256, 256,
            nn.Sequential(MaxPool2x2(), ConvBlock(128, 256))
        )

        self.up5 = Up(256, 128)
        self.conv5d = ConvBlock(256, 128)

        self.up4 = Up(128, 64)
        self.conv4d = ConvBlock(128, 64)

        self.up3 = Up(64, 32)
        self.conv3d = ConvBlock(64, 32)

        self.up2 = Up(32, 16)
        self.conv2d = ConvBlock(32, 16)

        self.conv_out = Conv1x1(16, out_ch)

    def encode(self, x):
        x1, y = self.block1(x)
        x2, y = self.block2(y)
        x3, y = self.block3(y)
        x4, y = self.block4(y)
        x5, y = self.block5(y)
        return x1, x2, x3, x4, x5

    def decode(self, x1, x2, x3, x4, x5):
        d5 = self.up5(x5)
        d5 = torch.cat([d5,x4], 1)
        d5 = self.conv5d(d5)

        d4 = self.up4(d5)
        d4 = torch.cat([d4,x3], 1)
        d4 = self.conv4d(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3,x2], 1)
        d3 = self.conv3d(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2,x1], 1)
        d2 = self.conv2d(d2)

        d1 = self.conv_out(d2)

        return d1

    def forward(self, t1, t2):
        input = self.pair_to_video(t1, t2)

        x1,x2,x3,x4,x5 = self.encode(input)
        out = self.decode(x1,x2,x3,x4,x5)

        return out

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