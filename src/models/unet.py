# Implementation of
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.
# with modifications (remove dropout layers, add residual connections, double number of channels, and change decoding blocks).

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._blocks import Conv3x3, MaxPool2x2, ResBlock, ResBlock2, DecBlock


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        C = [32, 64, 128, 256, 512]

        self.conv1 = ResBlock(in_ch, C[0])
        self.pool1 = MaxPool2x2()

        self.conv2 = ResBlock(C[0], C[1])
        self.pool2 = MaxPool2x2()

        self.conv3 = ResBlock2(C[1], C[2])
        self.pool3 = MaxPool2x2()

        self.conv4 = ResBlock2(C[2], C[3])
        self.pool4 = MaxPool2x2()

        self.conv4d = DecBlock(C[3], C[3], C[2])

        self.conv3d = DecBlock(C[2], C[2], C[1])

        self.conv2d = DecBlock(C[1], C[1], C[0])

        self.conv1d = DecBlock(C[0], C[0], out_ch, bn=False, act=False)

        self.act_out = nn.LogSoftmax(dim=1)

    def forward(self, t1, t2):
        # Encode branch
        # Stage 1
        x1 = self.conv1(torch.cat([t1,t2], dim=1))
        xp = self.pool1(x1)

        # Stage 2
        x2 = self.conv2(xp)
        xp = self.pool2(x2)

        # Stage 3
        x3 = self.conv3(xp)
        xp = self.pool3(x3)

        # Stage 4
        x4 = self.conv4(xp)
        xp = self.pool4(x4)

        # Decode
        # Stage 4d
        xd = self.conv4d(x4, xp)

        # Stage 3d
        xd = self.conv3d(x3, xd)

        # Stage 2d
        xd = self.conv2d(x2, xd)

        # Stage 1d
        xd = self.conv1d(x1, xd)

        return self.act_out(xd)