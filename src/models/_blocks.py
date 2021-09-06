import torch
import torch.nn as nn
import torch.nn.functional as F


def relu():
    return nn.ReLU(inplace=True)


class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pad='zero', bn=False, act=False, **kwargs):
        super().__init__()
        self.seq = nn.Sequential()
        if kernel>=2:
            self.seq.add_module('_pad', getattr(nn, pad.capitalize()+'Pad2d')(kernel//2))
        self.seq.add_module('_conv', nn.Conv2d(
            in_ch, out_ch, kernel,
            stride=1, padding=0,
            bias=not bn,
            **kwargs
        ))
        if bn:
            self.seq.add_module('_bn', nn.BatchNorm2d(out_ch))
        if act:
            self.seq.add_module('_act', relu())

    def forward(self, x):
        return self.seq(x)


class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, pad='zero', bn=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, pad=pad, bn=bn, act=act, **kwargs)


class MaxPool2x2(nn.MaxPool2d):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, bn=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, bn=True, act=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, bn=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, bn=True, act=False)
    
    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))


class ResBlock2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, bn=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, bn=True, act=True)
        self.conv3 = Conv3x3(out_ch, out_ch, bn=True, act=False)
    
    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv3(self.conv2(x)))


class DecBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, bn=True, act=True):
        super().__init__()
        self.deconv =  nn.ConvTranspose2d(in_ch2, in_ch2, kernel_size=2, padding=0, stride=2)
        self.conv_feat = ResBlock(in_ch1+in_ch2, in_ch2)
        self.conv_out = Conv3x3(in_ch2, out_ch, bn=bn, act=act)

    def forward(self, x1, x2):
        x2 = self.deconv(x2)
        pl = 0
        pr = x1.size(3)-x2.size(3)
        pt = 0
        pb = (x1.size(2)-x2.size(2))
        x2 = F.pad(x2, (pl, pr, pt, pb), 'replicate')
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_feat(x)
        return self.conv_out(x)