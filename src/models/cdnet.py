# Implementation of 
# Alcantarilla, P. F., Stent, S., Ros, G., Arroyo, R., & Gherardi, R. (2018). Street-view change detection with deconvolutional networks. Autonomous Robots, 42(7), 1301–1322.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._blocks import Conv7x7, MaxPool2x2, MaxUnPool2x2


class CDNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = Conv7x7(in_ch, 64, norm=True, act=True)
        self.pool1 = MaxPool2x2(return_indices=True)
        self.conv2 = Conv7x7(64, 64, norm=True, act=True)
        self.pool2 = MaxPool2x2(return_indices=True)
        self.conv3 = Conv7x7(64, 64, norm=True, act=True)
        self.pool3 = MaxPool2x2(return_indices=True)
        self.conv4 = Conv7x7(64, 64, norm=True, act=True)
        self.pool4 = MaxPool2x2(return_indices=True)
        self.conv5 = Conv7x7(64, 64, norm=True, act=True)
        self.upool4 = MaxUnPool2x2()
        self.conv6 = Conv7x7(64, 64, norm=True, act=True)
        self.upool3 = MaxUnPool2x2()
        self.conv7 = Conv7x7(64, 64, norm=True, act=True)
        self.upool2 = MaxUnPool2x2()
        self.conv8 = Conv7x7(64, 64, norm=True, act=True)
        self.upool1 = MaxUnPool2x2()
        
        self.conv_out = Conv7x7(64, out_ch, norm=False, act=False)
    
    def forward(self, t1, t2):
        # Concatenation
        x = torch.cat([t1, t2], dim=1)

        # Contraction
        x, ind1 = self.pool1(self.conv1(x))
        x, ind2 = self.pool2(self.conv2(x))
        x, ind3 = self.pool3(self.conv3(x))
        x, ind4 = self.pool4(self.conv4(x))

        # Expansion
        x = self.conv5(self.upool4(x, ind4))
        x = self.conv6(self.upool3(x, ind3))
        x = self.conv7(self.upool2(x, ind2))
        x = self.conv8(self.upool1(x, ind1))

        # Out
        return self.conv_out(x)
