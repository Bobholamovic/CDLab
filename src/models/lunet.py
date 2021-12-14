# Adapted from https://github.com/mpapadomanolaki/multi-task-L-UNet/blob/main/code/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.in_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.forget_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.cell_gate = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)

    def forward(self, input, h_state, c_state):

        conc_inputs = torch.cat( (input, h_state), 1)

        in_gate = self.in_gate(conc_inputs)
        forget_gate = self.forget_gate(conc_inputs)
        out_gate = self.out_gate(conc_inputs)
        cell_gate = self.cell_gate(conc_inputs)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        c_state = (forget_gate * c_state) + (in_gate * cell_gate)
        h_state = out_gate * torch.tanh(c_state)

        return h_state, c_state


class set_values(nn.Module):
    def __init__(self, hidden_size, height, width):
            super(set_values, self).__init__()
            self.hidden_size=int(hidden_size)
            self.height=int(height)
            self.width=int(width)
            self.dropout = nn.Dropout(0.7)
            self.RCell = RNNCell(self.hidden_size, self.hidden_size)


    def forward(self, seq, xinp):
        xout = torch.zeros(xinp.size()[0], xinp.size()[1], self.hidden_size, self.height, self.width).to(xinp.device)

        h_state, c_state = ( torch.zeros(xinp[0].shape[0], self.hidden_size, self.height, self.width).to(xinp.device),
                             torch.zeros(xinp[0].shape[0], self.hidden_size, self.height, self.width).to(xinp.device) )

        for t in range(xinp.size()[0]):
            input_t = seq(xinp[t])
            xout[t] = input_t
            h_state, c_state = self.RCell(input_t, h_state, c_state)

        return h_state, xout


class U_Net(nn.Module):
    def __init__(self,img_ch, output_ch, patch_size, video_len=2):
        super(U_Net,self).__init__()
        
        if video_len < 2:
            raise ValueError
        self.video_len = video_len

        self.patch_size = patch_size
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=16)
        self.set1 = set_values(16, self.patch_size, self.patch_size)

        self.Conv2 = conv_block(ch_in=16,ch_out=32)
        self.set2 = set_values(32, self.patch_size/2, self.patch_size/2)

        self.Conv3 = conv_block(ch_in=32,ch_out=64)
        self.set3 = set_values(64, self.patch_size/4, self.patch_size/4)

        self.Conv4 = conv_block(ch_in=64,ch_out=128)
        self.set4 = set_values(128, self.patch_size/8, self.patch_size/8)

        self.Conv5 = conv_block(ch_in=128,ch_out=256)
        self.set5 = set_values(256, self.patch_size/16, self.patch_size/16)

        self.Up5 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)

        self.Up4 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)

        self.Up3 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)

        self.Up2 = up_conv(ch_in=32,ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16,output_ch,kernel_size=1,stride=1,padding=0)


    def encoder(self, x):
        x1, xout = self.set1(self.Conv1, x)

        x2, xout = self.set2( nn.Sequential(self.Maxpool, self.Conv2), xout)

        x3, xout = self.set3( nn.Sequential(self.Maxpool, self.Conv3), xout)
 
        x4, xout = self.set4( nn.Sequential(self.Maxpool, self.Conv4), xout)

        x5, xout = self.set5( nn.Sequential(self.Maxpool, self.Conv5), xout)

        return x1,x2,x3,x4,x5


    def decoder_lstm(self, x1, x2, x3, x4, x5):
        d5 = self.Up5(x5)
        d5 = torch.cat((d5,x4),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((d4,x3),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3,x2),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2,x1),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

    def forward(self, t1, t2):
        input = self.pair_to_video(t1, t2)
        input = input.transpose(0,1)   # (t,b,c,h,w)
        
        x1,x2,x3,x4,x5 = self.encoder(input)

        d1 = self.decoder_lstm(x1,x2,x3,x4,x5)
        return d1

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