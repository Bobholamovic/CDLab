# Adapted from https://github.com/Bobholamovic/ESCNet/blob/main/escnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (Conv3x3, MaxPool2x2, ResBlock, ResBlock2, DecBlock)
from .ssn import SSN


class RefineNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        C = [in_ch, 32, 64, 128]

        # Fusion layers
        self.fuse1 = Conv3x3(in_ch+C[0], in_ch, bn=True, act=True)
        self.fuse2 = Conv3x3(in_ch+C[1], in_ch, bn=True, act=True)
        self.fuse3 = Conv3x3(in_ch+C[2], in_ch, bn=True, act=True)
        self.fuse4 = Conv3x3(in_ch+C[3], in_ch, bn=True, act=True)

        self.conv_out = nn.Sequential(
            Conv3x3(in_ch, in_ch, bn=True, act=True),
            Conv3x3(in_ch, out_ch, bn=False, act=False)
        )

    def forward(self, x, feats_to_fuse):
        y = x + self.fuse1(torch.cat([x, feats_to_fuse[0]], dim=1))
        interp_configs = dict(size=x.shape[2:], mode='bilinear', align_corners=True)
        y = y + self.fuse2(torch.cat([x, F.interpolate(feats_to_fuse[1], **interp_configs)], dim=1))
        y = y + self.fuse3(torch.cat([x, F.interpolate(feats_to_fuse[2], **interp_configs)], dim=1))
        y = y + self.fuse4(torch.cat([x, F.interpolate(feats_to_fuse[3], **interp_configs)], dim=1))

        return self.conv_out(y)


class SiamUNet_diff(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        C = [32, 64, 128, 256]

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

    def forward(self, t1, t2, merge=False):
        # Encode t1
        # Stage 1
        x1_1 = self.conv1(t1)
        xp = self.pool1(x1_1)

        # Stage 2
        x2_1 = self.conv2(xp)
        xp = self.pool2(x2_1)

        # Stage 3
        x3_1 = self.conv3(xp)
        xp = self.pool3(x3_1)

        # Stage 4
        x4_1 = self.conv4(xp)
        xp_1 = self.pool4(x4_1)

        # Encode t2
        # Stage 1
        x1_2 = self.conv1(t2)
        xp = self.pool1(x1_2)

        # Stage 2
        x2_2 = self.conv2(xp)
        xp = self.pool2(x2_2)

        # Stage 3
        x3_2 = self.conv3(xp)
        xp = self.pool3(x3_2)

        # Stage 4
        x4_2 = self.conv4(xp)
        xp_2 = self.pool4(x4_2)

        # Decode t2
        # Stage 4d
        xd4 = self.conv4d(torch.abs(x4_1-x4_2), xp_2)

        # Stage 3d
        xd3 = self.conv3d(torch.abs(x3_1-x3_2), xd4)

        # Stage 2d
        xd2 = self.conv2d(torch.abs(x2_1-x2_2), xd3)

        # Stage 1d
        xd1 = self.conv1d(torch.abs(x1_1-x1_2), xd2)
        
        return xd1, xd2, xd3, xd4


class ESCNet(nn.Module):
    def __init__(
        self, 
        feat_cvrter,
        n_iters=10, 
        n_spixels=100,
        n_filters=64, in_ch=8, out_ch=16,
        alpha=0.01
    ):
        super().__init__()
        self.ssn = SSN(feat_cvrter, n_iters, n_spixels, n_filters, in_ch, out_ch, cnn=True)
        self.cd_net = SiamUNet_diff(out_ch-2, out_ch)
        self.conv_ds = nn.Sequential(
            Conv3x3(out_ch, 2, bn=False, act=False)
        )
        self.fuse_net = RefineNet(out_ch, 2)
        self.omega2 = (alpha*n_spixels)**2

    def forward(self, f1, f2, merge=False):
        # Compute Qs
        Q1, ops1, f1, spf1, pf1 = self.ssn(f1)
        Q2, ops2, f2, spf2, pf2 = self.ssn(f2)

        Q1_d, Q2_d = Q1.detach(), Q2.detach()

        # Extract pixel-level features
        # pf means pixel features and hf means hidden-layer features
        hf = self.cd_net(pf1[:,2:], pf2[:,2:])
        pf = hf[0]

        # Super-pixelation
        if merge:
            # Adaptive superpixel merging
            b, c, s = spf1.size()
            
            spf1.detach_()
            rels = spf1.unsqueeze(-2) - spf1.unsqueeze(-1)
            rels = torch.exp(-(rels**2).sum(dim=1, keepdim=True)/self.omega2)
            coeffs = ops1['map_sp2p'](rels.view(b, s, s), Q1_d).view(b, s, -1)
            spf1 = torch.matmul(pf.view(b, c, -1), coeffs.transpose(1,2)) / (coeffs.unsqueeze(1).sum(-1)+1e-32)

            spf2.detach_()
            rels = spf2.unsqueeze(-2) - spf2.unsqueeze(-1)
            rels = torch.exp(-(rels**2).sum(dim=1, keepdim=True)/self.omega2)
            coeffs = ops2['map_sp2p'](rels.view(b, s, s), Q2_d).view(b, s, -1)
            spf2 = torch.matmul(pf.view(b, c, -1), coeffs.transpose(1,2)) / (coeffs.unsqueeze(1).sum(-1)+1e-32)
            
            del rels, coeffs
        else:
            spf1 = ops1['map_p2sp'](pf, Q1_d)
            spf2 = ops2['map_p2sp'](pf, Q2_d)
        
        pf1 = ops1['map_sp2p'](spf1, Q1_d)
        pf2 = ops2['map_sp2p'](spf2, Q2_d)
        pf_sp = pf1 + pf2
        prob_ds = self.conv_ds(pf_sp)
        
        # Pixel-level refinement
        pf_out = self.fuse_net(pf_sp, hf)
        
        prob = pf_out
        
        return prob, prob_ds, (Q1,Q2), (ops1,ops2), (f1,f2)


class ESCNet_Pixel(ESCNet):
    def forward(self, f1, f2, merge=False):
        # Compute Qs
        Q1, ops1, f1, spf1, pf1 = self.ssn(f1)
        Q2, ops2, f2, spf2, pf2 = self.ssn(f2)

        Q1_d, Q2_d = Q1.detach(), Q2.detach()

        # Extract pixel-level features
        # pf means pixel features and hf means hidden-layer features
        hf = self.cd_net(pf1[:,2:], pf2[:,2:])
        pf = hf[0]

        prob_ds = self.conv_ds(pf)
        
        # Pixel-level refinement
        pf_out = self.fuse_net(pf, hf)
        
        prob = pf_out
        
        return prob, prob_ds, (Q1,Q2), (ops1,ops2), (f1,f2)


class ESCNet_Detach(ESCNet):
    def forward(self, f1, f2, merge=False):
        # Compute Qs
        Q1, ops1, f1, spf1, pf1 = self.ssn(f1)
        Q2, ops2, f2, spf2, pf2 = self.ssn(f2)

        Q1_d, Q2_d = Q1.detach(), Q2.detach()

        # Extract pixel-level features
        # pf means pixel features and hf means hidden-layer features
        hf = self.cd_net(pf1[:,2:].detach(), pf2[:,2:].detach())
        pf = hf[0]

        # Super-pixelation
        if merge:
            # Adaptive superpixel merging
            b, c, s = spf1.size()
            
            spf1.detach_()
            rels = spf1.unsqueeze(-2) - spf1.unsqueeze(-1)
            rels = torch.exp(-(rels**2).sum(dim=1, keepdim=True)/self.omega2)
            coeffs = ops1['map_sp2p'](rels.view(b, s, s), Q1_d).view(b, s, -1)
            spf1 = torch.matmul(pf.view(b, c, -1), coeffs.transpose(1,2)) / (coeffs.unsqueeze(1).sum(-1)+1e-32)

            spf2.detach_()
            rels = spf2.unsqueeze(-2) - spf2.unsqueeze(-1)
            rels = torch.exp(-(rels**2).sum(dim=1, keepdim=True)/self.omega2)
            coeffs = ops2['map_sp2p'](rels.view(b, s, s), Q2_d).view(b, s, -1)
            spf2 = torch.matmul(pf.view(b, c, -1), coeffs.transpose(1,2)) / (coeffs.unsqueeze(1).sum(-1)+1e-32)
            
            del rels, coeffs
        else:
            spf1 = ops1['map_p2sp'](pf, Q1_d)
            spf2 = ops2['map_p2sp'](pf, Q2_d)
        
        pf1 = ops1['map_sp2p'](spf1, Q1_d)
        pf2 = ops2['map_sp2p'](spf2, Q2_d)
        pf_sp = pf1 + pf2
        prob_ds = self.conv_ds(pf_sp)
        
        # Pixel-level refinement
        pf_out = self.fuse_net(pf_sp, hf)
        
        prob = pf_out
        
        return prob, prob_ds, (Q1,Q2), (ops1,ops2), (f1,f2)