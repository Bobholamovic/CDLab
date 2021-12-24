import torch
import torch.nn as nn
import torch.nn.functional as F


class CCLoss(nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, reduction='none')

    def forward(self, cm, gen, ori):
        return ((1-cm.unsqueeze(1).float()) * super().forward(gen, ori)).mean()


class CompactnessLoss(nn.MSELoss):
    def forward(self, posclr_s, posclr):
        return super().forward(posclr_s[:,:2], posclr[:,:2].detach())