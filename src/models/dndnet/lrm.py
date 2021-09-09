# Thanks to Hongyu Chen for providing the LRM implementation

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LRCWH_NoBN(nn.Module):
    def __init__(self, c, k, ratio = 8):
        super(LRCWH_NoBN, self).__init__()

        self.conv1 = nn.Conv3d(c, c, 1)
        if ratio == 8:
            self.conv2 = nn.Sequential(nn.Conv3d(c,c,(5,1,1),stride= (2,1,1),padding=(2,0,0)),
                                        nn.ReLU(),
                                        nn.Conv3d(c,c,(5,1,1),stride= (2,1,1),padding=(2,0,0)),
                                        nn.ReLU(),
                                        nn.Conv3d(c,c,(5,1,1),stride= (2,1,1),padding=(2,0,0)),)
            self.conv3 = nn.Sequential(nn.Conv3d(c,c,(5,1,1),stride= (2,1,1),padding=(2,0,0)),
                                        nn.ReLU(),
                                        nn.Conv3d(c,c,(5,1,1),stride= (2,1,1),padding=(2,0,0)),
                                        nn.ReLU(),
                                        nn.Conv3d(c,c,(5,1,1),stride= (2,1,1),padding=(2,0,0)),)
        elif ratio == 4:
            self.conv2 = nn.Sequential(nn.Conv3d(c,c,(5,1,1),stride= (2,1,1),padding=(2,0,0)),
                                        nn.ReLU(),
                                        nn.Conv3d(c,c,(5,1,1),stride= (2,1,1),padding=(2,0,0)),
                                        nn.ReLU(),)
            self.conv3 = nn.Sequential(nn.Conv3d(c,c,(5,1,1),stride= (2,1,1),padding=(2,0,0)),
                                        nn.ReLU(),
                                        nn.Conv3d(c,c,(5,1,1),stride= (2,1,1),padding=(2,0,0)),
                                        nn.ReLU(),)      

        self.convn = nn.Conv3d(c, c, 1, bias=False)     
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
 
    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
 
        # The Low Rank decomposition
        b, m, c, h, w = x1.size()  # b * m * c * h * w
        _, _, k, _, _ = x2.size()  # b * m * k * h * w
        x1 = x1.view(b*m, c, h*w)  # bm * c * hw 
        x2 = x2.view(b*m, k, h*w).permute(0,2,1)  # bm * hw * k
        x3 = x3.view(b*m, k, h*w)  # bm * k * hw
        x3 = F.softmax(x3, dim=1)
        # x3_ = x3 / (1e-6 + x3.sum(dim=1, keepdim=True))      warining!!!!!!
        
        x1 = torch.bmm(x1, x2) # bm * c * k
        x1 = self._l2norm(x1, dim=1)

        x3 = torch.bmm(x1, x3) # bm * c * hw   x3_
        x3 = x3.view(b, m, c, h, w)              # b * c * h * w

        # The second 1x1 conv
        x3 = self.convn(x3)
        x3 = x3 + idn                 #   warning ~!!!!!!!!!!!!
        x3 = F.relu(x3, inplace=True)
        return x3            #

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))