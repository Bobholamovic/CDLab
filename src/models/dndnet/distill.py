import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Distill(nn.Module):
    def __init__(self, c, k, p=8):
        super(Distill, self).__init__()

        self.conv_v = nn.Conv2d(2*c, k, 1)

        self.conv1_1 = nn.Conv2d(c, c, 1)
        self.conv2_1 = nn.Conv2d(c, k, 1)
        self.conv_out_1 = nn.Conv2d(c, c, 1, bias=False)

        self.conv1_2 = nn.Conv2d(c, c, 1)
        self.conv2_2 = nn.Conv2d(c, k, 1)
        self.conv_out_2 = nn.Conv2d(c, c, 1, bias=False)

        self.k = k
        self.p = p
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
 
    def forward(self, x1, x2):
        b, c, h, w = x1.shape

        x1 = self._image_to_patches(x1)
        y1_1 = self.conv1_1(x1)
        y2_1 = self.conv2_1(x1)

        x2 = self._image_to_patches(x2)
        y1_2 = self.conv1_2(x2)
        y2_2 = self.conv2_2(x2)

        v = self.conv_v(torch.cat([x1,x2], dim=1))

        k = self.k
        pp = self.p*self.p
        bpp = b*pp

        y1_1 = y1_1.permute((0,2,1,3)).reshape(bpp, c, -1)
        y1_2 = y1_2.permute((0,2,1,3)).reshape(bpp, c, -1)
        y2_1 = y2_1.permute((0,2,3,1)).reshape(bpp, -1, k)
        y2_2 = y2_2.permute((0,2,3,1)).reshape(bpp, -1, k)

        v = v.permute((0,2,1,3)).reshape(bpp, k, -1)
        v = F.softmax(v, dim=1)
        
        u1 = torch.bmm(y1_1, y2_1)
        u1 = self._l2norm(u1, dim=1)
        u2 = torch.bmm(y1_2, y2_2)
        u2 = self._l2norm(u2, dim=1)
        
        y3_1 = torch.bmm(u1, v)
        y3_2 = torch.bmm(u2, v)

        y3_1 = self._patches_to_image(y3_1.view(b,pp,c,-1).permute((0,2,1,3)), h, w)
        y3_1 = self.conv_out_1(y3_1)
        y3_2 = self._patches_to_image(y3_2.view(b,pp,c,-1).permute((0,2,1,3)), h, w)
        y3_2 = self.conv_out_2(y3_2)
        
        return y3_1, y3_2

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def _image_to_patches(self, x):
        # In: (b,c,h,w)
        # Out: (b,c,pp,hw/pp)
        p = self.p
        b, c, h, w = x.shape
        return x.view(b, c, p, h//p, p, w//p).permute((0,1,2,4,3,5)).reshape(b,c,p*p,-1)

    def _patches_to_image(self, x, h, w):
        # In: (b,c,pp,hw/pp)
        # Out: (b,c,h,w)
        p = self.p
        b, c = x.shape[:2]
        return x.reshape(b, c, p, p, h//p, w//p).permute((0,1,2,4,3,5)).reshape(b,c,h,w)