import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Distill(nn.Module):
    def __init__(self, c, k, p=8):
        super(Distill, self).__init__()

        self.conv_u_1 = nn.Conv2d(c, k, 1)
        self.conv_u_2 = nn.Conv2d(c, k, 1)

        self.conv_v_left = nn.Conv2d(2*c, c, 1)
        self.conv_v_right = nn.Conv2d(2*c, k, 1)

        self.conv_out_1 = nn.Conv2d(c, c, 1, bias=False)
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
        u1 = self.conv_u_1(x1)

        x2 = self._image_to_patches(x2)
        u2 = self.conv_u_2(x2)

        x = torch.cat([x1,x2], dim=1)
        left = self.conv_v_left(x)
        right = self.conv_v_right(x)

        k = self.k
        pp = self.p*self.p
        bpp = b*pp

        left = left.permute((0,2,1,3)).reshape(bpp, c, -1)
        right = right.permute((0,2,3,1)).reshape(bpp, -1, k)

        u1 = u1.permute((0,2,1,3)).reshape(bpp, k, -1)
        u1 = F.softmax(u1, dim=1)
        u2 = u2.permute((0,2,1,3)).reshape(bpp, k, -1)
        u2 = F.softmax(u2, dim=1)
        
        v = torch.bmm(left, right)
        v = self._l2norm(v, dim=1)
        
        y3_1 = torch.bmm(v, u1)
        y3_2 = torch.bmm(v, u2)

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