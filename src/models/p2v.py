import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self.seq.add_module('_act', nn.ReLU(True))

    def forward(self, x):
        return self.seq(x)


class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, pad='zero', bn=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, pad=pad, bn=bn, act=act, **kwargs)


class MaxPool2x2(nn.MaxPool2d):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)


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
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv_fuse = ResBlock(in_ch1+in_ch2, out_ch)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x = torch.cat([x1, x2], dim=1)
        return self.conv_fuse(x)


class BasicConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, bn=False, act=False, **kwargs):
        super().__init__()
        self.seq = nn.Sequential()
        self.seq.add_module('_conv', nn.Conv3d(
            in_ch, out_ch, kernel,
            padding=kernel//2,
            bias=not bn,
            **kwargs
        ))
        if bn:
            self.seq.add_module('_bn', nn.BatchNorm3d(out_ch))
        if act:
            self.seq.add_module('_act', nn.ReLU(True))

    def forward(self, x):
        return self.seq(x)


class Conv3x3x3(BasicConv3D):
    def __init__(self, in_ch, out_ch, bn=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, bn=bn, act=act, **kwargs)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, itm_ch, stride=1, ds=None):
        super().__init__()
        self.conv1 = BasicConv3D(in_ch, itm_ch, 1, bn=True, act=True, stride=stride)
        self.conv2 = Conv3x3x3(itm_ch, itm_ch, bn=True, act=True)
        self.conv3 = BasicConv3D(itm_ch, out_ch, 1, bn=True, act=False)
        self.ds = ds
        
    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.ds is not None:
            res = self.ds(res)
        y = F.relu(y+res)
        return y


class PairEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(16,32,64), add_chs=(0,0)):
        super().__init__()

        self.num_layers = 3

        self.conv1 = ResBlock(2*in_ch, enc_chs[0])
        self.pool1 = MaxPool2x2()

        self.conv2 = ResBlock(enc_chs[0]+add_chs[0], enc_chs[1])
        self.pool2 = MaxPool2x2()

        self.conv3 = ResBlock2(enc_chs[1]+add_chs[1], enc_chs[2])
        self.pool3 = MaxPool2x2()

    def forward(self, x1, x2, add_feats=None):
        x = torch.cat([x1,x2], dim=1)
        feats = [x]

        for i in range(self.num_layers):
            conv = getattr(self, f'conv{i+1}')
            if i > 0 and add_feats is not None:
                add_feat = F.interpolate(add_feats[i-1], size=x.shape[2:])
                x = torch.cat([x,add_feat], dim=1)
            x = conv(x)
            pool = getattr(self, f'pool{i+1}')
            x = pool(x)
            feats.append(x)

        return feats


class VideoEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(64,128)):
        super().__init__()
        if in_ch != 3:
            raise NotImplementedError

        self.num_layers = 2
        self.expansion = 4
        self.tem_scales = (1.0, 0.5)

        self.stem = nn.Sequential(
            nn.Conv3d(3, enc_chs[0], kernel_size=(3,9,9), stride=(1,4,4), padding=(1,4,4), bias=False),
            nn.BatchNorm3d(enc_chs[0]),
            nn.ReLU(True)
        )
        exps = self.expansion
        self.layer1 = nn.Sequential(
            ResBlock3D(enc_chs[0], enc_chs[0]*exps, enc_chs[0], ds=BasicConv3D(enc_chs[0], enc_chs[0]*exps, 1, bn=True)),
            ResBlock3D(enc_chs[0]*exps, enc_chs[0]*exps, enc_chs[0])
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(enc_chs[0]*exps, enc_chs[1]*exps, enc_chs[1], stride=(2,2,2), ds=BasicConv3D(enc_chs[0]*exps, enc_chs[1]*exps, 1, stride=(2,2,2), bn=True)),
            ResBlock3D(enc_chs[1]*exps, enc_chs[1]*exps, enc_chs[1])
        )

    def forward(self, x):
        feats = [x]

        x = self.stem(x)
        for i in range(self.num_layers):
            layer = getattr(self, f'layer{i+1}')
            x = layer(x)
            feats.append(x)

        return feats


class SimpleDecoder(nn.Module):
    def __init__(self, itm_ch, enc_chs, dec_chs):
        super().__init__()
        
        enc_chs = enc_chs[::-1]
        self.conv_bottom = Conv3x3(itm_ch, itm_ch, bn=True, act=True)
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, out_ch)
            for in_ch1, in_ch2, out_ch in zip(enc_chs, (itm_ch,)+dec_chs[:-1], dec_chs)
        ])
        self.conv_out = BasicConv(dec_chs[-1], 1, 1, bn=False, act=False)
    
    def forward(self, x, feats):
        feats = feats[::-1]
        
        x = self.conv_bottom(x)
        
        for feat, blk in zip(feats, self.blocks):
            x = blk(feat, x)

        y = self.conv_out(x)

        return y


class P2VNet(nn.Module):
    def __init__(self, in_ch, video_len=8, enc_chs_p=(32,64,128), enc_chs_v=(64,128), dec_chs=(256,128,64,32)):
        super().__init__()
        if video_len < 2:
            raise ValueError
        self.video_len = video_len
        self.encoder_v = VideoEncoder(in_ch, enc_chs=enc_chs_v)
        enc_chs_v = tuple(ch*self.encoder_v.expansion for ch in enc_chs_v)
        self.encoder_p = PairEncoder(in_ch, enc_chs=enc_chs_p, add_chs=enc_chs_v)
        self.conv_out_v = BasicConv(enc_chs_v[-1], 1, 1)
        self.convs_video = nn.ModuleList(
            [
                BasicConv(2*ch, ch, 1, bn=True, act=True)
                for ch in enc_chs_v
            ]
        )
        self.decoder = SimpleDecoder(enc_chs_p[-1], (2*in_ch,)+enc_chs_p, dec_chs)
    
    def forward(self, t1, t2, return_aux=False):
        frames = self.pair_to_video(t1, t2)
        feats_v = self.encoder_v(frames.transpose(1,2))
        feats_v.pop(0)

        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        feats_p = self.encoder_p(t1, t2, feats_v)

        pred = self.decoder(feats_p[-1], feats_p)

        if return_aux:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])
            return pred, pred_v
        else:
            return pred

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

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)


# 2donly
class VideoEncoder_2DOnly(nn.Module):
    def __init__(self, in_ch, enc_chs=(64,128)):
        super().__init__()

        self.num_layers = 2
        self.expansion = 4

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, enc_chs[0], kernel_size=(1,9,9), stride=(1,4,4), padding=(0,4,4), bias=False),
            nn.BatchNorm3d(enc_chs[0]),
            nn.ReLU(True)
        )
        exps = self.expansion
        self.layer1 = nn.Sequential(
            ResBlock3D(enc_chs[0], enc_chs[0]*exps, enc_chs[0], ds=BasicConv3D(enc_chs[0], enc_chs[0]*exps, 1, bn=True)),
            ResBlock3D(enc_chs[0]*exps, enc_chs[0]*exps, enc_chs[0])
        )
        self.layer1[0].conv2.seq[0] = nn.Conv3d(enc_chs[0], enc_chs[0], (1,3,3), padding=(0,1,1))
        self.layer1[1].conv2.seq[0] = nn.Conv3d(enc_chs[0], enc_chs[0], (1,3,3), padding=(0,1,1))
        self.layer2 = nn.Sequential(
            ResBlock3D(enc_chs[0]*exps, enc_chs[1]*exps, enc_chs[1], stride=(2,2,2), ds=BasicConv3D(enc_chs[0]*exps, enc_chs[1]*exps, 1, stride=(2,2,2), bn=True)),
            ResBlock3D(enc_chs[1]*exps, enc_chs[1]*exps, enc_chs[1])
        )
        self.layer2[0].conv2.seq[0] = nn.Conv3d(enc_chs[1], enc_chs[1], (1,3,3), padding=(0,1,1))
        self.layer2[1].conv2.seq[0] = nn.Conv3d(enc_chs[1], enc_chs[1], (1,3,3), padding=(0,1,1))

    def forward(self, x):
        feats = [x]

        x = self.stem(x)
        for i in range(self.num_layers):
            layer = getattr(self, f'layer{i+1}')
            x = layer(x)
            feats.append(x)

        return feats
        

class P2VNet_2DOnly(P2VNet):
    def __init__(self, in_ch, video_len=8, enc_chs_p=(32,64,128), enc_chs_v=(64,164), dec_chs=(256,128,64,32)):
        super().__init__(in_ch, video_len, enc_chs_p, enc_chs_v, dec_chs)
        self.encoder_v = VideoEncoder_2DOnly(in_ch, enc_chs=enc_chs_v)


# 2donlyflat
class P2VNet_2DOnlyFlat(P2VNet):
    def __init__(self, in_ch, video_len=8, enc_chs_p=(32,64,128), enc_chs_v=(64,164), dec_chs=(256,128,64,32)):
        super().__init__(in_ch, video_len, enc_chs_p, enc_chs_v, dec_chs)
        self.encoder_v = VideoEncoder_2DOnly(in_ch*video_len, enc_chs=enc_chs_v)

    def pair_to_video(self, im1, im2, rate_map=None):
        return super().pair_to_video(im1, im2, rate_map).flatten(1,2).unsqueeze(1)


# notemporal
class P2VNet_NoTemporal(nn.Module):
    def __init__(self, in_ch, video_len=8, enc_chs_p=(64,256,256), enc_chs_v=(64,128), dec_chs=(256,128,64,32)):
        super().__init__()
        self.encoder_p = PairEncoder(in_ch, enc_chs=enc_chs_p)
        self.conv_out_p = BasicConv(enc_chs_p[-1], 1, 1)
        self.decoder = SimpleDecoder(enc_chs_p[-1], (2*in_ch,)+enc_chs_p, dec_chs)
    
    def forward(self, t1, t2, return_aux=False):
        feats_p = self.encoder_p(t1, t2)

        pred = self.decoder(feats_p[-1], feats_p)
        
        if return_aux:
            pred_p = self.conv_out_p(feats_p[-1])
            pred_p = F.interpolate(pred_p, size=pred.shape[2:])
            return pred, pred_p
        else:
            return pred


# latefusion
class P2VNet_LateFusion(P2VNet):
    def __init__(self, in_ch, video_len=8, enc_chs_p=(32,64,128), enc_chs_v=(64,128), dec_chs=(256,128,64,32)):
        super().__init__(in_ch, video_len, enc_chs_p, enc_chs_v, dec_chs)

        self.encoder_p = PairEncoder(in_ch, enc_chs=enc_chs_p)
        enc_chs_v = tuple(ch*self.encoder_v.expansion for ch in enc_chs_v)
        self.convs_video = nn.ModuleList([BasicConv(2*enc_chs_v[-1], enc_chs_v[-1], 1, bn=True, act=True)])
        self.conv_fuse = BasicConv(enc_chs_p[-1]+enc_chs_v[-1], 2*enc_chs_p[-1], 1, bn=True, act=False)
        self.decoder = SimpleDecoder(2*enc_chs_p[-1], (2*in_ch,)+enc_chs_p, dec_chs)
    
    def forward(self, t1, t2, return_aux=False):
        frames = self.pair_to_video(t1, t2)
        feats_v = self.encoder_v(frames.transpose(1,2))
        feats_v.pop(0)
        feats_v[-1] = self.convs_video[-1](self.tem_aggr(feats_v[-1]))

        feats_p = self.encoder_p(t1, t2)
        feat_f = torch.cat([feats_p[-1], F.interpolate(feats_v[-1], feats_p[-1].shape[2:])], 1)
        feat_f = self.conv_fuse(feat_f)
        pred = self.decoder(feat_f, feats_p)

        if return_aux:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])
            return pred, pred_v
        else:
            return pred


# halfhalf
class P2VNet_Halfhalf(P2VNet):
    def pair_to_video(self, im1, im2, rate_map=None):
        return torch.stack([im1]*(self.video_len//2)+[im2]*(self.video_len//2), dim=1)


# alter
class P2VNet_Alter(P2VNet):
    def pair_to_video(self, im1, im2, rate_map=None):
        return torch.stack([im1,im2], dim=1).repeat(1,self.video_len//2,1,1,1)


# perm
class P2VNet_Perm(P2VNet):
    def pair_to_video(self, im1, im2, rate_map=None):
        return torch.stack([im2,im1,im1,im1,im2,im1,im2,im2], dim=1)


# decouple
class VideoEncoder_1DOnly(nn.Module):
    def __init__(self, in_ch, enc_chs=(64,128)):
        super().__init__()
        if in_ch != 3:
            raise NotImplementedError

        self.num_layers = 2
        self.expansion = 4

        self.stem = nn.Sequential(
            nn.Conv3d(3, enc_chs[0], kernel_size=(3,1,1), stride=(1,4,4), padding=(1,0,0), bias=False),
            nn.BatchNorm3d(enc_chs[0]),
            nn.ReLU(True)
        )
        exps = self.expansion
        self.layer1 = nn.Sequential(
            ResBlock3D(enc_chs[0], enc_chs[0]*exps, enc_chs[0], ds=BasicConv3D(enc_chs[0], enc_chs[0]*exps, 1, bn=True)),
            ResBlock3D(enc_chs[0]*exps, enc_chs[0]*exps, enc_chs[0])
        )
        self.layer1[0].conv2.seq[0] = nn.Conv3d(enc_chs[0], enc_chs[0], (3,1,1), padding=(1,0,0))
        self.layer1[1].conv2.seq[0] = nn.Conv3d(enc_chs[0], enc_chs[0], (3,1,1), padding=(1,0,0))
        self.layer2 = nn.Sequential(
            ResBlock3D(enc_chs[0]*exps, enc_chs[1]*exps, enc_chs[1], stride=(2,2,2), ds=BasicConv3D(enc_chs[0]*exps, enc_chs[1]*exps, 1, stride=(2,2,2), bn=True)),
            ResBlock3D(enc_chs[1]*exps, enc_chs[1]*exps, enc_chs[1])
        )
        self.layer2[0].conv2.seq[0] = nn.Conv3d(enc_chs[1], enc_chs[1], (3,1,1), padding=(1,0,0))
        self.layer2[1].conv2.seq[0] = nn.Conv3d(enc_chs[1], enc_chs[1], (3,1,1), padding=(1,0,0))

    def forward(self, x):
        feats = [x]

        x = self.stem(x)
        for i in range(self.num_layers):
            layer = getattr(self, f'layer{i+1}')
            x = layer(x)
            feats.append(x)

        return feats
        

class P2VNet_Decouple(P2VNet):
    def __init__(self, in_ch, video_len=8, enc_chs_p=(32,64,128), enc_chs_v=(64,180), dec_chs=(256,128,64,32)):
        super().__init__(in_ch, video_len, enc_chs_p, enc_chs_v, dec_chs)
        self.encoder_v = VideoEncoder_1DOnly(in_ch, enc_chs=enc_chs_v)


class P2VNet_Prior(P2VNet):
    def __init__(self, in_ch, video_len=8, enc_chs_p=(32,64,128), enc_chs_v=(64,128), dec_chs=(256,128,64,32)):
        super().__init__(in_ch, video_len, enc_chs_p, enc_chs_v, dec_chs)
        self.prior_model = P2VNet_NoTemporal(3, 8)
        self.prior_model.load_state_dict(torch.load('../exp/svcd/weights/checkpoint_010_p2v_notemporal.pth')['state_dict'])
        for p in self.prior_model.parameters():
            p.requires_grad = False
    
    def forward(self, t1, t2, return_aux=False):
        with torch.no_grad():
            self.prior_model.eval()
            prior_map = self.prior_model(t1, t2)
            rate_map = torch.sigmoid(prior_map*0.1)
        frames = self.pair_to_video(t1, t2, rate_map)
        feats_v = self.encoder_v(frames.transpose(1,2))
        feats_v.pop(0)

        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        feats_p = self.encoder_p(t1, t2, feats_v)

        pred = self.decoder(feats_p[-1], feats_p)

        if return_aux:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])
            return pred, pred_v
        else:
            return pred


class P2VNet_CVA(P2VNet):
    def forward(self, t1, t2, return_aux=False):
        with torch.no_grad():
            cv = torch.pow(t1-t2, 2)
            cv = torch.sqrt(torch.sum(cv, dim=1, keepdims=True))
            rate_map = (cv - cv.min()) / (cv.max() - cv.min() + 1e-32)
        frames = self.pair_to_video(t1, t2, rate_map)
        feats_v = self.encoder_v(frames.transpose(1,2))
        feats_v.pop(0)

        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        feats_p = self.encoder_p(t1, t2, feats_v)

        pred = self.decoder(feats_p[-1], feats_p)

        if return_aux:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])
            return pred, pred_v
        else:
            return pred