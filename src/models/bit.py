# Implementation of 
# H. Chen, Z. Qi, and Z. Shi, “Remote Sensing Image Change Detection With Transformers,” IEEE Trans. Geosci. Remote Sensing, pp. 1–14, 2021, doi: 10.1109/TGRS.2021.3095166.

# Refer to https://github.com/justchenhao/BIT_CD
# The weight initialization method is different from the official implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import resnet
from ._blocks import Conv3x3, Conv1x1, get_norm_layer
from ._utils import Identity, KaimingInitMixin


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            Conv3x3(in_ch, in_ch, norm=True, act=True),
            Conv3x3(in_ch, out_ch)
        )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x1, x2, **kwargs):
        return self.fn(x1, x2, **kwargs) + x1


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm(x1), self.norm(x2), **kwargs)


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout_rate=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate)
        )


class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads=8, head_dim=64, dropout_rate=0., apply_softmax=True):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.scale = dim ** -0.5

        self.apply_softmax = apply_softmax
        
        self.fc_q = nn.Linear(dim, inner_dim, bias=False)
        self.fc_k = nn.Linear(dim, inner_dim, bias=False)
        self.fc_v = nn.Linear(dim, inner_dim, bias=False)

        self.fc_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, ref):
        b, n = x.shape[:2]
        h = self.n_heads

        q = self.fc_q(x)
        k = self.fc_k(ref)
        v = self.fc_v(ref)

        q = q.reshape((b,n,h,-1)).permute((0,2,1,3))
        k = k.reshape((b,ref.shape[1],h,-1)).permute((0,2,1,3))
        v = v.reshape((b,ref.shape[1],h,-1)).permute((0,2,1,3))

        mult = torch.matmul(q, k.transpose(-1,-2)) * self.scale

        if self.apply_softmax:
            mult = F.softmax(mult, dim=-1)

        out = torch.matmul(mult, v)
        out = out.permute((0,2,1,3)).flatten(2)
        return self.fc_out(out)


class SelfAttention(CrossAttention):
    def forward(self, x):
        return super().forward(x, x)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, n_heads, head_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, SelfAttention(dim, n_heads, head_dim, dropout_rate))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate)))
            ]))

    def forward(self, x):
        for att, ff in self.layers:
            x = att(x)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, n_heads, head_dim, mlp_dim, dropout_rate, apply_softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, CrossAttention(dim, n_heads, head_dim, dropout_rate, apply_softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate)))
            ]))

    def forward(self, x, m):
        for att, ff in self.layers:
            x = att(x, m)
            x = ff(x)
        return x


class Backbone(nn.Module, KaimingInitMixin):
    def __init__(
        self, 
        in_ch, out_ch=32,
        arch='resnet18',
        pretrained=True,
        n_stages=5
    ):
        super().__init__()

        expand = 1
        strides = (2,1,2,1,1)
        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        else:
            raise ValueError

        self.n_stages = n_stages

        if self.n_stages == 5:
            itm_ch = 512 * expand
        elif self.n_stages == 4:
            itm_ch = 256 * expand
        elif self.n_stages == 3:
            itm_ch = 128 * expand
        else:
            raise ValueError

        self.upsample = nn.Upsample(scale_factor=2)    
        self.conv_out = Conv3x3(itm_ch, out_ch)

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_ch, 
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        if not pretrained:
            self._init_weight()

    def forward(self, x):
        y = self.resnet.conv1(x)
        y = self.resnet.bn1(y)
        y = self.resnet.relu(y)
        y = self.resnet.maxpool(y)

        y = self.resnet.layer1(y)
        y = self.resnet.layer2(y)
        y = self.resnet.layer3(y)
        y = self.resnet.layer4(y)

        y = self.upsample(y)

        return self.conv_out(y)

    def _trim_resnet(self):
        if self.n_stages > 5:
            raise ValueError

        if self.n_stages < 5:
            self.resnet.layer4 = Identity()

        if self.n_stages <= 3:
            self.resnet.layer3 = Identity()

        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()


class BIT(nn.Module):
    def __init__(
        self, in_ch, out_ch,
        backbone='resnet18', n_stages=4, 
        use_tokenizer=True, token_len=4, 
        pool_mode='max', pool_size=2,
        enc_with_pos=True, 
        enc_depth=1, enc_head_dim=64, 
        dec_with_softmax=True,
        dec_depth=1, dec_head_dim=64,
        **backbone_kwargs
    ):
        super().__init__()

        # TODO: reduce hard-coded parameters
        dim = 32
        mlp_dim = 2*dim
        chn = dim

        self.backbone = Backbone(in_ch, chn, arch=backbone, n_stages=n_stages, **backbone_kwargs)

        self.use_tokenizer = use_tokenizer
        if not use_tokenizer:
            # If a tokenzier is not to be used，then downsample the feature maps
            self.pool_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = pool_size * pool_size
        else:
            self.conv_att = Conv1x1(32, token_len, bias=False)
            self.token_len = token_len

        self.enc_with_pos = enc_with_pos
        if enc_with_pos:
            self.enc_pos_embedding = nn.Parameter(torch.randn(1,self.token_len*2,chn))

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.enc_head_dim = enc_head_dim
        self.dec_head_dim = dec_head_dim
        
        self.encoder = TransformerEncoder(
            dim=dim, 
            depth=enc_depth, 
            n_heads=8,
            head_dim=enc_head_dim,
            mlp_dim=mlp_dim,
            dropout_rate=0.
        )
        self.decoder = TransformerDecoder(
            dim=dim, 
            depth=dec_depth,
            n_heads=8, 
            head_dim=dec_head_dim, 
            mlp_dim=mlp_dim, 
            dropout_rate=0.,
            apply_softmax=dec_with_softmax
        )

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.classifier = DoubleConv(chn, out_ch)

    def _get_semantic_tokens(self, x):
        b, c = x.shape[:2]
        att_map = self.conv_att(x)
        att_map = att_map.reshape((b,self.token_len,1,-1))
        att_map = F.softmax(att_map, dim=-1)
        x = x.reshape((b,1,c,-1))
        tokens = (x*att_map).sum(-1)
        return tokens

    def _get_reshaped_tokens(self, x):
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode == 'avg':
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        else:
            x = x
        tokens = x.permute((0,2,3,1)).flatten(1,2)
        return tokens

    def encode(self, x):
        if self.enc_with_pos:
            x += self.enc_pos_embedding
        x = self.encoder(x)
        return x

    def decode(self, x, m):
        b, c, h, w = x.shape
        x = x.permute((0,2,3,1)).flatten(1,2)
        x = self.decoder(x, m)
        x = x.transpose(1,2).reshape((b,c,h,w))
        return x

    def forward(self, t1, t2):
        # Extract features via shared backbone
        x1 = self.backbone(t1)
        x2 = self.backbone(t2)

        # Tokenization
        if self.use_tokenizer:
            token1 = self._get_semantic_tokens(x1)
            token2 = self._get_semantic_tokens(x2)
        else:
            token1 = self._get_reshaped_tokens(x1)
            token2 = self._get_reshaped_tokens(x2)

        # Transformer encoder forward
        token = torch.cat([token1, token2], dim=1)
        token = self.encode(token)
        token1, token2 = torch.chunk(token, 2, dim=1)

        # Transformer decoder forward
        y1 = self.decode(x1, token1)
        y2 = self.decode(x2, token2)

        # Feature differencing
        y = torch.abs(y1 - y2)
        y = self.upsample(y)

        # Classifier forward
        pred = self.classifier(y)

        return pred


