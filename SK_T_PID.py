# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS
from mmengine.model import Sequential
import torch
from mmcv.cnn import ConvModule
import itertools
import torch.nn.functional as F
class Conv_BN_HW(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.action=nn.Hardswish()

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.action(x)
        return x


class DWC_BN_HW(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, bias=False,padding=padding,groups=in_channels)
        self.conv2=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.action = nn.Hardswish()

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.bn(x)
        x=self.action(x)
        return x

class Residual(torch.nn.Module):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

class IRMB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 t,
                 ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t,
                      bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.Hardswish(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.stride = stride

    def forward(self, x):
        out = self.conv(x)
        if self.stride == 1:
            out =out+ self.shortcut(x)
        return out

class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem_layers = []
        stem_net1=Conv_BN_HW(3,32,3,2,1)
        self.add_module("layer1", stem_net1)
        self.stem_layers.append("layer1")

        stem_net2=IRMB(32,64,2,1)
        self.add_module("layer2", stem_net2)
        self.stem_layers.append("layer2")

        stem_net3=IRMB(64,64,2,1)
        self.add_module("layer3", stem_net3)
        self.stem_layers.append("layer3")

        stem_net4 = IRMB(64, 128, 2, 1)
        self.add_module("layer4", stem_net4)
        self.stem_layers.append("layer4")

    def forward(self,x):
        outs = []
        for i,layer_name in enumerate(self.stem_layers):
            conv_layer = getattr(self, layer_name)
            x = conv_layer(x)
            if i>1:
                outs.append(x)
        return tuple(outs)

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=nn.Hardswish,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Conv_BN_HW(dim, h)
        self.proj = torch.nn.Sequential(activation(), Conv_BN_HW(
            self.dh, dim))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(
            B, self.num_heads, -1, H * W
        ).split([self.key_dim, self.key_dim, self.d], dim=2)
        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        x = self.proj(x)
        return x

class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=nn.Hardswish,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        h = self.dh + nh_kd
        self.kv = Conv_BN_HW(in_dim, h)
        self.q = torch.nn.Sequential(
            torch.nn.AvgPool2d(1, stride, 0),
            Conv_BN_HW(in_dim, nh_kd))
        self.proj = torch.nn.Sequential(
            activation(), Conv_BN_HW(self.d * num_heads, out_dim))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, C, H, W = x.shape
        k, v = self.kv(x).view(B, self.num_heads, -1, H *
                               W).split([self.key_dim, self.d], dim=2)
        q = self.q(x).view(B, self.num_heads, self.key_dim, self.resolution_2)

        attn = (q.transpose(-2, -1) @ k) * self.scale + \
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).reshape(
            B, -1, self.resolution_, self.resolution_)
        x = self.proj(x)
        return x



class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        #一般情况下，mid_channels是in_channels的一半#
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )

    def forward(self, x, y):
        #x是p，y是i
        input_size = x.size()

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)
        x=sim_map*(y-x)
        return x

class Skip_block(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.in_channels=in_channels
        self.mid_channels=mid_channels
        self.conv1 = DWC_BN_HW(self.in_channels, self.mid_channels,kernel_size=3,padding=1)
        self.max_plooling=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2=Conv_BN_HW(self.mid_channels,self.mid_channels*2)
    def forward(self,x):
        x = self.conv1(x)
        x=self.max_plooling(x)
        x=self.conv2(x)
        return x

class Edge_fuse_block(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.in_channels=in_channels
        self.mid_channels=mid_channels
        self.conv1=Conv_BN_HW(self.in_channels,self.mid_channels)
        self.Upsamle=nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)

    def forward(self,x):
        x=self.conv1(x)
        x=self.Upsamle(x)
        return x

class Conv_UP(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=False,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.action=nn.Hardswish()

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.action(x)
        return x

@MODELS.register_module()
class SK_T_PID_backbone(BaseModule):
    def __init__(self):
        super().__init__()
        self.stem_layer=Stem()
        #I分支从1/16开始，（128，32，32）
        self.vit_layer1=Residual(Attention(dim=128,key_dim=16,resolution=32),drop=0)

        self.vit_layer2=AttentionSubsample(in_dim=128,out_dim=256,key_dim=16,resolution=32,resolution_=16)

        self.vit_layer3=Residual(Attention(dim=256,key_dim=16,resolution=16),drop=0)

        self.vit_layer4=AttentionSubsample(in_dim=256, out_dim=512, key_dim=16, resolution=16, resolution_=8)

        #P分支从1/16开始
        self.p_layer1=IRMB(128,128,1,1)

        self.p_layer2 = IRMB(128, 256, 2, 1)

        self.p_layer3 = IRMB(256, 256, 1, 1)
        # self.p_layer3=Conv_BN_HW(256,256)

        self.pag1=PagFM(128,64)

        self.pag2=PagFM(256,128)

        #d分支从1/8开始，（64，64，64）
        self.d_layer1=IRMB(64,128,2,1)

        self.d_layer2=IRMB(128,128,1,1)

        self.skip_block=Skip_block(64,64)

        self.edge_fuse=Edge_fuse_block(256,128)

        self.d_layer3=Conv_BN_HW(128,128)

        #数据融合
        self.conv_i=Conv_BN_HW(512,256)
        self.bn_i=nn.BatchNorm2d(256)
        self.bn_p = nn.BatchNorm2d(256)
        self.bn_i_p=nn.BatchNorm2d(128)
        self.bn_d = nn.BatchNorm2d(128)

        self.conv_p=Conv_BN_HW(256,128)

        # self.bn_i_p_d=nn.BatchNorm2d(128)

        self.convpid=Conv_BN_HW(128,128,kernel_size=3,padding=1)

        # self.conv_fuse=Conv_BN_HW(128,128)
        #辅助训练
        self.conv_p_t=Conv_BN_HW(256,64)
        self.conv_d_t=Conv_BN_HW(128,64)

    def forward(self,x):
        width = x.shape[-1]//8
        height = x.shape[-2]//8
        #每一个加法后面都应当增加一个归一化层#
        #增加反卷积和上采样，平衡PID，之后直接试试原生的pidhead#
        # outs=[]
        x1,x2=self.stem_layer(x)#x1(128,64,64)x2(256,32,32)
        x_i1=self.vit_layer1(x2)
        x_i2=self.vit_layer2(x_i1)
        x_i3 = self.vit_layer3(x_i2)
        x_i4 = self.vit_layer4(x_i3)

        # outs.append(x_i4)

        x_p1=self.p_layer1(x2)
        x_p1=x_p1+self.pag1(x_p1,x_i1)
        x_p2=self.p_layer2(x_p1)
        x_p2=x_p2+self.pag2(x_p2,x_i3)
        x_p2=self.p_layer3(x_p2)
        # outs.append(x_p2)

        x_d1=self.d_layer1(x1)
        x_fd=self.edge_fuse(x_i3)
        x_d2=self.d_layer2(x_d1+x_fd)
        x_skip=self.skip_block(x1)
        x_d2=self.d_layer3(x_d2+x_skip)
        # outs.append(x_d2)

        #数据融合
        # print('xxxxxxxxxxxxxx',x_p2.shape,x_i4.shape,x_d2.shape)
        x_i=F.sigmoid(self.conv_i(x_i4))
        x_i=F.interpolate(x_i,size=[x_p2.shape[-2],x_p2.shape[-1]],mode='bilinear',
                          align_corners=False)
        x_i=self.bn_i(x_i)

        x_p=F.sigmoid(x_p2)
        x_p=self.bn_p(x_p)
        x_p_i=x_i*x_p
        x_p_i=F.interpolate(self.conv_p(x_p_i),size=[x_d2.shape[-2],x_d2.shape[-1]],
                            mode='bilinear',align_corners=False)
        x_p_i=self.bn_i_p(x_p_i)

        x_d2=F.sigmoid(x_d2)
        x_d2=self.bn_d(x_d2)

        x_p_i_d=x_p_i*x_d2
        x_p_i_d=F.interpolate(x_p_i_d,size=[height,width],mode='bilinear',
                              align_corners=False)
        x_p_i_d=self.convpid(x_p_i_d)
        # x_p_i_d=self.conv_fuse(x_p_i_d)

        # if self.training:
        #     # x_p2
        #     x_p2=F.interpolate(self.conv_p_t(x_p2),size=[height,width],mode='bilinear',align_corners=False)
        #     x_d2=F.interpolate(self.conv_d_t(x_d2),size=[height,width],mode='bilinear',align_corners=False)
        #     return (x_p2,x_p_i_d,x_d2)
        # else:
        #     return x_p_i_dIn
        x_p2 = F.interpolate(self.conv_p_t(x_p2), size=[height, width], mode='bilinear', align_corners=False)
        x_d2 = F.interpolate(self.conv_d_t(x_d2), size=[height, width], mode='bilinear', align_corners=False)
        return (x_p2, x_p_i_d, x_d2)

# model=Stem()
# inputs = torch.rand(1, 3, 512, 512)

# model=AttentionSubsample(256,512,key_dim=16,resolution=32,resolution_=16)
# model=Attention(256,key_dim=16,resolution=32)
# inputs = torch.rand(1, 256, 32, 32)
# outs=model.forward(inputs)
# print(outs[0].shape,outs[1].shape)

#pag测试
# model=PagFM(64,32)
# inputs_x = torch.rand(1, 64, 32, 32)
# inputs_y = torch.rand(1, 64, 16, 16)
# outs=model.forward(inputs_x,inputs_y)

# model=Edge_fuse_block(512,256)
# inputs = torch.rand(1, 512, 16, 16)
# outs=model.forward(inputs)

model=SK_T_PID_backbone()
# model.eval()
inputs = torch.rand(1, 3, 512, 512)
outs=model.forward(inputs)
print(outs[0].shape,outs[1].shape,outs[2].shape)

# model=Skip_block(64,64,)
# # model.eval()
# inputs = torch.rand(1, 64, 64, 64)
# outs=model.forward(inputs)
# print(outs.shape)
