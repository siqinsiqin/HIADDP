# -*-coding:utf-8 -*-
"""
# Time       ：2022/9/1 20:44
# Author     ：comi
# version    ：python 3.8
# Description：
# todo u2netV2
# todo 通道打乱，# 还没实现 改用cat ，废弃add
"""
import torch
from ptflops import get_model_complexity_info
from torch import nn
from torch.nn.functional import interpolate
from torch.utils import checkpoint

from models.u2net3p.cbam import CBAM


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width, z = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width, z)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width, z)
    return x


def channel_spilt(x):
    # 分别用于残差分支和u结构
    x1, x2 = x.chunk(2, dim=1)
    return x1, x2


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=0, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        super(ConvBNReLU, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=False)
        )


class DEPTHWISECONV(nn.Module):
    """
    深度可分离卷积V1
    """

    def __init__(self, in_ch, out_ch, checkpoint=False):
        super(DEPTHWISECONV, self).__init__()
        self.checkpoint = checkpoint
        # 也相当于分组为1的分组卷积
        self.dw = nn.Sequential(
            # todo 逐层卷积
            nn.Conv3d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1,
                      groups=in_ch, bias=False, ),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=False),

            # 逐点卷积
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, input):
        if self.checkpoint:
            out = checkpoint.checkpoint(self.dw, input)
        else:
            out = self.dw(input)
        return out


class InvertedResidual(nn.Module):
    """
    mobile net v2
    """

    def __init__(self, inp, oup, stride, expand_ratio=6, norm_layer=nn.BatchNorm3d):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        print(hidden_dim)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class U2baseblock(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, checkpoint=False):
        super(U2baseblock, self).__init__()
        self.checkpoint = checkpoint
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        if self.checkpoint:
            x = checkpoint.checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class baseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1, dep=True, checkpoint=False):
        super(baseBlock, self).__init__()
        self.checkpoint = checkpoint
        self.dep = dep
        if dep:
            self.conv = DEPTHWISECONV(in_ch, out_ch)
        else:
            self.conv = U2baseblock(in_ch, out_ch, dirate)

    def forward(self, x):
        if self.checkpoint:
            return checkpoint.checkpoint(self.conv, x)
        else:
            return self.conv(x)


class singleKernal(nn.Module):
    def __init__(self, inc, outc, p=1, size=3, ):
        super(singleKernal, self).__init__()
        if p == 1:
            self.conv = nn.Conv3d(inc, outc, (size, 1, 1), stride=1, padding=(1, 0, 0))
        elif p == 2:
            self.conv = nn.Conv3d(inc, outc, (1, size, 1), stride=1, padding=(0, 1, 0))
        elif p == 3:
            self.conv = nn.Conv3d(inc, outc, (1, 1, size), stride=1, padding=(0, 0, 1))
        else:
            self.conv = nn.Conv3d(inc, outc, (1, 1, 1), stride=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(outc)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class mutiScale(nn.Module):
    def __init__(self, inc, outc, dirate=1, dep=False):
        super(mutiScale, self).__init__()
        size = 3
        self.lenear1 = singleKernal(inc, outc, p=1, size=size)
        self.lenear2 = singleKernal(inc, outc, p=2, size=size)
        self.lenear3 = singleKernal(inc, outc, p=3, size=size)
        self.lenear4 = singleKernal(inc, outc, p=4, size=size)
        # self.conv = baseBlock(mid, mid, dirate=dirate, dep=False)

    def forward(self, x):
        conv1 = self.lenear1(x)
        conv2 = self.lenear2(x)
        conv3 = self.lenear3(x)
        conv4 = self.lenear4(x)
        # conv5 = self.conv(x)
        return conv1 + conv2 + conv3 + conv4


def _upsample(src, tar):
    return interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)


class posAttn(nn.Module):

    def __init__(self, channel, depth):  #
        super(posAttn, self).__init__()

        self.depth = depth
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(CBAM(channel, channel))
            self.layers.append(nn.MaxPool3d(2, 2))

    def forward(self, x):

        assert len(x) == self.depth, 'depth error'
        pool = None
        results = []
        idx = 0
        for i, layer in enumerate(self.layers):

            if i % 2 == 0:
                if pool is not None:
                    features = x[idx] + pool
                else:
                    features = x[idx]
                features = layer(features)
                results.append(features)
                idx += 1
            else:
                if pool is None:
                    pool = features
                pool = layer(pool)

        return results


class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=16, out_ch=32, side=False, upsample=False, dep=False):
        super(RSU7, self).__init__()
        self.side = side
        # self.upsample = upsample

        # self.posAttn = posAttn(mid_ch, 5)

        self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)

        self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
        self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)

        self.rebnconv7 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)

        self.rebnconv6d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv5d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv4d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)  # RSU7: 9*9*16

        hx7 = self.rebnconv7(hx6)

        # cbam
        # pa2, pa3, pa4, pa5, pa6 = self.posAttn([hx2, hx3, hx4, hx5, hx6])

        # hx6d = self.rebnconv6d(torch.cat((hx7, pa6), 1))  # plan 3
        # hx6dup = _upsample(hx6d, hx5)
        #
        # hx5d = self.rebnconv5d(torch.cat((hx6dup, pa5), 1))
        # hx5dup = _upsample(hx5d, hx4)
        #
        # hx4d = self.rebnconv4d(torch.cat((hx5dup, pa4), 1))
        # hx4dup = _upsample(hx4d, hx3)
        #
        # hx3d = self.rebnconv3d(torch.cat((hx4dup, pa3), 1))
        # hx3dup = _upsample(hx3d, hx2)
        #
        # hx2d = self.rebnconv2d(torch.cat((hx3dup, pa2), 1))
        # hx2dup = _upsample(hx2d, hx1)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))  # normal
        hx6dup = _upsample(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d  # + hxin  # 不加hxin u block
# class RSU7(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, side=False, upsample=False, dep=False):
#         super(RSU7, self).__init__()
#         self.side = side
#         self.upsample = upsample
#
#         # self.posAttn = posAttn(mid_ch, 5)
#
#         self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)
#
#         self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)
#         self.sp1 = SpatialAttention(7)
#         self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.sp2 = SpatialAttention(5)
#         self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.sp3 = SpatialAttention(5)
#         self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.sp4 = SpatialAttention(3)
#         self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv5 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.sp5 = SpatialAttention(3)
#         self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv6 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#
#         self.rebnconv7 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)
#
#         self.rebnconv6d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#
#         self.sp5d = SpatialAttention(3)
#         self.rebnconv5d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#
#         self.sp4d = SpatialAttention(3)
#         self.rebnconv4d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#
#         self.sp3d = SpatialAttention(5)
#         self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#
#         self.sp2d = SpatialAttention(5)
#         self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#
#         self.sp1d = SpatialAttention(7)
#         self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)
#
#     def forward(self, x):
#         hx = x
#
#         hxin = self.rebnconvin(hx)
#
#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1 * self.sp1(hx1))
#
#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2 * self.sp2(hx2))
#
#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3 * self.sp3(hx3))
#
#         hx4 = self.rebnconv4(hx)
#         hx = self.pool4(hx4 * self.sp4(hx4))
#
#         hx5 = self.rebnconv5(hx)
#         hx = self.pool5(hx5 * self.sp1(hx5))
#
#         hx6 = self.rebnconv6(hx)  # RSU7: 9*9*16
#
#         hx7 = self.rebnconv7(hx6)
#
#         # cbam
#         # pa2, pa3, pa4, pa5, pa6 = self.posAttn([hx2, hx3, hx4, hx5, hx6])
#
#         # hx6d = self.rebnconv6d(torch.cat((hx7, pa6), 1))  # plan 3
#         # hx6dup = _upsample(hx6d, hx5)
#         #
#         # hx5d = self.rebnconv5d(torch.cat((hx6dup, pa5), 1))
#         # hx5dup = _upsample(hx5d, hx4)
#         #
#         # hx4d = self.rebnconv4d(torch.cat((hx5dup, pa4), 1))
#         # hx4dup = _upsample(hx4d, hx3)
#         #
#         # hx3d = self.rebnconv3d(torch.cat((hx4dup, pa3), 1))
#         # hx3dup = _upsample(hx3d, hx2)
#         #
#         # hx2d = self.rebnconv2d(torch.cat((hx3dup, pa2), 1))
#         # hx2dup = _upsample(hx2d, hx1)
#
#         hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))  # normal
#         hx6dup = _upsample(hx6d, hx5)
#
#         hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
#         hx5dup = _upsample(hx5d * self.sp5d(hx5d), hx4)
#
#         hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
#         hx4dup = _upsample(hx4d * self.sp4d(hx4d), hx3)
#
#         hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
#         hx3dup = _upsample(hx3d * self.sp3d(hx3d), hx2)
#
#         hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
#         hx2dup = _upsample(hx2d * self.sp2d(hx2d), hx1)
#
#         hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
#         hx1d = hx1d * self.sp1d(hx1d)
#         return hx1d  # + hxin  # 不加hxin u block


class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, side=False, upsample=False, dep=False):
        super(RSU6, self).__init__()
        self.side = side
        self.upsample = upsample
        self.res = nn.Sequential(
            nn.Conv3d(in_ch // 2, out_ch // 2, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm3d(out_ch // 2),
            nn.ReLU()
        )

        self.rebnconvin = baseBlock(in_ch // 2, out_ch // 2, dirate=1, dep=dep)

        self.rebnconv1 = baseBlock(out_ch // 2, mid_ch, dirate=1, dep=dep)

        # self.posAttn = posAttn(mid_ch, 4)

        # self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)
        #
        # self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)

        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)

        self.rebnconv6 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)

        self.rebnconv5d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv4d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv1d = baseBlock(mid_ch * 2, out_ch // 2, dirate=1, dep=dep)
        # self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)

    def forward(self, x):
        hx = x

        cx, ux = channel_spilt(x)
        cx = self.res(cx)

        hxin = self.rebnconvin(ux)

        # hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        # cbam plan 2
        # pa2, pa3, pa4, pa5 = self.posAttn([hx2, hx3, hx4, hx5])
        # hx5d = self.rebnconv5d(torch.cat((hx6, hx5 + pa5), 1))
        # hx5dup = _upsample(hx5d, hx4)
        # hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4 + pa4), 1))
        # hx4dup = _upsample(hx4d, hx3)
        # hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3 + pa3), 1))
        # hx3dup = _upsample(hx3d, hx2)
        # hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2 + pa2), 1))
        # hx2dup = _upsample(hx2d, hx1)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))

        hx5dup = _upsample(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))

        hx4dup = _upsample(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))

        hx3dup = _upsample(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))

        hx2dup = _upsample(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        x = channel_shuffle(torch.cat((hx1d + hxin, cx), dim=1), 4)  #
        return x  # + hxin


class RSU5(nn.Module):

    def __init__(self, in_ch=64, mid_ch=32, out_ch=128, side=False, upsample=False, dep=False):
        super(RSU5, self).__init__()
        self.side = side
        self.upsample = upsample
        self.res = nn.Sequential(
            nn.Conv3d(in_ch // 2, out_ch // 2, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm3d(out_ch // 2),
            nn.ReLU()
        )

        self.rebnconvin = baseBlock(in_ch // 2, out_ch // 2, dirate=1, dep=dep)

        self.rebnconv1 = baseBlock(out_ch // 2, mid_ch, dirate=1, dep=dep)

        # self.posAttn = posAttn(mid_ch, 3)

        # self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)

        # self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)

        self.rebnconv5 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)

        self.rebnconv4d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)

        self.rebnconv1d = baseBlock(mid_ch * 2, out_ch // 2, dirate=1, dep=dep)
        # self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)

    def forward(self, x):
        hx = x

        cx, ux = channel_spilt(x)
        cx = self.res(cx)
        hxin = self.rebnconvin(ux)

        # hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        # cbam
        # pa2, pa3, pa4 = self.posAttn([hx2, hx3, hx4])
        # hx4d = self.rebnconv4d(torch.cat((hx5, hx4 + pa4), 1))
        # hx4dup = _upsample(hx4d, hx3)
        # hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3 + pa3), 1))
        # hx3dup = _upsample(hx3d, hx2)
        # hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2 + pa2), 1))
        # hx2dup = _upsample(hx2d, hx1)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        x = channel_shuffle(torch.cat((hx1d + hxin, cx), dim=1), 4)  #
        return x  # hx1d + hxin


class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=128, mid_ch=64, out_ch=256, side=False, dep=False, upsample=False):
        super(RSU4, self).__init__()
        self.upsample = upsample
        self.side = side
        self.res = nn.Sequential(
            nn.Conv3d(in_ch // 2, out_ch // 2, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm3d(out_ch // 2),
            nn.ReLU()
        )
        self.rebnconvin = baseBlock(in_ch // 2, out_ch // 2, dirate=1, dep=dep)

        self.rebnconv1 = baseBlock(out_ch // 2, mid_ch, dirate=1, dep=dep)

        # self.posAttn = posAttn(mid_ch, 2)

        # self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)
        #
        # self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)

        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)

        self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)

        self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
        self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)

        self.rebnconv1d = baseBlock(mid_ch * 2, out_ch // 2, dirate=1, dep=dep)

        # self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)

    def forward(self, x):
        hx = x

        cx, ux = channel_spilt(x)
        cx = self.res(cx)
        hxin = self.rebnconvin(ux)

        # hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)  # 9*9*16

        hx4 = self.rebnconv4(hx3)

        # cbam
        # pa2, pa3 = self.posAttn([hx2, hx3])
        # hx3d = self.rebnconv3d(torch.cat((hx4, hx3 + pa3), 1))
        # hx3dup = _upsample(hx3d, hx2)
        #
        # hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2 + pa2), 1))
        # hx2dup = _upsample(hx2d, hx1)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        x = channel_shuffle(torch.cat((hx1d + hxin, cx), dim=1), 4)  # + hxin
        return x  # hx1d + hxin


class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, side=False, upsample=False, dep=False):
        super(RSU4F, self).__init__()
        self.side = side
        self.upsample = upsample
        self.res = nn.Sequential(
            nn.Conv3d(in_ch // 2, out_ch // 2, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm3d(out_ch // 2),
            nn.ReLU()
        )

        self.rebnconvin = baseBlock(in_ch // 2, out_ch // 2, dirate=1, dep=dep)

        self.rebnconv1 = baseBlock(out_ch // 2, mid_ch, dirate=1, dep=dep)

        # self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)
        #
        # self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)

        self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)
        self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=4, dep=False)

        self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=8, dep=False)

        self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=4, dep=False)
        self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=2, dep=False)

        self.rebnconv1d = baseBlock(mid_ch * 2, out_ch // 2, dirate=1, dep=dep)
        # self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)

    def forward(self, x):
        hx = x
        cx, ux = channel_spilt(x)
        cx = self.res(cx)
        hxin = self.rebnconvin(ux)

        # hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))

        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        x = channel_shuffle(torch.cat((hx1d + hxin, cx), dim=1), 4)  #
        return x  # hx1d + hxin


class shuffleU2net(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, filter=None, side=False, sup=False, expR=True):
        super(shuffleU2net, self).__init__()
        self.side = side
        self.sup = sup
        times = 2
        if filter is None:
            # filter = [64, 128, 256, 512, 512]
            filter = [32, 64, 128, 256, 256]

        self.stage1 = RSU7(in_ch, filter[0] // times, filter[0])
        self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(filter[0], filter[0] // times, filter[1])
        self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(filter[1], filter[1] // times, filter[2])
        self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(filter[2], filter[2] // times, filter[3])
        self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(filter[3], filter[3] // times, filter[4])
        self.pool56 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(filter[4], filter[4] // times, filter[4])

        # decoder
        self.stage5d = RSU4F(filter[4] * 2, filter[4] // times, filter[4])
        self.stage4d = RSU4(filter[4] * 2, filter[2] // times, filter[2])
        self.stage3d = RSU5(filter[3], filter[1] // times, filter[1])
        self.stage2d = RSU6(filter[2], filter[0] // times, filter[0])
        self.stage1d = RSU7(filter[1], filter[0] // times, filter[0])

        self.side1 = nn.Conv3d(filter[0], out_ch, 3, padding=1)
        self.side2 = nn.Conv3d(filter[0], out_ch, 3, padding=1)
        self.side3 = nn.Conv3d(filter[1], out_ch, 3, padding=1)
        self.side4 = nn.Conv3d(filter[2], out_ch, 3, padding=1)
        self.side5 = nn.Conv3d(filter[3], out_ch, 3, padding=1)
        self.side6 = nn.Conv3d(filter[4], out_ch, 3, padding=1)

        self.outconv = nn.Conv3d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample(d6, d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        if self.side:
            return d0
        if self.sup:
            return [d0, d1, d2, d3, d4, d5, d6]
        return d1


if __name__ == '__main__':
    from torch.autograd import Variable

    var = torch.rand(1, 64, 32, 32, 32).cuda()

    model = RSU6(64, 64, 128, ).cuda()
    macs, params = get_model_complexity_info(model, (64, 32, 32, 32), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(var)
    print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print(model)
#
# # -*-coding:utf-8 -*-
# """
# # Time       ：2022/9/1 20:44
# # Author     ：comi
# # version    ：python 3.8
# # Description：
# # todo u2netV2
# # todo 通道打乱，# 还没实现 改用cat ，废弃add
# """
# import torch
# from ptflops import get_model_complexity_info
# from torch import nn
# from torch.nn.functional import interpolate
# from torch.utils import checkpoint
#
# from models.u2net3p.cbam import CBAM
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         padding = (kernel_size - 1) // 2
#         self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.conv1(out)
#         out = self.sigmoid(out)
#         return out
#
#
# def channel_shuffle(x, groups):
#     batch_size, num_channels, height, width, z = x.size()
#     channels_per_group = num_channels // groups
#
#     # reshape
#     x = x.view(batch_size, groups, channels_per_group, height, width, z)
#
#     x = torch.transpose(x, 1, 2).contiguous()
#
#     # flatten
#     x = x.view(batch_size, -1, height, width, z)
#     return x
#
#
# def channel_spilt(x):
#     # 分别用于残差分支和u结构
#     x1, x2 = x.chunk(2, dim=1)
#     return x1, x2
#
#
# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=0, norm_layer=None):
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm3d
#         super(ConvBNReLU, self).__init__(
#             nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
#             norm_layer(out_planes),
#             nn.ReLU(inplace=False)
#         )
#
#
# class DEPTHWISECONV(nn.Module):
#     """
#     深度可分离卷积V1
#     """
#
#     def __init__(self, in_ch, out_ch, checkpoint=False):
#         super(DEPTHWISECONV, self).__init__()
#         self.checkpoint = checkpoint
#         # 也相当于分组为1的分组卷积
#         self.dw = nn.Sequential(
#             # todo 逐层卷积
#             nn.Conv3d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1,
#                       groups=in_ch, bias=False, ),
#             nn.BatchNorm3d(in_ch),
#             nn.ReLU(inplace=False),
#
#             # 逐点卷积
#             nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0,
#                       groups=1, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=False),
#         )
#
#     def forward(self, input):
#         if self.checkpoint:
#             out = checkpoint.checkpoint(self.dw, input)
#         else:
#             out = self.dw(input)
#         return out
#
#
# class InvertedResidual(nn.Module):
#     """
#     mobile net v2
#     """
#
#     def __init__(self, inp, oup, stride, expand_ratio=6, norm_layer=nn.BatchNorm3d):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = int(round(inp * expand_ratio))
#         print(hidden_dim)
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         layers = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
#
#         layers.extend([
#             # dw
#             ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
#             # pw-linear
#             nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
#             norm_layer(oup),
#         ])
#         self.conv = nn.Sequential(*layers)
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
# class U2baseblock(nn.Module):
#     def __init__(self, in_ch=3, out_ch=3, dirate=1, checkpoint=False):
#         super(U2baseblock, self).__init__()
#         self.checkpoint = checkpoint
#         self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
#         self.bn = nn.BatchNorm3d(out_ch)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         if self.checkpoint:
#             x = checkpoint.checkpoint(self.conv, x)
#         else:
#             x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
#
# class baseBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, dirate=1, dep=True, checkpoint=False):
#         super(baseBlock, self).__init__()
#         self.checkpoint = checkpoint
#         self.dep = dep
#         if dep:
#             self.conv = DEPTHWISECONV(in_ch, out_ch)
#         else:
#             self.conv = U2baseblock(in_ch, out_ch, dirate)
#
#     def forward(self, x):
#         if self.checkpoint:
#             return checkpoint.checkpoint(self.conv, x)
#         else:
#             return self.conv(x)
#
#
# class singleKernal(nn.Module):
#     def __init__(self, inc, outc, p=1, size=3, ):
#         super(singleKernal, self).__init__()
#         if p == 1:
#             self.conv = nn.Conv3d(inc, outc, (size, 1, 1), stride=1, padding=(1, 0, 0))
#         elif p == 2:
#             self.conv = nn.Conv3d(inc, outc, (1, size, 1), stride=1, padding=(0, 1, 0))
#         elif p == 3:
#             self.conv = nn.Conv3d(inc, outc, (1, 1, size), stride=1, padding=(0, 0, 1))
#         else:
#             self.conv = nn.Conv3d(inc, outc, (1, 1, 1), stride=1)
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm3d(outc)
#
#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))
#
#
# class mutiScale(nn.Module):
#     def __init__(self, inc, outc, dirate=1, dep=False):
#         super(mutiScale, self).__init__()
#         size = 3
#         self.lenear1 = singleKernal(inc, outc, p=1, size=size)
#         self.lenear2 = singleKernal(inc, outc, p=2, size=size)
#         self.lenear3 = singleKernal(inc, outc, p=3, size=size)
#         self.lenear4 = singleKernal(inc, outc, p=4, size=size)
#         # self.conv = baseBlock(mid, mid, dirate=dirate, dep=False)
#
#     def forward(self, x):
#         conv1 = self.lenear1(x)
#         conv2 = self.lenear2(x)
#         conv3 = self.lenear3(x)
#         conv4 = self.lenear4(x)
#         # conv5 = self.conv(x)
#         return conv1 + conv2 + conv3 + conv4
#
#
# def _upsample(src, tar):
#     return interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)
#
#
# class posAttn(nn.Module):
#
#     def __init__(self, channel, depth):  #
#         super(posAttn, self).__init__()
#
#         self.depth = depth
#         self.layers = nn.ModuleList()
#         for i in range(depth):
#             self.layers.append(CBAM(channel, channel))
#             self.layers.append(nn.MaxPool3d(2, 2))
#
#     def forward(self, x):
#
#         assert len(x) == self.depth, 'depth error'
#         pool = None
#         results = []
#         idx = 0
#         for i, layer in enumerate(self.layers):
#
#             if i % 2 == 0:
#                 if pool is not None:
#                     features = x[idx] + pool
#                 else:
#                     features = x[idx]
#                 features = layer(features)
#                 results.append(features)
#                 idx += 1
#             else:
#                 if pool is None:
#                     pool = features
#                 pool = layer(pool)
#
#         return results
#
#
# class RSU7(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, side=False, upsample=False, dep=False):
#         super(RSU7, self).__init__()
#         self.side = side
#         self.upsample = upsample
#
#         # self.posAttn = posAttn(mid_ch, 5)
#
#         self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)
#
#         self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)
#         self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv5 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv6 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#
#         self.rebnconv7 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)
#
#         self.rebnconv6d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv5d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv4d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)
#
#     def forward(self, x):
#         hx = x
#
#         hxin = self.rebnconvin(hx)
#
#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)
#
#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)
#
#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)
#
#         hx4 = self.rebnconv4(hx)
#         hx = self.pool4(hx4)
#
#         hx5 = self.rebnconv5(hx)
#         hx = self.pool5(hx5)
#
#         hx6 = self.rebnconv6(hx)  # RSU7: 9*9*16
#
#         hx7 = self.rebnconv7(hx6)
#
#         # cbam
#         # pa2, pa3, pa4, pa5, pa6 = self.posAttn([hx2, hx3, hx4, hx5, hx6])
#
#         # hx6d = self.rebnconv6d(torch.cat((hx7, pa6), 1))  # plan 3
#         # hx6dup = _upsample(hx6d, hx5)
#         #
#         # hx5d = self.rebnconv5d(torch.cat((hx6dup, pa5), 1))
#         # hx5dup = _upsample(hx5d, hx4)
#         #
#         # hx4d = self.rebnconv4d(torch.cat((hx5dup, pa4), 1))
#         # hx4dup = _upsample(hx4d, hx3)
#         #
#         # hx3d = self.rebnconv3d(torch.cat((hx4dup, pa3), 1))
#         # hx3dup = _upsample(hx3d, hx2)
#         #
#         # hx2d = self.rebnconv2d(torch.cat((hx3dup, pa2), 1))
#         # hx2dup = _upsample(hx2d, hx1)
#
#         hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))  # normal
#         hx6dup = _upsample(hx6d, hx5)
#
#         hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
#         hx5dup = _upsample(hx5d, hx4)
#
#         hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
#         hx4dup = _upsample(hx4d, hx3)
#
#         hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
#         hx3dup = _upsample(hx3d, hx2)
#
#         hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
#         hx2dup = _upsample(hx2d, hx1)
#
#         hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
#         return hx1d  # + hxin  # 不加hxin u block
#
#
# # class RSU7(nn.Module):
# #
# #     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, side=False, upsample=False, dep=False):
# #         super(RSU7, self).__init__()
# #         self.side = side
# #         self.upsample = upsample
# #
# #         # self.posAttn = posAttn(mid_ch, 5)
# #
# #         self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)
# #
# #         self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)
# #         self.sp1 = SpatialAttention(7)
# #         self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
# #
# #         self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
# #         self.sp2 = SpatialAttention(5)
# #         self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
# #
# #         self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
# #         self.sp3 = SpatialAttention(5)
# #         self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
# #
# #         self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
# #         self.sp4 = SpatialAttention(3)
# #         self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
# #
# #         self.rebnconv5 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
# #         self.sp5 = SpatialAttention(3)
# #         self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
# #
# #         self.rebnconv6 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
# #
# #         self.rebnconv7 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)
# #
# #         self.rebnconv6d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
# #
# #         self.sp5d = SpatialAttention(3)
# #         self.rebnconv5d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
# #
# #         self.sp4d = SpatialAttention(3)
# #         self.rebnconv4d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
# #
# #         self.sp3d = SpatialAttention(5)
# #         self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
# #
# #         self.sp2d = SpatialAttention(5)
# #         self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
# #
# #         self.sp1d = SpatialAttention(7)
# #         self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)
# #
# #     def forward(self, x):
# #         hx = x
# #
# #         hxin = self.rebnconvin(hx)
# #
# #         hx1 = self.rebnconv1(hxin)
# #         hx = self.pool1(hx1 * self.sp1(hx1))
# #
# #         hx2 = self.rebnconv2(hx)
# #         hx = self.pool2(hx2 * self.sp2(hx2))
# #
# #         hx3 = self.rebnconv3(hx)
# #         hx = self.pool3(hx3 * self.sp3(hx3))
# #
# #         hx4 = self.rebnconv4(hx)
# #         hx = self.pool4(hx4 * self.sp4(hx4))
# #
# #         hx5 = self.rebnconv5(hx)
# #         hx = self.pool5(hx5 * self.sp1(hx5))
# #
# #         hx6 = self.rebnconv6(hx)  # RSU7: 9*9*16
# #
# #         hx7 = self.rebnconv7(hx6)
# #
# #         # cbam
# #         # pa2, pa3, pa4, pa5, pa6 = self.posAttn([hx2, hx3, hx4, hx5, hx6])
# #
# #         # hx6d = self.rebnconv6d(torch.cat((hx7, pa6), 1))  # plan 3
# #         # hx6dup = _upsample(hx6d, hx5)
# #         #
# #         # hx5d = self.rebnconv5d(torch.cat((hx6dup, pa5), 1))
# #         # hx5dup = _upsample(hx5d, hx4)
# #         #
# #         # hx4d = self.rebnconv4d(torch.cat((hx5dup, pa4), 1))
# #         # hx4dup = _upsample(hx4d, hx3)
# #         #
# #         # hx3d = self.rebnconv3d(torch.cat((hx4dup, pa3), 1))
# #         # hx3dup = _upsample(hx3d, hx2)
# #         #
# #         # hx2d = self.rebnconv2d(torch.cat((hx3dup, pa2), 1))
# #         # hx2dup = _upsample(hx2d, hx1)
# #
# #         hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))  # normal
# #         hx6dup = _upsample(hx6d, hx5)
# #
# #         hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
# #         hx5dup = _upsample(hx5d * self.sp5d(hx5d), hx4)
# #
# #         hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
# #         hx4dup = _upsample(hx4d * self.sp4d(hx4d), hx3)
# #
# #         hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
# #         hx3dup = _upsample(hx3d * self.sp3d(hx3d), hx2)
# #
# #         hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
# #         hx2dup = _upsample(hx2d * self.sp2d(hx2d), hx1)
# #
# #         hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
# #         hx1d = hx1d * self.sp1d(hx1d)
# #         return hx1d  # + hxin  # 不加hxin u block
#
#
# class RSU6(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, side=False, upsample=False, dep=False):
#         super(RSU6, self).__init__()
#         self.side = side
#         self.upsample = upsample
#         # self.res = nn.Sequential(
#         #     nn.Conv3d(in_ch // 2, out_ch // 2, kernel_size=1, padding=0, dilation=1),
#         #     nn.BatchNorm3d(out_ch // 2),
#         #     nn.ReLU()
#         # )
#         #
#         # self.rebnconvin = baseBlock(in_ch // 2, out_ch // 2, dirate=1, dep=dep)
#         #
#         # self.rebnconv1 = baseBlock(out_ch // 2, mid_ch, dirate=1, dep=dep)
#
#         # self.posAttn = posAttn(mid_ch, 4)
#
#         self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)
#
#         self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)
#
#         self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv5 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#
#         self.rebnconv6 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)
#
#         self.rebnconv5d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv4d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         # self.rebnconv1d = baseBlock(mid_ch * 2, out_ch // 2, dirate=1, dep=dep)
#         self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)
#
#     def forward(self, x):
#         hx = x
#
#         # cx, ux = channel_spilt(x)
#         # cx = self.res(cx)
#         #
#         # hxin = self.rebnconvin(ux)
#
#         hxin = self.rebnconvin(hx)
#
#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)
#
#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)
#
#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)
#
#         hx4 = self.rebnconv4(hx)
#         hx = self.pool4(hx4)
#
#         hx5 = self.rebnconv5(hx)
#
#         hx6 = self.rebnconv6(hx5)
#
#         # cbam plan 2
#         # pa2, pa3, pa4, pa5 = self.posAttn([hx2, hx3, hx4, hx5])
#         # hx5d = self.rebnconv5d(torch.cat((hx6, hx5 + pa5), 1))
#         # hx5dup = _upsample(hx5d, hx4)
#         # hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4 + pa4), 1))
#         # hx4dup = _upsample(hx4d, hx3)
#         # hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3 + pa3), 1))
#         # hx3dup = _upsample(hx3d, hx2)
#         # hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2 + pa2), 1))
#         # hx2dup = _upsample(hx2d, hx1)
#
#         hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
#         hx5dup = _upsample(hx5d, hx4)
#         hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
#         hx4dup = _upsample(hx4d, hx3)
#         hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
#         hx3dup = _upsample(hx3d, hx2)
#         hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
#         hx2dup = _upsample(hx2d, hx1)
#
#         hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
#         # x = channel_shuffle(torch.cat((hx1d + hxin, cx), dim=1), 4)  #
#         return hx1d  # + hxin
#
#
# class RSU5(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, side=False, upsample=False, dep=False):
#         super(RSU5, self).__init__()
#         self.side = side
#         self.upsample = upsample
#         # self.res = nn.Sequential(
#         #     nn.Conv3d(in_ch // 2, out_ch // 2, kernel_size=1, padding=0, dilation=1),
#         #     nn.BatchNorm3d(out_ch // 2),
#         #     nn.ReLU()
#         # )
#         #
#         # self.rebnconvin = baseBlock(in_ch // 2, out_ch // 2, dirate=1, dep=dep)
#         #
#         # self.rebnconv1 = baseBlock(out_ch // 2, mid_ch, dirate=1, dep=dep)
#
#         # self.posAttn = posAttn(mid_ch, 3)
#
#         self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)
#
#         self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)
#         self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#
#         self.rebnconv5 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)
#
#         self.rebnconv4d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#
#         # self.rebnconv1d = baseBlock(mid_ch * 2, out_ch // 2, dirate=1, dep=dep)
#         self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)
#
#     def forward(self, x):
#         hx = x
#
#         # cx, ux = channel_spilt(x)
#         # cx = self.res(cx)
#         # hxin = self.rebnconvin(ux)
#
#         hxin = self.rebnconvin(hx)
#
#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)
#
#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)
#
#         hx3 = self.rebnconv3(hx)
#         hx = self.pool3(hx3)
#
#         hx4 = self.rebnconv4(hx)
#
#         hx5 = self.rebnconv5(hx4)
#
#         # cbam
#         # pa2, pa3, pa4 = self.posAttn([hx2, hx3, hx4])
#         # hx4d = self.rebnconv4d(torch.cat((hx5, hx4 + pa4), 1))
#         # hx4dup = _upsample(hx4d, hx3)
#         # hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3 + pa3), 1))
#         # hx3dup = _upsample(hx3d, hx2)
#         # hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2 + pa2), 1))
#         # hx2dup = _upsample(hx2d, hx1)
#
#         hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
#         hx4dup = _upsample(hx4d, hx3)
#         hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
#         hx3dup = _upsample(hx3d, hx2)
#         hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
#         hx2dup = _upsample(hx2d, hx1)
#
#         hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
#         # x = channel_shuffle(torch.cat((hx1d + hxin, cx), dim=1), 4)  #
#         return hx1d  # hx1d + hxin
#
#
# class RSU4(nn.Module):  # UNet04DRES(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, side=False, dep=False, upsample=False):
#         super(RSU4, self).__init__()
#         self.upsample = upsample
#         self.side = side
#         # self.res = nn.Sequential(
#         #     nn.Conv3d(in_ch // 2, out_ch // 2, kernel_size=1, padding=0, dilation=1),
#         #     nn.BatchNorm3d(out_ch // 2),
#         #     nn.ReLU()
#         # )
#         # self.rebnconvin = baseBlock(in_ch // 2, out_ch // 2, dirate=1, dep=dep)
#         #
#         # self.rebnconv1 = baseBlock(out_ch // 2, mid_ch, dirate=1, dep=dep)
#
#         # self.posAttn = posAttn(mid_ch, 2)
#
#         self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)
#         #
#         self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)
#
#         self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#         self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1, dep=dep)
#
#         self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)
#
#         self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#         self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1, dep=dep)
#
#         # self.rebnconv1d = baseBlock(mid_ch * 2, out_ch // 2, dirate=1, dep=dep)
#
#         self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)
#
#     def forward(self, x):
#         hx = x
#
#         # cx, ux = channel_spilt(x)
#         # cx = self.res(cx)
#         # hxin = self.rebnconvin(ux)
#
#         hxin = self.rebnconvin(hx)
#
#         hx1 = self.rebnconv1(hxin)
#         hx = self.pool1(hx1)
#
#         hx2 = self.rebnconv2(hx)
#         hx = self.pool2(hx2)
#
#         hx3 = self.rebnconv3(hx)  # 9*9*16
#
#         hx4 = self.rebnconv4(hx3)
#
#         # cbam
#         # pa2, pa3 = self.posAttn([hx2, hx3])
#         # hx3d = self.rebnconv3d(torch.cat((hx4, hx3 + pa3), 1))
#         # hx3dup = _upsample(hx3d, hx2)
#         #
#         # hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2 + pa2), 1))
#         # hx2dup = _upsample(hx2d, hx1)
#
#         hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
#         hx3dup = _upsample(hx3d, hx2)
#
#         hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
#         hx2dup = _upsample(hx2d, hx1)
#
#         hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
#         # x = channel_shuffle(torch.cat((hx1d + hxin, cx), dim=1), 4)  # + hxin
#         return hx1d  # hx1d + hxin
#
#
# class RSU4F(nn.Module):
#
#     def __init__(self, in_ch=3, mid_ch=12, out_ch=3, side=False, upsample=False, dep=False):
#         super(RSU4F, self).__init__()
#         self.side = side
#         self.upsample = upsample
#         # self.res = nn.Sequential(
#         #     nn.Conv3d(in_ch // 2, out_ch // 2, kernel_size=1, padding=0, dilation=1),
#         #     nn.BatchNorm3d(out_ch // 2),
#         #     nn.ReLU()
#         # )
#         #
#         # self.rebnconvin = baseBlock(in_ch // 2, out_ch // 2, dirate=1, dep=dep)
#         #
#         # self.rebnconv1 = baseBlock(out_ch // 2, mid_ch, dirate=1, dep=dep)
#
#         self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1, dep=dep)
#
#         self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1, dep=dep)
#
#         self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=2, dep=False)
#         self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=4, dep=False)
#
#         self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=8, dep=False)
#
#         self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=4, dep=False)
#         self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=2, dep=False)
#
#         # self.rebnconv1d = baseBlock(mid_ch * 2, out_ch // 2, dirate=1, dep=dep)
#         self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1, dep=dep)
#
#     def forward(self, x):
#         hx = x
#         # cx, ux = channel_spilt(x)
#         # cx = self.res(cx)
#         # hxin = self.rebnconvin(ux)
#
#         hxin = self.rebnconvin(hx)
#         hx1 = self.rebnconv1(hxin)
#         hx2 = self.rebnconv2(hx1)
#         hx3 = self.rebnconv3(hx2)
#         hx4 = self.rebnconv4(hx3)
#
#         hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
#         hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
#
#         hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
#         # x = channel_shuffle(torch.cat((hx1d + hxin, cx), dim=1), 4)  #
#         return hx1d  # hx1d + hxin
#
#
# class shuffleU2net(nn.Module):
#
#     def __init__(self, in_ch=3, out_ch=1, filter=None, side=False, sup=False, expR=True):
#         super(shuffleU2net, self).__init__()
#         self.side = side
#         self.sup = sup
#         times = 2
#         if filter is None:
#             # filter = [64, 128, 256, 512, 512]
#             filter = [32, 64, 128, 256, 256]
#
#         self.stage1 = RSU7(in_ch, filter[0] // times, filter[0])
#         self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage2 = RSU6(filter[0], filter[0] // times, filter[1])
#         self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage3 = RSU5(filter[1], filter[1] // times, filter[2])
#         self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage4 = RSU4(filter[2], filter[2] // times, filter[3])
#         self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage5 = RSU4F(filter[3], filter[3] // times, filter[4])
#         self.pool56 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage6 = RSU4F(filter[4], filter[4] // times, filter[4])
#
#         # decoder
#         self.stage5d = RSU4F(filter[4] * 2, filter[4] // times, filter[4])
#         self.stage4d = RSU4(filter[4] * 2, filter[2] // times, filter[2])
#         self.stage3d = RSU5(filter[3], filter[1] // times, filter[1])
#         self.stage2d = RSU6(filter[2], filter[0] // times, filter[0])
#         self.stage1d = RSU7(filter[1], filter[0] // times, filter[0])
#
#         self.side1 = nn.Conv3d(filter[0], out_ch, 3, padding=1)
#         self.side2 = nn.Conv3d(filter[0], out_ch, 3, padding=1)
#         self.side3 = nn.Conv3d(filter[1], out_ch, 3, padding=1)
#         self.side4 = nn.Conv3d(filter[2], out_ch, 3, padding=1)
#         self.side5 = nn.Conv3d(filter[3], out_ch, 3, padding=1)
#         self.side6 = nn.Conv3d(filter[4], out_ch, 3, padding=1)
#
#         self.outconv = nn.Conv3d(6, out_ch, 1)
#
#     def forward(self, x):
#         hx = x
#
#         # stage 1
#         hx1 = self.stage1(hx)
#         hx = self.pool12(hx1)
#
#         # stage 2
#         hx2 = self.stage2(hx)
#         hx = self.pool23(hx2)
#
#         # stage 3
#         hx3 = self.stage3(hx)
#         hx = self.pool34(hx3)
#
#         # stage 4
#         hx4 = self.stage4(hx)
#         hx = self.pool45(hx4)
#
#         # stage 5
#         hx5 = self.stage5(hx)
#         hx = self.pool56(hx5)
#
#         # stage 6
#         hx6 = self.stage6(hx)
#         hx6up = _upsample(hx6, hx5)
#
#         # -------------------- decoder --------------------
#         hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
#         hx5dup = _upsample(hx5d, hx4)
#
#         hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
#         hx4dup = _upsample(hx4d, hx3)
#
#         hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
#         hx3dup = _upsample(hx3d, hx2)
#
#         hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
#         hx2dup = _upsample(hx2d, hx1)
#
#         hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
#
#         # side output
#         d1 = self.side1(hx1d)
#
#         d2 = self.side2(hx2d)
#         d2 = _upsample(d2, d1)
#
#         d3 = self.side3(hx3d)
#         d3 = _upsample(d3, d1)
#
#         d4 = self.side4(hx4d)
#         d4 = _upsample(d4, d1)
#
#         d5 = self.side5(hx5d)
#         d5 = _upsample(d5, d1)
#
#         d6 = self.side6(hx6)
#         d6 = _upsample(d6, d1)
#         d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
#         if self.side:
#             return d0
#         if self.sup:
#             return [d0, d1, d2, d3, d4, d5, d6]
#         return d1
#
#
# if __name__ == '__main__':
#     from torch.autograd import Variable
#
#     var = torch.rand(1, 64, 32, 32, 32).cuda()
#
#     model = RSU6(64, 64, 128, ).cuda()
#     macs, params = get_model_complexity_info(model, (64, 32, 32, 32), as_strings=True,
#                                              print_per_layer_stat=False, verbose=False)
#     y = model(var)
#     print('Output shape:', y.shape)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#     # print(model)
