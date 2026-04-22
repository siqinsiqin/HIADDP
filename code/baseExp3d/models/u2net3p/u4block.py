# -*-coding:utf-8 -*-
"""
# Time       ：2022/11/21 21:00
# Author     ：comi
# version    ：python 3.8
# Description：
todo 设计空间通道注意力模块
"""
import torch
from torch import nn
from torch.autograd import Variable

from models.u2netV.shuffleNet import channel_shuffle, _upsample, channel_spilt


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, groups=1, padding=1, norm_layer=None,
                 dilation=1):
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        super(ConvBNReLU, self).__init__(
            # nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
            #           bias=False),
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    """
    mobile net v2
    """

    def __init__(self, inp, oup, stride, expand_ratio=6, norm_layer=nn.BatchNorm3d):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
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


class SpatialAttn(nn.Module):
    """
    depth:actual depth is depth + 1
    """

    def __init__(self, in_out_chans=1, depth=5, dirate=2, pool=True, block=ConvBNReLU):
        super(SpatialAttn, self).__init__()
        self.depth = depth
        self.pool = pool
        mid = in_out_chans // 2
        # todo 空间注意力,通道不变
        self.layers = nn.ModuleList()
        for i_layer in range(depth):
            if i_layer == 0:
                self.layers.append(block(in_out_chans, mid, stride=1))
            else:
                self.layers.append(block(mid, mid, stride=1))
            if pool:
                self.layers.append(nn.MaxPool3d(2, 2))

        self.layers.append(
            nn.Sequential(
                nn.Conv3d(mid, mid, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(mid),
                nn.ReLU(),
            ))
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(mid, mid, kernel_size=3, stride=1, padding=1 * dirate, dilation=dirate),
                nn.BatchNorm3d(mid),
                nn.ReLU(),
            ))
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(mid * 2, mid, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(mid),
                nn.ReLU()
            ))

        for i_layer in range(depth):
            if pool:
                self.layers.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
            if i_layer == depth - 1:
                self.layers.append(block(mid * 2, in_out_chans, stride=1))
            else:
                self.layers.append(block(mid * 2, mid, stride=1))

    def forward(self, x):
        # todo 空间注意力
        enconder = []
        cnt = -1
        for i, layer in enumerate(self.layers):
            if self.pool:
                if i <= ((self.depth * 4) + 2) // 2:
                    x = layer(x)
                    if i % 2 == 0:
                        enconder.append(x)
                else:
                    if i % 2 == 0:
                        x = layer(torch.cat((x, enconder[cnt]), dim=1))
                        cnt -= 1
                    else:
                        x = layer(x)
            else:
                if i <= ((self.depth * 2) + 2) // 2:
                    x = layer(x)
                    enconder.append(x)
                else:
                    x = layer(torch.cat((x, enconder[cnt]), dim=1))
                    cnt -= 1
        return x


# class ChannelAttn(nn.Module):
#     """
#     通道变换,分辨率不变
#     直接maxpool一个较小的维度?
#     """
#
#     def __init__(self, mid=32, depth=3, dirate=2, block=ConvBNReLU):
#         super(ChannelAttn, self).__init__()
#         self.depth = depth
#
#         # todo 通道挤压注意力
#         self.layers = nn.ModuleList()
#         for i_layer in range(depth):
#             tmpChanns = mid // 2
#             self.layers.append(nn.Sequential(
#                 block(mid, tmpChanns, stride=1, padding=2 * dirate, dilation=dirate),
#                 ChannelAttention(tmpChanns, reduction_ratio=2)
#             ))
#             mid = tmpChanns
#
#         self.layers.append(
#             nn.Sequential(nn.Conv3d(mid, mid, kernel_size=3, stride=1, padding=1 * dirate, dilation=dirate),
#                           nn.BatchNorm3d(mid),
#                           nn.ReLU()))
#
#         for i_layer in range(depth):
#             tmpChanns = mid * 2
#             self.layers.append(nn.Sequential(
#                 block(mid * 2, tmpChanns, stride=1, padding=2 * dirate, dilation=dirate),
#                 ChannelAttention(tmpChanns, reduction_ratio=2)
#             ))
#             mid = tmpChanns
#
#     def forward(self, x):
#         hx = x
#
#         enconder = []
#         cnt = -2
#         for i, layer in enumerate(self.layers):
#             if i < ((self.depth * 2) + 2) // 2:
#                 x = layer(x)
#                 enconder.append(x)
#             else:
#                 x = layer(torch.cat((x, enconder[cnt]), dim=1))
#                 cnt -= 1
#
#         return _upsample(x, hx)


class ChannelAttn(nn.Module):
    """
    通道变换,分辨率不变
    直接maxpool一个较小的维度?
    """

    def __init__(self, mid=32, depth=3, dirate=2, block=ConvBNReLU):
        super(ChannelAttn, self).__init__()
        self.depth = depth

        # todo 通道挤压注意力
        self.layers = nn.ModuleList()
        for i_layer in range(depth):
            tmpChanns = mid // 2
            self.layers.append(block(mid, tmpChanns, stride=1))
            mid = tmpChanns

        self.layers.append(
            nn.Sequential(nn.Conv3d(mid, mid, kernel_size=3, stride=1, padding=1 * dirate, dilation=dirate),
                          nn.BatchNorm3d(mid),
                          nn.ReLU()))

        for i_layer in range(depth):
            tmpChanns = mid * 2
            self.layers.append(block(mid * 2, tmpChanns, stride=1))
            mid = tmpChanns

    def forward(self, x):
        hx = x

        enconder = []
        cnt = -2
        for i, layer in enumerate(self.layers):
            if i < ((self.depth * 2) + 2) // 2:
                x = layer(x)
                enconder.append(x)
            else:
                x = layer(torch.cat((x, enconder[cnt]), dim=1))
                cnt -= 1

        return _upsample(x, hx)


class u4block(nn.Module):

    def __init__(self, in_channs=1, out_channs=3, depth=6, dirate=2, split=True, pool=True, mode='sp',
                 block=ConvBNReLU):
        super(u4block, self).__init__()
        self.depth = depth
        self.mode = mode
        self.split = split
        total_depth = 7

        if split:
            self.res = ConvBNReLU(in_channs // 2, out_channs // 2, kernel_size=1, stride=1, padding=0)
            self.rebnconvin = ConvBNReLU(in_channs // 2, out_channs // 2)
            if mode == 'all':
                # 下一步尝试使用原来的rsu
                self.spitial = SpatialAttn(out_channs // 2, depth, dirate, pool, block)
                self.channel = ChannelAttn(out_channs, total_depth - depth, dirate, block)
            elif mode == 'sp':
                self.spitial = SpatialAttn(out_channs // 2, depth, dirate, pool, block)
            elif mode == 'ch':
                self.channel = ChannelAttn(out_channs // 2, total_depth - depth, dirate, block)
        else:
            self.rebnconvin = ConvBNReLU(in_channs, out_channs)
            if mode == 'all':
                self.spitial = SpatialAttn(out_channs, depth, dirate, pool, block)
                self.channel = ChannelAttn(out_channs, total_depth - depth, dirate, block)
            elif mode == 'sp':
                self.spitial = SpatialAttn(out_channs, depth, dirate, pool, block)
            elif mode == 'ch':
                self.channel = ChannelAttn(out_channs, total_depth - depth, dirate, block)

        # self.rebnconout = ConvBNReLU(out_channs, out_channs)

    def forward(self, x):
        if self.split:
            cx, ux = channel_spilt(x)
            cx = self.res(cx)
            hxin = self.rebnconvin(ux)
            if self.mode == 'all':
                x = self.spitial(hxin)
                x = channel_shuffle(torch.cat((x + hxin, cx), dim=1), 4)
                x = self.channel(x) * x
            elif self.mode == 'sp':
                x = self.spitial(hxin)
                x = channel_shuffle(torch.cat((x + hxin, cx), dim=1), 4)
            elif self.mode == 'ch':
                x = self.channel(hxin)
                x = channel_shuffle(torch.cat((x + hxin, cx), dim=1), 4)
        else:
            hx = self.rebnconvin(x)
            if self.mode == 'all':
                x = self.spitial(hx)
                x = x + hx
                x = self.channel(x)
                x = x + hx
            elif self.mode == 'sp':
                x = self.spitial(hx)
                x = x + hx
            elif self.mode == 'ch':
                x = self.channel(hx)
                x = x + hx
        return x


if __name__ == '__main__':
    size = 4
    var = torch.rand(3, 256, size, size, size)
    x = Variable(var).cuda()
    # model = ChannelAttn(mid=256, depth=5).cuda()
    # pool = nn.MaxPool3d(2, 2)
    # model = u4block(in_channs=1, out_channs=32, split=False, depth=5, dirate=2).cuda()
    # x = model(x)
    # # print(model)
    # print('feature ', x.shape)
    # x = pool(x)
    # print('x:pool', x.shape)
    #
    # model = u4block(in_channs=32, out_channs=64, depth=4, dirate=2).cuda()
    # x = model(x)
    # print('feature ', x.shape)
    # x = pool(x)
    # print('x:pool', x.shape)
    #
    # model = u4block(in_channs=64, out_channs=128, depth=3, dirate=2).cuda()
    # x = model(x)
    # print('feature ', x.shape)
    # x = pool(x)
    # print('x:pool', x.shape)
    #
    # model = u4block(in_channs=128, out_channs=256, depth=2, dirate=2).cuda()
    # x = model(x)
    # print('feature ', x.shape)
    # x = pool(x)
    # print('x:pool', x.shape)

    model = u4block(in_channs=256, out_channs=256, depth=2, split=False, pool=False, dirate=2, mode='ch').cuda()
    # model = SpatialAttn(in_out_chans=64).cuda()
    print(model)
    x = model(x)
    print('feature ', x.shape)
