# -*-coding:utf-8 -*-
"""
# Time       ：2023/3/20 11:39
# Author     ：comi
# version    ：python 3.8
# Description：
# 以rsu5 为 起始模块，加入 “王” 注意力
# 在瓶颈层加入卷积自注意力
# 多级连预测，使用并行，cat，映射1
"""
import torch
from ptflops import get_model_complexity_info
from torch import nn
from torch.utils import checkpoint
from torch.nn.functional import interpolate

from models.u2netV.AttentionModule import SelfAttention3d, CBAM


def _upsample(src, tar):
    return interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)


class U2baseblock(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(U2baseblock, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class baseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super(baseBlock, self).__init__()

        self.conv = U2baseblock(in_ch, out_ch, dirate)

    def forward(self, x):
        try:
            return self.conv(x)
        except MemoryError as e:
            return checkpoint.checkpoint(self.conv, x)


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


class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, ):
        super(RSU5, self).__init__()

        self.posAttn = posAttn(mid_ch, 4)
        self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=1)
        # self.rebnconv4 = selfAttnMolude(mid_ch, 4, parall=True)

        # self.rebnconv5 = baseBlock(mid_ch, mid_ch, dirate=2)
        self.rebnconv5 = selfAttnMolude(mid_ch, 4, parall=True)

        self.rebnconv4d = baseBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1)

        self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1)

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

        hx5 = self.rebnconv5(hx4)

        pa1, pa2, pa3, pa4 = self.posAttn([hx1, hx2, hx3, hx4])

        hx4d = self.rebnconv4d(torch.cat((hx5, pa4), 1))
        hx4dup = _upsample(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, pa3), 1))
        hx3dup = _upsample(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, pa2), 1))
        hx2dup = _upsample(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, pa1), 1))

        return hx1d + hxin


class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.posAttn = posAttn(mid_ch, 3)

        self.rebnconvin = baseBlock(in_ch, out_ch, dirate=1)
        self.rebnconv1 = baseBlock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = baseBlock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = baseBlock(mid_ch, mid_ch, dirate=1)

        # self.rebnconv4 = baseBlock(mid_ch, mid_ch, dirate=2)
        self.rebnconv4 = selfAttnMolude(mid_ch, 4, parall=True)

        self.rebnconv3d = baseBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = baseBlock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = baseBlock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = x

        hxin = self.rebnconvin(hxin)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)  # 9*9*16

        hx4 = self.rebnconv4(hx3)

        pa1, pa2, pa3, = self.posAttn([hx1, hx2, hx3])

        hx3d = self.rebnconv3d(torch.cat((hx4, pa3), 1))
        hx3dup = _upsample(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, pa2), 1))
        hx2dup = _upsample(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, pa1), 1))

        return hx1d + hxin


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, padding=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        super(ConvBNReLU, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=False)
        )


class ChannelAttn(nn.Module):
    """
    通道变换,分辨率不变
    直接maxpool一个较小的维度?
    """

    def __init__(self, mid=32, depth=5, dirate=2, block=ConvBNReLU):
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


class selfAttnMolude(nn.Module):

    def __init__(self, mid, depth=1, parall=False):
        super(selfAttnMolude, self).__init__()
        self.parall = parall
        self.selfattn = nn.ModuleList([
            SelfAttention3d(mid) for _ in range(depth)
        ])
        self.out = nn.Conv3d(mid * depth, mid, kernel_size=3, padding=1)

    def forward(self, x):
        if self.parall:
            features = []
            for layer in self.selfattn:
                tmp = layer(x)
                features.append(tmp)
            x = self.out(torch.cat(features, 1))
        else:
            for layer in self.selfattn:
                x = layer(x)
        return x


class U2PCSNet(nn.Module):

    def __init__(self, in_channels=1, side_len=6):
        super(U2PCSNet, self).__init__()
        self.in_channels = in_channels

        filters = [64, 128, 256]  # 最终实现方案 通道数

        # -------------Encoder--------------
        self.en1 = RSU5(in_channels, filters[0] // 2, filters[0])
        self.pool1 = nn.MaxPool3d(2, 2)

        self.en2 = RSU4(filters[0], filters[1] // 2, filters[1])
        self.pool2 = nn.MaxPool3d(2, 2)

        self.en3 = nn.ModuleList([
            nn.Conv3d(filters[1], filters[2], kernel_size=3, stride=2, padding=1),
            ChannelAttn(filters[2], 5, dirate=2),
        ])

        self.selfattn = selfAttnMolude(filters[2], 4, True)
        self.ca = ChannelAttn(filters[2], 5, dirate=2)
        # self.selfattn = SelfAttention3d(filters[2])
        # -------------Decoder--------------
        self.de3 = nn.ModuleList([
            nn.Conv3d(filters[2] + filters[2], filters[1], kernel_size=3, stride=1, padding=1),
            ChannelAttn(filters[1], 5, dirate=2)
        ])

        self.de2 = RSU4(filters[1] + filters[1], filters[1] // 2, filters[0])

        self.de1 = RSU5(filters[0] + filters[0], filters[0] // 2, filters[0])

        # ----------------------------------------------------------------
        # mid = 32
        # self.side3 = nn.Conv3d(filters[1], mid, kernel_size=3, stride=1, padding=1)
        # self.side2 = nn.Conv3d(filters[0], mid, kernel_size=3, stride=1, padding=1)
        # self.side1 = nn.Conv3d(filters[0], mid, kernel_size=3, stride=1, padding=1)
        # self.attns = nn.ModuleList()
        #
        # for i in range(side_len):
        #     self.attns.append(SelfAttention3d(mid * 3))
        self.out = nn.Conv3d(filters[0], in_channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = inputs
        x_1 = self.en1(x)
        x_1pool = self.pool1(x_1)

        x_2 = self.en2(x_1pool)
        x_2pool = self.pool2(x_2)

        x_3 = x_2pool
        for layer in self.en3:
            x_3 = layer(x_3)

        x_4 = self.selfattn(x_3)
        x_4 = self.ca(x_4)

        #  -----deconders --------------------------------
        x3 = torch.cat((x_4, x_3), 1)
        for layer in self.de3:
            x3 = layer(x3)

        x3_up = _upsample(x3, x_2)
        x2 = self.de2(torch.cat((x3_up, x_2), 1))

        x2_up = _upsample(x2, x_1)
        x1 = self.de1(torch.cat((x2_up, x_1), 1))

        # side3 = self.side3(x_3)
        # side2 = self.side2(x2)
        #
        # side1 = self.side1(x1)
        # side2 = _upsample(side2, side1)
        # side3 = _upsample(side3, side1)
        #
        # features = []
        # for layer in self.attns:
        #     features.append(layer(torch.cat((side1, side2, side3), 1)))
        # out = torch.hstack(features)
        return self.out(x1)


if __name__ == '__main__':
    var = torch.rand(1, 1, 64, 64, 16).cuda()

    model = U2PCSNet(1).cuda()
    macs, params = get_model_complexity_info(model, (1, 64, 64, 16), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(var)
    print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print(model)
