# -*-coding:utf-8 -*-
"""
# Time       ：2022/7/23 19:49
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
from torch import nn


class SpatialBlock(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(SpatialBlock, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        return x


class SpatialAttention(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, dirate=[1, 2, 3, 2, 1], ):
        super(SpatialAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool3d(2)
        self.avgpool = nn.AvgPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        tmpc = 8
        self.stage1 = SpatialBlock(in_channel, tmpc, dirate[0])
        self.stage2 = SpatialBlock(tmpc, tmpc * 2, dirate[1])
        self.bottleneck = SpatialBlock(tmpc * 2, tmpc * 2, dirate[2])
        self.upStage2 = SpatialBlock(tmpc * 2, tmpc * 2, dirate[3])
        self.upStage1 = SpatialBlock(tmpc * 4, tmpc, dirate[4])
        self.out = nn.Conv3d(tmpc * 2, out_channel, 1, )

        self.AVG = nn.MaxPool3d(8, 4, )
        self.MAX = nn.AvgPool3d(8, 4, )

    def forward(self, x):
        normalX = x
        # u structure
        stage1 = self.stage1(x)
        stage1 = self.maxpool(stage1)

        stage2 = self.stage2(stage1)
        stage2 = self.maxpool(stage2)

        bottleneck = self.bottleneck(stage2)

        maxpool = self.MAX(x)
        avgpool = self.AVG(x)

        weight = self.sigmoid(maxpool + avgpool)
        upstage2 = self.upStage2(bottleneck * weight)
        cat2 = torch.cat((stage2, upstage2), dim=1)
        upStage1 = self.upsample(cat2)

        upStage1 = self.upStage1(upStage1)
        cat2 = torch.cat((stage1, upStage1), dim=1)
        up = self.upsample(cat2)

        out = self.out(up)

        return self.sigmoid(out) * normalX


if __name__ == '__main__':
    size = 8
    x = torch.randn((1, 1, size, size, size))
    SA = SpatialAttention(1, 1)
    x = SA(x)
    print(x.shape)
