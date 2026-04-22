# -*-coding:utf-8 -*-
"""
# Time       ：2023/3/26 20:30
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
from torch import nn
from torch.autograd import Variable

from models.swinu2net.U2netV5 import _upsample
from models.u2netV.AttentionModule import AttenModule


class basic_block(nn.Module):
    def __init__(self, inchannels, outchannels, downsample=False):
        super(basic_block, self).__init__()

        self.conv1 = nn.Conv3d(inchannels, inchannels, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm3d(inchannels)

        self.relu = nn.PReLU()
        if downsample:
            self.conv2 = nn.Conv3d(inchannels, outchannels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv2 = nn.Conv3d(inchannels, outchannels, kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.BatchNorm3d(outchannels)

        self.relu2 = nn.PReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class bottleneck(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.bottle = AttenModule('u', channel, 3)

    def forward(self, x):
        return self.bottle(x)


class Deconder(nn.Module):
    def __init__(self, channels=[1, 64, 128, 256]):
        super().__init__()

        self.de1 = basic_block(channels[3] + channels[3], channels[2])

        self.de2 = basic_block(channels[2] + channels[2], channels[1])

        self.de3 = basic_block(channels[1] + channels[1], channels[0])

    def forward(self, x, skip):
        # torch.Size([1, 512, 8, 8, 8])->torch.Size([1, 128, 8, 8, 8])
        # torch.Size([1, 256, 8, 8, 8])->torch.Size([1, 64, 8, 8, 8])
        # torch.Size([1, 128, 8, 8, 8])->torch.Size([1, 1, 8, 8, 8])

        de1 = self.de1(torch.cat([x, skip[-1]], dim=1))

        de2 = _upsample(de1, skip[-2])
        de3 = self.de2(torch.cat([de2, skip[-2]], dim=1))

        de3 = _upsample(de3, skip[-3])
        de3 = self.de3(torch.cat([de3, skip[-3]], dim=1))

        out = _upsample(de3, torch.zeros(1, 1, 64, 64, 64))
        return out


class Enconder(nn.Module):

    def __init__(self, channels=[1, 64, 128, 256]):
        super(Enconder, self).__init__()

        self.en1 = basic_block(channels[0], channels[1], downsample=True)

        self.en2 = basic_block(channels[1], channels[2], downsample=True)

        self.en3 = basic_block(channels[2], channels[3], downsample=True)

        # self.en4 = basic_block(channels[3], channels[3], downsample=True)

    def forward(self, x, up_features=None):
        if up_features is not None:
            x1 = self.en1(x)
            x2 = self.en2(x1 + up_features[0])
            x3 = self.en3(x2 + up_features[1])
            # x4 = self.en4(x3 + up_features[2])
        else:
            x1 = self.en1(x)
            x2 = self.en2(x1)
            x3 = self.en3(x2)
            # x4 = self.en4(x3)
        return x1, x2, x3,  # x4


class DualB(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = Enconder()
        self.border = Enconder()
        self.features = Enconder()

        self.bottle = bottleneck(256)

        self.deconder = Deconder()

    def forward(self, x):
        img, border, pos = x.split(1, dim=1)

        posx = self.pos(pos)

        borderx = self.border(border, posx)

        features = self.features(img, borderx)

        tmp = posx[-1] + borderx[-1]
        tmp = tmp + features[-1]

        bottle = self.bottle(tmp)

        out = self.deconder(bottle, features)

        return out


if __name__ == '__main__':
    SIZE = 64
    x = Variable(torch.rand(1, 3, SIZE, SIZE, SIZE)).cuda()
    # model = Enconder().cuda()
    # x1 = model(x)
    # for x in x1:
    #     print(x.shape)
    #
    # x = Variable(torch.rand(1, 1, SIZE, SIZE, SIZE)).cuda()
    # model = Enconder().cuda()
    # x2 = model(x, x1)
    # for x in x1:
    #     print(x.shape)
    #
    # modelb = bottleneck(256).cuda()
    # xb = modelb(x2[-1])
    #
    # xd = Deconder().cuda()
    # x = xd(xb, x2)
    # print(x.shape)

    model = DualB().cuda()
    x = model(x)
    print(x.shape)
