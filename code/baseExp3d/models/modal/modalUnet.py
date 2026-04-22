# -*-coding:utf-8 -*-
"""
# Time       ：2023/3/27 14:39
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
import torch.nn as nn
from torch.autograd import Variable


class DoubleConv(nn.Module):
    """定义一个包含两个卷积层的块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MultiModalUnet(nn.Module):
    """定义多模态3D U-Net模型"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose3d(384, 128, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose3d(192, 64, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(64, 32)

        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        ct = x[:, 0:1, :, :, :]
        mri = x[:, 1:2, :, :, :]
        # 进行编码器操作
        ct1 = self.conv1(ct)
        ct2 = self.conv2(self.pool1(ct1))
        ct3 = self.conv3(self.pool2(ct2))
        ct4 = self.conv4(self.pool3(ct3))

        mri1 = self.conv1(mri)
        mri2 = self.conv2(self.pool1(mri1))
        mri3 = self.conv3(self.pool2(mri2))
        mri4 = self.conv4(self.pool3(mri3))

        # 进行解码器操作
        x = torch.cat([ct4, mri4], dim=1)
        x = self.conv5(self.up1(x))
        x = torch.cat([ct3, mri3, x], dim=1)
        x = self.conv6(self.up2(x))
        x = torch.cat([ct2, mri2, x], dim=1)
        x = self.conv7(self.up3(x))
        x = self.out_conv(x)
        return x


if __name__ == '__main__':
    SIZE = 64
    x = Variable(torch.rand(1, 2, SIZE, SIZE, SIZE)).cuda()
    # model = DualPathDenseNet(3, 1).cuda()
    model = MultiModalUnet(1, 1).cuda()
    out = model(x)
    print(out.shape)
