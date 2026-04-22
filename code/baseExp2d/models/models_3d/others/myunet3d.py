# -*-coding:utf-8 -*-
"""
# Time       ：2022/4/29 11:22
# Author     ：comi
# version    ：python 3.8
# Description：
"""
from ptflops import get_model_complexity_info

# -*-coding:utf-8 -*-
"""
# Time       ：2022/4/29 10:45
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
import torchvision.transforms.functional as tf
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET3d(nn.Module):

    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNET3d, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # down sample
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = tf.resize(x, size=skip_connection.shape[2:])
            conskip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](conskip)
        return self.final_conv(x)


if __name__ == '__main__':

    from torch.autograd import Variable

    SIZE = 64
    x = Variable(torch.rand(1, 1, SIZE, SIZE, SIZE)).cuda()
    model = UNET3d(1, 1).cuda()

    macs, params = get_model_complexity_info(model, (1, SIZE, SIZE, SIZE), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(x)
    print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
