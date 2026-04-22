import torch
from torch import nn

from models.models_2d.mipt.chdseg.acnet_builder import ACNetBuilder
from models.models_2d.mipt.chdseg.net_parts import DoubleConv, Down, Down_GloRe, Up, OutConv


class UNet_GloRe(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_GloRe, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        builder = ACNetBuilder(base_config=None, deploy=False)
        self.inc = DoubleConv(builder, n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.down4 = Down_GloRe(512, 512)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(builder, 64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    x = torch.randn((1, 1, 64, 64))
    model = UNet_GloRe(1, 1)
    x = model(x)
    print(x.shape)
