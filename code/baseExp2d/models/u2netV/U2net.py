# -*-coding:utf-8 -*-
"""
# Time       ：2022/7/17 16:41
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
from ptflops import get_model_complexity_info
from torch import nn
from torch.nn.functional import interpolate


class U2baseblock(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, relu='relu', norm='batch'):
        super(U2baseblock, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        if norm == 'batch':
            self.bn = nn.BatchNorm3d(out_ch)
        else:
            self.bn = nn.InstanceNorm3d(out_ch)

        if relu == 'relu':
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def _upsample(src, tar):
    # print(src.shape, tar.shape)
    # src = F.upsample(src, size=tar.shape[2:], mode='trilinear', align_corners=True)

    return interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)


class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, lr=True):
        super(RSU7, self).__init__()
        self.lr = lr
        self._upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.rebnconvin = U2baseblock(in_ch, out_ch, dirate=1)
        self.rebnconv1 = U2baseblock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = U2baseblock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = U2baseblock(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = U2baseblock(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = U2baseblock(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = U2baseblock(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = U2baseblock(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = U2baseblock(mid_ch * 2, out_ch, dirate=1)

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

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = self._upsample2(hx6d)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = self._upsample2(hx5d)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self._upsample2(hx4d)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self._upsample2(hx3d)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self._upsample2(hx2d)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        if self.lr:
            return hx1d  # + hxin
        return hx1d + hxin


class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self._upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.rebnconvin = U2baseblock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = U2baseblock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = U2baseblock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = U2baseblock(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = U2baseblock(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = U2baseblock(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = U2baseblock(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = U2baseblock(mid_ch * 2, out_ch, dirate=1)

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

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = self._upsample2(hx5d)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self._upsample2(hx4d)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self._upsample2(hx3d)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self._upsample2(hx2d)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self._upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.rebnconvin = U2baseblock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = U2baseblock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = U2baseblock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = U2baseblock(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = U2baseblock(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = U2baseblock(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = U2baseblock(mid_ch * 2, out_ch, dirate=1)

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

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = self._upsample2(hx4d)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self._upsample2(hx3d)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self._upsample2(hx2d)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self._upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.rebnconvin = U2baseblock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = U2baseblock(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = U2baseblock(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = U2baseblock(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = U2baseblock(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = U2baseblock(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = U2baseblock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)  # 9*9*16

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = self._upsample2(hx3d)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self._upsample2(hx2d)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = U2baseblock(in_ch, out_ch, dirate=1)

        self.rebnconv1 = U2baseblock(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = U2baseblock(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = U2baseblock(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = U2baseblock(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = U2baseblock(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = U2baseblock(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = U2baseblock(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, filter=None, side=False, sup=False):
        super(U2NET, self).__init__()
        self.side = side
        self.sup = sup

        if filter is None:
            filter = [32, 64, 128, 256, 256]

        self.stage1 = RSU7(in_ch, filter[0] // 2, filter[0])
        self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(filter[0], filter[0] // 2, filter[1])
        self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(filter[1], filter[1] // 2, filter[2])
        self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(filter[2], filter[2] // 2, filter[3])
        self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(filter[3], filter[3] // 2, filter[4])
        self.pool56 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(filter[4], filter[4] // 2, filter[4])

        # decoder
        self.stage5d = RSU4F(filter[4] * 2, filter[4] // 2, filter[4])
        self.stage4d = RSU4(filter[4] * 2, filter[2] // 2, filter[2])
        self.stage3d = RSU5(filter[3], filter[1] // 2, filter[1])
        self.stage2d = RSU6(filter[2], filter[0] // 2, filter[0])
        self.stage1d = RSU7(filter[1], filter[0] // 4, filter[0])

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


### U^2-Net small ###
# class U2NETP(nn.Module):
#
#     def __init__(self, in_ch=3, out_ch=1):
#         super(U2NETP, self).__init__()
#         # _upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#         self.stage1 = RSU7(in_ch, 16, 64)
#         self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage2 = RSU6(64, 16, 64)
#         self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage3 = RSU5(64, 16, 64)
#         self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage4 = RSU4(64, 16, 64)
#         self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage5 = RSU4F(64, 16, 64)
#         self.pool56 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage6 = RSU4F(64, 16, 64)
#
#         # decoder
#         self.stage5d = RSU4F(128, 16, 64)
#         self.stage4d = RSU4(128, 16, 64)
#         self.stage3d = RSU5(128, 16, 64)
#         self.stage2d = RSU6(128, 16, 64)
#         self.stage1d = RSU7(128, 16, 64)
#
#         self.side1 = nn.Conv3d(64, out_ch, 3, padding=1)
#         self.side2 = nn.Conv3d(64, out_ch, 3, padding=1)
#         self.side3 = nn.Conv3d(64, out_ch, 3, padding=1)
#         self.side4 = nn.Conv3d(64, out_ch, 3, padding=1)
#         self.side5 = nn.Conv3d(64, out_ch, 3, padding=1)
#         self.side6 = nn.Conv3d(64, out_ch, 3, padding=1)
#
#         self.outconv = nn.Conv3d(6, out_ch, 1)
#
#     def forward(self, x):
#         hx = x  # (288*288)
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
#         # decoder
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
#         d2 = _upsample(d2, d1)  # 288*288*1
#
#         d3 = self.side3(hx3d)
#         d3 = _upsample(d3, d1)  # 288*288*1
#
#         d4 = self.side4(hx4d)
#         d4 = _upsample(d4, d1)  # 288*288*1
#
#         d5 = self.side5(hx5d)
#         d5 = _upsample(d5, d1)
#
#         d6 = self.side6(hx6)
#         d6 = _upsample(d6, d1)
#
#         d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
#
#         # return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
#         #     d4), torch.sigmoid(d5), torch.sigmoid(d6)
#         # 深监督
#         # 返回多个
#         # side 引导
#         # return d0
#         # normal
#         return d1


if __name__ == '__main__':
    x = torch.rand((1, 64, 32, 32, 32))
    # model = U2NET(1, out_ch=1)
    # # d0, d1, d2, d3, d4, d5, d6 = model(x)
    # macs, params = get_model_complexity_info(model, (1, 64, 64, 64), as_strings=True,
    #                                          print_per_layer_stat=False, verbose=False)
    # y, = model(x)
    # print('Output shape:', y.shape)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # model = U2NET(1, 1, filter=[64, 128, 256, 512, 512])
    # macs, params = get_model_complexity_info(model, (1, 64, 64, 64), as_strings=True,
    #                                          print_per_layer_stat=False, verbose=False)
    model = RSU6(64, 64, 128)
    y, = model(x)
    print('Output shape:', y.shape)
    macs, params = get_model_complexity_info(model, (64, 32, 32, 32), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # u2net
    # Output shape: torch.Size([1, 64, 64, 64])
    # Computational complexity:       201.05 GMac
    # Number of parameters:           131.94 M
    # tp
    # Output shape: torch.Size([1, 64, 64, 64])
    # Computational complexity:       123.25 GMac
    # Number of parameters:           3.37 M
