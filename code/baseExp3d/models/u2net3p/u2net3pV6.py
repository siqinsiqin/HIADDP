# -*-coding:utf-8 -*-
"""
# Time       ：2022/8/28 8:34
# Author     ：comi
# version    ：python 3.8
# Description：
# todo 3+ 结构优化，最后的可分离卷积改变为conv，利用上u2net的残差链接用于最终的结构引导
# dice 80.8
"""

import torch
from ptflops import get_model_complexity_info
from torch import nn
from torch.nn.functional import interpolate
from models.swinu2net3p.ResLayers import weights_init_kaiming
# from models.u2netV.U2netV4 import DRSU7, DRSU6, DRSU5, DRSU4, DRSU4F, DEPTHWISECONV


def _upsample(src, tar):
    return interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, downstep=None, block=None, side=False, upsample=False, dep=False):
        super(Block, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.maxpool = None
        self.block = block
        self.return2 = False
        if block is None:  # skip
            self.blockf = DEPTHWISECONV(in_channel, out_channel)
            # self.blockf = nn.Conv3d(in_channel, out_channel, 3, 1, padding=1)
        else:  # base block
            self.blockf = block(self.in_channel, self.in_channel // 2, self.out_channel, side=side, upsample=upsample,
                                dep=dep)
            self.return2 = True

        if downstep is not None:
            self.maxpool = nn.MaxPool3d(downstep, downstep, ceil_mode=True)  # ceil_mode 向上取整

        for m in self.children():
            m.apply(weights_init_kaiming)

    def forward(self, x, target=None):
        if self.maxpool is not None:
            x = self.maxpool(x)

        if target is not None:  # 上采样目标大小
            x = _upsample(x, target)

        if self.return2:
            x, skipx = self.blockf(x)
            return x, skipx
        else:
            x = self.blockf(x)
            return x


# class U2net3p6(nn.Module):
#     """
#     Methods u2net 3+, 修改内部嵌套u结构，可以进行side引导，上采样再下采样，可分离卷积替换
#
#     """
#
#     def __init__(self, in_channel=1, n_classes=1, filters=None, sup=False, encoderOnly=False, side=False):
#         super(U2net3p6, self).__init__()
#         self.sup = sup
#         self.encoderOnly = encoderOnly
#         self.side = side
#         if filters is None:
#             filters = [32, 64, 128, 256, 256]  # [16, 32, 64, 128, 256]
#
#         self.CatChannels = filters[0]
#         self.CatBlocks = 6
#         self.UpChannels = self.CatChannels * self.CatBlocks
#
#         # encoder
#         self.stage1 = DRSU7(in_channel, filters[0] // 2, filters[0])
#         self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage2 = Block(filters[0], filters[1], block=DRSU6, )
#         self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage3 = Block(filters[1], filters[2], block=DRSU5, )
#         self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage4 = Block(filters[2], filters[3], block=DRSU4, )
#         self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage5 = Block(filters[3], filters[4], block=DRSU4F, )
#         self.pool56 = nn.MaxPool3d(2, stride=2, ceil_mode=True)
#
#         self.stage6 = Block(filters[4], filters[4], block=DRSU4F, )
#
#         # decoder
#         """stage 5d"""
#         self.h1_to_hd5 = Block(filters[0], self.CatChannels, downstep=16, )
#         self.h2_to_hd5 = Block(filters[1], self.CatChannels, downstep=8, )
#         self.h3_to_hd5 = Block(filters[2], self.CatChannels, downstep=4, )
#         self.h4_to_hd5 = Block(filters[3], self.CatChannels, downstep=2, )
#         self.h5_Cat_hd5 = Block(filters[4], self.CatChannels, )
#         self.hd6_up_hd5 = Block(filters[4], self.CatChannels, )
#         if self.encoderOnly:
#             self.fusion5d = Block(self.UpChannels, self.UpChannels)
#         else:
#             self.fusion5d = Block(self.UpChannels, self.UpChannels, block=DRSU4F)
#
#         '''stage 4d'''
#         self.h1_to_hd4 = Block(filters[0], self.CatChannels, downstep=8, )
#         self.h2_to_hd4 = Block(filters[1], self.CatChannels, downstep=4, )
#         self.h3_to_hd4 = Block(filters[2], self.CatChannels, downstep=2, )
#         self.h4_Cat_hd4 = Block(filters[3], self.CatChannels, )
#         self.hd5_Up_hd4 = Block(self.UpChannels, self.CatChannels, )
#         self.hd6_Up_hd4 = Block(filters[4], self.CatChannels, )
#         if self.encoderOnly:
#             self.fusion4d = Block(self.UpChannels, self.UpChannels)
#         else:
#             self.fusion4d = Block(self.UpChannels, self.UpChannels, block=DRSU4)
#
#         '''stage 3d'''
#         self.h1_to_hd3 = Block(filters[0], self.CatChannels, downstep=4, )
#         self.h2_to_hd3 = Block(filters[1], self.CatChannels, downstep=2, )
#         self.h3_Cat_hd3 = Block(filters[2], self.CatChannels, )
#         self.hd4_Up_hd3 = Block(self.UpChannels, self.CatChannels, )
#         self.hd5_Up_hd3 = Block(filters[4], self.CatChannels, )
#         self.hd6_Up_hd3 = Block(filters[4], self.CatChannels, )
#         if self.encoderOnly:
#             self.fusion3d = Block(self.UpChannels, self.UpChannels, )
#         else:
#             self.fusion3d = Block(self.UpChannels, self.UpChannels, block=DRSU5)
#
#         '''stage 2d '''
#         self.h1_to_hd2 = Block(filters[0], self.CatChannels, downstep=2, )
#         self.h2_cat_hd2 = Block(filters[1], self.CatChannels, )
#         self.hd3_up_hd2 = Block(self.UpChannels, self.CatChannels, )
#         self.hd4_up_hd2 = Block(filters[3], self.CatChannels, )
#         self.hd5_up_hd2 = Block(filters[4], self.CatChannels, )
#         self.hd6_Up_hd2 = Block(filters[4], self.CatChannels, )
#         if self.encoderOnly:
#             self.fusion2d = Block(self.UpChannels, self.UpChannels, )
#         else:
#             self.fusion2d = Block(self.UpChannels, self.UpChannels, block=DRSU6)
#
#         '''stage 1d'''
#         self.h1_Cat_hd1 = Block(filters[0], self.CatChannels)
#         self.hd2_up_hd1 = Block(self.UpChannels, self.CatChannels, )
#         self.hd3_up_hd1 = Block(filters[2], self.CatChannels, )
#         self.hd4_up_hd1 = Block(filters[3], self.CatChannels, )
#         self.hd5_up_hd1 = Block(filters[4], self.CatChannels, )
#         self.hd6_Up_hd1 = Block(filters[4], self.CatChannels, )
#         if self.encoderOnly:
#             self.fusion1d = Block(self.UpChannels, self.UpChannels, )
#         else:
#             self.fusion1d = Block(self.UpChannels, self.UpChannels, block=DRSU7)
#
#         # DeepSup
#         self.outconv1 = nn.Conv3d(self.UpChannels, n_classes, 3, 1, padding=1)
#         self.outconv2 = nn.Conv3d(self.UpChannels, n_classes, 3, 1, padding=1)
#         self.outconv3 = nn.Conv3d(self.UpChannels, n_classes, 3, 1, padding=1)
#         self.outconv4 = nn.Conv3d(self.UpChannels, n_classes, 3, 1, padding=1)
#         self.outconv5 = nn.Conv3d(self.UpChannels, n_classes, 3, 1, padding=1)
#         self.outconv6 = nn.Conv3d(filters[4], n_classes, 3, 1, padding=1)
#         self.outconv = nn.Conv3d(self.CatBlocks, n_classes, 3, 1, padding=1)
#         # self.outconv1 = DEPTHWISECONV(self.UpChannels, n_classes)
#         # self.outconv2 = DEPTHWISECONV(self.UpChannels, n_classes)
#         # self.outconv3 = DEPTHWISECONV(self.UpChannels, n_classes)
#         # self.outconv4 = DEPTHWISECONV(self.UpChannels, n_classes)
#         # self.outconv5 = DEPTHWISECONV(self.UpChannels, n_classes)
#         # self.outconv6 = DEPTHWISECONV(filters[4], n_classes)
#         # self.outconv = DEPTHWISECONV(self.CatBlocks, n_classes)
#
#         # initialise attns
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 m.apply(weights_init_kaiming)
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.apply(weights_init_kaiming)
#
#     def forward(self, x):
#         ## -------------Encoder-------------
#         hx = x
#
#         # stage 1
#         h1, skip1 = self.stage1(hx)
#         hx = self.pool12(h1)
#
#         # stage 2
#         h2, skip2 = self.stage2(hx)
#         hx = self.pool23(h2)
#
#         # stage 3
#         h3, skip3 = self.stage3(hx)
#         hx = self.pool34(h3)
#
#         # stage 4
#         h4, skip4 = self.stage4(hx)
#         hx = self.pool45(h4)
#
#         # stage 5
#         h5, skip5 = self.stage5(hx)
#         hx = self.pool56(h5)
#
#         # stage 6
#         hd6, skip6 = self.stage6(hx)
#
#         ## -------------Decoder-------------
#
#         h1_to_hd5 = self.h1_to_hd5(skip1)
#         h2_to_hd5 = self.h2_to_hd5(skip2)
#         h3_to_hd5 = self.h3_to_hd5(skip3)
#         h4_to_hd5 = self.h4_to_hd5(skip4)
#         h5_Cat_hd5 = self.h5_Cat_hd5(h5)
#         hd6_up_hd5 = self.hd6_up_hd5(hd6, h5_Cat_hd5)
#         hd5, hd5Res = self.fusion5d(
#             torch.cat((h1_to_hd5, h2_to_hd5, h3_to_hd5, h4_to_hd5, h5_Cat_hd5, hd6_up_hd5), dim=1))
#
#         h1_to_hd4 = self.h1_to_hd4(skip1)
#         h2_to_hd4 = self.h2_to_hd4(skip2)
#         h3_to_hd4 = self.h3_to_hd4(skip3)
#         h4_Cat_hd4 = self.h4_Cat_hd4(h4)
#         hd5_Up_hd4 = self.hd5_Up_hd4(hd5, h4_Cat_hd4)
#         hd6_Up_hd4 = self.hd6_Up_hd4(skip6, h4_Cat_hd4)
#         hd4, hd4Res = self.fusion4d(
#             torch.cat((h1_to_hd4, h2_to_hd4, h3_to_hd4, h4_Cat_hd4, hd5_Up_hd4, hd6_Up_hd4), dim=1))
#
#         h1_to_hd3 = self.h1_to_hd3(skip1)
#         h2_to_hd3 = self.h2_to_hd3(skip2)
#         h3_Cat_hd3 = self.h3_Cat_hd3(h3)
#         hd4_Up_hd3 = self.hd4_Up_hd3(hd4, h3_Cat_hd3)
#         hd5_Up_hd3 = self.hd5_Up_hd3(skip5, h3_Cat_hd3)
#         hd6_up_hd3 = self.hd6_Up_hd3(skip6, h3_Cat_hd3)
#         hd3, hd3Res = self.fusion3d(
#             torch.cat((h1_to_hd3, h2_to_hd3, h3_Cat_hd3, hd4_Up_hd3, hd5_Up_hd3, hd6_up_hd3), dim=1))
#
#         h1_to_hd2 = self.h1_to_hd2(skip1)
#         h2_cat_hd2 = self.h2_cat_hd2(h2)
#         hd3_up_hd2 = self.hd3_up_hd2(hd3, h2_cat_hd2)
#         hd4_up_hd2 = self.hd4_up_hd2(skip4, h2_cat_hd2)
#         hd5_up_hd2 = self.hd5_up_hd2(skip5, h2_cat_hd2)
#         hd6_up_hd2 = self.hd6_Up_hd2(skip6, h2_cat_hd2)
#         hd2, hd2Res = self.fusion2d(
#             torch.cat((h1_to_hd2, h2_cat_hd2, hd3_up_hd2, hd4_up_hd2, hd5_up_hd2, hd6_up_hd2), dim=1))
#
#         h1_Cat_hd1 = self.h1_Cat_hd1(h1)
#         hd2_UT_hd1 = self.hd2_up_hd1(hd2, skip1)
#         hd3_UT_hd1 = self.hd3_up_hd1(skip3, skip1)
#         hd4_UT_hd1 = self.hd4_up_hd1(skip4, skip1)
#         hd5_UT_hd1 = self.hd5_up_hd1(skip5, skip1)
#         hd5_up_hd1 = self.hd6_Up_hd1(skip6, skip1)
#         hd1, hd1Res = self.fusion1d(
#             torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1, hd5_up_hd1), dim=1))
#
#         d6 = self.outconv6(skip6)
#         d5 = self.outconv5(hd5Res)
#         d4 = self.outconv4(hd4Res)
#         d3 = self.outconv3(hd3Res)
#         d2 = self.outconv2(hd2Res)
#         d1 = self.outconv1(hd1Res)
#
#         d6 = _upsample(d6, d1)
#         d5 = _upsample(d5, d1)
#         d4 = _upsample(d4, d1)
#         d3 = _upsample(d3, d1)
#         d2 = _upsample(d2, d1)
#
#         if self.sup:
#             return [d1, d2, d3, d4, d5, d6]
#         elif self.side:
#             d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))
#             return d0
#         return d1


class U2net3p5V6(nn.Module):
    """
    Methods u2net 3+, 修改内部嵌套u结构，可以进行side引导，上采样再下采样，可分离卷积替换

    """

    def __init__(self, in_channel=1, n_classes=1, filters=None, sup=False, encoderOnly=False, side=False):
        super(U2net3p5V6, self).__init__()
        self.sup = sup
        self.encoderOnly = encoderOnly
        self.side = side
        if filters is None:
            filters = [32, 64, 128, 256, 256]  # [64, 128, 256, 512, 512]

        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        # encoder
        self.stage1 = DRSU7(in_channel, filters[0] // 2, filters[0])
        self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage2 = Block(filters[0], filters[1], block=DRSU6, )
        self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage3 = Block(filters[1], filters[2], block=DRSU5, )
        self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage4 = Block(filters[2], filters[3], block=DRSU4, )
        self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage5 = Block(filters[3], filters[4], block=DRSU4F, )
        self.pool56 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        # decoder
        '''stage 4d'''
        self.h1_to_hd4 = Block(filters[0], self.CatChannels, downstep=8, )
        self.h2_to_hd4 = Block(filters[1], self.CatChannels, downstep=4, )
        self.h3_to_hd4 = Block(filters[2], self.CatChannels, downstep=2, )
        self.h4_Cat_hd4 = Block(filters[3], self.CatChannels, )
        self.hd5_Up_hd4 = Block(filters[4], self.CatChannels, )
        if self.encoderOnly:
            self.fusion4d = Block(self.UpChannels, self.UpChannels)
        else:
            self.fusion4d = Block(self.UpChannels, self.UpChannels, block=DRSU4)

        '''stage 3d'''
        self.h1_to_hd3 = Block(filters[0], self.CatChannels, downstep=4, )
        self.h2_to_hd3 = Block(filters[1], self.CatChannels, downstep=2, )
        self.h3_Cat_hd3 = Block(filters[2], self.CatChannels, )
        self.hd4_Up_hd3 = Block(self.UpChannels, self.CatChannels, )
        self.hd5_Up_hd3 = Block(filters[4], self.CatChannels, )
        if self.encoderOnly:
            self.fusion3d = Block(self.UpChannels, self.UpChannels, )
        else:
            self.fusion3d = Block(self.UpChannels, self.UpChannels, block=DRSU5)

        '''stage 2d '''
        self.h1_to_hd2 = Block(filters[0], self.CatChannels, downstep=2, )
        self.h2_cat_hd2 = Block(filters[1], self.CatChannels, )
        self.hd3_up_hd2 = Block(self.UpChannels, self.CatChannels, )
        self.hd4_up_hd2 = Block(filters[3], self.CatChannels, )
        self.hd5_up_hd2 = Block(filters[4], self.CatChannels, )
        if self.encoderOnly:
            self.fusion2d = Block(self.UpChannels, self.UpChannels, )
        else:
            self.fusion2d = Block(self.UpChannels, self.UpChannels, block=DRSU6)

        '''stage 1d'''
        self.h1_Cat_hd1 = Block(filters[0], self.CatChannels)
        self.hd2_up_hd1 = Block(self.UpChannels, self.CatChannels, )
        self.hd3_up_hd1 = Block(filters[2], self.CatChannels, )
        self.hd4_up_hd1 = Block(filters[3], self.CatChannels, )
        self.hd5_up_hd1 = Block(filters[4], self.CatChannels, )
        if self.encoderOnly:
            self.fusion1d = Block(self.UpChannels, self.UpChannels, )
        else:
            self.fusion1d = Block(self.UpChannels, self.UpChannels, block=DRSU7)

        # DeepSup
        # self.outconv1 = nn.Conv3d(self.UpChannels, n_classes, 3, 1, padding=1)
        # self.outconv2 = nn.Conv3d(self.UpChannels, n_classes, 3, 1, padding=1)
        # self.outconv3 = nn.Conv3d(self.UpChannels, n_classes, 3, 1, padding=1)
        # self.outconv4 = nn.Conv3d(self.UpChannels, n_classes, 3, 1, padding=1)
        # self.outconv5 = nn.Conv3d(filters[4], n_classes, 3, 1, padding=1)
        self.outconv = nn.Conv3d(self.CatBlocks, n_classes, 3, 1, padding=1)
        self.outconv1 = DEPTHWISECONV(self.UpChannels, n_classes)
        self.outconv2 = DEPTHWISECONV(self.UpChannels, n_classes)
        self.outconv3 = DEPTHWISECONV(self.UpChannels, n_classes)
        self.outconv4 = DEPTHWISECONV(self.UpChannels, n_classes)
        self.outconv5 = DEPTHWISECONV(filters[4], n_classes)

        # initialise attns
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm3d):
                m.apply(weights_init_kaiming)

    def forward(self, x):
        ## -------------Encoder-------------
        hx = x

        # stage 1
        h1, skip1 = self.stage1(hx)
        hx = self.pool12(h1)

        # stage 2
        h2, skip2 = self.stage2(hx)
        hx = self.pool23(h2)

        # stage 3
        h3, skip3 = self.stage3(hx)
        hx = self.pool34(h3)

        # stage 4
        h4, skip4 = self.stage4(hx)
        hx = self.pool45(h4)

        # stage 5
        hd5, skip5 = self.stage5(hx)

        ## -------------Decoder-------------
        h1_to_hd4 = self.h1_to_hd4(skip1)
        h2_to_hd4 = self.h2_to_hd4(skip2)
        h3_to_hd4 = self.h3_to_hd4(skip3)
        h4_Cat_hd4 = self.h4_Cat_hd4(h4)
        hd5_Up_hd4 = self.hd5_Up_hd4(hd5, h4_Cat_hd4)
        hd4, hd4Conv = self.fusion4d(torch.cat((h1_to_hd4, h2_to_hd4, h3_to_hd4, h4_Cat_hd4, hd5_Up_hd4), dim=1))

        h1_to_hd3 = self.h1_to_hd3(skip1)
        h2_to_hd3 = self.h2_to_hd3(skip2)
        h3_Cat_hd3 = self.h3_Cat_hd3(h3)
        hd4_Up_hd3 = self.hd4_Up_hd3(hd4, h3_Cat_hd3)
        hd5_Up_hd3 = self.hd5_Up_hd3(skip5, h3_Cat_hd3)
        hd3, hd3Conv = self.fusion3d(torch.cat((h1_to_hd3, h2_to_hd3, h3_Cat_hd3, hd4_Up_hd3, hd5_Up_hd3), dim=1))

        h1_to_hd2 = self.h1_to_hd2(skip1)
        h2_cat_hd2 = self.h2_cat_hd2(h2)
        hd3_up_hd2 = self.hd3_up_hd2(hd3, h2_cat_hd2)
        hd4_up_hd2 = self.hd4_up_hd2(skip4, h2_cat_hd2)
        hd5_up_hd2 = self.hd5_up_hd2(skip5, h2_cat_hd2)
        hd2, hd2Conv = self.fusion2d(
            torch.cat((h1_to_hd2, h2_cat_hd2, hd3_up_hd2, hd4_up_hd2, hd5_up_hd2), dim=1))

        h1_Cat_hd1 = self.h1_Cat_hd1(h1)
        hd2_UT_hd1 = self.hd2_up_hd1(hd2, skip1)
        hd3_UT_hd1 = self.hd3_up_hd1(skip3, skip1)
        hd4_UT_hd1 = self.hd4_up_hd1(skip4, skip1)
        hd5_UT_hd1 = self.hd5_up_hd1(skip5, skip1)
        hd1, hd1Conv = self.fusion1d(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), dim=1))

        d5 = self.outconv5(skip5)
        d4 = self.outconv4(hd4Conv)
        d3 = self.outconv3(hd3Conv)
        d2 = self.outconv2(hd2Conv)
        d1 = self.outconv1(hd1Conv)

        d5 = _upsample(d5, d1)
        d4 = _upsample(d4, d1)
        d3 = _upsample(d3, d1)
        d2 = _upsample(d2, d1)

        if self.sup:
            return [d1, d2, d3, d4, d5]
        elif self.side:
            d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), dim=1))
            return d0
        return d1


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    var = torch.rand(3, 1, 64, 64, 64).cuda()

    model = U2net3p6(1, 1, side=True).cuda()
    macs, params = get_model_complexity_info(model, (1, 64, 64, 64), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(var)
    print('Output shape:', y[0].shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
