# -*-coding:utf-8 -*-
"""
# Time       ：2022/7/17 19:21
# Author     ：comi
# version    ：python 3.8
# Description：
"""
from typing import Tuple, Union

import torch
from monai.networks.blocks import UnetOutBlock
from monai.networks.nets import ViT
from ptflops import get_model_complexity_info
from torch import nn

from models.swinu2net3p.ResLayers import weights_init_kaiming
from models.u2netV.U2net import RSU7, RSU6, RSU5, RSU4, RSU4F


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


class TBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 img_size: Tuple[int, int, int],
                 feature_size: int = 16,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_heads: int = 12,
                 pos_embed: str = "perceptron",
                 norm_name: Union[Tuple, str] = "instance",
                 conv_block: bool = False,
                 res_block: bool = True,
                 dropout_rate: float = 0.0, ):
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x):
        pass


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, downstep=None, upstep=None, block=None, attn=False):
        super(Block, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.maxpool = None
        self.upsample = None
        self.attn = attn
        self.block = block
        self.n = 2

        if block is None and not self.attn:  # UNet 基本块，skip，
            for i in range(1, self.n + 1):
                conv = nn.Sequential(nn.Conv3d(in_channel, out_channel, 3, 1, 1),
                                     nn.BatchNorm3d(out_channel),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_channel = out_channel
        elif block is None and self.attn:  # 用于 skip connection
            self.blockf = SpatialAttention(self.in_channel, self.out_channel)
        else:  # base block
            self.blockf = block(self.in_channel, self.in_channel // 2, self.out_channel)

        if downstep is not None:
            self.maxpool = nn.MaxPool3d(downstep, downstep, ceil_mode=True)  # ceil_mode 向上取整
        if upstep is not None:
            self.upsample = nn.Upsample(scale_factor=upstep, mode='trilinear', align_corners=False)

        for m in self.children():
            m.apply(weights_init_kaiming)

    def forward(self, x):
        if self.maxpool is not None:
            x = self.maxpool(x)

        if self.upsample is not None:
            x = self.upsample(x)

        if self.block is None:
            for i in range(1, self.n + 1):
                conv = getattr(self, 'conv%d' % i)
                x = conv(x)
        else:
            x = self.blockf(x)

        return x


class U2net3p(nn.Module):

    def __init__(self, in_channel=1, n_classes=1, filters=None, sup=False, attn=False, encoderOnly=False, single=False):
        super(U2net3p, self).__init__()
        self.sup = sup
        self.attn = attn
        self.encoderOnly = encoderOnly
        self.single = single
        if filters is None:
            filters = [16, 32, 64, 128, 256]

        # encoder
        self.stage1 = RSU7(in_channel, filters[0] // 2, filters[0])
        # self.stage1 = unetConv3(in_channel, filters[0], is_batchnorm=True)
        self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage2 = Block(filters[0], filters[1], block=RSU6)
        self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage3 = Block(filters[1], filters[2], block=RSU5)
        self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage4 = Block(filters[2], filters[3], block=RSU4)
        self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage5 = Block(filters[3], filters[4], block=RSU4F)

        # decoder
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_to_hd4 = Block(filters[0], self.CatChannels, downstep=8, attn=self.attn, )

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_to_hd4 = Block(filters[1], self.CatChannels, downstep=4, attn=self.attn)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_to_hd4 = Block(filters[2], self.CatChannels, downstep=2, attn=self.attn)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4 = Block(filters[3], self.CatChannels, attn=self.attn)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_Up_hd4 = Block(filters[4], self.CatChannels, upstep=2, attn=self.attn)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        if self.encoderOnly:
            self.fusion4d = Block(self.UpChannels, self.UpChannels)
        else:
            self.fusion4d = Block(self.UpChannels, self.UpChannels, block=RSU4)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_to_hd3 = Block(filters[0], self.CatChannels, downstep=4, attn=self.attn)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_to_hd3 = Block(filters[1], self.CatChannels, downstep=2, attn=self.attn)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3 = Block(filters[2], self.CatChannels, attn=self.attn)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_Up_hd3 = Block(self.UpChannels, self.CatChannels, upstep=2, attn=self.attn)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_Up_hd3 = Block(filters[4], self.CatChannels, upstep=4, attn=self.attn)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        if self.encoderOnly:
            self.fusion3d = Block(self.UpChannels, self.UpChannels, )
        else:
            self.fusion3d = Block(self.UpChannels, self.UpChannels, block=RSU5, )

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_to_hd2 = Block(filters[0], self.CatChannels, downstep=2, attn=self.attn)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_cat_hd2 = Block(filters[1], self.CatChannels, attn=self.attn)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_up_hd2 = Block(self.UpChannels, self.CatChannels, upstep=2, attn=self.attn)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_up_hd2 = Block(self.UpChannels, self.CatChannels, upstep=4, attn=self.attn)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_up_hd2 = Block(filters[4], self.CatChannels, upstep=8, attn=self.attn)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        if self.encoderOnly:
            self.fusion2d = Block(self.UpChannels, self.UpChannels, )
        else:
            self.fusion2d = Block(self.UpChannels, self.UpChannels, block=RSU6, )
        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1 = Block(filters[0], self.CatChannels, block=RSU7, attn=self.attn)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_up_hd1 = Block(self.UpChannels, self.CatChannels, upstep=2, attn=self.attn)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_up_hd1 = Block(self.UpChannels, self.CatChannels, upstep=4, attn=self.attn)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_up_hd1 = Block(self.UpChannels, self.CatChannels, upstep=8, attn=self.attn)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_up_hd1 = Block(filters[4], self.CatChannels, upstep=16, attn=self.attn)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        if self.encoderOnly:
            self.fusion1d = Block(self.UpChannels, self.UpChannels, )
        else:
            self.fusion1d = Block(self.UpChannels, self.UpChannels, block=RSU7, )

        # deep sup
        self.upscore6 = nn.Upsample(scale_factor=32, mode='trilinear', align_corners=True)
        self.upscore5 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)
        self.upscore4 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.upscore3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # DeepSup
        self.outconv1 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv3d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv3d(filters[4], n_classes, 3, padding=1)
        self.outconv = nn.Conv3d(5, n_classes, 3, padding=1)

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
        h1 = self.stage1(hx)
        hx = self.pool12(h1)

        # stage 2
        h2 = self.stage2(hx)
        hx = self.pool23(h2)

        # stage 3
        h3 = self.stage3(hx)
        hx = self.pool34(h3)

        # stage 4
        h4 = self.stage4(hx)
        hx = self.pool45(h4)

        # stage 5
        hd5 = self.stage5(hx)
        # hx = self.pool56(hx5)

        ## -------------Decoder-------------
        h1_to_hd4 = self.h1_to_hd4(h1)
        h2_to_hd4 = self.h2_to_hd4(h2)
        h3_to_hd4 = self.h3_to_hd4(h3)
        h4_Cat_hd4 = self.h4_Cat_hd4(h4)
        hd5_Up_hd4 = self.hd5_Up_hd4(hd5)

        hd4 = self.fusion4d(
            torch.cat((h1_to_hd4, h2_to_hd4, h3_to_hd4, h4_Cat_hd4, hd5_Up_hd4), 1))  # hd4->40*40*UpChannels

        # print(self.h1_to_hd4.para, self.h2_to_hd4.para, self.h3_to_hd4.para, self.h4_Cat_hd4.para, self.hd5_Up_hd4.para)

        h1_to_hd3 = self.h1_to_hd3(h1)
        h2_to_hd3 = self.h2_to_hd3(h2)
        h3_Cat_hd3 = self.h3_Cat_hd3(h3)
        hd4_Up_hd3 = self.hd4_Up_hd3(hd4)
        hd5_Up_hd3 = self.hd5_Up_hd3(hd5)
        hd3 = self.fusion3d(
            torch.cat((h1_to_hd3, h2_to_hd3, h3_Cat_hd3, hd4_Up_hd3, hd5_Up_hd3), 1))  # hd3->80*80*UpChannels

        h1_to_hd2 = self.h1_to_hd2(h1)
        h2_cat_hd2 = self.h2_cat_hd2(h2)
        hd3_up_hd2 = self.hd3_up_hd2(hd3)
        hd4_up_hd2 = self.hd4_up_hd2(hd4)
        hd5_up_hd2 = self.hd5_up_hd2(hd5)
        hd2 = self.fusion2d(
            torch.cat((h1_to_hd2, h2_cat_hd2, hd3_up_hd2, hd4_up_hd2, hd5_up_hd2), 1))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1(h1)
        hd2_UT_hd1 = self.hd2_up_hd1(hd2)
        hd3_UT_hd1 = self.hd3_up_hd1(hd3)
        hd4_UT_hd1 = self.hd4_up_hd1(hd4)
        hd5_UT_hd1 = self.hd5_up_hd1(hd5)
        hd1 = self.fusion1d(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))  # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # 256

        if self.sup:
            return [d1, d2, d3, d4, d5]
        elif self.single:  # SIDE
            d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))
            return d0
        else:
            # d1 = self.outconv1(hd1)  # 256
            return d1


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    var = torch.rand(3, 1, 64, 64, 64)
    x = Variable(var)
    model = U2net3p(1, 1, filters=[64, 128, 256, 512, 512], sup=True, single=True)
    macs, params = get_model_complexity_info(model, (1, 64, 64, 64), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(x)
    print('Output shape:', y[0].shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


