# -*-coding:utf-8 -*-
"""
# Time       ：2022/11/13 20:09
# Author     ：comi
# version    ：python 3.8
# Description： 改进U2net block
"""

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from torch.nn import init
from torch.nn.functional import interpolate
from torch.utils import checkpoint

from models.models_3d.mipt.PCAMNet import soft_dilate, soft_erode, ChanCom
from models.u2netV.shuffleNet import RSU7, RSU6, RSU5, RSU4, RSU4F


class DEPTHWISECONV(nn.Module):
    """
    深度可分离卷积V1
    """

    def __init__(self, in_ch, out_ch, checkpoint=False):
        super(DEPTHWISECONV, self).__init__()
        self.checkpoint = checkpoint
        # 也相当于分组为1的分组卷积
        self.dw = nn.Sequential(
            # todo 逐层卷积
            nn.Conv3d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1,
                      groups=in_ch, bias=False),
            nn.InstanceNorm3d(in_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),

            # 逐点卷积
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

    def forward(self, input):
        if self.checkpoint:
            out = checkpoint.checkpoint(self.dw, input)
        else:
            out = self.dw(input)
        return out


def _upsample(src, tar):
    return interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class Downsample(nn.Module):

    def __init__(self, stride, channel=None, mode='pool'):
        super(Downsample, self).__init__()
        assert stride is not None
        if mode == 'pool':
            self.maxpool = nn.MaxPool3d(stride, stride, ceil_mode=True)  # ceil_mode 向上取整
        else:
            assert channel is not None
            self.maxpool = nn.Conv3d(channel, channel, stride, padding=0, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


class Block(nn.Module):
    """
    设置解码器基本块
    设置跳跃连接
    设置上下采样
    """

    def __init__(self, in_channel, out_channel, stride=None, block=None, block_mode='3p', maxpool_mode='pool'):
        super(Block, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.maxpool = None
        self.skip = False

        if block_mode == 'u2':
            assert block is not None
            self.blockf = block(self.in_channel, self.in_channel // 2, self.out_channel)
        elif block_mode == '3p':
            self.blockf = unetConv3(self.in_channel, self.out_channel, True)
        elif block_mode == 'conv':  # skip,base block
            self.skip = True
            self.blockf = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, padding=1),
                nn.InstanceNorm3d(out_channel),
                nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
        elif block_mode == 'dep':  # skip
            self.skip = True
            self.blockf = DEPTHWISECONV(in_channel, out_channel)

        if stride is not None:
            self.maxpool = Downsample(stride)

        self.relu = nn.LeakyReLU(inplace=False)

        try:
            for m in self.children():
                m.apply(weights_init_kaiming)
        except:
            print('weight not exist')

    def forward(self, x, target=None):
        if self.maxpool is not None and self.skip:
            x = self.maxpool(x)
            # x = self.relu(x)

        if target is not None:  # 上采样目标大小
            x = _upsample(x, target)
            # x = self.relu(x)

        x = self.blockf(x)
        return x


class unetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv3, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.resconect = nn.Conv3d(in_size, out_size, kernel_size=1)
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.InstanceNorm3d(out_size),
                                     nn.LeakyReLU(negative_slope=0.01, inplace=False))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=False), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            m.apply(weights_init_kaiming)

    def forward(self, inputs):
        res = self.resconect(inputs)
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return torch.add(x, res)


class clus_atten(nn.Module):
    def __init__(self):
        super(clus_atten, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.k = nn.Parameter(torch.ones(1))

    def forward(self, feat_area, fore_mask):

        dilate_mask = fore_mask
        erode_mask = fore_mask
        N, C, H, W, S = feat_area.size()
        iters = 1

        # 形态学操作
        for i in range(iters):
            dilate_mask = soft_dilate(fore_mask)
        for i in range(iters):
            erode_mask = soft_erode(fore_mask)

        fore_mask = erode_mask  # N,1,H,W,S cand_mask *
        back_mask = 1 - dilate_mask  # N,1,H,W,S
        # feat_area = feat_area * cand_mask #N,C,H,W,S

        fore_feat = fore_mask.contiguous().view(N, 1, -1)  # N,1,HWS
        fore_feat = fore_feat.permute(0, 2, 1).contiguous()  # N,HWS,1
        back_feat = back_mask.contiguous().view(N, 1, -1)  # N,1,HWS
        back_feat = back_feat.permute(0, 2, 1).contiguous()  # N,HWS,1
        feat = feat_area.contiguous().view(N, C, -1)  # N,C,HWS

        fore_num = torch.sum(fore_feat, dim=1, keepdim=True) + 1e-5
        back_num = torch.sum(back_feat, dim=1, keepdim=True) + 1e-5

        # 注意力
        fore_cluster = torch.bmm(feat, fore_feat) / fore_num  # N,C,1
        back_cluster = torch.bmm(feat, back_feat) / back_num  # N,C,1
        feat_cluster = torch.cat((fore_cluster, back_cluster), dim=-1)  # N,C,2

        feat_key = feat_area  # N,C,H,W,S
        feat_key = feat_key.contiguous().view(N, C, -1)  # N,C,HWS
        feat_key = feat_key.permute(0, 2, 1).contiguous()  # N,HWS,C

        feat_cluster = feat_cluster.permute(0, 2, 1).contiguous()  # N,2,C
        feat_query = feat_cluster  # N,2,C
        feat_value = feat_cluster  # N,2,C

        feat_query = feat_query.permute(0, 2, 1).contiguous()  # N,C,2
        feat_sim = torch.bmm(feat_key, feat_query)  # N,HWS,2
        feat_sim = self.softmax(feat_sim)

        feat_atten = torch.bmm(feat_sim, feat_value)  # N,HWS,C
        feat_atten = feat_atten.permute(0, 2, 1).contiguous()  # N,C,HWS
        feat_atten = feat_atten.view(N, C, H, W, S)
        feat_area = self.k * feat_atten + feat_area

        return feat_area


class Enconder(nn.Module):
    def __init__(self, block_mode='3p', filters=[16, 32, 64, 128, 256], down_mode='pool'):
        super(Enconder, self).__init__()

        if block_mode == 'u2':
            self.conv1 = RSU7(1, filters[0] // 2, filters[0])
            self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)

            self.conv2 = Block(filters[0], filters[1], block=RSU6, block_mode=block_mode)
            self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

            self.conv3 = Block(filters[1], filters[2], block=RSU5, block_mode=block_mode)
            self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

            self.conv4 = Block(filters[2], filters[3], block=RSU4, block_mode=block_mode)
            self.maxpool4 = Downsample(stride=2, channel=filters[3], mode=down_mode)

            self.conv5 = Block(filters[3], filters[4], block=RSU4F, block_mode=block_mode)
        else:
            self.conv1 = Block(1, filters[0], block_mode=block_mode)
            self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)

            self.conv2 = Block(filters[0], filters[1], block_mode=block_mode)
            self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

            self.conv3 = Block(filters[1], filters[2], block_mode=block_mode)
            self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

            self.conv4 = Block(filters[2], filters[3], block_mode=block_mode)
            self.maxpool4 = Downsample(stride=2, channel=filters[3], mode=down_mode)

            self.conv5 = Block(filters[3], filters[4], block_mode=block_mode)

    def forward(self, input):
        h1 = self.conv1(input)
        h2 = self.maxpool1(h1)

        h2 = self.conv2(h2)
        h3 = self.maxpool2(h2)

        h3 = self.conv3(h3)
        h4 = self.maxpool3(h3)

        h4 = self.conv4(h4)
        h5 = self.maxpool4(h4)

        hd5 = self.conv5(h5)
        return h1, h2, h3, h4, hd5


class UnetSkipDeconder(nn.Module):

    def __init__(self, sideout=False, filters=[16, 32, 64, 128, 256], deconder='conv'):
        super(UnetSkipDeconder, self).__init__()
        self.filters = filters
        self.sideout = sideout
        assert deconder in ['conv', 'u2']
        if deconder == 'conv':
            self.fusion4 = Block(filters[4] + filters[3], filters[3], block_mode=deconder)
            self.fusion3 = Block(filters[3] + filters[2], filters[2], block_mode=deconder)
            self.fusion2 = Block(filters[2] + filters[1], filters[1], block_mode=deconder)
            self.fusion1 = Block(filters[1] + filters[0], filters[0], block_mode=deconder)
        elif deconder == 'u2':
            self.fusion4 = Block(filters[4] + filters[3], filters[3], block=RSU4, block_mode=deconder)
            self.fusion3 = Block(filters[3] + filters[2], filters[2], block=RSU5, block_mode=deconder)
            self.fusion2 = Block(filters[2] + filters[1], filters[1], block=RSU6, block_mode=deconder)
            self.fusion1 = Block(filters[1] + filters[0], filters[0], block=RSU7, block_mode=deconder)

    def forward(self, features):
        h1, h2, h3, h4, hd5 = features[0], features[1], features[2], features[3], features[4]

        hd5up = _upsample(hd5, h4)
        hd4 = self.fusion4(torch.cat((hd5up, h4), 1))

        hd4up = _upsample(hd4, h3)
        hd3 = self.fusion3(torch.cat((hd4up, h3), 1))

        hd3up = _upsample(hd3, h2)
        hd2 = self.fusion2(torch.cat((hd3up, h2), 1))

        hd1up = _upsample(hd2, h1)
        hd1 = self.fusion1(torch.cat((hd1up, h1), 1))

        if self.sideout:
            return hd1, hd2, hd3, hd4, hd5
        return hd1


class Unet3pSkipDeconder(nn.Module):
    def __init__(self, sideout=False, CatChannels=None, UpChannels=None, filters=[16, 32, 64, 128, 256],
                 skip_block='conv', deconder='conv'):
        super(Unet3pSkipDeconder, self).__init__()
        self.UpChannels = UpChannels
        self.CatChannels = CatChannels
        self.sideout = sideout

        '''stage 4d'''
        self.h1_PT_hd4 = Block(filters[0], self.CatChannels, stride=8, block_mode=skip_block)
        self.h2_PT_hd4 = Block(filters[1], self.CatChannels, stride=4, block_mode=skip_block)
        self.h3_PT_hd4 = Block(filters[2], self.CatChannels, stride=2, block_mode=skip_block)
        self.h4_Cat_hd4 = Block(filters[3], self.CatChannels, block_mode=skip_block)
        self.hd5_UT_hd4 = Block(filters[4], self.CatChannels, block_mode=skip_block)

        '''stage 3d'''
        self.h1_PT_hd3 = Block(filters[0], self.CatChannels, stride=4, block_mode=skip_block)
        self.h2_PT_hd3 = Block(filters[1], self.CatChannels, stride=2, block_mode=skip_block)
        self.h3_Cat_hd3 = Block(filters[2], self.CatChannels, block_mode=skip_block)
        self.hd4_UT_hd3 = Block(self.UpChannels, self.CatChannels, block_mode=skip_block)
        self.hd5_UT_hd3 = Block(filters[4], self.CatChannels, block_mode=skip_block)

        '''stage 2d '''
        self.h1_PT_hd2 = Block(filters[0], self.CatChannels, stride=2, block_mode=skip_block)
        self.h2_Cat_hd2 = Block(filters[1], self.CatChannels, block_mode=skip_block)
        self.hd3_UT_hd2 = Block(self.UpChannels, self.CatChannels, block_mode=skip_block)
        self.hd4_UT_hd2 = Block(self.UpChannels, self.CatChannels, block_mode=skip_block)
        self.hd5_UT_hd2 = Block(filters[4], self.CatChannels, block_mode=skip_block)

        '''stage 1d'''
        self.h1_Cat_hd1 = Block(filters[0], self.CatChannels, block_mode=skip_block)
        self.hd2_UT_hd1 = Block(self.UpChannels, self.CatChannels, block_mode=skip_block)
        self.hd3_UT_hd1 = Block(self.UpChannels, self.CatChannels, block_mode=skip_block)
        self.hd4_UT_hd1 = Block(self.UpChannels, self.CatChannels, block_mode=skip_block)
        self.hd5_UT_hd1 = Block(filters[4], self.CatChannels, block_mode=skip_block)

        assert deconder in ['conv', 'u2']
        if deconder == 'conv':
            self.fusion4 = Block(self.UpChannels, self.UpChannels, block_mode=deconder)
            self.fusion3 = Block(self.UpChannels, self.UpChannels, block_mode=deconder)
            self.fusion2 = Block(self.UpChannels, self.UpChannels, block_mode=deconder)
            self.fusion1 = Block(self.UpChannels, self.UpChannels, block_mode=deconder)
        elif deconder == 'u2':
            self.fusion4 = Block(self.UpChannels, self.UpChannels, block=RSU4, block_mode=deconder)
            self.fusion3 = Block(self.UpChannels, self.UpChannels, block=RSU5, block_mode=deconder)
            self.fusion2 = Block(self.UpChannels, self.UpChannels, block=RSU6, block_mode=deconder)
            self.fusion1 = Block(self.UpChannels, self.UpChannels, block=RSU7, block_mode=deconder)

    def forward(self, features):
        h1, h2, h3, h4, hd5 = features[0], features[1], features[2], features[3], features[4]

        h1_PT_hd4 = self.h1_PT_hd4(h1)
        h2_PT_hd4 = self.h2_PT_hd4(h2)
        h3_PT_hd4 = self.h3_PT_hd4(h3)
        h4_Cat_hd4 = self.h4_Cat_hd4(h4)
        hd5_UT_hd4 = self.hd5_UT_hd4(hd5, h4_Cat_hd4)
        hd4 = self.fusion4(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))

        h1_PT_hd3 = self.h1_PT_hd3(h1)
        h2_PT_hd3 = self.h2_PT_hd3(h2)
        h3_Cat_hd3 = self.h3_Cat_hd3(h3)
        hd4_UT_hd3 = self.hd4_UT_hd3(hd4, h3_Cat_hd3)
        hd5_UT_hd3 = self.hd5_UT_hd3(hd5, h3_Cat_hd3)
        hd3 = self.fusion3(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))

        h1_PT_hd2 = self.h1_PT_hd2(h1)
        h2_Cat_hd2 = self.h2_Cat_hd2(h2)
        hd3_UT_hd2 = self.hd3_UT_hd2(hd3, h2_Cat_hd2)
        hd4_UT_hd2 = self.hd4_UT_hd2(hd4, h2_Cat_hd2)
        hd5_UT_hd2 = self.hd5_UT_hd2(hd5, h2_Cat_hd2)
        hd2 = self.fusion2(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))

        h1_Cat_hd1 = checkpoint.checkpoint(self.h1_Cat_hd1, h1)
        hd2_UT_hd1 = checkpoint.checkpoint(self.hd2_UT_hd1, hd2, h1_Cat_hd1)
        hd3_UT_hd1 = checkpoint.checkpoint(self.hd3_UT_hd1, hd3, h1_Cat_hd1)
        hd4_UT_hd1 = checkpoint.checkpoint(self.hd4_UT_hd1, hd4, h1_Cat_hd1)
        hd5_UT_hd1 = checkpoint.checkpoint(self.hd5_UT_hd1, hd5, h1_Cat_hd1)
        # h1_Cat_hd1 = self.h1_Cat_hd1(h1)
        # hd2_UT_hd1 = self.hd2_UT_hd1(hd2, h1_Cat_hd1)
        # hd3_UT_hd1 = self.hd3_UT_hd1(hd3, h1_Cat_hd1)
        # hd4_UT_hd1 = self.hd4_UT_hd1(hd4, h1_Cat_hd1)
        # hd5_UT_hd1 = self.hd5_UT_hd1(hd5, h1_Cat_hd1)
        hd1 = self.fusion1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))

        if self.sideout:
            return hd1, hd2, hd3, hd4, hd5
        return hd1


class Unet3pSkipDeconderWithPCAM(Unet3pSkipDeconder):
    def __init__(self, sideout=False, CatChannels=None, UpChannels=None, filters=[16, 32, 64, 128, 256],
                 skip_block='conv', deconder='conv'):
        super(Unet3pSkipDeconderWithPCAM, self).__init__(sideout=sideout, CatChannels=CatChannels,
                                                         UpChannels=UpChannels, filters=filters,
                                                         skip_block=skip_block, deconder=deconder)
        self.UpChannels = UpChannels
        self.CatChannels = CatChannels
        self.sideout = sideout
        self.deconder = deconder

        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        # side compress
        self.side4 = ChanCom(self.UpChannels)
        self.side3 = ChanCom(self.UpChannels)
        self.side2 = ChanCom(self.UpChannels)

        self.atten4 = clus_atten()
        self.atten3 = clus_atten()
        self.atten2 = clus_atten()

        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        h1, h2, h3, h4, hd5 = features[0], features[1], features[2], features[3], features[4]

        h1_PT_hd4 = self.h1_PT_hd4(h1)
        h2_PT_hd4 = self.h2_PT_hd4(h2)
        h3_PT_hd4 = self.h3_PT_hd4(h3)
        h4_Cat_hd4 = self.h4_Cat_hd4(h4)
        hd5_UT_hd4 = self.hd5_UT_hd4(hd5, h4_Cat_hd4)
        hd4 = self.fusion4(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))

        hd4_side = self.side4(hd4)  # 挤压为1通道
        # hd2_sideout = _upsample(hd2_side, h1)
        hd4_mask = self.sigmoid(hd4_side)
        hd4_atten = self.atten4(hd4, hd4_mask)
        hd3_atten = _upsample(hd4_atten, h3)

        h1_PT_hd3 = self.h1_PT_hd3(h1)
        h2_PT_hd3 = self.h2_PT_hd3(h2)
        h3_Cat_hd3 = self.h3_Cat_hd3(h3)
        hd4_UT_hd3 = self.hd4_UT_hd3(hd4, h3_Cat_hd3)
        hd5_UT_hd3 = self.hd5_UT_hd3(hd5, h3_Cat_hd3)
        hd3 = self.fusion3(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))
        hd3 += hd3_atten

        hd3_side = self.side3(hd3)  # 挤压为1通道
        # hd2_sideout = _upsample(hd2_side, h1)
        hd3_mask = self.sigmoid(hd3_side)
        hd3_atten = self.atten3(hd3, hd3_mask)
        hd2_atten = _upsample(hd3_atten, h2)

        h1_PT_hd2 = self.h1_PT_hd2(h1)
        h2_Cat_hd2 = self.h2_Cat_hd2(h2)
        hd3_UT_hd2 = self.hd3_UT_hd2(hd3, h2_Cat_hd2)
        hd4_UT_hd2 = self.hd4_UT_hd2(hd4, h2_Cat_hd2)
        hd5_UT_hd2 = self.hd5_UT_hd2(hd5, h2_Cat_hd2)
        hd2 = self.fusion2(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))
        hd2 += hd2_atten

        hd2_side = self.side2(hd2)  # 挤压为1通道
        # hd2_sideout = _upsample(hd2_side, h1)
        hd2_mask = self.sigmoid(hd2_side)
        hd2_atten = self.atten2(hd2, hd2_mask)
        hd2_atten = _upsample(hd2_atten, h1)

        h1_Cat_hd1 = self.h1_Cat_hd1(h1)
        hd2_UT_hd1 = self.hd2_UT_hd1(hd2, h1_Cat_hd1)
        hd3_UT_hd1 = self.hd3_UT_hd1(hd3, h1_Cat_hd1)
        hd4_UT_hd1 = self.hd4_UT_hd1(hd4, h1_Cat_hd1)
        hd5_UT_hd1 = self.hd5_UT_hd1(hd5, h1_Cat_hd1)
        hd1 = self.fusion1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))

        hd1 += hd2_atten

        if self.sideout:
            return hd1, hd2, hd3, hd4, hd5
        return hd1


# class UnetSkipDeconderWithPCAM(UnetSkipDeconder):
#
#     def __init__(self, sideout=False, filters=[16, 32, 64, 128, 256], deconder='conv'):
#         super(UnetSkipDeconderWithPCAM, self).__init__(sideout=sideout, filters=filters, deconder=deconder)
#
#         # side compress
#         self.side4 = ChanCom(filters[3])
#         self.side3 = ChanCom(filters[2])
#         self.side2 = ChanCom(filters[1])
#
#         self.atten4 = clus_atten()
#         self.atten3 = clus_atten()
#         self.atten2 = clus_atten()
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, features):
#         h1, h2, h3, h4, hd5 = features[0], features[1], features[2], features[3], features[4]
#
#         hd5up = _upsample(hd5, h4)
#         hd4 = self.fusion4(torch.cat((hd5up, h4), 1))
#
#         hd4_side = self.side4(hd4)  # 挤压为1通道
#         # hd2_sideout = _upsample(hd2_side, h1)
#         hd4_mask = self.sigmoid(hd4_side)
#         hd4_atten = self.atten4(hd4, hd4_mask)
#         hd3_atten = _upsample(hd4_atten, h3)
#
#         hd4up = _upsample(hd4, h3)
#         hd3 = self.fusion3(torch.cat((hd4up, h3), 1))
#
#         hd3 += hd3_atten
#
#         hd3_side = self.side3(hd3)  # 挤压为1通道
#         # hd2_sideout = _upsample(hd2_side, h1)
#         hd3_mask = self.sigmoid(hd3_side)
#         hd3_atten = self.atten3(hd3, hd3_mask)
#         hd2_atten = _upsample(hd3_atten, h2)
#
#         hd3up = _upsample(hd3, h2)
#         hd2 = self.fusion2(torch.cat((hd3up, h2), 1))
#         hd2 += hd2_atten
#
#         hd2_side = self.side2(hd2)  # 挤压为1通道
#         # hd2_sideout = _upsample(hd2_side, h1)
#         hd2_mask = self.sigmoid(hd2_side)
#         hd1_atten = self.atten2(hd2, hd2_mask)
#         hd1_atten = _upsample(hd1_atten, h1)
#
#         hd1up = _upsample(hd2, h1)
#         hd1 = self.fusion1(torch.cat((hd1up, h1), 1))
#
#         hd1 += hd1_atten
#         if self.sideout:
#             return hd1, hd2, hd3, hd4, hd5
#         return hd1


class SkipDeconderFactory(nn.Module):
    def __init__(self, frame_config, filters, sideout=False, atten=False, n_classes=1):
        super(SkipDeconderFactory, self).__init__()
        name = frame_config[1]
        self.atten = atten
        self.sideout = sideout

        if name == '3p':
            CatChannels = filters[0]
            CatBlocks = 5
            UpChannels = CatChannels * CatBlocks
            if atten:
                self.deconder = Unet3pSkipDeconderWithPCAM(sideout, CatChannels, UpChannels, filters=filters,
                                                           skip_block=frame_config[2], deconder=frame_config[3])
            elif not atten:
                self.deconder = Unet3pSkipDeconder(sideout, CatChannels, UpChannels, filters=filters,
                                                   skip_block=frame_config[2], deconder=frame_config[3])
            self.outconv = nn.Conv3d(UpChannels, n_classes, 3, padding=1)
        elif name == 'cat':
            if not atten:
                self.deconder = UnetSkipDeconder(sideout, filters=filters, deconder=frame_config[3])
            elif atten:
                assert False
            self.outconv = nn.Conv3d(filters[0], n_classes, 3, padding=1)

    def forward(self, features):
        if not self.sideout:
            hd1 = self.deconder(features)
            d1 = self.outconv(hd1)
        else:
            pass
        return d1


class UNet3PlusV7(nn.Module):
    """
    frame_config = ['3p',  '3p', 'conv', 'conv']  # unet3+
    编码器可选：3p, u2
    跳跃连接可选：cat，3p
    跳跃连接卷积可选：conv，dep
    解码器基本块可选：conv，u2
    注意力可选:true,false
    """

    def __init__(self, in_channels=1, frame_config=['3p', '3p', 'conv', 'conv', False], atten=False):
        super(UNet3PlusV7, self).__init__()
        self.in_channels = in_channels
        self.atten = atten

        # 编码器基本块，下采样，跳跃连接结构，跳跃连接卷积，解码器基本块
        # frame_config = ['3p', '3p', 'conv', 'conv', False]  # unet3+
        # frame_config = ['u2', '3p', 'conv', 'conv', False]  # 编码器替换u2net
        # frame_config = ['u2', '3p', 'dep', 'conv', False]  # 编码器替换u2net
        # frame_config = ['u2', '3p', 'dep', 'u2', False]  #
        # frame_config = ['3p', '3p', 'dep', 'conv', False]  # 跳跃连接替换为dep
        # frame_config = ['3p', 'cat', 'conv', 'conv', False]  # 三阶段pcam
        #
        # filters = [64, 128, 256, 512, 512]  # [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        filters = [16, 32, 64, 128, 256]
        # assert False
        # -------------Encoder--------------
        self.enconder = Enconder(block_mode=frame_config[0], filters=filters)

        # -------------Decoder--------------
        self.deconder = SkipDeconderFactory(frame_config, filters, atten=frame_config[-1])

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm3d):
                m.apply(weights_init_kaiming)

    def forward(self, inputs):
        h1, h2, h3, h4, hd5 = self.enconder(inputs)
        d1 = self.deconder([h1, h2, h3, h4, hd5])

        return d1


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    var = torch.rand(3, 1, 64, 64, 64)
    x = Variable(var).cuda()
    model = UNet3PlusV6(1).cuda()
    macs, params = get_model_complexity_info(model, (1, 64, 64, 64), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(x)
    print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # #### Test Case ###
    # Output shape: torch.Size([3, 1, 64, 64, 64])
    # Computational complexity:       1949.44 GMac
    # Number of parameters:           80.87 M
