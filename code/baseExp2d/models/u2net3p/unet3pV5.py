# -*-coding:utf-8 -*-
"""
# Time       ：2022/8/30 9:58
# Author     ：comi
# version    ：python 3.8
# Description：
# todo enconder基本块，池化，跳跃连接 可选， pcamnet解码器
"""

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from torch.nn import init
from torch.nn.functional import interpolate
from torch.utils import checkpoint

from models.models_3d.mipt.PCAMNet import soft_dilate, soft_erode
from models.u2net3p.tfblock import BasicTFBlock, Embed
from models.u2net3p.u4block import u4block
from models.u2netV.AttentionModule import AttenModule
# from models.u2netV.U2net import RSU7, RSU6, RSU5, RSU4, RSU4F
from models.u2netV.shuffleNet import channel_shuffle, channel_spilt


# from models.u2netV.U2net import RSU6, RSU5, RSU4, RSU4F
from models.u2netV.shuffleNet import RSU7, RSU6, RSU5, RSU4, RSU4F, U2baseblock


# assert False


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
            nn.ReLU(inplace=True),

            # 逐点卷积
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
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

    def __init__(self, in_channel, out_channel, stride=None, block=None, block_mode='3p'):
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
                # nn.Conv3d(in_channel, out_channel, 1, 1, padding=0, dilation=2),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=False)
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
                                     nn.ReLU(inplace=False), )
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


class CResBlock(nn.Module):
    """
    通道残差
    """

    def __init__(self, inc, midc, outc, mode='cr'):
        super(CResBlock, self).__init__()
        self.mode = mode
        if mode == 'cr':
            self.res = nn.Sequential(
                nn.Conv3d(inc // 2, outc // 2, kernel_size=1, padding=0, dilation=1),
                nn.BatchNorm3d(outc // 2),
                nn.ReLU()
            )
            self.convx = nn.Sequential(
                nn.Conv3d(inc // 2, midc, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(midc),
                nn.ReLU(),
                nn.Conv3d(midc, outc // 2, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(outc // 2),
                nn.ReLU()
            )
        elif mode == 'res':
            self.res = nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=1, padding=0, dilation=1),
            )
            self.convx = nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(outc),
                nn.ReLU(),
                nn.Conv3d(outc, outc, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(outc),
                nn.ReLU()
            )
        elif mode == 'rs':
            self.res = nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=1, padding=0, dilation=1),
                nn.BatchNorm3d(outc),
                nn.ReLU()
            )
            self.convx = nn.Sequential(
                nn.Conv3d(inc, midc, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(midc),
                nn.ReLU(),
                nn.Conv3d(midc, outc, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(outc),
                nn.ReLU()
            )
        else:
            self.convx = nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(outc),
                nn.ReLU(),
                nn.Conv3d(outc, outc, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(outc),
                nn.ReLU()
            )

    def forward(self, x):
        if self.mode == 'cr':
            cx, ux = channel_spilt(x)
            cx = self.res(cx)
            ux = self.convx(ux)

            x = channel_shuffle(torch.cat((ux, cx), dim=1), 4)
        elif self.mode == 'res':
            X = self.res(x)
            Y = self.convx(x)
            x = X + Y
        elif self.mode == 'rs':
            x = channel_shuffle(x, 4)
            X = self.res(x)
            Y = self.convx(x)
            x = X + Y
        else:
            x = self.convx(x)

        return x


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
    def __init__(self, block_mode='3p', filters=[32, 64, 128, 256, 256], down_mode='pool'):
        super(Enconder, self).__init__()

        if block_mode == 'u2':
            self.conv1 = RSU7(1, filters[0] // 2, filters[0])  # , lr=True
            # self.conv1_b = AttenModule('u', filters[0], depth=5)  # _b
            self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)

            self.conv2 = Block(filters[0], filters[1], block=RSU6, block_mode=block_mode)
            # self.conv2_b = AttenModule('u', filters[1], depth=4)  # _b
            self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

            self.conv3 = Block(filters[1], filters[2], block=RSU5, block_mode=block_mode)
            # self.conv3_b = AttenModule('u', filters[2], depth=3)  # _b
            self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

            self.conv4 = Block(filters[2], filters[3], block=RSU4, block_mode=block_mode)
            # self.conv4_b = AttenModule('u', filters[3], depth=2)  # _b
            self.maxpool4 = Downsample(stride=2, channel=filters[3], mode=down_mode)

            # v2
            self.conv5 = Block(filters[3], filters[4], block=RSU4F, block_mode=block_mode)
            # self.conv5_b = AttenModule('cbam', filters[4], depth=3)  # _b
            self.conv5_b = u4block(filters[3], filters[4], depth=3, dirate=2, split=False, mode='ch')

            # 6 stage
            # self.maxpool5 = Downsample(stride=2, channel=filters[4], mode=down_mode)
            # self.conv6 = u4block(filters[4], filters[4], depth=2, dirate=8, split=False, mode='ch')
            # self.conv6_b = Block(filters[4], filters[4], block=RSU4F, block_mode=block_mode)

            # 挤压conv5
            # self.conv5 = CBAM(filters[3], filters[4],)
            # 直接32倍下采样
            # self.conv5 = nn.Sequential(unetConv3(filters[3], filters[4] // 32, is_batchnorm=True),
            #                            unetConv3(filters[4] // 32, filters[4], is_batchnorm=True),
            #                            )
            # bottlenek,测试
            # self.conv5 = nn.Sequential(
            #     nn.Conv3d(filters[3], filters[3] // 4, kernel_size=1, stride=1, padding=0),
            #     nn.BatchNorm3d(filters[3] // 4),
            #     nn.ReLU(inplace=True),
            #     nn.Conv3d(filters[3] // 4, filters[3] // 4, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm3d(filters[3] // 4),
            #     nn.ReLU(inplace=True),
            #     nn.Conv3d(filters[3] // 4, filters[4], kernel_size=1, stride=1, padding=0),
            #     nn.BatchNorm3d(filters[4]),
            # )
        elif block_mode == 'conv':
            self.conv1 = Block(1, filters[0], block_mode=block_mode)
            self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)

            self.conv2 = Block(filters[0], filters[1], block_mode=block_mode)
            self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

            self.conv3 = Block(filters[1], filters[2], block_mode=block_mode)
            self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

            self.conv4 = Block(filters[2], filters[3], block_mode=block_mode)
            self.maxpool4 = Downsample(stride=2, channel=filters[3], mode=down_mode)

            self.conv5 = Block(filters[3], filters[4], block_mode=block_mode)
        elif block_mode == 'tf':
            self.conv1 = BasicTFBlock(1, filters[0], heads=4, attn=False, trans=False)
            if not False:
                self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)
            else:
                self.maxpool1 = Embed(in_chans=filters[0], embed_dim=filters[0], )

            self.conv2 = BasicTFBlock(filters[0], filters[1], heads=4, attn=False, trans=False)
            self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

            self.conv3 = BasicTFBlock(filters[1], filters[2], heads=8, attn=False, trans=False)
            self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

            self.conv4 = BasicTFBlock(filters[2], filters[3], heads=8, attn=False, trans=False)
            self.maxpool4 = Downsample(stride=2, channel=filters[3], mode=down_mode)

            self.conv5 = BasicTFBlock(filters[3], filters[4], heads=16, attn=False, trans=False)
            # self.conv5 = Block(filters[3], filters[4], block=RSU4F, block_mode='u2')
        else:  # block_mode == 'u4'
            if block_mode == 'cr':
                self.conv1 = CResBlock(inc=1, midc=filters[0] // 2, outc=filters[0],
                                       mode='none')  # cr:none,res=block_mode
                self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)

                self.conv2 = CResBlock(filters[0], filters[0] // 2, filters[1], mode=block_mode)
                self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

                self.conv3 = CResBlock(filters[1], filters[1] // 2, filters[2], mode=block_mode)
                self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

                self.conv4 = CResBlock(filters[2], filters[2] // 2, filters[3], mode=block_mode)
                self.maxpool4 = Downsample(stride=2, channel=filters[3], mode=down_mode)

                # self.conv5 = CResBlock(filters[3], filters[3] // 2, filters[4], mode='res')

                self.conv5 = CResBlock(filters[3], filters[3] // 2, filters[4], mode=block_mode)

                # self.conv5_b = u4block(filters[3], filters[4], depth=2, dirate=2, split=False, mode='ch')
            else:
                self.conv1 = CResBlock(inc=1, midc=filters[0] // 2, outc=filters[0],
                                       mode='none')  # cr:none,res=block_mode
                self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)

                self.conv2 = CResBlock(filters[0], filters[0] // 2, filters[1], mode=block_mode)
                self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

                self.conv3 = CResBlock(filters[1], filters[1] // 2, filters[2], mode=block_mode)
                self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

                self.conv4 = CResBlock(filters[2], filters[2] // 2, filters[3], mode=block_mode)
                self.maxpool4 = Downsample(stride=2, channel=filters[3], mode=down_mode)

                self.conv5 = CResBlock(filters[3], filters[3] // 2, filters[4], mode=block_mode)

    def forward(self, input):
        h1 = self.conv1(input)
        h2 = self.maxpool1(h1)

        h2 = self.conv2(h2)
        h3 = self.maxpool2(h2)

        h3 = self.conv3(h3)
        h4 = self.maxpool3(h3)

        h4 = self.conv4(h4)
        h5 = self.maxpool4(h4)

        # h1 = self.conv1(input)
        # h1_b = self.conv1_b(h1)
        # h2 = self.maxpool1(h1 + h1_b)
        #
        # h2 = self.conv2(h2)
        # h2_b = self.conv2_b(h2)
        # h3 = self.maxpool2(h2 + h2_b)
        #
        # h3 = self.conv3(h3)
        # h3_b = self.conv3_b(h3)
        # h4 = self.maxpool3(h3 + h3_b)
        #
        # h4 = self.conv4(h4)
        # h4_b = self.conv4_b(h4)
        # h5 = self.maxpool4(h4 + h4_b)

        # 80.61 结构
        # h5_a = self.conv5(h5)
        # h5_b = self.conv5_b(h5)
        # h5 = (self.sigmold(h5_a) * h5_b) + h5

        # h5_a = self.conv5(h5)
        # h5 = (self.sigmold(h5_a) * h5) + h5
        # h6 = self.maxpool5(h5)
        #
        # h6_a = self.conv6(h6)
        # h6 = (self.sigmold(h6_a) * h6) + h6
        #
        # return h1, h2, h3, h4, h5, h6

        # v13
        # h5 = self.conv5(h5)  # 单独conv5 80.74

        # v2 80.89 ，最终
        h5_b = self.conv5(h5)
        h5 = self.conv5_b(h5_b + h5)

        # 通道残差+ 普通卷积
        # h5 = self.conv5(h5)

        # 通道残差+ 普通卷积
        # h5_b = self.conv5_b(h5)
        # h5 = self.conv5(h5_b + h5)

        # v3 1 添加残差 80.6
        # h5 = h5_a + h5
        # h5 = (self.sigmold(h5_x) * h5) + h5 # sigmoid方法
        # v3 2 直接挤压
        # h5 = self.conv5(h5)

        return h1, h2, h3, h4, h5,


class UnetSkipDeconder(nn.Module):

    def __init__(self, sideout=False, filters=[16, 32, 64, 128, 256], deconder='conv'):
        super(UnetSkipDeconder, self).__init__()
        self.filters = filters
        self.sideout = sideout

        # assert deconder in ['conv', 'u2', 'tf', 'u4']
        if deconder == 'conv':
            self.fusion4 = Block(filters[4] + filters[3], filters[3], block_mode=deconder)
            self.fusion3 = Block(filters[3] + filters[2], filters[2], block_mode=deconder)
            self.fusion2 = Block(filters[2] + filters[1], filters[1], block_mode=deconder)
            self.fusion1 = Block(filters[1] + filters[0], filters[0], block_mode=deconder)
        elif deconder == 'u2':
            self.fusion4 = Block(filters[4] + filters[3], filters[3], block=RSU4, block_mode=deconder)
            # self.conv4_b = AttenModule('u', filters[3], depth=2)  # _b
            self.fusion3 = Block(filters[3] + filters[2], filters[2], block=RSU5, block_mode=deconder)
            # self.conv3_b = AttenModule('u', filters[2], depth=3)  # _b
            self.fusion2 = Block(filters[2] + filters[1], filters[1], block=RSU6, block_mode=deconder)
            # self.conv2_b = AttenModule('u', filters[1], depth=4)  # _b
            self.fusion1 = Block(filters[1] + filters[0], filters[0], block=RSU7, block_mode=deconder)
            # self.conv1_b = AttenModule('u', filters[0], depth=5)  # _b

        elif deconder == 'tf':
            self.fusion4 = BasicTFBlock(filters[4] + filters[3], filters[3], heads=8, attn=False, trans=False)
            self.fusion3 = BasicTFBlock(filters[3] + filters[2], filters[2], heads=8, attn=False, trans=False)
            self.fusion2 = BasicTFBlock(filters[2] + filters[1], filters[1], heads=4, attn=False, trans=False)
            self.fusion1 = BasicTFBlock(filters[1] + filters[0], filters[0], heads=4, attn=False, trans=False)

        else:
            if deconder == 'cr':
                self.fusion4 = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                         mode='none')
            else:
                self.fusion4 = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                         mode=deconder)

            self.fusion3 = CResBlock(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2], mode=deconder)
            self.fusion2 = CResBlock(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1], mode=deconder)
            self.fusion1 = CResBlock(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0], mode=deconder)

        if self.sideout:
            # self.side1 = nn.Conv3d(filters[0], filters[0] // 2, 3, dilation=1, padding=1)
            # self.side2 = nn.Conv3d(filters[1], filters[1] // 2, 3, dilation=2, padding=1 * 2)
            # self.side3 = nn.Conv3d(filters[2], filters[2] // 2, 3, dilation=2, padding=1 * 2)
            # self.side4 = nn.Conv3d(filters[3], filters[3] // 2, 3, dilation=4, padding=1 * 4)
            # self.side5 = nn.Conv3d(filters[4], filters[4] // 2, 3, dilation=8, padding=1 * 8)
            # self.outconv = nn.Conv3d(sum(filters) // 2, 1, kernel_size=3, stride=1, padding=1)

            mid = 32

            # self.side1 = nn.Conv3d(filters[0], mid, 3, padding=1)
            # self.side2 = nn.Conv3d(filters[1], mid, 3, padding=1)
            # self.side3 = nn.Conv3d(filters[2], mid, 3, padding=1)
            # self.side4 = nn.Conv3d(filters[3], mid, 3, padding=1)
            # self.side5 = nn.Conv3d(filters[4], mid, 3, padding=1)

            self.side1 = nn.Conv3d(filters[0], 1, 3, padding=1)
            self.side2 = nn.Conv3d(filters[1], 1, 3, padding=1)
            self.side3 = nn.Conv3d(filters[2], 1, 3, padding=1)
            self.side4 = nn.Conv3d(filters[3], 1, 3, padding=1)
            self.side5 = nn.Conv3d(filters[4], 1, 3, padding=1)
            self.outconv = nn.Sequential(
                nn.Conv3d(5, 1, kernel_size=3, stride=1, padding=1),
            )

            # 逐层上采样
            # self.side1 = U2baseblock(filters[0] + mid, mid, )
            # self.side2 = U2baseblock(filters[1] + mid, mid, )
            # self.side3 = U2baseblock(filters[2] + mid, mid, )
            # self.side4 = U2baseblock(filters[3] + mid, mid, )
            # self.side5 = U2baseblock(filters[4], mid, )
            # self.outconv = nn.Sequential(
            #     U2baseblock(mid, 1, ),
            # )

            # # unet3pv3 80.92
            # self.outconv = nn.Sequential(
            #     u4block(mid * 3, mid * 2, depth=2, dirate=2, split=False, mode='ch'),
            #     u4block(mid * 2, mid // 2, depth=5, dirate=2, split=False, mode='ch'),
            #     nn.Conv3d(mid // 2, 1, kernel_size=3, stride=1, padding=1)
            # )

            # 修改
            # self.outconv = nn.Sequential(
            #     nn.Conv3d(mid * 4, mid * 2, 3, padding=1 * 4, dilation=1 * 4),
            #     nn.LeakyReLU(),
            #     nn.Conv3d(mid * 2, mid, 3, padding=1 * 2, dilation=1 * 2),
            #     nn.ReLU(),
            #     nn.Conv3d(mid, 1, 3, padding=1, dilation=1),
            #     nn.ReLU(),
            # )

            # unet3pv4 添加激活函数
            # self.outconv = nn.Sequential(
            #     RSU4F(mid * 5, mid, mid),
            #     RSU4F(mid, mid, mid),
            #     nn.Conv3d(mid, 1, kernel_size=3, stride=1, padding=1),
            # )
            # self.outconv = nn.Sequential(
            #     RSU4F(mid * 5, mid, mid),
            #     nn.Conv3d(mid, 1, kernel_size=3, stride=1, padding=1),
            # )

            # self.outconv = nn.Sequential(
            #     RSU4F(mid * 5, mid, mid),
            #     u4block(mid, mid, depth=3, dirate=2, split=False, mode='ch'),
            #     nn.Conv3d(mid, 1, kernel_size=3, stride=1, padding=1)
            # )

            # RSU7
            # self.outconv = nn.Sequential(
            #     RSU4F(mid * 5, mid, mid),
            #     u4block(mid, mid, 2, 2, False, False, mode='ch', ),
            #     nn.Conv3d(mid, 1, kernel_size=3, stride=1, padding=1)
            # )

    def forward(self, features):
        # h1, h2, h3, h4, hd5, hd6 = features[0], features[1], features[2], features[3], features[4], features[5]
        # 80.61
        h1, h2, h3, h4, hd5 = features[0], features[1], features[2], features[3], features[4]
        # hd6up = _upsample(hd6, hd5)
        # hd5 = self.fusion5(torch.cat((hd6up, hd5), 1))

        # hd6up = _upsample(hd6, hd5)
        # hd5_a = self.fusion5(torch.cat((hd6up, hd5), 1))
        # hd5_b = self.fusion5_b(torch.cat((hd6up, hd5), 1))
        # hd5 = (self.sigmold(hd5_a) * hd5_b) + hd6up

        # hd6up = _upsample(hd6, hd5)
        # hd5_a = self.fusion5(torch.cat((hd6up, hd5), 1))
        # hd5 = (self.sigmold(hd5_a) * hd6up) + hd6up
        #  pre
        hd5up = _upsample(hd5, h4)
        hd4 = self.fusion4(torch.cat((hd5up, h4), 1))

        hd4up = _upsample(hd4, h3)
        hd3 = self.fusion3(torch.cat((hd4up, h3), 1))

        hd3up = _upsample(hd3, h2)
        hd2 = self.fusion2(torch.cat((hd3up, h2), 1))

        hd1up = _upsample(hd2, h1)
        try:
            hd1 = self.fusion1(torch.cat((hd1up, h1), 1))
        except:
            hd1 = checkpoint.checkpoint(self.fusion1, torch.cat((hd1up, h1), 1))
        # now
        # hd5up = _upsample(hd5, h4)
        # hd4 = self.fusion4(torch.cat((hd5up, h4), 1))
        # hd4_b = self.conv4_b(hd4)
        #
        # hd4up = _upsample(hd4 + hd4_b, h3)
        # hd3 = self.fusion3(torch.cat((hd4up, h3), 1))
        # hd3_b = self.conv3_b(hd3)
        #
        # hd3up = _upsample(hd3 + hd3_b, h2)
        # hd2 = self.fusion2(torch.cat((hd3up, h2), 1))
        # hd2_b = self.conv2_b(hd2)
        #
        # hd1up = _upsample(hd2 + hd2_b, h1)
        # try:
        #     hd1 = self.fusion1(torch.cat((hd1up, h1), 1))
        #     hd1 = self.conv1_b(hd1)
        # except:
        #     hd1 = checkpoint.checkpoint(self.fusion1, torch.cat((hd1up, h1), 1))
        #     hd1 = self.conv1_b(hd1)

        if self.sideout:
            d1 = self.side1(hd1)

            d2 = self.side2(hd2)
            d2 = _upsample(d2, d1)

            d3 = self.side3(hd3)
            d3 = _upsample(d3, d1)

            d4 = self.side4(hd4)
            d4 = _upsample(d4, d1)

            d5 = self.side5(hd5)
            d5 = _upsample(d5, d1)

            # d6 = self.side6(hd6)
            # d6 = _upsample(d6, d1)
            # d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
            # d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))
            #
            # d0 = self.outconv(channel_shuffle(torch.cat((d1, d2, d3, d4, d5), 1), 4))
            all = torch.cat((d1, d2, d3, d4, d5), 1)
            d0 = self.outconv(all)  # side mid = 1
            # d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1), 4)
            return d0

            # 逐层上采样
            # d5 = self.side5(hd5)
            # d5up = _upsample(d5, h4)
            #
            # d4 = self.side4(torch.cat((hd4, d5up), 1))
            # d4up = _upsample(d4, h3)
            #
            # d3 = self.side3(torch.cat((hd3, d4up), 1))
            # d3up = _upsample(d3, h2)
            #
            # d2 = self.side2(torch.cat((hd2, d3up), 1))
            # d2up = _upsample(d2, h1)
            #
            # d1 = self.side1(torch.cat((hd1, d2up), 1))
            #
            # d0 = self.outconv(d1)
            # return d0
        else:
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

        try:
            h1_Cat_hd1 = self.h1_Cat_hd1(h1)
            hd2_UT_hd1 = self.hd2_UT_hd1(hd2, h1_Cat_hd1)
            hd3_UT_hd1 = self.hd3_UT_hd1(hd3, h1_Cat_hd1)
            hd4_UT_hd1 = self.hd4_UT_hd1(hd4, h1_Cat_hd1)
            hd5_UT_hd1 = self.hd5_UT_hd1(hd5, h1_Cat_hd1)
        except:
            h1_Cat_hd1 = checkpoint.checkpoint(self.h1_Cat_hd1, h1)
            hd2_UT_hd1 = checkpoint.checkpoint(self.hd2_UT_hd1, hd2, h1_Cat_hd1)
            hd3_UT_hd1 = checkpoint.checkpoint(self.hd3_UT_hd1, hd3, h1_Cat_hd1)
            hd4_UT_hd1 = checkpoint.checkpoint(self.hd4_UT_hd1, hd4, h1_Cat_hd1)
            hd5_UT_hd1 = checkpoint.checkpoint(self.hd5_UT_hd1, hd5, h1_Cat_hd1)
        hd1 = self.fusion1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))

        if self.sideout:
            return hd1, hd2, hd3, hd4, hd5
        return hd1


class DenseDeconder(nn.Module):

    def __init__(self, sideout=False, filters=[16, 32, 64, 128, 256], skip_block='conv', deconder='conv',
                 hamberger=False):
        super(DenseDeconder, self).__init__()
        self.filters = filters
        self.sideout = sideout
        self.hamberger = hamberger

        self.hd3_UT_hd1 = Block(filters[2], filters[0], block_mode=skip_block)

        self.hd4_UT_hd2 = Block(filters[3], filters[1], block_mode=skip_block)
        self.hd4_UT_hd1 = Block(filters[3], filters[0], block_mode=skip_block)

        self.hd5_UT_hd3 = Block(filters[4], filters[2], block_mode=skip_block)
        self.hd5_UT_hd2 = Block(filters[4], filters[1], block_mode=skip_block)
        self.hd5_UT_hd1 = Block(filters[4], filters[0], block_mode=skip_block)

        assert deconder in ['conv', 'u2']
        if deconder == 'conv':
            self.fusion4 = Block(filters[4] + filters[3], filters[3], block_mode=deconder)
            self.fusion3 = Block(filters[3] + filters[2] + filters[2], filters[2], block_mode=deconder)

            # self.fusion2 = Block(filters[2] + filters[1] + filters[1], filters[1], block_mode=deconder)
            self.fusion2 = Block(filters[2] + (filters[1] * 3), filters[1], block_mode=deconder)
            if hamberger:
                self.fusion1 = Block(filters[1] + (filters[0] * 3), filters[0], block_mode=deconder)
            else:
                # self.fusion1 = Block(filters[1] + (filters[0]*2), filters[0], block_mode=deconder)
                self.fusion1 = Block(filters[1] + (filters[0] * 4), filters[0], block_mode=deconder)
        elif deconder == 'u2':
            self.fusion4 = Block(filters[4] + filters[3], filters[3], block=RSU4, block_mode=deconder)
            self.fusion3 = Block(filters[3] + filters[2] + filters[2], filters[2], block=RSU5,
                                 block_mode=deconder)
            # self.fusion2 = Block(filters[2] + filters[1] + filters[1], filters[1], block=RSU6, block_mode=deconder)
            self.fusion2 = Block(filters[2] + (filters[1] * 3), filters[1], block=RSU6, block_mode=deconder)
            if hamberger:
                self.fusion1 = Block(filters[1] + (filters[0] * 3), filters[0], block=RSU7, block_mode=deconder)
            else:
                # self.fusion1 = Block(filters[1] + (filters[0] * 2), filters[0], block=RSU7, block_mode=deconder)
                self.fusion1 = Block(filters[1] + (filters[0] * 4), filters[0], block=RSU7, block_mode=deconder)

    def forward(self, features):
        h1, h2, h3, h4, hd5 = features[0], features[1], features[2], features[3], features[4]

        hd5up = _upsample(hd5, h4)
        hd4 = self.fusion4(torch.cat((hd5up, h4), 1))
        # hd4 = self.fusion4(channel_shuffle(torch.cat((hd5up, h4), 1), 4))

        hd4up = _upsample(hd4, h3)
        hd5_UT_hd3 = self.hd5_UT_hd3(hd5, h3)
        hd3 = self.fusion3(torch.cat((h3, hd4up, hd5_UT_hd3), 1))
        # hd3 = self.fusion3(channel_shuffle(torch.cat((h3, hd4up, hd5_UT_hd3), 1), 4))

        hd3up = _upsample(hd3, h2)
        hd5_UT_hd2 = self.hd5_UT_hd2(hd5, h2)
        hd4_UT_hd2 = self.hd4_UT_hd2(hd4, h2)
        hd2 = self.fusion2(torch.cat((h2, hd3up, hd4_UT_hd2, hd5_UT_hd2,), 1))
        # hd2 = self.fusion2(channel_shuffle(torch.cat((h2, hd3up, hd4_UT_hd2, hd5_UT_hd2,), 1), 4))

        hd1up = _upsample(hd2, h1)
        hd5_UT_hd1 = self.hd5_UT_hd1(hd5, h1)
        hd4_UT_hd1 = self.hd4_UT_hd1(hd4, h1)
        hd3_UT_hd1 = self.hd3_UT_hd1(hd3, h1)
        if self.hamberger:
            hd1 = self.fusion1(torch.cat((hd1up, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))
            # hd1 = self.fusion1(channel_shuffle(torch.cat((hd1up, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1), 4))
        else:
            hd1 = self.fusion1(torch.cat((h1, hd1up, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))
            # hd1 = self.fusion1(channel_shuffle(torch.cat((h1, hd1up, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1), 5))

        if self.sideout:
            return hd1, hd2, hd3, hd4, hd5
        return hd1

    # class DenseDeconderUP(nn.Module):
    #     def __init__(self, sideout=False, filters=[16, 32, 64, 128, 256], skip_block='conv', deconder='conv', ):
    #         super(DenseDeconderUP, self).__init__()
    #         self.filters = filters
    #         self.sideout = sideout
    #
    #         self.hd3_UT_hd1 = Block(filters[2], filters[0], block_mode=skip_block)
    #
    #         self.hd4_UT_hd2 = Block(filters[3], filters[1], block_mode=skip_block)
    #         self.hd4_UT_hd1 = Block(filters[3], filters[0], block_mode=skip_block)
    #
    #         self.hd5_UT_hd3 = Block(filters[4], filters[2], block_mode=skip_block)
    #         self.hd5_UT_hd2 = Block(filters[4], filters[1], block_mode=skip_block)
    #         self.hd5_UT_hd1 = Block(filters[4], filters[0], block_mode=skip_block)
    #
    #         assert deconder in ['conv', 'u2']
    #         if deconder == 'conv':
    #             self.fusion4 = Block(filters[4] + filters[3], filters[3], block_mode=deconder)
    #             self.fusion3 = Block(filters[3] + filters[2] + filters[2], filters[2], block_mode=deconder)
    #             self.fusion2 = Block(filters[2] + filters[1] + filters[1], filters[1], block_mode=deconder)
    #             self.fusion1 = Block(filters[1] + filters[0] + filters[0], filters[0], block_mode=deconder)
    #         elif deconder == 'u2':
    #             self.fusion4 = Block(filters[4] + filters[3], filters[3], block=RSU4, block_mode=deconder)
    #             self.fusion3 = Block(filters[3] + filters[2] + filters[2], filters[2], block=RSU5, block_mode=deconder)
    #             self.fusion2 = Block(filters[2] + filters[1] + filters[1], filters[1], block=RSU6, block_mode=deconder)
    #             self.fusion1 = Block(filters[1] + filters[0] + filters[0], filters[0], block=RSU7, block_mode=deconder)
    #
    #     def forward(self, features):
    #         h1, h2, h3, h4, hd5 = features[0], features[1], features[2], features[3], features[4]
    #
    #         hd5up = _upsample(hd5, h4)
    #         hd4 = self.fusion4(torch.cat((hd5up, h4), 1))
    #
    #         hd4up = _upsample(hd4, h3)
    #         hd5_UT_hd3 = self.hd5_UT_hd3(hd5, h3)
    #         hd3 = self.fusion3(torch.cat((hd4up, h3, hd5_UT_hd3), 1))
    #
    #         hd3up = _upsample(hd3, h2)
    #         hd5_UT_hd2 = self.hd5_UT_hd2(hd5, h2)
    #         hd2 = self.fusion2(torch.cat((hd3up, h2, hd5_UT_hd2), 1))
    #
    #         hd1up = _upsample(hd2, h1)
    #         hd5_UT_hd1 = self.hd5_UT_hd1(hd5, h1)
    #
    #         hd1 = self.fusion1(torch.cat((hd1up, h1, hd5_UT_hd1), 1))
    #
    #         if self.sideout:
    #             return hd1, hd2, hd3, hd4, hd5
    #         return hd1

    # class Unet3pSkipDeconderWithPCAM(Unet3pSkipDeconder):
    #     def __init__(self, sideout=False, CatChannels=None, UpChannels=None, filters=[16, 32, 64, 128, 256],
    #                  skip_block='conv', deconder='conv'):
    #         super(Unet3pSkipDeconderWithPCAM, self).__init__(sideout=sideout, CatChannels=CatChannels,
    #                                                          UpChannels=UpChannels, filters=filters,
    #                                                          skip_block=skip_block, deconder=deconder)
    #         self.UpChannels = UpChannels
    #         self.CatChannels = CatChannels
    #         self.sideout = sideout
    #         self.deconder = deconder
    #
    #         self.CatChannels = filters[0]
    #         self.CatBlocks = 5
    #         self.UpChannels = self.CatChannels * self.CatBlocks
    #
    #         # side compress
    #         self.side4 = ChanCom(self.UpChannels)
    #         self.side3 = ChanCom(self.UpChannels)
    #         self.side2 = ChanCom(self.UpChannels)
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
    #         h1_PT_hd4 = self.h1_PT_hd4(h1)
    #         h2_PT_hd4 = self.h2_PT_hd4(h2)
    #         h3_PT_hd4 = self.h3_PT_hd4(h3)
    #         h4_Cat_hd4 = self.h4_Cat_hd4(h4)
    #         hd5_UT_hd4 = self.hd5_UT_hd4(hd5, h4_Cat_hd4)
    #         hd4 = self.fusion4(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))
    #
    #         hd4_side = self.side4(hd4)  # 挤压为1通道
    #         # hd2_sideout = _upsample(hd2_side, h1)
    #         hd4_mask = self.sigmoid(hd4_side)
    #         hd4_atten = self.atten4(hd4, hd4_mask)
    #         hd3_atten = _upsample(hd4_atten, h3)
    #
    #         h1_PT_hd3 = self.h1_PT_hd3(h1)
    #         h2_PT_hd3 = self.h2_PT_hd3(h2)
    #         h3_Cat_hd3 = self.h3_Cat_hd3(h3)
    #         hd4_UT_hd3 = self.hd4_UT_hd3(hd4, h3_Cat_hd3)
    #         hd5_UT_hd3 = self.hd5_UT_hd3(hd5, h3_Cat_hd3)
    #         hd3 = self.fusion3(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))
    #         hd3 += hd3_atten
    #
    #         hd3_side = self.side3(hd3)  # 挤压为1通道
    #         # hd2_sideout = _upsample(hd2_side, h1)
    #         hd3_mask = self.sigmoid(hd3_side)
    #         hd3_atten = self.atten3(hd3, hd3_mask)
    #         hd2_atten = _upsample(hd3_atten, h2)
    #
    #         h1_PT_hd2 = self.h1_PT_hd2(h1)
    #         h2_Cat_hd2 = self.h2_Cat_hd2(h2)
    #         hd3_UT_hd2 = self.hd3_UT_hd2(hd3, h2_Cat_hd2)
    #         hd4_UT_hd2 = self.hd4_UT_hd2(hd4, h2_Cat_hd2)
    #         hd5_UT_hd2 = self.hd5_UT_hd2(hd5, h2_Cat_hd2)
    #         hd2 = self.fusion2(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))
    #         hd2 += hd2_atten
    #
    #         hd2_side = self.side2(hd2)  # 挤压为1通道
    #         # hd2_sideout = _upsample(hd2_side, h1)
    #         hd2_mask = self.sigmoid(hd2_side)
    #         hd2_atten = self.atten2(hd2, hd2_mask)
    #         hd2_atten = _upsample(hd2_atten, h1)
    #
    #         h1_Cat_hd1 = self.h1_Cat_hd1(h1)
    #         hd2_UT_hd1 = self.hd2_UT_hd1(hd2, h1_Cat_hd1)
    #         hd3_UT_hd1 = self.hd3_UT_hd1(hd3, h1_Cat_hd1)
    #         hd4_UT_hd1 = self.hd4_UT_hd1(hd4, h1_Cat_hd1)
    #         hd5_UT_hd1 = self.hd5_UT_hd1(hd5, h1_Cat_hd1)
    #         hd1 = self.fusion1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))
    #
    #         hd1 += hd2_atten
    #
    #         if self.sideout:
    #             return hd1, hd2, hd3, hd4, hd5
    #         return hd1

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
    def __init__(self, frame_config, filters, sideout=False, atten=True, n_classes=1):
        super(SkipDeconderFactory, self).__init__()
        name = frame_config[1]
        self.atten = atten
        self.sideout = sideout

        # assert self.sideout == True
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
            self.deconder = UnetSkipDeconder(sideout, filters=filters, deconder=frame_config[3], )

            self.outconv = nn.Conv3d(filters[0], n_classes, 3, padding=1)
        elif name == 'cat3p':
            if not atten:
                self.deconder = DenseDeconder(sideout, filters=filters, skip_block=frame_config[2],
                                              deconder=frame_config[3], hamberger=frame_config[-1])
            elif atten:
                self.deconder = DenseDeconder(sideout, filters=filters, skip_block=frame_config[2],
                                              deconder=frame_config[3], hamberger=frame_config[-1])
            self.outconv = nn.Conv3d(filters[0], n_classes, 3, padding=1)

    def forward(self, features):
        if not self.sideout:
            hd1 = self.deconder(features)
            d1 = self.outconv(hd1)
        else:
            d1 = self.deconder(features)
        return d1


class UNet3PlusV5(nn.Module):
    """
    frame_config = ['3p',  '3p', 'conv', 'conv']  # unet3+
    编码器可选：3p, u2
    跳跃连接可选：cat，3p, cat3p
    跳跃连接卷积可选：conv，dep
    解码器基本块可选：conv，u2
    注意力可选:true,false
    """

    def __init__(self, in_channels=1, frame_config=['3p', '3p', 'conv', 'conv', False], attn=True,
                 trans=False):
        super(UNet3PlusV5, self).__init__()
        self.in_channels = in_channels

        # 编码器基本块，下采样，跳跃连接结构，跳跃连接卷积，解码器基本块
        # frame_config = ['3p', '3p', 'conv', 'conv', False]  # unet3+
        # frame_config = ['3p', '3p', 'dep', 'conv', False]  # 跳跃连接替换为dep
        # frame_config = ['u2', '3p', 'conv', 'conv', False]  # 编码器替换u2net
        # frame_config = ['u2', '3p', 'dep', 'conv', False]  # 编码器替换u2net 3p跳跃连接 dep
        # frame_config = ['3p', '3p', 'dep', 'u2', False]  # enconder conv 3P dep u2 deconder
        # frame_config = ['u2', 'cat', 'conv', 'u2', False]  # u2net
        # frame_config = ['u2', '3p', 'dep', 'u2', False]  # u2net 3P dep

        # frame_config = ['3p', 'cat', 'conv', 'conv', False]  # 三阶段pcam
        #
        filters = [64, 128, 256, 512, 512]
        filters = [32, 64, 128, 256, 256]  # 最终实现方案 通道数
        # filters = [16, 32, 64, 128, 256]  # 测试
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
        # h1, h2, h3, h4, hd5, hd6 = self.enconder(inputs)
        # d1 = self.deconder([h1, h2, h3, h4, hd5, hd6])
        h1, h2, h3, h4, hd5, = self.enconder(inputs)
        d1 = self.deconder([h1, h2, h3, h4, hd5, ])

        return d1


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    var = torch.rand(3, 1, 64, 64, 16)
    x = Variable(var).cuda()
    model = UNet3PlusV5(1, frame_config=['u2', 'cat', 'conv', 'u2', False]).cuda()
    macs, params = get_model_complexity_info(model, (1, 64, 64, 16), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(x)
    print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # #### Test Case ###
    # Output shape: torch.Size([3, 1, 64, 64, 64])
    # Computational complexity:       1949.44 GMac
    # Number of parameters:           80.87 M
