# -*-coding:utf-8 -*-
"""
# Time       ：2023/6/2 21:02
# Author     ：comi
# version    ：python 3.8
# Description：
"""
# -*-coding:utf-8 -*-
"""
# Time       ：2023/3/13 9:56
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import interpolate

from models.u2net3p.u4block import u4block
from models.modal.norm2d import Swish, unetConv2d, channel_shuffle, channel_spilt, Softmax, Conv2d

"""
Woo, Sanghyun, et al. "CBAM: Convolutional Block Attention Module." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
"""


class Res_CBAM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Res_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        if downsample is None:
            # self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)
            self.downsample = None
        else:
            self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class CBAM(nn.Module):

    def __init__(self, inplanes):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * x

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out


"""
Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
"""

import torch.nn as nn


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


"""
Zhang, H., et al. "SKNet: Selective kernel networks." Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2019.
"""

import torch.nn as nn
from collections import OrderedDict


class SKConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, M=2, r=16, L=32):
        super(SKConv2d, self).__init__()
        d = max(L, out_channels // r)
        self.M = M
        self.out_channels = out_channels
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1 + i, dilation=1 + i))
        self.fc = nn.Linear(out_channels, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, out_channels))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, d, h, w = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_out = conv(x)
            conv_outs.append(conv_out)
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,d,h,w

        ### fuse
        U = torch.sum(torch.stack(conv_outs, dim=-1), dim=-1)
        U = U.permute(1, 0, 2, 3).reshape(bs, -1, d)

        ### reduction channel
        S = U.mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, self.out_channels, 1, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


class SKAttention(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=(k, 1, 1), padding=(k // 2, 0, 0), groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=3)  # softmax along the depth dimension

    def forward(self, x):
        bs, c, d, h, w = x.size()
        conv_outs = []
        # split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, dim=0)  # k,bs,channel,d,h,w

        # fuse
        U = sum(conv_outs)  # bs,c,d,h,w

        # reduction channel
        S = U.mean(-1).mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        # calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1, 1))  # bs,channel
        attention_weights = torch.stack(weights, dim=0)  # k,bs,channel,1,1,1
        attention_weights = self.softmax(attention_weights)  # k,bs,channel,1,1,1

        # fuse
        V = (attention_weights * feats).sum(0)
        return V


"""
Zhang, X., et al. "ResNeSt: Split-Attention Networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020.
"""


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        assert radix > 0
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class Splat(nn.Module):
    def __init__(self, channels, radix, cardinality, reduction_factor=4):
        super(Splat, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = channels
        inter_channels = max(channels * radix // reduction_factor, 32)
        self.fc1 = nn.Conv2d(channels // radix, inter_channels, 1, groups=cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=cardinality)
        self.rsoftmax = rSoftMax(radix, cardinality)

    def forward(self, x):
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool3d(gap, 1)
        gap = self.fc1(gap)

        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


# 自注意力
class SelfAttention3d(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SelfAttention3d, self).__init__()

        self.in_channels = in_channels

        # Query, Key, and Value convolutions
        self.query_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels // reduction,
                                    kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels // reduction,
                                  kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        # Attention convolution
        self.att_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(self.in_channels // reduction)
        self.bn2 = nn.BatchNorm2d(self.in_channels // reduction)
        self.bn3 = nn.BatchNorm2d(self.in_channels)

        # Activation function
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        batch_size, channels, height, width, depth = x.size()

        # Calculate Query, Key, and Value
        query = self.query_conv(x).view(batch_size, -1, height * width * depth).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width * depth)
        value = self.value_conv(x).view(batch_size, -1, height * width * depth)

        # Calculate attention weights
        att = torch.bmm(query, key)
        att = self.softmax(att)

        # Apply attention weights to Value
        att_value = torch.bmm(value, att.permute(0, 2, 1))
        att_value = att_value.view(batch_size, channels, height, width, depth)

        # Apply attention convolution
        att_value = self.att_conv(att_value)
        att_value = self.bn3(att_value)
        att_value = self.relu(att_value)

        # Add attention output to input
        out = att_value + x

        return out


class Attention(nn.Module):
    def __init__(self, channel, norm=nn.BatchNorm2d):
        super(Attention, self).__init__()

        self.channel = channel
        self.softmax = Softmax()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.active = nn.Sequential(
            norm(channel),
            Swish()
        )

    def att(self, query, key, value):
        # 计算注意力权重
        scores = torch.matmul(query.transpose(2, 3), key)
        scores = self.softmax(scores)
        # 使用注意力权重计算加权和
        weighted_sum = torch.matmul(scores, value.transpose(2, 3))
        output = weighted_sum.transpose(2, 3)

        return self.active(output + value)


class CASPv6(nn.Module):
    now = 1

    def __init__(self, in_channels, depth=1):
        super(CASPv6, self).__init__()

        self.sp = DenseSP(in_channels, depth, block=Conv2d)
        self.ca = DenseCA(in_channels, depth, block=unetConv2d)
        # Conv2d
        self.weight = Conv2d(in_channels, 1, n=1, ks=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.cbs1 = Conv2d(in_channels // 2, in_channels // 2, n=1, ks=3, padding=1)
        self.cbs2 = Conv2d(in_channels // 2, in_channels // 2, n=1, ks=3, padding=1)

    def hot(self, name, x):
        tmp = torch.sigmoid(x)
        b, c, h, w, d = tmp.size()

        def _upsample(src, tar):
            return interpolate(src, size=(64, 64, 64), mode='bilinear', align_corners=True)

        tmp = _upsample(tmp, None)
        tmp = tmp.cpu().detach().numpy()[0][c // 2, :, :, d // 2]
        import SimpleITK as sitk
        out = sitk.GetImageFromArray(tmp)
        sitk.WriteImage(out, name)

    def forward(self, x1, x2):
        # server = 'zljteam'
        x = torch.cat((x1, x2), dim=1)
        # self.hot(f'/{server}/jwj/baseExpV5/cat_.nrrd', x)

        w = self.sigmoid(self.weight(x))

        x = (w * self.sp(x) * x) + x

        # self.hot(f'/{server}/jwj/baseExpV5/sp_.nrrd', x)

        x = ((1 - w) * self.ca(x) * x) + x
        # self.hot(f'/{server}/jwj/baseExpV5/ca_.nrrd', x)

        x1a, x2a = channel_spilt(x)

        x1 = self.cbs1(x1a + x1)
        x2 = self.cbs2(x2a + x2)

        return x1, x2


class CASPv7(nn.Module):

    def __init__(self, in_channels, depth=1):
        super(CASPv7, self).__init__()

        self.sp = DenseSP(in_channels, depth, block=Conv2d)
        self.ca = DenseCA(in_channels, depth, block=Conv2d)
        # Conv2d
        self.weight = Conv2d(in_channels, 1, n=1, ks=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        w = self.sigmoid(self.weight(x))

        x = (w * self.sp(x) * x) + x
        x = ((1 - w) * self.ca(x) * x) + x

        return channel_spilt(x)


class CASPv8(nn.Module):

    def __init__(self, in_channels, depth=1):
        super(CASPv8, self).__init__()

        self.sp = DenseSP(in_channels, depth)
        self.ca = DenseCA(in_channels, depth)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)

        x = (self.sp(x) * x) + x
        x = (self.ca(x) * x) + x

        return channel_spilt(x)


class CASP(nn.Module):

    def __init__(self, in_channels, depth=1):
        super(CASP, self).__init__()

        self.sp = DenseSP(in_channels, depth)
        self.ca = DenseCA(in_channels, depth)
        self.bnswish = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            Swish()
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)

        x = (self.sp(x) * x) + x
        x = (self.ca(x) * x) + x

        x = self.bnswish(x)

        return channel_spilt(x)


class CA(Attention):
    def __init__(self, inchannel, shuffle=8, reduction=2, sigmoid=True, block=unetConv2d, norm=nn.BatchNorm2d):
        super(CA, self).__init__(inchannel)
        self.useSigmoid = sigmoid

        self.shuffle = shuffle

        self.catfusion = block(inchannel * 2, inchannel, n=1, ks=1, padding=0)

        self.squee1 = block(inchannel, inchannel // reduction, n=1, ks=1, padding=0)
        self.expd1 = block(inchannel // reduction, inchannel, n=1, ks=1, padding=0)

        self.squee2 = block(inchannel, inchannel // (reduction * 2), n=1, ks=1, padding=0)
        self.expd2 = block(inchannel // (reduction * 2), inchannel, n=1, ks=1, padding=0)

        self.squee3 = block(inchannel, inchannel // (reduction * 4), n=1, ks=1, padding=0)
        self.expd3 = block(inchannel // (reduction * 4), inchannel, n=1, ks=1, padding=0)

        self.norm = norm(inchannel)
        self.swish = Swish()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgv = self.avg_pool(x)
        maxv = self.max_pool(x)

        hx = self.catfusion(torch.cat((avgv, maxv), dim=1))

        sx = channel_shuffle(hx, self.shuffle)

        sq = self.squee1(hx)
        exp1 = self.expd1(sq) + hx

        sq = self.squee2(exp1)
        exp2 = self.expd2(sq) + exp1 + hx

        sq = self.squee3(exp2)
        exp = self.expd3(sq) + hx + exp2 + exp1

        if self.useSigmoid:
            return self.sigmoid(self.swish(self.norm((sx + exp + hx) * x)))
        return self.swish(self.norm((sx + exp + hx) * x))


class DenseCA(nn.Module):
    """
    串行
    """

    def __init__(self, inchannel, depth=8, block=unetConv2d):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CA(inchannel, sigmoid=False, block=block))

        #
        self.fusion = block(inchannel, inchannel, n=1, ks=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hx = x
        for layer in self.blocks:
            hx = layer(hx)

        return self.sigmoid(self.fusion(hx) + x)


class kernal(Attention):
    def __init__(self, in_channels=1, block=unetConv2d):
        super(kernal, self).__init__(1)
        # 替换残差
        self.spAttn5_1 = block(in_channels, in_channels, n=1, ks=3, padding=1)
        self.spAttn5_2 = block(in_channels, in_channels, n=1, ks=3, padding=1)

        self.spAttn3 = block(in_channels, in_channels, n=1, ks=3, padding=1)
        self.spAttn1 = block(in_channels, in_channels, n=1, ks=1, padding=0)

        self.norm = nn.BatchNorm2d(in_channels)
        self.swish = Swish()

    def forward(self, x):
        sp1 = self.spAttn1(x)
        sp3 = self.spAttn3(x)
        sp5 = self.spAttn5_2(self.spAttn5_1(x))

        return self.swish(self.norm(self.att(sp1, sp5, sp3) + x))


class DenseSP(nn.Module):
    """
    串行
    """

    def __init__(self, in_channels, depth=8, block=unetConv2d):
        super(DenseSP, self).__init__()
        # 替换残差
        self.reducePool = block(in_channels, 1, n=1, ks=3, padding=1)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(kernal(1, block=block))

        self.fusion = block(1, in_channels, ks=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hx = self.reducePool(x)

        for layer in self.blocks:
            hx = layer(hx)

        return self.sigmoid(self.fusion(hx) + x)


class CrossAttention(Attention):
    def __init__(self, channels):
        super(CrossAttention, self).__init__(channels)
        self.conv_query = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_key = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv_value = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x1, x2):
        query = self.conv_query(x1)
        key = self.conv_key(x2)
        value = self.conv_value(x2)

        return self.att(query, key, value)


class AttenModule(nn.Module):

    def __init__(self, name, inchannels, depth=2):
        super(AttenModule, self).__init__()
        self.name = name
        attens = ['se', 'sk', 'cbam', 'rest', 'att', 'u', 'csa', 'mutiscale',
                  'casp', 'sp', 'dsp', 'ca', 'dca', 'c2asp']
        assert name in attens
        self.blocks = None
        if name == attens[0]:
            self.blocks = SEAttention(inchannels).cuda()
        elif name == attens[1]:
            self.blocks = SKAttention(inchannels).cuda()
        elif name == attens[2]:
            self.blocks = CBAM(inchannels).cuda()
        elif name == attens[3]:
            self.blocks = Splat(channels=inchannels, radix=1, cardinality=8, reduction_factor=4).cuda()
        elif name == attens[4]:
            self.blocks = SelfAttention3d(inchannels).cuda()
        elif name == attens[5]:
            self.blocks = u4block(inchannels, inchannels, depth=depth, dirate=2, split=False, mode='ch').cuda()
        elif name == attens[6]:
            self.blocks = CSAtten(inchannels, depth=depth, fusion_block='dconv').cuda()
        elif name == attens[7]:
            self.blocks = MutiScaleDenseAttn(inchannels, depth=depth)
        elif name == attens[8]:
            self.blocks = CASP(inchannels)  # MutiScaleCASP(inchannels)
        elif name == attens[9]:
            self.blocks = MutiScaleCASP(inchannels)
        elif name == attens[10]:
            self.blocks = DenseSP(inchannels)
        elif name == attens[11]:
            self.blocks = CA(inchannels)
        elif name == attens[12]:
            self.blocks = DenseCA(inchannels)
        elif name == attens[13]:
            self.blocks = crossCASP(inchannels)

    def forward(self, x):
        if self.name in ['sp', 'dsp', 'ca', 'dca']:
            return self.blocks(x) * x
        return self.blocks(x) + x


if __name__ == '__main__':
    channel = 256
    size = 16
    var = torch.rand(8, channel, size, size, size)

    x = Variable(var).cuda()

    # # model = MixAttenModule('att', 'att', 256, depth=3, paralle=True).cuda()
    # model = CSAtten(256, fusion_block='cbam', depth=2).cuda()
    # model = SpatialAttention().cuda()
    # model = ChannelAttention(256).cuda()
    model = CASPv8(512).cuda()
    # model = CA(channel).cuda()
    # # model = MultiScaleDownsample(256).cuda()
    #
    # macs, params = get_model_complexity_info(model, (channel, size, size, size), as_strings=True,
    #                                          print_per_layer_stat=False, verbose=False)
    y = model(x, x)
    print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
