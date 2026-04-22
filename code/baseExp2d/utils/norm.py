# -*-coding:utf-8 -*-
"""
# Time       ：2023/4/12 18:00
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
from torch import nn
from torch.autograd import Variable


class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class LogSoftmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        x_max, _ = x.max(dim=self.dim, keepdim=True)  # 数值平移，避免溢出
        return x - x_max - torch.log(torch.sum(torch.exp(x - x_max), dim=self.dim, keepdim=True))


class Softmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim
        self.log_softmax = LogSoftmax(dim=self.dim)

    def forward(self, x):
        return torch.exp(self.log_softmax(x))


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width, z = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width, z)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width, z)
    return x


def channel_spilt(x):
    x1, x2 = x.chunk(2, dim=1)
    return x1, x2


# todo v1
# 基础实验基本块 conv3d
# 基本块使用卷积，而 1x1使用残差
class conv3d(nn.Module):
    def __init__(self, in_size, out_size, n=2, ks=3, s=1, padding=1, dilation=1, norm=nn.BatchNorm3d):
        super(conv3d, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = s
        self.padding = padding
        p = padding
        if n == 1:
            # self.block = resnetblock(in_size, out_size, n=n, ks=ks, s=s, padding=padding)
            self.block = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p, dilation=dilation, bias=False),
                                       norm(out_size),
                                       Swish())
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p, dilation=dilation, bias=False),
                                     norm(out_size),
                                     Swish())
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        if self.n == 1:
            return self.block(inputs)
        else:
            x = inputs
            for i in range(1, self.n + 1):
                conv = getattr(self, 'conv%d' % i)
                x = conv(x)

            return x


# todo v2
# resnet resnetblock
# 基本块使用残差卷积，而其他不变
class unetconv3d(nn.Module):
    def __init__(self, in_size, out_size, n=2, ks=3, s=1, padding=1, dilation=1, norm=nn.BatchNorm3d):
        super(unetconv3d, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = s
        self.padding = padding
        p = padding

        if n == 1:
            self.cb = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=ks, stride=s, padding=p, dilation=dilation, bias=False),
                norm(out_size)
            )
            self.swish = Swish()
            # 使用卷积
            # self.block = conv3d(in_size, out_size, n=n, ks=ks, s=s, padding=padding)
        else:
            self.cb = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size=ks, stride=s, padding=p, dilation=dilation, bias=False),
                norm(out_size),
                Swish(),
                nn.Conv3d(out_size, out_size, kernel_size=ks, stride=s, padding=p, dilation=dilation, bias=False),
                norm(out_size)
            )
            self.swish = Swish()

        self.res = nn.Conv3d(in_size, out_size, kernel_size=1, padding=0, bias=False)

    def forward(self, inputs):
        # if self.n == 1:
        #     return self.block(inputs)

        cb = self.cb(inputs)
        x = self.res(inputs)
        return self.swish(cb + x)


# todo v3
# resnetv2
# class unetConv3d(nn.Module):
#     def __init__(self, in_size, out_size, n=2, ks=3, s=1, padding=1, norm=nn.BatchNorm3d):
#         super(unetConv3d, self).__init__()
#         self.n = n
#         self.ks = ks
#         self.stride = s
#         self.padding = padding
#         p = padding
#         self.res = nn.Conv3d(in_size, out_size, kernel_size=1, padding=0, bias=False)
#         for i in range(1, n + 1):
#             conv = nn.Sequential(norm(in_size),
#                                  Swish(),
#                                  nn.Conv3d(in_size, out_size, ks, s, p, bias=False))
#             setattr(self, 'conv%d' % i, conv)
#             in_size = out_size
#
#     def forward(self, inputs):
#
#         x = inputs
#         inputs = self.res(inputs)
#         for i in range(1, self.n + 1):
#             conv = getattr(self, 'conv%d' % i)
#             x = conv(x)
#
#         return x + inputs


# class CResBlock(nn.Module):
#     """
#     通道残差 resnet v2
#     """
#
#     def __init__(self, inc, midc, outc, mode='cr', norm=nn.BatchNorm3d):
#         super(CResBlock, self).__init__()
#         self.mode = mode
#         if mode == 'cr':
#             self.res = nn.Sequential(
#                 norm(inc // 2),
#                 Swish(),
#                 nn.Conv3d(inc // 2, outc // 2, kernel_size=1, padding=0),
#             )
#             self.convx = nn.Sequential(
#                 norm(inc // 2),
#                 Swish(),
#                 nn.Conv3d(inc // 2, midc, kernel_size=3, padding=1),
#                 norm(midc),
#                 Swish(),
#                 nn.Conv3d(midc, outc // 2, kernel_size=3, padding=1),
#             )
#         else:
#             self.convx = nn.Sequential(
#                 norm(inc),
#                 Swish(),
#                 nn.Conv3d(inc, outc, kernel_size=3, padding=1),
#                 norm(outc),
#                 Swish(),
#                 nn.Conv3d(outc, outc, kernel_size=3, padding=1),
#             )
#
#     def forward(self, x):
#         if self.mode == 'cr':
#             cx, ux = channel_spilt(x)
#             cx = self.res(cx)
#             ux = self.convx(ux)
#             x = channel_shuffle(torch.cat((ux, cx), dim=1), 8)
#         else:
#             x = self.convx(x)
#
#         return x


class CResBlock(nn.Module):
    """
    通道残差
    """

    def __init__(self, inc, midc, outc, mode='cr', norm=nn.BatchNorm3d):
        super(CResBlock, self).__init__()
        self.mode = mode
        if mode == 'cr':
            self.res = nn.Sequential(
                nn.Conv3d(inc // 2, outc // 2, kernel_size=1, padding=0),
                norm(outc // 2),
                Swish()
            )
            self.convx = nn.Sequential(
                nn.Conv3d(inc // 2, midc, kernel_size=3, padding=1),
                norm(midc),
                Swish(),
                nn.Conv3d(midc, outc // 2, kernel_size=3, padding=1),
                norm(outc // 2),
                Swish()
            )
        else:
            self.convx = nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=3, padding=1),
                norm(outc),
                Swish(),
                nn.Conv3d(outc, outc, kernel_size=3, padding=1),
                norm(outc),
                Swish()
            )

    def forward(self, x):
        if self.mode == 'cr':
            cx, ux = channel_spilt(x)
            cx = self.res(cx)
            ux = self.convx(ux)
            x = channel_shuffle(torch.cat((ux, cx), dim=1), 8)
        else:
            x = self.convx(x)

        return x


class CdResBlock(nn.Module):
    """
    双残差基本块
    """

    def __init__(self, inc, midc, outc, mode='cr', norm=nn.BatchNorm3d):
        super(CdResBlock, self).__init__()
        self.mode = mode
        if mode == 'cr':
            self.resa = nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=1, padding=0),
                norm(outc),
                Swish()
            )
            self.res = nn.Sequential(
                nn.Conv3d(inc // 2, outc // 2, kernel_size=1, padding=0),
                norm(outc // 2),
                Swish()
            )
            self.convx = nn.Sequential(
                nn.Conv3d(inc // 2, midc, kernel_size=3, padding=1),
                norm(midc),
                Swish(),
                nn.Conv3d(midc, outc // 2, kernel_size=3, padding=1),
                norm(outc // 2),
                Swish()
            )
        else:
            self.convx = nn.Sequential(
                nn.Conv3d(inc, outc, kernel_size=3, padding=1),
                norm(outc),
                Swish(),
                nn.Conv3d(outc, outc, kernel_size=3, padding=1),
                norm(outc),
                Swish()
            )

    def forward(self, x):
        if self.mode == 'cr':
            xout = self.resa(x)
            cx, ux = channel_spilt(x)
            cx = self.res(cx)
            ux = self.convx(ux)
            x = channel_shuffle(torch.cat((ux, cx), dim=1), 8) + xout
        else:
            x = self.convx(x)

        return x


# todo v4
# 自适应通道残差卷积块unetConv3d
# class unetConv3d(nn.Module):
#     def __init__(self, in_size, out_size, n=2, ks=3, s=1, padding=1, norm=nn.BatchNorm3d):
#         super(unetConv3d, self).__init__()
#         self.n = n
#         self.ks = ks
#         self.stride = s
#         self.padding = padding
#         self.out_size = out_size
#         p = padding
#         if n == 1:
#             self.block = conv3d(in_size, out_size, n=n, ks=ks, s=s, padding=padding)
#         else:
#             self.conv3 = nn.Sequential(nn.Conv3d(in_size, out_size // 2, ks, s, p, bias=False),
#                                        norm(out_size // 2),
#                                        Swish()
#                                        )
#             self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size // 2, 1, 1, 0, bias=False),
#                                        norm(out_size // 2),
#                                        Swish()
#                                        )
#             self.res = nn.Sequential(nn.Conv3d(in_size, out_size, 1, 1, 0, bias=False),
#                                      norm(out_size),
#                                      Swish())
#
#     def forward(self, inputs):
#         x = inputs
#         if self.n == 1:
#             return self.block(inputs)
#
#         res = self.res(x)
#         conv3 = self.conv3(x)
#         conv1 = self.conv1(x)
#
#         return channel_shuffle(torch.cat((conv3, conv1), dim=1), 8) + res


if __name__ == '__main__':
    var = torch.rand(1, 32, 64, 64, 64)
    x = Variable(var).cuda()
    model = acres(32, 64, 2).cuda()
    x = model(x)
    print(x.size())
