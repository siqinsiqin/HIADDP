# -*-coding:utf-8 -*-
"""
# Time       ：2023/10/28 13:47
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
from torch import nn
from torch.nn import Linear, Dropout

from utils.norm import Swish, unetconv3d, channel_shuffle, channel_spilt, Softmax, conv3d


class Attention(nn.Module):
    def __init__(self, channel, norm=nn.BatchNorm3d):
        super(Attention, self).__init__()

        self.channel = channel
        self.softmax = Softmax()  # nn.Softmax(dim=-1)  #

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.active = nn.Sequential(
            norm(channel),
            Swish()
        )

    def att(self, query, key, value):
        # 计算注意力权重
        scores = torch.matmul(query, key)  #
        scores = torch.softmax(scores, dim=-1)  # self.softmax(scores)
        # 使用注意力权重计算加权和
        weighted_sum = torch.matmul(scores, value)  #
        output = weighted_sum  #

        return self.active(output + value)


class Mlp(nn.Module):
    def __init__(self, inchannels, drop=0.1):
        super(Mlp, self).__init__()
        self.fc1 = Linear(inchannels, inchannels * 2)
        self.fc2 = Linear(inchannels * 2, inchannels)
        self.dropout = Dropout(drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# class mScale(Attention):
#     def __init__(self, in_channels=1, hwd=32, block=unetconv3d, dilation=False, norm=nn.BatchNorm3d):
#         super(mScale, self).__init__(in_channels)
#
#         if not dilation:
#             self.spAttn7 = block(in_channels, in_channels, n=3, ks=3, dilation=1, padding=1)  # 15x15
#             self.spAttn5 = block(in_channels, in_channels, n=2, ks=3, dilation=1, padding=1)  # 7x7
#             self.spAttn3 = block(in_channels, in_channels, n=1, ks=3, dilation=1, padding=1)  # 3x3
#
#             self.layer1 = block(in_channels, in_channels, n=1, ks=(1, 1, hwd), padding=(0, 0, 0))
#             self.layer2 = block(in_channels, in_channels, n=1, ks=(1, hwd, 1), padding=(0, 0, 0))
#             self.layer3 = block(in_channels, in_channels, n=1, ks=(hwd, 1, 1), padding=(0, 0, 0))
#         else:
#             self.spAttn7 = block(in_channels, in_channels, n=3, ks=3, dilation=2, padding=2)  # 15x15
#             self.spAttn5 = block(in_channels, in_channels, n=2, ks=3, dilation=2, padding=2)  # 7x7
#             self.spAttn3 = block(in_channels, in_channels, n=1, ks=3, dilation=1, padding=1)  # 3x3
#
#             self.layer1 = block(in_channels, in_channels, n=1, ks=(3, 3, hwd), padding=(1, 1, 0))
#             self.layer2 = block(in_channels, in_channels, n=1, ks=(3, hwd, 3), padding=(1, 0, 1))
#             self.layer3 = block(in_channels, in_channels, n=1, ks=(hwd, 3, 3), padding=(0, 1, 1))
#
#         self.active1 = nn.Sequential(
#             norm(in_channels),
#             Swish()
#         )
#
#         self.active2 = nn.Sequential(
#             norm(in_channels),
#             Swish()
#         )
#         self.catf = block(in_channels * 3, in_channels, n=1, ks=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def attaxis(self, query, key, value):
#         # 计算注意力权重
#         b, c, h, w, d = query.shape
#         key = key.transpose(3, 4)
#         value = value.transpose(2, 4)
#
#         sequence_length = h * w * d
#
#         Q = query.reshape(b, sequence_length, c)
#         K = key.reshape(b, sequence_length, c)
#         v = value.reshape(b, sequence_length, c)
#
#         attention_scores = torch.bmm(Q, K.permute(0, 2, 1))
#         qk = self.softmax(attention_scores)
#
#         weighted_sum = torch.matmul(qk, v).reshape(b, c, h, w, -1)
#
#         return weighted_sum
#
#     # def forward(self, x):
#     #     sp3 = self.spAttn3(x)
#     #     sp5 = self.spAttn5(x + sp3)
#     #     sp7 = self.spAttn7(x + sp3 + sp5)
#     #     spa = self.sigmoid(self.att(sp3, sp5, sp7)) * x + x
#     #     spa = self.active1(spa)
#     #
#     #     # x
#     #     zaxis = self.layer1(spa)
#     #     yaxis = self.layer2(spa)
#     #     xaxis = self.layer3(spa)
#     #     spx = self.sigmoid(self.attaxis(zaxis, yaxis, xaxis)) * x + x
#     #     spx = self.active2(spx)
#     #     return spx
#
#     def forward(self, x):
#         sp3 = self.spAttn3(x)
#         sp5 = self.spAttn5(x + sp3)
#         sp7 = self.spAttn7(x + sp3 + sp5)
#         spa = self.sigmoid(self.active1(self.att(sp3, sp5, sp7))) * x + x
#         spa = self.active1(spa) + x
#
#         # x
#         zaxis = self.layer1(spa)
#         yaxis = self.layer2(spa)
#         xaxis = self.layer3(spa)
#         spx = self.sigmoid(self.active2(self.attaxis(zaxis, yaxis, xaxis))) * x + x
#         spx = self.active2(spx) + spa
#         return spx
#
#     # def forward(self, x):  # 84.85
#     #     sp3 = self.spAttn3(x)
#     #     sp5 = self.spAttn5(x + sp3)
#     #     sp7 = self.spAttn7(x + sp3 + sp5)
#     #     # spa = self.active1(self.att(sp3, sp5, sp7)) + x
#     #     spa = self.catf(torch.cat([sp3, sp5, sp7], dim=1)) + x
#     #     # x
#     #     zaxis = self.layer1(spa)
#     #     yaxis = self.layer2(spa)
#     #     xaxis = self.layer3(spa)
#     #     # spx = self.active2(self.attaxis(zaxis, yaxis, xaxis)) + spa
#     #     spx = self.active2(zaxis + yaxis + xaxis) + spa
#     #
#     #     return spx

# class mScale(Attention):
#     def __init__(self, in_channels=1, hwd=32, block=unetconv3d, norm=nn.BatchNorm3d):
#         super(mScale, self).__init__(1)
#
#         self.spAttn7 = block(in_channels, in_channels, n=3, ks=3, padding=1)  # 7x7
#         self.spAttn5 = block(in_channels, in_channels, n=2, ks=3, padding=1)  # 5x5
#         self.spAttn3 = block(in_channels, in_channels, n=1, ks=3, padding=1)  # 3x3
#
#         self.layer1 = block(in_channels, in_channels, n=1, ks=(1, 1, hwd), padding=0)
#         self.layer2 = block(in_channels, in_channels, n=1, ks=(1, hwd, 1), padding=0)
#         self.layer3 = block(in_channels, in_channels, n=1, ks=(hwd, 1, 1), padding=0)
#
#         self.active1 = nn.Sequential(
#             norm(in_channels),
#             Swish()
#         )
#
#         self.active2 = nn.Sequential(
#             norm(in_channels),
#             Swish()
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def attaxis(self, query, key, value):
#         # 计算注意力权重
#         b, c, h, w, d = query.shape
#         key = key.transpose(3, 4)
#         value = value.transpose(2, 4)
#
#         sequence_length = h * w * d
#
#         Q = query.reshape(b, sequence_length, c)
#         K = key.reshape(b, sequence_length, c)
#         v = value.reshape(b, sequence_length, c)
#
#         attention_scores = torch.bmm(Q, K.permute(0, 2, 1))
#         qk = self.softmax(attention_scores)
#
#         weighted_sum = torch.matmul(qk, v).reshape(b, c, h, w, -1)
#
#         return weighted_sum
#
#     def forward(self, x):
#         b, c, h, w, d = x.shape
#         sp3 = self.spAttn3(x)
#         sp5 = self.spAttn5(x + sp3)
#         sp7 = self.spAttn7(x + sp3 + sp5)
#         spa = self.sigmoid(self.att(sp3, sp5, sp7)) * x + x
#         spa = self.active1(spa) + x
#
#         # x
#         zaxis = self.layer1(spa)
#         yaxis = self.layer2(spa)
#         xaxis = self.layer3(spa)
#         spx = self.sigmoid(self.attaxis(zaxis, yaxis, xaxis)) * spa + x
#         spx = self.active2(spx) + spa
#         return spx
#     # def forward(self, x):
#     #     sp3 = self.spAttn3(x)
#     #     sp5 = self.spAttn5(x + sp3)
#     #     sp7 = self.spAttn7(x + sp3 + sp5)
#     #     spa = self.sigmoid(self.att(sp3, sp5, sp7)) * x  # + x
#     #     spa = self.active1(spa) + x
#
#     # x
#     # zaxis = self.layer1(spa)
#     # yaxis = self.layer2(spa)
#     # xaxis = self.layer3(spa)
#     # spx = self.sigmoid(self.attaxis(zaxis, yaxis, xaxis)) * spa  # + x
#     # spx = self.active2(spx) + spa
#     # return spx + spa
#     # return spa


# class DenseSP(nn.Module):
#     """
#     串行
#     """
#
#     def __init__(self, in_channels, hwd=1, block=unetconv3d, norm=nn.BatchNorm3d):
#         super(DenseSP, self).__init__()
#         tmpc = 1  # in_channels // 16
#         # 替换残差
#         self.spf = block(2, tmpc, n=1, ks=3)
#         # self.spf = block(in_channels, tmpc, n=1, ks=3)
#         # self.sps = block(tmpc, in_channels, n=1, ks=3)
#         dilation = True
#         self.spx = mScale(tmpc, hwd, block=block, dilation=dilation)
#         self.spx1 = mScale(tmpc, hwd, block=block, dilation=dilation)
#         self.spb = mScale(tmpc, hwd, block=block, dilation=dilation)
#         self.spb1 = mScale(tmpc, hwd, block=block, dilation=dilation)
#         self.sph = mScale(tmpc, hwd, block=block, dilation=dilation)
#         self.sph1 = mScale(tmpc, hwd, block=block, dilation=dilation)
#
#         # self.spd = mScale(tmpc, hwd, block=block, dilation=dilation)
#         # # self.spd1 = mScale(tmpc, hwd, block=block, dilation=dilation)
#         # self.spl = mScale(tmpc, hwd, block=block, dilation=dilation)
#         # # self.spl1 = mScale(tmpc, hwd, block=block, dilation=dilation)
#         # self.spr = mScale(tmpc, hwd, block=block, dilation=dilation)
#         # # self.spr1 = mScale(tmpc, hwd, block=block, dilation=dilation)
#
#         self.active = nn.Sequential(
#             norm(in_channels),
#             Swish()
#         )
#         self.sigmold = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         fin = self.spf(torch.cat([avg_out, max_out], dim=1))  # 前
#
#         fb = fin.permute(0, 1, 2, 4, 3)  # 后
#         fh = fin.permute(0, 1, 3, 2, 4)  # 上
#         fd = fin.permute(0, 1, 3, 4, 2)  # 下
#         fl = fin.permute(0, 1, 4, 3, 2)  # 左
#         fr = fin.permute(0, 1, 4, 2, 3)  # 右
#
#         fx = self.spx(fin)
#         fb = self.spx1(fb).permute(0, 1, 2, 4, 3)
#         fh = self.sph(fh).permute(0, 1, 3, 2, 4)
#         fd = self.sph(fd).permute(0, 1, 3, 4, 2)
#         fl = self.spl(fl).permute(0, 1, 4, 3, 2)
#         fr = self.spl(fr).permute(0, 1, 4, 2, 3)
#         sp = self.active((fx + fb + fh + fd + fl + fr) * x)
#         # sp = self.active(self.spf6(torch.cat([fin, fx, fb, fh, fd, fl, fr], dim=1)) * x)  #
#
#         return self.sigmold(sp)
#
#     # def forward(self, x):
#     #     avg_out = torch.mean(x, dim=1, keepdim=True)
#     #     max_out, _ = torch.max(x, dim=1, keepdim=True)
#     #     fin = self.spf(torch.cat([avg_out, max_out], dim=1))  # 前
#     #
#     #     fb = self.spx1(self.spx(fin)) + fin
#     #     fd = self.sph1(self.sph(fb.permute(0, 1, 3, 2, 4))).permute(0, 1, 3, 2, 4) + fin
#     #     fr = self.spb1(self.spb(fd.permute(0, 1, 4, 3, 2))).permute(0, 1, 4, 3, 2)
#     #     # 并联
#     #     spd = self.spd(fin)
#     #     spd1 = self.spd1(fin.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
#     #     spl = self.spl(fin.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)
#     #     spl1 = self.spl1(fin.permute(0, 1, 3, 4, 2)).permute(0, 1, 3, 4, 2)
#     #     spr = self.spr(fin.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2)
#     #     spr1 = self.spr1(fin.permute(0, 1, 4, 2, 3)).permute(0, 1, 4, 2, 3)
#     #
#     #     sp = self.active(self.sps(fb + fd + fr + spd + spd1 + spl + spl1 + spr + spr1) * x)
#     #
#     #     return self.sigmold(sp)
#
#     # def forward(self, x):
#     #     avg_out = torch.mean(x, dim=1, keepdim=True)
#     #     max_out, _ = torch.max(x, dim=1, keepdim=True)
#     #     fin = self.spf(torch.cat([avg_out, max_out], dim=1))  # 前
#     #
#     #     # 串联
#     #     fb = self.spx(fin) + fin
#     #     fd = self.spb(fb.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4) + fb
#     #     fr = self.sph(fd.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2) + fd
#     #     # fb = self.spx1(self.spx(fin)) + fin
#     #     # fd = self.sph1(self.sph(fb.permute(0, 1, 3, 2, 4))).permute(0, 1, 3, 2, 4) + fin
#     #     # fr = self.spb1(self.spb(fd.permute(0, 1, 4, 3, 2))).permute(0, 1, 4, 3, 2)
#     #     # 并联
#     #     pd = self.spd(fin)
#     #     pl = self.spl(fin.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)
#     #     pr = self.spr(fin.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2)
#     #
#     #     sp = self.active(self.sps(fb + fd + fr + pd + pl + pr) * x)
#     #
#     #     return self.sigmold(sp)
#     # def forward(self, x):
#     #     # avg_out = torch.mean(x, dim=1, keepdim=True)
#     #     # max_out, _ = torch.max(x, dim=1, keepdim=True)
#     #     # fin = self.spf(torch.cat([avg_out, max_out], dim=1))  # 前
#     #     fin = self.spf(x)
#     #     # 串联
#     #     # fb = self.spx(fin) + fin
#     #     # fd = self.spb(fb.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4) + fb
#     #     # fr = self.sph(fd.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2) + fd
#     #     fb = self.spx1(self.spx(fin)) + fin
#     #     fd = self.sph1(self.sph(fb.permute(0, 1, 3, 2, 4))).permute(0, 1, 3, 2, 4) + fin
#     #     fr = self.spb1(self.spb(fd.permute(0, 1, 4, 3, 2))).permute(0, 1, 4, 3, 2) + fin
#     #     # 并联
#     #     # pd = self.spd(fin)
#     #     # pl = self.spl(fin.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)
#     #     # pr = self.spr(fin.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2)
#     #
#     #     sp = self.sps(fb + fd + fr) + x
#     #
#     #     return self.sigmold(sp)

# class mScale(Attention):
#     def __init__(self, in_channels=1, hwd=32, block=unetconv3d, dilation=False, norm=nn.BatchNorm3d):
#         super(mScale, self).__init__(in_channels)
#
#         # self.spAttn7 = block(in_channels, in_channels, n=3, ks=3, padding=2)  # 15x15
#         # self.spAttn5 = block(in_channels, in_channels, n=2, ks=3, padding=2)  # 7x7
#         # self.spAttn3 = block(in_channels, in_channels, n=1, ks=3, padding=1)  # 3x3
#
#         self.layer1 = block(in_channels, in_channels, n=1, ks=(3, 3, hwd), padding=(1, 1, 0))
#         self.layer2 = block(in_channels, in_channels, n=1, ks=(3, hwd, 3), padding=(1, 0, 1))
#         self.layer3 = block(in_channels, in_channels, n=1, ks=(hwd, 3, 3), padding=(0, 1, 1))
#
#         self.active2 = nn.Sequential(
#             norm(in_channels),
#             # Swish()
#         )
#         # self.catf = block(in_channels * 3, in_channels, n=1, ks=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def attaxis(self, query, key, value):
#         # 计算注意力权重
#         b, c, h, w, d = query.shape
#         key = key.transpose(3, 4)
#         value = value.transpose(2, 4)
#
#         sequence_length = h * w * d
#
#         Q = query.reshape(b, sequence_length, c)
#         K = key.reshape(b, sequence_length, c)
#         v = value.reshape(b, sequence_length, c)
#
#         attention_scores = torch.bmm(Q, K.permute(0, 2, 1))
#         qk = self.softmax(attention_scores)
#
#         weighted_sum = torch.matmul(qk, v).reshape(b, c, h, w, -1)
#
#         return weighted_sum
#
#     def forward(self, x):  # 84.85
#
#         # x
#         zaxis = self.layer1(x)
#         yaxis = self.layer2(x)
#         xaxis = self.layer3(x)
#
#         spx = self.active2(zaxis + yaxis + xaxis) + x
#
#         return spx
#
#
# class DenseSP(nn.Module):
#     """
#     串行
#     """
#
#     def __init__(self, in_channels, hwd=1, block=unetconv3d, norm=nn.BatchNorm3d):
#         super(DenseSP, self).__init__()
#         # 替换残差
#         self.spf = block(2, 1, n=1, ks=3)
#
#         self.spx = mScale(1, hwd, block=block)
#         self.spb = mScale(1, hwd, block=block)
#         self.sph = mScale(1, hwd, block=block)
#         self.spd = mScale(1, hwd, block=block)
#         self.spl = mScale(1, hwd, block=block)
#         self.spr = mScale(1, hwd, block=block)
#
#         self.active = nn.Sequential(
#             norm(in_channels),
#             # Swish()
#         )
#         # self.fusion = block(1, in_channels, n=1, ks=3, padding=1)
#         self.sigmold = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         fx = self.spf(torch.cat([avg_out, max_out], dim=1))  # 前
#         fb = fx.permute(0, 1, 2, 4, 3)  # 后
#         fh = fx.permute(0, 1, 3, 2, 4)  # 上
#         fd = fx.permute(0, 1, 3, 4, 2)  # 下
#         fl = fx.permute(0, 1, 4, 3, 2)  # 左
#         fr = fx.permute(0, 1, 4, 2, 3)  # 右
#
#         fx = self.spx(fx) * x
#         fb = self.spb(fb).permute(0, 1, 2, 4, 3) * x
#         fh = self.sph(fh).permute(0, 1, 3, 2, 4) * x
#         fd = self.spd(fd).permute(0, 1, 3, 4, 2) * x
#         fl = self.spl(fl).permute(0, 1, 4, 3, 2) * x
#         fr = self.spr(fr).permute(0, 1, 4, 2, 3) * x
#         # sp = self.active((fx + fb + fh + fd + fl + fr) * x + x)
#         sp = self.active(fx + fb + fh + fd + fl + fr + x)
#
#         # return self.sigmold(self.fusion(sp) + x + (sp * x))
#         return self.sigmold(sp)
#
#
# class DenseCA(Attention):
#     def __init__(self, inchannel, shufinle=8, reduction=2, block=unetconv3d, norm=nn.BatchNorm3d):
#         super(DenseCA, self).__init__(inchannel)
#         self.inchannel = inchannel
#         self.shufinle = shufinle
#
#         self.catfusion = block(inchannel * 2, inchannel, n=1, ks=1, padding=0)
#
#         self.se1 = nn.Sequential(
#             block(inchannel, inchannel // reduction, n=1, ks=1, padding=0),
#             block(inchannel // reduction, inchannel, n=1, ks=1, padding=0)
#         )
#
#         self.se2 = nn.Sequential(
#             block(inchannel, inchannel // (reduction * 2), n=1, ks=1, padding=0),
#             block(inchannel // (reduction * 2), inchannel, n=1, ks=1, padding=0)
#         )
#
#         self.se3 = nn.Sequential(
#             block(inchannel, inchannel // (reduction * 4), n=1, ks=1, padding=0),
#             block(inchannel // (reduction * 4), inchannel, n=1, ks=1, padding=0)
#         )
#
#         self.se4 = nn.Sequential(
#             block(inchannel, inchannel // (reduction * 8), n=1, ks=1, padding=0),
#             block(inchannel // (reduction * 8), inchannel, n=1, ks=1, padding=0)
#         )
#         # self.mlp = Mlp(inchannel)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avgv = self.avg_pool(x)
#         maxv = self.max_pool(x)
#         hx = self.catfusion(torch.cat((avgv, maxv), dim=1))
#
#         sx = channel_shuffle(hx, self.shufinle)
#
#         exp1 = self.se1(hx)
#         exp2 = self.se2(exp1) + exp1
#         exp3 = self.se3(exp2) + exp2 + exp1
#         exp = self.se4(exp3) + exp2 + exp1 + exp3
#
#         return self.sigmoid((sx + exp) * x)  # self.sigmoid()
#
#
# class SP(nn.Module):
#
#     def __init__(self, in_channels, hwd=32, norm=nn.BatchNorm3d):
#         super(SP, self).__init__()
#
#         self.sp = DenseSP(in_channels, hwd=hwd, block=conv3d)
#         # self.sp = mScale(in_channels, hwd, block=conv3d)
#
#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), dim=1)
#
#         xsp = self.sp(x) * x
#
#         x1a, x2a = channel_spilt(xsp)
#
#         return x1a, x2a
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         padding = (kernel_size - 1) // 2
#         self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.conv1(out)
#         out = self.sigmoid(out)
#         return out
#
#
# class CA(nn.Module):
#
#     def __init__(self, in_channels, depth=1, norm=nn.BatchNorm3d):
#         super(CA, self).__init__()
#
#         self.ca = DenseCA(in_channels, depth, block=conv3d)
#         # self.cbs1 = conv3d(in_channels // 2, in_channels // 2, n=1, ks=3, padding=1)
#         # self.cbs2 = conv3d(in_channels // 2, in_channels // 2, n=1, ks=3, padding=1)
#         # self.sp = SpatialAttention()
#         rate = 32
#         self.spatial_attention = nn.Sequential(
#             nn.Conv3d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
#             nn.BatchNorm3d(int(in_channels / rate)),
#             Swish(),
#             nn.Conv3d(int(in_channels / rate), in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(in_channels)
#         )
#
#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), dim=1)
#
#         xsp = self.ca(x) * x
#         # xsp = self.sp(xsp) * x
#         xsp = self.spatial_attention(xsp).sigmoid() * x
#         x1a, x2a = channel_spilt(xsp)
#         # x1a = self.cbs1(x1a + x1)
#         # x2a = self.cbs2(x1a + x2)
#
#         return x1a, x2a
#
#
# class CASP(nn.Module):
#
#     def __init__(self, in_channels, depth=1, hwd=32, norm=nn.BatchNorm3d):
#         super(CASP, self).__init__()
#         # self.sp = DenseSP(in_channels, hwd=hwd, block=conv3d)
#
#         rate = depth
#
#         self.sp = nn.Sequential(
#             unetconv3d(in_channels, int(in_channels / rate), n=1, ks=5, padding=2),
#             unetconv3d(int(in_channels / rate), int(in_channels / rate), n=2, ks=3, padding=1),
#             unetconv3d(int(in_channels / rate), in_channels, n=1, ks=5, padding=2),
#         )
#
#         self.ca = DenseCA(in_channels, depth, block=conv3d)
#
#     def forward(self, x1, x2):
#         hx = torch.cat((x1, x2), dim=1)
#         xb1 = self.sp(hx).sigmoid() * hx
#         xc1 = self.ca(xb1) * hx
#         xa, xb = channel_spilt(xc1)
#
#         return xa, xb

class mScale(Attention):
    def __init__(self, in_channels=1, hwd=32, block=unetconv3d, dilation=False, norm=nn.BatchNorm3d):
        super(mScale, self).__init__(in_channels)

        # self.spAttn7 = block(in_channels, in_channels, n=3, ks=3, padding=2)  # 15x15
        # self.spAttn5 = block(in_channels, in_channels, n=2, ks=3, padding=2)  # 7x7
        # self.spAttn3 = block(in_channels, in_channels, n=1, ks=3, padding=1)  # 3x3

        self.layer1 = block(in_channels, in_channels, n=1, ks=(3, 3, hwd), padding=(1, 1, 0))
        self.layer2 = block(in_channels, in_channels, n=1, ks=(3, hwd, 3), padding=(1, 0, 1))
        self.layer3 = block(in_channels, in_channels, n=1, ks=(hwd, 3, 3), padding=(0, 1, 1))

        self.active2 = nn.Sequential(
            norm(in_channels),
            # Swish()
        )
        # self.catf = block(in_channels * 3, in_channels, n=1, ks=3)
        self.sigmoid = nn.Sigmoid()

    def attaxis(self, query, key, value):
        # 计算注意力权重
        b, c, h, w, d = query.shape
        key = key.transpose(3, 4)
        value = value.transpose(2, 4)

        sequence_length = h * w * d

        Q = query.reshape(b, sequence_length, c)
        K = key.reshape(b, sequence_length, c)
        v = value.reshape(b, sequence_length, c)

        attention_scores = torch.bmm(Q, K.permute(0, 2, 1))
        qk = self.softmax(attention_scores)

        weighted_sum = torch.matmul(qk, v).reshape(b, c, h, w, -1)

        return weighted_sum

    def forward(self, x):  # 84.85

        # x
        zaxis = self.layer1(x)
        yaxis = self.layer2(x)
        xaxis = self.layer3(x)

        spx = self.active2(zaxis + yaxis + xaxis) + x

        return spx


class DenseSP(nn.Module):
    """
    串行
    """

    def __init__(self, in_channels, hwd=1, block=unetconv3d, norm=nn.BatchNorm3d):
        super(DenseSP, self).__init__()
        # 替换残差
        self.spf = block(2, 1, n=1, ks=3)

        self.spx = mScale(1, hwd, block=block)
        self.spb = mScale(1, hwd, block=block)
        self.sph = mScale(1, hwd, block=block)
        self.spd = mScale(1, hwd, block=block)
        self.spl = mScale(1, hwd, block=block)
        self.spr = mScale(1, hwd, block=block)

        self.active = nn.Sequential(
            norm(in_channels),
            # Swish()
        )
        # self.fusion = block(1, in_channels, n=1, ks=3, padding=1)
        self.sigmold = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        fx = self.spf(torch.cat([avg_out, max_out], dim=1))  # 前
        fb = fx.permute(0, 1, 2, 4, 3)  # 后
        fh = fx.permute(0, 1, 3, 2, 4)  # 上
        fd = fx.permute(0, 1, 3, 4, 2)  # 下
        fl = fx.permute(0, 1, 4, 3, 2)  # 左
        fr = fx.permute(0, 1, 4, 2, 3)  # 右

        fx = self.spx(fx) * x
        fb = self.spb(fb).permute(0, 1, 2, 4, 3) * x
        fh = self.sph(fh).permute(0, 1, 3, 2, 4) * x
        fd = self.spd(fd).permute(0, 1, 3, 4, 2) * x
        fl = self.spl(fl).permute(0, 1, 4, 3, 2) * x
        fr = self.spr(fr).permute(0, 1, 4, 2, 3) * x
        # sp = self.active((fx + fb + fh + fd + fl + fr) * x + x)
        sp = self.active(fx + fb + fh + fd + fl + fr + x)

        # return self.sigmold(self.fusion(sp) + x + (sp * x))
        return self.sigmold(sp)


class DenseCA(Attention):
    def __init__(self, inchannel, shufinle=8, reduction=2, block=unetconv3d, norm=nn.BatchNorm3d):
        super(DenseCA, self).__init__(inchannel)
        self.inchannel = inchannel
        self.shufinle = shufinle

        self.catfusion = block(inchannel * 2, inchannel, n=1, ks=1, padding=0)

        self.se1 = nn.Sequential(
            block(inchannel, inchannel // reduction, n=1, ks=1, padding=0),
            block(inchannel // reduction, inchannel, n=1, ks=1, padding=0)
        )

        self.se2 = nn.Sequential(
            block(inchannel, inchannel // (reduction * 2), n=1, ks=1, padding=0),
            block(inchannel // (reduction * 2), inchannel, n=1, ks=1, padding=0)
        )

        self.se3 = nn.Sequential(
            block(inchannel, inchannel // (reduction * 4), n=1, ks=1, padding=0),
            block(inchannel // (reduction * 4), inchannel, n=1, ks=1, padding=0)
        )

        self.se4 = nn.Sequential(
            block(inchannel, inchannel // (reduction * 8), n=1, ks=1, padding=0),
            block(inchannel // (reduction * 8), inchannel, n=1, ks=1, padding=0)
        )
        # self.mlp = Mlp(inchannel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgv = self.avg_pool(x)
        maxv = self.max_pool(x)
        hx = self.catfusion(torch.cat((avgv, maxv), dim=1))

        sx = channel_shuffle(hx, self.shufinle)

        exp1 = self.se1(hx)
        exp2 = self.se2(exp1) + exp1
        exp3 = self.se3(exp2) + exp2 + exp1
        exp = self.se4(exp3) + exp2 + exp1 + exp3

        return self.sigmoid((sx + exp) * x)  # self.sigmoid()


class SP(nn.Module):

    def __init__(self, in_channels, hwd=32, norm=nn.BatchNorm3d):
        super(SP, self).__init__()

        self.sp = DenseSP(in_channels, hwd=hwd, block=conv3d)
        # self.sp = mScale(in_channels, hwd, block=conv3d)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)

        xsp = self.sp(x) * x

        x1a, x2a = channel_spilt(xsp)

        return x1a, x2a


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out


class CA(nn.Module):

    def __init__(self, in_channels, depth=1, norm=nn.BatchNorm3d):
        super(CA, self).__init__()

        self.ca = DenseCA(in_channels, depth, block=conv3d)
        # self.cbs1 = conv3d(in_channels // 2, in_channels // 2, n=1, ks=3, padding=1)
        # self.cbs2 = conv3d(in_channels // 2, in_channels // 2, n=1, ks=3, padding=1)
        # self.sp = SpatialAttention()
        rate = 32
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm3d(int(in_channels / rate)),
            Swish(),
            nn.Conv3d(int(in_channels / rate), in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels)
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)

        xsp = self.ca(x) * x
        # xsp = self.sp(xsp) * x
        xsp = self.spatial_attention(xsp).sigmoid() * x
        x1a, x2a = channel_spilt(xsp)
        # x1a = self.cbs1(x1a + x1)
        # x2a = self.cbs2(x1a + x2)

        return x1a, x2a


class CASP(nn.Module):

    def __init__(self, in_channels, depth=1, hwd=32, norm=nn.BatchNorm3d):
        super(CASP, self).__init__()
        # self.sp = DenseSP(in_channels, hwd=hwd, block=conv3d)

        rate = depth
        # 层
        self.sp = nn.Sequential(
            nn.Conv3d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm3d(int(in_channels / rate)),
            mScale(int(in_channels / rate), hwd, block=conv3d),
            nn.Conv3d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm3d(in_channels)
        )
        self.ca = DenseCA(in_channels, depth, block=conv3d)

    def forward(self, x1, x2):
        hx = torch.cat((x1, x2), dim=1)

        xc1 = self.ca(hx) * hx
        xb1 = self.sp(xc1).sigmoid() * hx
        xa, xb = channel_spilt(xb1)

        return xa, xb