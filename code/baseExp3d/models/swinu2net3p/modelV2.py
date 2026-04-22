# -*-coding:utf-8 -*-
"""
# Time       ：2022/8/16 9:53
# Author     ：comi
# version    ：python 3.8
# Description：
# 添加上采样
"""

import numpy as np
import torch
from einops import rearrange
from timm.models.layers import to_3tuple
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.nn.functional import interpolate
from torch.nn.init import trunc_normal_
from torch.utils import checkpoint

from models.swinu2net.transformer import SwinTransformerBlock, window_partition, PatchEmbed
# from models.u2netV.U2netV3 import DEPTHWISECONV, DRSU5, DRSU4, DRSU4F, DRSU6, DRSU7
import torch.nn.functional as F

# from models.u2net3p.u2net3pV4 import U2net3p5dc


def _upsample(src, tar):
    return interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)


def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def _init_weights_trans(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class Block(nn.Module):
    """
    this block implment:
    1. swin transformer
    2. swin transformer + other conv block
    3. conv max pool down sample
    """

    def __init__(self, in_channel, mid_dim, out_channel, refine_block=None, side=False, num_heads=2, mlp_ratio=4.,
                 window_size=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., upsample=False,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, depth=2):

        super(Block, self).__init__()
        self.window_size = to_3tuple(window_size)
        self.shift_size = to_3tuple(window_size // 2)
        self.use_checkpoint = use_checkpoint
        self.side = side  # side 特征融合
        self.mid_dim = mid_dim
        if mid_dim is None:
            self.mid_dim = max(in_channel // 2, out_channel // 2)
        else:
            self.mid_dim = mid_dim

        assert 0 <= min(self.shift_size) < min(
            self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}" \
            .format(self.shift_size, self.window_size)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=in_channel,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=to_3tuple(0),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[0 + (i * 2)] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer))
            self.blocks.append(
                SwinTransformerBlock(
                    dim=in_channel,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[1 + (i * 2)] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer)
            )

        if refine_block is not None:
            self.refine_block = refine_block(in_ch=in_channel, mid_ch=self.mid_dim, out_ch=out_channel, side=side,
                                             upsample=upsample)
        else:
            self.refine_block = connectBlock(in_channel, out_channel)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def toConv(self, x, H, W, T, out_dim):
        return x.view(-1, H, W, T, out_dim).permute(0, 4, 1, 2, 3).contiguous()

    def createmask(self, x, H, W, T):
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        H, W, T = x.size(2), x.size(3), x.size(4)
        x = rearrange(x, 'b c h w t-> b (h w t) c', h=H, w=W, t=T)

        attn_mask = self.createmask(x, H, W, T)

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        x = rearrange(x, 'b (w h t) c -> b c w h t', h=H, w=W, t=T)

        if self.refine_block is not None:
            x = self.refine_block(x)  # block 优化

        return x


class connectBlock(nn.Module):
    """
    用于上下采样和跳跃连接
    """

    def __init__(self, in_channel, out_channel, downstep=None, block=None):
        super(connectBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.maxpool = None
        self.block = block

        if block is None:  # UNet 基本块，skip，
            self.blockf = nn.Sequential(DEPTHWISECONV(in_channel, out_channel),
                                        nn.BatchNorm3d(out_channel),
                                        nn.ReLU(inplace=True), )
        else:  # base block
            self.blockf = block(self.in_channel, self.in_channel // 2, self.out_channel, side=False)

        if downstep is not None:
            self.maxpool = nn.MaxPool3d(downstep, downstep, ceil_mode=True)  # ceil_mode 向上取整

        for m in self.children():
            m.apply(weights_init_kaiming)

    def forward(self, x, target=None):
        if self.maxpool is not None:
            x = self.maxpool(x)

        if target is not None:  # 上采样目标大小
            x = _upsample(x, target)

        x = self.blockf(x)

        return x


class swinu2net3plus(nn.Module):
    """
    include ： 3+、u block 、 swin
    """

    def __init__(self, in_channel, out_channel, filters=None, side=False, img_size=64, window_size=4, patch_size=2,
                 ape=True, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_rate=0.2, upsample=False,
                 use_checkpoint=False):
        super(swinu2net3plus, self).__init__()
        self.side = side
        self.ape = ape
        self.upsample = upsample
        self.use_checkpoint = use_checkpoint
        self.filters = [32, 64, 128, 256, 256]  # [32, 64, 128, 256, 320]
        self.CatChannels = self.filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.en1 = DRSU7(in_channel, self.filters[0] // 2, self.filters[0], side=side, upsample=upsample)
        self.pool12 = nn.MaxPool3d(2, 2)

        embed_dim = self.filters[0] * 2
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=self.filters[0], embed_dim=embed_dim, norm_layer=nn.LayerNorm)

        if self.ape:
            img_size = to_3tuple(img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1],
                                  img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop)

        self.en2 = DRSU6(self.filters[1], self.filters[1] // self.filters[0], self.filters[1], side=side,
                         upsample=upsample)
        self.pool23 = nn.MaxPool3d(2, 2)

        depths = [2, 4, 8, 4, 2]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.en3 = Block(in_channel=self.filters[1], mid_dim=self.filters[1] // 2, out_channel=self.filters[2],
                         refine_block=DRSU5, num_heads=4, side=side, window_size=window_size, qkv_bias=qkv_bias,
                         qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, upsample=upsample,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         use_checkpoint=use_checkpoint, depth=1)
        self.pool34 = nn.MaxPool3d(2, 2)

        self.en4 = Block(in_channel=self.filters[2], mid_dim=self.filters[2] // 2, out_channel=self.filters[3],
                         refine_block=DRSU4, num_heads=8, side=side, window_size=window_size, qkv_bias=qkv_bias,
                         qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, upsample=upsample,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         use_checkpoint=use_checkpoint, depth=2)
        self.pool45 = nn.MaxPool3d(2, 2)

        self.bottle = Block(in_channel=self.filters[3], mid_dim=self.filters[3] // 2, out_channel=self.filters[4],
                            refine_block=DRSU4F, side=side, num_heads=16, window_size=window_size, qkv_bias=qkv_bias,
                            qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, upsample=upsample,
                            drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                            use_checkpoint=use_checkpoint, depth=4)

        """to 4 stage"""
        self.h1_to_hd4 = connectBlock(self.filters[0], self.CatChannels, downstep=8)
        self.h2_to_hd4 = connectBlock(self.filters[1], self.CatChannels, downstep=4)
        self.h3_to_hd4 = connectBlock(self.filters[2], self.CatChannels, downstep=2)
        self.h4_Cat_hd4 = connectBlock(self.filters[3], self.CatChannels)
        self.hd5_Up_hd4 = connectBlock(self.filters[4], self.CatChannels)
        self.fusion4d = Block(in_channel=self.UpChannels, mid_dim=self.UpChannels // 2, out_channel=self.UpChannels,
                              refine_block=DRSU4, side=side, num_heads=8, window_size=window_size, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, upsample=upsample,
                              drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                              use_checkpoint=use_checkpoint, depth=2)

        """to 3 stage"""
        self.h1_to_hd3 = connectBlock(self.filters[0], self.CatChannels, downstep=4, )
        self.h2_to_hd3 = connectBlock(self.filters[1], self.CatChannels, downstep=2, )
        self.h3_Cat_hd3 = connectBlock(self.filters[2], self.CatChannels)
        self.hd4_Up_hd3 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd5_Up_hd3 = connectBlock(self.filters[4], self.CatChannels)
        self.fusion3d = Block(in_channel=self.UpChannels, mid_dim=self.UpChannels // 2, out_channel=self.UpChannels,
                              refine_block=DRSU5, num_heads=4, side=side, window_size=window_size, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, upsample=upsample,
                              drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                              use_checkpoint=use_checkpoint, depth=1)

        """to 2 stage"""
        self.h1_to_hd2 = connectBlock(self.filters[0], self.CatChannels, downstep=2)
        self.h2_cat_hd2 = connectBlock(self.filters[1], self.CatChannels)
        self.hd3_up_hd2 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd4_up_hd2 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd5_up_hd2 = connectBlock(self.filters[4], self.CatChannels)
        self.fusion2d = DRSU6(in_ch=self.UpChannels, mid_ch=self.UpChannels // 2, out_ch=self.UpChannels,
                              upsample=upsample, side=side)

        '''stage 1d'''
        self.h1_Cat_hd1 = connectBlock(self.filters[0], self.CatChannels)
        self.hd2_up_hd1 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd3_up_hd1 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd4_up_hd1 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd5_up_hd1 = connectBlock(self.filters[4], self.CatChannels)
        self.fusion1d = DRSU7(in_ch=self.UpChannels, mid_ch=self.UpChannels // 2, out_ch=self.UpChannels,
                              upsample=upsample, side=side)

        self.outconv1 = DEPTHWISECONV(self.UpChannels, out_channel)
        self.outconv2 = DEPTHWISECONV(self.UpChannels, out_channel)
        self.outconv3 = DEPTHWISECONV(self.UpChannels, out_channel)
        self.outconv4 = DEPTHWISECONV(self.UpChannels, out_channel)
        self.outconv5 = DEPTHWISECONV(self.filters[4], out_channel)
        self.outconv = DEPTHWISECONV(self.CatBlocks, out_channel)

        # self.apply(_init_weights_trans)
        self.apply(weights_init_kaiming)

    def forward(self, x):
        hx = x
        en1 = self.en1(x)
        # en1pool = self.pool12(en1)

        patch = self.patch_embed(en1)
        H, W, T = patch.size(2), patch.size(3), patch.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(H, W, T), mode='trilinear',
                                               align_corners=True)
            patch = (patch + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        else:
            patch = patch.flatten(2).transpose(1, 2)  # [b l c]
        patch = self.pos_drop(patch)
        patch = rearrange(patch, 'b (h w t) c -> b c h w t', h=H, w=W, t=T)

        en2 = self.en2(patch)
        en2pool = self.pool23(en2)  # c

        en3 = self.en3(en2pool)  # t
        en3pool = self.pool34(en3)

        en4 = self.en4(en3pool)
        en4pool = self.pool45(en4)

        bottle = self.bottle(en4pool)

        h1_to_hd4 = checkpoint.checkpoint(self.h1_to_hd4, en1)  # 将其他通道融合为统一通道
        h2_to_hd4 = checkpoint.checkpoint(self.h2_to_hd4, en2)
        h3_to_hd4 = checkpoint.checkpoint(self.h3_to_hd4, en3)
        h4_Cat_hd4 = checkpoint.checkpoint(self.h4_Cat_hd4, en4)
        hd5_Up_hd4 = checkpoint.checkpoint(self.hd5_Up_hd4, bottle, h4_Cat_hd4)  # h4_Cat_hd4上采样目标大小
        hd4 = self.fusion4d(torch.cat((h1_to_hd4, h2_to_hd4, h3_to_hd4, h4_Cat_hd4, hd5_Up_hd4), dim=1))

        h1_to_hd3 = checkpoint.checkpoint(self.h1_to_hd3, en1)
        h2_to_hd3 = checkpoint.checkpoint(self.h2_to_hd3, en2)
        h3_Cat_hd3 = checkpoint.checkpoint(self.h3_Cat_hd3, en3)
        hd4_Up_hd3 = checkpoint.checkpoint(self.hd4_Up_hd3, hd4, h3_Cat_hd3)
        hd5_Up_hd3 = checkpoint.checkpoint(self.hd5_Up_hd3, bottle, h3_Cat_hd3)
        hd3 = self.fusion3d(torch.cat((h1_to_hd3, h2_to_hd3, h3_Cat_hd3, hd4_Up_hd3, hd5_Up_hd3), dim=1))

        h1_to_hd2 = checkpoint.checkpoint(self.h1_to_hd2, en1)
        h2_cat_hd2 = checkpoint.checkpoint(self.h2_cat_hd2, en2)
        hd3_up_hd2 = checkpoint.checkpoint(self.hd3_up_hd2, hd3, h2_cat_hd2)
        hd4_up_hd2 = checkpoint.checkpoint(self.hd4_up_hd2, hd4, h2_cat_hd2)
        hd5_up_hd2 = checkpoint.checkpoint(self.hd5_up_hd2, bottle, h2_cat_hd2)
        hd2 = self.fusion2d(torch.cat((h1_to_hd2, h2_cat_hd2, hd3_up_hd2, hd4_up_hd2, hd5_up_hd2), dim=1))

        h1_Cat_hd1 = checkpoint.checkpoint(self.h1_Cat_hd1, en1)
        hd2_UT_hd1 = checkpoint.checkpoint(self.hd2_up_hd1, hd2, h1_Cat_hd1)
        hd3_UT_hd1 = checkpoint.checkpoint(self.hd3_up_hd1, hd3, h1_Cat_hd1)
        hd4_UT_hd1 = checkpoint.checkpoint(self.hd4_up_hd1, hd4, h1_Cat_hd1)
        hd5_UT_hd1 = checkpoint.checkpoint(self.hd5_up_hd1, bottle, h1_Cat_hd1)
        hd1 = self.fusion1d(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1,), dim=1))

        d5 = checkpoint.checkpoint(self.outconv5, bottle)
        d4 = checkpoint.checkpoint(self.outconv4, hd4)
        d3 = checkpoint.checkpoint(self.outconv3, hd3)
        d2 = checkpoint.checkpoint(self.outconv2, hd2)
        d1 = checkpoint.checkpoint(self.outconv1, hd1)

        d5 = _upsample(d5, d1)
        d4 = _upsample(d4, d1)
        d3 = _upsample(d3, d1)
        d2 = _upsample(d2, d1)

        d0 = checkpoint.checkpoint(self.outconv, torch.cat((d1, d2, d3, d4, d5), dim=1))
        return d0


class depu2net3plus(nn.Module):

    def __init__(self, in_channel, out_channel, side=False, upsample=False):
        super(depu2net3plus, self).__init__()
        self.model = U2net3p5dc(in_channel, out_channel, filters=[32, 64, 128, 256, 256], side=side, upsample=upsample)

    def forward(self, x):
        return self.model(x)


class swinu2net(nn.Module):
    """
    去除3+结构,6 stage  79.25±1.28
    """

    def __init__(self, in_channel, out_channel, img_size=64, side=False, ape=True, upsample=False,
                 window_size=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_rate=0.2, patch_size=2,
                 use_checkpoint=False):
        super(swinu2net, self).__init__()
        self.side = side
        self.upsample = upsample
        self.ape = ape
        self.filter = [32, 64, 128, 256, 256]
        self.en1 = DRSU7(in_channel, 16, 32, side=side, upsample=upsample, )
        self.pool12 = nn.MaxPool3d(2, 2)

        embed_dim = self.filter[0] * 2
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=self.filter[0], embed_dim=embed_dim, norm_layer=nn.LayerNorm)

        if self.ape:
            img_size = to_3tuple(img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1],
                                  img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop)

        self.en2 = DRSU6(self.filter[1], self.filter[0], self.filter[1], side=side, upsample=upsample, )
        self.pool23 = nn.MaxPool3d(2, 2)

        depths = [2, 4, 4, 8, 4, 4, 2]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.en3 = Block(in_channel=self.filter[1], mid_dim=self.filter[1] // 2, out_channel=self.filter[2],
                         refine_block=DRSU5, num_heads=4, side=side, upsample=upsample,
                         window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])], use_checkpoint=use_checkpoint, depth=1)
        self.pool34 = nn.MaxPool3d(2, 2)

        self.en4 = Block(in_channel=self.filter[2], mid_dim=self.filter[2] // 2, out_channel=self.filter[3],
                         refine_block=DRSU4, num_heads=8, side=side, upsample=upsample,
                         window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         use_checkpoint=use_checkpoint, depth=2)
        self.pool45 = nn.MaxPool3d(2, 2)

        self.en5 = Block(in_channel=self.filter[3], mid_dim=self.filter[3] // 2, out_channel=self.filter[4],
                         refine_block=DRSU4F, num_heads=16, side=side, upsample=upsample,
                         window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                         use_checkpoint=use_checkpoint, depth=2)
        self.pool56 = nn.MaxPool3d(2, 2)

        self.bottle = Block(in_channel=self.filter[4], mid_dim=self.filter[4] // 2, out_channel=self.filter[4],
                            refine_block=DRSU4F, num_heads=16, side=side, upsample=upsample,
                            window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                            use_checkpoint=use_checkpoint, depth=4)

        self.de5 = Block(in_channel=self.filter[4] * 2, mid_dim=self.filter[4] // 2, out_channel=self.filter[4],
                         refine_block=DRSU4F, num_heads=16, side=side, upsample=upsample,
                         window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         use_checkpoint=use_checkpoint, depth=2)

        self.de4 = Block(in_channel=self.filter[3] * 2, mid_dim=self.filter[3] // 2, out_channel=self.filter[2],
                         refine_block=DRSU4, num_heads=8, side=side, upsample=upsample,
                         window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         use_checkpoint=use_checkpoint, depth=2)

        self.de3 = Block(in_channel=self.filter[2] * 2, mid_dim=self.filter[2] // 2, out_channel=self.filter[1],
                         refine_block=DRSU5, num_heads=4, side=side, upsample=upsample,
                         window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:6]):sum(depths[:7])],
                         use_checkpoint=use_checkpoint, depth=1)

        self.de2 = DRSU6(self.filter[1] * 2, self.filter[1] // 2, self.filter[0], side=side, upsample=upsample, )

        if True:
            self.de1 = DRSU7(self.filter[0] * 2, self.filter[0] // 2, self.filter[0], side=side)
            self.bottlesie = DEPTHWISECONV(self.filter[4], out_channel)
            self.de5side = DEPTHWISECONV(self.filter[4], out_channel)
            self.de4side = DEPTHWISECONV(self.filter[2], out_channel)
            self.de3side = DEPTHWISECONV(self.filter[1], out_channel)
            self.de2side = DEPTHWISECONV(self.filter[0], out_channel)
            self.de1side = DEPTHWISECONV(self.filter[0], out_channel)
            # self.fusion = Block(64 * 5, out_channel, RSU7, num_heads=2, window_size=window_size, qkv_bias=qkv_bias,
            #                     qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            #                     use_checkpoint=use_checkpoint, depth=2)
            self.fusion = DEPTHWISECONV(out_channel * 6, out_channel)
        else:
            self.de1 = DRSU7(32 * 2, 16, out_ch=out_channel, side=side, upsample=upsample)

        # self.apply(_init_weights_trans)
        # self.apply(weights_init_kaiming)

    def _upsample(self, src, tar):
        return F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)

    def forward(self, x):
        hx = x
        en1 = self.en1(x)
        # en1pool = self.pool12(en1)

        patch = self.patch_embed(en1)
        H, W, T = patch.size(2), patch.size(3), patch.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(H, W, T), mode='trilinear',
                                               align_corners=True)
            patch = (patch + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        else:
            patch = patch.flatten(2).transpose(1, 2)  # [b l c]
        patch = self.pos_drop(patch)
        patch = rearrange(patch, 'b (h w t) c -> b c h w t', h=H, w=W, t=T)

        en2 = self.en2(patch)
        en2pool = self.pool23(en2)  # c

        en3 = self.en3(en2pool)  # t
        en3pool = self.pool34(en3)

        en4 = self.en4(en3pool)
        en4pool = self.pool45(en4)

        en5 = self.en5(en4pool)
        en5pool = self.pool56(en5)

        bottle = self.bottle(en5pool)
        bottleup = self._upsample(bottle, en5)

        de5 = self.de5(torch.cat((bottleup, en5), dim=1))
        de5up = self._upsample(de5, en4)

        de4 = self.de4(torch.cat((de5up, en4), dim=1))
        de4up = self._upsample(de4, en3)

        de3 = self.de3(torch.cat((de4up, en3), dim=1))
        de3up = self._upsample(de3, en2)

        de2 = self.de2(torch.cat((de3up, en2), dim=1))
        de2up = self._upsample(de2, en1)

        if True:
            de1 = self.de1(torch.cat((de2up, en1), dim=1))
            bottleside = self.bottlesie(bottle)
            bottleside = self._upsample(bottleside, de1)
            de5side = self.de5side(de5)
            de5side = self._upsample(de5side, de1)
            de4side = self.de4side(de4)
            de4side = self._upsample(de4side, de1)
            de3side = self.de3side(de3)
            de3side = self._upsample(de3side, de1)
            de2side = self.de2side(de2)
            de2side = self._upsample(de2side, de1)
            de1side = self.de1side(de1)
            out = self.fusion(torch.cat((de1side, de2side, de3side, de4side, de5side, bottleside), dim=1))
            return out
        else:
            de1 = self.de1(torch.cat((de2up, en1), dim=1))
            return de1


class swin3plus(nn.Module):
    """
    去除 u2net结构，只保留swin 和 3+ ,76.83±1.63
    """

    def __init__(self, in_channel, out_channel, filters=[32, 64, 128, 184, 256], side=False, img_size=64, window_size=4,
                 patch_size=2, ape=True, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_rate=0.2,
                 use_checkpoint=False):
        super(swin3plus, self).__init__()
        self.side = side
        self.ape = ape

        self.filters = [32, 64, 128, 256, 256]  # [32, 64, 128, 256, 320]
        self.CatChannels = self.filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.en1 = connectBlock(in_channel, self.filters[0])
        self.pool12 = nn.MaxPool3d(2, 2)

        embed_dim = self.filters[0] * 2
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=self.filters[0], embed_dim=embed_dim, norm_layer=nn.LayerNorm)

        if self.ape:
            img_size = to_3tuple(img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1],
                                  img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop)

        self.en2 = connectBlock(embed_dim, self.filters[1])
        self.pool23 = nn.MaxPool3d(2, 2)

        depths = [2, 4, 8, 4, 2]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.en3 = Block(in_channel=self.filters[1], mid_dim=self.filters[1] // 2, out_channel=self.filters[2],
                         num_heads=4, side=side, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         use_checkpoint=use_checkpoint, depth=1)
        self.pool34 = nn.MaxPool3d(2, 2)

        self.en4 = Block(in_channel=self.filters[2], mid_dim=self.filters[2] // 2, out_channel=self.filters[3],
                         num_heads=8, side=side, window_size=window_size, qkv_bias=qkv_bias,
                         qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         use_checkpoint=use_checkpoint, depth=2)
        self.pool45 = nn.MaxPool3d(2, 2)

        self.bottle = Block(in_channel=self.filters[3], mid_dim=self.filters[3] // 2, out_channel=self.filters[4],
                            side=side, num_heads=16, window_size=window_size, qkv_bias=qkv_bias,
                            qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                            drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                            use_checkpoint=use_checkpoint, depth=4)

        """to 4 stage"""
        self.h1_to_hd4 = connectBlock(self.filters[0], self.CatChannels, downstep=8)
        self.h2_to_hd4 = connectBlock(self.filters[1], self.CatChannels, downstep=4)
        self.h3_to_hd4 = connectBlock(self.filters[2], self.CatChannels, downstep=2)
        self.h4_Cat_hd4 = connectBlock(self.filters[3], self.CatChannels)
        self.hd5_Up_hd4 = connectBlock(self.filters[4], self.CatChannels)
        self.fusion4d = Block(in_channel=self.UpChannels, mid_dim=self.UpChannels // 2, out_channel=self.UpChannels,
                              side=side, num_heads=8, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                              use_checkpoint=use_checkpoint, depth=2)

        """to 3 stage"""
        self.h1_to_hd3 = connectBlock(self.filters[0], self.CatChannels, downstep=4)
        self.h2_to_hd3 = connectBlock(self.filters[1], self.CatChannels, downstep=2)
        self.h3_Cat_hd3 = connectBlock(self.filters[2], self.CatChannels)
        self.hd4_Up_hd3 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd5_Up_hd3 = connectBlock(self.filters[4], self.CatChannels)
        self.fusion3d = Block(in_channel=self.UpChannels, mid_dim=self.UpChannels // 2, out_channel=self.UpChannels,
                              num_heads=4, side=side, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                              use_checkpoint=use_checkpoint, depth=1)

        """to 2 stage"""
        self.h1_to_hd2 = connectBlock(self.filters[0], self.CatChannels, downstep=2)
        self.h2_cat_hd2 = connectBlock(self.filters[1], self.CatChannels)
        self.hd3_up_hd2 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd4_up_hd2 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd5_up_hd2 = connectBlock(self.filters[4], self.CatChannels)
        self.fusion2d = connectBlock(self.UpChannels, self.UpChannels)

        '''stage 1d'''
        self.h1_Cat_hd1 = connectBlock(self.filters[0], self.CatChannels)
        self.hd2_up_hd1 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd3_up_hd1 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd4_up_hd1 = connectBlock(self.UpChannels, self.CatChannels)
        self.hd5_up_hd1 = connectBlock(self.filters[4], self.CatChannels)
        self.fusion1d = connectBlock(self.UpChannels, self.UpChannels)

        self.outconv1 = DEPTHWISECONV(self.UpChannels, out_channel)
        self.outconv2 = DEPTHWISECONV(self.UpChannels, out_channel)
        self.outconv3 = DEPTHWISECONV(self.UpChannels, out_channel)
        self.outconv4 = DEPTHWISECONV(self.UpChannels, out_channel)
        self.outconv5 = DEPTHWISECONV(self.filters[4], out_channel)
        self.outconv = DEPTHWISECONV(self.CatBlocks, out_channel)

        # self.apply(_init_weights_trans)
        self.apply(weights_init_kaiming)

    def forward(self, x):
        hx = x
        en1 = self.en1(x)
        # en1pool = self.pool12(en1)

        patch = self.patch_embed(en1)
        H, W, T = patch.size(2), patch.size(3), patch.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(H, W, T), mode='trilinear',
                                               align_corners=True)
            patch = (patch + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        else:
            patch = patch.flatten(2).transpose(1, 2)  # [b l c]
        patch = self.pos_drop(patch)
        patch = rearrange(patch, 'b (h w t) c -> b c h w t', h=H, w=W, t=T)

        en2 = self.en2(patch)
        en2pool = self.pool23(en2)  # c

        en3 = self.en3(en2pool)  # t
        en3pool = self.pool34(en3)

        en4 = self.en4(en3pool)
        en4pool = self.pool45(en4)

        bottle = self.bottle(en4pool)

        h1_to_hd4 = self.h1_to_hd4(en1)  # 将其他通道融合为统一通道
        h2_to_hd4 = self.h2_to_hd4(en2)
        h3_to_hd4 = self.h3_to_hd4(en3)
        h4_Cat_hd4 = self.h4_Cat_hd4(en4)
        hd5_Up_hd4 = self.hd5_Up_hd4(bottle, h4_Cat_hd4)  # h4_Cat_hd4上采样目标大小
        hd4 = self.fusion4d(torch.cat((h1_to_hd4, h2_to_hd4, h3_to_hd4, h4_Cat_hd4, hd5_Up_hd4), dim=1))

        h1_to_hd3 = self.h1_to_hd3(en1)
        h2_to_hd3 = self.h2_to_hd3(en2)
        h3_Cat_hd3 = self.h3_Cat_hd3(en3)
        hd4_Up_hd3 = self.hd4_Up_hd3(hd4, h3_Cat_hd3)
        hd5_Up_hd3 = self.hd5_Up_hd3(bottle, h3_Cat_hd3)
        hd3 = self.fusion3d(torch.cat((h1_to_hd3, h2_to_hd3, h3_Cat_hd3, hd4_Up_hd3, hd5_Up_hd3), dim=1))

        h1_to_hd2 = self.h1_to_hd2(en1)
        h2_cat_hd2 = self.h2_cat_hd2(en2)
        hd3_up_hd2 = self.hd3_up_hd2(hd3, h2_cat_hd2)
        hd4_up_hd2 = self.hd4_up_hd2(hd4, h2_cat_hd2)
        hd5_up_hd2 = self.hd5_up_hd2(bottle, h2_cat_hd2)
        hd2 = self.fusion2d(torch.cat((h1_to_hd2, h2_cat_hd2, hd3_up_hd2, hd4_up_hd2, hd5_up_hd2), dim=1))

        h1_Cat_hd1 = self.h1_Cat_hd1(en1)
        hd2_UT_hd1 = self.hd2_up_hd1(hd2, h1_Cat_hd1)
        hd3_UT_hd1 = self.hd3_up_hd1(hd3, h1_Cat_hd1)
        hd4_UT_hd1 = self.hd4_up_hd1(hd4, h1_Cat_hd1)
        hd5_UT_hd1 = self.hd5_up_hd1(bottle, h1_Cat_hd1)
        hd1 = self.fusion1d(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1,), dim=1))

        d5 = self.outconv5(bottle)
        d4 = self.outconv4(hd4)
        d3 = self.outconv3(hd3)
        d2 = self.outconv2(hd2)
        d1 = self.outconv1(hd1)

        d5 = _upsample(d5, d1)
        d4 = _upsample(d4, d1)
        d3 = _upsample(d3, d1)
        d2 = _upsample(d2, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), dim=1))
        return d0


if __name__ == '__main__':
    var = torch.rand(3, 1, 64, 64, 64)
    x = Variable(var)
    model = swinu2net(1, 1, side=True)
    x = model(x)
    print(x.shape)
