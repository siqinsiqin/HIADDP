# -*-coding:utf-8 -*-
"""
# Time       ：2022/8/10 18:24
# Author     ：comi
# version    ：python 3.8
# Description：
"""
# -*-coding:utf-8 -*-
from models.swinu2net3p.model import Block
# from models.u2netV.U2netV2 import DRSU7, DRSU6, DRSU5, DRSU4, DRSU4F, DEPTHWISECONV

import torch
from einops import rearrange
from timm.models.layers import trunc_normal_, to_3tuple
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.nn.init import trunc_normal_

import torch.nn.functional as F


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


# class Block(nn.Module):
#     """
#     两种设想
#     sw msa -> u2block -> redusial -> muti
#     w msa-> u2block sigmold -> redusial -> muti-> sw msa
#     """
#
#     def __init__(self, in_channel, mid_dim, out_channel, refine_block=None, num_heads=2, mlp_ratio=4., window_size=4,
#                  qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., side=False,
#                  norm_layer=nn.LayerNorm, use_checkpoint=False, depth=2):
#
#         super(Block, self).__init__()
#         self.window_size = to_3tuple(window_size)
#         self.shift_size = to_3tuple(window_size // 2)
#         self.use_checkpoint = use_checkpoint
#         self.mid_dim = mid_dim
#
#         assert 0 <= min(self.shift_size) < min(
#             self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}" \
#             .format(self.shift_size, self.window_size)
#
#         self.blocks = nn.ModuleList()
#         for i in range(depth):
#             self.blocks.append(
#                 SwinTransformerBlock(
#                     dim=in_channel,
#                     num_heads=num_heads,
#                     window_size=self.window_size,
#                     shift_size=to_3tuple(0),
#                     mlp_ratio=mlp_ratio,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop,
#                     attn_drop=attn_drop,
#                     drop_path=drop_path[0 + (i * 2)] if isinstance(drop_path, list) else drop_path,
#                     norm_layer=norm_layer))
#             self.blocks.append(
#                 SwinTransformerBlock(
#                     dim=in_channel,
#                     num_heads=num_heads,
#                     window_size=self.window_size,
#                     shift_size=self.shift_size,
#                     mlp_ratio=mlp_ratio,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop,
#                     attn_drop=attn_drop,
#                     drop_path=drop_path[1 + (i * 2)] if isinstance(drop_path, list) else drop_path,
#                     norm_layer=norm_layer)
#             )
#
#         if refine_block is not None:
#             self.refine_block = refine_block(in_channel, self.mid_dim, out_channel, side=side)
#         else:
#             self.refine_block = None
#
#     def proj_feat(self, x, hidden_size, feat_size):
#         x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
#         x = x.permute(0, 4, 1, 2, 3).contiguous()
#         return x
#
#     def toConv(self, x, H, W, T, out_dim):
#         return x.view(-1, H, W, T, out_dim).permute(0, 4, 1, 2, 3).contiguous()
#
#     def createmask(self, x, H, W, T):
#         Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
#         Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
#         Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
#         img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
#         h_slices = (slice(0, -self.window_size[0]),
#                     slice(-self.window_size[0], -self.shift_size[0]),
#                     slice(-self.shift_size[0], None))
#         w_slices = (slice(0, -self.window_size[1]),
#                     slice(-self.window_size[1], -self.shift_size[1]),
#                     slice(-self.shift_size[1], None))
#         t_slices = (slice(0, -self.window_size[2]),
#                     slice(-self.window_size[2], -self.shift_size[2]),
#                     slice(-self.shift_size[2], None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 for t in t_slices:
#                     img_mask[:, h, w, t, :] = cnt
#                     cnt += 1
#
#         mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#         mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         return attn_mask
#
#     def forward(self, x):
#         H, W, T = x.size(2), x.size(3), x.size(4)
#         x = rearrange(x, 'b c h w t-> b (h w t) c', h=H, w=W, t=T)
#
#         attn_mask = self.createmask(x, H, W, T)
#
#         for blk in self.blocks:
#             blk.H, blk.W, blk.T = H, W, T
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x, attn_mask)
#             else:
#                 x = blk(x, attn_mask)
#
#         x = rearrange(x, 'b (w h t) c -> b c w h t', h=H, w=W, t=T)
#         if self.refine_block is not None:
#             x = self.refine_block(x)  # block 优化
#
#         return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x


class swinU(nn.Module):
    """ "
    5 stage  79.44
    """

    def __init__(self, in_channel, out_channel, img_size=64, side=False, ape=True,
                 window_size=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_rate=0.2, patch_size=2,
                 use_checkpoint=False, ):
        # window size ，patch size 4，4
        super(swinU, self).__init__()
        self.side = side
        self.ape = ape

        self.en1 = DRSU7(in_channel, 16, 32, side=side)
        self.pool12 = nn.MaxPool3d(2, 2)

        embed_dim = 64
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=32, embed_dim=embed_dim, norm_layer=nn.LayerNorm)

        if self.ape:
            img_size = to_3tuple(img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1],
                                  img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop)

        self.en2 = DRSU6(64, 32, 64, side=side)
        self.pool23 = nn.MaxPool3d(2, 2)

        depths = [2, 4, 8, 4, 2]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.en3 = Block(64, 32, 128, DRSU5, num_heads=4, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         side=side,
                         drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         use_checkpoint=use_checkpoint, depth=1)
        self.pool34 = nn.MaxPool3d(2, 2)

        self.en4 = Block(128, 64, 256, DRSU4, num_heads=8, window_size=window_size, qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         side=side,
                         drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         use_checkpoint=use_checkpoint, depth=2)
        self.pool45 = nn.MaxPool3d(2, 2)

        self.bottle = Block(256, 128, 256, DRSU4F, num_heads=16, window_size=window_size, qkv_bias=qkv_bias, side=side,
                            qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                            use_checkpoint=use_checkpoint, depth=4)

        self.de4 = Block(256 * 2, 128, 128, DRSU4, num_heads=8, window_size=window_size, qkv_bias=qkv_bias, side=side,
                         qk_scale=qk_scale,
                         drop=drop, attn_drop=attn_drop, drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                         use_checkpoint=use_checkpoint, depth=2)

        self.de3 = Block(128 * 2, 128, 64, DRSU5, num_heads=4, window_size=window_size, qkv_bias=qkv_bias, side=side,
                         qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         use_checkpoint=use_checkpoint, depth=1)

        self.de2 = DRSU6(64 * 2, 64, 32, side=side)

        if True:
            self.de1 = DRSU7(32 * 2, 32, 32, side=side)
            self.bottlesie = DEPTHWISECONV(256, out_channel)
            self.de4side = DEPTHWISECONV(128, out_channel)
            self.de3side = DEPTHWISECONV(64, out_channel)
            self.de2side = DEPTHWISECONV(32, out_channel)
            self.de1side = DEPTHWISECONV(32, out_channel)
            # self.fusion = Block(64 * 5, out_channel, RSU7, num_heads=2, window_size=window_size, qkv_bias=qkv_bias,
            #                     qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            #                     use_checkpoint=use_checkpoint, depth=2)
            self.fusion = DEPTHWISECONV(out_channel * 5, out_channel)
        else:
            self.de1 = DRSU7(32 * 2, 16, out_ch=out_channel, side=side)

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

        bottle = self.bottle(en4pool)
        bottleup = self._upsample(bottle, en4)

        de4 = self.de4(torch.cat((bottleup, en4), dim=1))
        de4up = self._upsample(de4, en3)

        de3 = self.de3(torch.cat((de4up, en3), dim=1))
        de3up = self._upsample(de3, en2)

        de2 = self.de2(torch.cat((de3up, en2), dim=1))
        de2up = self._upsample(de2, en1)

        if True:
            de1 = self.de1(torch.cat((de2up, en1), dim=1))
            bottleside = self.bottlesie(bottle)
            bottleside = self._upsample(bottleside, de1)
            de4side = self.de4side(de4)
            de4side = self._upsample(de4side, de1)
            de3side = self.de3side(de3)
            de3side = self._upsample(de3side, de1)
            de2side = self.de2side(de2)
            de2side = self._upsample(de2side, de1)
            de1side = self.de1side(de1)
            out = self.fusion(torch.cat((de1side, de2side, de3side, de4side, bottleside), dim=1))
            return out
        else:
            de1 = self.de1(torch.cat((de2up, en1), dim=1))
            return de1


if __name__ == '__main__':
    SIZE = 64

    x = Variable(torch.rand(8, 1, SIZE, SIZE, SIZE)).cuda()

    model = swinU(1, 1, side=True).cuda()
    # model = Block(1, 1).cuda()
    x = model(x)
    print(x.shape)
    # Output shape: torch.Size([8, 1, 64, 64, 64]) v3
    # Computational complexity:       86.93 GMac
    # Number of parameters:           65.15 M
    # Output shape: torch.Size([8, 1, 64, 64, 64]) v2
    # Computational complexity:       218.66 GMac
    # Number of parameters:           136.54 M
