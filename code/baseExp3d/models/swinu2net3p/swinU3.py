# -*-coding:utf-8 -*-
"""
# Time       ：2022/8/3 19:30
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import numpy as np
import torch
from einops import rearrange
from timm.models.layers import trunc_normal_, to_3tuple
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.nn.init import trunc_normal_
from torch.utils import checkpoint

from models.swinu2net3p.TransMorph3D import SwinTransformerBlock, window_partition
from models.u2netV.U2net import RSU7, RSU6, RSU5, RSU4, RSU4F, U2baseblock
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


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class Block(nn.Module):
    """
    两种设想
    sw msa -> u2block -> redusial -> muti
    w msa-> u2block sigmold -> redusial -> muti-> sw msa
    """

    def __init__(self, dim, attn_block=None, num_heads=2, mlp_ratio=4., window_size=8, qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., downsample=None, norm_layer=nn.LayerNorm,
                 use_checkpoint=False, reduce_factor=2):

        super(Block, self).__init__()
        self.window_size = to_3tuple(window_size)
        self.shift_size = to_3tuple(window_size // 2)
        self.use_checkpoint = use_checkpoint
        self.out_dim = dim

        assert 0 <= min(self.shift_size) < min(
            self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}" \
            .format(self.shift_size, self.window_size)

        # print('drop :', drop_path)
        self.wMSA = SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            shift_size=to_3tuple(0),
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer)

        if attn_block is not None:
            self.attn_block = attn_block(dim, dim // 2, dim)
            self.sigmoid = nn.Sigmoid()
        else:
            self.attn_block = None

        self.swMSA = SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            shift_size=self.shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[1] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=reduce_factor)
        else:
            self.downsample = None

        self.norm = nn.LayerNorm(dim)

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

    def forward(self, x, H, W, T):
        hx = x
        attn_mask = self.createmask(x, H, W, T)

        self.wMSA.H, self.wMSA.W, self.wMSA.T = H, W, T
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.wMSA, x, attn_mask)
        else:
            x = self.wMSA(x, attn_mask)

        if self.attn_block is not None:
            x = self.proj_feat(x, self.out_dim, (H, W, T))
            x = self.attn_block(x)
            scale = self.sigmoid(x)
            scale = rearrange(scale, 'b c w h t-> b (w h t) c')
            x = hx * scale

        self.swMSA.H, self.swMSA.W, self.swMSA.T = H, W, T
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.swMSA, x, attn_mask)
        else:
            x = self.swMSA(x, attn_mask)

        x_out = self.norm(x)
        x_out = self.toConv(x_out, H, W, T, self.out_dim)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x_out, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x_out, H, W, T, x, H, W, T


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


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        # self.reduction = nn.Linear(8 * dim, reduce_factor * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = rearrange(x, 'b (h w t) c -> b h w t c', h=H, w=W, t=T)
        # x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class RSUT5(nn.Module):
    """
    设想：在bottle neck层加入transformer
    添加side引导
    多尺度连接
    """

    def __init__(self, in_channel, out_channel, num_heads=2, window_size=4, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., reduce_factor=2, side=False,
                 muticnt=False, depth=2, dpr=0, outd=None, downsample=PatchMerging, norm_layer=nn.LayerNorm):
        super(RSUT5, self).__init__()
        self.tmp_dim = in_channel
        self.side = side  # side 引导
        self.muticnt = muticnt  # 多尺度连接

        if outd is not None:
            self.downsample = outd(dim=out_channel, norm_layer=norm_layer, reduce_factor=reduce_factor)
        else:
            self.downsample = None

        dpr = [x.item() for x in torch.linspace(dpr[0], dpr[1], 8 + depth * 2)]
        self.en1 = Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=dpr[0:2], reduce_factor=reduce_factor,
                         downsample=downsample
                         )
        self.en2 = Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=dpr[2:4], reduce_factor=reduce_factor,
                         downsample=downsample
                         )
        self.en3 = Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=dpr[4:6], reduce_factor=reduce_factor,
                         downsample=downsample
                         )
        self.en4 = Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=dpr[6:8], reduce_factor=reduce_factor,
                         downsample=downsample
                         )

        self.bottle = nn.ModuleList()
        for i in range(depth):
            self.bottle.append(
                Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=dpr[8 + i * 2:10 + i * 2], reduce_factor=reduce_factor,
                      downsample=None
                      )
                , )

        self.de4 = U2baseblock(self.tmp_dim * 2, self.tmp_dim, dirate=1)
        self.de3 = U2baseblock(self.tmp_dim * 2, self.tmp_dim, dirate=1)
        self.de2 = U2baseblock(self.tmp_dim * 2, self.tmp_dim, dirate=1)

        if self.side:
            self.de1 = U2baseblock(self.tmp_dim * 2, self.tmp_dim, dirate=1)
        else:
            self.de1 = U2baseblock(self.tmp_dim * 2, out_channel, dirate=1)

        self.bottleside = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.side4 = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.side3 = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.side2 = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.side1 = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.outconv = nn.Conv3d(5 * out_channel, out_channel, 1)

    def _upsample(self, src, tar):
        return F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)

    def forward(self, x, H, W, T):

        en1, H, W, T, x, H, W, T = self.en1(x, H, W, T)
        # print(en1.shape, H, W, T, x.shape, H, W, T)
        en2, H, W, T, x, H, W, T = self.en2(x, H, W, T)
        # print(en2.shape, H, W, T, x.shape, H, W, T)
        en3, H, W, T, x, H, W, T = self.en3(x, H, W, T)
        # print(en3.shape, H, W, T, x.shape, H, W, T)
        en4, H, W, T, x, H, W, T = self.en4(x, H, W, T)
        # print(en4.shape, H, W, T, x.shape, H, W, T)

        for blk in self.bottle:
            bottle, H, W, T, x, H, W, T = blk(x, H, W, T)
        bottleup = self._upsample(bottle, en4)

        de4 = self.de4(torch.cat((bottleup, en4), dim=1))
        de4up = self._upsample(de4, en3)

        de3 = self.de3(torch.cat((de4up, en3), dim=1))
        de3up = self._upsample(de3, en2)

        de2 = self.de2(torch.cat((de3up, en2), dim=1))
        de2up = self._upsample(de2, en1)

        de1 = self.de1(torch.cat((de2up, en1), dim=1))

        if self.side:
            bottleside = self.bottleside(bottle)
            bottleside = self._upsample(bottleside, en1)

            side4 = self.side4(de4)
            side4 = self._upsample(side4, en1)

            side3 = self.side3(de3)
            side3 = self._upsample(side3, en1)

            side2 = self.side2(de2)
            side2 = self._upsample(side2, en1)

            side1 = self.side1(de1)
            out = self.outconv(torch.cat((side1, side2, side3, side4, bottleside), dim=1))
            H, W, T = out.size(2), out.size(3), out.size(4)
            if self.downsample is not None:
                x_down = self.downsample(rearrange(out, 'b c h w t -> b (h w t) c'), H, W, T)
                Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
                return out, H, W, T, x_down, Wh, Ww, Wt
            return out, H, W, T, rearrange(out, 'b c h w t -> b (h w t) c'), H, W, T,
        else:
            H, W, T = de1.size(2), de1.size(3), de1.size(4)
            if self.downsample is not None:
                x_down = self.downsample(rearrange(de1, 'b c h w t -> b (h w t) c', ), H, W, T)
                Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
                return de1, H, W, T, x_down, Wh, Ww, Wt
            return de1, H, W, T, rearrange(de1, 'b c h w t -> b (h w t) c'), H, W, T,


class RSUT4(nn.Module):
    """
    设想：在bottle neck层加入transformer
    添加side引导
    多尺度连接
    """

    def __init__(self, in_channel, out_channel, num_heads=2, window_size=4, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., reduce_factor=2, side=False,
                 muticnt=False, depth=2, dpr=0, outd=None, downsample=PatchMerging, norm_layer=nn.LayerNorm):
        super(RSUT4, self).__init__()
        self.tmp_dim = in_channel
        self.side = side  # side 引导
        self.muticnt = muticnt  # 多尺度连接
        if outd is not None:
            self.downsample = downsample(dim=out_channel, norm_layer=norm_layer, reduce_factor=reduce_factor)
        else:
            self.downsample = None

        dpr = [x.item() for x in torch.linspace(dpr[0], dpr[1], 6 + depth * 2)]

        self.en1 = Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=dpr[0:2], reduce_factor=reduce_factor,
                         downsample=downsample
                         )
        self.en2 = Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=dpr[2:4], reduce_factor=reduce_factor,
                         downsample=downsample
                         )
        self.en3 = Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=dpr[4:6], reduce_factor=reduce_factor,
                         downsample=downsample
                         )

        self.bottle = nn.ModuleList()
        for i in range(depth):
            self.bottle.append(
                Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=dpr[6 + i * 2:8 + i * 2], reduce_factor=reduce_factor,
                      downsample=None
                      )
                , )

        self.de3 = U2baseblock(self.tmp_dim * 2, self.tmp_dim, dirate=1)
        self.de2 = U2baseblock(self.tmp_dim * 2, self.tmp_dim, dirate=1)

        if self.side:
            self.de1 = U2baseblock(self.tmp_dim * 2, self.tmp_dim, dirate=1)
        else:
            self.de1 = U2baseblock(self.tmp_dim * 2, out_channel, dirate=1)

        self.bottleside = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.side3 = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.side2 = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.side1 = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.outconv = nn.Conv3d(4 * out_channel, out_channel, 1)

    def _upsample(self, src, tar):
        return F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)

    def forward(self, x, H, W, T):

        en1, H, W, T, x, H, W, T = self.en1(x, H, W, T)
        # print(en1.shape, H, W, T, x.shape, H, W, T)
        en2, H, W, T, x, H, W, T = self.en2(x, H, W, T)
        # print(en2.shape, H, W, T, x.shape, H, W, T)
        en3, H, W, T, x, H, W, T = self.en3(x, H, W, T)
        # print(en3.shape, H, W, T, x.shape, H, W, T)

        for blk in self.bottle:
            bottle, H, W, T, x, H, W, T = blk(x, H, W, T)
        bottleup = self._upsample(bottle, en3)

        de3 = self.de3(torch.cat((bottleup, en3), dim=1))
        de3up = self._upsample(de3, en2)

        de2 = self.de2(torch.cat((de3up, en2), dim=1))
        de2up = self._upsample(de2, en1)

        de1 = self.de1(torch.cat((de2up, en1), dim=1))

        if self.side:
            bottleside = self.bottleside(bottle)
            bottleside = self._upsample(bottleside, en1)

            side3 = self.side3(de3)
            side3 = self._upsample(side3, en1)

            side2 = self.side2(de2)
            side2 = self._upsample(side2, en1)

            side1 = self.side1(de1)
            out = self.outconv(torch.cat((side1, side2, side3, bottleside), dim=1))

            H, W, T = out.size(2), out.size(3), out.size(4)
            if self.downsample is not None:
                x_down = self.downsample(rearrange(out, 'b c h w t -> b (h w t) c'), H, W, T)
                Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
                return out, H, W, T, x_down, Wh, Ww, Wt
            return out, H, W, T, rearrange(out, 'b c h w t -> b (h w t) c'), H, W, T,
        else:
            H, W, T = de1.size(2), de1.size(3), de1.size(4)
            if self.downsample is not None:
                x_down = self.downsample(rearrange(de1, 'b c h w t -> b (h w t) c', ), H, W, T)
                Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
                return de1, H, W, T, x_down, Wh, Ww, Wt
            return de1, H, W, T, rearrange(de1, 'b c h w t -> b (h w t) c'), H, W, T,


class RSUT3(nn.Module):
    """
    设想：在bottle neck层加入transformer
    添加side引导
    多尺度连接
    """

    def __init__(self, in_channel, out_channel, num_heads=4, window_size=4, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., reduce_factor=2, side=False,
                 muticnt=False, depth=2, dpr=0, outd=None, downsample=PatchMerging, norm_layer=nn.LayerNorm):
        super(RSUT3, self).__init__()
        self.tmp_dim = in_channel
        self.side = side  # side 引导
        self.muticnt = muticnt  # 多尺度连接

        if outd is not None:
            self.downsample = outd(dim=out_channel, norm_layer=norm_layer, reduce_factor=reduce_factor)
        else:
            self.downsample = None

        dpr = [x.item() for x in torch.linspace(dpr[0], dpr[1], 4 + depth * 2)]

        self.en1 = Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=dpr[0:2], reduce_factor=reduce_factor,
                         downsample=downsample
                         )
        self.en2 = Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                         attn_drop=attn_drop_rate, drop_path=dpr[2:4], reduce_factor=reduce_factor,
                         downsample=downsample
                         )

        self.bottle = nn.ModuleList()
        for i in range(depth):
            self.bottle.append(
                Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=dpr[4 + i * 2:6 + i * 2], reduce_factor=reduce_factor,
                      downsample=None
                      )
                , )

        self.de2 = U2baseblock(self.tmp_dim * 2, self.tmp_dim, dirate=1)

        if self.side:
            self.de1 = U2baseblock(self.tmp_dim * 2, self.tmp_dim, dirate=1)
        else:
            self.de1 = U2baseblock(self.tmp_dim * 2, out_channel, dirate=1)

        self.bottleside = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.side2 = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.side1 = nn.Conv3d(self.tmp_dim, out_channel, 3, padding=1)
        self.outconv = nn.Conv3d(3 * out_channel, out_channel, 1)

    def _upsample(self, src, tar):
        return F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)

    def forward(self, x, H, W, T):

        en1, H, W, T, x, H, W, T = self.en1(x, H, W, T)
        # print(en1.shape, H, W, T, x.shape, H, W, T)
        en2, H, W, T, x, H, W, T = self.en2(x, H, W, T)
        # print(en2.shape, H, W, T, x.shape, H, W, T)

        for blk in self.bottle:
            bottle, H, W, T, x, H, W, T = blk(x, H, W, T)

        de3up = self._upsample(bottle, en2)

        de2 = self.de2(torch.cat((de3up, en2), dim=1))
        de2up = self._upsample(de2, en1)

        de1 = self.de1(torch.cat((de2up, en1), dim=1))

        if self.side:
            bottleside = self.bottleside(bottle)
            bottleside = self._upsample(bottleside, en1)

            side2 = self.side2(de2)
            side2 = self._upsample(side2, en1)

            side1 = self.side1(de1)
            out = self.outconv(torch.cat((side1, side2, bottleside), dim=1))
            H, W, T = out.size(2), out.size(3), out.size(4)
            if self.downsample is not None:
                x_down = self.downsample(rearrange(out, 'b c h w t -> b (h w t) c'), H, W, T)
                Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
                return out, H, W, T, x_down, Wh, Ww, Wt
            return out, H, W, T, rearrange(out, 'b c h w t -> b (h w t) c'), H, W, T,
        else:
            H, W, T = de1.size(2), de1.size(3), de1.size(4)
            if self.downsample is not None:
                x_down = self.downsample(rearrange(de1, 'b c h w t -> b (h w t) c', ), H, W, T)
                Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
                return de1, H, W, T, x_down, Wh, Ww, Wt
            return de1, H, W, T, rearrange(de1, 'b c h w t -> b (h w t) c'), H, W, T,


class RSUT2(nn.Module):
    """
    设想：在bottle neck层加入transformer
    添加side引导
    多尺度连接
    """

    def __init__(self, dim, num_heads=4, window_size=4, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., reduce_factor=2, side=False,
                 muticnt=False, depth=2, dpr=0, outd=None, norm_layer=nn.LayerNorm):
        super(RSUT2, self).__init__()
        self.tmp_dim = dim  # max((in_channel // 2), (out_channel // 2))
        self.side = side  # side 引导
        self.muticnt = muticnt  # 多尺度连接
        if outd is not None:
            self.downsample = outd(dim=dim, norm_layer=norm_layer, reduce_factor=reduce_factor // 2)
        else:
            self.downsample = None

        dpr = [x.item() for x in torch.linspace(dpr[0], dpr[-1], depth * 2)]
        self.bottle = nn.ModuleList()
        for i in range(depth):
            self.bottle.append(
                Block(dim=self.tmp_dim, attn_block=None, num_heads=num_heads, window_size=window_size,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=dpr[i * 2:2 + i * 2], reduce_factor=reduce_factor,
                      downsample=None))

    def _upsample(self, src, tar):
        return F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)

    def forward(self, x, H, W, T):

        for blk in self.bottle:
            bottle, H, W, T, x, H, W, T = blk(x, H, W, T)

        return bottle, H, W, T, x, H, W, T


class swinU3(nn.Module):

    def __init__(self,
                 img_size=64,
                 patch_size=2,
                 in_chans=1,
                 embed_dim=64,
                 depths=[2, 2, 2, 4],
                 num_heads=[4, 8, 16, 16],
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 ape=True,
                 patch_norm=True,
                 frozen_stages=-1,
                 reduce_factor=2,
                 side=False,
                 sup=False
                 ):
        super(swinU3, self).__init__()
        self.side = side
        self.sup = sup
        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages

        self.features = [64, 128, 256, 384, 512]
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm)

        if self.ape:
            img_size = to_3tuple(self.img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1],
                                  img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.encoder1 = Block(dim=embed_dim,
                              attn_block=RSU4,
                              num_heads=num_heads[0],
                              window_size=window_size,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              drop=drop_rate,
                              attn_drop=attn_drop_rate,
                              drop_path=dpr[sum(depths[:0]):sum(depths[: 1])],
                              reduce_factor=reduce_factor,
                              downsample=PatchMerging
                              )

        self.encoder2 = Block(dim=embed_dim * 2,
                              attn_block=RSU4,
                              num_heads=num_heads[1],
                              window_size=window_size,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              drop=drop_rate,
                              attn_drop=attn_drop_rate,
                              drop_path=dpr[sum(depths[:1]):sum(depths[: 2])],
                              reduce_factor=reduce_factor,
                              downsample=PatchMerging
                              )

        self.encoder3 = Block(dim=embed_dim * 4,
                              attn_block=RSU4,
                              num_heads=num_heads[2],
                              window_size=window_size,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              drop=drop_rate,
                              attn_drop=attn_drop_rate,
                              drop_path=dpr[sum(depths[:2]):sum(depths[: 3])],
                              reduce_factor=reduce_factor,
                              downsample=PatchMerging
                              )

        self.encoder4 = Block(dim=embed_dim * 8,
                              attn_block=RSU4,
                              num_heads=num_heads[3],
                              window_size=window_size,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              drop=drop_rate,
                              attn_drop=attn_drop_rate,
                              drop_path=dpr[sum(depths[:3]):sum(depths[: 4])],
                              reduce_factor=reduce_factor,
                              downsample=PatchMerging
                              )

        self.bottleneck = RSU4F(embed_dim * 8, embed_dim * 8 // 2, embed_dim * 8)
        self.deconder4 = RSU4(embed_dim * 16, embed_dim * 16 // 2, embed_dim * 4)
        self.deconder3 = RSU5(embed_dim * 8, embed_dim * 8 // 2, embed_dim * 2)
        self.deconder2 = RSU6(embed_dim * 4, embed_dim * 4 // 2, embed_dim)
        self.deconder1 = RSU7(embed_dim * 2, embed_dim * 2 // 2, embed_dim)

        self.side1 = nn.Conv3d(embed_dim, 1, 3, padding=1)
        self.side2 = nn.Conv3d(embed_dim, 1, 3, padding=1)
        self.side3 = nn.Conv3d(embed_dim * 2, 1, 3, padding=1)
        self.side4 = nn.Conv3d(embed_dim * 4, 1, 3, padding=1)
        self.side5 = nn.Conv3d(embed_dim * 8, 1, 3, padding=1)
        self.outconv = nn.Conv3d(5, 1, 1)

        self.target = torch.Tensor(1, 1, self.img_size, self.img_size, self.img_size)  # 目标大小

        # self.apply(_init_weights)
        #
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.apply(weights_init_kaiming)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.apply(weights_init_kaiming)

    def _upsample(self, src, tar):
        return F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)

    def forward(self, x):

        x = self.patch_embed(x)
        H, W, T = x.size(2), x.size(3), x.size(4)

        # x = rearrange(x, 'b c h w t -> b (h w t) c', )

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(H, W, T), mode='trilinear',
                                               align_corners=True)
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        x_out1, H, W, T, x_down, Wh, Ww, Wt = self.encoder1(x, H, W, T)
        # print(x_out1.shape, )
        x_out2, H, W, T, x_down, Wh, Ww, Wt = self.encoder2(x_down, Wh, Ww, Wt)
        # print(x_out2.shape, )
        x_out3, H, W, T, x_down, Wh, Ww, Wt = self.encoder3(x_down, Wh, Ww, Wt)
        # print(x_out3.shape)
        x_out4, H, W, T, x_down, Wh, Ww, Wt = self.encoder4(x_down, Wh, Ww, Wt)
        # print(x_out4.shape, )

        bottle = self.bottleneck(x_out4)
        bottle = self._upsample(bottle, x_out4)

        d4 = self.deconder4(torch.cat((bottle, x_out4), dim=1))
        d4_up = self._upsample(d4, x_out3)
        # print('d4:', d4.shape)

        d3 = self.deconder3(torch.cat((d4_up, x_out3), dim=1))
        d3_up = self._upsample(d3, x_out2)
        # print('d3:', d3.shape)

        d2 = self.deconder2(torch.cat((d3_up, x_out2), dim=1))
        d2_up = self._upsample(d2, x_out1)
        # print('d2:', d2.shape)

        d1 = self.deconder1(torch.cat((d2_up, x_out1), dim=1))
        d1_up = self._upsample(d1, self.target)

        d1 = self.side1(d1_up)

        if self.side:
            d2 = self.side2(d2)
            d2 = self._upsample(d2, d1)

            d3 = self.side3(d3)
            d3 = self._upsample(d3, d1)

            d4 = self.side4(d4)
            d4 = self._upsample(d4, d1)

            d5 = self.side5(bottle)
            d5 = self._upsample(d5, d1)

            d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))
            return d0
        if self.sup:
            d2 = self.side2(d2)
            d2 = self._upsample(d2, d1)

            d3 = self.side3(d3)
            d3 = self._upsample(d3, d1)

            d4 = self.side4(d4)
            d4 = self._upsample(d4, d1)

            d5 = self.side5(bottle)
            d5 = self._upsample(d5, d1)

            d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))
            return [d0, d1, d2, d3, d4, d5]
        return d1


class swinV(nn.Module):
    def __init__(self, in_channel, out_channel, img_size=64, ape=True, num_heads=[4, 4, 8, 8],
                 depths=[2, 2, 2, 4], window_size=4, patch_size=2, drop_rate=0., drop_path_rate=0.2,
                 side=True):
        super(swinV, self).__init__()
        self.filters = [64, 128, 256, 512]
        self.side = side
        self.ape = ape
        embed_dim = self.filters[1]

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm)

        if self.ape:
            img_size = to_3tuple(img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1],
                                  img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # enconder
        self.en1 = RSU7(in_channel, self.filters[0] // 2, self.filters[0])
        self.pool12 = nn.MaxPool3d(2, 2)
        self.en2 = RSU6(self.filters[0], self.filters[1] // 2, self.filters[1])
        self.fusion = nn.Conv3d(self.filters[1] * 2, self.filters[1], 3, padding=1)

        self.en3 = RSUT5(self.filters[1], self.filters[2], num_heads=4, window_size=window_size, outd=PatchMerging,
                         reduce_factor=8, dpr=dpr[0:2], depth=2, side=True)

        self.en4 = RSUT4(self.filters[2], self.filters[3], num_heads=4, window_size=window_size, outd=PatchMerging,
                         reduce_factor=8, dpr=dpr[2:4], depth=2, side=True)

        self.en5 = RSUT3(self.filters[3], self.filters[3], num_heads=4, window_size=window_size, outd=None,
                         reduce_factor=8, depth=2, dpr=dpr[4:6], side=True)

        self.bottle = RSUT2(self.filters[3], num_heads=4, window_size=window_size, outd=None,
                            reduce_factor=8, depth=2, dpr=dpr[6:10], side=True)

        # deconder
        self.de5 = RSUT3(self.filters[3] * 2, self.filters[3], num_heads=8, window_size=window_size, outd=None,
                         reduce_factor=8, depth=2, dpr=dpr[4:6], side=True)

        self.de4 = RSUT4(self.filters[3] * 2, self.filters[2], num_heads=4, window_size=window_size, outd=None,
                         reduce_factor=8, dpr=dpr[2:4], depth=2, side=True)

        self.de3 = RSUT5(self.filters[2] * 2, self.filters[1], num_heads=4, window_size=window_size, outd=None,
                         reduce_factor=8, dpr=dpr[0:2], depth=2, side=True)

        self.de2 = RSU6(self.filters[1] * 2, self.filters[1] // 2, self.filters[0])

        self.de1 = RSU7(self.filters[0] * 2, self.filters[0] // 2, self.filters[0])

        # side
        self.botleside = nn.Conv3d(self.filters[3], out_channel, 3, padding=1)
        self.side5 = nn.Conv3d(self.filters[3], out_channel, 3, padding=1)
        self.side4 = nn.Conv3d(self.filters[2], out_channel, 3, padding=1)
        self.side3 = nn.Conv3d(self.filters[1], out_channel, 3, padding=1)
        self.side2 = nn.Conv3d(self.filters[0], out_channel, 3, padding=1)
        self.side1 = nn.Conv3d(self.filters[0], out_channel, 3, padding=1)
        self.outconv = nn.Conv3d(6, out_channel, 3, padding=1)

        # self.apply(_init_weights)
        #
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.apply(weights_init_kaiming)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.apply(weights_init_kaiming)

    def _upsample(self, src, tar):
        return F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)

    def forward(self, x):
        hx = x
        en1 = self.en1(x)
        en1pool = self.pool12(en1)

        en2 = self.en2(en1pool)

        x = self.patch_embed(x)  # todo ->128 dim
        H, W, T = x.size(2), x.size(3), x.size(4)

        # x = rearrange(x, 'b c h w t -> b (h w t) c', )

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(H, W, T), mode='trilinear',
                                               align_corners=True)
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        en3_out, H, W, T, x_down, Wh, Ww, Wt = self.en3(x, H, W, T)
        print(en3_out.shape, H, W, T, x_down.shape, Wh, Ww, Wt)

        en4_out, H, W, T, x_down, Wh, Ww, Wt = self.en4(x_down, Wh, Ww, Wt)
        print(en4_out.shape, H, W, T, x_down.shape, Wh, Ww, Wt)

        en5_out, H, W, T, x_down, Wh, Ww, Wt = self.en5(x_down, Wh, Ww, Wt)
        print(en5_out.shape, H, W, T, x_down.shape, Wh, Ww, Wt)

        bottle_out, H, W, T, x_down, Wh, Ww, Wt = self.bottle(x_down, Wh, Ww, Wt)
        print(bottle_out.shape, H, W, T, x_down.shape, Wh, Ww, Wt)

        de5_out, H, W, T, x_down, Wh, Ww, Wt = self.de5(
            rearrange(torch.cat((bottle_out, en5_out), dim=1), 'b c h w t-> b (h w t) c'), Wh, Ww, Wt)
        de5up = self._upsample(de5_out, en4_out)
        Wh, Ww, Wt = de5up.size(2), de5up.size(3), de5up.size(4)

        de4_out, H, W, T, x_down, Wh, Ww, Wt = self.de4(
            rearrange(torch.cat((de5up, en4_out), dim=1), 'b c h w t-> b (h w t) c'), Wh, Ww, Wt)
        de4up = self._upsample(de4_out, en3_out)
        Wh, Ww, Wt = de4up.size(2), de4up.size(3), de4up.size(4)

        de3_out, H, W, T, x_down, Wh, Ww, Wt = self.de3(
            rearrange(torch.cat((de4up, en3_out), dim=1), 'b c h w t-> b (h w t) c'), Wh, Ww, Wt)
        de3up = self._upsample(de3_out, en2)

        de2 = self.de2(torch.cat((de3up, en2), dim=1))
        de2up = self._upsample(de2, en1)

        de1 = self.de1(torch.cat((de2up, en1), dim=1))

        if self.side:
            bottleside = self.botleside(bottle_out)
            bottleside = self._upsample(bottleside, de1)

            d5 = self.side5(de5_out)
            d5 = self._upsample(d5, de1)

            d4 = self.side4(de4_out)
            d4 = self._upsample(d4, de1)

            d3 = self.side3(de3_out)
            d3 = self._upsample(d3, de1)

            d2 = self.side2(de2)
            d2 = self._upsample(d2, de1)

            d1 = self.side1(de1)
            out = self.outconv(torch.cat((d1, d2, d3, d4, d5, bottleside), dim=1))
            return out
        else:
            return de1


if __name__ == '__main__':
    SIZE = 64

    x = Variable(torch.rand(1, 1, SIZE, SIZE, SIZE)).cuda()

    # model = swinU3(side=True).cuda()
    model = swinV(1, 1, side=True).cuda()
    x = model(x)
    print(x.shape)
