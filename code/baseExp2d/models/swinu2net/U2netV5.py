# -*-coding:utf-8 -*-
"""
# Time       ：2022/8/25 21:54
# Author     ：comi
# version    ：python 3.8
# Description：
# todo cnn 高通滤波，swin 低通滤波，结合二者，在加上U结构channel方向, Iformer
# todo 4，64，more mlp hidden
"""
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_3tuple
from torch import nn
from torch.nn.functional import interpolate
from torch.nn.init import trunc_normal_
from torch.utils import checkpoint

from models.swinu2net.transformer import SwinTransformerBlock, window_partition, PatchEmbed


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
                      groups=in_ch, bias=False, ),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),

            # 逐点卷积
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0,
                      groups=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        if self.checkpoint:
            out = checkpoint.checkpoint(self.dw, input)
        else:
            out = self.dw(input)
        return out


class expblock(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ratio=2):
        super(expblock, self).__init__()
        mid_dim = ratio * out_ch
        self.block = nn.Sequential(
            DEPTHWISECONV(in_ch, mid_dim),
            nn.Conv3d(mid_dim, mid_dim, 3, padding=1 * 2, dilation=1 * 2),
            nn.BatchNorm3d(mid_dim),
            nn.ReLU(inplace=True),
            DEPTHWISECONV(mid_dim, out_ch)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class bottleblock(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, ):
        super(bottleblock, self).__init__()
        mid_dim = out_ch // 2
        self.block = nn.Sequential(
            DEPTHWISECONV(in_ch, mid_dim, ),
            DEPTHWISECONV(mid_dim, mid_dim, ),
            DEPTHWISECONV(mid_dim, out_ch, ),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Block(nn.Module):

    def __init__(self, in_channel, num_heads=8, mlp_ratio=4., downsample=None,
                 window_size=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, depth=2):

        super(Block, self).__init__()
        self.window_size = to_3tuple(window_size)
        self.shift_size = to_3tuple(window_size // 2)
        self.use_checkpoint = use_checkpoint

        assert 0 <= min(self.shift_size) < min(
            self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}" \
            .format(self.shift_size, self.window_size)

        self.blocks = nn.ModuleList()
        for i in range(depth // 2):
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
        if downsample is not None:
            self.downsample = downsample(dim=in_channel, norm_layer=norm_layer)
        else:
            self.downsample = None

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

        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            x = rearrange(x, 'b (w h t) c -> b c w h t', h=H, w=W, t=T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            x_down = rearrange(x_down, 'b (w h t) c -> b c w h t', h=Wh, w=Ww, t=Wt)
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            x = rearrange(x, 'b (w h t) c -> b c w h t', h=H, w=W, t=T)
            return x, H, W, T, x, H, W, T


class baseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super(baseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class cbtBlock(nn.Module):
    """
    todo swin  and conv block combination
    """

    def __init__(self, in_ch, out_ch, useSwin=True):
        super(cbtBlock, self).__init__()
        self.useSwin = useSwin
        self.bottle = bottleblock(in_ch, out_ch)
        self.exp = expblock(in_ch, out_ch)
        self.conv = baseBlock(in_ch, out_ch)
        if useSwin:
            self.swinblock = Block(in_channel=in_ch, downsample=None, use_checkpoint=True)

    def forward(self, x):
        bx = self.bottle(x)
        # ex = self.exp(x)
        cx = self.conv(x)
        if self.useSwin:
            sx = self.swinblock(x)
            return bx + cx + sx[5]
        return bx + cx  # + ex


def _upsample(src, tar):
    return interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)


class swinU2NET(nn.Module):
    """
    78.41
    """

    def __init__(self, in_ch=3, out_ch=1, filter=None, side=False, sup=False, img_size=64, patch_size=4,
                 drop=0.):
        super(swinU2NET, self).__init__()
        self.side = side
        self.sup = sup

        if filter is None:
            filter = [32, 64, 128, 256, 256]  # [64, 128, 256, 512, 512]

        self.stage1 = cbtBlock(in_ch, filter[0], useSwin=False)
        self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage2 = cbtBlock(filter[0], filter[1], useSwin=False)
        self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        embed_dim = filter[2]
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_ch, embed_dim=embed_dim, norm_layer=nn.LayerNorm)

        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1],
                              img_size[2] // patch_size[2]]

        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
        trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop)

        self.stage3 = cbtBlock(filter[1], filter[2], useSwin=False)
        self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage4 = cbtBlock(filter[2], filter[3])
        self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage5 = cbtBlock(filter[3], filter[4])
        self.pool56 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage6 = cbtBlock(filter[4], filter[4])

        # decoder
        self.stage5d = cbtBlock(filter[4] * 2, filter[4])
        self.stage4d = cbtBlock(filter[4] * 2, filter[2])
        self.stage3d = cbtBlock(filter[3], filter[1])
        self.stage2d = cbtBlock(filter[2], filter[0])
        self.stage1d = cbtBlock(filter[1], filter[0])

        mid = out_ch
        self.side1 = DEPTHWISECONV(filter[0], mid)
        self.side2 = DEPTHWISECONV(filter[0], mid)
        self.side3 = DEPTHWISECONV(filter[1], mid)
        self.side4 = DEPTHWISECONV(filter[2], mid)
        self.side5 = DEPTHWISECONV(filter[3], mid)
        self.side6 = DEPTHWISECONV(filter[4], mid)

        self.outconv = nn.Conv3d(mid * 6, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        patch = self.patch_embed(x)
        H, W, T = patch.size(2), patch.size(3), patch.size(4)
        absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(H, W, T), mode='trilinear',
                                           align_corners=True)
        patch = (patch + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        patch = self.pos_drop(patch)
        patch = rearrange(patch, 'b (h w t) c -> b c h w t', h=H, w=W, t=T)

        # stage 3
        hx3 = self.stage3(hx)
        hx3 += patch
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        if self.side:
            return d0
        if self.sup:
            return [d0, d1, d2, d3, d4, d5, d6]
        return d1


class myU2NET(nn.Module):
    "77.5±2.65"

    def __init__(self, in_ch=3, out_ch=1, filter=None, side=False, sup=False, img_size=64, patch_size=4,
                 drop=0.):
        super(myU2NET, self).__init__()
        self.side = side
        self.sup = sup

        if filter is None:
            filter = [32, 64, 128, 256, 256]  # [64, 128, 256, 512, 512]

        self.stage1 = cbtBlock(in_ch, filter[0], useSwin=False)
        self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage2 = cbtBlock(filter[0], filter[1], useSwin=False)
        self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage3 = cbtBlock(filter[1], filter[2], useSwin=False)
        self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage4 = cbtBlock(filter[2], filter[3], useSwin=False)
        self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage5 = cbtBlock(filter[3], filter[4], useSwin=False)
        self.pool56 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage6 = cbtBlock(filter[4], filter[4], useSwin=False)

        # decoder
        self.stage5d = cbtBlock(filter[4] * 2, filter[4], useSwin=False)
        self.stage4d = cbtBlock(filter[4] * 2, filter[2], useSwin=False)
        self.stage3d = cbtBlock(filter[3], filter[1], useSwin=False)
        self.stage2d = cbtBlock(filter[2], filter[0], useSwin=False)
        self.stage1d = cbtBlock(filter[1], filter[0], useSwin=False)

        self.side1 = nn.Conv3d(filter[0], out_ch, 3, padding=1)
        self.side2 = nn.Conv3d(filter[0], out_ch, 3, padding=1)
        self.side3 = nn.Conv3d(filter[1], out_ch, 3, padding=1)
        self.side4 = nn.Conv3d(filter[2], out_ch, 3, padding=1)
        self.side5 = nn.Conv3d(filter[3], out_ch, 3, padding=1)
        self.side6 = nn.Conv3d(filter[4], out_ch, 3, padding=1)

        self.outconv = nn.Conv3d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample(d6, d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        if self.side:
            return d0
        if self.sup:
            return [d0, d1, d2, d3, d4, d5, d6]
        return d1


if __name__ == '__main__':
    x = torch.randn((1, 1, 64, 64, 64)).cuda()
    model = myU2NET(1, 1, side=True).cuda()
    x = model(x)
    print(x.shape)
