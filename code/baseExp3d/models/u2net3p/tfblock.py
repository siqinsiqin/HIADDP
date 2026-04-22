# -*-coding:utf-8 -*-
"""
# Time       ：2022/11/18 18:51
# Author     ：comi
# version    ：python 3.8
# Description：
# todo 2.5D transformer
"""
import copy
import math
import torch.nn.functional as nnf
import torch
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, to_3tuple
from torch import nn
import torch.nn.functional as F
from models.model_trans.utnet.conv_trans_utils import RelativePositionBias, LinearAttention
from models.swinu2net3p.TransMorph3D import BasicLayer, PatchMerging
from models.u2net3p.cbam import CBAM
from models.u2netV.shuffleNet import channel_shuffle, ConvBNReLU


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=64, norm_layer=None):
        super(PatchEmbed, self).__init__()
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
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x


class Embed(nn.Module):
    def __init__(self, img_size=64, patch_size=2, in_chans=1, embed_dim=32, norm_layer=nn.LayerNorm, drop_rate=0.,
                 ape=True, patch_norm=True, ):
        super(Embed, self).__init__()
        self.ape = ape

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        if self.ape:
            pretrain_img_size = to_3tuple(img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.patch_embed(x)

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed)  # .flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        else:
            # x = x.flatten(2).transpose(1, 2)
            pass

        x = self.pos_drop(x)

        return x


class InvertedResidual(nn.Module):
    """
    mobile net v2
    """

    def __init__(self, inp, oup, stride, expand_ratio=4, norm_layer=nn.InstanceNorm3d):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv3d(hidden_dim, oup, 1, 1, 1, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BasicTFBlock(nn.Module):

    def __init__(self, in_chans, out_chans, trans=True, attn=False, heads=4, depths=2):
        super(BasicTFBlock, self).__init__()
        self.trans = trans
        self.attn = attn
        self.conv1 = ConvBNReLU(in_chans, out_chans, stride=1, padding=1)
        self.conv2 = ConvBNReLU(out_chans, out_chans, stride=1, padding=1)

        self.res = ConvBNReLU(in_chans, out_chans, kernel_size=1, stride=1)
        # 可扩展为多尺度
        if trans:
            num_heads = heads
            dpr = [x.item() for x in torch.linspace(0, 0.2, 2)]
            self.layers = nn.ModuleList()
            self.layers.append(BasicLayer(dim=out_chans,
                                          depth=depths,
                                          num_heads=num_heads,
                                          window_size=to_3tuple(2),
                                          mlp_ratio=4,
                                          qkv_bias=True,
                                          rpe=True,
                                          qk_scale=None,
                                          drop=0,
                                          attn_drop=0,
                                          drop_path=dpr,
                                          norm_layer=nn.LayerNorm,
                                          downsample=PatchMerging,
                                          use_checkpoint=False,
                                          pat_merg_rf=4))

        if self.attn:
            self.cbam = CBAM(out_chans, out_chans)

    def forward(self, x):
        res = self.res(x)

        convx = self.conv1(x)
        transx, cbamx = None, None
        if self.trans:
            layer = self.layers[0]
            B, C, H, W, Z = convx.size()
            transx = rearrange(convx, 'b c h w z-> b (h w z) c', h=H, w=W, z=Z)
            transx, H, W, T, x_down, Wh, Ww, Wt = layer(transx, H, W, Z)
            # transx = rearrange(x_down, 'b (h w z) c -> b c h w z', h=Wh, w=Ww, z=Wt)
            transx = rearrange(transx, 'b (h w z) c -> b c h w z', h=H, w=W, z=Z)
        if self.attn:
            cbamx = self.cbam(convx)

        if cbamx is not None:
            convx = torch.add(convx, cbamx)
        if transx is not None:
            convx = torch.add(convx, transx)

        convx = self.conv2(convx)

        return torch.add(convx, res)


if __name__ == '__main__':
    from torch.autograd import Variable

    # var = torch.rand(3, 1, 64, 64)
    # var = torch.rand(3, 1, 64, 64, 64)
    # x = Variable(var).cuda()
    # # model = OverlapPatchEmbed(64, patch_size=2, stride=1, in_chans=1, embed_dim=64).cuda()
    # # y, H, W = model(x)
    #
    # model = PatchEmbed(patch_size=2, in_chans=1, embed_dim=64).cuda()
    # y = model(x)
    # model = LinearAttention(64, heads=4, dim_head=64 // 4, attn_drop=0,
    #                         proj_drop=0, reduce_size=8, projection='interp',
    #                         rel_pos=True).cuda()
    # print(y[:, :, :, :, 0].shape)
    # y = y[:, :, :, :, 0]
    # y.to('cuda:0')
    # y, _ = model(y)
    # print('Output shape:', y.shape)

    size = 64
    var = torch.rand(3, 1, size, size, size)
    x = Variable(var).cuda()
    # model = BasicTFBlock(32, 64, ).cuda()
    model = Embed().cuda()
    y = model(x)
    print(y.shape)
