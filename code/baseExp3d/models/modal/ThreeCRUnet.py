# -*-coding:utf-8 -*-
"""
# Time       ：2023/4/2 20:35
# Author     ：comi
# version    ：python 3.8
# Description：
"""
from models.u2net3p.u4block import u4block
from models.u2netV.AttentionModule import AttenModule

# -*-coding:utf-8 -*-
"""
# Time       ：2023/3/28 15:23
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
from ptflops import get_model_complexity_info
from torch import nn

from models.swinu2net.U2netV5 import _upsample
from models.u2net3p.unet3pV5 import Enconder, CResBlock, Downsample, Block
from models.u2netV.U2net import RSU4, RSU5, RSU6, RSU7


# from models.u2netV.shuffleNet import RSU4, RSU5, RSU6, RSU7


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width, z = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width, z)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width, z)
    return x


class unetConv3d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=2, ks=3, stride=1, padding=1):
        super(unetConv3d, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.InstanceNorm3d(out_size),
                                     nn.ReLU(inplace=False))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):

        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class Enconder(nn.Module):
    def __init__(self, block_mode='3p', filters=[16, 32, 64, 128, 256], down_mode='pool'):
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
            # self.conv5_b = AttenModule('u', filters[4], depth=3)  # _b
            self.conv5_b = u4block(filters[3], filters[4], depth=2, dirate=2, split=False, mode='ch')

        elif block_mode == 'conv':
            self.conv1 = unetConv3d(1, filters[0], )
            self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)

            self.conv2 = unetConv3d(filters[0], filters[1], )
            self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

            self.conv3 = unetConv3d(filters[1], filters[2], )
            self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

            self.conv4 = unetConv3d(filters[2], filters[3], )
            self.maxpool4 = Downsample(stride=2, channel=filters[3], mode=down_mode)

            self.conv5 = unetConv3d(filters[3], filters[4], )
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

        # v13
        h5 = self.conv5(h5)  # 单独conv5 80.74

        return h1, h2, h3, h4, h5


class ThreeCRUNet(nn.Module):

    def __init__(self, deconder='cr'):
        super(ThreeCRUNet, self).__init__()

        filters = [32, 64, 128, 256, 256]  # 最终实现方案 通道数
        self.up = Enconder(block_mode=deconder, filters=filters)  # 替换了enconder的基本块
        self.mid = Enconder(block_mode=deconder, filters=filters)
        self.down = Enconder(block_mode=deconder, filters=filters)

        if deconder == 'u2':
            self.fusion4_up = RSU4(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3])
            self.fusion3_up = RSU5(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2])
            self.fusion2_up = RSU6(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1])
            self.fusion1_up = RSU7(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0])

            self.fusion4_mid = RSU4(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3])
            self.fusion3_mid = RSU5(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2])
            self.fusion2_mid = RSU6(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1])
            self.fusion1_mid = RSU7(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0])

            self.fusion4_down = RSU4(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3])
            self.fusion3_down = RSU5(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2])
            self.fusion2_down = RSU6(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1])
            self.fusion1_down = RSU7(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0])
        elif deconder == 'cr':
            self.fusion4_up = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                        mode=deconder)
            self.fusion3_up = CResBlock(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2],
                                        mode=deconder)
            self.fusion2_up = CResBlock(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1],
                                        mode=deconder)
            self.fusion1_up = CResBlock(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0],
                                        mode='none')

            self.fusion4_mid = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                         mode=deconder)
            self.fusion3_mid = CResBlock(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2],
                                         mode=deconder)
            self.fusion2_mid = CResBlock(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1],
                                         mode=deconder)
            self.fusion1_mid = CResBlock(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0],
                                         mode='none')

            self.fusion4_down = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                          mode=deconder)
            self.fusion3_down = CResBlock(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2],
                                          mode=deconder)
            self.fusion2_down = CResBlock(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1],
                                          mode=deconder)
            self.fusion1_down = CResBlock(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0],
                                          mode='none')
        elif deconder == 'conv':
            self.fusion4_up = unetConv3d(filters[4] + filters[3], filters[3])
            self.fusion3_up = unetConv3d(filters[3] + filters[2], filters[2])
            self.fusion2_up = unetConv3d(filters[2] + filters[1], filters[1])
            self.fusion1_up = unetConv3d(filters[1] + filters[0], filters[0])

            self.fusion4_mid = unetConv3d(filters[4] + filters[3], filters[3])
            self.fusion3_mid = unetConv3d(filters[3] + filters[2], filters[2])
            self.fusion2_mid = unetConv3d(filters[2] + filters[1], filters[1])
            self.fusion1_mid = unetConv3d(filters[1] + filters[0], filters[0])

            self.fusion4_down = unetConv3d(filters[4] + filters[3], filters[3])
            self.fusion3_down = unetConv3d(filters[3] + filters[2], filters[2])
            self.fusion2_down = unetConv3d(filters[2] + filters[1], filters[1])
            self.fusion1_down = unetConv3d(filters[1] + filters[0], filters[0])
        elif deconder == 'rs':
            self.fusion4_up = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                        mode=deconder)
            self.fusion3_up = CResBlock(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2],
                                        mode=deconder)
            self.fusion2_up = CResBlock(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1],
                                        mode=deconder)
            self.fusion1_up = CResBlock(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0],
                                        mode='none')

            self.fusion4_mid = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                         mode=deconder)
            self.fusion3_mid = CResBlock(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2],
                                         mode=deconder)
            self.fusion2_mid = CResBlock(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1],
                                         mode=deconder)
            self.fusion1_mid = CResBlock(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0],
                                         mode='none')

            self.fusion4_down = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                          mode=deconder)
            self.fusion3_down = CResBlock(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2],
                                          mode=deconder)
            self.fusion2_down = CResBlock(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1],
                                          mode=deconder)
            self.fusion1_down = CResBlock(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0],
                                          mode='none')

        # self.neck_attn = ThreeAttnFusion(768, fusion_block='dconv', depth=1)
        # self.f_attn = ThreeAttnFusion(768, fusion_block='dconv', depth=1)
        # self.t_attn = ThreeAttnFusion(384, fusion_block='dconv', depth=1)
        # self.s_attn = ThreeAttnFusion(192, fusion_block='dconv', depth=1)

        self.neck_attn = AttenModule('cbam', 768, depth=1)
        self.f_attn = AttenModule('cbam', 768, depth=1)
        self.t_attn = AttenModule('cbam', 384, depth=1)
        self.s_attn = AttenModule('cbam', 192, depth=1)

        self.out = nn.Conv3d(96, 1, kernel_size=3, padding=1)

    def forward(self, x):
        img = x[:, 0:1, :, :, :]
        reverse = x[:, 1:2, :, :, :]
        border = x[:, 2:3, :, :, :]

        hu1, hu2, hu3, hu4, hu5 = self.up(img)
        rx1, rx2, rx3, rx4, rx5 = self.mid(reverse)
        hx1, hx2, hx3, hx4, hx5 = self.down(border)

        # 第一次混合
        fusion_neck = self.neck_attn(torch.cat([hu5, rx5, hx5], dim=1))
        img_f, reverse_f, border_f = torch.split(fusion_neck, 256, dim=1)

        hd5up = _upsample(img_f, hu4)
        fusion4_up = self.fusion4_up(torch.cat((hd5up, hu4), 1))

        rd5mid = _upsample(reverse_f, rx4)
        fusion4_mid = self.fusion4_mid(torch.cat((rd5mid, rx4), 1))

        hd5down = _upsample(border_f, hx4)
        fusion4_down = self.fusion4_down(torch.cat((hd5down, hx4), 1))

        # 第二次融合
        fusion_neck = self.f_attn(torch.cat([fusion4_up, fusion4_mid, fusion4_down], dim=1))
        img_f, reverse_f, border_f = torch.split(fusion_neck, 256, dim=1)

        hd4up = _upsample(img_f, hu3)
        fusion3_up = self.fusion3_up(torch.cat((hd4up, hu3), 1))

        rd4down = _upsample(reverse_f, hx3)
        fusion3_mid = self.fusion3_mid(torch.cat((rd4down, rx3), 1))

        hd4down = _upsample(border_f, hx3)
        fusion3_down = self.fusion3_down(torch.cat((hd4down, hx3), 1))

        # 第三次融合
        fusion_neck = self.t_attn(torch.cat([fusion3_up, fusion3_mid, fusion3_down], dim=1))
        img_f, reverse_f, border_f = torch.split(fusion_neck, 128, dim=1)

        hd3up = _upsample(img_f, hu2)
        fusion2_up = self.fusion2_up(torch.cat((hd3up, hu2), 1))

        rd3down = _upsample(reverse_f, hx2)
        fusion2_mid = self.fusion2_mid(torch.cat((rd3down, rx2), 1))

        hd3down = _upsample(border_f, hx2)
        fusion2_down = self.fusion2_down(torch.cat((hd3down, hx2), 1))

        # 第四次融合
        fusion_neck = self.s_attn(torch.cat([fusion2_up, fusion2_mid, fusion2_down], dim=1))
        img_f, reverse_f, border_f = torch.split(fusion_neck, 64, dim=1)

        hd2up = _upsample(img_f, hu1)
        fusion1_up = self.fusion1_up(torch.cat((hd2up, hu1), 1))

        rd2down = _upsample(reverse_f, hx1)
        fusion1_mid = self.fusion1_mid(torch.cat((rd2down, rx1), 1))

        hd2down = _upsample(border_f, hx1)
        fusion1_down = self.fusion1_down(torch.cat((hd2down, hx1), 1))

        return self.out(torch.cat([fusion1_up, fusion1_mid, fusion1_down], dim=1))


if __name__ == '__main__':
    from torch.autograd import Variable

    var = torch.rand(3, 3, 64, 64, 64)
    x = Variable(var).cuda()
    model = ThreeCRUNet(deconder='conv').cuda()
    macs, params = get_model_complexity_info(model, (3, 64, 64, 64), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(x)
    print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
