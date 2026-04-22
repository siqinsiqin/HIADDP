# -*-coding:utf-8 -*-
"""
# Time       ：2023/3/28 15:23
# Author     ：comi
# version    ：python 3.8
# Description：
深度为4，通道数为32起始
"""
import torch
from ptflops import get_model_complexity_info
from torch import nn
from torch.nn.functional import interpolate

from models.swinu2net.U2netV5 import _upsample
from models.u2net3p.tfblock import BasicTFBlock
from models.u2net3p.unet3pV5 import Downsample, Block
from models.u2netV.AttentionModule import Attention, CASP, CBAM
from models.u2netV.U2net import RSU4, RSU5, RSU6, RSU7
from utils.norm import Swish, unetconv3d, CResBlock, conv3d, channel_spilt


# from models.u2netV.shuffleNet import RSU4, RSU5, RSU6, RSU7


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class MultiScaleDownsample(Attention):
    def __init__(self, in_channels):
        super(MultiScaleDownsample, self).__init__(in_channels)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool3d(2, 2)
        self.casp = CASP(in_channels)
        self.norm = nn.InstanceNorm3d(in_channels)

    def forward(self, x):
        query = self.conv1(x)
        key = self.conv2(x)
        value = self.pool(x)

        # part 1
        mix = self.att(query, key, value)

        return self.norm(mix * self.casp(value))


class Upsample(nn.Module):
    def __init__(self, channel, mode='ins'):
        super().__init__()
        self.mode = mode
        if mode != 'ins':
            self.up = nn.ConvTranspose3d(channel, channel, 2, 2, padding=0, bias=False)
            self.norm = nn.BatchNorm3d(channel)
            self.swish = Swish()

    def _upsample(self, src, tar):
        return interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)

    def forward(self, src, target=None):
        if self.mode == 'ins':
            return self._upsample(src, target)
        else:
            return self.swish(self.norm(self.up(src)))


class Downsample(nn.Module):

    def __init__(self, stride, channel=None, mode='pool'):  # pool 设置在enconder
        super(Downsample, self).__init__()
        assert stride is not None
        self.mode = mode
        if mode == 'pool':
            self.maxpool = nn.MaxPool3d(stride, stride, ceil_mode=True)  # ceil_mode 向上取整
        else:
            # self.maxpool = nn.MaxPool3d(stride, stride, ceil_mode=True)
            # # self.avgpool = nn.AvgPool3d(stride, stride)
            # self.convpool = nn.Conv3d(channel, channel, kernel_size=3, padding=1, stride=stride)
            # self.softmax = nn.Softmax(dim=-1)
            #
            # # 膨胀率2
            # self.fusion = nn.Sequential(
            #     nn.Conv3d(channel * 2, channel, kernel_size=3, padding=1, stride=1, dilation=1),
            #     nn.BatchNorm3d(channel),
            #     Swish()
            # )
            # self.swish = Swish()
            # self.sigmoid = nn.Sigmoid()
            # plan 4
            self.mutiscale = MultiScaleDownsample(channel)

    def att(self, query, key, value):

        # 计算注意力权重
        scores = torch.matmul(query.transpose(2, 3), key)
        scores = self.softmax(scores)

        # 使用注意力权重计算加权和
        weighted_sum = torch.matmul(scores, value.transpose(2, 3))
        output = weighted_sum.transpose(2, 3)

        return output

    def forward(self, x):
        if self.mode == 'pool':
            return self.maxpool(x)
        else:

            # # # plan 2
            # query = self.maxpool(x)
            # key = self.avgpool(x)
            #
            # output1 = self.att(query, key, key)
            # output2 = self.att(key, query, query)
            #
            # return (self.sigmoid(self.fusion(output1 + output2)) * query) + query

            return self.mutiscale(x)


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


        elif block_mode == 'conv':
            self.conv1 = unetconv3d(1, filters[0], )
            self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)

            self.conv2 = unetconv3d(filters[0], filters[1], )
            self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

            self.conv3 = unetconv3d(filters[1], filters[2], )
            self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

            self.conv4 = unetconv3d(filters[2], filters[3], )

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

            else:
                self.conv1 = CResBlock(inc=1, midc=filters[0] // 2, outc=filters[0],
                                       mode='none')  # cr:none,res=block_mode
                self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)

                self.conv2 = CResBlock(filters[0], filters[0] // 2, filters[1], mode=block_mode)
                self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

                self.conv3 = CResBlock(filters[1], filters[1] // 2, filters[2], mode=block_mode)
                self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

                self.conv4 = CResBlock(filters[2], filters[2] // 2, filters[3], mode=block_mode)

    def forward(self, input):
        h1 = self.conv1(input)
        h2 = self.maxpool1(h1)

        h2 = self.conv2(h2)
        h3 = self.maxpool2(h2)

        h3 = self.conv3(h3)
        h4 = self.maxpool3(h3)

        h4 = self.conv4(h4)

        return h1, h2, h3, h4


class dualCRUNetD4(nn.Module):

    def __init__(self, deconder='cr', depth=1):
        super(dualCRUNetD4, self).__init__()
        upmodel = 'ins'
        filters = [32, 64, 128, 256]  # 最终实现方案 通道数
        self.up = Enconder(block_mode=deconder, filters=filters, down_mode='pool')  # 替换了enconder的基本块
        self.down = Enconder(block_mode=deconder, filters=filters, down_mode='pool')

        if deconder == 'u2':
            self.fusion4_up = RSU4(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3])
            self.fusion3_up = RSU5(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2])
            self.fusion2_up = RSU6(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1])
            self.fusion1_up = RSU7(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0])

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

            self.fusion4_down = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                          mode=deconder)

            self.fusion3_down = CResBlock(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2],
                                          mode=deconder)

            self.fusion2_down = CResBlock(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1],
                                          mode=deconder)

            self.fusion1_down = CResBlock(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0],
                                          mode='none')
        elif deconder == 'conv':
            self.fusion3_up = unetconv3d(filters[3] + filters[2], filters[2])
            self.fusion2_up = unetconv3d(filters[2] + filters[1], filters[1])
            self.fusion1_up = unetconv3d(filters[1] + filters[0], filters[0])

            self.fusion3_down = unetconv3d(filters[3] + filters[2], filters[2])
            self.fusion2_down = unetconv3d(filters[2] + filters[1], filters[1])
            self.fusion1_down = unetconv3d(filters[1] + filters[0], filters[0])
        elif deconder == 'rs':
            self.fusion4_up = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                        mode=deconder)
            self.fusion3_up = CResBlock(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2],
                                        mode=deconder)
            self.fusion2_up = CResBlock(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1],
                                        mode=deconder)
            self.fusion1_up = CResBlock(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0],
                                        mode='none')

            self.fusion4_down = CResBlock(filters[4] + filters[3], (filters[4] + filters[3]) // 2, filters[3],
                                          mode=deconder)
            self.fusion3_down = CResBlock(filters[3] + filters[2], (filters[3] + filters[2]) // 2, filters[2],
                                          mode=deconder)
            self.fusion2_down = CResBlock(filters[2] + filters[1], (filters[2] + filters[1]) // 2, filters[1],
                                          mode=deconder)
            self.fusion1_down = CResBlock(filters[1] + filters[0], (filters[1] + filters[0]) // 2, filters[0],
                                          mode='none')

        self.up3_up = Upsample(filters[3], mode=upmodel)
        self.up2_up = Upsample(filters[2], mode=upmodel)
        self.up1_up = Upsample(filters[1], mode=upmodel)

        self.up3_down = Upsample(filters[3], mode=upmodel)
        self.up2_down = Upsample(filters[2], mode=upmodel)
        self.up1_down = Upsample(filters[1], mode=upmodel)

        self.res3 = conv3d(filters[3] * 2, filters[2], n=1, ks=1, padding=0)
        self.res2 = conv3d(filters[2] * 2, filters[1], n=1, ks=1, padding=0)
        self.res1 = conv3d(filters[1] * 2, filters[0], n=1, ks=1, padding=0)

        self.f_attn = CBAM(512)  # CASPv6(512, depth)
        self.t_attn = CBAM(256)  # CASPv6(256, depth)
        self.s_attn = CBAM(128)  # CASPv6(128, depth)

        # self.neck_attn = CASPv6(512, depth)
        # self.f_attn = CASPv6(512, depth)
        # self.t_attn = CASPv6(256, depth)
        # self.s_attn = CASPv6(128, depth)

        # self.neck_attn = SPv6(512, depth)
        # self.f_attn = SPv6(512, depth)
        # self.t_attn = SPv6(256, depth)
        # self.s_attn = SPv6(128, depth)
        #
        # self.neck_attn = CAv6(512, depth)
        # self.f_attn = CAv6(512, depth)
        # self.t_attn = CAv6(256, depth)
        # self.s_attn = CAv6(128, depth)


        self.out = nn.Conv3d(64, 1, kernel_size=3, padding=1)

    def fusion(self, fx1, fx2, target_1, targer_2, block_1, block_2, res_block, _upsample_1=_upsample,
               _upsample_2=_upsample):
        # todo：channel shfulle
        # target_1, targer_2 = channel_spilt(channel_shuffle(torch.cat((target_1, targer_2), dim=1), 8))

        hdup = _upsample_1(fx1, target_1)
        fusion_up = block_1(torch.cat((hdup, target_1), dim=1))

        hddown = _upsample_2(fx2, targer_2)
        fusion_down = block_2(torch.cat((hddown, targer_2), dim=1))

        res = res_block(torch.cat((hdup, hddown), dim=1))

        return fusion_up + res, fusion_down + res

    def forward(self, x):
        img = x[:, 0:1, :, :, :]
        border = x[:, 1:2, :, :, :]

        hu1, hu2, hu3, hu4 = self.up(img)
        hx1, hx2, hx3, hx4 = self.down(border)

        # 第二次融合
        # attnu4, attnx4 = self.f_attn(hu4, hx4)
        attnu4, attnx4 = channel_spilt(self.f_attn(torch.cat((hu4, hx4), dim=1)))
        fu3, fx3 = self.fusion(attnu4, attnx4, hu3, hx3, self.fusion3_up, self.fusion3_down, self.res3, self.up3_up,
                               self.up3_down)
        # 第三次融合
        # attnu3, attnx3 = self.t_attn(fu3, fx3)
        attnu3, attnx3 = channel_spilt(self.t_attn(torch.cat((fu3, fx3), dim=1)))
        fu2, fx2 = self.fusion(attnu3, attnx3, hu2, hx2, self.fusion2_up, self.fusion2_down, self.res2, self.up2_up,
                               self.up2_down)

        # 第四次融合
        # attnu2, attnx2 = self.s_attn(fu2, fx2)
        attnu2, attnx2 = channel_spilt(self.s_attn(torch.cat((fu2, fx2), dim=1)))
        fu1, fx1 = self.fusion(attnu2, attnx2, hu1, hx1, self.fusion1_up, self.fusion1_down, self.res1, self.up1_up,
                               self.up1_down)

        return self.out(torch.cat((fu1, fx1), dim=1))


if __name__ == '__main__':
    from torch.autograd import Variable

    var = torch.rand(1, 2, 64, 64, 64)
    x = Variable(var).cuda()
    model = dualCRUNetD4(deconder='conv').cuda()
    # model = Downsample(mode='att', stride=2, channel=32).cuda()
    # model = Upsample(channel=32, mode='conv').cuda()
    # model = Upsample(32, 'up').cuda()
    # model = ChannelAttention(32, x.size()[0][0]).cuda()

    macs, params = get_model_complexity_info(model, (2, 64, 64, 64), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(x)
    print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    #
    # model = dualCRUNet(deconder='conv')
    # for name, param in model.named_parameters():
    #
    #     if not param.requires_grad:
    #         print(name)
