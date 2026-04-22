import math

import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def make_model(args, parent=False):
    return MainNet()


class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        self.conv1 = self.conv_block(1, 32)
        # self.convt = self.conv_block(1,1)
        self.conv2 = self.conv_block(32, 64)
        self.conv3 = self.conv_block(64, 128)
        self.conv4 = self.conv_block(128, 128 * 2)
        self.conv5 = self.conv_block(128 * 2, 128 * 4)
        self.pool = torch.nn.MaxPool2d(2)
        self.upconv1 = self.upconv(64, 32)
        self.upconv2 = self.upconv(128, 64)
        self.upconv3 = self.upconv(128 * 2, 128)
        self.upconv4 = self.upconv(128 * 4, 128 * 2)
        self.conv6 = self.conv_block(128 * 4, 128 * 2)
        self.conv7 = self.conv_block(128 * 2, 128)
        self.conv8 = self.conv_block(128, 64)
        self.conv9 = self.conv_block(64, 32)
        # self.conv10 = self.conv_block(35,1)
        self.conv11 = self.conv_block(33, 1)
        self.last_act = nn.PReLU()

    def conv_block(self, channel_in, channel_out):
        if channel_in == 3:
            return nn.Sequential(
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )
        else:
            return nn.Sequential(
                nn.PReLU(),
                nn.BatchNorm2d(channel_in),
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )

    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)

    def forward(self, x):
        # x = x / 255.
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        x5 = self.pool(x4)
        x5 = self.conv5(x5)
        u4 = self.upconv4(x5)
        u4 = torch.cat([u4, x4], 1)
        u4 = self.conv6(u4)
        u3 = self.upconv3(u4)
        u3 = torch.cat([u3, x3], 1)
        u3 = self.conv7(u3)
        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, x2], 1)
        u2 = self.conv8(u2)
        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, x1], 1)
        u1 = self.conv9(u1)
        # u1 = self.last_act(u1)
        u1 = torch.cat([u1, x], 1)
        # pred = self.conv10(u1) + x
        # out_pred = F.sigmoid(self.conv11(u1))
        out_pred = torch.sigmoid(self.conv11(u1))
        # return F.sigmoid(pred)
        return out_pred


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.conv1 = self.conv_block(1, 32)
        self.conv2 = self.conv_block(32, 64)
        self.conv3 = self.conv_block(64, 128)
        self.conv4 = self.conv_block(128, 128 * 2)
        self.conv5 = self.conv_block(128 * 2, 128 * 4)
        self.pool = torch.nn.MaxPool2d(2)
        self.upconv1 = self.upconv(64, 32)
        self.upconv2 = self.upconv(128, 64)
        self.upconv3 = self.upconv(128 * 2, 128)
        self.upconv4 = self.upconv(128 * 4, 128 * 2)
        self.conv6 = self.conv_block(128 * 4, 128 * 2)
        self.conv7 = self.conv_block(128 * 2, 128)
        self.conv8 = self.conv_block(128, 64)
        self.conv9 = self.conv_block(64, 32)
        self.conv11 = self.conv_block(33, 1)
        self.last_act = nn.PReLU()

    def conv_block(self, channel_in, channel_out):
        if channel_in == 3:
            return nn.Sequential(
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )
        else:
            return nn.Sequential(
                nn.PReLU(),
                nn.BatchNorm2d(channel_in),
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )

    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in, channel_out, kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        x5 = self.pool(x4)
        x5 = self.conv5(x5)
        u4 = self.upconv4(x5)
        u4 = torch.cat([u4, x4], 1)
        u4 = self.conv6(u4)
        u3 = self.upconv3(u4)
        u3 = torch.cat([u3, x3], 1)
        u3 = self.conv7(u3)
        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, x2], 1)
        u2 = self.conv8(u2)
        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, x1], 1)
        u1 = self.conv9(u1)
        u1 = torch.cat([u1, x], 1)
        out_pred = self.conv11(u1)
        # out_pred = torch.sigmoid(self.conv11(u1))
        return out_pred

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.s1 = EnhanceNet()
        self.s2 = SegNet()

    def forward(self, x):
        x1 = self.s1(x)
        out = self.s2(x1)
        # return x1, out
        return out


if __name__ == '__main__':
    # todo Study Group Learning: Improving Retinal Vessel Segmentation Trained with Noisy Labels
    x = torch.randn((1, 1, 64, 64))
    model = MainNet()
    x = model(x)
    print(x.shape)
