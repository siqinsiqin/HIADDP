# -*- coding: utf-8 -*-

import torch

from torch import nn
from torch.nn.functional import max_pool3d


class LossV2(nn.Module):
    def __init__(self, loss_name):
        super(LossV2, self).__init__()
        if loss_name == 'dice':
            self.loss = dice_coef()
        elif loss_name == 'mse':
            self.loss = MSE()
        elif loss_name == 'mae':
            self.loss = MAE()
        elif loss_name == 'mix':
            self.loss = mix_loss()
        elif loss_name == 'ce':
            self.loss = crossentropy()
        elif loss_name == 'bce':
            self.loss = B_crossentropy()
        elif loss_name == 'iou':
            self.loss = IOU(size_average=True)
        elif loss_name == 'tversky':
            self.loss = TverskyLoss()
        else:
            raise NotImplementedError

    def __call__(self, gt, pred):
        return self.loss(gt, pred)


class dice_coef(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1.
        size = y_true.shape[-1] // y_pred.shape[-1]
        y_true = max_pool3d(y_true, size, size)
        a = torch.sum(y_true * y_pred, (2, 3, 4))
        b = torch.sum(y_true, (2, 3, 4))
        c = torch.sum(y_pred, (2, 3, 4))
        dice = (2 * a) / (b + c + smooth)
        return torch.mean(dice)


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class MAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))


class mix_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return crossentropy()(y_true, y_pred) + 1 - dice_coef()(y_true, y_pred)


class crossentropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth))


class B_crossentropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth) + (1 - y_true) * torch.log(1 - y_pred + smooth))


def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)


def IOU_loss(pred, label):
    iou_loss = IOU(size_average=True)
    iou_out = iou_loss(pred, label)
    print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, pred, gt, smooth=1, alpha=0.3, beta=0.7, gamma=1):
        inputs = pred.view(-1)
        targets = gt.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        loss = (1 - Tversky) ** gamma

        return loss
