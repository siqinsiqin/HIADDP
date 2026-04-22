# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

import os
from collections import OrderedDict
from functools import partial
from glob import glob
from random import random, seed, getstate, setstate, choice

import cv2
import numpy as np
import pandas
import torch
import torchvision.utils
from batchgenerators.augmentations.utils import resize_segmentation
from matplotlib import pyplot as plt
from matplotlib.image import imsave
from ptflops import get_model_complexity_info
from scipy.ndimage import shift, rotate
from scipy.ndimage import zoom
from skimage.transform import resize
from volumentations import Compose, RandomRotate90, Rotate, Flip

from configs import config
from models.modal.DualPathDenseNet import DualSingleDenseNet
from models.modal.dualCRUNet import dualCRUNetv1
from models.modal.dualCRUNetD2 import dualCRUNetD2
from models.modal.dualCRUNetv2 import dualCRUNet
from models.modal.dualCRUNetv22d import dualCRUNet2d
from models.modal.dualCRUNetv3 import dualCRUNetD4
from models.modal.dualCRUNetv4 import dualCRUNetD3
from models.model_trans.medt import MedT
from models.model_trans.swin.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from models.model_trans.transbts.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from models.model_trans.transfuse.TransFuse import TransFuse_S
from models.model_trans.uctrans.UCTRANSNET import UCTransNet
from models.model_trans.uctrans.uctransnetconfig import get_CTranS_config
from models.model_trans.unetr import UNETR
from models.model_trans.utnet.utnet import UTNet
from models.models_2d.mipt.bionet import BiONet
from models.models_2d.mipt.cenet.cenet import CE_Net
from models.models_2d.mipt.chdseg.unet_GloRe import UNet_GloRe
from models.models_2d.mipt.cpfnet import CPFNet
from models.models_2d.mipt.kiunet import densekiunet
from models.models_2d.mipt.msnet.msnet import MSNet
from models.models_2d.mipt.raunet import RAUNet
from models.models_2d.mipt.sgl import MainNet
from models.models_2d.mipt.sgunet import SGU_Net
from models.models_2d.mipt.unet3p import UNet_3Plus
from models.models_2d.mipt.unetpp import UNet_Nested2d
from models.models_2d.mipt.unext import UNext
from models.models_2d.others.myunet import UNET
from models.models_2d.others.unet2 import U_Net
from models.models_3d.mipt.ASA.ASA import MEDIUMVIT
from models.models_3d.mipt.FedCRLD import mainNetwork
from models.models_3d.mipt.PCAMNet import PCAMNet
from models.models_3d.mipt.VTNET.VTUNET import VTUNet
from models.models_3d.mipt.kiunet3d import kiunet3d
from models.models_3d.mipt.reconnet import ReconNet
from models.models_3d.mipt.ucaps.ucaps import UCaps3D
from models.models_3d.mipt.unet_nested.networks.UNet_Nested import UNet_Nested
from models.models_3d.mipt.vnet import VNet
from models.models_3d.mipt.wingsnet import WingsNet
from models.models_3d.mipt.ynet3d import YNet3D
from models.models_3d.others.resunet.model import ResidualUNet3D, UNet3d
from models.swinu2net.U2netV5 import swinU2NET, myU2NET
from models.swinu2net3p.modelV2 import swinu2net3plus, depu2net3plus, swin3plus
from models.swinu2net3p.swinU3 import swinV
from models.swinu2net3p.swinU3V3 import swinU
from models.u2net3p.u2net3pV6 import U2net3p5V6
from models.u2netV.U2net import U2NET
from models.u2netV.shuffleNet import shuffleU2net
from models.u2pcsnet.u2pcsnet import U2PCSNet
from utils.logger import logs

# stage = config.stage
# if stage == 7:
#     from models.u2net3p.unet3pV5 import UNet3PlusV5
# elif stage == 6:
#     from models.u2netV.u2net6 import UNet3PlusV5
# elif stage == 5:
#     from models.u2netV.u2net5 import UNet3PlusV5

pth_path = config.pth_luna_path
pred_path = config.pred_path


def getAllAttrs(evaluate=False):
    attrs = dict()
    subtlety = ['ExtremelySubtle', 'ModeratelySubtle', 'FairlySubtle', 'ModeratelyObvious', 'Obvious']  #
    internalStructure = ['SoftTissue', 'Fluid', 'Fat', 'Air']
    calcification = ['Popcorn', 'Laminated', 'cSolid', 'Noncentral', 'Central', 'Absent']
    sphericity = ['Linear', 'OvoidLinear', 'Ovoid', 'OvoidRound', 'Round']
    margin = ['PoorlyDefined', 'NearPoorlyDefined', 'MediumMargin', 'NearSharp', 'Sharp']
    lobulation = ['NoLobulation', 'NearlyNoLobulation', 'MediumLobulation', 'NearMarkedLobulation',
                  'MarkedLobulation']

    spiculation = ['NoSpiculation', 'NearlyNoSpiculation', 'MediumSpiculation', 'NearMarkedSpiculation',
                   'MarkedSpiculation']

    texture = ['NonSolidGGO', 'NonSolidMixed', 'PartSolidMixed', 'SolidMixed', 'tSolid']
    maligancy = ['benign', 'uncertain', 'malignant']

    if evaluate:
        subtlety = ['ExtremelySubtle', 'ModeratelySubtle']
        attrs.update({'subtlety': subtlety})
        #
        # # attrs.update({'internalStructure': internalStructure})  # 不评估该属性  x
        # # attrs.update({'calcification': calcification})  # x6
        # # 'Linear',
        # sphericity = ['Linear', 'OvoidLinear', 'Ovoid', 'OvoidRound', 'Round']  # luna Linear 不存在, 'Linear',
        # sphericity = ['OvoidLinear', ]
        # attrs.update({'sphericity': sphericity})
        # margin = ['PoorlyDefined', ]
        # attrs.update({'margin': margin})
        # attrs.update({'lobulation': lobulation})
        # attrs.update({'spiculation': spiculation})
        #
        # # attrs.update({'texture': texture})
        # maligancy = ['benign', 'uncertain', ]
        # attrs.update({'maligancy': maligancy})
        # size = ['sub36', 'sub6p', 'solid36', 'solid68', 'solid8p']  # 投票时手动指定
        # size = ['sub36', 'sub6p', 'solid36']  # 投票时手动指定
        # attrs.update({'size': size})
    else:
        attrs = [subtlety, internalStructure, calcification, sphericity, margin, lobulation,
                 spiculation, texture]

    return attrs


def get_gate(model):
    try:
        gate = model.gate.cpu().detach().numpy()
    except Exception as e:
        gate = 0.5
        logs(e.args)
    if gate > 0.5:
        gate = 0.5
    return gate


def caculateDiameter(img, msk):
    """
    input:1x64x64x64
    description: 通过最小外接球或最小外接矩阵估算肺结节的最大直径
    """
    msk = msk[0]

    msk[msk >= 0.5] = 1
    msk[msk < 0.5] = 0

    # 将掩码转换为 CV_8UC1 格式的图像
    mask_cv8uc1 = (msk * 255).astype(np.uint8)
    max_diameter = 0
    for i in range(mask_cv8uc1.shape[0]):
        try:
            contours, _ = cv2.findContours(mask_cv8uc1[:, :, i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(contours[0])  # 最小外接矩阵求肺结节直径
            # 提取矩形的尺寸
            width, height = rect[1]

            # 计算直径
            diameter = max(width, height)

            # (x, y), radius = cv2.minEnclosingCircle(contours[0])  # 最小外接球体求肺结节直径
            # # 计算直径
            # diameter = 2 * radius
            # print('circle:', diameter)
            # imshows(img[0], mask_cv8uc1, k=i)

        except Exception as e:
            continue

        if diameter > max_diameter:
            max_diameter = diameter

    return np.round(max_diameter, 2)


def get_set(k, lesion_list):
    set_len = len(lesion_list)
    copies = int(set_len * config.val_domain)

    val_sidx = (k - 1) * copies
    val_eidx = val_sidx + copies
    if k == 5:
        val_eidx = max(val_eidx, set_len)
    # todo 得到验证集
    val_set = lesion_list[val_sidx:val_eidx]

    train_set = []
    train_set.extend(lesion_list[:val_sidx])
    train_set.extend(lesion_list[val_eidx:])

    return [train_set, val_set]


def set_init(k, seg_path, re, lists, format='*.npy', all=False):
    if re is not None:
        lesion_list = glob(seg_path + re)
    else:
        # lesion_list = glob(seg_path + '*.nii.gz')
        lesion_list = glob(seg_path + format)
    print(len(glob(seg_path + format)))
    lesion_list.sort()
    lesion_list = [item for item in lesion_list if 'sub3c' not in item]
    lesion_list = [item for item in lesion_list if 'solid3c' not in item]
    # 添加验证集
    # train_val = lesion_list[:-len(lesion_list) // 6]
    # test_list = lesion_list[len(train_val):]
    #
    if len(lesion_list) != 0 and not all:
        set_list = get_set(k, lesion_list)
        for i in range(len(set_list)):
            lists[i].extend(set_list[i])
        # lists[-1].extend(test_list)  # 添加测试集
        return lists
    else:
        return lesion_list


def seed_torch(sd=24, original_torch_rng_state=None, original_numpy_rng_state=None, original_gpu_seed=None,
               original_random_state=None):
    #
    if sd:
        # 获取当前状态
        original_torch_rng_state = torch.get_rng_state()
        original_numpy_rng_state = np.random.get_state()
        original_gpu_seed = torch.cuda.initial_seed()
        original_random_state = getstate()
        # fix
        seed(sd)
        os.environ['PYTHONHASHSEED'] = str(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        torch.cuda.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return original_torch_rng_state, original_numpy_rng_state, original_gpu_seed, original_random_state
    else:
        setstate(original_random_state)
        os.environ['PYTHONHASHSEED'] = ''
        torch.set_rng_state(original_torch_rng_state)
        np.random.set_state(original_numpy_rng_state)
        torch.cuda.manual_seed(original_gpu_seed)
        torch.cuda.manual_seed_all(original_gpu_seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return None


def imshows(img, msk=None, k=31, save=False, savename='new_png'):
    """
    input size：64x64x64
    """
    if msk is not None:
        # 创建多个子图，每行显示一个影像
        fig, axs = plt.subplots(1, 2, figsize=(5 * 2, 5))

        axs[0].imshow(img[:, :, k], cmap=plt.cm.gray, alpha=1, interpolation='sinc')
        axs[0].axis('off')

        axs[1].imshow(msk[:, :, k] * img[:, :, k], cmap=plt.cm.gray, alpha=1, interpolation='sinc')
        axs[1].axis('off')  #
        if save:
            print('save png file')
            imsave('img.png', img[:, :, k], cmap=plt.cm.gray)
            imsave('msk.png', msk[:, :, k], cmap=plt.cm.gray)

        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        # 创建多个子图，每行显示一个影像
        fig, axs = plt.subplots(figsize=(5, 5))

        axs.imshow(img[:, :, k], cmap=plt.cm.gray, alpha=1, interpolation='sinc')
        if save:
            print(f'save new png file:{savename}.png')
            imsave(f'{savename}.png', img[:, :, k], cmap=plt.cm.gray)

        axs.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()


def z_score_norm(img):
    # z-score
    img_mean = np.mean(img)  # torch
    img_std = np.std(img)
    norm_img = (img - img_mean) / img_std
    return norm_img


def cut_norm(img):
    """
    师兄实现
    """
    # 截断归一化
    lungwin = np.array([-1000., 400.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg


def save_tmp(path, _img, _msk, _pred, name):
    # _img = lumTrans(_img)
    # tm = torch.stack([_img, _msk, _pred], dim=0)
    # torchvision.utils1.save_image(tm, path + '/_' + name + '.png')
    _img = _img.cpu().numpy()
    _msk = _msk.cpu().numpy()
    _pred = _pred.cpu().numpy()

    fig, plots = plt.subplots(1, 3)
    plots[0].imshow(_img[0], cmap='gray')
    plots[1].imshow(_msk[0], cmap='gray')
    plots[2].imshow(_pred[0], cmap='gray')
    plots[0].axis('off')
    plots[1].axis('off')
    plots[2].axis('off')
    plt.tight_layout()
    plt.savefig(path + '/' + name + '.png', pad_inches=0)
    plt.close()


def transforms2d(img, label, flip=False):
    if flip:
        rn = random.random()
        if rn < 0.35:  # 水平翻转
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=1)
        elif rn < 0.75:  # 垂直
            img = np.flip(img, axis=0)
            label = np.flip(label, axis=0)

    return img, label


def flips(image, label, axis):
    return np.flip(image, axis=axis).copy(), np.flip(label, axis=axis).copy()


def get_rotate():
    return Compose([
        RandomRotate90((0, 1), p=0.3),
        Rotate((0, 0), (0, 0), (-45, 45), p=0.2),
        Flip(1, p=0.5),
        Flip(0, p=0.5),
    ], p=1.0)


def transforms3d(img, label):
    val = choice([1, 2, 3, 4])
    if val == 1 or val == 2:
        # img, label = flips(img, label, random.choice([1, 2]))  # x，y翻转

        data = {'image': img[0], 'mask': label[0]}
        aug = get_rotate()
        aug_data = aug(**data)
        img, msk = aug_data['image'], aug_data['mask']
        img = img[np.newaxis, ...]
        label = msk[np.newaxis, ...]
    # elif val == 2:
    #     # img, label, angle_z = random_rotation_3d(img, label)  # z 轴旋转
    #     # with mask

    elif val == 3:
        img, label = translate_image(img[0], label[0])  # 平移

    return img, label


def random_rotation_3d(img, label, angle_z=None):
    """
    随机旋转3D体积数据
    :param volume: 输入的3D体积数据，形状为(1, depth, height, width)
    :return: 旋转后的3D体积数据
    """
    if angle_z is None:
        angle_z = random.randint(5, 175)
        angle_z = random.choice([-1, 1]) * random.choice([0, 90, 180, 360])
        angle_z = 45

    # # 旋转3D体积
    # # cval = np.median(img)
    print(angle_z)
    axes = (0, 1)
    rotated_img = rotate(img[0], angle_z, axes=axes, reshape=False,
                         order=4, cval=0)

    rotated_label = rotate(label[0], angle_z, axes=axes, reshape=False,
                           order=4, cval=0)

    rotated_label[rotated_label < 0.5] = 0
    rotated_label[rotated_label >= 0.5] = 1

    return rotated_img[np.newaxis, ...], rotated_label[np.newaxis, ...], angle_z


def translate_image(image, msk):
    """
    平移图像
    :param image: 输入的图像
    :param shift_amount: 平移的距离，一个包含(x, y, z)的元组
    :return: 平移后的图像
    """
    pixel = 8
    val = choice([1, 2, 3])
    if val == 1:
        random_shift_x = np.random.randint(-pixel, pixel)
        random_shift_y = 0
    elif val == 2:
        random_shift_x = 0
        random_shift_y = np.random.randint(-pixel, pixel)
    else:
        random_shift_x = np.random.randint(-pixel, pixel)
        random_shift_y = np.random.randint(-pixel, pixel)

    # Z轴上的平移设为0（不进行Z轴平移）
    shift_amount = (random_shift_x, random_shift_y, 0)

    image = shift(image, shift_amount, mode='nearest', cval=0)
    msk = shift(msk, shift_amount, mode='nearest', cval=0)

    msk[msk < 0.5] = 0
    msk[msk >= 0.5] = 1

    return image[np.newaxis, ...], msk[np.newaxis, ...]


# todo 截断归一化，只关注兴趣区域
def lumTrans(img, left=-1000., right=400.):
    lungwin = np.array([left, right])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def avgStd(xs, log=False):
    import numpy as np
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        s = "nan"
    elif arr.size == 1:
        s = f"{arr.mean():.2f}"             # 只有一个值，就不加 ±
    else:
        s = f"{arr.mean():.2f}±{arr.std(ddof=1):.2f}"
    return s



def showTime(fold, start, end):
    times = round(end - start, 2)
    hours = round(times / 3600, 2)
    days = round(times / (3600 * 24), 2)
    if fold not in [1, 2, 3, 4, 5]:
        logs(f"Fold {fold}, time: {hours:.2f} hours, {days:.2f} days")
    else:
        logs(f"{fold}, time: {hours:.2f} hours, {days:.2f} days")

    return times


# # 重采样：原始CT分辨率往往不一致，为便于应用网络，需要统一分辨率
def resample3d(imgs, spacing, new_spacing=[1., 1., 1.], order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)  #
        true_spacing = imgs.shape * spacing / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, order=order, mode='nearest')  # mode='nearest', 缩放img，resize_factor 为缩放系数
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample3d(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


# # 重采样：原始CT分辨率往往不一致，为便于应用网络，需要统一分辨率
def resample2d(imgs, spacing, new_spacing=[1., 1.], is_seg=False):
    # if len(imgs.shape) == 2:
    #     new_shape = np.round(((np.array(spacing) / np.array(new_spacing)).astype(float) * imgs.shape)).astype(int)
    #     true_spacing = imgs.shape * spacing / new_shape
    #     resize_factor = new_shape / imgs.shape
    #     imgs = zoom(imgs, resize_factor, order=order, mode='nearest')  # mode='nearest', 缩放img，resize_factor 为缩放系数
    #     return imgs, true_spacing
    # else:
    #     raise ValueError('wrong shape')
    new_shape = np.round(((np.array(spacing) / np.array(new_spacing)).astype(float) * imgs.shape)).astype(int)
    if is_seg:
        order = 0
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
        imgs = resize_fn(imgs, new_shape, order)
    else:
        order = 3
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
        imgs = resize_fn(imgs, new_shape, order, **kwargs)

    return imgs


def fliter(imgs, masks, verbose=False):
    # 过滤 黑色背景
    ti = np.zeros_like(imgs)
    tm = np.zeros_like(masks)
    t = 0
    for i in range(imgs.shape[2]):
        a = imgs[:, :, i]
        if np.count_nonzero(a) != 0:
            ti[:, :, t] = imgs[:, :, i]
            tm[:, :, t] = masks[:, :, i]
            t += 1
    imgs = ti[:, :, :t]
    masks = tm[:, :, :t]
    if verbose:
        print(f'now {imgs.shape, masks.shape}')
    return imgs, masks


def img_crop_or_fill(img, mode='2d'):
    """
    避免出现填充的情况，即结节窗口取得足够大，有足够的窗口完成裁剪
    """
    if mode == '2d':
        size = config.input_size_2d
        if img.shape[0] > size and img.shape[1] > size:
            img = center_crop(img, size, size)
        else:
            img = fill(img, size)
    elif mode == '3d':
        size = config.input_size_3d
        if img.shape[0] > size and img.shape[1] > size and img.shape[2] > size:
            img = center_crop(img, size, size, size)
        else:
            print('Invalid mode')
            assert False
            img = fill(img, size)

    return img


def gapCal(img_size, size):
    gap = 0
    if img_size > size:
        gap = int(np.ceil((img_size - size) / 2))
    return gap


def fill(img, size):
    if len(img.shape) == 2:
        arr = np.zeros([size, size])
        gapx = gapCal(img.shape[0], size)
        gapy = gapCal(img.shape[1], size)
        img = img[gapx:img.shape[0] - gapx, gapy:img.shape[1] - gapy]
        # arr[:img.shape[0], :img.shape[1]] = img  # 边缘填充
        # 居中填充
        x_start = (size - img.shape[0]) // 2
        x_end = x_start + img.shape[0]
        y_start = (size - img.shape[1]) // 2
        y_end = y_start + img.shape[1]
        arr[x_start:x_end, y_start:y_end] = img

    elif len(img.shape) == 3:
        arr = np.zeros([size, size, size])
        gapx = gapCal(img.shape[0], size)
        gapy = gapCal(img.shape[1], size)
        gapz = gapCal(img.shape[2], size)
        img = img[gapx:img.shape[0] - gapx, gapy:img.shape[1] - gapy, gapz:img.shape[2] - gapz]
        # arr[:img.shape[0], :img.shape[1], :img.shape[2]] = img # 边缘填充

        # 居中填充
        x_start = (size - img.shape[0]) // 2
        x_end = x_start + img.shape[0]
        y_start = (size - img.shape[1]) // 2
        y_end = y_start + img.shape[1]
        z_start = (size - img.shape[2]) // 2
        z_end = z_start + img.shape[2]
        arr[x_start:x_end, y_start:y_end, z_start:z_end] = img
    else:
        raise RuntimeError('wrong shape')
    return arr


def img_to_reverse(img):
    min_value = img.min()
    max_value = img.max()
    scaled_img = (img - min_value) / (max_value - min_value) * 255

    # 计算亮度值的相反数并更新像素值
    inverted_img = 255 - scaled_img

    # 还原像素范围
    restored_img = inverted_img / 255 * (max_value - min_value) + min_value
    return restored_img


import SimpleITK as sitk


# def sobel_filter(image):
#     image = sitk.GetImageFromArray(image)
#     # image = Gaussian(image)
#     # 创建Sobel算子的滤波器
#     sobel_filter = sitk.SobelEdgeDetectionImageFilter()
#
#     # 应用Sobel算子滤波器
#     sobel_image = sobel_filter.Execute(image)
#
#     arr = sitk.GetArrayFromImage(sobel_image)
#     return arr[np.newaxis, ...]
def sobel_filter(image):
    """
    cv2更加清晰
    """
    # Apply Sobel filter to all 2D slices at once
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    filtered_volume = np.expand_dims(sobel, axis=0)

    return filtered_volume


def canny_filter(image):
    """
    自适应canny算子
    """
    imgs = []

    for i in range(image.shape[2]):
        # 确保像素值的数据类型为CV_8U
        gray_image = cv2.convertScaleAbs(image[:, :, i])
        adaptive_threshold = cv2.adaptiveThreshold(gray_image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 0.05)

        # # 创建CLAHE对象并应用对比度有限自适应直方图均衡化
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # clahe_image = clahe.apply(gray_image)

        imgs.append(adaptive_threshold)

    return np.stack(imgs, axis=2)[np.newaxis, ...]


def gradint_filter(image):
    image = sitk.GetImageFromArray(image)

    scharr_filter = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    scharr_filter.SetSigma(1.0)
    # 对图像应用Scharr边缘检测
    edge_image = scharr_filter.Execute(image)

    arr = sitk.GetArrayFromImage(edge_image)
    return arr[np.newaxis, ...]


def get_counter(msk):
    msk = sitk.GetImageFromArray(msk)
    msk = sitk.Cast(msk, sitk.sitkUInt8)
    label_contour = sitk.LabelContourImageFilter()
    label_contour.SetBackgroundValue(0)
    contour_img = label_contour.Execute(msk)
    contour_arr = sitk.GetArrayFromImage(contour_img)
    return contour_arr[np.newaxis, ...]


def resample3d(image, mask):
    # 原始输入尺寸
    old_shape = image.shape

    # 目标输出尺寸

    # new_shape = (64, 64, 64)
    new_shape = (128, 128, 16)
    # 计算缩放因子
    zoom_factor = tuple(np.array(new_shape) / np.array(old_shape))

    # 使用scipy的zoom函数进行重采样
    image_resampled = zoom(image, zoom_factor, order=1)
    mask_resampled = zoom(mask, zoom_factor, order=0)

    return image_resampled[np.newaxis, ...], mask_resampled[np.newaxis, ...]


def center_crop(img, new_width=None, new_height=None, new_z=None):
    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        z = img.shape[2]
        up_z = int(np.ceil((z - new_z) / 2))
        floor_z = z - int(np.floor((z - new_z) / 2))
        center_cropped_img = img[top:bottom, left:right, up_z:floor_z]

    return center_cropped_img


def load_model_k_checkpoint(pthPath, mode, model_name, optimizer, loss_name, model, k, verbose=True):
    if verbose:
        logs(f'============load {model_name} == Fold {k} check point============')
    file = os.path.join(pthPath, f'{mode}_{model_name}_{str(k)}_{optimizer}_{loss_name}_checkpoint.pth')
    print(file)
    if not os.path.exists(file):
        logs('pth not exist')
        exit(0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        check_point = torch.load(file, map_location=device)
        # print(device)
        # 检查当前模型和state_dict的键
        print('load keys')  # 当前模型的键
        # print(check_point.keys())  # 加载的state_dict的键

        try:
            model.load_state_dict(check_point)
        except Exception as e:
            # print(e.args)
            model.load_state_dict(check_point, strict=True)
            print('pth与model不一致！')
            raise ValueError


def save_predictions_as_imgs(loader, model, folder=pred_path, device='cuda:0', verbose=True):
    if verbose:
        logs('============save prediction============')
    os.makedirs(pred_path, exist_ok=True)
    model.eval()
    for idx, data in enumerate(loader):
        name, img, msk = data['name'], data['image'], data['mask']

        img = img.type(torch.FloatTensor)
        msk = msk.type(torch.FloatTensor)
        img = img.to(device)
        msk = msk.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(img))  # 判断是否进行了softmax 或者sigmoid
            preds = (preds > 0.5).float()
        logs(preds.shape)
        # res = torch.cat((img, msk, preds), dim=0)
        res = preds
        torchvision.utils.save_image(res, f"{folder}/predict_{name[0]}.png")
    model.train()


def get_parm(model='2d', model_name='None', verbose=False):
    from torch.autograd import Variable
    model = model.lower()
    print(model_name)
    if model == '2d':
        SIZE = config.input_size_2d
        x = Variable(torch.rand(8, 1, SIZE, SIZE)).cuda()
        model = get_model2d(model_name, config.device)
        macs, params = get_model_complexity_info(model, (1, SIZE, SIZE), as_strings=True,
                                                 print_per_layer_stat=verbose, verbose=verbose)
    else:
        SIZE = config.input_size_3d
        x = Variable(torch.rand(8, 1, SIZE, SIZE, SIZE)).cuda()
        model = get_model3d(model_name, config.device)
        macs, params = get_model_complexity_info(model, (1, SIZE, SIZE, SIZE), as_strings=True,
                                                 print_per_layer_stat=verbose, verbose=verbose)

    # if verbose:
    #     y = model(x)
    #     print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params


def findmax(imgs, masks):
    # todo 找出2d mask中结节最大的一个
    imgs = np.array(imgs)
    masks = np.array(masks)

    cnt = []
    for t in range(imgs.shape[-1]):
        n = np.count_nonzero(masks[:, :, t])
        cnt.append(n)

    a = np.array(cnt)

    return np.where(a == np.max(a))[0][0]


def checkThickSlice(seriesuid):
    """
    判断是否是厚层CT
    """
    df = pandas.read_csv('/zsm/jwj/baseExpV7/LIDCXML/annos/all_device_thick_info.csv')
    row_data = df[df['seriesuid'] == seriesuid].iloc[0]

    if row_data['slice_thickness'] <= 2.5:
        return True
    else:
        return False


import torch.nn as nn


def weights_init(m, xohnorm='he'):
    classname = m.__class__.__name__
    try:
        if classname.find('Conv') != -1:
            if xohnorm == 'x':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    except Exception as e:
        print(e.args)


def get_not_none_img(imgs, masks, range_num=None):
    """
    依据msk掩码获取所有含有结节的切片,并返回指定range_num的img
    """

    idx = findmax(imgs, masks)
    if idx + (range_num // 2) > imgs.shape[-1]:
        imgs = imgs[:, :, :, idx - (range_num - (imgs.shape[-1] - idx)):]
        masks = masks[:, :, :, idx - (range_num - (imgs.shape[-1] - idx)):]
    else:
        imgs = imgs[:, :, :, idx - (range_num // 2):idx + (range_num // 2)]
        masks = masks[:, :, :, idx - (range_num // 2):idx + (range_num // 2)]

    assert imgs.shape[-1] == masks.shape[-1] == range_num, 'shape error'
    return imgs, masks


def get_model2d(model_name, device, verbose=False):
    if model_name == 'unet':#dui
        # model = UNet(1, 1)
        model = U_Net(1, 1)
        # model = UNet2D(1, 1)
    elif model_name == 'myunet':
        model = UNET(1, 1)
    elif model_name == 'unetpp':#dui

        # model = smp.UnetPlusPlus(classes=1, in_channels=1)
        model = UNet_Nested2d(1, 1)
        # model = UNet_Nested(1, 1)
    elif model_name == 'unet3p':#dui
        model = UNet_3Plus()
    elif model_name == 'cpfnet':#dui
        model = CPFNet()
    elif model_name == 'raunet':#dui
        model = RAUNet()
    elif model_name == 'bionet':#dui
        model = BiONet()
    elif model_name == 'sgunet':#dui
        model = SGU_Net(1, 1)
    elif model_name == 'kiunet':#dui
        # model = kiunet()
        model = densekiunet()
    elif model_name == 'sgl':#dui
        model = MainNet()
    elif model_name == 'chdseg':#dui
        model = UNet_GloRe(1, 1)
    elif model_name == 'uctransnet':#dui
        model = UCTransNet(get_CTranS_config())
    elif model_name == 'medt':#dui
        model = MedT()  # todo 修改img size
        # model = logo()  # todo 修改img size
        # model = gated()  # todo 修改img size
    elif model_name == 'utnet':#dui
        # model = UTNet(in_chan=1, base_chan=32, block_list='1234', num_blocks=[1, 1, 1, 1], num_heads=[4, 4, 4, 4],
        #               attn_drop=0.1, proj_drop=0.1, reduce_size=8)  # 256 todo 修改reduce size
        model = UTNet(in_chan=1, base_chan=32, num_classes=1, reduce_size=4, block_list='234', num_blocks=[1, 2, 4],
                      projection='interp', num_heads=[2, 4, 8], attn_drop=0., proj_drop=0., bottleneck=False,
                      maxpool=True, rel_pos=True, aux_loss=False)
    elif model_name == 'swinunet':#dui
        model = SwinTransformerSys(in_chans=1, num_classes=1, img_size=64,
                                   window_size=8)  # size 224,todo 修改 window size 为8 默认7
        # todo 窗口大小应该为2才与官方的相比比较合适，
    elif model_name == 'transfuse':
        model = TransFuse_S()  # size  256,192
    elif model_name == 'mctrans':  # 考虑不要
        model = TransFuse_S()  # 网页安装包
    elif model_name == 'cenet':
        model = CE_Net(1)  # att
    elif model_name == 'msnet':#dui
        model = MSNet()
    elif model_name == 'unext':#dui
        model = UNext(1, input_channels=1, in_chans=1, img_size=64)
    elif model_name == 'u2nets2d':
        model = U2NET(1, 1)
    elif model_name == 'dualb':
        # model = DualB()  # 解码器编码器替换
        model = dualCRUNet2d('conv')
    else:
        raise Exception(f"no model name as {model_name}")

    if config.device != 'cpu' and torch.cuda.is_available():
        if verbose:
            logs(f'Use {device}')
        model.to(device)
    else:
        if verbose:
            logs('Use CPU')
        model.to('cpu')
    return model


def get_model3d(model_name, device, verbose=False):
    model_name = model_name.lower()
    filtersless = [16, 32, 64, 128, 256]
    filtersmid = [32, 64, 128, 184, 256]
    filtersbig = [32, 64, 128, 256, 320]
    filterhuge = [64, 128, 256, 512, 512]

    side = False
    upsample = True
    # unet++3d
    if model_name == 'unet':
        #     model = UNet3D(1, 1).cuda()
        # elif model_name == 'myunet3d':
        # model = UNET3d(1, 1).cuda()
        model = UNet3d(1, 1, False)
        # model = UNet3D(1, 1)
    elif model_name == 'resunet':
        model = ResidualUNet3D(1, 1, False, )
    elif model_name == 'vnet':
        model = VNet()
    elif model_name == 'kiunet3d':  # 参数过多，无法运行
        model = kiunet3d(1, 1, num_classes=1)
    elif model_name == 'ynet':
        model = YNet3D()
    elif model_name == 'unetpp':
        model = UNet_Nested()
    elif model_name == 'wingsnet':
        model = WingsNet()
    elif model_name == 'reconnet':
        model = ReconNet(32, 1)
    elif model_name == 'ucaps':
        model = UCaps3D(in_channels=2, out_channels=1)
    elif model_name == 'unetr':
        # model = UNETR(in_channels=1, out_channels=1, img_size=(64, 64, 64), feature_size=32,
        #               norm_name='batch', )  # todo 修改img size
        model = UNETR(in_channels=1, out_channels=1, img_size=(64, 64, 64), pos_embed='conv', norm_name='instance')
    elif model_name == 'transbts':
        _, model = TransBTS(img_dim=64, num_classes=1)  # todo 修改位置编码 4096-》512
    elif model_name == 'vtunet':
        model = VTUNet(num_classes=1)
    elif model_name == 'fedcrld':
        model = mainNetwork(1)
    elif model_name == 'pcamnet':
        model = PCAMNet(1, 1, )
    elif model_name == 'asa':
        model = MEDIUMVIT(in_channels=1, out_channels=1, img_size=(64, 64, 64), norm_name='instance')
    elif model_name == 'u2net':
        model = U2NET(1, 1, )
    elif model_name == 'u2nets':
        model = U2NET(1, 1, side=True)  # 直接返回d0
    elif model_name == 'shuffleu2net':
        model = shuffleU2net(1, 1, side=True)  # 直接返回d0
    elif model_name == 'u2netd':  # 深监督
        model = U2NET(1, 1, sup=True)
    # """
    #  pure u2net3+z
    # """
    elif model_name == 'dualb':  # CASPV6
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbsp':
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbspa':
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbca':
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbcaa':
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbasa':
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbd5df':  # 不同归一化方法
        model = dualCRUNet('conv')
    elif model_name == 'dualbd5zsc':  # zscore+sobel + cut norm
        model = dualCRUNet('conv')
    elif model_name == 'dualbd5cs':  # cut+sobel
        model = dualCRUNet('conv')
    elif model_name == 'dualbd5zs':  # zscore+sobel
        model = dualCRUNet('conv')
    elif model_name == 'dualbd2':  # 同归一化方法
        model = dualCRUNetD2('conv')
    elif model_name == 'dualbd3':  # 同归一化方法
        model = dualCRUNetD3('conv')
    elif model_name == 'dualbd4':  # 同归一化方法
        model = dualCRUNetD4('conv')
    elif model_name == 'dualbd5':  # 同归一化方法
        model = dualCRUNet('conv')
    elif model_name == 'u2net3pv6':
        model = U2net3p5V6(1, 1, side=True)
    # """
    # swin u2net
    # """
    elif model_name == 'u2pcsnet':
        model = U2PCSNet()
    elif model_name == 'duabpathdense':
        # model = DualPathDenseNet(2, 1)
        model = DualSingleDenseNet(2, 1)
    elif model_name == 'swinu':
        model = swinU(1, 1, side=side)
    elif model_name == 'swinv':
        model = swinV(1, 1, side=side, window_size=8, patch_size=4, )  # u2net transformer
    elif model_name == 'swinu2net3plus':
        model = swinu2net3plus(1, 1, side=side, use_checkpoint=True, upsample=False)  # upsample不行
        # model = nn.Sequential(model)
    # elif model_name == 'swinu2net':
    #     model = swinu2net(1, 1, side=side, upsample=upsample)
    elif model_name == "depu2net3plus":
        model = depu2net3plus(1, 1, side=side, upsample=False)  # upsample不行
    elif model_name == "swin3plus":
        model = swin3plus(1, 1, side=side, )  # u2net 嵌套
    elif model_name == 'swinu2net':
        model = swinU2NET(1, 1, side=True)
    elif model_name == 'myu2net':
        model = myU2NET(1, 1, side=True)
    else:
        raise Exception(f"no model name as {model_name}")

    if config.device != 'cpu' and torch.cuda.is_available():
        if verbose:
            logs(f'Use {device}')
        model.to(device)
    else:
        if verbose:
            logs('Use CPU')
        model.to('cpu')

    if config.train and not config.loadModel:
        if model_name.find('dualb') != -1:
            model.apply(partial(weights_init, xohnorm='x'))
        else:
            model.apply(weights_init)

    return model


if __name__ == '__main__':
    # TODO
    model2d = ['unet', 'unetpp', 'raunet', 'cpfnet', 'unet3p', 'sgunet', 'sgl',
               'bionet', 'kiunet', 'msnet']

    model3d = ['unet', 'vnet', 'ynet', 'kiunet3d', 'unetpp', 'wingsnet', ]

    # todo 需要重新设置
    modelTrans = ['uctransnet', 'medt', 'utnet', 'swinunet', 'transbts', 'unetr', ]

    get_parm('3d', 'asa', True)
