# -*-coding:utf-8 -*-
"""
# Time       ：2023/9/15 15:02
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import glob
import warnings

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.image import imsave
from pytorch_grad_cam import GradCAM

from utils.helper import load_model_k_checkpoint, get_model3d, sobel_filter

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from torch.backends import cudnn

cudnn.benchmark = False


def get_net():
    """
    Get Network for evaluation
    """
    modelname = 'dualbasa'
    model = get_model3d(modelname, 'cuda:0')
    load_model_k_checkpoint('/zsm/jwj/baseExpV7/pth_luna/', '3d', modelname, 'adam',
                            'dice', model, 2, verbose=False)
    model.eval()
    return model


def z_score_norm(img):
    # z-score
    img_mean = np.mean(img)  # torch
    img_std = np.std(img)
    norm_img = (img - img_mean) / img_std
    return norm_img


def cut_norm(img):
    # 截断归一化
    lungwin = np.array([-1000., 400.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg


"""

227962600322799211676960828223_solid8p_Obvious_SoftTissue_Absent_Ovoid_MediumMargin_MediumLobulation_MediumSpiculation_SolidMixed_malignant_35.png


254254303842550572473665729969_sub6p_ModeratelySubtle_SoftTissue_Absent_Ovoid_NearPoorlyDefined_NoLobulation_NoSpiculation_NonSolidGGO_uncertain_64.png

259123825760999546551970425757_solid8p_Obvious_SoftTissue_Absent_Ovoid_MediumMargin_MarkedLobulation_MarkedSpiculation_SolidMixed_malignant_99.png
"""

glist = glob.glob('/zsm/jwj/baseExpV7/$segmentation/seg_luna_3d/' + '*NearMarkedSpiculation*.npy')  # 2 18  250
glist = glob.glob(
    '/zsm/jwj/baseExpV7/$segmentation/seg_luna_3d/' +
    '*259123825760999546551970425757_solid8p_Obvious_SoftTissue_Absent_Ovoid_MediumMargin_MarkedLobulation_MarkedSpiculation_SolidMixed*.npy')  # 2 18  250

data = np.load(glist[0])
img, msk = np.split(data, 2, axis=0)
img_z = z_score_norm(img)
# img_s = z_score_norm(img)
# img_s = cut_norm(img)
# imshows(img_s[0])

img_sobel = sobel_filter(img_z[0])
# imshows(img_z[0], save=True, savename='img')
# imshows(img_sobel[0], save=True, savename='sobel')
img = np.stack([img_z, img_sobel], axis=1).squeeze(axis=0)

img = img[np.newaxis, ...]
imgs = img[0, 0, :]

img = torch.as_tensor(img).float().contiguous()

model = get_net()

if torch.cuda.is_available():
    model = model.cuda()
    img = img.cuda()

output = model(img)

normalized_masks = torch.sigmoid(output).cpu()
normalized_masks = (normalized_masks > 0.5).float()
import SimpleITK as sitk

out = sitk.GetImageFromArray(normalized_masks[0, 0, :, :, :])
sitk.WriteImage(out, 'show.nrrd')
# 开始cam 画图
sem_classes = [
    'background', 'real'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

# 将需要进行CAM的类名写至此处
plaque_category = sem_class_to_idx["real"]
plaque_mask = normalized_masks[0, 0, :, :, :].detach().cpu().numpy()
plaque_mask_uint8 = 255 * np.uint8(plaque_mask == plaque_category)
plaque_mask_float = np.float32(plaque_mask == plaque_category)

imgs = np.repeat(imgs[:, :, :, None], 3, axis=-1)


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[0] * self.mask).sum()


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# 此处修改希望得到Grad-CAM图所在的网络层
target_layers = [model.neck_attn.sp[-1]]  # neck_attn, f_attn, t_attn, s_attn, out  .cbs2
# target_layers = [model.neck_attn.ca]  # neck_attn, f_attn, t_attn, s_attn, out  .cbs2
# target_layers = [model.up.conv5]  # neck_attn, f_attn, t_attn, s_attn, out  .cbs2
# target_layers = [model.down.conv4]  # neck_attn, f_attn, t_attn, s_attn, out  .cbs2

targets = [SemanticSegmentationTarget(plaque_category, plaque_mask_float)]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
grayscale_cam = cam(input_tensor=img, targets=targets)[0, :]  # 这里的32是不同层对应的中间尺寸大小

imgs = (imgs.astype(np.float32) - np.min(imgs.astype(np.float32))) / (
        np.max(imgs.astype(np.float32)) - np.min(imgs.astype(np.float32)))

print("z-depth", grayscale_cam.shape)
for i in range(grayscale_cam.shape[2]):
    # cam_image = show_cam_on_image(imgs[:, :, 32], grayscale_cam[:, :, 32], use_rgb=True)  # 1 3 8 16 32
    cam_image = show_cam_on_image(imgs[:, :, 32], grayscale_cam[:, :, i], use_rgb=True)  # 1 3 8 16 32

    fig, axs = plt.subplots(figsize=(5, 5))
    savename = 'neck_attn'
    axs.imshow(cam_image, alpha=1, interpolation='sinc')
    axs.axis('off')
    imsave(f'/zsm/jwj/baseExpV7/{savename}_{i}.png', cam_image, cmap=plt.cm.gray)
    plt.tight_layout()
    plt.show()
    plt.close()
