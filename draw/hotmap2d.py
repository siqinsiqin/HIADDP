# -*-coding:utf-8 -*-
"""
# Time       ：2023/9/15 15:02
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM

from utils.helper import get_model2d, load_model_k_checkpoint

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from torch.backends import cudnn

cudnn.benchmark = False


def get_net():
    """
    Get Network for evaluation
    """
    modelname = 'unet'
    model = get_model2d(modelname, 'cuda:0')
    load_model_k_checkpoint('/zsm/jwj/baseExpV5/pth_luna/', '2d', modelname, 'adam',
                            'dice', model, 2, verbose=False)
    model.eval()
    return model


def z_score_norm(img):
    # z-score
    img_mean = np.mean(img)  # torch
    img_std = np.std(img)
    norm_img = (img - img_mean) / img_std
    return norm_img


data = np.load(
    '/zsm/jwj/baseExpV5/$segmentation/seg_luna_2d/1.3.6.1.4.1.14519.5.2.1.6279.6001.107109359065300889765026303943_solid8p_Obvious_SoftTissue_Absent_Ovoid_Sharp_NearlyNoLobulation_NoSpiculation_tSolid_malignant_72.npy',
    allow_pickle=True)
img, msk = np.split(data, 2, axis=0)
img = z_score_norm(img)

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

# 开始cam 画图
sem_classes = [
    'background', 'real'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

# 将需要进行CAM的类名写至此处
plaque_category = sem_class_to_idx["real"]
plaque_mask = normalized_masks[0, 0, :, :].detach().cpu().numpy()
plaque_mask_uint8 = 255 * np.uint8(plaque_mask == plaque_category)
plaque_mask_float = np.float32(plaque_mask == plaque_category)

imgs = np.repeat(imgs[:, :, None], 3, axis=-1)


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
                      image_weight: float = 0.7) -> np.ndarray:
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
target_layers = [model.Conv]
targets = [SemanticSegmentationTarget(plaque_category, plaque_mask_float)]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
grayscale_cam = cam(input_tensor=img, targets=targets)[0, :]

imgs = (imgs.astype(np.float32) - np.min(imgs.astype(np.float32))) / (
        np.max(imgs.astype(np.float32)) - np.min(imgs.astype(np.float32)))

cam_image = show_cam_on_image(imgs, grayscale_cam, use_rgb=True)

img = Image.fromarray(cam_image)
# 保存位置
img.save("result.png")
