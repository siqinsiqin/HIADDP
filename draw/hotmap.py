# -*-coding:utf-8 -*-
"""
# Time       ：2023/5/4 18:51
# Author     ：comi
# version    ：python 3.8
# Description：
# todo 热图
"""

import cv2
import matplotlib.pyplot as plt
from PIL import Image


def visulize_spatial_attention(img_path, attention_mask, ratio=1, cmap="jet"):  # jet,inferno ,viridis
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask: 2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:   attention style, default: "jet"
    """
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    # scale表示放大或者缩小图片的比率
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')
    plt.tight_layout()

    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.3, interpolation='nearest', cmap=cmap)
    plt.show()
    plt.tight_layout()


if __name__ == "__main__":
    import SimpleITK as sitk

    # 读取nrrd文件
    image = sitk.ReadImage(r"D:\backup\learning\DeepLearning\DL\segmentation\baseExpV5\cat_1.nrrd")

    # 获取图像的像素数组
    array = sitk.GetArrayFromImage(image)

    img_path = r"C:\Users\comi\Desktop\duab.png"  # 图像路径
    visulize_spatial_attention(img_path=img_path, attention_mask=array)
