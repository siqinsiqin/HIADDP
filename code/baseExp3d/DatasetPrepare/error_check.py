# -*-coding:utf-8 -*-
"""
# Time       ：2023/6/3 16:10
# Author     ：comi
# version    ：python 3.8
# Description：
"""
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from configs import config

path = r'/zljteam/jwj/baseExpV5/$segmentation/error_2d/'


lesion_list = glob(path + '*.npy')
import os
from PIL import Image


# 遍历图像文件
for i, image_file in enumerate(lesion_list):
    loadimg = np.load(image_file)
    img, msk = np.split(loadimg, 2, axis=0)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img[0, :, :], cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[1].imshow(msk[0, :, :], cmap=plt.cm.gray, )
    ax[1].axis('off')
    ax[2].imshow(msk[0, :, :] * img[0, :, :], )
    ax[2].axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig(config.pic_path + f'/check_{i}.png', bbox_inches="tight", pad_inches=0)
    plt.close()