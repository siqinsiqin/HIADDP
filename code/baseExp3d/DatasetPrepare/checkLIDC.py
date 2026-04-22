# -*-coding:utf-8 -*-
"""
# Time       ：2023/5/29 8:55
# Author     ：comi
# version    ：python 3.8
# Description：
"""
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

from configs import config

seg_path = config.seg_path_lidc_2d

lesion_list = glob(seg_path + '*.npy')
lesion_list = [item for item in lesion_list if 'sub3c' not in item]
lesion_list = [item for item in lesion_list if 'solid3c' not in item]
print(len(lesion_list))

idx = [365, 377, 402, 407, 409, 413, 442, 445, 454, 458, 462, 474, 486, 502, 534, 548, 554, 555, 560, 589, 595, 607,
       616, 625, 634, 663, 667, 672, 678, 680, 688, 703, 704, 715, 718, 725, 732, 733, 738, 764, 5, 10, 12, 13, 16, 32,
       37, 51, 52, 67, 101, 111, 123, 128, 135, 141, 144, 150, 153, 163, 199, 201, 203, 206, 208, 218, 249, 252, 255,
       267, 270, 281, 285, 319, 327, 335, 337, 338, 345, 353, 354, 358]

delete = [365, 445, 462, 486, 502, 548, 607, 715, 10, 12, 13, 16, 37, 32, 51, 101, 111, 123, 135, 218, 281, 319, 345,
          358, 458, 589, 725, 764, 141, 153, 252, 270, 337, 667, 680, 5, 206, 327]
exclude = []
for i in range(len(delete)):
    img = lesion_list[delete[i]]
    exclude.append(lesion_list[delete[i]])
    print(img)
    img = np.load(img)
    img, msk = np.split(img, 2, axis=0)

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
