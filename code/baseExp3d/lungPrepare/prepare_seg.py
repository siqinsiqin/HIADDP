# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os

import numpy as np
import pylidc as pl
from skimage.transform import resize

from configs import linuxConfig
from utils.helper import lumTrans

seg_path = linuxConfig['seg_path']
CTs = pl.query(pl.Scan).allc()
padding = [(30, 10), (10, 25), (0, 0)]  # todo 背景填充

print('total CTs len', len(CTs))
os.makedirs(seg_path, exist_ok=True)
for i in (range(0, 100)):

    ann = pl.query(pl.Annotation).filter(
        CTs[i].id == pl.Annotation.scan_id and pl.Annotation.calcification in [1, ]).allc()
    name = CTs[i].series_instance_uid
    vol = CTs[i].to_volume()
    print(name, len(ann))
    # vol = ann[i].scan.to_volume()

    # ann[i].visualize_in_3d()  # todo 3d展示
    # ann[i].visualize_in_scan()  # todo ct内部展示
    # name = pl.query(pl.Scan).filter(pl.Scan.id == ann[i].scan_id).first().series_instance_uid

    for t in range(len(ann)):
        masks = ann[t].boolean_mask(pad=padding)
        bboxes = ann[t].bbox(pad=padding)

        for k in range(masks.shape[2]):
            mask = masks[:, :, k]
            img = vol[bboxes][:, :, k]

            mask = resize(mask, (512, 512))
            img = lumTrans(img)
            img = resize(img, (512, 512))

            # fig, ax = plt.subplots(1, 2, figsize=(5, 3))
            # ax[0].imshow(img, cmap=plt.cm.gray)
            # # ax[0].axis('off')
            #
            # ax[1].imshow(mask, cmap=plt.cm.gray)
            # # ax[1].axis('off')
            #
            # plt.tight_layout()
            # # plt.savefig("../images/mask_bbox.png", bbox_inches="tight")
            # plt.show()
            np.save(seg_path + name + '_mask_' + str(i) + str(t) + str(k) + '_.npy', mask[np.newaxis, ...])
            np.save(seg_path + name + '_img_' + str(i) + str(t) + str(k) + '_.npy', img[np.newaxis, ...]),

print('Done')
