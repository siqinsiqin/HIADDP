# -*-coding:utf-8 -*-
"""
# Time       ：2022/9/27 9:18
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import cv2
import numpy as np

from draw.Anisotropic_diffusion import anisodiff2D
from draw.median_filter import MedianFilter

img = cv2.imread('D:\\backup\\learning\\DeepLearning\\DL\\segmentation\\baseExpV3\\img.png')
# print(img.shape)    # (1280, 1920, 3)
# img = cv2.pyrUp(img)
img = cv2.pyrDown(img)
# print(img.shape)    # (320, 480, 3)
# cv2.imshow('up_nodule', img)
an_filter = anisodiff2D()
# result = MedianFilter(img)

diff_im = an_filter.fit(img)
diff_im_img = diff_im.astype(np.uint8)
result = cv2.pyrUp(diff_im_img)
cv2.imshow("input", img)
cv2.imshow("output", result)
cv2.imshow("diff", diff_im_img)

cv2.waitKey(0)
cv2.destroyAllWindows()