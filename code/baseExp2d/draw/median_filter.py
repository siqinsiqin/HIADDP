# -*-coding:utf-8 -*-
"""
# Time       ：2022/9/27 9:13
# Author     ：comi
# version    ：python 3.8
# Description：
"""
# 中值滤波#
import cv2
import numpy as np

from draw.Anisotropic_diffusion import anisodiff2D


def MedianFilter(img, k=3, padding=None):
    imarray = img
    height = imarray.shape[0]
    width = imarray.shape[1]
    if not padding:
        edge = int((k - 1) / 2)
        if height - 1 - edge <= edge or width - 1 - edge <= edge:
            print("The parameter k is to large.")
            return None
        new_arr = np.zeros((height, width), dtype="uint8")
        for i in range(edge, height - edge):
            for j in range(edge, width - edge):
                new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1])  # 调用np.median求取中值
    return new_arr


# img = cv2.imread("D:\\backup\\learning\\DeepLearning\\DL\\segmentation\\baseExpV3\\zhanlian.png", 0)
# result = MedianFilter(img)
#
# # cv2.imwrite('re-cat.jpg', result)
# # median3 = cv2.medianBlur(result, 3)
# # median5 = cv2.medianBlur(result, 5)
# cv2.imshow("input", img.astype('int8'))
# cv2.imshow("output", result.astype('int8'))
#
# an_filter = anisodiff2D()
# diff_im = an_filter.fit(result)
# diff_im_img = diff_im.astype(np.uint8)
#
# cv2.imshow("diff", diff_im_img.astype('int8'))
#
#
# # cv2.imshow("Median3", median3)
# # cv2.imshow("Median5", median5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
