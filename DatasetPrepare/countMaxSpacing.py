# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import numpy as np
import pandas as pd
import pylidc as pl
from tqdm import tqdm


def findAllRow(data):
    """
    根据serious id   找出data中对应的结节ids
    """
    data = data.groupby('seriesuid').apply(
        lambda d: tuple(d.index) if len(d.index) > 0 else None
    ).dropna()

    return data


def findmax(slice_thickness, pixel_spacing, mode=1):
    if mode == 2:
        val1 = 2 * slice_thickness + 2 * pixel_spacing
        val2 = 2 * slice_thickness + 2 * pixel_spacing
    else:
        val1 = 2 * slice_thickness + 1 * pixel_spacing
        val2 = 2 * slice_thickness + 1 * pixel_spacing

    if val1 >= val2:
        return np.round(val1, 2)
    else:
        return np.round(val2, 2)


"""
# todo 2*层厚 + 2*层间距 的最大值
# todo 2*层厚 + 层间距 的最大值
"""
data = pd.read_csv('/zsm/jwj/baseExp/LIDCXML/annotations.csv', header=0)
mhd_path = pd.read_csv('/zsm/jwj/baseExp/DatasetPrepare/mhd_path.csv', header=0)  # todo 加载luna16文件路径

max22 = 0
max21 = 0
count = findAllRow(data)
path = findAllRow(mhd_path)
for i, item in tqdm(enumerate(data['seriesuid'])):
    CT = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == item).first()
    slice_thickness = CT.slice_thickness
    pixel_spacing = CT.pixel_spacing
    max21 = findmax(slice_thickness, pixel_spacing, 1)
    max22 = findmax(slice_thickness, pixel_spacing, 2)

print(max21, max22)
