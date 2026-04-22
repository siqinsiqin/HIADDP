# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pylidc as pl

from configs import config


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    # numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    # return numpyImage, numpyOrigin, numpySpacing, isflip
    return numpyOrigin, numpySpacing, isflip


# todo 获取所有文件名称
def getPath(filename):
    #  todo 读取文件路径
    for i in range(10):
        path = f'/zsm/xu/$DOWNLOADLUNA16PATH/subset{i}/{filename}.mhd'
        if os.path.exists(path):
            return path


def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def findAllRow(data):
    # todo 根据对应csv文件找出对应的idx
    data = data.groupby('seriesuid').apply(
        lambda d: tuple(d.index) if len(d.index) > 0 else None
    ).dropna()
    return data


"""
# todo 手动查看结节位置对比
"""

ids = [
    '1.3.6.1.4.1.14519.5.2.1.6279.6001.286422846896797433168187085942'
]
data = pd.read_csv(f'{config.csv_path}/annotations.csv', header=0)
ex = []
maxdiff = 0  # todo 最大偏差
for item in ids:
    CT = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == item).first()
    nods = CT.cluster_annotations(tol=CT.pixel_spacing, verbose=False)
    anns = pl.query(pl.Annotation).join(pl.Scan).filter(pl.Scan.series_instance_uid == item).all()
    # todo 加载luna16的数据
    path = getPath(item)
    if path is not None:
        numpyOrigin, numpySpacing, isflip = load_itk_image(path)

        ids = findAllRow(data)[item]

        for nod in anns:
            # v3, v1, v2, diameter = [], [], [], []

            # for ann in nod:
            #     diameter.append(ann.diameter)
            #     v3.append(ann.centroid[2])
            #     v1.append(ann.centroid[0])
            #     v2.append(ann.centroid[1])

            v1 = np.round(nod.centroid[0], 2)
            v2 = np.round(nod.centroid[1], 2)
            v3 = np.round(nod.centroid[2], 2)

            diameter = np.round(nod.diameter, 2)

            # todo 查找对应id 的所有结节，进行一一对比统计
            # todo 遍历该serious id 中所有结节与已知的结节坐标进行比对，将得到的坐标存入ex
            for row in ids:
                xyz = [data['coordX'][row], data['coordY'][row], data['coordZ'][row]]
                xyz = np.round(worldToVoxelCoord(xyz[::-1], numpyOrigin, numpySpacing)[::-1], 2)
                x = np.round(xyz[0], 2)
                y = np.round(xyz[1], 2)
                z = np.round(xyz[2], 2)
                diameter_mm = np.round(data['diameter_mm'][row], 2)

                print([data['coordX'][row], data['coordY'][row], data['coordZ'][row]], xyz, v2, v1, v3)
