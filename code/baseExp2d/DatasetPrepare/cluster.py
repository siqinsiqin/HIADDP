# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import time

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pylidc as pl

from utils.helper import showTime


def load_itk_image(filename):
    # with open(filename) as f:
    #     contents = f.readlines()
    #     line = [k for k in contents if k.startswith('TransformMatrix')][0]
    #     transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
    #     transformM = np.round(transformM)
    #     if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
    #         isflip = True
    #     else:
    #         isflip = False

    itkimage = sitk.ReadImage(filename)
    # numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    # return numpyImage, numpyOrigin, numpySpacing, isflip
    return numpyOrigin, numpySpacing


def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def findAllRow(data):
    """
    根据serious id   找出data中对应的结节ids
    """
    # ids = []
    # for t, row in (enumerate(data['seriesuid'])):
    #     if row == id:
    #         ids.append(t)
    data = data.groupby('seriesuid').apply(
        lambda d: tuple(d.index) if len(d.index) > 0 else None
    ).dropna()

    return data


def printt(bias, item, row, xyz, vvv, diff):
    print(f'small {bias}', item, row, xyz, vvv, 'diff：', diff)


diffs = []


def findNodule(sample):
    data = pd.read_csv('/zsm/jwj/baseExp/LIDCXML/annotations.csv', header=0)
    mhd_path = pd.read_csv('/zsm/jwj/baseExp/DatasetPrepare/mhd_path.csv', header=0)  # todo 加载luna16文件路径

    ex = []
    maxdiff = 0  # todo 最大偏差
    # todo 对luna16的指定id聚类找到结节
    # todo 利用z轴值来判断是否是 同一结节
    count = findAllRow(data)
    path = findAllRow(mhd_path)
    for i, item in (enumerate(data['seriesuid'])):
        # todo 加载lidc
        CT = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == item).first()
        nods = CT.cluster_annotations(tol=CT.pixel_spacing, verbose=False)

        excepts = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.159996104466052855396410079250',
                   '1.3.6.1.4.1.14519.5.2.1.6279.6001.897684031374557757145405000951',
                   '1.3.6.1.4.1.14519.5.2.1.6279.6001.137763212752154081977261297097',
                   '1.3.6.1.4.1.14519.5.2.1.6279.6001.214252223927572015414741039150',
                   '1.3.6.1.4.1.14519.5.2.1.6279.6001.323753921818102744511069914832'
                   ]

        numpyOrigin, numpySpacing = load_itk_image(mhd_path['path'][path[f'{item}.mhd'][0]])

        for nod in nods:
            v3, v1, v2, diameter = [], [], [], []

            if item not in excepts:
                if len(nod) < 3:
                    continue

            for ann in nod:
                diameter.append(ann.diameter)
                v3.append(ann.centroid[2])
                v1.append(ann.centroid[0])
                v2.append(ann.centroid[1])
            v1 = np.round(np.average(v1), 2)
            v2 = np.round(np.average(v2), 2)
            v3 = np.round(np.average(v3), 2)
            diameter = np.round(np.average(diameter), 2)

            if 3. <= diameter:
                # todo 查找对应id 的所有结节，进行一一对比统计
                # todo 遍历该serious id 中所有结节与已知的结节坐标进行比对，将得到的坐标存入ex

                for row in count[item]:
                    if row not in ex:
                        xyz = [data['coordX'][row], data['coordY'][row], data['coordZ'][row]]
                        xyz = np.round(worldToVoxelCoord(xyz[::-1], numpyOrigin, numpySpacing)[::-1], 2)
                        x, y, z = xyz[0], xyz[1], xyz[2]
                        diameter_mm = np.round(data['diameter_mm'][row], 2)

                        diameter_lim = 50.
                        diff = np.round(abs(diameter_mm - diameter), 2)
                        diffs.append(diff)
                        if diff > maxdiff:
                            maxdiff = diff

                        diffz = abs(z - v3)
                        diffy = abs(y - v1)
                        diffx = abs(x - v2)
                        diffd = diff <= diameter_lim
                        gap = .5
                        if item in excepts:
                            anns = pl.query(pl.Annotation).join(pl.Scan).filter(
                                pl.Scan.series_instance_uid == item).all()
                            one_nodule = []
                            """遍历serious id中所有掩膜,比较xyz,将在bias内的结节进行合并"""
                            for ann in anns:
                                v3 = np.round(ann.centroid[2], 2)
                                v1 = np.round(ann.centroid[0], 2)
                                v2 = np.round(ann.centroid[1], 2)
                                diffz = abs(z - v3)
                                diffy = abs(y - v1)
                                diffx = abs(x - v2)
                                bias = 3.  # 中心位置偏差
                                if diffx <= bias and diffy <= bias and diffz <= bias:
                                    one_nodule.append(ann)
                            if len(one_nodule) != 0:
                                printt('zzz', item, row, xyz, [v2, v1, v3], diff)
                                ex.append(row)
                            # if item == '1.3.6.1.4.1.14519.5.2.1.6279.6001.137763212752154081977261297097':
                            #     if diffx <= 6 and diffy <= 7.5 and diffz <= 3 and diffd:
                            #         printt('xy', item, row, xyz, [v2, v1, v3], diff)
                            #         ex.append(row)
                            # elif diffx <= 5 and diffy <= 3 and diffz <= 3 and diffd:
                            #     printt('x', item, row, xyz, [v2, v1, v3], diff)
                            #     ex.append(row)
                            # elif diffx <= 3 and diffy <= 5 and diffz <= 3 and diffd:
                            #     printt('y', item, row, xyz, [v2, v1, v3], diff)
                            #     ex.append(row)
                            # elif diffx <= 3 and diffy <= 3 and diffz <= 5 and diffd:
                            #     printt('z', item, row, xyz, [v2, v1, v3], diff)
                            #     ex.append(row)
                        else:
                            if diffx <= gap and diffy <= gap and diffz <= gap and diffd:
                                printt(f'{gap}', item, row, xyz, [v2, v1, v3], diff)
                                ex.append(row)
                            elif diffx <= gap * 2 and diffy <= gap * 2 and diffz <= gap * 2 and diffd:
                                printt(f'{gap * 2}', item, row, xyz, [v2, v1, v3], diff)
                                ex.append(row)
                            elif diffx <= gap * 3 and diffy <= gap * 3 and diffz <= gap * 3 and diffd:
                                printt(f'{gap * 2}', item, row, xyz, [v2, v1, v3], diff)
                                ex.append(row)
                            elif diffx <= gap * 4 and diffy <= gap * 4 and diffz <= gap * 4 and diffd:
                                printt(f'{gap * 4}', item, row, xyz, [v2, v1, v3], diff)
                                ex.append(row)
                            elif diffx <= gap * 5 and diffy <= gap * 5 and diffz <= gap * 5 and diffd:
                                printt(f'{gap * 5}', item, row, xyz, [v2, v1, v3], diff)
                                ex.append(row)
                            elif diffx <= gap * 6 and diffy <= gap * 6 and diffz <= gap * 6 and diffd:
                                printt(f'{gap * 6}', item, row, xyz, [v2, v1, v3], diff)
                                ex.append(row)
                            # todo row +1 为anno
        # todo 查看哪个结节没有被匹配
        for k in count[item]:
            if k not in ex:
                xyz = [item, data['coordX'][k], data['coordY'][k], data['coordZ'][k], data['diameter_mm'][k]]
                print('attn:', xyz)
                # todo k+2 为anno

    print(len(ex), maxdiff)
    print(ex)


if __name__ == '__main__':
    start_time = time.time()
    sample = pd.read_csv('/zsm/jwj/baseExp/DatasetPrepare/ones.csv')
    findNodule(sample)
    end_time = time.time()
    showTime('Total', start_time, end_time)
    # start_time = time.time()
    # sample = pd.read_csv('/zsm/jwj/baseExp/DatasetPrepare/mores.csv')
    # findNodule(sample)
    # end_time = time.time()
    # showTime('Total', start_time, end_time)
    # print(np.average(diffs))
