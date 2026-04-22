# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm

from configs import config


def findAllRow(data):
    # todo 根据对应csv文件找出对应的idx
    data = data.groupby('seriesuid').apply(
        lambda d: tuple(d.index) if len(d.index) > 0 else None
    ).dropna()
    return data


def load_itk_image_simple(filename):
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


"""
保存luna16中对应id的origin 和spacing
"""
data = pd.read_csv(f'{config.csv_path}/annotations.csv', header=0)  # 加载luna16结节信息
mhd_path = pd.read_csv(f'{config.csv_path}/mhd_path.csv', header=0)  # todo 加载luna16文件路径
count = findAllRow(data)
path = findAllRow(mhd_path)
attrs = []
for i, item in tqdm(enumerate(np.unique(data['seriesuid']))):
    numpyOrigin, numpySpacing = load_itk_image_simple(mhd_path['path'][path[f'{item}.mhd'][0]])
    attrs.append({'seriesuid': item, 'originX': numpyOrigin[0], 'originY': numpyOrigin[1], 'originZ': numpyOrigin[2],
                  'spacingX': numpySpacing[0], 'spacingY': numpySpacing[1], 'spacingZ': numpySpacing[2], })
df = pd.DataFrame(attrs)
df.to_csv(f'{config.csv_path}/MhdOriginAndSpaving.csv')
print('Save luna annotation MhdOriginAndSpaving !!!')
