# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
from glob import glob

from configs import config


def cntDiffLabel(seg_path, re):
    lesion_list = glob(seg_path + re)
    lesion_list.sort()
    return len(lesion_list)


luna_2d = config.seg_path_luna_2d
luna_3d = config.seg_path_luna_3d

subtletycls = ['ExtremelySubtle', 'ModeratelySubtle', 'FairlySubtle', 'ModeratelyObvious', 'Obvious']
internalStructurecls = ['SoftTissue', 'Fluid', 'Fat', 'Air']
calcificationcls = ['Popcorn', 'Laminated', 'Solid', 'Noncentral', 'Central', 'Absent']
sphericitycls = ['Linear', 'OvoidLinear', 'Ovoid', 'OvoidRound', 'Round']
margincls = ['PoorlyDefined', 'NearPoorlyDefined', 'MediumMargin', 'NearSharp', 'Sharp']
lobulationcls = ['NoLobulation', 'NearlyNoLobulation', 'MediumLobulation', 'NearMarkedLobulation',
                 'MarkedLobulation']
spiculationcls = ['NoSpiculation', 'NearlyNoSpiculation', 'MediumSpiculation', 'NearMarkedSpiculation',
                  'MarkedSpiculation']
texturecls = ['NonSolidGGO', 'NonSolidMixed', 'PartSolidMixed', 'SolidMixed', 'solid']

maligancy = ['benign', 'uncertain', 'malignant']
arrs = [subtletycls, internalStructurecls, calcificationcls, sphericitycls, margincls, lobulationcls, spiculationcls,
        texturecls, maligancy]
for arr in arrs:
    lens = 0
    for item in arr:
        re = f'*_{item}_*.npy'
        val = cntDiffLabel(luna_2d, re)
        lens += val
        print(f'{item}：{val}')
    print(f'{arr}:', lens)
