# -_-coding:utf-8 -_-
"""
# Time       ：2022/5/19 18:31
# Author     ：comi
# version    ：python 3.8
# Description：
"""
from glob import glob

from configs import config
from utils.helper import getAllAttrs

seg_path = config.seg_path_lidc_3d
# seg_path = config.seg_path_luna_3d
seg_path = f'/{config.server}/jwj/baseExpV7/$segmentation/seg_lidc_3d_rd2/'

# # # todo 统计各类别个数
# for key, labels in getAllAttrs(True).items():
#     print(key)
#     for label in labels:
#         lesion_list = glob(seg_path + f'*_{label}_*.npy')
#         lesion_list = [item for item in lesion_list if 'sub3c' not in item]
#         lesion_list = [item for item in lesion_list if 'solid3c' not in item]
#
#         print(label, len(lesion_list))
#     print('=' * 20)

# todo  统计全部个数
# seg_path = config.seg_path_luna_3d
# seg_path = '/zljteam/jwj/baseExpV5/$segmentation/error_2d/'
print(seg_path)
lesion_list = glob(seg_path + '*.npy')
print(len(lesion_list))

# print(
# metrics.keys())
#
# seg_path = config.seg_path_lidc_2d
# #
# lesion_list = glob(seg_path + '*.npy')
# print(len(lesion_list))
# lesion_list = [item for item in lesion_list if 'sub3c' not in item]
# print(len(lesion_list))
# lesion_list = [item for item in lesion_list if 'solid3c' not in item]
# print(len(lesion_list))

# delete = [365, 445, 462, 486, 502, 548, 607, 715, 10, 12, 13, 16, 37, 32, 51, 101, 111, 123, 135, 218, 281, 319,
#           345, 358, 458, 589, 725, 764, 141, 153, 252, 270, 337, 667, 680, 5, 206, 327]
# exclude = []
# for i in range(len(delete)):
#     exclude.append(lesion_list[delete[i]])
#
# import shutil
#
# for i in range(len(exclude)):
#     # if re.search(r'solid3c', lesion_list[i]):
#     #     # 指定要移动的文件路径
#     #     source_file = lesion_list[i]
#     #
#     #     # 指定目标位置
#     #     target_directory = '/zsm/jwj/baseExpV5/$segmentation/error_2d/'
#     #
#     #     # 使用shutil的move函数移动文件
#     #     shutil.move(source_file, target_directory)
#     # 指定要移动的文件路径
#     source_file = exclude[i]
#
#     # 指定目标位置
#     target_directory = '/zsm/jwj/baseExpV5/$segmentation/error_2d/'
#
#     # 使用shutil的move函数移动文件
#     shutil.move(source_file, target_directory)
# 筛选2d 3d样本
# import glob
#
# prefix_to_remove = '/zljteam/jwj/baseExpV5//$segmentation/seg_lidc_2d/'
#
# seg_path = config.seg_path_lidc_2d
# lesion_list = glob.glob(seg_path + '*.npy')
#
# # 去除前缀
# lesion_list = [item.replace(prefix_to_remove, '') for item in lesion_list]
#
# seg_path = config.seg_path_lidc_3d
# lesion_list1 = glob.glob(seg_path + '*.npy')
#
# # 去除前缀
# prefix_to_remove = '/zljteam/jwj/baseExpV5//$segmentation/seg_lidc_3d/'
# lesion_list1 = [item.replace(prefix_to_remove, '') for item in lesion_list1]
# prefix_to_add = '/zljteam/jwj/baseExpV5//$segmentation/seg_lidc_2d/'
#
# exclude = [prefix_to_add + item for item in lesion_list if item in lesion_list1]
# print(len(exclude))
#
# seg_path = config.seg_path_lidc_2d
# lesion_list = glob.glob(seg_path + '*.npy')
#
# # 删除不在 exclude 列表中的文件
# for item in lesion_list:
#     if item not in exclude:
#         file_path = item
#         # 指定目标位置
#         target_directory = '/zljteam/jwj/baseExpV5/$segmentation/error_2d'
#
#         # 使用shutil的move函数移动文件
#         shutil.move(file_path, target_directory)
#
# seg_path = config.seg_path_lidc_2d
# lesion_list = glob.glob(seg_path + '*.npy')
#
# print(len(lesion_list))
