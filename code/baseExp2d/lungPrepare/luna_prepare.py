# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os
from glob import glob

import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm

from configs import linuxConfig

"""
# 对结节进行mask标记
"""
#  对整个肺部进行处理
first_step = linuxConfig['first_step']
sec_step = linuxConfig['sec_step']
anno_path = linuxConfig['anno_path']

out_images = []  # final set of images
out_nodemasks = []  # final set of nodemasks
out_masks = []
np.seterr(divide='ignore', invalid='ignore')


def make_mask(center, diam, z, width, height, spacing, origin, point):
    '''
        Center : centers of circles px -- list of coordinates x,y,z
        diam : diameters of circles px -- diameter
        widthXheight : pixel dim of image
        spacing = mm/px conversion rate np array x,y,z
        origin = x,y,z mm np.array
        z = z position of slice in world coordinates mm 切片在世界坐标的位置
    '''
    xmin, xmax, ymin, ymax = point
    mask = np.zeros([height, width])  # mask大小
    # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    # 定义结节落入的体素范围
    v_center = (center - origin) / spacing
    v_diam = int(diam / spacing[0] + 5)  # 间距除以x ？[0.72265625 0.72265625 1.79999995] x间距
    # v_diam = 11.5不加5会怎么样?
    # 计算边界范围
    v_xmin = np.max([0, int(v_center[0] - v_diam) - 5])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + 5])
    v_ymin = np.max([0, int(v_center[1] - v_diam) - 5])
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + 5])

    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    # Convert back to world coordinates for distance calculation
    # x_data = [x * spacing[0] + origin[0] for x in range(width)]
    # y_data = [x * spacing[1] + origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:  # 便利区间内部，判断中心点减去该点坐标是否小于diam
                mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0

    return mask[ymin: ymax, xmin:xmax]


def crop64(img, center, width, height, spacing, origin):
    # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    # 定义结节落入的体素范围
    v_center = (center - origin) / spacing
    v_diam = 64  # 间距除以x ？[0.72265625 0.72265625 1.79999995] x间距
    # v_diam = 11.5不加5会怎么样?
    # 计算边界范围
    v_xmin = np.max([0, int(v_center[0] - v_diam)])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam)])
    v_ymin = np.max([0, int(v_center[1] - v_diam)])
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam)])
    return img[v_ymin:v_ymax, v_xmin:v_xmax, ], v_xmin, v_xmax, v_ymin, v_ymax


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


def deal_nodule(show=False):
    # 获取文件列表
    path = r'/zsm/xu/'  # linuxConfig['base_path']

    if not os.path.exists(first_step):
        os.makedirs(first_step)
    t = 0
    for i in range(1):
        luna_path = os.path.join(path, '$DOWNLOADLUNA16PATH/', "subset" + str(i) + '/')
        file_list = glob(luna_path + "*.mhd")  # 获取路径下全部的mhd结尾文件

        df_node = pd.read_csv(anno_path + "annotations.csv")
        df_node["file"] = df_node["seriesuid"].map(
            lambda file_name: get_filename(file_list, file_name))  # 将对应文件加入文件路径
        # df_node.to_csv(output_path+'/DatasetPrepare.csv') # 保存查看文件
        df_node = df_node.dropna()  # 删除丢失的文件行

        # Looping over the image files
        print('---------possess subset ' + str(i) + ':Nodule Mask ---------')
        for fcount, img_file in enumerate(tqdm(file_list)):

            mini_df = df_node[df_node["file"] == img_file]  # 获取文件对应的全部结节信息

            if mini_df.shape[0] > 0:  # 一些文件有多个结节，只取其中一个加载制作整肺mask
                # load the data once
                itk_img = sitk.ReadImage(img_file)  # 加载文件
                img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
                num_z, height, width = img_array.shape  # height X width constitute the transverse plane
                origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm) # 世界坐标中的X、Y、Z原点
                spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm) # 世界坐标中体素的间距。

                # go through all nodes (why just the biggest?)
                for node_idx, cur_row in mini_df.iterrows():  # 遍历每一个结节
                    node_x = cur_row["coordX"]
                    node_y = cur_row["coordY"]
                    node_z = cur_row["coordZ"]
                    diam = cur_row["diameter_mm"]  # 4.681381581

                    center = np.array([node_x, node_y, node_z])  # 世界坐标，肺结节中心
                    # nodule center in voxel space (still x,y,z ordering)
                    v_center = np.rint((center - origin) / spacing)  # 体素坐标，取整
                    # 结节中心v_center[2],取结节上一层和下一层的横切面，切片取值范围为[0,numz-1]，防止切片越界
                    for i, i_z in enumerate(np.arange(int(v_center[2]) - 1, int(v_center[2]) + 2).clip(0, num_z - 1)):
                        # 制作mask
                        # i_z * spacing[2] + origin[2]   -313 z:体素坐标*间距+原始z
                        nomal_z = i_z * spacing[2] + origin[2]
                        new_img, v_xmin, v_xmax, v_ymin, v_ymax = crop64(img_array[i_z], center, width, height, spacing,
                                                                         origin)
                        mask = make_mask(center, diam, nomal_z, width, height, spacing, origin,
                                         (v_xmin, v_xmax, v_ymin, v_ymax))

                        new_img = resize(new_img, (512, 512))
                        mask = resize(mask, (512, 512))

                        np.save(
                            os.path.join('/zsm/jwj/baseExp/$LUNA16PROPOCESSPATH/luna/', '_img_' + str(t) + '_.npy'),
                            new_img[np.newaxis, ...])
                        np.save(
                            os.path.join('/zsm/jwj/baseExp/$LUNA16PROPOCESSPATH/luna/', '_mask_' + str(t) + '_.npy'),
                            mask[np.newaxis, ...])

                        t += 1
                        # plt.figure()
                        # fig, plots = plt.subplots(1, 2)
                        # plots[0].imshow(new_img, cmap='gray')  # 原图
                        # plots[1].imshow(mask, cmap='gray')  # mask
                        # plt.show()


if __name__ == '__main__':
    deal_nodule()  # 处理结节
