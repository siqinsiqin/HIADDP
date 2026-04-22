# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os
from glob import glob

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure
from skimage import morphology
from skimage.transform import resize
from sklearn.cluster import KMeans
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


# 得到肺部mask
def segmentation(file_list, show=False):
    print('---------possess on function : segmentation---------')
    for img_file in tqdm(file_list):

        imgs_to_process = np.load(img_file).astype(np.float64)

        for i in range(len(imgs_to_process)):  # 3张
            img = imgs_to_process[i]

            "标准化"
            mean = np.mean(img)
            std = np.std(img)
            img = img - mean
            img = img / std

            # 找到接近肺部的平均像素值去重新归一化褪色图像
            # 选取图像行列为[100:400]
            middle = img[100:400, 100:400]
            if np.isnan(middle).any():
                continue
            # 为了改进阈值查找，将最大值和最小值换为均值

            mean = np.mean(middle)
            max = np.max(img)
            min = np.min(img)
            # To improve threshold finding, I'm moving the
            # underflow and overflow on the pixel spectrum
            img[img == max] = mean
            img[img == min] = mean

            if show:
                plt.figure()
                fig, plots = plt.subplots(1, 4)
                plots[0].imshow(img, cmap='gray')

            # 使用K均值将前景(放射不透明组织)和背景(放射透明组织，即肺部)分开
            # 仅在图像中心进行此操作，以尽可能避免图像的非组织部分
            # n_clusters 将预测结果分为几簇
            # np.prod(middle.shape): 横向依次相乘
            # 将middle reshape为  90000 1
            # kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))  # 90000 1
            # 利用kmeans将middle的文件分为两类
            kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))  # 90000 1
            centers = sorted(kmeans.cluster_centers_.flatten())
            threshold = np.mean(centers)  # 得到阈值
            thresh_img = np.where(img < threshold, 1.0, 0.0)  # 二值化

            if show:
                plots[1].imshow(thresh_img, cmap='gray')

            # 我发现最初的侵蚀有助于去除一些区域的颗粒状物质，
            # 然后大范围的扩张被用来使肺区吞噬血管，并被不透明的放射组织侵入肺腔。
            # 腐蚀，膨胀
            eroded = morphology.erosion(thresh_img, np.ones([4, 4]))  # 邻域
            dilation = morphology.dilation(eroded, np.ones([10, 10]))
            if show:
                plots[2].imshow(dilation, cmap='gray')

            # 标记每个区域并获取区域属性通过使用BBox移除区域来移除背景区域，该BBox在任一维度上都太大。
            # 此外，肺部通常远离图像的顶部和底部，所以任何太靠近顶部和底部的区域都会被移除。
            # 这不会从图像中产生完美的肺部分割，但考虑到其简单性，这是出人意料的好。
            labels = measure.label(dilation)  # 对每个区域进行分类
            # label_vals = np.unique(labels)  # 去除重复数字，排序输出，但是没用到
            regions = measure.regionprops(labels)
            good_labels = []
            for prop in regions:
                B = prop.bbox  # 得到label的边界，选出其中中心肺实质的部分
                if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
                    good_labels.append(prop.label)

            # 将肺实质部分重新填补到新的数组中，合并所有label，得到肺实质mask
            mask = np.ndarray([512, 512], dtype=np.int8)
            mask[:] = 0
            for N in good_labels:
                mask = mask + np.where(labels == N, 1, 0)
            mask = morphology.dilation(mask, np.ones([10, 10]))  # 膨胀
            imgs_to_process[i] = mask
            if show:
                plots[3].imshow(mask, cmap='gray')
                plt.show()
            # plt.figure()
            # fig, plots = plt.subplots(1, 3)
            # plots[0].imshow(img, cmap='gray')  # 原图
            # plots[1].imshow(mask, cmap='gray')  # mask
            # plots[2].imshow(mask * img, cmap='gray')  # 截取原图对应的mask
            # plt.show()
        # 将传入的images文件中的images替换为lungmask并保存文件
        np.save(img_file.replace("_origin_img_", "_lung_msk_"), imgs_to_process)


#    用mask和图像进行取值，得到感兴趣区域
def crop(file_list, show=False):
    print('---------possess on function : crop---------')
    os.makedirs(sec_step, exist_ok=True)
    for fname in tqdm(file_list):

        imgs_to_process = np.load(fname.replace("_lung_msk_", "_origin_img_"))  # 加载图片
        masks = np.load(fname)  # 肺实质mask
        node_masks = np.load(fname.replace("_lung_msk_", "_nodule_msk_"))  # 肺结节mask
        assert len(masks) == len(node_masks) == len(imgs_to_process)
        fname = fname.split('.')[-2].replace("_lung_msk_", "_")
        for i in range(len(imgs_to_process)):
            # 取出对应的mask图片
            mask = masks[i]
            node_mask = node_masks[i]
            img = imgs_to_process[i]
            new_size = [512, 512]  # 缩放到图像的原始大小
            img = mask * img  # 得到切割后的肺部图像

            if show:
                plt.figure()
                fig, plots = plt.subplots(1, 4)
                plots[0].imshow(node_mask, cmap='gray')
                plots[1].imshow(img, cmap='gray')

            # 重新规格化mask图像(在mask区域)
            # todo plan 1 do nothing
            # if show:
            #     plots[1].imshow(img, cmap='gray')

            # todo plan 2
            new_mean = np.mean(img[mask > 0])
            new_std = np.std(img[mask > 0])
            old_min = np.min(img)  # 背景颜色
            img[img == old_min] = new_mean - 1.2 * new_std  # 重新设置背景颜色
            # 归一化
            img = img - new_mean
            img = img / new_std

            # 制作图像的边界框(min row, min col, max row, max col)
            labels = measure.label(mask)
            regions = measure.regionprops(labels)

            # 求整个图片所有区域的的最小和最大值
            min_row = 512
            max_row = 0
            min_col = 512
            max_col = 0
            for prop in regions:
                B = prop.bbox
                if min_row > B[0]:
                    min_row = B[0]
                if min_col > B[1]:
                    min_col = B[1]
                if max_row < B[2]:
                    max_row = B[2]
                if max_col < B[3]:
                    max_col = B[3]

            width = max_col - min_col
            height = max_row - min_row
            # 向外扩展一行？
            if width > height:
                max_row = min_row + width
            else:
                max_col = min_col + height

            # 裁剪图片到边界框大小
            img = img[min_row:max_row, min_col:max_col]
            mask = mask[min_row:max_row, min_col:max_col]

            if max_row - min_row < 5 or max_col - min_col < 5:  # 跳过该图片，因为没有好的区域
                pass
            else:
                # todo plan 1
                # img = lumTrans(img)
                # moving range to -1 to 1 to accomodate the resize function
                # todo plan 2
                mean = np.mean(img)
                img = img - mean
                min = np.min(img)
                max = np.max(img)
                img = img / (max - min)

                new_img = resize(img, new_size)
                mask = resize(mask, new_size)
                new_node_mask = resize(node_mask[min_row:max_row, min_col:max_col], new_size)
                # todo plan 1
                # new_node_mask[new_node_mask > 0] = 1.0
                # 使用多种对比度增强的方法，得到结果求平均
                # contrast_img = new_node_mask * new_img
                #
                # a = 4  # 线性变换
                # y = a * contrast_img
                # y[y > 255] = 255.
                # img_bright = y.astype(np.uint8)
                # img_bright = img_bright ** 1.5  # 指数变换，提高黑色部分亮度
                #
                # new_node_mask = morphology.erosion(img_bright, np.ones([3, 3]))  # 邻域
                # new_node_mask = morphology.dilation(new_node_mask, np.ones([6, 6]))  # 膨胀
                # new_node_mask[new_node_mask > 0] = 1.
                # todo
                # plt.figure()
                # fig, plots = plt.subplots(1, 3)
                # plots[0].imshow(new_img, cmap='gray')  # 原图
                # plots[1].imshow(new_node_mask, cmap='gray')  # mask
                # plots[2].imshow(mask, cmap='gray')  # 截取原图对应的mask
                # plt.show()
                np.save(os.path.join(sec_step, fname + '_img_' + str(i) + '_.npy'), new_img[np.newaxis, ...])
                np.save(os.path.join(sec_step, fname + '_node_' + str(i) + '_.npy'), new_node_mask[np.newaxis, ...])
                np.save(os.path.join(sec_step, fname + '_lung_' + str(i) + '_.npy'), mask[np.newaxis, ...])


def make_mask(center, diam, z, width, height, spacing, origin):
    '''
        Center : centers of circles px -- list of coordinates x,y,z
        diam : diameters of circles px -- diameter
        widthXheight : pixel dim of image
        spacing = mm/px conversion rate np array x,y,z
        origin = x,y,z mm np.array
        z = z position of slice in world coordinates mm 切片在世界坐标的位置
    '''
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

    return mask

    # Helper function to get rows in data frame associated with each file


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


def deal_nodule(show=False):
    # 获取文件列表
    path = r'/zsm/xu/'  # linuxConfig['base_path']

    if not os.path.exists(first_step):
        os.makedirs(first_step)
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

                    # just keep 3 slices,只选择最大的靠近结节中心的切片
                    imgs = np.ndarray([3, height, width], dtype=np.float32)
                    masks = np.ndarray([3, height, width], dtype=np.uint8)

                    center = np.array([node_x, node_y, node_z])  # 世界坐标，肺结节中心
                    # nodule center in voxel space (still x,y,z ordering)
                    v_center = np.rint((center - origin) / spacing)  # 体素坐标，取整
                    # 结节中心v_center[2],取结节上一层和下一层的横切面，切片取值范围为[0,numz-1]，防止切片越界
                    for i, i_z in enumerate(np.arange(int(v_center[2]) - 1, int(v_center[2]) + 2).clip(0, num_z - 1)):
                        # 制作mask
                        # i_z * spacing[2] + origin[2]   -313 z:体素坐标*间距+原始z
                        nomal_z = i_z * spacing[2] + origin[2]
                        mask = make_mask(center, diam, nomal_z, width, height, spacing, origin)
                        # mask[mask > 0.] = 1.0
                        masks[i] = mask
                        imgs[i] = img_array[i_z]
                        # 展示效果
                        # if show:
                        # plt.figure()
                        # fig, plots = plt.subplots(1, 3)
                        # plots[0].imshow(img_array[i_z], cmap='gray')  # 原图
                        # plots[1].imshow(mask, cmap='gray')  # mask
                        # plots[2].imshow(mask * img_array[i_z], cmap='gray')  # 截取原图对应的mask
                        # plt.show()
                    img_file_name = img_file.split('/')[-1][:-4]
                    np.save(os.path.join(first_step, img_file_name + '_origin_img_' + str(node_idx)), imgs)
                    np.save(os.path.join(first_step, img_file_name + '_nodule_msk_' + str(node_idx)), masks)


def lung_and_nodule_possess(show=False):
    deal_nodule(show)  # 处理结节
    # file_list_image = glob(first_step + "*_origin_img_*.npy")
    # segmentation(file_list_image, show)
    file_list_mask = glob(first_step + "*_lung_msk_*.npy")
    crop(file_list_mask, show)


if __name__ == '__main__':
    lung_and_nodule_possess()
