# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os
from collections import OrderedDict
from glob import glob

import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import roberts
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.segmentation import clear_border
from skimage.transform import resize
from tqdm import tqdm

from configs import linuxConfig
from utils.helper import lumTrans

first_step = linuxConfig['first_step']
train_path = linuxConfig['train_path']
test_path = linuxConfig['test_path']
anno_path = linuxConfig['anno_path']

out_images = OrderedDict()  # final set of images
out_nodemasks = OrderedDict()  # final set of nodemasks
out_masks = OrderedDict()


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
    output_path = os.path.join('/zsm/jwj/baseExp/', '$LUNA16PROPOCESSPATH/first_step/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(2):

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
                        mask[mask > 0] = 1.0
                        masks[i] = mask
                        imgs[i] = img_array[i_z]
                        # 展示效果
                    img_file_name = img_file.split('/')[-1][:-4]
                    np.save(os.path.join(output_path, img_file_name + '_origin_img_' + str(node_idx)), imgs)
                    np.save(os.path.join(output_path, img_file_name + '_nodule_msk_' + str(node_idx)), masks)


def get_segmented_lungs(im):
    binary = im < -600

    cleared = clear_border(binary)
    label_image = label(cleared)
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:  # 如果其中的区域小于第二个最大的区域，将其坐标置为0
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    selem = disk(10)  # 生成扁平的盘形结构元素。如果像素之间的欧几里德距离，则像素在邻域内它和原点不大于半径。
    binary = binary_closing(binary, selem)
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    return im


def segmentation(file_list):
    print('---------possess on function : segmentation---------')
    for img_file in tqdm(file_list):

        imgs_to_process = np.load(img_file).astype(np.float64)

        for i in range(len(imgs_to_process)):  # 3张

            img = get_segmented_lungs(imgs_to_process[i])
            imgs_to_process[i] = img

        np.save(img_file.replace("_origin_img_", "_lung_msk_"), imgs_to_process)


def crop(file_list):
    print('---------possess on function : crop---------')
    t = 0
    for fname in tqdm(file_list):
        imgs_to_process = np.load(fname.replace("_lung_msk_", "_origin_img_"))  # 加载图片
        masks = np.load(fname)  # 肺实质mask
        node_masks = np.load(fname.replace("_lung_msk_", "_nodule_msk_"))  # 肺结节mask
        assert len(masks) == len(node_masks) == len(imgs_to_process)

        for i in range(len(imgs_to_process)):
            mask = masks[i]
            node_mask = node_masks[i]
            img = imgs_to_process[i]
            new_size = [512, 512]  # 缩放到图像的原始大小

            labels = label(mask)
            regions = regionprops(labels)

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
                new_img = resize(img, new_size)
                mask = resize(mask, new_size)
                new_img = lumTrans(new_img * mask)
                new_node_mask = resize(node_mask[min_row:max_row, min_col:max_col], new_size)

                out_images['' + str(t) + ''] = new_img
                out_nodemasks['' + str(t) + ''] = new_node_mask
                out_masks['' + str(t) + ''] = mask
                t += 1

    # 划分训练数据
    num_images = len(out_images)
    #  Writing out images and masks as 1 channel arrays for input into network
    assert len(out_masks) == len(out_nodemasks) == len(out_images)
    final_images = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
    final_nodule_masks = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
    final_lung_masks = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)

    for t in range(num_images):
        final_images[t, 0] = out_images['' + str(t) + '']  # out_images[i]
        final_nodule_masks[t, 0] = out_nodemasks['' + str(t) + '']  # out_nodemasks[i]
        final_lung_masks[t, 0] = out_masks['' + str(t) + '']  # out_masks[i]

        # 划分训练和测试部分
    rand_i = np.random.choice(range(num_images), size=num_images, replace=False)
    test_i = int(0.1 * num_images)
    print('num:', num_images, 'test:', test_i)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # 测试数据集存储
    print("save test")
    for i in tqdm(range(test_i)):
        np.save(os.path.join(test_path, 'test_img_' + str(i) + '_.npy'), final_images[rand_i[i]])
        np.save(os.path.join(test_path, 'test_nodule_msk_' + str(i) + '_.npy'), final_nodule_masks[rand_i[i]])
        np.save(os.path.join(test_path, 'test_lung_msk_' + str(i) + '_.npy'), final_lung_masks[rand_i[i]])
    # 训练数据集 存储
    print("save luna")
    for i in tqdm(range(test_i, num_images)):
        np.save(os.path.join(train_path, 'train_img_' + str(i) + '_.npy'), final_images[rand_i[i]])
        np.save(os.path.join(train_path, 'train_nodule_msk_' + str(i) + '_.npy'), final_nodule_masks[rand_i[i]])
        np.save(os.path.join(train_path, 'train_lung_msk_' + str(i) + '_.npy'), final_lung_masks[rand_i[i]])
    print('end save')


def lung_and_nodule_possess(show=False):
    # deal_nodule(show)  # 处理结节
    file_list_image = glob(first_step + "*_origin_img_*.npy")
    segmentation(file_list_image)
    file_list_mask = glob(first_step + "*_lung_msk_*.npy")
    crop(file_list_mask)


if __name__ == '__main__':
    lung_and_nodule_possess()
