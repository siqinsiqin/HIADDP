# -*-coding:utf-8 -*-
"""
# Time       ：2023/9/20 16:15
# Author     ：comi
# version    ：python 3.8
# Description：
# todo 1：按照层厚统计不同医生标注个数
"""

from __future__ import absolute_import

import multiprocessing
import os
import random
import sys

# todo 无法运行cmd时，取消注释下一行
from utils.logger import logs
from utils.resampleV2 import uniform_cubic_resample

sys.path.append(os.pardir)  # 环境变量
import time
from functools import partial
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
import pandas as pd
from torch import nn

from configs import config
from utils.helper import showTime, getAllAttrs
import pylidc as pl
from pylidc.utils import consensus


class FindLunaNoduleV2(nn.Module):
    """
    通过结节点xyz与lidc的结节进行对比,依次找到对应结节
    """
    seg_path3d = None
    seg_path2d = None
    dataset = 'luna'

    def dirInit(self):
        self.seg_path3d = config.seg_path_luna_3d
        self.seg_path2d = config.seg_path_luna_2d

        os.makedirs(self.seg_path2d, exist_ok=True)
        os.makedirs(self.seg_path3d, exist_ok=True)
        print(f'dir init in {self.seg_path2d},{self.seg_path3d}')

    @classmethod
    def findAllRow(cls, data):
        # todo 根据对应csv文件找出对应的idx
        data = data.groupby('seriesuid').apply(
            lambda d: tuple(d.index) if len(d.index) > 0 else None
        ).dropna()
        return data

    @classmethod
    def load_itk_image_simple(cls, filename):
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

    @classmethod
    def worldToVoxelCoord(cls, worldCoord, origin, spacing):
        stretchedVoxelCoord = np.absolute(worldCoord - origin)
        voxelCoord = stretchedVoxelCoord / spacing
        return voxelCoord

    def findmax(self, imgs, masks):
        # todo 找出2d mask中结节最大的一个
        imgs = np.array(imgs)
        masks = np.array(masks)

        cnt = []
        for t in range(imgs.shape[2]):
            n = np.count_nonzero(masks[:, :, t])
            cnt.append(n)

        idx = self.maxidx(cnt)
        return imgs[:, :, idx], masks[:, :, idx], idx

    @classmethod
    def maxidx(cls, a):
        a = np.array(a)
        return np.where(a == np.max(a))[0][0]

    @classmethod
    def lumTrans(cls, img):
        """
        截断归一化，只关注兴趣区域
        """
        lungwin = np.array([-1000., 400.])
        newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        # newimg = (newimg * 255).astype('uint8')
        return newimg

    @classmethod
    def vote(cls, values):
        """
        通过投票决定四个医生的标注属性的结节最终属性
        """
        cnt = dict()
        for val in values:
            if str(val) in cnt.keys():
                cnt.update({f'{val}': cnt.get(f'{val}') + 1})
            else:
                cnt.update({f'{val}': 1})

        # todo 检查是否有相同大小的值
        highest = max(cnt.values())
        idxs = [int(k) for k, v in cnt.items() if v == highest]
        if len(idxs) == 1:
            return idxs[0] - 1, idxs[0]
        else:
            return int(np.floor(np.median(idxs))) - 1, int(np.floor(np.median(idxs)))

    def voteAllAttr(self, arrs, features):
        labels = getAllAttrs()
        result = []
        for i, arr in enumerate(arrs):
            labelIdx, doctorIdx = self.vote(arr)
            features[i] = doctorIdx
            result.append(labels[i][labelIdx])
        return result, features

    def mix(self, anns, lunaDiameters, UseLunaDiameter=True):
        """
        良恶性判断使用中位数判断，大于3为恶性，等于3为不确定，小于3为良性
        直径：不同掩码直径均值。luna实际测量值
        结节大小分类：
            实性结节(Solid)：3~6，6~8，>8
            半实性结节(Subsolid Nodules)：3~6，>=6
        其余属性均进行投票
        """
        features_name = ('subtlety',
                         'internalStructure',
                         'calcification',
                         'sphericity',
                         'margin',
                         'lobulation',
                         'spiculation',
                         'texture',
                         'malignancy')
        diameters = []
        subtlety = []
        internalStructure = []
        calcification = []
        sphericity = []
        margin = []
        lobulation = []
        spiculation = []
        texture = []
        malignancy = []
        centroid = np.zeros(3)
        features = np.zeros(9)

        for ann in anns:
            # attr
            subtlety.append(ann.subtlety)
            internalStructure.append(ann.internalStructure)
            calcification.append(ann.calcification)
            sphericity.append(ann.sphericity)
            margin.append(ann.margin)
            lobulation.append(ann.lobulation)
            spiculation.append(ann.spiculation)
            texture.append(ann.texture)
            malignancy.append(ann.malignancy)
            diameters.append(ann.diameter)
            centroid += ann.centroid

        centroid /= len(anns)
        arrs = [subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture]
        result, features = self.voteAllAttr(arrs, features)

        """
        使用luna结节直径进行判断结节大小
        """
        if UseLunaDiameter:
            # 使用luna的直径
            avgDiameter = lunaDiameters
        else:
            # 使用官方预估的直径
            avgDiameter = np.average(diameters)

        noduleSize = 'unassignment'
        if result[7] in ['NonSolidGGO', 'NonSolidMixed', 'PartSolidMixed']:  # 改变结节名称
            # result[7] = 'subsolid'  # 部分实性
            if 3. <= avgDiameter < 6:
                noduleSize = 'sub36'
            elif avgDiameter >= 6:
                noduleSize = 'sub6p'
            elif avgDiameter < 3.:
                noduleSize = 'sub3c'
        elif result[7] in ['SolidMixed', 'tSolid']:
            # result[7] = 'tSolid'  # 实性
            if 3. <= avgDiameter < 6.:
                noduleSize = 'solid36'
            elif 6. <= avgDiameter <= 8.:
                noduleSize = 'solid68'
            elif avgDiameter > 8:
                noduleSize = 'solid8p'
            elif avgDiameter < 3.:
                noduleSize = 'solid3c'
        else:
            logs(f'error {result[7]}')

        """
        中位数决定结节良恶性
        """
        malignancy.sort()
        malignancy = np.round(np.median(malignancy), 2)
        features[8] = malignancy
        if malignancy < 3:
            malignancy = 'benign'
        elif malignancy == 3:
            malignancy = 'uncertain'
        else:
            malignancy = 'malignant'
        result.append(malignancy)

        return result, noduleSize, avgDiameter, features, centroid

    def main(self, idx, mode=None):
        # data = pd.read_csv(f'/zsm/jwj/baseExp/DatasetPrepare/annotations/annotations{idx}.csv', header=0)  # 加载luna16结节信息
        data = pd.read_csv(f'{config.csv_path}/annotations.csv', header=0)  # 加载luna16结节信息
        mhd_origin_spacing = pd.read_csv(f'{config.csv_path}/MhdOriginAndSpaving.csv', header=0)

        count = self.findAllRow(data)  # 统计id下的所有结节数量
        orgins_count = self.findAllRow(mhd_origin_spacing)
        a, b, c, d, e, f, g, h, ii, jj, kk = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        attrs = []
        # todo 对相应点的结节进行掩码聚类
        all_nodules = []
        # 多线程并发
        size = 18
        start = size * idx
        end = start + size
        if idx == 33:
            end = None
        if idx == -1:
            start = 0
            end = None

        print(f'Thread : {idx}, start {start} to {end}')
        for i, item in (enumerate(np.unique(data['seriesuid'])[start:end])):
            if i >= 601:
                break

            anns = pl.query(pl.Annotation).join(pl.Scan).filter(pl.Scan.series_instance_uid == item).all()
            CT = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == item).first()

            spacing = CT.spacings
            thickness = CT.slice_thickness
            pixel_spacing = CT.pixel_spacing
            patient_id = CT.patient_id

            ann_ex = []
            idx = orgins_count[item][0]
            mhdOrigin = np.array([mhd_origin_spacing['originX'][idx], mhd_origin_spacing['originY'][idx],
                                  mhd_origin_spacing['originZ'][idx]], dtype=np.float64)
            mhdSpacing = np.array([mhd_origin_spacing['spacingX'][idx], mhd_origin_spacing['spacingY'][idx],
                                   mhd_origin_spacing['spacingZ'][idx]], dtype=np.float64)
            for row in count[item]:
                xyz = np.array([data['coordX'][row], data['coordY'][row], data['coordZ'][row]], dtype=np.float64)
                xyz = np.round(self.worldToVoxelCoord(xyz[::-1], mhdOrigin, mhdSpacing)[::-1], 2)
                x, y, z = xyz[0], xyz[1], xyz[2]

                one_nodule = []
                """遍历serious id中所有掩膜,比较xyz,将在bias内的结节进行合并"""
                for k, ann in enumerate(anns):
                    if k not in ann_ex:
                        v1 = np.round(ann.centroid[0], 2)
                        v2 = np.round(ann.centroid[1], 2)
                        v3 = np.round(ann.centroid[2], 2)
                        diffz = abs(z - v3)
                        diffy = abs(y - v1)
                        diffx = abs(x - v2)
                        bias = 5.  # 中心位置偏差
                        if len(anns) == 3 or item in [  # 特殊结节
                            '1.3.6.1.4.1.14519.5.2.1.6279.6001.321935195060268166151738328001',
                            '1.3.6.1.4.1.14519.5.2.1.6279.6001.286422846896797433168187085942'
                        ]:
                            if diffx <= bias * 2 and diffy <= bias * 2 and diffz < 8.8:  # 一张切片只有三个标注，可能都是
                                ann_ex.append(k)
                                one_nodule.append(ann)
                        else:
                            """包含三层切片，如果还取不到，则认为没有同一个结节"""
                            if diffx <= bias and diffy <= bias and diffz < bias:
                                ann_ex.append(k)
                                one_nodule.append(ann)

                if len(one_nodule) == 3:
                    if thickness == 0.6:
                        a += 1
                    elif thickness == 0.75:
                        b += 1
                    elif thickness == 0.9:
                        c += 1
                    elif thickness == 1:
                        d += 1
                    elif thickness == 1.25:
                        e += 1
                    elif thickness == 1.5:
                        f += 1
                    elif thickness == 2.:
                        g += 1
                    elif thickness == 2.5:
                        h += 1
                    elif thickness == 3:
                        ii += 1
                    elif thickness == 4:
                        jj += 1
                    elif thickness == 5:
                        kk += 1
                    else:
                        print('error:', thickness)
                        continue
                elif len(one_nodule) == 4:
                    if thickness == 0.6:
                        a1 += 1
                    elif thickness == 0.75:
                        b1 += 1
                    elif thickness == 0.9:
                        c1 += 1
                    elif thickness == 1:
                        d1 += 1
                    elif thickness == 1.25:
                        e1 += 1
                    elif thickness == 1.5:
                        f1 += 1
                    elif thickness == 2.:
                        g1 += 1
                    elif thickness == 2.5:
                        h1 += 1
                    elif thickness == 3:
                        i1 += 1
                    elif thickness == 4:
                        j1 += 1
                    elif thickness == 5:
                        k1 += 1
                    else:
                        print('error:', thickness)
                        continue

            # todo 保存不同掩码均值属性,获取该结节的相关属性信息
            result, noduleSize, avgDiameter, features, centroid = self.mix(one_nodule, data['diameter_mm'][row])
            
            attrs.append(
                {'seriesuid': item, 'centroidX': centroid[0], 'centroidY': centroid[1], 'centroidZ': centroid[2],
                 'diameter': avgDiameter, 'subtlety': features[0], 'internalStructure': features[1],
                 'calcification': features[2], 'sphericity': features[3], 'margin': features[4],
                 'lobulation': features[5], 'spiculation': features[6], 'texture': features[7],
                 'malignancy': features[8], 'spacingX': spacing[0], 'spacingY': spacing[1], 'spacingZ': spacing[2],
                 'slice_thickness': thickness, 'pixel_spacing': pixel_spacing, 'patient_id': patient_id, })
            
            # todo 保存结节
            
            self.save(item, one_nodule, [result, noduleSize], CT)
            
            # todo 检查是否有遗漏
            if len(one_nodule) != 0:
                if len(one_nodule) < 3:
                    print(f'nodule len:{len(one_nodule)},anns:{len(anns)},', item, one_nodule)
                all_nodules.append(one_nodule)
            else:
                print('none', item)

        print(a, +b, +c, +d, +e, +f, +g, +h, +ii, +jj, +kk, )
        print(a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, )

        # if mode is None:
        #     """保存结节属性"""
        #     df = pd.DataFrame(attrs)
        #     df.to_csv(f'{config.csv_path}/all_luna_nodules_info.csv')
        #     print('Save luna annotation csv !!!')
        #     # todo 统计信息
        #     print('total', len(all_nodules))
        #     print(all_nodules)

    def save(self, name, nodules, attrs, CT):

        spacing = CT.slice_spacing
        pixel_spacing = CT.pixel_spacing
        slice_zvals = CT.slice_zvals

        # 单个结节重采样
        imgs, masks = nodules[0].uniform_cubic_resample(side_length=63)
        # fig, ax = plt.subplots(1, 2)
        # for t in range(25, 35):
        #     ax[0].imshow(imgs[:, :, t], cmap=plt.cm.gray)
        #     ax[0].axis('off')
        #     ax[1].imshow(masks[:, :, t] * imgs[:, :, t], cmap=plt.cm.gray)
        #     # ax[1].imshow(masks[:, :, t], cmap=plt.cm.gray)
        #     ax[1].axis('off')
        #     plt.tight_layout()
        #     # plt.show()
        #     os.makedirs(config.pic_path, exist_ok=True)
        #     plt.savefig(config.pic_path + f'pre_{t}.png', bbox_inches="tight", pad_inches=0)
        # plt.close()

        # todo 求掩膜均值
        try:
            masks, bbox, _ = consensus(nodules, clevel=0.5)

            imgs, masks = uniform_cubic_resample(side_length=63, bbox=bbox, bbox_dims=nodules[0].bbox_dims(),
                                                 pixel_spacing=pixel_spacing, slice_spacing=spacing,
                                                 slice_zvals=slice_zvals, mask=masks, scan=CT)

            # sigma = (1., 1, 1.)
            # truncate = 4.0
            #
            # # 使用ndimage的高斯滤波函数对图像进行平滑
            # imgs = ndimage.gaussian_filter(imgs, sigma=sigma, order=0, output=None, mode='reflect', cval=0.0,
            #                                truncate=truncate)

            # structuring_element = ndimage.generate_binary_structure(3, 1)
            #
            # # 对掩码进行形态学开运算
            # masks = ndimage.binary_opening(masks, structuring_element, iterations=1)  # 一般是1-3

            # fig, ax = plt.subplots(1, 2)
            # for t in range(25, 35):
            #     ax[0].imshow(imgs[:, :, t], cmap=plt.cm.gray)
            #     ax[0].axis('off')
            #     ax[1].imshow(masks[:, :, t] * imgs[:, :, t], cmap=plt.cm.gray)
            #     # ax[1].imshow(masks[:, :, t], cmap=plt.cm.gray)
            #     ax[1].axis('off')
            #     plt.tight_layout()
            #     # plt.show()
            #     os.makedirs(config.pic_path, exist_ok=True)
            #     plt.savefig(config.pic_path + f'single_{t}.png', bbox_inches="tight", pad_inches=0)
            # plt.close()

            # 2d 处理
            img_2d, msk_2d, _ = self.findmax(imgs, masks)

        except Exception as e:

            print(nodules)
            print(e.args)
            print(name, '====error====')

        # todo 过滤插值失败的img
        # imgs, masks = fliter(imgs, masks)

        # imgs = img_crop_or_fill(imgs, '3d')
        # masks = img_crop_or_fill(masks, '3d')
        # assert False
        while True:
            randomN = random.randint(0, 100)
            if self.filename == 'npy':
                suffix = f'{name}_{attrs[1]}_{attrs[0][0]}_{attrs[0][1]}_{attrs[0][2]}_{attrs[0][3]}' \
                         f'_{attrs[0][4]}_{attrs[0][5]}_{attrs[0][6]}_{attrs[0][7]}_{attrs[0][8]}_{randomN}.npy'
                lesion_name = self.seg_path3d + suffix

            else:
                suffix = f'{name}_{attrs[1]}_{attrs[0][0]}_{attrs[0][1]}_{attrs[0][2]}_{attrs[0][3]}' \
                         f'_{attrs[0][4]}_{attrs[0][5]}_{attrs[0][6]}_{attrs[0][7]}_{attrs[0][8]}_{randomN}.nii.gz'
                lesion_name = self.seg_path3d + suffix
            if not os.path.exists(lesion_name):
                break
            print('exist')

        lesion3d = np.concatenate((imgs[np.newaxis, ...], masks[np.newaxis, ...]))
        lesion2d = np.concatenate((img_2d[np.newaxis, ...], msk_2d[np.newaxis, ...]))

        lesion2dname = self.seg_path2d + suffix

        if self.filename == 'npy':
            np.save(lesion_name, lesion3d)
            np.save(lesion2dname, lesion2d)
        else:
            out = sitk.GetImageFromArray(lesion_name)
            sitk.WriteImage(out, lesion_name)
            out = sitk.GetImageFromArray(lesion2dname)
            sitk.WriteImage(out, lesion2dname)

        print(name)

    def __init__(self, mode=None, filename='npy'):
        super(FindLunaNoduleV2, self).__init__()
        self.padValue = 0  # 4  #
        self.filename = filename

        if mode == '-1':
            logs(f'lidc prepare {mode}')
        else:  # 非LIDC数据集
            # logs(f'luna prepare {mode}')
            self.dirInit()
            if mode is not None:
                pool = Pool(multiprocessing.cpu_count())  # 开启线程池
                func = partial(self.main, mode=mode)
                N = 1  # 线程数-34
                _ = pool.map(func, range(N))
                pool.close()  # 关闭线程池
                pool.join()
            else:  # 统计信息
                self.main(-1, mode)
            # logs(f'end luna prepare {mode}')


if __name__ == '__main__':
    """
    todo 1:统计结节信息   mode=None
    todo 2：保存2d最大横截面  mode=2d
    todo 3：保存3d 结节块     mode=3d
    """
    """
    远程ssh命令
    conda activate jwj
    cd /zsm/jwj/baseExpV5/DatasetPrepare/
    cd /zljteam/jwj/baseExpV5/DatasetPrepare/
    nohup python LunaPrepare.py >/dev/null 2>&1 &
    nohup python LunaPrepare.py 
    """
    # zsm  182130 --- 中值滤波，高斯滤波
    # zlj  142123 --- 数据集默认重采样
    # filename = 'npy'
    # mode = '2d'
    # print(mode)
    # start_time = time.time()
    # FindLunaNoduleV2(mode).to('cuda:0')
    # end_time = time.time()
    # showTime('2d Total', start_time, end_time)

    filename = 'npy'
    mode = '3d'
    print(mode)
    start_time = time.time()
    FindLunaNoduleV2(mode).to('cuda:0')
    end_time = time.time()
    showTime('3d Total', start_time, end_time)
