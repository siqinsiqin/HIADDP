"""
# Time       ：2022/5/18 9:22
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os
import sys
from functools import partial
from multiprocessing import Pool

from DatasetPrepare.LunaPrepareV2 import FindLunaNoduleV2

# # todo 无法运行cmd时，取消注释下一行

sys.path.append(os.pardir)  # 环境变量

import numpy as np
import pandas as pd
import pylidc as pl

from configs import config


class LidcPrepareV2(FindLunaNoduleV2):

    def __init__(self, mode):
        super(LidcPrepareV2, self).__init__('-1')
        self.dataset = 'lidc'
        self.dirInit()

        pool = Pool()  # 开启线程池
        func = partial(self.main, mode=mode)
        N = 34  # 线程数 34
        _ = pool.map(func, range(N))
        pool.close()  # 关闭线程池
        pool.join()
        # self.main(-1)

    def dirInit(self):
        self.seg_path3d = config.seg_path_lidc_3d
        self.seg_path2d = config.seg_path_lidc_2d

        self.seg_path3d = '/zsm/jwj/baseExpV7/$segmentation/mthickimg3d/'
        self.seg_path2d = '/zsm/jwj/baseExpV7/$segmentation/mthickimg2d/'

        os.makedirs(self.seg_path2d, exist_ok=True)
        os.makedirs(self.seg_path3d, exist_ok=True)
        print(f'dir init in {self.seg_path2d},{self.seg_path3d}')

    def inList(self, data, item):
        # for serious in np.unique(data['seriesuid']):
        #     if item == serious:
        #         return True
        # return False

        if item in np.unique(data['seriesuid']):
            return True
        return False

    def checkFalse(self, node):
        existFalse = False
        for nod in node:  # 遍历lidc中所有的标注，查看是否出错
            if nod.subtlety in range(1, 6) and nod.internalStructure in range(1, 5) and nod.calcification in range(1, 7) \
                    and nod.sphericity in range(1, 6) and nod.margin in range(1, 6) and nod.lobulation in range(1, 6) \
                    and nod.spiculation in range(1, 6) and nod.texture in range(1, 6):
                continue
            else:
                existFalse = True
                break
        return existFalse

    def checknodes(self, node):
        pointx = []
        pointy = []
        pointz = []
        for nod in node:  # todo 计算 所有注解平均值
            pointx.append(nod.centroid[0])
            pointy.append(nod.centroid[1])
            pointz.append(nod.centroid[2])

        # 定义允许的最大差异值
        max_difference = 5

        # 将坐标列表转换为NumPy数组
        x_coords = np.array(pointx)
        y_coords = np.array(pointy)
        z_coords = np.array(pointz)

        # 计算坐标的差异
        x_diff = np.abs(x_coords - np.mean(x_coords))
        y_diff = np.abs(y_coords - np.mean(y_coords))
        z_diff = np.abs(z_coords - np.mean(z_coords))

        # 创建一个掩码来标记差异过大的点
        mask = np.logical_or(x_diff > max_difference,
                             np.logical_or(y_diff > max_difference, z_diff > max_difference))

        # 根据掩码过滤掉差异过大的点
        filtered_x_coords = x_coords[~mask]
        filtered_y_coords = y_coords[~mask]
        filtered_z_coords = z_coords[~mask]

        new_node = []
        for nod in node:
            for i in range(len(filtered_x_coords)):
                if nod.centroid[0] == filtered_x_coords[i] and nod.centroid[1] == filtered_y_coords[
                    i] and nod.centroid[2] == filtered_z_coords[i]:
                    new_node.append(nod)
        return new_node

    def main(self, idx, mode=None):

        # CTs = pl.query(pl.Scan).filter(
        #     pl.Scan.series_instance_uid == '1.3.6.1.4.1.14519.5.2.1.6279.6001.161855583909753609742728521805').all()
        CTs = pl.query(pl.Scan).all()
        data = pd.read_csv(f'{config.csv_path}/annotations.csv', header=0)  # 加载luna16结节信息
        mhd_origin_spacing = pd.read_csv(f'{config.csv_path}/MhdOriginAndSpaving.csv', header=0)
        orgins_count = self.findAllRow(mhd_origin_spacing)
        count = self.findAllRow(data)

        # 多线程并发
        attrs = []
        size = 30
        start = size * idx
        end = start + size
        if idx == 33:
            end = None
        if idx == -1:
            start = 0
            end = None

        print(f'Thread : {idx}, start {start} to {end}，{len(CTs[start:end]) == size},{len(CTs[start:end])}')
        for ct in CTs[start:end]:
            """
            1. 如果 series_instance_uid 在luna16中，排除luna16的结节，将剩余结节进行划分
            2. 如果不在，直接聚类，输出结果
            """
            nods = ct.cluster_annotations()
            slice_thickness = ct.slice_thickness
            spacing = ct.spacings
            pixel_spacing = ct.pixel_spacing
            patient_id = ct.patient_id

            item = ct.series_instance_uid

            if self.inList(data, item):  # todo luna
                # print('inlist')
                citem = count[item]
                idx = orgins_count[item][0]
                mhdOrigin = np.array([mhd_origin_spacing['originX'][idx], mhd_origin_spacing['originY'][idx],
                                      mhd_origin_spacing['originZ'][idx]], dtype=np.float64)
                mhdSpacing = np.array([mhd_origin_spacing['spacingX'][idx], mhd_origin_spacing['spacingY'][idx],
                                       mhd_origin_spacing['spacingZ'][idx]], dtype=np.float64)
                # plan 1
                for node in nods:
                    pointx = []
                    pointy = []
                    pointz = []
                    for nod in node:  # todo 计算 所有注解平均值
                        pointx.append(nod.centroid[0])
                        pointy.append(nod.centroid[1])
                        pointz.append(nod.centroid[2])
                    centroid = [np.average(pointx), np.average(pointy), np.average(pointz)]
                    for row in citem:  # 统计id下的所有结节数量，这个luna数据集的判断
                        xyz = np.array([data['coordX'][row], data['coordY'][row], data['coordZ'][row]],
                                       dtype=np.float64)
                        diameater = np.floor(data['diameter_mm'][row])
                        xyz = np.round(self.worldToVoxelCoord(xyz[::-1], mhdOrigin, mhdSpacing)[::-1], 2)
                        x, y, z = xyz[0], xyz[1], xyz[2]
                        v1 = np.round(centroid[0], 2)
                        v2 = np.round(centroid[1], 2)
                        v3 = np.round(centroid[2], 2)
                        diffz = abs(z - v3)
                        diffy = abs(y - v1)
                        diffx = abs(x - v2)

                        bias = 5.

                        if diffx <= bias and diffy <= bias and diffz <= bias:
                            # 匹配成功,删除点
                            citem = [i for i in citem if i != row]
                            nods = [nod for nod in nods if nod != node]

                for node in nods:  # todo 将剩余点进行生成
                    # 检查是否标注出错，如果存在标注出错，直接排除
                    # if self.checkFalse(node) or len(node) < 2:
                    #     continue
                    try:
                        if self.checkFalse(node) or len(node) < 3:  # 只保存标注为1的
                            continue
                        result, noduleSize, avgDiameter, features, centroid = self.mix(node, None,
                                                                                       UseLunaDiameter=False)
                        # todo 保存结节信息
                        attrs.append(
                            {'seriesuid': item, 'centroidX': centroid[0], 'centroidY': centroid[1],
                             'centroidZ': centroid[2],
                             'diameter': avgDiameter, 'subtlety': features[0], 'internalStructure': features[1],
                             'calcification': features[2], 'sphericity': features[3], 'margin': features[4],
                             'lobulation': features[5], 'spiculation': features[6], 'texture': features[7],
                             'malignancy': features[8], 'spacingX': spacing[0], 'spacingY': spacing[1],
                             'spacingZ': spacing[2],
                             'slice_thickness': slice_thickness, 'pixel_spacing': pixel_spacing,
                             'patient_id': patient_id, })

                        # todo 保存结节
                        self.save(item, node, [result, noduleSize], ct)
                    except Exception as e:
                        print(e.args)

            else:
                # plan 2
                # todo 保存不同掩码均值属性,获取该结节的相关属性信息
                for node in nods:
                    # 检查是否标注出错，如果存在标注出错，直接排除
                    # if self.checkFalse(node) or len(node) < 2:  # 排除标注数量少于2的
                    #     continue

                    try:
                        if self.checkFalse(node) or len(node) < 3:  # 只保存标注为1的
                            continue
                        result, noduleSize, avgDiameter, features, centroid = self.mix(node, None,
                                                                                       UseLunaDiameter=False)

                        attrs.append(
                            {'seriesuid': item, 'centroidX': centroid[0], 'centroidY': centroid[1],
                             'centroidZ': centroid[2],
                             'diameter': avgDiameter, 'subtlety': features[0], 'internalStructure': features[1],
                             'calcification': features[2], 'sphericity': features[3], 'margin': features[4],
                             'lobulation': features[5], 'spiculation': features[6], 'texture': features[7],
                             'malignancy': features[8], 'spacingX': spacing[0], 'spacingY': spacing[1],
                             'spacingZ': spacing[2], 'slice_thickness': slice_thickness, 'pixel_spacing': pixel_spacing,
                             'patient_id': patient_id, })

                        # todo 保存结节

                        self.save(item, node, [result, noduleSize], ct)
                    except Exception as e:
                        print(e.args)

        # if mode is None:
        #     """保存结节属性"""
        #     df = pd.DataFrame(attrs)
        #     df.to_csv(f'{config.csv_path}/all_lidc_nodules_info.csv')
        #     print('Save lidc annotation csv !!!')

        print("end lidc")


if __name__ == '__main__':
    """
    远程ssh命令
    conda activate jwj
    cd /zsm/jwj/baseExpV5/DatasetPrepare/
    cd /zljteam/jwj/baseExpV5/DatasetPrepare/
    nohup python LidcPrepare.py >/dev/null 2>&1 &
    nohup python LidcPrepare.py  # 无法后台执行时进行此命令，查看错误记录
    """

    # todo zlteam
    # 3d 156578
    # 2d

    # todo 2d 3d 必须分开跑
    mode = config.mode
    LidcPrepareV2(mode).to(config.device)
