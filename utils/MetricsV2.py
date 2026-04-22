# # -*-coding:utf-8 -*-
# """
# # Time       ：2022/7/11 9:34
# # Author     ：comi
# # version    ：python 3.8
# # Description：
# """
#
# import numpy as np
# import torch
# import torchmetrics
# from mindspore import Tensor
# from mindspore.nn import HausdorffDistance, MeanSurfaceDistance
# from torch import nn
#
# from utils.helper import avgStd
# from utils.logger import logs
#
#
# class MetricsV2(nn.Module):
#
#     def __init__(self):
#         super(MetricsV2, self).__init__()
#         self.HD = []
#         self.DSC = []
#         self.PPV = []
#         self.MSD = []
#
#     def __call__(self, pred, msk, index=1):
#         """将msk，pred保存为文件然后测量距离"""
#         # todo 1
#         pred = pred.type(torch.FloatTensor)
#         msk = msk.type(torch.IntTensor)
#         self.DSC.append(self.numDeal(torchmetrics.functional.f1_score(pred, msk).data, self.DSC, name='dice'))
#
#         # mind spore eva
#         pred = Tensor(np.array(pred.cpu().numpy()))
#         msk = Tensor(np.array(msk.cpu().numpy()))
#
#         metric = HausdorffDistance(directed=True, percentile=95.)
#         metric.clear()
#         metric.update(pred, msk, index)
#         distance = metric.eval()
#         # print('hd: ', distance)
#         self.HD.append(self.numDeal(distance, self.HD, name='hd'), )
#
#         metric = MeanSurfaceDistance()
#         metric.clear()
#         metric.update(pred, msk, index)
#         mean_average_distance = metric.eval()  # 平均表面距离
#         print('msd: ', mean_average_distance)
#         self.MSD.append(self.numDeal(mean_average_distance, self.MSD, name='msd'))
#
#     def numDeal(self, nums, lists, name=None):
#         if np.isinf(nums):  # 如果是inf，则排除
#             logs(f'{name} exist inf')
#             nums = 100  # hd 的最大距离
#         if np.isnan(nums):
#             logs('exist nan')
#             nums = np.mean(lists)
#         return np.round((nums * 100), 2)  # 百分制，小数点后两位
#
#     def evluation(self, fold):
#         logs(
#             f"Fold {fold}"
#             f",dice : " + avgStd(self.DSC, log=True) +
#             f",HD: " + avgStd(self.HD, log=True)
#             + f",MSD: " + avgStd(self.MSD, log=True)
#             # +f",ppv: " + avgStd(self.PPV, log=True)
#         )
#         # return avgStd(self.DSC), avgStd(self.HD), avgStd(self.AHD), avgStd(self.PPV)
#         return avgStd(self.DSC), avgStd(self.HD), avgStd(self.MSD)
