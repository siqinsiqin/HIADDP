# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os
from functools import singledispatch

import numpy as np
import pandas as pd

import configs
from configs import config
from utils.helper import avgStd
from utils.logger import logs


class Writer(object):
    '''
    评估单个结节
    todo 用于保存五折交叉验证的各项精度指标
    '''
    mode = config.mode
    evaluatetype = config.log_name
    csv_path = config.csv_path
    metricsV = configs.MetricsV

    def __init__(self, dataset):

        self.dicts = {}
        self.dataset = dataset
        # metric 1
        self.rve = []
        self.voe = []
        self.dsc = []
        self.precision = []
        self.sensitivity = []
        self.mIou = []
        self.oneFolddsc = []
        self.oneFoldprecision = []
        self.oneFoldsensitivity = []
        self.oneFoldmIou = []
        self.oneFoldvoe = []
        self.oneFoldrve = []
        # metric 2
        self.HD = []
        self.MSD = []
        self.oneFoldHD = []
        self.oneFoldMSD = []

    @singledispatch
    def __call__(self, dice=None, hd=None, msd=None, avg=False):
        if avg:
            # logs(
            #     f'avg ,dice : {avgStd(self.oneFolddsc, log=True)}, '
            #     f'hd: {avgStd(self.oneFoldHD, log=True)},'
            #     f'msd: {avgStd(self.oneFoldMSD, log=True)},')
            self.oneFoldHD.append(avgStd(self.oneFoldHD))
            self.oneFoldMSD.append(avgStd(self.oneFoldMSD))
            self.oneFolddsc.append(avgStd(self.oneFolddsc))

            for i in range(len(self.oneFolddsc)):
                self.dsc.append(self.oneFolddsc[i])
                self.HD.append(self.oneFoldHD[i])
                self.MSD.append(self.oneFoldMSD[i])

            self.clear()
        else:
            if dice is None and hd is None and msd is None:
                if len(self.oneFolddsc) != 0:
                    # 如果这一fold里面没有对应的标签，那么就求其他fold的均值
                    self.oneFolddsc.append(np.mean(self.oneFolddsc))
                    self.oneFoldMSD.append(np.mean(self.oneFoldMSD))
                    self.oneFoldHD.append(np.mean(self.oneFoldHD))
            else:
                self.oneFolddsc.append(dice)
                self.oneFoldMSD.append(msd)
                self.oneFoldHD.append(hd)

    @singledispatch
    def __call__(self, precision=None, sensitivity=None, dsc=None, mIou=None, voe=None, rve=None, avg=False, ):
        if avg:
            # todo 求 现有的五折平均值
            logs(
                f'avg ,Precision : {avgStd(self.oneFoldprecision, log=True)}, '
                f'Sensitivity: {avgStd(self.oneFoldsensitivity, log=True)},'
                f'DSC: {avgStd(self.oneFolddsc, log=True)},'
                f'Iou: {avgStd(self.oneFoldmIou, log=True)}\n'
                f'voe, : {avgStd(self.oneFoldvoe, log=True)}, '
                f'rve, : {avgStd(self.oneFoldrve, log=True)}, '
            )
            self.oneFoldprecision.append(avgStd(self.oneFoldprecision))
            self.oneFoldsensitivity.append(avgStd(self.oneFoldsensitivity))
            self.oneFolddsc.append(avgStd(self.oneFolddsc))
            self.oneFoldmIou.append(avgStd(self.oneFoldmIou))
            self.oneFoldvoe.append(avgStd(self.oneFoldvoe))
            self.oneFoldrve.append(avgStd(self.oneFoldrve))
            # 单个loss的全部精度
            for i in range(len(self.oneFolddsc)):
                self.dsc.append(self.oneFolddsc[i])
                self.precision.append(self.oneFoldprecision[i])
                self.sensitivity.append(self.oneFoldsensitivity[i])
                self.mIou.append(self.oneFoldmIou[i])
                self.voe.append(self.oneFoldvoe[i])
                self.rve.append(self.oneFoldrve[i])

            self.clear()
        else:
            if precision is None and sensitivity is None and dsc is None and mIou is None:
                if len(self.oneFoldprecision) != 0:
                    # 如果这一fold里面没有对应的标签，那么就求其他fold的均值
                    self.oneFoldprecision.append(np.mean(self.oneFoldprecision))
                    self.oneFoldsensitivity.append(np.mean(self.oneFoldsensitivity))
                    self.oneFolddsc.append(np.mean(self.oneFolddsc))
                    self.oneFoldmIou.append(np.mean(self.oneFoldmIou))
                    self.oneFoldvoe.append(np.mean(self.oneFoldvoe))
                    self.oneFoldrve.append(np.mean(self.oneFoldrve))
            else:
                self.oneFoldprecision.append(precision)
                self.oneFoldsensitivity.append(sensitivity)
                self.oneFolddsc.append(dsc)
                self.oneFoldmIou.append(mIou)
                self.oneFoldvoe.append(voe)
                self.oneFoldrve.append(rve)

    def clear(self, allClear=False):
        if allClear:
            # 置空
            self.oneFolddsc = []
            self.oneFoldprecision = []
            self.oneFoldsensitivity = []
            self.oneFoldmIou = []
            self.oneFoldvoe = []
            self.oneFoldrve = []
            self.dsc = []
            self.precision = []
            self.sensitivity = []
            self.mIou = []
            self.voe = []
            self.rve = []

            # metric 2
            self.HD = []
            self.MSD = []
            self.oneFoldHD = []
            self.oneFoldMSD = []

        else:
            # 置空
            self.oneFolddsc = []
            self.oneFoldprecision = []
            self.oneFoldsensitivity = []
            self.oneFoldmIou = []
            self.oneFoldvoe = []
            self.oneFoldrve = []
            self.oneFoldHD = []
            self.oneFoldMSD = []

    @classmethod
    def reshape(cls, arrs):
        # print(arrs)
        arr = []
        for i in range(len(arrs)):
            for t in range(len(arrs[i])):
                arr.append(arrs[i][t])

        return arr

    def update(self, model_name):
        if self.metricsV == 1:
            data = self.reshape([self.precision, self.sensitivity, self.dsc, self.mIou, self.voe, self.rve, ])
        else:
            data = self.reshape([self.dsc, self.HD, self.MSD])
        # logs(f'model name {model_name}, len {len(data)} and data len == 6 {len(data) == 6}')
        # print(data)
        self.dicts.update({f"{model_name}": data})  # todo 一个模型的三个loss的全部精度
        # 置空
        self.clear(True)

    def save(self, model_name):
        df = pd.DataFrame(self.dicts)
        ''' 
            保存 其中一个模型的三种不同loss的各项精度指标
            依次是 精度，敏感度，dice，miou的五折分数及平均值
        '''
        os.makedirs(self.csv_path + '/evaluate/', exist_ok=True)
        df.to_csv(f'{self.csv_path}/evaluate/{model_name}_{self.dataset}_{self.mode}_{self.evaluatetype}_evaluate.csv')
        # 置空
        self.clear(True)
