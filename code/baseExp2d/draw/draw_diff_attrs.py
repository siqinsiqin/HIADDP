# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import pandas as pd
from matplotlib import pyplot as plt

# 电线图
from draw.colors import getColors
from utils.helper import getAllAttrs
from utils.reader import reader


def draw(plt, color, vals, labels):
    mask = '^'
    # fig.set_title('Precision')

    # todo 同一模型 五折验证在同一属性的评估
    # for i, (xlabel, ylabel) in enumerate(zip(fold, [0.77, 0.71, 0.72, 0.22, 0.55])):
    #     plt.plot(xlabel, ylabel, color=colors[i], linestyle=':', marker=mask, label=xlabel)

    # for attr in arrs:
    #     colors = getColors(u2net, len(u2net))
    #     for i, (key, val) in enumerate(zip(vals.keys(), vals.values())):
    #         plt.plot(attr, val, color=colors[i], linestyle=':', marker=mask, label=f'{key}')

    if labels == ['small', 'middle', 'huge']:
        labels = ['3~5mm', '5~10mm', '10mm++']

    for i, (key, val) in enumerate(zip(vals.keys(), vals.values())):
        plt.plot(labels, val, color=color, linestyle=':', marker=mask, label=f'{key}')


class drawDiffAttrs(reader):

    def __init__(self, dataset):
        super(drawDiffAttrs, self).__init__(dataset)
        # plt.figure()
        # self.fig, self.plots = plt.subplots(1)
        self.mask = 'o'

    @staticmethod
    def closeplt():
        # plt.legend(loc='best', ncol=4)
        # plt.tight_layout()
        plt.show()  # 显示图像

    def loadOneAttrs(self, labels, model_name):
        """
        加载一个属性中全部的label
        """
        datas = []
        for label in labels:
            path = f'{self.csv_path}/evaluate/{model_name}_{self.dataset}_{self.mode}_{label}_evaluate.csv'
            print(path)
            data = pd.read_csv(path, header=0)
            datas.append(data)
        return datas

    def combination(self, key, labels, model_name, size=6, baseIdx=17):
        """
        将同一label的不同属性的dsc属性进行组合，其中包括不同模型的不同loss
        """

        datas = self.loadOneAttrs(labels, model_name, )

        # self.plots.set_title(f'{key}')
        colors = getColors(length=datas[0].columns[1:].size * 3)
        k = 0
        for model in datas[0].columns[1:]:
            for i, loss in enumerate(self.losses):
                tmplist = []
                for data in datas:  # 不同csv

                    tmplist.append(round(data[model][(i * size) + baseIdx], 2))
                    """
                    输出长度与label
                    """
                print(f'{model}+{loss}:{tmplist}')
                # if tmplist[0] < 70:
                #     continue
                # draw(self.plots, colors[k], {f'{model}+{loss}': tmplist}, labels)
                k += 1
        # self.closeplt()


if __name__ == '__main__':

    model2d = ['unet', 'raunet', 'unetpp', 'cpfnet', 'unet3p', 'sgunet', 'bionet',
               'uctransnet', 'utnet', 'swinunet', 'unext']
    model3d = ['unet', 'ynet', 'unetpp', 'reconnet', 'unetr', 'transbts', 'wingsnet', 'pcamnet',
               'asa']  # 'vnet', 'resunet',, 'vtunet'
    model3d = [
        'unet',
        'unetpp',
        'reconnet',
        'pcamnet',
        'unetr',
        'asa',
        'dualbd5zs',
    ]
    # model3d = ['dualbd5zs']
    for model in model3d:
        for x, y in getAllAttrs(True).items():
            print(x)
            drawDiffAttrs('luna').combination(x, y, model)
