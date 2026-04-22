# -*-coding:utf-8 -*-
"""
# Time       ：2022/4/18 9:56
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from configs import config
from draw.colors import getColors
from utils.reader import reader


class entity():

    def __init__(self, name):
        self.name = name
        self.pre = []
        self.sen = []
        self.dice = []
        self.miou = []

    def updata(self, pre, sen, dice, miou):
        self.pre.append(pre)
        self.sen.append(sen)
        self.dice.append(dice)
        self.miou.append(miou)


class draw_entirety(reader):

    def __init__(self):
        super(draw_entirety, self).__init__(config.dataset)

    def loadEntirety(self, size=6):
        plt.figure()  # 设置画布的尺寸
        fig, plots = plt.subplots(1, 1)

        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)

        mask = 'o'
        # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型

        # twoddata = pd.read_csv(f'{self.csv_path}/evaluate/{self.dataset}_2d_eva_all_evaluate.csv', header=0)
        twoddata = pd.read_csv(f'{self.csv_path}/evaluate/{self.dataset}_3d_eva_all_evaluate.csv', header=0)
        models = []  # todo 模型名称
        """不同损失"""
        dsc = entity('dsc')
        bce = entity('bce')
        focal = entity('focal')
        totalLossFn = [dsc, bce, focal]

        preBase = 5
        senBase = 23
        dscBase = 41
        mIouBase = 59
        pre = []
        sen = []
        dsc = []
        miou = []
        colors = getColors(length=len(twoddata.columns[1:]))
        i = 1
        for model in twoddata.columns[1:]:
            tmp = entity('tmp')
            for i in [0, 1, 2]:
                print(f'{model} {i}  pre,sen,dsc,miou',
                      [round(twoddata[model][preBase + i * size] * 100, 2),
                       round(twoddata[model][senBase + i * size] * 100, 2),
                       round(twoddata[model][dscBase + i * size] * 100, 2),
                       round(twoddata[model][mIouBase + i * size] * 100, 2)])

            # print(len(models), len(colors))
            # plots.set_title('precision')
            # plots.plot(models, tmp.pre, color=colors, linestyle=':', marker=mask, label=models)

            # plots[1].set_title('sensitivity')
            # for color in colors:
            #     plots[1].plot(models, tmp.sen, color=colors, linestyle=':', marker=mask, label=models)
            #
            # plots[2].set_title('dsc')
            # for color in colors:
            #     plots[2].plot(models, tmp.dice, color=color, linestyle=':', marker=mask, label=models)
            #
            # plots[3].set_title('mIou')
            # for color in colors:
            #     plots[3].plot(models, tmp.miou, color=color, linestyle=':', marker=mask, label=models)

        # for model in threedata.columns[1:]:
        #     for i, loss in enumerate(self.losses):
        #         if threedata[model][preBase + i * size] < .2:
        #             continue
        #         models.append(model)
        #         print('pre,sen,dsc,miou', [threedata[model][preBase + i * size], threedata[model][senBase + i * size],
        #                                    threedata[model][dscBase + i * size], threedata[model][mIouBase + i * size]])
        #         totalLossFn[i].updata(threedata[model][preBase + i * size], threedata[model][senBase + i * size],
        #                               threedata[model][dscBase + i * size], threedata[model][mIouBase + i * size])

        # colors = getColors(length=len(models))
        # # print(len(models), len(colors))
        #
        #
        # # plt.legend(['Dice', 'Focal', 'Bce'], loc='best', ncol=3)
        # plt.tight_layout()
        # plt.show()  # 显示图像
        #
        # fig_leg = plt.figure(figsize=(3.5, 0.4))  # todo 修改label大小， 一行
        #
        # ax_leg = fig_leg.add_subplot()
        # ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=3)
        # ax_leg.axis('off')
        # # fig_leg.savefig('legend.png', pad_inches=0)
        # plt.show()


if __name__ == '__main__':
    draw_entirety().loadEntirety()
