# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 电线图
from draw.colors import getColors


def draw(model, Precision, Sensitivity, DSC, mIou):
    colors = getColors(model, len(Precision.items()))
    fold = [1, 2, 3, 4, 5, 'avg']

    plt.figure()  # 设置画布的尺寸
    fig, plots = plt.subplots(2, 2)

    y_major_locator = MultipleLocator(0.05)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    mask = 'o'
    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    plots[0, 0].set_title('Precision')
    for i, (model, val) in enumerate(Precision.items()):
        plots[0, 0].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    plots[0, 1].set_title('Sensitivity')
    for i, (model, val) in enumerate(Sensitivity.items()):
        plots[0, 1].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    plots[1, 0].set_title('DSC')
    for i, (model, val) in enumerate(DSC.items()):
        plots[1, 0].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    plots[1, 1].set_title('mIou')
    for i, (model, val) in enumerate(mIou.items()):
        plots[1, 1].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    plt.tight_layout()
    plt.show()  # 显示图像


if __name__ == '__main__':
    dice, focal, bce = False, True, False
    # todo dice
    if dice:
        model = ['unet', 'unet++', 'unet3+', 'cpfnet', 'raunet', 'bionet', 'sgunet', ]
        Precision = {
            'unet': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'unet++': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'unet3+': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'cpfnet': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'raunet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
            'bionet': [0.82, 0.82, 0.82, 0.78, 0.83, 0.81],
            'sgunet': [0.81, 0.82, 0.81, 0.82, 0.82, 0.82],
            # 'kiunet': [0.53, 0.47, 0.53, 0.47, 0.55, 0.51]
        }
        Sensitivity = {
            'unet': [0.84, 0.84, 0.86, 0.86, 0.88, 0.86],
            'unet++': [0.83, 0.88, 0.86, 0.83, 0.84, 0.85],
            'unet3+': [0.86, 0.85, 0.84, 0.86, 0.85, 0.85],
            'cpfnet': [0.83, 0.84, 0.84, 0.83, 0.87, 0.84],
            'raunet': [0.81, 0.83, 0.80, 0.80, 0.83, 0.81],
            'bionet': [0.84, 0.85, 0.86, 0.88, 0.85, 0.86],
            'sgunet': [0.86, 0.86, 0.87, 0.87, 0.88, 0.87],
            # 'kiunet': [0.98, 0.99, 0.99, 0.99, 0.99, 0.99]
        }
        DSC = {
            'unet': [0.81, 0.82, 0.82, 0.83, 0.83, 0.82],
            'unet++': [0.81, 0.82, 0.82, 0.81, 0.82, 0.82],
            'unet3+': [0.82, 0.82, 0.82, 0.82, 0.82, 0.82],
            'cpfnet': [0.80, 0.80, 0.80, 0.80, 0.81, 0.80],
            'raunet': [0.78, 0.79, 0.78, 0.79, 0.79, 0.79],
            'bionet': [0.81, 0.81, 0.81, 0.81, 0.82, 0.81],
            'sgunet': [0.81, 0.82, 0.81, 0.82, 0.82, 0.82],
            # 'kiunet': [0.66, 0.61, 0.66, 0.61, 0.67, 0.64]
        }
        mIou = {
            'unet': [0.86, 0.86, 0.86, 0.86, 0.86, 0.86],
            'unet++': [0.85, 0.86, 0.86, 0.85, 0.86, 0.86],
            'unet3+': [0.86, 0.86, 0.86, 0.86, 0.86, 0.86],
            'cpfnet': [0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
            'raunet': [0.84, 0.84, 0.84, 0.84, 0.84, 0.84],
            'bionet': [0.86, 0.85, 0.86, 0.85, 0.86, 0.86],
            'sgunet': [0.85, 0.86, 0.86, 0.86, 0.86, 0.86],
            # 'kiunet': [0.76, 0.73, 0.76, 0.73, 0.76, 0.75]
        }
    # todo focal
    if focal:
        model = ['unet', 'unet++', 'unet3+', 'cpfnet', 'raunet', 'bionet', 'sgunet']
        Precision = {
            'unet': [0.89, 0.88, 0.89, 0.88, 0.90, 0.89],
            'unet++': [0.86, 0.89, 0.88, 0.87, 0.87, 0.87],
            'unet3+': [0.87, 0.89, 0.87, 0.88, 0.84, 0.87],
            'cpfnet': [0.86, 0.89, 0.85, 0.85, 0.86, 0.86],
            'raunet': [0.79, 0.77, 0.80, 0.79, 0.83, 0.80],
            'bionet': [0.86, 0.9, 0.86, 0.86, 0.89, 0.87],
            'sgunet': [0.84, 0.86, 0.83, 0.84, 0.86, 0.85],
        }
        Sensitivity = {
            'unet': [0.67, 0.64, 0.68, 0.62, 0.71, 0.66],
            'unet++': [0.65, 0.70, 0.67, 0.60, 0.73, 0.67],
            'unet3+': [0.66, 0.67, 0.66, 0.69, 0.59, 0.65],
            'cpfnet': [0.62, 0.65, 0.69, 0.66, 0.62, 0.65],
            'raunet': [0.49, 0.49, 0.53, 0.50, 0.60, 0.52],
            'bionet': [0.56, 0.66, 0.59, 0.67, 0.7, 0.64],
            'sgunet': [0.57, 0.64, 0.57, 0.59, 0.66, 0.61],
        }
        DSC = {
            'unet': [0.74, 0.71, 0.74, 0.69, 0.77, 0.73, ],
            'unet++': [0.71, 0.76, 0.74, 0.68, 0.77, 0.73],
            'unet3+': [0.73, 0.74, 0.72, 0.74, 0.67, 0.72],
            'cpfnet': [0.69, 0.72, 0.73, 0.72, 0.69, 0.71],
            'raunet': [0.58, 0.57, 0.61, 0.58, 0.66, 0.60],
            'bionet': [0.65, 0.74, 0.67, 0.72, 0.75, 0.71],
            'sgunet': [0.65, 0.70, 0.64, 0.66, 0.71, 0.67],
        }
        mIou = {
            'unet': [0.81, 0.80, 0.81, 0.79, 0.83, 0.81],
            'unet++': [0.80, 0.82, 0.81, 0.78, 0.83, 0.81],
            'unet3+': [0.81, 0.81, 0.80, 0.81, 0.78, 0.80],
            'cpfnet': [0.79, 0.80, 0.81, 0.80, 0.79, 0.80],
            'raunet': [0.73, 0.73, 0.75, 0.73, 0.77, 0.74],
            'bionet': [0.77, 0.81, 0.78, 0.8, 0.82, 0.8],
            'sgunet': [0.77, 0.79, 0.76, 0.77, 0.80, 0.78],
        }
    # todo bce
    if bce:
        model = ['unet', 'unet++', 'unet3+', 'cpfnet', 'raunet', 'sgunet', 'bionet']
        Precision = {
            'unet': [0.83, 0.85, 0.86, 0.83, 0.84, 0.84],
            'unet++': [0.83, 0.82, 0.84, 0.81, 0.81, 0.82],
            'unet3+': [0.83, 0.83, 0.81, 0.85, 0.83, 0.83],
            'cpfnet': [0.81, 0.81, 0.82, 0.83, 0.80, 0.81],
            'raunet': [0.81, 0.79, 0.81, 0.80, 0.78, 0.80],
            'sgunet': [0.84, 0.83, 0.82, 0.83, 0.80, 0.82],
            'bionet': [0.83, 0.85, 0.82, 0.84, 0.82, 0.83],
        }
        Sensitivity = {
            'unet': [0.83, 0.83, 0.81, 0.83, 0.82, 0.82],
            'unet++': [0.82, 0.86, 0.83, 0.87, 0.87, 0.85],
            'unet3+': [0.82, 0.85, 0.87, 0.79, 0.86, 0.84],
            'cpfnet': [0.83, 0.83, 0.82, 0.80, 0.88, 0.83],
            'raunet': [0.78, 0.80, 0.77, 0.81, 0.84, 0.80],
            'sgunet': [0.79, 0.84, 0.83, 0.81, 0.87, 0.83],
            'bionet': [0.83, 0.79, 0.82, 0.80, 0.85, 0.82],
        }
        DSC = {
            'unet': [0.80, 0.82, 0.81, 0.80, 0.80, 0.81],
            'unet++': [0.80, 0.82, 0.81, 0.82, 0.81, 0.81],
            'unet3+': [0.80, 0.82, 0.81, 0.80, 0.82, 0.81],
            'cpfnet': [0.80, 0.80, 0.79, 0.79, 0.81, 0.80],
            'raunet': [0.77, 0.77, 0.76, 0.78, 0.78, 0.77],
            'sgunet': [0.79, 0.82, 0.80, 0.80, 0.81, 0.80],
            'bionet': [0.81, 0.79, 0.80, 0.79, 0.81, 0.80],
        }
        mIou = {
            'unet': [0.85, 0.86, 0.85, 0.85, 0.85, 0.85],
            'unet++': [0.85, 0.86, 0.85, 0.86, 0.85, 0.85],
            'unet3+': [0.85, 0.86, 0.86, 0.85, 0.86, 0.86],
            'cpfnet': [0.85, 0.85, 0.84, 0.84, 0.85, 0.85],
            'raunet': [0.83, 0.83, 0.83, 0.83, 0.84, 0.83],
            'sgunet': [0.84, 0.86, 0.85, 0.85, 0.85, 0.85],
            'bionet': [0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
        }

    draw(model, Precision, Sensitivity, DSC, mIou, )
