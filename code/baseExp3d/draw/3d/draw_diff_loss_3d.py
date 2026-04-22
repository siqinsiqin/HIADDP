from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 电线图
from draw.colors import getColors


def draw(models, Precision, Sensitivity, DSC, mIou):
    colors = getColors(models, len(Precision.items()))
    fold = [1, 2, 3, 4, 5, 'avg']

    plt.figure()  # 设置画布的尺寸
    fig, plots = plt.subplots(2, 2)
    # ax1 = plt.subplot(3, 2, 1)
    # ax2 = plt.subplot(3, 2, 2)
    # ax3 = plt.subplot(3, 2, 3)
    # ax4 = plt.subplot(3, 2, 4)

    y_major_locator = MultipleLocator(0.25)
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

    plots[1, 0].set_title('Dsc')
    for i, (model, val) in enumerate(DSC.items()):
        plots[1, 0].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    plots[1, 1].set_title('mIou')
    for i, (model, val) in enumerate(mIou.items()):
        plots[1, 1].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)
    # plt.legend(['unet', 'vnet', 'ynet', 'unet++', ], loc='best')
    plt.tight_layout()
    plt.show()  # 显示图像

    fig_leg = plt.figure(figsize=(5, 0.6))  # todo 修改label大小， 两行
    # fig_leg = plt.figure(figsize=(4.5, 0.5))  # todo 修改label大小， 一行

    ax_leg = fig_leg.add_subplot()
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=4)
    ax_leg.axis('off')
    # fig_leg.savefig('legend.png', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    dice, bce, focal, = False, False, True
    # todo dice
    if dice:
        model = ['unet3d', 'vnet', 'ynet', 'unet3d++', 'wingsnet']
        Precision = {
            'unet3d': [0.81, 0.8, 0.79, 0.78, 0.8, 0.8],
            'vnet': [0.8, 0.76, 0.8, 0.8, 0.66, 0.76],
            'ynet': [0.79, 0.81, 0.78, 0.83, 0.8, 0.8],
            'unetpp3d': [0.83, 0.81, 0.82, 0.81, 0.81, 0.82],
            'wingsnet': [0.81, 0.79, 0.81, 0.79, 0.78, 0.8]
        }
        Sensitivity = {
            'unet3d': [0.81, 0.83, 0.84, 0.84, 0.82, 0.83],
            'vnet': [0.83, 0.83, 0.83, 0.82, 0.85, 0.83],
            'ynet': [0.83, 0.78, 0.85, 0.77, 0.82, 0.81],
            'unetpp3d': [0.82, 0.84, 0.84, 0.83, 0.81, 0.83],
            'wingsnet': [0.83, 0.87, 0.86, 0.83, 0.83, 0.84]
        }
        DSC = {
            'unet3d': [0.79, 0.8, 0.8, 0.79, 0.79, 0.79],
            'vnet': [0.8, 0.77, 0.79, 0.79, 0.71, 0.77],
            'ynet': [0.78, 0.77, 0.78, 0.77, 0.78, 0.78],
            'unetpp3d': [0.81, 0.81, 0.82, 0.8, 0.78, 0.8],
            'wingsnet': [0.81, 0.81, 0.82, 0.79, 0.79, 0.8]
        }
        mIou = {
            'unet3d': [0.83, 0.84, 0.84, 0.83, 0.83, 0.83],
            'vnet': [0.84, 0.82, 0.84, 0.83, 0.78, 0.82],
            'ynet': [0.83, 0.82, 0.83, 0.82, 0.83, 0.83],
            'unetpp3d': [0.85, 0.84, 0.85, 0.84, 0.83, 0.84],
            'wingsnet': [0.84, 0.85, 0.85, 0.83, 0.83, 0.84]
        }
    # todo bce
    if bce:
        model = ['unet3d', 'vnet', 'ynet', 'unet3d++', 'wingsnet']
        Precision = {
            'unet3d': [0.84, 0.8, 0.79, 0.77, 0.79, 0.8], 'vnet': [0.78, 0.81, 0.81, 0.7, 0.65, 0.75],
            'ynet': [0.78, 0.78, 0.73, 0.78, 0.79, 0.77], 'unetpp3d': [0.69, 0.74, 0.77, 0.76, 0.73, 0.74],
            'wingsnet': [0.0, 0.57, 0.0, 0.0, 0.0, 0.11]
        }
        Sensitivity = {
            'unet3d': [0.75, 0.7, 0.74, 0.8, 0.75, 0.75], 'vnet': [0.66, 0.65, 0.56, 0.7, 0.75, 0.66],
            'ynet': [0.73, 0.76, 0.74, 0.79, 0.65, 0.73], 'unetpp3d': [0.41, 0.52, 0.61, 0.62, 0.49, 0.53],
            'wingsnet': [0.0, 0.35, 0.0, 0.0, 0.0, 0.07]
        }
        DSC = {
            'unet3d': [0.77, 0.71, 0.73, 0.76, 0.73, 0.74], 'vnet': [0.67, 0.68, 0.62, 0.65, 0.65, 0.65],
            'ynet': [0.72, 0.73, 0.69, 0.76, 0.67, 0.71], 'unetpp3d': [0.46, 0.56, 0.63, 0.65, 0.55, 0.57],
            'wingsnet': [0.0, 0.39, 0.0, 0.0, 0.0, 0.08]
        }
        mIou = {
            'unet3d': [0.82, 0.8, 0.8, 0.81, 0.8, 0.81], 'vnet': [0.77, 0.78, 0.75, 0.76, 0.76, 0.76],
            'ynet': [0.8, 0.8, 0.79, 0.82, 0.78, 0.8], 'unetpp3d': [0.67, 0.73, 0.76, 0.77, 0.72, 0.73],
            'wingsnet': [0.5, 0.65, 0.5, 0.5, 0.5, 0.53]
        }
        # todo focal
    if focal:
        model = ['unet3d', 'vnet', 'ynet', 'unet3d++', 'wingsnet']
        Precision = {
            'unet3d': [0.84, 0.78, 0.73, 0.8, 0.86, 0.8], 'vnet': [0.58, 0.63, 0.56, 0.62, 0.57, 0.59],
            'ynet': [0.85, 0.75, 0.79, 0.81, 0.73, 0.79], 'unetpp3d': [0.65, 0.46, 0.52, 0.69, 0.67, 0.6],
            'wingsnet': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        Sensitivity = {
            'unet3d': [0.56, 0.4, 0.31, 0.65, 0.52, 0.49], 'vnet': [0.26, 0.26, 0.22, 0.27, 0.32, 0.27],
            'ynet': [0.51, 0.33, 0.38, 0.52, 0.39, 0.43], 'unetpp3d': [0.3, 0.15, 0.15, 0.32, 0.29, 0.24],
            'wingsnet': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        DSC = {
            'unet3d': [0.63, 0.48, 0.4, 0.68, 0.61, 0.56], 'vnet': [0.33, 0.33, 0.28, 0.34, 0.37, 0.33],
            'ynet': [0.59, 0.42, 0.47, 0.58, 0.46, 0.5], 'unetpp3d': [0.37, 0.2, 0.21, 0.4, 0.36, 0.31],
            'wingsnet': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        mIou = {
            'unet3d': [0.75, 0.69, 0.65, 0.78, 0.74, 0.72], 'vnet': [0.62, 0.62, 0.6, 0.63, 0.64, 0.62],
            'ynet': [0.73, 0.66, 0.68, 0.72, 0.68, 0.69], 'unetpp3d': [0.64, 0.57, 0.57, 0.65, 0.63, 0.61],
            'wingsnet': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        }
    draw(model, Precision, Sensitivity, DSC, mIou, )
