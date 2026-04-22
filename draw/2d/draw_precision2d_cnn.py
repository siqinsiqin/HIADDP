from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 电线图
from draw.colors import getColors


def draw(models, Precision, Sensitivity, DSC, mIou):
    colors = getColors(models, len(Precision.items()))
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

    # plt.legend(model, loc='lower left', )
    plt.tight_layout()
    plt.show()  # 显示图像


if __name__ == '__main__':
    dice, focal, bce = False, True, False
    # todo dice
    if dice:
        model = ['unet3d', 'vnet', 'ynet', 'unet3d++', 'wingsnet']
        Precision = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
        Sensitivity = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
        DSC = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
        mIou = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
    # todo focal
    if focal:
        model = ['unet3d', 'vnet', 'ynet', 'unet3d++', 'wingsnet']
        Precision = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
        Sensitivity = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
        DSC = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
        mIou = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
    # todo bce
    if bce:
        model = ['unet3d', 'vnet', 'ynet', 'unet3d++', 'wingsnet']
        Precision = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
        Sensitivity = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
        DSC = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }
        mIou = {
            'unet3d': [0.83, 0.84, 0.83, 0.84, 0.81, 0.83],
            'vnet': [0.83, 0.81, 0.82, 0.84, 0.85, 0.83],
            'ynet': [0.82, 0.84, 0.84, 0.83, 0.83, 0.83],
            'unet3d++': [0.82, 0.82, 0.81, 0.82, 0.80, 0.81],
            'wingsnet': [0.80, 0.80, 0.81, 0.83, 0.80, 0.81],
        }

    draw(model, Precision, Sensitivity, DSC, mIou, )
