from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 电线图
from draw.colors import getColors


def draw(model, small, middle, huge):
    colors = getColors(model, len(model))
    fold = ['unet3d', 'vnet', 'ynet', 'unet++3d', 'winsgnet', ]
    plt.figure()  # 设置画布的尺寸
    fig, plots = plt.subplots(3, 1)

    # y_major_locator = MultipleLocator()
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    mask = 'o'
    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型

    plots[0].set_title('Size:Small')
    for i, (model, val) in enumerate(small.items()):
        if len(val) == 5:
            fold = ['unet3d', 'vnet', 'ynet', 'unet++3d', 'winsgnet', ]
        if len(val) == 4:
            fold = ['unet3d', 'vnet', 'ynet', 'unet++3d', ]
        plots[0].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    plots[1].set_title('Size:Middle')
    for i, (model, val) in enumerate(middle.items()):
        if len(val) == 5:
            fold = ['unet3d', 'vnet', 'ynet', 'unet++3d', 'winsgnet', ]
        if len(val) == 4:
            fold = ['unet3d', 'vnet', 'ynet', 'unet++3d', ]
        plots[1].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    plots[2].set_title('Size:Huge')
    for i, (model, val) in enumerate(huge.items()):
        if len(val) == 5:
            fold = ['unet3d', 'vnet', 'ynet', 'unet++3d', 'winsgnet', ]
        if len(val) == 4:
            fold = ['unet3d', 'vnet', 'ynet', 'unet++3d', ]
        plots[2].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    # plt.legend(['Dice', 'Focal', 'Bce'], loc='best', ncol=3)
    plt.tight_layout()
    plt.show()  # 显示图像

    fig_leg = plt.figure(figsize=(3.5, 0.4))  # todo 修改label大小， 一行

    ax_leg = fig_leg.add_subplot()
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=3)
    ax_leg.axis('off')
    # fig_leg.savefig('legend.png', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    model = ['dice', 'focal', 'bce']
    # todo 对不同大小结节进行评估
    small = {
        # todo 不同模型不同结节的DSC
        # TODO MODEL UNET	UNET++	UNET3+	CPFNET	RAUNET	BIONET	SGUNET
        'dice': [0.78, 0.75, 0.77, 0.79, 0.78],
        'bce': [0.7, 0.6, 0.69, 0.46, 0.05],
        'focal': [0.48, 0.19, 0.47, 0.17, 0.0],
    }
    middle = {
        'dice': [0.83, 0.81, 0.8, 0.83, 0.82],
        'bce': [0.79, 0.74, 0.77, 0.73, 0.12],
        'focal': [0.68, 0.53, 0.6, 0.52, 0.0],
    }
    huge = {
        'dice': [0.84, 0.83, 0.76, 0.84, 0.86],
        'bce': [0.83, 0.8, 0.74, 0.78, 0.14],
        'focal': [0.74, 0.64, 0.42, 0.59, 0.0],
    }

    draw(model, small, middle, huge)
