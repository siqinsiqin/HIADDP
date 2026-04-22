from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 电线图
from draw.colors import getColors


def draw(model, solid, ggo, mixed):
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
    plots[0].set_title('Nodule:Solid')
    for i, (model, val) in enumerate(solid.items()):
        plots[0].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    plots[1].set_title('Nodule:GGO')
    for i, (model, val) in enumerate(ggo.items()):
        plots[1].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    plots[2].set_title('Nodule:Mixed')
    for i, (model, val) in enumerate(mixed.items()):
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
    # todo 以dsc为主要指标，进行构图
    solid = {
        'dice': [0.81, 0.78, 0.79, 0.81, 0.81],
        'bce': [0.75, 0.69, 0.73, 0.59, 0.08],
        'focal': [0.58, 0.36, 0.53, 0.34, 0.0],
    }
    ggo = {
        'dice': [0.66, 0.63, 0.62, 0.69, 0.68],
        'bce': [0.61, 0.33, 0.54, 0.32, 0.05],
        'focal': [0.36, 0.05, 0.26, 0.07, 0.0],
    }
    mixed = {
        'dice': [0.72, 0.73, 0.72, 0.76, 0.75],
        'bce': [0.66, 0.41, 0.62, 0.47, 0.08],
        'focal': [0.41, 0.04, 0.3, 0.14, 0.0],
    }

    draw(model, solid, ggo, mixed)
