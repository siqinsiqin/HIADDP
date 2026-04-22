from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 电线图
from draw.colors import getColors


def draw(model, good, bad):
    colors = getColors(model, len(model))

    plt.figure()  # 设置画布的尺寸
    fig, plots = plt.subplots(2, 1)

    # y_major_locator = MultipleLocator(0.1)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # ax.yaxis.set_major_locator(y_major_locator)
    mask = 'o'
    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    plots[0].set_title('Malignancy:Benign')
    for i, (model, val) in enumerate(good.items()):
        fold = ['unet3d', 'vnet', 'ynet', 'unet++3d', 'winsgnet', ]
        plots[0].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

    plots[1].set_title('Malignancy:Malignant')
    for i, (model, val) in enumerate(bad.items()):
        fold = ['unet3d', 'vnet', 'ynet', 'unet++3d', 'winsgnet', ]
        plots[1].plot(fold, val, color=colors[i], linestyle=':', marker=mask, label=model)

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
    # todo 对不同 loss 的良性恶性结节进行评估
    good = {
        'dice': [0.79, 0.77, 0.78, 0.8, 0.8],
        'bce': [0.73, 0.64, 0.71, 0.53, 0.07],
        'focal': [0.54, 0.28, 0.51, 0.26, 0.0],
    }
    bad = {
        'dice': [0.83, 0.81, 0.79, 0.82, 0.83],
        'bce': [0.81, 0.76, 0.75, 0.74, 0.12],
        'focal': [0.7, 0.56, 0.51, 0.54, 0.0],
    }

    draw(model, good, bad)
