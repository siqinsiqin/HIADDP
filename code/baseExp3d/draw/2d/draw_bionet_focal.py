from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# 电线图
from draw.colors import getColors


# 点图
# fig, ax = plt.subplots()
# UNETD = [0.83, 0.84, 0.83, 0.84, 0.81]
# UNETPPD = [0.83, 0.81, 0.82, 0.84, 0.85]
# UNETD1 = [0.83, 0.82, 0.83, 0.82, 0.81]
# UNETD2 = [0.83, 0.85, 0.87, 0.84, 0.81]
#
# ax.plot(UNETD, 'o', label='UNET')
# ax.plot(UNETPPD, 'd', label='UNET++')
# ax.plot(UNETD1, 'v', label='data3')
# ax.plot(UNETD2, 'x', label='data4')
# ax.plot(UNETD2, 's', label='data4')
# ax.plot(UNETD2, '^', label='data4')
# ax.plot(UNETD2, 'v', label='data4')
# ax.plot(UNETD2, '>', label='data4')
# ax.plot(UNETD2, '<', label='data4')
# ax.plot(UNETD2, '>', label='data4')
# ax.plot(UNETD2, '<', label='data4')
# # 把x轴的刻度间隔设置为1，并存在变量里
# x_major_locator = MultipleLocator(1)
#
# ax = plt.gca()
# # ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)
# plt.xlim((-0.2, 4.2), )
# plt.title('5-FOLD DICE ')
# ax.legend()
# plt.show()
# plt.close()


def draw(round, Precision, Sensitivity, DSC, mIou):
    colors = getColors(round, len(Precision.items()))
    fold = [1, 2, 3, 4, 5, 'avg']

    plt.figure()  # 设置画布的尺寸
    fig, plots = plt.subplots(2, 2)

    y_major_locator = MultipleLocator(0.05)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    plots[0, 0].set_title('Precision')
    for i, (model, val) in enumerate(Precision.items()):
        plots[0, 0].plot(fold, val, color=colors[i], linestyle=':', marker='s', label=model)

    plots[0, 1].set_title('Sensitivity')
    for i, (model, val) in enumerate(Sensitivity.items()):
        plots[0, 1].plot(fold, val, color=colors[i], linestyle=':', marker='s', label=model)

    plots[1, 0].set_title('DSC')
    for i, (model, val) in enumerate(DSC.items()):
        plots[1, 0].plot(fold, val, color=colors[i], linestyle=':', marker='s', label=model)

    plots[1, 1].set_title('mIou')
    for i, (model, val) in enumerate(mIou.items()):
        plots[1, 1].plot(fold, val, color=colors[i], linestyle=':', marker='s', label=model)

    plt.legend(round, loc='best')
    plt.tight_layout()
    plt.show()  # 显示图像

    plt.tight_layout()
    plt.show()  # 显示图像


if __name__ == '__main__':
    # todo bce
    Precision = {
        'round1': [0.86, 0.9, 0.86, 0.86, 0.89, 0.87],
        'round2': [0.86, 0.87, 0.45, 0.88, 0.88, 0.79],
        'round3': [0.87, 0.82, 0.63, 0.86, 0.86, 0.81],
    }
    Sensitivity = {
        'round1': [0.56, 0.66, 0.59, 0.67, 0.7, 0.64],
        'round2': [0.62, 0.68, 0.12, 0.66, 0.59, 0.53],
        'round3': [0.72, 0.51, 0.24, 0.64, 0.62, 0.55],
    }
    DSC = {
        'round1': [0.65, 0.74, 0.67, 0.72, 0.75, 0.71],
        'round2': [0.7, 0.73, 0.17, 0.72, 0.68, 0.6],
        'round3': [0.76, 0.6, 0.33, 0.71, 0.69, 0.62],
    }
    mIou = {
        'round1': [0.77, 0.81, 0.78, 0.8, 0.82, 0.8],
        'round2': [0.79, 0.81, 0.55, 0.8, 0.78, 0.75],
        'round3': [0.82, 0.75, 0.62, 0.79, 0.79, 0.75],
    }
    round = ['round1', 'round2', 'round3']
    draw(round, Precision, Sensitivity, DSC, mIou)
