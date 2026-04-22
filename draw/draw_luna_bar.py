# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import matplotlib.pyplot as plt

# todo luna
plt.figure()  # 设置画布的尺寸
size = 5
width = 0.3
xlabel = ['1', '2', '3', '4', '5']
fig, plots = plt.subplots(3, 3)
plots[0, 0].set_title('Subtlety')
plots[0, 0].bar(xlabel, [21, 69, 310, 366, 420], width=width)

plots[0, 1].set_title('Internal Structure')
internalStructure = ['1', '2', '3', '4']
plots[0, 1].bar(internalStructure, [1181, 3, 0, 2], width=width)
plots[0, 2].set_title('Calcification')
calcification = ['1', '2', '3', '4', '5', '6']
plots[0, 2].bar(calcification, [0, 3, 114, 10, 8, 1051], width=width)
plots[1, 0].set_title('Sphericity')
plots[1, 0].bar(xlabel, [0, 56, 437, 469, 224], width=width)
plots[1, 1].set_title('Margin')
plots[1, 1].bar(xlabel, [33, 80, 160, 471, 442], width=width)
plots[1, 2].set_title('Lobulation')
plots[1, 2].bar(xlabel, [777, 241, 113, 42, 13], width=width)
plots[2, 0].set_title('Spiculation')
plots[2, 0].bar(xlabel, [921, 141, 59, 32, 33], width=width)
plots[2, 1].set_title('Texture')
plots[2, 1].bar(['1~3', '4~5'], [130, 1056], width=width)
plots[2, 2].set_title('Malignancy')
plots[2, 2].bar(['1~2', '3', '4~5'], [386, 440, 360], width=width)
fig.tight_layout()  # 调整整体空白
plt.subplots_adjust(wspace=.35, hspace=0.5)  # 调整子图间距

fig, plots = plt.subplots(1)
plots.set_title('Diameter(mm)')
diameterL = ['SubSolid 3~6', 'SubSolid >6', 'Solid 3~6', 'Solid 6~8', 'Solid >8']
plots.bar(diameterL, [56, 74, 446, 247, 363], width=.3)
fig.tight_layout()  # 调整整体空白
plt.subplots_adjust(wspace=.15, hspace=0.45)  # 调整子图间距
plt.show()
plt.close()

# todo lidc
plt.figure()  # 设置画布的尺寸
size = 5
width = 0.3
xlabel = ['1', '2', '3', '4', '5']
fig, plots = plt.subplots(3, 3)
plots[0, 0].set_title('Subtlety')
# xlabel = ['Extremely Subtle', 'Moderately Subtle', 'Fairly Subtle', 'Moderately Obvious', 'Obvious']
plots[0, 0].bar(xlabel, [160, 292, 691, 764, 746], width=width)
plots[0, 1].set_title('Internal Structure')
internalStructure = ['1', '2', '3', '4']
plots[0, 1].bar(internalStructure, [2639, 10, 0, 4], width=width)
plots[0, 2].set_title('Calcification')
calcification = ['1', '2', '3', '4', '5', '6']
plots[0, 2].bar(calcification, [0, 3, 256, 36, 25, 2333], width=width)
plots[1, 0].set_title('Sphericity')
plots[1, 0].bar(xlabel, [5, 213, 899, 988, 548], width=width)
plots[1, 1].set_title('Margin')
plots[1, 1].bar(xlabel, [153, 254, 413, 908, 925], width=width)
plots[1, 2].set_title('Lobulation')
plots[1, 2].bar(xlabel, [1849, 457, 212, 88, 47], width=width)
plots[2, 0].set_title('Spiculation')
plots[2, 0].bar(xlabel, [2045, 334, 136, 77, 61], width=width)
plots[2, 1].set_title('Texture')
plots[2, 1].bar(['1~3', '4~5'], [465, 2188], width=width)
plots[2, 2].set_title('Malignancy')
plots[2, 2].bar(['1~2', '3', '4~5'], [1148, 959, 546], width=width)
fig.tight_layout()  # 调整整体空白
plt.subplots_adjust(wspace=.35, hspace=0.5)  # 调整子图间距

fig, plots = plt.subplots(1)
plots.set_title('Diameter(mm)')
diameterL = ['SubSolid 3~6', 'SubSolid >6', 'Solid 3~6', 'Solid 6~8', 'Solid >8']
plots.bar(diameterL, [136, 329, 846, 598, 744], width=.3)
fig.tight_layout()  # 调整整体空白
plt.subplots_adjust(wspace=.15, hspace=0.45)  # 调整子图间距
plt.show()
plt.close()
