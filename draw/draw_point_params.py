# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
# import matplotlib.pyplot as plt
#
# # todo twoD
# params = [17.26, 26.07, 22.13, 43.26, 26.97, 14.97, 4.99]
# flops = [2.5, 1.15, 0.43, 0.5, 12.38, 2.24, 19.83]
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
# scatt = plt.scatter(flops, params, c=colors, marker='s')
# # todo 不含cenet ，
# # model msnet,kiunet,unet3d,vnet,ynet,kiunet3d,unetr
# params = [29.74, 0.29, 1.78, 45.6, 33.28, 0.23, 100.35]
# flops = [0.05, 17.53, 7.2, 93.91, 297.05, 51.88, 66.54]
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
#
# scatt = plt.scatter(flops, params, c=colors, marker='o')
#
# plt.xlabel('FLOPs(GMac)')
# plt.ylabel('Params(M)')
# # plt.set(xlim=(0, 50), xticks=np.arange(1, 8),
# #        ylim=(-5, 1000), yticks=np.arange(1, 8))
#
# plt.show()
# plt.close()
import matplotlib.pyplot as plt

from colors import getColors

"""
transformer 的最终参数还需要确认
"""
models = ['Unet', 'Raunet', 'Unet++', 'Cpfnet', 'Unet3+', 'Bionet', 'Unet Df', 'Chdseg', 'D-Kiunet', 'Sgunet', 'Sgl',
          'Msnet',
          'ThreeD Unet', 'Vnet', 'Residual Unet', 'Ynet', 'ThreeD Kiunet', '3D Unet++', 'Wingsnet', 'ReconNet', 'Ucaps',
          'Uctransnet', 'MedT', 'Utnet', 'Swin Unet', 'TransBTS', 'Unetr', 'UNeXt']

params = [31.04, 22.13, 26.07, 43.26, 26.97, 14.97, 34.53, 22.8, 0.488, 4.99, 15.53, 29.74,
          16.32, 45.6, 141.24, 33.28, 0.235, 5.73, 1.47, 4.08, 5.4,
          66.26, 1.32, 14.41, 27.15, 30.95, 92.34, 0.253
          ]
flops = [3.42, 0.43, 1.15, 0.5, 12.38, 2.24, 4.09, 3.25, 17.53, 0.31, 1.72, 0.05,
         237.01, 93.91, 398.6, 297.05, 51.88, 49.2, 38.55, 59.64, 173.07,
         2.7, 0.06, 1.28, 0.48, 32.68, 21.49, 0.01
         ]
print(len(params), len(models), len(flops))
colors = getColors(models, len(params))

for i, (x, y) in enumerate(zip(flops, params)):
    scatt = plt.scatter(x, y, c=colors[i], marker='^', label=models[i])

plt.xlabel('FLOPs(GMac)')
plt.ylabel('Params(M)')
plt.legend(loc='best', ncol=3)
plt.grid()
plt.show()
plt.close()
