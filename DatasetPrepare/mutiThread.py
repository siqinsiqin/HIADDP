# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import numpy as np
import pandas as pd

from configs import config

data = pd.read_csv(f'{config.csv_path}/annotations.csv', header=0)  # 加载luna16结节信息
size = 17
idx = 35
start = size * idx
for i, item in (enumerate(np.unique(data['seriesuid'])[size * idx:start + size])):
    print(i + start, item)

# print(np.unique(data['seriesuid'])[size * idx:size * idx + size])
# print(multiprocessing.cpu_count())
