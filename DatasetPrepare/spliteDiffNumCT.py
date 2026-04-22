# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import numpy as np
import pandas as pd

from configs import config


def findOneTime(id):
    """
    判断id出现几次
    """
    return np.sum(data['seriesuid'] == id) == 1


"""
# todo 将一个和多个结节划分
"""
data = pd.read_csv(f'{config.csv_path}/annotations.csv', header=0)

one = []
more = []

for i, item in (enumerate(np.unique(data['seriesuid']))):
    if findOneTime(item):
        one.append(item)
    else:
        more.append(item)
ones = dict()
mores = dict()

ones.update({"seriesuid": one})
mores.update({"seriesuid": more})

df = pd.DataFrame(ones)
df.to_csv(f'{config.csv_path}/ones.csv')
df = pd.DataFrame(mores)
df.to_csv(f'{config.csv_path}/mores.csv')
