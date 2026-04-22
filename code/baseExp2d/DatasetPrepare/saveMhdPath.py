# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os

import pandas as pd

from configs import config


def getPath():
    basepath = '/zsm/xu/$DOWNLOADLUNA16PATH/'
    id = []
    path = []
    data = dict()

    for dirpath, dirnames, filenames in os.walk(basepath):
        for filename in filenames:
            if filename.endswith('.mhd'):
                id.append(filename)
                path.append(os.path.join(dirpath, filename))
    data.update({'seriesuid': id})
    data.update({'path': path})
    df = pd.DataFrame(data)
    df.to_csv(f'{config.csv_path}/mhd_path.csv')


"""
将mhd 的id 与路径写入csv，加载文件部分删除读写说明文件，或者将相应的文件写为csv，
将luna16的mhd 路径存为一个csv
"""
getPath()
