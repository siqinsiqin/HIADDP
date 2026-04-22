# -*-coding:utf-8 -*-
"""
# Time       ：2022/5/24 10:11
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import pylidc as pl

cnt = 0

CTs = pl.query(pl.Scan).all()
for ct in CTs:
    nods = ct.cluster_annotations()
    cnt += len(nods)

print(cnt)
