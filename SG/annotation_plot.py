#!/usr/bin/env python
#-*- coding:utf-8 -*-

#接k_means.py
#k_means.py中得到三维规范化数据data_zs；
#r增加了最后一列，列索引为“聚类类别”
import pandas as pd
from sklearn.manifold import TSNE
import torch
s=torch.load('obj_direct.dat')
#print(s.shape)
s5=s[0:10,0:10]
#print(s5)
s25=s[0:23,0:23]
s50=s[0:49,0:49]
#data=s50
#data_zs = 1.0*(data - data.mean())/data.std()
# coding=utf-8

from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np

from thordata.FloorPlan1 import k_means as k

# 用TSNE进行数据降维并展示聚类结果

tsne = TSNE ()
tsne.fit_transform (k.data_zs)  # 进行数据降维,并返回结果
tsne = pd.DataFrame (tsne.embedding_, index=k.data_zs.index)  # 转换数据格式

import matplotlib.pyplot as plt

plt.rcParams ['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams ['axes.unicode_minus'] = False  # 用来正常显示负号

# 不同类别用不同颜色和样式绘图
d = tsne [k.r [u'聚类类别'] == 0]  # 找出聚类类别为0的数据对应的降维结果
plt.plot (d [0], d [1], 'r.')
d = tsne [k.r [u'聚类类别'] == 1]
plt.plot (d [0], d [1], 'go')
# d = tsne[k.r[u'聚类类别'] == 2]
# plt.plot(d[0], d[1], 'b*')
plt.savefig ("data.png")
plt.show ()

