# k_mean算法
import pandas as pd
import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
# 参数初始化
inputfile = 'comgraph.xlsx'  # 销量及其他属性数据
outputfile = 'x_1.xlsx'  # 保存结果的文件名
k = 92  # 聚类的类别
iteration = 100  # 聚类最大循环次数

data = pd.read_excel (inputfile, index_col='id')  # 读取数据
#print(data)
data_zs=data
data_zs.fillna('0.0')
data_zs.dropna(inplace=True)
#data_zs=preprocessing.normalize(data, norm='l2')
#data_zs=pd.DataFrame(data_zs)
print(np.isnan(data_zs).any())
data_zs = 1.0 * (data - data.mean ()) / data.std ()  # 数据标准化，std()表示求总体样本方差(除以n-1),numpy中std()是除以n
#print(data_zs)
#data_zs.fillna('0.0')
#data_zs.dropna(inplace=True)
#print(np.isnan(data_zs).any())
print(data_zs)
print(np.isnan(data_zs).any())
from sklearn.cluster import KMeans
model = KMeans (n_clusters=k, max_iter=iteration)  # 分为k类
#model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) #分为k类，并发数4
print(data_zs,data_zs.shape)
model.fit (data_zs)  # 开始聚类

# 简单打印结果
r1 = pd.Series (model.labels_).value_counts ()  # 统计各个类别的数目
r2 = pd.DataFrame (model.cluster_centers_)  # 找出聚类中心
r = pd.concat ([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
print ('data_zs')
print (r)
r.columns = list (data.columns) + [u'类别数目']  # 重命名表头
print (r)

# 详细输出原始数据及其类别

r = pd.concat ([data, pd.Series (model.labels_, index=data.index)], axis=1)  # 详细输出每个样本对应的类别
r.columns = list (data.columns) + [u'聚类类别']  # 重命名表头
r.to_excel (outputfile)  # 保存结果
