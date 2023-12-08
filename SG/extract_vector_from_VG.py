import csv
import pandas as pd

import pandas as pd

import numpy as np

from scipy import interpolate

df= pd.read_csv("corpus.csv",header=None)
df = df.values.tolist()

z = []

for row in df:

    #z.append(row)
    #print("z=",z)
    import codecs

    f = codecs.open ('direction_obj.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取
    line = f.readline ()  # 以行的形式进行读取文件
    list1 = []
    while line:
        a = line.split ()
        list1.append (a)  # 将其添加在列表之中
        line = f.readline ()
    f.close ()
    subject_bbox=[]
    object_bbox=[]
    relations=[]
    #记录obj文件中的词，是否在VG语料库中，得到其bbox和置信度
    #print(row)
    for i in list1:

           for k in list1:
              print (i, k, "111111", row)
              if i == row[1] or  k == row[7]:
                  print(i,k,"---",row)
                  subject_bbox.append(row[1:5])
                  relations.append(row[6])
                  object_bbox.append(row[8:-1])
                  
    #column1 = [l[i]for l in j]
    #print("SSSS=",subject_bbox)
    #print("OOO=",object_bbox)
    #print("RRR=",relations)