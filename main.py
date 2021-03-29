from function import mlp
from function import cnn
from function import lstm
from function import cleaning
from function import lr
from function import rf
import numpy as np
import sys
import pandas as pd
import os
import re 


news=pd.read_table("text3.csv",names = ["text", "label"],encoding= 'unicode_escape')
###text3前面是Amazon新闻后面是前十的国际新闻####
num=pd.read_table("num2.csv",names = ["open", "high","low","volume","label"])
####数值数据按照OPEN,HIGH,LOW,VOLUME###########
df=cleaning(news)
possi_mlp=mlp(df)
possi_cnn=cnn(df)
possi_ls=lstm(df)
possi_rf=rf(num)
possi_lr=lr(num)
possi1 = np.append(possi_cnn,possi_rf)
possi2 = np.append(possi_mlp,possi_ls)
possi3 = np.append(possi1,possi2)
possi  = np.append(possi3,possi_lr)
equation_inputs = possi
# print(type(equation_inputs))
# print(equation_inputs)
print(equation_inputs)

k = 0
product = np.zeros((1,equation_inputs.shape[0]), dtype = float, order = 'C') #create an empty array
# print(product)
while k<10:
    # prod = equation_inputs[k]*gene[k]
    prod = equation_inputs[k]
    product[0][k] = prod
    #product = np.insert(product,k,values=prod,axis=0)
    #print(prod)
    k=k+1
#product = np.delete(product, 4, 0)  #4*3324
# print("1",product)
product = product.T
# pjob = np.zeros(0,1),dtype = float,order = 'C')
# print(product)
l_p = product[0]+product[2]+product[4]+product[6]+product[8]
r_p = product[1]+product[3]+product[5]+product[7]+product[9]

l_pp = l_p / (l_p+r_p)
r_pp = r_p / (l_p+r_p)
presult = np.append(l_p,r_p)
presult_p = np.append(l_pp,r_pp)
# print(presult_p)
# print(presult)

prediction_last = np.argmax(presult)
print(presult_p[0])
####PRESULT_P[0]##X数据集中第一条信息预测结果的置信度###
print(prediction_last)
#####prediction_last##预测的结果：0是下降，1是预测#####

