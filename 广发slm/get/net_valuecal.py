# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 11:26:39 2018

@author: caoqiliang
"""
import numpy as np
import pandas as pd
#此程序用于计算利用SLM计算自然语言模型
#构造判断下一时点上涨下跌概率值的判断
import index_cal
index = 3000
data=pd.read_csv('shcomp.csv',index_col=0,parse_dates=True)
#print (data.CLOSE)
def prob(iterable,n):
    #iterable为array_like的可迭代类型
    #表示所用语言模型为n元模型
    #n元模型序列之前的所需序列的可能情况为2**n次方
    #先将iterable转化为0,1标注值
    #iterable值为0表示该日价格比昨日高，值为1表示不比昨日高
    iterable=np.array(iterable)
    label_get=[0 if iterable[i]>iterable[i-1] else 1 for i in range(1,len(iterable))]
    #print (label_get)
    #根据label_get得到2**n次方结果中的可能性概率，为了避免有些可能性未出现过，每个可能性的初始频率赋值为1
    #计算每个可能性出现的次数，并将其加至freq1中
    freq1=[1]*(2**n)
    #print ('freq1')
    #print (freq1)
    #计算frq中每个可能性之后跟随0的次数，加至于freq2中,可得
    freq2=[0]*(2**n)
    #print (freq2)
    
    multiply=[]
    for i in range(n):
        multiply.append(2**i)
    #print ('multiply')
    #print (multiply)
    
    for i in range(n-1,len(label_get)-1):
        bin_need=label_get[i-n+1:i+1]
        #print('bin_need')
        #print (bin_need)
        index_need=(np.array(bin_need)*np.array(multiply)).sum()
        #print('index need')
        #print (index_need)
        freq1[index_need]+=1
        if label_get[i+1]==0:
            freq2[index_need]+=1
    #print ('freq1')
    #print (freq1)
    #print (freq2)
    #以上即可得到贝叶斯概率的各个值
    #下面对明日上涨概率进行判断
    bin_need=label_get[-n:]
    #print('bin need')
    #print(bin_need)
    index_need=(np.array(bin_need)*np.array(multiply)).sum()
    prob=freq2[index_need]/freq1[index_need]
    return prob

#下面可根据prob计算下一日升值的概率
#为了计算准确，设最少有400个数据之后才能计算，即从第401个开始估计其上涨概率
#从第400个开始计算data之后的概率
def prob_cal(data,n):
    probability=[]
    for i in range(index,len(data)):
        prob_need=prob(data[0:i],n)
        probability.append(prob_need)
        #print('prob_need')
        #print(prob_need)
    return probability

def prob_cal_temp(data,n):
    i =len(data)
    prob_need = prob(data[0:i],n)
    return prob_need


#k = prob_cal_temp(data.CLOSE,6)
#print (k)
k = prob_cal(data.CLOSE,6)
print (k)
pr=[]
pr.append(k)
print (len(k))
#for i in range(1,7):
#for i in range(1,7):
    #将1到6阶段模型进行所预测出的概率值放入pr[i]中
#    k = prob_cal(data.CLOSE,i)
#    print (k)
#    pr.append(k)
    
    #pr.append(prob_cal(data.CLOSE,i))
      
#下面进行买入卖出判断，并作图
#初始净值为1,未设置止损线
def net_value(pro,data):
    #pro为对应的概率矩阵
    #data为对应的价格数据,为DataFrame且以datetimeindex为index的类型
    buy_signal=[1 if pro[i]>0.5 else 0 for i in range(len(pro))]
    print(buy_signal)
    #print(buy_signal[0])
    #利用buy_signal进行买卖判断，并计算净值：
    #只做多头，不做空头，因为股票融券难无法做空
    pct_change=pd.DataFrame(data).pct_change().values
    #print (pct_change)
    buy=0 #指示是否处于买入状态，若是，则为1，不是则为0
    net_value=1
    net_value_all=[]
    #if buy_signal[0]==1:
    #    buy=1
    #if buy_signal[0]==0:
    #    buy=0
    for i in range(len(buy_signal)):
        if buy==1:
            net_value=net_value*(1+pct_change[i][0])
        else:
            net_value=net_value #0.03/250设置为每日的现金收益
        if buy_signal[i]==1:
            buy=1
        if buy_signal[i]==0:
            buy=0
        net_value_all.append(net_value)
    return net_value_all
#下面根据pr的值，计算n=1:6时的各个净值的曲线图,分别放在net_value1到net_value5中
print (len(data.CLOSE[index:]))
print (data.CLOSE[index:])
net_value1=net_value(pr[0],data.CLOSE[index:])
print (net_value1)
#net_value2=net_value(pr[1],data.CLOSE[400:])
#net_value3=net_value(pr[2],data.CLOSE[400:])
#net_value4=net_value(pr[3],data.CLOSE[400:])
#net_value5=net_value(pr[4],data.CLOSE[400:])
#net_value6=net_value(pr[5],data.CLOSE[400:])
#做出个图
data_want=pd.DataFrame(data.CLOSE[index:]/data.CLOSE[0])
data_want['net_value1']=net_value1
#data_want['net_value2']=net_value2
#data_want['net_value3']=net_value3
#data_want['net_value4']=net_value4
#data_want['net_value5']=net_value5
#data_want['net_value6']=net_value6

data_want.plot()
index_cal.index_cal_output(data_want)      
        
        
        

       

  






    
    
    
    
    
        
        
        
    
    
