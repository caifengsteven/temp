# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 11:49:58 2018

@author: JIE
"""

import numpy as np
import pickle
import pandas as pd
import itertools

"""PART I. 构建波段形态 """

"""Step 1.1: 导入原始数据 """
stockid = '000300.SH'
with open(stockid+'.pickle','rb') as handle:
        df = pickle.load(handle)
#df = pd.read_csv('000300.csv')
df = pd.DataFrame.from_dict(df) 
df.index = pd.to_datetime(df.index)
df = df.sort_index()

"""Step 1.2: 计算布林通道"""    
sp = pd.DataFrame()
sp['close'] = df['close']
sp['ave'], sp['upper'], sp['lower'] = Bolinger_Bands(sp.close, 
                                  window_size=26, num_of_std = 2) 

"""Step 1.3: 计算指标 %b"""    
sp['%b'] = (sp['close'] - sp['lower'])/(sp['upper'] - sp['lower'])
sp = sp.dropna() #剔除由于计算布林带时，刚开始20天的数据缺失情况

"""Step 1.4: 分割指标 %b"""    
sp['Date'] = sp.index
sp = sp.reset_index(drop=True) #需要先reset_index，否在因为index时间下的不连续，导致分割不合理

x = sp['%b'].values
sp['%b_adj'] = Boll_Spilt(x) #分割

"""Step 1.5: 将分割指标 %b 返回收盘价数据"""      
mask = sp['%b_adj'].apply(np.isnan) #判定是否为缺失；True缺失；
sp['close_adj'] = np.nan  
sp['close_adj'].loc[mask==False] = sp['close'].loc[mask==False] #返回分割点的原始数据

"""Step 1.6: 由于单纯映射不到位，需要再次处理，裁剪，分割"""        
#sp['close_adj_cut'] = Boll_Cut(sp['close_adj'] ,sigma=250) 
sp['close_adj_cut'] = Boll_Cut(sp['close_adj'] ,sigma=100) 
#sigma确定比较关键，对原始数据进行裁剪; sigma越大，意味着更加注重把握大趋势；
sp['close_adj_cut'] = sp['close_adj_cut'].interpolate(method='linear', axis=0) #剔除缺失值，并线性插值
#实际上，由于%b分割与收盘价的问题，%b的分割，无法确保收盘价分割到位；因此需要对之再分割裁剪一次；
#sp['close_adj_cut'] = Boll_Cut(Boll_Spilt(sp['close_adj_cut']),sigma=250)
sp['close_adj_cut'] = Boll_Cut(Boll_Spilt(sp['close_adj_cut']),sigma=100)


"""  五、交易策略设计及实证结果 """
#利用一年的数据作为基准数据，从2011开始回测
#多头组合：买入--持有--直到当前信号消失--卖出；
#空头组合：卖出--持有--直到当前信号消失--买入；

date_list = sp.loc[sp['close_adj_cut'].apply(np.isnan)==False]['Date']
date_list = date_list.apply(lambda x:str(x.date()))

re = [] #记录交易信号
length = 7
for dateid in date_list:
    b = forecast_trend(length,dateid,sp)
    re.append(b)

pfm = cal_performance(re,sp) #依据交易信号，对策略进行评估
