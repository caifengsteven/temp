# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:12:26 2018

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
df = pd.DataFrame.from_dict(df) 
df.index = pd.to_datetime(df.index)
df = df.sort_index()

"""Step 1.2: 计算布林通道"""    
sp = pd.DataFrame()
sp['close'] = df['close']
sp['ave'], sp['upper'], sp['lower'] = Bolinger_Bands(sp.close, 
                                  window_size=26, num_of_std = 2) 
sp.plot(figsize=(12,9))

"""Step 1.3: 计算指标 %b"""    
sp['%b'] = (sp['close'] - sp['lower'])/(sp['upper'] - sp['lower'])
sp = sp.dropna() #剔除由于计算布林带时，刚开始20天的数据缺失情况
sp['%b'][-50:].plot() #看看最后50天的b数值情况
same_period_plot2(sp,-1,0)#叠加收盘价，看看b与收盘价关系


"""Step 1.4: 指标分割 %b""" 
#原文：我们将阈值δ1设得较小，以在减小噪音的情况下尽量缩短确认一个可能的高低点位所需的时间
#这里直接考虑最短时间情况，也即δ1逼近0的时
sp['Date'] = sp.index
sp = sp.reset_index(drop=True) #需要先reset_index，否在因为index时间下的不连续，导致分割不合理
x = sp['%b'].values
sp['%b_adj'] = Boll_Spilt(x) 
sp['%b_adj'][-50:].plot() #作图观察切割后的%b值
sp['%b_adj'] = sp['%b_adj'].interpolate(method='linear', axis=0) #剔除缺失值，并线性插值
sp[['%b_adj','%b']][-50:].plot() #作图对比，切割后的%b值与原始%b值
      
  
"""Step 1.5: 指标裁剪 %b"""    
""" 1.5.1 分析%b的分布情况，确定sigma的初始值   """  
import seaborn as sns 
sns.set_style('ticks')
mu, sigma = np.nanmean(x), np.nanvar(x) # mean and standard deviation
s = np.random.normal(mu, sigma, 100000) #为了画正太分布图，生成均值、方差与b一致的随机数
p1 = sns.kdeplot(np.array(x),kernel='gau',color="r",legend = True,label =  '%b distribution')
p2 = sns.kdeplot(np.array(s),kernel='gau',legend=True,color="b",label = 'normal distribution')
#sns.plt.show() #画分布图，并与正态分布对比，结果双峰分布
#sns.plt()
sigma1 = 0.5*sp['%b'].std()#sp['%b'].quantile(.25) 
 
#一倍标准差的概率，68.3%；2倍95.5%

""" 1.5.2 单次裁剪分析 """
sp['%b_adj_cut'] = Boll_Cut(Boll_Spilt(x) ,sigma)
sp['%b_adj_cut'] = sp['%b_adj_cut'].interpolate(method='linear', axis=0) #剔除缺失值，并线性插值
sp['%b_adj_cut'][-50:].plot() #作图观察裁剪后的%b值
sp[['%b_adj_cut','%b_adj']][-50:].plot() #作图对比

  
""" 1.5.3 对之进行多次裁剪,每次裁减前，需要重新分割 """
i=0
while i<=5: #这里反复分割裁剪操作5次
    print('currently ' + str(i) + 'th time of  spilting and cutteing data')
    if i ==0:
        v = Boll_Cut(Boll_Spilt(x) ,sigma1)
    else:
        v = Boll_Cut(v ,sigma1)
    i += 1  

sp['%b_adj_cut'] = v #最后再分割一次，使之变成直线Z形；
sp['%b_adj_cut'] = sp['%b_adj_cut'].interpolate(method='linear', axis=0)#剔除缺失值，并线性插值
sp['%b_adj_cut'][-25:].plot() #作图观察裁剪后的%b值
sp[['%b_adj_cut','%b_adj']][-25:].plot() #作图对比
sp[['%b_adj_cut','%b_adj','%b']][-25:].plot() #作图对比

# 备注，这里的sigma取值比较关键，这是一个可以优化的参数；
# 这个参数决定了交易规则；现实中调参回测，其实往往因为这个参数的原因，用到了未来信息；
  
"""Step 1.6: %b分割点返回原始数据 """ 
sp['%b_adj'] = Boll_Spilt(x) 
sp['%b_adj'][-50:].plot()  
mask = sp['%b_adj'].apply(np.isnan) #判定是否为缺失；True缺失；
sp['close_adj'] = np.nan  
sp['close_adj'].loc[mask==False] = sp['close'].loc[mask==False] #返回分割点的原始数据

"""Step 1.7: 分割点映射后，非完美，原始数据进行裁剪""" #备注，sigma的重要性再次显现
sp['close_adj'] = sp['close_adj'].interpolate(method='linear', axis=0) #剔除缺失值，并线性插值
sp[['close_adj','close']][-50:].plot() #作图对比，非完美；

sp['%b_adj'] = Boll_Spilt(x)  
mask = sp['%b_adj'].apply(np.isnan) #判定是否为缺失；True缺失；
sp['close_adj'] = np.nan  
sp['close_adj'].loc[mask==False] = sp['close'].loc[mask==False] #返回分割点的原始数据
sp['close_adj_cut'] = Boll_Cut(sp['close_adj'] ,sigma=250) #sigma确定比较关键，这里依据报告推测
sp['close_adj'] = sp['close_adj'].interpolate(method='linear', axis=0) #剔除缺失值，并线性插值
sp['close_adj_cut'] = sp['close_adj_cut'].interpolate(method='linear', axis=0) #剔除缺失值，并线性插值
#实际上，由于%b分割与收盘价的问题，%b的分割，无法确保收盘价分割到位；因此需要对之再分割裁剪一次；
sp['close_adj_cut'] = Boll_Cut(Boll_Spilt(sp['close_adj_cut']),sigma=250)
sp['close_adj_cut'] = sp['close_adj_cut'].interpolate(method='linear', axis=0) #剔除缺失值，并线性插值

sp['close_adj_cut'][-50:].plot()
sp[['close_adj_cut','close_adj']][-50:].plot() #作图对比，非完美；
sp[['close_adj_cut','close_adj','close']][-50:].plot() #作图对比，非完美；

  
#################################################################################
#################################################################################
