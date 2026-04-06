# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 22:18:24 2020

@author: Asus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering as bicluster
from sklearn.preprocessing import MinMaxScaler

from sqlalchemy import create_engine
import json

import warnings
warnings.filterwarnings('ignore')

#must be set before using
with open('para.json','r',encoding='utf-8') as f:
    para = json.load(f)
    
pn = para['yuqerdata_dir']

user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name1 = 'yuqerdata'
#eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name1)
engine = create_engine(eng_str)

sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''  


def chs_factor(ticker = '000005',begin = None ,end = None , 
               field = [u'symbol',  u'tradeDate', u'openPrice',
                        u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol',
                        u'turnoverValue',u'dealAmount', u'chgPct',
                        'turnoverRate',u'marketValue',u'accumAdjFactor']):
    sql_str1 = sql_str_select_data1 % (','.join(field),ticker,begin,end)
    dataday = pd.read_sql(sql_str1,engine)
    dataday = dataday.applymap(lambda x: np.nan if x == 0 else x)
    dataday.rename(columns={'symbol':'ticker'},inplace=True)
    ## 对数据补全
    return dataday.fillna(method = 'ffill')


def refine_adjfactor(df):
    df_new = df.copy()
    df_new.set_index('tradeDate', inplace= True)
    define_cols = [col for col in df_new.columns if 'Price' in col]
    for col in define_cols:
        ## 价格修正复权
        df_new[col] = df_new[col] * df_new['accumAdjFactor']
        # 对价格缺失日期价格修正
        df_new[col] = df_new[col].replace(0, np.nan).fillna(method = 'ffill')
    return df_new

def mean_index(s, n = 5):
    return s.rolling(n).mean()

def rsi_index(s, n = 5):
    gain = s.rolling(n).apply(lambda x: x[x > 0].mean() if (x > 0).sum() != 0 else 0)
    loss = s.rolling(n).apply(lambda x: - x[x < 0].mean() if (x < 0).sum() != 0 else 0)
    return gain / (gain + loss)

def wr_index(s_h, s_l, s_c, n = 5):
    h_n = s_h.rolling(n).apply(lambda x: x.max())
    l_n = s_l.rolling(n).apply(lambda x: x.min())
    return (h_n - s_c) / (h_n - l_n)

def roc_index(s, n= 5):
    return s / s.shift(n) - 1

def ema_index(s, n=5):
    return s.ewm(n, min_periods= n).mean()

def atr_index(s_h, s_l, s_c, n=5):
    sc_shift = s_c.shift(1)
    tr = pd.concat([s_h - s_l, (s_h - sc_shift ).abs(), (s_l - sc_shift ).abs()], axis= 1).max(axis = 1)
    return tr.rolling(n).mean()

def adx_index(s_h, s_l, s_c, n=5):
    sc_shift = s_c.shift(1)
    tr = pd.concat([s_h - s_l, (s_h - sc_shift).abs(), (s_l - sc_shift).abs()], axis=1).max(axis=1)
    ## 计算hd与ld
    hd = s_h - s_h.shift(1)
    ld = s_l.shift(1) - s_l
    hd_index = hd[(hd <= 0) | (hd < ld)].index
    hd.loc[hd_index] = 0
    ld_index = ld[(ld <= 0) | (ld < hd)].index
    ld.loc[ld_index] = 0

    ## 计算hdi与ldi
    hdi = hd.rolling(n).sum() / tr
    ldi = ld.rolling(n).sum() / tr

    adx = (hdi - ldi).abs() / (hdi + ldi)
    return adx.ewm(span = n).mean()

def label_func(x, thr = 0.005):
    if x> thr:
        l = 1
    elif x < -thr:
        l = -1
    else:
        l = 0
    return l

def knn_detect(s, df , thr = 0.2):
    def dist(x, y):
        return np.sqrt(np.square(x - y)).mean()
    clu_vec = [s[cols] for cols in df['cols'].values]
    sim = np.array([dist(i,j) for i,j in zip(clu_vec, res_df['col_vec'])])
    # return sim
    return (np.array([1 if i > 0 else -1 for i in res_df['row_score'].values])[sim < thr]).sum()

def knn_get_thr_each(s, df):
    def dist(x, y):
        return np.sqrt(np.square(x - y)).mean()
    clu_vec = [s.loc[cols] for cols in df['cols'].values]
    sim = np.array([dist(i,j) for i,j in zip(clu_vec, res_df['col_vec'])])
    return sim


# df = pd.read_csv('./data/sample01.csv')
## 选取股票测试
ticker = '600008'
df = chs_factor(ticker=ticker ,begin=u"20090101",end=u"20200101")
df = refine_adjfactor(df)
df_factor = pd.DataFrame(index= df.index)
# roc
for n in [12, 24, 36]:
    df_factor['roc_' + str(n)] = roc_index(df['closePrice'], n= n)
# ema
for n in [12, 24, 36]:
    df_factor['ema_' + str(n)] = ema_index(df['closePrice'], n= n)
# adx
for n in [14, 28, 42]:
    df_factor['adx_' + str(n)] = adx_index(df['highestPrice'], df['lowestPrice'],
                                           df['closePrice'], n= n)
# atr
for n in [14, 28, 42]:
    df_factor['atr_' + str(n)] = atr_index(df['highestPrice'], df['lowestPrice'],
                                           df['closePrice'], n= n)
# sma
for n in [10, 20, 30, 40]:
    df_factor['sma_' + str(n)] = mean_index(df['closePrice'], n= n)
# wr
for n in [9, 18, 27, 36]:
    df_factor['wr_' + str(n)] = wr_index(df['highestPrice'], df['lowestPrice'],
                                         df['closePrice'], n= n)
# rsi
for n in [6, 12, 18, 24, 30, 36]:
    df_factor['rsi_' + str(n)] = rsi_index(df['chgPct'], n= n)
    
    
# 标准化
df_factor.dropna(inplace= True)
std_model = MinMaxScaler()
x_scale = std_model.fit_transform(df_factor)
df_scale = pd.DataFrame(x_scale, columns= df_factor.columns, index= df_factor.index )
# 构成
frv = df_factor['sma_10'] / df['closePrice'] - 1
df_factor['label'] = frv.apply(lambda x: label_func(x, thr= 0.005))
model = bicluster(n_clusters= 5, random_state=0)
model.fit(x_scale)
df_factor['clus'] = model.row_labels_
print(df_factor.groupby('clus')['label'].mean())


set_b = df_factor.groupby('clus')['label'].apply(lambda x: np.array([x == 1]).sum() / np.float(x.shape[0]))
set_h = df_factor.groupby('clus')['label'].apply(lambda x: np.array([x == 0]).sum() / np.float(x.shape[0]))
set_s = df_factor.groupby('clus')['label'].apply(lambda x: np.array([x == -1]).sum() / np.float(x.shape[0]))

set_bhs = pd.concat([set_b, set_h, set_s], axis = 1)
set_bhs.columns = ['b', 'h', 's']

sup_dict = set_bhs.apply(lambda x: x.argmax(), axis = 1).to_dict()

dec_df = df_factor['clus'].map(sup_dict).map({'s':-1, 'b':1, 'h':np.nan})

flag= 0
sig_value = dec_df.fillna(method= 'ffill').bfill().values
for i, v in enumerate(sig_value):
    if v == flag:
        sig_value[i] = 0
    elif v != flag:
        flag = v
        
s = np.arange(df_scale.shape[0])[sig_value > 0]
b = np.arange(df_scale.shape[0])[sig_value < 0]
print(len(s), len(b))

plt.figure(figsize= (16, 8))
plt.plot(df['closePrice'].values)
for i in b:
    plt.plot(i, df['closePrice'].iloc[i], 'r*')
for i in s:
    plt.plot(i, df['closePrice'].iloc[i], 'y*')
    
plt.show()    


## 原文测试有问题，信号的过程是一个滞后过程，出现信号后第二天才能根据信号进行测试。
signal_df = dec_df.fillna(method= 'ffill').apply(lambda x: 0 if x > 0 else 1).shift(1)
z = df['chgPct'].loc[signal_df.index] * signal_df
p = df['chgPct'].loc[signal_df.index]
(z + 1).cumprod().plot()
(p + 1).cumprod().plot()
plt.show()

