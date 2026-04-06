# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 21:55:07 2020

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

def knn_detect(s, df, thr = 0.2):
    def dist(x, y):
        return np.sqrt(np.square(x - y)).mean()
    clu_vec = [s[cols] for cols in df['cols'].values]
    sim = np.array([dist(i,j) for i,j in zip(clu_vec, df['col_vec'])])
    # return sim
    return (np.array([1 if i > 0 else -1 for i in df['row_score'].values])[sim < thr]).sum()

def knn_get_thr_each(s, df):
    def dist(x, y):
        return np.sqrt(np.square(x - y)).mean()
    clu_vec = [s.loc[cols] for cols in df['cols'].values]
    sim = np.array([dist(i,j) for i,j in zip(clu_vec, res_df['col_vec'])])
    return sim

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
# df = pd.read_csv('./data/sample01.csv')
## 选取股票测试
ticker = '000001'
df = chs_factor(ticker=ticker ,begin=u"20090101",end=u"20190101")
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
    
#%% 全局变量
df_factor.dropna(inplace= True)
std_model = MinMaxScaler()
x_scale = std_model.fit_transform(df_factor)
df_scale = pd.DataFrame(x_scale, columns= df_factor.columns, index= df_factor.index )

#%% 标识过程
for label in ['short', 'mid', 'long']:
    if label == 'short':
        frv = df_factor['sma_10'] / df['closePrice'] - 1
    if label == 'mid':
        frv = df_factor['sma_20'] / df['closePrice'] - 1
    if label == 'long':
        frv = df_factor['sma_40'] / df['closePrice'] - 1
    df_factor[label] = frv.apply(lambda x: label_func(x,thr= 0.01))

#%% 聚类过程
model = bicluster(n_clusters= 6, random_state=0)
model.fit(x_scale)
df_factor['clus'] = model.row_labels_
bah = (df['chgPct'] + 1).cumprod()[-1]


#%% 粒子群算法
def sub_fit_func(clu_thr, row_thr, label = 'short'):
    # 对原始类别改造, 建立聚类数据res_df
    res_df = pd.DataFrame(columns=['cols'])
    res_df['row_score'] = df_factor.groupby('clus')[label].mean()
    col_label = pd.DataFrame([model.column_labels_, [col for col in df_factor.columns if '_' in col]],
                             index=['col_label', 'col_name']).T
    res_df['cols'] = col_label.groupby('col_label')['col_name'].apply(lambda x: x.values)
    col_vec = df_scale.groupby(df_factor['clus']).mean()
    res_df.dropna(inplace=True)
    res_df['col_vec'] = [col_vec.loc[clu, res_df.loc[clu, 'cols']].values for clu in res_df.index]
    # 去掉值小的
    if (res_df['row_score'].abs() > row_thr).sum() != 0:
        res_df = res_df[res_df['row_score'].abs() > row_thr]
    else:
        res_df['row_score'] = 1
    # 根据thr获得投资收益
    # 值控制过程
    dec_df = df_scale.apply(lambda x: knn_detect(x, df= res_df, thr= clu_thr), axis=1)
    # 补全过程
    dec_df.replace(0, np.nan, inplace=True)
    # o_chpPct = df['openPrice'].shift(-1) / df['openPrice'] - 1
    # o_chpPct = o_chpPct.fillna(0)
    o_chgPct = df['openPrice'] / df['openPrice'].shift(1) - 1
    signal_df = dec_df.fillna(method='ffill').apply(lambda x: 0 if x > 0 else 1).shift(1)
    # 信号第二天开始操作
    z = o_chgPct.loc[signal_df.index] * signal_df
    v = (z + 1).cumprod()[-1]
    return v

def fit_func(w, ports):
    ports = np.array(ports)
    w = np.zeros(3)
    w[ports.argmax()] = 1
    # 此处采用最高收益作为结果，则三个过程中的最高收益为最优，无需优化
    stage = np.dot(w, ports)
    # print('Stage :%.4f'%stage)
    return stage, w

def compute_v(pbest, gbest, x, v = 0):
    v = v + np.random.uniform(0,1) * (pbest - x) + \
        np.random.uniform(0,1) * (gbest - x)
    # 速度限制
    v[v > 1] = 1
    v[v < -1] = -1
    return v

def compute_wv(pbest, gbest, w, v = 0):
    w_re = w[:2]
    v = 0.8 * v + np.random.random() * (pbest - w_re) + \
        np.random.random() * (gbest - w_re)
    # 速度限制
    v[v > 0.5] = 0.5
    v[v < -0.5] = -0.5
    return v


def update_w(w, wv):
    w_re = w[:2]
    w_re = w_re+ wv
    # w 限制
    w_re[w_re > 0.8] = 0.8
    w_re[w_re < 0] = 0
    w[:2] = w_re
    w[2] = 1 - w_re
    return w

def update_thr(x_thr, v):
    x_thr = x_thr + v
    # 位置限制
    x_thr[x_thr > 0.8] = 0.8
    x_thr[x_thr < 0] = 0
    return x_thr


#%% 初始化参数
x_thr = np.random.uniform(0, 1, size= (10, 2))
pbest_list = np.zeros(shape = (10, 2))
w = np.random.uniform(0, 1, size = 3)
best_score = 0
v_list = np.zeros(10)
fit_ness_list = np.zeros(10)
for _ in range(10):
# 进行粒子群算法选择最优阈值
    for i, thr in enumerate(x_thr):
        clu_thr, row_thr = thr
        ps = sub_fit_func(clu_thr=clu_thr, row_thr=row_thr, label='short')
        pm = sub_fit_func(clu_thr=clu_thr, row_thr=row_thr, label='mid')
        pl = sub_fit_func(clu_thr=clu_thr, row_thr=row_thr, label='long')
        fit_ness, w = fit_func(w, [ps, pm, pl])
        if fit_ness_list[i] < fit_ness:
            fit_ness_list[i] = fit_ness
            pbest_list[i] = thr
    if fit_ness_list.max() > best_score:
        best_score = fit_ness_list.max()
        gbest = pbest_list[fit_ness_list.argmax()]

    v_list = np.array([compute_v(pbest, gbest, thr, v = v) for pbest, thr, v in zip(pbest_list, x_thr, v_list)])
    x_thr = update_thr(x_thr, v_list)
    print('BAH %.4f best stage in %s has %.4f return with %s threads'%(bah, _, best_score, gbest))
    
# 进行knn择时
# 对原始类别改造, 建立聚类数据res_df
res_df = pd.DataFrame(columns= ['cols'])
res_df['row_score'] = df_factor.groupby('clus')[['short', 'mid', 'long'][np.argmax(w)]].mean()
## 去掉值小的
res_df = res_df[res_df['row_score'].abs() > gbest[1]]
col_label = pd.DataFrame([model.column_labels_, [col for col in df_factor.columns if '_' in col]],
                         index= ['col_label', 'col_name']).T
res_df['cols'] = col_label.groupby('col_label')['col_name'].apply(lambda x: x.values)
col_vec = df_scale.groupby(df_factor['clus']).mean()
res_df.dropna(inplace= True)
res_df['col_vec'] = [col_vec.loc[clu, res_df.loc[clu, 'cols']].values for clu in res_df.index]


#%% 检测过程
dec_df = df_scale.apply(lambda x: knn_detect(x, df= res_df, thr= gbest[0]), axis= 1)
## 补全过程
dec_df.replace(0, np.nan, inplace = True)

dec_df.hist(bins= 30)
plt.show()
## 同一买卖点
dec_df = dec_df.apply(lambda x: 1 if x > 0 else x).apply(lambda x:-1 if x < 0 else x)


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
(z + 1).cumprod().plot(legend = 'strategy')
(p + 1).cumprod().plot(legend = 'original')
plt.show()

o_chpPct = df['openPrice']/df['openPrice'].shift(1) - 1
## 按开盘价操作
signal_df = dec_df.fillna(method= 'ffill').apply(lambda x: 0 if x > 0 else 1).shift(1)
z = o_chpPct.loc[signal_df.index] * signal_df
p = o_chpPct.loc[signal_df.index]
(z + 1).cumprod().plot(legend ='strategy')
(p + 1).cumprod().plot(legend ='original')
plt.show()