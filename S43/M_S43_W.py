
# coding: utf-8

#双底策略
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#import statsmodels.api as sm
#import math
from yq_tools import chg_factor as chs_factor #载入日线数据
from yq_tools import get_IdxCons as IdxConsGet #载入成分股数据
from yq_tools import get_index_tradeDate as MktIdxdGet #载入指数数据
##股票选取

def get_points(n, index_down, index_up, s_ori ,back_w = 1):
    #print('s_ori',s_ori)
    qsdu_e = index_up[n]
    point_b, point_d = index_down[n - 3 :n - 1]
    point_a, point_c = index_up[n - 3 :n - 1]
    point_a, point_c = s_ori.loc[point_a -back_w - 1 : point_a + back_w].argmax(), s_ori.loc[point_c -back_w - 1: point_c + back_w].argmax() 
    point_b, point_d = s_ori.loc[point_b -back_w - 1: point_b + back_w].argmin(), s_ori.loc[point_d -back_w - 1: point_d + back_w].argmin()
    qsdu_e = s_ori.loc[qsdu_e - back_w - 1: qsdu_e + back_w].argmax()
    #pre_a = index_down[n - 4]
    return point_a, point_b, point_c, point_d, qsdu_e


#加上前期30%的这个条件，找不到信号
def is_bottom(a, b, c, d, e, s_ori, s_high, s_vol):
    ## 前期30% 涨幅
    ## 给定一个时间，在时间内与最低点差30%即可
    #pre_day = a - 200
    #r_pre = s_ori[a] / s_ori[pre_day: a].min() - 1
    #cond1 = r_pre > 0.3  ## bug
    ## a点高于c点 第一个高点大于第二个高点
    cond2 = s_high[a] > s_high[c]
    ## d低于b点 第一个低点大于第二个低点
    cond3 = s_ori[b] > s_ori[d]
    ## ad之间的限制  下降50%以内
    cond5 = s_ori[a] / s_ori[d] - 1 < 0.5
    ## 判断是否出现突破
    ## 其中e点为伪e 第三个高点大于第二个高点
    cond4 = s_ori[e] > s_ori[c]
    if cond2 and cond3 and cond4 and cond5:
        ## 判断e点
        ## 最高点
        e_cond1 = s_high.loc[d:e] > s_high[c]
        e_supply = np.arange(d, e+1)[e_cond1.values]
        # 判断成交量
        # 成交量判断无效
        for i in e_supply:
            if s_vol[i] > s_vol.loc[c:i].mean()*1.2:
                e = i
                return [a, b, c, d], e
                break

## 信号
ticker= '000010'
begin = '20090101'
end = '20190201'
df = chs_factor(ticker= ticker ,begin = begin ,end = end)
def get_stock_signal(df , window = 5, is_plot = True):
    ### 数据获取
    s_ori = df['closePrice'] * df['accumAdjFactor']
    s_high = df['highestPrice'] * df['accumAdjFactor']
    s_vol = df['turnoverVol']
    ## 去噪过程
    trend = s_ori.rolling(window).mean()
    back_w = int(window / 2.0)
    # trend.plot()
    ## 获取去噪之后的高低点
    index_down = signal.find_peaks(-trend)[0]
    index_up = signal.find_peaks(trend)[0]
    index_down = index_down[index_down > index_up[0]]
    ## 获取信号
    e_list = []
    lenth_w = []
    for n in range(5, index_up.shape[0]):
        a,b,c,d,qsdu_e = get_points(n, index_down, index_up, s_ori, back_w)
        if is_bottom(a, b, c, d, qsdu_e, s_ori, s_high, s_vol):
            [a, b ,c, d] , e = is_bottom(a, b, c, d, qsdu_e, s_ori, s_high, s_vol)
            e_list.append(e)
            lenth_w.append(e - a)
    if is_plot:
        s_ori.plot()
        plt.plot(e_list, s_ori[e_list], 'r*')
    return np.array(lenth_w), np.array(e_list)


# In[ ]:

## 统计盈利
def stat_ratio(e_list, df, n):
    return df.loc[e_list + n]['closePrice'].values / df.loc[e_list + 1]['openPrice'].values - 1
## 统计到df
def stock_df(df, elist, lenth_w, n_list = [5, 10, 15, 30]):
    df_1 = df.loc[e_list][['tradeDate', 'ticker']]
    ratio_list = [stat_ratio(e_list, df, n) for n in [5, 10, 15, 30]]
    df_new = pd.DataFrame(ratio_list + [lenth_w], index = ['r_5', 'r_10', 'r_15','r_30', 'lenth']).T
    df_1.index = df_new.index
    return pd.concat([df_1, df_new], axis = 1)

def back_test(df, e_list, hold_days = 5, is_plot = True):
    e_signal = np.hstack([e_list + i for i in range(1, hold_days + 1)])
    e_signal = np.unique(e_signal)
    e_signal = e_signal[e_signal < df.shape[0]]
    e_signal.sort()
    v = np.zeros(df.shape[0])
    v[e_signal] = 1
    
    if is_plot:
        r = ((v * df['chgPct']) + 1).cumprod()
        r.plot()
        (df['chgPct'] + 1).cumprod().plot()
        plt.show()

    return (v * df['chgPct']) + 1

def back_test_more(df, e_list, hold_days = [5, 7, 10 ,15, 20, 30, 35, 60]):
    df_back = pd.DataFrame([back_test(df, e_list, day, is_plot= False) for day in hold_days], index= ['r_%s'%i for i in hold_days]).T
    df_back['tradeDate'] = df['tradeDate']
    df_back['ticker'] = df['ticker']
    return df_back


hs_300_pool = IdxConsGet(u"20171231",ticker=u"000300")
hs_300_pool = hs_300_pool

# ## 策略修改时间段
begin = '20101001'
end = '20200501'
## 以沪深300为例
flag = 0
all_dfs = []
back_dfs = []
for ticker in hs_300_pool:
    df = chs_factor(ticker= ticker ,begin = begin ,end = end)
    lenth_w, e_list = get_stock_signal(df , window = 3, is_plot = False)
    all_dfs.append(stock_df(df, e_list, lenth_w))
    flag += 1
    if flag % 100 == 0:
        print(flag)
    df_back = back_test_more(df, e_list)
    back_dfs.append(df_back)

## 天数分布
tickers_df = pd.concat(all_dfs)
tickers_df['lenth'].hist(bins = 30)


# In[ ]:

## 收益、胜率分析
tickers_df.groupby(pd.qcut(tickers_df['lenth'], 6))['r_5'].mean().plot.bar()
plt.show()
tickers_df.groupby(pd.qcut(tickers_df['lenth'], 6))['r_5'].apply(lambda x: (x > 0).mean()).plot.bar()
plt.show()
tickers_df.groupby(pd.qcut(tickers_df['lenth'], 6))['r_10'].mean().plot.bar()
plt.show()
tickers_df.groupby(pd.qcut(tickers_df['lenth'], 6))['r_10'].apply(lambda x: (x > 0).mean()).plot.bar()
plt.show()

back_df = pd.concat(back_dfs)
#del back_dfs
#back_df.to_csv('back_300.csv') ###################################

hs_300 = MktIdxdGet("000300",begin,end)
hs_300 = pd.Series(hs_300['openIndex'].values / hs_300['openIndex'].values[0] , index= hs_300['tradeDate'])

def stat_back(s, ind):
    shape_r = s[-1] / s.std()
    ratio = 250 * s[-1] / s.shape[0]
    beta, alpha = np.polyfit(ind,s, 1)
    return shape_r, ratio, beta, alpha

res = []
for i in ['r_5', 'r_7', 'r_10', 'r_15', 'r_20', 'r_30', 'r_35', 'r_60']:
    s = back_df.groupby('tradeDate')[i].mean().cumprod()
    result = stat_back(s, hs_300)
    res.append(result)

pd.DataFrame(res, columns= ['shape_r', 'return', 'beta', 'alpha'], index = ['r_5', 'r_7', 'r_10', 'r_15', 'r_20', 'r_30', 'r_35', 'r_60'])

## 策略构建
hs_300.plot()
back_df.groupby('tradeDate')['r_5'].mean().cumprod().plot()
back_df.groupby('tradeDate')['r_7'].mean().cumprod().plot()
back_df.groupby('tradeDate')['r_10'].mean().cumprod().plot()
back_df.groupby('tradeDate')['r_15'].mean().cumprod().plot()
back_df.groupby('tradeDate')['r_20'].mean().cumprod().plot()
back_df.groupby('tradeDate')['r_30'].mean().cumprod().plot()
back_df.groupby('tradeDate')['r_35'].mean().cumprod().plot()
back_df.groupby('tradeDate')['r_60'].mean().cumprod().plot()

