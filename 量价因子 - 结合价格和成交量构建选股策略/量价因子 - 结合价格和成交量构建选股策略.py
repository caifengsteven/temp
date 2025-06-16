# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:19:52 2020

@author: Asus
"""
'''
1. 概述
本文以海通证券《选股因子系列研究(十二)——“量”与“价”的结合》的研究方法为模板，试图分析量价相关关系作为因子的效果：

将股票在短期内的量价走势分类为量价背离与量价同向，并通过量价相关性来衡量量价走势的背离与同向程度
按照量价因子选股的月度多空收益在1%以上，得到了很显著的alpha
纯多头组合在六年回测中年化收益达到22.4%，信息比率达到2.22
量价因子等权叠加了反转因子后，六年回测年化收益达到26.0%，信息比率达到2.55

2. 量价因子构建
股票交易中，最显然的指标无非价格和成交量，大多经典的技术指标其实都是围绕着价格和成交量来构建，本文中尝试将这两者结合起来构建量价因子。中短周期上，量价走势分类为量价背离与量价同向，并通过量价相关性来衡量量价走势的背离与同向的程度。因此，量价相关性，也就是本文中的量价因子，可以简单定义为：

一段时间窗口内，股票收盘价与股票日换手率之间的秩相关系数
本文中的量价相关系数计算，采取的时间窗口为15个交易日

下面给出本文中用来计算量价因子的程序代码
'''


import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import dates
rc('mathtext', default='regular')
import seaborn as sns
sns.set_style('white')
import datetime
import numpy as np
import pandas as pd
import time
import scipy.stats as st
from CAL.PyCAL import *    # CAL.PyCAL中包含font

def getVolPriceCorrAll(universe, begin, end, window, file_name):
    # 计算各股票历史区间window天窗口移动的量价相关系数
    
    # 拿取上海证券交易所日历
    cal_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin, endDate=end).sort('calendarDate')
    cal_dates = cal_dates[cal_dates['isOpen']==1]
    all_dates = cal_dates['calendarDate'].values.tolist()                          # 工作日列表
    
    print str(window) + ' days Price-Volume-Corr will be calculated for ' + str(len(universe)) + ' stocks:'
    count = 0
    secs_time = 0
    start_time = time.time()
    
    ret_data = pd.DataFrame()   # 保存计算出来的收益率数据
    ret_data.to_csv(file_name)
    
    N = 10
    for i in range(len(universe)/N+1):
        sub_univ = universe[i*N:(i+1)*N]
        if len(sub_univ) == 0:
            continue
        data = DataAPI.MktEqudAdjGet(secID=sub_univ, beginDate=begin, endDate=end, 
                                     field='secID,tradeDate,turnoverRate,preClosePrice,closePrice')    # 拿取数据
        for stk in sub_univ:        # 对每一只股票分别计算历史window天前望收益率     
            tmp_ret_data = data[data.secID==stk].sort('tradeDate')
            
            corr_data = range(len(tmp_ret_data))
            for i in range(window-1, len(tmp_ret_data)):
                x = tmp_ret_data['turnoverRate'].values[i-window+1:i+1]
                y = tmp_ret_data['closePrice'].values[i-window+1:i+1]
                corr_data[i] = st.spearmanr(x, y)[0]
                
            # 计算前向收益率
            tmp_ret_data['corr'] = corr_data
            tmp_ret_data = tmp_ret_data[['tradeDate','corr']]
            tmp_ret_data.columns = ['tradeDate', stk]

            ret_data = pd.read_csv(file_name)
            if ret_data.empty:
                ret_data = tmp_ret_data
            else:
                ret_data = ret_data[ret_data.columns[1:]]
                ret_data = ret_data.merge(tmp_ret_data, on='tradeDate', how='outer')

            ret_data = ret_data.sort('tradeDate')
            ret_data.to_csv(file_name)
            
        # 打印进度部分
        count += 1
        if count > 0 and count % 2 == 0:
            finish_time = time.time()
            print count*N,
            print '  ' + str(np.round((finish_time-start_time) - secs_time, 0)) + ' seconds elapsed.'
            secs_time = (finish_time-start_time)
    return ret_data

def getBackwardReturnsAll(universe, begin, end, window, file_name):
    # 计算各股票历史区间回报率，过去window天的收益率
    
    print str(window) + ' days backward returns will be calculated for ' + str(len(universe)) + ' stocks:'
    count = 0
    secs_time = 0
    start_time = time.time()
    
    N = 50
    ret_data = pd.DataFrame()
    for stk in universe:
        data = DataAPI.MktEqudAdjGet(secID=stk, beginDate=begin, endDate=end, 
                                     field='secID,tradeDate,closePrice')    # 拿取数据
        tmp_ret_data = data.sort('tradeDate')
        # 计算历史窗口收益率
        tmp_ret_data['forwardReturns'] = tmp_ret_data['closePrice'] / tmp_ret_data['closePrice'].shift(window) - 1.0
        tmp_ret_data = tmp_ret_data[['tradeDate','forwardReturns']]
        tmp_ret_data.columns = ['tradeDate', stk]

        if ret_data.empty:
            ret_data = tmp_ret_data
        else:
            ret_data = ret_data.merge(tmp_ret_data, on='tradeDate', how='outer')

        # 打印进度部分
        count += 1
        if count > 0 and count % N == 0:
            finish_time = time.time()
            print count,
            print '  ' + str(np.round((finish_time-start_time) - secs_time, 0)) + ' seconds elapsed.'
            secs_time = (finish_time-start_time)
    
    ret_data.to_csv(file_name)
    return ret_data

def getForwardReturnsAll(universe, begin, end, window, file_name):
    # 计算各股票历史区间前瞻回报率，未来window天的收益率
    
    print str(window) + ' days forward returns will be calculated for ' + str(len(universe)) + ' stocks:'
    count = 0
    secs_time = 0
    start_time = time.time()
    
    N = 50
    ret_data = pd.DataFrame()
    for stk in universe:
        data = DataAPI.MktEqudAdjGet(secID=stk, beginDate=begin, endDate=end, 
                                     field='secID,tradeDate,closePrice')    # 拿取数据
        tmp_ret_data = data.sort('tradeDate')
        # 计算历史窗口前瞻收益率
        tmp_ret_data['forwardReturns'] = tmp_ret_data['closePrice'].shift(-window) / tmp_ret_data['closePrice']  - 1.0
        tmp_ret_data = tmp_ret_data[['tradeDate','forwardReturns']]
        tmp_ret_data.columns = ['tradeDate', stk]

        if ret_data.empty:
            ret_data = tmp_ret_data
        else:
            ret_data = ret_data.merge(tmp_ret_data, on='tradeDate', how='outer')

        # 打印进度部分
        count += 1
        if count > 0 and count % N == 0:
            finish_time = time.time()
            print count,
            print '  ' + str(np.round((finish_time-start_time) - secs_time, 0)) + ' seconds elapsed.'
            secs_time = (finish_time-start_time)
    
    ret_data.to_csv(file_name)
    return ret_data


def getMarketValueAll(universe, begin, end, file_name):
    # 获取股票历史每日市值
    
    print  'MarketValue will be calculated for ' + str(len(universe)) + ' stocks:'
    count = 0
    secs_time = 0
    start_time = time.time()
    
    N = 50
    ret_data = pd.DataFrame()
    for stk in universe:
        data = DataAPI.MktEqudAdjGet(secID=stk, beginDate=begin, endDate=end, 
                                     field='secID,tradeDate,marketValue')    # 拿取数据
        tmp_ret_data = data.sort('tradeDate')
        # 市值部分
        tmp_ret_data = tmp_ret_data[['tradeDate','marketValue']]
        tmp_ret_data.columns = ['tradeDate', stk]

        if ret_data.empty:
            ret_data = tmp_ret_data
        else:
            ret_data = ret_data.merge(tmp_ret_data, on='tradeDate', how='outer')

        # 打印进度部分
        count += 1
        if count > 0 and count % N == 0:
            finish_time = time.time()
            print count,
            print '  ' + str(np.round((finish_time-start_time) - secs_time, 0)) + ' seconds elapsed.'
            secs_time = (finish_time-start_time)
    
    ret_data.to_csv(file_name)
    return ret_data

def getWindowMeanTurnoverRateAll(universe, begin, end, window, file_name):
    # 获取股票历史滚动窗口平均换手率
    
    print  'WindowMeanTurnoverRate will be calculated for ' + str(len(universe)) + ' stocks:'
    count = 0
    secs_time = 0
    start_time = time.time()
    
    N = 100
    ret_data = pd.DataFrame()
    for stk in universe:
        data = DataAPI.MktEqudAdjGet(secID=stk, beginDate=begin, endDate=end, 
                                     field='secID,tradeDate,turnoverRate')    # 拿取数据
        tmp_ret_data = data.sort('tradeDate')
        # 市值部分
        tmp_ret_data['windowMeanTurnoverRate'] = pd.rolling_mean(tmp_ret_data['turnoverRate'], window=window)
        tmp_ret_data = tmp_ret_data[['tradeDate','windowMeanTurnoverRate']]
        tmp_ret_data.columns = ['tradeDate', stk]

        if ret_data.empty:
            ret_data = tmp_ret_data
        else:
            ret_data = ret_data.merge(tmp_ret_data, on='tradeDate', how='outer')

        # 打印进度部分
        count += 1
        if count > 0 and count % N == 0:
            finish_time = time.time()
            print count,
            print '  ' + str(np.round((finish_time-start_time) - secs_time, 0)) + ' seconds elapsed.'
            secs_time = (finish_time-start_time)
    
    ret_data.to_csv(file_name)
    return ret_data

'''

上面分别定义了计算本文关心的几个变量的函数，其中包括：

价量相关系数，getVolPriceCorrAll
历史收益率，getBackwardReturnsAll
未来收益率，getForwardReturnsAll
市值，getMarketValueAll
历史窗口日均换手率，getWindowMeanTurnoverRateAll
下面利用这五个函数分别计算我们需要的各种变量（我们只用了全A股中的50只作为示例，感兴趣的读者只需要将下面cell中第5行中的universe修改即可计算更大股票池的数据），并将这些变量保存在文件中以供调用。


'''

begin_date = '20060101'  # 开始日期
end_date = '20180626'    # 结束日期

universe = set_universe('HS300')      # 股票池
universe = universe[0:300]         # 计算速度缓慢，仅以部分股票池作为sample

# ----------- 计算量价相关系数部分 ----------------
window_corr = 15                
print ('=======================')
start_time = time.time()
forward_returns_data = getVolPriceCorrAll(universe=universe, begin=begin_date, end=end_date, window=window_corr, file_name='VolPriceCorr_W15_CSI300_sample.csv')
finish_time = time.time()
print ('')
print (str(finish_time-start_time) + ' seconds elapsed in total.')

# ----------- 计算股票历史窗口（一个月）收益率部分 ----------------
window_return = 20                
print ('=======================')
start_time = time.time()
forward_returns_data = getBackwardReturnsAll(universe=universe, begin=begin_date, end=end_date, window=window_return, file_name='BackwardReturns_W20_CSI300_Sample.csv')
finish_time = time.time()
print ('')
print (str(finish_time-start_time) + ' seconds elapsed in total.')

# ----------- 计算股票历史窗口（三个月）收益率部分 ----------------
window_return = 60                
print ('=======================')
start_time = time.time()
forward_returns_data = getBackwardReturnsAll(universe=universe, begin=begin_date, end=end_date, window=window_return, file_name='BackwardReturns_W60_CSI300_Sample.csv')
finish_time = time.time()
print ('')
print (str(finish_time-start_time) + ' seconds elapsed in total.')

# ----------- 计算股票前瞻收益率部分 ----------------
window_return = 20                
print ('=======================')
start_time = time.time()
forward_returns_data = getForwardReturnsAll(universe=universe, begin=begin_date, end=end_date, window=window_return, file_name='ForwardReturns_W20_CSI300_Sample.csv')
finish_time = time.time()
print ('')
print (str(finish_time-start_time) + ' seconds elapsed in total.')

# ----------- 计算股票历史市值部分 ----------------
print ('=======================')
start_time = time.time()
forward_returns_data = getMarketValueAll(universe=universe, begin=begin_date, end=end_date, file_name='MarketValues_CSI300_Sample.csv')
finish_time = time.time()
print ('')
print (str(finish_time-start_time) + ' seconds elapsed in total.')

# ----------- 计算历史月度日均换手率部分 ----------------
window = 20                
print ('=======================')
start_time = time.time()
forward_returns_data = getWindowMeanTurnoverRateAll(universe=universe, begin=begin_date, end=end_date, window=window, file_name='TurnoverRateWindowMean_W20_CSI300_Sample.csv')
finish_time = time.time()
print ('')
print (str(finish_time-start_time) + ' seconds elapsed in total.')


# 提取数据
corr_data = pd.read_csv('VolPriceCorr_W15_CSI300_sample.csv')                    # 15天窗口量价相关系数
forward_20d_return_data = pd.read_csv('ForwardReturns_W20_CSI300_Sample.csv')    # 未来20天收益率    
backward_20d_return_data = pd.read_csv('BackwardReturns_W20_CSI300_Sample.csv')  # 过去20天收益率 
backward_60d_return_data = pd.read_csv('BackwardReturns_W60_CSI300_Sample.csv')  # 过去60天收益率 
mkt_value_data = pd.read_csv('MarketValues_CSI300_Sample.csv')                    # 市值数据
turnover_rate_data = pd.read_csv('TurnoverRateWindowMean_W20_CSI300_Sample.csv') # 过去20天日均换手率数据

corr_data['tradeDate'] = map(Date.toDateTime, map(DateTime.parseISO, corr_data['tradeDate']))
forward_20d_return_data['tradeDate'] = map(Date.toDateTime, map(DateTime.parseISO, forward_20d_return_data['tradeDate']))
backward_20d_return_data['tradeDate'] = map(Date.toDateTime, map(DateTime.parseISO, backward_20d_return_data['tradeDate']))
backward_60d_return_data['tradeDate'] = map(Date.toDateTime, map(DateTime.parseISO, backward_60d_return_data['tradeDate']))
mkt_value_data['tradeDate'] = map(Date.toDateTime, map(DateTime.parseISO, mkt_value_data['tradeDate']))
turnover_rate_data['tradeDate'] = map(Date.toDateTime, map(DateTime.parseISO, turnover_rate_data['tradeDate']))

corr_data = corr_data[corr_data.columns[1:]].set_index('tradeDate')
forward_20d_return_data = forward_20d_return_data[forward_20d_return_data.columns[1:]].set_index('tradeDate')
backward_20d_return_data = backward_20d_return_data[backward_20d_return_data.columns[1:]].set_index('tradeDate')
backward_60d_return_data = backward_60d_return_data[backward_60d_return_data.columns[1:]].set_index('tradeDate')
mkt_value_data = mkt_value_data[mkt_value_data.columns[1:]].set_index('tradeDate')
turnover_rate_data = turnover_rate_data[turnover_rate_data.columns[1:]].set_index('tradeDate')


# 量价相关性历史表现

n_quantile = 10
# 和海通研报一样，统计十分位数
cols_mean = ['meanQ'+str(i+1) for i in range(n_quantile)]
cols = cols_mean
corr_means = pd.DataFrame(index=corr_data.index, columns=cols)

# 计算相关系数分组平均值
for dt in corr_means.index:
    qt_mean_results = []

    # 相关系数去掉nan和绝对值大于1的
    tmp_corr = corr_data.ix[dt].dropna()
    tmp_corr = tmp_corr[(tmp_corr<=1.0) & (tmp_corr>=-1.0)]
    
    pct_quantiles = 1.0/n_quantile
    for i in range(n_quantile):
        down = tmp_corr.quantile(pct_quantiles*i)
        up = tmp_corr.quantile(pct_quantiles*(i+1))
        mean_tmp = tmp_corr[(tmp_corr<=up) & (tmp_corr>=down)].mean()
        qt_mean_results.append(mean_tmp)
    corr_means.ix[dt] = qt_mean_results

# corr_means是对历史每一天，求量价相关系数在各个十分位里面的平均值
corr_means.tail()


# 量价相关性历史表现作图

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(111)

lns1 = ax1.plot(corr_means.index, corr_means.meanQ1, label='Q1')
lns2 = ax1.plot(corr_means.index, corr_means.meanQ5, label='Q5')
lns3 = ax1.plot(corr_means.index, corr_means.meanQ10, label='Q10')

lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=[0.5, 0.1], loc='', ncol=3, mode="", borderaxespad=0., fontsize=12)
ax1.set_ylabel(u'量价相关系数', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'日期', fontproperties=font, fontsize=16)
ax1.set_title(u"量价相关性历史表现", fontproperties=font, fontsize=16)
ax1.grid()


# ‘过去十五天量价相关系数’和‘之后20天收益’的秩相关系数计算

ic_data = pd.DataFrame(index=corr_data.index, columns=['IC','pValue'])

# 计算相关系数
for dt in ic_data.index:
    tmp_corr = corr_data.ix[dt]
    tmp_ret = forward_20d_return_data.ix[dt]
    cor = pd.DataFrame(tmp_corr)
    ret = pd.DataFrame(tmp_ret)
    cor.columns = ['corr']
    ret.columns = ['ret']
    cor['ret'] = ret['ret']
    cor = cor[~np.isnan(cor['corr'])][~np.isnan(cor['ret'])]
    if len(cor) < 5:
        continue
    # ic,p_value = st.pearsonr(q['Q'],q['ret'])                 # 计算相关系数   IC
    # ic,p_value = st.pearsonr(q['Q'].rank(),q['ret'].rank())   # 计算秩相关系数 RankIC
    ic, p_value = st.spearmanr(cor['corr'],cor['ret'])   # 计算秩相关系数 RankIC
    ic_data['IC'][dt] = ic
    ic_data['pValue'][dt] = p_value
    
# print len(ic_data['IC']), len(ic_data[ic_data.IC>0]), len(ic_data[ic_data.IC<0])
print ('mean of IC: ', ic_data['IC'].mean())
print ('median of IC: ', ic_data['IC'].median())
print ('the number of IC(all, plus, minus): ', (len(ic_data), len(ic_data[ic_data.IC>0]), len(ic_data[ic_data.IC<0])))

# ‘过去十五天量价相关系数’和‘之后20天收益’的秩相关系数作图

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(111)

lns1 = ax1.plot(ic_data.index, ic_data.IC, label='IC')

lns = lns1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=[0.5, 0.1], loc='', ncol=3, mode="", borderaxespad=0., fontsize=12)
#ax1.set_ylabel(u'相关系数', fontproperties=font, fontsize=16)
#ax1.set_xlabel(u'日期', fontproperties=font, fontsize=16)
#ax1.set_title(u"量价因子和未来20日收益之间的秩相关系数", fontproperties=font, fontsize=16)
ax1.set_ylabel(u'IC', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'Date', fontproperties=font, fontsize=16)
ax1.set_title(u"PV factor and 20 days forward return IC", fontproperties=font, fontsize=16)

ax1.grid()

n_quantile = 10
# 和海通研报一样，统计十分位数
cols_mean = [i+1 for i in range(n_quantile)]
cols = cols_mean

excess_returns_means = pd.DataFrame(index=corr_data.index, columns=cols)

# 计算相关系数分组的超额收益平均值
for dt in excess_returns_means.index:
    qt_mean_results = []
    
    # 相关系数去掉nan和绝对值大于1的
    tmp_corr = corr_data.ix[dt].dropna()
    tmp_corr = tmp_corr[(tmp_corr<=1.0) & (tmp_corr>=-1.0)]
    tmp_return = forward_20d_return_data.ix[dt].dropna()
    tmp_return_mean = tmp_return.mean()
    
    pct_quantiles = 1.0/n_quantile
    for i in range(n_quantile):
        down = tmp_corr.quantile(pct_quantiles*i)
        up = tmp_corr.quantile(pct_quantiles*(i+1))
        i_quantile_index = tmp_corr[(tmp_corr<=up) & (tmp_corr>=down)].index
        mean_tmp = tmp_return[i_quantile_index].mean() - tmp_return_mean
        qt_mean_results.append(mean_tmp)
        
    excess_returns_means.ix[dt] = qt_mean_results

excess_returns_means.dropna(inplace=True)
excess_returns_means.tail()


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

excess_returns_means_dist = excess_returns_means.mean()
# lns1 = ax1.plot(excess_returns_means_dist.index, excess_returns_means_dist.values, '--o', label='IC')
excess_dist_plus = excess_returns_means_dist[excess_returns_means_dist>0]
excess_dist_minus = excess_returns_means_dist[excess_returns_means_dist<0]
lns2 = ax1.bar(excess_dist_plus.index, excess_dist_plus.values, align='center', color='r', width=0.35)
lns3 = ax1.bar(excess_dist_minus.index, excess_dist_minus.values, align='center', color='g', width=0.35)

ax1.set_xlim(left=0.5, right=len(excess_returns_means_dist)+0.5)
ax1.set_ylim(-0.01, 0.004)
ax1.set_ylabel(u'超额收益', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'十分位分组', fontproperties=font, fontsize=16)
ax1.set_xticks(excess_returns_means_dist.index)
ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
ax1.set_yticklabels([str(x*100)+'0%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
ax1.set_title(u"量价相关性选股因子超额收益", fontproperties=font, fontsize=16)
ax1.grid()


n_quantile = 10
# 和海通研报一样，统计十分位数
cols_mean = [i+1 for i in range(n_quantile)]
cols = cols_mean

mkt_value_means = pd.DataFrame(index=corr_data.index, columns=cols)

# 计算相关系数分组的超额收益平均值
for dt in mkt_value_means.index:
    qt_mean_results = []
    
    # 相关系数去掉nan和绝对值大于1的
    tmp_corr = corr_data.ix[dt].dropna()
    tmp_corr = tmp_corr[(tmp_corr<=1.0) & (tmp_corr>=-1.0)]
    tmp_mkt_value = mkt_value_data.ix[dt].dropna()
    tmp_mkt_value = tmp_mkt_value.rank()/len(tmp_mkt_value)
    
    pct_quantiles = 1.0/n_quantile
    for i in range(n_quantile):
        down = tmp_corr.quantile(pct_quantiles*i)
        up = tmp_corr.quantile(pct_quantiles*(i+1))
        i_quantile_index = tmp_corr[(tmp_corr<=up) & (tmp_corr>=down)].index
        mean_tmp = tmp_mkt_value[i_quantile_index].mean()
        qt_mean_results.append(mean_tmp)
        
    mkt_value_means.ix[dt] = qt_mean_results

mkt_value_means.dropna(inplace=True)
mkt_value_means.tail()

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

mkt_value_means_dist = mkt_value_means.mean()
lns1 = ax1.bar(mkt_value_means_dist.index, mkt_value_means_dist.values, align='center', width=0.35)
lns2 = ax2.plot(excess_returns_means_dist.index, excess_returns_means_dist.values, 'o-r')

ax1.legend(lns1, ['market value(left axis)'], loc=2, fontsize=12)
ax2.legend(lns2, ['excess return(right axis)'], fontsize=12)
ax1.set_ylim(0.4, 0.6)
ax2.set_ylim(-0.01, 0.004)
ax1.set_xlim(left=0.5, right=len(mkt_value_means_dist)+0.5)
ax1.set_ylabel(u'市值百分位数', fontproperties=font, fontsize=16)
ax2.set_ylabel(u'超额收益', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'十分位分组', fontproperties=font, fontsize=16)
ax1.set_xticks(mkt_value_means_dist.index)
ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
ax1.set_yticklabels([str(x*100)+'%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
ax2.set_yticklabels([str(x*100)+'0%' for x in ax2.get_yticks()], fontproperties=font, fontsize=14)
ax1.set_title(u"量价相关性选股因子市值分布特征", fontproperties=font, fontsize=16)
ax1.grid()

n_quantile = 10
# 和海通研报一样，统计十分位数
cols_mean = [i+1 for i in range(n_quantile)]
cols = cols_mean
turnover_rate_means = pd.DataFrame(index=corr_data.index, columns=cols)

# 计算相关系数分组的超额收益平均值
for dt in turnover_rate_means.index:
    qt_mean_results = []
    
    # 相关系数去掉nan和绝对值大于1的
    tmp_corr = corr_data.ix[dt].dropna()
    tmp_corr = tmp_corr[(tmp_corr<=1.0) & (tmp_corr>=-1.0)]
    tmp_turnover_rate = turnover_rate_data.ix[dt].dropna()
    
    pct_quantiles = 1.0/n_quantile
    for i in range(n_quantile):
        down = tmp_corr.quantile(pct_quantiles*i)
        up = tmp_corr.quantile(pct_quantiles*(i+1))
        i_quantile_index = tmp_corr[(tmp_corr<=up) & (tmp_corr>=down)].index
        mean_tmp = tmp_turnover_rate[i_quantile_index].mean()
        qt_mean_results.append(mean_tmp)
        
    turnover_rate_means.ix[dt] = qt_mean_results

turnover_rate_means.dropna(inplace=True)
turnover_rate_means.tail()


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

turnover_rate_means_dist = turnover_rate_means.mean()
lns1 = ax1.bar(turnover_rate_means_dist.index, turnover_rate_means_dist.values, align='center', width=0.35)
lns2 = ax2.plot(excess_returns_means_dist.index, excess_returns_means_dist.values, 'o-r')

ax1.legend(lns1, ['turnover rate(left axis)'], loc=2, fontsize=12)
ax2.legend(lns2, ['excess return(right axis)'], fontsize=12)
ax1.set_ylim(0, 0.05)
ax2.set_ylim(-0.01, 0.004)
ax1.set_xlim(left=0.5, right=len(turnover_rate_means_dist)+0.5)
ax1.set_ylabel(u'换手率', fontproperties=font, fontsize=16)
ax2.set_ylabel(u'超额收益', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'十分位分组', fontproperties=font, fontsize=16)
ax1.set_xticks(turnover_rate_means_dist.index)
ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
ax1.set_yticklabels([str(x*100)+'%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
ax2.set_yticklabels([str(x*100)+'0%' for x in ax2.get_yticks()], fontproperties=font, fontsize=14)
ax1.set_title(u"量价相关性选股因子换手率分布特征", fontproperties=font, fontsize=16)
ax1.grid()

'''

4.4 量价因子选股的一个月反转分布特征

'''


n_quantile = 10
# 和海通研报一样，统计十分位数
cols_mean = [i+1 for i in range(n_quantile)]
cols = cols_mean
hist_returns_means = pd.DataFrame(index=corr_data.index, columns=cols)

# 计算相关系数分组的超额收益平均值
for dt in hist_returns_means.index:
    qt_mean_results = []
    
    # 相关系数去掉nan和绝对值大于1的
    tmp_corr = corr_data.ix[dt].dropna()
    tmp_corr = tmp_corr[(tmp_corr<=1.0) & (tmp_corr>=-1.0)]
    tmp_return = backward_20d_return_data.ix[dt].dropna()
    tmp_return_mean = tmp_return.mean()
    
    pct_quantiles = 1.0/n_quantile
    for i in range(n_quantile):
        down = tmp_corr.quantile(pct_quantiles*i)
        up = tmp_corr.quantile(pct_quantiles*(i+1))
        i_quantile_index = tmp_corr[(tmp_corr<=up) & (tmp_corr>=down)].index
        mean_tmp = tmp_return[i_quantile_index].mean() - tmp_return_mean
        qt_mean_results.append(mean_tmp)
        
    hist_returns_means.ix[dt] = qt_mean_results

hist_returns_means.dropna(inplace=True)
hist_returns_means.tail()

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

hist_returns_means_dist = hist_returns_means.mean()
lns1 = ax1.bar(hist_returns_means_dist.index, hist_returns_means_dist.values, align='center', width=0.35)
lns2 = ax2.plot(excess_returns_means_dist.index, excess_returns_means_dist.values, 'o-r')

ax1.legend(lns1, ['20 day return(left axis)'], loc=2, fontsize=12)
ax2.legend(lns2, ['excess return(right axis)'], fontsize=12)
ax1.set_ylim(-0.03, 0.07)
ax2.set_ylim(-0.01, 0.004)
ax1.set_xlim(left=0.5, right=len(hist_returns_means_dist)+0.5)
ax1.set_ylabel(u'历史一个月收益率', fontproperties=font, fontsize=16)
ax2.set_ylabel(u'超额收益', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'十分位分组', fontproperties=font, fontsize=16)
ax1.set_xticks(hist_returns_means_dist.index)
ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
ax1.set_yticklabels([str(x*100)+'%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
ax2.set_yticklabels([str(x*100)+'0%' for x in ax2.get_yticks()], fontproperties=font, fontsize=14)
ax1.set_title(u"量价相关性选股因子一个月历史收益率（一个月反转因子）分布特征", fontproperties=font, fontsize=16)
ax1.grid()


'''
4.5 量价因子选股的三个月反转分布特征

'''

n_quantile = 10
# 和海通研报一样，统计十分位数
cols_mean = [i+1 for i in range(n_quantile)]
cols = cols_mean
hist_returns_means = pd.DataFrame(index=corr_data.index, columns=cols)

# 计算相关系数分组的超额收益平均值
for dt in hist_returns_means.index:
    qt_mean_results = []
    
    # 相关系数去掉nan和绝对值大于1的
    tmp_corr = corr_data.ix[dt].dropna()
    tmp_corr = tmp_corr[(tmp_corr<=1.0) & (tmp_corr>=-1.0)]
    tmp_return = backward_60d_return_data.ix[dt].dropna()
    tmp_return_mean = tmp_return.mean()
    
    pct_quantiles = 1.0/n_quantile
    for i in range(n_quantile):
        down = tmp_corr.quantile(pct_quantiles*i)
        up = tmp_corr.quantile(pct_quantiles*(i+1))
        i_quantile_index = tmp_corr[(tmp_corr<=up) & (tmp_corr>=down)].index
        mean_tmp = tmp_return[i_quantile_index].mean() - tmp_return_mean
        qt_mean_results.append(mean_tmp)
        
    hist_returns_means.ix[dt] = qt_mean_results

hist_returns_means.dropna(inplace=True)
hist_returns_means.tail()

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

hist_returns_means_dist = hist_returns_means.mean()
lns1 = ax1.bar(hist_returns_means_dist.index, hist_returns_means_dist.values, align='center', width=0.35)
lns2 = ax2.plot(excess_returns_means_dist.index, excess_returns_means_dist.values, 'o-r')

ax1.legend(lns1, ['60 day return(left axis)'], loc=2, fontsize=12)
ax2.legend(lns2, ['excess return(right axis)'], fontsize=12)
ax1.set_ylim(-0.02, 0.04)
ax2.set_ylim(-0.01, 0.004)
ax1.set_xlim(left=0.5, right=len(hist_returns_means_dist)+0.5)
ax1.set_ylabel(u'历史三个月收益率', fontproperties=font, fontsize=16)
ax2.set_ylabel(u'超额收益', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'十分位分组', fontproperties=font, fontsize=16)
ax1.set_xticks(hist_returns_means_dist.index)
ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
ax1.set_yticklabels([str(x*100)+'%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
ax2.set_yticklabels([str(x*100)+'0%' for x in ax2.get_yticks()], fontproperties=font, fontsize=14)
ax1.set_title(u"量价相关性选股因子三个月历史收益率（三个月反转因子）分布特征", fontproperties=font, fontsize=16)
ax1.grid()

'''

5. 量价因子历史回测净值表现
接下来，考察上述量价因子的选股能力的回测效果。历史回测的基本设置如下：

回测时段为2010年1月1日至2016年8月1日
股票池为A股全部股票
组合每15个交易日调仓，交易费率设为双边万分之二
调仓时，涨停、停牌不买入，跌停、停牌不卖出；
每月底调仓时，选择股票池中量价因子最小的20%的股票；
5.1 量价因子最小20%股票

start = '2010-01-01'                       # 回测起始时间
end = '2018-06-26'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')               # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 15                          # 调仓频率，表示执行handle_data的时间间隔

corr_data = pd.read_csv('VolPriceCorr_W15_CSI300_sample.csv')     # 读取量价因子数据
corr_data = corr_data[corr_data.columns[1:]].set_index('tradeDate')
corr_dates = corr_data.index.values

quantile_five = 1                           # 选取股票的量价因子五分位数，1表示选取股票池中因子最小的10%的股票
commission = Commission(0.0002,0.0002)      # 交易费率设为双边万分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in corr_dates:            # 只在计算过量价因子的交易日调仓
        return
    
    # 拿取调仓日前一个交易日的量价因子，并按照相应十分位选择股票
    pre_corr = corr_data.ix[pre_date]
    pre_corr = pre_corr.dropna()
    pre_corr = pre_corr[(pre_corr<=1.0) & (pre_corr>=-1.0)]
    
    pre_corr_min = pre_corr.quantile((quantile_five-1)*0.2)
    pre_corr_max = pre_corr.quantile(quantile_five*0.2)
    my_univ = pre_corr[pre_corr>=pre_corr_min][pre_corr<pre_corr_max].index.values
    
    # 调仓逻辑
    univ = [x for x in my_univ if x in account.universe]
    
    # 不在股票池中的，清仓
    for stk in account.valid_secpos:
        if stk not in univ:
            order_to(stk, 0)
    # 在目标股票池中的，等权买入
    for stk in univ:
        order_pct_to(stk, 1.1/len(univ))
        
'''
bt_all = {}   # 用来保存三个策略运行结果：量价因子，20日反转因子，量价因子与20日反转因子等权重叠加
bt_all['corr'] = bt   # 保存量价因子回测结果

'''
5.2 一个月反转因子最小（近一个月涨幅最低的）20%股票

start = '2010-01-01'                       # 回测起始时间
end = '2018-06-26'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')               # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 15                          # 调仓频率，表示执行handle_data的时间间隔

revs_data = pd.read_csv('BackwardReturns_W20_CSI300_Sample.csv')     # 读取反转因子数据
revs_data = revs_data[revs_data.columns[1:]].set_index('tradeDate')
revs_dates = revs_data.index.values

quantile_five = 1                           # 选取股票的20日反转因子的五分位数，1表示选取股票池中因子最小的20%的股票
commission = Commission(0.0002,0.0002)     # 交易费率设为双边万分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in revs_dates:            # 只在计算过反转因子的交易日调仓
        return
    
    # 拿取调仓日前一个交易日的反转因子，并按照相应十分位选择股票
    pre_revs = revs_data.ix[pre_date]
    pre_revs = pre_revs.dropna()
    
    pre_revs_min = pre_revs.quantile((quantile_five-1)*0.2)
    pre_revs_max = pre_revs.quantile(quantile_five*0.2)
    my_univ = pre_revs[pre_revs>=pre_revs_min][pre_revs<pre_revs_max].index.values
    
    # 调仓逻辑
    univ = [x for x in my_univ if x in account.universe]
    
    # 不在股票池中的，清仓
    for stk in account.valid_secpos:
        if stk not in univ:
            order_to(stk, 0)
    # 在目标股票池中的，等权买入
    for stk in univ:
        order_pct_to(stk, 1.1/len(univ))

5.3 量价因子叠加反转因子选股

量价因子和反转因子分别标准化，之后相加生成叠加因子，选叠加因子最小的20%股票


start = '2010-01-01'                       # 回测起始时间
end = '2018-06-26'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')               # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 15                          # 调仓频率，表示执行handle_data的时间间隔

corr_data = pd.read_csv('VolPriceCorr_W15_CSI300_sample.csv')     # 读取量价因子数据
corr_data = corr_data[corr_data.columns[1:]].set_index('tradeDate')
corr_dates = corr_data.index.values

revs_data = pd.read_csv('BackwardReturns_W20_CSI300_Sample.csv')     # 读取反转因子数据
revs_data = revs_data[revs_data.columns[1:]].set_index('tradeDate')

quantile_five = 1                           # 选取股票的因子五分位数，1表示选取股票池中因子最小的20%的股票
commission = Commission(0.0002,0.0002)     # 交易费率设为双边万分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in corr_dates:            # 只在计算过量价因子的交易日调仓
        return

    # 拿取调仓日前一个交易日的量价因子和反转因子，并按照相应分位选择股票
    pre_corr = corr_data.ix[pre_date]
    pre_corr = pre_corr[(pre_corr<=1.0) & (pre_corr>=-1.0)]
    pre_revs = revs_data.ix[pre_date]
    
    # 量价因子和反转因子只做简单的等权叠加
    pre_data = pd.Series(standardize(pre_corr.to_dict())) + pd.Series(standardize(pre_revs.to_dict()))   # 因子标准化使用了uqer的函数standardize
    pre_data = pre_data.dropna()
    
    pre_data_min = pre_data.quantile((quantile_five-1)*0.2)
    pre_data_max = pre_data.quantile(quantile_five*0.2)
    my_univ = pre_data[pre_data>=pre_data_min][pre_data<pre_data_max].index.values
    
    # 调仓逻辑
    univ = [x for x in my_univ if x in account.universe]
    
    # 不在股票池中的，清仓
    for stk in account.valid_secpos:
        if stk not in univ:
            order_to(stk, 0)
    # 在目标股票池中的，等权买入
    for stk in univ:
        order_pct_to(stk, 1.1/len(univ))
        
bt_all['corr + revs'] = bt


'''


results = {}
for x in bt_all.keys():
    results[x] = {}
    results[x]['bt'] = bt_all[x]
    
fig = plt.figure(figsize=(10,8))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.grid()
ax2.grid()

for qt in ['corr','revs','corr + revs']:
    bt = results[qt]['bt']

    data = bt[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0   # 总头寸每日回报率
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return                 # 总头寸每日超额回报率
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()                # 总头寸对冲指数后的净值序列
    data['portfolio'] = data.portfolio_return + 1.0     
    data['portfolio'] = data.portfolio.cumprod()          # 总头寸不对冲时的净值序列
    data['benchmark'] = data.benchmark_return + 1.0
    data['benchmark'] = data.benchmark.cumprod()          # benchmark的净值序列
    results[qt]['hedged_max_drawdown'] = max([1 - v/max(1, max(data['excess'][:i+1])) for i,v in enumerate(data['excess'])])  # 对冲后净值最大回撤
    results[qt]['hedged_volatility'] = np.std(data['excess_return'])*np.sqrt(252)
    results[qt]['hedged_annualized_return'] = (data['excess'].values[-1])**(252.0/len(data['excess'])) - 1.0
    # data[['portfolio','benchmark','excess']].plot(figsize=(12,8))
    # ax.plot(data[['portfolio','benchmark','excess']], label=str(qt))
    ax1.plot(data['tradeDate'], data[['portfolio']], label=str(qt))
    ax2.plot(data['tradeDate'], data[['excess']], label=str(qt))
    

ax1.legend(loc=0, fontsize=12)
ax2.legend(loc=0, fontsize=12)
ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲净值", fontproperties=font, fontsize=16)
ax1.set_title(u"量价因子和反转因子选股能力对比 - 净值走势", fontproperties=font, fontsize=16)
ax2.set_title(u"量价因子和反转因子选股能力对比 - 对冲中证500指数后净值走势", fontproperties=font, fontsize=16)

'''
上图中可以发现：

蓝色曲线为量价因子，绿色为反转因子，红色为量价因子叠加反转因子
量价因子的漫长的熊市中走势稳健，并一直打败反转因子
反转因子在15年之后表现出色
量价因子叠加反转因子，能起到意想不到的叠加效果

5.5 量价因子选股 —— 不同五分位数组合回测走势比较

'''

# 可编辑部分与 strategy 模式一样，其余部分按本例代码编写即可

# -----------回测参数部分开始，可编辑------------
start = '2010-01-01'                       # 回测起始时间
end = '2018-06-26'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')               # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 15                          # 调仓频率，表示执行handle_data的时间间隔

corr_data = pd.read_csv('VolPriceCorr_W15_CSI300_sample.csv')     # 读取量价因子数据
corr_data = corr_data[corr_data.columns[1:]].set_index('tradeDate')
corr_dates = corr_data.index.values
# ---------------回测参数部分结束----------------


# 把回测参数封装到 SimulationParameters 中，供 quick_backtest 使用
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base)
# 获取回测行情数据
idxmap, data = quartz.get_daily_data(sim_params)
# 运行结果
results_corr = {}

# 调整参数(选取股票的量价因子五分位数)，进行快速回测
for quantile_five in range(1, 6):
    
    # ---------------策略逻辑部分----------------
    commission = Commission(0.0002,0.0002)      # 交易费率设为双边万分之二

    def initialize(account):                   # 初始化虚拟账户状态
        pass

    def handle_data(account):                  # 每个交易日的买入卖出指令
        pre_date = account.previous_date.strftime("%Y-%m-%d")
        if pre_date not in corr_dates:            # 只在计算过量价因子的交易日调仓
            return

        # 拿取调仓日前一个交易日的量价因子，并按照相应十分位选择股票
        pre_corr = corr_data.ix[pre_date]
        pre_corr = pre_corr.dropna()
        pre_corr = pre_corr[(pre_corr<=1.0) & (pre_corr>=-1.0)]

        pre_corr_min = pre_corr.quantile((quantile_five-1)*0.2)
        pre_corr_max = pre_corr.quantile(quantile_five*0.2)
        my_univ = pre_corr[pre_corr>=pre_corr_min][pre_corr<pre_corr_max].index.values

        # 调仓逻辑
        univ = [x for x in my_univ if x in account.universe]

        # 不在股票池中的，清仓
        for stk in account.valid_secpos:
            if stk not in univ:
                order_to(stk, 0)
        # 在目标股票池中的，等权买入
        for stk in univ:
            order_pct_to(stk, 1.1/len(univ))
    # ---------------策略逻辑部分结束----------------

    # 把回测逻辑封装到 TradingStrategy 中，供 quick_backtest 使用
    strategy = quartz.TradingStrategy(initialize, handle_data)
    # 回测部分
    bt, acct = quartz.quick_backtest(sim_params, strategy, idxmap, data, refresh_rate=refresh_rate, commission=commission)

    # 对于回测的结果，可以通过 perf_parse 函数计算风险指标
    perf = quartz.perf_parse(bt, acct)

    # 保存运行结果
    tmp = {}
    tmp['bt'] = bt
    tmp['annualized_return'] = perf['annualized_return']
    tmp['volatility'] = perf['volatility']
    tmp['max_drawdown'] = perf['max_drawdown']
    tmp['alpha'] = perf['alpha']
    tmp['beta'] = perf['beta']
    tmp['sharpe'] = perf['sharpe']
    tmp['information_ratio'] = perf['information_ratio']
    
    results_corr[quantile_five] = tmp
    print (str(quantile_five))
print ('done')


fig = plt.figure(figsize=(10,8))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.grid()
ax2.grid()

for qt in results_corr:
    bt = results_corr[qt]['bt']

    data = bt[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0   # 总头寸每日回报率
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return                 # 总头寸每日超额回报率
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()                # 总头寸对冲指数后的净值序列
    data['portfolio'] = data.portfolio_return + 1.0     
    data['portfolio'] = data.portfolio.cumprod()          # 总头寸不对冲时的净值序列
    data['benchmark'] = data.benchmark_return + 1.0
    data['benchmark'] = data.benchmark.cumprod()          # benchmark的净值序列
    results_corr[qt]['hedged_max_drawdown'] = max([1 - v/max(1, max(data['excess'][:i+1])) for i,v in enumerate(data['excess'])])  # 对冲后净值最大回撤
    results_corr[qt]['hedged_volatility'] = np.std(data['excess_return'])*np.sqrt(252)
    results_corr[qt]['hedged_annualized_return'] = (data['excess'].values[-1])**(252.0/len(data['excess'])) - 1.0
    # data[['portfolio','benchmark','excess']].plot(figsize=(12,8))
    # ax.plot(data[['portfolio','benchmark','excess']], label=str(qt))
    ax1.plot(data['tradeDate'], data[['portfolio']], label=str(qt))
    ax2.plot(data['tradeDate'], data[['excess']], label=str(qt))
    

ax1.legend(loc=0, fontsize=12)
ax2.legend(loc=0, fontsize=12)
ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲净值", fontproperties=font, fontsize=16)
ax1.set_title(u"量价因子 - 不同五分位数分组选股净值走势", fontproperties=font, fontsize=16)
ax2.set_title(u"量价因子 - 不同五分位数分组选股对冲中证500指数后净值走势", fontproperties=font, fontsize=16)


'''
上面的图片显示“量价因子-不同五分位数分组选股”的净值走势，其中下面一张图片展示出各组头寸对冲完中证500指数后的净值走势，可以看到：

不同的五分位数组对应的净值走势顺序区分度很高！
下面的表格展示出不同分位数组合的各项风险指标，每次调仓均买入量价因子最小的20%股票的策略，即最小分位数的组合(组合1)各项指标表现都非常出色：

'''
# results 转换为 DataFrame
import pandas
results_pd = pandas.DataFrame(results_corr).T.sort_index()

results_pd = results_pd[[u'alpha', u'beta', u'information_ratio', u'sharpe', 
                        u'annualized_return', u'max_drawdown', u'volatility', 
                         u'hedged_annualized_return', u'hedged_max_drawdown', u'hedged_volatility']]

for col in results_pd.columns:
    results_pd[col] = [np.round(x, 3) for x in results_pd[col]]
    
cols = [(u'风险指标', u'Alpha'), (u'风险指标', u'Beta'), (u'风险指标', u'信息比率'), (u'风险指标', u'夏普比率'),
        (u'纯股票多头时', u'年化收益'), (u'纯股票多头时', u'最大回撤'), (u'纯股票多头时', u'收益波动率'), 
        (u'对冲后', u'年化收益'), (u'对冲后', u'最大回撤'), 
        (u'对冲后', u'收益波动率')]
results_pd.columns = pd.MultiIndex.from_tuples(cols)
results_pd.index.name = u'五分位组别'
results_pd
