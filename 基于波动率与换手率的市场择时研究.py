# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:11:35 2020

@author: Asus
"""

'''

导读
A. 研究目的：本文利用优矿提供的个股和指数行情数据，参考华泰证券《波动率与换手率构造牛熊指标》（原作者：林晓明等）中，探究波动率与换手率与市场走势的关系，并构造牛熊指标对指数构建择时策略。

B. 研究结论：

利用波动率和换手率可以对市场状态进行划分为4个状态：波动率换手率同时上行（牛市）、波动率换手率同时下行（震荡市）、波动率上行换手率下行（熊市）、波动率下行换手率上行（上升市）。

利用换手率和波动率构建牛熊指标。牛熊指标和指数走势正相关，且拐点一致。

用牛熊指标和换手率指标对各个指数构建择时策略，与指数本身择时策略进行对比，换手率指标择时策略和牛熊指标择时策略均优于指数本身择时策略。以上证综指择时为例，使用双均线择时策略，换手率指标择时策略最优，年化收益达到14.57%，信息比率达到0.85，胜率为73.33%，盈亏比达到8.22；牛熊指标择时策略次之，年化收益达到12.64%，信息比率达到0.78，胜率为60.00%，盈亏比达到5.14。指数本身择时策略的年化换手率为7.67%，信息比率仅0.45，胜率仅29%。

构造牛熊指标的过程中，发现用换手率进行择时也能取得不错的效果。相比牛熊指标，换手率指标在对上证综指进行择时时的回撤更大，在其它方面表现比牛熊更优，因此引入了波动率的牛熊指标还有很大的提升空间。

C. 文章结构：本文共分为4个部分，具体如下

一、获取基础数据。

二、分析波动率、换手率指标的历史特征，并基于它们对市场状态进行划分。

三、构建牛熊指标，并使用牛熊指标和换手率指标构建上证综指的择时策略。

四、探究牛熊指标和换手率指标在沪深300和中证500上的表现。

D. 时间说明

一、第一部分运行需要6分钟
二、第二部分运行需要1分钟
三、第三部分运行需要1分钟
四、第四部分运行需要1分钟
总耗时9分钟左右 (为了方便修改程序，文中将很多中间数据存储下来以缩短运行时间)
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from CAL.PyCAL import *

data_path = 'bear_bull_timing'
if not os.path.exists(data_path):
    os.makedirs(data_path)
    
def proc_float_scale(df, col_name, format_str):
    """
    格式化输出
    参数：
        df: DataFrame, 需要格式化的数据
        col_name： list, 需要格式化的列名
        format_str： 格式类型, 如".2%",".2f"
    """
    df = df.copy()
    for col in col_name:
        for index in df.index:
            if not pd.isnull(df.ix[index, col]):
                df.ix[index, col] = format(df.ix[index, col], format_str)
    return df


'''

第一部分：获取基础数据
该部分耗时 6分钟
该部分内容为：

1.1 获取原始行情数据。起始时间为2000-01-01, 结束时间为2020-01-31。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 获取行情数据

获取从2000-01-01至2020-01-31的全市场行情数据；
获取上证综指的行情数据；
计算上证综指的日换手率：成交量/所有成分股的流通股本之和。

'''


start_time = time.time()
print ("该部分进行基础参数设置和数据准备...")

sdate = '20000101'
edate = '20200131'

# 全A投资域
universe_list = DataAPI.EquGet(equTypeCD=u"A,B",listStatusCD=u"L,S,DE",field=u"secID",pandas="1")['secID'].tolist()
universe_list.remove('DY600018.XSHG')

# 日交易日历
cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=sdate, endDate=edate).sort('calendarDate')
cal_dates_df['calendarDate'] = cal_dates_df['calendarDate'].apply(lambda x: x.replace('-', ''))
trade_dates_list = cal_dates_df[cal_dates_df['isOpen']==1]['calendarDate'].values.tolist()

# 个股行情数据
all_dmkt_df = DataAPI.MktEqudGet(secID=universe_list, beginDate=sdate,endDate=edate, field=u"ticker,tradeDate,closePrice,negMarketValue",pandas="1")
all_dmkt_df['float_share'] = all_dmkt_df['negMarketValue'] / all_dmkt_df['closePrice']
all_dmkt_df['tradeDate'] = all_dmkt_df['tradeDate'].apply(lambda x: str(x).replace('-', ''))
all_dmkt_df.to_pickle(os.path.join(data_path, 'all_dmkt_df.pkl'))
print ("个股行情数据：", all_dmkt_df.head().to_html())

# 上证综指行情数据
indexid = '000001.ZICN'
szzz_dmkt_df = DataAPI.MktIdxdGet(indexID=indexid, beginDate=sdate, endDate=edate, exchangeCD=u"XSHE,XSHG", field=u"ticker,tradeDate,CHGPct,closeIndex,turnoverVol",pandas="1")
szzz_dmkt_df['tradeDate'] = szzz_dmkt_df['tradeDate'].apply(lambda x: str(x).replace('-', ''))

# 上证综指月成分股
szzz_cons_df = DataAPI.IdxCloseWeightGet(secID=indexid, beginDate=sdate,endDate=edate,field=u"effDate,consTickerSymbol",pandas="1")
szzz_cons_df['effDate'] = szzz_cons_df['effDate'].apply(lambda x: str(x).replace('-', ''))
szzz_cons_df['cons'] = 1
szzz_cons_df = szzz_cons_df.pivot_table(values='cons', index='effDate', columns='consTickerSymbol').fillna(0)

# 月成分转换为日成分（因为成分变动不大，且后面会与行情数据merge，所以可以直接向后填充，误差可以忽略不计）
szzz_cons_df = szzz_cons_df.loc[trade_dates_list, :].sort_index()
szzz_cons_df = szzz_cons_df.fillna(method='ffill')
szzz_cons_df = szzz_cons_df.fillna(method='bfill') # 因为成分股数据从20110531开始, 所以之前的成分股，就用2011年之前的向前填充

szzz_cons_df = szzz_cons_df.stack()
szzz_cons_df = szzz_cons_df[szzz_cons_df==1].reset_index()
szzz_cons_df = szzz_cons_df[['effDate', 'consTickerSymbol']]

# 计算指数换手率
szzz_cons_df = szzz_cons_df.merge(all_dmkt_df[['ticker', 'tradeDate', 'float_share']], left_on=['effDate', 'consTickerSymbol'], right_on=['tradeDate', 'ticker'], how='left')
szzz_share_df = szzz_cons_df.groupby('effDate')[['float_share']].sum()
szzz_share_df = szzz_share_df.reset_index().rename(columns={'effDate': 'tradeDate'})
szzz_dmkt_df = szzz_dmkt_df.merge(szzz_share_df, on=['tradeDate'], how='left')
szzz_dmkt_df['turnover'] = szzz_dmkt_df['turnoverVol'] / szzz_dmkt_df['float_share']

szzz_dmkt_df.to_pickle(os.path.join(data_path, 'szzz_dmkt_df.pkl'))
print ("上证综指的行情数据：", szzz_dmkt_df.head().to_html())

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))



# 若已经储存过基础数据，可直接读取
sdate = '20000101'
edate = '20200131'

# 日交易日历
cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=sdate, endDate=edate).sort('calendarDate')
cal_dates_df['calendarDate'] = cal_dates_df['calendarDate'].apply(lambda x: x.replace('-', ''))
trade_dates_list = cal_dates_df[cal_dates_df['isOpen']==1]['calendarDate'].values.tolist()

# 个股行情数据
all_dmkt_df = pd.read_pickle((os.path.join(data_path, 'all_dmkt_df.pkl')))
# 指数行情数据
szzz_dmkt_df = pd.read_pickle((os.path.join(data_path, 'szzz_dmkt_df.pkl')))

'''

第二部分：分析波动率、换手率的历史特征
该部分耗时 1分钟
该部分内容为：

2.1 波动率的历史特征
2.2 换手率的历史特征
2.3 利用波动率和换手率对市场状态进行划分
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
2.1 波动率的历史特征

广义上来讲，波动率可以分为实际波动率、历史波动率、隐含波动率和预测波动率等类别。本报告研究历史波动率。
波动率指标计算：基于股票指数的过去历史n个交易日的收益率，计算标准差。

'''

# 计算历史波动率
szzz_vol_df = szzz_dmkt_df.sort_values('tradeDate')
hist_days_list = [60, 120, 200, 250]
for n in hist_days_list:
    szzz_vol_df['%s_std' %n] = szzz_vol_df['CHGPct'].rolling(n).apply(lambda x: x.std())
sub_szzz_vol_df = szzz_vol_df.dropna()
    
# 画图: 不同参数下的历史波动率
fig, ax = plt.subplots(figsize=(15, 5))
sub_szzz_vol_df[['tradeDate']+['%s_std' %n for n in hist_days_list]].set_index('tradeDate').plot(ax=ax)
ax.set_title(u'上证综指不同参数下的历史波动率(日波动率，未年化)', fontproperties=font, fontsize=15);

'''

历史波动率的特点就是波动率变化特征受到历史长度参数n的影响，类似于移动平均的特点。
n越大，波动率就越平滑；n越小，波动率变化速度就越快。

'''

# 画图： 上证综指和250日波动率
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(sub_szzz_vol_df['tradeDate']), sub_szzz_vol_df['closeIndex'], color='r', label=u'收盘价')
ax1.plot(pd.to_datetime(sub_szzz_vol_df['tradeDate']), sub_szzz_vol_df['250_std'], label=u'250日波动率')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'上证综指及其250日波动率', fontproperties=font, fontsize=15);

'''
本文中，我们重点关注长期限的波动率指标——250日波动率，因为我们希望指标能够描述一段时间的市场特征，而且最好是能够持续一段时间、反映市场涨跌结构的特征。
从上图看出，波动率的上升并非单单来源于市场的下跌。当市场出现快速上涨，会是波动率整体走高；上涨之后的下跌将会使波动率进一步上升。也就是说，在一轮完整的上涨下跌里面，波动率是持续走高的。如上图的2007年-2009年和2014年末-2016年。
在波动率下跌的情况下，市场方向多数处于震荡的情况。

'''


# 计算收盘价和波动率的相关系数
sub_szzz_vol_df['ind'] = sub_szzz_vol_df.index
sub_szzz_vol_df['250_corr'] = sub_szzz_vol_df['ind'].rolling(250).apply(lambda x: sub_szzz_vol_df.loc[x, ['closeIndex', '250_std']].corr().values[0,1])

# 画图：上证综指与250日波动率的滚动相关系数
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(sub_szzz_vol_df['tradeDate']), sub_szzz_vol_df['closeIndex'], color='r', label=u'收盘价')
ax1.plot(pd.to_datetime(sub_szzz_vol_df['tradeDate']), sub_szzz_vol_df['250_corr'], label=u'250日波动率与收盘价滚动1年相关性')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'上证综指与250日波动率的滚动相关系数', fontproperties=font, fontsize=15);


'''

2.2 换手率的历史特征

换手率反映一定时间内，股票流动性的强弱和交易活跃度。

'''
szzz_turnover_df = szzz_dmkt_df.sort_values('tradeDate')

# 画图：上证综指换手率
fig, ax = plt.subplots(figsize=(15, 5))
szzz_turnover_df[['tradeDate', 'turnover']].set_index('tradeDate').plot.area(ax=ax)
ax.set_title(u'上证综指日换手率——基于总流通股本', fontproperties=font, fontsize=15);

'''

为了探究换手率的趋势，借鉴移动平均的方法，计算一定时间内换手率的均值
同样地，换手率的趋势受历史时长参数n影响。

'''

# 计算换日均换手率
hist_days_list = [60, 120, 200, 250]
for n in hist_days_list:
    szzz_turnover_df['%s_turnover' %n] = szzz_turnover_df['turnover'].rolling(n).apply(lambda x: x.mean())
sub_szzz_turnover_df = szzz_turnover_df.dropna()
    
# 画图: 不同参数下的日均换手率
fig, ax = plt.subplots(figsize=(15, 5))
sub_szzz_turnover_df[['tradeDate']+['%s_turnover' %n for n in hist_days_list]].set_index('tradeDate').plot(ax=ax)
ax.set_title(u'上证综指不同参数下的日均换手率', fontproperties=font, fontsize=15);

# 画图： 上证综指和250日均换手率
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(sub_szzz_turnover_df['tradeDate']), sub_szzz_turnover_df['closeIndex'], color='r', label=u'收盘价')
ax1.plot(pd.to_datetime(sub_szzz_turnover_df['tradeDate']), sub_szzz_turnover_df['250_turnover'], label=u'250日均换手率')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'上证综指及其250日均换手率', fontproperties=font, fontsize=15);

'''

为了与波动率统一，我们依然观察250日均换手率与上证综指的关系。
从上图看出，换手率指标和指数走势呈现出明显的正相关。
换手率本质上属于量的概念，在市场明显上涨时，量也明显上升，可以直观的理解为价的稳定上涨，需要量的上涨来推动。
因为采用了均线的方法，存在滞后，如2015年的高点，换手率高点滞后于指数的高点。
'''

# 计算收盘价和日均换手率的相关系数
sub_szzz_turnover_df['ind'] = sub_szzz_turnover_df.index
sub_szzz_turnover_df['250_corr'] = sub_szzz_turnover_df['ind'].rolling(250).apply(lambda x: sub_szzz_turnover_df.loc[x, ['closeIndex', '250_turnover']].corr().values[0,1])

# 画图：上证综指与250日均换手率的滚动相关系数
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(sub_szzz_turnover_df['tradeDate']), sub_szzz_turnover_df['closeIndex'], color='r', label=u'收盘价')
ax1.plot(pd.to_datetime(sub_szzz_turnover_df['tradeDate']), sub_szzz_turnover_df['250_corr'], label=u'250日均换手率与收盘价滚动1年相关性')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'上证综指与250日均换手率的滚动相关系数', fontproperties=font, fontsize=15);


'''

计算指数和250日均换手率的滚动相关性，同样可以看出，大部分时间，两者呈现正相关。
在市场明显上涨和下跌时，两者强正相关。
从2001-2020年初，两者的滚动相关性均值为0.53。



2.3 利用波动率和换手率对市场状态进行划分

根据上述分析，可以利用波动率指标和换手率指标将市场分为4个状态：波动率换手率同时上行、波动率换手率同时下行、波动率上行换手率下行、波动率下行换手率上行。
特别说明，此处划分状态时先验划分，不具备预测能力。

'''

def get_turning_point(se, trend_last_thres=100):
    """
    计算时间序列数据的拐点
    参数：
        se: Series, 时间序列
        trend_last_thres: 趋势持续时间的界限
    返回：
        max_point_list: 局部最高点的index
        min_point_list: 局部最低点的index
    """
    max_point_list = []
    min_point_list = []
    if se.loc[0] >= se.loc[trend_last_thres / 2]:
        flag = 'min'
        max_point_list.append(0)
    else:
        flag = 'max'
        min_point_list.append(0)
    i = 0
    while i < len(se):
        if flag == 'min':
            min_point = np.argmin(se.loc[max(max_point_list):(i+trend_last_thres)])
            if min_point < i:
                min_point_list.append(min_point)
                flag = 'max'
            i = i+trend_last_thres-1
        else:
            max_point = np.argmax(se.loc[max(min_point_list):(i+trend_last_thres)])
            if max_point < i:
                max_point_list.append(max_point)
                flag = 'min'
            i = i+trend_last_thres-1
    return max_point_list, min_point_list


szzz_std_to_df = sub_szzz_vol_df[['tradeDate', 'CHGPct', 'closeIndex', '250_std']].merge(sub_szzz_turnover_df[['tradeDate', '250_turnover']], on='tradeDate')

# 计算波动率指标、换手率指标的拐点
std_max_point_list, std_min_point_list = get_turning_point(szzz_std_to_df['250_std'], trend_last_thres=120)
to_max_point_list, to_min_point_list = get_turning_point(szzz_std_to_df['250_turnover'], trend_last_thres=120)

# 按照波动率与换手率对上证综指走势状态的划分
szzz_std_to_df['std_tp'] = np.nan
szzz_std_to_df.loc[std_max_point_list, 'std_tp'] = -1
szzz_std_to_df.loc[std_min_point_list, 'std_tp'] = 1
szzz_std_to_df['std_tp'] = szzz_std_to_df['std_tp'].fillna(method='ffill')
szzz_std_to_df['to_tp'] = np.nan
szzz_std_to_df.loc[to_max_point_list, 'to_tp'] = -1
szzz_std_to_df.loc[to_min_point_list, 'to_tp'] = 1
szzz_std_to_df['to_tp'] = szzz_std_to_df['to_tp'].fillna(method='ffill')
# 状态划分：1——牛市；2——熊市；3——震荡市；4——上升市
szzz_std_to_df['status'] = np.where(szzz_std_to_df['to_tp']>0, np.where(szzz_std_to_df['std_tp']>0, 1, 4), np.where(szzz_std_to_df['std_tp']>0, 2, 3))

# 走势状态标记
szzz_std_to_df['status_id'] = np.nan
for i in szzz_std_to_df.index:
    if i == 0:
        status_id = 1
    elif szzz_std_to_df.loc[i, 'status'] != szzz_std_to_df.loc[i-1, 'status']:
        status_id = status_id+1
    szzz_std_to_df.loc[i, 'status_id'] = status_id
    
    
# 画图： 波动率指标按趋势将时间分段
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(szzz_std_to_df.index, szzz_std_to_df['250_std'], label=u'250日波动率')
ax.fill_between(szzz_std_to_df.index, 0, np.where(szzz_std_to_df['std_tp']==1, 0.03, 0), color='red', alpha=0.3)
ax.fill_between(szzz_std_to_df.index, 0, np.where(szzz_std_to_df['std_tp']==-1, 0.03, 0), color='darkseagreen', alpha=0.3)
ax.legend(loc=0, prop=font)
ax.grid(False)
ax.set_ylim(0, 0.03)
ax.set_xlim(0, len(szzz_std_to_df))
ind = np.linspace(0, len(szzz_std_to_df)-1, 10).astype(int)
ax.set_xticks(ind)
ax.set_xticklabels(szzz_std_to_df.loc[ind, 'tradeDate'])
ax.set_title(u'波动率走势状态的划分', fontproperties=font, fontsize=15);

# 画图： 换手率指标按趋势将时间分段
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(szzz_std_to_df.index, szzz_std_to_df['250_turnover'], label=u'250日均换手率')
ax.fill_between(szzz_std_to_df.index, 0, np.where(szzz_std_to_df['to_tp']==1, 0.05, 0), color='red', alpha=0.3)
ax.fill_between(szzz_std_to_df.index, 0, np.where(szzz_std_to_df['to_tp']==-1, 0.05, 0), color='darkseagreen', alpha=0.3)
ax.legend(loc=0, prop=font)
ax.grid(False)
ax.set_ylim(0, 0.05)
ax.set_xlim(0, len(szzz_std_to_df))
ind = np.linspace(0, len(szzz_std_to_df)-1, 10).astype(int)
ax.set_xticks(ind)
ax.set_xticklabels(szzz_std_to_df.loc[ind, 'tradeDate'])
ax.set_title(u'换手率走势状态的划分', fontproperties=font, fontsize=15);

# 统计
status_dict = {1: u'牛市', 2: u'熊市', 3: u'震荡市', 4: u'上升市'}
status_sdate = szzz_std_to_df.groupby('status_id')['tradeDate'].apply(lambda x: x.iloc[0])
status_edate = szzz_std_to_df.groupby('status_id')['tradeDate'].apply(lambda x: x.iloc[-1])
status = szzz_std_to_df.groupby('status_id')['status'].apply(lambda x: status_dict[x.values[0]])
summary = pd.concat([status_sdate, status_edate, status], axis=1)
summary.columns = ['开始时间', '结束时间', '状态']
print ('上证综指走势划分的起始时间：', summary.to_html())

# 画图：波动率与换手率对上证综指走势状态的划分
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(szzz_std_to_df.index, szzz_std_to_df['closeIndex'], label=u'收盘价', color='k')
ax1.plot(szzz_std_to_df.index, szzz_std_to_df['250_std'], label=u'250日波动率')
ax1.plot(szzz_std_to_df.index, szzz_std_to_df['250_turnover'], label=u'250日均换手率')
ax.fill_between(szzz_std_to_df.index, 1000, np.where(szzz_std_to_df['status']==1, 7000, 1000), color='red', alpha=0.3, label = status_dict[1])
ax.fill_between(szzz_std_to_df.index, 1000, np.where(szzz_std_to_df['status']==2, 7000, 1000), color='darkseagreen', alpha=0.3, label = status_dict[2])
ax.fill_between(szzz_std_to_df.index, 1000, np.where(szzz_std_to_df['status']==3, 7000, 1000), color='slateblue', alpha=0.3, label = status_dict[3])
ax.fill_between(szzz_std_to_df.index, 1000, np.where(szzz_std_to_df['status']==4, 7000, 1000), color='orangered', alpha=0.3, label = status_dict[4])
ax.grid(False)
ax1.grid(False)
ax.set_xlim(0, len(szzz_std_to_df))
ind = np.linspace(0, len(szzz_std_to_df)-1, 10).astype(int)
ax.set_xticks(ind)
ax.set_xticklabels(szzz_std_to_df.loc[ind, 'tradeDate'])
ax.set_title(u'波动率与换手率对上证综指走势状态的划分', fontproperties=font, fontsize=15)

ax.legend(bbox_to_anchor=(0.7, -0.1), prop=font, ncol=5)
ax1.legend(bbox_to_anchor=(1.2, 0.8), prop=font, ncol=1)

'''

以上就是根据基于波动率和换手率指标对上证综指的状态划分。
对上证综指的各段状态进行分析：
2001年7月下旬-2002年6月末（status_id=2），波动率一路走高，换手率快速下降，是典型的熊市特征。同样的时间段还有2004年5月末-2005年5月末（status_id=8），2007年10月中旬-2008年11月中旬（status_id=12），2017年末-2019年2月初（status_id=24）。
2005年9月初-2006年11月初（status_id=10），波动率下行，换手率迅速上升，指数也快速上升，进入了牛市初期的状态。同样的时间段还有2003年12月初-2004年4月中（status_id=6），2009年1月末-2009年9月末（status_id=14），2014年7月末-2014年11月末（status_id=20）.
2006年11月初-2007年10月中旬（status_id=11），换手率继续上行，波动率也开始快速上升，进入典型的牛市特征。同样的时间段还有2014年11月末-2015年11月（status_id=21），2019年2月中-2019年6月末（status_id=25）。实际上，这两段时间的后半段，指数已经开始下跌，市场进入熊市，但是因为换手率指标的滞后性，换手率指标只是减速上升，并未下降。
2009年9月末-2012年11月末（status_id=25），波动率和换手率同时下降，上证综指总体上呈现一个震荡市的特征, 该段时间为震荡下跌。同样的时间段2002年7月-2003年4月末（status_id=3），2013年12月末-2014年7月末（status_id=19），2016年3月末-2017年12月末（status_id=23），这段时间为震荡上升。
也存在一些划分错误的时间段。例如2003年4月-2004年4月中（status_id=4、5、6），换手率上升，因此该段时间被划分为上升市和牛市，但是实际上指数处于一个震荡市；2012年11月末-2013年12月中（status_id=16、17、18），算法将波动率和换手率定义为上升，但是实际上两者基本走平，没有明显的趋势，因此市场也处于一个震荡的状态。
总体上，基于波动率和换手率指标对上证综指的状态划分大概率还是能将市场进行准确的划分。


第三部分：波动率换手率构建的牛熊指标
该部分耗时 1分钟
该部分内容为：

3.1 构建牛熊指标，并分析牛熊指标的历史特征
3.2 利用牛熊指标和换手率指标对上证综指构建择时策略
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

3.1 构建牛熊指标，并分析牛熊指标的历史特征

基于上述分析，为了更直观的理解，本文构建牛熊指标：换手率/波动率。（研报为波动率/换手率）。当牛熊指标上升时，市场往往处于上升状态；当牛熊指标下跌时，市场往往也处于下跌状态。


'''

# 计算牛熊指标
szzz_std_to_df['bbi'] = szzz_std_to_df['250_turnover'] / szzz_std_to_df['250_std']

# 画图： 牛熊指标和波动率、换手率指标
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(szzz_std_to_df['tradeDate']), szzz_std_to_df['bbi'], label=u'牛熊指标', color='r')
ax1.plot(pd.to_datetime(szzz_std_to_df['tradeDate']), szzz_std_to_df['250_std'], label=u'250日波动率')
ax1.plot(pd.to_datetime(szzz_std_to_df['tradeDate']), szzz_std_to_df['250_turnover'], label=u'250日均换手率')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'牛熊指标和波动率、换手率指标', fontproperties=font, fontsize=15);

# 画图： 上证综指和牛熊指标
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(szzz_std_to_df['tradeDate']), szzz_std_to_df['closeIndex'], color='r', label=u'收盘价')
ax1.plot(pd.to_datetime(szzz_std_to_df['tradeDate']), szzz_std_to_df['bbi'], label=u'牛熊指标')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'上证综指和牛熊指标', fontproperties=font, fontsize=15);


'''

从牛熊指标的定义可以得到，当上升市特征（波动率下行、换手率上行）时，牛熊指标表现为上涨；当熊市特征时（波动率上行、换手率下行），牛熊市指标表现为下降。当牛市特征和震荡市特征时，牛熊指标的方向并不确定。
从图中可以看出，2007和2015年两个典型大牛市时，换手率上行，波动率上行，牛熊指标表现为上行。在震荡市特征时（波动率下行、换手率下行），牛熊指标随着换手率和波动率的强弱进行波动，典型的时间段就是2016年3月开始的震荡市。
以上说明，牛市的波动率上行都是换手率上升所带来的结果，市场的波动放大是交易热情高涨所形成的。

'''
# 计算收盘价和牛熊指标的相关系数
szzz_std_to_df['ind'] = szzz_std_to_df.index
szzz_std_to_df['bbi_corr'] = szzz_std_to_df['ind'].rolling(250).apply(lambda x: szzz_std_to_df.loc[x, ['closeIndex', 'bbi']].corr().values[0,1])

# 画图：上证综指与牛熊指标的滚动相关系数
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(szzz_std_to_df['tradeDate']), szzz_std_to_df['closeIndex'], color='r', label=u'收盘价')
ax1.plot(pd.to_datetime(szzz_std_to_df['tradeDate']), szzz_std_to_df['bbi_corr'], label=u'250日均换手率与收盘价滚动1年相关性')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'上证综指与牛熊指标的滚动相关系数', fontproperties=font, fontsize=15);

'''

计算指数和牛熊指标的滚动相关性，同样可以看出，大部分时间，两者呈现正相关。在部分震荡市特征下，两者会跳变到负相关。
从2001-2020年初，两者的滚动相关性均值为0.62。


3.2 利用牛熊指标和换手率指标对上证综指构建择时策略

根据上述分析，牛熊指标和换手率指标和上证综指的滚动相关性均较高，下面利用他们分别对上证综指构建择时策略。
当牛熊指标（换手率指标）上行，则对指数看多；
当牛熊指标（换手率指标）下行，则对指数看空。
对指数本身也构建择时策略，作为比较基准。
常见的择时策略有两种：双均线策略、布林带策略。
双均线策略：计算标的的20日均线和60日均线，若20日均线自下而上穿过60日均线，则对标的看多，若20日均线自上而下穿过60日均线，则对标的看空。
布林带策略：计算标的的20日均线和标的价格过去20日的标准差，以均线加上两个标准差作为布林带上轨，以均线减去两个标准差作为布林带下轨。当标的突破上轨时，对标的看多，当标的突破下轨时，对标的看空。
回测20020101-20200131这段时间内，上证综指的择时策略。

'''

def get_performance_summary(trade_df):
    """
    计算择时策略的绩效表现
    参数：
        trade_df: DataFrame, 策略的持仓状态和日收益率, columns=['hold', 'ret']
    返回：
        策略绩效表现
    """
    init_trade = pd.DataFrame([False, 0], index=['hold', 'ret']).T
    trade_df = init_trade.append(trade_df).reset_index(drop=True)
    trade_df['nav'] = (trade_df['ret']+1).cumprod()
    
    # 年化收益
    annual_ret = trade_df['ret'].mean()*250.0
    # 年化波动
    annual_std = trade_df['ret'].std()*np.sqrt(250.0)
    # 信息比率
    ir = annual_ret / annual_std
    # 最大回撤
    nav_max = trade_df['nav'].cummax()
    max_drawdown = ((nav_max-trade_df['nav']) / nav_max).max()
    
    trade_df['rebalance'] = (trade_df['hold'] - trade_df['hold'].shift(1))
    trade_df['rebalance'] = trade_df['rebalance'].shift(-1)

    rebalance_df = trade_df[trade_df['rebalance']!=0]
    rebalance_df['pnl'] = (rebalance_df['nav'] - rebalance_df['nav'].shift(1)) / rebalance_df['nav'].shift(1)
    pnl_df = rebalance_df[rebalance_df['rebalance']==-1]['pnl']
    
    if len(pnl_df) > 0:
        # 做多胜率
        win_ratio = 1.0*(pnl_df>0).sum()/len(pnl_df) 
        # 盈亏比
        pnl_ratio = pnl_df[pnl_df>0].mean() / np.abs(pnl_df[pnl_df<0].mean())
        # 交易次数
        rebalance_times = len(pnl_df)*2
        # 交易频率
        rebalance_freq = len(trade_df) / rebalance_times
    else:
        win_ratio = np.nan
        pnl_ratio = np.nan
        rebalance_times = np.nan
        rebalance_freq = np.nan
    
    res = pd.DataFrame([annual_ret, annual_std, ir, max_drawdown, win_ratio, pnl_ratio, rebalance_times, rebalance_freq], index=[u'年化收益', u'年化波动', u'信息比率', u'最大回撤', u'做多胜率', u'盈亏比', u'交易次数', u'交易频率(天/次)']).T
    return res


'''

3.2.1 双均线策略

'''

# 双均线择时
short_span = 20
long_span = 60
szzz_ma_df = szzz_std_to_df[['tradeDate', 'CHGPct', 'closeIndex', '250_turnover', 'bbi']]
szzz_ma_df['nxt_date'] = szzz_ma_df['tradeDate'].shift(-1)
szzz_ma_df['nxt_ret'] = szzz_ma_df['CHGPct'].shift(-1)
szzz_ma_df['close_ma%s' % short_span] = szzz_ma_df['closeIndex'].rolling(short_span).mean()
szzz_ma_df['close_ma%s' % long_span] = szzz_ma_df['closeIndex'].rolling(long_span).mean()
szzz_ma_df['bbi_ma%s'% short_span] = szzz_ma_df['bbi'].rolling(short_span).mean()
szzz_ma_df['bbi_ma%s' % long_span] = szzz_ma_df['bbi'].rolling(long_span).mean()
szzz_ma_df['to_ma%s'% short_span] = szzz_ma_df['250_turnover'].rolling(short_span).mean()
szzz_ma_df['to_ma%s' % long_span] = szzz_ma_df['250_turnover'].rolling(long_span).mean()
szzz_ma_df = szzz_ma_df.query("tradeDate>='20020101'")

# 计算策略调仓点
szzz_ma_df['close_ma_balance'] = szzz_ma_df['close_ma%s' % short_span] > szzz_ma_df['close_ma%s' % long_span]
szzz_ma_df['bbi_ma_balance'] = szzz_ma_df['bbi_ma%s' % short_span] > szzz_ma_df['bbi_ma%s' % long_span]
szzz_ma_df['to_ma_balance'] = szzz_ma_df['to_ma%s' % short_span] > szzz_ma_df['to_ma%s' % long_span]
szzz_ma_df = szzz_ma_df.dropna()

# 计算策略日收益
szzz_ma_df['close_ma_ret'] = np.where(szzz_ma_df['close_ma_balance'], szzz_ma_df['nxt_ret'], 0)
szzz_ma_df['bbi_ma_ret'] = np.where(szzz_ma_df['bbi_ma_balance'], szzz_ma_df['nxt_ret'], 0)
szzz_ma_df['to_ma_ret'] = np.where(szzz_ma_df['to_ma_balance'], szzz_ma_df['nxt_ret'], 0)

# 计算策略表现
banchmark = szzz_ma_df[['CHGPct']]
banchmark.columns = ['ret']
banchmark['hold'] = 1
banchmark_pref = get_performance_summary(banchmark)

close_trade = szzz_ma_df[['close_ma_balance', 'close_ma_ret']]
close_trade.columns = ['hold', 'ret']
close_pref = get_performance_summary(close_trade)

bbi_trade = szzz_ma_df[['bbi_ma_balance', 'bbi_ma_ret']]
bbi_trade.columns = ['hold', 'ret']
bbi_pref = get_performance_summary(bbi_trade)

to_trade = szzz_ma_df[['to_ma_balance', 'to_ma_ret']]
to_trade.columns = ['hold', 'ret']
to_pref = get_performance_summary(to_trade)

ma_perf = pd.concat([to_pref, bbi_pref, close_pref, banchmark_pref])
ma_perf.index = [u'换手率指标择时策略', u'牛熊指标择时策略', u'指数本身择时策略', u'指数']
ma_perf = proc_float_scale(ma_perf, [u'年化收益', u'年化波动', u'最大回撤', u'做多胜率'], '.2%')
ma_perf = proc_float_scale(ma_perf, [u'信息比率', u'盈亏比'], '.2f')
ma_perf = proc_float_scale(ma_perf, [u'交易次数', u'交易频率(天/次)'], '.0f')

print (u'上证综指双均线策略绩效表现:', ma_perf.to_html())

# 画图：上证综指双均线策略净值对比
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(pd.to_datetime(szzz_ma_df['nxt_date']), (szzz_ma_df['nxt_ret']+1).cumprod(), label=u'指数净值')
ax.plot(pd.to_datetime(szzz_ma_df['nxt_date']), (szzz_ma_df['close_ma_ret']+1).cumprod(), label=u'指数本身择时净值')
ax.plot(pd.to_datetime(szzz_ma_df['nxt_date']), (szzz_ma_df['bbi_ma_ret']+1).cumprod(), label=u'牛熊指标择时净值')
ax.plot(pd.to_datetime(szzz_ma_df['nxt_date']), (szzz_ma_df['to_ma_ret']+1).cumprod(), label=u'换手率指标择时净值')
ax.legend(loc=0, prop=font)
ax.set_title(u'上证综指双均线策略净值对比', fontproperties=font, fontsize=15);

'''

在上证综指上，三个双均线策略，换手率指标择时策略最优，年化收益达到14.57%，信息比率达到0.85，胜率为73.33%，盈亏比达到8.22；牛熊指标择时策略次之，年化收益达到12.64%，信息比率达到0.78，胜率为60.00%，盈亏比达到5.14。两者的各项指标均超过指数本身择时策略。
换手率指标择时策略和牛熊指标择时策略胜率明显高于指数本身择时策略，且交易次数明显少于指数本身择时策略。
比较换手率指标和牛熊指标的择时策略，换手率指标的择时策略收益和胜率均较优，但最大回撤也较大。说明牛熊指标对波动率的处理存在优化的空间。

3.2.2 布林带策略

'''

# 布林带策略
span = 20
boll_width = 2
szzz_bolling_df = szzz_std_to_df[['tradeDate', 'CHGPct', 'closeIndex', '250_turnover', 'bbi']]
szzz_bolling_df['nxt_date'] = szzz_bolling_df['tradeDate'].shift(-1)
szzz_bolling_df['nxt_ret'] = szzz_bolling_df['CHGPct'].shift(-1)
szzz_bolling_df['close_ma%s' % span] = szzz_bolling_df['closeIndex'].rolling(span).mean()
szzz_bolling_df['close_std%s' % span] = szzz_bolling_df['closeIndex'].rolling(span).std()
szzz_bolling_df['close_bollup'] = szzz_bolling_df['close_ma%s' % span] + boll_width * szzz_bolling_df['close_std%s' % span]
szzz_bolling_df['close_bolldown'] = szzz_bolling_df['close_ma%s' % span] - boll_width * szzz_bolling_df['close_std%s' % span]

szzz_bolling_df['bbi_ma%s'% span] = szzz_bolling_df['bbi'].rolling(span).mean()
szzz_bolling_df['bbi_std%s' % span] = szzz_bolling_df['bbi'].rolling(span).std()
szzz_bolling_df['bbi_bollup'] = szzz_bolling_df['bbi_ma%s' % span] + boll_width * szzz_bolling_df['bbi_std%s' % span]
szzz_bolling_df['bbi_bolldown'] = szzz_bolling_df['bbi_ma%s' % span] - boll_width * szzz_bolling_df['bbi_std%s' % span]

szzz_bolling_df['to_ma%s'% span] = szzz_bolling_df['250_turnover'].rolling(span).mean()
szzz_bolling_df['to_std%s' % span] = szzz_bolling_df['250_turnover'].rolling(span).std()
szzz_bolling_df['to_bollup'] = szzz_bolling_df['to_ma%s' % span] + boll_width * szzz_bolling_df['to_std%s' % span]
szzz_bolling_df['to_bolldown'] = szzz_bolling_df['to_ma%s' % span] - boll_width * szzz_bolling_df['to_std%s' % span]

szzz_bolling_df = szzz_bolling_df.query("tradeDate>='20020101'")

# 计算策略调仓点
szzz_bolling_df['close_bolling_balance'] = np.where(szzz_bolling_df['closeIndex'] > szzz_bolling_df['close_bollup'], 1, np.where(szzz_bolling_df['closeIndex'] < szzz_bolling_df['close_bolldown'], 0, np.nan))
szzz_bolling_df['close_bolling_balance'] = szzz_bolling_df['close_bolling_balance'].fillna(method='ffill').fillna(0)
szzz_bolling_df['bbi_bolling_balance'] = np.where(szzz_bolling_df['bbi'] > szzz_bolling_df['bbi_bollup'], 1, np.where(szzz_bolling_df['bbi'] < szzz_bolling_df['bbi_bolldown'], 0, np.nan))
szzz_bolling_df['bbi_bolling_balance'] = szzz_bolling_df['bbi_bolling_balance'].fillna(method='ffill').fillna(0)
szzz_bolling_df['to_bolling_balance'] = np.where(szzz_bolling_df['250_turnover'] > szzz_bolling_df['to_bollup'], 1, np.where(szzz_bolling_df['250_turnover'] < szzz_bolling_df['to_bolldown'], 0, np.nan))
szzz_bolling_df['to_bolling_balance'] = szzz_bolling_df['to_bolling_balance'].fillna(method='ffill').fillna(0)
szzz_bolling_df = szzz_bolling_df.dropna()

# 计算策略日收益
szzz_bolling_df['close_bolling_ret'] = np.where(szzz_bolling_df['close_bolling_balance'], szzz_bolling_df['nxt_ret'], 0)
szzz_bolling_df['bbi_bolling_ret'] = np.where(szzz_bolling_df['bbi_bolling_balance'], szzz_bolling_df['nxt_ret'], 0)
szzz_bolling_df['to_bolling_ret'] = np.where(szzz_bolling_df['to_bolling_balance'], szzz_bolling_df['nxt_ret'], 0)

# 计算策略表现
banchmark = szzz_bolling_df[['CHGPct']]
banchmark.columns = ['ret']
banchmark['hold'] = 1
banchmark_pref = get_performance_summary(banchmark)

close_trade = szzz_bolling_df[['close_bolling_balance', 'close_bolling_ret']]
close_trade.columns = ['hold', 'ret']
close_pref = get_performance_summary(close_trade)

bbi_trade = szzz_bolling_df[['bbi_bolling_balance', 'bbi_bolling_ret']]
bbi_trade.columns = ['hold', 'ret']
bbi_pref = get_performance_summary(bbi_trade)

to_trade = szzz_bolling_df[['to_bolling_balance', 'to_bolling_ret']]
to_trade.columns = ['hold', 'ret']
to_pref = get_performance_summary(to_trade)

bolling_perf = pd.concat([to_pref, bbi_pref, close_pref, banchmark_pref])
bolling_perf.index = [u'换手率指标择时策略', u'牛熊指标择时策略', u'指数本身择时策略', u'指数']
bolling_perf = proc_float_scale(bolling_perf, [u'年化收益', u'年化波动', u'最大回撤', u'做多胜率'], '.2%')
bolling_perf = proc_float_scale(bolling_perf, [u'信息比率', u'盈亏比'], '.2f')
bolling_perf = proc_float_scale(bolling_perf, [u'交易次数', u'交易频率(天/次)'], '.0f')

print (u'上证综指布林带策略绩效表现:', bolling_perf.to_html())

# 画图：上证综指布林带策略净值对比
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(pd.to_datetime(szzz_bolling_df['nxt_date']), (szzz_bolling_df['nxt_ret']+1).cumprod(), label=u'指数净值')
ax.plot(pd.to_datetime(szzz_bolling_df['nxt_date']), (szzz_bolling_df['close_bolling_ret']+1).cumprod(), label=u'指数本身择时净值')
ax.plot(pd.to_datetime(szzz_bolling_df['nxt_date']), (szzz_bolling_df['bbi_bolling_ret']+1).cumprod(), label=u'牛熊指标择时净值')
ax.plot(pd.to_datetime(szzz_bolling_df['nxt_date']), (szzz_bolling_df['to_bolling_ret']+1).cumprod(), label=u'换手率指标择时净值')
ax.legend(loc=0, prop=font)
ax.set_title(u'上证综指布林带策略净值对比', fontproperties=font, fontsize=15);

'''

使用布林带策略，基本结果和双均线策略相近。较双均线策略，指数本身择时策略有所提升，换手率指标择时策略和牛熊指标择时策略均下降。但是换手率指标择时策略和牛熊指标择时策略仍优于指数本身择时策略。
牛熊指标择时策略的胜率高于换手率指标择时策略，但换手率指标择时策略的盈亏比更高。
整体上，换手率指标择时策略最优。
   调试 运行
文档
 代码  策略  文档
第四部分：牛熊指标在其他指数的择时表现
该部分耗时 1分钟
该部分内容为：

4.1 牛熊指标在沪深300的择时表现
4.2 牛熊指标在中证500的择时表现
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
4.1 牛熊指标在沪深300的择时表现

'''

# 指数行情数据
indexid = '000300.ZICN'
hs300_dmkt_df = DataAPI.MktIdxdGet(indexID=indexid, beginDate=sdate, endDate=edate, exchangeCD=u"XSHE,XSHG", field=u"ticker,tradeDate,CHGPct,closeIndex,turnoverVol",pandas="1")
hs300_dmkt_df['tradeDate'] = hs300_dmkt_df['tradeDate'].apply(lambda x: str(x).replace('-', ''))
hs300_dmkt_df = hs300_dmkt_df.dropna()

# 指数月成分股
hs300_cons_df = DataAPI.IdxCloseWeightGet(secID=indexid, beginDate=sdate,endDate=edate,field=u"effDate,consTickerSymbol",pandas="1")
hs300_cons_df['effDate'] = hs300_cons_df['effDate'].apply(lambda x: str(x).replace('-', ''))
hs300_cons_df['cons'] = 1
hs300_cons_df = hs300_cons_df.pivot_table(values='cons', index='effDate', columns='consTickerSymbol').fillna(0)

# 月成分转换为日成分（因为成分变动不大，且后面会与行情数据merge，所以可以直接向后填充，误差可以忽略不计）
hs300_cons_df = hs300_cons_df.loc[trade_dates_list, :].sort_index()
hs300_cons_df = hs300_cons_df.fillna(method='ffill').dropna()
hs300_cons_df = hs300_cons_df.stack()
hs300_cons_df = hs300_cons_df[hs300_cons_df==1].reset_index()
hs300_cons_df = hs300_cons_df[['effDate', 'consTickerSymbol']]

# 计算指数换手率
hs300_cons_df = hs300_cons_df.merge(all_dmkt_df[['ticker', 'tradeDate', 'float_share']], left_on=['effDate', 'consTickerSymbol'], right_on=['tradeDate', 'ticker'], how='left')
hs300_share_df = hs300_cons_df.groupby('effDate')[['float_share']].sum()
hs300_share_df = hs300_share_df.reset_index().rename(columns={'effDate': 'tradeDate'})
hs300_dmkt_df = hs300_dmkt_df.merge(hs300_share_df, on=['tradeDate'])
hs300_dmkt_df['turnover'] = hs300_dmkt_df['turnoverVol'] / hs300_dmkt_df['float_share']

n = 250
# 计算牛熊指标
hs300_dmkt_df['250_std'] = hs300_dmkt_df['CHGPct'].rolling(n).std()
hs300_dmkt_df['250_turnover'] = hs300_dmkt_df['turnover'].rolling(n).mean()
hs300_dmkt_df['bbi'] = hs300_dmkt_df['250_turnover'] / hs300_dmkt_df['250_std']

hs300_dmkt_df = hs300_dmkt_df.dropna()

# 画图： 牛熊指标和波动率、换手率指标
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(hs300_dmkt_df['tradeDate']), hs300_dmkt_df['bbi'], label=u'牛熊指标', color='r')
ax1.plot(pd.to_datetime(hs300_dmkt_df['tradeDate']), hs300_dmkt_df['250_std'], label=u'250日波动率')
ax1.plot(pd.to_datetime(hs300_dmkt_df['tradeDate']), hs300_dmkt_df['250_turnover'], label=u'250日均换手率')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'HS300: 牛熊指标和波动率、换手率指标', fontproperties=font, fontsize=15);

# 画图： 沪深300和牛熊指标
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(hs300_dmkt_df['tradeDate']), hs300_dmkt_df['closeIndex'], color='r', label=u'收盘价')
ax1.plot(pd.to_datetime(hs300_dmkt_df['tradeDate']), hs300_dmkt_df['bbi'], label=u'牛熊指标')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'沪深300和牛熊指标', fontproperties=font, fontsize=15);

'''

4.1.1 双均线策略

'''

# 双均线择时
short_span = 20
long_span = 60
hs300_ma_df = hs300_dmkt_df[['tradeDate', 'CHGPct', 'closeIndex', '250_turnover', 'bbi']]
hs300_ma_df['nxt_date'] = hs300_ma_df['tradeDate'].shift(-1)
hs300_ma_df['nxt_ret'] = hs300_ma_df['CHGPct'].shift(-1)

hs300_ma_df['close_ma%s' % short_span] = hs300_ma_df['closeIndex'].rolling(short_span).mean()
hs300_ma_df['close_ma%s' % long_span] = hs300_ma_df['closeIndex'].rolling(long_span).mean()
hs300_ma_df['bbi_ma%s'% short_span] = hs300_ma_df['bbi'].rolling(short_span).mean()
hs300_ma_df['bbi_ma%s' % long_span] = hs300_ma_df['bbi'].rolling(long_span).mean()
hs300_ma_df['to_ma%s'% short_span] = hs300_ma_df['250_turnover'].rolling(short_span).mean()
hs300_ma_df['to_ma%s' % long_span] = hs300_ma_df['250_turnover'].rolling(long_span).mean()

# 计算策略调仓点
hs300_ma_df['close_ma_balance'] = hs300_ma_df['close_ma%s' % short_span] > hs300_ma_df['close_ma%s' % long_span]
hs300_ma_df['bbi_ma_balance'] = hs300_ma_df['bbi_ma%s' % short_span] > hs300_ma_df['bbi_ma%s' % long_span]
hs300_ma_df['to_ma_balance'] = hs300_ma_df['to_ma%s' % short_span] > hs300_ma_df['to_ma%s' % long_span]
hs300_ma_df = hs300_ma_df.dropna()

# 计算策略日收益
hs300_ma_df['close_ma_ret'] = np.where(hs300_ma_df['close_ma_balance'], hs300_ma_df['nxt_ret'], 0)
hs300_ma_df['bbi_ma_ret'] = np.where(hs300_ma_df['bbi_ma_balance'], hs300_ma_df['nxt_ret'], 0)
hs300_ma_df['to_ma_ret'] = np.where(hs300_ma_df['to_ma_balance'], hs300_ma_df['nxt_ret'], 0)

# 计算策略表现
banchmark = hs300_ma_df[['CHGPct']]
banchmark.columns = ['ret']
banchmark['hold'] = 1
banchmark_pref = get_performance_summary(banchmark)

close_trade = hs300_ma_df[['close_ma_balance', 'close_ma_ret']]
close_trade.columns = ['hold', 'ret']
close_pref = get_performance_summary(close_trade)

bbi_trade = hs300_ma_df[['bbi_ma_balance', 'bbi_ma_ret']]
bbi_trade.columns = ['hold', 'ret']
bbi_pref = get_performance_summary(bbi_trade)

to_trade = hs300_ma_df[['to_ma_balance', 'to_ma_ret']]
to_trade.columns = ['hold', 'ret']
to_pref = get_performance_summary(to_trade)

ma_perf = pd.concat([to_pref, bbi_pref, close_pref, banchmark_pref])
ma_perf.index = [u'换手率指标择时策略', u'牛熊指标择时策略', u'指数本身择时策略', u'指数']
ma_perf = proc_float_scale(ma_perf, [u'年化收益', u'年化波动', u'最大回撤', u'做多胜率'], '.2%')
ma_perf = proc_float_scale(ma_perf, [u'信息比率', u'盈亏比'], '.2f')
ma_perf = proc_float_scale(ma_perf, [u'交易次数', u'交易频率(天/次)'], '.0f')

print (u'沪深300双均线策略绩效表现:', ma_perf.to_html())

# 画图：沪深300双均线策略净值对比
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(pd.to_datetime(hs300_ma_df['nxt_date']), (hs300_ma_df['nxt_ret']+1).cumprod(), label=u'指数净值')
ax.plot(pd.to_datetime(hs300_ma_df['nxt_date']), (hs300_ma_df['close_ma_ret']+1).cumprod(), label=u'指数本身择时净值')
ax.plot(pd.to_datetime(hs300_ma_df['nxt_date']), (hs300_ma_df['bbi_ma_ret']+1).cumprod(), label=u'牛熊指标择时净值')
ax.plot(pd.to_datetime(hs300_ma_df['nxt_date']), (hs300_ma_df['to_ma_ret']+1).cumprod(), label=u'换手率择时净值')
ax.legend(loc=0, prop=font)
ax.set_title(u'沪深300双均线策略净值对比', fontproperties=font, fontsize=15);


'''

4.1.2 布林带策略

'''

# 布林带策略
span = 20
boll_width = 2
hs300_bolling_df = hs300_dmkt_df[['tradeDate', 'CHGPct', 'closeIndex', '250_turnover', 'bbi']].dropna()
hs300_bolling_df['nxt_date'] = hs300_bolling_df['tradeDate'].shift(-1)
hs300_bolling_df['nxt_ret'] = hs300_bolling_df['CHGPct'].shift(-1)
hs300_bolling_df['close_ma%s' % span] = hs300_bolling_df['closeIndex'].rolling(span).mean()
hs300_bolling_df['close_std%s' % span] = hs300_bolling_df['closeIndex'].rolling(span).std()
hs300_bolling_df['close_bollup'] = hs300_bolling_df['close_ma%s' % span] + boll_width * hs300_bolling_df['close_std%s' % span]
hs300_bolling_df['close_bolldown'] = hs300_bolling_df['close_ma%s' % span] - boll_width * hs300_bolling_df['close_std%s' % span]

hs300_bolling_df['bbi_ma%s'% span] = hs300_bolling_df['bbi'].rolling(span).mean()
hs300_bolling_df['bbi_std%s' % span] = hs300_bolling_df['bbi'].rolling(span).std()
hs300_bolling_df['bbi_bollup'] = hs300_bolling_df['bbi_ma%s' % span] + boll_width * hs300_bolling_df['bbi_std%s' % span]
hs300_bolling_df['bbi_bolldown'] = hs300_bolling_df['bbi_ma%s' % span] - boll_width * hs300_bolling_df['bbi_std%s' % span]

hs300_bolling_df['to_ma%s'% span] = hs300_bolling_df['250_turnover'].rolling(span).mean()
hs300_bolling_df['to_std%s' % span] = hs300_bolling_df['250_turnover'].rolling(span).std()
hs300_bolling_df['to_bollup'] = hs300_bolling_df['to_ma%s' % span] + boll_width * hs300_bolling_df['to_std%s' % span]
hs300_bolling_df['to_bolldown'] = hs300_bolling_df['to_ma%s' % span] - boll_width * hs300_bolling_df['to_std%s' % span]

# 计算策略调仓点
hs300_bolling_df['close_bolling_balance'] = np.where(hs300_bolling_df['closeIndex'] > hs300_bolling_df['close_bollup'], 1, np.where(hs300_bolling_df['closeIndex'] < hs300_bolling_df['close_bolldown'], 0, np.nan))
hs300_bolling_df['close_bolling_balance'] = hs300_bolling_df['close_bolling_balance'].fillna(method='ffill').fillna(0)
hs300_bolling_df['bbi_bolling_balance'] = np.where(hs300_bolling_df['bbi'] > hs300_bolling_df['bbi_bollup'], 1, np.where(hs300_bolling_df['bbi'] < hs300_bolling_df['bbi_bolldown'], 0, np.nan))
hs300_bolling_df['bbi_bolling_balance'] = hs300_bolling_df['bbi_bolling_balance'].fillna(method='ffill').fillna(0)
hs300_bolling_df['to_bolling_balance'] = np.where(hs300_bolling_df['250_turnover'] > hs300_bolling_df['to_bollup'], 1, np.where(hs300_bolling_df['250_turnover'] < hs300_bolling_df['to_bolldown'], 0, np.nan))
hs300_bolling_df['to_bolling_balance'] = hs300_bolling_df['to_bolling_balance'].fillna(method='ffill').fillna(0)
hs300_bolling_df = hs300_bolling_df.dropna()

# 计算策略日收益
hs300_bolling_df['close_bolling_ret'] = np.where(hs300_bolling_df['close_bolling_balance'], hs300_bolling_df['nxt_ret'], 0)
hs300_bolling_df['bbi_bolling_ret'] = np.where(hs300_bolling_df['bbi_bolling_balance'], hs300_bolling_df['nxt_ret'], 0)
hs300_bolling_df['to_bolling_ret'] = np.where(hs300_bolling_df['to_bolling_balance'], hs300_bolling_df['nxt_ret'], 0)


# 计算策略表现
banchmark = hs300_bolling_df[['CHGPct']]
banchmark.columns = ['ret']
banchmark['hold'] = 1
banchmark_pref = get_performance_summary(banchmark)

close_trade = hs300_bolling_df[['close_bolling_balance', 'close_bolling_ret']]
close_trade.columns = ['hold', 'ret']
close_pref = get_performance_summary(close_trade)

bbi_trade = hs300_bolling_df[['bbi_bolling_balance', 'bbi_bolling_ret']]
bbi_trade.columns = ['hold', 'ret']
bbi_pref = get_performance_summary(bbi_trade)

to_trade = hs300_bolling_df[['to_bolling_balance', 'to_bolling_ret']]
to_trade.columns = ['hold', 'ret']
to_pref = get_performance_summary(to_trade)

bolling_perf = pd.concat([to_pref, bbi_pref, close_pref, banchmark_pref])
bolling_perf.index = [u'换手率指标择时策略', u'牛熊指标择时策略', u'指数本身择时策略', u'指数']
bolling_perf = proc_float_scale(bolling_perf, [u'年化收益', u'年化波动', u'最大回撤', u'做多胜率'], '.2%')
bolling_perf = proc_float_scale(bolling_perf, [u'信息比率', u'盈亏比'], '.2f')
bolling_perf = proc_float_scale(bolling_perf, [u'交易次数', u'交易频率(天/次)'], '.0f')

print (u'沪深300布林带策略绩效表现:', bolling_perf.to_html())

# 画图：沪深300布林带策略净值对比
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(pd.to_datetime(hs300_bolling_df['nxt_date']), (hs300_bolling_df['nxt_ret']+1).cumprod(), label=u'指数净值')
ax.plot(pd.to_datetime(hs300_bolling_df['nxt_date']), (hs300_bolling_df['close_bolling_ret']+1).cumprod(), label=u'指数本身择时净值')
ax.plot(pd.to_datetime(hs300_bolling_df['nxt_date']), (hs300_bolling_df['bbi_bolling_ret']+1).cumprod(), label=u'牛熊指标择时净值')
ax.plot(pd.to_datetime(hs300_bolling_df['nxt_date']), (hs300_bolling_df['to_bolling_ret']+1).cumprod(), label=u'换手率指标择时净值')
ax.legend(loc=0, prop=font)
ax.set_title(u'沪深300布林带策略净值对比', fontproperties=font, fontsize=15);


'''

在沪深300中，双均线策略下，换手率指标择时策略效果最好，牛熊指标择时策略和指数本身择时策略相近。布林带策略下，三者均相近，时间序列上，还是换手率指标择时策略占优。
   调试 运行
文档
 代码  策略  文档
4.2 牛熊指标在中证500的择时表现

'''

# 指数行情数据
indexid = '000905.ZICN'
zz500_dmkt_df = DataAPI.MktIdxdGet(indexID=indexid, beginDate=sdate, endDate=edate, exchangeCD=u"XSHE,XSHG", field=u"ticker,tradeDate,CHGPct,closeIndex,turnoverVol",pandas="1")
zz500_dmkt_df['tradeDate'] = zz500_dmkt_df['tradeDate'].apply(lambda x: str(x).replace('-', ''))
zz500_dmkt_df = zz500_dmkt_df.dropna()

# 指数月成分股
zz500_cons_df = DataAPI.IdxCloseWeightGet(secID=indexid, beginDate=sdate,endDate=edate,field=u"effDate,consTickerSymbol",pandas="1")
zz500_cons_df['effDate'] = zz500_cons_df['effDate'].apply(lambda x: str(x).replace('-', ''))
zz500_cons_df['cons'] = 1
zz500_cons_df = zz500_cons_df.pivot_table(values='cons', index='effDate', columns='consTickerSymbol').fillna(0)

# 月成分转换为日成分（因为成分变动不大，且后面会与行情数据merge，所以可以直接向后填充，误差可以忽略不计）
zz500_cons_df = zz500_cons_df.loc[trade_dates_list, :].sort_index()
zz500_cons_df = zz500_cons_df.fillna(method='ffill').dropna()
zz500_cons_df = zz500_cons_df.stack()
zz500_cons_df = zz500_cons_df[zz500_cons_df==1].reset_index()
zz500_cons_df = zz500_cons_df[['effDate', 'consTickerSymbol']]

# 计算指数换手率
zz500_cons_df = zz500_cons_df.merge(all_dmkt_df[['ticker', 'tradeDate', 'float_share']], left_on=['effDate', 'consTickerSymbol'], right_on=['tradeDate', 'ticker'], how='left')
zz500_share_df = zz500_cons_df.groupby('effDate')[['float_share']].sum()
zz500_share_df = zz500_share_df.reset_index().rename(columns={'effDate': 'tradeDate'})
zz500_dmkt_df = zz500_dmkt_df.merge(zz500_share_df, on=['tradeDate'])
zz500_dmkt_df['turnover'] = zz500_dmkt_df['turnoverVol'] / zz500_dmkt_df['float_share']

n = 250
# 计算牛熊指标
zz500_dmkt_df['250_std'] = zz500_dmkt_df['CHGPct'].rolling(n).std()
zz500_dmkt_df['250_turnover'] = zz500_dmkt_df['turnover'].rolling(n).mean()
zz500_dmkt_df['bbi'] = zz500_dmkt_df['250_turnover'] / zz500_dmkt_df['250_std']

zz500_dmkt_df = zz500_dmkt_df.dropna()

# 画图： 牛熊指标和波动率、换手率指标
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(zz500_dmkt_df['tradeDate']), zz500_dmkt_df['bbi'], label=u'牛熊指标', color='r')
ax1.plot(pd.to_datetime(zz500_dmkt_df['tradeDate']), zz500_dmkt_df['250_std'], label=u'250日波动率')
ax1.plot(pd.to_datetime(zz500_dmkt_df['tradeDate']), zz500_dmkt_df['250_turnover'], label=u'250日均换手率')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'ZZ500: 牛熊指标和波动率、换手率指标', fontproperties=font, fontsize=15);

# 画图： 中证500和牛熊指标
fig, ax = plt.subplots(figsize=(15, 5))
ax1 = ax.twinx()
ax.plot(pd.to_datetime(zz500_dmkt_df['tradeDate']), zz500_dmkt_df['closeIndex'], color='r', label=u'收盘价')
ax1.plot(pd.to_datetime(zz500_dmkt_df['tradeDate']), zz500_dmkt_df['bbi'], label=u'牛熊指标')
ax1.grid(False)
ax.legend(loc=2, prop=font)
ax1.legend(loc=1, prop=font)
ax.set_title(u'中证500和牛熊指标', fontproperties=font, fontsize=15);

'''

4.2.1 双均线策略

'''

# 双均线择时
short_span = 20
long_span = 60
zz500_ma_df = zz500_dmkt_df[['tradeDate', 'CHGPct', 'closeIndex', '250_turnover', 'bbi']]
zz500_ma_df['nxt_date'] = zz500_ma_df['tradeDate'].shift(-1)
zz500_ma_df['nxt_ret'] = zz500_ma_df['CHGPct'].shift(-1)

zz500_ma_df['close_ma%s' % short_span] = zz500_ma_df['closeIndex'].rolling(short_span).mean()
zz500_ma_df['close_ma%s' % long_span] = zz500_ma_df['closeIndex'].rolling(long_span).mean()
zz500_ma_df['bbi_ma%s'% short_span] = zz500_ma_df['bbi'].rolling(short_span).mean()
zz500_ma_df['bbi_ma%s' % long_span] = zz500_ma_df['bbi'].rolling(long_span).mean()
zz500_ma_df['to_ma%s'% short_span] = zz500_ma_df['250_turnover'].rolling(short_span).mean()
zz500_ma_df['to_ma%s' % long_span] = zz500_ma_df['250_turnover'].rolling(long_span).mean()

# 计算策略调仓点
zz500_ma_df['close_ma_balance'] = zz500_ma_df['close_ma%s' % short_span] > zz500_ma_df['close_ma%s' % long_span]
zz500_ma_df['bbi_ma_balance'] = zz500_ma_df['bbi_ma%s' % short_span] > zz500_ma_df['bbi_ma%s' % long_span]
zz500_ma_df['to_ma_balance'] = zz500_ma_df['to_ma%s' % short_span] > zz500_ma_df['to_ma%s' % long_span]
zz500_ma_df = zz500_ma_df.dropna()

# 计算策略日收益
zz500_ma_df['close_ma_ret'] = np.where(zz500_ma_df['close_ma_balance'], zz500_ma_df['nxt_ret'], 0)
zz500_ma_df['bbi_ma_ret'] = np.where(zz500_ma_df['bbi_ma_balance'], zz500_ma_df['nxt_ret'], 0)
zz500_ma_df['to_ma_ret'] = np.where(zz500_ma_df['to_ma_balance'], zz500_ma_df['nxt_ret'], 0)


# 计算策略表现
banchmark = zz500_ma_df[['CHGPct']]
banchmark.columns = ['ret']
banchmark['hold'] = 1
banchmark_pref = get_performance_summary(banchmark)

close_trade = zz500_ma_df[['close_ma_balance', 'close_ma_ret']]
close_trade.columns = ['hold', 'ret']
close_pref = get_performance_summary(close_trade)

bbi_trade = zz500_ma_df[['bbi_ma_balance', 'bbi_ma_ret']]
bbi_trade.columns = ['hold', 'ret']
bbi_pref = get_performance_summary(bbi_trade)

to_trade = zz500_ma_df[['to_ma_balance', 'to_ma_ret']]
to_trade.columns = ['hold', 'ret']
to_pref = get_performance_summary(to_trade)

ma_perf = pd.concat([to_pref, bbi_pref, close_pref, banchmark_pref])
ma_perf.index = [u'换手率指标择时策略', u'牛熊指标择时策略', u'指数本身择时策略', u'指数']
ma_perf = proc_float_scale(ma_perf, [u'年化收益', u'年化波动', u'最大回撤', u'做多胜率'], '.2%')
ma_perf = proc_float_scale(ma_perf, [u'信息比率', u'盈亏比'], '.2f')
ma_perf = proc_float_scale(ma_perf, [u'交易次数', u'交易频率(天/次)'], '.0f')

print (u'中证500双均线策略绩效表现:', ma_perf.to_html())

# 画图：中证500双均线策略净值对比
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(pd.to_datetime(zz500_ma_df['nxt_date']), (zz500_ma_df['nxt_ret']+1).cumprod(), label=u'指数净值')
ax.plot(pd.to_datetime(zz500_ma_df['nxt_date']), (zz500_ma_df['close_ma_ret']+1).cumprod(), label=u'指数本身择时净值')
ax.plot(pd.to_datetime(zz500_ma_df['nxt_date']), (zz500_ma_df['bbi_ma_ret']+1).cumprod(), label=u'牛熊指标择时净值')
ax.plot(pd.to_datetime(zz500_ma_df['nxt_date']), (zz500_ma_df['to_ma_ret']+1).cumprod(), label=u'换手率择时净值')
ax.legend(loc=0, prop=font)
ax.set_title(u'中证500双均线策略净值对比', fontproperties=font, fontsize=15);

'''

4.2.2 布林带策略

'''

# 布林带策略
span = 20
boll_width = 2
zz500_bolling_df = zz500_dmkt_df[['tradeDate', 'CHGPct', 'closeIndex', '250_turnover', 'bbi']].dropna()
zz500_bolling_df['nxt_date'] = zz500_bolling_df['tradeDate'].shift(-1)
zz500_bolling_df['nxt_ret'] = zz500_bolling_df['CHGPct'].shift(-1)
zz500_bolling_df['close_ma%s' % span] = zz500_bolling_df['closeIndex'].rolling(span).mean()
zz500_bolling_df['close_std%s' % span] = zz500_bolling_df['closeIndex'].rolling(span).std()
zz500_bolling_df['close_bollup'] = zz500_bolling_df['close_ma%s' % span] + boll_width * zz500_bolling_df['close_std%s' % span]
zz500_bolling_df['close_bolldown'] = zz500_bolling_df['close_ma%s' % span] - boll_width * zz500_bolling_df['close_std%s' % span]

zz500_bolling_df['bbi_ma%s'% span] = zz500_bolling_df['bbi'].rolling(span).mean()
zz500_bolling_df['bbi_std%s' % span] = zz500_bolling_df['bbi'].rolling(span).std()
zz500_bolling_df['bbi_bollup'] = zz500_bolling_df['bbi_ma%s' % span] + boll_width * zz500_bolling_df['bbi_std%s' % span]
zz500_bolling_df['bbi_bolldown'] = zz500_bolling_df['bbi_ma%s' % span] - boll_width * zz500_bolling_df['bbi_std%s' % span]

zz500_bolling_df['to_ma%s'% span] = zz500_bolling_df['250_turnover'].rolling(span).mean()
zz500_bolling_df['to_std%s' % span] = zz500_bolling_df['250_turnover'].rolling(span).std()
zz500_bolling_df['to_bollup'] = zz500_bolling_df['to_ma%s' % span] + boll_width * zz500_bolling_df['to_std%s' % span]
zz500_bolling_df['to_bolldown'] = zz500_bolling_df['to_ma%s' % span] - boll_width * zz500_bolling_df['to_std%s' % span]

# 计算策略调仓点
zz500_bolling_df['close_bolling_balance'] = np.where(zz500_bolling_df['closeIndex'] > zz500_bolling_df['close_bollup'], 1, np.where(zz500_bolling_df['closeIndex'] < zz500_bolling_df['close_bolldown'], 0, np.nan))
zz500_bolling_df['close_bolling_balance'] = zz500_bolling_df['close_bolling_balance'].fillna(method='ffill').fillna(0)
zz500_bolling_df['bbi_bolling_balance'] = np.where(zz500_bolling_df['bbi'] > zz500_bolling_df['bbi_bollup'], 1, np.where(zz500_bolling_df['bbi'] < zz500_bolling_df['bbi_bolldown'], 0, np.nan))
zz500_bolling_df['bbi_bolling_balance'] = zz500_bolling_df['bbi_bolling_balance'].fillna(method='ffill').fillna(0)
zz500_bolling_df['to_bolling_balance'] = np.where(zz500_bolling_df['250_turnover'] > zz500_bolling_df['to_bollup'], 1, np.where(zz500_bolling_df['250_turnover'] < zz500_bolling_df['to_bolldown'], 0, np.nan))
zz500_bolling_df['to_bolling_balance'] = zz500_bolling_df['to_bolling_balance'].fillna(method='ffill').fillna(0)
zz500_bolling_df = zz500_bolling_df.dropna()

# 计算策略日收益
zz500_bolling_df['close_bolling_ret'] = np.where(zz500_bolling_df['close_bolling_balance'], zz500_bolling_df['nxt_ret'], 0)
zz500_bolling_df['bbi_bolling_ret'] = np.where(zz500_bolling_df['bbi_bolling_balance'], zz500_bolling_df['nxt_ret'], 0)
zz500_bolling_df['to_bolling_ret'] = np.where(zz500_bolling_df['to_bolling_balance'], zz500_bolling_df['nxt_ret'], 0)


# 计算策略表现
banchmark = zz500_bolling_df[['CHGPct']]
banchmark.columns = ['ret']
banchmark['hold'] = 1
banchmark_pref = get_performance_summary(banchmark)

close_trade = zz500_bolling_df[['close_bolling_balance', 'close_bolling_ret']]
close_trade.columns = ['hold', 'ret']
close_pref = get_performance_summary(close_trade)

bbi_trade = zz500_bolling_df[['bbi_bolling_balance', 'bbi_bolling_ret']]
bbi_trade.columns = ['hold', 'ret']
bbi_pref = get_performance_summary(bbi_trade)

to_trade = zz500_bolling_df[['to_bolling_balance', 'to_bolling_ret']]
to_trade.columns = ['hold', 'ret']
to_pref = get_performance_summary(to_trade)

bolling_perf = pd.concat([to_pref, bbi_pref, close_pref, banchmark_pref])
bolling_perf.index = [u'换手率指标择时策略', u'牛熊指标择时策略', u'指数本身择时策略', u'指数']
bolling_perf = proc_float_scale(bolling_perf, [u'年化收益', u'年化波动', u'最大回撤', u'做多胜率'], '.2%')
bolling_perf = proc_float_scale(bolling_perf, [u'信息比率', u'盈亏比'], '.2f')
bolling_perf = proc_float_scale(bolling_perf, [u'交易次数', u'交易频率(天/次)'], '.0f')

print (u'中证500布林带策略绩效表现:', bolling_perf.to_html())

# 画图：中证500布林带策略净值对比
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(pd.to_datetime(zz500_bolling_df['nxt_date']), (zz500_bolling_df['nxt_ret']+1).cumprod(), label=u'指数净值')
ax.plot(pd.to_datetime(zz500_bolling_df['nxt_date']), (zz500_bolling_df['close_bolling_ret']+1).cumprod(), label=u'指数本身择时净值')
ax.plot(pd.to_datetime(zz500_bolling_df['nxt_date']), (zz500_bolling_df['bbi_bolling_ret']+1).cumprod(), label=u'牛熊指标择时净值')
ax.plot(pd.to_datetime(zz500_bolling_df['nxt_date']), (zz500_bolling_df['to_bolling_ret']+1).cumprod(), label=u'换手率指标择时净值')
ax.legend(loc=0, prop=font)
ax.set_title(u'中证500布林带策略净值对比', fontproperties=font, fontsize=15);

'''
在中证500中，双均线策略和布林带策略差异较大。
在双均线策略中，指数本身择时策略无法跑赢指数本身；换手率指标择时策略和牛熊指标择时策略相近，时间序列上，还是换手率指标择时策略占优。
在布林带策略中，三个策略的效果均略微领先指数。指数本身择时策略的信息比率最高。

'''