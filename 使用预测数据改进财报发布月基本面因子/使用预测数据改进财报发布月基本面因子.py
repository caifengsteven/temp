# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:07:26 2020

@author: Asus
"""
'''
导读
A. 研究目的：本文利用优矿提供的行情、财报、因子、一致预期等数据，参考国盛证券《多因子系列之五：使用预测数据改进财报月基本面因子》（原作者：殷明、刘富兵）中的研究方法，利用不同的预测模型对财务数据进行预测，进而对财报月（4、8、10三个月）的基本面因子加以改进，旨在提升财报月的基本面因子的表现。

B. 研究结论：

基本面因子在非财报月的alpha效应均比财报月显著。整体上，财报月的基本面因子效果不佳。

在前视模型下（100%预测正确下个月的基本面数据），财报发布月（4、8、10月）的因子表现效果确实有大幅提升，但是非财报发布月提升不明显。不同的基本面因子在非财报月的提升是随机的。

使用一致预期数据作为预测数据，对基本面因子进行改进。以市盈率倒数和营业收入增长率两个因子为例，在某些财报月有提升，但是提升效果不太稳定。2010年-2019年累计提升相对收益分别为4.2%、4.8%。

构造线性预测模型，对基本面因子进行改进。以市盈率倒数和营业收入增长率两个因子为例，在财报月普遍有提升。2010年-2019年累计提升相对收益分别为11.8%、5.4%。

C. 文章结构：本文共分为4个部分，具体如下

一、描述财报信息的时间滞后问题，并展示其对基本面因子在财报发布月的表现的影响。

二、使用前视模型，分析使用预测数据对基本面因子的改进效果。

三、使用一致预期数据改进财报月的基本面因子，并分析改进效果。

四、构建线性预测模型，使用线性预测模型预测财报月数据，以此改进财报月的基本面因子，并分析改进效果。

D. 时间说明

一、第一部分运行需要2分钟
二、第二部分运行需要2分钟
三、第三部分运行需要4分钟
四、第三部分运行需要25分钟
总耗时33左右
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
https://uqer.datayes.com/community/share/DO4eNMu4iULhXmlEd4vDTTvpK700/private；密码：0332
请在运行之前，克隆上面的代码，并存成lib（右上角->另存为lib,不要修改名字）

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


第一部分：基本面因子在财报月的滞后问题
该部分耗时 2分钟
该部分内容为：

1.1 获取原始行情数据。起始时间为2010-01-01, 结束时间为2019-04-30。
1.2 统计2010年-2019年4月之间发布的所有财报，距离其月末的平均滞后天数。
1.3 在因子库中挑选出几个基本面因子，验证财报信息的时间滞后使得基本面因子在财报月表现不佳。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 行情数据、基础函数准备

'''


import pandas as pd
import numpy as np
import datetime
import time
#from CAL.PyCAL import *
import matplotlib.pyplot as plt
import lib.quant_util as qutil
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

start_time = time.time()
print ("该部分进行基础参数设置和数据准备...")

# 基础数据
sdate = '20100101'
edate = '20190430'

# 全A投资域
a_universe_list = DataAPI.EquGet(equTypeCD=u"A", field=u"secID",pandas="1")['secID'].tolist()
a_universe_list.remove('DY600018.XSHG')

# 获取月末交易日、日频交易日
cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=sdate, endDate='20190530').sort('calendarDate')
monthly_dates_list = cal_dates_df[cal_dates_df['isMonthEnd']==1]['calendarDate'].values.tolist()
monthly_dates_list = map(lambda x: x.replace('-', ''), monthly_dates_list)
daily_dates_list = cal_dates_df[cal_dates_df['isOpen']==1]['calendarDate'].values.tolist()
daily_dates_list = map(lambda x: x.replace('-', ''), daily_dates_list)

# 获取个股月度收益率
mret_df = DataAPI.MktEqumAdjGet(beginDate=sdate, endDate=edate, secID=a_universe_list, field=u"ticker,endDate,chgPct", pandas="1")
mret_df.rename(columns={'endDate':'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
mret_df['tradeDate'] = mret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
mret_df['nxt1m_ret'] = mret_df.groupby('ticker')['curr_ret'].shift(-1)
print ("个股收益率:", mret_df.head().to_html())

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


'''

1.2 问题一：换仓日滞后时间较长

假设固定月频进行策略调仓，则会统一使用月末计算得到的基本面因子，在下个月初进行换仓。那么，在财报公布月，有些公司可能在月初或月中就发出报告，但是在月末，报告信息才会被使用在因子中进行换仓，这造成了财报信息的时间滞后。
下面，统计了2010年-2019年4月之间发布的所有财报，距离其月末的平均滞后天数：一季报滞后6.75天，中报滞后9.22天，三季报滞后5.21天，年报滞后10.02天。

'''

start_time = time.time()
print ("该部分统计财报的平均滞后天数...")

def caltimedelta(date1, date2):
    '''
    计算日期差：date2-date1
    参数：
         date1, date2: 日期，格式为"%Y%m%d"
    返回：
         日期差（天）
    '''
    date1=time.strptime(date1,"%Y%m%d")
    date2=time.strptime(date2,"%Y%m%d")
    date1=datetime.datetime(date1[0],date1[1],date1[2])
    date2=datetime.datetime(date2[0],date2[1],date2[2])
    return (date2-date1).days

# 获取财报数据
fiscal_df = DataAPI.FdmtBSGet(secID=a_universe_list,beginDate=sdate,publishDateEnd=edate,field=u"ticker,publishDate,endDate,reportType,fiscalPeriod",pandas="1")
fiscal_df = fiscal_df.sort_values(['ticker', 'endDate', 'publishDate'])
fiscal_df = fiscal_df.drop_duplicates(subset=['ticker', 'endDate'], keep='first')
fiscal_df['publishDate'] = fiscal_df['publishDate'].apply(lambda x: x.replace('-', ''))
fiscal_df['rebalance_date'] = fiscal_df['publishDate'].apply(lambda x: [date for date in monthly_dates_list if date >= x][0])

# 计算滞后天数
fiscal_df['lag'] = fiscal_df.apply(lambda x: caltimedelta(x['publishDate'], x['rebalance_date']), axis=1)
q_lag = [fiscal_df[fiscal_df['reportType'] == q ].groupby('endDate')['lag'].mean() for q in ['Q1', 'S1', 'Q3', 'A']]

# 统计各个季报的平均滞后天数
title_list = [u'一季报', u'中报', u'三季报', u'年报']
fig = plt.figure(figsize=(20, 4))
for i in range(len(q_lag)):
    ax = fig.add_subplot(1, 4, i+1)
    ind = np.arange(len(q_lag[i]))
    ax.bar(ind+0.2, q_lag[i], 0.3, color='r')
    ax.set_ylim((0, 12))
    ax.set_xticks(ind+0.35)
    ax.set_xticklabels(q_lag[i].index, fontproperties=font, rotation=90)
    ax.set_title(title_list[i]+u' 平均滞后:%s天' % q_lag[i].mean().round(2), fontproperties=font, fontsize=16) 
    
end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


'''

1.3 问题二：月频调仓下，基本面因子在财报月的表现不佳

下面，需要检验，财报信息的时间滞后是否会对基本面因子的表现产生影响呢？下文对一些基本面因子的表现进行了统计。

在通联优矿因子库中选择了几个基本面因子，列表如下。其中，将PS（市销率）进行了倒数形式的方向调整。

因子名	因子含义	因子名	因子含义
ETOP	市盈率倒数	PS	市销率（进行了倒数处理）
EPS	每股收益	ROE	权益回报率
OperatingRevenueGrowRate	营业收入增长率	EARNMOM	八季度净利润变化趋势
SUE	未预期盈利		
获取因子数据后，对因子进行去极值、标准化、行业市值中性化处理。

'''
start_time = time.time()
print ("该部分获取基本面因子数据...")

fundemantal_factor_list = ['ETOP', 'PS', 'EPS', 'ROE', 'OperatingRevenueGrowRate', 'EARNMOM', 'SUE']
factor_df = qutil.get_data_items(a_universe_list, monthly_dates_list, fundemantal_factor_list)
factor_df = pd.concat(factor_df)
# 调整市销率因子方向
factor_df['PS'] = 1.0 / factor_df['PS']
# 因子处理：去极值，标准化，行业市值中性化
factor_ap_df = qutil.mad_winsorize(factor_df, fundemantal_factor_list)
factor_ap_df[fundemantal_factor_list] = factor_ap_df.groupby('tradeDate').apply(lambda x: (x[fundemantal_factor_list] - x[fundemantal_factor_list].mean()) / x[fundemantal_factor_list].std())
exclude_style = ['BETA', 'MOMENTUM', 'SIZENL', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY']
factor_ap_df = qutil.netralize_dframe(factor_ap_df, fundemantal_factor_list, exclude_style)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

start_time = time.time()
print ("该部分统计基本面因子的分月表现...")

# 计算因子收益率
perf_list = []
for fn in fundemantal_factor_list:
    perf,_ = qutil.easy_backtest(factor_ap_df, mret_df, fn, 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
    perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
    perf.columns = [fn]
    perf_list.append(perf)
perf_df = pd.concat(perf_list, axis=1)

# 分析因子在不同月份的收益率
perf_df['month'] = [date[4:6]for date in perf_df.index]
month_perf = perf_df.groupby('month')[fundemantal_factor_list].mean()

# 画图
fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ax.plot(month_perf.index, month_perf)
ax.set_xlim((0, 13))
ax.legend(month_perf.columns, bbox_to_anchor=(0.9, -0.1), ncol=len(fundemantal_factor_list))
ax.set_title(u'不同月份基本面因子收益率表现', fontproperties=font, fontsize=16)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


'''

从图看出，上述基本面因子在2月、4月、8月、9月均出现回撤。10月份因子因子虽没有出现回撤，但是大部分因子表现一般。
4月是年报、一季报发布月，8月是中报发布月，10月是三季报发布月。在这三个月，基本面因子表现一般。
2月份因子表现不佳，可能是春节效应的影响，不是财报信息滞后的原因，因此暂不考虑。

- 进一步，将月份分为财报月组、非财报月组。计算基本面因子在这两组月份中的平均收益，统计是否有显著差异。

'''

start_time = time.time()
print ("该部分统计财报月和非财报月的收益情况...")

month_perf['flag'] = ['fiscal' if m in ['04', '08', '10'] else 'no fiscal' for m in month_perf.index]
fiscal_perf = month_perf.groupby('flag')[fundemantal_factor_list].mean()

# 画图
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ind = np.arange(len(fundemantal_factor_list))
rect1 = ax.bar(ind+0.3, fiscal_perf.loc['no fiscal', :], width=0.3, color='y')
rect2 = ax.bar(ind+0.6, fiscal_perf.loc['fiscal', :], width=0.3, color='r')
ax.set_xticks(ind+0.6)
ax.set_xticklabels(fiscal_perf.columns)
ax.set_title(u'基本面因子在财报月和非财报月的表现差异', fontproperties=font, fontsize=16)
ax.legend([rect1, rect2], [u'非财报发布月', u'财报发布月'], prop=font, bbox_to_anchor=(0.6, -0.1), ncol=2)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


'''


从上图明显看出，上述基本面因子在非财报月的alpha效应均比财报月显著。整体上，财报月的基本面因子效果不佳。
   调试 运行
文档
 代码  策略  文档
因为财报发布时间和调仓时间的间隔，使得财报信息无法及时通过基本面因子反映到持仓上。那么改进这一现象的思路就有两个：
思路一：提高调仓频率：在财报发布月中，提高调仓频率，使得财报信息能及时反映到持仓上。
提高调仓频率，在不考虑交易摩擦时，确实能提高组合的收益。但是考虑到实际交易，提高调仓频率，必然带来更高的交易成本，当增加的交易成本超过提高收益时，这种方法就失去了操作性。（相关研报可以参考东方证券《因子选股系列研究之五十一：适宜快节奏的年报公告季节》）
思路二： 在财报发布月的前一个月，预测财报发布结果，使用预测数据计算基本面因子，提前将财报信息反映到持仓上。
这一方法，能够解决增加交易成本的问题。下文来验证这种方法能否提升基本面因子的表现。
   调试 运行
文档
 代码  策略  文档
第二部分：使用前视模型分析基本面因子改进效果
该部分耗时 2分钟
首先，我们需要验证，假设我们能够100%预测正确未来的财报数据，能否提高基本面因子的表现。因此，我们使用前视模型：在每个月月末，均能完美预测下一个月发布的财报数据, 使用预测数据计算基本面因子，比较和原始基本面因子的表现情况。下文以ETOP、OperatingRevenueGrowRate两个因子为例说明，旨在说明涉及归母净利润和营业收入财报字段的基本面因子的改进方法和效果。其他基本面因子方法类似。

该部分内容为：

2.1 使用前视模型计算ETOP因子，比较因子分月表现。
2.2 使用前视模型计算OperatingRevenueGrowRate因子，比较因子分月表现。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
2.1 使用前视模型计算ETOP因子

前视模型的具体做法是：例如在3月末，使用4月份的归母净利润TTM数据，除以当前的总市值，得到前视模型下的ETOP因子值。
为了具有可比性，前视模型下的ETOP因子也会进行去极值、标准化、行业市值中性化处理。如无特别说明，下文的提及的因子均会进行相同处理。


'''


def get_foresight_fiscal(fiscal_df, sdate, edate):
    '''
    获取一段时间内的前视模型的财务数据，即在当月底，获取下月份的财务数据
    参数：
        fiscal_df：DataFrame，真实财务数据
        sdate: 起始时间
        edate: 结束时间
    返回：
        前视模型下的财务数据，日期为连续的交易日
    '''
    fiscal_df['publishDate']= fiscal_df['publishDate'].apply(lambda x: x.replace('-', ''))
    fiscal_df['endDate']= fiscal_df['endDate'].apply(lambda x: x.replace('-', ''))
    
    # 去重
    fiscal_df = fiscal_df.sort_values(['ticker', 'publishDate', 'endDate'])
    fiscal_df = fiscal_df.drop_duplicates(subset=['ticker', 'endDate'], keep='first')
    fiscal_df = fiscal_df.drop_duplicates(subset=['ticker', 'publishDate'], keep='last')
    
    # 去掉非最新财报期记录
    fiscal_df['flag'] = fiscal_df.groupby(['ticker'])['endDate'].rolling(window=8, min_periods=1).max().values.astype(int).astype(str)
    fiscal_df = fiscal_df[fiscal_df['endDate'] == fiscal_df['flag']]
    
    cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG").sort('calendarDate')
    cal_dates_df['calendarDate'] = cal_dates_df['calendarDate'].apply(lambda x: x.replace('-', ''))
    monthly_dates_list = cal_dates_df[cal_dates_df['isMonthEnd']==1]['calendarDate'].values.tolist()
    
    # 将每条财务数据的可用时间，提前到发布日的上一个月底
    fiscal_df['pred_publishDate'] = fiscal_df['publishDate'].apply(lambda x: [date for date in monthly_dates_list if date <= x][-1])
    fiscal_df = fiscal_df.rename(columns={'pred_publishDate': 'tradeDate'})

    # PIT转换为连续时间
    trade_date_frame = cal_dates_df[cal_dates_df['isMonthEnd']==1]  
    trade_date_frame = trade_date_frame.rename(columns={"calendarDate": "tradeDate"})
    trade_date_frame = trade_date_frame[(trade_date_frame['tradeDate']>=sdate) & (trade_date_frame['tradeDate']<=edate)]
    trade_date_frame['tradeDate'] = trade_date_frame['tradeDate'].apply(lambda x: x.replace('-', ''))
    factor_df = fiscal_df.groupby(['ticker']).apply(lambda x: x.merge(trade_date_frame[['tradeDate', 'isOpen']], on=['tradeDate'], how='outer'))
    del factor_df['ticker']
    factor_df.reset_index(level='ticker', inplace=True)

    factor_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    factor_df = factor_df.groupby('ticker').apply(lambda x: x.fillna(method='ffill', limit=365))
    factor_df = factor_df[factor_df['isOpen'] == 1]
    
    # 去重
    factor_df = factor_df.sort_values(['ticker', 'tradeDate', 'endDate'])
    factor_df = factor_df.drop_duplicates(subset=['ticker', 'tradeDate'], keep='last')
    return factor_df.dropna()

start_time = time.time()
print ("该部分计算前视模型的ETOP, 并比较分月表现...")

# 计算前视模型的归母净利润
nincome_df = DataAPI.FdmtISTTMPITGet(secID=a_universe_list, publishDateEnd=edate, field=u"ticker,publishDate,endDate,NIncomeAttrP",pandas="1")
foresight_nincome_df = get_foresight_fiscal(nincome_df, sdate, edate)

# 获取总市值数据
mkt_df = qutil.get_data_items(a_universe_list, monthly_dates_list, ['MktValue'])
mkt_df = pd.concat(mkt_df)

# 计算前视模型的ETOP
foresight_ep_df = foresight_nincome_df.merge(mkt_df, on=['ticker', 'tradeDate'], how='left')
foresight_ep_df['foresight_ep'] = foresight_ep_df['NIncomeAttrP'] / foresight_ep_df['MktValue']
foresight_ep_df = foresight_ep_df[['ticker', 'tradeDate', 'foresight_ep']]

# 因子处理：去极值，标准化，行业市值中性化
foresight_ep_ap_df = qutil.mad_winsorize(foresight_ep_df, ['foresight_ep'])
foresight_ep_ap_df['foresight_ep'] = foresight_ep_ap_df.groupby('tradeDate')['foresight_ep'].apply(lambda x: (x - x.mean()) / x.std())
exclude_style = ['BETA', 'MOMENTUM', 'SIZENL', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY']
foresight_ep_ap_df = qutil.netralize_dframe(foresight_ep_ap_df, ['foresight_ep'], exclude_style)

# 比较前视模型的ETOP和原始ETOP的提升
perf_list = []
perf,_ = qutil.easy_backtest(factor_ap_df, mret_df, 'ETOP', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['ETOP']
perf_list.append(perf)
perf,_ = qutil.easy_backtest(foresight_ep_ap_df, mret_df, 'foresight_ep', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['foresight_ep']
perf_list.append(perf)

perf_df = pd.concat(perf_list, axis=1)

# 分析因子在不同月份的收益率
perf_df['month'] = [date[4:6]for date in perf_df.index]
month_perf = perf_df.groupby('month')['ETOP', 'foresight_ep'].mean()
month_perf['diff'] = month_perf['foresight_ep'] - month_perf['ETOP']
# 画图
fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ind = np.arange(12)
ax.bar(ind+0.2, month_perf['diff'], width=0.5)
ax.set_xticks(ind+0.45)
ax.set_xticklabels(month_perf.index)
ax.set_title(u'前视模型下ETOP的收益分月提升表现', fontproperties=font, fontsize=16)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''
对于ETOP因子，使用前视模型，除了9月份，在其他月份均能提高因子收益。
其中，3、4、8、10月份的因子表现提高最为显著，其他月份的提升效果一般。
   调试 运行
文档
 代码  策略  文档
2.2 使用前视模型计算OperatingRevenueGrowRate因子

前视模型的具体做法是：例如在3月末，使用4月份的营业收入TTM数据，结合去年同期的营业收入TTM数据，计算增长率，得到前视模型下的OperatingRevenueGrowRate因子值。

'''

start_time = time.time()
print ("该部分计算前视模型的OperatingRevenueGrowRate, 并比较分月表现...")

# 计算前视模型的营业收入
sales_df = DataAPI.FdmtISTTMPITGet(secID=a_universe_list, publishDateEnd=edate, field=u"ticker,publishDate,endDate,revenue",pandas="1")
foresight_sales_df = get_foresight_fiscal(sales_df, sdate, edate)

# 计算前视模型的OperatingRevenueGrowRate
def get_pre_end(date):
    """
    获取上一个期末日期
    params:
        date: str, %Y%m%d
    return:
        pre_end: str, %Y%m%d
    """
    if date[4:6] == '03':
        pre_end = str(int(date[:4])-1)+'1231'
    elif date[4:6] == '06':
        pre_end = date[:4]+'0331'
    elif date[4:6] == '09':
        pre_end = date[:4]+'0630'
    elif date[4:6] == '12':
        pre_end = date[:4]+'0930'
    return pre_end

foresight_sales_df['pre_end'] = foresight_sales_df['endDate'].apply(get_pre_end)
foresight_org_df = foresight_sales_df.merge(sales_df, left_on=['ticker', 'pre_end'], right_on=['ticker', 'endDate'], how='left', suffixes=['_curr', '_pre'])
foresight_org_df = foresight_org_df[foresight_org_df['tradeDate'] >= foresight_org_df['publishDate_pre']]
foresight_org_df = foresight_org_df.sort_values(['ticker', 'tradeDate', 'publishDate_pre'])
foresight_org_df = foresight_org_df.drop_duplicates(subset=['ticker', 'tradeDate'], keep='last')
foresight_org_df['foresight_org'] = (foresight_org_df['revenue_curr'] - foresight_org_df['revenue_pre']) / abs(foresight_org_df['revenue_pre'])
foresight_org_df = foresight_org_df[['ticker', 'tradeDate', 'foresight_org']]

# 因子处理：去极值，标准化，行业市值中性化
foresight_org_ap_df = qutil.mad_winsorize(foresight_org_df, ['foresight_org'])
foresight_org_ap_df['foresight_org'] = foresight_org_ap_df.groupby('tradeDate')['foresight_org'].apply(lambda x: (x - x.mean()) / x.std())
exclude_style = ['BETA', 'MOMENTUM', 'SIZENL', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY']
foresight_org_ap_df = qutil.netralize_dframe(foresight_org_ap_df, ['foresight_org'], exclude_style)

# 比较前视模型的OperatingRevenueGrowRate和原始OperatingRevenueGrowRate的提升
perf_list = []
perf,_ = qutil.easy_backtest(factor_ap_df, mret_df, 'OperatingRevenueGrowRate', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['OperatingRevenueGrowRate']
perf_list.append(perf)
perf,_ = qutil.easy_backtest(foresight_org_ap_df, mret_df, 'foresight_org', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['foresight_org']
perf_list.append(perf)

perf_df = pd.concat(perf_list, axis=1)

# 分析因子在不同月份的收益率
perf_df['month'] = [date[4:6]for date in perf_df.index]
month_perf = perf_df.groupby('month')['OperatingRevenueGrowRate', 'foresight_org'].mean()
month_perf['diff'] = month_perf['foresight_org'] - month_perf['OperatingRevenueGrowRate']
# 画图
fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ind = np.arange(12)
ax.bar(ind+0.2, month_perf['diff'], width=0.5)
ax.set_xticks(ind+0.45)
ax.set_xticklabels(month_perf.index)
ax.set_title(u'前视模型下OperatingRevenueGrowRate的收益分月提升表现', fontproperties=font, fontsize=16)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

对于ETOP因子，使用前视模型，除了2月份，在其他月份均能提高因子收益。
其中，4、7、8、10月份的因子表现提高最为显著，其他月份的提升效果一般。
   调试 运行
文档
 代码  策略  文档
从上述结果来看，如果能够100%预测正确下个月的基本面数据，财报发布月（4、8、10月）的因子表现效果确实有大幅提升，但是非财报发布月的提升效果不显著。
基于上述结论，我们每年实际上只需要预测三次，在3、7、9月使用预测模型分别预测4、8、10月的财报数据，更新因子截面数据皆可。
图片注释
（截图来自国盛证券研报）

   调试 运行
文档
 代码  策略  文档
第三部分：使用一致预期数据改进财报月的基本面因子
该部分耗时 4分钟
第二部分验证了预测模型能够达到改进基本面因子在财报月的表现。下文解决如何在财报月之前，得到预测数据的问题。

最便捷的得到预测数据的方法，就是熟知的分析师一致预期数据。下面，我们使用一致预期数据作为财报的预测数据，检验一致预期数据对基本面因子在财报月表现的提升效果。

该部分内容为：

3.1 使用一致预期预测模型计算ETOP因子，比较因子分月表现。
3.2 使用一致预期预测模型计算OperatingRevenueGrowRate因子，比较因子分月表现。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 使用一致预期数据计算ETOP因子

具体做法是：在3、7、9月底，使用一致预期数据。例如，2018年3月底，使用当天的2017年的一致预期净利润；2018年7月底和9月底，使用当天的2018年的一致预期净利润。然后除以当前的总市值，得到一致预期预测模型下的ETOP因子值。
在3、7、9月底，已发布财报的个股，就是用正式报告的数据计算因子；没有发布财报的公司，则用一致预期数据计算因子。
特别说明，因为预测类型为3的一致预期数据，等价于历史的TTM值，因此不对一致预期数据的预测类型进行筛选。

'''

def get_con_fiscal(date, con_col):
    '''
    获取财报月的一致预期数据, 已公布财报的，则为正式报告的数据。
    参数：
        date：str，3、7、10月的月底交易日日期，格式为"%Y%m%d"
        con_col: 一致预期数据字段，['profit', 'sales']中的一个，'profit'为一致预期净利润；'sales'为一致预期营业收入
    返回：
        一致预期预测模型下的财务数据
    '''
    # 获取一致预期数据
    if con_col == 'profit':
        con_df = DataAPI.ResConSecDataGet(endDate=date,beginDate=date,field="secCode,ConProfitType,foreYear,conProfit",pandas="1")
        fiscal_col = 'NIncomeAttrP'
    elif con_col == 'sales':
        con_df = DataAPI.ResConSecIncomeGet(endDate=date,beginDate=date,field="secCode,conIncomeType,foreYear,conIncome",pandas="1")
        fiscal_col = 'revenue'
    else:
        con_df = None
    con_df = con_df.sort_values(['secCode', 'foreYear'])
    con_df.columns = ['ticker', 'con_type', 'year', 'con_value']
    con_df['con_value'] = con_df['con_value'] * 10000
    
    # 筛选一致预期数据
    if date[4:6] == '03':
        con_df = con_df[con_df['year'] == int(date[:4])-1]
        ispub_list = [str(int(date[:4])-1)+'12', date[:4]+'03']
        rep_end = str(int(date[:4])-1)+'1231'
    elif date[4:6] == '07':
        con_df = con_df[con_df['year'] == int(date[:4])]
        ispub_list = [date[:4]+'06']
        rep_end = date[:4]+'0630'
    elif date[4:6] == '09':
        con_df = con_df[con_df['year'] == int(date[:4])]
        ispub_list = [date[:4]+'09']
        rep_end = date[:4]+'0930'
        
    # 获取财报数据
    fiscal_df = DataAPI.FdmtISTTMPITGet(secID=a_universe_list, publishDateEnd=date, endDate=get_pre_end(rep_end), field=u"ticker,publishDate,endDate,"+fiscal_col,pandas="1")
    fiscal_df['publishDate']= fiscal_df['publishDate'].apply(lambda x: x.replace('-', ''))
    fiscal_df['endDate']= fiscal_df['endDate'].apply(lambda x: x.replace('-', ''))
    fiscal_df = fiscal_df.sort_values(['ticker', 'endDate', 'publishDate'])
    fiscal_df = fiscal_df.drop_duplicates(['ticker'], keep='last')
    fiscal_df['flag'] = fiscal_df['endDate'].apply(lambda x: x[:6] in ispub_list)
    
    # 没有发布财报的个股，用一致预期数据替代；发布财报的个股，使用正式报告的数据
    merge_df = con_df.merge(fiscal_df, on='ticker', how='outer')
    merge_df['flag'] = np.where(merge_df['con_value'].isnull() | merge_df['con_type'] == 3, True, merge_df['flag'])
    merge_df['value'] = np.where(merge_df['flag'], merge_df[fiscal_col], merge_df['con_value'])
    merge_df['type'] = np.where(merge_df['flag'], 'history_ttm', 'consensus')
    merge_df['endDate'] = np.where(merge_df['flag'], merge_df['endDate'], rep_end)
    merge_df['tradeDate'] = date
    merge_df = merge_df[['ticker', 'tradeDate', 'endDate', 'value', 'type']]
    merge_df.rename(columns={'value': fiscal_col}, inplace=True)
    return merge_df.dropna()

start_time = time.time()
print ("该部分计算一致预期预测模型的ETOP, 并比较分月表现...")

# 计算一致预期预测模型下的归母净利润
fiscal_date_list = [date for date in monthly_dates_list if date[4:6] in ['03', '07', '09']]
con_nincome_df = []
for date in fiscal_date_list:
    con_nincome = get_con_fiscal(date, con_col='profit')
    con_nincome_df.append(con_nincome)
con_nincome_df = pd.concat(con_nincome_df)

#  获取总市值数据
mkt_df = qutil.get_data_items(a_universe_list, monthly_dates_list, ['MktValue'])
mkt_df = pd.concat(mkt_df)

# 计算一致预期预测模型的ETOP
con_ep_df = con_nincome_df.merge(mkt_df, on=['ticker', 'tradeDate'], how='left').dropna()
con_ep_df['con_ep'] = con_ep_df['NIncomeAttrP'] / con_ep_df['MktValue']
con_ep_df = con_ep_df[['ticker', 'tradeDate', 'con_ep']]

# 因子处理：去极值，标准化，行业市值中性化
con_ep_ap_df = qutil.mad_winsorize(con_ep_df, ['con_ep'])
con_ep_ap_df['con_ep'] = con_ep_ap_df.groupby('tradeDate')['con_ep'].apply(lambda x: (x - x.mean()) / x.std())
exclude_style = ['BETA', 'MOMENTUM', 'SIZENL', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY']
con_ep_ap_df = qutil.netralize_dframe(con_ep_ap_df, ['con_ep'], exclude_style)
con_ep_ap_df_all =  factor_ap_df[['ticker', 'tradeDate', 'ETOP']].merge(con_ep_ap_df, on=['ticker', 'tradeDate'], how='left')
con_ep_ap_df_all['con_ep'] = np.where(con_ep_ap_df_all['con_ep'].isnull(), con_ep_ap_df_all['ETOP'], con_ep_ap_df_all['con_ep'])

# 比较一致预期预测模型的ETOP和原始ETOP的提升
perf_list = []
perf,_ = qutil.easy_backtest(factor_ap_df, mret_df, 'ETOP', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['ETOP']
perf_list.append(perf)
perf,_ = qutil.easy_backtest(con_ep_ap_df_all, mret_df, 'con_ep', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['con_ep']
perf_list.append(perf)

perf_df = pd.concat(perf_list, axis=1)

# 分析因子在不同月份的收益率
perf_df['month'] = [date[4:6]for date in perf_df.index]
month_perf = perf_df.groupby('month')['ETOP', 'con_ep'].mean()
month_perf['diff'] = month_perf['con_ep'] - month_perf['ETOP']

# 画图
fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ind = np.arange(12)
ax.bar(ind+0.2, month_perf['diff'], width=0.5)
ax.set_xticks(ind+0.45)
ax.set_xticklabels(month_perf.index)
ax.set_title(u'一致预期预测模型下ETOP的收益分月提升表现', fontproperties=font, fontsize=16)

fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ax1 = ax.twinx()
ax.plot(pd.to_datetime(perf_df.index), (perf_df['ETOP']+1).cumprod(), label=u'原始')
ax.plot(pd.to_datetime(perf_df.index), (perf_df['con_ep']+1).cumprod(), label=u'一致预期预测')
ax1.plot(pd.to_datetime(perf_df.index), (perf_df['con_ep']-perf_df['ETOP']+1).cumprod(), label=u'相对收益(右轴)', color='r')
ax.grid(False)
ax1.grid(False)
ax.legend(bbox_to_anchor=(0.5, -0.1), prop=font, ncol=2)
ax1.legend(bbox_to_anchor=(0.65, -0.1), prop=font)
ax.set_title(u'一致预期预测模型下ETOP的累计收益提升表现', fontproperties=font, fontsize=16)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''
一致预期预测模型计算的ETOP因子，在8月份的改进效果最好，10月份一般，4月份最差。且在2017下半年开始，模型丧失改进效果。
ETOP因子在2010年-2019年的累计提升相对收益为4.2%。
   调试 运行
文档
 代码  策略  文档
3.2 使用一致预期数据计算OperatingRevenueGrowRate因子

具体做法是：在3、7、9月底，使用一致预期数据。例如，2018年3月底，使用当天的2017年的一致预期营业收入；2018年7月底和9月底，使用当天的2018年的一致营业收入。结合去年同期的营业收入TTM数据，计算增长率，得到一致预期预测模型下的OperatingRevenueGrowRate因子值。
在3、7、9月底，已发布财报的个股，就是用正式报告的数据计算因子；没有发布财报的公司，则用一致预期数据计算因子。
特别说明，因为预测类型为3的一致预期数据，等价于历史的TTM值，因此不对一致预期数据的预测类型进行筛选。

'''

start_time = time.time()
print ("该部分计算一致预期预测模型下的OperatingRevenueGrowRate, 并比较分月表现...")

# 计算一致预期预测模型下的营业收入
fiscal_date_list = [date for date in monthly_dates_list if date[4:6] in ['03', '07', '09']]
con_sales_df = []
for date in fiscal_date_list:
    con_sales = get_con_fiscal(date, con_col='sales')
    con_sales_df.append(con_sales)
con_sales_df = pd.concat(con_sales_df)

# 计算一致预期预测模型下的OperatingRevenueGrowRate
sales_df = DataAPI.FdmtISTTMPITGet(secID=a_universe_list, publishDateEnd=edate, field=u"ticker,publishDate,endDate,revenue",pandas="1")
sales_df['publishDate']= sales_df['publishDate'].apply(lambda x: x.replace('-', ''))
sales_df['endDate']= sales_df['endDate'].apply(lambda x: x.replace('-', ''))
con_sales_df['pre_end'] = con_sales_df['endDate'].apply(get_pre_end)
con_org_df = con_sales_df.merge(sales_df, left_on=['ticker', 'pre_end'], right_on=['ticker', 'endDate'], how='left', suffixes=['_curr', '_pre'])
con_org_df = con_org_df[con_org_df['tradeDate'] >= con_org_df['publishDate']]
con_org_df = con_org_df.sort_values(['ticker', 'tradeDate', 'publishDate'])
con_org_df = con_org_df.drop_duplicates(subset=['ticker', 'tradeDate'], keep='last')
con_org_df['con_org'] = (con_org_df['revenue_curr'] - con_org_df['revenue_pre']) / abs(con_org_df['revenue_pre'])
con_org_df = con_org_df[['ticker', 'tradeDate', 'con_org']]

# 因子处理：去极值，标准化，行业市值中性化
con_org_ap_df = qutil.mad_winsorize(con_org_df, ['con_org'])
con_org_ap_df['con_org'] = con_org_ap_df.groupby('tradeDate')['con_org'].apply(lambda x: (x - x.mean()) / x.std())
exclude_style = ['BETA', 'MOMENTUM', 'SIZENL', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY']
con_org_ap_df = qutil.netralize_dframe(con_org_ap_df, ['con_org'], exclude_style)
con_org_ap_df_all =  factor_ap_df[['ticker', 'tradeDate', 'OperatingRevenueGrowRate']].merge(con_org_ap_df, on=['ticker', 'tradeDate'], how='left')
con_org_ap_df_all['con_org'] = np.where(con_org_ap_df_all['con_org'].isnull(), con_org_ap_df_all['OperatingRevenueGrowRate'], con_org_ap_df_all['con_org'])

# 比较一致预期预测模型的OperatingRevenueGrowRate和原始OperatingRevenueGrowRate的提升
perf_list = []
perf,_ = qutil.easy_backtest(factor_ap_df, mret_df, 'OperatingRevenueGrowRate', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['OperatingRevenueGrowRate']
perf_list.append(perf)
perf,_ = qutil.easy_backtest(con_org_ap_df_all, mret_df, 'con_org', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['con_org']
perf_list.append(perf)

perf_df = pd.concat(perf_list, axis=1)

# 分析因子在不同月份的收益率
perf_df['month'] = [date[4:6]for date in perf_df.index]
month_perf = perf_df.groupby('month')['OperatingRevenueGrowRate', 'con_org'].mean()
month_perf['diff'] = month_perf['con_org'] - month_perf['OperatingRevenueGrowRate']

# 画图
fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ind = np.arange(12)
ax.bar(ind+0.2, month_perf['diff'], width=0.5)
ax.set_xticks(ind+0.45)
ax.set_xticklabels(month_perf.index)
ax.set_title(u'一致预期预测模型下OperatingRevenueGrowRate的收益分月提升表现', fontproperties=font, fontsize=16)

fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ax1 = ax.twinx()
ax.plot(pd.to_datetime(perf_df.index), (perf_df['OperatingRevenueGrowRate']+1).cumprod(), label=u'原始')
ax.plot(pd.to_datetime(perf_df.index), (perf_df['con_org']+1).cumprod(), label=u'一致预期预测')
ax1.plot(pd.to_datetime(perf_df.index), (perf_df['con_org']-perf_df['OperatingRevenueGrowRate']+1).cumprod(), label=u'相对收益(右轴)', color='r')
ax.grid(False)
ax1.grid(False)
ax.legend(bbox_to_anchor=(0.5, -0.1), prop=font, ncol=2)
ax1.legend(bbox_to_anchor=(0.65, -0.1), prop=font)
ax.set_title(u'一致预期预测模型下OperatingRevenueGrowRate的累计收益提升表现', fontproperties=font, fontsize=16)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

一致预期预测模型计算的OperatingRevenueGrowRate因子，在4月份的改进效果最好，8月份一般，10月份最差。整体改进效果不太稳定。
OperatingRevenueGrowRate因子在2010年-2019年的累计提升相对收益为4.8%。
   调试 运行
文档
 代码  策略  文档
总体上，一致预期预测模型能够达到一定的改进作用，但是效果不太稳定。
   调试 运行
文档
 代码  策略  文档
第四部分：使用线性预测模型预测财务数据，改进财报月的基本面因子
该部分耗时 25分钟
下文使用线性预测模型，预测财报月的数据。然后以此预测数据，计算线性预测模型下在财报月下的基本面因子，检验因子表现提升效果。

该部分内容为：

4.1 建立线性预测模型
4.2 使用线性预测模型预测数据计算ETOP因子，比较因子分月表现。
4.3 使用线性预测模型预测数据计算OperatingRevenueGrowRate因子，比较因子分月表现。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
4.1 建立线性预测模型
因为不同行业财报数据存在差异性，根据申万一级行业分类，对每一个行业构建线性预测模型。具体步骤如下：
步骤一：抽取预测所需的特征变量，分为五大类：
1）财务指标TTM类：营业收入TTM(revenue_ttm), 营业成本TTM(COGS_ttm), 费用TTM（fee_ttm = 销售费用sellExp+管理费用adminExp+财务费用finanExp）, 归母净利润TTM(NIncomeAttrP_ttm);
当期财务指标类：货币资金（cashCEquiv）, 应收账款（AR）, 存货（inventories）, 其他流动资产（othCA）, 固定资产（fixedAssets）, 流动负债（TCL）, 应付账款（AP）, 应交税费（taxesPayable）, 其他流动负债（othCL）, 总负债（TLiab）
3）财务指标当季度数据类：单季度营业收入(revenue_q), 单季度营业成本(COGS_q), 单季度费用（fee_q = 销售费用sellExp+管理费用adminExp+财务费用finanExp）, 单季度归母净利(NIncomeAttrP_q)
4）财务指标同比增长类：营业收入同比增长（revenueYOY）, 归母净利润同比增长（niAttrPYOY）, 经营活动产生的现金流量净额同比增长（nCfOpaYOY）
5）财务指标盈利能力类：销售毛利率（grossMARgin）, 销售净利润率（npMARgin）, 净资产收益率（ROE）
步骤二：对所抽取的特征进行处理：填充缺失值（行业中位数填充）、标准化。对预测数据进行缩尾（4倍绝对中位数差以外数据拉回）、标准化处理。
步骤三：对每一个行业，使用该行业所有个股在历史上过去3年（过去12个财报期）的数据作为训练集，采用Lasso回归进行L1正则处理，对特征进行降维处理，训练出线性预测模型。
步骤四：在3、7、9月底，使用当前的特征数据和线性预测模型，计算得到财报月的财报预测数据。

'''

start_time = time.time()
print ("该部分为线性预测模型准备数据...")

def get_pre_end(date):
    """
    获取上一个期末日期
    参数:
        date: str, %Y%m%d
    返回:
        pre_end: str, %Y%m%d
    """
    if date[4:6] == '03':
        pre_end = str(int(date[:4])-1)+'1231'
    elif date[4:6] == '06':
        pre_end = date[:4]+'0331'
    elif date[4:6] == '09':
        pre_end = date[:4]+'0630'
    elif date[4:6] == '12':
        pre_end = date[:4]+'0930'
    return pre_end


def process_feature(fiscal_df):
    '''
    特征的去重处理
    '''
    fiscal_df = fiscal_df.sort_values(['ticker', 'endDate', 'publishDate'])
    fiscal_df = fiscal_df.drop_duplicates(subset=['ticker', 'endDate'], keep='last')
    return fiscal_df

def win(df, col_list, date_col):
    '''
    特征的缩尾处理:绝对中位数差法
    参数：
        df: DataFrame, 特征数据
        col_list: list, 特征列表
        date_col: str, 时间字段名称
    返回：
        返回缩尾处理后的特征数据
    '''
    for col in col_list:
        med = df.groupby(date_col)[col].median()
        dm = med.reset_index()
        dm.columns = [date_col, 'median']
        df = pd.merge(df, dm, on=date_col)
        df['dm'] = (df[col] - df['median']).abs()
        dm1 = df.groupby(date_col)['dm'].median()
        upper = (med + 4*dm1).reset_index()
        upper.columns = [date_col, 'upper']
        lower = (med - 4*dm1).reset_index()
        lower.columns = [date_col, 'lower']
        df = df.merge(upper, on=[date_col]).merge(lower, on=[date_col])
        df[col] = np.where(df[col] > df['upper'], df['upper'], df[col])
        df[col] = np.where(df[col] < df['lower'], df['lower'], df[col])
        df = df.drop(['upper', 'lower', 'median', 'dm'], axis=1)
    return df

def fill_indu_median(df, col_list):
    '''
    特征的填充缺失处理（行业中位数填充）
    参数：
        df: DataFrame, 特征数据
        col_list: list, 特征列表
    返回：
        返回填充缺失处理后的特征数据
    '''
    df = df.copy()
    for col in col_list:
        df[col] = df[col].fillna(df[col].median())
    return df

def lasso_fit(df, x_col_list, y_col):
    '''
    Lasso回归模型
    参数：
        df: DataFrame, 训练集数据
        x_col_list: list, 特征列表
        y_col: str, 预测字段
    返回：
        训练得到的预测模型
    '''
    alpha = 0.001
    lasso = Lasso(alpha=alpha)
    model = lasso.fit(df[x_col_list], df[y_col])
    return model

## 特征数据准备
# 财务指标TTM类
fiscal_ttm_df = DataAPI.FdmtISTTMPITGet(secID=a_universe_list, beginDate='20060930',field=u"ticker,publishDate,endDate,revenue,COGS,sellExp,adminExp,finanExp,NIncomeAttrP",pandas="1")
fiscal_ttm_df['publishDate'] = fiscal_ttm_df['publishDate'].apply(lambda x: x.replace('-', ''))
fiscal_ttm_df['endDate'] = fiscal_ttm_df['endDate'].apply(lambda x: x.replace('-', ''))
fiscal_ttm_df['fee_ttm'] = fiscal_ttm_df[['sellExp', 'adminExp', 'finanExp']].sum(axis=1)
fiscal_ttm_df = fiscal_ttm_df.drop(['sellExp', 'adminExp', 'finanExp'], axis=1)
# 当期财务指标类
fiscal_mrq_df = DataAPI.FdmtBSGet(secID=a_universe_list, beginDate='20060930',field=u"ticker,publishDate,endDate,cashCEquiv,AR,inventories,othCA,fixedAssets,TCL,AP,taxesPayable,othCL,TLiab",pandas="1")
fiscal_mrq_df['publishDate'] = fiscal_mrq_df['publishDate'].apply(lambda x: x.replace('-', ''))
fiscal_mrq_df['endDate'] = fiscal_mrq_df['endDate'].apply(lambda x: x.replace('-', ''))
# 财务指标当季度数据类
fiscal_q_df = DataAPI.FdmtISQPITGet(secID=a_universe_list, beginDate='20060930',field=u"ticker,publishDate,endDate,revenue,COGS,sellExp,adminExp,finanExp,NIncomeAttrP",pandas="1")
fiscal_q_df['publishDate'] = fiscal_q_df['publishDate'].apply(lambda x: x.replace('-', ''))
fiscal_q_df['endDate'] = fiscal_q_df['endDate'].apply(lambda x: x.replace('-', ''))
fiscal_q_df['fee_q'] = fiscal_q_df[['sellExp', 'adminExp', 'finanExp']].sum(axis=1)
fiscal_q_df = fiscal_q_df.drop(['sellExp', 'adminExp', 'finanExp'], axis=1)
# 财务指标同比增长类
fiscal_growth_df = DataAPI.FdmtIndiGrowthPitGet(secID=a_universe_list, beginDate='20060930',field=u"ticker,publishDate,endDate,revenueYOY,niAttrPYOY,nCfOpaYOY",pandas="1")
fiscal_growth_df['publishDate'] = fiscal_growth_df['publishDate'].apply(lambda x: x.replace('-', ''))
fiscal_growth_df['endDate'] = fiscal_growth_df['endDate'].apply(lambda x: x.replace('-', ''))
# 财务指标盈利能力类
fiscal_rtn_df = DataAPI.FdmtIndiRtnPitGet(secID=a_universe_list, beginDate='20060930',field=u"ticker,publishDate,endDate,grossMARgin,npMARgin,ROE",pandas="1")
fiscal_rtn_df['publishDate'] = fiscal_rtn_df['publishDate'].apply(lambda x: x.replace('-', ''))
fiscal_rtn_df['endDate'] = fiscal_rtn_df['endDate'].apply(lambda x: x.replace('-', ''))

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


def linear_predict_model(date, y_col, x_col_list, data_len=3, r_thredshold=0):
    '''
    构建线性预测模型，预测财报月的财报数据; 已公布财报的，则为正式报告的数据
    参数：
        date: str, 3、7、10月的月底交易日日期，格式为"%Y%m%d"
        y_col: str, 需要进行预测的财报字段
        x_col_list：list, 特征列表
        data_len：int, 训练集长度
    返回：
        使用线性预测模型预测得到的财报数据
    '''
    # 确定预测数据的财报期，以及训练集的财报期范围
    if date[4:6] == '03':
        y_pred_end = str(int(date[:4])-1)+'1231'
        ispub_list = [str(int(date[:4])-1)+'12', date[:4]+'03']
        rep_end = str(int(date[:4])-1)+'1231'
    elif date[4:6] == '07':
        y_pred_end = date[:4]+'0630'
        ispub_list = [date[:4]+'06']
        rep_end = date[:4]+'0630'
    else:
        y_pred_end = date[:4]+'0930'
        ispub_list = [date[:4]+'09']
        rep_end = date[:4]+'0930'
    x_pred_end = get_pre_end(y_pred_end)
    y_train_end_begin = str(int(y_pred_end) - data_len*10000)
    x_train_end_begin = str(int(x_pred_end) - data_len*10000)
    
    ## 步骤一：抽取预测所需的特征变量
    d_fiscal_ttm_df = fiscal_ttm_df[(fiscal_ttm_df['publishDate'] <= date) & (fiscal_ttm_df['endDate'] <= y_pred_end) & (fiscal_ttm_df['endDate'] >= x_train_end_begin)]
    d_fiscal_ttm_df = process_feature(d_fiscal_ttm_df)
    
    d_fiscal_mrq_df = fiscal_mrq_df[(fiscal_mrq_df['publishDate'] <= date) & (fiscal_mrq_df['endDate'] <= y_pred_end) & (fiscal_mrq_df['endDate'] >= x_train_end_begin)]
    d_fiscal_mrq_df = process_feature(d_fiscal_mrq_df)
    
    d_fiscal_q_df = fiscal_q_df[(fiscal_q_df['publishDate'] <= date) & (fiscal_q_df['endDate'] <= y_pred_end) & (fiscal_q_df['endDate'] >= x_train_end_begin)]
    d_fiscal_q_df = process_feature(d_fiscal_q_df)
    
    d_fiscal_growth_df = fiscal_growth_df[(fiscal_growth_df['publishDate'] <= date) & (fiscal_growth_df['endDate'] <= y_pred_end) & (fiscal_growth_df['endDate'] >= x_train_end_begin)]
    d_fiscal_growth_df = process_feature(d_fiscal_growth_df)
    
    d_fiscal_rtn_df = fiscal_rtn_df[(fiscal_rtn_df['publishDate'] <= date) & (fiscal_rtn_df['endDate'] <= y_pred_end) & (fiscal_rtn_df['endDate'] >= x_train_end_begin)]
    d_fiscal_rtn_df = process_feature(d_fiscal_rtn_df)
    
    # 行业分类
    indu_df = DataAPI.EquIndustryGet(industryVersionCD=u"010303",intoDate=date,field=u"ticker,industryName1",pandas="1")
    
    data_df = d_fiscal_ttm_df.merge(d_fiscal_mrq_df, on=['ticker', 'endDate']).merge(d_fiscal_q_df, on=['ticker', 'endDate'], suffixes=('_tmm', '_q')).merge(d_fiscal_growth_df, on=['ticker', 'endDate']).merge(d_fiscal_rtn_df, on=['ticker', 'endDate'])
    data_df['endDate']= data_df['endDate'].apply(lambda x: x.replace('-', ''))
    
    # 整理出训练集和预测集
    x_pred = data_df[data_df['endDate'] == x_pred_end][['ticker', 'endDate'] + x_col_list]
    y_train =  data_df[(data_df['endDate'] >= y_train_end_begin) & (data_df['endDate'] <= x_pred_end)][['ticker', 'endDate', y_col]]
    y_train['x_endDate'] = y_train['endDate'].apply(get_pre_end)
    train_data = y_train.merge(data_df[['ticker', 'endDate'] + x_col_list], left_on=['ticker', 'x_endDate'], right_on=['ticker', 'endDate'], how='left', suffixes=('_y', '_x'))
    train_data = train_data.merge(indu_df, on=['ticker']) 
    x_col_list = [col+'_x' if col == y_col else col for col in x_col_list]
    y_col = y_col+'_y'
    x_pred.columns = ['ticker', 'endDate'] + x_col_list
    x_pred = x_pred.merge(indu_df, on=['ticker'])
    
    
    ## 步骤二：特征和预测变量的标准化处理
    # 对预测变量进行缩尾处理
    w_train_data = train_data.groupby('industryName1', group_keys=False).apply(lambda df: win(df, [y_col], 'endDate_x'))
    w_x_pred = x_pred.copy()

    # 对抽取的特征进行填值处理
    wf_train_data = w_train_data.groupby(['industryName1', 'endDate_x'], group_keys=False).apply(fill_indu_median, x_col_list)
    wf_x_pred = w_x_pred.groupby(['industryName1'], group_keys=False).apply(fill_indu_median, x_col_list)

    # 对抽取的特征和预测变量进行标准化处理
    ss_x = StandardScaler()
    wf_train_data = wf_train_data.dropna(subset=[y_col])
    wf_train_data[x_col_list] = ss_x.fit_transform(wf_train_data[x_col_list].fillna(0)) 
    wf_x_pred[x_col_list] = ss_x.transform(wf_x_pred[x_col_list].fillna(0))  
    wf_train_data = wf_train_data.sort_values(['industryName1', 'ticker', 'endDate_y'])
    wf_x_pred = wf_x_pred.sort_values(['industryName1', 'ticker'])
    ss = StandardScaler()
    wf_train_data[y_col] = ss.fit_transform(wf_train_data[y_col])
    

    ## 步骤三：分行业建模
    lasso_model = wf_train_data.groupby('industryName1').apply(lasso_fit, x_col_list, y_col).reset_index()
    lasso_model.columns = ['industryName1', 'model']
    wf_x_pred = wf_x_pred.merge(lasso_model, on=['industryName1'])
    
    # 计算预测值
    wf_x_pred['y_pred'] = wf_x_pred.apply(lambda x: x['model'].predict(x[x_col_list])[0], axis=1)
    wf_x_pred['y_pred'] = ss.inverse_transform(wf_x_pred['y_pred'])
    
    # 计算样本内R2
    wf_train_data = wf_train_data.merge(lasso_model, on=['industryName1'])
    wf_train_data['y_insimple'] = wf_train_data.apply(lambda x: x['model'].predict(x[x_col_list])[0], axis=1)
    r2_insample = wf_train_data.groupby('industryName1').apply(lambda df: r2_score(df[y_col], df['y_insimple'])).reset_index()
    r2_insample.columns = ['industryName1', 'r2_is']
    y_pred = wf_x_pred[['ticker', 'y_pred', 'industryName1']]
    y_pred = y_pred.merge(r2_insample, on=['industryName1'])
    
    ## 没有发布财报的个股，用一致预期数据替代；发布财报的个股，使用正式报告的数据
    # 获取财报数据
    fiscal_col = y_col.split('_')[0]
    fiscal_df = fiscal_ttm_df[(fiscal_ttm_df['publishDate'] <= date)][['ticker', 'publishDate', 'endDate', fiscal_col]]
    fiscal_df = fiscal_df.sort_values(['ticker', 'endDate', 'publishDate'])
    fiscal_df = fiscal_df.drop_duplicates(['ticker'], keep='last')
    fiscal_df['flag'] = fiscal_df['endDate'].apply(lambda x: x[:6] in ispub_list)
    
    merge_df = y_pred.merge(fiscal_df, on='ticker', how='outer')
    merge_df['flag'] = np.where(merge_df['y_pred'].isnull(), True, merge_df['flag'])
    merge_df['flag'] = np.where(merge_df['r2_is'] < r_thredshold, True, merge_df['flag'])
    merge_df['value'] = np.where(merge_df['flag'], merge_df[fiscal_col], merge_df['y_pred'])
    merge_df['type'] = np.where(merge_df['flag'], 'history_ttm', 'linear_predict')
    merge_df['endDate'] = np.where(merge_df['flag'], merge_df['endDate'], rep_end)
    merge_df['tradeDate'] = date
    merge_df = merge_df[['ticker', 'tradeDate', 'endDate', 'value', 'industryName1', 'r2_is', 'y_pred', 'type']]
    
    return merge_df

'''

4.2 使用线性预测模型计算ETOP因子

具体做法是：在3、7、9月底，构建线性预测模型，然后用当期特征数据，预测财报月的归母净利润TTM数据，然后除以当前的总市值，得到线性预测模型下的ETOP因子值。
在3、7、9月底，已发布财报的个股，就是用正式报告的数据计算因子；没有发布财报的公司，则用预测数据计算因子。

'''

start_time = time.time()
print ("该部分计算线性预测模型下财报月的ETOP, 并比较分月表现..")

# 线性模型预测财报月归母净利润
fiscal_date_list = [date for date in monthly_dates_list if date[4:6] in ['03', '07', '09']]
y_col = 'NIncomeAttrP_tmm'
x_col_list = ['revenue_tmm', 'COGS_tmm', 'NIncomeAttrP_tmm', 'fee_ttm', 'cashCEquiv', 'AR', 'inventories', 'othCA', 'fixedAssets', 'TCL', 'AP', 'taxesPayable', 'othCL', 'TLiab', 'revenue_q', 'COGS_q', 'NIncomeAttrP_q', 'fee_q', 'revenueYOY', 'niAttrPYOY', 'nCfOpaYOY', 'grossMARgin', 'npMARgin', 'ROE']
linear_predcit_nincome_df = []
for date in fiscal_date_list:
    y_pred = linear_predict_model(date, y_col, x_col_list, data_len=3, r_thredshold=0)
    linear_predcit_nincome_df.append(y_pred)
linear_predcit_nincome_df = pd.concat(linear_predcit_nincome_df)

#  获取总市值数据
mkt_df = qutil.get_data_items(a_universe_list, monthly_dates_list, ['MktValue'])
mkt_df = pd.concat(mkt_df)

# 计算线性预测模型的ETOP
linear_ep_df = linear_predcit_nincome_df.merge(mkt_df, on=['ticker', 'tradeDate'], how='left').dropna()
linear_ep_df['linear_ep'] = linear_ep_df['value'] / linear_ep_df['MktValue']
linear_ep_df = linear_ep_df[['ticker', 'tradeDate', 'linear_ep']]

# 因子处理：去极值，标准化，行业市值中性化
linear_ep_ap_df = qutil.mad_winsorize(linear_ep_df, ['linear_ep'])
linear_ep_ap_df['linear_ep'] = linear_ep_ap_df.groupby('tradeDate')['linear_ep'].apply(lambda x: (x - x.mean()) / x.std())
exclude_style = ['BETA', 'MOMENTUM', 'SIZENL', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY']
linear_ep_ap_df = qutil.netralize_dframe(linear_ep_ap_df, ['linear_ep'], exclude_style)
linear_ep_ap_df_all =  factor_ap_df[['ticker', 'tradeDate', 'ETOP']].merge(linear_ep_ap_df, on=['ticker', 'tradeDate'], how='left')
linear_ep_ap_df_all['linear_ep'] = np.where(linear_ep_ap_df_all['linear_ep'].isnull(), linear_ep_ap_df_all['ETOP'], linear_ep_ap_df_all['linear_ep'])

# 比较线性预测模型的ETOP和原始ETOP的提升
perf_list = []
perf,_ = qutil.easy_backtest(factor_ap_df, mret_df, 'ETOP', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['ETOP']
perf_list.append(perf)
perf,_ = qutil.easy_backtest(linear_ep_ap_df_all, mret_df, 'linear_ep', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['linear_ep']
perf_list.append(perf)

perf_df = pd.concat(perf_list, axis=1)

# 分析因子在不同月份的收益率
perf_df['month'] = [date[4:6]for date in perf_df.index]
month_perf = perf_df.groupby('month')['ETOP', 'linear_ep'].mean()
month_perf['diff'] = month_perf['linear_ep'] - month_perf['ETOP']
# 画图
fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ind = np.arange(12)
ax.bar(ind+0.2, month_perf['diff'], width=0.5)
ax.set_xticks(ind+0.45)
ax.set_xticklabels(month_perf.index)
ax.set_title(u'线性预测模型下ETOP的收益分月提升表现', fontproperties=font, fontsize=16)

fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ax1 = ax.twinx()
ax.plot(pd.to_datetime(perf_df.index), (perf_df['ETOP']+1).cumprod(), label=u'原始')
ax.plot(pd.to_datetime(perf_df.index), (perf_df['linear_ep']+1).cumprod(), label=u'线性模型预测')
ax1.plot(pd.to_datetime(perf_df.index), (perf_df['linear_ep']-perf_df['ETOP']+1).cumprod(), label=u'相对收益(右轴)', color='r')
ax.grid(False)
ax1.grid(False)
ax.legend(bbox_to_anchor=(0.5, -0.1), prop=font, ncol=2)
ax1.legend(bbox_to_anchor=(0.65, -0.1), prop=font)
ax.set_title(u'线性预测模型下ETOP的累计收益提升表现', fontproperties=font, fontsize=16)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''


线性预测模型计算的ETOP因子，在8、10月份的改进效果均不错，4月份比原始因子差，和一致预期预测模型存在一样的问题。
相较于一致预期预测模型，线性预测模型的效果更佳稳定。
ETOP因子在2010年-2019年的累计提升相对收益为11.8%。
   调试 运行
文档
 代码  策略  文档
4.3 使用线性预测模型计算OperatingRevenueGrowRate因子

具体做法是：在3、7、9月底，构建线性预测模型，然后用当期特征数据，预测财报月的营业收入TTM数据，结合去年同期的营业收入TTM数据，计算增长率，得到线性预测模型下的OperatingRevenueGrowRate因子值。
在3、7、9月底，已发布财报的个股，就是用正式报告的数据计算因子；没有发布财报的公司，则用线性预测模型预测数据计算因子。

'''

start_time = time.time()
print ("该部分计算线性预测模型下财报月的OperatingRevenueGrowRate, 并比较分月表现..")

# # 线性模型预测财报月营业收入
fiscal_date_list = [date for date in monthly_dates_list if date[4:6] in ['03', '07', '09']]
y_col = 'revenue_tmm'
x_col_list = ['revenue_tmm', 'COGS_tmm', 'NIncomeAttrP_tmm', 'fee_ttm', 'cashCEquiv', 'AR', 'inventories', 'othCA', 'fixedAssets', 'TCL', 'AP', 'taxesPayable', 'othCL', 'TLiab', 'revenue_q', 'COGS_q', 'NIncomeAttrP_q', 'fee_q', 'revenueYOY', 'niAttrPYOY', 'nCfOpaYOY', 'grossMARgin', 'npMARgin', 'ROE']
linear_predcit_sales_df = []
for date in fiscal_date_list:
    y_pred = linear_predict_model(date, y_col, x_col_list, data_len=3, r_thredshold=0.8)
    linear_predcit_sales_df.append(y_pred)
linear_predcit_sales_df = pd.concat(linear_predcit_sales_df)

# 计算线性预测模型的OperatingRevenueGrowRate
sales_df = DataAPI.FdmtISTTMPITGet(secID=a_universe_list, publishDateEnd=edate, field=u"ticker,publishDate,endDate,revenue",pandas="1")
sales_df['publishDate'] = sales_df['publishDate'].apply(lambda x: x.replace('-', ''))
sales_df['endDate'] = sales_df['endDate'].apply(lambda x: x.replace('-', ''))
linear_predcit_sales_df['pre_end'] = linear_predcit_sales_df['endDate'].apply(get_pre_end)
linear_org_df = linear_predcit_sales_df.merge(sales_df, left_on=['ticker', 'pre_end'], right_on=['ticker', 'endDate'], how='left')
linear_org_df = linear_org_df[linear_org_df['tradeDate'] >= linear_org_df['publishDate']]
linear_org_df = linear_org_df.sort_values(['ticker', 'tradeDate', 'publishDate'])
linear_org_df = linear_org_df.drop_duplicates(subset=['ticker', 'tradeDate'], keep='last')
linear_org_df['linear_org'] = (linear_org_df['value'] - linear_org_df['revenue']) / abs(linear_org_df['revenue'])
linear_org_df = linear_org_df[['ticker', 'tradeDate', 'linear_org']]

# 因子处理：去极值，标准化，行业市值中性化
linear_org_ap_df = qutil.mad_winsorize(linear_org_df, ['linear_org'])
linear_org_ap_df['linear_org'] = linear_org_ap_df.groupby('tradeDate')['linear_org'].apply(lambda x: (x - x.mean()) / x.std())
exclude_style = ['BETA', 'MOMENTUM', 'SIZENL', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY']
linear_org_ap_df = qutil.netralize_dframe(linear_org_ap_df, ['linear_org'], exclude_style)
linear_org_ap_df_all =  factor_ap_df[['ticker', 'tradeDate', 'OperatingRevenueGrowRate']].merge(linear_org_ap_df, on=['ticker', 'tradeDate'], how='left')
linear_org_ap_df_all['linear_org'] = np.where(linear_org_ap_df_all['linear_org'].isnull(), linear_org_ap_df_all['OperatingRevenueGrowRate'], linear_org_ap_df_all['linear_org'])

# 比较线性预测模型的OperatingRevenueGrowRate和原始OperatingRevenueGrowRate的提升
perf_list = []
perf,_ = qutil.easy_backtest(factor_ap_df, mret_df, 'OperatingRevenueGrowRate', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['OperatingRevenueGrowRate']
perf_list.append(perf)
perf,_ = qutil.easy_backtest(linear_org_ap_df_all, mret_df, 'linear_org', 'nxt1m_ret', method='long_short', direction=1, ngrp=10)
perf = perf[['tradeDate', 'period_ret']].set_index('tradeDate')
perf.columns = ['linear_org']
perf_list.append(perf)

perf_df = pd.concat(perf_list, axis=1)

# 分析因子在不同月份的收益率
perf_df['month'] = [date[4:6]for date in perf_df.index]
month_perf = perf_df.groupby('month')['OperatingRevenueGrowRate', 'linear_org'].mean()
month_perf['diff'] = month_perf['linear_org'] - month_perf['OperatingRevenueGrowRate']

# 画图
fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ind = np.arange(12)
ax.bar(ind+0.2, month_perf['diff'], width=0.5)
ax.set_xticks(ind+0.45)
ax.set_xticklabels(month_perf.index)
ax.set_title(u'线性预测模型下OperatingRevenueGrowRate的收益分月提升表现', fontproperties=font, fontsize=16)

fig= plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
ax1 = ax.twinx()
ax.plot(pd.to_datetime(perf_df.index), (perf_df['OperatingRevenueGrowRate']+1).cumprod(), label=u'原始')
ax.plot(pd.to_datetime(perf_df.index), (perf_df['linear_org']+1).cumprod(), label=u'线性模型预测')
ax1.plot(pd.to_datetime(perf_df.index), (perf_df['linear_org']-perf_df['OperatingRevenueGrowRate']+1).cumprod(), label=u'相对收益(右轴)', color='r')
ax.grid(False)
ax1.grid(False)
ax.legend(bbox_to_anchor=(0.5, -0.1), prop=font, ncol=2)
ax1.legend(bbox_to_anchor=(0.65, -0.1), prop=font)
ax.set_title(u'线性预测模型下OperatingRevenueGrowRate的累计收益提升表现', fontproperties=font, fontsize=16)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

线性预测模型计算的OperatingRevenueGrowRate因子，在4、8、10月均有提升，但是提升幅度较小。
相较于一致预期预测模型，线性预测模型的效果依然更佳稳定。
OperatingRevenueGrowRate因子在2010年-2019年的累计提升相对收益为5.4%。

'''
