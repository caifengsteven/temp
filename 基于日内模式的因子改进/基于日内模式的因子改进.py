# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:34:40 2020

@author: Asus
"""
'''

基于日内模式的因子改进
导读
A. 研究目的：本文利用优矿提供的行情数据，参考东吴证券《因子方法论之一：基于日内模式的因子改进》（原作者：高子剑、魏建榕）中的研究方法，对研报的结果进行了实证分析，提供一个日内信息改进因子的普适思路。

B. 研究结论：

基于2013-04-01至2017-10-31月度调仓回测结果，基于换手率的局部流动性因子存在额外的收益贡献，10：30-11：30时段的局部换手率因子收益预测能力表现不俗且稳定。

合并局部流动性因子和传统流动性因子，得到改进后的流动性因子。基于2013-01-01到2017-12-31的月度调仓回测结果，相对于原始流动性因子，改进后的因子的优点是降低了收益预测的波动性，从而降低收益波动率，提高收益风险比，且对原始因子的市值衰减性有一定的缓解效果，更具有行业普适性；缺点是提高了因子换手率。

C. 文章结构：本文共分为3个部分，具体如下

一、函数工具准备，数据准备以及因子值计算。

二、对传统流动性因子和局部流动性因子进行绩效评估分析，包括IC统计、多空对冲回测、剔除风格因子和行业因子后的测试。

三、逐年合成局部流动性因子和传统流动性因子，计算改进后的流动性因子。然后比较改进后因子和原始因子的绩效表现，包括IC值统计、行业内选股能力、市值衰减特性检验、分组多头回测表现、不同样本空间内的表现。

D. 其他说明

一、数据准备和因子值计算，需要30分钟

二、本文测试都剔除了上市不足60个交易日的次新股、ST股以及停牌个股。

三、本文涉及值分组，均是值越大，组别越大。

总耗时40分钟左右

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


第一部分：工具函数和因子值计算
该部分耗时 30分钟
该部分内容为：

1.1 提供一些工具函数，为下文的因子计算和因子分析提供基础。 特别说明：如无特别规定，下文中的分组，均是因子值越大，组别越大.

1.2 利用uqer的日行情数据，计算传统的换手率因子，并关于对数市值因子做中性化处理,消除市值影响，得到基于换手率的流动性因子LIQ，以下简称传统流动性因子LIQ.

1.3 利用优矿分钟级别的行情数据，提取流动性因子的日内信息。将一个交易日分成5段时间, 类似于传统流动性因子的计算方式，计算局部流动性因子, 最后与传统流动性因子正交, 得到5个纯净的局部流动性因子pureLIQ.

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

1.1 工具函数

该部分提供因子计算和分析的工具函数。具体如下。

ticker2secID(ticker)： ticker转换secID

secID2ticker(secID)： secID转换ticker

winsorize_standardize(df, col_name)： 去极值和标准化处理

neutralize_my(df, x_col, y_col)： 中性化处理

get_monthend(df, trade_calendar)： 获取数据中每个月最后一个交易日的数据

calc_ic(signal_df, return_df, factor_name, ret_name, method=‘spearman’): 计算因子IC值，本月和下月因子值的秩相关

ic_describe(ic_df)： 统计IC的均值、标准差、IC_IR、大于0的比例以及下月IC相关系数均值

signal_grouping(signal_df, factor_name, ngrp): 因子分组， 每天根据因子值将股票进行等分

long_short_backtest(signal_df, return_df, factor_name, return_name, direction=1)： 简易因子多空回测组合

perf_describe(perf_df)： 统计因子的回测绩效， 包括年化收益率、年化波动率、夏普比率、最大回撤

'''


import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
#from CAL.PyCAL import *
import matplotlib.pyplot as plt
import time
import os

# 新建文件夹存放数据
if not os.path.exists('raw_data'):
    os.makedirs('raw_data')
    
    
def ticker2secID(ticker):
    """
    ticker转换secID
    转换规则：secID = ticker + 后缀：如果股票属于沪市，则后缀为'.XSHG'，如果属于深市，则后缀为'.XSHE'
    """
    ticker = '0'*(6-len(ticker)) + ticker
    if ticker[0] == '6':
        secID = ticker + '.XSHG'
    else:
        secID = ticker + '.XSHE'
    return secID

def secID2ticker(secID):
    """
    secID转换ticker
    """
    return secID[:6]


def winsorize_standardize(df, col_name):
    """
    去极值和标准化处理
    params:
            df: Dataframe, columns=['ticker', [col_name]], 股票的因子值
            col_name: str, 要做处理的列名
    return：
            Series， 返回去极值和标准化后的值
    """
    tmp_df = df.copy()
    tmp_df = tmp_df[['ticker', col_name]].set_index('ticker')
    after_winsorize = winsorize(tmp_df[col_name].to_dict(), win_type='NormDistDraw', n_draw=1)
    after_standardize = standardize(after_winsorize)
    tmp_df[col_name] = pd.Series(after_standardize)
    return tmp_df[col_name]

def neutralize_my(df, x_col, y_col):
    """
    中性化处理
    params:
            df: Dataframe, columns=['ticker', [col_name]], 股票的因子值
            x_col:str, 回归时的自变量， 即需要消除影响的列名
            y_col:str, 待中性化的列名
    return:
            Series， 返回中性化后的值
    """
    y = df[y_col].values
    x = df[x_col].values
    x = sm.add_constant(x)
    results = sm.OLS(y, x).fit()
    df['res'] = results.resid
    return df['res']

def get_monthend(df, trade_calendar):
    """
    获取数据中每个月最后一个交易日的数据
    params:
            df:Dataframe, 待处理数据
            trade_calendar：Dataframe, 交易日历
    return:
            Dataframe, 返回原始数据中每个月最后一个交易日的数据
    """
    col = df.columns.values
    df = df.merge(trade_calendar, left_on='tradeDate', right_on='calendarDate', how='left')
    df = df[df['isMonthEnd'] == 1]
    return df[col]

def calc_ic(signal_df, return_df, factor_name, ret_name, method='spearman'):
    """
    计算因子IC值, 本月和下月因子值的秩相关
    params: 
            signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一列为股票当日的因子值
            return_df: DataFrame, colunms=['ticker, 'tradeDate'， [next_period_ret]], 收益率， next_period_ret一列为下月的收益率
            factor_name:　str, signal_df中因子值的列名
            ret_name: str, return_df中收益率的列名
            method: : {'spearman', 'pearson'}, 默认'spearman', 指定计算rank IC('spearman')或者Normal IC('pearson')
    return:
            DataFrame, 返回IC值和本月和下月因子值的秩相关
    """
    merge_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'])
    # 计算IC
    ic_df = merge_df.groupby('tradeDate').apply(lambda x: x[[factor_name, ret_name]].corr(method=method).values[0,1]).dropna()
    # 计算邻月IC
    merge_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    merge_df[factor_name+'_next'] = merge_df.groupby('ticker')[factor_name].shift(-1)
    merge_df.dropna(inplace=True)
    next_ic_df = merge_df.groupby('tradeDate').apply(lambda x: x[[factor_name, factor_name+'_next']].corr(method='spearman').values[0,1])
    
    result = pd.concat([ic_df,next_ic_df], axis=1, names=[factor_name, factor_name+'_next_ic'])
    result.columns = [factor_name, factor_name+'_next_ic']
    return result

def ic_describe(ic_df):
    """
    统计IC的均值、标准差、IC_IR、大于0的比例以及下月IC相关系数均值
    params:
            ic_df: DataFrame, IC值， index为日期， columns为因子名， values为各个因子的IC值
    return:
            DataFrame, IC统计
    """
    ic_df = ic_df.dropna()
    # 记录因子个数和因子名
    factor_name = [fname for fname in ic_df.columns.values if '_next_ic' not in fname]
    n = len(factor_name)
    # IC均值
    ic_mean = ic_df[factor_name].mean()
    # IC标准差
    ic_std = ic_df[factor_name].std()
    # IC均值的T统计量
    ic_t = pd.Series(st.ttest_1samp(ic_df[factor_name], 0)[0], index=factor_name)
    # IC_IR
    ic_ir = ic_mean/ic_std
    # IC大于0的比例
    ic_p_pct = (ic_df[factor_name] > 0).sum()/len(ic_df)
    # 下月IC相关系数均值
    ic_auto_corr = ic_df[[fname+'_next_ic' for fname in factor_name]].mean()
    ic_auto_corr.index = factor_name
    
    # IC统计
    ic_table = pd.DataFrame([ic_mean, ic_std, ic_t, ic_ir, ic_p_pct, ic_auto_corr], index=['平均IC', 'IC标准差', 'IC均值T统计量','IC_IR', 'IC大于0的比例', '下月IC相关系数均值'])
    return ic_table.T

def signal_grouping(signal_df, factor_name, ngrp):
    """
    因子分组， 每天根据因子值将股票进行等分，编号0 ~ ngrp-1, 编号越大， 因子值越大
    params:
            signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一类为股票当日的因子值
            factor_name:　str, signal_df中因子值的列名
            ngrp: int, 分组组数
    return:
            DataFrame, signal_df在原本的基础上增加一列'group', 记录每日分组
    """
    signal_df_tmp = signal_df.copy()
    signal_df_tmp.dropna(subset=[factor_name], inplace=True)
    signal_df_tmp['group'] = signal_df_tmp.groupby('tradeDate')[factor_name].apply(lambda x: (x.rank()-1)/len(x)*ngrp).astype(int)                         
    return signal_df_tmp

def long_short_backtest(signal_df, return_df, factor_name, return_name, direction=1):
    """
    简易因子多空回测组合， 根据因子值将个股等分成5组，根据方向指定， 正向操作：做多因子值最大的一组， 做空因子值最小的一组；反向操作：做空因子值最大的一组， 做多因子值最小的一组。
    根据调仓频率，进行交易，返回最后的累计收益率。
    params:
            signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一类为股票当日的因子值
            return_df: DataFrame, columns=['ticker', 'tradeDate', [next_period_return]], 收益率，只含有调仓日，以及下期累计收益率
            factor_name:　str, signal_df中因子值的列名 
            return_name： str, return_df中收益率的列名
            direction： {1,-1}, 操作方向， 1为正向操作， 2为反向操作， 默认为1
    return:
            DataFrame, columns=['tradeDate', 'cum_ret'], 返回累计收益率
    """
    bt_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'], how='right')
    
    # 分成五祖, 保留因子值最大和最小的两组
    bt_df.dropna(subset=[factor_name], inplace=True)
    bt_df = signal_grouping(bt_df, factor_name=factor_name, ngrp=5)
    bt_df = bt_df[bt_df['group'].isin([0, 4])]
    
    # 计算权重：每组等权    
    count_df = bt_df.groupby(['tradeDate', 'group']).apply(lambda x:len(x)).reset_index()
    count_df.columns=['tradeDate', 'group', 'count']
    bt_df = bt_df.merge(count_df, on=['tradeDate', 'group'])
    bt_df['weight'] = 1.0/bt_df['count']
    
    # 如果direction=1, 则做多因子值最大的一组， 做空因子值最小的一组；如果direction=-1, 则做空因子值最大的一组， 做多因子值最小的一组
    bt_df.loc[bt_df['group'] == 4, 'weight'] = bt_df.loc[bt_df['group'] == 4, 'weight']*direction
    bt_df.loc[bt_df['group'] == 0, 'weight'] = bt_df.loc[bt_df['group'] == 0, 'weight']*(-direction)
    
    perf = bt_df.groupby('tradeDate').apply(lambda x: sum(x[return_name]*x['weight'])).reset_index()
    perf.columns = ['tradeDate', 'period_ret']
    perf.sort_values('tradeDate', inplace=True)
    perf['cum_ret'] = (perf['period_ret']+1).cumprod()
    
    # 调整时间
    perf['period_ret'] = perf['period_ret'].shift(1)
    perf.fillna(0, inplace=True)
    perf['cum_ret'] = perf['cum_ret'].shift(1)
    perf.fillna(1, inplace=True)
    
    return perf[['tradeDate', 'period_ret','cum_ret']]

def perf_describe(perf_df):
    """
    统计因子的回测绩效， 包括年化收益率、年化波动率、夏普比率、最大回撤
    params:
            perf_df: DataFrame, 回测的期间收益率， index为日期， columns为因子名， values为因子回测的期间收益率
    return:
            DataFrame, 返回回测绩效
    """
    # 记录因子个数和因子名
    factor_name = perf_df.columns.values
    n = len(factor_name)
    
    # 年化收益率
    ret_mean = perf_df.mean()*12
    # 年化波动率
    ret_std = perf_df.std()*np.sqrt(12.0)
    # 年化IR
    ir = ret_mean / ret_std
    # 最大回撤
    maxdrawdown = {}
    for i in range(n):
        fname = factor_name[i]
        cum_ret = pd.DataFrame((perf_df[fname]+1).cumprod())
        cum_max = cum_ret.cummax()
        maxdrawdown[fname] = ((cum_max-cum_ret)/cum_max).max().values[0]
    maxdrawdown = pd.Series(maxdrawdown)
    # 月度胜率
    win_ret = (perf_df > 0).sum()/len(perf_df)
    
    perf_table = pd.DataFrame([ret_mean, ret_std, ir, maxdrawdown, win_ret], index=['年化收益率', '年化波动率', '夏普比率', '最大回撤', '月度胜率'])
    return perf_table.T

'''

1.2 计算传统流动性因子(8分钟)

首先，计算换手率因子，就是计算近20个交易日的换手率之和，再取对数。记 TRs,t 为股票s第t日的换手率， 则换手率因子计算如下： 
Fs,t=ln(120∑i=019TRs,t−i)
 特别说明，当个股在过去20个交易日出现停牌时，计算非停牌日的换手率均值的对数值作为换手率因子值。

因为原始的换手率因子和市值具有高度负相关性，为了消除市值的影响， 我们将换手率因子关于对数流通市值做中性化处理， 即将换手率因子关于对数流通市值做横截面回归，取残差为基于换手率的流动性因子。记LnFMVs,t 为对数流通市值，流动性因子 LIQs,t 计算如下： 
Fs,t=at+btLnFMVs,t+LIQs,t
在剔除市值影响之前，我们会对因子做以下处理，下面类似操作，也会做同样处理。：

在做横截面回归时，我们会剔除上市不足60个交易日的次新股、ST股以及停牌个股。在该部分会计算一个股票禁止池，包含上述股票，数据文件储存在raw_data/forbidden_pool.csv.

将因子值做去极值、标准化，其中去极值的具体做法是将3倍标准差外的数据压缩到3倍标准差。

计算2009-01-01～2017-12-31之间的因子值，因uqer运行内存原因，只储存月度数据，最后生产的数据文件存储在raw_data/liq.csv.


'''


# 设置起始时间和结束时间
start_date = '2009-01-01'
end_date = '2017-12-31'

# 获取交易日历
trade_calendar = DataAPI.TradeCalGet(exchangeCD=U"XSHG", field=u"calendarDate,isOpen,isMonthEnd")
# 获取全A的secID
a_universe = DataAPI.EquGet(equTypeCD=u"A", field=u"secID",pandas="1")['secID'].tolist()

start_time = time.time()
print ("该部分计算换手率因子...")

# 取日级别行情数据
dmkt_df = DataAPI.MktEqudGet(secID=a_universe, beginDate=start_date, endDate=end_date, field=['ticker', 'tradeDate', 'turnoverRate', 'negMarketValue', 'turnoverVol'])
dmkt_df.sort_values(['ticker', 'tradeDate'], inplace=True)
dmkt_df.loc[dmkt_df['turnoverRate']==0, 'turnoverRate'] = np.nan
#计算换手率因子
dmkt_df['f_raw'] = dmkt_df.groupby('ticker')['turnoverRate'].rolling(20, min_periods=1).apply(lambda x: np.nanmean(x)).values
dmkt_df['f_raw'] = np.log(dmkt_df['f_raw'])
dmkt_df.loc[np.isinf(dmkt_df['f_raw']), 'f_raw'] = np.nan

# 计算对数流通市值
dmkt_df['negMarketValue'] = np.log(dmkt_df['negMarketValue'])
dmkt_df.loc[np.isinf(dmkt_df['negMarketValue']), 'negMarketValue'] = np.nan
print (dmkt_df.tail().to_html())
dmkt_df.to_csv('raw_data/dmkt.csv', index=False)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

dmkt_df = pd.read_csv('raw_data/dmkt.csv', dtype={'ticker':str})

start_time = time.time()
print ("该部分计算禁止池，包括上市不足60个交易日的次新股、ST股以及停牌个股...")

# 获得交易日历
calendar = trade_calendar[trade_calendar['isOpen'] == 1]
calendar = calendar['calendarDate'].tolist()

# 次新股
print('计算次新股')
ipo_info = DataAPI.SecIDGet(assetClass=u"E", field=['ticker', 'listDate'], pandas="1")
ipo_info.dropna(inplace=True)
ticker_list = [ticker for ticker in ipo_info['ticker'] if len(ticker) == 6 and ticker[0] in ['0', '3', '6']]
ipo_info = ipo_info[ipo_info['ticker'].isin(ticker_list)]
ipo_info['permit_date'] = [calendar[calendar.index(date) + 60] if date in calendar else  calendar[0] for date in ipo_info['listDate']]

calendar = np.array(calendar)
new_df = pd.DataFrame()
for date in calendar[(calendar > start_date) & (calendar < end_date)]:
    new_list = ipo_info[(ipo_info['permit_date'] >= date) & (ipo_info['listDate'] <= date)]['ticker'].values
    d_new_df = pd.DataFrame({'tradeDate': [date] * len(new_list), 'ticker': new_list})
    new_df = new_df.append(d_new_df)

new_df['remove_flag'] = 'new'

# ST股
print('计算ST股')
st_info = DataAPI.SecSTGet(beginDate=start_date,endDate=end_date,field=['tradeDate', 'ticker'],pandas="1")
st_df = st_info.copy()
st_df['remove_flag'] = 'st'

# 停牌
print('计算停牌个股')
halt_info = DataAPI.SecHaltGet(beginDate=start_date, endDate=end_date, field=['ticker', 'haltBeginTime', 'haltEndTime'],pandas="1")
halt_info.fillna(calendar[-1], inplace=True)
halt_info['haltBeginTime'] = halt_info['haltBeginTime'].apply(lambda x: x[:10])
halt_info['haltEndTime'] = halt_info['haltEndTime'].apply(lambda x: x[:10])

halt_df = pd.DataFrame()
for date in calendar[(calendar > start_date) & (calendar < end_date)]:
    halt_list = halt_info[(halt_info['haltEndTime'] >= date) & (halt_info['haltBeginTime'] <= date)]['ticker'].values
    d_halt_df = pd.DataFrame({'tradeDate': [date] * len(halt_list), 'ticker': halt_list})
    halt_df = halt_df.append(d_halt_df)

halt_df['remove_flag'] = 'halt'

remove_df = new_df.append(st_df).append(halt_df)
remove_df = remove_df[['tradeDate', 'ticker', 'remove_flag']]
print (remove_df.head().to_html())
remove_df.to_csv('raw_data/forbidden_pool.csv', index=False)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

remove_df = pd.read_csv('raw_data/forbidden_pool.csv', dtype={'ticker':str})

start_time = time.time()
print ("该部分计算消除市值影响的基于换手率的流动性因子...")

# 剔除缺失值以及禁止池股票
ticker_df = dmkt_df.dropna(subset=['f_raw', 'negMarketValue'])
ticker_df = ticker_df.merge(remove_df, on=['ticker', 'tradeDate'], how='left')
ticker_df = ticker_df[pd.isnull(ticker_df['remove_flag'])]
ticker_df.sort_values(['tradeDate', 'ticker'], inplace=True)

# 对换手率因子和对数流通市值做去极值和标准化处理
ticker_df['f_raw'] = ticker_df.groupby('tradeDate').apply(winsorize_standardize, 'f_raw').values
ticker_df['negMarketValue'] = ticker_df.groupby('tradeDate').apply(winsorize_standardize, 'negMarketValue').values
#将换手率因子关于对数流通市值做中性化， 得到流动性因子
ticker_df['liq'] = ticker_df.groupby('tradeDate').apply(neutralize_my, x_col='negMarketValue', y_col='f_raw').values

# 保存月度数据
liq_df = ticker_df[['ticker', 'tradeDate', 'liq']]
liq_df = get_monthend(liq_df, trade_calendar)
print (liq_df.head().to_html())
liq_df.to_csv('raw_data/liq.csv', index=False)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

liq_df = pd.read_csv('raw_data/liq.csv', dtype={'ticker':str})

'''

1.3 计算局部流动性因子(21分钟)

将每个交易日分为互不相交的5个时段：隔夜（开盘前的集合竞价）、9：30-10：30、10：30-11：30、13：00-14:00、14：00-15：00, 则每天的换手率就是这5个时间段的换手率之和， 每个时段的换手率在传统的换手率因子中以等权的形式体现。这样处理的方式会忽略个股之间日内换手率分布情况的差异， 例如假设A股票和B股票日换手率相同， 但是A上午流动性高于B，B下午的流动性高于A，这样的差异在传统流动性因子中并不能体现出来，但是这种信息是否会对选股贡献收益呢？下文进行进一步研究。

研报中通过对分时段的收益率与市场收益率进行回归， 研究局部R方， 发现每个时段包含的信息确实具有差异， 因此A股确实存在日内特定的交易模式。

参照传统换手率因子的计算方法，我们可以计算局部换手率因子。记 TR(0)s,t、 TR(1)s,t、TR(2)s,t、TR(3)s,t、TR(4)s,t为股票s第t日在上述5个交易时段的换手率， 则局部换手率因子计算如下： 
F(k)s,t=ln(120∑i=019TR(k)s,t−i)(k=0,1,2,3,4)
 停牌处理方式与传统流动性因子处理一致。

同样地， 将局部换手率因子关于对数流通市值做中性化处理
F(k)s,t=at+btLnFMVs,t+LIQ(k)s,t(k=0,1,2,3,4)
得到的残差LIQ(k)s,t即为局部流动性因子。

局部流动性因子可近似看作是传统流动性因子的一部分，因此局部流动性因子LIQ(k)s,t和传统流动性因子LIQs,t必然具有高相关性，为了考察局部流动性因子是否能提供额外的选股增益信息，将局部流动性因子LIQ(k)s,t关于传统流动性因子LIQs,t进行回归，最终得到纯净的局部流动性因子，处理如下： 
LIQ(k)s,t=at+btLIQs,t+pureLIQ(k)s,t(k=0,1,2,3,4)
残差pureLIQ(k)s,t就是纯净的局部流动性因子。为了简便，若没有特别说明，以下提及的局部流动性因子均指纯净的局部流动性因子。

同样 计算2009-01-01～2017-12-31之间的因子值，用uqer的datacube接口获取日内行情数据，因为运行内存的原因，代码会分年度进行计算，最后整合成一个csv文件，存储在raw_data/pureliq.csv.

'''

start_time = time.time()
print ("该部分计算局部换手率因子...")

# 需要计算的年份
start_year = int(start_date[:4])
end_year = int(end_date[:4])
year_list = range(start_year, end_year+1)

# 获取股本数据
dmkt_df['shares'] = dmkt_df['turnoverVol']/dmkt_df['turnoverRate']  #当turnoverRate=0（停牌），shares计算为空

# 每年取数据，进行计算
for year in year_list:
    
    print ("计算%d年的因子值..."% year)
    # 获得小时级别交易量数据
    data = get_data_cube(symbol=a_universe, field=['turnoverVol'], start=str(year-1)+'-12-01', end=str(year)+'-12-31', freq='60m', style='ast')
    
    # 将panel转换为dataframe
    raw_data_df = data.to_frame().reset_index()
    raw_data_df.rename(columns={'minor': 'secID'}, inplace=True)
    raw_data_df['ticker'] = raw_data_df['secID'].apply(secID2ticker)
    raw_data_df['tradeDate'] = raw_data_df['tradeTime'].apply(lambda x: x[:10])
    raw_data_df['Time'] = raw_data_df['tradeTime'].apply(lambda x: x[11:])
    pturnover_df = raw_data_df.pivot_table(values=['turnoverVol'], index=['ticker', 'tradeDate'], columns=['Time'])
    pturnover_df.reset_index(col_level=1, inplace=True)
    pturnover_df = pturnover_df.T.reset_index(drop=True).T
    pturnover_df.columns=['ticker', 'tradeDate', 'tv0', 'tv1', 'tv2', 'tv3', 'tv4']
    
    # 计算局部换手率因子
    pturnover_df = pturnover_df.merge(dmkt_df[['ticker', 'tradeDate', 'shares', 'negMarketValue']], on=['ticker', 'tradeDate'])
    pturnover_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    for i in np.arange(5).astype(str):
        pturnover_df['tv'+i] = pturnover_df['tv'+i]/pturnover_df['shares']  # 停牌时，局部成交量为nan
        pturnover_df.loc[pturnover_df['tv'+i] == 0, 'tv'+i] = np.nan
        pturnover_df['f_raw'+i] = pturnover_df.groupby('ticker')['tv'+i].rolling(20, min_periods=1).apply(lambda x: np.nanmean(x)).values
        pturnover_df['f_raw'+i] = np.log(pturnover_df['f_raw'+i])
        pturnover_df.loc[np.isinf(pturnover_df['f_raw'+i]), 'f_raw'+i] = np.nan
    pturnover_df = pturnover_df[['ticker', 'tradeDate', 'f_raw0', 'f_raw1', 'f_raw2', 'f_raw3', 'f_raw4', 'negMarketValue']]
    pturnover_df = pturnover_df[pturnover_df['tradeDate'] >= str(year)+'-01-01']
    pturnover_df.dropna(inplace=True)
    pturnover_df = get_monthend(pturnover_df, trade_calendar)
    
    pturnover_df.to_csv('raw_data/pturnover_%d.csv'%year, index=False)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


start_time = time.time()
print ("该部分将几年的换手率因子合成一个DataFrame,然后计算纯净的局部流动性因子...")

# 合成一个DtaFrame
pturnover_df = pd.DataFrame()
for year in year_list:
    tmp_df = pd.read_csv('raw_data/pturnover_%d.csv'%year, dtype={'ticker': str})
    pturnover_df = pturnover_df.append(tmp_df)

# 剔除缺失值和禁止池股票
pturnover_df.dropna(subset=['f_raw0', 'f_raw1', 'f_raw2', 'f_raw3', 'f_raw4', 'negMarketValue'], inplace=True)
pturnover_df = pturnover_df.merge(remove_df, on=['ticker', 'tradeDate'], how='left')
pturnover_df = pturnover_df[pd.isnull(pturnover_df['remove_flag'])]
pturnover_df = pturnover_df.merge(liq_df, on=['ticker', 'tradeDate'])
pturnover_df.sort_values(['tradeDate', 'ticker'], inplace=True)

# 对局部换手率和对数流通市值做去极值标准化处理， 然后将局部换手率因子关于对数流通市值做中性化， 得到局部流动性因子再与传统流动性因子中性化，
pturnover_df['negMarketValue'] = pturnover_df.groupby('tradeDate').apply(winsorize_standardize, 'negMarketValue').values
for i in np.arange(5).astype(str):
    pturnover_df['f_raw'+i] = pturnover_df.groupby('tradeDate').apply(winsorize_standardize, 'f_raw'+i).values
    pturnover_df['liq'+i] = pturnover_df.groupby('tradeDate').apply(neutralize_my, x_col='negMarketValue', y_col='f_raw'+i).values
    pturnover_df['pure_liq'+i] = pturnover_df.groupby('tradeDate').apply(neutralize_my, x_col='liq', y_col='liq'+i).values
    
pureliq_df = pturnover_df[['ticker', 'tradeDate', 'pure_liq0', 'pure_liq1', 'pure_liq2', 'pure_liq3', 'pure_liq4']]
print (pureliq_df.head().to_html())
pureliq_df.to_csv('raw_data/pureliq.csv', index=False)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

pureliq_df = pd.read_csv('raw_data/pureliq.csv', dtype={'ticker':str})

'''
第二部分：传统流动性因子和局部流动性的因子分析
该部分耗时 2分钟
该部分内容为因子的效果分析, 具体包括：

2.1 传统流动性因子LIQ的效果分析

2.2 局部流动性因子pureLIQ的效果分析

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

2.1 传统流动性因子LIQ的效果分析

考察传统流动性因子的预测能力，计算它的IC表现。

计算IC： 计算月底因子值与下个月的收益率的秩相关性系数。

统计IC： 统计出因子IC的平均值、标准差、IC_IR、大于0的比例以及下月IC的相关系数

对传统流动性因子LIQ进行多空对冲测试，具体操作为：

因为在计算因子时，已经剔除了次新股、ST股、停牌个股等禁止池的股票，因此在对空测试时，样本空间为剔除禁止池以外的全体A股。

调仓频率为月度， 每个月底根据因子值从大到小将因子等分为5组。因为传统流动性因子为负向因子，因此做多因子值最小的一组，做空因子值最大的一组，每组等权配置获得多空组合。

多空对冲测试为简易测试，不考虑交易时的涨跌停情况和交易费用。

下文的IC统计和多空对冲测试操作都是类似的。

为了对比研报中的结果，在分析因子效果部分，我们选定2013-04-26至2017-10-31这个时间段作为测试区间，与研报设置一致。

基于回测结果可知， 传统流动性因子本身就是一个较好的负向因子，其IC值达到0.12， 已经有不错的预测能力。因子分析结果作为改进因子的比较基础。


'''

anls_start_date = '2013-04-26'
anls_end_date = '2017-10-31'

anls_liq_df = liq_df[(liq_df['tradeDate'] >= anls_start_date) & (liq_df['tradeDate'] <= anls_end_date)]
anls_pureliq_df = pureliq_df[(pureliq_df['tradeDate'] >= anls_start_date) & (pureliq_df['tradeDate'] <= anls_end_date)]

start_time = time.time()
print ("该部分统计传统流动因子的IC表现和多空对冲表现...")

# 获得月收益率
month_return = DataAPI.MktEqumGet(secID=a_universe, beginDate=start_date, endDate=end_date, field=u"ticker,endDate,chgPct,",pandas="1")
month_return.rename(columns={'endDate': 'tradeDate', 'chgPct': 'month_return'}, inplace=True)
month_return.sort_values(['ticker', 'tradeDate'], inplace=True)
month_return['next_month_return'] = month_return.groupby('ticker')['month_return'].shift(-1)
month_return.fillna(0, inplace=True)

# 计算IC
liq_ic_df = calc_ic(anls_liq_df, month_return, factor_name='liq', ret_name='next_month_return')
liq_ic_table = ic_describe(liq_ic_df)
print ("---传统流动性因子LIQ的IC统计表---")
print (liq_ic_table.to_html())

# 多空对冲表现
liq_ls_perf = long_short_backtest(anls_liq_df, month_return, factor_name='liq', return_name='next_month_return', direction=-1)
liq_ls_period = liq_ls_perf[['tradeDate', 'period_ret']].iloc[1:].set_index('tradeDate')
liq_ls_period.rename(columns={'period_ret': 'liq'}, inplace=True)
liq_perf_table = perf_describe(liq_ls_period)
print ("---传统流动性因子LIQ的多空对冲表现---")
print (liq_perf_table.to_html())
# 多空曲线
fig = plt.figure(figsize=(10,4))
fig.set_tight_layout(True)
ax = fig.add_subplot(111)
ax.grid()
ax.plot(pd.to_datetime(liq_ls_perf['tradeDate']), liq_ls_perf[['cum_ret']])
ax.set_ylim(0.5,5)
ax.set_ylabel(u"净值",fontproperties=font, fontsize=16)
ax.set_title(u"传统流动性因子LIQ多空净值曲线", fontproperties=font, fontsize=16)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))
'''
2.2 局部流动性因子pureLIQ的效果分析

考察局部流动性因子的预测能力，计算它的IC表现。

'''
start_time = time.time()
print ("该部分统计局部流动因子的IC表现...")

# 计算IC
pureliq_ic_df = pd.DataFrame(index=anls_pureliq_df['tradeDate'].unique())
for i in np.arange(5).astype(str):
    tmp_ic_df = calc_ic(anls_pureliq_df, month_return, factor_name='pure_liq'+i, ret_name='next_month_return', method='spearman')
    pureliq_ic_df = pd.concat([pureliq_ic_df, tmp_ic_df], axis=1, join='inner')

pureliq_ic_table = ic_describe(pureliq_ic_df)
print ("---局部流动性因子pureLIQ的IC统计表---")
print (pureliq_ic_table.to_html())
fig = plt.figure(figsize=(20,20))
fig.set_tight_layout(True)
ind = np.arange(len(pureliq_ic_df))
width = 0.2
for i in range(5):
    ax = fig.add_subplot(5,1,i+1)   
    ax.bar(ind, pureliq_ic_df['pure_liq'+str(i)], width, color='r')
    ax.set_xlim((0, ind[-1]+1))
    ax.set_ylim((-0.25, 0.25))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(pureliq_ic_df.index, rotation=90)
    ax.set_ylabel(u"IC", fontproperties=font, fontsize=16)
    ax.set_title(u"pureLIQ%d的IC序列"%i, fontproperties=font, fontsize=16)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

'''
从上述结果来看， 隔夜和14:00-15:00这两个时段的局部换手率因子（pureLIQ0, pureLIQ4）的IC均值小于-0.02，T统计量小于-2， IC大于0的占比小于50%，年化收益率小于0，因此有负向收益贡献；10：30-11：30和13:00-14:00这两个时段的局部换手率因子（pureLIQ2, pureLIQ3）的IC均值大于0.02，T统计量大于2， IC大于0的占比小于50%，因此有正向收益贡献。9:30-10:30时段的局部换手率因子（pureLIQ1）的IC均值接近于0，T统计量小于-2，且IC均值方向为负，但IC大于0的占比大于50%，方向矛盾，因此认为没有收益贡献。

综合IC各种统计结果，pureLIQ2的IC均值绝对值最大，且IC标准差最小，IC大于0的占比高达77.8%，因此pureLIQ的因子预测效果最好，且效果稳定。其次是pureLIQ3和pureLIQ4，pureLIQ4的IC均值更高，预测效果更好， 但是pureLIQ3的IC标准差更小，预测效果更稳定，因此它们的ICIR相近。最后是pureLIQ0，IC均值和puerLIQ3相近，但是稳定性不如pureLIQ3。

观察局部流动性因子的IC序列，考察因子的IC值是否受个别极值影响，从图上也可以看出， pureLIQ2的效果最佳，基本符合上述结论。


进一步,对局部流动性因子pureLIQ进行多空对冲测试， 在此处多空对冲操作中， 做空因子值最小的一组，做多因子值最大的一组，观察局部流动因子的方向。

'''
start_time = time.time()
print ("该部分统计局部流动因子的多空对冲表现...")

# 多空对冲表现
pureliq_ls_period = pd.DataFrame(anls_pureliq_df['tradeDate'].unique(), columns=['tradeDate'])
pureliq_ls_perf = pd.DataFrame(anls_pureliq_df['tradeDate'].unique(), columns=['tradeDate'])
for i in np.arange(5).astype(str):
    tmp_ls_perf = long_short_backtest(anls_pureliq_df, month_return, factor_name='pure_liq'+i, return_name='next_month_return', direction=1)
    pureliq_ls_period = pureliq_ls_period.merge(tmp_ls_perf[['tradeDate', 'period_ret']], on=['tradeDate'])
    pureliq_ls_period.rename(columns={'period_ret': 'pure_liq'+i}, inplace=True)
    pureliq_ls_perf = pureliq_ls_perf.merge(tmp_ls_perf[['tradeDate', 'cum_ret']], on=['tradeDate'])
    pureliq_ls_perf.rename(columns={'cum_ret': 'pure_liq'+i}, inplace=True)
    
pureliq_ls_period = pureliq_ls_period.iloc[1:].set_index('tradeDate')
pureliq_perf_table = perf_describe(pureliq_ls_period)
print ("---局部流动性因子pureLIQ的多空对冲表现---")
print (pureliq_perf_table.to_html())
# 多空曲线
fig = plt.figure(figsize=(10,4))
fig.set_tight_layout(True)
ax = fig.add_subplot(111)
ax.grid()
for i in np.arange(5).astype(str):
    ax.plot(pd.to_datetime(pureliq_ls_perf['tradeDate']), pureliq_ls_perf[['pure_liq'+i]], label='pure_liq'+i)
ax.legend(loc=0)
ax.set_ylabel(u"净值",fontproperties=font, fontsize=16)
ax.set_title(u"局部流动性因子pureLIQ多空净值曲线", fontproperties=font, fontsize=16)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


'''
从多空对冲曲线来看,pureLIQ0、pureLIQ4的负向因子，pureLIQ2、pureLIQ3是正向因子，这与IC统计结果一致。pureLIQ1在多空回测中收益为正，与IC均值方向矛盾。

从对空对冲的表现来看，pureLIQ4的夏普率最高，其次是pureLIQ2。

   调试 运行
文档
 代码  策略  文档
进一步，为了验证局部流动性因子pureLIQ的增益部分是否来自于已知的风格因子或行业因子，我们将pureLIQ关于10个风格因子(Beta，残差波动率，动量，市值，非线性市值，盈利能力，净市率，成长，杠杆，流动性)和申万一级行业因子做中性化处理，得到中性化后的残差。对残差进行IC统计和多空对冲测试
'''

start_time = time.time()
print ("该部分将局部流动性因子关于风格因子和行业因子中性化后， 进行IC统计和对空对冲测试...")

pureliq_wsn_df = anls_pureliq_df.copy()
# winsorize, standardize, neturalize
for i in np.arange(5).astype(str):
    print ("pureLIQ%s处理中..."%i)
    pureliq_wsn_df['pure_liq'+i] = pureliq_wsn_df.groupby('tradeDate').apply(winsorize_standardize, 'pure_liq'+i).values
    pureliq_wsn_df['pure_liq'+i] = pureliq_wsn_df.groupby('tradeDate').apply(lambda x: pd.Series(neutralize(x[['ticker', 'pure_liq'+i]].set_index('ticker')['pure_liq'+i].to_dict(), target_date=x['tradeDate'].unique()[0]))).values

# 计算IC
pureliq_wsn_ic_df = pd.DataFrame(index=pureliq_wsn_df['tradeDate'].unique())
for i in np.arange(5).astype(str):
    tmp_ic_df = calc_ic(pureliq_wsn_df, month_return, factor_name='pure_liq'+i, ret_name='next_month_return', method='spearman')
    pureliq_wsn_ic_df = pd.concat([pureliq_wsn_ic_df, tmp_ic_df], axis=1)
    
pureliq_wsn_ic_table = ic_describe(pureliq_wsn_ic_df)
print ("---剔除风格行业因子后的局部流动性因子的IC统计表---")
print (pureliq_wsn_ic_table.to_html())

pureliq_wsn_ls_period = pd.DataFrame(pureliq_wsn_df['tradeDate'].unique(), columns=['tradeDate'])
pureliq_wsn_ls_perf = pd.DataFrame(pureliq_wsn_df['tradeDate'].unique(), columns=['tradeDate'])
for i in np.arange(5).astype(str):
    tmp_ls_perf = long_short_backtest(pureliq_wsn_df, month_return, factor_name='pure_liq'+i, return_name='next_month_return', direction=1)
    pureliq_wsn_ls_period = pureliq_wsn_ls_period.merge(tmp_ls_perf[['tradeDate', 'period_ret']], on=['tradeDate'])
    pureliq_wsn_ls_period.rename(columns={'period_ret': 'pure_liq'+i}, inplace=True)
    pureliq_wsn_ls_perf = pureliq_wsn_ls_perf.merge(tmp_ls_perf[['tradeDate', 'cum_ret']], on=['tradeDate'])
    pureliq_wsn_ls_perf.rename(columns={'cum_ret': 'pure_liq'+i}, inplace=True)
    
pureliq_wsn_ls_period = pureliq_wsn_ls_period.iloc[1:].set_index('tradeDate')
pureliq_wsn_perf_table = perf_describe(pureliq_wsn_ls_period)
print ("---剔除风格行业因子后的局部流动性因子的多空对冲表现---")
print (pureliq_wsn_perf_table.to_html())
# 多空曲线
fig = plt.figure(figsize=(10,4))
fig.set_tight_layout(True)
ax = fig.add_subplot(111)
ax.grid()
for i in np.arange(5).astype(str):
    ax.plot(pd.to_datetime(pureliq_wsn_ls_perf['tradeDate']), pureliq_wsn_ls_perf[['pure_liq'+i]], label='pure_liq'+i)
ax.legend(loc=0)
ax.set_ylabel(u"净值",fontproperties=font, fontsize=16)
ax.set_title(u"剔除风格行业因子后的局部流动性因子多空净值曲线", fontproperties=font, fontsize=16)


end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


'''
剔除了已知风格因子和行业因子后，pureLIQ2的IC值仍大于0.02，多空对冲表现依然不错。pureLIQ3、pureLIQ4的IC值都下降到0.02以下。pureLIQ0表现提高。以上说明，局部流动性因子还是存在额外的选股收益的。

综合上述所有局部流动性因子的绩效分析，从统计角度来看，在回测区间内，局部流动性因子存在额外的选股能力。特别地，10:30-11:30时段的局部流动性因子pureLIQ2的表现最好且表现稳定。

以上分析结果，与研报的结果有差异，其原因可能是1）因子计算对停牌等特殊情况的处理不同；2）多空对冲回测细节处理不一致；3）剔除已知风格因子部分，已知风格因子选择不一致。

第三部分：因子合成及其绩效表现
该部分耗时 10分钟 （主要分组回测用时8分钟左右）
该部分内容为利用不同模型对因子进行合成, 具体包括：

3.1 根据历史回测区间局部流动性因子的结果，筛选出两个较强的局部流动性因子，合并到传统流动性因子上，得到一个改进后的流动性因子MixLIQ.

3.2 对MixLiQ的绩效表现进行评估，并与LIQ对比。绩效评估包括：IC值统计、行业内选股能力、市值衰减特性检验、分组多头回测表现、不同样本空间内的表现。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

3.1 改进后的流动性因子MixLIQ构造

为了不引入未来数据，本文根据历史回测区间的局部流动性因子结果，筛选出两个表现较强的因子，与传统流动性因子进行合并。具体做法如下：

合成因子构造区间为2013-01-01至2017-12-31.

每年年初，对过去4年时间的局部流动性因子进行效果测试，选出回测区间内ICIR最大的两个局部流动性因子，考虑方向，与传统流动性因子进行等权合并。例如，2013-01-01，回测2009-01-01至2012-12-31时段内5个局部流动性因子的IC，ICIR绝对值最大的2个局部流动性因子是pureLIQ2，pureLIQ0，方向分别是正向因子、负向因子，因此最后LIQ、pureLIQ2、pureLIQ0， 分别配置1，-1，1的权重合成，得到MixLIQ.

本文是为了考察加入局部流动性因子，是否对传统流动性因子有加成效果，因此采取等权的加权方式。但是实际上，LIQ的因子效果水平远高于pureLIQ，因此采取IC加权，ICIR加权等方式将pureLIQ加入LIQ,MixLIQ的因子效果也许会更好。


'''
start_time = time.time()
print ("该部分统计改进后的流动性因子MixLIQ...")

mix_start_date = '2013-01-01'
mix_end_date = '2017-12-31'

all_liq_df = liq_df.merge(pureliq_df, on=['ticker', 'tradeDate'])
mixliq_df = pd.DataFrame()

for year in range(int(mix_start_date[:4]), int(mix_end_date[:4])+1):
    back_start_date = str(year - 4) + '-01-01'
    back_end_date = str(year - 1) + '-12-31'
    print ("回测%s-%s" % (back_start_date, back_end_date))
    
    # 历史数据计算IC_IR
    bt_data = all_liq_df[(all_liq_df['tradeDate'] >= back_start_date) & (all_liq_df['tradeDate'] <= back_end_date)]
    pureliq_ic_df = pd.DataFrame(index=bt_data['tradeDate'].unique())
    for i in np.arange(5).astype(str):
        pureliq_ic_df['pure_liq'+i] = calc_ic(bt_data, month_return, factor_name='pure_liq'+i, ret_name='next_month_return', method='spearman')['pure_liq'+i]

    # 筛选出IC_IR绝对值最大的两个局部流动性因子
    icir_table = (pureliq_ic_df.mean()/pureliq_ic_df.std())
    abs_icir_table = abs(icir_table).sort_values(ascending=False)
    select_pureliq = abs_icir_table.index[:2].values.tolist()
    pure_liq_weight = (np.sign(icir_table[abs_icir_table.index[:2]]).values * (-1)).tolist()
    print ("%d年筛选出的因子为%s, 方向为%s" % (year, select_pureliq, pure_liq_weight))

    # 开始合并
    all_liq = ['liq'] + select_pureliq
    all_liq_weight = [1] + pure_liq_weight
    year_df = all_liq_df[(all_liq_df['tradeDate'] >= str(year)+'-01-01') & (all_liq_df['tradeDate'] <= str(year)+'-12-31')]
    for fname in all_liq:
        year_df[fname] = year_df.groupby('tradeDate')[fname].apply(lambda x: (x-x.mean())/x.std())
    year_df['mix_liq'] = (year_df[all_liq]*all_liq_weight).sum(axis=1)
    year_df = year_df[['ticker', 'tradeDate', 'liq', 'mix_liq']]

    mixliq_df = mixliq_df.append(year_df)
    
print (mixliq_df.head().to_html()  )
mixliq_df.to_csv('raw_data/mixliq_df.csv', index=False)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

mixliq_df = pd.read_csv('raw_data/mixliq_df.csv', dtype={'ticker': str})

'''
3.2 改进后的流动性因子MixLIQ和传统流动性因子LIQ的绩效比较

该部分对MixLIQ和LIQ进行比较，绩效评估包括：IC值统计、行业内选股能力、市值衰减特性检验、分组多头回测表现、不同样本空间内的表现。

首先进行IC统计比较。

'''
start_time = time.time()
print ("该部分比较MixLIQ和LIQ的IC表现...")

mixliq_ic_df = pd.DataFrame(index=mixliq_df['tradeDate'].unique())
liq_ic_df = calc_ic(mixliq_df, month_return, factor_name='liq', ret_name='next_month_return', method='spearman')
ml_ic_df = calc_ic(mixliq_df, month_return, factor_name='mix_liq', ret_name='next_month_return', method='spearman')
mixliq_ic_df = pd.concat([mixliq_ic_df, liq_ic_df, ml_ic_df], axis=1)
mixliq_ic_df.dropna(inplace=True)
mixliq_ic_table = ic_describe(mixliq_ic_df)

print ("---改进因子MixLIQ和传统流动性因子LIQ的IC统计表---")
print (mixliq_ic_table.to_html())

fig = plt.figure(figsize=(15,8))
fig.set_tight_layout(True)
ax = fig.add_subplot(111)
ind = np.arange(len(mixliq_ic_df))
width=0.2
rect1 = ax.bar(ind, mixliq_ic_df['liq'], width, color='r')
rect2 = ax.bar(ind+width, mixliq_ic_df['mix_liq'], width, color='y')
ax.set_xlim((0, ind[-1]+1))
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(mixliq_ic_df.index, rotation=90)
ax.set_ylabel(u"IC", fontproperties=font, fontsize=16)
ax.set_title(u"MixLIQ和LIQ的IC序列对比", fontproperties=font, fontsize=16)
ax.legend((rect1[0], rect2[0]), ('LIQ', 'MixLIQ'), fontsize=16)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


'''
从IC统计结果来看，MixLIQ的IC均值比LIQ低一些，从IC序列来看，整体上，MixLIQ的IC值也比LIQ的IC值小。MixLIQ的IC标准差将近是LIQ的一半，IC大于0的占比也是LIQ的一半，说明MixLIQ的预测稳定性提升了一倍。因此，MixLIQ的ICIR为-1.29，相对与LIQ，提升很多。其次，MixLIQ的下月IC相关系数下降到了33.17%，说明相对与LIQ，MixLIQ的换手率会有一定程度的提高。

局部流动性因子对传统流动因子的效果提升在于提升了因子的预测稳定性。

MixLIQ的优点是预测稳定性提升一倍，缺点是提高了因子换手率。

   调试 运行
文档
 代码  策略  文档
下面，对MixLIQ和LIQ的行业内选股能力进行比较，计算了两个因子在28个申万一级行业内的IC均值。

'''
start_time = time.time()
print ("该部分比较MixLIQ和LIQ的行业内选股能力...")

# 申万一级行业分类
indu_group = DataAPI.EquIndustryGet(industryVersionCD=u"010303", secID=a_universe, field=u"ticker,industryName1,outDate", pandas="1")
indu_group = indu_group[pd.isnull(indu_group.outDate)]
mixliq_indu_df = mixliq_df.merge(indu_group[['ticker', 'industryName1']], on=['ticker'])

# 计算行业内IC均值
liq_indu_ic = mixliq_indu_df.groupby('industryName1').apply(lambda x: calc_ic(x, month_return, 'liq', 'next_month_return')['liq'].mean())
liq_indu_ic = pd.DataFrame(liq_indu_ic, columns=['liq'])
mixliq_indu_ic = mixliq_indu_df.groupby('industryName1').apply(lambda x: calc_ic(x, month_return, 'mix_liq', 'next_month_return')['mix_liq'].mean())
mixliq_indu_ic = pd.DataFrame(mixliq_indu_ic, columns=['mix_liq'])
indu_ic = pd.concat([liq_indu_ic, mixliq_indu_ic], axis=1)
indu_ic.sort_values('mix_liq', inplace=True)

fig = plt.figure(figsize=(20,8))
fig.set_tight_layout(True)
ax = fig.add_subplot(111)
ind = np.arange(len(indu_ic))
width=0.2
rect1 = ax.bar(ind, indu_ic['liq'], width, color='r')
rect2 = ax.bar(ind+width, indu_ic['mix_liq'], width, color='y')
ax.set_xlim((0, ind[-1]+1))
ax.set_ylim((-0.2,0.01))
ax.xaxis.tick_top()
ax.set_xticks(ind + width / 2)
ax.set_xticklabels([i.decode('utf8') for i in indu_ic.index], rotation=70, fontproperties=font)
ax.set_ylabel(u"IC", fontproperties=font, fontsize=16)
ax.set_title(u"MixLIQ和LIQ的行业内IC均值对比", fontproperties=font, fontsize=16, y=1.15)
ax.legend((rect1[0], rect2[0]), ('LIQ', 'MixLIQ'), fontsize=16, loc=0)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


'''
从上图结果看出，MixLIQ在各个行业内都有较好的选股能力，其中，在农林牧渔行业内最强，IC均值达到-0.14， 最差的银行行业，IC均值也有将近-0.04。

从上图来看，整体上，LIQ在行业内的IC值更高，但是MixLIQ在不同行业内的选股效果更稳定。LIQ在银行行业基本没有选股能力，但是MixLIQ在银行行业的IC均值有-0.04左右。因此MixLIQ的行业普适性更好。

   调试 运行
文档
 代码  策略  文档
下面检验MixLIQ的市值衰减性。传统流动性因子具有较强的实质衰减性，其在大市值的个股中的表现较差，上文中LIQ在银行行业的IC表现，也印证了这一点。

将个股按市值等分成十组，市值越大，组别越大，分别计算MixLIQ和LIQ在各组内的IC值和ICIR值。

'''

start_time = time.time()
print ("该部分比较MixLIQ和LIQ的市值衰减特性...")

# 按流通市值将股票分成10组，组数越大，流通市值越大
mixliq_cap_df = mixliq_df.merge(dmkt_df[['ticker', 'tradeDate', 'negMarketValue']], on=['ticker', 'tradeDate'], how='left')
mixliq_cap_df = signal_grouping(mixliq_cap_df, factor_name='negMarketValue', ngrp=10)

# 各组计算IC
liq_cap_ic = mixliq_cap_df.groupby('group').apply(lambda x: calc_ic(x, month_return, 'liq', 'next_month_return')['liq'].mean())
mixliq_cap_ic = mixliq_cap_df.groupby('group').apply(lambda x: calc_ic(x, month_return, 'mix_liq', 'next_month_return')['mix_liq'].mean())
cap_ic = pd.concat([liq_cap_ic, mixliq_cap_ic], axis=1)
cap_ic.columns=['liq', 'mix_liq']

# 各组计算ICIR
liq_cap_ic_std = mixliq_cap_df.groupby('group').apply(lambda x: calc_ic(x, month_return, 'liq', 'next_month_return')['liq'].std())
mixliq_cap_ic_std = mixliq_cap_df.groupby('group').apply(lambda x: calc_ic(x, month_return, 'mix_liq', 'next_month_return')['mix_liq'].std())
liq_cap_ic_ir = liq_cap_ic / liq_cap_ic_std
mixliq_cap_ic_ir = mixliq_cap_ic / mixliq_cap_ic_std
cap_ic_ir = pd.concat([liq_cap_ic_ir, mixliq_cap_ic_ir], axis=1)
cap_ic_ir.columns=['liq', 'mix_liq']

# 做图
fig = plt.figure(figsize=(10,10))
fig.set_tight_layout(True)
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ind = np.arange(len(cap_ic))
width=0.2

rect1 = ax.bar(ind+width, cap_ic['liq'], width, color='r')
rect2 = ax.bar(ind+width*2, cap_ic['mix_liq'], width, color='y')
ax.set_xlim((0, ind[-1]+1))
ax.set_ylim((-0.16,0))
ax.xaxis.tick_top()
ax.set_xticks(ind + width*2)
ax.set_xticklabels([('第%d组'%(i+1)).decode('utf8') for i in cap_ic.index], rotation=70, fontproperties=font)
ax.set_ylabel(u"IC", fontproperties=font, fontsize=16)
ax.set_title(u"MixLIQ和LIQ在市值分组下的IC对比", fontproperties=font, fontsize=16, y=1.13)
ax.legend((rect1[0], rect2[0]), ('LIQ', 'MixLIQ'), fontsize=16, loc=0)

rect3 = ax2.bar(ind+width, cap_ic_ir['liq'], width, color='r')
rect4 = ax2.bar(ind+width*2, cap_ic_ir['mix_liq'], width, color='y')
ax2.set_xlim((0, ind[-1]+1))
ax2.set_ylim((-1.5,0))
ax2.xaxis.tick_top()
ax2.set_xticks(ind + width*2)
ax2.set_xticklabels([('第%d组'%(i+1)).decode('utf8') for i in cap_ic_ir.index], rotation=70, fontproperties=font)
ax2.set_ylabel(u"ICIR", fontproperties=font, fontsize=16)
ax2.set_title(u"MixLIQ和LIQ在市值分组下的ICIR对比", fontproperties=font, fontsize=16, y=1.13)
ax2.legend((rect3[0], rect4[0]), ('LIQ', 'MixLIQ'), fontsize=16, loc=0)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

'''

从IC对比结果来看，MixLIQ仍然具有一定的市值衰减性，但各组的IC均值相对稳定很多，相比LIQ有缓解。

从ICIR对比结果看，MixLIQ全面提高各组的ICIR值，各组ICIR值均达到0.6以上。但是MixLIQ仍然存在市值衰减性。

   调试 运行
文档
 代码  策略  文档
下面比较MixLIQ和LIQ的多空对冲表现，并分年度展示结果

'''
start_time = time.time()
print ("该部分比较MixLIQ和LIQ的多空对冲表现...")

# MixLIQ的多空表现
mixliq_ls_perf = long_short_backtest(mixliq_df, month_return, factor_name='mix_liq', return_name='next_month_return', direction=-1)
mixliq_perf_period = mixliq_ls_perf[['tradeDate', 'period_ret']].iloc[1:].set_index('tradeDate')
mixliq_perf_period.rename(columns={'period_ret': 'mix_liq'}, inplace=True)

# LIQ的多空表现
liq_ls_perf = long_short_backtest(mixliq_df, month_return, factor_name='liq', return_name='next_month_return', direction=-1)
liq_ls_period = liq_ls_perf[['tradeDate', 'period_ret']].iloc[1:].set_index('tradeDate')
liq_ls_period.rename(columns={'period_ret': 'liq'}, inplace=True)

#结合LIQ的多空表现
ls_period = liq_ls_period.merge(mixliq_perf_period, left_index=True, right_index=True)
ls_perf = liq_ls_perf[['tradeDate', 'cum_ret']].merge(mixliq_ls_perf[['tradeDate', 'cum_ret']], on=['tradeDate'])
ls_perf.columns=['tradeDate', 'liq', 'mix_liq']

# 分年度统计
for year in range(2013, 2018):
    print ("---%d年多空表现对比---"%year)
    print (perf_describe(ls_period[[str(year) in date for date in ls_period.index.values]]).to_html())
    
liq_perf_table = perf_describe(ls_period)
print ("---MixLIQ和LIQ的全历史多空对冲表现对比---")
print (liq_perf_table.to_html())

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

'''

观察分年度多空对冲表现，整体上，MixLIQ的年化收益率均低于LIQ,年化波动率也低于LIQ，夏普比率高于LIQ。2014年和2016年，MixLIQ的收益率比LIQ低很多，以至于夏普比率也低于LIQ。但是MixLIQ的月度胜率均高于LIQ, 2017年月度胜率高达100%。

从全历史的多空对冲表现来看，同样得出的结论是，MixLIQ相对于LIQ，降低了风险，提高了风险收益比。

   调试 运行
文档
 代码  策略  文档
下面检验MixLIQ的实际收益能力和因子单调性。将股票根据因子值等分成5组，每组组内等权配置，获得5条纯多头净值曲线。以全A指数作为基准，检验因子的收益来源于多头还是多头。

'''

start_time = time.time()
print ("该部分考察MixLIQ的实际收益水平...")

#--------------- 回测参数 ---------------
start = mix_start_date                     # 回测起始时间
end = mix_end_date                       # 回测结束时间
benchmark = '000002.ZICN'                        # 策略参考标准
universe = DynamicUniverse('A')           # 证券池，支持股票和基金
capital_base = 10000000                     # 起始资金
freq = 'd'                              
refresh_rate = Monthly(1)  

# ---------------回测参数部分结束----------------

# 读取因子数据
factor_data = mixliq_df[[u'ticker', u'tradeDate', u'mix_liq']]
factor_data['ticker'] = factor_data['ticker'].apply(ticker2secID)
factor_data.rename(columns={'mix_liq': 'factor'}, inplace=True)
factor_data = factor_data.set_index('tradeDate', drop=True)
q_dates = factor_data.index.values

accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
}

# 把回测参数封装到 SimulationParameters 中，供 quick_backtest 使用
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate, accounts=accounts)
# 获取回测行情数据
data = quartz.get_backtest_data(sim_params)
# 运行结果
results = {}

# # 将因子划分为5分位，并进行快速回测
for quantile_five in range(1, 6):
    
    # ---------------策略逻辑部分----------------
    
    def initialize(context):                   # 初始化虚拟账户状态
        pass

    def handle_data(context): 
        account = context.get_account('fantasy_account')
        current_universe = context.get_universe('stock', exclude_halt=True)
        pre_date = context.previous_date.strftime("%Y-%m-%d")
        if pre_date not in q_dates:            
            return

        # 拿取调仓日前一个交易日的因子，并按照相应分位选择股票
        q = factor_data.ix[pre_date]
        q = q.set_index('ticker', drop=True)
        q = q.ix[current_universe]
        q = q.dropna()
        
        q_min = q['factor'].quantile((quantile_five-1)*0.2)
        q_max = q['factor'].quantile(quantile_five*0.2)
        my_univ = q[(q['factor']>=q_min) & (q['factor']<q_max)].index.values

       # 交易部分
        positions = account.get_positions()
        sell_list = [stk for stk in positions if stk not in my_univ]
        for stk in sell_list:
            account.order_to(stk,0)
        
        # 在目标股票池中的，等权买入
        for stk in my_univ:
            account.order_pct_to(stk, 1.0/len(my_univ))


    # 生成策略对象
    strategy = quartz.TradingStrategy(initialize, handle_data)
    # ---------------策略定义结束----------------
    
    # 开始回测
    bt, perf = quartz.quick_backtest(sim_params, strategy, data=data)

    # 保存运行结果，1为因子最小组，5为因子最大组
    results[quantile_five] = {'max_drawdown': perf['max_drawdown'], 'sharpe': perf['sharpe'], 'alpha': perf['alpha'], 'beta': perf['beta'], 'information_ratio': perf['information_ratio'], 'annualized_return': perf['annualized_return'], 'bt': bt}    

    print (str(quantile_five))
print ('done')

# 画图展示回测结果
fig = plt.figure(figsize=(10,8))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.grid()
ax2.grid()

for qt in results:
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
    ax1.plot(data['tradeDate'], data[['portfolio']], label=str(qt))
    ax2.plot(data['tradeDate'], data[['excess']], label=str(qt))
      
ax1.legend(loc=0)
ax2.legend(loc=0)
ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲净值", fontproperties=font, fontsize=16)
ax1.set_title(u"因子不同五分位数分组选股净值走势", fontproperties=font, fontsize=16)
ax2.set_title(u"因子不同五分位数分组选股对冲全A指数后净值走势", fontproperties=font, fontsize=16)

# results 转换为 DataFrame
results_pd = pd.DataFrame(results).T.sort_index()

results_pd = results_pd[[u'alpha', u'beta', u'information_ratio', u'sharpe', u'annualized_return', u'max_drawdown',  
                         u'hedged_annualized_return', u'hedged_max_drawdown', u'hedged_volatility']]

cols = [(u'风险指标', u'Alpha'), (u'风险指标', u'Beta'), (u'风险指标', u'信息比率'), (u'风险指标', u'夏普比率'), (u'纯股票多头时', u'年化收益'),
        (u'纯股票多头时', u'最大回撤'), (u'对冲后', u'年化收益'), (u'对冲后', u'最大回撤'), (u'对冲后', u'收益波动率')]
results_pd.columns = pd.MultiIndex.from_tuples(cols)
results_pd.index.name = u'五分位组别'
print ("---MixLIQ等分5组多头表现---")
print (results_pd.to_html())

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

'''
从分组回测结果来看，MixLIQ具有较好的因子单调性。第1组的年化收益率远高于第5组，超额收益率来看，第1组表现优异，说明MixLIQ具有良好的收益水平。
   调试 运行
文档
 代码  策略  文档
下面比较MixLIQ和LIQ在不同样本空间内的表现。将两个因子在全A、中证500、沪深300三个样本空间内做多空对冲测试，比较它们的多空表现。
'''
start_time = time.time()
print ("该部分统计MixLIQ在不同样本空间内的表现...")

# 沪深300成份股
hs300 = DataAPI.mIdxCloseWeightGet(ticker=u"000300",beginDate=start_date,endDate=end_date,field=u"consID,effDate",pandas="1")
hs300.rename(columns={'consID': 'ticker', 'effDate': 'tradeDate'}, inplace=True)
hs300['ticker'] = hs300['ticker'].apply(secID2ticker)
hs300['hs300_flag'] = 1

# 中证500成份股
zz500 = DataAPI.mIdxCloseWeightGet(ticker=u"000905",beginDate=start_date,endDate=end_date,field=u"consID,effDate",pandas="1")
zz500.rename(columns={'consID': 'ticker', 'effDate': 'tradeDate'}, inplace=True)
zz500['ticker'] = zz500['ticker'].apply(secID2ticker)
zz500['zz500_flag'] = 1

mixliq_index_df = mixliq_df.merge(hs300, on=['ticker', 'tradeDate'], how='left').merge(zz500, on=['ticker', 'tradeDate'], how='left')

# 不同样本空间的多空测试
mixliq_hs_perf = long_short_backtest(mixliq_index_df[-pd.isnull(mixliq_index_df['hs300_flag'])], month_return, factor_name='mix_liq', return_name='next_month_return', direction=-1)
mixliq_zz_perf = long_short_backtest(mixliq_index_df[-pd.isnull(mixliq_index_df['zz500_flag'])], month_return, factor_name='mix_liq', return_name='next_month_return', direction=-1)

mixliq_diff_sample_period = mixliq_ls_perf[['tradeDate', 'period_ret']].merge(mixliq_zz_perf[['tradeDate', 'period_ret']], on='tradeDate').merge(mixliq_hs_perf[['tradeDate', 'period_ret']], on='tradeDate')
mixliq_diff_sample_period.columns=['tradeDate', 'all', 'zz500', 'hs300']

mixliq_diff_sample_perf = mixliq_ls_perf[['tradeDate', 'cum_ret']].merge(mixliq_zz_perf[['tradeDate', 'cum_ret']], on='tradeDate').merge(mixliq_hs_perf[['tradeDate', 'cum_ret']], on='tradeDate')
mixliq_diff_sample_perf.columns=['tradeDate', 'all', 'zz500', 'hs300']

mixliq_diff_sample_period = mixliq_diff_sample_period.iloc[1:].set_index('tradeDate')
mixliq_perf_table = perf_describe(mixliq_diff_sample_period)
print ("---MixLIQ在不同样本空间的多空对冲表现---")
print (mixliq_perf_table.to_html())

# 多空曲线
fig = plt.figure(figsize=(10,4))
fig.set_tight_layout(True)
ax = fig.add_subplot(111)
ax.grid()
ax.plot(pd.to_datetime(mixliq_diff_sample_perf['tradeDate']), mixliq_diff_sample_perf[['all']], label=u'全样本内')
ax.plot(pd.to_datetime(mixliq_diff_sample_perf['tradeDate']), mixliq_diff_sample_perf[['zz500']], label=u'中证500内')
ax.plot(pd.to_datetime(mixliq_diff_sample_perf['tradeDate']), mixliq_diff_sample_perf[['hs300']], label=u'沪深300内')
ax.legend(loc=0, prop=font)
ax.set_ylabel(u"净值",fontproperties=font, fontsize=16)
ax.set_title(u"MixLIQ在不同样本空间的对空净值曲线", fontproperties=font, fontsize=16)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

start_time = time.time()
print ("该部分统计LIQ在不同样本空间内的表现...")

# 不同样本空间的多空测试
liq_hs_perf = long_short_backtest(mixliq_index_df[-pd.isnull(mixliq_index_df['hs300_flag'])], month_return, factor_name='liq', return_name='next_month_return', direction=-1)
liq_zz_perf = long_short_backtest(mixliq_index_df[-pd.isnull(mixliq_index_df['zz500_flag'])], month_return, factor_name='liq', return_name='next_month_return', direction=-1)

liq_diff_sample_period = liq_ls_perf[['tradeDate', 'period_ret']].merge(liq_zz_perf[['tradeDate', 'period_ret']], on='tradeDate').merge(liq_hs_perf[['tradeDate', 'period_ret']], on='tradeDate')
liq_diff_sample_period.columns=['tradeDate', 'all', 'zz500', 'hs300']

liq_diff_sample_perf = liq_ls_perf[['tradeDate', 'cum_ret']].merge(liq_zz_perf[['tradeDate', 'cum_ret']], on='tradeDate').merge(liq_hs_perf[['tradeDate', 'cum_ret']], on='tradeDate')
liq_diff_sample_perf.columns=['tradeDate', 'all', 'zz500', 'hs300']

liq_diff_sample_period = liq_diff_sample_period.iloc[1:].set_index('tradeDate')
liq_perf_table = perf_describe(liq_diff_sample_period)
print ("---LIQ在不同样本空间的多空对冲表现---")
print (liq_perf_table.to_html())

# 多空曲线
fig = plt.figure(figsize=(10,4))
fig.set_tight_layout(True)
ax = fig.add_subplot(111)
ax.grid()
ax.plot(pd.to_datetime(liq_diff_sample_perf['tradeDate']), liq_diff_sample_perf[['all']], label=u'全样本内')
ax.plot(pd.to_datetime(liq_diff_sample_perf['tradeDate']), liq_diff_sample_perf[['zz500']], label=u'中证500内')
ax.plot(pd.to_datetime(liq_diff_sample_perf['tradeDate']), liq_diff_sample_perf[['hs300']], label=u'沪深300内')
ax.legend(loc=0, prop=font)
ax.set_ylabel(u"净值",fontproperties=font, fontsize=16)
ax.set_title(u"LIQ在不同样本空间的对空净值曲线", fontproperties=font, fontsize=16)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

