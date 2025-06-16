# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 19:42:44 2020

@author: Asus

导读
A. 研究目的：本文基于优矿提供的因子数据构建股票的网络动量因子，文中部分方法参考长江证券《基于网络的动量选股策略》（原作者：覃川桃）中的研究方法，通过实证分析揭示基于网络的动量选股策略的应用价值。

B. 文章结构：本文共分为3个部分，具体如下

一、数据准备和处理：利用uqer API获取动量指标数据；对每个动量指标在时间序列上进行正态化；计算个股网络动量。

二、网络动量因子测试：利用uqer API获取股票周度收益率，结合因子数据对构建因子进行测试，包括计算IC、分组、多空收益等；探讨因子在市值上的暴露情况

三、中性化之后因子测试：考虑到因子在市值上的暴露情况， 对因子做中性化之后进行测试；

四、总结：对网络动量因子及相应的选股策略表现进行总结；

C. 研究结论：

网络动量选股因子是一个有效的alpha因子，因子周度IC为5.5%,分组单调性明显；网络动量因子在市值上暴露明显，对市值和行业做中性化后，因子表现仍然较好，周度IC为4.6%。

基于网络的动量选股策略具有较好的表现， 原始因子多空年化收益30.5%，ir为0.925，中性话之后因子多空年化收益18.9%，ir为0.932。

D. 时间说明

本文主要分为四个部分，第一部分约耗时12分钟，其它部分耗时均在2分钟以内，总耗时在20分钟以内
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
链接：https://uqer.datayes.com/community/share/OEzAS7PI0D08Zya9Z6nyb5l0O2w0/private；密码：9378。请前往查看并注意保密。
请在运行之前，克隆上面的代码，并存成lib(右上角->另存为lib，不要修改名字)

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

"""


import pandas as pd
#import lib.uqer_utl1 as quant_util
import lib.quant_util as quant_util
import pandas as pd, numpy as np
import matplotlib.lines as mlines
from quartz_extensions.SignalAnalysis.tears import analyse_return, analyse_monthly_return, analyse_IC, analyse_construction, analyse_general
import time
import numpy as np
from scipy import stats

'''
第一部分：数据准备和处理
该部分耗时 ** 12分钟**
该部分内容为：

通过API取出因子数据:利用uqer API获取动量指标数据;

对动量指标进行标准化: 对每个动量指标在时间序列上进行正态化；

计算个股网络动量；

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''

start_date = "2012-01-01"
end_date = "2019-08-23"
dates = pd.date_range(start_date,end_date,fred="D").astype(str)
calendar_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
week_end_list = calendar_df[calendar_df['isWeekEnd']==1]['calendarDate'].values
month_end_list = calendar_df[calendar_df['isMonthEnd']==1]['calendarDate'].values
trade_date_list = calendar_df[calendar_df['isOpen']==1]['calendarDate'].values

'''
1.1:通过API取出因子数据（5分钟）

1)MassIndex:梅斯线（Mass Index），本指标是Donald Dorsey累积股价波幅宽度之后所设计的震荡曲线。其最主要的作用，在于寻找飙涨股或者极度弱势股的重要趋势反转点。

2)KDJ_J:随机指标。它综合了动量观念、强弱指标及移动平均线的优点，用来度量股价脱离价格正常范围的变异程度

3)RSI:相对强弱指标（Relative Strength Index），通过比较一段时期内的平均收盘涨数和平均收盘跌数来分析市场买沽盘的意向和实力，据此预测趋势的持续或者转向。

4)CCI10:10日顺势指标（Commodity Channel Index），专门测量股价是否已超出常态分布范围。

5)CMO:钱德动量摆动指标（Chande Momentum Osciliator），与其他动量指标摆动指标如相对强弱指标（RSI）和随机指标（KDJ）不同，钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据

6)MFI:资金流量指标（Money Flow Index），该指标是通过反映股价变动的四个元素：上涨的天数、下跌的天数、成交量增加幅度、成交量减少幅度来研判量能的趋势，预测市场供求关系和买卖力道。

'''

# 取数据
factor_df_list  = []
factor_list = ['MassIndex','KDJ_J','RSI','CCI10','CMO','MFI']
for wnenddate in week_end_list:
    factor_dfi = DataAPI.MktStockFactorsOneDayGet(tradeDate=wnenddate,secID=u"",ticker=u"",field=['secID','ticker','tradeDate']+factor_list,pandas="1")
    factor_df_list.append(factor_dfi)
factor_df = pd.concat(factor_df_list,axis=0)
factor_df.head()

#设置数据目录
raw_data_dir = "./net_momentum_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)
factor_df.to_csv('%s/mum_factor_df4.csv'%raw_data_dir,index=False,encoding='utf-8')        


'''

1.2:对动量指标进行标准化

虽然上述的指标都能有效的衡量股票的动量，但是却存在着以下两个问题：
1） 股价对指标计算结果的影响：以RSI为例，高股价和低股价的股票指标值差异也比较明显，所以同一指标在不同股票之间是难以比较的。2） 不同指标的数值范围不同：
由于计算方法的不同，不同指标的数值范围也是不同的，比如KDJ的值在0到100之间，而CCI则是波动于正无穷大和负无穷大之间，这使得我们不能简单将这些指标放在一起比较。

动量指标标准化的方法：对各个指标进行时间序列标准化，假设市场有N个股票，我们可以通过对历史行情数据的计算得出每个股票每周收盘之后的MassIndex，KDJ_J，RSI，
CCI10，CMO，MFI 指标的值。我们将第i 个股票最近一周的动量指标表示如下：
图片注释
然后我们对每个指标进行正态化， 以RSI为例正态化公式为:
图片注释

其中μ为最近一年内RSI的平均值，σ为最近一年内RSI的标准差，以此类推我们可以得到正态化后的动量指标：
图片注释

'''
#对指标值进行时间序列标准化


#对指标值进行时间序列标准化
def cal_std_factor(df, step=52):
    factor_list = ['MassIndex','KDJ_J','RSI','CCI10','CMO','MFI']
    df = df.copy()
    df = df.sort_values('tradeDate')
    for f in factor_list:
        df[f] = (df[f] - df[f].rolling(step,min_periods=30).mean()) / df[f].rolling(step,min_periods=30).std()
    return df    

factor_df1 = factor_df.groupby(['secID','ticker'],as_index=False).apply(lambda x: cal_std_factor(x,52)).reset_index().drop(['level_0','level_1'],axis=1)    

factor_df1 = factor_df1.dropna()
factor_df1 = factor_df1[['secID','ticker','tradeDate','KDJ_J','RSI','CCI10','MFI','MassIndex','CMO']]
factor_df1[:2]

'''

1.3:计算个股网络动量(6分钟)

在得到股票标准化的动量的基础上， 对于N只股票，我们可以得出一个N 行的动量指标矩阵：
图片注释

我们可以将这个矩阵转化为一个网络，矩阵中的一行即为网络中的一个节点，接下来我们使用欧式距离公式计算每个节点与其他节点之间的距离。假设有两个n 维向量X=(x1, x2 … xn)，Y=(y1, y2 …yn)，X 与Y 的欧式距离为：
图片注释

通过计算所有节点之间的距离，我们得到一个N 行N 列的距离矩阵：
图片注释
接着我们计算距离矩阵中每一行的平均值，也就是每个节点与其他节点的平均距离，即：
图片注释

将所有节点与其他节点的平均距离从小到大排序，节点与其他节点的平均距离越小表示节点在网络中处在更中心的位置，反之则表示节点处在更边缘的位置。

从股票动量的角度来理解平均距离：我们知道网络中的每个节点代表的是一个股票正态化之后的动量指标，两个节点之间的距离就可以理解为两个股票动量的差异度，平均距离即为一个股票与网络中其他股票的平均差异度。那么如果一个股票与其他股票的平均差异度越小，就可以理解为这个股票相对于网络中其他的股票处在更中心的位置。根据我们最初的猜测，我们将最近一个月内每个股票的平均距离取平均值，并选出平均值最小的股票，我们认为这些股票的价格在过去一个月内以稳定的速度行进，未来股价会受到更多资金的驱动从而获得更高的收益。


'''

#根据上述方法，计算每个节点与其他节点的平均距离
def cal_distance(df):
    df1 = df.copy()
    df1 = df1.set_index('secID').drop(['ticker','tradeDate'],axis=1).sort_index()
    df2 =df1.values.repeat([len(df1)]*len(df1),axis=0)
    df3 = np.concatenate([df1]*len(df1))
    df4 = pd.DataFrame(np.sqrt(np.sum((df2 -df3)**2,axis=1)).reshape(len(df1),-1),index=df1.index,columns=df1.index)
    df5 = df4.mean(axis=1)
    del df1,df2,df3,df4
    return df5
stock_distance_metrix = factor_df1.groupby('tradeDate').apply(lambda x: cal_distance(x) )
stock_distance_metrix = stock_distance_metrix.reset_index().rename(columns={0:'mean_distance'})
stock_distance_metrix[:2]

# 计算过去四周网络距离的平均值
def cal_period_mean_distance(df, period_step=4):
    df = df.copy()
    df = df.sort_values('tradeDate')
    df['period_mean_distance'] = df['mean_distance'].rolling(period_step).mean()
    return df
stock_distance_metrix1 = stock_distance_metrix.groupby('secID',as_index=False).apply(lambda x: cal_period_mean_distance(x, 4)).dropna().reset_index().drop(['level_0','level_1'],axis=1)
stock_distance_metrix1[:2]


'''
第二部分：网络动量因子测试
该部分耗时 ** 2分钟**
该部分内容为：

获取股票月度收益率:利用uqer API获取;

因子生成及测试：对网络动量因子进行测试，包括计算分组、IC、ICIR、多空收益等；

因子在市值上的分布情况

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

2.1:个股周度收益率

'''

# 获取个股周度收益率，由于数据量比较大， 鉴于优矿返回数据条数的限制，这里分段进行数据获取
bt_mret_df1 = DataAPI.MktEquwAdjGet(beginDate=start_date,endDate="2013-12-31",secID="",field=u"secID,endDate,chgPct",pandas="1")
bt_mret_df2 = DataAPI.MktEquwAdjGet(beginDate='2014-01-01',endDate="2016-12-31",secID="",field=u"secID,endDate,chgPct",pandas="1")
bt_mret_df3 = DataAPI.MktEquwAdjGet(beginDate='2017-01-01',endDate=end_date,secID="",field=u"secID,endDate,chgPct",pandas="1")
bt_mret_df = bt_mret_df1.append(bt_mret_df2).append(bt_mret_df3)
bt_mret_df.rename(columns={'endDate':'tradeDate', 'chgPct':'curr_ret'}, inplace=True)
bt_mret_df['ticker'] = bt_mret_df['secID'].str.slice(0,6)
bt_mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
bt_mret_df['nxt_ret'] = bt_mret_df.groupby('ticker')['curr_ret'].shift(-1)
bt_mret_df = bt_mret_df.dropna(subset=['nxt_ret'])
bt_mret_df[:2]

'''
2.2:因子测试

所用的因子即为上文计算得到的过去四周股票网络距离的平均值，回测的具体细节如下：
1）回测区间：2014 年1月至2019年8月；
2）选股范围：全部A股；
3）划分方法：根据平均距离从小到大排序划分为10 组，组内等权配置；
4）调仓频率：每周末调仓；
5）交易成本：双边千分之二

'''


back_test_date = '2014-01-01'
stock_distance_metrix2 = stock_distance_metrix1[stock_distance_metrix1['tradeDate']>=back_test_date]

factor_rtn_df = stock_distance_metrix2.merge(bt_mret_df, on=['secID', 'tradeDate'])
period_ic = factor_rtn_df.groupby('tradeDate').apply(lambda x: x[['period_mean_distance','nxt_ret']].corr(method="spearman").values[0, 1])
ic = period_ic.mean()
std = period_ic.std()
icir = ic / std
ic_t = stats.ttest_1samp(period_ic, 0)[0]

ic_summary = pd.DataFrame([ic, std, icir, ic_t], index = [u'IC均值', u'IC波动率',u'ICIR', u't值'], columns=['网络动量因子']).T.applymap(lambda x: round(x,3)) 
ic_summary

'''
可见网络动量为显著的反向因子， IC达到-5.5%
'''

#计算超额收益
def excess_rtn(s):
    r = s.iloc[-1]/s.iloc[0] - 1 
    return r
#计算胜率
def winper(s):
    s = s[s!=0]
    return (s>0).sum() / float(len(s))
#计算最大回测
def maxDrawdown(s):
    cum_max = s.cummax()
    maxdrawdown =((cum_max-s)/cum_max).max()
    return maxdrawdown
#计算年化收益
def annual_rtn(s, l,step=250):
    r = s.iloc[-1]/s.iloc[0] - 1
    ar = r / l * step
    return ar
#计算信息比率
def cal_ir(s):
    m = s.mean()
    m1 = m*12
    std1 = s.std()* np.sqrt(12)
    ir = m1/std1
    return ir

import matplotlib.pyplot as plt
import matplotlib as mpl
from CAL.PyCAL import *
mpl.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei'] 

group_num = 5
stock_distance_metrix2['ticker'] = stock_distance_metrix2['secID'].str.slice(0,6)
perf = quant_util.simple_group_backtest(stock_distance_metrix2.copy(), bt_mret_df, 'period_mean_distance', 'nxt_ret', ngrp=group_num)

fig = plt.figure(figsize=(40,15))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
nav = []
label_dict = {}
for i in range(group_num):
    label_dict[i+1] = u'第%s组'%(i+1)
    if i == 0:
        label_dict[i+1] += '(low)'
    elif i == group_num-1:
        label_dict[i+1] += '(high)'
    gperf = perf[perf['group'] == i]
    nav = nav + [gperf['cum_ret'].values[-1]]
    _=ax1.plot(pd.to_datetime(gperf['tradeDate']), gperf[['cum_ret']], label=label_dict[i+1])
ax1.set_ylabel(u"净值",fontproperties=font, fontsize=16)
ax1.set_title(u"传统net_momentum因子五档回测净值", fontproperties=font, fontsize=16)
ax1.legend(loc=0, prop=font)

ind = np.arange(group_num)
ax2.bar(ind+1.0/group_num, nav, 0.3, color='r')
ax2.set_xlim((0, ind[-1]+1))
ax2.set_xticks(ind+0.35)
_=ax2.set_xticklabels([label_dict[i+1] for i in ind], fontproperties=font)

# long-short 净值曲线
#perf,bt_df = quant_util.long_short_backtest(stock_distance_metrix2.copy(), bt_mret_df, 'period_mean_distance', 'nxt_ret', -1,commission=0.002)

perf,bt_df = quant_util.long_short_backtest(stock_distance_metrix2.copy(), bt_mret_df, 'period_mean_distance', 'nxt_ret', -1)

f, ax= plt.subplots(nrows=1, ncols=1, figsize = (15, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.4)
perf = perf.set_index('tradeDate')
_ = perf["cum_ret"].plot(ax=ax)
_ = ax.set_title(u"long-short net value")

r = []
excess_rtn1 = excess_rtn(perf['cum_ret'].dropna())
winper1 = winper(perf['period_ret'].dropna())
maxDrawdown1 = maxDrawdown(perf['cum_ret'].dropna())
ir1 = cal_ir(perf['period_ret'])
ar1 = annual_rtn(perf['cum_ret'].dropna(),len(perf['cum_ret'].dropna()),52)
r.append([excess_rtn1,ar1,winper1,maxDrawdown1,ir1])
gb_p1 = pd.DataFrame(r,columns=['excess_rtn','annual_rtn','winper','maxDrawdown','ir'],index=['网络动量']).applymap(lambda x: round(x,3))   
gb_p1

'''
因子分组测试区分度比较大，单调性很强，年化多空收益达到30.5%，多空胜率达到63.2%。从多空净值曲线走势来看，2014年和2017年表现一般，其他年份表现均比较好。
2.3:因子在市值上的分布
'''
df_list = []
for dt in week_end_list:
    df = DataAPI.MktStockFactorsOneDayGet(tradeDate=dt, secID=u"", ticker=u"",field=u"secID,tradeDate,LCAP", pandas="1")
    df_list.append(df)
size_df = pd.concat(df_list)
size_df = size_df.dropna()


signal_group_df = quant_util.signal_grouping(stock_distance_metrix2.copy(), 'period_mean_distance',group_num)
size_df['market_value'] = np.exp(size_df['LCAP'])
group_size = signal_group_df[['ticker','tradeDate','secID','period_mean_distance','group']].merge(size_df, on=['secID','tradeDate']).groupby('group')['market_value'].mean()
fig = plt.figure(figsize=(15,  6))
ax1 = fig.add_subplot(1, 1, 1)
_ = group_size.plot(kind='bar',ax=ax1, color='r')
_ =ax1.set_ylabel(u"Market Value")
_ = ax1.set_title(u"Market value in different group")
_ = ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

'''
可以看到第1组到第5组的平均市值逐渐变大。结合上文因子分组表现和从市值分布来看，因子和市值负相关， 市值小的股票表现越好。

第三部分：网络动量因子中性化后测试
该部分耗时 ** 2分钟**
该部分内容为：

对因子做中性化:利用uqer API，获取风险模型数据，对因子中性化;

中性化后因子生成及测试：对网络动量因子进行测试，包括计算分组、IC、ICIR、多空收益等；

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

3.1:对市值和行业进行中性化
'''

# 中性化
neu_stock_distance_metrix1 = stock_distance_metrix2.copy()
neu_stock_distance_metrix1 = quant_util.netralize_dframe(neu_stock_distance_metrix1.copy(), ['period_mean_distance'], exclude_style=['BETA','RESVOL','MOMENTUM','SIZENL','EARNYILD','BTOP','GROWTH','LEVERAGE','LIQUIDTY'])

# IC 测试
factor_rtn_df = neu_stock_distance_metrix1.merge(bt_mret_df, on=['secID', 'tradeDate'])
period_ic = factor_rtn_df.groupby('tradeDate').apply(lambda x: x[['period_mean_distance','nxt_ret']].corr(method="spearman").values[0, 1])
ic = period_ic.mean()
std = period_ic.std()
icir = ic / std
ic_t = stats.ttest_1samp(period_ic, 0)[0]

neu_ic_summary = pd.DataFrame([ic, std, icir, ic_t], index = [u'IC均值', u'IC波动率',u'ICIR', u't值'], columns=['网络动量因子']).T.applymap(lambda x: round(x,3)) 
neu_ic_summary

'''
可见对市值和行业中性化后，因子的IC值稍有下降， 但仍比较显著。
'''
# 分组及多空测试
perf = quant_util.simple_group_backtest(neu_stock_distance_metrix1.copy(), bt_mret_df, 'period_mean_distance', 'nxt_ret', ngrp=group_num)

fig = plt.figure(figsize=(40,15))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(223)
nav = []
label_dict = {}
for i in range(group_num):
    label_dict[i+1] = u'第%s组'%(i+1)
    if i == 0:
        label_dict[i+1] += '(low)'
    elif i == group_num-1:
        label_dict[i+1] += '(high)'
    gperf = perf[perf['group'] == i]
    nav = nav + [gperf['cum_ret'].values[-1]]
    _=ax1.plot(pd.to_datetime(gperf['tradeDate']), gperf[['cum_ret']], label=label_dict[i+1])
ax1.set_ylabel(u"净值",fontproperties=font, fontsize=16)
ax1.set_title(u"中性化net_momentum因子五档回测净值", fontproperties=font, fontsize=16)
ax1.legend(loc=0, prop=font)
ind = np.arange(group_num)
ax2.bar(ind+1.0/group_num, nav, 0.3, color='r')
ax2.set_xlim((0, ind[-1]+1))
ax2.set_xticks(ind+0.35)
_=ax2.set_xticklabels([label_dict[i+1] for i in ind], fontproperties=font)

perf,bt_df = quant_util.long_short_backtest(neu_stock_distance_metrix1.copy(), bt_mret_df, 'period_mean_distance', 'nxt_ret', -1,commission=0.002)
f, ax= plt.subplots(nrows=1, ncols=1, figsize = (15, 8))
plt.subplots_adjust(wspace=0.2, hspace=0.4)
perf = perf.set_index('tradeDate')
_ = perf["cum_ret"].plot(ax=ax)
_ = ax.set_title(u"long-short net value")

r = []
excess_rtn1 = excess_rtn(perf['cum_ret'].dropna())
winper1 = winper(perf['period_ret'].dropna())
maxDrawdown1 = maxDrawdown(perf['cum_ret'].dropna())
ir1 = cal_ir(perf['period_ret'])
ar1 = annual_rtn(perf['cum_ret'].dropna(),len(perf['cum_ret'].dropna()),52)
r.append([excess_rtn1,ar1,winper1,maxDrawdown1,ir1])
gb_p = pd.DataFrame(r,columns=['excess_rtn','annual_rtn','winper','maxDrawdown','ir'],index=['网络动量_中性化（size+industry）']).applymap(lambda x: round(x,3))   

'''
中性化后的因子分组测试单调性仍然很强，年化多空收益达到19%,ir为0.934,相比中性化之前有所提升。

第四部分：总结
基于优矿提供的因子数据构建股票的网络动量因子,通过实证分析表明该选股因子具有较好的应用价值。构建的股票网络动量因子周度IC达到5.5%，long-short测试年化收益为30.5%,对市值和行业中性化之后，因子仍非常有效，周度IC达到4.6%，年化多空收益达到19%。
'''