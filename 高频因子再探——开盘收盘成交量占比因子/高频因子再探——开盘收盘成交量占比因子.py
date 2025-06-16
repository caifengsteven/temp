# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:29:48 2020

@author: Asus
"""

import time
import datetime
import numpy as np
import pandas as pd
import scipy.stats as st
#import quartz as qf
#from quartz.api import *
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('mathtext', default='regular')
import seaborn as sns
sns.set_style('white')
from matplotlib import dates
#from CAL.PyCAL import font, Date, DateTime    # CAL.PyCAL中包含的font可以用于展示中文

begin_date = '20110101'  # 开始日期
end_date = '20180601'    # 结束日期
universe = set_universe('HS300')


'''

** 策略思想 **

去年社区新华哥写了一篇关于集合竞价成交占比因子，前几天看到微信公众号量化投资与机器学习的一篇文章，这篇文章构造了开盘收盘成交占比因子。两者有一定的相似之处，所以写了一篇再探高频因子的文章。其实构造这个因子的思想与集合竞价成交因子是类似的，背后具体的逻辑可以关注公众号看看该文章，本文就不细说了。

** 因子计算 **

这里主要受到微信公众号“量化投资与机器学习”的一篇文章的启发，然后在优矿上做了回测。主要参考了高频因子初探-集合竞价成交占比因子，追踪聪明钱 - A股市场交易的微观结构初探这两篇文章。按照作者的说法，开盘收盘成交量占比因子的计算方式为：
ratio=Volm+VolaVol

其中Volm表示上午9：30到10：00该股票的成交量，Vola表示下午14：30到15：00的成交量，Vol表示该支股票一天的总成交量。原作者不同的是，本文在此基础上对因子取了20天的移动平均。


** 数据准备 **

主要使用了下面两个函数，进行因子数据的计算，算的很慢，没有办法。为了防止意外情况，造成数据丢失，因子计算的过程中，每添加一条记录，就保存为csv文件。

'''


def get_factor(universe, begin, end, filename=None):
    """
    计算开盘收盘成交量占比因子
    
    universe: list, secID组成的list
    begin: datetime string, 起始日期，格式为"%Y%m%d"
    end: datetime string, 终止日期，格式为"%Y%m%d"
    file_name： string, 以".csv"结尾且符合文件命名规范的字符串
    """
    cal_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin, endDate=end).sort('calendarDate')
    cal_dates = cal_dates[cal_dates['isOpen']==1]
    all_dates = cal_dates['calendarDate'].values.tolist()
    all_dates = [x.replace('-', '') for x in all_dates]
    print ('Factor data will be calculated for ' + str(len(all_dates)) + ' days:')
    count = 0
    secs_time = 0
    start_time = time.time()
    ticker = [symbol[:6] for symbol in universe]
    data = pd.DataFrame(columns=ticker)
    
    for dt in all_dates:  
        morning_data = DataAPI.MktBarHistOneDayGet(securityID=universe, date=dt, startTime='09:31', endTime='10:00', unit=30, field='ticker,totalVolume')
        morning_data.dropna(inplace=True)
        #print morning_data
        afternoon_data = DataAPI.MktBarHistOneDayGet(securityID=universe, date=dt, startTime='14:31', endTime='15:00', unit=30, field='ticker,totalVolume')
        afternoon_data.dropna(inplace=True)
        #print afternoon_data
        
        morning_factor = morning_data[['totalVolume']]
        morning_factor.index = morning_data['ticker']
        
        afternoon_factor = afternoon_data[['totalVolume']]
        afternoon_factor.index = afternoon_data['ticker']
       
        factor = morning_factor+afternoon_factor
        factor.rename(columns={'totalVolume':dt}, inplace='True')
        factor.dropna(inplace=True)
        vol_data = DataAPI.MktEqudGet(secID=universe, tradeDate=dt, field='ticker,turnoverVol')
        vol_data.index=vol_data['ticker']
        vol_data = vol_data[['turnoverVol']]
        vol_data.rename(columns={'turnoverVol':dt}, inplace='True')
       
        factor = factor/vol_data
        factor = factor.T
        #print factor
        data = data.append(factor)
        
        count += 1
        if count > 0 and count % 10 == 0:
            finish_time = time.time()
            print (dt)
            print ('  ' + str(np.round((finish_time-start_time) - secs_time, 0)) + ' seconds elapsed.')
            secs_time = (finish_time-start_time)
        if filename:
            data.to_csv(filename)
    return data

data = get_factor(universe, '20110101', '20180601', 'factor.csv')

def getMarketValueAll(universe, begin, end, file_name=None):
    """
    获取股票历史每日市值
    universe: list, secID组成的list
    begin: datetime string, 起始日期，格式为"%Y%m%d"
    end: datetime string, 终止日期，格式为"%Y%m%d"
    file_name: string, 以".csv"结尾且符合文件命名规范的字符串
    """
    print  ('MarketValue will be calculated for ' + str(len(universe)) + ' stocks:')
    count = 0
    secs_time = 0
    start_time = time.time()
    N = 50
    ret_data = pd.DataFrame()
    for stk in universe:
        data = DataAPI.MktEqudAdjGet(secID=stk, beginDate=begin, endDate=end, field='secID,tradeDate,marketValue')    # 拿取数据
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
            print (count)
            print ('  ' + str(np.round((finish_time-start_time) - secs_time, 0)) + ' seconds elapsed.')
            secs_time = (finish_time-start_time)
    if file_name:
        ret_data.to_csv(file_name)
    return ret_data


maketvalue_data = getMarketValueAll(universe, begin_date, end_date, file_name='MarketValues_FullA.csv')


factor_data = pd.read_csv('factor.csv')
factor_data.rename(columns={'Unnamed: 0':'tradeDate'}, inplace=True)
factor_data.index = factor_data['tradeDate']
del factor_data['tradeDate']

s1 = factor_data.unstack().unstack().T
WINDOW_LENGTH = 20
s_ma = pd.rolling_mean(s1, window=WINDOW_LENGTH)

# 这里是补前面的坑，好不容易把数据算完了，懒得再弄了。
for symbol in s_ma.columns:
    if symbol[0] == '6':
        s_ma.rename(columns={symbol:symbol+'.XSHG'}, inplace=True)
    else:
        s_ma.rename(columns={symbol:symbol+'.XSHE'}, inplace=True)
s_ma.to_csv('Factor_MA%s_FullA.csv' % WINDOW_LENGTH)


factor_data = pd.read_csv('Factor_MA20_FullA.csv', parse_dates=['tradeDate'])    # 选股因子
mkt_value_data = pd.read_csv('MarketValues_FullA.csv', parse_dates=['tradeDate'])                    # 市值数据

factor_data = factor_data[factor_data.columns[:]].set_index('tradeDate')
mkt_value_data = mkt_value_data[mkt_value_data.columns[1:]].set_index('tradeDate')

factor_data[factor_data.columns[0:10]].tail()

# 因子历史表现
n_quantile = 10
# 统计十分位数
cols_mean = ['meanQ'+str(i+1) for i in range(n_quantile)]
cols = cols_mean
corr_means = pd.DataFrame(index=factor_data.index, columns=cols)

# 计算相关系数分组平均值
for dt in corr_means.index:
    qt_mean_results = []

    # 相关系数去掉nan和绝对值大于1的
    tmp_factor = factor_data.ix[dt].dropna()
    tmp_factor = tmp_factor[(tmp_factor<=1.0) & (tmp_factor>=-1.0)]
    
    pct_quantiles = 1.0/n_quantile
    for i in range(n_quantile):
        down = tmp_factor.quantile(pct_quantiles*i)
        up = tmp_factor.quantile(pct_quantiles*(i+1))
        mean_tmp = tmp_factor[(tmp_factor<=up) & (tmp_factor>=down)].mean()
        qt_mean_results.append(mean_tmp)
    corr_means.ix[dt] = qt_mean_results
    
    
# ------------- 因子历史表现作图 ------------------------
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
lns1 = ax1.plot(corr_means.index, corr_means.meanQ1, label='Q1')
lns2 = ax1.plot(corr_means.index, corr_means.meanQ5, label='Q5')
lns3 = ax1.plot(corr_means.index, corr_means.meanQ10, label='Q10')
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=[0.5, 0.1], loc='', ncol=3, mode="", borderaxespad=0., fontsize=12)
ax1.set_ylabel(u'因子', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'日期', fontproperties=font, fontsize=16)
ax1.set_title(u"因子历史表现", fontproperties=font, fontsize=16)
ax1.grid()


# 计算因子分组的市值分位数平均值
def quantile_mkt_values(signal_df, mkt_df):
    n_quantile = 10
    # 统计十分位数
    cols_mean = [i+1 for i in range(n_quantile)]
    cols = cols_mean

    mkt_value_means = pd.DataFrame(index=signal_df.index, columns=cols)

    # 计算相关系数分组的市值分位数平均值
    for dt in mkt_value_means.index:
        qt_mean_results = []

        tmp_factor = signal_df.ix[dt].dropna()
        tmp_mkt_value = mkt_df.ix[dt].dropna()
        tmp_mkt_value = tmp_mkt_value.rank()/len(tmp_mkt_value)

        pct_quantiles = 1.0/n_quantile
        for i in range(n_quantile):
            down = tmp_factor.quantile(pct_quantiles*i)
            up = tmp_factor.quantile(pct_quantiles*(i+1))
            i_quantile_index = tmp_factor[(tmp_factor<=up) & (tmp_factor>=down)].index
            mean_tmp = tmp_mkt_value[i_quantile_index].mean()
            qt_mean_results.append(mean_tmp)
        mkt_value_means.ix[dt] = qt_mean_results
    mkt_value_means.dropna(inplace=True)
    return mkt_value_means.mean()
    
# 计算因子分组的市值分位数平均值
origin_mkt_means = quantile_mkt_values(factor_data, mkt_value_data)


# 因子分组的市值分位数平均值作图
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

width = 0.3
lns1 = ax1.bar(origin_mkt_means.index, origin_mkt_means.values, align='center', width=width)

ax1.set_ylim(0.3,0.6)
ax1.set_xlim(left=0.5, right=len(origin_mkt_means)+0.5)
ax1.set_ylabel(u'市值百分位数', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'十分位分组', fontproperties=font, fontsize=16)
ax1.set_xticks(origin_mkt_means.index)
ax1.set_xticklabels([int(x) for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
ax1.set_yticklabels([str(x*100)+'0%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
ax1.set_title(u"因子分组市值分布特征", fontproperties=font, fontsize=16)
ax1.grid()

start = '2011-02-01'                       # 回测起始时间
end = '2018-06-01'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')               # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = Weekly(1)                           # 调仓频率，表示执行handle_data的时间间隔

factor_data = pd.read_csv('Factor_MA20_FullA.csv', parse_dates=['tradeDate'])     # 读取因子数据
factor_data = factor_data[factor_data.columns[:]].set_index('tradeDate')
        
factor_dates = factor_data.index.values

quantile_ten = 1                           # 选取股票的因子十分位数，1表示选取股票池中因子最小的10%的股票
commission = Commission(0.0008,0.0008)     # 交易费率设为双边万分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    q = factor_data.ix[pre_date].dropna()
    q = q[q>0]
    q_min = q.quantile((quantile_ten-1)*0.1)
    q_max = q.quantile(quantile_ten*0.1)
    my_univ = q[q>=q_min][q<q_max].index.values
    
    # 调仓逻辑
    univ = [x for x in my_univ if x in account.universe]
    
    # 不在股票池中的，清仓
    for stk in account.security_position:
        if stk not in univ:
            order_to(stk, 0)
    # 在目标股票池中的，等权买入
    for stk in univ:
        if(len(univ)>1): 
            #print len(univ)
            #print univ
            order_pct_to(stk, 1.1/len(univ))
            #order_pct_to(stk, 1/len(univ))
            
fig = plt.figure(figsize=(12,5))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.grid()

bt_quantile_ten = bt
data = bt_quantile_ten[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0
data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
data['excess_return'] = data.portfolio_return - data.benchmark_return
data['excess'] = data.excess_return + 1.0
data['excess'] = data.excess.cumprod()
data['portfolio'] = data.portfolio_return + 1.0
data['portfolio'] = data.portfolio.cumprod()
data['benchmark'] = data.benchmark_return + 1.0
data['benchmark'] = data.benchmark.cumprod()
# ax.plot(data[['portfolio','benchmark','excess']], label=str(qt))
ax1.plot(data['tradeDate'], data[['portfolio']], label='portfolio(left)')
ax1.plot(data['tradeDate'], data[['benchmark']], label='benchmark(left)')
ax2.plot(data['tradeDate'], data[['excess']], label='hedged(right)', color='r')

ax1.legend(loc=2)
ax2.legend(loc=1)
# ax2.set_ylim(bottom=0.5, top=2.5)
ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲指数净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲指数净值", fontproperties=font, fontsize=16)
ax1.set_title(u"因子最小的10%股票月度调仓走势", fontproperties=font, fontsize=16)


def backtest(start, end, quantile_five):
    benchmark = 'HS300'                        # 策略参考标准
    universe = set_universe('HS300')               # 证券池，支持股票和基金
    capital_base = 10000000                     # 起始资金
    freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
    refresh_rate = Weekly(1)                           # 调仓频率，表示执行handle_data的时间间隔
    commission = Commission(0.0002,0.0012)
    def initialize(account):                   # 初始化虚拟账户状态
        pass

    def handle_data(account):                  # 每个交易日的买入卖出指令
        pre_date = account.previous_date.strftime("%Y-%m-%d")

        # 拿取调仓日前一个交易日的Q因子，并按照相应十分位选择股票
        q = factor_data.ix[pre_date]
        q = q[q>0]
        q_min = q.quantile((quantile_five-1)*0.2)
        q_max = q.quantile(quantile_five*0.2)
        my_univ = q[q>=q_min][q<q_max].index.values
        
        # 调仓逻辑
        univ = [x for x in my_univ if x in account.universe]
        # 不在股票池中的，清仓
        for stk in account.valid_secpos:
            if stk not in univ:
                order_to(stk, 0)
        # 在目标股票池中的，等权买入
        for stk in univ:
            order_pct_to(stk, 1.1/len(univ))
    bt, perf = qf.backtest(universe=universe, start=start, end=end, initialize=initialize, handle_data=handle_data, capital_base=capital_base, refresh_rate=refresh_rate, freq=freq, commission = commission)
    return bt, perf

results = {}

factor_data = pd.read_csv('Factor_MA20_FullA.csv', parse_dates=['tradeDate'])     # 读取因子数据
factor_data = factor_data[factor_data.columns[:]].set_index('tradeDate')
factor_dates = factor_data.index.values

for quantile_five in range(1, 6):
    print (quantile_five)
    bt, perf = backtest('2014-01-01', '2018-01-08', quantile_five)
    tmp = {}
    tmp['bt'] = bt
    tmp['annualized_return'] = perf['annualized_return']
    tmp['volatility'] = perf['volatility']
    tmp['max_drawdown'] = perf['max_drawdown']
    tmp['alpha'] = perf['alpha']
    tmp['beta'] = perf['beta']
    tmp['sharpe'] = perf['sharpe']
    tmp['information_ratio'] = perf['information_ratio']
    
    results[quantile_five] = tmp
print ('done')


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
    # data[['portfolio','benchmark','excess']].plot(figsize=(12,8))
    # ax.plot(data[['portfolio','benchmark','excess']], label=str(qt))
    ax1.plot(data['tradeDate'], data[['portfolio']], label=str(qt))
    ax2.plot(data['tradeDate'], data[['excess']], label=str(qt))
    

ax1.legend(loc=0)
ax2.legend(loc=0)
ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲净值", fontproperties=font, fontsize=16)
ax1.set_title(u"开盘收盘成交占比因子 - 不同五分位数分组选股净值走势", fontproperties=font, fontsize=16)
ax2.set_title(u"开盘收盘成交占比因子 - 不同五分位数分组选股对冲中证500指数后净值走势", fontproperties=font, fontsize=16)

results_pd = pd.DataFrame(results).T.sort_index()
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

