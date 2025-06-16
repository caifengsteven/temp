# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:40:52 2020

@author: Asus
"""


'''
导读
A. 研究目的：本文利用优矿的分钟级别数据与回测框架，参考海通证券《选股因子系列研究(十九)——高频因子之股票收益分布特征》与《选股因子系列研究（二十五）——高频因子之已实现波动分解》中的研究方法，对研报的结果进行了实证分析，用以探索日内高频数据因子在选股方面的应用

B. 研究结论：

本文主要利用了股票分钟线数据，计算了收益的方差、偏度、峰度因子，并且将波动拆分成“上行波动+下行波动”的形式，实证偏度因子与上行波动因子具有选股的能力

对偏度因子与上行波动因子剔除市值、行业、换手及反转因子相关性后，虽然选股效果有所减弱，但仍具备选股能力

从Fama-MacBeth回归结果、因子权重占比及TOP50股票纯多头回测来看，增加偏度因子与上行波动因子进入传统多因子模型是能提高模型效果的

C. 文章结构: 本文共分为3个部分，具体如下

一、数据准备，利用分钟线数据提取因子、同时利用API调取多因子模型所需因子，并做相关处理

二、单因子回测，该部分主要包括分组回测及正交化处理后的回测

三、多因子回测，该部分主要包括Fama-MacBeth回归检验、多因子合成的权重分配情况及合成后的因子回测

D. 运行时间说明

一、数据准备，因为读取高频数据过多，需要6小时左右，读者可以减少回测区间缩减时间

二、单因子回测，需要1分钟左右

三、多因子回测，需要7分钟左右

总耗时6小时左右

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
一、数据准备
该部分耗时 6小时左右
该部分内容为：

获取高频数据，提取高频方差因子、高频偏度因子、高频峰度因子，并且将波动拆分成“上行波动+下行波动”的形式，最后进行标准化、去极值等操作

利用uqer的因子API获取市值、换手、反转等常规因子，并进行标准化、去极值等操作

对高频因子进行正交化处理、剔除行业、市值、换手等常规因子的影响

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 获取高频数据，并计算因子

   调试 运行
文档
 代码  策略  文档
由于内存限制，该章节分为两小节:

每15天读取一次高频数据，进行方差、偏度、峰度因子的计算，并存储在raw_data/high_freq_signal.csv

读取上述文件，在每个月末计算该月因子的均值，当作最终因子值，存储在raw_data/high_freq_month_signal.csv

   调试 运行
文档
 代码  策略  文档
1.1.1 计算每日因子

因子内存及时间限制，本文只考虑1分钟级别的行情数据

因为本章节读取的分钟级别数据过多，需要占用很多资源，建议该章节运行结束后重启环境释放已占资源，最终的结果进行了存储，重启不会影响后续章节运行

重启研究环境的步骤为：

网页版：先点击左上角的“Notebook”图标，然后点击左下角的“内存占用x%”图标，在弹框中点击重启研究环境
客户端：点击左下角的“内存x%”, 在弹框中点击重启研究环境
特别说明: 由于本节读取数据过多，时间过长，如果存在网络连接断开、内存不足系统强制重启等情况时，只需重跑相应cell中的代码即可，本节支持断点再续功能

   调试 运行
文档
 代码  策略  文档
本小节共计算了9种因子，其中包括:

两种方法计算的方差、偏度、峰度因子，共2 X 3 = 6种因子:
计算方法1:
方差:Vari=∑j=1Nr2ij

偏度:Skewi=N−−√∑Nj=1r3ijVar3/2i

峰度:Kurti=N∑Nj=1r4ijVar2i
计算方法2:
方差:Vari=∑j=1N(rij−ri~)2

偏度:Skewi=N−−√∑Nj=1(rij−ri~)3Var3/2i

峰度:Kurti=N∑Nj=1(rij−ri~)4Var2i
上行波动因子、下行波动因子、上行波动占比因子，共3种因子:
计算方法:
上行波动:(∑t(rtiIrti>0)2)12
下行波动:(∑t(rtiIrti<0)2)12
上行波动占比:∑t(rtiIrti>0)2∑t(rti)2
其中，rij代表着股票i在j时刻分钟线的对数收益率, 本文分钟线采用的均是1分钟。 N代表着1天内共多少个分钟线样本，ri~代表着股票当天的分钟线收益率均值。Irti>0是指示函数，当该时刻分钟线的收益率(rti)大于0时，其取1，否则为0。

'''


import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
from CAL.PyCAL import *
import matplotlib.pyplot as plt
import time
import os
import datetime as dt
from dateutil.relativedelta import relativedelta

# 新建文件夹存放数据
raw_data_dir = "./raw_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

# 设置起始时间和结束时间
start_date = '2012-12-01'
end_date = '2018-06-01'

cal_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
month_end_list = cal_df[cal_df['isMonthEnd']==1]['calendarDate'].values
trade_date_list = cal_df[cal_df['isOpen']==1]['calendarDate'].values


def cal_signal(org_data):
    """
    计算高频信号
    params:
        org_data: Dataframe, columns=['ticker', 'openPrice', 'closePrice'], 股票的分钟级行情
    return:
        Dataframe， 返回计算后的高频信号
    """
    data = org_data.copy()
    data = data[[True if (item[11:]>'09:30' and item[11:]<='14:57') else False  for item in data.index]]
    data['ret'] = data['closePrice'].apply(lambda x : np.log(x)) - data['openPrice'].apply(lambda x : np.log(x))
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    data['tradeDate'] = [item[:10]  for item in data.index]
    
    ticker = data.iloc[0]['ticker']
    
    
    def _cal_indicator(input_data):
        """
        计算方差、偏度、峰度
        params:
            org_data: Dataframe, columns包括['ticker', 'ret'], 股票的分钟级对数收益率数据
        return:
            Dataframe， 返回计算后的高频信号
        """
        input_data = input_data.copy()
        num = len(input_data)
        input_data['ret_'] = input_data['ret'] - input_data['ret'].mean()
        
        # 两种方法计算方差、偏度、峰度
        var1 = np.sum(input_data['ret']**2)
        skew1 = np.sum(input_data['ret']**3) * np.sqrt(num) / var1**1.5
        kurt1 = np.sum(input_data['ret']**4) * num / var1**2
        
        var2 = np.sum(input_data['ret_']**2)
        skew2 = np.sum(input_data['ret_']**3) * np.sqrt(num) / var2**1.5
        kurt2 = np.sum(input_data['ret_']**4) * num / var2**2
        
        # 上下行波动拆解
        up = input_data[input_data['ret'] > 0]
        down = input_data[input_data['ret'] < 0]
        
        up_var = np.sum(up['ret'] ** 2)
        down_var = np.sum(down['ret'] ** 2)
        up_var_pert = up_var / var1
        
        return pd.Series(index=['var1','skew1','kurt1','var2','skew2','kurt2', 'up_var', 'down_var', 'up_var_pert'], 
                         data=[var1, skew1, kurt1, var2, skew2, kurt2, np.sqrt(up_var), np.sqrt(down_var), up_var_pert])
    
    res = data.groupby(['tradeDate', 'ticker']).apply(lambda x : _cal_indicator(x)).dropna().reset_index()
    
    return res


signal_df = None
start_time = time.time()
fields = ['ticker','closePrice', 'openPrice']

# 每15天取一次数据，进行计算
node_list = range(0, len(trade_date_list), 15) 
is_header = True
# 获取全A的secID
a_universe = DataAPI.EquGet(equTypeCD=u"A", field=u"secID",pandas="1")['secID'].tolist()

max_date = start_date
signal_file = os.path.join(raw_data_dir, 'high_freq_signal.csv')

# 断点继续功能：如果已经有该文件，则读取文件的最大日期并记录，并删除最大日期数据(该天数据可能未全部完成)
if os.path.exists(signal_file):
    origin_signal_df = pd.read_csv(signal_file, dtype={"ticker": np.str, "tradeDate": np.str}, index_col=0)
    max_date = origin_signal_df['tradeDate'].max()
    origin_signal_df = origin_signal_df[origin_signal_df['tradeDate'] != max_date]
    origin_signal_df.to_csv(signal_file, chunksize=10000)
    is_header = False
    print('文件已存在，上次计算时间为%s, 现在继续计算.....'%max_date)
    
    

for i in range(1, len(node_list)):
    t0 = time.time()
    # 获取高频数据
    begin_date = trade_date_list[node_list[i-1]]
    end_date = trade_date_list[node_list[i]-1]
    if end_date < max_date: # 如果上次程序断掉之前已计算该日期数据，继续跑程序后不计算该数据
        continue
    times = 0
    try:
        data = get_data_cube(a_universe, fields, begin_date, end_date, freq='1m', style='sat')
    except Exception, e:
        times += 1
        if times >= 3:
            print('取不到数据，断开此次运行，之前运行数据已保存；请重启notebook，继续运行此cell中代码')
            break
    
    data_list = [data[secID] for secID in a_universe if secID in data]
    del data
    all_data = pd.concat(data_list, axis=0)
    signal = cal_signal(all_data)
    del all_data
    signal = signal[signal['tradeDate'] >= max_date] # 防止重复数据
    signal.to_csv(signal_file, chunksize=10000, mode='a', header=is_header) #追加本次计算信号进入文件
    if is_header:
        is_header = False
            
    if i % 10 == 0:
        print('trade date: %s, cost time: % s' % (trade_date_list[node_list[i]-1], time.time()-t0))

print "Time cost: %s seconds" % (time.time() - start_time)


import time
import os
import datetime as dt
from dateutil.relativedelta import relativedelta

raw_data_dir = "./raw_data"
# 设置起始时间和结束时间
start_date = '2012-12-01'
end_date = '2018-06-01'

cal_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
month_end_list = cal_df[cal_df['isMonthEnd']==1]['calendarDate'].values
trade_date_list = cal_df[cal_df['isOpen']==1]['calendarDate'].values

# 获取所有全A股票
equ_df = DataAPI.EquGet(equTypeCD=u"A", listStatusCD=u"", field=['secID', 'ticker', 'listDate', 'delistDate'], pandas="1")
equ_df['listDate'] = equ_df['listDate'].apply(str)
equ_df['delistDate'] = equ_df['delistDate'].apply(str)

def str2date(date_str):
    """
    转换字符串为时间类型
    params:
        date_str: str, YYYYMMDD或YYYY-MM-DD, 时间类型的字符串
    return:
        date， 返回转换后的时间格式
    """
    date_str = date_str.replace("-", "")
    date_obj = dt.datetime(int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]))
    return date_obj

def get_universe(date_str, list_date=90):
    """
    获取给定日期满足条件的所有A股股票ticker
    params:
        date_str: str, YYYYMMDD或YYYY-MM-DD, 时间类型的字符串
        list_date： int，默认90天，即截至给定日期需满足上市大于90天
    return:
        ticker， Set, 返回满足条件的所有A股股票ticker
    """
    list_date_need = (str2date(date_str) + relativedelta(days=-list_date)).strftime("%Y-%m-%d")
    A_ticker = set(equ_df[(equ_df['listDate'] <= list_date_need) & ((equ_df['delistDate'] > date_str) | (equ_df['delistDate'].isnull()))]['ticker'])
    st_ticker = set(DataAPI.SecSTGet(beginDate=list_date_need, endDate=date_str, pandas="1")['ticker'])
    
    return A_ticker - st_ticker

def remove_halt(df, no_halt_day=5):
    """
    去除停牌过多的股票
    params:
        df: DataFrame, 给定的数据，必须包括['ticker', 'tradeDate']列
        no_halt_day： int，默认5天，即股票必须含有大于5天的数据
    return:
        cal_signal_sample， DataFrame, 返回满足条件的数据
    """
    df = df.dropna()
    num_of_sample = df.groupby(by='ticker').count()['tradeDate']
    cal_universe = (num_of_sample[num_of_sample >= no_halt_day]).index.tolist()
    cal_signal_sample = df[df['ticker'].isin(cal_universe)]
    return cal_signal_sample

start_time = time.time()
# 读取上一章节计算的每日因子值
origin_signal_df = pd.read_csv(os.path.join(raw_data_dir, 'high_freq_signal.csv'), dtype={"ticker": np.str, "tradeDate": np.str},index_col=0)

signal_df = None
# 每个月末，取该月因子平均值作为最终因子值
for end_date in month_end_list:
    a_universe= get_universe(end_date)
    begin_date = end_date[:8] + '01'
    
    data = origin_signal_df[(origin_signal_df['tradeDate']>=begin_date) & (origin_signal_df['tradeDate']<=end_date)]
    
    # 去除停牌过多的股票
    data = remove_halt(data)
    signal = data.groupby('ticker').mean()
    signal = signal.loc[a_universe].dropna().reset_index()
    signal['tradeDate'] = end_date.replace("-", '')
    if signal_df is None:
        signal_df = signal
    else:
        signal_df = pd.concat([signal_df, signal], axis=0)

signal_df.to_csv(os.path.join(raw_data_dir, 'high_freq_month_signal.csv'))
print('因子文件格式为:')
print(signal_df.head(5).to_html())
print ("Time cost: %s seconds" % (time.time() - start_time))


'''

1.2 获取uqer因子数据、并进行winsorize, neutralize, standardize处理

   调试 运行
文档
 代码  策略  文档
本章节读取常见的几个因子，并进行相关处理

winsorize

上界值=因子中位数+5*|中位数（因子值-因子中位数）|，下界值=因子中位数-5*|中位数（因子值-因子中位数）|，超过上下界的值用上下界值填充
neutralize和standardize

直接调用优矿的neutralize函数进行市值、行业的中性化

对中性化后的因子进行标准化，直接调用优矿的standardize函数


'''


import pandas as pd
import numpy as np
import os
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import gevent


# 设置起始时间和结束时间
begin_date = '2012-12-01'
end_date = '2018-05-01'
factor_name = ['LCAP', 'REVS20', 'VOL20', 'Beta20']

raw_data_dir = "./raw_data"

a_universe = DataAPI.EquGet(equTypeCD=u"A", field=u"secID",pandas="1")['secID'].tolist()

def get_factor_by_day(tdate):
    '''
    获取给定日期的因子信息
    参数： 
        tdate, 时间，格式%Y%m%d
    返回:
        DataFrame, 返回给定日期的因子值
    '''
    cnt = 0
    while True:
        try:
            data = get_data_cube(a_universe, ['ticker', 'tradeDate'] + factor_name, tdate, tdate, style='tas')
            x = data[tdate]
            x['tradeDate'] = tdate.replace("-", "")

            return x
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                break


def winsorize_by_date(cdate_input):
    '''
    按照[dm+5*dm1, dm-5*dm1]进行winsorize
    参数:
        cdate_input: 某一期的因子值的dataframe
    返回:
        DataFrame, 去极值后的因子值
    '''
    cdate_input = cdate_input.copy()
    dm = cdate_input.median()
    dm1 = (cdate_input - dm).abs().median()

    upper = dm + 5 * dm1
    lower = dm - 5 * dm1
    cdate_input[cdate_input > upper] = upper
    cdate_input[cdate_input < lower] = lower
    return cdate_input
  

def standardize_winsorize_neutralize_factor(input_data):
    """
    进行行业内标准化
    输入：
        input_data：tuple, 传入的是(因子值，时间)。因子值为DataFrame
    返回：
        DataFrame, 行业标准化后的因子值
    """
    cdate_input, tdate = input_data
    
    cdate_input = cdate_input.dropna()
    cdate_input = cdate_input.set_index('ticker')
    for a_factor in factor_name:
        cnt = 0
        while True:
            try:
                if a_factor != 'LCAP':
                    cdate_input.loc[:, a_factor] = neutralize(winsorize_by_date(cdate_input[a_factor]), target_date=tdate, 
            exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'SIZENL'])
                else:
                    cdate_input.loc[:, a_factor] = neutralize(winsorize_by_date(cdate_input[a_factor]), target_date=tdate, 
            exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'SIZENL', 'SIZE'])
                    
                cdate_input.loc[:, a_factor] = standardize(cdate_input.loc[:, a_factor])
                break
            except Exception as e:
                cnt += 1
                if cnt >= 3:
                    cdate_input.loc[:, a_factor] = standardize(winsorize_by_date(cdate_input.loc[:, a_factor]))
                    break

    return cdate_input


if __name__ == "__main__":
    start_time = time.time()

    # 拿到交易日历，得到月末日期
    trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin_date, endDate=end_date, field=u"", pandas="1")
    trade_date = trade_date[trade_date.isMonthEnd == 1]

    print("begin to get factor value for each stock...")
    # # 取得每个月末日期，所有股票的因子值
    pool = ThreadPool(processes=16)
    date_list = [tdate for tdate in trade_date.calendarDate.values]
    frame_list = pool.map(get_factor_by_day, date_list)
    pool.close()
    pool.join()
    print "ALL FINISHED"

    
    # 遍历每个月末日期，利用协程对因子进行标准化，中性化处理
    print('standardize & neutralize & winsorize factor...')
    jobs = [gevent.spawn(standardize_winsorize_neutralize_factor, value) for value in zip(frame_list, date_list)]
    gevent.joinall(jobs)
    new_frame_list = [e.value for e in jobs]
    print('standardize neutralize factor finished!')
    
            
    # 将不同月份的数据合并到一起
    all_frame = pd.concat(new_frame_list, axis=0)
    all_frame.reset_index(inplace=True)
    
    ################################ 数据存储下来 ################################
    all_frame.to_csv(os.path.join(raw_data_dir, 'month_factor.csv'), chunksize=1000)

    end_time = time.time()
    print "Time cost: %s seconds" % (end_time - start_time)
    
'''    
1.3 正交化处理

   调试 运行
文档
 代码  策略  文档
对1.1章节计算的高频因子进行正交化处理，主要是剥离市值、行业、换手等的影响


'''

import statsmodels.api as sm
import pandas as pd
import numpy as np
import os
import gevent
import time
raw_data_dir = "./raw_data"

start_time = time.time()
# 读取高频月末因子
signal_df = pd.read_csv(os.path.join(raw_data_dir, 'high_freq_month_signal.csv'), dtype={"ticker": np.str, "tradeDate": np.str},index_col=0)

orthog_signal_df = signal_df.copy()

# 高频因子对市值、行业进行中性化
date_list = orthog_signal_df['tradeDate'].unique()
frame_list = [orthog_signal_df[orthog_signal_df['tradeDate']==trade_date] for trade_date in date_list]
factor_name = ['var1', 'skew1', 'kurt1', 'var2', 'skew2', 'kurt2', 'up_var', 'down_var', 'up_var_pert']
jobs = [gevent.spawn(standardize_winsorize_neutralize_factor, value) for value in zip(frame_list, date_list)]
gevent.joinall(jobs)
new_frame_list = [e.value for e in jobs]
orthog_signal_df = pd.concat(new_frame_list, axis=0)
orthog_signal_df.reset_index(inplace=True)

# 读取月末的常见因子，包括动量、换手等
factor_df = pd.read_csv(os.path.join(raw_data_dir, 'month_factor.csv'), dtype={"ticker": np.str, "tradeDate": np.str},index_col=0)
merge_factor = pd.merge(factor_df, orthog_signal_df, on=['ticker', 'tradeDate'])

orthog_factor_names = ['var1', 'skew1', 'kurt1', 'var2', 'skew2', 'kurt2', 'up_var', 'down_var', 'up_var_pert']
other_factor_names = ['REVS20', 'VOL20', 'Beta20']

def orthog_by_given_factors(signal):
    """
    正交化高频因子
    输入：
        signal：DataFrmae, 因子数据。
    返回：
        DataFrame, 正交化后的高频因子
    """    
    signal = signal.dropna()
    
    for factor in orthog_factor_names:
        results = sm.OLS(signal[factor].values, sm.add_constant(signal[other_factor_names].values)).fit()
        signal[factor] = standardize(pd.Series(index=signal['ticker'], data=results.resid)).values
    
    return signal

# 正交化高频因子
orthog_signal_df = merge_factor.groupby('tradeDate').apply(lambda x : orthog_by_given_factors(x))
orthog_signal_df = orthog_signal_df.reset_index(drop=True)

# 合并原始高频因子、正交化后的高频因子
orthog_signal_df = orthog_signal_df.rename(columns=dict(zip(orthog_factor_names, ['orthog_'+factor for factor in orthog_factor_names])))
merge_factor = pd.merge(orthog_signal_df, signal_df, on=['ticker', 'tradeDate'])
merge_factor.to_csv(os.path.join(raw_data_dir, 'all_high_freq_signal.csv'))

print "Time cost: %s seconds" % (time.time() - start_time)


'''
二、单因子回测
该部分耗时 小于1分钟
本章节主要测试单因子的选股效果，进行了分组测试和IC的计算。总共有4部分

第一部分测试高频方差因子与按照"上行波动+下行波动"拆分后的方差因子
第二部分测试高频偏度因子
第三部分测试高频峰度因子
第四部分考察了因子的分组特征及正交化后的因子测试情况
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)
'''

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
import seaborn as sns
sns.set_style('white')
from CAL.PyCAL import *    # CAL.PyCAL中包含font
from scipy.stats.mstats import gmean
import time


def get_trade_dates(start_date, end_date, frequency='d'):
    """
    输入起始日期和频率，即可获得日期列表（daily包括起始日，其余的都是位于起始日中间的）
    输入：
       start_date，开始日期，'YYYYMMDD'形式
       end_date，截止日期，'YYYYMMDD'形式
       frequency，频率，daily为所有交易日，weekly为每周最后一个交易日，monthly为每月最后一个交易日，quarterly为每季最后一个交易日
    返回：
       获得list型日期列表，以'YYYYMMDD'形式存储
    """
    data = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date,
                               field=u"calendarDate,isOpen,isWeekEnd,isMonthEnd,isQuarterEnd", pandas="1")
    if frequency == 'd':
        data = data[data['isOpen'] == 1]
    elif frequency == 'w':
        data = data[data['isWeekEnd'] == 1]
    elif frequency == 'm':
        data = data[data['isMonthEnd'] == 1]
    elif frequency == 'q':
        data = data[data['isQuarterEnd'] == 1]
    else:
        raise ValueError('调仓频率必须为d/w/m！！')
    date_list = map(lambda x: x[0:4] + x[5:7] + x[8:10], data['calendarDate'].values.tolist())
    return date_list

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
    signal_df_tmp.sort_values(factor_name, inplace=True)
    signal_df_tmp.dropna(subset=[factor_name], inplace=True)
    signal_df_tmp['group'] = signal_df_tmp.groupby('tradeDate')[factor_name].apply(
        lambda x: (x.rank() - 1) / len(x) * ngrp).astype(int)
    return signal_df_tmp


def plot_quantile_excess_return(perf, title):
    # 因子分组的超额收益作图
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    
    x = range(len(perf))
    perf.plot(kind='bar', ax=ax1, legend=False)
    plt.legend(perf.columns, prop=font, loc='best', handlelength=4, handletextpad=1, borderpad=0.5, ncol=2)

    ax1.set_ylabel(u'超额收益', fontproperties=font, fontsize=16)
    ax1.set_xlabel(u'分位组', fontproperties=font, fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels([int(x)+1 for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
    ax1.set_yticklabels([str(x * 100) + '0%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
    ax1.set_title(title, fontproperties=font, fontsize=16)
    ax1.grid()
    plt.show()
    

class FactorTest:
    def __init__(self, n_quantile=10):
        """
        构造FactorTest类
        params: 
            n_quantile: int, 默认为10，及根据因子值对股票划分10组进行回测
        """        
        self.n_quantile = n_quantile


    def calc_ic(self, signal_df, return_df, factor_name, ret_name, method='spearman'):
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
        ic_df = merge_df.groupby('tradeDate').apply(
            lambda x: x[[factor_name, ret_name]].corr(method=method).values[0, 1]).dropna()
        # 计算邻月IC
        merge_df.sort_values(['ticker', 'tradeDate'], inplace=True)
        merge_df[factor_name + '_next'] = merge_df.groupby('ticker')[factor_name].shift(-1)
        merge_df.dropna(inplace=True)
        next_ic_df = merge_df.groupby('tradeDate').apply(
            lambda x: x[[factor_name, factor_name + '_next']].corr(method='spearman').values[0, 1])

        result = pd.concat([ic_df, next_ic_df], axis=1, names=[factor_name, factor_name + '_next_ic'])
        result.columns = [factor_name, factor_name + '_next_ic']
        return result

    def ic_describe(self, ic_df):
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
        ic_ir = ic_mean / ic_std * np.sqrt(12.0)
        # IC>0的比例
        ic_p_pct = (ic_df[factor_name] > 0).sum() / len(ic_df)
        # 下月IC相关系数均值
        ic_auto_corr = ic_df[[fname + '_next_ic' for fname in factor_name]].mean()
        ic_auto_corr.index = factor_name

        # IC统计
        ic_table = pd.DataFrame([ic_mean, ic_std, ic_t, ic_ir, ic_p_pct, ic_auto_corr],
                                index=[u'平均IC', u'IC标准差', u'IC均值T统计量', u'IC_IR', u'IC大于0的比例', u'下月IC相关系数均值'])
        return ic_table.T



    def cal_quantile_return(self, signal_df, return_df, factor_name, return_name, direction=1):
        """
        分组回测， 根据因子值将个股等分成给定组数，进行回测
        根据调仓频率，进行交易，返回最后的累计超额收益率。
        params:
                signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一列为股票当日的因子值
                return_df: DataFrame, columns=['ticker', 'tradeDate', [next_period_return]], 收益率，只含有调仓日，以及下期累计收益率
                factor_name:　str, signal_df中因子值的列名 
                return_name： str, return_df中收益率的列名
                direction： {1,-1}, 操作方向， 1为正向操作， -1为反向操作， 默认为1
        return:
                DataFrame, columns=['tradeDate', 'cum_ret'], 返回累计超额收益率
        """
        bt_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'])

        # 分祖
        bt_df.dropna(subset=[factor_name], inplace=True)
        bt_df = signal_grouping(bt_df, factor_name=factor_name, ngrp=self.n_quantile)

        # 计算权重：每组等权    
        count_df = bt_df.groupby(['tradeDate', 'group']).apply(lambda x: len(x)).reset_index()
        count_df.columns = ['tradeDate', 'group', 'count']
        bt_df = bt_df.merge(count_df, on=['tradeDate', 'group'])
        bt_df['weight'] = 1.0 / bt_df['count']

        # 如果direction=1, 则做多因子值最大的一组， 做空因子值最小的一组；如果direction=-1, 则做空因子值最大的一组， 做多因子值最小的一组
        bt_df['longshort'] = 0.0
        bt_df.loc[bt_df['group']==self.n_quantile-1, 'longshort'] = bt_df.loc[bt_df['group']==self.n_quantile-1, 'weight'] * direction
        bt_df.loc[bt_df['group']==0, 'longshort'] = bt_df.loc[bt_df['group']==0, 'weight'] * (-direction)

        longshort_perf = bt_df.groupby('tradeDate').apply(lambda x: np.sum(x[return_name] * x['longshort'])).reset_index()
        longshort_perf.columns = ['tradeDate', 'longshort']
        
        # 统计每组的超额收益率
        group_pref = bt_df.groupby(['tradeDate', 'group']).apply(lambda x: np.sum(x[return_name] * x['weight'])).reset_index()
        group_pref.columns = ['tradeDate', 'group', 'ret']
        market_pref = bt_df.groupby(['tradeDate']).apply(lambda x: np.sum(x[return_name] * x['weight'])/np.sum(x['weight'])).reset_index()
        market_pref.columns = ['tradeDate', 'market_ret']
        merge_pref = pd.merge(group_pref, market_pref, on='tradeDate')
        merge_pref['ret'] = merge_pref['ret'] - merge_pref['market_ret']
        merge_pref = merge_pref[['tradeDate', 'group', 'ret']]
        
        group_ret = pd.pivot_table(merge_pref, index='tradeDate', values='ret', columns='group').reset_index()
        group_ret.columns = ['tradeDate'] + ['group%s'%(item+1) for item in range(self.n_quantile)]
        
        all_ret = pd.merge(group_ret, longshort_perf, on='tradeDate')
        all_ret.sort_values('tradeDate', inplace=True)

        return all_ret.set_index('tradeDate')
    
    def perf_describe(self, perf_df):
        """
        统计因子的回测绩效， 包括年化超额收益率、年化波动率、信息比率、最大回撤
        params:
                perf_df: DataFrame, 回测的期间超额收益率， index为日期， columns为因子名， values为因子回测的期间收益率
        return:
                DataFrame, 返回回测绩效
        """
        # 记录因子个数和因子名
        factor_name = perf_df.columns.values
        n = len(factor_name)

        # 年化超额收益率
        ret_mean = pd.Series(index=factor_name, data=gmean(perf_df+1.)**12 - 1.)
        # 年化波动率
        ret_std = perf_df.std() * np.sqrt(12.0)
        # 年化IR
        sr = ret_mean / ret_std
        # 最大回撤
        maxdrawdown = {}
        for i in range(n):
            fname = factor_name[i]
            cum_ret = pd.DataFrame((perf_df[fname] + 1).cumprod())
            cum_max = cum_ret.cummax()
            maxdrawdown[fname] = ((cum_max - cum_ret) / cum_max).max().values[0]
        maxdrawdown = pd.Series(maxdrawdown)
        # 月度胜率
        win_ret = (perf_df > 0).sum() / len(perf_df)

        perf_table = pd.DataFrame([ret_mean, sr, win_ret], index=[u'年化超额收益率', u'信息比率', u'月度胜率']).T
        
        perf_table = perf_table.ix[factor_name]
        
        return perf_table
    
    
import pandas as pd
import numpy as np

def factor_test(signal_df, month_return, factor_name_list, n_quantile=10):
    """
    单因子测试，包括分组回测情况及IC
    params:
            signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一类为股票当日的因子值
            month_return: DataFrame, columns=['ticker', 'tradeDate', [next_period_return]], 收益率
            factor_name_list:　list, signal_df中因子值的列名 
            n_quantile： int, 回测组数
    """
    factor_test = FactorTest(n_quantile)
    
    quantile_res_list = []
    return_name_list = []
    ic_res_list = []
    for factor_name in factor_name_list:
        var_ret = factor_test.cal_quantile_return(signal_df, month_return, factor_name, 'next_month_return', -1)
        var_perf = factor_test.perf_describe(var_ret)
        var_perf.columns = [column + '_' + factor_name for column in var_perf.columns]
        
        ic = factor_test.calc_ic(signal_df, month_return, factor_name, 'next_month_return')
        ic_des = factor_test.ic_describe(ic)
        
        quantile_res_list.append(var_perf)
        return_name_list.append(u'年化超额收益率_' + factor_name)
        ic_res_list.append(ic_des)
        
        

    var_perf = pd.concat(quantile_res_list, axis=1).round(4)
    plot_quantile_excess_return(var_perf.iloc[:n_quantile][return_name_list], u'高频因子分组超额收益(相对于全市场)')
    print('\n高频因子分组超额收益统计')
    print(var_perf.to_html())

    ic = pd.concat(ic_res_list, axis=0).round(4)
    ic.index = factor_name_list
    
    print('\n高频因子 rankIC 情况')
    print(ic.to_html())

# 加载因子数据
raw_data_dir = "./raw_data"
signal_df = pd.read_csv(os.path.join(raw_data_dir, 'all_high_freq_signal.csv'), dtype={"ticker": np.str, "tradeDate": np.str},index_col=0)

# 设置起始时间和结束时间
begin_date = '2012-01-01'
end_date = '2018-05-31'

# 获取全A的secID
a_universe = DataAPI.EquGet(equTypeCD=u"A", field=u"secID",pandas="1")['secID'].tolist()

# 获得月收益率
month_return = DataAPI.MktEqumAdjGet(secID=a_universe, beginDate=begin_date, endDate=end_date, field=u"ticker,endDate,chgPct,",pandas="1")
month_return.rename(columns={'endDate': 'tradeDate', 'chgPct': 'month_return'}, inplace=True)
month_return.sort_values(['ticker', 'tradeDate'], inplace=True)
month_return['next_month_return'] = month_return.groupby('ticker')['month_return'].shift(-1)
month_return.dropna(inplace=True)
month_return['tradeDate'] = month_return['tradeDate'].apply(lambda x : x.replace("-", ''))


'''

2.1 高频收益因子—方差因子

'''

factor_test(signal_df, month_return, ['var1', 'var2'])


factor_test(signal_df, month_return, ['up_var', 'down_var', 'up_var_pert'])

factor_test(signal_df, month_return, ['skew1','skew2'])

factor_test(signal_df, month_return, ['kurt1', 'kurt2'])


'''
2.4 正交化后的因子效用

   调试 运行
文档
 代码  策略  文档
首先考察使用因子分组后的各组股票组合在市值、换手、动量等因子上的单调特征

'''

orthog_factor_names = ['var1', 'skew1', 'kurt1', 'var2', 'skew2', 'kurt2']
other_factor_names = ['LCAP' ,'REVS20', 'VOL20', 'Beta20']


def get_group_attribute(signal_df, factor_name, title, n_quantile=10):
    """
    对给定因子分组后，考察每组的股票组合特征
    params:
        signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一类为股票当日的因子值
        factor_name:　str, 对该列进行分组
        n_quantile： int, 回测组数
    """    
    bt_df = signal_grouping(signal_df, [factor_name], ngrp=10)
    group_feature = bt_df.groupby(['group']).agg(dict(zip(other_factor_names, ['median']*len(other_factor_names))))
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    
    x = range(len(group_feature))
    group_feature.plot(ax=ax1, legend=False)
    plt.legend(group_feature.columns, prop=font, loc='best', handlelength=4, handletextpad=1, borderpad=0.5, ncol=2)
    
    ax1.set_ylabel(u'因子均值', fontproperties=font, fontsize=16)
    ax1.set_xlabel(u'分位组', fontproperties=font, fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels([int(x)+1 for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
    ax1.set_title(title, fontproperties=font, fontsize=16)
    ax1.grid()
    plt.show()
    
    
get_group_attribute(signal_df, 'up_var_pert', u'高频上行波动占比因子分组特征')

factor_test(signal_df, month_return, ['orthog_up_var_pert'], 10)

get_group_attribute(signal_df, 'skew1', u'高频偏度因子分组特征')

factor_test(signal_df, month_return, ['orthog_skew1', 'orthog_skew2'], 10)

'''


同样的，正交化后的偏度因子虽然选股作用比之前减弱，但还是有较好的区分度。IC均值下降到0.033左右，但ICIR绝对值仍然高于2.5以上。

   调试 运行
文档
 代码  策略  文档
三、多因子模型测试
该部分耗时 7分钟
对于新因子的研究最终还是要加入多因子模型，本节考察加入高频偏度因子与上行波动占比因子前后的多因子模型效果变化。
该部分分为三大部分

第一部分利用Fama-MacBeth回归来考察加入因子前后的效果

第二部分考察多因子模型中高频因子每期的权重分配情况

第三部分考察原始模型与加入高频因子后的模型回测情况

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


'''

import pandas as pd
import numpy as np
import os
import time

# 加载因子数据
raw_data_dir = "./raw_data"
signal_df = pd.read_csv(os.path.join(raw_data_dir, 'all_high_freq_signal.csv'), dtype={"ticker": np.str, "tradeDate": np.str},index_col=0)

# 设置起始时间和结束时间
begin_date = '2011-01-01'
end_date = '2018-05-31'


# 获取全A的secID
a_universe = DataAPI.EquGet(equTypeCD=u"A", field=u"secID",pandas="1")['secID'].tolist()

# 获得月收益率
month_return = DataAPI.MktEqumAdjGet(secID=a_universe, beginDate=begin_date, endDate=end_date, field=u"ticker,endDate,chgPct,",pandas="1")
month_return.rename(columns={'endDate': 'tradeDate', 'chgPct': 'month_return'}, inplace=True)
month_return.sort_values(['ticker', 'tradeDate'], inplace=True)
month_return['next_month_return'] = month_return.groupby('ticker')['month_return'].shift(-1)
month_return.dropna(inplace=True)
month_return['tradeDate'] = month_return['tradeDate'].apply(lambda x : x.replace("-", ''))

'''
3.1 Fama-MacBeth 回归检验

分别对原始因子、及加入高频因子后的数据进行回归

'''

import statsmodels.api as sm
import scipy.stats as st
def linear_reg(signal, y_column, x_column):
    """
    进行回归
    params:
        signal: DataFrame, 回归数据
        y_column: str, 列名，代表回归的因变量
        x_column: str or list, 列名，代表回归的自变量
    return:
        Series, 返回回归系数
    """
    signal = signal.dropna()
    results = sm.OLS(signal[y_column].values, sm.add_constant(signal[x_column].values)).fit()
    
    return pd.Series(index=x_column, data=results.params[1:])

def stat_reg(df, columns):
    """
    回归，并统计回归结果
    params:
        df: DataFrame, 包括因子值及股票下期收益
        columns: str or list, 列名，代表回归的因变量
    return:
        Series, 返回回归系数均值及T统计量
    """
    para = df.groupby('tradeDate').apply(lambda x : linear_reg(x, 'next_month_return', columns))
    mean_df = para.mean()
    t_value = pd.Series(st.ttest_1samp(para, 0)[0], index=para.columns)
    res = pd.concat([mean_df, t_value], axis=1)
    res.columns = [u'系数均值', u'T统计量']
    return res.T

# 计算原始因子的回归、加入高频后的回归
merge_df = signal_df.merge(month_return, on=['ticker', 'tradeDate'])
stat1 = stat_reg(merge_df, ['LCAP' ,'REVS20', 'VOL20', 'Beta20'])
stat2 = stat_reg(merge_df, ['LCAP' ,'REVS20', 'VOL20', 'Beta20', 'orthog_skew1'])
stat3 = stat_reg(merge_df, ['LCAP' ,'REVS20', 'VOL20', 'Beta20', 'orthog_skew2'])
stat4 = stat_reg(merge_df, ['LCAP' ,'REVS20', 'VOL20', 'Beta20', 'orthog_up_var_pert'])
stat = pd.concat([stat1, stat2, stat3, stat4], axis=0, keys=[u'原始信号', u'偏度因子_方法1', u'偏度因子_方法2', u'上行波动占比因子'])
print(u'Fama-MacBeth 回归检验（高频因子）')
print(stat.round(4).to_html())


'''
查看可知, 分别加入偏度因子和上行波动占比因子后的T统计量均满足显著条件，说明有一定的选股作用

   调试 运行
文档
 代码  策略  文档
3.2 因子权重分配情况

利用过去12期的ICIR作为当期因子的权重，考察每期高频因子的权重占比

'''

def get_factor_weight(signal_df, return_df, factor_name_list, ret_name):
    """
    利用IC计算每期因子权重
    params:
        signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一类为股票当日的因子值
        return_df: DataFrame, columns=['ticker', 'tradeDate', [next_period_return]], 收益率
        factor_name_list:　list, signal_df中因子值的列名 
        ret_name： str, return_df中的收益列名
    return:
        DataFrame, 返回计算的每期因子权重
    """
        
    merge_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'])
    
    ic_list = []
    # 计算IC
    for factor_name in factor_name_list:
        ic_df = merge_df.groupby('tradeDate').apply(
            lambda x: x[[factor_name, ret_name]].corr(method='spearman').values[0, 1])
        ic_list.append(ic_df)
        
    IC = pd.concat(ic_list, axis=1).shift(1).dropna()
    IC.columns = factor_name_list

    # 根据短期因子IC选择因子
    short_ic_mean = IC.rolling(12).mean()
    short_ic_std = IC.rolling(12).std()
    short_ic_ir = (short_ic_mean/short_ic_std).dropna()
    ic_weight = short_ic_ir.divide(short_ic_ir.abs().sum(axis=1), axis=0)
    
    return ic_weight

ic_weight1 = get_factor_weight(signal_df, month_return, ['LCAP' ,'REVS20', 'VOL20', 'Beta20'], 'next_month_return')
ic_weight2 = get_factor_weight(signal_df, month_return, ['LCAP' ,'REVS20', 'VOL20', 'Beta20', 'orthog_skew1'], 'next_month_return')
ic_weight3 = get_factor_weight(signal_df, month_return, ['LCAP' ,'REVS20', 'VOL20', 'Beta20', 'orthog_skew2'], 'next_month_return')
ic_weight4 = get_factor_weight(signal_df, month_return, ['LCAP' ,'REVS20', 'VOL20', 'Beta20', 'orthog_up_var_pert'], 'next_month_return')


#查看历年来各因子的系数权重变化
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

plt.style.use('ggplot')

# 画图比较因子各期的权重变化图，总权重按照上面所说已进行归一化
def plot_factor_coefficient(data, ax, this_type):
    """
    利用IC计算每期因子权重
    params:
        data: DataFrame, 股票的因子权重
        ax: matplotlib的画图模板
        this_type:　str, 类型，代表着是哪种方法计算的偏度因子
    """
        
    factor1 = data.iloc[:, :1].abs().sum(axis=1) * 100.0
    factor2 = data.iloc[:, :2].abs().sum(axis=1) * 100.0
    factor3 = data.iloc[:, :3].abs().sum(axis=1) * 100.0
    factor4 = data.iloc[:, :4].abs().sum(axis=1) * 100.0
    factor5 = data.iloc[:, :5].abs().sum(axis=1) * 100.0
    columns = data.columns.tolist()

    ax.plot(data.index, factor1, factor2, factor3, factor4, factor5, color='black')
    ax.fill_between(data.index.tolist(), 0, factor1.tolist(), facecolor='blue', label=columns[0])
    ax.fill_between(data.index.tolist(), factor1.tolist(), factor2.tolist(), facecolor='red', label=columns[1])
    ax.fill_between(data.index.tolist(), factor2.tolist(), factor3.tolist(), facecolor='green', label=columns[2])
    ax.fill_between(data.index.tolist(), factor3.tolist(), factor4.tolist(), facecolor='yellow', label=columns[3])
    ax.fill_between(data.index.tolist(), factor4.tolist(), factor5.tolist(), facecolor='grey', label=columns[4])
    
    legend = ax.legend(fontsize=12, loc='best')
    ax.set_xlim(data.index.tolist()[0], data.index.tolist()[-1])
    ax.set_ylim(-1, 101)
    ymajorFormatter = FormatStrFormatter('%.f%%')     
    ax.yaxis.set_major_formatter(ymajorFormatter)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Factor Coefficient Cumsum Percent', fontsize=12)
    ax.set_title('Type: %s'%this_type, fontsize=14, fontproperties=font)

# 对于计算的权重画图
weights = [ic_weight2, ic_weight3, ic_weight4]
title_list = [u'偏度因子权重占比情况(方法1)', u'偏度因子权重占比情况(方法2)', u'上行波动占比因子权重占比情况']
fig, axes = plt.subplots(nrows=1, ncols=len(title_list), figsize=(27, 4))

for i, weight in enumerate(weights):
    data = weight.copy()
    data.index = pd.to_datetime(data.index)
    plot_factor_coefficient(data, axes[i], title_list[i])
plt.show()

'''

图形中可以看出，每期的高频因子占比大约在20%~40%左右，说明因子在模型中发挥了一定的作用。

   调试 运行
文档
 代码  策略  文档
3.3 因子合成及回测

   调试 运行
文档
 代码  策略  文档
3.3.1 因子合成

利用上述的IC计算的每期权重值，结合原始的因子暴露，可以进行因子的合成。

'''

def combine_factor(signal_df, weight):
    """
    利用权重与因子暴露，进行每期的因子合成
    params:
        signal_df: DataFrame, 股票的因子值
        weight: DataFrame, 每期的因子权重
    return:
        DataFrame, 返回计算的合成因子
    """    
    factor_list = weight.columns.tolist()
    merge_factor = pd.merge(signal_df[['ticker', 'tradeDate']+factor_list], weight.reset_index(), on='tradeDate')
   
    merge_factor['score'] = merge_factor.apply(lambda x : np.sum(np.array([x[item+"_x"]*x[item+"_y"] for item in factor_list])), axis=1)
    
    return merge_factor[['ticker', 'tradeDate', 'score']]
    

factor_score_dict = {}
weight_dict = {u'origin_signal':ic_weight1, u"skew_method1":ic_weight2, 'skew_method2':ic_weight3, 'up_var_pert':ic_weight4}
for method, weight in weight_dict.items():
    factor_score_dict[method] = combine_factor(signal_df, weight)
    
'''

3.3.2 回测详情

选取中证全指作为选股股票池
每期选取TOP50股票做多，进行回测

'''


import time
import pickle
from CAL.PyCAL import * 

# 运行结果保存pickle的位置
save_dir = "./raw_data"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

backtest_results_dict = {}

start_time = time.time()
# -----------回测参数部分开始，可编辑------------
start = '2014-01-01'                       # 回测起始时间
end = '2018-05-31'                         # 回测结束时间
benchmark = '000985.ZICN'                        # 策略参考标准
universe = DynamicUniverse('000985.ZICN')           # 证券池，支持股票和基金
capital_base = 10000000                     # 起始资金
freq = 'd'                              
refresh_rate = Monthly(1)  


accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=capital_base)
}

# ---------------回测参数部分结束----------------

# 把回测参数封装到 SimulationParameters 中，供 quick_backtest 使用
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate, accounts=accounts)
# 获取回测行情数据
data = quartz.get_backtest_data(sim_params)

# 对4种不同类型分别进行测试
for this_type, factor_data in factor_score_dict.items():
    factor_data = factor_data.set_index('tradeDate')
    factor_data['ticker'] = factor_data['ticker'].apply(lambda x: x+'.XSHG' if x[:2] in ['60'] else x+'.XSHE')
    
    q_dates = factor_data.index.unique()
    # 调整参数, 选取top股票，进行快速回测

    # ---------------策略逻辑部分----------------

    def initialize(context):                   # 初始化虚拟账户状态
        pass

    def handle_data(context): 
        account = context.get_account('fantasy_account')
        current_universe = context.get_universe('stock')
        pre_date = context.previous_date.strftime("%Y%m%d")
        if pre_date not in q_dates:            
            return

        # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
        q = factor_data.ix[pre_date].dropna()
        q = q.set_index('ticker', drop=True)
        q = q.ix[current_universe]

        q_top = q.nlargest(50, 'score')
        my_univ = q_top.index.values

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

    # 保存运行结果
    backtest_results_dict[this_type] = {'max_drawdown': perf['max_drawdown'], 'sharpe': perf['sharpe'], 'alpha': perf['alpha'], 'beta': perf['beta'], 'information_ratio': perf['information_ratio'], 'annualized_return': perf['annualized_return'], 'bt': bt}    

    print ('backtesting for type %s..................................' % (this_type))
    
# 保存该次回测结果为文件
with open(os.path.join(save_dir, 'high_freq_backtest.pickle'), 'wb') as handle:
    pickle.dump(backtest_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ('Done! Time Cost: %s seconds' % (time.time()-start_time))

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
import os
import pandas as pd
import numpy as np
import time
import pickle

# 读取上述回测的结果，都存在了backtest_results_dict里。
# 如果读者内存过小，不能保证一次运行成功而是分批测试的，需要再写一段读取回测结果pickle文件的代码进行融合
save_dir = "./raw_data"
with open(os.path.join(save_dir, 'high_freq_backtest.pickle'), 'rb') as fHandler:
    backtest_results_dict = pickle.load(fHandler)

backtest_origin_indic = [u'alpha', u'beta', u'information_ratio', u'sharpe', u'annualized_return', u'max_drawdown']
backtest_heged_indic = [u'hedged_annualized_return', u'hedged_max_drawdown', u'hedged_volatility']
method_list = [u'origin_signal', u"skew_method1", 'skew_method2', 'up_var_pert']

def get_backtest_result(results):  
    """
    分析多因子模型回测结果
    params:
        results: dict, 三种模型的回测结果
    return:
        DataFrame, 返回计算的指标
    """        
    backtest_pd = pd.DataFrame(index=method_list, columns=backtest_origin_indic+backtest_heged_indic)

    for methond in method_list:
        bt = results[methond]['bt']

        data = bt[[u'tradeDate', u'portfolio_value', u'benchmark_return']]
        data['portfolio_return'] = data.portfolio_value / data.portfolio_value.shift(1) - 1.0  # 总头寸每日回报率
        data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0] / 10000000.0 - 1.0
        data['excess_return'] = data.portfolio_return - data.benchmark_return  # 总头寸每日超额回报率
        data['excess'] = data.excess_return + 1.0
        data['excess'] = data.excess.cumprod()  # 总头寸对冲指数后的净值序列
        data['portfolio'] = data.portfolio_return + 1.0
        data['portfolio'] = data.portfolio.cumprod()  # 总头寸不对冲时的净值序列
        data['benchmark'] = data.benchmark_return + 1.0
        data['benchmark'] = data.benchmark.cumprod()  # benchmark的净值序列
        
        hedged_max_drawdown = max([1 - v / max(1, max(data['excess'][:i + 1])) for i, v in enumerate(data['excess'])])  # 对冲后净值最大回撤
        hedged_volatility = np.std(data['excess_return']) * np.sqrt(252)
        hedged_annualized_return = (data['excess'].values[-1]) ** (252.0 / len(data['excess'])) - 1.0
        
        backtest_pd.loc[methond] = np.array([results[methond][item] for item in backtest_origin_indic] + [hedged_annualized_return, hedged_max_drawdown, hedged_volatility])

    cols = [(u'风险指标', u'Alpha'), (u'风险指标', u'Beta'), (u'风险指标', u'信息比率'), (u'风险指标', u'夏普比率'), (u'纯股票多头时', u'年化收益'),
            (u'纯股票多头时', u'最大回撤'), (u'对冲后', u'年化收益'), (u'对冲后', u'最大回撤'), (u'对冲后', u'收益波动率')]
    backtest_pd.columns = pd.MultiIndex.from_tuples(cols)
    backtest_pd.index.name = u'不同类型'
    return backtest_pd.astype(float).round(4)

backtest_pd = get_backtest_result(backtest_results_dict)
print(backtest_pd.to_html())

'''

可以看出

加入高频偏度因子后模型有少许提升，纯多头的年化收益从34.7%升至35.9%，IR从1.65提升到1.74。
加入高频上行波动占比因子后，模型效果提升比偏度因子更为明显。纯多头的年化收益从34.7%升至36.2%，IR从1.65提升到1.88。
虽然从Fama-MacBeth回归与因子权重时序变化图上看出高频因子效果显著，但选取top50股票进行回测提升却很少，特别是偏度因子，主要原因可能是

因子的收益显著主要集中在空头(从分组回测中可以看出)
回测结果跟因子权重分配方式关系密切，本文利用过去12期的ICIR作为分配标准，可以尝试其他方法进行分配再查看回测结果。

'''

