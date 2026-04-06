# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:11:36 2020

@author: Asus
"""
'''

导读
A.研究目的：A股市场是订单驱动型市场。从动力学的角度讲，股票行情的所有演化过程，都能由订单簿（order book）自下而上精确决定。常见的日级别的价格与交易量数据（高开低收、成交量）已经被A股市场的各类量化投资者成熟运用，相同化的因子会造成“因子拥挤”问题，而逐笔成交与逐笔委托数据的信息量非常丰富，本文旨在通过对成交与委托数据的研究，对传统的量价因子进行改进。本文我们参考东吴证券：《“订单簿的温度”系列研究之一：反转因子的精细结构》，从成交笔数指标入手，结合每日的成交金额、成交笔数等交易信息，对传统的反转因子进行改进，构造了选股能力更加卓越的因子，并完成了一系列测试，获得了能完美取代传统反转因子的理想反转因子。

B.研究结论：我们借助成交笔数的信息，通过优矿平台对因子进行构造和测试，对传统反转因子的进行改进，得到了理想反转因子，其rankIC均值为-0.081，五分组净值曲线完全线性，且多头组合与其他四组区分显著，多空对冲组合年化收益20.28%，年化波动为7.38%，月度胜率82.22%，信息比率高达2.75，而且在剔除Barra风格因子和行业因子的影响之后，纯净因子的信息比率提升至3.26。

C.文章结构：本文共分为3个部分，具体如下

一、数据准备及处理：这部分主要包括股票池的界定，文中所涉及到的因子的构造、预处理等

二、反转因子的W式切割：这部分主要是研究传统反转因子的改进方向，构建理想反转因子，并进行选股能力测试

三、理想反转因子的若干深入讨论：这部分主要是利用优矿平台对构建的理想反转因子的深入讨论及相关测试

D.运行时间说明

一、数据准备及处理，需要25分钟左右

二、反转因子的W式切割，需要20分钟左右

三、理想反转因子的若干讨论，需要25分钟左右

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


第一部分：数据准备及处理
该部分耗时 25分钟
该部分内容为：

每期股票池的选取（剔除ST及上市不满60天的新股）

文中涉及的因子的构造，因子的预处理

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

1.1 每期股票池的选取

生成的股票池文件存储在reverse_improve/stock_pool.csv

股票池属性为date，code，考虑到上海证券交易所在2011年2月1号开始才有成交笔数的数据（深市一直都有），为了后续比较的一致，向后推60个交易日，我们将回测区间定为2011年6月至2018年12月初



'''


# coding: utf-8

import pandas as pd
import numpy as np
import os
import time
import multiprocessing
#from CAL.PyCAL import * 
import statsmodels.api as sm
import gevent
import datetime as dt
import pickle
import seaborn as sns
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
#from quartz_extensions.SignalAnalysis.tears import analyse_return, analyse_monthly_return, analyse_IC, analyse_construction, analyse_general

def get_calender():
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" order by tradeDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x
    
def get_month_calender():
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" and endDate>="%s" order by endDate'''%(begin)
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

cal = pd.Series(get_month_calender())


raw_data_dir = "./reverse_improve"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)
    
def get_stock_pool_one_day(end, day):
    """
    获取单期的股票池
    参数:
        end: 月末日期
        day: 新股上市判定区间长度
    返回: 
        df: DataFrame，columns = ['date', 'code']，date为此月第一个交易日
    """
    univ=DynamicUniverse('A')
    date = cal.advanceDate(end, '1B').strftime("%Y%m%d")
    all_code = univ.preview(date, skip_halted=False)
    # 剔除ST
    st_date = date if date is not None else datetime.datetime.now().strftime('%Y%m%d')
    df_ST = DataAPI.SecSTGet(secID=all_code, beginDate=st_date, endDate=st_date, field=['secID'])
    all_code_not_ST = [s for s in all_code if s not in list(df_ST['secID'])]
    # 剔除新股
    ticker = [x.split('.')[0] for x in all_code_not_ST]
    period = '-' + str(day) + 'B'
    pastDate = cal.advanceDate(date, period).strftime("%Y-%m-%d")
    ipo_date = DataAPI.SecIDGet(partyID=u"",assetClass=u"e",ticker=ticker,cnSpell=u"",field=u"ticker,listDate",pandas="1")
    remove_list = ipo_date[ipo_date['listDate'] > pastDate]['ticker'].tolist()
    all_ticker_not_new = [stk for stk in ticker if stk not in remove_list]
    # ticker2secID
    universe = DataAPI.EquGet(equTypeCD=u"A",listStatusCD="L,S,DE,UN",field=u"ticker,secID",pandas="1") # 获取所有的A股（包括已退市）
    universe = dict(universe.set_index('ticker')['secID'])
    if isinstance(ticker, list):
        res = []
        for i in ticker:
            if i in universe:
                res.append(universe[i])
            else:
                print (i, ' 在universe中不存在，没有找到对应的secID！')
    df = pd.DataFrame({'code': res})
    df['date'] = date
    df = df[['date', 'code']]
    return df

# 股票池
tic = time.time()
path = 'reverse_improve/'
start_date = '20100101'
end_date = '20190101'        
day = 60
trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date).sort('calendarDate')
trade_date = trade_date[trade_date['isMonthEnd'] == 1]
date_list = [tdate.replace("-", "") for tdate in trade_date.calendarDate.values]
all_stock = []
for date in date_list:
    stock_pool = get_stock_pool_one_day(date, day)
    all_stock.append(stock_pool)
        
all_stock = pd.concat(all_stock)
all_stock.to_csv(path + 'stock_pool.csv', index=False)
toc = time.time()
print('***********股票池示例************')
print(all_stock.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


'''

1.2 因子构造及预处理

因子构造

我们在此部分会将因子计算涉及的所有行情数据一次性取完并存储，方便后续使用；
我们在此部分会将文中涉及的所有因子都构造完，用于后续的测试比较及深入讨论，具体包括如下几个部分：
part1:为了考察因子对窗口N的敏感性，我们分别构造了20日、40日、60日反转因子(ret_20、ret_40、ret_60)及对应的20日、40日、60日M反转因子(M_high_by_per_value、M_low_by_per_value、M_by_per_value、M_high_40、M_low_40、M_40、M_high_60、M_low_60、M_60)；
part2:为了考察不同切割维度的差异，我们分别构造了用成交金额进行切割的M类因子(M_high_by_value、M_low_by_value、M_by_value)、用成交笔数进行切割的M类因子(M_high_by_deal、M_low_by_deal、M_by_deal)、用平均单笔成交金额进行切割的M类因子(M_high_by_per_value、M_low_by_per_value、M_by_per_value)；
part3:为了考察不同高低分组比例下的M因子的效果，我们在60的观察窗口期下分别考察了将单笔成交金额大的X个交易日作为高D组，剩余60-X个交易日作为低D组的情形下因子的选股能力，即(M_x=10，M_x=12，…，M_x=50)。
因子预处理
在得到原始因子后，考虑到需要计算IC值，我们对每个因子进行如下处理：

Step1: 采用MAD（Median Absolute Deviation 绝对中位数法）进行边界压缩处理，剔除异常值；
Step2: 对第二步的残差项做z-score标准化处理；
Step3: 计算每个股票的次月收益数据备用。
factor_csv就是做完预处理后的标准化因子数据，格式如下：（列数太多截取部分）


'''

# 所有行情数据读取及存储
tic = time.time()
all_code = sorted(all_stock['code'].unique())
pre_date = cal.advanceDate(start_date, '-60B').strftime('%Y%m%d')
mkt_info = DataAPI.MktEqudGet(secID=all_code, beginDate=pre_date,endDate=end_date,isOpen="1",
                              field=u"secID,tradeDate,chgPct,turnoverValue,dealAmount",pandas="1")
mkt_info['tradeDate'] = mkt_info['tradeDate'].apply(lambda x: x.replace('-', ''))
mkt_info['per_value'] = mkt_info['turnoverValue'] / mkt_info['dealAmount']
# 存储为pkl
with open(path + 'mkt_info_frame.pkl', 'wb') as f:
    pickle.dump(mkt_info, f, pickle.HIGHEST_PROTOCOL)
    
toc = time.time()
print('***********行情数据示例************')
print(mkt_info.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''
注意事项 

 因为上述取行情数据占用了不少内存空间，而因子计算部分要用到多进程优化，消耗资源较大，需要在执行以下代码前重启研究环境以释放资源

之前的数据都进行了存储，可以直接运行下面的代码而不需要重跑之前的代码

重启研究环境的步骤为：

网页版：先点击左上角的“Notebook”图标，然后点击左下角的“内存占用x%”图标，在弹框中点击重启研究环境
客户端：点击左下角的“内存x%”, 在弹框中点击重启研究环境
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''


# coding: utf-8

import pandas as pd
import numpy as np
import os
import time
import multiprocessing
from CAL.PyCAL import * 
import statsmodels.api as sm
import gevent
import datetime as dt
import cPickle as pickle
import seaborn as sns
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from quartz_extensions.SignalAnalysis.tears import analyse_return, analyse_monthly_return, analyse_IC, analyse_construction, analyse_general
cal = Calendar('China.SSE')

start_date = '20100101'
end_date = '20190101'
path = 'reverse_improve/'
all_stock = pd.read_csv(path + 'stock_pool.csv', dtype={"date": np.str})
with open(path + 'mkt_info_frame.pkl') as f:
    mkt_info = pickle.load(f)
    
    
# 所有因子计算
def cal_ret(data, N):
    """
    传统反转因子计算
    参数:
        data: 单个股票过去N个交易日的交易数据
        N: 计算窗口期长度
    返回: 
        cum_ret: float，区间累计收益
    """    
    df = data.copy()
    if len(df) < N / 2:
        return np.nan
    else:
        cum_ret = reduce(lambda x,y: (1 + x)*(1 + y) - 1, list(df['chgPct']))
    return cum_ret

def cal_w_ret(data, N, rank_factor):
    """
    W切割下的反转因子计算
    参数:
        data: 单个股票过去N个交易日的交易数据
        N: 计算窗口期长度
        rank_factor：str，排序的列名，成交金额/成交笔数/平均单笔成交金额
    返回: 
        cum_ret_high: float，高组别的区间累计收益
        cum_ret_low: float，低组别的区间累计收益
    """     
    df = data.copy()
    if len(df) < N / 2:
        cum_ret_high = np.nan
        cum_ret_low = np.nan
    else:
        df.sort_values(by=rank_factor, ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)            
        num = len(df) / 2
        # M_high
        df_h = df[0: num]
        cum_ret_high = reduce(lambda x,y: (1 + x)*(1 + y) - 1, list(df_h['chgPct']))
        # M_low
        df_l = df[num: ]
        cum_ret_low = reduce(lambda x,y: (1 + x)*(1 + y) - 1, list(df_l['chgPct']))           
    return cum_ret_high, cum_ret_low

def cal_m_by_percent(data, days, rank_factor):
    """
    不同分组比例下的W切割反转因子，只针对60天窗口期的情况
    参数:
        data: 单个股票过去N个交易日的交易数据
        days: 高组别天数序列，[10, 12, 14,..., 50]
        rank_factor：str，排序的列名，成交金额/成交笔数/平均单笔成交金额
    返回: 
        factor_list:单个股票在不同组别下的M因子值序列
    """
    df = data.copy()
    if len(df) > 30:
        df.sort_values(by=rank_factor, ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        factor_list = []
        for x in days:
            num = len(df) * x / 60 # 按实际交易天数比例分组
            # M_high
            df_h = df[0: num]
            cum_ret_high = reduce(lambda x,y: (1 + x)*(1 + y) - 1, list(df_h['chgPct']))
            # M_low
            df_l = df[num: ]
            cum_ret_low = reduce(lambda x,y: (1 + x)*(1 + y) - 1, list(df_l['chgPct'])) 
            cum_ret_diff = cum_ret_high - cum_ret_low
            factor_list.append(cum_ret_diff)
    else:
        factor_list = [np.nan] * len(days)
    return factor_list

def cal_factor_one_day(args):
    """
    单日的所有股票因子值计算
    参数:
        date: 月末日期
        pre_20: 往前推20个交易日的日期
        pre_40: 往前推40个交易日的日期
        stock_pool: 对应date这一天的股票池数据
        mkt_info: date向前推60天的所有股票的市场行情数据
    返回: 
        all_factor: 单期所有股票的所有因子值数据，DataFrame，列名为['tradeDate', 'secID', 因子名1，因子名2，...]
    """
    date, pre_20, pre_40, stock_pool, mkt_info = args
    # 20日的因子集合（ret_20/按照成交金额划分/按照成交笔数划分/按照单笔成交金额划分）
    N = 20
    period = mkt_info[mkt_info['tradeDate'] >= pre_20]
    period = period[period['secID'].isin(list(stock_pool['code']))]        
    ret_20 = period.groupby(['secID']).apply(lambda x: cal_ret(x, N)).reset_index(name='ret_20')
    ret_20['tradeDate'] = date
    ret_20 = ret_20[['tradeDate', 'secID', 'ret_20']]
    # 成交金额划分
    ret_cut_by_value = period.groupby(['secID']).apply(lambda x: cal_w_ret(x, N, 'turnoverValue')).reset_index(name='temp') 
    ret_cut_by_value['M_high_by_value'] = ret_cut_by_value['temp'].apply(lambda x: x[0])
    ret_cut_by_value['M_low_by_value'] = ret_cut_by_value['temp'].apply(lambda x: x[1])
    ret_cut_by_value['M_by_value'] = ret_cut_by_value['M_high_by_value'] - ret_cut_by_value['M_low_by_value']
    del ret_cut_by_value['temp']
    # 成交笔数划分
    ret_cut_by_deal = period.groupby(['secID']).apply(lambda x: cal_w_ret(x, N, 'dealAmount')).reset_index(name='temp') 
    ret_cut_by_deal['M_high_by_deal'] = ret_cut_by_deal['temp'].apply(lambda x: x[0])
    ret_cut_by_deal['M_low_by_deal'] = ret_cut_by_deal['temp'].apply(lambda x: x[1])
    ret_cut_by_deal['M_by_deal'] = ret_cut_by_deal['M_high_by_deal'] - ret_cut_by_deal['M_low_by_deal']
    del ret_cut_by_deal['temp']
    # 单笔成交金额划分
    ret_cut_by_per_value = period.groupby(['secID']).apply(lambda x: cal_w_ret(x, N, 'per_value')).reset_index(name='temp') 
    ret_cut_by_per_value['M_high_by_per_value'] = ret_cut_by_per_value['temp'].apply(lambda x: x[0])
    ret_cut_by_per_value['M_low_by_per_value'] = ret_cut_by_per_value['temp'].apply(lambda x: x[1])
    ret_cut_by_per_value['M_by_per_value'] = ret_cut_by_per_value['M_high_by_per_value'] - ret_cut_by_per_value['M_low_by_per_value']
    del ret_cut_by_per_value['temp']
    # 20日因子整合
    all_factor_20 = ret_20.merge(ret_cut_by_value,on='secID').merge(ret_cut_by_deal,on='secID').merge(ret_cut_by_per_value,on='secID')

    # 40日的因子集合（ret_40/按照单笔成交金额划分）
    N = 40
    period = mkt_info[mkt_info['tradeDate'] >= pre_40]
    period = period[period['secID'].isin(list(stock_pool['code']))] 
    ret_40 = period.groupby(['secID']).apply(lambda x: cal_ret(x, N)).reset_index(name='ret_40')
    # 单笔成交金额划分
    ret_40_m = period.groupby(['secID']).apply(lambda x: cal_w_ret(x, N, 'per_value')).reset_index(name='temp') 
    ret_40_m['M_high_40'] = ret_40_m['temp'].apply(lambda x: x[0])
    ret_40_m['M_low_40'] = ret_40_m['temp'].apply(lambda x: x[1])
    ret_40_m['M_40'] = ret_40_m['M_high_40'] - ret_40_m['M_low_40']           
    del ret_40_m['temp']
    all_factor_40 = pd.merge(ret_40, ret_40_m, on='secID')

    # 60日的因子集合
    N = 60
    period = mkt_info.copy()
    period = period[period['secID'].isin(list(stock_pool['code']))]       
    ret_60 = period.groupby(['secID']).apply(lambda x: cal_ret(x, N)).reset_index(name='ret_60')
    # 单笔成交金额划分
    ret_60_m = period.groupby(['secID']).apply(lambda x: cal_w_ret(x, N, 'per_value')).reset_index(name='temp') 
    ret_60_m['M_high_60'] = ret_60_m['temp'].apply(lambda x: x[0])
    ret_60_m['M_low_60'] = ret_60_m['temp'].apply(lambda x: x[1])
    ret_60_m['M_60'] = ret_60_m['M_high_60'] - ret_60_m['M_low_60']           
    del ret_60_m['temp']
    # 高低分组比例
    days = range(10, 52, 2)
    ret_percent = period.groupby(['secID']).apply(lambda x: cal_m_by_percent(x, days, 'per_value')).reset_index(name='temp')
    for i in range(len(days)):
        ret_percent['M_x=' + str(days[i])] = ret_percent['temp'].apply(lambda x: x[i])
    del ret_percent['temp']
    all_factor_60 = ret_60.merge(ret_60_m, on='secID').merge(ret_percent, on='secID')
    # 全部整合
    all_factor = all_factor_20.merge(all_factor_40, on='secID', how='left').merge(all_factor_60, on='secID', how='left')
    return all_factor

# 多进程计算
tic = time.time()
date_list = sorted(all_stock['date'].unique())
arg_list = []
for date in date_list:
    month_end = cal.advanceDate(date, '-1B').strftime("%Y%m%d")
    pre_20 = cal.advanceDate(date, '-20B').strftime("%Y%m%d")
    pre_40 = cal.advanceDate(date, '-40B').strftime("%Y%m%d")
    s_date = cal.advanceDate(date, '-60B').strftime("%Y%m%d")
    period_data = mkt_info[(mkt_info['tradeDate'] >= s_date) & (mkt_info['tradeDate'] <= month_end)]
    single_pool = all_stock[all_stock['date'] == date]
    arg_list.append((month_end, pre_20, pre_40, single_pool, period_data))
pool = multiprocessing.Pool(4)
res = pool.map(cal_factor_one_day, arg_list)
pool.close()
pool.join()

rev_factor = pd.concat(res)
rev_factor.to_csv(path + 'raw_factor.csv', index=False)
toc = time.time()
print('***********原始因子数据示例************')
print(rev_factor.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


# 添加单期收益数据
if __name__ == "__main__":
    start_time = time.time()

    # 拿到交易日历，得到月末日期
    trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
    trade_date = trade_date[trade_date.isMonthEnd == 1]

    ########################## 取得个股的行情数据 ################################
    print("\nbegin to get price ratio for stocks and index ...")
    # 个股绝对涨幅
    chgframe = DataAPI.MktEqumAdjGet(beginDate=start_date, endDate=end_date, field=['ticker', 'endDate', 'chgPct', 'return'], pandas="1")
    chgframe['endDate'] = chgframe['endDate'].apply(lambda x: x.replace("-", ""))

    ################################ 对齐数据 ################################
    print("begin to align data ...")
    # 得到月度关系
    month_frame = trade_date[['calendarDate', 'isOpen']]
    month_frame['prev_month_end'] = month_frame['calendarDate'].shift(1)
    month_frame = month_frame[['prev_month_end', 'calendarDate']]
    month_frame.columns = ['month_end', 'next_month_end']
    month_frame.dropna(inplace=True)
    month_frame['month_end'] = month_frame['month_end'].apply(lambda x: x.replace("-", ""))
    month_frame['next_month_end'] = month_frame['next_month_end'].apply(lambda x: x.replace("-", ""))

    # 对齐月度关系
    factor_frame = rev_factor.merge(month_frame, left_on=['tradeDate'], right_on=['month_end'], how='left')

    # 得到个股下个月的涨幅数据
    factor_frame['ticker'] = factor_frame['secID'].apply(lambda x: str(x.split('.')[0]))
    factor_frame = factor_frame.merge(chgframe, left_on=['ticker', 'next_month_end'], right_on=['ticker', 'endDate'], how='left')
    del factor_frame['month_end']
    del factor_frame['endDate']
    
    print('因子数据格式为')
    print(factor_frame.head(5).to_html())
    end_time = time.time()
    print ("Time cost: %s seconds" % (end_time - start_time))
    
def winsorize_by_date(cdate_input):
    '''
    按照[dm+5*dm1, dm-5*dm1]进行去极值
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
  
def winsorize_standardize_factor(input_data):
    """
    进行去极值标准化
    输入：
        input_data：tuple, 传入的是(因子值，时间)。因子值为DataFrame
    返回：
        DataFrame, 处理后的数据
    """
    data, tdate = input_data
    alpha_factors = [x for x in data.columns if x not in ['tradeDate', 'secID', 'next_month_end', 'ticker', 'chgPct', 'return']]
    cdate_input = data[data['tradeDate'] == tdate]
    cdate_input = cdate_input.set_index('ticker')
        
    cdate_input.loc[:, alpha_factors] = cdate_input.loc[:, alpha_factors].apply(lambda x : winsorize_by_date(x))
    cdate_input.loc[:, alpha_factors] = cdate_input.loc[:, alpha_factors].apply(lambda x : standardize(x))
                    
    return cdate_input


if __name__ == "__main__":
    tic = time.time()
    
    # 遍历每个月末日期，利用协程对因子进行去极值、标准化处理
    print('winsorize & standardize factor...')
    date_list = factor_frame['tradeDate'].unique()
    jobs = [gevent.spawn(winsorize_standardize_factor, value) for value in zip([factor_frame]*len(date_list), date_list)]
    gevent.joinall(jobs)
    new_frame_list = [e.value for e in jobs]
    print ("ALL FINISHED")
    
    factor_csv = pd.concat(new_frame_list, axis=0)
    factor_csv.reset_index(inplace=True)

    ################################ 数据存储下来 ################################
    factor_csv.to_csv(path + 'standard_frame.csv', index=False)
    print('***********标准化因子数据示例************')
    print(factor_csv.head(10).to_html())
    toc = time.time()
    print ("Time cost: %s seconds" % (toc - tic))
    
'''
    
第二部分：反转因子的W式切割
该部分耗时 20分钟
该部分内容为：

反转因子的切割问题：反转因子如何改进？

反转因子的W式切割

理想反转因子

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)
 注意事项 

以下的多空组合的年化收益与波动率的计算我们采取多头-空头组合的收益；而优矿的因子收益回测接口中是以(多头-空头)/2来作为每日的多空收益的，本文我们将优矿接口的结果乘以2倍处理，统一以上述做法为准。



2.1 反转因子的切割问题：反转因子如何改进？

传统反转因子
众所周知，A股市场呈现较为显著的中长期反转效应。以20日收益率因子(ret_20)为例，经过我们的测试，从2011年6月初至2018年11月底期间：

因子月度的IC均值为-0.059，rankIC的均值为-0.073；
将其作为反转因子对股票进行排序分组，五分位组多空对冲的信息比率为1.24，月度胜率为66.67%。
稳定性问题
尽管传统反转因子的测试结果似乎具备一定的选股能力，但是令人遗憾的是，反转效应的稳定性不是很理想，从反转因子的多空对冲净值曲线上可以看到，至少对于2013年上半年、2014年下半年和2017年这些时段，反转因子基本失效，市场甚至呈现为动量效应，对40日收益率因子(ret_40)和60日收益率因子(ret_60)的考查，也存在类似的结论。

稳定性问题的改进—切割
传统反转因子在稳定性上的困难，将我们的思路引向“反转因子的切割问题”。思维的过程是这样展开的。首先，我们注意到，传统反转因子本质上是一段区间的涨跌幅，因此可以很自然地拆分为许多更小的时段，那么，会不会存在这样的情况：组成传统反转因的各个时段中，某些时段贡献了很强的反转，而剩余时段只是贡献了很弱的反转，甚至可能是贡献了动量的效果？


上图，提供了切割问题的一个直观图景。我们将传统收益率因子想象为一个柱状体，它的重心(橘色圆点)处于水位线下方，其寓意是“传统收益率因子呈现反转效应”。所谓切割问题，就是说我们能否找到一个好的切割方案，使得柱状体被分割为蓝色因子和红色因子两个部分呢？在这个理想的切割方案下：蓝色因子的重心(蓝色圆点)处于水位线下方更深处，也就是呈现为更强而有效的反转因子；红色因子的重心(红色圆点)则略高于水位线，呈现为弱的动量因子。

'''

# IC测试及多空净值函数
def cal_ic(factor_name, data, method, mode=False):
    """
    Args:
        factor_name: 需要计算IC的因子名称
        data: 因子数据，至少需要3列：日期、股票因子值、股票下期收益('Month_ret')
        method:计算IC的方法，pearson/spearman
        mode: 取值为True或者False,True代表舍弃最后一期计算IC（最后一期的未来月度收益可能没有），False不舍弃
    Returns:
        IC_data: IC结果，dataframe格式，列名为日期，IC值
    """
    IC = []
    date_list = sorted(list(set(data['tradeDate'])))
    if mode:
        date_list = date_list[: -1]
    
    IC_data = data.groupby(['tradeDate']).apply(lambda x: x[[factor_name, 'chgPct']].corr(method=method).values[0, 1])

    IC_data.name = 'IC_'+factor_name
    IC_data = IC_data.reset_index()
    return IC_data

def long_short_curve(data, factor_name):
    """
    计算多空对冲组合净值
    参数:
        data: 因子数据
        factor_name: 因子名
    返回: 
        ret: 组合收益，DataFrame，列名分别为日期、多头组合收益、空头组合收益
    """
    df = data.copy()
    df = df[['tradeDate', 'secID', factor_name, 'chgPct']]
    date_list = sorted(data['tradeDate'].unique())
    long_ret = []
    short_ret = []
    for date in date_list:
        tdata = data[data['tradeDate'] == date]
        tdata.dropna(inplace=True)
        tdata.sort_values(by=factor_name, inplace=True)
        tdata.reset_index(drop=True, inplace=True)
        num = len(tdata) / 5
        long = tdata[0: num]
        long_ret.append(long['chgPct'].mean())
        short = tdata[-num: ]
        short_ret.append(short['chgPct'].mean())
    
    ret = pd.DataFrame({"date": date_list, "long_ret": long_ret, "short_ret": short_ret})
    return ret

tic = time.time()
# 传统反转因子的测试
factor_name = 'ret_20'
ic_raw_s = cal_ic(factor_name, factor_csv, 'spearman', True)
ic_raw_p = cal_ic(factor_name, factor_csv, 'pearson', True)
ic_mean_s = [ic_raw_s['IC_' + factor_name].mean()]
ic_mean_p = [ic_raw_p['IC_' + factor_name].mean()]
icir = [abs(np.sqrt(12) * ic_raw_s['IC_' + factor_name].mean() / ic_raw_s['IC_' + factor_name].std())]
ic_win = [len(ic_raw_s[ic_raw_s['IC_' + factor_name] < 0]) / (len(ic_raw_s) + 0.0)]
ic_count = pd.DataFrame({u'因子': [factor_name], u'IC均值': ic_mean_p, u'rankIC均值': ic_mean_s, u'信息比率': icir,
                        u'月度胜率': ic_win})
ic_count = ic_count[[u'因子', u'IC均值', u'rankIC均值', u'信息比率', u'月度胜率']]
print('***********传统反转因子的IC指标************')
print(ic_count.round(4).to_html())
# 传统反转因子(ret_20)多空净值曲线
ls_ret_20 = long_short_curve(factor_frame, 'ret_20')
ls_ret_20 = ls_ret_20[: -1]
ls_ret_20['hedge'] = ls_ret_20['long_ret'] - ls_ret_20['short_ret']
ls_ret_20['capital'] = (1 + ls_ret_20['hedge']).cumprod()
# 多空收益图
ls_ret_20['datetime'] = pd.to_datetime(ls_ret_20['date'])
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(111)
ax1.plot(ls_ret_20['datetime'], ls_ret_20['capital'], label='long_short')
ax1.set_ylabel(u'capital', fontsize=16)
ax1.set_xlabel(u'date', fontsize=16)
title = u'多空净值曲线'
_ = ax1.set_title(title, fontproperties=font, fontsize=16)
plt.grid(True)
annual = ls_ret_20['hedge'].mean() * 12
vol = ls_ret_20['hedge'].std() * np.sqrt(12)
ir = annual / vol
win_percent = 100 *len(ls_ret_20[ls_ret_20['hedge'] > 1e-8]) / (len(ls_ret_20) + 0.0)
print ("多空对冲信息比率为: %.2f" % ir)
print ("多空对冲的月度胜率为: %.2f%%" % win_percent)
toc = time.time()
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''
2.2 反转因子的W式切割

W式切割
经过长期反复的探索，我们找到了一个反转因子的有效切割方案，简称W式切割。具体操作步骤如下：

Step1：在每个月底，对于股票s，回溯其过去N个交易日的数据(为了后续方便处理，N取偶数)；
Step2：对于股票s，逐日计算平均单笔成交金额D(D=当日成交金额/当日成交笔数)，将N个交易日按照D值从大到小排序，前N/2个交易日称为高D组，后N/2个交易日称为低D组；
Step3：对于股票s，将高D组交易日的涨跌幅加总[1],得到因子M_high；将低D组交易日的涨跌幅汇加总，得到因子M_low；
Step4：对于所有股票，分别按照上述流程计算因子值。
W式切割的核心步骤是，按照“单笔成交金额”对交易日进行排序分组，我们下面以20日收益率因子为例，来说明W式切割方案的出色效果：

样本空间为全部A股(剔除ST和上市未满60天的股票)
回测时间为2011年6月至2018年
回测结论：M_high因子是非常强的反转因子(rankIC均值为-0.095),而M_low因子是较弱的动量因子(rankIC均值为0.021),而且M_high在剔除了ret_20后依然是一个非常显著的选股因子(rankIC仍有-0.0684)。

同时我们发现M_high和M_low在回归剔除ret_20之后，一个是强反转，一个是强动量，选股能力大致是对称的。这个结果几乎是必然的，上述水位线的图景能为我们提供判断的直觉：所谓“回归剔除ret_20因子”的操作，实际上是将水位线从原来的位置调整到橘色圆点所在的高度；显然，红色圆点与蓝色圆点关于调整后的水位线是上下对称的。

PS:[1]处说的加总实际上是通过累乘来实现的，是计算的复合涨跌幅。

'''

# 反转因子的W式切割
def linear_reg(data, y_name_list, x_name_list):
    """
    计算多空对冲组合净值
    参数:
        data: 单期因子数据
        y_name_list: 回归的因变量名列表
        x_name_list: 回归的自变量名列表
    返回: 
        df: 包含回归后残差的单期因子数据
    """
    df = data.copy()
    df.dropna(subset=y_name_list + x_name_list, how='any', inplace=True)
    x = np.array(df[x_name_list])
    x = sm.add_constant(x, has_constant='add')
    for y_name in y_name_list:
        y = np.array(df[y_name])
        model = sm.OLS(y, x)
        results = model.fit()
        df[y_name + '_residual'] = results.resid
        
    return df

# 回归
reg_data = factor_csv[['tradeDate', 'secID', 'ret_20', 'M_high_by_per_value', 'M_low_by_per_value', 'chgPct']]
reg_data = reg_data.groupby(['tradeDate']).apply(lambda x: linear_reg(x, 
                                     ['M_high_by_per_value','M_low_by_per_value'], ['ret_20'])).reset_index(drop=True)
# 对比
compare_list = ['M_high_by_per_value', 'M_low_by_per_value', 'ret_20']
ic_mean_s = []
ic_mean_p = []
ic_ir = []
ic_win = []
residual_ic = []
for factor_name in compare_list:
    ic_raw_s = cal_ic(factor_name, reg_data, 'spearman', True)
    ic_raw_p = cal_ic(factor_name, reg_data, 'pearson', True)
    ic_mean_s.append(ic_raw_s['IC_' + factor_name].mean())
    ic_mean_p.append(ic_raw_p['IC_' + factor_name].mean())
    ic_ir.append(abs(np.sqrt(12) * ic_raw_s['IC_' + factor_name].mean() / ic_raw_s['IC_' + factor_name].std()))
    ic_win.append(len(ic_raw_s[ic_raw_s['IC_' + factor_name] < 0]) / (len(ic_raw_s) + 0.0))
    if factor_name != 'ret_20':
        tmp = cal_ic(factor_name + '_residual', reg_data, 'spearman', True)
        residual_ic.append(tmp['IC_' + factor_name + '_residual'].mean())
    else:
        residual_ic.append(np.nan)
            
ic_count = pd.DataFrame({u'因子': compare_list, u'IC均值': ic_mean_p, u'rankIC均值': ic_mean_s, u'信息比率': ic_ir,
                        u'月度胜率': ic_win, u'回归剔除Ret20后的rankIC均值': residual_ic})
ic_count = ic_count[[u'因子', u'IC均值', u'rankIC均值', u'信息比率', u'月度胜率', u'回归剔除Ret20后的rankIC均值']]
print('**************************************因子对比**************************************')
print(ic_count.round(4).to_html())

'''
2.3 理想反转因子

经过前文的讨论，我们提出了一个理想的反转因子M，其定义式如下：M = M_high - M_low

从定义式我们很容易预判，由于M_high是强反转、M_low是弱动量，M因子大概率会是更强的反转因子。我们通过优矿的回测接口对M因子进行回测，历史回测显示，对于全部A股(剔除ST和上市未满60日的股票)，在2011年6月初至2018年11月底期间，测试结果如下：

M因子的rankIC月度均值为-0.081，从柱形图上可以看到，IC的序列非常稳定，负向IC的期数占比达到了82.22%；
五分组的净值曲线排序良好(完全线性)，且多头组合Q5与其他四组有更大的区分度，多空对冲的年化收益为20.28%，年化波动率7.38%，信息比率高达2.75，从分年度上来看，每个年度的多空受益均为正值(优矿默认正向，用Q1-Q5，实际Q5才是该因子的多头，因子调换方向即可)，非常稳定。
'''

# 理想反转因子的IC测试
tic = time.time()
signal_df = factor_csv[['tradeDate', 'secID', 'M_by_per_value']]
signal_df['ticker'] = signal_df['secID'].apply(lambda x: str(x.split('.')[0]))
signal_df.rename(columns={"tradeDate": "date", "M_by_per_value": "value"}, inplace=True)
signal_df = signal_df[['date', 'ticker', 'value']]
date_list = sorted(signal_df['date'].unique())
# 调用优矿的因子IC分析框架
signal_ic = analyse_IC(factor_value_frame=signal_df, start_date=date_list[0], end_date=date_list[-1], frequency='month', corr_method='spearman', quantile_num=5, universe='TLQA', benchmark='HS300', decay_list=[1, 2, 3, 4])
toc = time.time()
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

# 理想反转因子的收益率分析
tic = time.time()
signal_return = analyse_return(factor_value_frame=signal_df, start_date=date_list[0], end_date=date_list[-1], frequency='month',        quantile_num=5, weight_type='equal', universe='TLQA', benchmark='TLQA', init_cash=100000000.0, decay_list=[1, 2, 3, 4])
toc = time.time()
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''
多头or空头？

经过前文的讨论，我们提出的理想反转因子M对传统反转因子无论是在IC还是多空组合收益表现上都有巨大提升，但是我们知道，传统的反转因子在空头的表现比多头的表现要好得多，因此我们有必要考察一下我们的理想反转因子M的提升是集中在多头还是空头?
'''

# 多空净值对比
ls_m = long_short_curve(factor_frame, 'M_by_per_value')
ls_m = ls_m[: -1]
ls_m['hedge'] = ls_m['long_ret'] - ls_m['short_ret']
ls_m['capital'] = (1 + ls_m['hedge']).cumprod()
ls_m['long_cap'] = (1 + ls_m['long_ret']).cumprod()
ls_m['short_cap'] = (1 + ls_m['short_ret']).cumprod()
ls_ret_20['long_cap'] = (1 + ls_ret_20['long_ret']).cumprod()
ls_ret_20['short_cap'] = (1 + ls_ret_20['short_ret']).cumprod()
# 绘图
ls_m['datetime'] = pd.to_datetime(ls_m['date'])
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(111)
ax1.plot(ls_m['datetime'], ls_m['capital'], label='M_reverse')
ax1.plot(ls_m['datetime'], ls_ret_20['capital'], label='ret_20')
ax1.set_ylabel(u'capital', fontsize=16)
ax1.set_xlabel(u'date', fontsize=16)
title = u'多空净值曲线对比'
plt.legend()
_ = ax1.set_title(title, fontproperties=font, fontsize=16)
plt.grid(True)

# 多头净值对比
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(111)
ax1.plot(ls_m['datetime'], ls_m['long_cap'], label='M_reverse')
ax1.plot(ls_m['datetime'], ls_ret_20['long_cap'], label='ret_20')
ax1.set_ylabel(u'capital', fontsize=16)
ax1.set_xlabel(u'date', fontsize=16)
title = u'多头净值曲线'
plt.legend()
_ = ax1.set_title(title, fontproperties=font, fontsize=16)
plt.grid(True)

# 空头净值对比
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(111)
ax1.plot(ls_m['datetime'], ls_m['short_cap'], label='M_reverse')
ax1.plot(ls_m['datetime'], ls_ret_20['short_cap'], label='ret_20')
ax1.set_ylabel(u'capital', fontsize=16)
ax1.set_xlabel(u'date', fontsize=16)
title = u'空头净值曲线'
plt.legend()
_ = ax1.set_title(title, fontproperties=font, fontsize=16)
plt.grid(True)


# 提升比例
percent_ls = 100 * (list(ls_m['capital'])[-1] / list(ls_ret_20['capital'])[-1] - 1)
percent_lo = 100 * (list(ls_m['long_cap'])[-1] / list(ls_ret_20['long_cap'])[-1] - 1)
percent_so = 100 * (list(ls_m['short_cap'])[-1] / list(ls_ret_20['short_cap'])[-1] - 1)

print ("多空提升比例: %.2f%%" % percent_ls)
print ("多头提升比例: %.2f%%" % percent_lo)
print ("空头提升比例: %.2f%%" % percent_so)

'''
结论

不论是从图像还是从净值的提升比例来看，理想反转因子M在多空两侧相比与传统反转因子均有很大提升，而且在多头的提升幅度还大于空头！这打消了我们关于理想反转因子是否集中提升在空头侧的疑虑，因而将理想反转因子替代传统反转因子加入多因子模型必然会提升量化策略的效果！

第三部分：理想反转因子的若干讨论
该部分耗时 25分钟
该部分内容主要是对因子的若干深入讨论，主要分为以下六个部分：

与风格因子的关联

参数N的敏感度

其他分组指标分割下的效果

其他样本空间的情况

因子收益的累积过程

高/低D组的分组比例的影响

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

3.1 与风格因子的关联

与风格因子的相关性
由于M因子是由两个涨跌幅相减得到，我们预判它与传统反转因子的关联会较低，作为一个量价因子，可能与Beta、波动率等因子的关联比较明显。我们下面展示了M因子与风险模型的10个风格因子的相关系数矩阵，可以看到，

M因子(在图中为value)与其余的风格因子的相关性都比较低，前三高的分别是LIQUIDITY(27%)、REVSOL(23%)、BETA(17%)，与其余的风格因子的相关度都在10%以下。
中性化后的因子
我们在横截面上对10个风格因子与行业因子的哑变量进行回归，将中性化后的残差作为新的选股因子。同样，我们调用优矿的因子收益接口进行测试，结果如下：

中性化后的纯净因子，多空对冲的年化收益为13.30%，年化波动率4.08%，信息比率由2.75—>3.26。


'''

# 若干重要的讨论之一：与风格因子的关联
tic = time.time()
factor_with_risk = []
risk_factors = ['BETA', 'MOMENTUM', 'SIZE', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY', 'SIZENL']
for date in date_list:
    tdata = signal_df[signal_df['date'] == date]
    temp = tdata.set_index('ticker')
    # 中性化
    ex_risk_factors = []
    after_neu = neutralize(temp['value'].to_dict(), date, industry_type='SW1', exclude_style_list=ex_risk_factors)
    temp['neu_value'] = np.nan
    temp.loc[after_neu.keys(), 'neu_value'] = after_neu.values()
    # 标准化
    after_standardize = standardize(temp['neu_value'].to_dict())
    temp['std_value'] = np.nan
    temp.loc[after_standardize.keys(), 'std_value'] = after_standardize.values()    
    temp.reset_index(inplace=True)
    # 风险模型数据
    ti = list(temp['ticker'])
    risk_data = DataAPI.RMExposureDayGet(ticker=ti,tradeDate=date,field=['tradeDate', 'ticker'] + risk_factors,pandas="1")
    risk_data.rename(columns={"tradeDate": "date"}, inplace=True)
    tdata = pd.merge(temp, risk_data, on=['date', 'ticker'])
    factor_with_risk.append(tdata)

factor_with_risk = pd.concat(factor_with_risk)
# 因子相关性矩阵
date_list = sorted(factor_with_risk['date'].unique())
mean_corr = pd.DataFrame()
for i in range(len(date_list)):
    tdata = factor_with_risk[factor_with_risk['date'] == date_list[i]]
    col = [x for x in factor_with_risk.columns if x not in ['date', 'ticker', 'neu_value', 'std_value']]
    corr_frame = tdata[col].corr()    
    if i == 0:
        mean_corr = corr_frame
    else:
        mean_corr += corr_frame

mean_corr = mean_corr / len(date_list)
mean_corr = mean_corr.round(2)

f, ax= plt.subplots(figsize = (15, 8))
_ = sns.heatmap(mean_corr, alpha=1.0, annot=True, center=0.0, annot_kws={"size": 8}, linewidths=0.02, 
                linecolor='white', linewidth=0,  ax=ax)
title=u'因子与风格因子间的相关性矩阵'
_ = ax.set_title(title, fontproperties=font, fontsize=16)
toc = time.time()
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


# 中性化后信号的收益率分析
tic = time.time()
neu_signal_df = factor_with_risk[['date', 'ticker', 'std_value']]
neu_signal_df.rename(columns={"std_value": "value"}, inplace=True)
neu_signal_df.reset_index(drop=True, inplace=True)
neu_signal_return = analyse_return(factor_value_frame=neu_signal_df, start_date=date_list[0], end_date=date_list[-1], frequency='month',        quantile_num=5, weight_type='equal', universe='TLQA', benchmark='TLQA', init_cash=100000000.0, decay_list=[1, 2, 3, 4])
toc = time.time()
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''

3.2 参数N的敏感度

本问前面都是以20日收益率为例，来对传统的反转因子进行改进，下面展示了参数N在三种去之下，原始因子与切割后因子的IC值
结论：可以看到，提供的切割方案用于改进40日收益率因子(ret_40)与60日收益率因子(ret_60)，同样效果出色！

'''


# 若干重要的讨论之二：参数N的敏感度
compare_list = ['M_high_by_per_value', 'M_high_40', 'M_high_60', 'M_low_by_per_value', 'M_low_40', 'M_low_60',
               'ret_20', 'ret_40', 'ret_60']

ic_mean = []
for factor_name in compare_list:
    ic_raw = cal_ic(factor_name, factor_csv, 'pearson', True)
    ic_mean.append(ic_raw['IC_' + factor_name].mean())

ic_mean = np.array(ic_mean).reshape((3, 3))
ic_compare = pd.DataFrame(data = ic_mean, index=['M_high', 'M_low', u'原始因子'], columns=['Ret_20', 'Ret_40', 'Ret_60'])
print('**N=20，40，60三种情况下的切割效果**')
print(ic_compare.round(4).to_html())

'''
3.3 其他分组指标分割下的效果

其实本文在进行切割的时候，对分组指标的寻找并非一步到位，我们也可以考察其他分组方式下的效果，比如按“成交金额”或“成交笔数”分组，下面我们比较了不同分组方式下的效果比较(rankIC均值)，总体来看，按照平均单笔成交金额作为分组指标的效果最好！
'''

# 若干重要的讨论之三：其他分组指标分割下的效果
compare_list = ['M_high_by_deal', 'M_high_by_value', 'M_high_by_per_value', 'M_low_by_deal', 'M_low_by_value', 'M_low_by_per_value']

ic_mean = []
for factor_name in compare_list:
    ic_raw = cal_ic(factor_name, factor_csv, 'spearman', True)
    ic_mean.append(ic_raw['IC_' + factor_name].mean())

ic_mean = np.array(ic_mean).reshape((2, 3))
ic_compare = pd.DataFrame(data = ic_mean, index=['M_high', 'M_low'], columns=[u'按成交笔数分组', u'按成交金额分组', u'按单笔成交金额分组'])
print('**************不同分组指标分割下的因子效果**************')
print(ic_compare.round(4).to_html())

'''
3.4 其他样本空间的情况

第二章节测试的时候是以全A为样本空间进行测试的，这部分我们考虑在不同样本空间中(沪深300、中证500)，理想的反转因子M是否同样能有优异的表现呢？


'''

# 若干重要的讨论之三：其他样本空间的情况
def plot_long_short_capital(data, index_ticker):
    """
    样本空间上的多空净值对比
    参数：
        data：因子数据，列名为tradeDate，secID，因子名1，因子名2，...
        index_ticker: 样本空间的代码，如000300，000905
    返回：
        fig：样本空间上的多空净值曲线对比图
        ratio：改进前后的多空对冲指标
    """
    df = []
    date_list = sorted(data['tradeDate'].unique())
    for date in date_list:
        tdata = data[data['tradeDate'] == date]
        bench = DataAPI.IdxCloseWeightGet(ticker=index_ticker, beginDate=date, endDate=date, field=u"consID,weight",pandas="1")
        tdata = tdata[tdata['secID'].isin(list(bench['consID']))]
        df.append(tdata)
        
    df = pd.concat(df)
    ret_old = long_short_curve(df, 'ret_20')
    ret_old['hedge_ret_old'] = ret_old['long_ret'] - ret_old['short_ret']
    ret_old['capital_old'] = (1 + ret_old['hedge_ret_old']).cumprod()
    ret_new = long_short_curve(df, 'M_by_per_value')
    ret_new['hedge_ret_new'] = ret_new['long_ret'] - ret_new['short_ret']
    ret_new['capital_new'] = (1 + ret_new['hedge_ret_new']).cumprod()
    compare = pd.merge(ret_old, ret_new, on='date')
    compare = compare[: -1]
    # 指标
    annual_old = compare['hedge_ret_old'].mean() * 12
    annual_new = compare['hedge_ret_new'].mean() * 12
    annual = [annual_old, annual_new]
    vol_old = compare['hedge_ret_old'].std() * np.sqrt(12)
    vol_new = compare['hedge_ret_new'].std() * np.sqrt(12)
    vol = [vol_old, vol_new]
    ir_old = annual_old / vol_old
    ir_new = annual_new / vol_new
    ir = [ir_old, ir_new]
    ratio = pd.DataFrame({u"因子": [u'原始反转', u'理想反转'], u"年化对冲收益": annual, u"年化波动率": vol, u"信息比率": ir})
    ratio = ratio[[u"因子", u"年化对冲收益", u"年化波动率", u"信息比率"]]
    # 多空收益对比图
    compare['date'] = pd.to_datetime(compare['date'])
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(111)
    ax1.plot(compare['date'], compare['capital_old'], label=u'traditional_ret_20')
    ax1.plot(compare['date'], compare['capital_new'], label=u'M_by_per_value')
    title = u'多空对冲净值曲线'
    _ = ax1.set_title(title, fontproperties=font, fontsize=16)
    plt.grid(True)
    plt.legend()
    return fig, ratio

# 沪深300上面的选股能力对比
tic = time.time()
index_ticker = '000300'
fig, ratio = plot_long_short_capital(factor_csv, index_ticker)
print (ratio.round(4).to_html())
toc = time.time()
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''

在沪深300成分股中：

原始反转因子的五分组多空对冲年化收益6.87%，年化波动21.40%，信息比率0.32；
理想反转因子的五分组多空对冲年化收益18.01%，年化波动12.84%，信息比率1.40；
从如上的净值曲线对比来看，沪深300中，理想反转因子相对原始反转因子的提升非常显著！

'''
# 中证500上面的选股能力对比
tic = time.time()
index_ticker = '000905'
fig, ratio = plot_long_short_capital(factor_csv, index_ticker)
print (ratio.round(4).to_html())
toc = time.time()
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''

在中证500成分股中：

原始反转因子的五分组多空对冲年化收益12.42%，年化波动15.04%，信息比率0.83；
理想反转因子的五分组多空对冲年化收益15.21%，年化波动8.37%，信息比率1.82；
从如上的净值曲线对比来看，在中证500中，理想反转因子相对原始反转因子的提升依旧非常显著！

3.5 因子收益的累积过程

在本文中，因子的回测采用的都是月度调仓。读者可能关心更高频率的交易效果，因此我们下面展示了N=20的时候，理想反转因子在月初建仓后(全市场股票、分五组)，多空对冲收益的累积过程，可以从柱状图看到，在T+0 -->T+20的过程中，收益的累积过程比较均匀，我们定性地判断，可以尝试利用理想反转因子进行周频/半月频调仓。

'''


# 若干重要的讨论之四：因子收益的累积过程
# 多头
bucket_ret_5 = pd.DataFrame(signal_return['absolute_return']['1']['Q5']['accumulated_return_history'], index=[0]).T
bucket_ret_5.columns=['cum_ret_5']
bucket_ret_5['date'] = bucket_ret_5.index
bucket_ret_5.reset_index(drop=True, inplace=True)
# 空头
bucket_ret_1 = pd.DataFrame(signal_return['absolute_return']['1']['Q1']['accumulated_return_history'], index=[0]).T
bucket_ret_1.columns=['cum_ret_1']
bucket_ret_1['date'] = bucket_ret_1.index
bucket_ret_1.reset_index(drop=True, inplace=True)
# merge
bucket_ret = bucket_ret_5.merge(bucket_ret_1, on='date')
bucket_ret['ret_5'] = (1 + bucket_ret['cum_ret_5']).pct_change()
bucket_ret['ret_1'] = (1 + bucket_ret['cum_ret_1']).pct_change()
bucket_ret['hedge_ret'] = bucket_ret['ret_5'] - bucket_ret['ret_1']
bucket_ret = bucket_ret[1: ]
# 累积过程
new_ret = []
for i in range(len(date_list) - 1):
    end = date_list[i]
    next_end = date_list[i + 1]
    period = bucket_ret[(bucket_ret['date'] > end) * (bucket_ret['date'] <= next_end)]
    period.sort_values(by='date', inplace=True)
    period.reset_index(drop=True, inplace=True)
    period['order'] = period.index
    new_ret.append(period)
    
new_ret = pd.concat(new_ret)
# 累积收益过程
cum_ret = new_ret.groupby(['order'])['hedge_ret'].mean().reset_index(name='ret')
cum_ret = cum_ret[0: 21]
cum_ret['cum_ret'] = (1 + cum_ret['ret']).cumprod() - 1
cum_ret['day'] = ['T+' + str(x) for x in cum_ret['order']]
_ax = cum_ret.plot(x='day', y='cum_ret', kind='bar')


'''
3.6 高/低D组的分组比例的影响

在W式切割方案中，高D组与低D组的交易日，各占回溯交易日的一半，也即N/2个。如果调整分组的比例，效果会有多大的区别呢？

实验
我们以N=60为例，将单笔成交金额大的X个交易日作为高D组，将剩余60-X个交易日作为低D组，遍历X的值，分别计算M因子的信息比率，结果如下图所示，X在30附近取值，都有很好的选股效果，这个结论支持了“对半分组”的做法。

'''
# 若干重要讨论之五：高/低D组的分组比例的影响
tic = time.time()
cut_list = []
ir_list = []
for factor_name in factor_csv.columns:
    if len(factor_name.split('=')) == 2:
        temp = long_short_curve(factor_csv, factor_name)
        temp = temp[: -1]
        temp['hedge'] = temp['long_ret'] - temp['short_ret']
        annual = temp['hedge'].mean() * 12
        vol = temp['hedge'].std() * np.sqrt(12)
        ir = annual / vol
        cut_list.append(factor_name.split('_')[1])
        ir_list.append(ir)
    else:
        continue

ir_count = pd.DataFrame({"cut": cut_list, "ir": ir_list})
_ax = ir_count.plot(x='cut', y='ir', kind='line')
toc = time.time()
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''

结论

量化逻辑
在构造因子的时候，我们始终会关心量化模型背后的逻辑。在W式切割中，“按单笔成交金额对交易日进行分组”，似乎在暗示：

对于大单交易活跃(单笔成交金额高)的交易日，涨跌幅因子有更强的反转特性；
对于大单交易不活跃(单笔成交金额低)的交易日，涨跌幅因子有更弱的反转特性(甚至呈现较弱的动量特性)。
总结
本文在对反转因子进行W式切割后构造了理想的反转因子M，M因子对反转因子的改进效果相当出色，M因子rankIC均值为-0.081，五分组净值曲线完全线性，且多头组合与其他四组区分显著，多空对冲组合年化收益20.28%，年化波动为7.38%，月度胜率82.22%，信息比率高达2.75，而且在剔除Barra风格因子和行业因子的影响之后，信息比率提升至3.26，M因子借助成交笔数数据，能完全取代传统反转因子的位置，增强多因子模型的效果。

'''