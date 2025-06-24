# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:04:21 2020

@author: Asus
"""
'''

导读
A. 研究目的：本文利用优矿提供的行情数据和的因子数据，参考广发证券《基于V型处置效应的选股策略研究》（原作者：陈原文等）中的因子计算的方法，构造了CYQ因子，并将对其进行解析。

B. 研究结论：

CYQ因子是对相对资本收益的度量，将相对资本分为盈利资本（Gain因子）和亏损资本（Loss因子），该因子的效果主要来源于亏损资本。

CYQ因子、Gain因子、Loss因子均为负向因子。基于CYQ因子周度、月度回测的结果，CYQ周度收益明显优于月度。且在不同投资域中的多头效应均很明显。Gain因子的分组单调性不佳，Loss因子分组单调。

因为CYQ因子的效果主要来源于Loss因子，但Gain因子、Loss因子有抵消效果。基于此构造出CYQ_reform因子。

基于2007-01-01至2019-03-01周度回测结果，CYQ因子和CYQ_reform因子风险模型中的风格因子的相关性较低。CYQ因子的IC达到-6.28%，年化ICIR为-2.74，扣费后的多空对冲收益有11.55%，夏普比率为1.03。CYQ_reform因子的IC达到-6.32%，年化ICIR为-2.77，扣费后的多空对冲收益有12.40%，夏普比率为1.10。它们在2019年初表现良好。

CYQ_reform因子各项指标略优于CYQ因子，但是提升效果并不明显。

C. 文章结构：本文共分为3个部分，具体如下

一、基础数据、函数准备，并介绍筹码分布原理。

二、计算CYQ因子、Gain因子、Loss因子,并解析它们的关系。回测其在不同投资域、不同调仓频率下的分组回测结果。最后构造CYQ_reform因子。

三、对CYQ因子、CYQ_reform因子进行相关性分析、IC分析、收益分析。最后以CYQ_reform因子构造ZZ800、ZZ500Smart Beta指数。

D. 时间说明

一、第一部分运行需要6分钟
二、第二部分运行需要65分钟
三、第三部分运行需要30分钟
总耗时101左右 (为了方便修改程序，文中将很多中间数据存储下来以缩短运行时间)
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
https://uqer.datayes.com/community/share/DO4eNMu4iULhXmlEd4vDTTvpK700/private；密码：0332
请在运行之前，克隆上面的代码，并存成lib（右上角->另存为lib,不要修改名字）

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


第一部分：基础数据、函数准备
该部分耗时 6分钟
该部分内容为：

1.1 获取原始行情数据，以及基础函数准备。起始时间为2007-01-01, 结束时间为2019-03-01.
1.2 筹码分布介绍。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 行情数据、基础函数准备(6分钟)

'''
import sys
sys.path
sys.path.append('G:\dropbox\Dropbox\Dropbox\project folder from my asua computer\Project\lib')

import pandas as pd
import numpy as np
import time
#from CAL.PyCAL import *
import matplotlib.pyplot as plt
import quant_util as qutil
import statsmodels.api as sm
import scipy.stats as st
import pickle
import seaborn as sns
from multiprocessing.dummy import Pool as ThreadPool
import os
if not os.path.exists('CYQ'):
    os.makedirs('CYQ')
    
    
start_time = time.time()
print ("该部分进行基础参数设置和数据准备...")

# 基础数据
sdate = '20070101'
edate = '20190301'
lookback = '20060101'

from sqlalchemy import create_engine
import json

import warnings



warnings.filterwarnings('ignore')

# must be set before using
with open('para.json', 'r', encoding='utf-8') as f:
    para = json.load(f)

user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name1 = 'yuqerdata'
db_name2 = 'yuqer_cubdata'
eng_str = 'mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name, pass_wd, port, db_name1)
eng_str2 = 'mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name, pass_wd, port, db_name2)
engine = create_engine(eng_str)
engine2 = create_engine(eng_str2)

sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
def get_IdxCons(intoDate,ticker='000300'):
    #nearst 时间
    sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
            and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
        ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' %(ticker,
        ticker,intoDate)
    x = pd.read_sql(sql_str1,engine)
    x = x['symbol'].values   
    return x

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


## 得到月度日历
def get_calender_range(begin, end):
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" and tradeDate >="%s" and tradeDate <="%s" order by tradeDate'''%(begin, end)
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

def get_calender():
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" order by tradeDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x
    
def get_month_calender(start_date, end_date):
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" and endDate>="%s" and endDate <="%s" order by endDate'''%(start_date, end_date)
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

def get_week_calender(start_date, end_date):
    sql_str = '''select endDate from yuqerdata.yq_MktIdxwGet where ticker = "000001" and endDate>="%s" and endDate <="%s" order by endDate'''%(start_date, end_date)
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x


raw_data_dir = "./raw_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

def MktStockFactorsOneDayProGet(tradeDate, fields):
    df = pd.DataFrame()
    i = 1
    for field in fields:
        field = field.lower()
        sql_str1 = '''select symbol, tradingdate, f_val from ''' + field
        sql_str2 = ''' where tradingdate = "%s"''' % (tradeDate)
        sql_str = sql_str1 + sql_str2
        # sql_str = '''select symbol, tradingdate, f_val from '%s' where tradingdate = "%s"'''%(field, tradeDate)
        x = pd.read_sql(sql_str, engine2)
        x = x.rename(columns={"f_val": field})
        if i == 1:
            df = x
        else:
            df = df.merge(x, on=["symbol", "tradingdate"])
        i = i + 1

    return df

def MktEqudAdjGet(beginDate, endDate, universe, fields):
    df = pd.DataFrame()
    k = 1
    for i in universe:
        x = chs_factor(ticker = i, begin = beginDate, end = endDate,field=fields)
    if k == 1:
        df = x
    else:
        df = df.merge(x, on=["ticker","tradingdate"])
    k = k+1
    return df
def MktEqumAdjGet(beginDate, endDate, universe, fields):
    df = pd.DataFrame()
    for i in universe:
        sql_str = '''select ''' +fields
        sql_str +=''' from mktequmadjafget where ticker = "%s" and endDate >="%s" and endDate <= "%s"''' %(i,beginDate, endDate)
        x = pd.read_sql(sql_str,engine)
        df = df.append(x)
    return df

def MktEquwAdjGet(beginDate, endDate, universe, fields):
    df = pd.DataFrame()
    for i in universe:
        sql_str = '''select ''' +fields
        sql_str +=''' from yq_mktequwadjafget where ticker = "%s" and endDate >="%s" and endDate <= "%s"''' %(i,beginDate, endDate)
        x = pd.read_sql(sql_str,engine)
        df = df.append(x)
    return df
    
def MktIdxwGet(beginDate, endDate, index, fields):
    sql_str = '''select ''' + fields
    sql_str += ''' from yq_mktidxwget where indexID = "%s" and endDate >="%s" and endDate <= "%s"''' % (index, beginDate, endDate)
    x = pd.read_sql(sql_str, engine)
    return x
# 全A投资域
#a_universe_list = DataAPI.EquGet(equTypeCD=u"A", field=u"secID",pandas="1")['secID'].tolist()
a_universe_list = get_IdxCons(edate, ticker ='000001')
cal_dates_df = get_calender_range(sdate, edate)
monthly_dates_list = get_month_calender(sdate,edate)
weekly_dates_list = get_week_calender(sdate, edate)
# 获取月末交易日
'''
#cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=sdate, endDate=edate).sort('calendarDate')
monthly_dates_list = cal_dates_df[cal_dates_df['isMonthEnd']==1]['calendarDate'].values.tolist()
monthly_dates_list = map(lambda x: x.replace('-', ''), monthly_dates_list)
weekly_dates_list = cal_dates_df[cal_dates_df['isWeekEnd']==1]['calendarDate'].values.tolist()
weekly_dates_list = map(lambda x: x.replace('-', ''), weekly_dates_list)
'''


# 获取个股月度收益率
mret_df = MktEqumAdjGet(sdate, edate, a_universe_list, fields = "ticker, endDate, chgPct")
mret_df.rename(columns={'endDate':'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
mret_df['tradeDate'] = mret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
mret_df['nxt1m_ret'] = mret_df.groupby('ticker')['curr_ret'].shift(-1)

'''
mret_df = DataAPI.MktEqumAdjGet(beginDate=sdate, endDate=edate, secID=a_universe_list, field=u"ticker,endDate,chgPct", pandas="1")
mret_df.rename(columns={'endDate':'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
mret_df['tradeDate'] = mret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
mret_df['nxt1m_ret'] = mret_df.groupby('ticker')['curr_ret'].shift(-1)
'''
# 获取个股周度收益率

wret_df = MktEquwAdjGet(sdate, edate, a_universe_list, fields = "ticker, endDate, chgPct")
wret_df.rename(columns={'endDate':'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
wret_df['tradeDate'] = wret_df['tradeDate'].astype('str').apply(lambda x: x.replace('-', ''))
wret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
wret_df['nxt1w_ret'] = wret_df.groupby('ticker')['curr_ret'].shift(-1)

'''
wret_df = DataAPI.MktEquwAdjGet(beginDate=sdate, endDate=edate, secID=a_universe_list, field=u"ticker,endDate,chgPct", pandas="1")
wret_df.rename(columns={'endDate':'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
wret_df['tradeDate'] = wret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
wret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
wret_df['nxt1w_ret'] = wret_df.groupby('ticker')['curr_ret'].shift(-1)
'''

# 剔除上市不满250个交易日、st股票和停牌个股
forbidden_pool = qutil.stock_special_tag(sdate, edate, pre_new_length=100)
f_mret_df = mret_df.merge(forbidden_pool, on=['ticker', 'tradeDate'], how='left')
f_mret_df = f_mret_df[f_mret_df['special_flag'].isnull()]
del f_mret_df['special_flag']
f_mret_df.to_pickle('CYQ/f_mret_df.pkl')
f_wret_df = wret_df.merge(forbidden_pool, on=['ticker', 'tradeDate'], how='left')
f_wret_df = f_wret_df[f_wret_df['special_flag'].isnull()]
del f_wret_df['special_flag']
f_wret_df.to_pickle('CYQ/f_wret_df.pkl')

# 个股日收益率
dret_df = MktEqudAdjGet(beginDate=lookback, endDate=edate, universe= a_universe_list, fields=["symbol","tradeDate","closePrice","turnoverRate"])
#print(dret_df)
dret_df['tradeDate'] = dret_df['tradeDate'].astype('str').apply(lambda x: x.replace('-', ''))
dret_df = dret_df.sort_values(['ticker', 'tradeDate'])
dret_df.to_pickle('CYQ/d_ret_df.pkl')

# 指数收益率
index_name = 'ZZ800.ZICN'
#index_symbol = DataAPI.SecIDGet(cnSpell=index_name)['secID'].values[0]
week_idx_ret_df = MktIdxwGet(index=index_name,beginDate=sdate,endDate=edate,fields="endDate,chgPct")
week_idx_ret_df.columns = ['tradeDate','%s_ret' % index_name]
week_idx_ret_df['tradeDate'] = week_idx_ret_df['tradeDate'].apply(lambda x: x.replace('-', ''))

# 风险模型
week_style_factor_list = []
style_factor_list = ['BETA', 'MOMENTUM', 'SIZE', 'SIZENL', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY']
for tdate in weekly_dates_list:
    factor_td_df = RMExposureDayGet(tradeDate=tdate,beginDate=u"",endDate=u"",field=['ticker', 'tradeDate' ]+style_factor_list,pandas="1")
    week_style_factor_list.append(factor_td_df)
week_style_factor_df = pd.concat(week_style_factor_list)
week_style_factor_df.to_pickle('CYQ/week_style_factor_df.pkl')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

def proc_float_scale(df, col_name, format_str):
    """
    格式化输出
    输入：
        df: DataFrame, 需要格式化的数据
        col_name： list, 需要格式化的列名
        format_str： 格式类型
    """
    for col in col_name:
        for index in df.index:
            df.ix[index, col] = format(df.ix[index, col], format_str)
    return df

def ic_describe(ic_df, annual_len):
    """
    统计IC的均值、标准差、IC_IR、大于0的比例
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
    ic_ir = ic_mean/ic_std*np.sqrt(annual_len)
    # IC大于0的比例
    ic_p_pct = (ic_df[factor_name] > 0).sum()/len(ic_df)
    
    # IC统计
    ic_table = pd.DataFrame([ic_mean, ic_std, ic_t, ic_ir, ic_p_pct], index=['平均IC', 'IC标准差', 'IC均值T统计量','IC_IR', 'IC大于0的比例']).T
    ic_table = proc_float_scale(ic_table, ['平均IC', 'IC标准差', 'IC大于0的比例'], ".2%")
    ic_table = proc_float_scale(ic_table, ['IC均值T统计量','IC_IR'], ".2f")
    return ic_table

def perf_describe(perf_df, annual_len):
    """
    统计因子的回测绩效， 包括年化收益率、年化波动率、夏普比率、最大回撤
    params:
            perf_df: DataFrame, 回测的期间收益率， index为日期， columns为因子名， values为因子回测的期间收益率
            annual_len: int, 年化周期数。若是月频结果，则通常为12；若是周频结果，则通常为52
    return:
            DataFrame, 返回回测绩效
    """
    # 记录因子个数和因子名
    factor_name = perf_df.columns.values
    n = len(factor_name)
    
    # 年化收益率
    ret_mean = perf_df.mean()*annual_len
    # 年化波动率
    ret_std = perf_df.std()*np.sqrt(annual_len)
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
    
    perf_table = pd.DataFrame([ret_mean, ret_std, ir, maxdrawdown, win_ret], index=['年化收益率', '年化波动率', '夏普比率', '最大回撤', '月度胜率']).T
    perf_table = proc_float_scale(perf_table, ['年化收益率', '年化波动率', '最大回撤', '月度胜率'], ".2%")
    perf_table = proc_float_scale(perf_table, ['夏普比率'], ".2f")
    perf_table = perf_table.loc[perf_df.columns,:]
    return perf_table

def perf_describe_by_year(perf_df, annual_len=12):
    """
    分年度统计因子的回测绩效， 包括年化收益率、年化波动率、夏普比率、最大回撤
    params:
            perf_df: DataFrame, 回测的期间收益率， index为日期， columns为因子名， values为因子回测的期间收益率
            annual_len: int, 年化周期数。若是月频结果，则通常为12；若是周频结果，则通常为52
    return:
            DataFrame, 返回回测绩效
    """
    syear = int(perf_df.index.min()[:4])
    edate = int(perf_df.index.max()[:4])
    perf_df_table = []
    for year in range(syear, edate+1):
        perf_table_y = perf_describe(perf_df[[str(year) in date for date in perf_df.index.values]], annual_len=annual_len)
        perf_table_y = pd.DataFrame(perf_table_y.unstack()).T
        perf_table_y.index = [year]
        perf_df_table.append(perf_table_y)
    perf_table_all = perf_describe(perf_df, annual_len=annual_len)
    perf_table_all = pd.DataFrame(perf_table_all.unstack()).T
    perf_table_all.index = ['all']
    perf_df_table.append(perf_table_all)
    perf_df_table = pd.concat(perf_df_table)
    return perf_df_table

'''

筹码分布理论，就是根据股票交易筹码流动性的特点，对大盘或个股的历史成交情况进行分析，得到其筹码分布，然后根据这个筹码分布预测其未来的走势。
股票交易都是通过买卖双方在某个价位进行买卖成交而实现的。随着股票的上涨或下跌，在不同的价格区域产生着不同的成交量，这些成交量在不同的价位的分布量，形成了股票不同价位的持仓成本。
一轮行情的发展就是成本转换的过程，即持仓筹码由一个价位向另一个价位转移的过程。股票的走势在表象上体现了股价的变化，而其内在的本质却体现了持仓成本的转换。 图片注释 图片注释
由上图看出,今年年初一轮上涨行情,平安银行的筹码帆布在不断上移.
   调试 运行
文档
 代码  策略  文档
根据前景理论对人们风险决策行为的描述，处于收益状态时，人们往往小心翼翼、厌恶风险喜欢见好就收；处于亏损状态时，人们往往会极不甘心，宁愿承受更大的风险来赌一把。
因为前景理论，形成了股票市场上的处置效应：投资者急于卖出盈利的股票，而不愿意卖出亏损的股票；损失股票的持有时间比收益股票的持有时间长。
从股价角度来看，当绝大多数投资者盈利卖出股票时，对股价产生下行压力，未来股价大概率下行；当绝大多数投资者亏损，当亏损幅度过大时，会对股价形成支撑，未来股价大概率反弹。
基于上述逻辑，本文利用筹码理论来计算个股的投资者平均的盈利（亏损）程序，以此来描述股价的超买超卖程度。
   调试 运行
文档
 代码  策略  文档
第二部分：CYQ因子构造和解析
该部分耗时 65分钟
该部分内容为：

2.1 利用行情数据，计算CYQ因子、Gain因子、Loss因子。
2.2 解析CYQ因子。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
2.1 CYQ因子、Gain因子、Loss因子计算(62分钟)

给定任意一只股票，其在t日的成交价为Pt, 在t-1日的成交价为Pt−1, 同理在t-n日的成交价为Pt−n。

则在任意t-n日，该股票的相对资本收益为
RCt−n=Pt−Pt−nPt
再进行分解，盈利的相对资本收益为
gaint−n=Pt−Pt−nPt∗1{Pt≥Pt−n}

亏损的相对资本收益为
losst−n=Pt−Pt−nPt∗1{Pt≤Pt−n}
股票在t日的换手率为Vt, 在t-n的换手率为Vt−n。则在t-n日的调整换手率为
wt−n=1kVt−n∏i=1n−1[1−Vt−n+i]
 
k=∑nVt−n∏i=1n−1[1−Vt−n+i]
因为越久远的成交信息对投资者当前时刻的参考意义越小，越近的成交信息参考意义越大，所以将调整换手率作为相对资本的加权权重。计算得到CYQ因子、Gain因子、Loss因子
CYQt=∑n=1∞wt−nRCt−n
 
Gaint=∑n=1∞wt−ngaint−n
 
Losst=∑n=1∞wt−nlosst−n
实际上，CYQ因子=Gain因子+Loss因子。参考研报，我们取n=100。

具体算法举例说明如下。
假设t时刻的成交价为48.5元，则Gain因子和Loss因子的计算过程如下表所示。
图片注释
（截图来自广发证券研报）

下面开始计算周度和月度的CYQ因子、Gain因子、Loss因子，其中月度因子耗时12分钟，周度因子计算耗时50分钟，并将数据存储到201903/cyq_week.pkl和201903/cyq_month.pkl中。

'''

def cal_cyq(df, n):
    if len(df) < n/2.0:
        return [np.nan] * 3
        
    # 计算cqy、gain、loss
    p = df['closePrice'].values
    p_chg = (p[-1] - p[:-1]) /p[-1]
    rc = p_chg
    gain = np.where(p_chg>=0, p_chg, 0)
    loss = np.where(p_chg>=0, 0, p_chg)
    # 计算权重
    w = df['turnoverRate'].values[:-1]
    w_1 = 1-w
    weight = [np.prod(w_1[i+1:]) for i in range(len(w_1))] * w
    weight = weight / sum(weight)

    # 计算VNSP
    CYQ = sum(rc*weight)
    Gain = sum(gain*weight)
    Loss = sum(loss*weight)
    return [CYQ, Gain, Loss]

def calc_by_date(params, n=100):
    """
    按日计算CYQ等因子
    """
    mkt_df, date = params
    tmp_mkt_df = mkt_df[mkt_df['tradeDate'] <= date]
    tmp_mkt_df = tmp_mkt_df.sort_values(['ticker', 'tradeDate'])
    tmp_mkt_df = tmp_mkt_df.groupby('ticker').tail(n+1)
    vnsp_df = tmp_mkt_df.groupby('ticker').apply(cal_cyq, n)
    factor_df = pd.DataFrame(vnsp_df.values.tolist(), columns=['CYQ', 'Gain', 'Loss'], index=vnsp_df.index)
    factor_df['tradeDate'] = date
    return factor_df.reset_index().dropna()

start_time = time.time()
print ("该部分计算月度CYQ因子、Gain因子、Loss因子...")

nday = len(monthly_dates_list)
pool = ThreadPool(processes=16)
pool_args = zip([dret_df] * nday, monthly_dates_list)
frame_list = pool.map(calc_by_date, pool_args)
pool.close()
pool.join()
cyq_month_df = pd.concat(frame_list)
cyq_month_df = cyq_month_df.merge(dret_df[['ticker', 'tradeDate']], on=['ticker', 'tradeDate'])
cyq_month_df.to_pickle('CYQ/cyq_month.pkl')

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


start_time = time.time()
print ("该部分计算周度CYQ因子、Gain因子、Loss因子...")

nday = len(weekly_dates_list)
pool = ThreadPool(processes=16)
pool_args = zip([dret_df] * nday, weekly_dates_list)
frame_list = pool.map(calc_by_date, pool_args)
pool.close()
pool.join()
cyq_week_df = pd.concat(frame_list)
cyq_week_df = cyq_week_df.merge(dret_df[['ticker', 'tradeDate']], on=['ticker', 'tradeDate'])
cyq_week_df.to_pickle('CYQ/cyq_week.pkl')

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


# 若已储存因子，则直接读取
cyq_month_df = pd.read_pickle('CYQ/cyq_month.pkl')
cyq_week_df = pd.read_pickle('CYQ/cyq_week.pkl')

'''

2.2 解析CYQ因子（3分钟）

   调试 运行
文档
 代码  策略  文档
以个股"平安银行"为例,观察CYQ因子、Gain因子、Loss因子的形态。下图展示个股周度因子和股价的走势图。


'''


ticker = '000001'
stock_cyq_df = cyq_week_df[cyq_week_df.ticker == ticker]
stock_cyq_df = stock_cyq_df.merge(f_wret_df, on=['ticker', 'tradeDate'])

fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
for fn in ['Gain', 'Loss']:
    ax1.plot(pd.to_datetime(stock_cyq_df['tradeDate']), stock_cyq_df[fn], label=fn)
ax1.plot(pd.to_datetime(stock_cyq_df['tradeDate']), (stock_cyq_df['curr_ret']+1).cumprod(), label=u'股价', color='r')
ax1.legend(loc=0, prop=font)
ax1.set_title(u'Gain因子和Loss因子与股价的走势图', fontsize=16, fontproperties=font)
for fn in ['Gain', 'Loss', 'CYQ']:
    ax2.plot(pd.to_datetime(stock_cyq_df['tradeDate']), stock_cyq_df[fn], label=fn)
ax2.legend(loc=0, prop=font)
ax2.set_title(u'CYQ因子与Gain因子、Loss因子的走势图', fontsize=16, fontproperties=font)

'''

观察Gain因子、Loss因子和股价走势，符合上述逻辑。当Loss较大时，即亏损幅度过大时，会对股价形成支撑，未来股价大概率反弹，在没有Loss因子的局部低点均得到印证。Gain因子的效果在图中不太明确。
CYQ因子就是对相对资本收益盈利和亏损状态的权衡。从CYQ因子和Gain因子、Loss因子的走势图，可以发现，CYQ要么和Gain因子重合，要么与Loss因子重合。个股有两个状态：1）Gain因子较大，Loss因子趋于0；2）Loss因子较大，Gain因子趋于0。
因此，可以理解为，股价由Gain因子和Loss因子其中的一个主导走势。
   调试 运行
文档
 代码  策略  文档
下面对CYQ因子、Gain因子、Loss因子月度、周度在全A、中证800、中证500成分股内进行分组回测。观察其因子性质。
回测说明：
选股范围：在投资域中，剔除上市不满250个交易日的股票、剔除ST股票、剔除停牌的股票。（为了方便起见，本文在第一部分数据准备中，将用于回测的收益率数据进行了上述个股的剔除。）
回测区间：2007-01-01至2019-03-01
分档方式：按照当期因子值大小，从小到大分为10组。（序号越大，因子值越大）
结果：在各个投资域下，展示超额收益的年化均值收益, 基准为各投资域等权组合。

'''

start_time = time.time()
print ("该部分获取ZZ500、ZZ800的成份股...")

def get_universe(index_name, date_list):
    """
    获取指数的成份股
    params:
        index_name: str, 指数名称，如'ZZ800'
        date_list: list, 日期列表
    return:
        DataFrame，每日的指数成份股
    """
    index_symbol = DataAPI.SecIDGet(cnSpell=index_name)['secID'].values[0]
    universe_df = []
    for date in date_list:
        cont_df = pd.DataFrame(DynamicUniverse(index_symbol).preview(date), columns=['ticker'])
        cont_df['tradeDate'] = date
        universe_df.append(cont_df)
    universe_df = pd.concat(universe_df).reset_index(drop=True)
    universe_df['ticker'] = universe_df['ticker'].apply(lambda x: x[:6])
    return universe_df

week_universe = {}
month_universe = {}
for index_name in ['ZZ800', 'ZZ500']:
    week_universe[index_name] = get_universe(index_name, weekly_dates_list)
    month_universe[index_name] = get_universe(index_name, monthly_dates_list)
with open('CYQ/week_universe.pkl', 'wb') as f:
    pickle.dump(week_universe, f)
with open('CYQ/month_universe.pkl', 'wb') as f:
    pickle.dump(month_universe, f)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


# 若已储存投资域，则直接读取
week_universe = pd.read_pickle('CYQ/week_universe.pkl')
month_universe = pd.read_pickle('CYQ/month_universe.pkl')


def cal_quantile_return(signal_df, return_df, factor_name, return_name, ngrp=10):
    """
    分组回测， 根据因子值将个股等分成给定组数，进行回测
    根据调仓频率，进行交易，返回最后的超额收益率, 基准为投资域等权组合。
    params:
            signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一列为股票当日的因子值
            return_df: DataFrame, columns=['ticker', 'tradeDate', [next_period_return]], 收益率，只含有调仓日，以及下期累计收益率
            factor_name:　str, signal_df中因子值的列名 
            return_name： str, return_df中收益率的列名
            ngrp： int, 分组数
    return:
            DataFrame, columns=['tradeDate', 'group', 'period_ret'], 返回超额收益率
    """
    bt_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'])

    # 分祖
    bt_df.dropna(subset=[factor_name], inplace=True)
    bt_df = qutil.signal_grouping(bt_df, factor_name=factor_name, ngrp=ngrp)

    # 计算权重：每组等权    
    count_df = bt_df.groupby(['tradeDate', 'group']).apply(lambda x: len(x)).reset_index()
    count_df.columns = ['tradeDate', 'group', 'count']
    bt_df = bt_df.merge(count_df, on=['tradeDate', 'group'])
    bt_df['weight'] = 1.0 / bt_df['count']

    # 统计每组的超额收益率
    group_pref = bt_df.groupby(['tradeDate', 'group']).apply(lambda x: np.sum(x[return_name] * x['weight'])).reset_index()
    group_pref.columns = ['tradeDate', 'group', 'period_ret']
    market_pref = bt_df.groupby(['tradeDate']).apply(lambda x: np.sum(x[return_name] * x['weight'])/np.sum(x['weight'])).reset_index()
    market_pref.columns = ['tradeDate', 'market_ret']
    merge_pref = pd.merge(group_pref, market_pref, on='tradeDate')
    merge_pref['period_ret'] = merge_pref['period_ret'] - merge_pref['market_ret']
    merge_pref = merge_pref[['tradeDate', 'group', 'period_ret']]

    # 调整时间
    merge_pref.sort_values(['group', 'tradeDate'], inplace=True)
    merge_pref['period_ret'] = merge_pref.groupby('group')['period_ret'].shift(1)
    merge_pref['period_ret'].fillna(0, inplace=True)

    return merge_pref

def plot_signal_group_backtest(perf, annual_len, ax, fig_title, ngrp=5):
    """
    绘制因子分组回测绝对收益的柱状体
    params:
        perf: DataFrame， 因子分组回测结果。
        ax: 坐标轴
        fig_title：str, 图片标题
        ngrp: int, 分组数
    """
    nav = []
    label_dict = {1: u'第1组(Low)', ngrp: u'第%s组(High)' %ngrp}
    for i in range(2, ngrp):
        label_dict[i] = u'第%s组'%i
    for i in range(ngrp):
        gperf = perf[perf['group'] == i]
        nav.append(gperf['period_ret'].mean() * annual_len)
    ind = np.arange(ngrp)
    ax.bar(ind+0.2, nav, 0.3, color='r')
    ax.set_xlim((0, ind[-1]+1))
    ax.set_xticks(ind+0.35)
    ax.set_xticklabels([label_dict[i+1] for i in ind], fontproperties=font);
    ax.set_title(fig_title, fontproperties=font, fontsize=16)
    
def universe_group_backtest(factor_name, universe='TLQA'):
    """
    不同投资域中的分组回测结果
    params:
        factor_name: 因子名
        universe: str， 投资域名称。取值'TLQA'、'ZZ800'、'ZZ500'，对应投资域。
    """
    week_factor_df = cyq_week_df.copy()
    month_factor_df = cyq_month_df.copy()
    if universe is not 'TLQA':
        week_factor_df = week_factor_df.merge(week_universe[universe], on=['ticker', 'tradeDate'])
        month_factor_df = month_factor_df.merge(month_universe[universe], on=['ticker', 'tradeDate'])
    perf1 = cal_quantile_return(month_factor_df, f_mret_df, factor_name, 'nxt1m_ret', ngrp=10)
    perf2 = cal_quantile_return(week_factor_df, f_wret_df, factor_name, 'nxt1w_ret', ngrp=10)
    
    fig = plt.figure(figsize=(20,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    universe_dict = {'TLQA': u'全市场', 'ZZ800': u'中证800指数', 'ZZ500': u'中证500指数'}
    fig_title1 = u'%s%s因子分档-月度调仓' %(universe_dict[universe], factor_name)
    fig_title2 = u'%s%s因子分档-周度调仓' %(universe_dict[universe], factor_name)
    plot_signal_group_backtest(perf1, 12, ax1, fig_title1, ngrp=10)
    plot_signal_group_backtest(perf2, 52, ax2, fig_title2, ngrp=10)
    return perf1,perf2


start_time = time.time()
print ("该部分获取CYQ因子分组回测结果...")

fn = 'CYQ'
for univ in ['TLQA', 'ZZ800', 'ZZ500']:
    universe_group_backtest(factor_name=fn, universe=univ)
    
end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


'''
CYQ因子是一个负向因子。
历史分组回测，全市场的单调性较好；在指数投资域中，多头部分效果较好，在空头部分的单调性不太稳定。
CYQ因子的多头效应明显。
对比周度回测和月度回测结果，第一组的周度收益大幅超过月度收益。说明因子效果，衰退较快。因子对短期的未来收益预测效果更佳。

'''


start_time = time.time()
print ("该部分获取Gain因子分组回测结果...")

fn = 'Gain'
for univ in ['TLQA', 'ZZ800', 'ZZ500']:
    universe_group_backtest(factor_name=fn, universe=univ)
    
end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

'''
Gain因子是一个负向因子。
历史分组回测，Gain因子的单调性不佳，中间组的收益最大。这是符合实际情况的，当盈利较小时，投资者不会急于卖出清仓；当相对资本盈利较大时，才会对股价产生压力。
Gain因子的空头效应明显。指数投资域内，周度调仓基本上失去超额收益。
对比周度回测和月度回测结果，月度收益优于周度。

'''

start_time = time.time()
print ("该部分获取Loss因子分组回测结果...")

fn = 'Loss'
for univ in ['TLQA', 'ZZ800', 'ZZ500']:
    universe_group_backtest(factor_name=fn, universe=univ)
    
end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

'''
Loss因子是一个负向因子。对比Gain因子，Loss因子明显更优。
Loss因子的结果与CYQ因子基本一致。所以CYQ因子的收益主要来源于Loss部分。
   调试 运行
文档
 代码  策略  文档
因为Gain因子均大于0，Loss因子小于0，进行等全组合后得到CYQ因子，Gain因子和Loss因子效果会相互抵消。尝试将他们分开，当Gain因子为主导因素时，因子值取Gain因子；当Loss因子为主导因素时，因子值取Loss因子，构造CYQ_reform因子。

'''

cyq_week_df['CYQ_reform'] = np.where(cyq_week_df['CYQ'] > 0, cyq_week_df['Gain'], cyq_week_df['Loss'])


'''

第三部分：CYQ因子分析
该部分耗时 2分钟
该部分内容为因子的效果分析，根据第二部分的分析，因子的周度效果明显优于月度，因此下文仅分析周度结果， 具体包括：

3.1 计算CYQ系列因子和常见的风格因子的相关性

3.2 CYQ系列因子的IC分析

3.3 CYQ系列因子的收益分析

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 相关性分析

   调试 运行
文档
 代码  策略  文档
将CYQ系列因子和常见的风格因子（风险模型中的风格因子）进行相关性分析。

'''


start_time = time.time()
print ("该部分进行相关性分析...")

cyq_list = ['CYQ', 'CYQ_reform', 'Gain', 'Loss']
merge_week_df = week_style_factor_df.merge(cyq_week_df, on=['ticker', 'tradeDate'])
date_corr_df = merge_week_df.groupby('tradeDate').corr(method='spearman')[cyq_list]
corr_df = sum([date_corr_df.loc[date] for date in weekly_dates_list]) / len(weekly_dates_list)
corr_df = corr_df.loc[style_factor_list+cyq_list,:]

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corr_df, linewidths=0.05, ax=ax, vmax=1, vmin=-1, cmap='RdYlGn_r', annot=True)
ax.set_title(u'CYQ因子与常见风格因子的相关性',  fontproperties=font, fontsize=16)

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

'''
CYQ系列因子与常见风格因子的相关性均较低。
CYQ因子和CYQ_reform因子本质上一样。
'''


week_ic_df = qutil.calc_ic(merge_week_df, f_wret_df, cyq_list, 'nxt1w_ret')
week_ic_table = ic_describe(week_ic_df.set_index('tradeDate'), 52)
print (week_ic_table.to_html())


'''


从IC看，整体上综合Gain和Loss两方面的CYQ因子由于单方面的因子。
CYQ因子和CYQ_reform因子效果相近，CYQ_reform因子的IC均值和波动率均略好于CYQ因子。
   调试 运行
文档
 代码  策略  文档
3.3 收益分析
   调试 运行
文档
 代码  策略  文档
下面回测CYQ因子和CYQ_reform因子的收益表现。调仓考虑交易费，交易费千分之三，在卖出时收取。

'''

start_time = time.time()
print ("该部分对CYQ因子进行收益分析...")

factor_name = 'CYQ'
perf_lo,_ = qutil.easy_backtest(cyq_week_df, f_wret_df, factor_name, 'nxt1w_ret', method='long_only', direction=-1, ngrp=10, commission=0.003)
perf_ls_cost,_ = qutil.easy_backtest(cyq_week_df, f_wret_df, factor_name, 'nxt1w_ret', method='long_short', direction=-1, ngrp=10, commission=0.003)
perf_ls,_ = qutil.easy_backtest(cyq_week_df, f_wret_df, factor_name, 'nxt1w_ret', method='long_short', direction=-1, ngrp=10, commission=0)
perf_lo = perf_lo.merge(week_idx_ret_df, on=['tradeDate'])
perf_lo['excess_ret'] = (perf_lo['period_ret'] - perf_lo['ZZ800_ret']) / 2
perf_lo.ix[0, 'excess_ret'] = 0
perf_lo['excess_cum_ret'] = (perf_lo['excess_ret']+1).cumprod()

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111)
ax1 = ax.twinx()
ax1.plot(pd.to_datetime(perf_lo['tradeDate']), perf_lo['cum_ret'], label=u'多头净值（扣费）(右轴)', color='y')
ax.plot(pd.to_datetime(perf_lo['tradeDate']), perf_lo['excess_cum_ret'], label=u'多头对冲中证800净值')
ax.plot(pd.to_datetime(perf_ls['tradeDate']), perf_ls['cum_ret'], label=u'多空对冲净值（不扣费）')
ax.plot(pd.to_datetime(perf_ls['tradeDate']), perf_ls_cost['cum_ret'], label=u'多空对冲净值(扣费)')
ax.legend(loc=0, prop=font)
ax1.legend(loc=0, prop=font)
ax.grid(False)
ax.set_title(u'CYQ因子周度回测净值走势',  fontproperties=font, fontsize=16)

perf_all = pd.concat([perf_lo.set_index('tradeDate')[['period_ret', 'excess_ret']], perf_ls.set_index('tradeDate')[['period_ret']], perf_ls_cost.set_index('tradeDate')[['period_ret']]], axis=1)
perf_all.columns=['多头（扣费）', '多头(扣费)对冲中证800', '多空对冲（不扣费）',  '多空对冲（扣费）']
perf_table = perf_describe(perf_all, annual_len=52)
print ("--%s因子周度回测表现--" % factor_name)
print (perf_table.to_html())

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

'''
CYQ因子多头收益明显。年化收益达到35.64%。
扣费后，多空年化收益达到11.55%，夏普比率为1.03。多头对冲中证800的年化收益达到13.27%，夏普比率为1.22。
'''

perf_ls = perf_all[['多头(扣费)对冲中证800', '多空对冲（不扣费）',  '多空对冲（扣费）']]
perf_describe_by_year(perf_ls, annual_len=52)


start_time = time.time()
print ("该部分对CYQ_reform因子进行收益分析...")

factor_name = 'CYQ_reform'
perf_lo,_ = qutil.easy_backtest(cyq_week_df, f_wret_df, factor_name, 'nxt1w_ret', method='long_only', direction=-1, ngrp=10, commission=0.003)
perf_ls_cost,_ = qutil.easy_backtest(cyq_week_df, f_wret_df, factor_name, 'nxt1w_ret', method='long_short', direction=-1, ngrp=10, commission=0.003)
perf_ls,_ = qutil.easy_backtest(cyq_week_df, f_wret_df, factor_name, 'nxt1w_ret', method='long_short', direction=-1, ngrp=10, commission=0)
perf_lo = perf_lo.merge(week_idx_ret_df, on=['tradeDate'])
perf_lo['excess_ret'] = (perf_lo['period_ret'] - perf_lo['ZZ800_ret']) / 2
perf_lo.ix[0, 'excess_ret'] = 0
perf_lo['excess_cum_ret'] = (perf_lo['excess_ret']+1).cumprod()

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(111)
ax1 = ax.twinx()
ax1.plot(pd.to_datetime(perf_lo['tradeDate']), perf_lo['cum_ret'], label=u'多头净值（扣费）(右轴)', color='y')
ax.plot(pd.to_datetime(perf_lo['tradeDate']), perf_lo['excess_cum_ret'], label=u'多头对冲中证800净值')
ax.plot(pd.to_datetime(perf_ls['tradeDate']), perf_ls['cum_ret'], label=u'多空对冲净值（不扣费）')
ax.plot(pd.to_datetime(perf_ls['tradeDate']), perf_ls_cost['cum_ret'], label=u'多空对冲净值(扣费)')
ax.legend(loc=0, prop=font)
ax1.legend(loc=0, prop=font)
ax.grid(False)
ax.set_title(u'CYQ因子周度回测净值走势',  fontproperties=font, fontsize=16)

perf_all = pd.concat([perf_lo.set_index('tradeDate')[['period_ret', 'excess_ret']], perf_ls.set_index('tradeDate')[['period_ret']], perf_ls_cost.set_index('tradeDate')[['period_ret']]], axis=1)
perf_all.columns=['多头（扣费）', '多头(扣费)对冲中证800', '多空对冲（不扣费）',  '多空对冲（扣费）']
perf_table = perf_describe(perf_all, annual_len=52)
print ("--%s因子周度回测表现--" % factor_name)
print (perf_table.to_html())

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


perf_ls = perf_all[['多头(扣费)对冲中证800', '多空对冲（不扣费）',  '多空对冲（扣费）']]
perf_describe_by_year(perf_ls, annual_len=52)

'''
整体上，CYQ_reform因子的历史回测结果，均优于CYQ因子，但是提升效果并不明显。
   调试 运行
文档
 代码  策略  文档
3.4 构造Smart Beta指数
在第二部分，对CYQ因子在ZZ800,ZZ500投资域中的分组较好，且多头效应明显。因此，利用CYQ_reform因子构造ZZ800,ZZ500的Smart Beta指数。
构造细节：
回测时间：2008-01-01至2019-03-20；
将指数成分股从高到低分成10组，选出因子值最小的一组；
组合方式：等权；
调仓：周度调仓，考虑手续费，买入千分之三，卖出千分之一。

'''

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


cyq_week_df['secID'] = cyq_week_df['ticker'].apply(ticker2secID)

start = '2008-01-01'                       # 回测起始时间
end = '2019-03-20'                         # 回测结束时间
universe = DynamicUniverse('ZZ800')        # 证券池，支持股票、基金、期货、指数四种资产
benchmark = '000906.ZICN'                  # 策略参考标准
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟
  
# 配置账户信息，支持多资产多账户
accounts = {
    'account': AccountConfig(account_type='security', capital_base=10000000, commission=Commission(0.0003, 0.0001)),
}
  
def initialize(context):
    pass
  
def handle_data(context): 
    account = context.get_account('account')
    current_universe = context.get_universe('stock', exclude_halt=True)
    pre_date = context.previous_date.strftime("%Y%m%d")
    if pre_date not in cyq_week_df['tradeDate'].values:            
        return

    # 拿取调仓日前一个交易日的因子，并按照相应分位选择股票
    q = cyq_week_df[cyq_week_df['tradeDate'] == pre_date]
    q = q.set_index('secID', drop=True)
    q = q.ix[current_universe]
    q = q.dropna()
    q_min = q['CYQ_reform'].quantile(0.1)
    q = q[q['CYQ_reform']<=q_min]
    my_univ = q.index.values
    
    
    # 计算调仓权重
    q['weight'] = 1.0 / len(my_univ)

   # 交易部分
    positions = account.get_positions()
    sell_list = [stk for stk in positions if stk not in my_univ]
    for stk in sell_list:
        account.order_to(stk,0)
    for stk in my_univ:
        account.order_pct_to(stk, q.ix[stk, 'weight'])
        
        
start = '2008-01-01'                       # 回测起始时间
end = '2019-03-20'                         # 回测结束时间
universe = DynamicUniverse('ZZ500')        # 证券池，支持股票、基金、期货、指数四种资产
benchmark = 'ZZ500'                  # 策略参考标准
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟
  
# 配置账户信息，支持多资产多账户
accounts = {
    'account': AccountConfig(account_type='security', capital_base=10000000, commission=Commission(0.0003, 0.0001)),
}
  
def initialize(context):
    pass
  
def handle_data(context): 
    account = context.get_account('account')
    current_universe = context.get_universe('stock', exclude_halt=True)
    pre_date = context.previous_date.strftime("%Y%m%d")
    if pre_date not in cyq_week_df['tradeDate'].values:            
        return

    # 拿取调仓日前一个交易日的因子，并按照相应分位选择股票
    q = cyq_week_df[cyq_week_df['tradeDate'] == pre_date]
    q = q.set_index('secID', drop=True)
    q = q.ix[current_universe]
    q = q.dropna()
    q_min = q['CYQ_reform'].quantile(0.1)
    q = q[q['CYQ_reform']<=q_min]
    my_univ = q.index.values
    
    
    # 计算调仓权重
    q['weight'] = 1.0 / len(my_univ)

   # 交易部分
    positions = account.get_positions()
    sell_list = [stk for stk in positions if stk not in my_univ]
    for stk in sell_list:
        account.order_to(stk,0)
    for stk in my_univ:
        account.order_pct_to(stk, q.ix[stk, 'weight'])
        
        
