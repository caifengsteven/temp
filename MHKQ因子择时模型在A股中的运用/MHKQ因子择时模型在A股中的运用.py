# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:16:56 2020

@author: Asus
"""
'''
导读
A.研究目的：量化研究中因子的加权方式对策略的表现起着至关重要的作用，在2017年的价值投资行情中，之前一直长期有效的市值、反转及特质波动率等因子纷纷失效，而传统的因子加权方式不能及时适应市场短期的快速变化，从而导致很多量化多因子组合出现了较大的回撤，而好的因子择时模型能及时调整因子权重，扭转组合颓势，参考天风证券:《MHKQ因子择时模型在A股中的运用》，书籍：《量化投资策略——如何实现超额收益Alpha》，对研报中的基于条件期望的因子择时模型进行了实证及分析，利用因子择时构造能够及时适应市场风格变化的稳健模型，改善多因子模型的表现。

B.研究结论：实证表明，通过优矿的因子库及宏观数据进行实证，我们的因子择时模型能显著改善市场风格切换带来的回撤，通过择时模型对量化因子进行加权复合构建全市场多因子组合，从2011年初回测至2018年5月底，月度调仓，多头部分年化Alpha24.0%，信息比率2.66，静态组合在17年的绝对收益为-21.58%，而因子择时模型在17年能获得（10.06%）的正收益，能很好地适应市场风格切换，根据因子构建的500成分内增强组合在最大回撤不超过5%的情况下能获得10.8%的稳健年化超额收益。

C.文章结构：本文共分为3个部分，具体如下

一、数据准备及处理：包括股票池的界定，所用选股因子的构造、预处理，外生变量的获取等

二、因子择时模型构造：该部分利用外生变量构建择时模型，用IC加权法与构造好的模型进行因子合成，分析不同风格下两者因子权重比较

三、策略回测：利用优矿平台，对第二部分的复合因子，回测择时模型是否确实有效，并基于模型结果构造指数增强组合

D.其它说明

一、本文测试的股票池剔除了上市不足60天，ST以及*ST的股票

二、外生变量中的宏观因子均来源于优矿宏观行业特色数据，调用函数文中已注明

三、由于对外生变量的处理及因子加权观察期的需要，组合回测的时间从2011年初至2018年5月底

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

第一部分：数据准备及处理
该部分耗时 60分钟(主要是因子的构造环节需要45分钟，其他加起来15分钟)
该部分内容为：

每期股票池的选取（剔除ST及上市不满半年的新股）

选股因子的构造，对因子的预处理（去极值、中性化、标准化）

从uqer的DataAPI中获取本文所需的外生变量数据

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

1.1 每期股票池的选取

生成的股票池文件存储在MHKQ_data/stock_pool.csv

股票池属性为date，code，区间为日期为（20060428 - 20180531）之间的每月月末

'''

# coding: utf-8

import numpy as np
import pandas as pd
import datetime
import time
import os
import copy
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
#from CAL.PyCAL import *    # CAL.PyCAL中包含font
#universe = set_universe('A')
#cal = Calendar('China.SSE')

from sqlalchemy import create_engine
import json

import warnings
warnings.filterwarnings('ignore')

#must be set before using
with open('para.json','r',encoding='utf-8') as f:
    para = json.load(f)
    
pn = para['yuqerdata_dir']

user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name1 = 'yuqerdata'
#eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name1)
engine = create_engine(eng_str)
sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate''' 
## 数据的起始与终止时间
start_date = '20070101'                       # 回测起始时间
end_date = '20190924'
def get_IdxCons(intoDate,ticker='000300'):
    #nearst 时间
    sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
            and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
        ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' %(ticker,
        ticker,intoDate)
    x = pd.read_sql(sql_str1,engine)
    x = x['symbol'].values   
    return x

universe = get_IdxCons(intoDate='20200101',ticker='000001')


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
def get_calender():
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" order by tradeDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x
cal = pd.DataFrame(get_calender())
def get_month_calender():
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" and endDate>="%s" order by endDate'''%(start_date)
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

    
# 股票池文件存放目录，如果目录不存在，程序自动新建一个
raw_data_dir = "./MHKQ_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

def time_change(x):
    y = datetime.datetime.strptime(x, '%Y-%m-%d')
    y = y.strftime('%Y%m%d')
    return y

# 获取回测区间的交易日、月末以及月初时间
def get_trade_list(start_date, end_date):
    """
    Args:
        start_date: 时间区间起点
        end_date: 时间区间终点
    Returns: 
        trade_list: 时间区间内的交易日
        month_end: 月末时间
        month_start: 月初时间
    """
    #cal_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date).sort('calendarDate')
    #cal_dates = cal_dates[cal_dates['isOpen'] == 1]
    cal_dates = get_calender()
    trade_list = cal_dates.tolist()
    #trade_list = [time_change(x) for x in trade_list]
    month_end = get_month_calender()
    month_end = month_end.tolist()
    #month_end = [time_change(x) for x in month_end]
    cal = get_calender()
    month_start = [cal.advanceDate(x, '1B').strftime('%Y%m%d') for x in month_end]
    return trade_list, month_end, month_start

# 剔除ST股票
def st_remove(source_universe, st_date=None):
    """
    Args:
        source_universe (list of str): 需要进行筛选的股票列表
        st_date (datetime): 进行筛选的日期,默认为调用当天
    Returns:
        list: 去掉ST股票之后的股票列表
    """
    st_date = st_date if st_date is not None else datetime.datetime.now().strftime('%Y%m%d')
    df_ST = DataAPI.SecSTGet(secID=source_universe, beginDate=st_date, endDate=st_date, field=['secID'])
    return [s for s in source_universe if s not in list(df_ST['secID'])]

# 剔除某个日期前多少个交易日,之后上市的新股
def new_remove(ticker,tradeDate= None,day = 1):
    """
    Args:
        ticker (list of str): 需要进行筛选的股票列表（无后缀）
        tradeDate (datetime): 进行筛选的日期,默认为调用当天
        day (int): 向前漂移的交易日的个数
    Returns:
        list: 去掉新股股票之后的股票列表（无后缀）
    """
    tradeDate = tradeDate if tradeDate is not None else datetime.datetime.now()
    period = '-' + str(day) + 'B'
    pastDate = cal.advanceDate(tradeDate,period)
    pastDate = pastDate.strftime("%Y-%m-%d")
    ipo_date = DataAPI.SecIDGet(partyID=u"",assetClass=u"e",ticker=ticker,cnSpell=u"",field=u"ticker,listDate",pandas="1")
    remove_list = ipo_date[ipo_date['listDate'] > pastDate]['ticker'].tolist()
    return [stk for stk in ticker if stk not in remove_list]

# 将股票代码转化为股票内部编码
def ticker2secID(ticker):
    """
    Args:
        tickers (list): 需要转化的股票代码列表
    Returns:
        list: 转化为内部编码的股票编码列表
    """
    universe = DataAPI.EquGet(equTypeCD=u"A",listStatusCD="L,S,DE,UN",field=u"ticker,secID",pandas="1") # 获取所有的A股（包括已退市）
    universe = dict(universe.set_index('ticker')['secID'])
    if isinstance(ticker, list):
        res = []
        for i in ticker:
            if i in universe:
                res.append(universe[i])
            else:
                print (i, ' 在universe中不存在，没有找到对应的secID！')
        return res
    else:
        raise ValueError('ticker should be list！')

# 获取股票池函数
def get_stock_pool(date, N):
    """
    Args:
        date: 月初时间
        N: 新股的定义时间
    Returns:
        stock_pool: 此月初的股票池
    """
    univ=get_IdxCons(intoDate='20200101',ticker='000001')
    all_code = univ.preview(date,skip_halted=False)
    all_code_not_ST = st_remove(all_code, st_date=date)
    ticker = [x.split('.')[0] for x in all_code_not_ST]
    all_code_need = new_remove(ticker, tradeDate=date, day=N)
    code = ticker2secID(all_code_need)
    df = pd.DataFrame({'code': code})
    df['date'] = date
    df = df[['date', 'code']]
    return df

# 获取每月初的股票池(20060428 - 20180531)
tic = time.time()
#start_date = '20060428'
#end_date = '20180531'

start_date = '20100201'                       # 回测起始时间
end_date = '20190924'
trade_list, month_end, month_start = get_trade_list(start_date, end_date)
N = 180
all_stock = pd.DataFrame()
for date in month_start:
    stock = get_stock_pool(date, N)
    all_stock = pd.concat([all_stock, stock],axis=0)

all_stock.to_csv('MHKQ_data/stock_pool.csv', index=False)
toc = time.time()
print('***********股票池示例************')
print(all_stock.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''

1.2 因子构造及预处理

因子构造

考虑到因子的全面性和代表性，我们在规模、技术反转、流动性、波动性、估值、成长、质量这七个维度选取了7个典型因子来构造多因子组合，如下： 图片注释
这里采用市值当做因子之一的是考虑到，市值作为市场风格切换的最显著特征，通过对市值因子权重的跟踪可以彰显模型择时的及时性以及有效性。
这里我们单独计算单季度净利润增速，财务数据来源于合并利润表，它是根据所有会计期末最新披露的数据计算的，可能部分股票会有未来数据，但考虑到这部分股票数量其实很少，而且对所有股票统一进行处理并不影响因子的质量，因此我们仍然采用这种构造方法，仅在此处说明。
我们对每个因子进行如下处理：

Step1: 采用MAD（Median Absolute Deviation 绝对中位数法）进行边界压缩处理，剔除异常值；
Step2: 对除lnmkt以外的其他因子进行市值 + 行业的中性化；
Step3: 对第二步的残差项做z-score标准化处理。 factor_stand就是做完预处理后的标准化因子数据，格式如下： 图片注释

'''

# 因子计算与整合

# 计算对数市值因子
def cal_lnmkt(date, code):
    """
    Args:
        date: 月初时间
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        mkt: 对数市值因子数据,dataframe格式，列名为日期，股票代码，因子值
    """
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')   
    mkt = DataAPI.MktEqudGet(tradeDate=end,secID=code,field=u"tradeDate,secID,marketValue",pandas="1")
    mkt.columns = ['date', 'code', 'mkt']
    mkt['mkt'] = np.log(mkt['mkt'])
    mkt['date'] = map(time_change, mkt['date'])
    return mkt

# 计算反转因子：过去20个交易日的涨跌幅
def cal_ret_20(date, code):
    """
    Args:
        date: 月初时间
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        ret: 反转因子数据,dataframe格式，列名为日期，股票代码，因子值
    """
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')
    pre_20 = cal.advanceDate(end, '-20B').strftime('%Y%m%d')
    close_before = DataAPI.MktEqudAdjAfGet(secID=code,beginDate=pre_20,endDate=pre_20,field=u"secID,closePrice",pandas="1")
    close_before.columns = ['code', 'close_pre_20']
    close = DataAPI.MktEqudAdjAfGet(secID=code,beginDate=end,endDate=end,field=u"secID,closePrice",pandas="1")
    close.columns = ['code', 'close_end']
    ret = pd.merge(close_before, close, on='code')
    ret['ret_20'] = ret['close_end'] / ret['close_pre_20'] - 1   
    ret['date'] = end
    ret = ret[['date', 'code', 'ret_20']]
    return ret

# 计算换手率因子
def cal_mean_to(date, code):
    """
    Args:
        date: 月初时间
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        mean_to: 换手率因子数据,dataframe格式，列名为日期，股票代码，因子值
    """
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')   
    pre_19 = cal.advanceDate(end, '-19B').strftime('%Y%m%d')
    temp = DataAPI.MktEqudGet(secID=code,beginDate=pre_19,endDate=end,field=u"tradeDate,secID,turnoverRate",pandas="1")
    grouped = temp.groupby(['secID'])['turnoverRate'].mean()
    mean_to = grouped.reset_index()
    mean_to.columns = ['code', 'mean_to']
    mean_to['date'] = end
    mean_to = mean_to[['date', 'code', 'mean_to']]
    return mean_to

# 计算特质波动率、特异度因子
def cal_specificity(date, code):
    """
    Args:
        date: 月初时间
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        spec: 特质波动率及特异度因子数据,dataframe格式，列名为日期，股票代码，因子值
    """
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')
    pre_19 = cal.advanceDate(date, '-19B').strftime('%Y%m%d')
    all_data = DataAPI.MktEqudGet(secID=code,beginDate=pre_19,endDate=end,field=u"tradeDate,secID,PB,negMarketValue,chgPct",pandas="1")
    all_data['tradeDate'] = map(time_change, all_data['tradeDate'])
    tdata = all_data[all_data['tradeDate'] == end]
    # 市场组合
    tdata['weight'] = tdata['negMarketValue'] / tdata['negMarketValue'].sum()
    port_all = tdata[['secID', 'weight']]
    port_all.columns = ['secID', 'weight_market']    
    # 大小市值组合
    temp = tdata.copy()
    temp.sort_values(by='negMarketValue', inplace=True)
    temp.reset_index(drop=True, inplace=True)
    num = len(temp) / 3
    port_small_mkv = temp[0: num]
    port_small_mkv['weight_sm'] = port_small_mkv['negMarketValue'] / port_small_mkv['negMarketValue'].sum()
    port_small_mkv = port_small_mkv[['secID', 'weight_sm']]
    port_large_mkv = temp[-num:]
    port_large_mkv['weight_lm'] = port_large_mkv['negMarketValue'] / port_large_mkv['negMarketValue'].sum()    
    port_large_mkv = port_large_mkv[['secID', 'weight_lm']]
    # 高低PB组合
    temp = tdata.copy()
    temp.sort_values(by='PB', inplace=True)
    temp.reset_index(drop=True, inplace=True)
    port_low_pb = temp[0: num]
    port_low_pb['weight_lp'] = port_low_pb['negMarketValue'] / port_low_pb['negMarketValue'].sum()
    port_low_pb = port_low_pb[['secID', 'weight_lp']]
    port_high_pb = temp[-num:]
    port_high_pb['weight_hp'] = port_high_pb['negMarketValue'] / port_high_pb['negMarketValue'].sum()    
    port_high_pb = port_high_pb[['secID', 'weight_hp']]
    # 整合
    weight = pd.merge(port_all, port_small_mkv, on='secID', how='left')
    weight = pd.merge(weight, port_large_mkv, on='secID', how='left')
    weight = pd.merge(weight, port_low_pb, on='secID', how='left')
    weight = pd.merge(weight, port_high_pb, on='secID', how='left')
    weight.fillna(0, inplace=True)
    # 收益矩阵
    cal_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=pre_19, endDate=end).sort('calendarDate')
    cal_dates = cal_dates[cal_dates['isOpen']==1]
    date_list = cal_dates['calendarDate'].values.tolist()
    date_list = [time_change(x) for x in date_list]
    for day in date_list:
        pct_chg = all_data[all_data['tradeDate'] == day]
        pct_chg = pct_chg[['secID', 'chgPct']]
        pct_chg.columns = ['secID', 'chgPct__' + str(day)]
        weight = pd.merge(weight, pct_chg, on='secID')
    # 日收益序列
    w_mat = np.matrix(weight.iloc[:, 1: 6]).T
    change_mat = np.matrix(weight.iloc[:, 6: ])                 
    ret = w_mat * change_mat
    cols = ['ret_all', 'ret_small_mkv', 'ret_large_mkv', 'ret_low_pb', 'ret_high_pb']
    ret = pd.DataFrame(ret.T,columns=cols)
    ret['date'] = date_list
    ret['ret_mkv'] = ret['ret_small_mkv'] - ret['ret_large_mkv']
    ret['ret_pb'] = ret['ret_low_pb'] - ret['ret_high_pb']
    ret['constant'] = 1
    ret = ret[['date', 'constant', 'ret_all', 'ret_mkv', 'ret_pb']]
    # 回归
    pct_table = weight.iloc[:, 6: ]
    pct_table.columns = [str(x.split('__')[1]) for x in list(pct_table.columns)]
    pct_table = pct_table.T
    pct_table.columns = list(weight['secID'])
    reg_data = pd.merge(ret, pct_table, left_on='date', right_index=True)
    x = reg_data.iloc[:, 1:5]  # 这里加了常数项constant
    col = reg_data.columns[5:]
    IV_col = []
    IVR = []
    all_code = []
    for name in col:
        y = reg_data[name]
        y = y.replace(0, np.nan)
        # 做个判定，回归天数太少的要剔除
        if len(y[y.isnull()]) < 10:
            model = sm.OLS(y, x, missing='drop')
            results = model.fit()
            IV = np.std(results.resid) * np.sqrt(252)
            R2_single = results.rsquared        
            all_code.append(name)            
            IV_col.append(IV)
            IVR.append(1 - R2_single)
        else:
            continue
    spec = pd.DataFrame({'secID': all_code, 'IVFF': IV_col, 'IVR': IVR})
    spec = pd.merge(port_all, spec, on='secID', how='left') # 没有的记为nan
    spec['date'] = end
    spec = spec[['date', 'secID', 'IVFF', 'IVR']]
    spec.columns = ['date', 'code', 'IVFF', 'IVR']
    return spec

# 计算BP因子
def cal_bp(date, code):
    """
    Args:
        date: 月初时间
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        bp: BP因子数据,dataframe格式，列名为日期，股票代码，因子值
    """
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')   
    bp = DataAPI.MktStockFactorsOneDayGet(tradeDate=end,secID=code,field=u"tradeDate,secID,PB",pandas="1")
    bp['BP'] = 1.0 / bp['PB']
    bp = bp[['secID', 'BP']]
    bp['date'] = end
    bp = bp[['date', 'secID', 'BP']]
    bp.columns = ['date', 'code', 'BP']
    return bp

# 计算roe_ttm因子
def cal_roe_ttm(date, code):
    """
    Args:
        date: 月初时间
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        roe_ttm: roe_ttm因子数据,dataframe格式，列名为日期，股票代码，因子值
    """    
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')    
    roe_ttm = DataAPI.MktStockFactorsOneDayProGet(tradeDate=end,secID=code,field=u"secID,ROE",pandas="1")
    roe_ttm['date'] = end
    roe_ttm = roe_ttm[['date', 'secID', 'ROE']]
    roe_ttm.columns = ['date', 'code', 'roe_ttm']
    return roe_ttm

# 获取某个时点股票所属行业
def get_industry(date, code):
    """
    Args:
        date: 月初时间
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        indu: 申万一级行业因子数据,dataframe格式，列名为日期，股票代码，行业名
    """ 
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d') 
    indu = DataAPI.MdSwBackGet(secID=code,field=u"secID,isNew,industryName1",pandas="1")
    indu = indu[indu['isNew'] == 1]
    indu.drop_duplicates(subset=['secID'], inplace=True)
    indu['date'] = end
    indu = indu[['date', 'secID', 'industryName1']]
    indu.columns = ['date', 'code', 'industry']
    return indu

# 单季度净利润增速计算
tic = time.time()
all_code = sorted(list(set(all_stock['code'])))
temp = DataAPI.FdmtISQGet(secID=all_code,beginYear=u"2007",field=u"secID,endDate,NIncome",pandas="1") # 财务数据选取
profit = pd.DataFrame()
for code in all_code:
    cdata = temp[temp['secID'] == code]
    cdata.sort_values(by='endDate', inplace=True)
    cdata.reset_index(drop=True, inplace=True)
    cdata['qfa_yoyprofit'] = (cdata['NIncome'] - cdata['NIncome'].shift(4)) / np.abs(cdata['NIncome'].shift(4))
    cdata = cdata[cdata['endDate'] >= '2007-09-30']
    profit = pd.concat([profit, cdata], axis=0)
    
profit = profit[['secID', 'endDate', 'qfa_yoyprofit']]
profit.columns = ['code', 'rptdate', 'qfa_yoyprofit']
profit['rptdate'] = map(time_change, profit['rptdate'])
profit.dropna(inplace=True)
# 将计算好的单季度净利润增速根据财报披露日期对应到每个月末
qfa_pro = pd.DataFrame()
for date in month_start:
    stock = all_stock[all_stock['date'] == date]
    code = list(stock['code'])
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')    
    date1 = datetime.datetime.strptime(end, '%Y%m%d')
    if date1.month in [1, 2, 3]: # 去年3季报
        x = datetime.datetime(date1.year - 1, 9, 30).strftime('%Y%m%d')
        tdata = profit[profit['rptdate'] == x]
        stock = pd.merge(stock, tdata, on='code', how='left')
        stock['date'] = end  
    elif date1.month in [4, 5, 6, 7]: # 今年一季报 
        x = datetime.datetime(date1.year, 3, 31).strftime('%Y%m%d')
        tdata = profit[profit['rptdate'] == x]
        stock = pd.merge(stock, tdata, on='code', how='left')
        stock['date'] = end
    elif date1.month in [8, 9]: # 今年二季报 
        x = datetime.datetime(date1.year, 6, 30).strftime('%Y%m%d')
        tdata = profit[profit['rptdate'] == x]
        stock = pd.merge(stock, tdata, on='code', how='left')
        stock['date'] = end
    elif date1.month in [10, 11, 12]: # 今年三季报 
        x = datetime.datetime(date1.year, 9, 30).strftime('%Y%m%d')
        tdata = profit[profit['rptdate'] == x]
        stock = pd.merge(stock, tdata, on='code', how='left')
        stock['date'] = end
    qfa_pro = pd.concat([qfa_pro, stock], axis=0)
    
# 按月完成因子计算
all_factor = pd.DataFrame()
for date in month_start[24: ]: # 所需因子从08年开始即可，选择06年开始是出于后面外生变量的区间考虑
    stock = all_stock[all_stock['date'] == date]
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')
    stock['date'] = end
    code = list(stock['code'])
    # 对数市值
    temp = cal_lnmkt(date, code)
    stock = pd.merge(stock, temp, on=['date', 'code'], how='left')
    # 反转
    temp = cal_ret_20(date, code)
    stock = pd.merge(stock, temp, on=['date', 'code'], how='left')
    # 换手
    temp = cal_mean_to(date, code)
    stock = pd.merge(stock, temp, on=['date', 'code'], how='left')
    # 特异度
    temp = cal_specificity(date, code)
    stock = pd.merge(stock, temp, on=['date', 'code'], how='left')
    # BP
    temp = cal_bp(date, code)
    stock = pd.merge(stock, temp, on=['date', 'code'], how='left')
    # roe_ttm
    temp = cal_roe_ttm(date, code)
    stock = pd.merge(stock, temp, on=['date', 'code'], how='left')
    # 行业虚拟变量
    temp = get_industry(date, code)
    stock = pd.merge(stock, temp, on=['date', 'code'], how='left')    
    all_factor = pd.concat([all_factor, stock], axis=0)

all_factor = pd.merge(all_factor, qfa_pro, on=['date', 'code'])
del all_factor['IVFF']
del all_factor['rptdate']
all_factor = all_factor[['date', 'code', 'ret_20', 'mean_to', 'IVR', 'BP', 'roe_ttm', 'qfa_yoyprofit', 'mkt', 'industry']]
all_factor.dropna(inplace=True)
all_factor.sort_values(by=['date', 'code'])
all_factor.reset_index(drop=True, inplace=True)
all_factor.to_csv('MHKQ_data/raw_factor.csv', index=False, encoding='gbk')
toc = time.time()
print('***************原始因子示例**************')
print(all_factor.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


# 因子预处理

def time_change(x):
    y = datetime.datetime.strptime(x, '%Y-%m-%d')
    y = y.strftime('%Y%m%d')
    return y

# 因子预处理函数，中位数去极值-->对市值及行业中性化-->标准化，得到处理好的因子数据
def factor_process(factor_name, data, mode):
    """
    Args:
        factor_name: 需要进行预处理的因子名
        data: 某日的原始因子数据
        mode: 对市值因子不需要执行中性化过程，需要作区分，'yes'代表中性化，'no'代表不做中性化
    Returns:
        data: 对指定factor_name做完处理的因子数据
    """
    # 中位数去极值
    D_mad = abs(data[factor_name] - data[factor_name].median()).median()
    D_m = data[factor_name].median()
    upper = D_m + 5 * D_mad
    lower = D_m - 5 * D_mad
    temp = [max(lower, min(x, upper)) for x in list(data[factor_name])] # 边界压缩 
    data[factor_name] = temp
    # 中性化
    if mode == 'yes':
        y = np.array(data[factor_name])
        x = np.array(data[data.columns[8: ]])
        x = sm.add_constant(x, has_constant='add')
        model = sm.OLS(y, x, missing='drop')
        results = model.fit()
        data[factor_name] = results.resid
    # 标准化
    data[factor_name] = (data[factor_name] - data[factor_name].mean()) / (data[factor_name].std())
    return data

tic = time.time()
factor_stand = pd.DataFrame()
date_list = sorted(list(set(all_factor['date'])))
for date in date_list:
    tdata = all_factor[all_factor['date'] == date]
    tdata.reset_index(drop=True ,inplace=True)
    # 将行业转换成虚拟变量
    indu_dummies = pd.get_dummies(tdata['industry'])
    del tdata['industry']
    tdata = pd.concat([tdata, indu_dummies], axis=1)
    # 先对市值标准化，方便后续其他因子的中性化
    tdata = factor_process('mkt', tdata, 'no')
    # 其他因子
    for factor_name in tdata.columns[2: 8]:
        tdata = factor_process(factor_name, tdata, 'yes')
    tdata = tdata[['date', 'code', 'mkt', 'ret_20', 'mean_to', 'IVR', 'BP', 'roe_ttm', 'qfa_yoyprofit']]
    factor_stand = factor_stand.append(tdata)
    
# 后面会用到因子IC数据，所以需要次月收益计算因子IC，这里计算次月收益
def cal_month_ret(code, this_month, next_month):
    """
    Args:
        code: 当月末的股票列表
        this_month: 当月月末时间
        next_month: 次月月末时间
    Returns:
        m_ret: 月末股票的次月收益数据,列名为日期、股票代码、次月收益
    """
    close_tm = DataAPI.MktEqudAdjAfGet(secID=code,beginDate=this_month,endDate=this_month,field=u"secID,closePrice",pandas="1")
    close_tm.columns = ['code', 'close_tm']
    close_nm = DataAPI.MktEqudAdjAfGet(secID=code,beginDate=next_month,endDate=next_month,field=u"secID,closePrice",pandas="1")
    close_nm.columns = ['code', 'close_nm']    
    close = pd.merge(close_tm, close_nm, on='code')
    close['Month_ret'] = close['close_nm'] / close['close_tm'] - 1
    close['date'] = this_month
    m_ret = close[['date', 'code', 'Month_ret']]    
    return m_ret

month_ret = pd.DataFrame()
date_list = sorted(list(set(factor_stand['date'])))
for i in range(len(date_list) - 1):
    this_month = date_list[i]
    next_month = date_list[i + 1]
    code = list(factor_stand[factor_stand['date'] == this_month]['code'])
    ret = cal_month_ret(code, this_month, next_month)
    month_ret = pd.concat([month_ret, ret], axis=0)
    
factor_stand = pd.merge(factor_stand, month_ret, on=['date', 'code'], how='left')
factor_stand.fillna(0, inplace=True)
# 标准化因子存储
factor_stand.sort_values(by=['date', 'code'])
factor_stand.reset_index(drop=True, inplace=True)
factor_stand.to_csv('MHKQ_data/factor_stand.csv', index=False)
toc = time.time()
print('***************标准化因子示例***************')
print(factor_stand.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


'''
1.3 外生变量

外生变量的选择
本文在Ronald Hua,Dmitri Kantsyrev, Edward Qian设计的基于条件期望的因子择时模型（简称HKQ模型）的外生变量的基础上进行扩展，共选择了如下几类外生变量：

涨跌幅类：上证50、沪深300、中证500过去20个交易日涨跌幅；
时间序列波动率类：上证50、中证1000过去60日收益率标准差；
截面波动率类：全市场股票单日收益率标准差，全市场股票过去20个交易日收益率标准差；
利率类：SHIBOR1W、SHIBOR2W、SHIBOR1M、信用利差（1年AAA中短期票据收益率 - 1年期国债收益率）；
货币类：M1同比 - M2同比；
通货膨胀类：CPI同比 - PPI同比；
工业产业类：采购经理指数PMI、工业增加值当月同比IAV；
日历类：月份；
外生变量的处理

由于GDP数据季度更新，因此我们选择了对GDP数据有较高预测能力且按月更新的PMI和IAV作为替代；
货币类、通货膨胀类、工业产业类月底的数据公布时间是在下一月，因此我们取上月的数据；
月份的取值为下月的月份；
对除月份外的变量，其他的外生变量的取值都取当期值在过去24个月末的取值的相对位置并进行min-max标准化到0~100之间，即 图片注释

'''

# 外生变量获取

# 1.涨跌幅类
def get_index_pct(date_list, ticker_list, N):
    """
    Args:
        date_list：月末序列
        ticker_list: 指数代码列表，如['000016', '000905']
        N: 时间窗口
    Returns:
        ret: 给定日期给定指数的前N日涨跌幅数据
    """    
    cal = Calendar('China.SSE')
    date_list = sorted(date_list)
    ret = pd.DataFrame()
    for date in date_list:
        start = cal.advanceDate(date, '-%sB'%N).strftime('%Y%m%d')
        temp1 = DataAPI.MktIdxdGet(ticker=ticker_list,tradeDate=start,exchangeCD=u"XSHE,XSHG",field=u"tradeDate,indexID,closeIndex",pandas="1")
        temp2 = DataAPI.MktIdxdGet(ticker=ticker_list,tradeDate=date,exchangeCD=u"XSHE,XSHG",field=u"tradeDate,indexID,closeIndex",pandas="1")
        temp2['ret_' + str(N)] = temp2['closeIndex'] / temp1['closeIndex'] - 1
        ret = pd.concat([ret, temp2],axis=0)    
    ret = ret[['tradeDate', 'indexID', 'ret_' + str(N)]]
    ret = ret.pivot_table(index='tradeDate', columns='indexID', values='ret_' + str(N))
    ret['date'] = ret.index
    ret.reset_index(drop=True, inplace=True)
    ret['date'] = map(time_change, ret['date'])
    return ret

# 2.时间序列波动率类                               '%sB'%hold_window
def get_index_vol(date_list, ticker, N):
    """
    Args:
        date_list：月末序列
        ticker: 指数代码,如'000016'
        N: 时间窗口
    Returns:
        vol: 给定日期序列中给定指数的N日波动率
    """
    cal = Calendar('China.SSE')
    start = cal.advanceDate(date_list[0], '-%sB'%(N-1)).strftime('%Y%m%d')
    end = date_list[-1]
    temp = DataAPI.MktIdxdGet(ticker=ticker,beginDate=start,endDate=end,exchangeCD=u"XSHE,XSHG",field=u"tradeDate,CHGPct",pandas="1")
    temp['vol_' + str(ticker)] = pd.rolling_std(temp['CHGPct'], window=N)
    temp = temp[['tradeDate', 'vol_' + str(ticker)]]
    vol = temp.copy()
    vol['tradeDate'] = map(time_change, vol['tradeDate'])
    vol = vol[vol['tradeDate'].isin(date_list)]
    vol.columns = ['date', 'vol_' + str(ticker)] 
    return vol
    
# 3.截面波动率类    
def get_section_vol(date_list, all_stock, N):
    """
    Args:
        date_list：月初序列
        all_stock: 股票池数据,列名为日期、股票代码
        N: 时间窗口
    Returns:
        section_vol: 给定日期序列的市场股票N日收益率的波动率，
    """    
    cal = Calendar('China.SSE')
    s = []
    l = []
    for date in date_list:
        tdata = all_stock[all_stock['date'] == date]
        code = list(tdata['code'])
        end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')
        pre = cal.advanceDate(end, '-%sB'%N).strftime('%Y%m%d')
        temp1 = DataAPI.MktEqudAdjAfGet(secID=code,tradeDate=end,field=u"tradeDate,secID,closePrice",pandas="1")
        temp2 = DataAPI.MktEqudAdjAfGet(secID=code,tradeDate=pre,field=u"tradeDate,secID,closePrice",pandas="1")
        temp1['close_pre'] = temp2['closePrice']
        temp1['ret'] = temp1['closePrice'] / temp1['close_pre'] - 1    
        l.append(end)
        s.append(temp1['ret'].std()) 
    section_vol = pd.DataFrame({'date': l, 'section_vol_' + str(N): s})
    return section_vol

# 4.利率类
def get_rate_factor(start_date, end_date, month_end):
    """
    Args:
        start_date： 起始日期
        end_date: 结束日期
        month_end: 月末序列
    Returns:
        rate: 月末利率类数据
    """    
    # Shibor
    rate = DataAPI.MktIborGet(ticker=u"Shibor1W,Shibor2W,Shibor1M",beginDate=start_date,endDate=end_date,currency='CNY',field=u"tradeDate,secID,rate",pandas="1")
    rate = rate.pivot_table(index='tradeDate', columns='secID', values='rate')
    rate['date'] = rate.index
    rate.reset_index(drop=True, inplace=True)
    rate['date'] = map(time_change, rate['date'])
    rate = rate[rate['date'].isin(month_end)]
    # 信用利差
    temp1 = DataAPI.EcoDataProGet('1090001558',start_date,end_date)
    temp1 = temp1[['periodDate', 'dataValue']]
    temp1['periodDate'] = map(time_change, temp1['periodDate'])
    temp1.columns = ['date', 'short_medium_ret']
    temp2 = DataAPI.EcoDataProGet('1090001390',start_date,end_date)
    temp2 = temp2[['periodDate', 'dataValue']]
    temp2['periodDate'] = map(time_change, temp2['periodDate'])
    temp2.columns = ['date', 'bond_ret']
    temp = pd.merge(temp1, temp2, on='date')
    temp['credit_spread'] = temp['short_medium_ret'] - temp['bond_ret']
    temp = temp[temp['date'].isin(month_end)]
    temp = temp[['date', 'credit_spread']]
    rate = pd.merge(rate, temp, on='date')   
    return rate

# 5.货币类
def get_curr_factor(start_date, end_date, month_end):
    """
    Args:
        start_date： 起始日期
        end_date: 结束日期
        month_end: 月末序列
    Returns:
        curr: 月末货币类数据
    """
    # M1同比
    temp1 = DataAPI.EcoDataProGet('1070000007',start_date,end_date)
    temp1 = temp1[['periodDate', 'dataValue']]
    temp1['periodDate'] = map(time_change, temp1['periodDate'])
    temp1.columns = ['date', 'M1_yoy']
    # M2同比
    temp2 = DataAPI.EcoDataProGet('1070000009',start_date,end_date)
    temp2 = temp2[['periodDate', 'dataValue']]
    temp2['periodDate'] = map(time_change, temp2['periodDate'])
    temp2.columns = ['date', 'M2_yoy']
    curr = pd.merge(temp1, temp2, on='date')
    curr['curr'] = curr['M1_yoy'] - curr['M2_yoy']
    curr = curr[['date', 'curr']]
    curr.sort_values(by='date', inplace=True)
    curr.reset_index(drop=True, inplace=True)
    curr['date'] = month_end
    return curr

# 6.通货膨胀类
def get_inflation_factor(start_date, end_date, month_end):
    """
    Args:
        start_date： 起始日期
        end_date: 结束日期
        month_end: 月末序列
    Returns:
        inflation: 月末通胀类数据
    """
    # CPI同比
    temp1 = DataAPI.EcoDataProGet('1040000050',start_date,end_date)
    temp1 = temp1[['periodDate', 'dataValue']]
    temp1['periodDate'] = map(time_change, temp1['periodDate'])
    temp1.columns = ['date', 'CPI_yoy']
    # PPI同比
    temp2 = DataAPI.EcoDataProGet('1040000702',start_date,end_date)
    temp2 = temp2[['periodDate', 'dataValue']]
    temp2['periodDate'] = map(time_change, temp2['periodDate'])
    temp2.columns = ['date', 'PPI_yoy']
    inflation = pd.merge(temp1, temp2)
    inflation['tz'] = inflation['CPI_yoy'] - inflation['PPI_yoy']
    inflation = inflation[['date', 'tz']]
    inflation.sort_values(by='date', inplace=True)
    inflation.reset_index(drop=True, inplace=True)
    inflation['date'] = month_end
    return inflation

# 7.工业产业类
def get_indus_factor(start_date, end_date, month_end):
    """
    Args:
        start_date： 起始日期
        end_date: 结束日期
        month_end: 月末序列
    Returns:
        PMI: 月末PMI数据
        IAV: 月末IAV数据
    """
    # PMI
    PMI = DataAPI.EcoDataProGet('1030000011',start_date,end_date)
    PMI = PMI[['periodDate', 'dataValue']]
    PMI['periodDate'] = map(time_change, PMI['periodDate'])
    PMI = PMI[PMI['periodDate'] < end_date] # PMI当月底就直接公布了
    PMI.columns = ['date', 'PMI']
    PMI.sort_values(by='date', inplace=True)
    PMI.reset_index(drop=True, inplace=True)
    
    PMI['date'] = month_end
    # IAV
    #IAV = DataAPI.EcoDataProGet('1020000004',start_date,end_date)
    #IAV = IAV[['periodDate', 'dataValue']]
    #IAV['periodDate'] = map(time_change, IAV['periodDate'])
    #IAV.columns = ['date', 'IAV']
    #IAV.sort_values(by='date', inplace=True)
    #IAV.reset_index(drop=True, inplace=True)
   
    return PMI
def get_indus_factor1(start_date, end_date, month_end):
    """
    Args:
        start_date： 起始日期
        end_date: 结束日期
        month_end: 月末序列
    Returns:
        PMI: 月末PMI数据
        IAV: 月末IAV数据
    """
    # IAV
    IAV = DataAPI.EcoDataProGet('1020000004',start_date,end_date)
    IAV = IAV[['periodDate', 'dataValue']]
    IAV['periodDate'] = map(time_change, IAV['periodDate'])
    IAV.columns = ['date', 'IAV']
    IAV.sort_values(by='date', inplace=True)
    IAV.reset_index(drop=True, inplace=True)
   
    return IAV


# 日历类
def get_month(x):
    cal = Calendar('China.SSE')
    start = cal.advanceDate(x, '1B').strftime('%Y%m%d')
    date = datetime.datetime.strptime(start, '%Y%m%d')
    y = date.month
    return y


# 外生变量获取
tic = time.time()
# 上证50、沪深300、中证500过去20个交易日涨跌幅
N = 20
ticker_list = ['000016', '000300', '000905']
index_pct_chg = get_index_pct(month_end, ticker_list, N)
# 上证50、中证1000过去60日收益率标准差
N = 60
index_vol = pd.DataFrame({'date': month_end})
ticker_list = ['000016', '000852']
for ticker in ticker_list:
    temp = get_index_vol(month_end, ticker, N) 
    index_vol = pd.merge(index_vol, temp, on=['date']) 
# 全市场股票单日收益率标准差，全市场股票过去20个交易日收益率标准差
N_list = [1, 20]
section_vol = pd.DataFrame({'date': month_end})
for N in N_list:
    temp = get_section_vol(month_start, all_stock, N)
    section_vol = pd.merge(section_vol, temp, on=['date']) 
# SHIBOR1W、SHIBOR2W、SHIBOR1M、信用利差（1年AAA中短期票据收益率 - 1年期国债收益率）
start_date = month_end[0]
end_date = month_end[-1]
rate = get_rate_factor(start_date, end_date, month_end)
# M1同比 - M2同比,因数据次月公布，须平移一月
date = datetime.datetime.strptime(month_end[0], '%Y%m%d')
print(date)
start_date = datetime.datetime(date.year, date.month - 1, 1).strftime('%Y%m%d')
#end_date = month_end[-1]
#need to program to get rid this bug

#end_date ="20180924"
end_date ="20190924"
print(start_date)
print(end_date)
print(month_end)

curr = get_curr_factor(start_date, end_date, month_end)
# CPI同比 - PPI同比

start_date = month_end[0]
end_date = month_end[-1]

inflation = get_inflation_factor(start_date, end_date, month_end)
# 采购经理指数PMI、工业增加值当月同比IAV
date = datetime.datetime.strptime(month_end[0], '%Y%m%d')
start_date = datetime.datetime(date.year, date.month - 1, 1).strftime('%Y%m%d')
end_date = month_end[-1]

PMI = get_indus_factor(start_date, end_date, month_end)

start_date = month_end[0]
end_date = month_end[-1]

IAV = get_indus_factor1(start_date, end_date, month_end)
# IAV数据在07-10年1月份一月份免报，为了数据严谨考虑，我们补齐这部分数据
supplement = pd.DataFrame({'date':['20070131', '20080131', '20090131', '20100131', '20110131'], 'IAV': [24.71, 15.40, -2.93, 29.20, 13.30]})
IAV = IAV.append(supplement)
IAV.sort_values(by='date', inplace=True)
IAV.reset_index(drop=True, inplace=True)
IAV['date'] = month_end
industry = pd.merge(PMI, IAV, on='date')
# 月份
month = [get_month(x) for x in month_end]
calendar = pd.DataFrame({'date': month_end, 'Month': month})

# 整合外生变量数据
exogenous = pd.merge(index_vol, index_pct_chg, on='date')
exogenous = pd.merge(exogenous, section_vol, on='date')
exogenous = pd.merge(exogenous, rate, on='date')
exogenous = pd.merge(exogenous, curr, on='date')
exogenous = pd.merge(exogenous, inflation, on='date')
exogenous = pd.merge(exogenous, industry, on='date')
exogenous = pd.merge(exogenous, calendar, on='date') # 这就是原始的外生变量数据
# 外生变量处理
def get_period_data(currentdate, df, N): # N代表滚动窗口长度
    """
    Args:
        currentdate: 当期时间
        df: 数据，dataframe格式
        N: 前溯窗口长度
    Returns:
        wdata: 前溯N期（包括当期）的数据集合
    """
    l_date = sorted(list(set(df['date'])))
    date = [l_date[l_date.index(currentdate) - i] for i in range(N)]
    wdata = df[df['date'].isin(date)]
    return wdata

N = 24
all_date = sorted(list(set(exogenous['date'])))
exo_data = pd.DataFrame()
for i in range(N, len(all_date) + 1):    
    currentdate = all_date[i - 1]
    wdata = get_period_data(currentdate, exogenous, N)
    for col in wdata.columns[1: -1]: # 除月份以外
        wdata[col] = 100 * (wdata[col] - wdata[col].min()) / (wdata[col].max() - wdata[col].min())
    tdata = wdata[wdata['date'] == currentdate]
    exo_data = pd.concat([exo_data, tdata])

# 存储
exo_data.to_csv('MHKQ_data/exogenous.csv', index=False)
toc = time.time()
print('***************归一化外生变量示例***************')
print(exo_data.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


'''

第二部分：因子择时模型
该部分耗时 15分钟
该部分内容为本文因子择时模型的核心部分, 具体包括：

2.1 基于条件期望的因子择时模型原理说明

2.2 因子择时模型构建

2.3 因子择时模型结果分析

深度报告版权归优矿所有，禁止直接转载或编辑后转载。

2.1 因子择时模型原理

在2017年传统强势因子稳定性降低的环境下，市场对因子择时模型的需求愈发强烈，而实际应用因子择时模型则存在诸多困难：

难以平衡稳健的传统模型以及灵活的择时模型：由于传统的多因子模型极度适用于过去这种风格稳定的市场环境，基于传统模型构建的组合在历史上往往能够获得惊人的收益表现，但是因子择时模型在2014年12月以及2017年初等风格明显切换的市场环境能为组合带来较好的收益表现。因此，传统模型以及择时模型间的平衡一直是一个重要问题；
无法量化地将因子择时观点转化为因子权重：没有一个量化模型指导择时因子权重的分配，更多的是基于人为的主观判断；
无法有效筛选因子择时指标：择时指标并不是多多益善，缺乏一个稳定的筛选择时指标的模型。
因子择时的实质是因子权重的动态分配，我们可以看看常见的几种因子权重分配方法：
图片注释
可见，因子择时这一问题实际上就是如何对于因子的收益以及协方差矩阵进行预测，而引入外生变量改变因子收益与协方差矩阵的估计，使得模型对市场反应更及时。

基于条件期望的因子择时原理

我们认为因子收益和外生变量之间存在相关性，当给定外生变量取值时，可以求解出因子收益的条件期望，[Hua 2012]假设因子收益和外生变量满足联合正态分布，
那么我们就能计算出因子收益的条件期望和条件期望协方差矩阵，进而结合因子权重的加权方法完成因子的复合。（HKQ模型）

传统的最大化复合因子IC或者IR加权的方式其实可以看成对因子收益以及协方差矩阵的无条件期望，我们假设因子的收益向量R和外生变量向量V服从联合正态分布，即
图片注释
根据统计中多元条件期望的理论，在给定当期外生变量向量v时，因子收益的期望和协方差矩阵修正为
图片注释
其中
图片注释
由此
图片注释
其中s是归一化常数。

外生变量筛选原则

在实际的建模过程中，我们不会把所有的外生变量都纳入HKQ模型中，过多的外生变量也会造成模型的过拟合问题，因此我们考虑使用AIC信息准则来衡量模型的拟合优度，以此为
标准每期对外生变量进行筛选。

基于条件期望模型的AIC计算公式如下：
图片注释
其中，T为样本窗口长度，N为因子数量，K为外生变量数量，而当样本窗口长度T不显著大于外生变量K时，例如样本期为24个月，16个外生变量的情形下，基于AIC准则的筛选方法可能
会出现过拟合，这种情况下基于AICc(修正的AIC)进行变量筛选效果会更好，AICc是在AIC的基础上增加了一个误差修正量，如下：
图片注释

每个月末的变量筛选步骤如下：

Step1：初始化时，最优外生变量集合𝑆0为空，且初始AICc取值为AICc0 = T·log[|Σ𝑅𝑅|]；
Step2：对于每一个外生变量k，计算将其加入到𝑆𝑖后的AICc取值；
Step3：如果第二步中的最低AICc比现有AICc要低，则跳转步骤4，否则步骤结束，当前的外生变量集合𝑆𝑖即为筛选结果；
Step4：将具有最低AICc的外生变量k加入到𝑆𝑖，即 𝑆𝑖+1=𝑆𝑖∪𝑘，更新𝐴𝐼𝐶c𝑖+1为使用𝑆𝑖+1后模型的AICc值；
Step5：继续执行Step2。
每个月末，我们基于T个观察期的因子IC及外生变量数据，利用AICc准则筛选出当月的外生变量集合，利用MHKQ条件期望模型得到𝑅|𝑣，结合最大化IC的方法完成多因子的加权
（PS：协方差矩阵的估计误差较大，因此这里我们只用𝑅|𝑣来进行IC加权，读者可自行根据压缩估计等方法来降低Σ的估计误差，进行复合IR加权。）


2.2 因子择时模型构建

模型时间周期说明

我们将模型的观察期T定为24个月，以第一个月（2010年12月末）为例，如下图所示：
图片注释
以后的每一个月都滚动地执行因子择时模型，由此就可以得到每个月模型输出的因子权重向量，结合每期因子暴露，合成的复合因子格式如下：
图片注释
factor_IC和factor_MHKQ分别是无条件期望和条件期望下的复合因子值。

'''


# 模型构建

# 计算因子IC的函数
def cal_ic(factor_name, data):
    """
    Args:
        factor_name: 需要计算IC的因子名称
        data: 因子数据，至少需要3列：日期、股票因子值、股票下期收益('Month_ret')
    Returns:
        IC_data: IC结果，dataframe格式，列名为日期，IC值
    """
    IC = []    
    date_list = sorted(list(set(data['date'])))
    for date in date_list:
        tdata = data[data['date'] == date]
        tdata.sort_values(by='code', inplace=True)
        tdata.reset_index(drop=True, inplace=True)        
        factor = np.array(tdata[factor_name])
        ret = np.array(tdata['Month_ret'])
        cor, pval = st.spearmanr(factor, ret) # spearman秩相关系数
        IC.append(cor)
    IC_data = pd.DataFrame({'date': date_list, 'IC_' + factor_name: IC})
    IC_data = IC_data[['date', 'IC_' + factor_name]]
    return IC_data

# 计算模型AICc值的函数
def cal_aicc(IC_data, exo_data, s, T):
    """
    Args:
        IC_data: 因子IC序列，dataframe格式，columns=['date', 'factor_name_1',...'factor_name_N']
        exo_data：外生变量序列，dataframe格式， columns=['date', 'exogenous1',..., 'exogenous_N']
        s: 外生变量变量列表，形如['section_vol_20', 'PMI']
        T: 时间窗口
    Returns:
        AICc：在外生变量集合为s时的模型的AICc值
    """
    s.append('date')
    IC_and_exo = pd.merge(IC_data, exo_data[s], on='date')
    all_matrix = IC_and_exo.iloc[:, 1:]
    all_sigma = np.cov(all_matrix.T)
    # 对协方差矩阵进行分割
    n = len(list(IC_data.columns[1: ])) # n就是因子的数量
    # 将all_sigma矩阵分割成4个部分
    sigma_RR = np.matrix(all_sigma[: n, : n])
    sigma_RV = np.matrix(all_sigma[: n, n: ])
    sigma_VR = np.matrix(all_sigma[n: , : n])
    sigma_VV = np.matrix(all_sigma[n: , n: ])
    # 条件期望的协方差矩阵sigma_v
    sigma_v = sigma_RR - sigma_RV*(sigma_VV.I)*sigma_VR
    # 因子数量k
    k = sigma_RV.shape[1]
    AICc = T*np.log(np.linalg.det(sigma_v)) + 2*n*k + 2*k*(k+1)/(T-k-1)
    return AICc

# 外生变量筛选函数
def exogenous_variables_select(IC_data, exo_data, N):
    """
    Args:
        IC_data: 因子IC序列，dataframe格式，columns=['date', 'factor_name_1',...'factor_name_N']
        exo_data：外生变量序列，dataframe格式， columns=['date', 'exogenous1',..., 'exogenous_N']
        N: 模型窗口参数
    Returns:
        select：筛选出的外生变量名列表(list)
    """
    # 计算初始AICc
    T = N
    n = len(list(IC_data.columns[1: ])) # 因子数目
    K = 0
    IC_matrix = IC_data.iloc[:, 1:]
    sigma = np.cov(IC_matrix.T)
    AICc_0 = T*np.log(np.linalg.det(sigma))

    select = [] # 记录选中的外生变量
    
    # 第一次筛选
    AICc_list = []
    for name in exo_data.columns[1: ]:
        AICc = cal_aicc(IC_data, exo_data, [name], T)
        AICc_list.append(AICc)
    compare = pd.DataFrame({'name': list(exo_data.columns[1: ]), 'AICc': AICc_list}) # 分别加入不同变量后的模型AICc值
    compare.sort_values(by='AICc', inplace=True)
    compare.reset_index(drop=True, inplace=True)
    # 判断是否加入变量
    if compare.iat[0, 0] < AICc_0:
        select.append(compare.iat[0, 1])
        AICc_0 = compare.iat[0, 0]
        
    # 后续筛选
    for j in range(9): # 已经做过第一次筛选了
        unselect = list(set(exo_data.columns[1: ]).difference(set(select)))
        AICc_list = []
        for name in unselect:
            select1 = copy.deepcopy(select)
            select1.append(name)
            AICc = cal_aicc(IC_data, exo_data, select1, T)
            AICc_list.append(AICc)
        compare = pd.DataFrame({'name': unselect, 'AICc': AICc_list})
        compare.sort_values(by='AICc', inplace=True)
        compare.reset_index(drop=True, inplace=True)        
        if compare.iat[0, 0] < AICc_0:
            select.append(compare.iat[0, 1])
            AICc_0 = compare.iat[0, 0]   
        else:
            break
    return select

# 计算条件期望的函数
def cal_conditional_expectation(IC_data, exo_data, s, v):
    """
    Args:
        IC_data: 因子IC序列，dataframe格式，columns=['date', 'factor_name_1',...'factor_name_N']
        exo_data：外生变量序列，dataframe格式， columns=['date', 'exogenous1',..., 'exogenous_N']
        s: 外生变量变量列表，形如['section_vol_20', 'PMI']
        v:当期的外生变量， 列表格式
    Returns:
        R_v：当期的条件期望下的因子权重
    """
    s.append('date')
    IC_and_exo = pd.merge(IC_data, exo_data[s], on='date')
    all_matrix = IC_and_exo.iloc[:, 1:]
    all_sigma = np.cov(all_matrix.T)
    # 对协方差矩阵进行分割
    n = len(list(IC_data.columns[1: ])) # n就是因子的数量
    # 将all_sigma矩阵分割成4个部分
    sigma_RR = np.matrix(all_sigma[: n, : n])
    sigma_RV = np.matrix(all_sigma[: n, n: ])
    sigma_VR = np.matrix(all_sigma[n: , : n])
    sigma_VV = np.matrix(all_sigma[n: , n: ])
    # 条件期望的协方差矩阵sigma_v
    sigma_v = sigma_RR - sigma_RV*(sigma_VV.I)*sigma_VR
    # 无条件期望
    R_mean = list((init[init.columns[1: ]].mean()).to_frame(name='mean_IC')['mean_IC'])
    wdata_select = wdata[select]
    V_mean = list((wdata_select[wdata_select.columns[: -1]].mean()).to_frame(name='mean_V')['mean_V']) # 外生变量均值向量
    delta_R = sigma_RV*(sigma_VV.I)*(np.matrix(list(map(lambda x: x[0]-x[1], zip(v, V_mean)))).T)
    delta_R = delta_R.T.tolist()[0]
    R_v = list(map(lambda x: x[0]+x[1], zip(R_mean, delta_R))) # 这就是权重向量
    return R_v

# 主函数
tic = time.time()
date_list = sorted([x for x in month_end if x >= '20101128'])
weight_IC = []
weight_MHKQ = []
N = 24 # 建立模型的时间跨度
for i in range(len(date_list) - 1):
    date = date_list[i] # 模型样本期最后一期的日期
    currentdate = date_list[i + 1] # 当期日期
    wdata = get_period_data(date, exo_data, N) # date向前回溯N期的外生变量数据(包括date当天)
    period_date = list(wdata['date'])
    period = factor_stand[factor_stand['date'].isin(period_date)] # 同期的标准化因子数据
    # init就是过去N个月各因子的IC序列
    init = pd.DataFrame({'date': period_date})
    for factor_name in period.columns[2: -1]:
        temp = cal_ic(factor_name, period)
        init = pd.merge(init, temp, on='date')  
    # 外生变量筛选
    select = exogenous_variables_select(init, wdata, N)
    # 计算条件期望
    if len(select) > 0:
        v = exo_data[exo_data['date'] == currentdate]
        v = v[select]
        v.reset_index(drop=True, inplace=True)
        v = list(v.iloc[0, :]) # 当期的外生变量
        R_v = cal_conditional_expectation(init, wdata, select, v)    
    else:
        R_v = list((init[init.columns[1: ]].mean()).to_frame(name='mean_IC')['mean_IC'])
    # 无条件期望
    R_mean = list((init[init.columns[1: ]].mean()).to_frame(name='mean_IC')['mean_IC'])
    # 分别记录两种权重数据
    weight_IC.append(R_mean)
    weight_MHKQ.append(R_v)
    print (currentdate, select) # 输出每期的外生变量筛选结果)
    
# 记录权重数据
# IC权重
weight_IC = pd.DataFrame(weight_IC, columns=factor_stand.columns[2: -1])
weight_IC['date'] = [y for y in month_end if y >= '20101231']
weight_IC.to_csv('MHKQ_data/weight_IC.csv', index=False)
# 择时模型权重
weight_MHKQ = pd.DataFrame(weight_MHKQ, columns=factor_stand.columns[2: -1])
weight_MHKQ['date'] = [y for y in month_end if y >= '20101231']
weight_MHKQ.to_csv('MHKQ_data/weight_MHKQ.csv', index=False)
print('***********因子权重示例************')
print(weight_MHKQ.head(10).to_html())

# 根据权重及因子暴露计算复合因子 
mixed_factor = pd.DataFrame()
for date in sorted(list(weight_MHKQ['date'])):
    # 每期因子暴露
    single_day_factor = factor_stand[factor_stand['date'] == date]
    single_day_factor.reset_index(drop=True, inplace=True)
    factor_loading = np.array(single_day_factor.iloc[:, 2: -1])
    # 权重
    w_MHKQ = weight_MHKQ[weight_MHKQ['date'] == date]
    w_IC = weight_IC[weight_IC['date'] == date]
    w_MHKQ = list((w_MHKQ.iloc[0, : -1]).to_frame(name='weight')['weight'])
    w_IC= list((w_IC.iloc[0, : -1]).to_frame(name='weight')['weight'])
    # 复合
    factor_MHKQ = np.dot(factor_loading, w_MHKQ)
    factor_IC = np.dot(factor_loading, w_IC)
    single_day_factor['factor_MHKQ'] = factor_MHKQ
    single_day_factor['factor_IC'] = factor_IC
    single_day_factor = single_day_factor[['date', 'code', 'factor_IC', 'factor_MHKQ']]
    mixed_factor = pd.concat([mixed_factor, single_day_factor], axis=0)
    
mixed_factor.to_csv('MHKQ_data/Mixed_factor.csv', index=False)
toc = time.time()
print('***********复合因子权重示例************')
print(mixed_factor.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


'''
2.2 因子择时模型结果分析

因子权重分析

1.市值因子
图片注释
可见，基于IC加权时，因子权重是过去T个观察期的IC的均值，权重序列一直很平稳，对市场的反应极为缓慢，现在看择时模型的市值权重，模型在2014年底将市值权重由负转正，并在15年一开始又立马转为负，
这一段刚好契合A股在2014年底的短暂大盘行情以及2015年众所周知的小票行情；不仅如此，模型在2017年初，迅速地将权重调由负转正，并在2017年全年大部分时间里面为正，这和去年的大盘行情
完全一致，同时IC模型在2017年底才缓慢地将市值权重转正，可此时行情已经结束，凸显出择时模型的优越性。

2.反转因子与特异度因子
反转
图片注释
特异度
图片注释
将这两个因子放在一起是因为这两个技术面的因子在历史上一直稳定有效，直到2017年初两者又同时开始失效，由图可见，我们的择时模型在17年初突然对因子进行正向调整，在年中半数以上的时间内
将两因子的权重定为正，结合市值权重的调整，使得量化多因子的回测组合在2017年不仅不亏，还能拥有不俗的表现。

3.换手、BP、roe_ttm与单季度净利润增速因子一直比较稳定，在17年也没有大的波动，择时模型对这几个因子的权重和IC模型几乎一致。

'''


#　结果分析函数

# 权重序列图
def plot_weight_series(weight, factor_name):
    """
    Args:
        weight：因子权重数据，dataframe格式，列名如['date', 'mkt_IC', 'mkt_MHKQ']
        factor_name：因子名(str)
    Returns:
        权重走势图输出
    """
    w = weight[['date', factor_name + '_IC', factor_name + '_MHKQ']]
    w['date'] = pd.to_datetime(w['date'])
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(111)
    ax1.plot(w['date'], w[factor_name + '_IC'], label=factor_name + '_IC')
    ax1.plot(w['date'], w[factor_name + '_MHKQ'], label=factor_name + '_MHKQ')
    ax1.set_ylabel(u'weight', fontsize=16)
    ax1.set_xlabel(u'date', fontsize=16)
    ax1.set_title(u"capital curve", fontsize=16)
    plt.grid(True)
    plt.legend()

tic = time.time()
weight = pd.merge(weight_IC, weight_MHKQ, on='date',  suffixes=('_IC', '_MHKQ'))
# 指定某个因子权重序列输出
factor_name = 'mkt'
plot_weight_series(weight, factor_name)
# 可视化
# 计算复合因子IC
mixed_factor = pd.merge(mixed_factor, factor_stand[['date', 'code', 'Month_ret']], on=['date', 'code'])
# 原始权重
ic_init = cal_ic('factor_IC', mixed_factor)
weight_IC = pd.merge(weight_IC, ic_init)
weight_IC = weight_IC[['date', 'mkt', 'ret_20', 'mean_to', 'IVR', 'BP', 'roe_ttm', 'qfa_yoyprofit', 'IC_factor_IC']]
weight_IC = weight_IC[(weight_IC['date'] >= '20161230') & (weight_IC['date'] <= '20171130')]
weight_IC.columns = ['月末日期', '市值权重', '反转权重', '换手率权重', '特异度权重', 'BP权重', 'REO_TTM权重', '净利润增速权重', '复合因子IC']
weight_IC.reset_index(drop=True, inplace=True)
print('*********************原始IC复合因子17年权重**********************')
print(weight_IC.to_html())
ic_mhkq = cal_ic('factor_MHKQ', mixed_factor)
weight_MHKQ = pd.merge(weight_MHKQ, ic_mhkq)
weight_MHKQ = weight_MHKQ[['date', 'mkt', 'ret_20', 'mean_to', 'IVR', 'BP', 'roe_ttm', 'qfa_yoyprofit', 'IC_factor_MHKQ']]
weight_MHKQ = weight_MHKQ[(weight_MHKQ['date'] >= '20161230') & (weight_MHKQ['date'] <= '20171130')]
weight_MHKQ.columns = ['月末日期', '市值权重', '反转权重', '换手率权重', '特异度权重', 'BP权重', 'REO_TTM权重', '净利润增速权重', '复合因子IC']
weight_MHKQ.reset_index(drop=True, inplace=True)
toc = time.time()
print('*********************择时模型复合因子17年权重**********************')
print(weight_MHKQ.to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''
第三部分：策略回测
该部分耗时 15分钟 (3个组合的回测，每个回测5分钟左右)
该部分内容为对因子结果的回测, 具体包括：

3.1 对原始IC以及择时模型的复合因子进行回测，并进行对比分析

3.2 基于择时模型构建指数增强组合

深度报告版权归优矿所有，禁止直接转载或编辑后转载。


3.1 复合因子组合回测

我们按照如上计算出的facotr_IC以及factor_MHKQ分别进行回测（进行对比），调仓空间定义如下：

全A非ST股票
上市时间6个月以上
调仓当天非停牌，非一字涨跌停
回测的参数如下：

回测时间2011年1月4日 ~ 2018年5月31日
每次选取样本空间中的100只股票等权重分配作为持仓
交易成本双边千分之二


'''

def plot_under_water(bt, title):
    """
    绘制回撤及收益率曲线图
    输入：
        bt：quartz回测结束自动生成的dict
        title：str
    返回：
        ax：matplotlib figure 对象
    """
    bt_quantile_ten = bt
    data = bt_quantile_ten[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()

    df_cum_rets = data['excess']
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -((running_max - df_cum_rets) / running_max)
    underwater.index = data['tradeDate']

    fig = plt.figure(figsize=(12, 5))
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    x = range(len(underwater))
    ax2.grid(False)
    ax1.set_ylim(-0.30, 0)
    ax1.set_ylabel(u'回撤', fontproperties=font, fontsize=16)
    ax1.fill_between(underwater.index, 0, np.array(underwater), color='#000066', alpha=1)
    ax2.set_ylabel(u'净值', fontproperties=font, fontsize=16)
    ax2.plot(data['tradeDate'], data[['excess']], label='hedged(right)', color='r')
    ax2.set_ylim(bottom=0.9, top=7)
    s = ax1.set_title(title, fontproperties=font, fontsize=16)
    return fig

factor_name = 'factor_IC'
factor = pd.read_csv('MHKQ_data/Mixed_factor.csv')
factor['date'] = map(str, factor['date'])
factor = factor[['date', 'code', factor_name]]
factor.columns = ['date', 'secID', factor_name]

def time_change(x):
    y = datetime.datetime.strptime(x, '%Y%m%d')
    y = y.strftime('%Y-%m-%d')
    return y

factor['date'] = map(time_change, factor['date'])
factor = factor.pivot_table(index='date', columns='secID', values=factor_name)
factor_pure = factor.copy() # fator_pure就是输入优矿回测框架的因子数据


#start = '2010-12-31'                       # 回测起始时间
#end = '2018-05-31'                         # 回测结束时间

start = '2010-02-01'                       # 回测起始时间
end = '2019-09-24'
                         # 回测结束时间


benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('A')        # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                          # 调仓频率，表示执行handle_data的时间间隔

factor_dates = factor_pure.index.values
  
commission = Commission(0.001, 0.001)     # 交易费率设为双边千分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in factor_dates:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    signal = pd.Series(dict(factor_pure.ix[pre_date, account.universe].dropna()))
    # signal = dict(signal)
    
    # 取前100只等权组合
    signal.sort_values(ascending=False, inplace=True)
    signal = signal[0: 100]
    wts = pd.Series([0.01] * 100, index=signal.index)
    wts = dict(wts)
    
    # 组合构建                
    # wts = long_only(signal, select_type=0, top_ratio=0.1, weight_type=0, target_date=''.join(pre_date.split('-')), universe_type='ZZ500')
    
    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        account.order_to(stk, 0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        account.order(stock, change[stock])
        


# 上图就是因子IC加权组合的回测曲线
plot_under_water(bt, title=u"原始IC加权组合曲线")
# 下图中，阴影部分对应着当前时点的回撤

factor_name = 'factor_MHKQ'
factor = pd.read_csv('MHKQ_data/Mixed_factor.csv')
factor['date'] = map(str, factor['date'])
factor = factor[['date', 'code', factor_name]]
factor.columns = ['date', 'secID', factor_name]

def time_change(x):
    y = datetime.datetime.strptime(x, '%Y%m%d')
    y = y.strftime('%Y-%m-%d')
    return y

factor['date'] = map(time_change, factor['date'])
factor = factor.pivot_table(index='date', columns='secID', values=factor_name)
factor_pure = factor.copy() # fator_pure就是输入优矿回测框架的因子数据

#start = '2010-12-31'                       # 回测起始时间
#end = '2018-05-31'                         # 回测结束时间

start = '2010-02-01'                       # 回测起始时间
end = '2019-09-24'

benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('A')        # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                          # 调仓频率，表示执行handle_data的时间间隔

factor_dates = factor_pure.index.values
  
commission = Commission(0.001, 0.001)     # 交易费率设为双边千分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in factor_dates:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    signal = pd.Series(dict(factor_pure.ix[pre_date, account.universe].dropna()))
    # signal = dict(signal)
    
    # 取前100只等权组合
    signal.sort_values(ascending=False, inplace=True)
    signal = signal[0: 100]
    wts = pd.Series([0.01] * 100, index=signal.index)
    wts = dict(wts)
    
    # 组合构建                
    # wts = long_only(signal, select_type=0, top_ratio=0.1, weight_type=0, target_date=''.join(pre_date.split('-')), universe_type='ZZ500')
    
    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        account.order_to(stk, 0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        account.order(stock, change[stock])
        
        
plot_under_water(bt, title=u"因子择时模型组合曲线")
# 下图中，阴影部分对应着当前时点的回撤
'''

策略结果分析

综合对比两种模型可以发现：

IC加权组合在2017年以前能获得较高的超额收益，而在市场变化较快的2014年底相对于中证500指数有一个较大的回撤，并且在2017年初至今一直处于回撤阶段；
因子择时组合的表现较为稳定，年化超额收益24.0%，略低于上述组合的25.7%，得益于因子的正确择时，组合在2014年底及2017年全年表现均远远好于IC组合，净值走势很稳健；
IC加权组合2017年绝对收益为-21.58%，因子择时组合2017年绝对收益为+10.06%，因子择时在风格切换市场的优越性不言而喻。
3.2 基于择时模型构建指数增强组合

指数增强组合

我们这里尝试因子择时模型的结果去构建指数增强组合，在月末调仓时对指数做行业中性化：

每个行业选择因子值最大的前20%配置，行业内等权，行业间参照基准权重
行业分类采用申万一级
对停牌或涨跌停做处理
交易费率双边千二
按照如上规则构建中证500成分内增强组合。



'''        


# 指数成分内因子

def get_cons_stock(data, date, index_ticker):
    """
    Args:
        data：横截面因子数据
        date：日期
        index_ticker：需要筛选成分的指数ID
    Returns:
        cons_stock：成分股因子数据
    """
    cons_id = DataAPI.IdxConsGet(ticker=index_ticker,intoDate=date,field=u"consID",pandas="1")
    cons_stock = data[data['code'].isin(list(cons_id['consID']))]
    return cons_stock

# 筛选成分股
mixed_factor = pd.read_csv('MHKQ_data/Mixed_factor.csv')
mixed_factor['date'] = map(str, mixed_factor['date'])
date_list = sorted(list(set(mixed_factor['date'])))
stock_in_index = pd.DataFrame()
for date in date_list:
    tdata = mixed_factor[mixed_factor['date'] == date]
    cons_stock = get_cons_stock(tdata, date, '000905')
    stock_in_index = stock_in_index.append(cons_stock)     
    
factor_name = 'factor_MHKQ'
stock_in_index = stock_in_index[['date', 'code', factor_name]]
stock_in_index.columns = ['date', 'secID', factor_name]

def time_change(x):
    y = datetime.datetime.strptime(x, '%Y%m%d')
    y = y.strftime('%Y-%m-%d')
    return y

stock_in_index['date'] = map(time_change, stock_in_index['date'])
stock_in_index = stock_in_index.pivot_table(index='date', columns='secID', values=factor_name)


#start = '2010-12-31'                       # 回测起始时间
#end = '2018-07-03'                         # 回测结束时间

start = '2010-02-01'                       # 回测起始时间
end = '2019-09-24'


benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('A')        # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                          # 调仓频率，表示执行handle_data的时间间隔

factor_dates = stock_in_index.index.values
  
commission = Commission(0.001, 0.001)     # 交易费率设为双边千分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in factor_dates:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    signal = pd.Series(dict(stock_in_index.ix[pre_date, account.universe].dropna()))
    signal = dict(signal)
    
    # 组合构建                
    wts = long_only(signal, select_type=1, top_ratio=0.2, weight_type=0, target_date=''.join(pre_date.split('-')), universe_type='ZZ500')
    
    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        account.order_to(stk, 0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        account.order(stock, change[stock])
        
def plot_under_water_enhance(bt, title):
    """
    绘制回撤及收益率曲线图
    输入：
        bt：quartz回测结束自动生成的dict
        title：str
    返回：
        ax：matplotlib figure 对象
    """
    bt_quantile_ten = bt
    data = bt_quantile_ten[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()

    df_cum_rets = data['excess']
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -((running_max - df_cum_rets) / running_max)
    underwater.index = data['tradeDate']

    fig = plt.figure(figsize=(12, 5))
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    x = range(len(underwater))
    ax2.grid(False)
    ax1.set_ylim(-0.30, 0)
    ax1.set_ylabel(u'回撤', fontproperties=font, fontsize=16)
    ax1.fill_between(underwater.index, 0, np.array(underwater), color='#000066', alpha=1)
    ax2.set_ylabel(u'净值', fontproperties=font, fontsize=16)
    ax2.plot(data['tradeDate'], data[['excess']], label='hedged(right)', color='r')
    ax2.set_ylim(bottom=0.9, top=3)
    s = ax1.set_title(title, fontproperties=font, fontsize=16)
    return fig

plot_under_water_enhance(bt, title=u"中证500增强组合回测曲线")


