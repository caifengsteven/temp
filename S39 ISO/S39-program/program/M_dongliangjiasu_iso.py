# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:35:57 2020
动量加速项目
文献原始做法
我们使用后复权月度数据统计收益，作者使用每日前复权数据计算收益，结果差距比较大，但是趋势相同
@author: adair2019
"""
import os
import pandas as pd
from yq_toolsS45 import time_use_tool
from yq_toolsS45 import create_db
from yq_toolsS45 import get_file_name
from sqlalchemy.types import NVARCHAR, Float, Integer,DATE
from tqdm import tqdm
from yq_toolsS45 import do_sql_order

eg_pro = create_db('data_pro')
#tn = 'main_index_zx02_v2'
tn = 'main_index_s68'
sql_t0 = 'select * from %s where index_id = "%s" order by tradeDate'
sub_index_id ='HSI'
tmp_t0 = pd.read_sql(sql_t0 % (tn,sub_index_id),eg_pro)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
# import statsmodels.api as sm
# import math

from sqlalchemy import create_engine
import json
from datetime import datetime, timedelta

# import multiprocessing as mp
# pool = mp.Pool(8)
t0 = datetime.now()
import warnings

warnings.filterwarnings('ignore')

# must be set before using
with open('para.json', 'r', encoding='utf-8') as f:
    para = json.load(f)

pn = para['yuqerdata_dir']

user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name1 = 'yuqerdata'
# eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str = 'mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name, pass_wd, port, db_name1)
engine = create_engine(eng_str)

db_name2 = 's37'
# eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str = 'mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name, pass_wd, port, db_name1)
engine_s37 = create_engine(eng_str)
tn_symbol = 'symbol_pool_S39'
# 设置参数
## 数据的起始与终止时间
begin = '2010-01-01'
end = '2020-04-01'
## 股票池的信息日期
info_date = '2015-07-01'
delta_1year = timedelta(days=365)
info_date_1year_before = datetime.strptime(info_date, '%Y-%m-%d') - delta_1year
info_date_1year_before = info_date_1year_before.strftime('%Y-%m-%d')
## 回测起始时间
# back_date = '2008-06-01'
import datetime

back_date = datetime.date(2010, 6, 1)
## 选择回测区间，选择A或者hs300
flag_m = 'A'


# flag_m = 'hs_300'
# flag_m = 'hs_500'

# 数字型股票代码补充0
def add_0(x):
    if isinstance(x, int):
        x = '%0.6d' % x
    else:
        x = x.rjust(6, '0')
    return x


##股票选取
# 获取每日数据
# 获取基本数据
sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''

def chs_factor(ticker='00005',begin=begin,end = end,)
def chs_factor(ticker='000005', begin=begin, end=end,
               field=[u'symbol', u'tradeDate', u'openPrice',
                      u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol',
                      u'turnoverValue', u'dealAmount', u'chgPct',
                      'turnoverRate', u'marketValue', u'accumAdjFactor']):
    sql_str1 = sql_str_select_data1 % (','.join(field), ticker, begin, end)
    dataday = pd.read_sql(sql_str1, engine)
    dataday = dataday.applymap(lambda x: np.nan if x == 0 else x)
    dataday.rename(columns={'symbol': 'ticker'}, inplace=True)
    ## 对数据补全
    return dataday.fillna(method='ffill')


# 获取月度数据收益
# 原文根据每日数据收益转化月度数据，我们直接读取月度收益即可
sql_str_month_data1 = '''select %s from MktEqumAdjAfGet where ticker="%s" and monthBeginDate>="%s"
    and endDate<="%s" and tradeDays>0 order by endDate'''


def get_month_return(ticker='000005', begin=begin, end=end,
                     field=[u'ticker', u'endDate', u'preClosePrice',
                            u'closePrice']):
    sql_str1 = sql_str_month_data1 % (','.join(field), ticker, begin, end)
    df_month = pd.read_sql(sql_str1, engine)
    df_month.index = df_month['endDate'].values
    # df_close = df_month['closePrice']
    # df_return = df_month['closePrice'] /df_month['preClosePrice'] - 1
    df_return = df_month['closePrice'] / df_month['closePrice'].shift(1) - 1
    return df_return.fillna(method='bfill')  # df_return.fillna(method = 'bfill')


### 得到月度日历
def get_month_calender():
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" order by endDate'''
    x = pd.read_sql(sql_str, engine)
    x = x['endDate'].values
    # b=[i.strftime('%Y-%m-%d') for i in x]
    return x


cal_data = get_month_calender()
cal_dict = {i.strftime('%Y-%m-%d'): True for i in cal_data}  # 链式引用


## 获取300股票代码
def get_IdxCons(intoDate, ticker='000300'):
    # nearst 时间
    sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
            and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
        ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' % (ticker,
                                                                                    ticker, intoDate)
    x = pd.read_sql(sql_str1, engine)
    x = x['symbol'].values
    return x


## 计算动量加速度
def moment_speed(df_return, n=18):
    ## 几何平均
    return df_return.rolling(n).apply(lambda x: np.prod(np.power(1 + x, 1.0 / n)) - 1)


def curve_fit(df_return, n=18):
    ## 二次曲线拟合
    # return df_return.rolling(n).apply(lambda x: np.polyfit(np.arange(1, n + 1), x, 2, full = False)[-1])
    return df_return.rolling(n).apply(lambda x: np.polyfit(np.arange(1, n + 1), x, 2, full=False)[0])


# 收益分组
def get_top_bot_group(s):
    bottom = s[s < s.quantile(0.3)].index
    top = s[s > s.quantile(0.7)].index
    return top, bottom


def get_top_bot_group_per(s, percent):
    bottom = s[s < s.quantile(percent)].index
    top = s[s > s.quantile(1 - percent)].index
    return top, bottom


# 曲线参数统计
def result(v1, v2):
    d1 = v1[1:] / v1[:-1] - 1
    d2 = v2[1:] / v2[:-1] - 1
    beta, alpha = np.polyfit(d1, d2, 1)
    ratio = (v2[-1] - 1) / v2.shape[0] * 12
    sigma = d2.std() * np.sqrt(12)
    info = ratio / sigma
    hedge_index = (d2 - beta * d1)
    hedge_ratio = ((hedge_index + 1).prod() - 1) / v2.shape[0] * 12
    hedge_sigma = hedge_index.std() * np.sqrt(12)
    hedge_info = hedge_ratio / hedge_sigma
    return ratio, info, hedge_ratio, hedge_info


hs_300 = get_IdxCons(info_date)
hs_500 = get_IdxCons(info_date, '000905')
# 上证综指成分股
# 排除B股和次新股， B股我们写入数据库的时候已经去掉了，现在是排除回测初始时间点的次新股
hs_a = get_IdxCons(info_date, '000001')
## 排除st与B股
sql_str_st = 'select distinct(ticker) from yuqerdata.st_info where tradedate = "%s"'
st = pd.read_sql(sql_str_st % (info_date), engine)
st['ticker'] = st['ticker'].apply(add_0)
hs_a = np.setdiff1d(hs_a, st)
## 排除次新股 上市1年以内我们认为是次新股
sql_str = '''select distinct(ticker)  from yuqerdata.equget  where equTypeCD = "A" 
and ListSectorCD<=3 and  listDate <="%s" order by ticker'''
symbol_a = pd.read_sql(sql_str % info_date_1year_before, engine).values
hs_a = np.intersect1d(symbol_a, hs_a)

# 计算因子数据、记录每月收益数据
speed_list = []
curve_list = []
return_list = []
flag = 0
## 此处讲hs_300_dict改成hs_a_dict 即可回测全市场
if flag_m == 'A':
    hs_pool = hs_a
elif flag_m == 'hs_300':
    hs_pool = hs_300
elif flag_m == 'hs_500':
    hs_pool = hs_500
for ticker in hs_pool:
    try:
        df_return = get_month_return(ticker=ticker)
        df_speed = moment_speed(df_return)
        df_curve = curve_fit(df_return)
        df_speed.name = ticker
        df_curve.name = ticker
        df_return.name = ticker
        speed_list.append(df_speed)
        curve_list.append(df_curve)
        return_list.append(df_return)
        flag += 1
        if flag % 10 == 0:
            print(u'%s 股票计算完毕' % flag)
    except:
        print(ticker, u'数据缺失，影响计算进而剔除')
        continue

speed_df = pd.concat(speed_list, axis=1)
curve_df = pd.concat(curve_list, axis=1)
return_df = pd.concat(return_list, axis=1)

speed_df = speed_df[speed_df.index > back_date]
curve_df = curve_df[curve_df.index > back_date]
return_df = return_df[return_df.index > back_date]

## 计算动量组 根据动量区分上下组
if flag_m == 'A':
    top_bot_df = speed_df.apply(lambda x: get_top_bot_group_per(x.dropna(), 0.3), axis=1)
else:
    # top_bot_df = speed_df.apply(lambda x: get_top_bot_group_per(x.dropna(),0.05), axis = 1)
    top_bot_df = speed_df.apply(lambda x: get_top_bot_group_per(x.dropna(), 0.3), axis=1)

## 计算加减速
top = top_bot_df.apply(lambda x: x[0]).to_dict()
bot = top_bot_df.apply(lambda x: x[1]).to_dict()

## 统计最后股票性质 根据曲线拟合结果识别上升下降
fin_dict = {}
for i in curve_df.index:
    ## 新建一个嵌套词典
    fin_dict[i] = {'top_up': [], 'top_down': [], 'bot_up': [], 'bot_down': []}
    ## 计算top up down
    s = curve_df.loc[i, top[i]]
    s_up = s[s > 0].index
    s_down = s[s < 0].index
    fin_dict[i]['top_up'] = s_up
    fin_dict[i]['top_down'] = s_down
    ## 计算bot up down
    s = curve_df.loc[i, bot[i]]
    s_up = s[s > 0].index
    s_down = s[s < 0].index
    fin_dict[i]['bot_up'] = s_up
    fin_dict[i]['bot_down'] = s_down

## 下个月收益基于这个月的决策
redi_date_dict = {i: j for i, j in zip(return_df.index[1:], return_df.index[:-1])}

top_up_return = []
top_down_return = []
bot_up_return = []
bot_down_return = []
for i in return_df.index[1:]:
    date = redi_date_dict[i]
    ## top_up
    tickers = fin_dict[date]['top_up']
    if len(tickers) != 0:
        top_up_return.append(return_df.loc[i, tickers].mean())
    else:
        top_up_return.append(0)
    ## top_down
    tickers = fin_dict[date]['top_down']
    if len(tickers) != 0:
        top_down_return.append(return_df.loc[i, tickers].mean())
    else:
        top_down_return.append(0)
    ## bot_up
    tickers = fin_dict[date]['bot_up']
    if len(tickers) != 0:
        bot_up_return.append(return_df.loc[i, tickers].mean())
    else:
        bot_up_return.append(0)
    ## bot_down
    tickers = fin_dict[date]['bot_down']
    if len(tickers) != 0:
        bot_down_return.append(return_df.loc[i, tickers].mean())
    else:
        bot_down_return.append(0)

return_df['top_up'] = [0] + top_up_return
return_df['top_down'] = [0] + top_down_return
return_df['bot_up'] = [0] + bot_up_return
return_df['bot_down'] = [0] + bot_down_return

plt.figure()
(return_df['top_up'] + 1).cumprod().plot(figsize=(16, 8), color='k')
(return_df['top_down'] + 1).cumprod().plot(color='g')
(return_df['bot_up'] + 1).cumprod().plot(color='r')
(return_df['bot_down'] + 1).cumprod().plot(color='b')
plt.legend(loc=0)
plt.show()

hs_300 = pd.read_sql('''select * from yq_index_month where symbol = "%s" 
                          and endDate>="%s" and endDate<="%s"''' % ('000300', begin, end), engine)
hs_300.index = hs_300['endDate'].values

temp_t0 = back_date  # datetime.strptime(back_date,'%Y-%m-%d').date()
hs_300_index = hs_300[hs_300.index > temp_t0].sort_index()['closePrice']
hs_300_index = hs_300_index / hs_300_index[0]

top_up_index = (return_df['top_up'] + 1).cumprod()
top_down_index = (return_df['top_down'] + 1).cumprod()
bot_up_index = (return_df['bot_up'] + 1).cumprod()
bot_down_index = (return_df['bot_down'] + 1).cumprod()

v1 = hs_300_index.values
v2 = top_up_index.values

for i in [top_up_index, top_down_index, bot_up_index, bot_down_index]:
    v1 = hs_300_index.values
    v2 = i.values
    a, b, c, d = result(v1=v1, v2=v2)
    print('收益 %s, 信息比率%s, 对冲收益 %s, 对冲信息比率 %s' % (a, b, c, d))

tt = datetime.datetime.now()
print('time used %s' % ((tt - t0).total_seconds()))

return_df[['top_up', 'top_down', 'bot_up', 'bot_down']].to_csv('re_%s.csv' % flag_m)