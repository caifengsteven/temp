# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:55:00 2020

@author: adair-9960
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statsmodels.api as sm
import math
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

"""
#后复权 前复权一直在变，无法
sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''    
##前复权需要换为后复权    
def chs_factor(ticker = '000005',begin = begin ,end = end , 
               field = [u'symbol',  u'tradeDate', u'openPrice',
                        u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol',
                        u'turnoverValue',u'dealAmount', u'chgPct',
                        'turnoverRate',u'marketValue']):
    sql_str1 = sql_str_select_data1 % (','.join(field),ticker,begin,end)
    dataday = pd.read_sql(sql_str1,engine)
    field2 = [u'tradeDate',u'accumAdjFactor']
    sql_str2 = sql_str_select_data2 % (','.join(field2),ticker,begin,end)
    dataday2 = pd.read_sql(sql_str2,engine)
    dataday=pd.merge(dataday,dataday2,on=['tradeDate','tradeDate'],how='inner')
    dataday = dataday.applymap(lambda x: np.nan if x == 0 else x)
    ## 对数据补全
    return dataday.fillna(method = 'ffill')
"""
#获取基本数据
sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''    
##前复权需要换为后复权    
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
#获取行业分类数据
def get_industry_class(t):
    sql_str1 = '''select ticker,industryID1 from yuqerdata.yq_industry where 
                industryVersionCD="010303" and intodate <= "%s" and 
                (outDate>"%s" or outDate is null)''' % (t,t)
    x = pd.read_sql(sql_str1,engine)
    return x
## 数据的起始与终止时间
begin = '2017-05-01'
end = '2017-06-01'
rolling_day = 5
## 股票池的信息日期
info_date = '2015-07-01'
## 回测起始时间
back_date = '2008-06-01'
## 选择回测区间，选择A或者hs300
flag_m = 'A'
# flag_m = 'hs_300'
##得到月度日历
def get_month_calender():
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" order by endDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

month_date = get_month_calender()
##获取成分股
def get_IdxCons(intoDate,ticker='000300'):
    #nearst 时间
    sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
            and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
        ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' %(ticker,
        ticker,intoDate)
    x = pd.read_sql(sql_str1,engine)
    x = x['symbol'].values   
    return x
    
hs_300=get_IdxCons(intoDate=end)
hs_300.sort()
def get_reg_param(df, rolling_day = rolling_day, is_disp = True):
    df.index = df['tradeDate'].values
    ## 计算B_k
    ## 复权
    df_bk = (df[['openPrice', 'highestPrice', 'lowestPrice', 'closePrice']].T * df['accumAdjFactor']).T
    ## 研报写的sum，此处感觉应该用mean
    df_bk = df_bk.rolling(rolling_day).mean().mean(axis = 1)
    vr = df_bk / df_bk.shift(1) - 1
    sp = (df['turnoverValue'] - df['turnoverVol'] * df_bk) / ( (df['turnoverVol'] / df['turnoverRate'])  * df_bk.shift(1) )
    ## 回归过程
    x = sp.dropna()
    y = vr.dropna()
    result = sm.OLS(y, x).fit()
    table = result.summary2().tables
    if is_disp:
        print(result.summary())
        #plt.figure()
        #plt.plot(x, y, '.')
    return [result.rsquared, result.pvalues.values[0] , result.params.values[0], float(table[2].loc[0, 3])]

## 数据的起始与终止时间
begin = '2017-05-01'
end = '2017-07-01'
rolling_day = 10
df = chs_factor(ticker= '000001', begin = begin, end= end)
get_reg_param(df, rolling_day= rolling_day, is_disp= True)

result_list = []
for i in hs_300:
    try:
        res = get_reg_param(chs_factor(i,begin = begin, end = end), rolling_day= rolling_day, is_disp= False)
        result_list.append(res)
    except:
        print(i, 'got error data')
        continue

#nan值处理，前复权和后复权差异
pd.DataFrame(result_list, columns= ['R2', 'pvalue', 'coef', 'dw']).dropna().mean()
print(pd.DataFrame(result_list, columns= ['R2', 'pvalue', 'coef', 'dw']).dropna().mean())

def get_root_reg_param(df, rolling_day = rolling_day, is_disp = True):
    df.index = df['tradeDate'].values
    ## 计算B_k
    ## 复权
    df_bk = (df[['openPrice', 'highestPrice', 'lowestPrice', 'closePrice']].T * df['accumAdjFactor']).T
    ## 研报写的sum，此处感觉应该用mean
    df_bk = df_bk.rolling(rolling_day).mean().mean(axis = 1)
    vr = df_bk / df_bk.shift(1) - 1
    sp = (df['turnoverValue'] - df['turnoverVol'] * df_bk) / ( (df['turnoverVol'] / df['turnoverRate'])  * df_bk.shift(1) )
    ## 回归过程
    x = sp.dropna()
    x = x.apply(lambda x: np.sign(x)*np.sqrt(np.abs(x)))
    y = vr.dropna()
    result = sm.OLS(y, x).fit()
    table = result.summary2().tables
    if is_disp:
        print(result.summary())
        #plt.figure()
        #plt.plot(x, y, '.')
    return [result.rsquared, result.pvalues.values[0] , result.params.values[0], float(table[2].loc[0, 3])]


## 数据的起始与终止时间
begin = '2017-05-01'
end = '2017-07-01'
df = chs_factor(ticker= '000001', begin = begin, end= end)
get_root_reg_param(df, is_disp= True)

result_list = []
for i in hs_300:
    try:
        res = get_root_reg_param(chs_factor(i, begin = begin, end= end),is_disp= False)
        result_list.append(res)
    except:
        print(i, 'got error data')
        continue
    
pd.DataFrame(result_list, columns= ['R2', 'pvalue', 'coef', 'dw']).dropna().mean()
print(pd.DataFrame(result_list, columns= ['R2', 'pvalue', 'coef', 'dw']).dropna().mean())

rolling_day = 20
## 选择root 回归
begin = '20100501'
end = '20200401'
df = chs_factor(ticker= '000010', begin = begin, end= end)

#alpha系数的大小进行选股策略
rolling_day = 20
## 选择root 回归
begin = '20100501'
end = '20200401'
df = chs_factor(ticker= '000010', begin = begin, end= end)

#获取因子值，最终得到每个月底的alpha和每个月月底的收益，二者一一对应
def get_spring_factor(df, ticker = '000001'):
    df.index = df['tradeDate'].values
    ## 计算B_k
    ## 复权
    df_bk = (df[['openPrice', 'highestPrice', 'lowestPrice', 'closePrice']].T * df['accumAdjFactor']).T
    ## 计算vr，sp
    df_bk = df_bk.rolling(rolling_day).mean().mean(axis = 1)
    vr = df_bk / df_bk.shift(1) - 1
    sp = (df['turnoverValue'] - df['turnoverVol'] * df_bk) / ( (df['turnoverVol'] / df['turnoverRate'])  * df_bk.shift(1) )
    
    coef_list = []
    ## 按月回归得到塑性系数
    for i,j in zip(month_date[:-1], month_date[1:]):
        y = vr.loc[i:j][1:].dropna()
        x = sp.loc[i:j][1:].dropna()
        x = x.apply(lambda x: np.sign(x)*np.sqrt(np.abs(x)))
        try:
            coef = sm.OLS(y, x).fit().params.values[0]
            coef_list.append(coef)
        except:
            coef = np.nan
            coef_list.append(coef)
            continue
    ## 塑性系数序列
    coef_sr = pd.Series(coef_list, index= month_date[1:], name = ticker).dropna()
    ## 取N = 12进行指数加权
    coef_t = coef_sr.ewm(span= 12).mean().dropna()
    ## 排除市值因子
    ## 数值补全，保证ols计算
    x_m = df.loc[coef_t.index]['marketValue'].fillna(method = 'bfill').fillna(method = 'ffill')
    alpha = sm.OLS(coef_t, x_m).fit().resid
    alpha.name = ticker
    return alpha, df.loc[coef_sr.index]

## 计算收益
def get_return(df1):
    name = df1['ticker'][-1]
    df1 = df1['closePrice'] * df1['accumAdjFactor']
    df_return = (df1 / df1.shift(1) - 1).shift(-1)
    df_return.name = name
    return df_return.dropna()

## 获取300股票代码
hs_300 = get_IdxCons(end,ticker='000300')
## 回测过程
fac_list = []
df_mon_list = []
for i in hs_300:
    ticker = i
    df = chs_factor(ticker= ticker, begin = begin, end= end)
    alpha, df_mon = get_spring_factor(df, ticker= ticker)
    df_mon = get_return(df_mon)
    fac_list.append(alpha)
    df_mon_list.append(df_mon)
    
## 原文使用500和分类进行回测，此处选300的分类前2个股票回测 因子和因子收益
fac_df = pd.concat(fac_list, axis = 1)
ret_df = pd.concat(df_mon_list, axis = 1)


ticker_class = get_industry_class(end)
ticker_class =pd.merge(ticker_class,pd.DataFrame({'ticker':hs_300}),on=['ticker','ticker'],how='inner')
ticker_class = ticker_class[['ticker','industryID1']].drop_duplicates()
ticker_group = ticker_class.groupby('industryID1')['ticker'].apply(lambda x: x.values)

#根据截面上每个行业分类的因子大小，选择分类前2个股票
def get_fac_ticker(c):
    fac_ticker = []
    fac_gc= fac_df.loc[:, c].apply(lambda x: x.sort_values(ascending= False)[:2], axis = 1)
    for i in fac_gc.applymap(lambda x: False if np.isnan(x) else True).values:
        fac_ticker.append(list(fac_gc.columns[i]))
    return pd.DataFrame(fac_ticker)

#获取每个月底的所有选择的股票序列
tickers_df = pd.concat([get_fac_ticker(ticker_group.iloc[i]) for i in range(ticker_group.shape[0])],axis = 1)
tickers_df.index = fac_df.index
ret_df = ret_df.loc[tickers_df.index].fillna(0)

#这里感觉应该顺延一个月，因为当月月底选出的股票，应该使用下一个月的收益来计算 !!!!!!!!!!!!!!
ret_list = []
for i in tickers_df.index:
    tickers = tickers_df.loc[i].dropna().values
    rets = ret_df.loc[i, tickers].mean()
    ret_list.append(rets)
ret_sr = pd.Series(ret_list, index= tickers_df.index).shift(1)

#hs_300_data = DataAPI.MktIdxmGet(beginDate= begin , endDate= end, indexID=u"000300.ZICN", ticker=u"",field=u"",pandas="1")
hs_300_data = pd.read_sql('''select * from yq_index_month where indexID = "000300.ZICN" 
                          and endDate>="%s" and endDate<="%s"''' %(begin,end),engine)
hs_300_data.index = hs_300_data['endDate'].values
hs_300_index = hs_300_data.loc[ret_sr.index].sort_index()['closePrice']
hs_300_index = hs_300_index / hs_300_index[0]

plt.figure()
(ret_sr.fillna(0) + 1).cumprod().plot()
hs_300_index.plot()
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

a, b, c, d = result(hs_300_index.values, (ret_sr.fillna(0) + 1).cumprod().values)
print('收益 %s, 信息比率%s, 对冲收益 %s, 对冲信息比率 %s'%(a, b , c, d))

outperformance = (ret_sr.fillna(0) + 1).cumprod().values - hs_300_index.values
plt.figure()
plt.plot(outperformance)