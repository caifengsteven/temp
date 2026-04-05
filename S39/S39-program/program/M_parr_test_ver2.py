# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:55:32 2020

@author: adair2019
"""
import dill
import sys,traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statsmodels.api as sm
import math
from sqlalchemy import create_engine
import json

#并行需要
from gplearn.utils import _partition_estimators
from joblib import Parallel, delayed
import itertools


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

def _parall_fun1(ticker_pool,begin='20100510',end='20200401'):
    fac_list = []
    #df_mon_list = []
    for ticker in ticker_pool:
        df = chs_factor(ticker= ticker, begin = begin, end= end)
        #alpha, df_mon = get_spring_factor(df, ticker= ticker)
        #df_mon = get_return(df_mon)
        #fac_list.append(alpha)
        fac_list.append(df)
        #df_mon_list.append(df_mon)
    return fac_list
#alpha系数的大小进行选股策略
rolling_day = 20
## 选择root 回归
begin = '20100501'
end = '20200401'
df = chs_factor(ticker= '000010', begin = begin, end= end)

## 获取300股票代码
hs_300 = get_IdxCons(end,ticker='000300')
## 回测过程
if __name__ == '__main__':
    #results = [pool.apply_async(chs_factor, args=(ticker,)) for ticker in hs_300[0:8]]
    #results = [p.get() for p in results]
    try:
        n_jobs, _, starts = _partition_estimators(len(hs_300),8)
        all_t = Parallel(n_jobs=n_jobs,verbose=False)(
                                  delayed(_parall_fun1)(hs_300[starts[i]:starts[i + 1]])
        for i in range(n_jobs))
        
        y = list(itertools.chain.from_iterable(all_t))
    except Exception:
        traceback.print_exc(file=sys.stdout)