# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:45:10 2020

@author: adair2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import json
from datetime import date,datetime
import pymysql
import warnings
import sys

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

db_name42 = 'S42'
#eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name42)
engine42 = create_engine(eng_str)

db_name_us = 'us_stock'
#eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name_us)
engine_us = create_engine(eng_str)

sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''

#创建表格并分区
def create_table_update(db_name,tn_name,var_name,var_type,key_str,p_num=1):
    #连接本地数据库
    db = pymysql.connect("localhost",user_name,pass_wd,db_name)
    #创建游标
    cursor = db.cursor()
    #创建
    var_info=''
    for id,sub_var in enumerate(var_name):
        var_info=var_info + sub_var + ' ' + var_type[id] + ','
    var_info = var_info[:-1]    
    if len(key_str)>0:
        sql = 'create table  `%s`(%s,primary key(%s)) partition by key(ticker) partitions %d' % (tn_name,var_info,key_str,p_num)    
    else:
        sql = 'create table  `%s`(%s)' % (tn_name,var_info)  

    try:
        # 执行SQL语句
        cursor.execute(sql)
        print("创建数据库成功")
    except Exception as e:
        print("创建数据库失败：case%s"%e)
    finally:
        #关闭游标连接
        cursor.close()
        # 关闭数据库连接
        db.close()
        
        
def create_table(db_name,tn_name,var_name,var_type,key_str):
    #连接本地数据库
    db = pymysql.connect("localhost",user_name,pass_wd,db_name)
    #创建游标
    cursor = db.cursor()
    #创建
    var_info=''
    for id,sub_var in enumerate(var_name):
        var_info=var_info + sub_var + ' ' + var_type[id] + ','
    var_info = var_info[:-1]    
    if len(key_str)>0:
        sql = 'create table  `%s`(%s,primary key(%s))' % (tn_name,var_info,key_str)    
    else:
        sql = 'create table  `%s`(%s)' % (tn_name,var_info)  

    try:
        # 执行SQL语句
        cursor.execute(sql)
        print("创建数据库成功")
    except Exception as e:
        print("创建数据库失败：case%s"%e)
    finally:
        #关闭游标连接
        cursor.close()
        # 关闭数据库连接
        db.close()
        
## 数据的起始与终止时间
def get_index_tradeDate(index,begin,end):
    sql_str_index = '''select * from yq_index where symbol = "%s" and tradeDate>="%s" and tradeDate<="%s" order by tradeDate'''
    sql_str_index = sql_str_index % (index,begin,end)
    hs300_index = pd.read_sql(sql_str_index,engine)
    hs300_index = hs300_index.sort_values('tradeDate')
    return hs300_index
def get_cf_future_tradeDate(index,begin='2000-01-01',end='2033-01-01'):
    sql_str_index = '''select contractObject as symbol,tradeDate,openPrice as openIndex,highestPrice as highestIndex,
    lowestPrice as lowestIndex,closePrice  as closeIndex,turnoverVol,chgPct from yq_MktMFutdGet 
    where contractObject = "%s" and mainCon = 1 and tradeDate>="%s" and 
    tradeDate<="%s" order by tradeDate'''
    sql_str_index = sql_str_index % (index,begin,end)
    hs300_index = pd.read_sql(sql_str_index,engine)
    hs300_index = hs300_index.sort_values('tradeDate')
    return hs300_index

def get_a_stock_tradeDate(index,begin='2000-01-01',end='2033-01-01'):
    #每日数据
    sql_str_index = '''select symbol,tradeDate,openPrice as openIndex,highestPrice as highestIndex,
    lowestPrice as lowestIndex,closePrice  as closeIndex from yq_dayprice
    where symbol = "%s"  and tradeDate>="%s" and 
    tradeDate<="%s" order by tradeDate'''
    sql_str_index = sql_str_index % (index,begin,end)
    hs300_index = pd.read_sql(sql_str_index,engine)
    hs300_index = hs300_index.sort_values('tradeDate')
    
    #后复权系数 MktEqudAdjAfGet
    sql_str_fq = """select tradeDate,accumAdjFactor from MktEqudAdjAfGet where 
    ticker = "%s" order by tradeDate"""
    sql_str_fq = sql_str_fq % index
    y = pd.read_sql(sql_str_fq,engine)
    y=pd.merge(hs300_index,y,on=['tradeDate'])
    y['openIndex'] = y['openIndex']*y['accumAdjFactor']
    y['highestIndex'] = y['highestIndex']*y['accumAdjFactor']
    y['lowestIndex'] = y['lowestIndex']*y['accumAdjFactor']
    y['closeIndex'] = y['closeIndex']*y['accumAdjFactor']
    return y

def get_exchange_tradeDate(index,begin='2000-01-01',end='2033-01-01'):
    #每日数据
    sql_str_index = '''select symbol,tradingdate as tradeDate,openPrice as openIndex,highestPrice as highestIndex,
    lowestPrice as lowestIndex,closePrice  as closeIndex,turnoverVol from exchange_dayly
    where symbol = "%s"  and tradingdate>="%s" and 
    tradingdate<="%s" order by tradingdate'''
    sql_str_index = sql_str_index % (index,begin,end)
    hs300_index = pd.read_sql(sql_str_index,engine42)
    hs300_index = hs300_index.sort_values('tradeDate')    
    return hs300_index
#dowjones data
def get_dowjones_tradeDate(index,begin='2000-01-01',end='2033-01-01'):
    #每日数据
    sql_str_index = '''select symbol,tradeDate,openPrice as openIndex,highestPrice as highestIndex,
    lowestPrice as lowestIndex,closePrice  as closeIndex, totalVolume as turnoverVol from dowjones_dayly
    where symbol = "%s"  and tradeDate>="%s" and 
    tradeDate<="%s" order by tradeDate'''
    sql_str_index = sql_str_index % (index,begin,end)
    hs300_index = pd.read_sql(sql_str_index,engine42)
    hs300_index = hs300_index.sort_values('tradeDate')    
    return hs300_index

#美股后复权数据
def get_american_stock_tradeDate(index,begin='2000-01-01',end='2033-01-01'):
    #每日数据
    sql_str_index = '''select symbol,tradingdate as tradeDate,openprice_adj as openIndex,highprice_adj as highestIndex,
    lowprice_adj as lowestIndex,closeprice_adj  as closeIndex,volume_adj as turnoverVol from us_stock_daytick
    where symbol = "%s"  and tradingdate>="%s" and 
    tradingdate<="%s" order by tradingdate'''
    sql_str_index = sql_str_index % (index,begin,end)
    hs300_index = pd.read_sql(sql_str_index,engine_us)
    return hs300_index
#获取成分股
def get_IdxCons(intoDate,ticker='000300'):
    #nearst 时间
    sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
            and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
        ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' %(ticker,
        ticker,intoDate)
    x = pd.read_sql(sql_str1,engine)
    x = x['symbol'].values   
    return x

#日线数据
def chg_factor(ticker = '000005',begin = '2001-01-01' ,end = '2090-01-01' , 
               field = [u'symbol',  u'tradeDate', u'openPrice',
                        u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol',
                        u'turnoverValue',u'dealAmount', u'chgPct',
                        'turnoverRate',u'marketValue']):
    sql_str1 = sql_str_select_data1 % (','.join(field),ticker,begin,end)
    dataday = pd.read_sql(sql_str1,engine)
    dataday = dataday.applymap(lambda x: np.nan if x == 0 else x)
    dataday.rename(columns={'symbol':'ticker'},inplace=True)
    #升级后复权系数
    #后复权系数 MktEqudAdjAfGet
    sql_str_fq = """select tradeDate,accumAdjFactor from MktEqudAdjAfGet where 
    ticker = "%s" order by tradeDate"""
    sql_str_fq = sql_str_fq % ticker
    y = pd.read_sql(sql_str_fq,engine)
    dataday=pd.merge(dataday,y,on=['tradeDate'])
    return dataday.fillna(method = 'ffill')


## 得交易日历
def get_calender_range(begin, end):
    sql_str = """select tradeDate from yuqerdata.yq_index where symbol = "000001" 
    and tradeDate >="%s" and tradeDate <="%s" order by tradeDate""" % (begin, end)
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

#获取所有交易日历
def get_calender():
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" order by tradeDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x
#获取月度日历    
def get_month_calender(begin = '2000-01-01'):
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" and endDate>="%s" order by endDate''' % (begin)
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x
#获取A股所有的ticker
def get_symbol_A():
    sql_str = """select distinct(ticker)  from equget
                where equTypeCD = "A" and listStatusCD !="UN" and 
                ListSectorCD<=3 and length(ticker)=6  order by ticker"""
    x = pd.read_sql(sql_str,engine)
    return x.ticker.tolist()
                