# -*- coding: utf-8 -*-
"""
Created on Fri May 29 08:59:45 2020
A stock data 写入一个csv
@author: adair-9960
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


sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
    
def chg_factor(ticker = '000005',begin = '2000-01-01' ,end = '2001-01-01' , 
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
    
    dataday['openPrice'] = dataday['openPrice']*y['accumAdjFactor']
    dataday['highestPrice'] = dataday['highestPrice']*y['accumAdjFactor']
    dataday['lowestPrice'] = dataday['lowestPrice']*y['accumAdjFactor']
    dataday['closePrice'] = dataday['closePrice']*y['accumAdjFactor']
    
    return dataday.fillna(method = 'ffill')

#x = chg_factor('000001')

sql_str = """select distinct(ticker)  from yuqerdata.equget where equTypeCD = 'A' 
    and listStatusCD !='UN' and ListSectorCD<=3 and ticker not like 'D'"""
symbol = pd.read_sql(sql_str,engine)
symbol = symbol.ticker.values

X = pd.DataFrame()
T = len(symbol)
for i,sub_symbol in enumerate(symbol):
    X = X.append(chg_factor(sub_symbol))
    print('%d-%d',(i,T))