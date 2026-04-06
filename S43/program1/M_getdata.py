# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:14:12 2020
DataAPI.MktBarHistOneDayGet(securityID=universe, date=dt, 
startTime='09:31', endTime='10:00', unit=30, field='ticker,totalVolume')

select * from yczdata.ycz_min_30 where date(tradingdate) = '2010-01-04' and 
time(tradingdate)>=time('9:31:00') and time(tradingdate)<=time('9:32:00') and symbol = 'sz000001'

field='ticker,barTime,totalVolume,totalValue'


@author: adair2019
"""
import pandas as pd

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

def MktBarHistOneDayGet(ticker,tradedate,startTime,endTime):
    if len(tradedate)==8:
        tradedate = '%s-%s-%s' % (tradedate[0:4],tradedate[4:6],tradedate[6:])
    if ticker[0]=='6':
        symbol1 = 'sh' + ticker
    else:
        symbol1 = 'sz' + ticker
    sql_str = """select symbol as ticker,tradingdate,volume,amount from yczdata.ycz_min_30 where symbol = "%s" and date(tradingdate)
             ="%s" and time(tradingdate)>= time("%s") and time(tradingdate)<=time("%s")
             order by tradingdate"""
    sql_str = sql_str % (symbol1,tradedate,startTime,endTime)
    x = pd.read_sql(sql_str,engine)
    if len(x)>0:
        x['ticker'] = ticker
        x.rename(columns={'tradingdate':'barTime','volume':'totalVolume','amount':'totalValue'},inplace=True)
    return x
    
x = MktBarHistOneDayGet(ticker='000001',tradedate='2010-01-04',startTime='9:31',endTime='10:00')
print(x)