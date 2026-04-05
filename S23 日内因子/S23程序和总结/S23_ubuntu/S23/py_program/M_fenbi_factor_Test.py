#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:25:56 2019

@author: adair
"""

#read fenbi data
import os, time, sys
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:liudehua@localhost:3306/S23?charset=utf8')

def get_file_name(file_dir,file_type):
    L=[]   
    info=[]
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(root, file))
                temp=file.split(' ')
                info.append(temp[1])                
    return L, info

def calculate_fenbi_data(fn):
    _,info=os.path.split(fn)
    info = info.split(' ')
    tradingdate = info[0]
    symbol = info[1][2:]
    
    x = pd.read_csv(fn,header=1,encoding='GBK',engine='python');
    x.rename(columns={'时间':'tradingdate', '价格':'price','笔数':'dealAmount',
                      '成交额':'turnoverValue','成交量':'volume','买卖盘':'BSsel',
                      '买一价':'BP1','买二价':'BP2','买三价':'BP3','买四价':'BP4',
                      '买五价':'BP5','卖一价':'SP1','卖二价':'SP2','卖三价':'SP3',
                      '卖四价':'SP4','卖五价':'SP5','买一量':'BV1','买二量':'BV2',
                      '买三量':'BV3','买四量':'BV4','买五量':'BV5','卖一量':'SV1',
                      '卖二量':'SV2','卖三量':'SV3','卖四量':'SV4','卖五量':'SV5'}, inplace = True)
    x['symbol'] = symbol
    try:
        #暂时先不去
        #t1 = '%s 09:30:00' % tradingdate
        #t2 = '%s 15:00:00' % tradingdate
        #x.where((x.tradingdate>=t1) & (x.tradingdate<=t2),inplace=True)
        #x.dropna(axis=0,inplace=True)
        x.eval('BID = (BV1*BP1*10+BV2*BP2*9+BV3*BP3*8+BV4*BP4*7+BV5*BP5*6)/(10+9+8+7+6)' , inplace=True)
        x.eval('ASK = (SV1*SP1*10+SV2*SP2*9+SV3*SP3*8+SV4*SP4*7+SV5*SP5*6)/(10+9+8+7+6)' , inplace=True)
        x.eval('spread_tick = (BID-ASK)/(BID+ASK)' , inplace=True)
        #spread_date
        spread_date=x.spread_tick.mean()
        #BuyRate
        buy_rate = x.loc[:,['BSsel','volume']].groupby(by='BSsel',axis=0).mean()
        buy_rate = buy_rate.iloc[0,0]/(buy_rate.iloc[1,0]+buy_rate.iloc[1,0])
        #spread_date adjust
        spread_date_adj = x.loc[:,['volume','spread_tick']].where(x.volume<x.volume.median()).mean().spread_tick
        #out_re
        data_day = pd.DataFrame({'tradingdate':tradingdate,'symbol':symbol,
                                 'spread_date':spread_date,'buy_rate':buy_rate,
                                'spread_date_adj':spread_date_adj},index=[0])
        table_name='fenbifactor1'
        data_day.to_sql(table_name,engine,if_exists='append',index=False)
        return data_day,tradingdate,symbol
    except IndexError:
        print('%s-%s' %(tradingdate,symbol))
        return [],tradingdate,symbol


    
file_dir = '/home/adair/workspool/YCZ_fenbi/2011'
fns,info = get_file_name(file_dir,'.csv')
temp = pd.DataFrame({'a':fns,'b':info})
temp.sort_values(by='b',inplace=True)
fns=temp['a'].tolist()
    