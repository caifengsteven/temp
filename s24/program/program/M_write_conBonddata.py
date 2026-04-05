# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:50:59 2019

@author: adair2019
"""

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:liudehua@localhost:3306/yuqerdata?charset=utf8')
table_name = 'ConvertibleBond_dayprice'

fn1 = r'D:\worksPool\works2019\SOME\项目\S24\Data\consBonddata2019.csv'
fn2 = r'D:\worksPool\works2019\SOME\项目\S24\Data\consBonddata2006_2018.csv'

def add_0(x):
    x= '%0.6d' % x
    return x

x1 = pd.read_csv(fn1,header=0,engine='python',encoding = "utf-8",index_col=0)
x2 = pd.read_csv(fn2,header=0,engine='python',encoding = "utf-8",index_col=0)

x1['tickerEqu'] = x1['tickerEqu'].apply(add_0)
x2['tickerEqu'] = x2['tickerEqu'].apply(add_0)
x1.to_sql(table_name,engine,if_exists='append',index=False,chunksize=3000)
x2.to_sql(table_name,engine,if_exists='append',index=False,chunksize=3000)