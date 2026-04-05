# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:58:09 2019
每日更新程序
最终的就是设定文件名称
@author: adair
"""

#import zipfile
#import os
from sqlalchemy import create_engine
import pandas as pd
import json
#must be set 用户名，密码，端口
with open('para.json','r',encoding='utf-8') as f:
    para = json.load(f)
    
user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name1 = 'yuqerdata'
#eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name1)
engine = create_engine(eng_str)

tn = 'bond_impliedvol_wind_update'

data_fn = '隐含波动率1(1).xlsx'
x = pd.read_excel(data_fn,header=3)

sql_str1 = 'select tradingdate from %s order by tradingdate desc limit 1' % tn
t0 = pd.read_sql(sql_str1,engine)
if t0.empty:
    t0_str = '1900-01-01'
else:
    t0_str = str(t0.tradingdate[0])
#x = pd.read_excel('波动率数据补充.xls',header=3)
col_name = x.columns.values
m,n = x.shape
t_name = col_name[0]
x=x[x[t_name]>t0_str]
if not x.empty:
    col_name = list(col_name[1:])
    i = 0
    y = pd.DataFrame()
    for sub_str in col_name:
        symbol_str,type_str = sub_str.split('.')
        sub_x = x[[t_name,sub_str]].copy()
        sub_x.dropna(inplace=True)
        if not sub_x.empty:
            sub_x['symbol'] = symbol_str
            sub_x['symboltype']=type_str
            sub_x.rename(columns={sub_str:'f_val',t_name:'tradingdate'},inplace=True)
            y = y.append(sub_x)
        i +=1
        print(i)
    
    y.to_sql(tn,engine,if_exists='append',index=False,chunksize=3000)
    print('wind 隐含波动率数据更新至%s' % y.tradingdate.max())
else:
    print('wind隐含波动率数据已经是最新 %s' % t0_str)