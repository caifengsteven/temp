# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:34:00 2021
支线02数据
数据转换
和以前的保持一致
@author: adair-9960
"""

import os
import pandas as pd
from yq_toolsS45 import time_use_tool
from yq_toolsS45 import create_db
from yq_toolsS45 import get_file_name
from sqlalchemy.types import NVARCHAR, Float, Integer,DATE
from tqdm import tqdm
from yq_toolsS45 import do_sql_order


def ticker_code_trans(sub_index,x):
    if 'Index'.lower() in x.lower():
        x = x.split(' ')[0]
    elif sub_index.upper() in ['HSI','HSI_NEW']:
        x = '%0.5d' % int(x.split(' ')[0])
    elif sub_index.lower() in ['topix','twse','hsce','msci','nky']:
        x = '%0.4d' % int(x.split(' ')[0])
    elif sub_index.lower() in ['kosdaq','kospi','xin9i']:
        x = '%0.6d' % int(x.split(' ')[0])
    else:
        x = x.split(' ')[0]
    return x

var_info = ['index_id','ticker', 'tradeDate', 'openPrice', 'closePrice', 'highPrice', 'lowPrice',
       'Volume']

dtypedict1 = dict(zip(var_info,[NVARCHAR(20),NVARCHAR(20),DATE,Float,Float,Float,Float,Float]))
eg_pro = create_db('data_pro')
#tn = 'main_index_zx02_v2'
tn = 'main_index_s68'

obj_t = time_use_tool()
pn = r'H:\bloomberg data'
_,fns = get_file_name(pn,'.xlsx')
fns.sort(reverse=True)
sql_t0 = 'select tradeDate from %s where index_id = "%s" order by tradeDate desc limit 1'

for sub_fn0 in tqdm(fns):
    sub_index_id = sub_fn0.split(' ')[0]
    print(sub_index_id)
    #do_sql_order('delete from %s where index_id ="%s"' % (tn,sub_index_id),'data_pro')
    #sub_fn0 = 'set50 index and component prices 小于50.xlsx'
    sub_fn = os.path.join(pn,sub_fn0)
    x =pd.read_excel(sub_fn)
    #obj_t.use('complete')
    x.dropna(how='all',inplace=True)
    x.dropna(axis=1,how='all',inplace=True)
    x.reset_index(inplace=True,drop=True)
    
    #x.drop(labels=[0,1],inplace=True)
    tmp_ind = max(x.index[x[x.columns[0]] =='End Date'])
    x.drop(labels=range(tmp_ind+1),inplace=True)
    
    b = x[x.columns[1]] == '#N/A Requesting Data...'
    b.name=  'c1'
    b = b.to_frame()
    b['c2'] = range(len(b))
    b = b[b.c1].c2.tolist()
    x.reset_index(inplace=True,drop=True)
    x.drop(labels=b,inplace=True)
    x = x.T
    x.iloc[0,0] = 'ticker'
    x.columns = x.iloc[0]
    x.reset_index(inplace=True,drop=True)
    x.drop(labels=0,inplace=True)
    x.ticker.fillna(method='ffill',inplace=True)
    
    x.set_index(keys=['Dates','ticker'],inplace=True,drop=True)
    x = x[x.columns[pd.notna(x.columns)]]
    tmp_t0 = pd.read_sql(sql_t0 % (tn,sub_index_id),eg_pro)
    if len(tmp_t0)>0:
        tmp_t0 = tmp_t0.tradeDate.astype(str).values[0]
        x =  x[x.columns[x.columns.astype(str)> '%s 13:00:00' % tmp_t0]]
        
    
    x = x.unstack().T
    x.reset_index(inplace=True)
    x.rename(columns={x.columns[0]:'tradeDate'},inplace=True)
    x.dropna(subset=x.columns[2:],inplace=True)
    
    #to mysql
    dic1 = dict(zip(['PX_HIGH', 'PX_LAST', 'PX_LOW', 'PX_OPEN',
           'PX_VOLUME'],['highPrice','closePrice','lowPrice','openPrice','Volume']))
    x.rename(columns=dic1,inplace=True)
    #x.ticker = x.ticker.apply(lambda x:x.split(' ')[0])
    
    x.ticker = x.ticker.apply(lambda x:ticker_code_trans(sub_index_id,x))
    x['index_id'] = sub_index_id
    tmp_t0 = pd.read_sql(sql_t0 % (tn,sub_index_id),eg_pro)
    if len(tmp_t0)>0:
        tmp_t0 = tmp_t0.tradeDate.astype(str).values[0]
        x=x[x.tradeDate>tmp_t0]
    if len(x)>0:
        if sub_index_id=='simsci':
            x= x[x.highPrice.apply(lambda x:not isinstance(x,str))]
        x.to_sql(tn,eg_pro,if_exists='append',index=False,chunksize=3000,dtype=dtypedict1) 
    obj_t.use('complete')