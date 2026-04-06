# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:54:12 2020

@author: adair-9960
"""
import os
import pandas as pd
from yq_toolsS45 import  get_file_name
#from yq_toolsS45 import create_table
from yq_toolsS45 import create_db
from yq_toolsS45 import save_pickle
import multiprocessing
num_core = multiprocessing.cpu_count()
var_info = ['indexID','ticker', 'tradeDate', 'openPrice', 'closePrice', 'highestPrice', 'lowestPrice',
       'Volume']

var_type = []
for i in var_info:
    if i in ['ticker']:
        var_type.append('varchar(20)')
    elif i in ['indexID']:
        var_type.append('varchar(20)')
    elif i in ['tradeDate']:
        var_type.append('datetime')
    else:
        var_type.append('float')

db_pro = 'data_pro'
eg_pro = create_db(db_pro)


def get_sub_data(inputData):
    sub_ticker,x,sub_index = inputData
    sub_x=x[(x[x.columns[0]]==sub_ticker) | (x[x.columns[0]]=='a')].T.iloc[1:]
    sub_x.columns=sub_x.iloc[0].tolist()
    sub_x=sub_x[1:]
    x.dropna(how='all',inplace=True)
    if sub_index.upper() in ['HSI','MSCI','NKY']:
        sub_x['ticker'] = '%0.5d' % int(sub_ticker.split(' ')[0])
    elif sub_index.upper() in ['topix'.upper()]:
        sub_x['ticker'] = '%0.4d' % int(sub_ticker.split(' ')[0])
    else:
        sub_x['ticker'] = sub_ticker.split(' ')[0]
    return sub_x

if __name__ == '__main__':
    """
    for sub_ticker in ticker[1:]:
        sub_x = get_sub_data(sub_ticker)
        #sub_x['index'] =sub_index
        X.append(sub_x)
        print(sub_ticker)
    X=pd.concat(X)
    """
    pn= r'D:\worksPool\works2020\adair2020_W\datasets\S61主要指数前复权日线'
    _,fn = get_file_name(pn,'.xlsx')
    for sub_fn in fn:
        sub_fn = os.path.join(pn,sub_fn)   
        sub_index= os.path.split(sub_fn)[1].split(' ')[0]
        print('begin %s' % sub_index)
        tn='main_index_S61'
        #tn = 'S54minInd_%s' % sub_index
        #create_table(db_pro,tn,var_info,var_type,[])    
        #
        x=pd.read_excel(sub_fn)
        x.dropna(how='all',inplace=True)
        x.dropna(how='all',axis='columns',inplace=True)
        x=x.iloc[2:]
        x.iloc[0].fillna(method='ffill',inplace=True)
        x=x.T
        x.drop(columns=x.columns[1],inplace=True)
        #x=x[x[x.columns[1]].isin(['Dates','Open'])]
        #X=[]
        x.iloc[0,0]='a'
        ticker=x[x.columns[0]].unique().tolist()
        
        X_pool = [x] * len(ticker[1:])
        p2 = [sub_index]*len(X_pool)
        pool = multiprocessing.Pool(num_core) 
        X=pool.map(get_sub_data, zip(ticker[1:],X_pool,p2))
        pool.close()
        pool.join()
        X=pd.concat(X)
        X.rename(columns={'Dates':'tradeDate','PX_OPEN':'openPrice','PX_LAST':'closePrice',
                          'PX_HIGH':'highPrice','PX_LOW':'lowPrice','PX_VOLUME':'Volume'},inplace=True)
        #X.rename(columns={'Dates':'tradeDate','Open':'openPrice','Close':'closePrice','High':'highestPrice','Low':'lowestPrice','index':'indexID'},inplace=True)
        X.dropna(subset=['closePrice'],inplace=True)
        X['index_id'] = sub_index
        X.to_sql(tn,eg_pro,if_exists='append',index=False,chunksize=3000) 
        
    
    
    
        
        
    
    