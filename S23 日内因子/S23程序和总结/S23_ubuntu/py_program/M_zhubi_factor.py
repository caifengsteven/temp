# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:15:55 2019
group by
@author: adair2019
"""

import pandas as pd
import numpy as np
import os,time,sys
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:liudehua@localhost:3306/S23?charset=utf8')

def get_file_name(file_dir,file_type):
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(root, file))             
    return L

def get_fenbi_data(fn):
    
    _,info=os.path.split(fn)
    sub_path = fn.split('/')  #muse check 
    if len(sub_path)>3:
        if sub_path[-3] == '2018':
            file_type=2
        else:
            file_type = 1
    else:
        file_type=1
            
    if file_type==1:
        x = pd.read_csv(fn,header=0,encoding='GBK',engine='python')        
        info = info.split(' ')
        tradingdate = info[0]
        symbol = info[1][2:-4]
    else:
        x = pd.read_csv(fn,header=None,names=['Time','Price','BuySell','Volume'],encoding='GBK',engine='python')
        symbol = info[2:][0:-4]
        tradingdate= sub_path[-2]
        tradingdate='%s-%s-%s' % (tradingdate[0:4],tradingdate[4:6],tradingdate[6:])
    return x,file_type,tradingdate,symbol

def cal_fenbifactordata(x,file_type):
    x.eval('cash = Volume*Price' , inplace=True)
    stdx = x.cash.std()
    mx = x.cash.mean()
    
    cut1 = mx + stdx
    cut2 = mx + 3*stdx
    v_all = x.cash.sum()
    
    value1 = x.where(x.cash>cut1).groupby(by='BuySell',axis=0).sum().cash
    value2 = x.where(x.cash>cut2).groupby(by='BuySell',axis=0).sum().cash
    value3 = x.groupby(by='BuySell',axis=0).sum().cash
    
    bigB1=0
    bigB2=0
    focusB=0
    bigS1=0
    bigS2=0
    focusS=0
    if value1.index.contains('B'):
        bigB1 = value1.loc['B']/v_all    
    if value1.index.contains('S'):
        bigS1 = value1.loc['S']/v_all
    
    if value2.index.contains('B'):    
        bigB2 = value2.loc['B']/v_all    
    if value2.index.contains('S'):
        bigS2 = value2.loc['S']/v_all
    
    if value3.index.contains('B'): 
        focusB = value3.loc['B']/v_all
    if value3.index.contains('S'):
        focusS = value3.loc['S']/v_all
    
    result1 = pd.DataFrame({'bigB1':[bigB1],'bigS1':[bigS1],'bigB2':[bigB2],'bigS2':[bigS2],
                            'focusB':[focusB],'focusS':[focusS]})    
    
    if file_type==2:
        x.where((x.Time>=93000) & (x.Volume>=100),inplace=True)
    else:
        x.where((x.Time>='09:30:00') & (x.Volume>=100),inplace=True)
    x.dropna(axis=0,inplace=True)
        
    sub_y = np.arange(1,8) / 8
    
    sub_x = x.loc[:,['BuySell','Volume']].groupby(by='BuySell',axis=0).quantile(sub_y).reset_index()    
    
    BS_type =['B','S']
    
    re = np.empty(0)
    re_columns = []
    for BS in BS_type:
        
        sub_sub_x0 = sub_x.where(sub_x.BuySell==BS).dropna(axis=0).Volume
        sub_sub_x = np.log10(sub_sub_x0)
        if not sub_sub_x.empty:
            sub_sub_y = np.log10(1-sub_y)
            f1 = np.polyfit(sub_sub_x,sub_sub_y, 1)
            
            beta = 1-f1[0]
            r = np.square(np.corrcoef(sub_sub_x,sub_sub_y)[0,1])
            
            sub_sub_y_adj = 1-sub_y
            f2 = np.polyfit(sub_sub_x,sub_sub_y_adj, 1)
            
            beta_adj = 1-f2[0]
            r_adj = np.square(np.corrcoef(sub_sub_x,sub_sub_y_adj)[0,1])
            v_all = sub_sub_x0.values
        else:
            r=np.nan
            beta=np.nan
            r_adj=np.nan
            beta_adj=np.nan
            sub_sub_x0=np.nan
            v_all = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
        
        re = np.hstack((re,r,beta,r_adj,beta_adj,v_all))
        re_columns.extend(['%sr' % BS,'%sbeta' % BS, '%sr_adj' % BS,'%sbeta_adj' % BS,
                           '%sV1' %BS, '%sV2' %BS,'%sV3' %BS,'%sV4' %BS,
                           '%sV5' %BS,'%sV6' %BS,'%sV7' %BS])
    
    re.reshape([1,22])    
    result2 = pd.DataFrame(re,index=re_columns).T    
    result = result1.join(result2)
    return result

def zhubi_flow(fn):
    x,file_type,tradingdate,symbol=get_fenbi_data(fn)
    try:
        result = cal_fenbifactordata(x,file_type)
        result['symbol'] = symbol
        result['tradingdate'] = tradingdate
        table_name='zhubifactor_basic'
        result.to_sql(table_name,engine,if_exists='append',index=False)

        return result,tradingdate,symbol
    except Exception:
        print(fn)
        return [],tradingdate,symbol


if __name__ == '__main__':
    t0 = time.time()
    print(sys.argv)
    if len(sys.argv) == 2:
        id = int(sys.argv[1])
    else:
        id =0
    print('%s' % id)
    
    file_dir = '/home/adair/workspool/YCZ_zhubi/'
    fns = get_file_name(file_dir,'.csv')
    
    T = len(fns)
    cut = 6
    while id < T:
        _,tradingdate,symbol=zhubi_flow(fns[id])
        print('Complete : %s-%s %d-%d' % (tradingdate,symbol,id,T))
        id= id+cut