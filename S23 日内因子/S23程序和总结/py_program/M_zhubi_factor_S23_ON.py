# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:15:55 2019
group by
@author: adair2019
"""

import pandas as pd
import numpy as np
import os
#import time
#import sys
import datetime
import zipfile
from yq_toolsS45 import eg_23 as engine
table_name='zhubifactor_basic'


def get_ycz_fenbi_file_tradingdate(info):
    ind = info.index('20')
    t = info[ind:-4]
    #t = datetime.datetime.strptime(t, '%Y%m%d').strftime('%Y-%m-%d')
    t=datetime.datetime.strptime(t, '%Y%m%d')
    t=datetime.datetime.date(t)
    return t
    
def get_file_name(file_dir,file_type):
    L=[]   
    info=[]
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(root, file))
                temp=file.split(' ')
                info.append(temp[0])                
    return L, info

def get_fenbi_data(fn):
    
    _,info=os.path.split(fn)
    sub_path = fn.split('\\')
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
    if 'B' in value1.index:
        bigB1 = value1.loc['B']/v_all    
    if 'S' in value1.index:
        bigS1 = value1.loc['S']/v_all
    
    if 'B' in value2.index:    
        bigB2 = value2.loc['B']/v_all    
    if 'S' in value2.index:
        bigS2 = value2.loc['S']/v_all
    
    if 'B' in value3.index: 
        focusB = value3.loc['B']/v_all
    if 'S' in value3.index:
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
        result.to_sql(table_name,engine,if_exists='append',index=False)

        return result,tradingdate,symbol
    except Exception:
        print(fn)
        return [],tradingdate,symbol

if __name__ == '__main__':
    file_dir = r'K:\datasets\YCZ_Update\zhubi-data-push'
    fns,info = get_file_name(file_dir,'.zip')
   
    tref_complete = pd.read_sql(('select distinct(tradingdate) from %s' % table_name),engine)
    ind = []
    fns_t = []
    for i,sub_info in enumerate(info):
        sub_t = get_ycz_fenbi_file_tradingdate(sub_info)
        fns_t.append(sub_t)
        if sub_t not in tref_complete.tradingdate.tolist():
            ind.append(i)
    
    T1 = len(ind)
    
    if T1==0:
        print('S23逐笔无可更新数据')
    for i,sub_ind in enumerate(ind):
        fn = fns[sub_ind]
        sub_path = fns_t[sub_ind]
        sub_path = datetime.datetime.strftime(sub_path,'%Y-%m-%d')
        sub_path = os.path.join(file_dir,sub_path)
        print(sub_path)
        print(fn)
        z = zipfile.ZipFile(fn, "r")    
        names_all = z.namelist()
        T2 = len(names_all)
        for j,sub_fn in enumerate(names_all):
            z.extract(sub_fn,sub_path)
            temp_fn=os.path.join(sub_path,sub_fn)
            if temp_fn.endswith('.csv'):
                _,t,s=zhubi_flow(temp_fn)
            print('complete %d-%d %d-%d %s' % (j,T2,i,T1,sub_fn))
        z.close()
import os
os.system("pause")