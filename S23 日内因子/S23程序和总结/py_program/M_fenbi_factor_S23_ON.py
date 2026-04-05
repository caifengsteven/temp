# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:35:29 2020
Stk_Tick_202001.rar
rar = rarfile.RarFile('Stk_Tick_202001.rar',pwd='www.jinshuyuan.net')
#x.namelist()[0]
data = rar.open(rar.namelist()[0])

升级，使得程序可以读取金数源数据
升级，可以直接读取压缩包数据
升级，包含B股数据，需要剔除
@author: adair2019


"""
import os
#import time
#import sys
#import rarfile
import pandas as pd
import datetime
#from sqlalchemy import create_engine
import zipfile
import zlib


from yq_toolsS45 import eg_23 as engine

table_name='fenbifactor1'

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
        if len(buy_rate)==2:
            buy_rate = buy_rate.iloc[0,0]/(buy_rate.iloc[1,0]+buy_rate.iloc[1,0])
        elif 'B' in buy_rate.index:
            buy_rate = 1
        else:
            buy_rate = 0
        #spread_date adjust
        spread_date_adj = x.loc[:,['volume','spread_tick']].where(x.volume<x.volume.median()).mean().spread_tick
        #out_re
        data_day = pd.DataFrame({'tradingdate':tradingdate,'symbol':symbol,
                                 'spread_date':spread_date,'buy_rate':buy_rate,
                                'spread_date_adj':spread_date_adj},index=[0])
        table_name='fenbifactor1'
        data_day.to_sql(table_name,engine,if_exists='append',index=False)
        return data_day,tradingdate,symbol
    except Exception:
        print('%s-%s' %(tradingdate,symbol))
        return [],tradingdate,symbol

if __name__ == '__main__':  
    file_dir = r'k:\datasets\YCZ_Update\fenbi-data-push'
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
        print('S23分笔无可更新数据')
    for i,sub_ind in enumerate(ind):
        fn = fns[sub_ind]
        sub_path = fns_t[sub_ind]
        sub_path = datetime.datetime.strftime(sub_path,'%Y-%m-%d')
        sub_path = os.path.join(file_dir,sub_path)
        print(sub_path)
        z = zipfile.ZipFile(fn, "r")
        for zinfo in z.infolist():
            if zinfo.compress_type == 9:
                print('type 9')
                continue
        names_all = z.namelist()
        T2 = len(names_all)
        for j,sub_fn in enumerate(names_all):
            z.extract(sub_fn,sub_path)
            temp_fn=os.path.join(sub_path,sub_fn)
            if temp_fn.endswith('.csv'):
                _,t,s=calculate_fenbi_data(temp_fn)
            print('complete %d-%d %d-%d %s' % (j,T2,i,T1,sub_fn))
        z.close()
import os
os.system("pause")