# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:46:57 2020
update20200323
增加了S26需要的10因子数据
@author: adair002
"""

import os
from sqlalchemy import create_engine
import pandas as pd

#must be set 用户名，密码，端口
engine = create_engine('mysql+pymysql://root:liudehua@localhost:3306/yuqerdata?charset=utf8')
engineS26 = create_engine('mysql+pymysql://root:liudehua@localhost:3306/S26?charset=utf8')

def get_file_name(file_dir,file_type):
    L=[]
    L_s = []   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(root, file))  
                L_s.append(file)
    return L,L_s

def add_0(x):
    if isinstance(x,int):
        x= '%0.6d' % x
    return x

#must be set before using
pn = r"D:\worksPool\works2020\SOME\yuqerdata\data"

"""
fns,_ = get_file_name(pn,'.zip')
_,fns_csv = get_file_name(pn,'.csv')
for sub_fn in fns:
    f = zipfile.ZipFile(sub_fn,'r')
    for file in f.namelist():
        if file not in fns_csv:
            f.extract(file,pn)
"""        
_,fns_csv = get_file_name(pn,'.csv')

fns_data = [];
for i in range(30):
    fns_data.append([])

for sub_fn in fns_csv:
    if 'EquGet' in sub_fn:
        fns_data[0] = os.path.join(pn,sub_fn)
    elif 'indicator_data' in sub_fn:
        fns_data[1] = os.path.join(pn,sub_fn)
    elif 'tickerday_data' in sub_fn:
        fns_data[2] = os.path.join(pn,sub_fn)
    elif 'tradingdate' in sub_fn:
        fns_data[3] = os.path.join(pn,sub_fn)
    elif 'MktEqumAdjAfGet' in sub_fn:
        fns_data[4] = os.path.join(pn,sub_fn)
    elif 'EquIndustryGet' in sub_fn:
        fns_data[5] = os.path.join(pn,sub_fn)
    elif 'st_data' in sub_fn:
        fns_data[6] = os.path.join(pn,sub_fn)
    elif 'IdxCloseWeightGet' in sub_fn:
        fns_data[7].append(os.path.join(pn,sub_fn))
    elif 'yuqer_cal' in sub_fn:
        fns_data[8].append(os.path.join(pn,sub_fn))
    elif 'MktEqudAdjAfGet' in sub_fn:
        fns_data[9].append(os.path.join(pn,sub_fn))
    elif 'MktIdxmGet' in sub_fn:
        fns_data[10]= os.path.join(pn,sub_fn)
    elif 'MktStockFactorsOneDayGet' in sub_fn and 'MktStockFactorsOneDayGet_add' not in sub_fn:
        fns_data[11] = os.path.join(pn,sub_fn)
    elif 'MktStockFactorsOneDayGet_add' in sub_fn:
        fns_data[12] = os.path.join(pn,sub_fn)

data_id = 0
#1  股票基本数据已更新
if len(fns_data[data_id])>0:
    tn = 'equget'
    x = pd.read_csv(fns_data[data_id],dtype={'ticker':str},engine='python',encoding='utf-8')
    x = x[['ticker','exchangeCD','ListSectorCD','ListSector','secShortName',
           'listStatusCD','listDate','delistDate','equTypeCD','equType','partyID',
           'totalShares','nonrestFloatShares','nonrestfloatA','endDate','TShEquity']]
    x.to_sql(tn,engine,if_exists='replace',index=False,chunksize=3000)
    print('股票基本数据已更新')
    os.remove(fns_data[data_id])
    
#2 index
data_id = 1
if len(fns_data[data_id])>0:
    x = pd.read_csv(fns_data[data_id],dtype={'ticker':str},engine='python',encoding='utf-8')
    tn = 'yq_index'
    sql_str1 = 'select tradedate from %s order by tradedate desc limit 1' % tn
    t = pd.read_sql(sql_str1,engine)
    x2 = x.loc[x.tradeDate>str(t.tradedate[0])]
    if not x2.empty:
        x2.rename(columns={'ticker':'symbol'},inplace=True)
        x2.to_sql(tn,engine,if_exists='append',index=False,chunksize=3000)
        print('每日指数数据更新至%s' % x2.tradeDate.max())
        os.remove(fns_data[data_id])
    else:
        print('每日指数数据已经是最新的%s，无需更新' % str(t.tradedate[0]))            
#3 day price
data_id=2
if len(fns_data[data_id])>0:
    x = pd.read_csv(fns_data[data_id],dtype={'ticker':str},engine='python')
    tn1 = 'yq_dayprice'
    sql_str1 = 'select tradedate from %s order by tradedate desc limit 1' % tn1
    t = pd.read_sql(sql_str1,engine)
    x2 = x.loc[x.tradeDate>str(t.tradedate[0])]
    if not x2.empty:
        x2.rename(columns={'ticker':'symbol'},inplace=True)
        x2.to_sql(tn1,engine,if_exists='append',index=False,chunksize=3000)
        print('每日股票数据更新至%s' % x2.tradeDate.max())
        os.remove(fns_data[data_id])
    else:
        print('每日股数据已经是最新的%s，无需更新' % str(t.tradedate[0]))
#4 tradingdate
data_id = 3
if len(fns_data[data_id])>0:
    x = pd.read_csv(fns_data[data_id],engine='python')
    tn1 = 'yq_tradingdate_future'
    """    
    sql_str1 = 'select tradingdate from %s order by tradingdate desc limit 1' % tn1
    try:
        t = pd.read_sql(sql_str1,engine)
    except:
        t = pd.DataFrame()            
    if not t.empty:
        tt_str = str(t.tradingdate[0])
        x2 = x.loc[x.tradingdate>tt_str]
    else:
        tt_str = '1990-01-01'
        x2 = x
    """
    tt_str = '1990-01-01'
    x2 = x
    if not x2.empty:
        x2.to_sql(tn1,engine,if_exists='replace',index=False,chunksize=3000)   #每次都重新更新
        print('交易日数据更新至%s' % x2.tradingdate.max())
        os.remove(fns_data[data_id])
    else:
        print('交易日数据已经是最新的%s，无需更新' % tt_str)        
#5 mongth data
data_id = 4
if len(fns_data[data_id])>0:
    x = pd.read_csv(fns_data[data_id],dtype={'ticker':str},engine='python')
    tn1 = 'MktEqumAdjAfGet'
    sql_str1 = 'select enddate from %s order by enddate desc limit 1' % tn1
    t = pd.read_sql(sql_str1,engine)
    x2 = x.loc[x.endDate>str(t.enddate[0])]
    if not x2.empty:
        #x2.rename(columns={'ticker':'symbol'},inplace=True)
        x2.to_sql(tn1,engine,if_exists='append',index=False,chunksize=3000)
        print('每月股票数据更新至%s' % x2.endDate.max())        
    else:
        print('每月股数据已经是最新的%s，无需更新' % str(t.enddate[0]))
    os.remove(fns_data[data_id])
#6 行业数据
data_id = 5
if len(fns_data[data_id])>0:
    tn = 'yq_industry_sw'
    x = pd.read_csv(fns_data[data_id],dtype={'ticker':str},engine='python',encoding='utf-8')
    x.to_sql(tn,engine,if_exists='replace',index=False,chunksize=3000)
    print('行业数据已更新')
    os.remove(fns_data[data_id])
#7 st
data_id = 6
if len(fns_data[data_id])>0:
    x = pd.read_csv(fns_data[data_id],dtype={'ticker':str},engine='python')
    tn1 = 'st_info'
    sql_str1 = 'select tradedate from %s order by tradedate desc limit 1' % tn1
    t = pd.read_sql(sql_str1,engine)
    x2 = x.loc[x.tradeDate>str(t.tradedate[0])]
    if not x2.empty:
        #x2.rename(columns={'ticker':'symbol'},inplace=True)
        x2.to_sql(tn1,engine,if_exists='append',index=False,chunksize=3000)
        print('st数据更新至%s' % x2.tradeDate.max())
        os.remove(fns_data[data_id])
    else:
        print('st数据已经是最新的%s，无需更新' % str(t.tradedate[0])) 
#8 指数成分股数据
data_id = 7
if len(fns_data[data_id])>0:
    for sub_fn in fns_data[data_id]:
        x = pd.read_csv(sub_fn,dtype={'ticker':str,'consTickerSymbol':str},engine='python')
        os.remove(sub_fn)
        if len(x.index)==0:
            continue
        
        x.rename(columns={'effDate':'tradingdate','consTickerSymbol':'symbol'},inplace=True)
        tn1 = 'IdxCloseWeightGet'
        sql_str1 = 'select tradingdate from %s where ticker="%s" order by tradingdate desc limit 1' % (tn1,x.ticker[0])
        t = pd.read_sql(sql_str1,engine)
        if not t.empty:
            tt_str = str(t.tradingdate[0])
        else:
            tt_str = '1990-01-01'        
        #tt_str = '1990-01-01'
        x2 = x.loc[x.tradingdate>tt_str]
        if not x2.empty:
            #x2.rename(columns={'ticker':'symbol'},inplace=True)
            x2.to_sql(tn1,engine,if_exists='append',index=False,chunksize=3000)
            print('%s指数成分股数据更新至%s' % (x.ticker[0],x2.tradingdate.max()))
        else:
            print('%s指数成分股数据已经是最新的%s，无需更新' % (x.ticker[0],tt_str)) 
            
#9 补充指数数据
data_id = 8
if len(fns_data[data_id])>0:
    for sub_fn in fns_data[data_id]:
        x = pd.read_csv(sub_fn,index_col=0,engine='python')
        os.remove(sub_fn)
        if len(x.index)==0:
            continue
        
        tn1 = 'yuqer_cal'
        sql_str1 = 'select calendarDate from %s  order by calendarDate desc limit 1' % (tn1)
        t = pd.read_sql(sql_str1,engine)
        if not t.empty:
            tt_str = str(t.calendarDate[0])
        else:
            tt_str = '1990-01-01'        
        #tt_str = '1990-01-01'
        x2 = x.loc[x.calendarDate>tt_str]
        if not x2.empty:
            #x2.rename(columns={'ticker':'symbol'},inplace=True)
            x2.to_sql(tn1,engine,if_exists='append',index=False,chunksize=3000)
            print('交易日数据更新至%s' % (x2.calendarDate.max()))
        else:
            print('交易日数据已经是最新的%s，无需更新' % (tt_str)) 
#10 后复权数据
data_id = 9
if len(fns_data[data_id])>0:
    for sub_fn in fns_data[data_id]:
        x = pd.read_csv(sub_fn,engine='python',dtype={'ticker':str})
        os.remove(sub_fn)
        if len(x.index)==0:
            continue
        
        tn1 = 'MktEqudAdjAfGet'
        sql_str1 = 'select tradeDate from %s  order by tradeDate desc limit 1' % (tn1)
        t = pd.read_sql(sql_str1,engine)
        if not t.empty:
            tt_str = str(t.tradeDate[0])
        else:
            tt_str = '1990-01-01'        
        #tt_str = '1990-01-01'
        x2 = x.loc[x.tradeDate>tt_str]
        if not x2.empty:
            #x2.rename(columns={'ticker':'symbol'},inplace=True)
            x2.to_sql(tn1,engine,if_exists='append',index=False,chunksize=3000)
            print('后复权数据更新至%s' % (x2.tradeDate.max()))
        else:
            print('后复权数据已经是最新的%s，无需更新' % (tt_str))             
#11 指数月度数据mongth data
data_id = 10
if len(fns_data[data_id])>0:
    x = pd.read_csv(fns_data[data_id],dtype={'ticker':str},engine='python')
    tn1 = 'yq_index_month'
    x.rename(columns={'ticker':'symbol'},inplace=True)
    sql_str1 = 'select enddate from %s order by enddate desc limit 1' % tn1
    t = pd.read_sql(sql_str1,engine)
    x2 = x.loc[x.endDate>str(t.enddate[0])]
    if not x2.empty:
        #x2.rename(columns={'ticker':'symbol'},inplace=True)
        x2.to_sql(tn1,engine,if_exists='append',index=False,chunksize=3000)
        print('指数每月数据更新至%s' % x2.endDate.max())
    else:
        print('指数每月数据已经是最新的%s，无需更新' % str(t.enddate[0]))
    os.remove(fns_data[data_id])
#S26计算中性化收益需要因子
data_id = 11    
if len(fns_data[data_id])>0:
    sub_fn = fns_data[data_id]
    x1 = pd.read_csv(sub_fn,header=0,engine='python',encoding = "utf-8",index_col=False)
    x1['ticker'] = x1['ticker'].apply(add_0)
    table_name = 'yq_MktStockFactorsOneDayGet_S26'
    sql_str1 = 'select tradeDate from %s order by tradeDate desc limit 1' % table_name
    t = pd.read_sql(sql_str1,engineS26)
    info = 'S26 10 Factors data'
    if not t.empty:
        tt_str = str(t.tradeDate[0])
    else:
        tt_str = '1990-01-01'
    x2 = x1.loc[x1.tradeDate>tt_str]
    if not x2.empty:    
        x2.to_sql(table_name,engineS26,if_exists='append',index=False,chunksize=3000)
        print('%s:%s' % (info,x2.tradeDate.max()))
    else:
        print('%s已经是最新的%s，无需更新' % (info,tt_str))
    os.remove(sub_fn) 
    
data_id=12
if len(fns_data[data_id])>0:
    sub_id = data_id
    x1 = pd.read_csv(fns_data[sub_id],header=0,engine='python',encoding = "utf-8",index_col=False)
    x1['ticker'] = x1['ticker'].apply(add_0)
    table_name = 'yq_MktStockFactorsOneDayGet_add_S26'
    sql_str1 = 'select tradeDate from %s order by tradeDate desc limit 1' % table_name
    t = pd.read_sql(sql_str1,engineS26)
    info = 'S26 3 added Factors'
    if not t.empty:
        tt_str = str(t.tradeDate[0])
    else:
        tt_str = '1990-01-01'
    x2 = x1.loc[x1.tradeDate>tt_str]
    if not x2.empty:    
        x2.to_sql(table_name,engineS26,if_exists='append',index=False,chunksize=3000)
        print('%s:%s' % (info,x2.tradeDate.max()))
    else:
        print('%s已经是最新的%s，无需更新' % (info,tt_str))
    os.remove(fns_data[sub_id]) 