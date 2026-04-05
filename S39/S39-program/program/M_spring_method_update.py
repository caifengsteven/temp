# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:55:00 2020
增加空头组结果
需要将数据写入数据库，减少计算量

s1=tickers_df.apply(lambda x :','.join(x.astype(str)),1).to_frame()
s2=tickers_df_less.apply(lambda x :','.join(x.astype(str)),1).to_frame()
s1.columns = ['more_r']
s2.columns = ['less_r']
s3 = s1.join(s2)
s3['index_code'] = index_code
s3['tradingdate'] = s3.index
s3['method_ID'] = 1
s3.to_sql(tn_symbol,engine_s37,if_exists='append',index=False,chunksize=3000)
s3.to_csv('re%s.csv' % index_code)


@author: adair-9960
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal
import statsmodels.api as sm
#import math
from sqlalchemy import create_engine
import json
from datetime import datetime,timedelta
#import multiprocessing as mp
#pool = mp.Pool(8)
t0 = datetime.now()
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
db_name2 = 's37'
#eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name2)
engine_s37 = create_engine(eng_str)
tn_symbol = 'symbol_pool_S39'
"""
#后复权 前复权一直在变，无法
sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''    
##前复权需要换为后复权    
def chs_factor(ticker = '000005',begin = begin ,end = end , 
               field = [u'symbol',  u'tradeDate', u'openPrice',
                        u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol',
                        u'turnoverValue',u'dealAmount', u'chgPct',
                        'turnoverRate',u'marketValue']):
    sql_str1 = sql_str_select_data1 % (','.join(field),ticker,begin,end)
    dataday = pd.read_sql(sql_str1,engine)
    field2 = [u'tradeDate',u'accumAdjFactor']
    sql_str2 = sql_str_select_data2 % (','.join(field2),ticker,begin,end)
    dataday2 = pd.read_sql(sql_str2,engine)
    dataday=pd.merge(dataday,dataday2,on=['tradeDate','tradeDate'],how='inner')
    dataday = dataday.applymap(lambda x: np.nan if x == 0 else x)
    ## 对数据补全
    return dataday.fillna(method = 'ffill')
"""
#数字型股票代码补充0
def add_0(x):
    if isinstance(x,int):
        x= '%0.6d' % x
    else:
        x=x.rjust(6,'0')
    return x
#获取基本数据
sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''    
##前复权需要换为后复权    
def chs_factor(ticker = '000005',begin = None ,end = None , 
               field = [u'symbol',  u'tradeDate', u'openPrice',
                        u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol',
                        u'turnoverValue',u'dealAmount', u'chgPct',
                        'turnoverRate',u'marketValue',u'accumAdjFactor']):
    sql_str1 = sql_str_select_data1 % (','.join(field),ticker,begin,end)
    dataday = pd.read_sql(sql_str1,engine)
    dataday = dataday.applymap(lambda x: np.nan if x == 0 else x)
    dataday.rename(columns={'symbol':'ticker'},inplace=True)
    ## 对数据补全
    return dataday.fillna(method = 'ffill')
#获取行业分类数据
def get_industry_class(t):
    sql_str1 = '''select ticker,industryID1 from yuqerdata.yq_industry where 
                industryVersionCD="010303" and intodate <= "%s" and 
                (outDate>"%s" or outDate is null)''' % (t,t)
    x = pd.read_sql(sql_str1,engine)
    return x

##得到月度日历
def get_month_calender():
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" order by endDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

month_date = get_month_calender()
##获取成分股
def get_IdxCons(intoDate,ticker='000300'):
    #nearst 时间
    if ticker=='a':
        sql_str1 = '''
        select symbol from yq_dayprice where tradeDate = (select tradingdate from
        yq_tradingdate_future where tradingdate <="%s" order by tradingdate desc
        limit 1) 
        ''' % intoDate
    else:            
        sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
                and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
            ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' %(ticker,
            ticker,intoDate)
        x = pd.read_sql(sql_str1,engine)
        if len(x)==0:
            sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
                    and tradingdate = "2011-05-31"''' %(ticker)
            x = pd.read_sql(sql_str1,engine)        
    x = x['symbol'].values   
    return x

#获取因子值，最终得到每个月底的alpha和每个月月底的收益，二者一一对应
def get_spring_factor(df, ticker = '000001'):
    df.index = df['tradeDate'].values
    ## 计算B_k
    ## 复权
    df_bk = (df[['openPrice', 'highestPrice', 'lowestPrice', 'closePrice']].T * df['accumAdjFactor']).T
    ## 计算vr，sp
    df_bk = df_bk.rolling(rolling_day).mean().mean(axis = 1)
    vr = df_bk / df_bk.shift(1) - 1
    sp = (df['turnoverValue'] - df['turnoverVol'] * df_bk) / ( (df['turnoverVol'] / df['turnoverRate'])  * df_bk.shift(1) )
    
    coef_list = []
    ## 按月回归得到塑性系数
    for i,j in zip(month_date[:-1], month_date[1:]):
        y = vr.loc[i:j][1:].dropna()
        x = sp.loc[i:j][1:].dropna()
        x = x.apply(lambda x: np.sign(x)*np.sqrt(np.abs(x)))
        try:
            coef = sm.OLS(y, x).fit().params.values[0]
            coef_list.append(coef)
        except:
            coef = np.nan
            coef_list.append(coef)
            continue
    ## 塑性系数序列
    coef_sr = pd.Series(coef_list, index= month_date[1:], name = ticker).dropna()
    ## 取N = 12进行指数加权
    coef_t = coef_sr.ewm(span= 12).mean().dropna()
    ## 排除市值因子
    ## 数值补全，保证ols计算
    x_m = df.loc[coef_t.index]['marketValue'].fillna(method = 'bfill').fillna(method = 'ffill')
    alpha = sm.OLS(coef_t, x_m).fit().resid
    alpha.name = ticker
    return alpha, df.loc[coef_sr.index]

## 计算收益
def get_return(df1,name):
    #name = df1['ticker'][-1]
    df1 = df1['closePrice'] * df1['accumAdjFactor']
    df_return = df1 / df1.shift(1) - 1
    df_return.name = name
    return df_return.dropna()

#根据截面上每个行业分类的因子大小，选择分类前2个股票
def get_fac_ticker(c,num_code_keeped=2,acs_sel = False):
    fac_ticker = []
    fac_gc= fac_df.loc[:, c].apply(lambda x: x.sort_values(ascending= acs_sel)[:num_code_keeped], axis = 1)
    for i in fac_gc.applymap(lambda x: False if np.isnan(x) else True).values:
        fac_ticker.append(list(fac_gc.columns[i]))
    return pd.DataFrame(fac_ticker)

def result(v1, v2):
    d1 = v1[1:] / v1[:-1] - 1
    d2 = v2[1:] / v2[:-1] - 1
    beta, alpha = np.polyfit(d1, d2, 1)
    ratio = (v2[-1] - 1) / v2.shape[0] * 12
    sigma = d2.std() * np.sqrt(12)
    info = ratio / sigma
    hedge_index = (d2 - beta * d1)
    hedge_ratio = ((hedge_index + 1).prod() - 1) / v2.shape[0] * 12
    hedge_sigma = hedge_index.std() * np.sqrt(12)
    hedge_info = hedge_ratio / hedge_sigma
    return ratio, info, hedge_ratio, hedge_info

index_pool = ['a','000300','000905','000001']
for index_code in index_pool:
    num_code_keeped = 3
    #alpha系数的大小进行选股策略
    rolling_day = 20
    ## 选择root 回归
    begin = '2010-05-01'
    end = '2090-04-01'
    
    info_date=begin
    delta_1year = timedelta(days=365)
    info_date_1year_before = datetime.strptime(info_date,'%Y-%m-%d')-delta_1year
    info_date_1year_before = info_date_1year_before.strftime('%Y-%m-%d')    
    
    ## 获取300股票代码
    hs_300 = get_IdxCons(begin,ticker=index_code)
    
    sql_str_st= 'select distinct(ticker) from yuqerdata.st_info where tradedate = "%s"'
    st = pd.read_sql(sql_str_st % (begin),engine)
    st['ticker'] = st['ticker'].apply(add_0)
    hs_300 = np.setdiff1d(hs_300,st)
    ## 排除次新股 上市1年以内我们认为是次新股
    sql_str = '''select distinct(ticker)  from yuqerdata.equget  where equTypeCD = "A" 
    and ListSectorCD<=3 and  listDate <="%s" order by ticker'''
    symbol_a = pd.read_sql(sql_str % info_date_1year_before,engine).values
    hs_300 = np.intersect1d(symbol_a,hs_300)    
    
    ## 回测过程
    fac_list=[]
    df_mon_list=[]
    for recorder_num,i in enumerate(hs_300):
        if np.mod(recorder_num,10)==0:
            print('%s-%d' % (index_code,recorder_num))
        ticker = i
        df = chs_factor(ticker= ticker, begin = begin, end= end)
        alpha, df_mon = get_spring_factor(df, ticker= ticker)
        df_mon = get_return(df_mon,ticker)
        fac_list.append(alpha)
        df_mon_list.append(df_mon)
        
    ## 原文使用500和分类进行回测，此处选300的分类前2个股票回测 因子和因子收益
    fac_df = pd.concat(fac_list, axis = 1)
    ret_df = pd.concat(df_mon_list, axis = 1)
    ret_df_less = ret_df.copy()
    
    ticker_class = get_industry_class(end)
    ticker_class =pd.merge(ticker_class,pd.DataFrame({'ticker':hs_300}),on=['ticker','ticker'],how='inner')
    ticker_class = ticker_class[['ticker','industryID1']].drop_duplicates()
    ticker_group = ticker_class.groupby('industryID1')['ticker'].apply(lambda x: x.values)
    
    #获取每个月底的所有选择的股票序列
    tickers_df = pd.concat([get_fac_ticker(ticker_group.iloc[i],num_code_keeped) for i in range(ticker_group.shape[0])],axis = 1)
    tickers_df.index = fac_df.index
    ret_df = ret_df.loc[tickers_df.index].fillna(0)
    #空头组
    tickers_df_less = pd.concat([get_fac_ticker(ticker_group.iloc[i],num_code_keeped,True) for i in range(ticker_group.shape[0])],axis = 1)
    tickers_df_less.index = fac_df.index
    ret_df_less = ret_df_less.loc[tickers_df_less.index].fillna(0)
    #这里感觉应该顺延一个月，因为当月月底选出的股票，应该使用下一个月的收益来计算 !!!!!!!!!!!!!!
    ret_list = []
    for i,j in zip(tickers_df.index[:-1], tickers_df.index[1:]):
        tickers = tickers_df.loc[i].dropna().values
        rets = ret_df.loc[j, tickers].mean()
        ret_list.append(rets)
    #策略每日收益
    ret_sr = pd.Series(ret_list, index= tickers_df.index[1:])
    
    ret_list_less = []
    for i,j in zip(tickers_df_less.index[:-1], tickers_df_less.index[1:]):
        tickers = tickers_df_less.loc[i].dropna().values
        rets = ret_df_less.loc[j, tickers].mean()
        ret_list_less.append(rets)
    #策略每日空头收益
    ret_less_sr = pd.Series(ret_list_less, index= tickers_df_less.index[1:]) 
    
    #hs_300_data = DataAPI.MktIdxmGet(beginDate= begin , endDate= end, indexID=u"000300.ZICN", ticker=u"",field=u"",pandas="1")
    if index_code=='a':
        sub_index_code = '000001'
    hs_300_data = pd.read_sql('''select * from yq_index_month where symbol = "%s" 
                              and endDate>="%s" and endDate<="%s"''' %(sub_index_code,begin,end),engine)
    hs_300_data.index = hs_300_data['endDate'].values
    hs_300_index = hs_300_data.loc[ret_sr.index].sort_index()['closePrice']
    hs_300_index = hs_300_index / hs_300_index[0]
    
    plt.figure()
    (ret_sr.fillna(0) + 1).cumprod().plot()
    hs_300_index.plot()
    
    plt.figure()
    (ret_sr.fillna(0)-ret_less_sr.fillna(0) + 1).cumprod().plot()
    
    a, b, c, d = result(hs_300_index.values, (ret_sr.fillna(0) + 1).cumprod().values)
    print('收益 %s, 信息比率%s, 对冲收益 %s, 对冲信息比率 %s'%(a, b , c, d))
    #df_return = (df1 / df1.shift(1) - 1).shift(-1)
    outperformance = (ret_sr.fillna(0) + 1).cumprod().values - hs_300_index.values
    plt.figure()
    plt.plot(outperformance)
tt = datetime.now()
print('time used %s' % ((tt-t0).total_seconds()))