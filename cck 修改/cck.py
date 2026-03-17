# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:37:56 2020

@author: Asus
"""

'''

基本思想与模型

现象
龙头股的上涨使得投资者对板块内其它尚未起涨的股票形成盈利预期，导致资金对此类股票形成追逐效应，体现为“头羊”领涨，“群羊”跟涨的现象。

龙头起涨，随后该行业指数亦跟风上涨，板块成分股收益率相对市场收益率的离散程度降低。据此通过离散度来识别羊群效应。

CCK

其核心思想是通过组合成分股收益率相对于市场收益率Rm离散程度的变化识别羊群效应的发生。
CSADt=1N∑i=1N∣∣Ri,t−Rm,t∣∣

有CAMP推导：
E(CSADt)=E(1N∑i=1N∣∣Ri,t−Rm,t∣∣) =1N(∑i=1N∣∣E(Ri,t)−E(Rm,t)∣∣) =1N(∑i=1N∣∣γ0+βiE(Rm,t−γ0)−(γ0+βmE(Rm,t−γ0))∣∣) =1N∑i=1N|βi−βm|E(Rm,t−γ0)

其中，βi代表股票的beta，βm代表了行业的beta。

导数关系：∂E(CSADt)∂E(Rm,t)=1N∑Ni=1|βi−βm|>0。

当存在羊群效应时，线性正向关系打破。

构造全市场的回归检验：
CSADt=α+β1∣∣Rm,t∣∣+β2R2m,t+εt

R2m,t的系数显著为负时说明羊群效应发生.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statsmodels.api as sm
from sqlalchemy import create_engine
import json

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
sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate''' 
## 数据的起始与终止时间
begin = '20070101'
end = '20171231'
## 得到月度日历
def get_calender():
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" order by tradeDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x
    
def get_month_calender():
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" and endDate>="%s" order by endDate'''%(begin)
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

month_date = pd.Series(get_month_calender())
cal_data = pd.Series(get_calender())
#cal_data = cal_data.set_index(cal_data.values)
#print(cal_data)

hs_df = pd.read_sql('''select * from yq_index where indexID = "000300.ZICN" 
                          and tradeDate>="%s" and tradeDate<="%s"''' %(begin,end),engine)
hs_df.index = hs_df['tradeDate'].values
#print(hs_df.head())

for i,j in zip(month_date[:-1], month_date[1:]):
    dt = hs_df.loc[i:j, :][1:]['CHGPct'].apply(lambda x: np.abs(x))
    
    
date_1, date_2 = i,j
m_name = '000300'

def get_IdxCons(intoDate,ticker='000300'):
    #nearst 时间
    sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
            and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
        ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' %(ticker,
        ticker,intoDate)
    x = pd.read_sql(sql_str1,engine)
    x = x['symbol'].values   
    return x
    
tickers=get_IdxCons(intoDate=date_2)
tickers.sort()

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

def get_return(date, tickers, field = ['symbol', 'chgPct']):
    ## date已改，写demo自动补全的时候没注意
    df_begin = pd.DataFrame()
    for ticker in tickers:
        df_begin_1 = chs_factor(ticker=ticker ,begin = date,end= date,field=field)
        #print(df_begin_1)
        df_begin = df_begin.append(df_begin_1)
    #print(df_begin)
    df_begin.index = df_begin['ticker'].values
    
    return df_begin['chgPct']

    
def get_csda(date, tickers, field = ['symbol', 'chgPct'], m_name = '000300.ZICN',engine=engine):
    r_i = get_return(date = date, tickers = tickers, field= field)
    hs_300_data = pd.read_sql('''select * from yq_index where indexID = "%s" and tradeDate>="%s" and tradeDate<="%s"''' %(m_name,date,date),engine)

    #r_m = DataAPI.MktIdxdGet(indexID=u"",ticker=m_name ,tradeDate = date ,beginDate= '', endDate='' ,exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")
    r_m = hs_300_data['CHGPct'].values
    csda = np.abs(r_i - r_m).mean()
    return csda

def get_date_range_csda(date_1, date_2, field = ['symbol', 'chgPct'], m_name = '000300.ZICN',engine=engine):
    csda_list = []
    #for i in cal_data.loc[date_1 : date_2, 'calendarDate'][1:]:
    begin_index = cal_data[cal_data == date_1].index[0]
    end_index = cal_data[cal_data == date_2].index[0]
    for i in cal_data.loc[begin_index:end_index]:
        tickers = get_IdxCons(intoDate=i)
        csda = get_csda(date = i, tickers = tickers, field= field, m_name= m_name, engine=engine)
        csda_list.append(csda)
    return pd.Series(csda_list, index= cal_data.loc[begin_index : end_index])

## 获得月度数据
m_name = '000300.ZICN'
print('*'*50)
r_m = pd.read_sql('''select * from yq_index where indexID = "%s" and tradeDate>="%s" and tradeDate<="%s"''' %(m_name,date_1,date_2),engine)
#r_m = DataAPI.MktIdxdGet(indexID=u"",ticker=m_name ,tradeDate = '',beginDate= date_1, endDate=date_2,exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")
r_m.index = r_m['tradeDate'].values
r_m = r_m['CHGPct']

#print(date_1)
#print(date_2)
csda_month = get_date_range_csda(date_1, date_2)
## 回归判定羊群效应
x = sm.add_constant(pd.concat([np.abs(r_m), np.square(r_m)], axis = 1))
x.columns = ['const','ads_rm','sq_rm']
#print('*'*50)
#print(csda_month)
#print(x)
model = sm.OLS(csda_month, x).fit()
## 显著性判别
p = model.pvalues['sq_rm']
## 参数判别
beta = model.params['sq_rm']
## 趋势判断
r = np.prod(r_m + 1)

date_list = []
p_list = []
v_list = []
r_list = []
for i,j in zip(month_date[:-1], month_date[1:]):
    m_name = '000300.ZICN'
    date_1, date_2 = i,j
    print(i,j)
    #r_m = DataAPI.MktIdxdGet(indexID=u"",ticker=m_name ,tradeDate = '',beginDate= date_1, endDate=date_2,exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")
    r_m = pd.read_sql('''select * from yq_index where indexID = "%s" and tradeDate>="%s" and tradeDate<="%s"''' %(m_name,date_1,date_2),engine)
    r_m.index = r_m['tradeDate'].values
    #r_m = r_m['CHGPct'].iloc[1:]
    r_m = r_m['CHGPct']
    csda_month = get_date_range_csda(date_1, date_2)
    ## 回归判定羊群效应
    x = sm.add_constant(pd.concat([np.abs(r_m), np.square(r_m)], axis = 1))
    x.columns = ['const','ads_rm','sq_rm']
    model = sm.OLS(csda_month, x).fit()
    ## 显著性判别
    p = model.pvalues['sq_rm']
    ## 参数判别
    beta = model.params['sq_rm']
    ## 趋势判断
    r = np.prod(r_m + 1)
    ## 保存数据
    date_list.append(date_2)
    p_list.append(p)
    v_list.append(beta)
    r_list.append(r)
print(m_name)
print(month_date[0],month_date[-1])    
hs_index = pd.read_sql('''select * from yq_index where indexID = "%s" and tradeDate>="%s" and tradeDate<="%s"''' %(m_name,month_date[0],month_date[-1]),engine)    
#hs_index = DataAPI.MktIdxdGet(indexID=u"",ticker=m_name ,tradeDate = '',beginDate= month_date[0], endDate=month_date[-1],exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")

hs_index['closeIndex'].plot()
plt.show()
for d,p,v,r in zip(date_list, p_list, v_list, r_list):
    if v < 0:
        print(d, r, p, v)
        h = hs_index[hs_index['tradeDate'] == d]
        x = h.index
        y = h['closeIndex'].values
        if r > 1:
            plt.plot(x, y, 'r*')
        if r < 1:
            plt.plot(x, y, 'y*')
            
            
result_df = pd.DataFrame([v_list, p_list, r_list], columns= date_list, index = ['v', 'p','r']).T
re = result_df.copy()
## 胜率统计
result_df['r'] = result_df['r'].shift(-1)

for p in np.linspace(0, 0.8, 9):
    print('p values', p)
    r1 = result_df[(re['v'] < 0) & (re['p'] > p)]
    # print('win rate', (r1['r'] > 1).sum() / float(r1.shape[0]))
    r2 = result_df[(re['v'] < 0) & (re['r'] > 1) & (re['p'] > p)]
    r2 = r2[r2['v'] < 0]
    # print('up win rate', (r2['r'] > 1).sum() / float(r2.shape[0]))
    r3 = result_df[(re['v'] < 0) & (re['r'] < 1) & (re['p'] > p)]
    r3 = r3[r3['v'] < 0]
    # print('down win rate', (r3['r'] > 1).sum() / float(r3.shape[0]))
    print('up win rate', (r2['r'] > 1).sum(), (r2['r'] > 1).sum() / float(r2.shape[0]), 'down win rate', (r3['r'] > 1).sum(), (r3['r'] > 1).sum() / float(r3.shape[0]) )
    

'''    
模型改进

CSADt=α+β1∣∣Rm,t∣∣+β2R2m,t 

CSADt=α+β1∣∣Rsmb,t∣∣+β2R2smb,t+εt
对市场的收益率模型的构建可以很容易的推广到其他因子上面。
smb做为市值因子，是全市场的因子。在选择300指数，行业指数等是不能用了。研报除了全市场使用了smb因子之外，都是用的指数市场因子。因子，没有做smb因子的羊群。不过逻辑是一样的。

'''

#t = DataAPI.MktIdxdGet(indexID=u"",ticker=u"",tradeDate=u"20161219",beginDate=u"",endDate=u"",exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")
#print(t[t['secShortName'].apply(lambda x:True if '50' in x else False)])


'''
说明

以下是指数的动态回测，只需要将m_name改成对应的指数代码，begin与end代表时间区间，修改即可回测。
研报回测太多，全部回测耗时太久。可以根据需要自行回测。
其中，t数据框已有基本所有的指数代码。
研报并没有说明显著性水平的参数选择过程，试验之后，觉得不考虑显著性有一个较高的胜率。
'''

## 数据的起始与终止时间
begin = '20070101'
end = '20171231'
## 得到月度日历
cal_data = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate= begin, endDate= end, field=['calendarDate','isMonthEnd','isOpen'])
cal_data = cal_data[cal_data['isOpen'] == 1]
cal_data.index = cal_data['calendarDate'].values
month_date = cal_data[cal_data['isMonthEnd'] == 1]['calendarDate'].values


date_list = []
p_list = []
v_list = []
r_list = []
for i,j in zip(month_date[:-1], month_date[1:]):
    ### 在此处修改指数代码
    m_name = '000016'
    date_1, date_2 = i,j
    r_m = DataAPI.MktIdxdGet(indexID=u"",ticker=m_name ,tradeDate = '',beginDate= date_1, endDate=date_2,exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")
    r_m.index = r_m['tradeDate'].values
    r_m = r_m['CHGPct'].iloc[1:]
    csda_month = get_date_range_csda(date_1, date_2)
    ## 回归判定羊群效应
    x = sm.add_constant(pd.concat([np.abs(r_m), np.square(r_m)], axis = 1))
    x.columns = ['const','ads_rm','sq_rm']
    model = sm.OLS(csda_month, x).fit()
    ## 显著性判别
    p = model.pvalues['sq_rm']
    ## 参数判别
    beta = model.params['sq_rm']
    ## 趋势判断
    r = np.prod(r_m + 1)
    ## 保存数据
    date_list.append(date_2)
    p_list.append(p)
    v_list.append(beta)
    r_list.append(r)
    print(date_2)
    
hs_index = DataAPI.MktIdxdGet(indexID=u"",ticker=m_name ,tradeDate = '',beginDate= month_date[0], endDate=month_date[-1],exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")

hs_index['closeIndex'].plot()
for d,p,v,r in zip(date_list, p_list, v_list, r_list):
    # ## 加入显著性得到的结果
    # if v < 0 and p < 0.15:
    #     print(d, r, p, v)
    #     h = hs_index[hs_index['tradeDate'] == d]
    #     x = h.index
    #     y = h['closeIndex'].values
    #     if r > 1:
    #         plt.plot(x, y, 'r*')
    #     if r < 1:
    #         plt.plot(x, y, 'y*')
    # 不加显著新得到的结果
    if v < 0:
        print(d, r, p, v)
        h = hs_index[hs_index['tradeDate'] == d]
        x = h.index
        y = h['closeIndex'].values
        if r > 1:
            plt.plot(x, y, 'r*')
        if r < 1:
            plt.plot(x, y, 'y*')
            
result_df = pd.DataFrame([v_list, p_list, r_list], columns= date_list, index = ['v', 'p','r']).T
re = result_df.copy()
## 胜率统计
result_df['r'] = result_df['r'].shift(-1)
## p值检验
for p in np.linspace(0, 0.8, 9):
    print('p values', p)
    r1 = result_df[(re['v'] < 0) & (re['p'] > p)]
    # print('win rate', (r1['r'] > 1).sum() / float(r1.shape[0]))
    r2 = result_df[(re['v'] < 0) & (re['r'] > 1) & (re['p'] > p)]
    r2 = r2[r2['v'] < 0]
    # print('up win rate', (r2['r'] > 1).sum() / float(r2.shape[0]))
    r3 = result_df[(re['v'] < 0) & (re['r'] < 1) & (re['p'] > p)]
    r3 = r3[r3['v'] < 0]
    # print('down win rate', (r3['r'] > 1).sum() / float(r3.shape[0]))
    print('up win rate', (r2['r'] > 1).sum(), (r2['r'] > 1).sum() / float(r2.shape[0]), 'down win rate', (r3['r'] > 1).sum(), (r3['r'] > 1).sum() / float(r3.shape[0]) )
    

'''
一系列检验，将p值定在0.3

'''



result_df = pd.DataFrame([v_list, p_list, r_list], columns= date_list, index = ['v', 'p','r']).T
re = result_df.copy()
## 胜率统计
result_df['r'] = result_df['r'].shift(-1)
r1 = result_df[(re['v'] < 0) & (re['p'] > 0.3)]
# print('win rate', (r1['r'] > 1).sum() / float(r1.shape[0]))
r2 = result_df[(re['v'] < 0) & (re['r'] > 1) & (re['p'] > 0.3)]
r2 = r2[r2['v'] < 0]
# print('up win rate', (r2['r'] > 1).sum() / float(r2.shape[0]))
r3 = result_df[(re['v'] < 0) & (re['r'] < 1) & (re['p'] > 0.3)]
r3 = r3[r3['v'] < 0]
# print('down win rate', (r3['r'] > 1).sum() / float(r3.shape[0]))
print('up win rate', (r2['r'] > 1).sum(), (r2['r'] > 1).sum() / float(r2.shape[0]), 'down win rate', (r3['r'] > 1).sum(), (r3['r'] > 1).sum() / float(r3.shape[0]) )


rs_2 = np.array([(hs_index[hs_index['tradeDate'] > date][:22]['CHGPct'] + 1).cumprod().values - 1 for date in r2.index])
rs_3 = np.array([(hs_index[hs_index['tradeDate'] > date][:22]['CHGPct'] + 1).cumprod().values - 1 for date in r3.index])

plt.plot(rs_2.mean(axis = 0))
plt.plot(rs_3.mean(axis = 0))
plt.show()


## 数据的起始与终止时间
begin = '20070101'
end = '20180101'
## 得到月度日历
cal_data = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate= begin, endDate= end, field=['calendarDate','isMonthEnd','isOpen'])
cal_data = cal_data[cal_data['isOpen'] == 1]
cal_data.index = cal_data['calendarDate'].values
month_date = cal_data[cal_data['isMonthEnd'] == 1]['calendarDate'].values


rs2_list = []
rs3_list = []
for m_name in ['000016','000300','000905','399101']:
    date_list = []
    p_list = []
    v_list = []
    r_list = []
    for i,j in zip(month_date[:-1], month_date[1:]):
        ### 在此处修改指数代码
        # m_name = '000016'
        date_1, date_2 = i,j
        r_m = DataAPI.MktIdxdGet(indexID=u"",ticker=m_name ,tradeDate = '',beginDate= date_1, endDate=date_2,exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")
        r_m.index = r_m['tradeDate'].values
        r_m = r_m['CHGPct'].iloc[1:]
        csda_month = get_date_range_csda(date_1, date_2)
        ## 回归判定羊群效应
        x = sm.add_constant(pd.concat([np.abs(r_m), np.square(r_m)], axis = 1))
        x.columns = ['const','ads_rm','sq_rm']
        model = sm.OLS(csda_month, x).fit()
        ## 显著性判别
        p = model.pvalues['sq_rm']
        ## 参数判别
        beta = model.params['sq_rm']
        ## 趋势判断
        r = np.prod(r_m + 1)
        ## 保存数据
        date_list.append(date_2)
        p_list.append(p)
        v_list.append(beta)
        r_list.append(r)
    ## 计算指数
    hs_index = DataAPI.MktIdxdGet(indexID=u"",ticker=m_name ,tradeDate = '',beginDate= month_date[0], endDate=month_date[-1],exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")
    ##
    result_df = pd.DataFrame([v_list, p_list, r_list], columns= date_list, index = ['v', 'p','r']).T
    re = result_df.copy()
    ## 胜率统计
    result_df['r'] = result_df['r'].shift(-1)
    r1 = result_df[(re['v'] < 0) & (re['p'] > 0.3)]
    print(m_name, 'win rate', (r1['r'] > 1).sum() / float(r1.shape[0]))
    r2 = result_df[(re['v'] < 0) & (re['r'] > 1) & (re['p'] > 0.3)]
    r2 = r2[r2['v'] < 0]
    print(m_name, 'up win rate', (r2['r'] > 1).sum() / float(r2.shape[0]))
    r3 = result_df[(re['v'] < 0) & (re['r'] < 1) & (re['p'] > 0.3)]
    r3 = r3[r3['v'] < 0]
    print(m_name, 'down win rate', (r3['r'] > 1).sum() / float(r3.shape[0]))
    rs_2 = np.array([(hs_index[hs_index['tradeDate'] > date][:22]['CHGPct'] + 1).cumprod().values - 1 for date in r2.index])
    rs_3 = np.array([(hs_index[hs_index['tradeDate'] > date][:22]['CHGPct'] + 1).cumprod().values - 1 for date in r3.index])
    rs2_list.append(rs_2)
    rs3_list.append(rs_3)
'''
    
避免一种情况：
当数据的最有一个月出现信号，则无法后推22天检测其累计收益。
故，最后一个回测不取。i[:-1]的情况。
如果想回测20190201一般来说，推后一个月，end时间取20190301就可以了。
'''
plt.figure(figsize= (16, 8))
for i,name in zip(rs2_list, ['50','300','500','small']):
    plt.plot(np.array(i[:-1]).mean(axis = 0), label= name)
plt.legend()
plt.show()

plt.figure(figsize= (16, 8))
for i,name in zip(rs3_list, ['50','300','500','small']):
    plt.plot(np.array(i[:-1]).mean(axis = 0), label = name)
plt.legend()
plt.show()