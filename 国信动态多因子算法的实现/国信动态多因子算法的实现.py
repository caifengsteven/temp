# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:59:04 2020

@author: Asus
"""
'''
本文是国信《45数量化投资技术系列之四十五：基于A股市场选股因子边际效用和有效分散的动态区分度动量策略》的实现。
文章中因子贡献度的含义如下：
1.首先将股票池中的股票按因子进行排名，分别选出排名靠前的20%和排名靠后的20%股票构成两个组合；
2.我们将两个组合的平均收益差和股票池中所有股票前后20%股票的平均收益差相除，得到的比值即为因子贡献度；
3.两个组合的收益率相差越大，则说明该时点此因子对股票的区分度越大；对于正的贡献度，因子值越大，平均收益率越小；对于负的贡献度，因子值越大，平均收益率越大。
因子贡献度的概念如下图所示：
因子贡献度

本文对于通过因子贡献的选股的策略如下：
1.首先计算股票池中各个因子的数值（标准化，中性化，去极值）；
2.然后计算各个因子的贡献度；
3.通过因子的大小对于股票进行排序，序号大的股票理论上上涨的可能性越大；
4.如果因子贡献度小于门限menTH则舍弃该因子；
5.通过以下公式计算股票池中股票的得分：
stockpointi=∑j=1Nrankij∗derrj

其中stockpointi rankij derrj 分别代表第i支股票的得分，第i支股票第j个因子的排名，第j个因子的贡献度；
6.选取得分最高的stocknum支股票；
下面的代码是实现了上述基于因子贡献的选股策略

'''

#coding=utf-8
import sys
sys.path
sys.path.append('G:\dropbox\Dropbox\Dropbox\project folder from my asua computer\Project\lib')
import pandas as pd
import numpy as np
#from CAL.PyCAL import  
from quant_util import *
from datetime import datetime
from pandas.tseries.offsets import BDay
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
def get_industry_class(t):
    sql_str1 = '''select ticker,industryID1 from yuqerdata.yq_industry where 
                industryVersionCD="010303" and intodate <= "%s" and 
                (outDate>"%s" or outDate is null)''' % (t,t)
    x = pd.read_sql(sql_str1,engine)
    return x

def get_month_calender():
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" order by endDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

def get_calender():
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" order by tradeDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

cal= get_calender()

def MktStockFactorsOneDayProGet(tradeDate, fields):
    df = pd.DataFrame()
    i = 1
    for field in fields:
        field = field.lower()
        sql_str1 = '''select symbol, tradingdate, f_val from ''' + field
        sql_str2 = ''' where tradingdate = "%s"''' % (tradeDate)
        sql_str = sql_str1 + sql_str2
        # sql_str = '''select symbol, tradingdate, f_val from '%s' where tradingdate = "%s"'''%(field, tradeDate)
        x = pd.read_sql(sql_str, engine2)
        x = x.rename(columns={"f_val": field})
        if i == 1:
            df = x
        else:
            df = df.merge(x, on=["symbol", "tradingdate"])
        i = i + 1

    return df
#from quartz.api import *

###############
#计算贡献度函数
###############
def get_index_data(index,begin, end):
    
    r_m = pd.read_sql('''select * from yq_index where indexID = "%s" and tradeDate>="%s" and tradeDate<="%s" order by tradeDate''' %(ticker,begin,end),engine)
    return r_m

def ConDegree(index,factorname,conf,beginday,endday): 
    
    ##############################################
    # 提取从beginday到endday的股票池index中的股票数据
    ##############################################
    
    #stock_data=DataAPI.MktEqudGet(secID=index,beginDate=beginday,endDate=endday
    #                              ,field=['secID','closePrice','tradeDate'],pandas="1").set_index('secID').dropna()
    stock_data = get_index_data(index,beginday, endday)
    stock_data = stock_data['secID', 'closeIndex', 'tradeDate'].dropna()
    stock_data['Rerate']=stock_data[stock_data['tradeDate']==endday]['closeIndex']/stock_data[stock_data['tradeDate']==beginday]['closePrice']-1
    stock_data=stock_data[stock_data['tradeDate']==endday]
    stock_data=stock_data.dropna()
    
    ####################
    # 计算前后20%的收益差
    ####################
    
    different=np.mean((stock_data.sort(columns='Rerate').tail(int(len(stock_data)/5))['Rerate'])-np.mean(stock_data.sort(columns='Rerate').head(int(len(stock_data)/5))['Rerate'])) 
    
    #################################################
    # 提取beginday的股票池index中的因子factorname的数据
    #################################################
    
    '''
    factor_data=DataAPI.MktStockFactorsOneDayGet(tradeDate=beginday,secID=stock_data.index
                            ,field=['secID',factorname],pandas='1').set_index('secID').dropna()  
    '''
    factor_data = MktStockFactorsOneDayProGet(beginday, factorname)
    factor_data = factor_data[factor_data['symbol'] in stock_data.index]
    factor_data.set_index('symbol').dropna(inplace=True)
    
    print (factor_data[factorname])
    '''
    stock_data[factorname]=standardize(neutralize(winsorize(factor_data[factorname]),endday.replace('-',''))).dropna()
    '''
    stock_data[factorname]=standardize(neutralize(winsorize(factor_data[factorname]),endday.replace('-',''))).dropna()
    ########################
    # 计算因子前后20%的收益差
    ########################
    
    if(conf):
        different2=np.mean(stock_data.sort(columns=factorname).tail(int(len(stock_data)/5))['Rerate'].dropna())-np.mean(stock_data.sort(columns=factorname).head(int(len(stock_data)/5))['Rerate'].dropna())
    else:
         different2=np.mean(stock_data.sort(columns=factorname).head(int(len(stock_data)/5))['Rerate'].dropna())-np.mean(stock_data.sort(columns=factorname).tail(int(len(stock_data)/5))['Rerate'].dropna()) 
            
    ########################
    # 得出贡献度
    ########################
    
    return different2/different

#########
#选股函数
#########

def dy_stockchoice(index,factorname1='REC',factorname2='CFO2EV',factorname3='LFLO'
                   ,factorname4='OperatingProfitGrowRate',factorname5='ILLIQUIDITY',conf1=1,conf2=1,
                   conf3=0,conf4=1,conf5=0,currday='20160902',periodstr=-5,stocknum=30,menTH=0.08):
    
    ############################
    #提取股票池index中诸因子的数据
    ############################
    #print index
    #print currday
    #endday=cal.advanceDate(currday,Period(periodstr)).strftime("%Y-%m-%d")
    endday = datetime.strptime(currday,"%Y%m%d") +BDay(periodstr)
    print(endday)
    
    
    #print endday
    currday_use=currday.replace('-','')
    factor_list = [factorname1, factorname2, factorname3, factorname4]
    factor_data = MktStockFactorsOneDayProGet(currday, factor_list)
    factor_data['symbol']= factor_data['symbol'].astype(str)
    print(factor_data['symbol'])
    print(index)
    factor_data = factor_data[factor_data['symbol'].isin(index)]
    factor_data = factor_data.set_index('symbol')
    factor_data = factor_data.dropna(inplace = True)
    print(factor_data)
    '''
    factor_data=DataAPI.MktStockFactorsOneDayGet(tradeDate=currday,secID=index
                                ,field=['secID',factorname1,factorname2,factorname3,factorname4,factorname5]
                                ,pandas='1').set_index('secID').dropna()
    '''
#    factor_data[factorname1]=standardize(neutralize(winsorize(factor_data[factorname1])
#                                                 ,currday_use,'short','SW2')).dropna()
#    factor_data[factorname2]=standardize(neutralize(winsorize(factor_data[factorname2])
#                                                 ,currday_use,'short','SW2')).dropna()
#    factor_data[factorname3]=standardize(neutralize(winsorize(factor_data[factorname3])
#                                                 ,currday_use,'short','SW2')).dropna()
#    factor_data[factorname4]=standardize(neutralize(winsorize(factor_data[factorname4])
#                                                 ,currday_use,'short','SW2')).dropna()
#    factor_data[factorname5]=standardize(neutralize(winsorize(factor_data[factorname5])
#                                                 ,currday_use,'short','SW2')).dropna()
    factor_data[factorname1]=standardize(neutralize(winsorize(factor_data[factorname1]),currday_use)).dropna()
    factor_data[factorname2]=standardize(neutralize(winsorize(factor_data[factorname2]),currday_use)).dropna()
    factor_data[factorname3]=standardize(neutralize(winsorize(factor_data[factorname3]),currday_use)).dropna()
    factor_data[factorname4]=standardize(neutralize(winsorize(factor_data[factorname4]),currday_use)).dropna()
    factor_data[factorname5]=standardize(neutralize(winsorize(factor_data[factorname5]),currday_use)).dropna()

    ##################
    #计算诸因子的贡献度
    ##################   
    
    derr1=ConDegree(index,factorname1,conf1,endday,currday)
    derr2=ConDegree(index,factorname2,conf2,endday,currday)
    derr3=ConDegree(index,factorname3,conf3,endday,currday)
    derr4=ConDegree(index,factorname4,conf4,endday,currday)
    derr5=ConDegree(index,factorname5,conf5,endday,currday)
    
    #########################
    #计算股票池下诸因子的rank排序
    #########################  
    
    '''factorname1'''
    if(conf1):
        factor_data=factor_data.sort(columns = factorname1,ascending=True).dropna()
    else:
        factor_data=factor_data.sort(columns = factorname1,ascending=False).dropna()        
    factor_data[factorname1+'rank']=pd.Series()
    i=1
    for stock in factor_data.index:
        factor_data[factorname1+'rank'][stock]=i
        i=i+1
    '''factorname2'''
    if(conf2):
        factor_data=factor_data.sort(columns = factorname2,ascending=True).dropna()
    else:
        factor_data=factor_data.sort(columns = factorname2,ascending=False).dropna()  
    factor_data[factorname2+'rank']=pd.Series()
    i=1
    for stock in factor_data.index:
        factor_data[factorname2+'rank'][stock]=i
        i=i+1
    '''factorname3'''
    if(conf3):
        factor_data=factor_data.sort(columns = factorname3,ascending=True).dropna()
    else:
        factor_data=factor_data.sort(columns = factorname3,ascending=False).dropna() 
    factor_data[factorname3+'rank']=pd.Series()
    i=1
    for stock in factor_data.index:
        factor_data[factorname3+'rank'][stock]=i
        i=i+1
    '''factorname4'''
    if(conf4):
        factor_data=factor_data.sort(columns = factorname4,ascending=True).dropna()
    else:
        factor_data=factor_data.sort(columns = factorname4,ascending=False).dropna() 
    factor_data[factorname4+'rank']=pd.Series()
    i=1
    for stock in factor_data.index:
        factor_data[factorname4+'rank'][stock]=i     
        i=i+1
    '''filename5 该因子为双向因子，因子值越大，股票收益越高则对于因子排序使用升序，否则降序'''
    if(derr5<0):
        if(conf5):
            conf5=0
        else:
            conf5=1
    if(conf5):
        factor_data=factor_data.sort(columns = factorname5,ascending=True).dropna()
    else:
        factor_data=factor_data.sort(columns = factorname5,ascending=False).dropna() 
    factor_data[factorname5+'rank']=pd.Series()
    i=1
    for stock in factor_data.index:
        factor_data[factorname5+'rank'][stock]=i     
        i=i+1

    ######################################################################
    #如果贡献度大于阈值则将因子作为选股打分权重,当confi=-1时该因子不计入选股打分
    ######################################################################
        
    if(derr1>menTH and conf1!=-1):
        factor_data[factorname1+'use']=derr1
    else:
        factor_data[factorname1+'use']=0
    if(derr2>menTH and conf2!=-1):
        factor_data[factorname2+'use']=derr2
    else:
        factor_data[factorname2+'use']=0
    if(derr3>menTH and conf3!=-1):
        factor_data[factorname3+'use']=derr3
    else:
        factor_data[factorname3+'use']=0
    if(derr4>menTH and conf4!=-1):
        factor_data[factorname4+'use']=derr4
    else:
        factor_data[factorname4+'use']=0
    if(abs(derr5)>menTH*1.2 and conf5!=-1):
        factor_data[factorname5+'use']=abs(derr5)
    else:
        factor_data[factorname5+'use']=0
        
    #################
    #按照因子进行打分
    #################
    
    if(derr5>0):
        factor_data['score']=factor_data[factorname1+'rank']*factor_data[factorname1+'use']+factor_data[factorname2+'rank']*factor_data[factorname2+'use']+factor_data[factorname3+'rank']*factor_data[factorname3+'use']+factor_data[factorname4+'rank']*factor_data[factorname4+'use']+factor_data[factorname5+'rank']*factor_data[factorname5+'use']
    else:
            factor_data['score']=factor_data[factorname1+'rank']*factor_data[factorname1+'use']+factor_data[factorname2+'rank']*factor_data[factorname2+'use']+factor_data[factorname3+'rank']*factor_data[factorname3+'use']+factor_data[factorname4+'rank']*factor_data[factorname4+'use']+(len(factor_data)-factor_data[factorname5+'rank'])*factor_data[factorname5+'use']    
            
    return factor_data.sort(columns = 'score').tail(stocknum).index


'''

下面我们对于策略进行总体的实现，为了减少回撤，我们引入二八大小盘轮动策略，具体如下：
1.计算沪深300和创业板的20天收益率：
2.1如果沪深300和创业板的20天收益率小于0，则清盘；
2.2如果沪深300的20天收益率大于创业板的20天收益，则在沪深300中得到动态多因子打分的股票num支，如果打分所得的股票与持有的股票有60%的不同则换仓；
2.3如果沪深300的20天收益率小于创业板的20天收益，则在创业板中得到动态多因子打分的股票num支，如果打分所得的股票与持有的股票有60%的不同则换仓；
其中每3个工作日判断大小盘风格轮动的状况，创业板每6个工作日进行一次换仓判断，沪深300每9个工作日进行一次换仓判断。
下面是策略的具体的实现：

'''
#from lib.guoxin_factor import *
start = '20110401'                       # 回测起始时间
end = '20180627'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = get_IdxCons(end, ticker ='000001') # 证券池，支持股票和基金
capital_base = 1000000                      # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate =3  


buylist=dy_stockchoice(index=universe,factorname1='REC',factorname2='ROE',factorname3='CFO2EV',factorname4='PE',factorname5='ILLIQUIDITY'
                    ,conf1=1,conf2=1,conf3=1,conf4=0,conf5=1,currday=start,periodstr=-20,stocknum=25
                    ,menTH=0.05)
print('*'*50)
print(buylist)
                                   # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟
def handle_data(account):                  # 每个交易日的买入卖出指令
    N1=1
    N2=0
    current=account.current_date.strftime("%Y-%m-%d")
    yesterday_use=account.previous_date.strftime("%Y%m%d")
    close_price = account.reference_price 
    yesterdaystr=cal.advanceDate(current,Period('-1B')).strftime("%Y-%m-%d") # 获取执行日前一个工作日的日期
    yesterdaystr_use=cal.advanceDate(current,Period('-1B')).strftime("%Y%m%d") 
    yesterdaystr20=cal.advanceDate(current,Period('-21B')).strftime("%Y-%m-%d")  # 获取执行日前二十一个工作日的日期
    HS_return=DataAPI.MktIdxdGet(indexID='000300.ZICN',beginDate=cal.advanceDate(current,Period('-21B')),
    endDate=yesterdaystr,field=['tradeDate','closeIndex'],pandas="1")
    HS_return=HS_return.set_index('tradeDate')
    CYB_return=DataAPI.MktIdxdGet(indexID='399006.ZICN',beginDate=cal.advanceDate(current,Period('-21B')),
                                  endDate=yesterdaystr,field=['tradeDate','preCloseIndex','closeIndex'],pandas="1")
    CYB_return=CYB_return.set_index('tradeDate') 
    HS_return_use=HS_return['closeIndex'][yesterdaystr]/HS_return['closeIndex'][yesterdaystr20]-1 # 计算HS300的20日涨幅
    CYB_return_use=CYB_return['closeIndex'][yesterdaystr]/CYB_return['closeIndex'][yesterdaystr20]-1 # 计算创业板的20日涨幅
    if HS_return_use<0 and CYB_return_use<0: #HS300和创业板20日涨幅小于0则平仓
        account.init=1
        account.count=0
        for s in account.security_position:
              order_pct_to(s,0)
    elif HS_return_use>CYB_return_use: #HS300的20日涨幅大于创业板则判断是否要买入沪深300中筛选出的股票
        if(account.count==0):
            curDate=account.previous_date.strftime("%Y-%m-%d")
            buylist=dy_stockchoice(index=DynamicUniverse('HS300').preview(current),factorname1='REC'
                    ,factorname2='ROE',factorname3='CFO2EV',factorname4='PE',factorname5='ILLIQUIDITY'
                    ,conf1=1,conf2=1,conf3=1,conf4=0,conf5=1,currday=curDate,periodstr='-20B',stocknum=25
                    ,menTH=0.05)
            cnt=0
            cnt_total=25
            for s in account.security_position:
                if s not in buylist:
                    cnt=cnt+1
            if(cnt>cnt_total*0.6 or account.init==1):
                account.init=0
                for s in account.security_position:
                    if s not in buylist:
                        order_pct_to(s,0)
                for s in buylist:
                    order_pct_to(s,1.0/(len(buylist)))
        if(account.count>N1):
            account.count=0
        else:
            account.count=account.count+1            
    else: #HS300的20日涨幅小于创业板则判断是否要买入创业板中筛选出的股票
        if(account.count==0):
            curDate=account.previous_date.strftime("%Y-%m-%d")
            buylist=dy_stockchoice(index=DynamicUniverse('CYB').preview(current),factorname1='REC'
                    ,factorname2='CFO2EV',factorname3='LFLO',factorname4='GREC'
                    ,factorname5='ILLIQUIDITY',conf1=1,conf2=1,conf3=0,conf4=1,conf5=0,currday=curDate
                    ,periodstr='-20B',stocknum=25,menTH=0.05)
            cnt=0
            cnt_total=25
            for s in account.security_position:
                if s not in buylist:
                    cnt=cnt+1
            if(cnt>cnt_total*0.6 or account.init==1):
                account.init=0
                for s in account.security_position:
                    if s not in buylist:
                        order_pct_to(s,0)
                for s in buylist:
                    order_pct_to(s,1.0/(len(buylist))) 
        if(account.count>N2):
            account.count=0
        else:
            account.count=account.count+1   
    return

'''
由回测的情况可知在2011-04-01到2017-03-27的时间段中本策略取得了46.10%的年化收益率,Alpha为43.50%，夏普比例为1.69，收益波动为24.9%，最大回撤17.7%！，可以看到策略的性能很好。
同时对于2011-04-01到2012-12-31这段股市下跌时期，本策略得了4.4%的年化收益率（高出基准17.7%）,收益波动为18.40%，最大回撤17.7%!,这说明该策略在长期熊市下有效！读者可以自己测试。
'''