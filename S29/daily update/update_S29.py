# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:45:51 2020

@author: adair2019
"""

"""
S29涉及所有因子 有一个需要转换后使用
更新策略 每个交易日更新上个季度的数据，先删除，后将新的数据添加
策略频率 月度
基本因子指标数据更新频率 季度 一年3次（一季度和上年四季度重合）
程序需要在4.30 更新第一季度、上年四季度数据；8.30日更新二季度；10.31更新三季度，其余时间不更新
逐个核查数据，有
5
7
9
无法核查到，需要转换    

#factor1  2007
x=DataAPI.FdmtIndiPSGet(ticker=u"000001",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"",field=u"secID,endDate,EPS",pandas="1")
#factor2  2008
x=DataAPI.FdmtIndiRtnGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"",field=u"secID,endDate,ROEA",pandas="1")
#factor3
x = DataAPI.FdmtIndiRtnGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"",field=u"secID,endDate,ROA",pandas="1")
#factor 4 200712
x= DataAPI.FdmtIndiRtnGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"A",field=u"secID,endDate,grossMARgin",pandas="1")
#成长能力
#factor 5 2008 #需要自己转换为同比增长率
x=DataAPI.FdmtISGet(ticker=u"000001",secID=u"",reportType=u"",endDate=u"",beginDate=u"",publishDateEnd=u"",
                  publishDateBegin=u"",endDateRep="",beginDateRep="",beginYear="",endYear="",fiscalPeriod="12",field=u"secID,endDate,TProfit",pandas="1")
#factor 6 经营活动产生的现金流量净额(同比增长率) 财务指标-成长能力
x=DataAPI.FdmtIndiGrowthGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"A",field=u"secID,endDate,nCfOpaYOY",pandas="1")
#factor 7 净资产(同比增长率)   -yq 归属于母公司的股东权益相对年初增长(%)
x=DataAPI.FdmtIndiGrowthGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"A",field=u"secID,endDate,teAttrPYTD",pandas="1")
#factor 8 
x=DataAPI.FdmtIndiGrowthGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"A",field=u"secID,endDate,niYOY",pandas="1")
#factor 9 
x=DataAPI.FdmtIndiGrowthGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"A",field=u"secID,endDate,taYTD",pandas="1")
#安全性指标 - 偿还债能力因子
#factor 10  资产负债率的倒数  asseTLiabRatio
x=DataAPI.FdmtIndiLqdGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"A",field=u"secID,endDate,asseTLiabRatio",pandas="1")
#factor 11
x=DataAPI.FdmtIndiLqdGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"A",field=u"secID,endDate,currenTRatio",pandas="1")
#factor 12
x=DataAPI.FdmtIndiLqdGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"A",field=u"secID,endDate,quickRatio",pandas="1")
#factor 13
x=DataAPI.FdmtIndiLqdGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"A",field=u"secID,endDate,nCFOpaCL",pandas="1")
#factor 14
x=DataAPI.FdmtIndiLqdGet(ticker=u"000002",secID="",endDate="",beginDate="",beginYear=u"",endYear=u"",reportType=u"A",field=u"secID,endDate,nCFOpaID",pandas="1")
#factor 15
x=DataAPI.RbEquDivGet(ticker=u"000001",publishDate=u"",beginDate=u"",endDate=u"",divDate=u"",cashDate=u"",field=u"secID,endDate,profitPnt",pandas="1")
"""

#tools
import zipfile
import pandas as pd
import time
import datetime
import numpy as np 

tt=time.strftime("%Y%m%d", time.localtime())
t0 = '20200301'
#tt = '20100101'

def save_data_adair(fn_d1,x):
    fn1_d1 = '%s.csv' % fn_d1
    fn2_d1 = '%s.zip' % fn_d1
    x.to_csv(fn1_d1,index=False)
    z=zipfile.ZipFile(fn2_d1,'w',zipfile.ZIP_DEFLATED)
    z.write(fn1_d1)
    z.close()
    toolkits.delete_files([fn1_d1])
def compress_all_zip_data():
    fns = list_files()
    z=zipfile.ZipFile('ad_test.zip','w',zipfile.ZIP_DEFLATED)
    for fn1_d1 in fns:
        z.write(fn1_d1)
    z.close()
    toolkits.delete_files(fns)
    
def get_symbol_adair():
    x=DataAPI.EquGet(secID=u"",ticker=u"",equTypeCD=u"A",listStatusCD=u"",field=u"",pandas="1")
    y=x['ticker']
    z=''
    z1=[]
    k=0;
    for i in y:
        if len(i)==6:
            if i[0]=='3' or i[0]=='0' or i[0:2]=='60':
                z1.append(i)
                if k==0:
                    z=i
                else:
                    z=z+','+i
                k = k +1
    return z1,z

z1,z=get_symbol_adair()



#确定需要更新的发布日期，在当前时间点，获取证交所规定的上个会计期间截止日  4.30 8.30 10.31
def get_pub_date_S29(t0):
    out_f = '%Y%m%d'
    y = t0.tm_year
    t1 = time.strptime(('%d0430' % y),out_f)
    t2 = time.strptime(('%d0830' % y),out_f)
    t3 = time.strptime(('%d1031' % y),out_f)
    
    if t0>=t3:
        t ='%d0930' % y
    elif t0>=t2:
        t ='%d0630' % y
    elif t0>=t1:
        t =['%d0331' % y ,'%d1231' % (y-1)]
    elif t0<t1:
        t ='%d0930' % (y-1)
    else:
        t=None
    return t


t0=time.localtime()
#pub_date=get_pub_date_S29(t0)
pub_date=""
def get_S29_f1(pub_date):
    x=DataAPI.FdmtIndiPSGet(ticker=z1,secID="",endDate=pub_date,beginDate=pub_date,beginYear=u"",
                            endYear=u"",reportType=u"",field=u"ticker,endDate,EPS",pandas="1")
    x.rename(columns={'EPS':'f1'},inplace=True)
    x.dropna(inplace=True)
    x.drop_duplicates(['ticker','endDate'],inplace=True)
    print('%d -14 complete' % 1)
    return x
def get_S29_f2_4(pub_date):
    x=DataAPI.FdmtIndiRtnGet(ticker=z1,secID="",endDate=pub_date,beginDate=pub_date,beginYear=u"",
                             endYear=u"",reportType=u"",field=u"ticker,endDate,ROEA,ROA,grossMARgin",pandas="1")
    x.rename(columns={'ROEA':'f2','ROA':'f3','grossMARgin':'f4'},inplace=True)
    x.drop_duplicates(['ticker','endDate'],inplace=True)
    print('%d -14 complete' % 4)
    return x
#需要后续转换
def get_S29_f5(pub_date):
    x=DataAPI.FdmtISGet(ticker=z1,secID=u"",reportType=u"Q1,S1,CQ3,A",endDate=pub_date,beginDate=pub_date,publishDateEnd=u"",
                  publishDateBegin=u"",endDateRep="",beginDateRep="",beginYear="",endYear="",fiscalPeriod="",
                    field=u"ticker,endDate,TProfit",pandas="1")
    x.rename(columns={'TProfit':'f5'},inplace=True)
    x.drop_duplicates(['ticker','endDate'],inplace=True)
    print('%d -14 complete' % 5)
    return x
def get_S29_f6_8(pub_date):
    x=DataAPI.FdmtIndiGrowthGet(ticker=z1,secID="",endDate=pub_date,beginDate=pub_date,
                            beginYear=u"",endYear=u"",reportType=u"",field=u"ticker,endDate,nCfOpaYOY,niYOY",pandas="1")
    x.rename(columns={'nCfOpaYOY':'f6','niYOY':'f8'},inplace=True)
    x.drop_duplicates(['ticker','endDate'],inplace=True)
    print('%d -14 complete' % 6)
    return x
#需要转换
#合并资产负债表
def get_S29_f7_9(pub_date):
    x=DataAPI.FdmtBSGet(ticker=z1,secID=u"",reportType=u"Q1,S1,Q3,A",endDate=u"",beginDate=u"",publishDateEnd=u"",
                    publishDateBegin=u"",endDateRep="",beginDateRep="",beginYear="",endYear="",fiscalPeriod="",
                        field=u"ticker,endDate,TEquityAttrP,TAssets",pandas="1")
    x.rename(columns={'TEquityAttrP':'f7','TAssets':'f9'},inplace=True)
    x.drop_duplicates(['ticker','endDate'],inplace=True)
    print('%d -14 complete' % 7)
    return x
#f14定义完全相同，但是数值有差异
def get_S29_f10_14(pub_date):
    x=DataAPI.FdmtIndiLqdGet(ticker=z1,secID="",endDate=pub_date,beginDate=pub_date,beginYear=u"",endYear=u"",
                         reportType=u"",field=u"ticker,endDate,asseTLiabRatio,currenTRatio,quickRatio,nCFOpaCL,nCFOpaID",pandas="1")  
    x.rename(columns={'asseTLiabRatio':'f10','currenTRatio':'f11','quickRatio':'f12','nCFOpaCL':'f13','nCFOpaID':'f14'},inplace=True)
    x.drop_duplicates(['ticker','endDate'],inplace=True)
    print('%d -14 complete' % 10)
    return x
def get_S29_f15(pub_date):
    x=DataAPI.RbEquDivGet(ticker=z1,publishDate=u"",beginDate=u"",endDate=u"",divDate=u"",cashDate=u"",field=u"ticker,endDate,profitPnt",pandas="1")
    x.dropna(inplace=True)
    x.rename(columns={'profitPnt':'f15'},inplace=True)
    x.drop_duplicates(['ticker','endDate'],inplace=True)
    print('%d -14 complete' % 14)
    return x


#清空
import toolkits
toolkits.delete_files(list_files())

#getdata and save data
f1 = get_S29_f1(pub_date)
f2 = get_S29_f2_4(pub_date)
f5= get_S29_f5(pub_date)
f6=get_S29_f6_8(pub_date)
f7=get_S29_f7_9(pub_date)
f10=get_S29_f10_14(pub_date)
f15=get_S29_f15(pub_date)

save_data_adair('S29F01',f1)
save_data_adair('S29F02',f2)
save_data_adair('S29F05',f5)
save_data_adair('S29F06',f6)
save_data_adair('S29F07',f7)
save_data_adair('S29F10',f10)
save_data_adair('S29F15',f15)

compress_all_zip_data()
