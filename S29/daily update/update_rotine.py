# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:44:43 2020

@author: adair2019
"""

#清空 csv
import toolkits
toolkits.delete_files(list_files())


#routine数据 借鉴S24项目的数据

#1获取所有可转债日度数据
#获取symbol的list
import zipfile
import pandas as pd
import time
import datetime
import numpy as np 

tt=time.strftime("%Y%m%d", time.localtime())
t0 = '20200315'
#tt = '20100101'

def save_data_adair(fn_d1,x):
    fn1_d1 = '%s.csv' % fn_d1
    fn2_d1 = '%s.zip' % fn_d1
    x.to_csv(fn1_d1,index=False)
    z=zipfile.ZipFile(fn2_d1,'w',zipfile.ZIP_DEFLATED)
    z.write(fn1_d1)
    z.close()
    toolkits.delete_files([fn1_d1])
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

def get_tradingdate_adair(tt):
    x=DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=u"20000101",endDate=tt,field=u"calendarDate,isOpen",pandas="1")
    t=x.calendarDate[x.isOpen==1].values
    return t

def compress_all_zip_data():
    fns = list_files()
    z=zipfile.ZipFile('ad_test.zip','w',zipfile.ZIP_DEFLATED)
    for fn1_d1 in fns:
        z.write(fn1_d1)
    z.close()
    toolkits.delete_files(fns)

def get_ticker_data():
    #x.loc[x.tradeDate>='2019-12-01']
    #2 股票日行情
    #股票日行情 半年
    fn_d2 = 'tickerday_data%s' % tt
    fn1_d2 = '%s.csv' % fn_d2
    fn2_d2=  '%s.zip' % fn_d2
    if fn2_d2 not in list_files():
        z1,z=get_symbol_adair()
        x=DataAPI.MktEqudGet(secID=u"",ticker=z1,tradeDate=u"",beginDate=t0,endDate=tt,isOpen=1,field=u"",pandas="1") 
        x.drop(['secID','secShortName','exchangeCD','vwap','isOpen'],axis=1, inplace=True)

        save_data_adair(fn_d2,x)
        print('正股日数据已经更新到%s' % fn2_d2) 
    else:
        print('%s已经存在，正股日数据已经更新，未执行' % fn2_d2) 
    #3 指数日行情
    t= get_tradingdate_adair(tt)
    fn_d3 = 'indicator_data%s' % tt
    fn1_d3 = '%s.csv' % fn_d3
    fn2_d3=  '%s.zip' % fn_d3
    t0_f1 = '%s-%s-%s' % (t0[0:4],t0[4:6],t0[6:])
    tt_f1 = '%s-%s-%s' % (tt[0:4],tt[4:6],tt[6:])
    t=t[t>=t0_f1]
    t=t[t<=tt_f1]
    if fn2_d3 not in list_files():
        i=0
        for sub_t in t:
            x=DataAPI.MktIdxdGet(indexID=u"",ticker=u"",tradeDate=sub_t,beginDate=u"",
                                 endDate=u"",exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")
            if i==0:
                re=x
            else:
                re = re.append(x)
            i = i +1
            if i%3 ==0:
                print('%d-%s' % (i,sub_t))
        test_v = 0;
        if len(t)>0:
            if len(re)>0:
                save_data_adair(fn_d3,re)
                test_v = test_v+1
        if test_v>0:
            print('指数日数据已经更新到%s' % fn2_d3) 
        else:
            print('指数日数据没有更新')
    else:
        print('%s已经存在，指数日数据已经更新，未执行' % fn2_d3) 

    #4交易日期更新
    fn_d4= 'tradingdate%s' % tt
    fn1_d4 = '%s.csv' % fn_d4
    fn2_d4 = '%s.zip' % fn_d4

    if fn2_d4 not in list_files():
        tt2 = (datetime.datetime.now()+datetime.timedelta(days=5)).strftime("%Y-%m-%d")
        x = get_tradingdate_adair(tt2)
        x = pd.DataFrame({'tradingdate':x})
        save_data_adair(fn_d4,x)
        print('交易日数据已经更新到%s' % fn2_d2) 
    else:
        print('%s已经存在，交易日数据已经更新，未执行' % fn2_d2) 

#ST标记
def update_st_data():   
    fn_d2 = 'st_data%s' % tt
    fn2_d2=  '%s.zip' % fn_d2
    if fn2_d2 not in list_files():
        x=DataAPI.SecSTGet(beginDate=t0,endDate=tt,secID=u"",ticker=z1,field=['ticker','tradeDate','STflg'],pandas="1")
        save_data_adair(fn_d2,x)
        print('st交易日数据已经更新到%s' % fn_d2)
    else:
        print('%s已经存在，交易日数据已经更新，未执行' % fn2_d2)    

#后复权月度行情
def get_month_data():    
    fn_d2 = 'MktEqumAdjAfGet%s' % tt
    fn2_d2=  '%s.zip' % fn_d2
    if fn2_d2 not in list_files():
        x=DataAPI.MktEqumAdjAfGet(secID=u"",ticker=z1,monthEndDate=u"",beginDate=t0,
                                  endDate=tt,isOpen=u"",field=u"",pandas="1")
        save_data_adair(fn_d2,x)
        print('后复权月度行情已经更新到%s' % fn_d2)
    else:
        print('%s已经存在，交易日数据已经更新，未执行' % fn2_d2)   
#股票基本信息
def get_symbol_basic_info():     
    fn_d2 = 'EquGet%s' % tt
    fn2_d2=  '%s.zip' % fn_d2
    if fn2_d2 not in list_files():
        x = DataAPI.EquGet(secID=u"",ticker=u"",equTypeCD=u"A",listStatusCD=u"",
                   exchangeCD="",ListSectorCD=u"",field=u"",pandas="1")   
        save_data_adair(fn_d2,x)
        print('股票基本信息已经更新到%s' % fn_d2)
    else:
        print('%s已经存在，股票基本信息已经更新，未执行' % fn2_d2) 

#申万获取行业
def get_industry_data_adair():    
    fn_d2 = 'EquIndustryGet%s' % tt
    fn2_d2=  '%s.zip' % fn_d2
    if fn2_d2 not in list_files():
        x =DataAPI.EquIndustryGet(secID=u"",ticker=u"",
                              industryVersionCD=u"010303",industry=u"",industryID=u"",industryID1=u"",industryID2=u"",
                              industryID3=u"",intoDate=u"",equTypeID=u"",field=u"",pandas="1")  
        save_data_adair(fn_d2,x)
        print('申万获取行业信息已经更新到%s' % fn_d2)
    else:
        print('%s已经存在，申万获取行业信息已经更新，未执行' % fn2_d2) 
#获取后复权因子
#DataAPI.MktEqudAdjAfGet(secID=u"",ticker=u"688001",tradeDate=u"",beginDate=u"20190801",endDate=u"20190805",isOpen="",field=u"",pandas="1")
def MktEqudAdjAfGet():
    fn_d2 = 'MktEqudAdjAfGet_data%s' % tt
    fn1_d2 = '%s.csv' % fn_d2
    fn2_d2=  '%s.zip' % fn_d2
    if fn2_d2 not in list_files():
        z1,z=get_symbol_adair()
        t= get_tradingdate_adair(tt)
        #t0_f1 = t0
        #tt_f1 = tt
        t0_f1 = '%s-%s-%s' % (t0[0:4],t0[4:6],t0[6:])
        tt_f1 = '%s-%s-%s' % (tt[0:4],tt[4:6],tt[6:])
        t=t[t>=t0_f1]
        t=t[t<=tt_f1]
        i = 0
        z_fn=zipfile.ZipFile(fn2_d2,'w',zipfile.ZIP_DEFLATED)
        for sub_t in t:
            sub_x=DataAPI.MktEqudAdjAfGet(secID=u"",ticker=z1,tradeDate=sub_t,beginDate=u"",endDate=u"",
                                      isOpen=1,field=u"ticker,tradeDate,accumAdjFactor",pandas="1") 
            sub_fn = 'MktEqudAdjAfGet_data%s.csv' % sub_t
            sub_x.to_csv(sub_fn,index=False)
            z_fn.write(sub_fn)
            toolkits.delete_files([sub_fn])
            #if i==0:
            #    x = sub_x
            #else:
            #    x = x.append(x)
            i = i +1
            if np.mod(i,20)==0:
                print(sub_t)
        z_fn.close()
        
        #save_data_adair(fn_d2,x)
        #return x
        print('后复权因子数据已经更新到%s' % fn2_d2) 
    else:
        #return None
        print('%s已经存在，后复权因子数据已经更新，未执行' % fn2_d2) 

#DataAPI.IdxCloseWeightGet(secID=u"",ticker=u"000300",beginDate=u"20151101",endDate=u"20151130",field=u"",pandas="1")  指数成分股数据
"""
'000001','000002','000003','000004','000005','000006','000007','000008','000009','000010','000011','000012',
'000013','000015','000016','000020','000090','000132','000133','000300','000852','000902','000903','000904',
'000905','000906','000907','000922','399001','399002','399004','399005','399006','399007','399008','399009',
'399010','399011','399012','399013','399015','399107','399108','399301','399302','399306','399307','399324',
'399330','399333','399400','399401','399649','DY1101','DY1102','DY1201','DY1202','DY1203','DY1204','DY1205','DY1206','DY1207','DY1208','DY1301','DY1302','DY1303','DY1304','DY1305','DY1306','DY1307','DY2201','DY2202','DY2203','DY2204'

"""
def IdxCloseWeightGet_adair(): 
    fn_d2 = 'IdxCloseWeightGet%s' % tt
    fn1_d2 = '%s.csv' % fn_d2
    fn2_d2=  '%s.zip' % fn_d2
    tickercode = ['000001','000002','000003','000004','000005','000006','000007','000008','000009','000010','000011','000012','000013','000015',
                  '000016','000020','000090','000132','000133','000300','000852','000902','000903','000904','000905','000906','000907','000922',
                  '399001','399002','399004','399005','399006','399007','399008','399009','399010','399011','399012','399013','399015','399107',
                  '399108','399301','399302','399306','399307','399324','399330','399333','399400','399401','399649']
    z_fn=zipfile.ZipFile(fn2_d2,'w',zipfile.ZIP_DEFLATED)
    for i,sub_ticker in enumerate(tickercode):
        sub_x=DataAPI.IdxCloseWeightGet(secID=u"",ticker=sub_ticker,beginDate=t0,endDate=tt,field=u"effDate,ticker,consTickerSymbol,weight",pandas="1")
        sub_fn = 'IdxCloseWeightGet%s.csv' % sub_ticker
        sub_x.to_csv(sub_fn,index=False)
        z_fn.write(sub_fn)
        toolkits.delete_files([sub_fn])
        print('%d-%s' % (i,sub_ticker))
    z_fn.close()

#指数月度行情
def MktIdxmGet_adair():
    fn_d2 = 'MktIdxmGet%s' % tt
    x=DataAPI.MktIdxmGet(beginDate=t0,endDate=tt,indexID=u"",ticker=u"",field=u"",pandas="1")
    save_data_adair(fn_d2,x)



get_ticker_data()
update_st_data()
get_month_data()
get_symbol_basic_info()
get_industry_data_adair()
x=MktEqudAdjAfGet()
IdxCloseWeightGet_adair()
MktIdxmGet_adair()