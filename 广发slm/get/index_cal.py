# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:53:11 2018

@author: caoqiliang
"""
#此程序用于计算公司组合产品的各个指标，以及图表。并计算各个指标的值再输出到excel
import pandas as pd
import numpy as np
from datetime import datetime
#下述表达式会报错：initializing  from file failed,一般是因为文件有中文名的原因
#data_ruitian=pd.read_table('沣谊工作文件夹\\锐天净值更新20180803.xls')
#改正如下：
''''f=open('沣谊投资\\data_huizong.csv','rb')  注意这里rb一定要有
a1=pd.read_excel(f)'''
#以下data均为pandas类型的基金净值数据，其中index为datetimeIndex,频率为日频
def ret_corr_Daily(data):
    #计算日收益率及相关性
    ret=data/data.shift(1)-1
    corr=ret.corr()
    return ret,corr

def ret_corr_Weekly(data):
    #计算每周的收益率及相关性
    data_weekly=data.asfreq('w',method='pad')
    ret,corr=ret_corr_Daily(data_weekly)
    return ret,corr

def ret_sigma_year(data):
    #计算几何平均年化收益率和年化收益率方差
    #年华收益率和方差的计算我们都利用周收益率进行计算
    ret_weekly,_=ret_corr_Weekly(data)
    std=ret_weekly.std()
    std_year=52**0.5*std
    ret_cumprod=(1+ret_weekly).cumprod()
    #ret_cumprod=ret_cumprod[pd.notnull(ret_cumprod)]
    
    week_times=np.array([len(ret_cumprod.iloc[:,i][pd.notnull(ret_cumprod.iloc[:,i])]) for i in range(len(ret_cumprod.columns))])
    ret_finally=[]
    for i in range(len(data.columns)):
        ret1=ret_cumprod.iloc[:,i][pd.notnull(ret_cumprod.iloc[:,i])]
        #得到最终的累计收益率
        ret_finally.append(ret1[-1])
    ret_finally=np.array(ret_finally)**(52/week_times)-1
    ret_year=pd.Series(ret_finally,index=data.columns)
    return ret_year,std_year

'''def ret_netvalue_this_year(data):
    #此程序用于返回今年收益和最终净值
    year_now=str(datetime.now().year)
    #将数据截断为从今年开始
    data_this_year=data.loc[year_now:,:]
    ret_this_year=[]
    NetValue=[]
    for i in range(len(data.columns)):
        data_need=data_this_year.iloc[:,i][pd.notnull(data_this_year.iloc[:,i])]
        NetValue.append(data_need[-1])
        ret1=data_need[-1]/data_need[0]-1
        ret_this_year.append(ret1)
    return pd.Series(ret_this_year,index=data.columns),pd.Series(NetValue,index=data.columns)'''

def max_draw_down(data):
    #此程序用于计算其最大回撤
    draw_down=[]
    for i in range(len(data.columns)):
        max_draw_down=0
        for j in range(len(data.iloc[:,i])):
            max_draw_down=min(max_draw_down,data.iloc[:,i][j]/data.iloc[:,i][0:j].max()-1)
        draw_down.append(-max_draw_down)
    return pd.Series(draw_down,index=data.columns)

def xiapu_ratio(data,risk_free_rate=0):
    #此程序用于计算年度夏普比率
    ret_year,std_year=ret_sigma_year(data)
    return (ret_year-risk_free_rate)/std_year

def sortino_ratio(data,risk_free_rate=0,benchmark=0):
    #此程序用于计算年度索提诺比例，benchmark表示所选定的阈值
    ret_year,_=ret_sigma_year(data)
    ret_weekly,_=ret_corr_Weekly(data)
    mean=ret_weekly.mean()
    suotinuo_ratio=[]
    for i in range(len(ret_weekly.columns)):
        #在此处，我们认为下方差也具有可加性，因此可以如之前一样计算
        data_need=ret_weekly.iloc[:,i][ret_weekly.iloc[:,i]<benchmark]
        down_std=((data_need-mean[i])**2).mean()**0.5*(52**0.5)
        suotinuo_ratio.append((ret_year[i]-risk_free_rate)/down_std)
    return pd.Series(suotinuo_ratio,index=data.columns)

def kama_ratio(data,risk_free_rate=0):
    #卡马比例
    ret_year,_=ret_sigma_year(data)
    draw_down=max_draw_down(data)
    kama_ratio=(ret_year-risk_free_rate)/draw_down
    return kama_ratio
    
    

def win_ratio(data,freq='d'):
    if freq=='d':
        ret_daily,_=ret_corr_Daily(data)
        bigger_than0=ret_daily[ret_daily>0].count()
        less_than0=ret_daily[ret_daily<0].count()
        count=ret_daily.count()
        win_rate=bigger_than0/(less_than0+bigger_than0)
    if freq=='w':
        ret_weekly,_=ret_corr_Weekly(data)
        bigger_than0=ret_weekly[ret_weekly>0].count()
        less_than0=ret_weekly[ret_weekly<0].count()
        count=ret_weekly.count()
        win_rate=bigger_than0/(less_than0+bigger_than0)
    return win_rate

def profit_loss_ratio(data,freq='d'):
    #计算盈亏比，即盈利的平均幅度除以亏损的平均幅度
    if freq=='d':
        ret_daily,_=ret_corr_Daily(data)
        bigger_than0=ret_daily[ret_daily>0].mean()
        smaller_than0=-ret_daily[ret_daily<0].mean()
        profit_loss_ratio=bigger_than0/smaller_than0
    if freq=='w':
        ret_weekly,_=ret_corr_Weekly(data)
        bigger_than0=ret_weekly[ret_weekly>0].mean()
        smaller_than0=-ret_weekly[ret_weekly<0].mean()
        profit_loss_ratio=bigger_than0/smaller_than0
    return profit_loss_ratio

def time_range(data):
    #此程序用于计算每个数据的周期跨度
    time=[]
    for i in range(len(data.columns)):
        data_need=data.iloc[:,i][pd.notnull(data.iloc[:,i])]
        start=str(data_need.index[0].date())
        end=str(data_need.index[-1].date())
        time.append(start+'-->'+end)
    return pd.Series(time,data.columns)
        
#最低净值，统计周期数，时间范围并未用函数计算，因为可以直接算
def index_cal_output(data,risk_free_rate=0,benchmark=0):
    #此程序计算所有指标并输出
    ret_year,std_year=ret_sigma_year(data)
    '''ret_this_year,netvalue_final=ret_netvalue_this_year(data)'''
    min_netvalue=data.min()
    xiapu=xiapu_ratio(data,risk_free_rate)
    sortino=sortino_ratio(data,risk_free_rate,benchmark)
    kama=kama_ratio(data,risk_free_rate)
    win_ratio_daily=win_ratio(data,'d')
    win_ratio_weekly=win_ratio(data,'w')
    profit_loss_daily=profit_loss_ratio(data,freq='d')
    profit_loss_weekly=profit_loss_ratio(data,freq='w')
    time_count=data.count()
    timeRange=time_range(data)
    index=['年化收益率','年化标准差','最低净值','夏普比率',\
           '索提诺比率','卡玛比率','日胜率','周胜率','日盈亏比','周盈亏比','统计周期数','时间跨度']
    index_all=pd.DataFrame([ret_year,std_year,min_netvalue,xiapu,\
                            sortino,kama,win_ratio_daily,win_ratio_weekly,profit_loss_daily,\
                            profit_loss_weekly,time_count,timeRange],index=index)
    index_all.T.to_excel('index_all.xls') #输出到表
    
    
    
    
    
    
    
    
    
    
    



        
        

        
        

        
        
        
    
    
    
            
            
        
        
        
        

        
        
    
    
    
    
    
    
    
    
    


    
    
    


#NetValue_ruitian=pd.read_table('沣谊工作文件夹\\天演.xlsx',header=[1],index_col=0,parse_dates=True,encoding='gb2312')

#下面开始对各个资产收益率以及相关性矩阵进行计算，按照口径分为日相关性，月相关性











