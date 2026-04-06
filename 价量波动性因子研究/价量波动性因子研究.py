# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:57:52 2020

@author: Asus
"""
'''
A. 研究目的：本文利用价格数据和传统量价指标，参考海通证券-选股因子系列研究(五十五)：价量波动幅度-190924（原作者：冯佳睿）中的研究方法，对研报的结果进行了实证分析，用以探寻股价振幅因子的表现

B. 文章结构：本文共分为3个部分，具体如下

一、数据准备和处理

二、测试因子与传统量价因子的互补特性

三、测试因子表现

四、总结

C. 研究结论：

股价振幅因子本身并不具有较好的选股效应，但是在将传统量价因子中性化掉之后股份振幅因子与股票未来收益显著正相关；同时股价振幅因子和传统的量价因子在空头效应和多头效应上可以互补，在涨幅低、换手率低、波动率低的样本集中，传统技术因子失效，而股价振幅因子在此样本集中存在显著选股效果，且多头效应强

在传统量价因子等权构建的组合中加入股价振幅因子可以显著提高组合的表现， 其中IC从-5.92%提高到-6.64%；年化收益从8.48%提高到11.0%；信息比率从1.26提高到1.76；回测表现证实了股价振幅因子对传统量价因子的互补作用

D. 时间说明

本文主要分为四个部分，第一部分约耗时25分钟，其它部分耗时均在5分钟以内，总耗时在50分钟左右
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
链接：https://uqer.datayes.com/v3/community/share/9sM4tSmTlqlcS7KLKSfnADGEt880/private；密码：7137。请前往查看并注意保密。
请在运行之前，克隆上面的代码，并存成lib(右上角->另存为lib，不要修改名字)

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


第一部分：数据准备和处理
该部分耗时 45分钟
该部分内容为：

读取禁投股票池

读取从2010-2019的A股基础价量数据，并从中生成本文的股价振幅因子，同时从优矿api中读取一些传统的价量因子作为后续与股价振幅因子的对比研究。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


'''


import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from scipy import stats
#from CAL.PyCAL import *
mpl.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei'] 

data_start_date = '2009-10-01'
data_end_date = '2019-12-01'

# 获取全A的secID
all_secid_list = DataAPI.EquGet(equTypeCD=u"A", field=u"secID",pandas="1")['secID'].tolist()
all_secid_list = [x for x in all_secid_list if len(x) == 11]

# 获取交易日历
trade_calendar = DataAPI.TradeCalGet(exchangeCD=U"XSHG", field=u"calendarDate,isOpen,isMonthEnd")
date_list = trade_calendar.query("calendarDate>=@data_start_date").query("calendarDate<=@data_end_date").query("isOpen==1")['calendarDate'].tolist()
# 月末交易日
month_end_list = trade_calendar.query("calendarDate>=@data_start_date").query("calendarDate<=@data_end_date").query("isMonthEnd==1")['calendarDate'].tolist()



t1 = time.time()
print("计算禁止池，包括上市不足60个交易日的次新股、ST股以及停牌个股...")

# 获得交易日历
calendar = trade_calendar[trade_calendar['isOpen'] == 1]
calendar = calendar['calendarDate'].tolist()

# 次新股
ipo_info = DataAPI.SecIDGet(assetClass=u"E", field=['ticker', 'listDate'], pandas="1")
ipo_info.dropna(inplace=True)
ticker_list = [ticker for ticker in ipo_info['ticker'] if len(ticker) == 6 and ticker[0] in ['0', '3', '6']]
ipo_info = ipo_info[ipo_info['ticker'].isin(ticker_list)]
ipo_info['permit_date'] = [calendar[calendar.index(date) + 60] if date in calendar else  calendar[0] for date in ipo_info['listDate']]

calendar = np.array(calendar)
new_df = pd.DataFrame()
for date in calendar[(calendar > data_start_date) & (calendar < data_end_date)]:
    new_list = ipo_info[(ipo_info['permit_date'] >= date) & (ipo_info['listDate'] <= date)]['ticker'].values
    d_new_df = pd.DataFrame({'tradeDate': [date] * len(new_list), 'ticker': new_list})
    new_df = new_df.append(d_new_df)

new_df['remove_flag'] = 'new'

# ST股
st_info = DataAPI.SecSTGet(beginDate=data_start_date,endDate=data_end_date,field=['tradeDate', 'ticker'],pandas="1")
st_df = st_info.copy()
st_df['remove_flag'] = 'st'

# 停牌
halt_info = DataAPI.SecHaltGet(beginDate=data_start_date, endDate=data_end_date, field=['ticker', 'haltBeginTime', 'haltEndTime'],pandas="1")
halt_info.fillna(calendar[-1], inplace=True)
halt_info['haltBeginTime'] = halt_info['haltBeginTime'].apply(lambda x: x[:10])
halt_info['haltEndTime'] = halt_info['haltEndTime'].apply(lambda x: x[:10])

halt_df = pd.DataFrame()
for date in calendar[(calendar > data_start_date) & (calendar < data_end_date)]:
    halt_list = halt_info[(halt_info['haltEndTime'] >= date) & (halt_info['haltBeginTime'] <= date)]['ticker'].values
    d_halt_df = pd.DataFrame({'tradeDate': [date] * len(halt_list), 'ticker': halt_list})
    halt_df = halt_df.append(d_halt_df)

halt_df['remove_flag'] = 'halt'

remove_df = new_df.append(st_df).append(halt_df)
remove_df = remove_df[['tradeDate', 'ticker', 'remove_flag']]

t2 = time.time()
print ('记录的禁止股票池样式如下，耗时:%s seconds'%(t2 - t1))
print (remove_df.head().to_html())

'''

1.2 获取基础价量数据生成股价振幅因子和获取传统价量因子

获取从2009-2019年的所有A股最高价、最低价、收盘价、换手率、市值等基础价量数据。
计算生成股价振幅因子：每个月末，我们基于过去一段时间内，日度价格最高值与最低值之间的变动幅度构建股价振幅因子（下简称振幅因子）。即振幅因子=日度价格最高值/最低值-1。本篇文章中接下来的因子都采用3个月的振幅因子（观察期为3个月）作为研究。公式中日度价格最高值是过去三个月该股票的每日收盘价的最大值，日度价格最低值是过去三个月该股票的每日收盘价的最小值。
计算生成未来20天的股票收益，用于后续对股价振幅因子和传统价量因子互补性的研究。
计算生成股票波动率因子price_vol（股票过去20天每天涨跌幅的标准差）作为传统价量因子中的一个，用于后续对股价振幅因子和传统价量因子互补性的研究。
获取2009-2019年所有A股的市值因子和传统价量因子（反转、流动性、换手率），其中市值因子采用LFLO（流通市值的对数），反转因子采用REVS20（股票的20日收益），流动性因子采用ILLIQUIDITY（过去20个交易日收益相对金额的比例）， 换手率因子采用VOl20（20日平均换手率）。

'''

import matplotlib.pyplot as plt
import pandas as pd, numpy as np 

#创建origin文件夹
origin_path = 'pricevolumevol/origin_byticker'
if not os.path.exists(origin_path):
    os.makedirs(origin_path)

#设定日期
start_date, end_date = '2009-10-01', '2019-12-01'

#下载全市场A股从2009-2019年的每天的最高价、最低价、换手率和市值等原始数据
def download_origin_data():
    """
    原始数据下载完之后存储在pricevolumevol/origin_byticker文件夹中
    按照股票的ticker分别存储
    """
    all_atickers_df = DataAPI.EquGet(secID=u"",ticker=u"",equTypeCD=u"A",listStatusCD=u"",field=u"",pandas="1")
    all_atickers = sorted(list(all_atickers_df.ticker))  
    for ticker in all_atickers:
        tmp = DataAPI.MktEqudAdjGet(secID=u"",ticker=ticker,tradeDate=u"",beginDate=start_date,endDate=end_date,isOpen="",
                       field=u"ticker,tradeDate,closePrice,highestPrice,lowestPrice,turnoverRate,marketValue",pandas="1")
        tmp.to_csv("{}/{}_origin.csv".format(origin_path, ticker), index=None)

#从原始数据中生成股价振幅因子，以及换手率、涨幅、波动率等指标，以进行进一步的分析        
def handle_origin_data():
    """
    处理完的因子数据存储在pricevolumevol/handle_byticker文件夹中
    按照股票的ticker分别存储
    """    
    calendar_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
    month_end_list = sorted(list(calendar_df[(calendar_df['isMonthEnd']==1)&(calendar_df['isOpen']==1)]['calendarDate'].values))
    trade_date_list = sorted(list(calendar_df[calendar_df['isOpen']==1]['calendarDate'].values))
    origin_path = 'pricevolumevol/origin_byticker'
    #创建handle文件夹用于存储
    handle_path = 'pricevolumevol/handle_byticker'
    if not os.path.exists(handle_path):
        os.makedirs(handle_path)
    all_files = sorted(os.listdir(origin_path))
    #每一个个股分别处理，每一个file一个ticker
    for file in all_files:
        ticker = file[:6]
        #不是A股的过滤掉        
        if file[:2] not in ['00', '30', '60']:
            continue
        #获取上市日期，用于后续代码剔除上市未满一年的个股
        ipo_date_df = DataAPI.EquIPOGet(secID=u"",ticker=u"{}".format(ticker),eventProcessCD=u"",beginDate=u"",endDate=u"",field=u"",pandas="1")
        #上市日期异常处理
        if len(ipo_date_df)>=1:
            ipo_date = ipo_date_df.sort_values(by='listDate').listDate.iloc[0]
            if type(ipo_date)==str:
                ipo_date = ipo_date
            else:
                ipo_date = '2020-01-01'
        else:
            ipo_date = '2020-01-01'
        #读取该个股的原始数据
        tmp = pd.read_csv('{}/{}'.format(origin_path, file))
        if len(tmp)==0:
            continue
        tmp = tmp.sort_values(by='tradeDate')
        #去掉价格中的0,因为有可能会被误当成最小值
        tmp = tmp[tmp['highestPrice']!=0]
        tmp = tmp[tmp['closePrice']!=0]
        tmp = tmp[tmp['lowestPrice']!=0]
        
        #rolling计算后取月末收据
        tmp = tmp.sort_values(by='tradeDate')
        price_amplitude_se = pd.rolling_apply(arg=tmp[['closePrice']], func=lambda x: x.max()/x.min()-1, window=60, min_periods=60)
        price_amplitude_se.columns = ['price_amplitude']
        tmp['pct'] = tmp['closePrice'].pct_change()
        price_vol_se = pd.rolling_apply(arg=tmp[['pct']], func=lambda x: x.std(), window=20, min_periods=20)
        price_vol_se.columns = ['price_vol']
        #未来一个月收益
        nextone_ret_se = tmp['closePrice'].pct_change(20).shift(-20)
        nextone_ret_se.name = 'nextone_ret'
        total_df = pd.concat([tmp, price_amplitude_se, price_vol_se, nextone_ret_se], axis=1)
        total_df = total_df[['ticker', 'tradeDate', 'price_amplitude', 'price_vol', 'nextone_ret']]
        total_df['ticker'] = total_df['ticker'].apply(lambda x: str(x).zfill(6))
        #取月末数据
        total_df = total_df.dropna()
        #空dataframe略过
        if len(total_df)==0: continue
        total_df = total_df[total_df['tradeDate'].apply(lambda x: x in month_end_list)]
        total_df['tradeDate'] = total_df['tradeDate'].apply(lambda x: str(x).replace('-', ''))
        total_df.to_csv("{}/{}_handle.csv".format(handle_path, ticker), index=None)
        
        
        
#下载传统的量价因子
def download_techfactors():
    """
    获得每个月末传统量价因子的数据
    all_techfactors: 类型：DataFrame，  columns中有 ticker,tradeDate
    """    
    start_date, end_date = '2009-10-01', '2019-12-01'
    calendar_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
    month_end_list = sorted(list(calendar_df[(calendar_df['isMonthEnd']==1)&(calendar_df['isOpen']==1)]['calendarDate'].values))
    all_techfactors = pd.DataFrame()
    for month_end in month_end_list:
        t_df = DataAPI.MktStockFactorsOneDayGet(tradeDate=month_end,secID=u"",ticker='', field=u"ticker,tradeDate,REVS20,VOL20,ILLIQUIDITY,LFLO",pandas="1")
        all_techfactors = pd.concat([all_techfactors, t_df])
    all_techfactors['ticker'] = all_techfactors['ticker'].apply(lambda x: str(x).zfill(6))
    all_techfactors['tradeDate'] = all_techfactors['tradeDate'].apply(lambda x: str(x).replace('-', ''))    
    return all_techfactors
        

#将全市场所有股票的因子和指标数据concat起来
def concat_handle_data():
    """
    concat后的数据存储在pricevolumevol/handle_byticker中
    """    
    handle_path = 'pricevolumevol/handle_byticker'
    all_files = sorted(list(os.listdir(handle_path)))
    total_df = pd.DataFrame()
    for file in all_files[::-1]:
        ticker = file[:6]
        if ticker[:2] not in ['00', '30', '60']:
            continue        
        a = pd.read_csv('{}/{}'.format(handle_path, file))
        if len(a)==0: continue
        total_df = pd.concat([total_df, a], ignore_index=True)
    total_df['ticker'] = total_df['ticker'].apply(lambda x: str(x).zfill(6))
    total_df['tradeDate'] = total_df['tradeDate'].apply(lambda x: str(x).replace('-', ''))
    #读取传统价量因子
    tech_factors = download_techfactors()
    concat_df = total_df.merge(tech_factors, on=['ticker', 'tradeDate'], how='left')
    concat_df = concat_df.dropna()
    
    concatsave_path = 'pricevolumevol/handle_concat'    
    if not os.path.exists(concatsave_path):
        os.makedirs(concatsave_path)
    concat_df.to_csv("{}/total_handle.csv".format(concatsave_path), index=None)

    
#下载全市场A股从2009-2019年的每天的最高价、最低价、换手率和市值等数据    
time0 = time.time()
download_origin_data()
print('#################')
print("下载基础价量数据用时： {}s".format(time.time()-time0))
origin_path = 'pricevolumevol/origin_byticker'
example = pd.read_csv("{}/{}".format(origin_path, '000001_origin.csv'))
print('基础价量数据样式如下：')
print(example.head(5).to_html())


#从原始数据中生成股价振幅因子，以及换手率、涨幅、波动率等指标，以进行进一步的分析        
time0 =time.time()
handle_origin_data()
print('\n#################')
print("生成股价振幅因子用时： {}s".format(time.time()-time0))
handle_path = 'pricevolumevol/handle_byticker'
example = pd.read_csv("{}/{}".format(handle_path, '000001_handle.csv'))
print('股价振幅因子数据样式如下：')
print(example.head(5).to_html())
      
# 将全市场所有股票的因子和指标数据concat起来，生成一个DataFrame
time0 =time.time()
concat_handle_data()
print('\n#################')
print("合并股价振幅因子与传统量价因子数据用时： {}s".format(time.time()-time0))
concatsave_path = 'pricevolumevol/handle_concat'    
example = pd.read_csv("{}/{}".format(concatsave_path, 'total_handle.csv'))
print('振幅因子与传统量价因子合并后的数据样式如下：')
print(example.head(5).to_html())

'''

第二部分：因子特性
该部分耗时 5分钟
该部分内容为：

因子特性：测试传统量价因子和股价振幅因子在空头效应和多头效应上的互补性;
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

2.1 传统量价因子和股价振幅因子在空头效应和多头效应上的互补性

传统价量因子存在一个普遍特征：因子的空头效应强，而多头效应弱。从因子在不同样本空间选股效果差异的角度来看，这个现象即意味着，因子在涨幅高、换手率高、波动率高的股票集中选股效果强，而在涨幅低、换手率低、波动率低的股票集中选股效果弱。
具体地，若我们基于前一个月涨跌幅、换手率、波动率3个指标，将全市场股票分为2x2x2=8个子样本空间。在每一个子样本空间，基于目标因子（反转、换手率等）将股票分为3组，并计算因子得分最高一组股票相对于因子得分最低一组股票的月均多空收益，分析结果如下图。
'''


##分析在不同因子细分区间上的各个因子的多空效应
import matplotlib.pyplot as plt
import pandas as pd, numpy as np 
from scipy import stats

#计算对某一个因子进行分档后，各档的未来收益的均值
def cal_ret(df, factor):
    """
    参数：
    df:       Dataframe, 包含因子值列、未来收益列
    factor：  str,       要进行分档的因子名称
    输出：
    group：   Series， 各档对应的平均收益 
    """
    df = df.sort_values(by=factor)
    df['{}_rank'.format(factor)] = df[factor].rank()
    divide_n = 2
    df['{}_class'.format(factor)] = df['{}_rank'.format(factor)].apply(lambda x: (df.shape[0]*(1+np.arange(divide_n-1))/float(divide_n)).searchsorted(x))
    group  = df.groupby('{}_class'.format(factor)).apply(lambda x: x['nextone_ret'].mean())
    return group
    
#读取所有股票2009-2019的数据
totalsave_path = 'pricevolumevol/handle_concat'
total_df = pd.read_csv("{}/total_handle.csv".format(totalsave_path))
all_dates = sorted(list(set(total_df['tradeDate'])))
#对股票域进行切分的三个因子（换手率、波动率、涨幅）
use_factors = ['REVS20', 'price_vol', 'VOL20']
use_factor_classes = ['{}_class'.format(x) for x in use_factors]
#需要与股价振幅因子进行对比的传统量价因子
see_return_factors = use_factors + ['price_amplitude']
for see_return_factor in see_return_factors:
    exec('{}_df = pd.DataFrame()'.format(see_return_factor))
#逐期对不同股票域的各因子多空收益进行分析
for date in all_dates:
    part = total_df[total_df['tradeDate']==date]
    for factor in use_factors:
        factor_mean = part[factor].median()
        part['{}_zscore'.format(factor)] = (part[factor]-part[factor].mean())/part[factor].std()
        part['{}_class'.format(factor)] = part['{}_zscore'.format(factor)].apply(lambda x: 'B' if x >= 0.0 else 'S')
    for see_return_factor in see_return_factors:
        date_ret = part.groupby(use_factor_classes).apply(lambda x: cal_ret(x, see_return_factor))
        #数据过少小于要分档的数量时会报错，那么应该置为nan，等同于try
        date_ret = date_ret.sort_index(axis=1)
        #价量幅度因子与未来收益是正相关，其他三个因子是负相关
        if see_return_factor == 'price_amplitude':
            date_ret_se = date_ret.iloc[:, -1] - date_ret.iloc[:, 0]
        else:
            date_ret_se = date_ret.iloc[:, 0] - date_ret.iloc[:, -1]
        date_ret_se.name = date
        exec('{0}_df = pd.concat([{0}_df, date_ret_se], axis=1)'.format(see_return_factor))
    
total_mean_df = pd.DataFrame()
total_pvalue_df = pd.DataFrame()
for see_return_factor in see_return_factors:
    exec('tmp = {}_df'.format(see_return_factor))
    tmp1 = tmp.apply(lambda x: stats.ttest_1samp(list(x.values), 0).pvalue, axis=1)
    exec('{}_pvalue = tmp1'.format(see_return_factor))
    exec('{0}_pvalue.name = "{0}"'.format(see_return_factor))
    exec('{0}_mean = {0}_df.mean(axis=1)'.format(see_return_factor))
    exec('{0}_mean.name = "{0}"'.format(see_return_factor))    
    total_mean_df = pd.concat([total_mean_df, eval('{}_mean'.format(see_return_factor))], axis=1)
    total_pvalue_df = pd.concat([total_pvalue_df, eval('{}_pvalue'.format(see_return_factor))], axis=1)

total_mean_resetindex = total_mean_df.reset_index()
total_mean_resetindex.columns = [u'涨幅', u'波动率', u'换手率', u'动量因子多空收益', u'波动率因子多空收益', u'换手率因子多空收益', u'股价振幅因子多空收益'] 
total_mean_resetindex = total_mean_resetindex.replace({'B': '高', 'S': '低'})
total_mean_resetindex = total_mean_resetindex.applymap(lambda x: '{}%'.format(round(100*x, 2)) if type(x)!=str else x)
print('##各个量价指标在不同股票域下的多空表现')
print(total_mean_resetindex.set_index([u'涨幅', u'波动率', u'换手率']).to_html())
print('##各个量价指标在不同股票域下的多空表现图示')
total_mean_df.iloc[:, :].plot.bar(figsize=(15, 9))
plt.legend([u'动量因子', u'波动率因子', u'换手率因子', u'股价振幅因子'], prop=font)
plt.title(u'各个价量因子在不同股票域里的多空收益', fontproperties=font, fontsize=16)
plt.xlabel(u'股票域', fontproperties=font, fontsize=16)
plt.ylabel(u'因子多空收益', fontproperties=font, fontsize=16)
plt.show()

'''
注释：上图中（B，B，B）表示在过去一个月涨幅高、波动率高、换手率高的股票域, （S，S，S）表示在过去一个月涨幅低、波动率低、换手率低的股票域

从上图可以看出，常见技术因子（反转、换手率、波动率），主要在涨幅高、换手率高的样本空间中存在显著选股效果；而在涨幅低、换手率低的样本空间，月均多空收益并不显著。但是振幅因子可以对常见价量类因子进行有效补充，常见技术因子在涨幅低、换手率低的样本集中，选股效果非常有限；而在此样本空间，3个月振幅因子选股效果显著。

   调试 运行
文档
 代码  策略  文档
第三部分：因子测试
该部分耗时 15分钟
该部分内容为：

计算股价振幅因子与其他量价因子(反转、换手率、波动率、流动性)的相关性；

测试股价振幅因子的IC、多空收益表现以及分组收益等; 测试将传统量价因子中性化掉之后的股价振幅因子的IC、多空收益表现以及分组收益等；

比较传统量价因子构建的因子、股价振幅因子、传统量价因子与股价振幅因子相结合三组的表现差异;

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 计算股价振幅因子与其他量价因子(反转、换手率、波动率、流动性)和市值因子的相关性

振幅因子是一个价量指标，与传统的价量因子必然存在着一定的相关性，该部分主要探究股价振幅因子与传统量价因子和市值因子的相关性。
具体地，先将传统价量因子和市值因子做zscore变换。接着按照振幅因子从小到大将股票池分为十组，分别统计在各个组内各因子的zscore值的均值。若某个因子随着股价振幅因子的增大，它在该分组内的zscore的均值也越大，说明股价振幅因子与该因子越正相关，反之亦然。
'''

import pandas as pd, numpy as np
import matplotlib.pylab as plt

totalsave_path = 'pricevolumevol/handle_concat'
total_df = pd.read_csv("{}/total_handle.csv".format(totalsave_path))
all_dates = sorted(list(set(total_df['tradeDate'])))
feature_factors = ['REVS20', 'VOL20', 'ILLIQUIDITY', 'LFLO', 'price_vol']
feature_zscores = ['{}_zscore'.format(x) for x in feature_factors]
total_out = []
for date in all_dates:
    part = total_df[total_df['tradeDate']==date]
    part['price_amplitude_rank'] = part['price_amplitude'].rank()
    #将因子标准化
    for factor in feature_factors: 
        part['{}_zscore'.format(factor)] = (part[factor] - part[factor].mean())/part[factor].std()
    divide_n = 10
    part['price_amplitude_class'] = part['{}_rank'.format('price_amplitude')].apply(lambda x: (part.shape[0]*(1+np.arange(divide_n-                                                                      1))/float(divide_n)).searchsorted(x))
    out = part.groupby('price_amplitude_class').apply(lambda x: x[feature_zscores].mean())
    total_out.append(out)
#求平均
total_sum = sum(total_out)
total_mean = total_sum/float(len(total_out))
total_mean.plot()
plt.title(u'传统量价因子在股价振幅因子不同分组上的分布', fontproperties=font, fontsize=16)
plt.ylabel(u'各量价因子zscore的均值', fontproperties=font, fontsize=16)
plt.show()

    
    
'''
注释：图中横坐标表示股价振幅因子的分组。从0-9数字越小，表明股价振幅因子越小；数字越大，股价振幅因子越大。纵坐标是各个因子在该分组的zcore的均值。

从上图中可以看出，股价振幅因子与换手率和波动率因子呈现非常明显的正相关性。振幅大的股票组合，其前期价格波动率和换手率也越高。此外振幅因子与市值因子也呈现一定的负相关性，振幅大的股票组合其市值通常相对较低。
'''


# 剔除次新股、ST股、停牌股
def remove_special_stocks(input_factor_df):
    '''
    input_factor_df: 因子值dataframe，列包括 ticker, tradeDate(%Y%m%d)
    '''
    factor_df = input_factor_df.copy()
    ori_len = len(factor_df)
    factor_df['tradeDate'] = factor_df['tradeDate'].apply(lambda x: x.replace("-", ""))
    # 剔除掉次新股、ST股、停牌股
    remove_df['tradeDate'] = remove_df['tradeDate'].apply(lambda x: x.replace("-", ""))
    factor_df = factor_df.merge(remove_df, on=['ticker', 'tradeDate'], how='left')
    factor_df = factor_df[factor_df['remove_flag'].isnull()]
    after_remove_len = len(factor_df)
    print (u'剔除掉次新股、ST股、停牌股减少了%s条记录'%(ori_len-after_remove_len))
    return factor_df


# 提取出月末因子值dataframe
def extract_month_end_factor(in_factor_df, start_date, end_date, factor_name):
    '''
    in_factor_df: 因子值dataframe，至少包含列： ticker, tradeDate(str或者int格式), <factor_name>
    start_date/end_date: %Y%m%d
    return: 和输入同等列的dataframe， tradeDate格式为:%Y%m%d
    '''
    # 区间内的月末交易日期
    trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
    trade_date = trade_date[trade_date.isMonthEnd == 1]
    trade_date_list = [x.replace("-", "") for x in trade_date['calendarDate'].tolist()]
    # 筛选因子值
    in_factor_df['tradeDate'] = in_factor_df['tradeDate'].apply(lambda x: str(x).replace("-", ""))
    factor_df = in_factor_df[in_factor_df['tradeDate'].isin(trade_date_list)]
    return factor_df



# 获取个股的月度收益和下个月收益数据
def get_monthly_return(start_date, end_date):
    '''
    start_date/end_date:%Y%m%d
    返回：
    收益dataframe，列为: ticker, tradeDate(%Y%m%d, 月末日期), ret(当月收益), nx_ret(下月收益)
    '''
    chgframe = DataAPI.MktEqumAdjGet(beginDate=start_date, endDate=end_date, field=['ticker', 'endDate', 'chgPct'], pandas="1")
    chgframe['endDate'] = chgframe['endDate'].apply(lambda x: x.replace("-", ""))
    chgframe.columns = ['ticker', 'tradeDate', 'ret']
    chgframe = chgframe.sort_values(by=['ticker', 'tradeDate'], ascending=True)
    chgframe['nx_return'] = chgframe.groupby(['ticker'])['ret'].shift(-1)
    return chgframe


# 根据分组收益率序列，画图展示分组收益时间序列图和柱状图
def plot_group_perf(perf, quantile_num, direction=1):
    """
    perf: 分组收益,index为日期, columns为分组组名
    quantile_num: 分组数
    """
    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(211)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(212)

    for i in range(quantile_num):
        gperf = perf.iloc[:, i]
        ax1.plot(pd.to_datetime(gperf.index), (gperf + 1).cumprod(), label=gperf.name + u'左轴',
                 color=matplotlib.colors.cnames.values()[i])
    if direction ==1 :
        label = u'Top-Bottom(右轴)'
    else:
        label = u'Top-Bottom(右轴, 反向)'
        
    _ = ax2.plot(pd.to_datetime(perf.index), (perf['Top-Bottom'] + 1).cumprod(), label=label, color='k')
    _ = ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
    _ = ax1.set_title(u"%s档回测净值" % quantile_num, fontproperties=font, fontsize=16)
    _ = ax1.legend(loc=2, prop=font)
    _ = ax2.legend(loc=1, prop=font)
    _ = ax2.grid(False)

    label_dict = {u'Q1': u'Q1(High)', u'Q%s'%quantile_num: u'Q%s(Low)' % quantile_num}
    for gnroup_num in range(2, quantile_num):
        label_dict[u'Q%s'%gnroup_num] = u'Q%s' % gnroup_num

    nav = ((perf + 1).prod() - 1)[:quantile_num]
    ind = np.arange(quantile_num)
    _ = ax3.bar(ind + 0.2, nav, 0.3, color='r')
    _ = ax3.set_xlim((0, ind[-1] + 1))
    _ = ax3.set_xticks(ind + 0.35)
    _ = ax3.set_xticklabels([label_dict[x] for x in nav.index], fontproperties=font)
    _ = ax3.set_title(u"%s档回测累计绝对收益" % quantile_num, fontproperties=font, fontsize=16)
    _ = plt.show()
    return


# 进行简易分组，等权构建组合回测
def group_bt(factor_df, quantile_num, direction=1):
    """
    进行简易分组，等权构建组合回测。(组数越小，因子值越大)
    factor_df：信号dataframe，列至少包含： date("%Y%m%d", 或者"%Y-%m-%d, 或者int), ticker, value, return(下周期收益)
    direction: 1为正向信号的top-bottom, -1为负向信号的top-bottom
    return:
           perf: 每组每期的绝对收益率
    """
    factor_df = factor_df.dropna(subset=['value', 'return'])
    # 根据因子值进行分组
    factor_df['group'] = factor_df.groupby('date')['value'].apply(
        lambda x: 1.0 * (x.rank(method='first') - 1) / len(x) * quantile_num).astype(int)
    factor_df['group'] = quantile_num - factor_df['group']

    count_df = factor_df.groupby(['date', 'group']).apply(lambda x: len(x)).reset_index()
    count_df.columns = ['date', 'group', 'count']
    count_df['weight'] = 1.0 / count_df['count']

    # 等权构建组合
    bt_df = factor_df.merge(count_df[['date', 'group', 'weight']], on=['date', 'group'])
    perf = bt_df.groupby(['group', 'date']).apply(lambda x: sum(x['return'] * x['weight'])).reset_index()
    perf.columns = ['group', 'date', 'period_return']
    perf = perf.pivot_table(values='period_return', index='date', columns='group')

    perf = perf.sort_index()
    perf = perf.shift(1)
    perf.iloc[0, :] = 0
    perf.columns = ['Q' + str(i) for i in perf.columns]

    # 计算多空收益率
    if direction == 1:
        perf['Top-Bottom'] = (perf['Q1'] - perf['Q' + str(quantile_num)]) / 2.0
    else:
        perf['Top-Bottom'] = (perf['Q' + str(quantile_num)] - perf['Q1']) / 2.0
    return perf, bt_df


# 统计Top-Bottom组合的年化收益、回撤、IR等指标
def get_perf_summary(perf, col='Top-Bottom'):
    '''
    perf: index为日期，列至少包括<col>， 值为每一个月的period_return
    '''
    # 累计净值
    final_nav = (perf[col] + 1).cumprod().values[-1]
    total_month_len = len(perf)
    # 年化收益
    # annual_return = (final_nav-1)*12.0/total_month_len
    annual_return = 12*perf[col].mean()
    # 年化波动率
    annual_std = np.sqrt(12)*perf[col].std()
    # 信息比率
    ir = np.sqrt(12)*perf[col].mean()/perf[col].std()
    # 月度胜率
    hit_rate = 1.0*len(perf[perf[col]>=0])/len(perf)
    # 最大回撤
    nav_df = (perf[col] + 1).cumprod()
    cum_max = nav_df.cummax()
    maxdrawdown =((cum_max-nav_df)/cum_max).max()
    summary_df = pd.DataFrame()
    summary_df.loc[0, u'总收益'] = "%s"%(round(100*(final_nav-1), 2))+"%"
    summary_df.loc[0, u'区间长度(年)'] = total_month_len/12.0
    summary_df.loc[0, u'年化收益'] = "%s"%(round(100*annual_return, 2))+"%"
    summary_df.loc[0, u'年化波动率'] = "%s%%"%(round(100*annual_std, 2))
    summary_df.loc[0, u'信息比率'] = round(ir, 2)
    summary_df.loc[0, u'月度胜率'] = "%s"%(round(100*hit_rate, 2))+"%"
    summary_df.loc[0, u'最大回撤'] = "%s"%(round(100*maxdrawdown, 2))+"%"
    return summary_df


# 计算信号IC
def calc_ic(input_df, ic_type='spearman'):
    '''
    input_df: 列至少包括, ticker, tradeDate, value, return
    '''
    ic_summary = pd.DataFrame()
    
    ic_df = input_df.groupby(['tradeDate']).apply(
        lambda x: x[['value', 'return']].dropna().corr(method=ic_type).values[0, 1])
    
    ic_df = ic_df.reset_index()
    ic_df.columns = ['date', 'ic']
    
    # IC的T值
    ic_t = stats.ttest_1samp(ic_df['ic'].dropna(), 0)[0]
    ic_t = round(ic_t, 2)
    # IC均值
    ic_mean = ic_df['ic'].mean()
    ic_mean = "%s%%" % (round(ic_mean * 100, 2))
    # ICIR
    ic_std = ic_df['ic'].std()
    ic_std = "%s%%" % (round(ic_std * 100, 2))
    ic_ir = ic_df['ic'].mean() / ic_df['ic'].std()
    ic_ir = round(ic_ir, 2)
    ic_summary.loc[ic_type, 'IC'] = ic_mean
    ic_summary.loc[ic_type, 'T'] = ic_t
    ic_summary.loc[ic_type, 'IC_IR'] = ic_ir
    return ic_summary
    
# 回测因子表现
def backtest_factor_perf(factor_frame, factor_name, start_date, end_date, group_num=5, direction=1):
    '''
    factor_frame: 因子列dataframe, 列至少包括：ticker, tradeDate(%Y%m%d), <factor_name>
    group_num: 收益分组的组数
    start_date/end_date: 回测的开始和结束时间,%Y%m%d
    direction: 构造多空组合时，最大-最小组(direction=1)，还是最小-最大组
    return:
            perf: 分组收益,index为日期, columns为分组组名
    '''
    # 只保留月末因子值
    factor_df = extract_month_end_factor(factor_frame, start_date, end_date, factor_name)
    
    # # 剔除掉次新股、ST股、停牌股
    factor_df = remove_special_stocks(factor_df)
    
    # 个股的月度收益dataframe，列至少包括 ticker, tradeDate(%Y%m%d, 月末日期), ret(当月收益), nx_return(下月收益)
    return_frame = get_monthly_return(start_date, end_date)
        
    # 合并因子值和下一期收益, 合并之后的列为: ticker, tradeDate, <factor_name>, 'return'
    factor_df = factor_df.merge(return_frame[['ticker', 'tradeDate', 'nx_return']], on=['ticker', 'tradeDate'], how='left')
    
    factor_df = factor_df.rename(columns={"nx_return": "return"})[['ticker', 'tradeDate', factor_name, 'return']]
        
    factor_df = factor_df.dropna(subset=[factor_name, 'return']).rename(columns={factor_name:"value"})
    
    # 计算因子的IC， RankIC, ICIR, 年化ICIR
    ic_df = calc_ic(factor_df, ic_type='pearson')
    rankic_df = calc_ic(factor_df, ic_type='spearman')
    ic_df = pd.concat([ic_df, rankic_df])
    print (u'## 信号IC表现为：\n')
    print (ic_df.to_html())
    
    # 分组回测
    perf, bt_df = group_bt(factor_df.rename(columns={"tradeDate":"date"}), group_num, direction=direction)
    
    summary_df = get_perf_summary(perf)
    print (u'## 多空对冲组合表现：\n')
    print (summary_df.to_html())
    
    print (u'## 因子分组表现: ')
    # 画图展示
    _ = plot_group_perf(perf, group_num, direction=direction)
    return perf, bt_df



# 比较因子表现， 计算多个因子的IC，T值，IR，年化收益，净值曲线
def compare_signal_perf(factor_dict, start_date, end_date, group_num=5, direction=1):
    '''
    :param factor_dict: 多个信号dataframe的dict, key为信号名, value为<signal_df>
    signal_df格式为: ticker, tradeDate("%Y%m%d"), 'value'
    :param start_date: 回测起始时间，%Y%m%d
    :param end_date:回测结束时间，%Y%m%d
    :param group_num: 多空对冲时，分组的组数
    :param direction: 多空对冲组合的方向
    :return:
    perf_dict: key为信号名，perd_dict = {<信号名1>:{"IC":IC值, "IC_T":IC的T值, "ls_annual_ret":多空年化收益, "ls_ir":多空IR, "nav_df":多空的净值df, index为date}} 
    '''
    perf_dict = {}
    return_frame = None # 个股月度收益数据
    for factor_name in factor_dict.keys():
        print (u'回测%s信号的表现...'%factor_name)
        factor_frame = factor_dict[factor_name]
        # 只保留月末因子值
        factor_df = extract_month_end_factor(factor_frame, start_date, end_date, 'value')

        # 剔除掉次新股、ST股、停牌股
        remove_df['tradeDate'] = remove_df['tradeDate'].apply(lambda x: x.replace("-", ""))
        factor_df = factor_df.merge(remove_df, on=['ticker', 'tradeDate'], how='left')
        factor_df = factor_df[factor_df['remove_flag'].isnull()]

        if return_frame is None:
            return_frame = get_monthly_return(start_date, end_date)

        # 合并因子值和下一期收益, 合并之后的列为: ticker, tradeDate, 'value', 'return'
        factor_df = factor_df.merge(return_frame[['ticker', 'tradeDate', 'nx_return']], on=['ticker', 'tradeDate'],
                                    how='left')
        factor_df = factor_df.rename(columns={"nx_return": "return"})
        factor_df = factor_df.dropna(subset=['value', 'return'])

        # 得到因子的IC和T值
        rankic_df = calc_ic(factor_df, ic_type='spearman')
        ic_mean = float(rankic_df.loc['spearman', 'IC'].replace("%", ""))
        ic_t = rankic_df.loc['spearman', 'T']

        # 得到多空对冲的年化收益、IR、净值曲线
        perf, bt_df = group_bt(factor_df.rename(columns={"tradeDate": "date"}), group_num, direction=direction)
        summary_df = get_perf_summary(perf)
        ## 年化收益
        ls_annual_ret = float(summary_df.loc[0, u'年化收益'].replace("%", ""))/100
        ## IR
        ls_ir = summary_df.loc[0, u'信息比率']
        ## 净值曲线
        nav_df = (perf['Top-Bottom'] + 1).cumprod()


        # 记录到dict中
        perf_dict[factor_name] = {"IC": ic_mean, "IC_T": ic_t, "ls_annual_ret": ls_annual_ret, "ls_ir": ls_ir,
                                  "nav_df": nav_df}
    return perf_dict


# 画图比较多个信号的表现
def plot_compare_signal_perf(perf_dict):
    '''
    perf_dict: key为信号名，perd_dict = {<信号名1>:{"IC":IC值, "IC_T":IC的T值, "ls_annual_ret":多空年化收益, "ls_ir":多空IR, "nav_df":多空的净值df, index为date}} 
    '''
    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    fig2 = plt.figure(figsize=(15,8))
    ax5 = fig2.add_subplot(111)
    
    factor_names = perf_dict.keys()
    factor_names.sort()
    # 画图展示不同因子的IC均值
    ic_plot_df = pd.Series()
    ic_t_plot_df = pd.Series()
    annual_ret_df = pd.Series()
    annual_ir_df = pd.Series()
    nav_df_list = []
    for factor_name in factor_names:
        ic_plot_df[factor_name] = perf_dict[factor_name]['IC']
        ic_t_plot_df[factor_name] = perf_dict[factor_name]['IC_T']
        annual_ret_df[factor_name] = perf_dict[factor_name]['ls_annual_ret']
        annual_ir_df[factor_name] = perf_dict[factor_name]['ls_ir']
        fnav_df = perf_dict[factor_name]['nav_df']
        fnav_df.name=factor_name
        nav_df_list.append(fnav_df)
    nav_df = pd.concat(nav_df_list, axis=1)
    
    # IC均值
    ax1 = ic_plot_df.plot.bar(ax=ax1)
    ax1.set_xticklabels(ic_plot_df.index.values, fontproperties=font, rotation=45)
    ax1.set_title(u'RankIC均值', fontproperties=font, fontsize=16)
    
    # IC的T值
    ax2 = ic_t_plot_df.plot.bar(ax=ax2)
    ax2.set_xticklabels(ic_t_plot_df.index.values, fontproperties=font, rotation=45)
    ax2.set_title(u'RankIC T值', fontproperties=font, fontsize=12)
    
    # 多空组合的年化收益
    ax3 = annual_ret_df.plot.bar(ax=ax3)
    ax3.set_xticklabels(annual_ret_df.index.values, fontproperties=font, rotation=45)
    ax3.set_title(u'多空组合年化收益', fontproperties=font, fontsize=12)
    
    # 多空组合的IR
    ax4 = annual_ir_df.plot.bar(ax=ax4)
    ax4.set_xticklabels(annual_ir_df.index.values, fontproperties=font, rotation=45)
    ax4.set_title(u'多空组合IR', fontproperties=font, fontsize=12)
    
    # 多空组合的累计净值
    ax5 = nav_df.plot(ax=ax5)
    _ = ax5.legend(loc=2, prop=font)
    ax5.set_title(u'多空组合累计净值', fontproperties=font, fontsize=12)
    
    return 

'''
3.2.2 测试原始股价振幅因子

该部分主要测试股价振幅因子的表现，主要包括IC、因子多空收益和五分组收益。

'''


totalsave_path = 'pricevolumevol/handle_concat'
total_df = pd.read_csv("{}/total_handle.csv".format(totalsave_path))
total_df['tradeDate'] = total_df['tradeDate'].apply(lambda x: str(x).replace('-', ''))
all_dates = sorted(list(set(total_df['tradeDate'])))
price_amplitude = total_df[['ticker', 'tradeDate', 'price_amplitude']]
price_amplitude['ticker'] = price_amplitude['ticker'].apply(lambda x: str(x).zfill(6))
price_amplitude['tradeDate'] = price_amplitude['tradeDate'].apply(lambda x: str(x).replace('-', ''))
print (u'股价振幅因子格式为:')
print (price_amplitude.head().to_html())
 
backtest_start_date = all_dates[0]
backtest_end_date = all_dates[-1]

# 测试股价振幅因子的表现
perf_old_price_amplitude, bt = backtest_factor_perf(price_amplitude, 'price_amplitude', backtest_start_date, backtest_end_date, group_num=5, direction=1)

'''
从上图中可以看出，原始的股价振幅因子表现较差,pearson系统为正但是spearman系数为负；且因子的分组收益也不具有区分度。股价振幅因子本身不是一个很好的alpha因子。

   调试 运行
文档
 代码  策略  文档
3.2.3 测试将传统量价因子中性化之后的股价振幅因子

上一章节显示股价振幅因子本身不是一个很好的alpha因子，由于该因子与传统价量因子相关性较大，考虑将传统价量因子中性化掉之后回测其表现；具体的回测内容包括因子IC、多空收益和五分组收益。

'''

from sklearn.linear_model import LinearRegression
totalsave_path = 'pricevolumevol/handle_concat'
total_df = pd.read_csv("{}/total_handle.csv".format(totalsave_path))
total_df['tradeDate'] = total_df['tradeDate'].apply(lambda x: str(x).replace('-', ''))
total_df['ticker'] = total_df['ticker'].apply(lambda x: str(x).zfill(6))
all_dates = sorted(list(set(total_df['tradeDate'])))

total_out = pd.DataFrame()
for date in all_dates:
    part = total_df[total_df['tradeDate']==date]    
    #dropna 用于回归
    part = part.dropna()
    #先会对市值 波动率 换手率 动量 给中性化掉
    zscore_factors = list(set(part.columns) -set(['ticker', 'tradeDate', 'nextone_ret']))
    for factor in zscore_factors: 
        part['{}_zscore'.format(factor)] = (part[factor] - part[factor].mean())/part[factor].std()
    X_factors = list(set(zscore_factors)-set(['price_amplitude']))
    Y_factor = 'price_amplitude'
    X_values = part[X_factors].values
    Y_values = part[Y_factor].values
    linreg = LinearRegression()
    linreg.fit(X_values, Y_values)
    Y_predict = linreg.predict(X_values)
    Y_resi = Y_values - Y_predict
    part = part.rename(columns={'price_amplitude': 'price_amplitude_origin'})
    part['price_amplitude'] = Y_resi
    total_out = pd.concat([total_out, part])
total_out = total_out
price_amplitude_neu = total_out[['ticker', 'tradeDate', 'price_amplitude']]

print (u'将传统量价因子中性化后的股价振幅因子格式为:')
print (price_amplitude_neu.head().to_html())
 
backtest_start_date = all_dates[0]
backtest_end_date = all_dates[-1]

# 测试股价振幅因子的表现
perf_old_price_amplitude_neu, bt = backtest_factor_perf(price_amplitude_neu, 'price_amplitude', backtest_start_date, backtest_end_date, group_num=5, direction=1)

'''

从上图中可以看出，股价振幅因子在将传统的量价因子中性化掉之后，表现非常好。在IC方面 pearson相关系数为2.97%， spearman相关系数为4.55%；多空对冲组合的曲线信息比率有1.88；因子的五分组收益也具有明显的区分度。

   调试 运行
文档
 代码  策略  文档
3.3 比较传统量价因子构建的因子和加入了股价振幅因子后的表现差异

   调试 运行
文档
 代码  策略  文档
3.3.1 分别构建传统量价因子的等权组合和加入了股价振幅因子后的等权组合

为了进一步说明股价振幅因子对传统价量因子的互补作用，设置对比实验。一个是用传统价量因子等权构建的因子，另一个是传统价量因子加上股价振幅因子后等权构建的因子（传统量价因子均为负向因子，而股价振幅因子为正向因子，因此在合成时将股价振幅因子乘以-1之后再与其他量价因子等权合成）。


'''

#传统量价因子等权构建的因子
factor_df = total_out.copy()
use_tech_factors = ['REVS20', 'VOL20', 'ILLIQUIDITY', 'LFLO', 'price_vol']
factor_df[use_tech_factors] = factor_df[use_tech_factors].apply(lambda x: (x-x.mean())/x.std())
use_tech_factors = ['REVS20', 'VOL20', 'ILLIQUIDITY', 'LFLO', 'price_vol']
factor_df[use_tech_factors] = factor_df[use_tech_factors].apply(lambda x: (x-x.mean())/x.std())
factor_df['multifactor_origin'] = factor_df[use_tech_factors].mean(axis=1)

#传统量价因子与股价振幅因子构建的多因子
#股价振幅因子与传统量价因子反向
factor_df['price_amplitude_origin'] = (factor_df['price_amplitude_origin']-factor_df['price_amplitude_origin'].mean())/factor_df['price_amplitude_origin'].std()
factor_df['price_amplitude'] = (factor_df['price_amplitude_origin']-factor_df['price_amplitude'].mean())/factor_df['price_amplitude'].std()
factor_df['price_amplitude_origin_neg'] = -factor_df['price_amplitude_origin']
factor_df['price_amplitude_neg'] = -factor_df['price_amplitude']
factor_df['multifactor_with_hamplitude'] = factor_df[['price_amplitude_neg']+use_tech_factors].mean(axis=1)
factor_df['multifactor_with_hamplitude_origin'] = factor_df[['price_amplitude_origin_neg']+use_tech_factors].mean(axis=1)

print('##factor_df的数据形式为 ')
print(factor_df.head().to_html())

'''

3.3.2 分别测试传统量价因子等权构建组合的表现和加入了股价振幅因子后的表现

分别回测上个章节构造的两个因子的表现，观察每个因子的IC、多空收益和分组收益。

'''


# 测试传统量价因子等权构建组合的表现
print('#############################')
print('###传统量价因子等权构建组合的表现')
perf_old_multifactor_origin, bt = backtest_factor_perf(factor_df[['tradeDate', 'ticker', 'multifactor_origin']], 'multifactor_origin', backtest_start_date, backtest_end_date, group_num=5, direction=-1)
# 测试传统量价因子与股价振幅因子等权构建组合的表现
print("######################################")
print('###传统量价因子与股价振幅因子等权构建组合的表现')
perf_old_multifactor_with_hamplitude_origin, bt = backtest_factor_perf(factor_df[['tradeDate', 'ticker', 'multifactor_with_hamplitude_origin']], 'multifactor_with_hamplitude_origin', backtest_start_date, backtest_end_date, group_num=5, direction=-1)


'''

从上面两个因子的对比表现可以看出，在将股价振幅因子加入传统的量价因子中等权构建组合后，无论是IC还是因子多空收益都有明显的提升！其中IC的绝对值从5.92%提高到6.64%；年化收益从8.48%提高到11.0%；信息比率从1.26提高到1.76；五分组的收益的区分度也明显提高。

   调试 运行
文档
 代码  策略  文档
3.3.3 对比传统量价因子构建的因子、股价振幅因子、传统量价与股价振幅因子结合三组的表现差异

从IC和净值曲线角度观察传统量价因子构建的因子、股价振幅因子、传统量价与股价振幅因子结合三组的表现差异

'''


# 将信号放到dict中
factor_dict = {u"传统量价":factor_df[['tradeDate', 'ticker', 'multifactor_origin']].rename(columns={'multifactor_origin': 'value'}),
               u'振幅因子': factor_df[['tradeDate', 'ticker', 'price_amplitude_neg']].rename(columns={'price_amplitude_neg': 'value'}),
               u'二者结合': factor_df[['tradeDate', 'ticker', 'multifactor_with_hamplitude_origin']].rename(columns={'multifactor_with_hamplitude_origin': 'value'})}


# 回测各个信号的表现并画图展示
perf_dict = compare_signal_perf(factor_dict, backtest_start_date, backtest_end_date, group_num=5, direction=-1)
_ = plot_compare_signal_perf(perf_dict)


'''

从上图可以我们可以知道，股价振幅因子本身是一个较差的因子，但由于它本身在低波动率、低换手率、低涨幅的股票区域上的有着传统量价因子所不具备的明显选股效应，所以对传统的量价因子是一个很好的补充。实践也表明在加入了股份振幅因子之后，等权组合的收益显著提高。
   调试 运行
文档
 代码  策略  文档
第四部分：总结
   调试 运行
文档
 代码  策略  文档
股价振幅因子和传统的量价因子在空头效应和多头效应上可以互补，是对传统量价因子体系一个很好的补充；

股价振幅因子本身是一个较差的因子，但由于它本身在低波动率、低换手率、低涨幅的股票区域上的有着传统量价因子所不具备的明显选股效应, 在加入到传统的量价因子中等权构建组合后，发现无论是IC还是因子多空收益都有明显的提升！其中IC从-5.92%提高到-6.64%；年化收益从8.48%提高到11.0%；信息比率从1.26提高到1.76；五分组的收益的区分度也明显提高。

'''