# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:08:51 2020

@author: Asus
"""
'''
导读
A.研究目的：随着金融市场的发展，人们越来越认识到极端事件对金融市场的巨大影响，而在极端的负面事件发生后，市场往往会呈现出异常的同跌现象，这种现象是常规的相关系数度量方法所无法捕捉到的，本文参考东方证券：《基于copula的尾部相关性研究：上尾异常相关系数因子》，通过引入copula函数来度量股票与市场间的尾部相关性，并构建相应的选股因子。

B.研究结论：实证表明，通过优矿平台及研报模型进行实证，我们构建及改进后的选股因子表现如下：

风格中性后的上尾相关系数因子rankIC达到5.38%，IC_IR为2.29，多空收益非常稳定，且分组收益完全单调，但是该因子和特异度因子的相关性很高，在剥离掉特异度之后的残差因子不具备显著的选股效果；
基于回归改进的上尾异常相关系数因子在风格中性后的rankIC为2.45%，IC_IR为1.78，因子在中证500空间内的选股效果出色，除了与特异度因子的相关系数为36%外，与其余常见因子的相关性都在20%以下，该因子在原有因子的基础上提供了增量信息，可以放入多因子模型增强模型的选股效果。
C.文章结构：本文共分为3个部分，具体如下

一、尾部相关性理论：这部分主要包括尾部相关性理论的介绍、尾部相关性因子构造、尾部相关性和市场的关系

二、上下尾相关系数因子：这部分主要是研究因子的选股效果，主要包括因子的测试和相关性分析

三、上尾相关系数因子改进：上尾异常相关系数因子的构建和分析

D.运行时间说明:

一、尾部相关性理论，需要30分钟左右

二、上下尾相关系数因子，需要30分钟左右

三、上尾相关系数因子改进，需要10分钟左右

四、总耗时：70分钟左右

 注意事项 
为便于阅读，本文将所有与文章主题无关的函数放在函数库里面：
链接：https://uqer.io/community/share/MxenTSYbXtSIdXJvSLEoaVYcq5E0/private；密码：1293。请前往查看并注意保密。
请在运行之前，克隆上面的代码，并存成lib(右上角->另存为lib，改名为quant)

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

第一部分：尾部相关性理论
该部分耗时 30分钟
该部分内容为：

尾部相关性理论

尾部相关系数因子构造

尾部相关性和市场的关系

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


1.1 尾部相关性理论

研究背景

在金融市场的负面事件发生后，市场之间往往呈现出异常的同跌现象，而这种现象是常规的相关系数度量方法无法捕捉到的，因此如何度量资产组合的尾部风险得到了越来越多的重视；

VaR方法由于在估计资产组合的时通常用多元高斯分布作为资产收益率的分布，而实际中金融资产往往具有尖峰厚尾的特征，因此VaR方法会大幅度低估资产组合的下行风险；

关于尾部相关性的度量，目前主要有如下三种方法：

条件相关性：条件相关性在市场发生剧烈变动时的相关性会大于波动较小时的相关性，但是条件相关性的计算需要假设序列间的分布情况，分布选择不当会带来较大的误差；
极值理论的多变量分布：根据尾部数据去拟合分布的参数然后估计出尾部的相关性，这个方法的确能比较好地描述不同资产间的尾部分布，但是需要估计的参数太多，比较适合用于衡量少数资产长期的相关性情况，不适合用于构建选股因子；
copula函数法：通过copula函数去描述不同资产的联合分布，然后根据copula函数的形式和参数来度量随机变量间的尾部相关性，这也是目前学界度量尾部相关性运用最广泛的方法
Copula函数
copula函数边缘分布函数和联合分布函数的连接函数，定义由Sklar定理引出：


Copula函数族
常用的copula函数族根据分类不同有很多类型，按照分布类型可以分为椭圆族copula和阿基米德copula。

椭圆族copula函数族中最常用的是Normal copula和t-Student copula，这两中在金融领域的应用非常广泛：
Normal copula也叫高斯copula，也就是多元正态分布的相依函数，令随机变量服从多元正态分布假设，当且仅当期边缘分布均为正态分布，则存在唯一copula函数使得：

t-Student copula是多元t-Student分布的copula函数，假设随机变量服从多元t-Student分布，自由度为ν，均值向量为μ，则存在唯一copula函数使得： 
阿基米德copul函数在实际应用中主要有如下两个形式（Frank copula由于其对称的性质，本篇文章中计算尾部相关性用不到，就不做介绍）：
Gumbel copula，其表达形式为：

      其中θ大于0，Gumbel copula是一个有偏的函数，能较好地拟合上尾的数据，所以在应用中通常用来刻画上尾风险；
Clayton copula，其表达形式为：

      其中θ大于-1且不为0，Clayton copula也是一个有偏的函数，它能较好地拟合下尾数据，所以在应用中通常用来刻画下尾风险。
尾部相关系数
尾部相关系数是指二维分布中尾部数据的相关系数，分为上尾相关系数和下尾相关系数，假设两个随机变量X，Y分别具有边缘分布F1(X)和F2(X)，以及copula函数C(u1, u2)，那么上尾相关系数的定义为


我们一般用Gumbel copula来度量上尾的相关性，代入上尾相关系数的计算公式，λU = 2 - 21/θ； 同理，下尾相关系数的定义为

我们一般用Clayton copula来度量下尾的相关性，代入下尾相关系数的计算公式，λL= 2 -1/θ


每期股票池的选取

生成的股票池文件存储在copula_corr_factor/stock_pool.csv

股票池属性为date，code，月频，日期为每月月初

剔除中证全指中当期上市不足60天的新股，区间为2005年9月至2019年7月初


'''


# coding: utf-8

import pandas as pd
import numpy as np
import os
import time
import multiprocessing
#from CAL.PyCAL import * 
import statsmodels.api as sm
import gevent
import datetime
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cal = Calendar('China.SSE')
from lib.quant import *

raw_data_dir = "./copula_corr_factor"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

def get_stock_pool_one_day(end, day):
    """
    获取单期的股票池
    参数:
        end: 月末日期
        day: 新股上市判定区间长度
    返回: 
        df: DataFrame，columns = ['date', 'code']，date为此月第一个交易日
    """
    #univ=DynamicUniverse('A')
    #changed by steven cai from All A share to CSI300
    
    univ=DynamicUniverse('HS300')
    date = cal.advanceDate(end, '1B').strftime("%Y%m%d")
    all_code = univ.preview(date, skip_halted=False)
    # 剔除ST
    st_date = date if date is not None else datetime.datetime.now().strftime('%Y%m%d')
    df_ST = DataAPI.SecSTGet(secID=all_code, beginDate=st_date, endDate=st_date, field=['secID'])
    all_code_not_ST = [s for s in all_code if s not in list(df_ST['secID'])]
    # 剔除新股
    ticker = [x.split('.')[0] for x in all_code_not_ST]
    period = '-' + str(day) + 'B'
    pastDate = cal.advanceDate(date, period).strftime("%Y-%m-%d")
    ipo_date = DataAPI.SecIDGet(partyID=u"",assetClass=u"e",ticker=ticker,cnSpell=u"",field=u"ticker,listDate",pandas="1")
    remove_list = ipo_date[ipo_date['listDate'] > pastDate]['ticker'].tolist()
    all_ticker_not_new = [stk for stk in ticker if stk not in remove_list]
    # ticker2secID
    universe = DataAPI.EquGet(equTypeCD=u"A",listStatusCD="L,S,DE,UN",field=u"ticker,secID",pandas="1") # 获取所有的A股（包括已退市）
    universe = dict(universe.set_index('ticker')['secID'])
    if isinstance(ticker, list):
        res = []
        for i in ticker:
            if i in universe:
                res.append(universe[i])
            else:
                print (i, ' 在universe中不存在，没有找到对应的secID！')
    df = pd.DataFrame({'code': res})
    df['date'] = date
    df = df[['date', 'code']]
    return df    
    
# 股票池
tic = time.time()
path = 'copula_corr_factor/'
start_date = '20050930'
end_date = '20190715'        
day = 60
trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date).sort('calendarDate')
trade_date = trade_date[trade_date['isMonthEnd'] == 1]
date_list = [tdate.replace("-", "") for tdate in trade_date.calendarDate.values]
all_stock = []
for date in date_list:
    stock_pool = get_stock_pool_one_day(date, day)
    all_stock.append(stock_pool)
        
all_stock = pd.concat(all_stock)
all_stock.to_csv(path + 'stock_pool.csv', index=False)
toc = time.time()
print('***********股票池示例************')
print(all_stock.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


'''

1.2 尾部相关性因子构造

copula参数估计
我们在计算上下尾相关系数的时候用到的都是Archimdean copula族中的函数，对于此族中的参数的估计，我们一般是考虑先估计出Kendall相关系数τ，而对于二维分布而言，Kendall相关系数和copula函数的关系如下：


带入copula函数的表达式，得到对应参数θ和τ的关系如下：
Gumbel: τ = 1 - 1θ；

Clayton：τ = θθ+2；

由此我们就能得到θ的估计，带入公式即可求得λU和λL。

因子构造方法

我们以中证全指作为市场走势的代理变量，在2006.1-2019.6的每个月末用中证全指过去3个月的收益率和股票过去3个月的日度收益率（剔除了正常交易天数少于一半的样本）来分别估计Gumbel copula和Clayton copula的参数，并据此计算股票和市场的短期上下尾相关系数。

'''

# 所有行情数据读取及存储
tic = time.time()
all_code = sorted(all_stock['code'].unique())
mkt_info = DataAPI.MktEqudGet(secID=all_code, beginDate=start_date,endDate=end_date,isOpen="1",
                              field=u"secID,tradeDate,chgPct,turnoverValue,PB,negMarketValue",pandas="1")
mkt_info['tradeDate'] = mkt_info['tradeDate'].apply(lambda x: x.replace('-', ''))
# 存储为pkl
with open(path + 'mkt_info_frame.pkl', 'wb') as f:
    pickle.dump(mkt_info, f, pickle.HIGHEST_PROTOCOL)   
    
toc = time.time()
print('***********行情数据示例************')
print(mkt_info.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


# 计算股票和中证全指、中证500、沪深300的上尾和下尾相关系数因子
tic = time.time()
# 指数收益率及行情数据
mkt_info_pv = mkt_info.pivot_table(index='tradeDate', columns='secID', values='chgPct')
index_ticker_list = ['000300', '000905', '000985']
index_ret = DataAPI.MktIdxdGet(ticker=index_ticker_list,beginDate=u"20051010",endDate=u"20190715",exchangeCD=u"XSHE,XSHG",
                               field=u"ticker,tradeDate,CHGPct",pandas="1")
index_ret['tradeDate'] = index_ret['tradeDate'].apply(lambda x: x.replace('-', ''))
index_ret = index_ret.pivot_table(index='tradeDate', columns='ticker', values='CHGPct')
data = pd.merge(mkt_info_pv, index_ret, left_index=True, right_index=True)

# 因子计算
month_start = sorted(all_stock['date'].unique())
stock_list = list(set(data.columns) - set(index_ticker_list))
up_corr = []
low_corr = []
for i in range(4, len(month_start)):
    pre_month_start = month_start[i - 3]
    next_month_start = month_start[i]
    this_month_end = cal.advanceDate(next_month_start,'-1B').strftime("%Y%m%d")
    period_data = data[(data.index >= pre_month_start) & (data.index < next_month_start)]
    n = len(period_data) / 2
    temp = [period_data[[secID] + index_ticker_list].corr(method="kendall", min_periods=n).values[0][1: ] for secID in stock_list]
    # kendall相关系数
    kendall_corr = pd.DataFrame(index=stock_list, columns=index_ticker_list, data=np.array(temp))
    kendall_corr.dropna(axis=0, how='all', inplace=True)
    # 反推copula参数
    gumbel_copula_param = 1 / (1 - kendall_corr)
    clayton_copula_param = 2 * kendall_corr / (1 - kendall_corr)
    # 上尾相关系数
    lambda_u = 2 - 2**(1.0 / gumbel_copula_param)
    lambda_u['tradeDate'] = this_month_end
    lambda_u['secID'] = lambda_u.index
    lambda_u.reset_index(drop=True, inplace=True)
    up_corr.append(lambda_u)
    # 下尾相关系数
    lambda_l = 2**(-1.0 / clayton_copula_param)
    lambda_l['tradeDate'] = this_month_end
    lambda_l['secID'] = lambda_l.index
    lambda_l.reset_index(drop=True, inplace=True)
    low_corr.append(lambda_l)
    
toc = time.time()
up_corr_factor = pd.concat(up_corr)
low_corr_factor = pd.concat(low_corr)
up_corr_factor.to_csv(path + 'raw_up_copula_factor.csv')
low_corr_factor.to_csv(path + 'raw_low_copula_factor.csv')
print('***********上下尾相关系数示例************')
print(up_corr_factor.head(5).to_html())
print(low_corr_factor.head(5).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


if __name__ == "__main__":
    tic = time.time()
    
    # 遍历每个月末日期，利用协程对因子进行去极值处理
    print('winsoriz factor...')
    date_list = up_corr_factor['tradeDate'].unique()
    jobs = [gevent.spawn(winsorize_factor, value) for value in zip([up_corr_factor]*len(date_list), date_list)]
    gevent.joinall(jobs)
    new_frame_list = [e.value for e in jobs]
    up_factor = pd.concat(new_frame_list, axis=0)
    up_factor.reset_index(inplace=True)    
    
    jobs = [gevent.spawn(winsorize_factor, value) for value in zip([low_corr_factor]*len(date_list), date_list)]
    gevent.joinall(jobs)
    new_frame_list = [e.value for e in jobs]
    low_factor = pd.concat(new_frame_list, axis=0)
    low_factor.reset_index(inplace=True)    
    
    print ("ALL FINISHED")
    print('***********去极值化后上尾因子数据示例************')
    print(up_factor.head(5).to_html())
    print('***********去极值化下后尾因子数据示例************')
    print(low_factor.head(5).to_html())
    toc = time.time()
    print ("Time cost: %s seconds" % (toc - tic))
    


# 不同时间平均上下尾相关系数和市场关系
index_month_price = DataAPI.MktIdxdGet(beginDate=start_date, endDate=end_date, 
                                       indexID=[u"000985.ZICN", u"000300.ZICN", u"000905.ZICN"],
                                       field=u"tradeDate,ticker,closeIndex", pandas="1")
index_month_price['tradeDate'] = index_month_price['tradeDate'].apply(lambda x: x.replace('-', ''))
index_month_price = index_month_price[index_month_price['tradeDate'].isin(date_list)]
index_month_price = index_month_price.pivot_table(index='tradeDate', columns='ticker', values='closeIndex')
up_mean = up_factor.groupby('tradeDate').mean()
low_mean = low_factor.groupby('tradeDate').mean()
time_df = pd.merge(up_mean, low_mean, left_index=True, right_index=True, suffixes=('_up', '_low'))
time_df = time_df.merge(index_month_price, left_index=True, right_index=True)
time_df.index = pd.to_datetime(time_df.index)
# 上证全指
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(311)
plt.subplots_adjust(hspace=0.2)
ax.plot(time_df.index, time_df['000985_up'], '-', label=u"上尾平均相关系数(左)")
ax.plot(time_df.index, time_df['000985_low'], '-', label=u"下尾平均相关系数(左)")
ax.set_ylim(0, 1)
ax2 = ax.twinx()
ax2.plot(time_df.index, time_df['000985'], '-r', label=u"中证全指(右)")
ax.legend(loc=2, prop=font)
ax2.legend(loc=0, prop=font)
title = u"中证全指"
_ = ax.set_title(title, fontproperties=font, fontsize=12)
ax2.grid()
# 中证500
ax = fig.add_subplot(312)
ax.plot(time_df.index, time_df['000905_up'], '-', label=u"上尾平均相关系数(左)")
ax.plot(time_df.index, time_df['000905_low'], '-', label=u"下尾平均相关系数(左)")
ax.set_ylim(0, 1)
ax2 = ax.twinx()
ax2.plot(time_df.index, time_df['000905'], '-r', label=u"中证500(右)")
ax2.set_ylim(0, 12000)
ax.legend(loc=2, prop=font)
ax2.legend(loc=0, prop=font)
title = u"中证500"
_ = ax.set_title(title, fontproperties=font, fontsize=12)
ax2.grid()
# 沪深300
ax = fig.add_subplot(313)
ax.plot(time_df.index, time_df['000300_up'], '-', label=u"上尾平均相关系数(左)")
ax.plot(time_df.index, time_df['000300_low'], '-', label=u"下尾平均相关系数(左)")
ax.set_ylim(0, 1)
ax2 = ax.twinx()
ax2.plot(time_df.index, time_df['000300'], '-r', label=u"沪深300(右)")
ax2.set_ylim(0, 6000)
ax.legend(loc=2, prop=font)
ax2.legend(loc=0, prop=font)
title = u"沪深300"
_ = ax.set_title(title, fontproperties=font, fontsize=12)
ax2.grid()


'''

1.3 尾部相关性和市场的关系

在剔除了异常值之后，我们根据计算出来的股票和各市场指数的平均上下尾相关系数和市场指数在时间序列上的关系如上所示，从图中我们可以看到：

对于不同指数而言，其实上下尾相关系数的变动非常同步，而且下尾相关系数几乎一直大于上尾相关系数，这与常规的实证结果是一致的；

从变动上可以看到，市场在大幅上涨的区间内（07年、09年，15年）的尾部相关系数都比较低，但是大幅下跌的区间内(08年、15年下半年、18年)的尾部相关系数则较高，也就是说，"牛市一起涨"的概率平均而言是小于"熊市一起跌"的，这与我们通常的认知也是一致的。

第二部分：上下尾相关系数因子
该部分耗时 30分钟
该部分内容为：

上下尾相关系数因子的测试

与其他因子的相关性分析与增量Alpha分析

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

2.1 上下尾相关系数因子的测试

因子预处理
在得到原始因子后，考虑到需要计算IC值，我们对每个因子进行如下处理：

Step1: 采用MAD（Median Absolute Deviation 绝对中位数法）进行边界压缩处理，剔除异常值；
Step2: 如果需要，我们将经过去极值处理的因子对市值+行业作中性化处理；
Step3: 对第二步的残差项做z-score标准化处理；
Step4: 计算每个股票的次月收益数据备用，factor_stand就是做完预处理后的标准化因子数据，格式如下： 图片注释
经过如上步骤得到的factor_stand就是我们需要的标准化因子数据，主要包括：

lambda_up/lambda_down: 股票与中证全指的上下尾相关系数原始因子[1]；

lambda_up_neu/lambda_down_neu: 行业市值中性化后的上下尾相关系数因子；

IVR: 3个月特异度因子；

lambda_up_neu_ivr/lambda_down_neu_ivr: 剥离了3个月特异度后的上下尾相关系数因子

得到所需因子后，我们参考东方证券的研报中的因子测试方法构建了相应的因子测试框架对上述因子进行测试。

PS：[1]处需要说明的是，我们在文章中用的是非参数的方法对阿基米德copula函数族的参数进行估计，因此Gumbel copula的参数θ1和Clayton copula的参数θ2都是由τ计算出来的，因此该方法下上下尾相关系数因子的rank排序在Kendall秩相关系数大于0的情形下是完全一致的，即原始的上下尾相关系数因子的IC测试结果和简单等权构造的组合的表现应该是几乎一致的。


'''


# 因子数据准备，上/下尾相关性，3个月特异度，市值，行业
# 上下尾相关性因子
lambda_up = up_corr_factor[['tradeDate', 'secID', '000985']]
lambda_up.rename(columns={"tradeDate": "date", "secID": "code", "000985": "lambda_up"}, inplace=True)
lambda_down = low_corr_factor[['tradeDate', 'secID', '000985']]
lambda_down.rename(columns={"tradeDate": "date", "secID": "code", "000985": "lambda_down"}, inplace=True)
lambda_factor = pd.merge(lambda_up, lambda_down, on=['date', 'code'])
# 特异度、市值、行业
tic = time.time()
all_factor = []
date_list = sorted(lambda_factor['date'].unique())
for date in date_list:
    tdata = lambda_factor[lambda_factor['date'] == date]
    code = list(tdata['code'])
    # 特异度
    temp = cal_specificity(date, code, mkt_info)
    tdata = pd.merge(tdata, temp, on=['date', 'code'], how='left')
    # 对数市值
    temp = cal_lnmkt(date, code)
    tdata = pd.merge(tdata, temp, on=['date', 'code'], how='left')
    # 行业
    temp = get_industry(date, code)
    tdata = pd.merge(tdata, temp, on=['date', 'code'], how='left')    
    all_factor.append(tdata)
    
all_factor = pd.concat(all_factor)
all_factor.to_csv(path + 'raw_factor.csv', index=False, encoding='gbk')
toc = time.time()
print('***********原始因子示例************')
print(all_factor.head(10).to_html())
print ("\n ----- factor Computation time = " + str((toc - tic)) + "s")

# 因子预处理
tic = time.time()
factor_stand = []
all_factor['lambda_up_neu'] = all_factor['lambda_up']
all_factor['lambda_down_neu'] = all_factor['lambda_down']
all_factor['lambda_up_neu_ivr'] = all_factor['lambda_up']
all_factor['lambda_down_neu_ivr'] = all_factor['lambda_down']
factor_list = ['lambda_up', 'lambda_up_neu', 'lambda_up_neu_ivr', 'lambda_down', 'lambda_down_neu', 'lambda_down_neu_ivr',
               'IVR', 'mkt'] # 因子集合
all_factor = all_factor[['date', 'code'] + factor_list + ['industry']]
date_list = sorted(all_factor['date'].unique())
for date in date_list:
    tdata = all_factor[all_factor['date'] == date]
    tdata.reset_index(drop=True ,inplace=True)
    # 缺失值填充
    for factor_name in factor_list:
        tdata = nafill_by_sw1(tdata, factor_name)
    tdata = tdata.dropna()
    # 将行业转换成虚拟变量
    indu_dummies = pd.get_dummies(tdata['industry'])
    del tdata['industry']
    tdata = pd.concat([tdata, indu_dummies], axis=1)
    # 先对市值标准化，方便后续其他因子的中性化
    tdata = factor_process('mkt', tdata, 'no')
    # 其他因子
    for factor_name in ['lambda_up_neu', 'lambda_down_neu', 'lambda_up_neu_ivr', 'lambda_down_neu_ivr', 'IVR']:
        tdata = factor_process(factor_name, tdata, 'yes')
    for factor_name in ['lambda_up', 'lambda_down']:
        tdata = factor_process(factor_name, tdata, 'no')
    # copula相关系数对IVR中性化
    y1 = np.array(tdata['lambda_up_neu_ivr'])   
    y2 = np.array(tdata['lambda_down_neu_ivr']) 
    x = np.array(tdata['IVR'])        
    x = sm.add_constant(x, has_constant='add')        
    model_up = sm.OLS(y1, x, missing='drop').fit()
    tdata['lambda_up_neu_ivr'] = model_up.resid          
    model_down = sm.OLS(y2, x, missing='drop').fit()
    tdata['lambda_down_neu_ivr'] = model_down.resid
    factor_stand.append(tdata)

factor_stand = pd.concat(factor_stand)
    
month_ret = []
date_list = sorted(list(set(factor_stand['date'])))
for i in range(len(date_list) - 1):
    this_month = date_list[i]
    next_month = date_list[i + 1]
    code = list(factor_stand[factor_stand['date'] == this_month]['code'])
    ret = cal_month_ret(code, this_month, next_month)
    month_ret.append(ret)
    
month_ret = pd.concat(month_ret) 

# 标准化因子存储  
factor_stand.sort_values(by=['date', 'code'])
factor_stand.reset_index(drop=True, inplace=True)
n = list(factor_stand.columns).index('mkt')
factor_stand = factor_stand[factor_stand.columns[: n + 1]]
factor_stand = pd.merge(factor_stand, month_ret, on=['date', 'code'], how='left')
factor_stand.fillna(0, inplace=True)    
factor_stand.to_csv(path + 'factor_stand.csv', index=False)
toc = time.time()
# print('***************标准化因子示例***************')
# print(factor_stand.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''

2.1.1 上下尾相关系数原始因子测试

回测结论：如上所述，由于两者的rank次序的一致性，所以原始的上下尾相关性因子的表现几乎一致，原始的上下尾选股因子在中证全指内具有非常显著的选股效果，rankIC达到了4.7%，从rankIC的时序图上可见该因子的IC稳定性也很好，ICIR有1.58，多空组合较为稳健，唯一的缺点是分组的单调性略微欠缺，而且和常规的技术类因子类似，空头端的负超额收益较强。

'''


# 上尾相关系数因子中证全指内测试结果
factor_name = 'lambda_up'
n = 10
result, IC_series, long_short_df, bucket_excess = factor_test(factor_stand, factor_name, n)
print (result.to_html())
test_result_show(factor_name, IC_series, long_short_df, bucket_excess)


# 下尾相关系数因子中证全指内测试结果
factor_name = 'lambda_down'
n = 10
result, IC_series, long_short_df, bucket_excess = factor_test(factor_stand, factor_name, n)
print (result.to_html())
test_result_show(factor_name, IC_series, long_short_df, bucket_excess)

'''
2.1.2 行业市值中性化的上下尾相关系数因子测试

回测结论：和原始因子相比，风格中性化后的因子的选股能力更强，以上尾因子为例，风格中性化后的因子的rankIC值从4.73%提升至5.38%，ICIR从1.58提升至2.29，且在回测期间内，IC正向的比例达到71.6%，多空组合的净值曲线的回撤大幅降低，且分组收益完全线性，多头超额收益也进一步提升，风格中性化后的λU和λL是一个具有显著稳定选股效果的Alpha因子。

'''

# 行业市值中性化后上尾相关系数因子中证全指测试结果
factor_name = 'lambda_up_neu'
n = 10
result, IC_series, long_short_df, bucket_excess = factor_test(factor_stand, factor_name, 10)
print (result.to_html())
test_result_show(factor_name, IC_series, long_short_df, bucket_excess)

# 行业市值中性化后下尾相关系数因子中证全指测试结果
factor_name = 'lambda_down_neu'
n = 10
result, IC_series, long_short_df, bucket_excess = factor_test(factor_stand, factor_name, 10)
print (result.to_html())
test_result_show(factor_name, IC_series, long_short_df, bucket_excess)

# 与其他因子的相关性
factors = pd.read_csv('copula_corr_factor/factor_stand.csv')
factors['date'] = map(str, factors['date'])
del factors['IVR']

factors = pd.merge(factors, factor_stand[['date', 'code', 'IVR', 'lambda_up_neu', 'lambda_down_neu']],
                   on=['date', 'code'])
# 因子相关性矩阵
date_list = sorted(factors['date'].unique())
factor_list = [x for x in factors.columns if x not in ['date', 'code', 'Month_ret', 'mkt']]
mean_corr = pd.DataFrame()
for i in range(len(date_list)):
    tdata = factors[factors['date'] == date_list[i]]
    corr_frame = tdata[factor_list].corr()    
    if i == 0:
        mean_corr = corr_frame
    else:
        mean_corr += corr_frame

mean_corr = mean_corr / len(date_list)
mean_corr = mean_corr.round(2)

f, ax= plt.subplots(figsize = (20, 10))
_ = sns.heatmap(mean_corr, alpha=1.0, annot=True, center=0.0, annot_kws={"size": 8}, linewidths=0.02, 
                linecolor='white', linewidth=0,  ax=ax)
title=u'行业市值中性化后各因子的相关性矩阵'
_ = ax.set_title(title, fontproperties=font, fontsize=16)

'''

相关性分析结果

从结果来看，风格中性后的相关性因子和估值以及成长类因子的相关性非常低；

尾部相关性因子基于价格数据构建，因此和反转因子、换手率因子、波动因子之间具有一定的相关性（25%-40%）；

上下尾相关因子和三个月特异度因子的相关性接近80%，这是因为特异度因子是股票收益率通过Fama-French三因子回归后的1-R2，而在三因子回归中，市场的收益率是占据主导的，而同时上下尾相关性与常规的线性相关也比较接近，所以会与特异度因子有很高的负向相关性。

那么相关系数因子相对于特异度因是否能提供额外的增量Alpha？

我们尝试通过Fama-Macbeth回归把风格中性后的三个月特异度从风格中性后的上下尾相关系数因子中剔除，来考察残差的选股效果，结果如下。

'''

# 剥离了3个月特异度后上尾相关系数因子表现
factor_name = 'lambda_up_neu_ivr'
n = 10
result, IC_series, long_short_df, bucket_excess = factor_test(factor_stand, factor_name, n)
print (result.to_html())
test_result_show(factor_name, IC_series, long_short_df, bucket_excess)

# 剥离了3个月特异度后下尾相关系数因子表现
factor_name = 'lambda_down_neu_ivr'
n = 10
result, IC_series, long_short_df, bucket_excess = factor_test(factor_stand, factor_name, n)
print (result.to_html())
test_result_show(factor_name, IC_series, long_short_df, bucket_excess)

'''

第三部分：上尾相关系数因子改进：上尾异常相关系数因子
该部分耗时 10分钟
该部分内容为：

上尾异常相关系数因子构造

上尾异常相关系数因子测试

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


3.1 上尾异常相关系数因子构造

上尾异常相关系数
从上面的结果来看，仅仅研究股票和市场的上尾或下尾相关系数并不能带来显著的增量Alpha，上面我们只考虑了股票和市场单边的相关关系，即：

上尾相关系数高说明在市场异常上涨的时候股票也能有大的涨幅；
下尾相关系数高说明在市场异常下跌的时候股票也可能有较大的跌幅；
而我们期望找到在市场上涨的时能跟着上涨而下跌时不跟着下跌的股票，所以我们要找到一个综合考虑上下尾相关系数的因子，定义如下：


其中ϵt就是上尾相关系数剥离掉下尾相关系数的残差，这个残差就是我们想要找的上尾异常相关系数

'''


# 计算上尾异常相关系数因子
tic = time.time()
date_list = sorted(all_factor['date'].unique())
up_idio_lambda = []
for date in date_list:
    tdata = all_factor[all_factor['date'] == date]
    tdata.reset_index(drop=True ,inplace=True)
    tdata['up_idio_lambda'] = 1
    tdata['up_idio_lambda_neu'] = 1
    tdata = tdata[['date', 'code', 'lambda_up', 'lambda_down', 'up_idio_lambda', 'up_idio_lambda_neu', 'mkt', 'industry']]
    # 去极值
    alpha_factors = ['lambda_up', 'lambda_down']
    tdata.loc[:, alpha_factors] = tdata.loc[:, alpha_factors].apply(lambda x : winsorize_by_date(x))
    # 计算上尾异常相关性
    y = np.array(tdata['lambda_up'])
    x = np.array(tdata['lambda_down'])
    x = sm.add_constant(x, has_constant='add')
    result = sm.OLS(y, x, missing='drop').fit()
    tdata['up_idio_lambda'] = result.resid
    tdata['up_idio_lambda_neu'] = result.resid
    # 将行业转换成虚拟变量
    indu_dummies = pd.get_dummies(tdata['industry'])
    del tdata['industry']
    tdata = pd.concat([tdata, indu_dummies], axis=1)
    # 先对市值标准化，方便后续其他因子的中性化
    tdata = factor_process('mkt', tdata, 'no')
    # 中性化
    tdata = factor_process('up_idio_lambda_neu', tdata, 'yes')            
    tdata = tdata[['date', 'code', 'up_idio_lambda', 'up_idio_lambda_neu']]
    up_idio_lambda.append(tdata)

up_idio_lambda = pd.concat(up_idio_lambda)
up_idio_lambda = pd.merge(up_idio_lambda, factor_stand[['date', 'code', 'Month_ret']], on=['date', 'code'])
toc = time.time()
print('***********上尾异常相关系数因子示例************')
print(up_idio_lambda.head(10).to_html())
print ("\n ----- factor Computation time = " + str((toc - tic)) + "s")


'''

3.2 上尾异常相关系数因子测试

我们分别对原始的上尾异常相关系数因子和行业市值中性化后的上尾异常相关系数因子做如上的测试，结论如下：

原始的上尾异常相关系数因子的选股效果一般， 多空收益和分组的线性都不如原始的上尾相关系数因子；

行业市值中性化后的上尾异常相关系数因子的IC值为2.45%，IC_IR达到1.78，从IC_IR及多空组合的净值曲线来看，具有不错的选股效果，但是缺点是多头组的收益欠佳，相比于上述行业市值中性化后的上尾相关系数因子而言，该因子不适合用于单独选股；

我们考察了风格中性后的因子与其他因子的相关性，可以看到，因子与IVR因子的相关性为从78%骤降至36%，且除了IVR外与其他因子的相关性都在20%以下，说明该因子在原有因子的基础上提供了增量的Alpha信息，可以放入多因子模型增强模型的选股效果!

 注意事项 

相关性分析的代码依赖之前深度报告中的因子数据，如无相关因子数据，请不要运行下方相关性分析代码！


'''


# 上尾异常相关系数在中证全指的表现
factor_name = 'up_idio_lambda'
n = 10
result, IC_series, long_short_df, bucket_excess = factor_test(up_idio_lambda, factor_name, n)
print (result.to_html())
test_result_show(factor_name, IC_series, long_short_df, bucket_excess)


# 行业市值中性化后上尾异常相关系数在中证全指的表现
factor_name = 'up_idio_lambda_neu'
n = 10
result, IC_series, long_short_df, bucket_excess = factor_test(up_idio_lambda, factor_name, n)
print (result.to_html())
test_result_show(factor_name, IC_series, long_short_df, bucket_excess)

# 看一下上尾异常相关因子和其他因子的相关性
factors = pd.read_csv('copula_corr_factor/factor_stand.csv')
factors['date'] = map(str, factors['date'])
del factors['IVR']

factors = pd.merge(factors, factor_stand[['date', 'code', 'IVR']], on=['date', 'code'])
factors = pd.merge(factors, up_idio_lambda[['date', 'code', 'up_idio_lambda', 'up_idio_lambda_neu']], on=['date', 'code'])
# 因子相关性矩阵
date_list = sorted(factors['date'].unique())
factor_list = [x for x in factors.columns if x not in ['date', 'code', 'Month_ret', 'mkt']]
mean_corr = pd.DataFrame()
for i in range(len(date_list)):
    tdata = factors[factors['date'] == date_list[i]]
    corr_frame = tdata[factor_list].corr()    
    if i == 0:
        mean_corr = corr_frame
    else:
        mean_corr += corr_frame

mean_corr = mean_corr / len(date_list)
mean_corr = mean_corr.round(2)

f, ax= plt.subplots(figsize = (20, 10))
_ = sns.heatmap(mean_corr, alpha=1.0, annot=True, center=0.0, annot_kws={"size": 8}, linewidths=0.02, 
                linecolor='white', linewidth=0,  ax=ax)
title=u'行业市值中性化后各因子的相关性矩阵'
_ = ax.set_title(title, fontproperties=font, fontsize=16)


'''

样本空间测试

我们进一步在沪深300和中证500内分别对因子做测试：

该因子作为一个价量因子，在沪深300内的选股效果很一般（下边不做展示）；

因子在中证500内的IC值为3.09%，IC_IR为1.46，且分组效果相对中证全指中更为单调，选股效果较为稳健，如下所示。

'''

# 中证500股票池筛选（沪深300的结果经测试一般，这里不做展示）
universe_500 = []
for date in date_list:
    tdata = up_idio_lambda[up_idio_lambda['date'] == date]
    universe = set_universe('ZZ500', date)
    tdata = tdata[tdata['code'].isin(universe)]
    universe_500.append(tdata)

factor_in_500 = pd.concat(universe_500)

# 上尾异常相关系数因子在中证500内的表现
factor_name = 'up_idio_lambda'
n = 5
result, IC_series, long_short_df, bucket_excess = factor_test(factor_in_500, factor_name, n)
print (result.to_html())
test_result_show(factor_name, IC_series, long_short_df, bucket_excess)

# 行业市值中性化后的上尾异常相关系数因子在中证500内的表现
factor_name = 'up_idio_lambda_neu'
n = 5
result, IC_series, long_short_df, bucket_excess = factor_test(factor_in_500, factor_name, n)
print (result.to_html())
test_result_show(factor_name, IC_series, long_short_df, bucket_excess)

'''


结论

我们基于copula方法度量股票和市场之间的尾部相关性，从过去10年来看，股票和市场的下尾相关系数均值要高于上尾相关系数均值，这说明市场极端上涨时股票同涨的可能性要小于市场极端下跌时股票同跌的可能性，这点与其他的实证研究是一致的；
基于过去3个月收益率数据构建的上下尾相关系数因子原始值和风格中性因子值在2006.1-2019.7均有比较好的选股效果，其中风格中性后的上尾相关系数因子rankIC为5.38%，IC_IC达到2.29，多空收益非常稳定，且分组收益完全单调，但是该因子和3个月特异度的相关性很高，在剥离了特异度因子后没有显著的选股效果，说明单纯地观察尾部相关系数并不能带来独立的Alpha源；
在此基础上，我们结合上下尾相关系数的特性构建了上尾异常相关系数因子，风格中性后的上尾异常相关系数因子IC为2.45%，但IC_IR达到1.78，具有不错的选股效果，缺点是多头组的收益欠佳，因此不适合单独选股，但在中证500空间内的选股效果出色，我们也考察了其与常见因子的相关性，除了与3个月IVR的相关性为36%以外，与其余因子的相关性都在20%以下，说明该因子在原有因子的基础上提供了增量信息，可以放入多因子模型增强模型的选股效果！

'''