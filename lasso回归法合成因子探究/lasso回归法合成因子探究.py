# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:25:38 2020

@author: Asus
"""

'''
导读
A. 研究目的： 本文参考国盛证券《Alpha因子高维度与非线性问题——基于Lasso的收益预测模型》中的研究方法，对几种因子合成方法进行了研究，并用优矿的行情数据和优矿因子库中的因子进行了因子合成效果的测试，对比不同方法的优缺点和收益、风险表现

B. 研究结论：

本文选取了59个因子，研究了使用传统ICIR方法、lasso方法、adaptive lasso方法、group lasso方法进行因子合成的效果（20140101到20181231）
用ICIR方法合成，第一组超额收益为39%，信息比率0.91，ICIR值1.08
lasso方法效果优于ICIR方法，第一组超额收益为53%，信息比率为1.22，ICIR值为1.39
adaptiv lasso方法同lasso收益接近，但在因子筛选上有所提升，将lasso方法中的54个有效因子压缩到了21个
group lasso则考虑了因子与收益中可能存在的非线性关系，虽然group lasso方法ICIR仅为0.9，逊色于lasso方法的ICIR 1.24，但在市场处于非线性关系稳定时期，group lasso法相对其他方法存在明显收益。综合来看第一组年化超额收益达到63%，远超其它方法的水平。
C. 文章结构：本文共分为四个部分，具体如下：

一、数据准备和数据处理方法详解
二、ICIR方法合成因子
三、lasso回归和ada_lasso回归方法合成因子
四、group lasso方法合成因子
D. 运行时间说明:

第一部分运行需要5分钟
第二部分运行需要10分钟
第三部分运行需要55分钟
第四部分运行需要50分钟
总耗时：2小时左右
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
https://uqer.io/community/share/YID08xqxvohJ4tjxvyVrD7Bb53g0/private；密码：1507
请在运行之前，克隆上面的代码，并存成lib（右上角->另存为lib,不要修改名字）

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

第一部分：数据准备和数据处理方法详解
该部分耗时5分钟
该部分的主要内容为：

从优矿因子库中选取59个常用因子，并标注该因子所属大类，取得2010-2018年全A股月度因子值
调整因子方向，先验的将负向因子调正
对因子数据进行标准化、MAD法去极值等数据预处理
获取个股的月度收益数据

'''


##获取筛选出的因子数据
# coding: utf-8

import pandas as pd
import numpy as np
import os
import time
import math

import sys
sys.path
sys.path.append('G:\dropbox\Dropbox\Dropbox\project folder from my asua computer\Project\lib')

import quant_util as quant_util
from numpy import linalg as LA
import statsmodels.api as sm

import seaborn as sns


import matplotlib.pyplot as plt
#from CAL.PyCAL import *    # CAL.PyCAL中包含font

# 取因子数据日期区间
start_date = '20100101'
end_date = '20181231'

# 回测起始时间
backtest_start = '20140101'
# 计算ICIR需要历史长度的数据，文中最长为24个月，因此截断数据的时间为回测开始时间前2年
backtest_data_start = '20120101'


# 因子文件存放目录，如果目录不存在，程序自动新建一个
raw_data_dir = "./uqer_report_201904"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)
    
    
# 设定用到的因子池及所属的大类
factor_dict = {
   "MLEV":"杠杆",
    "NOCFToTLiability":"杠杆",
    "NOCFToInterestBearDebt":"杠杆",
    "OperCashInToCurrentLiability":"杠杆",
    "EquityFixedAssetRatio":"杠杆",
    "CMRA":"波动率",
    "DASTD":"波动率",
    "DVRAT":"波动率",
    "Variance60":"波动率",
    "HSIGMA":"波动率",
    "DEGM":"成长",
    "EARNMOM":"成长",
    "NPParentCompanyCutYOY":"成长",
    "NPParentCompanyGrowRate":"成长",
    "OperatingProfitGrowRate":"成长",
    "OperatingRevenueGrowRate":"成长",
    "OperCashGrowRate":"成长",
    "TotalProfitGrowRate":"成长",
    "CETOP":"价值",
    "ETOP":"价值",
    "ETP5":"价值",
    "PCF":"价值",
    "PCFIndu":"价值",
    "PEG3Y":"价值",
    "PEG5Y":"价值",
    "PEHist120":"价值",
    "PS":"价值",
    "ILLIQUIDITY":"流动性",
    "TOBT":"流动性",
    "STOA":"流动性",
    "STOM":"流动性",
    "STOQ":"流动性",
    "ROECut":"盈利",
    "NetProfitRatio":"盈利",
    "ROAEBIT":"盈利",
    "SUOI":"盈利",
    "REVS250":"动量",
    "RSTR12":"动量",
    "Price3M":"动量",
    "Price1Y":"动量",
    "DAREV":"分析师",
    "DASREV":"分析师",
    "EPIBS":"分析师",
    "FEARNG":"分析师",
    "FY12P":"分析师",
    "GREC":"分析师",
    "DividendPS":"红利",
    "DividendPaidRatio":"红利",
    "CashDividendCover":"红利",
    "CTOP":"红利",
    "CTP5":"红利",
    "DividendCover":"红利",
    "CFO2EV":"质量",
    "NOCFToOperatingNILatest":"质量",
    "OperCashInToAsset":"质量",
    "TA2EV":"质量",
    "TaxRatio":"质量",
    "CashRateOfSalesLatest":"质量",
    "NetProfitCashCover":"质量"
    
}

'''
1.1 获取因子数据并预处理

提取数据并对因子数据方向调整、标准化、空值填充、去极值处理
最终的因子数据存储于[raw_data_dir]/raw_factor_data.pickle


'''


# 将负向因子调正
inv_factors = {
    "reciprocal":['PE', 'PB', 'PCF', 'PS', 'PEIndu', 'PBIndu', 'PCFIndu', 'PEHist120', 'PSIndu'],  #倒数
    "reverse":['Price1M', 'Price3M','RSTR12','Price1Y','RC12', 'RSTR24', 'REVS250','HSIGMA', 'CMRA', 'DASTD', 'Variance60', 'DVRAT'] # 负数
}


    
# 所有的因子名列表
all_factors = factor_dict.keys()
# 因子和对应的大类df
factor_group_df = pd.DataFrame()
tcount = 0
for fname, ftype in factor_dict.items():
    factor_group_df.loc[tcount, 'factor_name'] = fname
    factor_group_df.loc[tcount, 'factor_type'] = ftype
    tcount += 1

##################################################### 第一步：取因子库中存在的因子 ##############################################
factor_pool_items = all_factors

def get_calender_range(begin, end):
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" and tradeDate <="%s" and tradeDate >="%s" order by tradeDate'''%(begin, end)
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

def get_month_calender(start_date, end_date):
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" and endDate>="%s" and endDate <="%s" order by endDate'''%(start_date, end_date)
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

# 拿到交易日历，得到月末日期
trade_date = get_month_calender(start_date, end_date)
def get_IdxCons(intoDate,ticker='000300'):
    #nearst 时间
    sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
            and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
        ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' %(ticker,
        ticker,intoDate)
    x = pd.read_sql(sql_str1,engine)
    x = x['symbol'].values   
    return x
set_universe = get_IdxCons(end_date, ticker ='000001')
# 得到因子库的因子数据
factor_list = quant_util.get_data_items(set_universe, trade_date, factor_pool_items)
factor_frame = pd.concat(factor_list, axis=0)


print (u'对因子进行方向调整、标准化、空值填充、去极值处理...')
for inv_type in inv_factors:
    # 取倒数
    if inv_type == 'reciprocal':
        for factor_name in inv_factors[inv_type]:
            if factor_name not in factor_frame.columns:
                continue
            factor_frame[factor_name] = 1.0/factor_frame[factor_name]
    # 取负数
    elif inv_type == 'reverse':
        for factor_name in inv_factors[inv_type]:
            if factor_name not in factor_frame.columns:
                continue
            factor_frame[factor_name] = -1.0*factor_frame[factor_name]

#对因子数据进行行业中性及极值处理

# 标准化: 行业内zscore
factor_frame = quant_util.zscore_by_indu(factor_frame, all_factors)

# 空值填充
# 其它因子的空值用行业中位数
factor_frame = quant_util.fillna_indu_median(factor_frame, all_factors)

# 极值处理： 用MAD法，拉回3倍标准差外的值
factor_frame = quant_util.mad_winsorize(factor_frame, all_factors, sigma_n=3)
factor_frame.dropna(subset=['ticker'], inplace=True)

del factor_frame['industryName1']
all_factors = [x for x in factor_frame.columns if x not in ['ticker', 'tradeDate']]
# 数据存储下来
factor_frame.to_pickle(os.path.join(raw_data_dir, "raw_factor_data.pickle"))

factor_frame.head()


# 拿到交易日历，得到月末日期
trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
trade_date = list(trade_date[trade_date.isMonthEnd == 1]['calendarDate'].apply(lambda x:x.replace('-','')))
month_return=DataAPI.MktEqumGet(secID=u"",ticker=u"",monthEndDate=u"",beginDate=start_date,endDate=end_date,isOpen=u"",field=u"",pandas="1")[['ticker','endDate','return']]

# 使收益率变为未来一月的收益率，此处的month_return是按时间从晚到早排列的
month_return['return']=month_return['return'].shift(1) 

month_return.sort(['ticker','endDate'],ascending=[1,1],inplace=True) # 转换成从早到晚排列的形式
month_return.columns=['ticker','tradeDate','target_return']
month_return['tradeDate']=month_return['tradeDate'].apply(lambda x:x.replace('-',''))
month_return.head()

'''

第二部分：ICIR方法合成因子
该部分耗时10分钟
该部分的主要内容为：
遍历参数组(N, M, K)，用ICIR方法合成因子后，测试各参数组下的组合表现，ICIR方法合成因子的具体方法为：

计算因子每月的IC值及过去N个月的因子IC_IR值
设定ICIR阈值k，筛选出IC_IR>K的因子
根据因子所属大类合成大类因子：同组因子等权合成为对应大类因子，并进行zscore标准化处理
计算大类因子过去M个月的ICIR值，并根据ICIR值进行加权得到最后的综合因子
测试得到的综合因子的效果
N取值为12和24，M取值为24个月、12个月、6个月，K取值为0.4, 0.6, 0.8, 最后进行组合得到(N, M, K)参数组，要求M小于等于N

'''
# 计算得到因子每个月的IC值
factor_ic_frame = quant_util.calc_ic(factor_frame, month_return, all_factors)
factor_ic_frame.sort_values(by=['tradeDate'], inplace=True)
factor_ic_frame.tail()

# ICIR法合成因子的函数
def icir_combine_factor(factor_ic_frame, N, K, M):
    '''
    factor_ic_frame: 各个因子的月度IC值，dataframe，列为：tradeDate, [各个因子名]，值为因子在每期的IC值
    N, M: 计算ICIR时的时间窗口
    K: 挑选因子的ICIR阈值
    return: 
    total_df:合成后的因子值，dataframe格式，列为：ticker, tradeDate, value
    factor_count_df: 每一期用到的因子个数，列为： tradeDate, variable(因子个数)
    '''
    # 计算每个因子过去N个月的ICIR
    factor_icir_frame = factor_ic_frame.copy()
    factor_icir_frame[all_factors] = factor_ic_frame[all_factors].shift(1).rolling(window=N, min_periods=1).apply(
            lambda x: x.mean() / x.std())

    icir_filter_frame = pd.melt(factor_icir_frame, id_vars=['tradeDate'], value_vars=all_factors, value_name='ICIR').dropna()

    # # 只要ICIR > K的因子
    icir_filter_frame = icir_filter_frame.query("ICIR>=@K")

    # 增加保留下来因子所属的大类
    icir_filter_frame = icir_filter_frame.merge(factor_group_df.rename(columns={"factor_name":"variable"}), on=['variable'], how='left')
    # 每一期用到的因子数
    factor_count_df = icir_filter_frame.groupby(['tradeDate'])['variable'].count().reset_index()
    
    # 增加每期，因子在大类中的权重
    factor_group_weight = icir_filter_frame.groupby(['tradeDate', 'factor_type'])['variable'].count().reset_index()
    factor_group_weight['weight'] = 1.0/factor_group_weight['variable']
    factor_group_weight = factor_group_weight[['tradeDate', 'factor_type', 'weight']]

    # 计算每一期的大类因子值
    icir_filter_frame = icir_filter_frame.merge(factor_group_weight, on=['tradeDate', 'factor_type'], how='left')

    group_df_list = []
    # 遍历每一个大类， 计算得到合成后的大类因子值
    for group_type in np.unique(factor_group_df.factor_type.values):
        group_weight_df = icir_filter_frame.query("factor_type==@group_type")
        # 大类中，子因子的权重
        group_weight_df = group_weight_df.pivot(index='tradeDate', columns='variable', values='weight').reset_index()
        # 该大类的因子值dataframe
        group_factor_list = factor_group_df.query("factor_type==@group_type")['factor_name'].tolist()
        group_factor_df = factor_frame[['ticker', 'tradeDate']+group_factor_list]
        # 构造一个dataframe，同因子值dataframe一模一样, 值为因子的对应权重
        weight_df_header = factor_frame[['ticker', 'tradeDate']]
        weight_df = weight_df_header.merge(group_weight_df, on=['tradeDate'], how='left')
        # 补充上没用到的因子权重，0
        useless_factors = [x for x in group_factor_list if x not in weight_df.columns]
        for ufactors in useless_factors:
            weight_df[ufactors] = np.NAN
        weight_df = weight_df[['ticker', 'tradeDate']+group_factor_list]
        # 因子值乘以对应的因子权重，得到合成后的大类因子值
        factor_weighted_df = weight_df.set_index(['ticker', 'tradeDate'])*group_factor_df.set_index(['ticker', 'tradeDate'])
        factor_weighted_df.reset_index(inplace=True)
        factor_weighted_df[group_type] = factor_weighted_df[group_factor_list].sum(axis=1)
        factor_weighted_df = factor_weighted_df[['ticker', 'tradeDate', group_type]].set_index(['ticker', 'tradeDate'])
        group_df_list.append(factor_weighted_df)
    group_df = pd.concat(group_df_list, axis=1).reset_index()

    # 得到大类因子每个月的IC
    factors = np.unique(factor_group_df.factor_type.values)
    group_factor_ic_frame = quant_util.calc_ic(group_df, month_return, factors)
    group_factor_ic_frame.sort_values(by=['tradeDate'], inplace=True)

    # # # 得到大类因子过去M个月的ICIR
    group_factor_ic_frame[factors] = group_factor_ic_frame[factors].shift(1).rolling(window=M, min_periods=2).apply(
            lambda x: np.nanmean(x) / np.nanstd(x))
    global coefs_icir
    coefs_icir=group_factor_ic_frame.fillna(0)
    
    # 按各个大类因子值进行zscore标准化
    def normal_zscore_frame(df, col_list):
        df[col_list] = (df[col_list] - df[col_list].mean()) / df[col_list].std()
        return df

    group_df = group_df.groupby(['tradeDate']).apply(normal_zscore_frame, factors)
    group_factor_icir_frame = group_factor_ic_frame
    # 根据大类因子的ICIR值合成最后的因子
    group_factor_icir_frame['ICIR_total'] = group_factor_icir_frame[factors].fillna(0).sum(axis=1)
    for tfactor in factors:
        group_factor_icir_frame[tfactor] = group_factor_icir_frame[tfactor].fillna(0)/group_factor_icir_frame['ICIR_total']
    group_factor_icir_frame = group_factor_icir_frame[['tradeDate']+list(factors)]
    coefs_icir.index=group_factor_icir_frame.index
    group_header = group_df[['ticker', 'tradeDate']]
    group_weighted_df = group_header.merge(group_factor_icir_frame, on=['tradeDate'], how='left')

    combine_df = group_weighted_df.set_index(['ticker', 'tradeDate'])[factors]*group_df.set_index(['ticker', 'tradeDate'])[factors]
    combine_df=combine_df.reset_index()
    total_df=combine_df[['ticker','tradeDate']]
    total_df['value']=combine_df[factors].fillna(0).sum(axis=1)
    return total_df, factor_count_df



def quick_cal_perf(factor_df, month_return):
    '''简易统计因子表现, 统计区间由factor_df的最大、最小日期决定
    factor_df: 因子值dataframe， 列为 ticker, tradeDate, value, 分别代表股票代码，时间（必须月度），因子值
    month_return: 个股的月度收益dataframe， 列为：ticker, tradeDate, [下一期的收益]
    return:
    收益率、波动率、信息比率、IC与IC_IR值
    '''
    factor_df.sort(['ticker','tradeDate'],ascending=[1,1],inplace=True)
    return_loc=month_return[(month_return.tradeDate>=factor_df.iloc[0,1]) & (month_return.tradeDate<=factor_df.iloc[-1,1])]
    ans=quant_util.calc_ic(factor_df,return_loc,['value']) # 计算因子的IC
    IC=ans.mean()
    IC_IR=ans.mean()/ans.std()
    # 十分组，因子值最大的一组表现
    perf,bt_df=quant_util.easy_backtest(factor_df,return_loc,'value','target_return','long_only',ngrp=10)
    # 基准收益率
    Idx=DataAPI.MktIdxmGet(beginDate=factor_df.iloc[0,1],endDate=factor_df.iloc[-1,1],indexID=u"000001.ZICN",ticker=u"",field=u"",pandas="1")['chgPct']
    # 超额收益指标
    ret=perf['period_ret']-Idx
    rev=ret.mean()*12
    vol=ret.std()*(12**0.5)
    I_ratio=rev/vol
    return_df = pd.DataFrame()
    return_df.loc[0, u'第一组年化收益'] = rev
    return_df.loc[0, u'第一组年化波动'] = vol
    return_df.loc[0, u'第一组信息率'] = I_ratio
    return_df.loc[0, u'IC'] = IC['value']
    return_df.loc[0, u'ICIR'] = IC_IR['value']
    return return_df



import time
stime = time.time()
frame_list = []
for N in [12, 24]:
    for M in [12, 24, 6]:
        if(N<M):
            continue
        for K in [0.4,0.6, 0.8]:
            # 合成因子值
            total_df, factor_count_df = icir_combine_factor(factor_ic_frame, N, K, M)
            # 回测因子收益和风险表现
            tmp_df = quick_cal_perf(total_df.query("tradeDate>=@backtest_start"),month_return)
            # 统计用到的因子个数
            use_factor_count = round(factor_count_df.query("tradeDate>=@backtest_start")['variable'].mean(), 2)
            tmp_df.loc[0, 'N'] = N
            tmp_df.loc[0, 'M'] = M
            tmp_df.loc[0, 'K'] = K
            tmp_df.loc[0, 'use_factor'] = use_factor_count
            frame_list.append(tmp_df)
icir_perf_frame = pd.concat(frame_list, axis=0)
icir_perf_frame.index = range(len(icir_perf_frame))
etime = time.time()
print ('总耗时：',etime-stime)
icir_perf_frame.sort(['ICIR'],ascending=[0])

'''

根据上表的结果，参数N=12,M=12,K=0.6时，综合因子的IC、ICIR值及信息率较高

ICIR方法合成因子是一种常用的方法，理解上比较直观，主要存在如下两个问题：

在小类因子合成大类的时候，对于因子的分类存在主观性。有时候一些因子蕴含的信息并不相同，也可能被分到一组内，最典型的例子是质量类因子中，一些财务比率的相关程度并不是很高。
在大类合成的时候，等权和ICIR加权并没有考虑大类因子间的相关性。
   调试 运行
文档
 代码  策略  文档
第三部分 lasso回归和ada_lasso回归方法合成因子
该部分耗时 25分钟
该部分包括：

3.1 用lasso法回归收益和因子值，用得到的回归模型预测未来收益率，将预测收益率作为合成后的因子，测试该因子效果
3.2 用adaptive lasso回归收益和因子值，用得到的回归模型预测未来收益率，将预测收益率作为合成后的因子，测试该因子效果
3.3 对比ICIR方法与和Adaptive Lasso回归方法


'''

# 数值运算时，存在空值会报错，填充成0
factor_frame=factor_frame.fillna(0)
trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=backtest_data_start, endDate=end_date, field=u"", pandas="1")
trade_date = list(trade_date[trade_date.isMonthEnd == 1]['calendarDate'].apply(lambda x:x.replace('-','')))

#对齐函数 -对齐因子DataFrame 与未来收益率DataFrame
def check(factor_loc,return_loc):
    '''使两列数对齐'''
    factor_loc['oldindex']=factor_loc.index
    sumup=factor_loc.merge(return_loc,on=['ticker','tradeDate'])#筛选出前M个月的因子暴露及月度未来收益率
    sumup=sumup[sumup.target_return!=0]
    sumup=sumup.set_index('oldindex')
    return_loc=sumup[['ticker','tradeDate','target_return']]
    factor_loc=sumup[['ticker','tradeDate']+all_factors]
    return factor_loc,return_loc

'''
3.1 lasso回归

回归方法在因子筛选中的作用

为规避传统ICIR方法存在的问题，采用因子池中尽可能多的信息，我们需要建立一个模型来综合所有因子的贡献，最简单的模型就是回归模型。
但由于因子数量众多，因子之间有高度的相关性，如果直接进行 OLS 回归，会产生极大的方差，那么对于因子的统计检验是不可信的，几乎不具备因子筛选的能力。
更重要的是，特征过多，数据过少，很容易造成过拟合，造成预测的失效
 Lasso 方法加入正则化项来解决线性回归问题中高维度以及变量筛选的问题，非常适合应用于因子筛选
lasso回归的损失函数为 
min||Y−Xβ||n+λ||β||1
在线性回归的损失函数上添加了L1正则项，超参数λ用于调节正则化的强度，λ大则筛选出更少的因子
具体回测步骤

使用滚动回归统计前M个月的因子数据及未来收益率求解参数为λ的lasso的参数，预测股票下月收益率
M=[12,24]
λ=[0.001，0.0005，0.0001，0.00005]

'''

def lasso_predict(M,alp,method,start):
    '''
    M: 训练集的时间窗口
    alp: 调节参数lambda
    method: lasso/ada_lasso
    start：截取时间区间的参数
    return: 
    factor_lasso:预测的因子值，dataframe格式，列为：ticker, tradeDate, value
    预测用到的平均每期因子个数
    MSE
    '''
    from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV  
    from sklearn.linear_model import LinearRegression
    factor_lasso=pd.DataFrame()
    factor_num=[]
    global coefs_lasso
    coefs_lasso=pd.DataFrame()
    for i in range(start,len(trade_date)-1):
        stdate=trade_date[i-M]
        eddate=trade_date[i]
        factor_loc=factor_frame[(factor_frame.tradeDate>=stdate) & (factor_frame.tradeDate<eddate)]
        return_loc=month_return[(month_return.tradeDate>=stdate) & (month_return.tradeDate<eddate)]
        factor_loc,return_loc=check(factor_loc,return_loc)
        if(method=='lasso'):#如果是lasso，使用sklearn中自带的lasso进行回归
            model = Lasso(alpha=alp)
            model.fit(factor_loc[all_factors],return_loc['target_return'])
        if(method=='ada_lasso'):#如果是adaptive lasso,先用OLS回归得出系数beta(j)，用这些系数调整因子暴露后再度进行lasso回归
            model=ada_lasso(factor_loc,return_loc,alp)
        coefs=model.coef_
        factor_num.append(len(coefs[coefs!=0]))
        coefs_lasso=coefs_lasso.append(pd.Series(coefs.T,index=factor_loc.iloc[:,2:].columns).rename(eddate))
        ad_factor=factor_frame[factor_frame.tradeDate==eddate]
        ad_factor['value']=model.predict(factor_frame[factor_frame.tradeDate==eddate][all_factors])
        factor_lasso=factor_lasso.append(ad_factor[['ticker','tradeDate','value']])#用模型预测的下期收益率作为因子
    factor_mse=(factor_lasso['value']-month_return['target_return']).dropna()
    return factor_lasso,sum(factor_num)/len(factor_num),factor_mse.apply(lambda x:x**2).sum()/len(factor_mse)

import time
stime = time.time()
frame_list = []
for M in [12, 24]:
    for alp in [0.001, 0.0005, 0.00005,0.0001]:
        # 合成因子值
        total_df, factor_num,factor_mse = lasso_predict(M,alp,'lasso',24)
        # 回测表现
        tmp_df = quick_cal_perf(total_df.query("tradeDate>=@backtest_start"),month_return)
        tmp_df.loc[0, 'M'] = M
        tmp_df.loc[0, 'lambda'] = alp
        tmp_df.loc[0, 'use_factor'] = factor_num
        tmp_df.loc[0, 'MSE'] = factor_mse
        frame_list.append(tmp_df)
lasso_perf_frame = pd.concat(frame_list, axis=0)
lasso_perf_frame.index = range(len(lasso_perf_frame))
etime = time.time()
print ('总耗时:',etime-stime)
lasso_perf_frame.sort(['ICIR'],ascending=[0])


'''
从上表可以看出如下结论：

当调节参数变小时，筛选得出因子数量增多，同时因子的IC_IR值在变大，这说明新增加进来的因子能提供稳定的增量信息。
但调节参数变小时，伴随着样本外均方误差上升，说明惩罚项系数减小会带来泛化性能的减弱
M=24时的ICIR普遍大于M=12时的表现，说明随着选取时间窗口的增大,因子预测效果的稳定性有所上升
当组合效果最好的时候，M=24 lambda=0.00005 此时平均选取了54个因子，一共才有58个因子，lasso方法在因子筛选上的效果并不明显
   调试 运行
文档
 代码  策略  文档
3.2 adaptive lasso回归方法 （20分钟）

Lasso并不是筛选因子很好的方法，因为Lasso对不同权重的因子相同的惩罚相同。
若调节系数过小，则系数相对分散，模型不具有很好的筛选因子效果。且容易导致过拟合，模型外误差增大。
若调节系数过大，则将无效因子压缩至0的时候，也会丢失部分的有效因子的信息。
这里使用adaptive lasso 对不同的因子根据对模型的贡献赋予不同的惩罚项权重

损失函数如下:
min||Y−Xβ||n+λ|β||β^init,j|
具体方法如下：

首先进行ols回归得到每个因子的系数 β^init,j
将这些系数的绝对值作为权重进行lasso回归，得到预测
参数组取M=[12,24] lambda=[0.0001, 0.00005, 0.00001]

'''

def ada_lasso(factor_loc,return_loc,alp):
    from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV  
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(factor_loc[all_factors],return_loc['target_return'])
    factor_save=factor_loc
    coef1=model.coef_
    for j in range(len(model.coef_)):
        factor_loc.iloc[:,j+2]=factor_loc.iloc[:,j+2]*abs(model.coef_[j])
    model = Lasso(alpha=alp)
    model.fit(factor_loc[all_factors],return_loc['target_return'])
    for j in range(len(model.coef_)):
        model.coef_[j]=model.coef_[j]*abs(coef1[j])#得出的是beta/beta(j)的系数，需要再乘以beta(j)
    factor_loc=factor_save
    return model


import time
stime = time.time()
frame_list = []
for M in [12, 24]:
    for alp in [0.0001, 0.00005, 0.00001]:
        total_df, factor_num,factor_mse = lasso_predict(M,alp,'ada_lasso',24)
        tmp_df = quick_cal_perf(total_df.query("tradeDate>=@backtest_start"),month_return)
        tmp_df.loc[0, 'M'] = M
        tmp_df.loc[0, 'alpha'] = alp
        tmp_df.loc[0, 'use_factor'] = factor_num
        tmp_df.loc[0, 'MSE'] = factor_mse
        frame_list.append(tmp_df)
ada_lasso_perf_frame = pd.concat(frame_list, axis=0)
etime = time.time()
print ('总耗时:',etime-stime)
ada_lasso_perf_frame.sort(['ICIR'],ascending=[0])

'''

从上表可以看出如下结论：

当调节参数变小时，筛选得出因子数量增多，同时因子的IC_IR值在变大，第一组超额收益增加，信息比率也随之增大。
adaptive lasso方法得到的结果与lasso方法差距不大(收益：53.2% VS 53.7%， MSE：5.1% VS 5.1%)，但可以看出，筛选出的因子少了很多，效果最好的参数组合仅剩下21个因子，比lasso方法的54个有显著提升
   调试 运行
文档
 代码  策略  文档
3.3 ICIR 方法与 Adaptive Lasso 回归的对比

该部分展示了ICIR合成方法同Adaptive Lasso回归方法的因子表现、大类因子权重和累计收益的对比

3.3.1 因子效果对比

下表是上述方法最优参数组合下的信息比率与IC_IR值
name	信息比率	IC_IR	保留因子
ICIR	0.91	1.08	13
lasso	1.21	1.39	54
ada_lasso	1.25	1.20	21
可以看出：

lasso 方法与adaptive lasso方法效果都显著优于传统的ICIR方法
adaptive lasso方法比lasso方法在筛选因子方面效果出色
   调试 运行
文档
 代码  策略  文档
3.3.2 因子权重对比

这一部分用堆积柱状图来表现不同时期各大类因子的贡献情况

'''

# 堆叠图画图
def stack_plot(coefs,):
    x=range(coefs.shape[0])
    y0=coefs.iloc[:,0]
    y1=coefs.iloc[:,1]
    y2=coefs.iloc[:,2]
    y3=coefs.iloc[:,3]
    y4=coefs.iloc[:,4]
    y5=coefs.iloc[:,5]
    y6=coefs.iloc[:,6]
    y7=coefs.iloc[:,7]
    y8=coefs.iloc[:,8]
    y9=coefs.iloc[:,9]
    fig=plt.figure(figsize=(14,5))
    ax1=fig.add_subplot(111)
    ax1.stackplot(x,y0,y1,y2,y3,y4,y5,y6,y7,y8,y9, baseline='zero',labels=['value','analyist','moment','growth','leverage','volatility','liquidity','earning','dvd','quality'],colors=['r','g','b','aliceblue','antiquewhite','black','darkcyan','magenta','gold','limegreen'])
    ax1.set_xlim(0,len(coefs)-1)
    ax1.set_ylim(0,1)
    ax1.set_xlabel(u"时间", fontproperties=font,)
    ax1.set_ylabel(u"因子贡献（百分比）", fontproperties=font,)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width , box.height* 0.8])
    ax1.legend(loc='EastOutside',bbox_to_anchor=(1.11, 1))
    label=np.linspace(0,len(coefs)-1,13)
    ax1.set_xticks(label)
    label=coefs.index[[int(math.floor(i)) for i in label]]
    ax1.set_xticklabels(label)
    
ICIR_df, factor_count_df = icir_combine_factor(factor_ic_frame, 12, 0.6,12)
ICIR_df=ICIR_df.query("tradeDate>=@backtest_start")
coefs_icir=coefs_icir.query("tradeDate>=@backtest_start")
coefs_icir=abs(coefs_icir.set_index('tradeDate'))
coefs_icir=coefs_icir.apply(lambda x:x/coefs_icir.sum(axis=1))
stack_plot(coefs_icir)
print (u'ICIR方法中各大类因子的权重时序图')

lasso_df,num,mse=lasso_predict(24,0.00001,'ada_lasso',24)
lasso_df=lasso_df.query("tradeDate>=@backtest_start")
coefs_type=pd.DataFrame(0,index=coefs_lasso.index,columns=set(factor_dict.values()))
for i in coefs_lasso.columns:
    coefs_type[factor_dict[i]]=coefs_type[factor_dict[i]]+coefs_lasso[i]
coefs_type=abs(coefs_type)
coefs_lasso1=coefs_type.apply(lambda x:x/coefs_type.sum(axis=1))
stack_plot(coefs_lasso1)
print (u'ada_lasso方法中各大类因子的权重时序图')

'''
可以看出：

在ICIR方法下，经常有一段时间一大类因子完全没有贡献，另一段时间贡献突然暴增的情况出现，并且有的时间段单个大类因子贡献超过一半，如17年的分红因子，18年的动量因子等，因子权重非常不稳定，这是由于ICIR方法将权重集中在过去几个月表现最为优秀的几个因子上，忽视了其他因子可能存在的额外信息。
而adaptive lasso方法中因子权重变动比较稳定，可以看出从16到17年成长因子的占比增加，流动性因子占比减少等状况，比较符合逻辑，一些效果一般的因子也可以提供增量信息，在因子众多时效果优势较大。
另外我们可以发现，lasso方法在leverage和quality因子上投入了大量的权重，而在ICIR中这两个因子因为单因子表现差权重不高甚至被忽视，分析师因子在lasso中占比很小，可能是由于分析师因子的增量信息大多数可以被其他因子所解释
   调试 运行
文档
 代码  策略  文档
3.3.3 累计收益对比

'''

# 回测组合表现
def get_bt_df(factor_df,month_return):
    factor_df.sort(['ticker','tradeDate'],ascending=[1,1],inplace=True)
    return_loc=month_return[(month_return.tradeDate>=factor_df.iloc[0,1]) & (month_return.tradeDate<=factor_df.iloc[-1,1])]
    # 十分组，因子值最大的一组表现
    perf,bt_df=quant_util.easy_backtest(factor_df,return_loc,'value','target_return','long_only',ngrp=10)
    return perf

#画图展示净值曲线
def line_plot(plot_df,month_return,name):
    fig=plt.figure(figsize=(14,5))
    ax1=fig.add_subplot(111)
    for i in range(len(plot_df)):
        y=get_bt_df(plot_df[i],month_return)
        ax1.plot(y['cum_ret'],label=name[i])
    ax1.legend()
    label=np.linspace(0,len(y)-1,13)
    ax1.set_xticks(label)
    label=y['tradeDate'][[int(math.floor(i)) for i in label]]
    ax1.set_xticklabels(label)

print (u'adaptive lasso和ICIR方法合成因子的组合净值对比')
line_plot([lasso_df,ICIR_df],month_return,['ada_lasso','ICIR'])

'''

可以看出在14年到15年及17年10月向后，两种方法效果相差无几，但16年1月到17年9月，adaptive lasso方法远优于ICIR方法，这可能是发生了小市值因子等失效的情况导致ICIR方法变差。
   调试 运行
文档
 代码  策略  文档
小结

从预测角度来讲，adaptive lasso同lasso方法差不多，但是从因子筛选角度来讲，Adaptive lasso方法在对因子进行筛选时更加有效
Adaptive lasso方法对收益的预测精度高于ICIR方法，构建的第一组组合收益率显著高于ICIR方法
ICIR方法更加看重前期表现优秀的单个因子，Adaptive Lasso则更加重视增量信息，两种方法谁优谁劣取决于市场环境，当几个因子表现优秀而其他因子表现很差，难以提供增量信息时，ICIR方法反而要好
   调试 运行
文档
 代码  策略  文档
第四部分 group lasso方法合成因子
该部分耗时 60分钟
该部分包括：

4.1 因子效果中的非线性问题
4.2 用group lasso法回归收益和因子值，用得到的回归模型预测未来收益率，将预测收益率作为合成后的因子，测试该因子效果
   调试 运行
文档
 代码  策略  文档
4.1 因子效果中的非线性问题

在评价因子效果的过程中,常常遇到这样的问题：因子与收益之间的关系并非线性关系，有时整体上有较为显著的正相关，但可能出现中间高两头低的情况，这时因子的第一组收益就会走平甚至出现回撤。

用多项式回归取代线性回归，因子的高次项来拟合非线性关系，这些非线性项的显著确实能够证明因子和收益之间存在非线性的关系。

但多项式次数过低会导致欠拟合。但是随着次数的增加，模型的复杂度也在剧烈迅速增加。

因此统计学家想出了一个新的方法，叫做分段多项式回归，简单来说，就是将X切分成不同的取值区间（区段），在每一个区段中分别构建多项式回归，这其实是一种化繁为简的做法，又称样条回归
用较小次数的样条回归便可以达到与高次多项式回归类似的效果
   调试 运行
文档
 代码  策略  文档
4.2 group lasso方法合成因子

4.2.1 模型假设与数据准备

我们研究的问题是求出股票收益率与因子暴露之间的非线性关系
首先，将因子分为十组，在每组内进行二次函数拟合
mts(f)=∑k=112βtskpk(f)
其中
p1(f)=1
p2(f)=f
pk(f)=max{f−tk−3,0}2k=3…L+2
相当于把因子暴露拆成12个分量 转变为59*12=708个因子值

'''

#数据处理部分:将因子暴露转换为各个分量
def max0(x):
    return max(x,0)

def percen(x,c,factor):#分为10组
    return x[factor]-np.percentile(x[factor],c)

factor_frame.index=range(factor_frame.shape[0])
factor_newframe={}
for factor in all_factors:  #将因子暴露转为12个分量
    Newfactor_loc=factor_frame[['ticker','tradeDate',factor]]
    factor_spread=Newfactor_loc[factor]
    for j in range(10):
        factor_newspread=Newfactor_loc.groupby('tradeDate').apply(percen,j*10,factor)
        if(min(factor_newspread.sum(level=0))>0):
            factor_spread=np.c_[factor_spread,factor_newspread.apply(max0)**2]
    factor_newframe[factor]=factor_spread


'''

4.2.2 模型建立与求解实现

对于如此高维的问题，我们采用lasso进行变量压缩，同时我们希望在剔除某一个因子时，将他的所有分量统统降为0。
因此使用group lasso法，将每一个因子的所有分量列为一个组，组间使用L1范数，组内使用L2范数。
损失函数如下：
min||Y−Xβ||n+λ∑s=1S||βsk||
求解方法

Yuan & Lin(2007)提出了group lasso 的一个解法：
对每一组独立地求解这一组对于总体而言最优的系数。
如果满足||XTl(y−∑k≠lXkβ^k)||<λ 则β^k=0
否则
β^l=(XTlXl+λ/||β^l||)−1XTlrl(1)
其中
rl=y−∑k≠lXk
在满足XTlXl=I时,(1)式子转为
β^l=(1−λ||vl||)vl
这里为了使用解析解法，将p1(f)=1统一独立作为一组，其他的进行正交化使之满足XTlXl=I条件

这里使用对称正交法进行正交，尽可能保留因子分量间的相关性，且能够避免常用的施密特正交法中的路径依赖问题
参数选择M=12,24 lambda=0.00005,0.00003,0.00001

'''

import numpy as np
from scipy import linalg, optimize
from scipy.linalg import *
MAX_ITER = 100          #最多进行100次

# group lasso的回归实现
def sparse_group_lasso(X, y, alpha,  groups, max_iter=MAX_ITER, rtol=1e-1,
                verbose=False):#lasso回归
    # .. local variables ..
    X, y, groups = map(np.asanyarray, (X, y, groups))
    if groups.shape[0] != X.shape[1]:
        raise ValueError('Groups should be of shape %s got %s instead' % ((X.shape[1],), groups.shape))
    w_new = np.zeros(X.shape[1], dtype=X.dtype)
    n_samples = X.shape[0]
    alpha = alpha * n_samples
    # .. use integer indices for groups ..
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]#获取每组编号

    for n_iter in range(max_iter):
        w_old = w_new.copy()
        perm = np.random.permutation(len(group_labels))
        for i in perm:#按随机顺序对所有组进行系数估计
            group = group_labels[i]
            #X_l是这组的X值
            X_l=X.T[group]                          
            w_new[group]=0
            #X*r_k=X_l.T*(y-sigma(Xk*beta(k)))        (k!=l)
            X_r_k = np.dot(X_l,y-np.dot(X,w_new))   
            # 如果X_r_k<=alpha,则该组系数为0，否则为X_r_k*(1 -  alpha / np.linalg.norm(X_r_k))
            if np.linalg.norm(X_r_k) <=  alpha:
                w_new[group] = 0.
            else:
                w_new[group] = X_r_k*(1 -  alpha / np.linalg.norm(X_r_k))
                
                assert np.isfinite(w_new[group]).all()

        norm_w_new = max(np.linalg.norm(w_new), 1e-10)
        if np.linalg.norm(w_new - w_old) / norm_w_new < rtol:
            break
    return w_new


# 对输入的list进行lowdin正交（）
def lowdin_orthog_list(x_list):
    '''
    x_list = [x1, x2, x3, ...xk], 同一个横截面上，k个因子的因子集合
    x1 = [v11, v21, v31, ...vn1], 其中一个因子集合中，n个股票的某个因子值
    return: 对应的np.array([x1, x2, x3, ...xn])
    '''
    # 对X进行均值归零化，以便于在算overlap矩阵的时候直接用cov matrix
    x_list = [x-np.array(x).mean() for x in x_list]
    
    # 矩阵格式, 格式为:
    '''
    [[v11, v21, v31, v41, ...vn1],
     [v21, v22, v32, v42, ...vn2],
     ...
     [v1k, v2k, v3k, v4k, ...vnk]
     ]
    (由于是np.array转成的matrix, 所以矩阵都是行向量模式)
    '''
    factor_array = np.array(x_list)
    cov_m = np.cov(factor_array)
    
    # overlap矩阵
    overlap_m = (len(x_list[0])-1)*cov_m
    
    # 接下来，求overlap矩阵的特征值和特征根向量，以求解过度矩阵
    eig_d, eig_u = LA.eig(overlap_m)
    eig_d = np.power(eig_d, -0.5)
    
    # 处理后的特征根对角阵
    d_trans = np.diag(eig_d)
    eig_u_T = eig_u.T
    
    # 过渡矩阵
    transfer_s = np.matrix(eig_u)*d_trans*eig_u_T
    # 最终，正交处理后的矩阵
    out_m = (np.matrix(factor_array).T*transfer_s)
    out_m = np.array(out_m.T)
    return out_m,transfer_s


# 用group lasso合成因子的实现代码
def operate(i,M,alp):
    '''
        i表示预测位置
        M是训练集时间窗口
        alp是lambda值
        return list
        list第一项是DataFrame columns=['ticker','tradeDate','value']
        第二项是非0的coefs个数
    '''
    stdate=trade_date[i-M]
    eddate=trade_date[i]
    factor_loc=factor_frame[(factor_frame.tradeDate>stdate) & (factor_frame.tradeDate<=eddate)]#包括当前天数
    return_loc=month_return[(month_return.tradeDate>stdate) & (month_return.tradeDate<=eddate)]#注意[:-1]
    factor_loc,return_loc=check(factor_loc,return_loc)
    groups=[]
    number=0
    A1=factor_loc[factor_loc.tradeDate<eddate].index
    A2=factor_loc[factor_loc.tradeDate==eddate].index
    factor_array=np.array([])
    factor_y=np.array([])
    trans_dict={}
    for factor in all_factors:
        # 正交化
        factor_spread,trans=lowdin_orthog_list(factor_newframe[factor][A1].T)
        if(factor_spread[0][0]!=factor_spread[0][0]):
            continue
        factor_spread=factor_spread.T
        groups=groups+[number]*factor_spread.shape[1]
        if(number==0):
            factor_array=factor_spread
            factor_y=lowdin_orthog_list(factor_newframe[factor][A2].T)[0].T
        else:
            factor_array=np.c_[factor_array,factor_spread]
            factor_y=np.c_[factor_y,lowdin_orthog_list(factor_newframe[factor][A2].T)[0].T]
        number=number+1
    factor_array=np.c_[factor_array,(factor_array.shape[0]**-0.5)*np.ones(factor_array.shape[0])]
    groups=groups+[number]
    factor_y=np.c_[factor_y,(factor_y.shape[0]**-0.5)*np.ones(factor_y.shape[0])]
    #A1是之前的数据的index A2是当天的数据的index
    X=factor_array                                             
    y=return_loc[return_loc.tradeDate<eddate]
    y=y['target_return']
    # group lasso法进行回归
    coefs = sparse_group_lasso(X,y, alp, groups, verbose=True)
    ad_factor=factor_loc[factor_loc.tradeDate==eddate]
    ad_factor['value']=factor_y.dot(coefs).T
    return [ad_factor[['ticker','tradeDate','value']],len(coefs[coefs!=0])]


import time
stime = time.time()
frame_list = []
M_list=[12,24]
alp_list=[0.00003,0.00001]
bestgroup_lasso=pd.DataFrame()
maxIFratio=0
for M in M_list:
    for alp in alp_list:
        ans=[]
        for i in range(M_list[-1],len(trade_date)-1):
            ans=ans+[operate(i,M,alp)]
        factor_lasso=[i[0] for i in ans]
        factor_num=[i[1] for i in ans]
        factor_grouplasso=pd.DataFrame()
        for factor in factor_lasso:
            factor_grouplasso=factor_grouplasso.append(factor)
        factor_mse=(factor_grouplasso['value']-month_return['target_return']).dropna()
        factor_num=sum(factor_num)/len(factor_num)
        factor_mse=factor_mse.apply(lambda x:x**2).sum()/len(factor_mse)
        tmp_df = quick_cal_perf(factor_grouplasso,month_return)
        tmp_df.loc[0, 'M'] = M
        tmp_df.loc[0, 'lambda'] = alp
        tmp_df.loc[0, 'use_factor'] = factor_num/10
        tmp_df.loc[0, 'MSE'] = factor_mse
        frame_list.append(tmp_df)
        if(tmp_df[u'第一组信息率'][0]>maxIFratio):
            bestgroup_lasso=factor_grouplasso
            maxIFratio=tmp_df[u'第一组信息率'][0]
group_lasso_perf_frame = pd.concat(frame_list, axis=0)
etime = time.time()
print ('总耗时:',etime-stime)
group_lasso_perf_frame.sort([u'第一组信息率'],ascending=[0])

print (u'group_lasso, adaptive lasso和ICIR方法合成因子的组合净值对比')
line_plot([bestgroup_lasso,lasso_df,ICIR_df],month_return,['group_lasso','ada_lasso','ICIR'])

'''
从回测结果看，Group lasso方法合成的因子相比于ada_lasso和ICIR方法有显著的提升：

在信息比率差不多的情况下，年化收益为65%, 远超过ada_lasso和ICIR的表现
根据净值曲线，我们大致可以发现，group lasso 带来的超额收益主要集中在15年至17年，这段时间市场存在较强的非线性关系，但其他时刻平于甚至劣于lasso，这是由于很少存在长期稳定的非线性关系，当因子从非线性转为线性时，会导致group lasso失效，从而劣于ada_lasso方法。
   调试 运行
文档
 代码  策略  文档
结论
对于机器学习类算法，令人头疼的往往是对结果的解释，哪怕是lasso类方法这样简单的方法，他的解释性也往往远逊于ICIR方法这样直观的方法。
ICIR方法是根据过去一段时间单因子的表现来筛选表现强而稳定的因子，而lasso类方法则从回归的角度来解释每一个因子对收益的贡献，group lasso更是进一步加入了非线性关系的描述。
我们用lasso方法解决高维度的问题，adaptive lasso 进一步帮助lasso做因子筛选，而用group lasso 解决因子与收益率之间的非线性问题
可以发现，lasso类方法全面的优于ICIR方法的表现，group lasso方法在市场中因子与收益存在非线性关系时较为优秀，整体来言收益优于lasso方法

'''

