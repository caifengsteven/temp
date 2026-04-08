# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:23:43 2020

@author: Asus
"""

'''
导读
A. 研究目的：本文利用优矿提供的数据，参考国盛证券《宏观逻辑的量化验证：映射关系混沌初开》（原作者：叶尔乐、刘富兵）中的研究方法，选取了90个宏观因子、59个股票资产，利用ANAVA方差检验法研究宏观因子状态和股票资产的关系，并进一步探究了显著关系的样本外持续性。读者可以根据自己的需要进一步丰富宏观因子、资产类别以挖掘所需要的状态映射关系。

B. 研究结论：

HP滤波可以较好的反映日度，月度和季度的趋势，且PIT的趋势比非PIT的趋势在拐点时变化要慢。利用单变量检验法对所选的90个因子和股票资产进行的状态映射关系显著性检验（2014到2018年样本）得到的显著性较强的pair用来进行择时时可以获得非常不错的择时效果
C. 文章结构：本文共分为3个部分，具体如下

一、数据准备和数据处理方法详解

二、宏观因子状态和股票资产收益的显著性检验

三、显著性关系的样本外表现

D. 时间说明

第一部分： 4分钟
第二部分： 4分钟
第三部分： 10分钟
总耗时18分钟左右
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''

import pandas as pd
import numpy as np
from CAL.PyCAL import * 
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import statsmodels.api as sm
import scipy.stats as stats
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False



'''
第一部分：数据准备和数据处理方法详解
该部分耗时 4分钟
该部分内容为：

获取宏观因子数据
对比分析平滑处理、HP滤波的效果
对比分析PIT计算HP趋势和后验计算HP趋势的差异
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 获取宏观因子数据

宏观数据有很多种类，本文选取的指标参考了研报中的指标并做了一些精简以节省运行时间，最终选取的因子为：
图片注释
图片注释

此外，宏观数据的发布时间并不规整，为了便于处理，统一将季度、月度数据的发布时间处理成数据期的下个月，以确保没有用到未来数据

'''

# 优矿中的宏观因子ID
indic_ids = '1020000004,1020000032,1020000015,1020000010,1020000017,1020000018,1020000019,1020000009,1020000008,1020000020,2020100021,1020001608,1020001609,1020001612,1020001613,1020001629,1020001630,2050100952,2050100948,1020001635,1020001636,2090102055,2090102059,1030001008,1030000012,1030000013,1030000014,1030000015,1030000016,1030000017,1030000018,1030000740,1030000019,1030000020,1030000021,1030000022,1030000023,1040000050,1040000047,1040000048,1040000030,1040000046,1040000045,1040001022,1070000007,1070000005,1070000009,1050000046,1050000013,1050000021,1050000022,1050000023,1100006031,1100006030,1110000008,1170001132,1170001133,1070013762,1010008415,1010008416,1010008417,1010008414,1010008418,1010008419,1010008420,1010008421,1010008422,1010008423,1010008424,1010008425,1070000030,1070002270,1070001973,1070001974,1070001979,1130000019,1130000018,1080000566,1090000004,1090000005,1090000353,1090000356,1090000358,1090000537,1090000544,1090000540,1090001390,1090001399,1090001385,1090001386'

start_date = '20090101'
end_date = '20190423'

# 指标ID和指标名称对齐
indic_data_df = DataAPI.EcoDataProGet(indic_ids,start_date,end_date, field=['indicID','publishDate', 'periodDate', 'dataValue'])
indic_name_df = DataAPI.EcoInfoProGet(indicID=indic_ids,field=['indicID','indicName', 'frequency'],pandas="1")
indic_data_df = indic_data_df.merge(indic_name_df, on=['indicID'], how='left')
indic_data_df.drop_duplicates(subset=['indicID', 'periodDate'], inplace=True)
indic_data_df.head()

# publishDate设定
def set_pubdate(df):
    pub_date = df['publishDate']
    period_date = df['periodDate']
    # 发布日期的deadline: 1个月之后
    pub_dll = datetime.datetime.strptime(period_date, '%Y-%m-%d')
    pub_dll = pub_dll + datetime.timedelta(days=28)
    pub_dll = pub_dll.strftime("%Y-%m-%d")
    return pub_dll

# 季频数据
q_data_df = indic_data_df.query("frequency=='季'")
q_data_df['publishDate'] = q_data_df.apply(lambda x: set_pubdate(x), axis=1)

# 月频数据
m_data_df = indic_data_df.query("frequency=='月'")
m_data_df['publishDate'] = m_data_df.apply(lambda x: set_pubdate(x), axis=1)

# 日频数据
d_data_df = indic_data_df.query("frequency=='日'")
d_data_df['publishDate'].fillna(d_data_df['periodDate'], inplace=True)


'''

1.2 对比分析平滑处理、HP滤波的效果

在划分因子状态时，用到的HP算法依赖于经验参数lamb，而为了消除季节性、月度的周期性因素，需要事先进行平滑处理，我们既可以先进行平滑处理再进行HP滤波，也可以依赖于lamb参数进行平滑处理，在本文中，根据经验，对季度、月度、日度因子都设定了对应的处理参数，读者可以根据需要进行自由调整

'''


# 增加滑动平均值列
def add_rolling_col(df, value_col, rolling_len):
    '''
    df: 指标数值dataframe，列至少包括 'periodDate', value_col
    value_col: 需要进行滑动平均的列名
    rolling_len: 滑动平均长度
    return:
    df: 增加滑动平均列的指标数值dataframe， 列在原来的基础上增加了一个 'rolling_value'
    '''
    df.sort_values(by=['periodDate'], ascending=True, inplace=True)
    df['rolling_value'] = df[value_col].rolling(window=rolling_len, min_periods=1).apply(np.mean)
    return df

# 增加HP滤波后的趋势列
def add_simple_hptrend_col(df, value_col, lamb):
    '''
    df: 指标数值dataframe，列至少包括 'periodDate', value_col
    value_col: 需要进行滑动平均的列名
    lamb: HP滤波时的平滑参数
    return:
    df: 增加滑动平均列的指标数值dataframe， 列在原来的基础上增加了一个 'hp_trend'
    '''
    df.sort_values(by=['periodDate'], ascending=True, inplace=True)
    data_cyclical, data_trend=sm.tsa.filters.hpfilter(df[value_col], lamb=lamb)
    df['hp_trend'] = data_trend
    return df

# 画图展示HP滤波情况
def plot_hp_result(data_df, rolling_len, hp_lamb, title_name):
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(111)

    # 增加滑动平均
    data_df = add_rolling_col(data_df, 'dataValue', rolling_len=2)
    # 直接进行hp滤波
    data_df = add_simple_hptrend_col(data_df, 'dataValue', lamb=hp_lamb)
    data_df.rename(columns={"hp_trend":"no_rolling_trend"}, inplace=True)
    # 对滑动平均后的数值进行hp滤波
    data_df = add_simple_hptrend_col(data_df, 'rolling_value', lamb=hp_lamb)
    data_df.rename(columns={"hp_trend":"rolling_hp_trend"}, inplace=True)
    _ = data_df[['periodDate','dataValue', 'no_rolling_trend', 'rolling_hp_trend']].set_index('periodDate').plot(ax=ax1)
    _ = ax1.set_title(title_name, fontproperties=font, fontsize=16)
    return None


# 各个频率下，波动最大和最小的两组，看滤波后的效果
freq_df_dict = {
    "m":m_data_df,
    "q":q_data_df,
    "d":d_data_df
}


hp_parms = { 
    'q':[1, 2],  # hp_lamb, rolling_len
    "m":[10, 6],
    'd':[14400, 30]
}


for freq in freq_df_dict.keys():
    cal_df = freq_df_dict[freq]
    # 方差/均值最大和最小的两个指标
    std_ratio = cal_df.groupby(['indicName'])['dataValue'].std()/cal_df.groupby(['indicName'])['dataValue'].mean()
    std_ratio.sort_values(inplace=True)
    min_indic_name = std_ratio.index.values[0]
    max_indic_name = std_ratio.index.values[-1]
    min_indic_df = cal_df.query('indicName==@min_indic_name')
    max_indic_df = cal_df.query('indicName==@max_indic_name')
    hp_lamb, rolling_len = hp_parms[freq]
    # 画图比较HP效果
    plot_hp_result(min_indic_df, rolling_len=rolling_len, hp_lamb=hp_lamb, title_name=min_indic_name.decode("utf8"))
    plot_hp_result(max_indic_df, rolling_len=rolling_len, hp_lamb=hp_lamb, title_name=max_indic_name.decode("utf8"))
    
    
'''

从上面的图可以看出，HP滤波算法对于季度、月度、周度的因子都能较好的反应趋势，本文选取的日度、月度、季度的参数技能较好反应趋势，时效性也足够。不经过平滑处理的趋势线（绿线）比经过平滑处理的趋势线（红线）波动性更大，因此下文的HP滤波都是先经过平滑处理，再使用HP算法得到趋势

   调试 运行
文档
 代码  策略  文档
1.3 对比分析PIT计算HP趋势和后验计算HP趋势的差异

HP滤波算法对于同一段时间序列，在全样本和局部样本下可能得到不一样的结果，为了防止使用中使用了未来数据，需要以PIT（point in time）的方法计算HP趋势值，该部分比较了PIT计算的趋势和全样本计算得到的趋势之间的差异


'''

# 增加HP趋势值列，PIT值
def calc_hp_pit(df, rolling_len, lamb):
    '''
    df: 指标值dataframe，列至少包含 indicID, periodDate, dataValue
    rolling_len: 进行HP滤波前进行滑动平均的窗口长度
    lamb: 进行HP滤波时的参数
    return:
    df: 指标值dataframe, 比输入增加了一列 'hp_trend'
    '''
    df = df.copy()
    df.sort_values(by=['indicID', 'periodDate'], ascending=[True, True], inplace=True)
    df.index = range(len(df))
    # 对每个指标进行平滑
    df['rolling_value'] = df.groupby(['indicID'])['dataValue'].rolling(rolling_len, min_periods=1).apply(np.mean).values

    # 计算当前的HP值
    def get_hp_value(df, lamb):
        data_cyclical, data_trend=sm.tsa.filters.hpfilter(df, lamb=lamb)
        return data_trend[-1]


    # 计算HP趋势值
    rolling_max_len = df.groupby(['indicID'])['frequency'].count().max() + 10
    result = df.set_index('periodDate').groupby(['indicID'])['rolling_value'].rolling(rolling_max_len, min_periods=2).apply(lambda x: get_hp_value(x, lamb))
    result = result.reset_index()
    result.rename(columns={"rolling_value":"hp_trend"}, inplace=True)
    df = df.merge(result, on=['indicID', 'periodDate'], how='left')
    df['hp_trend'].fillna(df['rolling_value'], inplace=True)
    del df['rolling_value']
    return df

# 对每个频率段的数据，计算PIT的HP趋势值
for freq in freq_df_dict.keys():
    cal_df = freq_df_dict[freq]
    rolling_len, lamb = hp_parms[freq]
    cal_df = calc_hp_pit(cal_df, rolling_len, lamb)
    freq_df_dict[freq] = cal_df

m_data_df = freq_df_dict['m']
q_data_df = freq_df_dict['q']
d_data_df = freq_df_dict['d']



# 比较PIT计算HP滤波后的值同非PIT计算的差异
fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(111)
rolling_len, lamb = hp_parms['m']
sample_df = m_data_df.query("indicName=='产量:原油:当月同比'")
sample_df['rolling_value'] = sample_df.groupby(['indicID'])['dataValue'].rolling(rolling_len, min_periods=1).apply(np.mean).values
data_cyclical, data_trend=sm.tsa.filters.hpfilter(sample_df['rolling_value'], lamb=lamb)
sample_df['back_hp_trend'] = data_trend
sample_df.rename(columns={"hp_trend":"pit_hp_trend"}, inplace=True)
_ = sample_df.set_index('periodDate')[['dataValue', 'pit_hp_trend', 'back_hp_trend']].plot(ax = ax1)

'''

从上图可以看出，PIT的趋势值和非PIT的值还是存在一定的差异，不过总体来看差异并不是很大，在趋势发生变化时，PIT计算的趋势会比全样本趋势滞后一段时间

   调试 运行
文档
 代码  策略  文档
第二部分：宏观因子状态和股票资产收益的显著性检验
该部分耗时 4分钟
该部分内容为：

为了便于统计，将前面的日度、季度数据都采样至月频
计算宏观因子和资产收益之间的显著性关系
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
2.1 对数据频率进行处理便于后续的统计

'''

# 得到趋势变化值
def add_trend_status(in_df):
    '''
    in_df: 指标值dataframe，至少包括列 indicID, periodDate, hp_trend
    返回：
    in_df: 指标值dataframe，列增加一列， trend_delta,  trend_delta= 本期值-上期值
    '''
    in_df = in_df.copy()
    in_df = in_df.sort_values(by=['indicID', 'periodDate'], ascending=[True, True])
    trend_delta = in_df.set_index('periodDate').groupby(['indicID']).apply(lambda x:x['hp_trend'] - x['hp_trend'].shift(1)).reset_index()
    trend_delta.columns = ['indicID', 'periodDate', 'trend_delta']
    in_df = in_df.merge(trend_delta, on=['indicID', 'periodDate'], how='left')
    in_df['trend_delta'].fillna(0.001, inplace=True)
    return in_df

m_data_df = add_trend_status(m_data_df)
d_data_df = add_trend_status(d_data_df)
q_data_df = add_trend_status(q_data_df)
    

# 日度数据转成月度数据: 取均值
d_data_df = d_data_df[['indicID','publishDate', 'periodDate', 'dataValue', 'hp_trend', 'trend_delta']]
d_data_df['publishDate'] = d_data_df['publishDate'].apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d"))
m_d_data_df = d_data_df.groupby(['indicID']).apply(lambda x: x.set_index("publishDate")[['dataValue', 'hp_trend', 'trend_delta']].resample("M", how='mean')).reset_index()
m_d_data_df['month'] = m_d_data_df['publishDate'].apply(lambda x: x.strftime("%Y-%m"))

# 季度数据转成月度数据：前向填充
q_data_df['publishDate'] = q_data_df['publishDate'].apply(lambda x:datetime.datetime.strptime(x, "%Y-%m-%d"))
m_q_data_df = q_data_df.groupby(['indicID']).apply(lambda x: x.set_index("publishDate")[['dataValue', 'hp_trend', 'trend_delta']].resample("M").ffill()).reset_index()
m_q_data_df['month'] = m_q_data_df['publishDate'].apply(lambda x: x.strftime("%Y-%m"))

# 调整月度数据的month格式
m_data_df['month'] = m_data_df['publishDate'].apply(lambda x: "-".join(x.split("-")[:2]))

# 将所有指标合并成一个月度数据dataframe
month_indic_df = pd.concat([m_data_df[['indicID', 'publishDate','month', 'dataValue', 'hp_trend', 'trend_delta']], m_q_data_df, m_d_data_df], axis=0)
month_indic_df.head()

# # 转换数据格式
indic_status_df = month_indic_df[['indicID', 'month', 'trend_delta']]
indic_status_df['trend_flag'] = indic_status_df['trend_delta'].apply(lambda x: 'UP' if x>=0 else 'DN')
indic_status_df = indic_status_df.pivot(index='month', columns='indicID', values='trend_flag').reset_index()
indic_status_df.tail()

'''
2.2 观察2014年以来，宏观因子和资产的显著性关系

取2014年1月到2018年12月的数据作为全样本，研究这5年中宏观因子状态和资产收益的显著性关系
'''

# 各种资产的收益率
sample_start = '20131201'
sample_end = '20190131'
import datetime


indic_status_df = indic_status_df.query("month>='2014-01'")

# 风格因子组合、行业因子组合收益率（日度）
ret_df1 = DataAPI.RMFactorRetDayGet(beginDate=sample_start.replace("-", ""),endDate=sample_end.replace("-", ""),tradeDate=u"",field=u"",pandas="1")
# 转成月度收益
ret_df1['tradeDate'] = ret_df1['tradeDate'].apply(lambda x: datetime.datetime.strptime(x, "%Y%m%d"))
ret_df1.set_index('tradeDate', inplace=True)
ret_df1 = ret_df1.resample('M',convention='end', how=lambda x:(x+1.0).prod()-1.0)
ret_df1 = ret_df1.reset_index()
ret_df1['tradeDate'] = ret_df1['tradeDate'].apply(lambda x: x.strftime("%Y%m"))


# 指数收益率（月度）
zs_df = DataAPI.MktIdxmGet(indexID=u"000300.ZICN,000905.ZICN,000852.ZICN,000016.ZICN,000918.ZICN,000919.ZICN,399374.ZICN,399375.ZICN,399376.ZICN,399377.ZICN,000968.ZICN,000969.ZICN",ticker=u"",beginDate=sample_start,endDate=sample_end,field=u"indexID,secShortName,endDate,chgPct",pandas="1")
zs_df = zs_df.pivot(index='endDate', columns='secShortName', values='chgPct').reset_index()
zs_df['endDate'] = zs_df['endDate'].apply(lambda x: "".join(x.split("-")[:2]))
zs_df.rename(columns={'endDate':"tradeDate"}, inplace=True)

# 合并所有资产的收益率
ret_df = zs_df.merge(ret_df1, on=['tradeDate'])

# 计算衍生组合的收益率
ret_df['中证500-沪深300'] = ret_df['中证500'] - ret_df['沪深300']
ret_df['中证1000-沪深300'] = ret_df['中证1000'] - ret_df['沪深300']
ret_df['沪深300-上证50'] = ret_df['沪深300'] - ret_df['上证50']
ret_df['中证500-上证50'] = ret_df['中证500'] - ret_df['上证50']
ret_df['中证1000-上证50'] = ret_df['中证1000'] - ret_df['上证50']
ret_df['沪深300-上证50'] = ret_df['沪深300'] - ret_df['上证50']
ret_df['300成长-300价值'] = ret_df['300成长'] - ret_df['300价值']
ret_df['中盘成长-中盘价值'] = ret_df['巨潮中盘成长'] - ret_df['巨潮中盘价值']
ret_df['小盘成长-小盘价值'] = ret_df['巨潮小盘成长'] - ret_df['巨潮小盘价值']


# 计算下个月
def get_next_month(m):
    year = int(m[:4])
    month = int(m[4:])
    if month == 12:
        nx_month = 1
        nx_year = year + 1
    else:
        nx_month = month + 1
        nx_year = year
    return '%s-%s'%(nx_year, str(nx_month).zfill(2))

# 增加下个月日期
ret_col = [x for x in ret_df.columns]
ret_df['nx_month'] = ret_df['tradeDate'].apply(lambda x: get_next_month(x))

# 得到下个月的收益
nx_ret_df = ret_df[['nx_month'] + [x for x in ret_col if x != 'tradeDate']]
nx_ret_df.head()


# 单变量状态检验显著性
def f_one_stat(indic_name, asset_name):
# if 1:
#     indic_name = 1010008414
#     asset_name = '沪深300' 
    '''
    遍历indic_name的不同状态，统计在各个状态下，资产上涨比例（相对全样本）、收益均值差（相对全样本）、检验显著性
    indic_name: 宏观指标名称
    asset_name: 资产名称
    '''
    indic_df = indic_status_df[['month', indic_name]]
    indic_df.columns = ['month', 'status']
    asset_ret_df = nx_ret_df[['nx_month',asset_name]]
    asset_ret_df.columns = ['month', 'ret']
    md_df = indic_df.merge(asset_ret_df, on=['month'], how='left')
    
    total_len = len(md_df)
    count_pct = md_df.groupby(['status'])['ret'].count()/total_len
    ret_mean = md_df.groupby(['status'])['ret'].mean() - md_df['ret'].mean()
    md_df.dropna(inplace=True)
    dn_value = md_df.query("status=='DN'").ret.values
    up_value = md_df.query("status=='UP'").ret.values
    all_value = md_df.ret.values

    p_series = pd.Series()
    # 下降样本的检验显著性
    F_statistic, DN_p = stats.f_oneway(dn_value,all_value)
    F_statistic, UP_p = stats.f_oneway(up_value,all_value)
    p_series.loc['DN'] = DN_p
    p_series.loc['UP'] = UP_p
    stat_df = pd.concat([count_pct, ret_mean, p_series], axis=1).reset_index()
    stat_df.columns = [u'指标状态', u'上涨样本差值', u'收益均值差', u'显著性P值']
    stat_df['indic_name'] = indic_name
    stat_df['assset_name'] = asset_name
    return stat_df

stat_list = []
# 遍历所有指标和资产，得到不同状态下的显著性表现
indic_name_list = [x for x in indic_status_df.columns if x != 'month']
asset_name_list = [x for x in nx_ret_df.columns if x != 'nx_month']
for indic_name in indic_name_list:
    for asset_name in asset_name_list:
        stat_df = f_one_stat(indic_name, asset_name)
        stat_list.append(stat_df)
        
        


name_df = indic_data_df[['indicID', 'indicName']].drop_duplicates()
stat_all_df = pd.concat(stat_list, axis=0)
stat_all_df = stat_all_df.rename(columns={"indic_name":"indicID"}).merge(name_df, on=['indicID'], how='left')
stat_all_df[stat_all_df[u"收益均值差"]>0].sort_values(by=[u'显著性P值'], ascending=True).head(20)

# 宏观指标对资产择时的简易回测
def back_test_timing(indicID, asset_name, direct='long_only', status='UP'):
    indic_df = indic_status_df[['month', indicID]]
    indic_df.columns = ['month', 'status']
    asset_ret_df = nx_ret_df[['nx_month',asset_name]]
    asset_ret_df.columns = ['month', 'ret']
    md_df = indic_df.merge(asset_ret_df, on=['month'], how='left')
    
    # 做多、做空乘数
    def long_short_multip(row_df, direct, status):
        c_status = row_df['status']
        if c_status == status:
            return 1
        else:
            if direct == 'long_only':
                return 0
            else:
                return -1
    # 根据indic状态择时的收益
    md_df['multip'] = md_df.apply(lambda x: long_short_multip(x, direct, status), axis=1)
    md_df['fret'] = md_df['ret'] * md_df['multip']
    md_df['timing_net_value'] = md_df['fret'] + 1
    md_df['timing_net_value'] = md_df['timing_net_value'].cumprod()
    
    # 不择时，一直纯做多的收益
    md_df['ori_net_value'] = md_df['ret'] + 1
    md_df['ori_net_value'] = md_df['ori_net_value'].cumprod()
    return md_df[['month', 'status','timing_net_value', 'ori_net_value']]

# 通过indic_id找到对应的中文名
def get_indic_name_by_id(in_id):
    return indic_data_df.query("indicID==@in_id").indicName.values[0].decode("utf8")

# 遍历前10个最显著的正向关系
for idx, row_df in stat_all_df[stat_all_df[u"收益均值差"]>0].sort_values(by=[u'显著性P值'], ascending=True).head(10).iterrows():
    indic_id = row_df['indicID']
    asset_name = row_df['assset_name']
    indic_name = get_indic_name_by_id(indic_id)
    status = row_df[u'指标状态']
    ls_ret_df = back_test_timing(indic_id, asset_name, direct='long_short', status=status)
    lo_ret_df = back_test_timing(indic_id, asset_name, direct='long_only', status=status)

    # 画图展示择时效果
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ls_ret_df.set_index('month').plot(ax=ax1)
    _ = ax1.set_title(u'%s多空择时%s'%(indic_name, asset_name), fontproperties=font, fontsize=16)
    lo_ret_df.set_index('month').plot(ax=ax2)
    _ = ax2.set_title(u'%s纯多择时%s'%(indic_name, asset_name), fontproperties=font, fontsize=16)
    
'''

从上图结果来看，用宏观因子对资产进行择时，无论是用纯多组合还是多空组合，都能取得非常不错的超额收益，从数据层面上证明用宏观因子对资产进行择时是可行的，不过如果需要应用到实际投资者，读者还需要进一步挖掘择时关系背后的经济逻辑，本文限于篇幅不对该部分内容进行展开

   调试 运行
文档
 代码  策略  文档
第三部分：显著性关系的样本外表现
上面的宏观因子择时效果是在统计样本内做的，存在使用了未来函数的可能，因此该部分我们研究显著性关系的样本外表现
不同于常规意义上理解的样本外研究，本文参考国盛研报中的方法，计算研究区间内的显著性关系变化情况，具体为：
取全历史数据(data_start到data_end)
从outsample_start月份开始遍历每个月current_month，以data_start到current_month为样本，计算得到p<=0.05的样本
计算不同月份的显著性指标关系差异

'''


# 
data_start = '20081231'
data_end = '20190131'
outsample_month_start = '2014-01'
outsample_month_end = '2019-01'


indic_status_df = month_indic_df[['indicID', 'month', 'trend_delta']]
indic_status_df['trend_flag'] = indic_status_df['trend_delta'].apply(lambda x: 'UP' if x>=0 else 'DN')
indic_status_df = indic_status_df.pivot(index='month', columns='indicID', values='trend_flag').reset_index()

# 风格因子组合、行业因子组合收益率（日度）
ret_df1 = DataAPI.RMFactorRetDayGet(beginDate=data_start.replace("-", ""),endDate=data_end.replace("-", ""),tradeDate=u"",field=u"",pandas="1")
# 转成月度收益
ret_df1['tradeDate'] = ret_df1['tradeDate'].apply(lambda x: datetime.datetime.strptime(x, "%Y%m%d"))
ret_df1.set_index('tradeDate', inplace=True)
ret_df1 = ret_df1.resample('M',convention='end', how=lambda x:(x+1.0).prod()-1.0)
ret_df1 = ret_df1.reset_index()
ret_df1['tradeDate'] = ret_df1['tradeDate'].apply(lambda x: x.strftime("%Y%m"))


# 指数收益率（月度）
zs_df = DataAPI.MktIdxmGet(indexID=u"000300.ZICN,000905.ZICN,000852.ZICN,000016.ZICN,000918.ZICN,000919.ZICN,399374.ZICN,399375.ZICN,399376.ZICN,399377.ZICN,000968.ZICN,000969.ZICN",ticker=u"",beginDate=data_start,endDate=data_end,field=u"indexID,secShortName,endDate,chgPct",pandas="1")
zs_df = zs_df.pivot(index='endDate', columns='secShortName', values='chgPct').reset_index()
zs_df['endDate'] = zs_df['endDate'].apply(lambda x: "".join(x.split("-")[:2]))
zs_df.rename(columns={'endDate':"tradeDate"}, inplace=True)

# 合并所有资产的收益率
ret_df = zs_df.merge(ret_df1, on=['tradeDate'])

# 计算衍生组合的收益率
ret_df['中证500-沪深300'] = ret_df['中证500'] - ret_df['沪深300']
ret_df['中证1000-沪深300'] = ret_df['中证1000'] - ret_df['沪深300']
ret_df['沪深300-上证50'] = ret_df['沪深300'] - ret_df['上证50']
ret_df['中证500-上证50'] = ret_df['中证500'] - ret_df['上证50']
ret_df['中证1000-上证50'] = ret_df['中证1000'] - ret_df['上证50']
ret_df['沪深300-上证50'] = ret_df['沪深300'] - ret_df['上证50']
ret_df['300成长-300价值'] = ret_df['300成长'] - ret_df['300价值']
ret_df['中盘成长-中盘价值'] = ret_df['巨潮中盘成长'] - ret_df['巨潮中盘价值']
ret_df['小盘成长-小盘价值'] = ret_df['巨潮小盘成长'] - ret_df['巨潮小盘价值']


# 计算下个月
def get_next_month(m):
    year = int(m[:4])
    month = int(m[4:])
    if month == 12:
        nx_month = 1
        nx_year = year + 1
    else:
        nx_month = month + 1
        nx_year = year
    return '%s-%s'%(nx_year, str(nx_month).zfill(2))

# 增加下个月日期
ret_col = [x for x in ret_df.columns]
ret_df['nx_month'] = ret_df['tradeDate'].apply(lambda x: get_next_month(x))

# 得到下个月的收益
nx_ret_df = ret_df[['nx_month'] + [x for x in ret_col if x != 'tradeDate']]


'''

特别说明

由于逐月遍历每个资产和宏观因子的状态对需要花费很长时间，在此处本文以2014到2018样本得到的显著性关系最强的前10条记录对应的宏观因子和资产为遍历对象，可以缩短下文的运行时间以便进行展示，读者可根据自己需要调整遍历因子池和资产池

'''

example_df = stat_all_df[stat_all_df[u"收益均值差"]>0].sort_values(by=[u'显著性P值'], ascending=True).head(10)
example_indicID_list = example_df['indicID'].unique().tolist()
example_asset_list = example_df['assset_name'].unique().tolist()
example_df.head()

# 检查指标和资产之间是否有显著性，返回有显著性的状态
def filter_significance(indic_name, asset_name, max_month='2019-01'):
    '''
    遍历indic_name的不同状态，统计在各个状态下，资产上涨比例（相对全样本）、收益均值差（相对全样本）、检验显著性
    indic_name: 宏观指标名称
    asset_name: 资产名称
    max_month: 样本最大的时间点
    '''
    indic_df = indic_status_df[['month', indic_name]]
    indic_df.columns = ['month', 'status']
    asset_ret_df = nx_ret_df[['nx_month',asset_name]]
    asset_ret_df.columns = ['month', 'ret']
    md_df = indic_df.merge(asset_ret_df, on=['month'], how='left')
    # 根据日期筛选
    md_df = md_df.query("month<@max_month")
    md_df.dropna(inplace=True)
    dn_value = md_df.query("status=='DN'").ret.values
    up_value = md_df.query("status=='UP'").ret.values
    all_value = md_df.ret.values

    p_series = pd.Series()
    # 下降样本的检验显著性
    F_statistic, DN_p = stats.f_oneway(dn_value,all_value)
    F_statistic, UP_p = stats.f_oneway(up_value,all_value)
    p_series.loc['DN'] = DN_p
    p_series.loc['UP'] = UP_p
    stat_df = p_series.reset_index()
    stat_df.columns = [u'指标状态', u'显著性P值']
    stat_df['indic_name'] = indic_name
    stat_df['assset_name'] = asset_name
    return stat_df[stat_df[u'显著性P值']<=0.05]

import time
start_time = time.time()
filter_list = []
# 遍历所有指标和资产，得到不同状态下的显著性表现
# indic_name_list = [x for x in indic_status_df.columns if x != 'month']
# asset_name_list = [x for x in nx_ret_df.columns if x != 'nx_month']

# 将因子池和资产池设定为上文中最显著的10组关系对应的范围
indic_name_list = example_indicID_list
asset_name_list = example_asset_list
month_list = list(np.unique(indic_status_df.month.values))
month_list = [x for x in month_list if x>=outsample_month_start]
for month in month_list:
    print ("\n",month)
    tcount = 0
    for indic_name in indic_name_list:
        for asset_name in asset_name_list:
            if tcount %100 == 0:
                print (tcount)
            filter_df = filter_significance(indic_name, asset_name, max_month=month)
            filter_df['month'] = month
            filter_list.append(filter_df)
            tcount += 1
end_time = time.time()
print (u'花费总时间为:%s'%(end_time - start_time))

# 合并所有的显著性记录
filter_df = pd.concat(filter_list, axis=0)
# 计算每个月的显著性对个数
count_df = filter_df.groupby('month')['indic_name'].count()
count_df = count_df.reset_index().rename(columns={"indic_name":"total_count"})

# 每个月新增和减少的显著性关系对
pre_record = []
ad_df = pd.DataFrame()
tcount = 0
for month in month_list:
    new_record = []
    month_filter_df = filter_df.query("month==@month")
    for idx, row_df in month_filter_df.iterrows():
        indic_name = row_df['indic_name']
        asset_name = row_df['assset_name']
        new_record.append([indic_name, asset_name])
    new_add = [x for x in new_record if x not in pre_record]
    new_delete = [x for x in pre_record if x not in new_record]
    ad_df.loc[tcount, 'month'] = month
    ad_df.loc[tcount, 'add'] = len(new_add)
    ad_df.loc[tcount, 'del'] = len(new_delete)
    tcount += 1
    pre_record = new_add

# 合并统计数据
m_stat_df = count_df.merge(ad_df, on=['month'], how='outer')


# 画图展示结果
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(111)
_ = m_stat_df.set_index('month').sort_index().plot(ax=ax1)
ax1.set_title(u'每个月显著性关系对总数和增减情况', fontproperties=font, fontsize=16)

'''
从上图来看，在2014到2019年样本中状态映射关系最显著的10组对应的因子和资产，在2014年1月到11月之间不存在显著性关系，显著性从2014年12月开始出现，从15年9月后开始剧增，说明显著关系的出现同市场风格密切相关。另外相比上个月减少的显著性组合数大部分都处在0的水平，说明一个因子-资产组表现出了显著性之后会大概率继续体现出显著性
'''