# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:46:12 2020

@author: Asus
"""

'''
导读
A. 研究目的：

一般基本面因子主要采用定期财务报告的数据计算生成，忽略了业绩预告中的财务信息。本文利用优矿的财务数据与回测框架，参考东方证券研报《上市公司业绩预告信息研究》中的研究方法，用以探索业绩预告中数据对投资的影响
B. 研究结论：

自从2012年前，A股发布业绩预告数量大幅提升。以18年全年为例，共有2836家公司发布业绩预告数据，覆盖度较高

利用业绩预告数据直接构建业绩偏离度因子，其RankIC均值0.021，ICIR达2.76；多空组合信息比为1.60，最大回撤-1.66%

利用业绩预告数据去增强现有财务因子EP_TTM, ROE，加入业绩预告前后因子平均相关性均约95%，说明并未改变原有因子选股逻辑。但从回测上看，加入业绩预告后比原有因子有所提升。从IC来看，EP_TTM的ICIR从2.33提升至2.63，ROE因子的ICIR从1.17提升至1.52。从构建组合上看，EP_TTM因子Top100组合收益从10.7%提升到13.1%，ROE因子Top100组合收益从7.8提升到8.4%

C. 文章结构: 本文共分为3个部分，具体如下

一、业绩预告数据简介，该部分主要评估历史业绩发布预告的数量及数据质量

二、业绩预告偏离度因子，该部分主要构建了业绩偏离度因子，并且进行回测

三、基于业绩预告提高财务因子的效果，该部分主要利用业绩预告数据来改进EP_TTM、ROE，随后与原有因子进行回测比较

D. 运行时间说明

一、业绩预告数据简介，需要5分钟左右

二、业绩预告偏离度因子，需要25分钟左右

三、基于业绩预告提高财务因子的效果，需要30分钟左右

总耗时60分钟左右

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
一、业绩预告数据简介
该部分耗时 5分钟
该部分内容包括:

1.1 业绩预告历史发布情况分析
1.2 业绩预告发布数据质量分析
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 业绩预告历史发布情况分析

本小节统计了A股公司2007年至今每年发布的业绩预告数量，及公布业绩预告的上市公司数量

'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
#from CAL.PyCAL import *    # CAL.PyCAL中包含font

# 加载PIT的业绩预告数据
forecast_df = DataAPI.FdmtEfGet(pandas="1")
#print(forecast_df)
forecast_df = forecast_df[forecast_df['ticker'].apply(lambda x: x[0] not in ['2', '9'])]
#print(forecast_df)
forecast_df = forecast_df.sort_values(['ticker', 'endDate', 'publishDate', 'reportType'])
#print(forecast_df)
forecast_df = forecast_df.drop_duplicates(subset=['ticker', 'endDate', 'publishDate']) # 去除CQ3,Q3中的Q3
#print(forecast_df)
forecast_df['publishDate'] = forecast_df['publishDate'].apply(lambda x: x[:10])
#print(forecast_df)

'''

1.1.1 不同年份业绩预告数量和公布业绩预告的上市公司数量

从下图中可以看到，A股中发布业绩预告的公司逐年增长，业绩预告数量也呈同样趋势。12年1月，证监会发布规定创业板公司必须披露业绩预告；同年8月，深交所发布规定中小板企业同样必须披露业绩预告。所以从结果中可以看到12年后，业绩预告数量有大幅提升，2018年全年A股有2836家公司发布业绩预告数据。

'''


df = forecast_df.copy()
df['year'] = df['publishDate'].apply(lambda x: x[:4])
number_df = df.groupby('year').apply(lambda x: pd.Series(index=[u'预告数量', u'公司数量'], data=[len(x), len(x['ticker'].unique())]))
print(number_df)
fig, ax = plt.subplots(figsize=(16, 5))
number_df.plot(kind='bar', legend=False, ax=ax)
ax.legend([u'预告数量', u'公司数量'], prop=font, fontsize=16)
ax.set_title(u"不同年份业绩预告数量和公布业绩预告的上市公司数量", fontproperties=font, fontsize=16)
plt.show()

'''

1.1.2 不同年份不同季度的业绩预告和上市公司数量

下图统计了分季度的业绩预告发布情况。图中可知，一季报因为强制披露的要求较低，披露数量最小；二季度与三季度的数量差不多；年度的业绩预告数量最多。

'''

report_type_dict = {'A': 4, 'S1': 2, 'Q1': 1, 'Q3': 3, 'CQ3': 3}
df['reportNum'] = df['reportType'].apply(lambda x: report_type_dict[x])
quarter_ticker_number_df = df.groupby('year').apply(lambda x: x.drop_duplicates(subset=['ticker', 'reportNum']).groupby('reportNum').agg({'ticker': 'count'}))
quarter_report_number_df = df.groupby('year').apply(lambda x: x.groupby('reportNum').agg({'ticker': 'count'}))

print(u'不同年份不同期业绩预告和上市公司数量')
fig, axes = plt.subplots(figsize=(16, 10), nrows=2, ncols=1)
quarter_ticker_number_df.unstack(level=1).plot(kind='bar', legend=False, ax=axes[0])
axes[0].legend([u'一季度', u'中报', u'三季度', u'年报'], prop=font, fontsize=16)
axes[0].set_title(u"公司数量", fontproperties=font, fontsize=16)
axes[0].xaxis.label.set_visible(False)

quarter_report_number_df.unstack(level=1).plot(kind='bar', legend=False, ax=axes[1])
axes[1].legend([u'一季度', u'中报', u'三季度', u'年报'], prop=font, fontsize=16)
axes[1].set_title(u"报告数量", fontproperties=font, fontsize=16)
axes[1].xaxis.label.set_visible(False)
plt.show()

'''
1.1.3 不同月份的业绩预告发布数量

下图统计了分月份的业绩预告发布情况。图中可知，全年发布业绩预告主要在1、3、4、7、8、10这几个月份。1月份主要发布的是年报的预告，3月份主要发布一季报的预告，4月份发布年报与一季报的预告，7月份主要是中报的预告，8月份主要是三季报的预告，10月份的主要是三季报与年报的预告。其中，很多月份的业绩预告都要比正式财报发布要早，比如说10月份、1月份的年报预告都要比正式公布时间早很多。

'''


df['month'] = df['publishDate'].apply(lambda x: int(x[5:7]))
month_report_number_df = df.groupby('month').apply(lambda x: x.groupby('reportNum').agg({'ticker': 'count'}))

fig, ax = plt.subplots(figsize=(16, 5))
month_report_number_df.unstack(level=1).plot(kind='bar', legend=False, ax=ax)
ax.legend([u'一季度', u'中报', u'三季度', u'年报'], prop=font, fontsize=16)
ax.set_title(u"各月份公布业绩预告数量", fontproperties=font, fontsize=16)
ax.xaxis.label.set_visible(False)

plt.show()

'''


1.2 业绩预告发布数据质量分析

一般来说，上市公司公布业绩预告中的数据均有上下限，有些上下限相隔较近，有些范围极广。另外，业绩预告中的数据与最终真实财报的数据也有偏差。本小节选取了预告中的净利润数据，从上下限范围、及准确度两方面来分析其质量。

本小节定义两个指标来衡量上述问题

业绩预告区间比: 主要分析上下限的差距
业绩预告区间比=上限−下限(上限+下限)/2
业绩预告均值偏离度: 主要分析预告数据与真实财报数据的差距
业绩预告均值偏离度=实际的累积净利润−(上限+下限)/2实际的单季度净利润

'''

# 不同业绩预告上下限区间和对应的均值偏离度统计
field=[u'secID', u'publishDate', u'endDate', u'ticker', u'secShortName', u'reportType', u'expnIncAPLL', u'expnIncAPUPL']
df = df[field].dropna(subset=[u'expnIncAPLL', u'expnIncAPUPL'], how='all')
df['expnIncAPLL'] = df['expnIncAPLL'].fillna(df['expnIncAPUPL'])
df['expnIncAPUPL'] = df['expnIncAPUPL'].fillna(df['expnIncAPLL'])

# 读取真实的归母净利润, 同时计算单季度真实归母净利润
income_df = DataAPI.FdmtISAllLatestGet(beginDate=u"20070101", reportType=['Q1', 'S1', 'CQ3', 'A'], field=[u'ticker', u'endDate', u'reportType', u'NIncomeAttrP'], pandas="1").dropna(subset=['NIncomeAttrP'])
income_df['year'] = income_df['endDate'].apply(lambda x: x[:4])
income_df = income_df[income_df['ticker'].apply(lambda x: x[0] not in ['2', '9'])]

def format_data(x):
    # 给入财务数据，根据累计净利润计算出单季度净利润
    report_type_date_dict = {'Q1': '03-31', 'S1': '06-30', 'CQ3': '09-30', 'A': '12-31'}
    format_df = pd.pivot_table(x[[u'year', u'reportType', u'NIncomeAttrP']].copy(), index='reportType', columns='year', values='NIncomeAttrP')
    format_df = format_df.loc[['Q1', 'S1', 'CQ3', 'A']]
    first_quarter_data = format_df.iloc[0]
    format_df = format_df - format_df.shift(1)
    format_df.iloc[0] = first_quarter_data
    format_df = format_df.stack().reset_index()
    format_df.columns = [u'reportType', u'year', u'QuarterNIncomeAttrP']
    format_df = format_df.dropna(subset=['QuarterNIncomeAttrP'])
    if len(format_df) == 0:
        x['QuarterNIncomeAttrP'] = np.nan
        return x.dropna(subset=['endDate', 'NIncomeAttrP'])
    
    format_df['endDate'] =  format_df.apply(lambda x: '%s-%s' % (x['year'], report_type_date_dict[x['reportType']]), axis=1)
    df = pd.merge(x, format_df[['endDate', u'QuarterNIncomeAttrP']], on='endDate', how='left')
    
    return df.dropna(subset=['endDate', 'NIncomeAttrP'])

# 计算指标
final_income_df = income_df.groupby(['ticker']).apply(lambda x: format_data(x)).reset_index(drop=True)
data = pd.merge(df, final_income_df[['ticker', 'endDate', 'NIncomeAttrP', 'QuarterNIncomeAttrP']], on=['ticker', 'endDate'])
data['expnIncInterval'] = (data['expnIncAPUPL'] - data['expnIncAPLL']) * 2 / (data['expnIncAPUPL'] + data['expnIncAPLL']).abs()
data['deviation'] = data.apply(lambda x: (x['QuarterNIncomeAttrP'] - (x['expnIncAPLL'] + x['expnIncAPUPL'])/2.0) / np.abs(x['QuarterNIncomeAttrP']+1e-10)  if x['reportType']=='Q3' else (x['NIncomeAttrP'] - (x['expnIncAPLL'] + x['expnIncAPUPL'])/2.0) / np.abs(x['QuarterNIncomeAttrP']+1e-10), axis=1)
data['deviation'] = data['deviation'].clip(-20, 20)
data = data.dropna(subset=['deviation'])

'''

下图中衡量了不同业绩预告区间比的预测偏离度情况。可以看出，大部分公司的上下限区间比在0.3以下，但仍有4%的公司区间比在1以上。另外，上下限区间比越小的公司，其偏离度也很小，随着上下限区间变大，其预测偏离度也同样变大。同时，偏离度一般负值，说明业绩预告中数据一般都会高估净利润数据。

注： “业绩预告均值偏离度”中某些公司预测出现极大偏差，这里限定了偏离度在-20~20范围内

'''

data = data.dropna()
test = data.groupby(pd.cut(data["expnIncInterval"], list(np.arange(0, 1.1, 0.1)) + [np.Inf])).apply(lambda x: pd.Series(index=['interval', 'dev'], data=[len(x)*1.0/len(data), np.mean(x['deviation'])]))

fig, ax = plt.subplots(figsize=(16, 5))
ax1 = ax.twinx()

test['interval'].plot(kind='bar', color='blue', ax=ax, legend=False)
test['dev'].plot(kind='bar', color='red', ax=ax1, legend=False)
ax.set_xticklabels(test.index, rotation=45)

ax.set_ylim(0, 0.5)
ax.set_yticklabels(['%.1f' % (i * 100) + '%' for i in ax.get_yticks()])
ax1.set_ylim(-1.0, 0)

ax.legend([u'对应区间预告占比(左轴)'], prop=font, fontsize=16, loc=[0.4, 0.2])
ax1.legend([u'均值偏离度(右轴)'], prop=font, fontsize=16, loc=[0.6, 0.2])
ax.set_title(u"不同业绩预告上下限区间和对应的均值偏离度统计", fontproperties=font, fontsize=16)
ax.xaxis.label.set_visible(False)
plt.show()

'''
此外，同样统计了均值偏离度的分布情况。下图可以看出整体分布呈左偏，验证了上图多数公司高估净利润的情况。多数公司的偏离度均在-0.2~0.2之间。偏离度<-2的公司占比3.6%左右，此类公司有夸大业绩的动机，值得投资者警惕。
'''


fig, ax = plt.subplots(figsize=(16, 5))
test = data.groupby(pd.cut(data["deviation"], [-np.Inf] + list(np.arange(-2, 2.1, 0.1).round(2)) + [np.Inf])).apply(lambda x: pd.Series(index=['pert'], data=[len(x) * 1.0 / len(data)]))

test['pert'].plot(kind='bar', color='blue', ax=ax, legend=False)
ax.set_xticklabels(test.index, rotation=45)

ax.set_ylim(0, 0.3)
ax.set_yticklabels(['%.1f' % (i * 100) + '%' for i in ax.get_yticks()])
ax.set_title(u"业绩预告均值偏离度分布情况", fontproperties=font, fontsize=16)
ax.xaxis.label.set_visible(False)
plt.show()


'''
二、业绩预告偏离度因子
该部分耗时 25分钟左右
本章节利用第一小节定义的业绩预告偏离度指标作为因子进行测试。首先考察因子在通联全A的覆盖度情况，然后对因子进行去极值、填缺失值、中性化等处理，最后讨论了因子的IC及回测情况。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''

import pandas as pd
import numpy as np
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from CAL.PyCAL import * 
import gevent
import datetime as dt
import calendar
from dateutil.relativedelta import relativedelta
from quartz_extensions.SignalAnalysis.tears import analyse_return, analyse_monthly_return, analyse_IC, analyse_construction, analyse_general

def get_findata_nlatest(fin_data_frame, n, col_name='value', is_annual=False, is_duplicate=False):
    """
    :param fin_data_frame: df, column= ['ticker','pub_date','end_date',[col_name]], index=num, pub_date='%Y%m%d'
    :param n: int， n>=0, 计算多少期
    :param col_name: str, 财务字段名称
    :param is_annual: bool, 为True时，取最近几期年报数据; 为False时，取最近几期财报数据（包括季度报告）。默认为False。
    :param is_duplicate: bool, 为True时，按照publish_date取能获取到的最近财季的数据
    :return: df, column= ['ticker','pub_date', 'end_date', 'value_0','value_1','value_2','value_3',...,'value_N'], 'ticker','pub_date' 是唯一性约束
    """
    fin_df = fin_data_frame[['ticker', 'pub_date', 'end_date', col_name]]
    fin_df.dropna(inplace=True)
    pub_list = ['pub_date_' + str(i) for i in range(n)]
    value_list = ['value_' + str(i) for i in range(n)]

    def get_end_date(date):
        if is_annual:
            pre_end = str(int(date[:4]) - 1) + '1231'
        else:
            if date[4:6] == '03':
                pre_end = str(int(date[:4]) - 1) + '1231'
            elif date[4:6] == '06':
                pre_end = date[:4] + '0331'
            elif date[4:6] == '09':
                pre_end = date[:4] + '0630'
            elif date[4:6] == '12':
                pre_end = date[:4] + '0930'
        return pre_end

    def get_nlatest_perticker(df, col_name):
        tmp_df = df.copy()
        tmp_df.sort_values(['pub_date', 'end_date'], inplace=True)

        tmp_df_1 = tmp_df.copy()
        tmp_df_1.rename(columns={col_name: 'value_0', 'pub_date': 'pub_date_0', 'end_date': 'end_date_0'},
                        inplace=True)

        for i in range(1, n):
            # 标记i期前的财报日期
            tmp_df_1['end_date_' + str(i)] = tmp_df_1['end_date_' + str(i - 1)].apply(get_end_date)
            # 计算i期前财务数据
            tmp_df_2 = tmp_df.rename(columns={col_name: 'value_' + str(i), 'pub_date': 'pub_date_' + str(i),
                                              'end_date': 'end_date_' + str(i)})
            tmp_df_1 = tmp_df_1.merge(tmp_df_2, on=['ticker', 'end_date_' + str(i)], how='left')

        # 计算增长率
        if tmp_df_1.empty:
            return None
        else:
            # 去重
            # 标记最大pub_date，为记录可用时间
            tmp_df_1['max_pub_date'] = np.max(tmp_df_1[pub_list].fillna(method='ffill', axis=1), axis=1)
            tmp_df_1['max_pub_date'] = tmp_df_1['max_pub_date'].astype(np.int64).astype(np.str)
            tmp_df_1.sort_values(['max_pub_date', 'end_date_0'], inplace=True)

            if is_duplicate:
                tmp_df_1 = tmp_df_1.drop_duplicates(subset=['max_pub_date'], keep='last')
                tmp_df_1['max_end_date'] = tmp_df_1['end_date_0'].rolling(window=8, min_periods=1).max()
                tmp_df_1['max_end_date'] = tmp_df_1['max_end_date'].astype(np.int64).astype(np.str)
                tmp_df_1 = tmp_df_1[tmp_df_1['end_date_0'] == tmp_df_1['max_end_date']]  # 得到最新财报数据

                return tmp_df_1[['ticker', 'max_pub_date', 'max_end_date'] + value_list].rename(
                    columns={'max_pub_date': 'pub_date', 'max_end_date': 'end_date'})
            else:
                return tmp_df_1[['ticker', 'max_pub_date', 'end_date_0'] + value_list].rename(
                    columns={'max_pub_date': 'pub_date', 'end_date_0': 'end_date'})
    fin_df = fin_df.groupby(['ticker']).apply(get_nlatest_perticker, col_name)

    fin_df.reset_index(inplace=True, drop=True)
    return fin_df

def get_fin_data_latest(fin_data_frame, col_name=['value']):
    """
    获取最新财务数据， 也就是说给定publishDate，当时能取到的最新数据
    :param fin_data_frame: financial column= ['ticker','pub_date',’end_date',[fin_value]], index=num, pub_date='%Y%m%d'
    :param col_name: list, column name of value, 可以有多个列
    :return: column= ['ticker','pub_date','end_date',[fin_value]], 'ticker','pub_date' 是唯一性约束
    """

    fin_df = fin_data_frame.copy()

    def get_latest_perticker(df, col_name):
        tmp_df = df.copy()

        tmp_df.dropna(subset=col_name, how='all', inplace=True)
        tmp_df.sort_values(['pub_date', 'end_date'], inplace=True)
        tmp_df.drop_duplicates(subset=['pub_date'], keep='last', inplace=True)
        tmp_df['max_end_date'] = tmp_df['end_date'].rolling(window=8, min_periods=1).max()
        tmp_df['max_end_date'] = tmp_df['max_end_date'].astype(np.int64).astype(np.str)
        tmp_df = tmp_df[tmp_df['end_date'] == tmp_df['max_end_date']]
        return tmp_df[['ticker', 'pub_date', 'end_date'] + col_name]

    fin_df = fin_df.groupby(['ticker']).apply(get_latest_perticker, col_name)
    fin_df.reset_index(inplace=True, drop=True)
    return fin_df

# 将PIT数据转成连续数据
def fin_data_pit2cont(pit_data_frame, sdate, edate, fill_days=360):  
    """
    将PIT数据转成连续数据
    pit_data_frame: 财务报表数据, column= ['ticker','pub_date',[fin_value]], index=num, pub_date='%Y%m%d'
    sdate: 起始时间, '%Y%m%d'
    edate: 终止时间, '%Y%m%d'
    fill_days: fillna最长回溯时间
    返回：
         连续日的因子值dataframe, 列为：['ticker','pub_date',[fin_value]]
    """

    trade_date_frame = DataAPI.TradeCalGet(exchangeCD=u"XSHE", beginDate='20080101', endDate=edate,
                                           field=['calendarDate', 'isMonthEnd'])
    trade_date_frame.rename(columns={"calendarDate": "pub_date"}, inplace=True)
    trade_date_frame['pub_date'] = trade_date_frame['pub_date'].apply(lambda x: str(x).replace('-', ''))

    tmp_frame = pit_data_frame.groupby(['ticker']).apply(lambda x: x.merge(trade_date_frame,
                                                                           on=['pub_date'], how='outer'))
    del tmp_frame['ticker']
    tmp_frame.reset_index(inplace=True)
    del tmp_frame['level_1']

    tmp_frame = tmp_frame.sort_values(by=['ticker', 'pub_date'], ascending=True)
    tmp_frame = tmp_frame.groupby(['ticker']).apply(lambda x: x.fillna(method='pad', limit=fill_days))
    tmp_frame.dropna(inplace=True)
    tmp_frame = tmp_frame[(tmp_frame.pub_date >= sdate) & (tmp_frame.pub_date <= edate)]
    tmp_frame = tmp_frame[tmp_frame['isMonthEnd'] == 1]
    del tmp_frame['isMonthEnd']
    return tmp_frame

def fin_data_ttm(fin_data_frame, col_name='value'):
    """
    计算最新ttm值
    :param fin_data_frame:  column= ['ticker','pub_date','end_date','fis_period',[col_name]], index=num, pub_date='%Y%m%d'
    :param col_name: str, 财务字段名称
    :return: column= ['ticker','pub_date','end_date',[fin_value]], 'ticker','pub_date' 是唯一性约束
    """

    # 计算逻辑: ttm = 最近财报数据 + 去年年报数据 - 去年同期财报数据
    # 如果股票新上市,季报数据缺失,就选取最近能计算到的ttm值填充
    fin_df = fin_data_frame.copy()
    if 'fis_period' not in fin_df.columns:
        fin_df['fis_period'] = fin_df['end_date'].apply(lambda x: int(x[4:6]))
        
    def get_ttm_perticker(df, col_name):

        tmp_df = df.copy()
        tmp_df.sort_values(['pub_date', 'end_date'], inplace=True)

        # 标记去年年底和去年同期时间
        tmp_df_1 = tmp_df.copy()
        tmp_df_1.rename(columns={'fis_period': 'fis_period_std'}, inplace=True)
        tmp_df_1['last_year_end_date'] = tmp_df_1['end_date'].apply(lambda x: str(int(x[:4]) - 1) + '1231')
        tmp_df_1['start_end_date'] = tmp_df_1['end_date'].apply(lambda x: str(int(x[:4]) - 1) + x[4:8])

        # 计算去年年报数据
        tmp_df_2 = tmp_df[tmp_df['fis_period'] == 12]
        tmp_df_2.rename(columns={col_name: 'last_year_value', 'pub_date': 'last_year_pub_date',
                                 'end_date': 'last_year_end_date'}, inplace=True)
        tmp_df_1 = tmp_df_1.merge(tmp_df_2, on=['ticker', 'last_year_end_date'])

        # 计算去年同期数据
        tmp_df_3 = tmp_df.copy()
        tmp_df_3.rename(columns={col_name: 'start_value', 'pub_date': 'start_pub_date',
                                 'end_date': 'start_end_date'}, inplace=True)
        tmp_df_1 = tmp_df_1.merge(tmp_df_3, on=['ticker', 'start_end_date'])

        # 计算ttm
        tmp_df_1['ttm_'+col_name] = tmp_df_1[col_name] + tmp_df_1['last_year_value'] - tmp_df_1['start_value']
        tmp_df_1.loc[tmp_df_1['fis_period_std'] == 12, 'ttm_'+col_name] = tmp_df_1.loc[tmp_df_1['fis_period_std'] == 12, col_name]

        # 去重
        tmp_df_1.dropna(subset=['ttm_' + col_name], inplace=True)
        # 标记最大pub_date，为记录可用时间
        tmp_df_1['max_pub_date'] = np.max(tmp_df_1[['pub_date', 'last_year_pub_date', 'start_pub_date']], axis=1)
        tmp_df_1['max_pub_date'] = tmp_df_1['max_pub_date'].astype(np.int64).astype(np.str)
        tmp_df_1.sort_values(['max_pub_date', 'end_date'], inplace=True)
        tmp_df_1 = tmp_df_1.drop_duplicates(subset=['max_pub_date'], keep='last')
        tmp_df_1['max_end_date'] = tmp_df_1['end_date'].rolling(window=8, min_periods=1).max()
        tmp_df_1['max_end_date'] = tmp_df_1['max_end_date'].astype(np.int64).astype(np.str)
        tmp_df_1 = tmp_df_1[tmp_df_1['end_date'] == tmp_df_1['max_end_date']] #得到最新财报数据

        return tmp_df_1[['ticker', 'max_pub_date', 'max_end_date', 'ttm_'+col_name]]

    fin_df = fin_df.groupby(['ticker']).apply(get_ttm_perticker, col_name)
    fin_df.reset_index(inplace=True, drop=True)
    fin_df.rename(columns={'max_pub_date': 'pub_date', 'max_end_date': 'end_date',
                           'ttm_' + col_name: col_name}, inplace=True)
    return fin_df

def get_cap_by_day(tdate):
    '''
    获取给定日期的市值信息
    参数： 
        tdate, 时间，格式%Y%m%d
    返回:
        DataFrame, 返回给定日期的因子值
    '''
    cnt = 0
    while True:
        try:
            # 利用DataAPI查询因子
            x = DataAPI.MktEqudGet(tradeDate=tdate, isOpen="",field=u"ticker,tradeDate,negMarketValue",pandas="1")
            x['tradeDate'] = x['tradeDate'].apply(lambda x: x.replace('-', ''))
            x = x.rename(columns={'tradeDate': 'date', 'negMarketValue': 'cap'})

            return x
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                print('error get factor data: ', tdate, e)
                return pd.DataFrame()
            
            
            
# 保存地址
raw_data_dir = "./deep_rep_forecast"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

# 加载PIT的业绩预告数据的归母净利润数据
forecast_df = DataAPI.FdmtEfGet(field=[u'ticker', u'publishDate', u'endDate', u'reportType', u'expnIncAPLL', u'expnIncAPUPL'], pandas="1").dropna(subset=[u'expnIncAPLL', u'expnIncAPUPL'], how='all')
forecast_df = forecast_df[forecast_df['ticker'].apply(lambda x: x[0] not in ['2', '9'])]
forecast_df = forecast_df.sort_values(['ticker', 'endDate', 'publishDate', 'reportType'])
forecast_df = forecast_df.drop_duplicates(subset=['ticker', 'endDate', 'publishDate']) # 去除CQ3,Q3中的Q3
forecast_df['expnIncAPLL'] = forecast_df['expnIncAPLL'].fillna(forecast_df['expnIncAPUPL'])
forecast_df['expnIncAPUPL'] = forecast_df['expnIncAPUPL'].fillna(forecast_df['expnIncAPLL'])
forecast_df = forecast_df.rename(columns={'publishDate': 'pub_date', 'endDate': 'end_date'})
forecast_df['pub_date'] = forecast_df['pub_date'].apply(lambda x: x[:10].replace('-', ''))
forecast_df['end_date'] = forecast_df['end_date'].apply(lambda x: x.replace('-', ''))
forecast_df['forcastNIncome'] = (forecast_df['expnIncAPLL'] + forecast_df['expnIncAPUPL']) / 2 # 以预告中的上下限均值当作当季预测的净利润

# 加载PIT的真实利润表数据
income_df = DataAPI.FdmtISGet(beginYear='2007', reportType=['Q1', 'S1', 'CQ3', 'A'], field=[u'ticker', u'publishDate', u'endDate', u'NIncomeAttrP'], pandas="1")
income_df = income_df[income_df['ticker'].apply(lambda x: x[0] not in ['2', '9'])]
income_df = income_df.rename(columns={'publishDate': 'pub_date', 'endDate': 'end_date'})
income_df['pub_date'] = income_df['pub_date'].apply(lambda x: x.replace('-', ''))
income_df['end_date'] = income_df['end_date'].apply(lambda x: x.replace('-', ''))
# 计算单季度净利润
income_df = get_findata_nlatest(income_df, 2, col_name='NIncomeAttrP')
income_df['quaterNIncomeAttrP'] = income_df['value_0'] - income_df['value_1']
income_df = income_df.rename(columns={'value_0': 'NIncomeAttrP'}).drop('value_1', axis=1)

# 加载PIT的真实资产负债表
asset_df = DataAPI.FdmtBSGet(beginYear='2007', field=[u'ticker', u'publishDate', u'endDate', u'TEquityAttrP'], pandas="1")
asset_df = asset_df[asset_df['ticker'].apply(lambda x: x[0] not in ['2', '9'])]
asset_df = asset_df.rename(columns={'publishDate': 'pub_date', 'endDate': 'end_date'})
asset_df['pub_date'] = asset_df['pub_date'].apply(lambda x: x.replace('-', ''))
asset_df['end_date'] = asset_df['end_date'].apply(lambda x: x.replace('-', ''))


# ----------------------------------------------  均值偏离度因子
start_date = '2008-01-01'
end_date = '2019-01-31'

start_time = time.time()
# 计算均值偏离度因子
merge_data = pd.merge(forecast_df[[u'ticker', u'pub_date', u'end_date', u'reportType', 'forcastNIncome']], income_df, on=['ticker', 'end_date'])
merge_data['deviation'] = merge_data.apply(lambda x: (x['quaterNIncomeAttrP'] - x['forcastNIncome']) / np.abs(x['quaterNIncomeAttrP']+1e-10)  if x['reportType']=='Q3' else (x['NIncomeAttrP'] - x['forcastNIncome']) / np.abs(x['quaterNIncomeAttrP']+1e-10), axis=1)
merge_data = merge_data.dropna(subset=['deviation'])

merge_data['max_pub_date'] = np.max(merge_data[[u'pub_date_x', u'pub_date_y']].fillna(method='ffill', axis=1), axis=1)
merge_data['max_pub_date'] = merge_data['max_pub_date'].astype(np.int64).astype(np.str)
merge_data = merge_data[['ticker', 'max_pub_date', 'end_date', 'deviation']].rename(columns={'max_pub_date': 'pub_date'})

# 获取当天能取得的最新财报季信号，并填补给定日期范围内的月末信号
deviation_df = get_fin_data_latest(merge_data, col_name=['deviation']).drop('end_date', axis=1)
deviation_df = fin_data_pit2cont(deviation_df, start_date, end_date).rename(columns={'pub_date': 'date'})
print('因子数据格式为')
print(deviation_df.head(5).to_html())
print('Done! Cost time: %s' % (time.time() - start_time))

'''
下面分析了因子偏离度在通联全A的覆盖度情况。图中可以看出，2008-2012之间因子覆盖率较低，2012年之后覆盖率稳步提升至50%以上，这与12年之后报告数量的提升有关。本节截取20130101至今来测试该因子表现。

同时，查验该因子分布情况，发现因子有极值情况，下一小节将对因子进行去极值等处理。

注意: 该因子计算全部采用了PIT数据。

'''

factor_df = deviation_df.reset_index(drop=True).rename(columns={'deviation':'value'})
signal_discrip = analyse_general(factor_value_frame=factor_df, start_date=factor_df['date'].min(), end_date=factor_df['date'].max(), universe='TLQA', frequency='month')


'''
该部分对因子进行去极值、填充缺失值、中性化、标准化等操作。
用行业中位数填充空值
用MAD法处理5倍标准差外的异常值
利用优矿neutralize函数做中性化处理，主要是去除行业、市值的影响

'''

#--------------------------------去极值、填充值、中性化-----------------------------------------------
import datetime as dt
from dateutil.relativedelta import relativedelta

start_date = '20121201'
end_date = '20190131'

alpha_factors = ['value']
industry_df = DataAPI.EquIndustryGet(industryVersionCD=u"010303", field='ticker,intoDate,outDate,industryName1', pandas="1")
industry_df['intoDate'] = industry_df['intoDate'].apply(lambda x: str(x).replace('-', ''))
industry_df['outDate'] = industry_df['outDate'].apply(lambda x: str(x).replace('-', ''))

# 获取所有全A股票
equ_df = DataAPI.EquGet(equTypeCD=u"A", listStatusCD=u"", field=['secID', 'ticker', 'listDate', 'delistDate'], pandas="1")
equ_df['listDate'] = equ_df['listDate'].apply(str)
equ_df['delistDate'] = equ_df['delistDate'].apply(str)
equ_df['listDate'] = equ_df['listDate'].apply(lambda x: x.replace('-', ''))
equ_df['delistDate'] = equ_df['delistDate'].apply(lambda x: x.replace('-', ''))

def str2date(date_str):
    """转换日期格式 "YYYYMMDD" / "YYYY-MM-DD"string to datetime object"""
    date_str = date_str.replace("-", "")
    date_obj = dt.datetime(int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]))
    return date_obj

def get_universe(date_str, list_date=90):
    '''
    给定日期，选取符合条件的所有A股ticker
    '''
    format_date = str2date(date_str).strftime("%Y%m%d")
    list_date_need = (str2date(date_str) + relativedelta(days=-list_date)).strftime("%Y%m%d")
    A_ticker = set(equ_df[(equ_df['listDate'] <= list_date_need) & ((equ_df['delistDate'] > format_date) | (equ_df['delistDate'].isnull()))]['ticker'])
    
    cnt = 0
    while True:
        try:
            st_ticker = set(DataAPI.SecSTGet(beginDate=list_date_need, endDate=format_date, pandas="1")['ticker'])
            return A_ticker - st_ticker
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                return A_ticker
    

def winsorize_by_date(cdate_input):
    '''
    按照[dm+5*dm1, dm-5*dm1]进行去极值
    参数:
        cdate_input: 某一期的因子值的dataframe
    返回:
        DataFrame, 去极值后的因子值
    '''
    cdate_input = cdate_input.copy()
    dm = cdate_input.median()
    dm1 = (cdate_input - dm).abs().median()

    upper = dm + 5 * dm1
    lower = dm - 5 * dm1
    cdate_input[cdate_input > upper] = upper
    cdate_input[cdate_input < lower] = lower
    return cdate_input

def standardize_winsorize_neutralize_factor(input_data):
    """
    进行去极值、中性化、标准化
    输入：
        input_data：tuple, 传入的是(因子值，时间)。因子值为DataFrame
    返回：
        DataFrame, 处理后的数据
    """
    data, tdate = input_data
    cdate_input = data[data['date'] == tdate]
    cdate_input = cdate_input.set_index('ticker')
    # 去极值
    cdate_input.loc[:, alpha_factors] = cdate_input.loc[:, alpha_factors].apply(lambda x : winsorize_by_date(x))
    
    #填缺失值
    a_universe = get_universe(tdate)
    cdate_input = cdate_input.loc[a_universe]
    cdate_input['date'] = tdate
    industry = industry_df[(industry_df['intoDate']<=tdate) & ((industry_df['outDate']>tdate) | pd.isnull(industry_df['outDate']))]
    cdate_input = pd.merge(cdate_input, industry[['ticker', 'industryName1']].set_index('ticker'), left_index=True, right_index=True)
    cdate_input.loc[:, alpha_factors] = cdate_input.loc[:, alpha_factors].fillna(cdate_input.groupby('industryName1')[alpha_factors].transform("median"))
    cdate_input = cdate_input.fillna(0.0)
    
    # 标准化
    cdate_input.loc[:, alpha_factors] = cdate_input.loc[:, alpha_factors].apply(lambda x : standardize(x))
    
    # 中性化
    for a_factor in alpha_factors:
        sig = cdate_input[a_factor]
        cnt = 0
        while True:
            try:
                cdate_input.loc[:, a_factor] = neutralize(sig, target_date=tdate, exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'SIZENL'])       
                break
            except Exception as e:
                cnt += 1
                if cnt >= 3:
                    break
    # 中性化
    cdate_input.loc[:, alpha_factors] = cdate_input.loc[:, alpha_factors].apply(lambda x : standardize(x))
    return cdate_input.reset_index()[['ticker', 'date'] + alpha_factors]

if __name__ == "__main__":
    start_time = time.time()
    # 只保留start_date, end_date的因子值
    factor_df = factor_df[(factor_df['date']>=start_date) & (factor_df['date']<=end_date)]
    # 遍历每个月末日期，利用协程对因子及收益进行去极值、标准化，中性化处理
    print('standardize & neutralize & winsorize factor...')
    date_list = factor_df['date'].unique()
    jobs = [gevent.spawn(standardize_winsorize_neutralize_factor, value) for value in zip([factor_df]*len(date_list), date_list)]
    gevent.joinall(jobs)
    new_frame_list = [e.value for e in jobs]
    print ("ALL FINISHED")
    
    factor_csv = pd.concat(new_frame_list, axis=0)
    factor_csv.reset_index(drop=True, inplace=True)

    ################################ 数据存储下来 ################################
    factor_csv.to_csv(os.path.join(raw_data_dir, 'handle_forcast_income_alpha.csv'), chunksize=1000)
    end_time = time.time()
    print ("Time cost: %s seconds" % (end_time - start_time))
    
# 业绩预告偏离度因子的IC测试
tic = time.time()
date_list = sorted(factor_csv['date'].unique())

# 调用优矿的因子IC分析框架
signal_ic = analyse_IC(factor_value_frame=factor_csv, start_date=date_list[0], end_date=date_list[-1], frequency='month', corr_method='spearman', quantile_num=5, universe='TLQA', benchmark='TLQA', decay_list=[1, 2])

toc = time.time()
print ("\n ----- Computation time = " + str((toc - tic)) + "seconds")


# 业绩预告偏离度因子的收益率分析
tic = time.time()
signal_return = analyse_return(factor_value_frame=factor_csv, start_date='20130101', end_date='20181231', frequency='month', quantile_num=5, weight_type='equal', universe='TLQA', benchmark='TLQA', init_cash=100000000.0, decay_list=[0, 1])
toc = time.time()

print ("\n ----- Computation time = " + str((toc - tic)) + "seconds")



'''

从分组收益来看，因子单调性并不是很好，读者可尝试其他方法构建因子。另外，由于业绩预告数据较定期财报数据更为提前，可以用来增强财务因子的表现。
   调试 运行
文档
 代码  策略  文档
三、基于业绩预告提高财务因子的效果
该部分耗时 30分钟左右
本节选取了EP_TTM及ROE进行分析，在原有的财报数据中加入业绩预告数据进而计算因子。对比加入业绩预告数据后的因子与原有因子的回测结果。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 EP_TTM, ROE因子计算

3.1.1 EP_TTM因子

净利润采用TTM计算

'''

# ----------------------------------------------  EP_TTM
start_date = '20121201'
end_date = '20190131'

def cap_ep(df, cap_df, col_name):
    # 计算TTM的净利润
    ttm_df = fin_data_ttm(df.copy(), col_name=col_name)
    latest_ttm_df = get_fin_data_latest(ttm_df, col_name=[col_name]).drop('end_date', axis=1)
    fill_ttm_df = fin_data_pit2cont(latest_ttm_df, start_date, end_date)

    # 合并市值，计算EP_TTM
    ep_df = pd.merge(fill_ttm_df.rename(columns={'pub_date': 'date'}), cap_df, on=['ticker', 'date'])
    ep_df['ep'] = ep_df[col_name] / ep_df['cap']
    
    return ep_df[['ticker', 'date', 'ep']]

start_time = time.time()

# 拿到交易日历，得到月末日期
trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
trade_date = trade_date[trade_date.isMonthEnd == 1]

print("begin to get factor value for each stock...", time.asctime())
# # 取得每个月末日期，所有股票的市值
pool = ThreadPool(processes=16)
date_list = [tdate.replace("-", "") for tdate in trade_date.calendarDate.values]
frame_list = pool.map(get_cap_by_day, date_list)
pool.close()
pool.join()
cap_df = pd.concat(frame_list, axis=0)

print("begin to cal ep_ttm ...", time.asctime())
ep_df = cap_ep(income_df[income_df['end_date']>'20110101'], cap_df, 'NIncomeAttrP')

print("begin to cal ep_ttm with forecast income ...", time.asctime())
# 合并预告收入
income_with_forecast_df = pd.concat([income_df[['ticker', 'pub_date', 'end_date', 'NIncomeAttrP']], forecast_df[['ticker', 'pub_date', 'end_date', 'forcastNIncome']].rename(columns={'forcastNIncome': 'NIncomeAttrP'})])
forecast_ep_df = cap_ep(income_with_forecast_df[income_with_forecast_df['end_date']>'20110101'], cap_df, 'NIncomeAttrP')

print('Done! Cost time: %s seconds' % (time.time() - start_time))

'''

3.1.2 ROE因子计算

ROE中的净利润选择当季度净利润，资产选择期初资产与期末资产均值。对于业绩预告计算的ROE来说，期末资产当时尚未发布，假设业绩预告中的净利润加上起初资产为期末资产。

'''
# ----------------------------------------------  ROE
start_date = '20121201'
end_date = '20190131'

start_time = time.time()

def get_end_date(date, is_annual=False):
    if is_annual:
        pre_end = str(int(date[:4]) - 1) + '1231'
    else:
        if date[4:6] == '03':
            pre_end = str(int(date[:4]) - 1) + '1231'
        elif date[4:6] == '06':
            pre_end = date[:4] + '0331'
        elif date[4:6] == '09':
            pre_end = date[:4] + '0630'
        elif date[4:6] == '12':
            pre_end = date[:4] + '0930'
    return pre_end

# 计算财报中的ROE
print('begin to cal roe ...', time.asctime())
asset_df_copy = get_findata_nlatest(asset_df, 2, col_name='TEquityAttrP')
asset_df_copy['TEquityAttrP'] = (asset_df_copy['value_0'] + asset_df_copy['value_1']) / 2.0
merge_data2 = pd.merge(income_df, asset_df_copy, on=['ticker', 'end_date'])
merge_data2['roe'] = merge_data2['quaterNIncomeAttrP'] / merge_data2['TEquityAttrP']
merge_data2['max_pub_date'] = np.max(merge_data2[[u'pub_date_x', u'pub_date_y']].fillna(method='ffill', axis=1), axis=1)
merge_data2['max_pub_date'] = merge_data2['max_pub_date'].astype(np.int64).astype(np.str)
merge_data2 = merge_data2[['ticker', 'max_pub_date', 'end_date', 'roe']].rename(columns={'max_pub_date': 'pub_date'}).dropna()
latest_roe_df = get_fin_data_latest(merge_data2, col_name=['roe']).drop('end_date', axis=1)
roe_df = fin_data_pit2cont(latest_roe_df, start_date, end_date, fill_days=180).rename(columns={'pub_date': 'date'})

# 合并业绩预告净利润与财报净利润数据
print('begin to cal quarter forecast data ...', time.asctime())
forecast_df['pre_end_date'] = forecast_df['end_date'].apply(lambda x: get_end_date(x))
forecast_df = pd.merge(forecast_df[['ticker', 'pub_date', 'end_date', 'reportType', 'forcastNIncome', 'pre_end_date']], income_df, left_on=['ticker', 'pre_end_date'], right_on=['ticker', 'end_date'], how='left')
forecast_df['max_pub_date'] = np.max(forecast_df[['pub_date_x', 'pub_date_y']].fillna(method='ffill', axis=1), axis=1)
forecast_df['max_pub_date'] = forecast_df['max_pub_date'].astype(np.int64).astype(np.str)

# 计算业绩预告的预测当季度数据，也就说去除已发布的财报数据
forecast_df['forcastNIncome'] = forecast_df.apply(lambda x: x['forcastNIncome']+x['NIncomeAttrP'] if x['reportType']=='Q3' else x['forcastNIncome'], axis=1)
forecast_df['forcastQuarterNIncome'] = forecast_df['forcastNIncome'] - forecast_df['NIncomeAttrP']
forecast_df = forecast_df[['ticker', 'max_pub_date', 'end_date_x', 'forcastNIncome', 'forcastQuarterNIncome', 'pre_end_date']].rename(columns={'max_pub_date': 'pub_date', 'end_date_x': 'end_date'})

# 计算业绩预告中的ROE
print('begin to cal roe with forecast income ...', time.asctime())
merge_data1 = pd.merge(forecast_df, asset_df, left_on=['ticker', 'pre_end_date'], right_on=['ticker', 'end_date'])
merge_data1['roe'] = merge_data1['forcastQuarterNIncome'] * 2.0 / (merge_data1['TEquityAttrP'] * 2 + merge_data1['forcastQuarterNIncome']) 
merge_data1['max_pub_date'] = np.max(merge_data1[[u'pub_date_x', u'pub_date_y']].fillna(method='ffill', axis=1), axis=1)
merge_data1['max_pub_date'] = merge_data1['max_pub_date'].astype(np.int64).astype(np.str)
merge_data1 = merge_data1[['ticker', 'max_pub_date', 'end_date_x', 'roe']].rename(columns={'max_pub_date': 'pub_date', 'end_date_x': 'end_date'}).dropna()
latest_forecast_roe_df = get_fin_data_latest(pd.concat([merge_data1, merge_data2]), col_name=['roe']).drop('end_date', axis=1).dropna()
forecast_roe_df = fin_data_pit2cont(latest_forecast_roe_df, start_date, end_date, fill_days=180).rename(columns={'pub_date': 'date'})

print('Done! Cost time: %s seconds' % (time.time() - start_time))


# 合并上述计算的ep_ttm, roe因子
factor_df = reduce(lambda left, right: pd.merge(left, right, on=['date', 'ticker'], how='outer'), [ep_df, forecast_ep_df, roe_df, forecast_roe_df])
factor_df.columns = ['ticker', 'date', 'ep', 'ep_forecast', 'roe', 'roe_forecast']
factor_df.to_csv(os.path.join(raw_data_dir, 'forcast_ep_roe.csv'), chunksize=1000)

print('因子数据格式为')
print(factor_df.head(5).to_html())
# 计算业绩预告增强因子前后的相关性
print(u'业绩预告数据增强EP_TTM因子前后相关性: %.2f%%' % (factor_df[['ep', 'ep_forecast']].corr().round(4).iloc[0,1] * 100))
print(u'业绩预告数据增强ROE因子前后相关性: %.2f%%' % (factor_df[['roe', 'roe_forecast']].corr().round(4).iloc[0,1] * 100))


'''

3.1.3 因子预处理

同样对因子进行去极值、填缺失值、中性化、标准化等操作。

'''

start_time = time.time()

# 遍历每个月末日期，利用协程对因子及收益进行去极值、标准化，中性化处理
print('standardize & neutralize & winsorize factor...')
alpha_factors = ['ep', 'ep_forecast', 'roe', 'roe_forecast']
date_list = factor_df['date'].unique()
jobs = [gevent.spawn(standardize_winsorize_neutralize_factor, value) for value in zip([factor_df]*len(date_list), date_list)]
gevent.joinall(jobs)
new_frame_list = [e.value for e in jobs]
print ("ALL FINISHED")

factor_csv = pd.concat(new_frame_list, axis=0)
factor_csv.reset_index(drop=True, inplace=True)

################################ 数据存储下来 ################################
factor_csv.to_csv(os.path.join(raw_data_dir, 'handle_forcast_ep_roe.csv'), chunksize=1000)
end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

'''

3.2 IC比较

对上述计算的EP_TTM, ROE进行IC分析。
从结果上来看，加入了业绩预告数据后，原有的EP_TTM的ICIR从2.33提升至2.63，ROE因子的ICIR从1.17提升至1.52。说明业绩预告数据确实可以帮助我们提升现有因子效果。


'''


import pandas as pd
import numpy as np
import scipy.stats as st

def calc_ic(signal_df, return_df, factor_name, ret_name, method='spearman'):
    """
    计算因子IC值, 本月和下月因子值的秩相关
    params: 
            signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一列为股票当日的因子值
            return_df: DataFrame, colunms=['ticker, 'tradeDate'， [next_period_ret]], 收益率， next_period_ret一列为下月的收益率
            factor_name:　str, signal_df中因子值的列名
            ret_name: str, return_df中收益率的列名
            method: : {'spearman', 'pearson'}, 默认'spearman', 指定计算rank IC('spearman')或者Normal IC('pearson')
    return:
            DataFrame, 返回IC值和本月和下月因子值的秩相关
    """
    if isinstance(factor_name, str):
        factor_name = [factor_name]
    merge_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'])
    # 计算IC
    ic_df = merge_df.groupby('tradeDate').apply(
        lambda x: pd.Series(data=x[factor_name + [ret_name]].corr(method=method).values[0, 1:], index=factor_name)).dropna()
    # 计算邻月IC
    factor_name_next = ["%s_next"%item for item in factor_name]
    merge_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    merge_df[factor_name_next] = merge_df.groupby('ticker')[factor_name].shift(-1)
    merge_df.dropna(inplace=True)
    next_ic_df = merge_df.groupby('tradeDate').apply(
        lambda x: pd.Series(data=x[factor_name + factor_name_next].corr(method='spearman').values[0:len(factor_name), len(factor_name):].diagonal(), index=["%s_next_ic"%item for item in factor_name]))

    result = pd.concat([ic_df, next_ic_df], axis=1).dropna()

    return result

def ic_describe(ic_df):
    """
    统计IC的均值、标准差、IC_IR、大于0的比例以及下月IC相关系数均值
    params:
            ic_df: DataFrame, IC值， index为日期， columns为因子名， values为各个因子的IC值
    return:
            DataFrame, IC统计
    """
    ic_df = ic_df.dropna()
    # 记录因子个数和因子名
    factor_name = [fname for fname in ic_df.columns.values if '_next_ic' not in fname]
    n = len(factor_name)
    # IC均值
    ic_mean = ic_df[factor_name].mean()
    # IC标准差
    ic_std = ic_df[factor_name].std()
    # IC均值的T统计量
    ic_t = pd.Series(st.ttest_1samp(ic_df[factor_name].values, 0)[0], index=factor_name)
    # IC_IR
    ic_ir = ic_mean / ic_std * np.sqrt(12.0)
    # IC>0的比例
    ic_p_pct = (ic_df[factor_name] > 0).sum() / len(ic_df)
    # 下月IC相关系数均值
    ic_auto_corr = ic_df[[fname + '_next_ic' for fname in factor_name]].mean()
    ic_auto_corr.index = factor_name

    # IC统计
    ic_table = pd.DataFrame([ic_mean, ic_std, ic_t, ic_ir, ic_p_pct, ic_auto_corr],
                            index=[u'平均IC', u'IC标准差', u'IC均值T统计量', u'年化IC_IR', u'IC大于0的比例', u'下月IC相关系数均值'])
    return ic_table.T


import pickle

# 设置起始时间和结束时间
begin_date = '20130101'
end_date = '20190131'

# 获得月收益率
month_return = DataAPI.MktEqumAdjGet(beginDate=begin_date, endDate=end_date, field=u"ticker,endDate,chgPct,",pandas="1")
month_return.rename(columns={'endDate': 'tradeDate', 'chgPct': 'month_return'}, inplace=True)
month_return.sort_values(['ticker', 'tradeDate'], inplace=True)
month_return['next_month_return'] = month_return.groupby('ticker')['month_return'].shift(-1)
month_return.dropna(inplace=True)
month_return['tradeDate'] = month_return['tradeDate'].apply(lambda x : x.replace("-", ''))

factor_csv = pd.read_csv(os.path.join(raw_data_dir, 'handle_forcast_ep_roe.csv'), index_col=0, dtype={'ticker': str, 'date': str})
method_list = ['ep', 'ep_forecast', 'roe', 'roe_forecast']
ic_res_list = []
for method in method_list:
    signal_df = factor_csv[['ticker', 'date', method]].rename(columns={'date': 'tradeDate'})
    ic = calc_ic(signal_df, month_return, method, 'next_month_return')
    ic_des = ic_describe(ic)

    ic_res_list.append(ic_des)
        
ic = pd.concat(ic_res_list, axis=0).round(4)
ic.index = method_list

print('rankIC 情况')
print(ic.to_html())    


'''

3.3 组合回测

对上述因子构建组合进行回测，共测试了2种组合形式: Top100等权选股组合，long-short多空组合，具体参数如下:

选股池: 中证全指成分股
时间范围: 20130101-20181231
调仓参数: 月度调仓，买卖交易费千分之1.5

'''

import time
from CAL.PyCAL import * 

start_time = time.time()
# -----------回测参数部分开始，可编辑------------
start = '2013-01-01'                       # 回测起始时间
end = '2018-12-31'                         # 回测结束时间
benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('000985.ZICN')           # 证券池，支持股票和基金
capital_base = 1000000000.0                     # 起始资金
freq = 'd'                              
refresh_rate = Monthly(1)  
commission = Commission(buycost=0.0015, sellcost=0.0015, unit='perValue')

# 把回测参数封装到 SimulationParameters 中，供 quick_backtest 使用
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate)
# 获取回测行情数据
data = quartz.get_backtest_data(sim_params)

backtest_results_dict = {}

method_list = ['ep', 'ep_forecast', 'roe', 'roe_forecast']
# 对4种不同类型分别进行测试
for method in method_list:
    factor_data = factor_csv[['ticker', 'date', method]].rename(columns={'date': 'tradeDate', method: 'alpha'})
    factor_data = factor_data.set_index('tradeDate')
    factor_data['ticker'] = factor_data['ticker'].apply(lambda x: x+'.XSHG' if x[:2] in ['60'] else x+'.XSHE')
    
    q_dates = factor_data.index.unique()
    
    for portfolio_type in ['long', 'short', 'top100']: 
        print ('backtesting for method %s portfolio %s ..................................' % (method, portfolio_type))
        
        # 注册一个账户
        accounts = {'fantasy_account': AccountConfig(account_type='security', capital_base=capital_base, commission=commission)}
        sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate, accounts=accounts)

        # ---------------策略逻辑部分----------------

        def initialize(context):                   # 初始化虚拟账户状态
            pass

        def handle_data(context): 
            account = context.get_account('fantasy_account')
            current_universe = context.get_universe('stock', exclude_halt=True)
            pre_date = context.previous_date.strftime("%Y%m%d")
            if pre_date not in q_dates:            
                return
            wts = {}
            # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
            q = factor_data.ix[pre_date].dropna()
            q = q.set_index('ticker', drop=True)
            q = q.ix[current_universe].dropna()
            if len(q) == 0:
                print('no alpha ', pre_date)
                print(current_universe)
            
            if portfolio_type == 'top100':
                q_top = q.nlargest(100, 'alpha')
                my_univ = q_top.index.values
            elif portfolio_type == 'long':
                q_min = q['alpha'].quantile(0.8)
                q_max = q['alpha'].quantile(1.0)
                my_univ = q[(q['alpha']>q_min) & (q['alpha']<=q_max)].index.values
            elif portfolio_type == 'short':
                q_min = q['alpha'].quantile(0.0)
                q_max = q['alpha'].quantile(0.2)
                my_univ = q[(q['alpha']>q_min) & (q['alpha']<=q_max)].index.values
            else:
                print('no portfolio_type: %s' % portfolio_type)
                return
           # 交易部分
            positions = account.get_positions()
            sell_list = [stk for stk in positions if stk not in my_univ]
            for stk in sell_list:
                account.order_to(stk, 0)

            # 在目标股票池中的，买入
            for stk in my_univ:
                account.order_pct_to(stk, wts.get(stk, 1.0/len(my_univ)))

        # 生成策略对象
        strategy = quartz.TradingStrategy(initialize, handle_data)
        # ---------------策略定义结束----------------

        # 开始回测
        bt, perf = quartz.quick_backtest(sim_params, strategy, data=data)

        # 保存运行结果
        backtest_results_dict[method + "_" + portfolio_type] = {'max_drawdown': perf['max_drawdown'], 'sharpe': perf['sharpe'], 'alpha': perf['alpha'], 'beta': perf['beta'], 'information_ratio': perf['information_ratio'], 'annualized_return': perf['annualized_return'], 'bt': bt[[u'tradeDate', u'portfolio_value', u'benchmark_return']]}    
        
    
# 保存该次回测结果为文件
with open(os.path.join(raw_data_dir, 'forecast_backtest.pickle'), 'wb') as handle:
    pickle.dump(backtest_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ('Done! Time Cost: %s seconds' % (time.time()-start_time))


# 读取上述回测的结果，都存在了backtest_results_dict里，便于后续组合分析

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
import os
import pandas as pd
import numpy as np
import time
import pickle
from CAL.PyCAL import * 

with open(os.path.join(raw_data_dir, 'forecast_backtest.pickle'), 'rb') as fHandler:
    backtest_results_dict = pickle.load(fHandler)

capital_base = 1000000000.0

key_list = ['ep', 'ep_forecast', 'roe', 'roe_forecast']


'''

3.3.1 Top100组合分析

该小节分析了top100的纯多头组合表现
结果上看，加入业绩预告后的EP_TTM, ROE均比原有因子有所提升。EP_TTM因子Top100组合收益从10.7%提升到13.1%，ROE因子Top100组合收益从7.8提升到8.4%。

'''


backtest_origin_indic = [u'annualized_return', u'sharpe', u'max_drawdown']

def get_top100_result(results):  
    """
    top100组合的回测结果展示及分析
    params:
        results: dict, 回测结果
    return:
        DataFrame, 返回计算的指标
    """        
    backtest_pd = pd.DataFrame(index=key_list, columns=backtest_origin_indic+[u'volatility'])
    
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.grid()

    for key in key_list:
        bt = results[key+"_top100"]['bt']
        data = bt[[u'tradeDate', u'portfolio_value', u'benchmark_return']]
        data['portfolio_return'] = data.portfolio_value / data.portfolio_value.shift(1) - 1.0  # 总头寸每日回报率
        data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0] / capital_base - 1.0
        data['portfolio'] = data.portfolio_return + 1.0
        data['portfolio'] = data.portfolio.cumprod()  # 总头寸不对冲时的净值序列
        volatility = np.std(data['portfolio_return']) * np.sqrt(252)
        backtest_pd.loc[key] = np.array([results[key+"_top100"][item] for item in backtest_origin_indic] + [volatility])
        ax.plot(data['tradeDate'], data[['portfolio']], label=key)

    backtest_pd.columns = [u'年化收益', u'夏普比率', u'最大回撤', u'年化波动']
    backtest_pd.index.name = u'不同类型'
    
    ax.legend(loc=0)
    ax.set_ylabel(u"净值", fontproperties=font, fontsize=16)
    ax.set_title(u"Top100组合净值走势", fontproperties=font, fontsize=16)

    return backtest_pd.astype(float).round(4)

backtest_pd = get_top100_result(backtest_results_dict)
print(backtest_pd.to_html())


'''


4.2.2 long-short组合分析

该小节分析了Top20% - Bottom20%的多空组合走势
结果上看，加入业绩预告数据后, EP_TTM因子多空组合收益从11.3%提升到13.1%，夏普比从1.79提升到2.08；ROE因子则相对原有因子有所降低，主要是因为空头组合不一致导致。


'''

backtest_heged_indic = [u'年复合收益', u'夏普比率', u'最大回撤', u'收益波动率']

def get_long_short_result(results):  
    """
    多空组合回测结果展示及分析
    params:
        results: dict, 回测结果
    return:
        DataFrame, 返回计算的指标
    """        
    
    backtest_pd = pd.DataFrame(index=key_list, columns=backtest_heged_indic)
    
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.grid()
    
    for key in key_list:   
        data_list = []
        for portfolio in ['long', 'short']:
            data = results["%s_%s"%(key, portfolio)]['bt'][[u'tradeDate', u'portfolio_value']]
            data['portfolio_return'] = data.portfolio_value / data.portfolio_value.shift(1) - 1.0  # 总头寸每日回报率
            data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0] / capital_base - 1.0
            data_list.append(data.set_index('tradeDate')['portfolio_return'])
        long_short_return = (data_list[0] - data_list[1]).fillna(0.0)
        long_short_value = (long_short_return + 1.0).cumprod()
        
        hedged_max_drawdown = max([1 - v / max(1, max(long_short_value[:i + 1])) for i, v in enumerate(long_short_value)])  # 对冲后净值最大回撤
        hedged_volatility = np.std(long_short_return) * np.sqrt(252)
        hedged_annualized_return = (long_short_value.values[-1]) ** (252.0 / len(long_short_value)) - 1.0
        sharpe_ratio = hedged_annualized_return / hedged_volatility
        backtest_pd.loc[key] = [hedged_annualized_return, sharpe_ratio, hedged_max_drawdown, hedged_volatility]

        ax.plot(long_short_value.index, long_short_value.values, label=key)

    ax.legend(loc=0)
    ax.set_ylabel(u"净值", fontproperties=font, fontsize=16)
    ax.set_title(u"long-short组合净值走势", fontproperties=font, fontsize=16)

    return backtest_pd.astype(float).round(4)

backtest_pd = get_long_short_result(backtest_results_dict)
print(backtest_pd.to_html())


