# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:43:13 2020

@author: Asus
"""
"""

导读
A. 研究目的：本文利用优矿提供的分钟行情数据和因子数据，参考天风证券《基于净利润断层的选股策略》（原作者：张欣慰，陈可）中超预期事件和盈余跳空因子思路，验证其有效性，并进一步构建基本面和技术面共振的选股策略。

B. 研究结论：

利用研报中超预期的样本触发的事件具有超额收益。其中，业绩预告类型超预期事件的超额能力、持续性最强，财报居中，快报最差；
盈余跳空因子JOR，相对于传统的盈余漂移因子EAR，在A股市场的选股能力更好其更稳定。全中性后的JOR因子的IC为2.21%，月度胜率为79.84%，多空年化收益为8.87%，夏普比为2.16。JOR因子对超预期事件也有显著的区分能力。
利用超预期事件构造的超预期样本空间具有超额收益。叠加JOR因子，构造净利润断层金股组合，在201607-202005期间，组合稳定持续跑赢中证500，且年华超额收益达到21.0%，月度胜率达到76.09％。
C. 文章结构：本文共分为4个部分，具体如下

一、数据准备
二、超预期事件的测试
三、盈余公告跳空因子
四、净利润断层组合构建
D. 时间说明

一、第一部分运行需要6分钟
二、第二部分运行需要1分钟
三、第三部分运行需要4分钟
四、第三部分运行需要2分钟
总耗时13分钟左右 (为了方便修改程序，文中将很多中间数据存储下来以缩短运行时间)
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
https://uqer.datayes.com/community/share/eLNeQy0p3r0lRu9I5WoZ5YOw2ng0/private；密码：6278
请在运行之前，克隆上面的代码，并存成lib（右上角->另存为lib,不要修改名字）

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

"""

import pandas as pd
import numpy as np
import os
import datetime
import time
import lib.quant_util as qutil
import scipy.stats as st
from CAL.PyCAL import *
import matplotlib.pyplot as plt
import seaborn as sns

# 建立缓存数据的文件夹
file_path = './jor'
if not os.path.exists(file_path):
    os.makedirs(file_path)
    
"""

第一部分：数据准备
该部分耗时 6分钟
该部分内容为：

1.1 获取基础行情数据，包括：个股日度行情数据和月度收益率数据、中证500的日度行情数据和月度收益数据，以及后续需要剔除的股票池（上市不满60个交易日的次新股、st股、停牌个股、一字板个股）。时间范围为20100101-20200531.
1.2 获取个股的盈余公告数据，包括业绩预告，业绩快报、定期财报。其中业绩预告剔除公告覆盖财务日期距离公告发布日期大于365天或者小于-125天的记录；定期财报只保留第一次公布的记录。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

### 1.1 基础行情数据

"""


start_time = time.time()
print"该部分进行基础参数设置和数据准备..."

start_date = '20100101'
end_date = '20200531'

# 全A投资域
a_universe_list = DataAPI.EquGet(equTypeCD=u"A",listStatusCD=u"L,S,DE",field=u"secID",pandas="1")['secID'].tolist()
a_universe_list.remove('DY600018.XSHG')

# 交易日历
cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate='', endDate='20201231').sort('calendarDate')
cal_dates_df['calendarDate'] = cal_dates_df['calendarDate'].apply(lambda x: x.replace('-', ''))
cal_dates_df['prevTradeDate'] = cal_dates_df['prevTradeDate'].apply(lambda x: x.replace('-', ''))
daily_trade_list = sorted(cal_dates_df.query("isOpen==1")['calendarDate'].tolist())

# 个股日度行情数据
if not os.path.exists(os.path.join(file_path, 'dmkt.pkl')):
    dmkt_df = DataAPI.MktEqudGet(secID=a_universe_list, beginDate=start_date, endDate=end_date, isOpen="", field=u"ticker,tradeDate,preClosePrice,highestPrice,lowestPrice,chgPct", pandas="1")
    dmkt_df['tradeDate'] = dmkt_df['tradeDate'].apply(lambda x: x.replace('-', ''))
    dmkt_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    dmkt_df.to_pickle(os.path.join(file_path, 'dmkt.pkl'))
else:
    dmkt_df = pd.read_pickle(os.path.join(file_path, 'dmkt.pkl'))
print ("个股日度行情数据:", dmkt_df.head().to_html())

# 股票池筛选：上市不满60个交易日的次新股、st股、停牌个股、一字板个股
if not os.path.exists(os.path.join(file_path, 'forbidden.pkl')):
    forbidden_pool = qutil.stock_special_tag(start_date, end_date, pre_new_length=60)
    # 筛选一字板个股
    mkt_df = dmkt_df.copy()
    limit_df = mkt_df[(mkt_df['highestPrice'] == mkt_df['lowestPrice']) & (mkt_df['highestPrice']>0)][['ticker', 'tradeDate']]
    limit_df['special_flag'] = 'limit'
    forbidden_pool = forbidden_pool.append(limit_df)
    forbidden_pool = forbidden_pool.merge(cal_dates_df, left_on=['tradeDate'], right_on=['calendarDate'])
    forbidden_pool = forbidden_pool[['ticker', 'tradeDate', 'prevTradeDate', 'special_flag']]
    forbidden_pool.to_pickle(os.path.join(file_path, 'forbidden.pkl'))
else:
    forbidden_pool = pd.read_pickle(os.path.join(file_path, 'forbidden.pkl'))
print ("禁止股票池:", forbidden_pool.head().to_html())

# 获取个股月度收益率
if not os.path.exists(os.path.join(file_path, 'mret.pkl')):
    mret_df = DataAPI.MktEqumAdjGet(beginDate=start_date, endDate=end_date, secID=a_universe_list, field=u"ticker,endDate,chgPct", pandas="1")
    mret_df.rename(columns={'endDate':'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
    mret_df['tradeDate'] = mret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
    mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    mret_df['nxt1m_ret'] = mret_df.groupby('ticker')['curr_ret'].shift(-1)
    mret_df.to_pickle(os.path.join(file_path, 'mret.pkl'))
else:
    mret_df = pd.read_pickle(os.path.join(file_path, 'mret.pkl'))
print ("个股月度收益率:", mret_df.head().to_html())

# zz500日度行情
if not os.path.exists(os.path.join(file_path, 'idx_ret.pkl')):
    idx_ret_df = DataAPI.MktIdxdGet(indexID=u"",ticker=u"000905",tradeDate=u"",beginDate=start_date,endDate=end_date,exchangeCD=u"XSHE,XSHG",field=u"tradeDate,preCloseIndex,lowestIndex,CHGPct",pandas="1")
    idx_ret_df['tradeDate'] = idx_ret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
    idx_ret_df.to_pickle(os.path.join(file_path, 'idx_ret.pkl'))
else:
    idx_ret_df = pd.read_pickle(os.path.join(file_path, 'idx_ret.pkl'))
print ("zz500日度行情:", idx_ret_df.head().to_html())

# zz500月度收益
idx_mret_df = DataAPI.MktIdxmGet(beginDate=start_date,endDate=end_date,indexID=u"000905.ZICN",ticker=u"",field=u"endDate,chgPct",pandas="1")
idx_mret_df.columns = ['tradeDate', 'mret']
idx_mret_df['tradeDate'] = idx_mret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
print ("zz500月度收益率:", idx_mret_df.head().to_html())

# 计算个股超额收益
dret_df = dmkt_df.pivot_table(values='chgPct', index='tradeDate', columns='ticker')
for c in dret_df.columns:
    dret_df[c] = dret_df[c] - idx_ret_df.set_index('tradeDate')['CHGPct']
    
end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

"""

### 1.2 个股盈余公告数据

"""

start_time = time.time()
print ("该部分进行个股盈余公告数据获取...")

# 业绩快报
if not os.path.exists(os.path.join(file_path, 'ee.pkl')):
    ee_df = DataAPI.FdmtEeGet(ticker=u"",secID=u"",reportType=u"",endDate=u"",beginDate=u"",publishDateEnd=end_date,publishDateBegin=start_date,field=u"ticker,actPubtime,publishDate,endDate,fiscalPeriod,reportType",pandas="1")   
    ee_df['pubtime'] = ee_df['actPubtime'].apply(lambda x: x[:10].replace('-', ''))
    ee_df.to_pickle(os.path.join(file_path, 'ee.pkl'))
else:
    ee_df = pd.read_pickle(os.path.join(file_path, 'ee.pkl'))
ee_df = ee_df.sort_values(['ticker', 'publishDate', 'endDate'])
print ("业绩快报:", ee_df.head().to_html())

# 业绩预告
if not os.path.exists(os.path.join(file_path, 'ef.pkl')):
    ef_df = DataAPI.FdmtEfGet(ticker=u"",secID=u"",reportType=u"",endDate=u"",beginDate=u"",publishDateEnd=end_date,publishDateBegin=start_date,forecastType="",field=u"ticker,actPubtime,publishDate,endDate,fiscalPeriod,reportType",pandas="1")
    ef_df['days'] = ef_df.apply(lambda x: (datetime.datetime.strptime(x['publishDate'][:10], "%Y-%m-%d") - datetime.datetime.strptime(x['endDate'], "%Y-%m-%d")).days, axis=1)
    # 剔除公告覆盖财务日期距离公告发布日期大于365天或者小于-125天的记录
    ef_df = ef_df[(ef_df['days']<125) & ((ef_df['days']>-365))].drop('days', axis=1)
    ef_df['pubtime'] = ef_df['actPubtime'].apply(lambda x: x[:10].replace('-', ''))
    ef_df.to_pickle(os.path.join(file_path, 'ef.pkl'))
else:
    ef_df = pd.read_pickle(os.path.join(file_path, 'ef.pkl'))
ef_df = ef_df.sort_values(['ticker', 'publishDate', 'endDate'])
print ("业绩预告:", ef_df.head().to_html())

# 定期财报
if not os.path.exists(os.path.join(file_path, 'is.pkl')):
    is_df = DataAPI.FdmtISGet(ticker=u"",secID=u"",reportType=u"",endDate=u"",beginDate=u"",publishDateEnd=end_date,publishDateBegin=start_date,endDateRep="",beginDateRep="",beginYear="",endYear="",fiscalPeriod="",field=u"ticker,actPubtime,publishDate,endDate,fiscalPeriod,reportType",pandas="1")
    is_df['pubtime'] = is_df['actPubtime'].apply(lambda x: x[:10].replace('-', ''))
    is_df.to_pickle(os.path.join(file_path, 'is.pkl'))
else:
    is_df = pd.read_pickle(os.path.join(file_path, 'is.pkl'))
is_df = is_df.sort_values(['ticker', 'endDate', 'publishDate', 'actPubtime'])
is_df = is_df.drop_duplicates(['ticker', 'endDate', 'publishDate'], keep='last')  # 同一天发布的保留最新记录
is_df = is_df.drop_duplicates(['ticker', 'endDate'], keep='first')  # 删除对历史修正的记录
print ("定期财报:", is_df.head().to_html())

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


"""

第二部分：超预期事件的测试
该部分耗时 1分钟
该部分内容为：

2.1 说明超预期研报的筛选方式和不同类型和时间的超预期研报的统计
2.2 测试不同类型的超预期事件的超额收益
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
2.1 超预期研报的文本刻画

2.1.1 有效研报样本的筛选

由于优矿平台不能获取通联数据底层的研报数据，因此这部分没有提供研报处理代码。以下说明底层研报的处理方式，以及提供最后的超预期研报结果。
首先，采集分析师对盈余公告（业绩预告、业绩快报和正式财报）进行点评的研报,撰写时间始于20160101，截至于20200531，总共有40993篇研报。 图片注释
从分布来看，研报的公告类型和参考研报中有所不同，主要可能是因为监管趋严后，2016年业绩预告发布数量逐年增多。整体上，定期财报点评类型的研报占比最高，高达54.9%，其次是业绩预告点评类型的研报占比也很高，达到35.5%。
因为研报的入库时间相对于撰写日期有所之后，从统计结果来看，接近90%的研报的入库时间在2天以内（包括2天），99.8%的研报入库时间在5天以内（包括5天）。样本剔除入库时间滞后超过5天的研报后，有效研报数量为40771篇。
   调试 运行
文档
 代码  策略  文档
2.1.2 超预期的文本定义

本文从有效研报的标题和摘要信息中识别出关键信息。本文采用极简单文本处理方式筛选出“超预期”研报。
关键词汇规则如下：
1）屏蔽词汇： 符合, 下, 低, 不, 或, 将, 预计, 望, 可能, 待, 略, 小幅, 复合, 亏损, 可期
2）必须词：预期
3）必选词：高,好,优, 超, 胜
筛选规则如下：
1）将标题中包含屏蔽词汇的研报剔除；
2）首先针对评级进行筛选：评级为强烈推荐, 强推, 优于大市, 跑赢行业的入选超预期样本，评级为中性, 审慎推荐, 谨慎增持, 卖出, 谨慎推荐, 回避, 减持, 观望, 审慎增持, 跑输行业的剔除；
3）在标题和摘要中出现预期的位置的前6个文字中，搜索必选词汇，若出现必选词汇。则入选超预期样本；出现屏蔽词汇，则剔除。
根据上述规则，最后筛选出6417篇“超预期”研报。
数据下载地址：http://qbqo3su8y.bkt.clouddn.com/exceed_expectation_report.csv
可自行下载数据后，上传到优矿的私有数据的jor文件目录下，即可运行后续代码。

"""

# 超预期研报
ee_report_df = pd.read_csv(os.path.join(file_path, 'exceed_expectation_report.csv'), dtype={'ticker': str, 'write_date': str, 'into_date': str, 'anno_pubdate': str})
print ("超预期研报样例： 共%s篇" %len(ee_report_df), ee_report_df.head().to_html())

"""

下面对超预期样本的字段进行说明：
1）report_id: 研报ID，可以根据这个在“萝卜投资”网页上找到该研报；
2）ticker: 个股证券代码； sec_name: 个股证券简称；
3）write_date: 研报撰写日期； into_date: 研报入库日期；into_dalay_days：研报入库滞后天数；
4）report_title: 研报标题；report_rate: 研报评级；report_type: 研报对应公告类型（业绩快报，业绩预告，定期财报）；report_type_id：研报对应公告类型ID，1表示为定期财报，2表示为业绩快报，3表示为业绩预告；
5）anno_pubdate：对应公告的发布时间；into_anno_gap_days：公告发布时间和研报入库时间的滞后天数（研报入库日期-公告发布时间）

"""

start_time = time.time()
print ("该部分进行超预期研报的统计...")

# 统计不同类型报告的数量
report_type_count = ee_report_df.groupby('report_type')['ticker'].count()
report_type_count['整体'] = report_type_count.sum()
ax = report_type_count.plot(kind='bar')
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font, fontsize=15, rotation=0)
ax.set_title(u'超预期分析师样本不同类型报告的数量', fontproperties=font, fontsize=16)
for n in range(len(report_type_count)):
    plt.text(n-0.1, report_type_count[n]+50, report_type_count[n])
plt.show()
    
# 不同类型报告在不同月份的分布情况
ee_report_df['month'] = ee_report_df['write_date'].apply(lambda x: x[4:6])
report_month_count = ee_report_df.groupby(['month', 'report_type'])['ticker'].count().reset_index()
report_month_count = report_month_count.pivot_table(index='month', columns='report_type', values='ticker').fillna(0)
report_month_count = report_month_count / report_month_count.sum()

fig, ax = plt.subplots(figsize=(10, 5))
width = 0.2
ind = np.arange(12)
rect1 = ax.bar(ind+width, report_month_count.loc[:, '定期财报'], width=width, color='dodgerblue')
rect2 = ax.bar(ind+width*2, report_month_count.loc[:, '业绩预告'], width=width, color='darkorange')
rect3 = ax.bar(ind+width*3, report_month_count.loc[:, '业绩快报'], width=width, color='silver')
ax.set_xticks(ind+width*2.5)
ax.set_xticklabels(report_month_count.index)
ax.set_title(u'超预期分析师样本不同类型报告在不同月份的分布情况', fontproperties=font, fontsize=16)
ax.legend([rect1, rect2, rect3], [u'定期财报', u'业绩预告', u'业绩快报'], prop=font, bbox_to_anchor=(0.6, -0.1), ncol=3)
plt.show()

# 不同类型超预期报告分年度统计
ee_report_df['year'] = ee_report_df['write_date'].apply(lambda x: x[:4])
report_year_count = ee_report_df.groupby(['year', 'report_type'])['ticker'].count().reset_index()
report_year_count = report_year_count.pivot_table(index='year', columns='report_type', values='ticker')
print ('不同类型超预期报告分年度统计', report_year_count.to_html())

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time)


'''

超预期研报占所有有效研报的15.7%。 其中，定期财报类占所有超预期研报的比例最高，占比为56.8%，其次是业绩预告，占比为33.6%。
从入库月份统计来看，定期财报类研报基本集中在4、8、10月份；业绩预告类研报基本集中在1、4、7、10月份；业绩快报类研报基本集中在1、2、3、7月份。
从年度来看，依然是业绩预告和定期财报类研报数量较多，业绩快报类研报相对较少。


2.2 超预期事件效应测试
定于超预期事件：假设公司𝑖在𝑇时刻盈余公告后一共获得N份超预期样本，我们记录第𝑛份（其中 0<𝑛≤𝑁）超预期样本的入库时间为tn, 接着我们在tn+1日s收盘价买入 股票 并持有至tn+81个交易日，研报采样期为5天。 图片注释

'''

def analyse_event_return(event_df, dret_df, hold_days):
    """
    测试事件收益
    参数：
        event_df: 事件买入时间点，DataFrame, columns=['ticker', 'buy_date']
        dret_df: 个股收益率数据，DataFrame， index为时间，columns为个股代码
        hold_days: 持有时间，int
    返回：
        每个事件hold_days天收益
    """
    max_date = dret_df.index.max()
    event_df = event_df.query("buy_date<=@max_date")
    
    def _get_ret(buy_date, ticker):
        find_idx = dret_df.index.tolist().index(buy_date)
        ret_s = dret_df.ix[find_idx:(find_idx+hold_days), ticker].reset_index(drop=True)
        return ret_s
    
    event_ret_df = event_df.apply(lambda x: _get_ret(x['buy_date'], x['ticker']), axis=1)
    event_df = pd.concat([event_df, event_ret_df], axis=1).set_index(['ticker', 'buy_date']).dropna()
    return event_df

def report_sample_event_return(sample_df, hold_days=80):
    """
    测试超预期不同财报类型的事件收益
    参数：
        sample_df: 研报数据，DataFrame
        hold_days: 持有时间，int
    返回：
        每个超预期事件hold_days天收益
    """
    sample_df = sample_df.sort_values(['ticker', 'anno_pubdate', 'into_date'])
    sample_df = sample_df.groupby(['ticker', 'anno_pubdate']).head(2)
    report_count  = sample_df.groupby(['ticker', 'anno_pubdate'])['into_date'].count().reset_index()
    sample_n2 = report_count.query("into_date==2")
    sample_df1 = sample_df.drop_duplicates(['ticker', 'anno_pubdate'], keep='first')
    sample_df2 = sample_df.merge(sample_n2[['ticker', 'anno_pubdate']], on=['ticker', 'anno_pubdate'])
    sample_df2 = sample_df2.drop_duplicates(['ticker', 'anno_pubdate'], keep='last')

    sample_df1_ret_df = analyse_event_return(sample_df1[['ticker', 'buy_date']], dret_df, hold_days)
    sample_df2_ret_df = analyse_event_return(sample_df2[['ticker', 'buy_date']], dret_df, hold_days)
    return sample_df1_ret_df, sample_df2_ret_df


start_time = time.time()
print ("该部分进行超预期事件测试...")

ee_report_df = ee_report_df.sort_values(['ticker', 'anno_pubdate', 'report_type_id', 'into_date'])
ee_report_df['buy_date'] = ee_report_df['into_date'].apply(lambda x: [td for td in daily_trade_list if td > x][1])

# 保留5天内的报告
is_sample_df = ee_report_df.query("report_type_id==1").query("into_anno_gap_days<=5")  # 定期财报
ee_sample_df = ee_report_df.query("report_type_id==2").query("into_anno_gap_days<=5")  # 业绩快报
ef_sample_df = ee_report_df.query("report_type_id==3").query("into_anno_gap_days<=5")  # 业绩预告
is_sample_df1_ret_df, is_sample_df2_ret_df = report_sample_event_return(is_sample_df, hold_days=80)
print ("定期财报类超预期事件个数（n=1）：%s, (n=2): %s" %(len(is_sample_df1_ret_df), len(is_sample_df2_ret_df)))
ee_sample_df1_ret_df, ee_sample_df2_ret_df = report_sample_event_return(ee_sample_df, hold_days=80)
print ("业绩快报类超预期事件个数（n=1）：%s, (n=2): %s" %(len(ee_sample_df1_ret_df), len(ee_sample_df2_ret_df)))
ef_sample_df1_ret_df, ef_sample_df2_ret_df = report_sample_event_return(ef_sample_df, hold_days=80)
print ("业绩预告类超预期事件个数（n=1）：%s, (n=2): %s" %(len(ef_sample_df1_ret_df), len(ef_sample_df2_ret_df)))

fig, ax = plt.subplots(figsize=(15,6))
is_sample_df1_ret_df.mean().cumsum().plot(color='green', label=u'定期财报，n=1')
is_sample_df2_ret_df.mean().cumsum().plot(color='green', linestyle='--', label=u'定期财报，n=2')
ee_sample_df1_ret_df.mean().cumsum().plot(color='grey', label=u'业绩快报，n=1')
ee_sample_df2_ret_df.mean().cumsum().plot(color='grey', linestyle='--', label=u'业绩快报，n=2')
ef_sample_df1_ret_df.mean().cumsum().plot(color='darkorange', label=u'业绩预报，n=1')
ef_sample_df2_ret_df.mean().cumsum().plot(color='darkorange', linestyle='--', label=u'业绩预报，n=2')
ax.legend(prop=font, bbox_to_anchor=(0.6, -0.1), ncol=3)
ax.set_title(u'不同 n 取值下不同类型超预期事件的超额收益表现', fontproperties=font, fontsize=16)
plt.show()

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

测试不同类型的超预期事件触发后，相对中证500的超额收益，结果如上图。
业绩预告、年报的超额收益相当稳定、且衰退较慢，长达80个交易日内仍然贡献持续为正的超额收益，在70个交易日收益达到巅峰。而快报的超额收益虽然为正，但稳定性不足；
预告类型超预期事件的超额能力、持续性最强，财报居中，快报最差；
无论是业绩预告还是定期财报 ，多份超预期报告样本验证下的超预期事件（n=2) 超额能力显著强于单份超预期报告样本验证下的超预期事件。

    
第三部分：盈余公告跳空因子
该部分耗时 4分钟
该部分内容为：

3.1 构建因子测试的函数集
3.2 计算EAR因子并测试
3.3 计算JOR因子并测试
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 因子的测试函数

下面因子数据处理，IC分析、分组回测分析的函数。
理论基础，参数设置以及说明参考7月的深度报告《因子合成方法实证分析》


'''

def factor_process(factor_df, factor_list):
    """
    因子处理：去极值、标准化、行业市值中性化
    参数：
        factor_df: DataFrame, 待处理因子值
        factor_list: list, 待处理因子列表
    返回：
        DataFrame, 处理后因子
    """
    # 去极值
    w_factor_df = qutil.mad_winsorize(factor_df, factor_list, sigma_n=5)
    # 标准化
    w_factor_df[factor_list] = w_factor_df.groupby('tradeDate')[factor_list].apply(lambda df: (df-df.mean())/df.std())
    # 行业市值中性化
    wsn_factor_df1 = qutil.netralize_dframe(w_factor_df, factor_list, exclude_style=['BETA', 'RESVOL', 'MOMENTUM', 'SIZENL', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY'])
    # 全中性化
    #wsn_factor_df2 = qutil.neutralize_dframe(w_factor_df, factor_list, exclude_style=[])
    # modified by steven cai
    
    wsn_facotr_df2 = qutil.netralize_dframe(w_factor_df, factor_list, exclude_style=[])
    wsn_factor_df = wsn_factor_df1.merge(wsn_factor_df2, on=['ticker', 'tradeDate'], suffixes=('_n1', '_n2'))
    return w_factor_df, wsn_factor_df

def factor_test_summary(factor_df, factor_list, ngrp):
    """
    综合因子测试方法：回归法、IC分析法、分组测试分析法
    参数：
        factor_df: DataFrame, 因子值
        factor_df: list, 因子列表
    返回：
        因子收益率和t值、IC序列、分组收益率序列
    """
    # IC测试
    ic_res = qutil.calc_ic(factor_df, mret_df, factor_list, return_col_name='nxt1m_ret', ic_type='spearman')
    # 分层回测测试
    perf_list = []
    for fn in factor_list:
        perf, _ = qutil.simple_group_backtest(factor_df, mret_df, factor_name=fn, return_name='nxt1m_ret', commission=0, ngrp=ngrp)
        perf_list.append(perf.pivot_table(values='period_ret', index='tradeDate', columns='group'))
    perf_df = pd.concat(perf_list, axis=1)
    perf_df.columns = pd.MultiIndex.from_tuples([(fn, col) for fn in factor_list for col in range(ngrp)])
    return ic_res, perf_df

def proc_float_scale(df, col_name, format_str):
    """
    格式化输出
    参数：
        df: DataFrame, 需要格式化的数据
        col_name： list, 需要格式化的列名
        format_str： 格式类型
    """
    for col in col_name:
        for index in df[~df[col].isnull()].index:
            df.ix[index, col] = format(df.ix[index, col], format_str)
    return df

def ic_describe(ic_df, factor_list,annual_len):
    """
    统计IC的均值、标准差、IC_IR、大于0的比例
    参数:
        ic_df: DataFrame, IC值， index为日期， columns为因子名， values为各个因子的IC值
        factor_df: list, 因子列表
        annual_len: int, 年化周期数。若是月频结果，则通常为12；若是周频结果，则通常为52
    返回:
        DataFrame, IC统计
    """
    ic_df = ic_df.dropna()
    
    # 记录因子个数和因子名
    n = len(factor_list)
    # IC均值
    ic_mean = ic_df[factor_list].mean()
    # IC标准差
    ic_std = ic_df[factor_list].std()
    # IC均值的T统计量
    ic_t = pd.Series(st.ttest_1samp(ic_df[factor_list], 0)[0], index=factor_list)
    # IC_IR
    ic_ir = ic_mean/ic_std*np.sqrt(annual_len)
    # IC大于0的比例
    ic_p_pct = (ic_df[factor_list] > 0).sum()/len(ic_df)
    
    # IC统计
    ic_table = pd.DataFrame([ic_mean, ic_std, ic_t, ic_ir, ic_p_pct], index=['平均IC', 'IC标准差', 'IC均值T统计量','IC_IR', 'IC大于0的比例']).T
    ic_table = proc_float_scale(ic_table, ['平均IC', 'IC标准差', 'IC大于0的比例'], ".2%")
    ic_table = proc_float_scale(ic_table, ['IC均值T统计量','IC_IR'], ".2f")
    return ic_table

def group_perf_describe(perf_df, factor_list, annual_len):
    """
    统计因子的回测绩效， 包括年化收益率、年化波动率、夏普比率、最大回撤
    参数:
        perf_df: DataFrame, 回测的期间收益率， index为日期， columns为因子名， values为因子回测的期间收益率
        factor_df: list, 因子列表
        annual_len: int, 年化周期数。若是月频结果，则通常为12；若是周频结果，则通常为52
    返回:
        DataFrame, 返回回测绩效
    """
    # 记录因子个数
    n = len(factor_list)
    group_res = (perf_df.mean()*annual_len).reset_index()
    group_res.columns = ['factor_name', 'group', 'value']
    group_res = group_res.pivot_table(values='value', index='factor_name', columns='group')
    ngrp = group_res.columns.max()+1

    sub_res = pd.concat([perf_df[(fn, ngrp-1)] - perf_df[(fn, 0)] for fn in factor_list], axis=1)
    sub_res.columns = factor_list

    # 年化收益率
    ret_mean = sub_res.mean()*annual_len
    # 年化波动率
    ret_std = sub_res.std()*np.sqrt(annual_len)
    # 年化IR
    ir = ret_mean / ret_std
    # 最大回撤
    maxdrawdown = {}
    for i in range(n):
        fname = factor_list[i]
        cum_ret = pd.DataFrame((sub_res[fname]+1).cumprod())
        cum_max = cum_ret.cummax()
        maxdrawdown[fname] = ((cum_max-cum_ret)/cum_max).max().values[0]
    maxdrawdown = pd.Series(maxdrawdown)
    # 月度胜率
    win_ret = (sub_res > 0).sum()/(len(sub_res)-1)

    ls_res = pd.DataFrame([ret_mean, ret_std, ir, maxdrawdown, win_ret], index=['ls_ret', 'ls_std', 'ls_ir', 'ls_md', 'ls_win']).T

    group_table = pd.concat([group_res, ls_res], axis=1)
    group_table.columns = ['第%s组年化收益率'%i for i in range(1, ngrp+1)]+['多空组合年化收益率', '多空组合年化波动率', '多空组合夏普比率', '多空组合最大回撤', '多空组合月度胜率']
    group_table = proc_float_scale(group_table, ['第%s组年化收益率'%i for i in range(1, ngrp+1)]+['多空组合年化收益率', '多空组合年化波动率', '多空组合最大回撤', '多空组合月度胜率'], ".2%")
    group_table = proc_float_scale(group_table, ['多空组合夏普比率'], ".2f")
    return group_table.loc[factor_list, :]

def test_discribe(ic_res, perf_df, factor_list, annual_len=12):
    """
    综合因子分析结果统计
    参数:
        ic_res: DataFrame, IC值， index为日期， columns为因子名， values为各个因子的IC值
        perf_df: DataFrame, 回测的期间收益率， index为日期， columns为因子名， values为因子回测的期间收益率
        factor_df: list, 因子列表
    """
    ic_table = ic_describe(ic_res, factor_list, annual_len=annual_len)
    group_table = group_perf_describe(perf_df, factor_list, annual_len=annual_len)
    print ('IC结果分析', ic_table.to_html())
    print ('分组回测结果分析', group_table.to_html())
    
    fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    ic_res.set_index('tradeDate').cumsum().plot(ax=ax[1])
    ax[1].set_title(u'因子在不同中性化要求下的 IC 累计图', fontproperties=font, fontsize=15)
    for fn in factor_list:
        tmp = perf_df[fn]
        ls_ret = (tmp[tmp.columns.max()] - tmp[0]).add(1).cumprod()
        ls_ret.plot(ax=ax[0])
    ax[0].legend(factor_list, loc=0)
    ax[0].set_title(u'因子在不同中性化要求下的多空对冲净值（分%s组）'%(tmp.columns.max()+1), fontproperties=font, fontsize=15)
    plt.show()
    
    
'''

3.2 EAR因子的计算和测试

盈余惯性（PEAD）现象是指盈余公告后价格漂移，由Ball and Brown(1968)的经典文献最早发现。文献发现：盈余公告发布后，股价长期的累积超额收益朝着盈利意外的方向漂移。
EAR因子就是基于这一现象设计的。其定义为：公司在T日发布盈余公告，则将T-2日收盘～T+1日收盘期间的累积收益记为因子值。
每个交易日，取最近一个盈余公告时EAR作为因子值。

'''

start_time = time.time()
print ("该部分计算EAR因子...")

# 找到每个自然日的下一个交易日
all_day_df = cal_dates_df[['calendarDate', 'isOpen']]
all_day_df['trade_date'] = np.where(all_day_df['isOpen']==1, all_day_df['calendarDate'], np.nan)
all_day_df  = all_day_df.sort_values(['calendarDate'])
all_day_df['trade_date'] = all_day_df['trade_date'].fillna(method='ffill')
trade_df = cal_dates_df.query("isOpen==1")[['calendarDate']]
trade_df['t+1'] = trade_df['calendarDate'].shift(-1)
all_day_df = all_day_df.merge(trade_df, left_on=['trade_date'], right_on=['calendarDate'], suffixes=('', '_a'))

# 计算3个交易日的收益
un_dret_df = dmkt_df[['ticker', 'tradeDate', 'chgPct']]
un_dret_df.columns = ['ticker', 'tradeDate', 'dret']
un_dret_df = un_dret_df.sort_values(['ticker', 'tradeDate'])
un_dret_df['pre3_ret'] = un_dret_df.groupby(['ticker'])['dret'].rolling(3).apply(lambda x: (1+x).prod()-1).values

# 计算EAR因子
ee_df['tag'] = 2
ef_df['tag'] = 3
is_df['tag'] = 1
fis_report_df = pd.concat([ef_df, ee_df, is_df])
fis_report_df = fis_report_df.drop_duplicates(['ticker', 'pubtime'])

# 计算T日盈余公告发布后，T-2收盘～T+1收盘期间的累积收益，称作为盈余惯性
ear_df = fis_report_df.merge(all_day_df[['calendarDate', 't+1']], left_on=['pubtime'], right_on=['calendarDate'])
ear_df = ear_df.drop_duplicates(subset=['ticker', 't+1'])
ear_df = ear_df[['ticker', 'pubtime', 't+1', 'tag']].merge(un_dret_df[['ticker', 'tradeDate', 'pre3_ret']], left_on=['ticker', 't+1'], right_on=['ticker', 'tradeDate'])
ear_df = ear_df.sort_values(['ticker', 'pubtime'])[['ticker', 't+1', 'pre3_ret']]
ear_df.columns = ['ticker', 'tradeDate', 'ear']

# 每个交易日，取最近一个盈余公告时盈余惯性作为因子值
factor_date_df = mret_df[['ticker', 'tradeDate']]
factor_date_df['flag'] = 1
ear_df = ear_df.merge(factor_date_df, how='outer')
ear_df = ear_df.sort_values(['ticker', 'tradeDate'])
ear_df['ear']= ear_df.groupby(['ticker'])['ear'].fillna(method='ffill')
ear_df = ear_df[ear_df['flag']==1]
ear_df = ear_df[['ticker', 'tradeDate', 'ear']]

# 剔除调仓当日的停牌、次新股、st
ear_df = ear_df.merge(forbidden_pool[['ticker', 'prevTradeDate', 'special_flag']], left_on=['ticker', 'tradeDate'], right_on=['ticker', 'prevTradeDate'], how='left')
ear_df = ear_df[ear_df['special_flag'].isnull()]
ear_df = ear_df.drop(['prevTradeDate', 'special_flag'], axis=1)
print ("EAR因子：", ear_df.head().to_html())

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


start_time = time.time()
print ("该部分测试EAR因子...")

# 因子处理
w_ear_df, wsn_ear_df = factor_process(ear_df, factor_list=['ear'])
all_ear_df = w_ear_df.merge(wsn_ear_df, on=['ticker', 'tradeDate'])
# 因子测试
ear_ic_res, ear_perf_df = factor_test_summary(all_ear_df, ['ear', 'ear_n1', 'ear_n2'], ngrp=10)
test_discribe(ear_ic_res, ear_perf_df, ['ear', 'ear_n1', 'ear_n2'], annual_len=12)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

上图结果中，ear为去极值，标准化后的原始因子，ear_n1为市值行业中性化后的因子值，ear_n2为风险模型所有风格和行业中性化后的因子值。
从IC来看，全中性后的EAR因子的IC为1.36%，月度胜率为64.52%；从多空组合来看，中性化有助于提高因子稳定性。全中性后的EAR因子的多空年化收益为4.89%，夏普比为1.00。
整体上看，EAR因子在A股市场有一定的选股效果，但是波动行较大，表现并不出色。
   调试 运行
文档
 代码  策略  文档
3.3 JOR因子的计算和测试

部分文献认为学术文献认为盈余公告日之后的价格跳跃行为比 盈余公告日前后的超额收益更有信息含量。据此，设计了JOR因子。
JOR因子的定义：
JORi=rc2l,t,i−indexrc2l,t,i
其中，t是盈余公告的下一个交易日，rc2l,t,i是股票i在t日最低价相对前一日收盘价的收益率，indexrc2l,t,i是中证500指数在t日最低价相对前一日收盘价的收益率。

'''


start_time = time.time()
print ("该部分计算JOR因子...")

# 计算JOR因子
ee_df['tag'] = 2
ef_df['tag'] = 3
is_df['tag'] = 1
fis_report_df = pd.concat([ef_df, ee_df, is_df])
fis_report_df = fis_report_df.drop_duplicates(['ticker', 'pubtime'])

# 计算T日盈余公告发布后，T+1日最低价相对前收盘价的收益率，称作盈余股价跳跃
raw_jor_df = fis_report_df.merge(all_day_df[['calendarDate', 't+1']], left_on=['pubtime'], right_on=['calendarDate'])
raw_jor_df = raw_jor_df.drop_duplicates(subset=['ticker', 't+1'])
raw_jor_df = raw_jor_df[['ticker', 'pubtime', 't+1', 'tag']].merge(dmkt_df[['ticker', 'tradeDate', 'preClosePrice', 'lowestPrice']], left_on=['ticker', 't+1'], right_on=['ticker', 'tradeDate'])
raw_jor_df = raw_jor_df.merge(idx_ret_df, on=['tradeDate'])
raw_jor_df['jor'] = raw_jor_df['lowestPrice'] / raw_jor_df['preClosePrice'] - raw_jor_df['lowestIndex'] / raw_jor_df['preCloseIndex']

# 每个交易日，取最近一个盈余股价跳跃作为因子值
factor_date_df = mret_df[['ticker', 'tradeDate']]
factor_date_df['flag'] = 1
jor_df = raw_jor_df.merge(factor_date_df, how='outer')
jor_df = jor_df.sort_values(['ticker', 'tradeDate'])
jor_df['jor']= jor_df.groupby(['ticker'])['jor'].fillna(method='ffill')
jor_df = jor_df[jor_df['flag']==1]
jor_df = jor_df[['ticker', 'tradeDate', 'jor']]

# 剔除调仓当日的停牌、次新股、st
jor_df = jor_df.merge(forbidden_pool[['ticker', 'prevTradeDate', 'special_flag']], left_on=['ticker', 'tradeDate'], right_on=['ticker', 'prevTradeDate'], how='left')
jor_df = jor_df[jor_df['special_flag'].isnull()]
jor_df = jor_df.drop(['prevTradeDate', 'special_flag'], axis=1)
print ("JOR因子：", jor_df.head().to_html())

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


start_time = time.time()
print ("该部分测试JOR因子...")

# 因子处理
w_jor_df, wsn_jor_df = factor_process(jor_df, factor_list=['jor'])
all_jor_df = w_jor_df.merge(wsn_jor_df, on=['ticker', 'tradeDate'])
# 因子测试
jor_ic_res, jor_perf_df = factor_test_summary(all_jor_df, ['jor', 'jor_n1', 'jor_n2'], ngrp=10)
test_discribe(jor_ic_res, jor_perf_df, ['jor', 'jor_n1', 'jor_n2'], annual_len=12)

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


'''

上图结果中，jor为去极值，标准化后的原始因子，jor_n1为市值行业中性化后的因子值，jor_n2为风险模型所有风格和行业中性化后的因子值。
从IC来看，全中性后的JOR因子的IC为2.21%，月度胜率为79.84%；从多空组合来看，全中性后的EAR因子的多空年化收益为8.87%，夏普比为2.16。
整体上看，JOR因子相对EAR因子有明显提升，不仅提高了选股能力，波动率也有明显的降低。
   调试 运行
文档
 代码  策略  文档
进一步，我们测试JOR因子对超预期事件的分组效果。
具体地，根据超预期事件对应的公告发布时JOR，将超预期事件等分为3组。

'''

start_time = time.time()
print ("该部分测试JOR因子对超预期事件的分组效果...")

# 定期财报超预期样本
is_sample_df1_ret_df['excess_ret'] = is_sample_df1_ret_df[range(80)].apply(lambda x: x.sum(), axis=1)
is_sample_exret_df = is_sample_df1_ret_df[['excess_ret']].reset_index()
tmp_is_sample_df = is_sample_df[['ticker', 'anno_pubdate', 'buy_date']]
tmp_is_sample_df = tmp_is_sample_df.sort_values(['ticker', 'buy_date', 'anno_pubdate'])
tmp_is_sample_df = tmp_is_sample_df.drop_duplicates(['ticker', 'buy_date'], keep='last')
is_sample_exret_df = is_sample_exret_df.merge(tmp_is_sample_df, on=['ticker', 'buy_date'], how='left')
is_sample_exret_df = is_sample_exret_df.merge(raw_jor_df[['ticker', 'pubtime', 'jor']], left_on=['ticker', 'anno_pubdate'], right_on=['ticker', 'pubtime'], how='left')
is_sample_exret_df = is_sample_exret_df.dropna()
# 分三组：序号越小，因子值越小
is_sample_exret_df['group'] = is_sample_exret_df['jor'].rank(method='first')
is_sample_exret_df['group'] = ((is_sample_exret_df['group']-1) / len(is_sample_exret_df) * 3).astype(int)
is_sample_groupret_df = is_sample_exret_df.groupby('group')['excess_ret'].mean()

# 业绩快报超预期样本
ee_sample_df1_ret_df['excess_ret'] = ee_sample_df1_ret_df[range(80)].apply(lambda x: x.sum(), axis=1)
ee_sample_exret_df = ee_sample_df1_ret_df[['excess_ret']].reset_index()
tmp_ee_sample_df = ee_sample_df[['ticker', 'anno_pubdate', 'buy_date']]
tmp_ee_sample_df = tmp_ee_sample_df.sort_values(['ticker', 'buy_date', 'anno_pubdate'])
tmp_ee_sample_df = tmp_ee_sample_df.drop_duplicates(['ticker', 'buy_date'], keep='last')
ee_sample_exret_df = ee_sample_exret_df.merge(tmp_ee_sample_df, on=['ticker', 'buy_date'], how='left')
ee_sample_exret_df = ee_sample_exret_df.merge(raw_jor_df[['ticker', 'pubtime', 'jor']], left_on=['ticker', 'anno_pubdate'], right_on=['ticker', 'pubtime'], how='left')
ee_sample_exret_df = ee_sample_exret_df.dropna()
# 分三组：序号越小，因子值越小
ee_sample_exret_df['group'] = ee_sample_exret_df['jor'].rank(method='first')
ee_sample_exret_df['group'] = ((ee_sample_exret_df['group']-1) / len(ee_sample_exret_df) * 3).astype(int)
ee_sample_groupret_df = ee_sample_exret_df.groupby('group')['excess_ret'].mean()

# 业绩预告超预期样本
ef_sample_df1_ret_df['excess_ret'] = ef_sample_df1_ret_df[range(80)].apply(lambda x: x.sum(), axis=1)
ef_sample_exret_df = ef_sample_df1_ret_df[['excess_ret']].reset_index()
tmp_ef_sample_df = ef_sample_df[['ticker', 'anno_pubdate', 'buy_date']]
tmp_ef_sample_df = tmp_ef_sample_df.sort_values(['ticker', 'buy_date', 'anno_pubdate'])
tmp_ef_sample_df = tmp_ef_sample_df.drop_duplicates(['ticker', 'buy_date'], keep='last')
ef_sample_exret_df = ef_sample_exret_df.merge(tmp_ef_sample_df, on=['ticker', 'buy_date'], how='left')
ef_sample_exret_df = ef_sample_exret_df.merge(raw_jor_df[['ticker', 'pubtime', 'jor']], left_on=['ticker', 'anno_pubdate'], right_on=['ticker', 'pubtime'], how='left')
ef_sample_exret_df = ee_sample_exret_df.dropna()
# 分三组：序号越小，因子值越小
ef_sample_exret_df['group'] = ef_sample_exret_df['jor'].rank(method='first')
ef_sample_exret_df['group'] = ((ef_sample_exret_df['group']-1) / len(ef_sample_exret_df) * 3).astype(int)
ef_sample_groupret_df = ef_sample_exret_df.groupby('group')['excess_ret'].mean()

# 画图
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
fig.suptitle(u'不同断层因子分组下不同公告类型超预期事件的超额收益水平', fontproperties=font, fontsize=15)
is_sample_groupret_df.plot(kind='bar', ax=ax[0])
ax[0].set_xticklabels(['low', 'middle', 'high'], rotation=0)
ax[0].set_title(u'定期财报', fontproperties=font, fontsize=10)

ee_sample_groupret_df.plot(kind='bar', ax=ax[1])
ax[1].set_xticklabels(['low', 'middle', 'high'], rotation=0)
ax[1].set_title(u'业绩快报', fontproperties=font, fontsize=10)

ef_sample_groupret_df.plot(kind='bar', ax=ax[2])
ax[2].set_xticklabels(['low', 'middle', 'high'], rotation=0)
ax[2].set_title(u'业绩预告', fontproperties=font, fontsize=10)
# plt.tight_layout()

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

整体上，三种类型的超预期事件，JOR因子均具有显著的区分能力。高JOR因子的超预期事件，能获得更高的超额收益。
   调试 运行
文档
 代码  策略  文档
第四部分：净利润断层组合构建
该部分耗时 2分钟
该部分内容为：

4.1 构建超预期样本空间，测试JOR因子在其中的选股效果
4.2 构建净利润断层金股组合，并回测
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
4.1 超预期样本空间以及JOR因子在其中的选股效果

综合上面的分析，超预期事件的触发是随机的，如果出发一次调一次仓，对交易操作要求较高。为了结合JOR因子的选股，利用超预期事件构造超预期样本空间。
超预期样本空间：每个月末，回溯过去180天的研报，若某股票在回溯期内出发了超预期时间，则该股票入选超预期样本空间。
'''

start_time = time.time()
print ("该部分构造超预期样本空间...")

# 计算每个月底的超预期样本
ee_report_df = ee_report_df.query("into_anno_gap_days<=5")
month_end_dates = cal_dates_df.query("isMonthEnd==1").query("calendarDate>='20160601'").query("calendarDate<='20200601'")['calendarDate'].tolist()
ee_ticker_dict = {}
for td in month_end_dates:
    begin_td = (pd.to_datetime(td)-datetime.timedelta(days=180)).strftime('%Y%m%d')
    tmp_ee_sample = ee_report_df.query("into_date<=@td").query("into_date>=@begin_td")['ticker'].unique().tolist()
    ee_ticker_dict[td] = tmp_ee_sample
ee_ticker = pd.DataFrame.from_dict(ee_ticker_dict, orient='index').unstack().reset_index()[['level_1', 0]].dropna()
ee_ticker.columns = ['tradeDate', 'ticker']
ee_ticker = ee_ticker.sort_values(['tradeDate', 'ticker'])

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

# 统计每个月度的超预期样本个数
ee_ticker_count = ee_ticker.groupby('tradeDate')['ticker'].count()

print ('超预期样本空间统计：')
print (ee_ticker_count.describe()[['count', 'mean', 'min', '50%', 'max']])

fig, ax = plt.subplots(figsize=(15, 4))
ee_ticker_count.plot(kind='bar', ax=ax)
ax.set_title(u'超预期样本空间样本数的时序分布', fontproperties=font, fontsize=15);
plt.show()

# 行业分布情况
indu_df = DataAPI.EquIndustryGet(secID=u"",ticker=u"",industryVersionCD=u"010303",industry=u"",industryID=u"",industryID1=u"",industryID2=u"",industryID3=u"",intoDate=u"",equTypeID=u"",field=u"ticker,intoDate,outDate,industryName1",pandas="1")
indu_df['outDate'] = indu_df['outDate'].fillna('20201231')
ee_ticker_indu = ee_ticker.merge(indu_df, on=['ticker'])
ee_ticker_indu = ee_ticker_indu.query("tradeDate>=intoDate").query("tradeDate<outDate")
# 统计每个行业分布
ee_ticker_indu_count = ee_ticker_indu.groupby(['tradeDate', 'industryName1'])['ticker'].count().reset_index()
ee_ticker_indu_count = ee_ticker_indu_count.pivot_table(index='industryName1', columns='tradeDate', values='ticker').fillna(0)
ee_ticker_indu_dist = ee_ticker_indu_count / ee_ticker_indu_count.sum()
ee_ticker_indu_dist = ee_ticker_indu_dist.sort_values('20200529', ascending=False)
ee_ticker_indu_dist.index = [i.decode('utf8') for i in ee_ticker_indu_dist.index]

fig, ax = plt.subplots(figsize=(15, 8))
fig = ax.stackplot(pd.to_datetime(ee_ticker_indu_dist.columns), ee_ticker_indu_dist, labels=ee_ticker_indu_dist.index, colors=sns.xkcd_rgb.values()[:len(ee_ticker_indu_dist.index)*3:3])
ax.set_ylim((0,1))
ax.legend(prop=font, bbox_to_anchor=(1.2, 1), ncol=1)
ax.set_title(u'超预期样本空间的行业分布', fontproperties=font, fontsize=15);

'''

整体上，超预期样本空间每月的数量在189-475只之间，平均每个月有349只。
通过对超预期样本空间的行业分布分析，超预期样本空间在电子、计算机、化工、医药生物这4个行业的分布占比较大。
   调试 运行
文档
 代码  策略  文档
进一步，测试超预期样本空间是否具有超额收益。每个月末，利用超预期样本空间，流通市值加权构造组合，与中证500进行比较。

'''


start_time = time.time()
print ("该部分测试超预期样本空间的超额收益...")

# 获取流通市值
mv_df = []
for td in month_end_dates:
    td_mv_df = DataAPI.MktEqudGet(secID=u"",ticker=u"",tradeDate=td,beginDate=u"",endDate=u"",isOpen="",field=u"ticker,tradeDate,negMarketValue",pandas="1")
    mv_df.append(td_mv_df)
mv_df = pd.concat(mv_df)
mv_df['tradeDate'] = mv_df['tradeDate'].apply(lambda x: x.replace('-', ''))

# 超预期样本空间相对500的表现
ee_ticker_ret_df = ee_ticker.merge(mret_df, on=['ticker', 'tradeDate']).merge(mv_df, on=['ticker', 'tradeDate'])
ee_ticker_ret_df['weight'] = ee_ticker_ret_df.groupby('tradeDate')['negMarketValue'].apply(lambda x: x/sum(x))

# 计算组合收益
ee_ticker_perf_df = ee_ticker_ret_df.groupby('tradeDate').apply(lambda df: (df['nxt1m_ret']*df['weight']).sum()).reset_index()
ee_ticker_perf_df.columns = ['tradeDate', 'ee_mret']
ee_ticker_perf_df = ee_ticker_perf_df.sort_values('tradeDate')
ee_ticker_perf_df['ee_mret'] = ee_ticker_perf_df['ee_mret'].shift(1).fillna(0)

# 对比中证500
m_perf_df = ee_ticker_perf_df.merge(idx_mret_df, on=['tradeDate'])
m_perf_df.ix[0, 'mret'] = 0
m_perf_df['ee_exret'] = m_perf_df['ee_mret'] - m_perf_df['mret']

# 年化收益率
ret_mean = m_perf_df.mean()*12.0
# 年化波动率
ret_std = m_perf_df.std()*np.sqrt(12.0)
# 年化IR
ir = ret_mean / ret_std

m_perf_stat = pd.DataFrame(pd.concat([ret_mean[['ee_mret', 'mret', 'ee_exret']], ir[['ee_mret', 'ee_exret']]])).T
m_perf_stat.columns = ['绝对收益', '中证500', '相对收益', '绝对信息比', '相对信息比']
m_perf_stat = proc_float_scale(m_perf_stat, ['绝对收益', '中证500', '相对收益'], '.2%')
m_perf_stat = proc_float_scale(m_perf_stat, ['绝对信息比', '相对信息比'], '.2f')
print (m_perf_stat.to_html())

fig, ax = plt.subplots(figsize=(15, 5))
(m_perf_df.set_index('tradeDate')+1).cumprod().plot(ax=ax)
ax.legend([u'超预期样本净值', u'中证500', u'相对强弱'], prop=font, loc=2)
ax.set_title(u'超预期样本空间相对中证 00指数的表现', fontproperties=font, fontsize=15);
plt.show()

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


'''

在201607-202005期间，超预期样本空间组合能够持续跑赢中证500。2019年以来，效果有所减弱，但基本和中证500持平。
   调试 运行
文档
 代码  策略  文档
验证JOR因子在超预期样本空间的选股效果。

'''


# JOR因子在超预期样本空间的分组表现
jor_ee_df = jor_df.merge(ee_ticker, on=['ticker', 'tradeDate'])
jor_ee_perf, _ = qutil.simple_group_backtest(jor_ee_df, mret_df, factor_name='jor', return_name='nxt1m_ret', commission=0, ngrp=5)
jor_ee_perf = jor_ee_perf.pivot_table(index='tradeDate', columns='group', values='period_ret')
jor_ee_perf['top-bottom'] = jor_ee_perf[4] - jor_ee_perf[0]

fig, ax = plt.subplots(figsize=(15, 5))
(jor_ee_perf[range(5)]+1).cumprod().plot(ax=ax)
ax.set_title(u'超预期样本空间在JOR不同分组下的业绩表现', fontproperties=font, fontsize=15)
(jor_ee_perf[['top-bottom']]+1).cumprod().plot(ax=ax)
ax.legend(range(5)+['top-bottom(right)'], loc=2);

'''

从分组分析结果来看，JOR因子在超预期样本空间仍然具有显著的选股效果，分组单调性很多，多空组合具有稳定的收益。
   调试 运行
文档
 代码  策略  文档
4.2 净利润断层金股组合

结合JOR因子和超预期样本空间，我们尝试构造净利润断层金股组合。
调仓时点：基于正式财报、 业绩预告信息的分布月份，我们在1、4、7、8、10这5个月月末进行调仓。
选股方式：每个月末，筛选出过去60天内发生业绩预告和定期财报的超预期事件的样本，按照JOR因子排序，选出因子值最大的50个股票，等权配置。剔除ST、停牌、涨跌停、次新股等股票。
回测时间：20160701-20200531。
交易成本：双边千三。

'''
start_time = time.time()
print ("该部分构建净利润断层金股组合...")

part_ee_report_df = ee_report_df[ee_report_df['report_type_id'].isin([1,3])]
adjust_month_date = [td for td in month_end_dates if td[4:6] in ['01', '04', '07', '08', '10']]
# 选出调仓月的超预期样本空间
gold_portfolio = {}
for td in adjust_month_date:
    begin_td = (pd.to_datetime(td)-datetime.timedelta(days=60)).strftime('%Y%m%d')
    tmp_ee_sample = part_ee_report_df.query("into_date<=@td").query("into_date>=@begin_td")['ticker'].unique().tolist()
    gold_portfolio[td] = tmp_ee_sample
gold_portfolio = pd.DataFrame.from_dict(gold_portfolio, orient='index').unstack().reset_index()[['level_1', 0]].dropna()
gold_portfolio.columns = ['tradeDate', 'ticker']
gold_portfolio = gold_portfolio.sort_values(['tradeDate', 'ticker'])

# 用jor因子选出排名前50的个股
gold_portfolio = gold_portfolio.merge(all_jor_df, on=['ticker', 'tradeDate'])
gold_portfolio = gold_portfolio.sort_values(['tradeDate', 'jor_n1'])
gold_portfolio = gold_portfolio.groupby('tradeDate').tail(50)[['tradeDate', 'ticker']]
gold_portfolio['secID'] = gold_portfolio['ticker'].apply(lambda x: x+'.XSHG' if x[0]== '6' else x+'.XSHE')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

# 行业分布情况
gold_portfolio_indu = gold_portfolio.merge(indu_df, on=['ticker'])
gold_portfolio_indu = gold_portfolio_indu.query("tradeDate>=intoDate").query("tradeDate<outDate")
# 统计每个行业分布
gold_portfolio_indu_count = gold_portfolio_indu.groupby(['tradeDate', 'industryName1'])['ticker'].count().reset_index()
gold_portfolio_indu_count = gold_portfolio_indu_count.pivot_table(index='industryName1', columns='tradeDate', values='ticker').fillna(0)
gold_portfolio_indu_dist = gold_portfolio_indu_count / gold_portfolio_indu_count.sum()
gold_portfolio_indu_dist = gold_portfolio_indu_dist.sort_values('20200430', ascending=False)
gold_portfolio_indu_dist.index = [i.decode('utf8') for i in gold_portfolio_indu_dist.index]

fig, ax = plt.subplots(figsize=(15, 8))
fig = ax.stackplot(pd.to_datetime(gold_portfolio_indu_dist.columns), gold_portfolio_indu_dist, labels=gold_portfolio_indu_dist.index, colors=sns.xkcd_rgb.values()[:len(gold_portfolio_indu_dist.index)*3:3])
ax.set_ylim((0,1))
ax.legend(prop=font, bbox_to_anchor=(1.2, 1), ncol=1)
ax.set_title(u'净利润断层金股组合的行业分布', fontproperties=font, fontsize=15);

fig, ax = plt.subplots(figsize=(15, 4))
gold_portfolio_indu_count.mean(axis=1).sort_values(ascending=False).plot(kind='bar', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font, fontsize=12)
ax.set_title(u'净利润断层金股组合的行业平均入选个股', fontproperties=font, fontsize=15)

'''

和上述分析一样，净利润断层金股组合在电子、计算机、化工、医药生物这4个行业的分布占比较大。

'''

start = '2016-08-01'                       # 回测起始时间
end = '2020-06-01'                         # 回测结束时间
universe = DynamicUniverse('A')        # 证券池，支持股票、基金、期货、指数四种资产
benchmark = 'ZZ500'                        # 策略参考标准
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = Monthly(1)                              # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟
  
# 配置账户信息，支持多资产多账户
accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000, commission = Commission(buycost=0.003, sellcost=0.003, unit='perValue'))
}
  
def initialize(context):
    pass
  
# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context):    
    previous_date = context.previous_date.strftime('%Y%m%d')
    if previous_date in adjust_month_date:
        target_position = gold_portfolio.query("tradeDate==@previous_date")['secID'].tolist()

        # 获取当前账户信息
        account = context.get_account('fantasy_account')   
        current_position = account.get_positions(exclude_halt=True)       

        # 卖出当前持有，但目标持仓没有的部分
        for stock in set(current_position).difference(target_position):
            account.order_to(stock, 0)

        # 根据目标持仓权重，逐一委托下单
        for stock in target_position:
            account.order_pct_to(stock, 1.0/len(target_position))
            
            
# 超额收益
perf_df = pd.concat([perf['returns'], perf['benchmark_returns']], axis=1)
perf_df = perf_df[perf_df.index>='2016-07-29']
perf_df.columns = ['ret', 'bret']
perf_df.iloc[0,] = 0
perf_df['eret'] = perf_df['ret'] - perf_df['bret']
ax = (perf_df['eret']+1).cumprod().plot()
ax.set_title(u'净利润断层金股策略相对中证500的超额业绩表现', fontproperties=font, fontsize=15);
plt.show()

perf_df['year'] = [x.year for x in perf_df.index]
perf_df['month'] = [x.month for x in perf_df.index]
ym_perf = perf_df.groupby(['year', 'month'])['eret'].apply(lambda x: (x+1).prod()-1).reset_index()
print ("月度胜率：", format(((ym_perf['eret']>0).sum()-1)* 1.0 / (len(ym_perf)-1), '.2%'))
ym_perf = ym_perf.pivot_table(index='year', columns='month', values='eret')
ym_perf.loc[2016, 7] = np.nan
ym_perf.loc[u'均值', :] = ym_perf.mean()
ym_perf = proc_float_scale(ym_perf, ym_perf.columns, '.2%')
print ('净利润断层金股组合相对中证500收益月度汇总:', ym_perf.fillna('').to_html())

'''

通过对净利润断层金股组合的回测，在201607-202005期间，组合稳定持续跑赢中证500，且年华超额收益达到21.0%，月度胜率达到76.09％。
分月度表现来看，组合在1、4、6月业绩表现相对更优，在3、 9、11月业绩表现相对较弱。
   调试 运行
文档
 代码  策略  文档
总结
利用研报中超预期的样本触发的事件具有超额收益。其中，业绩预告类型超预期事件的超额能力、持续性最强，财报居中，快报最差；
盈余跳空因子JOR，相对于传统的盈余漂移因子EAR，在A股市场的选股能力更好其更稳定。全中性后的JOR因子的IC为2.21%，月度胜率为79.84%，多空年化收益为8.89%，夏普比为2.16。JOR因子对超预期事件也有显著的区分能力。
利用超预期事件构造的超预期样本空间具有超额收益。叠加JOR因子，构造净利润断层金股组合，在201607-202005期间，组合稳定持续跑赢中证500，且年华超额收益达到21.0%，月度胜率达到76.09％。

'''



