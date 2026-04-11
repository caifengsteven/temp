# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:39:40 2020

@author: Asus
"""

'''

导读
A. 研究目的：本文利用优矿提供的行情数据和财务数据，参考光大证券《成长企业新视角：税收科目全解析》（原作者：古翔等），对三大报表中的税收科目进行分析，并构建一系列从税收角度出发的成长因子和质量因子，并且进一步分析因子的有效性和叠加性。

B. 研究结论：

三大报表中均有与税收相关的会计科目，其中，税务口径核算的税费可参考资产负债表中的应交税费科目；会计口径核算的税费可参考利润表中的所得税费用和税金及附加；税务口径与会计口径不一致导致的暂时性差异形成资产负债表中的递延所得税资产（负债）。现金流量表中，支付的各项税费为当期实际缴纳税费的现金流。
利用三大报表的所有税收会计科目，构造了42个因子，包含成长因子和质量因子，并与传统成长因子进行比较。单因子测试结果显示，应交税费占比、税金及附加环比、支付的各项税费环比这个三个税收因子选股效果最好，且和净利润环比相关性较低。且整体来说，时序标准化处理可以提升基本面因子的选股效果。
三个税收因子合成后，20100101-20200731期间，月度调仓，IC均值为2.01%，ICIR为2.95，五分组多空年化收益率为7.31%，夏普比率达到2.78，且五分组区分度显著，单调性好。合成税收因子叠加净利润环比因子，多空年化收益为12.64%，夏普比率为3.95，对比净利润环比因子多空年化收益为11.52%，夏普比率为3.68，叠加后的因子优于单个净利润成长因子，说明税收因子有增益信息。
C. 文章结构：本文共分为3个部分，具体如下

一、数据准备
二、税收科目的分析
三、税收因子计算和分析
D. 时间说明

一、第一部分运行需要2分钟
二、第二部分运行需要1分钟
三、第三部分运行需要35分钟
总耗时38分钟左右 (为了方便修改程序，文中将很多中间数据存储下来以缩短运行时间)
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
https://uqer.datayes.com/community/share/eLNeQy0p3r0lRu9I5WoZ5YOw2ng0/private；密码：6278
请在运行之前，克隆上面的代码，并存成lib（右上角->另存为lib,不要修改名字）

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)




'''
import pandas as pd
import numpy as np
import os
import datetime
from pandas.tseries.offsets import QuarterEnd
import time
import lib.quant_util as qutil
import scipy.stats as st
from CAL.PyCAL import *
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# 建立缓存数据的文件夹
file_path = './tax'
if not os.path.exists(file_path):
    os.makedirs(file_path)
    

'''

第一部分：数据准备
该部分耗时 2分钟
该部分内容为：

获取基础行情数据，包括：个股月度收益率数据、后续需要剔除的股票池（上市不满60个交易日的次新股、st股、停牌个股）。时间范围为20100101-20200731.
获取个股的三大财报中税收科目相关数据，以供后文税收科目分析的因子计算。特别说明，利润表和现金流量表的数据，均获取TTM数据。
获取2019年各个投资域成分股和行业分类，以供后文税收科目分析。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


'''
start_time = time.time()
print"该部分进行基础参数设置和数据准备..."

sdate = '20100101'
edate = '20200731'

# 全A投资域
a_universe_list = DataAPI.EquGet(equTypeCD=u"A",listStatusCD=u"L,S,DE",field=u"secID",pandas="1")['secID'].tolist()
a_universe_list.remove('DY600018.XSHG')
a_ticker_list = [t[:6] for t in a_universe_list]

# 交易日历
cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate='', endDate='20201231').sort('calendarDate')
cal_dates_df['calendarDate'] = cal_dates_df['calendarDate'].apply(lambda x: x.replace('-', ''))
cal_dates_df['prevTradeDate'] = cal_dates_df['prevTradeDate'].apply(lambda x: x.replace('-', ''))
daily_trade_list = sorted(cal_dates_df.query("isOpen==1")['calendarDate'].tolist())
year_end_list = cal_dates_df.query('isYearEnd==1')['calendarDate'].tolist()
year_end_list = [td for td in year_end_list if (td>'20070101') & (td<'20200101')]
month_dates_df = cal_dates_df.query("isMonthEnd==1")[['calendarDate']]

# 股票池筛选：上市不满60个交易日的次新股、st股、停牌个股
if not os.path.exists(os.path.join(file_path, 'forbidden.pkl')):
    forbidden_pool = qutil.stock_special_tag(sdate, edate, pre_new_length=60)
    forbidden_pool = forbidden_pool.merge(cal_dates_df, left_on=['tradeDate'], right_on=['calendarDate'])
    forbidden_pool = forbidden_pool[['ticker', 'tradeDate', 'prevTradeDate', 'special_flag']]
    forbidden_pool.to_pickle(os.path.join(file_path, 'forbidden.pkl'))
else:
    forbidden_pool = pd.read_pickle(os.path.join(file_path, 'forbidden.pkl'))
print "禁止股票池:", forbidden_pool.head().to_html()

# 获取个股月度收益率
if not os.path.exists(os.path.join(file_path, 'mret.pkl')):
    mret_df = DataAPI.MktEqumAdjGet(beginDate=sdate, endDate=edate, secID=a_universe_list, field=u"ticker,endDate,chgPct", pandas="1")
    mret_df.rename(columns={'endDate':'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
    mret_df['tradeDate'] = mret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
    mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    mret_df['nxt1m_ret'] = mret_df.groupby('ticker')['curr_ret'].shift(-1)
    mret_df.to_pickle(os.path.join(file_path, 'mret.pkl'))
else:
    mret_df = pd.read_pickle(os.path.join(file_path, 'mret.pkl'))
print "个股月度收益率:", mret_df.head().to_html()

if not os.path.exists(os.path.join(file_path, 'bs.pkl')):
    # 资产负债表
    bs_df = DataAPI.FdmtBSGet(ticker=u"",secID=u"",reportType=u"",endDate=u"20200630",beginDate=u"20061231",publishDateEnd=u"",publishDateBegin=u"",endDateRep="",beginDateRep="",beginYear="",endYear="",fiscalPeriod="",field=u"ticker,actPubtime,publishDate,endDate,fiscalPeriod,deferTaxAssets,deferTaxLiab,taxesPayable,TEquityAttrP",pandas="1")
    bs_df = bs_df[bs_df['ticker'].isin(a_ticker_list)]
    bs_df['publishDate'] = bs_df['publishDate'].apply(lambda x: x.replace('-', ''))
    bs_df.to_pickle(os.path.join(file_path, 'bs.pkl'))
else:
    bs_df = pd.read_pickle(os.path.join(file_path, 'bs.pkl'))
print "资产负债表:", bs_df.head().to_html()
    
if not os.path.exists(os.path.join(file_path, 'is.pkl')):
    # 利润表TTM
    is_df = DataAPI.FdmtISTTMPITGet(endDate=u"20200630",beginDate=u"20061231",field=u"ticker,actPubtime,publishDate,endDate,fiscalPeriod,bizTaxSurchg,incomeTax,NIncomeAttrP",pandas="1")
    is_df = is_df[is_df['ticker'].isin(a_ticker_list)]
    is_df['publishDate'] = is_df['publishDate'].apply(lambda x: x.replace('-', ''))
    is_df.to_pickle(os.path.join(file_path, 'is.pkl'))
else:
    is_df = pd.read_pickle(os.path.join(file_path, 'is.pkl'))
print "利润表:", is_df.head().to_html()
    
if not os.path.exists(os.path.join(file_path, 'cf.pkl')):
    # 现金流量表TTM
    cf_df = DataAPI.FdmtCFTTMPITGet(endDate=u"20200630",beginDate=u"20061231",field=u"ticker,actPubtime,publishDate,endDate,fiscalPeriod,CPaidForTaxes,refundOfTax,NCFOperateA",pandas="1")
    cf_df = cf_df[cf_df['ticker'].isin(a_ticker_list)]
    cf_df['publishDate'] = cf_df['publishDate'].apply(lambda x: x.replace('-', ''))
    cf_df.to_pickle(os.path.join(file_path, 'cf.pkl'))
else:
    cf_df = pd.read_pickle(os.path.join(file_path, 'cf.pkl'))
print "现金流量表:", cf_df.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)

start_time = time.time()
print"该部分获取2019年各投资域成分股和行业分类..."

#### 为分析税收数据所需
# 指数2019成份股
# HS300 
hs300_df = DataAPI.mIdxCloseWeightGet(secID=u"",ticker=u"000300",beginDate=u"20191231",endDate=u"20191231",field=u"consTickerSymbol,effDate",pandas="1")
# ZZ500
zz500_df = DataAPI.mIdxCloseWeightGet(secID=u"",ticker=u"000905",beginDate=u"20191231",endDate=u"20191231",field=u"consTickerSymbol,effDate",pandas="1")
# ZZ1000
zz1000_df = DataAPI.mIdxCloseWeightGet(secID=u"",ticker=u"000852",beginDate=u"20191231",endDate=u"20191231",field=u"consTickerSymbol,effDate",pandas="1")
# 全A
qa_df = []
for td in year_end_list:
    td_qa_df = pd.DataFrame([ticker[:6] for ticker in DynamicUniverse('A').preview(td)], columns=['consTickerSymbol'])
    td_qa_df['effDate'] = td[:4]+'-12-31'
    qa_df.append(td_qa_df)
qa_df = pd.concat(qa_df)

# 行业分类
indu_df = DataAPI.EquIndustryGet(secID=u"",ticker=u"",industryVersionCD=u"010303",industry=u"",industryID=u"",industryID1=u"",industryID2=u"",industryID3=u"",intoDate=u"20191231",equTypeID=u"",field=u"ticker,industryName1",pandas="1")

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)

'''


第二部分：税收科目的分析
该部分耗时 1分钟
该部分内容为：

2.1 资产负债表中的税收科目分析
2.2 利润表中的税收科目分析
2.3 现金流量表中的税收科目分析
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


2.1 资产负债表中的税收科目

资产负债表中，税收相关科目有：递延所得税资产（deferTaxAssets）、递延所得税负债（deferTaxLiab）、应交税费（taxesPayable），反映未来须缴纳的税费或可抵减税的金额。
递延所得税资产/负债：主要来源于税务口径与会计口径的不一致所导致的暂时性差异，未来可抵减/应纳税的金额，即：当期实际缴纳的税费高于会计核算的税费时，可计入递延所得税资产，未来可抵减所得税金额；反之亦然。
应交税费：类似于应付账款，是按权责发生制原则确认计提但尚未缴纳的部分，可能包括应交增值税、应交企业所得税、应交城市维护建设税等，一定程度可近似理解为税务口径下计算的企业税负。
下面分析三个税收科目占归属股东所有者权益的占比，以及A股中的覆盖度。



'''

start_time = time.time()
print"该部分进行递延所得税分析..."

annual_bs_df = bs_df.query("fiscalPeriod=='12'").sort_values(['ticker', 'publishDate', 'actPubtime', 'endDate'])
annual_bs_df = annual_bs_df.drop_duplicates(['ticker', 'publishDate', 'endDate'], keep='last')
annual_bs_df = annual_bs_df.drop_duplicates(['ticker', 'endDate'], keep='first')

annual_bs_df['dta2na'] = annual_bs_df['deferTaxAssets'] / annual_bs_df['TEquityAttrP']
annual_bs_df['dtl2na'] = annual_bs_df['deferTaxLiab'] / annual_bs_df['TEquityAttrP']
annual_bs_df['tp2na'] = annual_bs_df['taxesPayable'] / annual_bs_df['TEquityAttrP']

## 计算递延所得占比统计
dt_med = annual_bs_df.groupby('endDate')[['dta2na', 'dtl2na', 'tp2na']].median()
qa_count = qa_df.merge(annual_bs_df[['ticker', 'endDate', 'dta2na', 'dtl2na', 'tp2na']], left_on=['consTickerSymbol', 'effDate'], right_on=['ticker', 'endDate'], how='left')
qa_cov = qa_count.groupby('effDate')[['dta2na', 'dtl2na', 'tp2na', 'consTickerSymbol']].count()
qa_cov['dta2na_cov'] = qa_cov['dta2na'] /qa_cov['consTickerSymbol']
qa_cov['dtl2na_cov'] = qa_cov['dtl2na'] /qa_cov['consTickerSymbol']
qa_cov['tp2na_cov'] = qa_cov['tp2na'] /qa_cov['consTickerSymbol']

# 计算不同宽基内，递延所得税统计
qa_2019 = qa_df.query("effDate=='2019-12-31'")
qa_2019['is_zz1000'] = np.where(qa_2019['consTickerSymbol'].isin(zz1000_df['consTickerSymbol']), 1, np.nan)
qa_2019['is_zz500'] = np.where(qa_2019['consTickerSymbol'].isin(zz500_df['consTickerSymbol']), 1, np.nan)
qa_2019['is_hs300'] = np.where(qa_2019['consTickerSymbol'].isin(hs300_df['consTickerSymbol']), 1, np.nan)
qa_2019['is_other'] = np.where(qa_2019[['is_zz1000', 'is_zz500', 'is_hs300']].sum(axis=1)>=1, np.nan, 1)

univ_count = qa_2019.merge(annual_bs_df.query("endDate=='2019-12-31'")[['ticker', 'dta2na', 'dtl2na']], left_on='consTickerSymbol', right_on='ticker', how='left')
dta_median = {}
dta_cov = {}
dtl_median = {}
dtl_cov = {}
univ_list = ['other', 'zz1000', 'zz500', 'hs300']
for univ in univ_list:
    dta_median[univ] = (univ_count['dta2na']*univ_count['is_'+univ]).median()
    dta_cov[univ] = (univ_count['dta2na']*univ_count['is_'+univ]).count()*1.0 / univ_count['is_'+univ].count()
    dtl_median[univ] = (univ_count['dtl2na']*univ_count['is_'+univ]).median()
    dtl_cov[univ] = (univ_count['dtl2na']*univ_count['is_'+univ]).count()*1.0 / univ_count['is_'+univ].count()

# 画图
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(121)
ind = np.arange(len(dt_med.index))
width = 0.3
ax.bar(ind+width, dt_med['dta2na'], width, label=u'递延所得税资产占净资产比例', color='cornflowerblue')
ax.bar(ind+width*2, dt_med['dtl2na'], width, label=u'递延所得税负债占净资产比例', color='lightsteelblue')
ax.set_xticks(ind+width)
ax.set_xticklabels(dt_med.index, rotation=70);
ax.grid(False)
ax1 = ax.twinx()
ax1.plot(ind+width*2, qa_cov['dta2na_cov'], label=u'递延所得税资产覆盖度(右轴)', color='chocolate', linestyle='--')
ax1.plot(ind+width*2, qa_cov['dtl2na_cov'], label=u'递延所得税负债覆盖度(右轴)', color='gold', linestyle='--')
ax1.grid(False)
ax1.set_ylim((0,1))
ax.legend(prop=font, bbox_to_anchor=(0.9,-0.25), ncol=2)
ax1.legend(prop=font, bbox_to_anchor=(0.9,-0.35), ncol=2)
ax.set_title(u'不同时段内，递延所得税资产占比统计（中位数）', fontproperties=font, fontsize=15)

ax2 = fig.add_subplot(122)
ind = np.arange(len(univ_list))
width = 0.3
ax2.bar(ind+width, pd.Series(dta_median)[univ_list], width, label=u'递延所得税资产占净资产比例', color='cornflowerblue')
ax2.bar(ind+width*2, pd.Series(dtl_median)[univ_list], width, label=u'递延所得税资负债净资产比例', color='lightsteelblue')
ax2.set_xticks(ind+width*2)
ax2.set_xticklabels([u'非宽基指数内', u'中证1000', u'中证500', u'沪深300'], fontproperties=font);
ax2.grid(False)
ax3 = ax2.twinx()
ax3.plot(ind+width*2, pd.Series(dta_cov)[univ_list], label=u'递延所得税资产覆盖度(右轴)', color='chocolate', linestyle='--')
ax3.plot(ind+width*2, pd.Series(dtl_cov)[univ_list], label=u'递延所得税负债覆盖度(右轴)', color='gold', linestyle='--')
ax3.grid(False)
ax3.set_ylim((0,1))
ax2.legend(prop=font, bbox_to_anchor=(1,-0.1), ncol=2)
ax3.legend(prop=font, bbox_to_anchor=(1,-0.2), ncol=2)
ax2.set_title(u'不同宽基指数，递延所得税资产占比统计（中位数）', fontproperties=font, fontsize=15);

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)



start_time = time.time()
print"该部分进行应交税费分析..."

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ind = np.arange(len(dt_med.index))
width = 0.3
ax.bar(ind+width, dt_med['tp2na'], width, label=u'应交税费占净资产比例', color='cornflowerblue')
ax.set_xticks(ind+width)
ax.set_xticklabels(dt_med.index, rotation=70);
ax.grid(False)
ax1 = ax.twinx()
ax1.plot(ind+width*2, qa_cov['tp2na_cov'], label=u'应交税费覆盖度(右轴)', color='chocolate', linestyle='--')
ax1.grid(False)
# ax1.set_ylim((0,1))
ax.legend(prop=font, bbox_to_anchor=(0.4,-0.25))
ax1.legend(prop=font, bbox_to_anchor=(0.8,-0.25))
ax.set_title(u'不同时段内，应交税费占比统计（中位数）', fontproperties=font, fontsize=15);

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''


应交税费占归母净资产的比例在1%-2.5%内，整体占比也较低。且近年来，占比逐年下降，反映企业税负有所减轻。
   调试 运行
文档
 代码  策略  文档
2.2 利润表中的税收科目

利润表中，税收相关科目有：税金及附加（bizTaxSurchg）、所得税费用（incomeTax），反映当期核算的税费。
税金及附加：在营业利润内计量，包括除企业所得税外的大部分税种，如：城市维护建设税，教育费附加，印花税，房产税，土地使用税等。
所得税费用：即为会计核算的所得税费用，与企业当期的利润总额直接相关。企业所得税率一般是25%，符合条件的小型微利企业，所得税的税率一般为20%。国家重点扶持的高新技术企业，所得税的税率一般为15%。
下面分析两个税收科目占归属母公司净利润的占比，A股中的覆盖度，以及行业分析。


'''

start_time = time.time()
print "该部分进行利润表税收科目分析..."

is_df['end_month'] = is_df['endDate'].apply(lambda x: x[5:7])
annual_is_df = is_df.query("end_month=='12'").sort_values(['ticker', 'publishDate', 'endDate'])
annual_is_df = annual_is_df.drop_duplicates(['ticker', 'endDate'], keep='first')

annual_is_df['bts2np'] = annual_is_df['bizTaxSurchg'] / annual_is_df['NIncomeAttrP']
annual_is_df['it2np'] = annual_is_df['incomeTax'] / annual_is_df['NIncomeAttrP']

## 计算占比统计
ist_med = annual_is_df.groupby('endDate')[['bts2np', 'it2np']].median()
qa_count = qa_df.merge(annual_is_df[['ticker', 'endDate', 'bts2np', 'it2np']], left_on=['consTickerSymbol', 'effDate'], right_on=['ticker', 'endDate'], how='left')
qa_cov = qa_count.groupby('effDate')[['bts2np', 'it2np', 'consTickerSymbol']].count()
qa_cov['bts2np_cov'] = qa_cov['bts2np'] /qa_cov['consTickerSymbol']
qa_cov['it2np_cov'] = qa_cov['it2np'] /qa_cov['consTickerSymbol']

# 画图
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ind = np.arange(len(ist_med.index))
width = 0.3
ax.bar(ind+width, ist_med['bts2np'], width, label=u'税金及附加占利润比例', color='cornflowerblue')
ax.bar(ind+width*2, ist_med['it2np'], width, label=u'所得税占利润比例', color='lightsteelblue')
ax.set_xticks(ind+width)
ax.set_xticklabels(ist_med.index, rotation=70);
ax.grid(False)
ax1 = ax.twinx()
ax1.plot(ind+width*2, qa_cov['bts2np_cov'], label=u'税金及附加覆盖度(右轴)', color='chocolate', linestyle='--')
ax1.plot(ind+width*2, qa_cov['it2np_cov'], label=u'所得税覆盖度(右轴)', color='gold', linestyle='--')
ax1.grid(False)
ax.legend(prop=font, bbox_to_anchor=(0.7,-0.25), ncol=2)
ax1.legend(prop=font, bbox_to_anchor=(0.7,-0.35), ncol=2)
ax.set_title(u'不同时段内，所得税费用、税金及附加占比统计（中位数）', fontproperties=font, fontsize=15)

# 行业统计
indu_stat = indu_df.merge(annual_is_df.query("endDate=='2019-12-31'")[['ticker', 'bts2np', 'it2np']], on='ticker')
indu_med = indu_stat.groupby('industryName1')[['bts2np', 'it2np']].median().sort_values('bts2np', ascending=False)

# 画图
fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(111)
ind = np.arange(len(indu_med.index))
width = 0.3
ax.bar(ind+width, indu_med['bts2np'], width, label=u'税金及附加占利润比例', color='cornflowerblue')
ax.bar(ind+width*2, indu_med['it2np'], width, label=u'所得税占利润比例', color='lightsteelblue')
ax.set_xticks(ind+width)
ax.set_xticklabels([i.decode('utf8') for i in indu_med.index], rotation=70,fontproperties=font);
ax.grid(False)
ax.set_title(u'不同行业内，所得税费用、税金及附加占比统计（中位数，申万一级分类）', fontproperties=font, fontsize=15)
ax.legend(prop=font)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''

所得税费用、税金及附加数据的覆盖度均较高，超过95%。
从占归母净利润的比例来看，所得税费用占比显著高于税金及附加，说明各项税种中，企业所得税对于企业净利润影响最大。
所得税费用占比呈逐年下降的趋势；税金及附加占比相对稳定，大约占6%-8%。
从行业角度来看。不同行业税收占比差异较大。TMT所在的高新技术企业的行业，所得税费用占比和税金及附加占比较低；房地产、采掘等资本开支较大的传统行业所得税费用占比和税金及附加占比较高；金融、休闲服务等服务性行业所得税费用占比较高，税金及附加占比较低。
   调试 运行
文档
 代码  策略  文档
2.3 现金流量表中的税收科目

现金流量表中，税收相关科目有：支付的各项税费（CPaidForTaxes）、收到的税收返还（refundOfTax），反映当期与税收相关的现金流情况。
支付的各项税费：归属于经营活动产生的现金流，是企业当期通过现金缴付的税费。
收到的税收返还：指的是政府按照国家有关规定采取先征后返、即征即退等办法向企业返还的税款，属于以税收优惠形式给予的一种政府补助。
下面分析两个税收科目占经营性净现金流的占比，以及A股中的覆盖度。

'''

start_time = time.time()
print "该部分进行现金流量表税收科目分析..."

cf_df['end_month'] = cf_df['endDate'].apply(lambda x: x[5:7])
annual_cf_df = cf_df.query("end_month=='12'").sort_values(['ticker', 'publishDate', 'endDate'])
annual_cf_df = annual_cf_df.drop_duplicates(['ticker', 'endDate'], keep='first')

annual_cf_df['cpt2cfo'] = annual_cf_df['CPaidForTaxes'] / annual_cf_df['NCFOperateA']
annual_cf_df['rot2cfo'] = annual_cf_df['refundOfTax'] / annual_cf_df['NCFOperateA']

## 计算支付和收到税费占比统计
cft_med = annual_cf_df.groupby('endDate')[['cpt2cfo', 'rot2cfo']].median()
qa_count = qa_df.merge(annual_cf_df[['ticker', 'endDate', 'cpt2cfo', 'rot2cfo']], left_on=['consTickerSymbol', 'effDate'], right_on=['ticker', 'endDate'], how='left')
qa_cov = qa_count.groupby('effDate')[['cpt2cfo', 'rot2cfo', 'consTickerSymbol']].count()
qa_cov['cpt2cfo_cov'] = qa_cov['cpt2cfo'] /qa_cov['consTickerSymbol']
qa_cov['rot2cfo_cov'] = qa_cov['rot2cfo'] /qa_cov['consTickerSymbol']

# 画图
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ind = np.arange(len(cft_med.index))
width = 0.3
ax.bar(ind+width, cft_med['cpt2cfo'], width, label=u'支付的各项税费占经营性净现金流比例', color='cornflowerblue')
ax.bar(ind+width*2, cft_med['rot2cfo'], width, label=u'收到的税收返还占经营性净现金流比例', color='lightsteelblue')
ax.set_xticks(ind+width)
ax.set_xticklabels(cft_med.index, rotation=70);
ax.grid(False)
ax1 = ax.twinx()
ax1.plot(ind+width*2, qa_cov['cpt2cfo_cov'], label=u'支付的各项税费覆盖度(右轴)', color='chocolate', linestyle='--')
ax1.plot(ind+width*2, qa_cov['rot2cfo_cov'], label=u'收到的税收覆盖度(右轴)', color='gold', linestyle='--')
ax1.grid(False)
ax1.set_ylim((0,1))
ax.legend(prop=font, bbox_to_anchor=(0.7,-0.25), ncol=2)
ax1.legend(prop=font, bbox_to_anchor=(0.7,-0.35), ncol=2)
ax.set_title(u'不同时段内，支付的各项税费、收到的税收返还占经营性净现金流占比统计（中位数）', fontproperties=font, fontsize=15)

# 行业统计
indu_stat = indu_df.merge(annual_cf_df.query("endDate=='2019-12-31'")[['ticker', 'rot2cfo']], on='ticker', how='left')
indu_med = indu_stat.groupby('industryName1')[['rot2cfo']].median().sort_values('rot2cfo', ascending=False)
indu_cov = indu_stat.groupby('industryName1').apply(lambda x: x['rot2cfo'].count()*1.0/x['ticker'].count())[indu_med.index]

# 画图
fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(111)
ind = np.arange(len(indu_med.index))
width = 0.3
ax.bar(ind+width, indu_med['rot2cfo'], width, label=u'收到的税费返还占经营性现金流比例', color='cornflowerblue')
ax.set_xticks(ind+width)
ax.set_xticklabels([i.decode('utf8') for i in indu_med.index], rotation=70,fontproperties=font);
ax.grid(False)
ax.legend(prop=font, bbox_to_anchor=(0.5,-0.25))
ax1 = ax.twinx()
ax1.plot(ind+width*1.5, indu_cov, label=u'收到的税收返还覆盖度(右轴)', color='chocolate', linestyle='--')
ax1.legend(prop=font, bbox_to_anchor=(0.7,-0.25))
ax1.grid(False)
ax.set_title(u'不同行业内，收到的税费返还占经营性现金流比例统计（中位数，申万一级分类）', fontproperties=font, fontsize=15)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''

支付的各项税费占经营性现金流比重，在40%上下浮动，对企业现金流具有较大的影响；收到的税收返还占比很小，对企业现金流影响有限。
收到的税收返回一定程度反应政策的优惠程度。从行业层面来看。电子、家用电器、机械设备、汽车等行业的税收政策较多，但最多也只能达到14%左右。
   调试 运行
文档
 代码  策略  文档
第三部分：税收因子计算和分析
该部分耗时 35分钟
该部分内容为：

3.1 计算因子和分析因子的函数
3.2 计算税收因子
3.3 税收因子的单因子测试
3.4 税收因子的叠加性分析
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 计算因子和分析因子的函数

该部分提供了2类函数，第一类函数是计算税收因子的函数；第二类是提供进行因子测试的函数。第二类函数说明参考7月的深度报告《因子合成方法实证分析》。
第一类函数提供了基本面因子计算的框架：
1）提供了三个算子函数：计算比率、计算季度同比、计算时间序列标准化 图片注释
2）理论上，每个会计科目，均可以计算比率，本身的环比，比率的环比、本身的时间序列标准化、比率的时间序列标准化、环比的时间序列标准化。可以结合实际含义，进行因子计算。
3）将算子函数得到的数据，通过“获取最新财务数据”、“PIT转连续”两步计算，得到最终的因子数据。


'''

## 计算基本面因子的函数

# PIT转连续
def fin_data_pit2cont(fin_data_frame, sdate, edate):

    month_dates = month_dates_df.rename(columns={"calendarDate":"publishDate"})
    tmp_frame = fin_data_frame.groupby(['ticker']).apply(lambda x: x.merge(month_dates,
                                                                         on=['publishDate'], how='outer'))
    del tmp_frame['ticker']
    tmp_frame = tmp_frame.reset_index(level=0)

    tmp_frame = tmp_frame.sort_values(by=['ticker', 'publishDate'], ascending=True)
    tmp_frame = tmp_frame.groupby(['ticker']).apply(lambda x: x.fillna(method='pad'))
    tmp_frame = tmp_frame[tmp_frame['publishDate'].isin(month_dates['publishDate'])]
    tmp_frame.dropna(inplace=True)
    tmp_frame = tmp_frame[(tmp_frame['publishDate'] >= sdate) & (tmp_frame['publishDate'] <= edate)]

    return tmp_frame

# 获取最新财务数据
def get_fin_data_latest(fin_data_frame, col_name=['value']):

    fin_df = fin_data_frame.copy()
    if type(col_name) != list:
        col_name = [col_name]

    def get_latest_perticker(df, col_name):
        tmp_df = df.copy()
        tmp_df.dropna(subset=col_name, how='all', inplace=True)
        tmp_df.sort_values(['publishDate', 'endDate'],inplace=True)
        tmp_df.drop_duplicates(subset=['publishDate'], keep='last', inplace=True)
        tmp_df['max_end_date'] = tmp_df['endDate'].rolling(window=6, min_periods=1).max()
        tmp_df['max_end_date'] = tmp_df['max_end_date'].astype(np.str)
        tmp_df = tmp_df[tmp_df['endDate'] == tmp_df['max_end_date']]
        return tmp_df[['ticker', 'publishDate', 'endDate'] + col_name]

    fin_df = fin_df.groupby(['ticker']).apply(get_latest_perticker, col_name)
    fin_df.reset_index(inplace=True, drop=True)
    return fin_df

# 计算比率的算子
def _operator_ratio(df, num_col, den_col, ratio_name):
    df = df.copy()
    df[ratio_name] = df[num_col] / df[den_col]
    return df[['ticker', 'publishDate', 'endDate', ratio_name]].dropna()

# 计算环比
def _operator_qoq(df, col, qoq_name):
    df = df.copy()
    pre_end_dict = {ed: (pd.to_datetime(ed) - QuarterEnd(1)).strftime("%Y-%m-%d") for ed in df.endDate.unique()}
    pre_df = df.copy()
    df['pre_end'] = df['endDate'].apply(lambda x: pre_end_dict[x])
    df = df.merge(pre_df, left_on=['ticker', 'pre_end'], right_on=['ticker', 'endDate'], suffixes=('', '_pre'))
    df[qoq_name] = (df[col] - df[col+'_pre']) / abs(df[col+'_pre']) * 100.0
    df['publishDate'] = df[['publishDate', 'publishDate_pre']].max(axis=1).astype(int).astype(str)
    return df[['ticker', 'publishDate', 'endDate', qoq_name]].dropna()

# 计算时间序列标准化
def _operator_z(df, col, z_name, n=8):
    df = df.copy()
    df = df.dropna(subset=[col]).sort_values(['ticker', 'publishDate', 'endDate'])
    df = df.drop_duplicates(['ticker', 'publishDate', 'endDate'], keep='last')
    pre_df = df.copy()
    for i in range(1, n+1):
        pre_end_dict = {ed: (pd.to_datetime(ed) - QuarterEnd(i)).strftime("%Y-%m-%d") for ed in df.endDate.unique()}
        df['pre_end'] = df['endDate'].apply(lambda x: pre_end_dict[x])
        df = df.merge(pre_df, left_on=['ticker', 'pre_end'], right_on=['ticker', 'endDate'], suffixes=('', '_'+str(i)), how='left')
        df = df[(df['publishDate']>=df['publishDate_'+str(i)]) | df['publishDate_'+str(i)].isnull()]
        df = df.sort_values(['ticker', 'publishDate', 'endDate', 'publishDate_'+str(i)])
        df = df.drop_duplicates(['ticker', 'publishDate', 'endDate'], keep='last')
    history_col = [col+'_'+str(i) for i in range(1, n+1)]
    df[z_name] = (df[col] - df[history_col].mean(axis=1)) / df[history_col].std(axis=1)
    df[z_name] = np.where(df[history_col].count(axis=1)<n/2.0, np.nan, df[z_name])
    return df[['ticker', 'publishDate', 'endDate', z_name]].dropna()

# 基本面因子连续化
def fund_factor_process(factor_df_list, sdate, edate):
    fund_factor = []
    for df in factor_df_list:
        col_name = np.setdiff1d(df.columns, ['ticker', 'publishDate', 'endDate']).tolist()
        tmp = get_fin_data_latest(df, col_name=col_name)
        tmp = tmp[['ticker', 'publishDate']+col_name].set_index(['ticker', 'publishDate'])
        fund_factor.append(tmp)
    fund_factor = pd.concat(fund_factor, axis=1).reset_index()
    fund_factor = fin_data_pit2cont(fund_factor, sdate, edate)
    return fund_factor.rename(columns={'publishDate': 'tradeDate'})



## 因子分析函数

def filter_nopub(data):
    """
    过滤数据中非上市个股记录
    data: 待过滤数据, columns=['ticker', 'date', ...]
    """
    # 获取股票上市退市时间
    equ_info = DataAPI.EquGet(secID=u"",ticker=u"",equTypeCD=u"A",listStatusCD=u"L,S,DE",field=u"ticker,listDate,delistDate",pandas="1")
    equ_info['listDate'] = equ_info['listDate'].apply(lambda x: str(x).replace('-', ''))
    equ_info['delistDate'] = equ_info['delistDate'].apply(lambda x: str(x).replace('-', ''))
    
    # 过滤
    data_m = data.merge(equ_info, on=['ticker'], how='left')
    data_m_filter = data_m[(data_m['tradeDate']>=data_m['listDate']) & (data_m['tradeDate']<data_m['delistDate'])]
    data_m_filter = data_m_filter[data.columns]
    return data_m_filter

def factor_process(factor_df, factor_list, is_win=True, is_stand=True, is_neut=True):
    """
    因子处理：去极值、标准化、行业市值中性化
    参数：
        factor_df: DataFrame, 待处理因子值
        factor_list: list, 待处理因子列表
    返回：
        DataFrame, 处理后因子
    """
    # 剔除非上市个股
    factor_df = filter_nopub(factor_df)
    # # 剔除调仓当日的停牌、次新股、st
    factor_df = factor_df.merge(forbidden_pool[['ticker', 'prevTradeDate', 'special_flag']], left_on=['ticker', 'tradeDate'], right_on=['ticker', 'prevTradeDate'], how='left')
    factor_df = factor_df[factor_df['special_flag'].isnull()]
    # 剔除inf
    factor_df[factor_list] = np.where(np.isinf(factor_df[factor_list]), np.nan, factor_df[factor_list])
    
    # 去极值
    if is_win:
        w_factor_df = qutil.mad_winsorize(factor_df, factor_list, sigma_n=5)
    else:
        w_factor_df = factor_df.copy()
    
    # 标准化
    if is_stand:
        w_factor_df[factor_list] = w_factor_df.groupby('tradeDate')[factor_list].apply(lambda df: (df-df.mean())/df.std()) 

    if is_neut:
        # 全中性化
        wsn_factor_df = qutil.neutralize_dframe(w_factor_df, factor_list, exclude_style=[])
    else:
        wsn_factor_df = w_factor_df.copy()
        
    return wsn_factor_df[['ticker', 'tradeDate']+factor_list]

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

def test_discribe(ic_res, perf_df, factor_list, annual_len=12, show_fig=False):
    """
    综合因子分析结果统计
    参数:
        ic_res: DataFrame, IC值， index为日期， columns为因子名， values为各个因子的IC值
        perf_df: DataFrame, 回测的期间收益率， index为日期， columns为因子名， values为因子回测的期间收益率
        factor_df: list, 因子列表
    """
    ic_table = ic_describe(ic_res, factor_list, annual_len=annual_len)
    group_table = group_perf_describe(perf_df, factor_list, annual_len=annual_len)
    print 'IC结果分析', ic_table.to_html()
    print '分组回测结果分析', group_table.to_html()
    if show_fig:
        for fn in factor_list:
            ngrp = len(perf_df[fn].columns)
            fig, ax = plt.subplots(figsize=(18, 4), nrows=1, ncols=2)
            ax[0].plot(pd.to_datetime(perf_df[fn].index), (perf_df[fn]+1).cumprod())
            ax1 = ax[0].twinx()
            ax1.plot(pd.to_datetime(perf_df[fn].index), (perf_df[fn][ngrp-1]-perf_df[fn][0]+1).cumprod(), label='top-bottom', color='k')
            ax[0].legend(perf_df[fn].columns.tolist(), loc=2)
            ax1.legend(loc=1)
            ax[0].grid(False)
            ax1.grid(False)
            ax[0].set_title(u'%s%s分组净值'%(fn, ngrp), fontproperties=font)
        
            ind = np.arange(ngrp)
            width = 0.3
            ax[1].bar(ind+width, (perf_df[fn]+1).prod(), width)
            ax[1].set_xticks(ind+width*1.5)
            ax[1].set_xticklabels(perf_df[fn].columns)
            ax[1].set_title(u'%s%s累计收益'%(fn, ngrp), fontproperties=font)
            plt.show()
    
def calc_factor_corr(factor_frame, factor_list, show_corr_plot=True, corr_method='spearman'):
    """
    计算因子相关性
    """
    dates_list = factor_frame['tradeDate'].unique()
    date_corr_df = factor_frame.groupby('tradeDate')[factor_list].corr(method=corr_method)
    corr_df = sum([date_corr_df.loc[date].fillna(0) for date in dates_list]) / sum(
        [~date_corr_df.loc[date].isnull() for date in dates_list])

    if show_corr_plot:
        fig, ax = plt.subplots(figsize=(len(factor_list) * 0.9, len(factor_list) * 0.6))
        sns.heatmap(corr_df, linewidths=0.05, ax=ax, vmax=1, vmin=-1, cmap='RdYlGn_r', annot=True)
        ax.set_title(u'因子相关性', fontproperties=font, fontsize=16)
        plt.show()
    return corr_df

'''
3.2 计算税收因子（25分钟）

利用三大财务报表中的各个税收科目，计算税收因子。所用基础数据如下。
简称	缩写	含义
递延所得税	DT	递延所得税资产（空值填充为0）-递延所得税负债（空值填充为0）
应交税费	PT	可理解为税务口径下核算的税费
所得税	OT	企业所得税
各项税费之和	AT	企业所得税+税金及附加
税金及附加	OT	经营过程中产生的税费（车船税、印花税等）
支付的各项税费	PTCF	企业缴纳各项税费的现金流
支付税费的净现金流	NTCF	支付的各项税费 - 收到的税收返还（空值填充为0）
归属母公司的所有者权益	NA	-
归属母公司净利润	NP	-
经营性净现金流	OCF	-
- 上述税收字段，计算比率，本身的环比，比率的环比、本身的时间序列标准化、比率的时间序列标准化、环比的时间序列标准化，共6个衍生因子。同时，利用归属母公司的所有者权益、归属母公司净利润、经营性净现金流，计算环比和环比的时间序列标准化，共6个因子，作为传统成长因子的比较。


'''

start_time = time.time()
print"该部分计算递延所得税相关因子..."

# 递延所得税相关因子
bs_df['DT'] = bs_df['deferTaxAssets'].fillna(0) - bs_df['deferTaxLiab'].fillna(0)
factor1 = _operator_ratio(bs_df, num_col='DT', den_col='TEquityAttrP', ratio_name='dt2na')
factor2 = _operator_qoq(bs_df[['ticker', 'publishDate', 'endDate', 'DT']], col='DT', qoq_name='DT_qoq')
factor3 = _operator_qoq(factor1, col='dt2na', qoq_name='dt2na_qoq')
factor4 = _operator_z(factor1, col='dt2na', z_name='dt2na_z', n=8)
factor5 = _operator_z(factor2, col='DT_qoq', z_name='DT_qoq_z', n=8)
factor6 = _operator_z(factor3, col='dt2na_qoq', z_name='dt2na_qoq_z', n=8)

factor_df_list = [factor1, factor2, factor3, factor4, factor5, factor6]
dt_factor = fund_factor_process(factor_df_list, sdate, edate)
dt_factor.to_pickle(os.path.join(file_path, 'dt.pkl'))

print "递延所得税相关因子：", dt_factor.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


start_time = time.time()
print"该部分计算应交税费相关因子..."

# 应交税费相关因子
factor1 = _operator_ratio(bs_df, num_col='taxesPayable', den_col='TEquityAttrP', ratio_name='pt2na')
factor2 = _operator_qoq(bs_df[['ticker', 'publishDate', 'endDate', 'taxesPayable']], col='taxesPayable', qoq_name='PT_qoq')
factor3 = _operator_qoq(factor1, col='pt2na', qoq_name='pt2na_qoq')
factor4 = _operator_z(factor1, col='pt2na', z_name='pt2na_z', n=8)
factor5 = _operator_z(factor2, col='PT_qoq', z_name='PT_qoq_z', n=8)
factor6 = _operator_z(factor3, col='pt2na_qoq', z_name='pt2na_qoq_z', n=8)

factor_df_list = [factor1, factor2, factor3, factor4, factor5, factor6]
pt_factor = fund_factor_process(factor_df_list, sdate, edate)
pt_factor.to_pickle(os.path.join(file_path, 'pt.pkl'))

print "应交税费相关因子：", pt_factor.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


start_time = time.time()
print"该部分计算所得税相关因子..."

# 所得税相关因子
factor1 = _operator_ratio(is_df, num_col='incomeTax', den_col='NIncomeAttrP', ratio_name='it2np')
factor2 = _operator_qoq(is_df[['ticker', 'publishDate', 'endDate', 'incomeTax']], col='incomeTax', qoq_name='IT_qoq')
factor3 = _operator_qoq(factor1, col='it2np', qoq_name='it2np_qoq')
factor4 = _operator_z(factor1, col='it2np', z_name='it2np_z', n=8)
factor5 = _operator_z(factor2, col='IT_qoq', z_name='IT_qoq_z', n=8)
factor6 = _operator_z(factor3, col='it2np_qoq', z_name='it2np_qoq_z', n=8)

factor_df_list = [factor1, factor2, factor3, factor4, factor5, factor6]
it_factor = fund_factor_process(factor_df_list, sdate, edate)
it_factor.to_pickle(os.path.join(file_path, 'it.pkl'))

print "所得税相关因子：", it_factor.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


start_time = time.time()
print"该部分计算各项税费之和相关因子..."

# 各项税费之和相关因子
is_df['AT'] = is_df['bizTaxSurchg']+is_df['incomeTax']
factor1 = _operator_ratio(is_df, num_col='AT', den_col='NIncomeAttrP', ratio_name='at2np')
factor2 = _operator_qoq(is_df[['ticker', 'publishDate', 'endDate', 'AT']], col='AT', qoq_name='AT_qoq')
factor3 = _operator_qoq(factor1, col='at2np', qoq_name='at2np_qoq')
factor4 = _operator_z(factor1, col='at2np', z_name='at2np_z', n=8)
factor5 = _operator_z(factor2, col='AT_qoq', z_name='AT_qoq_z', n=8)
factor6 = _operator_z(factor3, col='at2np_qoq', z_name='at2np_qoq_z', n=8)

factor_df_list = [factor1, factor2, factor3, factor4, factor5, factor6]
at_factor = fund_factor_process(factor_df_list, sdate, edate)
at_factor.to_pickle(os.path.join(file_path, 'at.pkl'))

print "各项税费之和相关因子：", at_factor.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


start_time = time.time()
print"该部分计算税金及附加相关因子..."

# 税金及附加相关因子
factor1 = _operator_ratio(is_df, num_col='bizTaxSurchg', den_col='NIncomeAttrP', ratio_name='ot2np')
factor2 = _operator_qoq(is_df[['ticker', 'publishDate', 'endDate', 'bizTaxSurchg']], col='bizTaxSurchg', qoq_name='OT_qoq')
factor3 = _operator_qoq(factor1, col='ot2np', qoq_name='ot2np_qoq')
factor4 = _operator_z(factor1, col='ot2np', z_name='ot2np_z', n=8)
factor5 = _operator_z(factor2, col='OT_qoq', z_name='OT_qoq_z', n=8)
factor6 = _operator_z(factor3, col='ot2np_qoq', z_name='ot2np_qoq_z', n=8)

factor_df_list = [factor1, factor2, factor3, factor4, factor5, factor6]
ot_factor = fund_factor_process(factor_df_list, sdate, edate)
ot_factor.to_pickle(os.path.join(file_path, 'ot.pkl'))

print "税金及附加相关因子：", ot_factor.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)

start_time = time.time()
print"该部分计算支付的各项税费相关因子..."

# 支付的各项税费相关因子
factor1 = _operator_ratio(cf_df, num_col='CPaidForTaxes', den_col='NCFOperateA', ratio_name='ptcf2ocf')
factor2 = _operator_qoq(cf_df[['ticker', 'publishDate', 'endDate', 'CPaidForTaxes']], col='CPaidForTaxes', qoq_name='PTCF_qoq')
factor3 = _operator_qoq(factor1, col='ptcf2ocf', qoq_name='ptcf2ocf_qoq')
factor4 = _operator_z(factor1, col='ptcf2ocf', z_name='ptcf2ocf_z', n=8)
factor5 = _operator_z(factor2, col='PTCF_qoq', z_name='PTCF_qoq_z', n=8)
factor6 = _operator_z(factor3, col='ptcf2ocf_qoq', z_name='ptcf2ocf_qoq_z', n=8)

factor_df_list = [factor1, factor2, factor3, factor4, factor5, factor6]
ptcf_factor = fund_factor_process(factor_df_list, sdate, edate)
ptcf_factor.to_pickle(os.path.join(file_path, 'ptcf.pkl'))

print "支付的各项税费相关因子：", ptcf_factor.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


start_time = time.time()
print"该部分计算支付税费净现金流相关因子..."

# 支付税费净现金流相关因子
cf_df['NTCF'] = cf_df['CPaidForTaxes'] -cf_df['refundOfTax'].fillna(0)
factor1 = _operator_ratio(cf_df, num_col='NTCF', den_col='NCFOperateA', ratio_name='ntcf2ocf')
factor2 = _operator_qoq(cf_df[['ticker', 'publishDate', 'endDate', 'NTCF']], col='NTCF', qoq_name='NTCF_qoq')
factor3 = _operator_qoq(factor1, col='ntcf2ocf', qoq_name='ntcf2ocf_qoq')
factor4 = _operator_z(factor1, col='ntcf2ocf', z_name='ntcf2ocf_z', n=8)
factor5 = _operator_z(factor2, col='NTCF_qoq', z_name='NTCF_qoq_z', n=8)
factor6 = _operator_z(factor3, col='ntcf2ocf_qoq', z_name='ntcf2ocf_qoq_z', n=8)

factor_df_list = [factor1, factor2, factor3, factor4, factor5, factor6]
ntcf_factor = fund_factor_process(factor_df_list, sdate, edate)
ntcf_factor.to_pickle(os.path.join(file_path, 'ntcf.pkl'))

print "支付税费净现金流相关因子：", ntcf_factor.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)

start_time = time.time()
print"该部分计算传统成长因子..."

# 对照组: 传统成长因子
factor1 =  _operator_qoq(bs_df[['ticker', 'publishDate', 'endDate', 'TEquityAttrP']], col='TEquityAttrP', qoq_name='NA_qoq')
factor2 = _operator_z(factor1, col='NA_qoq', z_name='NA_qoq_z', n=8)
factor3 =  _operator_qoq(is_df[['ticker', 'publishDate', 'endDate', 'NIncomeAttrP']], col='NIncomeAttrP', qoq_name='NP_qoq')
factor4 = _operator_z(factor3, col='NP_qoq', z_name='NP_qoq_z', n=8)
factor5 =  _operator_qoq(cf_df[['ticker', 'publishDate', 'endDate', 'NCFOperateA']], col='NCFOperateA', qoq_name='OCF_qoq')
factor6 = _operator_z(factor5, col='OCF_qoq', z_name='OCF_qoq_z', n=8)

factor_df_list = [factor1, factor2, factor3, factor4, factor5, factor6]
growth_factor = fund_factor_process(factor_df_list, sdate, edate)
growth_factor.to_pickle(os.path.join(file_path, 'growth.pkl'))

print "传统成长因子相关因子：", growth_factor.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


# 若已缓存因子，可以直接读取
dt_factor = pd.read_pickle(os.path.join(file_path, 'dt.pkl'))
pt_factor = pd.read_pickle(os.path.join(file_path, 'pt.pkl'))
it_factor = pd.read_pickle(os.path.join(file_path, 'it.pkl'))
at_factor = pd.read_pickle(os.path.join(file_path, 'at.pkl'))
ot_factor = pd.read_pickle(os.path.join(file_path, 'ot.pkl'))
ptcf_factor = pd.read_pickle(os.path.join(file_path, 'ptcf.pkl'))
ntcf_factor = pd.read_pickle(os.path.join(file_path, 'ntcf.pkl'))
growth_factor = pd.read_pickle(os.path.join(file_path, 'growth.pkl'))


'''

3.3 税收因子的单因子测试

对上述税收因子和传统成长因子进行有效性分析。
因子进行去极值、标准化、全中性化处理。

'''

start_time = time.time()
print"该部分分析递延所得税相关因子..."

# 因子处理
factor_df = dt_factor.copy()
factor_list = np.setdiff1d(factor_df.columns, ['ticker', 'tradeDate']).tolist()
wsn_factor = factor_process(factor_df, factor_list)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''

上述结果表明，递延所得税相关因子，并没有显著的选股能力。

'''

start_time = time.time()
print"该部分分析应交税费相关因子..."

# 因子处理
factor_df = pt_factor.copy()
factor_list = np.setdiff1d(factor_df.columns, ['ticker', 'tradeDate']).tolist()
wsn_factor = factor_process(factor_df, factor_list)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''
从上述结果看出，应交税费环比(PT_qoq)、应交税费占比（时序标准化）（pt2na_z）的选股效果较好，ICIR均值为1.65、1.89， 五分组多空组合的年化收益达到4.03%、 4.92%， 夏普比率为1.33、1.90，且分组单调。

'''

start_time = time.time()
print"该部分分析所得税相关因子..."

# 因子处理
factor_df = it_factor.copy()
factor_list = np.setdiff1d(factor_df.columns, ['ticker', 'tradeDate']).tolist()
wsn_factor = factor_process(factor_df, factor_list)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)

'''

从上述结果看出，所得税环比（时序标准化）(IT_qoq_z)的选股效果较好，ICIR均值为3.40， 五分组多空组合的年化收益达到8.09%， 夏普比率为2.79，且分组单调。
所得税占比相关因子无明显选股效果。


'''

start_time = time.time()
print"该部分分析各项税费之和相关因子..."

# 因子处理
factor_df = at_factor.copy()
factor_list = np.setdiff1d(factor_df.columns, ['ticker', 'tradeDate']).tolist()
wsn_factor = factor_process(factor_df, factor_list)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''

结论和所得税相关因子类似。
各项税费之和环比(AT_qoq)的选股效果较好，ICIR均值为3.49， 五分组多空组合的年化收益达到8.39%， 夏普比率为2.58，且分组单调。
各项税费之和占比相关因子无明显选股效果。

'''
start_time = time.time()
print"该部分分析税金及附加相关因子..."

# 因子处理
factor_df = ot_factor.copy()
factor_list = np.setdiff1d(factor_df.columns, ['ticker', 'tradeDate']).tolist()
wsn_factor = factor_process(factor_df, factor_list)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)

'''
从上述结果看出，税金及附加环比（时序标准化）(OT_qoq_z)、税金及附加占比（时序标准化）（ot2np_z）、税金及附加占比环比（时序标准化）（ot2np_qoq_z）的选股效果较好，且后两者是负向因子，ICIR均值为2.22、1.66、2.69， 五分组多空组合的年化收益达到4.08%、 3.51%、5.97%， 夏普比率为1.75、1.40、2.36，且分组单调。
因为税金及附加占比的公司间可比性相对较弱，时序标准化处理可以明显提高因子的选股效果。

'''

start_time = time.time()
print"该部分分析支付的各项税费相关因子..."

# 因子处理
factor_df = ptcf_factor.copy()
factor_list = np.setdiff1d(factor_df.columns, ['ticker', 'tradeDate']).tolist()
wsn_factor = factor_process(factor_df, factor_list)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)

'''

从上述结果看出，支付的各项税费环比（时序标准化）(PTCF_qoq_z)，ICIR均值为1.88， 五分组多空组合的年化收益达到4.28%， 夏普比率为1.83，且分组单调。
支付的各项税占比相关因子无明显选股效果

'''

start_time = time.time()
print"该部分分析支付税费净现金流相关因子..."

# 因子处理
factor_df = ntcf_factor.copy()
factor_list = np.setdiff1d(factor_df.columns, ['ticker', 'tradeDate']).tolist()
wsn_factor = factor_process(factor_df, factor_list)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)

'''

支付的各项税费相关因子类似。
支付税费净现金流环比（时序标准化）(NTCF_qoq_z)，ICIR均值为1.82， 五分组多空组合的年化收益达到3.63%， 夏普比率为1.51，且分组单调。
支付税费净现金流占比相关因子无明显选股效果。
两者比较，支付的各项税费季度同比（时序标准化）(PTCF_qoq_z)的选股效果更好。


'''

start_time = time.time()
print"该部分分析传统成长相关因子..."

# 因子处理
factor_df = growth_factor.copy()
factor_list = np.setdiff1d(factor_df.columns, ['ticker', 'tradeDate']).tolist()
wsn_factor = factor_process(factor_df, factor_list)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''

从上述结果看出，净资产环比（时序标准化）（NA_qoq_z）、净利润环比（时序标准化）（NP_qoq_z）的选股效果较好，ICIR均值为2.93、4.55， 五分组多空组合的年化收益达到8.51%、 11.52%， 夏普比率为2.49、3.68，且分组单调。
经营性净现金流环比因子没有明显的选股能力。
时序标准化处理能够增强传统成长因子的选股能力。
   调试 运行
文档
 代码  策略  文档
3.4 税收因子的叠加性分析

综合3.3的分析，一共筛选出了9个税收因子，具有较好的选股效果。分别为：‘PT_qoq’, ‘pt2na_z’, ‘IT_qoq_z’, ‘AT_qoq_z’, ‘OT_qoq_z’, ‘ot2np_z’, ‘ot2np_qoq_z’, ‘PTCF_qoq_z’, ‘NTCF_qoq_z’。
其中，‘pt2na_z’, 'pt2na_qoq_z’为负向因子，故调整方向。
总结它们的选股效果如下。

'''
start_time = time.time()
print"该部分整合选股效果较好的税相关因子..."

select_factor_list = ['PT_qoq', 'pt2na_z', 'IT_qoq_z', 'AT_qoq_z', 'OT_qoq_z', 'ot2np_z', 'ot2np_qoq_z', 'PTCF_qoq_z', 'NTCF_qoq_z']
ref_factor_list = ['NA_qoq_z', 'NP_qoq_z', 'OCF_qoq_z']

df1 = pt_factor[['ticker', 'tradeDate', 'PT_qoq', 'pt2na_z', 'pt2na_qoq_z']].set_index(['ticker', 'tradeDate'])
df2 = it_factor[['ticker', 'tradeDate', 'IT_qoq_z']].set_index(['ticker', 'tradeDate'])
df3 = at_factor[['ticker', 'tradeDate', 'AT_qoq_z']].set_index(['ticker', 'tradeDate'])
df4 = ot_factor[['ticker', 'tradeDate', 'OT_qoq_z', 'ot2np_z', 'ot2np_qoq_z']].set_index(['ticker', 'tradeDate'])
df5 = ptcf_factor[['ticker', 'tradeDate', 'PTCF_qoq_z']].set_index(['ticker', 'tradeDate'])
df6 = ntcf_factor[['ticker', 'tradeDate', 'NTCF_qoq_z']].set_index(['ticker', 'tradeDate'])
df7 = growth_factor[['ticker', 'tradeDate', 'NA_qoq_z', 'NP_qoq_z', 'OCF_qoq_z']].set_index(['ticker', 'tradeDate'])
all_df = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=1).reset_index()
all_df['ot2np_qoq_z'] = -all_df['ot2np_qoq_z']
all_df['ot2np_z'] = -all_df['ot2np_z']
print "筛选出的税收因子：", all_df.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''
首先观察税收因子之间，以及和传统成长因子的相关性。

'''

_ = calc_factor_corr(all_df, select_factor_list+ref_factor_list, show_corr_plot=True, corr_method='spearman')

'''

PT_qoq和pt2na_z的相关性较高，达到0.48。
IT_qoq_z和AT_qoq_z的相关性达到0.83， 主要是因为税金及附加占比较小，各项税费之和的主体就是所得税。它们与NP_qoq_z的相关性较高，达到0.5左右，主要是因为说所得税和净利润高度相关。
ot2np_qoq_z与NP_qoq_z的相关性较高，达到0.41，说明税金及附加占比的变化又要来源于净利润的变化。
PTCF_qoq_z和NTCF_qoq_z的相关性较高，主要是因为收到的税收返还占比较小。

'''

start_time = time.time()

factor_df = all_df.copy()
factor_list = ['PT_qoq', 'pt2na_z', 'IT_qoq_z', 'AT_qoq_z', 'OT_qoq_z', 'ot2np_z', 'ot2np_qoq_z', 'PTCF_qoq_z', 'NTCF_qoq_z', 'NA_qoq_z', 'NP_qoq_z', 'OCF_qoq_z']
wsn_factor = factor_process(factor_df, factor_list)

# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12, show_fig=False)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''

因子选股效果对比来看，税收因子的选股效果比不上传统成长因子，因此税收因子的作用应该是能否提供辅助的增量信息。
税收因子中，IT_qoq_z和AT_qoq_z的表现最好，但是他们和NP_qoq_z的相关性较高，因此他们很难提供增量信息。ot2np_qoq_z也是类似。
考虑到相关性和因子效果，pt2na_z优于PT_qoq，PTCF_qoq_z优于NTCF_qoq_z。
故最终筛选出’pt2na_z’、‘OT_qoq_z’、‘ot2np_z’、'PTCF_qoq_z’4个因子。

'''

factor_list = ['pt2na_z', 'OT_qoq_z', 'ot2np_z', 'PTCF_qoq_z', 'NP_qoq_z']
_ = calc_factor_corr(all_df, factor_list, show_corr_plot=True, corr_method='spearman')
test_discribe(ic_res, perf_df, factor_list, annual_len=12, show_fig=True)


'''

具体地，从多空净值曲线来看，近年来，ot2np_z有失效的迹象，故删除。
pt2na_z、OT_qoq_z、PTCF_qoq_z的单调性和多空净值曲线均表现较好。
   调试 运行
文档
 代码  策略  文档
－首先，先对税收因子进行等权合成

'''


# 因子处理
factor_df = all_df.copy()
factor_df = factor_process(factor_df, ['pt2na_z', 'OT_qoq_z', 'PTCF_qoq_z', 'NP_qoq_z'], is_neut=False)
factor_df['tax_comb'] = factor_df[['pt2na_z', 'OT_qoq_z', 'PTCF_qoq_z']].mean(axis=1)
_ = calc_factor_corr(factor_df, ['tax_comb', 'NP_qoq_z'], show_corr_plot=True, corr_method='spearman')
factor_list = ['tax_comb']
wsn_factor = factor_process(factor_df, factor_list, is_win=False)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12, show_fig=True)


'''
从上述结果可以看出，合成税收因子好于单个因子，说明三者的选股效果相对独立，合成后达到“一加一大于二”的效果。
合成税收因子的ICIR达到2.95，五分组的多空年化收益达到7.31%，夏普比率达到2.78。五分组区分度明显，多空净值曲线收益稳定。
   调试 运行
文档
 代码  策略  文档
进一步，将合成税收因子，和净利润成长因子进行等权合成。

'''

start_time = time.time()

# 因子处理
factor_df = all_df.copy()
factor_df = factor_process(factor_df, ['pt2na_z', 'OT_qoq_z', 'PTCF_qoq_z', 'NP_qoq_z'], is_neut=False)
factor_df['tax_comb'] = factor_df[['pt2na_z', 'OT_qoq_z', 'PTCF_qoq_z']].mean(axis=1)
factor_df['growth_comb'] = factor_df[['tax_comb', 'NP_qoq_z']].mean(axis=1, skipna=False)
factor_list = ['growth_comb', 'NP_qoq_z']
wsn_factor = factor_process(factor_df, factor_list, is_win=False)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12, show_fig=True)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)

'''

对比结果来看，合成税收因子的净利润成长因子多空年化收益为12.64%，夏普比率为3.95，优于净利润成长因子。
从分组结果可以看出，税收因子的增益在于，加强了多头的收益。
   调试 运行
文档
 代码  策略  文档
若每日，将合成税收因子的因子值大于50%的个股，记为1；小于50%的记为0。再和净利润成长因子进行合成。


'''

start_time = time.time()

# 因子处理
factor_df = all_df.copy()
factor_df = factor_process(factor_df, ['pt2na_z', 'OT_qoq_z', 'PTCF_qoq_z', 'NP_qoq_z'], is_neut=False)
factor_df['tax_comb'] = factor_df[['pt2na_z', 'OT_qoq_z', 'PTCF_qoq_z']].mean(axis=1)
factor_df['tax_comb_flag'] = factor_df.groupby('tradeDate')['tax_comb'].apply(lambda x: x>x.dropna().quantile(0.5))
factor_df['tax_comb_flag'] = np.where(factor_df['tax_comb_flag'], 0.5, -0.5)
factor_df['growth_comb'] = factor_df[['tax_comb_flag', 'NP_qoq_z']].mean(axis=1, skipna=False)
factor_list = ['growth_comb', 'NP_qoq_z']
wsn_factor = factor_process(factor_df, factor_list, is_win=False)
# 因子测试
ic_res, perf_df = factor_test_summary(wsn_factor, factor_list, ngrp=5)
test_discribe(ic_res, perf_df, factor_list, annual_len=12, show_fig=True)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)

'''

从对比结果来看，合成税收因子的净利润成长因子，多空年化收益为11.57%，夏普比率为3.95，依然略优于单个净利润成长因子。
该方法，税收因子的主要增益是降低波动率，提高月度胜率。
   调试 运行
文档
 代码  策略  文档
总结：

三大报表中均有与税收相关的会计科目，其中，税务口径核算的税费可参考资产负债表中的应交税费科目；会计口径核算的税费可参考利润表中的所得税费用和税金及附加；税务口径与会计口径不一致导致的暂时性差异形成资产负债表中的递延所得税资产（负债）。现金流量表中，支付的各项税费为当期实际缴纳税费的现金流。
利用三大报表的所有税收会计科目，构造了42个因子，包含成长因子和质量因子，并与传统成长因子进行比较。单因子测试结果显示，应交税费占比、税金及附加环比、支付的各项税费环比这个三个税收因子选股效果最好，且和净利润环比相关性较低。且整体来说，时序标准化处理可以提升基本面因子的选股效果。
三个税收因子合成后，20100101-20200731期间，月度调仓，IC均值为2.01%，ICIR为2.95，五分组多空年化收益率为7.31%，夏普比率达到2.78，且五分组区分度显著，单调性好。合成税收因子叠加净利润环比因子，多空年化收益为12.64%，夏普比率为3.95，对比净利润环比因子多空年化收益为11.52%，夏普比率为3.68，叠加后的因子优于单个净利润成长因子，说明税收因子有增益信息。
'''

