# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:21:40 2020

@author: Asus
"""

'''
导读
A. 研究目的：

公募基金持仓股票的组成及收益情况一直备受关注，本报告参考海通证券研报《基金重仓超配因子及其对指数增强组合的影响》中的研究方法，用以探索基金重仓超配个股的超额收益情况。
B. 研究结论：

2016年前基金重仓股票个数持续提升，之后持股日渐集中，2020一季度基金池重仓持股个数为1143，占全市场总股票数之比为30%。对超配股进行风格与行业分析发现，近三年超配股行业上偏向医药生物、电子，风格上偏好高成长，高流行性的股票。

从截面回归的角度来看，重仓超配因子溢价显著的月度占比高达63.72%。从多空分组的角度看，该因子的多头组合相对于空头组合存在年化3.61%的超额收益，月度胜率58.93%，夏普比0.45。

选择不同业绩的基金池构建因子，会影响HS300增强模型的表现，业绩靠后50%基金所构建的重仓超配因子表现最好，其年化收益为3.8%，信息比率为0.51，而业绩前50%基金构建的因子表现最差，其年化收益仅为1.9%，信息比率为0.26。

C. 文章结构: 本文共分为3个部分，具体如下

一、基金重仓超配个股的特征，该部分主要进行基金池的筛选，及其超配个股的风格、行业信息

二、基金重仓超配因子，该部分主要利用不同类型的基金池构建超配因子，并进行IC、分组回测

三、指数增强组合回测，该部分主要依据超配因子构建HS300增强组合回测

D. 运行时间说明

一、基金重仓超配个股的特征，需要25分钟左右

二、基金重仓超配因子，需要5分钟左右

三、指数增强组合回测，需要40分钟左右

总耗时70分钟左右

特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
(链接：[https://uqer.datayes.com/community/share/eLNeQy0p3r0lRu9I5WoZ5YOw2ng0/private]）；密码：6278。请前往查看并注意保密。)
请在运行之前，克隆上面的代码，并存成lib（右上角->另存为lib,不要修改名字）

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

一、基金重仓超配个股的特征
该部分耗时 25分钟
该部分内容包括:

1.1 基金重仓股及重仓超配个股
1.2 重仓超配个股的风格特征与行业特征
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


1.1 基金重仓股及重仓超配个股

首先根据基金类型选取股票型基金和混合型基金。混合型基金需要过滤非偏股型的基金
基金池主要为主动型基金，去除被动型基金，同时剔除封闭式基金，剔除分级基金，FOF基金等

'''

#--------------------------------定义函数, 获取基金基本信息、资产信息、季度收益情况----------------------------------------------
import os, time, math
import datetime as dt
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
import scipy.stats as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from CAL.PyCAL import *

start_time = time.time()
# 起始、终止时间
begin_date = '2010-01-01'
end_date = '2020-06-30'

raw_data_dir = "./fund_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

def get_fund_info():
    # 获取基金基本信息
    data_path = os.path.join(raw_data_dir, 'fund_basic_info.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0)
    else:
        df = DataAPI.FundGet(category=u"E,H", operationMode=u"O", field=['secID', 'secShortName', 'category', 'establishDate', 'expireDate', 'isClass', 'isFof', 'isQdii', 'indexFund', 'managementFullName'],pandas="1").dropna(subset=['secID', 'establishDate'])
        df = df.query("isClass==0")
        df = df.query("isFof==0")
        df = df.query("isQdii==0")
        df = df[~df['indexFund'].isin(['I', 'EI'])]

        df['secShortName2'] = df['secShortName'].apply(lambda x: x[:-2] if x[-2:] in ['-A', '-B', '-C'] else x)
        df = df.sort_values('secShortName2').drop_duplicates(subset=['category', 'establishDate', 'managementFullName', 'secShortName2'])
        df = df.drop(['secShortName2', 'isClass', 'isFof', 'isQdii', 'indexFund'], axis=1).sort_values(['managementFullName', 'establishDate'])
        df.to_csv(data_path)
    return df

def get_fund_asset_info(secID_list, equity_ratio=0.7, quarters=4):
    # 获取基金规模信息，并获取过去四个季度持股比例均超过70%的基金
    data_path = os.path.join(raw_data_dir, 'fund_asset.csv')
    if os.path.exists(data_path) and False:
        asset_info = pd.read_csv(data_path, index_col=0)
    else:
        asset_info = DataAPI.FundAssetsGet(secID=secID_list, field=['secID', 'reportDate', 'totalAsset', 'netAsset', 'equityMarketValue', 'equityRatioInTa', 'publishDate', 'currencyCd'], pandas="1")
        asset_info = asset_info.query("currencyCd=='CNY'")
        asset_info['equityRatioInTa'] = asset_info['equityRatioInTa'].fillna(0)
        report_asset_info = asset_info[asset_info['reportDate'].apply(lambda x: x[-5:] in ['03-31', '06-30', '09-30', '12-31'])]
        equity_ratio_df = pd.pivot_table(report_asset_info, index='reportDate', columns='secID', values='equityRatioInTa')
        equity_ratio_df = (equity_ratio_df >= equity_ratio)
        equity_ratio_df = (equity_ratio_df.rolling(quarters).sum() == quarters)
        equity_ratio_df = equity_ratio_df.stack().reset_index()
        equity_ratio_df.columns = ['reportDate', 'secID', 'isStockFund']
        asset_info = pd.merge(asset_info, equity_ratio_df, on=['reportDate', 'secID'], how='left')
        asset_info['isStockFund'] = asset_info['isStockFund'].fillna(False)
        isstock_num = asset_info.groupby('secID')['isStockFund'].sum()
        isstock_cnt = asset_info.groupby('secID')['isStockFund'].count()
        isstock_pert = isstock_num * 1.0 / isstock_cnt
        secID_list = list(set(isstock_num[isstock_num>0].index.tolist()) & set(isstock_pert[isstock_pert>equity_ratio].index.tolist()))
        asset_info = asset_info[asset_info['secID'].isin(secID_list)]
        asset_info.to_csv(data_path)
    
    return asset_info

def get_quarter_period(x):
    # 获取季度信息
    year, month = x[:4], x[5:7]
    if month <= '03':
        return year + '-03-31', year + '-06-30'
    elif month <= '06':
        return year + '-06-30', year + '-09-30'
    elif month <= '09':
        return year + '-09-30', year + '-12-31'
    else:
        return year + '-12-31', str(int(year)+1) + '-03-31'
    
def cal_fund_ret(secID_list, begin_date, end_date):
    # 未来季度收益率，过去一年的夏普率
    def _cal_sp(x, year_days=250, riskfree_rate=0.03):
        if len(x) <= year_days * 0.8:
            return np.nan
        x = x.copy()
        x['cumRet'] = (1 + x['adjNavChgPct']).cumprod()
        volatility = np.std(x['adjNavChgPct']) * np.sqrt(year_days)
        if volatility == 0:
            return None
        annualized_return = (x['cumRet'].values[-1]) ** (year_days / len(x['cumRet'])) - 1.0
        sp = (annualized_return - riskfree_rate) / volatility
        
        return sp
    
    def _cal_ret(x, month_days=21):
        if len(x) < month_days * 0.5:
            return np.nan        
        if x.min() < -0.12 or x.max() > 0.12:
            return np.nan
        return (1 + x).prod() - 1
        
    data_path = os.path.join(raw_data_dir, 'fund_quarter_perf.csv')
    if os.path.exists(data_path):
        perf_df = pd.read_csv(data_path, index_col=0)
    else:
        data_list = []
        all_quarter_list = [str(year) + quarter for year in range(int(begin_date[:4]), int(end_date[:4]) + 1) for quarter in ['-03-31', '-06-30', '-09-30', '-12-31']]
        df = None
        for i, quarter in enumerate(all_quarter_list):
            if quarter > '2020-06-30':
                break
            begin_quarter = str(int(quarter[:4])-1) + quarter[4:]
            if '03-31' in quarter:
                print quarter, time.asctime()
            if df is None:
                df = DataAPI.FundNavGet(secID=secID_list, beginDate=begin_quarter, endDate=quarter, field=['secID', 'endDate', 'adjNavChgPct'])
                df['adjNavChgPct'] = df['adjNavChgPct'].fillna(0) / 100.0
                df[['quarter', 'nextQuarter']] = df['endDate'].apply(lambda x: pd.Series(index=['quarter', 'nextQuarter'], data=get_quarter_period(x)))
            else:
                new_df = DataAPI.FundNavGet(secID=secID_list, beginDate=all_quarter_list[i-1], endDate=quarter, field=['secID', 'endDate', 'adjNavChgPct'])
                new_df['adjNavChgPct'] = new_df['adjNavChgPct'].fillna(0) / 100.0
                new_df[['quarter', 'nextQuarter']] = new_df['endDate'].apply(lambda x: pd.Series(index=['quarter', 'nextQuarter'], data=get_quarter_period(x)))
                df = pd.concat([df, new_df]).drop_duplicates()
                df = df[df['quarter'] > begin_quarter]
                
            sp = df.groupby('secID').apply(lambda x: _cal_sp(x))
            sp.index.name = 'secID'
            quarter_df = df[df['quarter'] == quarter]
            ret = quarter_df.groupby('secID')['adjNavChgPct'].apply(lambda x: _cal_ret(x))
            ret.index.name = 'secID'
            
            quarter_perf_df = pd.concat([sp, ret], axis=1).dropna(how='all')
            quarter_perf_df.columns = ['SP', 'ret']
            quarter_perf_df['reportDate'] = quarter
            quarter_perf_df.index.name = 'secID'
            data_list.append(quarter_perf_df.reset_index())
            
        perf_df = pd.concat(data_list)
        perf_df = perf_df[['secID', 'reportDate', 'SP', 'ret']].drop_duplicates().dropna(subset=['SP', 'ret'], how='all')
        perf_df[['quarter', 'nextQuarter']] = perf_df['reportDate'].apply(lambda x: pd.Series(index=['quarter', 'nextQuarter'], data=get_quarter_period(x)))
        perf_df = pd.merge(perf_df[['secID', 'reportDate', 'SP', 'ret', 'nextQuarter']].rename(columns={'nextQuarter': 'nextReportDate'}), perf_df[['secID', 'quarter', 'ret']].rename(columns={'ret': 'nextRet'}), left_on=['secID', 'nextReportDate'], right_on=['secID', 'quarter']).sort_values(['secID', 'reportDate']).drop(['quarter'], axis=1)
        
        perf_df.to_csv(data_path)

    return perf_df


def get_cap_info(begin_date, end_date):
    '''
    获取个股每个季度末的市值
    '''
    
    data_path = os.path.join(raw_data_dir, 'ticker_quarter_cap.csv')
    if os.path.exists(data_path):
        all_cap_df = pd.read_csv(data_path, index_col=0, dtype={'ticker': 'str'})
    else:
        cal_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin_date, endDate=end_date, field=u"", pandas="1")
        cal_df = cal_df.query("isOpen==1")
        equ_df = DataAPI.EquGet(equTypeCD=u"A", listStatusCD=u"", field=['secID', 'ticker', 'listDate', 'delistDate'], pandas="1")
        hs300_cons_df = DataAPI.IdxCloseWeightGet(ticker='000300', beginDate=begin_date, endDate=end_date, field=['consTickerSymbol', 'effDate', 'weight'],pandas="1")

        data_list = []
        all_quarter_list = [str(year) + quarter for year in range(int(begin_date[:4]), int(end_date[:4]) + 1) for quarter in ['-03-31', '-06-30', '-09-30', '-12-31']]

        for i, quarter in enumerate(all_quarter_list):
            if quarter > '2020-06-30':
                break
            
            A_ticker_list = list(equ_df[(equ_df['listDate'] <= quarter) & ((equ_df['delistDate'] > quarter) | (equ_df['delistDate'].isnull()))]['ticker'])
            tdate = cal_df.query("calendarDate<=@quarter").iloc[-1]['calendarDate']
            cap_df = DataAPI.MktEqudGet(tradeDate=tdate, field=u"ticker,tradeDate,negMarketValue",pandas="1")
            cap_df = cap_df[cap_df['ticker'].isin(A_ticker_list)]
            cap_df = cap_df.rename(columns={'tradeDate': 'reportDate', 'negMarketValue': 'cap'})
            cap_df['reportDate'] = quarter
            
            tdate = hs300_cons_df.query("effDate<=@quarter").iloc[-1]['effDate']
            hs300_weight = hs300_cons_df.query("effDate==@tdate").set_index('consTickerSymbol')['weight'].to_dict()
            cap_df['hs300_weight'] = cap_df['ticker'].apply(lambda x: hs300_weight.get(x, 0) / 100.0)
            
            data_list.append(cap_df)
        all_cap_df = pd.concat(data_list).reset_index(drop=True)
        all_cap_df.to_csv(data_path)
    
    return all_cap_df


# 获取基金基本信息、资产信息、季度收益情况
fund_info_df = get_fund_info()
fund_asset_df = get_fund_asset_info(fund_info_df['secID'].unique().tolist())
fund_quarter_perf_df = cal_fund_ret(fund_asset_df['secID'].unique().tolist(), begin_date, end_date)
ticker_quarter_cap_df = get_cap_info(begin_date, end_date)

print "Time cost: %s seconds" % (time.time() - start_time)
print(u'基金信息格式为')
print(fund_info_df.head(5).to_html())
print(u'基金资产信息格式为')
print(fund_asset_df.head(5).to_html())
print(u'基金季度收益格式为')
print(fund_quarter_perf_df.head(5).to_html())
print(u'个股季度市值格式为')
print(ticker_quarter_cap_df.head(5).to_html())

'''
将上节构造的基金池当期季报披露的十大重仓股取并集，作为股票池进行分析
'''

# ---------------------------  获取基金持仓信息
start_time = time.time()
def get_fund_hold_info(fund_info, begin_date, end_date):
    # 获取基金持仓信息

    hold_info = DataAPI.FundHoldingsGet(secID=fund_info['secID'].unique(), beginDate=begin_date, endDate=end_date, secType="E", field=['secID', 'reportDate', 'holdingTicker', 'ratioInNa', 'publishDate', 'currencyCd'],pandas="1")
    hold_info = hold_info.query("currencyCd=='CNY'")
    hold_info['ratioInNa'] = hold_info['ratioInNa'].fillna(0)
    hold_info = hold_info[hold_info['reportDate'].apply(lambda x: x[-5:] in ['03-31', '06-30', '09-30', '12-31'])]
        
    hold_info = pd.merge(hold_info, fund_info[['secID', 'reportDate', 'totalAsset', 'netAsset', 'equityMarketValue', 'equityRatioInTa']], on=['secID', 'reportDate'])
    hold_info = hold_info.sort_values(['secID', 'reportDate', 'ratioInNa'], ascending=[True, True, False])
    
    return hold_info

fund_hold_info = get_fund_hold_info(fund_asset_df, begin_date, end_date)
print "Time cost: %s seconds" % (time.time() - start_time)
print(u'基金持仓信息格式为')
print(fund_hold_info.head(5).to_html())


'''

统计基金重仓股票池中基金对于每只个股的持仓市值占比，并跟选取基准中该股票的权重进行对比，若重仓股票池权重占比高于基准权重占比，称之为超配个股
为了统一，本报告中基准选取HS300

'''

# ---------------------------  获取重仓股个数占比、市值占比
def get_stock_info(fund_hold_info, ticker_quarter_cap_df, fund_quarter_perf_df):
    # 按过去一年收益率分成两组
    fund_quarter_perf_df = fund_quarter_perf_df.dropna(subset=['ret'])
    fund_quarter_perf_df['group'] = fund_quarter_perf_df.groupby('reportDate')['ret'].apply(lambda x: (x.rank(method='first') - 1) / len(x) * 2).astype(int)
    fund_hold_info = pd.merge(fund_hold_info, fund_quarter_perf_df[['secID', 'reportDate', 'group']], on=['secID', 'reportDate'])
    
    overweight_dict = {}
    cnt_dict = {}
    key_list = ['bottom', 'top', 'all']
    for i, group in enumerate([[0], [1], [0, 1]]):
        key = key_list[i]
        fund_info = fund_hold_info[fund_hold_info['group'].isin(group)]

        fund_cnt = fund_info.groupby('reportDate').apply(lambda x: len(x['holdingTicker'].unique()))
        ticker_cnt = ticker_quarter_cap_df.groupby('reportDate').apply(lambda x: len(x['ticker'].unique()))

        cap_df = ticker_quarter_cap_df.copy()
        cap_df['cap_pert'] = cap_df.groupby('reportDate')['cap'].apply(lambda x: x / np.sum(x))

        fund_stat = fund_info.groupby(['holdingTicker', 'reportDate']).apply(lambda x: (x['equityMarketValue'] * x['ratioInNa'] / 100.0).sum()).reset_index()
        fund_stat.columns = ['ticker', 'reportDate', 'hold_cap']
        fund_stat['hold_cap_pert'] = fund_stat.groupby('reportDate')['hold_cap'].apply(lambda x: x / np.sum(x))

        overweight_info = pd.merge(fund_stat, cap_df, on=['ticker', 'reportDate'])
        overweight_info['overweight'] = np.where(overweight_info['hold_cap_pert'] > overweight_info['hs300_weight'], 1, 0)

        overweight_cnt = overweight_info.groupby('reportDate')['overweight'].sum()
        overweight_cap_pert = overweight_info.groupby('reportDate').apply(lambda x: (x['hold_cap_pert'] * x['overweight']).sum())

        cnt_info = pd.concat([fund_cnt, ticker_cnt, overweight_cnt, overweight_cap_pert], axis=1)
        cnt_info.columns = ['fund_ticker_cnt', 'A_ticker_cnt', 'overweight_cnt', 'overweight_cap_pert']
        
        overweight_dict[key] = overweight_info
        cnt_dict[key] = cnt_info
    
    return overweight_dict, cnt_dict
    
top10_hold_info = fund_hold_info.groupby(['secID', 'reportDate']).head(10)
overweight_dict, cnt_dict = get_stock_info(top10_hold_info, ticker_quarter_cap_df, fund_quarter_perf_df)


'''

下节统计了每期重仓持有的股票总个数及其占市场总股票数之比。可以看出16年前，基金重仓股票个数持续提升，之后持股日渐集中，2020一季度基金池重仓持股个数为1143，占全市场总股票数之比为30%

'''

# ---------------------------  股票总个数及其占市场总股票数之比
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

stat_info = cnt_dict['all']
ind = np.arange(len(stat_info))
width = 0.5

ax1.plot(ind + width, stat_info['fund_ticker_cnt']/stat_info['A_ticker_cnt'], color='b', label=u'重仓股个股/总个股数')
ax1.legend(loc=2, prop=font)
ax1.set_ylim(0, 1)
ax1.set_yticklabels(['%d%%' % (item*100) for item in ax1.get_yticks()])
ax1.set_xticks(ind + width)
ax1.set_xticklabels(stat_info.index, rotation=45)

ax2.plot(ind + width, stat_info['fund_ticker_cnt'], color='r', label=u'重仓股个数(右轴)')
ax2.legend(loc=1, prop=font)
ax2.grid(False)
ax2.set_ylim(0, 2000)
ax1.set_title(u'基金重仓股个数', fontsize=16, fontproperties=font)
plt.show()


'''

下节统计了超配个股的数目及其市值占比，2017年以来，超配个股数逐渐减少，而持仓市值占比逐渐攀升。

'''

# ---------------------------  超配个股的数目及其市值占比
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

ind = np.arange(len(stat_info))
width = 0.5

ax1.plot(ind + width, stat_info['overweight_cnt']/stat_info['fund_ticker_cnt'], color='b', label=u'个股数占比')
ax1.legend(loc=2, prop=font)
ax1.set_yticklabels(['%d%%' % (item*100) for item in ax1.get_yticks()])
ax1.set_xticks(ind + width)
ax1.set_xticklabels(stat_info.index, rotation=45)

ax2.plot(ind + width, stat_info['overweight_cap_pert'], color='r', label=u'持仓市值占比(右轴)')
ax2.legend(loc=1, prop=font)
ax2.grid(False)
ax2.set_yticklabels(['%d%%' % (item*100) for item in ax2.get_yticks()])

ax1.set_title(u'重仓超配股个数及持仓市值占比', fontsize=16, fontproperties=font)
plt.show()


'''

1.2 重仓超配个股的风格特征与行业特征

下面分析重仓超配个股的风格与行业偏好情况，依据风险模型数据计算每期持仓的风格因子与行业因子暴露，并计算近3年的情况与历史10年进行对比

'''


# ---------------------------  获取超配股行业与风格信息
StyleName = ['BETA', 'MOMENTUM', 'SIZE', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY', 'SIZENL']
IndustryName = ['Bank', 'RealEstate', 'Health', 'Transportation', 'Mining', 'NonFerMetal', 'HouseApp', 'LeiService', 'MachiEquip', 'BuildDeco', 'CommeTrade', 'CONMAT', 'Auto', 'Textile', 'FoodBever', 'Electronics', 'Computer', 'LightIndus', 'Utilities', 'Telecom', 'AgriForest', 'CHEM', 'Media', 'IronSteel', 'NonBankFinan', 'ELECEQP', 'AERODEF', 'Conglomerates']

def get_style_industry_factor(df):
    # 获取超配股的风格与行业偏好
    cal_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin_date, endDate=end_date)
    quarter_list = cal_df.query("isQuarterEnd==1")['calendarDate'].tolist()
    
    df = df.query("overweight==1")
    hold_quarter_list = sorted(df['reportDate'].unique().tolist())
    hold_style_industry_df = pd.DataFrame(index=hold_quarter_list, columns=StyleName+IndustryName)
    for quarter in quarter_list:
        style_factor_df = DataAPI.RMExposureDayGet(tradeDate=quarter.replace('-', ''), field=['ticker', 'tradeDate'] + StyleName + IndustryName)
        format_quarter = quarter[:7] + ('-31' if quarter[5:7] in ['03', '12'] else '-30')
        hold_ticker_list = df.query("reportDate==@format_quarter")['ticker'].tolist()
        hold_style = style_factor_df.set_index('ticker')[StyleName + IndustryName].loc[hold_ticker_list].mean()
        hold_style_industry_df.loc[format_quarter] = hold_style

    return hold_style_industry_df

overweight_info = overweight_dict['all']
hold_style_industry_df = get_style_industry_factor(overweight_info)


'''

从10年至20年，超配股的风格发生了部分变化，近3年偏好高成长，高流行性，高beta的股票，其中动量、盈利风格因子发生的变化较明显

'''
# ---------------------------  展示超配股风格信息
hold_style_df = hold_style_industry_df[StyleName]
all_avg = hold_style_df.mean()
latest_3years_avg = hold_style_df[hold_style_df.index > '2018-01-01'].mean()
style_avg = pd.concat([all_avg, latest_3years_avg], axis=1)
style_avg.columns = [u'历史平均(2010-2020)', u'最近3年平均(2018-2020)']
style_avg[u'最近三年相对历史配置的变化'] = style_avg[u'最近3年平均(2018-2020)'] - style_avg[u'历史平均(2010-2020)']
style_avg = style_avg.sort_values(u'最近3年平均(2018-2020)', ascending=False)

fig = plt.figure(figsize=(24, 6))
for i in range(style_avg.shape[1]):
    k = 100 + 70 + i + 1
    ax = style_avg.iloc[:, i].plot(kind='barh', ax=fig.add_subplot(k), color='r')
    ax.set_xlabel(style_avg.columns[i], fontproperties=font)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    if i == 0:
        s = ax.set_ylabel(u'风格因子', fontproperties=font, fontsize=14)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
        

'''

行业上，近些年超配股偏好电子，医药生物行业，其中电子行业相对历史配置增加最多。

'''

# ---------------------------  展示超配股行业信息
industry_name_dict = {'Bank': '银行',
    'RealEstate': '房地产',
    'Health': '医药生物',
    'Transportation': '交通运输',
    'Mining': '采掘',
    'NonFerMetal': '有色金属',
    'HouseApp': '家用电器',
    'LeiService': '休闲服务',
    'MachiEquip': '机械设备',
    'BuildDeco': '建筑装饰',
    'CommeTrade': '商业贸易',
    'CONMAT': '建筑材料',
    'Auto': '汽车',
    'Textile': '纺织服装',
    'FoodBever': '食品饮料',
    'Electronics': '电子',
    'Computer': '计算机',
    'LightIndus': '轻工制造',
    'Utilities': '公用事业',
    'Telecom': '通信',
    'AgriForest': '农林牧渔',
    'CHEM': '化工',
    'Media': '传媒',
    'IronSteel': '钢铁',
    'NonBankFinan': '非银金融',
    'ELECEQP': '电气设备',
    'AERODEF': '国防军工',
    'Conglomerates': '综合'
}

hold_industry_df = hold_style_industry_df[IndustryName]
hold_industry_df.columns = [industry_name_dict[item] for item in hold_industry_df.columns]
all_avg = hold_industry_df.mean()
latest_3years_avg = hold_industry_df[hold_industry_df.index > '2018-01-01'].mean()
style_avg = pd.concat([all_avg, latest_3years_avg], axis=1)
style_avg.columns = [u'历史平均(2010-2020)', u'最近3年平均(2018-2020)']
style_avg[u'最近三年相对历史配置的变化'] = style_avg[u'最近3年平均(2018-2020)'] - style_avg[u'历史平均(2010-2020)']
style_avg = style_avg.sort_values(u'最近3年平均(2018-2020)', ascending=False)

fig = plt.figure(figsize=(32, 6))
for i in range(style_avg.shape[1]):
    k = 100 + 70 + i + 1
    ax = style_avg.iloc[:, i].plot(kind='barh', ax=fig.add_subplot(k), color='r')
    ax.set_xlabel(style_avg.columns[i], fontproperties=font)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    if i == 0:
        s = ax.set_ylabel(u'行业', fontproperties=font, fontsize=14)
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font, rotation=0)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
        

'''

二、基金重仓超配因子
该部分耗时 5分钟
该部分内容包括:

2.1 重仓超配因子的构建
2.2 重仓超配因子的测试
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


2.1 重仓超配因子的构建

对第一节的每期重仓超配个股赋予哑变量1，其余个股赋值0，构建当期的信号值
同时为了考察不同基金池对因子的影响，根据过去一年的收益将基金池分为两组：业绩排名靠前和业绩排名靠后的基金，对每组基金池分别构建重仓超配因子


'''


# ---------------------------  计算每期因子值
import datetime as dt
from dateutil.relativedelta import relativedelta

start_time = time.time()

equ_df = DataAPI.EquGet(equTypeCD=u"A", listStatusCD=u"", field=['secID', 'ticker', 'listDate', 'delistDate'], pandas="1")
publish_date_dict = top10_hold_info.groupby('reportDate')['publishDate'].max().to_dict()

def str2date(date_str):
    """转换日期格式 "YYYYMMDD" / "YYYY-MM-DD"string to datetime object"""
    date_str = date_str.replace("-", "")
    date_obj = dt.datetime(int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]))
    return date_obj

def get_universe(date_str, list_date=90):
    '''
    给定日期，选取符合条件的所有A股ticker
    '''
    format_date = str2date(date_str).strftime("%Y-%m-%d")
    list_date_need = (str2date(date_str) + relativedelta(days=-list_date)).strftime("%Y-%m-%d")
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

def get_overweight_signal(overweight_info):
    '''
        基于重仓信息，构建月度哑变量因子
    '''
    cal_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin_date, endDate=end_date, field=u"", pandas="1")
    cal_df = cal_df[cal_df.isMonthEnd == 1]
    
    data = overweight_info.copy()
    data['date'] = data['reportDate'].apply(lambda x: publish_date_dict[x])
    data = data.drop('reportDate', axis=1)
    
    data_list = []
    for tdate in cal_df['calendarDate']:
        A_ticker_list = get_universe(tdate)
        max_date = data.query("date<=@tdate")['date'].max()
        signal_df = data.query("date==@max_date").set_index('ticker')
        signal_df = signal_df.reindex(A_ticker_list).fillna(0).reset_index()
        signal_df['date'] = tdate
        
        data_list.append(signal_df)
    
    all_signal_df = pd.concat(data_list, axis=0)
    return all_signal_df

overweight_list = []
key_list = ['bottom', 'top', 'all']
for key in key_list:
    overweight_list.append(overweight_dict[key].set_index(['ticker', 'reportDate'])['overweight'])

overweight_info = pd.concat(overweight_list, axis=1)
overweight_info.columns = key_list

signal_df = get_overweight_signal(overweight_info.reset_index())
signal_df = signal_df.query("date >= '2011-01-01'")
signal_df = signal_df[['date', 'ticker'] + key_list].sort_values(['date', 'ticker']).reset_index(drop=True)

print "Time cost: %s seconds" % (time.time() - start_time)
print(u'因子格式为')
print(signal_df.head(5).to_html())


'''

2.2 重仓超配因子的测试

构建因子测试方法，测试因子的IC、分组、因子收益率显著情况
分组测试: 月度再平衡，每个月初等权买入重仓超配因子值为1 的股票，构建因子多头组合；等权买入因子值为0 的股票，构建因子空头组合

'''

# ---------------------------  因子测试框架
import copy
import lib.quant_util as qutil
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.style.use('seaborn-white')

def regression_test(factor_df, mret_df, factor_list):
    """
    计算回归法的因子收益率和t值
    参数：
        factor_df: DataFrame, 因子值
        factor_df: list, 因子列表
    返回：
        DataFrame, 因子收益率和t值
    """
    def reg(df, fn):
        y = df['nxt1m_ret']
        x = df[[fn, 'SIZE']+indu_list]
        try:
            res = sm.OLS(y, sm.add_constant(x), missing='drop').fit()
            return [res.params[1], res.tvalues[1]]
        except:
            print fn, df['tradeDate'].unique()
            return [np.nan, np.nan]
    merged_df = factor_df.merge(size_indu_df, on=['ticker', 'tradeDate']).merge(mret_df, on=['ticker', 'tradeDate'])
    merged_df = merged_df.dropna(subset=['nxt1m_ret'])
    # 计算历史因子收益率
    t_value = []
    for fn in factor_list:
        tmp = merged_df.groupby('tradeDate').apply(lambda df: reg(df, fn))
        t_value.append(pd.DataFrame(tmp.values.tolist(), index=tmp.index, columns=['fret', 't_value']))
    t_value = pd.concat(t_value, axis=1)
    t_value.columns = pd.MultiIndex.from_tuples([(fn, col) for fn in factor_list for col in ['fret', 't_value']])
    return t_value

def simple_group_backtest(signal_df, return_df, factor_name, return_name, ngrp=5, commission=0):
    """
    对因子进行简单的分组多头回测。返回各组收益率和累计收益率， 编号越大，因子值越大。
    参数：
        signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一类为股票当日的因子值
        return_df: DataFrame, columns=['ticker', 'tradeDate', [period_return]], 收益率，只含有调仓日，以及下期累计收益率
        factor_name:　str, signal_df中因子值的列名
        return_name： str, return_df中收益率的列名
        ngrp: int, 分组数, 默认为5
        commission: float, 交易费用设置, 卖出时收取，默认不考虑交易费
    返回：
        DataFrame, 列为[’group'， tradeDate', 'period_ret', 'cum_ret'], 返回每期收益率和累计收益率
    """
    bt_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'], how='right')

    # 因子分组
    bt_df.dropna(subset=[factor_name, return_name], inplace=True)
    bt_df['group'] = bt_df[factor_name]

    # 等权
    count_df = bt_df.groupby(['tradeDate', 'group']).apply(lambda x: len(x)).reset_index()
    count_df.columns = ['tradeDate', 'group', 'count']
    bt_df = bt_df.merge(count_df, on=['tradeDate', 'group'])
    bt_df['weight'] = 1.0 / bt_df['count']

    perf = bt_df.groupby(['group', 'tradeDate']).apply(lambda x: sum(x[return_name] * x['weight'])).reset_index()
    perf.columns = ['group', 'tradeDate', 'period_ret']
    if commission > 0:
        # 在卖出时收取交易费用
        adj_df = bt_df.pivot_table(values='weight', index='tradeDate', columns=['group', 'ticker']).fillna(0)
        adj_df1 = adj_df.diff().fillna(0)
        comm = (adj_df1[adj_df1 < 0] * commission).sum(level='group', axis=1).fillna(0)
        comm = comm.stack().reset_index()
        comm.columns = ['tradeDate', 'group', 'cost']
        perf = perf.merge(comm, on=['group', 'tradeDate'])
        perf['period_ret'] = perf['period_ret'] + perf['cost']
    perf.sort_values(['group', 'tradeDate'], inplace=True)
    perf['cum_ret'] = perf.groupby('group')['period_ret'].apply(lambda x: (x + 1).cumprod())

    # 调整时间
    perf['period_ret'] = perf.groupby('group')['period_ret'].shift(1)
    perf['period_ret'].fillna(0, inplace=True)
    perf['cum_ret'] = perf.groupby('group')['cum_ret'].shift(1)
    perf['cum_ret'].fillna(1, inplace=True)

    return perf[['group', 'tradeDate', 'period_ret', 'cum_ret']], bt_df

def factor_test_summary(factor_df, mret_df, factor_list, ngrp=2):
    """
    综合因子测试方法：回归法、IC分析法、分组测试分析法
    参数：
        factor_df: DataFrame, 因子值
        factor_df: list, 因子列表
    返回：
        因子收益率和t值、IC序列、分组收益率序列
    """
    # 回归法测试
    reg_res = regression_test(factor_df, mret_df, factor_list)
    # IC测试
    ic_res = qutil.calc_ic(factor_df, mret_df, factor_list, return_col_name='nxt1m_ret', ic_type='spearman')
    # 分层回测测试
    perf_list = []
    for fn in factor_list:
        perf, _ = simple_group_backtest(factor_df, mret_df, factor_name=fn, return_name='nxt1m_ret', commission=0.0015, ngrp=ngrp)
        perf_list.append(perf.pivot_table(values='period_ret', index='tradeDate', columns='group'))
    perf_df = pd.concat(perf_list, axis=1)
    perf_df.columns = pd.MultiIndex.from_tuples([(fn, col) for fn in factor_list for col in range(ngrp)])
    return reg_res, ic_res, perf_df

def proc_float_scale(df, col_name, format_str):
    """
    格式化输出
    参数：
        df: DataFrame, 需要格式化的数据
        col_name： list, 需要格式化的列名
        format_str： 格式类型
    """
    for col in col_name:
        for index in df.index:
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

def group_perf_describe(perf_df, factor_list, annual_len, ngrp=2):
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

def reg_res_discribe(reg_res, factor_list):
    """
    统计因子回归法分析结果
    参数:
        reg_res: DataFrame, 因子收益率和t值
        factor_df: list, 因子列表
    返回:
        DataFrame, 回归法分析结果统计
    """
    sub_res = reg_res[pd.MultiIndex.from_tuples([(fn, 't_value') for fn in factor_list])]
    tvalues_res = pd.concat([sub_res.abs().mean(), (abs(sub_res)>2).sum() / len(sub_res), (sub_res>2).sum() / len(sub_res), (sub_res<-2).sum() / len(sub_res)], axis=1).reset_index()
    tvalues_res.columns = ['factor_name', 'col_type', 'abs_t_mean', 'abs_t_m2_pct', 'pos_t_m2_pct', 'neg_t_m2_pct']
    tvalues_res = tvalues_res[['factor_name', 'abs_t_mean', 'abs_t_m2_pct', 'pos_t_m2_pct', 'neg_t_m2_pct']].set_index('factor_name')

    res_mean = reg_res.mean().reset_index()
    res_mean.columns = ['factor_name', 'col_type', 'value']
    res_mean = res_mean.pivot_table(values='value', index='factor_name', columns='col_type')

    reg_table = pd.concat([tvalues_res, res_mean], axis=1)
    reg_table.columns = ['|t|均值', '|t|>2占比', 't>2占比', 't<-2占比', '因子收益率均值', 't均值']
    reg_table = proc_float_scale(reg_table, ['|t|>2占比', 't>2占比', 't<-2占比', '因子收益率均值'], ".2%")
    reg_table = proc_float_scale(reg_table, ['|t|均值', 't均值'], ".2f")
    return reg_table.loc[factor_list, :]

def test_discribe(reg_res, ic_res, perf_df, factor_list, index_list):
    """
    综合因子分析结果统计
    参数:
        reg_res: DataFrame, 因子收益率和t值
        ic_res: DataFrame, IC值， index为日期， columns为因子名， values为各个因子的IC值
        perf_df: DataFrame, 回测的期间收益率， index为日期， columns为因子名， values为因子回测的期间收益率
        factor_df: list, 因子列表
    """
    reg_table = reg_res_discribe(reg_res, factor_list)
    reg_table.index = index_list
    ic_table = ic_describe(ic_res, factor_list, annual_len=12)
    ic_table.index = index_list
    group_table = group_perf_describe(perf_df, factor_list, annual_len=12)
    group_table.index = index_list
    print '基金业绩与重仓超配因子回归法结果分析', reg_table.to_html()
    print '基金业绩与重仓超配因子IC结果分析', ic_table.to_html()
    print '基金业绩与重仓超配因子分组回测结果分析', group_table.to_html()

def plot_group_perf(perf_df, factor_name, title_part, ngrp=2):
    """
    展示分组回测收益结果
    参数：
        参数:
        perf_df: DataFrame, 回测的期间收益率， index为日期， columns为因子名， values为因子回测的期间收益率
        factor_name: str, 因子名称
        title_part: str, 图片标题
    """
    fig = plt.figure(figsize=(18, 10))
    
    for i in range(len(title_part)):
        perf_dfk = copy.deepcopy(perf_df)
        perf_dfk = perf_dfk[[(factor_name[i], col) for col in range(ngrp)]]
        perf_dfk.columns = range(ngrp)
        
        k = 220 + i+1
        ax1 = fig.add_subplot(k)
        ax1.plot(pd.to_datetime(perf_dfk.index), (perf_dfk[range(ngrp)]+1).cumprod())
        ax1.legend([u"空头", u"多头"], loc="upper left", shadow=True, fancybox=True, prop=font, fontsize=10)
        ax1.set_ylabel(u'净值', fontsize=12, fontproperties=font)
        ax1.set_title(u'%s二分组回测收益表现' % title_part[i], fontsize=16, fontproperties=font)
        
        perf_dfk[ngrp] = perf_dfk[ngrp-1]-perf_dfk[0]
        ax2 = ax1.twinx()
        ax2.plot(pd.to_datetime(perf_dfk.index), (perf_dfk[ngrp]+1).cumprod(), color='red')
        ax2.legend([u"多空(右轴)"], loc="upper right", shadow=True, fancybox=True, prop=font, fontsize=10)
        ax2.grid(False)
        
    plt.show()
    
    
'''

获取行情数据，并剔除上市不满90个交易日的次新股、st股、停牌个股、一字板个股
同时为了利用回归方式考察因子收益率，加载风险模型的市值与行业因子

'''

# ---------------------------  行情数据处理
start_time = time.time()

# 全A投资域
universe_list = DataAPI.EquGet(equTypeCD=u"A", field=u"secID",pandas="1")['secID'].tolist()
universe_list.remove('DY600018.XSHG')

# 获取月末交易日
cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin_date, endDate=end_date).sort('calendarDate')
cal_dates_df[['calendarDate', 'prevTradeDate']] = cal_dates_df[['calendarDate', 'prevTradeDate']].applymap(lambda x: x.replace('-', ''))

# 获取个股月度收益率
mret_df = DataAPI.MktEqumAdjGet(beginDate=begin_date, endDate=end_date, secID=universe_list, field=u"ticker,endDate,chgPct", pandas="1")
mret_df.rename(columns={'endDate':'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
mret_df['tradeDate'] = mret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
mret_df['nxt1m_ret'] = mret_df.groupby('ticker')['curr_ret'].shift(-1)

# 股票池筛选：上市不满90个交易日的次新股、st股、停牌个股、一字板个股
forbidden_pool = qutil.stock_special_tag(begin_date, end_date, pre_new_length=90)
# 筛选一字板个股
mkt_df = DataAPI.MktEqudGet(beginDate=begin_date, endDate=end_date, secID=universe_list, field=u"ticker,tradeDate,highestPrice,lowestPrice", pandas="1")
mkt_df['tradeDate'] = mkt_df['tradeDate'].apply(lambda x: x.replace('-', ''))
limit_df = mkt_df[(mkt_df['highestPrice'] == mkt_df['lowestPrice']) & (mkt_df['highestPrice']>0)][['ticker', 'tradeDate']]
limit_df['special_flag'] = 'limit'
forbidden_pool = forbidden_pool.append(limit_df)
forbidden_pool = forbidden_pool.merge(cal_dates_df, left_on=['tradeDate'], right_on=['calendarDate'])
forbidden_pool = forbidden_pool[['ticker', 'tradeDate', 'prevTradeDate', 'special_flag']]

# 在月行情数据中剔除个股
mret_df = mret_df.merge(forbidden_pool[['ticker', 'prevTradeDate', 'special_flag']], left_on=['ticker', 'tradeDate'], right_on=['ticker', 'prevTradeDate'], how='left')
mret_df = mret_df[mret_df['special_flag'].isnull()]
mret_df = mret_df.drop(['prevTradeDate', 'special_flag'], axis=1)

print "个股收益率:", mret_df.head().to_html()

# 行业市值数据
tmp = DataAPI.RMExposureDayGet(secID=u"",ticker=u"000001",tradeDate=u"",beginDate=u"20170410",endDate=u"20170410",field=u"",pandas="1")
indu_list = tmp.columns[15:-2].tolist()
size_indu_list = []
for tdate in mret_df['tradeDate'].unique():
    factor_td_df = DataAPI.RMExposureDayGet(tradeDate=tdate,beginDate=u"",endDate=u"",field=['ticker', 'tradeDate' , 'SIZE']+indu_list,pandas="1")
    size_indu_list.append(factor_td_df)
size_indu_df = pd.concat(size_indu_list)
print "行业市值数据:", size_indu_df.head().to_html()

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''

如下表所示:
从截面回归的角度来看，重仓超配因子溢价显著的月度占比高达63.72%，显示重仓超配虚拟变量因子在截面上对个股收益存在显著影响。
从多空分组的角度看，该因子的多头组合相对于空头组合存在年化3.61%的超额收益，月度胜率58.93%，夏普比0.45。即平均来看，基金重仓超配的股票确实存在正向超额收益。但该因子并非在每年都有正超额收益，从多空曲线来看，该组合在14-16表现较差，发生明显回撤。
同时，基于业绩前50%的基金所构建的重仓超配因子，具有更高的IC及多空收益。业绩前50%基金的重仓超配因子年化多空收益为3.63%，信息比0.4；而业绩后50%基金的重仓超配因子年化多空收益为2.50%，信息比0.33。
'''

# ---------------------------  因子测试
start_time = time.time()

factor_list = key_list
name_list = [u'业绩后50%', u'业绩前50%', u'全部基金']
data = signal_df.rename(columns={'date': 'tradeDate'})
data['tradeDate'] = data['tradeDate'].apply(lambda x: x.replace('-', ''))
comp_reg_res, comp_ic_res, comp_perf_df = factor_test_summary(data, mret_df, factor_list)
test_discribe(comp_reg_res, comp_ic_res, comp_perf_df, factor_list, name_list)
plot_group_perf(comp_perf_df, factor_list, name_list)

end_time = time.time()
print "耗时: %s seconds" % (end_time - start_time)


'''

三、基金重仓超配因子及其对指数增强组合的影响
该部分耗时 40分钟
该部分内容包括:

3.1 个股打分模型
3.2 指数增强组合回测
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

3.1 个股打分模型

研报中个股收益采用风格、低频技术因子、基本面因子和高频因子进行预测，本节为了去除alpha因子选取对结果的影响，只考察基金重仓超配因子的使用
采用过去12个月IC的均值符号对当期因子进行赋权，构建最终的alpha因子


''''
# ---------------------------  alpha因子构建
ic_df = comp_ic_res.set_index('tradeDate').shift(1)
ic_mean = ic_df.rolling(12).mean().dropna().applymap(np.sign)

factor_df = pd.merge(data, ic_mean.reset_index(), on='tradeDate').sort_values(['tradeDate', 'ticker'])

for key in key_list:
    factor_df[key] = factor_df[key + "_x"] * factor_df[key + "_y"]
factor_df = factor_df[['tradeDate', 'ticker'] + key_list]

'''
3.2 指数增强组合回测

具体参数如下:

选股池: ZZ800
基准: HS300
时间范围: 20130101-20200630
调仓参数: 月度调仓，买卖交易费千分之1.5
'''

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
import os
import pandas as pd
import numpy as np
import time
import pickle
from CAL.PyCAL import * 
from quartz_extensions.SignalAnalysis.tears import portfolio_construction

start_time = time.time()
# -----------回测参数部分开始，可编辑------------
start = '2013-01-01'                       # 回测起始时间
end = '2020-06-30'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = DynamicUniverse('000906.ZICN')           # 证券池，支持股票和基金
capital_base = 1000000000                     # 起始资金
freq = 'd'                              
refresh_rate = Monthly(1)  
commission = Commission(buycost=0.0015, sellcost=0.0015, unit='perValue')

# 因子值
    
# 把回测参数封装到 SimulationParameters 中，供 quick_backtest 使用
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate)
# 获取回测行情数据
data = quartz.get_backtest_data(sim_params)

backtest_results_dict = {}

factor_list = ['all', 'top', 'bottom']
for method in factor_list:
    factor_data = factor_df[['ticker', 'tradeDate', method]].copy().rename(columns={method: 'value', 'tradeDate': 'date'}).dropna()
    factor_data['value'] = factor_data.groupby('date')['value'].apply(lambda x: (x - x.mean())/x.std())
    factor_data['secID'] = factor_data['ticker'].apply(lambda x: x+'.XSHG' if x[:2] in ['60'] else x+'.XSHE')
    print ('backtesting for factor %s ..................................' % method, time.asctime())
    # 注册一个账户
    accounts = {'fantasy_account': AccountConfig(account_type='security', capital_base=capital_base, commission=commission)}    
    sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate, accounts=accounts)

    def initialize(context):                   # 初始化虚拟账户状态
        pass

    def handle_data(context): 
        previous_date = context.previous_date.strftime('%Y%m%d')
        current_date = context.current_date.strftime('%Y%m%d')
        current_universe = context.get_universe('stock', exclude_halt=True)
        # 因子值
        hist = factor_data[factor_data["date"] == previous_date]
        hist["date"] = current_date
        hist = hist[hist['secID'].isin(current_universe)]
        hist = hist[['ticker', 'date', 'value']].reset_index(drop=True)
        
        # 获取当前账户信息
        account = context.get_account('fantasy_account')   
        current_position = account.get_positions(exclude_halt=True)

        # 持仓转化
        bf_positions = {}
        bf_positions['cash'] = account.cash
        if len(current_position) > 0: 
            for stock, p in current_position.iteritems():
                bf_positions[stock] = current_position[stock]['value']
        else:
            pass
        bf_positions = pd.Series(bf_positions)

        # 单期组合优化
        result = portfolio_construction(hist, 'limit_active_risk', start_date=current_date, end_date=current_date, universe='ZZ800', benchmark='HS300', frequency='month', sector_exposure_lower_boundary=-0.05, sector_exposure_upper_boundary=0.05, factor_exposure_lower_boundary=-1.0, factor_exposure_upper_boundary=1.0, target_risk=0.04, init_holding=bf_positions, asset_lower_boundary=0, asset_upper_boundary=0.1, construct_type='start')
 
        # 每期的成分股权重
        positions = account.get_positions() 
        res = result.get(current_date.replace('-',''))
        opt_weight = pd.Series(res['expected_weight']).to_frame(name='opt_w')

        # 每一期的跟踪误差
        status = res['opt_status']
        tracking_error = res['exante_active_risk']

        wts = pd.Series(res['expected_volume'])
        wts = wts.drop('cash')[wts>0].to_dict()
        if len(wts)>0:
            sell_list = [stk for stk in positions if stk not in wts]
            for stk in sell_list:
                account.order_to(stk,0)
            c = account.portfolio_value
            change = {}
            for stock, w in wts.iteritems():
                p = context.current_price(stock)
                if not np.isnan(p) and p > 0:
                    if stock not in positions:
                        available_amount = 0
                    else:
                        available_amount = positions[stock].available_amount
                    change[stock] = round(w) - available_amount

            for stock in sorted(change, key=change.get):
                if abs(change[stock])>100:
                    account.order(stock, change[stock])

    # 生成策略对象
    strategy = quartz.TradingStrategy(initialize, handle_data)
    # ---------------策略定义结束----------------

    # 开始回测
    bt, perf = quartz.quick_backtest(sim_params, strategy, data=data)

    # 保存运行结果
    backtest_results_dict[method] = {'max_drawdown': perf['max_drawdown'], 'sharpe': perf['sharpe'], 'alpha': perf['alpha'], 'beta': perf['beta'], 'information_ratio': perf['information_ratio'], 'annualized_return': perf['annualized_return'], 'bt': bt[[u'tradeDate', u'portfolio_value', u'benchmark_return']]}  

# 保存该次回测结果为文件
with open(os.path.join(raw_data_dir, 'backtest.pickle'), 'wb') as handle:
    pickle.dump(backtest_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ('Done! Time Cost: %s seconds' % (time.time()-start_time))


'''

下图分析了指数增强组合相较于基准的表现，可以看出三个基金池构建的因子均有超额收益，与第二节业绩靠前50%基金池构建的多空因子表现最好不同的是，在HS300指数增强中，业绩靠后50%基金所构建的重仓超配因子表现最好，其年化收益为3.8%，信息比率为0.51，而业绩前50%基金构建的因子表现最差，其年化收益仅为1.9%，信息比率为0.26
研报解释这个现象主要是由于历史上重仓超配因子并非一直都为alpha 因子，基于业绩后50%基金所构建的重仓超配因子，溢价方向延续比例更高，在根据历史溢价预估因子收益的模型下更为有利

'''

backtest_origin_indic = [u'information_ratio']
backtest_heged_indic = [u'hedged_annualized_return', u'hedged_max_drawdown', u'hedged_volatility']

name_list = [u'业绩后50%', u'业绩前50%', u'全部基金']
def get_tracking_index_result(results):  
    """
    指数增强组合的回测结果展示及分析
    params:
        results: dict, 回测结果
    return:
        DataFrame, 返回计算的指标
    """        
    backtest_pd = pd.DataFrame(index=factor_list, columns=backtest_heged_indic+backtest_origin_indic)
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax1 = ax.twinx()
    ax.grid()
    
    color_list = ['blue', 'black', 'red']
    for number, key in enumerate(factor_list):
        bt = results[key]['bt']
        data = bt[[u'tradeDate', u'portfolio_value', u'benchmark_return']]
        data['portfolio_return'] = data.portfolio_value / data.portfolio_value.shift(1) - 1.0  # 总头寸每日回报率
        data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0] / capital_base - 1.0
        data['excess_return'] = data.portfolio_return - data.benchmark_return  # 总头寸每日超额回报率
        data['excess'] = data.excess_return + 1.0
        data['excess'] = data.excess.cumprod()  # 总头寸对冲指数后的净值序列
        
        running_max = np.maximum.accumulate(data['excess'].values)
        max_drawdown_line = -((running_max - data['excess'].values) / running_max)

        hedged_max_drawdown = max([1 - v / max(1, max(data['excess'][:i + 1])) for i, v in enumerate(data['excess'])])  # 对冲后净值最大回撤
        hedged_volatility = np.std(data['excess_return']) * np.sqrt(252)
        hedged_annualized_return = (data['excess'].values[-1]) ** (252.0 / len(data['excess'])) - 1.0
        
        backtest_pd.loc[key] = np.array([hedged_annualized_return, hedged_max_drawdown, hedged_volatility] + [results[key][item] for item in backtest_origin_indic])
        
        ax.plot(data['tradeDate'], data[['excess']], linewidth=2, color=color_list[number])
        ax1.fill_between(data['tradeDate'].values, 0, max_drawdown_line, color=color_list[number])
   
    backtest_pd.columns = [ u'年化收益', u'最大回撤', u'收益波动率', u'信息比率']
    backtest_pd.index.name = u'不同类型'
    backtest_pd.index = name_list
   
    
    ax.set_ylim(0.2, 1.5)
    ax1.set_ylim(-0.3, 0.3)
    ax.legend(loc=0)
    ax.legend(name_list, loc="upper left", shadow=True, fancybox=True, prop=font, fontsize=10)
    ax.set_ylabel(u"对冲净值（曲线图）", fontproperties=font, fontsize=16)
    ax1.set_ylabel(u"回撤（柱状图）", fontproperties=font, fontsize=16)
    ax.set_title(u"HS300指数增强组合对冲净值走势", fontproperties=font, fontsize=16)
    
    return backtest_pd.astype(float).round(4)

backtest_pd = get_tracking_index_result(backtest_results_dict)
print(backtest_pd.to_html())

