# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:39:19 2020

@author: Asus
"""


'''
基于IVX回归的因子择时研究
导读
A. 研究目的：

由于金融时序数据中常见的内生性和持续性问题，导致OLS方法估计有偏，且统计失效。本文利用优矿的因子及宏观数据，参考东方证券研报《因子择时》，对研报的结果进行了实证分析，采用IVX回归方法来检验宏观、市场指标与因子收益率的显著性关系
B. 研究结论：

本文利用2007.01-2019.06数据进行实证，发现在样本区间内，大部分因子均能找到显著性的指标，且OLS方法与IVX方法结果有较大差异

利用样本外区间数据实证，发现规模、反转及ILLIQ因子的IVX模型均显著优于移动平均预测模型，其中规模因子预测的方向准确率最高，能达到70%

行业和市值风险中性化处理后，大多数因子的有效预测次数变多，方向准确率也有所提高，其中反转因子的可预测性最强

C. 文章结构: 本文共分为3个部分，具体如下

一、数据准备，利用API调取因子及收益数据，并做相关处理；同时构建市场、与宏观指标

二、宏观、市场指标与因子收益率的显著性检验

三、显著性关系的样本外表现，该部分主要考察样本内显著的关系在样本外能否延续，并考察了风险因子中性化对显著性关系的影响

D. 运行时间说明

一、数据准备，需要6分钟左右

二、宏观、市场指标与因子收益率的显著性检验，需要1分钟左右

三、显著性关系的样本外表现，需要3分钟左右

总耗时10分钟左右

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
一、数据准备
该部分耗时 6分钟
该部分内容为：

因子数据准备

市场、宏观指标准备

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 因子数据准备

该部分内容为：

获取uqer因子、下月收益数据

对因子进行去极值、填缺失值、标准化及中性化操作

   调试 运行
文档
 代码  策略  文档
1.1.1 获取uqer因子、下月收益数据

获取因子数据包括:

本文从规模、反转、动量、换手率、估值、盈利、成长及ILLIQ等8类风格类别中共选取了16个基础alpha因子。单个因子等权合成成大类因子 图片注释


'''


# ---------------------------  因子数据读取

import pandas as pd
import numpy as np
import os
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
#from CAL.PyCAL import * 
import gevent
import datetime as dt
from dateutil.relativedelta import relativedelta

raw_data_dir = "./factor_timing"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

#定义需要获取的因子
origin_alpha_factors = ['LCAP', 'PB', 'PE', 'PS', 'PCF', 'ROE', 'TotalAssetsTRate', 'ROIC', 'NetProfitGrowRate', 'OperatingRevenueGrowRate', 'SUE', 'Variance20', 'HSIGMA', 'VOL20', 'REVS20', 'ILLIQUIDITY']

# 起始时间
start_date = '2007-01-01'
end_date = '2019-06-01'

# 获取所有全A股票
equ_df = DataAPI.EquGet(equTypeCD=u"A", listStatusCD=u"", field=['secID', 'ticker', 'listDate', 'delistDate'], pandas="1")
equ_df['listDate'] = equ_df['listDate'].apply(str)
equ_df['delistDate'] = equ_df['delistDate'].apply(str)

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
    st_ticker = set(DataAPI.SecSTGet(beginDate=list_date_need, endDate=format_date, pandas="1")['ticker'])
    
    return A_ticker - st_ticker

def get_factor_by_day(tdate):
    '''
    获取给定日期的因子信息
    参数： 
        tdate, 时间，格式%Y-%m-%d
    返回:
        DataFrame, 返回给定日期的因子值
    '''
    cnt = 0
    while True:
        try:
            A_ticker = get_universe(tdate)
            # 利用DataAPI查询因子
            x = DataAPI.MktStockFactorsOneDayProGet(tradeDate=tdate, field=['ticker', 'tradeDate'] + origin_alpha_factors, pandas="1")
            # 转换方向
            x['BP'] = 1.0 / x['PB']
            x['EP'] = 1.0 / x['PE']
            x['SP'] = 1.0 / x['PS']
            x['CFP'] = 1.0 / x['PCF']
            x = x.drop(['PB', 'PE', 'PS', 'PCF'], axis=1)
            x = x[x['ticker'].isin(A_ticker)]
            
            # 查询行业信息
            industry = DataAPI.EquIndustryGet(industryVersionCD=u"010303", intoDate=tdate,field='ticker,industryName1',pandas="1").dropna()
            x = pd.merge(x, industry, on='ticker')
            
            cap = DataAPI.MktEqudGet(tradeDate=tdate, isOpen="",field=u"ticker,tradeDate,negMarketValue,marketValue",pandas="1")
            x = pd.merge(x, cap, on=['ticker', 'tradeDate'])
            x['tradeDate'] = x['tradeDate'].apply(lambda x: x.replace('-', ''))

            x = x.dropna(thresh=int(len(x.columns)*0.4))
            return x
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                print('error get factor data: ', tdate, e)
                return None

if __name__ == "__main__":
    start_time = time.time()

    # 拿到交易日历，得到月末日期
    trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
    trade_date = trade_date[trade_date.isMonthEnd == 1]

    print("begin to get factor value for each stock...")
    # # 取得每个月末日期，所有股票的因子值
    pool = ThreadPool(processes=16)
    date_list = [tdate.replace("-", "") for tdate in trade_date.calendarDate.values]
    frame_list = pool.map(get_factor_by_day, date_list)
    pool.close()
    pool.join()

    factor_df = pd.concat(frame_list, axis=0)
    ########################## 取得个股的行情数据 ################################
    print("\nbegin to get price ratio for stocks and index ...")
    # 个股绝对涨幅
    chgframe = DataAPI.MktEqumAdjGet(beginDate=start_date, endDate=end_date, field=['ticker', 'endDate', 'return'], pandas="1")
    chgframe['endDate'] = chgframe['endDate'].apply(lambda x: x.replace("-", ""))

    ################################ 对齐数据 ################################
    print("begin to align data ...")
    # 得到月度关系
    month_frame = trade_date[['calendarDate', 'isOpen']]
    month_frame['prev_month_end'] = month_frame['calendarDate'].shift(1)
    month_frame = month_frame[['prev_month_end', 'calendarDate']]
    month_frame.columns = ['month_end', 'next_month_end']
    month_frame.dropna(inplace=True)
    month_frame['month_end'] = month_frame['month_end'].apply(lambda x: x.replace("-", ""))
    month_frame['next_month_end'] = month_frame['next_month_end'].apply(lambda x: x.replace("-", ""))

    # 对齐月度关系
    factor_frame = factor_df.merge(month_frame, left_on=['tradeDate'], right_on=['month_end'], how='left')

    # 得到个股下个月的涨幅数据
    factor_frame = factor_frame.merge(chgframe, left_on=['ticker', 'next_month_end'], right_on=['ticker', 'endDate'])
    del factor_frame['month_end']
    del factor_frame['endDate']
    
    end_time = time.time()
    print ("Time cost: %s seconds" % (end_time - start_time))
    
    
'''

1.1.2 对因子进行去极值、填缺失值、标准化及中性化操作

本章节对上一小节的数据进行相关处理

用MAD法处理4倍标准差外的异常值
用行业中位数填充空值
对因子及收益实现市值及行业的中性化
单个因子等权合成大类因子


'''


# ---------------------------  因子数据处理

alpha_factors = ['LCAP', 'BP', 'EP', 'SP', 'CFP', 'ROE', 'TotalAssetsTRate', 'ROIC', 'NetProfitGrowRate', 'OperatingRevenueGrowRate', 'SUE', 'Variance20', 'HSIGMA', 'VOL20', 'REVS20', 'ILLIQUIDITY']

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

    upper = dm + 4 * dm1
    lower = dm - 4 * dm1
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
    cdate_input = data[data['tradeDate'] == tdate]
    cdate_input = cdate_input.set_index('ticker')
        
    cdate_input.loc[:, alpha_factors] = cdate_input.loc[:, alpha_factors].apply(lambda x : winsorize_by_date(x))
    cdate_input.loc[:, alpha_factors] = cdate_input.loc[:, alpha_factors].fillna(cdate_input.groupby('industryName1')[alpha_factors].transform("median"))
    cdate_input = cdate_input.fillna(0.0)
    cdate_input.loc[:, alpha_factors] = cdate_input.loc[:, alpha_factors].apply(lambda x : standardize(x))
    
    cdate_input['Value'] = standardize(cdate_input[['BP', 'EP', 'SP', 'CFP']].mean(axis=1))
    cdate_input['Profit'] = standardize(cdate_input[['ROE', 'TotalAssetsTRate', 'ROIC']].mean(axis=1))
    cdate_input['Growth'] = standardize(cdate_input[['NetProfitGrowRate', 'OperatingRevenueGrowRate', 'SUE']].mean(axis=1))
    cdate_input['Volaitility'] = standardize(cdate_input[['Variance20', 'HSIGMA']].mean(axis=1))
    cdate_input = cdate_input.rename(columns={'VOL20': 'Turnover', 'REVS20': 'Reversal', 'ILLIQUIDITY': 'ILLIQ', 'LCAP': 'LogMV', 'BP': 'BTOP'})

    cdate_input.drop(list(set(alpha_factors) & set(cdate_input.columns)), axis=1, inplace=True)
    new_factors = ['Value', 'Profit', 'Growth', 'Volaitility', 'Turnover', 'Reversal', 'ILLIQ', 'LogMV', 'BTOP']
    
    for a_factor in new_factors:
        sig = cdate_input[a_factor]
        cnt = 0
        while True:
            try:
                if a_factor == 'LogMV':
                    cdate_input.loc[:, 'neu_' + a_factor] = standardize(neutralize(sig, target_date=tdate, exclude_style_list=['SIZE', 'BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'SIZENL']))    
                else:
                    cdate_input.loc[:, 'neu_' + a_factor] = standardize(neutralize(sig, target_date=tdate, exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'SIZENL']))   
                break
            except Exception as e:
                cnt += 1
                if cnt >= 3:
                    break

    return cdate_input

if __name__ == "__main__":
    start_time = time.time()
    
    # 遍历每个月末日期，利用协程对因子及收益进行去极值、标准化，中性化处理
    print('standardize & neutralize & winsorize factor...')
    date_list = factor_frame['tradeDate'].unique()
    jobs = [gevent.spawn(standardize_winsorize_neutralize_factor, value) for value in zip([factor_frame]*len(date_list), date_list)]
    gevent.joinall(jobs)
    new_frame_list = [e.value for e in jobs]
    print ("ALL FINISHED")

    factor_df = pd.concat(new_frame_list, axis=0)
    factor_df.reset_index(inplace=True)
    factor_df = factor_df.rename(columns={'return':'next_return'})

    # ################################ 数据存储下来 ################################
    factor_df.to_csv(os.path.join(raw_data_dir, 'factor.csv'), chunksize=1000)
    
    print('因子数据格式为')
    print(factor_df.head(5).to_html())
    end_time = time.time()
    print ("Time cost: %s seconds" % (end_time - start_time))
    
    
'''

1.2 市场、宏观指标准备

该部分内容为：

市场指标准备

宏观指标准备

涉及指标参考东方证券研报如下:
图片注释

1.2.1 市场指标准备

构建市场EP、市场BP、市场换手率、市场波动率、资金敏感度、ValueSpread、上期因子收益率
因子收益率利用因子top 10%组合与bottom 10%组合的收益率构建


'''
# ---------------------------  基础函数
import pandas as pd
import numpy as np
import time

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
def fin_data_pit2cont(pit_data_frame, sdate, edate, fill_days=None):  
    """
    将PIT数据转成连续数据
    pit_data_frame: 财务报表数据, column= ['ticker','pub_date',[fin_value]], index=num, pub_date='%Y%m%d'
    sdate: 起始时间, '%Y%m%d'
    edate: 终止时间, '%Y%m%d'
    fill_days: fillna最长回溯时间
    返回：
         连续日的因子值dataframe, 列为：['ticker','pub_date',[fin_value]]
    """

    trade_date_frame = DataAPI.TradeCalGet(exchangeCD=u"XSHE", beginDate=sdate, endDate=edate,
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
    tmp_frame = tmp_frame[(tmp_frame.pub_date >= sdate.replace('-', '')) & (tmp_frame.pub_date <= edate.replace('-', ''))]
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

def cal_market_info(data, factors, ngrp=10):
    '''
        计算市场指标，包括市场EP、市场BP、市场换手率、资金敏感度、各个因子的ValueSpread、各个因子的上期因子收益率
    '''
    res_dict = {}
    ep_info = data[['marketValue', 'NIncomeAttrP']].dropna()
    bp_info = data[['marketValue', 'TEquityAttrP']].dropna()
    turnover_info = data[['negMarketValue','Turnover']].dropna()
    illiq_info = data[['negMarketValue','ILLIQ']].dropna()
    
    res_dict['mkt_ep'] = ep_info['NIncomeAttrP'].sum() / ep_info['marketValue'].sum()
    res_dict['mkt_bp'] = bp_info['TEquityAttrP'].sum() / bp_info['marketValue'].sum()
    res_dict['turnover'] = (turnover_info['negMarketValue'] * turnover_info['Turnover']).sum() / turnover_info['negMarketValue'].sum()
    res_dict['illiq'] = (illiq_info['negMarketValue'] * illiq_info['ILLIQ']).sum() / illiq_info['negMarketValue'].sum()
    
    for factor in factors + ['neu_%s' % item for item in factors]:
        factor_info = data[[factor, 'BTOP', 'next_return']].dropna(subset=[factor])
        factor_info.sort_values(factor, inplace=True)
        factor_info['group'] = ((factor_info[factor].rank(method='first') - 1) / len(factor_info) * ngrp).astype(int)
        res_dict['%s_next_ret' % factor] = factor_info.query('group==%s' % (ngrp-1))['next_return'].mean() - factor_info.query('group==0')['next_return'].mean()
        res_dict['%s_value_spread' % factor] = factor_info.query('group==%s' % (ngrp-1))['BTOP'].median() / factor_info.query('group==0')['BTOP'].median()
    
    return pd.Series(res_dict)

# ---------------------------  读取财报数据
start_time = time.time()

# 读取PIT净利润表数据，计算市场EP用
income_df = DataAPI.FdmtISGet(beginYear='2005', reportType=['Q1', 'S1', 'CQ3', 'A'], field=[u'ticker', u'publishDate', u'endDate', u'NIncomeAttrP'], pandas="1").rename(columns={'publishDate': 'pub_date', 'endDate': 'end_date'})
income_df = income_df[income_df['ticker'].apply(lambda x: x[0] in ['0' ,'3', '6'])]
income_df[['pub_date', 'end_date']] = income_df[['pub_date', 'end_date']].applymap(lambda x: x.replace('-', ''))
ttm_income_df = fin_data_ttm(income_df, col_name=u'NIncomeAttrP')
latest_ttm_income_df = get_fin_data_latest(ttm_income_df, col_name=[u'NIncomeAttrP']).drop('end_date', axis=1)
fill_ttm_income_df = fin_data_pit2cont(latest_ttm_income_df, start_date, end_date)

# 读取PIT资产负债表数据，计算市场BP用
bs_df = DataAPI.FdmtBSGet(beginYear='2006', field=[u'ticker', u'publishDate', u'endDate', u'TEquityAttrP'], pandas="1").rename(columns={'publishDate': 'pub_date', 'endDate': 'end_date'})
bs_df = bs_df[bs_df['ticker'].apply(lambda x: x[0] in ['0' ,'3', '6'])]
bs_df[['pub_date', 'end_date']] = bs_df[['pub_date', 'end_date']].applymap(lambda x: x.replace('-', ''))
latest_bs_df = get_fin_data_latest(bs_df, col_name=['TEquityAttrP'])
fill_bs_df = fin_data_pit2cont(latest_bs_df, start_date, end_date).drop('end_date', axis=1)

# 合并上述财务数据
fdmt_df = pd.merge(fill_ttm_income_df, fill_bs_df, on=['ticker', 'pub_date'], how='outer').rename(columns={'pub_date': 'tradeDate'})

print('数据格式为')
print(fdmt_df.head(5).to_html())
end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


# ---------------------------  计算市场指标
start_time = time.time()

# 合并因子数据、及财务数据
alpha_fdmt_df = pd.merge(factor_df, fdmt_df, on=['ticker', 'tradeDate'])
alpha_factors = ['LogMV', 'Value', 'Profit', 'Growth', 'Volaitility', 'Turnover', 'Reversal', 'ILLIQ']

mkt_df = alpha_fdmt_df.groupby('tradeDate').apply(lambda x: cal_market_info(x, alpha_factors))

# 当期因子收益率
ret_columns = ['%s_ret' % factor for factor in alpha_factors] + ['neu_%s_ret' % item for item in alpha_factors]
next_ret_columns = ['%s_next_ret' % factor for factor in alpha_factors] + ['neu_%s_next_ret' % item for item in alpha_factors]
mkt_df[ret_columns] = mkt_df[next_ret_columns].shift(1)

# 利用中证全指的收益来计算市场波动率
mkt_vol = DataAPI.MktIdxdGet(ticker=u"000985",beginDate=u"20060101",endDate=u"20190601",field=u"tradeDate,CHGPct",pandas="1").rolling(window=63).std().dropna()
mkt_vol.columns = ['tradeDate', 'mkt_vol']
mkt_vol['tradeDate'] = mkt_vol['tradeDate'].apply(lambda x: x.replace('-', ''))
mkt_vol.set_index('tradeDate', inplace=True)
mkt_df['mkt_vol'] = mkt_vol

print('数据格式为')
print(mkt_df.head(5).to_html())
end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))


'''


1.2.2 宏观数据准备

本文选取的指标参考了研报中的指标, 如下所示
图片注释

此外，东方证券研报中未考虑宏观数据实际公布时点滞后问题；本小节从实际出发，依据宏观数据的发布时间来获取实际已发布数据，避免引入未来数据

'''


# ---------------------------  获取宏观数据，并处理成月末形式
start_time = time.time()

# 优矿中的宏观因子ID
indic_ids = '1040000050,1040000702,1070000007,1070000009,1020000004,1070003530,1090000035,1090001387,1090001399,1090002003'

# 指标ID和指标名称对齐
indic_data_df = DataAPI.EcoDataProGet(indic_ids, start_date, end_date, field=['indicID','publishDate', 'periodDate', 'dataValue'])
indic_name_df = DataAPI.EcoInfoProGet(indic_ids, field=['indicID','indicName', 'frequency'],pandas="1")
indic_data_df = indic_data_df.merge(indic_name_df, on=['indicID'], how='left')
indic_data_df.drop_duplicates(subset=['indicID', 'periodDate'], inplace=True)
indic_data_df['publishDate'] = indic_data_df['publishDate'].fillna(indic_data_df['periodDate'])
indic_data_df['publishDate'] = indic_data_df['publishDate'].apply(lambda x: x[:10])
indic_data_df = indic_data_df.sort_values(['indicID', 'publishDate'])

indic_data_df['dataValue'] = indic_data_df['dataValue'] / 100.0
# 替换国家外汇储备值为环比
indic_data_df.loc[indic_data_df['indicID']==1070003530, 'dataValue'] = indic_data_df.loc[indic_data_df['indicID']==1070003530, :]['dataValue'].pct_change()

# 获取月末的宏观数据
cal_df =  DataAPI.TradeCalGet(exchangeCD=u"XSHE", beginDate=start_date, endDate=end_date, field='calendarDate,prevTradeDate,isOpen')
monthend_list = [item.strftime("%Y-%m-%d") for item in pd.date_range(start_date, end_date, freq='1M')]
cal_df = cal_df[cal_df['calendarDate'].isin(monthend_list)]
cal_df['prevTradeDate'] = np.where(cal_df['isOpen']==1, cal_df['calendarDate'], cal_df['prevTradeDate'])

data = indic_data_df.groupby('indicID').apply(lambda x: pd.merge(x, cal_df.rename(columns={"calendarDate": 'publishDate'}), on='publishDate', how='outer'))
del data['indicID']
data.reset_index(inplace=True)
del data['level_1']

# 简单宏观数据处理
data = data.sort_values(by=['indicID', 'publishDate'], ascending=True)
data['dataValue'] = data.groupby(['indicID'])['dataValue'].transform(lambda x: x.fillna(method='pad'))
data = data[['indicID', 'prevTradeDate', 'dataValue']].dropna()
data = pd.pivot_table(data, index='prevTradeDate', columns='indicID', values='dataValue')
data[u'M2-M1'] = data[1070000009] - data[1070000007]
data[u'期限利差'] = data[1090001399] - data[1090001387]
data[u'信用利差'] = data[1090002003] - data[1090001399]
data = data.rename(columns={1040000050: u'CPI', 1040000702: u'PPI', 1020000004: u'工业增加值变动', 1070003530: u'外汇储备变动', 1090000035: u'存款准备金率', 1090001387: u'利率'})
data = data.drop([1070000007, 1070000009, 1090001399, 1090002003], axis=1).dropna()
data.index = [item.replace('-', '') for item in data.index]

print('宏观数据格式为')
print(data.head(5).to_html())
end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))

'''
- 合并上述市场指标、宏观指标数据，构建成最终预测因子收益率的需要数据
'''

# ---------------------------  合并宏观及市场指标
indic_df = pd.concat([mkt_df, data], axis=1).dropna()
indic_df.index.name = 'tradeDate'
indic_df = indic_df.reset_index()
indic_df.to_csv(os.path.join(raw_data_dir, 'indic_info.csv'), chunksize=1000, encoding='utf-8')


'''

二、宏观、市场指标与因子收益率的显著性检验
该部分耗时 1分钟左右
本节主要检测能否利用宏观、市场指标来预测上述因子的收益率。常规检测显著性关系的方法是OLS线性回归，但在时间序列预测时，该方法会产生较大误差，主要原因是金融时序数据有内生性问题及近似非平稳性问题，导致参数估计不准确
研报采取了Kostakis, Magdalinos and Stamatogiannis (2015)提出的IVX方法进行回归检测，使用了工具变量的方法，可以应对回归中的内生性问题
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
东方研报测试了1998年至今的样本区间，因优矿数据时间限制，本小节只考察了2007年至今的情况
在整个样本区间内，拿单个因子对单个指标做回归，考察哪个指标在样本内对下月因子收益率有显著预测作用

'''


# ---------------------------  IVX函数定义
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import os
import time
import math

def ivx_estimation(y, x, k=1):
    '''
        IVX回归，参考Robust Econometric Inference for Stock Return Predictability_Kostakis, Magdalinos and Stamatogiannis (2015)
    '''
    row, col = x.shape
    xlag = x[:row-1, :]
    xt = x[1:, :]
    yt = y[1:, :]

    nn, l = xlag.shape
    df = [l, nn - l]

    # join_horiz to include intercept
    Xols = np.insert(xlag, 0, 1, axis=1)
    Aols, _, _, _ = np.linalg.lstsq(Xols, yt)
    epshat = yt - Xols.dot(Aols)
    s2 = np.sum(np.power(epshat, 2)) / (nn - l)
    std_err = np.sqrt(s2 * np.diag(np.linalg.pinv(Xols.T.dot(Xols))).reshape(-1, 1))
    tstat = Aols / std_err

    Rn = np.zeros((l, l))
    for i in range(l):
        Rn[i, i] = np.asscalar(np.linalg.lstsq(xlag[:, [i]], xt[:, [i]])[0])

    # autoregressive residual estimation
    u = xt - xlag.dot(Rn)
    #residuals correlation matrix
    corrmat = np.corrcoef(epshat, u, rowvar=False)
    # covariance matrix estimation (predictive regression)
    covepshat = epshat.T.dot(epshat) / nn

    # covariance matrix estimation(autoregression)
    covu = np.zeros((l, l))
    for i in range(nn):
        covu += u[[i], :].T.dot(u[[i], :])
    covu = covu / nn

    # covariance matrix between 'epshat' and 'u'
    covuhat = np.zeros((1, l))
    for i in range(l):
        covuhat[0, i] = np.sum(epshat * u[:, [i]], axis=0)
    covuhat = covuhat.T / nn

    m = int(math.floor(nn ** 0.3333333))
    uu = np.zeros((l, l))
    for h in range(1, m+1):
        a = np.zeros((l, l))
        for t in range(h, nn):
            a += u[[t], :].T.dot(u[[t-h], :])
        con = 1.0 - h / (1.0 + m)
        uu += con * a
    uu = uu / nn
    Omegauu = covu + uu + uu.T

    q = np.zeros((m, l))
    for h in range(1, m+1):
        p = np.zeros((nn-h, l))
        for t in range(h, nn):
            p[[t-h], :] = u[[t], :] * epshat[[t-h], :] 
        con = 1.0 - h / (1.0 + m)
        q[[h-1], :] = con * np.sum(p, axis=0)
    residue = np.sum(q, axis=0) / nn
    Omegaeu = covuhat + residue.reshape(-1, 1)

    # instrument construction
    Rz = (1 - 1.0 / (nn ** 0.95)) * np.eye(l)
    diffx = xt - xlag
    z = np.zeros((nn, l))
    z[0, :] = diffx[0, :]
    for i in range(1, nn):
        z[[i], :] = z[[i-1], :].dot(Rz) + diffx[[i], :]

    n = nn - k + 1
    Z = np.insert(z[:n-1, :], 0, 0, axis=0)
    zz = np.insert(z[:nn-1, :], 0, 0, axis=0)
    ZK = np.zeros((n, l))
    for i in range(n):
        ZK[[i], :] = np.sum(zz[i: i+k, :], axis=0)
    meanzK = np.mean(ZK, axis=0).reshape(1, -1)

    yy = np.zeros((n, 1))
    for i in range(n):
        yy[[i], :] = np.sum(yt[i: i+k, :], axis=0)
    Yt = yy - np.mean(yy, axis=0)

    xK = np.zeros((n, l))
    for i in range(n):
        value = np.sum(xlag[i: i+k, :], axis=0)
        xK[[i], :] = value
    Xt = xK - np.mean(xK, axis=0)

    Aivx = Yt.T.dot(Z).dot(np.linalg.pinv(Xt.T.dot(Z)))
    fitted = Xt.dot(Aivx.T)
    residuals = Yt - fitted

    FM = np.asscalar(covepshat - Omegaeu.T.dot(np.linalg.inv(Omegauu)).dot(Omegaeu))
    M = ZK.T.dot(ZK) * np.asscalar(covepshat) - n * meanzK.T.dot(meanzK) * FM
    H = np.eye(l)
    Q = H.dot(np.linalg.pinv(Z.T.dot(Xt))).dot(M).dot(np.linalg.pinv(Xt.T.dot(Z))).dot(H.T)

    wivx = (H.dot(Aivx.T)).T.dot(np.linalg.pinv(Q)).dot(H.dot(Aivx.T))
    wivxind_z = Aivx / np.sqrt(np.diag(Q)).reshape(1, -1)
    wivxind = wivxind_z.T ** 2

    p_value_ivx = stats.chi2.sf(wivxind, 1)
    pv_waldjoint = stats.chi2.sf(wivx, df[0])

    return {"Params IVX": Aivx.T,
            "Join Wald T": wivx,
            "Join Wald P": pv_waldjoint,
            "Ind Wald T": wivxind,
            "Ind Wald P": p_value_ivx,
            "IVX MEAN X" : np.mean(xK, axis=0),
            "IVX MEAN Y" : np.mean(yy, axis=0),
            "Params OLS": Aols,
            "OLS T": tstat}



# ---------------------------  OLS与IVX样本内检测显著性关系

start_time = time.time()

columns = [u'市场EP', u'市场BP', 'CPI', 'PPI', 'M2-M1', u'工业增加值变动', u'外汇储备变动', u'存款准备金率', u'市场换手率', u'市场波动率', u'资金敏感度', 'ValueSpread', '上期因子收益率', u'利率', u'期限利差', u'信用利差']
df_ols = pd.DataFrame(index=alpha_factors, columns=columns)
df_ivx = pd.DataFrame(index=alpha_factors, columns=columns)

# 对每个因子分别利用OLS与IVX检测
for factor in alpha_factors:
    y_col = '%s_ret' % factor
    x_col = ['mkt_ep', 'mkt_bp', 'CPI', 'PPI', 'M2-M1', u'工业增加值变动', u'外汇储备变动', u'存款准备金率', 'turnover', 'mkt_vol', 
            'illiq', '%s_value_spread' % factor, '%s_ret' % factor, u'利率', u'期限利差', u'信用利差']
    
    # OLS方法
    res_ols = indic_df[x_col].apply(lambda x: sm.OLS(indic_df[y_col].values[1:], sm.add_constant(x.values[:-1], has_constant='add')).fit())
    p_value = res_ols.apply(lambda x: x.pvalues[1])
    params = res_ols.apply(lambda x: x.params[1]) 
    filter_p = np.where(p_value>0.1, 'nan', np.where(p_value>0.05, '*', np.where(p_value>0.01, '**', '***')))
    df_ols.loc[factor] = ['%.3f%s'%(params[i], p) if p != 'nan' else '' for i, p in enumerate(filter_p)] 
    
    # IVX方法
    res_ivx = indic_df[x_col].apply(lambda x: ivx_estimation(indic_df[y_col].values.reshape(-1, 1), x.values.reshape(-1, 1))) 
    p_value = res_ivx.apply(lambda x: np.asscalar(x['Ind Wald P']))    
    params = res_ivx.apply(lambda x: np.asscalar(x['Params IVX']))    
    filter_p = np.where(p_value>0.1, 'nan', np.where(p_value>0.05, '*', np.where(p_value>0.01, '**', '***')))
    df_ivx.loc[factor] = ['%.3f%s'%(params[i], p) if p != 'nan' else '' for i, p in enumerate(filter_p)] 

print('****** OLS显著性检查结果 2007.03-2019.05')
print(df_ols.to_html())

print('\n****** IVX显著性检查结果 2007.03-2019.05')
print(df_ivx.to_html())

end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time) )


'''
上述结果中，回归系数后的一颗星、两颗星、三颗星分别代表回归系数在10%、5%和1%置信度下显著不等于0，不显著的系数没有列出
OLS及IVX方法结果有显著差别，比如“资金敏感度”指标在OLS方法下呈现一定的预测作用，但使用IVX方法，该指标无显著预测作用
IVX方法中，除去盈利、成长因子，其余因子在样本内均有显著的可用预测指标
预测指标中，PPI、市场波动率的作用最明显，对多个因子的下月收益率有显著预测作用；其中外汇储备变动、ValueSpread、利率等指标预测作用不明显
   调试 运行
文档
 代码  策略  文档
三、显著性关系的样本外表现
该部分耗时 3分钟
本节主要考察样本内显著的关系在样本外能否延续，主要内容为：

滚动测试样本外表现

规模因子的时序结果变化

风险中性化的影响

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 滚动测试样本外表现

上一节的因子显著关系是在统计样本内做的，而投资者更关注样本内有效的指标是否在样本外同样有效
本节参考东方研报，采用了滚动窗口的检测方法来考察样本外表现
滚动测试步骤如下:

Step 1. 每个月月初，基于过去100个月的数据，用单个因子的因子收益率对单个预测指标做一元IVX回归，看哪个指标显著（10%置信度）
Step 2. 如果有多个指标显著，则把它们合在一起做多元 IVX 回归，联合检验整个回归方程是否显著。如果显著，则代入最新的预测指标数据，预测下一期因子收益率
Step 3. 如此滚动向前，重复前两步，统计预测结果的准确性

'''

# ---------------------------  滚动测试的基础函数

from CAL.PyCAL import *    # CAL.PyCAL中包含font
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import lines

def choose_significant_indic(data, choose_p=0.05):
    '''
       对每个因子，挑选能显著预测的指标
    '''
    choose_indic_dict = {}
    for factor in alpha_factors + ['neu_%s' % item for item in alpha_factors]:
        y_col = '%s_ret' % factor
        x_col = ['mkt_ep', 'mkt_bp', 'CPI', 'PPI', 'M2-M1', u'工业增加值变动', u'外汇储备变动', u'存款准备金率', 'turnover', 'mkt_vol', 
                'illiq', '%s_value_spread' % factor, '%s_ret' % factor, u'利率', u'期限利差', u'信用利差']

        res = data[x_col].apply(lambda x: ivx_estimation(data[y_col].values.reshape(-1, 1), x.values.reshape(-1, 1))['Ind Wald P'].T[0][0])
        choose_indic_dict[factor] = res[res <= choose_p].index.tolist()
    return choose_indic_dict

def rolling_predict(x, choose_p=0.05, k=1):
    '''
        滚动考察样本外表现
    '''
    train_data = x.iloc[:-1]
    choose_indic_dict = choose_significant_indic(train_data, choose_p)
    
    predict_dict = {}
    for factor, choose_indic in choose_indic_dict.iteritems():
        if len(choose_indic) == 0:
            continue
        y_col = '%s_ret' % factor
        
        res = ivx_estimation(train_data[y_col].values.reshape(-1, 1), train_data[choose_indic].values.reshape(-1, len(choose_indic)))
        if res['Join Wald P'][0][0] <= choose_p:
            predict_x = np.sum(x.iloc[-k:][choose_indic].values, axis=0) - res['IVX MEAN X']
            predict_dict[factor] = np.asscalar(predict_x.dot(res['Params IVX']) + res['IVX MEAN Y'])
    choose_indic_df = pd.DataFrame.from_dict(choose_indic_dict, orient='index').stack().reset_index(level=0)
    choose_indic_df.columns = ['factor', 'indic']
    return pd.Series(predict_dict), choose_indic_df

def clark_west_test(y1, y2, y):
    '''
        CW检测，y1与y2均为预测值，y为真实值，判断y1是否比y2预测的更好
    '''
    e1 = (y - y1) ** 2
    e2 = (y - y2) ** 2
    e3 = (y1 - y2) ** 2
    f_hat = e1 - e2 + e3
    
    t_value = np.sqrt(len(f_hat)) * np.mean(f_hat) / np.std(f_hat)
    p_value = stats.norm.sf(t_value)
    return t_value, p_value

def polt_time_series_effective_indic(choose_indic_df, factor):
    '''
        画图，有效指标随着时间的变化图
    '''
    df = choose_indic_df.query('factor=="%s"' % factor).reset_index(drop=True)
    uniques, X = np.unique(df['indic'], return_inverse=True)
    df['indic_num'] = X
    df = df.set_index(['date', 'indic'])['indic_num'].unstack(level=1)
    ax = df.plot(kind='line', marker="o", ylim=[X.min() - 0.5, X.max() + 0.5], legend=False)
    ax.legend(df.columns.tolist(), prop=font, fontsize=16)
    ax.set_title(u"因子有效预测指标变化", fontproperties=font, fontsize=16)
    ax.xaxis.label.set_visible(False)
    plt.show()
    
def plot_time_series_sign_accuracy(df, factor):
    '''
        画图，预测方向准确性随着时间的变化图
    '''
    data = df[['ivx_%s_ret' % factor, '%s_ret' % factor, '%s_next_ret' % factor]].copy()
    data.columns = ['ivx', 'rolling', 'true']
    acc_df = data.apply(lambda x: np.sign(x['ivx']) == np.sign(x['true']) if not pd.isnull(x['ivx'])  else 2, axis=1).astype(int).reset_index()
    acc_df.columns = ['date', 'acc']
    color_map = {0: 'green', 1: 'red', 2: 'grey'}
    acc_df['acc_col'] = acc_df['acc'].apply(lambda x: color_map[x])
    acc_df['acc'] = 1

    ax = acc_df[['date', 'acc']].set_index('date').plot(kind='bar', color=acc_df['acc_col'].tolist(), rot=45, ylim=[-0.1, 1.2], legend=False, width=1, figsize=(10, 4))
    ax.xaxis.label.set_visible(False)
    ax.get_yaxis().set_ticks([])

    n = 5
    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::n])
    ax.xaxis.set_ticklabels(ticklabels[::n])

    line1 = lines.Line2D(range(10), range(10), linewidth=10, color="grey")
    line2 = lines.Line2D(range(10), range(10), linewidth=10, color="green")
    line3 = lines.Line2D(range(10), range(10), linewidth=10, color="red")
    plt.legend((line1,line2,line3), (u'无预测', u'错误预测', u'正确预测'), ncol=3,  prop=font, fontsize=16) 
    ax.set_title(u"IVX 回归模型预测因子收益率的方向准确性" , fontproperties=font, fontsize=16)

    plt.show()
    
'''

上述第一步骤中，可能在该窗口中没有发现显著的预测指标，此时无法利用模型进行预测，相反的，如果发现了显著预测指标，称本次预测为有效预测
对于这些有效预测，统计如下的两个指标来衡量模型准确度:
方向准确性，即模型预测的涨跌与真实涨跌方向是否一致的比例
MSE，即均方误差
同时，引入了移动平均值作为参考,具体来说利用因子过去24个月的因子收益率均值作为下期因子收益率的预测。构建样本外R-squared指标：
Ros=1−MSEivx/MSE均值
如果 Ros>0， 说明IVX回归模型更准确，反之则说明滚动均值更准确

'''

# ---------------------------  滚动测试的结果展示

start_time = time.time()

ret_columns = ['%s_ret' % item for item in alpha_factors] + ['neu_%s_ret' % item for item in alpha_factors]
next_ret_columns = ['%s_next_ret' % item for item in alpha_factors] + ['neu_%s_next_ret' % item for item in alpha_factors]

# 过去24个月的均值
rolling_mean_ret_df = indic_df.set_index('tradeDate')[ret_columns].rolling(24).mean()
next_ret_df = indic_df.set_index('tradeDate')[next_ret_columns]

min_step = 100
predict_return_df = pd.DataFrame(index=indic_df['tradeDate'][min_step:], columns=alpha_factors + ['neu_%s' % item for item in alpha_factors])
choose_indic_df = pd.DataFrame()
for i in range(min_step+1, len(indic_df)):
    this_data = indic_df.iloc[:i]
    this_date = this_data.iloc[-1]['tradeDate']
    predict_ret, choose_indic = rolling_predict(this_data, 0.1)
    predict_return_df.loc[this_date] = predict_ret
    choose_indic['date'] = this_date
    choose_indic_df = pd.concat([choose_indic_df, choose_indic])

predict_return_df.columns = ['ivx_%s_ret' % item for item in predict_return_df.columns]
df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), [predict_return_df, rolling_mean_ret_df, next_ret_df])

predict_perf_df = pd.DataFrame(index=alpha_factors, columns=[u'有效预测次数', u'有效预测占比', u'方向准确率', u'MSE_IVX', u'MSE_滚动均值', 'Ros', 'CW-pVal'])
neu_predict_perf_df = pd.DataFrame(index=['neu_%s' % item for item in alpha_factors], columns=[u'有效预测次数', u'有效预测占比', u'方向准确率', u'MSE_IVX', u'MSE_滚动均值', 'Ros', 'CW-pVal'])

for factor in alpha_factors + ['neu_%s' % item for item in alpha_factors]:
    data = df[['ivx_%s_ret' % factor, '%s_ret' % factor, '%s_next_ret' % factor]].copy()
    data.columns = ['ivx', 'rolling', 'true']
    effective_num = len(data.dropna())
    effective_pert = effective_num * 1.0 / len(data)

    effective_data = data.dropna()
    if len(effective_data) == 0:
        continue
    sign_accuracy_pert = (effective_data['ivx'].apply(lambda x: np.sign(x)) == effective_data['true'].apply(lambda x: np.sign(x))).sum() * 1.0 / len(effective_data)
    mse_ivx  = np.mean(np.power(effective_data['ivx'] - effective_data['true'], 2))
    mse_rolling = np.mean(np.power(effective_data['rolling'] - effective_data['true'], 2))
    Ros = 1 - mse_ivx / mse_rolling
    t_value, p_value = clark_west_test(effective_data['rolling'], effective_data['ivx'], effective_data['true'])
    if factor in predict_perf_df.index:
        predict_perf_df.loc[factor] = [effective_num, effective_pert, sign_accuracy_pert, mse_ivx, mse_rolling, Ros, p_value]
    else:
        neu_predict_perf_df.loc[factor] = [effective_num, effective_pert, sign_accuracy_pert, mse_ivx, mse_rolling, Ros, p_value]

print(u'样本外滚动预测结果(2015.07-2019.04)')
print(predict_perf_df.dropna().to_html())
end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time) )


'''
上述表格最后一列代表着CW检验的P值，比较的是IVX模型与滚动均值预测的准确性，原假设是回归模型不比滚动均值模型准确
有效预测次数占比最多的是换手率因子，次数达到45次，但查看样本外R2，不如滚动均值表现好，波动率因子有类似表现
IVX回归模型比滚动均值模型表现好的因子有规模、价值、盈利、反转及ILLIQ，其中达到5%显著性的有规模、反转及ILLIQ因子，其中规模因子预测的方向准确率最高，能达到70%
   调试 运行
文档
 代码  策略  文档
3.2 规模因子的时序结果变化

   调试 运行
文档
 代码  策略  文档
以规模因子为例，查看有效预测指标随时间的变化情况，下图所知，市场波动率是一个长期有效的预测指标，存款准备金率与工业增加值同比变动也曾阶段性有效
目前预测规模因子收益率有效的指标是市场波动率及PPI

'''

polt_time_series_effective_indic(choose_indic_df, 'LogMV')
plot_time_series_sign_accuracy(df, 'LogMV')


'''
3.3 风险中性化的影响

上述章节考察的全部是原始因子收益率的预测效果，有时需要关注风险中性化的因子收益率。该节测试了对因子进行市值+行业中性化的因子效果
下表结果可知，除价值因子外，对大多数因子来说，中性化后的有效预测次数变多，方向准确率也有所提高，预测效果同样有提升

'''

print(u'样本外滚动预测结果(2015.07-2019.04)')
print(neu_predict_perf_df.dropna().drop('neu_LogMV').to_html())


polt_time_series_effective_indic(choose_indic_df, 'neu_Reversal')

plot_time_series_sign_accuracy(df, 'neu_Reversal')

