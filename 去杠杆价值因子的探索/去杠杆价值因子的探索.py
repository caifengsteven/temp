# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:57:47 2020

@author: Asus
"""

'''
导读
A. 研究目的：

传统价值因子中分子包括股权与债券的影响，而分母只包括股票市值，二者并不匹配；另一方面，传统价值因子采用的财务项目未区分公司经营性活动、金融性活动的影响。本文利用优矿的财务数据与回测框架，参考国盛证券研报《多因子系列之四-对价值因子的思考和改进》中的研究方法，用以探索去杠杆价值因子的选股作用
B. 研究结论：

相对于传统价值因子，去杠杆价值因子与常见风格因子的相关性更低；同时因为引入了非线性的关系，去杠杆价值因子比直接对杠杆因子中性化的传统因子效果要好

相对于传统价值因子，去杠杆的SP/EP/CFP在IC测试中均有不同幅度的提升：全A域中，SP因子IC从 0.0348 提升至 0.0374，EP从 0.0530 提升至 0.0570，CFP从 0.0354 提升至 0.0367；其多空收益也有大幅提升，去杠杆SP多空组合夏普从1.13提升至1.38，去杠杆EP夏普从1.68提升至2.13，去杠杆CFP夏普比从1.48提升至1.79

C. 文章结构: 本文共分为3个部分，具体如下

一、去杠杆价值因子计算，该部分主要介绍传统价值因子问题，随之进行去杠杆价值因子计算

二、去杠杆价值因子分析，该部分主要进行IC测试，相关性分析、及对杠杆因子中性化的对比分析

三、因子回测，该部分主要对比去杠杆因子的多空组合收益

D. 运行时间说明

一、去杠杆价值因子计算，需要20分钟左右

二、去杠杆价值因子分析，需要5分钟左右

三、因子回测，需要40分钟左右

总耗时65分钟左右

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
一、去杠杆价值因子计算
该部分耗时 20分钟
该部分内容包括:

1.1 传统价值因子的问题探讨
1.2 改进后的去杠杆价值因子
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 传统价值因子的问题探讨

传统价值因子，例如 SP(营业收入/总市值)、EP(净利润/总市值)、BP(净资产/总市值)、CFP(经营性净现金流/总市值)等，分母通常选用为股票市值，其仅包含代表股东权益的净资产的市场价值。但是分子端诸如营业收入，净利润等，其包含的利益同时影响到债权人与股东。
对于负债率高的企业，公司需要分拨利润支付给债权人，从而影响净利润。传统价值因子未考虑公司杠杆率的因素。
企业收入一部分来自于主营业务，一部分来自于投资等其他业务。对于大多数企业，投资者更专注于其主营的收入情况。所以有必要在财报中区分经营性科目及金融性科目。
图片注释

   调试 运行
文档
 代码  策略  文档
下面对三大报表中财务项区分金融性科目及经营性科目。

'''

# ---------------------------  财报基础函数
import pandas as pd
import numpy as np
import time

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

# ---------------------------  区分财报中的金融性科目及经营性科目
def get_oper_fin_asset():
    '''
        计算资产负债表种的金融性资产与经营性资产
    '''
    df = DataAPI.FdmtBSInduGet(beginDate=u"20070101", field=u"",pandas="1")
    df = df[df['ticker'].apply(lambda x: x[0] in ['0' ,'3', '6'])]
    df = df[~(pd.isnull(df['TLiab']) | pd.isnull(df['TAssets']) | pd.isnull(df['TShEquity']))]
    
    # 计算无息流动负债
    df['IntFreeCL'] = df[['AP', 'commisPayable', 'payrollPayable', 'taxesPayable', 'othPayable', 'accruedExp', 'deferRevenue', 'othCL']].fillna(0).sum(axis=1)    
    # 计算带息非流动负债
    df['IntNCL'] = df[['LTBorr', 'bondPayable']].fillna(0).sum(axis=1)    
    # 计算金融资产
    df['FinAsset'] = df[['cashCEquiv', 'tradingFA', 'availForSaleFa', 'htmInvest', 'investRealEstate', 'derivAssets', 'divReceiv', 'intReceiv', 'purResaleFa', 'CAE', 'othCA']].fillna(0).sum(axis=1)   
    # 计算金融负债, 把涉及利息支付的负债均划分为金融负债
    df['FinLiab'] =  df[['TCL', 'IntNCL']].fillna(0.0).sum(axis=1) - df['IntFreeCL'].fillna(0.0)
    # 计算净负债: 金融负债 − 金融资产
    df['NetFinLiab'] = df['FinLiab'] - df['FinAsset']
    
    # 计算经营性净资产
    df['OperAsset'] = df[['TCL', 'IntNCL', 'TShEquity']].fillna(0.0).sum(axis=1) - df[['IntFreeCL', 'FinAsset']].fillna(0.0).sum(axis=1) 
    
    df.loc[:, ['publishDate', 'endDate']] = df.loc[:, ['publishDate', 'endDate']].applymap(lambda x: x.replace('-', ''))
    column_list = ['TLiab', 'TAssets', 'TShEquity', 'IntFreeCL', 'IntNCL', 'FinAsset', 'FinLiab', 'NetFinLiab', 'OperAsset']
    return df[['ticker', 'secShortName', 'endDate', 'publishDate'] + column_list].rename(columns={'endDate': 'end_date', 'publishDate': 'pub_date'})


def get_oper_fin_income():
    '''
        拆分利润表的经营性部分
    '''
    column_list = ['revenue', 'COGS', 'NIncomeAttrP', 'sellExp', 'adminExp', 'finanExp']
    df = DataAPI.FdmtISInduGet(beginDate=u"20070101", field=['ticker', 'secShortName', 'endDate', 'publishDate'] + column_list, pandas="1")
    df = df[df['ticker'].apply(lambda x: x[0] in ['0' ,'3', '6'])]
    # 计算经营性利润
    df['OperIncome'] = df['revenue'] - df['COGS']    
    df.loc[:, ['publishDate', 'endDate']] = df.loc[:, ['publishDate', 'endDate']].applymap(lambda x: x.replace('-', ''))
    return df.rename(columns={'endDate': 'end_date', 'publishDate': 'pub_date'})

def get_oper_cash_flow():
    df = DataAPI.FdmtCFInduGet(beginDate=u"20070101", field=['ticker', 'secShortName', 'endDate', 'publishDate', 'NCFOperateA'],pandas="1")
    df = df[df['ticker'].apply(lambda x: x[0] in ['0' ,'3', '6'])]
    
    df.loc[:, ['publishDate', 'endDate']] = df.loc[:, ['publishDate', 'endDate']].applymap(lambda x: x.replace('-', ''))
    return df.rename(columns={'endDate': 'end_date', 'publishDate': 'pub_date'})

bs_df = get_oper_fin_asset()
is_df = get_oper_fin_income()
cf_df = get_oper_cash_flow()

'''
1.2 改进后的去杠杆价值因子

根据上一节讨论，为了避免公司杠杆的影响，同时剥离公司金融性活动对财报的影响，引入经营性净资产的概念。
经营性净资产=经营性资产−经营性负债=金融负债−金融资产+权益
对于经营性净资产的市场价值方面，权益部分仍可以采用股票市值代替。但对于金融负债、金融资产来说，二者交易不活跃，这里直接采用账目价值代替。
综上所述，对于传统价值因子的改进如下(参考国盛证券):
图片注释


'''

import gevent
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import datetime as dt
import calendar
from dateutil.relativedelta import relativedelta
from quartz_extensions.SignalAnalysis.tears import analyse_return, analyse_monthly_return, analyse_IC, analyse_construction, analyse_general

StyleName = ['BETA', 'MOMENTUM', 'SIZE', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY', 'SIZENL']

IndustryName = ['Bank', 'RealEstate', 'Health', 'Transportation', 'Mining', 'NonFerMetal', 'HouseApp', 'LeiService',
            'MachiEquip', 'BuildDeco', 'CommeTrade', 'CONMAT', 'Auto', 'Textile', 'FoodBever', 'Electronics',
            'Computer', 'LightIndus', 'Utilities', 'Telecom', 'AgriForest', 'CHEM', 'Media', 'IronSteel',
            'NonBankFinan', 'ELECEQP', 'AERODEF', 'Conglomerates']

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
            x = DataAPI.MktEqudGet(tradeDate=tdate, isOpen="",field=u"ticker,tradeDate,marketValue",pandas="1")
            x['tradeDate'] = x['tradeDate'].apply(lambda x: x.replace('-', ''))
            
            # 加载指数信息
            hs300_cons = DataAPI.IdxConsGet(ticker='000300', intoDate=tdate, field='', pandas="1")['consTickerSymbol'].tolist()
            zz500_cons = DataAPI.IdxConsGet(ticker='000905', intoDate=tdate, field='', pandas="1")['consTickerSymbol'].tolist()
            x.loc[x['ticker'].isin(hs300_cons), 'is_hs300'] = 1
            x.loc[x['ticker'].isin(zz500_cons), 'is_zz500'] = 1
            x = x.fillna(0)
            
            # 加载风险模型因子数据
            y = DataAPI.RMExposureDayGet(tradeDate=tdate, field=['ticker', 'tradeDate'] + StyleName + IndustryName)
            
            df = pd.merge(x, y, on=['ticker', 'tradeDate'])
            df = df.rename(columns={'tradeDate': 'date', 'marketValue': 'cap'})
            return df
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                print('error get factor data: ', tdate, e)
                return pd.DataFrame()

def cal_operate_marketvalue(cap_df, bs_df):
    start_date = cap_df['date'].min()
    end_date = cap_df['date'].max()
    latest_bs_df = get_fin_data_latest(bs_df, col_name=['NetFinLiab']).drop('end_date', axis=1)
    fill_bs_df = fin_data_pit2cont(latest_bs_df, start_date, end_date)
    
    # 合并市值，计算经营性市值
    operate_cap_df = pd.merge(fill_bs_df.rename(columns={'pub_date': 'date'}), cap_df, on=['ticker', 'date'])
    operate_cap_df['OperateCap'] = operate_cap_df['NetFinLiab'] + operate_cap_df['cap']
    
    return operate_cap_df
    

def cal_factor(df, cap_df, factor_name, numerator_col_name, denominator_col_name='cap', is_ttm=False):
    df = df.copy()
    if is_ttm:
        df = fin_data_ttm(df, col_name=numerator_col_name)
        
    latest_df = get_fin_data_latest(df, col_name=[numerator_col_name]).drop('end_date', axis=1)
    fill_df = fin_data_pit2cont(latest_df, start_date, end_date)

    # 合并市值
    ep_df = pd.merge(fill_df.rename(columns={'pub_date': 'date'}), cap_df, on=['ticker', 'date'])
    ep_df[factor_name] = ep_df[numerator_col_name] / ep_df[denominator_col_name]
    
    return ep_df[['ticker', 'date', factor_name]]


# ---------------------------------------------- 计算因子
start_time = time.time()
raw_data_dir = './remove_leverage'

start_date = '20071231'
end_date = '20190331'

# 拿到交易日历，得到月末日期
cal_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
monthend_list = cal_df.query('isMonthEnd == 1')

# 取得每个月末日期，所有股票的市值、风险因子、是否属于HS300、ZZ500成分股
print("begin to get stock cap ...", time.asctime())
pool = ThreadPool(processes=16)
date_list = [tdate.replace("-", "") for tdate in monthend_list['calendarDate']]
frame_list = pool.map(get_cap_by_day, date_list)
pool.close()
pool.join()
monthend_stock_info_df = pd.concat(frame_list, axis=0)
monthend_stock_info_df.to_csv(os.path.join(raw_data_dir, 'monthend_stock_info.csv'), chunksize=1000)
# 计算经营性净资产的市场价值
operate_cap_df = cal_operate_marketvalue(monthend_stock_info_df[['ticker', 'date', 'cap']], bs_df)

# 开始计算传统价值因子、及去杠杆的价值因子
print("begin to cal sp_ttm ...", time.asctime())
sp_df = cal_factor(is_df, operate_cap_df, 'sp', 'revenue', 'cap', True)
sp_adjust_df = cal_factor(is_df, operate_cap_df, 'adjust_sp','revenue', 'OperateCap', True)

print("begin to cal ep_ttm ...", time.asctime())
ep_df = cal_factor(is_df, operate_cap_df, 'ep', 'NIncomeAttrP', 'cap', True)
ep_adjust_df = cal_factor(is_df, operate_cap_df, 'adjust_ep','OperIncome', 'OperateCap', True)

print("begin to cal bp ...", time.asctime())
bp_df = cal_factor(bs_df, operate_cap_df, 'bp', 'TShEquity', 'cap', False)
bp_adjust_df = cal_factor(bs_df, operate_cap_df, 'adjust_bp', 'OperAsset', 'OperateCap', False)

print("begin to cal cfp_ttm ...", time.asctime())
cfp_df = cal_factor(cf_df, operate_cap_df, 'cfp', 'NCFOperateA', 'cap', True)
cfp_adjust_df = cal_factor(cf_df, operate_cap_df, 'adjust_cfp', 'NCFOperateA', 'OperateCap', True)

print('Done! Cost time: %s seconds' % (time.time() - start_time))

# 合并上述计算的价值因子
factor_df = reduce(lambda left, right: pd.merge(left, right, on=['date', 'ticker'], how='outer'), [sp_df, sp_adjust_df, ep_df, ep_adjust_df, bp_df, bp_adjust_df, cfp_df, cfp_adjust_df])

factor_df.to_csv(os.path.join(raw_data_dir, 'remove_alpha_value_factor.csv'), chunksize=1000)

print('因子数据格式为')
print(factor_df.head(5).to_html())

'''


该部分对因子进行去极值、填充缺失值、中性化、标准化等操作。
用行业中位数填充空值
用MAD法处理4倍标准差外的异常值
利用优矿neutralize函数做中性化处理，主要是去除行业、市值的影响

'''


#--------------------------------去极值、填充值、中性化-----------------------------------------------
import datetime as dt
from dateutil.relativedelta import relativedelta

alpha_factors = ['sp', 'adjust_sp', 'ep', 'adjust_ep', 'bp', 'adjust_bp', 'cfp', 'adjust_cfp']
industry_df = DataAPI.EquIndustryGet(industryVersionCD=u"010303", field='ticker,intoDate,outDate,industryName1', pandas="1")
industry_df.loc[:, ['intoDate', 'outDate']] = industry_df.loc[:, ['intoDate', 'outDate']].applymap(lambda x: str(x).replace('-', ''))
industry_df = industry_df[~industry_df['industryName1'].isin(['金融服务', '银行', '非银金融'])]

# 获取所有全A股票
equ_df = DataAPI.EquGet(equTypeCD=u"A", listStatusCD=u"", field=['secID', 'ticker', 'listDate', 'delistDate'], pandas="1")
equ_df.loc[:, ['listDate', 'delistDate']] = equ_df.loc[:, ['listDate', 'delistDate']].applymap(lambda x: str(x).replace('-', ''))


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
    按照[dm+4*dm1, dm-4*dm1]进行去极值
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
                cdate_input.loc[:, 'neu_' + a_factor] = standardize(neutralize(sig, target_date=tdate, exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'SIZENL'])) 
                if a_factor == 'ep':
                    cdate_input.loc[:, 'neu_leverage_ep'] = standardize(neutralize(sig, target_date=tdate, exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LIQUIDTY', 'SIZENL']))                     
                break
            except Exception as e:
                cnt += 1
                if cnt >= 3:
                    break
    neu_alpha_factors = ['neu_' + a_factor for a_factor in alpha_factors] + ['neu_leverage_ep']
    return cdate_input.reset_index()[['ticker', 'date'] + alpha_factors + neu_alpha_factors]

if __name__ == "__main__":
    start_time = time.time()

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
    factor_csv.to_csv(os.path.join(raw_data_dir, 'neu_remove_alpha_value_factor.csv'), chunksize=1000)
    end_time = time.time()
    print ("Time cost: %s seconds" % (end_time - start_time))
    print(u'因子格式为: ')
    print(factor_csv.head(5).to_html())
    
    
'''

二、去杠杆价值因子测试
该部分耗时 5分钟左右
本节主要对比传统价值因子与去杠杆价值因子的IC表现；及直接对传统价值因子进行杠杆因子中性化的分析
2.1 因子IC测试
2.2 与风格因子相关性分析
2.3 对杠杆因子中性化的分析
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
2.1 因子IC测试

本节比较原始与去杠杆后的价值因子IC表现，并考察其在不同域的情况。

'''

import pandas as pd
import numpy as np
import scipy.stats as st
from quartz_extensions.SignalAnalysis.tears import analyse_return, analyse_monthly_return, analyse_IC, analyse_construction, analyse_general

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
        lambda x: pd.Series(data=x[factor_name + [ret_name]].corr(method=method).values[-1, :-1], index=factor_name)).dropna()
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

start_date = '20081231'
end_date = '20190331'

# 获得月收益率
month_return = DataAPI.MktEqumAdjGet(beginDate=start_date, endDate=end_date, field=u"ticker,endDate,chgPct,",pandas="1")
month_return.rename(columns={'endDate': 'tradeDate', 'chgPct': 'month_return'}, inplace=True)
month_return.sort_values(['ticker', 'tradeDate'], inplace=True)
month_return['next_month_return'] = month_return.groupby('ticker')['month_return'].shift(-1)
month_return.dropna(inplace=True)
month_return['tradeDate'] = month_return['tradeDate'].apply(lambda x : x.replace("-", ''))

raw_data_dir = './remove_leverage'
# 加载价值因子
factor_df = pd.read_csv(os.path.join(raw_data_dir, 'neu_remove_alpha_value_factor.csv'), index_col=0, dtype={'ticker': str, 'date': str})
method_list = ['sp', 'adjust_sp', 'ep', 'adjust_ep', 'bp', 'adjust_bp', 'cfp', 'adjust_cfp']
neu_method_list = ['neu_' + item for item in method_list]

monthend_stock_info_df = pd.read_csv(os.path.join(raw_data_dir, 'monthend_stock_info.csv'), index_col=0, dtype={'ticker': str, 'date': str})
cons_df = monthend_stock_info_df[['ticker', 'date', 'is_hs300', 'is_zz500']]

# 合并收益、因子数据，计算IC
ic_list = []
for universe in ['hs300', 'zz500', 'A']:
    if universe == 'A':
        signal_df = factor_df.rename(columns={'date': 'tradeDate'})
    else:
        column = 'is_' + universe
        this_cons_df = cons_df.query('%s==1' % column)
        this_data_df = pd.merge(factor_df, this_cons_df, on=['ticker', 'date'])
        signal_df = this_data_df.rename(columns={'date': 'tradeDate'})
    
    ic_df = calc_ic(signal_df, month_return, method_list, 'next_month_return', method='spearman')
    neu_ic_df = calc_ic(signal_df, month_return, neu_method_list, 'next_month_return', method='spearman')
    ic_desc = ic_describe(ic_df)
    neu_ic_desc = ic_describe(neu_ic_df)

    neu_ic_desc.index = [item.replace('neu_', '') for item in neu_ic_desc.index]
    neu_ic_desc.columns = [u'中性化' + item for item in neu_ic_desc.columns]

    final_ic = pd.concat([ic_desc[[u'平均IC', u'年化IC_IR']].T, neu_ic_desc[[u'中性化平均IC', u'中性化年化IC_IR']].T]).round(4)
    final_ic.columns = [item.replace('adjust_', '去杠杆') for item in final_ic.columns]
    
    ic_list.append(final_ic)

ic_df = pd.concat(ic_list, axis=1, keys=['hs300', 'zz500', 'A'])
for key in ['sp', 'ep', 'bp', 'cfp']:
    this_df = pd.concat([ic_df.xs(key, level=1, axis=1), ic_df.xs('去杠杆'+key, level=1, axis=1)], axis=1, keys=[key, '去杠杆'+key])
    print('%s 分域 rankIC 情况 ' % key)
    print(this_df.to_html())

'''
从上述结果可知，除了BP因子外，SP、EP及CFP均有不同幅度提升。且经过去杠杆后，未经过中性化的因子同样有不错表现。
具体的说，全A域中，SP因子IC从 0.0348 提升至 0.0374，EP从 0.0530 提升至 0.0570，CFP从 0.0354 提升至 0.0367。
   调试 运行
文档
 代码  策略  文档
2.2 与风格因子相关性分析

考察去杠杆前后价值因子与风险模型风格因子的相关性

'''

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')

StyleName = ['BETA', 'MOMENTUM', 'SIZE', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY', 'SIZENL']

def _plot_corr_heatmap(df, factor_name1, factor_name2):
    fig, ax = plt.subplots(figsize=((20, 10)))
    corr_df = df.groupby('FACTOR').apply(lambda x: x[factor_name1].mean()).reindex(factor_name2)
    corr_df = (100 * corr_df).round(2)
    _ = sns.heatmap(corr_df, alpha=1.0, annot=True, center=0.0, annot_kws={"size": 12},
                    linecolor='white', linewidth=0.02, ax=ax, cmap='RdYlGn_r')

def cal_corr(df, factors, StyleName):
    trade_date = df.iloc[0]['date']
    style_industry_corr_df = pd.concat([df[StyleName], df[factors]], axis=1, keys=['df1', 'df2']).corr().loc['df1']['df2']
    style_industry_corr_df.index.name = 'FACTOR'
    style_industry_corr_df.reset_index(inplace=True)
    style_industry_corr_df['TRADE_DATE'] = trade_date
    
    return style_industry_corr_df

factors = ['sp', 'adjust_sp', 'ep', 'adjust_ep', 'bp', 'adjust_bp', 'cfp', 'adjust_cfp']
data = pd.merge(factor_df, monthend_stock_info_df, on=['ticker', 'date'])
df = data.groupby('date').apply(lambda x: cal_corr(x, factors, StyleName))
_plot_corr_heatmap(df, factors, StyleName)


'''

从上述结果中可以看出，与传统价值因子相比，去杠杆价值因子在大多数情况下与风格因子的相关性更低一些。
   调试 运行
文档
 代码  策略  文档
2.3 对杠杆因子中性化的分析

本小节考察原始价值因子直接对杠杆因子进行中性化的效果，对比其与去杠杆价值因子的分组回测

'''

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
import seaborn as sns
sns.set_style('white')
from CAL.PyCAL import *    # CAL.PyCAL中包含font
from scipy.stats.mstats import gmean
import time

def signal_grouping(signal_df, factor_name, ngrp):
    """
    因子分组， 每天根据因子值将股票进行等分，编号0 ~ ngrp-1, 编号越大， 因子值越大
    params:
            signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一类为股票当日的因子值
            factor_name:　str, signal_df中因子值的列名
            ngrp: int, 分组组数
    return:
            DataFrame, signal_df在原本的基础上增加一列'group', 记录每日分组
    """
    signal_df_tmp = signal_df.copy()
    signal_df_tmp.sort_values(factor_name, inplace=True)
    signal_df_tmp.dropna(subset=[factor_name], inplace=True)
    signal_df_tmp['group'] = signal_df_tmp.groupby('tradeDate')[factor_name].apply(
        lambda x: (x.rank() - 1) / len(x) * ngrp).astype(int)
    return signal_df_tmp

def cal_quantile_return(signal_df, return_df, factor_name, return_name, direction=1, n_quantile=10):
    """
    分组回测， 根据因子值将个股等分成给定组数，进行回测
    根据调仓频率，进行交易，返回最后的累计超额收益率。
    params:
            signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一列为股票当日的因子值
            return_df: DataFrame, columns=['ticker', 'tradeDate', [next_period_return]], 收益率，只含有调仓日，以及下期累计收益率
            factor_name:　str, signal_df中因子值的列名 
            return_name： str, return_df中收益率的列名
            direction： {1,-1}, 操作方向， 1为正向操作， -1为反向操作， 默认为1
    return:
            DataFrame, columns=['tradeDate', 'cum_ret'], 返回累计超额收益率
    """
    bt_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'])

    # 分祖
    bt_df.dropna(subset=[factor_name], inplace=True)
    bt_df = signal_grouping(bt_df, factor_name=factor_name, ngrp=n_quantile)

    # 计算权重：每组等权    
    count_df = bt_df.groupby(['tradeDate', 'group']).apply(lambda x: len(x)).reset_index()
    count_df.columns = ['tradeDate', 'group', 'count']
    bt_df = bt_df.merge(count_df, on=['tradeDate', 'group'])
    bt_df['weight'] = 1.0 / bt_df['count']

    # 如果direction=1, 则做多因子值最大的一组， 做空因子值最小的一组；如果direction=-1, 则做空因子值最大的一组， 做多因子值最小的一组
    bt_df['longshort'] = 0.0
    bt_df.loc[bt_df['group']==n_quantile-1, 'longshort'] = bt_df.loc[bt_df['group']==n_quantile-1, 'weight'] * direction
    bt_df.loc[bt_df['group']==0, 'longshort'] = bt_df.loc[bt_df['group']==0, 'weight'] * (-direction)

    longshort_perf = bt_df.groupby('tradeDate').apply(lambda x: np.sum(x[return_name] * x['longshort'])).reset_index()
    longshort_perf.columns = ['tradeDate', 'longshort']

    # 统计每组的超额收益率
    group_pref = bt_df.groupby(['tradeDate', 'group']).apply(lambda x: np.sum(x[return_name] * x['weight'])).reset_index()
    group_pref.columns = ['tradeDate', 'group', 'ret']
    market_pref = bt_df.groupby(['tradeDate']).apply(lambda x: np.sum(x[return_name] * x['weight'])/np.sum(x['weight'])).reset_index()
    market_pref.columns = ['tradeDate', 'market_ret']
    merge_pref = pd.merge(group_pref, market_pref, on='tradeDate')
    merge_pref['ret'] = merge_pref['ret'] - merge_pref['market_ret']
    merge_pref = merge_pref[['tradeDate', 'group', 'ret']]

    group_ret = pd.pivot_table(merge_pref, index='tradeDate', values='ret', columns='group').reset_index()
    group_ret.columns = ['tradeDate'] + ['group%s'%(item+1) for item in range(n_quantile)]

    all_ret = pd.merge(group_ret, longshort_perf, on='tradeDate')
    all_ret.sort_values('tradeDate', inplace=True)

    return all_ret.set_index('tradeDate')

def perf_describe(perf_df):
    """
    统计因子的回测绩效， 包括年化超额收益率、年化波动率、信息比率、最大回撤
    params:
            perf_df: DataFrame, 回测的期间超额收益率， index为日期， columns为因子名， values为因子回测的期间收益率
    return:
            DataFrame, 返回回测绩效
    """
    # 记录因子个数和因子名
    factor_name = perf_df.columns.values
    n = len(factor_name)

    # 年化超额收益率
    ret_mean = pd.Series(index=factor_name, data=gmean(perf_df+1.)**12 - 1.)
    # 年化波动率
    ret_std = perf_df.std() * np.sqrt(12.0)
    # 年化IR
    sr = ret_mean / ret_std
    # 最大回撤
    maxdrawdown = {}
    for i in range(n):
        fname = factor_name[i]
        cum_ret = pd.DataFrame((perf_df[fname] + 1).cumprod())
        cum_max = cum_ret.cummax()
        maxdrawdown[fname] = ((cum_max - cum_ret) / cum_max).max().values[0]
    maxdrawdown = pd.Series(maxdrawdown)
    # 月度胜率
    win_ret = (perf_df > 0).sum() / len(perf_df)

    perf_table = pd.DataFrame([ret_mean, sr, win_ret], index=[u'年化超额收益率', u'信息比率', u'月度胜率']).T

    perf_table = perf_table.ix[factor_name]

    return perf_table

def plot_quantile_excess_return(perf, title):
    # 因子分组的超额收益作图
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    
    x = range(len(perf))
    perf.plot(kind='bar', ax=ax1, legend=False)
    plt.legend(perf.columns, prop=font, loc='best', handlelength=4, handletextpad=1, borderpad=0.5, ncol=2)

    ax1.set_ylabel(u'超额收益', fontproperties=font, fontsize=16)
    ax1.set_xlabel(u'分位组', fontproperties=font, fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels([int(x)+1 for x in ax1.get_xticks()], fontproperties=font, fontsize=14)
    ax1.set_yticklabels([str(x * 100) + '0%' for x in ax1.get_yticks()], fontproperties=font, fontsize=14)
    ax1.set_title(title, fontproperties=font, fontsize=16)
    ax1.grid()
    plt.show()
    
    
n_quantile = 10
quantile_res_list = []
return_name_list = []
method_dict = {'neu_adjust_ep': u'去杠杆EP因子', 'neu_leverage_ep': u'对传统EP做风险模型杠杆因子中性化后的因子'}
for factor_name in ['neu_adjust_ep', 'neu_leverage_ep']:
    signal_df = factor_df[['ticker', 'date', factor_name]].rename(columns={'date': 'tradeDate'})
    
    var_ret = cal_quantile_return(signal_df, month_return, factor_name, 'next_month_return', -1)
    var_perf = perf_describe(var_ret)
    var_perf.columns = [column + '_' + method_dict[factor_name] for column in var_perf.columns]

    quantile_res_list.append(var_perf)
    return_name_list.append(u'年化超额收益率_' + method_dict[factor_name])

var_perf = pd.concat(quantile_res_list, axis=1).round(4)
plot_quantile_excess_return(var_perf.iloc[:n_quantile][return_name_list], u'因子分组超额收益(相对于全市场)')
print('\n因子分组超额收益统计')
print(var_perf.to_html())

'''
上述分组回测可知，对传统价值因子直接对杠杆因子中性化的分组效果并不单调，而去杠杆价值因子有更好的相关性。这是因为在计算因子过程中，对传统因子的分子分母均进行了处理，并不是简单的线性关系。
   调试 运行
文档
 代码  策略  文档
三、因子回测分析
该部分耗时 40分钟左右
本节对中性化后的价值因子进行回测，对比去杠杆前后的因子表现。

因为本章节读取的历史数据过多，需要占用很多资源，建议该章节运行前重启环境释放已占资源，上述因子结果进行了存储，重启不会影响后续章节运行

重启研究环境的步骤为：

网页版：先点击左上角的“Notebook”图标，然后点击左下角的“内存占用x%”图标，在弹框中点击重启研究环境
客户端：点击左下角的“内存x%”, 在弹框中点击重启研究环境
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 组合回测

测试了组合形式: long-short多空组合,具体参数如下:

选股池: A
时间范围: 20090101-20190331
调仓参数: 月度调仓，买卖交易费千分之1.5

'''


import os
import pandas as pd
import numpy as np
import time
import pickle
from CAL.PyCAL import * 

start_time = time.time()
# -----------回测参数部分开始，可编辑------------
start = '2008-12-31'                       # 回测起始时间
end = '2019-03-31'                         # 回测结束时间
benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('A')           # 证券池，支持股票和基金
capital_base = 1000000000.0                     # 起始资金
freq = 'd'                              
refresh_rate = Monthly(1)  
commission = Commission(buycost=0.0015, sellcost=0.0015, unit='perValue')

raw_data_dir = './remove_leverage'
# 运行结果保存pickle的位置
factor_df = pd.read_csv(os.path.join(raw_data_dir, 'neu_remove_alpha_value_factor.csv'), index_col=0, dtype={'ticker': str, 'date': str})
method_list = ['sp', 'adjust_sp', 'ep', 'adjust_ep', 'bp', 'adjust_bp', 'cfp', 'adjust_cfp']
neu_method_list = ['neu_' + item for item in method_list] 

# 把回测参数封装到 SimulationParameters 中，供 quick_backtest 使用
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate)
# 获取回测行情数据
data = quartz.get_backtest_data(sim_params)

backtest_results_dict = {}

# 对4种不同类型分别进行测试
for method in neu_method_list:
    factor_data = factor_df[['ticker', 'date', method]].rename(columns={'date': 'tradeDate', method: 'alpha'})
    factor_data = factor_data.set_index('tradeDate')
    factor_data['ticker'] = factor_data['ticker'].apply(lambda x: x+'.XSHG' if x[:2] in ['60'] else x+'.XSHE')
    
    q_dates = factor_data.index.unique()
    
    for portfolio_type in ['long', 'short']: 
        print ('backtesting for method %s portfolio %s ..................................' % (method, portfolio_type), time.asctime())
        
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
                wts, optimal = optimize_rhac(q['alpha'], pre_date, benchmark=benchmark)
                if not optimal:
                    print('td:{0}, rhac not solved, holdings would not change.'.format(pre_date))
                    return
                wts = wts['optimal_weights'].to_dict()
                my_univ = wts.keys()
                    
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
with open(os.path.join(raw_data_dir, 'remove_leverage_backtest.pickle'), 'wb') as handle:
    pickle.dump(backtest_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ('Done! Time Cost: %s seconds' % (time.time()-start_time))


'''

3.2 long-short组合分析

该小节分析了Top20% - Bottom20%的多空组合走势

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

capital_base = 1000000000.0
with open(os.path.join(raw_data_dir, 'remove_leverage_backtest.pickle'), 'rb') as fHandler:
    backtest_results_dict = pickle.load(fHandler)

backtest_heged_indic = [u'年复合收益', u'夏普比率', u'最大回撤', u'收益波动率']

def get_long_short_result(results):  
    """
    多空组合回测结果展示及分析
    params:
        results: dict, 回测结果
    return:
        DataFrame, 返回计算的指标
    """        
    method_list = ['sp', 'adjust_sp', 'ep', 'adjust_ep', 'bp', 'adjust_bp', 'cfp', 'adjust_cfp']

    backtest_pd = pd.DataFrame(index=method_list, columns=backtest_heged_indic)
    color_list = ['grey', 'red']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    for k, key in enumerate(method_list):   
        data_list = []
        for portfolio in ['long', 'short']:
            data = results["neu_%s_%s"%(key, portfolio)]['bt'][[u'tradeDate', u'portfolio_value']]
            data['portfolio_return'] = data.portfolio_value / data.portfolio_value.shift(1) - 1.0  # 总头寸每日回报率
            data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0] / capital_base - 1.0
            data_list.append(data.set_index('tradeDate')['portfolio_return'])
        long_short_return = (data_list[0] - data_list[1]).fillna(0.0)
        long_short_value = (long_short_return + 1.0).cumprod()
        long_short_value.index = pd.to_datetime(long_short_value.index)
        
        hedged_max_drawdown = max([1 - v / max(1, max(long_short_value[:i + 1])) for i, v in enumerate(long_short_value)])  # 对冲后净值最大回撤
        hedged_volatility = np.std(long_short_return) * np.sqrt(252)
        hedged_annualized_return = (long_short_value.values[-1]) ** (252.0 / len(long_short_value)) - 1.0
        sharpe_ratio = hedged_annualized_return / hedged_volatility
        backtest_pd.loc[key] = [hedged_annualized_return, sharpe_ratio, hedged_max_drawdown, hedged_volatility]
        ax = axes[k/2/2][k/2%2]
        ax.plot(long_short_value.index, long_short_value.values, color=color_list[k%2])
        
        if k % 2 == 0:
            ax.set_title(u"%s 因子多空组合净值走势" % key, fontproperties=font, fontsize=16)
            ax.set_ylabel(u"净值", fontproperties=font, fontsize=16)
    

    return backtest_pd.astype(float).round(4)

backtest_pd = get_long_short_result(backtest_results_dict)
print(backtest_pd.to_html())


