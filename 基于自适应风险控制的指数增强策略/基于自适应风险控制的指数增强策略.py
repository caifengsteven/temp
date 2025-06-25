# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:48:14 2020

@author: Asus
"""

'''
导读
A.研究目的：指数增强基金力求在对基准指数跟踪的同时实现超额收益，在市场经历了2017年市值、反转等因子的大幅波动及回撤，通过组合优化来构建指数增强组合的方式受到了越来越多的关注。组合优化模型的最大优势是可以进行灵活的风险控制，能够在最大化组合收益的同时满足一系列的风险控制约束条件，使得组合能够在跟踪基准指数的基础上实现稳定超额。收益预测模型和风险控制模型是组合优化模型的两个重要的构成部分，本文参考天风证券：《基于自适应风险控制的指数增强策略》、天风证券：《因子正交全攻略，理论、框架与实践》，我们将两篇研报中的因子模型与风险控制方法结合起来，通过一种自适应控制跟踪误差的方法在A股构建指数增强策略进行实证，实现在不同风格指数上的稳健超额收益。

B.研究结论：实证表明，通过优矿平台及研报模型进行实证，我们结合收益预测模型、风险控制模型、自适应风险控制方法下的策略能在各种市场风格下获取稳定的超额收益，其中，

自适应风险控制下的沪深300指数增强组合从2010年初回测至2018年8月底，年化超额收益8.23%，相对最大回撤4.01%，收益回撤比2.05，夏普比率2.48，跟踪误差3.31%；
自适应风险控制下的中证500指数增强组合从2010年初回测至2018年8月底，年化超额收益13.23%，相对最大回撤3.23%，收益回撤比4.10，夏普比率3.99，跟踪误差3.32%。
C.文章结构：本文共分为3个部分，具体如下

一、数据准备及处理：这部分主要包括股票池的界定，所用选股因子的构造、预处理等

二、收益预测模型：该部分主要是利用对称正交方法、因子权重反向归零以及ICIR加权方法对因子进行加权，得到复合因子，作为对股票的收益预测

三、风险控制模型与组合优化：该部分主要是根据收益预测模型及一系列风险约束通过组合优化方法获得每期组合权重并用优矿平台进行回测

D.运行时间说明

一、数据准备及处理，需要50分钟左右

二、收益预测模型，需要5分钟左右

三、风险控制模型与组合优化，需要60分钟左右

 注意事项 

 第三部分涉及回测及多进程优化，消耗资源较大，需要重启研究环境以释放资源

之前的数据都进行了存储，第三部分的代码可以直接运行而不需要重跑第一、二部分的代码

重启研究环境的步骤为：

网页版：先点击左上角的“Notebook”图标，然后点击左下角的“内存占用x%”图标，在弹框中点击重启研究环境
客户端：点击左下角的“内存x%”, 在弹框中点击重启研究环境
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
第一部分：数据准备及处理
该部分耗时 50分钟(主要是因子的构造环节需要40分钟，其他合计10分钟)
该部分内容为：

每期股票池的选取（剔除ST及上市不满半年的新股）

选股因子的构造，对因子的预处理（去极值、中性化、标准化）

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 每期股票池的选取

生成的股票池文件存储在enhance_strategy_data/stock_pool.csv

股票池属性为date，code，区间为日期为（20080102 - 20180903）之间的每月月初

'''


# coding: utf-8

import numpy as np
import pandas as pd
import datetime
import time
import os
import copy
import cvxpy as cvx
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
import seaborn as sns
from multiprocessing import Pool
import cPickle as pickle
from CAL.PyCAL import *    # CAL.PyCAL中包含font
universe = set_universe('A')
cal = Calendar('China.SSE')

# 时间格式转变函数
def time_change(x):
    y = datetime.datetime.strptime(x, '%Y-%m-%d')
    y = y.strftime('%Y%m%d')
    return y

# 获取回测区间的交易日、月末以及月初时间
def get_trade_list(start_date, end_date):
    """
    Args:
        start_date: 时间区间起点
        end_date: 时间区间终点
    Returns: 
        trade_list: 时间区间内的交易日
        month_end: 月末时间
        month_start: 月初时间
    """
    cal_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date).sort('calendarDate')
    cal_dates = cal_dates[cal_dates['isOpen'] == 1]
    trade_list = cal_dates['calendarDate'].values.tolist()
    trade_list = [time_change(x) for x in trade_list]
    month_end = cal_dates[cal_dates['isMonthEnd'] == 1]
    month_end = month_end['calendarDate'].values.tolist()
    month_end = [time_change(x) for x in month_end]
    cal = Calendar('China.SSE')
    month_start = [cal.advanceDate(x, '1B').strftime('%Y%m%d') for x in month_end]
    return trade_list, month_end, month_start

# 剔除ST股票
def st_remove(source_universe, st_date=None):
    """
    Args:
        source_universe (list of str): 需要进行筛选的股票列表
        st_date (datetime): 进行筛选的日期,默认为调用当天
    Returns:
        list: 去掉ST股票之后的股票列表
    """
    st_date = st_date if st_date is not None else datetime.datetime.now().strftime('%Y%m%d')
    df_ST = DataAPI.SecSTGet(secID=source_universe, beginDate=st_date, endDate=st_date, field=['secID'])
    return [s for s in source_universe if s not in list(df_ST['secID'])]

# 剔除某个日期前多少个交易日,之后上市的新股
def new_remove(ticker,tradeDate= None,day = 1):
    """
    Args:
        ticker (list of str): 需要进行筛选的股票列表（无后缀）
        tradeDate (datetime): 进行筛选的日期,默认为调用当天
        day (int): 向前漂移的交易日的个数
    Returns:
        list: 去掉新股股票之后的股票列表（无后缀）
    """
    tradeDate = tradeDate if tradeDate is not None else datetime.datetime.now()
    period = '-' + str(day) + 'B'
    pastDate = cal.advanceDate(tradeDate,period)
    pastDate = pastDate.strftime("%Y-%m-%d")
    ipo_date = DataAPI.SecIDGet(partyID=u"",assetClass=u"e",ticker=ticker,cnSpell=u"",field=u"ticker,listDate",pandas="1")
    remove_list = ipo_date[ipo_date['listDate'] > pastDate]['ticker'].tolist()
    return [stk for stk in ticker if stk not in remove_list]

# 将股票代码转化为股票内部编码
def ticker2secID(ticker):
    """
    Args:
        tickers (list): 需要转化的股票代码列表
    Returns:
        list: 转化为内部编码的股票编码列表
    """
    universe = DataAPI.EquGet(equTypeCD=u"A",listStatusCD="L,S,DE,UN",field=u"ticker,secID",pandas="1") # 获取所有的A股（包括已退市）
    universe = dict(universe.set_index('ticker')['secID'])
    if isinstance(ticker, list):
        res = []
        for i in ticker:
            if i in universe:
                res.append(universe[i])
            else:
                print i, ' 在universe中不存在，没有找到对应的secID！'
        return res
    else:
        raise ValueError('ticker should be list！')

# 获取股票池函数
def get_stock_pool(date, N):
    """
    Args:
        date: 月初时间
        N: 新股的定义时间
    Returns:
        stock_pool: 此月初的股票池
    """
    univ=DynamicUniverse('A')
    all_code = univ.preview(date,skip_halted=False)
    all_code_not_ST = st_remove(all_code, st_date=date)
    ticker = [x.split('.')[0] for x in all_code_not_ST]
    all_code_need = new_remove(ticker, tradeDate=date, day=N)
    code = ticker2secID(all_code_need)
    df = pd.DataFrame({'code': code})
    df['date'] = date
    df = df[['date', 'code']]
    return df


# 股票池文件存放目录，如果目录不存在，程序自动新建一个
raw_data_dir = "./enhance_strategy_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

# 获取每月初的股票池(start_date - end_date)
tic = time.time()
path = 'enhance_strategy_data/'
start_date = '20071220'
end_date = '20180831'        
trade_list, month_end, month_start = get_trade_list(start_date, end_date)
N = 180
all_stock = []
for date in month_start:
    stock = get_stock_pool(date, N)
    all_stock.append(stock)

all_stock = pd.concat(all_stock)
all_stock.to_csv(path + 'stock_pool.csv', index=False)
toc = time.time()
print('***********股票池示例************')
print(all_stock.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")


'''


1.2 因子构造及预处理

因子构造

考虑到因子的全面性和代表性，我们从规模、估值、成长、盈利、技术、流动性、波动等维度来筛选具有长期稳定选股能力的因子，因子集合如下： 图片注释
我们将因子的计算分成两个部分：

第一部分是财务相关因子的计算，计算方法包含在FinancialFactor类中；
第二部分是技术因子的计算，我们在月末完成单期因子的计算；
在得到原始因子后，我们对每个因子进行如下处理：

Step1: 用申万一级行业的中位数对缺失值进行填充；
Step2: 采用MAD（Median Absolute Deviation 绝对中位数法）进行边界压缩处理，剔除异常值；
Step3: 对除LNCAP以外的其他因子进行市值 + 行业的中性化，对LNCAP做行业中性化；
Step4: 对第二步的残差项做z-score标准化处理，
Step5: 计算每个股票的次月收益数据备用。 factor_stand就是做完预处理后的标准化因子数据，格式如下：（列数太多截取部分） 图片注释


'''

# 第一部分：财务相关因子计算
class FinancialFactor(object):
    """
    计算财务相关因子
    """

    def __init__(self, income_statement, asset_statement, end_date_list):
        self.income_statement = income_statement  # 利润及收入相关数据
        self.asset_statement = asset_statement  # 资产数据
        self.end_date_list = end_date_list

    @classmethod
    def cal_signal(cls, df, columns, end_date_list, shift=True):
        '''
        计算业绩惊喜因子, V = (Qt - Qt-4)/std(delta(Q))
        delta(Q)为过去12期的 Qt-Qt-4的标准差，过去12期不包含当期
        :param df:至少包括: secID, publishDate, endDate, [columns]
        :param columns: list, 用来计算惊喜因子的会计科目
        :param end_date_list: 时间区间
        :param shift: 计算时候是否进行漂移
        :return: 因子值dataframe, 列为 publishDate, [columns], publishDate格式和输入一致
        '''
        df1 = df.copy()
        df1.sort_values(by=['publishDate', 'endDate'], ascending=False, inplace=True)
        df2 = df.set_index('publishDate')[columns]
        date_list = df1['publishDate'].unique()
        date_list.sort()

        for date in sorted(date_list):
            tmp = df1[df1.publishDate <= date]
            tmp.drop_duplicates(subset=['endDate'], inplace=True, keep='first')

            tmp = tmp.sort_values(by='endDate', ascending=False).set_index('endDate')
            report_end_date = tmp.index[0]
            report_date_list = end_date_list[end_date_list <= report_end_date][-13:][::-1]
            tmp = tmp.reindex(report_date_list).head(13)

            tmp[columns] = tmp[columns].diff(-4)
            for column in columns:
                sigma = tmp[column][1:].std() if len(tmp[column].dropna()) >= 4 else np.NaN
                if shift:
                    df2.loc[date, column] = (tmp[column].iloc[0] - tmp[column].iloc[1:].mean()) / sigma
                else:
                    df2.loc[date, column] = tmp[column].iloc[0] / sigma
        df2 = df2.reset_index()
        return df2[['publishDate'] + columns]

    @classmethod
    def cal_yoy_signal(cls, df, columns):
        '''
        计算同比增长率， value = (Qt-Q(t-4))/abs(Q(t-4))
        :param df: 至少包括: secID, publishDate, endDate, [columns]
        :param columns: 列名，用来计算同比的会计科目，list格式
        :return: dataframe， 列为: secID, publishDate, [columns]
        return的columns虽然和输入同名，但值为同比值， publishDate格式为"%Y-%m-%d"
        '''
        df1 = df.copy()
        # 转成int类型，便于进行rolling计算
        df1['endDate'] = df1['endDate'].apply(lambda x: int(x.replace("-", "")))
        df1['publishDate'] = df1['publishDate'].apply(lambda x: int(x.replace("-", "")))
        # 去年同期endDate
        df1['pre_end_date'] = df1['endDate'] - 10000
        # 需要合并的去年同期值
        m_df = df1[['publishDate', 'endDate'] + columns].rename(columns={"publishDate": "pre_pub",
                                                                         'endDate': 'pre_end_date'})

        df1 = df1.merge(m_df, on=['pre_end_date'], how='inner', suffixes=['', "_pre"])

        df1['max_pub_date'] = np.max(df1[['publishDate', 'pre_pub']], axis=1)
        # 同一个发布日期，保留最大的endDate的值
        df1.sort_values(by=['max_pub_date', 'endDate', 'pre_pub'], ascending=True, inplace=True)
        df1 = df1.drop_duplicates(subset=['max_pub_date'], keep='last')

        # 得到最近8条记录对应的最大max_endDate
        df1['max_end_date'] = df1['endDate'].rolling(window=8, min_periods=1).max()
        # 取endDate为最新的记录
        df1 = df1[df1['endDate'] == df1['max_end_date']]
        # 计算因子值
        for column in columns:
            pre_value_col = column + "_pre"
            df1[column] = (df1[column] - df1[pre_value_col]) / abs(df1[pre_value_col])
        # 将publishDate转成 '%Y-%m-%d'格式
        df1 = df1[['secID', 'max_pub_date'] + columns].rename(columns={"max_pub_date": "publishDate"})
        df1['publishDate'] = df1['publishDate'].apply(lambda x: "%s-%s-%s" % (str(x)[:4], str(x)[4:6], str(x)[6:8]))
        return df1

    def cal_sue_sur(self):
        """
        不带漂移项的业绩惊喜因子
        :return: sue和sur的DataFrame, 公告日发布后计算的因子数据
        列为： secID, publishDate, signal
        """
        su_df = self.income_statement.groupby(by='secID').apply(lambda x: FinancialFactor.cal_signal(x,
                                                                                                     ['NIncomeAttrP',
                                                                                                      'revenue'],
                                                                                                     self.end_date_list,
                                                                                                     False))
        if 'secID' in su_df.columns:
            su_df = su_df.drop('secID', axis=1)
        su_df = su_df.reset_index()
        su_df.drop_duplicates(subset=['secID', 'publishDate'], inplace=True, keep='first')
        sue = su_df[['secID', 'publishDate', 'NIncomeAttrP']].dropna()
        sue = sue[['secID', 'publishDate', 'NIncomeAttrP']].rename(columns={"NIncomeAttrP":'signal'})
        sur = su_df[['secID', 'publishDate', 'revenue']].dropna()
        sur = sur[['secID', 'publishDate', 'revenue']].rename(columns={"revenue": 'signal'})
        return sue, sur

    def cal_growth_yoy(self):
        """
        计算净利润增长率单季度同比因子
        :return:DataFrame, 公告日发布后计算的因子数据
        """
        growth_yoy = self.income_statement.groupby(by='secID').apply(lambda x: FinancialFactor.cal_yoy_signal(x,
                                                                                                              [
                                                                                                                  'NIncomeAttrP',
                                                                                                                  'revenue']))
        profit_growth_yoy = growth_yoy[['secID', 'publishDate', 'NIncomeAttrP']].dropna().rename(
            columns={"NIncomeAttrP": "signal"})
        sales_growth_yoy = growth_yoy[['secID', 'publishDate', 'revenue']].dropna().rename(
            columns={"revenue": "signal"})
        profit_growth_yoy.reset_index(drop=True, inplace=True)
        sales_growth_yoy.reset_index(drop=True, inplace=True)
        return profit_growth_yoy, sales_growth_yoy

    @classmethod
    def cal_latest_pit_num(cls, df, col):
        '''
        取财务数据中最新的col列的值（最大的endDate), endDate指财务发布期，如2018-03-30
        :param df: 财务数据dataframe，至少包括 publishDate, endDate, col
        :param col: 财务科目
        :return: 每个公告日对应的最新endDate的col值
        dataframe格式，列为 publishDate,max_end_date, col, publishDate格式为"%Y-%m-%d"
        '''
        df1 = df[['publishDate', 'endDate', col]]
        # 转成int类型，便于进行rolling计算
        df1['endDate'] = df1['endDate'].apply(lambda x: int(x.replace("-", "")))
        df1['publishDate'] = df1['publishDate'].apply(lambda x: int(x.replace("-", "")))
        # 根据发布日、endDate升序
        df1.sort_values(by=['publishDate', 'endDate'], ascending=True, inplace=True)
        # 得到最近8条记录对应的最大max_endDate
        df1['max_end_date'] = df1['endDate'].rolling(window=8, min_periods=1).max()
        # 合并max_end_date的数值
        merge_df = df1[['publishDate', 'endDate', col]]
        merge_df.columns = ['m_pubdate', 'max_end_date', 'signal']
        df1 = df1.merge(merge_df, on=['max_end_date'], how='left')
        # 剔除未来的数据
        df1 = df1[df1.publishDate >= df1.m_pubdate]
        # 取已发布的最新endDate的值
        df1.sort_values(by=['publishDate', 'max_end_date', 'm_pubdate'], ascending=True, inplace=True)
        df1.drop_duplicates(subset=['publishDate'], keep='last', inplace=True)
        # 取endDate为最新的记录
        df1['publishDate'] = df1['publishDate'].apply(lambda x: "%s-%s-%s" % (str(x)[:4], str(x)[4:6], str(x)[6:8]))
        df1['max_end_date'] = df1['max_end_date'].apply(lambda x: "%s-%s-%s" % (str(x)[:4], str(x)[4:6], str(x)[6:8]))
        return df1[['publishDate', 'max_end_date', 'signal']]

    @classmethod
    def cal_latest_2pit_mean(cls, df, col):
        '''
        取财务数据中最新的两个col列的值的均值（最大的endDate和次大endDate), endDate指财务发布期，如2018-03-30
        :param df: 财务数据dataframe，至少包括 publishDate, endDate, col
        :param col: 财务科目
        :return: 每个公告日对应的最新endDate的col值
        dataframe格式，列为 publishDate,max_end_date,col, publishDate格式为"%Y-%m-%d"
        '''
        df1 = df[['publishDate', 'endDate', col]]
        # 转成int类型，便于进行rolling计算
        df1['endDate'] = df1['endDate'].apply(lambda x: int(x.replace("-", "")))
        df1['publishDate'] = df1['publishDate'].apply(lambda x: int(x.replace("-", "")))
        # 根据发布日、endDate升序
        df1.sort_values(by=['publishDate', 'endDate'], ascending=True, inplace=True)
        # 得到最近8条记录对应的最大max_endDate
        df1['max_end_date'] = df1['endDate'].rolling(window=8, min_periods=1).max()
        # 合并max_end_date的数值
        merge_df = df1[['publishDate', 'endDate', col]]
        merge_df.columns = ['m_pubdate', 'max_end_date', 'signal']
        df1 = df1.merge(merge_df, on=['max_end_date'], how='left')
        # 剔除未来的数据
        df1 = df1[df1.publishDate >= df1.m_pubdate]
        # 取已发布的最新endDate的值
        df1.sort_values(by=['publishDate', 'max_end_date', 'm_pubdate'], ascending=True, inplace=True)
        df1.drop_duplicates(subset=['publishDate'], keep='last', inplace=True)

        def calc_prev_enddate(x):
            '''
            计算上一个财报的enddate, x为int，如20180330
            返回值为int类型
            '''
            c_m = str(x)[4:6]
            c_y = str(x)[:4]
            if c_m == '03':
                prev_enddate = int("%s1231" % (int(c_y) - 1))
            elif c_m == '06':
                prev_enddate = int("%s0331" % c_y)
            elif c_m == '09':
                prev_enddate = int("%s0630" % c_y)
            elif c_m == '12':
                prev_enddate = int("%s0930" % c_y)
            else:
                raise Exception("not valid month, %s" % c_m)
            return prev_enddate

        # 次大end_date的值
        df1['prev_end_date'] = df1['max_end_date'].apply(lambda x: calc_prev_enddate(x))
        # 合并次大end_date的值
        merge_df.columns = ['prev_m_pubdate', 'prev_end_date', 'prev_signal']
        df1 = df1.merge(merge_df, on=['prev_end_date'], how='left')
        # 剔除未来的数据
        df1 = df1[df1.publishDate >= df1.prev_m_pubdate]
        # 取已发布的最新endDate的值
        df1.sort_values(by=['publishDate', 'prev_end_date', 'prev_m_pubdate'], ascending=True, inplace=True)
        df1.drop_duplicates(subset=['publishDate'], keep='last', inplace=True)
        # 取近期两个值的均值
        df1['signal'] = df1[['signal', 'prev_signal']].mean(axis=1)
        # 取endDate为最新的记录
        df1['publishDate'] = df1['publishDate'].apply(lambda x: "%s-%s-%s" % (str(x)[:4], str(x)[4:6], str(x)[6:8]))
        df1['max_end_date'] = df1['max_end_date'].apply(lambda x: "%s-%s-%s" % (str(x)[:4], str(x)[4:6], str(x)[6:8]))
        return df1[['publishDate', 'max_end_date', 'signal']]

    def cal_roe(self):
        """
        计算ROE相关因子
        :return:DataFrame, 公告日发布后计算的因子数据
        列为：'secID', 'publishDate', 'roe', 'delta_roe'
        """
        # 计算每个公告日对应的净利润值（最新的财报期)
        numerator = self.income_statement.groupby(by='secID').apply(
            lambda x: FinancialFactor.cal_latest_pit_num(x, 'NIncomeAttrP'))
        if 'secID' in numerator.columns:
            numerator = numerator.drop('secID', axis=1)
        numerator = numerator.reset_index()
        numerator = numerator.rename(columns={'signal': "signal_NIAttrP", 'max_end_date': "endDate"}).dropna()

        # 计算每个公告日对应的净资产值(最新的财报期)
        denominator = self.asset_statement.groupby(by='secID').apply(
            lambda x: FinancialFactor.cal_latest_2pit_mean(x, 'TEquityAttrP'))
        if 'secID' in denominator.columns:
            denominator = denominator.drop('secID', axis=1)
        denominator = denominator.reset_index()
        denominator = denominator.rename(columns={'signal': "signal_TEAttrP", 'max_end_date': "endDate"}).dropna()

        df = pd.merge(numerator, denominator, on=['secID', 'publishDate', 'endDate'], how='left')
        df['roe'] = df['signal_NIAttrP'] / df['signal_TEAttrP']
        df = df.dropna()
        # 去年同期的roe
        df['pre_endDate'] = df['endDate'].apply(lambda x: "%s%s" % (int(x[:4]) - 1, x[4:]))
        merge_df = df[['secID', 'publishDate', 'endDate', 'roe']]
        merge_df.columns = ['secID', 'pre_pubdate', 'pre_endDate', 'pre_roe']
        # 合并去年的值
        df = df.merge(merge_df, on=['secID', 'pre_endDate'], how='left')
        df = df[(df.publishDate >= df.pre_pubdate) | (df.pre_pubdate.isnull())]
        df.sort_values(by=['secID', 'publishDate', 'endDate', 'pre_pubdate'], ascending=True, inplace=True)
        df.drop_duplicates(subset=['secID', 'publishDate'], keep='last', inplace=True)

        df['delta_roe'] = df['roe'] - df['pre_roe']
        df = df[['secID', 'publishDate', 'roe', 'delta_roe']]
        df.reset_index(drop=True, inplace=True)
        return df
    
def fill_factor(df, name):
    """
    处理财务因子数据的格式
    """
    df = df.pivot(index='publishDate', columns='secID', values='signal').loc[trade_date_list, :].fillna(method='ffill').loc[month_date_list, :].unstack().reset_index()
    df.columns = ['secID', 'publishDate', name]
    return df


# 第一部分：财务相关因子计算
tic = time.time()
# 利润及收入数据
income_data = DataAPI.FdmtISQPITGet(field=u"secID,publishDate,endDate,NIncomeAttrP,NIncome,revenue", pandas="1")
income_data = income_data[income_data['secID'].str[0].isin(['0', '3', '6'])]
# 资产数据
asset_data = DataAPI.FdmtBSGet(field=u"secID,publishDate,endDate,TEquityAttrP,TShEquity", pandas="1")
asset_data = asset_data[asset_data['secID'].str[0].isin(['0', '3', '6'])]  
    
date_list = np.array(sorted(income_data['endDate'].unique()))
financial_factor = FinancialFactor(income_data, asset_data, date_list)
#　披露期因子计算    
sue, sur = financial_factor.cal_sue_sur()

profit_growth_yoy, sales_growth_yoy = financial_factor.cal_growth_yoy()

earning_factor = financial_factor.cal_roe()
    
# 将PIT的因子数据转成月末因子值
calendar = DataAPI.TradeCalGet(exchangeCD='XSHG', beginDate='20070101', endDate='20180831')
calendar = calendar[calendar['isOpen'] == 1]
trade_date_list = calendar['calendarDate'].tolist()

month_date_list = calendar[calendar['isMonthEnd'] == 1]['calendarDate']
month_date_list = month_date_list[month_date_list > '2007-01-01'].tolist()

# 业绩惊喜因子
sue = fill_factor(sue, 'sue')
sur = fill_factor(sur, 'sur')
profit_growth_yoy = fill_factor(profit_growth_yoy, 'profit_growth_yoy')
sales_growth_yoy = fill_factor(sales_growth_yoy, 'sales_growth_yoy')
# 盈利相关因子
roe = fill_factor(earning_factor[['secID', 'publishDate', 'roe']].rename(columns={"roe": "signal"}), 'roe')
delta_roe = fill_factor(earning_factor[['secID', 'publishDate', 'delta_roe']].rename(columns={"delta_roe": "signal"}), 'delta_roe')
# 成长和盈利因子整合
growth = sue.merge(sur, on=['secID', 'publishDate']).merge(profit_growth_yoy, on=['secID', 'publishDate']). \
    merge(sales_growth_yoy, on=['secID', 'publishDate'])
growth.columns = ['code', 'date', 'sue', 'sur', 'profit_growth_yoy', 'sales_growth_yoy']
growth['date'] = map(time_change, growth['date'])
earning = roe.merge(delta_roe, on=['secID', 'publishDate'])
earning.columns = ['code', 'date', 'roe', 'delta_roe']
earning['date'] = map(time_change, earning['date'])

toc = time.time()
print ("\n ----- Financial factor Computation time = " + str((toc - tic)) + "s")


# 计算对数市值因子
def cal_lnmkt(date, code):
    """
    Args:
        date: 日期
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        mkt: 对数市值因子数据,dataframe格式，列名为日期，股票代码，因子值
    """   
    mkt = DataAPI.MktEqudGet(tradeDate=date,secID=code,field=u"tradeDate,secID,marketValue",pandas="1")
    mkt.columns = ['date', 'code', 'mkt']
    mkt['mkt'] = np.log(mkt['mkt'])
    mkt['date'] = map(time_change, mkt['date'])
    return mkt

# 估值因子：BP、EPTTM、SPTTM
def cal_value_factor(date, code):
    """
    Args:
        date: 日期
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns: 
        value: 估值因子数据，dataframe格式，列名为日期，股票代码，BP，EPTTM，SPTTM
    """
    temp = DataAPI.MktStockFactorsOneDayProGet(tradeDate=date,secID=code,field=u"tradeDate,secID,PB,PE,PS",pandas="1")
    temp['BP'] = 1.0 / temp['PB']
    temp['EPTTM'] =  1.0 / temp['PE']
    temp['SPTTM'] = 1.0 / temp['PS']
    value = temp[['tradeDate', 'secID', 'BP', 'EPTTM', 'SPTTM']]
    value.columns = ['date', 'code', 'BP', 'EPTTM', 'SPTTM']
    value['date'] = map(time_change, value['date'])
    return value

# 技术因子：一月、三月反转
def cal_reverse(date, code):
    """
    Args:
        date: 日期
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        ret: 反转因子数据，dataframe格式，列名为日期，股票代码，因子值
    """
    cal = Calendar('China.SSE')
    pre_20 = cal.advanceDate(date, '-20B').strftime('%Y%m%d')
    pre_60 = cal.advanceDate(date, '-60B').strftime('%Y%m%d')
    close_20 = DataAPI.MktEqudAdjAfGet(secID=code,tradeDate=pre_20,field=u"secID,closePrice",pandas="1")
    close_20.columns = ['code', 'close_20']
    close_60 = DataAPI.MktEqudAdjAfGet(secID=code,tradeDate=pre_60,field=u"secID,closePrice",pandas="1")
    close_60.columns = ['code', 'close_60']
    close = DataAPI.MktEqudAdjAfGet(secID=code,tradeDate=date,field=u"tradeDate,secID,closePrice",pandas="1")
    close.columns = ['date', 'code', 'close']
    ret = close.merge(close_20, on='code').merge(close_60, on='code')
    ret['ret_20'] = ret['close'] / ret['close_20'] - 1
    ret['ret_60'] = ret['close'] / ret['close_60'] - 1 
    reverse = ret[['date', 'code', 'ret_20', 'ret_60']]
    reverse['date'] = map(time_change, reverse['date'])
    return reverse

# 获取某个时点股票所属行业（申万一级行业，不回填）
def get_industry(date, code):
    """
    Args:
        date: 月初时间
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        indu: 申万一级行业因子数据,dataframe格式，列名为日期，股票代码，行业名
    """ 
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d') 
    indu = DataAPI.MdSwBackGet(secID=code,field=u"secID,isNew,oldTypeName,industryName1,intoDate,outDate",pandas="1")
    indu['outDate'].fillna('2050-01-01', inplace=True)
    indu['intoDate'] = map(time_change, indu['intoDate'])
    indu['outDate'] = map(time_change, indu['outDate'])
    indu = indu[(indu['intoDate']<=end) & (indu['outDate']>end)]
    indu.drop_duplicates(subset=['secID'], inplace=True)
    indu['date'] = end
    indu = indu[['date', 'secID', 'industryName1']]
    indu.columns = ['date', 'code', 'industry']
    return indu

# 流动性因子： 非流动性冲击，一月日均换手，三月日均换手
def cal_liquidity(date, code, mkt_info):
    """
    Args:
        date: 月初日期
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
        mkt_info: 行情数据
    Returns:
        liquidity: 流动性因子数据，dataframe格式，列名为日期，股票代码，ILLIQ，turn_1M，turn_3M
    """
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')
    pre_20 = cal.advanceDate(end, '-19B').strftime('%Y%m%d')
    pre_60 = cal.advanceDate(end, '-59B').strftime('%Y%m%d')
    # ILLIQ
    pct_chg = mkt_info[(mkt_info['tradeDate'] >= pre_20) & (mkt_info['tradeDate'] <= end)]
    pct_chg['daily'] = np.where(pct_chg['turnoverValue'] == 0, np.nan, 10e9 * np.abs(pct_chg['chgPct']) / pct_chg['turnoverValue'])
    illiq = pct_chg.groupby(by='secID').apply(lambda x: x['daily'].mean())
    illiq = illiq.reset_index(level=0)
    illiq.columns = ['code', 'ILLIQ']
    illiq['date'] = end
    # turnover_3M
    turn = mkt_info[(mkt_info['tradeDate'] >= pre_60) & (mkt_info['tradeDate'] <= end)]
    turn['turn'] = np.where(turn['turnoverRate'] == 0, np.nan, turn['turnoverRate'])
    turn_3M = turn.groupby(by='secID').apply(lambda x: x['turn'].mean())
    turn_3M = turn_3M.reset_index(level=0)
    turn_3M.columns = ['code', 'turn_3M']
    turn_3M['date'] = end
    # turnover_1M
    turn = turn[turn['tradeDate'] >= pre_20]
    turn_1M = turn.groupby(by='secID').apply(lambda x: x['turn'].mean())
    turn_1M = turn_1M.reset_index(level=0)
    turn_1M.columns = ['code', 'turn_1M']
    turn_1M['date'] = end
    
    liquidity = illiq.merge(turn_3M, on=['date', 'code']).merge(turn_1M, on=['date', 'code'])
    liquidity = liquidity[['date', 'code', 'ILLIQ', 'turn_1M', 'turn_3M']]
    return liquidity

# 波动： 特异度，一个月真实波幅，三个月真实波幅
# 特异度因子
def cal_specificity(date, code, mkt_info):
    """
    Args:
        date: 月初日期
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
        mkt_info: 行情数据
    Returns:
        spec: 特异度因子数据,dataframe格式，列名为日期，股票代码，因子值
    """
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')
    pre_20 = cal.advanceDate(date, '-20B').strftime('%Y%m%d')
    all_data = mkt_info[(mkt_info['tradeDate'] >= pre_20) & (mkt_info['tradeDate'] <= end)]
    tdata = all_data[(all_data['tradeDate'] == end) & (all_data['secID'].isin(code))]
    # 市场组合
    tdata['weight'] = tdata['negMarketValue'] / tdata['negMarketValue'].sum()
    port_all = tdata[['secID', 'weight']]
    port_all.columns = ['secID', 'weight_market']    
    # 大小市值组合
    temp = tdata.copy()
    temp.sort_values(by='negMarketValue', inplace=True)
    temp.reset_index(drop=True, inplace=True)
    num = len(temp) / 3
    port_small_mkv = temp[0: num]
    port_small_mkv['weight_sm'] = port_small_mkv['negMarketValue'] / port_small_mkv['negMarketValue'].sum()
    port_small_mkv = port_small_mkv[['secID', 'weight_sm']]
    port_large_mkv = temp[-num:]
    port_large_mkv['weight_lm'] = port_large_mkv['negMarketValue'] / port_large_mkv['negMarketValue'].sum()    
    port_large_mkv = port_large_mkv[['secID', 'weight_lm']]
    # 高低PB组合
    temp = tdata.copy()
    temp.sort_values(by='PB', inplace=True)
    temp.reset_index(drop=True, inplace=True)
    port_low_pb = temp[0: num]
    port_low_pb['weight_lp'] = port_low_pb['negMarketValue'] / port_low_pb['negMarketValue'].sum()
    port_low_pb = port_low_pb[['secID', 'weight_lp']]
    port_high_pb = temp[-num:]
    port_high_pb['weight_hp'] = port_high_pb['negMarketValue'] / port_high_pb['negMarketValue'].sum()    
    port_high_pb = port_high_pb[['secID', 'weight_hp']]
    # 整合
    weight = pd.merge(port_all, port_small_mkv, on='secID', how='left')
    weight = pd.merge(weight, port_large_mkv, on='secID', how='left')
    weight = pd.merge(weight, port_low_pb, on='secID', how='left')
    weight = pd.merge(weight, port_high_pb, on='secID', how='left')
    weight.fillna(0, inplace=True)
    # 收益矩阵
    cal_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=pre_20, endDate=end).sort('calendarDate')
    cal_dates = cal_dates[cal_dates['isOpen']==1]
    date_list = cal_dates['calendarDate'].values.tolist()
    date_list = [time_change(x) for x in date_list]
    for day in date_list:
        pct_chg = all_data[all_data['tradeDate'] == day]
        pct_chg = pct_chg[['secID', 'chgPct']]
        pct_chg.columns = ['secID', 'chgPct__' + str(day)]
        weight = pd.merge(weight, pct_chg, on='secID')
    # 日收益序列
    w_mat = np.matrix(weight.iloc[:, 1: 6]).T
    change_mat = np.matrix(weight.iloc[:, 6: ])                 
    ret = w_mat * change_mat
    cols = ['ret_all', 'ret_small_mkv', 'ret_large_mkv', 'ret_low_pb', 'ret_high_pb']
    ret = pd.DataFrame(ret.T,columns=cols)
    ret['date'] = date_list
    ret['ret_mkv'] = ret['ret_small_mkv'] - ret['ret_large_mkv']
    ret['ret_pb'] = ret['ret_low_pb'] - ret['ret_high_pb']
    ret['constant'] = 1
    ret = ret[['date', 'constant', 'ret_all', 'ret_mkv', 'ret_pb']]
    # 回归
    pct_table = weight.iloc[:, 6: ]
    pct_table.columns = [str(x.split('__')[1]) for x in list(pct_table.columns)]
    pct_table = pct_table.T
    pct_table.columns = list(weight['secID'])
    reg_data = pd.merge(ret, pct_table, left_on='date', right_index=True)
    x = reg_data.iloc[:, 1:5]  # 这里加了常数项constant
    col = reg_data.columns[5:]
    IV_col = []
    IVR = []
    all_code = []
    for name in col:
        y = reg_data[name]
        y = y.replace(0, np.nan)
        # 做个判定，回归天数太少的要剔除
        if len(y[y.isnull()]) < 10:
            model = sm.OLS(y, x, missing='drop')
            results = model.fit()
            IV = np.std(results.resid) * np.sqrt(252)
            R2_single = results.rsquared        
            all_code.append(name)            
            IV_col.append(IV)
            IVR.append(1 - R2_single)
        else:
            continue
    spec = pd.DataFrame({'secID': all_code, 'IVFF': IV_col, 'IVR': IVR})
    spec = pd.merge(port_all, spec, on='secID', how='left') # 没有的记为nan
    spec['date'] = end
    spec = spec[['date', 'secID', 'IVR']]
    spec.columns = ['date', 'code', 'IVR']
    return spec

def cal_atrp(date, code, mkt_info):
    """
    Args:
        date: 月初日期
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
        mkt_info: 行情数据
    Returns:
        : atrp因子数据，dataframe格式，列名为日期，股票代码，ATRP_20, ATRP_60
    """
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d')
    pre_20 = cal.advanceDate(end, '-19B').strftime('%Y%m%d')
    pre_60 = cal.advanceDate(end, '-59B').strftime('%Y%m%d')
    period_info = mkt_info[(mkt_info['tradeDate'] >= pre_60) & (mkt_info['tradeDate'] <= end)]
    period_info.rename(columns={"secID": "code", "tradeDate": "date", "preClosePrice": "pre_close", "highestPrice": "high",
                             "lowestPrice": "low", "closePrice": "close", "turnoverVol": "volume"}, inplace=True)  
    period_info['cand1'] = period_info['high'] - period_info['low']
    period_info['cand2'] = abs(period_info['high'] - period_info['pre_close'])
    period_info['cand3'] = abs(period_info['low'] - period_info['pre_close'])
    period_info['tr'] = np.maximum(period_info['cand1'], period_info['cand2'])
    period_info['tr'] = np.maximum(period_info['tr'], period_info['cand3'])
    period_info['trp'] = period_info['tr'] / period_info['close']
    period_info['trp'] = np.where(period_info['volume'] < 1e-8, np.nan, period_info['trp'])
    # ATRP_3M                         
    atrp_3M = period_info.groupby(by='code').apply(lambda x: x['trp'].mean())
    atrp_3M = atrp_3M.reset_index(level=0)
    atrp_3M.columns = ['code', 'ATRP_3M']
    # ATRP_1M
    period_info = period_info[period_info['date'] >= pre_20]
    atrp_1M = period_info.groupby(by='code').apply(lambda x: x['trp'].mean())
    atrp_1M = atrp_1M.reset_index(level=0)
    atrp_1M.columns = ['code', 'ATRP_1M']
        
    atrp = atrp_1M.merge(atrp_3M, on='code')
    atrp['date'] = end
    return atrp

# 多线程取数据备用
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(processes=16)

# 获取给定区间的行情信息
def get_mkt_info(params):
    '''
    Args：
        params = [code, date, equd_list, adj_list]
        code: 股票代码集合
        date: 起始时间和结束时间列表， ['20080101', '20081231']
        equd_list: mktequd数据列表
        adj_list: 前复权数据列表
    Return:
        DataFrame, 返回日期区间的数据值
        '''
    code, date, equd_list, adj_list = params
    
    cnt = 0
    while True:
        try:
            tmp_frame1 = DataAPI.MktEqudGet(secID=code,beginDate=date[0],endDate=date[1],
                                           field=["secID","tradeDate"] + equd_list,pandas="1")
            tmp_frame2 = DataAPI.MktEqudAdjGet(secID=code,beginDate=date[0],endDate=date[1],
                                           field=["secID","tradeDate"] + adj_list,pandas="1")
            tmp_frame = tmp_frame1.merge(tmp_frame2, on=['secID', 'tradeDate'])
            return tmp_frame
        except Exception as e:
            cnt += 1
            print "get data failed in get_mkt_info, reason:%s, retry again, retry count:%s" % (e, cnt)
            if cnt >= 3:
                print "max get data retry, will exit"
                raise Exception(e)
        return

tic = time.time()
# 多线程取数据
all_code = sorted(all_stock['code'].unique())
date_list = sorted(all_stock['date'].unique())
# 时间准备，一年取一次
calendar = DataAPI.TradeCalGet(exchangeCD='XSHG', beginDate='20070601', endDate='20180831')
calendar = calendar[calendar['isOpen'] == 1]
year_end = list(calendar[calendar['isYearEnd'] == 1]['calendarDate'])
year_end.append('2007-06-01')
year_end.append('2018-08-31')
year_end = sorted([x.replace('-', '') for x in year_end])
date_l = [[year_end[i], year_end[i + 1]] for i in range(len(year_end) - 1)]

factor_list_1 = ['chgPct', 'turnoverValue', 'turnoverRate', 'PB', 'negMarketValue']
factor_list_2 = ['preClosePrice', 'highestPrice', 'lowestPrice', 'closePrice', 'turnoverVol']
pool_args = zip([all_code] * len(date_l), date_l, [factor_list_1] * len(date_l), [factor_list_2] * len(date_l))
mkt_info_list = pool.map(get_mkt_info, pool_args)
pool.close()
pool.join()
mkt_info = pd.concat(mkt_info_list)
mkt_info['tradeDate'] = map(time_change, mkt_info['tradeDate'])
mkt_info.drop_duplicates(subset=['secID', 'tradeDate'], inplace=True)
toc = time.time()
print ("\n ----- get_mkt_data time = " + str((toc - tic)) + "s")


# 第二部分：月末因子计算
tic = time.time()
date_list = sorted(all_stock['date'].unique())
all_factor = []
for date in date_list:
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d') # 月末时间
    stock = all_stock[all_stock['date'] == date]
    stock['date'] = end
    code = list(stock['code'])
    factor = stock.copy()
    # 估值因子
    temp = cal_value_factor(end, code)
    factor = pd.merge(factor, temp, on=['date', 'code'], how='left')
    # 技术因子
    temp = cal_reverse(end, code)
    factor = pd.merge(factor, temp, on=['date', 'code'], how='left')
    # 流动性因子
    temp = cal_liquidity(date, code, mkt_info)
    factor = pd.merge(factor, temp, on=['date', 'code'], how='left')
    # 特异度
    temp = cal_specificity(date, code, mkt_info)
    factor = pd.merge(factor, temp, on=['date', 'code'], how='left')    
    # atrp
    temp = cal_atrp(date, code, mkt_info)
    factor = pd.merge(factor, temp, on=['date', 'code'], how='left')     
    # 对数市值
    temp = cal_lnmkt(end, code)
    factor = pd.merge(factor, temp, on=['date', 'code'], how='left')
    # 行业虚拟变量
    temp = get_industry(date, code)
    factor = pd.merge(factor, temp, on=['date', 'code'], how='left')    
    all_factor.append(factor)

all_factor = pd.concat(all_factor)
# 原始因子整合
all_factor = all_factor.merge(growth, on=['date', 'code'], how='left').merge(earning, on=['date', 'code'], how='left')
# 调整列的次序
tmp1 = all_factor[all_factor.columns[: 13]].copy()
tmp2 = all_factor[['mkt', 'industry']].copy()
tmp3 = all_factor[all_factor.columns[15: ]].copy()
all_factor = pd.concat([tmp1, tmp3, tmp2], axis=1)
all_factor.to_csv(path + 'raw_factor.csv', index=False, encoding='gbk')
toc = time.time()
print('***********原始因子示例************')
print(all_factor.head(10).to_html())
print ("\n ----- factor Computation time = " + str((toc - tic)) + "s")

# 因子预处理
# 缺失值填充
def nafill_by_sw1(data, factor_name):
    """
    缺失值填充，使用用申万一级行业中位数
    Args：
        data: 因子值，DataFrame
        factor_name: 因子名
    Returns：
        DataFrame, 填充缺失值后的因子值
    """
    data_input = data.copy()
    data_input.loc[:, factor_name] = data_input.loc[:, factor_name].fillna(
        data_input.groupby('industry')[factor_name].transform("median"))

    return data_input

# 因子预处理函数，中位数去极值-->对市值及行业中性化-->标准化，得到处理好的因子数据
def factor_process(factor_name, data, mode):
    """
    Args:
        factor_name: 需要进行预处理的因子名
        data: 某日的原始因子数据
        mode: 对市值因子不需要执行中性化过程，需要作区分，'yes'代表中性化，'no'代表不做中性化
    Returns:
        data: 对指定factor_name做完处理的因子数据
    """    
    # 中位数去极值
    D_mad = abs(data[factor_name] - data[factor_name].median()).median()
    D_m = data[factor_name].median()
    upper = D_m + 5 * D_mad
    lower = D_m - 5 * D_mad
    temp = [max(lower, min(x, upper)) for x in list(data[factor_name])] # 边界压缩 
    data[factor_name] = temp
    n = list(data.columns).index('mkt')
    # 中性化
    if mode == 'yes':
        y = np.array(data[factor_name])
        x = np.array(data[data.columns[n: ]]) # 市值加行业
        x = sm.add_constant(x, has_constant='add')
        model = sm.OLS(y, x, missing='drop')
        results = model.fit()
        data[factor_name] = results.resid
    # 标准化
    data[factor_name] = (data[factor_name] - data[factor_name].mean()) / (data[factor_name].std())
    return data
  
# 后面会用到因子IC数据，所以需要次月收益计算因子IC，这里计算次月收益
def cal_month_ret(code, this_month, next_month):
    """
    Args:
        code: 当月末的股票列表
        this_month: 当月月末时间
        next_month: 次月月末时间
    Returns:
        m_ret: 月末股票的次月收益数据,列名为日期、股票代码、次月收益
    """
    close_tm = DataAPI.MktEqudAdjAfGet(secID=code,beginDate=this_month,endDate=this_month,field=u"secID,closePrice",pandas="1")
    close_tm.columns = ['code', 'close_tm']
    close_nm = DataAPI.MktEqudAdjAfGet(secID=code,beginDate=next_month,endDate=next_month,field=u"secID,closePrice",pandas="1")
    close_nm.columns = ['code', 'close_nm']    
    close = pd.merge(close_tm, close_nm, on='code')
    close['Month_ret'] = close['close_nm'] / close['close_tm'] - 1
    close['date'] = this_month
    m_ret = close[['date', 'code', 'Month_ret']]    
    return m_ret

# 因子预处理
tic = time.time()
factor_stand = []
factor_list = ['BP', 'EPTTM', 'SPTTM', 'ret_20', 'ret_60', 'ILLIQ', 'turn_1M', 'turn_3M', 'IVR', 'ATRP_1M', 'ATRP_3M', 'sue', 'sur',
              'profit_growth_yoy', 'sales_growth_yoy', 'roe', 'delta_roe', 'mkt'] # 因子集合
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
    for factor_name in factor_list[: -1]:
        tdata = factor_process(factor_name, tdata, 'yes')
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
print('***************标准化因子示例***************')
print(factor_stand.head(10).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''
第二部分：收益预测模型
该部分耗时 5分钟
该部分内容是如何对股票的预期收益进行预测，而我们将收益的预测转化为因子的复合，具体包括：

2.1 因子的多重共线性处理：对称正交

2.2 因子权重的反向归零及ICIR加权法

深度报告版权归优矿所有，禁止直接转载或编辑后转载。

   调试 运行
文档
 代码  策略  文档
2.1 因子多重共线性处理：对称正交

因子共线性的困扰
在多因子选股时，我们通常会从规模、技术反转、流动性、波动性、估值、成长、质量等维度根据多个因子的线性加权来为个股进行综合打分，这个打分法有一个隐含假设是因子之间相关性较低，但是我们绘出了本文选取的因子在2010年初至2018年8月底相关系数的均值。从下图可以看到，虽然有些因子属于不同的维度，但是仍然存在明显的相关性，同一维度内因子的相关性则更为突出，如果直接采用常见的加权方法直接对这些因子进行加权，会导致加权后的组合整体在某些因子上的重复暴露，从而会影响组合的长期表现。

因子正交化的原理
因子正交化，本质上是对原始因子（通过一系列线性变换）进行旋转，旋转后得到一组两两正交的新因子，他们之间的相关性为零并且对于收益的解释度（即整体的方差）保持不变。相关性为零保证了旋转后的因子之间没有共线性，而解释度保持不变保证了原始因子包含的信息能否被完全保留。

正交化公式

 
正交化方法
常见的正交化方法就是施密特正交和对称正交，而对称正交相比于传统的施密特正交法有如下优点：

相比于施密特正交，对称正交不需要提供正交次序，对每个因子平等看待；
所有正交过渡矩阵中，对称正交后的矩阵和原始矩阵的相似性最大，我们用变换前后的矩阵的Frobenius范数𝜑来衡量因子正交前后的变化大小。在所有过渡矩阵中，存在唯一解使𝜑最小，该解即为对称正交的过渡矩阵；
对称正交的计算只需要截面因子数据，并不依赖历史数据，因此计算效率非常高。 两种正交方法的示意图如下：

'''


# 因子相关性矩阵
begin_point = '20091231'
date_list = sorted(factor_stand['date'].unique())
date_list = [x for x in date_list if x >= begin_point]
mean_corr = pd.DataFrame()
for i in range(len(date_list)):
    tdata = factor_stand[factor_stand['date'] == date_list[i]]
    corr_frame = tdata[tdata.columns[2: -1]].corr()    
    if i == 0:
        mean_corr = corr_frame
    else:
        mean_corr += corr_frame

mean_corr = mean_corr / len(date_list)
mean_corr = mean_corr.round(2)

f, ax= plt.subplots(figsize = (20, 10))
_ = sns.heatmap(mean_corr, alpha=1.0, annot=True, center=0.0, annot_kws={"size": 8}, linewidths=0.02, 
                linecolor='white', linewidth=0,  ax=ax)
title=u'原始因子的相关性矩阵'
_ = ax.set_title(title, fontproperties=font, fontsize=16)


# 对称正交的实现代码
from numpy import linalg as LA
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
    return out_m

# 正交dataframe
def lowdin_orthog_frame(df, cols):
    '''
    df: 包含因子值的dataframe，示例格式为: [ticker, tradeDate, factor1, factor2, factor3, factor4, ...], 可为横截面或者panel的因子数据
    cols: 需要进行正交的列，如 cols = [factor1,factor2,factor3,factor4...]
    返回:
        对cols进行了正交处理后的dataframe，格式同输入df完全一致
    说明： 如果df的tradeDate不止一个值，则分别在每个tradeDate,对横截面的多个因子值进行正交
    '''
    def orthog_tdate_frame(dframe, cols):
        dframe = dframe.copy()
        dframe[cols] = pd.DataFrame(lowdin_orthog_list(np.array(dframe[cols]).T).T, index=dframe.index, columns = [cols])
        return dframe
    
    df = df.groupby(['date']).apply(orthog_tdate_frame, cols)
    df.index = range(len(df))
    return df

# 对原始因子进行对称正交
using_factors = [x for x in factor_stand.columns if x not in ['date', 'code', 'Month_ret']]
all_orth_factor_df = lowdin_orthog_frame(factor_stand, using_factors)


# 对称正交后的因子相关性矩阵
date_list = sorted(all_orth_factor_df['date'].unique())
date_list = [x for x in date_list if x >= begin_point]
mean_corr = pd.DataFrame()
for i in range(len(date_list)):
    tdata = all_orth_factor_df[all_orth_factor_df['date'] == date_list[i]]
    corr_frame = tdata[tdata.columns[2: -1]].corr()    
    if i == 0:
        mean_corr = corr_frame
    else:
        mean_corr += corr_frame

mean_corr = mean_corr / len(date_list)
mean_corr = mean_corr.round(2)

f, ax= plt.subplots(figsize = (20, 10))
_ = sns.heatmap(mean_corr, alpha=1.0, annot=True, center=0.0, annot_kws={"size": 8}, linewidths=0.02, 
                linecolor='white', linewidth=0,  ax=ax)
title=u'对称正交后因子的相关性矩阵'
_ = ax.set_title(title, fontproperties=font, fontsize=16)

'''


从上图可以看到，对称正交之后的因子两两相关性为零，下面我们检验一下对称正交前后因子加权复合的效果.

   调试 运行
文档
 代码  策略  文档
2.2 因子权重的反向归零及ICIR加权法

原理
在多因子模型中，我们常常会用以下几种常见的加权方式：

因子IC均值加权
因子ICIR加权
最优化复合因子ICIR加权
半衰IC加权
首先本文采用的因子加权方法是ICIR加权，窗口期为12，及用该因子过去12期的IR值作为该因子在当期的权重。
其次，多因子选股策略选择的因子通常都有其合理的投资逻辑，例如对于估值类因子，一般我们都会认为低估值的股票未来的表现要优于高估值的股票票，所以当我们在某个截面上预期高估值股票占优（和长期的投资逻辑不一致时），我们并不建议反向配置该因子，而选择对当期的因子权重作归零处理。

例如下图，这是SPTTM因子的IC序列与滚动12期IC均值，从投资逻辑上来讲SPTTM的IC应该是正的，但是该因子在2013年下半年至2014年上半年中IC的滚动12期均值为负，我们选择在收益预测模型中将该因子的权重配置设为0，即不反向使用该因子来预测收益。

'''

# 计算因子IC的函数
def cal_ic(factor_name, data, mode=False):
    """
    Args:
        factor_name: 需要计算IC的因子名称
        data: 因子数据，至少需要3列：日期、股票因子值、股票下期收益('Month_ret')
        mode: 取值为True或者False,True代表舍弃最后一期计算IC（最后一期的未来月度收益可能没有），False不舍弃
    Returns:
        IC_data: IC结果，dataframe格式，列名为日期，IC值
    """
    IC = []
    date_list = sorted(list(set(data['date'])))
    if mode:
        date_list = date_list[: -1]

    IC_data = data.groupby(['date']).apply(
        lambda x: x[[factor_name, 'Month_ret']].corr(method='spearman').values[0, 1])
    IC_data.name = 'IC_'+factor_name
    IC_data = IC_data.reset_index()
    return IC_data

def cal_weight(df):
    """
    用因子过去N期的IR值作为当期因子的权重
    Args:
        df：因子数据，列名为['date', 'code', 因子名1, 因子名2,..., 次月收益]
    Returns:
        all_weight：DataFrame格式，列名为因子名1,因子名2,... ,日期    
    """
    date_list = sorted(df['date'].unique())
    N = 12
    all_weight = []
    for i in range(N, len(date_list)):
        currentdate = date_list[i]
        period_date = date_list[i - N: i]
        period = df[(df['date'] >= period_date[0]) & (df['date'] <= period_date[-1])]
        # init就是过去N个月各因子的IC序列
        init = pd.DataFrame({'date': period_date})
        for factor_name in period.columns[2: -1]:
            temp = cal_ic(factor_name, period, False)
            init = pd.merge(init, temp, on='date')        
        init = init[init.columns[1: ]]
        weight = np.array(init.mean() / init.std()) # 权重
        weight = pd.DataFrame(weight.reshape(1, len(weight)), columns=period.columns[2: -1])
        weight['date'] = currentdate
        all_weight.append(weight)
    all_weight = pd.concat(all_weight)
    return all_weight

def factor_compose(df, all_weight):
    """
    Args:
        df：因子数据，列名为['date', 'code', 因子名1, 因子名2,..., 次月收益]
        all_weight：DataFrame格式，列名为因子名1,因子名2,... ,日期
    Returns:
        frame：DataFrame格式，列名为['date', 'code', 'compose', 'Month_ret']    
    """
    weight = all_weight.copy()
    date_list = sorted(all_weight['date'].unique())
    weight = weight.set_index('date')
    frame = []
    for date in date_list:
        tdata = df[df['date'] == date]
        factor_loading = np.array(tdata[tdata.columns[2: -1]])
        w = np.array(weight.loc[date, :])
        composed_factor = np.dot(factor_loading, w)
        tdata['compose'] = composed_factor
        frame.append(tdata)
    frame = pd.concat(frame)
    frame = frame[['date', 'code', 'compose', 'Month_ret']]
    return frame

# 原始因子及对称正交化后的因子合成
tic = time.time()
# 每期权重
all_weight = cal_weight(factor_stand)
all_weight.to_csv(path + 'weight.csv', index=False)
# 原始复合因子
compose_raw = factor_compose(factor_stand, all_weight)
compose_orth = factor_compose(all_orth_factor_df, all_weight)
compose_raw.to_csv(path + 'compose_raw.csv', index=False)
compose_orth.to_csv(path + 'compose_orth.csv', index=False)
# 计算正交前后的复合因子的IC情况
ic_raw = cal_ic('compose', compose_raw, True)
ic_orth = cal_ic('compose', compose_orth, True)
# 因子权重反向归零
weight = all_weight[all_weight.columns[: -1]]
tmp = np.sign(weight * weight.mean()).replace(-1, 0)
weight_zero = np.multiply(weight, tmp)
weight_zero['date'] = all_weight['date']
compose_orth_zero = factor_compose(all_orth_factor_df, weight_zero)
compose_orth_zero.to_csv(path + 'compose_orth_zero.csv', index=False)
ic_orth_zero = cal_ic('compose', compose_orth_zero, True)
# 整合
ic_mean = [ic_raw['IC_compose'].mean(), ic_orth['IC_compose'].mean(), ic_orth_zero['IC_compose'].mean()]
icir = [ic_raw['IC_compose'].mean() / ic_raw['IC_compose'].std(), ic_orth['IC_compose'].mean() / ic_orth['IC_compose'].std(),
       ic_orth_zero['IC_compose'].mean() / ic_orth_zero['IC_compose'].std()]
icir = [np.sqrt(12) * x for x in icir]
ic_win = [len(ic_raw[ic_raw['IC_compose'] > 0]) / (len(ic_raw) + 0.0), len(ic_orth[ic_orth['IC_compose'] > 0]) / (len(ic_orth) + 0.0),
         len(ic_orth_zero[ic_orth_zero['IC_compose'] > 0]) / (len(ic_orth_zero) + 0.0)]
ic_count = pd.DataFrame({'Method': [u'原始', u'对称正交', u'对称正交带反向归零'], u'复合因子IC均值': ic_mean, 
                         u'复合因子年化ICIR': icir, u'复合因子IC胜率': ic_win})
toc = time.time()
print('***********IC指标************')
print(ic_count.round(4).to_html())
print ("\n ----- Computation time = " + str((toc - tic)) + "s")

'''
IC结论

对称正交后的复合因子IC均值从0.1285上升到0.1312，ICIR从6.08显著提升到了6.74，IC的胜率也有提升，可见对称正交对因子的复合效果有显著提升；
带反向归零处理使得复合因子的稳健型得到进一步提升，复合因子的ICIR提升至6.93。
   调试 运行
文档
 代码  策略  文档
第三部分：风险控制模型与组合优化
该部分耗时 60分钟
该部分内容是如何在各种风险约束下实现组合优化，根据风险约束的区别，包括如下三个部分：

3.1 风险控制模型及静态指数增强模型

3.2 基于自适应风险控制的指数增强模型

3.3 组合回测及分析

深度报告版权归优矿所有，禁止直接转载或编辑后转载。

   调试 运行
文档
 代码  策略  文档
3.1 风险控制模型及静态指数增强模型

风险控制模型

稳健的收益预测模型是多因子选股策略成功的基石，但是如果仅选用得分最高的一篮子股票构建组合，在一些极端市场环境下可能会产生较大的回撤风险，因此需要对组合进行风险控制，避免组合在某些风格或行业上有过大的暴露。常见的风险控制形式主要包括以下几种：风格暴露约束、行业暴露约束、相对于基准的跟踪误差约束、个股权重约束等。这些约束条件都能有效控制组合相对基准指数的偏离，使组合能稳定地战胜基准指数。

本文采用的组合优化模型如下：


该优化问题的目标函数为最大化组合的预期收益，模型对应的约束条件如下：

第一个约束条件限制了组合相对于基准指数的风格偏移；
第二个约束条件限制了组合相对基准指数的行业偏离；
第三个约束条件限制了组合相对于基准指数成分股的偏离；
第四个约束限制了组合在成份股在权重占比的上限及下限；
第五个约束限制了卖空，并且限制了个股权重的上线为l；
第六个约束要求组合的权重和为1，即组合始终满仓运作。
跟以往的组合优化模型有区别的是，我们摒弃了二次项的跟踪误差约束来控制组合对基准的偏离，取而代之的是用个股相对基准指数成份股的偏离度，这有两个方面的考虑：

直接用跟踪误差作为约束条件进行风险控制需要估计协方差矩阵，对跟踪误差的控制是否成功依赖于协方差矩阵的估计准确性；而直接控制个股相对基准指数成分股偏离度对组合的跟踪误差控制的传导机制更直接，个股偏离度越小，对基准指数的跟踪误差就越小，极端情况下，将个股相对基准成分股权重的偏离设为0时，组合即完全复制基准指数，此时跟踪误差为零；
跟踪误差约束是二次项约束，需要用二阶锥规划来求解，而上述模型中目标函数、个股权重偏离约束、成分股权重占比约束等都是线性的，线性规划问题的求解比二阶锥规划的求解更高效，尤其在变量数急剧增加的时候。
注：研报中的指数增强模型皆是在全A的投资域中进行选股的，因此个股偏离度到跟踪误差的传导并不是完全直接的，因为要考虑到成分股之外的股票的风险影响（例如某只指数成分股之外的股票的权重为1%，那么偏离度就是1%，但是这只股票的风险不能用指数成份股的风险矩阵估计），但是正向关系还是存在的。
静态指数增强模型
下面我们根据前文介绍的收益预测模型、风险控制模型，对沪深300、中证500这两个指数利用组合优化模型进行增强实证回测。
我们对组合优化及回测设定的参数如下：

回测区间从2010年初至2018年8月31日
在剔除ST及新股后的全A投资域进行选股
交易费用双边0.2%
风险维度我们需要市值与基准行业完全一致，行业暴露敞口最大为0.005（一般会限制到0.001以内）
 注意事项 

 第三部分涉及回测及多进程优化，消耗资源较大，需要重启研究环境以释放资源

之前的数据都进行了存储，下面的代码可以直接运行而不需要重跑上面的代码

重启研究环境的步骤为：

网页版：先点击左上角的“Notebook”图标，然后点击左下角的“内存占用x%”图标，在弹框中点击重启研究环境
客户端：点击左下角的“内存x%”, 在弹框中点击重启研究环境

'''

# coding: utf-8

import numpy as np
import pandas as pd
import datetime
import time
import os
import copy
import cvxpy as cvx
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
import seaborn as sns
from multiprocessing import Pool
import cPickle as pickle
from CAL.PyCAL import *    # CAL.PyCAL中包含font
universe = set_universe('A')
cal = Calendar('China.SSE')

# 时间格式转变函数
def time_change(x):
    y = datetime.datetime.strptime(x, '%Y-%m-%d')
    y = y.strftime('%Y%m%d')
    return y

# 获取约束所需数据
def get_size_con_data(df, date, index_ticker):
    """
    Args:
        df: 月末股票池数据
        date: 月末时间
        index_ticker: 指数代码
    Returns:
        mkt: 对数市值数据，dataframe，列名为股票代码，对数流通市值
        sh: 基准指数的对数市值暴露
    """ 
    # 市值暴露向量
    code = list(df['code'])
    mkt = DataAPI.MktEqudGet(tradeDate=date,secID=code,field=u"tradeDate,secID,marketValue",pandas="1")
    mkt['marketValue'] = np.log(mkt['marketValue'])
    mkt = mkt[['secID', 'marketValue']]
    mkt.columns = ['code', 'mkt']
    mkt = mkt.sort_values(by=['code'])
    mkt.reset_index(drop=True, inplace=True)
    #　指数市值暴露
    bench = DataAPI.IdxCloseWeightGet(ticker=index_ticker, beginDate=date, endDate=date, field=u"consID,weight",pandas="1")
    bench.columns = ['code', 'weight']
    bench['weight'] = bench['weight'] / 100
    mkt_b = DataAPI.MktEqudGet(tradeDate=date,secID=list(bench['code']),field=u"tradeDate,secID,marketValue",pandas="1")
    mkt_b['marketValue'] = np.log(mkt_b['marketValue'])
    mkt_b = mkt_b[['secID', 'marketValue']]
    mkt_b.columns = ['code', 'mkt_bench']
    tmp = pd.merge(mkt, bench, on=['code'])
    sh = (tmp['mkt'] * tmp['weight']).sum() # 指数的市值暴露
    return mkt, sh

# 获取某个时点股票所属行业
def get_industry_end(date, code):
    """
    Args:
        date: 月末时间
        code: 股票代码列表，如['000001.XSHE', '000002.XSHE']
    Returns:
        indu: 申万一级行业因子数据,dataframe格式，列名为日期，股票代码，行业名
    """  
    indu = DataAPI.MdSwBackGet(secID=code,field=u"secID,isNew,oldTypeName,industryName1,intoDate,outDate",pandas="1")
    indu['outDate'].fillna('2050-01-01', inplace=True)
    indu['intoDate'] = map(time_change, indu['intoDate'])
    indu['outDate'] = map(time_change, indu['outDate'])
    indu = indu[(indu['intoDate'] <= date) & (indu['outDate'] > date)]
    indu.drop_duplicates(subset=['secID'], inplace=True)
    indu['date'] = date
    indu = indu[['date', 'secID', 'industryName1']]
    indu.columns = ['date', 'code', 'industry']
    return indu

def get_indu_con_data(df, date, index_ticker):
    """
    Args:
        df: 月末股票池数据
        date: 月末时间
        index_ticker: 指数代码
    Returns:
        all_indu: 行业虚拟变量数据
    """ 
    # 行业暴露向量
    code = list(df['code'])
    indu = get_industry_end(date, code)
    indu.set_index('code', inplace=True)
    indu_dummies = pd.get_dummies(indu['industry'])
    H = indu_dummies.T
    # 指数行业暴露
    bench = DataAPI.IdxCloseWeightGet(ticker=index_ticker, beginDate=date, endDate=date, field=u"consID,weight",pandas="1")
    bench.columns = ['code', 'weight']
    bench['weight'] = bench['weight'] / 100
    indu_b = get_industry_end(date, list(bench['code']))
    data = pd.merge(bench, indu_b, on='code')
    index_indu = data.groupby(by='industry').sum()
    all_indu = pd.merge(H, index_indu, left_index=True, right_index=True, how='left').fillna(0)
    return all_indu

def get_stock_diff(df, date, index_ticker):
    """
    Args:
        df: 月末股票池数据
        date: 月末时间
        index_ticker: 指数代码
    Returns:
        w: 组合月基准指数的个股权重差异
    """ 
    bench = DataAPI.IdxCloseWeightGet(ticker=index_ticker, beginDate=date, endDate=date, field=u"consID,weight",pandas="1")
    bench.columns = ['code', 'weight']
    bench['weight'] = bench['weight'] / 100
    w = pd.merge(df[['code']], bench, on='code', how='left')
    w.fillna(0, inplace=True)
    return w

# 利用线性规划求解组合
# 为了应用多进程，我们将单期的组合优化函数的取数据与计算模块拆开
def prepare_optimal_data(df, date, index_ticker):
    """
    线性规划取数据函数
    Args：
        df：全体因子数据，DataFrame格式，列名为['date', 'code', 'compose', 'Month_ret']
        date：月末日期
        index_ticker：指数编号，比如沪深300就是'000300'
    Returns：
        r：预期收益向量
        tmp：市值约束及权重和为1约束数据
        indu：行业约束数据
        w：个股偏离度约束数据
    """ 
    factor = df[df['date'] == date]
    factor = factor.sort_values(by=['code'])
    factor.reset_index(drop=True, inplace=True)    
    # 预期收益向量
    r = np.array(factor['compose'])
    # 风格暴露约束（约束市值中性）
    size, sh = get_size_con_data(factor, date, index_ticker)
    size['cosntant'] = 1
    tmp = size.set_index('code').T
    tmp['weight'] = [sh, 1]  
    # 行业中性约束
    indu = get_indu_con_data(factor, date, index_ticker)    
    # 个股偏离度约束
    w = get_stock_diff(factor, date, index_ticker)    
    return r, tmp, indu, w

def single_period_allocation(arg):
    """
    组合优化函数
    Args：
        arg：参数集合，分别为tmp, indu, w, dev,具体如下
        r：预期收益向量
        tmp：市值约束及权重和为1约束数据
        indu：行业约束数据
        w_con：个股偏离度约束数据
        dev：个股偏离度
        date：日期
    Returns：
        w：每期组合权重数据
        res.success：每期优化状态
    """ 
    r, tmp, indu, w_con, dev, date = arg
    w = cvx.Variable(len(r))
    size = np.array(tmp.iloc[0, : -1]).reshape(1, tmp.shape[1] - 1)
    w_con['upper'] = w_con['weight'] + dev
    w_con['lower'] = np.where(w_con['weight'] - dev < 0, 0, w_con['weight'] - dev)    
    # 目标函数
    obj = cvx.Maximize(r.reshape(1, len(r)) * w)
    # 循环放宽约束    
    delta = 0.001
    indu_expo_init = 0.001
    for i in range(5): # 行业偏移上线暂时设定为0.005
        indu_expo = indu_expo_init + i * delta
        constraint = [cvx.sum(w) == 1.0,
                      size * w <= tmp['weight'][0],
                      size * w >= tmp['weight'][0],
                      np.array(indu[indu.columns[: -1]]) * w <= np.array(indu['weight'] + indu_expo),
                      np.array(indu[indu.columns[: -1]]) * w >= np.array(indu['weight'] - indu_expo),
                      w <= np.array(w_con['upper'].values),
                      w >= np.array(w_con['lower'].values)
                      ]
        prob = cvx.Problem(obj, constraint)
        prob.solve(solver=cvx.ECOS, verbose=False, max_iters=3000)
        if prob.status == 'optimal':
            w_con['optimal_weight'] = w.value    
            w_con['date'] = date
            w_con = w_con[['date', 'code', 'optimal_weight']]
            break
        if (prob.status != 'optimal') & (i == 4):
            break
    
    return w_con, dev, prob.status
    
def get_mkt_data(df, date, index_ticker):
    """
    Args：
        df：全体因子数据，DataFrame格式，列名为['date', 'code', 'compose', 'Month_ret']
        date：月末日期
        index_ticker：指数编号，比如沪深300就是'000300'
    Returns：
        pct：股票每日涨跌幅数据
        bench：基准涨跌幅序列数据
    """ 
    date_list = sorted(df['date'].unique())
    pre_date = date_list[date_list.index(date) - 2] # T-2
    factor = df[df['date'] == date]
    code = list(factor['code'])
    pct = DataAPI.MktEqudGet(secID=code,beginDate=pre_date,endDate=date,field=u"secID,tradeDate,chgPct",pandas="1")
    pct = pct[pct['secID'].str[0].isin(['0', '3', '6'])]
    pct.columns = ['code', 'date', 'ret']
    pct = pct.pivot(index='code', columns='date', values='ret')
    pct = pct[pct.columns[1: ]]
    pct.fillna(0,inplace=True)
    # 组合
    pct = (1 + pct).cumprod(axis=1)
    pct['code'] = pct.index
    # 基准
    bench = DataAPI.MktIdxdGet(ticker=index_ticker,beginDate=pre_date,endDate=date,field=u"tradeDate,closeIndex",pandas="1")
    bench['ret_bench'] = bench['closeIndex'] / bench['closeIndex'].shift(1) - 1
    bench = bench[1: ]
    bench.columns = ['date', 'close', 'ret_bench']
    return pct, bench

def cal_tracking_error(pct, bench, weight):
    """
    Args：
        pct：股票每日涨跌幅数据
        bench：基准涨跌幅序列数据
        weight：优化好的组合权重
    Returns：
        te：组合的年化跟踪误差
    """ 
    
    pct = pd.merge(pct, weight, on='code')
    days = [x for x in pct.columns if x not in ['date', 'code', 'optimal_weight']]
    capital = []
    for day in days:
        capital.append((pct[day] * pct['optimal_weight']).sum())
    portfolio = pd.DataFrame({'date': days, 'capital': capital})
    portfolio['temp'] = portfolio['capital'].shift(1).fillna(1)
    portfolio['ret'] = portfolio['capital'] / portfolio['temp'] - 1
    portfolio = pd.merge(portfolio, bench, on='date')
    portfolio['excess'] = portfolio['ret'] - portfolio['ret_bench']
    te = portfolio['excess'].std() * np.sqrt(252)
    return te

def portfolio_get(pickle_data, static_dev, target_te):
    """
    从pickle文件中提取信息
    Args：
        pickle_data：优化结果pickle文件
        static_dev：默认的静态的个股偏离度，str格式，例如'0.02'
        target_te： 预期跟踪误差上限，例如0.03
    Returns：
        static_port：静态组合，DataFrame，列名为日期，股票代码，优化权重
        dynamic_port：动态组合，DataFrame，列名为日期，股票代码，优化权重
    """
    date_list = sorted(pickle_data.keys())
    static_port = pd.DataFrame()
    dynamic_port = pd.DataFrame()
    for date in date_list:
        single_data = pickle_data[date]
        dev_list = sorted(single_data.keys())
        status = []
        tracking_error = []
        for dev in dev_list:
            status.append(single_data[dev]['status'])
            tracking_error.append(single_data[dev]['tracking_error'])
        df = pd.DataFrame({'dev': dev_list, 'status': status, 'tracking_error': tracking_error})
        df['date'] = date
        df = df[['date', 'dev', 'status', 'tracking_error']]
        # 静态组合
        static_port = static_port.append(single_data[static_dev]['weight'])
        # 动态组合
        temp = df[df['status'] == 'optimal']
        if float(temp['tracking_error'].min()) > target_te:
            dynamic_dev = temp['dev'].min()
        else:
            temp = temp[temp['tracking_error'] <= target_te]
            dynamic_dev = temp['dev'].max()
        dynamic_port = dynamic_port.append(single_data[dynamic_dev]['weight'])
    static_port.rename(columns={'code': 'secID'}, inplace=True)
    dynamic_port.rename(columns={'code': 'secID'}, inplace=True)
    static_port['date'] = static_port['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d').strftime('%Y-%m-%d'))
    dynamic_port['date'] = dynamic_port['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d').strftime('%Y-%m-%d'))
    return static_port, dynamic_port


begin_point = '20091231'
path = 'enhance_strategy_data/'
compose_orth_zero = pd.read_csv(path + 'compose_orth_zero.csv', dtype={"date": np.str})


# 多进程实现不同个股偏离度下的组合优化：沪深300增强组合
if __name__ == '__main__':
    tic = time.time()
    date_list = sorted(compose_orth_zero['date'].unique())
    date_list = [x for x in date_list if x >= begin_point]
    # 沪深300增强
    index_ticker = '000300'
    results = {}
    for date in date_list:
        # 数据准备
        r, tmp, indu, w = prepare_optimal_data(compose_orth_zero, date, index_ticker)
        # 偏离度集合下的多进程计算
        dev_list = list(np.linspace(0.005, 0.02, 16))
        arg_list = []
        for dev in dev_list:
            arg_list.append((r, tmp, indu, w, dev, date))
        pool = Pool(processes=16)
        res = pool.map(single_period_allocation, arg_list)
        pool.close()
        pool.join()
        pct, bench = get_mkt_data(compose_orth_zero, date, index_ticker)
        temp = {}
        for k in range(len(res)):
            weight = res[k][0]
            if 'optimal_weight' in list(weight.columns):
                te = cal_tracking_error(pct, bench, weight) # 跟踪误差计算
                weight = weight[weight['optimal_weight'] > 1e-8]
            else:
                te = np.nan
            temp[str(res[k][1])] = {"weight": weight, "status": res[k][2], "tracking_error": te}
        results[date] = temp

    # 存储
    with open(path + 'HS300_weight.pkl', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    toc = time.time()
    print ("\n ----- Computation time = " + str((toc - tic)) + "s")
    
    
# 多进程实现不同个股偏离度下的组合优化：中证500增强组合
if __name__ == '__main__':
    tic = time.time()
    date_list = sorted(compose_orth_zero['date'].unique())
    date_list = [x for x in date_list if x >= begin_point]
    # 中证500增强
    index_ticker = '000905'
    results = {}
    for date in date_list:
        # 数据准备
        r, tmp, indu, w = prepare_optimal_data(compose_orth_zero, date, index_ticker)
        # 偏离度集合下的多进程计算
        dev_list = list(np.linspace(0.001, 0.005, 5))
        arg_list = []
        for dev in dev_list:
            arg_list.append((r, tmp, indu, w, dev, date))
        pool = Pool(processes=16)
        res = pool.map(single_period_allocation, arg_list)
        pool.close()
        pool.join()
        pct, bench = get_mkt_data(compose_orth_zero, date, index_ticker)
        temp = {}
        for k in range(len(res)):
            weight = res[k][0]
            if 'optimal_weight' in list(weight.columns):
                te = cal_tracking_error(pct, bench, weight) # 跟踪误差计算
                weight = weight[weight['optimal_weight'] > 1e-8]
            else:
                te = np.nan
            temp[str(res[k][1])] = {"weight": weight, "status": res[k][2], "tracking_error": te}
        toc = time.time()
        results[date] = temp
        
    # 存储
    with open(path + 'ZZ500_weight.pkl', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    toc = time.time()
    print ("\n ----- Computation time = " + str((toc - tic)) + "s")
    
with open(path + 'HS300_weight.pkl') as f:
    pickle_300 = pickle.load(f)
with open(path + 'ZZ500_weight.pkl') as f:
    pickle_500 = pickle.load(f)

# 沪深300
static_dev = '0.02'
target_te = 0.03
static_port_300, dynamic_port_300 = portfolio_get(pickle_300, static_dev, target_te)
# 中证500
static_dev = '0.005'
target_te = 0.035
static_port_500, dynamic_port_500 = portfolio_get(pickle_500, static_dev, target_te)


# 跟踪误差说明，以静态中证500为例
factor = static_port_500.copy()
factor = factor.pivot_table(index='date', columns='secID', values='optimal_weight') # 静态中证500


# 静态中证500组合回测
start = '2009-12-31'                       # 回测起始时间
end = '2018-08-31'                         # 回测结束时间

benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('A')        # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                          # 调仓频率，表示执行handle_data的时间间隔

factor_dates = factor.index.values
  
commission = Commission(0.001, 0.001)     # 交易费率设为双边千分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in factor_dates:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    wts = pd.Series(dict(factor.ix[pre_date, account.universe].dropna()))
    wts = dict(wts)
        
    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        account.order_to(stk, 0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        account.order(stock, change[stock])
        
        
# 静态中证500增强组合的滚动三个月年化跟踪误差图
df = bt[['tradeDate', 'portfolio_value', 'benchmark_return']]
df['tradeDate'] = df['tradeDate'].apply(lambda x: x.strftime('%Y%m%d'))
df['return'] = df['portfolio_value'].pct_change()
df = df[1: ]
df['excess_return'] = df['return'] - df['benchmark_return']
date_list = sorted(results.keys())
actual_te = []
for i in range(3, len(date_list)):
    start = date_list[i - 3]
    end = date_list[i]
    temp =  df[(df['tradeDate'] >= start) & (df['tradeDate'] <= end)]
    actual_te.append(temp['excess_return'].std() * np.sqrt(252))
actual_te = pd.DataFrame({'date': date_list[3: ], 'actual_te': actual_te})
actual_te['date'] = pd.to_datetime(actual_te['date'])
ax = plt.plot(actual_te['date'], actual_te['actual_te'])

'''


跟踪误差约束结果
上图是上述中证500静态增强组合在个股偏离度为0.5%条件下的的实际滚动三个月年化跟踪误差，计算可知，组合总体的跟踪误差为4.45%，2011年的跟踪误差仅有2.93%，但2015年的跟踪误差却达到7.14%，因此在不同的市场状态下设置相同的跟踪误差约束参数会导致组合的实际跟踪误差动态变化，这会使得组合在大部分的时间内满足跟踪误差小于TE的约束，但是在极端行情下的风险控制就无法保证了。

   调试 运行
文档
 代码  策略  文档
3.2 基于自适应风险控制的指数增强模型

自适应风险控制
上述的静态模型在相同的约束下，在不同的市场环境下实现的跟踪误差是不尽相同的，静态的个股偏离度约束并不能完美地适应市场波动的变化。因此我们参考天风证券的做法，采取了一种自适应的跟踪误差约束方法，根据组合过去一段时间内以不同的个股权重偏离约束得到的组合实际跟踪误差与预期跟踪误差的关系来动态地自适应地确定每期调仓时的个股权重偏离度约束，具体而言：

在T月底建仓时，首先计算[T-3, T]月时间内以个股权重偏离度w_i优化得到的组合的年化跟踪误差TE_i；
对于给定的目标跟踪误差TE_target，找到满足TE_k <= TE_target的个股权重偏离度的最大值w_k作为T月底的个股权重偏离度约束条件。

上图是文中的静态中证500增强组合在0.1%-0.5%分五档个股权重偏离度约束下的实际年化跟踪误差图，给定的跟踪误差约束为3.5%，我们以上图分节点说明：

整体来看，个股权重偏离度越宽，则组合的实际跟踪误差越大；
在20131231时，以最大偏离0.2%的组合过去3个月的年化跟踪误差为3.06%，以0.3%为约束的组合过去3个月的年化跟踪误差为3.88%，因此在当期约束跟踪误差时，我们以0.2%作为个股权重最大偏离的约束来求解下一期组合；
在20150731时，以最大偏离0.1%的组合过去3个月的年化跟踪误差为3.76%，其他约束下的跟踪误差都高于4%，因此在当期我们以0.1%作为个股权重最大偏离约束；
在20170630时，以0.5%为约束的组合过去3个月的年化跟踪误差为3.39%，其他约束下的跟踪误差都低于3%，因此在当期我们以0.5%作为个股权重最大偏离约束。
   调试 运行
文档
 代码  策略  文档
3.3 组合回测及分析

'''


with open(path + 'HS300_weight.pkl') as f:
    pickle_300 = pickle.load(f)
with open(path + 'ZZ500_weight.pkl') as f:
    pickle_500 = pickle.load(f)

# 沪深300
static_dev = '0.02'
target_te = 0.03
static_port_300, dynamic_port_300 = portfolio_get(pickle_300, static_dev, target_te)
# 中证500
static_dev = '0.005'
target_te = 0.035
static_port_500, dynamic_port_500 = portfolio_get(pickle_500, static_dev, target_te)


def plot_under_water(bt, title):
    """
    绘制回撤及收益率曲线图，输出策略指标
    输入：
        bt：quartz回测结束自动生成的dict
        title：str
    返回：
        ax：matplotlib figure 对象
        df_ratio：策略指标
    """
    bt_quantile_ten = bt.copy()
    data = bt_quantile_ten[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()
    # 指标计算
    df = data.copy()
    df = df[['tradeDate', 'excess_return', 'excess']]
    df.columns = ['tradeDate', 'rtn', 'capital']
    df['tradeDate'] = df['tradeDate'].apply(lambda x: x.strftime('%Y%m%d'))
    df.sort_values(by='tradeDate', inplace=True)
    df.reset_index(drop=True, inplace=True)
    annual = pow((df.ix[len(df.index) - 1, 'capital']) / df.ix[0, 'capital'], 250.0 / len(df)) - 1 # 年化收益
    volatility = df['rtn'].std() * np.sqrt(250)
    df['max2here'] = df['capital'].expanding(min_periods=1).max()
    df['dd2here'] = df['capital'] / df['max2here'] - 1
    temp = df.sort_values(by='dd2here').iloc[0][['tradeDate', 'dd2here']]
    max_dd = temp['dd2here'] # 最大回撤
    end_date = temp['tradeDate']
    df = df[df['tradeDate'] <= end_date]
    start_date = df.sort_values(by='capital', ascending=False).iloc[0]['tradeDate']
    sharpe = annual / volatility # 夏普比率
    rtn_ratio = annual / np.abs(max_dd) # 收益回撤比
    df_ratio = pd.DataFrame({u'策略': [title], u'年化超额收益': [annual], u'相对最大回撤': [max_dd], u'收益回撤比': rtn_ratio,
                             u'最大回撤起始': start_date, u'最大回撤结束': end_date, u'跟踪误差': volatility, u'夏普比率': sharpe})
    # 画图
    df_cum_rets = data['excess']
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -((running_max - df_cum_rets) / running_max)
    underwater.index = data['tradeDate']

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    x = range(len(underwater))
    ax2.grid(False)
    ax1.set_ylim(-0.30, 0)
    ax1.set_ylabel(u'回撤', fontproperties=font, fontsize=16)
    ax1.fill_between(underwater.index, 0, np.array(underwater), color='#000066', alpha=1)
    ax2.set_ylabel(u'净值', fontproperties=font, fontsize=16)
    ax2.plot(data['tradeDate'], data[['excess']], label='hedged(right)', color='r')
    ax2.set_ylim(bottom=0.9, top=5)
    s = ax1.set_title(title, fontproperties=font, fontsize=16)
    return fig, df_ratio

# 将数据处理成优矿回测所需格式
factor1 = static_port_300.copy()
factor1 = factor1.pivot_table(index='date', columns='secID', values='optimal_weight') # 静态沪深300
factor2 = dynamic_port_300.copy()
factor2 = factor2.pivot_table(index='date', columns='secID', values='optimal_weight') # 动态沪深300
factor3 = static_port_500.copy()
factor3 = factor3.pivot_table(index='date', columns='secID', values='optimal_weight') # 静态中证500
factor4 = dynamic_port_500.copy()
factor4 = factor4.pivot_table(index='date', columns='secID', values='optimal_weight') # 动态中证500


# 静态沪深300组合回测
start = '2009-12-31'                       # 回测起始时间
end = '2018-08-31'                         # 回测结束时间

benchmark = 'HS300'                        # 策略参考标准
universe = DynamicUniverse('A')        # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                          # 调仓频率，表示执行handle_data的时间间隔

factor_dates = factor1.index.values
  
commission = Commission(0.001, 0.001)     # 交易费率设为双边千分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in factor_dates:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    wts = pd.Series(dict(factor1.ix[pre_date, account.universe].dropna()))
    wts = dict(wts)
    
    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        account.order_to(stk, 0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        account.order(stock, change[stock])
        
        


bt1 = bt.copy()
bt1.to_csv(path + 'static_300.csv', index=False)
figure1, ratio1 = plot_under_water(bt1, u'静态沪深300')

# 动态沪深300组合回测，跟踪误差上限设置为3%
start = '2009-12-31'                       # 回测起始时间
end = '2018-08-31'                         # 回测结束时间

benchmark = 'HS300'                        # 策略参考标准
universe = DynamicUniverse('A')        # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                          # 调仓频率，表示执行handle_data的时间间隔

factor_dates = factor2.index.values
  
commission = Commission(0.001, 0.001)     # 交易费率设为双边千分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in factor_dates:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    wts = pd.Series(dict(factor2.ix[pre_date, account.universe].dropna()))
    wts = dict(wts)

    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        account.order_to(stk, 0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        account.order(stock, change[stock])
        
        

'''
由此，我们基于静态方法与自适应风险控制方法分别对沪深300、中证500指数构建指数增强组合，组合回测结果如下

'''

bt2 = bt.copy()
bt2.to_csv(path + 'dynamic_300.csv', index=False)
figure2, ratio2 = plot_under_water(bt2, u'动态沪深300')

# 静态中证500组合回测
start = '2009-12-31'                       # 回测起始时间
end = '2018-08-31'                         # 回测结束时间

benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('A')        # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                          # 调仓频率，表示执行handle_data的时间间隔

factor_dates = factor3.index.values
  
commission = Commission(0.001, 0.001)     # 交易费率设为双边千分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in factor_dates:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    wts = pd.Series(dict(factor3.ix[pre_date, account.universe].dropna()))
    wts = dict(wts)

    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        account.order_to(stk, 0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        account.order(stock, change[stock])
        
        
bt3 = bt.copy()
bt3.to_csv(path + 'static_500.csv', index=False)
figure3, ratio3 = plot_under_water(bt3, u'静态中证500')


# 动态中证500组合回测，跟踪误差上限设置为3.5%
start = '2009-12-31'                       # 回测起始时间
end = '2018-08-31'                         # 回测结束时间

benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('A')        # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                          # 调仓频率，表示执行handle_data的时间间隔

factor_dates = factor4.index.values
  
commission = Commission(0.001, 0.001)     # 交易费率设为双边千分之二

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in factor_dates:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    wts = pd.Series(dict(factor4.ix[pre_date, account.universe].dropna()))
    wts = dict(wts)

    # 交易部分
    sell_list = [stk for stk in account.security_position if stk not in wts]
    for stk in sell_list:
        account.order_to(stk, 0)

    c = account.reference_portfolio_value
    change = {}
    for stock, w in wts.iteritems():
        p = account.reference_price[stock]
        if not np.isnan(p) and p > 0:
            change[stock] = int(c * w / p) - account.security_position.get(stock, 0)

    for stock in sorted(change, key=change.get):
        account.order(stock, change[stock])
        
bt4 = bt.copy()
bt4.to_csv(path + 'dynamic_500.csv', index=False)
figure4, ratio4 = plot_under_water(bt4, u'动态中证500')

# 回测结果整合
ratio = pd.concat([ratio1, ratio2, ratio3, ratio4], axis=0)
ratio = ratio[[u'策略', u'年化超额收益', u'跟踪误差', u'夏普比率', u'收益回撤比', u'相对最大回撤', u'最大回撤起始', u'最大回撤结束']]
print('***********组合结果对比************')
print(ratio.round(4).to_html())

'''
结论

由上述指数增强组合的表现得到如下结论：

在自适应风险控制下，沪深300指数增强组合的收益略微下降，年化超额收益从10.7%降至8.2%，但是跟踪误差从原先的4.55%降至3.31%，总体的跟踪误差较好地约束在目标跟踪误差范围左右（目标跟踪误差3%），组合的夏普比率由2.36上升至2.48，相对基准的最大回撤大幅度降低（7.36% -> 4.01%），而且我们从净值曲线上可以看到，自适应风险控制的组合在任何市场行情中都非常稳健；
同理，在自适应风险控制下，中证500指数增强组合的收益也是下降的，这是强化风险约束带来的必然结果，但是跟踪误差从原先的4.44%降至3.32%，能够完全地约束在目标跟踪误差3.5%的范围内，同时相对基准的最大回撤也降低了（3.53% -> 3.23%），控制了在诸如2015年极端行情下的跟踪误差，自适应风险控制组合能适应任何行情并获取稳定超额收益；
因此，自适应的风险控制约束能有效地控制组合的风险，在牺牲一小部分收益的情况下大幅度提升组合的稳健性，更好地适应各种市场风格。

'''

