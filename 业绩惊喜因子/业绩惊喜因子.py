# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:31:27 2020

@author: Asus
"""

'''

1. 导读
研究目的：

1968年，Ball和Brown在美国市场首次发现了盈余公告后的价格偏移现象，即对于盈利超预期（Positive Eanings Surprise）的股票在公告后有持续正向的异常收益，相反，盈利低于预期的股票后期有持续负向的异常收益。自Ball 和Brown(1968)之后，大量研究学者在不同的国家、采用不同的研究或度量方法均发现了类似的结论，早期的研究主要集中于对净利润的研究，Ertimur 等（2003）和Jegadeesh 等（2006）的研究则将关注点从净利润转移到营业收入上，结果表明营业收入的预期外部分在公告后也有显著的异常收益，而且这部分异常收益并不能被盈利的预期外部分解释，说明营业收入相对于盈利有额外的信息。
本文结合优矿底层提供的相关财务数据，参考了东方证券《业绩超预期类因子——因子选股系列研究之三十九》，计算了业绩惊喜相关因子，并对相关因子进行了测试。
研究结论：

本文对业绩惊喜相关因子数据进行了测试，从单因子分析的结果来看，财报公告后，业绩超预期的股票有着显著的正向超额收益，而不及预期的股票有着负向的收益。
同时，从IC均值，多空收益来看，基于净利润的业绩惊喜因子要好于基于营业收入的业绩惊喜因子。
四个因子的原始值有着明显的选股效果，但中性化后的因子选股稳定性更高，预测能力也更强。
本文对业绩惊喜因子进行了信息增量分析，研究发现，剔除了其他大类因子后，业绩惊喜因子仍具有不错的选股能力，而成长因子在剥离了业绩惊喜因子后，几乎没有选股效果，该因子可以作为成长因子的替代
文章结构：

数据及工具函数的准备：该部分主要是一些工具函数的编写，包括交易日历的获取，因子的计算，因子分析的代码以及分析报告的绘图展示等等。（45分钟）
业绩惊喜因子介绍：介绍了本文4个业绩惊喜因子的计算逻辑和计算方式
单因子有效性分析：从因子的角度来分析业绩惊喜因子能否带来超额收益 （<5分钟）
信息增量分析：分析业绩惊喜因子与大类因子之间的相关性，与成长因子的替代性 （<5分钟）
指数增强策略：比较了使用业绩惊喜因子与成长因子的指数增强策略的业绩表现 (10分钟)
总结：根据分析的结果，对业绩惊喜因子进行总结
   调试 运行
文档
 代码  策略  文档
1. 数据及工具函数的准备
该部分，我们主要进行数据的预加载以及处理。包括因子值的获取和处理，工具函数的编写等等。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


'''

import pandas as pd
import numpy as np
import scipy.stats as st
import gevent
import pandas as pd
import numpy as np
import os
import time
import multiprocessing
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from sqlalchemy import create_engine
import json


#from CAL.PyCAL import font

path = "./surprise_data/"
if not os.path.exists(path):
    os.mkdir(path)
sns.set_style('ticks')

# must be set before using
with open('para.json', 'r', encoding='utf-8') as f:
    para = json.load(f)

pn = para['yuqerdata_dir']

user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name1 = 'yuqerdata'
db_name2 = 'yuqer_cubdata'
eng_str = 'mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name, pass_wd, port, db_name1)
eng_str2 = 'mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name, pass_wd, port, db_name2)
engine = create_engine(eng_str)
engine2 = create_engine(eng_str2)

sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''


##前复权需要换为后复权
def chs_factor(ticker='000005', begin=None, end=None,
               field=[u'symbol', u'tradeDate', u'openPrice',
                      u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol',
                      u'turnoverValue', u'dealAmount', u'chgPct',
                      'turnoverRate', u'marketValue', u'accumAdjFactor']):
    sql_str1 = sql_str_select_data1 % (','.join(field), ticker, begin, end)
    dataday = pd.read_sql(sql_str1, engine)
    dataday = dataday.applymap(lambda x: np.nan if x == 0 else x)
    dataday.rename(columns={'symbol': 'ticker'}, inplace=True)
    ## 对数据补全
    return dataday.fillna(method='ffill')


def get_industry_class(t):
    sql_str1 = '''select ticker,industryID1 from yuqerdata.yq_industry where 
                industryVersionCD="010303" and intodate <= "%s" and 
                (outDate>"%s" or outDate is null)''' % (t, t)
    x = pd.read_sql(sql_str1, engine)
    return x


def get_month_calender():
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" order by endDate'''
    x = pd.read_sql(sql_str, engine)
    x = x['endDate'].values
    # b=[i.strftime('%Y-%m-%d') for i in x]
    return x

def get_month_calender_range(beginDate, endDate):
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" and endDate >="%s" and endDate<="%s" order by endDate''' %(beginDate, endDate)
    x = pd.read_sql(sql_str, engine)
    x = x['endDate'].values
    # b=[i.strftime('%Y-%m-%d') for i in x]
    return x
def get_calender():
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" order by tradeDate'''
    x = pd.read_sql(sql_str, engine)
    x = x['tradeDate'].values
    # b=[i.strftime('%Y-%m-%d') for i in x]
    return x


cal = get_calender()


def get_IdxCons(intoDate, ticker='000300'):
    # nearst 时间
    sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
            and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
        ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' % (ticker,
                                                                                    ticker, intoDate)
    x = pd.read_sql(sql_str1, engine)
    x = x['symbol'].values
    return x


def MktStockFactorsOneDayProGet(tradeDate, fields):
    df = pd.DataFrame()
    i = 1
    for field in fields:
        field = field.lower()
        sql_str1 = '''select symbol, tradingdate, f_val from ''' + field
        sql_str2 = ''' where tradingdate = "%s"''' % (tradeDate)
        sql_str = sql_str1 + sql_str2
        # sql_str = '''select symbol, tradingdate, f_val from '%s' where tradingdate = "%s"'''%(field, tradeDate)
        x = pd.read_sql(sql_str, engine2)
        x = x.rename(columns={"f_val": field})
        if i == 1:
            df = x
        else:
            df = df.merge(x, on=["symbol", "tradingdate"])
        i = i + 1

    return df


# from quartz.api import *

###############
# 计算贡献度函数
###############
def get_index_data(index, begin, end):
    r_m = pd.read_sql(
        '''select * from yq_index where indexID = "%s" and tradeDate>="%s" and tradeDate<="%s" order by tradeDate''' % (
        index, begin, end), engine)
    return r_m


'''
1.1 数据准备

本文需要使用行情数据，优矿因子库相关因子数据，以及优矿整理过的单季度PIT利润表数据
优矿因子库因子数据获取:
    
'''

def get_factor_by_day(args):
    """
    根据日期，获取当天的因子值
    输入：
        args：tuple:(list, str)，第一个元素为因子名称列表，第二个元素为'YYYYMMDD'格式的字符串，表示时间
    返回：
        DataFrame：因子数据
    """
    factor_names, date = args
    cnt = 0
    while True:
        try:
            x = MktStockFactorsOneDayProGet(tradeDate=date,fields= factor_names)
            return x
        except Exception as e:
            print(e)
            cnt += 1
            if cnt >= 3:
                print('error get factor data: ', date)
                break

def get_multi_factor(factor_names, trade_date_list):
    """
    取多个因子数据
    输入：
        factor_names: list, 因子名
        trade_date_list: list, 交易日列表
    返回：
        DataFrame：记录多个因子数据
    """
    pool = ThreadPool(processes=4)
    factor_names = [factor_names] * len(trade_date_list)
    frame_list = pool.map(get_factor_by_day, zip(factor_names, trade_date_list))
    pool.close()
    pool.join()
    factor = pd.concat(frame_list, axis=0)
    #print(factor)
    factor['tradingdate'] = factor['tradingdate'].str.replace('-', '')
    return factor

def winsorize_by_date(cdate_input):
    '''
    按照[dm+5*dm1, dm-5*dm1]进行winsorize
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

def nafill_by_sw1(cdate_input):
    """
    缺失值填充，使用用申万一级行业中位数
    输入：
        cdate_input: 因子值，DataFrame
    返回：
        DataFrame, 填充缺失值后的因子值
    """
    func_input = cdate_input.copy()
    func_input = func_input.merge(sw_map_frame[['secID', 'industryName1']], on=['secID'], how='left')

    func_input.loc[:, factor_name] = func_input.loc[:, factor_name].fillna(
        func_input.groupby('industryName1')[factor_name].transform("median"))

    return func_input.fillna(0.0)

def winsorize_fillna_date(tdate):
    """
    对某一天的数据进行去极值，填充缺失值
    输入：
        tdate： str， 'YYYYMMDD'格式
    返回：
        DataFrame, 去极值，填充缺失值后的因子值
    """
    cnt = 0
    while True:
        try:
            # 缺失值填充, 用同行业的均值
            cdate_input = input_frame[input_frame.tradeDate == tdate]
            cdate_input = nafill_by_sw1(cdate_input)
            cdate_input.set_index('secID', inplace=True)
            # print("####Running single_date for %s" % tdate)
            # winsorize
            cdate_input = winsorize_by_date(cdate_input)
            return cdate_input
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                cdate_input = input_frame[input_frame.tradeDate == tdate]
                # 缺失值填充, 用同行业的均值
                cdate_input = nafill_by_sw1(cdate_input)
                cdate_input.set_index('secID', inplace=True)
                return cdate_input
            
def standardize(x):
    return (x - x.mean()) / x.std()


def standardize_neutralize_factor(input_data):
    """
    进行中性化和标准化
    输入：
        input_data：tuple, 传入的是(因子值，时间)。因子值为DataFrame
    返回：
        DataFrame, 行业标准化后的因子值
    """
    cdate_input, tdate = input_data
    for a_factor in factor_name + ['sue0', 'sue1', 'sur0', 'sur1', 'sales_growth_yoy', 'profit_growth_yoy']:
        cnt = 0
        while True:
            try:
                cdate_input.loc[:, a_factor] = standardize(neutralize(cdate_input.loc[:, a_factor],
                                                                      target_date=tdate, 
                                                                      exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 
                                                                                          'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'SIZENL']))
                break
            except Exception as e:
                cnt += 1
                if cnt >= 3:
                    break

    return cdate_input

def standardize_factor(input_data):
    """
    进行行业内标准化
    输入：
        input_data：tuple, 传入的是(因子值，时间)。因子值为DataFrame
    返回：
        DataFrame, 行业标准化后的因子值
    """
    cdate_input, tdate = input_data
    for a_factor in factor_name:
        cnt = 0
        while True:
            try:
                cdate_input.loc[:, a_factor] =standardize(cdate_input[a_factor])
                break
            except Exception as e:
                cnt += 1
                if cnt >= 3:
                    break

    return cdate_input

if __name__ == "__main__":
    # 设置起始时间和结束时间
    begin_date = '20070101'
    end_date = '20180630'
    #factor_name = ['OperatingRevenueGrowRate', 'NetProfitGrowRate', 'PE', 'PB', 'PS', 'PCF', 'CETOP',
    #               'ROE', 'ROA', 'EPS', 'ROIC', 'GrossIncomeRatio', 'VOL20', 'DAVOL20', 'ILLIQUIDITY',
    #               'REVS20', 'REVS60']

    factor_name = [ 'PE', 'PB', 'PS', 'PCF', 'CETOP',
                   'ROE', 'ROA', 'EPS', 'ROIC', 'GrossIncomeRatio', 'VOL10', 'DAVOL20',
                   'REVS20', 'REVS60']

    trade_date_list = get_month_calender_range(begin_date, end_date)
    #trade_date_list = trade_date_list[trade_date_list['isMonthEnd'] == 1]['calendarDate'].tolist()
    
    # 因子数据的获取
    input_frame = get_multi_factor(factor_name, trade_date_list)

    # 部分因子方向的调整
    input_frame['pb'] = 1. / input_frame['pb']
    input_frame['pe'] = 1. / input_frame['pe']
    input_frame['ps'] = 1. / input_frame['ps']
    input_frame['pcf'] = 1. / input_frame['pcf']
    input_frame['revs20'] = -1 * input_frame['revs20']
    input_frame['revs60'] = -1 * input_frame['revs60']
    input_frame['davol20'] = -1 * input_frame['davol20']
    input_frame['vol10'] = -1 * input_frame['vol10']
    
class SurpriseFactor(object):
    """
    计算业绩惊喜相关因子
    """

    def __init__(self, income_statement, end_date_list):
        self.income_statement = income_statement
        self.end_date_list = end_date_list
    
    @classmethod
    def cal_signal(cls, df, columns, end_date_list, shift=True):
        df1 = df.copy()
        df1.sort_values(by=['publishDate', 'endDate'], ascending=False, inplace=True)
        df2 = df.set_index('publishDate')
        date_list = df1['publishDate'].unique()
        date_list.sort()

        for date in sorted(date_list):
            tmp = df1[df1.publishDate <= date]

            tmp.drop_duplicates(subset=['secID', 'endDate'], inplace=True, keep='first')

            tmp = tmp.sort_values(by='endDate', ascending=False).set_index('endDate')
            report_end_date = tmp.index[0]
            report_date_list = end_date_list[end_date_list <= report_end_date][-13:][::-1]
            tmp = tmp.reindex(report_date_list).head(13)
            tmp[columns] = tmp[columns].diff(-4)
            sigma = tmp[columns][1:].std() if len(tmp.dropna()) >= 4 else np.NaN
            if shift:
                df2.loc[date, 'signal'] = (tmp[columns].iloc[0] - tmp[columns].iloc[1:].mean()) / sigma
            else:
                df2.loc[date, 'signal'] = tmp[columns].iloc[0] / sigma
        df2 = df2.reset_index()
        return df2

    @classmethod
    def cal_yoy_signal(cls, df, columns, end_date_list):
        df1 = df.copy()
        df1.sort_values(by=['publishDate', 'endDate'], ascending=False, inplace=True)
        df2 = df.set_index('publishDate')
        date_list = df1['publishDate'].unique()
        date_list.sort()

        for date in sorted(date_list):
            tmp = df1[df1.publishDate <= date]
            tmp.drop_duplicates(subset=['secID', 'endDate'], inplace=True, keep='first')

            tmp = tmp.sort_values(by='endDate', ascending=False).set_index('endDate')
            report_end_date = tmp.index[0]
            report_date_list = end_date_list[end_date_list <= report_end_date][-5:][::-1]
            tmp = tmp.reindex(report_date_list).head(5)
            df2.loc[date, 'signal'] = (tmp[columns].iloc[0] - tmp[columns].iloc[-1]) / np.abs(tmp[columns].iloc[-1])
        df2 = df2.reset_index()
        return df2

    def cal_sue0(self):
        """
        带漂移项的净利润业绩惊喜因子
        :return: DataFrame, 公告日发布后计算的因子数据
        """
        sue0 = self.income_statement.groupby(by='secID').apply(lambda x: SurpriseFactor.cal_signal(x,
                                                                                                   'NIncomeAttrP',
                                                                                                   self.end_date_list,
                                                                                                   True))
        sue0 = sue0.drop('secID', axis=1).reset_index()
        sue0.drop_duplicates(subset=['secID', 'publishDate'], inplace=True, keep='first')
        sue0 = sue0[['secID', 'publishDate', 'signal']].dropna()
        return sue0

    def cal_sue1(self):
        """
        不带漂移项的营业收入业绩惊喜因子
        :return: DataFrame, 公告日发布后计算的因子数据
        """
        sue1 = self.income_statement.groupby(by='secID').apply(lambda x: SurpriseFactor.cal_signal(x,
                                                                                                   'NIncomeAttrP',
                                                                                                   self.end_date_list,
                                                                                                   False))
        sue1 = sue1.drop('secID', axis=1).reset_index()
        sue1.drop_duplicates(subset=['secID', 'publishDate'], inplace=True, keep='first')
        sue1 = sue1[['secID', 'publishDate', 'signal']].dropna()
        return sue1

    def cal_sur0(self):
        """
        带漂移项的营业收入业绩惊喜因子
        :return: DataFrame, 公告日发布后计算的因子数据
        """

        sur0 = self.income_statement.groupby(by='secID').apply(lambda x: SurpriseFactor.cal_signal(x,
                                                                                                   'revenue',
                                                                                                   self.end_date_list,
                                                                                                   True))
        sur0 = sur0.drop('secID', axis=1).reset_index()
        sur0.drop_duplicates(subset=['secID', 'publishDate'], inplace=True, keep='first')
        sur0 = sur0[['secID', 'publishDate', 'signal']].dropna()
        return sur0

    def cal_sur1(self):
        """
        带漂移项的净利润业绩惊喜因子
        :return: DataFrame, 公告日发布后计算的因子数据
        """

        sur1 = self.income_statement.groupby(by='secID').apply(lambda x: SurpriseFactor.cal_signal(x,
                                                                                                   'revenue',
                                                                                                   self.end_date_list,
                                                                                                   False))
        sur1 = sur1.drop('secID', axis=1).reset_index()
        sur1.drop_duplicates(subset=['secID', 'publishDate'], inplace=True, keep='first')
        sur1 = sur1[['secID', 'publishDate', 'signal']].dropna()
        return sur1

    def cal_profit_growth_yoy(self):
        """
        计算净利润增长率单季度同比因子
        :return:
        """
        profit_growth_yoy = self.income_statement.groupby(by='secID').apply(lambda x: SurpriseFactor.cal_yoy_signal(x,
                                                                                                   'NIncomeAttrP',
                                                                                                   self.end_date_list))
        profit_growth_yoy.drop_duplicates(subset=['secID', 'publishDate'], inplace=True, keep='first')
        profit_growth_yoy = profit_growth_yoy[['secID', 'publishDate', 'signal']].dropna()
        profit_growth_yoy.reset_index(drop=True, inplace=True)
        return profit_growth_yoy

    def cal_sales_growth_yoy(self):
        """
        计算营业收入增长率单季度同比因子
        :return:
        """
        sales_growth_yoy = self.income_statement.groupby(by='secID').apply(lambda x: SurpriseFactor.cal_yoy_signal(x,
                                                                                                   'revenue',
                                                                                                   self.end_date_list))
        sales_growth_yoy.drop_duplicates(subset=['secID', 'publishDate'], inplace=True, keep='first')
        sales_growth_yoy = sales_growth_yoy[['secID', 'publishDate', 'signal']].dropna()
        sales_growth_yoy.reset_index(drop=True, inplace=True)
        return sales_growth_yoy
    
    
if __name__ == '__main__':
    data = DataAPI.FdmtISQPITGet(field=u"secID,publishDate,endDate,NIncomeAttrP,revenue",pandas="1")
    data = data[data['secID'].str[0].isin(['0', '3', '6'])]
    data.to_csv('income_statement.csv')
    data = pd.read_csv('income_statement.csv', index_col=0)
    
    date_list = np.array(sorted(data['endDate'].unique()))
    surprise_factor = SurpriseFactor(data, date_list)

    sue0 = surprise_factor.cal_sue0()
    sue0.to_csv(path + 'sue0.csv')

    sue1 = surprise_factor.cal_sue1()
    sue1.to_csv(path + 'sue1.csv')

    sur0 = surprise_factor.cal_sur0()
    sur0.to_csv(path + 'sur0.csv')
    
    sur1 = surprise_factor.cal_sur1()
    sur1.to_csv(path + 'sur1.csv')
    
    profit_growth_yoy = surprise_factor.cal_profit_growth_yoy()
    profit_growth_yoy.to_csv(path + 'profit_growth_yoy.csv')
    
    sales_growth_yoy = surprise_factor.cal_sales_growth_yoy()
    sales_growth_yoy.to_csv(path + 'sales_growth_yoy.csv')
    
def get_dates(start_date, end_date, frequency='daily'):
    """
    输入起始日期和频率，即可获得日期列表（daily包括起始日，其余的都是位于起始日中间的）
    输入：
       start_date，开始日期，'YYYYMMDD'形式
       end_date，截止日期，'YYYYMMDD'形式
       frequency，频率，daily为所有交易日，weekly为每周最后一个交易日，monthly为每月最后一个交易日，quarterly为每季最后一个交易日
    返回：
       获得list型日期列表，以'YYYYMMDD'形式存储
    """
    data = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date,
                               field=u"calendarDate,isOpen,isWeekEnd,isMonthEnd,isQuarterEnd", pandas="1")
    if frequency == 'daily':
        data = data[data['isOpen'] == 1]
    elif frequency == 'weekly':
        data = data[data['isWeekEnd'] == 1]
    elif frequency == 'monthly':
        data = data[data['isMonthEnd'] == 1]
    elif frequency == 'quarterly':
        data = data[data['isQuarterEnd'] == 1]
    else:
        raise ValueError('调仓频率必须为daily/daily1/weekly/weekly2/monthly/quarterly！！！')
    date_list = map(lambda x: x[0:4] + x[5:7] + x[8:10], data['calendarDate'].values.tolist())
    return date_list

def shift_date(date, n, direction='back'):
    """
    日期平移函数，获取date向前/向后移动Ndays个交易日所对应的交易日
    输入：
        date: str， 'YYYYMMDD'各式
        n: int，长度不要超过700
        direction:str, 可选'back'或者'forward'
    返回：
        date：str，'YYYYMMDD'格式
    """
    last_two_year = str(int(date[:4]) - 2) + '0101'
    forward_two_year = str(int(date[:4]) + 2) + '1231'
    if direction == 'back':
        date_list = get_dates(last_two_year, date, 'daily')
        return date_list[len(date_list) - 1 - n]
    elif direction == 'forward':
        date_list = get_dates(date, forward_two_year, 'daily')
        if len(date_list) <= n:  # 当未来数据长度不够用时，返回最后一个能取到的交易日
            return date_list[-1]
        else:
            return date_list[n]
    else:
        raise ValueError('direction should be back/forward！！！')
        
def get_st_tickers(date):
    """
    获取历史上某一天的ST股票
    输入:
        date: str, 'YYYYMMDD'格式
    返回：
        list: 元素为股票ticker
    """
    data = DataAPI.SecSTGet(beginDate=date, endDate=date, field='')
    return data['secID'].tolist()

def get_all_tickers(date):
    """
    给定日期，获取这一天上市时间不低于60天的股票（参照中证全指指数编制）
    输入：
        date: str， 'YYYYMMDD'格式
    返回：
        list： 元素为股票ticker
    """
    universe = DataAPI.EquGet(equTypeCD=u"A", listStatusCD="L,S,DE", field=u"secID,listDate,delistDate")
    universe['listdate'] = universe['listDate'].apply(lambda x: x.replace('-', ''))
    universe['delistdate'] = universe['delistDate'].apply(
        lambda x: x.replace('-', '') if isinstance(x, unicode) else '99999999')
    list_d_need = shift_date(date, 60, 'back')
    universe = universe[(universe['listdate'] <= list_d_need) & (universe['delistdate'] > date)]
    tickers = list(set(universe['secID'].tolist()) - set(get_st_tickers(date)))
    return tickers

def get_idx_cons(idx, date):
    """
    获取某天指数成分股ticker列表
    输入:
        idx:str，指数代码
        date:str，'YYYYMMDD'格式
    返回：
        list:指数成份股的ticker
    """
    if idx != 'A':
        try:
            data = DataAPI.IdxConsGet(ticker=idx, intoDate=date, field='', pandas="1")['consID']
        except Exception as e:
            raise ValueError(u'DataAPI.IdxConsGet出错了！！！: %s' % e)
        if len(data) < 50:
            raise ValueError('{0}该日指数成分股API取出来的成分股数不足50个！！！'.format(date))
    else:
        universe = get_all_tickers(date)
        st_stk = get_st_tickers(date)
        return list(set(universe) - set(st_stk))
    return list(set(data))

def pretreat_factor(factor_df, neu=True):
    """
    去极值，中性化，标准化
    """
    pretreat_data = factor_df.copy(deep=True)
    for dt in pretreat_data.index:
        try:
            factor_dt = pretreat_data.ix[dt].dropna()
            factor_dt_dict = factor_dt.to_dict()
            if neu:
                pretreat_data.ix[dt] = pd.Series(standardize(neutralize(winsorize(factor_df.ix[dt].to_dict()), target_date=''.join(dt.split('-')), industry_type='SW1',                                                             exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'SIZENL', 'GROWTH', 'LEVERAGE', 'LIQUIDTY'])))
            else:
                pretreat_data.ix[dt] = pd.Series(standardize(winsorize(factor_dt_dict)))
        except Exception as excp:
            print (dt)
            print (excp)
            continue
    return pretreat_data


def fill_surprise_factor(df, name):
    """
    处理业绩增长相关因子数据的格式
    """
    df = df.pivot(index='publishDate', columns='secID', values='signal').loc[trade_date_list, :].fillna(method='ffill').loc[month_date_list, :].unstack().reset_index()
    df.columns = ['secID', 'publishDate', name]
    return df

def select_universe(x, universe):
    try:
        return x[x['secID'].isin(universe.loc[x['tradeDate'].iloc[0], :].dropna().tolist())]
    except Exception as e:
        print (x)
        raise Exception(e)
        
        
       
def bar_plot_wrapper(func):   
    def warpper(*args, **kwargs):
        ax=func(*args, **kwargs)  
        ax.spines['right']
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data', 0))
        sns.despine()
        return ax
    return warpper

def line_plot_wrapper(func):   
    def warpper(*args,**kwargs):
        ax=func(*args,**kwargs)  
        ax.spines['right']
        ax.spines['top'].set_color('none')
        sns.despine()
        return ax
    return warpper

@bar_plot_wrapper
def barplot(data, ax):
    """
    条形图绘制函数
    """
    return data.plot(kind='bar', ax=ax, width=0.3, alpha=0.9)

@line_plot_wrapper
def lineplot(data, ax):
    """
    线图绘制函数
    """
    return data.plot(ax=ax, color='r')


class FactorTest:
    def __init__(self):
        self.hs300_cons = hs300_cons.copy(deep=True)
        self.zz500_cons = zz500_cons.copy(deep=True)
        self.all_a = a_cons.copy(deep=True)
        self.return_df = forward_20d_return_data.copy(deep=True)

    def _get_universe_factor(self, factor, idx=None):
        """
        筛选出某指数成份股或者指定域内的因子值
        输入：
            factor:DataFrame，index为日期，columns为股票代码，value为因子值
            idx:str，成份股简称，目前只支持hs300,zz500,all_a
        返回：
            factor:DataFrame，指定域下的因子值，index为日期，columns为股票代码，value为因子值
        """
        universe_factor = pd.DataFrame()
        if idx == 'hs300':
            universe = self.hs300_cons
        elif idx == 'zz500':
            universe = self.zz500_cons
        elif idx == 'all_a':
            universe = self.all_a
        else:
            raise Exception('目前idx只支持hs300, zz500, all_a')
        for date in factor.index:
            date_universe = universe.loc[date, :].dropna()
            universe_factor = universe_factor.append(factor.loc[date, date_universe].to_frame(date).T)
        return universe_factor

    @staticmethod
    def calc_ic(signal_df, return_df, factor_name, ret_name, method='spearman'):
        """
        计算因子IC值, 本月和下月因子值的秩相关
        params:
                signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一列为股票当日的因子值
                return_df: DataFrame, columns=['ticker, 'tradeDate'， [next_period_ret]], 收益率，next_period_ret一列为下月的收益率
                factor_name:　str, signal_df中因子值的列名
                ret_name: str, return_df中收益率的列名
                method: : {'spearman', 'pearson'}, 默认'spearman', 指定计算rank IC('spearman')或者Normal IC('pearson')
        return:
                DataFrame, 返回IC值和本月和下月因子值的秩相关
        """
        merge_df = signal_df.merge(return_df, on=['secID', 'tradeDate'])

        # 计算IC
        ic_df = merge_df.groupby('tradeDate').apply(
            lambda x: x[factor_name + [ret_name]].corr(method=method).iloc[-1, range(len(factor_name))]).dropna()

        return ic_df
    
    def ic_return_analysis_simple(self, n_bucket, *args):
        """
        简单的IC，Return分析函数(仅针对月度收益率和IC的分析)
        """
        ic_result = pd.DataFrame()
        for factor in args:
            factor_name = factor.columns[-1]
            ic_result = ic_result.append(self.ic_describe(FactorTest.calc_ic(factor, self.return_df, [factor_name], 'forward_1m_ret')))
        return_result = pd.DataFrame()
        for factor in args:
            factor_return_result = self.bucket_back_test(factor, n_bucket)
            long_short = factor_return_result[n_bucket] - factor_return_result[1]
            long_short_net_value = (1 + long_short.fillna(0)).cumprod()
            month_return = long_short.mean()
            win_ratio = float(len(long_short[long_short > 0])) / len(long_short)
            ir = ((long_short_net_value.iloc[-1]) ** (12. / len(long_short)) - 1) / np.sqrt(12 * long_short.fillna(0).var())
            return_result = return_result.append(pd.DataFrame(data=[month_return, ir, win_ratio, 0], index=['月均收益', '信息比率', '月胜率', '最大回撤']).T)
        return_result.index = ic_result.index
        final_result = pd.concat((ic_result, return_result), axis=1)
        final_result.index.name = '因子'
        final_result.columns = [['RankIC', 'RankIC', 'RankIC', '多空组合', '多空组合', '多空组合', '多空组合'], 
                                ['IC均值', 'IC_IR', 't统计量', '月均收益', '信息比率', '月胜率', '最大回撤']]
        return final_result
        
    def ic_analysis(self, factor, factor_neu):
        """
        IC分析函数
        :param factor:
        :param factor_neu:
        :return:
        """
        data = pd.merge(factor, factor_neu, on=['secID', 'tradeDate'])
        all_factor_name = []
        
        for signal in [factor, factor_neu]:
            all_factor_name.append(signal.columns[-1])
            for cons in ['hs300', 'zz500']:
                signal_cons = signal[['secID', 'tradeDate', all_factor_name[-1]]].copy()
                if cons == 'hs300':
                    universe = self.hs300_cons
                else:
                    universe = self.zz500_cons
                signal_cons = signal_cons.groupby(by=['tradeDate']).apply(lambda x: select_universe(x, universe))
                signal_cons.columns = ['secID', 'tradeDate', all_factor_name[-1] + '_' + cons]
                data = pd.merge(data, signal_cons, on=['secID', 'tradeDate'], how='left')
        ic_result = FactorTest.calc_ic(data, self.return_df,
                                       all_factor_name + [factor_name + '_' + cons for factor_name in all_factor_name
                                                          for cons in ['hs300', 'zz500']],
                                       'forward_1m_ret')
        return ic_result

    def ic_describe(self, ic_df):
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
        ic_std = ic_df[factor_name].std()
        # IC均值的T统计量
        ic_t = pd.Series(st.ttest_1samp(ic_df[factor_name], 0)[0], index=factor_name)
        # IC_IR
        ic_ir = ic_mean / ic_std * np.sqrt(12.0)
        # IC统计
        ic_table = pd.DataFrame([ic_mean, ic_ir, ic_t],
                                 index=[u'平均IC', u'IC_IR', u'IC均值T统计量'])
        return ic_table.T

    def bucket_back_test(self, factor, n_bucket):
        factor_name = factor.columns[-1]
        cols = range(1, 1 + n_bucket)

        # 计算因子分组的超额收益平均值
        def cal_group_return(df):
            df['bucket'] = pd.qcut(df[factor_name].rank(method='first'), n_bucket, range(1, 1 + n_bucket))
            return df.groupby('bucket')['forward_1m_ret'].mean() - df['forward_1m_ret'].mean()

        factor = pd.merge(factor, self.return_df, on=['secID', 'tradeDate'], how='left')
        excess_returns_means = factor.groupby(by='tradeDate').apply(lambda x: cal_group_return(x))
        return excess_returns_means

    def bucket_back_test_summary(self, factor, n_bucket, ax1, ax2):
        result = self.bucket_back_test(factor, n_bucket)
        ax1 = barplot(result.mean(), ax1)
        ax2 = lineplot((1 + result[n_bucket] - result[1]).cumprod(), ax2)
        return ax1, ax2
    
calendar = DataAPI.TradeCalGet(exchangeCD='XSHG', beginDate='20060101', endDate='20180531')
calendar = calendar[calendar['isOpen'] == 1]
trade_date_list = calendar['calendarDate'].tolist()

month_date_list = calendar[calendar['isMonthEnd'] == 1]['calendarDate']
month_date_list = month_date_list[month_date_list > '2007-04-01'].tolist()

sue0 = fill_surprise_factor(pd.read_csv(path + 'sue0.csv', index_col=0), 'sue0')
sue1 = fill_surprise_factor(pd.read_csv(path + 'sue1.csv', index_col=0), 'sue1')
sur0 = fill_surprise_factor(pd.read_csv(path + 'sur0.csv', index_col=0), 'sur0')
sur1 = fill_surprise_factor(pd.read_csv(path + 'sur1.csv', index_col=0), 'sur1')
profit_growth_yoy = fill_surprise_factor(pd.read_csv(path + 'profit_growth_yoy.csv', index_col=0), 'profit_growth_yoy')
sales_growth_yoy = fill_surprise_factor(pd.read_csv(path + 'sales_growth_yoy.csv', index_col=0), 'sales_growth_yoy')

# 定义input_frame 
surprise_factor = sue0.merge(sue1, on=['secID', 'publishDate']).merge(sur0, on=['secID', 'publishDate']) \
                      .merge(sur1, on=['secID', 'publishDate']).merge(profit_growth_yoy, on=['secID', 'publishDate']).merge(sales_growth_yoy, on=['secID', 'publishDate'])
surprise_factor['publishDate'] = surprise_factor['publishDate'].str.replace('-', '')
surprise_factor.rename(columns={'publishDate': 'tradeDate'}, inplace=True)
input_frame = pd.merge(input_frame, surprise_factor, on=['secID', 'tradeDate'], how='inner')

hs300_cons, zz500_cons, a_cons = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
for date in month_date_list:
    hs300_cons = hs300_cons.append(pd.Series(get_idx_cons('000300', date)).to_frame(date).T)
    zz500_cons = zz500_cons.append(pd.Series(get_idx_cons('000905', date)).to_frame(date).T)
    a_cons = a_cons.append(pd.Series(get_idx_cons('A', date)).to_frame(date).T)
    
# 遍历每个月末日期，对因子进行去极值
# 申万行业分类
sw_map_frame = DataAPI.EquIndustryGet(industryVersionCD=u"010303", 
                                      field=['secID', 'secShortName', 'industry', 'intoDate', 'outDate', 
                                             'industryName1', 'industryName2', 'industryName3', 'isNew'], pandas="1")
sw_map_frame = sw_map_frame[sw_map_frame.isNew == 1]

print('winsorize factor data...')
pool = Pool(processes=8)
date_list = [tdate for tdate in np.unique(input_frame.tradeDate.values) if int(tdate) > 20061231]
dframe_list = pool.map(winsorize_fillna_date, date_list)

# 1. 准备一份不做中性化的因子数据
print('standardize factor...')
jobs = [gevent.spawn(standardize_factor, value) for value in zip(copy.deepcopy(dframe_list), date_list)]
gevent.joinall(jobs)
new_dframe_list = [e.value for e in jobs]
print('standardize factor finished!')

# 将不同月份的数据合并到一起
all_frame = pd.concat(new_dframe_list, axis=0)
all_frame.reset_index(inplace=True)

all_frame['tradeDate'] = all_frame['tradeDate'].apply(lambda x: datetime.strptime(x, '%Y%m%d').strftime('%Y-%m-%d'))
all_frame.to_csv(path + 'all_uqer_factor.csv')

# 2. 准备一份中性化后的因子数据
## 遍历每个月末日期，利用协程对因子进行中性化， 标准化处理
print('standardize & neutralize factor...')
jobs = [gevent.spawn(standardize_neutralize_factor, value) for value in zip(copy.deepcopy(dframe_list), date_list)]
gevent.joinall(jobs)
new_dframe_list = [e.value for e in jobs]
print('standardize neutralize factor finished!')

# 将不同月份的数据合并到一起
all_neu_frame = pd.concat(new_dframe_list, axis=0)
all_neu_frame.reset_index(inplace=True)

all_neu_frame['tradeDate'] = all_neu_frame['tradeDate'].apply(lambda x: datetime.strptime(x, '%Y%m%d').strftime('%Y-%m-%d'))
all_neu_frame.to_csv(path + 'all_uqer_factor_neu.csv')
pool.close()
pool.join()

all_frame = pd.read_csv(path + 'all_uqer_factor.csv', index_col=0)
all_neu_frame = pd.read_csv(path + 'all_uqer_factor_neu.csv', index_col=0)

# 加载行情数据
pct_chg = DataAPI.MktEqumAdjGet(beginDate='20070101', endDate='20180630', field='secID,endDate,chgPct')
forward_20d_return_data = pct_chg.set_index('endDate').groupby(by='secID').apply(lambda x: x['chgPct'].shift(-1)).reset_index().dropna()
forward_20d_return_data.columns = ['secID', 'tradeDate', 'forward_1m_ret']


'''

2. 业绩惊喜因子介绍
很多文献研究表明，公司盈利是否符合预期会在股票的价格当中有所体现。本文顺着这些学术研究的思路，定义了标准化的预期外盈利作为公司的盈利预期程度的一种度量。
定义公司i在季度t的标准化预期外盈利的计算公式为：
SUEi,t=Qi,t−E(Qi,t)σi,t
其中Qi,t是单季度的净利润，而E(Qi,t)则是在财报发布之前，期望的净利润，σi,t是盈利同比增长的标准差。那么问题就来了，如何对净利润的期望值，以及盈利增长的标准差进行估计？
公司盈利预期的估计一般来说有两种，一种是整理分析师的观点进行计算得到，还有一种则是通过时间序列上的一些模型进行计算得出。对于时间序列模型，一些早期的研究（Foster, Olson和Shevlin(1984)，Bernard和Thomas(1989)）假设单季度的净利润同比增长服从AR(1)过程，然后去估计净利润的期望值。但是Freeman和Tse(1989)以及其他一些学者发现，公告日后的收益率与季节随机游走模型给出的预测相关性更高。因此，本文也采用季节随机游走模型来估计盈利的市场预期。针对随机游走模型是否存在漂移项的问题，本文参考东方证券的做法，分别进行了测试了SUE0和SUE1两个指标。其中SUE0包含漂移项，SUE1不包含漂移项。带不带漂移项的差别在于假设市场是否会根据历史业绩增长对未来产生预期。
Qi,t=Qi,t−4+ci,t+ϵt                   (带漂移项)
  Qi,t=Qi,t−4+ϵt                           (不带漂移项)
对于带漂移项的季节性随机游走模型，漂移项的估计可以通过过去两年盈利的差分进行计算：
ci,t=∑8j=1(Qi,t−j−Qi,t−j−4)8
标准差的估计为：
σi,t=17∑j=18(Qi,t−j−Qi,t−j−4−ci,t)2−−−−−−−−−−−−−−−−−−−−−−⎷
这时候，盈利预期为：
E(Qi,t)=Qi,t−4+ci,t
对于不带漂移项的季节性随机游走模型，标准差的估计为：
σi,t=17∑j=18(Qi,t−j−Qi,t−j−4)2−−−−−−−−−−−−−−−−−⎷
而盈利预期为：
E(Qi,t)=Qi,t−4
我们计算了8期的盈利差分，所以，为了保证能计算出对应的数据，我们需要12期净利润的数据。
类似，我们将季节性随机游走过程应用在营业收入上，得到两个标准化预期外营业收入（standardized unexpected revenue, SUR）作为营业收入惊喜的度量。
SURi,t=REVi,t−E(REVi,t)ξi,t
其中，E(REVi,t)表示预期的营业收入，ξi,t表示预期营业收入的标准差，这两个都是通过季节性随机游走模型估计得到，SUR0表示模型包含漂移项，SUR1表示模型不含随机项。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3. 单因子有效性分析
前面，我们计算了每次财报发布后相关的业绩惊喜指标。从上一节事件研究的结果上来看，这些事件在异常收益的存续期内衰减并不明显。因此，本节将从因子的角度，将其进行因子化。为了简化问题，我们每个交易日取最新的数据来计算业绩超预期指标SUE0，SUE1，SUR0，SUR1，并将其作为因子值，以此来考查业绩惊喜因子的选股效果。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 IC分析


'''

factor_test = FactorTest()


sue0 = all_frame[['secID', 'tradeDate', 'sue0']]
sue0_neu= all_neu_frame[['secID', 'tradeDate', 'sue0']].rename(columns={'sue0': 'sue0_neu'})
sue0_ic_df = factor_test.ic_analysis(sue0.copy(deep=True), sue0_neu.copy(deep=True))
print ('基于归母净利润的SUE0因子的IC分析')
print (factor_test.ic_describe(sue0_ic_df).round(4).to_html())


sue1 = all_frame[['secID', 'tradeDate', 'sue1']]
sue1_neu= all_neu_frame[['secID', 'tradeDate', 'sue1']].rename(columns={'sue1': 'sue1_neu'})
sue1_ic_df = factor_test.ic_analysis(sue1.copy(deep=True), sue1_neu.copy(deep=True))
print ('基于归母净利润的SUE1因子的IC分析')
print (factor_test.ic_describe(sue1_ic_df).round(4).to_html())

sur0 = all_frame[['secID', 'tradeDate', 'sur0']]
sur0_neu= all_neu_frame[['secID', 'tradeDate', 'sur0']].rename(columns={'sur0': 'sur0_neu'})
sur0_ic_df = factor_test.ic_analysis(sur0.copy(deep=True), sur0_neu.copy(deep=True))
print ('基于营业收入的SUR0因子的IC分析')
print (factor_test.ic_describe(sur0_ic_df).round(4).to_html())

sur1 = all_frame[['secID', 'tradeDate', 'sur1']]
sur1_neu= all_neu_frame[['secID', 'tradeDate', 'sur1']].rename(columns={'sur1': 'sur1_neu'})
sur1_ic_df = factor_test.ic_analysis(sur1.copy(deep=True), sur1_neu.copy(deep=True))
print ('基于营业收入的SUR1因子的IC分析')
print (factor_test.ic_describe(sur1_ic_df).round(4).to_html())

'''

从IC的角度来看，基于归母净利润的SUE0因子最好，该因子的初始IC更高，但是中性化后的IC IR更加稳定。
而对于SUE1来说，中性化后的因子表现要更好一些。基于营业收入的业绩惊喜因子预测能力要稍弱些。
综合来看，中性化后的SUE1 IC最高，IC_IR也最高
   调试 运行
文档
 代码  策略  文档
3.2 分组收益测试

'''

all_a = a_cons.copy()

sue0_neu_ = sue0_neu.groupby(by='tradeDate').apply(lambda x: select_universe(x, all_a))
sue0_neu_.index = range(len(sue0_neu_))

sue1_neu_ = sue1_neu.groupby(by='tradeDate').apply(lambda x: select_universe(x, all_a))
sue1_neu_.index = range(len(sue1_neu_))

sur0_neu_ = sur0_neu.groupby(by='tradeDate').apply(lambda x: select_universe(x, all_a))
sur0_neu_.index = range(len(sur0_neu_))

sur1_neu_ = sur1_neu.groupby(by='tradeDate').apply(lambda x: select_universe(x, all_a))
sur1_neu_.index = range(len(sur1_neu_))

factor_test = FactorTest()
fig = plt.figure(figsize=(16, 7))
ax1, ax2, ax3, ax4 = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223),fig.add_subplot(224)
factor_test.bucket_back_test_summary(sue0_neu_, 10, ax1, ax3)
factor_test.bucket_back_test_summary(sue1_neu_, 10, ax2, ax4)
plt.show()

fig = plt.figure(figsize=(16, 7))
ax1, ax2, ax3, ax4 = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223),fig.add_subplot(224)
factor_test.bucket_back_test_summary(sur0_neu_, 10, ax1, ax3)
factor_test.bucket_back_test_summary(sur1_neu_, 10, ax2, ax4)
plt.show()

factor_neu = ((sue0_neu_.set_index(['secID', 'tradeDate'])['sue0_neu'] + sue1_neu_.set_index(['secID', 'tradeDate'])['sue1_neu'] +\
           sur0_neu_.set_index(['secID', 'tradeDate'])['sur0_neu'] + sur1_neu_.set_index(['secID', 'tradeDate'])['sur1_neu']) / 4).reset_index()
factor_neu.columns = ['secID', 'tradeDate', 'surprise']

factor = ((sue0.set_index(['secID', 'tradeDate'])['sue0'] + sue1.set_index(['secID', 'tradeDate'])['sue1'] +\
           sur0.set_index(['secID', 'tradeDate'])['sur0'] + sur1.set_index(['secID', 'tradeDate'])['sur1']) / 4).reset_index()
factor.columns = ['secID', 'tradeDate', 'surprise']

fig = plt.figure(figsize=(16, 3))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
ax1, ax2 = factor_test.bucket_back_test_summary(factor_neu, 10, ax1, ax2)


'''
从分组角度来看，中性化后的业绩惊喜因子都展现了不错的选股能力，其中中性化后的SUR0选股能力最强。和前面的报告《质优股量化投资》当中测试的基本面因子叫类似，业绩惊喜因子在2014年也发生了回撤。我们将因子进行等权的合成一个大类的业绩惊喜因子，从结果来看，选股能力非常不错，从2007年以来，多空收益一直比较稳健。

   调试 运行
文档
 代码  策略  文档
小结:

从单因子分析的结果来看，财报公告后，业绩超预期的股票有着显著的正向超额收益，而不及预期的股票有着负向的收益。
同时，从IC均值，多空收益来看，基于净利润的业绩惊喜因子要好于基于营业收入的业绩惊喜因子。
除了SUR1以外，其余三个因子的原始值有着十分显著的选股效果，但中性化后的因子选股稳定性更高，预测能力也更强些。
等权合成的大类业绩惊喜因子选股能力最为显著。
   调试 运行
文档
 代码  策略  文档
4. 信息增量分析
4.1 因子说明

我们重新回顾因子的构建逻辑，发现该因子与成长因子的结构比较相似。对于成长因子来说，为了不同股票之间的可比性，一般来说会除以上一期的值，但当上一期净利润为负时，这时候就会出现问题，简单起见，本文此处做法采用的是对分母取绝对值的方式。与成长因子不同，业绩惊喜因子则是除以波动进行调整（类似于标准化的做法）。但不管怎样，直观上来看，这两类因子分子相同，因此它们之间的相关性应该较高。这时候，我们需要先考察它是否给我们的模型带来额外的增量信息。

首先，为了降低问题复杂度，我们主要考虑大类因子之间的相关性。我们首先将这四个业绩惊喜因子进行等权合成，得到业绩惊喜这个大类因子，然后我们从当前的因子库当中，挑选一些比较成熟的因子，进行大类的合成（等权）。如下所示：

图片注释

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''
all_frame['growth'] = (all_frame['OperatingRevenueGrowRate'] + all_frame['NetProfitGrowRate'] + all_frame['sales_growth_yoy'] + all_frame['profit_growth_yoy']) / 4 
all_frame['surprise'] = (all_frame['sue0'] + all_frame['sue1'] + all_frame['sur0'] + all_frame['sur1']) / 4
all_neu_frame['growth'] = (all_neu_frame['OperatingRevenueGrowRate'] + all_neu_frame['NetProfitGrowRate'] + \
                           all_neu_frame['sales_growth_yoy'] + all_neu_frame['profit_growth_yoy']) / 4
all_neu_frame['surprise'] = (all_neu_frame['sue0'] + all_neu_frame['sue1'] + all_neu_frame['sur0'] + all_neu_frame['sur1']) / 4

all_frame['value'] = (all_frame['PE'] + all_frame['PB'] + all_frame['PS'] + all_frame['PCF'] + all_frame['CETOP']) / 5
all_frame['earnyild'] = (all_frame[u'ROE'] + all_frame[u'ROA'] + all_frame[u'EPS'] + all_frame[u'ROIC'] + all_frame[u'GrossIncomeRatio']) / 5
all_frame['liq'] = (all_frame[u'VOL20'] + all_frame[u'DAVOL20'] + all_frame['ILLIQUIDITY']) / 3
all_frame['lottry'] = (all_frame[u'REVS20'] + all_frame[u'REVS60']) / 2

print ('原始因子之间的Spearman相关系数矩阵')
all_frame[['growth', 'value', 'surprise', 'earnyild', 'lottry', 'liq']].corr(method='spearman').round(4)

all_neu_frame['value'] = (all_neu_frame['PE'] + all_neu_frame['PB'] + all_neu_frame['PS'] + all_neu_frame['PCF'] + all_neu_frame['CETOP']) / 5
all_neu_frame['earnyild'] = (all_neu_frame[u'ROE'] + all_neu_frame[u'ROA'] + all_neu_frame[u'EPS'] + all_neu_frame[u'ROIC'] + all_neu_frame[u'GrossIncomeRatio']) / 5
all_neu_frame['liq'] = (all_neu_frame[u'VOL20'] + all_neu_frame[u'DAVOL20'] + all_neu_frame['ILLIQUIDITY']) / 3
all_neu_frame['lottry'] = (all_neu_frame[u'REVS20'] + all_neu_frame[u'REVS60']) / 2

print ('中性化后因子之间的Spearman相关系数矩阵')
all_neu_frame[['growth', 'value', 'surprise', 'earnyild', 'lottry', 'liq']].corr(method='spearman').round(4)

'''

我们可以直接从斯皮尔曼相关系数矩阵上看出，业绩惊喜因子与成长因子高度相关，原始因子值之间的相关性达到了70%，而中性化之后的因子值的相关性也在35%以上。这表明两者之间存在很强的信息的重合。除此之外，业绩惊喜因子与盈利因子也存在一定的相关性，但不到20%，在可接受的范围之内。

   调试 运行
文档
 代码  策略  文档
4.3 回归分析

通过相关性分析，我们发现该因子与成长因子存在很大的相关性。因此，我们可以通过回归的方式，考察剔除成长因子，以及其他大类因子后业绩惊喜因子是否还有显著的选股效果。我们一个是将业绩惊喜大类因子通过回归，得到残差因子，同时也分析了每个细分因子（SUE0，SUE1，SUR0，SUR1）剔除了其他各类因子前后的表现。
从另外一个角度，我们还考察了其他大类因子剔除业绩惊喜因子前后的表现。
   调试 运行
文档
 代码  策略  文档
4.3.1 剔除其他大类因子后的业绩惊喜因子分析

'''


import statsmodels.api as sm
def regression(x, y, name):
    model = sm.OLS(y, x, missing='drop').fit()
    value = pd.Series(model.resid)
    value.name = name
    return value

all_frame = all_frame[all_frame['tradeDate'] > '2007-04-01']

all_frame = all_frame.set_index('secID')
surprise_without_other = all_frame.groupby(by='tradeDate').apply(lambda x: regression(x[['value', 'growth', 'earnyild', 'liq', 'lottry']], 
                                                                                         x['surprise'], 'surprise')).reset_index()
sur_detail_without_other = []
for name in ['sue0', 'sue1', 'sur0', 'sur1']:
    sur_detail_without_other.append(all_frame.groupby(by='tradeDate').apply(lambda x: regression(x[['value', 'growth', 'earnyild', 'liq', 'lottry']], 
                                                                                                    x[name], name)).reset_index())
sue0_without_other, sue1_without_other, sur0_without_other, sur1_without_other = tuple(sur_detail_without_other)


print ('剔除各大类因子前(因子原始值)表现')
print (factor_test.ic_return_analysis_simple(10, factor, sue0, sue1, sur0, sur1).round(4).to_html())
print ('剔除各大类因子后(因子原始值)表现')
print (factor_test.ic_return_analysis_simple(10, surprise_without_other, sue0_without_other, sue1_without_other, sur0_without_other, sur1_without_other).round(4).to_html())

all_neu_frame = all_neu_frame[all_neu_frame['tradeDate'] > '2007-04-01']
all_neu_frame = all_neu_frame.set_index('secID')
surprise_without_other = all_neu_frame.groupby(by='tradeDate').apply(lambda x: regression(x[['value', 'growth', 'earnyild', 'liq', 'lottry']], 
                                                                                             x['surprise'], 'surprise')).reset_index()
sur_detail_without_other = []
for name in ['sue0', 'sue1', 'sur0', 'sur1']:
    sur_detail_without_other.append(all_neu_frame.groupby(by='tradeDate').apply(lambda x: regression(x[['value', 'growth', 'earnyild', 'liq', 'lottry']], 
                                                                                                        x[name], name)).reset_index())
sue0_without_other, sue1_without_other, sur0_without_other, sur1_without_other = tuple(sur_detail_without_other)

print ('剔除各大类因子前(行业市值中性化)表现')
print (factor_test.ic_return_analysis_simple(10, factor_neu, sue0_neu_, sue1_neu_, sur0_neu_, sur1_neu_).round(4).to_html())
print ('剔除各大类因子后(行业市值中性化)表现')
print (factor_test.ic_return_analysis_simple(10, surprise_without_other, sue0_without_other, sue1_without_other, sur0_without_other, sur1_without_other).round(4).to_html())

'''

4.3.2 成长因子与业绩惊喜因子信息重叠分析

   调试 运行
文档
 代码  策略  文档
成长因子剔除业绩惊喜因子分析

'''

growth_without_surprise = all_frame.groupby(by='tradeDate').apply(lambda x: regression(x['surprise'],  x['growth'], 'growth_without_surprise')).reset_index()
print (factor_test.ic_return_analysis_simple(10, all_frame.reset_index()[['secID', 'tradeDate', 'growth']], growth_without_surprise).round(4).to_html())

growth_without_surprise = all_neu_frame.groupby(by='tradeDate').apply(lambda x: regression(x['surprise'],  x['growth'], 'growth_without_surprise')).reset_index()
print (factor_test.ic_return_analysis_simple(10, all_neu_frame.reset_index()[['secID', 'tradeDate', 'growth']], growth_without_surprise).round(4).to_html())

surprise_without_growth = all_frame.groupby(by='tradeDate').apply(lambda x: regression(x['growth'],  x['surprise'], 'surprise_without_growth')).reset_index()
print (factor_test.ic_return_analysis_simple(10, all_frame.reset_index()[['secID', 'tradeDate', 'surprise']], surprise_without_growth).round(4).to_html())

surprise_without_growth = all_neu_frame.groupby(by='tradeDate').apply(lambda x: regression(x['growth'],  x['surprise'],
                                                                                           'surprise_without_growth')).reset_index()
print (factor_test.ic_return_analysis_simple(10, all_neu_frame.reset_index()[['secID', 'tradeDate', 'surprise']], surprise_without_growth).round(4).to_html())

all_neu_frame.to_csv('surprise_data/final_factor.csv')


'''


小结:

我们比较了在剔除其它大类因子后，业绩惊喜因子的表现，发现无论是原始的业绩惊喜因子还是中性化之后的业绩惊喜因子，依然还有十分显著的选股能力。IC，IC_IR不降反升，这表明，业绩惊喜因子相对于这些因子而言，存在信息增量，选股效果几乎不受其它大类因子的影响
我们发现，与业绩惊喜因子高度相关的成长因子在剔除了业绩惊喜因子后，变得很不稳定，几乎没有选股能力
   调试 运行
文档
 代码  策略  文档
5. 指数增强
前面的分析表明，业绩惊喜因子在剔除了其他大类风格因子后，依然有不错的选股表现，而成长因子在剔除了业绩惊喜因子后，几乎没有了选股的能力，因此，通过业绩惊喜因子来替代成长因子，理论上来说应该会加强组合业绩的表现。本文验证了两个模型在沪深300成份股内增强，中证500成份股内增强共2个组合在20100106-20180631期间的表现。
简单起见，因子采用大类因子之间等权合成，然后构造行业中性组合。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''
factor = pd.read_csv(path + 'final_factor.csv')
factor['signal1'] = factor['growth'] + factor['value'] + factor['earnyild'] + factor['lottry'] + factor['liq']
factor['signal2'] = factor['surprise'] + factor['value'] + factor['earnyild'] + factor['lottry'] + factor['liq']
growth = factor.pivot(index='tradeDate', columns='secID', values='signal1')
surprise = factor.pivot(index='tradeDate', columns='secID', values='signal2')

def get_pf(bt):
    '''
    计算净值和回撤, 待画图用
    参数:
       bt: DataFrame，详细介绍参考：https://uqer.io/help/faq/#bt
    返回:
       data: DataFrame, 收益率等信息
       underwater：DataFrame，回撤信息
    '''
    data = bt[[u'tradeDate',u'portfolio_value',u'benchmark_return']].set_index('tradeDate')
    data.index = pd.to_datetime(data.index)
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()

    df_cum_rets = data['excess']
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -((running_max - df_cum_rets) / running_max)
    return data, underwater

'''

5.1 沪深300成份股指数增强


'''

universe = hs300_cons.copy()
factor_with_growth = pd.DataFrame()
factor_with_surprise = pd.DataFrame()
for date in growth.index:
    date_ = date.replace('-', '')
    signal1 = growth.ix[date, universe.loc[date, :].dropna().values].dropna()
    signal2 = surprise.ix[date, universe.loc[date, :].dropna().values].dropna()
    # 组合构建                
    wts1 = pd.Series(long_only(signal1, select_type=1, top_ratio=0.1, weight_type=1, target_date=date, universe_type='HS300')).to_frame(date).T
    wts2 = pd.Series(long_only(signal2, select_type=1, top_ratio=0.1, weight_type=1, target_date=date, universe_type='HS300')).to_frame(date).T
    factor_with_growth = factor_with_growth.append(wts1)
    factor_with_surprise = factor_with_surprise.append(wts2)
    
start = '2010-01-01'                       # 回测起始时间
end = '2018-06-30'                         # 回测结束时间

benchmark = 'HS300'                        # 策略参考标准
universe = DynamicUniverse('HS300')            # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔

factor_dates = factor_with_growth.index.values

# 配置账户信息，支持多资产多账户
accounts = {
    'growth_account': AccountConfig(account_type='security', capital_base=10000000),
    'surprise_account': AccountConfig(account_type='security', capital_base=10000000)
}

def initialize(context):                   # 初始化虚拟账户状态
    pass

def handle_data(context):                  # 每个交易日的买入卖出指令
    growth_account = context.get_account('growth_account')  
    surprise_account = context.get_account('surprise_account') 
    pre_date = context.previous_date.strftime("%Y-%m-%d")
    if pre_date not in factor_with_growth.index:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 组合构建                
    wts1 = factor_with_growth.ix[pre_date, :].dropna().to_dict()
    wts2 = factor_with_surprise.ix[pre_date, :].dropna().to_dict()
    
    # 全市场模型交易部分
    current_position = growth_account.get_positions(exclude_halt=True)    
    
    # 卖出当前持有，但目标持仓没有的部分
    for stk in set(current_position).difference(wts1):
        growth_account.order_to(stk, 0)

    for stk in sorted(wts1, key=wts1.get):
        growth_account.order_pct_to(stk, wts1[stk])
        
    # 分行业模型交易部分
    current_position = surprise_account.get_positions(exclude_halt=True)    
    
    # 卖出当前持有，但目标持仓没有的部分
    for stk in set(current_position).difference(wts2):
        surprise_account.order_to(stk, 0)

    for stk in sorted(wts2, key=wts2.get):
        surprise_account.order_pct_to(stk, wts2[stk])
        

# 读取回测数据
data_growth_factor, underwater_growth_factor = get_pf(bt_by_account['growth_account'])
data_surprise_factor, underwater_surprise_factor = get_pf(bt_by_account['surprise_account'])


# 画图展示
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.grid(True)
ax1.set_ylim(-0.12, 0.12)
ax1.fill_between(underwater_growth_factor.index, 0, np.array(underwater_growth_factor), color='r')
ax1.fill_between(underwater_surprise_factor.index, 0, np.array(underwater_surprise_factor), alpha=0.5, color='b')

(data_growth_factor['excess'] - 1).plot(ax=ax2, label='传统成长因子', color='r', fontsize=20)
(data_surprise_factor['excess'] - 1).plot(ax=ax2, label='业绩惊喜因子', color='b', fontsize=20)
ax2.set_ylim(-2, 2)
ax2.legend(loc='best', prop=font)
s = ax1.set_title(u"对冲组合超额收益走势（曲线图）", fontproperties=font, fontsize=16)
s = ax1.set_ylabel(u"回撤（柱状图）", fontproperties=font, fontsize=16)
s = ax2.set_ylabel(u"累计超额收益（曲线图）", fontproperties=font, fontsize=16)
s = ax1.set_xlabel(u"日期", fontproperties=font, fontsize=16)


universe = zz500_cons.copy()
factor_with_growth = pd.DataFrame()
factor_with_surprise = pd.DataFrame()
for date in growth.index[32:]:
    date_ = date.replace('-', '')
    signal1 = growth.ix[date, universe.loc[date, :].dropna().values].dropna()
    signal2 = surprise.ix[date, universe.loc[date, :].dropna().values].dropna()
    # 组合构建                
    wts1 = pd.Series(long_only(signal1, select_type=1, top_ratio=0.1, weight_type=1, target_date=date, universe_type='ZZ500')).to_frame(date).T
    wts2 = pd.Series(long_only(signal2, select_type=1, top_ratio=0.1, weight_type=1, target_date=date, universe_type='ZZ500')).to_frame(date).T
    factor_with_growth = factor_with_growth.append(wts1)
    factor_with_surprise = factor_with_surprise.append(wts2)
    
    
start = '2010-01-01'                       # 回测起始时间
end = '2018-06-30'                         # 回测结束时间

benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('ZZ500')        # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔

factor_dates = factor_with_growth.index.values

# 配置账户信息，支持多资产多账户
accounts = {
    'growth_account': AccountConfig(account_type='security', capital_base=10000000),
    'surprise_account': AccountConfig(account_type='security', capital_base=10000000)
}

def initialize(context):                   # 初始化虚拟账户状态
    pass

def handle_data(context):                  # 每个交易日的买入卖出指令
    growth_account = context.get_account('growth_account')  
    surprise_account = context.get_account('surprise_account') 
    pre_date = context.previous_date.strftime("%Y-%m-%d")
    if pre_date not in factor_with_growth.index:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 组合构建                
    wts1 = factor_with_growth.ix[pre_date, :].dropna().to_dict()
    wts2 = factor_with_surprise.ix[pre_date, :].dropna().to_dict()
    
    # 全市场模型交易部分
    current_position = growth_account.get_positions(exclude_halt=True)    
    
    # 卖出当前持有，但目标持仓没有的部分
    for stk in set(current_position).difference(wts1):
        growth_account.order_to(stk, 0)

    for stk in sorted(wts1, key=wts1.get):
        growth_account.order_pct_to(stk, wts1[stk])
        
    # 分行业模型交易部分
    current_position = surprise_account.get_positions(exclude_halt=True)    
    
    # 卖出当前持有，但目标持仓没有的部分
    for stk in set(current_position).difference(wts2):
        surprise_account.order_to(stk, 0)

    for stk in sorted(wts2, key=wts2.get):
        surprise_account.order_pct_to(stk, wts2[stk])
        

# 读取回测数据
data_growth_factor, underwater_growth_factor = get_pf(bt_by_account['growth_account'])
data_surprise_factor, underwater_surprise_factor = get_pf(bt_by_account['surprise_account'])


# 画图展示
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.grid(True)
ax1.set_ylim(-0.12, 0.12)
ax1.fill_between(underwater_growth_factor.index, 0, np.array(underwater_growth_factor), color='r')
ax1.fill_between(underwater_surprise_factor.index, 0, np.array(underwater_surprise_factor), alpha=0.5, color='b')

(data_growth_factor['excess'] - 1).plot(ax=ax2, label='传统成长因子', color='r', fontsize=20)
(data_surprise_factor['excess'] - 1).plot(ax=ax2, label='业绩惊喜因子', color='b', fontsize=20)
ax2.set_ylim(-2, 2)
ax2.legend(loc='best', prop=font)
s = ax1.set_title(u"对冲组合超额收益走势（曲线图）", fontproperties=font, fontsize=16)
s = ax1.set_ylabel(u"回撤（柱状图）", fontproperties=font, fontsize=16)
s = ax2.set_ylabel(u"累计超额收益（曲线图）", fontproperties=font, fontsize=16)
s = ax1.set_xlabel(u"日期", fontproperties=font, fontsize=16)

'''
6. 总结
我们从时间序列角度出发，计算了基于随机游走模型，分别从净利润，营业收入两个角度计算了4个业绩惊喜指标。
基于我们事件分析框架的研究结论，我们认为盈利公告的价格漂移现象在A股也存在。
将业绩惊喜指标因子化之后，我们发现其选股能力也非常显著。
业绩惊喜因子与传统的成长 因子具有较高的相关性。业绩惊喜因子在剔除其他大类因子后，选股效果依然显著，而成长因子在剔除业绩惊喜因子后，选股能力丧失。
指数增强组合使用业绩惊喜因子替代成长因子后，业绩有稳定提升。
感兴趣的读者可以将业绩预告，业绩快报等信息纳入因子的计算当中，效果会有进一步的提升。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''

