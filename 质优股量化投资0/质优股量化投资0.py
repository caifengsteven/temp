# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:45:40 2020

@author: Asus
"""
'''

导读
研究目的：本文借助优矿的量化因子库，参考东方证券研报：《质优股量化投资——寻找A股的动量效应》，书籍：《量化投资策略——如何实现超额收益Alpha》，对研报及书中的一些结果进行了实证分析。从盈利、成长、财务安全、质量优良等维度，综合定量的考虑一个公司是否真的质地优良。

研究结论：实证表明，通过优矿的量化因子库，以及优矿平台，我们能够发现质地优良的股票。通过质量因子，构建全市场中证500行业中性组合，从2008年7月回测至2017年8月，月度调仓，多头部分年化收益达到20.5%，年化Alpha达到12.2%，信息比率达到1.95。自2016年以来，组合稳健的获得超额收益。

文章结构：

数据及工具函数的准备：该部分主要是一些工具函数的编写，包括交易日历的获取，因子的获取，股票池的界定，因子的处理函数，因子分析的代码以及分析报告的绘图展示等等。
质量相关因子的定义及分析：该部分主要从盈利、成长、财务安全、治理优良等维度来度量股票是否优良，对这些层面下的因子进行测试及分析
质量因子的合成：该部分主要对第二节中的因子进行合成，并结合优矿平台，真实的回测了质优股在历史上是否能获得超额收益
其它说明：

本文测试的股票池剔除了上市不足60天，ST以及*ST的股票
由于部分因子不包括申万一级行业分类下的银行，非银金融的股票，为了统一，本文对其他因子的测试也相应的剔除了这类股票
因子测试的时间为2007年1月至2017年7月，部分因子时间因为数据获取问题，起始时间会往后推，但结束时间不变
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


1. 数据及工具函数的准备
该部分，我们主要进行数据的预加载以及处理。包括因子值的获取和处理，工具函数的编写等等。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import time

#from CAL.PyCAL import *
from datetime import datetime, timedelta
from scipy.stats import ttest_ind
from multiprocessing.dummy import Pool as ThreadPool

sns.set_style('whitegrid')


def get_halt_tickers(date):
    """
    获取历史上某一天停牌的股票，包括上市和暂停上市的
    输入:
        date: str，'YYYYMMDD'格式
    返回: 
        list: 元素为停牌的股票代码
    """
    data = DataAPI.SecHaltGet(beginDate=date, endDate=date, field='')
    data = data[(data['assetClass'] == 'E') & (data['listStatusCD'].isin(['L', 'S', 'DE']))]
    return data['ticker'].tolist()

def get_st_tickers(date):
    '''
    获取历史上某一天的ST股票
    输入:
        date: str, 'YYYYMMDD'格式
    返回：
        list: 元素为股票ticker
    '''
    data = DataAPI.SecSTGet(beginDate=date, endDate=date, field='')
    return data['ticker'].tolist()

def get_Atickers(date):
    """
    给定日期，获取这一天上市时间不低于60天的股票（参照中证全指指数编制）
    输入：
        date: str， 'YYYYMMDD'格式
    返回：
        list： 元素为股票ticker
    """
    universe = DataAPI.EquGet(equTypeCD=u"A", listStatusCD="L,S,DE", field=u"ticker,listDate,delistDate")
    universe['listdate'] = universe['listDate'].apply(lambda x: x.replace('-', ''))
    universe['delistdate'] = universe['delistDate'].apply(
        lambda x: x.replace('-', '') if isinstance(x, unicode) else '99999999')
    list_d_need = shift_date(date, 60, 'back')
    universe = universe[(universe['listdate'] <= list_d_need) & (universe['delistdate'] > date)]
    tickers = list(set(universe['ticker'].tolist()) - set(get_st_tickers(date)))
    return tickers

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
        
def get_rank_ic(factor, forward_return):
    """
    计算因子的信息系数
    输入：
        factor:DataFrame，index为日期，columns为股票代码，value为因子值
        forward_return:DataFrame，index为日期，columns为股票代码，value为下一期的股票收益率
    返回：
        DataFrame:index为日期，columns为IC，IC t检验的pvalue
    注意：factor与forward_return的index及columns应保持一致
    """
    common_index = factor.index.intersection(forward_return.index)
    ic_data = pd.DataFrame(index=common_index, columns=['IC','pValue'])

    # 计算相关系数
    for dt in ic_data.index:
        tmp_factor = factor.ix[dt]
        tmp_ret = forward_return.ix[dt]
        cor = pd.DataFrame(tmp_factor)
        ret = pd.DataFrame(tmp_ret)
        cor.columns = ['corr']
        ret.columns = ['ret']
        cor['ret'] = ret['ret']
        cor.dropna(inplace=True)
        if len(cor) < 5:
            continue

        ic, p_value = st.spearmanr(cor['corr'], cor['ret'])   # 计算秩相关系数RankIC
        ic_data['IC'][dt] = ic
        ic_data['pValue'][dt] = p_value
    return ic_data

def getUqerStockIndu(trade_date_list, universe):
    """
    记录区间段内的股票的申万一级行业分类
    输入
        trade_date_list:list，元素为'YYYYMMDD'格式的字符串
        universe:list，元素为股票的secID
    返回:
        DataFrame:index为股票代码，columns为secID，values为所属的申万一级行业分类
    """
    # 按天拿取行业数据，并保存为一个dataframe
    df = pd.DataFrame()
    for dt in trade_date_list:
        # 拿取数据dataapi，必要时可以使用专业版api
        dt_df = DataAPI.EquIndustryGet(industryVersionCD=u"010303", secID=universe, intoDate=dt, field=u"secID,industryName1",pandas="1").set_index('secID').T
        dt_df.index = [dt]
        if df.empty:
            df = dt_df
        else:
            df = df.append(dt_df)
    return df

def get_universe_factor(factor, idx=None, univ=None):
    """
    筛选出某指数成份股或者指定域内的因子值
    输入：
        factor:DataFrame，index为日期，columns为股票代码，value为因子值
        idx:指数代码，000300:沪深300，000905：中证500，000985：中证全指
        univ:DataFrame，index为日期，'YYYYMMDD'格式。columns为'code'，value为股票代码
    返回：
        factor:DataFrame，指定域下的因子值，index为日期，columns为股票代码，value为因子值
    """
    universe_factor = pd.DataFrame()
    if idx is not None:
        for date in factor.index:
            universe = get_idx_cons(idx, date)
            universe = [i + '.XSHG' if i[0] == '6' else i + '.XSHE' for i in universe]
            universe_factor = universe_factor.append(factor.loc[date, universe].to_frame(date).T)
    else:
        if univ is not None:
            for date in factor.index:
                universe = univ.loc[date, 'code'].tolist()
                universe_factor = universe_factor.append(factor.loc[date, universe].to_frame(date).T)
        else:
            raise Exception('请指定成分股或域')
    return universe_factor

def get_factor_by_day(tdate):
    '''
    根据日期，获取当天的因子值
    tdate：str，'YYYYMMDD'格式
    '''
    cnt = 0
    while True:
        try:
            x = DataAPI.MktStockFactorsOneDayProGet(tradeDate=tdate,secID=u"",ticker=u"",field=['secID', 'tradeDate'] + factors,pandas="1")
            return x
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                print('error get factor data: ', tdate)
                break
                
def get_group_ret(factor, month_ret, n_quantile=10):
    """
    计算分组超额收益：组合构建方式为等权，基准也为等权.
    注意：month_ret和factor应该错开一期，也就是说，month_ret要比factor晚一期
    输入：
        factor:DataFrame，index为日期，columns为股票代码，value为因子值
        month_ret:DataFrame，index为日期，columns为股票代码，value为收益率，month_ret的日期频率应和factor保持一致
        n_quantile:int，分组数量
    返回：
        DataFrame：列为分组序号，index为日期，值为每个调仓周期的组合收益率
    """
    # 统计分位数
    cols_mean = [i+1 for i in range(n_quantile)]
    cols = cols_mean

    excess_returns_means = pd.DataFrame(index=month_ret.index[1:len(factor+1)], columns=cols)

    # 计算因子分组的超额收益平均值
    for t, dt in enumerate(excess_returns_means.index):
        qt_mean_results = []

        # ILLIQ去掉nan
        tmp_factor = factor.iloc[t].dropna()
        tmp_return = month_ret.loc[dt].dropna()
        tmp_return = tmp_return.loc[tmp_factor.index]
        tmp_return_mean = tmp_return.mean()

        pct_quantiles = 1.0 / n_quantile
        for i in range(n_quantile):
            down = tmp_factor.quantile(pct_quantiles*i)
            up = tmp_factor.quantile(pct_quantiles*(i + 1))
            i_quantile_index = tmp_factor[(tmp_factor <= up) & (tmp_factor >= down)].index
            mean_tmp = tmp_return[i_quantile_index].mean() - tmp_return_mean
            qt_mean_results.append(mean_tmp)
        excess_returns_means.ix[t] = qt_mean_results
    return excess_returns_means

def plot_under_water(bt, title):
    """
    绘制回撤及收益率曲线图
    输入：
        bt：quartz回测结束自动生成的dict
        title：str
    返回：
        ax：matplotlib figure 对象
    """
    bt_quantile_ten = bt
    data = bt_quantile_ten[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()

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
    ax2.set_ylim(bottom=0.9, top=4)
    s = ax1.set_title(title, fontproperties=font, fontsize=16)
    return fig

def pretreat_factor(factor_df, neu=True):
    """
    因子处理函数
    输入：
        factor_df：DataFrame，index为日期，columns为股票代码，value为因子值
        neu：Bool，是否进行行业+市值中性化，若为True，则进行去极值->中性化->标准化；若为否，则进行去极值->标准化
    返回：
        factor_df：DataFrame，处理之后的因子
    """
    pretreat_data = factor_df.copy(deep=True)
    for dt in pretreat_data.index:
        try:
            factor_dt = pretreat_data.ix[dt].dropna()
            factor_dt_dict = factor_dt.to_dict()
            if neu:
                pretreat_data.ix[dt] = pd.Series(standardize(neutralize(winsorize(factor_dt_dict), target_date=''.join(dt.split('-')), industry_type='SW1',                                                              exclude_style_list=['RESVOL', 'BETA', 'EARNYILD', 'BTOP', 'SIZENL', 'GROWTH', 'LEVERAGE', 'MOMENTUM', 'LIQUIDTY'])))
            else:
                pretreat_data.ix[dt] = pd.Series(standardize(winsorize(factor_dt_dict)))
        except Exception as excp:
            print (dt)
            print (excp)
            continue
    return pretreat_data

def group_mean_report_plot(group_return, direction=1):
    """
    分组收益绘图
    group_return：分组收益，columns为分组序号，index为日期，值为每个调仓周期的组合收益率。可由函数get_group_ret产生
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(212)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(211)
    ax2.grid(False)
    
    month_return = (group_return.iloc[:, np.sign(direction-1)] - group_return.iloc[:, -np.sign(direction+1)]).fillna(0)
    
    ax1.bar(pd.to_datetime(month_return.index), month_return.values)
    ax2.plot(pd.to_datetime(month_return.index), (month_return.values+1).cumprod(), color='r')
    ax1.set_title(u"因子在中证全指（扣除金融）的表现", fontproperties=font, fontsize=16)
    
    excess_returns_means_dist = group_return.mean()
    excess_dist_plus = excess_returns_means_dist[excess_returns_means_dist>0]
    excess_dist_minus = excess_returns_means_dist[excess_returns_means_dist<0]
    lns2 = ax3.bar(excess_dist_plus.index, excess_dist_plus.values, align='center', color='r', width=0.35)
    lns3 = ax3.bar(excess_dist_minus.index, excess_dist_minus.values, align='center', color='g', width=0.35)

    ax3.set_xlim(left=0.5, right=len(excess_returns_means_dist)+0.5)
    ax3.set_xticks(excess_returns_means_dist.index)
    ax3.set_title(u"因子分组超额收益", fontproperties=font, fontsize=16)
    ax3.grid(True)
    
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
            data = DataAPI.IdxConsGet(ticker=idx, intoDate=date, field='', pandas="1")['consTickerSymbol']
        except:
            raise ValueError(u'DataAPI.IdxConsGet出错了！！！')
        if len(data) < 50:
            raise ValueError('{0}该日指数成分股API取出来的成分股数不足50个！！！'.format(date))
    else:
        universe = get_Atickers(date)
        st_stk = get_st_tickers(date)
        return list(set(universe) - set(st_stk))
    return list(set(data))

def ticker2secID(ticker):
    """
    ticker代码转换为通联编制的secID
    转换规则：secID = ticker + 后缀：如果股票属于沪市，则后缀为'.XSHG'，如果属于深市，则后缀为'.XSHE'
    """
    ticker = '0'*(6-len(ticker)) + ticker
    if ticker[0] == '6':
        secID = ticker + '.XSHG'
    else:
        secID = ticker + '.XSHE'
    return secID

def get_easy_factor_report(factor, month_return, direction):
    """
    获得简单的因子分析报告，注意后面的分析会剔除金融行业。
    在输入的month_return中，索引应该和factor保持一致，
    输入：
        factor：DataFrame，index为日期，columns为股票代码，value为因子值
        month_return：DataFrame，index为日期，columns为股票代码，value为股票收益率。month_return
    返回：
        DataFrame：记录中性化前因子在不同域的IC，IC_IR，pValue，以及中性化后因子在不同域的IC，IC_IR，以及不同域的多空表现
    """
    factor_hs300 = get_universe_factor(factor, univ=univ_hs300).loc[:, factor.columns-finance]
    factor_zz500 = get_universe_factor(factor, univ=univ_zz500).loc[:, factor.columns-finance]

    factor_hs300_neu = pretreat_factor(factor_hs300)
    factor_zz500_neu = pretreat_factor(factor_zz500)
    factor_a_neu = pretreat_factor(factor)

    # 中性化前因子分析
    rank_ic_hs300 = get_rank_ic(factor_hs300, forward_1m_ret)
    rank_ic_zz500 = get_rank_ic(factor_zz500, forward_1m_ret)
    rank_ic_a = get_rank_ic(factor, forward_1m_ret)

    rank_ic_hs300_mean = rank_ic_hs300['IC'].mean()
    rank_ic_zz500_mean = rank_ic_zz500['IC'].mean()
    rank_ic_a_mean = rank_ic_a['IC'].mean()

    rank_ic_hs300_pvalue = ttest_ind(rank_ic_hs300['IC'].dropna().tolist(), [0] * len(rank_ic_hs300.dropna()))[1]
    rank_ic_zz500_pvalue = ttest_ind(rank_ic_zz500['IC'].dropna().tolist(), [0] * len(rank_ic_zz500.dropna()))[1]
    rank_ic_a_pvalue = ttest_ind(rank_ic_a['IC'].dropna().tolist(), [0] * len(rank_ic_a.dropna()))[1]

    rank_ic_ir_hs300 = rank_ic_hs300['IC'].mean() / rank_ic_hs300['IC'].std()
    rank_ic_ir_zz500 = rank_ic_zz500['IC'].mean() / rank_ic_zz500['IC'].std()
    rank_ic_ir_a = rank_ic_a['IC'].mean() / rank_ic_a['IC'].std()

    # 中性化后因子分析
    rank_ic_neu_hs300 = get_rank_ic(factor_hs300_neu, forward_1m_ret)
    rank_ic_neu_zz500 = get_rank_ic(factor_zz500_neu, forward_1m_ret)
    rank_ic_neu_a = get_rank_ic(factor_a_neu, forward_1m_ret)

    rank_ic_hs300_neu_pvalue = ttest_ind(rank_ic_neu_hs300['IC'].dropna().tolist(), [0] * len(rank_ic_neu_hs300['IC'].dropna()))[1]
    rank_ic_zz500_neu_pvalue = ttest_ind(rank_ic_neu_zz500['IC'].dropna().tolist(), [0] * len(rank_ic_neu_zz500['IC'].dropna()))[1]
    rank_ic_a_neu_pvalue = ttest_ind(rank_ic_neu_a['IC'].dropna().tolist(), [0] * len(rank_ic_neu_a['IC'].dropna()))[1]

    rank_ic_neu_hs300_mean = rank_ic_neu_hs300['IC'].mean()
    rank_ic_neu_zz500_mean = rank_ic_neu_zz500['IC'].mean()
    rank_ic_neu_a_mean = rank_ic_neu_a['IC'].mean()

    rank_ic_ir_neu_hs300 = rank_ic_neu_hs300['IC'].mean() / rank_ic_neu_hs300['IC'].std()
    rank_ic_ir_neu_zz500 = rank_ic_neu_zz500['IC'].mean() / rank_ic_neu_zz500['IC'].std()
    rank_ic_ir_neu_a = rank_ic_neu_a['IC'].mean() / rank_ic_neu_a['IC'].std()

    hs300_excess_returns = get_group_ret(factor_hs300_neu, month_return, n_quantile=10)
    zz500_excess_returns = get_group_ret(factor_zz500_neu, month_return, n_quantile=10)
    a_excess_returns = get_group_ret(factor_a_neu, month_return, n_quantile=10)

    hs300_long_short_ret = (hs300_excess_returns.iloc[:, np.sign(direction-1)] - hs300_excess_returns.iloc[:, -np.sign(direction+1)]).fillna(0)
    zz500_long_short_ret = (zz500_excess_returns.iloc[:, np.sign(direction-1)] - zz500_excess_returns.iloc[:, -np.sign(direction+1)]).fillna(0)
    a_long_short_ret = (a_excess_returns.iloc[:, np.sign(direction-1)] - a_excess_returns.iloc[:, -np.sign(direction+1)]).fillna(0)

    hs300_long_short_month_ret = hs300_long_short_ret.mean()
    zz500_long_short_month_ret = zz500_long_short_ret.mean()
    a_long_short_month_ret = a_long_short_ret.mean()

    hs300_long_short_win_ratio = float(len(hs300_long_short_ret[hs300_long_short_ret > 0])) / len(hs300_long_short_ret)
    zz500_long_short_win_ratio = float(len(zz500_long_short_ret[zz500_long_short_ret > 0])) / len(zz500_long_short_ret)
    a_long_short_win_ratio = float(len(a_long_short_ret[a_long_short_ret > 0])) / len(a_long_short_ret)

    hs300_long_short_sharp_ratio = hs300_long_short_ret.mean() / hs300_long_short_ret.std()
    zz500_long_short_sharp_ratio = zz500_long_short_ret.mean() / zz500_long_short_ret.std()
    a_long_short_sharp_ratio = a_long_short_ret.mean() / a_long_short_ret.std()

    # 最大回撤
    hs300_long_short_max_drawdown = max([1 - v/max(1, max((hs300_long_short_ret+1).cumprod()[:i+1])) for i,v in enumerate((hs300_long_short_ret+1).cumprod())])
    zz500_long_short_max_drawdown = max([1 - v/max(1, max((zz500_long_short_ret+1).cumprod()[:i+1])) for i,v in enumerate((zz500_long_short_ret+1).cumprod())])
    a_long_short_max_drawdown = max([1 - v/max(1, max((a_long_short_ret+1).cumprod()[:i+1])) for i,v in enumerate((a_long_short_ret+1).cumprod())])

    # 结果汇总
    report = pd.DataFrame(index=['沪深300', '中证500', '全A'], 
                          columns=[['原始因子', '原始因子', '原始因子', '行业和市值中性化后因子', '行业和市值中性化后因子','行业和市值中性化后因子',
                                    '行业和市值中性化后因子','行业和市值中性化后因子','行业和市值中性化后因子', '行业和市值中性化后因子'], 
                                   ['IC', 'IC_IR', 'pvalue', 'IC', 'IC_IR', 'pvalue', '多空组合月度收益', '胜率', '最大回撤', '夏普比率']])
    report.iloc[:, 0] = [rank_ic_hs300_mean, rank_ic_zz500_mean, rank_ic_a_mean]
    report.iloc[:, 1] = [rank_ic_ir_hs300, rank_ic_ir_zz500, rank_ic_ir_a]
    report.iloc[:, 2] = [rank_ic_hs300_pvalue, rank_ic_zz500_pvalue, rank_ic_a_pvalue]
    report.iloc[:, 3] = [rank_ic_neu_hs300_mean, rank_ic_neu_zz500_mean, rank_ic_neu_a_mean]
    report.iloc[:, 4] = [rank_ic_ir_neu_hs300, rank_ic_ir_neu_zz500, rank_ic_ir_neu_a]
    report.iloc[:, 5] = [rank_ic_hs300_neu_pvalue, rank_ic_zz500_neu_pvalue, rank_ic_a_neu_pvalue]
    report.iloc[:, 6] = [hs300_long_short_month_ret, zz500_long_short_month_ret, a_long_short_month_ret]
    report.iloc[:, 7] = [hs300_long_short_win_ratio, zz500_long_short_win_ratio, a_long_short_win_ratio]
    report.iloc[:, 8] = [hs300_long_short_max_drawdown, zz500_long_short_max_drawdown, a_long_short_max_drawdown]
    report.iloc[:, 9] = [hs300_long_short_sharp_ratio, zz500_long_short_sharp_ratio, a_long_short_sharp_ratio]
    return report

def replace_nan_indu(factor):
    """缺失值填充函数，使用行业中位数进行填充
    输入：
        factor：DataFrame，index为日期，columns为股票代码，value为因子值
    返回：
        factor：格式保持不变，为填充后的因子
    """ 
    fill_factor = pd.DataFrame()
    for date in factor.index:
        sec_list = map(lambda x: ticker2secID(x), univ.loc[date.replace('-', ''), 'code'].tolist())
        factor_array = factor.ix[date, :].to_frame('values')
        indu_array = indu.ix[date, :].dropna().to_frame('industryName1')
        factor_array = factor_array.merge(indu_array, left_index=True, right_index=True, how='inner')
        mid = factor_array.groupby('industryName1').median()
        factor_array = factor_array.merge(mid, left_on='industryName1', right_index=True, how='left')
        factor_array['values_x'][pd.isnull(factor_array['values_x'])] = factor_array['values_y'][pd.isnull(factor_array['values_x'])]
        fill_factor = fill_factor.append(factor_array['values_x'].to_frame(date).T)
    return fill_factor

########################## 取得个股月度的行情数据 ################################
print ('个股行情数据开始计算...')
month_return = DataAPI.MktEqumAdjGet(beginDate='20070101', endDate='20171231', field='endDate,secID,chgPct')
month_return = month_return.pivot(index='endDate', columns='secID', values='chgPct')
month_return.index = map(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%Y%m%d'), month_return.index)
forward_1m_ret = month_return.shift(-1)
print ('个股行情数据计算完成')
print ('---------------------')

print ('开始生成前文所定义的股票池...')
univ, univ_zz500, univ_hs300 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
trade_date_list = get_dates('20070101', '20170731', 'monthly')
for date in trade_date_list:
    current_universe = pd.Series(get_Atickers(date)).to_frame(name='code')
    current_universe.index = [date] * len(current_universe)
    univ = univ.append(current_universe)
    
    current_hs300_universe = pd.Series(get_idx_cons('000300', date)).to_frame(name='code')
    current_hs300_universe.index = [date] * len(current_hs300_universe)
    univ_hs300 = univ_hs300.append(current_hs300_universe)
    
    current_zz500_universe = pd.Series(get_idx_cons('000905', date)).to_frame(name='code')
    current_zz500_universe.index = [date] * len(current_zz500_universe)
    univ_zz500 = univ_zz500.append(current_zz500_universe)

univ['code'] = univ['code'].apply(lambda x: x + '.XSHG' if x[0] == '6' else x + '.XSHE')
univ_zz500['code'] = univ_zz500['code'].apply(lambda x: x + '.XSHG' if x[0] == '6' else x + '.XSHE')
univ_hs300['code'] = univ_hs300['code'].apply(lambda x: x + '.XSHG' if x[0] == '6' else x + '.XSHE')
print ('股票池生成结束')
print ('--------------------')
print ('开始计算因子数据...')
# 定义所需要的因子

factors = ['RevenueTTM', 'GrossProfitTTM', 'OperatingProfitRatio', 'NetProfitRatio', 'ROIC', 'ROA', 'ROE', 
           'TProfitTTM', 'OperateProfitTTM', 'TotalAssets', 'NetProfitGrowRate', 'OperatingRevenueGrowRate', 
           'TotalAssetGrowRate', 'currentRatio', 'NetTangibleAssets', 'InterestCover', 'NetOperateCFTTM', 
           'EquityToAsset', 'DebtEquityRatio', 'IntFreeNCL', 'FCFF', 'NOCFToTLiability', 'PE', 'PB', 'PS', 'PCF'
          ]

pool = ThreadPool(processes=16)
frame_list = pool.map(get_factor_by_day, trade_date_list)
pool.close()
pool.join()
factor_csv = pd.concat(frame_list, axis=0)
factor_csv.reset_index(inplace=True, drop=True)
factor_csv['tradeDate'] = factor_csv['tradeDate'].str.replace('-', '')
print ('因子数据计算完成')
print ('--------------------')

print ('开始计算行业数据...')
all_universe = DataAPI.EquGet(equTypeCD=u"A",listStatusCD=u"L,S,DE",field=u"secID",pandas="1")['secID'].tolist()
indu = getUqerStockIndu(trade_date_list, all_universe)
print ('行业数据计算完成')
print ('--------------------')

print ('计算高管前三薪酬因子...')
all_sala = pd.DataFrame()
for date in trade_date_list:
    code = univ.loc[date, 'code'].tolist()
    # 根据月份分类
    time = datetime.strptime(date, '%Y%m%d')
    if time.month <= 3:
        begin = datetime(time.year - 2, time.month, 28).strftime('%Y%m%d')
        end = datetime(time.year - 1, time.month, 28)
        end = end.strftime('%Y%m%d')
        temp = DataAPI.EquSalarySurGet(secID=code, beginDate=begin, endDate=end, field=u"secID,endDate,sumSalManaTop3",pandas="1")
    else:
        begin = datetime(time.year - 1, time.month, 28).strftime('%Y%m%d')
        end = datetime(time.year, time.month, 28)
        end = end.strftime('%Y%m%d')
        temp = DataAPI.EquSalarySurGet(secID=code, beginDate=begin, endDate=end, field=u"secID,endDate,sumSalManaTop3",pandas="1") 
    
    temp['date'] = date
    temp = temp[['date', 'secID', 'sumSalManaTop3']]    
    all_sala = all_sala.append(temp)
sumSalManaTop3 = all_sala.pivot(index='date', columns='secID', values='sumSalManaTop3')
sumSalManaTop3 = sumSalManaTop3.replace([None], np.NaN)
sumSalManaTop3 = sumSalManaTop3.applymap(lambda x:float(x) if isinstance(x, str) else x)
print ('高管前三薪酬因子计算完成')

# 找到金融类股票，便于后面进行剔除
finance = indu.iloc[-1, :]
finance = finance[finance.isin(['银行', '非银金融'])].index

'''

2 质量因子的定义
本节，我们主要从盈利，成长，财务安全，治理优良这四个维度，综合的考察质量相关因子的表现

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

2.1 盈利能力

本节主要从企业盈利能力的维度来度量股票是否质优。

盈利是投资者最为关注的指标，毕竟盈利是商业投机活动的基本要求。利润可以分为毛利润、营业利润和净利润三种。净利润是最为综合的指标，考虑到了各种收入与费用，是股票分红的基础，未分配部分将进入股东权益，常用的PE、ROE等指标也是基于此计算。但也有学者推荐使用其他指标，Novy-Marx(2013)推荐使用毛利润，原因在于计算营业利润和净利润时，研发投入和营销费用（广告费、分销商佣金等）是作为营业费用被剔除的，它们会降低当期营业利润和净利润，但事实这些费用可能对上市公司长期竞争力和市场地位的保持有利，而企业的核心竞争力能够推动企业的盈利能力，因此毛利更能反应公司的真实盈利状况（考虑到A股上市公司整体研发投入不大，这个理由在A股的有效性可能有限）。三个利润指标反应了上市公司不同的盈利层面，无法互相取代，本文对这三个指标也分别进行了测试。为了使不同大小公司的数据具有可比性，利润指标要除以一个值来消除影响，当分母选为营业收入时，可以得到三个常用的选股指标：毛利率(GPM，Gross Profit Margin)、营业利润率（OPM，Operating Profit Margin）和净利率（NPM，Net Profit Margin）。分母也可以是总资产或股东权益，这里我们考察三个指标ROA、ROE和Novy-Marx(2013)建议的GPOA（Gross Profit On Asset）。
ROE是主动投资中使用最为广泛的盈利指标之一，关于ROE的分析，在我们先前的文章《ROE选股策略的思考与验证》已经有所提及。但ROE指标的分母是股东权益，易受上市公司财务杠杆的影响。我们测试了Greenblatt（2010）在他书中提到的一个盈利指标：投资资本回报率（ROIC，Return on Investment Capital）。相较于ROE，投资资本回报率能够识别出公司具有真正具备生产力的资产的回报。如果投资资本回报率很高，理论上公司就更有可能具备成长性并更有可能在未来获得超额收益。
本节，我们主要对上述的因子进行测试，具体包括：GPOA，GPM，NPM，OPM，ROA，ROE，ROIC，营业利润占比
这些指标优矿量化因子库已经提供，或者提供了计算所需的原始数据，对于优矿量化因子库当中的指标，其具体的计算方式需要在购买专业版之后才能提供。


'''

revenue = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='RevenueTTM')), univ=univ)
gross_profit = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='GrossProfitTTM')), univ=univ)
opm = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='OperatingProfitRatio')), univ=univ)
npm = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='NetProfitRatio')), univ=univ)
roic = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='ROIC')), univ=univ)
roa = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='ROA')), univ=univ)
roe = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='ROE')), univ=univ)
total_profit_ttm = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='TProfitTTM')), univ=univ)
operate_profit_ttm = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='OperateProfitTTM')), univ=univ)
gpm = replace_nan_indu((gross_profit / revenue)).replace([-np.inf, np.inf], [-10000, 10000])
total_assets = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='TotalAssets')), univ=univ)
gpoa = replace_nan_indu((gross_profit / total_assets))
operate_to_total = replace_nan_indu(operate_profit_ttm / total_profit_ttm)

'''
2.1.1 行业分析

在进行分析之前，我们比较了这些因子在不同行业的平均水平，可以看到行业间差别还是较大的。行业因素会干扰指标在全市场选股效果。我们在后续实证测试时，选股因子都做了行业和市值的风险中性化处理。另外银行和非银金融两个行业和其它行业在会计计量有所不同，部分因子，如GPM、GPOA和ROIC指标在这两个行业内没有取值。为了保证结果的统一性，我们在测试选股因子表现时，都统一把这两个行业的股票拿掉。

'''

indu_last = indu.loc['20170731'].to_frame('indu')
opm_last = opm.iloc[-1, :].to_frame('opm')
opm_indu_mean = pd.merge(opm_last, indu_last, how='inner', right_index=True, left_index=True).groupby('indu').mean()
gpm_last = gpm.iloc[-1, :].to_frame('gpm')
gpm_indu_mean = pd.merge(gpm_last, indu_last, how='inner', right_index=True, left_index=True).groupby('indu').mean()
npm_last = npm.iloc[-1, :].to_frame('npm')
npm_indu_mean = pd.merge(npm_last, indu_last, how='inner', right_index=True, left_index=True).groupby('indu').mean()
roe_last = roe.iloc[-1, :].to_frame('roe')
roe_indu_mean = pd.merge(roe_last, indu_last, how='inner', right_index=True, left_index=True).groupby('indu').mean()
roa_last = roa.iloc[-1, :].to_frame('roa')
roa_indu_mean = pd.merge(roa_last, indu_last, how='inner', right_index=True, left_index=True).groupby('indu').mean()
roic_last = roic.iloc[-1, :].to_frame('roic')
roic_indu_mean = pd.merge(roic_last, indu_last, how='inner', right_index=True, left_index=True).groupby('indu').mean()
gpoa_last = gpoa.iloc[-1, :].to_frame('gpoa')
gpoa_indu_mean = pd.merge(gpoa_last, indu_last, how='inner', right_index=True, left_index=True).groupby('indu').mean()
indu_dist = pd.concat([opm_indu_mean, gpm_indu_mean, npm_indu_mean, roe_indu_mean, roa_indu_mean, roic_indu_mean, gpoa_indu_mean], axis=1)

fig = plt.figure(figsize=(16, 8))
for i in range(indu_dist.shape[1]):
    k = 100 + 70 + i + 1
    ax = indu_dist.iloc[:, i].plot(kind='barh', ax=fig.add_subplot(k), color='r')
    ax.set_xlabel(indu_dist.columns[i])
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    if i == 0:
        s = ax.set_yticklabels([i.decode('utf-8') for i in indu_dist.index], fontproperties=font)
        s = ax.set_ylabel(u'行业', fontproperties=font, fontsize=14)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel('')
# 各个因子在申万一级行业内的平均值

'''
2.1.2 GPOA因子测试

我们在测试因子时，给出了两部分结果：

第一部分给出了原始因子在不同域内的IC，IC_IR，以及IC t检验的p-value，同时还给出了行业和市值中性化后的因子的表现，也包括IC，IC_IR，p-value，多空组合的月度收益、胜率、最大回撤和夏普比率
第二部分展示了分组超额收益以及多空组合（先前定义的股票池内构建）的累积净值



'''
gpoa_report = get_easy_factor_report(gpoa.loc[:, gpoa.columns-finance], month_return, -1)
print (gpoa_report.to_html())

gpoa_neu = pretreat_factor(gpoa.loc[:, gpoa.columns-finance], neu=True)
gpoa_neu_excess_returns = get_group_ret(gpoa_neu.loc[:, gpoa.columns-finance], month_return, 10)
ax = group_mean_report_plot(gpoa_neu_excess_returns, -1)

gpm_report = get_easy_factor_report(gpm.loc[:, gpm.columns-finance], month_return, -1)
print (gpm_report.to_html())

gpm_neu = pretreat_factor(gpm.loc[:, gpm.columns-finance], neu=True)
gpm_neu_excess_returns = get_group_ret(gpm_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(gpm_neu_excess_returns, -1)

opm_report = get_easy_factor_report(opm.loc[:, opm.columns-finance], month_return, -1)
print (opm_report.to_html())

opm_neu = pretreat_factor(opm.loc[:, opm.columns-finance], neu=True)
opm_neu_excess_returns = get_group_ret(opm_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(opm_neu_excess_returns, -1)

npm_report = get_easy_factor_report(npm.loc[:, npm.columns-finance], month_return, -1)
print (npm_report.to_html())

npm_neu = pretreat_factor(npm.loc[:, npm.columns-finance], neu=True)
npm_neu_excess_returns = get_group_ret(npm_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(npm_neu_excess_returns, -1)


roa_report = get_easy_factor_report(roa.loc[:, roa.columns-finance], month_return, -1)
print (roa_report.to_html())

roa_neu = pretreat_factor(roa.loc[:, roa.columns-finance], neu=True)
roa_neu_excess_returns = get_group_ret(roa_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(roa_neu_excess_returns, -1)

roe_report = get_easy_factor_report(roe.loc[:, roe.columns-finance], month_return, -1)
print (roe_report.to_html())

roe_neu = pretreat_factor(roe.loc[:, roe.columns-finance], neu=True)
roe_neu_excess_returns = get_group_ret(roe_neu.loc[:, roe_neu.columns-finance], month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(roe_neu_excess_returns, -1)

roic_report = get_easy_factor_report(roic.loc[:, roic.columns-finance], month_return, -1)
print (roic_report.to_html())

roic_neu = pretreat_factor(roic.loc[:, roic.columns-finance], neu=True)
roic_neu_excess_returns = get_group_ret(roic_neu.loc[:, roic_neu.columns-finance], month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(roic_neu_excess_returns, -1)

operate_to_total_report = get_easy_factor_report(operate_to_total.loc[:, operate_to_total.columns-finance], month_return, -1)
print (operate_to_total_report.to_html())

operate_to_total_neu = pretreat_factor(operate_to_total.loc[:, operate_to_total.columns-finance], neu=True)
excess_returns = get_group_ret(operate_to_total_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

'''

小结:

原始因子因为受到行业和市值因素的影响，基本上没有什么选股能力，中性化处理后因子IC和IC_IR明显提升。
七个盈利因子都是在中证500成份股里面IC要高于沪深300成份股里的IC。本文构造的全A的股票池可以近似看作沪深300+中证500+其它小股票的总和，盈利因子在中证全指里的IC基本上介于沪深300和中证500之间。因而盈利因子在以中证500为代表的中盘股里选股效果应该最好。
不同盈利因子构建的多空组合在2014年都发生了一次较大回撤，这可能和当年资本市场借壳上市的需求飙升有关。50家企业计划通过借壳曲线上市，并有大概20家企业当年成功实现借壳，这两个数字分别是2013年的三倍和两倍，业绩差的壳资源公司受到市场热炒，导致盈利因子表现大幅回撤。
三个利润率指标里，GPM表现最好，但这三个指标在沪深300里基本都没有什么选股能力。从证券估值角度讲，ROIC比ROE更能准确判断上市公司是否在创造价值，但是从前面的分析来看，两者的选股能力相差不大。
盈利因子并没有良好的单调性，其最高分位组合的超额收益要低于次高分位组合。

2.2 成长性

成长是公司盈利向未来的延伸，是未来现金流的保障。成长最直接的度量方式是净利润的同比增长，但需要注意的是上一期净利润为负时，净利润增长率的计算方式，因为个别时期亏损公司的占比很高。常用的方法是用对分母取绝对值。
另一种计算成长能力的方法，是用利润的变动值做分子，分母则取为总资产、净资产等大概率为正数的指标。这样可以避免上述增长率指标计算时分母为负值的问题，但得到的成长指标却可能受到了上市公司盈利能力的影响，没有上面两个增长率指标纯净。我们也测试了GPOA变动，ROE变动、ROIC变动。
本节我们主要测试了净利润增长率，营业收入增长率，总利润增长率，GPOA变动，ROE变动、ROIC变动这六个因子

'''

n_profit_grow_rate = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='NetProfitGrowRate')), univ=univ)
operating_revenue_grow_rate = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='OperatingRevenueGrowRate')), univ=univ)
total_asset_grow_rate = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='TotalAssetGrowRate')), univ=univ)
roic_chg = replace_nan_indu(roic.pct_change().replace([-np.inf, np.inf], np.NaN))
roe_chg = replace_nan_indu(roe.pct_change().replace([-np.inf, np.inf], np.NaN))
gpoa_chg = replace_nan_indu(gpoa.pct_change().replace([-np.inf, np.inf], np.NaN))

roe_chg.replace(0, np.NaN, inplace=True)
roe_chg.fillna(method='ffill', inplace=True)

roic_chg.replace(0, np.NaN, inplace=True)
roic_chg.fillna(method='ffill', inplace=True)

gpoa_chg.replace(0, np.NaN, inplace=True)
gpoa_chg.fillna(method='ffill', inplace=True)

gpoa_chg_report = get_easy_factor_report(gpoa_chg.ix[1:, gpoa_chg.columns-finance], month_return.ix[1:, :], -1)
print (gpoa_chg_report.to_html())

gpoa_chg_neu = pretreat_factor(gpoa_chg.loc[:, gpoa_chg.columns-finance], neu=True)
excess_returns = get_group_ret(gpoa_chg_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

n_profit_grow_rate_report = get_easy_factor_report(n_profit_grow_rate.loc[:, n_profit_grow_rate.columns-finance], month_return, -1)
print (n_profit_grow_rate_report.to_html())

n_profit_grow_rate_neu = pretreat_factor(n_profit_grow_rate.loc[:, n_profit_grow_rate.columns-finance], neu=True)
excess_returns = get_group_ret(n_profit_grow_rate_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

operating_revenue_grow_rate_report = get_easy_factor_report(operating_revenue_grow_rate.loc[:, operating_revenue_grow_rate.columns-finance], month_return, -1)
print (operating_revenue_grow_rate_report.to_html())

operating_revenue_grow_rate_neu = pretreat_factor(operating_revenue_grow_rate.loc[:, operating_revenue_grow_rate.columns-finance], neu=True)
excess_returns = get_group_ret(operating_revenue_grow_rate_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

total_asset_grow_rate_report = get_easy_factor_report(total_asset_grow_rate, month_return, -1)
print (total_asset_grow_rate_report.to_html())

total_asset_grow_rate_neu = pretreat_factor(total_asset_grow_rate.loc[:, operating_revenue_grow_rate.columns-finance], neu=True)
excess_returns = get_group_ret(total_asset_grow_rate_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

roe_chg_report = get_easy_factor_report(roe_chg.loc['20070228':, roe_chg.columns-finance], month_return.loc['20070238':, :], -1)
print (roe_chg_report.to_html())

roe_chg_neu = pretreat_factor(roe_chg.loc['20070228':, roe_chg.columns-finance], neu=True)
excess_returns = get_group_ret(roe_chg_neu.loc['20070228':, :], month_return.loc['20070228':, :], 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)
roic_chg_report = get_easy_factor_report(roic_chg.loc['20070330':, roic_chg.columns-finance], month_return.loc['20070330':, :], -1)
print (roic_chg_report.to_html())

roic_chg_neu = pretreat_factor(roic_chg.loc['20070330':, roic_chg.columns-finance], neu=True)
excess_returns = get_group_ret(roic_chg_neu, month_return.loc['20070330':, :], 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

current_ratio = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='CurrentRatio')), univ=univ)
debt_equity_ratio = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='DebtEquityRatio')), univ=univ)
int_free_ncl = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='IntFreeNCL')), univ=univ)
fcff = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='FCFF')), univ=univ)
nocf_to_tliability = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='NOCFToTLiability')), univ=univ)

fcff_to_int_free_ncl = fcff / int_free_ncl
fcff_to_int_free_ncl = replace_nan_indu(fcff_to_int_free_ncl).replace([-np.inf, np.inf], np.NaN)

debt_equity_ratio_report = get_easy_factor_report(debt_equity_ratio.loc[:, debt_equity_ratio.columns-finance], month_return, 1)
print (debt_equity_ratio_report.to_html())

debt_equity_ratio_neu = pretreat_factor(debt_equity_ratio, neu=True)
excess_returns = get_group_ret((debt_equity_ratio_neu).loc['20070131':, debt_equity_ratio_neu.columns-finance], month_return.loc['20070131':, :], 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)


report = get_easy_factor_report(fcff_to_int_free_ncl.loc[:, fcff_to_int_free_ncl.columns-finance], month_return, 1)
print (report.to_html())

fcff_to_int_free_ncl_neu = pretreat_factor(fcff_to_int_free_ncl, neu=True)
excess_returns = get_group_ret(fcff_to_int_free_ncl_neu.loc[:, debt_equity_ratio_neu.columns-finance], month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

report = get_easy_factor_report(nocf_to_tliability.loc[:, nocf_to_tliability.columns-finance], month_return, -1)
print (report.to_html())

nocf_to_tliability_neu = pretreat_factor(nocf_to_tliability, neu=True)
excess_returns = get_group_ret(nocf_to_tliability_neu.loc[:, nocf_to_tliability.columns-finance], month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

current_ratio_report = get_easy_factor_report(current_ratio.loc[:, current_ratio.columns-finance], month_return, 1)
print (current_ratio_report.to_html())

current_ratio_neu = pretreat_factor(current_ratio.loc[:, current_ratio.columns-finance] , neu=True)
excess_returns = get_group_ret(current_ratio_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, 1)

sumSalManaTop3 = replace_nan_indu(sumSalManaTop3)


sum_sal_mana_top3_report = get_easy_factor_report(sumSalManaTop3.loc[:, sumSalManaTop3.columns-finance], month_return, -1)
print (sum_sal_mana_top3_report.to_html())

sum_sal_mana_top3_neu = pretreat_factor(sumSalManaTop3.loc[:, sumSalManaTop3.columns-finance] , neu=True)
excess_returns = get_group_ret(sum_sal_mana_top3_neu, month_return, 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

'''

3 合成的质量因子分析
这里首先把第二节里介绍的部分alpha因子通过因子IC_IR加权的方式合成一个大类因子，考察大类因子的表现，大类因子的构成如下：

盈利因子：净利率、ROA、ROE、GPOA、ROIC、营业利润占比
成长因子：净利润增长、GPOA变动、ROE变动、ROIC变动
财务安全：自由现金流比非流动负债，经营活动产生的现金流量净额/负债合计
治理优良：高管前三薪酬之和
估值因子：PB、PE、PS、PCF
我们采用同一类型内部IC_IR方式合成大类因子，盈利，成长，财务安全，治理优良这四个大类因子之间再采用等权进行合成质量因子。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''

class FactorWeight():
    def __init__(self):
        pass
    
    @staticmethod
    def weighted(factor_dict, factor_weight):
        """
        用于因子合成的函数。因子之间需要对齐，因子和其对应的权重也应进行对齐
        输入：
            factor_dict：列表，用于存储因子，key为因子名，值为DataFrame(index为日期，columns为股票代码)
            factor_weight：因子权重，用于对因子进行配权，为DataFrame，index为日期，列对应着因子名称，值为当期因子的权重
        返回：
            DataFrame：最终合成后的因子
        """
        weighted_factor = 0
        for factor_name, factor in factor_dict.items():
            weighted_factor += factor.multiply(factor_weight[factor_name], axis=0)
        return weighted_factor

    @staticmethod
    def equal_weight(factor_dict):
        factor_weight = pd.Series([1. / len(factor_dict)] * len(factor_dict), index=factor_dict.keys()).to_dict()
        weighted_factor = FactorWeight.weighted(factor_dict, factor_weight)
        return weighted_factor
    
    @staticmethod
    def ic_weight(factor_dict, forward_month_return, window):
        
        # 获得IC序列
        all_rolling_ic_list = []
        for factor_name, factor in factor_dict.items():
            ic = get_rank_ic(factor, forward_month_return)['IC']
            # 计算得到当前因子的IC
            ic = pd.rolling_mean(ic, window=window)
            ic = ic.shift(1)
            ic.name = factor_name
            all_rolling_ic_list.append(ic)
        
        # 合并成一个DataFrame
        all_rolling_ic_df = pd.concat(all_rolling_ic_list, axis=1)
        all_rolling_ic_df = all_rolling_ic_df.divide(all_rolling_ic_df.sum(axis=1), axis=0)
        # 因子汇总
        weighted_factor = FactorWeight.weighted(factor_dict, all_rolling_ic_df)
        return weighted_factor
    
    @staticmethod
    def ic_ir_weight(factor_dict, forward_month_return, window):
        
        # 获得IC_IR序列
        all_rolling_ic_ir_list = []
        for factor_name, factor in factor_dict.items():
            ic = get_rank_ic(factor, forward_month_return)['IC']
            # 计算得到当前因子的IC_IR
            ic_ir = pd.rolling_mean(ic, window=window) / pd.rolling_std(ic, window=window)
            ic_ir = ic_ir.shift(1)
            ic_ir.name = factor_name
            all_rolling_ic_ir_list.append(ic_ir)
        
        # 合并成一个DataFrame，并计算权重
        all_rolling_ic_ir_df = pd.concat(all_rolling_ic_ir_list, axis=1)
        all_rolling_ic_ir_df = all_rolling_ic_ir_df.divide(all_rolling_ic_ir_df.sum(axis=1), axis=0)
        
        # 因子汇总
        weighted_factor = FactorWeight.weighted(factor_dict, all_rolling_ic_ir_df)
        return weighted_factor, all_rolling_ic_ir_df
    
    
    
# 创建字典，用于因子合成
earning_dict = {'gpm':gpm_neu, 'gpoa':gpoa_neu, 'operate_to_total':operate_to_total_neu, 'roa':roa_neu, 'roe':roe_neu, 'roic':roic_neu}
growth_dict = {'roic_chg': roic_chg_neu, 
               'roe_chg': roe_chg_neu, 
               'gopa_chg':gpoa_chg_neu, 
               'n_profit_grow_rate':n_profit_grow_rate_neu}
safe_dict = {'fcff_to_int_free_ncl':fcff_to_int_free_ncl_neu, 'nocf_to_tliability':nocf_to_tliability_neu}
manage = {'manage': sum_sal_mana_top3_neu}

# 合成大类因子
earning, a = FactorWeight().ic_ir_weight(earning_dict, month_return, 12)
growth, a  = FactorWeight().ic_ir_weight(growth_dict, month_return, 12)
safe, a = FactorWeight().ic_ir_weight(safe_dict, month_return, 12)
manage = FactorWeight().equal_weight(manage)

# 合成质量因子
quality = FactorWeight().equal_weight({'earning':earning, 'growth':growth, 'safe':safe, 'manage':manage})

# 生成简单的因子分析报告
report = get_easy_factor_report(quality.loc['20080331':, :], month_return.loc['20080331':, :], -1)
print (report.to_html())

quality_std = pretreat_factor(quality.loc['20080331':, :], neu=False)
excess_returns = get_group_ret(quality_std, month_return.loc['20080331':, :], 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

pe = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='PE')), univ=univ)
ps = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='PS')), univ=univ)
pcf = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='PCF')), univ=univ)
pb = get_universe_factor(replace_nan_indu(factor_csv.pivot(index='tradeDate', columns='secID', values='PB')), univ=univ)

pe_ = pe[pe > 0]
ep = pretreat_factor((1 / pe_).replace([-np.inf, np.inf], np.NaN))
bp = pretreat_factor((1/ pb).replace([-np.inf, np.inf], np.NaN))
cfp = pretreat_factor((1/ pcf).replace([-np.inf, np.inf], np.NaN))
sp = pretreat_factor((1/ ps).replace([-np.inf, np.inf], np.NaN))
value, a = FactorWeight.ic_ir_weight({'ep':ep, 'bp':bp, 'sp':sp, 'cfp':cfp}, month_return, 12)

value_std = pretreat_factor(value.loc['20080331':, :], neu=False)

corlation = 0
for date in earning.index[14:]:
    tmp = pd.concat([earning.loc[date, :].to_frame('盈利'), 
                     growth.loc[date, :].to_frame('成长'),
                     safe.loc[date, :].to_frame('财务安全'),
                     sum_sal_mana_top3_neu.loc[date, :].to_frame('治理优良'),
                     value_std.loc[date, :].to_frame('估值')], axis=1)
    corlation = tmp.corr() + corlation
corlation / len(earning.index[14:])

quality_ep = FactorWeight.equal_weight({'quality':quality_std, 'value':value_std})

report = get_easy_factor_report(quality_ep.loc['20080331':, quality_ep.columns-finance], month_return.loc['20080331':, :], -1)
print (report.to_html())

excess_returns = get_group_ret(quality_ep.loc['20080331':, quality_ep.columns-finance], month_return.loc['20080331':, :], 10)
# 因子分组的超额收益作图
ax = group_mean_report_plot(excess_returns, -1)

# quality_ep = pd.read_csv('quality_ep.csv', index_col=0)
quality_ep.index = map(str, quality_ep.index)

start = '2008-07-31'                       # 回测起始时间
end = '2017-08-31'                         # 回测结束时间

benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('A')         # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                          # 调仓频率，表示执行handle_data的时间间隔

factor_dates = quality_ep.index.values

# 配置账户信息，支持多资产多账户
accounts = {
    'stock_account': AccountConfig(account_type='security', capital_base=10000000)
}


def initialize(context):                   # 初始化虚拟账户状态
    pass

def handle_data(context):                  # 每个交易日的买入卖出指令
    account = context.get_account('stock_account') 
    pre_date = context.previous_date.strftime("%Y%m%d")
    if pre_date not in factor_dates:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
    signal = pd.Series(dict(quality_ep.ix[pre_date, :].dropna()))
    signal = dict(signal)
    
    # 组合构建                
    wts = long_only(signal, select_type=1, top_ratio=0.1, weight_type=1, target_date=''.join(pre_date.split('-')), universe_type='ZZ500')
    
    # 全市场模型交易部分
    current_position = account.get_positions(exclude_halt=True)    
    
    # 卖出当前持有，但目标持仓没有的部分
    for stk in set(current_position).difference(wts):
        account.order_to(stk, 0)

    for stk in sorted(wts, key=wts.get):
        account.order_pct_to(stk, wts[stk])
        
ax = plot_under_water(bt, title=u'对冲中证500净值走势')
# 下图中，阴影部分对应着当前时点的回撤