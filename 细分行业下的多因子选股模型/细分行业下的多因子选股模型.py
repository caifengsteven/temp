# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:04:09 2020

@author: Asus
"""
'''

细分行业下的多因子选股模型
导读
研究目的：

传统的多因子选股模型在给股票打分或者预测股票收益率，然后通过一定的技术手段，例如IC相关的加权方式,确定每个因子的权重，然后根据权重汇总得到股票的得分或者预期收益率的估计。并由此来构建组合，一般对于量化绝对收益产品来说，为了控制风险，需要让做多的股票组合的行业配权重与基准达到一致。也就是说，我们的选股是在每个行业内进行的，这就隐含了一个前提，我们最终进行选股的因子在这个行业内具有预测能力。但这个前提并不是在每个行业当中都成立。

事实上是，不同的行业，板块的股票的价格驱动因素并不完全一样。例如，价量因子在小盘股当中预测性更强；财务质量因子在价值股当中更为显著；成长因子偏好那些高成长的行业等等，分行业或者分域建模结果可能会更好。

还有一点在于，同一只股票在不同域下去极值和标准化后的值不一样。以对数市值因子为例，银行行业有很多市值很大的股票，如果是全域去极值处理，将会导致一些银行在市值上的得分一致。而如果在行业内进行去极值并标准化，那么它们的因子得分是不一样的。

因此，如果使用全市场有效因子来对行业内的股票收益做预测，效果可能不如用行业内有效因子来对行业内股票收益做预测更加有效。本文目的在于以行业为单位，希望能够筛选出一些行业，并且在这些行业内进行单独建模（分行业模型），从而提升组合的整体表现。

研究结论：本文通过从优矿的量化因子库精选的若干因子，对分行业模型进行测试。从结果上来看，分行业模型确实能够显著提升组合的业绩表现。通过分行业模型整合得到的因子，构建沪深300指数增强组合，从2011年1月回测至2017年12月，月度调仓，组合主动年化收益为11%，IR为2，最大回撤为5.3%，而基于全市场模型得到沪深300指数增强组合，主动年化收益仅为6.3， IR为1.06，最大回撤为11%，各项指标提升效果显著。

文章结构：

数据及工具函数的准备：该部分主要是一些工具函数的编写，包括交易日历的获取，因子的获取，股票池的界定，因子的处理函数，因子分析的代码以及分析报告的绘图展示等等。
行业测试：该部分对一些行业进行测试，测试的内容主要为分行业模型与全市场模型的业绩比较，以及当前行业下，因子预测能力排序
行业因子的整合：该部分对行业因子进行整合生成一个新的因子，并对该因子进行测试分析
   调试 运行
文档
 代码  策略  文档
1. 数据及工具函数的准备
该部分，我们主要进行数据的预加载以及处理。包括因子值的获取和处理，工具函数的编写等等。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''

import DataAPI
import pandas as pd
import gevent
import numpy as np
import logging
import scipy.stats as st
import quartz as qz
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time

from matplotlib import font_manager as fm 
from matplotlib import cm 
from quartz.api import *
from CAL.PyCAL import *
from datetime import datetime, timedelta
from multiprocessing.dummy import Pool as ThreadPool  # 存取数据
from multiprocessing import Pool  # 计算
sns.set_style("white")


def plot_under_water(alpha, ax=None):
    """
    回撤加净值曲线图，其中回撤部分位于图的上方，净值曲线位于图的下方
    alpha：pd.Series，策略的每日收益
    ax：绘制成功后返回的画布
    """
    df_cum_rets = (alpha + 1).cumprod()
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -((running_max - df_cum_rets) / running_max)
    underwater.index = alpha.index

    ax1 = ax
    ax2 = ax1.twinx()
    x = range(len(underwater))
    ax1.grid(False)
    ax1.grid(True)
    ax1.set_ylim(-0.30, 0)
    ax1.fill_between(underwater.index, 0, np.array(underwater), color='#000066', alpha=1)
    
    ax2.plot(alpha.index, df_cum_rets.values, label='hedged(right)', color='r')
    ax2.set_ylim(bottom=0.9, top=3)
    s = ax1.set_title(u"股票月度调仓走势", fontproperties=font, fontsize=16)
    return ax2


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
            x = DataAPI.MktStockFactorsOneDayProGet(tradeDate=date,
                                                    secID=u"",
                                                    field=['secID', 'tradeDate'] + factor_names,
                                                    pandas="1")
            return x
        except Exception as e:
            logging.info(e)
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
    factor['tradeDate'] = factor['tradeDate'].str.replace('-', '')
    return factor


def get_trade_dates(start_date, end_date, frequency='d'):
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
    if frequency == 'd':
        data = data[data['isOpen'] == 1]
    elif frequency == 'w':
        data = data[data['isWeekEnd'] == 1]
    elif frequency == 'm':
        data = data[data['isMonthEnd'] == 1]
    elif frequency == 'q':
        data = data[data['isQuarterEnd'] == 1]
    else:
        raise ValueError('调仓频率必须为d/w/m！！')
    date_list = map(lambda x: x[0:4] + x[5:7] + x[8:10], data['calendarDate'].values.tolist())
    return date_list


def shift_date(date, n, direction='back'):
    """
    日期平移函数，获取date向前/向后移动Ndays个交易日所对应的交易日
    输入：
        date: str， 'YYYYMMDD'格式
        n: int，长度不要超过700
        direction:str, 可选'back'或者'forward'
    返回：
        date：str，'YYYYMMDD'格式
    """
    last_two_year = str(int(date[:4]) - 2) + '0101'
    forward_two_year = str(int(date[:4]) + 2) + '1231'
    if direction == 'back':
        date_list = get_trade_dates(last_two_year, date, 'd')
        return date_list[len(date_list) - 1 - n]
    elif direction == 'forward':
        date_list = get_trade_dates(date, forward_two_year, 'd')
        if len(date_list) <= n:  # 当未来数据长度不够用时，返回最后一个能取到的交易日
            return date_list[-1]
        else:
            return date_list[n]
    else:
        raise ValueError('direction should be back/forward！！！')


def paper_winsorize(v, a_factor):
    """
    winsorize去极值，给定上下界
    输入：
        v: Series, 因子值
        upper: 上界值
        lower: 下界值
    返回：
        Series, 规定上下界后因子值
    """
    v = v[a_factor]
    dm = v.median()

    new_factor_series = abs(v - dm)  # abs(di-dm)
    dm1 = 1.483 * new_factor_series.median()
    
    upper = dm + 3 * dm1
    lower = dm - 3 * dm1
    
    v[v > upper] = upper
    v[v < lower] = lower

    return v


def winsorize_by_date(cdate_input):
    """
    按照[dm+5*dm1, dm-5*dm1]进行winsorize
    输入：
        cdate_input: 某一期的因子值的dataframe，这里，我们做行业内的去极值和标准化
    返回：
        DataFrame, 去极值后的因子值
    """
    for a_factor in factor_name:
        cdate_input[a_factor] = cdate_input[[a_factor, 'industryName1']].groupby(by='industryName1') \
                                                                        .apply(lambda x: paper_winsorize(x, a_factor)).reset_index().set_index('level_1')[a_factor]
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
            cdate_input = nafill_by_sw1(cdate_input)
            cdate_input.set_index('secID', inplace=True)
            
                
            cdate_input = input_frame[input_frame.tradeDate == tdate]
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


def standardize_neutralize_factor(input_data):
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
                cdate_input.loc[:, a_factor] =standardize(neutralize(cdate_input[a_factor], 
                                                                     target_date=tdate, 
                                                                     exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'SIZENL']))
                break
            except Exception as e:
                cnt += 1
                if cnt >= 3:
                    break

    return cdate_input


# 获取某个时点的行业分类
def get_industry(date):
    """
    输入：
        date: str， 'YYYYMMDD'格式
    返回：
        indu: 申万一级行业因子数据,dataframe格式，列名为日期，股票代码，行业名
    """ 
    cal = Calendar('China.SSE')
    end = cal.advanceDate(date, '-1B').strftime('%Y%m%d') 
    indu = pd.read_csv('all_indu.csv', index_col=0, dtype={'ticker': np.str})
    current_indu = indu[(indu['intoDate'] <= date) & ((indu['outDate'] >= date) | (pd.isnull(indu['outDate'])))]
    return current_indu


def get_st_tickers(date):
    """
    获取历史上某一天的ST股票
    输入:
        date: str, 'YYYYMMDD'格式
    返回：
        list: 元素为股票ticker
    """
    data = DataAPI.SecSTGet(beginDate=date, endDate=date, field='')
    return data['ticker'].tolist()


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
        cor = cor[~pd.isnull(cor['corr'])][~pd.isnull(cor['ret'])]
        if len(cor) < 5:
            continue

        ic, p_value = st.spearmanr(cor['corr'], cor['ret'])   # 计算秩相关系数RankIC
        ic_data['IC'][dt] = ic
        ic_data['pValue'][dt] = p_value
    return ic_data


def easy_backtest(start, end, universe, factor, quantile):
    """
    回测函数
    输入：
        start: str， 'YYYYMMDD'格式
        end: str， 'YYYYMMDD'格式
        universe: list，股票池
        factor: pd.DataFrame，因子数据，index为'YYYY-mm-dd'格式的日期，columns为股票代码，值为因子值，用于组合构建
        quantile: tuple， list，组合构建选择的分位数，例如[0.8, 1]表示选择因子值在80%分位数以上的股票
    返回：
        perf：记录组合的业绩表现，具体参考：https://uqer.io/help/faq/#perf
    """
    benchmark = 'HS300'                        # 策略参考基准
    accounts = {'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)}
    factor_dates = factor.index
    def initialize(context):                   # 初始化策略运行环境
        pass

    def handle_data(context):                  # 核心策略逻辑
        account = context.get_account('fantasy_account')
        
        pre_date = context.previous_date.strftime("%Y-%m-%d")
        if pre_date not in factor.index:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
            return
        
        # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
        q = factor.ix[pre_date].dropna()
        q_min = q.quantile(quantile[0])
        q_max = q.quantile(quantile[1])
    
        my_univ = list(set(q[q>=q_min][q<=q_max].index.tolist()) & set(context.get_universe(exclude_halt=True)))
        # 组合构建
        try:
            wts = pd.Series(1./ len(my_univ), index=my_univ).to_dict()
        except Exception as e:
            print (e)
            print (pre_date, my_univ)
            raise Exception('组合列表为空')
        
        # 交易部分
        current_position = account.get_positions(exclude_halt=True)    

        # 卖出当前持有，但目标持仓没有的部分
        for stk in set(current_position).difference(wts):
            account.order_to(stk, 0)

        for stk in sorted(wts, key=wts.get):
            account.order_pct_to(stk, wts[stk])
    bt, perf, stock = qz.backtest(start=start, 
                     end=end, 
                     benchmark='HS300', 
                     universe=universe, 
                     capital_base=1000000.0, 
                     initialize=initialize, 
                     handle_data=handle_data, 
                     refresh_rate=1, 
                     freq='d', 
                     accounts=accounts,
                     display=False)
    return perf


class SwIndu(object):
    """
    动态获取申万行业成份股股票
    """
    def __init__(self, name):
        self.name = name
    
    def preview(self, date, skip_st=True, skip_new=True):
        """
        输入:
            date:日期，'YYYYY-MM-DD'格式
            skip_st:去除st股票，ST摘帽后不足3个月的股票
            skip_new:去除上市不足六个月的股票
        返回：
            list:股票池
        """
        current_indu = get_industry(date)
        current_indu = current_indu[current_indu['industryName1'] == self.name]['secID'].tolist()
        
        if skip_new:
            # 去除新股
            current_indu = DataAPI.EquGet(secID=current_indu, equTypeCD=u"A", listStatusCD="L,S,DE", field=u"secID,listDate,delistDate")
            current_indu['listdate'] = current_indu['listDate'].apply(lambda x: x.replace('-', ''))
            current_indu['delistdate'] = current_indu['delistDate'].apply(lambda x: x.replace('-', '') if isinstance(x, unicode) else '99999999')
            list_d_need = shift_date(date, 120, 'back')
            current_indu = current_indu[(current_indu['listdate'] <= list_d_need) & (current_indu['delistdate'] > date)]['secID'].tolist()
        
        if skip_st:
            st = get_st_tickers(date)
            # 去除st
            current_indu = list(set(current_indu) - set(st))
            
            # 去除st摘帽不足6个月的股票
            remove_st_less_3m = DataAPI.EquInstSstateGet(secID=current_indu, beginDate=shift_date(date, 60), endDate=date, field=u"", pandas="1")
            remove_st_less_3m = remove_st_less_3m[remove_st_less_3m['partyState'] == 1]['secID'].tolist()
            current_indu = list(set(current_indu) - set(remove_st_less_3m))
        
        return current_indu
    

def industry_test(factor, industry_universe, forward_return, back_test=True):
    """
    行业内挑选因子建模并回测
    输入：
        factor:pd.DataFrame，全行业因子数据。多列，第一列为tradeDate，表示交易日历；第二列为secID，记录股票代码。后面若干列记录因子数据
        industry_universe:dict，行业成份股。key为日期，值为list，元素为对应日期下的行业成份股
        forward_return:pd.DataFrame，未来一期股票收益率。index为日期，columns为股票代码，值为股票下一个月的收益率
        back_test:bool，是否进行回测
    返回：
        tuple：第一个元素为行业IC，合成后的因子值，多头组合的策略表现，空头组合的策略表现，因子加权权重
    """
    IC = {}
    industry_factor = pd.DataFrame()
    
    # 因子处理，动态股票池
    for date in sorted(industry_universe.keys()):
        current_factor = factor[factor['tradeDate'] == date]
        current_factor = current_factor[current_factor['secID'].isin(industry_universe[date])]
        industry_factor = industry_factor.append(current_factor)

    # IC计算
    for factor_name in industry_factor.columns[2:-1]:
        current_factor = industry_factor.loc[:, ['tradeDate', 'secID', factor_name]]
        current_factor = current_factor.pivot(index='tradeDate', columns='secID', values=factor_name)
        IC[factor_name] = get_rank_ic(current_factor, forward_return)['IC']
    
    IC = pd.DataFrame(IC).shift(1).dropna(how='all', axis=0)

    # 根据短期因子IC选择因子
    short_ic_mean = IC.rolling(12).mean()
    short_ic_std = IC.rolling(12).std()
    short_ic_ir = short_ic_mean / short_ic_std
    
    ## 选择滚动IC均值绝对值高于0.04，滚动IC_IR高于0.25
    short_ic_mean = short_ic_mean[short_ic_mean > 0.04]
    short_ic_ir = short_ic_ir[short_ic_ir.abs() > 0.25]
    
    short_ic_mean.dropna(how='all', axis=0, inplace=True)
    short_ic_ir.dropna(how='all', axis=0, inplace=True)
    
    ## 按IC_IR加权，得到每个因子的权重
    short_ic_mean[~pd.isnull(short_ic_mean)] = 0

    short_ic_ir = (short_ic_ir + short_ic_mean).fillna(0)
    ic_ir_weight = short_ic_ir.divide(short_ic_ir.abs().sum(axis=1), axis=0)
    
    # 计算合成的因子
    weighted_factor = pd.DataFrame()
    for date in ic_ir_weight.index:
        current_factor = industry_factor[industry_factor['tradeDate']==date].set_index(['tradeDate', 'secID'])
        current_factor = current_factor.multiply(ic_ir_weight.loc[date], axis=1).sum(axis=1).reset_index()
        weighted_factor = weighted_factor.append(current_factor)

    weighted_factor.rename(columns={0: 'values'}, inplace=True)
    weighted_factor = weighted_factor.pivot(index='tradeDate', columns='secID', values='values')
    weighted_factor = weighted_factor.fillna(method='ffill')
    
    if back_test:
        # # 回测，多头组合取前30%，空头组合取后30%
        perf1 = easy_backtest('2011-01-31', '2017-12-29', weighted_factor.columns.tolist(), weighted_factor, [0.7, 1])
        perf2 = easy_backtest('2011-01-31', '2017-12-29', weighted_factor.columns.tolist(), weighted_factor, [0, 0.3])
    else:
        perf1 = None
        perf2 = None
    return IC, weighted_factor, perf1, perf2, ic_ir_weight

    
def all_indu_test(factor, industry_unvierse, forward_return, start_date, end_date):
    """
    全行业测试，使用全市场模型得到的因子数据进行行业内回测
    输入：
        factor:pd.DataFrame，全行业因子数据。index为日期，columns为股票代码，值为合成后的因子值
        industry_universe:dict，行业成份股。key为日期，值为list，元素为对应日期下的行业成份股
        forward_return:pd.DataFrame，未来一期股票收益率。index为日期，columns为股票代码，值为股票下一个月的收益率
        start_date:回测起始时间，'YYYY-MM-DD'格式的字符串
        end_date:回测结束时间，'YYYY-MM-DD'格式的字符串
    返回：
        tuple。策略多头组合的策略表现，策略空头组合的策略表现
    """
    IC = {}
    industry_factor = pd.DataFrame()
    
    # 因子处理，动态股票池
    for date in sorted(industry_unvierse.keys()):
        current_factor = factor.ix[date, industry_unvierse[date]]
        industry_factor = industry_factor.append(current_factor)
    
    # # 回测，多头组合取前30%，空头组合取后30%
    perf1 = easy_backtest(start_date, end_date, industry_factor.columns.tolist(), industry_factor, [0.7, 1])
    perf2 = easy_backtest(start_date, end_date, industry_factor.columns.tolist(), industry_factor, [0, 0.3])
    return perf1, perf2
    
def plot(perf1, perf2, benchmark, y_upper=4, ax=None, title=None):
    """
    绘图函数
    输入：
        perf1：策略多头的表现
        perf2：策略空头的表现
        benchmark：行业基准
        y_upper：y轴上限
        ax：画布
        title：标题
    返回：
        绘制成功的画布
    """
    ax = (1 + perf1['returns']).cumprod().plot(ylim=(0, y_upper), ax=ax, label='Top')
    ax = (1 + perf2['returns']).cumprod().plot(ylim=(0, y_upper), ax=ax, label='Bottom')
    ax = (1 + perf1['returns'] - perf2['returns']).cumprod().plot(ylim=(0, y_upper), color='r', ax=ax, label='long_short')
    ax.legend(loc='best')
    ax.set_title(title, fontproperties=font)
    return ax


def performance(indu_long, indu_short, all_long, all_short, benchmark):
    """
    性能指标综合比较
    输入：
        indu_long：分行业模型多头组合的表现
        indu_short：分行业模型空头组合的表现
        all_long：全市场模型多头组合的表现
        all_short：全市场模型空头的表现
        benchmark：基准指数
    """
    # 分行业模型，性能指标
    indu_long_short_ret = (indu_long['returns'] - indu_short['returns'])
    indu_long_short_annual_ret = ((1 + indu_long_short_ret).cumprod().iloc[-1])**(252./len(indu_long_short_ret)) - 1
    indu_long_short_ir = indu_long_short_annual_ret / np.sqrt(252 * indu_long_short_ret.var())
    
    indu_excess_ret = (indu_long['returns'] - benchmark)
    indu_excess_annual_ret = ((1 + indu_excess_ret).cumprod().iloc[-1])**(252./len(indu_excess_ret)) - 1
    indu_ir = indu_excess_annual_ret / np.sqrt(252 * indu_excess_ret.var())
    
    # 全市场模型， 性能指标
    all_long_short_alpha = (all_long['returns'] - all_short['returns'])
    all_long_short_annual_ret = ((1 + all_long_short_alpha).cumprod().iloc[-1])**(252./len(all_long_short_alpha)) - 1
    all_long_short_ir = all_long_short_annual_ret / np.sqrt(252 * all_long_short_alpha.var())
    
    all_excess_ret = (all_long['returns'] - benchmark)
    all_excess_annual_ret = ((1 + all_excess_ret).cumprod().iloc[-1])**(252./len(all_excess_ret)) - 1
    all_ir = all_excess_annual_ret / np.sqrt(252 * all_excess_ret.var())
    
    perf = pd.DataFrame(index=['全市场模型', '分行业模型'], columns=['多头组合年化超额收益', '多头组合IR', '多空组合年化超额收益', '多空组合IR'])
    perf.iloc[0, 0] = all_long_short_annual_ret
    perf.iloc[0, 1] = all_long_short_ir
    perf.iloc[0, 2] = all_excess_annual_ret
    perf.iloc[0, 3] = all_ir
    
    perf.iloc[1, 0] = indu_long_short_annual_ret
    perf.iloc[1, 1] = indu_long_short_ir
    perf.iloc[1, 2] = indu_excess_annual_ret
    perf.iloc[1, 3] = indu_ir
    
    return perf


def pie_plot(data):
    """
    饼图的绘制
    """
    labels = data.index 
    sizes = data.values 
    fig, ax = plt.subplots(figsize=(4, 4)) # 设置绘图区域大小 
    colors = cm.Spectral(np.arange(len(sizes))/float(len(sizes)))
    patches, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.0f%%',shadow=False, colors=colors, startangle=170, wedgeprops = {'linewidth': 2, 'edgecolor': 'white'}) 
    ax.axis('equal') 
    for t in texts: 
        t.set_fontproperties(font)
    plt.show() 
    

def cal_alpha_by_bt(bt):
    """
    根据quartz回测结束所返回的bt计算超额收益
    输入：
        bt:DataFrame，详细介绍参考：https://uqer.io/help/faq/#bt
    """
    data = bt[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return
    return data['excess_return']


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


def detail_performance(alpha):
    """
    计算具体的策略评价，并评价，包括分年度收益，组合年化收益率，信息比率和最大回撤等等
    输入：
        alpha:pd.Series，index为datetime格式的日期，值为每期收益
    返回：
        None
    """
    data = alpha.copy(deep=True)
    year_return = alpha.groupby(by=data.index.strftime('%Y')).apply(lambda x: (x+1).cumprod().iloc[-1] ** (252./len(x))-1)
    data.index = pd.to_datetime(data.index)
    print ('年度收益统计: \n')
    print (year_return.to_frame('return').T.to_html())
    print ('年化收益率：', (data + 1).cumprod().iloc[-1] ** (252./len(data)) - 1)
    print ('信息比率： ', ((data + 1).cumprod().iloc[-1] ** (252./len(data)) - 1) / np.sqrt(252 * data.var()))
    print ('最大回撤: ', max([1 - v/max(1, max((data+1).cumprod()[:i+1])) for i,v in enumerate((data+1).cumprod())]))
    
    
factor_group = {u'因子类别':[u'规模', u'动量', u'动量', u'流动性', u'流动性', u'流动性', u'估值', u'估值', u'估值', u'估值',
                        u'估值', u'成长', u'成长', u'成长', u'成长', u'盈利', u'盈利', u'盈利', u'盈利', u'盈利', u'现金流', 
                        u'现金流', u'周转', u'周转', u'周转', u'负债'],
                u'因子在优矿信号库当中的代码': [u'LCAP', u'REVS20', u'REVS60', u'VOL20', u'DAVOL20', u'ILLIQUIDITY', u'PB', u'PE', u'PS',
                                             u'PCF', u'CETOP', u'NetProfitGrowRate', u'OperatingRevenueGrowRate', u'EGRO', u'SGRO', u'ROE',
                                             u'ROA', u'EPS', u'ROIC', u'GrossIncomeRatio', u'EnterpriseFCFPS', u'OperCashFlowPS', u'InventoryTRate',
                                             u'FixedAssetsTRate', u'TotalAssetsTRate', u'DebtsAssetRatio'],
                u'因子描述': [u'对数市值', u'过去20个交易日收益率', u'过去60个交易日收益率', u'20日平均换手率', u'最近一段时间的日平均换手率相对于120个交易日的变化', 
                            u'收益相对金融比', u'市净率', u'市盈率', u'市销率', u'市现率', u'现金收益滚动收益与市值比', u'净利润增长率', u'营业收入增长率', 
                            u'5年收益增长率', u'5年营业收入增长率', u'权益回报率', u'资产回报率', u'基本每股收益', u'投入资本回报率', u'销售毛利率', u'每股企业自由现金流量', 
                            u'每股经营现金流量', u'存货周转率', u'固定资产周转率', u'总资产周转率', u'债务总资产比']}
factor_group = pd.DataFrame(factor_group)


# 设置交易日历
trade_date_list = get_trade_dates('20080101', '20171229', 'm')

# 因子数据的获取
factor_name = factor_group[u'因子在优矿信号库当中的代码'].tolist()

# 股票池，全A
universe = set_universe('A')

all_indu = DataAPI.MdSwBackGet()
all_indu.to_csv('all_indu.csv')

# 计算股票月度收益率
month_ret = DataAPI.MktEqumAdjGet(beginDate='20080101', endDate='20180331', field='secID,endDate,chgPct')
month_ret = month_ret.pivot(index='endDate', columns='secID', values='chgPct')
forward_month_ret = month_ret.shift(-1).dropna(how='all', axis=0)

# 申万一级行业中英文对照
SW_name = {
       'Bank': '银行', 'RealEstate': '房地产', 'Health': '医药生物', 'Transportation': '交通运输', 'Mining': '采掘', 'NonFerMetal': '有色金属', 
       'HouseApp': '家用电器', 'LeiService': '休闲服务', 'MachiEquip': '机械设备', 'BuildDeco': '建筑装饰', 'CommeTrade': '商业贸易', 'CONMAT': '建筑材料', 
       'Auto': '汽车', 'Textile': '纺织服装', 'FoodBever': '食品饮料', 'Electronics': '电子', 'Computer': '计算机', 'LightIndus': '轻工制造', 
       'Utilities': '公用事业', 'Telecom': '通信', 'AgriForest': '农林牧渔', 'CHEM': '化工', 'Media': '传媒', 'IronSteel': '钢铁', 'NonBankFinan': '非银金融',            'ELECEQP': '电气设备', 'AERODEF': '国防军工', 'Conglomerates': '综合'
      }

# 申万一级行业代码
SW_ticker = {
       '银行': '801780.ZICN', '房地产': '801180.ZICN', '医药生物': '801150.ZICN', '交通运输': '801170.ZICN', '公用事业': '801160.ZICN', '综合': '801230.ZICN',
       '有色金属': '801050.ZICN', '休闲服务': '801210.ZICN', '家用电器': '801110.ZICN', '机械设备': '801890.ZICN', '商业贸易': '801200.ZICN', '建筑装饰': '801720.ZICN', 
       '建筑材料': '801710.ZICN', '汽车': '801880.ZICN', '纺织服装': '801130.ZICN', '食品饮料': '801120.ZICN', '电子': '801080.ZICN', '计算机': '801750.ZICN', 
       '轻工制造': '801140.ZICN', '通信': '801770.ZICN', '农林牧渔': '801010.ZICN', '化工': '801030.ZICN', '传媒': '801760.ZICN', '钢铁': '801040.ZICN', 
       '采掘': '801020.ZICN', '非银金融': '801790.ZICN', '电气设备': '801730.ZICN', '国防军工': '801740.ZICN'
       }

industry_name = SW_ticker.keys()

# 因子数据的获取
input_frame = get_multi_factor(factor_name, trade_date_list)

# 部分因子方向的调整
input_frame['PB'] = 1. / input_frame['PB']
input_frame['PE'] = 1. / input_frame['PE']
input_frame['PS'] = 1. / input_frame['PS']
input_frame['PCF'] = 1. / input_frame['PCF']
input_frame['REVS20'] = -1 * input_frame['REVS20']
input_frame['REVS60'] = -1 * input_frame['REVS60']
input_frame['DAVOL20'] = -1 * input_frame['DAVOL20']
input_frame['VOL20'] = -1 * input_frame['VOL20']
input_frame['LCAP'] = -1 * input_frame['LCAP']

# 申万行业分类
sw_map_frame = DataAPI.EquIndustryGet(industryVersionCD=u"010303", 
                                      field=[u'secID', 'secShortName', 'industry', 'intoDate', 'outDate', 
                                             'industryName1', 'industryName2', 'industryName3', 'isNew'], pandas="1")
sw_map_frame = sw_map_frame[sw_map_frame.isNew == 1]

# 遍历每个月末日期，对因子进行去极值、空值填充
print('winsorize factor data...')
pool = Pool(processes=8)
date_list = [tdate for tdate in np.unique(input_frame.tradeDate.values) if int(tdate) > 20061231]
dframe_list = pool.map(winsorize_fillna_date, date_list)

# 遍历每个月末日期，利用协程对因子进行标准化，中性化处理
print('standardize & neutralize factor...')
jobs = [gevent.spawn(standardize_neutralize_factor, value) for value in zip(dframe_list, date_list)]
gevent.joinall(jobs)
new_dframe_list = [e.value for e in jobs]
print('standardize neutralize factor finished!')

# 将不同月份的数据合并到一起
all_frame = pd.concat(new_dframe_list, axis=0)
all_frame.reset_index(inplace=True)

all_frame['tradeDate'] = all_frame['tradeDate'].apply(lambda x: datetime.strptime(x, '%Y%m%d').strftime('%Y-%m-%d'))
all_frame.to_csv('all_frame_neu.csv', index_col=0)
pool.close()
pool.join()

all_frame = pd.read_csv('all_frame_neu.csv', index_col=0)

trade_date = DataAPI.TradeCalGet(beginDate='20090101', endDate='20171231', exchangeCD='XSHG')
trade_month_date = trade_date[trade_date['isMonthEnd'] == 1]['calendarDate'].tolist()


'''
2.细分行业下的因子测试
2.1 测试方法

2.1.1 因子库

我们在优矿的因子库当中，精选了26个因子，如下：
图片注释
因子的详细计算方式在因子文档当中，本文不再赘述。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
2.1.2 测试方法

股票池：

行业内动态股票池
上市6个月以上
去除ST，*ST，以及ST摘帽不足3个月的股票
预处理：

去极值：我们首先对因子进行去极值的处理，这里，正如我们前面所说的，极值的处理使用分行业下的去极值操作，去极值的方法采用MAD法，关于极值的处理方法在选股因子数据的异常值处理和正态转换进行了详细的说明
中性化：行业内因子也可能收到市值风格上的干扰，为了排除市值的影响，本文统一做市值+行业中性化
标准化：传统的z-score方法
有效因子的筛选：

计算过去滚动12个月的IC均值和IC_IR
筛选IC均值高于0.04的股票
筛选IC_IR高于0.25的股票
因子的合成：

采用IC_IR加权的方式进行因子的合成
组合构建及回测参数设置：

等权构建组合
多头：行业内排名前30%；空头：行业内排名后30%
回测区间：2011年1月-2017年12月
月度调仓
基准选择申万对应的行业指数
测试的内容:

全市场模型与分行业模型多头组合的收益率，IR，多空组合的收益率和IR
展示了两个模型的多头组合，空头组合，多空组合的净值曲线
统计了，每期大类因子加权的平均权重，从而可以推断当前行业下比较有效的大类因子
注意：

在测试因子之前，需要当心模型引入前视偏差的问题。例如，如果我们在2017年年底进行因子的有效性分析，并将筛选出的因子用于历史回测，这样便存在明显的未来数据，原因在于我们得到的哪些因子有效的结论是从2017年底得到的，这是一个样本内的结论，将这样的结论应用在样本内并进行回测，显然是不够科学和准确的。本文极力避免未来数据的引入，采用滚动的因子筛选的方法
   调试 运行
文档
 代码  策略  文档
2.2 行业测试

2.2.1 全市场选股模型

在行业测试之前，我们动态的构建全市场选股模型。选择过去一年IC均值高于0.04，IC_IR高于0.25的因子，使用IC_IR加权的方式进行合成。

'''

all_universe = {}
for date in all_frame['tradeDate'].unique():
    all_universe[date] = list(all_frame[all_frame['tradeDate'] == date]['secID'].unique())
    
_, weighted_factor, _, _ , ic_ir_weight = industry_test(all_frame, all_universe, forward_month_ret, False)

'''

2.2.1 银行业

从 2008年10月开始，银行股在沪深300成分股中的总权重就始终处在15%以上，对于指数的影响很大，如果组合银行行业表现优于沪深300的银行业，那么对沪深300指数增强策略的业绩提升有很大帮助。
金融行业的因子逻辑与其它行业不太一样，单独建模必要性很强。
本文这里抛砖引玉，除了传统的因子之外，还应深挖行业内的专属因子。这一块，等我们银行业专属因子做好，可以针对银行业再出相关报告。

'''


bank = {}
indu = SwIndu('银行')
for date in trade_month_date:
    bank[date] = indu.preview(date)
    
benchmark = DataAPI.MktIdxdGet(indexID=u"801780.ZICN", beginDate=u"20110131", endDate=u"20171229", field=u"tradeDate,CHGPct",pandas="1").set_index('tradeDate')['CHGPct']
benchmark.index = pd.to_datetime(benchmark.index)

IC, bank_factor, perf1_indu, perf2_indu, weight = industry_test(all_frame, bank, forward_month_ret, back_test=True)
perf1_all, perf2_all = all_indu_test(weighted_factor, bank, forward_month_ret, '2011-01-31', '2017-12-29')


'''
从上面的结果来看：

相对于全市场选股模型来说，行业内选股模型的多头年化超额收益为13.7%，IR为0.95，而全市场模型多头年化超额收益仅为5.3%，IR为0.4
同时行业内选股模型的多空年化超额收益为7%，IR为0.91，而全市场模型几乎为0，IR为0.033
以上两点说明银行业内的选股模型较全市场模型效果要好很多
银行业内，全市场模型的因子方向并不稳定。2012年末至2015年中，空头组合的表现要优于多头组合的表现
银行业内，估值，盈利因子的预测能力要高于其它
   调试 运行
文档
 代码  策略  文档
2.2.3 非银金融

'''

non_bank_finan = {}
indu = SwIndu('非银金融')
for date in trade_month_date:
    non_bank_finan[date] = indu.preview(date)
    
benchmark = DataAPI.MktIdxdGet(indexID=u"801790.ZICN", beginDate=u"20110131", endDate=u"20171229", field=u"tradeDate,CHGPct",pandas="1").set_index('tradeDate')['CHGPct']
benchmark.index = pd.to_datetime(benchmark.index)

IC, non_bank_finan_factor, perf1_indu, perf2_indu, weight = industry_test(all_frame, non_bank_finan, forward_month_ret)
perf1_all, perf2_all = all_indu_test(weighted_factor, non_bank_finan, forward_month_ret, '2011-01-31', '2017-12-29')


p = performance(perf1_indu, perf2_indu, perf1_all, perf2_all, benchmark)

print (p.to_html())
fig = plt.figure(figsize=(18, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1 = plot(perf1_indu, perf2_indu, benchmark, ax=ax1, y_upper=5, title=u'分行业模型')
ax2 = plot(perf1_all, perf2_all, benchmark, ax=ax2, y_upper=5, title=u'全市场模型')
plt.show()

style_weight = pd.merge(ic_ir_weight.abs().sum(axis=0).to_frame('weight'), factor_group, left_index=True, right_on=u'因子在优矿信号库当中的代码').groupby(u'因子类别').mean()
style_weight['weight'] = style_weight['weight'] / style_weight['weight'].sum()

print ('\n全市场模型大类因子权重占比')
pie_plot(style_weight['weight'])


'''
从上面的结果来看：

对于非银金融行业，行业内建模的效果不及全行业模型
非银金融行业内，动量，流动性两类因子的预测能力最强，估值因子次之
   调试 运行
文档
 代码  策略  文档
2.2.3 家电行业

'''

house_app = {}
indu = SwIndu('家用电器')
for date in trade_month_date:
    house_app[date] = indu.preview(date)
    
benchmark = DataAPI.MktIdxdGet(indexID=u"801110.ZICN", beginDate=u"20110131", endDate=u"20171229", field=u"tradeDate,CHGPct",pandas="1").set_index('tradeDate')['CHGPct']
benchmark.index = pd.to_datetime(benchmark.index)

IC, house_app_factor, perf1_indu, perf2_indu, weight = industry_test(all_frame, house_app, forward_month_ret)
perf1_all, perf2_all = all_indu_test(weighted_factor, house_app, forward_month_ret, '2011-01-31', '2017-12-29')


p = performance(perf1_indu, perf2_indu, perf1_all, perf2_all, benchmark)
print (p.to_html())

fig = plt.figure(figsize=(18, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1 = plot(perf1_indu, perf2_indu, benchmark, ax=ax1, y_upper=8, title=u'分行业模型')
ax2 = plot(perf1_all, perf2_all, benchmark, ax=ax2, y_upper=8, title=u'全市场模型')
plt.show()

style_weight = pd.merge(weight.abs().sum(axis=0).to_frame('weight'), factor_group, left_index=True, right_on=u'因子在优矿信号库当中的代码').groupby(u'因子类别').mean()
style_weight['weight'] = style_weight['weight'] / style_weight['weight'].sum()

print ('\n分行业模型大类因子权重占比')
pie_plot(style_weight['weight'])


'''

从上面的结果来看：

相对于全市场选股模型来说，行业内选股模型的多头年化超额收益为33%，IR为1.98541，而全市场模型多头年化超额收益为30.4%，IR为1.76469
同时行业内选股模型的多空年化超额收益为11.4%，IR为0.8329，而全市场模型为5.3%，IR为0.39
从多头组合超额收益来看，两者相差不大；从多空收益来看，分行业模型显然更好一些
家电行业内，动量，流动性等技术类因子的预测能力最强，估值因子次之
   调试 运行
文档
 代码  策略  文档
2.2.4 休闲服务

'''

lei_service = {}
indu = SwIndu('休闲服务')
for date in trade_month_date:
    lei_service[date] = indu.preview(date)
    
benchmark = DataAPI.MktIdxdGet(indexID=u"801210.ZICN", beginDate=u"20110131", endDate=u"20171229", field=u"tradeDate,CHGPct",pandas="1").set_index('tradeDate')['CHGPct']
benchmark.index = pd.to_datetime(benchmark.index)

IC, lei_service_factor, perf1_indu, perf2_indu, weight = industry_test(all_frame, lei_service, forward_month_ret)
perf1_all, perf2_all = all_indu_test(weighted_factor, lei_service, forward_month_ret, '2011-01-31', '2017-12-29')


p = performance(perf1_indu, perf2_indu, perf1_all, perf2_all, benchmark)

print (p.to_html())
fig = plt.figure(figsize=(18, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1 = plot(perf1_indu, perf2_indu, benchmark, ax=ax1, y_upper=5, title=u'分行业模型')
ax2 = plot(perf1_all, perf2_all, benchmark, ax=ax2, y_upper=5, title=u'全市场模型')

'''

从上面的结果来看：

对于休闲服务行业，行业内建模的效果不及全行业模型
   调试 运行
文档
 代码  策略  文档
2.2.5 传媒行业

'''

media = {}
indu = SwIndu('传媒')
for date in trade_month_date:
    media[date] = indu.preview(date)
    
benchmark = DataAPI.MktIdxdGet(indexID=u"801760.ZICN", beginDate=u"20110131", endDate=u"20171229", field=u"tradeDate,CHGPct",pandas="1").set_index('tradeDate')['CHGPct']
benchmark.index = pd.to_datetime(benchmark.index)

IC, media_factor, perf1_indu, perf2_indu, weight = industry_test(all_frame, media, forward_month_ret)
perf1_all, perf2_all = all_indu_test(weighted_factor, media, forward_month_ret, '2011-01-31', '2017-12-29')


p = performance(perf1_indu, perf2_indu, perf1_all, perf2_all, benchmark)

print (p.to_html())
fig = plt.figure(figsize=(18, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1 = plot(perf1_indu, perf2_indu, benchmark, ax=ax1, y_upper=9)
ax2 = plot(perf1_all, perf2_all, benchmark, ax=ax2, y_upper=9)
plt.show()

style_weight = pd.merge(weight.abs().sum(axis=0).to_frame('weight'), factor_group, left_index=True, right_on=u'因子在优矿信号库当中的代码').groupby(u'因子类别').mean()
style_weight['weight'] = style_weight['weight'] / style_weight['weight'].sum()

print ('\n大类因子权重占比')
pie_plot(style_weight['weight'])


'''
从上面的结果来看：

相对于全市场选股模型来说，行业内选股模型的多头年化超额收益为15.8%，IR为0.99，而全市场模型多头年化超额收益为5.4%，IR为0.32
同时行业内选股模型的多空年化超额收益为14.1%，IR为1.32，而全市场模型为7.4%，IR为0.66
显然，分行业建模效果要优于全行业模型
传媒行业内，动量因子的预测能力最强，估值，流动性因子次之，两者权重占比也几乎相同
   调试 运行
文档
 代码  策略  文档
2.3 分行业模型表现对比

下面我们对申万一级行业中的各个行业用上面的方法进行分行业建模，并和全市场模型，以及不同行业之间进行纵向与横向对比

'''

s = time.time()
all_perf = pd.DataFrame()

for i, indu_name in enumerate(SW_name.values()):
    current_indu = {}
    indu = SwIndu(indu_name)
    for date in trade_month_date:
        current_indu[date] = indu.preview(date)

    benchmark = DataAPI.MktIdxdGet(indexID=SW_ticker[indu_name], beginDate=u"20110131", endDate=u"20171229", 
                                   field=u"tradeDate,CHGPct",pandas="1").set_index('tradeDate')['CHGPct']
    benchmark.index = pd.to_datetime(benchmark.index)
    
    IC, current_indu_factor, perf1_indu, perf2_indu, weight = industry_test(all_frame, current_indu, forward_month_ret)
    perf1_all, perf2_all = all_indu_test(weighted_factor, current_indu, forward_month_ret, '2011-01-31', '2017-12-29')
    
    p = performance(perf1_indu, perf2_indu, perf1_all, perf2_all, benchmark)
    p = (p.T.unstack()).to_frame(indu_name).T
    all_perf = all_perf.append(p)
e = time.time()

print ("Time cost(minutes):", (e - s) / 60)

print(all_perf)

'''
小结：

从上表可以多空组合的IR来看，传媒，有色金属，医药生物，化工，电子，商业贸易，家用电器，食品饮料，综合，计算机，房地产，银行这几个行业，基于全市场模型得到的因子并不完全适用，例如对于食品饮料行业，基于全市场模型构建的多空组合IR为负，而分行业模型得到的多空组合IR为0.75，相差明显
而采掘，非银金融，钢铁，通信，国防军工，休闲服饰，建筑装饰等行业内，行业内模型不如全市场模型
剩下的行业两者之间差别不大，简单起见，可直接使用全市场模型
   调试 运行
文档
 代码  策略  文档
3. 行业因子的整合
前面我们分析得到对部分行业，采用分行业建模的效果要优于全行业模型。因此，我们将这些行业进行分行业模型，而其他行业采用全市场模型，由此得到因子得分，并由此进一步构建出组合。
为了说明模型效果，本文以沪深300成份股内的指数增强组合和中证500的指数增强组合为例。
如果以指数成分股范围内进行行业有效性的研究，那么可能会因为行业内股票数量的问题，使得检验的结果未必可靠。我们对全市场范围内的行业进行测试，能够增加检验结果的准确性
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 因子整合

   调试 运行
文档
 代码  策略  文档
我们使用分行业模型来计算传媒，有色金属，医药生物，化工，电子，商业贸易，家用电器，食品饮料，综合，计算机，房地产，银行这几个行业的因子得分


'''

new_factor = pd.DataFrame()
final_factor = weighted_factor.copy(deep=True)
for i, name in enumerate(['传媒', '有色金属', '医药生物', '商业贸易', '家用电器', '食品饮料', '综合', '计算机', '房地产', '银行']):
    industry_universe = {}
    indu = SwIndu(name)
    for date in trade_month_date:
        industry_universe[date] = indu.preview(date)
    industry_factor = industry_test(all_frame, industry_universe, forward_month_ret)[1]
    new_factor = new_factor.add(industry_factor, fill_value=0)
    
final_factor[new_factor.columns] = new_factor
final_factor = final_factor.dropna(how='all', axis=0)

final_factor.tail()

hs300_weighted_factor = pd.DataFrame()
hs300_final_factor = pd.DataFrame()
for date in trade_month_date:
    hs300_date_universe = set_universe('HS300', date)
    hs300_weighted_factor = hs300_weighted_factor.append(weighted_factor.loc[date, hs300_date_universe].to_frame(date).T)
    hs300_final_factor = hs300_final_factor.append(final_factor.loc[date, hs300_date_universe].to_frame(date).T)
    
    
zz500_weighted_factor = pd.DataFrame()
zz500_final_factor = pd.DataFrame()
for date in trade_month_date:
    zz500_date_universe = set_universe('ZZ500', date)
    zz500_weighted_factor = zz500_weighted_factor.append(weighted_factor.loc[date, zz500_date_universe].to_frame(date).T)
    zz500_final_factor = zz500_final_factor.append(final_factor.loc[date, zz500_date_universe].to_frame(date).T)
    
    
hs300_final_factor_ic = get_rank_ic(hs300_final_factor, forward_month_ret)['IC']
hs300_weighted_factor_ic = get_rank_ic(hs300_weighted_factor, forward_month_ret)['IC']

zz500_final_factor_ic = get_rank_ic(zz500_final_factor, forward_month_ret)['IC']
zz500_weighted_factor_ic = get_rank_ic(zz500_weighted_factor, forward_month_ret)['IC']

ic_summary = pd.DataFrame(index=[['HS300', 'HS300', 'ZZ500', 'ZZ500'], ['分行业模型', '全市场模型', '分行业模型', '全市场模型']], 
                          columns=['IC', '年化ICIR', 'IC为正期数', '总期数'])

ic_summary.loc['HS300'].loc['分行业模型']= [hs300_final_factor_ic.mean(), 
                           hs300_final_factor_ic.mean() / hs300_final_factor_ic.std() * np.sqrt(12), 
                           len(hs300_final_factor_ic[hs300_final_factor_ic > 0]), 
                           len(hs300_final_factor_ic)]
ic_summary.loc['HS300'].loc['全市场模型'] = [hs300_weighted_factor_ic.mean(), 
                       hs300_weighted_factor_ic.mean() / hs300_weighted_factor_ic.std()* np.sqrt(12), 
                       len(hs300_weighted_factor_ic[hs300_weighted_factor_ic > 0]), 
                       len(hs300_weighted_factor_ic)]
ic_summary.loc['ZZ500'].loc['分行业模型']= [zz500_final_factor_ic.mean(), 
                           zz500_final_factor_ic.mean() / zz500_final_factor_ic.std() * np.sqrt(12), 
                           len(zz500_final_factor_ic[zz500_final_factor_ic > 0]), 
                           len(zz500_final_factor_ic)]
ic_summary.loc['ZZ500'].loc['全市场模型'] = [zz500_weighted_factor_ic.mean(), 
                       zz500_weighted_factor_ic.mean() / zz500_weighted_factor_ic.std()* np.sqrt(12), 
                       len(zz500_weighted_factor_ic[zz500_weighted_factor_ic > 0]), 
                       len(zz500_weighted_factor_ic)]

print (ic_summary.to_html())

'''
小结：

可以看到我们从优矿因子库精选出的因子，并通过全市场模型得到的合成的因子IC很高
对于沪深300成分股分行业模型的IC，IC_IR均要高于全市场模型，体现在因子的预测能力，稳定性会更好
对于中证500成分股分行业模型的IC均值要低于全市场模型，IC_IR要高于全市场模型，体现在因子的稳定性会更好
   调试 运行
文档
 代码  策略  文档
3.2 组合构建及回测

我们根据分行业模型整合得到的因子进行组合构建，每个行业内取前10%的股票，组合行业基准与基准的行业权重保持一致，采用流通市值加权的方式构建组合。

   调试 运行
文档
 代码  策略  文档
3.2.1 沪深300指数增强组合回测分析

'''

final_factor_weights = pd.DataFrame()
weighted_factor_weights = pd.DataFrame()

universe = DynamicUniverse('HS300')
for date in final_factor.index:
    date_ = date.replace('-', '')
    signal1 = hs300_final_factor.ix[date, universe.preview(date)].dropna()
    signal2 = hs300_weighted_factor.ix[date, universe.preview(date)].dropna()
    # 组合构建                
    wts1 = pd.Series(long_only(signal1, select_type=1, top_ratio=0.1, weight_type=1, target_date=date, universe_type='HS300')).to_frame(date).T
    wts2 = pd.Series(long_only(signal2, select_type=1, top_ratio=0.1, weight_type=1, target_date=date, universe_type='HS300')).to_frame(date).T
    final_factor_weights = final_factor_weights.append(wts1)
    weighted_factor_weights = weighted_factor_weights.append(wts2)
    
    
start = '2011-02-01'                       # 回测起始时间
end = '2017-12-31'                         # 回测结束时间

benchmark = 'HS300'                        # 策略参考标准
universe = DynamicUniverse('HS300')            # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔

factor_dates = weighted_factor_weights.index.values

# 配置账户信息，支持多资产多账户
accounts = {
    'weighted_factor_account': AccountConfig(account_type='security', capital_base=10000000),
    'final_factor_account': AccountConfig(account_type='security', capital_base=10000000)
}

def initialize(context):                   # 初始化虚拟账户状态
    pass

def handle_data(context):                  # 每个交易日的买入卖出指令
    weighted_factor_account = context.get_account('weighted_factor_account')  
    final_factor_account = context.get_account('final_factor_account') 
    pre_date = context.previous_date.strftime("%Y-%m-%d")
    if pre_date not in weighted_factor_weights.index:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 组合构建                
    wts1 = weighted_factor_weights.ix[pre_date, :].dropna().to_dict()
    wts2 = final_factor_weights.ix[pre_date, :].dropna().to_dict()
    
    # 全市场模型交易部分
    current_position = weighted_factor_account.get_positions(exclude_halt=True)    
    
    # 卖出当前持有，但目标持仓没有的部分
    for stk in set(current_position).difference(wts1):
        weighted_factor_account.order_to(stk, 0)

    for stk in sorted(wts1, key=wts1.get):
        weighted_factor_account.order_pct_to(stk, wts1[stk])
        
    # 分行业模型交易部分
    current_position = final_factor_account.get_positions(exclude_halt=True)    
    
    # 卖出当前持有，但目标持仓没有的部分
    for stk in set(current_position).difference(wts2):
        final_factor_account.order_to(stk, 0)

    for stk in sorted(wts2, key=wts2.get):
        final_factor_account.order_pct_to(stk, wts2[stk])
        
        
weighted_factor_alpha = cal_alpha_by_bt(bt_by_account['weighted_factor_account'])
final_factor_alpha = cal_alpha_by_bt(bt_by_account['final_factor_account'])

print ('全市场模型：')
detail_performance(weighted_factor_alpha)
print ('---------')
print ('分行业模型：')
detail_performance(final_factor_alpha)

# 读取回测数据
data_weighted_factor, underwater_weighted_factor = get_pf(bt_by_account['weighted_factor_account'])
data_final_factor, underwater_final_factor = get_pf(bt_by_account['final_factor_account'])


# 画图展示
fig = plt.figure(figsize=(12, 6))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.grid(True)
ax1.set_ylim(-0.12, 0.12)
ax1.fill_between(underwater_final_factor.index, 0, np.array(underwater_final_factor), color='r')
ax1.fill_between(underwater_weighted_factor.index, 0, np.array(underwater_weighted_factor),alpha=0.5, color='b')

(data_final_factor['excess']-1).plot(ax=ax2, label='分行业模型', color='r', fontsize=20)
(data_weighted_factor['excess']-1).plot(ax=ax2, label='全市场模型', color='b', fontsize=20)
ax2.set_ylim(-1.5, 1.5)
ax2.legend(loc='best', prop=font)
s = ax1.set_title(u"对冲组合超额收益走势（曲线图）", fontproperties=font, fontsize=16)
s = ax1.set_ylabel(u"回撤（柱状图）", fontproperties=font, fontsize=16)
s = ax2.set_ylabel(u"累计超额收益（曲线图）", fontproperties=font, fontsize=16)
s = ax1.set_xlabel(u"日期", fontproperties=font, fontsize=16)


final_factor_weights = pd.DataFrame()
weighted_factor_weights = pd.DataFrame()

universe = DynamicUniverse('ZZ500')
for date in zz500_final_factor.index:
    date_ = date.replace('-', '')
    signal1 = zz500_final_factor.ix[date, universe.preview(date)].dropna()
    signal2 = zz500_weighted_factor.ix[date, universe.preview(date)].dropna()
    # 组合构建                
    wts1 = pd.Series(long_only(signal1, select_type=1, top_ratio=0.1, weight_type=1, target_date=date, universe_type='ZZ500')).to_frame(date).T
    wts2 = pd.Series(long_only(signal2, select_type=1, top_ratio=0.1, weight_type=1, target_date=date, universe_type='ZZ500')).to_frame(date).T
    final_factor_weights = final_factor_weights.append(wts1)
    weighted_factor_weights = weighted_factor_weights.append(wts2)
    
    
start = '2011-02-01'                       # 回测起始时间
end = '2017-12-31'                         # 回测结束时间

benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('ZZ500')            # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔

factor_dates = weighted_factor_weights.index.values

# 配置账户信息，支持多资产多账户
accounts = {
    'weighted_factor_account': AccountConfig(account_type='security', capital_base=10000000),
    'final_factor_account': AccountConfig(account_type='security', capital_base=10000000)
}

def initialize(context):                   # 初始化虚拟账户状态
    pass

def handle_data(context):                  # 每个交易日的买入卖出指令
    weighted_factor_account = context.get_account('weighted_factor_account')  
    final_factor_account = context.get_account('final_factor_account') 
    pre_date = context.previous_date.strftime("%Y-%m-%d")
    if pre_date not in weighted_factor_weights.index:            # 因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
    # 组合构建                
    wts1 = weighted_factor_weights.ix[pre_date, :].dropna().to_dict()
    wts2 = final_factor_weights.ix[pre_date, :].dropna().to_dict()
    
    # 全市场模型交易部分
    current_position = weighted_factor_account.get_positions(exclude_halt=True)    
    
    # 卖出当前持有，但目标持仓没有的部分
    for stk in set(current_position).difference(wts1):
        weighted_factor_account.order_to(stk, 0)

    for stk in sorted(wts1, key=wts1.get):
        weighted_factor_account.order_pct_to(stk, wts1[stk])
        
    # 分行业模型交易部分
    current_position = final_factor_account.get_positions(exclude_halt=True)    
    
    # 卖出当前持有，但目标持仓没有的部分
    for stk in set(current_position).difference(wts2):
        final_factor_account.order_to(stk, 0)

    for stk in sorted(wts2, key=wts2.get):
        final_factor_account.order_pct_to(stk, wts2[stk])
        

weighted_factor_alpha = cal_alpha_by_bt(bt_by_account['weighted_factor_account'])
final_factor_alpha = cal_alpha_by_bt(bt_by_account['final_factor_account'])

print ('全市场模型：')
detail_performance(weighted_factor_alpha)

print ('---------')
print ('分行业模型：')
detail_performance(final_factor_alpha)

# 读取回测数据
data_weighted_factor, underwater_weighted_factor = get_pf(bt_by_account['weighted_factor_account'])
data_final_factor, underwater_final_factor = get_pf(bt_by_account['final_factor_account'])


# 画图展示
fig = plt.figure(figsize=(12, 6))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.grid(True)
ax1.set_ylim(-0.12, 0.12)
ax1.fill_between(underwater_final_factor.index, 0, np.array(underwater_final_factor), color='r')
ax1.fill_between(underwater_weighted_factor.index, 0, np.array(underwater_weighted_factor), alpha=0.5, color='b')

(data_final_factor['excess']-1).plot(ax=ax2, label='分行业模型', color='r', fontsize=20)
(data_weighted_factor['excess']-1).plot(ax=ax2, label='全市场模型', color='b', fontsize=20)
ax2.set_ylim(-2, 2)
ax2.legend(loc='best', prop=font)
s = ax1.set_title(u"对冲组合超额收益走势（曲线图）", fontproperties=font, fontsize=16)
s = ax1.set_ylabel(u"回撤（柱状图）", fontproperties=font, fontsize=16)
s = ax2.set_ylabel(u"累计超额收益（曲线图）", fontproperties=font, fontsize=16)
s = ax1.set_xlabel(u"日期", fontproperties=font, fontsize=16)

'''
小结：

对于沪深300增强组合：

从2011年初至2017年末，分行业模型超额年化收益率，IR几乎为全市场模型的两倍，而最大回撤仅为全市场模型的1/2
除了2014年，分行业模型超额收益均高于全市场模型，在2014年4月份以前，全市场模型几乎没有收益，而分行业模型持续稳定的获得超额收益
在2017年市场风格发生巨大变化的情况下，全市场模型持续回撤，而分行业模型要好很多
分行业模型较全市场模型，对沪深300成分股内指数增强组合有着显著的提升效果
对于中证500增强组合：

从2011年初至2017年末，分行业模型超额年化收益率，IR，最大回撤等性能指标依然优于全市场模型，但没有沪深300那么显著
   调试 运行
文档
 代码  策略  文档
总结
本文从优矿的因子库当中精选了规模，动量，波动性，流动性，估值，成长，盈利等不同维度下，共26个比较公认的有效因子
在传统的多因子模型的基础上，本文做了进一步的扩展。传统的多因子模型，在为了控制组合的风险的基础上，都会加行业中性的处理。这隐含着一个前提假设是因子在行业内也具有预测的能力，但这种假设并不完全成立
就像分域研究一样，有的因子在小市值范围内预测能力较好，而在大市值范围内却没有预测能力一样
沪深300域内，金融行业占比很高，保证组合的权重行业表现要优于基准也是策略成功的关键，因此分行业建模重要性不言而喻，本文最终的实验结果也验证了这一点
在选择的这些因子上，对部分行业验证分行业模型与全市场模型的不同。同时，为了避免未来数据的引入，本文采用动态的模型。从结果来看，部分行业采用分行业模型效果更好
还有一部分行业，分行业模型不如全市场模型，原因可能在于本文的因子库构建并不完全。例如，对于非银金融、银行等，因为经营业务的不同，一些在其它行业或者全行业有效的因子并不完全适用，需要单独寻找一些符合行业特征的因子；同时，针对不同行业，本文采取了同样一套因子的筛选准则，但是这并不一定对所有行业都适用，这也是导致模型是否有效的原因之一
因子合成的方式有很多，传统的静态，动态，以及机器学习的方法合成因子，事实上本文也提供了另外一种思路，通过部分行业，进行行业内的有效因子的配权，进而提升组合的业绩表现
综合来说，本文只是对分行业建模进行了初步的尝试，验证了这一想法的可行性。读者如果感兴趣，还可以针对一些特殊的行业，例如银行，构建特有的因子
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''

