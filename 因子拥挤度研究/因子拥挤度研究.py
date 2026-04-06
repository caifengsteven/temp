# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 07:32:44 2020

@author: Asus
"""
'''

导读
A. 研究目的：本文利用优矿提供的行情数据和的因子数据，参考海通证券《选股因子系列（四十二）————因子失效预警：因子拥挤》（原作者：冯佳睿、袁林青）和 《MSCI INTEGRATED FACTOR CROWDING MODEL: Assessing Crowding Risks in Equity Factor Strategies》（原作者：George Bonne等）中的研究方法，对研报的结果进行了实证分析，重点回测了4类拥挤度指标：1）估值价差，2）配对相关性，3）长期累积收益，4）多空波动率比率，最终合成复合拥挤度指标。

B. 研究结论：

基于2007-01-01至2018-11-30月度调仓回测结果，4个拥挤度指标中，估值价差、长期累积收益、多空波动率比率与未来因子收益负相关，配对相关性与未来因子收益负相关不显著。拥挤度对因子未来收益在中长期上影响最明显。拥挤度与未来收益波动无明显正相关性。

将3个拥挤度指标合成复合拥挤度指标，大部分因子的复合拥挤度指标与未来收益负相关，比4个指标更稳定，中长期影响明显。复合拥挤度与未来收益波动仍然无明显正相关。

从因子净值来看，拥挤度的增加，大概率出现大幅回撤。且因子净值的局部高点都能从因子拥挤度预先观测到。但不同因子的拥挤度分布不同，海外通常设定的阈值[-1,1]并不通用。

本文提出的因子拥挤度指标，在收益上的预测并不明显，但是可以作为风险管理的辅助工具，从一定程度上解释因子投资回撤的原因。

C. 文章结构：本文共分为3个部分，具体如下

一、基础数据准备，选取几个常见因子数据，并计算这些因子的因子收益率，未来因子收益、未来收益波动率。

二、计算4个拥挤度指标：1）估值价差，2）配对相关性，3）长期累积收益，4）多空波动率比率。回测他们对未来收益、未来收益波动的预测效果。

三、将估值价差、长期累积收益、多空波动率比率3个拥挤度指标等权合成复合拥挤度指标，回测复合拥挤度指标对未来收益、未来收益波动的预测效果。从因子净值角度展示拥挤度在因子失效预警上的效果。

D. 时间说明

总耗时10分钟左右
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
https://uqer.datayes.com/community/share/9sM4tSmTlqlcS7KLKSfnADGEt880/private；密码：7137
请在运行之前，克隆上面的代码，并存成lib（右上角->另存为lib,不要修改名字）

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


第一部分：基础数据、函数准备
该部分耗时 4分钟
该部分内容为：

1.1 获取原始行情数据，以及基础函数准备。
1.2 获取几个常见因子数据，并计算这些因子的在不同时间段内的未来收益和未来收益波动率。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

1.1 行情数据、基础函数准备



'''

import pandas as pd
import numpy as np
import time
import scipy.stats as st
import matplotlib.pyplot as plt
#import lib.quant_util as qutil
#from CAL.PyCAL import *


#coding=utf-8

# coding=utf-8

############################################################################################
# ---------------------深度报告工具函数目录---------------------------
# 1. 数据IO
#    get_data_items(universe_list, date_list, factor_list, [adj, thread_count, use_datacube])  取优矿中的因子库因子数据
#    add_indu_col(dframe, [indu_name])                                                         获取行业分类:在dataframe后增加一列，表示对应的申万行业分类
#    stock_special_tag(start_date, end_date, [halt, st, pre_new, pre_new_length])              获取个股标签信息(停牌,ST,次新股)：某一时间区间内，根据股票的是否满足某些条件，打上标签

# 2. 信号处理
#    zscore_by_indu(dframe, col_list, [indu_name])                                             各个因子在行业内进行标准化(ZSCORE)
#    fillna_indu_median(dframe, col_list, [indu_name])                                         用行业内中位数填充因子空值
#    netralize_dframe(dframe, col_list, [exclude_style])                                       批量因子中性化处理:对风险模型的风格因子和行业因子进行中性化
#    mad_winsorize(dframe, col_list, [sigma_n])                                                因子去极值处理: 绝对中位数差去极值
#    fin_data_pit2cont(pit_data_frame, sdate, edate)                                           将PIT数据转成时间连续数据  
#    signal_grouping(signal_df, factor_name, ngrp)                                             因子分组， 每天根据因子值将股票进行等分


# 3. 信号分析
#    calc_ic(factor_df, return_df, factor_list, [return_col_name, ic_type])                    给定factor_df， return_df，计算对于的IC
#    monthly_factor_ic(factor_df, factor_list, [start_date, end_date, ic_type, month_len])	   输入因子的dataframe，计算月度因子的IC序列（未来1个月，n个月，可自定义）

# 4. 信号合成
#    multifactor_icir_comb(factor_df, factor_list, window, [ic_type, month_len,...])           根据过去N期的IC_IR，得到因子的权重和加权得到的因子值

# 5. 组合回测
#    get_performance(bt, [excess])															   根据优矿的回测结果（或者类似的回测数据）计算净值和回撤
#    long_short_backtest(signal_df, return_df, factor_name, return_name, [direction])          简易回测（不考虑停牌、涨跌停无法交易）：因子多组合回测/纯多头组合回测
#    easy_backtest(signal_df, return_df, factor_name, return_name, return_name, [method,...])  简易因子回测组合， 根据因子值将个股等分成n组，指定回测方式，可进行多空回测或纯多头回测。
#    simple_group_backtest(signal_df, return_df, factor_name, return_name, [ngrp])             对因子进行简单的分组多头回测。返回各组收益率和累计收益率， 编号越大，因子值越大。


############################################################################################

from multiprocessing.dummy import Pool as ThreadPool
import time
import pandas as pd
import numpy as np
#from quartz_extensions import neutralize, standardize, winsorize
import gevent


############################################################################################
# Usage: get_data_items(set_universe("A"), ['20070101', '20080104'], ['LCAP', 'PE'])
############################################################################################
# 取优矿中的因子库因子数据
def get_data_items(universe_list, date_list, factor_list, adj=None, thread_count=16, use_datacube=False):
    '''
    universe_list: ['000001.XSHE', '600036.XSHG', ...]
    date_list:数据日期列表，["2007001", "20180706", '...']
    factor_list: 要取的数据列表(data_cube支持的)
    adj: 数据复权方式（比如取closeprice时）， None/pre
    thread_count: 取数据的线程数，默认16个
    返回:
        frame_list:[frame_t0, frame_t1, ...frame_tn], frame_tn为tn日对应的因子dataframe
        frame_tn的列为: ticker, tradeDate, factor_list, tradeDate格式为"%Y%m%d"
    '''

    t_start = time.time()
    pool = ThreadPool(processes=16)

    # 获取给定日期的因子信息
    def get_factor_by_day(parms):
        '''
        参数：
            params = [my_universe, tdate, data_item_list]
            my_universe: secID的列表
            tdate: 时间， %Y%m%d
            data_item_list: 要取的数据列表
        返回:
            DataFrame, 返回给定日期的因子值
        '''

        tdate, data_item_list, my_universe = parms

        cnt = 0
        while True:
            try:
                if use_datacube:
                    data = get_data_cube(my_universe, ['ticker', 'tradeDate'] + data_item_list, tdate, tdate,
                                         style='tas', adj=adj)
                    tmp_frame = data[tdate]
                else:
                    tmp_frame = DataAPI.MktStockFactorsOneDayProGet(tradeDate=tdate, secID=u"", ticker=u"",
                                                                    field=['ticker', 'tradeDate'] + data_item_list,
                                                                    pandas="1")
                tmp_frame['tradeDate'] = tdate.replace("-", "")
                return tmp_frame

            except Exception as e:
                cnt += 1
                print ("get data failed in get_factors, reason:%s, retry again, retry count:%s" % (e, cnt))
                if cnt >= 3:
                    print ("max get data retry, will exit")
                    raise Exception(e)
            return

    pool_args = zip(date_list, [factor_list] * len(date_list), [universe_list] * len(date_list))
    frame_list = pool.map(get_factor_by_day, pool_args)
    pool.close()
    pool.join()
    t_end = time.time()
    print ("[quant_util.get_data_items] finished!, time cost:%s" % (t_end - t_start))
    return frame_list


############################################################################################
# Usage: add_indu_col(factor_frame, indu_name='industryName1')
############################################################################################
# 在dataframe后增加一列，表示对应的申万行业分类
def add_indu_col(dframe, indu_name='industryName1'):
    '''
    dframe: panel/横截面/时间序列数据，至少包含[ticker, tradeDate]列， tradeDate为"%Y%m%d"格式
    返回：
          dframe，增加一列，标识对应的申万行业分类
    '''
    # 先拿到申万一级行业的分类
    sw_frame = DataAPI.EquIndustryGet(ticker=np.unique(dframe.ticker.values), industryVersionCD=u"010303",
                                      field=["ticker", indu_name, 'intoDate'], pandas="1")
    sw_frame['tradeDate'] = sw_frame['intoDate'].apply(lambda x: x.replace("-", ""))

    # 标志dframe原有的行
    dframe['original_row'] = 1

    # 合并行业分类
    dframe = dframe.merge(sw_frame[['ticker', 'tradeDate', indu_name]], on=['ticker', 'tradeDate'], how='outer')
    # 排序后，按股票的历史上行业分类进行前向填充
    dframe.sort_values(by=['ticker', 'tradeDate'], ascending=[True, True], inplace=True)
    dframe[indu_name] = dframe.groupby(['ticker']).apply(lambda x: x[indu_name].fillna(method='ffill')).values

    # 删除非dframe原有的行，保证输入输出的日期是一样的
    dframe.dropna(subset=['original_row'], inplace=True)
    del dframe['original_row']
    return dframe


############################################################################################
# Usage: zscore_by_indu(factor_frame,['LCAP', 'PE'])
############################################################################################
# 各个因子在行业内进行标准化(ZSCORE)
def zscore_by_indu(dframe, col_list, indu_name='industryName1'):
    '''
    dframe: panel/横截面/时间序列数据, 列至少包括: ['ticker','tradeDate', col_list], tradeDate为 "%Y%m%d"
    col_list: 需要进行中性化的因子列表
    返回：
         dframe，和输入dframe相比，多了indu_name一列
    '''
    # 得到对应的行业分类
    dframe = add_indu_col(dframe, indu_name=indu_name)

    # 对df的col_list每一列进行zscore标准化
    def zscore_frame(df, col_list):
        df[col_list] = (df[col_list] - df[col_list].mean()) / df[col_list].std()
        return df

    # 按行业进行ZSCORE
    dframe = dframe.groupby(['tradeDate', indu_name]).apply(zscore_frame, col_list)
    return dframe


############################################################################################
# Usage: fillna_indu_median(factor_frame,['LCAP', 'PE'])
############################################################################################
# 用行业内中位数填充因子空值
def fillna_indu_median(dframe, col_list, indu_name='industryName1'):
    '''
    dframe: panel/横截面/时间序列数据, 至少包含 ['ticker', 'tradeDate', col_list], tradeDate为"%Y%m%d"
    col_list: 需要进行中性化的因子列表
    返回：
        经过空值填充的dframe
    '''
    if indu_name not in dframe.columns:
        dframe = add_indu_col(dframe, indu_name=indu_name)

    # 中位数填充空值
    def fill_na_media(df, col):
        df[col] = df[col].fillna(df[col].median())
        return df

    dframe = dframe.groupby(['tradeDate', indu_name]).apply(fill_na_media, col_list)
    return dframe


############################################################################################
# Usage: netralize_dframe(factor_frame,['LCAP', 'PE'], exclude_stype=['BETA', 'SIZE', 'Bank'])
############################################################################################
def netralize_dframe(dframe, col_list, exclude_style=[]):
    '''
    dframe: panel/横截面/时间序列数据, 列至少包括['ticker', 'tradeDate', col_list]
    col_list: 需要进行中性化的因子列表
    exclude_style: 不进行中性的风格
    返回：
         经过中性化后的dframe
    '''

    # 在某一天对col_list的每一个因子进行中性化
    def neutralize_by_date(params):
        '''
        params=[dframe_by_tdate, col_list, exclude_style]
        dframe_by_tdate: tdate日的dframe，列至少包括['ticker', 'tradeDate', col_list]
        exclude_style: 不进行中性化的风格, list
        '''
        dframe_by_tdate, col_list, exclude_style = params
        tdate = dframe_by_tdate.tradeDate.values[0]
        # 对每个因子进行中性化
        for col in col_list:
            if len(dframe_by_tdate[col].dropna()) < 11:
                # print "Netralize skipped for %s, %s because  too many nan factor values" %(col, tdate)
                continue
            dframe_by_tdate[col] = neutralize(dframe_by_tdate[col], target_date=tdate, exclude_style_list=exclude_style)
        return dframe_by_tdate

    dframe.set_index('ticker', inplace=True)
    # 将dframe拆成list，便于利用协程加快计算
    col_lists = []
    frame_list = []
    exclude_lists = []
    for tdate, tdframe in dframe.groupby(['tradeDate']):
        col_lists.append(col_list)
        frame_list.append(tdframe)
        exclude_lists.append(exclude_style)
    # 利用协程进行计算
    jobs = [gevent.spawn(neutralize_by_date, value) for value in zip(frame_list, col_lists, exclude_lists)]
    gevent.joinall(jobs)
    new_frame_list = [result.value for result in jobs]
    dframe = pd.concat(new_frame_list, axis=0)
    dframe.reset_index(inplace=True)
    return dframe


############################################################################################
# Usage: netralize_dframe(factor_frame,['LCAP', 'PE'], sigma_n=3)
############################################################################################
# 绝对中位数差法
def mad_winsorize(dframe, col_list, sigma_n=3):
    '''
    dframe: panel/横截面/时间序列数据, 列至少包括: ['ticker','tradeDate', col_list], tradeDate为 "%Y%m%d"
    col_list: 需要进行winsorize的因子列表
    '''

    def mad_winsor_by_day(dframe_tdate, col_list, sigma_n):
        '''
        按照[dm+sigma_n*dm1, dm-sigma_n*dm1]进行winsorize
        dm: median
        dm1: median(abs(origin_data - median)), 即 MAD值
        参数:
            dframe_tdate: 某一期的多个因子值的dataframe
        返回:
            去极值后的dframe_tdate
        '''
        dm = dframe_tdate[col_list].median()
        dm1 = (dframe_tdate[col_list] - dm).abs().median()

        upper = dm + sigma_n * dm1
        lower = dm - sigma_n * dm1
        for col in col_list:
            tmp_col = dframe_tdate[col]
            tmp_col[tmp_col > upper[col]] = upper[col]
            tmp_col[tmp_col < lower[col]] = lower[col]
            dframe_tdate[col] = tmp_col
        return dframe_tdate

    dframe = dframe.groupby(['tradeDate']).apply(mad_winsor_by_day, col_list, sigma_n)
    return dframe


############################################################################################
# Usage: calc_ic(factor_frame, return_df, ['LCAP', 'PE'], ic_type='spearman')
############################################################################################
# 给定factor_df， return_df，计算对于的IC
def calc_ic(factor_df, return_df, factor_list, return_col_name='target_return', ic_type='spearman'):
    """
    计算因子IC值, 本月和下月因子值的秩相关
    params:
            factor_df: DataFrame, columns=['ticker', 'tradeDate', factor_list]
            return_df: DataFrame, colunms=['ticker, 'tradeDate'， return_col_name], 预先计算好的未来的收益率
            factor_list:　list， 需要计算IC的因子名list
            return_col_name: str, return_df中的收益率列名
            method: : {'spearman', 'pearson'}, 默认'spearman', 指定计算rank IC('spearman')或者Normal IC('pearson')
    return:
            DataFrame, 返回各因子的IC序列， 列为: ['tradeDate', factor_list]
    """
    merge_df = factor_df.merge(return_df, on=['ticker', 'tradeDate'])
    # 遍历每个因子，计算对应的IC
    factor_ic_list = []
    for factor_name in factor_list:
        tmp_factor_ic = merge_df.groupby(['tradeDate']).apply(
            lambda x: x[[factor_name, return_col_name]].corr(method=ic_type).values[0, 1])
        tmp_factor_ic.name = factor_name
        factor_ic_list.append(tmp_factor_ic)
    factor_ic_frame = pd.concat(factor_ic_list, axis=1)
    factor_ic_frame.reset_index(inplace=True)
    return factor_ic_frame


############################################################################################
# Usage: monthly_factor_ic(factor_frame,['LCAP', 'PE'], month_len=3)
############################################################################################
# 输入因子的dataframe，计算月度因子的IC序列（未来1个月，n个月，可自定义）
def monthly_factor_ic(factor_df, factor_list, start_date=None, end_date=None, ic_type='spearman', month_len=1):
    '''
    factor_df: panel/横截面/时间序列数据, 列至少包括: ['ticker','tradeDate', factor_list], tradeDate为 "%Y%m%d", 必须为月末日期
    factor_list: 需要计算IC的factor名list
    start_date: 返回的IC序列的最早时间，默认为None，和factor_df的最早时间保持一致；如果不为None, 格式为"%Y%m%d, 必须为月末日期
    end_date: 返回的IC序列的最大时间，默认为None，和factor_df的最大时间保持一致；如果不为None, 格式为"%Y%m%d， 必须为月末日期
    ic_type: spearman/pearson
    month_len: 计算IC时，看和未来N期收益的关系
    返回：
         IC的dataframe，columns为：[tradeDate, factor1_name, factor2_name,..., factorn_name]]
    '''
    if start_date is None:
        start_date = min(factor_df.tradeDate.values)
    else:
        start_date = max(str(start_date).replace("-", ""), min(factor_df.tradeDate.values))

    if end_date is None:
        end_date = max(factor_df.tradeDate.values)
    else:
        end_date = min(str(end_date).replace("-", ""), max(factor_df.tradeDate.values))
    factor_df = factor_df.query("(tradeDate>=@start_date) & (tradeDate<=@end_date)")

    # 由于计算IC用到未来期的收益，所以取行情数据的截止日应该比因子的截止日多month_len期
    date_frame = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=end_date, field=u"", pandas="1")
    date_frame = date_frame.query("isMonthEnd==1")
    if len(date_frame) < (month_len + 1):
        raise Exception(u"计算月度IC时，交易日历中取不到%s的下个月月末日期，请检查%s是否为月末交易日" % (end_date, end_date))
    data_end_date = date_frame.head(month_len + 1).calendarDate.values[-1].replace("-", "")

    ticker_list = list(np.unique(factor_df.ticker.values))

    # 获得月收益率
    month_return = DataAPI.MktEqumGet(ticker=ticker_list, beginDate=start_date, endDate=data_end_date,
                                      field=["ticker", "endDate", "closePrice"], pandas="1")
    month_return.rename(columns={'endDate': 'tradeDate'}, inplace=True)
    month_return['tradeDate'] = month_return['tradeDate'].apply(lambda x: x.replace("-", ""))
    month_return.sort_values(['ticker', 'tradeDate'], inplace=True)
    # 计算未来month_len期的累计收益率
    month_return['target_closePrice'] = month_return.groupby('ticker')['closePrice'].shift(-1 * month_len)
    month_return['target_return'] = (month_return['target_closePrice'] - month_return['closePrice']) / month_return[
        'closePrice']
    month_return = month_return[['ticker', 'tradeDate', 'target_return', 'closePrice']]
    month_return.dropna(inplace=True)

    # 得到IC值
    factor_ic_frame = calc_ic(factor_df, month_return, factor_list)
    factor_ic_frame = factor_ic_frame[['tradeDate'] + factor_list]
    factor_return_frame = factor_df.merge(month_return, on=['ticker', 'tradeDate'])
    return factor_ic_frame, factor_return_frame


############################################################################################
# Usage: multifactor_icir_comb(factor_frame,['LCAP', 'PE'], 3, month_len=3)
############################################################################################
# 根据过去N期的IC_IR，得到因子的权重和加权得到的因子值
def multifactor_icir_comb(factor_df, factor_list, window, ic_type='spearman', month_len=1, start_date=None,
                          end_date=None):
    '''
    factor_df: panel数据, 列至少包括: ['ticker','tradeDate', factor_list], tradeDate为 "%Y%m%d", 必须为月末日期
    factor_list: 参与权重分配的factor名list
    start_date: 返回权重的最早时间，默认为None，和factor_df的最早时间保持一致；如果不为None, 格式为"%Y%m%d, 必须为月末日期
    end_date: 返回的权重的最大时间，默认为None，和factor_df的最大时间保持一致；如果不为None, 格式为"%Y%m%d， 必须为月末日期
    ic_type: spearman/pearson
    返回：
         factor_weight_frame： 列为: ['tradeDate', factor_name1, factor_name2, ...factor_nameN], 同一个tradeDate，权重之和为1
         factor_frame：加上了合成因子值后的factor_frame, 列为['ticker', 'tradeDate', factor_list(原始因子值), 'multifactor_comb_value']
    '''
    # 调整factor_df的index，防止有duplicated的index
    ori_factor_df_index = factor_df.index.values
    factor_df.index = range(len(factor_df))
    factor_df = factor_df[['ticker', 'tradeDate'] + factor_list]
    # 得到因子每个月的IC
    factor_ic_frame, factor_return_frame = monthly_factor_ic(factor_df, factor_list)
    # 计算IC_IR值
    factor_ic_frame.sort_values(by=['tradeDate'], inplace=True)
    factor_icir_frame = factor_ic_frame.copy()
    factor_icir_frame[factor_list] = factor_ic_frame[factor_list].shift(month_len).rolling(window=window).apply(
        lambda x: x.mean() / x.std())
    # 得到因子的权重值（根据横截面的IC_IR做归一化）, 权重frame的列为
    factor_weight_frame = factor_icir_frame.copy()
    # for factor_name in factor_list:
    #     factor_weight_frame[factor_name] = factor_icir_frame[factor_name]/factor_icir_frame[factor_list].sum(axis=1)

    # 将因子权重乘以原始因子值，得到合成之后的因子值
    factor_df = factor_df.merge(factor_weight_frame, on=['tradeDate'], how='left', suffixes=("", "_weight"))
    weight_cols = [x + "_weight" for x in factor_list]
    factor_df['multifactor_comb_value'] = (np.array(factor_df[factor_list]) * (np.array(factor_df[weight_cols]))).sum(
        axis=1)

    if start_date is None:
        start_date = min(factor_df.tradeDate.values)
    else:
        start_date = max(str(start_date).replace("-", ""), min(factor_df.tradeDate.values))

    if end_date is None:
        end_date = max(factor_df.tradeDate.values)
    else:
        end_date = min(str(end_date).replace("-", ""), max(factor_df.tradeDate.values))
    factor_df = factor_df.query("(tradeDate>=@start_date) & (tradeDate<=@end_date)")
    factor_weight_frame = factor_weight_frame.query("(tradeDate>=@start_date) & (tradeDate<=@end_date)")
    return factor_df, [factor_weight_frame, factor_return_frame]


############################################################################################
# Usage: fin_data_pit2cont(factor_frame,'20160101', '20171231')
############################################################################################
# 将PIT数据转成连续数据
def fin_data_pit2cont(pit_data_frame, sdate, edate):
    """
    将PIT数据转成连续数据
    pit_data_frame: 财务报表数据, column= ['ticker','pub_date',[fin_value]], index=num, pub_date='%Y%m%d'
    sdate: 起始时间, '%Y%m%d'
    edate: 终止时间, '%Y%m%d'
    返回：
         连续日的因子值dataframe, 列为：['ticker','pub_date',[fin_value]]
    """

    trade_date_frame = DataAPI.TradeCalGet(exchangeCD=u"XSHE", beginDate='20060101', endDate=edate,
                                           field=['calendarDate', 'isOpen'])
    trade_date_frame.rename(columns={"calendarDate": "pub_date"}, inplace=True)
    trade_date_frame['pub_date'] = trade_date_frame['pub_date'].apply(lambda x: str(x).replace('-', ''))

    tmp_frame = pit_data_frame.groupby(['ticker']).apply(lambda x: x.merge(trade_date_frame,
                                                                           on=['pub_date'], how='outer'))
    del tmp_frame['ticker']
    tmp_frame.reset_index(inplace=True)
    del tmp_frame['level_1']

    tmp_frame = tmp_frame.sort_values(by=['ticker', 'pub_date'], ascending=True)
    tmp_frame = tmp_frame.groupby(['ticker']).apply(lambda x: x.fillna(method='pad'))
    tmp_frame.dropna(inplace=True)
    tmp_frame = tmp_frame[tmp_frame.pub_date >= sdate]
    tmp_frame = tmp_frame[tmp_frame.isOpen == 1]
    del tmp_frame['isOpen']
    return tmp_frame


############################################################################################
# Usage: stock_special_tag('20160101', '20171231')
############################################################################################
# 某一时间区间内，根据股票的是否满足某些条件，打上标签
def stock_special_tag(start_date, end_date, halt=1, st=1, pre_new=1, pre_new_length=60):
    '''
    某一时间区间内，根据股票的是否满足某些条件，打上标签
    start_date: 起始时间, %Y%m%d
    end_date: 结束时间, %Y%m%d
    halt: 停牌
    st: 正处于ST状态
    pre_new: 次新股
    pre_new_length: 定义新股上市后 pre_new_length的股票为次新股
    返回：
         tag_df：包含标签的dataframe， 列为： ['ticker', 'tradeDate', 'special_flag']
         special_flag为：{如果停牌，则为'halt'， 如果ST，则为'ST', 如果次新股，则为'new'}，一个股票在同一天如果满足多个条件，会有多条记录（多行）
    '''
    # 获取交易日历
    trade_calendar = DataAPI.TradeCalGet(exchangeCD=U"XSHG", field=u"calendarDate,isOpen,isMonthEnd")

    # 获得交易日历
    calendar = trade_calendar[trade_calendar['isOpen'] == 1]
    calendar = calendar['calendarDate'].tolist()

    # 次新股
    new_df = pd.DataFrame()
    if pre_new:
        ipo_info = DataAPI.SecIDGet(assetClass=u"E", field=['ticker', 'listDate'], pandas="1")
        ipo_info.dropna(inplace=True)
        ticker_list = [ticker for ticker in ipo_info['ticker'] if len(ticker) == 6 and ticker[0] in ['0', '3', '6']]
        ipo_info = ipo_info[ipo_info['ticker'].isin(ticker_list)]
        ipo_info['permit_date'] = [
            calendar[calendar.index(date) + int(pre_new_length)] if date in calendar else  calendar[0] for date in
            ipo_info['listDate']]

        calendar = np.array(calendar)
        new_df_list = []
        for date in calendar[(calendar > start_date) & (calendar < end_date)]:
            new_list = ipo_info[(ipo_info['permit_date'] >= date) & (ipo_info['listDate'] <= date)]['ticker'].values
            d_new_df = pd.DataFrame({'tradeDate': [date] * len(new_list), 'ticker': new_list})
            new_df_list.append(d_new_df)

        new_df = pd.concat(new_df_list, axis=0)
        new_df['special_flag'] = 'new'

    # ST股
    st_df = pd.DataFrame()
    if st:
        st_info = DataAPI.SecSTGet(beginDate=start_date, endDate=end_date, field=['tradeDate', 'ticker'], pandas="1")
        st_df = st_info.copy()
        st_df['special_flag'] = 'st'

    # 停牌
    halt_frame = pd.DataFrame()
    if halt:
        halt_info = DataAPI.SecHaltGet(beginDate=start_date, endDate=end_date,
                                       field=['ticker', 'haltBeginTime', 'haltEndTime'], pandas="1")
        halt_info.fillna(calendar[-1], inplace=True)
        halt_info['haltBeginTime'] = halt_info['haltBeginTime'].apply(lambda x: x[:10])
        halt_info['haltEndTime'] = halt_info['haltEndTime'].apply(lambda x: x[:10])

        halt_frame_list = []
        for date in calendar[(calendar > start_date) & (calendar < end_date)]:
            halt_list = halt_info[(halt_info['haltEndTime'] >= date) & (halt_info['haltBeginTime'] <= date)][
                'ticker'].values
            d_halt_df = pd.DataFrame({'tradeDate': [date] * len(halt_list), 'ticker': halt_list})
            halt_frame_list.append(d_halt_df)

        halt_df = pd.concat(halt_frame_list, axis=0)
        halt_df['special_flag'] = 'halt'

    tag_df = pd.concat([new_df, st_df, halt_df], axis=0)
    tag_df = tag_df[['ticker', 'tradeDate', 'special_flag']]
    tag_df['tradeDate'] = tag_df['tradeDate'].apply(lambda x: x.replace("-", ""))
    return tag_df


############################################################################################
# Usage: get_performance(bt)
############################################################################################
# 根据优矿的回测结果（或者类似的回测数据）计算净值和回撤
def get_performance(bt, excess=False):
    '''
    得到回测结果的净值和回撤
    bt: dataframe，columns至少为：['tradeDate', u'portfolio_value',u'benchmark_return']
    excess: 如果为True, 则收益代表超额收益，否则为绝对收益
    返回：
         return_data: 净值序列dataframe, 列为:['tradeDate', 'portfolio_value','portfolio_return','target_return'], 'target_return'为绝对或者超额的累计收益率
         drawback_data:最大回撤序列
    '''
    return_data = bt[[u'tradeDate', u'portfolio_value', u'benchmark_return']].set_index('tradeDate')
    if type(bt.tradeDate.values[0]) == np.datetime64:
        return_data.index = pd.to_datetime(return_data.index)
    return_data['portfolio_return'] = return_data.portfolio_value.pct_change()
    return_data['portfolio_return'].ix[0] = 0
    if excess:
        return_data['target_return'] = return_data.portfolio_return - data.benchmark_return
    else:
        return_data['target_return'] = return_data.portfolio_return
    return_data['target'] = return_data.target_return + 1.0
    return_data['target_return'] = return_data.target.cumprod()
    del return_data['target']

    df_cum_rets = return_data['portfolio_return']
    running_max = np.maximum.accumulate(df_cum_rets)
    drawback_data = -((running_max - df_cum_rets) / running_max)
    return return_data, drawback_data


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
    signal_df_tmp.dropna(subset=[factor_name], inplace=True)
    signal_df_tmp['group'] = signal_df_tmp.groupby('tradeDate')[factor_name].apply(
        lambda x: (x.rank() - 1) / len(x) * ngrp).astype(int)
    return signal_df_tmp
	

def long_short_backtest(signal_df, return_df, factor_name, return_name, direction=1):
    """
    简易因子多空回测组合， 根据因子值将个股等分成5组，根据方向指定， 正向操作：做多因子值最大的一组， 做空因子值最小的一组；反向操作：做空因子值最大的一组， 做多因子值最小的一组。
    根据调仓频率，进行交易，返回最后的累计收益率。
    params:
            signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一类为股票当日的因子值
            return_df: DataFrame, columns=['ticker', 'tradeDate', [period_return]], 收益率，只含有调仓日，以及下期累计收益率
            factor_name:　str, signal_df中因子值的列名
            return_name： str, return_df中收益率的列名
            direction： {1,-1}, 操作方向， 1为正向操作， 2为反向操作， 默认为1
    return:
            DataFrame, columns=['tradeDate', 'cum_ret'], 返回累计收益率
    """
    bt_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'], how='right')

    # 分成五祖, 保留因子值最大和最小的两组
    bt_df.dropna(subset=[factor_name], inplace=True)
    bt_df = signal_grouping(bt_df, factor_name=factor_name, ngrp=5)
    bt_df = bt_df[bt_df['group'].isin([0, 4])]

    # 计算权重：每组等权
    count_df = bt_df.groupby(['tradeDate', 'group']).apply(lambda x: len(x)).reset_index()
    count_df.columns = ['tradeDate', 'group', 'count']
    bt_df = bt_df.merge(count_df, on=['tradeDate', 'group'])
    bt_df['weight'] = 1.0 / bt_df['count']

    # 如果direction=1, 则做多因子值最大的一组， 做空因子值最小的一组；如果direction=-1, 则做空因子值最大的一组， 做多因子值最小的一组
    bt_df.loc[bt_df['group'] == 4, 'weight'] = bt_df.loc[bt_df['group'] == 4, 'weight'] * direction
    bt_df.loc[bt_df['group'] == 0, 'weight'] = bt_df.loc[bt_df['group'] == 0, 'weight'] * (-direction) * 0

    perf = bt_df.groupby('tradeDate').apply(lambda x: sum(x[return_name] * x['weight'])).reset_index()
    perf.columns = ['tradeDate', 'period_ret']
    perf.sort_values('tradeDate', inplace=True)
    perf['cum_ret'] = (perf['period_ret'] + 1).cumprod()

    # 调整时间
    perf['period_ret'] = perf['period_ret'].shift(1)
    perf.fillna(0, inplace=True)
    perf['cum_ret'] = perf['cum_ret'].shift(1)
    perf.fillna(1, inplace=True)

    return perf[['tradeDate', 'period_ret', 'cum_ret']], bt_df


def easy_backtest(signal_df, return_df, factor_name, return_name, method='long_short', direction=1, ngrp=5, weight_schemes=0, weights=None):
    """
    简易因子回测组合， 根据因子值将个股等分成n组，指定回测方式，可进行多空回测或纯多头回测。
    根据方向指定， 正向操作：做多因子值最大的一组， 做空因子值最小的一组；反向操作：做空因子值最大的一组， 做多因子值最小的一组。
    根据调仓频率，进行交易，返回每期收益率和累计收益率。
    params:
            signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一类为股票当日的因子值
            return_df: DataFrame, columns=['ticker', 'tradeDate', [period_return]], 收益率，只含有调仓日，以及下期累计收益率
            factor_name:　str, signal_df中因子值的列名
            return_name： str, return_df中收益率的列名
            method: {'long_only', 'long_only'}, 'long_only'纯多头回测, 'long_short'多空回测
            direction: {1,-1}, 1因子为正向, -1因子为反向
            ngrp: 因子分组的组数， 默认分5组
            weight_schemes: {0,1}. 0等权配置, 1自定义加权配置,需要给定weights
            weights: 当weight_schemes = 1时，weights为权重方式。

    return:
            DataFrame, columns=['tradeDate', 'period_ret', 'cum_ret'], 返回每期收益率和累计收益率
    """
    bt_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'], how='right')

    # 因子分组
    bt_df.dropna(subset=[factor_name], inplace=True)
    bt_df = signal_grouping(bt_df, factor_name=factor_name, ngrp=ngrp)

    if method == 'long_short':
        # 保留因子值最大和最小的两组
        bt_df = bt_df[bt_df['group'].isin([0, ngrp - 1])]
    elif method == 'long_only':
        if direction == 1:
            bt_df = bt_df[bt_df['group'].isin([ngrp - 1])]
        elif direction == -1:
            bt_df = bt_df[bt_df['group'].isin([0])]

    # 加权方式
    if weight_schemes == 0:
        # 计算权重：每组等权
        count_df = bt_df.groupby(['tradeDate', 'group']).apply(lambda x: len(x)).reset_index()
        count_df.columns = ['tradeDate', 'group', 'count']
        bt_df = bt_df.merge(count_df, on=['tradeDate', 'group'])
        bt_df['weight'] = 1.0 / bt_df['count']
    elif weight_schemes == 1:
        # 计算权重：自定义加权
        bt_df = bt_df.merge(weights, on=['ticker', 'tradeDate'])
        bt_df.sort_values(['group', 'tradeDate'], inplace=True)
        bt_df['weight'] = bt_df.groupby(['group', 'tradeDate'])['weight'].apply(lambda x: x / sum(x)).values

    if method == 'long_short':
        # 如果direction=1, 则做多因子值最大的一组， 做空因子值最小的一组；如果direction=-1, 则做空因子值最大的一组， 做多因子值最小的一组
        bt_df.loc[bt_df['group'] == ngrp - 1, 'weight'] = bt_df.loc[bt_df['group'] == ngrp - 1, 'weight'] * direction /2.0
        bt_df.loc[bt_df['group'] == 0, 'weight'] = bt_df.loc[bt_df['group'] == 0, 'weight'] * (-direction) /2.0

    perf = bt_df.groupby('tradeDate').apply(lambda x: sum(x[return_name] * x['weight'])).reset_index()
    perf.columns = ['tradeDate', 'period_ret']
    perf.sort_values('tradeDate', inplace=True)
    perf['cum_ret'] = (perf['period_ret'] + 1).cumprod()

    # 调整时间
    perf['period_ret'] = perf['period_ret'].shift(1)
    perf.fillna(0, inplace=True)
    perf['cum_ret'] = perf['cum_ret'].shift(1)
    perf.fillna(1, inplace=True)

    return perf[['tradeDate', 'period_ret', 'cum_ret']], bt_df

def simple_group_backtest(signal_df, return_df, factor_name, return_name, ngrp=5):
    """
    对因子进行简单的分组多头回测。返回各组收益率和累计收益率， 编号越大，因子值越大。
    参数：
        signal_df: DataFrame, columns=['ticker', 'tradeDate', [factor]], 股票的因子值, factor一类为股票当日的因子值
        return_df: DataFrame, columns=['ticker', 'tradeDate', [period_return]], 收益率，只含有调仓日，以及下期累计收益率
        factor_name:　str, signal_df中因子值的列名
        return_name： str, return_df中收益率的列名
        ngrp: int, 分组数, 默认为5
    返回：
        DataFrame, 列为[’group'， tradeDate', 'period_ret', 'cum_ret'], 返回每期收益率和累计收益率
    """
    bt_df = signal_df.merge(return_df, on=['ticker', 'tradeDate'], how='right')
    
    # 因子分组
    bt_df.dropna(subset=[factor_name], inplace=True)
    bt_df = signal_grouping(bt_df, factor_name=factor_name, ngrp=ngrp)
    
    # 等权
    count_df = bt_df.groupby(['tradeDate', 'group']).apply(lambda x: len(x)).reset_index()
    count_df.columns = ['tradeDate', 'group', 'count']
    bt_df = bt_df.merge(count_df, on=['tradeDate', 'group'])
    bt_df['weight'] = 1.0 / bt_df['count']
    
    perf = bt_df.groupby(['group', 'tradeDate']).apply(lambda x: sum(x[return_name] * x['weight'])).reset_index()
    perf.columns = ['group', 'tradeDate', 'period_ret']
    perf.sort_values(['group', 'tradeDate'], inplace=True)
    perf['cum_ret'] = perf.groupby('group')['period_ret'].apply(lambda x: (x+1).cumprod())
    
    # 调整时间
    perf['period_ret'] = perf.groupby('group')['period_ret'].shift(1)
    perf['period_ret'].fillna(0, inplace=True)
    perf['cum_ret'] = perf.groupby('group')['cum_ret'].shift(1)
    perf['cum_ret'].fillna(1, inplace=True)
    
    return perf


def proc_float_scale(df, col_name, format_str):
    """
    格式化输出
    输入：
        df: DataFrame, 需要格式化的数据
        col_name： list, 需要格式化的列名
        format_str： 格式类型
    """
    for col in col_name:
        for index in df.index:
            df.ix[index, col] = format(df.ix[index, col], format_str)
    return df

start_time = time.time()
print ("该部分进行基础参数设置和数据准备...")
sdate = '20070101'
edate = '20181130'

# 全A投资域
a_universe_list = DataAPI.EquGet(equTypeCD=u"A", field=u"secID",pandas="1")['secID'].tolist()

# 获取月末交易日
cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=sdate, endDate=edate).sort('calendarDate')
monthly_dates_list = cal_dates_df[cal_dates_df['isMonthEnd']==1]['calendarDate'].values.tolist()
monthly_dates_list = map(lambda x: x.replace('-', ''), monthly_dates_list)

# 个股日收益率
dret_df = DataAPI.MktEqudGet(secID=a_universe_list, beginDate=sdate,endDate=edate,field=u"ticker,tradeDate,chgPct",pandas="1")
dret_df['tradeDate'] = dret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
dret_df.rename(columns={'chgPct':'ret'}, inplace=True)
dret_df = dret_df.pivot_table(values='ret', index='ticker', columns='tradeDate')
print ("个股日收益率:", dret_df.head().to_html())

# 个股月度收益率
mret_df = DataAPI.MktEqumAdjGet(beginDate=sdate, endDate=edate, secID=a_universe_list, field=u"ticker,endDate,chgPct", pandas="1")
mret_df.rename(columns={'endDate':'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
mret_df['tradeDate'] = mret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
mret_df['nxt1m_ret'] = mret_df.groupby('ticker')['curr_ret'].shift(-1)
print ("个股未来收益率:", mret_df.head().to_html())

mkt_value_df = qutil.get_data_items(a_universe_list, monthly_dates_list, ['NegMktValue'])
mkt_value_df = pd.concat(mkt_value_df)
mkt_value_df = qutil.mad_winsorize(mkt_value_df,['NegMktValue'])
print ("个股流通市值:", mkt_value_df.head().to_html())

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

1.2 获取几个常见因子数据，并计算未来收益和未来收益波动率

本文选取风险模型中的因子：Beta、动量、市值、盈利、残差波动率、成长、估值、杠杆、流动性，获取因子暴露数据。
设定了因子的方向：Beta、动量、盈利、成长、估值为正向因子，即多头为因子值较大的组合；市值、残差波动率、杠杆、流动性为负向因子，即多头为因子值较小的组合。

'''
start_time = time.time()
print ("该部分获取几个常见因子数据...")

# 常见因子列表
selected_factor_list = ['BETA', 'MOMENTUM', 'SIZE', 'EARNYILD', 'RESVOL', 'GROWTH', 'BTOP', 'LEVERAGE', 'LIQUIDTY']
# 给定因子的多空方向
factor_direction_dict = {'BETA': 1, 'MOMENTUM': 1, 'SIZE': -1, 'EARNYILD': 1, 'RESVOL': -1, 'GROWTH': 1, 'BTOP': 1, 'LEVERAGE': -1, 'LIQUIDTY': -1}
factor_decs_dict = {'BETA': u'高贝塔', 'MOMENTUM': u'动量', 'SIZE': u'小市值', 'EARNYILD': u'高盈利', 'RESVOL': u'低残差波动率', 'GROWTH': u'高成长', 'BTOP': u'低估值', 'LEVERAGE': u'低杠杆', 'LIQUIDTY': u'低流动性'}
#获取因子数据
selected_factor_exp_list = []
for tdate in monthly_dates_list:
    factor_td_df = DataAPI.RMExposureDayGet(tradeDate=tdate,beginDate=u"",endDate=u"",field=['ticker', 'tradeDate' ]+selected_factor_list,pandas="1")
    selected_factor_exp_list.append(factor_td_df)
selected_factor_exp_df = pd.concat(selected_factor_exp_list)
print ('常见选股因子：', selected_factor_exp_df.head().to_html())

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

首先计算因子收益率，然后计算因子未来收益和未来收益波动率。考虑到大多数因子投资者是在月度调仓的框架下进行因子投资的，因此本文在月度调仓的框架计算收益率、波动率。
因子收益率：本文先根据因子值，将个股分成10组，每组按照市值加权构建组合，计算出每组组合的收益率。若因子为正向，则用因子值最大的组合收益率减去因子值最小的组合收益率，得到因子收益率；若因子为负向，则用因子值最小的组合收益率减去因子值最大的组合收益率，得到因子收益率。
因子未来收益： 根据因子收益率，本文计算未来6个月、未来12个月、未来18个月、未来24个月的因子收益率，以及未来6个月、未来第7-12个月、未来第13-18个月、未来第19-24个月的分段因子收益率。
因子未来收益波动率: 因子未来收益波动率：计算给定未来时段内的因子月收益率的标准差，作为因子未来收益的波动率。如未来12个月的收益波动率，为未来12个月的因子月收益率的标准差；未来第7-12个月的收益波动率，为未来第7-12个月的因子月收益率的标准差。


'''


start_time = time.time()
print ("该部分计算因子未来收益和未来收益波动率...")

# 获取因子分组
factor_group_df_dict = {}
for fn in selected_factor_list:
    #将因子等分10组,factor_group_df包含因子值和因子分组,0为因子值最小的一组,9为因子值最大的一组
    factor_group_df = qutil.signal_grouping(selected_factor_exp_df,fn, ngrp=10)[['ticker', 'tradeDate', fn, 'group']]
    factor_group_df_dict[fn] = factor_group_df

def calc_factor_futurn_retstd(factor_df, mret_df, direction=1):
    """
    计算因子未来收益率和未来收益波动率
    params:
            factor_df:　DataFrame, columns=['ticker', 'tradeDate', [factor_name], 'group'], 因子分组数据
            mret_df: DataFrame, columns=['ticker', 'tradeDate', 'curr_ret', 'nxt1m_ret'], 股票月收益率数据
            direction: int, {1, -1}, 1为正向因子，-1为负向因子
    return:
            perf: DataFrame, columns=['nxt_6m_ret','nxt_6m_retstd','nxt_1m_6m_ret','nxt_0m_6m_retstd',...,'fmret'], 返回因子未来收益率和未来收益波动率
    """
    ngrp = factor_df['group'].max()
    factor_df = factor_df[factor_df['group'].isin([0, ngrp])]
    fret_df = factor_df.merge(mret_df, on=['ticker', 'tradeDate'], how='left').merge(mkt_value_df, on=['ticker', 'tradeDate'], how='left')
    fret_df.dropna(inplace=True)
    fret_df['weight'] = fret_df.groupby(['group', 'tradeDate'])['NegMktValue'].apply(lambda x: x / sum(x)).values
    fret_df.loc[fret_df['group'] == ngrp, 'weight'] = fret_df.loc[fret_df['group'] == ngrp, 'weight'] * direction /2.0
    fret_df.loc[fret_df['group'] == 0, 'weight'] = fret_df.loc[fret_df['group'] == 0, 'weight'] * (-direction) /2.0
    perf = fret_df.groupby('tradeDate').apply(lambda x: sum(x['nxt1m_ret'] * x['weight'])).reset_index().dropna()
    perf.columns = ['tradeDate','nxt1m_ret']
    for i in [6, 12,18, 24]:
        perf['nxt_%dm_ret'%i] = perf['nxt1m_ret'].rolling(i).apply(lambda x: (x+1).prod()-1).shift(-i+1).values
        perf['nxt_%dm_retstd'%i] = perf['nxt1m_ret'].rolling(i).apply(lambda x: np.std(x, ddof=1)).shift(-i+1).values
        perf['nxt_%dm_%dm_ret'%(i-5,i)] = perf['nxt1m_ret'].rolling(i).apply(lambda x: (x[i-6:]+1).prod()-1).shift(-i+1).values
        perf['nxt_%dm_%dm_retstd'%(i-5,i)] = perf['nxt1m_ret'].rolling(i).apply(lambda x: np.std(x[i-6:], ddof=1)).shift(-i+1).values
    perf['fmret'] = perf['nxt1m_ret'].shift(1)
    perf.fillna(0, inplace=True)
    return perf.drop(['nxt1m_ret'], axis=1)

# 计算因子未来收益率和波动率
factor_ret_df_dict = {}
factor_fret_df_dict = {}
factor_fretstd_df_dict = {}
for fn in selected_factor_list:
    factor_fret_df = calc_factor_futurn_retstd(factor_group_df_dict[fn], mret_df, direction=factor_direction_dict[fn]).set_index('tradeDate')
    factor_ret_df_dict[fn] = factor_fret_df[['fmret']]
    factor_fret_df_dict[fn] = factor_fret_df[['nxt_%dm_ret'%i for i in [6,12,18,24]]+['nxt_%dm_%dm_ret'%(i-5,i) for i in [6,12,18,24]]]
    factor_fretstd_df_dict[fn] = factor_fret_df[['nxt_%dm_retstd'%i for i in [6,12,18,24]]+['nxt_%dm_%dm_retstd'%(i-5,i) for i in [6,12,18,24]]]
print ('市值因子收益率：', factor_ret_df_dict['SIZE'].head().to_html())
print ('市值因子未来收益：', factor_fret_df_dict['SIZE'].head().to_html())
print ('市值因子未来收益波动率：', factor_fretstd_df_dict['SIZE'].head().to_html())

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''
第二部分：因子拥挤度的四种指标计算及表现
该部分耗时 3分钟
本文定义了4中因子拥挤度的指标：估值价差、配对相关性、长期收益率、多空波动率比率。

该部分内容为：

2.1 估值价差计算，并展示估值价差和因子未来收益、未来收益波动率的相关性表现。

2.2 配对相关性计算，并展示估值价差和因子未来收益、未来收益波动率的相关性表现。

2.3 长期收益率计算，并展示估值价差和因子未来收益、未来收益波动率的相关性表现。

2.4 多空波动率比率计算，并展示估值价差和因子未来收益、未来收益波动率的相关性表现。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

本文先定义了一个因子拥挤度分析的类Analysis_Factor_Crowding，类中包含对因子拥挤度进行分析的一些通用函数。

'''

class Analysis_Factor_Crowding(object):      
    
    @classmethod
    def factor_crowding_history_plot(cls, factor_dict, factor_fc_df_dict, fc_name):
        """
        画图：画出因子拥挤度的历史走势
        params:
                factor_dict: dict, {因子名: 中文描述}
                factor_fc_df_dict:　dict, {因子名: 因子拥挤度df}
                fc_name: str, 因子拥挤度指标名称
        """
        fig = plt.figure(figsize=(18,6))
        ax = fig.add_subplot(111)
        for i in range(len(factor_dict)):
            factor_name = factor_dict.keys()[i]
            ax.plot(pd.to_datetime(factor_fc_df_dict[factor_name].index),factor_fc_df_dict[factor_name], label=factor_dict[factor_name])
        ax.plot(pd.to_datetime(factor_fc_df_dict['SIZE'].index), [0]*len(factor_fc_df_dict['SIZE']), color='k')
        ax.set_ylabel(fc_name,fontproperties=font, fontsize=16)
        ax.set_title(u"%s历史走势"%fc_name, fontproperties=font, fontsize=16)
        leg = ax.legend(loc=0, prop=font)
        plt.setp(leg.get_texts(), fontsize=16)
    
    @classmethod    
    def calc_corr_crowd_ret(cls, factor_fc_df_dict, factor_fret_df_dict, fc_name):
        """
        计算因子拥挤度和未来收益、波动率的相关性
        params:
                factor_fc_df_dict:　dict, {因子名: 因子拥挤度df}
                factor_fret_df_dict: dict, {因子名: 因子未来收益率df(或 因子未来收益波动率df)}
                fc_name: str, 因子拥挤度指标名称
        return:
                factor_fc_ret_corr_df: DataFrame, 相关系数
                factor_fc_ret_corrp_df: DataFrame, 相关性检验P值
        """
        selected_factor_list = factor_fc_df_dict.keys()
        factor_fc_ret_corr_df = []
        factor_fc_ret_corrp_df = []
        for fn in selected_factor_list:
            factor_fc_ret_df = factor_fc_df_dict[fn].merge(factor_fret_df_dict[fn], left_index=True, right_index=True)
            corr_list = []
            p_list = []
            for ret_col in factor_fret_df_dict[fn].columns:
                corr_df = factor_fc_ret_df[[fc_name, ret_col]].dropna()
                corr, p_value = st.pearsonr(corr_df[fc_name], corr_df[ret_col])
                corr_list.append(corr)
                p_list.append(p_value)
            tmp_fc_ret_corr_df = pd.DataFrame(corr_list, index=factor_fc_ret_df.columns[1:]+'_corr', columns=[fn]).T
            tmp_fc_ret_corrp_df = pd.DataFrame(p_list, index=factor_fc_ret_df.columns[1:]+'_p', columns=[fn]).T
            factor_fc_ret_corr_df.append(tmp_fc_ret_corr_df)
            factor_fc_ret_corrp_df.append(tmp_fc_ret_corrp_df)
        factor_fc_ret_corr_df = pd.concat(factor_fc_ret_corr_df)
        factor_fc_ret_corrp_df = pd.concat(factor_fc_ret_corrp_df)
        return factor_fc_ret_corr_df, factor_fc_ret_corrp_df
      
    @classmethod 
    def output_corrdata(cls, corr_df, corrp_df, fc_name, ret_type, factor_decs_dict=factor_decs_dict):
        """
        输出相关性具体数据
        params:
                corr_df:　DataFrame, 相关系数
                fcorrp_df: DataFrame, 相关性检验P值
                ret_type: str, ‘收益’、‘收益波动率’二者中的一个
        """
        fc_cum_df = pd.concat([corr_df.iloc[:, np.arange(4)], corrp_df.iloc[:, np.arange(4)]], axis=1)
        fc_cum_df = proc_float_scale(fc_cum_df, fc_cum_df.columns[:4], ".2f")
        fc_cum_df = proc_float_scale(fc_cum_df, fc_cum_df.columns[4:], ".2%")
        cum_cols = [(a, b) for a in [u'相关性', u'相关性检验P值'] for b in [u'未来6个月', u'未来12个月', u'未来18个月', u'未来24个月']]
        fc_cum_df.columns = pd.MultiIndex.from_tuples(cum_cols)
        fc_cum_df.index = [factor_decs_dict[fn] for fn in fc_cum_df.index]
        fc_cum_df.index.name = u'因子名称'
        fc_period_df = pd.concat([corr_df.iloc[:, np.arange(4, 8)], corrp_df.iloc[:, np.arange(4, 8)]], axis=1)
        fc_period_df = proc_float_scale(fc_period_df, fc_period_df.columns[:4], ".2f")
        fc_period_df = proc_float_scale(fc_period_df, fc_period_df.columns[4:], ".2%")
        period_cols = [(a, b) for a in [u'相关性', u'相关性检验P值'] for b in [u'未来6个月', u'未来第7-12个月', u'未来第13-18个月', u'未来第19-24个月']]
        fc_period_df.columns = pd.MultiIndex.from_tuples(period_cols)
        fc_period_df.index = [factor_decs_dict[fn] for fn in fc_period_df.index]
        fc_period_df.index.name = u'因子名称'
        print ('%s与因子未来累计%s相关性'%(fc_name, ret_type), fc_cum_df.to_html())
        print ('%s与因子未来分段%s相关性'%(fc_name, ret_type), fc_period_df.to_html())
    
    @classmethod
    def corrdata_plot(cls, corr_df, fc_name, ret_type, factor_decs_dict=factor_decs_dict):
        """
        画图：展示因子拥挤度和未来收益（或未来收益波动率）的相关性
        params:
                corr_df:　DataFrame, 相关系数
                fc_name: str, 因子拥挤度指标名称
                ret_type: str, ‘收益’、‘收益波动率’二者中的一个
        """
        ind = np.arange(corr_df.shape[0])
        color_list = ['indianred', 'bisque', 'lightblue','darkseagreen']
        rect_dict = {}
        fig = plt.figure(figsize=(20,6))
        ax1 = fig.add_subplot(121)
        for i in range(4):
            rect_dict[i] = ax1.bar(ind+0.2*i+0.1, corr_df.iloc[:, i], 0.2, color=color_list[i])
        ax1.set_xticks(ind + 0.5)
        ax1.set_xticklabels([factor_decs_dict[fn] for fn in corr_df.index], fontproperties=font, rotation=50)
        ax1.xaxis.tick_top()
        ax1.legend([rect_dict[i] for i in range(4)], [corr_df.columns[i] for i in range(4)], fontsize=15, bbox_to_anchor=(0.8, 0), ncol=2)
        ax1.set_title(u"%s与因子未来%s相关性"%(fc_name, ret_type), fontproperties=font, fontsize=16, y=1.3)
        ax2 = fig.add_subplot(122)
        for i in range(4):
            rect_dict[i] = ax2.bar(ind+0.2*i+0.1, corr_df.iloc[:, i+4], 0.2, color=color_list[i])
        ax2.set_xticks(ind + 0.5)
        ax2.set_xticklabels([factor_decs_dict[fn] for fn in corr_df.index], fontproperties=font, rotation=50)
        ax2.xaxis.tick_top()
        ax2.legend([rect_dict[i] for i in range(4)], [corr_df.columns[i+4] for i in range(4)], fontsize=15, bbox_to_anchor=(0.95, 0), ncol=2)
        ax2.set_title(u"%s与因子未来%s相关性（分段)"%(fc_name, ret_type), fontproperties=font, fontsize=16, y=1.3)
        
    @classmethod
    def factor_crowd_ret_plot(cls, factor_fc_df, factor_ret_df, fc_name, factor_name):
        """
        画图：因子拥挤度与因子净值
        params:
                factor_fc_df:　DataFrame, 因子拥挤度
                factor_ret_df: DataFrame, 因子收益率
                fc_name: str, 因子拥挤度指标名称
                factor_name: str, 因子描述
        """
        merge_df = pd.concat([factor_fc_df, factor_ret_df], axis=1).dropna()
        merge_df.ix[0, 'fmret'] = 0
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.plot(pd.to_datetime(merge_df.index), merge_df[fc_name], label=u'拥挤度（左轴）', color='r')
        ax1.plot(pd.to_datetime(merge_df.index), [1]*len(merge_df), "b--", color='k')
        ax1.plot(pd.to_datetime(merge_df.index), [-1]*len(merge_df), "b--", color='k')
        ax1.set_title(u'%s因子净值与复合拥挤度'%factor_name, fontproperties=font, fontsize=16)
        ax1.set_ylabel(u'拥挤度', fontproperties=font, fontsize=16)
        ax1.legend(bbox_to_anchor=(0.2, 1), prop=font)
        ax1.grid(True)
        ax2.plot(pd.to_datetime(merge_df.index),(merge_df['fmret']+1).cumprod(), label=u'因子净值（右轴）', color='b')
        ax2.set_ylabel(u'净值', fontproperties=font, fontsize=16)
        leg = ax2.legend(bbox_to_anchor=(0.215, 0.9), prop=font)
        plt.setp(leg.get_texts(), fontsize=16)
        ax2.grid(False)
        
'''

2.1 估值价差计算及表现

相关研究认为，资金对于因子度追捧会进一步推进因子多头端端固执或者压低因子空头端的估值水平，由此加大因子多空组合的估值价差。
估值价差的计算方法：
估值价差=log(因子多头估值因子空头估值)
根据因子值对全市场进行排序，若因子为正向，因子多（空）头组合为排序前（后）10%的股票；若因子为负向，因子多（空）头组合为排序后（前）10%的股票。下文的因子多空组合均与此处一样。
本文使用PB因子值来衡量个股估值，因子多头估值为因子多头组合中个股PB的中位数，因子空头估值类似方式计算。
对估值价差进行累积时间序列标准化：每个时间点的均值，为该时间点以前的时间序列的平均数；每个时间点的标准差，为该时间点以前的时间序列的标准差。
standardize估值价差t=估值价差t−mean(估值价差1,2,…,t)std(估值价差1,2,…,t)
下文的因子累积时间序列标准化均与此处一样。

'''

start_time = time.time()
print ("该部分计算常见因子的估值价差，并展示部分因子的历史走势...")

# 获取估值数据（使用PB衡量估值）
pb_df = qutil.get_data_items(a_universe_list, monthly_dates_list, ['PB'])
pb_df = pd.concat(pb_df)
value_df = pb_df.rename(columns={'PB':'value'})

# 函数：计算因子多空的估值价差
def calc_valuation_spread(factor_df, value_df, direction=1):
    ngrp = factor_df['group'].max()
    factor_df = factor_df[factor_df['group'].isin([0, ngrp])]
    vs_df = factor_df.merge(value_df, on=['ticker', 'tradeDate'], how='left')
    vs_df = vs_df.groupby(['group', 'tradeDate'])[['value']].median().reset_index()
    vs_df = vs_df.pivot_table(values=['value'], index=['tradeDate'], columns=['group'])
    vs_df.columns = vs_df.columns.droplevel()
    vs_df['vs'] = np.log((vs_df[ngrp] / vs_df[0])**direction)
    vs_df = vs_df.dropna()
    # 标准化
    vs_df['stand_vs'] = vs_df['vs'].rolling(1000, min_periods=12).apply(lambda x: (x[-1] - np.mean(x))/np.std(x, ddof=1)).values
    vs_df = vs_df[['stand_vs']].rename(columns={'stand_vs': 'vs'})
    return vs_df.dropna()

# 计算常见因子的估值价差
factor_vs_df_dict = {}
for fn in selected_factor_list:
    factor_vs_df = calc_valuation_spread(factor_group_df_dict[fn], value_df, direction=factor_direction_dict[fn])
    factor_vs_df_dict[fn] = factor_vs_df

# 画图
factor_plot_dict = {key: value for key, value in factor_decs_dict.items() if key in ['SIZE', 'BTOP', 'EARNYILD']}
Analysis_Factor_Crowding.factor_crowding_history_plot(factor_plot_dict, factor_vs_df_dict, u'估值价差')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


start_time = time.time()
print ("该部分展示估值价差与因子未来收益的相关性")

# 计算估值价差和未来因子收益的相关性
factor_vs_fret_corr_df, factor_vs_fret_corrp_df = Analysis_Factor_Crowding.calc_corr_crowd_ret(factor_vs_df_dict, factor_fret_df_dict, 'vs')

# 输出具体数据
Analysis_Factor_Crowding.output_corrdata(factor_vs_fret_corr_df, factor_vs_fret_corrp_df, '估值价差', '收益')

# 画图
Analysis_Factor_Crowding.corrdata_plot(factor_vs_fret_corr_df, u'估值价差', u'收益')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

整体上看，除了流动性，其他因子的估值价差和未来收益均呈负相关性。
大部分因子在未来12个月或18个月的负相关性最明显，说明估值价差对因子收益的影响在中长期上更为明显，这也是可以理解的，当市场中出现了一致性后，通常需要一定的时间来消化这种一致性。

'''

start_time = time.time()
print ("该部分展示估值价差与因子未来收益波动率的相关性")

# 计算估值价差和因子未来收益波动率的相关性
factor_vs_fretstd_corr_df, factor_vs_fretstd_corrp_df = Analysis_Factor_Crowding.calc_corr_crowd_ret(factor_vs_df_dict, factor_fretstd_df_dict, 'vs')

# 输出具体数据
Analysis_Factor_Crowding.output_corrdata(factor_vs_fretstd_corr_df, factor_vs_fretstd_corrp_df, '估值价差', '收益波动率')

# 画图
Analysis_Factor_Crowding.corrdata_plot(factor_vs_fretstd_corr_df, u'估值价差', u'收益波动率')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

2.2 配对相关性计算及表现

配对相关性从股票通同涨同跌的特征来度量因子的拥挤程度。相关研究认为，资金对因子或某一类股票的追捧会加剧这一类股票的同涨同跌的特性。
配对相关性的计算方法：
配对相关性=mean(∑i=1Ncorr(r多头,r多头,i))−mean(∑i=1Ncorr(r空头,r空头,i))
其中r多头=mean(r多头,i), r空头=mean(r空头,i).
本文取过去三个月的日收益率数据计算配对相关性指标，且剔除过去三个月内停牌超过10天的成份股。
对配对交易量进行累积时间序列标准化。

'''
start_time = time.time()
print ("该部分计算常见因子的配对相关性...")

# 函数：计算因子的配对相关性
def calc_pairwise_correlation(factor_df, dret_df, monthly_dates_list, direction=1):
    monthly_dates_list = ['20061231'] + monthly_dates_list
    date_list = dret_df.columns.values
    ngrp = factor_df['group'].max()
    factor_df = factor_df[factor_df['group'].isin([0, ngrp])]
    pc_list = []
    for i in range(4, len(monthly_dates_list)):
        # 获取多空组合成分股
        top_position_list = factor_df[(factor_df['tradeDate'] == monthly_dates_list[i]) & (factor_df['group'] == ngrp)]['ticker'].values.tolist()
        bottom_position_list = factor_df[(factor_df['tradeDate'] == monthly_dates_list[i]) & (factor_df['group'] == 0)]['ticker'].values.tolist()
        per_sdate = monthly_dates_list[i-3]
        per_edate = monthly_dates_list[i]
        # 获取成份股日收益率
        tdate_list = date_list[(date_list > per_sdate) & (date_list <=per_edate)]
        ret_df = dret_df.loc[top_position_list+bottom_position_list, tdate_list]
        # 剔除过去一个季度停牌超过10个交易日的成份股
        filter_secID_list = ret_df.index[ret_df.isnull().sum(axis=1) <= 10]
        filter_ret_df = ret_df.loc[filter_secID_list]
        # 计算多、空组合的平均收益率
        filter_ret_df.loc['top_ave_ret', :] = filter_ret_df.loc[top_position_list].mean()
        filter_ret_df.loc['bottom_ave_ret', :] = filter_ret_df.loc[bottom_position_list].mean()
        # 计算成份股corr    
        corr_df = filter_ret_df.T.corr()
        pc = (corr_df.loc[top_position_list, 'top_ave_ret'].mean() - corr_df.loc[bottom_position_list, 'bottom_ave_ret'].mean()) * direction
        pc_list.append(pc)
    
    pc_df = pd.DataFrame(pc_list, columns=['pc'], index=monthly_dates_list[4:])
    pc_df['stand_pc'] = pc_df['pc'].rolling(1000, min_periods=12).apply(lambda x: (x[-1] - np.mean(x))/np.std(x, ddof=1)).values
    pc_df = pc_df[['stand_pc']].rename(columns={'stand_pc': 'pc'})
    return pc_df.dropna()

# 计算常见因子的配对相关性
factor_pc_df_dict = {}
for fn in selected_factor_list:
    factor_pc_df = calc_pairwise_correlation(factor_group_df_dict[fn], dret_df, monthly_dates_list, direction=factor_direction_dict[fn])
    factor_pc_df_dict[fn] = factor_pc_df

# 画图
factor_plot_dict = {key: value for key, value in factor_decs_dict.items() if key in ['SIZE', 'BTOP', 'EARNYILD']}
Analysis_Factor_Crowding.factor_crowding_history_plot(factor_plot_dict, factor_pc_df_dict, u'配对相关性')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

start_time = time.time()
print ("该部分展示配对相关性与因子未来收益的相关性")

# 计算配对相关性和未来因子收益的相关性
factor_pc_fret_corr_df, factor_pc_fret_corrp_df = Analysis_Factor_Crowding.calc_corr_crowd_ret(factor_pc_df_dict, factor_fret_df_dict, 'pc')

# 输出具体数据
Analysis_Factor_Crowding.output_corrdata(factor_pc_fret_corr_df, factor_pc_fret_corrp_df, '配对相关性', '收益')

# 画图
Analysis_Factor_Crowding.corrdata_plot(factor_pc_fret_corr_df, u'配对相关性', u'收益')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

从图上看出，配对相关性和因子未来收益的负相关性并不明显。这可能是因为，在因子拥挤时，多头组合和空头组合的股票收益相关性可能都会增大，它们的差值可能会很小。以这种差值的方式来定义A股市场的因子拥挤度从结果来看效果并不好

'''


start_time = time.time()
print ("该部分展示配对相关性与因子未来收益波动率的相关性")

# 计算配对相关性和因子未来收益波动率的相关性
factor_pc_fretstd_corr_df, factor_pc_fretstd_corrp_df = Analysis_Factor_Crowding.calc_corr_crowd_ret(factor_pc_df_dict, factor_fretstd_df_dict, 'pc')

# 输出具体数据
Analysis_Factor_Crowding.output_corrdata(factor_pc_fretstd_corr_df, factor_pc_fretstd_corrp_df, '配对相关性', '收益波动率')

# 画图
Analysis_Factor_Crowding.corrdata_plot(factor_pc_fretstd_corr_df, u'配对相关性', u'收益波动率')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


'''
从结果来看，相关性有正有负，且P值并不显著，配对相关性与未来因子收益波动的正相关性也不成立。

2.3 长期累计收益计算及表现

因子长期累积收益也是投资者在选择因子时的重要考量因子，因此长期累积收益越高，因子更容易受到资金的追捧。
本文取上文中计算的因子收益率，以36个月为滚动时间窗口来计算长期累计收益率。
对长期累积收益进行滚动时间序列标准化: 每个时间点的均值，为该时间点以前36个月的时间序列的平均数；每个时间点的标准差，为该时间点以前36个月的时间序列的标准差。
standardize长期累积收益t=长期累积收益t−mean(长期累积收益t−35,…,t−1,t)std(长期累积收益t−35,…,t−1,t)
下文的因子滚动时间序列标准化均与此处一样。
'''


start_time = time.time()
print ("该部分计算常见因子的长期累计收益...")

# 函数：计算因子的长期收益率
def calc_longterm_ret(factor_ret_df):
    factor_ret_df = factor_ret_df.copy()
    factor_ret_df['lt_ret'] = factor_ret_df['fmret'].rolling(36).apply(lambda x: (x+1).prod()-1)
    factor_ret_df['stand_lt_ret'] = factor_ret_df['lt_ret'].rolling(36, min_periods=12).apply(lambda x: (x[-1] - np.nanmean(x))/np.nanstd(x, ddof=1)).values
    lt_ret_df = factor_ret_df[['stand_lt_ret']].rename(columns={'stand_lt_ret': 'lt_ret'}).dropna()
    return lt_ret_df

# 计算常见因子的长期收益率
factor_lr_df_dict = {}
for fn in selected_factor_list:
    factor_lr_df = calc_longterm_ret(factor_ret_df_dict[fn])
    factor_lr_df_dict[fn] = factor_lr_df

# 画图
factor_plot_dict = {key: value for key, value in factor_decs_dict.items() if key in ['SIZE', 'BTOP', 'EARNYILD']}
Analysis_Factor_Crowding.factor_crowding_history_plot(factor_plot_dict, factor_lr_df_dict, u'长期累计收益率')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

start_time = time.time()
print ("该部分展示长期累计收益与因子未来收益的相关性")

# 计算长期累计收益和未来因子收益的相关性
factor_lr_fret_corr_df, factor_lr_fret_corrp_df = Analysis_Factor_Crowding.calc_corr_crowd_ret(factor_lr_df_dict, factor_fret_df_dict, 'lt_ret')

# 输出具体数据
Analysis_Factor_Crowding.output_corrdata(factor_lr_fret_corr_df, factor_lr_fret_corrp_df, '长期累计收益', '收益')

# 画图
Analysis_Factor_Crowding.corrdata_plot(factor_lr_fret_corr_df, u'长期累计收益', u'收益')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


'''
整体上看，大部分因子的长期累积收益和未来因子收益呈负相关性（估值、动量、市值除外）
大部分因子在未来18个月或24个月的负相关性最明显，说明长期累积收益对因子收益的影响在长期上更为显著。因为长期累积收益刻画因子收益的长期趋势，“影响在长期上较明显”符合预期。
'''

start_time = time.time()
print ("该部分展示长期累计收益与因子未来收益波动率的相关性")

# 计算长期累计收益和因子未来收益波动率的相关性
factor_lr_fretstd_corr_df, factor_lr_fretstd_corrp_df = Analysis_Factor_Crowding.calc_corr_crowd_ret(factor_lr_df_dict, factor_fretstd_df_dict, 'lt_ret')

# 输出具体数据
Analysis_Factor_Crowding.output_corrdata(factor_lr_fretstd_corr_df, factor_lr_fretstd_corrp_df, '长期累计收益', '收益波动率')

# 画图
Analysis_Factor_Crowding.corrdata_plot(factor_lr_fretstd_corr_df, u'长期累计收益', u'收益波动率')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''

从结果来看，长期累积收益与未来因子收益波动的正相关性也不成立。

 多空波动率比率

海外相关研究认为，资金对于因子的追捧会加剧因子波动，因此使用因子波动率来衡量因子当前的拥挤程度。
多空波动率比率的计算方法：
多空波动率比率=vol(r多头)vol(r空头)
因子多头的波动率，为前3个月的多头成分股等权合成的日收益率数据计算得到，且剔除过去三个月内停牌超过10天的成份股。
对多空波动率比率进行累积时间序列标准化。

'''

start_time = time.time()
print ("该部分计算常见因子的多空波动率比率...")

# 函数：计算因子多空的多空波动率比率
def calc_factor_volatility(factor_df, dret_df, monthly_dates_list, direction=1):
    monthly_dates_list = ['20061231'] + monthly_dates_list
    date_list = dret_df.columns.values
    ngrp = factor_df['group'].max()
    factor_df = factor_df[factor_df['group'].isin([0, ngrp])]
    fv_list = []
    for i in range(4, len(monthly_dates_list)):
        # 获取多空组合成分股
        top_position_list = factor_df[(factor_df['tradeDate'] == monthly_dates_list[i]) & (factor_df['group'] == ngrp)]['ticker'].values.tolist()
        bottom_position_list = factor_df[(factor_df['tradeDate'] == monthly_dates_list[i]) & (factor_df['group'] == 0)]['ticker'].values.tolist()
        per_sdate = monthly_dates_list[i-3]
        per_edate = monthly_dates_list[i]
        # 获取成份股日收益率
        tdate_list = date_list[(date_list > per_sdate) & (date_list <=per_edate)]
        ret_df = dret_df.loc[top_position_list+bottom_position_list, tdate_list]
        # 剔除过去一个季度停牌超过10个交易日的成份股
        filter_secID_list = ret_df.index[ret_df.isnull().sum(axis=1) <= 10]
        filter_ret_df = ret_df.loc[filter_secID_list]
        # 计算多、空组合的平均收益率
        filter_ret_df.loc['top_ave_ret', :] = filter_ret_df.loc[top_position_list].mean()
        filter_ret_df.loc['bottom_ave_ret', :] = filter_ret_df.loc[bottom_position_list].mean()
        # 计算波动率    
        fv = (filter_ret_df.loc['top_ave_ret', :].std() / filter_ret_df.loc['bottom_ave_ret', :].std()) ** direction
        fv_list.append(fv)
    
    fv_df = pd.DataFrame(fv_list, columns=['fv'], index=monthly_dates_list[4:])
    fv_df['stand_fv'] = fv_df['fv'].rolling(24, min_periods=12).apply(lambda x: (x[-1] - np.mean(x))/np.std(x, ddof=1)).values
    fv_df = fv_df[['stand_fv']].rename(columns={'stand_fv': 'fv'})
    return fv_df.dropna()
        
# 计算常见因子的多空波动率比率
factor_fv_df_dict = {}
for fn in selected_factor_list:
    factor_fv_df = calc_factor_volatility(factor_group_df_dict[fn], dret_df, monthly_dates_list, direction=factor_direction_dict[fn])
    factor_fv_df_dict[fn] = factor_fv_df

# 画图
factor_plot_dict = {key: value for key, value in factor_decs_dict.items() if key in ['SIZE', 'BTOP', 'EARNYILD']}
Analysis_Factor_Crowding.factor_crowding_history_plot(factor_plot_dict, factor_fv_df_dict, u'多空波动率比率')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


start_time = time.time()
print ("该部分展示多空波动率比率与因子未来收益的相关性")

# 计算多空波动率比率和未来因子收益的相关性
factor_fv_fret_corr_df, factor_fv_fret_corrp_df = Analysis_Factor_Crowding.calc_corr_crowd_ret(factor_fv_df_dict, factor_fret_df_dict, 'fv')

# 输出具体数据
Analysis_Factor_Crowding.output_corrdata(factor_fv_fret_corr_df, factor_fv_fret_corrp_df, '多空波动率比率', '收益')

# 画图
Analysis_Factor_Crowding.corrdata_plot(factor_fv_fret_corr_df, u'多空波动率比率', u'收益')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''
从结果来看，多空波动率比率与未来因子收益的负相关性在部分因子上成立，整体来看，拥挤度和未来收益的负相关性没有估值价差、长期累计收益的方式显著，但对于部分因子还是一个不错的拥挤度描述方法。
多空波动率比率在12、18个月的负相关性最明显，说明多空波动率比率对因子收益的影响在中期上较显著。
'''

start_time = time.time()
print ("该部分展示多空波动率比率与因子未来收益波动率的相关性")

# 计算多空波动率比率和因子未来收益波动率的相关性
factor_fv_fretstd_corr_df, factor_fv_fretstd_corrp_df = Analysis_Factor_Crowding.calc_corr_crowd_ret(factor_fv_df_dict, factor_fretstd_df_dict, 'fv')

# 输出具体数据
Analysis_Factor_Crowding.output_corrdata(factor_fv_fretstd_corr_df, factor_fv_fretstd_corrp_df, '多空波动率比率', '收益波动率')

# 画图
Analysis_Factor_Crowding.corrdata_plot(factor_fv_fretstd_corr_df, u'多空波动率比率', u'收益波动率')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))


'''
同样的，多空波动率比率与未来因子收益波动的正相关性不成立。
   调试 运行
文档
 代码  策略  文档
从上述分析可得，估值价差、长期累计收益、多空波动率比率与未来因子收益呈负相关，且中长期影响较为显著。配对相关性对因子未来收益的影响不显著。
所有指标与因子未来收益波动率的正相关均不成立。

第三部分：复合因子拥挤度
该部分耗时 1分钟
因为不同拥挤度指标在不同因子上的效果不同，比如估值价差、多空波动率比率在流动性上失效，但是长期累计收益在流动性上有效；长期累计收益在市值上失效，但是估值价差、多空波动率比率在市值上有效。因此，该部分尝试组合上述3个指标，综合不同角度对拥挤度的刻画。

该部分内容为：

3.1 在第二部分的基础上，简单合成复合拥挤度指标。
3.2 探究复合拥挤度和因子净值的关系
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 复合因子拥挤度计算

将上述四个指标按照等权方式合成复合因子拥挤度。

'''

# 计算常见因子的复合拥挤度
factor_integrated_fc_df_dict = {}
for fn in selected_factor_list:
    factor_integrated_fc_df = pd.concat([factor_vs_df_dict[fn], factor_lr_df_dict[fn], factor_fv_df_dict[fn]], axis=1)
    factor_integrated_fc_df = factor_integrated_fc_df.dropna()
    factor_integrated_fc_df['fc'] = factor_integrated_fc_df.mean(axis=1)
    factor_integrated_fc_df_dict[fn] = factor_integrated_fc_df[['fc']]
    
# 画图
factor_plot_dict = {key: value for key, value in factor_decs_dict.items() if key in ['SIZE', 'BTOP', 'EARNYILD']}
Analysis_Factor_Crowding.factor_crowding_history_plot(factor_plot_dict, factor_integrated_fc_df_dict, u'复合拥挤度')

start_time = time.time()
print ("该部分展示复合拥挤度与因子未来收益的相关性")

# 计算复合拥挤度和未来因子收益的相关性
factor_integrated_fc_fret_corr_df, factor_integrated_fc_fret_corrp_df = Analysis_Factor_Crowding.calc_corr_crowd_ret(factor_integrated_fc_df_dict, factor_fret_df_dict, 'fc')

# 输出具体数据
Analysis_Factor_Crowding.output_corrdata(factor_integrated_fc_fret_corr_df, factor_integrated_fc_fret_corrp_df, '复合拥挤度', '收益')

# 画图
Analysis_Factor_Crowding.corrdata_plot(factor_integrated_fc_fret_corr_df, u'复合拥挤度', u'收益')

end_time = time.time()
print ("耗时: %s seconds" % (end_time - start_time))

'''
整体上看，除了流动性，其他因子的估值价差和未来收益均呈负相关性。
大部分因子在未来12个月或18个月的负相关性明显，且对未来第7到12个月的因子收益影响显著，这也符合我们的预期，合成后的复合拥挤度对因子收益的影响在中长期上更为明显。
   调试 运行
文档
 代码  策略  文档
3.2 拥挤度与因子净值

-下面看复合拥挤度历史走势和因子净值走走势之间的关系。

'''

for fn in selected_factor_list:
    Analysis_Factor_Crowding.factor_crowd_ret_plot(factor_integrated_fc_df_dict[fn],factor_ret_df_dict[fn], 'fc', factor_decs_dict[fn])
    
'''

从上图可以看出，当因子净值出现大幅回落之前，因子拥挤度往往提前预警。当因子从拥挤变为极度拥挤时，因子净值通在未来中长期将迎来回落。例如市值因子，2017年初因子净值出现回落，小市值因子拥挤度从2015年末到2017年初就已经显示出持续拥挤的信号。
海外研究通常将1和-1作为衡量因子拥挤的阈值，但是不同因子的复合拥挤度在[-1,1]之间的分布并不均匀，例如估值通常拥挤度小于0。因此以1和-1为阈值，并不通用。

总结：

总体来看，部分因子发生拥挤后会在中长期对收益有负向影响，但拥挤度达到什么水平后就能确定有负向影响从结果上来看并没有像国外一样有明显的阈值效应，后续可以进一步进行挖掘；
因子拥挤度可以作为风险管理的辅助工具。当因子拥挤度增大，可以根据历史上的拥挤度和后续走势情况，评估后续是否需要关注因子的失效可能。

'''


