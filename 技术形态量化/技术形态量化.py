# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:44:37 2020

@author: Asus
"""
'''
导读
A. 研究目的：本文利用个股的前复权价格数据，参考中信建投-技术形态选股研究之黎明曙光：深跌反转形态-180827(原作者：丁鲁明等)的研究方法，对研报的结果进行了实证分析，用以探寻当出现了深跌反转形态之后的股票后续的走势情况、以及策略构建的收益情况。并在最后对这种形态产生的信号进行因子化探究其表现。

B. 文章结构：本文共分为3个部分，具体如下

一、数据下载和处理

二、统计满足深跌反转形态的个股在未来一段时间内的平均收益和胜率情况

三、根据深跌反转形态的信号构建策略

四、深跌反转因子测试

五、总结

C. 研究结论：

深跌反转形态的个股在触发信号后20个交易日内达到了接近百分之90%的峰值，显示深跌反转信号在刚触发时，该股票有很强的反弹动能。在20个交易日之后，符合深跌反转信号的个股走势出现分化，导致胜率出现降低。在20日后仍然延续强劲反弹势头的个股收益很高，带动了整体的平均收益后续继续走高。
策略在20140101到20191231的回测区间内，共取得128.52%的总收益，年化收益为30.24%，夏普比率为1.05，最大回撤32.99%，胜率较高为73.08%。较高的胜率表明深跌反转形态的后续反弹动能很强。
通过回测发现，深跌反转因子的表现并不理想，IC均值不到1%，五分组没有体现出较好的线性性，并不是一个理想的alpha因子。可能是因子反弹动能衰减，因子的有效性在衰减期内未能维持。
D. 时间说明

本文主要分为五个部分，第一部分约耗时130分钟，其它部分耗时均在15分钟以内，总耗时在180分钟左右
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
链接：https://uqer.datayes.com/community/share/eLNeQy0p3r0lRu9I5WoZ5YOw2ng0/private；密码：6278
请在运行之前，克隆上面的代码，并存成lib(右上角->另存为lib，不要修改名字)

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


第一部分：数据下载和处理
该部分耗时 130分钟
该部分内容为：

读取禁投股票池。

读取从2013-2019的A股所有个股的前复权价格数据。

找出历史上所有的符合深跌反转的形态的股票，记录其发生的时间点以及历史下跌段数。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


1.1 数据下载

根据后续研究的需要，从优矿中下载的数据有：个股2013-2019的个股前复权收盘价数据


'''

import pandas as pd, numpy as np, os, pickle, time
import datetime
from collections import Counter
#import lib.quant_util as quant_util
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from scipy import stats
#from CAL.PyCAL import *
mpl.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei'] 

#建立数据存储文件夹
deep_fall_dir = 'deep_fall'
if not os.path.exists(deep_fall_dir):
    os.makedirs(deep_fall_dir)
    

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

    
##下载个股2013-2019的个股前复权收盘价数据并存储,由于数据量较大，鉴于优矿返回数据条数的限制，这里分段进行数据获取。

start_time = time.time()
print ("下载个股2013-2019的个股前复权收盘价数据并存储...")

start_date, end_date = '20130101', '20191231'
close_df = pd.DataFrame()
for year in range(2013, 2020):
    closedata1 = DataAPI.MktEqudAdjGet(secID=u"",ticker=u"",tradeDate=u"",beginDate=u"{}0101".format(year),endDate=u"{}0331".format(year),
                      isOpen="",field=u"ticker,tradeDate,closePrice",pandas="1")

    closedata2 = DataAPI.MktEqudAdjGet(secID=u"",ticker=u"",tradeDate=u"",beginDate=u"{}0401".format(year),endDate=u"{}0630".format(year),
                      isOpen="",field=u"ticker,tradeDate,closePrice",pandas="1")

    closedata3 = DataAPI.MktEqudAdjGet(secID=u"",ticker=u"",tradeDate=u"",beginDate=u"{}0701".format(year),endDate=u"{}0930".format(year),
                      isOpen="",field=u"ticker,tradeDate,closePrice",pandas="1")
    
    closedata4 = DataAPI.MktEqudAdjGet(secID=u"",ticker=u"",tradeDate=u"",beginDate=u"{}1001".format(year),endDate=u"{}1231".format(year),
                      isOpen="",field=u"ticker,tradeDate,closePrice",pandas="1")
    close_df = pd.concat([close_df, closedata1, closedata2, closedata3, closedata4])
with open('{}/close_adj.pkl'.format(deep_fall_dir), 'w') as f:
    pickle.dump(close_df, f)
    
end_time = time.time()
print ("Time cost: %s seconds" % (end_time - start_time))
print ('个股2013-2019的个股前复权收盘价数据样式如下：')
print (close_df.head().to_html())

'''
1.2 数据处理

   调试 运行
文档
 代码  策略  文档
1.2.1 找出历史上所有的符合深跌反转的形态的股票，记录其发生的时间点以及历史下跌段数

资产价格在经历极大跌幅的下跌之后，往往会进入反弹阶段，即便此时资产价格并未大幅低于其内在价值，也存在市场恐慌程度达到极限之后的反转，能够支持一段时间内的上涨；而在资产价格大幅上涨之后，同样存在回调的需求。
由于同一时段内涨跌幅相近的股票可能具有完全不同的价格路径，而想要研究连续下跌之后的样本表现，就必须考虑股票的走势形态。因此， 以下我们将首先从形态识别出发，根据波浪理论， 找到连续下跌，且跌幅达到一定程度的股票，统计其在此后出现反转的可能性，并尝试根据所选样本构造投资组合。
本文采用滚动窗口局部极值识别法
图片注释
在通过滚动窗口识别法得到高低点序列之后，我们通过一系列的规则筛选出我们所认同的深跌反转的样本，规则如下：
 1. 序列的最后的一个点为滚动窗口中的低点，且选股日距低点d1-d2个交易日（d1、d2分别表示选股日距离序列最后一个低点的最小天数和最大天数）；
 2. 从最后的低点向前，至少存在三段下跌；
 3. 每一段下跌的低点都低于前一低点；
 4. 连续下跌的跌幅累计超过fall（fall表示之前下跌时间段内累计跌幅的最小值，若累计跌幅小于fall，则不符合该形态）；
'''
 

#获取交易日历
def get_trade_cal(beginDate, endDate):
    trade_cal = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=u"{}".format(beginDate),
                                    endDate=u"{}".format(endDate),isOpen=u"1",field=u"",pandas="1")
    return trade_cal

#生成高低点序列
def get_high_low_point(price_df, tradingdates):
    '''
    输入：
    price_df: DataFrame， 价格数据，包含列名有 tradeDate, closePrice..
    tradingdates: list, 所有交易日的序列
    返回：
    point_date_df: dataframe，每个个股的高低点序列数据，包含列名有 tradeDate, closePrice, point_date...
    '''
    
    #全序列生成各个高点和低点
    price_df = price_df.sort_values(by='tradeDate')

    # 有些股票某些交易日价格不存在，向前填充
    price_df_period_tradingdays = sorted(
        list(filter(lambda x: x >= price_df['tradeDate'].min() and x <= price_df['tradeDate'].max(), tradingdates)))
    price_df_period_tradingdays = pd.DataFrame(price_df_period_tradingdays, columns=['tradeDate'])
    price_df = price_df.merge(price_df_period_tradingdays, on=['tradeDate'], how='right')
    price_df = price_df.sort_values(by='tradeDate')
    price_df = price_df.fillna(method='ffill')
    price_df.index = range(len(price_df))
    price_df['price_max'] = price_df['closePrice']
    price_df['price_min'] = price_df['closePrice']
    price_df['argmax_ind'] = price_df['price_max'].rolling(window=2 * winsize + 1, min_periods=2 * winsize + 1).apply(
        lambda x: int(x.argmax()))
    price_df['argmin_ind'] = price_df['price_min'].rolling(window=2 * winsize + 1, min_periods=2 * winsize + 1).apply(
        lambda x: int(x.argmin()))
    
    #当天必须是序列的中心 
    price_df_centered = price_df[price_df.apply(lambda x: x['argmax_ind']==winsize or x['argmin_ind']==winsize, axis=1)]

    if len(price_df_centered) == 0:
        return pd.DataFrame()
    price_df_centered['high_point_date'] = price_df_centered.apply(
        lambda x: tradingdates[tradingdates.index(x['tradeDate']) -  winsize ] if x['argmax_ind']==winsize else np.nan, axis=1)
    price_df_centered['low_point_date'] = price_df_centered.apply(
        lambda x: tradingdates[tradingdates.index(x['tradeDate']) - winsize] if x['argmin_ind'] == winsize else np.nan, axis=1)

    price_df_centered['high_flag'], price_df_centered['low_flag'] = 1, -1

    #point_date的price的价格必须是当天的，直接取到的closePrice不是当天的价格
    point_date_df = pd.concat([price_df_centered[['tradeDate', 'high_point_date', 'high_flag', 'argmax_ind']].rename(
                              columns= {'high_point_date': 'point_date', 'high_flag': 'flag', 'tradeDate': 'high_tradeDate'}), 
                              price_df_centered[['tradeDate', 'low_point_date', 'low_flag', 'argmin_ind']].rename(columns={'low_point_date':                                                                    'point_date', 'low_flag': 'flag', 'tradeDate': 'low_tradeDate'})])

    point_date_df = point_date_df.merge(price_df[['tradeDate', 'closePrice']].rename(columns={'tradeDate': 'point_date'}), on='point_date',                                                              how='left')

    point_date_df = point_date_df[point_date_df['point_date'] != 'none']
    point_date_df = point_date_df.sort_values(by=['point_date', 'flag'], ascending=False)
    #去除nan的point_date
    point_date_df = point_date_df.dropna(subset=['point_date'])
    #去除重复的高低点
    point_date_df['flag_diff'] = point_date_df['flag'].diff().fillna(1) #第一个flag的会为nan，但是不应该删除
    point_date_df = point_date_df[point_date_df['flag_diff'] != 0]
    #转为日期从小到大排序
    point_date_df = point_date_df.sort_values(by='point_date')
    
    return point_date_df
    
    
#搜索满足深跌反转形态的个例的函数
def get_example(close_preadj, roll_backdays, winsize, d1, d2, fall, trade_cal):
    '''
    输入：
    closeprice:    DataFrame， 股票的价格数据.
    roll_backdays: int， 回溯时间长度
    d1，d2:        int,    最后一个高低点距离触发点的长度范围
    fall：         float， 此前最小股价下滑幅度
    trade_cal      DataFrame,  交易日历
    返回：
    all_choose_sample_df: dataframe，每个个股符合深跌反转形态的样本，包含列名有 choose_date, down_periods...
    '''
    #交易日序列和每周最后一个交易日序列
    tradingdates = sorted([str(x).replace('-', '') for x in trade_cal[trade_cal['isOpen']==1]['calendarDate'].values])
    weekend_dates = sorted([str(x).replace('-', '') for x in trade_cal[trade_cal['isOpen']==1][trade_cal['isWeekEnd']==1]['calendarDate'].values])
    monthend_dates = sorted([str(x).replace('-', '') for x in trade_cal[trade_cal['isOpen']==1][trade_cal['isMonthEnd']==1]['calendarDate'].values])

    #生成高低点序列
    close_preadj['tradeDate'] = close_preadj['tradeDate'].apply(lambda x: str(x).replace('-', ''))    
    point_date_df = get_high_low_point(close_preadj, tradingdates)
    
    #如果高低点序列为空，则直接返回空的dataframe
    if len(point_date_df) == 0:
        return pd.DataFrame([], columns=['choose_date', 'down_periods'])
    
    #对每月最后一个交易日序列逐个匹配形态
    all_choosen_sample = []
    for choose_date in monthend_dates:
        if tradingdates.index(choose_date) < (roll_backdays - 1):
            continue
        start_date = tradingdates[tradingdates.index(choose_date) - roll_backdays + 1]
        point_date_df_use = point_date_df[point_date_df['point_date']>=start_date]
        point_date_df_use = point_date_df_use[point_date_df_use['point_date']<=choose_date]
        #指定区间内高低点序列不存在也跳过
        if len(point_date_df_use) ==0:
            continue
        
        #筛选标准1.序列最后一个点为滚动窗口中的低点，且选股日距低点的d1-d2个交易日
        select_flag_11 = point_date_df_use.loc[:, 'flag'].iloc[-1] == -1
        last_point_date = point_date_df_use['point_date'].iloc[-1]
        tradingday_interval = tradingdates.index(choose_date) - tradingdates.index(last_point_date)
        select_flag_12 = (tradingday_interval >= d1) & (tradingday_interval <= d2)
        select_flag_1 = select_flag_11 & select_flag_12 
        #筛选标准2.从最后的低点向前，至少存在三段下跌
        down_periods = (point_date_df_use['flag']==-1).sum()
        select_flag_2 = down_periods >= 3
        #筛选标准3.每一段下跌的低点都低于前一低点
        down_reverts = (point_date_df_use[point_date_df_use['flag']==-1].sort_values(by='point_date')['closePrice'].diff()>0).sum()
        select_flag_3 = down_reverts == 0
        #筛选标准4.连续下跌的跌幅累计超过fall
        closePrice_last = point_date_df_use['closePrice'].iloc[-1]
        select_flag_4 = (closePrice_last / point_date_df_use['closePrice'] - 1).min() < -fall

        if select_flag_1 & select_flag_2 & select_flag_3 & select_flag_4:
            choose_flag = 1
        else:
            choose_flag = 0
        if choose_flag == 1:
            all_choosen_sample.append([choose_date, down_periods])
        all_choosen_sample_df = pd.DataFrame(all_choosen_sample, columns=['choose_date', 'down_periods'])

    return all_choosen_sample_df

##找到所有符合深跌反转形态的样本

with open('{}/close_adj.pkl'.format(deep_fall_dir), 'r') as f:
    close_adj = pickle.load(f)
#从14年开始跑
close_adj = close_adj[close_adj['tradeDate']>='2014-01-01']    
all_tickers = sorted(list(set(close_adj['ticker'])))
#剔除科创板等股票，只选取 00 30 60等
all_tickers = sorted(list(filter(lambda x: x[:2] in ['00', '30', '60'], all_tickers)))
all_dates =sorted(list(set(close_adj['tradeDate'])))
trade_cal = get_trade_cal(all_dates[0], all_dates[-1])

all_examples = []
error_ticker = []
time0 = time.time()
for ticker in all_tickers:
    close_adj_part = close_adj[close_adj['ticker']==ticker]
    close_adj_part.index = range(len(close_adj_part))
    roll_backdays, winsize, d1, d2, fall = 242, 15, 6, 10, 0.5
    weekend_dates = sorted([str(x).replace('-', '') for x in trade_cal[trade_cal['isOpen']==1][trade_cal['isWeekEnd']==1]['calendarDate'].values])
    ticker_choosen = get_example(close_adj_part, roll_backdays, winsize, d1, d2, fall, trade_cal)
    with open('{}/monthend_{}.pkl'.format(deep_fall_dir, ticker), 'wb') as f:
        pickle.dump(ticker_choosen, f)
print('end looping for all tickers')
print ('该部分耗时: %s 秒！'%(time.time() - time0))



#整合所有的样本
time0 = time.time()
all_codes = sorted(list(filter(lambda x: x.startswith('monthend'), os.listdir(deep_fall_dir))))
all_example = []
for code in all_codes:
    with open('{}/{}'.format(deep_fall_dir, code), 'rb') as f:
        t_code_example = pickle.load(f)
    t_code_example['ticker'] = code[9:15]
    all_example.append(t_code_example)
all_example_df = pd.concat(all_example)
all_example_df['flag'] = 1

#对样本数据进行过滤，去除新股、ST和停牌的股票
spec_stocks = quant_util.stock_special_tag(all_example_df['choose_date'].min().replace('-',''), all_example_df['choose_date'].max().replace('-',''), halt=1, st=1, pre_new=1, pre_new_length=90)
spec_stocks['tradeDate'] = pd.to_datetime(spec_stocks['tradeDate'],format='%Y%m%d').dt.strftime('%Y-%m-%d')
all_example_df = all_example_df.merge(spec_stocks.rename(columns={'tradeDate': 'choose_date'}),on=['ticker','choose_date'],how='left')
all_example_df = all_example_df[~all_example_df['special_flag'].isin(['halt','st','new'])].drop(['special_flag'],axis=1)
all_example_pivot = all_example_df.pivot(index='choose_date', columns='ticker', values='flag')    

print (u'深跌反转形态全样本数据格式为:')
print(all_example_df.head().to_html())
print ('该部分耗时: %s 秒！'%(time.time() - time0))

'''
第二部分：统计满足深跌反转形态的个股在未来一段时间内的平均收益和胜率情况
该部分耗时 25分钟
该部分内容为：

全样本在被选后0-200交易日内的平均收益和胜率。

不同年份的样本在被选后0-200交易日内的平均收益和胜率

不同下跌段数下的样本在被选后0-200交易日内的平均收益和胜率。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)
'''
#读取已经存储好的股票价格数据，用于计算未来收益
with open('{}/close_adj.pkl'.format(deep_fall_dir), 'r') as f:
    close_adj = pickle.load(f)
close_adj['tradeDate'] = close_adj['tradeDate'].apply(lambda x: str(x).replace('-', ''))
close_df = close_adj.pivot(index='tradeDate', columns='ticker', values='closePrice')

#统计当深跌反转信号产生后，后续股票的表现
def get_ret_ratio(close_df, signal_df, interval, ax, title, length):
    '''
    输入：
    close_df: DataFrame， 价格数据，包含列名有 tradeDate, ticker, closePrice..
    signal_df: DataFrame, 因子数据，index为时间，columns是股票代码..
    interval:  int        间隔时间
    ax         plt.ax     绘图句柄
    title      str        图标的标题
    '''
    
    ret_stat = []
    for i in range(1, length, interval):
        t_pct = (close_df.pct_change(i).shift(-i) * signal_df).stack()
        t_win_ratio = (t_pct > 0).sum()/float(t_pct.shape[0])
        t_ret = t_pct.mean()
        ret_stat.append([i, t_ret, t_win_ratio])
    ret_stat_df = pd.DataFrame(ret_stat, columns=['day', 'pct_mean', 'win_ratio'])

    ax.plot(list(ret_stat_df.day.values), list(ret_stat_df.pct_mean.values), '-', label = 'ret')
    ax2 = ax.twinx()
    ax2.plot(list(ret_stat_df.day.values), list(ret_stat_df.win_ratio.values), '-r', label = 'win_ratio')
    ax.legend(loc=2)
    ax.grid()
    ax.set_xlabel(u"day")
    ax.set_ylabel(u"ret")
    ax2.set_ylabel(u"win_ratio")
    ax2.legend(loc=1)
    ax.set_title(title, fontproperties=font, fontsize=16)



#总的样本的未来收益情况和胜率
interval = 2
length = 200
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1,1,1)
get_ret_ratio(close_df, all_example_pivot, interval, ax, u'全样本在被选后0-200交易日内的平均净值及胜率', length)


#按照年份分割

interval = 2
#2019后半年所选出的样本可能会出现不存在长度为125天以上的收益，会导致125以上的天数的胜率统计由于样本太少而出现极端值，所以2019年只统计1-125天的平均收益和胜率
length_dict = {2016: 200, 2017: 200, 2018:200, 2019:125}
fig = plt.figure(figsize=(45, 7))
start_year, end_year = 2016, 2020
for i in range(start_year, end_year):
    ax = fig.add_subplot(1,5,i-2014)
    all_example_df['year'] = all_example_df['choose_date'].apply(lambda x: int(x[:4]))
    all_example_df_part = all_example_df[all_example_df['year']==i]
    all_example_df_part_pivot = all_example_df_part.pivot(index='choose_date', columns='ticker', values='flag')    
    get_ret_ratio(close_df, all_example_df_part_pivot, interval, ax, i, length_dict[i])


# 按照段数分割
interval = 2
length = 200
fig = plt.figure(figsize=(30, 8))
for i in range(3, 6):
    ax = fig.add_subplot(1,3,i-2)
    all_example_df_part = all_example_df[all_example_df['down_periods']==i]
    all_example_df_part_pivot = all_example_df_part.pivot(index='choose_date', columns='ticker', values='flag')    
    get_ret_ratio(close_df, all_example_df_part_pivot, interval, ax, i, length)
    

'''
第三部分：根据深跌反转形态的信号构建策略
该部分耗时 15分钟
该部分内容为：

全样本在被选后持仓20个交易日构建策略；

根据下跌段数分类，分别构建策略，对比不同的样本的策略表现;

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''
# 由于优矿返回数据条数的限制，分段获取个股月度收益率，
t0 = time.time()
bt_mret_df1 = DataAPI.MktEqumAdjGet(beginDate="2010-01-01",endDate="2013-12-31",secID="",field=u"secID,endDate,chgPct",pandas="1")
bt_mret_df2 = DataAPI.MktEqumAdjGet(beginDate='2014-01-01',endDate="2016-12-31",secID="",field=u"secID,endDate,chgPct",pandas="1")
bt_mret_df3 = DataAPI.MktEqumAdjGet(beginDate='2017-01-01',endDate='2020-03-23',secID="",field=u"secID,endDate,chgPct",pandas="1")
bt_mret_df = bt_mret_df1.append(bt_mret_df2).append(bt_mret_df3)
bt_mret_df.rename(columns={'endDate':'tradeDate', 'chgPct':'curr_ret'}, inplace=True)
bt_mret_df['ticker'] = bt_mret_df['secID'].str.slice(0,6)
bt_mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
bt_mret_df['nxt_ret'] = bt_mret_df.groupby('ticker')['curr_ret'].shift(-1)
bt_mret_df = bt_mret_df.dropna(subset=['nxt_ret'])
print(bt_mret_df.head().to_html())
print ('该部分耗时: %s 秒！'%(time.time() - t0))

#简易版的回测纯多头策略函数
def backtest_longonly_strategy(signal_df, ret_df):
    '''
    输入：
    signal_df: DataFrame， 信号数据，包含列名有 tradeDate, ticker..
    ret_df:    DataFrame,  收益率数据 包含列名  secID, ticker, tradeDate, nxt_ret
    返回：
    bt，       Series，    收益序列
    stat_se    Series,     回测结果的统计指标
    '''

    first_day = signal_df.tradeDate.min()
    last_day = signal_df.tradeDate.max()
    ret_df['tradeDate'] = ret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
    monthend_dates = sorted(list(ret_df['tradeDate']))
    bt_monthend_dates = sorted(list(set(filter(lambda x: x>=first_day and x<=last_day, monthend_dates))))
    merge_df = signal_df.merge(ret_df, on=['ticker', 'tradeDate'], how='left')
    bt = merge_df.groupby('tradeDate')['nxt_ret'].mean()
    bt = bt.loc[bt_monthend_dates]
    bt = bt.fillna(0)
    sharpe = (12)**(0.5) * bt.mean()/bt.std()
    annuel_ret = bt.mean() * 12
    all_ret = bt.sum()
    win_ratio = (bt>0).sum()/float(bt[bt!=0].shape[0])
    bt_net_value = bt + 1
    bt_net_value_rolling_max = bt_net_value.rolling(window=1000, min_periods=2).max()
    mdd = (bt_net_value/bt_net_value_rolling_max -1).min()
    bt_days = (datetime.datetime.strptime(last_day, '%Y%m%d') - datetime.datetime.strptime(first_day, '%Y%m%d')).days
    bt_years = round(bt_days/float(365), 2)
    stat_se = pd.Series(['{}%'.format(round(100*all_ret, 2)), round(bt_years, 2), '{}%'.format(round(100*annuel_ret, 2)),
                         round(sharpe,2 ), '{}%'.format(round(100*win_ratio, 2)), '{}%'.format(round(100*mdd, 2))], 
                         index=['总收益', '回测区间（年）', '年化收益', '夏普比率', '胜率', '最大回撤'])
    
    return bt, stat_se

def plot_bt(bt, stat_se, title_part):
    bt.cumsum().plot()
    plt.title(u'【{}】 年化收益: {}; 夏普比率: {}; 最大回撤: {}'.format(title_part, stat_se.loc['年化收益'], stat_se.loc['夏普比率'],stat_se.loc['最大回撤']),  fontproperties=font, fontsize=16)
    plt.show()
    
'''

3.1 全样本在被选后持仓20个交易日构建策略

在每个月末，在全市场搜寻满足深跌反转形态的个股，剔除掉上市不久、停牌等股票（获取信号时已经剔除），买入并持有至下个月末，获得策略的表现曲线。

'''

signal_df = all_example_df.rename(columns={'choose_date': 'tradeDate'})
bt, stat_se = backtest_longonly_strategy(signal_df, bt_mret_df.copy())
print(pd.DataFrame(stat_se).T.to_html())
plot_bt(bt, stat_se, u'全样本')


#分下跌段数可以都画图
for down_period in range(3, 6):
    signal_df = all_example_df.rename(columns={'choose_date': 'tradeDate'})
    signal_df = signal_df[signal_df['down_periods']==down_period]
    bt, stat_se = backtest_longonly_strategy(signal_df, bt_mret_df.copy())
    plt.figure()
    stat_se.name = u'down_period为{}的样本'.format(down_period)
    print('######################################\n')   
    print(pd.DataFrame(stat_se).T.to_html())
    plot_bt(bt, stat_se, u'down_period为{}的样本'.format(down_period))
   


'''

第四部分：深跌反转因子测试
该部分耗时 15分钟
该部分内容为：

用不同的decay长度分别生成两个深跌反转因子；

对深跌反转因子数据进行行业和市值中性化、去除停牌、涨停等股票；

对两个深跌反转因子数据进行对比测试，包括计算IC、分组、多空收益等；

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

## 4.1 用不同的decay长度分别生成两个深跌反转因子
- 深跌反转本身是一个事件型策略，为了将该信号因子化，我们将信号发生当天的因子值设置为1，然后每天不断地衰减，设置一长(100天)一短(50天)两个半衰期，分别考察该因子在短期和长期内的有效性。

'''

##生成因子数据
#数据处理部分
t0 = time.time()
all_example_df['flag'] = 1
trade_df = all_example_df.pivot(index='choose_date', columns='ticker', values='flag')
trade_cal = get_trade_cal(all_example_df.choose_date.min(), '20200309')
tradingdates = sorted([str(x).replace('-', '') for x in trade_cal[trade_cal['isOpen']==1]['calendarDate'].values])
trade_df = trade_df.loc[tradingdates, :]
monthend_dates = sorted([str(x).replace('-', '') for x in trade_cal[trade_cal['isOpen']==1][trade_cal['isMonthEnd']==1]['calendarDate'].values])
weekend_dates = sorted([str(x).replace('-', '') for x in trade_cal[trade_cal['isOpen']==1][trade_cal['isWeekEnd']==1]['calendarDate'].values])
trade_df = trade_df.fillna(0)

#生成做因子用的signal_df,分别使用不同的半衰期和rolling_window;因子dataframe只需要月末的因子
def get_decay_df(trade_df, signal_rolling_window, signal_half_decay_days, monthend_dates, tradingdates):
    
    '''
    输入：
    trade_df:                 DataFrame， 信号数据，index为交易日序列，columns为股票代码序列
    singal_rolling_window:    int,        信号往后衰减的最长长度
    signal_half_decay_days:   int,        信号衰减的半衰期
    monthend_dates:           list,       每个月月末的交易日序列
    tradedates：              list,       所有的交易日序列
    返回：
    decay_df_stack            DataFrame,  生成的因子数据， 列名包括 ticker, tradeDate..
    '''
    
    decay_ratio = (float(1)/2)**(float(1)/signal_half_decay_days)
    decay_df = []
    for date in monthend_dates:
        if tradingdates.index(date) < signal_rolling_window : continue
        monthend_1year_before = tradingdates[tradingdates.index(date) - signal_rolling_window]
        trade_df_part = trade_df.loc[monthend_1year_before:date]
        weight_se = pd.Series(decay_ratio ** np.arange(signal_rolling_window+1)[::-1], index=trade_df_part.index)
        decay_se = (trade_df_part.T * weight_se).sum(axis=1)
        decay_se.name = date
        decay_df.append(decay_se)
    decay_df = pd.concat(decay_df, axis=1).T
    decay_df_stack = decay_df.stack().reset_index()
    decay_df_stack.columns = ['tradeDate', 'ticker', 'decay_{}'.format(signal_half_decay_days)]
    decay_df_stack = decay_df_stack[decay_df_stack['decay_{}'.format(signal_half_decay_days)]!=0]
    return decay_df_stack

signal_rolling_window = 100
decay_period_1, decay_period_2 = 100, 50
signal_decay_1000 = get_decay_df(trade_df, signal_rolling_window, decay_period_1, monthend_dates, tradingdates)
signal_decay_500 = get_decay_df(trade_df, signal_rolling_window, decay_period_2, monthend_dates, tradingdates)
    
signal_decay = signal_decay_1000.merge(signal_decay_500, on=['ticker', 'tradeDate'], how='inner')
signal_decay['tradeDate'] = signal_decay['tradeDate'].apply(lambda x: '{}-{}-{}'.format(str(x)[:4], str(x)[4:6], str(x)[6:]))
deepfall_reverse = signal_decay.copy()
deepfall_reverse['secID'] = deepfall_reverse['ticker'].apply(lambda x: '{}.XSHE'.format(x) if x[:2] != '60' else '{}.XSHG'.format(x)) 
print('反转因子的数据样式为: ')
print(deepfall_reverse.head().to_html())
print ('该部分耗时: %s 秒！'%(time.time() - t0))

'''


4.2 对深跌反转因子数据进行行业和市值中性化、去除停牌、涨停等股票

'''

# # 对市值和行业进行中性化
t0 = time.time()
factor1, factor2 = 'decay_{}'.format(decay_period_1), 'decay_{}'.format(decay_period_2)
deepfall_reverse1 = deepfall_reverse.dropna(subset=[factor1, factor2])
deepfall_reverse1 = quant_util.neutralize_dframe(deepfall_reverse1.copy(), [factor1, factor2], exclude_style=['BETA','RESVOL','MOMENTUM','SIZENL','EARNYILD','BTOP','GROWTH','LEVERAGE','LIQUIDTY'])

# 测试之前需对因子数据进行过滤，去除新股、ST和停牌的股票
spec_stocks = quant_util.stock_special_tag(deepfall_reverse1['tradeDate'].min().replace('-',''), deepfall_reverse1['tradeDate'].max().replace('-',''), halt=1, st=1, pre_new=1, pre_new_length=90)
spec_stocks['tradeDate'] = pd.to_datetime(spec_stocks['tradeDate'],format='%Y%m%d').dt.strftime('%Y-%m-%d')
deepfall_reverse1 = deepfall_reverse1.copy().merge(spec_stocks,on=['ticker','tradeDate'],how='left')
deepfall_reverse1 = deepfall_reverse1[~deepfall_reverse1['special_flag'].isin(['halt','st','new'])].drop(['special_flag'],axis=1)
print('做完市值行业中性、去掉停牌和ST后的深跌反转因子样式为： ')
print(deepfall_reverse1.head().to_html())
print ('该部分耗时: %s 秒！'%(time.time() - t0))


'''

4.3 对两个深跌反转因子数据进行对比测试，包括计算IC、分组、多空收益等

'''

#计算IC
def ic_anlyst(factor_df, rtn_df, factor_col, nxt_rtn_col,cor_method='spearman'):
    factor_rtn_df = factor_df.merge(rtn_df, on=['secID', 'tradeDate'])
    period_ic = factor_rtn_df.groupby('tradeDate').apply(lambda x: x[[factor_col,nxt_rtn_col]].corr(method=cor_method).values[0, 1])
    ic = period_ic.mean()
    std = period_ic.std()
    icir = ic / std
    ic_t = stats.ttest_1samp(period_ic, 0)[0]
    ic_summary = pd.Series([ic, std, icir, ic_t], index = [u'IC均值', u'IC波动率',u'ICIR', u't值']) 
    return ic_summary

#计算超额收益
def excess_rtn(s):
    r = s.iloc[-1]/s.iloc[0] - 1 
    return r

#计算胜率
def winper(s):
    s = s[s!=0]
    return (s>0).sum() / float(len(s))

#计算最大回测
def maxDrawdown(s):
    cum_max = s.cummax()
    maxdrawdown =((cum_max-s)/cum_max).max()
    return maxdrawdown

#计算年化收益
def annual_rtn(s, l,step=250):
    r = s.iloc[-1]/s.iloc[0] - 1
    ar = r / l * step
    return ar

#计算信息比率
def cal_ir(s,step=250):
    m = s.mean()
    m1 = m*step
    std1 = s.std()* np.sqrt(step)
    ir = m1/std1
    return ir

#分组绘图
def plot_group_fig(perf,group_num,title="分组净值"):
    fig = plt.figure(figsize=(20,8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    nav = []
    label_dict = {}
    for i in range(group_num):
        label_dict[i+1] = u'第%s组'%(i+1)
        if i == 0:
            label_dict[i+1] += '(low)'
        elif i == group_num-1:
            label_dict[i+1] += '(high)'
        gperf = perf[perf['group'] == i]
        nav = nav + [gperf['cum_ret'].values[-1]]
        _=ax1.plot(pd.to_datetime(gperf['tradeDate']), gperf[['cum_ret']], label=label_dict[i+1])
    ax1.set_ylabel(u"净值",fontproperties=font, fontsize=16)
    ax1.set_title(title, fontproperties=font, fontsize=16)
    ax1.legend(loc=0, prop=font)
    ind = np.arange(group_num)
    ax2.bar(ind+1.0/group_num, nav, 0.3, color='r')
    ax2.set_xlim((0, ind[-1]+1))
    ax2.set_xticks(ind+0.35)
    ax2.set_title(title, fontproperties=font, fontsize=16)
    _=ax2.set_xticklabels([label_dict[i+1] for i in ind], fontproperties=font)
    return 

#多空净值曲线
def plot_ls_fig(perf1,perf2, title=u"net value"):    
    f= plt.figure(figsize=(20,5))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    _ = perf1.plot(ax=ax1)
    _ = ax1.set_title(perf1.name)
    _ = perf2.plot(ax=ax2)
    _ = ax2.set_title(perf2.name)
    return 

#多空表现指标统计
def ls_perf_stats(perf,step=12):
    r=[]
    excess_rtn1 = excess_rtn(perf['cum_ret'].dropna())
    winper1 = winper(perf['period_ret'].dropna())
    maxDrawdown1 = maxDrawdown(perf['cum_ret'].dropna())
    ir1 = cal_ir(perf['period_ret'],step)
    ar1 = annual_rtn(perf['cum_ret'].dropna(),len(perf['cum_ret'].dropna()),step)
    gb_p = pd.Series([excess_rtn1,ar1,winper1,maxDrawdown1,ir1],index=['excess_rtn','annual_rtn','winper','maxDrawdown','ir']) 
    return gb_p

##因子的IC、分组、多空测试

factor1, factor2 = 'decay_{}'.format(decay_period_1), 'decay_{}'.format(decay_period_2)
# IC 测试
factor1_ic_summary = ic_anlyst(deepfall_reverse1.copy(), bt_mret_df.copy(), factor1, "nxt_ret",cor_method='spearman')
factor2_ic_summary = ic_anlyst(deepfall_reverse1.copy(), bt_mret_df.copy(), factor2, "nxt_ret",cor_method='spearman')

ic_a_summary  = pd.DataFrame([factor1_ic_summary,factor2_ic_summary],index=['{}_signal'.format(factor1), '{}_signal'.format(factor2)]).applymap(lambda x: round(x,3)) 
print (ic_a_summary.to_html())

#分组测试
group_num = 5
hp_a_group_perf = quant_util.simple_group_backtest(deepfall_reverse1.copy(), bt_mret_df.copy(), factor1, 'nxt_ret', ngrp=group_num)
_ =plot_group_fig(hp_a_group_perf[0],group_num,title=u'{}因子分组净值'.format(factor1))
ori_a_group_perf = quant_util.simple_group_backtest(deepfall_reverse1.copy(), bt_mret_df.copy(), factor2, 'nxt_ret', ngrp=group_num)
_ =plot_group_fig(ori_a_group_perf[0],group_num,title=u'{}因子分组净值'.format(factor2))

#多空,手续费为双边0.2%
hp_a_ls_perf,hp_a_bt_df = quant_util.easy_backtest(deepfall_reverse1.copy(), bt_mret_df.copy(), factor1, 'nxt_ret', method='long_short', direction=1, ngrp=5, weight_schemes=0, weights=None, commission=0.002)
ori_a_ls_perf,ori_a_bt_df = quant_util.easy_backtest(deepfall_reverse1.copy(), bt_mret_df.copy(), factor2, 'nxt_ret', method='long_short', direction=1, ngrp=5, weight_schemes=0, weights=None, commission=0.002)
hp_a_ls_summary = ls_perf_stats(hp_a_ls_perf)
ori_a_ls_summary = ls_perf_stats(ori_a_ls_perf)
ls_a_summary = pd.DataFrame([hp_a_ls_summary,ori_a_ls_summary],index=['{}_signal'.format(factor1),'{}_signal'.format(factor2)]).applymap(lambda x: round(x,3)) 
print (ls_a_summary.to_html())
hp_a_ls_perf = hp_a_ls_perf.set_index('tradeDate')['cum_ret']
ori_a_ls_perf = ori_a_ls_perf.set_index('tradeDate')['cum_ret']
hp_a_ls_perf.name = "net_value of {} signal".format(factor1)
ori_a_ls_perf.name = "net_value of {} signal".format(factor2)
_ = plot_ls_fig(hp_a_ls_perf,ori_a_ls_perf)


'''

通过回测发现，深跌反转因子的表现并不理想，IC均值不到1%，五分组也没有体现出较好的线性性。虽然多空收益有年化8%的收益，但是最大回撤有23.5%，并不是一个理想的alpha因子。
究其原因，一方面根据前面回测结果，因子胜率在20个交易日后迅速下滑，反弹动能衰减，因子的有效性在衰减期内未能维持。另一方面可能是深跌反转形态确实是一个好的择时方法，但是在横截面上相对于不具有深跌反转形态的个股并不具有明显的超额收益。

第五部分：总结
   调试 运行
文档
 代码  策略  文档
深跌反转形态的个股在触发信号后20个交易日内达到了接近百分之90%的峰值，显示深跌反转信号在刚触发时，该股票有很强的反弹动能。在20个交易日之后，符合深跌反转信号的个股走势出现分化，导致胜率出现降低。在20日后仍然延续强劲反弹势头的个股收益很高，带动了整体的平均收益后续继续走高。
策略在20140101到20191231的回测区间内，共取得128.52%的总收益，年化收益为30.24%，夏普比率为1.05，最大回撤32.99%，胜率较高为73.08%。较高的胜率表明深跌反转形态的后续反弹动能很强。深跌反转形态在各个下跌段数上表现较为稳定，年化收益都在20%以上，胜率皆在64%以上。同时下跌段数为3段的样本最大回撤23.82%最小，同时也小于全样本的32.99%,表明过多的下跌段数会透露出该个股的长期下跌的隐忧，反转的动能趋缓，难以取得更高的反弹收益。
通过回测发现，深跌反转因子的表现并不理想，IC均值不到1%，五分组没有体现出较好的线性性，并不是一个理想的alpha因子。一方面可能是因子反弹动能衰减，因子的有效性在衰减期内未能维持；另一方面可能是深跌反转形态确实是一个好的择时方法，但是在横截面上相对于不具有深跌反转形态的个股并不具有明显的超额收益。

'''
