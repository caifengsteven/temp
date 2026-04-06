# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:25:45 2020

@author: Asus
"""

'''
导读
A. 研究目的：

传统的多因子模型对全市场股票不加区别的统一建模，忽略了股票自身属性的不同，以及Alpha因子在不同风格股票域内的差异性。
动态情景模型((Dynamic Contextual Alpha Model, DCAM)通过将股票进行分区间建模很好地解决了上述问题，同时该模型通过切分区间自然地引入了非线性特征，使得其更贴近市场真实逻辑。
本文利用优矿的因子数据与回测框架，参考东方证券研报1《动态情景多因子Alpha模型》、东方证券研报2《动态情景多因子Alpha模型再思考》及国泰君安研报《风格域划分下的基本面多因子选股策略》中的研究方法，对研报的结果进行了实证分析，用以探索DCAM模型与传统模型的区别
B. 研究结论：

基于规模、价值及成长3个情景因子对股票进行分区间研究发现，alpha因子在不同区间的股票收益预测能力存在显著差异，也说明了分区间研究的必要性

DCAM模型采取不同的加权方案，表现出的效果也不同，东方证券研报2《动态情景多因子Alpha模型再思考》中的加权方案表现最优

通过IC、多空组合，Top100组合及中证500指数增强组合四个方面效果的分析，发现DCAM相较于传统模型均有提升，以中证500指数增强为例，东方研报2的模型年化收益相较传统模型的14.6%提升到17.7%，信息比率从1.83提升到2.17

C. 文章结构: 本文共分为4个部分，具体如下

一、数据准备，利用API调取因子及收益数据，并做相关处理

二、情景分层下的因子测试，该部分主要测试情景因子分层后，Alpha因子在不同分层区间的表现差别

三、动态情景模型构建，该部分主要是对三篇研报中的加权方案进行分析

四、实证分析，该部分主要是从四个方面(IC、多空组合，Top100组合及中证500指数增强组合)对等权打分模型、ICIR打分模型和DCAM模型进行对比

D. 运行时间说明

一、数据准备，需要5分钟左右

二、情景分层下的因子测试，需要1分钟左右

三、动态情景模型构建，需要20分钟左右

四、实证分析，测试组合较多共20组，需要1小时左右

总耗时90分钟左右

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

一、数据准备
该部分耗时 5分钟
该部分内容为：

从优矿获取因子数据，下月收益数据

对数据进行去极值、填缺失值、标准化及中性化操作

对alpha因子进行正交化处理

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
1.1 获取uqer因子、下月收益数据

获取因子数据包括: 情景因子、及alpha因子库

情景因子
本文选取了三个情景因子，包括了规模、价值及成长的因素(见下表):
图片注释

本文从反转、动量、流动性、估值、盈利、成长、财务杠杆及现金流8类风格类别中共选取了18个基础alpha因子 图片注释



'''

import pandas as pd
import numpy as np
import os
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from CAL.PyCAL import * 
import gevent
import datetime as dt
from dateutil.relativedelta import relativedelta

raw_data_dir = "./raw_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

#定义需要获取的因子
contextual_factors = ['LCAP', 'PB', 'NetAssetGrowRate']
alpha_factors = ['REVS20', 'RSTR504', 'VOL20', 'DAVOL20', 'ILLIQUIDITY', 'PE', 'PS', 'PCF', 'CETOP', 'ROE', 'EPSTTM', 'ROIC', 'NetProfitGrowRate', 'OperatingRevenueGrowRate', 'MLEV', 'DebtsAssetRatio', 'EnterpriseFCFPS', 'OperCashFlowPS']

# 起始时间
start_date = '2009-12-01'
end_date = '2018-07-01'

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
    list_date_need = (str2date(date_str) + relativedelta(days=-list_date)).strftime("%Y-%m-%d")
    A_ticker = set(equ_df[(equ_df['listDate'] <= list_date_need) & ((equ_df['delistDate'] > date_str) | (equ_df['delistDate'].isnull()))]['ticker'])
    st_ticker = set(DataAPI.SecSTGet(beginDate=list_date_need, endDate=date_str, pandas="1")['ticker'])
    
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
            x = DataAPI.MktStockFactorsOneDayProGet(tradeDate=tdate, field=['ticker', 'tradeDate'] + contextual_factors + alpha_factors, pandas="1")
            x = x.dropna(subset=contextual_factors)           
            x['tradeDate'] = x['tradeDate'].apply(lambda x: x.replace("-", ""))
            x = x[x['ticker'].isin(A_ticker)]
            
            # 查询行业信息
            industry = DataAPI.EquIndustryGet(industryVersionCD=u"010303", intoDate=tdate,field='ticker,industryName1',pandas="1").dropna()
            x = pd.merge(x, industry, on='ticker')

            x = x.dropna(thresh=15)
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
    chgframe = DataAPI.MktEqumAdjGet(beginDate=start_date, endDate=end_date, field=['ticker', 'endDate', 'chgPct', 'return'], pandas="1")
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
    
    print('因子数据格式为')
    print(factor_frame.head(5).to_html())
    end_time = time.time()
    print "Time cost: %s seconds" % (end_time - start_time)
    
'''
1.2 对数据进行去极值、填缺失值、标准化及中性化操作

本章节对上一小节的数据进行相关处理

用MAD法处理5倍标准差外的异常值
用行业中位数填充空值
对因子及收益实现市值及行业的中性化

'''

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

    upper = dm + 5 * dm1
    lower = dm - 5 * dm1
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

    
    for a_factor in alpha_factors + ['chgPct']:
        sig = cdate_input[a_factor]
        cnt = 0
        while True:
            try:
                cdate_input.loc[:, a_factor] = neutralize(cdate_input[a_factor], target_date=tdate, exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'SIZENL'])       
                break
            except Exception as e:
                cnt += 1
                if cnt >= 3:
                    break
                    
    cdate_input.loc[:, alpha_factors] = cdate_input.loc[:, alpha_factors].apply(lambda x : standardize(x))
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
    
    factor_csv = pd.concat(new_frame_list, axis=0)
    factor_csv.reset_index(inplace=True)
    factor_csv = factor_csv.rename(columns={'return':'next_return', 'chgPct':'neu_next_return'})

    ################################ 数据存储下来 ################################
    factor_csv.to_csv(os.path.join(raw_data_dir, 'dcam_data.csv'), chunksize=1000)
    end_time = time.time()
    print "Time cost: %s seconds" % (end_time - start_time)
    
'''

1.3 正交化处理

为了消除因子之间共线性的影响，本文引入对称正交法对Alpha因子库中因子进行处理
对称正交相关原理见天风研报《因子正交全攻略，理论、框架与实践》或者8月的深度报告《风格的定义和择时价值判断》

'''


from numpy import linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt
from CAL.PyCAL import *    # CAL.PyCAL中包含font


def lowdin_orthog_list(x_list):
    '''
        对输入的list进行lowdin正交
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
    
    df = df.groupby(['tradeDate']).apply(orthog_tdate_frame, cols)
    df.index = range(len(df))
    return df

def plot_corr_heatmap(dframe, col_list, ax, title=None):
    '''
    因子间相关性图
    '''
    corr_frame = dframe[col_list].corr()
    corr_frame = (100*corr_frame).round(2)
    _ = sns.heatmap(corr_frame, alpha=1.0, annot=True, center=0.0, annot_kws={"size": 8}, linewidths=0.02, 
                         linecolor='white', linewidth=0, ax=ax)
    if title is not None:
        ax.set_title(title, fontproperties=font, fontsize=16)
        
start_time = time.time()

# 读入原始因子文件
factor_frame = pd.read_csv(os.path.join(raw_data_dir, 'dcam_data.csv'), index_col=0, dtype={'ticker':np.str, 'tradeDate':np.str})

# 对因子进行对称正交
all_orth_factor_df = lowdin_orthog_frame(factor_frame,  alpha_factors)
all_orth_factor_df.to_csv(os.path.join(raw_data_dir, "orth_dcam_data.csv"))

# 对比正交前后的因子相关性
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))      
plot_corr_heatmap(factor_frame, alpha_factors, axes[0], u'无对称正交的因子相关性矩阵')
plot_corr_heatmap(all_orth_factor_df, alpha_factors, axes[1], u'对称正交之后的因子相关性矩阵')

end_time = time.time()
print "Time cost: %s seconds" % (end_time - start_time)


'''

二、情景分层下的因子测试
该部分耗时 1分钟左右
本章节主要测试情景因子分层后，Alpha因子在不同分层区间的表现差别
具体的说，每个月末，按照选取的分层因子对全样本等分地切成两份，然后计算alpha因子在不同样本空间的IC，最后检验不同区间IC是否有明显区别
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
本文选取风险调整IC来代替传统IC计算，其定义如下：
ICadj=corr(fpure,rresidual)
其中， fpure是本期因子值中性化后的值，rresidual是下期个股收益中性化后的值，X代表着行业因素，mktcap代表着市值因素，回归的方程表示为：
fpure=f−b1X−b2log(mktcap)
rresidual=r−m1X−m2log(mktcap)
上述中性化在第一章节中已做处理

'''


import scipy.stats as st
from itertools import chain

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
    signal_df_tmp['group_%s'%factor_name] = signal_df_tmp.groupby('tradeDate')[factor_name].apply(lambda x: (x.rank() - 1) / len(x) * ngrp).astype(int)
    return signal_df_tmp

def analysis_ic(data, factor_name):
    '''
    分析不同区间的IC结果
    params:
        data: DataFrame, 计算的IC结果
        factor_name: 情景因子名称
    return:
        Series, IC检验结果
    '''
    mean = data.mean()
    std = data.std()
    ir = mean / std
    
    t_low = st.ttest_1samp(data[0].values, 0)[0]
    t_high = st.ttest_1samp(data[1].values, 0)[0]
    t_high_low = st.ttest_1samp((data[1]-data[0]).values, 0)[0]
    
    t, p_t = st.ttest_ind(data[0].values, data[1].values, equal_var=False)   #t检验
    F, p_F = st.levene(data[0].values, data[1].values)                       #关于方差的F检验
    ks, p_ks = st.ks_2samp(data[0].values, data[1].values)                   #关于分布的K-S检验
    
    values = list(chain(*zip(mean, std)))
    names = [['Mean', 'Mean', 'Tstat', 'Tstat', 'IR', 'IR', 'Two Sample Test', 'Two Sample Test', 'F Test','F Test', 'DiffTstat'], ['Low', 'High', 'Low', 'High', 'Low', 'High', 't','p value', 'F', 'p_value', 'High-Low']]
    index = pd.MultiIndex.from_tuples(zip(*names))   #生成多维列名
    
    return pd.Series(index = index, data = mean.tolist()+[t_low, t_high]+ir.tolist()+[t, p_t, F, p_F, t_high_low], name=factor_name)


import pandas as pd
import numpy as np
raw_data_dir = "./raw_data"
dcam_alpha  = pd.read_csv(os.path.join(raw_data_dir, 'orth_dcam_data.csv'), index_col=0, dtype={'ticker':np.str, 'tradeDate':np.str})

for factor_name in contextual_factors:
    factor_group = signal_grouping(dcam_alpha, factor_name, 2)
    ic = factor_group.groupby(['tradeDate', 'group_%s'%factor_name]).apply(lambda x:  pd.Series(index=alpha_factors,
                                                data=x[['neu_next_return']+alpha_factors].corr(method='spearman').values[0, 1:])).reset_index()
    data = pd.pivot_table(ic, values=alpha_factors, index=['tradeDate'], columns=['group_%s'%factor_name])
    res = pd.concat([analysis_ic(data[factor],  factor) for factor in alpha_factors], axis=1).T
    print('******************* %s 因子分层的IC检验 *******************\n'%factor_name)
    print(res.round(4).to_html())
    
'''

上述结果中，第一列代表不同区间的IC序列均值，第二列代表不同区间的IC序列T检验值，第三列代表不同区间IC序列的IR值，第四列代表两区间IC序列的t双边检验(主要检查均值是否一致)，第五列代表着IC序列的F检验(主要检查方差是否一致)，最后一列代表着两区间IC序列差值的T检验值。

可以看出，在情景因子分层下，有些alpha因子的IC显著不同。例如，仅从最后一列检验来说，市值因子分层下，REVS20、DAVOL20、ILLIQUIDITY、PS、OperCashFlowPS均有显著差别。从侧面也验证了分层建模的必要性。

   调试 运行
文档
 代码  策略  文档
三、动态情景模型构建
该部分耗时 20分钟
该部分流程可以参考东方证券《动态情景多因子Alpha模型再思考》中的流程
图片注释

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
具体的说，构建一个动态情景模型，需要考虑如下4个部分:

情景因子与Alpha因子的选取
本节沿用上述章节定义好的因子。每个情景因子对样本空间等分成5份
同一情景下的各情景区间内部如何分配因子的权重
这里分配权重类似常规alpha模型，比如常见的等权、IC、ICIR、最大化复合ICIR、最大化复合IC等
为了对比方便，这里统一采用过去24个月的ICIR作为该子区间的因子权重比例
同一情景下的不同情景区间结果如何进行合并
这里东方证券与国泰君安采取了不同的处理方法：

a) 东方证券首先利用情景因子进行区间划分，随后对分配到该区间的股票按照步骤2得到的该子区间ICIR权重进行加权打分，随后合并每个子区间的所属股票打分，即得到全部股票的分数

b) 国泰君安同样先利用情景因子划分区间，然后根据每个股票自身属性的不同，赋予该股票下的各子区间权重，接着对各子区间ICIR权重进行加权打分，最后利用合并权重对该股票进行打分，对每个股票都进行如上操作即得到全部股票分数

小结: 二者主要区别在于东方是先利用权重处理该子区间的股票，再合并各子区间打分结果；国君是先合并各子区间因子权重，再利用合并权重得到所有股票打分

不同情景下产生的结果如何进行合并
国君直接采用等权处理不同情景下的股票得分
东方引入了"个股情景特征"(利用该股票在该风格因子分位数衡量)及"模型区分度"(利用该情景下各子区间因子权重的距离来衡量)来进行不同情景间的加权

'''

import scipy.stats as st
from scipy.spatial import distance
import statsmodels.api as sm

contextual_factors = ['LCAP', 'PB', 'NetAssetGrowRate']
alpha_factors = ['REVS20', 'RSTR504', 'VOL20', 'DAVOL20', 'ILLIQUIDITY', 'PE', 'PS', 'PCF', 'CETOP', 'ROE', 'EPSTTM', 'ROIC', 'NetProfitGrowRate', 'OperatingRevenueGrowRate', 'MLEV', 'DebtsAssetRatio', 'EnterpriseFCFPS', 'OperCashFlowPS']

def combine_factor(signal_df, weight):
    """
    利用权重与因子暴露，进行每期的因子合成
    params:
        signal_df: DataFrame, 股票的因子值
        weight: DataFrame, 每期的因子权重
    return:
        DataFrame, 返回计算的合成因子
    """    
    factor_list = weight.columns.tolist()
    merge_factor = pd.merge(signal_df[['ticker', 'tradeDate']+factor_list], weight.reset_index(), on='tradeDate').sort_values(['tradeDate', 'ticker'])
   
    merge_factor['alpha'] = merge_factor.apply(lambda x : np.sum(np.array([x[item+"_x"]*x[item+"_y"] for item in factor_list])), axis=1)
    
    return merge_factor[['ticker', 'tradeDate', 'alpha']]


class DCAM:
    """
    构建动态情景模型，主要是实现了三篇研报中方法
    """
    def __init__(self, data, contextual_factors, alpha_factors, groups=5):
        """
        初始化函数
        params:
            data: DataFrame, alpha信号
            contextual_factors: list, 情景因子
            alpha_factors: list, alpha因子
            groups: int, 利用情景因子的分层数，默认为5
        """
        self.data = data.copy()
        self.contextual_factors = contextual_factors
        self.alpha_factors = alpha_factors
        self.groups = groups


    def get_dcam_alpha(self, method):
        """
        利用三篇研报的加权方案进行打分
        params:
            method: str, 具体哪一种方法，取值{guojun, dongfang_v1, dongfang_v2}，代表着国君研报、东方研报1，东方研报2中提到的方法
        return:
            DataFrame, 返回计算的合成信号
        """    
        data = self.data
        data[['%s_q'%item for item in self.contextual_factors]] = data.groupby('tradeDate')[self.contextual_factors].apply(lambda x: x.rank(pct=True)).rename(columns={item:'%s_q'%item for item in self.contextual_factors})

        contextual_alpha_weight_list = []
        for factor_name in self.contextual_factors:
            alpha_data = signal_grouping(data, factor_name, self.groups)
            factor_ic = alpha_data.groupby(['tradeDate', 'group_%s' % factor_name]).apply(
                lambda x: pd.Series(index=self.alpha_factors, data=x[['neu_next_return'] + self.alpha_factors].corr(method='spearman').values[0, 1:])).reset_index()
            factor_ic = factor_ic.sort_values(by=['tradeDate', 'group_%s' % factor_name], ascending=[True, True])
            factor_ic.loc[:, self.alpha_factors] = factor_ic.groupby('group_%s' % factor_name)[self.alpha_factors].shift(1)
            contextual_alpha_weight = self.get_contextual_alpha(alpha_data, factor_ic.dropna(), factor_name, method=method)

            contextual_alpha_weight_list.append(contextual_alpha_weight)

        all_alpha = reduce(lambda left, right: pd.merge(left, right, on=['tradeDate', 'ticker']), contextual_alpha_weight_list)
        alpha_columns = ['alpha_'+item for item in self.contextual_factors]
        weight_columns = ['weight_'+item for item in self.contextual_factors]
        all_alpha['alpha'] = (np.array(all_alpha[alpha_columns]) * np.array(all_alpha[weight_columns])).sum(axis=1) / np.array(all_alpha[weight_columns]).sum(axis=1)

        return all_alpha[['tradeDate', 'ticker', 'alpha']]


    def get_contextual_alpha(self, alpha_data, factor_ic, factor_name, window=24, method='dongfang_v2'):
        """
        对某一个具体情景因子下的加权打分模块
        params:
            alpha_data: DataFrame, alpha信号
            factor_ic: DataFrame, alpha信号的RankIC
            factor_name: str, 情景因子名称
            window: int, 窗口期, 默认值24
            method: str, 方法名称，具体涵义见上一个函数
        return:
            DataFrame, 返回计算的合成信号，及权重
        """    
        alpha_data = alpha_data.copy()
        #step 1  各子区间采用ICIR权重
        sub_interval_ic = factor_ic.set_index('tradeDate').groupby('group_%s' % factor_name)[self.alpha_factors].rolling(window).mean()
        sub_interval_ir = factor_ic.set_index('tradeDate').groupby('group_%s' % factor_name)[self.alpha_factors].rolling(window).std()
        sub_interval_ic_ir = (sub_interval_ic/sub_interval_ir).dropna()
        sub_interval_weight = sub_interval_ic_ir.divide(sub_interval_ic_ir.abs().sum(axis=1), axis=0)
        sub_interval_weight.columns = ['%s_weight' % item for item in sub_interval_weight.columns]

        #step 2 该情景下的股票得分
        if method == 'guojun':
            contextual_alpha = self.combine_subinterval_weight(alpha_data, sub_interval_weight, factor_name)
        else:
            # 各子区间利用ICIR加权加权打分
            # 东方研报《因子系列8》中直接将因子与ICIR权重进行加权；东方研报《因子系列19》中首先将因子值转化为分位数再加权，主要是因为子区间中的因子分布可能有偏，与正态相差较大
            if method == 'dongfang_v2':
                alpha_data.loc[:, self.alpha_factors] = alpha_data.groupby(['tradeDate', 'group_%s' % factor_name])[self.alpha_factors].apply(lambda x: x.rank(pct=True))
            merge_data = pd.merge(alpha_data, sub_interval_weight.reset_index(), on=['tradeDate', 'group_%s' % factor_name])

            weight_columns = ['%s_weight' % item for item in self.alpha_factors]
            merge_data['score'] = (np.array(merge_data[self.alpha_factors]) * np.array(merge_data[weight_columns])).sum(axis=1)
            merge_data['score'] = merge_data.groupby(['tradeDate','group_%s' % factor_name])['score'].transform(st.zscore)

            # 东方研报《因子系列8》直接将ZSCORE作为最终alpha输出；东方研报《因子系列19》中将上述子区间的ZSCORE转换为预期收益率
            if method == 'dongfang_v2':
                contextual_alpha = self.transform_zscore_to_ret(merge_data, factor_name)
            else:
                merge_data['alpha_%s'%factor_name] = merge_data['score']
                contextual_alpha = merge_data.copy()

        # step 3  该情景的权重
        contextual_alpha = contextual_alpha[['tradeDate', 'ticker', 'alpha_%s' % factor_name]]
        contextual_weight = self.get_contextual_weight(alpha_data, sub_interval_weight, factor_name, method)

        contextual_alpha_weight = pd.merge(contextual_alpha, contextual_weight, on=['tradeDate', 'ticker'])
        return contextual_alpha_weight


    def combine_subinterval_weight(self, alpha_data, sub_interval_weight, factor_name):
        """
        国泰君安研报中提到的合成子区间权重，然后利用合成权重进行股票打分
        params:
            alpha_data: DataFrame, alpha信号
            sub_interval_weight: DataFrame, 各子区间权重
            factor_name: str, 情景因子名称
        return:
            DataFrame, 返回计算的合成信号，及权重
        """    
        alpha_data = alpha_data.copy()
        # 国泰君安将截面上的所有股票按照这一风格的因子值进行排序，得到各个股票的分位数。以分位数作为该股票在这一风格维度上高低程度的度量，并以其倒数作为权重进行加权

        points = [i * 1.0 / self.groups for i in range(1, self.groups)]
        mid_points = [0.0] + [(points[i] + points[i - 1]) / 2 for i in range(1, len(points))] + [1.0]
        weight_columns = ['%s_weight' % item for item in self.alpha_factors]

        def _inverse_distance_weight(data):
            """
            倒数距离加权
            """
            q = round(data['%s_q' % factor_name], 6)

            group_weight = [0.0] * self.groups
            pos = np.searchsorted(mid_points, q)
            point = mid_points[pos]
            if point == q:
                group_weight[pos] = 1.0
            else:
                last_point = mid_points[pos - 1]
                weight = np.array([1.0 / (q - last_point), 1.0 / (point - q)])
                weight = weight / np.sum(weight)
                group_weight[pos - 1] = weight[0]
                group_weight[pos] = weight[1]

            return pd.Series(index=range(self.groups), data=group_weight, name='weight')

        alpha_data[range(self.groups)] = alpha_data.apply(lambda x : _inverse_distance_weight(x.copy()), axis=1)
        group_weight_df = pd.melt(alpha_data, id_vars=['ticker', 'tradeDate'], value_vars=range(self.groups), var_name='group_%s' % factor_name, value_name='weight')
        combine_weight_df = pd.merge(group_weight_df, sub_interval_weight.reset_index(), on=['tradeDate', 'group_%s' % factor_name])
        combine_weight_df[weight_columns] = combine_weight_df[weight_columns].apply(lambda x: x * combine_weight_df['weight'])
        final_weight_df = combine_weight_df.groupby(['ticker', 'tradeDate']).apply(lambda x: x[weight_columns].sum())

        merge_data = pd.merge(alpha_data, final_weight_df.reset_index(), on=['tradeDate', 'ticker'])
        merge_data['alpha_%s'%factor_name] = (np.array(merge_data[self.alpha_factors]) * np.array(merge_data[weight_columns])).sum(axis=1)
        
        return merge_data


    def transform_zscore_to_ret(self, data, factor_name, x_name='score', y_name='next_return', window=12):
        """
        东方研报2提到的将ZSCORE转化为预期收益
        params:
            data: DataFrame, ZSCORE信息
            factor_name: str, 情景因子名称
            x_name: str, ZSCORE列名称
            y_name: str, 预期收益列名称
            window: int, 窗口期
        return:
            DataFrame, 返回转换后的合成信号
        """   
        def _ols_fit(data):
            result = sm.OLS(data[y_name].values, sm.add_constant(data[x_name].values), missing='drop').fit()
            return pd.Series(index=['a', 'b'], data=[result.params[0], result.params[1]])

        transform_para = data.groupby(['tradeDate', 'group_%s'%factor_name]).apply(lambda x : _ols_fit(x)).reset_index()
        transform_para = transform_para.sort_values(by=['tradeDate', 'group_%s' % factor_name], ascending=[True, True])
        transform_para.loc[:, ['a', 'b']] = transform_para.groupby('group_%s' % factor_name)[['a', 'b']].shift(1)
        transform_para = transform_para.set_index('tradeDate').groupby('group_%s' % factor_name)[['a', 'b']].rolling(window).mean().dropna().reset_index()

        merge_data = pd.merge(data, transform_para, on=['tradeDate', 'group_%s' % factor_name])
        merge_data['alpha_%s'%factor_name] = merge_data['a'] + merge_data['score'] * merge_data['b']

        return merge_data


    def get_contextual_weight(self, data, sub_interval_weight, factor_name, method):
        """
        不同情景间的信号权重
        params:
            data: DataFrame, 信号数据
            sub_interval_weight: DataFrame, 各子区间权重
            factor_name: str, 情景因子名称
            method: str, 方法名称
        return:
            DataFrame, 返回权重信息
        """   
        data = data.copy()
        weight_columns = ['%s_weight' % item for item in self.alpha_factors]
        def _cal_contextual_feature(x):
            # 个股情景特征
            x = x * 100.0
            if x <= 50:
                return (x - 101) * 10 / 101.0
            else:
                return x * 10.0 / 101

        def _cal_contextual_distance(x):
            # 模型区分度
            x = x.copy()[weight_columns]
            dis = distance.pdist(x.values) / np.sqrt(len(self.alpha_factors))

            return np.mean(dis)

        if method == 'guojun':
            # 国君等权处理各个情景
            data.loc[:, 'weight_%s'%factor_name] = 1.0 / len(self.contextual_factors)
        elif method == 'dongfang_v1':
            # 东方《因子系列8》根据股票在各个情景下的取值极端程度(个股情景特征)进行加权
            data.loc[:, 'weight_%s'%factor_name] = data['%s_q'%factor_name].map(_cal_contextual_feature).abs()
        else:
            # 东方《因子系列19》采用该情景下不同情景区间的模型区分度作为情景间预期收益率的权重
            sub_interval_weight = sub_interval_weight.reset_index()
            weight = sub_interval_weight.groupby('tradeDate').apply(_cal_contextual_distance).reset_index()
            weight.columns = ['tradeDate', 'weight_%s'%factor_name]
            data = pd.merge(data, weight, on='tradeDate')

        return data[['tradeDate', 'ticker', 'weight_%s' % factor_name]]
    
    
'''


为了比较动态情景多因子模型的效果，该章节增加了两个对照组：等权因子合成、及利用ICIR的因子合成方法
动态情景模型的加权按照上述三篇研报中所述步骤进行，具体细节有所不同。比如国泰君安研报中子区间采用了最大化复合IC的因子权重，本文选取了常规ICIR进行替代，便于比较几篇研报后续加权步骤的效果。


'''
import pandas as pd
import numpy as np
import time, pickle

start_time = time.time()
raw_data_dir = "./raw_data"
dcam_alpha  = pd.read_csv(os.path.join(raw_data_dir, 'orth_dcam_data.csv'), index_col=0, dtype={'ticker':np.str, 'tradeDate':np.str})

factor_score_dict = {}

# 计算因子IC
window = 24
ic_all = dcam_alpha.groupby(['tradeDate']).apply(lambda x:  pd.Series(index=alpha_factors, data=x[['neu_next_return']+alpha_factors].corr(method='spearman').values[0, 1:]))
ic_all = ic_all.sort_index()
ic_all[alpha_factors] = ic_all[alpha_factors].shift(1)

# 等权权重构建
ic_mean = ic_all.rolling(window).mean()
equal_sign = ic_mean.apply(np.sign).replace(0, 1).dropna()
equal_weight = equal_sign.divide(equal_sign.abs().sum(axis=1), axis=0)
factor_score_dict['equal'] = combine_factor(dcam_alpha, equal_weight)

#ICIR加权权重
ic_std = ic_all.rolling(window).std()
ic_ir = (ic_mean/ic_std).dropna()
ic_ir_weight = ic_ir.divide(ic_ir.abs().sum(axis=1), axis=0)
factor_score_dict['icir'] = combine_factor(dcam_alpha, ic_ir_weight)

# 动态情景alpha
dcam_model = DCAM(dcam_alpha, contextual_factors, alpha_factors, groups=5)

# 分别对应着国泰君安、东方证券研报1，东方证券研报2中所述方法
for method in ['guojun', 'dongfang_v1', 'dongfang_v2']:  
    print('利用方法%s构建动态情景模型中.......' % method)
    factor_score_dict[method] = dcam_model.get_dcam_alpha(method)

with open(os.path.join(raw_data_dir, 'dcam_all_alpha.pickle'), 'wb') as handle:
    pickle.dump(factor_score_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print "Time cost: %s seconds" % (time.time() - start_time)


'''
四、实证分析
该部分耗时 1小时左右
为了考察动态情景模型的效果，本节从IC及组合回测两种方式进行了比较。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

4.1 IC比较

'''

import pandas as pd
import numpy as np
import scipy.stats as st

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
        lambda x: pd.Series(data=x[factor_name + [ret_name]].corr(method=method).values[0, 1:], index=factor_name)).dropna()
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

import pickle
save_dir = "./raw_data"
with open(os.path.join(save_dir, 'dcam_all_alpha.pickle'), 'rb') as fHandler:
    factor_score_dict = pickle.load(fHandler)
    
# 设置起始时间和结束时间
begin_date = '2013-01-01'
end_date = '2018-06-30'

# 获得月收益率
month_return = DataAPI.MktEqumAdjGet(beginDate=begin_date, endDate=end_date, field=u"ticker,endDate,chgPct,",pandas="1")
month_return.rename(columns={'endDate': 'tradeDate', 'chgPct': 'month_return'}, inplace=True)
month_return.sort_values(['ticker', 'tradeDate'], inplace=True)
month_return['next_month_return'] = month_return.groupby('ticker')['month_return'].shift(-1)
month_return.dropna(inplace=True)
month_return['tradeDate'] = month_return['tradeDate'].apply(lambda x : x.replace("-", ''))

method_list = ['equal', 'icir', 'guojun', 'dongfang_v1', 'dongfang_v2']
ic_res_list = []
for method in method_list:
    signal_df = factor_score_dict[method]
    ic = calc_ic(signal_df, month_return, 'alpha', 'next_month_return')
    ic_des = ic_describe(ic)

    ic_res_list.append(ic_des)
        
ic = pd.concat(ic_res_list, axis=0).round(4)
ic.index = method_list

print('rankIC 情况')
print(ic.to_html())    

'''

结果显示，等权是所有方法中最差的；动态情景模型在不同的加权方案下表现的也很不一致，具体的说

国君及东方研报1中所述的方法表现很类似，相较于传统ICIR加权，二者在IC均值上并未带来明显的提升，但从波动上来看，二者表现稍好
东方研报2中所述的方法在IC均值上带来最高的提升，但是明显加剧了波动，导致其IR变弱

4.2 组合回测

对第三章节5种方法构建的alpha进行回测，共测试了3种组合形式: Top100等权选股组合，long-short多空组合及指数增强组合，具体参数如下:

三种组合的共有参数
选股池: 中证全指成分股
时间范围: 20130101-20180630
调仓参数: 月度调仓，买卖交易费千分之1.5
指数增强组合特有参数
基准: 中证500
个股上限权重: 2.5%
市值敞口限制: [-0.01, 0.01]
行业敞口限制: [-0.002, 0.002]

'''

import quartz_extensions.Optimizer.optimize as opt

def optimize_rhac(signal, date, benchmark='ZZ500'):
    """
    参考知识库中的-组合优化器文档
    """
    # 创建优化器对象
    pspec = opt.UqerOptimizer(signal, date, benchmark_str=benchmark)
    # 添加约束
    # 个股上下限约束
    pspec.add_constraint(default_min_weight=0., default_max_weight=0.025)
    # 行业中性
    pspec.add_constraint(is_industry_neutralize=True, active_indu_lower=-0.002, active_indu_upper=0.002)
    # 风格约束
    pspec.add_constraint(spec_style={'SIZE':[-0.01, 0.01]})
    pspec.solve()
    weights = pspec.assets[pspec.assets.optimal_weights > 0.00001]
    return weights, pspec.optimal


import time
from CAL.PyCAL import * 

start_time = time.time()
# -----------回测参数部分开始，可编辑------------
start = '2013-01-01'                       # 回测起始时间
end = '2018-06-30'                         # 回测结束时间
benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('000985.ZICN')           # 证券池，支持股票和基金
capital_base = 1000000000.0                     # 起始资金
freq = 'd'                              
refresh_rate = Monthly(1)  
commission = Commission(buycost=0.0015, sellcost=0.0015, unit='perValue')

# 运行结果保存pickle的位置
save_dir = "./raw_data"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

with open(os.path.join(save_dir, 'dcam_all_alpha.pickle'), 'rb') as fHandler:
    factor_score_dict = pickle.load(fHandler)

# 把回测参数封装到 SimulationParameters 中，供 quick_backtest 使用
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate)
# 获取回测行情数据
data = quartz.get_backtest_data(sim_params)

backtest_results_dict = {}

# 对4种不同类型分别进行测试
for method, alpha in factor_score_dict.items():
    factor_data = alpha.copy()
    factor_data = factor_data.set_index('tradeDate')
    factor_data['ticker'] = factor_data['ticker'].apply(lambda x: x+'.XSHG' if x[:2] in ['60'] else x+'.XSHE')
    
    q_dates = factor_data.index.unique()
    
    for portfolio_type in ['rhac', 'long', 'short', 'top100']: 
        print ('backtesting for method %s portfolio %s ..................................' % (method, portfolio_type))
        
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
with open(os.path.join(save_dir, 'dcam_backtest.pickle'), 'wb') as handle:
    pickle.dump(backtest_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ('Done! Time Cost: %s seconds' % (time.time()-start_time))


# 读取上述回测的结果，都存在了backtest_results_dict里，便于后续组合分析

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
import os
import pandas as pd
import numpy as np
import time
import pickle
from CAL.PyCAL import * 

save_dir = "./raw_data"
with open(os.path.join(save_dir, 'dcam_backtest.pickle'), 'rb') as fHandler:
    backtest_results_dict = pickle.load(fHandler)

capital_base = 1000000000.0

key_list = ['equal', 'icir', 'guojun', 'dongfang_v1', 'dongfang_v2']

'''

4.2.1 Top100组合分析

该小节分析了top100的纯多头组合表现


'''
backtest_origin_indic = [u'alpha', u'beta', u'sharpe', u'annualized_return', u'max_drawdown']

def get_top100_result(results):  
    """
    top100组合的回测结果展示及分析
    params:
        results: dict, 回测结果
    return:
        DataFrame, 返回计算的指标
    """        
    backtest_pd = pd.DataFrame(index=key_list, columns=backtest_origin_indic+[u'volatility'])
    
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.grid()

    for key in key_list:
        bt = results[key+"_top100"]['bt']
        data = bt[[u'tradeDate', u'portfolio_value', u'benchmark_return']]
        data['portfolio_return'] = data.portfolio_value / data.portfolio_value.shift(1) - 1.0  # 总头寸每日回报率
        data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0] / capital_base - 1.0
        data['portfolio'] = data.portfolio_return + 1.0
        data['portfolio'] = data.portfolio.cumprod()  # 总头寸不对冲时的净值序列
        volatility = np.std(data['portfolio_return']) * np.sqrt(252)
        backtest_pd.loc[key] = np.array([results[key+"_top100"][item] for item in backtest_origin_indic] + [volatility])
        ax.plot(data['tradeDate'], data[['portfolio']], label=key)

        cols = [(u'指标', u'Alpha'), (u'指标', u'Beta'), (u'指标', u'夏普比率'), (u'纯股票多头时', u'年化收益'), (u'纯股票多头时', u'最大回撤'), (u'纯股票多头时', u'年化波动')]
    backtest_pd.columns = pd.MultiIndex.from_tuples(cols)
    backtest_pd.index.name = u'不同类型'
    
    ax.legend(loc=0)
    ax.set_ylabel(u"净值", fontproperties=font, fontsize=16)
    ax.set_title(u"Top100组合净值走势", fontproperties=font, fontsize=16)

    return backtest_pd.astype(float).round(4)

backtest_pd = get_top100_result(backtest_results_dict)
print(backtest_pd.to_html())

'''

结果显示与IC结果很类似，等权合成的方法是最差的，动态情景模型按照不同的加权方案来看，效果也不同：

国君及东方研报1中所述的方法相较于传统ICIR来说，优势均不太明显；东方研报1的方法在波动上表现最好，但牺牲了收益，导致夏普比率有所下降；国君的方法在各项指标上都较ICIR有提升
东方研报2的方法在收益方面表现最好，波动稍微差了一些，相较于ICIR，其绝对收益提升了11%，夏普比率从1.20提升到1.47

4.2.2 long-short组合分析

该小节分析了Top10% - Bottom10%的多空组合走势

'''

backtest_heged_indic = [u'年复合收益', u'夏普比率', u'最大回撤', u'收益波动率']

def get_long_short_result(results):  
    """
    多空组合回测结果展示及分析
    params:
        results: dict, 回测结果
    return:
        DataFrame, 返回计算的指标
    """        
    
    backtest_pd = pd.DataFrame(index=key_list, columns=backtest_heged_indic)
    
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    ax.grid()
    
    for key in key_list:   
        data_list = []
        for portfolio in ['long', 'short']:
            data = results["%s_%s"%(key, portfolio)]['bt'][[u'tradeDate', u'portfolio_value']]
            data['portfolio_return'] = data.portfolio_value / data.portfolio_value.shift(1) - 1.0  # 总头寸每日回报率
            data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0] / capital_base - 1.0
            data_list.append(data.set_index('tradeDate')['portfolio_return'])
        long_short_return = (data_list[0] - data_list[1]).fillna(0.0)
        long_short_value = (long_short_return + 1.0).cumprod()
        
        hedged_max_drawdown = max([1 - v / max(1, max(long_short_value[:i + 1])) for i, v in enumerate(long_short_value)])  # 对冲后净值最大回撤
        hedged_volatility = np.std(long_short_return) * np.sqrt(252)
        hedged_annualized_return = (long_short_value.values[-1]) ** (252.0 / len(long_short_value)) - 1.0
        sharpe_ratio = hedged_annualized_return / hedged_volatility
        backtest_pd.loc[key] = [hedged_annualized_return, sharpe_ratio, hedged_max_drawdown, hedged_volatility]

        ax.plot(long_short_value.index, long_short_value.values, label=key)

    ax.legend(loc=0)
    ax.set_ylabel(u"净值", fontproperties=font, fontsize=16)
    ax.set_title(u"long-short组合净值走势", fontproperties=font, fontsize=16)

    return backtest_pd.astype(float).round(4)

backtest_pd = get_long_short_result(backtest_results_dict)
print(backtest_pd.to_html())

'''

与top100组合表现类似，等权组合表现最差；国君与东方研报1中方法几乎与ICIR加权表现一致，只有微弱的优势；东方研报2中方法明显超过其余加权方案，其年化收益相较于传统ICIR模型的29.3%提升到了39.6%，代价是增加了波动

   调试 运行
文档
 代码  策略  文档
4.2.3 指数增强组合分析

上述两种组合并未考虑风险因子暴露的情况，该小节分析了在控制市值、行业等风险因子后的模型表现

'''
backtest_origin_indic = [u'information_ratio']
backtest_heged_indic = [u'hedged_annualized_return', u'hedged_max_drawdown', u'hedged_volatility']

def get_tracking_index_result(results):  
    """
    指数增强组合的回测结果展示及分析
    params:
        results: dict, 回测结果
    return:
        DataFrame, 返回计算的指标
    """        
    backtest_pd = pd.DataFrame(index=key_list, columns=backtest_heged_indic+backtest_origin_indic)
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax1 = ax.twinx()
    ax.grid()
    
    color_list = ['grey', 'black', 'green', 'blue', 'red']
    for number, key in enumerate(key_list):
        bt = results[key+"_rhac"]['bt']
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
        
        backtest_pd.loc[key] = np.array([hedged_annualized_return, hedged_max_drawdown, hedged_volatility] + [results[key+"_rhac"][item] for item in backtest_origin_indic])
        
        ax.plot(data['tradeDate'], data[['excess']], linewidth=2, color=color_list[number], label=key)
        ax1.fill_between(data['tradeDate'].values, 0, max_drawdown_line, color=color_list[number], label=key)
    cols = [(u'对冲后', u'年化收益'), (u'对冲后', u'最大回撤'), (u'对冲后', u'收益波动率'), (u'风险指标', u'信息比率')]
    backtest_pd.columns = pd.MultiIndex.from_tuples(cols)
    backtest_pd.index.name = u'不同类型'
    
    ax.set_ylim(-1, 3)
    ax1.set_ylim(-0.1, 0.1)
    ax.legend(loc=0)
    ax.set_ylabel(u"对冲净值（曲线图）", fontproperties=font, fontsize=16)
    ax1.set_ylabel(u"回撤（柱状图）", fontproperties=font, fontsize=16)
    ax.set_title(u"中证500指数增强组合对冲净值走势", fontproperties=font, fontsize=16)

    return backtest_pd.astype(float).round(4)

backtest_pd = get_tracking_index_result(backtest_results_dict)
print(backtest_pd.to_html())


'''

可以看出，在指数增强组合中，三种加权方案构建的动态情景模型相比较ICIR均有明显优势，表现最好的还是东方研报2中的方法，其年化收益相较ICIR的14.6%提升到17.7%，信息比率从1.83提升到2.17
该指数增强组合只限制了市值及行业敞口，读者也可以同时限制跟踪误差，换手率等更多条件来测试
   调试 运行
文档
 代码  策略  文档
参考

1、 东方证券 《动态情景多因子Alpha模型 - 因子选股系列研究之八》
2、 东方证券 《动态情景多因子Alpha模型再思考 - 因子选股系列研究之十九》
3、 国泰君安 《数量化专题之一百一十六：风格域划分下的基本面多因子选股策略》
4、 天风证券 《因子正交全攻略，理论、框架与实践》

'''
