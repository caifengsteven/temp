# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:59:52 2020

@author: Asus
"""

'''
导读
A. 研究目的：本文利用优矿因子库中的因子，参考广发证券《探寻资金流背后的风格轮动规律-——多因子Alpha系列报告之（三十七）》（原作者：史庆盛）中的研究方法，对研报的结果进行了实证分析，用以探究A股资金流背后的风格轮动规律

B. 文章结构：本文共分为3个部分，具体如下

一、数据准备和处理：资金流比率的计算、风格因子的获取及因子收益率计算、资金流偏好度的计算、资金流风格配置矩阵的生成

二、各个资金流策略的表现情况及资金流策略合并后的表现情况

三、采用滑动窗口的方法预测，各个资金流策略的表现情况及资金流策略合并后的表现情况

C. 研究结论：

采用资金流比率对资金强度进行刻画，找到当前市场中资金关注的焦点，并分析随后市场风格轮动的规律，可以为我们在风格轮动中提供重要的建议。
通过组合各个子资金流比率构建的策略优于单个策略，同时基于滑动预测的方式下，该风格轮动规律仍可获取超额收益。
D. 时间说明

总耗时25分钟左右
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
链接：https://uqer.datayes.com/community/share/0DfBcVxVqR8TAw9S1sNQJZ3p2Dk0/private；密码：6962。请前往查看并注意保密。
请在运行之前，克隆上面的代码，并存成lib(右上角->另存为lib，不要修改名字)

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
第一部分：数据准备和处理
该部分耗时 15分钟
该部分内容为：

资金流比率的计算:分别对三种资金流（净流入资金、融资余额、沪深股通资金），从横向和纵向两个角度构造6种资金流比率;

风格因子选取和构建:从优矿中获取计算因子所需的依赖数据和因子库中已有的因子数据，计算并对齐因子，对因子进行标准化、缺失值填充、行业中性化、归一化处理;

风格因子收益率计算:包括多空、多头、空头三种收益率;

资金流偏好度的计算: 资金流偏好度和风格因子有关，指定风格的资金流偏好度为某一种风格因子的多头组合以及空头组合按照既定的资金流和既定的计算方法计算的资金的偏离度；

资金流风格配置矩阵生成:根据当期的资金流偏好度及下期风格因子的表现情况进行统计分析， 生成风格配置矩阵，用于对风格因子择时。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''

import pandas as pd
import time
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
import copy
import lib.qnt_lib as quant_util


#设置数据目录
raw_data_dir = "./cash_flow_stype_rotation_data3"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)
    
    
#交易日历
start_date, end_date = '2010-01-01', '2018-08-01'
calendar_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
week_end_list = calendar_df[calendar_df['isWeekEnd']==1]['calendarDate'].values
trade_date_list = calendar_df[calendar_df['isOpen']==1]['calendarDate'].values


'''

1.1 资金流比率的计算(5分钟)

针对三种资金流（资金净流入、融资增量、沪港深通资金），从横向（资金在交易额中的占比）和纵向（资金环比增长率）构造构造6种资金流比率：
1)净流入比率(pct_netMoneyInflow);
2)融资余额交易额占比(pct_finVal);
3)沪深股通交易额占比(pct_partyVol);
4)主动流入资金增量环比增长(qoq_netMoneyInflow);
5)融资余额增量环比增长(qoq_finVal);
6)沪深股通增量环比增长(qoq_partyVol);
其中前3个资金流偏好度使用的是资金流在交易额中的占比的计算方法，后3个资金流偏好度使用的是环比增长率的计算方法。
资金在交易额中的占比 = 本期的资金增量 / 本期的交易额之和
资金环比增长率 = 本期的资金增量 / 上一期的资金增量 – 1

计算资金流比率关键是对股票前后两周资金增量的计算，具体步骤如下：

先通过uqer API 取到三种资金流日度的数据；
汇总每日的资金流数据得到周度的资金流数据；
计算前后两周资金流的变化； 以资金净流入为例，最后得到的数据格式如下：
图片注释

结果包含三列：股票ID、日期、当前日期资金流相对于上周的变化量

由于后续计算资金流偏好度和风格因子收益率时需要用到股票周度的成交额和周收益率数据，所以在该部分先获取这两类数据


'''
#获取日度的资金净流入、融资余额、沪深港通资金数据；获取票周度的成交额和周收益率数据

#资金净流入
t0 = time.time()
cash_df_list = []
for year in range(2010,2019):
    print (year)
    df_tmp1 =DataAPI.MktEquFlowGet(tradeDate="",secID="",ticker=u"",beginDate="%s0101"%str(year),endDate="%s0630"%str(year),field=['secID','tradeDate','netMoneyInflow'],pandas="1")
    df_tmp2 =DataAPI.MktEquFlowGet(tradeDate="",secID="",ticker=u"",beginDate="%s0701"%str(year),endDate="%s1231"%str(year),field=['secID','tradeDate','netMoneyInflow'],pandas="1")
    cash_df_list.append(df_tmp1)
    cash_df_list.append(df_tmp2)
cash_df = pd.concat(cash_df_list, axis=0)  
t1 = time.time()
print ("get daily cash frame finished, using %s seconds !"%round(t1-t0,2))
cash_df = cash_df[((cash_df['secID'].str.startswith('0'))|(cash_df['secID'].str.startswith('3'))|(cash_df['secID'].str.startswith('6')))]
cash_df.to_csv('%s/daily_cash_df.csv'%raw_data_dir,index=False,encoding='utf-8')

#融资余额
t0 = time.time()
finval_df_list = []
for year in range(2010,2019):
    print (year)
    df_tmp =DataAPI.FstDetailGet(secID="",ticker=u"",beginDate="%s0101"%str(year),endDate="%s1231"%str(year),field=['secID','tradeDate','finVal'],pandas="1")
    finval_df_list.append(df_tmp)
finval_df = pd.concat(finval_df_list, axis=0) 
finval_df = finval_df[((finval_df['secID'].str.startswith('0'))|(finval_df['secID'].str.startswith('3'))|(finval_df['secID'].str.startswith('6')))]
t1 = time.time()  
print ("get finval frame finished, using %s seconds !"%round(t1-t0,2))
finval_df.to_csv('%s/daily_finval_df.csv'%raw_data_dir,index=False,encoding='utf-8')

#沪深港通资金
t0 = time.time()
hkval_df1 = DataAPI.HKshszHoldGet(secID=u"",ticker=u"",tradeCD=["1","2"],ticketCode=u"",partyName=u"",beginDate=u"20170101",endDate=u"20171231",field=["secID","endDate","partyVol","tradeCD"],pandas="1")
hkval_df2 = DataAPI.HKshszHoldGet(secID=u"",ticker=u"",tradeCD=["1","2"],ticketCode=u"",partyName=u"",beginDate=u"20180101",endDate=u"20180801",field=["secID","endDate","partyVol","tradeCD","partyPct"],pandas="1")
hkval_df = pd.concat([hkval_df1, hkval_df2], axis=0)
hkval_df.rename(columns={'endDate':'tradeDate'},inplace=True)
hkval_df = hkval_df[((hkval_df['secID'].str.startswith('0'))|(hkval_df['secID'].str.startswith('3'))|(hkval_df['secID'].str.startswith('6')))]
t1 = time.time()  
print ("get hkval frame finished, using %s seconds !"%round(t1-t0,2))
hkval_df.to_csv('%s/daily_hkval_df.csv'%raw_data_dir,index=False,encoding='utf-8')

#周成交额和收益率
t0 = time.time()
trnval_df_list = []
for year in range(2010,2019):
    print (year)
    tmp_df = DataAPI.MktEquwGet(secID=u"",ticker=u"",weekEndDate=u"",beginDate="%s0101"%str(year),endDate="%s1231"%str(year),isOpen=u"1",field=['secID','endDate','turnoverValue','return'],pandas="1")
    trnval_df_list.append(tmp_df)
trnval_df = pd.concat(trnval_df_list, axis=0)  
trnval_df.rename(columns={'endDate':'tradeDate'},inplace=True)
trnval_df = trnval_df[((trnval_df['secID'].str.startswith('0'))|(trnval_df['secID'].str.startswith('3'))|(trnval_df['secID'].str.startswith('6')))]
t1 = time.time() 
print ("get trnval frame finished, using %s seconds !"%round(t1-t0,2))
trnval_df.to_csv('%s/weekly_mktval_df.csv'%raw_data_dir,index=False,encoding='utf-8') 


#数据存储后可直接读取每天的资金流数据
daily_cash_df = pd.read_csv('%s/daily_cash_df.csv'%raw_data_dir,encoding='utf-8')
daily_finval_df = pd.read_csv('%s/daily_finval_df.csv'%raw_data_dir,encoding='utf-8')
dailyhkval_df = pd.read_csv('%s/daily_hkval_df.csv'%raw_data_dir,encoding='utf-8')
weekly_trnval_df = pd.read_csv('%s/weekly_mktval_df.csv'%raw_data_dir,encoding='utf-8')


#计算周度资金流及周度资金流的变化
def cal_week_cash(df):
    df = df.copy()
    df = df.set_index('tradeDate').sort_index().resample('W').sum().reset_index()
    df['tradeDate'] =df['tradeDate'].dt.strftime('%Y-%m-%d')
    df = df.merge(calendar_df[['calendarDate','prevTradeDate']],left_on='tradeDate',right_on='calendarDate',how='left')
    df = df.drop_duplicates(subset=['prevTradeDate'],keep='first')[['prevTradeDate','netMoneyInflow']]
    return df
daily_cash_df['tradeDate'] =pd.to_datetime(daily_cash_df['tradeDate'])
weekly_cash_df = daily_cash_df.groupby('secID').apply(lambda x:cal_week_cash(x)).reset_index().drop(['level_1'],axis=1)
weekly_cash_df.rename(columns={'prevTradeDate':'tradeDate'},inplace=True)
weekly_finval_df=daily_finval_df[daily_finval_df['tradeDate'].isin(week_end_list)]
weekly_hkval_df=dailyhkval_df[dailyhkval_df['tradeDate'].isin(week_end_list)][['secID','tradeDate','partyVol']]

def cal_cash_change(df, cash_col):
    df = df.copy()
    df.sort_values('tradeDate',inplace=True)
    df[cash_col] = df[cash_col] - df[cash_col].shift(1)
    return df.dropna()
weekly_cashchg_df = weekly_cash_df.groupby('secID',as_index=False).apply(lambda x: cal_cash_change(x, cash_col='netMoneyInflow')).reset_index().drop(['level_0','level_1'],axis=1)
weekly_finvalchg_df = weekly_finval_df.groupby('secID',as_index=False).apply(lambda x: cal_cash_change(x, cash_col='finVal')).reset_index().drop(['level_0','level_1'],axis=1)
weekly_hkvalchg_df = weekly_hkval_df.groupby('secID',as_index=False).apply(lambda x: cal_cash_change(x, cash_col='partyVol')).reset_index().drop(['level_0','level_1'],axis=1)


'''

1.2 风格因子选取和构建(5分钟)

使用的风格因子从盈利、成长、杠杆、流动、技术、规模、质量、价值8大类因子中选取，由于同类因子的相关性及表现一般较为相似，因此我们仅在每个大类风格因子中各挑出一个风格因子进行研究，分别是ROE、净利润增长率(NetProfitGrowRate)、流通股本/总股本(fshareRatio)、6日成交金额的移动平均值(TVMA6)、股票的20日收益(REVS20)、BP(行业相对,PBIndu)、流动负债率(tclRatio)、总市值(MktValue)，如下图所示（红色字体因子代表选择的因子），图中预定方向表示在一般情况下，投资该因子的风格因子的方向，其中“+”表示在一般情况下，我们会配置该风格因子值大的股票,“-”表示配置该风格因子值小的股票。
图片注释


'''
#流动负债率因子不能从uqer 因子获取API直接得到，需要先读取相关数据计算得到。 
def get_fin_data_latest(fin_data_frame, col_name=['value']):

    """
    获取最新财务数据
    :param fin_data_frame: financial column= ['ticker','pub_date',’end_date',[fin_value]], index=num, pub_date='%Y%m%d'
    :param col_name: list, column name of value, 可以有多个列
    :return: column= ['ticker','pub_date','end_date',[fin_value]], 'ticker','pub_date' 是唯一性约束
    """

    fin_df = fin_data_frame.copy()

    def get_latest_perticker(df, col_name):
        tmp_df = df.copy()
        tmp_df.dropna(subset=col_name, how='all', inplace=True)
        tmp_df.sort_values(['publishDate', 'endDate'], inplace=True)
        tmp_df.drop_duplicates(subset=['publishDate'], keep='last', inplace=True)
        tmp_df['max_end_date'] = tmp_df['endDate'].rolling(window=6, min_periods=1).max()
        tmp_df['max_end_date'] = tmp_df['max_end_date'].astype(np.int64).astype(np.str)
        tmp_df = tmp_df[tmp_df['endDate'] == tmp_df['max_end_date']]
        return tmp_df[['secID', 'publishDate', 'endDate'] + col_name]

    fin_df = fin_df.groupby(['secID']).apply(get_latest_perticker, col_name)
    fin_df.reset_index(inplace=True, drop=True)
    return fin_df

lib_df = DataAPI.FdmtBSGet(ticker=u"",secID="",reportType=u"",endDate=u"20180801",beginDate=u"20100101",field=['secID','publishDate','endDate','ticker','endDateRep','fiscalPeriod','TCL','TLiab'],pandas="1")
lib_df['endDate_mon'] = pd.to_datetime(lib_df['endDate']).dt.month.astype(str)
lib_df = lib_df[lib_df['fiscalPeriod']==lib_df['endDate_mon']]
lib_df['publishDate'] = pd.to_datetime(lib_df['publishDate']).dt.strftime('%Y%m%d')
lib_df['endDate'] = pd.to_datetime(lib_df['endDate']).dt.strftime('%Y%m%d')
lib_df.fillna(0,inplace=True)
lib_latest=get_fin_data_latest(lib_df,col_name=['TCL','TLiab'])
lib_latest['tclRatio'] = lib_latest['TCL'] / lib_latest['TLiab']
tcl_factor = quant_util.fin_data_pit2cont(lib_latest, start_date, end_date)
tcl_factor['publishDate'] = pd.to_datetime(tcl_factor['publishDate'],format='%Y%m%d').dt.strftime('%Y-%m-%d')
weekly_tcl_factor = tcl_factor[tcl_factor['publishDate'].isin(week_end_list)]
weekly_tcl_factor.rename(columns={'publishDate':'tradeDate'}, inplace=True)
weekly_tcl_factor = weekly_tcl_factor[((weekly_tcl_factor['secID'].str.startswith('0'))|(weekly_tcl_factor['secID'].str.startswith('3'))|(weekly_tcl_factor['secID'].str.startswith('6')))]

#其他7个因子可通过因子获取API得到
t0 = time.time()
facor_df_list =[]
i = 0
for dt in week_end_list:
    if i%100 ==0:
        print (dt)
    tmp1 = DataAPI.MktStockFactorsOneDayProGet(tradeDate=dt,secID=u"",ticker=u"",field=['secID','tradeDate','ROE','NetProfitGrowRate','NegMktValue','MktValue','TVMA6','REVS20','PBIndu'],pandas="1")
    i+=1
    facor_df_list.append(tmp1)
factor_df = pd.concat(facor_df_list, axis=0)
t1 = time.time()
print ("get factor frame finished, using %s seconds !"%round(t1-t0,2))

factor_df['fshareRatio'] = factor_df['NegMktValue'] / factor_df['MktValue']
weekly_factor_df = factor_df.merge(weekly_tcl_factor,on=['secID','tradeDate'])[['secID','tradeDate','ROE','NetProfitGrowRate','fshareRatio','TVMA6','REVS20','PBIndu','tclRatio','MktValue']]
weekly_factor_df = weekly_factor_df.dropna(subset=['secID'])
weekly_factor_df = weekly_factor_df[((weekly_factor_df['secID'].str.startswith('0'))|(weekly_factor_df['secID'].str.startswith('3'))|(weekly_factor_df['secID'].str.startswith('6')))]
weekly_factor_df.to_csv('%s/raw_weekly_factor_df.csv'%raw_data_dir,encoding='utf-8',index=False)


#因子值填充、去极值、标准化: 基于行业中位数填充缺失的因子值;利用绝对中位数差法对因子去极值;最后对各个因子在行业内进行标准化(ZSCORE)
factor_list = weekly_factor_df.columns[2:]
standard_weekly_factor_df = quant_util.fillna_indu_median(weekly_factor_df,factor_list).dropna(subset=['secID'])
standard_weekly_factor_df = quant_util.mad_winsorize(standard_weekly_factor_df, factor_list, sigma_n=3)
del standard_weekly_factor_df['industryName1']
standard_weekly_factor_df = quant_util.zscore_by_indu(standard_weekly_factor_df,factor_list)
del standard_weekly_factor_df['industryName1']
standard_weekly_factor_df.to_csv('%s/standard_weekly_factor_df.csv'%raw_data_dir,encoding='utf-8',index=False)


#直接读取因子值
weekly_factor_df = pd.read_csv('%s/raw_weekly_factor_df.csv'%raw_data_dir,encoding='utf-8')
standard_weekly_factor_df = pd.read_csv('%s/standard_weekly_factor_df.csv'%raw_data_dir,encoding='utf-8')


'''


1.3 风格因子收益率计算(1分钟)

研报中在评估策略表现的时候，分别展示了多空（做多因子值最大的一组，做空因子值最小的一组）、多头（做多因子值最大的一组）、空头（做多因子值最小的一组）三种方式下的表现情况，所以在计算因子收益率时，也分别计算了多空、多头、空头三种方式的收益率。

'''


#考虑到市值对因子收益率的影响， 计算之前先对其他因子（除市值以外）进行市值中性化
neu_cols = ['MktValue']
standard_weekly_signal_df1 = standard_weekly_factor_df.copy()
standard_weekly_signal_df1 = standard_weekly_signal_df1.dropna()
need_neu_cols = standard_weekly_signal_df1.columns[2:-1]
def neu_dframe_cols(df, cols, neu_cols):
    tdframe = df.copy()
    # tdframe = df
    Y = np.array(tdframe[cols])
    X = np.array(tdframe[neu_cols])
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    tdframe[cols] = results.resid
    return tdframe

standard_weekly_signal_df1 = standard_weekly_signal_df1.groupby(['tradeDate'],as_index=False).apply(neu_dframe_cols, need_neu_cols, neu_cols)
standard_weekly_signal_df1 = standard_weekly_signal_df1.reset_index().drop(['level_0','level_1'],axis=1).dropna()


def signal_grouping(signal_df, factor_name, ngrp):
    """
    因子分组， 每天根据因子值将股票进行等分，编号0 ~ ngrp-1, 编号越大， 因子值越大
    params:
            signal_df: DataFrame, columns=['ticker', 'tradeDate', 'factor'], 股票的因子值, factor为股票当日的因子值
            factor_name:　str, signal_df中因子值的列名
            ngrp: int, 分组组数
    return:
            DataFrame, signal_df在原本的基础上增加一列'group', 记录每日分组
    """
    signal_df_tmp = signal_df.copy()
    signal_df_tmp.dropna(subset=[factor_name], inplace=True)
    signal_df_tmp['group'] = signal_df_tmp.groupby('tradeDate')[factor_name].apply(lambda x: (x.rank()-1)/len(x)*ngrp).astype(int)                         
    return signal_df_tmp

def long_short_backtest(signal_df, return_df, factor_name, return_name, direction=1, group_num=5): 
    bt_df = signal_df.merge(return_df, on=['secID', 'tradeDate'], how='right')    
    # 分成n组, 保留因子值最大和最小的两组
    bt_df.dropna(subset=[factor_name], inplace=True)
    bt_df = signal_grouping(bt_df, factor_name=factor_name, ngrp=group_num)
    bt_df = bt_df[bt_df['group'].isin([0, group_num-1])]    
    # 计算权重：每组等权    
    count_df = bt_df.groupby(['tradeDate', 'group']).apply(lambda x:len(x)).reset_index()
    count_df.columns=['tradeDate', 'group', 'count']
    bt_df = bt_df.merge(count_df, on=['tradeDate', 'group'])
    bt_df['weight'] = 1.0/bt_df['count']    
    # 如果direction=1, 则做多因子值最大的一组， 做空因子值最小的一组；如果direction=-1, 则做空因子值最大的一组， 做多因子值最小的一组
    bt_df.loc[bt_df['group'] == (group_num-1), 'weight'] = bt_df.loc[bt_df['group'] == (group_num-1), 'weight']*direction
    bt_df.loc[bt_df['group'] == 0, 'weight'] = bt_df.loc[bt_df['group'] == 0, 'weight']*(-direction)    
    ls_perf = bt_df.groupby('tradeDate').apply(lambda x: sum(x[return_name]*x['weight']))
    lo_bt_df = bt_df[bt_df['group']==(group_num-1)]
    lo_perf = lo_bt_df.groupby('tradeDate')[return_name].mean()
    so_bt_df = bt_df[bt_df['group']==0]
    so_perf = so_bt_df.groupby('tradeDate')[return_name].mean()
    all_perf_df = pd.DataFrame([ls_perf, lo_perf, so_perf]).T.sort_index()
    rtn_flag = ['ls', 'lo', 'so']    
    all_perf_cols = ['%s_period_ret'%i for i in rtn_flag]
    all_perf_df.columns = all_perf_cols
    for i in rtn_flag:
        all_perf_df['%s_cum_ret'%i] = (all_perf_df['%s_period_ret'%i]+1).cumprod()
        all_perf_df['%s_period_ret'%i] = all_perf_df['%s_period_ret'%i].shift(1)
        all_perf_df.fillna(0, inplace=True)
        all_perf_df['%s_cum_ret'%i] = all_perf_df['%s_cum_ret'%i].shift(1)
        all_perf_df.fillna(1, inplace=True) 
    return all_perf_df.reset_index()

#股票周收益率
weekly_trnval_df1 = weekly_trnval_df.copy()
weekly_trnval_df1.sort_values(['secID','tradeDate'],inplace=True)
weekly_trnval_df1['tar_rtn'] = weekly_trnval_df1.groupby('secID')['return'].shift(-1)
weekly_rtn_df = weekly_trnval_df1[['secID','tradeDate','tar_rtn']].dropna()

# 构建对冲组合，计算因子多空、多头、空头的收益率
style_factors = standard_weekly_signal_df1.columns[2:]
f, ax= plt.subplots(nrows=4, ncols=2, figsize = (20, 15))
plt.subplots_adjust(wspace=0.2, hspace=0.4)
weekly_rtn_df1 = weekly_rtn_df.copy()
perf_list =[]
for tcount, col_name in enumerate(style_factors):
    row = tcount / 2
    col = tcount - 2*(tcount/2)
    # 回测组合表现
    perf = long_short_backtest(standard_weekly_signal_df1, weekly_rtn_df1, col_name, 'tar_rtn', 1,10)
    perf = perf.set_index('tradeDate')
    perf = perf.rename(columns=dict(zip(perf.columns, perf.columns+'_%s'%col_name)))
    _ = perf[['ls_cum_ret_%s'%col_name,'lo_cum_ret_%s'%col_name,'so_cum_ret_%s'%col_name]].plot(ax=ax[row][col])
    _ = ax[row][col].set_title("style_%s"%col_name)
    perf_list.append(perf)
factor_rtn_df = pd.concat(perf_list,axis=1)  

'''
图例中ls_cum_ret 代表多空的净值曲线（做多因子值最大的一组，做空因子值最小的一组）、lo_cum_ret 代表多头的净值曲线（只做多因子值最大的一组）、so_cum_ret 代表空头净值曲线（只做多因子值最小的一组）；
曲线走势代表各个因子收益率的表现情况；

   调试 运行
文档
 代码  策略  文档
1.4 资金流偏好度计算(2分钟)

资金流偏好度是对于风格因子而言的， 指定风格的资金流偏好度为某一种风格因子的多头组合以及空头组合按照既定的资金流和既定的计算方法得到的资金的偏离度。具体计算方法如下：

在每一期按照既定的资金流分别计算风格多头组合和空头组合的资金流之和，并用多头组合的资金流之和减去空头组合的资金流之和得到当期该风格因子的资金流差；
若是按照资金在交易额中的占比计算比率，那么使用资金流差除以多头组合和空头组合两者的交易额之和得到比率；若是按照环比增长率，那么使用资金流差除以上一期的同一风格的资金流差得到风格的资金流环比增长率；
完成以上步骤后，将根据资金流的正负判断资金流入的方向并记录; 图片注释

'''

def cash_flow_preference(signal_df, factor_name, cash_df, cash_factor,  trn_df, trn_factor, group_num, direction):
    """
    计算某一风格因子上资金流的偏好度
    params:
            signal_df: DataFrame, columns=['secID', 'tradeDate', factor], 股票的因子值, factor为股票当日的因子值
            cash_df: DataFrame, columns=['secID', 'tradeDate', 'cash_value'], cash_vale代表资金流
            trn_df : DataFrame, columns=['secID', 'tradeDate', 'trn_value'], trn_value代表成交额
    return: 两个DataFrame,分别代表按照资金在交易额中的占比和资金环比增长率计算的资金流偏好度
    """
    bt_df = signal_df.merge(cash_df, on=['secID', 'tradeDate'], how='right').merge(trn_df,on=['secID', 'tradeDate'],how='left')
    
    # 分成n组, 保留因子值最大和最小的两组
    bt_df.dropna(subset=[factor_name], inplace=True)
    bt_df = signal_grouping(bt_df, factor_name=factor_name, ngrp=group_num)
    bt_df = bt_df[bt_df['group'].isin([0, group_num-1])]
    
    # 计算权重：每组等权    
    count_df = bt_df.groupby(['tradeDate', 'group']).apply(lambda x:len(x)).reset_index()
    count_df.columns=['tradeDate', 'group', 'count']
    bt_df = bt_df.merge(count_df, on=['tradeDate', 'group'])
    bt_df['weight'] = 1.0/bt_df['count']
    
    # 如果direction=1, 则做多因子值最大的一组， 做空因子值最小的一组；如果direction=-1, 则做空因子值最大的一组， 做多因子值最小的一组
    bt_df.loc[bt_df['group'] == (group_num-1), 'weight'] = bt_df.loc[bt_df['group'] == (group_num-1), 'weight']*direction
    bt_df.loc[bt_df['group'] == 0, 'weight'] = bt_df.loc[bt_df['group'] == 0, 'weight']*(-direction)
    new_col_name = '%s_%s_preference' % (cash_factor,factor_name)
    bt_df = bt_df.sort_values(['tradeDate','secID'])
    pct_perf = bt_df.groupby('tradeDate').apply(lambda x: (x[cash_factor]*x['weight']).sum() /  (x[trn_factor]*x['weight'].abs()).sum()).reset_index().rename(columns={0:'pct_%s' %new_col_name})
    qoq_perf = bt_df.groupby('tradeDate').apply(lambda x: (x[cash_factor]*x['weight']).sum() ).reset_index().rename(columns={0:'qoq_%s' %new_col_name})
    qoq_perf['qoq_%s1' %new_col_name] = qoq_perf['qoq_%s' %new_col_name].shift(1)
    qoq_perf['qoq_%s' %new_col_name] =(qoq_perf['qoq_%s' %new_col_name] - qoq_perf['qoq_%s1' %new_col_name])/qoq_perf['qoq_%s1' %new_col_name].abs()
    return pct_perf,qoq_perf[['tradeDate','qoq_%s' %new_col_name]].iloc[1:,:]


#各个资金流指标下, 资金流的偏好度
signal_df = standard_weekly_factor_df.copy()
style_factor_list = signal_df.columns[2:]
trn_df = weekly_trnval_df.copy()
trn_factor = 'turnoverValue'
group_num = 10
direction = 1
cash_df_list = [weekly_cashchg_df.copy(),weekly_finvalchg_df.copy(), weekly_hkvalchg_df.copy()]
cash_factor_list =['netMoneyInflow','finVal','partyVol']
perf_list =[]
for i in range(len(cash_df_list)):
    for f in style_factor_list:
        pct_perfi,qoq_perfi = cash_flow_preference(signal_df, f, cash_df_list[i],cash_factor_list[i],  trn_df, trn_factor, group_num,direction)
        pct_perfi.set_index('tradeDate',inplace=True)
        qoq_perfi.set_index('tradeDate',inplace=True)
        perf_list.append(pct_perfi)
        perf_list.append(qoq_perfi)
preference_df = pd.concat(perf_list,axis=1) 

#根据每一期每个风格的资金流偏好度，找到其中资金偏好度最强的风格（考虑方向），用于后续计算风格配置矩阵。
cash_style_list = ['pct_netMoneyInflow','qoq_netMoneyInflow','pct_partyVol','qoq_partyVol','pct_finVal','qoq_finVal']
style_preference_dic ={}
for cash_style_now in cash_style_list:
    style_cols = [i for i in preference_df.columns if cash_style_now in i]
    style_preference_df_tmp = preference_df[style_cols]
    style_preference_df_tmp = style_preference_df_tmp.dropna()
    style_preference_df_tmp['strongest_style'] = style_preference_df_tmp.apply(lambda x: x.abs().idxmax(),axis=1)
    style_preference_df_tmp['strongest_style_direction'] = style_preference_df_tmp.apply(lambda x: x[x['strongest_style']],axis=1)
    style_preference_df_tmp['strongest_style_direction'] =     style_preference_df_tmp['strongest_style_direction'].apply(lambda x: -1 if x<0 else 1)
    style_preference_df_tmp['strongest_style'] = style_preference_df_tmp['strongest_style'].apply(lambda x:x.replace('_preference','').split('_')[-1])
    style_preference_dic[cash_style_now] = style_preference_df_tmp
    
'''


1.5 资金流风格配置矩阵生成(1分钟)

配置矩阵的作用是对随后一期的风格因子进行择时: 根据每一期资金偏好度最强的风格，对随后一期的8种风格的多头组合对冲空头组合的平均收益率、胜率、波动率、IR等进行统计。若某一种最强风格在整个样本区间内出现次数大于或等于25次，则选出最强风格随后一期8种风格中风格IR绝对值最高的3个风格作为该最强风格的推荐配置，配置的方向将根据IR的正负符号表示，配置的权重将根据IR的大小进行归一化计算。
以净流入比率为例，风格配置矩阵如下图所示，以marketvalue为例，当资金在市值上的偏好最强且流入小市值大于大市值时（dorection=-1）,下一期推荐的风格因子为TVMA6、MktValue和NetProfitGrowRate，其中NetProfitGrowRate正向配置（short因子值小的股票，long因子值大的股票），TVMA6、MktValue反向配置（long因子值大的股票，short因子值小的股票），配置的权重根据“abs_ir”大小进行归一化计算。
图片注释

'''

#找到历史上发生情况超过特定阈值的风格及方向
def get_common_style(cash_style_now,style_preference_dic, happen_thres=8 ):
    style_preference_df = style_preference_dic[cash_style_now]
    style_count_df = style_preference_df.groupby(['strongest_style','strongest_style_direction'])['%s_ROE_preference'%cash_style_now].count().reset_index()
    common_style_df = style_count_df[style_count_df['%s_ROE_preference' %cash_style_now]>=happen_thres]
    return common_style_df

#计算最强风格因子下期各因子的表现，包括平均收益率、胜率、IR等
def get_cash_style_factor_stats(common_style_df,style_preference_df, factor_rtn_df,cash_style):    
    factor_stat_list =[]
    for _, s in common_style_df.iterrows():
        strongest_style = s['strongest_style'] 
        cash_direction = s['strongest_style_direction']
        dates = style_preference_df[(style_preference_df['strongest_style']==strongest_style)&(style_preference_df['strongest_style_direction']==cash_direction)].index
        extr_factor_rtn_df = factor_rtn_df[factor_rtn_df.index.isin(dates)].dropna()
        m = extr_factor_rtn_df.mean()
        m1 = m*52
        std = extr_factor_rtn_df.std()
        std1 = extr_factor_rtn_df.std()* np.sqrt(52)
        winper = extr_factor_rtn_df.apply(lambda x:len(x[x>0])/ float(len(x)))
        ir = m1/std1
        tmp_df = pd.DataFrame([m,std,winper,ir]).T.reset_index()
        tmp_df.columns = ['nxt_factor','mean','std','win_per','ir']
        tmp_df['nxt_factor'] = tmp_df['nxt_factor'].apply(lambda x:x.split('_',3)[-1])
        tmp_df['cash_style'] = cash_style
        tmp_df['strongest_style'] = strongest_style
        tmp_df['direction'] = cash_direction
        factor_stat_list.append(tmp_df)
    factor_stat_df = pd.concat(factor_stat_list)
    return factor_stat_df
        
#跟据最强风格因子下期各因子的表现，确定下一期表现最好的三个风格因子及配置方向
def get_optimize_style_factor(factor_stat_df, best_num):
    
    def sel_optimize_style_factor(df, best_num=3):
        df = df.copy()
        df['abs_ir'] = df['ir'].abs()
        df = df.sort_values('abs_ir', ascending=False)
        return df.iloc[:best_num,:]
    optimize_factor_df = factor_stat_df.groupby(['cash_style','strongest_style','direction'])[['nxt_factor','mean','std','win_per','ir']].apply(lambda x: sel_optimize_style_factor(x, best_num)).reset_index().drop(['level_3'],axis=1)
    return optimize_factor_df

cash_style_list = ['pct_netMoneyInflow','qoq_netMoneyInflow','pct_partyVol','qoq_partyVol','pct_finVal','qoq_finVal'] ####
style_preference_dic1 = style_preference_dic.copy()
factor_rtn_df1 = factor_rtn_df[[i for i in factor_rtn_df.columns if 'ls_period' in i]].copy()
factor_rtn_df1 = factor_rtn_df1.shift(-1)
best_num = 3
all_optimize_factor_dic = {}
for cash_style in cash_style_list:
    print (cash_style)
    happen_thres = 25
    if 'partyVol' in cash_style :
        happen_thres = 6
    common_style_df = get_common_style(cash_style, style_preference_dic1, happen_thres)
    factor_stat_df = get_cash_style_factor_stats(common_style_df,style_preference_dic1[cash_style], factor_rtn_df1,cash_style)
    optimize_factor_df = get_optimize_style_factor(factor_stat_df, best_num)
    optimize_factor_df['allot_direction'] = optimize_factor_df['ir'].apply(lambda x: -1 if x<0 else 1)
    all_optimize_factor_dic[cash_style]=optimize_factor_df
    
'''

第二部分：各个子资金流策略的表现情况及资金流策略合并后的表现情况
该部分耗时 6分钟
根据以上构造的资金流偏好度，可以基于以下的策略思想进行回测。
从个股角度上看，针对当期的大量资金地突然流入或流出，市场都会迅速将信息反馈到股市中，正如股市中常提及的“放量上涨”、“缩量下跌”等等。
从风格配置角度上看，资金流集聚在某种特定的风格或者资金流大量流入某一种风格都意味着该风格正迅速获得市场的关注并将能用于指导风格的轮转。

策略回测的基础框架
图片注释

该部分内容为：

子资金流策略的表现情况，包括净值曲线、超额收益、最大回撤、胜率
子资金流策略组合后的表现情况，包括净值曲线、因子权重、超额收益、最大回撤、胜率
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
2.1 子资金流策略的表现情况(5分钟)

分别对各个资金流策略进行回测，每个自然周末作为策略的起点，根据每个风格因子上的净流入比率判断当期资金流偏好度最高的风格因子，根据上述筛选训练机制得到的该风格因子对应的配置方案，对下一周的风格进行配置；回测时，加入指数收益(上证综指)，看策略多头对冲指数的表现情况。

'''

def cal_optimize_rtn(strongest_style,strongest_style_direction, style_sel_matrix,date,factor_rtn_df, factor_list,default_direction):
    factor_rtn_df = factor_rtn_df.copy()
    style_sel_matrix = style_sel_matrix.copy()
    sel_style_factor =style_sel_matrix[(style_sel_matrix['strongest_style']==strongest_style)&(style_sel_matrix['direction']==strongest_style_direction)][['nxt_factor','ir','abs_ir','allot_direction']]
    if len(sel_style_factor) == 0:
        ls_rtn = (factor_rtn_df[[i for i in factor_rtn_df.columns if 'ls' in i]].loc[date]*default_direction).mean()
        lo_cols = ['so_period_ret_%s'%factor_list[i] if default_direction[i]==-1 else 'lo_period_ret_%s'%factor_list[i] for i in range(len(factor_list)) ]
        so_cols = ['so_period_ret_%s'%factor_list[i] if default_direction[i]==1 else 'lo_period_ret_%s'%factor_list[i] for i in range(len(factor_list)) ]        
        lo_rtn = (factor_rtn_df.loc[date][lo_cols]).mean()       
        so_rtn = (factor_rtn_df.loc[date][so_cols]).mean()
        weight = [(1.0/len(factor_list))] * len(factor_list)
        f_buy = factor_list
        rec_flag = 0
    else:
        sel_style_factor['weight'] = sel_style_factor['abs_ir'] / sel_style_factor['abs_ir'].sum()
        sel_factors, sel_factors_dir = sel_style_factor['nxt_factor'].tolist(),sel_style_factor['allot_direction'].tolist()
        ls_cols = ['ls_period_ret_%s'%i for i in sel_factors]
        lo_cols = ['lo_period_ret_%s'%sel_factors[i] if sel_factors_dir[i]==1 else 'so_period_ret_%s'%sel_factors[i] for i in range(len(sel_factors)) ]
        so_cols = ['lo_period_ret_%s'%sel_factors[i] if sel_factors_dir[i]==-1 else 'so_period_ret_%s'%sel_factors[i] for i in range(len(sel_factors)) ]
        ls_rtn = (sel_style_factor['weight'].values * factor_rtn_df.loc[date][ls_cols].values*sel_style_factor['allot_direction'].values).sum()
        lo_rtn = (sel_style_factor['weight'].values * factor_rtn_df.loc[date][lo_cols].values).sum() 
        so_rtn = (sel_style_factor['weight'].values * factor_rtn_df.loc[date][so_cols].values).sum() 
        weight = list(sel_style_factor['weight'].values)
        f_buy = sel_factors
        rec_flag = 1
    return ls_rtn,lo_rtn,so_rtn, rec_flag,weight,f_buy

factor_rtn_df2 = factor_rtn_df.copy()
factor_rtn_df2 = factor_rtn_df2[[i for i in factor_rtn_df2.columns if 'period' in i]]
factor_rtn_df2 = factor_rtn_df2.shift(-1)

factor_list = ['ROE','NetProfitGrowRate','fshareRatio','TVMA6','REVS20','PBIndu','tclRatio','MktValue']

default_direction = [1, 1, 1, -1, -1, -1, -1, -1]
all_optimize_factor_dic1 = copy.deepcopy(all_optimize_factor_dic)
style_preference_dic1 = copy.deepcopy(style_preference_dic)

commision = 0.002  #考虑手续费

strategy_performance_dic = {}
for now_cash_style in cash_style_list:   
    t0 = time.time()
    now_preference_df = style_preference_dic1[now_cash_style][['strongest_style','strongest_style_direction']]
    now_optimize_factor_df = all_optimize_factor_dic1[now_cash_style]
    now_preference_df['%s_plan'%now_cash_style] = now_preference_df.apply(lambda x: cal_optimize_rtn(x['strongest_style'],x['strongest_style_direction'], now_optimize_factor_df,x.name,factor_rtn_df2, factor_list, default_direction), axis=1)
    now_preference_df['%s_ls_rtn'%now_cash_style] = now_preference_df['%s_plan'%now_cash_style].apply(lambda x: x[0] -commision)
    now_preference_df['%s_lo_rtn'%now_cash_style] = now_preference_df['%s_plan'%now_cash_style].apply(lambda x: x[1] -commision)
    now_preference_df['%s_so_rtn'%now_cash_style] = now_preference_df['%s_plan'%now_cash_style].apply(lambda x: x[2] -commision)
    now_preference_df['%s_rec_flag'%now_cash_style] = now_preference_df['%s_plan'%now_cash_style].apply(lambda x: x[3])
    now_preference_df['%s_ls_cum_rtn'%now_cash_style] = (now_preference_df['%s_ls_rtn'%now_cash_style]+1).cumprod()
    now_preference_df['%s_lo_cum_rtn'%now_cash_style] = (now_preference_df['%s_lo_rtn'%now_cash_style]+1).cumprod()
    now_preference_df['%s_so_cum_rtn'%now_cash_style] = (now_preference_df['%s_so_rtn'%now_cash_style]+1).cumprod()
    # 调整时间   
    now_preference_df['%s_ls_rtn'%now_cash_style] = now_preference_df['%s_ls_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(0, inplace=True)
    now_preference_df['%s_ls_cum_rtn'%now_cash_style] = now_preference_df['%s_ls_cum_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(1, inplace=True) 
    now_preference_df['%s_lo_rtn'%now_cash_style] = now_preference_df['%s_lo_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(0, inplace=True)
    now_preference_df['%s_lo_cum_rtn'%now_cash_style] = now_preference_df['%s_lo_cum_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(1, inplace=True)
    now_preference_df['%s_so_rtn'%now_cash_style] = now_preference_df['%s_so_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(0, inplace=True)
    now_preference_df['%s_so_cum_rtn'%now_cash_style] = now_preference_df['%s_so_cum_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(1, inplace=True)
    print ("%s finished, using %s seconds! " %(now_cash_style,time.time()-t0))
    strategy_performance_dic[now_cash_style] = now_preference_df

#加入指数收益，看多头对冲指数的表现情况
strategy_performance_dic1 = copy.deepcopy(strategy_performance_dic)
idx_rtn = DataAPI.MktIdxwGet(beginDate=u"20100101",endDate=u"20180801",indexID=u"000001.ZICN",ticker=u"",field=['indexID','endDate','chgPct'],pandas="1").sort_values('endDate')
cash_style_list = ['pct_netMoneyInflow','qoq_netMoneyInflow','pct_partyVol','qoq_partyVol','pct_finVal','qoq_finVal']
for i in cash_style_list:
    strategy_performance_dic1[i] = strategy_performance_dic1[i].merge(idx_rtn, left_index=True, right_on='endDate')
    strategy_performance_dic1[i]['%s_lo_relative_rtn'%i] = strategy_performance_dic1[i]['%s_lo_rtn'%i] - strategy_performance_dic1[i]['chgPct']
    strategy_performance_dic1[i]['%s_lo_relative_cum_rtn'%i] = (strategy_performance_dic1[i]['%s_lo_relative_rtn'%i]+1).cumprod()
    strategy_performance_dic1[i]['%s_lo_relative_rtn'%i] = strategy_performance_dic1[i]['%s_lo_relative_rtn'%i].shift(1)
    strategy_performance_dic1[i].fillna(0, inplace=True)
    strategy_performance_dic1[i]['%s_lo_relative_cum_rtn'%i] = strategy_performance_dic1[i]['%s_lo_relative_cum_rtn'%i].shift(1)
    strategy_performance_dic1[i].fillna(1, inplace=True) 
    strategy_performance_dic1[i]['idx_cum_rtn'] = (strategy_performance_dic1[i]['chgPct']+1).cumprod()
    strategy_performance_dic1[i]['idx_cum_rtn'] = strategy_performance_dic1[i]['idx_cum_rtn'].shift(1)
    strategy_performance_dic1[i].fillna(1, inplace=True)
    strategy_performance_dic1[i] = strategy_performance_dic1[i].set_index('endDate')
    
#各个子策略的净值曲线
f, ax= plt.subplots(nrows=3, ncols=2, figsize = (20, 15))
plt.subplots_adjust(wspace=0.1, hspace=0.4)
for tcount, col_name in enumerate(cash_style_list):
    row = tcount /2
    col = tcount - 2*(tcount/2)
    _ = strategy_performance_dic1[col_name][['%s_ls_cum_rtn'%col_name,'%s_lo_cum_rtn'%col_name,'%s_lo_relative_cum_rtn'%col_name,'idx_cum_rtn','%s_so_cum_rtn'%col_name]].plot(ax=ax[row][col])
    _ = ax[row][col].set_title(col_name)    
    
'''
    
图例中ls_cum_rtn 代表多空的净值曲线（做多因子值最大的一组，做空因子值最小的一组）、lo_cum_rtn 代表多头的净值曲线（只做多因子值最大的一组）、so_cum_rtn 代表空头净值曲线（只做多因子值最小的一组）;lo_relative_cum_rtn 代表多头对冲上证指数的净值曲线；idx_cum_rtn代表上证指数净值曲线。
从图上可以看出，6种资金流策略在多空的情况下均能取得显著的正收益；多头对冲上证指数除了沪港深资金策略外表现也比较好，但在2017年后，策略均遭遇比较明显的回测。

   调试 运行
文档
 代码  策略  文档
2.2 子资金流策略组合后的表现情况(1分钟)

在每期，我们将分别监测6个资金流偏好度当期的最强风格因子，若其有推荐的配置方案，则记录下对应的推荐风格及相应的权重，最终等权配置具有推荐配置因子的方案归一化后得到最终的综合策略配置方案。

'''

strategy_performance_dic2 = copy.deepcopy(strategy_performance_dic)
all_strategy_preference_df = pd.concat([i.iloc[:,2] for i in strategy_performance_dic2.values()],axis=1)
all_strategy_preference_df = all_strategy_preference_df[[i for i in all_strategy_preference_df.columns if 'partyVol' not in i]]#沪深港通时间区间较短，不再合并范围内
def cal_comb_rtn(s):
    s = s.dropna()
    s = s.tolist()
    m = [i[3] for i in s]
    if sum(m) == 0:
        r_ls = s[0][0]
        r_lo = s[0][1]
        r_so = s[0][2]
    else:
        ls_m = [i[0] for i in s if i[3]==1]
        lo_m = [i[1] for i in s if i[3]==1]
        so_m = [i[2] for i in s if i[3]==1]
        r_ls = sum(ls_m) / len(ls_m)
        r_lo = sum(lo_m) / len(lo_m)
        r_so = sum(so_m) / len(so_m)       
    return r_ls, r_lo, r_so
    
all_strategy_preference_df['comb_rtn'] = all_strategy_preference_df.apply(lambda x: cal_comb_rtn(x),axis=1)

all_strategy_preference_df['comb_ls_rtn'] = all_strategy_preference_df['comb_rtn'].apply(lambda x:x[0] - commision)
all_strategy_preference_df['comb_lo_rtn'] = all_strategy_preference_df['comb_rtn'].apply(lambda x:x[1] - commision)
all_strategy_preference_df['comb_so_rtn'] = all_strategy_preference_df['comb_rtn'].apply(lambda x:x[2] - commision)

all_strategy_preference_df['comb_ls_cum_rtn'] = (all_strategy_preference_df['comb_ls_rtn']+1).cumprod()
all_strategy_preference_df['comb_lo_cum_rtn'] = (all_strategy_preference_df['comb_lo_rtn']+1).cumprod()
all_strategy_preference_df['comb_so_cum_rtn'] = (all_strategy_preference_df['comb_so_rtn']+1).cumprod()

# 调整时间
all_strategy_preference_df['comb_ls_rtn'] = all_strategy_preference_df['comb_ls_rtn'].shift(1)
all_strategy_preference_df.fillna(0, inplace=True)
all_strategy_preference_df['comb_ls_cum_rtn'] = all_strategy_preference_df['comb_ls_cum_rtn'].shift(1)
all_strategy_preference_df.fillna(1, inplace=True) 

all_strategy_preference_df['comb_lo_rtn'] = all_strategy_preference_df['comb_lo_rtn'].shift(1)
all_strategy_preference_df.fillna(0, inplace=True)
all_strategy_preference_df['comb_lo_cum_rtn'] = all_strategy_preference_df['comb_lo_cum_rtn'].shift(1)
all_strategy_preference_df.fillna(1, inplace=True) 

all_strategy_preference_df['comb_so_rtn'] = all_strategy_preference_df['comb_so_rtn'].shift(1)
all_strategy_preference_df.fillna(0, inplace=True)
all_strategy_preference_df['comb_so_cum_rtn'] = all_strategy_preference_df['comb_so_cum_rtn'].shift(1)
all_strategy_preference_df.fillna(1, inplace=True) 
#加入指数收益，看多头对冲指数的表现情况
all_strategy_preference_df = all_strategy_preference_df.merge(idx_rtn, left_index=True, right_on='endDate')
all_strategy_preference_df['comb_lo_relative_rtn'] = all_strategy_preference_df['comb_lo_rtn'] - all_strategy_preference_df['chgPct']
all_strategy_preference_df['comb_lo_relative_cum_rtn'] = (all_strategy_preference_df['comb_lo_relative_rtn']+1).cumprod()
all_strategy_preference_df['comb_lo_relative_rtn'] = all_strategy_preference_df['comb_lo_relative_rtn'].shift(1)
all_strategy_preference_df.fillna(0, inplace=True)
all_strategy_preference_df['comb_lo_relative_cum_rtn'] = all_strategy_preference_df['comb_lo_relative_cum_rtn'].shift(1)
all_strategy_preference_df.fillna(1, inplace=True) 
all_strategy_preference_df['idx_cum_rtn'] = (all_strategy_preference_df['chgPct']+1).cumprod()
all_strategy_preference_df['idx_cum_rtn'] = all_strategy_preference_df['idx_cum_rtn'].shift(1)
all_strategy_preference_df.fillna(1, inplace=True)

#子策略合并后的净值曲线
f, ax= plt.subplots(nrows=1, ncols=1, figsize = (15, 6))
_= all_strategy_preference_df.set_index('endDate')[['comb_ls_cum_rtn','comb_lo_cum_rtn','comb_lo_relative_cum_rtn','idx_cum_rtn','comb_so_cum_rtn']].plot(ax=ax)


'''

可见，子策略合并后，无论是多空还是多头对冲指数，均比各个子策略单独表现的要好，说明策略合并是比较有效的。

'''

#查看组合策略各年的权重变化
plan_all_strategy_preference_df = all_strategy_preference_df.copy()
plan_all_strategy_preference_df.set_index('endDate',inplace=True)
plan_all_strategy_preference_df = plan_all_strategy_preference_df.iloc[:,:4]

#计算组合策略的权重
def cal_factor_weight(s,factor_list):
    default_weights = [1.0/len(factor_list)] * len(factor_list) 
    s = s[s!=0].dropna()
    s = s.tolist()
    all_f_weights = []
    for i in s:
        if i[3] == 0:
            f_weights = default_weights
        else:
            f_weights = [0] * len(factor_list)
            for j in range(len(i[4])):
                f_weights[factor_list.index(i[5][j])] = i[4][j]
        all_f_weights.append(f_weights)
    f_w = list(np.array(all_f_weights).mean(axis=0))
    return f_w  

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.style.use('ggplot')
#绘制权重的时间序列图
def plot_factor_coefficient(factor_weight_df):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))   
    factor_weight_df.index = pd.to_datetime(factor_weight_df.index)
    factor1 = factor_weight_df.iloc[:, :1].abs().sum(axis=1) * 100.0
    factor2 = factor_weight_df.iloc[:, :2].abs().sum(axis=1) * 100.0
    factor3 = factor_weight_df.iloc[:, :3].abs().sum(axis=1) * 100.0
    factor4 = factor_weight_df.iloc[:, :4].abs().sum(axis=1) * 100.0
    factor5 = factor_weight_df.iloc[:, :5].abs().sum(axis=1) * 100.0
    factor6 = factor_weight_df.iloc[:, :6].abs().sum(axis=1) * 100.0
    factor7 = factor_weight_df.iloc[:, :7].abs().sum(axis=1) * 100.0
    factor8 = factor_weight_df.iloc[:, :8].abs().sum(axis=1) * 100.0
    columns = factor_weight_df.columns.tolist()
    ax.plot(factor_weight_df.index, factor1, factor2, factor3, factor4, factor5, color='black')
    ax.fill_between(factor_weight_df.index.tolist(), 0, factor1.tolist(), facecolor='green', label=columns[0])
    ax.fill_between(factor_weight_df.index.tolist(), factor1.tolist(), factor2.tolist(), facecolor='red', label=columns[1])
    ax.fill_between(factor_weight_df.index.tolist(), factor2.tolist(), factor3.tolist(), facecolor='blue', label=columns[2])
    ax.fill_between(factor_weight_df.index.tolist(), factor3.tolist(), factor4.tolist(), facecolor='yellow', label=columns[3])
    ax.fill_between(factor_weight_df.index.tolist(), factor4.tolist(), factor5.tolist(), facecolor='grey', label=columns[4])
    ax.fill_between(factor_weight_df.index.tolist(), factor5.tolist(), factor6.tolist(), facecolor='coral', label=columns[5])
    ax.fill_between(factor_weight_df.index.tolist(), factor6.tolist(), factor7.tolist(), facecolor='darkorange', label=columns[6])
    ax.fill_between(factor_weight_df.index.tolist(), factor7.tolist(), factor8.tolist(), facecolor='pink', label=columns[7])
    legend = ax.legend(fontsize=12, loc='best',ncol=4)
    ax.set_xlim(factor_weight_df.index.tolist()[0], factor_weight_df.index.tolist()[-1])
    ax.set_ylim(-1, 101)
    ymajorFormatter = FormatStrFormatter('%.f%%')     
    ax.yaxis.set_major_formatter(ymajorFormatter)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Factor weights Cumsum Percent', fontsize=12)
    ax.set_title('factor weights of combined strategy', fontsize=14)

plan_all_strategy_preference_df['factor_weights'] = plan_all_strategy_preference_df.apply(lambda x:cal_factor_weight(x,factor_list),axis=1 )
factor_weight_df = pd.DataFrame(list(plan_all_strategy_preference_df['factor_weights'].values),index =plan_all_strategy_preference_df.index,columns=factor_list )
plot_factor_coefficient(factor_weight_df)

#多头对冲指数的情况下，子策略和合并后策略净值曲线比较
single_lo_relative_performance_df =pd.concat([strategy_performance_dic1[i][['%s_lo_relative_cum_rtn'%i,'%s_lo_relative_rtn'%i]] for i in cash_style_list],axis=1)
comb_lo_relative_preference_df = all_strategy_preference_df.set_index('endDate')[['comb_lo_relative_cum_rtn','comb_lo_relative_rtn']]
lo_relative_preference_df = pd.concat([single_lo_relative_performance_df,comb_lo_relative_preference_df],axis=1)

cum_col_list = [i for i in lo_relative_preference_df.columns if 'cum' in i]
f, ax= plt.subplots(nrows=1, ncols=1, figsize = (15, 6))
_= lo_relative_preference_df[cum_col_list].plot(ax=ax, color =['r','y','c','m','g','k','b'])

#统计子策略和合并后的策略的超额收益、胜率和最大回测
def excess_rtn(s):
    r = s.iloc[-1]/s.iloc[0] - 1
    return r
def winper(s):
    return (s.pct_change()>0).sum() / float(len(s))
def maxDrawdown(s):
    cum_max = s.cummax()
    maxdrawdown =((cum_max-s)/cum_max).max()
    return maxdrawdown

lo_relative_preference_df['year'] = lo_relative_preference_df.index.str.slice(0,4)
lo_relative_perf_list = []
for i in cum_col_list:
    tmp_perf_df = lo_relative_preference_df[[i,'year']].dropna().groupby('year')[i].agg([excess_rtn,winper,maxDrawdown])
    tmp_perf_df.loc['all','excess_rtn'] = excess_rtn(lo_relative_preference_df[i].dropna())
    tmp_perf_df.loc['all','winper'] = winper(lo_relative_preference_df[i].dropna())
    tmp_perf_df.loc['all','maxDrawdown'] = maxDrawdown(lo_relative_preference_df[i].dropna())
    lo_relative_perf_list.append(tmp_perf_df)
df1 = pd.concat(lo_relative_perf_list,axis=1)  
df1.columns= pd.MultiIndex.from_product([cum_col_list, ['excess_rtn','winper','maxDrawdown']])
df1=df1.apply(lambda x: np.round(x,3))
df1


'''

从净值曲线图来看， 合并策略的表现明显优于各个子策略， 不过净值在2017年有个明显下降，考虑到2017年的市场风格， 可能和资金流策略未能完全捕捉到大市值的股票有关;从以上数据也可以看到，策略合并后总体超额收益明显优于各个子策略2.521 VS (2.07、1.826、-0.239、-0.11.986、2.118)，最大回撤相比四个资金流子策略变化不大（最大回撤大于沪港深资金流策略，但沪港深资金流策略回测时间短、收益和胜率均比较低）、胜率相对于各个子策略也有一定的提高。

   调试 运行
文档
 代码  策略  文档
第三部分：采用滑动窗口的方法预测，各个资金流策略的表现情况及资金流策略合并后的表现情况
该部分耗时 4分钟
该部分内容为：

滑动预测下子资金流策略的表现情况
滑动预测下子资金流策略组合后的表现情况
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
3.1 滑动预测方式下，子资金流策略的表现情况(3分钟)

研报只做了样本内测试，本实验进一步实验滑动预测的效果，时间窗口取三年，用前三年的训练结果在第四年进行测试,由于沪深港通时间区间较短，所以这里面只对除此之外的四个资金流策略做试验


'''

step_window =3
factor_rtn_df2 = factor_rtn_df.copy()
factor_rtn_df2 = factor_rtn_df2[[i for i in factor_rtn_df2.columns if 'period' in i]]
factor_rtn_df2 = factor_rtn_df2.shift(-1)
factor_list = ['ROE','NetProfitGrowRate','fshareRatio','TVMA6','REVS20','PBIndu','tclRatio','MktValue']
cash_style_list = ['pct_netMoneyInflow','qoq_netMoneyInflow','pct_finVal','qoq_finVal']
default_direction = [1, 1, 1, -1, -1, -1, -1, -1]  #默认的买入方向
commision = 0.002  #考虑手续费
best_num = 3
happen_thres = 11
style_preference_dic1 = copy.deepcopy(style_preference_dic)

rolling_cash_style_perf_dic  ={}
for cash_style in cash_style_list:    
    min_year,max_year = int(style_preference_dic1[cash_style].index.min()[:4]),int(style_preference_dic1[cash_style].index.max()[:4])
    perf_df_list = []
    for yr in range(min_year,max_year-step_window+1):
        style_preference_dic_tmp = {}
        style_preference_dic_tmp[cash_style] = style_preference_dic1[cash_style][(style_preference_dic1[cash_style].index>='%s_01-01'%str(yr))&(style_preference_dic1[cash_style].index<='%s_12-31'%str(yr+2))]
        common_style_df = get_common_style(cash_style, style_preference_dic_tmp, happen_thres)
        factor_stat_df = get_cash_style_factor_stats(common_style_df,style_preference_dic_tmp[cash_style], factor_rtn_df1,cash_style)
        optimize_factor_df = get_optimize_style_factor(factor_stat_df, best_num)
        optimize_factor_df['allot_direction'] = optimize_factor_df['ir'].apply(lambda x: -1 if x<0 else 1)

        now_preference_df = style_preference_dic[cash_style][['strongest_style','strongest_style_direction']]
        now_preference_df = now_preference_df[(now_preference_df.index>='%s-01-01' %str(yr+step_window))&(now_preference_df.index<='%s-12-31'%str(yr+step_window))]
        now_preference_df['%s_plan'%cash_style] = now_preference_df.apply(lambda x: cal_optimize_rtn(x['strongest_style'],x['strongest_style_direction'], optimize_factor_df,x.name,factor_rtn_df2, factor_list, default_direction), axis=1)
        now_preference_df['%s_ls_rtn'%cash_style] = now_preference_df['%s_plan'%cash_style].apply(lambda x: x[0] -commision)
        now_preference_df['%s_lo_rtn'%cash_style] = now_preference_df['%s_plan'%cash_style].apply(lambda x: x[1] -commision)
        now_preference_df['%s_so_rtn'%cash_style] = now_preference_df['%s_plan'%cash_style].apply(lambda x: x[2] -commision)
        now_preference_df['%s_rec_flag'%cash_style] = now_preference_df['%s_plan'%cash_style].apply(lambda x: x[3])
        perf_df_list.append(now_preference_df)
    cash_style_perf_df = pd.concat(perf_df_list,axis=0)
    rolling_cash_style_perf_dic[cash_style] = cash_style_perf_df
    
#各子策略的表现情况
t0 = time.time()
rolling_strategy_performance_dic ={}
rolling_cash_style_perf_dic1 = copy.deepcopy(rolling_cash_style_perf_dic)
for now_cash_style in rolling_cash_style_perf_dic1.keys():
    now_preference_df = rolling_cash_style_perf_dic1[now_cash_style]
    now_preference_df['%s_ls_cum_rtn'%now_cash_style] = (now_preference_df['%s_ls_rtn'%now_cash_style]+1).cumprod()
    now_preference_df['%s_lo_cum_rtn'%now_cash_style] = (now_preference_df['%s_lo_rtn'%now_cash_style]+1).cumprod()
    now_preference_df['%s_so_cum_rtn'%now_cash_style] = (now_preference_df['%s_so_rtn'%now_cash_style]+1).cumprod()
    # 调整时间   
    now_preference_df['%s_ls_rtn'%now_cash_style] = now_preference_df['%s_ls_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(0, inplace=True)
    now_preference_df['%s_ls_cum_rtn'%now_cash_style] = now_preference_df['%s_ls_cum_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(1, inplace=True) 
    now_preference_df['%s_lo_rtn'%now_cash_style] = now_preference_df['%s_lo_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(0, inplace=True)
    now_preference_df['%s_lo_cum_rtn'%now_cash_style] = now_preference_df['%s_lo_cum_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(1, inplace=True)
    now_preference_df['%s_so_rtn'%now_cash_style] = now_preference_df['%s_so_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(0, inplace=True)
    now_preference_df['%s_so_cum_rtn'%now_cash_style] = now_preference_df['%s_so_cum_rtn'%now_cash_style].shift(1)
    now_preference_df.fillna(1, inplace=True)
    print ("%s finished, using %s seconds! " %(now_cash_style,time.time()-t0))
    rolling_strategy_performance_dic[now_cash_style] = now_preference_df

#加入指数收益，看多头对冲指数的表现情况
rolling_strategy_performance_dic1 = copy.deepcopy(rolling_strategy_performance_dic)
rolling_cash_style_list = ['pct_netMoneyInflow','qoq_netMoneyInflow','pct_finVal','qoq_finVal']
for i in rolling_cash_style_list:
    rolling_strategy_performance_dic1[i] = rolling_strategy_performance_dic1[i].merge(idx_rtn, left_index=True, right_on='endDate')
    rolling_strategy_performance_dic1[i]['%s_lo_relative_rtn'%i] = rolling_strategy_performance_dic1[i]['%s_lo_rtn'%i] - rolling_strategy_performance_dic1[i]['chgPct']
    rolling_strategy_performance_dic1[i]['%s_lo_relative_cum_rtn'%i] = (rolling_strategy_performance_dic1[i]['%s_lo_relative_rtn'%i]+1).cumprod()
    rolling_strategy_performance_dic1[i]['%s_lo_relative_rtn'%i] = rolling_strategy_performance_dic1[i]['%s_lo_relative_rtn'%i].shift(1)
    rolling_strategy_performance_dic1[i].fillna(0, inplace=True)
    rolling_strategy_performance_dic1[i]['%s_lo_relative_cum_rtn'%i] = rolling_strategy_performance_dic1[i]['%s_lo_relative_cum_rtn'%i].shift(1)
    rolling_strategy_performance_dic1[i].fillna(1, inplace=True) 
    rolling_strategy_performance_dic1[i]['idx_cum_rtn'] = (rolling_strategy_performance_dic1[i]['chgPct']+1).cumprod()
    rolling_strategy_performance_dic1[i]['idx_cum_rtn'] = rolling_strategy_performance_dic1[i]['idx_cum_rtn'].shift(1)
    rolling_strategy_performance_dic1[i].fillna(1, inplace=True)
    rolling_strategy_performance_dic1[i] = rolling_strategy_performance_dic1[i].set_index('endDate')


#滑动预测下, 各个子策略的净值曲线
f, ax= plt.subplots(nrows=2, ncols=2, figsize = (25, 10))
plt.subplots_adjust(wspace=0.1, hspace=0.4)
for tcount, col_name in enumerate(rolling_cash_style_list):
    row = tcount /2
    col = tcount - 2*(tcount/2)
    _ = rolling_strategy_performance_dic1[col_name][['%s_ls_cum_rtn'%col_name,'%s_lo_cum_rtn'%col_name,'%s_lo_relative_cum_rtn'%col_name,'idx_cum_rtn','%s_so_cum_rtn'%col_name]].plot(ax=ax[row][col])
    _ = ax[row][col].set_title(col_name)

'''
从结果上来看， 四个资金流策略在多空、多头对冲指数的情况下均有明显的正收益，四个子策略中，融资余额交易额占比（pct_finVal）表现最好。

   调试 运行
文档
 代码  策略  文档
3.2 滑动预测方式下，子资金流策略合并后的表现情况(1分钟)    
'''

rolling_strategy_performance_dic2 = copy.deepcopy(rolling_strategy_performance_dic)
rolling_all_strategy_preference_df = pd.concat([i.iloc[:,2] for i in rolling_strategy_performance_dic2.values()],axis=1)
rolling_all_strategy_preference_df = rolling_all_strategy_preference_df[[i for i in rolling_all_strategy_preference_df.columns if 'partyVol' not in i]]

rolling_all_strategy_preference_df['comb_rtn'] = rolling_all_strategy_preference_df.apply(lambda x: cal_comb_rtn(x),axis=1)

rolling_all_strategy_preference_df['comb_ls_rtn'] = rolling_all_strategy_preference_df['comb_rtn'].apply(lambda x:x[0] - commision)
rolling_all_strategy_preference_df['comb_lo_rtn'] = rolling_all_strategy_preference_df['comb_rtn'].apply(lambda x:x[1] - commision)
rolling_all_strategy_preference_df['comb_so_rtn'] = rolling_all_strategy_preference_df['comb_rtn'].apply(lambda x:x[2] - commision)

rolling_all_strategy_preference_df['comb_ls_cum_rtn'] = (rolling_all_strategy_preference_df['comb_ls_rtn']+1).cumprod()
rolling_all_strategy_preference_df['comb_lo_cum_rtn'] = (rolling_all_strategy_preference_df['comb_lo_rtn']+1).cumprod()
rolling_all_strategy_preference_df['comb_so_cum_rtn'] = (rolling_all_strategy_preference_df['comb_so_rtn']+1).cumprod()

# 调整时间
rolling_all_strategy_preference_df['comb_ls_rtn'] = rolling_all_strategy_preference_df['comb_ls_rtn'].shift(1)
rolling_all_strategy_preference_df.fillna(0, inplace=True)
rolling_all_strategy_preference_df['comb_ls_cum_rtn'] = rolling_all_strategy_preference_df['comb_ls_cum_rtn'].shift(1)
rolling_all_strategy_preference_df.fillna(1, inplace=True) 

rolling_all_strategy_preference_df['comb_lo_rtn'] = rolling_all_strategy_preference_df['comb_lo_rtn'].shift(1)
rolling_all_strategy_preference_df.fillna(0, inplace=True)
rolling_all_strategy_preference_df['comb_lo_cum_rtn'] = rolling_all_strategy_preference_df['comb_lo_cum_rtn'].shift(1)
rolling_all_strategy_preference_df.fillna(1, inplace=True) 

rolling_all_strategy_preference_df['comb_so_rtn'] = rolling_all_strategy_preference_df['comb_so_rtn'].shift(1)
rolling_all_strategy_preference_df.fillna(0, inplace=True)
rolling_all_strategy_preference_df['comb_so_cum_rtn'] = rolling_all_strategy_preference_df['comb_so_cum_rtn'].shift(1)
rolling_all_strategy_preference_df.fillna(1, inplace=True) 
#加入指数收益，看多头对冲指数的表现情况
rolling_all_strategy_preference_df = rolling_all_strategy_preference_df.merge(idx_rtn, left_index=True, right_on='endDate')
rolling_all_strategy_preference_df['comb_lo_relative_rtn'] = rolling_all_strategy_preference_df['comb_lo_rtn'] - all_strategy_preference_df['chgPct']
rolling_all_strategy_preference_df['comb_lo_relative_cum_rtn'] = (rolling_all_strategy_preference_df['comb_lo_relative_rtn']+1).cumprod()
rolling_all_strategy_preference_df['comb_lo_relative_rtn'] = rolling_all_strategy_preference_df['comb_lo_relative_rtn'].shift(1)
rolling_all_strategy_preference_df.fillna(0, inplace=True)
rolling_all_strategy_preference_df['comb_lo_relative_cum_rtn'] = rolling_all_strategy_preference_df['comb_lo_relative_cum_rtn'].shift(1)
rolling_all_strategy_preference_df.fillna(1, inplace=True) 
rolling_all_strategy_preference_df['idx_cum_rtn'] = (rolling_all_strategy_preference_df['chgPct']+1).cumprod()
rolling_all_strategy_preference_df['idx_cum_rtn'] = rolling_all_strategy_preference_df['idx_cum_rtn'].shift(1)
rolling_all_strategy_preference_df.fillna(1, inplace=True)


#滑动预测下，子策略合并后的净值曲线
f, ax= plt.subplots(nrows=1, ncols=1, figsize = (15, 6))
_= rolling_all_strategy_preference_df.set_index('endDate')[['comb_ls_cum_rtn','comb_lo_cum_rtn','comb_lo_relative_cum_rtn','idx_cum_rtn','comb_so_cum_rtn']].plot(ax=ax)

#查看组合策略各年的权重变化
rolling_plan_all_strategy_preference_df = rolling_all_strategy_preference_df.copy()
rolling_plan_all_strategy_preference_df.set_index('endDate',inplace=True)
rolling_plan_all_strategy_preference_df = rolling_plan_all_strategy_preference_df.iloc[:,:4]

rolling_plan_all_strategy_preference_df['factor_weights'] = rolling_plan_all_strategy_preference_df.apply(lambda x:cal_factor_weight(x,factor_list),axis=1 )
rolling_factor_weight_df = pd.DataFrame(list(rolling_plan_all_strategy_preference_df['factor_weights'].values),index =rolling_plan_all_strategy_preference_df.index,columns=factor_list )
rolling_factor_weight_df.index = pd.to_datetime(rolling_factor_weight_df.index).date
plot_factor_coefficient(rolling_factor_weight_df)

#滑动预测，多头对冲指数的情况下，子策略和合并后策略净值曲线比较
rolling_single_lo_relative_performance_df =pd.concat([rolling_strategy_performance_dic1[i][['%s_lo_relative_cum_rtn'%i,'%s_lo_relative_rtn'%i]] for i in rolling_cash_style_list],axis=1)
rolling_comb_lo_relative_preference_df = rolling_all_strategy_preference_df.set_index('endDate')[['comb_lo_relative_cum_rtn','comb_lo_relative_rtn']]
rolling_lo_relative_preference_df = pd.concat([rolling_single_lo_relative_performance_df,rolling_comb_lo_relative_preference_df],axis=1)

rolling_cum_col_list = [i for i in rolling_lo_relative_preference_df.columns if 'cum' in i]
f, ax= plt.subplots(nrows=1, ncols=1, figsize = (15, 6))
_= rolling_lo_relative_preference_df[rolling_cum_col_list].plot(ax=ax)

#统计子策略和合并后的策略的超额收益、胜率和最大回测
rolling_lo_relative_preference_df['year'] = rolling_lo_relative_preference_df.index.str.slice(0,4)
rolling_lo_relative_perf_list = []
for i in rolling_cum_col_list:
    tmp_perf_df = rolling_lo_relative_preference_df[[i,'year']].dropna().groupby('year')[i].agg([excess_rtn,winper,maxDrawdown])
    tmp_perf_df.loc['all','excess_rtn'] = excess_rtn(rolling_lo_relative_preference_df[i].dropna())
    tmp_perf_df.loc['all','winper'] = winper(rolling_lo_relative_preference_df[i].dropna())
    tmp_perf_df.loc['all','maxDrawdown'] = maxDrawdown(rolling_lo_relative_preference_df[i].dropna())
    rolling_lo_relative_perf_list.append(tmp_perf_df)
df2 = pd.concat(rolling_lo_relative_perf_list,axis=1)  
df2.columns= pd.MultiIndex.from_product([rolling_cum_col_list, ['excess_rtn','winper','maxDrawdown']])
df2=df2.apply(lambda x: np.round(x,3))
df2

