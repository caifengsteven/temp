# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:07:35 2021

@author: ASUS
"""

'''
导读
A. 研究目的：本文利用通联的财务数据、分析师数据等，参考天风证券《如何获取季度盈利预测》（原作者：吴先兴）中的研究方法，对如何得到季度盈利预测数据进行了研究，并通过构造量化因子验证季度盈利预测数据的使用效果。

B. 文章结构：本文共分为4个部分，具体如下

一、背景介绍：论证季度盈利数据的作用及得到季度盈利数据预测值的必要性

二、如何计算季度盈利预测数据：实现了利用业绩预告、2种规律外推得到季度预测、分析师预测季度拆解、简约外推共5种方法

三、预测效果分析：从季度预测误差、因子表现效果两个维度对比了不同季度盈利预测方法的好坏

四、总结

C. 研究结论：
5种得到季度预测的方法中：

覆盖度进行比较，
很显然是simple线性外推方法得到的覆盖度最高，平均在90%左右，其次是equalgrowth和equalratio方法得到的因子，覆盖度在80-90%左右，再是分析师季度预测因子，覆盖度在50-60%左右，最后是业绩预告因子；
从因子效果来看，
相对来说，业绩预告数据得到的因子效果最好，EP1Q因子IC为3.87%，EPTTM因子IC为3.33%，Grh1Q因子IC为1.56%， GrhTTM因子IC为0.34%，但覆盖度明显不足以用在量化策略上，需要叠加其它数据使用；
equalgrowth和equalratio因子在EP类因子表现其次，而分析师因子和线性外推因子在Grh因子表现其次
将这5中方法得到的季度预测数据合成后得到的预测数据，
从覆盖度来看，是覆盖度最高的；
从分季度的误差来看，除了比业绩预告的误差低，相比其它几种方法，预测误差最小；
从因子效果来看，在4个因子测试中，排名3左右，综合性能优异
D. 时间说明

本文主要分为四个部分，第一部分约耗时10分钟，第二部分耗时15分钟，第三部分耗时5分钟，总耗时在30分钟左右
特别说明
为便于阅读，本文将部分和文章主题无关的函数放在函数库里面：
https://uqer.datayes.com/community/share/9sM4tSmTlqlcS7KLKSfnADGEt880/private；密码：7137
请在运行之前，克隆上面的代码，并存成lib（右上角->另存为lib,不要修改名字）

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


'''


import pandas as pd
import numpy as np
import lib.quant_util as qutil
import time
import os
import matplotlib.pyplot as plt
from CAL.PyCAL import *

data_save_dir = './report_quarter_forecast'

if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
    
    
'''

第一部分：背景介绍和研究目标论证
该部分耗时 10分钟
根据DCF模型，股价是公司未来价值的现值，因此提前预知公司的基本面信息（如盈利）可以在当下更合理的给公司进行定价。但是，从理论上来说，更合理的定价不一定可以获得更好的二级市场收益率，当投资者的认知远远超过市场认知时，也可能长时间处于市场“不合理定价”而导致的亏损局面。因此，对于公司的未来预判不需要看的很远，在适度超过市场认知的时间窗口下，投资者可以获得因自己的“超额认知”而带来的超额收益。
本节选取未来1个季度和未来1年这两个尺度来论证，提前获得公司盈利信息是否可以获得显著的超额收益，以及提前获取季度盈利相比于提前获取年度盈利是否有额外的信息增益
该部分内容为：

提前得知季度和年度盈利数据是否能带来超额收益
季度盈利数据相比年度盈利数据是否有额外的增量信息
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)




1.1 原始财务数据获取
财务数据会存在发布后进行修正、因为重组等原因进行更新等情况，在本节中，为了简化处理，不考虑财务数据在首次发布后的修正、更新等情况，这类特殊样本相对数量比较少，不会影响分析结果

本节得到的数据格式如下，其中重要的字段包括，ticker, publishDate(发布日期), Q_ANttrP(单季度归母净利润), Q_flag(季度标识)
图片注释


'''

tstart = time.time()
begin_date = '20090331' # 取财务数据的起始时间
end_date = '20210331'  #  取财务数据的结束时间
# 因子的取数据起始和结束时间
factor_sdate = '20090101'
factor_edate = '20210501'
# 因子测试的起始时间和结束时间
factor_data_sdate = '20100501'
factor_data_edate = '20210501'

stock_info_df = DataAPI.EquGet(equTypeCD=u"A",listStatusCD=u"",exchangeCD="",ListSectorCD=u"",field=u"ticker,listDate",pandas="1")
all_stock_list = stock_info_df['ticker'].tolist()

# 各个股票的上市、退市日期
delist_df = DataAPI.EquGet(ticker=all_stock_list,equTypeCD=u"A",field=u"ticker,delistDate,listDate",pandas="1")
delist_df['delistDate'].fillna("2099-01-01", inplace=True)
delist_df.head()


earning_df = DataAPI.FdmtISQPIT2018Get(ticker=all_stock_list,endDate=end_date,beginDate=begin_date,
                          field=u"ticker,publishDate,endDate,endDateRep,mergedFlag,nIncomeAttrP",pandas="1")
earning_df = earning_df.query("mergedFlag=='1'") #取合并财务报表数据

# 对于公告重复发多次，虽然有数据修正，但都以第一次发的为准
earning_df = earning_df.sort_values(by=['ticker', 'publishDate', 'endDate'], ascending=True)
earning_df = earning_df.drop_duplicates(subset=['ticker','endDate'], keep='first')

# 重命名，打上季度标签Q_flag, 如2010Q1
earning_df = earning_df.rename(columns={"nIncomeAttrP":"Q_NAttrP"})
earning_df['Q_flag'] = earning_df['endDate'].apply(lambda x: "%sQ%s"%(x[:4], int(x[5:7])/3))

# 剔除掉上市之前的数据（有些在新三板上市，发了财报，再转板到A股，也会有早期的财务数据）以及退市之后的数据（A股转到新三板）
earning_df = earning_df.merge(delist_df, on=['ticker'],how='inner')
earning_df = earning_df.query("publishDate>=listDate").query("publishDate<=delistDate")
del earning_df['listDate']
del earning_df['delistDate']

print '单季度财务数据样式如下:'
print earning_df.head().to_html()


accum_earning_df = DataAPI.FdmtISGet(ticker=all_stock_list,endDate=end_date,beginDate=begin_date,
                          field=u"ticker,publishDate,endDate,endDateRep,mergedFlag,reportType,NIncomeAttrP",pandas="1")

accum_earning_df = accum_earning_df.query("mergedFlag=='1'") #取合并财务报表数据

# 对于公告重复发多次，虽然有数据修正，但都以第一次发的为准
accum_earning_df = accum_earning_df.sort_values(by=['ticker', 'publishDate', 'endDate'], ascending=True)
accum_earning_df = accum_earning_df.drop_duplicates(subset=['ticker','endDate'], keep='first')
# 此处的Q_flag并非单季度的财报，而是标记截至的季度，如2009Q3代表财报截至2009Q3
accum_earning_df['Q_flag'] = accum_earning_df['endDate'].apply(lambda x: "%sQ%s"%(x[:4], int(x[5:7])/3))

# 剔除掉上市之前的数据（有些在新三板上市，发了财报，再转板到A股，也会有早期的财务数据）以及退市之后的数据（A股转到新三板）
accum_earning_df = accum_earning_df.merge(delist_df, on=['ticker'],how='inner')
accum_earning_df = accum_earning_df.query("publishDate>=listDate").query("publishDate<=delistDate")
del accum_earning_df['listDate']
del accum_earning_df['delistDate']

print '累计财务数据样式如下:'
print accum_earning_df.head().to_html()


# 把年度和季度的财务值对齐到一起
quarter_df = earning_df[['ticker','publishDate','endDate','endDateRep','Q_NAttrP','Q_flag']]
quarter_df.columns = ['ticker','Q_pub_date','Q_endDate','Q_endRep','Q_NAttrP','Q_flag']

year_df = accum_earning_df[['ticker','publishDate','endDate','endDateRep','NIncomeAttrP','Q_flag','reportType']]

finan_df = year_df.merge(quarter_df, on=['ticker','Q_flag'],how='outer')

# 不相等的有300多条，背后都有一些特殊原因，因为数量少，直接过滤掉
finan_df = finan_df.query("publishDate==Q_pub_date")

finan_df = finan_df[['ticker','publishDate','endDate','endDateRep','NIncomeAttrP','reportType','Q_NAttrP','Q_flag']]

print '合并累计财务值和单季度财务值的数据样式如下：'
print finan_df.head().to_html()

tend = time.time()
print '该部分耗时:%s seconds'%(round(tend-tstart, 2))


'''

1.2 构造提前预知的数据
按照PIT的原则，根据如下规则构造提前预知盈利的数据结构

图片注释


'''

# 季度数据对齐
Q_future_df = earning_df[['ticker','publishDate','Q_flag','Q_NAttrP']]

# 得到下一个季度的季度标识
def get_next_quarter(x):
    '''
    x: 如2019Q4
    返回：2020Q1
    '''
    if x[-1] == '4':
        return "%sQ1"%(int(x[:4])+1)
    else:
        return "%sQ%s"%(x[:4], int(x[-1])+1)
    
    
Q_future_df['Next_Q'] = Q_future_df['Q_flag'].apply(lambda x: get_next_quarter(x))
Q_future_df = Q_future_df.merge(Q_future_df[['ticker','Q_flag', 'Q_NAttrP']].rename(columns={"Q_flag":"Next_Q", "Q_NAttrP":"Next_Q_NAttrP"}), on=['ticker','Next_Q'], how='left')
print "季度数据对齐后的数据如下:"
print Q_future_df.head().to_html()


# 年度数据对齐
Y_future_df = accum_earning_df.query("reportType=='A'")[['ticker','publishDate','endDate','NIncomeAttrP']]
Y_future_df['Next_Y'] = Y_future_df['endDate'].apply(lambda x: "%s-12-31"%(int(x[:4])+1))
Y_future_df = Y_future_df.merge(Y_future_df[['ticker','endDate','NIncomeAttrP']].rename(columns={"endDate":"Next_Y", "NIncomeAttrP":"Next_Y_NAttrP"}), on=['ticker','Next_Y'], how='left')
print '年度数据对齐后的数据如下:'
print Y_future_df.head().to_html()

'''
1.3 利用提前预知的数据，计算因子(月度)
参考天风证券的做法，计算提前预知未来季度和年度的EP因子，为了进一步比较，也计算了实际的季度和年度的EP因子

其中季度因子的定义为：

futureQep=未来1个季度的归母净利润当前的流通市值

currentQep=最近1个季度的归母净利润当前的流通市值
年度因子的定义为：

futureYep=未来1年的归母净利润当前的流通市值

currentYep=最近1年的归母净利润当前的流通市值

'''

tstart = time.time()
# 月度的交易日历
trade_calendar_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=factor_sdate,endDate=factor_edate,field=u"calendarDate,isMonthEnd",pandas="1").query("isMonthEnd==1")

date_list = trade_calendar_df['calendarDate'].unique().tolist()

# 获取市值数据
cap_df_list = []
for tdate in date_list:
    tmp_df = DataAPI.MktEqudGet(ticker=all_stock_list,tradeDate=tdate,field=u"ticker,tradeDate,negMarketValue",pandas="1")
    cap_df_list.append(tmp_df)
cap_df = pd.concat(cap_df_list,axis=0)

quarter_df = cap_df.copy()
quarter_df['monthEnd_flag'] = 1

Q_future_df = Q_future_df.sort_values(by=['ticker','publishDate','Q_flag'], ascending=False)
# 合并PIT的财务数据时间序列数据和定期的因子模板数据
quarter_df = quarter_df.merge(Q_future_df.drop_duplicates(subset=['ticker','publishDate'], keep='first').replace(np.nan, 'NULL').rename(columns={"publishDate":"tradeDate"}), on=['ticker','tradeDate'], how='outer')
quarter_df = quarter_df.sort_values(by=['ticker','tradeDate'], ascending=True)
quarter_df['Q_NAttrP'] = quarter_df.groupby(['ticker'])['Q_NAttrP'].fillna(method='ffill').values
quarter_df['Next_Q_NAttrP'] = quarter_df.groupby(['ticker'])['Next_Q_NAttrP'].fillna(method='ffill').values
quarter_df = quarter_df.query("monthEnd_flag==1").replace("NULL", np.nan)


# futureQep因子
futureQep_df = quarter_df[['ticker','tradeDate','negMarketValue','Next_Q_NAttrP']]
futureQep_df['futureQep'] = futureQep_df['Next_Q_NAttrP']/futureQep_df['negMarketValue']

# currentQep因子
currentQep_df = quarter_df[['ticker','tradeDate','negMarketValue','Q_NAttrP']]
currentQep_df['currentQep'] = currentQep_df['Q_NAttrP']/currentQep_df['negMarketValue']


year_df = cap_df.copy()
year_df['monthEnd_flag'] = 1

# 合并PIT的财务数据时间序列数据和定期的因子模板数据
year_df = year_df.merge(Y_future_df.replace(np.nan, 'NULL').rename(columns={"publishDate":"tradeDate"}), on=['ticker','tradeDate'], how='outer')
year_df = year_df.sort_values(by=['ticker','tradeDate'], ascending=True)
year_df['NIncomeAttrP'] = year_df.groupby(['ticker'])['NIncomeAttrP'].fillna(method='ffill').values
year_df['Next_Y_NAttrP'] = year_df.groupby(['ticker'])['Next_Y_NAttrP'].fillna(method='ffill').values
year_df = year_df.query("monthEnd_flag==1").replace("NULL", np.nan)


# futureYep因子
futureYep_df = year_df[['ticker','tradeDate','negMarketValue','Next_Y_NAttrP']]
futureYep_df['futureYep'] = futureYep_df['Next_Y_NAttrP']/futureYep_df['negMarketValue']

# currentYep因子
currentYep_df = year_df[['ticker','tradeDate','negMarketValue','NIncomeAttrP']]
currentYep_df['currentYep'] = currentYep_df['NIncomeAttrP']/currentYep_df['negMarketValue']


# 将所有因子对齐合并
factor_df = futureQep_df[['ticker','tradeDate','futureQep']].merge(currentQep_df[['ticker','tradeDate','currentQep']], on=['ticker','tradeDate'], how='outer')
factor_df = factor_df.merge(futureYep_df[['ticker','tradeDate','futureYep']], on=['ticker','tradeDate'], how='outer')
factor_df = factor_df.merge(currentYep_df[['ticker','tradeDate','currentYep']], on=['ticker','tradeDate'], how='outer')

factor_df['tradeDate'] = factor_df['tradeDate'].apply(lambda x:x.replace("-", ""))
factor_df = factor_df.query("tradeDate>=@factor_data_sdate").query("tradeDate<=@factor_data_edate")

factor_df.head()

tend = time.time()
print u'该部分耗时:%s seconds'%(round(tend-tstart, 2))


'''

1.4 测试因子的表现

'''

print"该部分进行回测相关的数据准备..."
start_time = time.time()

sdate = factor_sdate
edate = factor_edate


# 全A投资域
a_universe_list = DataAPI.EquGet(equTypeCD=u"A",field=u"secID",pandas="1")['secID'].tolist()
a_universe_list.remove('DY600018.XSHG')

# 交易日历
cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=sdate, endDate=edate).sort('calendarDate')
cal_dates_df['calendarDate'] = cal_dates_df['calendarDate'].apply(lambda x: x.replace('-', ''))
cal_dates_df['prevTradeDate'] = cal_dates_df['prevTradeDate'].apply(lambda x: x.replace('-', ''))
month_end_list = cal_dates_df[cal_dates_df['isMonthEnd']==1]['calendarDate'].values


# 股票池筛选：上市不满60个交易日的次新股、st股、停牌个股
if not os.path.exists(os.path.join(data_save_dir, 'forbidden.pkl')):
    forbidden_pool = qutil.stock_special_tag(sdate, edate, pre_new_length=60)
    forbidden_pool = forbidden_pool.merge(cal_dates_df, left_on=['tradeDate'], right_on=['calendarDate'])
    forbidden_pool = forbidden_pool[['ticker', 'tradeDate', 'prevTradeDate', 'special_flag']]
    forbidden_pool.to_pickle(os.path.join(data_save_dir, 'forbidden.pkl'))
else:
    forbidden_pool = pd.read_pickle(os.path.join(data_save_dir, 'forbidden.pkl'))
print "禁止股票池:", forbidden_pool.head().to_html()

# 获取个股月度收益率
if not os.path.exists(os.path.join(data_save_dir, 'mret.pkl')):
    mret_df = DataAPI.MktEqumAdjGet(beginDate=sdate, endDate=edate, secID=a_universe_list, field=u"ticker,endDate,chgPct", pandas="1")
    mret_df.rename(columns={'endDate':'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
    mret_df['tradeDate'] = mret_df['tradeDate'].apply(lambda x: x.replace('-', ''))
    mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    mret_df['nxt1m_ret'] = mret_df.groupby('ticker')['curr_ret'].shift(-1)
    mret_df.to_pickle(os.path.join(data_save_dir, 'mret.pkl'))
else:
    mret_df = pd.read_pickle(os.path.join(data_save_dir, 'mret.pkl'))
print "个股月度收益率:", mret_df.head().to_html()

#在月行情数据中剔除个股(ST股票、次新股、停牌股票)
mret_df_real = mret_df.merge(forbidden_pool[['ticker', 'prevTradeDate', 'special_flag']], left_on=['ticker',
'tradeDate'], right_on=['ticker', 'prevTradeDate'], how='left')
mret_df_real = mret_df_real[mret_df_real['special_flag'].isnull()]
mret_df_real = mret_df_real.drop(['prevTradeDate', 'special_flag'], axis=1)

# 沪深300、中证500和中证800指数成分股

idx_dict = {300:"000300", 500:"000905", 800:"000906"}
idx_stock_dict = {}
for idx in idx_dict:
    df_tmp = DataAPI.mIdxCloseWeightGet(secID=u"", ticker=idx_dict[idx], beginDate=sdate, endDate=edate,
                                        field=u"consTickerSymbol,effDate", pandas="1")
    df_tmp.rename(columns={'consTickerSymbol':'ticker', 'effDate':'tradeDate'}, inplace=True)
    df_tmp['tradeDate'] = df_tmp['tradeDate'].apply(lambda x: x.replace('-', ''))
    idx_stock_dict[idx] = df_tmp.copy()
df_hs300 = idx_stock_dict[300].copy()
df_zz500 = idx_stock_dict[500].copy()
df_zz800 = idx_stock_dict[800].copy()


print "沪深300指数成分股:", df_hs300.head().to_html()
print "中证500指数成分股:", df_zz500.head().to_html()
print "中证800指数成分股:", df_zz800.head().to_html()

print "耗时: %s seconds" % (time.time() - start_time)


#IC测试相关函数
def proc_float_scale(df, col_name, format_str):
    """
    函数：格式化输出
    参数：
        df：DataFrame，需要格式化的数据
        col_name：list，需要格式化的列名
        format_str：格式类型
    """
    for col in col_name:
        for index in df.index:
            df.ix[index, col] = format(df.ix[index, col], format_str)
    return df
def fac_process(factor_df, factor_list, forbidden_pool, neutral):
    """
    因子处理函数  
    neutral为中性化选项：0-不进行中性化，1-行业市值中性化，2-行业和Barra风格中性化
    """
    # 剔除调仓当日的停牌、次新股、st
    factor_df = factor_df.merge(forbidden_pool[['ticker', 'prevTradeDate', 'special_flag']], left_on=['ticker', 'tradeDate'], right_on=['ticker', 'prevTradeDate'], how='left')
    factor_df = factor_df[factor_df['special_flag'].isnull()]
    factor_df = factor_df.drop(['prevTradeDate', 'special_flag'], axis=1)
    # 去极值
    w_factor_df = qutil.mad_winsorize(factor_df, factor_list, sigma_n=3)
    # 行业市值中性化
    if neutral==1:
        n_factor_df = qutil.neutralize_dframe(w_factor_df.copy(), factor_list, exclude_style=['BETA', 'RESVOL', 'MOMENTUM', 'SIZENL', 'EARNYILD', 'BTOP', 'GROWTH','LEVERAGE', 'LIQUIDTY'])
    elif neutral==2:
        n_factor_df = qutil.neutralize_dframe(w_factor_df.copy(), factor_list, exclude_style=[])
    elif neutral==0:
        n_factor_df = w_factor_df
    else:
        print "neutral参数错误！"
    # 标准化
    s_factor_df = n_factor_df.copy()
    s_factor_df[factor_list] = s_factor_df.groupby('tradeDate')[factor_list].apply(lambda df: (df-df.mean())/df.std())
    return s_factor_df
def ic_calc(factor_df, mret_df, factor_list, ret_col_name, ic_type):
    """
    函数：计算IC
    参数:
        factor_df: DataFrame，因子值，columns为因子名
        mret_df: DataFrame，个股月度收益率
        ret_col_name：下月收益率在mret_df中的列名
        factor_list: list，因子列表     
        ic_type：指定计算normal IC/Rank IC
    返回:
        DataFrame，IC序列，index为tradeDate  
    """
    ic_df = qutil.calc_ic(factor_df, mret_df, factor_list, return_col_name=ret_col_name, ic_type=ic_type)
    ic_df.set_index(['tradeDate'], inplace=True)
    return ic_df

def ic_describe(ic_df):
    """
    函数：统计IC的均值、标准差、ICIR、IC大于0的比例
    参数：ic_df: Series，各期IC序列值
    返回：DataFrame，IC统计结果，columns为IC序列均值、ICIR
    """
    ic_mean = ic_df.mean()
    ic_std = ic_df.std()
    icir = ic_mean/ic_std*np.sqrt(12)
    ic_anlyst = pd.concat([ic_mean, icir], axis=1) 
    ic_anlyst.columns = ['IC序列均值', 'ICIR']
    ic_anlyst = proc_float_scale(ic_anlyst, ['IC序列均值'], ".2%")
    ic_anlyst = proc_float_scale(ic_anlyst, ['ICIR'], ".2f")
    return ic_anlyst
    
def ic_test_summary(factor_df, mret_df, ret_col_name, factor_list):
    '''
    函数：因子的IC测试与统计
    输入：
        factor_df: DataFrame，待测试的动量因子值，columns至少包含因子名和ticker、tradeDate
        mret_df: DataFrame，个股月度收益率   
        ret_col_name: 下月收益率在mret_df中的列名
        factor_list：list，改进的动量因子与原始动量因子列名
    返回：IC测试结果
    '''
    # IC测试
    ic_df = ic_calc(factor_df, mret_df, factor_list, ret_col_name, 'pearson')
    # IC测试统计
    ic_res = ic_describe(ic_df) 
    return ic_res
#分组测试相关函数
def group_test(factor_df, fac_name, mret_df, ret_col_name, group_num):
    '''
    函数：对某个因子做分组回测，获取每组每期（月）收益率
    输入：
        factor_df：DataFrame 待测因子值  column至少包含ticker、tradeDate、fac_name（待测因子列名） 
        mret_df：DataFrame  月收益   column至少包含ticker、tradeDate、ret_col_name（下月收益率的列名）
        group_num：分组组数
    返回：
        perf_df：DataFrame  因子各分组期间收益率  index为日期   columns为'0' ~ '(group_num-1)'
        cumret_df：DataFrame  分组净值收益  index为日期   column为'0' ~ '(group_num-1)' 
    '''
    perf, _ = qutil.simple_group_backtest(factor_df, mret_df, factor_name=fac_name, return_name=ret_col_name, commission=0, ngrp=group_num)  # 卖出不收取费率
    # 分组期间收益率
    perf_df = perf.pivot_table(values='period_ret', index='tradeDate', columns='group')  
    # 分组净值收益
    long_idx_df = perf_df.copy()
    long_idx_df.ix[0, :] = 0
    long_idx_df = long_idx_df.sort_index()
    cumret_df = (long_idx_df[range(group_num)]+1).cumprod()
    return perf_df, cumret_df-1

def group_ls_describe(perf_df, annual_len, group_num, col_name, ls_direction='reverse'):
    '''
    函数：根据每组每期（月）收益率，统计分组及多空表现
    输入：
        perf_df：DataFrame，因子分组期间收益率，index为日期，columns为'0' ~ '(group_num-1)'
        annual_len：年化期间
        col_name：统计结果表格中的因子index名
    返回：       
        long_short_cumret：Series，多空净值，index为日期   
        perf_desc.T：多空绩效指标，columns为指标名称 
    '''
    # 计算多空收益净值
    if ls_direction == 'reverse':
        long_short_ret = -(perf_df[group_num-1] - perf_df[0])   # 多空收益率序列
    else:
        long_short_ret = (perf_df[group_num-1] - perf_df[0])   # 多空收益率序列
        
    long_short_cumret = (long_short_ret+1).cumprod()   # 多空收益净值序列
    # 计算分组年化收益
    group_res = (perf_df.mean()*annual_len).tolist()
    # 计算年化指标
    perf_anlyst = group_res + perf_describe(long_short_ret, annual_len)
    perf_ls_desc = pd.DataFrame(perf_anlyst, columns=[col_name]).T
    perf_ls_desc.columns = ['第%s组年化收益率'%i for i in range(1, group_num+1)]+['多空组合年化收益率', '多空组合年化波动率', '多空组合夏普率', '月度胜率', '多空组合最大回撤率']
    perf_desc = proc_float_scale(perf_ls_desc, ['第%s组年化收益率'%i for i in range(1, group_num+1)]+['多空组合年化收益率', '多空组合年化波动率', '多空组合最大回撤率', '月度胜率'], ".2%")
    perf_desc = proc_float_scale(perf_desc, ['多空组合夏普率'], ".2f")
    return long_short_cumret, perf_desc 

def perf_describe(long_short_ret, annual_len):
    '''
    函数：计算多空对冲年化指标 
    输入：long_short_ret：Series，多空对冲收益每期数据
    返回：各类指标列表
    '''
    mean = long_short_ret.mean()*annual_len  # 计算均值
    std = long_short_ret.std()*np.sqrt(annual_len)  # 计算波动率
    sharp = mean / std  
    win = (long_short_ret>0).sum() / float(len(long_short_ret)-1)  # 计算月胜率 
    l_s_cumret = (long_short_ret+1).cumprod()  # 净值序列
    running_max = np.maximum.accumulate(l_s_cumret)    
    drawback = ((running_max-l_s_cumret) / running_max).dropna().max()  # 计算最大回撤     
    return [mean, std, sharp, win, drawback]
    
def plot_group_ls_cumret(group_cumret, long_short_cumret, legend1, legend2, title_part):
    '''
    函数：同时展示因子5分组回测和多空对冲净值走势，以及累计净值柱状图
    参数：
    group_cumret：DataFrame，分组净值，index为日期，column为'target_ret_0' ~ 'target_ret_4'
    long_short_cumret：Series，多空净值，index为日期 
    '''
    fig = plt.figure(figsize=(18,6))        
    ax1 = fig.add_subplot(121)
    ax2 = ax1.twinx()
    _ = ax1.plot(pd.to_datetime(group_cumret.index), group_cumret)
    _ = ax2.plot(pd.to_datetime(long_short_cumret.index), long_short_cumret, 'r--')
    _ = ax1.legend(legend1, loc=2, prop=font)  
    _ = ax2.legend(legend2, loc=1, prop=font)
    _ = ax1.set_title(u'%s 5分组回测及多空对冲净值走势' % title_part, fontsize=16, fontproperties=font)
    _ = plt.grid(b=None)
    
    ax3 = fig.add_subplot(122)
    plot_baseline = np.arange(5)
    _ = ax3.bar(plot_baseline, group_cumret.iloc[-1]-1)
    _ = ax3.set_title(u'%s 5分组累计收益柱状图' % title_part, fontsize=16, fontproperties=font)  
    _ = ax3.set_xticks(plot_baseline+0.3)
    _ = ax3.set_xticklabels([u'第%s组' %str(i+1) for i in range(5)], fontproperties=font, rotation=0)
    _ = plt.show()

    
# 画图展示数据的覆盖股票个数和覆盖度
def plot_coverage(indata_df, mret_df=None, universe='TLQA', title_tail=''):
    '''
    indata_df: 列为 :ticker, date(%Y%m%d), ,covered_ticker_count
    mret_df: 行情datafarme，列至少为 ticker, tradeDate(%Y%m%d), 需要输入是因为有时候要用到剔除次新股、ST的行情dataframe
    title_tail: 图标题为 Factor Coverge + <title_tail>
    用到了全局变量 df_<universe>: 列为ticker, tradeDate("%Y%m%d")
    '''
    # 市场总的股票个数计算覆盖度
    if mret_df is None:
        mret_df = pd.read_pickle(os.path.join(data_save_dir, 'mret.pkl'))
    if universe.upper() == 'ZZ800':
        mret_df = mret_df.merge(df_zz800, on=['ticker','tradeDate'], how='inner')
    elif universe.upper() == 'ZZ500':
        mret_df = mret_df.merge(df_zz500, on=['ticker','tradeDate'], how='inner')
    elif universe.upper() == 'HS300':
        mret_df = mret_df.merge(df_hs300, on=['ticker','tradeDate'], how='inner')
    elif universe.upper() == 'TLQA':
        pass
    else:
        raise Exception(u'不支持的投资域:%s'%universe)
    
    ashare_count = mret_df.groupby(['tradeDate'])['ticker'].count().reset_index()
    ashare_count.columns = ['date', 'all_ticker']

    statistic_df = indata_df.merge(ashare_count, on=['date'], how='inner')
    statistic_df['coverage'] = statistic_df['covered_ticker_count']*1.0/statistic_df['all_ticker']
    statistic_df.index = pd.to_datetime(statistic_df['date'], format='%Y%m%d')

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    
    _ = statistic_df['covered_ticker_count'].plot.area(ax=ax2, color='g', alpha=0.5)
    _ = statistic_df['coverage'].plot(ax=ax, color='r')
    _ = ax.legend(('coverage in all %s'%universe,),loc=2)
    _ = ax2.legend(('covered stocks num',), loc=1)
    _ = ax2.set_ylabel('covered stocks No')
    _ = ax.set_ylabel('coverage in %s'%universe)
    _ = plt.title(u"Factor Coverage "+title_tail)
    _ = plt.show()    

def bucket_performance(factor_df, mret_df, factor_name_list, factor_describe_list, ret_col_name,group_num,prefix=u'', universe='TLQA'):
    '''
    输入：
        factor_df: DataFrame，因子值的dataframe,列包括ticker, tradeDate(%Y%m%d), <factor_name_list>
        factor_name_list：factor_df中的因子名列表
        factor_describe_list: 因子名的中文注释
        mret_df: DataFrame，个股月度收益率,列包括 ticker,tradeDate("%Y%m%d"), <ret_col_name>
        ret_col_name: 下月收益率在mret_df中的列名
        group_num:分组数
    函数中用到了全局变了 df_<投资域>, 列为 ticker, tradeDate，用来筛选投资域
    返回：分组测试结果   
    '''  
    # 分组测试并统计结果
    group_cumret_list = []
    perf_desc_list = []
    long_short_cumret_list = []
    
    if universe.upper() == 'ZZ800':
        mret_df = mret_df.merge(df_zz800, on=['ticker','tradeDate'], how='inner')
    elif universe.upper() == 'ZZ500':
        mret_df = mret_df.merge(df_zz500, on=['ticker','tradeDate'], how='inner')
    elif universe.upper() == 'HS300':
        mret_df = mret_df.merge(df_hs300, on=['ticker','tradeDate'], how='inner')
    else:
        pass
    
    for i in range(len(factor_name_list)):
        factor_name = factor_name_list[i]
        group_perf, group_cumret = group_test(factor_df, factor_name, mret_df, ret_col_name, group_num)
        long_short_cumret, perf_desc = group_ls_describe(group_perf, 12, group_num, prefix+factor_describe_list[i], ls_direction='normal')
        group_cumret_list.append(group_cumret)
        perf_desc_list.append(perf_desc)
        long_short_cumret_list.append(long_short_cumret)
    
    bucket_perf_df = pd.concat(perf_desc_list, axis=0)
    print u"5分组多空对冲绩效指标对比", bucket_perf_df.to_html()
    # 展示因子覆盖度，5分组回测和多空对冲净值走势、累计净值分布
    for i in range(len(factor_name_list)):
        # 之所以用TLQA是因为mret_df在上面已经filter了
        factor_name = factor_name_list[i]
        plot_factor_df = factor_df[['ticker','tradeDate', factor_name]].dropna().merge(mret_df, on=['ticker','tradeDate'],how='inner')
        factor_count_df = plot_factor_df.groupby(['tradeDate'])['ticker'].count().reset_index()
        factor_count_df.columns = ['date', 'covered_ticker_count']
        print '-------------------------------------------------------------------------------------------'
        _ = plot_coverage(factor_count_df, mret_df=None, universe=universe) 
        _ = plot_group_ls_cumret(group_cumret_list[i], long_short_cumret_list[i], np.arange(5)+1, [u'多空对冲(右轴)'], prefix+factor_describe_list[i])
    return bucket_perf_df


print "因子测试结果..."
start_time = time.time()
fac_list = ['futureQep','currentQep','futureYep','currentYep']
# 因子处理(剔除禁止股票池、去极值、行业市值中性化、标准化)
c_p_df = fac_process(factor_df, fac_list, forbidden_pool, neutral=1)
# IC测试统计
c_ic_res = ic_test_summary(c_p_df, mret_df_real, 'nxt1m_ret', fac_list)
print "IC测试结果：", c_ic_res.to_html()
# print "分组测试分析..." 
_ = bucket_performance(c_p_df, mret_df_real, ['futureQep','currentQep','futureYep','currentYep'],[u'使用下个季度的数据',u'使用当前季度的数据', u'使用下一个年度财报的数据', u'使用当前年度财报的数据'],'nxt1m_ret', 5)

print "耗时: %s seconds" % (time.time() - start_time)

'''

上图future图的最后一期的覆盖度突变是因为还没有下个季度的真实值，可忽略；
从上面的结果来看，无论是提前预知未来1个季度的数据（futureQep)还是提前预知未来1年的数据(futureYep), 相比于使用当前的季度(currentQep)和当前的年度数据(currentYep)，IC都有有将近2倍以上的提高，ICIR提高了1.5倍以上，因此如果能够预知未来数据，对于提高收益是很有帮助的
    


第二部分：如何计算季度盈利预测数据
该部分耗时 15分钟
参考天风证券的思想，按照如下的流程来确定计算季度盈利预测的方法：
图片注释

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

2.1 数据统计（业绩预告和分析师预期数据的覆盖度）

2.1.1 业绩预告覆盖股票数

'''

tstart = time.time()
pre_report_df = DataAPI.FdmtEfGet(ticker=all_stock_list,endDate=u"20200930",beginDate=u"20100331",
                  field=u"",pandas="1")

# 业绩预告中至少应该包括如下科目中的任何一项
valid_columns = [u'revChgrLL', u'revChgrUPL', u'expRevLL', u'expRevUPL',
       u'NIncomeChgrLL', u'NIncomeChgrUPL', u'expnIncomeLL', u'expnIncomeUPL',
       u'NIncAPChgrLL', u'NIncAPChgrUPL', u'expnIncAPLL', u'expnIncAPUPL']

# 过滤掉没有实质性内容的业绩预告记录，可能出现在招股说明书等内容中
pre_report_df = pre_report_df.dropna(subset=valid_columns, how='all').query("mergedFlag=='1'")
pre_report_df['report'] = pre_report_df['endDate'].apply(lambda x: "%sQ%s"%(x[:4], int(x[5:7])/3))


# 只保留净利润的预测
pre_report_df =pre_report_df[['secID','publishDate','report','reportType','NIncAPChgrLL','NIncAPChgrUPL','expnIncAPLL','expnIncAPUPL']].dropna(how='all')
# 取预告上下限的非空均值作为预告值
pre_report_df['ChgMean'] = pre_report_df[['NIncAPChgrLL', 'NIncAPChgrUPL']].mean(skipna=True,axis=1)
pre_report_df['expnIncMean'] = pre_report_df[['expnIncAPLL', 'expnIncAPUPL']].mean(skipna=True,axis=1)

pre_report_df['ticker'] = pre_report_df['secID'].apply(lambda x: x.split(".")[0])
# 得到该预告对应财报期的上一期财报戳
pre_report_df['prev_report'] = pre_report_df['report'].apply(lambda x: "%s%s"%(int(x[:4])-1, x[-2:]))

# 基于同比增速得到的累计归母净利润
pre_report_df = pre_report_df.merge(accum_earning_df[['ticker','Q_flag','NIncomeAttrP','reportType']].rename(columns={"Q_flag":"prev_report", "NIncomeAttrP":"prev_acc_profit"}), on=['ticker','prev_report','reportType'], how='left')
pre_report_df['announce_acc_profit'] = pre_report_df['prev_acc_profit']*(pre_report_df['ChgMean']+1)
# 预告的归母净利润
pre_report_df['announce_predict_profit'] = pre_report_df['expnIncMean'].fillna(pre_report_df['prev_acc_profit'])

# 预告中可能是关于单季度的预告，也可能是累计的预告，如果有单季度的预告就直接用单季度的，没有再通过累计进行推算
pre_singleq_report_df = pre_report_df[pre_report_df['reportType'].isin(['Q1','Q3'])]
pre_singleq_report_df['fore_type'] = '1'
pre_singleq_report_df['publishDate'] = pre_singleq_report_df['publishDate'].apply(lambda x: x.split(" ")[0])
pre_singleq_report_df = pre_singleq_report_df[['ticker','publishDate','report','announce_predict_profit','fore_type']].rename(columns={"announce_predict_profit":"fore_q_annouce"})

# 处理预告中是累计预告的情况
pre_report_df = pre_report_df[~pre_report_df['reportType'].isin(['Q1','Q3'])]
# 对齐当前已发财报的累计归母净利润
def get_latest_end_date(x):
    year = x[:4]
    quarter = x[-1]
    # Q1之前无累计季度
    if quarter == '1':
        last_acc_quarter = '0'
    elif quarter == '2':
        last_acc_quarter = '%s-03-31'%year
    elif quarter == '3':
        last_acc_quarter = '%s-06-30'%year
    elif quarter == '4':
        last_acc_quarter = '%s-09-30'%year
    else:
        raise Exception("not valid quarter flag")
    return last_acc_quarter

pre_report_df['last_end_date'] = pre_report_df['report'].apply(lambda x: get_latest_end_date(x))
# 对齐上个季度的累计值
pre_report_df = pre_report_df.merge(accum_earning_df[['ticker','publishDate','endDate','NIncomeAttrP','reportType']].rename(columns={"endDate":"last_end_date", 'NIncomeAttrP':"latest_acc_profit", "publishDate":"latest_acc_pdate", "reportType":"latest_acc_reportType"}), on=['ticker', 'last_end_date'], how='left')

# 剔除掉预测季度非Q1，但是last_acc_profit为空的记录
pre_report_df = pre_report_df[~((pre_report_df['latest_acc_profit'].isnull())&(pre_report_df['reportType']!='Q1'))]
# 剔除掉业绩预告的发布日比最新财报期发布日还要晚的记录
pre_report_df['publishDate'] = pre_report_df['publishDate'].apply(lambda x: x.split(" ")[0])
# 填充空值为2000年
pre_report_df['latest_acc_pdate'] = pre_report_df['latest_acc_pdate'].fillna("2000-01-01")
pre_report_df = pre_report_df.query("publishDate>=latest_acc_pdate")

# 单季度的预告值
pre_report_df['fore_q_annouce'] = pre_report_df['announce_predict_profit'] - pre_report_df['latest_acc_profit'].fillna(0)
# 如果同时预告了归母净利润绝对值和增速，则优先取绝对值
pre_forecast_q_df = pre_report_df[['ticker', 'publishDate','report', 'fore_q_annouce']]
pre_forecast_q_df['fore_type'] = 0

## 合并之前的的单季度预测数据
pre_forecast_q_df = pd.concat([pre_singleq_report_df, pre_forecast_q_df], axis=0)
# 如果有重复的，优先使用单季度预告的
pre_forecast_q_df = pre_forecast_q_df.sort_values(by=['ticker', 'publishDate','report', 'fore_type'])
pre_forecast_q_df = pre_forecast_q_df.drop_duplicates(subset=['ticker','publishDate'], keep='last')
print '预告得到的季度归母净利润格式为:'
print pre_forecast_q_df.head().to_html()

tend = time.time()
print u'该部分耗时:%s seconds'%(round(tend-tstart, 2))

# 画图展示每个财务期的业绩预告个数
pre_count_df = pre_forecast_q_df.groupby(['report'])['ticker'].count()
_ = pre_count_df.plot.bar(figsize=(15,5))

print pre_count_df.tail(20)



'''

从上面的数据和图表中可以看出，2019年以来业绩快报相比于过去同期出于明显的下降趋势，从2018到2020年，Q1预告数从1634到1628到1868（20年Q1因为疫情原因，数据异常可以不考虑）， 半年报预告数从2376到1649到1356， 三季报预告数从2170到1595到1104，而这几年上市公司的数量是处于增加阶段的。因此，按该趋势下去，业绩预告覆盖的股票数会越来越少

    
文档
2.1.2 一致预期覆盖度（15分钟）

'''

stime = time.time()
print u'逐月读取一致预期数据...'
if os.path.exists(os.path.join(data_save_dir, "consensus_df.pkl")):
    consensus_df =  pd.read_pickle(os.path.join(data_save_dir, "consensus_df.pkl"))
else:
    consensus_list = []
    for tdate in date_list:
        print tdate,
        consensus_df = DataAPI.ResConSecDataGet(endDate=tdate.replace("-", ""),beginDate=tdate.replace("-", ""),
                                                field='secCode,repForeTime,ConProfitType,foreYear,foreType,conProfit',pandas="1")
        consensus_df = consensus_df[consensus_df['ConProfitType'].isin([1,2])]
        consensus_list.append(consensus_df)
    consensus_df = pd.concat(consensus_list, axis=0)

    consensus_df.to_pickle(os.path.join(data_save_dir, "consensus_df.pkl"))
etime = time.time()
print u'\n一致预期数据读取完成，耗时:%s seconds'%(etime-stime)


# 取预测年度为预测日当年
consensus_df['date']=consensus_df['repForeTime'].apply(lambda x: x.split(" ")[0].replace("-", ""))
consensus_df['forecast_year'] = consensus_df['date'].apply(lambda x: int(x[:4]))
consensus_df = consensus_df.query("foreYear==forecast_year")
consensus_df.head()

# 分析师一致预期覆盖的股票个数
consensus_cover_count = consensus_df.dropna().groupby(['date'])['secCode'].nunique().reset_index()
consensus_cover_count.columns = ['date', 'covered_ticker_count']

_ = plot_coverage(consensus_cover_count)


'''

从上图可以看出，一致预期覆盖的股票数在2017年中达到了顶峰，此后逐步下降，虽然在2020年后半年有所增加，但从对整个市场的覆盖度来看，2018年后的覆盖度明显低了很多，近两年都在55%的水平

    
文档
2.2 规律外推计算季度盈利预测
    
文档

假设一：预测季度(未来1个季度)占比等于历史占比均值
参考天风证券研报中比例外推的公式：
FPn+1,T=Pn,TPCT¯¯¯¯¯¯¯−Pn,T
将其进行等式变换，其等价于：

Pn,TFPn+1,T+Pn,T=PCT¯¯¯¯¯¯¯

图片注释

该假设成立需要一定先决条件，参考研报中的做法，对应:

条件1：基期盈利正常
图片注释

条件2：历史盈利分布稳定
图片注释

由于是用历史数据线性外推，基期盈利正常可以理解为公司最近季度的盈利没有出现明显的波动、且过去3年的历史中n季度和n+1季度的盈利关系没有太大的波动

假设二：预测季度(未来1个季度)同比增速等于历史增速
当历史盈利分布（占比）不稳定时，如果增速平稳，则可以用假设二进行线性外推，需要的条件是：

基期盈利正常（条件1）
历史盈利分布不稳定(占比，条件2)
预测期上年同期盈利正常（条件3），具体为：
ABS(Pn+1,T−1−AvgPn)/AvgPn<2
其中，AvgPn=(ABS(Pn+1,T−2)+ABS(Pn+1,T−3))/2
为了更好的理解该公式，假设P都是大于0的（对于绝大部分公司也是成立的），则条件3的公式可以推导成：
Pn+1,T−1<AvgPn<=max(Pn+1,T−2,Pn+1,T−3)

因此，这一组条件可以理解为上个季度的盈利正常（基期），虽然历史盈利不稳定，但是去年预测季度(n+1)的盈利并未超过前两年的季度盈利，也暗含了下一个预测季度值也在一定的范围内的假设。

整体来看，以上两组条件的设定都有一定的合理性，但对于线性外推的数学方法来说也都不是非常严格的数学假设，这里面有经验的成分，读者也可以根据自己的经验进一步丰富和完善

2.2.1 计算季度预期数据

'''

# 条件一判断，基期盈利是否正常
def add_col_basenormal(df):
    '''
    df:财报数据，dataframe格式，列至少包括：ticker, P_T_n, P_T-1_n三列
    返回：
    在df中增加一列,'is_base_normal'
    '''
    df['cum_prod'] = df['P_T_n']*df['P_T-1_n']
    df['delta'] = df['P_T_n'] - df['P_T-1_n']
    df['pct'] = df['delta']/df['P_T-1_n']
    df['pct'] = df['pct'].abs()
    df['is_base_normal'] = np.where(((df['pct']<0.5) &(df['cum_prod']>0)), 1, 0)
    del df['pct']
    del df['cum_prod']
    del df['delta']
    return df

# 条件二判断，历史盈利分布是否稳定
def add_col_historystable(df):
    '''
    df:财报数据，dataframe格式，列至少包括：ticker, <P_T-i_n, i=1,2,3>, <P_T-i_n+1,i=1,2,3>
    返回：
    在df中增加一列,'is_history_stable'
    '''
    for i in range(1,4):
        df['PCT_T-%s'%i] = df['P_T-%s_n'%i]/(df['P_T-%s_n'%i] + df['P_T-%s_n+1'%i])
    df['PCT_mean'] = df[['PCT_T-1', 'PCT_T-2', 'PCT_T-3']].mean(axis=1)
    
    
    delta_col_list = []
    for i in range(1,4):
        delta_col = 'delta%s'%i
        df[delta_col] = (df['PCT_T-%s'%i] - df['PCT_mean']).abs()
        delta_col_list.append(delta_col)
    df['MAD'] = df[delta_col_list].max(axis=1)
    df['is_history_stable'] = np.where(df['MAD']<0.1, 1, 0)
    
    # 删除中间过程列
    for i in range(1,4):
        del df['PCT_T-%s'%i]
        del df['delta%s'%i]
    del df['MAD']
    return df



# 条件三判断，预测期上年同期盈利是否正常
def add_col_prevyqnormal(df):
    '''
    df:财报数据，dataframe格式，列至少包括：ticker, P_T-2_n+1, 'P_T-3_n+1'
    返回：
    在df中增加一列,'is_base_normal'
    '''
    df['AvgPn'] = (df['P_T-2_n+1'].abs()+df['P_T-3_n+1'].abs())/2
    df['calc_v'] = (df['P_T-1_n+1'] - df['AvgPn']).abs()/df['AvgPn']
    df['is_prevyq_normal'] = np.where(df['calc_v']<2, 1, 0)
    del df['AvgPn']
    del df['calc_v']
    return df

# 比例外推得到的下期的预测值
def forecast_nxt_equlratio(df):
    '''
    df:财报数据，dataframe格式，列至少包括：ticker, PCT_mean, P_T_n
    返回：
    在df中增加一列,'fore_q_equalratio', 线性外推的值
    '''
    df['fore_q_equalratio'] = df['P_T_n']/df['PCT_mean'] - df['P_T_n']
    return df

# 增速外推得到的下期的预测值
def forecast_nxt_equlgrowth(df):
    df['gnT'] = (df['P_T_n']-df['P_T-1_n'])/df['P_T-1_n'].abs()  #最新季度的同比增长率
    df['Cg_T-1_n+1'] = (df['P_T-1_n'] + df['P_T-1_n+1'] - (df['P_T-2_n']+df['P_T-2_n+1']))/(df['P_T-2_n']+df['P_T-2_n+1']).abs() #去年两个季度的混合同比增长率
    df['Cg_T-2_n+1'] = (df['P_T-2_n'] + df['P_T-2_n+1'] - (df['P_T-3_n']+df['P_T-3_n+1']))/(df['P_T-3_n']+df['P_T-3_n+1']).abs() #前年两个季度的混合同比增长率
    df['growth'] = ((df['Cg_T-1_n+1'] + df['Cg_T-2_n+1'])/2 + df['gnT'] )/2
    df['fore_q_equalgrowth'] = df['P_T-1_n+1']*(1+df['growth'])
    del df['gnT']
    del df['Cg_T-1_n+1']
    del df['Cg_T-2_n+1']
    del df['growth']
    return df


tstart = time.time()
# 构造数据结构，对齐基期和预测期过去3年的季度数据
quarter_df = earning_df[['ticker','publishDate','Q_flag','Q_NAttrP']]    
quarter_df['Next_Q'] = quarter_df['Q_flag'].apply(lambda x: get_next_quarter(x))
aligned_earning_df = quarter_df[['ticker','publishDate', 'Q_flag', 'Q_NAttrP', 'Next_Q']].drop_duplicates()
aligned_earning_df.columns = ['ticker', 'date', 'T_n','P_T_n', 'T_n+1']
trace_year = 3
for i in range(1, trace_year+1):
    # T-i年的n期
    aligned_earning_df['T-%s_n'%i] = aligned_earning_df['T_n'].apply(lambda x: "%s%s"%(int(x[:4])-i,x[-2:]))
    # 对齐上T-i年的n期单季度数据
    add_earning_df = aligned_earning_df[['ticker','T_n','P_T_n']].rename(columns={"T_n":'T-%s_n'%i, 'P_T_n':"P_T-%s_n"%i})
    
    aligned_earning_df = aligned_earning_df.merge(add_earning_df, on=['ticker', 'T-%s_n'%i], how='left')
    # T-i年的n+1期
    aligned_earning_df['T-%s_n+1'%i] = aligned_earning_df['T_n+1'].apply(lambda x: "%s%s"%(int(x[:4])-i,x[-2:]))
    # 对齐上T-i年的n+1期单季度数据
    add_earning_df = aligned_earning_df[['ticker','T_n','P_T_n']].rename(columns={"T_n":'T-%s_n+1'%i, 'P_T_n':"P_T-%s_n+1"%i})
    aligned_earning_df = aligned_earning_df.merge(add_earning_df, on=['ticker', 'T-%s_n+1'%i], how='left')
print aligned_earning_df.query("ticker=='000001'").tail().to_html()

tend = time.time()
print u'该部分耗时:%s seconds'%(round(tend-tstart, 2))

# 增加基期盈利是否正常的标签
aligned_earning_df = add_col_basenormal(aligned_earning_df)

# 增加历史盈利分布是否稳定的标签
aligned_earning_df = add_col_historystable(aligned_earning_df)

# 增加预测期上年同期是否盈利正常的标签
aligned_earning_df = add_col_prevyqnormal(aligned_earning_df)

# 用等比例外推得到的下个季度的预测值
aligned_earning_df = forecast_nxt_equlratio(aligned_earning_df)

# 用等增速外推得到的下个季度的预测值
aligned_earning_df = forecast_nxt_equlgrowth(aligned_earning_df)

# 剔除掉同一天发布多份报告，如年报和1季报的记录，只保留最新的1季报数据
aligned_earning_df = aligned_earning_df.sort_values(by=['ticker', 'date', 'T_n+1'], ascending=True)
aligned_earning_df=aligned_earning_df.drop_duplicates(subset=['ticker', 'date'], keep='last')

# 处理一下aligned_earning_df, 如A公司，在20141029发了2014Q3公告，到20150402发了2015Q1报告，在20150420发了2014Q4报告，则2014Q4报告作废
aligned_earning_df = aligned_earning_df.sort_values(by=['ticker', 'date'])
aligned_earning_df['max_T_n+1'] = aligned_earning_df.groupby(['ticker'])['T_n+1'].rolling(3, min_periods=1).max().values
del_al = aligned_earning_df[aligned_earning_df['T_n+1']<aligned_earning_df['max_T_n+1']]
aligned_earning_df = aligned_earning_df[aligned_earning_df['T_n+1']>=aligned_earning_df['max_T_n+1']]
print aligned_earning_df.query("ticker=='000001'").tail().to_html()

'''

2.2.2 分析各种方法下的季度预期数据覆盖度
先不考虑各种假设条件是否成立，分析一下用不同方法得到的季度预测数据的覆盖度（月度）

'''

# 将零散的日期数据，变成规整的数据（如一周一个，一个月一个等）
def pit2fixedfreq(raw_data_frame, freq_kline_frame):
    '''
    raw_data_frame: 包含pit数据的datafame，列至少为ticker, date("%Y%m%d"), <各种PIT对应的财务列>
    freq_kline_frame: 具有固定期限的行情类dataframe，如日度、周度、月度行情数据，列至少为 ticker, date(%Y%m%d)
    return: PIT转成的和freq_kline_frame同周期的数据，如freq_kline_frame是月度的，则返回
    前向填充后，一个月一个因子数据的dataframe，列为: ticker, date, <各种财务列>
    '''
    
    raw_data_frame['date'] = raw_data_frame['date'].apply(lambda x: str(x).split(" ")[0].replace("-", ""))
    # 保留空值特征
    raw_data_frame = raw_data_frame.replace(np.nan, 'NULL_FILL')
    
    freq_kline_frame['date'] = freq_kline_frame['date'].apply(lambda x: str(x).split(" ")[0].replace("-", ""))
    freq_kline_frame['end_mark'] = 1
    
    cont_dframe = freq_kline_frame[['ticker', 'date', 'end_mark']].merge(raw_data_frame, on=['ticker', 'date'], how='outer')
    cont_dframe = cont_dframe.sort_values(by=['ticker', 'date'], ascending=True)
    
    # 前向填充
    for tcol in [x for x in cont_dframe.columns if x not in ['ticker', 'date','end_mark']]:
        cont_dframe[tcol] = cont_dframe.groupby('ticker')[tcol].fillna(method='ffill').values
    
    # 仅保周期末数据
    cont_dframe = cont_dframe.query("end_mark==1")
    
    # 空值还原
    cont_dframe = cont_dframe.replace("NULL_FILL", np.nan)
    return cont_dframe


for tcol in ['fore_q_equalratio', 'fore_q_equalgrowth']:
    predict_dframe = aligned_earning_df[['ticker', 'date', tcol]]
    predict_dframe['date'] = predict_dframe['date'].apply(lambda x: x.replace("-", ""))
    predict_dframe = pit2fixedfreq(predict_dframe, mret_df.rename(columns={"tradeDate":"date"}))
    print u'%s 计算得到的非空覆盖度股票数:'%tcol
    
    coverd_stock_df = predict_dframe.dropna().groupby(['date'])['ticker'].count().reset_index()
    coverd_stock_df.columns = ['date', 'covered_ticker_count']
    _ = plot_coverage(coverd_stock_df)


'''
从上图来看，两种衍生计算的季度预测值，对于全A股票的覆盖度近几年在80%以上，明显高于分析师一致预期覆盖的50%水平，单从覆盖度来看是可以提供增量信息的。此外，之所以覆盖度达不到95%以上的水平，是因为用了3年的历史财务数据，次新股的股票无法用这种方式推算出来。从下图可以看出，上市3年以上的股票比例近几年在80%到95%之间波动
'''

mret_df = mret_df.rename(columns={"tradeDate":"date"})
# 市场中1年、2年、3年以上的股票占比
stock_live_df = mret_df.merge(stock_info_df, on=['ticker'], how='left')
stock_live_df['listDate']=pd.to_datetime(stock_live_df['listDate'], format='%Y-%m-%d')
stock_live_df['curDate']=pd.to_datetime(stock_live_df['date'], format='%Y%m%d')
stock_live_df['live_days'] = stock_live_df['curDate']-stock_live_df['listDate']
stock_live_df['live_years']=stock_live_df['live_days'].apply(lambda x: round(x.days/365.0), 2)

year_1m_count = stock_live_df.query("live_years>=1").groupby(['date'])['ticker'].count()
year_2m_count = stock_live_df.query("live_years>=2").groupby(['date'])['ticker'].count()
year_3m_count = stock_live_df.query("live_years>=3").groupby(['date'])['ticker'].count()
year_count = pd.concat([year_1m_count, year_2m_count, year_3m_count], axis=1).reset_index()
year_count.columns = ['date', 'stocks_1year_longer', 'stocks_2year_longer','stocks_3year_longer']

# 
ashare_count = mret_df.groupby(['date'])['ticker'].count().reset_index()
ashare_count.columns = ['date', 'all_ticker']
year_count = year_count.merge(ashare_count,on=['date'], how='right')
for col in ['stocks_1year_longer', 'stocks_2year_longer','stocks_3year_longer']:
    year_count[col] = year_count[col]*1.0/year_count['all_ticker']

_ = year_count.set_index('date')[['stocks_1year_longer', 'stocks_2year_longer','stocks_3year_longer']].plot(figsize=[15,5])


'''

2.3 通过分析师年度预测推算季度预测
分析师对公司的预测只有年报数据，因此分析师一致预期数据中也只有年报的预测，将年报预测拆解成季度预测有多种方法，每种方法都有其固有的假设，参考天风证券在研报中的做法，此处做如下假设:

预测季度占所在年度未披露数据中的比例同去年保持一致
即，预测T年Q1季度时，Q1(预测) = T年全年一致预期值*T-1年Q1占全年的比例
预测T年Q3季度时，Q3(预测) = （T年全年一致预期值-T年的半年报数据）*(T-1年中Q3占下半年累计值的比例)，Q2和Q4以此类推


'''

tstart = time.time()
analyst_data = finan_df[['ticker','publishDate','Q_flag','Q_NAttrP','NIncomeAttrP','reportType']]

# 预测的季度
analyst_data['Next_Q'] = analyst_data['Q_flag'].apply(lambda x: get_next_quarter(x))

# 分析师预测的年份
analyst_data['year'] = analyst_data['Next_Q'].apply(lambda x: "%s1231"%x[:4])

merged_data = analyst_data[['ticker', 'Q_flag', 'Q_NAttrP']]
# 去年4个季度的单季度值
for i in range(1, 5):
    quarter_col = 'prev_y_Q%s'%i
    # 具体去年的第i个季度
    analyst_data[quarter_col] = analyst_data['Next_Q'].apply(lambda x: "%sQ%s"%(int(x[:4])-1, i))
    # 合并去年第i个季度的财报值
    analyst_data = analyst_data.merge(merged_data.rename(columns={"Q_flag":quarter_col, "Q_NAttrP":'prev_y_Q%s_v'%i}), on=['ticker',quarter_col], how='left')


# 去年，预测季度占当年剩余季度的比例
def get_ratio(df):
    try:
        fore_q = df['Next_Q']
        last_Q1 = float(df['prev_y_Q1_v'])
        last_Q2 = float(df['prev_y_Q2_v'])
        last_Q3 = float(df['prev_y_Q3_v'])
        last_Q4 = float(df['prev_y_Q4_v'])
        if fore_q[-1] == '4':
            ratio = 1
        elif fore_q[-1] == '3':
            ratio = last_Q3/(last_Q4 + last_Q3)
        elif fore_q[-1] == '2':
            ratio = last_Q2/(last_Q2+last_Q3+last_Q4)
        elif fore_q[-1] == '1':
            ratio = last_Q1/(last_Q1+last_Q2+last_Q3+last_Q4)
        else:
            raise Exception(u'不支持的Next_Q')
        return ratio
    except:
        return np.nan
analyst_data['ratio'] = analyst_data.apply(lambda x: get_ratio(x), axis=1)
analyst_data = analyst_data.dropna(subset=['ratio'])
analyst_data.head()


# 将一致预期比例数据和每月一个的一致预期数据合并到一起
ratio_data = analyst_data[['ticker', 'publishDate', 'Next_Q', 'year', 'ratio', 'reportType', 'NIncomeAttrP']]
ratio_data.columns = ['ticker', 'date', 'forecast_q', 'forecast_year', 'ratio', 'reportType', 'NIncomeAttrP']
ratio_data['date'] = ratio_data['date'].apply(lambda x: x.replace("-", ""))
# 后续使用了前向填充，为了保持原有的空缺值（如ratio在某期计算就为nan，则需要将nan一直保留，不能被前向填充改变nan的值）
ratio_data = ratio_data.replace(np.nan, 'NULL_FILL')

cons_df = pd.read_pickle(os.path.join(data_save_dir, "consensus_df.pkl"))
cons_df['date'] = cons_df['repForeTime'].apply(lambda x: x.split(" ")[0].replace("-", ""))
cons_df['forecast_year_analyst'] = cons_df['foreYear'].apply(lambda x: "%s1231"%x)

cons_df = cons_df[['secCode', 'date', 'forecast_year_analyst', 'conProfit']]
cons_df.columns = ['ticker', 'date', 'forecast_year_analyst', 'forecast_analyst']
cons_df['month_end_mark'] = 1

cons_df = cons_df.merge(ratio_data, on=['ticker', 'date'], how='outer')

# 前向填充得到来对齐ratio数据
cons_df = cons_df.sort_values(by=['ticker', 'date'], ascending=True)
for tcol in [x for x in cons_df.columns if x not in ['ticker', 'date','month_end_mark']]:
    cons_df[tcol] = cons_df.groupby('ticker')[tcol].fillna(method='ffill')
    
# 一致预期会有多年的预测，选择最近的一年
cons_df = cons_df.query("forecast_year_analyst==forecast_year")
cons_df = cons_df.query("month_end_mark==1")

# 分析师一致预期单位为万
cons_df['forecast_analyst'] = cons_df['forecast_analyst']*10000

# 预测全年值减去已发布的公告值
cons_df['forecast_remain'] = np.where(cons_df['reportType']=='A', cons_df['forecast_analyst'], cons_df['forecast_analyst']-cons_df['NIncomeAttrP'])

# 乘以比例，得到预测季度的预测值
cons_df['fore_q_analyst'] = cons_df['ratio']*cons_df['forecast_remain']
cons_df = cons_df.query('month_end_mark == 1')
cons_df = cons_df[['ticker', 'date', 'forecast_q', 'fore_q_analyst']]

print u'基于分析师一致预期年度预测，利用季度占比推算得到的季度预测值为：'
print cons_df.head().to_html()

tend = time.time()
print u'此部分耗时:%s seconds'%(round(tend-tstart,2))

# 分析覆盖度
tcol = 'fore_q_analyst'
predict_dframe = cons_df[['ticker', 'date', tcol]]
predict_dframe['date'] = cons_df['date'].apply(lambda x: x.replace("-", ""))
print u'%s 计算得到的非空覆盖度股票数:'%tcol

coverd_stock_df = predict_dframe.dropna().groupby(['date'])['ticker'].count().reset_index()
coverd_stock_df.columns = ['date', 'covered_ticker_count']
_ = plot_coverage(coverd_stock_df)


'''

从图中来看，在每年的Q1都有较大的覆盖波动，原因是一致预期中包括了公司的业绩预告数据，业绩预告在Q1发布的比较多，因此会出现覆盖度的毛刺。在上面一致预期的覆盖度图中之所以没有毛刺，是因为上面算一致预期覆盖度时，增加了预测年等于自然年的条件把业绩预告的数据过滤掉了

    
文档
2.4 通过简约模型来推算
参考天风证券在研报中的做法，用过去4期的平均季度盈利加上上年同期的盈利来作为预测季度的值

'''

# 简约模型

# 过去4期季度的平均值
mean_q_df = earning_df[['ticker', 'publishDate', 'Q_NAttrP', 'Q_flag', 'endDate']].sort_values(by=['ticker', 'Q_flag'], ascending=True)
mean_q_df['rolling_mean'] = mean_q_df.rolling(4, min_periods=1)['Q_NAttrP'].mean().values

aligned_earning_df = aligned_earning_df.merge(mean_q_df[['ticker','publishDate','Q_flag', 'rolling_mean']].rename(columns={"publishDate":"date", "Q_flag":"T_n"}), on=['ticker', 'date', 'T_n'], how='left')

aligned_earning_df['fore_q_simple'] = aligned_earning_df['rolling_mean'] + aligned_earning_df['P_T-1_n+1']
aligned_earning_df.head()


'''

2.5 汇总得到季度预测数据
按照前文的逻辑，按照 预告、线性外推、分析师预期、简约模型的顺序来得到季度盈利预测值

'''

tstart = time.time()
##### part1: 得到基于财报、业绩预告得到的预测数据，该类预测数据有个特点是发生时间不固定,但都是在财报发布日
# 合并业绩预告数据和基于财报的推测数据
forecast_q_data = aligned_earning_df[['ticker', 'date', 'T_n+1', 'is_base_normal',u'is_history_stable',u'is_prevyq_normal', u'fore_q_equalratio', u'fore_q_equalgrowth',u'fore_q_simple']].rename(columns={"T_n+1":"financial_forecast_q", "date":"financial_date"})
# 保持原有的nan
forecast_q_data = forecast_q_data.replace(np.nan, 'NULL_FILL')
pre_forecast_q_df = pre_forecast_q_df.replace(np.nan, 'NULL_FILL')


# 合并季度预测数据
forecast_q_data = pd.concat([forecast_q_data, pre_forecast_q_df.rename(columns={"publishDate":"annouce_date", 'report':"annouce_forecast_q"})], axis=0)


# 前向填充数据
forecast_q_data['event_date'] = forecast_q_data[['annouce_date', 'financial_date']].fillna('3030-01-01').min(axis=1).values
forecast_q_data = forecast_q_data.sort_values(by=['ticker', 'event_date'], ascending=True)
for tcol in [x for x in forecast_q_data.columns if x not in ['ticker']]:
        forecast_q_data[tcol] = forecast_q_data.groupby(['ticker'])[tcol].fillna(method='ffill').values
        
# 清理和对齐数据
## 当业绩预告的预测期低于财报的预测期时，说明发了新的财报，业绩预告无效了
for tcol in ['annouce_date', 'annouce_forecast_q', 'fore_q_annouce']:
    forecast_q_data[tcol] = np.where(forecast_q_data['annouce_forecast_q'].replace("NULL_FILL", '2000-01-01').fillna('2000-01-01')<forecast_q_data['financial_forecast_q'].replace("NULL_FILL", '2000-01-01').fillna('2000-01-01'), np.nan, forecast_q_data[tcol])

## 当基于财报的预测期小于业绩预告的预测期时，将基于财报做的预测都置空
for tcol in ['financial_forecast_q', 'fore_q_equalgrowth', 'fore_q_equalratio', 'fore_q_simple', 'is_base_normal', 'is_history_stable', 'is_prevyq_normal']:
    forecast_q_data[tcol] = np.where(forecast_q_data['annouce_forecast_q'].replace("NULL_FILL", '2000-01-01').fillna('2000-01-01')<=forecast_q_data['financial_forecast_q'].replace("NULL_FILL", '2000-01-01').fillna('2000-01-01'), forecast_q_data[tcol], np.nan)

forecast_q_data['fore_q'] = forecast_q_data['financial_forecast_q'].fillna(forecast_q_data['annouce_forecast_q'])

forecast_q_data = forecast_q_data.replace(np.nan, 'NULL_FILL')

## 对齐后的预测数据
forecast_q_data = forecast_q_data[['ticker','event_date', 'fore_q', 'fore_q_annouce','fore_q_equalgrowth', 'fore_q_equalratio', 'fore_q_simple', 'is_base_normal', 'is_history_stable', 'is_prevyq_normal']].rename(columns={"event_date":"date"})
forecast_q_data = forecast_q_data.drop_duplicates(subset=['ticker', 'date'], keep='last')
print u'基于财报和预告得到的预测数据原始数据格式如下:'
print forecast_q_data.query("ticker=='000001'").tail().to_html()

## 将基于财报的、不定期的预测数据，转换为一个月一个数据，用前向填充的方式

# 取月行情数据，并得到对齐的需要输出的dataframe结构
mret_df = pd.read_pickle(os.path.join(data_save_dir, 'mret.pkl'))  # 列为 ticker, tradeDate("%Y%m%d")
mret_df['end_mark'] = 1
mret_df = mret_df[['ticker','tradeDate','end_mark']].rename(columns={"tradeDate":"date"})

forecast_q_data['date'] = forecast_q_data['date'].apply(lambda x: x.replace("-", ""))
forecast_q_data = pit2fixedfreq(forecast_q_data, mret_df)


# ##### part2: 对齐一致预期季度预测数据

# 合并数据
forecast_data = cons_df.merge(forecast_q_data.replace(np.nan, 'NULL_FILL'), on=['ticker','date'], how='outer')

# 剔除预测季度不一致的情况
forecast_data['forecast_q'] = forecast_data['forecast_q'].fillna(forecast_data['fore_q'])
forecast_data['fore_q'] = forecast_data['fore_q'].fillna(forecast_data['forecast_q'])
forecast_data = forecast_data[(forecast_data["fore_q"]==forecast_data["forecast_q"])]
forecast_data = forecast_data[['ticker','date', 'fore_q', 'fore_q_annouce','fore_q_equalgrowth', 'fore_q_equalratio', 'fore_q_analyst','fore_q_simple', 'is_base_normal', 'is_history_stable', 'is_prevyq_normal']]

# 恢复NULL_FILL为nan
forecast_data = forecast_data.replace('NULL_FILL', np.nan)
forecast_data = forecast_data.dropna(subset=['fore_q'])

print u'所有季度预测对齐后的数据样式为:'
print forecast_data.head().to_html()


forecast_data.to_pickle(os.path.join(data_save_dir, "forecast_data.pkl"))

tend = time.time()
print u'此部分耗时:%s seconds'%(round(tend-tstart,2))


'''

第三部分：预测效果分析
该部分耗时 5分钟
比较一下以上几种方法的效果，比较的方法具体如下：

从预测准确度来看，各种方法的预测效果；
从因子效果来看，各种方法的预测效果；
参考天风证券研报中的做法，将多种预测方法得到的值按照一定逻辑合并成一份预测数据，对比合并后数据的准确性和因子效果
预测误差的定义:
error=abs((预测值−真实值)/真实值)
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''
forecast_data = pd.read_pickle(os.path.join(data_save_dir, "forecast_data.pkl"))
forecast_data = forecast_data.merge(earning_df[['ticker','Q_flag','Q_NAttrP']].rename(columns={"Q_flag":"fore_q",'Q_NAttrP':"real_q_value"}),
                                    on=['ticker','fore_q'],how='left')


'''

3.1 不同方法得到的季度预测准确性

'''
tstart = time.time()
name_dict = {
    "fore_q_annouce":u'业绩预告平均值',
    "fore_q_equalgrowth":u'历史业绩等增长率推算',
    "fore_q_equalratio":u'历史业绩等比例推算',
    "fore_q_analyst":u'分析师年度预测中拆解推算',
    "fore_q_simple":u'简单外推'
}

print u'############ 各种方法的预测准确性如下:'
for forecast_v in ['fore_q_annouce','fore_q_equalgrowth', 'fore_q_equalratio', 'fore_q_analyst', 'fore_q_simple']:
    print forecast_v, name_dict[forecast_v]
    fore_df = forecast_data[['ticker','date',forecast_v,'real_q_value', 'fore_q']].dropna()
    fore_df['error'] = abs(fore_df[forecast_v] - fore_df['real_q_value'])/abs(fore_df['real_q_value'])
    fore_df['q'] = fore_df['fore_q'].apply(lambda x: x[-2:])
    ## 按季度统计预测误差
    print fore_df.groupby(['q'])['error'].median().reset_index().to_html()
    ## 因为2021Q2还没有（写深度报告时，在20210430日计算forecast Q2误差的样本数太少，引起结果异常，故去掉）
    plot_df = fore_df.query("date<'20210430'").groupby(['date'])['error'].describe().reset_index().pivot_table(index=['date'],columns=['level_1'],values='error', aggfunc='mean')['50%']
    _ = plot_df.plot(figsize=(15,5))
    _ = plt.show()

tend = time.time()
print u'此部分耗时:%s seconds'%(round(tend-tstart,2))


'''

从上图的结果来看

历史业绩等增长率推算、历史业绩等比例推算的误差范围总体较小，而业绩预告的误差极小值更小，分析师的年度预测拆解相对中规中矩，而简单推算的误差结果最大，因此在综合多个方法得到的季度一致预期推算时，也应该按照误差从小到大的数据采纳顺序。此外，不同数据的误差高和低的位置是错开的，因此合并多个预测方法从逻辑上来说是可以取长补短，获得更好的综合表现
总体来看，Q3和Q4季度的预测误差更大，因此误差曲线时间序列呈现出周期性波动的特点
    
文档
3.2 不同方法得到的预测数据因子效果
此处测试预期EP因子和预期季度利润增长率因子，因子定义如下：
EP1Q=下季度利润预期值当前流通市值

EPTTM=包括下季度利润预期值的年度TTM预测值当前流通市值

Grh1Q=abs(下季度利润预期值−去年下季度同期利润真实值去年下季度同期利润真实值)

GrhTTM=abs(包括下季度利润预期值的年度TTM预测值−去年同期利润TTM真实值去年同期利润TTM真实值)


'''
tstart = time.time()
# 获取市值数据
cap_df_list = []
for tdate in month_end_list:
    day_df = DataAPI.MktEqudGet(ticker=all_stock_list,tradeDate=tdate,field=u"ticker,tradeDate,negMarketValue,PE",pandas="1")
    cap_df_list.append(day_df)

cap_df = pd.concat(cap_df_list, axis=0)
cap_df['date'] = cap_df['tradeDate'].apply(lambda x: x.replace("-", ""))

# 把forecast_data和市值数据合并到一起
EP_factor_df = forecast_data.merge(cap_df[['ticker','date','PE','negMarketValue']], on=['ticker','date'], how='inner')

# 计算EP1Q因子
forecast_col = ['fore_q_annouce','fore_q_equalgrowth', 'fore_q_equalratio','fore_q_analyst', 'fore_q_simple']
for factor_col in forecast_col:
    EP_factor_df['%s_EP1Q'%factor_col] = EP_factor_df[factor_col]/EP_factor_df['negMarketValue']
EP_NX1Q_df = EP_factor_df[['ticker','date', 'PE'] + ['%s_EP1Q'%x for x in forecast_col]]
EP_NX1Q_df['EP'] = 1/EP_NX1Q_df['PE']
del EP_NX1Q_df['PE']
EP_NX1Q_df.head()

ttm_fore_df = EP_factor_df.copy()

def get_previous_q(inq, prev_n):
    '''
    inq: 如 2019Q2
    prev_n， 前面第prev_n个季度
    '''
    current_year = int(inq[:4])
    current_q = int(inq[-1])
    minus_year = int(prev_n)/4
    minus_q = int(prev_n)%4
    
    target_year = current_year - minus_year
    if current_q > minus_q:
        target_q = current_q - minus_q
    else:
        target_year -= 1
        target_q = current_q + 4 - minus_q
    return "%sQ%s"%(target_year, target_q)
    
    
## 计算剩下3个因子
quarter_df = earning_df[['ticker','Q_flag','Q_NAttrP']]    

for i in range(1, 8):
    date_tag = 'date_q-%s'%i
    value_tag = 'q-%s_v'%i
    
    # q-i，指的是未来一个季度前面第i个季度
    ttm_fore_df[date_tag] = ttm_fore_df['fore_q'].apply(lambda x: get_previous_q(x, i))
    merge_df = quarter_df.rename(columns={"Q_flag":date_tag, "Q_NAttrP":value_tag})
    # q-i季度的季度财报值
    ttm_fore_df = ttm_fore_df.merge(merge_df, on=['ticker', date_tag], how='left')
    del ttm_fore_df[date_tag]

ttm_fore_df['last_y_ttm'] = ttm_fore_df[['q-4_v', 'q-5_v','q-6_v', 'q-7_v']].sum(axis=1, skipna=False)
for factor_col in forecast_col:
    ttm_fore_df['%s_ttm'%factor_col] = ttm_fore_df[[factor_col, 'q-1_v', 'q-2_v','q-3_v']].sum(axis=1, skipna=False)
    ttm_fore_df['%s_EPTTM'%factor_col] = ttm_fore_df['%s_ttm'%factor_col] / ttm_fore_df['negMarketValue']
    ttm_fore_df['%s_Grh1Q'%factor_col] = ((ttm_fore_df[factor_col] - ttm_fore_df['q-4_v'])/ ttm_fore_df['q-4_v']).abs()
    ttm_fore_df['%s_GrhTTM'%factor_col] = ((ttm_fore_df['%s_ttm'%factor_col] - ttm_fore_df['last_y_ttm'])/ ttm_fore_df['last_y_ttm']).abs()

# 因子合并，并测试这些因子的表现
forcast_factor_df = EP_NX1Q_df.merge(ttm_fore_df[['ticker','date']+[x for x in ttm_fore_df.columns if (('EPTTM' in x) or ('Grh' in x))]], on=['ticker','date'], how='outer')

forcast_factor_df = forcast_factor_df.query("date>=@factor_data_sdate").query("date<=@factor_data_edate")

print "季度预测数据得到的因子测试结果..."
ic_summary_list = []
bucket_summary_list = []
start_time = time.time()
for factor_demo in ['METHOD_EP1Q', 'METHOD_EPTTM', 'METHOD_Grh1Q', 'METHOD_GrhTTM']:
    factor_series = factor_demo.replace("_METHOD", "").replace("METHOD_", "")
    print '## 测试%s系列的因子...'%factor_series
    factor_list = []
    for pred_method in forecast_col:
        factor_list.append(factor_demo.replace("METHOD", pred_method))
    print '因子列表为:', factor_list

    # 因子处理(剔除禁止股票池、去极值、行业市值中性化、标准化)
    c_p_df = fac_process(forcast_factor_df.rename(columns={"date":"tradeDate"}), factor_list, forbidden_pool, neutral=1)
    # IC测试统计
    ic_df = ic_test_summary(c_p_df, mret_df_real, 'nxt1m_ret', factor_list)
    print "IC测试结果：",ic_df.to_html()
    ic_summary_list.append(ic_df)
    # print "分组测试分析..."
    bucket_df = bucket_performance(c_p_df, mret_df_real, factor_list, factor_list,'nxt1m_ret', 5)
    bucket_summary_list.append(bucket_df)
    print "耗时: %s seconds" % (time.time() - start_time)

bucket_summary_df = pd.concat(bucket_summary_list, axis=0)
ic_summary_df = pd.concat(ic_summary_list, axis=0)
bucket_summary_df.to_pickle(os.path.join(data_save_dir, "bucket_summary_df.pkl"))
ic_summary_df.to_pickle(os.path.join(data_save_dir, "ic_summary_df.pkl"))

tend = time.time()
print u'此部分耗时:%s seconds'%(round(tend-tstart,2))


'''

此处预测因子的覆盖度和前文的业绩预测数据覆盖度不太一样，这是因为处理时候合并多个预测值时，增加了对预测的有效性筛选功能，以000036的equalgrowth的预测为例：

在2018-10-31，公司发布了2018Q3季报，对应的预测季度是2018Q4；
在2019-03-20，公司发布了2019Q1的预告；
因此只考虑equalgrowth时，在2019年3月底，依然有2018Q4的预测值;
而合并预告后，在2019-03-20日就会将2018Q4的预测数据置为无效；
测试结果中，覆盖度进行比较，

很显然是simple线性外推方法得到的覆盖度最高，平均在90%左右，其次是equalgrowth和equalratio方法得到的因子，覆盖度在80-90%左右，再是分析师季度预测因子，覆盖度在50-60%左右，最后是业绩预告因子；
从因子效果来看，

相对来说，业绩预告数据得到的因子效果最好，EP1Q因子IC为3.87%，EPTTM因子IC为3.33%，Grh1Q因子IC为1.56%， GrhTTM因子IC为0.34%，但覆盖度明显不足以用在量化策略上，需要叠加其它数据使用；
equalgrowth和equalratio因子在EP类因子表现其次，而分析师因子和线性外推因子在Grh因子表现其次

3.3 合并不同方法得到的预测数据和对应因子效果
'''
tstart = time.time()

forecast_data = pd.read_pickle(os.path.join(data_save_dir, "forecast_data.pkl"))
# 合并的预测数据列
forecast_data['combine_fore_q'] = np.nan
# 合并方法标识:1,2,3,4,5或者0（代表所有预测值都为空）
forecast_data['combine_flag'] = np.nan


# 第一优先级，业绩预告,combine_flag=1
forecast_data['combine_fore_q'] = forecast_data['combine_fore_q'].fillna(forecast_data['fore_q_annouce'])
forecast_data['combine_flag'] = (forecast_data['fore_q_annouce'].notnull())*1 #预告不为空，则flag为1，否则为0
forecast_data['combine_flag'] = forecast_data['combine_flag'].replace(0, np.nan)

# 其次，基期盈利正常，且历史盈利分布稳定，可以用等比例外推,combine_flag=2
forecast_data['tmp_col']=np.where((forecast_data['is_base_normal']==1) & (forecast_data['is_history_stable']==1), forecast_data['fore_q_equalratio'],forecast_data['combine_fore_q'])
forecast_data['combine_fore_q'] = forecast_data['combine_fore_q'].fillna(forecast_data['tmp_col'])

forecast_data['tmp_col']=np.where((forecast_data['is_base_normal']==1) & (forecast_data['is_history_stable']==1), (forecast_data['fore_q_equalratio'].notnull())*2,forecast_data['combine_flag'])
forecast_data['combine_flag'] = forecast_data['combine_flag'].fillna(forecast_data['tmp_col'])
forecast_data['combine_flag'] = forecast_data['combine_flag'].replace(0, np.nan)


# 再次，基期盈利正常，且历史盈利分布不稳定，且上期盈利稳定，可以用等比例增长,combine_flag=3
forecast_data['tmp_col']=np.where((forecast_data['is_base_normal']==1) & (forecast_data['is_history_stable']==0) & (forecast_data['is_prevyq_normal']==1), forecast_data['fore_q_equalgrowth'],forecast_data['combine_fore_q'])
forecast_data['combine_fore_q'] = forecast_data['combine_fore_q'].fillna(forecast_data['tmp_col'])

forecast_data['tmp_col']=np.where((forecast_data['is_base_normal']==1) & (forecast_data['is_history_stable']==0) & (forecast_data['is_prevyq_normal']==1), (forecast_data['fore_q_equalgrowth'].notnull())*3,forecast_data['combine_flag'])
forecast_data['combine_flag'] = forecast_data['combine_flag'].fillna(forecast_data['tmp_col'])
forecast_data['combine_flag'] = forecast_data['combine_flag'].replace(0, np.nan)


# 分析师覆盖,combine_flag=4
forecast_data['combine_fore_q'] = forecast_data['combine_fore_q'].fillna(forecast_data['fore_q_analyst'])
forecast_data['combine_flag'] = forecast_data['combine_flag'].fillna(forecast_data['fore_q_analyst'].notnull()*4)
forecast_data['combine_flag'] = forecast_data['combine_flag'].replace(0, np.nan)

# 最后，简单外推，combine_flag=5
forecast_data['combine_fore_q'] = forecast_data['combine_fore_q'].fillna(forecast_data['fore_q_simple'])
forecast_data['combine_flag'] = forecast_data['combine_flag'].fillna(forecast_data['fore_q_simple'].notnull()*5)
del forecast_data['tmp_col']
forecast_data = forecast_data.query("date>=@factor_data_sdate").query("date<=@factor_data_edate")
print u'合并后的数据样式为：'
print forecast_data.head().to_html()


tend = time.time()
print u'此部分耗时:%s seconds'%(round(tend-tstart,2))


'''

3.3.1 计算每期的合并标识占比

'''

total_count = forecast_data.groupby(['date'])['ticker'].count().reset_index()
total_count.columns = ['date', 'total_count']
total_count = total_count.set_index('date')

flag_count = forecast_data.groupby(['date', 'combine_flag'])['ticker'].count().reset_index()
flag_count.columns = ['date', 'combine_flag', 'count']
flag_count = flag_count.set_index('date')
flag_count['flag_ratio'] = flag_count['count']*1.0/total_count['total_count']

flag_count_df = flag_count.reset_index().pivot_table(index='date', columns='combine_flag', values=['flag_ratio']).fillna(0)
flag_count_df.columns=pd.MultiIndex.from_product([['flag_ratio'], ['NO_VALUE', 'fore_annouce', 'fore_equalratio', 'fore_equalgrowth', 'fore_analyst', 'fore_simple']])

ax = flag_count_df['flag_ratio'].plot.bar(stacked=True, title="combine_flag ratio",figsize=(20,5))
_ = ax.legend(ncol=6,loc='upper left')


'''

从上图可以看出来，2019年以来，业绩预告所占的比例越来越小，而equalratio和equalgrowth外推所占的比例变大，分析师的比例在2021年明显变大

    
文档
3.3.2 合并后的预告数据准确性水平

'''

forecast_data = forecast_data.merge(earning_df[['ticker','Q_flag','Q_NAttrP']].rename(columns={"Q_flag":"fore_q",'Q_NAttrP':"real_q_value"}),
                                    on=['ticker','fore_q'],how='left')

forecast_v = 'combine_fore_q'
fore_df = forecast_data[['ticker','date',forecast_v,'real_q_value', 'fore_q']].dropna()
fore_df['error'] = abs(fore_df[forecast_v] - fore_df['real_q_value'])/abs(fore_df['real_q_value'])
fore_df['q'] = fore_df['fore_q'].apply(lambda x: x[-2:])
## 按季度统计预测误差
print fore_df.groupby(['q'])['error'].median().reset_index().to_html()
plot_df = fore_df.query("date<'20210430'").groupby(['date'])['error'].describe().reset_index().pivot_table(index=['date'],columns=['level_1'],values='error', aggfunc='mean')['50%']
_ = plot_df.plot(figsize=(15,5))
_ = plt.show()

'''

从分季度的误差来看，

Q1误差中位数为0.318，低于业绩预告的0.067，但比上述几个方法中次好的分析师0.426的误差更小；
Q2误差中位数为0.348，低于业绩预告的0.150，但比上述几个方法中次好的分析师0.483的误差更小；
Q3误差中位数为0.424，低于业绩预告的0.229，但比上述几个方法中次好的分析师和equalratio的0.52的误差更小；
Q4误差中位数为0.867，同上述几个方法中最好的equalratio的0.820的误差非常接近；
而合成后的预测数据，覆盖度远远大于业绩预告，是所有预测数据中最高的，因此从覆盖度和预测误差的角度来看，合成后的预测数据效果更好

    
文档
3.3.3 合并后的预告数据的因子效果

'''
# 把forecast_data和市值数据合并到一起
combine_factor_df = forecast_data.merge(cap_df[['ticker','date','PE','negMarketValue']], on=['ticker','date'], how='inner')

# 计算EP1Q因子
forecast_col ='combine_fore_q'
combine_factor_df['%s_EP1Q'%forecast_col] = combine_factor_df[forecast_col]/combine_factor_df['negMarketValue']

# 计算剩下3个因子
for i in range(1, 8):
    date_tag = 'date_q-%s'%i
    value_tag = 'q-%s_v'%i
    
    # q-i，指的是未来一个季度前面第i个季度
    combine_factor_df[date_tag] = combine_factor_df['fore_q'].apply(lambda x: get_previous_q(x, i))
    merge_df = quarter_df.rename(columns={"Q_flag":date_tag, "Q_NAttrP":value_tag})
    # q-i季度的季度财报值
    combine_factor_df = combine_factor_df.merge(merge_df, on=['ticker', date_tag], how='left')
    del combine_factor_df[date_tag]

combine_factor_df['last_y_ttm'] = combine_factor_df[['q-4_v', 'q-5_v','q-6_v', 'q-7_v']].sum(axis=1, skipna=False)

combine_factor_df['%s_ttm'%forecast_col] = combine_factor_df[[forecast_col, 'q-1_v', 'q-2_v','q-3_v']].sum(axis=1, skipna=False)
combine_factor_df['%s_EPTTM'%forecast_col] = combine_factor_df['%s_ttm'%forecast_col] / combine_factor_df['negMarketValue']
combine_factor_df['%s_Grh1Q'%forecast_col] = ((combine_factor_df[forecast_col] - combine_factor_df['q-4_v'])/ combine_factor_df['q-4_v']).abs()
combine_factor_df['%s_GrhTTM'%forecast_col] = ((combine_factor_df['%s_ttm'%forecast_col] - combine_factor_df['last_y_ttm'])/ combine_factor_df['last_y_ttm']).abs()

# 因子合并，并测试这些因子的表现
combine_factor_df = combine_factor_df[['ticker','date']+[x for x in combine_factor_df.columns if (('EP' in x) or ('Grh' in x)) and (forecast_col in x)]]

combine_factor_df = combine_factor_df.query("date>=@factor_data_sdate").query("date<=@factor_data_edate")
print "合并后的季度预测数据得到的因子测试结果..."
factor_list = []
for factor_demo in ['METHOD_EP1Q', 'METHOD_EPTTM', 'METHOD_Grh1Q', 'METHOD_GrhTTM']:
    factor_list.append(factor_demo.replace("METHOD", forecast_col))
print '因子列表为:', factor_list

# 因子处理(剔除禁止股票池、去极值、行业市值中性化、标准化)
c_p_df = fac_process(combine_factor_df.rename(columns={"date":"tradeDate"}), factor_list, forbidden_pool, neutral=1)
# IC测试统计
c_ic_res = ic_test_summary(c_p_df, mret_df_real, 'nxt1m_ret', factor_list)
print "IC测试结果：", c_ic_res.to_html()
combine_bucket_df = bucket_performance(c_p_df, mret_df_real, factor_list, factor_list,'nxt1m_ret', 5)

c_ic_res.to_pickle(os.path.join(data_save_dir, "combine_ic.pkl"))
combine_bucket_df.to_pickle(os.path.join(data_save_dir, "combine_bucket.pkl"))



## 把所有因子的IC合并到一起
ic_total_df = pd.concat([c_ic_res, ic_summary_df], axis=0)
## 把所有因子的分组多空合并到一起
bucket_total_df = pd.concat([combine_bucket_df, bucket_summary_df], axis=0)[['多空组合年化收益率','多空组合夏普率']]
## 把IC和多空合并到一起
factor_perf_df = pd.concat([ic_total_df, bucket_total_df], axis=1)
## 标识对应的因子名
factor_perf_df['factor'] = [x.split("_")[-1] for x in factor_perf_df.index.values]
## 展示因子表现
factor_perf_df.sort_values(by=['factor', 'ICIR'], ascending=[False, False])


'''

从上面的汇总结果来看，以ICIR为衡量标准：

GrhTTM因子，合成后的季度预测数据因子排名第4，ICIR为0.09，整体来看，季度预测类数据的表现都不太好，equal线性外推的数据因子表现最差；
Grh1Q因子，同GrhTTM类似，季度预测类数据的因子表现也都不太好，合成后的季度预测数据因子排名第3，ICIR为0.45，equal线性外推的数据因子表现差；
EPTTM因子，季度预测类数据得到的因子效果都还不错，合成后的季度预测数据因子排名第3，仅次于equalratio和equalgrowth得到的因子，ICIR为1.43， IC有2.78%，多空年化10%；
EP1Q因子，和EPTTM类似，因子表现效果较好，合成后的季度预测数据因子排名第3，ICIR为1.26，IC有2.09%，多空年化7.71%；
综合来看，在各种季度预测数据中，合成后的季度预测数据的因子表现比较平衡，无论是效果不佳的因子构建方法（Grh类）还是效果不错的因子构建方法（EP类），合成后的季度预测数据因子都排名在3左右。

    
文档
第四部分：总结
    
文档
本文对季度预测数据的几种方法进行了实证，从覆盖度、预测误差、因子效果等几个角度进行了多维度对比，从结果来看，合并多种季度预测数据得到的季度预测数据具有最优的预测误差（在保证一定覆盖度的条件下），且因子效果也是多种数据中最稳定的数据，具有不错的量化效果。不足之处在于，合并多种季度预测数据得到的因子效果并不是所有因子中最优的，不同季度预测数据的合并方法还有较大的空间进行进一步优化。



'''




