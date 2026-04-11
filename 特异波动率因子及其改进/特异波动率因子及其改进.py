# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:28:47 2020

@author: Asus
"""

第一部分：数据下载和处理
该部分耗时 90分钟
该部分内容为：

读取禁投股票池。

读取从2010-2019的A股所有个股的前复权价格数据、2010-2019的中证全指的每日收盘价数据、2009-2019年每天的市值因子收益率、估值因子收益率和流动性因子收益率。

计算普通波动率因子、特异率因子、特异波动率因子、新特异率和新特异波动率因子。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

1.1 数据下载

根据后续研究的需要，从优矿中下载的数据有：每天的因子收益率(size, btop, liquidity)数据；个股2009-2019的个股前复权收盘价数据；市场整体收益数据（以000985中证全指为代表）

import pandas as pd, numpy as np, os, pickle, time
from collections import Counter
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from scipy import stats
from CAL.PyCAL import *
mpl.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei'] 

#建立数据存储文件夹
special_vol_dir = 'special_vol'
if not os.path.exists(special_vol_dir):
    os.makedirs(special_vol_dir)
    
    
#记录的禁止股票池以及其他设置

data_start_date = '2009-10-01'
data_end_date = '2019-12-01'

# 获取全A的secID
all_secid_list = DataAPI.EquGet(equTypeCD=u"A", listStatusCD=u"L,S,DE", field=u"secID",pandas="1")['secID'].tolist()
all_secid_list.remove('DY600018.XSHG')
all_secid_list = [x for x in all_secid_list if len(x) == 11]

# 获取交易日历
trade_calendar = DataAPI.TradeCalGet(exchangeCD=U"XSHG", field=u"calendarDate,isOpen,isMonthEnd")
date_list = trade_calendar.query("calendarDate>=@data_start_date").query("calendarDate<=@data_end_date").query("isOpen==1")['calendarDate'].tolist()
# 月末交易日
month_end_list = trade_calendar.query("calendarDate>=@data_start_date").query("calendarDate<=@data_end_date").query("isMonthEnd==1")['calendarDate'].tolist()

t1 = time.time()
print"计算禁止池，包括上市不足60个交易日的次新股、ST股以及停牌个股..."

# 获得交易日历
calendar = trade_calendar[trade_calendar['isOpen'] == 1]
calendar = calendar['calendarDate'].tolist()

# 次新股
ipo_info = DataAPI.SecIDGet(assetClass=u"E", field=['ticker', 'listDate'], pandas="1")
ipo_info.dropna(inplace=True)
ticker_list = [ticker for ticker in ipo_info['ticker'] if len(ticker) == 6 and ticker[0] in ['0', '3', '6']]
ipo_info = ipo_info[ipo_info['ticker'].isin(ticker_list)]
ipo_info['permit_date'] = [calendar[calendar.index(date) + 60] if date in calendar else  calendar[0] for date in ipo_info['listDate']]

calendar = np.array(calendar)
new_df = pd.DataFrame()
for date in calendar[(calendar > data_start_date) & (calendar < data_end_date)]:
    new_list = ipo_info[(ipo_info['permit_date'] >= date) & (ipo_info['listDate'] <= date)]['ticker'].values
    d_new_df = pd.DataFrame({'tradeDate': [date] * len(new_list), 'ticker': new_list})
    new_df = new_df.append(d_new_df)

new_df['remove_flag'] = 'new'

# ST股
st_info = DataAPI.SecSTGet(beginDate=data_start_date,endDate=data_end_date,field=['tradeDate', 'ticker'],pandas="1")
st_df = st_info.copy()
st_df['remove_flag'] = 'st'

# 停牌
halt_info = DataAPI.SecHaltGet(beginDate=data_start_date, endDate=data_end_date, field=['ticker', 'haltBeginTime', 'haltEndTime'],pandas="1")
halt_info.fillna(calendar[-1], inplace=True)
halt_info['haltBeginTime'] = halt_info['haltBeginTime'].apply(lambda x: x[:10])
halt_info['haltEndTime'] = halt_info['haltEndTime'].apply(lambda x: x[:10])

halt_df = pd.DataFrame()
for date in calendar[(calendar > data_start_date) & (calendar < data_end_date)]:
    halt_list = halt_info[(halt_info['haltEndTime'] >= date) & (halt_info['haltBeginTime'] <= date)]['ticker'].values
    d_halt_df = pd.DataFrame({'tradeDate': [date] * len(halt_list), 'ticker': halt_list})
    halt_df = halt_df.append(d_halt_df)

halt_df['remove_flag'] = 'halt'

remove_df = new_df.append(st_df).append(halt_df)
remove_df = remove_df[['tradeDate', 'ticker', 'remove_flag']]

t2 = time.time()
print '记录的禁止股票池样式如下，耗时:%s seconds'%(t2 - t1)
print remove_df.head().to_html()


##下载个股2009-2019的个股前复权收盘价数据并存储,由于数据量较大，鉴于优矿返回数据条数的限制，这里分段进行数据获取。

start_time = time.time()
print"下载个股2009-2019的个股前复权收盘价数据并存储..."

start_date, end_date = '20090101', '20191001'
close_df = pd.DataFrame()
for year in range(2009, 2020):
    closedata1 = DataAPI.MktEqudAdjGet(secID=u"",ticker=u"",tradeDate=u"",beginDate=u"{}0101".format(year),endDate=u"{}0331".format(year),
                      isOpen="",field=u"ticker,tradeDate,closePrice",pandas="1")

    closedata2 = DataAPI.MktEqudAdjGet(secID=u"",ticker=u"",tradeDate=u"",beginDate=u"{}0401".format(year),endDate=u"{}0630".format(year),
                      isOpen="",field=u"ticker,tradeDate,closePrice",pandas="1")

    closedata3 = DataAPI.MktEqudAdjGet(secID=u"",ticker=u"",tradeDate=u"",beginDate=u"{}0701".format(year),endDate=u"{}0930".format(year),
                      isOpen="",field=u"ticker,tradeDate,closePrice",pandas="1")
    
    closedata4 = DataAPI.MktEqudAdjGet(secID=u"",ticker=u"",tradeDate=u"",beginDate=u"{}1001".format(year),endDate=u"{}1231".format(year),
                      isOpen="",field=u"ticker,tradeDate,closePrice",pandas="1")
    close_df = pd.concat([close_df, closedata1, closedata2, closedata3, closedata4])
with open('{}/close_adj.pkl'.format(special_vol_dir), 'w') as f:
    pickle.dump(close_df, f)
    
end_time = time.time()
print "Time cost: %s seconds" % (end_time - start_time)
print '个股2009-2019的个股前复权收盘价数据样式如下：'
print close_df.head().to_html()


##获取市场整体收益数据（以000985中证全指为代表）

start_time = time.time()
print"获取市场整体收益数据（以000985中证全指为代表）..."

start_date, end_date = '20090101', '20191231'
index_close_df = pd.DataFrame() 
ticker_list = ['000001','000985']
for ticker in ticker_list:
    t_df = DataAPI.MktIdxdGet(indexID=u"",ticker=ticker,tradeDate=u"",beginDate=start_date, endDate=end_date,
                   exchangeCD=u"XSHE,XSHG",field=["ticker", "tradeDate", "closeIndex"],pandas="1")
    index_close_df = pd.concat([index_close_df, t_df])
with open('{}/index_close.pkl'.format(special_vol_dir), 'w') as f:
    pickle.dump(index_close_df, f)

end_time = time.time()
print "Time cost: %s seconds" % (end_time - start_time)
print '市场整体收益数据样式如下：'
print index_close_df.head().to_html()

#获取每天的因子收益率(size, btop, liquidity)数据

start_time = time.time()
print"获取每天的因子收益率(size, btop, liquidity)数据...."

start_date, end_date = '20090101', '20191001'
factors = ['ticker', 'tradeDate', 'SIZE', 'BTOP', 'LIQUIDTY']
close_df = pd.DataFrame()
for year in range(2009, 2020):
    for month in range(1, 13):
        month = str(month).zfill(2)
        closedata1 = DataAPI.RMFactorRetDayGet(beginDate=u"{}{}01".format(year, month),endDate=u"{}{}32".format(year, month),tradeDate=u"",field=['tradeDate', "SIZE", 'BTOP', 'LIQUIDTY'],pandas="1")
        closedata1 = closedata1.sort_values(by='tradeDate')
        close_df = pd.concat([close_df, closedata1])
with open('{}/riskfactorret.pkl'.format(special_vol_dir), 'w') as f:
    pickle.dump(close_df, f)
    
end_time = time.time()
print "Time cost: %s seconds" % (end_time - start_time)
print '每天的因子收益率(size, btop, liquidity)数据样式如下：'
print close_df.head().to_html()

'''
1.2 数据处理

1.2.1 生成回归所需要的自变量和因变量数据

自变量是过去21天每日市场整体收益（用000905中证全指代替）和各个因子收益率（size, btop, liquidity）; 因变量是过去21天每日股票的收益。

'''


#获取个股每天的收益数据（线性回归中的Y）
start_time = time.time()
print("获取个股每天的收益数据（线性回归中的Y）....")

with open('{}/close_adj.pkl'.format(special_vol_dir), 'r') as f:
    close_adj = pickle.load(f)
close_adj = close_adj.pivot(index='tradeDate', columns='ticker', values='closePrice')
close_pct = close_adj.pct_change()
close_pct.index = [str(x).replace('-', '') for x in close_pct.index]
#禁止池中的股票删除（被禁止的股票当天收益不能得到）
close_pct_stack = close_pct.stack().reset_index()
close_pct_stack.columns = ['tradeDate', 'ticker', 'pct']
close_pct_stack_withremoveflag = close_pct_stack.merge(remove_df, on=['tradeDate', 'ticker'], how='left')
close_pct_stack = close_pct_stack_withremoveflag[pd.isnull(close_pct_stack_withremoveflag['remove_flag'])]
close_pct_stack = close_pct_stack[['tradeDate', 'ticker', 'pct']]
close_pct = close_pct_stack.pivot(index='tradeDate', columns='ticker', values='pct')

end_time = time.time()
print "Time cost: %s seconds" % (end_time - start_time)
print '个股每天的收益数据样式如下：'
print close_pct.head().to_html()

#获取指数每天的收益数据和每日各个因子收益率的数据（线性回归中的X）
start_time = time.time()
print("获取指数每天的收益数据和每日各个因子收益率的数据（线性回归中的X）....")

with open('{}/riskfactorret.pkl'.format(special_vol_dir), 'r') as f:
    riskfactorret = pickle.load(f)
with open('{}/index_close.pkl'.format(special_vol_dir), 'r') as f:
    index_close_df = pickle.load(f)
index_close_df = index_close_df.pivot(index='tradeDate', columns='ticker', values='closeIndex')
index_pct_df = index_close_df.pct_change()
index_pct_df.index = [str(x).replace("-", '') for x in index_pct_df.index]
index_pct_df.index.name = 'tradeDate'
index_pct_df = index_pct_df.reset_index()
#合并 风险因子收益 和 市场整体收益 作为线性回归中的X
X_concat = index_pct_df.merge(riskfactorret, on='tradeDate', how='left')
    
end_time = time.time()
print "Time cost: %s seconds" % (end_time - start_time)
print '指数每天的收益数据和每日各个因子收益率的数据样式如下：'
print X_concat.head().to_html()

'''

1.2.2 逐日逐股票进行回归得到拟合优度（该部分耗时较久，大概需要90分钟！）

在之前长江证券一篇研报《高频因子（五）：高频因子和交易行为》中，长江证券以特异率因子作为简单衡量个股交易行为异常的描述，其构建方式为对过去 21 天个股收益率做市场收益率、规模收益率和价值收益率的线性回归，取回归的拟合优度，以 1-拟合优度作为特异率因子，特异率因子值越大，个股存在更多的交易异常行为。
因为特异率因子本身是一个较为有效的因子（本文后续会证明），同时特异率因子也可以用来辅助计算特异波动率因子，所以逐日逐股票回归得到拟合优度数据进而得到特异率因子，可以帮助后文的进一步分析。

'''

start_time = time.time()
print("逐日逐股票进行回归得到拟合优度....")

# 获取交易日历
start_date, end_date = '2009-01-01', '2019-12-31'
trade_calendar = DataAPI.TradeCalGet(exchangeCD=U"XSHG", field=u"calendarDate,isOpen,isMonthEnd")
date_list = trade_calendar.query("calendarDate>=@start_date").query("calendarDate<=@end_date").query("isOpen==1")['calendarDate'].tolist()
date_list = [str(x).replace('-', '') for x in date_list]
# 月末交易日
month_end_list = trade_calendar.query("calendarDate>=@start_date").query("calendarDate<=@end_date").query("isMonthEnd==1")['calendarDate'].tolist()
month_end_list = [x.replace('-', '') for x in month_end_list]

#输入X和Y回归得到拟合优度也就是R2
def get_r2(y_se, x_df):
    """
    参数：
    x_df:       Dataframe, 包含线性回归中需要用到的X
    y_se：      Series,       包含线性回归中需要用到的Y
    输出：
    r2：        float，    线性回归的R2（拟合优度） 
    """
    regress_use_factors = list(set(x_df.columns) - set(['tradeDate']))
    y_se.index.name = 'tradeDate'
    y_se.name = 'close_pct'
    y_df = y_se.reset_index()
    regress_df = x_df.merge(y_df, on='tradeDate', how='left')
    regress_df = regress_df.dropna()
    #数据都是nan，无法回归，则返回nan
    if len(regress_df) == 0:
        return np.nan
    mod = sm.OLS(regress_df['close_pct'].values, sm.add_constant(regress_df[regress_use_factors].values))
    res = mod.fit()
    r2 = res.rsquared
    return r2

#逐日进行回归得到拟合优度
market_index = '000985'
use_factors_num_list = [3, 4]
for use_factors_num in use_factors_num_list: 
    if use_factors_num == 3:
        use_factors = ['SIZE', 'BTOP', market_index]
    else:
        use_factors = ['SIZE', 'BTOP', 'LIQUIDTY', market_index]
    special_df = pd.DataFrame()
    for month_end in month_end_list:
        date_before21 = date_list[date_list.index(month_end)-20]
        X_concat_part = X_concat[X_concat['tradeDate']>=date_before21][X_concat['tradeDate']<=month_end]
        #如果改天可回归数据长度为0，不回归，默认为np.nan
        if len(X_concat_part) == 0:
            continue
        close_pct_part = close_pct.loc[list(filter(lambda x: x>=date_before21 and x<= month_end, close_pct.index))]
        time0 = time.time()
        tmp = close_pct_part.apply(lambda x: get_r2(x, X_concat_part[use_factors+['tradeDate']]))
        tmp.name = month_end
        special_df = pd.concat([special_df, tmp], axis=1)

    #存储r2的数据
    with open('{}/special_strange_{}_{}.pkl'.format(special_vol_dir, market_index, use_factors_num), 'w') as f:
        pickle.dump(special_df, f)

end_time = time.time()
print "Time cost: %s seconds" % (end_time - start_time)

'''

第二部分：普通波动率、特异率和特异波动率的因子表现对比
该部分耗时 2分钟
该部分内容为：

生成特异率、普通波动率、特异波动率三个因子数据。

分别回测特异率、普通波动率、特异波动率三个因子的单因子表现。

特异率、普通波动率、特异波动率三个因子的对比表现。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

2.1 生成特异率、普通波动率、特异波动率三个因子数据

低波效应因子的投资逻辑在于由投资者的过度反应（羊群效应）带来的股价变动相对于市场异常现象的均值回归，当投资者对于个股在短期内存在非理性“追捧”时，个股波动迅速上升，相比于市场的价格变动也呈现同步性的异常，但这种非理性不可长期持续，最终个股价格会随着交易行为回归正常而回落至合理价格区间，波动率因子的构建即基于低波效应带来的确定性收益。 A 股市场个人投资者较多，非理性交易行为普遍存在，局部上存在特定个股的价格变动偏离市场变动，并在未来随交易行为理性而回归，且这种现象在不同规模、不同行业下均存在，这也是波动率因子成为 A 股市场中在各个宽基指数、行业指数下选股均有一定效果的少数量价因子的一个原因。
以过去 21 天收益率标准差作为基础波动率因子。
基础波动率因子的选股区分能力有限，原因在于构建因子时仅刻画了个股自身价格的异常变动，并非刻画个股价格相对于市场的异常变动。故目前市场上对低波效应的研究也多转向特异波动率，即以 fama 三因子模型为基础， 对过去 21 天个股收益率做市场收益率、规模因子收益率和价值因子收益率的线性回归，取残差计算标准差作为特异波动率因子。
以 fama 三因子模型为基础， 对过去 21 天个股收益率做市场收益率、规模因子收益率和价值因子收益率的线性回归，(1-拟合优度）作为特异率因子
经过数学推导可以证明：图片注释；采用该方法计算特异波动率可以简化计算过程。具体的推导过程如下： 图片注释 图片注释

'''

start_time = time.time()
print("生成特异率、普通波动率、特异波动率三个因子数据....")

#读取r2数据，生成特异率因子
with open('{}/special_strange_000985_3.pkl'.format(special_vol_dir), 'r') as f:
    r2_df = pickle.load(f)
r2_df = r2_df.T
special_df = 1 - r2_df

#生成普通波动率因子
vol_normal = pd.rolling_std(close_pct, window=21, min_periods=5)
vol_normal_part = vol_normal.loc[sorted(list(filter(lambda x: x in special_df.index, vol_normal.index)))]
vol_normal_part_stack = vol_normal_part.stack().reset_index()
vol_normal_part_stack.columns = ['tradeDate', 'ticker', 'vol_normal']

#生成特异波动率因子 
special_df_stack = special_df.stack().reset_index()
special_df_stack.columns = ['tradeDate', 'ticker', 'special']
concat = special_df_stack.merge(vol_normal_part_stack, on=['ticker', 'tradeDate'], how='left')
concat['vol_special'] = concat['vol_normal'] * np.sqrt(concat['special'])
#dropna
concat = concat.dropna()
#factor_old 是用市值因子、估值因子、市场收益做回归得到的各个因子
factor_old = concat

end_time = time.time()
print "Time cost: %s seconds" % (end_time - start_time)
print '特异率、普通波动率、特异波动率三个因子数据样式如下：'
print factor_old.head().to_html()


# 剔除次新股、ST股、停牌股
def remove_special_stocks(input_factor_df):
    '''
    input_factor_df: 因子值dataframe，列包括 ticker, tradeDate(%Y%m%d)
    '''
    factor_df = input_factor_df.copy()
    ori_len = len(factor_df)
    factor_df['tradeDate'] = factor_df['tradeDate'].apply(lambda x: x.replace("-", ""))
    # 剔除掉次新股、ST股、停牌股
    remove_df['tradeDate'] = remove_df['tradeDate'].apply(lambda x: x.replace("-", ""))
    factor_df = factor_df.merge(remove_df, on=['ticker', 'tradeDate'], how='left')
    factor_df = factor_df[factor_df['remove_flag'].isnull()]
    after_remove_len = len(factor_df)
    print u'剔除掉次新股、ST股、停牌股减少了%s条记录'%(ori_len-after_remove_len)
    return factor_df


# 提取出月末因子值dataframe
def extract_month_end_factor(in_factor_df, start_date, end_date, factor_name):
    '''
    in_factor_df: 因子值dataframe，至少包含列： ticker, tradeDate(str或者int格式), <factor_name>
    start_date/end_date: %Y%m%d
    return: 和输入同等列的dataframe， tradeDate格式为:%Y%m%d
    '''
    # 区间内的月末交易日期
    trade_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=start_date, endDate=end_date, field=u"", pandas="1")
    trade_date = trade_date[trade_date.isMonthEnd == 1]
    trade_date_list = [x.replace("-", "") for x in trade_date['calendarDate'].tolist()]
    # 筛选因子值
    in_factor_df['tradeDate'] = in_factor_df['tradeDate'].apply(lambda x: str(x).replace("-", ""))
    factor_df = in_factor_df[in_factor_df['tradeDate'].isin(trade_date_list)]
    return factor_df



# 获取个股的月度收益和下个月收益数据
def get_monthly_return(start_date, end_date):
    '''
    start_date/end_date:%Y%m%d
    返回：
    收益dataframe，列为: ticker, tradeDate(%Y%m%d, 月末日期), ret(当月收益), nx_ret(下月收益)
    '''
    chgframe = DataAPI.MktEqumAdjGet(beginDate=start_date, endDate=end_date, field=['ticker', 'endDate', 'chgPct'], pandas="1")
    chgframe['endDate'] = chgframe['endDate'].apply(lambda x: x.replace("-", ""))
    chgframe.columns = ['ticker', 'tradeDate', 'ret']
    chgframe = chgframe.sort_values(by=['ticker', 'tradeDate'], ascending=True)
    chgframe['nx_return'] = chgframe.groupby(['ticker'])['ret'].shift(-1)
    return chgframe


# 根据分组收益率序列，画图展示分组收益时间序列图和柱状图
def plot_group_perf(perf, quantile_num, direction=1):
    """
    perf: 分组收益,index为日期, columns为分组组名
    quantile_num: 分组数
    """
    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(211)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(212)

    for i in range(quantile_num):
        gperf = perf.iloc[:, i]
        ax1.plot(pd.to_datetime(gperf.index), (gperf + 1).cumprod(), label=gperf.name + u'左轴',
                 color=matplotlib.colors.cnames.values()[i])
    if direction ==1 :
        label = u'Top-Bottom(右轴)'
    else:
        label = u'Top-Bottom(右轴, 反向)'
        
    _ = ax2.plot(pd.to_datetime(perf.index), (perf['Top-Bottom'] + 1).cumprod(), label=label, color='k')
    _ = ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
    _ = ax1.set_title(u"%s档回测净值" % quantile_num, fontproperties=font, fontsize=16)
    _ = ax1.legend(loc=2, prop=font)
    _ = ax2.legend(loc=1, prop=font)
    _ = ax2.grid(False)

    label_dict = {u'Q1': u'Q1(High)', u'Q%s'%quantile_num: u'Q%s(Low)' % quantile_num}
    for gnroup_num in range(2, quantile_num):
        label_dict[u'Q%s'%gnroup_num] = u'Q%s' % gnroup_num

    nav = ((perf + 1).prod() - 1)[:quantile_num]
    ind = np.arange(quantile_num)
    _ = ax3.bar(ind + 0.2, nav, 0.3, color='r')
    _ = ax3.set_xlim((0, ind[-1] + 1))
    _ = ax3.set_xticks(ind + 0.35)
    _ = ax3.set_xticklabels([label_dict[x] for x in nav.index], fontproperties=font)
    _ = ax3.set_title(u"%s档回测累计绝对收益" % quantile_num, fontproperties=font, fontsize=16)
    _ = plt.show()
    return


# 进行简易分组，等权构建组合回测
def group_bt(factor_df, quantile_num, direction=1):
    """
    进行简易分组，等权构建组合回测。(组数越小，因子值越大)
    factor_df：信号dataframe，列至少包含： date("%Y%m%d", 或者"%Y-%m-%d, 或者int), ticker, value, return(下周期收益)
    direction: 1为正向信号的top-bottom, -1为负向信号的top-bottom
    return:
           perf: 每组每期的绝对收益率
    """
    factor_df = factor_df.dropna(subset=['value', 'return'])
    # 根据因子值进行分组
    factor_df['group'] = factor_df.groupby('date')['value'].apply(
        lambda x: 1.0 * (x.rank(method='first') - 1) / len(x) * quantile_num).astype(int)
    factor_df['group'] = quantile_num - factor_df['group']

    count_df = factor_df.groupby(['date', 'group']).apply(lambda x: len(x)).reset_index()
    count_df.columns = ['date', 'group', 'count']
    count_df['weight'] = 1.0 / count_df['count']

    # 等权构建组合
    bt_df = factor_df.merge(count_df[['date', 'group', 'weight']], on=['date', 'group'])
    perf = bt_df.groupby(['group', 'date']).apply(lambda x: sum(x['return'] * x['weight'])).reset_index()
    perf.columns = ['group', 'date', 'period_return']
    perf = perf.pivot_table(values='period_return', index='date', columns='group')

    perf = perf.sort_index()
    perf = perf.shift(1)
    perf.iloc[0, :] = 0
    perf.columns = ['Q' + str(i) for i in perf.columns]

    # 计算多空收益率
    if direction == 1:
        perf['Top-Bottom'] = (perf['Q1'] - perf['Q' + str(quantile_num)]) / 2.0
    else:
        perf['Top-Bottom'] = (perf['Q' + str(quantile_num)] - perf['Q1']) / 2.0
    return perf, bt_df


# 统计Top-Bottom组合的年化收益、回撤、IR等指标
def get_perf_summary(perf, col='Top-Bottom'):
    '''
    perf: index为日期，列至少包括<col>， 值为每一个月的period_return
    '''
    # 累计净值
    final_nav = (perf[col] + 1).cumprod().values[-1]
    total_month_len = len(perf)
    # 年化收益
    # annual_return = (final_nav-1)*12.0/total_month_len
    annual_return = 12*perf[col].mean()
    # 年化波动率
    annual_std = np.sqrt(12)*perf[col].std()
    # 信息比率
    ir = np.sqrt(12)*perf[col].mean()/perf[col].std()
    # 月度胜率
    hit_rate = 1.0*len(perf[perf[col]>=0])/len(perf)
    # 最大回撤
    nav_df = (perf[col] + 1).cumprod()
    cum_max = nav_df.cummax()
    maxdrawdown =((cum_max-nav_df)/cum_max).max()
    summary_df = pd.DataFrame()
    summary_df.loc[0, u'总收益'] = "%s"%(round(100*(final_nav-1), 2))+"%"
    summary_df.loc[0, u'区间长度(年)'] = total_month_len/12.0
    summary_df.loc[0, u'年化收益'] = "%s"%(round(100*annual_return, 2))+"%"
    summary_df.loc[0, u'年化波动率'] = "%s%%"%(round(100*annual_std, 2))
    summary_df.loc[0, u'信息比率'] = round(ir, 2)
    summary_df.loc[0, u'月度胜率'] = "%s"%(round(100*hit_rate, 2))+"%"
    summary_df.loc[0, u'最大回撤'] = "%s"%(round(100*maxdrawdown, 2))+"%"
    return summary_df


# 计算信号IC
def calc_ic(input_df, ic_type='spearman'):
    '''
    input_df: 列至少包括, ticker, tradeDate, value, return
    '''
    ic_summary = pd.DataFrame()
    
    ic_df = input_df.groupby(['tradeDate']).apply(
        lambda x: x[['value', 'return']].dropna().corr(method=ic_type).values[0, 1])
    
    ic_df = ic_df.reset_index()
    ic_df.columns = ['date', 'ic']
    
    # IC的T值
    ic_t = stats.ttest_1samp(ic_df['ic'].dropna(), 0)[0]
    ic_t = round(ic_t, 2)
    # IC均值
    ic_mean = ic_df['ic'].mean()
    ic_mean = "%s%%" % (round(ic_mean * 100, 2))
    # ICIR
    ic_std = ic_df['ic'].std()
    ic_std = "%s%%" % (round(ic_std * 100, 2))
    ic_ir = ic_df['ic'].mean() / ic_df['ic'].std()
    ic_ir = round(ic_ir, 2)
    ic_summary.loc[ic_type, 'IC'] = ic_mean
    ic_summary.loc[ic_type, 'T'] = ic_t
    ic_summary.loc[ic_type, 'IC_IR'] = ic_ir
    return ic_summary
    
# 回测因子表现
def backtest_factor_perf(factor_frame, factor_name, start_date, end_date, group_num=5, direction=1):
    '''
    factor_frame: 因子列dataframe, 列至少包括：ticker, tradeDate(%Y%m%d), <factor_name>
    group_num: 收益分组的组数
    start_date/end_date: 回测的开始和结束时间,%Y%m%d
    direction: 构造多空组合时，最大-最小组(direction=1)，还是最小-最大组
    return:
            perf: 分组收益,index为日期, columns为分组组名
    '''
    # 只保留月末因子值
    factor_df = extract_month_end_factor(factor_frame, start_date, end_date, factor_name)
    
    # # 剔除掉次新股、ST股、停牌股
    factor_df = remove_special_stocks(factor_df)
    
    # 个股的月度收益dataframe，列至少包括 ticker, tradeDate(%Y%m%d, 月末日期), ret(当月收益), nx_return(下月收益)
    return_frame = get_monthly_return(start_date, end_date)
        
    # 合并因子值和下一期收益, 合并之后的列为: ticker, tradeDate, <factor_name>, 'return'
    factor_df = factor_df.merge(return_frame[['ticker', 'tradeDate', 'nx_return']], on=['ticker', 'tradeDate'], how='left')
    
    factor_df = factor_df.rename(columns={"nx_return": "return"})[['ticker', 'tradeDate', factor_name, 'return']]
        
    factor_df = factor_df.dropna(subset=[factor_name, 'return']).rename(columns={factor_name:"value"})
    
    # 计算因子的IC， RankIC, ICIR, 年化ICIR
    ic_df = calc_ic(factor_df, ic_type='pearson')
    rankic_df = calc_ic(factor_df, ic_type='spearman')
    ic_df = pd.concat([ic_df, rankic_df])
    print u'## 信号IC表现为：\n'
    print ic_df.to_html()
    
    # 分组回测
    perf, bt_df = group_bt(factor_df.rename(columns={"tradeDate":"date"}), group_num, direction=direction)
    
    summary_df = get_perf_summary(perf)
    print u'## 多空对冲组合表现：\n'
    print summary_df.to_html()
    
    print u'## 因子分组表现: '
    # 画图展示
    _ = plot_group_perf(perf, group_num, direction=direction)
    return perf, bt_df



# 比较因子表现， 计算多个因子的IC，T值，IR，年化收益，净值曲线
def compare_signal_perf(factor_dict, start_date, end_date, group_num=5, direction=1):
    '''
    :param factor_dict: 多个信号dataframe的dict, key为信号名, value为<signal_df>
    signal_df格式为: ticker, tradeDate("%Y%m%d"), 'value'
    :param start_date: 回测起始时间，%Y%m%d
    :param end_date:回测结束时间，%Y%m%d
    :param group_num: 多空对冲时，分组的组数
    :param direction: 多空对冲组合的方向
    :return:
    perf_dict: key为信号名，perd_dict = {<信号名1>:{"IC":IC值, "IC_T":IC的T值, "ls_annual_ret":多空年化收益, "ls_ir":多空IR, "nav_df":多空的净值df, index为date}} 
    '''
    perf_dict = {}
    return_frame = None # 个股月度收益数据
    for factor_name in factor_dict.keys():
        print u'回测%s信号的表现...'%factor_name
        factor_frame = factor_dict[factor_name]
        # 只保留月末因子值
        factor_df = extract_month_end_factor(factor_frame, start_date, end_date, 'value')

        # 剔除掉次新股、ST股、停牌股
        remove_df['tradeDate'] = remove_df['tradeDate'].apply(lambda x: x.replace("-", ""))
        factor_df = factor_df.merge(remove_df, on=['ticker', 'tradeDate'], how='left')
        factor_df = factor_df[factor_df['remove_flag'].isnull()]

        if return_frame is None:
            return_frame = get_monthly_return(start_date, end_date)

        # 合并因子值和下一期收益, 合并之后的列为: ticker, tradeDate, 'value', 'return'
        factor_df = factor_df.merge(return_frame[['ticker', 'tradeDate', 'nx_return']], on=['ticker', 'tradeDate'],
                                    how='left')
        factor_df = factor_df.rename(columns={"nx_return": "return"})
        factor_df = factor_df.dropna(subset=['value', 'return'])

        # 得到因子的IC和T值
        rankic_df = calc_ic(factor_df, ic_type='spearman')
        ic_mean = float(rankic_df.loc['spearman', 'IC'].replace("%", ""))
        ic_t = rankic_df.loc['spearman', 'T']

        # 得到多空对冲的年化收益、IR、净值曲线
        perf, bt_df = group_bt(factor_df.rename(columns={"tradeDate": "date"}), group_num, direction=direction)
        summary_df = get_perf_summary(perf)
        ## 年化收益
        ls_annual_ret = float(summary_df.loc[0, u'年化收益'].replace("%", ""))/100
        ## IR
        ls_ir = summary_df.loc[0, u'信息比率']
        ## 净值曲线
        nav_df = (perf['Top-Bottom'] + 1).cumprod()


        # 记录到dict中
        perf_dict[factor_name] = {"IC": ic_mean, "IC_T": ic_t, "ls_annual_ret": ls_annual_ret, "ls_ir": ls_ir,
                                  "nav_df": nav_df}
    return perf_dict


# 画图比较多个信号的表现
def plot_compare_signal_perf(perf_dict):
    '''
    perf_dict: key为信号名，perd_dict = {<信号名1>:{"IC":IC值, "IC_T":IC的T值, "ls_annual_ret":多空年化收益, "ls_ir":多空IR, "nav_df":多空的净值df, index为date}} 
    '''
    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    fig2 = plt.figure(figsize=(15,8))
    ax5 = fig2.add_subplot(111)
    
    factor_names = perf_dict.keys()
    factor_names.sort()
    # 画图展示不同因子的IC均值
    ic_plot_df = pd.Series()
    ic_t_plot_df = pd.Series()
    annual_ret_df = pd.Series()
    annual_ir_df = pd.Series()
    nav_df_list = []
    for factor_name in factor_names:
        ic_plot_df[factor_name] = perf_dict[factor_name]['IC']
        ic_t_plot_df[factor_name] = perf_dict[factor_name]['IC_T']
        annual_ret_df[factor_name] = perf_dict[factor_name]['ls_annual_ret']
        annual_ir_df[factor_name] = perf_dict[factor_name]['ls_ir']
        fnav_df = perf_dict[factor_name]['nav_df']
        fnav_df.name=factor_name
        nav_df_list.append(fnav_df)
    nav_df = pd.concat(nav_df_list, axis=1)
    
    # IC均值
    ax1 = ic_plot_df.plot.bar(ax=ax1)
    ax1.set_xticklabels(ic_plot_df.index.values, fontproperties=font, rotation=45)
    ax1.set_title(u'RankIC均值', fontproperties=font, fontsize=16)
    
    # IC的T值
    ax2 = ic_t_plot_df.plot.bar(ax=ax2)
    ax2.set_xticklabels(ic_t_plot_df.index.values, fontproperties=font, rotation=45)
    ax2.set_title(u'RankIC T值', fontproperties=font, fontsize=12)
    
    # 多空组合的年化收益
    ax3 = annual_ret_df.plot.bar(ax=ax3)
    ax3.set_xticklabels(annual_ret_df.index.values, fontproperties=font, rotation=45)
    ax3.set_title(u'多空组合年化收益', fontproperties=font, fontsize=12)
    
    # 多空组合的IR
    ax4 = annual_ir_df.plot.bar(ax=ax4)
    ax4.set_xticklabels(annual_ir_df.index.values, fontproperties=font, rotation=45)
    ax4.set_title(u'多空组合IR', fontproperties=font, fontsize=12)
    
    # 多空组合的累计净值
    ax5 = nav_df.plot(ax=ax5)
    _ = ax5.legend(loc=2, prop=font)
    ax5.set_title(u'多空组合累计净值', fontproperties=font, fontsize=12)
    
    return 


perf_old_multifactor_with_hamplitude_origin, bt = backtest_factor_perf(factor_old[['tradeDate', 'ticker', 'vol_normal']], 'vol_normal', factor_old.tradeDate.min(), factor_old.tradeDate.max(), group_num=5, direction=-1)

perf_old_multifactor_with_hamplitude_origin, bt = backtest_factor_perf(factor_old[['tradeDate', 'ticker', 'special']], 'special', factor_old.tradeDate.min(), factor_old.tradeDate.max(), group_num=5, direction=-1)

perf_old_multifactor_with_hamplitude_origin, bt = backtest_factor_perf(factor_old[['tradeDate', 'ticker', 'vol_special']], 'vol_special', factor_old.tradeDate.min(), factor_old.tradeDate.max(), group_num=5, direction=-1)


# 将信号放到dict中
factor_dict = {u"普通波动率":factor_old[['tradeDate', 'ticker', 'vol_normal']].rename(columns={'vol_normal': 'value'}),
               u'特异率': factor_old[['tradeDate', 'ticker', 'special']].rename(columns={'special': 'value'}),
               u'特异波动率': factor_old[['tradeDate', 'ticker', 'vol_special']].rename(columns={'vol_special': 'value'}),
              }

# 回测各个信号的表现并画图展示
perf_dict = compare_signal_perf(factor_dict, factor_old.tradeDate.min(), factor_old.tradeDate.max(), group_num=5, direction=-1)
_ = plot_compare_signal_perf(perf_dict)

'''

第三部分：新特异波动率因子与新特异率因子表现回测
该部分耗时 2分钟
该部分内容为：

生成加入流动性因子收益率之后产生的新特异波动率因子与新特异率因子。

新特异率因子和新特异波动率因子的单因子表现回测。

新特异率因子、新特异波动率因子与特异率因子、特异波动率因子的对比表现。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

3.1 生成加入流动性因子收益率之后产生的新特异波动率因子与新特异率因子

特异率衡量在剥离了可以对个股收益解释的因素后个股的异动情况，目前多以 fama 三因子模型为基础（市场 beta、规模和估值），以线性回归剔除线性影响的方式，以拟合优度为异常程度的评价标准。这种构建特异率的方式可以较好的体现个股价格变动相对市场主要矛盾的异常，但针对波动率类因子描述的价格变动，则需尽可能的考虑更多可以解释个股收益的因素。我们知道波动率类因子和流动性因子线性相关性较高，在 fama 三因子的基础上加入流动性因子收益率，构建新的特异信息，则回归方程如下： 图片注释 并取上述方程回归结果的残差标准差及拟合优度，构建特异波动率及特异率因子，在下文中统称为新特异波动率和新特异率，本文中将特异率因子和新特异率因子统称为特异率类因子，将特异波动率因子和新特异波动率因子统称为特异波动率类因子

'''


start_time = time.time()
print("生成加入流动性因子收益率之后产生的新特异波动率因子与新特异率因子....")

#读取特异率因子
with open('{}/special_strange_000985_4.pkl'.format(special_vol_dir), 'r') as f:
    r2_df = pickle.load(f)
r2_df = r2_df.T
special_df = 1 - r2_df

#生成普通波动率因子
vol_normal = pd.rolling_std(close_pct, window=21, min_periods=5)
vol_normal_part = vol_normal.loc[sorted(list(filter(lambda x: x in special_df.index, vol_normal.index)))]
special_df_stack = special_df.stack().reset_index()
special_df_stack.columns = ['tradeDate', 'ticker', 'special']
vol_normal_part_stack = vol_normal_part.stack().reset_index()
vol_normal_part_stack.columns = ['tradeDate', 'ticker', 'vol_normal']

#生成特异波动率因子
concat = special_df_stack.merge(vol_normal_part_stack, on=['ticker', 'tradeDate'], how='left')
concat['vol_special'] = concat['vol_normal'] * np.sqrt(concat['special'])
#dropna
concat = concat.dropna()
#factor_new 是用市值因子、估值因子、流动性因子、市场收益做回归得到的各个因子
factor_new = concat

end_time = time.time()
print "Time cost: %s seconds" % (end_time - start_time)

perf_old_multifactor_with_hamplitude_origin, bt = backtest_factor_perf(factor_new[['tradeDate', 'ticker', 'vol_special']], 'vol_special', factor_new.tradeDate.min(), factor_new.tradeDate.max(), group_num=5, direction=-1)
perf_old_multifactor_with_hamplitude_origin, bt = backtest_factor_perf(factor_new[['tradeDate', 'ticker', 'special']], 'special', factor_new.tradeDate.min(), factor_new.tradeDate.max(), group_num=5, direction=-1)


# 将信号放到dict中
factor_dict = {
               u'特异率': factor_old[['tradeDate', 'ticker', 'special']].rename(columns={'special': 'value'}),
               u'特异波动率': factor_old[['tradeDate', 'ticker', 'vol_special']].rename(columns={'vol_special': 'value'}),
               u'新特异率': factor_new[['tradeDate', 'ticker', 'special']].rename(columns={'special': 'value'}),
               u'新特异波动率': factor_new[['tradeDate', 'ticker', 'vol_special']].rename(columns={'vol_special': 'value'})
              }

# 回测各个信号的表现并画图展示
perf_dict = compare_signal_perf(factor_dict, factor_old.tradeDate.min(), factor_old.tradeDate.max(), group_num=5, direction=-1)
_ = plot_compare_signal_perf(perf_dict)



'''

第四部分：非线性特异波动率因子与特异波动率因子的表现对比
该部分耗时 2分钟
该部分内容为：

生成对普通波动率进行非线性处理后产生的非线性特异波动率因子。

非线性特异波动率因子的单因子表现回测。

非线性特异波动率因子与特异波动率因子的表现对比。

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

4.1 生成对普通波动率进行非线性处理后产生的非线性特异波动率因子

从特异波动率因子的五分组收益图上我们可以看到，限制特异波动率因子表现的根本原因在于其空头组区分个股能力显著，而多头组区分个股能力较差，整体上拖累了杠杆加强个股股价变动异常信息时的头部组表现。从数学形式上看，特异波动率综合个股特异和个股波动的方式是非线性的： 图片注释
如果可以保留这种信息综合方式，同时在数学形式上给出一定改善，以增强波动率因子空头组信息，同时减弱波动率因子多头组信息，便可以改善特异波动率因子选股的线性。
特异波动率因子在基础波动率在进行截面归一化后，以线性关系均匀分布，作为乘数加强特异率体现的个股特异信息。高个股波动区域确实存在更多价格变动异常，而在低波动区域个股之间的价格变动差别不大，局部的扰动误差会干扰杠杆在尾端的加强。为了增强空头组区分，减弱多头组区分，首先对基础波动率做如截面归一化： 图片注释
对得到的归一化值进行非线性变换，得到基础波动率乘数部分： 图片注释

'''

#生成非线性的基础波动率乘数部分 
std_norm = factor_old[['tradeDate', 'ticker', 'vol_normal']].groupby('tradeDate').apply(
    lambda x: (x[['vol_normal', 'ticker']].set_index('ticker')-x['vol_normal'].min())/(x['vol_normal'].max()- x['vol_normal'].min()))
std_norm = std_norm.reset_index()
std_norm.columns = ['tradeDate', 'ticker', 'norm']
std_norm['non_linear_std'] = (np.e)**std_norm['norm']
std_norm['ticker'] = std_norm['ticker'].apply(lambda x: str(x).zfill(6))
#与原有因子merge
factor_nonlinear = factor_old.merge(std_norm, on=['tradeDate', 'ticker'], how='left')
#生成非线性特异波动率因子
factor_nonlinear['vol_special_nonlinear'] = factor_nonlinear['non_linear_std'] * np.sqrt(factor_nonlinear['special'])
#dropna
factor_nonlinear = factor_nonlinear.dropna()

factor_name = 'vol_special_nonlinear'
perf_old_multifactor_with_hamplitude_origin, bt = backtest_factor_perf(factor_nonlinear[['tradeDate', 'ticker', factor_name]], factor_name, factor_nonlinear.tradeDate.min(), factor_nonlinear.tradeDate.max(), group_num=5, direction=-1)

# 将信号放到dict中
factor_dict = {u"普通波动率":factor_nonlinear[['tradeDate', 'ticker', 'vol_normal']].rename(columns={'vol_normal': 'value'}),
               u'特异波动率': factor_nonlinear[['tradeDate', 'ticker', 'vol_special']].rename(columns={'vol_special': 'value'}),
              u'非线性特异波动率': factor_nonlinear[['tradeDate', 'ticker', 'vol_special_nonlinear']].rename(columns={'vol_special_nonlinear': 'value'})}


# 回测各个信号的表现并画图展示
perf_dict = compare_signal_perf(factor_dict, factor_nonlinear.tradeDate.min(), factor_nonlinear.tradeDate.max(), group_num=5, direction=-1)
_ = plot_compare_signal_perf(perf_dict)


'''

从上面三个因子的对比分析来看，相较于普通波动率因子和特异波动率因子，非线性特异波动率因子各项表现指标均为最优；这说明在进行非线性处理之后，特异波动率因子选股的线性得到了较大的改善。

   调试 运行
文档
 代码  策略  文档
第五部分：总结
   调试 运行
文档
 代码  策略  文档
低波效应的核心逻辑为股价变动相对于市场异常现象的均值回归，刻画相对价格变动的特异波动率更符合其逻辑；数学形式上看，特异波动率由波动率和特异率两个部分信息共同组成；从因子回测表现上看，特异波动率因子比普通波动率因子更为优秀，因子分组收益的单调性更为明显，并且因子多空组合年化收益从 2.84% 提高到 8.46%，信息比率从0.35提高到1.29。特异率因子表现则更为突出，因子IC均值的绝对值5.5%，多空对冲组合年化收益9.68%, 信息比率2.2。
在 fama 三因子的基础上加入流动性因子收益率回归得到的新特异率因子和新特异波动率因子表现并没有提升，反而在各个表现指标上均有下降。这说明在回归方程中加入更多可以解释个股收益的因子收益率能够提高特异率因子和特异波动率因子的表现的说法值得商榷。
在对特异波动率因子的个股波动部分做了非线性处理后，因子表现进一步提升，IC绝对值均值从5.35%提高到6.09%， 因子多空组合年化收益从 8.46% 提高到 10.1%，信息比率从1.29提高到1.91。

'''
