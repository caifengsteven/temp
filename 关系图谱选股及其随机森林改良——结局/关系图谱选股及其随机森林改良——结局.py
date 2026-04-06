# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:46:23 2020

@author: Asus
"""


import pandas as pd
import time 
import statsmodels.api as sm

universe = set_universe('A')
#universe = universe[:500]    #测试所用
begin = '20130101'
end = '20170101'



def get_match():
    a = {'建筑建材':801060,'建筑材料':801061,'建筑装饰':801720,'机械设备':801070,'电气设备':801730,'交运设备':801090,'国防军工':801740,'汽车':801880,'信息设备':801100,'信息服务':801220,'计算机':801750,'传媒':801221,'通信':801770,'金融服务':801190,'银行':801780,'非银金融':801790,'钢铁':801040,'休闲服务':801210,'农林牧渔':801010,'采掘':801020,'化工':801030,'有色金属':801050,'电子':801080,'家用电器':801110,'食品饮料':801120,'纺织服装':801130,'轻工制造':801140,'医药生物':801150,'公用事业':801160,'交通运输':801170,'房地产':801180,'商业贸易':801200,'综合':801230}
    match_all = pd.DataFrame(data=a,index=['id'])
    return match_all        

def get_datelist(begin_date,end_date):
    date_data=DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=begin_date,endDate=end_date,field=['calendarDate','isOpen'])
    date_data=date_data[date_data['isOpen'] == 1]
    #date= map(lambda x: x[0:4]+x[5:7]+x[8:10], date_data['calendarDate'].values.tolist())
    date_data=date_data['calendarDate'].values.tolist()
    date = pd.DataFrame(data=range(len(date_data)),index=date_data)
    date.columns = ['we']
    return date               #得到交易日


def get_all_R(universe,begin,end):
    all_match = get_match()
    all_R_data = pd.DataFrame()
    all_R_data2 = pd.DataFrame()
    all_R_data3 = pd.DataFrame()
    all_R_data4 = pd.DataFrame()
    for x in universe:
        print (x)
        tmp_match = DataAPI.EquIndustryGet(industryVersionCD=u"010303",industry=u"",secID=x,ticker=u"",intoDate=u"",field=u"secID,exchangeCD,intoDate,outDate,industryName1",pandas="1").sort('intoDate').set_index('intoDate')
        tmp_match = tmp_match.fillna('2017-05-01')     #填补唯一的缺失值
        for n in tmp_match.index:
            for nn in all_match.columns:
                if tmp_match.ix[n,'industryName1'] == nn:
                    tmp_match.ix[n,'id'] = all_match.ix['id',nn]
        #tmp_match.outDate[-1] = '2017-05-01'
        
        mkt2_data = pd.DataFrame()
        for m in tmp_match.index:
            beg = m
            beg = time.strptime(beg,"%Y-%m-%d")
            beg = time.strftime("%Y%m%d",beg)
            ed = tmp_match.ix[m,'outDate']
            ed = time.strptime(ed,"%Y-%m-%d")
            ed = time.strftime("%Y%m%d",ed)
            mkt2 = int(tmp_match.ix[m,'id'])
            mkt2_tmp = DataAPI.MktIdxdGet(beginDate=beg,endDate=ed,ticker=str(mkt2),exchangeCD=u"XSHE,XSHG",field=u"tradeDate,CHGPct",pandas="1")
            mkt2_data = mkt2_data.append(mkt2_tmp)
        mkt2_data.columns = ['tradeDate','mkt2']
        mkt2_data = mkt2_data.sort('tradeDate')
        mkt2_data.index = range(len(mkt2_data))
        #mkt2_data = mkt2_data.set_index('tradeDate')
        
        
        
        
        if tmp_match.exchangeCD[0] == 'XSHE':
            mkt1 = '399001'
        if tmp_match.exchangeCD[0] == 'XSHG':
            mkt1 = '000001'
        mkt1_data =DataAPI.MktIdxdGet(beginDate=begin,endDate=end,ticker=mkt1,exchangeCD=u"XSHE,XSHG",field=u"tradeDate,CHGPct",pandas="1").sort('tradeDate')
        mkt1_data.columns = ['tradeDate','mkt1']
        #mkt1_data = mkt1_data.set_index('tradeDate')
        
        
        stk_data = DataAPI.MktEqudGet(beginDate=begin,endDate=end,secID=x,field=u"tradeDate,chgPct",pandas="1")
        stk_data.columns = ['tradeDate','stk']
        stk_data = stk_data.sort('tradeDate')
        
        all_data = pd.merge(mkt1_data,mkt2_data,on='tradeDate',how='outer')
        all_data = pd.merge(all_data,stk_data,on='tradeDate',how='outer')
        all_data = all_data.sort('tradeDate')
        all_data = all_data.dropna().drop_duplicates(['tradeDate'])
        all_data.index = range(len(all_data))
        for i in range(len(all_data.index)-20):
            tmp_x = all_data.ix[i:i+20,['mkt1','mkt2']]
            tmp_x.mkt1 = tmp_x.mkt1.cumsum()
            tmp_x.mkt2 = tmp_x.mkt2.cumsum()
            tmp_y = all_data.ix[i:i+20,['stk']]
            tmp_y = tmp_y.stk.cumsum()
            tmp_x = sm.add_constant(tmp_x)
            est=sm.OLS(tmp_y,tmp_x).fit()
            tmp_R = est.rsquared
            tmp_R2 = est.params[1]          
            #tmp_R3 = est.params[2]    
            all_data.loc[i+15,'R'] = tmp_R
            all_data2 = all_data
            all_data3 = all_data
            all_data4 = all_data
            all_data2.loc[i+20,'xielv'] = tmp_R2
            all_data3.loc[i+20,'dapan'] = tmp_x.ix[tmp_x.index[-1],'mkt1']
            all_data4.loc[i+20,'gegu']  = tmp_y.data[-1]
        all_data = all_data.ix[:,['tradeDate','R']]
        all_data2 = all_data2.ix[:,['tradeDate','xielv']]
        all_data3 = all_data3.ix[:,['tradeDate','dapan']]
        all_data4 = all_data4.ix[:,['tradeDate','gegu']]

        all_data.columns = ['tradeDate',x]
        all_data2.columns = ['tradeDate',x]
        all_data3.columns = ['tradeDate',x]
        all_data4.columns = ['tradeDate',x]
        all_data = all_data.drop_duplicates(['tradeDate'])     #删除重复tradeDate 的行
        all_data2 = all_data2.drop_duplicates(['tradeDate'])
        all_data3 = all_data3.drop_duplicates(['tradeDate'])
        all_data4 = all_data.drop_duplicates(['tradeDate'])

        if all_R_data.empty:
            all_R_data = all_data
        else:all_R_data = pd.merge(all_R_data,all_data,on='tradeDate',how='outer')
        all_R_data = all_R_data.sort('tradeDate')
        all_R_data = all_R_data.drop_duplicates(['tradeDate'])
        
        if all_R_data2.empty:
            all_R_data2 = all_data2
        else:all_R_data2 = pd.merge(all_R_data2,all_data2,on='tradeDate',how='outer')
        all_R_data2 = all_R_data2.sort('tradeDate')
        all_R_data2 = all_R_data2.drop_duplicates(['tradeDate'])
        
        if all_R_data3.empty:
            all_R_data3 = all_data3
        else:all_R_data3 = pd.merge(all_R_data3,all_data3,on='tradeDate',how='outer')
        all_R_data3 = all_R_data3.sort('tradeDate')
        all_R_data3 = all_R_data3.drop_duplicates(['tradeDate'])
        
        if all_R_data4.empty:
            all_R_data4 = all_data4
        else:all_R_data4 = pd.merge(all_R_data4,all_data4,on='tradeDate',how='outer')
        all_R_data4 = all_R_data4.sort('tradeDate')
        all_R_data4 = all_R_data4.drop_duplicates(['tradeDate'])
    return all_R_data,all_R_data2,all_R_data3,all_R_data4

h,m,n,e = get_all_R(universe,begin,end)
h.to_csv('test1_R.csv')
m.to_csv('test1_xielv.csv')
n.to_csv('test1_dapan.csv')
e.to_csv('test1_gegu.csv')

#初学python后半部分得到数据合并数据和保存数据写得有点多，应该是有简短的语法的。    


import pandas as pd 

a_data = pd.read_csv('zengliang1.csv')
a_data = a_data[a_data.columns[1:]].set_index('tradeDate')
quantile_data = pd.DataFrame(data=0,index=a_data.index,columns=['R_25','R_50','R_75','R_max'])

for x in quantile_data.index:
    #quantile_data.loc[x,'R_min'] = a_data.loc[x,:].dropna().quantile(0)
    quantile_data.loc[x,'R_25'] = a_data.loc[x,:].dropna().quantile(0.25)
    quantile_data.loc[x,'R_50'] = a_data.loc[x,:].dropna().quantile(0.5)
    quantile_data.loc[x,'R_75'] = a_data.loc[x,:].dropna().quantile(0.75)
    quantile_data.loc[x,'R_max'] = a_data.loc[x,:].dropna().quantile(1)
print (quantile_data)
#得到因子分位数数据


mkt1_data =DataAPI.MktIdxdGet(beginDate='20090101',endDate='20170501',ticker='000001',exchangeCD=u"XSHE,XSHG",field=u"tradeDate,closeIndex",pandas="1").sort('tradeDate')
mkt1_data.columns=['tradeDate','dapan']
#print a_data
h_data = pd.merge(m_data,mkt1_data,on='tradeDate',how='left')
h_data = h_data.set_index('tradeDate')
h_data = h_data.reset_index()
h_data.to_csv('baogao.csv')

r_data = pd.read_csv('zengliang1.csv')
r_data = r_data[r_data.columns[1:]].set_index('tradeDate')
stk = r_data.columns[1600]
r_data2 = r_data.loc[:,stk]
r_data2 = r_data2.reset_index()

price = DataAPI.MktEqudGet(beginDate='20090101',endDate='20170501',secID=stk,field=u"tradeDate,closePrice",pandas="1")
price.columns = ['tradeDate','price']

al_data = pd.merge(r_data2,price,on='tradeDate',how='left')
print (al_data)
al_data.to_csv('300284data.csv')

import time
for n in quantile_data.index:
    n_tmp = time.strptime(n,"%Y-%m-%d")
    n_tmp = time.strftime("%Y%m%d",n_tmp)
    quantile_data.loc[n,'dapan'] = DataAPI.MktIdxdGet(tradeDate=n_tmp,ticker='000001',exchangeCD=u"XSHE,XSHG",field=u"closeIndex",pandas="1").closeIndex.values
    quantile_data.loc[n,'000001']=DataAPI.MktEqudGet(tradeDate=n_tmp,ticker='000001',field=u"closePrice",pandas="1").closePrice.values
    
    quantile_data.loc[n,'jhqc']=DataAPI.MktEqudGet(tradeDate=n_tmp,ticker='600418',field=u"closePrice",pandas="1").closePrice.values
    quantile_data.loc[n,'nba']=DataAPI.MktEqudGet(tradeDate=n_tmp,ticker='000012',field=u"closePrice",pandas="1").closePrice.values
print (quantile_data)


import pandas as pd
#调仓周期5天，最大百分位


start = '2013-01-01'                       # 回测起始时间
end = '2017-05-01'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('A')               # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate =  5                       # 调仓频率，表示执行handle_data的时间间隔

p_data1 = pd.read_csv('zengliang4.csv')
p_data2 = pd.read_csv('zengliang5.csv')

p_data1 = p_data1[p_data1.columns[1:]].set_index('tradeDate')
p_data2 = p_data2[p_data2.columns[1:]].set_index('tradeDate')

p_data = p_data1*p_data2

p_data = p_data[973:]        #：1457 是09-1-5    1000: 是130219开始至今
p_dates = p_data.index.values
result=pd.DataFrame()
commission = Commission(0.0002,0.0002)     # 交易费率设为双边万分之二


def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in p_dates:           # Q因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
        # 拿取调仓日前一个交易日的Q因子，并按照相应十分位选择股票
    p = p_data.ix[pre_date].dropna()
    p_min = p.quantile(0)     #乘以 0.9-1 1和5 不行，   乘以 0.1-0 5 不行，   
    p_max = p.quantile(0.2)
    my_univ = p[p>=p_min][p<p_max].index.values
    
        # 调仓逻辑
    univ = [x for x in my_univ if x in account.universe]
    
        # 不在股票池中的，清仓
    for stk in account.valid_secpos:    #valid_secpos 
        if stk not in univ:
            order_to(stk, 0)
        # 在目标股票池中的，等权买入
    for stk in univ:
        order_pct_to(stk, 1.1/len(univ))  

#注意！回测后会得到三个重要文件，最有用的为bt！    
        
        
#bt.to_csv('one.csv')
#bt.to_csv('two.csv')
#bt.to_csv('three.csv')
#bt.to_csv('five.csv')
bt.to_csv('four.csv')

import pandas as pd
#调仓周期5天，最大百分位


start = '2013-01-01'                       # 回测起始时间
end = '2017-05-01'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('A')               # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate =  5                       # 调仓频率，表示执行handle_data的时间间隔

p_data1 = pd.read_csv('zengliang4.csv')
p_data2 = pd.read_csv('zengliang5.csv')

p_data1 = p_data1[p_data1.columns[1:]].set_index('tradeDate')
p_data2 = p_data2[p_data2.columns[1:]].set_index('tradeDate')

p_data = p_data1*p_data2

p_data = p_data[973:]        #：1457 是09-1-5    1000: 是130219开始至今
p_dates = p_data.index.values
result=pd.DataFrame()
commission = Commission(0.0002,0.0002)     # 交易费率设为双边万分之二


def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    pre_date = account.previous_date.strftime("%Y-%m-%d")
    if pre_date not in p_dates:           # Q因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
        return
    
        # 拿取调仓日前一个交易日的Q因子，并按照相应十分位选择股票
    p = p_data.ix[pre_date].dropna()
    p_min = p.quantile(0.85)     #乘以 0.9-1 1和5 不行，   乘以 0.1-0 5 不行，   
    p_max = p.quantile(0.9)
    my_univ = p[p>=p_min][p<p_max].index.values
    
        # 调仓逻辑
    univ = [x for x in my_univ if x in account.universe]
    
        # 不在股票池中的，清仓
    for stk in account.valid_secpos:    #valid_secpos 
        if stk not in univ:
            order_to(stk, 0)
        # 在目标股票池中的，等权买入
    for stk in univ:
        order_pct_to(stk, 1.1/len(univ))  

#注意！回测后会得到三个重要文件，最有用的为bt！    
        
        
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from CAL.PyCAL import *    # CAL.PyCAL中包含font

fig = plt.figure(figsize=(10,8))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.grid()
ax2.grid()


A = pd.read_csv('one.csv')
B = pd.read_csv('two.csv')
C = pd.read_csv('three.csv')
D = pd.read_csv('four.csv')
E = pd.read_csv('five.csv')


A = A[A.columns[1:]]
B = B[B.columns[1:]]
C = C[C.columns[1:]]
D = D[D.columns[1:]]
E = E[E.columns[1:]]

all_data = {}
all_data[0] = A
all_data[1] = B
all_data[2] = C
all_data[3] = D
all_data[4] = E


for i in all_data:
    
    all_data[i]['portfolio_return'] = all_data[i].portfolio_value/all_data[i].portfolio_value.shift(1) - 1.0
    all_data[i]['portfolio_return'].ix[0] = all_data[i]['portfolio_value'].ix[0]/10000000.0 - 1.0
    all_data[i]['excess_return'] = all_data[i].portfolio_return - A.benchmark_return                 # 总头寸每日超额回报率
    all_data[i]['excess'] = all_data[i].excess_return + 1.0
    all_data[i]['excess'] = all_data[i].excess.cumprod()                # 总头寸对冲指数后的净值序列
    all_data[i]['portfolio'] = all_data[i].portfolio_return + 1.0     
    all_data[i]['portfolio'] = all_data[i].portfolio.cumprod()          # 总头寸不对冲时的净值序列
    all_data[i]['benchmark'] = all_data[i].benchmark_return + 1.0
    all_data[i]['benchmark'] = all_data[i].benchmark.cumprod()          # benchmark的净值序列
    all_data[i].tradeDate= pd.to_datetime(all_data[i].tradeDate)  #把日期变成plot可以识别的日期形式，改变前为str

    ax1.plot(all_data[i]['tradeDate'], all_data[i][['portfolio']], label=str(i))
    ax2.plot(all_data[i]['tradeDate'], all_data[i][['excess']], label=str(i))

ax1.plot(all_data[i]['tradeDate'], all_data[i][['benchmark']], label='bench')






'''for qt in results_illiq:
    bt = results_illiq[qt]['bt']

    data = bt[[u'tradeDate',u'portfolio_value',u'benchmark_return']]
    data['portfolio_return'] = data.portfolio_value/data.portfolio_value.shift(1) - 1.0   # 总头寸每日回报率
    data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/	10000000.0 - 1.0
    data['excess_return'] = data.portfolio_return - data.benchmark_return                 # 总头寸每日超额回报率
    data['excess'] = data.excess_return + 1.0
    data['excess'] = data.excess.cumprod()                # 总头寸对冲指数后的净值序列
    data['portfolio'] = data.portfolio_return + 1.0     
    data['portfolio'] = data.portfolio.cumprod()          # 总头寸不对冲时的净值序列
    data['benchmark'] = data.benchmark_return + 1.0
    data['benchmark'] = data.benchmark.cumprod()          # benchmark的净值序列
    results_illiq[qt]['hedged_max_drawdown'] = max([1 - v/max(1, max(data['excess'][:i+1])) for i,v in enumerate(data['excess'])])  # 对冲后净值最大回撤
    results_illiq[qt]['hedged_volatility'] = np.std(data['excess_return'])*np.sqrt(252)
    results_illiq[qt]['hedged_annualized_return'] = (data['excess'].values[-1])**(252.0/len(data['excess'])) - 1.0
    # data[['portfolio','benchmark','excess']].plot(figsize=(12,8))
    # ax.plot(data[['portfolio','benchmark','excess']], label=str(qt))
    ax1.plot(data['tradeDate'], data[['portfolio']], label=str(qt))
    ax1.plot(data['tradeDate'], data[['benchmark']], label=str(qt))
    ax2.plot(data['tradeDate'], data[['excess']], label=str(qt))'''
    
    

ax1.legend(loc=0, fontsize=12)
ax2.legend(loc=0, fontsize=12)
ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲净值", fontproperties=font, fontsize=16)
ax1.set_title(u"五分位的走势", fontproperties=font, fontsize=16)
ax2.set_title(u"五分位对冲全A股指数后净值走势", fontproperties=font, fontsize=16)

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')
import seaborn as sns
sns.set_style('white')
from matplotlib import dates
import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
import scipy.stats as st
from CAL.PyCAL import *    # CAL.PyCAL中包含font
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std



def plot_return(bt, title):
    bt['portfolio_return'] = bt['portfolio_value']/bt['portfolio_value'].shift(1) - 1.0
    bt['excess_return'] = bt.portfolio_return - bt.benchmark_return                 # 总头寸每日超额回报率
    bt['excess'] = bt.excess_return + 1.0
    bt['excess'] = bt.excess.cumprod()                # 总头寸对冲指数后的净值序列
    month_date = [bt.ix[0, 0].date()]
    portfolio_value = [bt.ix[0, 'portfolio_value']]
    portfolio_excess= [bt.ix[0, 'excess']]
    for i in range(len(bt)-1):
        if bt.ix[i+1, 0].month != month_date[-1].month:
            month_date.append(bt.ix[i+1, 0].date())
            portfolio_value.append(bt.ix[i+1, 'portfolio_value'])
            portfolio_excess.append(bt.ix[i+1, 'excess'])
    results = pd.DataFrame(data=portfolio_value, columns=['portfolio_value'])
    results['monthly_return'] = results['portfolio_value']/results['portfolio_value'].shift(1)-1
    results['cumulative_return'] = portfolio_excess
    results['monthly_return_1'] = results['cumulative_return']/results['cumulative_return'].shift(1)-1
    results.index = month_date
    results.dropna(inplace=True)
    
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    lns2 = ax1.plot(results.index, results['cumulative_return'], 'r')
    lns1 = ax2.bar(results.index, results['monthly_return_1'], align='center', width=10, color='b')
    ax2.legend(lns1, ['monthly return(right axis)'], loc=1, fontsize=12)
    ax1.legend(lns2, ['excess return(left axis)'], loc=2, fontsize=12)
    ax1.set_ylabel(u'组合净值/基准净值', fontproperties=font, fontsize=16)
    ax2.set_ylabel(u'月度收益率', fontproperties=font, fontsize=16)
    ax1.set_xlabel(u'时间', fontproperties=font, fontsize=16)
    ax2.set_yticklabels([str(x*100)+'0%' for x in ax2.get_yticks()], fontproperties=font, fontsize=14)
    s = ax1.set_title(title, fontproperties=font, fontsize=16)
    return ax1, ax2

s = plot_return(bt, u'因子的月度表现')


from CAL.PyCAL import *
import pandas as pd
import numpy as np
# 可编辑部分与 strategy 模式一样，其余部分按本例代码编写即可
# -----------回测参数部分开始，可编辑------------
start = '2013-01-01'                       # 回测起始时间
end = '2017-05-01'                           # 回测结束时间
benchmark ='HS300'                    # 策略参考标准
universe = set_universe('A')             # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测
refresh_rate = 5                          # 调仓频率，表示执行handle_data的时间间隔
cal = Calendar('China.SSE')
period = Period('-1B')
aa = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate='20130101', endDate='20170501')
aa = aa[aa['isMonthEnd']==1].sort('calendarDate')
    # 工作日列表
aa = aa['calendarDate'].values.tolist()   
    #print aa
# ---------------回测参数部分结束----------------
# 把回测参数封装到 SimulationParameters 中，供 quick_backtest 使用
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base)
# 获取回测行情数据
data = quartz.get_backtest_data(sim_params)
# 运行结果
results_illiq = {}
fac = pd.read_csv('zengliang1.csv')
fac = fac[fac.columns[1:]].set_index('tradeDate')
fac = fac[973:]        #：1457 是09-1-5    1000: 是130219开始至今
p_dates = fac.index.values

# 调整参数(选取股票的ILLIQ因子五分位数)，进行快速回测
for quantile_five in range(0,5):
    print ('start')
    # ---------------策略逻辑部分----------------
  
    cal = Calendar('China.SSE')#股票交易日历
    period = Period('-1B')#前一工作日长度
    
    def initialize(account):                   # 初始化虚拟账户状态
        pass

    def handle_data(account):# 每个交易日的买入卖出指令
        yesterday = account.previous_date.strftime("%Y-%m-%d")
        if yesterday in p_dates:            # Q因子只在每个月底计算，所以调仓也在每月最后一个交易日进行
            today = account.current_date
            today = Date.fromDateTime(account.current_date)  # 向前移动一个工作日，转换为cal模式日期
        #print yesterday.strftime('%Y%m%d')
            if yesterday in aa:#转换日期显示形式

                sell_list = account.valid_secpos
                for stk in sell_list:
                    order_to(stk, 0)
                
                t_len=len(fac.loc[yesterday].dropna()) 
                if quantile_five==0:
                    t_a=0
                    t_b=0.2
                if quantile_five==1:
                    t_a=0.2
                    t_b=0.4
                if quantile_five==2:
                    t_a=0.4
                    t_b=0.6
                if quantile_five==3:
                    t_a=0.6
                    t_b=0.8
                if quantile_five==4:
                    t_a=0.8
                    t_b=1
            #print t_a
            #print t_b
                ta=fac.loc[yesterday].rank()>t_a*t_len 
                tbb=fac.loc[yesterday].rank()<t_b*t_len
            #print ta
                buy_list_1 =  ta[ta==True].index#rsi中的secid
                buy_list_2 = tbb[tbb==True].index
                buy_list=[i for i in buy_list_1 if i in buy_list_2]
                total_money = account.referencePortfolioValue#投资组合参考价值（虽然不清楚具体公式）
                prices = account.referencePrice #前一日收盘价
                for stk in buy_list:#遍历买入list
                #print stk
                #print prices[stk]
                    if np.isnan(prices[stk]) or prices[stk] == 0:  # 停牌或是还没有上市等原因不能交易
                        continue
                    order(stk, int(total_money /len(buy_list) / prices[stk] /100)*100)#投资组合价值＊股票权重值/股票前一日收盘价（就是参考投资组合价值和权重买多少股股票，这里是不是多了个／100）＊100？）
            else:
                return
                           
     
    # ---------------策略逻辑部分结束----------------

    # 把回测逻辑封装到 TradingStrategy 中，供 quick_backtest 使用
    strategy = quartz.TradingStrategy(initialize, handle_data)
    # 回测部分
    bt, acct = quartz.quick_backtest(sim_params, strategy, data, refresh_rate=refresh_rate)
    # 对于回测的结果，可以通过 perf_parse 函数计算风险指标
    perf = quartz.perf_parse(bt, acct)

    # 保存运行结果
    tmp = {}
    tmp['bt'] = bt
    tmp['annualized_return'] = perf['annualized_return']
    tmp['volatility'] = perf['volatility']
    tmp['max_drawdown'] = perf['max_drawdown']
    tmp['alpha'] = perf['alpha']
    tmp['beta'] = perf['beta']
    tmp['sharpe'] = perf['sharpe']
    tmp['information_ratio'] = perf['information_ratio']
    
    results_illiq[quantile_five] = tmp
    print (str(quantile_five))
print ('done')


import pandas as pd
import time

new = pd.read_csv('zengliang1.csv')
new = new[new.columns[1:]]
match_data = pd.DataFrame(data=0,index=new.index,columns=['start'])
match_data['tradeDate'] = new['tradeDate']
begin_date = match_data.tradeDate.values[0]
beg = time.strptime(begin_date,"%Y-%m-%d")
beg = time.strftime("%Y%m%d",beg)
end_date = match_data.tradeDate.values[-1]
ed = time.strptime(end_date,"%Y-%m-%d")
ed = time.strftime("%Y%m%d",ed)
mkt1_data =DataAPI.MktIdxdGet(beginDate=beg,endDate=ed,ticker='000001',exchangeCD=u"XSHE,XSHG",field=u"tradeDate,CHGPct",pandas="1").sort('tradeDate')



for i in new.columns.values[1:]:
    stk_data = DataAPI.MktEqudGet(beginDate=beg,endDate=ed,secID=i,field=u"tradeDate,closePrice",pandas="1").sort('tradeDate')
    stk_data['closePrice'] = 1-(stk_data.closePrice/stk_data.shift(-5).closePrice)
    stk_data.columns=['tradeDate',i]
    match_data = pd.merge(left=match_data,right=stk_data,on='tradeDate',how='left')
print (match_data)
    
match_data.to_csv('ALL_A_forward_5.csv')


import numpy as np
import scipy.stats as st

ALL_A_forward_5 = pd.read_csv('ALL_A_forward_5.csv')
ALL_A_forward_5 = ALL_A_forward_5[ALL_A_forward_5.columns[2:]].set_index('tradeDate')
#factor_data1 = pd.read_csv('zengliang1.csv')
#factor_data2 = pd.read_csv('zengliang2.csv')

#factor_data1 = factor_data1[factor_data1.columns[1:]].set_index('tradeDate')
#factor_data2 = factor_data2[factor_data2.columns[1:]].set_index('tradeDate')
#factor_data = factor_data1*factor_data2
factor_data = pd.read_csv('zengliang1.csv')
factor_data = factor_data[factor_data.columns[1:]].set_index('tradeDate')
ic_data = pd.DataFrame(index=factor_data.index, columns=['IC','pValue'])
# 计算相关系数
for dt in ic_data.index:
    tmp_R = factor_data.ix[dt]
    tmp_ret = ALL_A_forward_5.ix[dt]
    cor = pd.DataFrame(tmp_R)
    ret = pd.DataFrame(tmp_ret)
    cor.columns = ['corr']
    ret.columns = ['ret']
    cor['ret'] = ret['ret']
    cor = cor[~np.isnan(cor['corr'])][~np.isnan(cor['ret'])]
    if len(cor) < 5:
        continue

    ic, p_value = st.spearmanr(cor['corr'],cor['ret'])   # 计算秩相关系数 RankIC
    ic_data['IC'][dt] = ic
    ic_data['pValue'][dt] = p_value
    
print ('mean of IC: ', ic_data['IC'].mean())
print ('median of IC: ', ic_data['IC'].median())
print ('the number of IC(all, plus, minus): ', (len(ic_data), len(ic_data[ic_data.IC>0]), len(ic_data[ic_data.IC<0])))

import matplotlib.pyplot as plt
from CAL.PyCAL import *    # CAL.PyCAL中包含font

ic_data.index= pd.to_datetime(ic_data.index)  #把日期变成plot可以识别的日期形式，改变前为str

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(111)

lns1 = ax1.plot(ic_data.index, ic_data.IC, label='IC')

lns = lns1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=[0.5, 0.1], loc='', ncol=3, mode="", borderaxespad=0., fontsize=12)
ax1.set_ylabel(u'相关系数', fontproperties=font, fontsize=16)
ax1.set_xlabel(u'日期', fontproperties=font, fontsize=16)
ax1.set_title(u"R平方和之后5日收益的秩相关系数", fontproperties=font, fontsize=16)
ax1.grid()

from sklearn.ensemble import RandomForestClassifier
from CAL.PyCAL import *
import pandas as pd
def train(stocklist,date,period):
    # 创建训练集
    preday=cal.advanceDate(date,period)
    yesterday=cal.advanceDate(date,'-1B')
    
    #创建因子df:fac
    fac=DataAPI.MktStockFactorsOneDayGet(tradeDate=preday,secID=stocklist,field=['secID','LCAP', 'LFLO', 'NPToTOR', 'OperatingProfitGrowRate', 'TotalAssetGrowRate', 'DHILO', 'DEGM', 'Skewness', 'DAREC', 'GREC'],pandas="1")
    #创建价格df:price
    price1=DataAPI.MktEqudAdjGet(secID=stocklist,tradeDate=preday,field=u"secID,closePrice",pandas="1")
    price2=DataAPI.MktEqudAdjGet(secID=stocklist,tradeDate=yesterday,field=u"secID,closePrice",pandas="1")
    price2['closePrice2']=price2['closePrice']
    del price2['closePrice']
    price=pd.merge(price1,price2)
    tmp1=[]
    tmp=(price['closePrice2']-price['closePrice'])/price['closePrice']*100
    for i in tmp:
        tmp1.append(int(i))
    price['zhangdie']=tmp1
    del price['closePrice']
    del price['closePrice2']
    traindf=pd.merge(fac,price)
    traindf.set_index(traindf.secID)
    del traindf['secID']
    traindf=traindf.dropna()
    target=list(traindf['zhangdie'])
    train=[]
    for x in range(0,len(traindf.iloc[:])):
        train.append(list(traindf.iloc[x][0:-1]))
    #构建train列表和target列表完毕
    
    test1 = DataAPI.MktStockFactorsOneDayGet(tradeDate=yesterday,secID=stocklist,field=['secID','LCAP', 'LFLO', 'NPToTOR', 'OperatingProfitGrowRate', 'TotalAssetGrowRate', 'DHILO', 'DEGM', 'Skewness', 'DAREC', 'GREC'],pandas="1")
    test1=test1.dropna()
    test=[]
    for x in range(0,len(test1.index)):
        test.append(list(test1.iloc[x][1:]))
    
    
    
    # 创建并且训练一个随机森林模型,根据上一个交易日的因子预测涨跌,返回预测涨幅最大的前20支股票
    rf = RandomForestClassifier(n_estimators = 1000)  #1000个决策树
    rf.fit(train, target)
    predicted_results = [x for index, x in enumerate(rf.predict(test))]
    test1['predict']=predicted_results    
    test1=test1.sort(columns='predict',ascending=False)
    stock=test1['secID'][:20]
    return stock


start = '2013-01-01'                       # 回测起始时间
end = '2016-06-01'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')  # 证券池，支持股票和基金
capital_base = 1000000                      # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 20                           # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟

data=DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=u"20110101",endDate=u"20160601",field=['calendarDate','isMonthEnd'],pandas="1")
data = data[data['isMonthEnd'] == 1]
date_list = data['calendarDate'].values.tolist()

cal = Calendar('China.SSE')
period = Period('-6M')   #训练日期1个月

def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令

    buylist=train(account.universe,account.current_date,period)
    
                
    for stock in account.avail_secpos.keys():
        if stock in account.universe and stock not in buylist:
            order_to(stock,0)
     

    for stock in buylist: 
        order_pct_to(stock,1./len(buylist))


from sklearn.ensemble import RandomForestClassifier
from CAL.PyCAL import *
import pandas as pd
import numpy as np
import time 
import statsmodels.api as sm



def train(stocklist,to_date):
    tra_day= cal.advanceDate(to_date,'-20B')
    tra_day = str(tra_day)
    tra_day = time.strptime(tra_day,"%Y-%m-%d")
    tra_day = time.strftime("%Y%m%d",tra_day)
    
    tes_day=cal.advanceDate(to_date,'-15B')
    tes_day=str(tes_day)
    tes_day = time.strptime(tes_day,"%Y-%m-%d")
    tes_day = time.strftime("%Y%m%d",tes_day)
    
    yesterday=cal.advanceDate(to_date,'-1B')
    yesterday = str(yesterday)
    yesterday = time.strptime(yesterday,"%Y-%m-%d")
    yesterday = time.strftime("%Y%m%d",yesterday)

    tra_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=tra_day,endDate=yesterday,field=['calendarDate','isOpen'])
    
    tra_dates=tra_dates[tra_dates['isOpen'] == 1]
    #date= map(lambda x: x[0:4]+x[5:7]+x[8:10], tra_dates['calendarDate'].values.tolist())
    tra_dates=tra_dates['calendarDate'].values.tolist()
    step1_date = pd.DataFrame(index=range(len(tra_dates)),columns=['tradeDate'],data=tra_dates)
    #print step1_date
    

    
    
    
    
    tes_dates = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=tes_day,endDate=yesterday,field=['calendarDate','isOpen'])
    tes_dates=tes_dates[tes_dates['isOpen'] == 1]
    #date= map(lambda x: x[0:4]+x[5:7]+x[8:10], date_data['calendarDate'].values.tolist())
    tes_dates=tes_dates['calendarDate'].values.tolist()
    step2_date = pd.DataFrame(index=range(len(tes_dates)),columns=['tradeDate'],data=tes_dates)
    #print step2_date
    
    #ok
    
    
    
    train_all_data = pd.DataFrame(index=stocklist,columns=['R','xielv','dapan','gegu','target'])
    test_all_data = pd.DataFrame(index=stocklist,columns=['R','xielv','dapan','gegu','target'])
    #print test_all_data
    #print train_all_data
    
    #ok
    
    
    mkt_tmp1 = DataAPI.MktIdxdGet(beginDate=tra_day,endDate=yesterday,ticker='000001',exchangeCD=u"XSHE,XSHG",field=u"tradeDate,CHGPct",pandas="1").sort('tradeDate')

    mkt_tmp1 = pd.merge(step1_date,mkt_tmp1,on='tradeDate',how='left')
    #print mkt_tmp1    
    #ok
    mkt_tmp2 = pd.merge(step2_date,mkt_tmp1,on='tradeDate',how='left')
    #print mkt_tmp2
    
    mkt_tmp1.CHGPct = mkt_tmp1.CHGPct.cumsum()
    mkt_tmp2.CHGPct = mkt_tmp2.CHGPct.cumsum()
    
#ok

    
    for stk in stocklist:
        match_stk = pd.DataFrame()
        stk_tmp = DataAPI.MktEqudGet(secID=stk,beginDate=tra_day,endDate=yesterday,field=u"tradeDate,chgPct",pandas="1")
        stk_tmp.chgPct = stk_tmp.chgPct.cumsum()
        match_stk = pd.merge(mkt_tmp1,stk_tmp,on='tradeDate',how='left')
        tmp_x = match_stk.CHGPct
        tmp_y = match_stk.chgPct
        tmp_x = sm.add_constant(tmp_x)
        est=sm.OLS(tmp_y,tmp_x).fit()
        train_all_data.loc[stk]['R'] = est.rsquared
        train_all_data.loc[stk]['xielv'] = est.params[1]
        train_all_data.loc[stk]['dapan'] = match_stk.iloc[-1]['CHGPct']
        train_all_data.loc[stk]['gegu'] = match_stk.iloc[-1]['chgPct']
        train_all_data.loc[stk]['target'] = int(stk_tmp.iloc[-1]['chgPct']*100)
        #ok
    train_all_data = train_all_data.dropna()
    train_all_data = train_all_data[(train_all_data.target>train_all_data.target.quantile(0.8)) | (train_all_data.target<train_all_data.target.quantile(0.2))]
    
    
    target=list(train_all_data['target'])
    train=[]
    for x in range(len(train_all_data.iloc[:])):
        train.append(list(train_all_data.iloc[x][:-2]))
    #print len(train[0])
    for stk2 in stocklist:
        match_train = pd.DataFrame()
        stk_tmp = DataAPI.MktEqudGet(beginDate=tes_day,endDate=yesterday,secID=stk2,field=u"tradeDate,chgPct",pandas="1")
        stk_tmp.chgPct = stk_tmp.chgPct.cumsum()
        match_train = pd.merge(mkt_tmp2,stk_tmp,on='tradeDate',how='left')
        tmp_x = match_train.CHGPct
        tmp_y = match_train.chgPct
        tmp_x = sm.add_constant(tmp_x)
        est=sm.OLS(tmp_y,tmp_x).fit()
        test_all_data.loc[stk2]['R'] = est.rsquared
        test_all_data.loc[stk2]['xielv'] = est.params[1]
        test_all_data.loc[stk2]['dapan'] = mkt_tmp2.iloc[-1]['CHGPct']
        test_all_data.loc[stk2]['gegu'] = stk_tmp.iloc[-1]['chgPct']
        test_all_data.loc[stk2]['target'] = int(stk_tmp.iloc[-1]['chgPct']*100)
    test_all_data = test_all_data.dropna()
    
    test=[]
    for i in range(0,len(test_all_data.index)):
        test.append(list(test_all_data.iloc[i][:-2]))
    print (len(test[0]))
    print (to_date)
        
        
        
        
    rf = RandomForestClassifier(n_estimators = 1000)  #1000个决策树
    rf.fit(train, target)
    results = rf.predict(test)
    test_all_data['predict']=results    
    test_all_data=test_all_data.sort(columns='predict',ascending=False)
    stock=test_all_data.index[:20]
    
    return stock
    




start = '2013-09-15'                       # 回测起始时间
end = '2015-01-01'                         # 回测结束时间
benchmark = 'HS300'                        # 策略参考标准
universe = set_universe('HS300')  # 证券池，支持股票和基金
capital_base = 1000000                      # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 5                           # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟


cal = Calendar('China.SSE')


def initialize(account):                   # 初始化虚拟账户状态
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令

    buylist=train(account.universe,account.current_date)
    
                
    for stock in account.avail_secpos.keys():
        if stock in account.universe and stock not in buylist:
            order_to(stock,0)
     

    for stock in buylist: 
        order_pct_to(stock,1./len(buylist))


bt2 = bt
bt2.portfolio_value = 1-(bt2.portfolio_value.shift()/bt2.portfolio_value)

bt2 = bt2.ix[:,['tradeDate','portfolio_value','benchmark_return']]

bt2.to_csv('bt_170424.csv')


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from CAL.PyCAL import *    # CAL.PyCAL中包含font


a = pd.read_csv('bt_130904.csv')
a = a[a.columns[1:]]
#a.portfolio_value = 1-(a.portfolio_value.shift()/a.portfolio_value)


b = pd.read_csv('bt_150105.csv')
b = b[b.columns[1:]]
b = b.ix[:,['benchmark_return','portfolio_value','tradeDate']]
b.portfolio_value[0] = 0


c = pd.read_csv('bt_150618.csv')
c = c[c.columns[1:]]
c.portfolio_value[0] = 0

d = pd.read_csv('bt_150803.csv')
d = d[d.columns[1:]]
d.portfolio_value[0] = 0

e = pd.read_csv('bt_151120.csv')
e = e[e.columns[1:]]
e.portfolio_value[0] = 0

f = pd.read_csv('bt_170424.csv')
f = f[f.columns[1:]]
f.portfolio_value[0] = 0


alll = a.append(b)
alll = alll.append(c)
alll = alll.append(d)
alll = alll.append(e)
alll = alll.append(f)


alll = alll.ix[:,['benchmark_return','portfolio_value','tradeDate']]


alll['excess_return'] = alll.portfolio_value - alll.benchmark_return
alll['excess'] = alll.excess_return + 1.0
alll['excess'] = alll.excess.cumprod()


alll['portfolio_value'] = alll.portfolio_value + 1.0
alll['portfolio_value'] = alll.portfolio_value.cumprod()

alll['benchmark_return'] = alll.benchmark_return + 1.0
alll['benchmark_return'] = alll.benchmark_return.cumprod()

alll.tradeDate = pd.to_datetime(alll.tradeDate)


fig = plt.figure(figsize=(10,8))
fig.set_tight_layout(True)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.grid()
ax2.grid()




ax1.plot(alll['tradeDate'], alll[['portfolio_value']], label='jingzhi')
ax2.plot(alll['tradeDate'], alll[['excess']], label='alpha')

ax1.plot(alll['tradeDate'], alll[['benchmark_return']], label='HS300')




ax1.legend(loc=0, fontsize=12)
ax2.legend(loc=0, fontsize=12)
ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
ax2.set_ylabel(u"对冲净值", fontproperties=font, fontsize=16)
ax1.set_title(u"随机森林组合净值走势", fontproperties=font, fontsize=16)
ax2.set_title(u"组合对冲指数净值走势", fontproperties=font, fontsize=16)


