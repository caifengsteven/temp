# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 08:29:57 2020

@author: Asus
"""

'''

导读
A. 研究目的：本文利用优矿提供的行情数据和宏观行业因子数据，参考兴业证券《系统化资产配置系列之四:基于长期、中期、中短期择时模型相结合的A股市场择时研究》（原作者：于明明等）中对市场择时模型的介绍，分长期、中期进行通联全A指数择时模型构建，测试各择时因子是否稳定有效。对稳定有效的因子进行中期内合成以及长中期结合合成，探索长中期择时模型在A股市场择时中的应用价值。

B. 研究结论：

长期（季度）择时模型以DRP因子发出的信号为准，进行调仓。在纯多头回测下，年化收益率达到11.3%，夏普比率为0.77，获得了远超标的指数的收益，并且波动率较低，年化波动率仅仅14.7%，收益较稳定。

通过筛选，中期（月度）择时模型因子库共13个因子在回测区间内都带来了超额收益，t值均大于0.8，说明因子预测效果较好。因子提供的择时策略有很大概率可以避开市场大规模的回撤，规避风险。

对于中期（月度）择时模型，剔除相关性高的因子后进行合成。合成后的信号表现大幅超过每个单独因子给出信号的表现。t值达到3.56，多头和多空年化收益率分别达到14.3%和25.3%,模型对于市场的预测能力非常强。

通过朴素贝叶斯模型对长、中期择时模型进行结合，最终结合模型给出的信号与中期（月度）择时模型给出的信号一致。

C. 文章结构：本文共分为5个部分，具体如下

一、基础数据获取、择时模型介绍。

二、构建长期（季度）择时模型。

三、构建中期（月度）择时模型。

四、长、中期择时模型结合。

五、总结。

D. 时间说明

各部分耗时均比较短，全文总耗时在5分钟左右
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)
'''

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import copy
from matplotlib.ticker import FuncFormatter
plt.style.use('seaborn-white')

'''

第一部分：基础数据获取、择时模型介绍
该部分内容为：

1.1 交易日数据获取， 包括交易日，月末交易日，季末交易日
1.2 介绍择时模型基本逻辑
1.3 信号生成机制
1.4 介绍择时效果的评价方法
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

1.1 基础数据获取

获取从2005-04-01至2019-10-01的日历数据，所有交易日，所有月末交易日，所有季末交易日。

'''

print"基础参数设置和数据准备..."

# 基础数据
sdate = '20050401'
edate = '20191001'

# 获取月末交易日
cal_dates_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=sdate, endDate=edate).sort('calendarDate')
cal_dates_df['calendarDate'] = cal_dates_df['calendarDate'].apply(lambda x: x.replace('-', ''))
cal_dates_df['prevTradeDate'] = cal_dates_df['prevTradeDate'].apply(lambda x: x.replace('-', ''))
tradedate = cal_dates_df[cal_dates_df["isOpen"] == 1][["calendarDate"]]
tradedate.rename(columns={"calendarDate":"tradeDate"},inplace=True)
dates = cal_dates_df[["calendarDate"]]
dates.rename(columns={"calendarDate":"tradeDate"},inplace=True)
monthly_dates_list = cal_dates_df[cal_dates_df['isMonthEnd']==1]['calendarDate'].values.tolist()
monthly_dates_df = pd.DataFrame(monthly_dates_list)
monthly_dates_df.rename(columns={0:"tradeDate"},inplace=True)
seasonly_dates_list = []
for date in monthly_dates_list:
    if date[4:6] in ["03","06","09","12"]:
        seasonly_dates_list.append(date)
print "月末交易日",monthly_dates_list[:10]
print "季末交易日",seasonly_dates_list[:10]

'''

1.2 择时模型基本逻辑

参考研报，针对大类资产的中长期择时方法往往有两种思路，一是周期视角下的大类资产轮动，典型模型如美林时钟；另一种是多因子择时方法，其尝试从估值、增长、通胀、流动性、情绪等不同角度对指数涨跌进行预测。由于投资者结构，市场机制等差异，A股市场与成熟市场相比，具有更高的波动，并且常常不与经济发展同步。因此，结合A股实际情况，当前多因子框架下的择时方案更能够全面描述A股市场的特征。
在长期、中期择时中，因子的内在逻辑更为关键。在长期（季度）择时模型中，投资者一般更加关注股票市场当前的估值水平、经济长期增长等指标；在中期（月度）择时模型中，投资者不仅要关注估值等指标，也会关心社会所处的经济环境、资金流动性水平、通胀水平、市场的风险偏好等。

1.3 信号生成机制

参考研报，使用两分位法则生成择时交易信号。统计通联全A历史收益率，并将其分成上涨和下跌两种情况，以每个季度为例，我们发现在季度频率上万得全A上涨的频率为50.9%，下跌的频率为49.1%，从季度层面来看上涨的频率大于下跌的频率，因此长期择时（季度）发出看涨的信号应该也要多于看跌的信号才能获得更高的胜率，所以根据择时标的本身的历史上涨下跌频率来确定因子的分位点。把时间段内每个因子按大小排序（以正向因子为例），因子值排前50.9%的发出看多信号，剩下49.1%发出看空信号。


'''


index_ret = DataAPI.MktIdxdGet(indexID=u"",ticker=u"DY0001",tradeDate=u"",beginDate=sdate,endDate=edate,exchangeCD=u"XSHE,XSHG",field=u"tradeDate,closeIndex",pandas="1")
index_ret["tradeDate"] = index_ret["tradeDate"].apply(lambda x: x.replace('-', ''))
index_ret_ = index_ret[index_ret["tradeDate"].isin(seasonly_dates_list)]
index_ret_["closeIndexlast"] = index_ret_["closeIndex"].shift(1)
index_ret_.dropna(inplace=True)
index_ret_["diff"] = index_ret_["closeIndex"] - index_ret_["closeIndexlast"]
up = (index_ret_["diff"]>0).sum()
down = (index_ret_["diff"]<0).sum()
point = float(up)/(up+down)
print '长期(季度)正向因子的分位点: %s' %round(point,3)

'''

1.4 择时效果处理方法

参考研报，构建t统计量如下，可以反映因子的择时效果：
t = F¯¯¯1 − F¯¯¯3(n1−1)S12 − (n3−1)S32n1+n3−2⸳(1n1+1n3)−−−−−−−−−−−−−−−−−−−−−−−−√
其中F¯¯¯1表示发出看多信号时未来一期标的收益率的均值；F¯¯¯3表示发出看空信号时未来一期标的收益率的均值；S21表示发出看多信号时未来一期标的收益率的方差；S23表示发出看空信号时未来一期标的收益率的方差；n1表示发出看多信号的样本容量；n3表示发出看空信号的样本容量。
t统计量的值和夏普比率（不考虑手续费和交易摩擦）相关性非常高，从而可以用t统计量是否显著作为因子预测效果的重要衡量指标。t值越大，说明该因子发出看多和看空信号情况下未来收益的差异越明显，其预测效果越好。

'''

#定义计算t值的函数
def t(f1,f3,s1,s3,n1,n3):
    return (f1-f3)/((((n1-1.0)*s1+(n3-1.0)*s3)/(n1+n3-2.0)*(1.0/n1+1.0/n3))**0.5)


'''

第二部分：构建长期（季度）择时模型
该部分耗时 1分钟
该部分内容为：

2.1 构建长期择时因子库。
2.2 对因子分别进行回测，筛选。
2.3 检验因子之间的相关性。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

2.1 构建长期择时因子库

如上文所述，长期择时模型主要关注估值和经济增长指标。具体而言，估值指标包括股权风险溢价、PE、PB、股息率等，除开最重要的估值指标，长期择时模型同样会关注经济的发展情况，如GDP增长。归纳起来，本研究中长期择时模型所选用的指标及择时方向展示如下：

– GDP:不变价:当季同比  正向
– ERP:股权风险溢价（基于通联全A） 正向
– DRP:中证红利的股息率-无风险利率 正向

下面获取宏观数据：无风险利率（指标ID:1090002455），股息率(指标ID:1170007470)，GDP:不变价:当季同比(指标ID:1010008414)。对于许多宏观数据，例如GDP来说，发布时间往往具有滞后性。为了避免利用未来数据，影响模型的真实性，当期调仓所使用的数据实际上是上一期的指标数据。




'''

def set_pubdate(df):
    # pub_date = df['publishDate']
    period_date = df['periodDate']
    # 发布日期调整: 1期之后
    pub_dll = datetime.datetime.strptime(period_date, '%Y-%m-%d')
    pub_dll = pub_dll + datetime.timedelta(days=28)
    pub_dll = pub_dll.strftime("%Y-%m-%d")
    return pub_dll
#获取GDP季度同比数据(指标ID:1010008414)
macro_data = DataAPI.EcoDataProGet('1010008414',sdate,edate)
macro_data["periodDate"] = macro_data.apply(lambda x: set_pubdate(x), axis=1)
#获取无风险利率(指标ID:1090002455)，股息率数据(指标ID:1170007470)
macro_data = pd.concat([macro_data,DataAPI.EcoDataProGet('1090002455,1170007470',sdate,edate)])
macro_data = pd.pivot_table(macro_data,index="periodDate",columns="indicID",values="dataValue").reset_index()
macro_data.columns.name=None
macro_data["periodDate"] =macro_data["periodDate"].apply(lambda x: x.replace('-', ''))
macro_data.rename(columns={1090002455:"riskfree_r",1170007470:"dividend_ratio",1010008414:"GDP","periodDate":"tradeDate"},inplace=1)
print macro_data[-3:]

#以上证A股为标的指数，获取PE数据
PE_data = DataAPI.MktIdxFactorDateRangeGet(secID=u"",ticker=u"000002",beginDate=sdate,endDate=edate,field=u"",pandas="1")
PE_data = PE_data[["tradeDate","PE1"]]
PE_data["tradeDate"] = PE_data["tradeDate"].apply(lambda x: x.replace('-', ''))

#计算因子
season_df = pd.merge(PE_data,macro_data,how="left")
season_df.fillna(method = 'ffill',inplace=True)
season_df = season_df[season_df["tradeDate"].isin(seasonly_dates_list)]
season_df["ERP"] = 100/season_df["PE1"]-season_df["riskfree_r"]
season_df["DRP"] = season_df["dividend_ratio"]-season_df["riskfree_r"]
factor_list = ["GDP","ERP","DRP"]
season_df = season_df[["tradeDate"]+factor_list]

'''

2.2 对因子分别进行回测，筛选

构建回测框架。交易费率设为0.001，每个季度末的交易日以收盘价调仓，交易标的为通联全A指数。
对上述生成的三个因子分别进行回测。

'''

#该函数用于判断是否需要调仓
def get_order_day(s,startday=False):
    r=[startday]
    for i in range(1,len(s)):
        if s[i]==s[i-1]:
            r.append(False)
        else:
            r.append(True)
    return r 

def calc_t(index_ret,df,factor,kt):
    calc_t = pd.merge(index_ret,df,how="right")
    calc_t[factor] = calc_t[factor].shift(1)
    calc_t["ret"] = calc_t["closeIndex"].pct_change()
    calc_t.dropna(inplace=True)
    calc_t_1 = calc_t[calc_t[factor] == 1]
    calc_t_0 = calc_t[calc_t[factor] == kt]
    calc_t_1 = calc_t_1["ret"].tolist()
    calc_t_0 = calc_t_0["ret"].tolist()
    f1 = np.mean(calc_t_1)
    f3 = np.mean(calc_t_0)
    s1 = np.var(calc_t_1)
    s3 = np.var(calc_t_0)
    n1 = len(calc_t_1)
    n3 = len(calc_t_0)
    return t(f1,f3,s1,s3,n1,n3)

#回测函数，并且计算收益率，波动率，换手次数，波动比，kt参数控制回测的方式，0代表多头回测，-1代表多空回测
def back_test_season(dff,factorlist,index_ret,dates,cut_point,show=1,kt=-1,cost = 0.001):
    print "回测中......\n"
    if show:
        if kt == 0:
            print "多头回测："
        else:
            print "多空回测："
    df = copy.deepcopy(dff)
    data = []
    rtt = pd.DataFrame()
    start = dff["tradeDate"].iloc[0]
    end = dff["tradeDate"].iloc[-1]
    #自然年数计算
    years = (dates[dates["tradeDate"]==end].index.values[0] - dates[dates["tradeDate"]==start].index.values[0])/365.0
    # print start,end
    # print dates[dates["tradeDate"]==end].index.values[0], dates[dates["tradeDate"]==start].index.values[0]
    for factor in factorlist:
        df[factor] = df[factor].rank(ascending=False,pct=True).apply(lambda x: 1.0 if x<=cut_point else float(kt))
        t = calc_t(index_ret,df,factor,kt)
        everyday_ret = pd.merge(index_ret,df,how="left")
        everyday_ret.fillna(method="ffill",inplace=True)
        everyday_ret.fillna(0,inplace=1)
        everyday_ret["trade_day"] = get_order_day(everyday_ret[factor].tolist())
        everyday_ret["pct"] = everyday_ret["closeIndex"].pct_change().shift(-1)
        #交易费率设置        
        everyday_ret["strategy_pct"] = everyday_ret["pct"]*everyday_ret[factor]
        everyday_ret["strategy_pct"] = everyday_ret["strategy_pct"] - cost * everyday_ret["trade_day"]
        huanshouci = everyday_ret["trade_day"].sum()/years
        var = np.std(everyday_ret["strategy_pct"].tolist()[:-1])*(250**0.5)
        dates_ = dates[(dates["tradeDate"]>=start) & (dates["tradeDate"]<=end)]
        everyday_ret = pd.merge(dates_,everyday_ret,how="left")
        everyday_ret = everyday_ret[["tradeDate","pct","strategy_pct"]]
        everyday_ret.fillna(0,inplace=1)
        everyday_ret["index_ret"] = (everyday_ret["pct"]+1).cumprod()
        everyday_ret["str_cum_pct"] = (everyday_ret["strategy_pct"]+1).cumprod()
        ret = (everyday_ret["str_cum_pct"].tolist()[-1])**(1.0/years)-1
        if show:
            plt.plot(pd.to_datetime(everyday_ret["tradeDate"]),everyday_ret.str_cum_pct,label=unicode(factor, "utf-8")) 
            plt.title(u"net value curve of factor timing and index ")
        line = pd.DataFrame({"t值":t,"年化收益率":ret,"年化波动率":var,"收益波动比":ret/var,"年换手次数":huanshouci},index=[factor])
        data.append(line)
        rt = everyday_ret["str_cum_pct"]
        rtt[factor] = rt
    if show:
        plt.plot(pd.to_datetime(everyday_ret["tradeDate"]),everyday_ret.index_ret)
        plt.legend()
        data1 = pd.concat(data).applymap(lambda x: round(x,3))
        # data1 = pd.concat(data)
        print(data1.to_html())
        rtt1 = rtt.corr().applymap(lambda x: round(x,3))
        print("指标收益序列的相关系数：")
        print (rtt1.to_html())
    return df

#长期择时模型多头回测
df_season = back_test_season(season_df,["GDP","ERP","DRP"],index_ret,dates,point,show=1,kt=0)

#长期择时模型多空回测
df_season = back_test_season(season_df,["GDP","ERP","DRP"],index_ret,dates,point,show=1,kt=-1)

'''

2.3 结果分析

DRP因子与ERP因子的t值均大于1，说明这两个因子发出看多和看空信号未来收益的差异明显，预测效果较好。GDP因子t值小于0，预测效果较差，从调仓结果来看，近年来GDP同比增速放缓，使得因子在近年持续放出看空信号。
DRP因子在纯多头和多空回测下，年化收益率分别达到11.3%和13%，夏普比率为0.77和0.52，获得了远超标的指数的收益，并且波动率较低，收益较稳定。
ERP因子的回测结果也较好，但是该指标和DRP因子的收益序列相关度较高，多头回测方式下达到0.96，实际应用中，选取DRP因子即可。
综上，长期（季度）择时模型以DRP因子发出的信号为准，进行调仓。

第三部分：构建中期（月度）择时模型。
该部分内容为：

3.1 构建中期择时因子库。
3.2 对因子进行回测，结果展示。
3.3 对所有入选因子进行信号合成。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

3.1 构建中期择时因子库

中期（月度）择时不仅关注市场的估值指标，它以股利贴现模型为核心，借助经济周期判断企业盈利状况，需要全面关注物价水平、资金流动性和市场风险偏好等多方面的影响。经过对通联宏观行业数据库中数百个因子的筛选，最终入选的中期择时因子及择时方向展示如下:
中债企业债(AA)与国债利差:1年 正向
全国居民人均可支配收入:累计值:差分 正向
中债国债到期收益率:1个月:差分 负向
银行间同业拆借加权利率:1天:差分 负向
月均银行间回购加权利率:7天:差分 负向
银行间同业拆借加权利率:14天:差分 负向
固定资产投资完成额:新建:累计同比:差分 正向
社会消费品零售总额:当月同比(扣除价格因素) 正向
原油价格指数:Brent 负向
股息率加权平均指标 正向
中债国债到期收益率:10年 负向
市场动量:一个月的变化率 正向
市场动量:三个月的变化率 正向
由于基本面指标更新频率存在差异，在进行中期因子的处理时，需要将因子统一到同一频率中：如果为低频因子（如全国居民人均可支配收入每个季度发布一次），将其填充为月频，未发布月份的数据用上一期数据填补；若为高频因子（如每日发布的Shibor利率），取月末数据或者月均值进行每月数据的填充，保证中期因子的数据频率统一为月频。




'''

#上述所选因子对应的指标ID如下：
indic_ids = ['1050000046','1110000009','1140000004','1090000554','1090002455','1090002720', '2030610441','1090000556','1170007470', '1090001385','1090000587']

#将ID与名称一一对应
name = DataAPI.EcoInfoProGet(indicID=indic_ids,field=['indicID','indicName', 'frequency'],pandas="1").set_index("indicID")
IDs = name.index.tolist()
names = name["indicName"].tolist()
dic = dict(zip(IDs,names))
IDs = map(str,IDs)
dic_df = dict(zip(IDs,names))
dic_df["index_1m"] = "市场动量:一个月的变化率"
dic_df["index_3m"] = "市场动量:三个月的变化率"
name_df = pd.DataFrame(pd.Series(dic_df)).rename(columns={0:"name"})
print name_df.to_html()

'''

3.2 对因子进行回测，结果展示

构建回测框架。交易费率设为0.001，每个月末的交易日以收盘价调仓，交易标的为通联全A指数。
对上述因子分别进行回测。

'''


#考虑到指标历史数据的长度不统一，设定回测区间为2008.04-2019.10。
sdate = '20080401'
edate = '20190930'

#生成相对应区间的分位点和指数收益
def generate_index(start,end):
    index_ret = DataAPI.MktIdxdGet(indexID=u"",ticker=u"DY0001",tradeDate=u"",beginDate=start,endDate=end,exchangeCD=u"XSHE,XSHG",field=u"tradeDate,closeIndex",pandas="1")
    index_ret["tradeDate"] = index_ret["tradeDate"].apply(lambda x: x.replace('-', ''))
    index_ret_ = index_ret[index_ret["tradeDate"].isin(monthly_dates_list)]
    index_ret_["closeIndexlast"] = index_ret_["closeIndex"].shift(1)
    index_ret_.dropna(inplace=True)
    index_ret_["diff"] = index_ret_["closeIndex"] - index_ret_["closeIndexlast"]
    up = (index_ret_["diff"]>0).sum()
    down = (index_ret_["diff"]<0).sum()
    point = float(up)/(up+down)
    return index_ret,point


#生成信号
# def get_signal(indic_id,delay,diff,start=sdate,end=edate):
def get_signal(indic_id,delay,diff,start,end):    
    print "当前指标：" + dic[int(indic_id)]
    factor_data = DataAPI.EcoDataProGet(indic_id,sdate,edate)
    factor_data["indicID"] = factor_data["indicID"].apply(str)
    #处理延迟发布数据
    if delay:
        factor_data["periodDate"] = factor_data.apply(lambda x: set_pubdate(x), axis=1)
        #print "发布日期延迟处理"
    factor_data["periodDate"] = factor_data["periodDate"].apply(lambda x: x.replace('-', ''))
    #确定回测结束期
    e = factor_data["periodDate"][0]
    e = [x for x in monthly_dates_list if x<=str(e)][-1]
    if e < end:
        end = e
    #统一为月度数据
    factor_data = pd.merge(dates,factor_data,how="left",right_on="periodDate",left_on="tradeDate")
    #print factor_data
    factor_data.fillna(method="ffill",inplace=True)
    #print factor_data
    factor_data.dropna(subset=['dataValue'],inplace=True)
    factor_data = factor_data[factor_data["tradeDate"].isin(monthly_dates_list)]
    factor_data = factor_data[factor_data["tradeDate"]<=e]
    if diff:
        factor_data["last"] = factor_data["dataValue"].shift(1)
        factor_data.dropna(subset=['dataValue'],inplace=True)
        factor_data["dataValue"] = factor_data["dataValue"] - factor_data["last"]
        print "处理方式：差分处理"
    #确定回测开始期
    s = factor_data["tradeDate"].iloc[0]
    if s > start:
        start = s
    index_ret,point = generate_index(start,end)

    factor_data = pd.pivot_table(factor_data,index="tradeDate",columns="indicID",values="dataValue").reset_index()
    factor_data.columns.name=None
    return index_ret,start,end,point,factor_data
    
    
#回测函数
def back_test_month(dff,index_ret,factor,point,start,end,kt,plot=False,show=True,adj=True,neg=False,cost = 0.001):
    if show:
        # print "回测中......\n"
        if kt == 0:
            print "多头回测："
        else:
            print "多空回测："

        print '回测区间：%s-%s'%(start,end)
    df = dff.copy()
    years = (dates[dates["tradeDate"]==end].index.values[0] - dates[dates["tradeDate"]==start].index.values[0])/365.0
    if adj:
        if neg:
            df[factor] = df[factor].rank(ascending=False,pct=True).apply(lambda x: kt if x<=(1-point) else 1)
            # print "负向处理"
        else:
            df[factor] = df[factor].rank(ascending=False,pct=True).apply(lambda x: 1 if x<=point else kt)
    #print df
    t = calc_t(index_ret,df,factor,kt)
    everyday_ret = pd.merge(index_ret,df,how="left")
    everyday_ret.fillna(method="ffill",inplace=True)
    everyday_ret.fillna(0,inplace=1)
    everyday_ret["trade_day"] = get_order_day(everyday_ret[factor].tolist())
    everyday_ret["pct"] = everyday_ret["closeIndex"].pct_change().shift(-1)
    
    everyday_ret["strategy_pct"] = everyday_ret["pct"]*everyday_ret[factor]
    everyday_ret["strategy_pct"] = everyday_ret["strategy_pct"] - cost * everyday_ret["trade_day"]
    huanshouci = everyday_ret["trade_day"].sum()/years
    var = np.std(everyday_ret["strategy_pct"].tolist()[:-1])*(250**0.5)
    dates_ = dates[(dates["tradeDate"]>=start) & (dates["tradeDate"]<=end)]
    everyday_ret = pd.merge(dates_,everyday_ret,how="left")
    everyday_ret = everyday_ret[["tradeDate","pct","strategy_pct"]]
    everyday_ret.fillna(0,inplace=1)
    everyday_ret["index_ret"] = (everyday_ret["pct"]+1).cumprod()
    everyday_ret["str_cum_pct"] = (everyday_ret["strategy_pct"]+1).cumprod()
    ret = (everyday_ret["str_cum_pct"].tolist()[-1])**(1.0/years)-1

    line = pd.DataFrame({"t值":t,"年化收益率":ret,"年化波动率":var,"收益波动比":ret/var,"年换手次数":huanshouci},index=[factor])
    if plot:
        plt.plot(pd.to_datetime(everyday_ret["tradeDate"]),everyday_ret.str_cum_pct,label=unicode(factor, "utf-8"))
        plt.plot(pd.to_datetime(everyday_ret["tradeDate"]),everyday_ret.index_ret)
        plt.title(u"net value curve of factor timing and index ")
        plt.legend()
        plt.show()
    if show:
        line1 = line.applymap(lambda x: round(x,3))
        print(line1.to_html())
    return df,line


def bt(indic_id,delay,diff,sdate,edate,kt=-1,plot=False,show=True,adj=True,neg=False):
    index_ret,start,end,point,dff = get_signal(indic_id,delay,diff,sdate,edate)
    df,line = back_test_month(dff,index_ret,indic_id,point,sdate,edate,kt,plot,show,adj,neg)
    return df,line


#进行回测
total_factors = ['1050000046','1110000009','1140000004','1090000554','1090002455','1090002720', '2030610441','1090000556','1170007470', '1090001385','1090000587']
adjust_factors =  ['1050000046','1110000009','1140000004']
neg_factors = ["1090000554","1090002455","2030610441","1090000556","1090001385","1090000587"]
dif_factors = ['1050000046', '1140000004', '1090000554', '1090000556', '1090001385','1090000587']
left_factors = list(set(total_factors).difference(set(adjust_factors)))
conclude = []
for i in adjust_factors:
    dif = i in dif_factors
    ne = i in neg_factors
    df,line = bt(str(i),1,dif,sdate,edate,kt=0,plot=1,neg=ne)
    conclude.append(line)
for i in left_factors:
    dif = i in dif_factors
    ne = i in neg_factors
    df,line = bt(str(i),0,dif,sdate,edate,kt=0,plot=1,neg=ne)
    conclude.append(line)
    
#回测动量指标
print "一月动量指标"
TLQA = DataAPI.MktIdxdGet(indexID=u"",ticker=u"DY0001",tradeDate=u"",beginDate=sdate,endDate=edate,exchangeCD=u"XSHE,XSHG",field=u"tradeDate,closeIndex",pandas="1")
TLQA["tradeDate"] = TLQA["tradeDate"].apply(lambda x: x.replace('-', ''))
TLQA = TLQA[TLQA["tradeDate"].isin(monthly_dates_list)]
TLQA["last1mIndex"] = TLQA["closeIndex"].shift(1)
TLQA.dropna(inplace=True)
TLQA["index_1m"] = TLQA["closeIndex"] > TLQA["last1mIndex"]
TLQA["index_1m"] = TLQA["index_1m"].apply(lambda x: 1 if x == True else 0)
TLQA_1m = TLQA[["tradeDate","index_1m"]]
index_1m,line = back_test_month(TLQA,index_ret,"index_1m",point,sdate,edate,0,plot=True,show=True,adj=False,neg=False)
conclude.append(line)

print "三月动量指标"
TLQA = DataAPI.MktIdxdGet(indexID=u"",ticker=u"DY0001",tradeDate=u"",beginDate=sdate,endDate=edate,exchangeCD=u"XSHE,XSHG",field=u"tradeDate,closeIndex",pandas="1")
TLQA["tradeDate"] = TLQA["tradeDate"].apply(lambda x: x.replace('-', ''))
TLQA = TLQA[TLQA["tradeDate"].isin(monthly_dates_list)]
TLQA["last3mIndex"] = TLQA["closeIndex"].shift(3)
TLQA.dropna(inplace=True)
TLQA["index_3m"] = TLQA["closeIndex"] > TLQA["last3mIndex"]
TLQA["index_3m"] = TLQA["index_3m"].apply(lambda x: 1 if x == True else 0)
TLQA = TLQA[["tradeDate","index_3m"]]
index_3m,line = back_test_month(TLQA,index_ret,"index_3m",point,sdate,edate,0,plot=True,show=True,adj=False,neg=False)
conclude.append(line)   
    


#展示汇总结果
per_f_all = pd.merge(name_df,pd.concat(conclude),how="right",left_index=True, right_index=True).set_index("name")
per_f_all = per_f_all.applymap(lambda x: round(x,3))
print per_f_all.to_html()

'''

由上面展示的图表可见，挑选出的因子库里面的因子在回测区间内，都带来了超额收益，t值均大于0.8，说明所选的因子预测效果较好。
以表现最好的股息率加权平均指标为例，多头回测中t值达到2.39，年化收益率达到11.6%，夏普比率达到0.71，说明该因子提供了稳定超越指数的收益。由收益图可以看出，因子提供的择时策略有很大概率可以避开市场较大的回撤，规避了风险。
   调试 运行
文档
 代码  策略  文档
3.3 对所有入选因子进行信号合成

对上述所有因子发出的信号进行投票式合成。规则是如果超过一半的因子发出同一信号即为合成信号，当期按照信号进行调仓；若发出信号的因子不足一半，则不生成当期信号，即当期不进行调仓。
由于"市场动量:一个月的变化率"与"市场动量:三个月的变化率"两个因子相关性高，故只选择表现较好的"市场动量:一个月的变化率"因子进入信号合成。
对信号合成结果进行回测。

'''

#投票式合成
def voting(factor_list,adjust_factors,dif_factors,neg_factors,start_date,end_date,others=False,kt=-1):
    monthly_in_range = monthly_dates_df[(monthly_dates_df["tradeDate"]>=start_date) & (monthly_dates_df["tradeDate"]<=end_date)]
    for i in factor_list:
        monthly_in_range = pd.merge(monthly_in_range,bt(i,(i in adjust_factors),(i in dif_factors),start_date,end_date,kt=0,plot=0,show=0,adj=True,neg=(i in neg_factors))[0],how="left")
    if others:
        for df in others:
            monthly_in_range = pd.merge(monthly_in_range,df,how="left")
    # monthly_in_range
    monthly_in_range["sum"] =(monthly_in_range==1).sum(axis=1)
    monthly_in_range["vote"] = len(total_factors) - monthly_in_range.isnull().sum(axis=1)
    monthly_in_range = monthly_in_range[monthly_in_range["vote"]>=(len(total_factors)/2.0)]
    monthly_in_range["signal"] = (monthly_in_range["sum"]>=(monthly_in_range["vote"]/2.0))
    monthly_in_range["signal"] = monthly_in_range["signal"].apply(lambda x : 1 if x == True else kt)
    monthly_in_range = monthly_in_range[["tradeDate","signal"]]
    monthly_in_range = pd.merge(monthly_dates_df[(monthly_dates_df["tradeDate"]>=start_date) & (monthly_dates_df["tradeDate"]<=end_date)],monthly_in_range,how="left")
    monthly_in_range.fillna(method="ffill",inplace=True)
    monthly_in_range.dropna(subset=['signal'],inplace=True)
    return monthly_in_range

#信号合成,多头回测
other = [index_1m]
kt=0
monthly_in_range = voting(total_factors,adjust_factors,dif_factors,neg_factors,sdate,edate,others=other,kt=kt)
s = monthly_in_range["tradeDate"].iloc[0]
e = monthly_in_range["tradeDate"].iloc[-1]
index_ret,point = generate_index(s,e)
df_month,line = back_test_month(monthly_in_range,index_ret,"signal",point,s,e,kt,plot=True,adj=False)

#信号合成，多空回测
other = [index_1m]
kt = -1
monthly_in_range = voting(total_factors,adjust_factors,dif_factors,neg_factors,sdate,edate,others=other,kt=kt)
s = monthly_in_range["tradeDate"].iloc[0]
e = monthly_in_range["tradeDate"].iloc[-1]
index_ret,point = generate_index(s,e)
df_month1,line = back_test_month(monthly_in_range,index_ret,"signal",point,s,e,kt,plot=True,adj=False)

'''

由上图可见，合成后的信号表现大幅超过每个单独因子给出信号的表现。t值达到3.56，多头和多空的年化收益分别达到14.3%和25.3%，说明模型对于市场的预测能力非常强。
   调试 运行
文档
 代码  策略  文档
第四部分：长、中期择时模型结合
该部分内容为：

4.1 介绍朴素贝叶斯分类器。
4.2 使用朴素贝叶斯分类器对长、中期择时模型进行结合。
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

   调试 运行
文档
 代码  策略  文档
4.1 介绍朴素贝叶斯分类器

根据贝叶斯定理，P(A/B)=P(B/A)∗P(A)P(B)。朴素贝叶斯分类器的目的即为考虑了一些现有的因素后，随机事件会以多大概率出现各种情况，通过参考这个结果，我们针对性地作出决策。
由于2个模型的信号结合，只可能有4种可能的情况：(0,0),(0,1),(1,0),(1,1)。只需要将其枚举，就可以判断在给定信号组合的条件下，市场上涨和下跌哪个可能性更大。
   调试 运行
文档
 代码  策略  文档
4.2 使用朴素贝叶斯分类器对长、中期择时模型进行结合

将因子信号结合后的中期（月度）择时模型与长期（季度）择时模型相结合，统计每种情况下市场涨跌情况。


'''

df_season = back_test_season(season_df,["GDP","ERP","DRP"],index_ret,dates,point,show=0,kt=0)
df_season["s_signal"] = df_season["DRP"]
df_season = df_season[["tradeDate","s_signal"]]

combine = pd.merge(df_month,df_season,how="left")
combine = pd.merge(combine,index_ret,how="left")

combine["nxt_index"] = combine["closeIndex"].shift(-1)
combine["index_ret"] = (combine["nxt_index"] > combine["closeIndex"])
combine.fillna(method="ffill",inplace=1)
combine.dropna(inplace=True)
combine["index_ret"] = combine["index_ret"].map(int)
combine["s_signal"] = combine["s_signal"].map(int)
combine["signal"] = combine["signal"].map(int)
combine["season,month"]=combine["s_signal"].map(str)+","+combine["signal"].map(str)

combine = combine[["season,month","index_ret"]]
combine = (pd.DataFrame(combine.groupby(["season,month","index_ret"])["season,month"].count()))
combine.rename(columns={"season,month":"count"},inplace=1)
combine.reset_index(inplace=1)
pd.pivot_table(combine,index="season,month",columns="index_ret")["count"]

'''

由上表可见，(季度，月度)信号为(0,0)时，市场有更大概率下跌，即发出看空信号；为(0,1)时，市场有更大概率上涨，即发出看多信号；为(1,0)时，市场有更大概率下跌，即发出看空信号；为(1,1)时，市场有更大概率上涨，即发出看多信号。
可以发现，最终结合模型给出的信号与中期（月度）择时模型给出的信号一致,同时，当季度模型与月度模型同时给出看多信号时，市场上涨的可能性很高，达到66.7%。
   调试 运行
文档
 代码  策略  文档
第五部分：总结
   调试 运行
文档
 代码  策略  文档
本文利用优矿提供的行情数据和宏观行业因子数据，分长期、中期进行A股指数择时模型构建，测试各择时因子是否稳定有效。进一步对稳定有效的因子进行中期内合成以及长中期结合合成，结果表明，长期（季度）择时模型以DRP因子发出的信号为准，进行调仓。在纯多头回测下，年化收益率达到11.3%，夏普比率为0.77，获得了远超标的指数的收益，并且波动率较低，年化波动率仅仅14.7%，收益较稳定；中期（月度）择时模型因子库共13个因子在回测区间内都带来了超额收益，t值均大于0.8，因子预测效果较好；合成后的信号表现大幅超过每个单独因子给出信号的表现，t值达到3.56，多头和多空年化收益率分别达到14.3%和25.3%,模型的择时能力非常强。

'''