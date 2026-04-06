# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:29:04 2020

@author: Asus
"""
'''

导读
A. 研究目的：本文复现广发、国泰、光大情绪类因子的相关研报，使用行为金融学理论，构建量价情绪类因子。

B. 研究结论：

1)极端收益对投资者的影响更大。不受投资者关注的股票，未来一段时间收益会更明显；投资者存在时间偏见，更关注近期的数据；个股持有者大多是理性投资者的话，未来收益会更高。
2)效用因子、显著性因子、行为偏差因子、CGO因子本身具有一定选股能力。但与动量、反转因子间相关性较高，剥离后，因子表现会变差。
3)由于因子间存在相关性，经过因子正交后，再进行ICIR合成，效果会更好。
4)使用合成后因子，构建主动增强策略，近3年在沪深300投资域，由于在市值风格上的负向暴露，导致主动收益有所回撤。
C. 文章结构：本文共分为4个部分，具体如下

一、数据准备和预处理。通过调用DataAPI或自算的方式，计算行为金融学相关因子。
二、单因子测试。进行传统单因子测试，并验证因子方向是否符合预期；验证与动量、反转因子的相关性，测试剥离后的表现。
三、合成因子。将前两步计算的单因子，进行对称正交化、ICIR合成，验证效果。
四、构建主动增强策略。利用合成后的因子，控制主动风险并进行组合构建，回测结果。
D. 运行时间说明

一、数据准备和预处理，需要4小时。由于计算频度为日频，平均每个因子耗时1h，可以通过修改start_date/end_date来缩短时间
二、单因子测试，需要20分钟
三、合成因子，需要10分钟
四、构建主动增强策略，需要10分钟
E. 参考文献

广发证券-行为金融因子研究系列之三：结合凸显理论的选股研究-180115.pdf
国泰君安_数量化专题之九十八：基于前景理论的选股策略_20170718.pdf
光大证券-多因子系列报告之六：行为金融因子：噪音交易者行为偏差-20171120.pdf
   调试 运行
文档
 代码  策略  文档
（一）、情感心理学

过度自信(Overconfidence)。Daniel Kahnemn认为，过度自信来源于投资者对概率事件的错误估计，人们对于小概率事件发生的可能性产生过高的估计，认为其总是可能发生的，这也是各种博彩行为的心理依据；而对于中等偏高程度的概率性事件，易产生过低的估计；但对于90%以上的概率性事件，则认为肯定会发生。

保守主义(Conservation)。指人们的思想大都存在一种惰性，改变个人的原有信念总是非常难的，新的证据对原有信念的修正往往不足，特别是当新的数据并非来源于一个显而易见的模型，人们就不会对它给予足够的重视，不能按照贝叶斯法则修正自己的信念。保守主义表现为人们过于重视了先验概率，而忽视了条件概率。

模糊厌恶(Ambiguity Aversion)。模糊厌恶，即对主观的或含糊的不确定性的厌恶程度要超过对客观不确定性的厌恶。具体到金融市场，客观的不确定性指诸如市场上的政策风险，国家风险等客观的风险因素，而主观上的不确定性则是指诸如人们对某一上市公司的价值判断，对某一政策变化对市场影响是正面或是负面等等主观判断。

心境(Mood)。心境是指人的情绪也即心境对人的投资判断等有着显著的影响一项研究表明，股市每日的交易量和阳光量有着明显的统计相关关系。因此，当人处于不同的心境之下时，对同一项投资做出的决策很可能就是不同的。

后悔厌恶(Regret Aversion)。由于害怕引起后悔，投资者会有强烈的从众心理，购买受到大家一致追捧的股票，因为既使股价下跌，当考虑到大家都同样遭受损失时，或许会减轻投资者的后悔反应。（这即是最小化后悔）。同样，根据忽略偏见（omission bias）理论，当股价下跌时，投资者会倾向于继续持有股票（不采取行动），以免出现一旦卖出股票后（采取行动）股价却反弹所带来的更为强烈的后悔心理的情况。

损失厌恶(Loss Aversion)。期望理论认为，损失厌恶反映了人们的风险偏好并不是一致的，当涉及的是收益时，人们表现为风险厌恶；当涉及的是损失时，人们则表现为风险寻求。

时间偏好(Time preferences)。大量的心理学实验研究指出人们是按照双曲线来贴现将来预测的效用值的，其特征是人们对近期的增加时差要比远期增加的时差的贴现值更大一些。因此，一个人今天对将来某个时差与将来对同一时差的偏好是不同的，也就是说偏好是时间不一致的。

（二）、认知心理学

1、认知的方式

代表性法则(Representative)。代表性启发是指当个体进行判断时，将所得信息与头脑中已存在的类似某种原形的概念进行比较，当偏差较小时，个体便迅速判断该信息很可能代表该原型概念。

可利用性法则(Availability)。它是指人们往往根据一个客体或事件在知觉或记忆中的可得性程度来评估其出现概率，而不是去寻找其它相关的信息，容易被知觉到或回想起的被认为更容易出现。

锚定与调整法则(Anchoring and Adjustment)。锚定与调整法则是指在没有把握的情况下，人们常常利用某个参照点和锚来降低模糊性，然后再通过一定的调整来得出最后的结论。金融市场上，当投资者对某种股票形成较稳定的看法后，其就会在一定程度上被锚定在这种看法上，并以此为基准形成对该股票将来表现的预期判断。因而当该股票基本面信息（如每股盈利）变化时，投资者在进行下一期预测时，受制于锚定的影响而不能做出充分调整。

2、认知的偏差

确认偏差（Confirmation Bias）。即一旦人们形成先验信念，他们就会有意识地寻找有利于证实先验信念的各种证据，而无论事后的结果正确与否。

阿Q精神（Action-induced attitude change）。指人们的信念会随着行动的成功与否而改变。如果行动失败，人们将向下修正自己的信念，人为地降低由于后悔带来的损失；假如行动成功，人们则会向上修正自己的信念，显示自己做决策的英明。

情景依赖（Framing）。又称框架依赖，是指决策者并不是孤立地知觉和记忆素材，他们是根据过去的经验，以及素材发生的背景，来解释新的信息。

3、认知的目标

期望理论（Prospect Theory）。所谓“期望”即是各种风险结果，期望选择所遵循的是特殊的心理过程和规律。其中主要包括效用价值函数、决策权重函数等。

心理账户（Mental Accounting）。人们根据资金的来源、资金的所在和资金的用途等因素对资金进行归类。人们对待不同的心理账户的风险的态度也是不一样的，投资者通常对于放入保值心理账户的资金具有较强的风险厌恶特点，而对放入升值心理账户的资金具有较弱的风险厌恶特点，有时候甚至主动寻求风险。

（三）、社会心理学

认知的系统偏差（Systematic biases）。指社会特有因素对人的信念与决策产生重要的影响。

信息串流(Information cascades)。指人们在决策时都会参考其他人的选择，而忽略自己已有的信息或可获得的信息。信息串流理论刻画了大量信息在传播与评估中的丢失现象。

羊群效应（Herd behavior）。就是从众行为，当情绪激动之后，由不断激发的情绪引发的行动也不断升级，并进一步刺激人们的情绪。

   调试 运行
文档
 代码  策略  文档
第一部分：数据准备和处理
该部分耗时 4小时，每个日级别因子计算耗时大约1小时，可以通过修改start_date/end_date来调整
我们先从DataAPI中提取以下五方面基础数据，再进行衍生计算得到原始因子。

该部分内容为：

缓存为bh/factor_tk|st|be|cgo.pickle，避免重复计算

'''

from multiprocessing.dummy import Pool as ThreadPool
#from CAL.PyCAL import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from collections import OrderedDict
from sklearn import linear_model
from scipy.stats import pearsonr
from dateutil.parser import parse
from datetime import date, timedelta
import datetime
import itertools
import pickle
import os
import time
from enum import Enum
import sys
import gevent
import math
import seaborn as sns

dir_path = "bh"
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)
    print(u"创建数据文件夹:{0}".format(dir_path))
else:
    print(u"数据文件夹:{0}已存在".format(dir_path))

class TimeDateFormat(Enum):
    """
    时间格式, 枚举类型
    """
    YMDHYPHEN = '%Y-%m-%d'
    YMD = '%Y%m%d'
    YMDHMSHYPHEN = '%Y-%m-%d %H:%M:%S'
    YMDHMS = '%Y:%m:%d %H:%M:%S'
    Y = '%Y'
    HM = '%H:%M'
    
class TimeDateUtil:
    @staticmethod
    def convert_str_to_date(src_str, src_format):
        """
        将timestamp格式日期，转换为date
        """
        try:
            d = datetime.datetime.strptime(src_str, src_format)
        except ValueError:
            d = None
        except TypeError:
            d = None
        return d

    @staticmethod
    def convert_date_to_str(src_date, tgt_format):
        """
        将date格式日期，转换为timestamp
        """
        return src_date.strftime(tgt_format)

    @staticmethod
    def convert_format(src_str, src_format, tgt_format):
        """
        转换日期格式
        :param src_str: 源格式日期
        :param src_format: 源格式
        :param tgt_format: 目标格式
        :return d: 转换格式后的日期
        """
        src_date = TimeDateUtil.convert_str_to_date(src_str, src_format)
        if src_date is None:
            return None
        return TimeDateUtil.convert_date_to_str(src_date, tgt_format)
    
    @staticmethod
    def get_previous_date_str(src_str, src_format, tgt_format, n):
        """
        指定日期前N天日期
        """
        last_days = TimeDateUtil.convert_str_to_date(src_str, src_format) - timedelta(n)
        return last_days.strftime(tgt_format)
    
    @staticmethod
    def reset_time_format(df, src_format, tgt_format, column_name='TRADE_DATE'):
        """
        转换trade_date格式为"yyyy"
        :return:
        """
        new_time_index = list(map(lambda x: TimeDateUtil.convert_format(str(x), src_format, tgt_format), df[column_name].values))
        df[column_name] = pd.Series(new_time_index, index=df.index)
        return df
    
    @staticmethod
    def get_time_delta(src_str, tgt_str, src_format, tgt_format, unit='DAY'):
        """
        两日期间时间差
        """
        last_days = TimeDateUtil.convert_str_to_date(src_str, src_format) - TimeDateUtil.convert_str_to_date(tgt_str, tgt_format)
        if unit == 'DAY':
            return last_days.days
        elif unit == 'SEC':
            return last_days.seconds
        return last_days.days
    
class StyleFactors(Enum):
    BETA = 'BETA'
    MOMENTUM = 'MOMENTUM'
    SIZE = 'SIZE'
    EARNYILD = 'EARNYILD'
    RESVOL = 'RESVOL'
    GROWTH = 'GROWTH'
    BTOP = 'BTOP'
    LEVERAGE = 'LEVERAGE'
    LIQUIDTY = 'LIQUIDTY'
    SIZENL = 'SIZENL'
    
class IndustryFactors(Enum):
    Bank = 'Bank'
    RealEstate = 'RealEstate'
    Health = 'Health'
    Transportation = 'Transportation'
    Mining = 'Mining'
    NonFerMetal = 'NonFerMetal'
    HouseApp = 'HouseApp'
    LeiService = 'LeiService'
    MachiEquip = 'MachiEquip'
    BuildDeco = 'BuildDeco'
    CommeTrade = 'CommeTrade'
    CONMAT = 'CONMAT'
    Auto = 'Auto'
    Textile = 'Textile'
    FoodBever = 'FoodBever'
    Electronics = 'Electronics'
    Computer = 'Computer'
    LightIndus = 'LightIndus'
    Utilities = 'Utilities'
    Telecom = 'Telecom'
    AgriForest = 'AgriForest'
    CHEM = 'CHEM'
    Media = 'Media'
    IronSteel = 'IronSteel'
    NonBankFinan = 'NonBankFinan'
    ELECEQP = 'ELECEQP'
    AERODEF = 'AERODEF'
    Conglomerates = 'Conglomerates'

def load_pickle(pickle_name, ori_obj):
    """
    加载之前处理好的缓存文件
    :param pickle_name: 存储文件名
    :param ori_obj: 变量默认值
    :return valid, ori_obj: 是否之前存在, 读取后的变量值
    """
    if not os.path.exists(pickle_name):
        return False, ori_obj
    else:
        with open(pickle_name, 'rb') as handle:
            ori_obj = pickle.load(handle)
    return True, ori_obj

def save_pickle(pickle_name, ori_obj):
    """
    存储变量为缓存文件
    :param pickle_name: 存储文件名
    :param ori_obj: 变量
    """
    with open(pickle_name, 'wb') as handle:
        pickle.dump(ori_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def filter_security(df, trade_cal, benchmark):
    """
    1) 按投资域筛选因子
    2) 统计覆盖率
    """
    new_df = []
    coverage = []
    bm_ticker = '000300' if benchmark == 'hs300' else '000906' if benchmark == 'zz800' else '000905' if benchmark == 'zz500' else '000985'
    for idx, (_, rows) in enumerate(trade_cal.iterrows()):
        t1 = time.time()
        td = rows['calendarDate']
        td_no_hyphen = rows['calendarDate'].replace('-', '')
        cons_tickers = DataAPI.IdxConsGet(secID=u"",ticker=bm_ticker,isNew=u"",intoDate=td_no_hyphen,field=u"consTickerSymbol",pandas="1")
        cons_tickers_set = cons_tickers['consTickerSymbol'].values.tolist()
        ticker_in_condition = df['ticker'].apply(lambda x: x in cons_tickers_set)
        sub_df = df[(ticker_in_condition) & (df['tradeDate'] == td_no_hyphen)].copy(deep=True)
        new_df.append(sub_df)
        coverage.append([td_no_hyphen, len(sub_df.index) / float(len(cons_tickers_set))])
        if idx % 10 == 0:
            print(u'[FilterSecurity][{0}/{1}] benchmark:{2} done. cost:{3:.2f}'.format(idx, len(trade_cal.index), benchmark, time.time()-t1))
    new_df = pd.concat(new_df, ignore_index=True)
    coverage_df = pd.DataFrame(data=coverage, columns=['tradeDate', 'coverage'])
    return new_df, coverage_df


# 用于因子计算、回测的重要时间戳
begin_date = '20130131'
end_date = '20180731'
# 由于计算某些因子需要前N周的行情数据，所以要比回测起始时间长一些
begin_date_five_years_ago = TimeDateUtil.get_previous_date_str(begin_date, TimeDateFormat.YMD.value, TimeDateFormat.YMD.value, 365*5)
# 月末交易日历，由于后面打算测试因子月度调仓的表现
trade_cal = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin_date, endDate=end_date, field=u"calendarDate,isOpen,isMonthEnd", pandas="1")
trade_cal = trade_cal[(trade_cal['isMonthEnd'] == 1) & (trade_cal['isOpen'] == 1)]
# 全A投资域，ticker->secID的转换关系
sec_df = DataAPI.SecIDGet(partyID=u"",ticker=u"",cnSpell=u"",assetClass=u"E",field=u"ticker,exchangeCD,listDate,secID",pandas="1")
sec_df = sec_df[sec_df['exchangeCD'].apply(lambda x: x in ['XSHE', 'XSHG'])]


'''
基于前景理论的TK因子计算(国泰)

Step 1. 由于效用在时间序列上存在衰减，故提高近期样本的效用权重。取过去60周收益，计算每周衰减系数=ρn (n=1,…,60)

每周衰减系数的累加和为，P=∑60i=1ρi，后续用于归一化

Step 2. 将过去60周收益由负到正进行排序。设共有m个负收益，n个正收益，(r−m,r−m+1,…,r−1;r1,…,rn−1,rn)
Step 3. 结合价值效用函数、概率权重函数、时间衰减系数，计算TK因子，

TK=1P∑i=−m−1ρtiv(ri)[w−(i+m+160)−w−(i+m60)]+1P∑i=1nρtiv(ri)[w+(n−i+160)−w+(n−i60)]
其中，价值效用函数 v(x)={xα,−λ(−x)α,if x≥0if x<0
概率权重函数 ⎧⎩⎨⎪⎪w+(p)=pγ(pγ+(1−p)γ)1/γw−(p)=pδ(pδ+(1−p)δ)1/δ
注：相关参数参考Tversky&Kahneman(1992)的论文，ρ=0.98，α=0.88，λ=2.25，γ=0.61，δ=0.69
'''

class TK:
    @staticmethod
    def time_decay_func(rho=0.98, n=60):
        """
        时间衰减函数
        """
        decay = rho ** np.arange(1, n + 1)
        return decay

    @staticmethod
    def value_utility_func(x, lamb=2.25, alpha=0.88):
        """
        基于前景理论中风险偏好的不对称性，给予盈利状态较低的风险偏好，同时给予亏损状态下更高的风险偏好；
        大家习惯于赚了一部分就卖掉兑换收益，而亏损时则继续持有承受亏损；
        沿用Tversky和Kahneman论文中的经验参数，源于心理学研究，恒定且敏感性较低
        """
        y = x ** alpha if x >= 0 else -lamb * ((-x) ** alpha)
        return y

    @staticmethod
    def probability_weighting_func(p, direction='pos', gamma=0.61, delta=0.69):
        """
        投资者由于赌博和恐惧心理，从而会放大极端事件发生的概率
        沿用Tversky和Kahneman论文中的经验参数
        """
        if direction == 'pos':
            y = (p ** gamma) / ((p ** gamma + (1 - p) ** gamma) ** (1.0 / gamma))
        else:
            y = (p ** delta) / ((p ** delta + (1 - p) ** delta) ** (1.0 / delta))
        return y

    @staticmethod
    def compute_tk_factor(weekly_df, n=60):
        """
        1) 取过去60周收益(5年)，计算每周衰减系数rho^n(n=1,...,60)
        2) 周收益由负到正，进行排序
        3) 价值效用函数、概率估计函数，合成TK因子
        :param weekly_df: 周线数据
        :return 因子数据
        """
        def comp_tk(idxs, weekly_df, decay):
            part_df = weekly_df.iloc[map(int, idxs)].copy(deep=True)
            tot_num = len(part_df.index)
            neg_num = np.sum(part_df['ret'] <= 0)
            part_rank = np.argsort(part_df['ret'])
            part_rank = {idx: rank - neg_num for rank, idx in enumerate(part_rank)}
            part_rank = np.array([rank for _, rank in sorted(part_rank.items())])
            # 处于盈利、亏损不同状态下，本期的盈利情况，对投资者的影响程度也是不同的
            part_prob_weight = np.array([TK.probability_weighting_func((rank + neg_num + 1) / float(tot_num), direction='neg') - 
                                         TK.probability_weighting_func((rank + neg_num) / float(tot_num), direction='neg') if rank < 0 else 
                                         TK.probability_weighting_func((tot_num - rank) / float(tot_num), direction='pos') - 
                                         TK.probability_weighting_func((tot_num - rank - 1) / float(tot_num), direction='pos') for rank in part_rank])
            part_decay = decay[:len(part_df.index)]
            part_value = np.array([TK.value_utility_func(x) for x in part_df['ret'].values])
            tk = np.sum(part_decay * part_prob_weight * part_value) / np.sum(part_decay)
            return tk

        decay = TK.time_decay_func(rho=0.98, n=n)
        weekly_df = weekly_df[weekly_df['tradeDays'] > 0]
        weekly_df['ret'] = weekly_df['closePrice'].pct_change(periods=1)
        weekly_df['idx'] = range(len(weekly_df.index))
        weekly_df['TK'] = weekly_df['idx'].rolling(window=n, min_periods=0, center=False).apply(lambda x: comp_tk(x, weekly_df, decay))
        return weekly_df

# 价值效用函数(value/utility function)
xs = np.arange(-2.0, 2.0, 0.01)
ys = np.array([TK.value_utility_func(x) for x in xs])
fig, ax1 = plt.subplots()

_ = ax1.plot(xs, ys, '--', linewidth=2)
_ = ax1.set_xlabel(u"x-收益", fontsize=18, fontproperties=font)
_ = ax1.set_ylabel(u"v(x)-风险偏好", fontsize=18, fontproperties=font)
_ = ax1.set_title(u"价值效用函数(投资者亏损时愿意继续承受)", fontsize=18, fontproperties=font)
_ = plt.tight_layout()

# 再看一下概率加权函数(probability weighting function)。可见，投资者心中对极端事件过度效应的扭曲程度，边际变化的程度不同
xs = np.arange(0.0, 1.0, 0.01)
ys_bm = np.arange(0.0, 1.0, 0.01)
ys_pos = np.array([TK.probability_weighting_func(x, direction='pos') for x in xs])
ys_neg = np.array([TK.probability_weighting_func(x, direction='neg') for x in xs])
fig, ax1 = plt.subplots()
ax1.plot(xs, ys_bm, '.', linewidth=2, label='linear')
ax1.plot(xs, ys_pos, '--', linewidth=2, label='w_pos')
ax1.plot(xs, ys_neg, '-', linewidth=2, label='w_neg')
ax1.legend(loc='upper left')
ax1.set_xlabel(u"P-赚钱/亏钱的概率", fontsize=18, fontproperties=font)
ax1.set_ylabel(u"w(P)-权重函数", fontsize=18, fontproperties=font)
ax1.set_title(u"概率加权函数(投资者对极端的盈利亏损更敏感)", fontsize=18, fontproperties=font)
_ = plt.tight_layout()


def compute_tk_factor_thread(ticker):
    """
    计算tk因子的单线程函数
    注：这里使用前复权股价计算收益率从而得到衍生TK效用因子
    """
    t1 = time.time()
    #print(ticker)
    weekly_df = DataAPI.MktEquwAdjGet(secID=u"",ticker=ticker,weekEndDate=u"",beginDate=begin_date_five_years_ago,endDate=end_date,isOpen=u"1",
                                      field=u"ticker,weekBeginDate,endDate,tradeDays,closePrice",pandas="1")
    weekly_df = TK.compute_tk_factor(weekly_df, n=60)
    data = []
    for td_hyphen in trade_cal['calendarDate'].values:
        td = TimeDateUtil.convert_format(td_hyphen, TimeDateFormat.YMDHYPHEN.value, TimeDateFormat.YMD.value)
        part_df = weekly_df[weekly_df['endDate'] <= td_hyphen]
        if len(part_df.index) == 0:
            continue
        data.append([ticker, td, part_df['TK'].iloc[-1]])
    tk_factor = pd.DataFrame(data=data, columns=['ticker', 'tradeDate', 'TK'])
    return tk_factor

# 计算TK因子
flag, tk_factor = load_pickle(pickle_name='{0}/factor_tk.pickle'.format(dir_path), ori_obj=pd.DataFrame())
#print(flag)
#print(tk_factor)
if not flag:
    t0 = time.time()
    pool_args = [rows['ticker'] for idx, (_, rows) in enumerate(sec_df.iterrows())]
    # 单线程
    tk_factor = [compute_tk_factor_thread(value) for value in pool_args]
    tk_factor = pd.concat(tk_factor, ignore_index=True)
    save_pickle(pickle_name='{0}/factor_tk.pickle'.format(dir_path), ori_obj=tk_factor)
    print(u'计算完成所有个股TK因子，总耗时:{0:.2f}秒'.format(time.time()-t0))
    
'''

基于显著性理论的ST因子计算(广发)

Step 1. 广发报告中认为，个股收益相对于基准收益的偏离程度，即显著性程度；发现投资者，相较于显著负收益，更关注显著正收益。我们取个股过去20日日线，计算显著性函数

σ(ris,rs¯)=|ris−rs¯||ris|+|rs¯|+θ∗e(ris−rs¯)
Step 2. 将个股过去20天的显著性σ(ris,rs¯)，从大到小降序排序，得到序号kis∈{1,…,20}
Step 3. 计算个股显著性权重，当股票收益率与市场平均收益偏离大时，σ(ris,rs¯)值较大，对应序号kis较小，从而分配到更大的权重。其中πs′=1/20
wis=δkis∑s′δkis′⋅πs′
Step 4. 显著性权重wis与股票收益率ris间的协方差，即为ST因子。ST值越小，说明有限注意的投资者由于过度关注显著信息而低估了股票价格
STi,t=cov(wis,t,ris,t)
注：相关参数，θ=0.9，δ=0.9

'''

class ST:
    @staticmethod
    def saliency_func(r_s, r_bm, theta=0.898):
        """
        当个股收益率大幅超过或大幅落后于市场平均收益水平时，显著性取值会较大；
        当个股收益率与市场水平收益水平相当时，显著性取值会趋于0；
        考虑：投资者对于显著正收益的关注度要高于显著负收益，引入指数项刻画关注度
        """
        sal = abs(r_s - r_bm) / (abs(r_s) + abs(r_bm) + theta) * np.exp(r_s - r_bm)
        return sal
    
    @staticmethod
    def saliency_weight_func(x, sigma=0.9):
        """
        根据上一步得到的saliency值，进行加权归一化
        """
        tmp_x = np.argsort(x)
        tmp_x = {idx: len(x) - rank for rank, idx in enumerate(tmp_x)}
        tmp_x = np.array([rank for _, rank in sorted(tmp_x.items())])
        den = np.sum(sigma ** tmp_x) / len(x)
        w = sigma ** tmp_x / den
        return w
    
    @staticmethod
    def compute_st_factor(daily_df, n=20):
        """
        基于凸显理论的ST因子
        投资者在选择风险资产时，倾向于回报最显著的资产，而不管其实际风险水平。
        """
        def comp_st(idxs, daily_df):
            part_df = daily_df.iloc[map(int, idxs)].copy(deep=True)
            w = ST.saliency_weight_func(part_df['SALIENCY'])
            return np.cov(w, part_df['ret'])[0,1]
        
        daily_df['ret'] = daily_df['closePrice'].pct_change(periods=1)
        daily_df = daily_df.dropna(how='any', axis=0)
        daily_df['SALIENCY'] = ST.saliency_func(daily_df['ret'], daily_df['ret_bm'])
        daily_df['idx'] = range(len(daily_df.index))
        daily_df['ST'] = daily_df['idx'].rolling(window=n, min_periods=0, center=False).apply(lambda x: comp_st(x, daily_df))
        return daily_df

# 我们看一下显著性函数
rs = np.arange(-0.1, 0.1, 0.01)
r_bms = np.arange(-0.1, 0.1, 0.01)
xs = []
ys = []
zs = []
fig = plt.figure(figsize=(18,8))
ax1 = fig.add_subplot(111, projection='3d')
for r, r_bm in itertools.product(rs, r_bms):
    xs.append(r)
    ys.append(r_bm)
    zs.append(ST.saliency_func(r, r_bm))
ax1.scatter(xs, ys, zs, c='b', marker='o')
ax1.set_xlabel(u'r(个股收益率)', fontsize=18, fontproperties=font)
ax1.set_ylabel(u'r_bm(基准收益率)', fontsize=18, fontproperties=font)
ax1.set_zlabel(u'saliency(显著度)', fontsize=18, fontproperties=font)
ax1.set_title(u"显著性函数(个股收益与基准收益偏离越大，则越受关注)", fontsize=18, fontproperties=font)
plt.show()


def compute_st_factor_thread(ticker):
    """
    计算st因子的单线程函数
    """
    t1 = time.time()
    daily_df = DataAPI.MktEqudAdjGet(secID=u"",ticker=ticker,tradeDate=u"",beginDate=begin_date_five_years_ago,endDate=end_date,isOpen="1",
                                      field=u"ticker,tradeDate,closePrice",pandas="1")
    daily_df = daily_df.merge(bm_df, left_on='tradeDate', right_on='tradeDate', how='left', suffixes=['', '_'])
    daily_df = ST.compute_st_factor(daily_df, n=20)
    print (ticker)
    data = []
    for td_hyphen in trade_cal['calendarDate'].values:
        td = TimeDateUtil.convert_format(td_hyphen, TimeDateFormat.YMDHYPHEN.value, TimeDateFormat.YMD.value)
        part_df = daily_df[daily_df['tradeDate'] <= td_hyphen]
        if len(part_df.index) == 0:
            continue
        data.append([ticker, td, part_df['ST'].iloc[-1]])
    st_factor = pd.DataFrame(data=data, columns=['ticker', 'tradeDate', 'ST'])
    return st_factor

# 计算ST因子
flag, st_factor = load_pickle(pickle_name='{0}/factor_st.pickle'.format(dir_path), ori_obj=pd.DataFrame()) 

if not flag:
    t0 = time.time()
    # 000001: 上证综指；000300: 沪深300，由于后面我们做沪深300投资域的测试，所以选用沪深300作为基准更为合理
    bm_df = DataAPI.MktIdxdGet(indexID=u"",ticker=u"000300",tradeDate=u"",beginDate=begin_date_five_years_ago,endDate=end_date,exchangeCD=u"XSHG",
                               field=u"tradeDate,CHGPct",pandas="1").rename(columns={'CHGPct': 'ret_bm'})
    pool_args = [rows['ticker'] for idx, (_, rows) in enumerate(sec_df.iterrows())]
    print (pool_args)
    st_factor = [compute_st_factor_thread(value) for value in pool_args]
    st_factor = pd.concat(st_factor, ignore_index=True)
    print (st_factor)
    save_pickle(pickle_name='{0}/factor_st.pickle'.format(dir_path), ori_obj=st_factor)
    print(u'计算完成所有个股ST因子，总耗时:{0:.2f}秒'.format(time.time()-t0))


'''    
基于噪音交易者的行为偏差(光大)

Step 1. 光大使用雪球帖子数据，构造投资者行为指数(MDI)。指数编制规则如下：筛选过去一年内帖子最多的前10只个股，按流通市值加权，每年年初调整。

Step 2. 横截面回归CAPM模型中的βCi，其中rft表示无风险利率，rmt为市场/基准的收益率

rit−rft=αi+βCi(rmt−rft)+ϵit
Step 3. 横截面回归BAPM(Behavior Asset Pricing Model)中的βBi，其中rft表示无风险利率，rBmt为"投资者行为指数"(也称为MDI)的收益率

rit−rft=αi+βBi(rBmt−rft)+ϵit
Step 4. 行为偏差BE，即为CAPM与BAPM中的beta差，也称为噪音交易者风险的代理变量

BEi=βCi−βBi
Step 5. 构造衍生因子，

行为偏差波动因子(BE_STD)，取行为偏差(BE)在过去6个月的标准差
行为偏差因子(BE_MEAN)，取行为偏差(BE)在过去6个月的平均值

'''

class BE:
    @staticmethod
    def load_xueqiu_post_df(trade_cal_xueqiu):
        """
        读取雪球帖子数数据，时间范围：从2011年末至今
        """
        xueqiu_post_df = []
        for idx, (_, rows) in enumerate(trade_cal_xueqiu.iterrows()):
            td = TimeDateUtil.convert_format(rows['calendarDate'], TimeDateFormat.YMDHYPHEN.value, TimeDateFormat.YMD.value)
            xueqiu_post_df.append(DataAPI.SocialDataXQByDateGet(statisticsDate=td,field=u"ticker,statisticsDate,postNum",pandas="1"))
            if idx % 100 == 0:
                print('[MDI][{0}/{1}] td:{2}, post data loaded done.'.format(idx, len(trade_cal_xueqiu.index), td))
        xueqiu_post_df = pd.concat(xueqiu_post_df, ignore_index=True)
        return xueqiu_post_df
    
    @staticmethod
    def compute_mdi_index(trade_cal_xueqiu, xueqiu_post_df):
        """
        计算MDI指数
        利用雪球等股吧的股票热度数据，筛选一定期限内股民讨论热度最高的股票，来组成投资者行为指数MDI(MomsandDadsIndex)
        """
        xueqiu_post_df['year'] = xueqiu_post_df['statisticsDate'].apply(lambda x: x[:4])
        xueqiu_post_df_ = xueqiu_post_df.groupby(by=['year','ticker'])['postNum'].sum().reset_index().sort_values(by=['year', 'postNum'], ascending=[True, False])
        g = xueqiu_post_df_.groupby(by=['year'])
        trade_cal_xueqiu = trade_cal_xueqiu[trade_cal_xueqiu['isOpen'] == 1].copy(deep=True)
        # 编制指数，合成指数收益
        mdi_df = []
        for year, part_df in g:
            index_begin_date = '{}0101'.format(int(year) + 1)
            index_begin_date_hyphen = TimeDateUtil.convert_format(index_begin_date, TimeDateFormat.YMD.value, TimeDateFormat.YMDHYPHEN.value)
            index_begin_date = trade_cal_xueqiu[trade_cal_xueqiu['calendarDate'] < index_begin_date_hyphen]['calendarDate'].iloc[-1]
            index_end_date = '{}1231'.format(int(year) + 1)
            # 选取每年雪球帖子最多的前10只个股，注：这里也用的是前复权数据
            tickers = part_df.head(10)['ticker'].values.tolist()
            daily_df = DataAPI.MktEqudAdjGet(secID=u"",ticker=",".join(tickers),tradeDate=u"",beginDate=index_begin_date,endDate=index_end_date,isOpen="",
                                             field=u"ticker,tradeDate,closePrice,negMarketValue,isOpen",pandas="1").sort_values(by=['ticker','tradeDate'], ascending=[True, True])
            # 计算个股收益
            daily_df_ = []
            g_ = daily_df.groupby(by=['ticker'])
            for ticker, pp_df in g_:
                pp_df = pp_df.copy(deep=True)
                pp_df['ret'] = pp_df['closePrice'].pct_change(periods=1)
                daily_df_.append(pp_df.dropna())
            daily_df_ = pd.concat(daily_df_, ignore_index=True)
            daily_df_.loc[daily_df_['closePrice'] == 0.0, 'ret'] = 0.0
            # 计算流通市值加权指数收益
            daily_df_ret = pd.crosstab(index=daily_df_['tradeDate'], columns=daily_df_['ticker'], values=daily_df_['ret'], aggfunc='sum').fillna(0.0)
            daily_df_neg = pd.crosstab(index=daily_df_['tradeDate'], columns=daily_df_['ticker'], values=daily_df_['negMarketValue'], aggfunc='sum').fillna(0.0)
            daily_df_neg = daily_df_neg.divide(daily_df_neg.sum(axis=1), axis=0)
            daily_df_ret = (daily_df_ret * daily_df_neg).sum(axis=1)
            mdi_df.append(daily_df_ret.reset_index().rename(columns={0: 'ret_mdi'}))
        mdi_df = pd.concat(mdi_df, ignore_index=True).sort_values(by=['tradeDate'], ascending=[True])
        mdi_df['index_mdi'] = np.cumprod(mdi_df['ret_mdi'] + 1.0) - 1.0
        return mdi_df

    @staticmethod
    def compute_be_factor(daily_df, n=252):
        """
        计算行为偏差因子(BE=CAPM中的beta-BAPM中的beta)
        用行为偏差变量来刻画噪音交易者的交易行为。
        我们选用全A指数作为CAPM的市场行为；选用雪球论坛的股票热度数据，来构建投资者行为指数MDI。
        """
        def comp_be(idxs, daily_df):
            part_df = daily_df.iloc[map(int, idxs)].copy(deep=True)
            reg1 = linear_model.LinearRegression()
            reg2 = linear_model.LinearRegression()
            tot_size = len(part_df['ret'].index)
            mdi_model = reg1.fit(part_df['ret'].values.reshape((tot_size, 1)), part_df['ret_mdi'].values)
            bm_model = reg2.fit(part_df['ret'].values.reshape((tot_size, 1)), part_df['ret_bm'].values)
            be = bm_model.coef_ - mdi_model.coef_
            return be
        
        daily_df['ret'] = daily_df['closePrice'].pct_change(periods=1)
        daily_df = daily_df.dropna(how='any', axis=0)
        daily_df['idx'] = range(len(daily_df.index))
        daily_df['BE'] = daily_df['idx'].rolling(window=n, min_periods=0, center=False).apply(lambda x: comp_be(x, daily_df))
        daily_df['BE_STD'] = daily_df['BE'].rolling(window=120, min_periods=0, center=False).std()
        daily_df['BE_MEAN'] = daily_df['BE'].rolling(window=120, min_periods=0, center=False).mean()
        return daily_df

# 我们看一下通过雪球论坛帖子数，构造的MDI指数
trade_cal_xueqiu = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate='20101130', endDate='20171231', field=u"calendarDate,isOpen,isMonthEnd", pandas="1")
xueqiu_post_df = BE.load_xueqiu_post_df(trade_cal_xueqiu)
mdi_df = BE.compute_mdi_index(trade_cal_xueqiu, xueqiu_post_df)
# 回测起始日前6个月
index_begin_date = (datetime.datetime.strptime(begin_date, "%Y%m%d") - datetime.timedelta(120)).strftime("%Y%m%d")
a_df = DataAPI.MktIdxdGet(indexID=u"",ticker=u"DY0001",tradeDate=u"",beginDate=index_begin_date,endDate=end_date,exchangeCD=u"XSHG",
                   field=u"ticker,tradeDate,closeIndex,CHGPct",pandas="1").rename(columns={'CHGPct': 'ret_bm'})
# 注：index_bm/index_mdi我们这里指的是指数净值/价格曲线
a_df['index_bm'] = np.cumprod(a_df['ret_bm'] + 1.0) - 1.0
index_ret_df = a_df.merge(mdi_df, left_on=['tradeDate'], right_on=['tradeDate'], how='left').set_index('tradeDate')
fig, ax1 = plt.subplots(figsize=(18,8))
index_ret_df['index_mdi'].plot(ax=ax1, label=u'MDI')
index_ret_df['index_bm'].plot(ax=ax1, label=u'A')
ax1.legend()
ax1.set_xlabel(u"tradeDate(日期)", fontsize=18, fontproperties=font)
ax1.set_ylabel(u"netValue(指数净值)", fontsize=18, fontproperties=font)
ax1.set_title(u"MDI指数与上证综指指数收益", fontsize=18, fontproperties=font)
plt.tight_layout()


def compute_be_factor_thread(args):
    """
    计算be因子的单线程函数
    """
    ticker, ticker_daily_df, index_ret_df = args
    print (ticker)
    daily_df = ticker_daily_df.query("ticker==@ticker")
    # 与市场基准指数、MDI基准指数merge，方便后续回归beta
    daily_df = daily_df.merge(index_ret_df.reset_index(), left_on='tradeDate', right_on='tradeDate', how='left', suffixes=['', '_'])
    daily_df = BE.compute_be_factor(daily_df, n=20)
    data = []
    for td_hyphen in trade_cal['calendarDate'].values:
        td = TimeDateUtil.convert_format(td_hyphen, TimeDateFormat.YMDHYPHEN.value, TimeDateFormat.YMD.value)
        part_df = daily_df[daily_df['tradeDate'] <= td_hyphen]
        if len(part_df.index) == 0:
            continue
        data.append([ticker, td, part_df['BE'].iloc[-1], part_df['BE_STD'].iloc[-1], part_df['BE_MEAN'].iloc[-1]])
    be_factor = pd.DataFrame(data=data, columns=['ticker', 'tradeDate', 'BE', 'BE_STD', 'BE_MEAN'])
    return be_factor
# # 计算BE因子
flag, be_factor = load_pickle(pickle_name='{0}/factor_be.pickle'.format(dir_path), ori_obj=pd.DataFrame())

if not flag:
    ticker_list = list(np.unique(sec_df.ticker.values))
    # 行情数据
    ticker_daily_df = DataAPI.MktEqudAdjGet(secID=u"",ticker=ticker_list,tradeDate=u"",beginDate=begin_date_five_years_ago,endDate=end_date,isOpen="1",
                                          field=u"ticker,tradeDate,closePrice",pandas="1")
    be_list = []
    t0 = time.time()
    pool_list = [(ticker, ticker_daily_df, index_ret_df) for ticker in ticker_list]
    for args in pool_list:
        tmp_df = compute_be_factor_thread(args)
        be_list.append(tmp_df)
    be_factor = pd.concat(be_list, axis=0)
    save_pickle(pickle_name='{0}/factor_be.pickle'.format(dir_path), ori_obj=be_factor)
    print(u'计算完成所有个股BE因子，总耗时:{0:.2f}秒'.format(time.time()-t0))
    
'''

基于资本利得的CGO因子(广发)

这个因子，可以参考社区已有的文章，处置效应。我们只是简单拿出来实现，后续做对比分析。

Step 1. 广发效仿Grinblatt(2005)论文，提出以100日为周期，使用日换手率加权的日成交均价(VWAP)来定义参考价格(RP)。其中P即为成交均价(VWAP)，v为换手率

RPt=1k∑n=1100(vt−n∏s=1n−1(1−vt−n+s))Pt−n
Step 2. 计算资本利得突出量CGO(Capital Gain Overhang)，即当前股价相对于参考价格的位置。其中Pclose为个股收盘价

CGOt=Pclose,t−1−RPtRPt

'''

class CGO:
    @staticmethod
    def decay_weight(x):
        """
        换手衰减加权
        认为前一日的持仓，会在今日按照一定衰减比例卖掉。从而得到过去一段时间内的平均成交价格(参考价RP)
        """
        tmp_x = x[::-1]
        x_prod = np.cumprod(1-tmp_x)
        new_x = map(lambda idx: x_prod[idx] * tmp_x[idx] / (1 - tmp_x[idx]), range(len(x)))
        new_x = new_x[::-1]
        return new_x
    
    @staticmethod
    def compute_cgo_factor(daily_df, n=100):
        """
        计算行为资本利得因子(当前价格相对于参考价格的上涨幅度)
        如果当前价格高于参考价格，说明近期涨幅较高，很多投资者获利；如果当前价格低于参考价格，说明近期高换手下跌，很多投资者亏损。
        """
        def comp_rp(idxs, daily_df):
            part_df = daily_df.iloc[map(int, idxs)].copy(deep=True)
            w = CGO.decay_weight(part_df['turnoverRate'].values)
            w = w / np.sum(w)
            return np.sum(w * part_df['vwap'].values)
        
        daily_df = daily_df.dropna(how='any', axis=0)
        daily_df['vwap'] = daily_df['vwap'].fillna(method='ffill')
        daily_df['turnoverRate'] = daily_df['turnoverRate'].fillna(method='ffill')
        daily_df['idx'] = range(len(daily_df.index))
        daily_df['RP'] = daily_df['idx'].rolling(window=n, min_periods=0, center=False).apply(lambda x: comp_rp(x, daily_df)).shift(1)
        daily_df['CGO'] = (daily_df['closePrice'] - daily_df['RP']) / daily_df['RP']
        return daily_df
    
# 我们看一下CGO因子在不同市场环境下的leading作用(由于CGO捕捉了换手率的信息，故比同期均线更容易发现趋势转变)
# 1) 牛市上涨行情
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax11 = ax1.twinx()
# 这里我们使用未复权的股价数据，所以要用前复权因子进行复权
daily_df = DataAPI.MktEqudAdjGet(secID=u"",ticker='600115',tradeDate=u"",beginDate=u"20141031",endDate=u"20150827",isOpen="",
                                  field=u"ticker,tradeDate,closePrice,turnoverRate,turnoverValue,turnoverVol",pandas="1")
daily_df['vwap'] = daily_df['turnoverValue'] / daily_df['turnoverVol']
daily_df = CGO.compute_cgo_factor(daily_df, n=100).set_index('tradeDate')
daily_df['CGO'].plot(ax=ax1, label='CGO', color='b', alpha=0.5, kind='bar', sharex=True)
daily_df['closePrice'].plot(ax=ax11, label='closePrice', color='r', alpha=0.5)
daily_df['RP'].plot(ax=ax11, label='RP', color='g', alpha=0.5)
ax1.set_xticks([i for i in range(0, len(daily_df.index), 40)])
ax1.set_xticklabels([daily_df.index.values[i] for i in range(0, len(daily_df.index), 40)], rotation=0)
ax1.legend(loc='upper left')
ax11.legend(loc='upper right')
ax1.set_xlabel(u"tradeDate(日期)", fontsize=18, fontproperties=font)
ax1.set_title(u"上升通道的某个股，CGO/RP/收盘价", fontsize=18, fontproperties=font)
# 2) 下降通道
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax22 = ax2.twinx()
# 这里我们使用未复权的股价数据，所以要用前复权因子进行复权
daily_df = DataAPI.MktEqudAdjGet(secID=u"",ticker='000002',tradeDate=u"",beginDate=u"20180101",endDate=u"20180806",isOpen="",
                                  field=u"ticker,tradeDate,closePrice,turnoverRate,turnoverValue,turnoverVol",pandas="1")
daily_df['vwap'] = daily_df['turnoverValue'] / daily_df['turnoverVol']
daily_df = CGO.compute_cgo_factor(daily_df, n=100).set_index('tradeDate')
daily_df['CGO'].plot(ax=ax2, label='CGO', color='b', alpha=0.5, kind='bar', sharex=True)
daily_df['closePrice'].plot(ax=ax22, label='closePrice', color='r', alpha=0.5)
daily_df['RP'].plot(ax=ax22, label='RP', color='g', alpha=0.5)
ax2.set_xticks([i for i in range(0, len(daily_df.index), 40)])
ax2.set_xticklabels([daily_df.index.values[i] for i in range(0, len(daily_df.index), 40)], rotation=0)
ax2.legend(loc='upper left')
ax22.legend(loc='upper right')
ax2.set_xlabel(u"tradeDate(日期)", fontsize=18, fontproperties=font)
ax2.set_title(u"下降通道的某个股，CGO/RP/收盘价", fontsize=18, fontproperties=font)
# 3) 震荡行情
fig = plt.figure()
ax3 = fig.add_subplot(111)
ax33 = ax3.twinx()
# 这里我们使用未复权的股价数据，所以要用前复权因子进行复权
daily_df = DataAPI.MktEqudAdjGet(secID=u"",ticker='600028',tradeDate=u"",beginDate=u"20180411",endDate=u"20180803",isOpen="",
                                  field=u"ticker,tradeDate,closePrice,turnoverRate,turnoverValue,turnoverVol",pandas="1")
daily_df['vwap'] = daily_df['turnoverValue'] / daily_df['turnoverVol']
daily_df = CGO.compute_cgo_factor(daily_df, n=100).set_index('tradeDate')
daily_df['CGO'].plot(ax=ax3, label='CGO', color='b', alpha=0.5, kind='bar', sharex=True)
daily_df['closePrice'].plot(ax=ax33, label='closePrice', color='r', alpha=0.5)
daily_df['RP'].plot(ax=ax33, label='RP', color='g', alpha=0.5)
ax3.set_xticks([i for i in range(0, len(daily_df.index), 40)])
ax3.set_xticklabels([daily_df.index.values[i] for i in range(0, len(daily_df.index), 40)], rotation=0)
ax3.legend(loc='upper left')
ax33.legend(loc='upper right')
ax3.set_xlabel(u"tradeDate(日期)", fontsize=18, fontproperties=font)
ax3.set_title(u"震荡行情的某个股，CGO/RP/收盘价", fontsize=18, fontproperties=font)
plt.tight_layout()


def compute_cgo_factor_thread(ticker):
    """
    计算cgo因子的单线程函数
    """
    t1 = time.time()
    print (ticker)
    daily_df = DataAPI.MktEqudAdjGet(secID=u"",ticker=ticker,tradeDate=u"",beginDate=begin_date_five_years_ago,endDate=end_date,isOpen="",
                                      field=u"ticker,tradeDate,closePrice,turnoverRate,turnoverValue,turnoverVol",pandas="1")
    daily_df['vwap'] = daily_df['turnoverValue'] / daily_df['turnoverVol']
    daily_df = CGO.compute_cgo_factor(daily_df, n=100)
    data = []
    for td_hyphen in trade_cal['calendarDate'].values:
        td = TimeDateUtil.convert_format(td_hyphen, TimeDateFormat.YMDHYPHEN.value, TimeDateFormat.YMD.value)
        part_df = daily_df[daily_df['tradeDate'] <= td_hyphen]
        if len(part_df.index) == 0:
            continue
        data.append([ticker, td, part_df['CGO'].iloc[-1]])
    cgo_factor = pd.DataFrame(data=data, columns=['ticker', 'tradeDate', 'CGO'])
    return cgo_factor

# 计算CGO因子
flag, cgo_factor = load_pickle(pickle_name='{0}/factor_cgo.pickle'.format(dir_path), ori_obj=pd.DataFrame())
if not flag:
    t0 = time.time()
    pool_args = [rows['ticker'] for idx, (_, rows) in enumerate(sec_df.iterrows())]
    cgo_factor = [compute_cgo_factor_thread(value) for value in pool_args]
    cgo_factor = pd.concat(cgo_factor, ignore_index=True)
    save_pickle(pickle_name='{0}/factor_cgo.pickle'.format(dir_path), ori_obj=cgo_factor)
    print(u'计算完成所有个股CGO因子，总耗时:{0:.2f}秒'.format(time.time()-t0))
    
'''

小结
本节复现了研报中的行为金融学相关因子，并做了样例结果展示

TK因子。综合考虑时间偏视，大家更注重近期表现；极端收益对投资者的情绪影响更大，复用Tversky和Kahneman的论文参数，从而构建出该因子。
ST显著性因子。综合考虑时间偏视，大家更关注近期表现；投资者会放大涨幅较高的个股在心中影响。因子值较小，代表不受市场关注的股票，反而有可能在未来有较高收益。
BE行为偏差因子。我们认为市场中充满噪音，BE因子刻画的是个股在噪音交易上的暴露程度。如果在暴露程度低，该个股的持有者更为理性，未来一定时间内具有较高回报。
CGO因子，像是综合考虑换手率估算平均成本的动量因子。CGO因子为负说明目前浮亏，但非理性投资者往往愿意继续持有，会产生负向收益。
   调试 运行
文档
 代码  策略  文档
第二部分：单因子分析
该部分耗时 20分钟
先对上述行为金融学因子，进行单因子分析

我们验证这些因子与动量、反转因子的相关性

进行剥离前后的单因子测试对比

该部分内容为：

缓存为bh/bh_single_factor.data，避免重复计算

'''


import seaborn as sns
import pandas as pd

class SingleFactor:
    @staticmethod
    def merge_factors(factor_df_list):
        """
        因子数据合并
        :param factor_df_list: list of df, 每一个df都需要包含tradeDate/ticker这两列
        """
        # 边界值处理
        if len(factor_df_list) == 1:
            return factor_df_list[0]
        elif len(factor_df_list) == 0:
            return factor_df_list
        # 依次合并
        tmp_factor = factor_df_list[0]
        for idx in range(1, len(factor_df_list)):
            tgt_factor = factor_df_list[idx]
            tmp_factor = tmp_factor.merge(tgt_factor, left_on=['tradeDate', 'ticker'], right_on=['tradeDate', 'ticker'], how='outer')
        return tmp_factor

    @staticmethod
    def load_return_data(factor_df):
        """
        读取调仓日之间的个股绝对收益数据
        :param factor_df: 因子数据
        """
        # 对于最后一期因子值，我们也希望知道对应的下期收益
        end_date_next_month = TimeDateUtil.get_previous_date_str(end_date, TimeDateFormat.YMD.value, TimeDateFormat.YMD.value, -60)
        # 1) 拿到交易日历，并进行日期对齐
        trade_cal = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begin_date_five_years_ago, endDate=end_date_next_month, field=u"calendarDate,isOpen,isMonthEnd", pandas="1")
        trade_cal = TimeDateUtil.reset_time_format(trade_cal, TimeDateFormat.YMDHYPHEN.value, TimeDateFormat.YMD.value, column_name='calendarDate')
        trade_cal = trade_cal[trade_cal['isOpen'] == 1]
        trade_cal['nextTradeDate'] = trade_cal['calendarDate'].shift(-1)
        trade_cal = trade_cal[trade_cal['isMonthEnd'] == 1] 
        trade_cal['nextMonthEnd'] = trade_cal['calendarDate'].shift(-1)
        trade_cal_dic = {rows['calendarDate']: (rows['nextTradeDate'], rows['nextMonthEnd']) for _, rows in trade_cal.iterrows()}
        # 2) 读取收盘价，并计算区间收益
        g = factor_df.groupby(by=['tradeDate'])
        data = []
        for idx, (td, part_df) in enumerate(g):
            t1 = time.time()
            tickers = part_df['ticker'].values.tolist()
            daily_df = DataAPI.MktEqudAdjGet(secID=u"",ticker=",".join(tickers),tradeDate=u"",beginDate=trade_cal_dic[td][0],endDate=trade_cal_dic[td][1],isOpen="1",
                                             field=u"ticker,tradeDate,closePrice",pandas="1").sort_values(by=['ticker', 'tradeDate'], ascending=[True, True])
            in_price = daily_df.groupby(by=['ticker'])['closePrice'].first()
            out_price = daily_df.groupby(by=['ticker'])['closePrice'].last()
            ret_df = ((out_price - in_price) / in_price).reset_index()
            ret_df['tradeDate'] = td
            data.append(ret_df)
            if idx % 10 == 0:
                print('[{0}/{1}]td:{2} loaded return data. cost:{3:.2f}'.format(idx, len(g), td, time.time()-t1))
        ret_df = pd.concat(data, ignore_index=True).rename(columns={'closePrice': 'ret'})
        return ret_df
    
    @staticmethod
    def run_backtest(factor_name, backtest_data, signal_df, capital_base=10000000.0, neu=True, direction=1):
        """
        为了得到分组收益图
        :param direction: 1的话代表因子具有正向选股能力；-1代表因子具有反向选股能力
        :param backtest_data: quartz返回的必要依赖数据
        :param signal_df: 因子数据
        """
        t1 = time.time()
        # 读取因子数据，并考虑因子方向
        factor_data = signal_df[['secID', 'tradeDate', factor_name]].copy(deep=True)
        factor_data[factor_name] = factor_data[factor_name] * direction
        factor_data = factor_data.set_index('tradeDate', drop=True)
        month_end_dates = factor_data.index.values
        t2 = time.time()
        # 运行结果
        results = {}
        # 将因子划分为5分位，并进行快速回测
        for quantile_five in range(1, 6):
            t2 = time.time()
            # ---------------策略逻辑部分----------------
            # 因为涉及多次回测，这样设置可以清空上次回测中的缓存数据
            accounts = {
                'security_account': AccountConfig(account_type='security', capital_base=capital_base) # 初始账户资金设为1kw
            }
            sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate, accounts=accounts) # 把回测参数封装到SimulationParameters中
            # 初始化虚拟账户状态
            def initialize(context):
                pass

            # 每个调仓期执行一次
            def handle_data(context): 
                account = context.get_account('security_account')
                pre_date = context.previous_date.strftime("%Y%m%d")
                current_universe = context.get_universe(exclude_halt=True)
                if pre_date not in month_end_dates: 
                    return
                # 拿取调仓日前一个交易日的因子，并按照大小排列取相应分位的股票代码
                q = factor_data.ix[pre_date]
                q = q.set_index('secID', drop=True)
                q = q.ix[current_universe]
                q.dropna(inplace=True)
                # 因子中性化
                if neu:
                    q = standardize(neutralize(winsorize(q), pre_date))
                q_min = q[factor_name].quantile((quantile_five-1)*0.2)
                q_max = q[factor_name].quantile(quantile_five*0.2)
                my_univ = q[(q[factor_name] >= q_min) & (q[factor_name] < q_max)].index.values
                # 将不在目标持仓中的股票卖出
                positions = account.get_positions()
                sell_list = [stk for stk in positions if stk not in my_univ]
                for stk in sell_list:
                    account.order_to(stk,0)
                # 将在目标股票池中的股票，等权买入
                for stk in my_univ:
                    account.order_pct_to(stk, 1.0/len(my_univ))
            # 生成策略对象
            strategy = quartz.TradingStrategy(initialize, handle_data)
            # 开始回测
            bt, perf = quartz.quick_backtest(sim_params, strategy, data=backtest_data)
            # 保存运行结果，1为因子最强组，5为因子最弱组
            results[6 - quantile_five] = {'max_drawdown': perf['max_drawdown'], 'sharpe': perf['sharpe'], 'alpha': perf['alpha'], 'beta': perf['beta'], 
                                          'information_ratio': perf['information_ratio'], 'annualized_return': perf['annualized_return'], 'bt': bt}

        print('fname:{0} quantile lony-only backtest done. cost:{1:.2f}'.format(factor_name, time.time()-t1))
        return results

    @staticmethod
    def collect_bt_result(results, capital_base):
        """
        从quartz.quick_backtest的回测结果中，进行汇总及衍生计算
        """
        quantile_long_only_df = []
        temp = results[1]['bt']
        temp = temp[['tradeDate']]
        for qt in results:
            bt = results[qt]['bt']
            data = bt[[u'tradeDate',u'portfolio_value',u'benchmark_return']].copy(deep=True)
            data['portfolio_return'] = data['portfolio_value']/data['portfolio_value'].shift(1) - 1.0   # 总头寸每日回报率
            data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0]/capital_base - 1.0
            data['excess_return'] = data['portfolio_return'] - data['benchmark_return']                 # 总头寸每日超额回报率
            data['excess'] = data['excess_return'] + 1.0
            data['excess'] = data['excess'].cumprod()                # 总头寸对冲指数后的净值序列
            data['portfolio'] = data['portfolio_return'] + 1.0     
            data['portfolio'] = data['portfolio'].cumprod()          # 总头寸不对冲时的净值序列
            data['benchmark'] = data['benchmark_return'] + 1.0
            data['benchmark'] = data['benchmark'].cumprod()          # benchmark的净值序列
            data['qt'] = qt # 分位数
            results[qt]['hedged_max_drawdown'] = max([1 - v/max(1, max(data['excess'][:i+1])) for i,v in enumerate(data['excess'])])  # 对冲后净值最大回撤
            results[qt]['hedged_volatility'] = np.std(data['excess_return'])*np.sqrt(252)
            results[qt]['hedged_annualized_return'] = (data['excess'].values[-1])**(252.0/len(data['excess'])) - 1.0
            ret = data[['tradeDate', 'portfolio_return']]
            ret.columns = ['tradeDate', 'ret_' + str(qt)]
            temp = pd.merge(temp, ret, on='tradeDate')
            quantile_long_only_df.append(data[['tradeDate', 'portfolio', 'qt']])
        quantile_long_only_df = pd.concat(quantile_long_only_df, ignore_index=True)
        # 1-5多空对冲
        temp['hedge_1_to_5'] = temp['ret_1'] - temp['ret_5']
        temp['portfolio'] = (1 + temp['hedge_1_to_5']).cumprod()   
        # 回测结果转换为DataFrame
        results_pd = pd.DataFrame(results).T.sort_index()
        results_pd = results_pd[[u'alpha', u'beta', u'information_ratio', u'sharpe', u'annualized_return', u'max_drawdown',  
                                 u'hedged_annualized_return', u'hedged_max_drawdown', u'hedged_volatility']]
        cols = [(u'风险指标', u'Alpha'), (u'风险指标', u'Beta'), (u'风险指标', u'信息比率'), (u'风险指标', u'夏普比率'), (u'纯股票多头时', u'年化收益'),
                (u'纯股票多头时', u'最大回撤'), (u'对冲后', u'年化收益'), (u'对冲后', u'最大回撤'), (u'对冲后', u'收益波动率')]
        results_pd.columns = pd.MultiIndex.from_tuples(cols)
        results_pd.index.name = u'五分位组别'
        return results_pd, quantile_long_only_df
    
    @staticmethod
    def neutralize_customer(factor_df, target_factor_name, source_factor_names):
        """
        将原始因子target_factor_name，剥离这些因子source_factor_names，即对这些因子做中性化
        TODO: 剥离动量、反转类因子后，再测试因子表现
        """
        subset = [target_factor_name] + source_factor_names
        factor_df = factor_df.copy(deep=True).dropna(subset=subset, axis=0, how='any')
        y_pre = linear_model.LinearRegression().fit(factor_df[source_factor_names].values, factor_df[target_factor_name].values).predict(factor_df[source_factor_names].values)
        factor_df.loc[:, target_factor_name] = factor_df[target_factor_name].values - y_pre
        return factor_df
    
    @staticmethod
    def neutralize_size_industry(factor_df, target_factors):
        """
        针对给定的因子列表，剥离市值、行业对其的影响
        """
        neutralized_df = []
        g = factor_df.groupby(by=['tradeDate'])
        for idx, (td, part_df) in enumerate(g):
            neu_df = pd.DataFrame()
            for factor in target_factors:
                part_df_cpy = part_df.copy(deep=True).set_index('ticker')
                part_df_cpy = standardize(neutralize(winsorize(part_df_cpy[factor]), target_date=td, 
                             industry_type='SW1', exclude_style_list=['BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY']))
                if len(neu_df.index) == 0:
                    neu_df = part_df_cpy.reset_index().rename(columns={'index': 'ticker', 0: factor})
                else:
                    neu_df = neu_df.merge(part_df_cpy.reset_index().rename(columns={'index': 'ticker', 0: factor}), left_on=['ticker'], right_on=['ticker'], how='outer')
            neu_df['tradeDate'] = td
            neutralized_df.append(neu_df)
            if idx % 10 == 0:
                print('[Neutralize][{0}/{1}] td:{2} neutralized done. {3}'.format(idx, len(g), td, len(part_df.index)))
        merged_factor = pd.concat(neutralized_df, ignore_index=True)
        return merged_factor
    
    @staticmethod
    def plot_grouped_net_value(quantile_long_only_dfs, factors):
        """
        画给定因子的五分组净值走势
        """
        fig = plt.figure(figsize=(20,12))
        col_num = 3 if len(factors) > 1 else 1
        row_num = int(math.ceil(len(factors) / float(col_num)))
        for idx, factor in enumerate(factors):
            ax1 = fig.add_subplot(int("{0}{1}{2}".format(row_num, col_num, idx+1)))
            ax1.grid()
            g = quantile_long_only_dfs[factor].groupby(by='qt')
            for qt, part_df in g:
                ax1.plot(part_df['tradeDate'], part_df[['portfolio']], label=str(qt))
            ax1.legend(loc=0)
            ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
            ax1.set_title(u"因子:{0},五分位选股,净值走势".format(factor), fontproperties=font, fontsize=16)
        return
    
    @staticmethod
    def plot_factor_corr_and_ic(factor_corr_df, factor_ic_corr_df, factor_ic_df, factors):
        """
        画因子间相关性的热力图、因子IC序列的柱状图
        :param factor_corr_df: 因子取值的相关性
        :param factor_ic_corr_df: 因子IC序列上的相关性
        :param factor_ic_df: 因子IC序列
        """
        # 因子值相关性画图展现
        fig = plt.figure(figsize=(20,9))
        ax1 = fig.add_subplot(121)
        sns.heatmap(factor_corr_df, annot=True, fmt='.2f', ax=ax1)
        ax1.set_title(u"因子值相关性", fontproperties=font, fontsize=16)
        # IC相关性画图展现
        ax2 = fig.add_subplot(122)
        sns.heatmap(factor_ic_corr_df, annot=True, fmt='.2f', ax=ax2)
        ax2.set_title(u"因子IC相关性", fontproperties=font, fontsize=16)
        plt.tight_layout()
        # 因子IC序列的柱状图
        fig = plt.figure(figsize=(20,12))
        col_num = 3
        row_num = int(math.ceil(len(factors) / float(col_num)))
        for idx, factor in enumerate(factors):
            ax1 = fig.add_subplot(int("{0}{1}{2}".format(row_num, col_num, idx+1)))
            ax1.grid()
            factor_ic_df[factor].plot(ax=ax1, kind='bar', width=0.5, color='b', label='IC')
            factor_ic_df[factor].rolling(window=12, min_periods=0, center=False).mean().plot(ax=ax1, color='r', label='MEAN(IC,12)')
            ax1.legend(loc=0)
            ax1.set_title(u"因子:{0},月度IC走势".format(factor), fontproperties=font, fontsize=16)
            ax1.set_xticks([i for i in range(0, len(factor_ic_df.index), 40)])
            ax1.set_xticklabels([factor_ic_df.index.values[i] for i in range(0, len(factor_ic_df.index), 40)], rotation=0)
        return
    
    
# 单因子测试：五分位、多空收益
factor_dfs = [tk_factor, st_factor, be_factor, cgo_factor]
factors = OrderedDict([('TK', 1), ('ST', -1), ('BE_STD', 1), ('BE_MEAN', -1), ('CGO', -1)])
benchmark = 'hs300'

# 1) 按投资域过滤
flag, merged_factor = load_pickle(pickle_name='{0}/merged_factor_{1}.pickle'.format(dir_path, benchmark.lower()), ori_obj=pd.DataFrame())
if not flag:
    t1 = time.time()
    merged_factor = SingleFactor.merge_factors(factor_dfs)
    # 2) 将上述原始因子，对行业、市值做中性化，再去分析因子表现
    merged_factor = SingleFactor.neutralize_size_industry(merged_factor, factors.keys())
    merged_factor, coverage_df = filter_security(merged_factor, trade_cal, benchmark)
    save_pickle(pickle_name='{0}/merged_factor_{1}.pickle'.format(dir_path, benchmark), ori_obj=merged_factor)
    print(u'[SingleFactor]按投资域:{0}, 过滤相关个股，总耗时:{1:.2f}秒'.format(benchmark, time.time()-t1))

# 优矿回测中会用到secID，所以做映射
ticker2secID = {rows['ticker']: rows['secID'] for _, rows in sec_df.iterrows()}
merged_factor['secID'] = merged_factor['ticker'].apply(lambda x: ticker2secID[x])

# 3) 单因子测试的主要参数
t2 = time.time()
start = TimeDateUtil.convert_format(begin_date, TimeDateFormat.YMD.value, TimeDateFormat.YMDHYPHEN.value)                       # 回测起始时间
end = TimeDateUtil.convert_format(end_date, TimeDateFormat.YMD.value, TimeDateFormat.YMDHYPHEN.value)                         # 回测结束时间
benchmark = benchmark.upper()              # 策略参考标准
universe = DynamicUniverse(benchmark.upper())# 证券池，支持股票和基金
capital_base = 10000000.0                   # 起始资金
freq = 'd'                                 # 日行情
refresh_rate = Monthly(1)                  # 每月第一个交易日进行调仓
accounts = {
    'security_account': AccountConfig(account_type='security', capital_base=capital_base) # 初始账户资金设为1kw
}
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate, accounts=accounts) # 把回测参数封装到SimulationParameters中
backtest_data = quartz.get_backtest_data(sim_params) # 获取回测行情数据
print('[SingleFactor] load back-test data:{0}, cost:{1:.2f}'.format(benchmark, time.time()-t2))

# 4) 单因子分析
flag, (results_dfs, quantile_long_only_dfs) = load_pickle(pickle_name='{0}/bh_single_factor_{1}.pickle'.format(dir_path, benchmark.lower()), 
                                                          ori_obj=tuple([pd.DataFrame(), pd.DataFrame()]))
if not flag:
    t1 = time.time()
    quantile_long_only_dfs = {}
    results_dfs = []
    for factor, direction in factors.items():
        results_bt = SingleFactor.run_backtest(factor, backtest_data, merged_factor[['secID', 'tradeDate', factor]], capital_base=capital_base, neu=False, direction=direction)
        results_df, quantile_long_only_df = SingleFactor.collect_bt_result(results_bt, capital_base)
        results_df[u'因子名'] = factor
        quantile_long_only_dfs[factor] = quantile_long_only_df
        results_dfs.append(results_df)
    results_dfs = pd.concat(results_dfs, ignore_index=False)
    save_pickle(pickle_name='{0}/bh_single_factor_{1}.pickle'.format(dir_path, benchmark.lower()), ori_obj=tuple([results_dfs, quantile_long_only_dfs]))
    print(u'[SingleFactor]单因子测试，总耗时:{1:.2f}秒'.format(benchmark, time.time()-t1))

# 5) 画图&展示
for col in results_dfs.columns:
    if u'因子名' in col:
        continue
    results_dfs[col] = results_dfs[col].astype(np.float).round(3)
print (results_dfs.to_html())
SingleFactor.plot_grouped_net_value(quantile_long_only_dfs, factors)

'''

我们先看上图中，剥离市值、行业后的原始因子，它们分组选股能力的表现如下

TK/ST/BE_STD/CGO因子在hs300投资域上的分组区分能力最强

BE_MEAN因子分组表现较差

'''

# 与动量、反转等大类因子间相关性分析
# 0) 读取因子成分股，对应的下期收益数据
return_df = SingleFactor.load_return_data(merged_factor)

# 1) 读取因子暴露并合并
other_factors = ['MOMENTUM', 'RESVOL', 'MassIndex', 'ILLIQUIDITY', 'VOL20']
sentiment_factors = factors.keys() + other_factors
g = merged_factor.groupby(by=['tradeDate'])
other_factor_df = []
for idx, (td, part_df) in enumerate(g):
    tickers = part_df['ticker'].values.tolist()
    other_df_1 = DataAPI.RMExposureDayGet(secID=u"",ticker=",".join(tickers),tradeDate=td,beginDate=u"",endDate=u"",field=u"ticker,tradeDate,MOMENTUM,RESVOL",pandas="1")
    other_df_2 = DataAPI.MktStockFactorsOneDayProGet(secID=u"",ticker=",".join(tickers),tradeDate=td,field=u"ticker,tradeDate,MassIndex,ILLIQUIDITY,VOL20",pandas="1")
    other_df_2 = TimeDateUtil.reset_time_format(other_df_2, TimeDateFormat.YMDHYPHEN.value, TimeDateFormat.YMD.value, 'tradeDate')
    other_factor_df.append(other_df_1.merge(other_df_2, left_on=['ticker', 'tradeDate'], right_on=['ticker', 'tradeDate'], how='outer'))
    if idx % 10 == 0:
        print('[OtherFactor][{0}/{1}] td:{2} loaded done. {3}'.format(idx, len(g), td, len(tickers)))
factor_ret_df = pd.concat(other_factor_df, ignore_index=True)
factor_ret_df = factor_ret_df.merge(merged_factor, left_on=['ticker', 'tradeDate'], right_on=['ticker', 'tradeDate'], how='outer')
factor_ret_df = factor_ret_df.merge(return_df, left_on=['ticker', 'tradeDate'], right_on=['ticker', 'tradeDate'], how='outer')

# 2) 因子相关性
g = factor_ret_df.groupby(by=['tradeDate'])
factor_corr_dfs = []
for td, part_df in g:
    tmp_df = part_df[sentiment_factors].corr(method='pearson')
    factor_corr_dfs.append(tmp_df)
factor_corr_df = reduce(lambda x, y: x.add(y, fill_value=0), factor_corr_dfs)
factor_corr_df = factor_corr_df / len(factor_corr_dfs)

# 3) 因子IC相关性
factor_ic_df = []
for td, part_df in g:
    for factor in sentiment_factors:
        part_df_not_nan = part_df[[factor, 'ret']].copy(deep=True).dropna(axis=0, how='any')
        corr, rho = pearsonr(part_df_not_nan[factor].values, part_df_not_nan['ret'].values)
        factor_ic_df.append([td, factor, corr])
factor_ic_df = pd.DataFrame(data=factor_ic_df, columns=['tradeDate', 'factor', 'ic'])
factor_ic_df = pd.crosstab(index=factor_ic_df['tradeDate'], columns=factor_ic_df['factor'], values=factor_ic_df['ic'], aggfunc='sum')
factor_ic_corr_df = factor_ic_df[sentiment_factors].corr(method='pearson')
# 画相关性热力图、原始因子IC走势图
SingleFactor.plot_factor_corr_and_ic(factor_corr_df, factor_ic_corr_df, factor_ic_df, factors.keys())

'''


由于本文计算的ST/TK/BE/CGO因子，主要使用了量价数据，故可能与动量、反转因子有较高相关性

从因子值相关性上看，TK因子与MOMENTUM风格相关性较高；ST因子与CGO/MassIndex相关性较高；BE类因子与动量、反转因子相关性均不高；CGO因子与ST、MassIndex相关性较高

从因子IC相关性上看，本文计算的金融行为学因子与动量、反转、以及相互之间的相关性都较高

印证了我们的想法后，我们对原始因子进行剥离，看下单因子测试结果是否会显著下降


'''

# 1) 将上述因子剥离动量、反转因子
g = factor_ret_df.groupby(by=['tradeDate'])
neutralized_df = []
for idx, (td, part_df) in enumerate(g):
    part_df = part_df.copy(deep=True)
    for factor in factors.keys():
        part_df = SingleFactor.neutralize_customer(part_df, factor, other_factors)
    neutralized_df.append(part_df)
    if idx % 10 == 0:
        print('[Neutralize][{0}/{1}] td:{2} neutralized done. {3}'.format(idx, len(g), td, len(part_df.index)))
neutralized_df = pd.concat(neutralized_df, ignore_index=True)

# 2) 对剥离后的因子，再次进行单因子测试
t1 = time.time()
quantile_long_only_dfs = {}
results_dfs = []
for factor, direction in factors.items():
    results_bt = SingleFactor.run_backtest(factor, backtest_data, neutralized_df[['secID', 'tradeDate', factor]], capital_base=capital_base, neu=False, direction=direction)
    results_df, quantile_long_only_df = SingleFactor.collect_bt_result(results_bt, capital_base)
    results_df[u'因子名'] = factor
    quantile_long_only_dfs[factor] = quantile_long_only_df
    results_dfs.append(results_df)
results_dfs = pd.concat(results_dfs, ignore_index=False)
print(u'[SingleFactor]单因子测试，总耗时:{1:.2f}秒'.format(benchmark, time.time()-t1))

# 3) 展示画分组收益图
for col in results_dfs.columns:
    if u'因子名' in col:
        continue
    results_dfs[col] = results_dfs[col].astype(np.float).round(3)
print(results_dfs.to_html())
SingleFactor.plot_grouped_net_value(quantile_long_only_dfs, factors)


'''


小结
本节对第一步计算的行为金融学因子，进行了单因子分析；验证了与动量、反转因子之间的相关性；剥离后的效果测试

ST/BE_STD/CGO因子分组表现最好，且多头部分收益明显；剩余因子次之

由于我们实现的这些因子，仍与量价有关，在逻辑上与动量、反转因子有一定关系，故验证了因子相关性，的确与Momentum、MassIndex因子相关性较高

原始因子剥离动量、反转因子后，发现TK因子表现会略微下降

   调试 运行
文档
 代码  策略  文档
第三部分：因子合成
该部分耗时 10分钟
由于因子间存在相关性，我们用因子正交化方法去除相关性；再使用ICIR的加权方式，合成因子，看效果是否有提升。


'''
import pandas as pd
import numpy as np
import seaborn
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr
from enum import Enum
from sklearn import linear_model

class CovMethod(Enum):
    SAMPLECOV = 'SAMPLECOV'
    LEDOITWOLF = 'LEDOITWOLF'

class IRIC:
    """
    QEPM: chapter7-Multifactor Alpha Models
    复合因子的计算中，需要注意IC的定义以避免未来函数
    """
    @staticmethod
    def comp_ic(factor_df, return_df, f_names):
        """
        计算因子IC
        :param factor_df: 因子数据，需要包括ticker/tradeDate两列
        :param return_df: 收益数据，需要包括ticker/tradeDate两列
        :param f_names: 需要计算ic的因子名，也需要出现在factor_df列名中
        :return df: 各因子的IC时间序列
        """
        df = factor_df.merge(return_df, left_on=['ticker', 'tradeDate'],
                             right_on=['ticker', 'tradeDate'], how='inner')
        g = df.groupby(by=['tradeDate'])
        data = []
        for td, part_df in g:
            for f_name in f_names:
                part_df = part_df.copy(deep=True).dropna(subset=[f_name, 'ret'], how='any', axis=0)
                rho, pval = spearmanr(part_df[f_name], part_df['ret'])
                data.append([td, f_name, rho])
        df = pd.DataFrame(data=data, columns=['tradeDate', 'factor', 'corr'])
        df = pd.crosstab(index=df['tradeDate'], columns=df['factor'], values=df['corr'], aggfunc='sum')
        return df

    @staticmethod
    def sample_cov(ic_dt):
        """
        样本协方差
        unshrunk covariance
        :return:
        """
        ic_cov_mat = np.mat(np.cov(ic_dt.T.as_matrix()).astype(float))
        return ic_cov_mat

    @staticmethod
    def shrink_cov(ic_dt):
        """
        Ledoit-Wolf shrink covariance
        Ledoit(2004) 单参数形式，可以表示为方差乘以一个单位矩阵
        Ledoit(2003b) CAPM单因子结构化模型估计
        Ledoit(2003a) 平均相关系数形式
        Newey-West(1987) statsmodels.OLS.fit('HAC')
        """
        ic_cov_mat = LedoitWolf().fit(ic_dt.as_matrix()).covariance_
        return ic_cov_mat

    @staticmethod
    def max_ir(ic_df, n=120, method='LEDOITWOLF'):
        """
        v^* = \delta \Sigma^{-1}\vec{\mathrm{IC}}
        \Sigma is factor IC correlation(covariance) matrix
        https://uqer.io/v3/community/share/57eca10d228e5b3663fac5a0
        page.203
        """
        ic_weight_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns).fillna(1.0/len(ic_df.columns))
        for dt in ic_df.index:
            ic_dt = ic_df[ic_df.index < dt].tail(n)
            if len(ic_dt) < n:
                continue
            if method == CovMethod.SAMPLECOV.value:
                ic_cov_mat = IRIC.sample_cov(ic_dt)
            else:
                ic_cov_mat = IRIC.shrink_cov(ic_dt)
            inv_ic_cov_mat = np.linalg.inv(ic_cov_mat)
            weight = inv_ic_cov_mat * np.mat(ic_dt.mean()).reshape(len(inv_ic_cov_mat), 1)
            weight = np.array(weight.reshape(len(weight), ))[0]
            ic_weight_df.ix[dt] = weight / np.sum(weight)
        return ic_weight_df

    @staticmethod
    def ic(ic_df, factor_df, n=120, method='LEDOITWOLF'):
        """
        v^* = \delta \PHI^{-1}\vec{\mathrm{IC}}
        \PHI is factor correlation(covariance) matrix
        page.207
        """
        ic_weight_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns).fillna(1.0/len(ic_df.columns))
        for dt in factor_df.index:
            factor_dt = factor_df[factor_df.index < dt].tail(n)
            ic_dt = factor_df[ic_df.index < dt].tail(n)
            if len(factor_dt) < n or len(ic_dt) < n:
                continue
            if method == CovMethod.SAMPLECOV.value:
                factor_cov_mat = IRIC.sample_cov(factor_dt)
            else:
                factor_cov_mat = IRIC.shrink_cov(factor_dt)
            inv_factor_cov_mat = np.linalg.inv(factor_cov_mat)
            weight = inv_factor_cov_mat * np.mat(ic_dt.mean()).reshape(len(inv_factor_cov_mat), 1)
            weight = np.array(weight.reshape(len(weight), ))[0]
            ic_weight_df.ix[dt] = weight / np.sum(weight)
        return ic_weight_df

    @staticmethod
    def ic_ir_rolling_mean(ic_df, n=120):
        """
        过去一段时间IC均值/过去一段事件IC标准差
        https://uqer.io/v3/community/share/5b1a1260d52680015f105c11
        :param ic_df:
        :param n:
        :return:
        """
        ic_weight_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns).fillna(1.0/len(ic_df.columns))
        for dt in ic_df.index:
            ic_dt = ic_df[ic_df.index < dt].tail(n)
            if len(ic_dt) < n:
                continue
            weight = np.array(ic_dt.mean() / ic_dt.std(ddof=1))
            ic_weight_df.ix[dt] = weight / np.sum(weight)
        return ic_weight_df

    @staticmethod
    def ic_rolling_mean(ic_df, n=120):
        """
        过去一段时间的IC均值除以标准差
        https://uqer.io/v3/community/share/5b1a1260d52680015f105c11
        """
        ic_weight_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns).fillna(1.0/len(ic_df.columns))
        for dt in ic_df.index:
            ic_dt = ic_df[ic_df.index < dt].tail(n)
            if len(ic_dt) < n:
                continue
            weight = np.array(ic_dt.mean())
            ic_weight_df.ix[dt] = weight / np.sum(weight)
        return ic_weight_df

    @staticmethod
    def half_life_ic(ic_df, h=2, n=120, interval=20):
        """
        半衰期IC加权，使用半衰的权重向量来刻画近期IC的影响
        :return:
        """
        ic_weight_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns).fillna(1.0/len(ic_df.columns))
        periods = int(n/interval+1)
        w = np.array([2**((i-periods-1)/float(h)) for i in range(1, periods)])
        w = w / np.sum(w)
        for dt in ic_df.index:
            ic_dt = ic_df[ic_df.index < dt].tail(n)
            if len(ic_dt) < n:
                continue
            ic_dt = ic_dt.rolling(window=interval, min_periods=0, center=False).mean()[interval::interval]
            weight = np.array(ic_dt.mean()) * w
            ic_weight_df.ix[dt] = weight / np.sum(weight)
        return ic_weight_df

    @staticmethod
    def weighting_factors(factor_df, ic_weight_df, factor_names):
        """
        使用因子IC权重，对因子进行加权合成
        :param factor_df:
        :param ic_weight_df:
        :param factor_names:
        :return:
        """
        factor_weight_names = ['{0}_'.format(f_name) for f_name in factor_names]
        factor_weight_renames = {'{0}_'.format(f_name): f_name for f_name in factor_names}
        tmp_factor_df = factor_df.merge(ic_weight_df.reset_index(), left_on=['tradeDate'], right_on=['tradeDate'], how='left', suffixes=['', '_']).set_index(['tradeDate', 'ticker'])
        weighted_factor_df = tmp_factor_df[factor_names].multiply(tmp_factor_df[factor_weight_names].copy(deep=True).rename(columns=factor_weight_renames), axis=1, fill_value=0.0)
        weighted_factor_df['factor'] = np.sum(weighted_factor_df, axis=1)
        weighted_factor_df = weighted_factor_df[['factor']].reset_index()
        return weighted_factor_df

# 1) 修改因子方向&归一化
merged_factor_direction = merged_factor.copy(deep=True)
for factor, direction in factors.items():
    merged_factor_direction[factor] = merged_factor_direction[factor] * direction
g = merged_factor_direction.groupby(by=['tradeDate'])
for idx, (td, part_df) in enumerate(g):
    for factor, direction in factors.items():
        not_nan_idx = part_df[~part_df[factor].isnull()]
        merged_factor_direction.loc[not_nan_idx.index, factor] = normalize(not_nan_idx[factor].values, norm='l2', axis=1, copy=True, return_norm=False).reshape(
            (len(not_nan_idx.index),1))
    if idx % 10 == 0:
        print('[Normalize][{0}/{1}] td:{2} normalized done.'.format(idx, len(g), td))

# 2) 不考虑因子间相关性，直接使用icir方法合成因子
combined_factors = ['TK', 'ST', 'BE_STD', 'CGO']
t1 = time.time()
ic_df = IRIC.comp_ic(merged_factor_direction, return_df, combined_factors)
ic_weight_df = IRIC.ic_ir_rolling_mean(ic_df, n=12)
synthesis_df = IRIC.weighting_factors(merged_factor_direction, ic_weight_df, combined_factors)
synthesis_df = synthesis_df[(synthesis_df['tradeDate'] > begin_date)]
synthesis_df = synthesis_df.merge(return_df, left_on=['ticker', 'tradeDate'], right_on=['ticker', 'tradeDate'], how='left')
synthesis_df = synthesis_df.merge(merged_factor_direction[['ticker', 'tradeDate', 'secID']], left_on=['ticker', 'tradeDate'], right_on=['ticker', 'tradeDate'], how='left')

# 3) 重新进行因子测试
results_bt = SingleFactor.run_backtest('factor', backtest_data, synthesis_df, capital_base=capital_base, neu=False, direction=1)
results_df, quantile_long_only_df = SingleFactor.collect_bt_result(results_bt, capital_base)
print(u'[MultiFactor]多因子合成，总耗时:{1:.2f}秒'.format(benchmark, time.time()-t1))
# 展示画分组收益图
for col in results_df.columns:
    if u'因子名' in col:
        continue
    results_df[col] = results_df[col].astype(np.float).round(3)
print(results_df.to_html())
SingleFactor.plot_grouped_net_value({'Combined':quantile_long_only_df}, ['Combined'])

'''
我们发现不考虑因子间相关性的情况下，经ICIR合成后，效果有一定提升。接下来我们采用因子正交化方法，再进行测试

我们仅选用了四个表现较好的因子，效用因子TK，显著性因子ST，异常行为因子BE_STD，资本利得因子CGO

发现经ICIR加权后，第一组的累计收益有提高

但这些因子之间仍存在相关性，我们接下来验证因子正交化+ICIR加权能否进一步提升

'''

from enum import Enum
from sklearn import linear_model
import numpy as np
import pandas as pd

class OrthogonalMethod(Enum):
    GRAMSCHMIDT = 'GRAMSCHMIDT'
    SYMMETRIC = 'SYMMETRIC'

class Orthogonal:
    """
    Klein R F, Chow V K. Orthogonalized factors and systematic risk decomposition[J]. The Quarterly Review of Economics and Finance, 2013, 53(2): 175-187.
    """
    @staticmethod
    def gram_schmidt(factor_df, f_names):
        """
        通用的gram_schmidt正交化方法
        """
        factor_num = len(f_names)
        ori_factors = factor_df[f_names].values
        orthogonal_factors = np.zeros_like(ori_factors)
        orthogonal_factors[:, 0] = ori_factors[:, 0]
        for i in range(1, factor_num):
            for j in range(i):
                neued = np.dot(np.dot(orthogonal_factors[:, j].T, ori_factors[:, i]), orthogonal_factors[:, j]) / \
                     np.dot(orthogonal_factors[:, j], orthogonal_factors[:, j].T)
                orth_factor = ori_factors[:, i] - neued
            orthogonal_factors[:, i] = orth_factor
        orthogonal_df = pd.DataFrame(data=orthogonal_factors, columns=f_names, index=factor_df.index)
        orthogonal_df['ticker'] = factor_df['ticker']
        orthogonal_df['tradeDate'] = factor_df['tradeDate']
        orthogonal_df['secID'] = factor_df['secID']
        return orthogonal_df
    
    @staticmethod
    def symmetric_orthogonal(factor_df, f_names):
        """
        天风证券-金工专题报告：因子正交全攻略，理论、框架与实践-20171030
        天风证券-金工专题报告：基于自适应风险控制的指数增强策略-180705
        1) overlap_mat = (N-1)cov
        :return:
        """
        asset_num = len(factor_df.index)
        factor_num = len(f_names)
        ori_factors = factor_df[f_names].values
        fmean = np.zeros_like(ori_factors)
        for i in range(factor_num):
            fmean[:, i] = ori_factors[:, i] - np.mean(ori_factors[:, i])
        fmean = np.mat(fmean)
        overlap_mat = np.mat((asset_num - 1) * np.cov(fmean.T))  # 获得重叠矩阵
        u, v = np.linalg.eig(overlap_mat)  # u是特征值，v是特征向量
        degree_mat = np.dot(np.dot(v, np.linalg.inv(np.diag(u ** 0.5))), v.T) # 获得度矩阵
        symmetric_mat = np.dot(degree_mat * ((asset_num - 1) ** 0.5), np.diag(ori_factors.var(axis=0) ** 0.5))
        orthogonal_factors = np.dot(ori_factors, symmetric_mat)
        orthogonal_df = pd.DataFrame(data=orthogonal_factors, columns=f_names, index=factor_df.index)
        orthogonal_df['ticker'] = factor_df['ticker']
        orthogonal_df['tradeDate'] = factor_df['tradeDate']
        orthogonal_df['secID'] = factor_df['secID']
        return orthogonal_df

    @staticmethod
    def othogonal_factor(factor_df, f_names, method='GRAMSCHMIDT'):
        """
        1) zscore
        2) 协方差矩阵->重叠矩阵
        :param factor_df: 多个因子合并后的df
        :param f_names: 因子列表
        :return:
        """
        factor_df = factor_df[['tradeDate', 'ticker', 'secID'] + f_names].copy(deep=True)
        g = factor_df.groupby(by=['tradeDate'])
        orthogonal_dfs = []
        for td, part_df in g:
            part_df = part_df.dropna(axis=0, how='any')
            if method == OrthogonalMethod.GRAMSCHMIDT.value:
                orthogonal_df = Orthogonal.gram_schmidt(part_df, f_names)
            elif method == OrthogonalMethod.SYMMETRIC.value:
                orthogonal_df = Orthogonal.symmetric_orthogonal(part_df, f_names)
            orthogonal_dfs.append(orthogonal_df)
        orthogonal_dfs = pd.concat(orthogonal_dfs, ignore_index=True)
        return orthogonal_dfs

# 1) 考虑因子间相关性，去掉后再进行合成
orthgonal_df = Orthogonal.othogonal_factor(merged_factor_direction, combined_factors, OrthogonalMethod.SYMMETRIC.value)
t1 = time.time()
ic_df = IRIC.comp_ic(orthgonal_df, return_df, combined_factors)
ic_weight_df = IRIC.ic_ir_rolling_mean(ic_df, n=12)
synthesis_df = IRIC.weighting_factors(orthgonal_df, ic_weight_df, combined_factors)
synthesis_df = synthesis_df[(synthesis_df['tradeDate'] > begin_date)]
synthesis_df = synthesis_df.merge(return_df, left_on=['ticker', 'tradeDate'], right_on=['ticker', 'tradeDate'], how='left')
synthesis_df = synthesis_df.merge(orthgonal_df[['ticker', 'tradeDate', 'secID']], left_on=['ticker', 'tradeDate'], right_on=['ticker', 'tradeDate'], how='left')

# 2) 重新进行因子测试
results_bt = SingleFactor.run_backtest('factor', backtest_data, synthesis_df, capital_base=capital_base, neu=False, direction=1)
results_df, quantile_long_only_df = SingleFactor.collect_bt_result(results_bt, capital_base)
print(u'[MultiFactor]多因子合成，总耗时:{1:.2f}秒'.format(benchmark, time.time()-t1))
# 展示画分组收益图
for col in results_df.columns:
    if u'因子名' in col:
        continue
    results_df[col] = results_df[col].astype(np.float).round(3)
print(results_df.to_html())
SingleFactor.plot_grouped_net_value({'Orthogonal-Combined':quantile_long_only_df}, ['Orthogonal-Combined'])


'''

小结
本节实现了因子正交化去除因子间相关性，ICIR加权的因子合成方法

由于我们实现的行为金融学因子，与量价有关，之间有一定相关性。如果直接通过ICIR进行合成，因子分组区分度有改善。

经过因子正交去除相关性，再通过ICIR合成，可进一步提高因子分组区分度，优质组的收益也有提升。

   调试 运行
文档
 代码  策略  文档
第四部分： 构建主动增强策略
该部分耗时 10分钟
如何将上述因子，应用到实际投资场景中？

我们通过在hs300投资域，进行组合构建，按月调仓，实现long only的主动增强策略

我们以经正交化&ICIR合成后的因子为例，验证这些行为金融学因子中仍包含一定的alpha信息


'''

import numpy as np
import pandas as pd
import quartz_extensions.Optimizer.optimize as opt

def optimize_rhac(signal, date, active_risk=0.02, need_neutralize=True, benchmark='ZZ500', exclude_style_list=[]):
    """
    优矿中的指数增强优化(控制行业、风格偏离度，给定能承受的追踪误差)
    参考知识库中的-组合优化器文档
    """
    if need_neutralize:
        signal = standardize(neutralize(winsorize(signal), date, exclude_style_list=exclude_style_list)).dropna()
    else:
        signal = standardize(winsorize(signal)).dropna()
    # 创建优化器对象
    pspec = opt.UqerOptimizer(signal, date, benchmark_str=benchmark)
    # 添加约束
    # 个股上下限约束
    pspec.add_constraint(default_min_weight=0.0, default_max_weight=0.05)
    # 主动风险
    pspec.add_constraint(tracking_error=active_risk)
    # 行业中性
    pspec.add_constraint(is_industry_neutralize=True, active_indu_lower=0, active_indu_upper=0)
    # 风格约束
    pspec.add_constraint(style_value=0.03)
    pspec.solve()
    weights = pspec.assets[pspec.assets.optimal_weights > 0.00001]
    return weights, pspec.optimal

from CAL.PyCAL import *
import numpy as np
from pandas import DataFrame

benchmark_ = ('HS300', '000300')
start = TimeDateUtil.convert_format(begin_date, TimeDateFormat.YMD.value, TimeDateFormat.YMDHYPHEN.value) # 回测起始时间
end = TimeDateUtil.convert_format(end_date, TimeDateFormat.YMD.value, TimeDateFormat.YMDHYPHEN.value) # 回测结束时间
benchmark = '{0}.ZICN'.format(benchmark_[1])                        # 策略参考标准
universe = DynamicUniverse(benchmark_[0])  # 证券池，支持股票和基金
capital_base = 10000000                    # 起始资金
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = Monthly(1)                  # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟
commission = Commission(buycost=0.0015, sellcost=0.0015) 

# 构建日期列表
factor = synthesis_df[['ticker', 'tradeDate', 'factor', 'secID']].copy(deep=True).rename(columns={'tradeDate': 'date', 'factor': 'value'})
data = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=start.replace('-', ''),endDate=end.replace('-', ''),field=['calendarDate','isMonthEnd'],pandas="1")
data = data[data['isMonthEnd'] == 1]
date_list = data['calendarDate'].values.tolist()

cal = Calendar('China.SSE')
period = Period('-1B')

accounts = {
 		'stock_account': AccountConfig(account_type='security', capital_base=capital_base)
}

def initialize(context):                   # 初始化虚拟账户状态
    pass

def handle_data(context):                  # 每个交易日的买入卖出指令
    today = context.current_date
    today = Date.fromDateTime(context.current_date)  # 向前移动一个工作日
    yesterday = cal.advanceDate(today, period)
    yesterday = yesterday.toDateTime()
    account = context.get_account('stock_account')
    
    if yesterday.strftime('%Y-%m-%d') in date_list:
        factor_df = factor[factor['date'] == yesterday.strftime('%Y%m%d')].copy(deep=True)
        factor_df = factor_df.set_index('secID', drop=False)
        if len(factor_df.index) == 0:
            return
        wts, optimal = optimize_rhac(factor_df['value'], yesterday.strftime('%Y%m%d'), active_risk=0.03, need_neutralize=False, benchmark=benchmark_[0],
                                     exclude_style_list=['BETA', 'BTOP', 'EARNYILD', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'MOMENTUM', 'RESVOL'])
        if not optimal:
            print('td:{0}, rhac not solved, holdings would not change.'.format(yesterday))
            return
        wts = wts['optimal_weights'].to_dict()
        # 找载体，将ticker转化为secID
        factor_df['wts'] = np.nan
        factor_df['wts'][wts.keys()] = wts.values()
        factor_df = factor_df[~np.isnan(factor_df['wts'])]
        factor_df.set_index('secID', inplace=True)
        # 先卖出
        sell_list = account.get_positions()
        for stk in sell_list:
            account.order_to(stk, 0)
        # 再买入
        buy_list = factor_df.index
        total_money = account.portfolio_value
        for stk in buy_list:
            if np.isnan(context.current_price(stk)) or context.current_price(stk) == 0:  # 停牌或是还没有上市等原因不能交易
                continue
            account.order(stk, int(total_money * factor_df.loc[stk]['wts'] / context.current_price(stk) /100)*100)
    else:
        return
    

from quartz_extensions.SignalAnalysis.tears import analyse_construction

def get_neued_signal(x):
    signal = x.copy(deep=True).dropna()
    date = x['date'].iloc[0]
    if signal.shape[0] > 0:
        series = signal[['ticker', 'value']].set_index('ticker')['value'].dropna()
        neued = standardize(neutralize(winsorize(series), target_date=date, 
                           exclude_style_list=['BETA', 'BTOP', 'EARNYILD', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'MOMENTUM', 'RESVOL']))
        return neued

# 合成后的因子，剥离所有风格因子影响
factor = synthesis_df[['ticker', 'tradeDate', 'factor', 'secID']].copy(deep=True).rename(columns={'tradeDate': 'date', 'factor': 'value'})

# 主动增强组合，参数设置
factor_upper_boundary = 0.03 # 风格因子偏离度上限
factor_boundary = pd.Series(index=['BETA', 'BTOP', 'COUNTRY', 'EARNYILD', 'GROWTH', 'LEVERAGE', 'LIQUIDTY', 'MOMENTUM', 'RESVOL', 'SIZE', 'SIZENL'])
factor_boundary.loc[:] = factor_upper_boundary
factor_boundary.loc['SIZE'] = 0.03 # 希望市值因子的偏离度小一些，因为我们已知17年以来市值反转
sector_boundary = 0.03 # 行业因子偏离度上限

# 主动增强组合构建与归因分析
all_data_rhac = analyse_construction(factor, 'limit_active_risk', min(factor['date']), max(factor['date']), universe='HS300', benchmark='HS300', frequency='month', 
                                     sector_exposure_lower_boundary=-sector_boundary, sector_exposure_upper_boundary=sector_boundary, 
                                     factor_exposure_lower_boundary=-factor_boundary, factor_exposure_upper_boundary=factor_boundary, 
                                     target_risk=0.03, asset_upper_boundary=0.05, init_cash=1e7)


'''

本节调用优矿提供的优化器，构建指数增强策略（控制行业、风格暴露，跟踪误差）

发现通过正交化&ICIR合成后的因子，在13~17年可以贡献稳定的超额收益

从归因结果看来组合有一定小市值暴露，从而解释17年以来组合为何表现不好

   调试 运行
文档
 代码  策略  文档
总结
1)本文通过复现相关金工研报，实现行为金融学因子，发现投资者的行为偏好，也是alpha的贡献来源，值得挖掘

2)将来可以结合其他信息源，不单单是从行情数据出发构建因子

'''

