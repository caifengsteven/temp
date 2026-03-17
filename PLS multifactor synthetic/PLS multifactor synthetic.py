# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:53:39 2020

@author: Asus
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import date, timedelta
import datetime
import pickle
import os
import time
from enum import Enum

dir_path = "pls"
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


class TradeDateProcess():
    """
    交易日历相关处理
    """
    def __init__(self):
        self.trade_date = None

    def update_trade_date(self):
        """
        更新获取交易日历
        """
        trade_cal = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=u"",endDate=u"",field=u"isOpen,calendarDate",pandas="1")
        trade_cal = trade_cal[trade_cal['isOpen']==1]
        self.trade_date = trade_cal['calendarDate'].map(lambda x:parse(x)).tolist()
    
    def _get_date_list(self,start_date,end_date,freq=None,point=None):
        """
        可以得到指定频率交易日期列表
        :param start_date: 起始日期
        :param end_date: 终止日期
        :param freq: 频率, 日、周、月
        :param point: 取哪一天作为调仓日
        """
        calendar_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=start_date,endDate=end_date,
                                            field=u"",pandas="1")[['isOpen','calendarDate']]
        calendar_date = calendar_date[calendar_date['isOpen']==1]
        calendar_date['calendarDate'] = pd.to_datetime(calendar_date['calendarDate'])
        trade_date = set(calendar_date['calendarDate'])
        if freq == 'monthly':
            calendar_date['year'] = calendar_date['calendarDate'].map(lambda x:x.year) 
            calendar_date['month'] = calendar_date['calendarDate'].map(lambda x:x.month)
            if point > 0:
                trade_date = set(calendar_date.groupby(['year','month']).head(point)['calendarDate']) - set(calendar_date.groupby(['year','month']).head(point-1)['calendarDate'])
            elif point < 0:
                point = - point
                trade_date = set(calendar_date.groupby(['year','month']).tail(point)['calendarDate']) - set(calendar_date.groupby(['year','month']).tail(point-1)['calendarDate'])
        elif freq == 'weekly':
            calendar_date['year_week'] = calendar_date['calendarDate'].map(lambda x:datetime.datetime.isocalendar(x)[0:2])
            if point > 0:
                trade_date = set(calendar_date.groupby(['year_week']).head(point)['calendarDate']) - set(calendar_date.groupby(['year_week']).head(point-1)['calendarDate']) 
            elif point < 0:
                point = - point
                trade_date = set(calendar_date.groupby(['year_week']).tail(point)['calendarDate']) - set(calendar_date.groupby(['year_week']).tail(point-1)['calendarDate'])
        elif freq == 'daily':
            trade_date = set(calendar_date['calendarDate'])
            return np.sort(list(trade_date))
        elif freq == 'quarterly':
            calendar_date['year'] = calendar_date['calendarDate'].map(lambda x:x.year) 
            calendar_date['month'] = calendar_date['calendarDate'].map(lambda x:x.month)
            calendar_date = calendar_date[calendar_date['month'].isin([4,8,10])]
            if point > 0:
                trade_date = set(calendar_date.groupby(['year','month']).head(point)['calendarDate']) - set(calendar_date.groupby(['year','month']).head(point-1)['calendarDate'])
            elif point < 0:
                point = - point
                trade_date = set(calendar_date.groupby(['year','month']).tail(point)['calendarDate']) - set(calendar_date.groupby(['year','month']).tail(point-1)['calendarDate'])
        return np.sort(list(trade_date))
    
    @staticmethod
    def step_trade_date(base_date, step, trading_days):
        """
        交易日向前向后漂移
        """
        base_date_index = trading_days.index(base_date)
        step_date_index = base_date_index + step
        return trading_days[step_date_index]


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

def convert_format(src_str, src_format, tgt_format):
    """
    转换日期格式
    :param src_str: 源格式日期
    :param src_format: 源格式
    :param tgt_format: 目标格式
    :return d: 转换格式后的日期
    """
    def convert_str_to_date(src_str, src_format):
        try:
            d = datetime.datetime.strptime(src_str, src_format)
        except ValueError:
            d = None
        except TypeError:
            d = None
        return d
    
    def convert_date_to_str(src_date, tgt_format):
        return src_date.strftime(tgt_format)
    
    src_date = convert_str_to_date(src_str, src_format)
    if src_date is None:
        return None
    return convert_date_to_str(src_date, tgt_format)

def load_fiscal_data(ticker='603288', begin_date='2007-01-01', end_date='2018-06-26'):
    """
    读取财务数据, 并计算相关因子
    1)资产负债表, FdmtBSGet
    TCA: 流动资产合计, cashCEquiv: 货币资金, NotesReceiv: 应收票据, AR: 应收账款, TCL: 流动负债合计, STBorr: 短期借款, CBBorr: 向银行借款
    fixedAssets: 固定资产, inventories: 存货, TAssets: 总资产, TLiab: 总负债
    2)利润表, FdmtISGet
    tRevenue: 营业总收入, incomeTax: 所得税费用, NIncome: 净利润, TProfit: 税前利润
    3)现金流量表, FdmtCFGet
    COutfFrInvestA: 投资活动现金流出小计, CPaidInvest: 投资支付现金
    4)现金流量表附表, FdmtCfsGet
    FAOGPBDepr: 固定资产折旧, intanAssetsAmor: 无形资产摊销
    :param ticker: 个股名
    :param begin_date: 读取财务数据的起始日期
    :param end_date: 读取财务数据的终止日期
    """
    bs = DataAPI.FdmtBSGet(ticker=ticker,secID=u"",reportType=u"",endDate=end_date,beginDate=u"",
                                    publishDateEnd=u"",publishDateBegin=u"",\
                                    endDateRep="",beginDateRep="",beginYear="",endYear="",fiscalPeriod="",\
                                    field=u"ticker,endDate,publishDate,fiscalPeriod,TCA,cashCEquiv,TCL,STBorr,CBBorr,fixedAssets,inventories,TAssets,TLiab,investRealEstate,intanAssets,depos",pandas="1")
    bs = bs.sort_values(by=['endDate', 'publishDate', 'fiscalPeriod'], ascending=[True, True, False]).drop_duplicates(subset=['endDate'], keep='first')
    income = DataAPI.FdmtISGet(ticker=ticker,secID=u"",reportType=u"",endDate=end_date,beginDate=u"", \
                                    publishDateEnd=u"",publishDateBegin=u"",\
                                    endDateRep="",beginDateRep="",beginYear="",endYear="",fiscalPeriod="", \
                                    field=u"ticker,secShortName,endDate,publishDate,fiscalPeriod,tRevenue,incomeTax,NIncome,TProfit",pandas="1")
    income = income.sort_values(by=['endDate', 'publishDate', 'fiscalPeriod'], ascending=[True, True, False]).drop_duplicates(subset=['endDate'], keep='first')
    cf = DataAPI.FdmtCFGet(ticker=ticker,secID=u"",reportType=u"",endDate=end_date,beginDate=u"", \
                                    publishDateEnd=u"",publishDateBegin=u"",\
                                    endDateRep="",beginDateRep="",beginYear="",endYear="",fiscalPeriod="", \
                                    field=u"ticker,endDate,publishDate,fiscalPeriod,COutfFrInvestA,CPaidInvest,NCFFrInvestA",pandas="1")
    cf = cf.sort_values(by=['endDate', 'publishDate', 'fiscalPeriod'], ascending=[True, True, False]).drop_duplicates(subset=['endDate'], keep='first')
    cfs = DataAPI.FdmtCfsGet(ticker=ticker,secID=u"",reportType=u"",endDate=end_date,beginDate=u"", \
                            beginYear="",endYear="",fiscalPeriod="", \
                            field=u"ticker,endDate,publishDate,fiscalPeriod,FAOGPBDepr,intanAssetsAmor",pandas="1")
    cfs = cfs.sort_values(by=['endDate', 'publishDate', 'fiscalPeriod'], ascending=[True, True, False]).drop_duplicates(subset=['endDate'], keep='first')
    cfs['fiscalPeriod'] = cfs['fiscalPeriod'].astype(int)
    cfs['publishDate'] = cfs['publishDate'].apply(lambda x: x[:10])
    bs = bs.merge(income, left_on=['ticker', 'endDate', 'publishDate', 'fiscalPeriod'], right_on=['ticker', 'endDate', 'publishDate', 'fiscalPeriod'])
    bs = bs.merge(cf, left_on=['ticker', 'endDate', 'publishDate', 'fiscalPeriod'], right_on=['ticker', 'endDate', 'publishDate', 'fiscalPeriod'])
    bs['fiscalPeriod'] = bs['fiscalPeriod'].astype(int)
    bs = bs.merge(cfs, left_on=['ticker', 'endDate', 'publishDate', 'fiscalPeriod'], right_on=['ticker', 'endDate', 'publishDate', 'fiscalPeriod'], how='inner')
    # 这样处理后就只剩下半年报、年报的数据了，因为季报不披露财务附表
    bs[['investRealEstate','fixedAssets','intanAssets','CBBorr','depos','NCFFrInvestA']] = bs[['investRealEstate','fixedAssets','intanAssets','CBBorr','depos','NCFFrInvestA']].fillna(0.0)
    bs['TCA'] = bs['TCA'].fillna(bs['TAssets']-bs['investRealEstate']-bs['fixedAssets']-bs['intanAssets'])
    bs['TCL'] = bs['TCL'].fillna(bs['TLiab'])
    bs['STBorr'] = bs['STBorr'].fillna(bs['CBBorr'] + bs['depos'])
    df = bs.fillna(0.0)
    # print(df)
    df['CAPX'] = -(df['COutfFrInvestA'] - df['CPaidInvest']).shift(1)
    df['CE'] = df['CAPX'] / df['tRevenue']
    df['CI'] = df['CE'] / df['CE'].rolling(window=3, min_periods=0, center=False).mean().shift(1) - 1.0
    df['I/K'] = df['CAPX'] / df['fixedAssets']
    df['I/A'] = (df['inventories'] + df['fixedAssets']).diff(periods=1) / df['TAssets']
    df['IG'] = df['NCFFrInvestA'].pct_change(periods=2)
    df['ACC'] = (((df['TCA']-df['cashCEquiv'])-(df['TCL']-df['STBorr']-df['CBBorr'])).diff(periods=1)-(df['FAOGPBDepr']+df['intanAssetsAmor'])) / df['TAssets'].shift(1)
    df['NOA'] = ((df['TAssets']-df['cashCEquiv'])-(df['TLiab']-df['STBorr']-df['CBBorr']))/df['TAssets'].shift(1)
    df['O'] = -1.32+6.03*df['TLiab']/df['TAssets'].shift(1)-1.43*(df['TCL']-df['TCA'])/df['TAssets'].shift(1)+0.076*df['TCL']/df['TCA']\
        -1.72*(df['TLiab']>df['TAssets'])-2.37*df['NIncome']/df['TAssets'].shift(1)-1.83*df['TProfit']/df['TLiab'].shift(1)\
        +0.285*(df['NIncome']<0.0)-0.521*df['NIncome'].diff(periods=1)/(abs(df['NIncome'])+abs(df['NIncome'].shift(1)))
    return df

def comp_factors(trade_cal, universe_ticker='000300'):
    """
    计算相关财务、量价因子
    Size (S): NLSIZE
    Book to markt ratio(B/M): 1/PB
    Earning to price ratio(E/P): 1/PE
    Cash flow to price ratio(C/P): CashFlowPS
    Market beta(B): Beta252
    Momemtom(MOM): REVS250
    Long-term reversal(LTR): 过去5年的累计收益率
    Short-term reversal(STR): REVS20, 上个月的收益率
    *Idiosyncratic volatility(IdVol): 特异性波动率
    Maximum daily return over the past month(MAX): 过去一月中最大日收益
    Expected idiosyncratic skewness(EIS): 特异性偏度
    Return on equity (ROE): ROE
    Returns on assets (ROA): ROA
    Total asset growth (AG): TotalAssetGrowRate
    Abnormal capital investments (CI): 资本支出增长率；论文中将其用level=0.01进行winsorize；反向指标
    Investment growth (IG): InvestCashGrowRate, 投资活动产生的现金流量净额增长率
    Investment-to-capital ratio (I/K): CAPEX/TotalFixedAssets
    Investment-to-assets ratio (I/A): 其中INVT表示存货
    Accruals (ACC): 应计项目
    Net operating assets (NOA): 净经营资产
    Net stock issues (NS): 流通市值变化；其中CSHO表示流通股
    Composite stock issuance (l): 其中r(t-6,t-1)表示前六年年底至去年年底
    Leverage (LV): BLEV, 杠杆率
    O-score (O): 描述财务困境: 其中TLTA表示总负债/滞后一期总资产
    Turnover (TO): 过去3-12月的日均换手率
    *Analysts’ forecasts dispersion (D): 分析师预期分布
    :param trade_cal: 仅按trade_cal计算该日期对应的因子
    :param universe_ticker: 投资域代码
    :return factors: dict(factor_name, list of dataframe)
    """
    # 0) 创建dict of empty dataframe
    factors = dict()
    for fname in ['S','B/M','E/P','C/P','B','MOM','LTR','STR','MAX','ROE','ROA','AG','CI','IG','I/K','I/A','ACC','NOA','NS','l','LV','O','TO']:
        factors[fname] = []
    # 1) 计算量价factor
    begin_date = min(trade_cal['calendarDateNoHyphen'])
    begin_date_five_years = (datetime.datetime.strptime(begin_date, '%Y%m%d') - timedelta(365*5)).strftime('%Y%m%d')
    end_date = max(trade_cal['calendarDateNoHyphen'])
    universe = DataAPI.IdxCloseWeightGet(secID=u"",ticker=universe_ticker,beginDate=begin_date,endDate=end_date,field=u"",pandas="1")['consTickerSymbol'].unique().tolist()
    data = []
    for idy, ticker in enumerate(universe):
        t1 = time.time()
        tmp_k = DataAPI.MktEqudAdjGet(tradeDate=u"",secID=u"",ticker=ticker,isOpen="1",
                                        beginDate=begin_date_five_years,endDate=end_date,field=u"",pandas="1").sort_values(by=['tradeDate'], ascending=[True])
        tmp_k['RET'] = tmp_k['closePrice'].pct_change(periods=1)
        tmp_k['LTR'] = (tmp_k['RET']+1.0).rolling(window=750, min_periods=500, center=False).apply(lambda x: np.cumprod(x)[-1]) - 1.0
        tmp_k['MAX'] = tmp_k['RET'].rolling(window=20, min_periods=20, center=False).max()
        tmp_k['TO'] = tmp_k['turnoverRate'].rolling(window=180, min_periods=180, center=False).mean().shift(60)
        tmp_k['NS'] = tmp_k['negMarketValue'] / tmp_k['negMarketValue'].shift(1)
        tmp_k['l'] = np.log(tmp_k['marketValue'] / tmp_k['marketValue'].shift(500)) - np.log(tmp_k['closePrice'].pct_change(periods=500)+1.0)
        tmp_k['tradeDate'] = tmp_k['tradeDate'].apply(lambda x: convert_format(x, TimeDateFormat.YMDHYPHEN.value, TimeDateFormat.YMD.value))
        tmp_k = tmp_k.set_index('tradeDate', drop=False)
        for idx, (_, rows) in enumerate(trade_cal.iterrows()):
            if len(tmp_k.loc[:rows['calendarDateNoHyphen']]) > 0:
                part_df = (tmp_k.loc[:rows['calendarDateNoHyphen']].iloc[-1:]).copy(deep=True)
                part_df['tradeDate'] = rows['calendarDateNoHyphen']
                part_df = part_df.set_index('tradeDate', drop=False)
                data.append(part_df)
        if idy % 50 == 0:
            print('[Single Factor][{0}/{1}] computed daily k factors. cost:{2:.2f}'.format(idy, len(universe), time.time()-t1))
    daily_k = pd.concat(data)
    # 2) 计算财务factor
    data = []
    for idy, ticker in enumerate(universe):
        t1 = time.time()
        tmp_f = load_fiscal_data(ticker, begin_date=convert_format(begin_date_five_years, TimeDateFormat.YMD.value, TimeDateFormat.YMDHYPHEN.value),
                                 end_date=convert_format(end_date, TimeDateFormat.YMD.value, TimeDateFormat.YMDHYPHEN.value))
        tmp_f['publishDate'] = tmp_f['publishDate'].apply(lambda x: convert_format(x, TimeDateFormat.YMDHYPHEN.value, TimeDateFormat.YMD.value))
        tmp_f = tmp_f.set_index('publishDate', drop=False)
        for idx, (_, rows) in enumerate(trade_cal.iterrows()):
            if len(tmp_f.loc[:rows['calendarDateNoHyphen']]) > 0:
                part_df = (tmp_f.loc[:rows['calendarDateNoHyphen']].iloc[-1:]).copy(deep=True)
                part_df['tradeDate'] = rows['calendarDateNoHyphen']
                part_df = part_df.set_index('tradeDate', drop=False)
                data.append(part_df)
        if idy % 50 == 0:
            print('[Single Factor][{0}/{1}] computed fiscal factors. cost:{2:.2f}'.format(idy, len(universe), time.time()-t1))
    fiscal_df = pd.concat(data)
    del data
    # 3) 计算分析师factor, 先不实现
    # 4) 已有的factor可以直接幅值
    for idx, (_, rows) in enumerate(trade_cal.iterrows()):
        t1 = time.time()
        const_ticker = DataAPI.IdxCloseWeightGet(beginDate=rows['calendarDateNoHyphen'],endDate=rows['calendarDateNoHyphen'],\
                                                 secID=u"",ticker=universe_ticker,field=u"",pandas="1")
        factor_df = DataAPI.MktStockFactorsOneDayProGet(tradeDate=rows['calendarDateNoHyphen'],secID=u"",\
                                                        ticker=','.join(const_ticker['consTickerSymbol'].unique().tolist()),\
                                        field=u"ticker,tradeDate,NLSIZE,PB,PE,CashFlowPS,Beta252,REVS250,REVS20,ROE,ROA,TotalAssetGrowRate,BLEV",pandas="1")
        factor_df['tradeDate'] = rows['calendarDateNoHyphen']
        factor_df['PB'] = 1.0 / factor_df['PB']
        factor_df['PE'] = 1.0 / factor_df['PE']
        for new_name, ori_name in {'S': 'NLSIZE', 'B/M': 'PB', 'E/P': 'PE', 'C/P': 'CashFlowPS', 'B': 'Beta252', 'MOM': 'REVS250', 'STR': 'REVS20', 'ROE': 'ROE',
                                  'ROA': 'ROA', 'AG': 'TotalAssetGrowRate', 'LV': 'BLEV'}.items():
            factors[new_name].append(factor_df[['ticker', ori_name, 'tradeDate']].rename(columns={ori_name: new_name}))
        # 技术指标factor
        if rows['calendarDateNoHyphen'] in daily_k.index:
            for fname in ['LTR', 'MAX', 'TO', 'NS', 'l']:
                new_df = daily_k.loc[rows['calendarDateNoHyphen'], ['ticker', fname, 'tradeDate']]
                factors[fname].append(new_df)
        # 财务factor
        if rows['calendarDateNoHyphen'] in fiscal_df.index:
            for fname in ['CI', 'I/K', 'I/A', 'ACC', 'NOA', 'O', 'IG']:
                new_df = fiscal_df.loc[rows['calendarDateNoHyphen'], ['ticker', fname, 'tradeDate']]
                factors[fname].append(new_df)
        if idx % 50 == 0:
            print('[factorLoad][{0}/{1}] done. cost:{2:.2f}'.format(idx, len(trade_cal.index), time.time()-t1))
    return factors


start_date = '20070101'
end_date = '20180729'
frequency = 'monthly'

# 设置交易日、调仓周期、投资域
date_proc = TradeDateProcess()
date_proc.update_trade_date()

trade_date = date_proc.trade_date
factor_date = date_proc._get_date_list(start_date,end_date,frequency,-1)
price_date = [date_proc.step_trade_date(date,1,trade_date) for date in factor_date]

# 读取相关因子
trade_cal = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=start_date,endDate=end_date,field=u"",pandas="1")
trade_cal = trade_cal[trade_cal['isMonthEnd'] == 1]
trade_cal['calendarDateNoHyphen'] = trade_cal['calendarDate'].apply(lambda x: convert_format(x, TimeDateFormat.YMDHYPHEN.value, TimeDateFormat.YMD.value))

flag, factors = load_pickle('pls/pls_single_factor.data', ori_obj=dict())
if not flag:
    t1 = time.time()
    # 设置投资域为hs300
    factors = comp_factors(trade_cal, universe_ticker='000300')
    # 转换dataframe index为timestamp格式
    for fname in factors:
        new_df = pd.concat(factors[fname])
        new_df = new_df.replace([np.inf, -np.inf], np.nan).dropna(how='any', axis=0)
        factors[fname] = pd.crosstab(index=new_df['tradeDate'], columns=new_df['ticker'], values=new_df[fname], aggfunc='sum')
        factors[fname].index = factors[fname].index.to_datetime()
    save_pickle('pls/pls_single_factor.data', factors)
    print('[Single Factor] computed done. cost:{0:.2f}'.format(time.time()-t1))
else:
    print('[Single Factor] loaded done.')
    
    
# 单因子测试框架
import pandas as pd
import numpy as np
import datetime
from quartz.universe import set_universe
from dateutil.parser import parse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# CAL.PyCAL中包含font
from CAL.PyCAL import *
cal = Calendar('China.SSE')
import scipy.stats as st
from scipy.stats.mstats import gmean
from random import randint
font.set_size(15)

class SignalGenerate():
    def __init__(self,DateProc):
        self.factor_date = None
        self.price_date = None
        self.date_proc = DateProc
        self.trade_date = self.date_proc.trade_date
        self.benchmark_symbol = None
    
    def set_analysis_parameter(self,start_date,end_date,benchmark_symbol,freq=None,point=None):
        """
        这里的price_date是factor_date的下一交易日
        :param start_date: 起始日期
        :param end_date: 终止日期
        :param benchmark_symbol: 基准代码
        :param freq: 频率,日、周、月
        :param point: 取哪一天作为调仓日
        """
        self.factor_date = self.date_proc._get_date_list(start_date,end_date,freq,point)
        self.price_date = [self.date_proc.step_trade_date(date,1,self.trade_date) for date in self.factor_date]
        self.benchmark_symbol = benchmark_symbol
        
    def get_analysis_factor_df(self,factor_compute):
        """
        根据动态投资域，得到相应因子值
        :param factor_compute: 因子dataframe
        """
        factor_dict ={}
        for date in self.factor_date:
            universe = set_universe(self.benchmark_symbol,date=date)
            factor_dict[date] = factor_compute.get_factor(universe, date)
        factor_df = pd.DataFrame(factor_dict).T
        return factor_df
    
    def get_o2o_return(self,signal_df):
        """
        获取对应时间点的开盘价，计算开盘价买入、下一调仓日开盘价卖出之间的收益
        :param signal_df: 因子dataframe
        :return eq_return_df: 个股收益率dataframe
        """
        quotation_list = []
        universe = list(signal_df.columns)
        for date in self.price_date:
            universe = set_universe(symbol=self.benchmark_symbol,date=date)
            one_day_quotation = DataAPI.MktEqudAdjGet(tradeDate=date.strftime('%Y%m%d'),secID=universe,ticker=u"",isOpen="",
                                                      beginDate=u"",endDate=u"",field=u"secID,tradeDate,openPrice",pandas="1")
            # print(universe, date, one_day_quotation)
            quotation_list.append(one_day_quotation)
        quotation_df = pd.concat(quotation_list,axis=0)
        quotation_df['tradeDate'] = quotation_df['tradeDate'].map(lambda x:parse(x))
        quotation_df['secID'] = quotation_df['secID'].map(lambda x: str(x)[:6])
        quotation_df = quotation_df.set_index(['tradeDate','secID'])
        eq_return_df = quotation_df.unstack().replace(0,np.NaN).pct_change().shift(-1).iloc[0:-1].stack()
        eq_return_df = eq_return_df.rename(columns={'openPrice':'o2o_ret'})
        return eq_return_df
    
    def get_index_o2o_return(self):
        """
        获取指数开买、开卖间的收益率
        :return eq_return_df: 指数收益率dataframe
        """
        index_quotation_list = []
        for date in self.price_date:
            index_day_quotation = DataAPI.MktIdxdGet(tradeDate=date.strftime('%Y%m%d'),indexID=self.benchmark_symbol,ticker=u"",
                               beginDate=u"",endDate=u"",exchangeCD=u"XSHE,XSHG",field=u"tradeDate,openIndex",pandas="1")
            # print(index_day_quotation)
            index_quotation_list.append(index_day_quotation)
        index_quotation_df = pd.concat(index_quotation_list,axis=0)
        index_quotation_df['tradeDate'] = index_quotation_df['tradeDate'].map(lambda x:parse(x))
        return index_quotation_df.set_index(['tradeDate']).pct_change().shift(-1)
    
    def get_analysis_summary(self,factor_df,eq_return_df):
        """
        将因子df与收益df做合并
        TODO: 这里没考虑开盘涨停无法买入，开盘跌停无法卖出等case
        :param factor_df: 因子dataframe
        :param eq_return_df: 收益dataframe
        """
        factor_price_dict = {}
        for i,date in enumerate(self.factor_date):
            factor_price_dict[date] = self.price_date[i]
        summary_df = pd.concat([eq_return_df,factor_df.rename(index=factor_price_dict).stack()],axis=1).rename(columns={0:'factor_value'})
        summary_df = summary_df.dropna(subset=['factor_value'])
        return summary_df


class SignalAnalysis():
    def __init__(self,generate_class):
        self.signal_generate = generate_class
        self.index_return = self.signal_generate.get_index_o2o_return()

    def plot_signal_coverage(self,analysis_df,universe_lens,plot=True):
        """
        画出信号覆盖率
        :param analysis_df: 因子、收益dataframe
        :param universe_lens: 投资域内股票数量
        :param plot: 是否画图
        """
        signal_range = universe_lens
        siganl_coverage = analysis_df.groupby(analysis_df.index.get_level_values(0)).apply(lambda x:len(x.dropna()))
        if plot:
            fig = plt.figure(figsize=(18,7))
            ax = fig.add_subplot(111)
            ax.fill_between(siganl_coverage.dropna().index,0,siganl_coverage.dropna().values,color='skyblue')
            ax.set_ylim([0,signal_range])    
            ax.set_title(u"测试区间信号覆盖", fontproperties=font, fontsize=18)
            return np.mean(siganl_coverage) / universe_lens
        return np.mean(siganl_coverage) / universe_lens, siganl_coverage

    def plot_random_distribution_of_signal(self, factor_df):
        """
        随机选取9期因子数据，看分布，左偏、右偏等问题
        """
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20,15))
        for i in range(3):
            for j in range(3):
                rand = randint(0, len(factor_df) - 1)
                factor_df.iloc[rand].dropna().hist(bins=100,ax=ax[i][j])

    def factor_grouper(self,summary_df,quantile_point,col='factor_value'):
        """
        对因子进行分组
        :param summary_df: 因子、收益dataframe
        :param quantile_point: 分组
        :return quantile_df: 分组后的dataframe
        """
        def quantile_function(df, quantile_point, col='factor_value'):
            #Get the quantile value
            #Select the data in the group that falls at or below the quantile value and return it
            def group_num(x):
                if quantile_point >= 3:
                    if x <= df[col].quantile(1.0/quantile_point):
                        return 1
                    for i in range(2,quantile_point):
                        if x <=df[col].quantile(1.0*i/quantile_point) and x > df[col].quantile(1.0*(i-1)/quantile_point):
                            return i
                    if x > df[col].quantile(1.0*i/quantile_point):
                        return quantile_point
                elif quantile_point == 2:
                    if x <= df[col].quantile(1.0/quantile_point):
                        return 1
                    else:
                        return 2
                else:
                    raise ValueError('quantile group num must larger than 2!')
            df['group'] = df[col].map(lambda x:group_num(x))
            return df
        return summary_df.groupby(level=0).apply(quantile_function,quantile_point)
                
    def fama_macbeth_decay(self,factor_df,eq_return_df,decay=6):
        """
        因子对收益的预测能力
        :param factor_df: 因子dataframe
        :param eq_return_df: 个股收益dataframe
        :param decay: 滞后测试用到的期数
        :return decay_summary_df: decay分析结果
        """
        summary_df_list = []
        for i in range(decay):
            factor_df_ = factor_df.shift(i).dropna(how='all')
            summary_df = self.signal_generate.get_analysis_summary(factor_df_,eq_return_df)
            summary_df = summary_df.dropna(subset=['factor_value'])
            fm = pd.fama_macbeth(y=summary_df['o2o_ret'],x=summary_df[['factor_value']])
            summary_df_list.append([fm.mean_beta['factor_value'],fm.t_stat['factor_value'],i])
        return pd.DataFrame(summary_df_list).rename(columns={0:'mean_Coeff',1:'t_stat',2:'Decay'})
    
    def _compute_turnover(self,select_df):
        """
        计算换手率
        """
        select_df = select_df.unstack()
        accumulate_turnover = 0
        for i in range(len(select_df)-1):
            diff_universe = set(select_df.iloc[i].dropna().index) - set(select_df.iloc[i+1].dropna().index)
            dropna_len = len(select_df.iloc[i].dropna())
            accumulate_turnover = accumulate_turnover + 1.0*len(diff_universe)/dropna_len*2 if dropna_len != 0 else accumulate_turnover
        return accumulate_turnover / len(select_df) * 12
    
    def _compute_annual_return(self,ret_series):
        """
        计算年化收益
        """
        return gmean(ret_series.dropna()+1.)**12 - 1.
    
    def _compute_annual_std(self,ret_series):
        """
        计算年化收益标准差，也是风险的一种
        """
        return ret_series.dropna().std() * np.sqrt(12)
    
    def _compute_sharp(self,ret_series):
        """
        计算夏普率
        """
        return 1.0*self._compute_annual_return(ret_series) / self._compute_annual_std(ret_series)
    
    def _compute_mean_return(self,ret_series):
        """
        计算平均收益
        """
        return np.mean(ret_series)
    
    def _compute_sample_standard_deviation(self,ret_series):
        """
        计算收益的样本标准差
        """
        return np.std(ret_series, ddof=1)
    
    def long_only_ret(self,summary_df,benchmark,threhold=None,quantile_num=None):
        """
        仅做多收益
        :param summary_df: 因子、收益dataframe
        :param benchmark: 基准dataframe
        :param threhold: 百分比分组
        :param quantile_num: 分组数
        """
        def cut_big(x):
            if threhold is not None:
                x = x[x['factor_value']>x['factor_value'].quantile(threhold)]
                x.index = x.index.droplevel()
                return x
            elif quantile_num is not None:
                x = x[x['factor_value']>x['factor_value'].quantile(1 - 1.0/quantile_num)]
                x.index = x.index.droplevel()
                return x
            else:
                raise ValueError('you must set one of threhold or quantile_num!')
        summary_df_select = summary_df.groupby(summary_df.index.get_level_values(0)).apply(lambda x:cut_big(x))
        summary_df_mean = summary_df_select.groupby([summary_df_select.index.get_level_values(0)]).mean()
        if benchmark == 'equal':
            time_df_summary = summary_df_mean['o2o_ret'] - summary_df.groupby(summary_df.index.get_level_values(0))['o2o_ret'].mean()
        elif benchmark == 'index':
            time_df_summary = summary_df_mean['o2o_ret'] - self.index_return['openIndex']
        else:
            raise ValueError('benchmark only support equal or index !')
        return summary_df_select, time_df_summary
    
    def long_short_ret(self,summary_df,threhold=None,quantile_num=None):
        """
        多空收益
        :param summary_df: 因子、收益dataframe
        :param threhold: 百分比分组
        :param quantile_num: 分组数
        """
        def cut_df(x,flag):
            if threhold is not None:
                if flag == 'big':  
                    x = x[x['factor_value']>x['factor_value'].quantile(threhold)]
                    x.index = x.index.droplevel()
                    return x
                elif flag == 'small':
                    x = x[x['factor_value']<x['factor_value'].quantile(threhold)]
                    x.index = x.index.droplevel()
                    return x
            elif quantile_num is not None:
                if flag == 'big':
                    x = x[x['factor_value']>x['factor_value'].quantile(1 - 1.0/quantile_num)]
                    x.index = x.index.droplevel()
                    return x
                elif  flag == 'small':
                    x = x[x['factor_value']<x['factor_value'].quantile(1.0/quantile_num)]
                    x.index = x.index.droplevel()
                    return x
            else:
                raise ValueError('you must set one of threhold or quantile_num!')
        
        summary_df_big = summary_df.groupby(summary_df.index.get_level_values(0)).apply(lambda x:cut_df(x,'big'))
        summary_df_small = summary_df.groupby(summary_df.index.get_level_values(0)).apply(lambda x:cut_df(x,'small'))
        summary_df_big_mean = summary_df_big.groupby([summary_df_big.index.get_level_values(0)]).mean()
        summary_df_small_mean = summary_df_small.groupby([summary_df_small.index.get_level_values(0)]).mean()
        ret_summary = summary_df_big_mean['o2o_ret'] - summary_df_small_mean['o2o_ret']
        return summary_df_big, summary_df_small,ret_summary
    
    def get_decay_return_plot(self,factor_df,eq_return_df,ret_type,benchmark=None,threhold=None,quantile_num=None,decay=6):
        """
        收益的decay分析
        :param factor_df: 因子dataframe
        :param eq_return_df: 收益dataframe
        :param ret_type: long_only / long_short
        :param benchmark: 基准dataframe
        :param threhold: 百分比分组
        :param quantile_num: 分组数
        :param decay: 滞后期数
        """
        summary_df_list = []
        legend_col = []
        result_dict = {'年化收益':[],'标准差':[],'夏普率':[],'年化换手率':[]}   
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        for i in range(decay):
            factor_df_ = factor_df.shift(i).dropna(how='all')
            summary_df = self.signal_generate.get_analysis_summary(factor_df_,eq_return_df)
            summary_df = summary_df.dropna(subset=['factor_value'])
            if ret_type == 'long_only':
                select_df, time_df_summary = self.long_only_ret(summary_df,quantile_num = quantile_num,threhold = threhold,benchmark=benchmark)
                result_dict['年化换手率'].append(self._compute_turnover(select_df['o2o_ret']))
                
            elif ret_type == 'long_short':
                select_big, select_small , time_df_summary = self.long_short_ret(summary_df,quantile_num = quantile_num,threhold = threhold)                      
                long_turnover = self._compute_turnover(select_big['o2o_ret'])
                short_turnover = self._compute_turnover(select_small['o2o_ret'])                            
                result_dict['年化换手率'].append(long_turnover + short_turnover)
            
            result_dict['年化收益'].append(self._compute_annual_return(time_df_summary))
            result_dict['标准差'].append(self._compute_annual_std(time_df_summary))
            result_dict['夏普率'].append(self._compute_sharp(time_df_summary))
            time_df_summary = (1+time_df_summary).cumprod()
            ax.plot(time_df_summary.index,time_df_summary.values,linewidth=2.5)
            legend_col.append('decay '+str(i))
            summary_df_list.append(time_df_summary)
        
        decay_summary_return = pd.concat(summary_df_list,axis=1)
        ax.legend(legend_col,loc='upper left',frameon= False)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if ret_type == 'long_only':
            ax.set_title(u"测试区间long_only累计超额收益衰减", fontproperties=font, fontsize=18)
        elif ret_type == 'long_short':
            ax.set_title(u"测试区间long_short累计收益衰减", fontproperties=font, fontsize=18)
        result_df = pd.DataFrame(result_dict)
        result_df.index.name = 'decay'
        result_df = result_df[['年化收益','标准差','夏普率','年化换手率']]
        return decay_summary_return,result_df
    
    def get_long_short_return_stats(self,factor_df,eq_return_df,ret_type,benchmark=None,threhold=None,quantile_num=None):
        """
        计算平均收益、标准差、t统计量(关于什么的t统计)
        :param factor_df: 因子dataframe
        :param eq_return_df: 收益dataframe
        :param ret_type: long_only / long_short
        :param benchmark: 基准dataframe
        :param threhold: 百分比分组
        :param quantile_num: 分组数
        """
        result_dict = {'平均对冲收益':[],'对冲收益标准差':[]} 
        factor_df_ = factor_df.dropna(how='all')
        summary_df = self.signal_generate.get_analysis_summary(factor_df_,eq_return_df)
        summary_df = summary_df.dropna(subset=['factor_value'])
        if ret_type == 'long_only':
            select_df, time_df_summary = self.long_only_ret(summary_df,quantile_num=quantile_num,threhold=threhold,benchmark=benchmark)
        elif ret_type == 'long_short':
            select_big, select_small , time_df_summary = self.long_short_ret(summary_df,quantile_num=quantile_num,threhold=threhold)
        result_dict['平均对冲收益'].append(self._compute_mean_return(time_df_summary))
        result_dict['对冲收益标准差'].append(self._compute_sample_standard_deviation(time_df_summary))
        result_df = pd.DataFrame(result_dict)
        result_df.index.name = 'decay'
        result_df = result_df[['平均对冲收益','对冲收益标准差']]
        return result_df     
    
    def get_quantile_cumulative_return_plot(self,summary_df,quantile_num,benchmark):
        """
        分组累计超额收益
        :param summary_df: 因子、收益dataframe
        :param quantile_num: 分组数
        :param benchmark: 基准dataframe
        """
        # if out_benchmark:
        #     index_o2o_return = self.get_index_o2o_return()
        group_summary_df = self.factor_grouper(summary_df,quantile_num)
        time_group_df = group_summary_df.groupby([group_summary_df.index.get_level_values(0),'group']).mean()
        # print time_group_df
        def get_return(x):
            x.index = x.index.get_level_values(0)
            if benchmark == 'equal':
                return x['o2o_ret'] - summary_df.groupby(summary_df.index.get_level_values(0))['o2o_ret'].mean()
            elif benchmark == 'index':
                return x['o2o_ret'] - self.index_return['openIndex']
            else:
                raise ValueError('benchmark only support equal or index !')
        time_group_df_summary = time_group_df.groupby(time_group_df.index.get_level_values(1)).apply(lambda x:get_return(x)).T
        ret = time_group_df_summary.apply(lambda x:self._compute_annual_return(x))
        sigma = time_group_df_summary.apply(lambda x:self._compute_annual_std(x))
        result = pd.DataFrame([ret,sigma],index = ['年化收益','标准差']).T
        result['夏普率'] = result['年化收益'] / result['标准差']
        result['年化换手率'] = group_summary_df.groupby(['group']).apply(lambda x:self._compute_turnover(x['o2o_ret']))
        result.index.name = 'group'
        time_group_df_summary_ = (1 + time_group_df_summary).cumprod()
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)
        for group in time_group_df_summary_.columns:
            ax.plot(time_group_df_summary_.index,time_group_df_summary_[group].values,linewidth=2.5)
        ax.legend(time_group_df_summary_.columns,loc='upper left',frameon= False)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_title(u"测试区间分位数分组累计超额收益", fontproperties=font, fontsize=18)
        return time_group_df_summary_ , result
    
    def get_quantile_return_mean_plot(self,summary_df,quantile_num,plot=True):
        """
        分组平均超额收益
        :param summary_df: 因子、收益dataframe
        :param quantile_num: 分组数
        :param plot: 是否画图
        """
        group_summary_df = self.factor_grouper(summary_df,quantile_num)
        time_group_df = group_summary_df.groupby([group_summary_df.index.get_level_values(0),'group']).mean()
        time_group_df_summary = time_group_df.groupby(time_group_df.index.get_level_values(1)).mean()
        if plot:
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_subplot(111)
            time_group_df_summary['o2o_ret'].plot(kind='bar',ax=ax,color='deepskyblue')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['right'].set_color('none')
            ax.yaxis.set_ticks_position('left')
            ax.spines['bottom'].set_position('zero')
            ax.set_title(u"测试区间分组平均收益", fontproperties=font, fontsize=18)
            ax.grid()
        return time_group_df_summary

    def get_ic_series(self,fator_return_df):
        """
        返回IC序列
        :param fator_return_df: 因子、收益dataframe
        """
        def omit_nan_spearmanr(x):
            x = x.dropna()
            if len(x) > 0:
                return st.spearmanr(x['o2o_ret'],x['factor_value'])
        ic_result = fator_return_df.groupby(level=0).apply(lambda x:omit_nan_spearmanr(x))
        ic_result = pd.DataFrame(ic_result.dropna())
        ic_result['IC'] = ic_result[0].map(lambda x:x[0])
        ic_result['IC_Pvalue'] = ic_result[0].map(lambda x:x[1])
        return ic_result[['IC','IC_Pvalue']]
    
    def get_ic_plot(self,fator_return_df):
        """
        ic分析并画图
        :param fator_return_df: 因子、收益dataframe
        """
        ic_series = self.get_ic_series(fator_return_df)
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(20,5))
        ax[0].plot(ic_series.index,ic_series['IC'],'o-',color='cyan')
        ax[0].plot(ic_series.index,pd.rolling_mean(ic_series['IC'],10),color = 'deepskyblue',linewidth = 2.5)
        ax[1].plot(ic_series.index,ic_series['IC_Pvalue'],'v-',color = 'lightgreen')
        ax[1].plot(ic_series.index,[0.05]*len(ic_series),'r',linewidth=2)
        ax[0].legend(['IC','MA10_IC'],loc='upper left',prop=font,frameon= False,handlelength = 3,handletextpad=0.1,borderpad = 0.1)
        ax[1].legend(['IC_Pvalue','Pvalue_0.05'],loc='upper left',prop=font,frameon= False,handlelength = 3,handletextpad=0.1,borderpad = 0.1)
        def myfunc(x, pos=0): 
            return "{0:.2f}%".format(x * 100)
        ax[1].yaxis.set_major_formatter(FuncFormatter(myfunc))
        ax[0].set_ylabel('IC')
        ax[1].set_ylabel('IC_Pvalue')
        ax[0].set_title(u"IC时序图", fontproperties=font, fontsize=18)
        ax[1].set_title(u"IC_Pvalue时序图", fontproperties=font, fontsize=18)
        ax[0].grid()
        ax[1].grid()

        
def plot_signal_coverage(signal_coverages, group_dfs, universe_lens, factor_type_dic, factor_type):
    """
    因子覆盖图
    :param signal_coverages: dict of series
    :param factor_type_dic: 因子类型字典
    :param factor_type: 因子类型
    """
    fig = plt.figure(figsize=(30,20))
    tot = len(factor_type_dic[factor_type])
    for idx, fname in enumerate(factor_type_dic[factor_type]):
        ax = fig.add_subplot(2, tot, idx+1)
        ax.fill_between(signal_coverages[fname].dropna().index,0,signal_coverages[fname].dropna().values,color='skyblue')
        ax.set_ylim([0,universe_lens])    
        ax.set_title(u"因子{0}:信号覆盖".format(fname), fontproperties=font, fontsize=12)
        plt.xticks(rotation=90)
        plt.suptitle(u'因子类型:{0}'.format(factor_type), fontproperties=font, fontsize=16)
        
        ax = fig.add_subplot(2, tot, tot+idx+1)
        group_dfs[fname]['o2o_ret'].plot(kind='bar',ax=ax,color='deepskyblue')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['right'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position('zero')
        ax.set_title(u"因子{0}:分组平均收益".format(fname), fontproperties=font, fontsize=12)
        ax.grid()
    return

# 计算开盘价间的收益dataframe
t1 = time.time()
signal_generate = SignalGenerate(date_proc)
signal_generate.set_analysis_parameter(start_date,end_date,'000300.ZICN',frequency,-1)
signal_analysis = SignalAnalysis(signal_generate)

factor_df = factors['S']
eq_return_df = signal_generate.get_o2o_return(factor_df)
print(u"[O2O-Return] compute o2o return, cost: {0:.2f}s".format(time.time()-t1))

# 为因子划分类型
factor_type_dic = {'techniqual': ['S', 'B', 'MAX', 'NS', 'l', 'TO'],
                   'value': ['B/M', 'E/P', 'C/P', 'MOM', 'LTR', 'STR'],
                   'quality': ['ROA', 'ROE', 'I/A', 'I/K', 'ACC', 'NOA', 'LV', 'O'],
                   'growth': ['AG', 'CI', 'IG']}

# 进行单因子测试
flag, data = load_pickle('pls/pls_single_factor_test.data', ori_obj=tuple())
if not flag:
    t1 = time.time()
    result_df = pd.DataFrame()
    signal_coverages = dict()
    group_dfs = dict()
    for fname in ['S', 'B/M', 'E/P', 'C/P', 'B', 'MOM','LTR', 'STR', 'MAX', 'ROE', 'ROA', 'AG', 'CI', 'IG', 'I/K', 'I/A','ACC', 'NOA', 'NS', 'l', 'LV', 'O', 'TO']:
        factor_df = factors[fname]
        # 针对原始因子，统计long-short平均收益等
        series = pd.Series(factor_df.values.ravel()).dropna()
        # factor_df.T.describe().T[['mean','std','min','max']].plot()
        stats_df = pd.DataFrame(series.describe()).T
        # print(stats_df.to_html())
        # 获取股票行情，得到信号与下一期收益率的汇总
        summary_df = signal_generate.get_analysis_summary(factor_df,eq_return_df)
        # 信号覆盖情况
        coverage, signal_coverage = signal_analysis.plot_signal_coverage(summary_df['factor_value'],300,plot=False)
        # 信号分组平均收益率
        group_summary_df = signal_analysis.get_quantile_return_mean_plot(summary_df,5,plot=False)
        # 信号long_short收益率及其decay
        tmp_df = signal_analysis.get_long_short_return_stats(factor_df,eq_return_df,'long_short',benchmark='equal',quantile_num=5)
        tmp_df['fname'] = fname
        tmp_df['oriSignalMean'] = stats_df['mean']
        tmp_df['oriSignalStd'] = stats_df['std']
        tmp_df['oriSignalMin'] = stats_df['min']
        tmp_df['oriSignalMax'] = stats_df['max']
        tmp_df['coverage'] = coverage
        result_df = result_df.append(tmp_df)
        signal_coverages[fname] = signal_coverage
        group_dfs[fname] = group_summary_df
    save_pickle('pls/pls_single_factor_test.data', (result_df, signal_coverages, group_dfs))
    print('[Single Factor] single factor test computed done. cost: {0:.2f}'.format(time.time()-t1))
else:
    result_df, signal_coverages, group_dfs = data
    print('[Single Factor] single factor test loaded done.')

print(result_df.to_html())
for factor_type in factor_type_dic:
    plot_signal_coverage(signal_coverages, group_dfs, universe_lens=300, factor_type_dic=factor_type_dic, factor_type=factor_type)
    
    
from collections import OrderedDict

def base_line_model(factors, eq_return_df, factor_price_dic, history_len=250):
    """
    1) 回归因子收益
    2) 回归个股预期收益
    :param factors: 因子dict of dataframe
    :param eq_return_df: 个股收益dataframe
    :param history_len: 历史数据长度
    :return baseline_df: 基准模型
    """
    eq_return_df = eq_return_df.reset_index().copy(deep=True)
    eq_return_df = pd.crosstab(index=eq_return_df['tradeDate'], columns=eq_return_df['secID'], values=eq_return_df['o2o_ret'], aggfunc='sum')
    price_factor_dic = {v:k for k,v in factor_price_dic.items()}
    eq_return_df.index = [price_factor_dic[k] if k in price_factor_dic else k for k in eq_return_df.index]
    next_td_dic = {td: eq_return_df.index[idx+1] for idx, td in enumerate(eq_return_df.index) if idx+1 < len(eq_return_df.index)}
    eq_return_df_unstack = eq_return_df.unstack().reset_index().rename(columns={'level_1': 'tradeDate', 0: 'RET', 'secID': 'ticker'})
    # step 1) 回归因子收益
    t1 = time.time()
    factors_ret = dict()
    factors_new = dict()
    # print(eq_return_df_unstack.head(5))
    for fn in factors:
        t4 = time.time()
        factor_df = factors[fn].copy(deep=True).unstack().reset_index().rename(columns={'level_1': 'tradeDate', 0: fn})
        factor_df = factor_df.merge(eq_return_df_unstack, left_on=['tradeDate', 'ticker'], right_on=['tradeDate', 'ticker'], how='outer')
        factors_new[fn] = factor_df
        g = factor_df.groupby(by=['tradeDate'])
        data = []
        for td, part_df in g:
            part_df = part_df.reset_index().set_index('ticker', drop=True).dropna()
            if td not in next_td_dic:
                continue
            next_td = next_td_dic[td]
            if len(part_df.index) == 0:
                data.append([next_td, np.nan])
                continue
            part_df[fn] = standardize(part_df[fn])
            part_df = part_df.set_index('index', drop=True)
            factors_new[fn].loc[part_df.index, fn] = part_df[fn]
            x = np.array(part_df[fn]).reshape((len(part_df[fn]), 1))
            y = np.array(part_df['RET'])
            linreg = LinearRegression()
            model = linreg.fit(x, y)
            data.append([next_td, model.coef_])
        factors_ret[fn] = pd.DataFrame(data=data, columns=['tradeDate', 'factorReturn']).set_index('tradeDate', drop=True)
        print('[Baseline][step1] regress factor:{0} return. cost:{1:.2f}'.format(fn, time.time()-t4))
    t2 = time.time()
    print('[Baseline][step1] regress factor return. cost:{0:.2f}'.format(t2-t1))
    # step 2) 使用过去T段时间因子收益平均值作为预测
    for fn in factors:
        factors_ret[fn] = factors_ret[fn].rolling(window=history_len, min_periods=0, center=False).apply(lambda x: x[~np.isnan(x)].mean())
        # print(factors_ret[fn].head(5))
    t3 = time.time()
    print('[Baseline][step2] moving average factor return. cost:{0:.2f}'.format(t3-t2))
    # step 3) 回归个股预期收益
    baseline_df = []
    for fn in factors:
        factor_df = factors_new[fn].rename(columns={fn: 'X'})
        factor_df = factor_df.merge(factors_ret[fn].reset_index().rename(columns={'factorReturn': 'Y'}), left_on=['tradeDate'], right_on=['tradeDate'], how='outer')
        baseline_df.append(factor_df)
    g = pd.concat(baseline_df, axis=0).groupby(by=['tradeDate', 'ticker'])
    data = []
    for (td, ticker), part_df in g:
        part_df = part_df.dropna()
        if len(part_df.index) == 0:
            data.append([td, ticker, np.nan])
            continue
        x = np.array(part_df['X']).reshape((len(part_df['X']), 1))
        y = np.array(part_df['Y'])
        linreg = LinearRegression()
        model = linreg.fit(x, y)
        data.append([td, ticker, model.coef_[0]])
    baseline_df = pd.DataFrame(data=data, columns=['tradeDate', 'ticker', 'factor'])
    baseline_df = pd.crosstab(index=baseline_df['tradeDate'], columns=baseline_df['ticker'], values=baseline_df['factor'], aggfunc='sum')
    print('[Baseline][step3] regress equity return. cost:{0:.2f}'.format(time.time()-t3))
    return baseline_df

factor_price_dic = {fd: pd for fd, pd in zip(factor_date, price_date)}
flag, baseline_df_dic = load_pickle('pls/pls_single_factor_baseline.data', ori_obj=OrderedDict())
if not flag:
    t1 = time.time()
    for history_len in [1, 12, 24, 36]:
        t2 = time.time()
        baseline_df = base_line_model(factors, eq_return_df, factor_price_dic, history_len=history_len)
        # print(baseline_df.head(5).to_html())
        baseline_df_dic[history_len] = baseline_df
        print('[Baseline] base line model computed done. history len: {0}. cost: {1:.2f}'.format(history_len, time.time()-t2))
    save_pickle('pls/pls_single_factor_baseline.data', baseline_df_dic)
    print('[Baseline] base line model computed done. cost: {0:.2f}'.format(time.time()-t1))
else:
    print('[Baseline] base line model loaded done.')

# 针对原始因子，统计long-short平均收益等
result_df = pd.DataFrame()
for history_len in baseline_df_dic:
    factor_df = baseline_df_dic[history_len]
    series = pd.Series(factor_df.values.ravel()).dropna()
    stats_df = pd.DataFrame(series.describe()).T
    summary_df = signal_generate.get_analysis_summary(factor_df,eq_return_df)
    coverage, signal_coverage = signal_analysis.plot_signal_coverage(summary_df['factor_value'],300,plot=False)
    group_summary_df = signal_analysis.get_quantile_return_mean_plot(summary_df,5,plot=False)
    tmp_df = signal_analysis.get_long_short_return_stats(factor_df,eq_return_df,'long_short',benchmark='equal',quantile_num=5)
    tmp_df['historyLength'] = '{0:02d}'.format(history_len)
    tmp_df['oriSignalMean'] = stats_df['mean']
    tmp_df['oriSignalStd'] = stats_df['std']
    tmp_df['oriSignalMin'] = stats_df['min']
    tmp_df['oriSignalMax'] = stats_df['max']
    tmp_df['coverage'] = coverage
    result_df = result_df.append(tmp_df)
print(result_df.to_html())


from collections import OrderedDict

def latent_factor_model(factors, eq_return_df, L=5, history_len=250):
    """
    1) 回归在指定代理因子上的因子暴露
    2) 回归个股预期收益
    3) 回归代理因子收益
    4) 计算预期收益
    :param factors:
    :param eq_return_df:
    :param L: 隐变量个数
    :param history_len:
    """
    eq_return_df = eq_return_df.reset_index().copy(deep=True)
    eq_return_df = pd.crosstab(index=eq_return_df['tradeDate'], columns=eq_return_df['secID'], values=eq_return_df['o2o_ret'], aggfunc='sum')
    price_factor_dic = {v:k for k,v in factor_price_dic.items()}
    eq_return_df.index = [price_factor_dic[k] if k in price_factor_dic else k for k in eq_return_df.index]
    next_td_dic = {td: eq_return_df.index[idx+1] for idx, td in enumerate(eq_return_df.index) if idx+1 < len(eq_return_df.index)}
    eq_return_df_unstack = eq_return_df.unstack().reset_index().rename(columns={'level_1': 'tradeDate', 0: 'RET', 'secID': 'ticker'})
    # 对因子进行标准化
    t1 = time.time()
    factors_new = dict()
    for fn in factors:
        factor_df = factors[fn].copy(deep=True).unstack().reset_index().rename(columns={'level_1': 'tradeDate', 0: fn})
        g = factor_df.groupby(by=['tradeDate'])
        for td, part_df in g:
            part_df = part_df.dropna().reset_index().set_index('ticker', drop=True)
            if len(part_df.index) == 0:
                continue
            part_df[fn] = standardize(part_df[fn])
            part_df = part_df.set_index('index', drop=True)
            factor_df.loc[part_df.index, fn] = part_df[fn]
        factors_new[fn] = factor_df
    # print(factors_new[fn].tail(5))
    print('[TwoFactor][step0] factor standardize done. cost:{0:.2f}'.format(time.time()-t1))
    # 逐步解释个股收益
    latent_factor_pred_df = pd.DataFrame()
    for l in range(L):
        # step1) 回归在指定代理因子上的因子暴露
        factors_proxy = dict()
        for fn in factors:
            t2 = time.time()
            factor_df = factors_new[fn].copy(deep=True)
            factor_df = factor_df.merge(eq_return_df_unstack, left_on=['tradeDate', 'ticker'], right_on=['tradeDate', 'ticker'], how='outer')
            g = factor_df.groupby(by=['tradeDate'])
            data = []
            for td, part_df in g:
                part_df = part_df.reset_index().set_index('ticker', drop=True).dropna()
                if td not in next_td_dic:
                    continue
                next_td = next_td_dic[td]
                if len(part_df.index) == 0:
                    data.append([next_td, np.nan])
                    continue
                x = np.array(part_df['RET']).reshape((len(part_df['RET']), 1))
                y = np.array(part_df[fn])
                linreg = LinearRegression()
                model = linreg.fit(x, y)
                data.append([next_td, model.coef_])
            factors_proxy[fn] = pd.DataFrame(data=data, columns=['tradeDate', 'factorProxy']).set_index('tradeDate', drop=True)
            print('[TwoFactor][level {0}][step1] regress factor:{1} proxy. cost:{2:.2f}'.format(l, fn, time.time()-t2))
        # print(factors_proxy[fn].head(5))
        # step2) 使用过去T段时间代理因子均值作为预测
        t3 = time.time()
        for fn in factors_proxy:
            factors_proxy[fn] = factors_proxy[fn].rolling(window=history_len, min_periods=0, center=False).apply(lambda x: x[~np.isnan(x)].mean())
        print('[TwoFactor][level {0}][step2] moving average factor proxy. cost:{1:.2f}'.format(l, time.time()-t3))
        # step3) 代理因子
        def regress_exposure_proxy(factors, factors_proxy, delay=0):
            """
            回归因子暴露，在隐空间上的映射/暴露
            """
            proxy = []
            for fn in factors:
                factor_df = factors[fn].copy(deep=True).rename(columns={fn: 'Y'})
                factor_df = factor_df.merge(factors_proxy[fn].shift(-delay).reset_index().rename(columns={'factorProxy': 'X'}), left_on=['tradeDate'], right_on=['tradeDate'], how='outer')
                proxy.append(factor_df)
            g = pd.concat(proxy, axis=0).groupby(by=['tradeDate', 'ticker'])
            data = []
            for (td, ticker), part_df in g:
                part_df = part_df.dropna()
                if delay > 0 and td in next_td_dic:
                    td = next_td_dic[td]
                elif delay > 0:
                    continue
                if len(part_df.index) == 0:
                    data.append([td, ticker, np.nan])
                    continue
                x = np.array(part_df['X']).reshape((len(part_df['X']), 1))
                y = np.array(part_df['Y'])
                linreg = LinearRegression()
                model = linreg.fit(x, y)
                data.append([td, ticker, model.coef_[0]])
            proxy = pd.DataFrame(data=data, columns=['tradeDate', 'ticker', 'factorProxy'])
            return proxy
        
        proxy = regress_exposure_proxy(factors_new, factors_proxy, delay=0)
        proxy_delay = regress_exposure_proxy(factors_new, factors_proxy, delay=1)
        t4 = time.time()
        print('[TwoFactor][level {0}][step3] proxy & proxy delay. cost:{1:.2f}'.format(l, t4-t3))
        # print(proxy.head(5))
        # 代理因子收益
        proxy_delay = proxy_delay.merge(eq_return_df_unstack, left_on=['tradeDate', 'ticker'], right_on=['tradeDate', 'ticker'], how='outer')
        g = proxy_delay.groupby(by=['tradeDate'])
        data = []
        for td, part_df in g:
            part_df = part_df.dropna()
            if len(part_df.index) == 0:
                data.append([td, np.nan])
                continue
            x = np.array(part_df['factorProxy']).reshape((len(part_df['factorProxy']), 1))
            y = np.array(part_df['RET'])
            linreg = LinearRegression()
            model = linreg.fit(x, y)
            data.append([td, model.coef_])
        proxy_ret = pd.DataFrame(data=data, columns=['tradeDate', 'proxyRet']).set_index('tradeDate', drop=True)
        proxy_ret = proxy_ret.rolling(window=history_len, min_periods=0, center=False).apply(lambda x: x[~np.isnan(x)].mean())
        t5 = time.time()
        print('[TwoFactor][level {0}][step4] proxy return. cost:{1:.2f}'.format(l, t5-t4))
        # print(proxy_ret.head(5))
        # step4) 回归个股预期收益
        def compute_predict_return(proxy, proxy_ret):
            proxy_ret = proxy_ret.copy(deep=True).reset_index()
            latent_factor_df = proxy.merge(proxy_ret, left_on=['tradeDate'], right_on=['tradeDate'], how='outer')
            latent_factor_df['RET_PRED'] = latent_factor_df['factorProxy'] * latent_factor_df['proxyRet']
            return latent_factor_df
        
        eq_return_df_unstack = eq_return_df_unstack.merge(compute_predict_return(proxy_delay, proxy_ret), left_on=['tradeDate', 'ticker'], right_on=['tradeDate', 'ticker'], suffixes=['', '_y'], how='outer')
        eq_return_df_unstack['RET'] = eq_return_df_unstack['RET'] - eq_return_df_unstack['RET_PRED']
        eq_return_df_unstack = eq_return_df_unstack[['tradeDate', 'ticker', 'RET']]
        if len(latent_factor_pred_df.index) == 0:
            latent_factor_pred_df = compute_predict_return(proxy, proxy_ret)
        else:
            latent_factor_pred_df = latent_factor_pred_df.merge(compute_predict_return(proxy, proxy_ret), left_on=['tradeDate', 'ticker'], right_on=['tradeDate', 'ticker'], suffixes=['', '_y'], how='outer')
            latent_factor_pred_df['RET_PRED'] = latent_factor_pred_df['RET_PRED'] + latent_factor_pred_df['RET_PRED_y']
            latent_factor_pred_df = latent_factor_pred_df[['tradeDate', 'ticker', 'RET_PRED']]
        print('[TwoFactor][level {0}][step5] factor predicted. cost:{1:.2f}'.format(l, time.time()-t5))
        # print(latent_factor_pred_df.head(5))
    latent_factor_pred_df = pd.crosstab(index=latent_factor_pred_df['tradeDate'], columns=latent_factor_pred_df['ticker'], values=latent_factor_pred_df['RET_PRED'], aggfunc='sum')
    return latent_factor_pred_df

# 用PLS合成因子
factor_price_dic = {fd: pd for fd, pd in zip(factor_date, price_date)}
flag, latent_factor_dic = load_pickle('pls/pls_single_factor_two_factor.data', ori_obj=OrderedDict())
if not flag:
    t1 = time.time()
    for l in [3, 5]:
        for history_len in [1, 12, 24, 36]:
        # for history_len in [1, 12]:
            latent_factor_df = latent_factor_model(factors, eq_return_df, L=l, history_len=history_len)
            # print(latent_factor_df.head(5).to_html())
            latent_factor_dic[tuple([l, history_len])] = latent_factor_df
    save_pickle('pls/pls_single_factor_two_factor.data', latent_factor_dic)
    print('[TwoFactor] two factor model computed done. cost: {0:.2f}'.format(time.time()-t1))
else:
    print('[TwoFactor] two factor model loaded done.')

# 针对原始因子，统计long-short平均收益等
result_df = pd.DataFrame()
for l, history_len in latent_factor_dic:
    factor_df = latent_factor_dic[tuple([l, history_len])]
    series = pd.Series(factor_df.values.ravel()).dropna()
    stats_df = pd.DataFrame(series.describe()).T
    summary_df = signal_generate.get_analysis_summary(factor_df,eq_return_df)
    coverage, signal_coverage = signal_analysis.plot_signal_coverage(summary_df['factor_value'],300,plot=False)
    group_summary_df = signal_analysis.get_quantile_return_mean_plot(summary_df,5,plot=False)
    tmp_df = signal_analysis.get_long_short_return_stats(factor_df,eq_return_df,'long_short',benchmark='equal',quantile_num=5)
    tmp_df['latentVarNum'] = '{0:02d}'.format(l)
    tmp_df['historyLength'] = '{0:02d}'.format(history_len)
    tmp_df['oriSignalMean'] = stats_df['mean']
    tmp_df['oriSignalStd'] = stats_df['std']
    tmp_df['oriSignalMin'] = stats_df['min']
    tmp_df['oriSignalMax'] = stats_df['max']
    tmp_df['coverage'] = coverage
    result_df = result_df.append(tmp_df)
print(result_df.to_html())
latent_factor_dic

load_pickle('pls/pls_single_factor_two_factor.data',latent_factor_dic)


# 我们选取上述表现最好的合成信号，用quartz做回测
# start = '2007-05-31'                       # 回测起始时间
start = '2013-03-31'                       # 回测起始时间
end = '2018-07-29'                         # 回测结束时间
universe = DynamicUniverse('HS300')        # 证券池，支持股票和基金、期货
benchmark = 'HS300'                        # 策略参考基准
freq = 'd'                                 # 'd'表示使用日频率回测，'m'表示使用分钟频率回测
refresh_rate = Monthly(1)                   # 执行handle_data的时间间隔
commission = Commission(buycost=0.0015, sellcost=0.0015, unit='perValue')
# commission = Commission(buycost=0.0, sellcost=0.0, unit='perValue')
#max_holding = 60
max_holding = 15

# 3层隐变量，仅使用过去1个月的历史数据，预测PLS中的代理变量
factor_df = latent_factor_dic[tuple([3, 1])]
trade_cal = DataAPI.TradeCalGet(exchangeCD=u"XSHG",beginDate=u"",endDate=u"",field=u"",pandas="1")
trade_cal = trade_cal[trade_cal['isOpen'] == 1]
trade_cal = trade_cal.set_index('calendarDate', drop=False)

tickers_df = DataAPI.EquGet(equTypeCD=u"A",secID=u"",ticker=u"",listStatusCD=u"",field=u"",pandas="1")
tickers_dict = {rows['ticker']:rows['ticker']+'.'+rows['exchangeCD'] for _, rows in tickers_df.iterrows()}

accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=100000000, commission=commission)
}

def initialize(context):                   # 初始化策略运行环境
    pass

def handle_data(context):                  # 核心策略逻辑
    account = context.get_account('fantasy_account')
    cur_date = context.current_date.strftime('%Y-%m-%d')
    cur_date_no_hyphen = cur_date.replace('-', '')
    prev_date = trade_cal.loc[cur_date, 'prevTradeDate']
    
    cur_universe = context.get_universe('stock', exclude_halt=False)
    cur_not_halt_universe = context.get_universe('stock', exclude_halt=True)
    cur_universe = set([ticker[:6] for ticker in cur_universe])
    cur_not_halt_universe = set([ticker[:6] for ticker in cur_not_halt_universe])
    cur_halt_universe = cur_universe - cur_not_halt_universe
    #print cur_halt_universe
    
    # print(len(cur_universe), len(cur_halt_universe))
    # 该天没有因子值
    
    tickers_factor = factor_df.loc[prev_date][list(cur_not_halt_universe)].fillna(0.0).to_dict()
    tickers_factor = sorted(tickers_factor.items(), key=lambda x: (-x[1], x[0]), reverse=False)
    tmp_df = pd.DataFrame(data=tickers_factor[:max_holding], columns=['ticker','score'])
    tmp_df['datadate'] = cur_date_no_hyphen
    
    to_buy_tickers = [ticker for ticker, value in tickers_factor[:max_holding]]
    cur_positions = set([ticker[:6] for ticker in account.get_positions(exclude_halt=False)])
    
    could_not_sell = cur_halt_universe & cur_positions
    to_sell = [ticker for ticker in cur_positions if ticker not in cur_halt_universe and ticker not in to_buy_tickers]
    to_hold = [ticker for ticker in cur_positions if ticker in cur_halt_universe]
    #to_buy = [ticker for ticker in to_buy_tickers[:max_holding-len(to_hold)] if ticker not in cur_positions]
    to_buy = [ticker for ticker in to_buy_tickers[:max_holding-len(to_hold)] if (ticker not in cur_halt_universe and ticker not in cur_positions)]
    to_hold += [ticker for ticker in to_buy_tickers[:max_holding-len(to_hold)] if ticker in cur_positions]
    for ticker in to_sell:
        ticker = tickers_dict[ticker]
        account.order_to(ticker, 0.0)
    for ticker in to_buy:
        ticker = tickers_dict[ticker]
        account.order_pct_to(ticker, 1.0/max_holding)
        
        
