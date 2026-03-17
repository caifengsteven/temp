'''
only breakout no machine learning

'''

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from datetime import datetime
from datatime import timedelta
from stockstats import StockDataFrame as Sdf
import pandas_datareader.data as web
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralCoclustering as bicluster
from sklearn.preprocessing import MinMaxScaler
from backtester.result import load_result
import pysnooper
from datasets.ibes.data_source import *
from lager.datasources.arm import *

def rsi_index(s, n=5):
	gain = s.rolling(n).apply(lambda x:x[x>0].mean() if(x>0).sum()!=0 else 0)
	loss = s.rolling(n).apply(lambda x: -x[x<0].mean() if(x<0).sum()!=0 else 0)
	return gain/(gain+loss)

from algorithm.algorithm import Algorithm
from algorithm.api import (order, order_target, order_target_value, add_pipeline, get_alg_params, get_logger, get_current_date, get_trade_date, get_universe)
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import datetime

import math
import pytz
import pdb
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from datetime import date
logger = get_logger('Algorithm')

def wr_index(s_h, s_l, s_c, n=5):
	h_n = s_h.rolling(n).apply(lambda x:x.max())
	l_n = s_l.rolling(n).apply(lambda x:x.min())
	return (h_n -s_c)/(h_n - l_n)

def roc_index(s, n=5):
	return s/s.shift(n)-1

def ema_index(s, n=5):
	return s.ewm(n, min_periods=n).mean()

def atr_index(s_h, s_l, s_c, n=5):
	sc_shift = s_c.shift(1)
	tr = pd.concat([s_h-s_l, (s_h-sc_shift).abs(),(s_l-sc_shift).abs()], axis=1).max(axis=1)
	return tr.rolling(n).mean()
def adx_index(s_h, s_l, s_c, n=5):
	sc_shift = s_c.shift(1)
	tr = pd.concat([s_h-s_l,(s_h-sc_shift).abs(),(s_l-sc_shift).abs()],axis=1).max(axis=1)
	hd = s_h - s_h.shift(1)
	ld = s_l.shift(1)-s_l
	hd_index = hd[(hd<=0)|(hd<ld)].index
	hd.loc[hd_index]=0
	ld_index = ld[(ld<=0)|(ld<hd)].index
	ld.loc[ld_index]=0
	hdi = hd.rolling(n).sum()/tr
	ldi = ld.rolling(n).sum()/tr
	adx = (hdi-ldi).abs()/(hdi+ldi)
	return adx.ewm(span=n).mean()

def create_lagged_series_ret(stock_ts, lags=20):
	ts = pd.DataFrame(index = stock_ts.index)
	ts['Today'] = stock_ts['closePrice']
	ts['volume'] = stock_ts['volume']
	tslag = pd.DataFrame(index = stock_ts.index)
	tslag['Today'] = ts['Today']
	tslag['Volume'] = ts['volume']
	tsret = pd.DataFrame(index = tslag.index)
	for i in range(0, lags):
		tslag['Ahead%s'%str(i+1)]= stock_ts['closePrice'].shift(-(i+1))
		tsret = pd.DataFrame(index = tslag.index)
		tsret['Volume'] = tslag['Volume']
		tsret['Today'] = tslag['Today'].pct_change()*100
	for i,x in enumerate(tsret['Today']):
		if(abs(x)<0.001):
			tsret['Today'][i]=0.001
	for i in range(0, lags):
		tsret['Ahead%s'%str(i+1)] = tslag['Ahead%s'%str(i+1)].pct_change()*100
	return tsret

def create_lagged_series(stock_ts, lags=20):
    ts = pd.DataFrame(index=stock_ts.index)
    ts['Today'] = stock_ts['closePrice']
    ts['volume'] = stock_ts['volume']
    # create the new lagged DataFrame\n",
    tslag = pd.DataFrame(index=ts.index)
    tslag['Today'] = ts['Today']
    tslag['Volume'] = ts['volume']
    tsret = pd.DataFrame(index=tslag.index)
    # create the shifted lag series of prior trading period close values\n",
    for i in range(0, lags):
        tslag['Lag%s' % str(i+1)] = ts['Adj Close'].shift(i+1)
        
        tsret['Volume'] = tslag['Volume']
        tsret['Today'] = tslag['Today']

    for i,x in enumerate(tsret['Today']):
        if (abs(x) < 0.0001):
            tsret['Today'][i] = 0.0001
    # create the lagged percentage returns columns\n",
    for i in range(0, lags):
        tsret['Lag%s' % str(i+1)] =  tslag['Lag%s' % str(i+1)]

    return tsret

def create_lagged_series_rsi(stock_ts, lags=20):
	ts = pd.DataFrame(index = stock_ts.index)
	ts['Today'] = stock_ts['closePrice']
	ts['volume'] = stock_ts['volume']
    ts['rsi']=stock_ts['rsi_14']
    tslag = pd.DataFrame(index=ts.index)
    tslag['rsi'] = ts['rsi']
    tslag['Volume'] = ts['volume']
    tsret = pd.DataFrame(index= tslag.index)
    # create the shifted lag series of prior trading period close values\n",
    for i in range(0, lags):
        tslag['Lag%s' % str(i+1)] = ts['rsi'].shift(i+1)
        tsret['Volume'] = tslag['Volume']
        tsret['rsi'] = tslag['rsi']

    for i,x in enumerate(tsret['rsi']):
        if (abs(x) < 0.0001):
            tsret['rsi'][i] = 0.0001
    # create the lagged percentage returns columns\n",
    for i in range(0, lags):
        tsret['Lag%s' % str(i+1)] =  tslag['Lag%s' % str(i+1)]

    return tsret
def generate_alpha(data):
	print('----------generate alpha --------')
	main_data = data['main']
	start_test_date = np.datetime64('2011-01-01')

	df = main_data[['open_adj','close_adj', 'low_adj', 'high_adj', 'turnvoer']]
	df_alpha =df 
	idx = pd.IndexSlice
	start_test_date = df_alpha.index[0]
	end_test_date = df_alpha.index[-1]
	for stock in main_date.index.levels[1]:
		df = main_data[['open_adj','close_adj', 'low_adj', 'high_adj', 'turnvoer']]
		df = df.xs(stock, level = 1)
		df.columns = ['openPrice','closePrice','lowestPrice', 'highestPrice', 'volume']
		df['chgPct'] = df['closePrice'].pct_change()
		df['rsi_14'] = rsi_idex(df['chgPct'],14)
		'''
		Signal for positive driver

		'''
		snpret_close = create_lagged_series(df)
		snpret_close = snpret_close[1:]
		snpret_min15 = snpret_close.iloc[:,-16:].min(axis=1)
		snpret_min5 = snpret_close.iloc[:,-21:-16].min(axis=1)
		snpret_max15 = snpret_close.iloc[:,-16:].max(axis=1)
		snpret_max5 = snpret_close.iloc[:,-21:-16].max(axis=1)
		snpret_rsi = create_lagged_series_rsi(df)
		snpret_rsi = snpret_rsi[1:]
		snpret_rsimin15 = snpret_rsi.iloc[:,-16:].min(axis=1)
		snpret_rsimin5 = snpret_rsi.iloc[:,-21:-16].min(axis=1)
		snpret_rsimax15 = snpret_rsi.iloc[:,-16:].max(axis=1)
		snpret_rsimax5 = snpret_rsi.iloc[:,-21:-16].max(axis=1)
		snpret_rsi1 = snpret_rsi.iloc[:,-21]
		snpret_rsi2 = snpret_rsi.iloc[:,-20]
		condlist = [snpret_min5<snpret_min15, snpret_max5>snpret_max15]
		choicelist =[1,2]
		direction_price = np.select(constlist, choicelist)

		condlist= [snpret_rsimin5>snpret_rsimin15, snpret_rsimax5<snpret_rsimax15]
		direction_rsi = np.select(condlist, choicelist)
		condlist = [snpret_rsi1>snpret_rsi2, snpret_rsi1<snpret_rsi2]
		direction_rsilast = np.select(condlist, choicelist)

		signal = direction_price*direction_rsi*direction_rsilast

		condlist = [signal==1, signal==0, signal==2, signal==4, signal==8]
		choicelist = [1,0,0,0,-1]
		direction = np.select(condlist, choicelist)
		snpret_close['Direction']=np.sign(direction)
		snpret_close['min15']=snpret_min15
		snpret_close['max15']= snpret_max15
		snpret_ret = create_lagged_series_ret(df)
		snpret_ret = snpret_ret[1:]
		snpret_ret5 = snpret_ret.iloc[:,-21:-16].sum(axis=1)
		snpret_ret10 = snpret_ret.iloc[:,-21:-11].sum(axis=1)
		snpret_ret21 = snpret_ret.iloc[:,-21:-1].sum(axis=1)
		snpret_close['Ret5d'] = np.sign(snpret_ret5)
		snpret_close['Ret10d'] = np.sign(snpret_ret10)
		snpret_close['Ret20d'] = np.sign(snpret_ret20)
		snpret_close['FinalDirection5d'] = snpret_close['Direction']*snpret_close['Ret5d']
		snpret_close['FinalDirection10d'] = snpret_close['Direction']*snpret_close['Ret10d']
		snpret_close['FinalDirection20d'] = snpret_close['Direction']*snpret_close['Ret20d']
		snpret_close['symbol']=stock
		snpret_close = snpret_close.reset_index().set_index(['date','symbol'])
		df_alpha.loc[(slice(None), stock),'Direction'] = snpret_close['Direction']
		df_alpha.loc[(slice(None), stock),'min15'] = snpret_close['min15']
		df_alpha.loc[(slice(None), stock),'max15'] = snpret_close['max15']
	return df_alpha

def init(context):
	params = get_alg_params()
	print('-------------------init -------------------')
	add_pipeline('alpha',generate_alpha)
	context.only_start = True

def before_trading_start(context, data):
	if context.only_start:
		context.only_start = False
		universe = get_universe()
		for i, j in universe.stocks.items():
			context[i]={}
			context[i]['buysignal']=0
			context[i]['buylimit']=0
			context[i]['buystop']=0
			context[i]['sellsignal']=0
			context[i]['selllimit']=0
			context[i]['sellstop']=0
	return True

def handle_data(context, data):
	main_data = data.current()
	universe=get_universe()
	for i, j in universe.stocks.items():
		temp = main_data['close_adj']
		if(temp is None):
			continue
		if not (i in main_data.index):
			continue
		close = temp.loc[i]

		if main_data is None:
			logger.error('None main current')
			return True
		alpha_data = data.get_data('alpha').current()
		alpha_data = alpha_data.loc[i]

		existing_long = False
		existing_short = False
		if(j.pos>0):
			existing_long= True
		if (j.pos<0):
			existing_short = True
		if(alpha_data is not None):
			if not np.isnan(alpha_data['Direction']):
				Direction = int(alpha_data['Direction'])
			else:
				Direction = 0
			if not np.isnan(alpha_data['min15']):
				min15 = (alpha_data['min15'])
			else: 
				min15 = 0
			if not np.isnan(alpha_data['max15']):
				max15 = (alpha_data['max15'])
			else: 
				max15 = 0

		if (Direction==1):
			context[i]['buysignal']=1
			context[i]['buylimit']=max15
			context[i]['buystop']=min15
			context[i]['sellsignal'] = 0
		if (Direction==-1):
			context[i]['sellsignal']=1
			context[i]['selllimit']=min15
			context[i]['sellstop']=max15
			context[i]['buysignal'] = 0
		if((context[i]['buysignal']==1) and (not existing_long) and (close>context[i]['buylimit'])):
			order_target_value (i, 100000, algo='close')
			existing_long = True

		if((context[i]['buysignal']==1) and (xisting_long) and (close<context[i]['buystop'])):
			order_target_value (i, 0, algo='close')
			existing_long = False
			context[i]['buysignal']=0

		if((context[i]['sellsignal']==1) and (not existing_short) and (close<context[i]['selllimit'])):
			order_target_value (i, -100000, algo='close')
			existing_short = True
		if((context[i]['sellsignal']==1) and (existing_short) and (close>context[i]['sellstop'])):
			order_target_value (i, 0, algo='close')
			existing_short = False
			context[i]['sellsignal']=0

alg = Algorithm(init= init, handle_data=handle_data, before_trading_start= before_trading_start, name='StatArb2')
algo_params_default ={}

from strategy.strategy import Strategy
from utils.dotdict import merge
import os

class StatArbStrategy(Strategy):
	def __init__(self, universe, alg_params=None, cfg_params=None, data_params = None):
		cfg_params_default ={
		'micro':'Dammy',
		'run_pipelines':True,
		'load_pipeline':False,
		'log_path':'/home/chechen/cc/strategies/StatArbStrategy'
		} 

		data_params_default ={
		'start_date': '20060101',
		'datasets':{'main':'dataset.price.px'},
		'external_data':None
		}
		if cfg_params is not None:
			cfg_params = merge(cfg_params, cfg_params_default)
		else:
			cfg_params = cfg_params_default

		if alg_params is not None:
			alg_params = merge(alg_params, algo_params_default)
		else:
			alg_params = algo_params_default

		if data_params is not None:
			data_params = merge(data_params, data_params_default)
		else:
			data_params = data_params_default
		super().__init__(alg, universe = universe, alg_params= alg_params, cfg_params= cfg_params, data_params= data_params, intraday = False)

from backtester.backtester import backtest_strategy
from utils.dotdict import merge
from pandas.tseries.offsets import BDay
import datetime
import pandas as pd

def run_backtest(sim_params=None, cfg_params=None, data_params=None, alg_params=None, universe= None, start_date='20140101', end_date= None, result_id=None):
	sim_params_default ={
	'slippage_cost':None,
	'tcost': 0,
	'scale_factor':1,
	'update_cache': False,
	'disable_lookforward_bias_check': True,
	'include_oss_data':True,
	'strategy_type':'daily',
	'fill_mode':'partial',
	'fill_ratio': [0,0.8],
	'timezone': 'Asia/Hong_Kong',
	'load_pipeline':False,
	'max_exec':0.5,
	'enable_tax_cost':True

	}
	if sim_params is not None:
		sim_params_default.update(sim_params)

	strategy = StatArbStrategy(universe=universe, alg_params= alg_params, data_params= data_params)
	return backtest_strategy(strategy=strategy, sim_params=sim_params_default, start_date = start_date, result_id=result_id)


import sys
sys.path.append('N:/bi_paticle')

universe ={
	'stocks':{ 'indices':['HSI Index','HSCEI Index']},
	'currencies':{'symbols':['HKD Curncy']}

}

sim_params ={
	'update_cache': False,
	'load_pipeline': False,
	'enable_tax_cost':True
}

alg_params={
	'alpha':{}
}
data_params ={
	'update_cache':False
}

result = run_backtest(universe = universe, alg_params=alg_params, start_date='20110101', end_date='20180101', sim_params=sim_params, data_params=data_params, result_id='starmine_test_hk')
result.portfolio_data['pnl'].cumsum().plot()
result.plot()
result.portfolio_data.to_csv('r_s_breakout_hk.csv')


