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
    tr = pd.concat([s_h - s_l, (s_h - sc_shift).abs(), (s_l - sc_shift).abs()], axis=1).max(axis=1)
    ## 计算hd与ld
    hd = s_h - s_h.shift(1)
    ld = s_l.shift(1) - s_l
    hd_index = hd[(hd <= 0) | (hd < ld)].index
    hd.loc[hd_index] = 0
    ld_index = ld[(ld <= 0) | (ld < hd)].index
    ld.loc[ld_index] = 0

    ## 计算hdi与ldi
    hdi = hd.rolling(n).sum() / tr
    ldi = ld.rolling(n).sum() / tr

    adx = (hdi - ldi).abs() / (hdi + ldi)
    return adx.ewm(span = n).mean()

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

	df = main_data[['open_adj','close_adj', 'low_adj', 'high_adj', 'turnover']]
	df = df.xs('510050 CH Equity', level=1)
	df.columns  = ['openPrice','closePrice','lowestPrice', 'highestPirce', 'volume']
	df['chgPct'] = df['closePrice'].pct_change()
	df['rsi_14'] = rsi_index(df['chgPct'],14)
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
	k = 0
	plt.plot(df['closePrice'].values)
	for i in snpret_close['Direction']:
		if(i==1):
			plt.plot(k,df['closePrice'].iloc[k],'r*')
		if(i==-1):
			plt.plot(k, df['closePirce'].iloc[k], 'y*')
		k+=1
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
	snpret_close['20ma'] = snpret_close['Today'].rolling(window=20, min_periods=0).mean()
	snpret_close['40ma'] = snpret_close['Today'].rolling(window=40, min_periods=0).mean()
	snpret_close['60ma'] = snpret_close['Today'].rolling(window=60, min_periods=0).mean()
	snpret_close['120ma'] = snpret_close['Today'].rolling(window=120, min_periods=0).mean()

	feature = ['20ma','40ma','60ma','120ma']
	# roc
	for n in [12, 24, 36]:
	    snpret_close['roc_' + str(n)] = roc_index(df['closePrice'], n= n)
	    feature.append('roc_' + str(n))
	# ema
	for n in [12, 24, 36]:
	    snpret_close['ema_' + str(n)] = ema_index(df['closePrice'], n= n)
	    feature.append('ema_' + str(n))
	# adx
	for n in [14, 28, 42]:
	    snpret_close['adx_' + str(n)] = adx_index(df['highestPrice'], df['lowestPrice'],df['closePrice'], n= n)
	    feature.append('adx_' + str(n))
	# atr
	for n in [14, 28, 42]:
	    snpret_close['atr_' + str(n)] = atr_index(df['highestPrice'], df['lowestPrice'], df['closePrice'], n= n)
	    feature.append('atr_' + str(n))
	# sma
	for n in [10, 20, 30, 40]:
	    snpret_close['sma_' + str(n)] = mean_index(df['closePrice'], n= n)
	    feature.append('sma_' + str(n))
	# wr
	for n in [9, 18, 27, 36]:
	    snpret_close['wr_' + str(n)] = wr_index(df['highestPrice'], df['lowestPrice'], df['closePrice'], n= n)
	    feature.append('wr_' + str(n))
	# rsi
	for n in [6, 12, 18, 24, 30, 36]:
	    snpret_close['rsi_' + str(n)] = rsi_index(df['chgPct'], n= n)
	    feature.append('rsi_' + str(n))

	snpret_close['Prediction'] =0
	std_model = MinMaxScaler()
	x_scale = std_model.fit_transform(snpret_close)
	snpret_close_scale = pd.DataFrame(x_scale, columns= snpret_close.columns, index = snpret_close.index)
	snpret_close_scale['FinalDirection5d'] = snpret_close['FinalDirection5d']
	snpret_close_scale['FinalDirection10d'] = snpret_close['FinalDirection10d']
	snpret_close_scale['FinalDirection20d'] = snpret_close['FinalDirection20d']
	snpret_close_scale.replace([np.inf, -np.inf], np.nan)
	snpret_close_scale.dropna(inplace = True)
	snpret_close_cp = snpret_close.copy()
	for i in range (8):
		idx = pd.IndexSlice
		loop_date = start_test_date +np.timedelta(365*i,'D')
		loop_end_date = loop_date+np.timedelta(365,'D')
		print(loop_date, loop_end_date)
		snpret_close_test = snpret_close_scale.loc[snpret_close.index <loop_date]
		snpret_close_test.dropna(inplace = True)
		X = snpret_close_test[feature]
		Y = snpret_close_test['FinalDirection5d']
		X_train = X
		y_train = Y
		X_test = snpret_close_scale.loc[(snpret_close.index < loop_end_date)]
		X_test = X_test.loc[X_test.index>loop_date]
		X_test = X_test[feature]
		models =[('RSVM', SVC(C=1000000.0, cache_size = 200, class_weight={1:10}, coef0 =0.0, degree =3, gamma = 0.0001, kernel ='linear', max_iter=-1, probability = False, random_state=None, shrinking=True, tol = 0.001,verbose= False))]
		for m in models:
			m[1].fit(X_train, y_train)
			pred = m[1].predict(X_test)
		pred_df = pd.DataFrame(index = X_test.index)
		pred_df['Prediction'] = pred
		snpret_close.loc[idx[loop_date:loop_end_date],'Prediction'] = pred_df

	df_alpha = pd.DataFrame(index = snpret_close.index)
	df_alpha['Direction'] = snpret_close['Direction']
	df_alpha['Prediction']= snpret_close['Prediction']

	return df_alpha

def init(context):
	params = get_alg_params()
	print('---------------------init-------------------')
	add_pipeline('alpha', generate_alpha)
	context.long_day = 0
	context.short_day = 0

def handle_data(context, data):
	main_data = data.current()
	if main_data is None:
		logger.error('None main current')
		return True
	alpha_data = data.get_data('alpha').current()

	if(alpha_data is not None):
		if not np.isnan(alpha_data['Direction']):
			Direction = int(alpha_data['Direction'])
		else:
			Direction = 0
		if not np.isnan(alpha_data['Prediction']):
			Prediction = int(alpha_data['Prediction'])
		else:
			Prediction = 0
		if(context.long_day>0):
			context.long_day +=1
		if(context.short_day>0):
			context.short_day +=1
		if(context.long_day==5):
			order_target_value('511050 CH Equity',0 , algo= 'close')
			print('unwinding buy')
			context.long_day=0
		if(context.short_day==5):
			order_target_value('511050 CH Equity',0 , algo= 'close')
			print('unwinding sell')
			context.short_day=0
		if(Direction == 1 and Prediction ==1):
			order_target_value('511050 CH Equity',1000000, algo='close')
			print('buying')
			context.long_day =1
		if(Direction == -1 and Prediction ==-1):
			order_target_value('511050 CH Equity',-1000000, algo='close')
			print('selling')
			context.short_day =1


alg = Algorithm(init = init,handle_data = handle_data, name = "StatArb2")
alg_params_default ={}

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
	'stocks':{ 'symbols':['510050 CH Equity']},
	'currencies':{'symbols':['CNY Curncy']}

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
result.portfolio_data.to_csv('temp_r_s.csv')

