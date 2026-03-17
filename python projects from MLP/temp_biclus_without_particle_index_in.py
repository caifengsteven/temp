import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering as bicluster
from sklearn.preprocessing import MinMaxScaler
from backtester.result import load_result
import pysnooper

def refine_adjfactor(df):
    df_new = df.copy()
    df_new.set_index('tradeDate', inplace= True)
    define_cols = [col for col in df_new.columns if 'Price' in col]
    for col in define_cols:
        df_new[col] = df_new[col] * df_new['accumAdjFactor']
        df_new[col] = df_new[col].replace(0, np.nan).fillna(method = 'ffill')
    return df_new

def mean_index(s, n = 5):
    return s.rolling(n).mean()

def rsi_index(s, n = 5):
    gain = s.rolling(n).apply(lambda x: x[x > 0].mean() if (x > 0).sum() != 0 else 0)
    loss = s.rolling(n).apply(lambda x: - x[x < 0].mean() if (x < 0).sum() != 0 else 0)
    return gain / (gain + loss)

def wr_index(s_h, s_l, s_c, n = 5):
    h_n = s_h.rolling(n).apply(lambda x: x.max())
    l_n = s_l.rolling(n).apply(lambda x: x.min())
    return (h_n - s_c) / (h_n - l_n)

def roc_index(s, n= 5):
    return s / s.shift(n) - 1

def ema_index(s, n=5):
    return s.ewm(n, min_periods= n).mean()

def atr_index(s_h, s_l, s_c, n=5):
    sc_shift = s_c.shift(1)
    tr = pd.concat([s_h - s_l, (s_h - sc_shift ).abs(), (s_l - sc_shift ).abs()], axis= 1).max(axis = 1)
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

def label_func(x, thr = 0.005):
    if x> thr:
        l = 1
    elif x < -thr:
        l = -1
    else:
        l = 0
    return l

def dist(x,y):
	return np.sqrt(np.square(x-y)).mean()

def dist_2(x,y):
	return np.linalg.norm(x-y)

def knn_detect(s, df , thr = 0.2):
    clu_vec = [s[cols] for cols in df['cols'].values]
    sim = np.array([dist2(i,j) for i,j in zip(clu_vec, res_df['col_vec'])])
    # return sim
    return (np.array([1 if i > 0 else -1 for i in res_df['row_score'].values])[sim < thr]).sum()

def knn_get_thr_each(s, df):
    clu_vec = [s.loc[cols] for cols in df['cols'].values]
    sim = np.array([dist2(i,j) for i,j in zip(clu_vec, res_df['col_vec'])])
    return sim

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

def generate_alpha(data):
	print('----------generate alpha --------')
	main_data = data['main']
	df = main_data[['open_adj','close_adj', 'low_adj', 'high_adj', 'turnover']]
	idx = pd.IndexSlice
	df_alpha = df
	start_test_date = df_alpha.index[0]
	end_test_date = df_alpha.index[-1]
	df_alpha['Direction'] =0
	for stock in main_data.index.levels[1]:
		df = main_data[['open_adj','close_adj', 'low_adj', 'high_adj', 'turnvoer']]
		df = df.xs(stock, level = 1)
		df.columns = ['openPrice','closePrice','lowestPrice', 'highestPrice', 'volume']
		df['chgPct'] = df['closePrice'].pct_change()
		df_factor = pd.DataFrame(index = df.index)
		for n in [12, 24, 36]:
		    df_factor['roc_' + str(n)] = roc_index(df['closePrice'], n= n)
		
		# ema
		for n in [12, 24, 36]:
	    	df_factor['ema_' + str(n)] = ema_index(df['closePrice'], n= n)
	    	
		# adx
		for n in [14, 28, 42]:
	    	df_factor['adx_' + str(n)] = adx_index(df['highestPrice'], df['lowestPrice'],df['closePrice'], n= n)
	    	
		# atr
		for n in [14, 28, 42]:
	    	df_factor['atr_' + str(n)] = atr_index(df['highestPrice'], df['lowestPrice'], df['closePrice'], n= n)
	    	
		# sma
		for n in [10, 20, 30, 40]:
	    	df_factor['sma_' + str(n)] = mean_index(df['closePrice'], n= n)
	    	
		# wr
		for n in [9, 18, 27, 36]:
	    	df_factor['wr_' + str(n)] = wr_index(df['highestPrice'], df['lowestPrice'], df['closePrice'], n= n)
	    	
		# rsi
		for n in [6, 12, 18, 24, 30, 36]:
	    	df_factor['rsi_' + str(n)] = rsi_index(df['chgPct'], n= n)
	    	
	    df_factor.dropna(inplace= True)
	    std_model = MinMaxScaler()
	    x_scale = std_model.fit_transform(df_factor)
	    df_scale = pd.DataFrame(x_scale, columns = df_factor.columns, index = df_factor.index)
	    frv = df_factor['sma_10']/df['closePrice'] -1
	    df_factor['laabel'] = frv.apply(lambda x:label_func(x, thr=0.005))
	    model = bicluster(n_clusters=5, random_state=0)
	    model.fit(x_scale)

	    df_factor['clus']= model.row_labels_

	    set_b = df_factor.groupby('clus')['label'].apply(lambda x: np.array([x == 1]).sum() / np.float(x.shape[0]))
		set_h = df_factor.groupby('clus')['label'].apply(lambda x: np.array([x == 0]).sum() / np.float(x.shape[0]))
		set_s = df_factor.groupby('clus')['label'].apply(lambda x: np.array([x == -1]).sum() / np.float(x.shape[0]))
		set_bhs = pd.concat([set_b, set_h, set_s], axis = 1)
		set_bhs.columns = ['b', 'h', 's']
		sup_dict = set_bhs.apply(lambda x: x.argmax(), axis = 1).to_dict()
		dec_df = df_factor['clus'].map(sup_dict).map({'s':-1, 'b':1, 'h':np.nan})
		flag= 0
		sig_value = dec_df.fillna(method= 'ffill').bfill().values
		for i, v in enumerate(sig_value):
    		if v == flag:
        		sig_value[i] = 0
    		elif v != flag:
        		flag = v
        s = np.arange(df_scale.shape[0])[sig_value > 0]
		b = np.arange(df_scale.shape[0])[sig_value < 0]
		print(len(s), len(b))

		signal_df = pd.DataFrame(dec_df.fillna(method='ffill').apply(lambda x:0 if x>0 else 1))
		ran = signal_df.index
		df_alpha.loc[(ran, stock),'Direction']=signal_df.shift(1).values
	return df_alpha

def init(context):
	params = get_alg_params()
	print('---------------------init-------------------')
	add_pipeline('alpha', generate_alpha)

def handle_data(context, data):
	main_data = data.current()
	universe = get_universe()
	if main_data is None:
		logger.error('None main current')
		return True
	if i, j in universe.stocks.items():
		alpha_data = data.get_data('alpha').current()
		try:
			alpha_data = alpha_data.loc[i]
		except:
			continue
		if isinstance(alpha_data, pd.Series):
			indicator = alpha_data['Direction']
			if(indicator==1):
				order_target_value(i, 1000000, algo='open')
			if(indicator ==0):
				order_target_value(i, 0, algo='open')

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
	'stocks':{ 'indices':['NIFTY Index']},
	'currencies':{'symbols':['INR Curncy']}

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
result.portfolio_data.to_csv('temp_biclus_without_particle_index.csv')