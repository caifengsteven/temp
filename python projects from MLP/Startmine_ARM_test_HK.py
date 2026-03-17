import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoClustering as bicluster
from sklearn.preprocessing import MinMaxScaler
from backtester.result import load_result
import pysnooper
from datasets.ibes.data_source import *
from lager.datasources.arm import *

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
	print('----------------generate_alpha-------------')
	main_data = data['main']
	ds = ARMAsiaDatasource()
	df = main_data.index.values
	start_date = df[0][0]
	end_date = df[-1][0]
	df_today = ds.query(start_date = start_date, end_date = end_date, split='HK')
	df_today.drop('cusip', axis=1, inplace = True)
	df_today= df_today.dropna(axis=0)
	df_alpha = df_today[['date', 'ticker','arm_score_5']]
	df_alpha.set_index('date', inplace=True)
	return df_alpha
def init(context):
	params = get_alg_params()
	print('---------------init-------------')
	add_pipeline('alpha', generate_alpha)

def handle_data(context, data):
	main_data = data.current()
	universe = get_universe()
	if main_data is None:
		logger.error('None main current')
		return True
	alpha_data = data.get_data('alpha').current()
	long_list = alpha_data[alpha_data['arm_score_5']==1]
	long_ticker = (long_list['ticker'].apply(lambda x: x+' HK Equity'))
	long_ticker = long_ticker.tolist()
	short_list = alpha_data[alpha_data['arm_score_5']==5]
	short_ticker = (short_list['ticker'].apply(lambda x: x+' HK Equity'))
	short_ticker = short_ticker.tolist()

	existing_long_list = []
	existing_short_list = []
	for i, j in universe.stocks.items():
		if(j.pos>0):
			existing_long_list.append(i)
		if(j.pos<0):
			existing_short_list.append(i)
	for i in existing_long_list:
		if (i not in long_ticker):
			order_target_value(i,0, algo='open')
	for i in existing_short_list:
		if (i not in short_ticker):
			order_target_value(i,0, algo='open')

	for i in long_ticker:
		order_target_value(i, 100000, algo='open')

	for i in short_ticker:
		order_target_value(i, -100000, algo='open')

alg = Algorithm(init = init, handle_data = handle_data, name='StatArbs')
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

