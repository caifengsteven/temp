'''
TD Rei classifc and modified
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
import time
import matplotlib.patches as mpatches
import matplotlib as mlp

#经典TD幅度膨胀指标计算
def cal_tdrei(df,k,m,p):
    df = df.copy()
    df = df.sort_values('tradeDate')
    df['h_k'] = pd.rolling_max(df['highestIndex'].shift(1), k)
    df['l_k'] = pd.rolling_min(df['lowestIndex'].shift(1), k)
    df['h_m'] = pd.rolling_max(df['highestIndex'].shift(1), m)
    df['l_m'] = pd.rolling_min(df['lowestIndex'].shift(1), m)
    df['h_p'] = pd.rolling_max(df['highestIndex'].shift(1), p)
    df['l_p'] = pd.rolling_min(df['lowestIndex'].shift(1), p)
    
    def cal_X(hi,hk,li,lk,lm,hm):
        if hi>=lm and li<=hm:
            X = (hi - hk) + (li - lk)
        else:
            X = 0
        return X    
    df['X'] = df.apply(lambda x: cal_X(x['highestIndex'],x['h_k'],x['lowestIndex'],x['l_k'],x['l_m'],x['h_m']),axis=1)
    df['x_rollingsum'] = pd.rolling_sum(df['X'], p+1) 
    df['TDREI'] = (df['x_rollingsum'] / (df['h_p'] - df['l_p'])) * 100
    return df

#计算截至当前日期指标连续超过阈值或连续低于负的阈值的天数
def s_num(s):
    if s[0] ==0:
        s1 = [0]
    else:
        s1 = [1]
    for i in range(1, len(s)):
        if s[i]==0:
            s1.append(0)
        elif s[i] == s[i-1]:
            s1.append(s1[i-1]+1)
        else:
            s1.append(1)
    return s1

#计算截至当前日期连续上涨的天数
def rise_num(s1,s2):
    r_num = []
    for i in range(len(s1)):
        if s1[i] > 1:
            r_num.append(sum(s2[(i-s1[i]+1):(i+1)]))
        else:
            r_num.append(np.nan)
    return r_num    

#标记下单的日期
def get_order_day(s):
    r=[True]
    for i in range(1,len(s)):
        if s[i]==s[i-1]:
            r.append(False)
        else:
            r.append(True)
    if s[-1]==0  or s[-1]==s[-2]:
        r[-1] = True
    return r 

#经典TD膨胀度指标的交易策略
def classic_tdrei(df, k=6, m=2, theta=40):
    df['theta'] = df['TDREI'].apply(lambda x: 1 if x>=theta else -1 if x<-theta else 0)
    df['con_num'] = s_num(df['theta'].tolist())
    df['rise_down'] = df['CHGPct'].apply(lambda x: 0 if x<0 else 1)
    df['rise_num']  = rise_num(df['con_num'].tolist(),df['rise_down'].tolist())    
    df['down_num'] = df['con_num'] - df['rise_num']
    df['h_k'] = pd.rolling_max(df['highestIndex'],k)
    df['l_k'] = pd.rolling_min(df['lowestIndex'],k)
    df['h_k1'] = pd.rolling_max(df['highestIndex'].shift(1),k)
    df['l_k1'] = pd.rolling_min(df['lowestIndex'].shift(1),k)    
    df['nxt_can'] = df.apply(lambda x: 1 if x['openIndex']>=x['l_k1'] and x['closeIndex']<=x['openIndex'] and x['lowestIndex']<=x['l_k1'] and x['theta']==1 else -1 if x['openIndex']<=x['h_k1'] and x['closeIndex']>=x['openIndex'] and x['highestIndex']>=x['h_k1'] and x['theta']==-1 else 0,axis=1)
    df['pre_can'] = df.apply(lambda x: 1 if x['con_num']>=2 and x['rise_num']>=1 and x['theta']==1 else -1 if x['con_num']>=2 and x['down_num']>=1 and x['theta']==-1 else 0,axis=1)
    df['pre_can'] = df['pre_can'].shift(1)    
    df['order'] = df.apply(lambda x: -1 if x['pre_can']==1 and x['nxt_can']==1 else 1 if x['pre_can']==-1 and x['nxt_can']==-1 else 0,axis=1 )    
    df1 = df[df['order'].isin([-1,1])]
    df1 = df1.append(df.iloc[-1,:])
    df1 = df1.drop_duplicates()    
    df1['order_day'] = get_order_day(df1['order'].tolist())    
    df2 = df1[df1['order_day']==True][['date','closeIndex','order']]        
    return df2

def Modify_TD(df,theta):
    df = df.copy()
    df['order'] = df['TDREI'].apply(lambda x: 1 if x>=theta else -1 if x<=-theta else 0)
    df1 = df[df['order'].isin([-1,1])]
    df1 = df1.append(df.iloc[-1,:])
    df1 = df1.drop_duplicates()
    df1['order_day'] = get_order_day(df1['order'].tolist())
    df1 = df1[df1['order_day']==True][['date','closeIndex','order','TDREI']]
    return df1

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
	df_alpha['order'] = 0
	for stock in main_data.index.levels[1]:
		df = main_data[['open_adj','close_adj', 'low_adj', 'high_adj', 'turnover']]
		df = df.xs(stock, level=1)
		df.columns = ['openIndex', 'closeIndex', 'lowestIndex','highestIndex', 'volume']
		df['CHGPct'] = df['closeIndex'].pct_change()
		result = cal_tdrei(df,6,2,5)
		result = result.dropna()

		try:
			result_tdrei_df = Modify_TD(result)
			result_tdrei_df = result_tdrei_df.set_inde('date')
		except:
			continue
		mask = result_tdrei_df.index
		result_tdrei_df['symbol']=stock
		result_tdrei_df.reset_index().set_index('date','symbol')
		ran = pd.to_datetime(result_tdrei_df.index).date
		df_alpha.loc[(ran, stock),'order'] = result_tdrei_df.loc[:,'order'].values
	return df_alpha

def init(context):
	params = get_alg_params()
	print('--------------------init ----------------------')
	add_pipeline('alpha', generate_alpha)

def handle_data(context, data):
	main_data = data.current()
	if (main_data) is None:
		logger.error('None main current')
		return True
	universe = get_universe()
	for i, j in universe.stocks.items():
		alpha_data = data.get_data('alpha').current()
		try:
			alpha_data = alpha_data.loc[i]
		except:
			continue

		if (alpha_data is not None):
			if not np.isnan(alpha_data['order']):
				Direction = int(alpha_data['order'])
			else:
				Direction = 0
			if(Direction == 1):
				order_target_value(i, 1000000, algo='close')
			if(Direction == -1):
				order_target_value(i, -1000000, algo='close')


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
