import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering as bicluster
from sklearn.preprocessing import MinMaxScaler
from backtester.result import load_result
import pysnooper
from scipy import signal
import statsmodels.api as sm
import math

from scipy.ndimage.interpolation import shift

def get_points(n, index_down, index_up, s_ori ,back_w = 1):
    print('s_ori',s_ori)
    qsdu_e = index_up[n]
    point_b, point_d = index_down[n - 3 :n - 1]
    point_a, point_c = index_up[n - 3 :n - 1]
    point_a, point_c = s_ori.loc[point_a -back_w - 1 : point_a + back_w].argmax(), s_ori.loc[point_c -back_w - 1: point_c + back_w].argmax() 
    point_b, point_d = s_ori.loc[point_b -back_w - 1: point_b + back_w].argmin(), s_ori.loc[point_d -back_w - 1: point_d + back_w].argmin()
    qsdu_e = s_ori.loc[qsdu_e - back_w - 1: qsdu_e + back_w].argmax()
    pre_a = index_down[n - 4]
    return point_a, point_b, point_c, point_d, qsdu_e

pre_day = 60
def is_bottom(a, b, c, d, e, s_ori, s_high, s_vol):
    ## 前期30% 涨幅
    ## 给定一个时间，在时间内与最低点差30%即可
    pre_day = a - 30
    r_pre = s_ori[a] / s_ori[pre_day: a].min() - 1
    cond1 = r_pre > 0.3
    ## a点高于c点
    cond2 = s_high[a] > s_high[c]
    ## d低于b点
    cond3 = s_ori[b] > s_ori[d]
    ## ad之间的限制
    cond5 = s_ori[a] / s_ori[d] - 1 < 0.5
    ## 判断是否出现突破
    ## 其中e点为伪e
    cond4 = s_ori[e] > s_ori[c]
    if cond2 and cond3 and cond4 and cond5:
        ## 判断e点
        ## 最高点
        e_cond1 = s_high.loc[d:e] > s_high[c]
        e_supply = np.arange(d, e+1)[e_cond1.values]
        # 判断成交量
        # 成交量判断无效
        for i in e_supply:
            if s_vol[i] > s_vol.loc[c:i].mean()*1.2:
                e = i
                return [a, b, c, d], e
                break

def get_stock_signal(df , window = 5, is_plot = True):
    ### 数据获取
    s_ori = df['closePrice'] 
    s_high = df['highestPrice'] 
    s_vol = df['volume']
    ## 去噪过程
    trend = s_ori.rolling(window).mean()
    back_w = int(window / 2.0)
    # trend.plot()
    ## 获取去噪之后的高低点
    index_down = signal.find_peaks(-trend)[0]
    index_up = signal.find_peaks(trend)[0]
    index_down = index_down[index_down > index_up[0]]
    ## 获取信号
    e_list = []
    lenth_w = []
    for n in range(5, index_up.shape[0]):
        a,b,c,d,qsdu_e = get_points(n, index_down, index_up, s_ori, back_w)
        if is_bottom(a, b, c, d, qsdu_e, s_ori, s_high, s_vol):
            [a, b ,c, d] , e = is_bottom(a, b, c, d, qsdu_e, s_ori, s_high, s_vol)
            e_list.append(e)
            lenth_w.append(e - a)
    if is_plot:
        s_ori.plot()
        plt.plot(e_list, s_ori[e_list], 'r*')
    return np.array(lenth_w), np.array(e_list)

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
	df = main_data[['open_adj','close_adj', 'low_adj', 'high_adj', 'turnover','volume','eqy_sh_out_real','cur_mkt_cap']]
	df.columns = ['openPrice', 'closePrice','lowestPrice', 'highestPrice','turnover', 'volume', 'eqy_sh_out_real', 'cur']
	df['chgPct'] = df['closePrice'].pct_change()
	df_alpha = df
	df_alpha['Direction'] = 0
	all_dfs = []
	back_dfs = []
	flag = 0
	for stock in main_data.index.levels[1]:
		df = main_data[['open_adj','close_adj', 'low_adj', 'high_adj', 'turnover','volume','eqy_sh_out_real','cur_mkt_cap']]
		df = df.xs(stock, level =1)
		df.columns = ['openPrice', 'closePrice','lowestPrice', 'highestPrice','turnover', 'volume', 'eqy_sh_out_real', 'cur']
		df['chgPct'] = df['closePrice'].pct_change()
		df = df.reset_index()
		try:
			length_w, e_list = get_stock_signal(df, window=3, is_plot=False)
		except:
			continue
		holding_day =5
		e_signal = np.hstack([e_list+i for i in range(1, holding_day+1)])
		e_signal = np.unique(e_signal)
		e_signal = e_signal[e_signal<df.shape[0]]
		e_signal.sort()
		if(len(e_signal)>0):
			v = np.zeros(df.shape[0])
			v[e_signal] =1
			df_alpha.loc[(slice(None), stock), 'Direction'] = shift(v, 1, cval=0)
	return df_alpha

def init(context):
	params = get_alg_params()
	print('--------------------init----------------------')
	add_pipeline('alpha',generate_alpha)

def handle_data(context, data):
	main_data = data.current()
	if main_data is None:
		logger.error('None main current')
		return True
	universe = get_universe()
	for i, j in universe.stocks.items()
		try:
			alpha_data = data.get_data('alpha').current()
			if(alpha_data is not None):
				if not np.isnan(alpha_data['Direction']):
					Direction= int(alpha_data['Direction'])
				else:
					Direction = 0
				if (Direction==1):
					order_target_value(i, 100000, algo='close')
				if(Direction == 0)
					order_target_value(i,0, algo='close')
		except:
			continue

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


