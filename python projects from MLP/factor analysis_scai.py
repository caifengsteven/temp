from gkd.factors.factor_analysis import analyze_factors
from lager.core.recipe import USRIYRecipe
from lager.core.recipe import ALLAURecipe
import pandas as pd
import numpy as np
import logging
import sys
from backtester.plotting import plot_nice_table, plot_ts
from sklearn.cluster import SpectralCoclustering as bicluster
from sklearn.preprocessing import MinMaxScaler
from backtester.result import load_result
import pysnooper
import datetime

au_ds = ALLAURecipe(min_adv_mio =2, start_date = datetime.datetime(1995,6,30))
df_total = au_ds.query(start_date=pd.Timestamp('20060101'), end_date=pd.Timestamp('20181231'))

df_total['return'] = df_total['dlyreturn']/100
df_total.reset_index().set_index(['date','barrid'])
All = slice(None)
stock_universe = df_total.index[1]
for i in stock_universe:
	df_total.loc[(slice(None),i),'rolling_sum_ret_5'] = df_total.loc[(slice(None),i),'return'].rolling(5).sum()
	df_total.loc[(slice(None),i),'rolling_ret_std_5'] = df_total.loc[(slice(None),i),'return'].rolling(5).std()
	df_total.loc[(slice(None),i), 'ratio'] = df_total.loc[(slice(None),i), 'rolling_sum_ret_5']/df_total.loc[(slice(None),i), 'rolling_ret_std_5']

universe_condition = 'as51_index>0'
factors = ['ratio_5']
result_factors = analyze_factors(df_total, factos= factors, universe_condition = universe_condition, portfolio ='quantile',quantile_low =0.1, quantile_high=0.9, beta_neutral=False, beta_col='usfastd_beta', return_lag=2, return_col = 'return')
plot_nice_table(result_factors.performance_metrics, title = 'performance')
plot_ts(result_factors.daily_metrics['return'].cumsum())



