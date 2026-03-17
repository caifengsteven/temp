from gkd.factors.factor_analysis import analyze_factors
from gkd.factors.factor_analysis import normalize_factors

import pandas as pd
import numpy as np
import logging
import sys
from backtester.plotting import plot_nice_table, plot_ts
from dtasets.starmine.data_source import *
from dataset.price import px
from lager.datasource import arm
import os
from operators.common import ema, rolling_std, expanding_std, expanding_mean, sum

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

file_name = 'japan_0519_data_v4.h5'
os.chdir('../data/')
df = pd.read_hdf(file_name, key='data')


from tpot import TPOTClassifier
import h2o
import pandas as pd
import numpy as np
from operators.common import ema, rolling_std, expanding_mean, expanding_std, rolling_sum
from backtester.plotting import plot_ts
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotencoder, LabelEncoder
import matplotlib.pyplot as plt

from gkd.models import model as all_model
from importlib import reload
from backtester.result import load_result
from backtester.plotting import plot_return_pnl, plot_nice_table
import pyfolio as pf
import copy
from operators.common import sma, ema


allgroup = ['ase2d_jpbeta','ase2d_jpbtop', 'ase2d_jpdivyild','ase2d_jpearnyild','ase2d_jpgrowth', 'ase2d_jpleverage', 'ase2d_jpliquidty', 'ase2d_jpmidcap','ase2d_jpmomentum', 'ase2d_jpresvol', 'ase2d_jpsize',\
'blev','btop','cetoe','cetop', 'cetoe','cins','cmra','css','dastd','dimv','dips','dns', 'dps','dsbeta','dtoa','egro','epibs','ess', 'etop','hbeta','histbeta','hsigma','indmom','lncap','midcap','mlev','predbeta',\
'rstr','season','sgro', 'specificreturn', 'specrisk','stoa','stom','stoq', 'totalrisk','yield','analyst_revision_score','arm_global_rank','arm_preferred_earnings_component','arm_revenue_component','arm_score_5',\
'arm_secondary_earning_component', 'price_mo_region_rank','price_mo_global_rank','price_mo_long_term_component', 'price_mo_mid_term_component','price_mo_short_term_component','price_mo_industry_component',\
'valmo_region_rank','valmo_global_rank','valmo_sector_rank','valmo_industry_rank', 'smartestimate_fy1_eps', 'predicted_surprise_pct_fy1_eps','smartestimate_fy2_eps', 'predicted_surprise_pct_fy2_eps', \
'smartestimate_fq1_eps', 'predicted_surprise_pct_fq1_eps','smartestimate_fq2_eps', 'predicted_surprise_pct_fq2_eps','smartestimate_f12m_eps', 'predicted_surprise_pct_f12m_eps','ase2d_jpairmarne_ret', \
'ase2d_jpautocomp_ret', 'ase2d_jpbanks_ret', 'ase2d_jpbuild_ret','ase2d_jpcapgoods_ret', 'ase2d_jpchemical_ret','ase2d_jpcnstrp_ret','ase2d_jpcomputer_ret','ase2d_jpcondur_ret', 'ase2d_jpconsrv_ret',\
'ase2d_jpconstap_ret', 'ase2d_jpdivfinan_ret','ase2d_jpelectron_ret', 'ase2d_jpenergy_ret', 'ase2d_jphealth_ret', 'ase2d_jpindtele_ret','ase2d_machine_ret','ase2d_mediaret_ret','ase2d_jpmetal',\
'ase2d_jprealest_ret', 'ase2d_jpsoftware_ret', 'ase2d_jpsteel_ret','ase2d_jputility_ret', 'ase2d_jpbeta_ret','ase2d_jpbtop_ret', 'ase2d_jpdivyild_ret', 'ase2d_jpgrowth_ret','ase2d_jpleverage_ret','ase2d_jpliquidty'\
'ase2d_jpmidcap_ret', 'ase2d_jpmomentum_ret', 'ase2d_jpresvol_ret', 'ase2d_jpsize_ret', 'abs_diff_1','ase2d_jpbeta_diff_1','ase2d_jpbtop_diff_1','ase2d_jpdivyild_diff_1', 'ase2d_jpearnyild_diff_1','ase2d_jpgrowth_diff_1'\
'ase2d_jpleverage_diff_1','ase2d_jpliquidty_diff_1', 'ase2d_jpmidcap_diff_1','ase2d_jpmomentum_diff_1', 'ase2d_jpresvol_diff_1', 'ase2d_jpsize_diff_1','blev_diff_1','btop_diff_1','capt_diff_1', 'cetoe_diff_1',\
'cetop_diff_1', 'cetoe_diff_1','cins_diff_1','cmra_diff_1','css_diff_1','dastd_diff_1','dimv_diff_1','dips_diff_1','dns_diff_1', 'dps_diff_1','dsbeta_diff_1','dtoa_diff_1','egro_diff_1','epibs_diff_1',\
'ess_diff_1', 'etop_diff_1','hbeta_diff_1','histbeta_diff_1','hsigma_diff_1','indmom_diff_1','lncap_diff_1','midcap_diff_1','mlev_diff_1','predbeta_diff_1','rstr_diff_1','season_diff_1','sgro_diff_1',\
'specificreturn_diff_1', 'specrisk_diff_1','stoa_diff_1','stom_diff_1','stoq_diff_1', 'totalrisk_diff_1','yield_diff_1', 'return_diff_1','abs_diff_5','ase2d_jpbeta_diff_5','ase2d_jpbtop_diff_5',\
'ase2d_jpdivyild_diff_5', 'ase2d_jpearnyild_diff_5','ase2d_jpgrowth_diff_5','ase2d_jpleverage_diff_5','ase2d_jpliquidty_diff_5', 'ase2d_jpmidcap_diff_5','ase2d_jpmomentum_diff_5', 'ase2d_jpresvol_diff_5', \
'ase2d_jpsize_diff_5','blev_diff_5','btop_diff_5','capt_diff_5', 'cetoe_diff_5','cetop_diff_5', 'cetoe_diff_5','cins_diff_5','cmra_diff_5','css_diff_5','dastd_diff_5','dimv_diff_5','dips_diff_5','dns_diff_5',\
'dps_diff_5','dsbeta_diff_5','dtoa_diff_5','egro_diff_5','epibs_diff_5','ess_diff_5', 'etop_diff_5','hbeta_diff_5','histbeta_diff_5','hsigma_diff_5','indmom_diff_5','lncap_diff_5','midcap_diff_5','mlev_diff_5',\
'predbeta_diff_5','rstr_diff_5','season_diff_5','sgro_diff_5','specificreturn_diff_5', 'specrisk_diff_5','stoa_diff_5','stom_diff_5','stoq_diff_5', 'totalrisk_diff_5','yield_diff_5', 'return_diff_5']


print(len(allgroup))



df['vwap_day_price'] = df['vwap_day']/df['price']
df['high_day_price'] = df['high_day']/df['price']
df['low_day_price'] = df['low_day']/df['price']
df['last_day_price'] = df['last_day']/df['price']

df[allgroup] = df[allgroup].replace([np.inf, -np.inf],np.nan)
df[allgroup] = df[allgroup].fillna(0)

for target in ['target_1', 'target_2','target_3','target_5','target_10','target_15','target_20']:
	df[target] = df[target].replace([np.inf, -np.inf],np.nan)
	df[target] = df[target].fillna(0)

from gkd.models import rolling_model

reload(all_model)
reload(rolling_model)
train_start_date = '20120101'
targets=['target_1', 'target_2','target_3','target_5','target_10','target_15','target_20']
df_pnl = pd.DataFrame(index = df.index)
df_weight = pd.DataFrame(index = df.index)

for target in targets:
	rm = rolling_model.RollingModel()
	rm.add_model_factory('xgb', lambda: all_model.XGBRegressor(max_depth=5, learning_rate =0.3, verbosity=1 , n_jobs=-1))
	rm.train(df, x_cols=allgroup, y_col=target, start_date=train_start_date, period=60, win=800, method='expanding', min_rows=10000)
	st = target.replace('_','0')[-2:]
	offset= int(st)+1
	y_pred = rm.predict(df, x_cols=allgroup, lag = offset)
	y_pred_mean = y_pred.groupby(level=0).mean().reindex(y_pred.index, level=0)
	y_pred_new = y_pred-y_pred_mean
	y_pred_sum = y_pred_new.abs().groupby(level=0).sum().reindex(y_pred.index, level=0)
	y_pred_weight = (y_pred_new)/y_pred_sum
	for col in y_pred_weight:
		df_weight[target+'_'+col+'_'+str(60)] = y_pred_weight[col]

filename = "allgroup_393features_xgb_depth5_export.h5"
os.chdir('../data/')
rm.save(filename)

y_pred_test = rm._model_df.loc['20190723']['xgb'].predict(df[group1])
y_pred_test_2016 = rm._model_df.loc['20160807']['xgb'].predict(df[group1])

df_weight['y_pred_test'] = y_pred_test
df_weight['y_pred_test_2016'] = y_pred_test_2016
df_pnl = sma (df_weight[['y_pred_test','y_pred_test_2016']],window=1, min_periods=1).mul(df['returnshift_2'],axis=0)
df_pnl_day = df_pnl.groupby(level=0).sum()
plot_ts(df_pnl_day['20120101':].cumsum())
print(df_pnl_day['20120101':].mean()/df_pnl_day['20120101':].std()*15.5)


# next is to take this y_pred and use it as input for optimizer
# I still don't know how this will be used for backtester though
# so the key question 
# 1) what is alpha to be sent to backtest algo in handle data
# 2) if you use y_pred as target return for optimizer you get your constraints optimized trade orders. How do you back test that

