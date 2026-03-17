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

file_name = 'aud_0519_data_v4.h5'
os.chdir('../data/')
df = pd.read_hdf(file_name, key='r19')
allgroup = ['return','vwap_day_price','low_day_price','high_day_price', 'last_day_price', 'ase2d_beta','ase2d_dsbeta', 'ase2d_indmom','ase2d_liquidty','ase2d_midcap', 'ase2d_momentum', 'ase2d_size', 'ase2d_strevrsl',\
'capt', 'dsbeta', 'hbeta','histbeta','index','indmom', 'lncap','midcap', 'predbeta', 'as51_index_tsnorm', 'spx_index_tsnorm', 'ase2d_beta_tsnorm', 'ase2d_dsbeta_tsnorm', 'ase2d_indmom_tsnorm', 'ase2d_liquidty_tsnorm', 'ase2d_midcap_tsnorm'\
'ase2d_momentum_tsnorm', 'ase2d_size_tsnorm', 'ase2d_strevrsl_tsnorm', 'capt_tsnorm', 'dsbeta_tsnorm','hbeta_tsnorm', 'histbeta_tsnorm', 'index_tsnorm','indmom_tsnorm', 'lncap_tsnorm', 'midcap_tsnorm', 'predbeta_tsnorm',\
'predbeta_tsnorm', 'ase2d_airmarne_ret', 'ase2d_autocomp_ret', 'ase2d_banks_ret', 'ase2d_build_ret','ase2d_capgoods_ret', 'ase2d_chemical_ret','ase2d_cnstrp_ret','ase2d_comtra_ret', 'ase2d_condur_ret', 'ase2d_consrv_ret',\
'ase2d_constap_ret', 'ase2d_divfinan_ret', 'ase2d_energy_ret','ase2d_gold_ret', 'ase2d_health_ret', 'ase2d_insuran_ret', 'ase2d_mediaret_ret', 'ase2d_metmin_ret','ase2d_realest_ret', 'ase2d_software_ret', 'ase2d_steel_ret',\
'ase2d_telecomm_ret', 'ase2d_utility_ret', 'ase2d_bet_ret', 'ase2d_dsbeta_ret', 'ase2d_indmom_ret','ase2d_liquidty_ret', 'ase2d_midcap_ret', 'ase2d_momentum_ret', 'ase2d_size_ret','ase2d_strevrsl_ret',\
 ##### group 2
 
 'ase2d_btop', 'ase2d_divyild','ase2d_earnyild','ase2d_growth', 'ase2d_leverage', 'ase2d_oilsen', 'ase2d_quality', 'ase2d_resvol','ase2d_season','blev','btop','cetoe','cetop', 'css','dastd','dtoa','egro','epibs',\
 'ess','yield','etop','hsigma','issuermarketcap','mlev','rstr','season','sgro','specificreturn','specrisk','totalrisk','ase2d_btop_tsnorm', 'ase2d_divyild_tsnorm','ase2d_earnyild_tsnorm','ase2d_growth_tsnorm', 'ase2d_leverage_tsnorm', 'ase2d_oilsen_tsnorm',\
  'ase2d_quality_tsnorm', 'ase2d_resvol_tsnorm','ase2d_season_tsnorm','blev_tsnorm','btop_tsnorm','cetoe_tsnorm','cetop_tsnorm', 'css_tsnorm','dastd_tsnorm','dtoa_tsnorm','egro_tsnorm','epibs_tsnorm',\
 'ess_tsnorm','yield_tsnorm','etop_tsnorm','hsigma_tsnorm','issuermarketcap_tsnorm','mlev_tsnorm','rstr_tsnorm','season_tsnorm','sgro_tsnorm','specificreturn_tsnorm','specrisk_tsnorm','totalrisk_tsnorm'\
 'ase2d_btop_ret', 'ase2d_divyild_ret','ase2d_earnyild_ret','ase2d_growth_ret', 'ase2d_leverage_ret', 'ase2d_oilsen_ret', 'ase2d_quality_ret', 'ase2d_resvol_ret','ase2d_season_ret','ret_sma_5','ret_sma_10','ret_sma_20','ret_sma_50',\
 'ret_std_30', 'ret_std_90', 'ret_std_180','ret_std_250','ret_ema_5','ret_ema_10','ret_ema_20','ret_ema_50','ret_csum_5','ret_csum_10','ret_csum_20','ret_csum_50','ret_csum_diff_50_20','ret_csum_diff_20_10',\
 'ret_csum_diff_10_5', 'rv_5_ema_30_sd_0.5','rv_5_ema_90_sd_0.5','rv_5_ema_180_sd_0.5','rv_5_ema_250_sd_0.5','rv_10_ema_30_sd_0.5','rv_10_ema_90_sd_0.5','rv_10_ema_180_sd_0.5','rv_10_ema_250_sd_0.5',\
 'rv_20_ema_30_sd_0.5','rv_20_ema_90_sd_0.5','rv_20_ema_180_sd_0.5''rv_20_ema_250_sd_0.5','rv_50_ema_30_sd_0.5','rv_50_ema_90_sd_0.5','rv_50_ema_180_sd_0.5','rv_50_ema_250_sd_0.5',\
 # Time series cross sectional and vol of vol features
 'ret_sma_5_csvol','ret_sma_10_csvol','ret_sma_20_csvol','ret_sma_50_csvol','ret_std_30_csvol','ret_std_90_csvol','ret_std_180_csvol','ret_std_250_csvol','ret_ema_5_csvol','ret_ema_10_csvol','ret_ema_20_csvol',\
 'ret_ema_50_csvol','ret_csum_5_csvol','ret_csum_10_csvol','ret_csum_20_csvol','ret_csum_50_csvol','ret_csum_diff_50_20_csvol','ret_csum_diff_20_10_csvol','ret_csum_diff_10_5_csvol','rv_5_ema_30_sd_0.5_csvol',\
 'rv_5_ema_90_sd_0.5_csvol','rv_5_ema_180_sd_0.5_csvol','rv_5_ema_250_sd_0.5_csvol','rv_10_ema_30_sd_0.5_csvol','rv_10_ema_90_sd_0.5_csvol','rv_10_ema_180_sd_0.5_csvol','rv_10_ema_250_sd_0.5_csvol',\
 'rv_20_ema_30_sd_0.5_csvol','rv_20_ema_90_sd_0.5_csvol','rv_20_ema_180_sd_0.5_csvol','rv_20_ema_250_sd_0.5_csvol','rv_50_ema_30_sd_0.5_csvol','rv_50_ema_90_sd_0.5_csvol','rv_50_ema_180_sd_0.5_csvol',\
 'rv_50_ema_250_sd_0.5_csvol',\
 #TS vol....

 'ase2d_beta_tsvol', 'ase2d_dsbeta_tsvol','ase2d_momentum_tsvol', 'ase2d_strevrsl_tsvol', 'ase2d_airmarne_ret_tsvol', 'ase2d_autocomp_ret_tsvol', 'ase2d_banks_ret_tsvol', 'ase2d_build_ret_tsvol','ase2d_capgoods_ret_tsvol',\
 'ase2d_chemical_ret_tsvol', 'ase2d_cnstrp_ret_tsvol','ase2d_comtra_ret_tsvol', 'ase2d_condur_ret_tsvol', 'ase2d_consrv_ret_tsvol','ase2d_constap_ret_tsvol', 'ase2d_divfinan_ret_tsvol', 'ase2d_energy_ret_tsvol',\
 'ase2d_gold_ret_tsvol', 'ase2d_health_ret_tsvol', 'ase2d_insuran_ret_tsvol', 'ase2d_mediaret_ret_tsvol', 'ase2d_metmin_ret_tsvol','ase2d_realest_ret_tsvol', 'ase2d_software_ret_tsvol', 'ase2d_steel_ret_tsvol',\
 'ase2d_telecomm_ret_tsvol', 'ase2d_utility_ret_tsvol','ase2d_bet_ret_tsvol', 'ase2d_dsbeta_ret_tsvol', 'ase2d_indmom_ret_tsvol','ase2d_liquidty_ret_tsvol', 'ase2d_midcap_ret_tsvol', 'ase2d_momentum_ret_tsvol',\
 'ase2d_size_ret_tsvol','ase2d_strevrsl_ret_tsvol','ase2d_btop_tsvol', 'ase2d_divyild_tsvol','ase2d_earnyild_tsvol','ase2d_growth_tsvol', 'ase2d_leverage_tsvol', 'ase2d_oilsen_tsvol', 'ase2d_quality_tsvol', \
 'ase2d_resvol_tsvol','ase2d_season_tsvol', 'ase2d_btop_vov_90', 'ase2d_divyild_vov_90','ase2d_earnyild_vov_90','ase2d_growth_vov_90', 'ase2d_leverage_vov_90', 'ase2d_oilsen_vov_90', 'ase2d_quality_vov_90', \
 'ase2d_resvol_vov_90','ase2d_season_vov_90','ase2d_btop_vov_180', 'ase2d_divyild_vov_180','ase2d_earnyild_vov_180','ase2d_growth_vov_180', 'ase2d_leverage_vov_180', 'ase2d_oilsen_vov_180', 'ase2d_quality_vov_180', \
 'ase2d_resvol_vov_180','ase2d_season_vov_180','ase2d_beta_vov_90', 'ase2d_dsbeta_vov_90','ase2d_momentum_vov_90', 'ase2d_strevrsl_vov_90', 'ase2d_airmarne_vov_90', 'ase2d_autocomp_vov_90', 'ase2d_banks_vov_90',\
 'ase2d_build_vov_90','ase2d_capgoods_vov_90', 'ase2d_chemical_vov_90', 'ase2d_cnstrp_vov_90','ase2d_comtra_vov_90', 'ase2d_condur_vov_90', 'ase2d_consrv_vov_90','ase2d_constap_vov_90', 'ase2d_divfinan_vov_90',\
 'ase2d_energy_vov_90','ase2d_gold_vov_90', 'ase2d_health_vov_90', 'ase2d_insuran_vov_90', 'ase2d_mediaret_vov_90', 'ase2d_metmin_vov_90','ase2d_realest_vov_90', 'ase2d_software_vov_90', 'ase2d_steel_vov_90',\
 'ase2d_telecomm_vov_90', 'ase2d_utility_vov_90','ase2d_indmom_vov_90','ase2d_liquidty_vov_90', 'ase2d_midcap_vov_90', 'ase2d_momentum_vov_90', 'ase2d_size_vov_90','ret_sma_5_vol_90','ret_sma_10_vol_90',\
 'ret_sma_20_vol_90','ret_sma_50_vol_90','ret_ema_5_vov_90','ret_ema_10_vov_90','ret_ema_20_vov_90', 'ret_ema_50_vov_90', 'ret_csum_5_vov_90','ret_csum_10_vov_90','ret_csum_20_vov_90','ret_csum_vov_90',\
 'ret_csum_diff_50_20_vov_90','ret_csum_diff_20_10_vov_90','ret_csum_diff_10_5_vov_90','rv_5_ema_30_sd_0.5_vov_90','rv_5_ema_90_sd_0.5_vov_90','rv_5_ema_180_sd_0.5_vov_90','rv_5_ema_250_sd_0.5_vov_90',\
 'rv_10_ema_30_sd_0.5_vov_90','rv_10_ema_90_sd_0.5_vov_90','rv_10_ema_180_sd_0.5_vov_90','rv_10_ema_250_sd_0.5_vov_90','rv_20_ema_30_sd_0.5_vov_90','rv_20_ema_90_sd_0.5_vov_90','rv_20_ema_180_sd_0.5_vov_90',\
 'rv_20_ema_250_sd_0.5_vov_90','rv_50_ema_30_sd_0.5_vov_90','rv_50_ema_90_sd_0.5_vov_90','rv_50_ema_180_sd_0.5_vov_90', 'rv_50_ema_250_sd_0.5_vov_90','ase2d_beta_vov_180', 'ase2d_dsbeta_vov_180','ase2d_momentum_vov_180',\
 'ase2d_strevrsl_vov_180', 'ase2d_airmarne_ret_vov_180', 'ase2d_autocomp_ret_vov_180', 'ase2d_banks_ret_vov_180', 'ase2d_build_ret_vov_180','ase2d_capgoods_ret_vov_180','ase2d_chemical_ret_vov_180', \
 'ase2d_cnstrp_ret_vov_180','ase2d_comtra_ret_vov_180', 'ase2d_condur_ret_vov_180', 'ase2d_consrv_ret_vov_180','ase2d_constap_ret_vov_180', 'ase2d_divfinan_ret_vov_180', 'ase2d_energy_ret_vov_180',\
 'ase2d_gold_ret_vov_180', 'ase2d_health_ret_vov_180', 'ase2d_insuran_ret_vov_180', 'ase2d_mediaret_ret_vov_180', 'ase2d_metmin_ret_vov_180','ase2d_realest_ret_vov_180', 'ase2d_software_ret_vov_180',\
 'ase2d_steel_ret_vov_180', 'ase2d_telecomm_ret_vov_180', 'ase2d_utility_ret_vov_180','ase2d_indmom_vov_180','ase2d_liquidty_vov_180', 'ase2d_midcap_vov_180', 'ase2d_momentum_vov_180', 'ase2d_size_vov_180',\
 'ret_sma_5_vol_180','ret_sma_10_vol_180','ret_sma_20_vol_180','ret_sma_50_vol_180','ret_ema_5_vov_180','ret_ema_10_vov_180','ret_ema_20_vov_180', 'ret_ema_50_vov_180', 'ret_csum_5_vov_180',\
 'ret_csum_10_vov_180','ret_csum_20_vov_180','ret_csum_vov_180','ret_csum_diff_50_20_vov_180','ret_csum_diff_20_10_vov_180','ret_csum_diff_10_5_vov_180','rv_5_ema_30_sd_0.5_vov_180','rv_5_ema_90_sd_0.5_vov_180',\
 'rv_5_ema_180_sd_0.5_vov_180','rv_5_ema_250_sd_0.5_vov_180','rv_10_ema_30_sd_0.5_vov_180','rv_10_ema_90_sd_0.5_vov_180','rv_10_ema_180_sd_0.5_vov_180','rv_10_ema_250_sd_0.5_vov_180','rv_20_ema_30_sd_0.5_vov_180',\
 'rv_20_ema_90_sd_0.5_vov_180','rv_20_ema_180_sd_0.5_vov_180','rv_20_ema_250_sd_0.5_vov_180','rv_50_ema_30_sd_0.5_vov_180','rv_50_ema_90_sd_0.5_vov_180','rv_50_ema_180_sd_0.5_vov_180', 'rv_50_ema_250_sd_0.5_vov_180']

print(len(allgroup))

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
