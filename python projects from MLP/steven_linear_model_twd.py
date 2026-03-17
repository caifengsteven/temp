import pandas as pd
import numpy as np
from tpot import TPOTClassifier
import h2o
from operators.common import ema, rolling_std, expanding_mean, expanding_std
from backtester.plotting import plot_ts
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from importlib import reload
import pyfolio as pf
from backtester.plotting import plot_nice_table

from sklearn.linear_model import Lasso, LarsCV, LassoCV, LassoLars, LassoLarsCV, LassoLarIC, LinearRegression, LogisticRegression, ElasticNet, ElasticNetCV
from sklearn.linear_model import Ridge, MultiTaskLasso, MultiTaskElasticNet, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor
from sklearn.linear_model import PassiveAgressiveRegressor, RANSACRegressor, TheilSenRegressor, HuberRegressor

df = pd.read_csv('/dat/golkonda_data_2/rates/notebooks/data/twd_10y_data.csv')

df.drop('Unnamed: 0', axis =1 , inplace = True)
df['date'] =pd.to_datetime(df['date'])
df = df.set_index(['date', 'pair_name']).sort_index()

targets = ['target_1','target_3', 'target_5', 'target_mix135']
pairs = df.index.levels[1]

pairs_map={'ntn11w':0, 'ntn121':1,'ntn123':2, 'ntn126':3, 'ntn129':4,'ntn31':5, 'ntn31w':6, 'ntn61':7,'ntn63':8, 'ntn91':9,'ntn93':10, 'ntn96':11}
macro_features = ['usd_swap_ois_6m', 'usd_swap_ois_12m', 'tw_ea_eco_surprise_citi', 'tw_elec_export_yoy_m', 'tw_ind_prod_yoy_m', 'tw_budge_balance_m','tw_10yr_govtnote', 'fxvix_index', 'rty_index','us_bond_index'\
'igbond_index', 'hybond_index', 'tybond_index', 'intbond_index', 'tips_index', 'us_cds_index', 'gb_bond_index', 'ge_bond_index', 'fr_bond_index', 'it_bond_index', 'sp_bond_index', 'eu_bond_index', 'eu_cds_index'\
'tw_frgn_invst', 'tw_prime_rate', 'tw_overnight_rate', 'tw_1d_repo', 'tw_10d_repo', 'tw_fix', 'tw_wmco_fix','tw_fix_diff', 'tw_first_future', 'tw_basis','tw_krwtwd_fx']
index_returns = ['px_last_spot_rtn', 'tamsci_index_rtn', 'spx_index_rtn', 'oil_future_rtn','dxy_index_rtn']
index_returns_1 =['px_last_spot_rtn', 'tamsci_index_rtn', 'spx_index_rtn', 'oil_future_rtn','dxy_index_rtn', 'rty_index_rtn', 'us_bond_index_rtn', 'igbond_index_rtn', 'hybond_index_rtn', 'tybond_index_rtn',\
'intybond_index_rtn', 'tips_index_rtn', 'us_cds_index_rtn', 'gb_bond_index_rtn','ge_bond_index_rtn','fr_bond_index_rtn', 'it_bond_index_rtn', 'sp_bond_index_rtn', 'eu_bond_index_rtn', 'eu_cds_index_rtn',\
]

df[macro_features] = df[macro_features].groupby(level=1).shift(1).fillna(0)
macro_change_cols = [col +'_chg' for col in marco_cols]
df[macro_change_cols] = df[marco_cols].groupby(level=1).diff().fillna(0)
macro_change_rtns = [col + '_rtn' for col in macro_cols_rtn]
df[macro_change_rtns] = df[macro_cols_rtn].groupby(level=1).diff().fillna(0)/df[macro_cols_rtn].groupby(level=1).shift(1)
df[macro_change_rtns] = df[macro_change_rtns].replace([np.inf, -np.inf], 0.0)
df[macro_change_rtns]= df[macro_change_rtns].fillna(0)

features = macro_features + macro_change_cols + pair_features + onshore_spreads
features_ema = ema (df[features], halflife=20, min_periods=1)
features_dev = df[features] - features_ema
features_std = expanding_std(features_dev, min_periods=60)
features_std = features_std.groupby(level=1).bfill()
features_norm = (features_dev)/features_std
df_norm = features_norm
df_norm[targets] = df[targets]
cutoff_date = '20160104'
df_train = df_norm.loc[:cutoff_date]
df_test = df_norm.loc[cutoff_date:]
df_clean = df_train.fillna(df_train.median())

print(len(df_clean))

'''
Testing LassorLarsIC AIC

'''
models={}
for target in targets:
	print(target)
	model = LassoLarIC(criterion = 'aic')
	model.fit(df_clean[features].values, df_clean[target].values)
	models[target] = model

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))

'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing LassorLarsIC BIC

'''
models={}
for target in targets:
	print(target)
	model = LassoLarIC(criterion = 'bic')
	model.fit(df_clean[features].values, df_clean[target].values)
	models[target] = model

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing LassoCV

'''
models={}
for target in targets:
	print(target)
	model = LassoCV(cv =10)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing Lasso

'''
models={}
for target in targets:
	print(target)
	model = Lasso(alpha = 0.00001)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing ElasticNet

'''
models={}
for target in targets:
	print(target)
	model = ElaticNet(alpha = 0.00001, l1_ratio=0.6)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing Linear Regression

'''
models={}
for target in targets:
	print(target)
	model = LinearRegression()
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)

'''
Testing Ridge

'''
models={}
for target in targets:
	print(target)
	model = Ridge(alpha=0.5)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)



'''
Testing OrthogonalMatchingPursuit

'''
models={}
for target in targets:
	print(target)
	model = OrthogonalMatchingPursuit(n_nonzero_coef=10)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing BayestimaRidge

'''
models={}
for target in targets:
	print(target)
	model = BayestimaRidge()
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing Logistic Regression

'''
models={}
for target in targets:
	print(target)
	model = LogisticRegression(penalty='l1', solver = 'saga', tol= 1e-6, max_iter=1000)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing SGDRegressor

'''
models={}
for target in targets:
	print(target)
	model = SGDRegressor(max_iter=1000, tol=1e-6)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing PassiveAggressiveRegressor

'''
models={}
for target in targets:
	print(target)
	model = PassiveAggressiveRegressor(max_iter=1000, tol=1e-6, random_state=0)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)




'''
Testing RANSAC

'''
models={}
for target in targets:
	print(target)
	model = RANSACRegressor()
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing TheilSenRegressor

'''
models={}
for target in targets:
	print(target)
	model = TheilSenRegressor(random_state=42)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing LassoCV

'''
models={}
for target in targets:
	print(target)
	model = HuberRegressor(fit_intercept = True, alpha = 0.0, max_iter=1000, epsilon=1.3)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing ARDRegression

'''
models={}
for target in targets:
	print(target)
	model = ARDRegression(alpha_1 = 1e-6, alpha_2=1e-6, compute_score = False, copy_X= True, fit_intercept = True, lambda_1=1e-6, lambda_2= 1e-6, n_iter=300)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Testing Random Forest

'''
models={}
for target in targets:
	print(target)
	model = RandomForestRegressor(n_estimators = 3000, oob_score=True, random_state=1, max_depth = 3, n_jobs=-1)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


def zscore_frame(dframe):
	for col in dframe.columns:
		if(dframe[col].dytpe =='object'):
			print(col)
		else:
			dframe[col] = (dframe[col]-dframe[col].mean())/dframe[col].std()
	return dframe
def fill_na_media(dframe):
	for col in dframe.columns:
		if(dframe[col].dytpe =='object'):
			print(col)
		else:
			dframe[col] = dframe[col].fillna(dframe[col].median())
	return dframe


'''
Testing SVM

'''

from sklearn import svm
models={}
for target in targets:
	print(target)
	model = svm.SVR()
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
Lets try autosklearn

'''

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.regression

for target in targets:
	automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task =120, per_run_time_limit=30, resampling_strategy = 'holdout', resampling_strategy_arguments={'train_size':0.67,'shuffle':False})
	automl.fit(df_clean[features].values, df_clean[target].values)
	models[target]=automl


from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import gradientBoostingRegressor


models={}
for target in targets:
	print(target)
	model = MLPRegressor(hiddle_layer_size = (500,), random_state=0)
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)


'''
select feature

'''

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, f_regression, SelectPercentile, SelectFromModel

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

models={}
for target in targets:
	print(target)
	model = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty='l1'))),('regression', RandomForestRegressor(n_estimators=3000))])
	models[target] = model
	model.fit(df_clean[features].values, df_clean[target].values)
	

df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	model.score(df_test[features].values, df_test[target].values)
	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_weight[target] = df_test['pred'].apply(lambda x:1 if x>0 else -1)

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1']), axis=0
df_pnl_day = df_pnl.groupby(level=0).sum()
df_pnl_day = df_pnl_day.loc[cutoff_date:]
plot_ts(df_pnl_day.cumsum())
print(df_pnl_day.mean()/df_pnl_day.std()*15.5)

combined_size =1
pf.plotting.plot_drawdown_periods(df_pnl_day['target_1']/combined_size, top=10, figsize=(15,10))
plot_nice_table(pf.timeseries,.gen_drawdown_table(df_pnl_day['target_1']/combined_size, top=10), title='dd')
'''
with consideration of transaction cost
'''

df_tran = df_weight.diff()
df_tran = df_tran.abs()
df_tran = df_tran.mul(0.00003, axis=0)
df_pnl_withtran = df_wieght.mul(df_test['target_1'], axis=0) - df_tran
df_pnl_day_withtran = df_pnl_withtran.groupby(level=0).sum()
df_pnl_day_withtran = df_pnl_day_withtran.loc[cutoff_date:]
plot_ts(df_pnl_day_withtran.cumsum())
print(df_pnl_day_withtran.mean()/df_pnl_day_withtran.std()*15.5)



