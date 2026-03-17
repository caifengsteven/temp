import autosklearn.pipeline.components.regression
from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS
import autosklearn.regression

class HuberRegression(AutoSklearnRegressionAlgorithm):
	def __init__(self, espilon, max_iter, alpha, fit_intercept= True):
		self.alpha = alpha
		self.espilon = espilon
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.estimator = None
	def fit(self, X,y):
		self.alpha = float(self.alpha)
		self.espilon = float(self.espilon)
		self.max_iter = int(self.max_iter)
		self.fit_intercept bool(self.fit_intercept)
		import sklearn.linear_model.HuberRegressor
		self.estimator = sklearn.linear_model.HuberRegressor(alpha=self.alpha, espilon = self.espilon, max_iter= self.max_iter, fit_intercept = self.fit_intercept,self.estimator.fit(X,y))
		return self
	def predict(self, X):
		if self.estimator is None:
			raise NotImplementedError
		return self.estimator.predict(X)

	@staticmethod
	def get_properties(dataset_properties = None):
		return {'shortname':'HR', 'name':'Huber Regressor', 'handles_regression':True, 'handles_classification':False, 'handles_multiclass':False, 'handles_multilabel':False, 'is_deterministic': True, 'input':(SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA), 'output':(PREDICTIONS,)}
	@staticmethod
	def get_hyperparameter_search_space(dataset_properties = None):
		cs = ConfigurationSpace()
		alpha = UniformFlooatHyperparameter(name='alpha', lower= 10**-5, upper=1, log=True, default_values=0.0001)
		espilon = UniformFlooatHyperparameter(name='espilon', lower= 1.35, upper=1.75, log=True, default_values=1.35)
		max_iter = UniformFlooatHyperparameter(name='max_iter', lower= 100, upper=200, log=True, default_values=100)
		fit_intercept = CategoricalHyperparameter(name='fit_intercept', choices=['True', 'False',],default_value= 'True')
		cs.add_hyperparameters([alpha, espilon, max_iter, fit_intercept])
		return cs
if __name__=='__main__':
	autosklearn.pipeline.components.regression.add_regressor(HuberRegressor)
	cs = HuberRegressor.get_hyperparameter_search_space()
	print cs


def sharpe_ratio (oneday_return, solutions, prediction):
	df = pd.DataFrame()
	df['pred'] = np.hstack(prediction)
	df['target'] = np.hstack(oneday_return)
	df['pred_sum'] = df['pred'].abs().sum()
	df['weight']= df['pred']/df['pred_sum']
	df['pnl'] = df['weight']*df['target']
	df['pnl_day']= df['pnl'].groupby(level=0).sum()
	s = df['pnl_day'].mean()/df['pnl_day'].std()*15.5
	print('{:.2f}'.format(s))
	return s

from functools import partial

p_sharpe_ratio = partial(sharpe_ratio, (df_clean['target_1'][int(len(df_clean['target_1'])*0.7):]).values)
import autosklearn.metrics
sharpe_scorer = autosklearn.metrics.make_scorer(name='sharpe', score_func= p_sharpe_ratio, greater_is_better=True, needs_proba=False, need_threshold=False)

import pickle
from sklearn.externals import joblib
filename = '/dat/golkonda_data_2/rates/notebooks/data/pickle/twd/finalmodel_'
today= '_20190730_'

models ={}
model_algo={}
for target in targets:
	print(target)
	model = autosklearn.regression.AutoSklearnRegressionAlgorithm(time_left_for_this_task = 3600, per_run_time_limt=360, resampling_strategy='holdout', resampling_strategy_argument={'train_size':0.7, 'shuffle':False}, n_jobs=-1,ensemble_memory_limit =4096,ml_memory_limit=18000, tmp_folder='./temp', output_folder='./output', delete_tmp_folder_after_terminate=True, delete_output_folder_after_terminate=True)
	model.fit(df_clean[features].values, df_clean[target].values, metrics= sharpe_scorer, feat_type = feature_type)
	filepick = str(filename)+str(target)+str(today)+'pickle.sav'
	pickle.dump(model, open(filepick,'wb'))
	model_algo[target] = model.show_models()
	models[target]=model

	df_pnl = pd.DataFrame(index = df_test.index)
df_weight = pd.DataFrame(index= df_test.index)
for target, model in models.items():
	print(target)
	models.show_models()

	y_pred = model.predict(df_test[features].values)
	df_test['pred'] = y_pred
	df_test['pred_sum'] = df_test['pred'].abs().groupby(level=0).sum().reindex(df_test.index, level=0)
	df_weight[target] = df_test['pred']/df_test['pred_sum']
	plot_ts((df_test['pred']*df_test['target_1']).groupby(level=0).sum().cumsum())
	

df_weight['target_135'] = df_weight[['target_1','target_3', 'target_5']].mean(axis=1)
df_pnl = df_weight.mul(df_test['target_1'], axis=0)
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
