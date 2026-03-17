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

df[features] = df[features].replace([np.inf, -np.inf],np.nan)
df[features] = df[features].fillna(0)

for target in ['target_1', 'target_2','target_3','target_5','target_10','target_15','target_20']:
	df[target] = df[target].replace([np.inf, -np.inf],np.nan)
	df[target] = df[target].fillna(0)
