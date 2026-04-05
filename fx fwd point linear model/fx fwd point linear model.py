# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:00:27 2019

@author: ASUS
"""

import pandas as pd
import numpy as np
from tpot import TPOTClassifier
import h2o
from operator.common import ema, rolling_std, expanding_mean, expanding_std
from backtester.plotting import plot_ts
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
from importlib import reload
import pyfolio as pf

from backtester.plotting import plot_nice_table
from sklearn.linear_model import Lasso, LarsCV, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, ElasticNetCV
from sklearn.linear_model import Ridge, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor, HuberRegressor

df = pd.read_csv('twd_10y_date.csv')

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(['date','pair_name']).sort_index()
targets = ['target_1','target_3','target_5','target_mix135']
pairs = df.index.levels[1]
