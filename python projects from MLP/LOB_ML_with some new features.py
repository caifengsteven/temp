'''
add some new features

'''

def drop_numerical_outlier(combine, z_thresh=3):
	cols = combine.columns.tolist()
	cols = cols[:-1]
	temp = combine[cols]
	constrains = temp.select_dtypes(include=[np.number]).apply(lambda x:np.abs(stats.zscore(x))<z_thresh, reduce=False).all(axis=1)
	print(constrains)
	combine.drop(combine.index[~constrains], inplace=True)
	return combine

def drop_numerical_outlier_x(X, z_thresh=3):
	constrains = X.select_dtypes(include=[np.number]).apply(lambda x:np.abs(stats.zscore(x))<z_thresh, reduce= False).all(axis=1)
	x.drop(X.index[!constrains], inplace = True)
	return X

deltaT=200
import pandas as pd
import talib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble.forest import RandomForestClassifier
from xgboost import XGBClassifier

import datetime
from dateime import date, time
from dateutil import rrule, parser
from datetime import timedelta

from talib import BBANDS, DEMA, EMA, HT_TRENDLINE, KAMA, MA, MAMA, MIDPOINT, MIDPRICE, SAR, SAREXT, SMA, T4, TEMA, TRIMA, WMA
from talib import ADX, ADXR, AROON, AROONOSC, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM, PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR100, RSI, STOCH, STOCHF, STOCHRSI, TRIX, ULTOSC, WILLR
from talib import AD, ADOSC, OBV
from talib import HT_DCPERIOD, HT_DCPHASE, HT_PPASOR, HT_SINE, HT_TRENDMOOD
from talib import AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE
from talib import ATR, NATR, TRANGE
from scipy import stats

def z_score(x, window):
	r = x.rolling(window=window)
	m = r.mean().shift(1)
	s = r.std(ddof=0).shift(1)
	z= (x-m)/s
	return z.values

date1 = '2019-06-25'
date2 = '2019-09-10'

datesx = list(rrule.rrule(rrle.DAILY, dtstart=parser.parse(date1), until=parser.parse(date2)))
f1= pd.DataFrame()
for i in datesx:
    try:
        filename = '000063.SZ.SHZ.'+i.strftime('%Y-%m-%d')+'.hdf'
        temp = pd.read_hdf(filename)
        beforeAuction = i+timedelta(hours=1,minutes=30,seconds=0)
        temp = temp[temp.index>beforeAuction]
        f1 = pd.cancat([f1,temp])
    except:
        continue

print(list(f1.keys()))
print(f1)

v1 = f1[['l1bidsize','l1asksize','l2bidsize','l2asksize','l3bidsize','l3asksize','l4bidsize','l4asksize','l5bidsize','l5asksize','l6bidsize','l6asksize','l7bidsize','l7asksize','l8bidsize','l8asksize','l9bidsize','l19asksize','l10bidsize','l10asksize']]
a = f1['l1askprice'] - f1['l1bidprice']
close = f1['price']
volume = f1['accvol']
v2 = pd.DataFrame(a, columns=['l1bid_ask_spread'])
v2['l1bid_ask_spread'] = f1['l1askprice']-f1['l1bidprice']
v2['l2bid_ask_spread'] = f1['l2askprice']-f1['l2bidprice']
v2['l3bid_ask_spread'] = f1['l3askprice']-f1['l3bidprice']
v2['l4bid_ask_spread'] = f1['l4askprice']-f1['l4bidprice']
v2['l5bid_ask_spread'] = f1['l5askprice']-f1['l5bidprice']
v2['l6bid_ask_spread'] = f1['l6askprice']-f1['l6bidprice']
v2['l7bid_ask_spread'] = f1['l7askprice']-f1['l7bidprice']
v2['l8bid_ask_spread'] = f1['l8askprice']-f1['l8bidprice']
v2['l9bid_ask_spread'] = f1['l9askprice']-f1['l9bidprice']
v2['l10bid_ask_spread'] = f1['l10askprice']-f1['l10bidprice']

v2['l1bid_ask_size_spread'] = f1['l1asksize']-f1['l1bidsize']
v2['l2bid_ask_size_spread'] = f1['l2asksize']-f1['l2bidsize']
v2['l3bid_ask_size_spread'] = f1['l3asksize']-f1['l3bidsize']
v2['l4bid_ask_size_spread'] = f1['l4asksize']-f1['l4bidsize']
v2['l5bid_ask_size_spread'] = f1['l5asksize']-f1['l5bidsize']
v2['l6bid_ask_size_spread'] = f1['l6asksize']-f1['l6bidsize']
v2['l7bid_ask_size_spread'] = f1['l7asksize']-f1['l7bidsize']
v2['l8bid_ask_size_spread'] = f1['l8asksize']-f1['l8bidsize']
v2['l9bid_ask_size_spread'] = f1['l9asksize']-f1['l9bidsize']
v2['l10bid_ask_size_spread'] = f1['l10asksize']-f1['l10bidsize']

v2['l1mid'] = (f1['l1askprice']+f1['l1bidprice'])/2
v2['l2mid'] = (f1['l2askprice']+f1['l2bidprice'])/2
v2['l3mid'] = (f1['l3askprice']+f1['l3bidprice'])/2
v2['l4mid'] = (f1['l4askprice']+f1['l4bidprice'])/2
v2['l5mid'] = (f1['l5askprice']+f1['l5bidprice'])/2
v2['l6mid'] = (f1['l6askprice']+f1['l6bidprice'])/2
v2['l7mid'] = (f1['l7askprice']+f1['l7bidprice'])/2
v2['l8mid'] = (f1['l8askprice']+f1['l8bidprice'])/2
v2['l9mid'] = (f1['l9askprice']+f1['l9bidprice'])/2
v2['l10mid'] = (f1['l10askprice']+f1['l10bidprice'])/2

c = f1['l2askprice'] - f1['l1askprice']
d = f1['l1bidprice'] - f1['l2bidprice']
v3 = pd.cancat([c,d], axis=1)
for i in range (8):
	str_ask = 'l'+str(i+3) +'askprice'
	str_bid = 'l'+str(i+3) +'bidprice'
	c = f1[str_ask] - f1['l1askprice']
	d = f1['l1bidprice'] - f1[str_bid]
	v3 = pd.cancat([v3,c,d], axis =1)
for i in range (9):
	str_ask1 = 'l'+str(i+1)+'askprice'
	str_ask2 = 'l'+str(i+2)+'askprice'
	str_bid1 = 'l'+str(i+1)+'bidprice'
	str_bid2 = 'l'+str(i+2)+'bidprice'
	c = abs(f1[str_ask2] - f1[str_ask1])
	d = abs(f1[str_bid2] - f1[str_bid1])
	v3 = pd.cancat([v3,c,d], axis =1)

v3 = pd.DataFrame(v3.values, columns= ['v3-'+str(i+1) for i in range(36)], index = v3.index)
v4 = pd.DataFrame(index = f1.index)

avg_ask = f1[['l1askprice','l2askprice','l3askprice','l4askprice','l5askprice','l6askprice','l7askprice','l8askprice','l9askprice','l10askprice']].T.mean().T
avg_bid = f1[['l1bidprice','l2bidprice','l3bidprice','l4bidprice','l5bidprice','l6bidprice','l7bidprice','l8bidprice','l9bidprice','l10bidprice']].T.mean().T

avg_ask_vol = f1[['l1asksize','l2asksize','l3asksize','l4asksize','l5asksize','l6asksize','l7asksize','l8asksize','l9asksize','l10asksize']].T.mean().T
avg_bid_vol = f1[['l1bidsize','l2bidsize','l3bidsize','l4bidsize','l5bidsize','l6bidsize','l7bidsize','l8bidsize','l9bidsize','l10bidsize']].T.mean().T
v4 = pd.cancat([v4, avg_ask,avg_bid, avg_ask_vol, avg_bid_vol], axis =1 )
v4 = pd.DataFrame(v4.values, columns = ['v4-'+str(i+1) for i in range(4)], index = v4.index)
bid_ask_spread = v2[['l1bid_ask_spread','l2bid_ask_spread','l3bid_ask_spread','l4bid_ask_spread','l5bid_ask_spread','l6bid_ask_spread','l7bid_ask_spread','l8bid_ask_spread','l9bid_ask_spread','l10bid_ask_spread']]
v5_1 = bid_ask_spread.T.sum().T
bid_ask_size_spread = v2[['l1bid_ask_size_spread','l2bid_ask_size_spread','l3bid_ask_size_spread','l4bid_ask_size_spread','l5bid_ask_size_spread','l6bid_ask_size_spread','l7bid_ask_size_spread','l8bid_ask_size_spread','l9bid_ask_size_spread','l10bid_ask_size_spread']]
v5_2 = bid_ask_size_spread.T.sum().T
v5 = pd.cancat([v5_1,v5_2], axis = 1)
v5 = pd.DataFrame(v5.value, columns = ['v5-'+str(i+1) for i in range(2)], index = f1.index)
v6 = pd.DataFrame(f1.loc[:, ['l1bidsize', 'l1asksize','l2bidsize', 'l2asksize','l3bidsize', 'l3asksize','l4bidsize', 'l4asksize','l5bidsize', 'l5asksize','l6bidsize', 'l6asksize','l7bidsize', 'l7asksize','l8bidsize', 'l8asksize','l9bidsize', 'l9asksize','l10bidsize', 'l10asksize', 'l1bidprice', 'l1askprice','l2bidprice', 'l2askprice','l3bidprice', 'l3askprice','l4bidprice', 'l4askprice','l5bidprice', 'l5askprice','l6bidprice', 'l6askprice','l7bidprice', 'l7askprice','l8bidprice', 'l8askprice','l9bidprice', 'l9askprice','l10bidprice', 'l10askprice']].diff())
v6 = pd.DataFrame(v6.values, columns=['v6-'+str(i+1) for i in range(40)], index = f1.index)
# deltaT is time window


deltaT = 200
e = pd.DataFrame(f1.iloc[deltaT:,].values, columns= f1.columns)
print('*'*40)
f = pd.DataFrame(f1.iloc[:-deltaT,].values, columns = f1.columns)
spread = pd.DataFrame(np.zeors(len(e)), index = e.index)
spread[e.loc[:,'price']<f.loc[:,'price']*0.995]=-1
spread[e.loc[:,'price']>f.loc[:,'price']*1.005]=1

X = pd.cancat([v1,v2,v3,v4,v5,v6], axis =1)
old_index = X.index
upperbank, middleband, lowerband = BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn =2, matype =0)
X['bolling_upperband_normal'] = upperband
X['bolling_lowerband_normal'] = lowerband
X['bolling_middleband_normal'] = middleband

real = DEMA(close, timeperiod=30)
X['DEMA_normal'] = real
real = EMA(close, timeperiod=30)
X['EMA_normal'] = real
real = HT_TRENDLINE(close)
X['HT_TRENDLINE_normal'] = real
real = KAMA(close, timeperiod=30)
X['KAMA_normal'] = real
real = MA(close, timeperiod=30, matype =0)
X['MA_normal'] = real
real = MIDPOINT(close, timeperiod=14)
X['MIDPOINT_normal'] = real
real = SMA(close, timeperiod=30)
X['SMA_normal'] = real
real = T3(close, timeperiod=5, vfactor=0)
X['T3_normal'] = real
real = TEMA(close, timeperiod=30)
X['TEMA_normal'] = real
real = TRIMA(close, timeperiod=30)
X['TRIMA_normal'] = real
real = WMA(close, timeperiod=30)
X['WMA_normal'] = real
real = APO(close, fastperiod=12, slowperiod =26, matype =0)
X['APO_normal'] = real
real = CMO(close, timeperiod=14)
X['CMO_normal'] = real
macd, macdsignal, macdhist = MACD(close, fastperiod =12, slowperiod=26, signalperiod=9)
X['macd_normal'] = macd
X['macdsignal_normal'] = macdsignal
X['macdhist_normal'] = macdhist
real = MOM(close, timeperiod=10)
X['MOM_normal'] = real
real = PPO(close, fastperiod=12, slowperiod=26, matype=0)
X['PPO_normal'] = real
real = ROC(close, timeperiod=10)
X['ROC_normal'] = real
real = ROCP(close, timeperiod=10)
X['ROCP_normal'] = real
real = ROCR(close, timeperiod=10)
X['ROCR_normal'] = real
real = RSI(close, timeperiod=14)
X['RSI_normal'] = real
fastk_rsi, fastd_rsi = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
X['fastk_rsi_normal']=fastk_rsi
X['fastd_rsi_normal']=fastd_rsi
real = TRIX(close, timeperiod=30)
X['TRIX_normal'] = real
real = OBV(close, volume)
X['OBV_normal'] = real
real = HT_DCPERIOD(close)
X['HT_DEPERIOD_normal'] = real
real = HT_DCPHASE(close)
X['HT_DEPHASE_normal'] = real
inphase, quadrature = HT_PHASOR(close)
X['inphase_normal']  = inphase
X['quadrature_normal'] = quadrature
sine, leadsine = HT_SINE(close)
X['sine_normal'] = sine
X['leadsine_normal'] = leadsine
integer = HT_TRENDMOOD(close)
X['HT_TRENDMOOD_normal'] = integer
X = pd.DataFrame(X.values, index = old_index, columns= X.columns)
Y = spread

X = pd.DataFrame(X.iloc[1:-deltaT,:].values, columns=X.columns,index=X.iloc[1:-deltaT, :].index)
Y = pd.DataFrame(Y.iloc[1:,:].values, columns=['spread'], index=X.index)

cols = X.columns.tolist()
combine = pd.cancat([X,Y], axis=1)
combine = combine.dropna()
combine = drop_numerical_outlier(combine)
X = combine[cols]
Y = combine['spread']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state =42)

clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(X_train, Y_train)
Y_predict = clf_tree.predict (X_test)
print('%'*40)
print('accuracy for Decision Tree')
print(accuracy_score(Y_test, Y_predict))

rfc = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
rfc = rfc.fit(X_train, Y_train)
Y_predict = rfc.predict(X_test)
print('%'*40)
print('accuracy for RandomForest')
print(accuracy_score(Y_test, Y_predict))


















