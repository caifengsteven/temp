# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:28:15 2019

@author: adair2019
"""

import pandas as pd
import talib
#import matplotlib.pyplot as plt
#import scipy.io as scio
import os
from scipy.stats import rankdata

path0 = r'D:\datasets\期货\FutAC_Min1_Std_Year_2004-2018'
def neg(X):
    return -X
def delay(X,d):
    y = X.rolling(window=d).apply(lambda x: x[0],raw=True)
    return y

def delta(X,d):
    y = X-delay(X,d)
    return y

def ts_rank(X,var,d):
    return X[var].rolling(window=d).apply(lambda x: rankdata(x)[-1]/d,raw=True)

def import_data(fn):
    x = pd.read_csv(fn,engine='python')
    x.rename(columns={'市场代码':'code','合约代码':'codenum','时间':'tradingdate',
              '开':'openprice', '高':'highprice', '低':'lowprice', '收':'closeprice', 
              '成交量':'turnoverVol', '成交额':'turnoverValue' , '持仓量':'openInt'},inplace=True)
    return x


def RB_data():
    fn = 'rb主力连续.csv'
    fn_tocsv = '%s_factor_data.csv' % 'rb'
    fn = os.path.join(path0,fn)
    x = import_data(fn)
    wid = 450
    x['f'] = talib.MIDPOINT(-x.closeprice,wid)
    z=x[['codenum','tradingdate','openprice','closeprice','f']]    
    z.to_csv(fn_tocsv)
    print('Complete %s' % 'RB')
    
def L_data():
    fn = 'l主力连续.csv'
    fn_tocsv = '%s_factor_data.csv' % 'L'
    fn = os.path.join(path0,fn)
    x = import_data(fn)
    wid = 630
    x['f'] = talib.DEMA(-x.closeprice,wid)
    z=x[['codenum','tradingdate','openprice','closeprice','f']]    
    z.to_csv(fn_tocsv)  
    print('Complete %s' % 'L')
    
def AL_data():
    # delta(ADX(high, low, close, 330), 630)
    fn = 'al主力连续.csv'
    fn_tocsv = '%s_factor_data.csv' % 'AL'
    fn = os.path.join(path0,fn)
    x = import_data(fn)
    x['f'] = delta(talib.ADX(x.highprice,x.lowprice,x.closeprice,330),630)
    z=x[['codenum','tradingdate','openprice','closeprice','f']]    
    z.to_csv(fn_tocsv) 
    print('Complete %s' % 'AL')
    
def RU_data():
    fn = 'ru主力连续.csv'
    fn_tocsv = '%s_factor_data.csv' % 'RU'
    fn = os.path.join(path0,fn)
    x = import_data(fn)
    x['f'] = neg(delta(x.turnoverVol,60))
    z=x[['codenum','tradingdate','openprice','closeprice','f']]  
    z.to_csv(fn_tocsv)
    print('Complete %s' % 'RU')

def RM_data():
    fn = 'RM主力连续.csv'
    fn_tocsv = '%s_factor_data.csv' % 'RM'
    fn = os.path.join(path0,fn)
    x = import_data(fn)
    x['f'] = talib.MA(x.turnoverVol,180)#neg(delta(x.turnoverVol,60))
    z=x[['codenum','tradingdate','openprice','closeprice','f']]  
    z.to_csv(fn_tocsv)
    print('Complete %s' % 'RM')

def J_data():
    fn = 'j主力连续.csv'
    fn_tocsv = '%s_factor_data.csv' % 'J'
    fn = os.path.join(path0,fn)
    x = import_data(fn)
    y = x.openInt.rolling(window=2).apply(lambda x: x[1]/x[0]-1,raw=True)
    x['f'] = talib.MIDPOINT(talib.DEMA(y,440),450)#neg(delta(x.turnoverVol,60))
    z=x[['codenum','tradingdate','openprice','closeprice','f']]  
    z.to_csv(fn_tocsv)
    print('Complete %s' % 'J')
    
def I_data():
    fn = 'i主力连续.csv'
    fn_tocsv = '%s_factor_data.csv' % 'I'
    fn = os.path.join(path0,fn)
    x = import_data(fn)
    x['f'] = talib.HT_DCPHASE(talib.ADX(x.highprice,x.lowprice,x.closeprice,210))#neg(delta(x.turnoverVol,60))
    z=x[['codenum','tradingdate','openprice','closeprice','f']]  
    z.to_csv(fn_tocsv)
    print('Complete %s' % 'I')
    
def HC_data():
    fn = 'hc主力连续.csv'
    fn_tocsv = '%s_factor_data.csv' % 'HC'
    fn = os.path.join(path0,fn)
    x = import_data(fn)
    x['f'] = talib.HT_DCPHASE(ts_rank(x,'highprice',210))#neg(delta(x.turnoverVol,60))
    z=x[['codenum','tradingdate','openprice','closeprice','f']]  
    z.to_csv(fn_tocsv)
    print('Complete %s' % 'HC')

if __name__ == "__main__":
    RB_data()
    L_data()
    AL_data()
    RU_data()
    RM_data()
    J_data()
    I_data()
    HC_data()