# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 08:07:20 2021

@author: ASUS
"""

import matplotlib


import logging
import signal, threading, os, time
import logging

from IPython.core.debugger import set_trace
from IPython.terminal.embed import embed

import os, sys
import asyncio

import nest_asyncio
nest_asyncio.apply()

import inspect
#import ccxt.async_support as ccxt
import ccxt
import pandas as pd

import mplfinance as mpf
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 12]



try:
    from settings import HTTP_PROXY
except ImportError:
    HTTP_PROXY = None
    
# (!!!)You should change this according to your network environment
if HTTP_PROXY is None:
    HTTP_PROXY = 'http://127.0.0.1:1087'    

def kline_to_df(arr) -> pd.DataFrame:
    kline = pd.DataFrame(
        arr,
        columns=['ts', 'open', 'high', 'low', 'close', 'volume' ])
    kline.index = pd.to_datetime(kline.ts, unit='ms')
    kline.drop('ts', axis=1, inplace=True)
    return kline
    

#%%


PROXIES = {
    'http': HTTP_PROXY,
    'https': HTTP_PROXY,
}

ccxt_options = {'proxies': PROXIES}

ok = 'okex'
bn = 'binance'
hb = 'huobipro'

client_list = [hb, bn, ok]


from HarmonicPatterns.harmonic_functions import HarmonicDetector
from HarmonicPatterns.harmonic import send_alert, search_function
from functools import partial
import time
from multiprocessing import Pool



detector = HarmonicDetector(error_allowed=0.07, strict=True, predict_err_rate=0.07)
search = partial(search_function, detector, ccxt_args=ccxt_options, only_last=False)
PERIODS = ['1h', '4h', '1d']



s = ['BTC/USDT', 'ETH/USDT']
search('binance', s, periods = ['15m', '30m', '1h', '4h'])



s = ['BTC/USDT', 'ETH/USDT', 'BCH/USDT', 'ETC/USDT', 'ADA/USDT','XMR/USDT', 'DOT/USDT', 'EOS/USDT', 'LTC/USDT']
#s = ['BTC/USDT', 'ETH/USDT', 'DOT/USDT','BCH/USDT']



#s = ['ETH/BTC', 'BCH/BTC', 'EOS/BTC', 'ETC/BTC', 'ADA/BTC', 'DOT/BTC', 'LTC/BTC']
s = ['ETH/BTC']

search(hb, s)



#from harmo import search
search(hb, ['HT/USDT'])
search(ok, ['OKB/USDT'])
search(bn, ['BNB/USDT'])


s = ['UNI/USDT', 'XMR/USDT', 'ATOM/USDT', 'COMP/USDT', 'SOL/USDT', 'ALGO/USDT', 'FIL/USDT', 'XLM/USDT', 'AAVE/USDT']
search(hb, s, periods=['15m','30m', '1h', '4h'], predict=True, only_last=True,plot=True)
#search(hb, s, periods=['5m'], predict=True, only_last=True)



search(hb, ['BTC/USDT', 'ETH/USDT', 'LTC/USDT'], periods=['5m', '15m', '30m', '1h', '4h'],predict=True) 

#%%

search(hb, ['ETH/USDT','ATOM/USDT', 'UNI/USDT','ADA/USDT', 'ETC/USDT'], periods=['5m', '15m', '30m', '1h', '4h'], plot=True, only_last=True) 

#%%

search(hb, ['FIL/USDT', ], periods=['5m', '15m', '30m', '1h', '4h'], plot=True, predict=True) 

#%% md

### LOOP

#%%


s = ['BTC/USDT', 'ETH/USDT', 'BCH/USDT', 'ETC/USDT', 'ADA/USDT','XRP/USDT', 'DOT/USDT', 'EOS/USDT', 'LTC/USDT']
s = ['UNI/USDT', 'XMR/USDT', 'ATOM/USDT', 'COMP/USDT', 'SOL/USDT', 'ALGO/USDT', 'FIL/USDT', 'XLM/USDT', 'AAVE/USDT', *s]

#%%

search(hb, ['ETC/USDT', ], periods=['5m', '15m', '30m', '1h', '4h'], plot=True, predict=True, only_last=True) 

#%%

periods = ['5m','15m','30m','1h','4h']
with Pool(8) as p:
    r = p.map(partial(search, hb, ['XMR/USDT'], predict=True, only_last=True, alert=False, plot=False), [[pi] for pi in periods])

#%%

with Pool(8) as p:
    r = p.map_async(partial(search, bn, periods=['5m','15m','30m','1h','4h'], predict=True, only_last=True, alert=False, plot=False), [[si] for si in s])
    r.get(timeout=240)

#%%

