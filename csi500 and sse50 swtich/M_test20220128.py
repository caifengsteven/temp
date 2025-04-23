# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:52:50 2022
如果在t0结束，上证50的结果好过中证500，就以t0的收盘价买入中证500，卖出上证50.
如果在t0结束，上证50的结果差过中证500，就以t0的收盘价卖出入中证500，买出上证50.

MktIdxdGet(index,begin,end,var_name = '*')
@author: adair-9960
"""
from yq_toolsS45_linux import MktIdxdGet
fee = 3/10000*2

indexID = ['000905','000016']

t0 = '2010-01-01'
tt = '2099-01-01'

var_key = 'tradeDate,CHGPct'
x = []
for ticker in indexID:
    tmp = MktIdxdGet(ticker,t0,tt,var_key)
    tmp.rename(columns = {'CHGPct':ticker},inplace=True)
    x.append(tmp)

x = x[0].merge(x[1],on='tradeDate')

x['sig1'] = x['000905']<x['000016']
x['sig2'] = x['000905']>=x['000016']
x['r1'] = x.sig1.shift(1)*x['000016']
x['r2'] = x.sig2.shift(1)*x['000905']
x['r'] = x['r1']+x['r2']

x['fee_sig'] = x.sig1 != x.sig1.shift(1)
x['fee'] = x['fee_sig']*fee

x.set_index('tradeDate',inplace=True)
(1+x.r-x.fee).cumprod().plot()