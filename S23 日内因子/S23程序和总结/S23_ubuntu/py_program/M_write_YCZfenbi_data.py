# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:23:07 2019

@author: adair
"""

import dill as pickle
import time
import os
import pandas as pd
from sqlalchemy import create_engine
from gplearn.utils import _partition_estimators
from joblib import Parallel, delayed
import itertools

engine = create_engine('mysql+pymysql://root:liudehua@localhost:3306/YCZfenbi?charset=utf8')

def get_file_name(file_dir,file_type):
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(root, file))  
    return L

def read_fenbi_data(fn):
    _,info=os.path.split(fn)
    info = info.split(' ')
    table_name = info[0]
    info = info[1]
    
    x = pd.read_csv(fn,header=1,encoding='GBK',engine='python');
    x.rename(columns={'时间':'tradingdate', '价格':'price','笔数':'dealAmount',
                      '成交额':'turnoverValue','成交量':'volume','买卖盘':'BSsel',
                      '买一价':'BP1','买二价':'BP2','买三价':'BP3','买四价':'BP4',
                      '买五价':'BP5','卖一价':'BS1','卖二价':'BS2','卖三价':'BS3',
                      '卖四价':'BS4','卖五价':'BS5','买一量':'BV1','买二量':'BV2',
                      '买三量':'BV3','买四量':'BV4','买五量':'BV5','卖一量':'SV1',
                      '卖二量':'SV2','卖三量':'SV3','卖四量':'SV4','卖五量':'SV5'}, inplace = True)
    x['symbol'] = info[2:]
    table_name = table_name.replace('-','')
    #write to table
    x.to_sql(table_name,engine,if_exists='append',index=False)
    return x,table_name,info

def _parallel_excute(fns):
    t_re = []
    for fn in fns:
        sub_t = False
        try:
            x,table_name,info = read_fenbi_data(fn)
            sub_t = True
        except Exception as e:
            print(e)
        print('%s-%s' % (table_name,info))
        t_re.append(sub_t)
    return t_re
#如何并行计算
file_dir = 'F:\\works2018\\some\\datasets\\YCZ\\2012'
fns = get_file_name(file_dir,'.csv')


t0 = time.time()
n_jobs, _, starts = _partition_estimators(20, 4)
all_t = Parallel(n_jobs=n_jobs,
                      verbose=False)(
    delayed(_parallel_excute)(fns[starts[i]:starts[i + 1]])
    for i in range(n_jobs))

y = list(itertools.chain.from_iterable(all_t))
tt = time.time()
print(tt-t0)
