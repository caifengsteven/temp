# -*- coding: utf-8 -*-
"""
DataAPI.RMExposureDayGet
函数接口，来获取股票的风格因子和行业因子暴露

行业因子：申万一级行业分类
风格因子：市值，Beta，动量，估值，盈利，成长，杠杆，波动，非线性市值，流动性

@author: adair2019
"""

#中性化需要
import pandas as pd
#import numpy as np
#import statsmodels.api as sm
#from statsmodels.sandbox.rls import RLS

#数据库
from sqlalchemy import create_engine
import json

import warnings
warnings.filterwarnings('ignore')

#must be set before using
with open('para.json','r',encoding='utf-8') as f:
    para = json.load(f)
    
user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name1 = 'yuqerdata'
eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name1)
engine = create_engine(eng_str)

db_name_yq_cub = 'yuqer_cubdata_update'
eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name_yq_cub)
engine_yq_cub = create_engine(eng_str)
#获取行业分类
def get_industry_class(t):
    sql_str1 = '''select ticker,industryID1 from yuqerdata.yq_industry where 
                industryVersionCD="010303" and intodate <= "%s" and 
                (outDate>"%s" or outDate is null)''' % (t,t)
    x = pd.read_sql(sql_str1,engine)
    return x

def RMExposureDayGet(target_date='2018-12-04'):
    indus = get_industry_class(target_date)
    indus.rename(columns={'industryID1':'classname'},inplace= True)
    indus.index = indus['ticker']
    indus = indus['classname']
    class_var = pd.get_dummies(indus,columns=['classname'],
                                   prefix='class',prefix_sep="_", dummy_na=False, drop_first = False)
    #风格因子
    x=pd.DataFrame()
    factor_style10 = ['Beta252','Rstr504','LCAP','EPIBS','DASTD','SGRO','PB','MLEV','STOQ','NLSIZE']
    factor_style10_str = 'select symbol,f_val from %s where tradingdate ="%s"'
    tns = pd.read_sql('show tables from %s'  % db_name_yq_cub,engine_yq_cub)
    tns = tns.Tables_in_yuqer_cubdata_update.tolist()
    for i,sub_f_name in enumerate(factor_style10):
        if sub_f_name.lower() not in tns:
            continue
        sub_sql_str = factor_style10_str % (sub_f_name,target_date)
        sub_x = pd.read_sql(sub_sql_str,engine_yq_cub)
        if len(sub_x)==0:
            continue
        if len(x)==0:
            x = sub_x
        else:
            x = pd.merge(x,sub_x,on='symbol',how = 'inner')
            
        print(i)
    if len(x)==0:
        return pd.DataFrame()
    else:
        x.rename(columns={'symbol':'ticker'},inplace= True)
        x = pd.merge(x,class_var,on='ticker',how = 'inner')   
        return x

x = RMExposureDayGet(target_date = '2018-01-05')
