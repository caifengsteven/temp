# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:18:51 2020
金数源tick 数据
@author: adair
"""


#将预测者的分钟数据写入数据库
import pandas as pd
from sqlalchemy import create_engine
import os
import pymysql
from zipfile import ZipFile
import datetime
import json

#must be set before using
with open('para.json','r',encoding='utf-8') as f:
    para = json.load(f)
    
pn = para['yuqerdata_dir']

user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name_jsy = 'Future_tick'
eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name_jsy)
engine_jsy = create_engine(eng_str)


file_dir = r'E:\datasets\Jingyuan\Jingyuan\future_tick5'


def get_file_name(file_dir,file_type):
    L=[]
    L_s = []   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(root, file))  
                L_s.append(file)
    return L,L_s

def create_table(db_name,tn_name,var_name,var_type,key_str=[]):
    #连接本地数据库
    db = pymysql.connect("localhost",user_name,pass_wd,db_name)

    #创建游标
    cursor = db.cursor()

    #创建
    var_info=''
    for id,sub_var in enumerate(var_name):
        var_info=var_info + sub_var + ' ' + var_type[id] + ','
    var_info = var_info[:-1]    
    if len(key_str)>0:
        sql = 'create table  `%s`(%s,primary key(%s))' % (tn_name,var_info,key_str)
    else:
        sql = 'create table  `%s`(%s)' % (tn_name,var_info)
    
    try:
        # 执行SQL语句
        cursor.execute(sql)
        print("创建数据库成功")
    except Exception as e:
        print("创建数据库失败：case%s"%e)
    finally:
        #关闭游标连接
        cursor.close()
        # 关闭数据库连接
        db.close()

#起始时间
t0=pd.read_sql('show tables',engine_jsy)
t0=t0.Tables_in_future_tick.max()
t0=int(t0[1:])

fns_full,fns = get_file_name(file_dir,'.csv')
fns_L = [0 if '连续' not in sub_fns else 1 for sub_fns in fns ]
T = len(fns_L)

fns_full_sel = [fns_full[i] for i in  range(T) if fns_L[i]==0]
fns_sel = [fns[i] for i in  range(T) if fns_L[i]==0]
fns_t0 = [int(x[-12:-6]) for x in fns_sel]

#去重
fns_s = []
fns_f = []
fns_t = []
for i,temp_str in enumerate(fns_sel):
    if temp_str not in fns_s:
        fns_s.append(fns_sel[i])
        fns_f.append(fns_full_sel[i])
        fns_t.append(fns_t0[i])
#找到所有的未写的数据
T = len(fns_s)
fns_L = [0 if t0<sub_t else 1 for sub_t in fns_t]
fns_s1 = [fns_s[i] for i in  range(T) if fns_L[i]==0]        
fns_f1 = [fns_f[i] for i in  range(T) if fns_L[i]==0]  
fns_t1 = [fns_t[i] for i in  range(T) if fns_L[i]==0]  

#获取所有表格时间，创建表格
t_u = []
for temp_str in fns_t1:
    if temp_str not in t_u:
        t_u.append(temp_str)

var_name = ['marketcode', 'ticker', 'tradingdate', 'infoNow', 'holdInt',
       'increaseInt', 'turnoverValue', 'turnoverVol', 'openposition',
       'closeposition', 'type1', 'bsdirection', 'Buy1', 'Buy2', 'Buy3', 'Buy4',
       'Buy5', 'Sail1', 'Sail2', 'Sail3', 'Sail4', 'Sail5', 'BV1', 'BV2',
       'BV3', 'BV4', 'BV5', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5']
var_type = []
for i in range(len(var_name)):
    if i in [0,1,8,9,10,11]:
        var_type.append('varchar(12)')
    elif i==2:
        var_type.append('datetime')
    else:
        var_type.append('float')


for sub_t in t_u:
    sub_tn = 'y%d' % sub_t
    create_table(db_name_jsy,sub_tn,var_name,var_type)
    
T = len(fns_s1)
for i,sub_fn in enumerate(fns_f1):
    table_name = 'y%d' % fns_t1[i]
    x = pd.read_csv(sub_fn,header=0,engine='python',encoding='GBK')
    x.rename(columns={'市场代码':'marketcode', '合约代码':'ticker', '时间':'tradingdate', 
                      '最新':'infoNow', '持仓':'holdInt', '增仓':'increaseInt',
                      '成交额':'turnoverValue', '成交量':'turnoverVol', '开仓':'openposition', 
                      '平仓':'closeposition','成交类型':'type1', '方向':'bsdirection',
                      '买一价':'Buy1', '买二价':'Buy2', '买三价':'Buy3', '买四价':'Buy4',
                      '买五价':'Buy5', '卖一价':'Sail1', '卖二价':'Sail2', '卖三价':'Sail3',
                      '卖四价':'Sail4', '卖五价':'Sail5', '买一量':'BV1', '买二量':'BV2',
                      '买三量':'BV3', '买四量':'BV4', '买五量':'BV5', '卖一量':'SV1', 
                      '卖二量':'SV2', '卖三量':'SV3','卖四量':'SV4', '卖五量':'SV5'},inplace=True)
    x.to_sql(table_name,engine_jsy,if_exists='append',index=False,chunksize=3000)
    print('Complete ( %d-%d )' % (i,T))
