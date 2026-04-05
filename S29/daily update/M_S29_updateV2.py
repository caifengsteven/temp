# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:39:36 2020
财务数据特点

时不时更新，不定时更新
我们数据也定时更新
@author: adair-9960

和原有程序尽量保持一致
conda install mysql-connector-python
"""

import os
from sqlalchemy import create_engine
import pandas as pd
import pymysql


user_name = 'root'
pass_wd = '352471Cf'
port = 3306
db_name = 'S29'
tn_name = 'factor_yuqer'
#eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str='mysql+mysqlconnector://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
engine = create_engine(eng_str)

tns_exist = pd.read_sql('show tables from %s' % db_name,engine)
tns_exist=tns_exist.iloc[:,0].tolist()

#must be set before using
pn = r"G:\dropbox\Dropbox\Dropbox\project folder from my asua computer\Project\S29\daily update\S29F07"

def get_file_name(file_dir,file_type):
    L=[]
    L_s = []   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(root, file))  
                L_s.append(file)
    return L,L_s

def add_0(x):
    x= '%0.6d' % x
    return x

def do_sql_order(db_name,order_str):
    db = pymysql.connect("localhost",user_name,pass_wd,db_name)
    #创建游标
    cursor = db.cursor()
    try:
        # 执行SQL语句
        cursor.execute(order_str)
        print("执行mysql命令成功")
    except Exception as e:
        print("执行mysql命令失败：case%s"%e)
    finally:
        #关闭游标连接
        cursor.close()
        # 关闭数据库连接
        db.close()
    
def create_table(db_name,tn_name,var_name,var_type,key_str=None):
    #连接本地数据库
    db = pymysql.connect("localhost",user_name,pass_wd,db_name)

    #创建游标
    cursor = db.cursor()

    #创建
    var_info=''
    for id,sub_var in enumerate(var_name):
        var_info=var_info + sub_var + ' ' + var_type[id] + ','
    var_info = var_info[:-1]    
    if key_str is None:
        sql = 'create table  `%s`(%s)' % (tn_name,var_info)
    else:            
        sql = 'create table  `%s`(%s,primary key(%s))' % (tn_name,var_info,key_str)
    
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
        
"""
fns,_ = get_file_name(pn,'.zip')
_,fns_csv = get_file_name(pn,'.csv')
for sub_fn in fns:
    f = zipfile.ZipFile(sub_fn,'r')
    for file in f.namelist():
        if file not in fns_csv:
            f.extract(file,pn)
"""        
_,fns_csv = get_file_name(pn,'.csv')

fns_data = [];
for i in range(30):
    fns_data.append([])

for sub_fn in fns_csv:
    if 'S29F01' in sub_fn:
        fns_data[0] = os.path.join(pn,sub_fn)
    elif 'S29F02' in sub_fn:
        fns_data[1] = os.path.join(pn,sub_fn)
    elif 'S29F05' in sub_fn:
        fns_data[2] = os.path.join(pn,sub_fn)
    elif 'S29F06' in sub_fn:
        fns_data[3] = os.path.join(pn,sub_fn)
    elif 'S29F07' in sub_fn:
        fns_data[4] = os.path.join(pn,sub_fn)
    elif 'S29F10' in sub_fn:
        fns_data[5] = os.path.join(pn,sub_fn)
    elif 'S29F15' in sub_fn:
        fns_data[6] = os.path.join(pn,sub_fn)
    
var_name=['factor_name','pub_date', 'symbol', 'f_val']
var_type=['varchar(8)','date','varchar(8)','float']
id_attention = ['f5','f7','f9']
create_table(db_name,tn_name,var_name,var_type,'%s,%s,%s' % (var_name[0],var_name[1],var_name[2]))



date_cols = ['endDate']
for id in range(7):
    x = pd.read_csv(fns_data[id],dtype={'ticker':str}, parse_dates=date_cols)
    os.remove(fns_data[id])
    if 'publishDate' in x.columns.tolist():
        x.drop(columns=['publishDate'],inplace=True)
    code0 = ['ticker', 'endDate']
    code1 = x.columns.values.tolist()
    code2 = set(code1).difference(set(code0))
    #write data to mysql
    for sub_key in code2:
        sub_key1 = code0.copy()
        sub_key1.append(sub_key)
        sub_x = x[sub_key1].copy()
        sub_x.rename(columns={sub_key:'f_val','ticker':'symbol','endDate':'pub_date'},
                     inplace=True)
        sub_x.dropna(inplace=True)
        #insert to database
        if sub_key in id_attention:
            sub_key2 = '-' + sub_key
        else:
            sub_key2 = sub_key
        sub_x[var_name[0]] =sub_key2 
        do_sql_order(db_name,('delete from %s where factor_name="%s"' % (tn_name,sub_key2)))
        
        sub_x.to_sql(tn_name,engine,if_exists='append',index=False,chunksize=3000)

