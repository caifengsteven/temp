# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:37:02 2020

@author: adair2019
"""

#import zipfile
import os
from sqlalchemy import create_engine
import pandas as pd
import pymysql
import time

t_max=time.strftime("%Y-%m-%d", time.localtime())
pn = r"D:\worksPool\works2020\SOME\yuqerdata\data"
#must be set 用户名，密码，端口
user_name = 'root'
pass_wd = 'liudehua'
port = 3306
db_name1 = 'yuqerdata'
tn_name = 'factor_yuqer'
#eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str='mysql+mysqlconnector://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name1)
engine = create_engine(eng_str)

db_name3 = 'gtadata'
eng_str='mysql+mysqlconnector://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name3)
engine3 = create_engine(eng_str)

#tt_str0 = '2019-01-01'

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
    if isinstance(x,int):
        x= '%0.6d' % x
    return x

def do_sql_order(order_str,db_name):
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
sql_str = 'delete from %s where %s >="%s"'
#must be set before using


_,fns_csv = get_file_name(pn,'.csv')
_,fns_csv1 = get_file_name(pn,'.txt')
for i in fns_csv1:
    fns_csv.append(i)

fns_data = []
for i in range(20):
    fns_data.append([])

for sub_fn in fns_csv:
    if 'EquRestructuringGet' in sub_fn:
        fns_data[0] = os.path.join(pn,sub_fn)
    elif 'FdmtISGet' in sub_fn:
        fns_data[1] = os.path.join(pn,sub_fn)
    elif 'FdmtBSGet' in sub_fn:
        fns_data[2] = os.path.join(pn,sub_fn)
    elif 'FdmtMainOperNGet' in sub_fn:
        fns_data[3] = os.path.join(pn,sub_fn)
    elif 'FdmtEeGet' in sub_fn:
        fns_data[4] = os.path.join(pn,sub_fn)
    elif 'MktStockFactorsOneDayGet' in sub_fn and 'MktStockFactorsOneDayGet_add' not in sub_fn:
        fns_data[5] = os.path.join(pn,sub_fn)
    elif 'FdmtDerPitGet' in sub_fn:
        fns_data[6] = os.path.join(pn,sub_fn)
    elif 'MktStockFactorsOneDayGet_add' in sub_fn:
        fns_data[7] = os.path.join(pn,sub_fn)
    elif 'FdmtIndiTrnovrPitGet' in sub_fn:
        fns_data[8] = os.path.join(pn,sub_fn)
    elif 'FAR_Finidx' in sub_fn:
        fns_data[9] = os.path.join(pn,sub_fn)
          
#1 EquRestructuringGet 重组数据
data_id = 0
if len(fns_data[data_id])>0:
    x1 = pd.read_csv(fns_data[0],header=0,engine='python',encoding = "utf-8",
                     dtype={'ticker':str})
    #x1['ticker'] = x1['ticker'].apply(add_0)
    table_name = 'EquRestructuringGet'
    sql_str1 = 'select publishDate from %s where publishDate <"%s" order by publishDate desc limit 1' % (table_name,t_max)
    t = pd.read_sql(sql_str1,engine)
    info = '重组数据'
    if not t.empty:
        tt_str = str(t.publishDate[0])
    else:
        tt_str = '1990-01-01'
    
    do_sql_order(sql_str % (table_name,'publishDate',tt_str),db_name1)
    x2 = x1.loc[x1.publishDate>=tt_str]
    if not x2.empty:
        y=x2[['secID','ticker','secShortName','exchangeCD','publishDate',
              'iniPublishDate','finPublishDate','program','isSucceed','restructuringType',
              'underlyingType','underlyingVal','expenseVal','isRelevance','isMajorRes','payType']]
        y.to_sql(table_name,engine,if_exists='append',index=False,chunksize=3000)
        print('%s:%s' % (info,y.publishDate.max()))
    else:
        print('%s已经是最新的%s，无需更新' % (info,tt_str))
    os.remove(fns_data[data_id])
#
data_id = 1    
if len(fns_data[data_id])>0:
    x1 = pd.read_csv(fns_data[data_id],header=0,engine='python',encoding = "utf-8",
                     dtype={'ticker':str})
    #x1['ticker'] = x1['ticker'].apply(add_0)
    table_name = 'nincome'
    sql_str1 = 'select publishDate from %s order by publishDate desc limit 1' % table_name
    t = pd.read_sql(sql_str1,engine)
    info = '合并利润表'
    if not t.empty:
        tt_str = str(t.publishDate[0])
    else:
        tt_str = '1990-01-01'
    do_sql_order(sql_str % (table_name,'publishDate',tt_str),db_name1)
    x2 = x1.loc[x1.publishDate>=tt_str]
    if not x2.empty:    
        x2.to_sql(table_name,engine,if_exists='append',index=False,chunksize=3000)
        print('%s:%s' % (info,x2.publishDate.max()))
    else:
        print('%s已经是最新的%s，无需更新' % (info,tt_str))
    os.remove(fns_data[data_id])
#
data_id=2
if len(fns_data[data_id])>0:
    x1 = pd.read_csv(fns_data[data_id],header=0,engine='python',encoding = "utf-8",
                     dtype={'ticker':str})
    #x1['ticker'] = x1['ticker'].apply(add_0)
    table_name = 'yq_FdmtBSGet'
    sql_str1 = 'select publishDate from %s order by publishDate desc limit 1' % table_name
    t = pd.read_sql(sql_str1,engine)
    info = '合并资产负债表'
    if not t.empty:
        tt_str = str(t.publishDate[0])
    else:
        tt_str = '1990-01-01'
    do_sql_order(sql_str % (table_name,'publishDate',tt_str),db_name1)
    x2 = x1.loc[x1.publishDate>=tt_str]
    if not x2.empty:    
        x2.to_sql(table_name,engine,if_exists='append',index=False,chunksize=3000)
        print('%s:%s' % (info,x2.publishDate.max()))
    else:
        print('%s已经是最新的%s，无需更新' % (info,tt_str))
    os.remove(fns_data[data_id])
    
data_id=3
if len(fns_data[data_id])>0:
    x1 = pd.read_csv(fns_data[data_id],header=0,engine='python',encoding = "utf-8",
                     dtype={'ticker':str})
    #x1['ticker'] = x1['ticker'].apply(add_0)
    table_name = 'yq_FdmtMainOperNGet_update'
    sql_str1 = 'select publishDate from %s order by publishDate desc limit 1' % table_name
    t = pd.read_sql(sql_str1,engine)
    info = '主营业务构成'
    if not t.empty:
        tt_str = str(t.publishDate[0])
    else:
        tt_str = '1990-01-01'
    do_sql_order(sql_str % (table_name,'publishDate',tt_str),db_name1)
    x2 = x1.loc[x1.publishDate>=tt_str]
    if not x2.empty:    
        x2.to_sql(table_name,engine,if_exists='append',index=False,chunksize=3000)
        print('%s:%s' % (info,x2.publishDate.max()))
    else:
        print('%s已经是最新的%s，无需更新' % (info,tt_str))
    os.remove(fns_data[data_id])

data_id = 6    
if len(fns_data[data_id])>0:
    x1 = pd.read_csv(fns_data[data_id],header=0,engine='python',encoding = "utf-8",
                     dtype={'ticker':str})
    #x1['ticker'] = x1['ticker'].apply(add_0)
    x1.rename(columns={'ticker':'symbol'},inplace=True)
    table_name = 'yq_FdmtDerPitGet'
    sql_str1 = 'select publishDate from %s order by publishDate desc limit 1' % table_name
    t = pd.read_sql(sql_str1,engine)
    info = '财务衍生数据'
    if not t.empty:
        tt_str = str(t.publishDate[0])
    else:
        tt_str = '1990-01-01'
    do_sql_order(sql_str % (table_name,'publishDate',tt_str),db_name1)
    x2 = x1.loc[x1.publishDate>=tt_str]
    if not x2.empty:    
        x2.to_sql(table_name,engine,if_exists='append',index=False,chunksize=3000)
        print('%s:%s' % (info,x2.publishDate.max()))
    else:
        print('%s已经是最新的%s，无需更新' % (info,tt_str))
    os.remove(fns_data[data_id])
#
data_id = 8    
if len(fns_data[data_id])>0:
    x1 = pd.read_csv(fns_data[data_id],header=0,engine='python',encoding = "utf-8",
                     dtype={'ticker':str})
    #x1['ticker'] = x1['ticker'].apply(add_0)
    x1.rename(columns={'ticker':'symbol'},inplace=True)
    table_name = 'yq_FdmtIndiTrnovrPitGet'
    sql_str1 = 'select publishDate from %s order by publishDate desc limit 1' % table_name
    t = pd.read_sql(sql_str1,engine)
    info = '财务指标-运营能力'
    if not t.empty:
        tt_str = str(t.publishDate[0])
    else:
        tt_str = '1990-01-01'
    do_sql_order(sql_str % (table_name,'publishDate',tt_str),db_name1)
    x2 = x1.loc[x1.publishDate>=tt_str]
    if not x2.empty:    
        x2.to_sql(table_name,engine,if_exists='append',index=False,chunksize=3000)
        print('%s:%s' % (info,x2.publishDate.max()))
    else:
        print('%s已经是最新的%s，无需更新' % (info,tt_str))
    os.remove(fns_data[data_id]) 
#
data_id = 9    
if len(fns_data[data_id])>0:
    x1 = pd.read_csv(fns_data[data_id],sep='\t',header=0,engine='python',encoding='utf-16')
    x1['Stkcd'] = x1['Stkcd'].apply(add_0)
    table_name = 'FAR_Finidx'
    sql_str1 = 'select Annodt from %s order by Annodt desc limit 1' % table_name
    t = pd.read_sql(sql_str1,engine3)
    info = '财务指标-运营能力'
    if not t.empty:
        tt_str = str(t.Annodt[0])
    else:
        tt_str = '1990-01-01'
    do_sql_order(sql_str % (table_name,'Annodt',tt_str),db_name3)
    x2 = x1.loc[x1.Annodt>=tt_str]
    if not x2.empty:    
        x2.to_sql(table_name,engine3,if_exists='append',index=False,chunksize=3000)
        print('%s:%s' % (info,x2.Annodt.max()))
    else:
        print('%s已经是最新的%s，无需更新' % (info,tt_str))
    os.remove(fns_data[data_id]) 