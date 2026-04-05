# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:23:07 2019

@author: adair
"""

import time
import os,sys
import pandas as pd
from sqlalchemy import create_engine


engine = create_engine('mysql+pymysql://root:liudehua@localhost:3306/YCZfenbi?charset=utf8')

def get_file_name(file_dir,file_type):
    L=[]   
    info=[]
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == file_type:  
                L.append(os.path.join(root, file))
                temp=file.split(' ')
                info.append(temp[1])                
    return L, info

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
	#drop duplicates
    x.drop_duplicates(subset=['tradingdate','symbol'],keep='first',inplace=True)
    #write to table
    x.to_sql(table_name,engine,if_exists='append',index=False)
    #print('%s-%s' % (table_name,info))
    return x,table_name,info

#å¦ä½å¹¶è¡è®¡ç®
    
if __name__ == '__main__':
    t0 = time.time()
    print(sys.argv)
    if len(sys.argv) == 2:
        id = int(sys.argv[1])
    else:
        id =0
    print('%s' % id)
    file_dir = 'D:\\datasets\\YCZ\\fenbishuju\\2010'
    fns,info = get_file_name(file_dir,'.csv')
    temp = pd.DataFrame({'a':fns,'b':info})
    temp.sort_values(by='b',inplace=True)
    fns=temp['a'].tolist()
    T = len(fns)
    cut = 8
    while id < T:
        _,tb_name,info=read_fenbi_data(fns[id])
        print('Complete : %s-%s %d-%d' % (tb_name,info,id,T))
        id= id+cut

