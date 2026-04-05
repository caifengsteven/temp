# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:18:34 2019
#时间,价格,笔数,成交额,成交量,买卖盘,买一价,买二价,买三价,买四价,买五价,卖一价,
    #卖二价,卖三价,卖四价,卖五价,买一量,买二量,买三量,买四量,买五量,卖一量,卖二量,卖三量,卖四量,卖五量
@author: adair2019
"""

import os
import sys
import pandas as pd
import time

def read_fenbi_data(fn):
    _,info=os.path.split(fn)
    info = info.split(' ')
    table_name = info[0]
    info = info[1]
    
    x = pd.read_csv(fn,header=1,encoding='GBK');
    x.rename(columns={'时间':'tradingdate', '价格':'price','笔数':'dealAmount',
                      '成交额':'turnoverValue','成交量':'volume','买卖盘':'BSsel',
                      '买一价':'BP1','买二价':'BP2','买三价':'BP3','买四价':'BP4',
                      '买五价':'BP5','卖一价':'BS1','卖二价':'BS2','卖三价':'BS3',
                      '卖四价':'BS4','卖五价':'BS5','买一量':'BV1','买二量':'BV2',
                      '买三量':'BV3','买四量':'BV4','买五量':'BV5','卖一量':'SV1',
                      '卖二量':'SV2','卖三量':'SV3','卖四量':'SV4','卖五量':'SV5'}, inplace = True)
    x['symbol'] = info
    return x,table_name,info    

if __name__ == '__main__':
    t0 = time.time()
    print(sys.argv)
    if len(sys.argv) == 2:
        fn = sys.argv[1]
    else:
        fn ='D:\\datasets\\YCZ\\fenbishuju\\2010\\1\\2010-01-04\\2010-01-04 sh600000 fenbi.csv'
        
    if os.path.exists(fn):
        x,tb_name,info=read_fenbi_data(fn)
        #insert into mysql
        print('Complete : %s-%s' % (tb_name,info))
    else:
        print('no this file %s' % fn)
    tt = time.time()
    print('time Used: %s' % (tt-t0))

