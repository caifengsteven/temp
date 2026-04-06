# coding: utf-8
'''
双底策略，补充结果
20210509
1）美国股票 SPX
2）香港股票 HSI
3）外汇日线和分钟线 

升级
记录信号位置，为后续寻找信号做准备

'''

from yq_toolsS45 import get_us_daytick_adj
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#import statsmodels.api as sm
#import math
from yq_toolsS45 import save_pickle
from yq_toolsS45 import table_in_database
from yq_tools import chg_factor as chs_factor #载入日线数据
#from yq_tools import get_IdxCons as IdxConsGet #载入成分股数据
from yq_tools import get_index_tradeDate as MktIdxdGet #载入指数数据
from yq_tools import get_symbol_A
from yq_tools import create_table_update
from datetime import datetime
from yq_toolsS45 import get_spx_com
import multiprocessing
num_core2 = int(multiprocessing.cpu_count()/2)
from yq_toolsS45 import create_db

engine37 = create_db('s37','localhost')
engine = create_db('yuqerdata','localhost')
eg_plg = create_db('polygon')
eg_pro = create_db('data_pro','localhost')
tn = 's43_addre'
para_sel = False


def get_points(n, index_down, index_up, s_ori ,back_w = 1):
    #print('s_ori',s_ori)
    qsdu_e = index_up[n]
    point_b, point_d = index_down[n - 3 :n - 1]
    point_a, point_c = index_up[n - 3 :n - 1]
    point_a, point_c = s_ori.loc[point_a -back_w - 1 : point_a + back_w].idxmax(), s_ori.loc[point_c -back_w - 1: point_c + back_w].idxmax() 
    point_b, point_d = s_ori.loc[point_b -back_w - 1: point_b + back_w].idxmin(), s_ori.loc[point_d -back_w - 1: point_d + back_w].idxmin()
    qsdu_e = s_ori.loc[qsdu_e - back_w - 1: qsdu_e + back_w].idxmax()
    #pre_a = index_down[n - 4]
    return point_a, point_b, point_c, point_d, qsdu_e


#加上前期30%的这个条件，找不到信号
def is_bottom(a, b, c, d, e, s_ori, s_high, s_vol):
    ## 前期30% 涨幅
    ## 给定一个时间，在时间内与最低点差30%即可
    #pre_day = a - 200
    #r_pre = s_ori[a] / s_ori[pre_day: a].min() - 1
    #cond1 = r_pre > 0.3  ## bug
    ## a点高于c点 第一个高点大于第二个高点
    cond2 = s_high[a] > s_high[c]
    ## d低于b点 第一个低点大于第二个低点
    cond3 = s_ori[b] > s_ori[d]
    ## ad之间的限制  下降50%以内
    cond5 = s_ori[a] / s_ori[d] - 1 < 0.5
    ## 判断是否出现突破
    ## 其中e点为伪e 第三个高点大于第二个高点
    cond4 = s_ori[e] > s_ori[c]
    if cond2 and cond3 and cond4 and cond5:
        ## 判断e点
        ## 最高点
        e_cond1 = s_high.loc[d:e] > s_high[c]
        e_supply = np.arange(d, e+1)[e_cond1.values]
        # 判断成交量
        # 成交量判断无效
        for i in e_supply:
            if s_vol[i] > s_vol.loc[c:i].mean()*1.2:
                e = i
                return [a, b, c, d], e
                break

def get_stock_signal(df , window = 5, is_plot = True):
    ### 数据获取
    s_ori = df['closePrice'] * df['accumAdjFactor']
    s_high = df['highestPrice'] * df['accumAdjFactor']
    s_vol = df['turnoverVol']
    ## 去噪过程
    trend = s_ori.rolling(window).mean()
    back_w = int(window / 2.0)
    # trend.plot()
    ## 获取去噪之后的高低点
    index_down = signal.find_peaks(-trend)[0]
    index_up = signal.find_peaks(trend)[0]
    index_down = index_down[index_down > index_up[0]]
    ## 获取信号
    e_list = []
    lenth_w = []
    for n in range(5, index_up.shape[0]):
        a,b,c,d,qsdu_e = get_points(n, index_down, index_up, s_ori, back_w)
        if is_bottom(a, b, c, d, qsdu_e, s_ori, s_high, s_vol):
            [a, b ,c, d] , e = is_bottom(a, b, c, d, qsdu_e, s_ori, s_high, s_vol)
            e_list.append(e)
            lenth_w.append(e - a)
    if is_plot:
        s_ori.plot()
        plt.plot(e_list, s_ori[e_list], 'r*')
    return np.array(lenth_w), np.array(e_list)

## 统计盈利
def stat_ratio(e_list, df, n):
    
    tmp_ind = e_list+n
    tmp_ind = tmp_ind[tmp_ind<len(df)]
    tmp_ind = tmp_ind-n
    
    return df.loc[tmp_ind + n]['closePrice'].values / df.loc[tmp_ind + 1]['openPrice'].values - 1
## 统计到df
def stock_df(df, e_list, lenth_w, n_list = [5, 10, 15, 30]):
    df_1 = df.loc[e_list][['tradeDate', 'ticker']]
    ratio_list = [stat_ratio(e_list, df, n) for n in [5, 10, 15, 30]]
    df_new = pd.DataFrame(ratio_list + [lenth_w], index = ['r_5', 'r_10', 'r_15','r_30', 'lenth']).T
    df_1.index = df_new.index
    return pd.concat([df_1, df_new], axis = 1)

def back_test(df, e_list, hold_days = 5, is_plot = True):
    e_signal = np.hstack([e_list + i for i in range(1, hold_days + 1)])
    e_signal = np.unique(e_signal)
    e_signal = e_signal[e_signal < df.shape[0]]
    e_signal.sort()
    v = np.zeros(df.shape[0])
    v[e_signal] = 1
    
    if is_plot:
        r = ((v * df['chgPct']) + 1).cumprod()
        r.plot()
        (df['chgPct'] + 1).cumprod().plot()
        plt.show()

    return (v * df['chgPct']) + 1

def back_test_more(df, e_list, hold_days = [5, 7, 10 ,15, 20, 30, 35, 60]):
    df_back = pd.DataFrame([back_test(df, e_list, day, is_plot= False) for day in hold_days], index= ['r_%s'%i for i in hold_days]).T
    df_back['tradeDate'] = df['tradeDate']
    df_back['ticker'] = df['ticker']
    return df_back

def get_single_s43(x):
    ticker,begin,end,dtype=x    
    if dtype=='csi':
        df = chs_factor(ticker= ticker ,begin = begin ,end = end)
    elif dtype=="US":
        df = get_us_daytick_adj(ticker,begin,end)
        df = df[['ticker','tradeDate','openPrice_adj',
       'closePrice_adj', 'highPrice_adj', 'lowPrice_adj','turnoverVol']]
        df.columns = ['ticker','tradeDate','openPrice',
       'closePrice', 'highestPrice', 'lowestPrice','turnoverVol']
        df['chgPct'] = df.closePrice.pct_change()
        df.chgPct.fillna(0,inplace=True)
        df['accumAdjFactor'] = 1
    elif dtype == "HK":
        sql_tmp = '''select * from mkthkequdgets54 where ticker = "%s" 
        and tradeDate>="%s" and tradeDate<="%s" and closePrice is not null order by tradeDate'''
        df = pd.read_sql(sql_tmp % (ticker,begin,end),engine)
        df.chgPct.fillna(0,inplace=True)
        df['accumAdjFactor'] = 1
    elif dtype == "forex_day":
        sql_tmp = '''select * from forex_day where ticker = "%s" 
        and tradeDate>="%s" and tradeDate<="%s" and closePrice is not null order by tradeDate'''
        df = pd.read_sql(sql_tmp % (ticker,begin,end),eg_plg)
        df.rename(columns={'highPrice':'highestPrice','lowPrice':'lowestPrice'},inplace=True)
        df['chgPct'] = df.closePrice.pct_change()
        df.chgPct.fillna(0,inplace=True)
        df['accumAdjFactor'] = 1
    if len(df)==0:
        return pd.DataFrame(),pd.DataFrame()
    if df.shape[0]<240*2:
        return pd.DataFrame(),pd.DataFrame()
    lenth_w, e_list = get_stock_signal(df , window = 3, is_plot = False)
    if len(e_list)==0:
        return pd.DataFrame(),pd.DataFrame()
    try:
        df_back = back_test_more(df, e_list) 
        print('S43 %s update %s' % (dtype,ticker))
        return df_back,df.iloc[e_list][['ticker','tradeDate']]
    except:
        print('S43 %s stock E %s' % (dtype,ticker))
        return pd.DataFrame(),pd.DataFrame()


if __name__ == "__main__":
    time_start = datetime.now()
    #'csi,US,HK,"forex_day","forex_min"'
    dtype = 'HK'
    begin = '2010-10-01'    
    end = datetime.strftime(datetime.now(),'%Y-%m-%d')
    if table_in_database('s37',tn):
        t0 = pd.read_sql('select tradeDate from %s where index0 = "%s" order by tradeDate desc limit 1' % (tn,dtype),engine37)
        if len(t0)>0:
            t0 = t0.tradeDate.astype(str)[0]
        else:
            t0 = begin
    else:
        t0 = begin
    if dtype == "csi":
        tt = pd.read_sql('select tradeDate from yq_dayprice order by tradeDate desc limit 1',engine)
    elif dtype== "US":
        tt = pd.read_sql('select tradeDate from  usastock_day order by tradeDate desc limit 1',eg_plg)    
    elif dtype == "HK":
        tt = pd.read_sql('select tradeDate from mkthkequdgets54 order by tradeDate desc limit 1',engine)
    elif dtype == "forex_day":
        tt = pd.read_sql('select tradeDate from  forex_day order by tradeDate desc limit 1',eg_plg)   
    tt = tt[tt.columns[0]].astype(str).values[0]
    if tt> t0:  
        if dtype=="csi":
            hs_300_pool = get_symbol_A()
        elif dtype == "US":
            hs_300_pool = get_spx_com('2021-05-09')    
        elif dtype == "HK":
            hs_300_pool = ' select distinct(ticker) from main_index_s68 where index_id = "HSI" and ticker != "HSI"'
            hs_300_pool = pd.read_sql(hs_300_pool,eg_pro)
            hs_300_pool = hs_300_pool.ticker.tolist()
        elif dtype in ["forex_day","forex_min"]:
            hs_300_pool = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD',
                   'USDCHF', 'EURGBP', 'EURCHF', 'EURAUD', 'EURNZD', 'EURCAD', 'AUDJPY', 'NZDJPY',
                   'EURJPY', 'CHFJPY', 'GBPJPY', 'CADJPY', 'USDCNH', 'USDINR', 'USDTRY', 'USDRUB',
                   'USDZAR']
            
        T_symbols = len(hs_300_pool)
        if para_sel:
            p1 = hs_300_pool
            p2 = T_symbols*[begin]
            p3 = T_symbols*[end]
            p4 = T_symbols*[dtype]
            pool = multiprocessing.Pool(num_core2)
            back_dfs = pool.map(get_single_s43, zip(p1,p2,p3))
            pool.close()
            pool.join() 
        else:    
            back_dfs = []
            for ticker in tqdm(hs_300_pool):
                df_back = get_single_s43([ticker,begin,end,dtype])
                back_dfs.append(df_back)
        signal = [i[1] for i in back_dfs]
        back_dfs = [i[0] for i in back_dfs]
        #记录了信号的初始位置
        signal = pd.concat(signal)
        
        back_df = pd.concat(back_dfs)
        var_name = ['tradeDate', 'ticker','r_5', 'r_7', 'r_10', 'r_15', 'r_20', 'r_30', 'r_35', 'r_60']
        var_type = []
        for i in var_name:
            var_type.append('float')
        var_type[0] = 'date'
        var_type[1] = 'varchar(8)'
        key_str = 'tradeDate,ticker'
        
        #create_table_update('S37',tn,var_name,var_type,key_str,4)        
        back_df=back_df[back_df.tradeDate.astype(str)>t0]
        if len(back_df)>0:
            back_df['index0'] = dtype
            back_df.to_sql(tn,engine37,if_exists='append',index=False,chunksize=3000)
            
        time_end = datetime.now()
        print('Time used %s' % (time_end-time_start))
    else:
        print('S45 %s 数据已经是最新，无需更新' % dtype)