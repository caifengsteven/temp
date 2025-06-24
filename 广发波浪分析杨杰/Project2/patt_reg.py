# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:28:42 2018

@author: JIE
"""
import itertools
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def same_period_plot2(data,cl1,cl2):
    """
    定义相关性函数，同期相关性作图（双y轴）
    """
    df = data.copy()
    df = df.iloc[:,[cl1,cl2]]
    df = df.interpolate(method='linear')
    df = df[np.isfinite(df.iloc[:,0])]
    df = df[np.isfinite(df.iloc[:,1])]    
   
    x = list(df.index)
    y1 = df.iloc[:,0]
    y2 = df.iloc[:,1]
    y1_name = list(df.columns)[0]
    y2_name = list(df.columns)[1]
    
    fig = plt.figure(figsize = (9,6))   
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()  # this is the important function

    ax1.plot(x, y1,label = y1_name)
    ax1.set_ylabel(y1_name)
    ax1.axhline(y=0.0, color='k', linestyle='--')
    #ax1.set_title("Double Y axis")
    
    ax2.plot(x, y2, 'r',label = y2_name)
    #ax2.set_xlim([0, np.e])
    ax2.set_ylabel(y2_name)
    ax1.plot(np.nan, 'r', label = y2_name)  # Make an agent in ax
    #ax2.set_xlabel('Same X for both exp(-x) and ln(x)')
    ax1.legend(loc=0)
    plt.show()
    
def Bolinger_Bands(stock_price, window_size=26, num_of_std=2):
    """
    计算布林线的上线，下线，中线；
    周期可以自定义，这里默认26个交易日；
    布林带带宽，默认两倍标准差 ;
    2倍标准差，95.5%概率
    """
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std  = stock_price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)

    return rolling_mean, upper_band, lower_band

def Boll_Spilt(x): 
    """
    输入%b指标，对之进行分割，得到Z字
    """        
    y = []
    y.append(x[0]) #b的第一个值，记录
    i = 1            
    while i<=len(x)-1:        
        if x[i]>=x[i-1]: #如果是上涨
            count = 0 #i+1+count<=len(x)-1 ,确保index不会溢出
            while i+1+count<=len(x)-1 and x[i+1+count]>=x[i+count]:# 后面继续上涨：
                y.append(np.nan) #上涨的时候，非最大值，直接记空,中间缺失值后面用线性插值法补充
                count += 1
            y.append(x[i+count]) #记录下最大值
            i += count + 1
            
        elif x[i]<x[i-1]:#如果是下跌
            count = 0
            while i+1+count<=len(x)-1 and x[i+1+count]<x[i+count]:# 后面继续下跌：
                y.append(np.nan) #下跌的时候，非最小值，直接记空，中间缺失值后面用线性插值法补充
                count += 1
            y.append(x[i+count]) #记录下最小值
            i += count + 1
    return y

def Boll_Cut(z,sigma):
    """
    输入切割后的%b,参数z
    裁剪的阈值，sigma
    对之进行裁剪
    """
    zcut = z.copy()
    idx = np.argwhere(~np.isnan(z)) #找到非nan 值得位置，也就是局域最大值，最小值位置
    idx = list(itertools.chain(*idx)) #把嵌套得list转化为非嵌套的list
    for i in range(len(idx)-1):
        po = idx[i] #当前这个极值
        if i==0:
            pass #保留第一个值
        elif po<len(z)-3: #最后值直接保留
            lst_po = idx[i-1] #上1个极值
            nxt_po1 = idx[i+1] #下1个极值
            nxt_po2 = idx[i+2] #下2个极值
            
            #Case 1
            #(1) 第i个极值大于第i-1个极值；同时小于第i+2个极值；
            #(2) 第i+1个极值小于第i个极值；同时偏离不超过sigma
            #(3)第i+1个极值大于第i-1个极值
            #以上3个条件同时满足，则中间两个点，i,i+1拿掉；
            flag11 =  z[lst_po]  < z[po] < z[nxt_po2]  #(1) 第i个极值大于第i-1个极值；同时小于第i+2个极值；
            flag12 =  z[po] -sigma <= z[nxt_po1] < z[po]  #(2) 第i+1个极值小于第i个极值；同时偏离不超过sigma
            flag13 =  z[nxt_po1] > z[lst_po]  #(3)第i+1个极值大于第i-1个极值
            
            #Case 2
            #(1) 第i个极值小于第i-1个极值,同时偏离不能超过sigma；
            #(2) 第i+1个极值大于第i个极值；且第i+1个极值小于第i-1个极值
            #(3)第i+2个极值大于第i个极值
            #以上三个条件同时满足，则中间两个点，i,i+1拿掉；
            flag21 =  z[lst_po] - sigma <= z[po] < z[lst_po]  #(1) 第i个极值小于第i-1个极值,同时偏离不能超过sigma
            flag22 =  z[po] < z[nxt_po1] < z[lst_po]  #(2) 第i+1个极值大于第i个极值；且第i+1个极值小于第i-1个极值
            flag23 =  z[nxt_po2] < z[po]  #(2)第i+2个极值大于第i个极值
            
            if (flag11 ==True and flag12 ==True and flag13 ==True) or (flag21 ==True and flag22 ==True and flag23 ==True):
                zcut[po] = np.nan
                zcut[nxt_po1] = np.nan
            else:
                 pass
                
        else:
            pass
        
    return zcut

def high_low_rank(length,dateid,sp):  
    """
    以当日时间为截止点，
    输出长度为 n 的高低序列 
    由于波浪理论的原因，默认n为奇数
    """
    n = length #取长度为n的点位,偶数
    dateid = datetime.strptime(str(dateid), '%Y-%m-%d')
    
    mask = sp['close_adj_cut'].apply(np.isnan) #判定是否为缺失；True缺失；
    ts = sp[['close','Date']].loc[mask==False]
    ts2 = ts.copy()
    ts2.index = ts.Date
    ts2 = ts2[:dateid]
    def nearest(items, pivot):#返回前面最近一个高低点日期
        return min(items, key=lambda x: abs(x - pivot))
    
    date = nearest(ts2.Date, dateid) #返回前面最近一个高低点日期
    idx = ts.index[ts.Date == date].tolist()[0]#找到对应点的位置
    
    index_list = ts.index.tolist() #由于ts的index不连续，重新生成一个index list
    ind = index_list.index(idx) #找到index的新位置
    idx_list = index_list[ind-length:ind] #取出最新日期，前面长度为n的index
    temp = ts.loc[ts.index.isin(idx_list)]
    temp = temp.reset_index()
    
    i1 = [i*2 for i in range(n//2+1)]
    i2 = [i*2+1 for i in range(n//2)]
    temp1 = temp.loc[temp.index.isin(i1)]
    temp2 = temp.loc[temp.index.isin(i2)]
    
    rankidx = temp['close'].rank().values #高低点排序
    
    return temp, temp1, temp2, rankidx

def cal_distance(length, dateid1,dateid2,sp):
    """
    相似度距离， length窗口的决定非常重要 
    输出高低点位置排序是否完全一致；"""
    temp1, temp11, temp12, rankidx1 = high_low_rank(length,dateid1,sp)
    temp2, temp21, temp22, rankidx2 = high_low_rank(length,dateid2,sp)
    flag = (rankidx1 == rankidx2).all() #判断高低点位置是否完全一致；
    
    close1,close2 = temp1.close.values,temp2.close.values
    diff1 = [close1[i+1]-close1[i] for i in range(length-1)]
    diff2 = [close2[i+1]-close2[i] for i in range(length-1)]

    diff = [abs(abs(a) -abs(b)) for a,b in zip(diff1,diff2)]
    dist = np.sum(diff)/(length-1)
    return flag, dist

def find_similarity(length,given_date,sp):
    """
    寻找过往，历史中相似的走势
    """
    #given_date = '2018-08-22'  
    given_date = datetime.strptime(str(given_date), '%Y-%m-%d')
    sp_slice = sp.copy()
    sp_slice.index = sp_slice.Date
    sp_slice = sp_slice[:given_date]
    
    mask = sp_slice['close_adj_cut'].apply(np.isnan) #判定是否为缺失；True缺失
    date_ts = sp_slice['Date'].loc[mask==False].apply(lambda x: str(x.date()))
    date_list = list(date_ts.values)[length:-1]#前面至少一个length；最后一个点不包含，是自身
    
    similar_date = []#记录相似位置
    dimilar_dist = []
    for dateid in date_list:
        flag, dist = cal_distance(length, str(given_date.date()),dateid,sp)
        if flag == True:
            similar_date.append(dateid)
            dimilar_dist.append(dist)
        else:
            pass
    return similar_date, dimilar_dist
           
def forecast_trend(length,given_date,sp):
    """
    历史图形一致的时候，依据历史中匹配相似点的下一个端点，对未来进行预测；
    上涨行情：t+1个端点高于t-1个端点；买入，并设定浮动动态止损策略
    下跌行情：t+1个端点低于t-1个端点；卖空，动态止损点--t-1端点点位
    """
    mask = sp['close_adj_cut'].apply(np.isnan) #判定是否为缺失；True缺失；
    ts = sp[['close','Date']].loc[mask==False]
    index_list = ts.index.tolist() #由于ts的index不连续，重新生成一个index list
  
    similar_date, dimilar_dist = find_similarity(length,given_date,sp)   
    nsignal = len(similar_date)
    
    def catch_signal(sig_date):
        idx = ts.index[ts.Date == sig_date].tolist()[0]#找到对应点的位置
        ind = index_list.index(idx) #找到index的新位置
        lst_idx = index_list[ind-1]
        nxt_idx = index_list[ind+1]
        lst_value = sp.iloc[lst_idx]['close']
        nxt_value = sp.iloc[nxt_idx]['close']
        
        if lst_value > nxt_value:
            signal = -1
            
        elif lst_value < nxt_value:
            signal = 1
            
        elif lst_value == nxt_value:
            signal = 0
        return signal
    
    a = [] #记录交易信号            
    if  nsignal == 0:#没有信号
        pass
        #print("没有历史形态与之匹配，无法预测未来走势")
    else:#有信号
        #print("形态与之匹配，可以预测未来走势")
        for i in range(nsignal):
            signal_date = similar_date[i]
            signal = catch_signal(sig_date = signal_date)
            print("当前日期为 " + given_date + ", 匹配日期为 " + signal_date +', 后市信号为' + str(signal) +'（其中1看涨，0不确定，-1看跌）')
            a.append(signal) 
            
    #输出整体买入，卖出信号
    b = 0  #整体买入，卖出信号,默认无信号
    if len(a) == 0:#没有信号
        pass
    else:
        if np.sum(a)==0: #有信号，但信号不明确
             pass
        else:
            total_signal = np.sum(a)/len(a)
            if total_signal>0.0:#明确卖出信号
                b = 1
            elif total_signal<0.0:#明确卖出信号
                b = -1
    return b            
            
def performance_attr(r,fre):
    """
    调用Qrisk包，完成收益评估
    """
    import qrisk as qr
    def cal_max_loss(r):
        loss = ((1+r).cumprod()).min() - 1
        return loss
    pf = dict()
    #pf['alpha'] = qr.alpha(r,period = fre) 
    #pf['beta'] = qr.beta(r,period = fre) 
    pf['annual_return'] = qr.annual_return(r,period = fre)
    pf['annual_volatility']= qr.annual_volatility(r,period = fre)
    pf['sharpe_ratio']= qr.sharpe_ratio(r,period = fre) # risk_free=0,TD=252
    pf['downside_risk']= qr.downside_risk(r,period = fre) 
    pf['max_drawdown'] = qr.max_drawdown(r)
    pf['max_loss'] = cal_max_loss(r)
    pf['sortino_ratio'] = qr.sortino_ratio(r,period = fre) # required return = 0
    #pf['information_ratio'] = qr.information_ratio(r)
    pf['omega_ratio'] = qr.omega_ratio(r) #risk_free=0.0, required_return=0.0
    pf['tail_ratio'] = qr.tail_ratio(r) 
    return pf    
        

def cal_win_ratio(re,sp):  
    date_list = sp.loc[sp['close_adj_cut'].apply(np.isnan)==False]['Date']
    date_list = date_list.apply(lambda x:str(x.date()))
    rt = pd.DataFrame(re)
    rt.columns = ['signal']
    sp['adj_close'] = sp['close'].shift(-1) #.shift(-1) #第二天才能买入/卖出
    rt['close'] = sp.loc[sp['close_adj_cut'].apply(np.isnan)==False]['adj_close'].values
    rt['return'] = rt['close'].pct_change().shift(-1) #对齐信号日的收益
    rt['port'] = rt['return']*rt['signal']
    rt.index = date_list
    rt = rt.dropna()
    win_loss = len(rt.loc[rt['port']>0])/len(rt.loc[rt['signal']!=0])
    return win_loss

def cal_performance(re,sp):
    
    date_list = sp.loc[sp['close_adj_cut'].apply(np.isnan)==False]['Date']
    date_list = date_list.apply(lambda x:str(x.date()))
    
    rt = pd.DataFrame(re)
    rt.columns = ['signal']
    sp['adj_close'] = sp['close'].shift(-1) #.shift(-1) #第二天才能买入/卖出
    rt['close'] = sp.loc[sp['close_adj_cut'].apply(np.isnan)==False]['adj_close'].values
    rt['return'] = rt['close'].pct_change().shift(-1) #对齐信号日的收益
    rt['port'] = rt['return']*rt['signal']
    rt.index = date_list
    rt = rt.dropna()
    win_loss = len(rt.loc[rt['port']>0])/len(rt.loc[rt['signal']!=0])
    #win_loss = cal_win_ratio(re,sp)
    
    rtt = pd.DataFrame(re)
    rtt.columns = ['signal']
    rtt.index = date_list
    close = sp['close']
    close.index = sp['Date'].apply(lambda x: str(x.date()))
    ts = pd.concat([rtt,close ], axis=1)
    ts['adj_close'] = ts['close'].shift(-1) #第二天才能买入/卖出
    ts['daily_ret'] = ts['adj_close'].pct_change().shift(-1) #对齐信号日的收益
    ts['adj_signal'] = ts['signal'].fillna(method = 'ffill')
    ts['port_ret'] = ts['daily_ret']*ts['adj_signal']
    
    tt = ts['2011-07-15':]
    r = tt['port_ret'].dropna()
    (r+1).cumprod().plot(figsize=(9,6))

    temp = performance_attr(r,fre='daily')
    temp['win_loss'] = win_loss
    tem_ts = pd.DataFrame.from_dict(temp,orient='index')    
    tem_ts = tem_ts.T
    tem_ts = tem_ts[['annual_return', 'annual_volatility','sharpe_ratio',
                       'max_drawdown', 'max_loss', 'downside_risk','win_loss',
                        'sortino_ratio','omega_ratio','tail_ratio']]  
    return tem_ts