# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 22:59:32 2020

@author: Asus
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import matplotlib.pylab as plt


from sqlalchemy import create_engine
import json

import warnings
warnings.filterwarnings('ignore')
global cov, corr, dist
global link
#must be set before using
with open('para.json','r',encoding='utf-8') as f:
    para = json.load(f)
    
pn = para['yuqerdata_dir']

user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name1 = 'yuqerdata'
#eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name)
eng_str='mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name,pass_wd,port,db_name1)
engine = create_engine(eng_str)

sql_str_select_data1 = '''select %s from yq_dayprice where symbol="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''
sql_str_select_data2 = '''select %s from MktEqudAdjAfGet where ticker="%s" and tradeDate>="%s"
    and tradeDate<="%s" order by tradeDate'''  
sql_str_select_data3 = '''select * from yuqerdata.yq_industry_sw where industryVersionCD="010303" and industryID1="01030317" '''
med_code = pd.read_sql(sql_str_select_data3,engine)
print (med_code)
## 医药行业代码
med_code = med_code.loc[med_code['ticker'].drop_duplicates().index]
med_code = med_code[med_code['equType'].apply(lambda x: True if 'A' in x else False)]['ticker'].tolist()

def get_month_calender():
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" order by endDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x
## 获取时间
cal = pd.Series(get_month_calender())
next_cal = cal.shift(-1)

def get_month_factor(date, tickers, factor = ['marketValue'], save_time = False, month = False):
    ## 当前股票
    ## 取规模因子-市值的大小
    ### size 分组
    if 'marketValue' in factor:
        fac_m = DataAPI.MktEqudGet(secID=u'' ,ticker= tickers ,tradeDate= date ,isOpen=u"",field= ['ticker'] + ['marketValue'] ,pandas="1")
        factor.remove('marketValue')
        fac = DataAPI.MktStockFactorsOneDayGet(tradeDate=date, secID=u'', ticker= tickers, field=['ticker'] + factor ,pandas="1")
        fac = pd.merge(fac_m, fac, on = 'ticker')
    else:
        fac = DataAPI.MktStockFactorsOneDayGet(tradeDate=date, secID=tickers, ticker=u'', field=['ticker'] + factor ,pandas="1")
    next_date = next_cal[cal['calendarDate'] == date]['calendarDate'].values[0]
    ret = DataAPI.MktEqumGet(secID=u"",ticker=med_code ,monthEndDate=next_date ,beginDate=u"",endDate=u"",isOpen=1 ,field=['ticker', 'chgPct'],pandas="1")
    ## 收益率 + 波动率 + 反转 因子
    if month:
        ret_mon = DataAPI.MktEqumGet(secID=u"",ticker=med_code ,monthEndDate= date ,beginDate=u"",endDate=u"",isOpen=1 ,field=['ticker', 'return', 'turnoverRate', 'sdReturn24'],pandas="1")
        ret_mon.columns = ['ticker', 'b_return', 'turnoverRate', 'sdReturn24']
        ret = pd.merge(ret_mon, ret, on= 'ticker', how= 'inner')
    ## 除去次新股
    new_tickers = DataAPI.MktSubnewEqudGet(secID=u"",ticker=u"",beginDate=date, endDate=next_date, field=u"ticker",pandas="1")['ticker'].values
    df = pd.merge(fac, ret, on = 'ticker', how= 'right')
    df = df.set_index('ticker')
    df = df.loc[[i for i in df.index if i not in new_tickers], ]
    ## 中位数填充
    df = df.fillna(df.median())
    if save_time: df['date'] = date
    return df


df = get_month_factor('2018-08-31', tickers = med_code, factor= ['marketValue', 'PB', 'PE'], month= True)

def stad_data(x):
    x = standardize(winsorize(x))
    return x


factor_list = ['marketValue', 'PB', 'PE']
date_list = [i for i in cal['calendarDate'] if i > '2009-12' and i < '2019-06']
factor_dict = {}
flag = 0
for i in date_list:
    df = get_month_factor(i, tickers = med_code, factor= ['marketValue', 'PB', 'PE'])
    for col in df.columns:
        if col != 'chgPct': df[col] = stad_data(df[col])
    ## 计算正交 除去市值因子的影响
    x = df['marketValue'].values
    df_ma = df.apply(lambda y: y - x * x.dot(y) / x.dot(x))
    df_ma = df_ma[[j for j in factor_list if j != 'marketValue']]
    df_ma.columns = ['ma_'+ j for j in factor_list if j != 'marketValue']
    df = pd.concat([df_ma, df], axis = 1)
    factor_dict[i] = df
    if flag % 10 == 0:
        print('%s has done'%i)
    flag += 1
    
rankic_result = pd.concat([factor_dict[date].corr('spearman')['chgPct'] for date in date_list], axis = 1).T
rankic_result.index = date_list


## 逐步筛选确定有效因子过程
### 选取因子
factor_list = ['marketValue', 'PB', 'PE', 'ROE', 'ROA','GrossIncomeRatio', 'NetProfitRatio', 'OperatingProfitRatio', 'CashRateOfSales', 'NOCFToOperatingNI', 'NetProfitGrowRate', 'OperatingRevenueGrowRate', 'FEARNG', 'FSALESG', 'SUOI', 'TotalAssetGrowRate', 'NetAssetGrowRate', 'CurrentRatio', 'OperCashInToCurrentLiability', 'DebtsAssetRatio', 'SFY12P', 'FY12P']

date_list = [i for i in cal['calendarDate'] if i > '2009-12' and i < '2019-06']
factor_dict = {}
flag = 0
for i in date_list:
    df = get_month_factor(i, tickers = med_code, factor= ['marketValue', 'PB', 'PE', 'ROE', 'ROA','GrossIncomeRatio', 'NetProfitRatio', 'OperatingProfitRatio', 'CashRateOfSales', 'NOCFToOperatingNI', 'NetProfitGrowRate', 'OperatingRevenueGrowRate', 'FEARNG', 'FSALESG', 'SUOI', 'TotalAssetGrowRate', 'NetAssetGrowRate', 'CurrentRatio', 'OperCashInToCurrentLiability', 'DebtsAssetRatio', 'SFY12P', 'FY12P'], month= True)
    for col in df.columns:
        if col != 'chgPct': df[col] = stad_data(df[col])
    # # 计算正交 除去市值因子的影响
    # x = df['marketValue'].values
    # df_ma = df.apply(lambda y: y - x * x.dot(y) / x.dot(x))
    # df_ma = df_ma[[j for j in factor_list if j != 'marketValue']]
    # df_ma.columns = ['ma_'+ j for j in factor_list if j != 'marketValue']
    # df = pd.concat([df_ma, df], axis = 1)
    factor_dict[i] = df
    if flag % 10 == 0:
        print('%s has done'%i)
    flag += 1
    
df = get_month_factor(date_list[2], tickers = med_code, factor= ['marketValue', 'PB', 'PE'], month= True)


## 回归参数
def step_regress(fix_list = ['marketValue']):
    fac_c_dict = {}
    fac_r_dict = {}
    ori_r = []
    for date in date_list:
        df = factor_dict[date]
        y,x = df['chgPct'], df[[i for i in df.columns if i != 'chgPct']]
        fix_x = x[fix_list]
        add_list = [i for i in x.columns if i not in fix_list]
        r = sm.OLS(y,sm.add_constant(fix_x)).fit()
        ori_r.append(r.rsquared)
        ## 计算添加后的参数
        for col in add_list:
            fix_x = x[fix_list + [col]]
            r = sm.OLS(y,sm.add_constant(fix_x)).fit()
            r_add = r.params.loc[fix_list + [col]]
            r_add_sq = r.rsquared
            if col not in fac_c_dict.keys():
                fac_c_dict[col] = [r_add]
                fac_r_dict[col] = [r_add_sq]
            else:
                fac_c_dict[col].append(r_add)
                fac_r_dict[col].append(r_add_sq)
    
    return fac_c_dict, fac_r_dict, ori_r, add_list, fix_list


## 回归的检验过程
fix_list = ['marketValue']
while 1:
    fac_c_dict, fac_r_dict, ori_r, add_list, fix_list = step_regress(fix_list = fix_list)
    print(fix_list , 'done')
    add_copy = []
    for i in add_list:
        s = pd.concat(fac_c_dict[i], axis = 1)
        ## 计算显著
        t_test = s.apply(lambda x: stats.ttest_1samp(x, 0)[1], axis = 1)
        if (t_test > 0.5).sum() == 0:
            add_copy.append(i)
    if len(add_copy) == 0:
        break
    ## 取最大R方
    fix_list.append(add_list[np.argmax([np.mean(fac_r_dict[i]) for i in add_copy])])
    
## 合成因子过程
fac_c_dict, fac_r_dict, ori_r, add_list, _ = step_regress(fix_list = fix_list[:-1])
mul_fac = pd.concat(fac_c_dict[fix_list[-1]], axis = 1)
wt = mul_fac.T.mean()

mul_fac_df = pd.concat([(factor_dict[i][fix_list] * wt).sum(axis = 1) for i in date_list], axis = 1)
mul_fac_df.columns = date_list


## 选前20只做多组合，选后20只做空组合
choose_top_tickers = {}
choose_bot_tickers = {}
x_ic_ar = []
for i in date_list:
    df = factor_dict[i].copy()
    df['x_fac'] = (factor_dict[i][fix_list] * wt).sum(axis = 1)
    choose_top_tickers[i] = df['x_fac'].sort_values()[-20:].index.tolist()
    choose_bot_tickers[i] = df['x_fac'].sort_values()[20:].index.tolist()
    x_ic = df[['x_fac', 'chgPct']].corr('spearman').iloc[1, 0]
    x_ic_ar.append(x_ic)
x_ic_ar = np.array(x_ic_ar)


top_ser = pd.Series(np.array([factor_dict[date].loc[choose_top_tickers[date], 'chgPct'].mean() + 1 for date in date_list]).cumprod(), index= date_list)
bot_ser = pd.Series(np.array([factor_dict[date].loc[choose_bot_tickers[date], 'chgPct'].mean() + 1 for date in date_list]).cumprod(), index= date_list)
hug_ser = pd.Series( (np.array([factor_dict[date].loc[choose_top_tickers[date], 'chgPct'].mean() for date in date_list]) - 
                      np.array([factor_dict[date].loc[choose_bot_tickers[date], 'chgPct'].mean() for date in date_list]) + 1).cumprod(), index= date_list)
top_ser.plot(label = 'top') # 高20组合
bot_ser.plot(label = 'bot') # 底20组合
hug_ser.plot(label = 'hug') # 多空对冲
plt.legend(loc = 'center left')
plt.show()