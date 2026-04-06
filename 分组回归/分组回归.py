
# coding: utf-8

# In[ ]:



import pandas as pd
import numpy as np
import os
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
#from CAL.PyCAL import * 

from sqlalchemy import create_engine
import json

import warnings



warnings.filterwarnings('ignore')

# must be set before using
with open('para.json', 'r', encoding='utf-8') as f:
    para = json.load(f)

user_name = para['mysql_para']['user_name']
pass_wd = para['mysql_para']['pass_wd']
port = para['mysql_para']['port']

db_name1 = 'yuqerdata'
db_name2 = 'yuqer_cubdata'
eng_str = 'mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name, pass_wd, port, db_name1)
eng_str2 = 'mysql+pymysql://%s:%s@localhost:%d/%s?charset=utf8' % (user_name, pass_wd, port, db_name2)
engine = create_engine(eng_str)
engine2 = create_engine(eng_str2)

def get_IdxCons(intoDate,ticker='000300'):
    #nearst 时间
    sql_str1 = '''select symbol from yuqerdata.IdxCloseWeightGet where ticker = "%s"
            and tradingdate = (select tradingdate from yuqerdata.IdxCloseWeightGet where 
        ticker="%s" and tradingdate<="%s"  order by tradingdate desc limit 1)''' %(ticker,
        ticker,intoDate)
    x = pd.read_sql(sql_str1,engine)
    x = x['symbol'].values   
    return x

def chs_factor(ticker = '000005',begin = None ,end = None , 
               field = [u'symbol',  u'tradeDate', u'openPrice',
                        u'highestPrice', u'lowestPrice', u'closePrice', u'turnoverVol',
                        u'turnoverValue',u'dealAmount', u'chgPct',
                        'turnoverRate',u'marketValue',u'accumAdjFactor']):
    sql_str1 = sql_str_select_data1 % (','.join(field),ticker,begin,end)
    dataday = pd.read_sql(sql_str1,engine)
    dataday = dataday.applymap(lambda x: np.nan if x == 0 else x)
    dataday.rename(columns={'symbol':'ticker'},inplace=True)
    ## 对数据补全
    return dataday.fillna(method = 'ffill')


## 得到月度日历
def get_calender_range(begin, end):
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" and tradeDate >="%s" and tradeDate <="%s" order by tradeDate'''%(begin, end)
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

def get_calender():
    sql_str = '''select tradeDate from yuqerdata.yq_index where symbol = "000001" order by tradeDate'''
    x=pd.read_sql(sql_str,engine)
    x=x['tradeDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x
    
def get_month_calender(start_date, end_date):
    sql_str = '''select endDate from yuqerdata.yq_index_month where symbol = "000001" and endDate>="%s" and endDate <="%s" order by endDate'''%(start_date, end_date)
    x=pd.read_sql(sql_str,engine)
    x=x['endDate'].values
    #b=[i.strftime('%Y-%m-%d') for i in x]
    return x

raw_data_dir = "./raw_data"
if not os.path.exists(raw_data_dir):
    os.mkdir(raw_data_dir)

def MktStockFactorsOneDayProGet(tradeDate, fields):
    df = pd.DataFrame()
    i = 1
    for field in fields:
        field = field.lower()
        sql_str1 = '''select symbol, tradingdate, f_val from ''' + field
        sql_str2 = ''' where tradingdate = "%s"''' % (tradeDate)
        sql_str = sql_str1 + sql_str2
        # sql_str = '''select symbol, tradingdate, f_val from '%s' where tradingdate = "%s"'''%(field, tradeDate)
        x = pd.read_sql(sql_str, engine2)
        x = x.rename(columns={"f_val": field})
        if i == 1:
            df = x
        else:
            df = df.merge(x, on=["symbol", "tradingdate"])
        i = i + 1

    return df

#定义需要获取的因子
factors = ['LCAP', 'REVS20', 'VOL10']
# 规模：对数市值，LCAP
# 反转：20日收益率，REVS20
# 流动性：20日日均换手率，VOL20

def get_factor_by_day(tdate):
    '''
    获取给定日期的因子信息
    参数： 
        tdate, 时间，格式%Y%m%d
    返回:
        DataFrame, 返回给定日期的70个因子值
    '''
    cnt = 0
    while True:
        try:
            # universe = set_universe('ZZ500', date=tdate)
            x = MktStockFactorsOneDayProGet(tradeDate=tdate,fields=factors)
            x['tradingdate'] = x['tradingdate'].astype(str).apply(lambda x: x.replace("-", ""))

            return x
        except Exception as e:
            print(e)
            cnt += 1
            if cnt >= 3:
                print('error get factor data: ', tdate)
                break


def MktEqumAdjGet(beginDate, endDate, universe, fields):
    df = pd.DataFrame()
    for i in universe:
        sql_str = '''select ''' + fields
        sql_str += ''' from mktequmadjafget where ticker = "%s" and endDate >="%s" and endDate <= "%s"''' % (
        i, beginDate, endDate)
        x = pd.read_sql(sql_str, engine)
        df = df.append(x)
    return df


def MktEquwAdjGet(beginDate, endDate, universe, fields):
    df = pd.DataFrame()
    for i in universe:
        sql_str = '''select ''' + fields
        sql_str += ''' from yq_mktequwadjafget where ticker = "%s" and endDate >="%s" and endDate <= "%s"''' % (
        i, beginDate, endDate)
        x = pd.read_sql(sql_str, engine)
        df = df.append(x)
    return df

def MktIdxmGet(beginDate, endDate, symbol=['000300'], fields='symbol,endDate, chgPct'):
    df = pd.DataFrame()
    for i in symbol:
        sql_str = '''select ''' + fields
        sql_str += ''' from yq_index_month where symbol = "%s" and endDate >="%s" and endDate <= "%s"''' % (i, beginDate, endDate)
        x = pd.read_sql(sql_str, engine)
        df = df.append(x)
    return df

if __name__ == "__main__":
    start_time = time.time()

    # 拿到交易日历，得到月末日期
    #trade_date = (exchangeCD=u"XSHG", beginDate="20070101", endDate="20171231", field=u"", pandas="1")
    #trade_date = trade_date[trade_date.isMonthEnd == 1]
    trade_date = pd.Series(get_month_calender(start_date ="20070101", end_date = "20171231" ))
    #trade_date['tradeDate'] = trade_date['month_end']
    #print(trade_date)

    print("begin to get factor value for each stock...")
    # # 取得每个月末日期，所有股票的因子值
    pool = ThreadPool(processes=16)
    date_list = [tdate.replace("-", "") for tdate in trade_date.astype(str).values if tdate < "20171101"]
    frame_list = pool.map(get_factor_by_day, date_list)
    pool.close()
    pool.join()
    print ("ALL FINISHED")

    factor_csv = pd.concat(frame_list, axis=0)
    factor_csv.reset_index(inplace=True, drop=True)
    stock_list = np.unique(factor_csv.symbol.values)

    ########################## 取得个股和指数的行情数据 ################################
    print("\nbegin to get price ratio for stocks and index ...")
    # 个股绝对涨幅
    chgframe = MktEqumAdjGet(beginDate="20070131", endDate="20171130", universe=stock_list,fields='ticker, endDate, tradeDays, chgPct')
    
    chgframe['endDate'] = chgframe['endDate'].apply(lambda x: x.replace("-", ""))

    # 沪深300指数涨幅
    hs300_chg_frame = MktIdxmGet(beginDate="20070131", endDate="20171130", symbol=["000300"],fields='symbol, endDate, chgPct')
    print(hs300_chg_frame)
    hs300_chg_frame['endDate'] = hs300_chg_frame['endDate'].astype('str').apply(lambda x: x.replace("-", ""))
    hs300_chg_frame.head()

    # 得到个股的相对收益
    hs300_chg_frame.columns = ['HS300', 'endDate', 'HS300_chgPct']
    pframe = chgframe.merge(hs300_chg_frame, on=['endDate'], how='left')
    pframe['active_return'] = pframe['chgPct'] - pframe['HS300_chgPct']
    #pframe = pframe[['ticker', 'endDate', 'return', 'active_return']]
    pframe = pframe[['ticker', 'endDate', 'active_return']]
    #pframe.rename(columns={"return": "abs_return"}, inplace=True)

    ################################ 对齐数据 ################################
    print("begin to align data ...")
    # 得到月度关系
    month_frame = pd.DataFrame(trade_date,columns=['month_end'])
    month_frame['tradingdate'] = month_frame['month_end']
    #print(month_frame)
    month_frame['prev_month_end'] = month_frame['month_end'].shift(1)
    month_frame = month_frame[['prev_month_end', 'month_end']]
    month_frame.columns = ['month_end', 'next_month_end']
    month_frame.dropna(inplace=True)
    month_frame['month_end'] = month_frame['month_end'].astype('str').apply(lambda x: x.replace("-", ""))
    month_frame['next_month_end'] = month_frame['next_month_end'].astype('str').apply(lambda x: x.replace("-", ""))
    #print(factor_csv)
    # 对齐月度关系
    factor_frame = factor_csv.merge(month_frame, left_on=['tradingdate'], right_on=['month_end'], how='left')

    # 得到个股下个月的涨幅数据
    print('*'*50)
    print(factor_frame)
    print('*'*50)
    print(pframe)
    factor_frame = factor_frame.merge(pframe, left_on=['ticker', 'next_month_end'], right_on=['ticker', 'endDate'])

    del factor_frame['month_end']
    del factor_frame['endDate']

    ################################ 数据存储下来 ################################
    factor_frame.to_csv(os.path.join(raw_data_dir, 'factor_chpct.csv'), chunksize=1000)

    end_time = time.time()
    print ("Time cost: %s seconds" % (end_time - start_time))


# In[ ]:


import pandas as pd
import numpy as np
import os
import shutil
import multiprocessing
import time
import gevent
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


######################################### 通用变量设置 #########################################
start_time = time.time()
raw_data_dir = "./raw_data"

pre_handle_dir = "./pre_handle_data"  # 存放中间数据
if not os.path.exists(pre_handle_dir):
    os.mkdir(pre_handle_dir)

# 申万一级行业分类
sw_map_frame = DataAPI.EquIndustryGet(industryVersionCD=u"010303", industry=u"", secID=u"", ticker=u"", intoDate=u"",field=[u'ticker', 'secShortName', 'industry', 'intoDate', 'outDate', 'industryName1', 'industryName2', 'industryName3', 'isNew'], pandas="1")
sw_map_frame = sw_map_frame[sw_map_frame.isNew == 1]
    

# 读入原始因子
input_frame = pd.read_csv(os.path.join(raw_data_dir, u'factor_chpct.csv'),
                          dtype={"ticker": np.str, "tradeDate": np.str, "next_month_end": np.str}, index_col=0)

# 得到因子名
extra_list = ['ticker', 'tradeDate', 'next_month_end', 'abs_return', 'active_return']
factor_name = [x for x in input_frame.columns if x not in extra_list]

print('init data done, cost time: %s seconds' % (time.time()-start_time))

################################### 定义数据处理的一些基本函数 ##################################

def paper_winsorize(v, upper, lower):
    '''
    winsorize去极值，给定上下界
    参数:    
        v: Series, 因子值
        upper: 上界值
        lower: 下界值
    返回:
        Series, 规定上下界后因子值
    '''
    if v > upper:
        v = upper
    elif v < lower:
        v = lower
    return v

def winsorize_by_date(cdate_input):
    '''
    按照[dm+5*dm1, dm-5*dm1]进行winsorize
    参数:
        cdate_input: 某一期的因子值的dataframe
    返回:
        DataFrame, 去极值后的因子值
    '''
    media_v = cdate_input.median()
    for a_factor in factor_name:
        dm = media_v[a_factor]
        new_factor_series = abs(cdate_input[a_factor] - dm)  # abs(di-dm)
        dm1 = new_factor_series.median()
        upper = dm + 5 * dm1
        lower = dm - 5 * dm1
        cdate_input[a_factor] = cdate_input[a_factor].apply(lambda x: paper_winsorize(x, upper, lower))
    return cdate_input


def nafill_by_sw1(cdate_input):
    '''
    用申万一级的均值进行填充
    参数:
        cdate_input: 因子值，DataFrame
    返回:
        DataFrame, 填充缺失值后的因子值
    '''
    func_input = cdate_input.copy()
    func_input = func_input.merge(sw_map_frame[['ticker', 'industryName1']], on=['ticker'], how='left')
    
    func_input.loc[:, factor_name] = func_input.loc[:, factor_name].fillna(func_input.groupby('industryName1')[factor_name].transform("mean"))
    
    return func_input.fillna(0.0)


def winsorize_fillna_date(tdate):
    '''
    对某一天的数据进行去极值，填充缺失值
    参数:
        tdate： 时间， 格式为 %Y%m%d
    返回:
        DataFrame, 去极值，填充缺失值后的因子值
    '''
    cnt = 0
    while True:
        try:
            cdate_input = input_frame[input_frame.tradeDate == tdate]
            # print("####Running single_date for %s" % tdate)
            # winsorize
            cdate_input = winsorize_by_date(cdate_input)

            # 缺失值填充, 用同行业的均值
            cdate_input = nafill_by_sw1(cdate_input)
            cdate_input.set_index('ticker', inplace=True)

            return cdate_input
        except Exception as e:
            cnt += 1
            if cnt >= 3:
                cdate_input = input_frame[input_frame.tradeDate == tdate]
                # 缺失值填充, 用同行业的均值
                cdate_input = nafill_by_sw1(cdate_input)
                cdate_input.set_index('ticker', inplace=True)
                return cdate_input
            
            
def standardize_neutralize_factor(input_data):
    '''
    行业中性化，并进行标准化
    参数: 
        input_data：tuple, 传入的是(因子值，时间)。因子值为DataFrame
    返回:
        DataFrame, 行业中性化，并进行标准化后的因子值
    '''
    cdate_input, tdate = input_data
    for a_factor in factor_name:
        cnt = 0
        while True:
            try:
                cdate_input.loc[:, a_factor] = standardize(neutralize(cdate_input[a_factor], target_date=tdate,
                        exclude_style_list=['SIZE', 'SIZENL','BETA', 'RESVOL', 'MOMENTUM', 'EARNYILD', 'BTOP', 'GROWTH', 'LEVERAGE', 'LIQUIDTY']))
                break
            except Exception as e:
                print('standardize false')
                cnt += 1
                if cnt >= 3:
                    break
    
    return cdate_input

            
if __name__ == "__main__":
    ############################################ 对每期的数据进行处理 ###########################################
    # 遍历每个月末日期，对因子进行去极值、空值填充
    print('winsorize factor data...')
    pool = Pool(processes=8)
    date_list = [tdate for tdate in np.unique(input_frame.tradeDate.values) if int(tdate) > 20061231]
    dframe_list = pool.map(winsorize_fillna_date, date_list)

    # 遍历每个月末日期，利用协程对因子进行标准化，中性化处理
    print('standardize & neutralize factor...')
    jobs = [gevent.spawn(standardize_neutralize_factor, value) for value in zip(dframe_list, date_list)]
    gevent.joinall(jobs)
    new_dframe_list = [e.value for e in jobs]
    print('standardize neutralize factor finished!')
    
            
    # 将不同月份的数据合并到一起
    all_frame = pd.concat(new_dframe_list, axis=0)
    all_frame.reset_index(inplace=True)

    # 存储下来
    all_frame.to_csv(os.path.join(raw_data_dir, "after_prehandle.csv"), encoding='gbk', chunksize=1000)
    end_time = time.time()
    print("\nData handle finished! Time Cost:%s seconds" % (end_time - start_time))


# In[ ]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

raw_data_dir = "./raw_data"
# 读入原始因子
factor_df = pd.read_csv(os.path.join(raw_data_dir, u'after_prehandle.csv'), 
                        dtype={"ticker": np.str, "tradeDate": np.str, "next_month_end": np.str}, index_col=0, encoding='gbk')
factor_df.head()


# In[ ]:

from datetime import timedelta, datetime

factors = ['LCAP', 'REVS20', 'VOL20']  
type_list = ['QR(0.1)', 'QR(0.5)', 'QR(0.9)', 'OLS']

# 获取剔除上市未满60天的全A股票
equ_df = DataAPI.EquGet(equTypeCD=u"A", listStatusCD=u"", field=['secID', 'ticker', 'listDate', 'delistDate'], pandas="1")
def get_A_ticker(date, list_date=60):
    list_date_need = (datetime.strptime(date, '%Y%m%d') - timedelta(days=list_date)).strftime('%Y%m%d')
    A_ticker = equ_df[(equ_df['listDate'] <= list_date_need) & ((equ_df['delistDate'] > date) | (equ_df['delistDate'].isnull()))]['ticker'].tolist()
    
    return A_ticker


def plot_quantile_reg_res(data, factor, date, universe, ax):
    # 取三组不同分位点，进行回归，保留参数
    mod = smf.quantreg('abs_return ~ %s'%factor, data)
    quantiles = [0.1, 0.5, 0.9]
    def fit_model(q):
        res = mod.fit(q=q)
        return [q, res.params['Intercept'], res.params[factor]] + res.conf_int().ix[factor].tolist()

    models = [fit_model(x) for x in quantiles]
    models = pd.DataFrame(models, columns=['q', 'a', 'b','lb','ub'])
    
    # 对照组，均值回归
    ols = smf.ols('abs_return ~ %s'%factor, data).fit()
    ols_ci = ols.conf_int().ix[factor].tolist()
    ols = dict(a = ols.params['Intercept'], b = ols.params[factor], lb = ols_ci[0], ub = ols_ci[1])

    # 画图比较
    x = np.arange(data[factor].min(), data[factor].max(), 0.1)
    get_y = lambda a, b: a + b * x

    colors = ['blue', 'red', 'green']
    for i in range(models.shape[0]):
        y = get_y(models.a[i], models.b[i])
        ax.plot(x, y, color=colors[i], label='quantile=%s'%quantiles[i])

    y = get_y(ols['a'], ols['b'])
    ax.plot(x, y, color='black', linestyle='dotted', label='OLS')
    
    ax.scatter(data[factor], data['abs_return'], alpha=.2, color='grey', label=None)
    legend = ax.legend(fontsize=12)
    
    ax.set_xlabel('Quantile', fontsize=12)
    ax.set_ylabel('Next Month Return', fontsize=12)
    ax.set_title("Factor: %s, Date: %s, Universe:%s"%(factor, date, universe), fontsize=12)

# 考察选取的一个时间点的因子数据，对每一个因子进行不同股票池的比较
this_date = '20170630'
factor_data = factor_df[factor_df['tradeDate']==this_date].set_index('ticker')
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))

hs300 = [secID[:6] for secID in set_universe('HS300', this_date)]
zz500 = [secID[:6] for secID in set_universe('ZZ500', this_date)]
universe_list = [('A', get_A_ticker(this_date)), ('HS300', hs300), ('ZZ500', zz500)]
for i, factor in enumerate(factors):
    for j, (universe, tickers) in enumerate(universe_list):
        plot_quantile_reg_res(factor_data.ix[tickers], factor, this_date, universe, axes[i][j])
    


# In[ ]:

#计算历年来所有的系数
date_list = factor_df['tradeDate'].unique()
iterables = [date_list, type_list]
factor_params_df = pd.DataFrame(index=pd.MultiIndex.from_product(iterables, names=['tradeDate', 'type']), columns=factors)
choose_universe = 'ZZ500'

for this_date in date_list:
    factor_data = factor_df[factor_df['tradeDate']==this_date].set_index('ticker')
    # 只选取ZZ500成分股
    universe_ticker = [secID[:6] for secID in set_universe(choose_universe, this_date)]
    factor_data = factor_data.ix[universe_ticker]
    
    # 进行分位数回归，保存参数
    mod = smf.quantreg('abs_return ~ LCAP + REVS20 + VOL20', factor_data)
    quantiles = [0.1, 0.5, 0.9]
    def fit_model(q):
        res = mod.fit(q=q)
        params = np.array([res.params[factor] for factor in factors])
        
        return params*1.0

    for q in quantiles:
        factor_params_df.loc[this_date, 'QR(%s)'%q] = fit_model(q)
    
    # 进行均值回归，保存参数
    ols = smf.ols('abs_return ~ LCAP + REVS20 + VOL20', factor_data).fit()
    params = np.array([ols.params[factor] for factor in factors])
    factor_params_df.loc[this_date, 'OLS'] = params*1.0
    
factor_params_df = factor_params_df.reset_index()   
factor_params_df.head(5)


# In[ ]:

# 计算6期滚动平均值，截选10年后的数据进行打分，便于后期回测
begin_date = '20100101'
factor_params_df.loc[:, factors] = factor_params_df.groupby('type').transform(lambda x : x.rolling(6).mean().shift())[factors]

# 这里进行了归一化，方便后期比较因子权重
factor_params_df.loc[:, factors] = factor_params_df[factors].apply(lambda x : x/np.abs(factor_params_df[factors]).sum(axis=1))
factor_params_df = factor_params_df[factor_params_df['tradeDate']>begin_date]

factor_params_df.head()


# In[ ]:

#查看历年来各因子的系数权重变化
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

plt.style.use('ggplot')

# 画图比较3个因子的各期的权重变化图，总权重按照上面所说已进行归一化
def plot_factor_coefficient(data, ax, this_type):
    factor1 = data[factors[0]].abs() * 100.0
    factor2 = (data[factors[0]].abs() + data[factors[1]].abs()) * 100.0
    factor3 = (data[factors[0]].abs() + data[factors[1]].abs() + data[factors[2]].abs()) * 100.0

    ax.plot(data.index, factor1, factor2, factor3, color='black')
    ax.fill_between(data.index.tolist(), 0, factor1.tolist(), facecolor='blue')
    ax.fill_between(data.index.tolist(), factor1.tolist(), factor2.tolist(), facecolor='red')
    ax.fill_between(data.index.tolist(), factor2.tolist(), factor3.tolist(), facecolor='green')
    
    ax.set_xlim(data.index.tolist()[0], data.index.tolist()[-1])
    ax.set_ylim(-1, 101)
    ymajorFormatter = FormatStrFormatter('%.f%%')     
    ax.yaxis.set_major_formatter(ymajorFormatter)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Factor Coefficient Cumsum Percent', fontsize=12)
    ax.set_title('Type: %s'%this_type, fontsize=14)

# 对三组分位数回归与均值回归共4组类型进行分析
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(28, 4))
for i, this_type in enumerate(type_list):
    data = factor_params_df[factor_params_df['type']==this_type].set_index('tradeDate', drop=True)
    data.index = pd.to_datetime(data.index)
    plot_factor_coefficient(data, axes[i], this_type)
plt.show()


# In[ ]:

#利用上述回归权重计算股票得分
factor_score_dict = {}
factor_df = factor_df[factor_df['tradeDate']>begin_date]
factor_df['score'] = np.nan
for this_type in type_list:
    params = factor_params_df[factor_params_df['type']==this_type].set_index('tradeDate', drop=True)
    
    final_factor_df = factor_df[['tradeDate', 'ticker', 'score']].set_index(['tradeDate', 'ticker'], drop=True)
    for tradeDate, data in factor_df.groupby('tradeDate'):
        this_param = params.loc[tradeDate]
        data['score'] = sum([data[factor] * this_param[factor] for factor in factors])
        need_data = data[['tradeDate', 'ticker', 'score']].set_index(['tradeDate', 'ticker'], drop=True)
        final_factor_df.loc[need_data.index, :] = need_data['score']

    factor_score_dict[this_type] = final_factor_df


# In[ ]:

factor_score_dict['OLS'].head(10)


# In[ ]:

import time
import pickle
from CAL.PyCAL import * 

# 运行结果保存pickle的位置
save_dir = "store_data"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#分组数
groups = 10
backtest_results_dict = {}


start_time = time.time()
# -----------回测参数部分开始，可编辑------------
start = '2010-01-01'                       # 回测起始时间
end = '2017-12-31'                         # 回测结束时间
benchmark = 'ZZ500'                        # 策略参考标准
universe = DynamicUniverse('ZZ500')           # 证券池，支持股票和基金
capital_base = 10000000                     # 起始资金
freq = 'd'                              
refresh_rate = Monthly(1)  


accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
}

# ---------------回测参数部分结束----------------

# 把回测参数封装到 SimulationParameters 中，供 quick_backtest 使用
sim_params = quartz.SimulationParameters(start, end, benchmark, universe, capital_base, refresh_rate=refresh_rate, accounts=accounts)
# 获取回测行情数据
data = quartz.get_backtest_data(sim_params)

# 对4种不同类型分别进行测试
for this_type in type_list:
    results = {}
    
    factor_data = factor_score_dict[this_type].reset_index().set_index('tradeDate', drop=True)
    factor_data['ticker'] = factor_data['ticker'].apply(lambda x: x+'.XSHG' if x[:2] in ['60'] else x+'.XSHE')
    
    q_dates = factor_data.index.unique()
    # 调整参数(选取股票的集成因子五分位数)，进行快速回测
    for quantile in range(1, groups+1):

        # ---------------策略逻辑部分----------------

        def initialize(context):                   # 初始化虚拟账户状态
            pass

        def handle_data(context): 
            account = context.get_account('fantasy_account')
            current_universe = context.get_universe('stock')
            pre_date = context.previous_date.strftime("%Y%m%d")
            if pre_date not in q_dates:            
                return

            # 拿取调仓日前一个交易日的因子，并按照相应十分位选择股票
            q = factor_data.ix[pre_date].dropna()
            q = q.set_index('ticker', drop=True)
            q = q.ix[current_universe]

            q_min = q['score'].quantile((quantile-1)*1.0/groups)
            q_max = q['score'].quantile(quantile*1.0/groups)
            my_univ = q[(q['score']>=q_min) & (q['score']<q_max)].index.values

           # 交易部分
            positions = account.get_positions()
            sell_list = [stk for stk in positions if stk not in my_univ]
            for stk in sell_list:
                account.order_to(stk,0)

            # 在目标股票池中的，等权买入
            for stk in my_univ:
                account.order_pct_to(stk, 1.0/len(my_univ))


        # 生成策略对象
        strategy = quartz.TradingStrategy(initialize, handle_data)
        # ---------------策略定义结束----------------

        # 开始回测
        bt, perf = quartz.quick_backtest(sim_params, strategy, data=data)

        # 保存运行结果，1为因子最强组
        results[groups+1-quantile] = {'max_drawdown': perf['max_drawdown'], 'sharpe': perf['sharpe'], 'alpha': perf['alpha'], 'beta': perf['beta'], 'information_ratio': perf['information_ratio'], 'annualized_return': perf['annualized_return'], 'bt': bt}    

        print ('backtesting for type %s group %s..................................' % (this_type, str(quantile)))
    backtest_results_dict[this_type] = results
    
    # 保存该次回测结果为文件
    with open(os.path.join(save_dir, 'backtest_%s.pickle'%this_type), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print ('Done! Time Cost: %s seconds' % (time.time()-start_time))


# In[ ]:

# 读取上述回测的结果，都存在了backtest_results_dict里。
# 如果读者内存过小，不能保证一次运行成功而是分批测试的，需要再写一段读取回测结果pickle文件的代码进行融合

return_df = pd.DataFrame(index=['Group%s'%i for i in range(1, groups+1)], columns=['Return_%s'%(item.upper()) for item in type_list])
for this_type in type_list:
    backtest_res = backtest_results_dict[this_type]
    for group in range(1, groups+1):
        return_df.loc['Group%s'%group]['Return_%s'%(this_type.upper())] = backtest_res[group]['annualized_return']

return_df.loc['Group1-Group%s'%groups] = return_df.loc['Group1'] - return_df.loc['Group%s'%groups]
return_df


# In[ ]:

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')

backtest_origin_indic = [u'alpha', u'beta', u'information_ratio', u'sharpe', u'annualized_return', u'max_drawdown']
backtest_heged_indic = [u'hedged_annualized_return', u'hedged_max_drawdown', u'hedged_volatility']
def plot_backtest_result(results):
    fig = plt.figure(figsize=(10,8))
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.grid()
    ax2.grid()
    
    backtest_pd = pd.DataFrame(index=type_list, columns=backtest_origin_indic+backtest_heged_indic)

    for this_type in results:
        bt = results[this_type][1]['bt']

        data = bt[[u'tradeDate', u'portfolio_value', u'benchmark_return']]
        data['portfolio_return'] = data.portfolio_value / data.portfolio_value.shift(1) - 1.0  # 总头寸每日回报率
        data['portfolio_return'].ix[0] = data['portfolio_value'].ix[0] / 10000000.0 - 1.0
        data['excess_return'] = data.portfolio_return - data.benchmark_return  # 总头寸每日超额回报率
        data['excess'] = data.excess_return + 1.0
        data['excess'] = data.excess.cumprod()  # 总头寸对冲指数后的净值序列
        data['portfolio'] = data.portfolio_return + 1.0
        data['portfolio'] = data.portfolio.cumprod()  # 总头寸不对冲时的净值序列
        data['benchmark'] = data.benchmark_return + 1.0
        data['benchmark'] = data.benchmark.cumprod()  # benchmark的净值序列
        ax1.plot(data['tradeDate'], data[['portfolio']], label=str(this_type))
        ax2.plot(data['tradeDate'], data[['excess']], label=str(this_type))
        
        
        hedged_max_drawdown = max([1 - v / max(1, max(data['excess'][:i + 1])) for i, v in enumerate(data['excess'])])  # 对冲后净值最大回撤
        hedged_volatility = np.std(data['excess_return']) * np.sqrt(252)
        hedged_annualized_return = (data['excess'].values[-1]) ** (252.0 / len(data['excess'])) - 1.0
        
        backtest_pd.loc[this_type] = [results[this_type][1][item] for item in backtest_origin_indic] + [hedged_annualized_return, hedged_max_drawdown, hedged_volatility]

    ax1.legend(loc=0)
    ax2.legend(loc=0)
    ax1.set_ylabel(u"净值", fontproperties=font, fontsize=16)
    ax2.set_ylabel(u"对冲净值", fontproperties=font, fontsize=16)
    ax1.set_title(u"组1选股净值走势", fontproperties=font, fontsize=16)
    ax2.set_title(u"组1选股对冲中证500指数后净值走势", fontproperties=font, fontsize=16)


    cols = [(u'风险指标', u'Alpha'), (u'风险指标', u'Beta'), (u'风险指标', u'信息比率'), (u'风险指标', u'夏普比率'), (u'纯股票多头时', u'年化收益'),
            (u'纯股票多头时', u'最大回撤'), (u'对冲后', u'年化收益'), (u'对冲后', u'最大回撤'), (u'对冲后', u'收益波动率')]
    backtest_pd.columns = pd.MultiIndex.from_tuples(cols)
    backtest_pd.index.name = u'不同类型'
    return backtest_pd


# In[ ]:

results_pd = plot_backtest_result(backtest_results_dict)
results_pd


# In[ ]:



