
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import datetime
import time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


# In[ ]:

import itertools


# In[ ]:

today = time.strftime("%Y-%m-%d")


# In[ ]:

fit_col = [u'secID', u'ticker', u'tradeDate', u'ratio', u'ind']


# In[ ]:

## 获取交易日历
def get_celedate(begin = '2010-01-01', end = today, symbol = 'day'):
    df_date = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", field=u"")
    df_date = df_date[(df_date['calendarDate'] > begin) & (df_date['calendarDate'] < end)]
    df_date = df_date[df_date['isOpen'] == 1]
    if symbol == 'day':
        return df_date['calendarDate']
    if symbol == 'month':
        return df_date[df_date['isMonthEnd'] == 1]['calendarDate']
    if symbol == 'week':
        return df_date[df_date['isWeekEnd'] == 1]['calendarDate']
    if symbol == 'all':
        return df_date


# In[ ]:

## 获取行业分类
def get_ind_class():
    df_ind = DataAPI.EquIndustryGet(secID=u"",ticker=u"",industryVersionCD=u"010303",industry=u"",industryID1=u"",industryID2=u"",industryID3=u"",intoDate=u"",field=u"",pandas="1")
    dict_ind = {k:v for k, v in df_ind[df_ind['isNew'] == 1][['ticker', 'industryName1']].values}
    return dict_ind
dict_ind = get_ind_class()


# In[ ]:

## 获取因子值
def get_fac_data(date_pre, date_late, field = u''):
    ## 获取数据
    df_fac = DataAPI.MktStockFactorsOneDayGet(tradeDate= date_pre ,secID=u"",ticker=u"",field=field, pandas="1")
    df_price_pre = DataAPI.MktEqudGet(secID=u"",ticker=u"",tradeDate= date_pre ,beginDate=u"",endDate=u"",isOpen="1",field=u"ticker,closePrice",pandas="1")
    df_price_late = DataAPI.MktEqudGet(secID=u"",ticker=u"",tradeDate= date_late ,beginDate=u"",endDate=u"",isOpen="1",field=u"ticker,closePrice",pandas="1")
    
    df_price = pd.merge(df_price_pre, df_price_late, on= 'ticker', how= 'inner')
    df_price['ratio'] = (df_price['closePrice_y'] - df_price['closePrice_x'])  / df_price['closePrice_x']
    dict_ratio = {k:v for k,v in df_price[['ticker', 'ratio']].values}
    ## 填充收益与行业分类
    df_fac['ratio'] = df_fac['ticker'].map(dict_ratio)
    df_fac['ind'] = df_fac['ticker'].map(dict_ind)
    # df_group = df_fac.groupby('ind').apply(lambda x: x[:])
    return df_fac.loc[df_fac['ratio'].dropna().index]


# ## 设计
# 1. 时间选择6月-7月进行回测
# 2. 选择5月到6月有效因子
# 3. 时间维度选取12个月

# In[ ]:

## 取前一个月有效因子
## 第一层删选
date_pre = '2018-04-27'
date_late = '2018-05-31'
df_fac = get_fac_data(date_pre, date_late)


# In[ ]:

fit_col = [u'secID', u'ticker', u'tradeDate', u'ratio', u'ind']
fac_list = [i for i in df_fac.columns if i not in fit_col] 


# In[ ]:

## 强有效
fac_field = []
for i in fac_list:
    val, p = spearmanr(df_fac['ratio'], df_fac[i])
    if p < 0.05 and np.abs(val) > 0.15:
        fac_field.append(i)

## 弱有效
newfac_field = []
for i in fac_list:
    val, p = spearmanr(df_fac['ratio'], df_fac[i])
    if p < 0.05 and np.abs(val) > 0.1:
        newfac_field.append(i)

## 差集候选
diff_field = [i for i in newfac_field if i not in fac_field]


# In[ ]:

df_fac = get_fac_data(date_pre, date_late, field= fac_field + ['ticker', 'tradeDate'])


# ## 标准化过程
# 1. 去极值
# 2. 标准化
# 3. 打label

# In[ ]:

## 标准化-去空值-生成训练集
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

def winsorize_by_date(cdate_input, fac_list):
    '''
    按照[dm+5*dm1, dm-5*dm1]进行winsorize
    参数:
        cdate_input: 某一期的因子值的dataframe
    返回:
        DataFrame, 去极值后的因子值
    '''
    media_v = cdate_input.median()
    for a_factor in fac_list:
        dm = media_v[a_factor]
        new_factor_series = abs(cdate_input[a_factor] - dm)  # abs(di-dm)
        dm1 = new_factor_series.median()
        upper = dm + 5 * dm1
        lower = dm - 5 * dm1
        cdate_input[a_factor] = cdate_input[a_factor].apply(lambda x: paper_winsorize(x, upper, lower))
    return cdate_input

def stand_data(df, fac_list):
    df_win = winsorize_by_date(df, fac_list).dropna()
    scale = StandardScaler()
    df_std = pd.DataFrame(scale.fit_transform(df_win[fac_list]), index= df_win.index, columns= fac_list)
    return pd.concat([df_std, df_win[[col for col in df_win.columns if col not in fac_list]]], axis = 1)

def label_data(df, tile = 0.25):
    new_df = df.copy()
    bot_val = new_df['ratio'].quantile(tile)
    top_val = new_df['ratio'].quantile(1 - tile)
    
    def label(x, top_val, bot_val):
        if x > top_val:
            x = 1
        elif x < bot_val:
            x = 0
        else:
            x = np.nan
        return x
    
    new_df['label'] = new_df['ratio'].apply(lambda x: label(x, top_val, bot_val))
    return new_df.dropna()


# In[ ]:

def get_train_data(num_day, end = today ,symbol = 'month', fac_field = fac_field):
    df_fac_list = [get_fac_data(date_pre, date_late, field= fac_field + ['ticker', 'tradeDate']) for date_pre, date_late in zip(get_celedate(symbol=symbol, end= end)[-num_day-1:-1], get_celedate(symbol= symbol , end= end)[-num_day:])]
    df_std_list = [stand_data(df, fac_field) for df in df_fac_list]
    df_label_list = [label_data(df) for df in df_std_list]
    return pd.concat(df_label_list)


# In[ ]:

df_data = get_train_data(12, end = '2018-07-01', fac_field= newfac_field)


# ## 训练，测试集合的选择
# 1. 选择最后一个月最为观察量。
# 2. 倒数第二个月作为test集合
# 3. 前面的集合作为train_data

# In[ ]:

def split_train_test(df, x_cols, y_col):
    new_df  = df.copy()
    time_list = new_df['tradeDate'].unique()
    time_list.sort()
    train_df = new_df[new_df['tradeDate'] < time_list[-2]]
    test_df = new_df[new_df['tradeDate'] == time_list[-2]]
    val_df = new_df[new_df['tradeDate'] == time_list[-1]]
    return (train_df[x_cols].values, train_df[y_col].values), (test_df[x_cols].values, test_df[y_col].values),  (val_df[x_cols].values, val_df[y_col].values)


# In[ ]:

(x_train, y_train) , (x_test, y_test) , (x_val, y_val)= split_train_test(df_data, x_cols = fac_field, y_col = 'label')


# In[ ]:

from sklearn.ensemble import RandomForestClassifier


# In[ ]:

def get_alg(alg, fac_field = fac_field):
    (x_train, y_train) , (x_test, y_test) , (x_val, y_val)= split_train_test(df_data, x_cols = fac_field, y_col = 'label')
    alg = RandomForestClassifier( random_state= 2018,  n_estimators= 50, max_depth= 8)
    alg = alg.fit(x_train, y_train)
    print('train_acc',alg.score(x_train, y_train), 'test_acc', alg.score(x_test, y_test))
    ## 估计月的准确率
    print('val_acc', alg.score(x_val, y_val))
    return alg


# In[ ]:

alg = get_alg(RandomForestClassifier(max_depth= 8, n_estimators= 50), fac_field= newfac_field)


# ## 我们将HS300作为评价指标，从全A股市场选择300只与HS300作为对比。

# In[ ]:

## 预测六月份股票
date = '2018-05-31'
df_pre =  get_fac_data(date, date,field= newfac_field + ['ticker', 'tradeDate'])
df_pre = stand_data(df_pre, fac_field)


# In[ ]:

ticker_list = pd.Series(alg.predict_proba(df_pre[newfac_field])[:, 1], index= df_pre['ticker'])
ticker_list.sort(ascending= False)


# In[ ]:

ticker_list = ticker_list[:300].index
signal_list = DataAPI.MktEqudGet(secID=u"",ticker=ticker_list ,tradeDate=date ,beginDate=u"",endDate=u"",isOpen="",field=u"secID",pandas="1")["secID"].values


# In[ ]:

start = '2018-06-01'                       # 回测起始时间
end = '2018-07-01'                         # 回测结束时间
universe = DynamicUniverse('A')        # 证券池，支持股票、基金、期货、指数四种资产
benchmark = 'HS300'                        # 策略参考标准
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟
  
# 配置账户信息，支持多资产多账户
accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
}
  
def initialize(context):
    pass
  
# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context):    
    
    target_position = signal_list
    # 获取当前账户信息
    account = context.get_account('fantasy_account')   
    current_position = account.get_positions(exclude_halt=True)       
     
    # 根据目标持仓权重，逐一委托下单
    for stock in target_position:
        account.order(stock, 10000)


# ## 利用遗传算法改进过程

# In[ ]:

## 遗传算法过程
### 初代种群生成
def ga_generate_ori(del_prob, add_prob,ori_field, add_field):
    new_field = []
    ## 选取原始特征过程
    for i in ori_field:
        prob = np.random.uniform(0, 1)
        if prob < del_prob:
            new_field.append(True)
        else:
             new_field.append(False)
    ## 添加新特征过程
    for j in add_field:
        prob = np.random.uniform()
        if prob < add_prob:
            new_field.append(True)
        else:
            new_field.append(False)
    new_field = np.array(new_field)
    return new_field

## 种群交配过程
def ga_cross_next_group(ori_group, dict_score = 'ori', change_prob = 0.2):
    new_dict = ori_group.copy()
    if dict_score == 'ori':
        score = 1.0 / len(ori_group)
        dict_score = {k:score for k in ori_group.keys()}
    g, p = np.array([[k, v] for k, v in dict_score.items()]).T
    flag = max(ori_group.keys())
    ## 按照种群分数进行选择交配
    ## 选择交配种群
    cross_group = np.random.choice(g, size = 5, p = p, replace= False)
    for (fa,mo) in itertools.combinations(cross_group, 2):
        flag += 1
        fa_code, mo_code = ori_group[fa], ori_group[mo]
        ## 随机选择切分点
        cut_point = np.random.randint(1, len(fa_code)-1)
        ## 切分基因
        fa_code0, fa_code1 = fa_code[:cut_point], fa_code[cut_point:]
        mo_code0, mo_code1 = mo_code[:cut_point], mo_code[cut_point:]
        # print(fa_code0, mo_code1)
        ## 基因交换
        new1 = np.hstack([fa_code0, mo_code1])
        ## 变异过程
        prob = np.random.uniform(0, 1)
        if prob < change_prob:
            ## 随机挑一个基因点
            change_point = np.random.randint(0, len(fa_code))
            ## 改变该点的值
            new1[change_point] = not new1[change_point]
        new_dict[flag] = new1
    return new_dict

## 对每一个个体评分
def ga_get_score(alg, df_data, x_cols):
    (x_train, y_train) , (x_test, y_test) , (x_val, y_val)= split_train_test(df_data, x_cols = x_cols, y_col = 'label')
    alg = alg.fit(x_train, y_train)
    train_score, test_score = alg.score(x_train, y_train), alg.score(x_test, y_test)
    ## 评价取 0.2的训练集与 0.8的测试集
    # print('val_acc', alg.score(x_val, y_val))
    return alg, 0.2 * train_score + 0.8 * test_score

## 种群个体能力评价
def ga_evalue_group(group, evalue_df, evalue_col):
    score_dict = {}
    for g, code in group.items():
        cols = evalue_col[code]
        _, score = ga_get_score(alg = RandomForestClassifier( random_state= 2018,  n_estimators= 50, max_depth= 8), df_data = evalue_df ,x_cols = cols)
        score_dict[g] = score
    return score_dict

## 丢弃弱者
def ga_kill_group(ori_group, dict_score):
    ## 二代目
    sub_group = ga_cross_next_group(ori_group, dict_score= dict_score)
    ## 评价
    score_dict = ga_evalue_group(sub_group, df_data, evalue_cols)
    score_se = pd.Series(score_dict)
    score_se = score_se.sort_values(ascending= False)[:10] / (score_se.sort_values(ascending= False)[:10].sum())
    score_dict = dict(score_se)
    liv_group = {i:sub_group[i] for i in score_dict.keys()}
    print('开启贤者模式')
    return liv_group, score_dict


# In[ ]:

alg = RandomForestClassifier( random_state= 2018,  n_estimators= 50, max_depth= 8)
np.random.seed(2018)
## 初始化过程
ori_field = fac_field
add_field = diff_field
evalue_cols = np.array(ori_field + add_field)

## 随机产生初代子类
group_num = 10
del_prob = 0.7
add_prob = 0.3
ori_group = {i:ga_generate_ori(del_prob, add_prob, ori_field, add_field) for i in range(group_num)}

## 产生第一代杂交类
for i in range(6):
    if i == 0:
        sub, sco = ga_kill_group(ori_group, 'ori')
    else:
        sub, sco = ga_kill_group(sub, sco)


# In[ ]:

best_code = pd.Series(sco).sort_values()[-1:].index[0]
best_field = list(evalue_cols[sub[best_code]])


# In[ ]:

alg = get_alg(RandomForestClassifier(max_depth= 8, n_estimators= 50), fac_field= best_field)


# In[ ]:

## 预测六月份股票
date = '2018-05-31'
df_pre =  get_fac_data(date, date, field= best_field + ['ticker', 'tradeDate'])
df_pre = stand_data(df_pre, best_field)
ticker_list = pd.Series(alg.predict_proba(df_pre[best_field])[:, 1], index= df_pre['ticker'])
ticker_list.sort(ascending= False)


# In[ ]:

ticker_list = ticker_list[:300].index
signal_list = DataAPI.MktEqudGet(secID=u"",ticker=ticker_list ,tradeDate=date ,beginDate=u"",endDate=u"",isOpen="",field=u"secID",pandas="1")["secID"].values


# In[ ]:

start = '2018-06-01'                       # 回测起始时间
end = '2018-07-01'                         # 回测结束时间
universe = DynamicUniverse('A')        # 证券池，支持股票、基金、期货、指数四种资产
benchmark = 'HS300'                        # 策略参考标准
freq = 'd'                                 # 策略类型，'d'表示日间策略使用日线回测，'m'表示日内策略使用分钟线回测
refresh_rate = 1                           # 调仓频率，表示执行handle_data的时间间隔，若freq = 'd'时间间隔的单位为交易日，若freq = 'm'时间间隔为分钟
  
# 配置账户信息，支持多资产多账户
accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
}
  
def initialize(context):
    pass
  
# 每个单位时间(如果按天回测,则每天调用一次,如果按分钟,则每分钟调用一次)调用一次
def handle_data(context):    
    
    target_position = signal_list
    # 获取当前账户信息
    account = context.get_account('fantasy_account')   
    current_position = account.get_positions(exclude_halt=True)       
     
    # 根据目标持仓权重，逐一委托下单
    for stock in target_position:
        account.order(stock, 10000)


# ## 结论
# 通过遗传算法，改进了模型的表现。
