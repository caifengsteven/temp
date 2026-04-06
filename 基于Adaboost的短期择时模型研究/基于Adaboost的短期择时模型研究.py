# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:15:02 2020

@author: Asus
"""
'''
导读
A. 研究目的：本文利用优矿提供的行情数据、宏观行业数据、个股因子数据等，参考兴业证券《系统化资产配置系列之三:基于AdaBoost机器学习算法的市场短期择时策略》（原作者：于明明等）的方法，挑选了一批因子，并分别用CART决策树模型、AdaBoost等机器学习模型进行市场短期择时的研究，探究用机器学习模型来解决因子筛选时的非线性、相关性等一系列传统择时模型遇到的难点问题的效果，以及不同机器学习模型的表现差异。

B. 研究结论：

在不考虑手续费的理想条件下，利用CART决策树模型对通联全A指数在201410月到2019年12月中进行日度择时，纯做多时可获得10.2%的年化收益，日度胜率达到了57%，最大回撤为27.2%；纯做空时获得-1.8%的年化收益，日度胜率为47.8%，最大回撤为28.8%；而如果构造多空的择时模型，可获得8.3%的年化收益，胜率为52.3%，最大回撤为27.1%，对应纯持有全A指数的年化收益为8.9%，最大回撤为49%。CART择时模型在进行做多择时上表现相比做空预测更好，但并未明显超过基准；

进一步，利用adaboost模型对通联全A指数在201410月到2019年12月中进行日度择时，纯做多时可获得12%的年化收益，日度胜率达到了55.8%，最大回撤为23.4%；纯做空时获得4.9%的年化收益，日度胜率为47.6%，最大回撤为24.2%；而如果构造多空的择时模型，可获得22.7%的年化收益，胜率为52.8%，最大回撤为27.5%；

从收益、最大回撤等性能指标来看，adaboost模型无论是在做多择时还是做空择时上，都相比CART模型有更好的表现，远超过基准（持有不动）的表现

C. 文章结构：本文共分为4个部分，具体如下

一、指标数据的获取和计算。

二、利用CART决策树模型进行择时。

三、利用AdaBoost模型进行择时。

四、总结。

D. 时间说明

一、第一部分运行需要5分钟
二、第二部分运行需要10分钟
三、第三部分运行需要20分钟
四、第三部分运行需要1分钟
总耗时35分钟左右
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

'''


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report



save_dir = 'uqer_report_202001_Adaboost'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
data_start_date = '20070101'
data_end_date = '20191231'


'''
第一部分：指标数据的获取和计算
该部分耗时 5分钟
该部分总共计算了46个指标从 <data_start_date> 到 <data_end_date> 之间的每日值，具体内容为：

1.1 取14个宏观指标
1.2 计算9个商品、股指、指数因子指标
1.3 计算23个指标的5日变化率指标
所有指标列表汇总如下：
图片注释

计算后的指标数据存储在<save_dir>/factor.pkl中，样式具体如下：

图片注释

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

1.1 取14个宏观指标

图片注释

'''

# 宏观指标ID
indic_id_str = '1090000537,1090000538,1090000539,1090000540,1090000541,1090000542,1090000554,1090000584,1170000954,1090001390,1090001385,1090001386,1090002102,2020000720'

# 各指标的数值
macro_df = DataAPI.EcoDataProGet(indic_id_str,data_start_date,data_end_date)

macro_df = macro_df[['indicID', 'periodDate', 'dataValue']]
macro_df['tradeDate'] = macro_df['periodDate'].apply(lambda x: x.replace("-", ""))
macro_df['ID'] = macro_df['indicID']
macro_df = macro_df[['ID', 'tradeDate', 'dataValue']].rename(columns={"dataValue":"value"})

# 南华商品指数
nh_idx_df = DataAPI.MktIdxdGet(indexID=u"NHCI.NHCI",beginDate=data_start_date,endDate=data_end_date,field=u"indexID,tradeDate,closeIndex",pandas="1")
nh_idx_df.columns = ['ID', 'tradeDate','value']
nh_idx_df['tradeDate'] = nh_idx_df['tradeDate'].apply(lambda x: x.replace("-", ""))

# 标普500指数
sp500_df = DataAPI.MktIdxdGet(ticker=u"spx",beginDate=data_start_date,endDate=data_end_date,field=u"indexID,tradeDate,closeIndex",pandas="1")
sp500_df.columns = ['ID', 'tradeDate','value']
sp500_df['tradeDate'] = sp500_df['tradeDate'].apply(lambda x: x.replace("-", ""))

# 通联全A指数
tlqa_df = DataAPI.MktIdxdGet(ticker=u"DY0001",beginDate=data_start_date,endDate=data_end_date,field=u"indexID,tradeDate,closeIndex",pandas="1")
tlqa_df.columns = ['ID', 'tradeDate','value']
tlqa_df['tradeDate'] = tlqa_df['tradeDate'].apply(lambda x: x.replace("-", ""))

# 医药行业指数
medic_df = DataAPI.MktIdxdGet(indexID=u"801150.ZICN",beginDate=data_start_date,endDate=data_end_date,field=u"indexID,tradeDate,closeIndex",pandas="1")
medic_df.columns = ['ID', 'tradeDate','value']
medic_df['tradeDate'] = medic_df['tradeDate'].apply(lambda x: x.replace("-", ""))

# 食品饮料行业指数
food_df = DataAPI.MktIdxdGet(indexID=u"801120.ZICN",beginDate=data_start_date,endDate=data_end_date,field=u"indexID,tradeDate,closeIndex",pandas="1")
food_df.columns = ['ID', 'tradeDate','value']
food_df['tradeDate'] = food_df['tradeDate'].apply(lambda x: x.replace("-", ""))

# 计算行情的衍生因子指标

## 市场动量指标, close(t)-close(t-10)
mtm_df = tlqa_df.sort_values(by=['tradeDate'], ascending=True)
mtm_df['mtm'] = mtm_df['value'] - mtm_df['value'].shift(10)
mtm_df = mtm_df[['ID', 'tradeDate', 'mtm']]
mtm_df['ID'] = 'MARKET_MOMENTUM'
mtm_df.columns = ['ID', 'tradeDate', 'value']


## 市场交易活跃指标,用全市场的成交金额代替
active_df = DataAPI.MktIdxdGet(ticker=u"DY0001",beginDate=data_start_date,endDate=data_end_date,field=u"indexID,tradeDate,turnoverValue",pandas="1")
active_df.columns = ['ID', 'tradeDate','value']
active_df['tradeDate'] = active_df['tradeDate'].apply(lambda x: x.replace("-", ""))
active_df['ID'] = 'active_rate'

## Beta分离度, A股全市场股票近12个月beta值的横截面标准差
### 取各个股票的Beta252因子数据
disp_df = get_data_cube(set_universe("A"), ['Beta252'], data_start_date, data_end_date,  freq='1d', style='ast')
disp_df = disp_df.to_frame(filter_observations=False).reset_index().rename(columns={"major":"tradeDate", "minor":"secID"})
### 计算横截面的标准差
disp_df = disp_df.groupby(['tradeDate'])['Beta252'].std().reset_index()
disp_df['tradeDate'] = disp_df['tradeDate'].apply(lambda x: x.replace("-", ""))
disp_df['ID'] = 'beta_disp'
disp_df = disp_df[['ID', 'tradeDate', 'Beta252']]
disp_df.columns = ['ID', 'tradeDate', 'value']

## 过去60日的波动率
volatility_60 = tlqa_df.sort_values(by=['tradeDate'], ascending=True)
volatility_60['vol_60'] = volatility_60['value'].rolling(60).std()
volatility_60 = volatility_60[['ID', 'tradeDate', 'vol_60']]
volatility_60['ID'] = 'VOLATILITY_60'
volatility_60.columns = ['ID', 'tradeDate', 'value']

## 过去120日的波动率
volatility_120 = tlqa_df.sort_values(by=['tradeDate'], ascending=True)
volatility_120['vol_120'] = volatility_120['value'].rolling(120).std()
volatility_120 = volatility_120[['ID', 'tradeDate', 'vol_120']]
volatility_120['ID'] = 'VOLATILITY_120'
volatility_120.columns = ['ID', 'tradeDate', 'value']

## 医药行业超额收益率
### 全A指数的日度收益率
ret_A_df = tlqa_df.copy()
ret_A_df['ret'] = ret_A_df.sort_values(by=['tradeDate'])['value'].pct_change()
ret_A_df = ret_A_df[['ID', 'tradeDate', 'ret']].set_index("tradeDate")

### 医药行业的日度收益率
ret_medic_df = medic_df.copy()
ret_medic_df['ret'] = ret_medic_df.sort_values(by=['tradeDate'])['value'].pct_change()
ret_medic_df = ret_medic_df[['ID', 'tradeDate', 'ret']].set_index("tradeDate")
ret_medic_df['value'] = ret_medic_df['ret'] - ret_A_df['ret']
ret_medic_df.reset_index(inplace=True)
ret_medic_df = ret_medic_df[['ID', 'tradeDate', 'value']]
ret_medic_df['ID'] = 'MEDIC_EXCESS'

## 食品饮料行业超额收益率
ret_food_df = food_df.copy()
ret_food_df['ret'] = ret_food_df.sort_values(by=['tradeDate'])['value'].pct_change()
ret_food_df = ret_food_df[['ID', 'tradeDate', 'ret']].set_index("tradeDate")
ret_food_df['value'] = ret_food_df['ret'] - ret_A_df['ret']
ret_food_df.reset_index(inplace=True)
ret_food_df = ret_food_df[['ID', 'tradeDate', 'value']]
ret_food_df['ID'] = 'FOOD_EXCESS'


factor_df = pd.concat([nh_idx_df,sp500_df,mtm_df,active_df,disp_df,volatility_60,volatility_120,ret_medic_df,ret_food_df], axis=0)

# 将上面的指标合并到一起
total_df = pd.concat([macro_df, factor_df], axis=0)
total_df.index = range(len(total_df))

# 计算每个指标、每日的5日变化率
change_rate_5day = total_df.sort_values(by=['ID','tradeDate'], ascending=True).groupby(['ID'])['value'].rolling(5).apply(lambda x: (x[-1] - x[0])/x[0]).reset_index()
change_rate_5day.rename(columns={"level_1":"idx"},inplace=True)

# 根据change_rate_5day中的idx和total_df中的index对齐，得到tradeDate
date_df = total_df['tradeDate'].reset_index().rename(columns={"index":"idx"})
change_rate_5day = change_rate_5day.merge(date_df, on=['idx'], how='left')
change_rate_5day['ID'] = change_rate_5day['ID'].apply(lambda x: str(x)+"_5d_pct")
change_rate_5day = change_rate_5day[['ID', 'tradeDate', 'value']]


ret_A_df['nxt_ret'] = ret_A_df['ret'].shift(1)
ret_A_df['nxt_direction'] = ret_A_df['nxt_ret'].apply(lambda x: 1 if x>=0 else 0)

# 存储全A指数的每日收益率, column为: ID, ret, nxt_ret, nxt_direction, index为日期(%Y%m%d)
ret_A_df.to_pickle(os.path.join(save_dir, "all_A_ret.pkl"))


## 合并所有的指标，总共46个
factor_df = pd.concat([total_df, change_rate_5day], axis=0).dropna()
factor_df.index = range(len(factor_df))


# 将指标和下一个交易日的涨跌情况对齐，以便于下文使用机器学习模型
factor_df = factor_df.pivot(index='tradeDate', columns='ID', values='value')
factor_df = pd.concat([factor_df, ret_A_df['nxt_direction']], axis=1)
factor_df.dropna(subset=['nxt_direction'], inplace=True)
factor_df.to_pickle(os.path.join(save_dir, 'factors.pkl'))
factor_df.tail()

'''
第二部分：利用CART决策树模型进行择时
该部分耗时 10分钟
该部分实现了CART决策树的择时模型，具体包括以下部分内容:

2.1 CART决策树理论介绍
2.2 模型训练和预测机制说明
2.3 CART决策树的实现代码
2.4 CART决策树的择时效果

(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)


'''

factor_df = pd.read_pickle(os.path.join(save_dir, 'factors.pkl'))
ret_A_df = pd.read_pickle(os.path.join(save_dir, "all_A_ret.pkl"))
factor_df.head()

'''
## 2.3 CART决策树的实现代码
CART决策树的超参数有很多个，其中最重要的是不纯度的度量方法（信息熵或者是基尼系数），以及所允许的最大决策树深度，为了得到更好的效果，下文使用了GridSearchCV来搜寻最佳的超参数，具体为：
- 所有参数中，选择在5层训练集的交叉验证中，得分函数值（准确率）最高的一组
- 事先定义了超参数的取值范围， 不纯度的度量方法取值为[信息熵， 基尼系数]之一，而最大的决策树深度为[5, 15, 20, 25, 30]之一
'''

# 提取出训练数据
def get_train_data_set(factor_df, fit_model_date, start_date='20070601'):
    '''
    factor_df: 特征数据dataframe，列依次为 <指标ID1>, <指标ID2>, .... <指标IDn>, 'nxt_direction', index为日期,%Y%m%d格式
    fit_model_date: 训练模型的日期(下个交易日开始使用本次训练的模型)
    返回: train_x和train_y， np.array类型
    train_x: shape为 （日期数， 特征数)
    train_y: shape为  (日期数, 1)
    
    '''
    fit_model_date = str(fit_model_date).replace("-", "")
    start_date = str(start_date).replace("-", "")
    train_df = factor_df[(factor_df.index>=start_date)&(factor_df.index<fit_model_date)].fillna(100000)
    train_x = np.array(train_df.iloc[:,:-1])  # 指标数据
    train_y = np.array(train_df.iloc[:, -1])  # 训练目标数据
    return train_x, train_y



# 准确率
def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    # 确保预测的数量与结果的数量一致
    if len(truth) == len(pred): 
        # Calculate and return the accuracy as a percent
        # 计算预测准确率（百分比）
        # 用bool的平均数算百分比
        return(truth == pred).mean()*100
    else:
        return 0


# 搜索不同超参数，选取最佳参数构建树模型，在测试集中使用交叉验证方法
def fit_model_k_fold(X, y, cross_len=5, max_depth=[5,10,20], criterion=np.array(['entropy', 'gini'])):
    k_fold = KFold(n_splits=cross_len)
    
    #  构造一个决策树
    clf = DecisionTreeClassifier(random_state=80)
    # 超参数组
    params = {'max_depth':max_depth,'criterion':criterion}

    # 定义目标函数
    scoring_fnc = make_scorer(accuracy_score)

    # 定义好模型超参数优化的框架
    grid = GridSearchCV(clf, param_grid=params,scoring=scoring_fnc,cv=k_fold)

    # 根据数据训练
    grid = grid.fit(X, y)
    return grid.best_estimator_


# 得到适合当前时点的模型
def choose_model(tdate, month_end_list, existing_model, depth_list=range(5,35,5)):
    '''
    tdate: 需要使用模型进行预测的日期, %Y%m%d格式
    month_end_list: 月末交易日的列表, 日期格式为 %Y%m%d
    existing_model: 之前已存在的模型
    depth_list: 训练新模型时，树的深度取值范围
    return: 适用于这一天的模型 useable_model, 更新后的模型 updated_model, 训练样本内的准确率insample_precision
    1. 如果tdate不是月末最后一个交易日，则updated_model为新训练的模型, 否则 updated_model=existing_model
    2. useable_model 永远为 existing_model
    3. 如果tdate不是月末最后一个交易日，则insample_precision为np.nan(延续前值), 否则 insample_precision为新训练的模型的样本内准确率
    '''
    if tdate not in month_end_list: # 非月末日，延续之前的模型
        useable_model = existing_model
        updated_model = existing_model
        insample_precision = np.nan
    else:  # 月末日， 则重新更新模型
        useable_model = existing_model
        # 用start_date到tdate之间的日期训练模型
        x_train, y_train = get_train_data_set(factor_df, tdate, start_date='20070601')
        t_start = time.time()
        updated_model = fit_model_k_fold(x_train, y_train, max_depth=depth_list)
        
        y_true, y_pred = y_train, updated_model.predict(x_train)
        insample_precision = float(classification_report(y_true, y_pred).split("avg / total       ")[1].split("      ")[0])
        t_end = time.time()
        print ('updateing model in %s, max_depth:%s, criterion:%s, time_cost:%s s'%(tdate, updated_model.get_params()['max_depth'],updated_model.get_params()['criterion'], round(t_end-t_start,2)))
    return useable_model, updated_model, insample_precision


start_time = time.time()
# 获取交易日历
trade_df = DataAPI.TradeCalGet(exchangeCD=u"XSHG",endDate=data_end_date,isOpen=u"1",field=u"calendarDate,isMonthEnd",pandas="1")
trade_df['calendarDate'] = trade_df['calendarDate'].apply(lambda x: x.replace("-", ""))
month_end_list = trade_df.query("isMonthEnd==1")['calendarDate'].tolist()
trade_date_list = trade_df['calendarDate'].tolist()


# 指标（特征）中的空缺值用100000填充，代表缺失值
factor_df.fillna(100000, inplace=True)


insample_df = pd.DataFrame()
# 预先训练好模型，保证回测期第一天能够运行起来
model_start_date = '20141027'
x_train, y_train = get_train_data_set(factor_df, model_start_date, start_date='20070601')
existing_model = fit_model_k_fold(x_train, y_train)

y_true, y_pred = y_train, existing_model.predict(x_train)
insample_precision = float(classification_report(y_true, y_pred).split("avg / total       ")[1].split("      ")[0])
insample_df.loc[model_start_date, 'insample_precision'] = insample_precision

# 20141027开始预训练好一个模型
result_dict = {"date":[], "predict_result":[]}
date_list = [x for x in trade_date_list if x > model_start_date]

for tdate in date_list:
    useable_model, updated_model, insample_precision = choose_model(tdate, month_end_list, existing_model)
    insample_df.loc[tdate, 'insample_precision'] = insample_precision
    # 预测下一天的涨跌方向
    day_factor_df = factor_df[factor_df.index==tdate]
    input_x = np.array(day_factor_df.iloc[:, :-1])
    tdate_output = useable_model.predict(input_x)
    result_dict['date'].append(tdate)
    result_dict['predict_result'].append(tdate_output[0])
    existing_model = updated_model

    
predict_df = pd.DataFrame(result_dict)
end_time = time.time()

print ("total time cost:%s"%round((end_time-start_time),2))


# 根据模型的预测值，得到纯做多、纯做空、多空和benchmark（一直持有不动），4个策略下的净值曲线
def get_model_return(predict_df, ret_A_df):
    '''
    predict_df: 预测的明日涨跌dataframe, 列为 date("%Y%m%d")， predict_result(0代表跌，1代表涨)
    ret_A_df: 全A指数的日度收益率，列至少包括 ret(当日收益率), index为日期("%Y%m%d")
    return: 
    all_holding_perf_df: 各个策略的每日净值, 列为: date(%Y%m%d格式), long_only, short_only, long_short, benchmark
    summary_df: 各个策略的表现统计
    '''
    
    daily_ret_df = ret_A_df[['ret']].reset_index()
    daily_ret_df.columns = ['date', 'ret']
    daily_ret_df = daily_ret_df.query("date>@model_start_date")
    
    # 1代表看涨，-1代表看跌
    predict_df['predict_result'] = predict_df['predict_result'].replace(0, -1)
    
    # 第一步，将predict_df中的predict_result往后移动一个单位
    predict_df['direction'] = predict_df['predict_result'].shift(1)
    
    # 第二步，将-1的方向用0填充，得到纯多头组合的每天持仓方向
    long_df = predict_df.copy()
    long_df['holding_direction'] = long_df['direction'].replace(-1, 0)
    long_df = long_df[['date', 'holding_direction']]
    
    # 第三步，将1的方向用0填充，得到纯空头组合的每天持仓方向
    short_df = predict_df.copy()
    short_df['holding_direction'] = short_df['direction'].replace(1, 0)
    short_df = short_df[['date', 'holding_direction']]
    
    
    # 第四步，多空组合的每天持仓方向
    long_short_df = predict_df.copy()
    long_short_df['holding_direction'] = predict_df['direction']
    long_short_df = long_short_df[['date', 'holding_direction']]
    
    # 第五步，构造基准，每天都是正向持有
    benchmark_df = predict_df.copy()
    benchmark_df['holding_direction'] = 1
    benchmark_df = benchmark_df[['date', 'holding_direction']]
    
    
    perf_list = []
    summary_list = []
    name_list = ['long_only', 'short_only', 'long_short','benchmark']
    tcount = 0
    for holding_df in [long_df, short_df, long_short_df,benchmark_df]:
        holding_ret_df = holding_df.merge(daily_ret_df, on=['date'], how='right')
        holding_ret_df['daily_ret'] = holding_ret_df['holding_direction'] * holding_ret_df['ret']
        holding_ret_df['daily_ret'].fillna(0, inplace=True)
        holding_ret_df['daily_perf'] = 1+holding_ret_df['daily_ret']
        # 实际的当日收益方向
        holding_ret_df['actual_direction'] = holding_ret_df['ret'].apply(lambda x: 1 if x>=0 else -1)
        
        # 累计净值
        holding_ret_df['total_perf'] = np.cumprod(holding_ret_df['daily_perf'].values)
        
        # 计算年化收益率, 年化波动率，最大回撤, 风险比率     
        max_drawdown = max(
        [1 - v / max(1, max(holding_ret_df['total_perf'][:i + 1])) for i, v in enumerate(holding_ret_df['total_perf'])])  # 对冲后净值最大回撤
        volatility = np.std(holding_ret_df['daily_ret']) * np.sqrt(250)    
        annualized_return = (holding_ret_df['total_perf'].values[-1]) ** (250.0 / len(holding_ret_df['total_perf'])) - 1.0
    
        ratio = annualized_return / volatility
        max_drawdown = str(round(float(max_drawdown)*100,1)) + "%"
        annualized_return = str(round(float(annualized_return)*100,1)) + "%"
        volatility = str(round(float(volatility)*100,1)) + "%"
        ratio = round(ratio,2)
        # 计算胜率，多(空)头组合：开多(空)头单的日期中，开对的比率, 多空组合：所有日期中，方向预测对的比率
        if name_list[tcount] == 'long_only':
            statistic_df = holding_ret_df[['date', 'holding_direction', 'actual_direction']].query("holding_direction==1")
            win_ratio = len(statistic_df.query("actual_direction==1"))*1.0/len(statistic_df)
            win_ratio = str(round(float(win_ratio)*100,1)) + "%"
        elif name_list[tcount] == 'short_only':
            statistic_df = holding_ret_df[['date', 'holding_direction', 'actual_direction']].query("holding_direction==-1")
            win_ratio = len(statistic_df.query("actual_direction==-1"))*1.0/len(statistic_df)
            win_ratio = str(round(float(win_ratio)*100,1)) + "%"
        elif name_list[tcount] == 'long_short':
            statistic_df = holding_ret_df[['date', 'holding_direction', 'actual_direction']]
            win_ratio = len(statistic_df[statistic_df['actual_direction'] == statistic_df['holding_direction']])*1.0/len(statistic_df)
            win_ratio = str(round(float(win_ratio)*100,1)) + "%"
        else:
            win_ratio = '-'
            
        summary_series = pd.Series({"annual_ret":annualized_return, "annual_volatility":volatility, u"收益风险比":ratio, "max_drawdown":max_drawdown, u"日胜率":win_ratio})
        summary_series.name = name_list[tcount]
        summary_list.append(summary_series)
        
        holding_ret_df = holding_ret_df[['date', 'total_perf']].rename(columns={"total_perf":name_list[tcount]}).set_index('date')
        perf_list.append(holding_ret_df)
        tcount += 1
    # 净值曲线
    all_holding_perf_df = pd.concat(perf_list, axis=1).reset_index()
    # 表现summary
    summary_df = pd.concat(summary_list, axis=1).reset_index()
    summary_df.rename(columns={"index":u"策略表现"}, inplace=True)
    return all_holding_perf_df, summary_df

perf_df, summary_df = get_model_return(predict_df, ret_A_df)
print (summary_df.to_html())
perf_df.index = pd.to_datetime(perf_df['date'], format='%Y%m%d')
_ = perf_df[['long_only', 'short_only', 'long_short','benchmark']].plot(figsize=(16,6), title=u'Net Values of CART-model-selection portfolio and benchark')


'''
从上面的结果可以看出，CART决策树在进行做多预测时，有不错的表现，纯做多时可获得10.2%的年化收益，日度胜率达到了57%，最大回撤为27.2%；纯做空时获得-1.8%的年化收益，日度胜率为47.8%，最大回撤为28.8%；而如果构造多空的择时模型，可获得8.3%的年化收益，胜率为52.3%，最大回撤为27.1%，对应纯持有全A指数的年化收益为8.9%，最大回撤为49%。CART择时模型在进行做多择时上表现相比做空预测更好，但并未明显超过基准，如果考虑实际操作时的手续费，用该择时模型并不能取得超额收益，原因可能有以下几个方面：

模型训练时候发生了过拟合，在样本内表现很好，但是样本外预测的泛化能力很差，这种情况可以通过分析样本内外的准确率等指标进行简单判断；
所选用的指标本身效果不行
机器学习模型并不适用等
下图展示了所用的CART模型在样本内的准确率，大概在71%左右，而根据上面的结果可知，在样本外预测准确率在50%左右，总体来看样本外的准确率有所下降，但是下降的并不是非常的极端，因此出了原因1，原因2和3也是需要进行深入研究的，有兴趣的读者可以将决策树图画出来进行详细的分析

# 展示模型在样本内的准确度
_ = insample_df.dropna().plot(figsize=(16,6), title='CART model insample precision')

第三部分：利用AdaBoost模型进行择时
该部分耗时 20分钟
该部分实现了CART决策树的择时模型，具体包括以下部分内容:

3.1 AdaBoost模型理论介绍
3.2 模型训练和预测机制说明
3.3 AdaBoost模型的实现代码
3.4 AdaBoost模型的择时效果
(深度报告版权归优矿所有，禁止直接转载或编辑后转载。)

3.3 AdaBoost模型的实现代码

AdaBoost模型同样有很多个超参数，除了基分类器中的不纯度的度量方法，基分类器的个数也是很重要的参数，此外基分类器的最大深度理论上也是很重要的参数，但是参考研报中的做法，设定好最大深度为1，因此不需要进行调仓。在使用了GridSearchCV进行调优超参数时，对于AdaBoost中基分类器的参数用"__"表示


'''

from sklearn.ensemble import AdaBoostClassifier
# 搜索不同超参数，选取最佳参数构建树模型，在测试集中使用交叉验证方法
def fit_adamodel_k_fold(X, y, cross_len=5, criterion=np.array(['entropy', 'gini']), n_estimators=[20, 30]):
    '''
    X: 训练数据的输入，np.array格式
    y: 训练数据的正确输出值,np.array格式
    '''
    k_fold = KFold(n_splits=cross_len)
    
    # 定义基分类器，是最大深度为1的CART决策树
    base_classier = DecisionTreeClassifier(max_depth=1)
    ada = AdaBoostClassifier(base_estimator=base_classier)

    params = {"base_estimator__criterion": ["gini", "entropy"],  # 基分类器中，构建CART树的不纯度的度量方法
          "n_estimators": n_estimators} # 基分类器个数
    
    # 定义目标函数
    scoring_fnc = make_scorer(accuracy_score)
    
    # 定义好模型超参数优化的框架
    grid_search_ada = GridSearchCV(ada, param_grid=params, scoring=scoring_fnc,cv=k_fold)
    # 根据数据训练
    grid_search_ada = grid_search_ada.fit(X, y)
    return grid_search_ada.best_estimator_


# 得到适合当前时点的模型
def choose_ada_model(tdate, month_end_list, existing_model, n_estimators=range(5,35,5)):
    '''
    tdate: 需要使用模型进行预测的日期, %Y%m%d格式
    month_end_list: 月末交易日的列表, 日期格式为 %Y%m%d
    existing_model: 之前已存在的模型
    n_estimators: 超参数选择时，基分类器的个数取值范围
    
    return: 
    适用于这一天的模型 useable_model, 更新后的模型 updated_model ,训练样本内的准确率insample_precision
    1. 如果tdate不是月末最后一个交易日，则updated_model为新训练的模型, 否则 updated_model=existing_model
    2. useable_model 永远为 existing_model
    3. 如果tdate不是月末最后一个交易日，则insample_precision为np.nan(延续前值), 否则 insample_precision为新训练的模型的样本内准确率
    '''
    if tdate not in month_end_list: # 非月末日，延续之前的模型
        useable_model = existing_model
        updated_model = existing_model
        insample_precision = np.nan
    else:  # 月末日， 则重新更新模型
        useable_model = existing_model
        # 用start_date到tdate之间的日期训练模型
        x_train, y_train = get_train_data_set(factor_df, tdate, start_date='20070601')
        t_start = time.time()
        updated_model = fit_adamodel_k_fold(x_train, y_train, n_estimators=n_estimators)
        
        y_true, y_pred = y_train, updated_model.predict(x_train)
        insample_precision = float(classification_report(y_true, y_pred).split("avg / total       ")[1].split("      ")[0])
        
        t_end = time.time()
        print ('updateing model in %s, n_estimators:%s, criterion:%s, time_cost:%s s'%(tdate, updated_model.get_params()['n_estimators'],updated_model.get_params()['base_estimator__criterion'], round(t_end-t_start,2)))
    return useable_model, updated_model, insample_precision


start_time = time.time()

ada_insample_df = pd.DataFrame()
# 预先训练好模型，保证回测期第一天能够运行起来
model_start_date = '20141027'
x_train, y_train = get_train_data_set(factor_df, model_start_date, start_date='20070601')
existing_model = fit_adamodel_k_fold(x_train, y_train, n_estimators=[20, 25, 30])

y_true, y_pred = y_train, existing_model.predict(x_train)
insample_precision = float(classification_report(y_true, y_pred).split("avg / total       ")[1].split("      ")[0])
ada_insample_df.loc[model_start_date, 'insample_precision'] = insample_precision


# 20141027开始预训练好一个模型
result_dict = {"date":[], "predict_result":[]}
date_list = [x for x in trade_date_list if x > model_start_date]
for tdate in date_list:
    useable_model, updated_model, insample_precision = choose_ada_model(tdate, month_end_list, existing_model, n_estimators=[20, 25, 30])
    ada_insample_df.loc[tdate, 'insample_precision'] = insample_precision
    # 预测下一天的涨跌方向
    day_factor_df = factor_df[factor_df.index==tdate]
    input_x = np.array(day_factor_df.iloc[:, :-1])
    tdate_output = useable_model.predict(input_x)
    result_dict['date'].append(tdate)
    result_dict['predict_result'].append(tdate_output[0])
    existing_model = updated_model

ada_predict_df = pd.DataFrame(result_dict)
end_time = time.time()

print ("total time cost:%s"%round((end_time-start_time),2))


# 展示adaboost模型的择时效果
ada_perf_df, ada_summary_df = get_model_return(ada_predict_df, ret_A_df)
print (ada_summary_df.to_html())
ada_perf_df.index = pd.to_datetime(ada_perf_df['date'], format='%Y%m%d')
_ = ada_perf_df[['long_only', 'short_only', 'long_short','benchmark']].plot(figsize=(16,6), title=u'Net Values of adaboost-model-selection portfolio and benchark')

'''
从上面的结果可以看出，纯做多时可获得12%的年化收益，日度胜率达到了55.8%，最大回撤为23.4%；纯做空时获得4.9%的年化收益，日度胜率为47.6%，最大回撤为24.2%；而如果构造多空的择时模型，可获得22.7%的年化收益，胜率为52.8%，最大回撤为27.5%，远超过CART模型的表现以及计准的表现，说明AdaBoost在进行短期择时上可以取得很好的效果。下图进一步分析了模型在样本内和样本外的准确性，可以看出在样本内的准确度为69%左右，上面的回测结果显示样本外的胜率为52%，样本外的准确性相比样本内下降的水平同第二章的CART接近，因此也说明CART表现不好，本身模型适用情况的关系更大。

第四部分：总结
   调试 运行
文档
 代码  策略  文档
经过上面的实证分析，CART模型在进行短周期预测时的效果并不好，但是AdaBoost择时模型可以取得2倍于基准（一直持有）的年化收益率，最大回撤也更低（23.4% VS 49%)，收益风险比也从基准的0.37提高到0.96，效果非常明显，是一个很不错的择时模型，当然上述结论只是基于46个指标通过回测得到的，在投入实盘之前还应该进行选用指标的敏感性分析、不同时间区间的测试分析以进一步验证模型的鲁棒性，有兴趣的读者可以自己进行深入研究，本文起一个抛砖引玉的作用不再进行详细的论述

'''
