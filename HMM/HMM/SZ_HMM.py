# pip install hmmlearn
from hmmlearn.hmm import GaussianHMM
from sklearn import preprocessing  # To center and standardize the data.
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle
import datetime
import os
#os.chdir("D:\\Project\\HMM\\HMM")  # 设定工作目录路径

#  PART I, Data Preparation, 直接拿申万28个一级行业指数数据做HMM， 这一步已经处理，跳过。
# from WindPy import w
# w.start()
# df = pd.read_excel('申万行业.xlsx', sheetname='Sheet1')
# index_pool = df['code'].values.tolist()
# index_data = dict()
# for i in range(len(index_pool)):
#     index_id = index_pool[i]
#     wsd = w.wsd(index_id, "open,high,low,close,pct_chg,turn,mkt_cap_ard", "2007-1-1", "2018-12-23", "unit=1")
#     index = dict(zip(wsd.Fields, wsd.Data))
#     index['Date'] = wsd.Times
#     index_data[index_id] = index
#
# with open('sw_index_data.pickle', 'wb') as handle:
#     pickle.dump(index_data, handle, protocol=pickle.HIGHEST_PROTOCOL


#  PART II, 取单个资产多空策略， 取其中一个指数；

with open('sw_index_data.pickle', 'rb') as handle:
    index_data = pickle.load(handle)

key_index = list(index_data.keys())
i = 0
df = pd.DataFrame.from_dict(index_data[key_index[i]])  # 将字典中改指数数据转为为dataframe
df.index = df['Date']
df = df.drop('Date', axis=1)
df['v1'] = df['CLOSE']/df['OPEN']-1
df['v2'] = df['HIGH']/df['OPEN']-1
df['v3'] = 1 - df['LOW']/df['OPEN']
df['v4'] = df['TURN']
df['v5'] = df['PCT_CHG']
df['v6'] = df['MKT_CAP_ARD']/10000  # 市值转化为以万为单位

#  特征标准化
df['v1'] = preprocessing.scale(df['v1'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
df['v2'] = preprocessing.scale(df['v2'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
df['v3'] = preprocessing.scale(df['v3'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
df['v4'] = preprocessing.scale(df['v4'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
df['v5'] = preprocessing.scale(df['v5'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
df['v6'] = preprocessing.scale(df['v6'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化


# 直观结果分析
numOfHiddenState = 3  # 状态数量为3，上涨-震荡-下跌三种；
X = df[['v1', 'v2', 'v3', 'v4', 'v5', 'v6']].values
model = GaussianHMM(n_components=numOfHiddenState, covariance_type="diag", n_iter=1000).fit(X)
hidden_states = model.predict(X)  # 得到隐藏状态

# Plot the hidden states，直观感受三种状态的分布
plt.figure(figsize=(9, 6))
date = pd.to_datetime(df.index)
for i in range(model.n_components):
    pos = (hidden_states == i)
    plt.plot_date(date[pos], df['CLOSE'][pos], 'o', label='hidden state %d' % i, lw=2)
    plt.legend(loc="best")
plt.show()

#  绘制各种状态的收益分布图
df['state'] = hidden_states
all_state_ret = dict()
for i in range(model.n_components):
    state = (hidden_states == i)  # i = 2
    idx = np.append(0, state[:-1])  # 第二天进行买入操作
    df['state %d_return' % i] = df['PCT_CHG'].multiply(idx, axis=0)
    state_ret = (1 + (df['state %d_return' % i]) / 100).cumprod()
    all_state_ret['state %d_return' % i] = state_ret[-1]
    plt.plot_date(date, (1 + (df['state %d_return' % i]) / 100).cumprod(), '-', label='hidden state %d' % i)
    plt.legend(loc="best")
    fig = plt.gcf()
    fig.savefig('单资产对冲.jpg')
plt.show()
#  输出上涨的状态
buy_signal = max(all_state_ret, key=lambda key: all_state_ret[key])

# 备注：
# HMM无法自己判断谁上涨，谁下跌，要自己判断。
# 直接依据状态多空，对股票进行多空操作。


# PART III: 多个资产多空策略
# 1. 提取交易日数据，从2007/2/1开始
with open('sw_index_data.pickle', 'rb') as handle:
    index_data = pickle.load(handle)

key_index = list(index_data.keys())
i = 0
df = pd.DataFrame.from_dict(index_data[key_index[i]])  # 将字典中改指数数据转为为dataframe
date_list = df['Date'].values.tolist()
first_day = '2007-02-01'
first_day = datetime.datetime.strptime(first_day, "%Y-%m-%d").date()
idx = date_list.index(first_day)
index_data_date_list = date_list  # 股票行情记录的日期
date_list = date_list[idx:]  # 策略中后续会用到的交易日，从2007-2-1日开始

# 定义评价函数
def performance_attr(r, fre):
    import qrisk as qr
    def cal_max_loss(r):
        loss = ((1 + r).cumprod()).min() - 1
        return loss
    pf = dict()
    pf['annual_return'] = qr.annual_return(r,  period=fre)
    pf['annual_volatility'] = qr.annual_volatility(r, period=fre)
    pf['sharpe_ratio'] = qr.sharpe_ratio(r, period=fre)  # risk_free=0,TD=252
    pf['downside_risk'] = qr.downside_risk(r, period=fre)
    pf['max_drawdown'] = qr.max_drawdown(r)
    pf['max_loss'] = cal_max_loss(r)
    pf['sortino_ratio'] = qr.sortino_ratio(r, period=fre)  # required return = 0
    pf['omega_ratio'] = qr.omega_ratio(r)  # risk_free=0.0, required_return=0.0
    pf['tail_ratio'] = qr.tail_ratio(r)
    return pf


# 考虑到实际情况，没必要分组。对所有股票，直接买入上涨概率最高的股票，做空下跌概率最高的股票即可
def divide_group(date_id, index_data):
    # T日， 依据T-5日与T日涨跌情况，将资产分为两组，上涨组和下跌组；
    upp_code, down_code = [], []
    for i in range(len(key_index)):
        df = pd.DataFrame.from_dict(index_data[key_index[i]])  # 转为dataframe
        df.index = df['Date']
        df = df.drop('Date', axis=1)
        idx = list(df.index).index(date_id)
        five_ago = list(df.index)[idx-5]
        sign = df.loc[date_id]['CLOSE']/df.loc[five_ago]['CLOSE'] - 1
        if sign > 0:
            upp_code.append(key_index[i])
        elif sign < 0:
            down_code.append(key_index[i])
    return upp_code, down_code


upp_code, down_code = divide_group(date_list[0], index_data)


# 去极值
def fun_normalizeData(myData):
    myData = myData.copy()
    tmpList = list(myData.keys())
    for stock in tmpList:
        std = np.std(myData[stock])
        MA = np.mean(myData[stock])
        for i in range(len(myData)):
            if myData[stock][i] > MA + 3*std:
                myData[stock][i] = MA + 3*std
            elif myData[stock][i] < MA - 3*std:
                myData[stock][i] = MA - 3*std
    return myData


# 多个资产依然按照单资产方式，不过每天对每一个资产交易，做多或者做空或者不操作。
def hmm_predict(key_index, date_id,  numOfHiddenState):
    # 训练模型
    long_ret = dict()
    short_ret = dict()
    for index_id in key_index:  # 取出T-20, T观测期内，同一组股票的特征
        df = pd.DataFrame.from_dict(index_data[index_id])  # 将字典中改指数数据转为为dataframe
        df.index = df['Date']
        df = df.drop('Date', axis=1)
        df['v1'] = df['CLOSE'] / df['OPEN'] - 1
        df['v2'] = df['HIGH'] / df['OPEN'] - 1
        df['v3'] = 1 - df['LOW'] / df['OPEN']
        df['v4'] = df['PCT_CHG']
        # 指数数据在14年以前缺少换手率、市值等指标，因此这两个指标拿去；个股数据存在，则加上即可
        # df['v5'] = df['TURN']
        # df['v6'] = df['MKT_CAP_ARD'] / 10000  # 市值转化为以万为单位
        df = df.loc[date_id:]
        ts = df.copy()
        # 特征标准化
        ts['v1'] = preprocessing.scale(ts['v1'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
        ts['v2'] = preprocessing.scale(ts['v2'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
        ts['v3'] = preprocessing.scale(ts['v3'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
        ts['v4'] = preprocessing.scale(ts['v4'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
        # ts['v5'] = preprocessing.scale(ts['v5'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
        # ts['v6'] = preprocessing.scale(ts['v6'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化

        # 去极值
        X = fun_normalizeData(ts[['v1', 'v2', 'v3', 'v4']])  # 'v5', 'v6'
        X = X.values
        # Make an HMM instance and execute fit
        model = GaussianHMM(n_components=numOfHiddenState, covariance_type="diag", n_iter=1000).fit(X)
        # Predict the optimal sequence of internal hidden state
        hidden_states = model.predict(X)

        all_state_ret = dict()
        for i in range(model.n_components):
            state = (hidden_states == i)  # i = 2
            idx = np.append(0, state[:-1])  # 第二天进行买入操作
            ts['%d' % i] = ts['PCT_CHG'].multiply(idx, axis=0)
            state_ret = (1 + (ts['%d' % i]) / 100).cumprod()
            all_state_ret['%d' % i] = state_ret[-1]
        # 判断涨跌状态
        long = max(all_state_ret, key=lambda key: all_state_ret[key])
        short = min(all_state_ret, key=lambda key: all_state_ret[key])
        long_ret[index_id] = ts[long].values.tolist()
        short_ret[index_id] = ts[short].values.tolist()

    return long_ret, short_ret


long_ret, short_ret = hmm_predict(key_index, date_id=date_list[0],  numOfHiddenState=3)

long_ts = pd.DataFrame.from_dict(long_ret)
short_ts = pd.DataFrame.from_dict(short_ret)
long_ts.index = date_list
short_ts.index = date_list
port = pd.DataFrame()
port['long'] = long_ts.sum(axis=1)/long_ts.shape[1]
port['short'] = short_ts.sum(axis=1)/short_ts.shape[1]
port['port'] = port['long'] - port['short']
(1+port/100).cumprod().plot()
fig = plt.gcf()
fig.savefig('多资产对冲.jpg')
pf = performance_attr(r=port['port']/100, fre='daily')


#  附录： 修正策略：如果每天预测前，重新训练一次模型。

def adj_hmm_predict(key_index, date_id,  numOfHiddenState):
    # 对所有指数，利用T-20 到T的数据训练，得到上涨、下跌概率；
    # 备注：HMM本身极不稳定，为提高模型实时性，每天train 一次模型
    # 训练模型
    long_stock = []
    short_stock = []
    for index_id in key_index:  # 取出T-20, T观测期内，同一组股票的特征
        df = pd.DataFrame.from_dict(index_data[index_id])  # 将字典中改指数数据转为为dataframe
        df.index = df['Date']
        df = df.drop('Date', axis=1)
        df['v1'] = df['CLOSE'] / df['OPEN'] - 1
        df['v2'] = df['HIGH'] / df['OPEN'] - 1
        df['v3'] = 1 - df['LOW'] / df['OPEN']
        df['v4'] = df['PCT_CHG']
        # 指数数据在14年以前缺少换手率、市值等指标，因此这两个指标拿去；个股数据存在，则加上即可
        # df['v5'] = df['TURN']
        # df['v6'] = df['MKT_CAP_ARD'] / 10000  # 市值转化为以万为单位
        idx = index_data_date_list.index(date_id)
        # five_ago = index_data_date_list[idx-1]
        # twenty_ago = index_data_date_list[idx-20]
        df = df.loc[:date_id]
        ts = df.copy()
        # 特征标准化
        ts['v1'] = preprocessing.scale(ts['v1'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
        ts['v2'] = preprocessing.scale(ts['v2'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
        ts['v3'] = preprocessing.scale(ts['v3'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
        ts['v4'] = preprocessing.scale(ts['v4'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
        # ts['v5'] = preprocessing.scale(ts['v5'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化
        # ts['v6'] = preprocessing.scale(ts['v6'], axis=0, with_mean=True, with_std=True, copy=False)  # 沿截面的特征标准化

        # 去极值
        X = fun_normalizeData(ts[['v1', 'v2', 'v3', 'v4']])  # 'v5', 'v6'
        X = X.values
        # Make an HMM instance and execute fit
        model = GaussianHMM(n_components=numOfHiddenState, covariance_type="diag", n_iter=1000).fit(X)
        # Predict the optimal sequence of internal hidden state
        hidden_states = model.predict(X)
        # print("Transition matrix")
        # print(model.transmat_)
        # print()
        #
        # print("Means and vars of each hidden state")
        # for i in range(model.n_components):
        #     print("{0}th hidden state".format(i))
        #     print("mean = ", model.means_[i])
        #     print("var = ", np.diag(model.covars_[i]))
        #     print()

        all_state_ret = dict()
        for i in range(model.n_components):
            state = (hidden_states == i)  # i = 2
            idx = np.append(0, state[:-1])  # 第二天进行买入操作
            ts['%d' % i] = ts['PCT_CHG'].multiply(idx, axis=0)
            state_ret = (1 + (ts['%d' % i]) / 100).cumprod()
            all_state_ret['%d' % i] = state_ret[-1]
        # 训练模型
        long = max(all_state_ret, key=lambda key: all_state_ret[key])
        short = min(all_state_ret, key=lambda key: all_state_ret[key])

        # long_prob = dict()  # 上涨概率
        # short_prob = dict()  # 下跌概率
        # for i in range(len(key_index)):
        #     pos = 21*(i+1) - 1
        #     index_id = key_index[i]
        #     prob_up = model.predict_proba(X)[pos][int(long)]
        #     prob_down = model.predict_proba(X)[pos][int(short)]
        #     long_prob[index_id] = prob_up
        #     short_prob[index_id] = prob_down
        if hidden_states[-1] == int(long):
            long_stock.append(index_id)
        elif hidden_states[-1] == int(short):
            short_stock.append(index_id)

    return long_stock, short_stock


numOfHiddenState = 3
# 策略：做多上涨状态的指数，做空下跌状态的指数，多空组合，暂时不考虑手续费；
# 等权
port_return = dict()  # 记录组合收益
port_hold = dict()  # 记录组合持仓
# port_return[date_list[0]] = 0
for i in range(len(date_list)-1):
    tem = dict()
    date_id = date_list[i]
    print('Currently back testing on ' + str(date_id))
    long_stock, short_stock = adj_hmm_predict(key_index, date_id, numOfHiddenState)
    #print(long_stock)
    #print(short_stock)
    idx = index_data_date_list.index(date_id)
    idx += 1
    long_ret = 0
    short_ret = 0
    if len(long_stock) > 0:
        for long_index in long_stock:
            long_ret += index_data[long_index]['PCT_CHG'][idx]
        long_ret = long_ret/len(long_stock)
    if len(short_stock) > 0:
        for short_index in short_stock:
            short_ret += -1*index_data[short_index]['PCT_CHG'][idx]
            short_ret = short_ret / len(short_stock)

    daily_ret = long_ret + short_ret
    tem['long'] = long_ret
    tem['short'] = short_ret
    tem['port'] = daily_ret
    port_return[date_list[i+1]] = tem
    port_hold[date_list[i+1]] = [long_stock, short_stock]

rets = pd.DataFrame.from_dict(port_return)
rets = rets.T
(1+rets['port']/100).cumprod().plot()
plt.show()
plt.savefig('1.png')

# 评价指标
pf = performance_attr(r=rets['port']/100, fre='daily')
print('annual return')
print(pf['annual_return'])
print('sharp ratio')
print(pf['sharpe_ratio'])
print('maximum drawdown')
print(pf['max_drawdown'])