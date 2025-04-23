import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import json

# %%
with open('./component/500.json', 'r') as f:
    com_500 = json.load(f)
stk_list = glob.glob('./ind500_week/*.csv')

# %% 获取数据
year_data = {}
for year in com_500.keys():
    if len(com_500[year]) != 0:
        year_dfs = []
        for stk in com_500[year]:
            path = './ind500_week/%s.csv' % stk.split('.')[0]
            if path in stk_list:
                df = pd.read_csv('./ind500_week\\%s.csv' % stk.split('.')[0], index_col=0)
                df = df.loc[(df.index > int(year) * 100) & (df.index < (int(year) + 1) * 100)]
                df['week'] = df.index.map(str)
                # df.set_index('ts_code', append=True, inplace=True)

                year_dfs.append(df)
        year_df = pd.concat(year_dfs)
        year_df.index = range(year_df.shape[0])

        year_data[year] = year_df


# %%

def read_index():
    df = pd.read_csv('./index_data/hs_500.csv', index_col=0)

    def split_date(x):
        y, m, d = x.split('-')
        return y + m + d

    def split_week(x):
        y, m, w = x.year, x.month, x.week
        if m == 1 and w > 50:
            y -= 1
        if w < 10:
            w = '0' + str(w)
        else:
            w = str(w)
        return str(y) + w

    df['trade_date'] = df['tradeDate'].apply(split_date)
    trade_date = pd.to_datetime(df['tradeDate'])
    df['week'] = trade_date.apply(split_week)
    return df.groupby('week')['CHGPct'].apply(lambda x: (x + 1).prod() - 1)


# %%
index_ser = read_index()
dummy_ser = pd.concat([index_ser.shift(i) for i in range(1, 5)], axis=1)
dummy_ser.columns = ['dummy_%s'%s for s in range(1, 5)]
dummy_ser = dummy_ser.sum(axis = 1)
dummy_ser = dummy_ser.apply(lambda x: 1 if x < 0 else 0)
dummy_ser.name = 'dummy'


# %%
def quantile_beta_diff(x, quantile=10):
    quan10 = np.quantile(x, quantile / 100)
    quan90 = np.quantile(x, 1 - quantile / 100)
    high, low = x[x > quan90].mean(), x[x < quan10].mean()
    return high - low

# %%
def weight_beta_diff(x):
    w = x['value'] / x['value'].sum()
    wb = w * x['beta']
    p = wb * (x['beta'] - wb.sum())**2
    return np.sqrt(p.sum())

# %%
beta_ser = pd.concat([year_data[year].groupby('week')['beta'].apply(lambda x: quantile_beta_diff(x, quantile=10))
                      for year in year_data.keys()])
# beta_ser = pd.concat([year_data[year].groupby('week').apply(lambda x: weight_beta_diff(x))
#                       for year in year_data.keys()])
beta_ser.name = 'beta'

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title('beta dispersion')
beta_ser.plot()
plt.subplot(122)
plt.title('index')
(index_ser.loc[beta_ser.index] + 1).cumprod().plot()
plt.show()

# %%
import statsmodels.api as sm

result = sm.OLS(index_ser.loc[beta_ser.index], sm.add_constant(beta_ser))
result = result.fit()

print(result.summary())

# %%
# result = sm.OLS(index_ser.loc[beta_ser.index],
#                 pd.concat([beta_ser, dummy_ser.loc[beta_ser.index].apply(lambda x: x * beta_ser)], axis=1))
result = sm.OLS(index_ser.loc[beta_ser.index],
                pd.concat([beta_ser, dummy_ser.loc[beta_ser.index] * beta_ser], axis=1))
result = result.fit()

print(result.summary())

# %% strategy- beta > quantile and dummy = 0
signal = pd.concat([beta_ser, dummy_ser.loc[beta_ser.index]], axis=1)
quan_signal = pd.Series([beta_ser.iloc[:i].quantile() for i in range(beta_ser.shape[0] - 1)], index=signal.index[1:])
quan_signal.loc[signal.index[0]] = np.nan
quan_signal.sort_index(inplace= True)
# 当前时间段的beta阈值是从开始时间到上一个时间点时间段的beta序列的中位数。

signal = (signal['beta'] > quan_signal) & (signal['dummy'] == 0)
(index_ser.loc[beta_ser.index] + 1).cumprod().plot()
(index_ser.loc[beta_ser.index] * signal + 1).cumprod().plot()
plt.show()

# %% rolling predict
data_x = sm.add_constant(pd.concat([beta_ser, dummy_ser.loc[beta_ser.index]], axis=1))
data_y = index_ser.loc[beta_ser.index]

# %%
dfs = []
window = 100
for i in range(data_y.shape[0] - window):
    x = data_x.iloc[:i + window]
    y = data_y.iloc[:i + window]
    x_pre = data_x.iloc[[i + window]]
    model = sm.OLS(y, x).fit()
    y_pre = model.predict(x_pre)
    dfs.append(y_pre)
signal = pd.concat(dfs)
(index_ser.loc[signal.index] + 1).cumprod().plot()
((signal > 0) * index_ser.loc[signal.index] + 1).cumprod().plot()
plt.show()

# %%
