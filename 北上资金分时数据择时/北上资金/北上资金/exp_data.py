import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file_list = os.listdir('./data/MMA min data')
# %%
hs_300 = pd.read_csv('./data/hs_300.csv', index_col=0)


def trans_trade_date(trade_data):
    year, month, day = trade_data.split('-')
    return year + month + day


# %%
date_list = hs_300['tradeDate'].apply(lambda x: trans_trade_date(x)).values.tolist()
hs_300.index = date_list

# %%
df = pd.read_csv('./data/MMA min data/getSec20190329.csv', index_col=['dataTime'])
df[df['connectMarket'] == 'SH']['netTurnover'].plot(figsize=(16, 8))
df[df['connectMarket'] == 'SZ']['netTurnover'].plot(figsize=(16, 8))
(df[df['connectMarket'] == 'SH']['netTurnover'] +
 df[df['connectMarket'] == 'SZ']['netTurnover']).plot(figsize=(16, 8))
plt.show()

# %% 观察数据
df = pd.read_csv('./data/MMA min data/getSec20200724.csv', index_col=['dataTime'])
df[df['connectMarket'] == 'SH']['netTurnover'].plot(figsize=(16, 8))
df[df['connectMarket'] == 'SZ']['netTurnover'].plot(figsize=(16, 8))
(df[df['connectMarket'] == 'SH']['netTurnover'] +
 df[df['connectMarket'] == 'SZ']['netTurnover']).plot(figsize=(16, 8))
plt.show()

# %%
for i in date_list:
    # 判定是否有北上数据
    flag = False
    for j in file_list:
        if i in j:
            flag = True
            break

    if flag:
        df = pd.read_csv('./data/MMA min data/getSec%s.csv' % i)
        if df.shape[0] != 0:
            # 取最后两个值，代表沪市，深市的净买入值
            sh, sz = df[df['connectMarket'] == 'SH']['netTurnover'], df[df['connectMarket'] == 'SZ']['netTurnover']
            hs_300.loc[i, 'netVol'] = sz.iloc[-1] + sh.iloc[-1]

# %%
chs_cols = ['tradeDate', 'preCloseIndex', 'openIndex', 'lowestIndex',
            'highestIndex', 'closeIndex', 'netVol']
index_df = hs_300.loc[:, chs_cols].copy()

# %%
index_df['nextOpen'] = index_df['openIndex'].shift(-1)  # T+1 开盘价
index_df['next2Open'] = index_df['openIndex'].shift(-2)  # T+2 开盘价
index_df['ret'] = index_df['next2Open'] / index_df['nextOpen'] - 1  # 平仓收益
index_df.dropna(inplace=True)
# %%
(index_df['ret'] + 1).cumprod().plot(figsize=(16, 8), label='base') # 持有收益
(index_df['netVol'].apply(lambda x: np.sign(x)) * index_df['ret'] + 1).cumprod().plot(label ='strategy') # 策略收益
plt.legend()
plt.show()


