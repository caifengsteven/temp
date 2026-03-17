# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# %% read data
data = pd.read_csv('./data/hs_300.csv', index_col=0, parse_dates=['tradeDate'])
data.index = data.pop('tradeDate')

# # %%
# data['closeIndex'].plot()
# plt.show()

# %% feat generate

def ACF_func(s, lag=5):
    n = s.shape[0]

    def get_corr(s1, s2):
        return np.corrcoef(s1, s2)[0, 1]

    s_acf = np.array([get_corr(s[i: i + lag], s[i + lag: i + 2 * lag])
                      for i in range(n - 2 * lag)])
    return s_acf


def Difuse_func(s, lag=5):
    n = s.shape[0]
    gams = []
    for i in range(n - 2 * lag):
        j = s[i: i + 2 * lag]
        fj = np.array([np.square(j[lag + i] - j[i:lag + i]).sum() for i in range(lag)])
        # reg
        x = np.array([np.ones_like(fj), np.ones_like(fj) + np.arange(lag)]).T
        c, gam = np.linalg.inv(x.T @ x) @ x.T @ fj
        gams.append(gam)
    return np.array(gams)


def Meanstate_func(s, lag=5):
    n = s.shape[0]
    s_mu = np.array([s[i: i + lag].mean() for i in range(n - lag)])

    def get_std(ps, mu, lag=lag):
        sq = np.square(ps - mu).sum() / (lag - 1)
        return np.sqrt(sq)

    s_std = np.array([get_std(s[i: i + lag], mu, lag) for i, mu in enumerate(s_mu)])
    return s_mu, s_std


def generate_condition_data(s, index, lag=5):
    acf_ser = pd.Series(ACF_func(s, lag=lag),
                        index=index[2 * lag:], name='acf')
    dif_ser = pd.Series(Difuse_func(s, lag=lag),
                        index=index[2 * lag:], name='difuse')
    mu, sig = Meanstate_func(s, lag=lag)
    mu_ser = pd.Series(mu, index=index[lag:], name='mu')
    sig_ser = pd.Series(sig, index=index[lag:], name='sigma')
    s_ser = pd.Series(s, index=index, name='close')
    return pd.concat([s_ser, acf_ser, dif_ser, mu_ser, sig_ser], axis=1)

# %%
s = data['closeIndex'].values
lag = 20
df = generate_condition_data(s, index=data.index, lag = lag)

# %%
# 状态描述 在方差范围内
df['sig_F'] = (df['close'] - df['mu']).apply(np.abs) / df['sigma']

# %%
# 超参数可调
def get_buy_signal(df, acf=0.7, F=1.3):
    df['sig_o1'] = df['acf'] < acf
    df['sig_o2'] = df['difuse'] < 0
    df['sig_o3'] = df['sig_F'] < F
    df['sig_o4'] = df['mu'] < df['close']
    return df


def get_sell_signal(df, acf=0.7, F=1.3):
    df['sig_s1'] = df['acf'] < acf
    df['sig_s2'] = df['difuse'] < 0
    df['sig_s3'] = df['sig_F'] < F
    df['sig_s4'] = df['mu'] > df['close']
    return df


df = get_buy_signal(df)
df = get_sell_signal(df)

# %%
sb = df[[i for i in df.columns if 'sig_o' in i]].apply(lambda x: np.all(x.values), axis=1)
ss = df[[i for i in df.columns if 'sig_s' in i]].apply(lambda x: np.all(x.values), axis=1)

# %%
sig_ser = []
flag = 0
for b, s in zip(sb, ss):
    if b:
        flag = 1
    if s:
        flag = 0
    sig_ser.append(flag)

# %%
plt.title('fast alg return')
close = data['closeIndex']
ret = close.diff() / close * pd.Series(sig_ser, index= data.index).shift(1)
(close.diff() / close + 1).cumprod().plot()
(ret + 1).cumprod().plot()
plt.show()

