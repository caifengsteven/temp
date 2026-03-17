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

    s_acf = np.array([get_corr(s[i - 2 * lag: i - lag], s[i - lag: i])
                      for i in range(2 * lag, n)])
    return s_acf


def Difuse_func(s, lag=5):
    n = s.shape[0]
    gams = []
    for i in range(2 * lag, n):
        j = s[i - 2 * lag: i]
        fj = np.array([np.square(j[i] - j[i - lag: i]).sum() for i in range(lag, 2 * lag)])
        # reg
        x = np.array([np.ones_like(fj), np.ones_like(fj) + np.arange(lag)]).T
        c, gam = np.linalg.inv(x.T @ x) @ x.T @ fj
        gams.append(gam)
    return np.array(gams)


def Meanstate_func(s, lag=5):
    n = s.shape[0]
    s_mu = np.array([s[i - lag: i].mean() for i in range(lag, n)])

    def get_std(ps, mu, lag=lag):
        sq = np.square(ps - mu).sum() / (lag - 1)
        return np.sqrt(sq)

    s_std = np.array([get_std(s[i - lag: i], mu, lag) for i, mu in zip(range(lag, n), s_mu)])
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
# 超参数可调
def get_buy_signal(df, acf=0.7, F=1.3):
    df['sig_o1'] = df['acf'] < acf
    df['sig_o2'] = df['difuse'] < 0
    df['sig_o3'] = df['sig_F'] < F
    return df


def get_sell_signal(df, acf=0.7, F=1.3):
    df['sig_s1'] = df['acf'] < acf
    df['sig_s2'] = df['difuse'] < 0
    df['sig_s3'] = df['sig_F'] < F
    return df


# %%
def main_result(data, lag=20, F=1.4, acf=0.7,
                if_short=True, is_plot=True):
    ''':arg
    data 是一个dataframe， 其中，收盘价列名为closeIndex
    '''
    s = data['closeIndex']
    df = generate_condition_data(s, index=data.index, lag=lag)

    # 状态描述 在方差范围内
    df['sig_F'] = (df['close'] - df['mu']).apply(np.abs) / df['sigma']

    # 获取开仓与平仓参数
    df = get_buy_signal(df, acf=acf, F=F)
    df = get_sell_signal(df, acf=acf, F=F)

    # 信号
    sopen = df[[i for i in df.columns if 'sig_o' in i]].apply(lambda x: np.all(x.values), axis=1)
    sclose = df[[i for i in df.columns if 'sig_s' in i]].apply(lambda x: np.all(x.values), axis=1)
    slong = df['close'] > df['mu']
    sshort = df['close'] < df['mu']

    # 根据信号的操作
    sig_ser = []
    flag = 0
    for o, c, l, s in zip(sopen, sclose, slong, sshort):
        if o and l:
            flag = 1
        if o and s and if_short:
            flag = -1
        if c and s and not if_short:
            flag = 0  # 当没有做空情况，即无任何操作
        sig_ser.append(flag)

    # result
    close = data['closeIndex']
    ret = close.diff() / close * pd.Series(sig_ser, index=data.index).shift(1)
    ori = close.diff() / close

    if is_plot:
        plt.title('fast alg return')
        (ori + 1).cumprod().plot()
        (ret + 1).cumprod().plot()
        plt.show()

    # 计算对冲后表现
    rets_charge = (ret - ori + 1).cumprod()
    rets = (ret + 1).cumprod()

    # 年化收益
    y_mean = (rets_charge[-1] - 1) / rets_charge.shape[0] * 250
    y_std = (ret - ori).std() * np.sqrt(rets_charge.shape[0] / 250)
    shp = y_mean / y_std

    return rets, rets_charge, shp


# %% 45, 0.7, 1.0
rets, rets_charge, shp = main_result(data, if_short=False, is_plot=True, lag= 45, F = 1, acf= 0.7)
print('------------- long only with default parameters ------------')
print('rets ', rets)
print('rets_charge ', rets_charge)
print('sharpe ', shp)

rets2, rets_charge2, shp2 = main_result(data, if_short=True, is_plot=True)
print('------------- long short with default parameters ------------')
print('rets ', rets2)
print('rets_charge ', rets_charge2)
print('sharpe ', shp2)
# %%
res_dict = {}
flag = 0
for lag in [5, 7, 10, 20, 30, 45]:
    for F in [1, 1.2, 1.4, 1.8]:
        for acf in [0.6, 0.7, 0.8]:
            _, _, shp = main_result(data, lag=lag, F=F, acf=acf,
                                    is_plot=False)
            res_dict[(lag, acf, F)] = shp
            flag += 1
            if flag % 10 == 0:
                print('%s has done' % flag)
# %%
lag, acf, F = pd.Series(res_dict).idxmax()
pd.Series(res_dict).to_csv('./data/grad_fast.csv')

# %%
rets, rets_charge, shp = main_result(data, lag= lag, acf= acf, F= F,
                                     if_short=False, is_plot=True)

print('------------- long only with best parameters ------------')
print('rets ', rets)
print('rets_charge ', rets_charge)
print('sharpe ', shp)

rets, rets_charge, shp = main_result(data, lag= lag, acf= acf, F= F,
                                     if_short=True, is_plot=True)

print('------------- long short with best parameters ------------')
print('rets ', rets)
print('rets_charge ', rets_charge)
print('sharpe ', shp)