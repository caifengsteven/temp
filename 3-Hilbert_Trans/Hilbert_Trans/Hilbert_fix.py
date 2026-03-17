import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from units import *
from tqdm import tqdm

# %%
df = pd.read_csv('./data/sh_index.csv', index_col=0)
df.index = df.pop('tradeDate')

# %%
sh_index = df[(df.index > '2005-01-01') & (df.index < '2013-05-05')]['closeIndex']


# %%
def get_hilbert_angle(ts):
    xh = hilbert(ts)
    x_ang = np.angle(xh, deg=False)
    sinx = np.sin(x_ang)
    return -np.sign(sinx)


def get_result_shp(sh_index, ma=20, window=20, is_mid=True, idx=-1):
    sh_ma = sh_index.rolling(ma).mean()
    sh_df = sh_ma.diff()
    sh_df.dropna(inplace=True)
    n = sh_df.shape[0]
    xh = hilbert(sh_df.values)
    # 固定时间窗口
    if is_mid:
        sigs = np.zeros(n - window)
        for i in range(n - window):
            ts = sh_df.values[:window + i]
            sigs[i] = get_hilbert_angle(ts)[int(window / 2)]
        signal = pd.Series(sigs, index=sh_df.index[window:])
    else:
        sigs = np.zeros(n - window)
        for i in range(n - window):
            ts = sh_df.values[:window + i]
            sigs[i] = get_hilbert_angle(ts)[idx]
        signal = pd.Series(sigs, index=sh_df.index[window:])
    signal.name = 'signal'
    signal.dropna(inplace=True)
    sig_df = df.loc[signal.index]
    sig_df['signal'] = signal.shift(1)
    ret = sig_df['CHGPct'] * sig_df['signal']
    cum_ret = (ret + 1).cumprod()
    return get_shape_ratio(ret), cum_ret


# %% 由于变换的对称性，此处选择中位数的角度应该是最符合逻辑的
bs_param = {}

sharpe_dict, cumret_dict = {}, {}
for ma in tqdm([10, 15, 20, 30, 45, 60, 80, 100, 120, 200]):
    for window in [10, 15, 20, 30, 45, 60, 80, 100, 120, 200]:
        a, b = get_result_shp(sh_index, ma, window, is_mid=True)
        sharpe_dict[(ma, window)], cumret_dict[(ma, window)] = a, b[-1]

pd.Series(sharpe_dict).unstack().to_csv('./rest/shp_mid.csv')
pd.Series(cumret_dict).unstack().to_csv('./rest/ret_mid.csv')
print('best params')
ma, win = pd.Series(sharpe_dict).idxmax()
bs_param['mid'] = (ma, win)

# %%
ret, cum_ret = get_result_shp(sh_index, ma=ma, window=win, is_mid=True)
# 训练数据寻找最优参数
cum_ret.plot(figsize=(16, 8), label='Training Result')
sig_df = df.loc[cum_ret.index]
(sig_df['CHGPct'] + 1).cumprod().plot(figsize=(16, 8), label='Sh_index Result')
plt.title('mid')
plt.legend()
plt.show()

# %% 按照研报的逻辑，还是应该选择最后几个作为信号,为此，同时做测试
idx = -1
sharpe_dict, cumret_dict = {}, {}
for ma in tqdm([10, 15, 20, 30, 45, 60, 80, 100, 120, 200]):
    for window in [10, 15, 20, 30, 45, 60, 80, 100, 120, 200]:
        a, b = get_result_shp(sh_index, ma, window, is_mid=False, idx=idx)
        sharpe_dict[(ma, window)], cumret_dict[(ma, window)] = a, b[-1]
pd.Series(sharpe_dict).unstack().to_csv('./rest/shp_%s.csv' % idx)
pd.Series(cumret_dict).unstack().to_csv('./rest/ret_%s.csv' % idx)
print('best params')
ma, win = pd.Series(sharpe_dict).idxmax()

#
ret, cum_ret = get_result_shp(sh_index, ma=ma, window=win, is_mid=False, idx=idx)
# 训练数据寻找最优参数
cum_ret.plot(figsize=(16, 8), label='Training Result')
sig_df = df.loc[cum_ret.index]
(sig_df['CHGPct'] + 1).cumprod().plot(figsize=(16, 8), label='Sh_index Result')
plt.title('%s index' % idx)
plt.legend()
plt.show()
bs_param['%s'%idx] = (ma, win)

# %%
pd.DataFrame(bs_param).to_csv('./rest/bs_fix_params.csv', index= False)