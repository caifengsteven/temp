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
sh_index = df[df.index > '2013-01-01']['closeIndex']
bs_param = pd.read_csv('./rest/bs_fix_params.csv').T
# %%
def get_hilbert_angle(ts):
    xh = hilbert(ts)
    x_ang = np.angle(xh, deg=False)
    sinx = np.sin(x_ang)
    return -np.sign(sinx)

def get_result_shp(sh_index ,ma = 20, window = 20, is_mid = True, idx = -1):
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
    signal.dropna(inplace= True)
    sig_df = df.loc[signal.index]
    sig_df['signal'] = signal.shift(1)
    return sig_df

# %%
for ty, ma, win in zip(bs_param.index, bs_param[0], bs_param[1]):
    if ty == 'mid':
        sig_df = get_result_shp(sh_index, ma = ma, window= win, is_mid= True)
        (sig_df['signal'] * sig_df['CHGPct'] + 1).cumprod().plot(figsize = (16, 8), label = 'mid')
        (sig_df['CHGPct'] + 1).cumprod().plot(label='sh_index')
        plt.legend()
        plt.show()
    else:
        sig_df = get_result_shp(sh_index, ma= ma, window= win, is_mid=False, idx= int(ty))
        (sig_df['signal'] * sig_df['CHGPct'] + 1).cumprod().plot(figsize=(16, 8), label='ind %s'%ty)
        (sig_df['CHGPct'] + 1).cumprod().plot(label='sh_index')
        plt.legend()
        plt.show()

