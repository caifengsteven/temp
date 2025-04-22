import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Regret import Regret_Minimize
from tqdm import tqdm

# %%
thread = 0.55

def get_Regret_result(df, K = 0.8):
    pre_pt = df['closeIndex'].iloc[:-1].values
    pt = df['closeIndex'].iloc[1:].values

    model = Regret_Minimize(K)
    model.fit(pt, pre_pt)

    res_df = df.iloc[1:].loc[:, ['closeIndex', 'CHGPct']]
    res_df['alpha_long'] = model.alpha_longs
    res_df['alpha_short'] = model.alpha_shorts
    res_df['alpha_long'] = res_df['alpha_long'].shift(1)
    res_df['alpha_short'] = res_df['alpha_short'].shift(1)


    res = (res_df['CHGPct'] * (res_df['alpha_long'] > thread).astype(int)
     - res_df['CHGPct'] * (res_df['alpha_short'] > thread).astype(int)
     + 1).prod()
    return res, res_df

# %%
df = pd.read_csv('./data/sh_index.csv', index_col=0, parse_dates=['tradeDate'])
df.index = df.pop('tradeDate')
df_train = df[(df.index > '2008-01-01') & (df.index < '2013-01-01')]

ret_dict = {}
for K in tqdm(np.linspace(0.1, 1, 90, endpoint= False)):
    ret,_ = get_Regret_result(df_train, K= K)
    ret_dict[K] = ret

# %%
K = pd.Series(ret_dict).idxmax()
# K = 0.8
df_test = df[df.index > '2013-01-01']

_, res_df = get_Regret_result(df_test, K = K)

# %%
(res_df['CHGPct'] + 1).cumprod().plot(figsize= (16, 8))
(res_df['CHGPct'] * (res_df['alpha_long'] > thread).astype(int) + 1).cumprod().plot()
plt.title(u'k = %s long return'%K)
plt.legend()
plt.show()

# %%
(res_df['CHGPct'] + 1).cumprod().plot(figsize= (16, 8))
(res_df['CHGPct'] * (res_df['alpha_long'] > thread).astype(int)
 - res_df['CHGPct'] * (res_df['alpha_short'] > thread).astype(int)
 + 1).cumprod().plot()
plt.title(u'k = %s long-short return'%K)
plt.legend()
plt.show()