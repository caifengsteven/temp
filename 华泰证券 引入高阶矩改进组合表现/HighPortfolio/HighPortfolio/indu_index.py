import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from algs import *
from tqdm import tqdm

# %%
df = pd.read_csv('./data/index_shenwan.csv', encoding='gbk')

# %%
base_dir = './data/indu_index'
if not os.path.exists(base_dir): os.makedirs(base_dir)
symbols = df['symbol'].unique()
for symbol in symbols:
    if not os.path.exists(os.path.join(base_dir, '%s.csv'%symbol)):
        sub_df = df[df['symbol'] == symbol]
        sub_df.to_csv(os.path.join(base_dir, '%s.csv'%symbol))

# %%
ms = os.listdir(base_dir)
ms.sort()
main_index = [name for name in ms if name[-5] == '0']

# %%
def read_return_data(ms):
    dfs = [pd.read_csv(os.path.join(base_dir, name), index_col='tradeDate')['closeIndex']
           for name in ms]
    
    dfs = [df.diff() / df.shift(1) for df in dfs]
    dfs = pd.concat(dfs, axis=1)
    dfs.columns = [i.split('.')[0] for i in ms]
    dfs.sort_index(inplace=True)
    # dfs = dfs[dfs.index < '2020-01-01']
    dfs.columns = [name.split('.')[0] for name in ms]
    return dfs.dropna()

# %%
indu_indexs = main_index[:5]
df = read_return_data(indu_indexs)
n_asset = len(indu_indexs)

test_window = 200
step_window = 20
n = df.shape[0]
def get_result(lam_1 = 1, lam_2 = 1):
    tests = []
    for i in tqdm(range(0, n - test_window - step_window, step_window)):
        df_sub = df.iloc[i: i + test_window + step_window, :]
        X_train, df_test = df_sub.iloc[:test_window].copy(), df_sub.iloc[test_window:].copy()
        X_train = X_train.values
        X_test = df_test.values

        # get base optimal
        model = PGP_skew(lam_1=lam_1, lam_2=lam_2, lam_3=0, n_asset=n_asset)
        w, opt = model.fit(X_train)
        df_test['opt_base'] = (X_test * w).sum(axis=1)

        # get 3order optimal
        ws = []
        for lam_3 in [1, 3, 5, 7, 9]:
            model = PGP_skew(lam_1=lam_1, lam_2=lam_2, lam_3=lam_3, n_asset=n_asset)
            w, opt = model.fit(X_train)
            ws.append(w)
        w = np.mean(ws, axis=0)
        df_test['opt_skn'] = (X_test * w).sum(axis=1)
        tests.append(df_test)
    ts_df = pd.concat(tests)
    return ts_df

# %%
ts_11 = get_result(1, 1)
ts_15 = get_result(1, 5)
ts_51 = get_result(5, 1)
# %%
# (ts_51[['opt_base', 'opt_skn']] + 1).cumprod().plot(figsize=(16, 8))
(ts_51+ 1).cumprod().plot(figsize=(16, 8))
plt.show()

# %%
for ts, ty in zip([ts_11, ts_15, ts_51], ['稳健', '激进', '保守']):
    base = get_shape_ratio(ts['opt_base'])
    opt = get_shape_ratio(ts['opt_skn'])
    print('type: %s = base sharpe: %.4f, opt sharpe: %.4f, up_ratio: %.4f'%(ty, base, opt, (opt/base - 1) * 100))