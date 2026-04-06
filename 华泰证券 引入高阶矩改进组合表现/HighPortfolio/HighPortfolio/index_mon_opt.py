import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from algs import *
from tqdm import tqdm

# %%
bs_dir = './data/index_month'
ms = os.listdir(bs_dir)
#ms = [i for i in ms if '.csv' in i and 'N225' not in i] # 中 欧 美
#ms = [i for i in ms if '.csv' in i and 'N100' not in i] # 中 美 日
ms = [i for i in ms if '.csv' in i and 'GSPC' not in i] # 中 欧 日
n_asset = len(ms)

def read_return_data():
    dfs = [pd.read_csv(os.path.join(bs_dir, name), index_col='Date')['Adj Close']
           for name in ms]

    dfs = [df.diff() / df.shift(1) for df in dfs]
    dfs = pd.concat(dfs, axis=1)
    dfs.columns = [i.split('.')[0] for i in ms]
    dfs.sort_index(inplace=True)
    dfs = dfs[dfs.index < '2020-01-01']
    return dfs.dropna()


df = read_return_data()

# %%
# df_sub = df.iloc[:, :].copy()
# X = df_sub.values
# model = PGP_skew(lam_1=1, lam_2=1, lam_3=0, n_asset=3)
# w, opt = model.fit(X)
# df_sub['opt_base'] = (X * w).sum(axis=1)
# print(w)
#
# model = PGP_skew(lam_1=1, lam_2=1, lam_3=3, n_asset=3)
# w, opt = model.fit(X)
# df_sub['opt_imp'] = (X * w).sum(axis=1)
# print(w)
#
# (df_sub + 1).cumprod().plot()
# plt.show()

# %%
test_window = 24
step_window = 1
n = df.shape[0]
# %%
# lam_1, lam_2= 1, 1
# tests = []
# w_base = []
# w_skn = []
# for i in tqdm(range(0, n - test_window - step_window, step_window)):
#     df_sub = df.iloc[i: i + test_window + step_window, :]
#     X_train, df_test = df_sub.iloc[:test_window].copy(), df_sub.iloc[test_window:].copy()
#     X_train = X_train.values
#     X_test = df_test.values
#
#     # get base optimal
#     model = PGP_skew(lam_1=lam_1, lam_2=lam_2, lam_3=0, n_asset=n_asset)
#     w, opt = model.fit(X_train)
#     w_base.append(w)
#     df_test['opt_base'] = (X_test * w).sum(axis=1)
#
#     # get 3order optimal
#     ws = []
#     for lam_3 in [1, 3, 5, 7, 9]:
#         model = PGP_skew(lam_1=lam_1, lam_2=lam_2, lam_3=lam_3, n_asset=n_asset)
#         w, opt = model.fit(X_train)
#         ws.append(w)
#     w = np.mean(ws, axis=0)
#     w_skn.append(w)
#     df_test['opt_skn'] = (X_test * w).sum(axis=1)
#     tests.append(df_test)
# ts_df = pd.concat(tests)

# %%

def get_result(lam_1=1, lam_2=1):
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
(ts_51 + 1).cumprod().plot(figsize=(16, 8))
plt.show()

# %%
for ts, ty in zip([ts_11, ts_15, ts_51], ['稳健', '激进', '保守']):
    base = get_shape_ratio(ts['opt_base'])
    opt = get_shape_ratio(ts['opt_skn'])
    print('type:%s = base sharpe: %.4f, opt sharpe: %.4f, up_ratio: %.4f' % (ty, base, opt, (opt / base - 1) * 100))
