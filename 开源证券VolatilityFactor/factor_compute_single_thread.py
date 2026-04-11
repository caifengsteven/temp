import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Process, Manager, Pool

# %%
# sub_df = pd.read_csv('./stk_data/000001.csv', index_col='tradeDate')
# sub_df = sub_df[sub_df.index > '2010-01-01']
with open('./data/date_index.npy', 'rb') as f:
    date_ind = np.load(f, allow_pickle=True)
date_ind.sort()


# 记录字典
# %% 重索引数据，包含停牌日
def reindex_data(sub_df, date_ind):
    begin_ind = sub_df.index[0]
    ind = date_ind[np.where(date_ind == begin_ind)[0][0]:]
    new_df = pd.DataFrame(index=ind)
    for col in sub_df.columns:
        new_df[col] = sub_df[col]
    return new_df


def get_vol_factor(stk_id, begin_date='2010-01-01', n_days=20, lam=0.25):
    res_dict = {}
    # 获取数据
    sub_df = pd.read_csv('./stk_data/%s.csv' % stk_id, index_col='tradeDate')
    if np.any(sub_df.index > begin_date):  # 提前退市股票不计算
        sub_df = sub_df[sub_df.index > begin_date]

        # 整理数据
        stk_df = reindex_data(sub_df, date_ind)
        stk_df['vol'] = stk_df['highestPrice'] / stk_df['lowestPrice'] - 1
        # 剔除一字板
        stk_df['vol'].replace(0, np.nan, inplace=True)

        # 计算因子值
        for ind in range(stk_df.shape[0] - n_days):
            sub_df = stk_df.iloc[ind: ind + n_days]
            next_day = stk_df.iloc[ind + n_days]
            if sub_df['vol'].isnull().sum() < 10:
                # 收盘价较高的 lam
                df_high = sub_df[sub_df['closePrice'] > np.quantile(sub_df['closePrice'].dropna(),
                                                                    1 - lam)]
                v_high = df_high['vol'].mean()

                # 收盘价较低的 lam
                df_low = sub_df[sub_df['closePrice'] < np.quantile(sub_df['closePrice'].dropna(),
                                                                   lam)]
                v_low = df_low['vol'].mean()

                # 理想振幅因子
                v_sp = v_high - v_low
            else:
                v_high = np.nan
                v_low = np.nan
                v_sp = np.nan
            res_dict[(stk_id, next_day.name)] = {'v_high': v_high, 'v_low': v_low, 'v_sp': v_sp,
                                                 'pre_close': sub_df.loc[:, 'closePrice'].iloc[-1],
                                                 'close': next_day.closePrice}
    return res_dict


# %%
import os

ms = os.listdir('./stk_data')
ms.sort()
ms = [i.split('.')[0] for i in ms]

# %%
# res_dict = {}
# n_pool = 8 # 4核8线程
# pool = Pool(n_pool)
# ran = int(len(ms) / n_pool) + 1
# for i in tqdm(range(ran)):
#     sub_ms = ms[i * n_pool : (i + 1) * n_pool]
#     s = pool.map(get_vol_factor, sub_ms)
#     for d in s: res_dict.update(d)
# pd.DataFrame(res_dict).T.to_csv('./data/FactorData.csv')

# %% 单线程
res_dict = {}
for i in tqdm(ms): res_dict.update(get_vol_factor(i))
pd.DataFrame(res_dict).T.to_csv('./data/FactorData.csv')













# %%
# 多核加速
# n_kernel = 4
# ran = int(len(ms) / 4) + 1
# n_stock = len(ms)
# for i in tqdm(range(ran)):
#     if i * n_kernel < n_stock:
#         p1 = Process(target=get_vol_factor, args=(ms[i * n_kernel],))
#     if (i * n_kernel + 1) < n_stock:
#         p2 = Process(target=get_vol_factor, args=(ms[i * n_kernel + 1],))
#     if (i * n_kernel + 2) < n_stock:
#         p3 = Process(target=get_vol_factor, args=(ms[i * n_kernel + 2],))
#     if (i * n_kernel + 3) < n_stock:
#         p4 = Process(target=get_vol_factor, args=(ms[i * n_kernel + 3],))
#
#     if i * n_kernel < n_stock:
#         p1.start()
#     if (i * n_kernel + 1) < n_stock:
#         p2.start()
#     if (i * n_kernel + 2) < n_stock:
#         p3.start()
#     if (i * n_kernel + 3) < n_stock:
#         p4.start()
#
#     if i * n_kernel < n_stock:
#         p1.join()
#     if (i * n_kernel + 1) < n_stock:
#         p2.join()
#     if (i * n_kernel + 2) < n_stock:
#         p3.join()
#     if (i * n_kernel + 3) < n_stock:
#         p4.join()

# %% save data


# %%
# stk_id = '000003'
# sub_df = pd.read_csv('./stk_data/%s.csv'%stk_id, index_col='tradeDate')

# %%
# stk_df = reindex_data(sub_df, date_ind)
# stk_df['vol'] = stk_df['highestPrice']/stk_df['lowestPrice'] - 1
# # 剔除一字板
# stk_df['vol'].replace(0, np.nan, inplace=True)
#
# # %% 计算因子值
#
# n_days = 20
# lam = 0.4
# for ind in range(stk_df.shape[0] - n_days):
#     sub_df = stk_df.iloc[ind: ind + n_days]
#     next_day = stk_df.iloc[ind + n_days]
#     if sub_df['vol'].isnull().sum() < 10:
#         # 收盘价较高的 40%
#         df_high = sub_df[sub_df['closePrice'] > np.quantile(sub_df['closePrice'].dropna(),
#                                                             1 - lam)]
#         v_high = df_high['vol'].mean()
#
#         # 收盘价较低的 40%
#         df_low = sub_df[sub_df['closePrice'] < np.quantile(sub_df['closePrice'].dropna(),
#                                                             lam)]
#         v_low = df_low['vol'].mean()
#
#         # 理想振幅因子
#         v_sp = v_high - v_low
#     else:
#         v_high = np.nan
#         v_low = np.nan
#         v_sp = np.nan
#     res_dict[('000001', next_day.name)] = {'v_high': v_high, 'v_low': v_low, 'v_sp': v_sp,
#                                       'pre_close': sub_df.loc[:, 'closePrice'].iloc[-1],
#                                       'close': next_day.closePrice}
#
# # %%
# pd.DataFrame(res_dict).T
