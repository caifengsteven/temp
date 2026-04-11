import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm

# %%
stk_list = glob('./data/*/*.csv')
df_sh = pd.read_csv('./data/sh_week.csv', dtype={'week': str, 'ticker': str})
df_sh.index = df_sh.pop('week')


# %% get beta
def read_stk_data(path):
    df = pd.read_csv(path, dtype={'week': str, 'ticker': str, 'industryID1': str})
    indu = df['industryID1'].iloc[0]
    return df, indu


def get_beta(df_x, df_y, window=100):
    week_list = df_y.index.tolist()
    week_list.sort()
    n = len(week_list)
    y_arr, x_arr = df_y['chgPct'].values, df_x.loc[df_y.index, 'CHGPct'].values
    alphas, betas, next_weeks = [], [], []
    for i in range(n - window):
        next_week = week_list[i + window]
        x_sub, y_sub = x_arr[i: i + window], y_arr[i: i + window]
        beta = (x_sub * y_sub).sum() / (x_sub * x_sub).sum()
        betas.append(beta), next_weeks.append(next_week)
    df_y.loc[next_weeks, 'beta'] = betas
    return df_y


def read_indu_index():
    indu_dfs = {}
    indu_list = glob('./data/103*.csv')
    for path in indu_list:
        # print(i)
        df_ = pd.read_csv(path, dtype={'week': str, 'ticker': str, 'industryID1': str})
        df_.set_index('week', inplace=True)
        indu_code = df_['industryID1'].unique()[0]
        indu_dfs[indu_code] = df_
    return indu_dfs


indu_dfs = read_indu_index()

# %%
# from pathos.pools import ProcessPool as Pool
# from multiprocessing import Pool

stk_data_dict = {}

# pool = Pool(12)
# step = 64
# n = len(stk_list) // step
# print('reading data')
# for i in tqdm(range(n)):
#     sub_stk_list = stk_list[i * step: (1 + i) * step]
#     dl = pool.map(read_stk_data, sub_stk_list)
#     for df, indu_code in dl:
#         if indu_code not in stk_data_dict.keys():
#             stk_data_dict[indu_code] = []
#         df.set_index('week', inplace=True)
#         df.sort_index(inplace=True)
#         stk_data_dict[indu_code].append(df)
#
# pool.close()

print('reading data')
for path in tqdm(stk_list):
    df_, indu = read_stk_data(path)
    if indu not in stk_data_dict.keys():
        stk_data_dict[indu] = []
    df_.set_index('week', inplace=True)
    df_.sort_index(inplace=True)
    stk_data_dict[indu].append(df_)
# %% get beta
print('get beta')
for key in tqdm(stk_data_dict.keys()):
    dfs = stk_data_dict[key]
    for df_y in dfs:
        if df_y.shape[0] > 200:
            df_y = get_beta(df_sh, df_y)
        else:
            del df_y  # del useless data

# %% get csad
print('get single csad')
for key in tqdm(stk_data_dict.keys()):
    dfs = stk_data_dict[key]
    indu_df = indu_dfs[key]
    for df_stk in dfs:
        df_stk['csad'] = (df_stk['chgPct'] - indu_df.loc[df_stk.index, 'CHGPct']).abs()

# %% get indu csad
print('get single cssd and indu csad')
for key in tqdm(stk_data_dict.keys()):
    dfs = stk_data_dict[key]
    indu_df = indu_dfs[key]
    indu_comp_df = pd.concat(dfs)
    indu_df['csad'] = indu_comp_df.groupby(level='week')['csad'].mean()
    indu_df['mean_ret'] = indu_comp_df.groupby(level='week')['chgPct'].mean()
    for df_stk in dfs:
        df_stk['cssd'] = (df_stk['chgPct'] - indu_df.loc[df_stk.index, 'mean_ret']) ** 2


# %%
def get_indu_cssd_func(x):
    '''
    x is (r_i - r_I)^2
    '''
    p = x.sum() / (len(x) - 1)
    return np.sqrt(p)


# %% get indu cssd
print('get indu cssd')
for key in tqdm(stk_data_dict.keys()):
    dfs = stk_data_dict[key]
    indu_df = indu_dfs[key]
    indu_comp_df = pd.concat(dfs)
    indu_df['cssd'] = indu_comp_df.groupby(level='week')['cssd'].apply(get_indu_cssd_func)

# %% 计算三种羊群效应 CSAD
import statsmodels.api as sm

print('get coef csad ...')
window = 60
for key in tqdm(stk_data_dict.keys()):
    indu_df = indu_dfs[key]
    # 计算
    week_list = indu_df.index.tolist()
    week_list.sort()
    # 数据记录列表
    next_weeks, coefs, p_vals = [], [], []
    for i in range(len(week_list) - window):
        df_xv = np.ones(shape=(window, 3))
        sub_weeks = week_list[i: i + window]
        df_x = df_sh.loc[sub_weeks, 'CHGPct'].abs().copy()
        df_xv[:, 1] = df_x.values
        df_xv[:, 2] = df_x.values ** 2
        df_y = indu_df.loc[sub_weeks, 'csad']
        est = sm.OLS(df_y, df_xv)
        est = est.fit()
        # 回归结果
        p, coef = est.pvalues['x2'], est.params['x2']
        next_weeks.append(week_list[i + window])
        # print(sub_weeks, week_list[i + window])
        coefs.append(coef)
        p_vals.append(p)
    indu_df.loc[next_weeks, 'coef_csad'] = coefs
    indu_df.loc[next_weeks, 'p_csad'] = p_vals

# %% 计算CSSD
print('get coef cssd ...')
window = 20
for key in tqdm(stk_data_dict.keys()):
    indu_df = indu_dfs[key]
    # 计算
    week_list = indu_df.index.tolist()
    week_list.sort()
    # 数据记录列表
    next_weeks, coefs, p_vals, coef2s, p_val2s = [], [], [], [], []
    for i in range(len(week_list) - window):
        df_xv = np.ones(shape=(window, 3))
        sub_weeks = week_list[i: i + window]
        df_x = df_sh.loc[sub_weeks, 'CHGPct']
        df_xv[:, 1] = (df_x < df_x.quantile(0.3)).astype(float).values
        df_xv[:, 2] = (df_x > df_x.quantile(0.7)).astype(float).values
        df_y = indu_df.loc[sub_weeks, 'cssd']
        est = sm.OLS(df_y, df_xv)
        est = est.fit()
        # 回归结果
        p1, coef1, p2, coef2 = est.pvalues['x1'], est.params['x1'], est.pvalues['x2'], est.params['x2']
        next_weeks.append(week_list[i + window])
        coefs.append(coef1)
        p_vals.append(p1)
        coef2s.append(coef2)
        p_val2s.append(p2)
    indu_df.loc[next_weeks, 'coef1_cssd'] = coefs
    indu_df.loc[next_weeks, 'p1_cssd'] = p_vals
    indu_df.loc[next_weeks, 'coef2_cssd'] = coef2s
    indu_df.loc[next_weeks, 'p2_cssd'] = p_val2s

# %% 计算beta
print('get indu beta ...')
for key in tqdm(stk_data_dict.keys()):
    dfs = stk_data_dict[key]
    indu_df = indu_dfs[key]
    indu_comp_df = pd.concat(dfs)
    if 'beta' in indu_comp_df.columns:
        indu_df['beta'] = indu_comp_df.groupby(level='week')['beta'].std()
        mean_beta = indu_comp_df.groupby(level='week')['beta'].mean()
    else:
        del indu_df  # 去除旧的行业，未计算出beta的行业。
    for df_stk in dfs:
        if 'beta' in df_stk.columns:
            df_stk['diff_beta'] = (df_stk['beta'] - mean_beta.loc[df_stk.index]).abs()
        else:
            del df_stk

# %% save data
print('saving new data...')
for key in tqdm(stk_data_dict.keys()):
    dfs = stk_data_dict[key]
    indu_df = indu_dfs[key]
    # 保存 指数数据
    indu_df.to_csv('./data/%s.CSV' % key)
    for df in dfs:
        ticker = df['ticker'].unique()[0]
        df.to_csv('./data/%s/%s.CSV' % (key, ticker))
