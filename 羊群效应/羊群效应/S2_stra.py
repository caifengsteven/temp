import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# %%
stk_list = glob('./data/*/*.CSV')


def read_stk_data(path):
    df = pd.read_csv(path, dtype={'week': str, 'ticker': str, 'industryID1': str})
    df['prePct'] = df['chgPct'].shift(1)
    indu = df['industryID1'].iloc[0]
    return df, indu


def read_indu_index():
    indu_dfs = {}
    indu_list = glob('./data/103*.CSV')
    for path in indu_list:
        # print(i)
        df_ = pd.read_csv(path, dtype={'week': str, 'ticker': str, 'industryID1': str})
        df_['prePct'] = df_['CHGPct'].shift(1)
        df_.set_index('week', inplace=True)
        indu_code = df_['industryID1'].unique()[0]
        indu_dfs[indu_code] = df_
    return indu_dfs

# %% 读取数据
stk_data_dict = {}
for path in stk_list:
    df_, indu = read_stk_data(path)
    if indu not in stk_data_dict.keys():
        stk_data_dict[indu] = []
    stk_data_dict[indu].append(df_)

for key in stk_data_dict.keys():
    indu_comp_df = pd.concat(stk_data_dict[key])
    indu_comp_df.set_index(['week', 'ticker'], inplace= True)
    stk_data_dict[key] = indu_comp_df

indu_dfs = read_indu_index()
# %% beta 策略
indu_df = pd.concat(indu_dfs.values())
indu_df.set_index('industryID1', append = True, inplace= True)
indu_df = indu_df[indu_df.index.get_level_values('week') >= '20140101']
week_list = indu_df.index.get_level_values('week').unique().tolist()
week_list.sort()


ret_beta_dict = {}
chs_beta_dict = {}
for week in week_list:
    temp_chs = []
    sub_df = indu_df.loc[week]
    # 选出标准差最小的前6个行业
    indu_sub_df = sub_df.sort_values('beta').iloc[:6]
    # 判断涨跌幅是否小于4%
    indu_sub_df = indu_sub_df[indu_sub_df['prePct'].abs() < 0.04]
    indu_codes = indu_sub_df.index.tolist()
    for indu_code in indu_codes:
        indu_comp_df = stk_data_dict[indu_code].loc[week]
        # 判断行业是否有龙头
        if indu_comp_df['prePct'].max() > 0.13:
            quan_50 = indu_comp_df['diff_beta'].quantile(0.5)
            temp_df = indu_comp_df[(indu_comp_df['prePct'] < 0.0007) & (indu_comp_df['diff_beta'] < quan_50)]
            temp_df = temp_df.sort_values('diff_beta').iloc[:5]
            temp_chs.append(temp_df)
    if len(temp_chs) != 0:
        temp_df = pd.concat(temp_chs)
        ret_beta_dict[week] = temp_df['chgPct'].mean()
        chs_beta_dict[week] = temp_df
    else:
        ret_beta_dict[week] = 0

(pd.Series(ret_beta_dict).fillna(0) + 1).cumprod().plot(figsize=(16, 8), label='beta')
plt.title('beta return')
plt.legend()
plt.show()

# %% CSAD 策略
ret_csad_dict = {}
chs_csad_dict = {}
for week in week_list:
    temp_chs = []
    sub_df = indu_df.loc[week]
    # 选出第二个参数为负且p , 0.5 的行业
    indu_sub_df = sub_df[(sub_df['coef_csad'] < 0) & (sub_df['p_csad'] < 0.5)].sort_values('p_csad')
    # 涨跌幅小于13的前2个行业
    indu_sub_df = indu_sub_df[indu_sub_df['prePct'].abs() < 0.13].iloc[:2]
    indu_codes = indu_sub_df.index.tolist()
    for indu_code in indu_codes:
        indu_comp_df = stk_data_dict[indu_code].loc[week]
        # 判断行业是否有龙头
        if indu_comp_df['prePct'].max() > 0.11:
            temp_df = indu_comp_df[(indu_comp_df['prePct'] < 0.11)]
            temp_df = temp_df.sort_values('csad').iloc[:6]
            temp_chs.append(temp_df)
    if len(temp_chs) != 0:
        temp_df = pd.concat(temp_chs)
        ret_csad_dict[week] = temp_df['chgPct'].mean()
        chs_csad_dict[week] = temp_df
    else:
        ret_csad_dict[week] = 0

(pd.Series(ret_csad_dict).fillna(0) + 1).cumprod().plot(figsize=(16, 8), label='csad')
plt.title('csad return')
plt.legend()
plt.show()

# %% CSSD 策略
chs_cssd_dict = {}
ret_cssd_dict = {}
for week in week_list:
    temp_chs = []
    sub_df = indu_df.loc[week]
    # 选出第二个参数为负且p , 0.7 的行业
    indu_sub_df = sub_df[(sub_df['coef2_cssd'] < 0) & (sub_df['p2_cssd'] < 0.7)]
    # 涨跌幅小于13的前3个行业
    indu_sub_df = sub_df.sort_values('coef1_cssd').iloc[:3]

    indu_sub_df = indu_sub_df[indu_sub_df['prePct'].abs() < 0.04]
    indu_codes = indu_sub_df.index.tolist()
    for indu_code in indu_codes:
        indu_comp_df = stk_data_dict[indu_code].loc[week]
        # 判断行业是否有龙头
        if indu_comp_df['prePct'].max() > 0.11:
            temp_df = indu_comp_df[(indu_comp_df['prePct'] < 0.04)]
            temp_df = temp_df.sort_values('cssd').iloc[:5]
            temp_chs.append(temp_df)
    if len(temp_chs) != 0:
        temp_df = pd.concat(temp_chs)
        ret_cssd_dict[week] = temp_df['chgPct'].mean()
        chs_cssd_dict[week] = temp_df
    else:
        ret_cssd_dict[week] = 0

(pd.Series(ret_cssd_dict).fillna(0) + 1).cumprod().plot(figsize=(16, 8), label='cssd')
plt.title('cssd return')
plt.legend()
plt.show()
# %% 混合策略
mix1_df = pd.concat([pd.Series(ret_beta_dict).fillna(0),
                     pd.Series(ret_csad_dict).fillna(0),
                     pd.Series(ret_cssd_dict).fillna(0)], axis=1)
(mix1_df.apply(lambda x: x[x != 0].mean() if (x == 0).sum() != 3 else 0, axis=1) + 1).cumprod().plot(figsize=(16, 8), label = 'mix1')

mix_ret_dict = {}
for week in week_list:
    temp_dfs = []
    if week in chs_beta_dict.keys():
        temp_dfs.append(chs_beta_dict[week])
    if week in chs_csad_dict.keys():
        temp_dfs.append(chs_csad_dict[week])
    if week in chs_cssd_dict.keys():
        temp_dfs.append(chs_cssd_dict[week])
    if len(temp_dfs) != 0:
        temp_df = pd.concat(temp_dfs)
        mix_ret_dict[week] = temp_df['chgPct'].mean()
    else:
        mix_ret_dict[week] = 0

(pd.Series(mix_ret_dict).fillna(0) + 1).cumprod().plot(label = 'mix2')

mix1_df = pd.concat([pd.Series(ret_beta_dict).fillna(0),
                     pd.Series(ret_cssd_dict).fillna(0)], axis=1)
(mix1_df.apply(lambda x: x[x != 0].mean() if (x == 0).sum() != 2 else 0, axis=1) + 1).cumprod().plot(figsize=(16, 8), label = 'beta_cssd')


plt.legend()
plt.show()