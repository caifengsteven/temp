import pandas as pd
import numpy as np
from eval_units import *
from tqdm import tqdm

# %% 定义
if_300 = True

# %%
df = pd.read_csv('./data/FactorData.csv')
df.columns = ['stock', 'date', 'close', 'preClose', 'v_high', 'v_low', 'v_sp']
df.set_index(['stock', 'date'], inplace= True)
# %%
ind_300 = pd.read_csv('./data/300index.csv')
ind_300 = ind_300['consTickerSymbol'].tolist()
if if_300:
    df = df.loc[ind_300]

# %%
df['ret'] = (df['close'] - df['preClose']) / df['preClose']
date_list = df.index.get_level_values(1).unique()
# %% 按日回测

quan = 5
res_dict = {}
for date in tqdm(date_list):
    sub_df = df.loc[df.index.get_level_values(1) == date]
    sub_df = sub_df.dropna()
    v_h = sub_df.groupby(pd.qcut(sub_df['v_high'], quan, range(quan)))['ret'].mean().tolist()
    v_l = sub_df.groupby(pd.qcut(sub_df['v_low'], quan, range(quan)))['ret'].mean().tolist()
    v_s = sub_df.groupby(pd.qcut(sub_df['v_sp'], quan, range(quan)))['ret'].mean().tolist()
    res_dict[(date, 'v_high')] = v_h
    res_dict[(date, 'v_low')] = v_l
    res_dict[(date, 'v_sp')] = v_s


# %% daily
res_df = pd.DataFrame(res_dict).T
v_high_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_high']
v_low_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_low']
v_sp_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_sp']

# %%
sp = get_shape_ratio(v_sp_ser[0] - v_sp_ser[4])
md = get_max_drawdown((v_sp_ser[0] - v_sp_ser[4] + 1).cumprod())
print('sharpe ratio : %.4f, max draw : %.4f, return: %.4f'%(sp, md, (v_sp_ser[0] - v_sp_ser[4] + 1).prod()))
res_df.to_csv('./data/300_val_quan.csv')
# %% 按月回测
month_date = {}
for i in date_list:
    month = i[:7]
    if month not in month_date.keys():
        month_date[month] = i
month_list = list(month_date.values())
month_list.sort()

# %%
month_data = [df.loc[df.index.get_level_values(1) == date] for date in month_list]
for data in month_data: data.index = data.index.get_level_values(0)

# %%
for pre_data, data in zip(month_data[:-1], month_data[1:]):
    pre_data['close'] = data['close']
    pre_data['ret'] = (pre_data['close'] - pre_data['preClose']) / pre_data['preClose']


# %%
quan = 5
res_mon_dict = {}
for i in range(0, len(month_data) - 1):
    sub_df = month_data[i]
    sub_df = sub_df.dropna()
    v_h = sub_df.groupby(pd.qcut(sub_df['v_high'], quan, range(quan)))['ret'].mean().tolist()
    v_l = sub_df.groupby(pd.qcut(sub_df['v_low'], quan, range(quan)))['ret'].mean().tolist()
    v_s = sub_df.groupby(pd.qcut(sub_df['v_sp'], quan, range(quan)))['ret'].mean().tolist()
    res_mon_dict[(month_list[i + 1], 'v_high')] = v_h
    res_mon_dict[(month_list[i + 1], 'v_low')] = v_l
    res_mon_dict[(month_list[i + 1], 'v_sp')] = v_s
# %%
res_df = pd.DataFrame(res_mon_dict).T
res_df.to_csv('./data/300_val_month_quan.csv')
v_high_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_high']
v_low_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_low']
v_sp_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_sp']

# %%
sp = get_shape_ratio(v_sp_ser[0] - v_sp_ser[4], 'month')
md = get_max_drawdown((v_sp_ser[0] - v_sp_ser[4] + 1).cumprod())
print('sharpe ratio : %.4f, max draw : %.4f, return : %.4f'%(sp, md, (v_sp_ser[0] - v_sp_ser[4] + 1).prod() - 1))
