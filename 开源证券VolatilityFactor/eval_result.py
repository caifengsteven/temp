import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eval_units import *

# %%
is_300 = True # 是否使用沪深300
is_service = False # 是否加入手续费


if is_300:
    month_dir = './data/300_val_month_quan.csv'
    day_dir = './data/300_val_quan.csv'
else:
    month_dir = './data/val_month_quan.csv'
    day_dir = './data/val_quan.csv'

if is_service:
    s_rate = 0.0016
else:
    s_rate = 0

# %%
res_df = pd.read_csv(month_dir, index_col=[0,1])
v_high_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_high']
v_high_ser.index = v_high_ser.index.get_level_values(0)
v_low_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_low']
v_low_ser.index = v_low_ser.index.get_level_values(0)
v_sp_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_sp']
v_sp_ser.index = v_sp_ser.index.get_level_values(0)

# %% 修改查看
sp = get_shape_ratio(v_sp_ser['0'] - v_sp_ser['4']- s_rate, 'month')
md = get_max_drawdown((v_sp_ser['0'] - v_sp_ser['4'] + 1- s_rate).cumprod())
print('month result \n sharpe ratio : %.4f, max draw : %.4f'%(sp, md))

# %%
(v_high_ser['0'] - v_high_ser['4'] + 1 - s_rate).cumprod().plot(figsize = (16, 8), label = 'high')
(v_low_ser['0'] - v_low_ser['4'] + 1- s_rate).cumprod().plot(figsize = (16, 8), label = 'low')
(v_sp_ser['0'] - v_sp_ser['4'] + 1- s_rate).cumprod().plot(figsize = (16, 8), label = 'sp')
plt.title('month result')
plt.legend()
plt.show()

# %%
res_df = pd.read_csv(day_dir, index_col=[0,1])
v_high_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_high']
v_high_ser.index = v_high_ser.index.get_level_values(0)
v_low_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_low']
v_low_ser.index = v_low_ser.index.get_level_values(0)
v_sp_ser = res_df.loc[res_df.index.get_level_values(1) == 'v_sp']
v_sp_ser.index = v_sp_ser.index.get_level_values(0)

# %% 修改查看
sp = get_shape_ratio(v_sp_ser['0'] - v_sp_ser['4']- s_rate)
md = get_max_drawdown((v_sp_ser['0'] - v_sp_ser['4'] + 1 - s_rate).cumprod())
print('daily result \n sharpe ratio : %.4f, max draw : %.4f'%(sp, md))

# %%
(v_high_ser['0'] - v_high_ser['4'] + 1- s_rate).cumprod().plot(figsize = (16, 8), label = 'high')
(v_low_ser['0'] - v_low_ser['4'] + 1- s_rate).cumprod().plot(figsize = (16, 8), label = 'low')
(v_sp_ser['0'] - v_sp_ser['4'] + 1- s_rate).cumprod().plot(figsize = (16, 8), label = 'sp')
plt.title('daily result')
plt.legend()
plt.show()
