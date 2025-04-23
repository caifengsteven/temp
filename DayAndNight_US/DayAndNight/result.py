import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
res_df = pd.read_csv('beta_dfs.csv', index_col=0)

# %%
ret_df = pd.pivot_table(res_df,
                        index=['ticker', 'month'],
                        values=['ticker_night', 'ticker_day', 'betas'],
                        aggfunc={'ticker_night': lambda x: np.prod(x + 1) - 1,
                                 'ticker_day': lambda x: np.prod(x + 1) - 1,
                                 'betas': np.mean})

# %%
n = 5
ret_df['quantile'] = ret_df.groupby(level=1)['betas'].apply(lambda x:
                                                            pd.qcut(x, n, range(n)))

# %% 根据因子分组观察
ret_day_dict = {}
ret_night_dict = {}
for i in range(n):
    sub_df = ret_df[ret_df['quantile'] == i]
    sub_ret = sub_df.groupby(level = 1)['ticker_day', 'ticker_night'].mean()
    sub_ret = (sub_ret + 1).cumprod()
    ret_day_dict[i] = sub_ret['ticker_day'].values
    ret_night_dict[i] = sub_ret['ticker_night'].values

# %%
pd.DataFrame(ret_day_dict).plot(title = 'day group')
plt.show()
pd.DataFrame(ret_night_dict).plot(title = 'night group')
plt.show()

# %% 晚上做多高beta，白天反转头寸, 万2手续费
# 说明，按月计算beta值，但是，操作是每日操作。晚上做多高beta的股票，白天反转头寸。
ret_df = pd.pivot_table(res_df,
                        index=['ticker', 'month'],
                        values=['ticker_night', 'ticker_day', 'betas'],
                        aggfunc={'ticker_night': lambda x: np.prod(x + 1 - 0.0002) - 1, # 晚上做多
                                 'ticker_day': lambda x: np.prod(1 - x - 0.0002) - 1, # 白天做空
                                 'betas': np.mean})

n = 10
ret_df['quantile'] = ret_df.groupby(level=1)['betas'].apply(lambda x:
                                                            pd.qcut(x, n, range(n)))
high_beta_df = ret_df[ret_df['quantile'] == (n - 1)]
sub_ret = high_beta_df.groupby(level = 1)['ticker_day', 'ticker_night'].mean()
sub_ret = (sub_ret['ticker_night'] + sub_ret['ticker_day'] + 1).cumprod()

sub_ret.plot(title = 'stratege return')
plt.show()