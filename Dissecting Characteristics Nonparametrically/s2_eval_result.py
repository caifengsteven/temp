import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

# %%
result_list = glob('./result/*.csv')
result_list.sort()

# %%
df = pd.read_csv(result_list[0])

# %%
n = 5
s_list = []
for path in result_list:
    df = pd.read_csv(path)
    s = df.groupby(pd.qcut(df['pred'], n, range(1, n + 1)))['real_return'].mean()
    s.name = path[-11:-4]
    s_list.append(s)
res_df = pd.concat(s_list, axis=1).T

# %%
(res_df + 1).cumprod().plot()
plt.show()

# %%
n = 20
s_list = []
for path in result_list:
    df = pd.read_csv(path)
    s = df.groupby(pd.qcut(df['pred'], n, range(1, n + 1)))['real_return'].mean()
    s.name = path[-11:-4]
    s_list.append(s)
res_df = pd.concat(s_list, axis=1).T

# %%
(res_df[n] - res_df[1] + 1).cumprod().plot()
plt.show()
