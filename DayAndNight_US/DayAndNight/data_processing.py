import pandas as pd
from tqdm import tqdm
import os

# %%
cols = ['uopen', 'uhigh', 'ulow', 'uclose', 'uvol', 'div',
        'split', 'open', 'high', 'low', 'close', 'vol']
df = pd.read_csv('./data/EOD_20200619.csv', header=None, names = cols)

# %%
tickers = df.index.get_level_values(0).unique()


# %%
def is_save(sub_df):
    # 满足上市时间三年以上
    flag = True
    n = sub_df.shape[0]
    if n < 1000:
        flag = False
    # 未退市
    end_date = sub_df.index[-1]
    if end_date < '2020-01-01':
        flag = False
    if sub_df['close'].median() < 20: # 去除低市值
        flag = False
    return flag

# %%
for ticker in tqdm(tickers):
    sub_df = df.loc[ticker]
    if not os.path.exists('./stk_data'): os.makedirs('./stk_data')
    if is_save(sub_df): sub_df.to_csv('./stk_data/stk_%s.csv'%ticker)

# %%
