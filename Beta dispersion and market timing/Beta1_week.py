import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import json

# %%
stk_list = glob('./new_stk_data/*/*.*.CSV')
use_cols = ['代码', '简称', '日期', '开盘价(元)', '最高价(元)',
            '最低价(元)', '收盘价(元)', '成交量(股)',
            '换手率(%)', 'A股流通市值(元)', '总市值(元)', '市盈率']
cols_name = ['ts_code', 'short_name', 'trade_date', 'open', 'high',
             'low', 'close', 'vol',
             'turnover', 'flowvol', 'value', 'pb']


# %%
def read_file(path):
    df = pd.read_csv(path, encoding='gbk', usecols=use_cols, na_values='--')
    df.columns = cols_name
    df.dropna(inplace=True)
    trade_date = pd.to_datetime(df['trade_date'])
    flag = True
    if df.shape[0] < 200:
        flag = False

    def split_week(x):
        y, m, w = x.year, x.month, x.week
        if m == 1 and w > 50:
            y -= 1
        if w < 10:
            w = '0' + str(w)
        else:
            w = str(w)
        return str(y) + w

    def split_date(x):
        y, m, d = x.split('-')
        return y + m + d

    if flag:
        df['week'] = trade_date.apply(split_week)
        df['trade_date'] = df['trade_date'].apply(split_date)
        df['pre_close'] = df['close'].shift(1)
        df['chg_pct'] = df['close'] / df['pre_close'] - 1
        df.sort_values('trade_date', inplace=True)
        return df.iloc[60:], flag
    else:
        return None, flag


def read_index():
    df = pd.read_csv('./index_data/hs_500.csv', index_col=0)

    def split_date(x):
        y, m, d = x.split('-')
        return y + m + d

    df['trade_date'] = df['tradeDate'].apply(split_date)
    df.index = df.pop('trade_date')
    return df


def get_beta(path, week_step=24):
    df, flag = read_file(path)
    df.index = df.pop('trade_date')
    ts_code = df['ts_code'].unique()[0]
    ret_data, index_month = [], []
    # get beta
    month_list = df[(df['week'] > '2005') & (df['week'] < '2021')]['week'].unique().tolist()
    month_list.sort()
    for ind, month in enumerate(month_list[:-week_step]):
        pre_month = month_list[ind: ind + week_step]
        now_month = month_list[ind + week_step]
        index_month.append(now_month)
        bool_ind = df['week'].apply(lambda x: True if x in pre_month else False)
        sub_df = df[bool_ind]
        sub_index = df_index.loc[sub_df.index]['CHGPct']
        beta, alpha = np.polyfit(sub_index, sub_df['chg_pct'], deg=1)
        ret = (df[df['week'] == now_month]['chg_pct'] + 1).prod() - 1
        val = df[df['week'] == now_month]['value'].mean()
        ret_data.append([beta, ret, val])
    ret_df = pd.DataFrame(ret_data, index=index_month, columns=['beta', 'chg_pct', 'value'])
    ret_df['ts_code'] = ts_code
    return ret_df, ts_code


# %%
# df, flag = read_file(stk_list[1])

# %%
df_index = read_index()
with open('./component/500.json', 'r') as f:
    com_500 = json.load(f)
vs = []
for v in com_500.values(): vs.extend(v)
vs = set(vs)
stk_500 = []
for v in vs:
    v = v.split('.')[0]
    for k in stk_list:
        if v in k:
            stk_500.append(k)

# %% 时间太长，使用多线程方法
import multiprocessing

if __name__ == '__main__':
    num_core = 8
    file_step = 16
    n = int(len(stk_500) / 16 + 1)

    for i in tqdm(range(n)):
        with multiprocessing.Pool(num_core) as pool:
            s = pool.map(get_beta, stk_500[i * file_step: (i + 1) * file_step])
            for (ret_df, ts_code) in s:
                if not os.path.exists('./ind500_week/'): os.mkdir('./ind500_week')
                ret_df.to_csv('./ind500_week/%s.csv' % ts_code.split('.')[0])
