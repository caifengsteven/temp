import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool

pool = Pool(12)

# %%
df = pd.read_csv('./data/A_stock.csv', dtype={'ticker': str, 'industryID1': str}, parse_dates=['tradeDate'])


# %%
def split_week(x):
    if x.week < 10:
        p = str(x.year) + '0' + str(x.week)
    else:
        p = str(x.year) + str(x.week)
    return p


def read_indu_index():
    indu_dfs = {}
    df = pd.read_csv('./data/indu_index.csv', parse_dates=['tradeDate'], index_col=0)
    indu_id = df['industryID1'].unique().tolist()
    for i in indu_id:
        # print(i)
        sub_ = df[df['industryID1'] == i].copy()
        sub_.index = sub_.pop('tradeDate')
        sub_ = sub_[sub_['CHGPct'].apply(lambda x: np.abs(x) < 0.10)]
        sub_ = sub_.loc[pd.to_datetime('2010-01-01'):, ]
        indu_dfs[str(i)] = sub_
    return indu_dfs


def trans_stk_data(sub_df):
    sub_df = sub_df[sub_df['tradeDate'] > pd.to_datetime('2010-01-01')].copy()
    if sub_df.shape[0] > 60:
        sub_df = sub_df.sort_values('tradeDate')
        sub_df['chgPct'] = sub_df['closePrice'] / sub_df['preClosePrice'] - 1
        indu = sub_df['industryID1'].iloc[-1] # 取最新的行业作为行业分类
        stk_code = sub_df['ticker'].unique()[0]
        sub_df['week'] = sub_df['tradeDate'].apply(split_week)
        week_df = pd.pivot_table(sub_df, index=['week'],
                                 values=['ticker', 'chgPct', 'preClosePrice', 'closePrice', 'industryID1'],
                                 aggfunc={'ticker': lambda x: x[0],
                                          'chgPct': lambda x: np.prod(1 + x) - 1,
                                          'preClosePrice': lambda x: x.iloc[0],
                                          'closePrice': lambda x: x.iloc[-1],
                                          'industryID1': lambda x: indu})
        return week_df, indu, stk_code
    else:
        return None, False, False


def read_index():
    df = pd.read_csv('./data/sh_index.csv',
                     usecols=['tradeDate', 'CHGPct', 'preCloseIndex', 'openIndex', 'lowestIndex',
                              'highestIndex', 'closeIndex'],
                     index_col='tradeDate',
                     parse_dates=['tradeDate'])
    return df


# %% 分割数据
ticker_list = df['ticker'].unique().tolist()
df.index = df['ticker']
print('reading stk data ...')

for ticker in tqdm(ticker_list):
    sub_df = df.loc[ticker]
    week_df, indu, stk_code = trans_stk_data(sub_df)
    if indu:
        if not os.path.exists('./data/%s' % indu):
            os.makedirs('./data/%s' % indu)
        week_df.to_csv('./data/%s/%s.csv' % (indu, ticker))


# %% 行业指数数据转换成周数据
print('transing indu data')
indu_dfs = read_indu_index()
for indu_code in tqdm(indu_dfs.keys()):
    indu_df = indu_dfs[indu_code]
    indu_df['week'] = indu_df.index.map(split_week)
    indu_week = pd.pivot_table(indu_df, index=['week'],
                               values=['ticker', 'CHGPct', 'preCloseIndex', 'closeIndex', 'industryID1'],
                               aggfunc={'ticker': lambda x: x[0],
                                        'CHGPct': lambda x: np.prod(1 + x) - 1,
                                        'preCloseIndex': lambda x: x.iloc[0],
                                        'closeIndex': lambda x: x.iloc[-1],
                                        'industryID1': lambda x: x.iloc[0]})
    indu_week.to_csv('./data/%s.csv' % indu_code)

# %% 上证指数转换
sh_df = read_index()
sh_df['week'] = sh_df.index.map(split_week)
sh_week = pd.pivot_table(sh_df, index=['week'],
                         values=['CHGPct', 'preCloseIndex', 'closeIndex'],
                         aggfunc={'CHGPct': lambda x: np.prod(1 + x) - 1,
                                  'preCloseIndex': lambda x: x.iloc[0],
                                  'closeIndex': lambda x: x.iloc[-1]})
indu_week.to_csv('./data/sh_week.csv')
