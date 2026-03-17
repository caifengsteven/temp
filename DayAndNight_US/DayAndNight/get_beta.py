import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import pickle

model = LinearRegression(fit_intercept=True)
# %%
tickers = os.listdir('./stk_data')
tickers = [i for i in tickers if '.csv' in i]
index_df = pd.read_csv('./data/index_sina.csv', index_col=0)
index_df = index_df[index_df['ticker'] == 'SPX']
index_df.index = index_df.pop('tradeDate')


# %%
def get_chgpct_by_df(df, close='close', open='open'):
    df['chgPct'] = df[close] / df[close].shift(1) - 1
    df['chg_day'] = df[close] / df[open] - 1
    df['chg_night'] = (1 + df['chgPct']) / (1 + df['chg_day']) - 1
    return df


def clear_ticker_data(df):
    # 删除 收益超过 200% 的日期
    chg_c2c = df['chgPct'] < 2
    # 删除 隔夜收益超过50% 的日期
    chg_night = df['chg_night'] < 0.5
    # 删除 当天涨幅超过 100% 的日期
    chg_day = df['chg_day'] < 1
    return df[chg_c2c & chg_night & chg_day]


def rolling_year_date(mon_dates, window=12):
    n = len(mon_dates)
    train_dates = []
    test_dates = []
    for i in range(n - window):
        train_date = mon_dates[i: i + window]
        test_date = mon_dates[i + window]
        train_dates.append(train_date)
        test_dates.append(test_date)
    return train_dates, test_dates


def get_night_beta(x, y):
    model.fit(x, y)
    return model.coef_


def get_ticker_beta(df_x, train_dates, test_dates):
    beta_dict = {}
    for train_date, test_date in zip(train_dates, test_dates):
        sub_df = df_x[df_x['month'].apply(lambda x: True if x in train_date else False)]
        X, y = sub_df[['market_night']], sub_df['ticker_night']
        model.fit(X, y)
        beta = model.coef_.item()
        beta_dict[test_date] = beta
    return beta_dict


index_df = get_chgpct_by_df(index_df, 'closeIndex', 'openIndex')

# %%
ticker_betas = []
for ticker in tqdm(tickers):
    ticker_df = pd.read_csv('./stk_data/%s' % ticker, index_col=0)
    ticker_df = get_chgpct_by_df(ticker_df)
    ticker_df = clear_ticker_data(ticker_df)

    df_x = pd.concat([index_df['chg_night'], ticker_df['chg_night'], ticker_df['chg_day']], axis=1).dropna()
    df_x.columns = ['market_night', 'ticker_night', 'ticker_day']
    # 标识月
    df_x['month'] = df_x.index
    df_x['month'] = df_x['month'].apply(lambda x: x[:7])

    # 计算beta
    mon_dates = df_x['month'].unique().tolist()
    train_dates, test_dates = rolling_year_date(mon_dates)
    beta_map = get_ticker_beta(df_x, train_dates, test_dates)
    df_x['betas'] = df_x['month'].map(beta_map)
    df_x.dropna(inplace=True)
    df_x['ticker'] = ticker[:-4]
    # if not os.path.exists('./beta_tickers'): os.mkdir('./beta_tickers')
    # df_x.to_csv('./beta_tickers/%s' % ticker)
    ticker_betas.append(df_x)

# %%
beta_dfs = pd.concat(ticker_betas)
beta_dfs.to_csv('beta_dfs.csv')