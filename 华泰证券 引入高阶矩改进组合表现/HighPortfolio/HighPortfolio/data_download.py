import pandas as pd
import yfinance as yf
from algs import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
ss = yf.Ticker('000001.ss')
ss_df = ss.history(period= '1mo', start='2000-01-01', end = '2020-01-01')

sp = yf.Ticker('^GSPC')
sp_df = sp.history(period= '1mo', start = '2000-01-01', end = '2020-01-01')

ni = yf.Ticker('^N225')
ni_df = ni.history(period= '1mo', start = '2000-01-01', end = '2020-01-01')

sx = yf.Ticker('^STOXX50E')
sx_df = sx.history(period= '1mo', start = '2000-01-01', end = '2020-01-01')

# %%
def get_return_df(dfs, names):
    df_list = [df['Close'].diff() / df['Close'].shift(1) for df in dfs]
    df_ = pd.concat(df_list, axis= 1)
    df_.columns = names
    return df_.dropna()

df = get_return_df([ss_df, sp_df, ni_df], names= ['ss', 'sp', 'ni'])

# %%
test_window = 24
step_window = 1
n = df.shape[0]


def get_result(lam_1=1, lam_2=1):
    tests = []
    for i in tqdm(range(0, n - test_window - step_window, step_window)):
        df_sub = df.iloc[i: i + test_window + step_window, :]
        X_train, df_test = df_sub.iloc[:test_window].copy(), df_sub.iloc[test_window:].copy()
        X_train = X_train.values
        X_test = df_test.values

        # get base optimal
        model = PGP_skew(lam_1=lam_1, lam_2=lam_2, lam_3=0, n_asset=3)
        w, opt = model.fit(X_train)
        df_test['opt_base'] = (X_test * w).sum(axis=1)

        # get 3order optimal
        ws = []
        for lam_3 in [1, 3, 5, 7, 9]:
            model = PGP_skew(lam_1=lam_1, lam_2=lam_2, lam_3=lam_3, n_asset=3)
            w, opt = model.fit(X_train)
            ws.append(w)
        w = np.mean(ws, axis=0)
        df_test['opt_skn'] = (X_test * w).sum(axis=1)
        tests.append(df_test)
    ts_df = pd.concat(tests)
    return ts_df


# %%
ts_11 = get_result(1, 1)
ts_15 = get_result(1, 5)
ts_51 = get_result(5, 1)
# %%
(ts_51 + 1).cumprod().plot(figsize=(16, 8))
plt.show()

# %%
for ts, ty in zip([ts_11, ts_15, ts_51], ['稳健', '激进', '保守']):
    base = get_shape_ratio(ts['opt_base'])
    opt = get_shape_ratio(ts['opt_skn'])
    print('type:%s = base sharpe: %.4f, opt sharpe: %.4f, up_ratio: %.4f' % (ty, base, opt, (opt / base - 1) * 100))
