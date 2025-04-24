import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os
from glob import glob
from cvxpy_glasso import *
import time
import statsmodels.api as sm
from tqdm import tqdm

# %%
data_list = glob('L:\Dropbox\Dropbox\project folder from my asua computer\Project\Adaboost\month_data\*.csv')
data_list.sort()


# %%
def get_train_test_path(begin_date='2010-01', window=12):
    for begin_path in data_list:
        if begin_date in begin_path: break
    beg_ind = data_list.index(begin_path)
    train_ind = data_list[beg_ind: beg_ind + window]
    test_ind = data_list[beg_ind + window]
    return train_ind, test_ind


def processing_data_order(s):
    sc = s.copy()
    s_order = np.arange(s.shape[0])[s.argsort()]
    sc.iloc[s_order] = np.arange(s.shape[0])
    return sc / sc.max()


def order_data(s):
    sc = s.copy()
    n = len(s)
    order = np.arange(n)[np.argsort(s)]
    sc[order] = np.arange(n)
    return sc / n


def group_Xdata(v, knot=10):
    c = np.ones_like(v)
    v_2 = v ** 2
    quantile = np.linspace(0, 1, knot + 1)
    p_ks = [c, v, v_2]
    for q in quantile[1:-1]:
        p_i = v_2.copy()
        p_i[v < q] = 0
        p_ks.append(p_i)
    return np.vstack(p_ks).T


def knot_group_Xdata(s, knot=10):
    v, name = s.values, s.name
    c = np.ones_like(v)
    v_2 = v ** 2
    quantile = np.linspace(0, 1, knot + 1)
    p_ks = [c, v, v_2]
    for q in quantile[1:-1]:
        p_i = v_2.copy()
        p_i[v < q] = 0
        p_ks.append(p_i)
    return np.vstack(p_ks).T, name


def processing_data(df, knot=10, is_train=True):
    order_cols = df.columns[3: -3]
    df_order = df[order_cols]
    df_x, df_y = df_order.iloc[:, :-1], df_order.iloc[:, -1]
    vx, names = df_x.values, df_x.columns
    v_order = np.apply_along_axis(lambda x: order_data(x), 0, vx)
    v_group = np.apply_along_axis(lambda x: group_Xdata(x, knot=knot), 0, v_order)
    # n = v_group.shape[-1]
    # X_g = [v_group[:, :, i] for i in range(n)]
    if is_train:
        df_y.iloc[np.arange(df_y.shape[0])[df_y.argsort()]] = np.arange(df_y.shape[0]) / df_y.shape[0]
        return (v_group, names), df_y.values
    else:
        return (v_group, names), df.loc[:, ['ticker', 'tradeDate', 'real_return']]


def get_train_test_data(train_paths, test_path):
    train_datas = []
    for train_path in train_paths:
        df_train = pd.read_csv(train_path, index_col=0)
        x_g, y = processing_data(df_train, is_train=True)
        train_datas.append((x_g, y))
    df_test = pd.read_csv(test_path, index_col=0)
    df_test.dropna(inplace=True)
    x_test, df_ret = processing_data(df_test, is_train=False)

    # generate x data
    data_trx = [data[0][0] for data in train_datas]
    data_trx = np.vstack(data_trx)

    data_tex, cols = x_test

    # generate y data
    data_try = [data[1] for data in train_datas]
    data_try = np.hstack(data_try)

    # split group x
    n = data_trx.shape[-1]
    Xtr_g = [data_trx[:, :, i] for i in range(n)]
    Xte_g = [data_tex[:, :, i] for i in range(n)]

    # concat datas
    return (Xtr_g, data_try), (Xte_g, df_ret), cols


def use_ols_filter_data(train_data):
    xs_, y_ = train_data
    chs_index = []
    for col_index, x_, in enumerate(xs_):
        ols_model = sm.OLS(y_, x_)
        result = ols_model.fit()
        fp = result.f_pvalue
        ps = result.pvalues[1:]
        if fp < 0.1 and (ps < 0.1).sum() > 1:
            chs_index.append(col_index)
    return chs_index


def update_data(train_data, test_data, names, chs_index):
    train_n, test_n, names_n = [[], train_data[1]], [[], test_data[1]], []
    for i in chs_index:
        train_n[0].append(train_data[0][i])
        test_n[0].append(test_data[0][i])
        names_n.append(names[i])
    return train_n, test_n, names_n


# %%
window = 12
for date in tqdm(data_list[:-window]):
    np.random.seed(1)
    train_paths, test_path = get_train_test_path(date, window=window)
    test_date = test_path[-11:]
    train_data, test_data, names = get_train_test_data(train_paths, test_path)
    chs_index = use_ols_filter_data(train_data)
    train_data, test_data, names = update_data(train_data, test_data, names, chs_index)
    print(len(chs_index))

    # %%
    print(time.asctime())
    print('optimizing')
    model = group_lasso_model(intercept=False, penalty=10e-3)
    X_g, y = train_data
    model.fit_with_panelty(X_g, y)
    print(time.asctime())

    # %%
    betas = model.sig_beta.copy()
    chs_index = np.arange(len(betas))[np.abs(np.vstack(betas)).sum(axis=1) > 10e-4]
    train_data, test_data, names = update_data(train_data, test_data, names, chs_index)

    # %%
    ws = np.array([np.power(np.sum(np.square(b)), -.5) for b in betas])[chs_index]

    # %%
    X_g, y = train_data
    model.penalty = 10e-5
    model.fit_with_panelty(X_g, y, weights=ws)

    betas = model.sig_beta.copy()
    chs_index = np.arange(len(betas))[np.abs(np.vstack(betas)).sum(axis=1) > 10e-3]
    print(np.array(names)[chs_index])

    test_data[1]['pred'] = model.predict_penalty(test_data[0])
    test_data[1]['chs_factor'] = np.hstack([np.array(names)[chs_index], np.zeros(test_data[1].shape[0] - len(chs_index)
                                                                                 , dtype=object)])
    if not os.path.exists('./result/'): os.mkdir('./result/')
    test_data[1].to_csv('./result/%s' % test_date, index = False)
