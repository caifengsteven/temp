
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import copy
import statsmodels.api as sm
from sklearn import linear_model


# In[ ]:

all_mon_date = []
for year in range(2010, 2020):
    for mon in range(1, 13):
        if mon < 10:
            mon_str = '%s-0%s'%(year, mon)
        else:
            mon_str = '%s-%s'%(year, mon)
        all_mon_date.append(mon_str)


# In[ ]:

def read_mon_fac_data(begin_date = '2010-01' ,window = 30):
    base_dir = './fac_data/'
    begin_mon_ind = all_mon_date.index(begin_date)
    end_mon_ind = begin_mon_ind + window
    train_paths = [base_dir + i + '.csv' for i in all_mon_date[begin_mon_ind : end_mon_ind]]
    test_path = base_dir + all_mon_date[end_mon_ind] + '.csv'
    train_data = [pd.read_csv(path, index_col = 0) for path in train_paths]
    test_data = pd.read_csv(test_path, index_col = 0)
    return train_data, test_data, all_mon_date[end_mon_ind]


# In[ ]:


def filter_icir(train_data, fac_cols, y_col):
    ic_df = pd.concat([data[fac_cols].corrwith(data[y_col]) for data in train_data], axis = 1)
    icir = ic_df.mean(axis = 1) / ic_df.std(axis = 1)
    filter_cols = icir.abs().sort_values( ascending = False).iloc[:70].index
    return filter_cols.tolist()

def filter_corr(train_data, filter_icir_cols):
    fil_df = pd.concat([data[filter_icir_cols] for data in train_data], axis = 0)
    
    fil_corr_cols = copy.deepcopy(filter_icir_cols)
    for col in fil_corr_cols:
        fil_cols = [fcol for fcol in fil_corr_cols if fcol not in col]
        corr_df = fil_df[fil_cols].corrwith(fil_df[col])
        if np.any(corr_df.abs() > 0.75):
            del_cols = corr_df[corr_df.abs() > 0.75].index.tolist()
            for i in del_cols: fil_corr_cols.remove(i)
    return fil_corr_cols

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def filter_stepreg(train_data, filter_corr_cols, verbose = True):
    fil_df = pd.concat([data[filter_corr_cols + ['return']] for data in train_data], axis = 0)
    filter_stepreg_cols = stepwise_selection(fil_df[filter_corr_cols], fil_df[y_col], verbose= verbose)
    return filter_stepreg_cols
    

# alpha * l1 = 0.05  0.5 * alpha * (1 - l1) = 0.3  => alpha = 0.65 l1 = 0.077 研报参数
def main_ela_model(train_data, test_data, filter_stepreg_cols):
    model = linear_model.ElasticNet(l1_ratio=0.077, alpha= 0.65)
    fil_df = pd.concat([data[filter_stepreg_cols + ['return']] for data in train_data], axis = 0)
    result = model.fit(fil_df[filter_stepreg_cols], fil_df[y_col])
    pred = result.predict(test_data[filter_stepreg_cols])
    test_data['pred'] = pred
    return test_data[['real_return', 'return', 'pred']]


# 如果出现内存不足问题，将window = 30修改成更小的window可以解决。例如 window = 20

# In[ ]:

window = 30 # 内存不足将30改成更小的数，例如20
res_dict = {}
for mon_str in all_mon_date[:-30]:
    train_data, test_data, test_date = read_mon_fac_data(begin_date= mon_str, window= window)
    fac_cols, y_col = test_data.columns[3:-5], 'return'
    filter_icir_cols = filter_icir(train_data, fac_cols, y_col) # 利用 iric 进行因子筛选
    filter_corr_cols = filter_corr(train_data, filter_icir_cols) # 利用 相关性 进行因子筛选
    filter_stepreg_cols = filter_stepreg(train_data, filter_corr_cols, verbose = False) # 利用 逐步回归 进行因子筛选
    test_result = main_ela_model(train_data, test_data, filter_stepreg_cols) # 使用 ela算法 选择因子
    res_dict[test_date] = test_result # 保存结果
    test_result.to_csv('./fac_data/res_%s.csv'%test_date)
    print(test_date, 'has done')
    


# In[ ]:

ret_dfs = []
for mon_str in res_dict.keys():
    res_df = res_dict[mon_str]
    ret_ser = res_df.groupby(pd.qcut(res_df['pred'], 5, range(1, 6)))['real_return'].mean()
    ret_ser.name = mon_str
    ret_dfs.append(ret_ser)


# In[ ]:

ret_df = pd.concat(ret_dfs, axis = 1).T
ret_df.sort_index(inplace = True)


# In[ ]:

# 因子分层检验，分层明显，说明因子是有效的
(ret_df + 1).cumprod().plot()


# In[ ]:

# 每月做多 分数最高的15只股票，做空 分数最低的15只股票
n_stocks = 15
rets = {}
for mon_str in res_dict.keys():
    res_df = res_dict[mon_str]
    ret_long = res_df.sort_values('pred', ascending = False).iloc[:n_stocks]['real_return'].mean()
    ret_short = res_df.sort_values('pred', ascending = True).iloc[:n_stocks]['real_return'].mean()
    rets[mon_str] = [ret_long, ret_short]
ret_ser = pd.DataFrame(rets).T


# In[ ]:

(ret_ser[0] - ret_ser[1]  + 1).cumprod().plot()


# In[ ]:



