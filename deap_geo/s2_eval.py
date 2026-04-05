import numpy as np
import pandas as pd
import pickle
from s1_deap_result import toolbox
from deap import tools
import matplotlib.pyplot as plt
# %% 读取数据
data = pd.read_csv('./data/hs_300.csv', index_col=0).dropna()
df_x = data[['closeIndex', 'openIndex', 'turnoverVol', 'lowestIndex', 'highestIndex', 'CHGPct']]
df_y = data['CHGPct'].shift(-1)
df_x.index, df_y.index = data['tradeDate'], data['tradeDate']
df_train_x = df_x.loc['2014': '2020', :]
df_train_y = df_y.loc[df_train_x.index]
df_test_x = df_x.loc['2014':, :]
df_test_y = df_y.loc['2014':]
pop = pickle.load(open('model_pop.pkl', 'rb'))

# %%
bests = tools.selBest(pop, k=100)
for i in range(30):
    func_p = toolbox.compile(bests[i])
    pred_ser = func_p(*[df_test_x[col] for col in df_test_x.columns])
    signal_up = (pred_ser > pred_ser.rolling(60).quantile(0.8)).astype(int)
    signal_down = (pred_ser < pred_ser.rolling(60).quantile(0.2)).astype(int)
    signal = signal_down - signal_up
    (df_test_y * signal + 1).cumprod().plot()
(df_test_y + 1).cumprod().plot()
plt.show()

# %%
print(bests[0])

# %%
bests = tools.selBest(pop, k=100)
for i in range(30):
    func_p = toolbox.compile(bests[i])
    pred_ser = func_p(*[df_test_x[col] for col in df_test_x.columns])
    signal_up = (pred_ser > pred_ser.rolling(60).quantile(0.8)).astype(int)
    signal_down = (pred_ser < pred_ser.rolling(60).quantile(0.2)).astype(int)
    signal = signal_down - signal_up
    ((df_test_y * signal).loc['2020':] + 1).cumprod().plot()
(df_test_y.loc['2020':] + 1).cumprod().plot()
plt.show()
