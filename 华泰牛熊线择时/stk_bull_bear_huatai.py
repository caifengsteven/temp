
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:

ind_code = '000001' # 此处修改股票代码
fields = ['tradeDate', 'closePrice', 'chgPct', 'turnoverRate','ticker']


# In[ ]:

df = DataAPI.MktEqudGet(secID=u"",ticker=ind_code,tradeDate=u"",beginDate=u"2000-01-01", endDate=u"",isOpen="1",field=fields ,pandas="1")


# In[ ]:

def get_bear_index(df_50, bear_window = 250, long_window = 120, short_window = 30):
    df = df_50.copy()
    df['bear_index'] = df['turnoverRate'].rolling(bear_window).mean() / df['chgPct'].rolling(bear_window).std()
    df['bear_long'] = df['bear_index'].rolling(long_window).mean()
    df['bear_short'] = df['bear_index'].rolling(short_window).mean()
    return df.dropna()
    


# In[ ]:

df_index = get_bear_index(df)


# In[ ]:

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(df_index['bear_short'].values,'g-')
ax1.plot(df_index['bear_long'].values, 'r-')
ax2.plot(df_index['closePrice'].values,'b-')


# 按照研报的参数，并不能复现研报的结果。应该是研报对指数换手率的计算有特殊方法。我们按照原始定义，对换手率计算。

# In[ ]:

df_index = get_bear_index(df,bear_window = 250,  long_window= 120, short_window= 30)
sig = (df_index['bear_short'] > df_index['bear_long']).astype(int).shift(1) # 信号滞后
(df_index['chgPct'] * sig + 1).cumprod().plot(figsize=(16, 8))
(df_index['chgPct'] + 1).cumprod().plot()


# In[ ]:

res_dict = {}
for bear_window in range(50, 260, 10):
    for long_window in range(120, 260, 10):
        for short_window in range(20, 65, 5):
            df_index = get_bear_index(df, bear_window= bear_window ,long_window= long_window, short_window= short_window)
            sig = (df_index['bear_short'] > df_index['bear_long']).astype(int).shift(1) # 信号滞后
            ret = (df_index['chgPct'] * sig + 1).prod()
            res_dict[(bear_window ,long_window, short_window)] = ret


# In[ ]:

res_df = pd.Series(res_dict)


# 使用最优参数搜寻得到的结果

# In[ ]:

bear_window, long_window, short_window = pd.Series(res_dict).argmax()
df_index = get_bear_index(df, bear_window = bear_window,  long_window= long_window, short_window= short_window)
sig = (df_index['bear_short'] > df_index['bear_long']).astype(int).shift(1) # 信号滞后
(df_index['chgPct'] * sig + 1).cumprod().plot(figsize=(16, 8))
(df_index['chgPct'] + 1).cumprod().plot()


# In[ ]:

print(bear_window, long_window, short_window)

