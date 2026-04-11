# %%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
# %%
print(os.listdir())

# %%
df = pd.read_csv('./data/Astock_dayly.csv', index_col= 0, dtype={'ticker': str})
df['ticker'] = df['ticker'].apply(lambda x: '0'*(6 - len(x)) + x)

# %%
for name in tqdm(df['ticker'].unique()):
    sub_df = df[df['ticker'] == name]
    sub_df.to_csv('./stk_data/%s.csv'%name, index= False)

# %% save date
with open('./data/date_index.npy', 'wb') as f:
    np.save(f, df['tradeDate'].unique())
