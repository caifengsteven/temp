import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
class Regret_Minimize:

    def __init__(self, K=0.8):
        self.alpha_long = 0.5
        self.alpha_short = 0.5
        self.R_long = 0
        self.R_short = 0
        self.K = K  # 遗忘系数

    def update_alpha(self, p1, p2):
        # 计算 决策点虚拟夹着
        v_long = p2 - p1
        v_short = p1 - p2
        vht = self.alpha_long * (p2 - p1) - self.alpha_short * (p2 - p1)

        if v_long > 0:
            self.r_long = v_long - vht
            if self.R_long == 0:
                self.R_long = self.r_long  # 初始赋值
            else:
                self.R_long = self.R_long * self.K + self.r_long

        if v_short > 0:
            self.r_short = v_short - vht
            if self.R_short == 0:
                self.R_short = self.r_short  # 初始赋值
            else:
                self.R_short = self.R_short * self.K + self.r_short

        self.alpha_long = self.R_long / (self.R_long + self.R_short)
        self.alpha_short = self.R_short / (self.R_short + self.R_long)

    def fit(self, pt, pre_pt):
        n = len(pt)
        self.alpha_longs = np.zeros(n)
        self.alpha_shorts = np.zeros(n)
        for i in range(n):
            # print(self.alpha_long, self.alpha_short)
            p1, p2 = pre_pt[i], pt[i]
            self.update_alpha(p1, p2)
            self.alpha_longs[i] = self.alpha_long
            self.alpha_shorts[i] = self.alpha_short


# %%
if __name__ == '__main__':

    df = pd.read_csv('./data/sh_index.csv', index_col=0, parse_dates=['tradeDate'])
    df.index = df.pop('tradeDate')
    df = df[df.index > '2008-01-01']
    # %%
    K = 0.8

    pre_pt = df['closeIndex'].iloc[:-1].values
    pt = df['closeIndex'].iloc[1:].values

    model = Regret_Minimize(K)
    model.fit(pt, pre_pt)

    res_df = df.iloc[1:].loc[:, ['closeIndex', 'CHGPct']]
    res_df['alpha_long'] = model.alpha_longs
    res_df['alpha_short'] = model.alpha_shorts
    res_df['alpha_long'] = res_df['alpha_long'].shift(1)
    res_df['alpha_short'] = res_df['alpha_short'].shift(1)


    (res_df['CHGPct'] + 1).cumprod().plot(figsize= (16, 8))
    (res_df['CHGPct'] * (res_df['alpha_long'] > 0.55).astype(int) + 1).cumprod().plot(figsize= (16, 8))
    plt.title(u'k = %s long return'%K)
    plt.legend()
    plt.show()

    (res_df['CHGPct'] + 1).cumprod().plot(figsize= (16, 8))
    (res_df['CHGPct'] * (res_df['alpha_long'] > 0.55).astype(int)
     - res_df['CHGPct'] * (res_df['alpha_short'] > 0.55).astype(int)
     + 1).cumprod().plot(figsize= (16, 8))
    plt.title(u'k = %s long-short return'%K)
    plt.legend()
    plt.show()
