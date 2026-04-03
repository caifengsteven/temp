import numpy as np
import pandas as pd
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


def calculate_unicorn_edge(prices_df, returns_df,n=63):
    """
    Unicorn Edge因子：
    1. 价值信号：逆价格（1/股价）横截面百分位排名（0-1）
    2. 反转信号：过去10日收益求和→取反→横截面Z标准化
    3. 基础信号BASE = 0.7*价值信号 + 0.3*反转信号
    4. 漂移制度REGIME：过去63天正收益日占比>60%则为1，否则为0
    5. 最终因子EDGE = BASE * REGIME
    """
    # -------- 1 计算价值信号 --------
    inv_price = 1 / prices_df  # 逆价格（纯价格维度，非市值）
    value_signal = inv_price.rank(axis=1, pct=True)  # 每日横截面百分位排名（0-1）

    # -------- 2 计算反转信号 --------
    # 过去10日收益滚动求和（满10天才计算）
    roll10_returns = returns_df.rolling(window=10, min_periods=10).sum()
    reversed_returns = -roll10_returns  # 收益取反（做空赢家，做多输家）
    # 横截面Z标准化（每日对所有股票做Z-score）
    reversal_signal = reversed_returns.apply(zscore, axis=1, nan_policy='omit')

    # -------- 3 计算基础信号BASE --------
    base_signal = 0.7 * value_signal + 0.3 * reversal_signal
    base_signal = base_signal.fillna(0)  # 填充NaN为0

    # -------- 4 计算漂移制度REGIME --------
    # 标记每日收益是否为正
    positive_returns = (returns_df > 0).astype(int)
    # 过去n天正收益日数求和（满63天才计算）
    rolln_pos_count = positive_returns.rolling(window=n, min_periods=n).sum()
    up_fraction = rolln_pos_count / n  # 正收益日占比
    # 制度判定：>60%为1，否则为0
    regime_signal = (up_fraction > 0.6).astype(int)
    regime_signal = regime_signal.fillna(0)  # 前63天填充为0

    # -------- 5 计算最终EDGE因子 --------
    edge_factor = base_signal * regime_signal

    return edge_factor