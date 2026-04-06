# -*- coding: utf-8 -*-
"""
市场压力概率指数
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 真实数据获取 (Yahoo Finance)
# ==========================================
def fetch_real_data(start_date="2005-01-01", end_date="2024-12-31"):
    """
    从 Yahoo Finance 获取市场指数和一组个股数据
    """
    # 定义市场基准 (标普 500 指数)
    market_ticker = "^GSPC"
    # 定义一组代表性的个股 (用于计算横截面信号)
    # 我们选择一组覆盖不同行业的流动性高的股票
    stock_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH",
        "MA", "PG", "HD", "DIS", "BAC", "VZ", "ADBE", "CMCSA", "NFLX", "PFE",
        "WMT", "KO", "PEP", "XOM", "CVX", "COST", "ORCL", "CSCO", "INTC", "IBM"
    ]

    print(f"正在从 Yahoo Finance 下载数据 (从 {start_date} 到 {end_date})...")

    # 获取市场收益率
    mkt_data_raw = yf.download(market_ticker, start=start_date, end=end_date)
    print(f"\n--- 市场指数 ({market_ticker}) 数据列 ---")
    print(mkt_data_raw.columns)
    print("------------------------------------------")

    # 对于市场指数，yf.download 返回的是 'Close' 而不是 'Adj Close'
    #if 'Adj Close' not in mkt_data_raw.columns:
    #    raise KeyError(f"'Adj Close' 列在市场指数 ({market_ticker}) 数据中未找到。可用列：{mkt_data_raw.columns.tolist()}")

    # The previous output showed mkt_data_raw had MultiIndex columns like ('Close', '^GSPC').
    # We need to select the specific Series for '^GSPC' under 'Close'.
    mkt_data = mkt_data_raw.loc[:, ('Close', market_ticker)]
    mkt_ret_daily = mkt_data.pct_change().dropna()

    # 获取个股收益率和成交量
    stocks_data = yf.download(stock_tickers, start=start_date, end=end_date)

    # Debug: 打印 stocks_data 的列信息
    print("\n--- 个股数据列 ---")
    print(stocks_data.columns)
    print("-------------------")

    # 提取收益率 (Pct Change) 和 成交量
    stock_rets = stocks_data['Close'].pct_change() # Changed 'Adj Close' to 'Close'
    stock_vols = stocks_data['Volume']

    # 转换为长表格式，以便后续按日期进行横截面计算
    daily_list = []
    for date in stock_rets.index:
        day_rets = stock_rets.loc[date].dropna()
        day_vols = stock_vols.loc[date].loc[day_rets.index] # 确保对齐

        if len(day_rets) < 10: # 如果有效数据太少则跳过该日
            continue

        df_day = pd.DataFrame({
            'date': date,
            'ret': day_rets.values,
            'volume': day_vols.values
        })
        daily_list.append(df_day)

    all_daily_data = pd.concat(daily_list)

    return all_daily_data, mkt_ret_daily

# ==========================================
# 2. 特征工程 (横截面脆弱性信号)
# ==========================================
def compute_fragility_signals(daily_df):
    """
    将每日个股数据转换为月度横截面信号 (Table 1)
    """
    print("计算横截面信号 (离散度, 偏度, 峰度, 尾部比例)...")

    # 每日横截面统计
    def get_daily_stats(x):
        return pd.Series({
            'sigma_xs': x['ret'].std(),
            'skew_xs': x['ret'].skew(),
            'kurt_xs': x['ret'].kurt(),
            'frac_dn': (x['ret'] <= -0.03).mean(), # 考虑到真实数据个股波动，设为-3%
            'frac_up': (x['ret'] >= 0.03).mean(),
            'avg_log_vol': np.log1p(x['volume']).mean()
        })

    daily_stats = daily_df.groupby('date').apply(get_daily_stats)

    # 聚合为月度数据 (取每日平均值)
    # 使用 'ME' (Month End) 频率
    monthly_features = daily_stats.resample('ME').mean()
    return monthly_features

# ==========================================
# 3. 标签定义 (S_t)
# ==========================================
def define_stress_labels(mkt_rets_daily):
    """
    定义压力状态 (公式 8): 收益率极低 OR 波动率极高
    """
    # 月度收益率
    mkt_monthly_ret = mkt_rets_daily.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    # 月度实现波动率 (标准差 * sqrt(21天交易日))
    mkt_monthly_vol = mkt_rets_daily.resample('ME').std() * np.sqrt(21)

    # 扩展窗口分位数 (避免前瞻偏差)
    vol_threshold = mkt_monthly_vol.expanding().quantile(0.90).shift(1)

    # Ensure all components are Series and handle potential NaNs before boolean operations
    condition_ret = (mkt_monthly_ret <= -0.05)
    condition_vol = (mkt_monthly_vol >= vol_threshold)

    # Fill NaN values in boolean Series with False to prevent ambiguity errors if they arise
    # For example, if vol_threshold is NaN, the comparison mkt_monthly_vol >= NaN results in NaN,
    # which can then cause ambiguity in later boolean ops if not handled.
    condition_ret = condition_ret.fillna(False)
    condition_vol = condition_vol.fillna(False)

    # 压力定义: 收益率 <= -0.05 或 波动率 > 90分位数
    stress_label = (condition_ret | condition_vol).astype(int)

    return stress_label, mkt_monthly_ret, mkt_monthly_vol

# ==========================================
# 4. 主程序：MSPI 构建与回测
# ==========================================
def run_mspi_pipeline():
    # 1. 获取数据
    daily_data, mkt_ret_daily = fetch_real_data()

    # 2. 提取特征
    X = compute_fragility_signals(daily_data)

    # 3. 提取标签
    y_raw, m_ret, m_vol = define_stress_labels(mkt_ret_daily)

    # 确保 X 和 y 对齐
    # 预测目标是下个月的压力状态 Y_{t+1}
    y = y_raw.shift(-1).dropna()
    X = X.loc[X.index.intersection(y.index)]
    y = y.loc[X.index]

    # Clear the name of the target Series to prevent potential ambiguity issues in scikit-learn
    y.name = None

    print(f"DEBUG: Head of aligned X:\n{X.head()}")
    print(f"DEBUG: Head of aligned y:\n{y.head()}")
    print(f"DEBUG: X contains NaNs: {X.isnull().any().any()}")
    print(f"DEBUG: y contains NaNs: {y.isnull().any()}")
    print(f"DEBUG: X contains inf: {np.isinf(X).any().any()}")
    print(f"DEBUG: y contains inf: {np.isinf(y).any()}")

    # 4. 扩展窗口回测
    n_total = len(X)
    train_size = 60  # 初始训练窗口设为 60 个月 (5年) 适合真实数据长度

    mspi_probs = []
    actual_stress = []

    print(f"开始扩展窗口回测 (从第 {train_size} 个月开始，共 {n_total} 个月)...")

    for t in range(train_size, n_total):
        # 划分训练集和测试集 (严格无前瞻)
        X_train = X.iloc[:t]
        y_train = y.iloc[:t]
        X_test = X.iloc[[t]]

        # 处理可能的缺失值
        X_train = X_train.fillna(method='ffill').fillna(0)
        X_test = X_test.fillna(method='ffill').fillna(0)

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Lasso Logit (C=0.1 增加正则化强度以应对真实数据的噪声)
        model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 预测下一期压力概率
        prob = model.predict_proba(X_test_scaled)[0, 1]
        mspi_probs.append(prob)
        actual_stress.append(y.iloc[t])

    # 5. 结果汇总
    results = pd.DataFrame({
        'MSPI': mspi_probs,
        'Actual': actual_stress
    }, index=X.index[train_size:])

    # 计算评估指标
    auc = roc_auc_score(results['Actual'], results['MSPI'])
    brier = brier_score_loss(results['Actual'], results['MSPI'])

    print("\n" + "="*30)
    print("--- 真实数据 MSPI 表现 ---")
    print(f"AUC: {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print("="*30)

    # 6. 可视化
    plt.figure(figsize=(14, 7))
    plt.plot(results.index, results['MSPI'], label='MSPI (预测压力概率)', color='steelblue', lw=2)

    # 标记实际发生的压力月份
    stress_dates = results.index[results['Actual'] == 1]
    plt.scatter(stress_dates, [1.02] * len(stress_dates),
                color='crimson', marker='v', s=40, label='实际压力状态 (S_t+1=1)')

    # 标注重大历史事件 (可选)
    events = {
        '2008-09-01': '雷曼兄弟',
        '2011-08-01': '美债降级',
        '2020-03-01': 'COVID-19',
        '2022-06-01': '高通胀/加息'
    }
    for date_str, label in events.items():
        ts = pd.to_datetime(date_str)
        if ts in results.index:
            plt.axvline(x=ts, color='gray', linestyle='--', alpha=0.5)
            plt.text(ts, 0.9, label, rotation=90, color='gray', fontsize=9)

    plt.title('基于真实数据的市场压力概率指数 (MSPI) 复现', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('概率', fontsize=12)
    plt.ylim(-0.05, 1.1)
    plt.legend(loc='upper left')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 检查是否安装了 yfinance
    try:
        run_mspi_pipeline()
    except Exception as e:
        print(f"程序运行出错: {e}")
        print("请确保已安装 yfinance: pip install yfinance")