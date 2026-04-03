import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']

# ===================== 1. 模拟1秒级高频成交数据 =====================
def simulate_high_freq_data(start_price=100, total_seconds=3600, seed=42):
    np.random.seed(seed)
    time_stamps = pd.date_range(start='2026-01-01 09:30:00', periods=total_seconds, freq='1S')
    price_changes = np.random.normal(loc=0, scale=0.001, size=total_seconds)
    prices = start_price * np.exp(np.cumsum(price_changes))
    volumes = np.random.randint(low=100, high=10000, size=total_seconds)

    df = pd.DataFrame({
        'time': time_stamps,
        'price': prices,
        'volume': volumes
    })
    return df


# ===================== 2. 数据预处理：计算核心高频变量 =====================
def preprocess_data(df):
    # 对数收益率（剔除零收益率）
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df[df['log_return'] != 0].reset_index(drop=True)

    # 对数交易量
    df['log_volume'] = np.log(df['volume'] / df['volume'].shift(1))
    df = df.dropna(subset=['log_volume']).reset_index(drop=True)

    # 交易持续时间
    df['trade_duration'] = 1 + np.random.normal(loc=0, scale=0.1, size=len(df))
    df['trade_duration'] = df['trade_duration'].abs()

    return df


# ===================== 3. 本福特定律：计算高频偏离因子 =====================
def get_first_significant_digit(x):
    if x == 0:
        return np.nan
    abs_x = abs(x)
    str_x = str(abs_x).replace('.', '').lstrip('0')
    return int(str_x[0]) if str_x else np.nan


def benford_theoretical_distribution():
    benford_probs = {}
    for d in range(1, 10):
        benford_probs[d] = math.log10(1 + 1 / d)
    return benford_probs


def calculate_benford_deviation_factor(series, window_size=200):
    benford_probs = benford_theoretical_distribution()
    deviation_factors = []

    for i in range(window_size, len(series)):
        window_data = series.iloc[i - window_size:i]
        first_digits = window_data.apply(get_first_significant_digit)
        first_digits = first_digits.dropna()
        if len(first_digits) < 10:
            deviation_factors.append(np.nan)
            continue

        observed_counts = first_digits.value_counts().reindex(range(1, 10), fill_value=0)
        total = len(first_digits)
        observed_freq = observed_counts / total

        # 卡方检验
        chi2_stat = 0
        for d in range(1, 10):
            expected = total * benford_probs[d]
            observed = observed_counts[d]
            if expected > 0:
                chi2_stat += (observed - expected) ** 2 / expected

        # 标准化+对数转换
        standardized_chi2 = sum([(observed_freq[d] - benford_probs[d]) ** 2 / benford_probs[d] for d in range(1, 10)])
        deviation_factor = np.log(standardized_chi2) if standardized_chi2 > 0 else np.nan
        deviation_factors.append(deviation_factor)

    deviation_factors = [np.nan] * window_size + deviation_factors
    return deviation_factors


# ===================== 4. LSTAR过渡函数 + STARX预测模型 =====================
def lstar_transition_function(x, c, gamma):
    """
    逻辑型过渡函数（LSTAR）
    :param x: 过渡变量（如收益率滞后项、本福特偏离因子）
    :param c: 阈值参数（区制切换临界值）
    :param gamma: 斜率参数（转换平滑度）
    :return: 过渡函数值F∈[0,1]
    """
    return 1 / (1 + np.exp(-gamma * (x - c)))


def starx_model(params, X, transition_var):
    """
    STARX模型核心计算（含LSTAR过渡）
    :param params: 模型参数 [a0_0,a0_1,a0_2,a0_3,a1_0,a1_1,a1_2,a1_3,c,gamma]
                   a0_*: 低区制系数；a1_*: 高区制增量系数；c:阈值；gamma:斜率
    :param X: 解释变量矩阵 [常数项, Rt-1, Vt-1, Dt-1]（可选加偏离因子）
    :param transition_var: 过渡变量
    :return: 预测的收益率Rt
    """
    # 拆分参数
    a0_0, a0_1, a0_2, a0_3 = params[0:4]
    a1_0, a1_1, a1_2, a1_3 = params[4:8]
    c, gamma = params[8], params[9]

    # 低区制回归
    low_regime = a0_0 + a0_1 * X[:, 1] + a0_2 * X[:, 2] + a0_3 * X[:, 3]
    # 高区制增量回归
    high_regime = a1_0 + a1_1 * X[:, 1] + a1_2 * X[:, 2] + a1_3 * X[:, 3]
    # LSTAR过渡函数
    F = lstar_transition_function(transition_var, c, gamma)
    # 总预测值
    y_pred = low_regime + high_regime * F
    return y_pred


def starx_model_with_benford(params, X, transition_var):
    """
    融入本福特偏离因子的STARX模型（Model4）
    :param params: 模型参数 [a0_0-a0_6, a1_0-a1_6, c, gamma]
                   a0_4-a0_6: 低区制偏离因子系数；a1_4-a1_6: 高区制偏离因子增量系数
    :param X: 解释变量矩阵 [常数项, Rt-1, Vt-1, Dt-1, Br, Bv, Bd]
    :param transition_var: 过渡变量
    :return: 预测的收益率Rt
    """
    a0_0, a0_1, a0_2, a0_3, a0_4, a0_5, a0_6 = params[0:7]
    a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6 = params[7:14]
    c, gamma = params[14], params[15]

    low_regime = a0_0 + a0_1 * X[:, 1] + a0_2 * X[:, 2] + a0_3 * X[:, 3] + a0_4 * X[:, 4] + a0_5 * X[:, 5] + a0_6 * X[:,
                                                                                                                    6]
    high_regime = a1_0 + a1_1 * X[:, 1] + a1_2 * X[:, 2] + a1_3 * X[:, 3] + a1_4 * X[:, 4] + a1_5 * X[:, 5] + a1_6 * X[
                                                                                                                     :,
                                                                                                                     6]
    F = lstar_transition_function(transition_var, c, gamma)
    y_pred = low_regime + high_regime * F
    return y_pred


def fit_starx_model(y, X, transition_var, with_benford=False):
    """
    拟合STARX模型（非线性最小二乘法）
    :param y: 目标变量（Rt）
    :param X: 解释变量矩阵
    :param transition_var: 过渡变量
    :param with_benford: 是否融入本福特偏离因子
    :return: 拟合后的参数、预测值、RMSE
    """
    # 初始化参数（经验值，可根据数据调整）
    if not with_benford:
        # 基准模型（Model3）参数：[a0_0,a0_1,a0_2,a0_3,a1_0,a1_1,a1_2,a1_3,c,gamma]
        init_params = [0.0001, 0.1, 0.05, 0.02, 0.0001, 0.08, 0.03, 0.01, np.mean(transition_var), 1.43]

        # 定义损失函数（均方误差）
        def loss(params):
            y_pred = starx_model(params, X, transition_var)
            return mean_squared_error(y, y_pred)
    else:
        # 融入偏离因子模型（Model4）参数：[a0_0-a0_6, a1_0-a1_6, c, gamma]
        init_params = [0.0001, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01,
                       0.0001, 0.08, 0.03, 0.01, 0.008, 0.008, 0.008,
                       np.mean(transition_var), 1.43]

        def loss(params):
            y_pred = starx_model_with_benford(params, X, transition_var)
            return mean_squared_error(y, y_pred)

    # 非线性最小二乘优化
    result = minimize(loss, init_params, method='L-BFGS-B')
    best_params = result.x
    # 计算预测值和RMSE
    if not with_benford:
        y_pred = starx_model(best_params, X, transition_var)
    else:
        y_pred = starx_model_with_benford(best_params, X, transition_var)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return best_params, y_pred, rmse


# ===================== 5. 主流程执行 =====================
if __name__ == "__main__":
    # Step1: 模拟+预处理数据
    raw_data = simulate_high_freq_data(start_price=100, total_seconds=3600)
    processed_data = preprocess_data(raw_data)

    # Step2: 计算本福特偏离因子
    window_size = 200
    processed_data['Br'] = calculate_benford_deviation_factor(processed_data['log_return'], window_size)
    processed_data['Bv'] = calculate_benford_deviation_factor(processed_data['log_volume'], window_size)
    processed_data['Bd'] = calculate_benford_deviation_factor(processed_data['trade_duration'], window_size)

    # Step3: 准备建模数据（剔除NaN，划分训练/测试集）
    model_data = processed_data.dropna(
        subset=['log_return', 'log_volume', 'trade_duration', 'Br', 'Bv', 'Bd']).reset_index(drop=True)
    # 滞后1期（用t-1期数据预测t期收益率）
    model_data['log_return_lag1'] = model_data['log_return'].shift(1)
    model_data['log_volume_lag1'] = model_data['log_volume'].shift(1)
    model_data['trade_duration_lag1'] = model_data['trade_duration'].shift(1)
    model_data = model_data.dropna().reset_index(drop=True)

    # 划分训练集（80%）、测试集（20%）
    train_size = int(len(model_data) * 0.8)
    train = model_data.iloc[:train_size]
    test = model_data.iloc[train_size:]

    # Step4: 构建解释变量矩阵（STARX模型输入）
    ## 基准模型（Model3）：无偏离因子
    X_train_base = np.column_stack([
        np.ones(len(train)),  # 常数项
        train['log_return_lag1'].values,
        train['log_volume_lag1'].values,
        train['trade_duration_lag1'].values
    ])
    X_test_base = np.column_stack([
        np.ones(len(test)),
        test['log_return_lag1'].values,
        test['log_volume_lag1'].values,
        test['trade_duration_lag1'].values
    ])

    ## 融合偏离因子模型（Model4）
    X_train_benford = np.column_stack([
        np.ones(len(train)),  # 常数项
        train['log_return_lag1'].values,
        train['log_volume_lag1'].values,
        train['trade_duration_lag1'].values,
        train['Br'].values,
        train['Bv'].values,
        train['Bd'].values
    ])
    X_test_benford = np.column_stack([
        np.ones(len(test)),
        test['log_return_lag1'].values,
        test['log_volume_lag1'].values,
        test['trade_duration_lag1'].values,
        test['Br'].values,
        test['Bv'].values,
        test['Bd'].values
    ])

    # 过渡变量（选收益率滞后1期）
    transition_var_train = train['log_return_lag1'].values
    transition_var_test = test['log_return_lag1'].values

    # 目标变量（t期对数收益率）
    y_train = train['log_return'].values
    y_test = test['log_return'].values

    # Step5: 拟合STARX模型
    ## 基准模型（Model3）
    params_base, y_pred_train_base, rmse_train_base = fit_starx_model(y_train, X_train_base, transition_var_train)
    y_pred_test_base, rmse_test_base = starx_model(params_base, X_test_base, transition_var_test), np.sqrt(
        mean_squared_error(y_test, starx_model(params_base, X_test_base, transition_var_test)))

    ## 融合偏离因子模型（Model4）
    params_benford, y_pred_train_benford, rmse_train_benford = fit_starx_model(y_train, X_train_benford,
                                                                               transition_var_train, with_benford=True)
    y_pred_test_benford = starx_model_with_benford(params_benford, X_test_benford, transition_var_test)
    rmse_test_benford = np.sqrt(mean_squared_error(y_test, y_pred_test_benford))

    # Step6: 输出模型结果
    print("===== STARX模型性能 =====")
    print(f"基准模型（无偏离因子）- 训练集RMSE: {rmse_train_base:.6f}, 测试集RMSE: {rmse_test_base:.6f}")
    print(f"融合偏离因子模型 - 训练集RMSE: {rmse_train_benford:.6f}, 测试集RMSE: {rmse_test_benford:.6f}")

    # Step7: 可视化预测结果（测试集）
    plt.figure(figsize=(14, 8))
    # 子图1：真实值vs基准模型预测值
    plt.subplot(2, 1, 1)
    plt.plot(test['time'].to_numpy(), y_test, label='真实收益率', alpha=0.8, color='blue')
    plt.plot(test['time'].to_numpy(), y_pred_test_base, label='基准STARX预测', alpha=0.7, color='orange')
    plt.title('基准STARX模型（无本福特偏离因子）- 测试集预测')
    plt.xlabel('时间')
    plt.ylabel('1秒级对数收益率')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    # 子图2：真实值vs融合偏离因子模型预测值
    plt.subplot(2, 1, 2)
    plt.plot(test['time'].to_numpy(), y_test, label='真实收益率', alpha=0.8, color='blue')
    plt.plot(test['time'].to_numpy(), y_pred_test_benford, label='STARX+本福特偏离因子预测', alpha=0.7, color='green')
    plt.title('融合本福特偏离因子的STARX模型 - 测试集预测')
    plt.xlabel('时间')
    plt.ylabel('1秒级对数收益率')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()