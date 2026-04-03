import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# ======================== 1. 核心公式定义与工具函数 ========================
def calculate_performance_gap(f_a, f_b, X_val, y_val):
    """
    计算两个模型的性能差距Δ_hat(t,ℓ)
    公式：Δ̂_{t,ℓ} = (1/n)Σ[(f_a(x)-y)² - (f_b(x)-y)²]
    """
    pred_a = f_a.predict(X_val)
    pred_b = f_b.predict(X_val)
    mse_a = np.mean((pred_a - y_val) ** 2)
    mse_b = np.mean((pred_b - y_val) ** 2)
    return mse_a - mse_b


def bias_proxy(performance_gaps, psi_vals, ell):
    """
    计算非平稳性偏差代理φ̂(t,ℓ,δ')
    公式：φ̂ = max_i∈[ℓ] (|Δ̂_{t,ℓ}-Δ̂_{t,i}| - [ψ(t,ℓ)+ψ(t,i)])_+
    """
    max_val = 0
    for i in range(1, ell + 1):
        if i >= len(performance_gaps):
            continue
        delta_diff = abs(performance_gaps[ell - 1] - performance_gaps[i - 1])
        psi_sum = psi_vals[ell - 1] + psi_vals[i - 1]
        current = max(delta_diff - psi_sum, 0)
        if current > max_val:
            max_val = current
    return max_val


def variance_proxy(var_est, n, delta_prime=0.05, M=1):
    """
    计算统计不确定性代理ψ̂(t,ℓ,δ')（Bernstein不等式推导）
    公式：ψ̂ = v̂√(2log(2/δ')/n) + 64M²log(2/δ')/(3(n-1))
    """
    log_term = np.log(2 / delta_prime)
    term1 = var_est * np.sqrt(2 * log_term / n)
    term2 = (64 * (M ** 2) * log_term) / (3 * (n - 1)) if n > 1 else 0
    return term1 + term2


def select_adaptive_validation_window(X, y, delta_prime=0.05, M=1,min_ell=400):
    """
    选择自适应验证窗口ℓ̂ = argmin_ℓ {φ̂(t,ℓ)+ψ̂(t,ℓ)}
    """
    max_ell = min(min_ell, len(X))  # 最大验证窗口长度（可调整）
    min_loss = float('inf')
    best_ell = 1

    # 预计算不同窗口的性能差距和方差代理
    performance_gaps = []
    psi_vals = []

    # 用基准模型（岭回归）计算性能差距
    base_model = Ridge()
    for ell in range(1, max_ell + 1):
        # 取前ell个样本作为验证数据
        X_ell = X[:ell]
        y_ell = y[:ell]
        if len(X_ell) < 5:  # 样本量过小跳过
            performance_gaps.append(0)
            psi_vals.append(float('inf'))
            continue

        # 训练基准模型
        base_model.fit(X_ell, y_ell)
        # 计算性能差距（与随机预测对比）
        random_pred = np.random.randn(len(y_ell)) * np.std(y_ell) + np.mean(y_ell)
        pg = np.mean((base_model.predict(X_ell) - y_ell) ** 2) - np.mean((random_pred - y_ell) ** 2)
        performance_gaps.append(pg)

        # 计算方差代理
        var_est = np.var(y_ell)
        psi = variance_proxy(var_est, len(X_ell), delta_prime, M)
        psi_vals.append(psi)

    # 遍历所有可能的窗口，找到最优ℓ
    for ell in range(1, max_ell + 1):
        if len(X[:ell]) < 5:
            continue
        # 计算偏差代理
        phi = bias_proxy(performance_gaps, psi_vals, ell)
        # 计算方差代理
        var_est = np.var(y[:ell])
        psi = variance_proxy(var_est, len(X[:ell]), delta_prime, M)
        # 总损失：φ + ψ
        total_loss = phi + psi

        if total_loss < min_loss:
            min_loss = total_loss
            best_ell = ell

    return best_ell


# ======================== 2. 数据预处理 ========================
def preprocess_kline_data(file_path):
    """
    处理K线CSV数据，生成预测特征和标签
    特征：OHLCV衍生特征（收益率、波动率、成交量占比等）
    标签：下一期收盘价收益率
    """
    # 读取数据
    df = pd.read_csv(file_path)

    # 转换时间戳为datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.sort_values('open_time').reset_index(drop=True)

    # 计算核心特征
    # 1. 收益率特征
    df['return'] = (df['close'] - df['open']) / df['open']  # 当期收益率
    df['high_low_ratio'] = (df['high'] - df['low']) / df['open']  # 波动率
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()  # 成交量相对20期均值
    df['buy_volume_ratio'] = df['buy_volume'] / df['volume']  # 主动买盘占比

    # 2. 滞后特征（用前N期数据预测）
    for lag in [1, 3, 5]:
        df[f'return_lag_{lag}'] = df['return'].shift(lag)
        df[f'volume_ratio_lag_{lag}'] = df['volume_ratio'].shift(lag)

    # 3. 标签：下一期收盘价收益率
    df['target'] = df['return'].shift(-1)

    # 去除缺失值
    df = df.dropna().reset_index(drop=True)

    # 分离特征和标签
    feature_cols = [col for col in df.columns if col not in ['open_time', 'end_time', 'target', 'ignore']]
    X = df[feature_cols].values
    y = df['target'].values

    return df, X, y


# ======================== 3. 候选模型池构建 ========================
def build_candidate_pool( train_windows = [160,320,640,800,1000]):
    """
    构建候选模型池：不同复杂度模型 + 不同训练窗口
    模型类：线性（Ridge/Lasso/ElasticNet）+ 非线性（RandomForest）
    训练窗口：[10, 20, 40, 80]（指数级跨度，适配K线数据）
    """
    # 定义不同复杂度的模型
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    }

    # 定义训练窗口（单位：K线周期数）
    # 构建候选组合：(模型名称, 模型实例, 训练窗口长度)
    candidate_pool = []
    for model_name, model in models.items():
        for window in train_windows:
            candidate_pool.append((f"{model_name}_window_{window}", model, window))

    return candidate_pool


# ======================== 4. 锦标赛筛选算法 ========================
def tournament_selection(candidate_pool, X, y, best_ell):
    """
    序贯淘汰锦标赛：选出最优模型组合
    步骤：
    1. 随机选基准模型
    2. 成对比较所有模型，保留优胜者
    3. 重复至只剩1个模型
    """
    # 复制候选池避免修改原数据
    remaining_candidates = candidate_pool.copy()

    while len(remaining_candidates) > 1:
        # 随机选基准模型（pivot）
        pivot_idx = np.random.randint(0, len(remaining_candidates))
        pivot_name, pivot_model, pivot_window = remaining_candidates[pivot_idx]

        # 训练基准模型
        X_train_pivot = X[-pivot_window - best_ell: -best_ell]  # 训练数据：基准窗口+验证窗口之前
        y_train_pivot = y[-pivot_window - best_ell: -best_ell]
        pivot_model.fit(X_train_pivot, y_train_pivot)

        # 验证数据：最优验证窗口
        X_val = X[-best_ell:]
        y_val = y[-best_ell:]

        # 存储本轮优胜者
        winners = []

        # 成对比较
        for candidate in remaining_candidates:
            if candidate[0] == pivot_name:
                continue  # 跳过基准模型

            cand_name, cand_model, cand_window = candidate
            # 训练候选模型
            X_train_cand = X[-cand_window - best_ell: -best_ell]
            y_train_cand = y[-cand_window - best_ell: -best_ell]
            cand_model.fit(X_train_cand, y_train_cand)

            # 计算性能差距：Δ̂ = pivot_mse - cand_mse
            perf_gap = calculate_performance_gap(pivot_model, cand_model, X_val, y_val)

            # 性能差距>0 → 候选模型更优，加入优胜者
            if perf_gap > 0:
                winners.append(candidate)

        # 如果有优胜者，更新剩余候选；否则保留基准模型
        if winners:
            remaining_candidates = winners
        else:
            remaining_candidates = [remaining_candidates[pivot_idx]]

    return remaining_candidates[0]


# ======================== 5. 主函数：ATOMS算法入口 ========================
def atoms_algorithm(file_path):
    """
    ATOMS算法主流程
    """
    # Step 1: 数据预处理
    print("Step 1: 数据预处理...")
    df, X, y = preprocess_kline_data(file_path)
    print(f"预处理完成，因子维度：{X.shape}，标签维度：{y.shape}")

    # Step 2: 构建候选模型池
    print("\nStep 2: 构建候选模型池...")
    candidate_pool = build_candidate_pool()
    print(f"候选池规模：{len(candidate_pool)} 个模型-窗口组合")

    # Step 3: 选择自适应验证窗口
    print("\nStep 3: 选择最优验证窗口...")
    # 用最后n个样本作为验证窗口选择的基础数据
    test_window = 1000
    X_for_val_window = X[-test_window:]
    y_for_val_window = y[-test_window:]
    best_ell = select_adaptive_validation_window(X_for_val_window, y_for_val_window)
    print(f"最优验证窗口长度：{best_ell} 个K线周期")

    # Step 4: 锦标赛筛选最优模型
    print("\nStep 4: 锦标赛筛选最优模型...")
    best_candidate = tournament_selection(candidate_pool, X, y, best_ell)
    best_name, best_model, best_window = best_candidate

    # Step 5: 评估最优模型性能
    print("\nStep 5: 评估最优模型性能...")
    # 训练最优模型
    X_train_best = X[-best_window - best_ell: -best_ell]
    y_train_best = y[-best_window - best_ell: -best_ell]
    X_test = X[-best_ell:]
    y_test = y[-best_ell:]
    best_model.fit(X_train_best, y_train_best)

    # 计算测试误差
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    # 输出结果
    print("\n======================= ATOMS算法结果 =======================")
    print(f"最优模型-窗口组合：{best_name}")
    print(f"测试集MSE：{test_mse:.6f}")
    print(f"测试集R²：{test_r2:.6f}")

    return {
        'best_model_name': best_name,
        'best_train_window': best_window,
        'best_val_window': best_ell,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'best_model': best_model
    }


# ======================== 6. 运行算法 ========================
if __name__ == "__main__":
    # 替换为你的K线CSV文件路径
    FILE_PATH = "klines.csv"
    # 运行ATOMS算法
    results = atoms_algorithm(FILE_PATH)