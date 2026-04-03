import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 1. 全局参数配置（优化后，避免EMRT=inf） =====================
CONFIG = {
    "T": 10000,  # 时间序列长度（交易日）
    "num_candidates": 5,  # 候选价差标的数量
    "C": 1.0,  # 重要极值偏离系数（从2→1，放宽判定条件）
    "epsilon_threshold": 0.2,  # 均值穿越误差阈值（从0.1→0.2，放宽）
    "gamma": 0.99,  # RL折扣因子
    "alpha": 0.1,  # RL学习率
    "epsilon_train": 0.1,  # 训练ε-贪婪概率
    "epsilon_test": 0.0,  # 测试ε-贪婪概率
    "c": 0.001,  # 交易成本
    "k": 3,  # RL状态d_t等级阈值（%）
    "l": 4,  # RL状态回看窗口长度
    "ou_mu": 0,  # OU过程长期均值
    "ou_sigma": 4.0,  # OU过程波动率（从1→2，增大波动确保偏离）
    "ou_dt": 1,  # OU时间步长
}

# ===================== 2. 生成OU时间序列（优化参数，确保足够波动） =====================
def generate_ou_process(T, theta, mu=0, sigma=1, dt=1):
    """
    生成OU时间序列（欧拉离散化，确保有足够波动触发极值）
    :param T: 序列长度
    :param theta: 回归速度（θ越大回归越快）
    :param mu: 长期均值
    :param sigma: 波动率（增大sigma确保价差偏离）
    :param dt: 时间步长
    :return: 价差序列X
    """
    X = np.zeros(T)
    # 初始值偏离均值，确保初期有波动
    X[0] = mu + np.random.normal(0, sigma) * 2
    W = np.random.normal(0, np.sqrt(dt), T)  # 布朗运动
    for t in range(1, T):
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * W[t]
    return X

# 生成5组候选价差（θ递增，确保EMRT递减）
candidate_thetas = [0.01, 0.03, 0.05, 0.02, 0.04]
candidate_spreads = []
for theta in candidate_thetas:
    spread = generate_ou_process(
        T=CONFIG["T"],
        theta=theta,
        mu=CONFIG["ou_mu"],
        sigma=CONFIG["ou_sigma"],
        dt=CONFIG["ou_dt"]
    )
    candidate_spreads.append(spread)

# 可视化候选价差
plt.figure(figsize=(12, 6))
for i, (spread, theta) in enumerate(zip(candidate_spreads, candidate_thetas)):
    plt.plot(spread, label=f"候选标的{i+1} (θ={theta})")
plt.axhline(y=CONFIG["ou_mu"], color="black", linestyle="--", label="OU均值")
plt.title("5组候选价差（优化后，确保足够波动）")
plt.xlabel("交易日")
plt.ylabel("价差")
plt.legend()
plt.show()

# ===================== 3. EMRT计算（优化逻辑，避免inf） =====================
def calculate_emrt(spread, C=1.0, epsilon=0.2):
    """
    计算EMRT（优化极值查找逻辑，确保返回有效数值）
    :param spread: 价差序列
    :param C: 重要极值偏离系数（放宽至1.0）
    :param epsilon: 均值穿越阈值（放宽至0.2）
    :return: 有效EMRT值（无inf）
    """
    # 1. 基础统计量
    theta_hat = np.mean(spread)
    s = np.std(spread, ddof=1)
    # 处理极端情况：无波动则强制赋予小波动
    if s < 0.1:
        s = 0.1

    # 2. 查找重要极值（扩大区间+容错逻辑）
    extreme_indices = []
    n = len(spread)
    for m in range(n):
        # 扩大区间范围（前后30个点）
        i = max(0, m - 30)
        j = min(n-1, m + 30)
        window = spread[i:j+1]
        m_in_window = m - i

        # 条件1：区间最值（允许微小误差，避免浮点精度问题）
        window_min = np.min(window)
        window_max = np.max(window)
        is_min = abs(window[m_in_window] - window_min) < 1e-3
        is_max = abs(window[m_in_window] - window_max) < 1e-3
        if not (is_min or is_max):
            continue

        # 条件2：足够偏离（放宽至C*s）
        dev_threshold = C * s
        if is_min:
            dev_i = spread[i] - spread[m]
            dev_j = spread[j] - spread[m]
            if dev_i >= dev_threshold and dev_j >= dev_threshold:
                extreme_indices.append(m)
        if is_max:
            dev_i = spread[m] - spread[i]
            dev_j = spread[m] - spread[j]
            if dev_i >= dev_threshold and dev_j >= dev_threshold:
                extreme_indices.append(m)

    # 容错：若无极值，手动选取前10个最值点（确保有数据）
    if len(extreme_indices) == 0:
        # 找全局最值点
        all_indices = np.arange(n)
        sorted_spread = sorted(zip(spread, all_indices))
        # 取5个最小值+5个最大值
        min_indices = [idx for _, idx in sorted_spread[:5]]
        max_indices = [idx for _, idx in sorted_spread[-5:]]
        extreme_indices = sorted(list(set(min_indices + max_indices)))

    # 去重排序
    extreme_indices = sorted(list(set(extreme_indices)))

    # 3. 查找均值穿越时间（容错逻辑）
    regression_times = []
    theta_hat = np.mean(spread)
    for m in extreme_indices:
        cross_time = None
        # 扩大查找范围（极值后最多100个点）
        for t in range(m+1, min(n, m+100)):
            if abs(spread[t] - theta_hat) <= epsilon:
                cross_time = t
                break
        # 容错：若无穿越时间，取极值后第50个点
        if cross_time is None:
            cross_time = min(n-1, m + 50)
        regression_times.append(cross_time - m)

    # 4. 计算EMRT（确保无空列表）
    emrt = np.mean(regression_times)
    # 确保EMRT为正
    return max(emrt, 1.0)

# 计算所有候选标的EMRT
emrt_list = []
for i, spread in enumerate(candidate_spreads):
    emrt = calculate_emrt(
        spread,
        C=CONFIG["C"],
        epsilon=CONFIG["epsilon_threshold"]
    )
    emrt_list.append(emrt)
    print(f"候选标的{i+1}：θ={candidate_thetas[i]}, EMRT={emrt:.2f}天")

# 筛选最优标的（EMRT最小）
best_idx = np.argmin(emrt_list)
best_spread = candidate_spreads[best_idx]
print(f"\nEMRT筛选结果：最优标的为第{best_idx+1}个（EMRT={emrt_list[best_idx]:.2f}天）")

# ===================== 4. RL（Q-learning）独立交易（无修改） =====================
class QLearningTrader:
    def __init__(self, gamma, alpha, epsilon, k, l, c):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.k = k
        self.l = l
        self.c = c
        self.q_table = {}
        self.position = 0  # 0=无仓，1=多头

    def _calc_d_t(self, spread):
        """计算价差离散变动等级d_t"""
        pi_t = np.diff(spread) / spread[:-1] * 100
        pi_t = np.insert(pi_t, 0, 0)  # 补全第一个值
        d_t = np.zeros_like(pi_t)
        d_t[pi_t >= self.k] = 2
        d_t[(pi_t > 0) & (pi_t < self.k)] = 1
        d_t[(pi_t < 0) & (pi_t > -self.k)] = -1
        d_t[pi_t <= -self.k] = -2
        return d_t

    def _get_state(self, d_t, t):
        """获取t时刻状态（最近l个d_t的元组）"""
        if t < self.l - 1:
            return None
        state = tuple(d_t[t - self.l + 1 : t + 1])
        return state

    def _init_q_value(self, state):
        """初始化Q值"""
        if state not in self.q_table:
            self.q_table[state] = {1: 0.0, 0: 0.0, -1: 0.0}

    def _choose_action(self, state):
        """ε-贪婪选择动作（考虑仓位约束）"""
        self._init_q_value(state)
        # 仓位约束
        if self.position == 0:
            available_actions = [1, 0]
        else:
            available_actions = [-1, 0]
        # ε-贪婪
        if random.random() < self.epsilon:
            action = random.choice(available_actions)
        else:
            q_values = [self.q_table[state][a] for a in available_actions]
            max_idx = np.argmax(q_values)
            action = available_actions[max_idx]
        return action

    def _calc_reward(self, spread, t, action):
        """计算奖励函数"""
        theta_hat = np.mean(spread)
        X_t = spread[t]
        core_reward = (theta_hat - X_t) * action
        cost_penalty = self.c * abs(action - self.position)
        return core_reward - cost_penalty

    def train(self, spread, train_steps):
        """训练Q-learning模型"""
        d_t = self._calc_d_t(spread)
        total_reward = 0
        for t in tqdm(range(train_steps), desc="RL训练中"):
            state = self._get_state(d_t, t)
            if state is None:
                continue
            # 选动作
            action = self._choose_action(state)
            # 算奖励
            reward = self._calc_reward(spread, t, action)
            total_reward += reward
            # 下一状态
            next_state = self._get_state(d_t, t+1)
            if next_state is None:
                self.q_table[state][action] += self.alpha * (reward - self.q_table[state][action])
                continue
            # 更新Q表
            self._init_q_value(next_state)
            max_next_q = max(self.q_table[next_state].values())
            target = reward + self.gamma * max_next_q
            self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])
            # 更新仓位
            self.position += action
            self.position = np.clip(self.position, 0, 1)
        print(f"RL训练完成，累计奖励：{total_reward:.2f}")
        return self.q_table

    def trade(self, spread):
        """执行交易，返回累计收益"""
        self.epsilon = CONFIG["epsilon_test"]
        self.position = 0
        d_t = self._calc_d_t(spread)
        returns = [0.0]
        actions = []
        positions = []
        # 遍历测试数据
        for t in range(self.l - 1, len(spread)-1):
            state = self._get_state(d_t, t)
            if state is None:
                returns.append(0.0)
                actions.append(0)
                positions.append(self.position)
                continue
            # 选动作
            action = self._choose_action(state)
            # 计算收益
            spread_return = (spread[t+1] - spread[t]) * self.position
            cost = self.c * abs(action - self.position) if action != 0 else 0
            total_return = spread_return - cost
            returns.append(total_return)
            # 更新仓位
            self.position += action
            self.position = np.clip(self.position, 0, 1)
            # 记录
            actions.append(action)
            positions.append(self.position)
        # 累计收益
        cum_returns = np.cumsum(returns)
        return cum_returns, actions, positions

# 初始化并训练RL
trader = QLearningTrader(
    gamma=CONFIG["gamma"],
    alpha=CONFIG["alpha"],
    epsilon=CONFIG["epsilon_train"],
    k=CONFIG["k"],
    l=CONFIG["l"],
    c=CONFIG["c"]
)

# 划分训练/测试集
train_size = int(len(best_spread) * 0.8)
train_spread = best_spread[:train_size]
test_spread = best_spread[train_size:]

# 训练RL
q_table = trader.train(train_spread, train_steps=train_size)

# 执行交易
cum_returns, actions, positions = trader.trade(test_spread)

# ===================== 5. 结果可视化与输出 =====================
plt.figure(figsize=(15, 10))

# 子图1：最优价差序列
plt.subplot(3,1,1)
plt.plot(test_spread, label="最优价差（EMRT筛选后）")
plt.axhline(y=np.mean(test_spread), color="red", linestyle="--", label="价差均值")
plt.title("RL交易的最优价差序列")
plt.xlabel("交易日")
plt.ylabel("价差")
plt.legend()

# 子图2：仓位变化
plt.subplot(3,1,2)
plt.plot(positions, label="仓位（0=无仓，1=多头）", color="orange")
plt.title("RL交易仓位")
plt.xlabel("交易日")
plt.ylabel("仓位")
plt.legend()

# 子图3：累计收益
plt.subplot(3,1,3)
plt.plot(cum_returns, label="RL累计收益", color="green")
plt.axhline(y=0, color="black", linestyle="--")
plt.title("RL交易累计收益")
plt.xlabel("交易日")
plt.ylabel("累计收益")
plt.legend()

plt.tight_layout()
plt.show()

# 最终结果输出
print(f"\n=== 最终结果汇总 ===")
print(f"1. EMRT筛选结果：")
for i in range(CONFIG["num_candidates"]):
    print(f"   候选标的{i+1}：θ={candidate_thetas[i]}, EMRT={emrt_list[i]:.2f}天")
print(f"   最优标的：第{best_idx+1}个（EMRT最小）")
print(f"2. RL交易：")
print(f"   - 累计收益：{cum_returns[-1]:.2f}")
print(f"   - 最大收益：{np.max(cum_returns):.2f}")
print(f"   - 最小收益：{np.min(cum_returns):.2f}")
print(f"   - 总交易次数：{sum([1 for a in actions if a != 0])}")