# -*- coding: utf-8 -*-
"""交易执行优化模型复现
"""


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# ==========================================
# 1. 市场模拟与冲击模型 (Propagator Model)
# ==========================================

class TransientImpactModel:
    """
    实现了论文 2.1.2 节所述的瞬时冲击模型。
    使用指数衰减核 G(l) = G0 * e^(-l/tau) 和平方根冲击函数。
    """
    def __init__(self, g0=1e-4, tau=10.0, gamma=0.0005):
        self.g0 = g0        # 初始冲击系数
        self.tau = tau      # 衰减周期 (分钟)
        self.gamma = gamma  # 瞬时冲击比例因子
        self.history = []   # 存储 (impact_contribution, timestamp)

    def calculate_impact(self, q_t, v_t, current_time):
        """
        计算当前时刻 t 的累积瞬时冲击 I_t
        q_t: 代理成交量, v_t: 市场成交量
        """
        if v_t <= 0: return 0.0

        # 1. 计算当前交易产生的即时冲击 contribution
        # f(q, V) = gamma * sqrt(|q|/V)
        immediate_contribution = self.gamma * np.sqrt(np.abs(q_t) / v_t)
        self.history.append({'val': immediate_contribution, 'sign': np.sign(q_t), 'time': current_time})

        # 2. 累积过去所有交易的衰减影响
        total_impact = 0.0
        for trade in self.history:
            lag = current_time - trade['time']
            decay = self.g0 * np.exp(-lag / self.tau)
            total_impact += decay * trade['val'] * trade['sign']

        return total_impact

# ==========================================
# 2. GEO 仿真环境 (Gymnasium Environment)
# ==========================================

class GEOEnv(gym.Env):
    """
    GEO: Gymnasium for Executing Optimally.
    实现了 2.3.3 节所述的环境设计。
    """
    def __init__(self, data_stream, horizon=390):
        super(GEOEnv, self).__init__()
        self.data_stream = data_stream
        self.horizon = horizon

        # 动作空间: alpha_t ∈ {-1, -0.75, ..., 0.75, 1} (共 9 个离散值)
        self.action_values = np.linspace(-1, 1, 9)
        self.action_space = spaces.Discrete(len(self.action_values))

        # 状态空间: 13 维特征向量 (见论文 2.3.3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.impact_model = TransientImpactModel()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.q_total = 100000  # 假设总订单量 Q0
        self.q_remaining = self.q_total
        self.side = 1  # 1 为买入, -1 为卖出

        # Clear impact model history for a new episode
        self.impact_model.history = []

        # 获取市场初始数据
        self.market_data = self.data_stream.get_episode_data()
        self.p_arrival = self.market_data['mid'].iloc[0]
        self.q_last = 0
        self.i_cum = 0

        return self._get_obs(), {}

    def _get_obs(self):
        """
        构造 13 维观测向量
        """
        # Ensure current_step does not exceed the valid index range of market_data
        # When self.current_step == self.horizon, it means the episode has ended.
        # In such cases, we should use the last valid data point for the observation.
        data_index = min(self.current_step, len(self.market_data) - 1)
        m = self.market_data.iloc[data_index]

        # Ensure the previous mid price index is also within bounds
        prev_mid_price_index = max(0, data_index - 1)

        # 简化版 13 维特征
        obs = np.array([
            m['volume'],            # 1. Vt (市场成交量)
            self.q_last,            # 2. q_{t-1} (上一步交易量)
            self.horizon - self.current_step, # 3. 剩余时间
            self.q_remaining,       # 4. 剩余库存
            0.02,                   # 5. ADV% (假设值)
            self.q_remaining / (self.horizon - self.current_step + 1e-5), # 6. EHV%
            0.0,                    # 7. I_imm (即时冲击)
            self.i_cum,             # 8. I_cum (累积冲击)
            m['mid'],               # 9. 当前中间价
            self.market_data['mid'].iloc[prev_mid_price_index], # 10. 上一步中间价
            self.p_arrival,         # 11. 到达价基准
            0.015,                  # 12. 1-day 波动率
            0.012                   # 13. 5-day 波动率
        ], dtype=np.float32)
        return obs

    def step(self, action_idx):
        at = self.action_values[action_idx]
        m = self.market_data.iloc[self.current_step]

        # 1. 计算执行数量 (根据论文 2.3.3 公式)
        # target_rate 基于剩余库存和预期市场成交量的平均分布
        if self.current_step >= self.horizon: # No future volume if episode is over
            e_vh = 0.0
        else:
            # Calculate expected future market volume only for remaining steps
            remaining_market_volume_series = self.market_data['volume'].iloc[self.current_step:self.horizon]
            if not remaining_market_volume_series.empty:
                e_vh = np.mean(remaining_market_volume_series) * (self.horizon - self.current_step)
            else:
                e_vh = 0.0 # No more future volume to consider

        p_target = self.q_remaining / (e_vh + 1e-9)
        p_action = p_target * (1 + at)

        q_t = p_action * m['volume']
        # 限制不超卖
        q_t = min(q_t, self.q_remaining)
        if self.current_step == self.horizon - 1: # 最后一分钟必须完结
            q_t = self.q_remaining

        # 2. 计算市场冲击和成交价
        i_t = self.impact_model.calculate_impact(q_t, m['volume'], self.current_step)
        p_fill = m['vwap'] * (1 + self.side * i_t)

        # 3. 更新状态
        self.q_last = q_t
        self.q_remaining -= q_t
        self.i_cum += np.abs(i_t)

        # 4. 计算奖励 (见论文 2.3.6)
        # 简化奖励: 最小化到达价滑点 (Arrival Slippage)
        slippage = self.side * (p_fill - self.p_arrival) / self.p_arrival * 1e4 # bps
        reward = -slippage

        # 进度控制
        self.current_step += 1
        terminated = self.current_step >= self.horizon or self.q_remaining <= 0
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, {"slippage": slippage}

# ==========================================
# 3. Mock 数据生成器 (用于演示)
# ==========================================

class MockMarketData:
    def __init__(self, length=390):
        self.length = length

    def get_episode_data(self):
        # 模拟生成 390 分钟的 K 线数据
        times = pd.date_range("09:30", periods=self.length, freq="1min")
        mid_prices = 150.0 + np.cumsum(np.random.normal(0, 0.05, self.length))
        volumes = np.random.lognormal(10, 1, self.length)
        return pd.DataFrame({
            'mid': mid_prices,
            'vwap': mid_prices + np.random.normal(0, 0.01, self.length),
            'volume': volumes
        }, index=times)

# ==========================================
# 4. MAP-Elites 质量-多样性管理器 (演示架构)
# ==========================================

class MAPElitesArchive:
    """
    MAP-Elites 存档。按照流动性(Liquidity)和波动率(Volatility)划分网格。
    """
    def __init__(self, bins=3):
        # 3x3 网格: [Low, Med, High] x [Low, Med, High]
        self.archive: Dict[Tuple[int, int], Dict] = {}
        self.bins = bins

    def get_cell(self, liquidity_rank, volatility_rank):
        # 将 [0,1] 的排名映射到网格索引
        l_idx = min(int(liquidity_rank * self.bins), self.bins - 1)
        v_idx = min(int(volatility_rank * self.bins), self.bins - 1)
        return (l_idx, v_idx)

    def update(self, policy, fitness, descriptors):
        cell = self.get_cell(descriptors['liq'], descriptors['vol'])
        if cell not in self.archive or fitness > self.archive[cell]['fitness']:
            print(f"-> 存档更新: 单元格 {cell}, 新适应度: {fitness:.4f}")
            self.archive[cell] = {'policy': policy, 'fitness': fitness}

# ==========================================
# 5. 主程序: 训练与复现
# ==========================================

if __name__ == "__main__":
    # 1. 初始化环境
    mock_data = MockMarketData()
    env_func = lambda: GEOEnv(mock_data)
    env = DummyVecEnv([env_func])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 2. 训练 PPO-MLP (基准模型)
    print("开始训练 PPO 策略...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01
    )
    model.learn(total_timesteps=10000)
    print("训练完成。\n")

    # 3. 演示 MAP-Elites 存档逻辑
    print("演示 MAP-Elites 存档更新逻辑...")
    archive = MAPElitesArchive(bins=3)

    # 模拟在不同市场条件下评估策略
    for _ in range(5):
        # 随机模拟不同的市场特质 (流动性, 波动率)
        test_liq = np.random.random()
        test_vol = np.random.random()

        # 在该特质下评估模型适应度 (这里使用负的滑点均值)
        # 在真实 MAP-Elites 中，这里会对策略权重进行高斯扰动 (Mutation)
        fake_fitness = -np.random.uniform(2, 5)

        archive.update(model.policy.state_dict(), fake_fitness, {'liq': test_liq, 'vol': test_vol})

    # 4. 测试模型
    print("\n测试训练后的 PPO 模型...")
    obs = env.reset()
    total_slippage = 0
    for _ in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_slippage += infos[0]['slippage']
        if dones[0]: break

    print(f"单次订单平均执行滑点: {total_slippage/10:.2f} bps")