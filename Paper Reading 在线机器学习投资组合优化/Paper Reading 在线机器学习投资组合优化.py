
# coding: utf-8

# #### **Bandit Learning**
# 
# Online机器学习方法在投资组合选择上的应用，参考文献
# 
# > *Wang J, Wang J, Jiang Y G, et al. Portfolio choices with orthogonal bandit learning[C] International Conference on Artificial Intelligence. AAAI Press, 2015:974-980.*
# 
# Bandit Learning 最初是为了解决多臂赌博机 (multi-armed bandit) 问题
# 
# >**for** t =1, 2 ... to **T**
# >&nbsp;   **play** $arm_{t} \in K$
# >&nbsp;   **observer** reward $r_{arm_{t}}^{t}$
# 
# 多臂赌博机问题目标是为了最大化 Long-term Reward，或者最小化Cumlative Pseudo Regret, 这里$r^{*}$为最优臂
# 
# >&nbsp;   **min** $ \rm{T}* \mathbb{E}(r^{*}) - \sum_{t=1}^{T} \mathbb{E}(r_{arm_{t}}^{t})$
# 
# 这里的臂 (arm) 可以抽象为不同的动作，对于量化模型来讲可以是一个资产也可以是一个资产组合，在t时刻我们可以看到对应arm过去的信息从而做出决策来最大化远期收益。和offline学习过程不同的是，online学习过程并不可见所有的数据集，优化目标并不是即时的损失而是远期的期望回报，本质上追求的是决策最优而不是的结果最优。
# 
# 一个最简单的方法是Follow-The-Leader(FTL)，每个时刻$t$都选择过去表现最好的arm，可以对应于常见的动量策略。FTl对每个arm的回报估计基于过去的平均表现。
# 
# >&nbsp;   $\forall t $ $ arm_{t} = \mathop{argmin}\limits_{k \in K}  \frac{1}{t-1} \sum_{i=1}^{t-1} r_{k}^{t}$ 
# 
# 在此基础上也有一些改进的版本例如Follow-the-Regularized-Leader(FTRL)，在FTL的估计后面增加一个约束项（通常是L2约束）增强解的稳定性。这篇论文里面作者考虑的是 Upper-Confidence-Bounds(UCB)算法，UCB是一个乐观的在线机器学习算法，它对每个arm的估计来源于过去的平均表现的上界。UCB算法的累积Pseudo Regret增长速率的上界为 $O(\frac{8n}{\Delta^{*}} ln(T) + 5K)$。
# 
# >&nbsp;   $\forall t $ $ arm_{t} = \mathop{argmin} \limits_{k \in K}  \frac{1}{t-1} \sum_{i=1}^{t-1} r_{k}^{t} + \sqrt{\frac{2ln(k)}{k_{i}}} $
# 
# 这里的$k_{i}$是我们选择第$i$个arm的次数。相对于FTL容易陷入局部最优，依赖于初始状态，UCB算法并不依赖于初始状态而总是会偏向最优的选择。

# In[ ]:

import numpy as np

class BernoulliArm:
    def __init__(self,r):
        self.r = r
        self.total_reward = 0.
        self.played_times = 0
    
    def get_average_reward(self):
        return self.total_reward / self.played_times
        
    def play(self):
        self.played_times += 1
        if np.random.random() < self.r:
            self.total_reward += 1
            return 1
        else:
            return 0

class FollowTheLeader:
    def __init__(self, arm_list):
        self.arm_list = arm_list
        self.arm_star = np.max([arm.r for arm in self.arm_list])
        self.persudo_regret = []
        
    def choose_arm(self):
        return np.argmax([arm.get_average_reward() for arm in self.arm_list])
    
    def run(self, T):
        for t in range(T):
            arm_t = self.choose_arm()
            self.arm_list[arm_t].play()
            self.persudo_regret.append(self.arm_star - self.arm_list[arm_t].r)
            
class UpperBoundConfidence(FollowTheLeader):
    def choose_arm(self):
        k = len(self.persudo_regret)
        return np.argmax([arm.get_average_reward() + np.sqrt(np.log(2*k)/arm.played_times) for arm in self.arm_list])


# Simple Experiment UCB vs FTL    
import matplotlib.pyplot as plt
import copy

arm_list = [BernoulliArm(r) for r in [0.1,0.11,0.12]]  

# Initial State
for arm in arm_list:
    arm.play()

# Perform FTL and UCB
FTL = FollowTheLeader(copy.deepcopy(arm_list))
UCB = UpperBoundConfidence(copy.deepcopy(arm_list))
FTL.run(5000)
UCB.run(5000)

# Plot Result
fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(111)
ax.plot(np.cumsum(FTL.persudo_regret))
ax.plot(np.cumsum(UCB.persudo_regret))
ax.set_xlabel('Epochs(t)')
ax.set_ylabel('Cumulative Regret')
ax.legend(['FTL', 'UCB'])


# 实验用的数据集为[FF48](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_48_ind_port.html)(数据为相对上期的百分比变化，缺失值默认用-99.99填充)，这个数据集涵盖了美股1926年至今的各行业的收益率，很多Portfolio Optimization的算法都会在这个数据集上进行实验。因为是美股的行业数据，一般并不会限制$w \geq 0$, 此时负的权重表示做空，如果要用在大A股的话要做一个投影操作把权重投影到我们的可行域里。
# >&nbsp;   $w_{new} = \mathop{argmin}\limits_{b \in \Delta}  \lVert b - w \lVert_{2}^{2} $  {$\Delta: \sum b = 1, b\geq 0$}

# In[ ]:

import pandas as pd

data = pd.read_txt('FF48.txt', index_col=['Date'], dtype= {'Date': str})
data = data[ ('197401' <= data.index.values.astype(str)) & (data.index.values.astype(str) < '201801')]
date = data.index.values

# The rate of return matrix, fill missing values with 100%
R = data.values.T.astype(np.float)
R[R < -99] = 0.
R = (R + 100) / 100


# 核心的问题是如何抽象出合适的arm用于投资组合选择的优化。
# 
# 这里用 $R$ 来表示$n$个资产的收益矩阵，$R_{k}$ 就表示$n$个资产在$k$时刻的收益率。用 $\sum_{k}$表示$n$个资产的协方差矩阵，因为资产收益之间是线性无关的，$\sum_{k}$一定是正定的，因此可以做一个特征值分解得到$n$个正的特征值。
# >&nbsp;   $\sum_{k} = H_{k}\Lambda_{k} H_{k}^{T}$
# 
# 其中 $H_{k}$是一个正交矩阵，表示了协方差矩阵的特征向量，$\Lambda_{k}$是对角矩阵，每个元素表示对应特征向量的非负特征值，这里让特征值由大到小排列$\lambda_{k,1} > \lambda_{k,2} > ... \lambda_{k,n} > 0$。特征向量就表示了互相不相关的$n$组投资组合的权重。为了满足投资组合权重的限制，需要对特征向量做归一化$\overline{H}\_{k} = \frac{H_{k}}{Z}$ 使得权重和为1。那么在$k$时刻的收益可以用$\overline{H}\_{k}^{T}R_{k}$来表示，风险可以用$\overline{\Lambda}\_{k} = \overline{H}\_{k} \sum_{k}\overline{H}\_{k}^{T}$来表示, 对角矩阵$\overline{\Lambda}\_{k}$的对角线上每个元素$\lambda_{k,i}$表示了每个组合的波动率。
# 
# 论文同时参考了金融学相关的实证研究的结论：
# >&nbsp;   $\overline{\sum}\_{k} = \underbrace{\sum_{i=1}^{l} \overline{\Lambda}\_{k,i} \overline{H}\_{k,i} \overline{H}\_{k,i}^{T}}\_{Passive\ Part}  +  \underbrace{\sum_{i=l+1}^{n} \overline{\Lambda}\_{k,i} \overline{H}\_{k,i} \overline{H}\_{k,i}^{T}}\_{Active\ Part}$
# 
# 考虑特征值分解的$n$个不相关投资组合，特征值最大的若干部分反映了这个市场的被动投资收益部分，最小的若干部分反映了主动投资的收益部分。论文希望通过组合Passive 和 Active 的Portfolio来在保证组合能够追随市场获得被动收益的同时也能够兼顾一定的主动超额收益。
# 
# 因此可以分别运行两个UCB算法，分别从前 $l$ 和后 $n-l$ 个投资组合中选择最优的投资组合来构建新的投资组合来兼顾被动和主动收益，目标是最大化夏普比率。
# >&nbsp;   $SharpeRatio_{k,i} = \frac{H_{k,i}\mathbb{E}(R_{k,i})}{\sqrt{\lambda_{k,i}}}$
# 
# 假设分别选择了 $i^{\*} \in \[1,j\],\ j^{\*} \in (l,n]$, 因为两组投资组合相互独立，因此通过最小化方差可以得到最优的分配权重$\theta$
# >&nbsp;   $\lambda_{k,p} = \theta^{2}\lambda_{k,j^{\*}} + (1-\theta)^{2}\lambda_{k,i^{\*}} \rightarrow \theta = \frac{\lambda_{k,i^{\*}}}{\lambda_{k,i^{\*}} +\lambda_{k,j^{\*}}}$
# 
# 因此最后的组合权重为
# >&nbsp;   $w = \theta \overline{H}\_{k,j^{\*}}+(1-\theta)\overline{H}\_{k,i^{\*}}$
# 
# 原文分别使用 Factor Model 和 James-Stein shrinkage estimator来估计协方差和均值，这里实现简单用平均来估计。

# In[ ]:

from sklearn.preprocessing import MinMaxScaler

class OrthogonalBanditPortfolio:
    def __init__(self, R):
        self.R = R
        self.n_arms, self.n_samples = R.shape
        
    def run(self, window_size, l):
        self.reward = np.ones(self.n_samples - window_size)
        self.played_times = np.zeros(self.n_arms)
        
        for t in range(window_size, self.n_samples):
            sliceR = self.R[:, t-window_size:t]
            
            # Compute the covariance matrix
            covariance_matrix =  np.cov(sliceR)
            
            # Eigenvalue Decomposition
            A, H = np.linalg.eig(covariance_matrix)
            
            # All eigenvalues are positive
            assert(np.sum(A<0) == 0)
            
            # Sort the eigenvalues
            idx = np.argsort(-A)
            A = np.diag(A[idx])
            H = H[:,idx]
            
            # Normalized weight
            H /= np.sum(H, axis= 0)
            ANew =  H.T.dot(covariance_matrix).dot(H)
            
            # Compute the sharpe ratio
            portfolio_reward = H.T.dot(sliceR)

            sharpe_ratio = np.mean(portfolio_reward, axis=1) / np.sqrt(ANew.diagonal())
            sharpe_ratio = MinMaxScaler().fit_transform(sharpe_ratio.reshape(-1,1)).reshape(-1)
            
            # Compute the upper bound of expected reward
            sharpe_ratio_upper_bound = sharpe_ratio + np.sqrt((2*np.log(t))/(window_size+self.played_times))
            
            # Select the optimal arm
            action1 = np.argmax(sharpe_ratio_upper_bound[:l])
            action2 = np.argmax(sharpe_ratio_upper_bound[l:])+l

            self.played_times[action1] += 1
            self.played_times[action2] += 1

            # Optimal weight
            Adiag = ANew.diagonal()
            theta = Adiag[action1] / (Adiag[action1] + Adiag[action2])
            weight = (1-theta)*H[:,action1] + theta*H[:,action2]
            
            self.reward[t-window_size] = weight.dot(self.R[:,t])


# 考虑两个基准策略，定期等权重调仓以及买入持有
# 
# >Constant Weight Rebalance: $ \prod_{t=1}^{T} w^{T}r_{t} $
# >
# >Equal weight Portfolio: $ w^{T}  \prod_{t=1}^{T} r_{t}$
# 
# $r_{t}$ 表示 $t$ 时刻的每个资产的收益率, $w$ 表示权重分配的向量，对于等权策略来说 $ \sum w^{T}  \mathcal{\bf{1}} = 1$

# In[ ]:

# Bandit Learning
window_size = 120
orthogonal_bandit_portfolio = OrthogonalBanditPortfolio(R)
orthogonal_bandit_portfolio.run(window_size = window_size, l = 3)

#Baseline
constant_weight_rebalance = np.cumprod(R[:,window_size:].mean(axis=0))
equal_weight_portfolio = np.mean(np.cumprod(R[:,window_size:], axis=1), axis=0)

date = data.index.values[window_size:]


# In[ ]:

import matplotlib.pyplot as plt
# Plot Result
fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(111)
ax.plot(constant_weight_rebalance)
ax.plot(equal_weight_portfolio)
ax.plot(np.cumprod(orthogonal_bandit_portfolio.reward))

xticks = np.arange(0, len(date), 15)
ax.set_xticks(xticks)
ax.set_xticklabels(date[xticks], rotation = 45)

ax.set_xlabel('Date')
ax.set_ylabel('Cumulateive Wealth')
ax.legend(['Constant Weight Rebalance', 'Equal Weight Portfolio', 'Bandit Learning'], loc='upper left')


# UCB Portfolio算法的缺陷:
# 1.算法假设市场流动性充足且并没有考虑交易成本
# 2.在大跌时过于乐观，使用上界估计夏普比率，在大跌时容易大幅回撤
# 
