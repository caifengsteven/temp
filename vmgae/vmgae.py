import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastdtw import fastdtw
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.data import Data

# 定义WDTW距离函数
def wdtw_distance(series1, series2, gamma=0.5, window_size=10):
    distance, path = fastdtw(series1, series2)
    return distance


# 计算WDTW距离矩阵
def compute_wdtw_matrix(data, gamma=0.5, window_size=10):
    n_stocks = data.shape[1]
    distance_matrix = np.zeros((n_stocks, n_stocks))

    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):  # 计算上三角矩阵
            distance = wdtw_distance(data.iloc[:, i].values, data.iloc[:, j].values, gamma, window_size)
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix


# 将距离矩阵转换为邻接矩阵
def convert_to_adjacency_matrix(distance_matrix, alpha=0.05):
    n_stocks = distance_matrix.shape[0]
    adjacency_matrix = np.zeros((n_stocks, n_stocks))
    # 计算阈值
    threshold = np.sort(distance_matrix[np.triu_indices(n_stocks, k=1)])[
        int(alpha * len(distance_matrix[np.triu_indices(n_stocks, k=1)]))]

    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            if distance_matrix[i, j] < threshold:
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1

    return adjacency_matrix



class VMGAE(nn.Module):
    def __init__(self, num_features, hidden_dim, out_dim, num_clusters):
        super(VMGAE, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.fc_mu = nn.Linear(out_dim, num_clusters)
        self.fc_logvar = nn.Linear(out_dim, num_clusters)

    def encode(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Simple inner product decoder
        adj_recon = torch.sigmoid(torch.matmul(z, z.T))
        return adj_recon

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        z_reparameterized = self.reparameterize(mu, logvar)
        adj_recon = self.decode(z_reparameterized)
        return adj_recon, mu, logvar

    def loss(self, adj_recon, edge_index, mu, logvar):
        recon_loss = 0
        for i, j in edge_index.t():
            # 获取节点i和j的mu向量
            mu_i = mu[i]
            mu_j = mu[j]
            # 计算重建损失
            recon_loss += torch.sum((adj_recon[i, j] - torch.sigmoid(torch.matmul(mu_i, mu_j.T))) ** 2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss



# 加载CSV文件
df = pd.read_csv('closes.csv', index_col=0)

df = df.pct_change().fillna(0)
data = df
# 计算WDTW距离矩阵
wdtw_matrix = compute_wdtw_matrix(data)

# 转换为邻接矩阵
adjacency_matrix = convert_to_adjacency_matrix(wdtw_matrix)

# 将邻接矩阵和节点特征转换为张量
adjacency_matrix = torch.FloatTensor(adjacency_matrix)
node_features = torch.FloatTensor(data.values).T

# Hyperparameters
num_features = node_features.shape[1]  #时序长度或者自定义特征数
hidden_dim = 64
out_dim = 64
num_clusters = 3  # 类别数
learning_rate = 0.0001
num_epochs = 10


# Initialize VMGAE model
model = VMGAE(num_features, hidden_dim, out_dim, num_clusters)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Create edge_index from adjacency matrix
edge_index = torch.nonzero(adjacency_matrix).t()

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    adj_recon, mu, logvar = model(node_features, edge_index)
    loss = model.loss(adj_recon, edge_index, mu, logvar)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

print("Training completed.")

# 获取每只股票最终特征
features = model(node_features, edge_index)[1]
features = features.detach().cpu().numpy()
# GMM
gmm = GaussianMixture(n_components=num_clusters)
cluster_labels = gmm.fit_predict(features)

print("聚类结果:")
print(cluster_labels)


#根据聚类结果分组股票
cluster_map = {}
for i,c in enumerate(cluster_labels):
    stock = df.columns[i]
    if cluster_map.get(c) is None:
        cluster_map[c] = [stock]
    else:
        cluster_map[c].append(stock)

sa = cluster_map.get(0)
sb = cluster_map.get(1)
window = 100

df['spread'] =df[sa].mean(axis=1).rolling(window).sum()-df[sb].mean(axis=1).rolling(window).sum()

df['mean_spread'] = df['spread'].rolling(window=window).mean()
df['std_spread'] = df['spread'].rolling(window=window).std()
k=2
df['upper_bound'] = df['mean_spread'] + k * df['std_spread']
df['lower_bound'] = df['mean_spread'] - k * df['std_spread']
df['signal'] = 0
#对冲
df.loc[df['spread'] > df['upper_bound'], 'signal'] = -1
df.loc[df['spread'] < df['lower_bound'], 'signal'] = 1


capital =  (1 + df[sa].mean(axis=1)* -df['signal'] + df[sb].mean(axis=1) * (df['signal'])).cumprod()

# 绘制最终收益曲线
plt.figure(figsize=(14, 7))
plt.plot(capital, label='Capital Over Time')
plt.title('Capital Over Time')
plt.xlabel('Date')
plt.ylabel('Capital')
plt.legend()
plt.show()

