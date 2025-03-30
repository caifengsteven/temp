import numpy as np
import matplotlib.pyplot as plt

# 趋势编码函数
def trend_encoding(data):
    if len(data) < 2:
        return np.array([])
    return np.where(data[1:] < data[:-1], -1, 1)

# 计算样本重叠函数
def vector_overlap(x, y):
    min_len = min(len(x), len(y))
    if min_len == 0:
        return 0
    x = x[:min_len]
    y = y[:min_len]
    return np.mean(x == y)

# 马氏距离计算函数
def mahalanobis_distance(x, mu, cov):
    if len(mu.shape) == 0 or mu.shape[0] == 1:
        return 0
    if x.shape != mu.shape:
        print(f"警告：x 形状 {x.shape} 与 mu 形状 {mu.shape} 不匹配，跳过马氏距离计算。")
        return 0
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return 0
    diff = x - mu
    return np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))

# 映射函数
def Dtp(u):
    # 缩放 u 以避免溢出
    if u > 700:  # 700 是一个经验值，可以根据实际情况调整
        u = 700
    return 2 / (1 + np.exp(u))

# 训练函数
def train(data, max_lag, m=3):
    all_trend_features = []
    all_positive_subsets = []
    all_negative_subsets = []
    for i in range(1, max_lag + 1):
        samples = []
        labels = []
        for j in range(len(data) - i):
            sample = data[j:j + i]
            label = data[j + i]
            samples.append(sample)
            labels.append(label)
        samples = np.array(samples)
        labels = np.array(labels)

        sample_trends = np.array([trend_encoding(s) for s in samples])
        unique_trends = np.unique(sample_trends, axis=0)
        subsets = []
        for trend in unique_trends:
            subset = samples[np.all(sample_trends == trend, axis=1)]
            # 确保子集样本长度和当前滞后阶一致
            valid_subset = [s for s in subset if len(s) == i]
            if len(valid_subset) >= m:
                subsets.append(np.array(valid_subset))

        positive_subsets = []
        negative_subsets = []
        for subset in subsets:
            if len(subset) < m:
                continue
            subset_indices = [np.where((samples == s).all(axis=1))[0][0] for s in subset]
            subset_labels = labels[subset_indices]

            positive_indices = []
            negative_indices = []
            for idx, sample in enumerate(subset):
                next_value = subset_labels[idx]
                last_value = sample[-1]
                if next_value > last_value:
                    positive_indices.append(idx)
                elif next_value < last_value:
                    negative_indices.append(idx)
            positive_subset = subset[positive_indices]
            negative_subset = subset[negative_indices]
            positive_subsets.append(positive_subset)
            negative_subsets.append(negative_subset)

            print(f"滞后阶 {i}, 子集: 正样本数量 {len(positive_subset)}, 负样本数量 {len(negative_subset)}")

        all_trend_features.append(unique_trends)
        all_positive_subsets.append(positive_subsets)
        all_negative_subsets.append(negative_subsets)

    return max_lag, all_trend_features, all_positive_subsets, all_negative_subsets

# 测试函数
def test(test_data, max_lag, all_trend_features, all_positive_subsets, all_negative_subsets):
    total_score_positive = 0
    total_score_negative = 0
    for i in range(1, max_lag + 1):
        test_sample = test_data[-i:]
        test_trend = trend_encoding(test_sample)
        scores_positive = []
        scores_negative = []
        for j, trends in enumerate(all_trend_features[:i]):
            for k, trend in enumerate(trends):
                if j >= len(all_positive_subsets) or k >= len(all_positive_subsets[j]):
                    continue
                U = vector_overlap(test_trend, trend)
                positive_subset = all_positive_subsets[j][k]
                negative_subset = all_negative_subsets[j][k]

                # 再次检查子集样本维度
                positive_subset = np.array([s for s in positive_subset if len(s) == i])
                negative_subset = np.array([s for s in negative_subset if len(s) == i])

                num_positive = len(positive_subset)
                num_negative = len(negative_subset)
                if num_positive + num_negative == 0:
                    score_positive_trend = 0
                    score_negative_trend = 0
                else:
                    score_positive_trend = (num_positive + 1) / (num_positive + num_negative + 2)
                    score_negative_trend = (num_negative + 1) / (num_positive + num_negative + 2)

                if num_positive > 0:
                    mu_positive = np.mean(positive_subset, axis=0)
                    cov_positive = np.cov(positive_subset.T)
                    md_positive = mahalanobis_distance(test_sample, mu_positive, cov_positive)
                    score_positive_sample = Dtp(md_positive)
                else:
                    score_positive_sample = 0

                if num_negative > 0:
                    mu_negative = np.mean(negative_subset, axis=0)
                    cov_negative = np.cov(negative_subset.T)
                    md_negative = mahalanobis_distance(test_sample, mu_negative, cov_negative)
                    score_negative_sample = Dtp(md_negative)
                else:
                    score_negative_sample = 0

                score_positive = U * (score_positive_trend + score_positive_sample)
                score_negative = U * (score_negative_trend + score_negative_sample)

                scores_positive.append(score_positive)
                scores_negative.append(score_negative)

                print(f"滞后阶 {i}, 趋势 {j}-{k}: 正趋势得分 {score_positive_trend}, 负趋势得分 {score_negative_trend}, "
                      f"正样本得分 {score_positive_sample}, 负样本得分 {score_negative_sample}, "
                      f"重叠度 {U}, 正总得分 {score_positive}, 负总得分 {score_negative}")

        total_score_positive += np.sum(scores_positive)
        total_score_negative += np.sum(scores_negative)

    max_score = max(total_score_positive, total_score_negative)
    total_score_positive -= max_score
    total_score_negative -= max_score

    probability = np.exp(total_score_positive) / (np.exp(total_score_positive) + np.exp(total_score_negative))
    return probability

# 生成正弦波训练数据和测试数据
t = np.linspace(0, 10 * np.pi, 500)
data = np.sin(t)
train_data = data[:400]
test_data = data[400:]

# 设置固定的 max_lag
fixed_max_lag = 3

# 训练模型
max_lag, all_trend_features, all_positive_subsets, all_negative_subsets = train(train_data, fixed_max_lag)
print(f"最大滞后阶 max_lag: {max_lag}")

# 测试模型
probability = test(test_data, max_lag, all_trend_features, all_positive_subsets, all_negative_subsets)
print(f"预测概率: {probability}")

# 可视化正弦波数据
plt.figure(figsize=(10, 6))
plt.plot(t[:400], train_data, label='Training Data')
plt.plot(t[400:], test_data, label='Testing Data')
plt.title('Sine Wave Data for Training and Testing')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()