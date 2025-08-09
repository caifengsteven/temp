import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# 1) Simulate a price series with stochastic returns and latent factor that many forecasters partially capture
n = 2000  # bars
# latent expected return process (slowly varying)
latent = 0.0005 * np.tanh(np.convolve(np.random.randn(n) * 0.2, np.ones(25)/25, mode='same'))
realized_ret = latent + 0.01 * np.random.randn(n)
price = 100 * np.cumprod(1 + realized_ret)

# 2) Build a panel of base forecasts via noisy, windowed estimates of the latent mean
window_set = [10, 20, 40, 80, 120, 160]
base_forecasts = []
for w in window_set:
    # rolling mean of recent returns + noise; lag by 1 to avoid lookahead
    mu = pd.Series(realized_ret).rolling(w).mean().shift(1)
    est = mu.fillna(0.0).values + np.random.normal(0, 0.002, size=n)
    base_forecasts.append(est)
# add variants to increase dimensionality
for w in [15, 30, 60, 100, 140]:
    mu = pd.Series(realized_ret).ewm(span=w, adjust=False).mean().shift(1)
    est = mu.fillna(0.0).values + np.random.normal(0, 0.002, size=n)
    base_forecasts.append(est)
F = np.vstack(base_forecasts).T  # shape (n, k)

# 3) LPCA pipeline in a walk-forward fashion
k = F.shape[1]
scaler = StandardScaler(with_mean=True, with_std=True)

start = 200  # burn-in for initial models
pca_components = 5

pred_ret = np.zeros(n)
coefs_history = []

for t in range(start, n-1):
    # Use information up to t-1 to train
    F_hist = F[:t]
    y_hist = realized_ret[1:t+1]  # align next-return as target

    # standardize forecasts
    F_hist_std = scaler.fit_transform(F_hist)

    # PCA
    pca = PCA(n_components=pca_components)
    PCs_hist = pca.fit_transform(F_hist_std)

    # LASSO with simple CV (time-series CV would be preferable)
    # To keep compatibility with sklearn 0.24.2, use LassoCV with default KFold
    # Note: In real time-series, use TimeSeriesSplit
    lcv = LassoCV(alphas=None, cv=5, fit_intercept=True, max_iter=5000, random_state=42)
    lcv.fit(PCs_hist, y_hist)

    # One-step-ahead prediction
    x_t = F[t:t+1]
    x_t_std = scaler.transform(x_t)
    pc_t = pca.transform(x_t_std)
    pred = lcv.predict(pc_t)[0]
    pred_ret[t+1] = pred

    if t % 200 == 0:
        coefs_history.append(lcv.coef_.copy())

# 4) Trading rule: continuous position proportional to predicted return, with cap
vol = pd.Series(realized_ret).rolling(50).std().shift(1).fillna(method='bfill').values
vol[vol == 0] = np.nan
vol = pd.Series(vol).fillna(np.nanmedian(vol)).values

risk_target = 0.1  # target annualized vol not used directly; use simple scaling here
scale = 5.0  # scale predictions to positions
position = np.clip(scale * pred_ret / (vol + 1e-6), -1.0, 1.0)
position = pd.Series(position).shift(1).fillna(0.0).values  # trade at next bar

# 5) PnL, metrics
pnl = position * realized_ret
cum_pnl = np.cumsum(pnl)
ann_factor = np.sqrt(252)  # if bars are daily; for intraday adjust
sharpe = np.mean(pnl) / (np.std(pnl) + 1e-12) * ann_factor

max_dd = 0
peak = -1e9
equity = cum_pnl
peak = equity[0]
for v in equity:
    peak = max(peak, v)
    max_dd = min(max_dd, v - peak)

print('Simulated LPCA strategy results:')
print(f'Bars: {n}, Forecasts: {k}, PCA components: {pca_components}')
print(f'Sharpe (ann.): {sharpe:.2f}')
print(f'Total PnL: {cum_pnl[-1]:.3f}')
print(f'Max Drawdown: {max_dd:.3f}')

# Compare to naive mean-forecast strategy
mean_forecast = F.mean(axis=1)
mean_pred = pd.Series(mean_forecast).shift(1).fillna(0.0).values
pos_mean = np.clip(scale * mean_pred / (vol + 1e-6), -1.0, 1.0)
pos_mean = pd.Series(pos_mean).shift(1).fillna(0.0).values
pnl_mean = pos_mean * realized_ret
cum_pnl_mean = np.cumsum(pnl_mean)
sharpe_mean = np.mean(pnl_mean) / (np.std(pnl_mean) + 1e-12) * ann_factor

print('Naive mean-forecast strategy:')
print(f'Sharpe (ann.): {sharpe_mean:.2f}, Total PnL: {cum_pnl_mean[-1]:.3f}')

# Output a small sample of equity curves
out = pd.DataFrame({
    'equity_lpca': cum_pnl,
    'equity_mean': cum_pnl_mean
})
print('Tail of equity curves:')
print(out.tail(5))

