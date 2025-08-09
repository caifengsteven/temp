import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

np.random.seed(123)

# 1) Simulate returns with regime-switching latent expectation driven by a few core moving-average signals
n = 2500
noise_sigma = 0.01

# Core signals that truly matter
core_windows = [10, 40, 120]
ret_white = np.random.randn(n) * 0.2
base_returns = 0.0005 * np.tanh(np.convolve(ret_white, np.ones(25)/25, mode='same')) + np.random.randn(n) * 0.0002

S_core = []
for w in core_windows:
    s = pd.Series(base_returns).rolling(w).mean().fillna(0.0).values
    S_core.append(s)
S_core = np.vstack(S_core).T  # (n, 3)

# Regime weights switch over time to emphasize different cores
w_regimes = [np.array([1.0, 0.5, 0.0]), np.array([0.0, 1.0, 0.8]), np.array([0.6, 0.0, 1.0])]
regime_len = n // 3
w = np.vstack([
    np.tile(w_regimes[0], (regime_len, 1)),
    np.tile(w_regimes[1], (regime_len, 1)),
    np.tile(w_regimes[2], (n - 2*regime_len, 1))
])
latent = (S_core * w).sum(axis=1)
latent = 0.5 * latent  # scale
realized_ret = latent + noise_sigma * np.random.randn(n)

# 2) Build a larger panel of base forecasts: many overlapping windows/filters that are noisy versions of signal
window_set = [5, 8, 10, 12, 16, 20, 30, 40, 60, 80, 100, 120, 140, 160]
base_forecasts = []
for w in window_set:
    mu = pd.Series(realized_ret).rolling(w).mean().shift(1)
    est = mu.fillna(0.0).values + np.random.normal(0, 0.002, size=n)
    base_forecasts.append(est)
for w in [7, 14, 28, 56, 112]:
    mu = pd.Series(realized_ret).ewm(span=w, adjust=False).mean().shift(1)
    est = mu.fillna(0.0).values + np.random.normal(0, 0.002, size=n)
    base_forecasts.append(est)
F = np.vstack(base_forecasts).T  # shape (n, k)

# 3) Three methods: (A) LPCA+LASSO, (B) LASSO on raw forecasts, (C) simple mean
k = F.shape[1]
start = 300
lookback = 600  # rolling training window to adapt to regimes
pca_components = 6

pred_lpca = np.zeros(n)
pred_lasso = np.zeros(n)
pred_mean = pd.Series(F.mean(axis=1)).shift(1).fillna(0.0).values

scaler = StandardScaler()

for t in range(start, n-1):
    i0 = max(0, t - lookback)
    F_hist = F[i0:t]
    y_hist = realized_ret[i0+1:t+1]  # next-return target aligned

    # Standardize forecasts
    F_hist_std = scaler.fit_transform(F_hist)

    # A) PCA then LASSO with TimeSeriesSplit
    pca = PCA(n_components=pca_components)
    PCs_hist = pca.fit_transform(F_hist_std)
    tscv = TimeSeriesSplit(n_splits=5)
    lcv_pca = LassoCV(cv=tscv, fit_intercept=True, max_iter=10000, random_state=123)
    lcv_pca.fit(PCs_hist, y_hist)

    x_t = F[t:t+1]
    x_t_std = scaler.transform(x_t)
    pc_t = pca.transform(x_t_std)
    pred_lpca[t+1] = lcv_pca.predict(pc_t)[0]

    # B) Direct LASSO on raw forecasts (standardized)
    lcv_raw = LassoCV(cv=tscv, fit_intercept=True, max_iter=10000, random_state=123)
    lcv_raw.fit(F_hist_std, y_hist)
    pred_lasso[t+1] = lcv_raw.predict(x_t_std)[0]

# 4) Positioning and PnL
ann_factor = np.sqrt(252)
vol = pd.Series(realized_ret).rolling(50).std().shift(1).fillna(method='bfill')
vol = vol.replace(0, np.nan).fillna(vol.median()).values

scale = 2.0
pos_lpca = np.clip(scale * pred_lpca / (vol + 1e-6), -1.0, 1.0)
pos_lasso = np.clip(scale * pred_lasso / (vol + 1e-6), -1.0, 1.0)
pos_mean = np.clip(scale * pred_mean / (vol + 1e-6), -1.0, 1.0)

pos_lpca = pd.Series(pos_lpca).shift(1).fillna(0.0).values
pos_lasso = pd.Series(pos_lasso).shift(1).fillna(0.0).values
pos_mean = pd.Series(pos_mean).shift(1).fillna(0.0).values

pnl_lpca = pos_lpca * realized_ret
pnl_lasso = pos_lasso * realized_ret
pnl_mean = pos_mean * realized_ret

cum_lpca = np.cumsum(pnl_lpca)
cum_lasso = np.cumsum(pnl_lasso)
cum_mean = np.cumsum(pnl_mean)

def metrics(pnl):
    sharpe = np.mean(pnl) / (np.std(pnl) + 1e-12) * ann_factor
    eq = np.cumsum(pnl)
    peak = eq[0]
    max_dd = 0
    for v in eq:
        if v > peak:
            peak = v
        dd = v - peak
        if dd < max_dd:
            max_dd = dd
    return sharpe, eq[-1], max_dd

sm_lpca = metrics(pnl_lpca)
sm_lasso = metrics(pnl_lasso)
sm_mean = metrics(pnl_mean)

print('Simulated regime-switching LPCA vs baselines')
print(f'Bars={n}, Forecasts={k}, PCA comps={pca_components}, Lookback={lookback}')
print('LPCA+LASSO: Sharpe={:.2f}, TotalPnL={:.3f}, MaxDD={:.3f}'.format(*sm_lpca))
print('Raw LASSO:  Sharpe={:.2f}, TotalPnL={:.3f}, MaxDD={:.3f}'.format(*sm_lasso))
print('Mean pool:  Sharpe={:.2f}, TotalPnL={:.3f}, MaxDD={:.3f}'.format(*sm_mean))

print('Tail of equity curves:')
print(pd.DataFrame({'lpca': cum_lpca, 'lasso': cum_lasso, 'mean': cum_mean}).tail(5))

