import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

np.random.seed(42)

@dataclass
class SimConfig:
    n_days: int = 1000
    train_days: int = 750
    mu_x: float = 0.0
    sigma_x: float = 1.0
    beta: float = 0.7
    alpha: float = 10.0
    ar_rho: float = 0.8  # AR(1) for residual noise (stationary)
    eps_sigma: float = 0.5
    z_entry: float = 1.0
    z_exit: float = 0.0
    txn_cost_bps: float = 5.0  # per leg per trade (enter or exit)


def simulate_cointegrated_pair(cfg: SimConfig) -> pd.DataFrame:
    n = cfg.n_days
    # Simulate X as a random walk
    x_noise = np.random.normal(cfg.mu_x, cfg.sigma_x, size=n)
    X = np.cumsum(x_noise) + 100.0  # start around 100

    # AR(1) stationary residual epsilon
    eps = np.zeros(n)
    eps[0] = np.random.normal(0, cfg.eps_sigma)
    for t in range(1, n):
        eps[t] = cfg.ar_rho * eps[t-1] + np.random.normal(0, cfg.eps_sigma)

    # Y follows cointegration relation: Y = alpha + beta*X + eps
    Y = cfg.alpha + cfg.beta * X + eps

    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    df = pd.DataFrame({"asset1": X, "asset2": Y}, index=dates)
    return df


def formation_stage(prices: pd.DataFrame, cfg: SimConfig):
    train = prices.iloc[:cfg.train_days]
    a1, a2 = train["asset1"].values, train["asset2"].values

    # Cointegration test (Engle-Granger)
    stat, pval, _ = coint(a1, a2)

    # OLS: choose predictor as higher mean close, as per paper convention
    mean1, mean2 = train["asset1"].mean(), train["asset2"].mean()
    if mean1 >= mean2:
        predictor = train["asset1"]
        target = train["asset2"]
        dir_flag = 1  # asset1 is predictor
    else:
        predictor = train["asset2"]
        target = train["asset1"]
        dir_flag = -1  # asset2 is predictor

    X = sm.add_constant(predictor.values)
    model = sm.OLS(target.values, X).fit()
    alpha_hat, beta_hat = model.params[0], model.params[1]

    # Residuals and ADF on residuals
    resid = target.values - (alpha_hat + beta_hat * predictor.values)
    adf_stat, adf_p, _, _, crit_vals, _ = adfuller(resid, autolag='AIC')

    info = {
        "coint_pvalue": float(pval),
        "ols_alpha": float(alpha_hat),
        "ols_beta": float(beta_hat),
        "predictor_asset": "asset1" if dir_flag == 1 else "asset2",
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "adf_crit_1pct": float(crit_vals['1%'])
    }
    return info


def build_signals(prices: pd.DataFrame, cfg: SimConfig, train_mean: float, train_std: float) -> pd.DataFrame:
    df = prices.copy()
    # Ratio as in paper: ratio = asset1 / asset2
    df["ratio"] = df["asset1"] / df["asset2"]
    df["z"] = (df["ratio"] - train_mean) / (train_std if train_std != 0 else 1e-8)

    # Signal rules (ratio z-score):
    # z > +z_entry: short asset1, long asset2
    # z < -z_entry: long asset1, short asset2
    df["sig1"] = 0
    df.loc[df["z"] > cfg.z_entry, "sig1"] = -1
    df.loc[df["z"] < -cfg.z_entry, "sig1"] = 1
    # Complementary signals for asset2
    df["sig2"] = -df["sig1"]

    # Convert to positions with exit rule: when |z| <= z_exit, flat
    in_pos = 0
    sig1_pos = []
    for z, sig in zip(df["z"].values, df["sig1"].values):
        if in_pos == 0:
            if sig != 0:
                in_pos = sig  # enter
        else:
            # if exit condition met
            if abs(z) <= cfg.z_exit:
                in_pos = 0
            else:
                # keep same direction until exit
                pass
        sig1_pos.append(in_pos)
    df["pos1"] = sig1_pos
    df["pos2"] = -df["pos1"]

    # Triggers (entries/exits)
    df["pos1_prev"] = df["pos1"].shift(1).fillna(0)
    df["trade"] = (df["pos1"] != df["pos1_prev"]).astype(int)
    return df.drop(columns=["pos1_prev"]) 


def backtest(prices: pd.DataFrame, signals: pd.DataFrame, beta_hat: float, cfg: SimConfig, start_test_idx: int):
    df = prices.copy()
    df = df.join(signals[["z", "pos1", "pos2", "trade"]])

    # Force flat start at the beginning of the test window to avoid pre-test leakage of positions
    df.loc[: df.index[start_test_idx], "pos1"] = 0
    df["pos2"] = -df["pos1"]
    # Recompute trade flags after enforcing flat start
    pos1_prev = df["pos1"].shift(1).fillna(0)
    df["trade"] = (df["pos1"] != pos1_prev).astype(int)

    # Daily returns from close-to-close price changes
    ret1 = df["asset1"].pct_change().fillna(0)
    ret2 = df["asset2"].pct_change().fillna(0)

    # Position sizing: beta-neutral notionals per pair:
    # leg1 notional = 1.0, leg2 notional = |beta_hat| (to hedge)
    notional1 = 1.0
    notional2 = abs(beta_hat)

    # Pair daily return when in position: pos1 * notional1 * ret1 + pos2 * notional2 * ret2
    pair_ret = df["pos1"] * notional1 * ret1 + df["pos2"] * notional2 * ret2

    # Apply transaction costs on changes in position (both legs). Cost per change event: 2 legs * bps
    cost_per_change = (2 * cfg.txn_cost_bps) / 10000.0
    # Charge cost when trade==1 (entry/exit/flip). For flips, it's counted once per day which approximates two trades.
    costs = df["trade"].astype(float) * cost_per_change

    net_ret = pair_ret - costs

    # Only evaluate in test window
    net_ret.iloc[:start_test_idx] = 0.0

    equity = (1.0 + net_ret).cumprod()
    dd = equity / equity.cummax() - 1.0

    daily_mean = net_ret.iloc[start_test_idx:].mean()
    daily_std = net_ret.iloc[start_test_idx:].std(ddof=0)
    sharpe = (daily_mean / (daily_std + 1e-12)) * np.sqrt(252)

    results = {
        "trades": int(df["trade"].iloc[start_test_idx:].sum()),
        "test_return": float(equity.iloc[-1] / equity.iloc[start_test_idx] - 1.0),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.iloc[start_test_idx:].min()),
    }
    out = pd.DataFrame({
        "equity": equity,
        "net_ret": net_ret,
        "pair_ret": pair_ret,
        "z": df["z"],
        "pos1": df["pos1"],
        "pos2": df["pos2"],
        "trade": df["trade"],
    })
    return results, out


def main():
    cfg = SimConfig()
    prices = simulate_cointegrated_pair(cfg)

    info = formation_stage(prices, cfg)

    # Prepare z-score using training statistics of ratio
    train = prices.iloc[:cfg.train_days]
    train_ratio = (train["asset1"] / train["asset2"]).values
    r_mean, r_std = float(np.mean(train_ratio)), float(np.std(train_ratio))

    signals = build_signals(prices, cfg, r_mean, r_std)

    # Use OLS beta from formation stage
    beta_hat = info["ols_beta"]

    results, bt = backtest(prices, signals, beta_hat, cfg, start_test_idx=cfg.train_days)

    # Print a concise report
    print("=== Formation Stage ===")
    print(f"Cointegration p-value: {info['coint_pvalue']:.4g}")
    print(f"OLS: alpha={info['ols_alpha']:.4f}, beta={beta_hat:.4f}, predictor={info['predictor_asset']}")
    print(f"ADF(residuals): stat={info['adf_stat']:.4f}, p={info['adf_pvalue']:.4g}, crit_1%={info['adf_crit_1pct']:.4f}")
    print("=== Trading (Test Window) ===")
    print(f"Trades: {results['trades']}")
    print(f"Test Return: {results['test_return']*100:.2f}%")
    print(f"Sharpe (daily, 252 ann.): {results['sharpe']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")

    # Save outputs
    bt.to_csv("sim_results_timeseries.csv")
    prices.to_csv("sim_prices.csv")


if __name__ == "__main__":
    main()

