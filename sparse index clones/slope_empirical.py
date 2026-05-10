"""
slope_empirical.py
Rolling window empirical analysis for index tracking and hedge fund replication.
Based on Section 4 of Kremer et al. (2022).

Framework:
- Rolling window estimation with in-sample window tau
- Out-of-sample tracking error computed at each rebalance
- Strategies: SLOPE, SLOPE-LO, SLOPE-SLC, SLOPE-LO-SLC, LASSO
- Equity indices: SP100/SP200/SP500 constituents (daily, tau=750, monthly rebalance)
- Hedge funds: HFR indices (monthly, tau=60, 17 risk factors)

Data sources (user must provide or use yfinance proxies):
- Equity: daily returns of S&P constituents (CSV or yfinance)
- Hedge funds: monthly HFR index returns + 17 factor returns (CSV)
"""

import numpy as np
import pandas as pd
from slope_solver import (solve_slope_fista, solve_lasso_pg,
                          compute_lambda_sequence_bogdan,
                          project_simplex, project_box_hyperplane)
import warnings


def slope_slc_strategy(R, Y, w, percentile=75.0, constraint='simplex'):
    """
    SLOPE-SLC (Select) strategy: keep only top groups by median partial correlation.

    Following the paper:
    - Identify groups by unique non-zero coefficient values
    - For each group, compute median partial correlation with the index
    - Keep groups with median partial correlation >= percentile threshold
    - Rescale weights to satisfy budget constraint

    Parameters
    ----------
    R : ndarray, shape (T, K)
    Y : ndarray, shape (T,)
    w : ndarray, shape (K,), SLOPE weights
    percentile : float, threshold (75 for equity, 25 for hedge funds in paper)
    constraint : str, 'simplex' or 'box_hyperplane'

    Returns
    -------
    w_selected : ndarray, shape (K,)
    """
    K = len(w)
    abs_w = np.abs(w)
    nz_idx = np.where(abs_w > 1e-10)[0]

    if len(nz_idx) == 0:
        # Fallback: equal weight
        if constraint == 'simplex':
            return np.ones(K) / K
        else:
            return np.ones(K) / K

    # Identify groups by rounding to 6 decimal places
    nz_vals = w[nz_idx]
    unique_vals = np.unique(np.round(nz_vals, 6))

    group_medians = []
    group_masks = []

    for val in unique_vals:
        mask = np.abs(np.round(w, 6) - val) < 1e-10
        group_idx = np.where(mask)[0]

        # Partial correlation: correlation of each asset with residual
        # after removing other groups
        other_idx = np.setdiff1d(np.arange(K), group_idx)
        if len(other_idx) > 0:
            r_P = Y - R[:, other_idx] @ w[other_idx]
        else:
            r_P = Y

        pcs = []
        for j in group_idx:
            # Centered partial correlation
            xj = R[:, j] - np.mean(R[:, j])
            rp = r_P - np.mean(r_P)
            denom = np.linalg.norm(xj) * np.linalg.norm(rp)
            if denom > 1e-14:
                pc = np.dot(xj, rp) / denom
                pcs.append(pc)

        median_pc = np.median(np.abs(pcs)) if len(pcs) > 0 else 0.0
        group_medians.append(median_pc)
        group_masks.append(mask)

    # Select groups above percentile threshold
    threshold_val = np.percentile(group_medians, percentile)

    w_selected = np.zeros(K)
    for median_pc, mask in zip(group_medians, group_masks):
        if median_pc >= threshold_val:
            w_selected[mask] = w[mask]

    # Re-normalize to satisfy budget constraint
    if constraint == 'simplex':
        s = np.sum(w_selected)
        if s > 1e-10:
            w_selected = w_selected / s
        else:
            w_selected = np.ones(K) / K
    elif constraint == 'box_hyperplane':
        s = np.sum(w_selected)
        if np.abs(s - 1.0) > 1e-6 or np.any(w_selected < -1.0) or np.any(w_selected > 1.0):
            w_selected = project_box_hyperplane(w_selected, a=-1.0, b=1.0, c=1.0)

    return w_selected


def rolling_window_backtest(R, Y, window_size, step_size, lambda_seq,
                            strategy='slope', constraint='simplex',
                            threshold=0.0005, percentile=75.0,
                            warm_start=True, verbose=False):
    """
    Rolling window backtest for index tracking.

    Parameters
    ----------
    R : ndarray, shape (T, K)  or DataFrame
    Y : ndarray, shape (T,)    or Series
    window_size : int, in-sample estimation window
    step_size : int, rebalancing frequency (e.g. 21 for monthly)
    lambda_seq : ndarray, shape (K,), SLOPE penalty sequence
    strategy : str
        'slope'     : standard SLOPE
        'slope-slc' : SLOPE with group selection
        'slope-lo'  : SLOPE long-only (same as slope + simplex constraint)
        'lasso'     : LASSO baseline
    constraint : str, 'simplex' or 'box_hyperplane'
    threshold : float, weights |w| < threshold set to zero (paper uses 0.0005)
    percentile : float, for SLC strategy
    warm_start : bool, use previous weights as initial guess
    verbose : bool

    Returns
    -------
    results : dict
    """
    if isinstance(R, pd.DataFrame):
        R = R.values
    if isinstance(Y, (pd.Series, pd.DataFrame)):
        Y = Y.values.flatten()

    T, K = R.shape
    n_steps = (T - window_size) // step_size

    oos_returns = []      # Y_out - R_out @ w
    oos_te_sq = []        # squared tracking errors
    weights_hist = []
    sparsity_hist = []
    turnover_hist = []
    groups_hist = []

    w_prev = None

    for t in range(n_steps):
        start = t * step_size
        end = start + window_size
        oos_end = min(end + step_size, T)

        R_in = R[start:end]
        Y_in = Y[start:end]
        R_out = R[end:oos_end]
        Y_out = Y[end:oos_end]

        # Determine solver and lambda
        if strategy.lower() in ('slope', 'slope-lo', 'slope-slc', 'slope-lo-slc'):
            w, info = solve_slope_fista(
                R_in, Y_in, lambda_seq,
                w0=w_prev if warm_start else None,
                constraint=constraint,
                max_iter=5000, tol=1e-7, verbose=False
            )
        elif strategy.lower() == 'lasso':
            lambda_lasso = lambda_seq[0] if len(lambda_seq) > 0 else 0.01
            w, info = solve_lasso_pg(
                R_in, Y_in, lambda_lasso,
                w0=w_prev if warm_start else None,
                constraint=constraint,
                max_iter=5000, tol=1e-7, verbose=False
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Apply threshold
        w[np.abs(w) < threshold] = 0.0

        # SLC post-processing
        if 'slc' in strategy.lower():
            w = slope_slc_strategy(R_in, Y_in, w,
                                   percentile=percentile,
                                   constraint=constraint)

        # Turnover
        if w_prev is not None:
            to = np.sum(np.abs(w - w_prev))
        else:
            to = 0.0
        turnover_hist.append(to)

        # Out-of-sample tracking
        if len(Y_out) > 0:
            diff = Y_out - R_out @ w
            oos_returns.extend(diff.tolist())
            oos_te_sq.extend((diff ** 2).tolist())

        # Diagnostics
        n_nz = int(np.sum(np.abs(w) > threshold))
        n_gr = len(np.unique(np.round(w[np.abs(w) > threshold], 6)))

        weights_hist.append(w.copy())
        sparsity_hist.append(n_nz)
        groups_hist.append(n_gr)
        w_prev = w.copy()

        if verbose and (t + 1) % 10 == 0:
            te_mean = np.sqrt(np.mean(oos_te_sq)) if len(oos_te_sq) > 0 else np.nan
            print(f"  Step {t+1}/{n_steps}: sparsity={n_nz}, groups={n_gr}, "
                  f"RMSE={te_mean:.4f}")

    oos_returns = np.array(oos_returns)
    oos_te_sq = np.array(oos_te_sq)

    # Annualized metrics (assume 252 trading days for daily, 12 for monthly)
    freq = 252 if step_size >= 20 else 12

    return {
        'oos_returns': oos_returns,
        'oos_te_sq': oos_te_sq,
        'weights': np.array(weights_hist),
        'sparsity': np.array(sparsity_hist),
        'groups': np.array(groups_hist),
        'turnover': np.array(turnover_hist),
        'n_steps': n_steps,
        'mean_te': np.mean(oos_returns) if len(oos_returns) > 0 else np.nan,
        'te_vol': np.std(oos_returns, ddof=1) * np.sqrt(freq) if len(oos_returns) > 1 else np.nan,
        'rmse': np.sqrt(np.mean(oos_te_sq)) if len(oos_te_sq) > 0 else np.nan,
        'mean_sparsity': np.mean(sparsity_hist),
        'mean_turnover': np.mean(turnover_hist),
        'mean_groups': np.mean(groups_hist),
    }


def run_equity_index_tracking(prices_df, index_series, lambda_alpha=0.01,
                              window_days=750, rebalance_days=21,
                              strategies=None):
    """
    Run equity index tracking backtest.

    Parameters
    ----------
    prices_df : DataFrame, shape (T, K), constituent prices or returns
    index_series : Series, shape (T,), index prices or returns
    lambda_alpha : float, scale for lambda sequence
    window_days : int, estimation window
    rebalance_days : int, rebalancing frequency
    strategies : list of str or None

    Returns
    -------
    summary : DataFrame
    """
    if strategies is None:
        strategies = ['slope', 'slope-slc', 'lasso']

    # Compute returns if prices provided
    if prices_df.max().max() > 0.5:  # heuristic: prices vs returns
        R = prices_df.pct_change().dropna().values
        Y = index_series.pct_change().dropna().values.flatten()
        # Align
        common_idx = prices_df.pct_change().dropna().index.intersection(
            index_series.pct_change().dropna().index)
        R = prices_df.pct_change().loc[common_idx].values
        Y = index_series.pct_change().loc[common_idx].values.flatten()
    else:
        R = prices_df.values
        Y = index_series.values.flatten()

    T, K = R.shape
    lambda_seq = compute_lambda_sequence_bogdan(K, lambda_alpha, theta=0.1)

    results = {}
    for strat in strategies:
        print(f"\nRunning strategy: {strat}")
        res = rolling_window_backtest(
            R, Y, window_size=window_days, step_size=rebalance_days,
            lambda_seq=lambda_seq, strategy=strat,
            constraint='simplex', percentile=75.0,
            verbose=True
        )
        results[strat] = res

    # Summary table
    rows = []
    for strat, res in results.items():
        rows.append({
            'Strategy': strat,
            'Mean_TE': res['mean_te'],
            'TE_Vol': res['te_vol'],
            'RMSE': res['rmse'],
            'Mean_Sparsity': res['mean_sparsity'],
            'Mean_Groups': res['mean_groups'],
            'Mean_Turnover': res['mean_turnover']
        })

    summary = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("EQUITY INDEX TRACKING SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    return summary, results


def run_hedge_fund_replication(hf_returns, factor_returns, lambda_alpha=0.01,
                               window_months=60, rebalance_months=1,
                               strategies=None):
    """
    Run hedge fund replication backtest.

    Parameters
    ----------
    hf_returns : DataFrame or Series, shape (T,), hedge fund index returns
    factor_returns : DataFrame, shape (T, K), factor returns
    lambda_alpha : float
    window_months : int
    rebalance_months : int
    strategies : list of str or None

    Returns
    -------
    summary : DataFrame
    """
    if strategies is None:
        strategies = ['slope', 'slope-lo', 'slope-slc', 'slope-lo-slc', 'lasso']

    if isinstance(hf_returns, pd.Series):
        hf_returns = hf_returns.to_frame()

    # Align
    common_idx = hf_returns.index.intersection(factor_returns.index)
    Y = hf_returns.loc[common_idx].values.flatten()
    R = factor_returns.loc[common_idx].values

    T, K = R.shape
    lambda_seq = compute_lambda_sequence_bogdan(K, lambda_alpha, theta=0.1)

    results = {}
    for strat in strategies:
        print(f"\nRunning strategy: {strat}")
        constraint = 'simplex' if 'lo' in strat.lower() else 'box_hyperplane'
        percentile = 25.0
        res = rolling_window_backtest(
            R, Y, window_size=window_months, step_size=rebalance_months,
            lambda_seq=lambda_seq, strategy=strat,
            constraint=constraint, percentile=percentile,
            verbose=True
        )
        results[strat] = res

    rows = []
    for strat, res in results.items():
        rows.append({
            'Strategy': strat,
            'Mean_TE': res['mean_te'],
            'TE_Vol': res['te_vol'],
            'RMSE': res['rmse'],
            'Mean_Sparsity': res['mean_sparsity'],
            'Mean_Groups': res['mean_groups'],
            'Mean_Turnover': res['mean_turnover']
        })

    summary = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("HEDGE FUND REPLICATION SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    return summary, results


if __name__ == '__main__':
    print("slope_empirical.py: import this module and call")
    print("  run_equity_index_tracking(prices_df, index_series)")
    print("  run_hedge_fund_replication(hf_returns, factor_returns)")
