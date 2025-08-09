import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from data.loader import load_price_csvs, compute_returns
from models.hmm import RegimeHMM, HMMConfig
from opt.mean_variance import mv_weights
from backtest.simulator import simulate_backtest


def belief_weighted_params(mu_list, cov_list, post):
    # Blend per-regime stats by current posterior belief
    mu_hat = (post @ mu_list).reshape(-1)
    # law of total covariance: E[Cov]+Cov[E]
    cov_hat = np.einsum('k, kij -> ij', post, cov_list)
    mu_diff = (mu_list - mu_hat)
    cov_of_mu = (post[:, None, None] * (mu_diff[:, :, None] * mu_diff[:, None, :])).sum(axis=0)
    return mu_hat, cov_hat + cov_of_mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--tickers', nargs='+', default=['SPY','QQQ','IWM','EFA','EEM','TLT','LQD','HYG','GLD','IEF'])
    ap.add_argument('--train_ratio', type=float, default=0.7)
    ap.add_argument('--freq', type=str, default='W-FRI')
    ap.add_argument('--n_regimes', type=int, default=3)
    ap.add_argument('--risk_aversion', type=float, default=5.0)
    ap.add_argument('--tx_cost_bps', type=float, default=2.0)
    args = ap.parse_args()

    px = load_price_csvs(args.data_dir, args.tickers)
    rets = compute_returns(px, freq=args.freq)

    T = len(rets)
    split = int(T * args.train_ratio)
    train, test = rets.iloc[:split], rets.iloc[split:]

    hmm = RegimeHMM(HMMConfig(n_regimes=args.n_regimes))
    hmm.fit_on_agg(train)
    post_train = hmm.filter_posteriors(train)
    mu_list, cov_list = RegimeHMM.per_regime_stats(train, post_train)

    # Online filtering on full series (train+test) for walk-forward beliefs
    post_full = hmm.filter_posteriors(rets)

    # Rebalance at each timestamp using belief-weighted params
    weights = []
    for t in rets.index:
        post = post_full.loc[t].values  # shape (K,)
        mu_hat, cov_hat = belief_weighted_params(mu_list, cov_list, post)
        w = mv_weights(mu_hat, cov_hat, risk_aversion=args.risk_aversion)
        weights.append(pd.Series(w, index=rets.columns, name=t))
    weights = pd.DataFrame(weights)

    port_rets, eq_curve, summary = simulate_backtest(rets, weights, tx_cost_bps=args.tx_cost_bps)

    print("Summary:", summary)
    # Save outputs
    outdir = Path('outputs')
    outdir.mkdir(exist_ok=True)
    weights.to_csv(outdir / 'baseline_weights.csv')
    port_rets.to_csv(outdir / 'baseline_port_returns.csv')
    eq_curve.to_csv(outdir / 'baseline_equity_curve.csv')


if __name__ == '__main__':
    main()

