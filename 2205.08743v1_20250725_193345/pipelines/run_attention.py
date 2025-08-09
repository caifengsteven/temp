import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from data.loader import load_price_csvs, compute_returns
from models.hmm import RegimeHMM, HMMConfig
from opt.mean_variance import mv_weights
from backtest.simulator import simulate_backtest
from attention.controller import attention_series
from attention.filtering import temper_posteriors


def belief_weighted_params(mu_list, cov_list, post):
    mu_hat = (post @ mu_list).reshape(-1)
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
    # attention params
    ap.add_argument('--a_max', type=float, default=1.0)
    ap.add_argument('--k1', type=float, default=1.0)
    ap.add_argument('--k2', type=float, default=1.0)
    ap.add_argument('--c2_bps', type=float, default=0.05, help='quadratic attention cost in bps per step')
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

    post_full = hmm.filter_posteriors(rets)

    # Attention control from posteriors
    a_t = attention_series(post_full, k1=args.k1, a_max=args.a_max)
    post_tempered = temper_posteriors(post_full, a_t, k2=args.k2)

    # Attention cost series (quadratic only here)
    attn_cost = (args.c2_bps * 1e-4) * (a_t ** 2)

    # Rebalance each timestamp with tempered beliefs
    weights = []
    for t in rets.index:
        post = post_tempered.loc[t].values
        mu_hat, cov_hat = belief_weighted_params(mu_list, cov_list, post)
        w = mv_weights(mu_hat, cov_hat, risk_aversion=args.risk_aversion)
        weights.append(pd.Series(w, index=rets.columns, name=t))
    weights = pd.DataFrame(weights)

    port_rets, eq_curve, summary = simulate_backtest(rets, weights, tx_cost_bps=args.tx_cost_bps, attn_cost=attn_cost)

    print("Summary (with attention):", summary)
    outdir = Path('outputs')
    outdir.mkdir(exist_ok=True)
    weights.to_csv(outdir / 'attention_weights.csv')
    port_rets.to_csv(outdir / 'attention_port_returns.csv')
    eq_curve.to_csv(outdir / 'attention_equity_curve.csv')
    a_t.to_csv(outdir / 'attention_series.csv')


if __name__ == '__main__':
    main()

