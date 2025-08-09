from pathlib import Path
import numpy as np
import pandas as pd
import argparse


def gen_gbm(T=1500, mu=0.08, sigma=0.18, dt=1/252, s0=100.0, seed=None):
    rng = np.random.default_rng(seed)
    n = T
    z = rng.standard_normal(n)
    x = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z
    logp = np.r_[np.log(s0), np.log(s0) + np.cumsum(x)]
    return np.exp(logp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--tickers', nargs='+', required=True)
    ap.add_argument('--T', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    # Create correlated series by mixing common factor and idiosyncratic
    dates = pd.bdate_range('2010-01-01', periods=args.T)
    common = gen_gbm(T=args.T, mu=0.06, sigma=0.15, s0=100, seed=rng.integers(1e9))
    df = pd.DataFrame(index=dates)
    for t in args.tickers:
        mu = 0.04 + 0.06*rng.random()
        sig = 0.10 + 0.20*rng.random()
        mix = 0.5 + 0.4*rng.random()
        idio = gen_gbm(T=args.T, mu=mu, sigma=sig, s0=100, seed=rng.integers(1e9))
        price = mix*common[:args.T+1] + (1-mix)*idio[:args.T+1]
        # normalize to start ~100
        price = (price/price[0])*100
        tmp = pd.DataFrame({'Date': dates, 'Adj Close': price[1:]})
        tmp.to_csv(out / f"{t}.csv", index=False)


if __name__ == '__main__':
    main()

