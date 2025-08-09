Baseline HMM Mean–Variance Portfolio (with Synthetic Data)

Overview
- Implements a baseline regime-aware mean–variance strategy:
  - Fit a univariate Gaussian HMM on an aggregate (equal-weight) return.
  - Compute forward (filter) regime probabilities out-of-sample.
  - Estimate per-regime asset return mean/covariance from the training period.
  - Form belief-weighted mean/covariance and compute long-only, fully-invested weights.
  - Backtest with transaction costs and weekly rebalancing.
- Includes a synthetic data generator so you can run it offline without downloads.

Quickstart (synthetic demo)
1) Generate synthetic ETF prices:
   python -m data.synthetic.generate_synth --out data/synth --tickers SPY QQQ IWM EFA EEM TLT LQD HYG GLD IEF --seed 42

2) Run the baseline pipeline on synthetic CSVs:
   python -m pipelines.run_baseline --data_dir data/synth --train_ratio 0.7 --n_regimes 3 --rebalance W-FRI --tx_cost_bps 2 --risk_aversion 5.0

Notes
- CSV format expected: columns [Date, Adj Close], one file per ticker named <TICKER>.csv.
- If you have real ETF data in the same format, point --data_dir to that folder.
- Requirements already present here: numpy, pandas, hmmlearn.

File structure
- data/loader.py               CSV reading and returns computation
- data/synthetic/generate_synth.py  Synthetic price generator
- models/hmm.py               HMM fit + online forward filter + regime stats
- opt/mean_variance.py        Mean–variance weights with simplex projection
- backtest/simulator.py       Rebalance simulation and metrics
- pipelines/run_baseline.py   End-to-end run script

Next
- After validating baseline, we’ll add dynamic attention as a control that modulates observation noise at a cost, and compare.

