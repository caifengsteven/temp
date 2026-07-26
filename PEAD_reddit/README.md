# PEAD Asymmetry Replication Pipeline

Testing the Reddit claim: *"Large earnings misses drift 4.5× more than large beats at T+20."*

## Quick Start

```bash
# Install
uv pip install -e ".[dev]"

# Run on synthetic data (validates pipeline mechanics)
python scripts/run_replication.py --synthetic --tickers 200 --quarters 40

# Run on Bloomberg data (production)
python scripts/run_replication.py --bloomberg --tickers-file tickers.txt \
    --start 1995-01-01 --end 2024-12-31 --output data/processed/
```

## What This Pipeline Tests

The Reddit claim makes three assertions:
1. PEAD is asymmetric (misses drift more than beats) — **direction**
2. The ratio is 4.5× at T+20 — **magnitude**
3. This contradicts the classic symmetric-drift assumption — **novelty**

This pipeline tests all three with proper methodology:

| Test | What It Does | Why It Matters |
|---|---|---|
| Raw asymmetry ratio | \|miss CAR\| / \|beat CAR\| at T+20 | Headline number |
| Bootstrap 95% CI | 10,000 resample iterations | Without CI, 4.5× is meaningless |
| Clustered difference test | Firm-clustered t-test on miss - beat | Formal hypothesis test |
| Mid-quote adjusted | CAR from (bid+ask)/2 returns | Removes bid-ask bounce (Zhang et al. 2024) |
| Transaction cost adjusted | Commission + Amihud slippage + borrow | Chordia et al.: costs eat 70-100% of PEAD |
| Liquidity double-sort | Ratio within Amihud buckets | Controls for composition effect |
| Size double-sort | Ratio within market-cap buckets | Bartov et al.: institutional ownership drives PEAD |
| SUE method comparison | SUE1 vs SUE3 | Livnat-Mendenhall: method changes drift magnitude |
| Sub-period analysis | Pre/post Reg-FD, post-2010 | PEAD has decayed over time |

## Architecture

```
src/pead/
├── schema.py              Column name constants + validation (shared contract)
├── synthetic.py           Synthetic data generator (known 2.0× asymmetry injected)
├── pipeline.py            End-to-end orchestrator
├── ingest/
│   ├── bloomberg.py       BQL/BLPAPI queries (lazy import; mock fallback)
│   └── factors.py         FF5 + momentum factor loading
├── sue/
│   ├── time_series.py     SUE1 (random walk) + SUE2 (ex-special items)
│   ├── analyst.py         SUE3 (analyst-median forecast)
│   └── validate.py        Livnat-Mendenhall filters + decile assignment
├── events/
│   ├── calendar.py        Trading day snapping + offsets
│   └── returns.py         Market model + CAR/BHAR + mid-quote returns
├── portfolios/
│   ├── sort.py            Per-cross-section decile formation
│   ├── calendar_time.py   Calendar-time portfolio assembly
│   └── costs.py           Amihud-scaled transaction costs + borrow
├── benchmarks/
│   ├── ff5.py             Fama-French 5-factor regression (Newey-West HAC)
│   └── dgtw.py            DGTW 5×5×5 characteristic-matched benchmarks
└── asymmetry/
    ├── ratio.py           Miss/beat ratio computation
    ├── double_sort.py     Liquidity + size conditional analysis
    └── inference.py       Bootstrap CIs + clustered t-tests
```

## Data Requirements (Bloomberg Mode)

| Dataset | Bloomberg Fields | Purpose |
|---|---|---|
| Analyst estimates | `BEST_EPS_EST_MED`, `BEST_EPS_EST_MEAN`, `BEST_EPS_NEST`, `EE_API` | SUE3 construction |
| Quarterly fundamentals | `FA_AMP_EPS`, `FA_DILUTED_EPS`, `FA_SPL_ITEM`, `FA_BP_BAS_EV` | SUE1/SUE2 construction |
| Daily prices | `PX_LAST`, `PX_BID`, `PX_ASK`, `PX_VOLUME`, `DAY_TO_DAY_TOT_RETURN_GROSS_DVDS` | Event returns + Amihud |
| Delisting events | Delisting date + return | Short-side survivorship bias correction |
| Factor returns | `FF_MKT_RF`, `FF_SMB`, `FF_HML`, `FF_RMW`, `FF_CMA` or Kenneth French library | FF5 risk adjustment |

## Methodology References

| Paper | Contribution to This Pipeline |
|---|---|
| Livnat & Mendenhall (2006) | SUE1/SUE2/SUE3 construction; I/B/E/S vs Compustat drift comparison |
| Bernard & Thomas (1989, 1990) | Decile portfolio formation; AR(1) expectations model |
| MacKinlay (1997) | Market model estimation window [−255, −46] |
| Lyon, Barber, Tsai (1999) | Calendar-time portfolio approach; bootstrap inference |
| Mitchell & Stafford (2000) | BHAR cross-correlation problem; CAR preferred for inference |
| Bartov, Radhakrishnan & Krinsky (2000) | Institutional ownership explains PEAD; size double-sort |
| Mendenhall (2004) | Arbitrage risk (idiosyncratic vol) drives PEAD |
| Narayanamoorthy (2006) | Accounting conservatism creates SUE autocorrelation asymmetry |
| Chordia, Goyal, Sadka et al. (2009) | Transaction costs consume 70-100% of PEAD in illiquid names |
| Korczak, Korczak & Pacelli (2013) | CTP + costs → no PEAD alpha |
| Bird, Choi & Yeung (2011) | Bad-news PEAD > good-news PEAD under high uncertainty |
| Zhang, Gregoriou & Wu (2024) | Bid-ask bounce inflates asymmetry; UK evidence ~2.4× |

## Testing

```bash
python -m pytest tests/ -v          # 58 tests across all modules
python -m pytest tests/test_sue.py  # SUE formula verification
```

The synthetic data generator injects a known 2.0× asymmetry (beats: +50bps, misses: −100bps over [1,20]). The `test_recovers_injected_asymmetry` test verifies the pipeline detects this signal.

## Configuration

All parameters are in `config/config.yaml`. Key sections:

- `sue:` — SUE deflator, analyst window, special-items tax adjustment
- `event_study:` — Estimation window [−255, −46], event windows, mid-quote toggle
- `portfolios:` — Decile count, weight scheme, cross-section sort groups
- `transaction_costs:` — Commission, slippage, Amihud scaling, borrow cost
- `asymmetry:` — Bootstrap samples, confidence level, clustering dimensions
- `subperiods:` — Reg-FD split (Oct 2000), post-2010 split

## Expected Findings (Based on Literature)

When run on real US equity data (1995–2024, I/B/E/S universe):

| Adjustment | Expected Ratio | Source |
|---|---|---|
| Raw CAR[1,20] | 1.5–3.0× | Bird et al., Zhang et al. |
| Mid-quote adjusted | Lower than raw | Zhang et al. 2024 |
| After transaction costs | 1.0–2.0× | Chordia et al. 2009 |
| Within liquid bucket | Closer to 1.0× | Composition effect |
| Post-2010 sub-period | Lower than pre-2000 | PEAD decay |

If the raw ratio exceeds 3.5×, check for:
- Survivorship bias (missing delisting returns on short side)
- Equal-weighting (overweights small/illiquid names)
- No transaction costs
- Event-time CAR (overstates vs calendar-time portfolio)
- Sign-based classification instead of SUE deciles
