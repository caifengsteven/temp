# US Recession Probability from Cross-Asset Markets

This repository implements two versions of a framework to estimate US recession probabilities from multiple asset classes:

- Minimal v1 Ensemble: interpretable, robust features, penalized logistic + gradient boosting with calibration and static/slowly updated weights
- Dynamic/TVP Ensemble: adaptive models using rolling/online learning and dynamic model averaging across submodels and horizons

It produces monthly probabilities for 3, 6, and 12-month horizons.

## Quick Start

1) Create a Python 3.10+ environment
2) Install dependencies

```
pip install -r requirements.txt
```

3) (Optional) Set a FRED API key in your environment to improve reliability:
- Windows PowerShell: `$env:FRED_API_KEY = "YOUR_KEY"`
- Bash: `export FRED_API_KEY=YOUR_KEY`

4) Run the minimal v1 pipeline (downloads data from FRED only):
```
python src/run_minimal.py
```

5) Run the dynamic/TVP pipeline:
```
python src/run_dynamic.py
```

Outputs (CSV and PNG charts) will be saved to `outputs/`.

## Data Sources

To avoid licensing issues, v1 uses freely available FRED series (Treasuries, S&P 500 index, USD broad dollar, FX spot rates, WTI, NBER recession series). Some optional series (e.g., credit OAS) may fail to fetch if symbols are retired—these are treated as optional.

If you have Bloomberg/Refinitiv access or your own CSVs, drop them into `data/external/` and update `config.yaml` to point to them. The loader will prefer local files over FRED when present.

## Horizons and Labels

- Horizons: 3, 6, 12 months ahead
- Label definition: 1 if any month in the next H months is an NBER recession month (inclusive), else 0. Configure in `src/labels.py`.

## Notes

- This code is designed for monthly endpoints (month-end). Daily series are converted to month-end using last available value.
- The dynamic pipeline adapts model weights using recent log-loss (Brier) performance and uses online-learning logistic models to allow time-varying coefficients.

## Disclaimer

This is a research framework and not investment advice. Use at your own risk.

