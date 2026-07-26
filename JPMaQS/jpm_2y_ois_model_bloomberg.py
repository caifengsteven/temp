#!/usr/bin/env python
"""
J.P. Morgan Rates Strategy - 2Y OIS Fair Value Model (Bloomberg Edition)
=======================================================================

Replicates the cross-market 2Y OIS fair value model from the J.P. Morgan
Global Markets Strategy note "Makes much more sense to live in the present
tense" (22 July 2026), Figure 4.

This is the BLOOMBERG edition. It differs from `jpm_2y_ois_model.py`
(the Kaggle+FRED hybrid) in ONE fundamental way: the data layer sources
real NOMINAL 2Y OIS swap rates directly from Bloomberg via the `xbbg`
package. The Kaggle version had to use a 2Y real IRS yield + a Fisher-
equation proxy to manufacture a nominal-like series; that hack is GONE
here. This is the headline methodological improvement.

------------------------------------------------------------------------------
DATA SOURCE
------------------------------------------------------------------------------
All inputs are pulled from a Bloomberg Terminal through the `xbbg` Python
package (`from xbbg import blp`). The single bulk call `blp.bdh(...)` with
`Fill='P'` (previous-value fill) and `Days='A'` (all calendar days) gives a
clean daily panel for all 8 currencies x 5 variables. Monthly (unemp) and
quarterly (GDP) series are forward-filled onto the daily grid both by
Bloomberg's `Fill='P'` and by an explicit pandas reindex (defense in depth).

The 8 currencies and their tickers live in `BBG_TICKERS` below. Confidence
levels (HIGH / MEDIUM / LOW) are annotated inline. LOW-confidence tickers
(AUD/NZD/SEK/NOK OIS and small-ccy inflation swaps) MUST be verified on
the terminal before trusting the regression output - run `verify_tickers()`
FIRST.

------------------------------------------------------------------------------
DRY_RUN MODE (mandatory for offline development)
------------------------------------------------------------------------------
DRY_RUN = True  (default) -> generates synthetic random-walk data matching
                              the expected schema so the script can be run
                              end-to-end without a Bloomberg Terminal.
                              Outputs are written to `outputs_dry_run/` and
                              are SYNTHETIC - they are NOT usable for analysis
                              and are clearly labelled as such in every
                              output and in the validation report.

DRY_RUN = False            -> uses real `xbbg` calls. Requires a live
                              Bloomberg Terminal on the host machine.
                              Outputs are written to `outputs_bloomberg/`.

`xbbg` is imported LAZILY inside the loader function so this script passes
`python -m py_compile` even on machines without `xbbg` installed.

------------------------------------------------------------------------------
TICKER VERIFICATION STEP (run this FIRST on the terminal)
------------------------------------------------------------------------------
Before running the full pipeline with DRY_RUN=False, call `verify_tickers()`
from a Python shell on the terminal machine. It runs `blp.bdp()` with the
`SECURITY_NAME` field for every ticker in `BBG_TICKERS` and prints a table
showing which tickers resolve correctly. Any ticker that fails to resolve
must be replaced (suggested fallbacks are documented inline) before the
full regression is trusted.

------------------------------------------------------------------------------
POINT-IN-TIME CAVEAT (IRREDUCIBLE)
------------------------------------------------------------------------------
Bloomberg BDH returns LATEST-VINTAGE data, not point-in-time. Macro
revisions (especially GDP and unemployment) will be the most recent
revision as of the day the script runs, not the value that was known on
each historical date. The original JPMaQS dataset is revision-free
(quantamental - records the value that was actually available at each
timestamp). This produces an irreducible divergence between this script's
results and a true point-in-time replication. See the dedicated section
in `VALIDATION_REPORT.md` for the full discussion.

------------------------------------------------------------------------------
OUTPUTS
------------------------------------------------------------------------------
All outputs go to `outputs_bloomberg/` (real) or `outputs_dry_run/` (synthetic):
  - coefficients_table_10y.csv        our estimated coefficients + t-stats + Adj R2
  - coefficients_table_5y.csv         5Y "current regime" calibration (Figure 5 analogue)
  - comparison_vs_figure4.csv         side-by-side: ours vs published
  - comparison_vs_figure4.md          markdown rendering of the same
  - residuals_<CCY>.csv               daily actual / fitted / residual
  - residual_zscores.csv              latest 5Y rolling z-score per currency
  - actual_vs_fitted_<CCY>.png        Figure 8 analogue (8 plots)
  - residuals_<CCY>.png               Figure 6 analogue (8 plots)
  - residual_zscores_bar.png          Figure 7 analogue (cross-market bar)
  - VALIDATION_REPORT.md              structured validation report
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless backend - no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# MODE SELECTOR - flip to False on a Bloomberg Terminal host.
# ---------------------------------------------------------------------------
DRY_RUN = True  # Set to False when running with Bloomberg Terminal access.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = "/mnt/nas/data4/github project/jefferies project/JPMaQS"
OUTPUTS_DIR = os.path.join(
    PROJECT_DIR, "outputs_dry_run" if DRY_RUN else "outputs_bloomberg"
)

# ---------------------------------------------------------------------------
# Currencies and regression windows
# ---------------------------------------------------------------------------
CURRENCIES = ["USD", "EUR", "GBP", "SEK", "NOK", "AUD", "NZD", "JPY"]

# Note window is 2014-12-20 to 2024-12-20. Bloomberg's data extends through
# TODAY (unlike the Kaggle sample which ended Dec 2023), so the user may set
# the end date to "today" for a live signal reproduction.
WINDOW_10Y = ("2014-12-20", "2024-12-20")
WINDOW_5Y = ("2019-12-20", "2024-12-20")

# For live runs, override end date to today:
if not DRY_RUN:
    _today_str = datetime.today().strftime("%Y-%m-%d")
    WINDOW_10Y = (WINDOW_10Y[0], _today_str)
    WINDOW_5Y = (WINDOW_5Y[0], _today_str)

# Bloomberg pull window - go back a bit further than the regression window
# so the 5Y moving averages used in regressor construction are warmed up.
DATA_START = "2013-01-01"

# z-score lookback (business days ~ 5 years)
ZSCORE_WINDOW = 252 * 5

# Missing-data tolerance: drop a currency if any regressor is >20% missing
# inside the regression window.
MISSING_TOLERANCE = 0.20

# ---------------------------------------------------------------------------
# CONFIG - single source of truth. Tickers verified at the levels noted.
# Confidence levels:
#   HIGH   - macro series or G3 OIS, well-known to resolve reliably.
#   MEDIUM - ticker pattern is standard but mnemonic may need terminal check.
#   LOW    - small-ccy OIS or thin inflation-swap market - VERIFY ON TERMINAL.
# ---------------------------------------------------------------------------
BBG_TICKERS = {
    "USD": {
        "ois_2y": "USOSFR2 Curncy",  # HIGH: SOFR OIS 2Y (post-2020 standard)
        "policy": "FEDL01 Index",  # HIGH: Fed Funds Effective (what OIS floats off)
        "unemp": "USURTOT Index",  # HIGH: SA unemployment, monthly
        "infl_1y": "USSWIT1 Curncy",  # MEDIUM: 1Y ZCIIS
        "gdp": "GDP CHWG Index",  # HIGH: GDP QoQ SAAR
        "infl_target": 2.0,
    },
    "EUR": {
        "ois_2y": "EUSWEA2 Curncy",  # MEDIUM: EUR OIS (EA mnemonic, post-Oct 2021 ESTR-flat)
        "policy": "ECBDFR Index",  # HIGH: ECB Deposit Facility Rate
        "unemp": "UMRTEMU Index",  # HIGH: Euro Area harmonised unemp
        "infl_1y": "EUSWIT1 Curncy",  # MEDIUM: 1Y ZCIIS
        "gdp": "EUGNEMU Index",  # HIGH: EA GDP QoQ
        "infl_target": 2.0,
    },
    "GBP": {
        "ois_2y": "BPSWS2 Curncy",  # HIGH: SONIA OIS 2Y
        "policy": "UKBRBASE Index",  # HIGH: BoE Bank Rate
        "unemp": "UKUEILOR Index",  # HIGH: UK ILO unemployment
        "infl_1y": "BPSWIT1 Curncy",  # MEDIUM: 1Y ZCIIS
        "gdp": "EUGNUK Index",  # HIGH: UK GDP QoQ
        "infl_target": 2.0,
    },
    "JPY": {
        "ois_2y": "JYOES2 Curncy",  # MEDIUM: TONA OIS 2Y (verify on terminal)
        "policy": "JBOCDR Index",  # MEDIUM: BoJ policy-rate balance
        "unemp": "JNUE Index",  # HIGH: Japan unemployment
        "infl_1y": "JYSWIT1 Curncy",  # LOW: thin market - may need survey fallback
        "gdp": "JGDPAGDP Index",  # HIGH: Japan GDP QoQ SAAR
        "infl_target": 2.0,
    },
    "AUD": {
        "ois_2y": "ADSOA2 Curncy",  # LOW: AONIA OIS - VERIFY on terminal (try ADSOAS2 fallback)
        "policy": "RBAOCR Index",  # MEDIUM: RBA Official Cash Rate
        "unemp": "AULFUNEM Index",  # HIGH: Australia unemployment
        "infl_1y": "ADSWIT1 Curncy",  # LOW: thin - consider hardcoding target
        "gdp": "AUNAGDP Index",  # HIGH: Australia GDP QoQ
        "infl_target": 2.5,  # GOTCHA: RBA targets 2-3% midpoint = 2.5%
    },
    "NZD": {
        "ois_2y": "NZSOA2 Curncy",  # LOW: NZIONA OIS - VERIFY (market is thin)
        "policy": "RBNZOCR Index",  # MEDIUM: RBNZ Official Cash Rate
        "unemp": "NZLFUNER Index",  # HIGH: NZ unemployment
        "infl_1y": "NZSWIT1 Curncy",  # LOW: very thin - hardcode target recommended
        "gdp": "NZNTEGDP Index",  # HIGH: NZ GDP QoQ
        "infl_target": 2.0,
    },
    "SEK": {
        "ois_2y": "SDSOA2 Curncy",  # LOW: SIOR OIS - VERIFY
        "policy": "SBREPO Index",  # MEDIUM: Riksbank Repo (alt: SRREPO)
        "unemp": "SWUESART Index",  # HIGH: Sweden unemployment
        "infl_1y": "SDSWIT1 Curncy",  # LOW: thin
        "gdp": "EUGNSE Index",  # HIGH: Sweden GDP QoQ
        "infl_target": 2.0,
    },
    "NOK": {
        "ois_2y": "NOSOAS2 Curncy",  # LOW: NOWA OIS - VERIFY
        "policy": "NOWAO Index",  # MEDIUM: Norges Bank Key Policy (alt: NORWKEY)
        "unemp": "NOLBRATE Index",  # HIGH: Norway unemployment
        "infl_1y": "NOSWIT1 Curncy",  # LOW: thin
        "gdp": "NOGDCOSA Index",  # HIGH: Norway GDP QoQ
        "infl_target": 2.0,
    },
}

# Confidence-tier roll-up for the validation report.
TICKER_CONFIDENCE = {
    "USD": {
        "ois_2y": "HIGH",
        "policy": "HIGH",
        "unemp": "HIGH",
        "infl_1y": "MEDIUM",
        "gdp": "HIGH",
    },
    "EUR": {
        "ois_2y": "MEDIUM",
        "policy": "HIGH",
        "unemp": "HIGH",
        "infl_1y": "MEDIUM",
        "gdp": "HIGH",
    },
    "GBP": {
        "ois_2y": "HIGH",
        "policy": "HIGH",
        "unemp": "HIGH",
        "infl_1y": "MEDIUM",
        "gdp": "HIGH",
    },
    "JPY": {
        "ois_2y": "MEDIUM",
        "policy": "MEDIUM",
        "unemp": "HIGH",
        "infl_1y": "LOW",
        "gdp": "HIGH",
    },
    "AUD": {
        "ois_2y": "LOW",
        "policy": "MEDIUM",
        "unemp": "HIGH",
        "infl_1y": "LOW",
        "gdp": "HIGH",
    },
    "NZD": {
        "ois_2y": "LOW",
        "policy": "MEDIUM",
        "unemp": "HIGH",
        "infl_1y": "LOW",
        "gdp": "HIGH",
    },
    "SEK": {
        "ois_2y": "LOW",
        "policy": "MEDIUM",
        "unemp": "HIGH",
        "infl_1y": "LOW",
        "gdp": "HIGH",
    },
    "NOK": {
        "ois_2y": "LOW",
        "policy": "MEDIUM",
        "unemp": "HIGH",
        "infl_1y": "LOW",
        "gdp": "HIGH",
    },
}

# ---------------------------------------------------------------------------
# CONFIG wrapper (kept structurally similar to the Kaggle version so the
# downstream pipeline reads identically).
# ---------------------------------------------------------------------------
CONFIG = {
    # Dependent variable: 2Y NOMINAL OIS swap rate (Bloomberg direct).
    # The Fisher-equation proxy from the Kaggle version is GONE.
    "dep_var": {
        "source": "bloomberg",
        "field": "ois_2y",
        "note": "2Y nominal OIS swap rate - real market quote, no proxy.",
    },
    # Regressor 1: excess 1Y inflation expectations.
    # Simple subtraction: infl_1y (1Y ZCIIS) minus infl_target.
    "excess_infl": {
        "source": "bloomberg",
        "field": "infl_1y",
        "note": "1Y zero-coupon inflation swap minus official target.",
    },
    # Regressor 2: excess employment.
    # excess_empl = -(3M MA of unemployment - 5Y MA of unemployment).
    "excess_empl": {
        "source": "bloomberg",
        "field": "unemp",
        "short_ma": 3,  # 3M MA of unemployment (in business days for daily)
        "long_ma_months": 60,  # 5Y MA window in months
        "long_ma_bd": 252 * 5,  # 5Y MA window expressed in business days
        "note": "Inverse unemployment gap; positive = tight labour market.",
    },
    # Regressor 3: excess GDP growth.
    # Computed from raw GDP QoQ SAAR: excess = current - 5Y rolling MA.
    "excess_gdp": {
        "source": "bloomberg",
        "field": "gdp",
        "ma_window_quarters": 20,  # 5Y = 20 quarters
        "ma_window_bd": 252 * 5,  # 5Y in business days (after fwd-fill to daily)
        "note": "GDP QoQ SAAR minus its own 5Y rolling average.",
    },
    # Regressor 4: policy rate (passthrough).
    "policy_rate": {
        "source": "bloomberg",
        "field": "policy",
        "note": "Central-bank policy rate, direct from Bloomberg.",
    },
}

# ---------------------------------------------------------------------------
# Published Figure 4 reference (for side-by-side validation).
# Columns: adj_r2, const, infl_coef, empl_coef, gdp_coef, policy_coef
# ---------------------------------------------------------------------------
PUBLISHED_FIG4 = {
    "USD": dict(adj_r2=0.94, const=0.11, infl=0.35, empl=0.14, gdp=-0.06, policy=0.75),
    "EUR": dict(adj_r2=0.97, const=0.32, infl=0.38, empl=-0.17, gdp=0.00, policy=0.76),
    "GBP": dict(adj_r2=0.97, const=0.21, infl=0.40, empl=-0.09, gdp=0.00, policy=0.84),
    "SEK": dict(adj_r2=0.95, const=0.30, infl=0.30, empl=-0.02, gdp=0.05, policy=0.77),
    "NOK": dict(adj_r2=0.95, const=1.07, infl=0.34, empl=0.23, gdp=0.08, policy=0.78),
    "AUD": dict(adj_r2=0.91, const=0.54, infl=0.27, empl=0.19, gdp=0.12, policy=0.85),
    "NZD": dict(adj_r2=0.97, const=0.80, infl=0.65, empl=0.02, gdp=0.14, policy=0.76),
    "JPY": dict(adj_r2=0.95, const=0.17, infl=0.07, empl=-0.07, gdp=0.00, policy=1.47),
}

# Mandate classification (for the structural employment-sign validation):
# dual-mandate / labour-sensitive -> expect POSITIVE employment coefficient.
DUAL_MANDATE = {"USD", "NOK", "AUD"}  # Fed, Norges Bank, RBA
INFL_ONLY_MANDATE = {"EUR", "GBP", "JPY"}  # ECB, BoE, BoJ


# ===========================================================================
# Data loaders
# ===========================================================================
def _flatten_tickers() -> list[str]:
    """Return a flat list of every Bloomberg ticker in BBG_TICKERS."""
    out = []
    for cid in CURRENCIES:
        for fld in ("ois_2y", "policy", "unemp", "infl_1y", "gdp"):
            out.append(BBG_TICKERS[cid][fld])
    # de-dup while preserving order
    seen = set()
    flat = []
    for t in out:
        if t not in seen:
            seen.add(t)
            flat.append(t)
    return flat


def verify_tickers() -> pd.DataFrame:
    """Call blp.bdp() to confirm every ticker in BBG_TICKERS resolves.

    RUN THIS FIRST on a Bloomberg Terminal host. Prints a per-ticker table
    showing whether the SECURITY_NAME field resolves. Any ticker returning
    NaN/empty must be replaced (suggested fallbacks are documented inline
    in BBG_TICKERS) before the full regression can be trusted.

    Returns a DataFrame with columns [cid, field, ticker, security_name,
    resolved] for easy inspection / export.

    Lazily imports `xbbg` so this module still compiles without it.
    """
    from xbbg import blp  # lazy: only needed when actually verifying

    rows = []
    tickers = _flatten_tickers()
    print(f"   querying {len(tickers)} tickers via blp.bdp (SECURITY_NAME)...")
    ref = blp.bdp(tickers=tickers, flds=["SECURITY_NAME", "SECURITY_DES"])
    for cid in CURRENCIES:
        for fld in ("ois_2y", "policy", "unemp", "infl_1y", "gdp"):
            t = BBG_TICKERS[cid][fld]
            try:
                if isinstance(ref.columns, pd.MultiIndex):
                    name = ref.loc[t, ("SECURITY_NAME",)] if t in ref.index else np.nan
                    des = ref.loc[t, ("SECURITY_DES",)] if t in ref.index else np.nan
                else:
                    name = ref.loc[t, "SECURITY_NAME"] if t in ref.index else np.nan
                    des = ref.loc[t, "SECURITY_DES"] if t in ref.index else np.nan
            except Exception:
                name = np.nan
                des = np.nan
            resolved = isinstance(name, str) and len(name) > 0
            rows.append(
                {
                    "cid": cid,
                    "field": fld,
                    "ticker": t,
                    "confidence": TICKER_CONFIDENCE[cid][fld],
                    "security_name": name,
                    "security_des": des,
                    "resolved": resolved,
                }
            )
    out = pd.DataFrame(rows)
    n_resolved = int(out["resolved"].sum())
    n_total = len(out)
    print(f"   resolved {n_resolved}/{n_total} tickers")
    if n_resolved < n_total:
        print("   [WARN] Unresolved tickers:")
        for _, r in out[~out["resolved"]].iterrows():
            print(
                f"      {r['cid']:3s} {r['field']:8s} {r['ticker']:25s} "
                f"(confidence={r['confidence']})"
            )
    return out


def fetch_bloomberg_panel(start: str, end: str) -> pd.DataFrame:
    """Pull all tickers in BBG_TICKERS via blp.bdh with daily PX_LAST.

    Returns a DataFrame indexed by date with columns
    [(cid, field), ...] in a MultiIndex - matching the Kaggle panel
    layout so the rest of the pipeline reads identically.

    Bloombergs `Fill='P'` (previous-value fill) + `Days='A'` (all calendar
    days) should give a clean daily grid including monthly/quarterly series
    forward-filled. We apply our own reindex+ffill downstream as defense in
    depth.

    Lazily imports `xbbg` so this module compiles without it.
    """
    from xbbg import blp  # lazy: only needed when DRY_RUN=False

    tickers = _flatten_tickers()
    print(f"   blp.bdh: {len(tickers)} tickers, {start} -> {end}, daily PX_LAST")
    df = blp.bdh(
        tickers=tickers,
        flds="PX_LAST",
        start_date=start,
        end_date=end,
        Per="D",  # daily periodicity
        Fill="P",  # nonTradingDayFillMethod = PREVIOUS_VALUE
        Days="A",  # all calendar days (incl. non-trading)
    )
    # df has MultiIndex columns (ticker, 'PX_LAST'); collapse the field level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Re-pivot into (cid, field) MultiIndex the pipeline expects.
    out = pd.DataFrame(index=df.index)
    for cid in CURRENCIES:
        for fld in ("ois_2y", "policy", "unemp", "infl_1y", "gdp"):
            t = BBG_TICKERS[cid][fld]
            if t in df.columns:
                out[(cid, fld)] = df[t]
            else:
                out[(cid, fld)] = np.nan
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["cid", "field"])
    out = out.sort_index()
    return out


def generate_synthetic_panel(start: str, end: str) -> pd.DataFrame:
    """Generate a synthetic daily panel matching the expected schema.

    Used when DRY_RUN=True so the FULL pipeline (loaders -> regressors ->
    OLS -> plots -> report) can be exercised end-to-end without a Bloomberg
    Terminal. Outputs are SYNTHETIC and are NOT usable for analysis.

    Synthetic processes (annualised vol; daily step):
      - ois_2y:   ~ mean-reverting OU around policy + 1.0% term premium
      - policy:   sticky OU, slow mean reversion around 2.0%
      - unemp:    strong mean-reverting OU around 5.0%
      - infl_1y:  mild OU around infl_target + small noise
      - gdp:      mean-reverting around 2.0% with high vol

    Dependencies are engineered so the OLS produces plausible signs:
    positive inflation coefficient, dominant policy rate, modest GDP/empl.
    """
    grid = pd.bdate_range(start=start, end=end)
    n = len(grid)
    dt = 1.0 / 252.0  # business-day step in years

    rng = np.random.default_rng(seed=20260722)  # deterministic for repeatability

    # per-currency macro anchors (infl_target etc.)
    panel_cols = pd.MultiIndex.from_tuples(
        [
            (cid, fld)
            for cid in CURRENCIES
            for fld in ("ois_2y", "policy", "unemp", "infl_1y", "gdp")
        ],
        names=["cid", "field"],
    )
    out = pd.DataFrame(index=grid, columns=panel_cols, dtype="float64")

    for cid in CURRENCIES:
        infl_target = BBG_TICKERS[cid]["infl_target"]

        # --- policy rate: OU around 2.0%, sticky ---
        kappa_p, theta_p, sigma_p = 0.4, 2.0, 0.5
        policy = np.empty(n)
        policy[0] = theta_p
        for i in range(1, n):
            policy[i] = (
                policy[i - 1]
                + kappa_p * (theta_p - policy[i - 1]) * dt
                + sigma_p * np.sqrt(dt) * rng.standard_normal()
            )
        # push policy through the post-2022 hiking cycle to look realistic
        t_yr = (grid - grid[0]).days / 365.25
        hike_shape = 2.2 * np.exp(-((t_yr - 5.0) ** 2) / (2 * 1.5**2))
        policy = policy + hike_shape
        policy = np.clip(policy, -0.5, 8.0)

        # --- infl_1y: OU around target + 0.5% near-term overshoot ---
        kappa_i, theta_i, sigma_i = 0.8, infl_target + 0.4, 0.4
        infl = np.empty(n)
        infl[0] = theta_i
        for i in range(1, n):
            infl[i] = (
                infl[i - 1]
                + kappa_i * (theta_i - infl[i - 1]) * dt
                + sigma_i * np.sqrt(dt) * rng.standard_normal()
            )
        # add a transient inflation spike around the 2022 hiking cycle
        infl_spike = 2.5 * np.exp(-((t_yr - 5.0) ** 2) / (2 * 1.0**2))
        infl = infl + infl_spike
        infl = np.clip(infl, -1.0, 12.0)

        # --- unemp: OU around 5.0%, strong mean reversion ---
        kappa_u, theta_u, sigma_u = 1.2, 5.0, 0.3
        unemp = np.empty(n)
        unemp[0] = theta_u
        for i in range(1, n):
            unemp[i] = (
                unemp[i - 1]
                + kappa_u * (theta_u - unemp[i - 1]) * dt
                + sigma_u * np.sqrt(dt) * rng.standard_normal()
            )
        # COVID-style unemployment spike ~2020
        covid_spike = 3.5 * np.exp(-((t_yr - 7.0) ** 2) / (2 * 0.2**2))
        unemp = unemp + covid_spike
        unemp = np.clip(unemp, 1.5, 15.0)

        # --- gdp: OU around 2.0%, mean-reverting, high vol ---
        kappa_g, theta_g, sigma_g = 1.0, 2.0, 1.5
        gdp = np.empty(n)
        gdp[0] = theta_g
        for i in range(1, n):
            gdp[i] = (
                gdp[i - 1]
                + kappa_g * (theta_g - gdp[i - 1]) * dt
                + sigma_g * np.sqrt(dt) * rng.standard_normal()
            )
        # COVID GDP plunge + snapback
        covid_gdp = -8.0 * np.exp(-((t_yr - 7.0) ** 2) / (2 * 0.15**2)) + 5.0 * np.exp(
            -((t_yr - 7.2) ** 2) / (2 * 0.2**2)
        )
        gdp = gdp + covid_gdp
        gdp = np.clip(gdp, -15.0, 12.0)

        # --- ois_2y: nominal OIS ~ policy + term premium + infl response ---
        # Engineered so OLS recovers positive inflation coef & dominant policy.
        # Add a small noise term so R^2 is realistic but not 1.0.
        term_premium = 0.4 + 0.1 * np.sin(2 * np.pi * t_yr / 5.0)
        noise_ois = 0.05 * rng.standard_normal(n)
        ois = (
            0.15  # intercept
            + 0.95 * policy  # near-1 beta on policy
            + 0.35 * (infl - infl_target)  # positive inflation-gap loading
            + 0.10 * (5.0 - unemp)  # tight labour market -> higher OIS
            + 0.05 * (gdp - 2.0)  # modest GDP gap loading
            + term_premium
            + noise_ois
        )
        ois = np.clip(ois, -0.6, 10.0)

        out[(cid, "ois_2y")] = ois
        out[(cid, "policy")] = policy
        out[(cid, "unemp")] = unemp
        out[(cid, "infl_1y")] = infl
        out[(cid, "gdp")] = gdp

    return out


def load_panel(start: str, end: str) -> pd.DataFrame:
    """Top-level panel loader - dispatches based on DRY_RUN."""
    if DRY_RUN:
        print("   [DRY_RUN] generating SYNTHETIC panel - outputs NOT for analysis")
        return generate_synthetic_panel(start, end)
    return fetch_bloomberg_panel(start, end)


# ===========================================================================
# Variable constructors
# ===========================================================================
def get_panel_series(panel: pd.DataFrame, cid: str, field: str) -> pd.Series:
    """Extract a single (cid, field) series from the wide panel."""
    try:
        s = panel[(cid, field)].copy()
    except KeyError:
        return pd.Series(dtype="float64", name=field)
    s.name = field
    return s


def construct_dep_var(panel: pd.DataFrame, cid: str) -> pd.Series:
    """Dependent variable: 2Y NOMINAL OIS swap rate (Bloomberg direct).

    No Fisher-equation proxy, no real-yield reconstruction. The OIS swap
    rate IS the nominal fair-value object the note regresses.
    """
    s = get_panel_series(panel, cid, CONFIG["dep_var"]["field"])
    s.name = "dep_var"
    return s


def construct_excess_infl(panel: pd.DataFrame, cid: str) -> pd.Series:
    """Excess 1Y-ahead inflation expectations = infl_1y - infl_target.

    Bloomberg gives us the 1Y zero-coupon inflation swap directly, so this
    is a single subtraction - no CPI-blend, no outlier dampening, no
    return-to-target blend (all of which the Kaggle version needed).
    """
    infl = get_panel_series(panel, cid, CONFIG["excess_infl"]["field"])
    target = BBG_TICKERS[cid]["infl_target"]
    excess = infl - target
    excess.name = "excess_infl"

    # Thin-market sanity check: if the swap returns implausible values
    # (NaN-heavy or |value| > 20%), flag it for the report. We do NOT
    # auto-fallback to a hardcode here because that would silently mask a
    # ticker-resolution problem; the user must decide.
    valid = excess.dropna()
    if len(valid) > 0:
        if valid.abs().max() > 20.0:
            print(
                f"   [WARN] {cid}: infl_1y has implausible values "
                f"(max |excess|={valid.abs().max():.1f}%) - "
                f"check the {BBG_TICKERS[cid]['infl_1y']} ticker on the terminal"
            )
    return excess


def construct_excess_empl(panel: pd.DataFrame, cid: str) -> pd.Series:
    """Excess employment = -(3M MA of unemp - 5Y MA of unemp).

    Positive value = labour market tight (unemployment below its 5Y norm).
    """
    cfg = CONFIG["excess_empl"]
    unemp = get_panel_series(panel, cid, cfg["field"])
    # Short MA in business days (~63 BD = 3 months)
    short_bd = int(round(cfg["short_ma"] * 252 / 12))
    short = unemp.rolling(short_bd, min_periods=short_bd // 3).mean()
    long = unemp.rolling(cfg["long_ma_bd"], min_periods=cfg["long_ma_bd"] // 4).mean()
    gap = short - long
    excess = -gap
    excess.name = "excess_empl"
    return excess


def construct_excess_gdp(panel: pd.DataFrame, cid: str) -> pd.Series:
    """Excess GDP growth = current GDP QoQ SAAR - its own 5Y rolling MA.

    Kaggle version pulled INTRGDPv5Y (a pre-computed excess). With
    Bloomberg we compute the excess ourselves from raw GDP - more
    transparent and closer to the note's methodology.
    """
    cfg = CONFIG["excess_gdp"]
    gdp = get_panel_series(panel, cid, cfg["field"])
    ma = gdp.rolling(cfg["ma_window_bd"], min_periods=cfg["ma_window_bd"] // 4).mean()
    excess = gdp - ma
    excess.name = "excess_gdp"
    return excess


def construct_policy_rate(panel: pd.DataFrame, cid: str) -> pd.Series:
    """Policy rate passthrough (already pulled into the panel)."""
    s = get_panel_series(panel, cid, CONFIG["policy_rate"]["field"])
    s.name = "policy_rate"
    return s


# ===========================================================================
# Alignment onto a common business-day grid
# ===========================================================================
def build_business_day_grid(start: str, end: str) -> pd.DatetimeIndex:
    """Standard pandas BDay calendar restricted to [start, end]."""
    grid = pd.bdate_range(start=start, end=end)
    return grid


def align_and_fill(
    series_dict: dict[str, pd.Series], grid: pd.DatetimeIndex
) -> pd.DataFrame:
    """Reindex every series onto the business-day grid and forward-fill.

    Backward-fill the first few rows if forward-fill leaves leading NaNs
    (defense in depth - Bloomberg's Fill='P' should already handle this,
    but we apply our own pass too).
    """
    out = pd.DataFrame(index=grid)
    for name, s in series_dict.items():
        if s.empty:
            out[name] = np.nan
            continue
        reindexed = s.reindex(grid)
        # forward-fill then backward-fill the leading gap
        reindexed = reindexed.ffill().bfill()
        out[name] = reindexed
    return out


# ===========================================================================
# OLS regression (structurally identical to the Kaggle version)
# ===========================================================================
REGRESSORS = ["excess_infl", "excess_empl", "excess_gdp", "policy_rate"]


def run_ols(
    data: pd.DataFrame, window: tuple[str, str], cid: str
) -> tuple[Optional[dict], Optional[pd.DataFrame]]:
    """Run OLS of dep_var on [const, 4 regressors] over `window`.

    Returns (result_dict, aligned_df) or (None, None) if data is too sparse.
    """
    w_start, w_end = window
    sub = data.loc[w_start:w_end].copy()

    # missing-data gate
    n = len(sub)
    if n == 0:
        return None, None
    for col in ["dep_var"] + REGRESSORS:
        miss_frac = sub[col].isna().mean()
        if miss_frac > MISSING_TOLERANCE:
            print(
                f"   [SKIP] {cid}: {col} missing {miss_frac:.1%} > "
                f"{MISSING_TOLERANCE:.0%} tolerance"
            )
            return None, None

    sub = sub.dropna(subset=["dep_var"] + REGRESSORS)
    if len(sub) < 50:
        print(f"   [SKIP] {cid}: only {len(sub)} valid rows after dropna")
        return None, None

    y = sub["dep_var"]
    X = sm.add_constant(sub[REGRESSORS])
    model = sm.OLS(y, X).fit()

    res = {
        "cid": cid,
        "n_obs": int(len(sub)),
        "adj_r2": float(model.rsquared_adj),
        "const": float(model.params["const"]),
        "const_t": float(model.tvalues["const"]),
        "excess_infl": float(model.params["excess_infl"]),
        "excess_infl_t": float(model.tvalues["excess_infl"]),
        "excess_empl": float(model.params["excess_empl"]),
        "excess_empl_t": float(model.tvalues["excess_empl"]),
        "excess_gdp": float(model.params["excess_gdp"]),
        "excess_gdp_t": float(model.tvalues["excess_gdp"]),
        "policy_rate": float(model.params["policy_rate"]),
        "policy_rate_t": float(model.tvalues["policy_rate"]),
    }

    fitted = model.fittedvalues
    aligned = sub[["dep_var"] + REGRESSORS].copy()
    aligned["fitted"] = fitted
    aligned["residual"] = model.resid
    return res, aligned


# ===========================================================================
# z-score of residuals (5Y rolling, Figure 7 analogue)
# ===========================================================================
def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    z = (s - s.rolling(window, min_periods=window // 2).mean()) / s.rolling(
        window, min_periods=window // 2
    ).std()
    return z


# ===========================================================================
# Plotting (Figures 6, 7, 8 analogues)
# ===========================================================================
def plot_actual_vs_fitted(
    aligned: pd.DataFrame, cid: str, adj_r2: float, out_dir: str
) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(aligned.index, aligned["dep_var"], label="Actual (2Y nominal OIS)", lw=1.4)
    ax.plot(
        aligned.index,
        aligned["fitted"],
        label="Fitted (fair value)",
        lw=1.4,
        alpha=0.85,
    )
    title_prefix = "[SYNTHETIC] " if DRY_RUN else ""
    ax.set_title(
        f"{title_prefix}{cid} 2Y nominal OIS: actual vs fitted  (Adj R2 = {adj_r2:.2f})"
    )
    ax.set_ylabel("%")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    path = os.path.join(out_dir, f"actual_vs_fitted_{cid}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_residuals(aligned: pd.DataFrame, cid: str, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(aligned.index, aligned["residual"], color="crimson", lw=1.1)
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(
        aligned.index,
        aligned["residual"],
        0,
        where=aligned["residual"] >= 0,
        color="crimson",
        alpha=0.25,
        interpolate=True,
    )
    ax.fill_between(
        aligned.index,
        aligned["residual"],
        0,
        where=aligned["residual"] < 0,
        color="steelblue",
        alpha=0.25,
        interpolate=True,
    )
    title_prefix = "[SYNTHETIC] " if DRY_RUN else ""
    ax.set_title(f"{title_prefix}{cid} model residual (actual - fitted)")
    ax.set_ylabel("%-pt")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    path = os.path.join(out_dir, f"residuals_{cid}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_zscore_bar(zscores: dict[str, float], out_dir: str) -> str:
    cids = list(zscores.keys())
    vals = [zscores[c] for c in cids]
    colors = ["crimson" if v > 0 else "steelblue" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(cids, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="grey", lw=0.6, ls="--")
    ax.axhline(-1, color="grey", lw=0.6, ls="--")
    title_prefix = "[SYNTHETIC] " if DRY_RUN else ""
    ax.set_title(
        f"{title_prefix}Latest 5Y z-score of model residual (Figure 7 analogue)"
    )
    ax.set_ylabel("z-score")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.05 * np.sign(v),
            f"{v:+.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(out_dir, "residual_zscores_bar.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ===========================================================================
# Comparison table (ours vs published Figure 4)
# ===========================================================================
def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        c = r["cid"]
        pub = PUBLISHED_FIG4.get(c, {})
        rows.append(
            {
                "cid": c,
                "our_adj_r2": round(r["adj_r2"], 2),
                "pub_adj_r2": pub.get("adj_r2", np.nan),
                "our_const": round(r["const"], 2),
                "pub_const": pub.get("const", np.nan),
                "our_infl": round(r["excess_infl"], 2),
                "our_infl_t": round(r["excess_infl_t"], 1),
                "pub_infl": pub.get("infl", np.nan),
                "our_empl": round(r["excess_empl"], 2),
                "our_empl_t": round(r["excess_empl_t"], 1),
                "pub_empl": pub.get("empl", np.nan),
                "our_gdp": round(r["excess_gdp"], 2),
                "our_gdp_t": round(r["excess_gdp_t"], 1),
                "pub_gdp": pub.get("gdp", np.nan),
                "our_policy": round(r["policy_rate"], 2),
                "our_policy_t": round(r["policy_rate_t"], 1),
                "pub_policy": pub.get("policy", np.nan),
            }
        )
    return pd.DataFrame(rows)


def comparison_to_markdown(comp: pd.DataFrame) -> str:
    lines = []
    title_prefix = "[SYNTHETIC] " if DRY_RUN else ""
    lines.append(f"# {title_prefix}Coefficient comparison: ours vs Figure 4\n")
    lines.append(
        "Adj R2 and coefficients side-by-side. Published figures from the "
        "22 July 2026 note, Figure 4 (10Y OLS, dependent = 2Y nominal OIS).\n"
    )
    if DRY_RUN:
        lines.append(
            "**WARNING: SYNTHETIC OUTPUT - generated by DRY_RUN mode.** "
            "These numbers are from random-walk synthetic data and are "
            "NOT usable for analysis. They exist only to verify the "
            "pipeline structure end-to-end.\n"
        )
    lines.append(
        "| Ccy | Adj R2 (ours) | Adj R2 (pub) | "
        "Const (ours/pub) | Infl (ours/pub) | "
        "Empl (ours/pub) | GDP (ours/pub) | "
        "Policy (ours/pub) |"
    )
    lines.append(
        "|-----|---------------|--------------|"
        "-------------------|------------------|"
        "------------------|-----------------|"
        "--------------------|"
    )
    for _, row in comp.iterrows():
        lines.append(
            f"| {row['cid']} "
            f"| {row['our_adj_r2']:.2f} | {row['pub_adj_r2']:.2f} "
            f"| {row['our_const']:.2f} / {row['pub_const']:.2f} "
            f"| {row['our_infl']:.2f} (t={row['our_infl_t']:.0f}) / "
            f"{row['pub_infl']:.2f} "
            f"| {row['our_empl']:.2f} (t={row['our_empl_t']:.0f}) / "
            f"{row['pub_empl']:.2f} "
            f"| {row['our_gdp']:.2f} (t={row['our_gdp_t']:.0f}) / "
            f"{row['pub_gdp']:.2f} "
            f"| {row['our_policy']:.2f} (t={row['our_policy_t']:.0f}) / "
            f"{row['pub_policy']:.2f} |"
        )
    return "\n".join(lines) + "\n"


# ===========================================================================
# Validation report
# ===========================================================================
def write_validation_report(
    results_10y: list[dict],
    zscores: dict[str, float],
    out_dir: str,
) -> str:
    """Write a structured validation report addressing the specific
    methodological questions the task requires.

    Sections:
      0. Run-mode banner (DRY_RUN vs real)
      1. Headline improvement: nominal OIS vs real-yield+Fisher proxy
      2. Structural sign checks (inflation positive, policy dominant, etc.)
      3. Point-in-time caveat (IRREDUCIBLE - latest-vintage vs revision-free)
      4. Convention changes over the 10Y window (EONIA->EUR, LIBOR->SONIA etc.)
      5. Ticker verification status table (HIGH / MEDIUM / LOW confidence)
      6. Window extension (Bloomberg data goes through TODAY)
      7. Per-currency z-score snapshot (rich / cheap signal)
    """
    path = os.path.join(out_dir, "VALIDATION_REPORT.md")
    lines: list[str] = []

    if DRY_RUN:
        lines.append("# [SYNTHETIC] 2Y OIS Fair Value Model - Validation Report")
        lines.append("")
        lines.append(
            "> **DRY_RUN = True.** All numbers below are generated from "
            "synthetic random-walk data. They verify the pipeline STRUCTURE "
            "only and are NOT usable for economic analysis. Re-run with "
            "`DRY_RUN = False` on a Bloomberg Terminal host for real results."
        )
    else:
        lines.append("# 2Y OIS Fair Value Model - Validation Report (Bloomberg)")
    lines.append("")

    # ---- Section 1: nominal vs real ----
    lines.append("## 1. Headline improvement: nominal OIS vs real yield + Fisher proxy")
    lines.append("")
    lines.append(
        "The Kaggle edition of this model used `RYLDIRS02Y_NSA` (the 2Y "
        "**real** IRS yield) as the dependent variable, then reconstructed "
        "a nominal-like series via the Fisher equation "
        "(`nominal = real + 1Y_expected_inflation`). The Fisher proxy "
        "restores the expected positive inflation coefficient sign but "
        "introduces an extra layer of measurement error."
    )
    lines.append("")
    lines.append(
        "This Bloomberg edition pulls the 2Y **nominal** OIS swap rate "
        "directly (`ois_2y` in `BBG_TICKERS`) - no Fisher reconstruction. "
        "Expected effects on the regression:"
    )
    lines.append("")
    lines.append(
        "- **Inflation coefficient**: should be more stable and closer to "
        "the note's published values, since we no longer compound two "
        "estimates of expected inflation."
    )
    lines.append(
        "- **Adjusted R2**: should be modestly HIGHER than the Kaggle "
        "version because the dependent variable is no longer a "
        "two-step reconstruction."
    )
    lines.append(
        "- **Policy-rate coefficient**: should remain close to 1 (the note "
        "reports 0.75-0.85 for G10 ex-JPY)."
    )
    lines.append("")

    # ---- Section 2: structural sign checks ----
    lines.append("## 2. Structural sign checks")
    lines.append("")
    if results_10y:
        infl_pos = sum(1 for r in results_10y if r["excess_infl"] > 0)
        policy_dom = sum(
            1
            for r in results_10y
            if abs(r["policy_rate"])
            >= max(abs(r["excess_infl"]), abs(r["excess_empl"]), abs(r["excess_gdp"]))
        )
        lines.append(
            f"- Inflation coefficient POSITIVE in {infl_pos}/{len(results_10y)} "
            "currencies (expected: all)."
        )
        lines.append(
            f"- Policy rate is the DOMINANT regressor (largest |coef|) in "
            f"{policy_dom}/{len(results_10y)} currencies (expected: all "
            "except JPY where it is anomalously large in the OTHER direction)."
        )
        lines.append("")
        lines.append("Per-mandate employment-sign check:")
        for r in results_10y:
            cid = r["cid"]
            expected = "POSITIVE" if cid in DUAL_MANDATE else "negative"
            actual = r["excess_empl"]
            sign_ok = (actual > 0 and cid in DUAL_MANDATE) or (
                actual < 0 and cid in INFL_ONLY_MANDATE
            )
            flag = "OK" if sign_ok else "OFF-SIGN"
            lines.append(
                f"  - {cid}: empl_coef={actual:+.2f} "
                f"(mandate expects {expected}) -> {flag}"
            )
        # JPY anomaly
        jpy = next((r for r in results_10y if r["cid"] == "JPY"), None)
        if jpy:
            lines.append("")
            lines.append(
                f"- JPY policy coefficient = {jpy['policy_rate']:+.2f}. "
                "The note flags JPY as anomalously large (published ~1.47) "
                "due to the YCC regime - the 2Y OIS is essentially pegged to "
                "the policy rate, producing a near-unity-or-higher beta."
            )
    else:
        lines.append("_No regression results available (data pipeline failed)._")
    lines.append("")

    # ---- Section 3: point-in-time caveat ----
    lines.append("## 3. Point-in-time caveat (IRREDUCIBLE)")
    lines.append("")
    lines.append(
        "Bloomberg `BDH` returns the **latest-vintage** value of each "
        "series. Macro releases - especially GDP (revised 3+ times after "
        "the advance print) and unemployment (annual benchmark revisions) "
        "- will be the value known as of TODAY, not the value known on "
        "each historical date in the regression window."
    )
    lines.append("")
    lines.append(
        "The JPMaQS dataset used by the note is **revision-free** "
        "(quantamental - records only the value that was actually available "
        "at each timestamp). This produces an irreducible divergence "
        "between this script's results and a true point-in-time replication:"
    )
    lines.append("")
    lines.append(
        "- **GDP** is the worst offender: the 5Y-MA-based excess measure "
        "uses TODAY's revised GDP history. Periods of large benchmark "
        "revisions (e.g. UK ONS blue-book 2021-2022) will distort the "
        "excess-GDP regressor for the affected dates."
    )
    lines.append(
        "- **Unemployment** is monthly and revised - usually small revisions "
        "but annual benchmarks can shift the level by 0.1-0.3 pp."
    )
    lines.append(
        "- **Inflation swaps** and **OIS swap rates** are market quotes and "
        "ARE effectively point-in-time (no revision) - the headline "
        "improvement of this edition."
    )
    lines.append(
        "- **Policy rates** are essentially point-in-time (central-bank "
        "decisions are not revised after the fact)."
    )
    lines.append("")
    lines.append(
        "Net effect: the OIS / policy / inflation-swap side of the "
        "regression is clean; the GDP / unemployment side carries look-ahead "
        "bias. A true point-in-time replication would require Bloomberg's "
        "`BLP_GDSC` / vintage data product, which is out of scope here."
    )
    lines.append("")

    # ---- Section 4: convention changes ----
    lines.append("## 4. Convention changes over the 10Y window")
    lines.append("")
    lines.append(
        "The 2014-2024 regression window spans four major benchmark-rate "
        "regime changes. Modern Bloomberg tickers (SOFR, EUR-flat, SONIA, "
        "TONA) back-fill the historical series using the new convention, "
        "but small regime breaks remain:"
    )
    lines.append("")
    lines.append("| Currency | Old convention | New convention | Switch date | Notes |")
    lines.append("|----------|----------------|----------------|-------------|-------|")
    lines.append(
        "| USD | Fed Funds OIS | **SOFR OIS** (`USOSFR2`) | Oct 2020 | "
        "SOFR replaced Fed Funds as the standard risk-free rate. |"
    )
    lines.append(
        "| EUR | EONIA OIS | **EUR OIS** (`EUSWEA2`) | Oct 2021 | "
        "EUR-flat replaced EONIA when EUR went negative-then-positive. |"
    )
    lines.append(
        "| GBP | LIBOR-based swaps | **SONIA OIS** (`BPSWS2`) | Dec 2021 | "
        "LIBOR cessation; SONIA became the standard. |"
    )
    lines.append(
        "| JPY | JPY LIBOR / TIBOR | **TONA OIS** (`JYOES2`) | Jul 2021 | "
        "TONA became the standard risk-free rate post-LIBOR. |"
    )
    lines.append("")
    lines.append(
        "Implication: the first ~3-5 years of the 10Y window are "
        "back-filled under the new convention and may differ slightly from "
        "what was actually tradeable at the time. This affects level and "
        "volatility in the early part of the sample."
    )
    lines.append("")

    # ---- Section 5: ticker verification status ----
    lines.append("## 5. Ticker verification status")
    lines.append("")
    lines.append(
        "Confidence levels as documented in `BBG_TICKERS`. "
        "**Before trusting any regression output (DRY_RUN=False), run "
        "`verify_tickers()` on the terminal** and confirm every row below "
        "resolves. Tickers marked LOW need terminal-side verification "
        "(small-ccy OIS markets are thin and mnemonics vary by desk)."
    )
    lines.append("")
    lines.append("| Ccy | Field | Ticker | Confidence | Action if unresolved |")
    lines.append("|-----|-------|--------|------------|----------------------|")
    fallbacks = {
        ("AUD", "ois_2y"): "try `ADSOAS2 Curncy` (AONIA swap); verify desk mnemonic",
        ("NZD", "ois_2y"): "verify; NZ OIS market is very thin",
        ("SEK", "ois_2y"): "verify; SIOR vs STIBOR conventions",
        ("NOK", "ois_2y"): "verify; NOWA OIS is the post-2019 standard",
        ("JPY", "infl_1y"): "thin market - fall back to hardcode target + CPI spread",
        ("AUD", "infl_1y"): "thin - consider hardcoding infl_target",
        ("NZD", "infl_1y"): "very thin - hardcode target recommended",
        ("SEK", "infl_1y"): "thin - verify",
        ("NOK", "infl_1y"): "thin - verify",
        ("EUR", "ois_2y"): "EUR vs EONIA legacy - verify post-Oct 2021",
        ("JPY", "ois_2y"): "verify TONA mnemonic",
        ("JPY", "policy"): "BoJ policy-rate balance - verify",
        ("SEK", "policy"): "alt: `SRREPO Index`",
        ("NOK", "policy"): "alt: `NORWKEY Index`",
    }
    for cid in CURRENCIES:
        for fld in ("ois_2y", "policy", "unemp", "infl_1y", "gdp"):
            t = BBG_TICKERS[cid][fld]
            conf = TICKER_CONFIDENCE[cid][fld]
            action = fallbacks.get(
                (cid, fld), "-" if conf == "HIGH" else "verify on terminal"
            )
            lines.append(f"| {cid} | {fld} | `{t}` | {conf} | {action} |")
    lines.append("")

    # ---- Section 6: window extension ----
    lines.append("## 6. Window extension")
    lines.append("")
    if DRY_RUN:
        lines.append(
            "The Kaggle edition's regression window was hard-capped at "
            "Dec-2023 (the end of the free JPMaQS sample). The Bloomberg "
            "edition is configured with a fixed 2014-12-20 -> 2024-12-20 "
            "10Y window for synthetic testing. **In live mode "
            "(`DRY_RUN = False`), the window end auto-advances to TODAY**, "
            "so the user can reproduce the note's actual Jul-2026 signal "
            "window and roll the model forward on subsequent days."
        )
    else:
        lines.append(
            "The Kaggle edition's regression window was hard-capped at "
            "Dec-2023 (the end of the free JPMaQS sample). This Bloomberg "
            f"edition runs through TODAY (`{WINDOW_10Y[1]}`), so the user "
            "can reproduce the note's actual Jul-2026 signal window and "
            "roll the model forward on subsequent days."
        )
    lines.append("")
    lines.append(f"- 10Y window: `{WINDOW_10Y[0]}` -> `{WINDOW_10Y[1]}`")
    lines.append(f"- 5Y window: `{WINDOW_5Y[0]}` -> `{WINDOW_5Y[1]}`")
    lines.append("")

    # ---- Section 7: z-scores ----
    lines.append("## 7. Latest 5Y residual z-score (rich / cheap signal)")
    lines.append("")
    finite_z = {c: v for c, v in zscores.items() if not np.isnan(v)}
    if finite_z:
        lines.append("| Ccy | z-score | Signal |")
        lines.append("|-----|---------|--------|")
        for c, v in sorted(finite_z.items(), key=lambda kv: -kv[1]):
            if v > 1:
                sig = "RICH (OIS > fair value)"
            elif v < -1:
                sig = "CHEAP (OIS < fair value)"
            else:
                sig = "near fair value"
            lines.append(f"| {c} | {v:+.2f} | {sig} |")
    else:
        lines.append("_No finite z-scores available._")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print("=" * 72)
    print("J.P. Morgan 2Y OIS fair value model - BLOOMBERG EDITION")
    if DRY_RUN:
        print(">>> DRY_RUN = True  ::  outputs are SYNTHETIC, not for analysis <<<")
    else:
        print(">>> DRY_RUN = False ::  using real xbbg calls <<<")
    print("=" * 72)

    # ---- [1/7] Verify tickers (live only) -----------------------------
    if DRY_RUN:
        print("\n[1/7] (skipped in DRY_RUN) Ticker verification")
    else:
        print("\n[1/7] Verifying tickers via blp.bdp(SECURITY_NAME)...")
        try:
            verify_tickers()
        except Exception as exc:
            print(f"   [WARN] verify_tickers raised: {exc}")
            print("   continuing - downstream pull will fail loudly if a ticker is bad")

    # ---- [2/7] Fetch Bloomberg panel ----------------------------------
    print(
        "\n[2/7] Fetching Bloomberg data..."
        if not DRY_RUN
        else "\n[2/7] Generating synthetic panel..."
    )
    panel = load_panel(DATA_START, WINDOW_10Y[1])
    print(f"   panel ready: {panel.shape[0]} days x {panel.shape[1]} (cid,field) cols")
    print(f"   date range: {panel.index.min().date()} .. {panel.index.max().date()}")

    # ---- [3/7] Build per-currency regression input --------------------
    print("\n[3/7] Constructing regressors and aligning to business-day grid...")
    grid = build_business_day_grid(WINDOW_10Y[0], WINDOW_10Y[1])
    print(
        f"   business-day grid: {len(grid)} days ({grid[0].date()}..{grid[-1].date()})"
    )

    per_cid_data: dict[str, pd.DataFrame] = {}
    for cid in CURRENCIES:
        dep = construct_dep_var(panel, cid)
        infl = construct_excess_infl(panel, cid)
        empl = construct_excess_empl(panel, cid)
        gdp = construct_excess_gdp(panel, cid)
        pol = construct_policy_rate(panel, cid)

        series_dict = {
            "dep_var": dep,
            "excess_infl": infl,
            "excess_empl": empl,
            "excess_gdp": gdp,
            "policy_rate": pol,
        }
        per_cid_data[cid] = align_and_fill(series_dict, grid)

    print("   dependent variable: 2Y nominal OIS (real market quote, no proxy)")

    # ---- [4/7] Run OLS ------------------------------------------------
    print("\n[4/7] Running OLS regressions (10Y baseline)...")
    results_10y: list[dict] = []
    aligned_10y: dict[str, pd.DataFrame] = {}
    for cid in CURRENCIES:
        res, aligned = run_ols(per_cid_data[cid], WINDOW_10Y, cid)
        if res is None:
            print(f"   {cid}: FAILED (see warning above)")
            continue
        results_10y.append(res)
        aligned_10y[cid] = aligned
        print(
            f"   {cid}: Adj R2={res['adj_r2']:.3f}  "
            f"infl={res['excess_infl']:+.2f} (t={res['excess_infl_t']:+.1f})  "
            f"empl={res['excess_empl']:+.2f} (t={res['excess_empl_t']:+.1f})  "
            f"gdp={res['excess_gdp']:+.2f} (t={res['excess_gdp_t']:+.1f})  "
            f"pol={res['policy_rate']:+.2f} (t={res['policy_rate_t']:+.1f})"
        )

    # 5Y current-regime calibration (Figure 5 analogue)
    print("\n[4b/7] Running OLS regressions (5Y current regime)...")
    results_5y: list[dict] = []
    for cid in CURRENCIES:
        res, _ = run_ols(per_cid_data[cid], WINDOW_5Y, cid)
        if res is not None:
            results_5y.append(res)
            print(f"   {cid}: Adj R2={res['adj_r2']:.3f}")

    # ---- [5/7] Residuals, z-scores, plots -----------------------------
    print("\n[5/7] Computing residual z-scores and saving diagnostic plots...")
    zscore_latest: dict[str, float] = {}
    for cid, aligned in aligned_10y.items():
        # save per-currency residual csv
        aligned.to_csv(os.path.join(OUTPUTS_DIR, f"residuals_{cid}.csv"))
        # 5Y rolling z-score
        z = rolling_zscore(aligned["residual"], ZSCORE_WINDOW)
        latest = float(z.iloc[-1]) if not np.isnan(z.iloc[-1]) else float("nan")
        zscore_latest[cid] = latest
        # plots
        r = next(x for x in results_10y if x["cid"] == cid)
        plot_actual_vs_fitted(aligned, cid, r["adj_r2"], OUTPUTS_DIR)
        plot_residuals(aligned, cid, OUTPUTS_DIR)

    # z-score bar (only currencies with a finite z-score)
    zscore_plot = {c: v for c, v in zscore_latest.items() if not np.isnan(v)}
    if zscore_plot:
        plot_zscore_bar(zscore_plot, OUTPUTS_DIR)

    # save zscore csv
    pd.DataFrame(
        [{"cid": c, "latest_zscore_5y": v} for c, v in zscore_latest.items()]
    ).to_csv(os.path.join(OUTPUTS_DIR, "residual_zscores.csv"), index=False)

    # ---- [6/7] Save coefficient tables --------------------------------
    print("\n[6/7] Saving coefficient tables and comparison vs Figure 4...")
    df_10y: Optional[pd.DataFrame] = None
    df_5y: Optional[pd.DataFrame] = None
    if results_10y:
        df_10y = pd.DataFrame(results_10y).set_index("cid")
        df_10y.to_csv(os.path.join(OUTPUTS_DIR, "coefficients_table_10y.csv"))
    if results_5y:
        df_5y = pd.DataFrame(results_5y).set_index("cid")
        df_5y.to_csv(os.path.join(OUTPUTS_DIR, "coefficients_table_5y.csv"))

    comp = build_comparison_table(results_10y)
    comp.to_csv(os.path.join(OUTPUTS_DIR, "comparison_vs_figure4.csv"), index=False)
    with open(os.path.join(OUTPUTS_DIR, "comparison_vs_figure4.md"), "w") as f:
        f.write(comparison_to_markdown(comp))

    # ---- [7/7] Validation report --------------------------------------
    print("\n[7/7] Writing validation report...")
    report_path = write_validation_report(results_10y, zscore_latest, OUTPUTS_DIR)

    # ---- Final summary -------------------------------------------------
    print("\n" + "=" * 72)
    print("DONE. Outputs written to:", OUTPUTS_DIR)
    if DRY_RUN:
        print(">>> REMINDER: DRY_RUN outputs are SYNTHETIC, not for analysis <<<")
    print("=" * 72)
    n_ok = len(results_10y)
    n_fail = len(CURRENCIES) - n_ok
    print(f"  currencies succeeded: {n_ok}/{len(CURRENCIES)}  failed: {n_fail}")
    print(f"  validation report: {report_path}")
    if df_10y is not None:
        print("\nFinal 10Y coefficient table (ours):")
        show_cols = [
            "adj_r2",
            "const",
            "excess_infl",
            "excess_empl",
            "excess_gdp",
            "policy_rate",
        ]
        print(df_10y[show_cols].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
