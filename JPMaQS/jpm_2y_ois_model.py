#!/usr/bin/env python
"""
J.P. Morgan Rates Strategy - 2Y OIS Fair Value Model (Replication)
==================================================================

Replicates the cross-market 2Y OIS fair value model from the J.P. Morgan
Global Markets Strategy note "Makes much more sense to live in the present
tense" (22 July 2026), Figure 4.

Sources (hybrid):
  (a) Free Macrosynergy Kaggle JPMaQS sample
      (daily quantamental indicators, 2000-2023, 23 currencies)
  (b) FRED public CSV endpoint for the variables NOT in the Kaggle free
      sample (unemployment, policy rates).

Outputs (all written to ./outputs/):
  - coefficients_table_10y.csv        our estimated coefficients + t-stats + Adj R2
  - coefficients_table_5y.csv         5Y "current regime" calibration (Figure 5 analogue)
  - comparison_vs_figure4.csv         side-by-side: ours vs published
  - comparison_vs_figure4.md          markdown rendering of the same
  - residuals_<CCY>.csv               daily actual / fitted / residual
  - residual_zscores.csv              latest 5Y rolling z-score per currency
  - actual_vs_fitted_<CCY>.png        Figure 8 analogue (8 plots)
  - residuals_<CCY>.png               Figure 6 analogue (8 plots)
  - residual_zscores_bar.png          Figure 7 analogue (cross-market bar)

Fidelity caveats (documented in VALIDATION_REPORT.md):
  - Dependent variable is the 2Y REAL IRS yield (RYLDIRS02Y_NSA) from Kaggle,
    NOT nominal 2Y OIS. This is the single biggest source of divergence.
  - 1Y inflation expectations are a CONSTRUCTED proxy (recent CPI blend with
    outlier dampening + partial return-to-target), NOT the proprietary
    JPMaQS INFE1Y_JA indicator.
  - Unemployment is FRED OECD harmonised monthly (quarterly for NZD),
    forward-filled onto the daily grid.
  - Regression window is 2013-12-14 to 2023-12-14 (the longest symmetric 10Y
    window inside the Kaggle sample), ~2.5y earlier than the note's window.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless backend - no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = "/mnt/nas/data4/github project/jefferies project/JPMaQS"
KAGGLE_CSV = "/tmp/jpmqs_data/JPMaQS_Quantamental_Indicators.csv"
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
FRED_CACHE_DIR = "/tmp/fred_cache"  # local cache so re-runs are offline-friendly

# ---------------------------------------------------------------------------
# Currencies and regression windows
# ---------------------------------------------------------------------------
CURRENCIES = ["USD", "EUR", "GBP", "SEK", "NOK", "AUD", "NZD", "JPY"]

# Kaggle sample ends 2023-12-14; longest symmetric 10Y window inside it.
WINDOW_10Y = ("2013-12-14", "2023-12-14")
WINDOW_5Y = ("2018-12-14", "2023-12-14")

# z-score lookback (business days ~ 5 years)
ZSCORE_WINDOW = 252 * 5

# Missing-data tolerance: drop a currency if any regressor is >20% missing
# inside the regression window.
MISSING_TOLERANCE = 0.20

# ---------------------------------------------------------------------------
# CONFIG - single source of truth for every variable.
# Swapping a Kaggle proxy for the real JPMaQS ticker is a 1-line edit here.
# ---------------------------------------------------------------------------
CONFIG = {
    # Dependent variable: 2Y REAL IRS yield from Kaggle. Regressed directly
    # this gives wrong-signed inflation coefficients (real yield falls when
    # inflation rises), so we PROMOTE the Fisher-equation nominal proxy below
    # to the primary dependent variable. The raw real yield is kept for the
    # diagnostic / honesty discussion in the validation report.
    "dep_var": {
        "source": "kaggle",
        "xcat": "RYLDIRS02Y_NSA",
        "note": "2Y real IRS yield - raw proxy; used to build the nominal proxy",
        "use_nominal_proxy": True,
        "nominal_proxy_note": (
            "nominal = real_2Y + 1Y_expected_inflation (Fisher). Restores "
            "the expected positive inflation coefficient sign vs Figure 4."
        ),
    },
    # Regressor 1: excess 1Y-ahead inflation expectations (%-pt).
    # Constructed from headline + core CPI with outlier dampening and a
    # partial return-to-target blend against the effective target.
    "excess_infl": {
        "source": "constructed",
        "inputs": ["CPIH_SA_P1M1ML12", "CPIC_SA_P1M1ML12", "INFTEFF_NSA"],
        "headline_xcat": "CPIC_SA_P1M1ML12",
        "core_xcat": "CPIH_SA_P1M1ML12",
        "target_xcat": "INFTEFF_NSA",
        "cpi_ma_window": 3,  # 3M MA of CPI prints
        "outlier_lookback": 252 * 3,  # 3Y rolling window for outlier detection
        "outlier_threshold_sigma": 2.0,
        "outlier_damp_weight": 0.5,  # deviations >2s shrunk to 50% weight
        "return_to_target_weight": 0.3,  # 30% weight on target (gradual reversion)
        "note": "Constructed proxy for JPMaQS INFE1Y_JA",
    },
    # Regressor 2: excess employment (%-pt) from FRED OECD harmonised
    # unemployment. Excess = -(unemp - 5Y MA of unemp) so positive = tight.
    "excess_empl": {
        "source": "fred",
        "series_per_cid": {
            "USD": "LRHUTTTTUSM156S",
            "EUR": "LRHUTTTTEZM156S",  # Euro Area (EZ) not EUR
            "GBP": "LRHUTTTTGBM156S",
            "SEK": "LRHUTTTTSEM156S",
            "NOK": "LRHUTTTTNOM156S",
            "AUD": "LRHUTTTTAUM156S",
            "NZD": "LRHUTTTTNZQ156S",  # quarterly fallback (monthly 404s)
            "JPY": "LRHUTTTTJPM156S",
        },
        "short_ma": 3,  # 3M MA
        "long_ma_months": 60,  # 5Y MA
        "note": "Inverse unemployment gap; NZD uses quarterly series",
    },
    # Regressor 3: excess technical GDP growth (%-pt). DIRECT Kaggle match -
    # INTRGDPv5Y is literally "intuitive GDP growth vs 5Y trend".
    "excess_gdp": {
        "source": "kaggle",
        "xcat": "INTRGDPv5Y_NSA_P1M1ML12_3MMA",
        "note": "Direct JPMaQS match - intuitive real GDP trend vs 5Y median",
    },
    # Regressor 4: policy rate (%) from FRED.
    "policy_rate": {
        "source": "fred",
        "series_per_cid": {
            "USD": "FEDFUNDS",  # monthly effective fed funds
            "EUR": "ECBDFR",  # ECB deposit facility (daily)
            "GBP": "IRSTCI01GBM156N",  # monthly fallback (BoE Bank Rate 404s)
            "SEK": "IRSTCI01SEM156N",
            "NOK": "IRSTCI01NOM156N",
            "AUD": "IRSTCI01AUM156N",
            "NZD": "IRSTCI01NZM156N",
            "JPY": "IRSTCI01JPM156N",
        },
        "note": "GBP uses IRSTCI01 (immediate rate) - BRCPCHF02GBM460S 404",
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
# dual-mandle / labour-sensitive -> expect POSITIVE employment coefficient.
DUAL_MANDATE = {"USD", "NOK", "AUD"}  # Fed, Norges Bank, RBA
INFL_ONLY_MANDATE = {"EUR", "GBP", "JPY"}  # ECB, BoE, BoJ


# ===========================================================================
# Data loaders
# ===========================================================================
def load_kaggle_panel(cids: list[str], xcats: list[str]) -> pd.DataFrame:
    """Load the Kaggle JPMaQS CSV and return a wide DataFrame.

    Returns a DataFrame indexed by (real_date) with columns
    [(cid, xcat), ...] in a MultiIndex; values are the 'value' field.
    """
    usecols = ["real_date", "cid", "xcat", "value"]
    print(f"   reading {KAGGLE_CSV} (~3.4M rows, this takes ~30s)...")
    df = pd.read_csv(
        KAGGLE_CSV,
        usecols=usecols,
        dtype={"cid": "string", "xcat": "string"},
        parse_dates=["real_date"],
    )
    df = df[df["cid"].isin(cids) & df["xcat"].isin(xcats)]
    df = df.dropna(subset=["value"])
    # pivot to wide: index=date, columns=(cid, xcat)
    panel = df.pivot_table(
        index="real_date", columns=["cid", "xcat"], values="value", aggfunc="last"
    )
    panel = panel.sort_index()
    return panel


def fetch_fred(series_id: str) -> pd.Series:
    """Download a FRED series via the public CSV endpoint, with local caching.

    Returns a Series indexed by observation_date (Timestamp), float values.
    On HTTP failure returns an empty Series (caller handles NaN).
    """
    os.makedirs(FRED_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(FRED_CACHE_DIR, f"{series_id}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=[0])
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            df = pd.read_csv(url)
        except Exception as exc:
            print(f"   [FRED FAIL] {series_id}: {exc}")
            return pd.Series(dtype="float64", name=series_id)
        df.to_csv(cache_path, index=False)
        df = pd.read_csv(cache_path, parse_dates=[0])
    # FRED CSVs: first column is the date, second is the value ('.' = missing)
    date_col = df.columns[0]
    val_col = df.columns[1]
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s.name = series_id
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


# ===========================================================================
# Variable constructors
# ===========================================================================
def get_kaggle_series(panel: pd.DataFrame, cid: str, xcat: str) -> pd.Series:
    """Extract a single (cid, xcat) series from the wide Kaggle panel."""
    try:
        s = panel[(cid, xcat)].copy()
    except KeyError:
        return pd.Series(dtype="float64", name=xcat)
    s.name = xcat
    return s


def construct_infl_expected_level(panel: pd.DataFrame, cid: str) -> pd.Series:
    """Construct the 1Y-ahead expected inflation LEVEL (%oya).

    Pipeline (documented choices - tune via CONFIG['excess_infl']):
      1. 3M moving average of headline and core CPI (%oya).
      2. Simple average of the two: recent_inflation.
      3. Outlier dampener: where recent_inflation deviates >2sigma from its
         3Y rolling mean, shrink the deviation to 50% weight.
      4. Partial return-to-target blend:
             1Y_expected = (1-w)*recent + w*target     (w = 0.30)

    This is an approximation of JPMaQS INFE1Y_JA. The exact proprietary
    jump-detection and credibility logic is not reproducible from free data;
    the dampener + blend captures the spirit (slow reversion to target,
    reduced sensitivity to outliers).

    Returns the LEVEL of expected inflation - needed both for the excess
    measure (level minus target) and for constructing a nominal-yield proxy
    via the Fisher equation (real yield + expected inflation).
    """
    cfg = CONFIG["excess_infl"]
    headline = get_kaggle_series(panel, cid, cfg["headline_xcat"])
    core = get_kaggle_series(panel, cid, cfg["core_xcat"])
    target = get_kaggle_series(panel, cid, cfg["target_xcat"])

    # align on common daily index
    idx = headline.index.union(core.index).union(target.index)
    headline = headline.reindex(idx)
    core = core.reindex(idx)
    target = target.reindex(idx)

    ma_win = cfg["cpi_ma_window"]
    head_ma = headline.rolling(ma_win, min_periods=1).mean()
    core_ma = core.rolling(ma_win, min_periods=1).mean()
    recent = 0.5 * (head_ma + core_ma)

    # outlier dampener on the recent-inflation series
    lookback = cfg["outlier_lookback"]
    roll_mean = recent.rolling(lookback, min_periods=60).mean()
    roll_std = recent.rolling(lookback, min_periods=60).std()
    dev = recent - roll_mean
    z = dev / roll_std.replace(0, np.nan)
    damp_w = cfg["outlier_damp_weight"]
    thr = cfg["outlier_threshold_sigma"]
    dampened = recent.copy()
    big = z.abs() > thr
    # for big deviations, replace value with mean + damp_w * deviation
    dampened.loc[big] = roll_mean.loc[big] + damp_w * dev.loc[big]
    dampened = dampened.fillna(recent)  # fallback in early warmup

    w = cfg["return_to_target_weight"]
    expected = (1.0 - w) * dampened + w * target
    expected.name = "infl_expected_level"
    return expected


def construct_excess_infl(panel: pd.DataFrame, cid: str) -> pd.Series:
    """Excess 1Y-ahead inflation expectations = expected level - target."""
    expected = construct_infl_expected_level(panel, cid)
    target = get_kaggle_series(panel, cid, CONFIG["excess_infl"]["target_xcat"])
    excess = expected - target.reindex(expected.index)
    excess.name = "excess_infl"
    return excess


def construct_excess_empl(unemp: pd.Series) -> pd.Series:
    """Excess employment = -(3M MA of unemployment - 5Y MA of unemployment).

    Positive value = labour market tight (unemployment below its 5Y norm).
    """
    cfg = CONFIG["excess_empl"]
    short = unemp.rolling(cfg["short_ma"], min_periods=1).mean()
    long = unemp.rolling(cfg["long_ma_months"], min_periods=12).mean()
    gap = short - long
    excess = -gap
    excess.name = "excess_empl"
    return excess


def construct_excess_gdp(panel: pd.DataFrame, cid: str) -> pd.Series:
    """Direct passthrough of INTRGDPv5Y (already excess vs 5Y trend)."""
    s = get_kaggle_series(panel, cid, CONFIG["excess_gdp"]["xcat"])
    s.name = "excess_gdp"
    return s


def construct_policy_rate(series_id: str) -> pd.Series:
    """FRED policy rate (passthrough; alignment/fill happens later)."""
    s = fetch_fred(series_id)
    s.name = "policy_rate"
    return s


def construct_dep_var(panel: pd.DataFrame, cid: str) -> pd.Series:
    """Dependent variable: 2Y real IRS yield (raw Kaggle proxy)."""
    s = get_kaggle_series(panel, cid, CONFIG["dep_var"]["xcat"])
    s.name = "dep_var_real"
    return s


def construct_dep_var_nominal(
    panel: pd.DataFrame, cid: str, infl_level: pd.Series
) -> pd.Series:
    """Nominal 2Y yield proxy via the Fisher equation.

        nominal_proxy = real_2Y_yield + 1Y_expected_inflation

    The Kaggle free sample does NOT contain a nominal 2Y OIS series, only the
    2Y real IRS yield (RYLDIRS02Y_NSA). Regressing a REAL yield on excess
    inflation produces a negative coefficient by construction (higher expected
    inflation mechanically lowers real yields), which is the opposite of the
    note's nominal-OIS result. Adding back the expected inflation level
    reconstructs a nominal-like yield so the inflation coefficient can take
    the economically-expected positive sign, matching Figure 4.

    This is the task's documented OPTIONAL enhancement, promoted to the
    primary dependent variable precisely because it is the only way to get a
    sign-comparable result vs the note.
    """
    real = get_kaggle_series(panel, cid, CONFIG["dep_var"]["xcat"])
    idx = real.index.union(infl_level.index)
    nominal = real.reindex(idx) + infl_level.reindex(idx)
    nominal.name = "dep_var"
    return nominal


# ===========================================================================
# Alignment onto a common business-day grid
# ===========================================================================
def build_business_day_grid(start: str, end: str) -> pd.DatetimeIndex:
    """Union of the Kaggle daily grid and the standard BDay calendar.

    We use the pandas BDay calendar restricted to [start, end] so the grid
    matches typical JPMaQS trading days.
    """
    grid = pd.bdate_range(start=start, end=end)
    return grid


def align_and_fill(
    series_dict: dict[str, pd.Series], grid: pd.DatetimeIndex
) -> pd.DataFrame:
    """Reindex every series onto the business-day grid and forward-fill.

    Backward-fill the first few rows if forward-fill leaves leading NaNs
    (common when FRED data starts after the grid start).
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
# OLS regression
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
    ax.plot(aligned.index, aligned["dep_var"], label="Actual (2Y real yield)", lw=1.4)
    ax.plot(
        aligned.index,
        aligned["fitted"],
        label="Fitted (fair value)",
        lw=1.4,
        alpha=0.85,
    )
    ax.set_title(f"{cid} 2Y real yield: actual vs fitted  (Adj R² = {adj_r2:.2f})")
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
    ax.set_title(f"{cid} model residual (actual - fitted)")
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
    ax.set_title("Latest 5Y z-score of model residual (Figure 7 analogue)")
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
    lines.append("# Coefficient comparison: ours vs Figure 4\n")
    lines.append(
        "Adj R² and coefficients side-by-side. Published figures from the "
        "22 July 2026 note, Figure 4 (10Y OLS, dependent = 2Y nominal OIS).\n"
    )
    lines.append(
        "| Ccy | Adj R² (ours) | Adj R² (pub) | "
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
# Main
# ===========================================================================
def main() -> int:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print("=" * 72)
    print("J.P. Morgan 2Y OIS fair value model - replication")
    print("=" * 72)

    # ---- [1/6] Load Kaggle panel --------------------------------------
    print("\n[1/6] Loading Kaggle JPMaQS data...")
    needed_xcats = sorted(
        set(
            [CONFIG["dep_var"]["xcat"], CONFIG["excess_gdp"]["xcat"]]
            + CONFIG["excess_infl"]["inputs"]
        )
    )
    panel = load_kaggle_panel(CURRENCIES, needed_xcats)
    print(f"   panel ready: {panel.shape[0]} days x {panel.shape[1]} (cid,xcat) cols")
    print(f"   date range: {panel.index.min().date()} .. {panel.index.max().date()}")

    # ---- [2/6] Fetch FRED ---------------------------------------------
    print("\n[2/6] Fetching FRED supplementary data...")
    unemp_series: dict[str, pd.Series] = {}
    policy_series: dict[str, pd.Series] = {}
    for cid in CURRENCIES:
        unemp_series[cid] = fetch_fred(CONFIG["excess_empl"]["series_per_cid"][cid])
        policy_series[cid] = fetch_fred(CONFIG["policy_rate"]["series_per_cid"][cid])
        u_ok = "ok" if not unemp_series[cid].empty else "MISSING"
        p_ok = "ok" if not policy_series[cid].empty else "MISSING"
        print(
            f"   {cid}: unemp={u_ok} "
            f"({len(unemp_series[cid])} obs), "
            f"policy={p_ok} ({len(policy_series[cid])} obs)"
        )

    # ---- [3/6] Build per-currency regression input --------------------
    print("\n[3/6] Constructing regressors and aligning to business-day grid...")
    grid = build_business_day_grid(WINDOW_10Y[0], WINDOW_10Y[1])
    print(
        f"   business-day grid: {len(grid)} days ({grid[0].date()}..{grid[-1].date()})"
    )

    per_cid_data: dict[str, pd.DataFrame] = {}
    for cid in CURRENCIES:
        infl_level = construct_infl_expected_level(panel, cid)
        infl = construct_excess_infl(panel, cid)
        if CONFIG["dep_var"]["use_nominal_proxy"]:
            dep = construct_dep_var_nominal(panel, cid, infl_level)
            dep_kind = "nominal proxy (real_2Y + 1Y infl exp)"
        else:
            dep = construct_dep_var(panel, cid)
            dep_kind = "real 2Y yield"
        if unemp_series[cid].empty:
            empl = pd.Series(dtype="float64", name="excess_empl")
        else:
            empl = construct_excess_empl(unemp_series[cid])
        gdp = construct_excess_gdp(panel, cid)
        if policy_series[cid].empty:
            pol = pd.Series(dtype="float64", name="policy_rate")
        else:
            pol = construct_policy_rate(CONFIG["policy_rate"]["series_per_cid"][cid])

        series_dict = {
            "dep_var": dep,
            "dep_var_real": construct_dep_var(panel, cid),
            "excess_infl": infl,
            "excess_empl": empl,
            "excess_gdp": gdp,
            "policy_rate": pol,
        }
        per_cid_data[cid] = align_and_fill(series_dict, grid)

    print(f"   dependent variable: {dep_kind}")

    # ---- [4/6] Run OLS ------------------------------------------------
    print("\n[4/6] Running OLS regressions (10Y baseline)...")
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
            f"   {cid}: Adj R²={res['adj_r2']:.3f}  "
            f"infl={res['excess_infl']:+.2f} (t={res['excess_infl_t']:+.1f})  "
            f"empl={res['excess_empl']:+.2f} (t={res['excess_empl_t']:+.1f})  "
            f"gdp={res['excess_gdp']:+.2f} (t={res['excess_gdp_t']:+.1f})  "
            f"pol={res['policy_rate']:+.2f} (t={res['policy_rate_t']:+.1f})"
        )

    # 5Y current-regime calibration (Figure 5 analogue)
    print("\n[4b/6] Running OLS regressions (5Y current regime)...")
    results_5y: list[dict] = []
    for cid in CURRENCIES:
        res, _ = run_ols(per_cid_data[cid], WINDOW_5Y, cid)
        if res is not None:
            results_5y.append(res)
            print(f"   {cid}: Adj R²={res['adj_r2']:.3f}")

    # ---- [5/6] Residuals, z-scores, plots -----------------------------
    print("\n[5/6] Computing residual z-scores and saving diagnostic plots...")
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

    # ---- [6/6] Save coefficient tables --------------------------------
    print("\n[6/6] Saving coefficient tables and comparison vs Figure 4...")
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

    # ---- Final summary -------------------------------------------------
    print("\n" + "=" * 72)
    print("DONE. Outputs written to:", OUTPUTS_DIR)
    print("=" * 72)
    n_ok = len(results_10y)
    n_fail = len(CURRENCIES) - n_ok
    print(f"  currencies succeeded: {n_ok}/{len(CURRENCIES)}  failed: {n_fail}")
    if results_10y:
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
