"""
DGTW (Daniel, Grinblatt, Titman, Wermers 1997) characteristic-matched benchmark
returns.

Methodology:
    1. Each firm is sorted independently on three characteristics into quintiles:
         - size       (market equity)
         - book-to-market
         - momentum   (12-2 month past return, skipping the most recent month)
       yielding 5×5×5 = 125 "DGTW" benchmark portfolios.
    2. Breakpoints are NYSE-style percentile cutoffs (percentiles of the cross
       section) computed using ONLY data available as of the formation date —
       no lookahead.
    3. Each bucket's return is the value-weighted average of its member firms.
    4. A firm's DGTW-adjusted abnormal return = R_firm − R_its_DGTW_bucket.

DGTW is more robust than FF5 regression for event studies because it matches on
firm characteristics directly (not just factor loadings), so it controls for the
size/BM/momentum composition that confounds the PEAD asymmetry ratio. Because
the synthetic data injects the asymmetry behaviorally (directly into returns
with no characteristic tilt), DGTW adjustment should NOT remove the injected
signal — a useful sanity check.

Reference: Daniel, Grinblatt, Titman, Wermers (1997), "Measuring Mutual Fund
           Performance with Characteristic-Based Benchmarks," JF 52(3).
Reference impl: github.com/Chihche-Liew/DGTW-Portfolio
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from pead.schema import Col

# ─── Characteristic / output column names (mirror schema.py) ─────────────────

TICKER = Col.TICKER
TRADING_DATE = Col.TRADING_DATE
PX_CLOSE = Col.PX_CLOSE
VOLUME = Col.VOLUME
RET = Col.RET
ANNOUNCE_DATE = Col.ANNOUNCE_DATE

MARKET_CAP = Col.MARKET_CAP  # "market_cap"
BTM = Col.BOOK_TO_MARKET  # "btm"
MOM_12_2 = Col.MOMENTUM_12_2  # "mom_12_2"
DGTW_BUCKET = Col.DGTW_BUCKET  # "dgtw_bucket"

# Internal bucket-rank column names.
COL_DGTW_SIZE = "dgtw_size"
COL_DGTW_BM = "dgtw_bm"
COL_DGTW_MOM = "dgtw_mom"

# Defaults pulled from config/config.yaml -> benchmarks.dgtw.
DEFAULT_N_SIZE = 5
DEFAULT_N_BM = 5
DEFAULT_N_MOM = 5

# Momentum formation window (months): cumulative return from t-12m to t-2m.
MOMENTUM_FORMATION_MONTHS = 12
MOMENTUM_SKIP_MONTHS = 2

# Window (in trading days) over which to average volume for the market-cap proxy.
VOLUME_LOOKBACK_DAYS = 60

# Scale factor for the synthetic market-cap proxy: px_close * volume / SCALE.
# Using a rolling-average volume stabilises the proxy relative to a single day.
MARKET_CAP_SCALE = 10_000.0

# Seed for the deterministic per-firm book-to-market proxy.
BTM_PROXY_SEED = 20240517


# ─── Panel construction ──────────────────────────────────────────────────────


def _dedupe_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (ticker, trading_date).

    The synthetic generator emits independent, overlapping price windows around
    each earnings event, so the same (ticker, date) can appear several times
    with different realisations. We keep the first occurrence to recover a
    single deterministic price history per ticker. Applied at every entry point
    so the panel-based and raw-price code paths see identical data.
    """
    if prices.duplicated(subset=[TICKER, TRADING_DATE]).any():
        prices = prices.drop_duplicates(subset=[TICKER, TRADING_DATE], keep="first").reset_index(
            drop=True
        )
    return prices


def _build_panels(prices: pd.DataFrame) -> dict[str, Any]:
    """Pivot the long price table into date × ticker panels.

    Returns a dict with ``close``, ``vol``, ``ret`` panels (sorted by trading
    date), the ordered ``tickers`` list, and a deterministic per-ticker
    book-to-market ``btm`` mapping.
    """
    prices = _dedupe_prices(prices)
    close = prices.pivot_table(index=TRADING_DATE, columns=TICKER, values=PX_CLOSE).sort_index()
    vol = prices.pivot_table(index=TRADING_DATE, columns=TICKER, values=VOLUME).sort_index()
    if RET in prices.columns:
        ret = prices.pivot_table(index=TRADING_DATE, columns=TICKER, values=RET).sort_index()
    else:
        # Fall back to close-to-close returns if no `ret` column supplied.
        ret = close.pct_change()

    tickers = list(close.columns)
    btm_map = _stable_btm_proxy(tickers)
    return {"close": close, "vol": vol, "ret": ret, "tickers": tickers, "btm": btm_map}


def _stable_btm_proxy(tickers: list[str]) -> dict[str, float]:
    """Generate a random but *persistent* book-to-market value per ticker.

    Real B/M requires book equity from fundamentals, which the synthetic price
    panel does not carry. We therefore synthesise a stable, firm-specific B/M
    draw (range ~[0.1, 1.0]) so each firm's bucket is consistent over time and
    across formation dates.
    """
    rng = np.random.default_rng(BTM_PROXY_SEED)
    unique = sorted(set(tickers))
    vals = rng.uniform(0.1, 1.0, size=len(unique))
    return dict(zip(unique, vals))


def _asof_row(panel: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Return the last available row of *panel* on or before *asof*.

    This is the no-lookahead primitive: for a formation date ``asof`` we only
    ever read data that had materialised by then. Returns an all-NaN Series if
    no history exists.
    """
    sub = panel.loc[panel.index <= asof]
    if len(sub) == 0:
        return pd.Series(np.nan, index=panel.columns)
    return sub.iloc[-1]


# ─── Public: characteristics ─────────────────────────────────────────────────


def compute_firm_characteristics(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    char_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute DGTW characteristics for the event universe as of ``char_date``.

    All inputs use only data on or before ``char_date`` (no lookahead):
      - ``market_cap``: ``px_close * avg_volume_60d / 10000`` — a synthetic
        scale proxy (real data would use ``price × shares`` from fundamentals).
      - ``btm``: a deterministic per-firm random book-to-market proxy (real data
        would use ``book_equity / market_equity``).
      - ``mom_12_2``: cumulative return from t-12m to t-2m (12-2 momentum),
        computed from the price history.

    Args:
        prices: long daily price frame (ticker, trading_date, px_close, volume).
        events: frame whose ``ticker`` column defines the firm universe.
        char_date: as-of formation date.

    Returns:
        DataFrame with columns ``ticker, market_cap, btm, mom_12_2`` — one row
        per firm with sufficient history.
    """
    panels = _build_panels(prices)
    tickers = list(pd.Series(events[TICKER].unique()))
    return _characteristics_from_panels(panels, char_date, tickers=tickers)


def _characteristics_from_panels(
    panels: dict[str, Any],
    char_date: pd.Timestamp,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Vectorised characteristic computation from pre-built panels.

    Reused by both :func:`compute_firm_characteristics` (single as-of date) and
    the benchmark-formation loop (many formation dates) so the heavy pivot step
    is paid only once.
    """
    close = panels["close"]
    vol = panels["vol"]
    btm_map = panels["btm"]

    char_date = pd.Timestamp(char_date)
    p_now = _asof_row(close, char_date)

    # Rolling-average volume over the lookback window (stabilises the mcap proxy).
    past_vol = vol.loc[vol.index <= char_date]
    avg_vol = past_vol.tail(VOLUME_LOOKBACK_DAYS).mean()

    # 12-2 momentum: return from t-12m to t-2m, skipping the most recent 2 months.
    date_2m = char_date - pd.DateOffset(months=MOMENTUM_SKIP_MONTHS)
    date_12m = char_date - pd.DateOffset(months=MOMENTUM_FORMATION_MONTHS)
    p_2m = _asof_row(close, date_2m)
    p_12m = _asof_row(close, date_12m)
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = p_2m / p_12m - 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        mcap = p_now * avg_vol / MARKET_CAP_SCALE

    btm = pd.Series([btm_map.get(t, np.nan) for t in close.columns], index=close.columns)

    df = pd.DataFrame(
        {
            TICKER: list(close.columns),
            MARKET_CAP: mcap.values,
            BTM: btm.values,
            MOM_12_2: mom.values,
        }
    )

    if tickers is not None:
        wanted = set(tickers)
        df = df[df[TICKER].isin(wanted)].reset_index(drop=True)

    # Drop firms without a usable market cap or momentum (insufficient history).
    df = df.dropna(subset=[MARKET_CAP, MOM_12_2]).reset_index(drop=True)
    return df


# ─── Public: bucket assignment ───────────────────────────────────────────────


def _quantile_assign(series: pd.Series, n: int) -> pd.Series:
    """Assign each value to bucket 0..n-1 using NYSE-style percentile cutoffs.

    Implemented via fractional ranks so the result is robust to ties (always
    yields exactly ``n`` buckets for continuous data) and equivalent to cutting
    at the [1/n, 2/n, ..., (n-1)/n] quantiles of the cross-section.
    """
    out = pd.Series(np.nan, index=series.index, dtype=float)
    mask = series.notna() & np.isfinite(series.values)
    valid = series.loc[mask]
    if valid.empty:
        return out
    ranks = valid.rank(method="average")
    count = float(len(valid))
    labels = np.floor((ranks - 1.0) / count * n).astype(float)
    labels = labels.clip(upper=n - 1.0)
    out.loc[mask] = labels.values
    return out


def assign_dgtw_buckets(
    characteristics: pd.DataFrame,
    n_size: int = DEFAULT_N_SIZE,
    n_bm: int = DEFAULT_N_BM,
    n_momentum: int = DEFAULT_N_MOM,
) -> pd.DataFrame:
    """Assign each firm to one of ``n_size × n_bm × n_momentum`` DGTW buckets.

    Sorting is *independent* on each characteristic (not sequential): each firm
    receives a 0-based rank for size, BM and momentum, then the three ranks are
    combined into a single bucket id.

    Adds columns: ``dgtw_size, dgtw_bm, dgtw_mom`` (each 0..n-1) and
    ``dgtw_bucket`` in ``[0, n_size*n_bm*n_momentum - 1]``.
    """
    df = characteristics.copy()
    df[COL_DGTW_SIZE] = _quantile_assign(df[MARKET_CAP], n_size)
    df[COL_DGTW_BM] = _quantile_assign(df[BTM], n_bm)
    df[COL_DGTW_MOM] = _quantile_assign(df[MOM_12_2], n_momentum)

    # Combine: size is the most-significant index, then BM, then momentum.
    df[DGTW_BUCKET] = (
        df[COL_DGTW_SIZE] * (n_bm * n_momentum) + df[COL_DGTW_BM] * n_momentum + df[COL_DGTW_MOM]
    )
    return df


# ─── Public: benchmark returns ───────────────────────────────────────────────


def compute_dgtw_benchmark_returns(
    prices: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Value-weighted return of each DGTW bucket over time.

    Uses a single (static) assignment table — i.e. one formation snapshot held
    across the whole sample. This matches the documented contract; the
    time-varying monthly formation used inside
    :func:`compute_dgtw_adjusted_returns` is the production path.

    Args:
        prices: long daily price frame with ``ticker, trading_date, ret``.
        assignments: frame with ``ticker, dgtw_bucket, market_cap`` (a single
            formation snapshot).

    Returns:
        Long frame with columns ``trading_date, dgtw_bucket, benchmark_ret``.
    """
    required = {TICKER, DGTW_BUCKET, MARKET_CAP}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"assignments missing required columns: {sorted(missing)}")

    prices = _dedupe_prices(prices)
    ret = prices[[TRADING_DATE, TICKER, RET]].copy()
    weights = assignments[[TICKER, DGTW_BUCKET, MARKET_CAP]].copy()

    merged = ret.merge(weights, on=TICKER, how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=[RET, DGTW_BUCKET, MARKET_CAP])
    if merged.empty:
        return pd.DataFrame(columns=[TRADING_DATE, DGTW_BUCKET, "benchmark_ret"])

    merged["_wret"] = merged[RET] * merged[MARKET_CAP]
    grouped = merged.groupby([TRADING_DATE, DGTW_BUCKET], observed=True)
    num = grouped["_wret"].sum()
    den = grouped[MARKET_CAP].sum()
    bench = (num / den).reset_index()
    bench.columns = [TRADING_DATE, DGTW_BUCKET, "benchmark_ret"]
    return bench.sort_values([TRADING_DATE, DGTW_BUCKET]).reset_index(drop=True)


# ─── Public: DGTW-adjusted event returns ─────────────────────────────────────


def compute_dgtw_adjusted_returns(
    event_returns: pd.DataFrame,
    prices: pd.DataFrame,
    n_size: int = DEFAULT_N_SIZE,
    n_bm: int = DEFAULT_N_BM,
    n_momentum: int = DEFAULT_N_MOM,
) -> pd.DataFrame:
    """Master DGTW adjustment for event-study returns.

    Pipeline:
      1. Parse each ``car_*`` window column (named ``car_{start}_{end}``).
      2. Build date × ticker return/price panels once.
      3. Form DGTW buckets at every quarter-end (using only data up to that
         date) for the full cross-section — no lookahead.
      4. Compute daily value-weighted returns per bucket, rolling assignments
         forward within each quarter.
      5. For each event, locate the firm's bucket at announcement (most recent
         formation) and sum the bucket's daily returns over the same window to
         get a benchmark CAR. DGTW-adjusted CAR = firm CAR − benchmark CAR.

    This is the most robust risk adjustment for the asymmetry test because
    misses and beats differ in size/BM/momentum; DGTW nets that out so the
    residual is behavioural drift.

    Args:
        event_returns: frame with ``ticker, announce_date`` and one or more
            ``car_{start}_{end}`` columns (cumulative abnormal returns over
            trading-day window [start, end]).
        prices: long daily price frame.

    Returns:
        Copy of ``event_returns`` with an added ``dgtw_adjusted_car_{window}``
        column for every input window.
    """
    car_cols, windows = _parse_car_columns(event_returns)
    if not car_cols:
        return event_returns.copy()

    panels = _build_panels(prices)
    ret_panel: pd.DataFrame = panels["ret"]
    trading_dates = pd.DatetimeIndex(ret_panel.index.sort_values())
    dates_arr = trading_dates.values  # numpy datetime64 for searchsorted

    # ── Quarterly formation of buckets for the full cross-section ───────────
    form_dates = _quarter_end_dates(trading_dates)
    assignments_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for fd in form_dates:
        chars = _characteristics_from_panels(panels, fd)
        if chars.empty:
            continue
        assigns = assign_dgtw_buckets(chars, n_size, n_bm, n_momentum)
        assignments_by_date[pd.Timestamp(fd)] = assigns[[TICKER, DGTW_BUCKET, MARKET_CAP]].copy()

    if not assignments_by_date:
        # No formation possible — return frame with NaN adjustments.
        out = event_returns.copy()
        for col in car_cols:
            out[f"dgtw_adjusted_{col}"] = np.nan
        return out

    # ── Daily value-weighted benchmark return per bucket, rolling formation ─
    benchmark = _build_rolling_benchmark_returns(ret_panel, assignments_by_date)
    if benchmark.empty:
        out = event_returns.copy()
        for col in car_cols:
            out[f"dgtw_adjusted_{col}"] = np.nan
        return out

    # ── Cumulative benchmark return per bucket for O(1) window summation ────
    bench_pivot = (
        benchmark.pivot_table(index=TRADING_DATE, columns=DGTW_BUCKET, values="benchmark_ret")
        .reindex(trading_dates)
        .fillna(0.0)
        .sort_index()
    )
    cumret = bench_pivot.cumsum()
    cum_arr = cumret.values  # shape (n_dates, n_buckets)
    n_dates, n_buckets = cum_arr.shape
    form_list = sorted(assignments_by_date.keys())

    # ── Map each event to (announce position, firm bucket) ──────────────────
    out = event_returns.copy()
    ev_tickers = out[TICKER].values
    ev_dates = pd.to_datetime(out[ANNOUNCE_DATE].values)
    announce_pos = _announce_positions(ev_dates, dates_arr)
    event_buckets = np.array(
        [
            _event_bucket(str(tkr), pd.Timestamp(d), assignments_by_date, form_list)
            for tkr, d in zip(ev_tickers, ev_dates)
        ],
        dtype=float,
    )

    # ── Benchmark CAR per window → subtract from firm CAR ───────────────────
    for col in car_cols:
        start, end = windows[col]
        start_pos = announce_pos + start
        end_pos = announce_pos + end
        bench_car = _gather_window_car(cum_arr, event_buckets, start_pos, end_pos, n_dates)
        out[f"dgtw_adjusted_{col}"] = out[col].astype(float).values - bench_car

    return out


# ─── Internal helpers (formation / matching) ─────────────────────────────────


def _parse_car_columns(event_returns: pd.DataFrame) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Identify ``car_{start}_{end}`` columns and extract their trading-day windows."""
    car_cols = [c for c in event_returns.columns if str(c).startswith("car_")]
    windows: dict[str, tuple[int, int]] = {}
    for col in car_cols:
        nums = re.findall(r"-?\d+", str(col))
        if len(nums) >= 2:
            windows[col] = (int(nums[0]), int(nums[1]))
        else:
            windows[col] = (0, 0)
    return car_cols, windows


def _quarter_end_dates(trading_dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each calendar quarter present in the panel."""
    s = pd.Series(pd.DatetimeIndex(trading_dates))
    last_per_q = s.groupby(s.dt.to_period("Q")).max()
    return [pd.Timestamp(d) for d in last_per_q.values]


def _build_rolling_benchmark_returns(
    ret_panel: pd.DataFrame,
    assignments_by_date: dict[pd.Timestamp, pd.DataFrame],
) -> pd.DataFrame:
    """Daily value-weighted bucket returns, re-forming buckets each quarter.

    Within the holding period following a formation date, every ticker keeps the
    bucket it was assigned at formation, and each bucket's daily return is the
    market-cap-weighted average of its members.
    """
    form_list = sorted(assignments_by_date.keys())
    panel_dates = ret_panel.index
    chunks: list[pd.DataFrame] = []

    for i, fd in enumerate(form_list):
        next_fd = form_list[i + 1] if i + 1 < len(form_list) else panel_dates[-1]
        assigns = assignments_by_date[fd]
        period_mask = (panel_dates > fd) & (panel_dates <= next_fd)
        period_ret = ret_panel.loc[period_mask]
        if period_ret.empty:
            continue

        long = period_ret.stack().reset_index()
        long.columns = [TRADING_DATE, TICKER, RET]
        merged = long.merge(assigns, on=TICKER, how="inner")
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[RET, DGTW_BUCKET, MARKET_CAP]
        )
        if merged.empty:
            continue

        merged["_wret"] = merged[RET] * merged[MARKET_CAP]
        grouped = merged.groupby([TRADING_DATE, DGTW_BUCKET], observed=True)
        num = grouped["_wret"].sum()
        den = grouped[MARKET_CAP].sum()
        bench = (num / den).reset_index()
        bench.columns = [TRADING_DATE, DGTW_BUCKET, "benchmark_ret"]
        chunks.append(bench)

    if not chunks:
        return pd.DataFrame(columns=[TRADING_DATE, DGTW_BUCKET, "benchmark_ret"])
    return pd.concat(chunks, ignore_index=True)


def _announce_positions(ev_dates: np.ndarray, dates_arr: np.ndarray) -> np.ndarray:
    """Position in ``dates_arr`` of the first trading day on/after each event.

    Events announced on a non-trading day are aligned to the next trading day
    (the conventional event-study convention for day-0).
    """
    positions = np.searchsorted(dates_arr, ev_dates, side="left")
    return positions


def _event_bucket(
    ticker: str,
    announce_date: pd.Timestamp,
    assignments_by_date: dict[pd.Timestamp, pd.DataFrame],
    form_list: list[pd.Timestamp],
) -> float:
    """Most-recent formation-date DGTW bucket for ``ticker`` on/before the event."""
    fd = None
    for f in reversed(form_list):
        if f <= announce_date:
            fd = f
            break
    if fd is None:
        return np.nan
    assigns = assignments_by_date[fd]
    row = assigns.loc[assigns[TICKER] == ticker, DGTW_BUCKET]
    if row.empty:
        return np.nan
    return float(row.iloc[0])


def _gather_window_car(
    cum_arr: np.ndarray,
    buckets: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    n_dates: int,
) -> np.ndarray:
    """Benchmark CAR = cumret[end] − cumret[start-1] for each event.

    Handles out-of-sample windows by clamping to the available date range; events
    whose window falls entirely outside the panel receive NaN.
    """
    n = len(buckets)
    car = np.full(n, np.nan, dtype=float)
    for i in range(n):
        b = buckets[i]
        if not np.isfinite(b):
            continue
        bucket_idx = int(b)
        if bucket_idx < 0 or bucket_idx >= cum_arr.shape[1]:
            continue
        sp = int(start_pos[i])
        ep = int(end_pos[i])
        # Clamp into the available panel range.
        sp_c = max(sp, 0)
        ep_c = min(ep, n_dates - 1)
        if ep_c < 0 or sp_c > n_dates - 1 or sp_c > ep_c:
            continue
        prev = cum_arr[sp_c - 1, bucket_idx] if sp_c - 1 >= 0 else 0.0
        car[i] = cum_arr[ep_c, bucket_idx] - prev
    return car


__all__ = [
    "compute_firm_characteristics",
    "assign_dgtw_buckets",
    "compute_dgtw_benchmark_returns",
    "compute_dgtw_adjusted_returns",
    "DEFAULT_N_SIZE",
    "DEFAULT_N_BM",
    "DEFAULT_N_MOM",
    "MOMENTUM_FORMATION_MONTHS",
    "MOMENTUM_SKIP_MONTHS",
]
