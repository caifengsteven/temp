"""
Liquidity double-sort for the PEAD asymmetry analysis.

The Reddit claim (4.5x asymmetry ratio at T+20) may be a composition effect:
if large misses happen to occur in more illiquid stocks (whose returns are
noisier and whose drift is mechanically larger), the 4.5x ratio is not a
behavioural phenomenon but an artifact of sorting.

We test this by **double-sorting**: first sort events into liquidity buckets
(by Amihud illiquidity or by market cap), then compute the asymmetry ratio
**within** each bucket. If the ratio converges toward 1.0 within buckets,
the asymmetry is composition-driven. If it stays high, the asymmetry is
behavioural.

Reference: Amihud (2002) for the illiquidity measure; Bartov, Krinsky &
           Kremers (2000) for the size/institutional-ownership proxy;
           Diether, Malloy & Scherbina (2002) for dispersion-based tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pead.schema import Col


def compute_amihud_by_event(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    estimation_days: int = 252,
) -> pd.DataFrame:
    """Compute Amihud (2002) illiquidity for each event.

    For each event we measure illiquidity over the estimation window
    ``[-estimation_days - 1, -1]`` trading days relative to ``announce_date``
    (i.e. the ``estimation_days`` trading days strictly before the
    announcement). Using a pre-announcement window ensures the measure is not
    contaminated by the announcement itself.

    .. math::
        \\text{Amihud}_i = \\frac{1}{T} \\sum_{t} \\frac{|R_{i,t}|}{\\text{DVOL}_{i,t}}

    where ``DVOL = px_close * volume`` is dollar volume.

    Args:
        prices: Daily prices with at least
            ``[ticker, trading_date, px_close, volume, ret]``.
        events: Events with ``[ticker, announce_date]``.
        estimation_days: Number of pre-announcement trading days to use.

    Returns:
        DataFrame with columns ``[ticker, announce_date, amihud]``.
    """
    required = [Col.TICKER, Col.TRADING_DATE, Col.PX_CLOSE, Col.VOLUME, Col.RET]
    missing = [c for c in required if c not in prices.columns]
    if missing:
        raise ValueError(f"prices missing required columns: {missing}")

    ev_missing = [c for c in (Col.TICKER, Col.ANNOUNCE_DATE) if c not in events.columns]
    if ev_missing:
        raise ValueError(f"events missing required columns: {ev_missing}")

    p = prices[list(required)].copy()
    p[Col.TRADING_DATE] = pd.to_datetime(p[Col.TRADING_DATE])
    p[Col.PX_CLOSE] = pd.to_numeric(p[Col.PX_CLOSE], errors="coerce")
    p[Col.VOLUME] = pd.to_numeric(p[Col.VOLUME], errors="coerce")
    p[Col.RET] = pd.to_numeric(p[Col.RET], errors="coerce")

    # Dollar volume and daily illiquidity. Guard against zero/inf/negative
    # dollar volumes (split-adjustment artifacts, no-trade days).
    dvol = p[Col.PX_CLOSE] * p[Col.VOLUME]
    dvol = dvol.where(dvol > 0, np.nan)
    illiq_daily = (p[Col.RET].abs() / dvol).replace([np.inf, -np.inf], np.nan)

    p = p[[Col.TICKER, Col.TRADING_DATE]].assign(_illiq=illiq_daily)
    p = p.sort_values([Col.TICKER, Col.TRADING_DATE])

    # Pre-index per-ticker series for fast slicing in the event loop.
    grouped: dict[str, pd.DataFrame] = {tkr: g for tkr, g in p.groupby(Col.TICKER, sort=False)}

    ev = events[[Col.TICKER, Col.ANNOUNCE_DATE]].copy()
    ev[Col.ANNOUNCE_DATE] = pd.to_datetime(ev[Col.ANNOUNCE_DATE])

    out_rows: list[dict] = []
    for _, row in ev.iterrows():
        tkr = row[Col.TICKER]
        announce = row[Col.ANNOUNCE_DATE]
        gp = grouped.get(tkr)
        if gp is None or gp.empty:
            out_rows.append(
                {Col.TICKER: tkr, Col.ANNOUNCE_DATE: announce, Col.AMIHUD_ILLIQUIDITY: float("nan")}
            )
            continue
        prior = gp.loc[gp[Col.TRADING_DATE] < announce].tail(estimation_days)
        amihud = float(prior["_illiq"].mean()) if not prior.empty else float("nan")
        out_rows.append(
            {Col.TICKER: tkr, Col.ANNOUNCE_DATE: announce, Col.AMIHUD_ILLIQUIDITY: amihud}
        )

    return pd.DataFrame(out_rows, columns=[Col.TICKER, Col.ANNOUNCE_DATE, Col.AMIHUD_ILLIQUIDITY])


def _compute_size_by_event(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    estimation_days: int = 252,
) -> pd.DataFrame:
    """Compute a market-cap proxy for each event.

    When shares outstanding are not directly available we proxy size by the
    average dollar volume over the pre-announcement estimation window. This
    is monotonically related to firm size (larger firms trade more dollars)
    and is the standard fallback in CRSP-based event studies.

    Returns:
        DataFrame with ``[ticker, announce_date, size]``.
    """
    required = [Col.TICKER, Col.TRADING_DATE, Col.PX_CLOSE, Col.VOLUME]
    missing = [c for c in required if c not in prices.columns]
    if missing:
        raise ValueError(f"prices missing required columns: {missing}")

    p = prices[list(required)].copy()
    p[Col.TRADING_DATE] = pd.to_datetime(p[Col.TRADING_DATE])
    p[Col.PX_CLOSE] = pd.to_numeric(p[Col.PX_CLOSE], errors="coerce")
    p[Col.VOLUME] = pd.to_numeric(p[Col.VOLUME], errors="coerce")
    dvol = (p[Col.PX_CLOSE] * p[Col.VOLUME]).where(
        (p[Col.PX_CLOSE] > 0) & (p[Col.VOLUME] > 0), np.nan
    )
    p = p[[Col.TICKER, Col.TRADING_DATE]].assign(_dvol=dvol)
    p = p.sort_values([Col.TICKER, Col.TRADING_DATE])
    grouped: dict[str, pd.DataFrame] = {tkr: g for tkr, g in p.groupby(Col.TICKER, sort=False)}

    ev = events[[Col.TICKER, Col.ANNOUNCE_DATE]].copy()
    ev[Col.ANNOUNCE_DATE] = pd.to_datetime(ev[Col.ANNOUNCE_DATE])

    out_rows: list[dict] = []
    for _, row in ev.iterrows():
        tkr = row[Col.TICKER]
        announce = row[Col.ANNOUNCE_DATE]
        gp = grouped.get(tkr)
        if gp is None or gp.empty:
            out_rows.append({Col.TICKER: tkr, Col.ANNOUNCE_DATE: announce, "size": float("nan")})
            continue
        prior = gp.loc[gp[Col.TRADING_DATE] < announce].tail(estimation_days)
        size = float(prior["_dvol"].mean()) if not prior.empty else float("nan")
        out_rows.append({Col.TICKER: tkr, Col.ANNOUNCE_DATE: announce, "size": size})

    return pd.DataFrame(out_rows, columns=[Col.TICKER, Col.ANNOUNCE_DATE, "size"])


def _bucketize(series: pd.Series, n_buckets: int, label: str) -> pd.Series:
    """Assign 1..n_buckets by quantile. Falls back to equal-width if qcut fails."""
    s = series.dropna()
    if s.nunique() < n_buckets or len(s) < n_buckets:
        # Not enough variation for n_buckets quantiles — return NaNs to drop later.
        return pd.Series(np.full(len(series), np.nan), index=series.index, name=label)
    try:
        buckets = pd.qcut(series, n_buckets, labels=False, duplicates="drop")
    except ValueError:
        buckets = pd.cut(series, n_buckets, labels=False, include_lowest=True)
    buckets = buckets.astype("float") + 1.0
    return buckets.rename(label)


def _double_sort(
    event_returns: pd.DataFrame,
    sort_col: str,
    return_col: str,
    decile_col: str,
    n_buckets: int,
    bucket_label: str,
) -> pd.DataFrame:
    """Shared double-sort engine.

    For each bucket of ``sort_col``, compute the miss/beat ratio using
    ``decile_col`` and ``return_col``.
    """
    df = event_returns.copy()
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df[decile_col] = pd.to_numeric(df[decile_col], errors="coerce")
    df[bucket_label] = _bucketize(df[sort_col], n_buckets, bucket_label)
    df = df.dropna(subset=[bucket_label, decile_col, return_col])
    if df.empty:
        return pd.DataFrame(
            columns=[
                bucket_label,
                f"{sort_col}_median",
                "miss_car",
                "beat_car",
                "ratio",
                "n",
                "n_miss",
                "n_beat",
            ]
        )

    rows: list[dict] = []
    for bucket, g in df.groupby(bucket_label, sort=True):
        miss = g.loc[g[decile_col] == 1, return_col]
        beat = g.loc[g[decile_col] == g[decile_col].max(), return_col]
        # NB: ``g[decile_col].max()`` resolves the top decile even when the
        # bucket only contains a subset of decile values; in the well-formed
        # case it equals ``n_deciles`` (10).
        miss_car = float(miss.mean()) if len(miss) else float("nan")
        beat_car = float(beat.mean()) if len(beat) else float("nan")
        if np.isfinite(beat_car) and beat_car != 0.0 and np.isfinite(miss_car):
            ratio = abs(miss_car) / abs(beat_car)
        else:
            ratio = float("nan")
        rows.append(
            {
                bucket_label: int(bucket),
                f"{sort_col}_median": float(g[sort_col].median()),
                "miss_car": miss_car,
                "beat_car": beat_car,
                "ratio": float(ratio),
                "n": int(len(g)),
                "n_miss": int(len(miss)),
                "n_beat": int(len(beat)),
            }
        )
    return pd.DataFrame(rows).sort_values(bucket_label).reset_index(drop=True)


def double_sort_asymmetry(
    event_returns: pd.DataFrame,
    prices: pd.DataFrame,
    return_col: str = "car_short_drift",
    n_liquidity_buckets: int = 5,
    decile_col: str = "sue3_decile",
    n_deciles: int = 10,
) -> pd.DataFrame:
    """CRITICAL TEST: compute the asymmetry ratio WITHIN liquidity buckets.

    Process:
      1. Compute Amihud illiquidity for each event (pre-announcement window).
      2. Merge Amihud into ``event_returns`` on ``[ticker, announce_date]``.
      3. Sort events into ``n_liquidity_buckets`` by Amihud.
      4. Within each liquidity bucket, compute ``|miss drift| / |beat drift|``.

    Hypothesis: if the 4.5x ratio is driven by composition (misses happen in
    more illiquid stocks whose drift is mechanically larger), the ratio should
    converge toward 1.0 within liquidity buckets. If it stays high, the
    asymmetry is behavioural.

    Args:
        event_returns: DataFrame with ``[ticker, announce_date, decile_col, return_col]``.
        prices: Daily prices used to compute Amihud.
        return_col: CAR column to test (default ``car_short_drift`` — T+20).
        n_liquidity_buckets: Number of Amihud buckets (default 5 = quintiles).
        decile_col: Column holding the 1..``n_deciles`` assignments.
        n_deciles: Total number of deciles; the top decile value defines a beat.

    Returns:
        DataFrame with one row per liquidity bucket and columns
        ``[liquidity_bucket, amihud_median, miss_car, beat_car, ratio, n, n_miss, n_beat]``.
        Buckets are sorted from most liquid (1) to most illiquid (n).
    """
    if Col.AMIHUD_ILLIQUIDITY in event_returns.columns:
        # Allow callers to pre-compute Amihud and pass it through.
        amihud_df = event_returns[
            [Col.TICKER, Col.ANNOUNCE_DATE, Col.AMIHUD_ILLIQUIDITY]
        ].drop_duplicates(subset=[Col.TICKER, Col.ANNOUNCE_DATE])
    else:
        events = event_returns[[Col.TICKER, Col.ANNOUNCE_DATE]].drop_duplicates()
        amihud_df = compute_amihud_by_event(prices, events)

    df = event_returns.merge(
        amihud_df, on=[Col.TICKER, Col.ANNOUNCE_DATE], how="inner", suffixes=("", "_amihud")
    )
    # If the merge produced a duplicate amihud column from a pre-existing one,
    # drop the pre-existing one and keep the freshly computed value.
    dup_col = f"{Col.AMIHUD_ILLIQUIDITY}_amihud"
    if dup_col in df.columns:
        df = df.drop(columns=[Col.AMIHUD_ILLIQUIDITY]).rename(
            columns={dup_col: Col.AMIHUD_ILLIQUIDITY}
        )

    out = _double_sort(
        df,
        sort_col=Col.AMIHUD_ILLIQUIDITY,
        return_col=return_col,
        decile_col=decile_col,
        n_buckets=n_liquidity_buckets,
        bucket_label="liquidity_bucket",
    )
    if not out.empty:
        out = out.rename(columns={f"{Col.AMIHUD_ILLIQUIDITY}_median": "amihud_median"})
    return out


def double_sort_by_size(
    event_returns: pd.DataFrame,
    prices: pd.DataFrame,
    return_col: str = "car_short_drift",
    n_size_buckets: int = 5,
    decile_col: str = "sue3_decile",
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Same as :func:`double_sort_asymmetry` but sorting by firm size.

    Bartov, Krinsky & Kremers (2000) show that institutional ownership
    (proxied here by firm size) explains PEAD: small firms drift more.
    If the asymmetry shrinks within size buckets, it is driven by size
    differences between misses and beats.

    Size is proxied by average dollar volume over the pre-announcement
    estimation window when ``market_cap`` is not present in
    ``event_returns``; otherwise the canonical ``market_cap`` column is used.

    Returns:
        DataFrame with columns
        ``[size_bucket, size_median, miss_car, beat_car, ratio, n, n_miss, n_beat]``.
    """
    if Col.MARKET_CAP in event_returns.columns:
        df = event_returns.copy()
        df["size"] = pd.to_numeric(df[Col.MARKET_CAP], errors="coerce")
    else:
        events = event_returns[[Col.TICKER, Col.ANNOUNCE_DATE]].drop_duplicates()
        size_df = _compute_size_by_event(prices, events)
        df = event_returns.merge(size_df, on=[Col.TICKER, Col.ANNOUNCE_DATE], how="inner")

    return _double_sort(
        df,
        sort_col="size",
        return_col=return_col,
        decile_col=decile_col,
        n_buckets=n_size_buckets,
        bucket_label="size_bucket",
    )


__all__ = [
    "compute_amihud_by_event",
    "double_sort_asymmetry",
    "double_sort_by_size",
]
