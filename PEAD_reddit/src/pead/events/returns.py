"""
Event-study abnormal returns for the PEAD asymmetry pipeline.

Implements the standard market-model event study (MacKinlay 1997):

* Estimate ``R_firm = alpha + beta * R_market + epsilon`` over a pre-event
  estimation window (default [−255, −46] trading days).
* Compute abnormal returns ``AR_t = R_firm_t - (alpha + beta * R_market_t)`` and
  cumulate them (CAR) over event windows expressed in event time (day 0 =
  announcement).
* Optionally use mid-quote returns ``(MID_t / MID_{t-1}) - 1`` to purge the
  bid-ask bounce that inflates short-side (miss) drift (Zhang, Gregoriou &
  Wu 2024).
* Report buy-and-hold abnormal returns (BHAR) for completeness — these are
  right-skewed (Mitchell & Stafford 2000) and must NOT be used for primary
  inference; CAR is the inferential statistic.

Market proxy
------------
Because the synthetic fixture has no single index series, the market return is
the cross-sectional equal-weight average of all stock returns each day
(computed from the prices panel). In production this is replaced by SPX Index
returns. The market-model *intercept* absorbs the unconditional mean drift, so
the procedure is robust even when the proxy is a weak correlate of any single
name.

Segment-aware panels
--------------------
The synthetic generator emits one self-contained price segment per event and
concatenates them, which produces duplicate ``(ticker, trading_date)`` rows.
For such panels :func:`compute_all_event_returns` reconstructs the per-event
segments (by row order, verified against the events table) so each event is
matched to its own price path. Clean production panels (one row per
ticker-date) are used directly without any such treatment.

References
----------
MacKinlay (1997), "Event Studies in Economics and Finance," JEL 35(1).
Mitchell & Stafford (2000), "Managerial Decisions and Long-Term Stock Price
    Performance," JBL 55(6).
Zhang, Gregoriou & Wu (2024), bid-ask bounce and PEAD asymmetry.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pead.events.calendar import build_trading_calendar, get_window_dates
from pead.schema import Col

logger = logging.getLogger(__name__)

# Minimum observations required to fit the market model (MacKinlay 1997 uses
# ~120–210; we allow shorter synthetic segments but demand enough for OLS).
MIN_ESTIMATION_OBS = 30

# Internal column used to tag reconstructed per-event segments.
_SEG_ID = "_seg_id"


# ─── Small aligned-Series primitives ────────────────────────────────────────


def _slice_by_date(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return the sub-series whose index lies in ``[start, end]`` (inclusive).

    Uses an O(n) boolean mask so behaviour is well-defined even if the index is
    not monotonic. The batch path uses :func:`_slice_loc` (label slicing) for
    speed on the large synthetic panel.
    """
    idx = series.index
    return series.loc[(idx >= start) & (idx <= end)]


def _slice_loc(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fast inclusive label slice for a sorted DatetimeIndex."""
    return series.loc[start:end]


def compute_market_returns(prices: pd.DataFrame, col: str = Col.RET) -> pd.Series:
    """Cross-sectional equal-weight average of ``col`` per trading day.

    This is the market proxy used by the market model when no single index
    series (e.g. SPX) is supplied.
    """
    return prices.groupby(Col.TRADING_DATE)[col].mean().sort_index()


def _ols_alpha_beta(
    firm: pd.Series,
    market: pd.Series,
    min_obs: int = MIN_ESTIMATION_OBS,
) -> tuple[float, float] | None:
    """OLS fit ``firm = alpha + beta * market`` via numpy.polyfit.

    Returns ``(alpha, beta)`` or ``None`` if fewer than ``min_obs`` overlapping
    non-null observations, or if the market has no variance. ``market`` is
    reindexed onto ``firm``'s index so the two need not share every label.
    """
    if len(firm) == 0:
        return None
    m = market.reindex(firm.index).to_numpy(dtype=float)
    f = firm.to_numpy(dtype=float)
    mask = np.isfinite(f) & np.isfinite(m)
    if int(mask.sum()) < min_obs:
        return None
    mm = m[mask]
    if np.std(mm) == 0.0:
        return None
    ff = f[mask]
    # polyfit returns highest-degree coefficient first: [slope, intercept].
    beta, alpha = np.polyfit(mm, ff, 1)
    if not (np.isfinite(alpha) and np.isfinite(beta)):
        return None
    return float(alpha), float(beta)


def _car_from_aligned(
    firm: pd.Series,
    market: pd.Series,
    alpha: float,
    beta: float,
    window_start: int,
    window_end: int,
    ticker: str | None = None,
) -> float | None:
    """CAR over aligned firm/market return series.

    ``CAR = sum_t AR_t`` where ``AR_t = R_firm_t - (alpha + beta * R_market_t)``.
    Returns ``None`` when no overlapping observations exist. Missing days
    (e.g. a delisting mid-window) truncate the sum and are logged.
    """
    if len(firm) == 0:
        return None
    m = market.reindex(firm.index).to_numpy(dtype=float)
    f = firm.to_numpy(dtype=float)
    mask = np.isfinite(f) & np.isfinite(m)
    n_obs = int(mask.sum())
    if n_obs == 0:
        return None
    expected = window_end - window_start + 1
    if n_obs < expected and ticker is not None:
        logger.info("CAR window truncated for %s: %d of %d days available", ticker, n_obs, expected)
    ar = f[mask] - (alpha + beta * m[mask])
    return float(ar.sum())


def _bhar_from_aligned(firm: pd.Series, bench: pd.Series) -> float | None:
    """BHAR = prod(1+R_firm) - prod(1+R_bench) over aligned series."""
    if len(firm) == 0:
        return None
    b = bench.reindex(firm.index).to_numpy(dtype=float)
    f = firm.to_numpy(dtype=float)
    mask = np.isfinite(f) & np.isfinite(b)
    if int(mask.sum()) == 0:
        return None
    return float(np.prod(1.0 + f[mask]) - np.prod(1.0 + b[mask]))


# ─── Mid-quote returns ──────────────────────────────────────────────────────


def _ensure_ret_midquote(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``prices`` with a ``ret_midquote`` column.

    Mid-quote return ``R_mid_t = MID_t / MID_{t-1} - 1`` is computed within
    each firm/segment so consecutive segments for the same ticker do not bleed
    into one another.
    """
    if Col.RET_MIDQUOTE in prices.columns:
        return prices
    df = prices.copy()
    if Col.PX_MIDQUOTE not in df.columns:
        if Col.PX_BID in df.columns and Col.PX_ASK in df.columns:
            df[Col.PX_MIDQUOTE] = (df[Col.PX_BID] + df[Col.PX_ASK]) / 2.0
        else:
            raise ValueError(
                "Cannot compute midquote returns: need px_midquote or (px_bid, px_ask)"
            )
    # Group within reconstructed segments when present, else within ticker.
    grp_col = _SEG_ID if _SEG_ID in df.columns else Col.TICKER
    df = df.sort_values([grp_col, Col.TRADING_DATE])
    df[Col.RET_MIDQUOTE] = df.groupby(grp_col, group_keys=False)[Col.PX_MIDQUOTE].transform(
        lambda s: s / s.shift(1) - 1.0
    )
    return df


def _extract_firm_series(prices: pd.DataFrame, ticker: str, col: str) -> pd.Series:
    """Return a single ticker's ``col`` series indexed by trading date (clean panel)."""
    sub = prices.loc[prices[Col.TICKER] == ticker, [Col.TRADING_DATE, col]]
    return sub.set_index(Col.TRADING_DATE)[col].sort_index()


def _ensure_midquote_panel(
    prices: pd.DataFrame,
    use_midquote: bool,
    firm_series: pd.Series | None,
    market_series: pd.Series | None,
) -> pd.DataFrame:
    """Ensure ``ret_midquote`` exists when the caller did not supply precomputed series.

    The per-event public functions accept optional precomputed firm/market series
    to avoid recomputation; only when those are absent (and mid-quote returns are
    requested) do we materialise the column from the prices panel.
    """
    if (
        use_midquote
        and Col.RET_MIDQUOTE not in prices.columns
        and (firm_series is None or market_series is None)
    ):
        return _ensure_ret_midquote(prices)
    return prices


# ─── Public per-event API (clean panel contract) ────────────────────────────


def estimate_market_model(
    prices: pd.DataFrame,
    announce_date: pd.Timestamp,
    ticker: str,
    calendar: pd.DatetimeIndex,
    est_start: int = -255,
    est_end: int = -46,
    market_index_col: str = Col.RET,
    market_returns: pd.Series | None = None,
    firm_series: pd.Series | None = None,
    min_obs: int = MIN_ESTIMATION_OBS,
) -> tuple[float, float] | None:
    """Estimate market-model ``(alpha, beta)`` over ``[est_start, est_end]``.

    Fits ``R_firm = alpha + beta * R_market + epsilon`` by OLS, where the
    market return is the cross-sectional equal-weight average of all stocks'
    ``market_index_col`` each day (precomputable via :func:`compute_market_returns`
    and passed as ``market_returns`` for efficiency).

    Returns ``None`` if there are fewer than ``min_obs`` usable observations.
    """
    start_date, end_date = get_window_dates(announce_date, est_start, est_end, calendar)
    firm = (
        firm_series
        if firm_series is not None
        else _extract_firm_series(prices, ticker, market_index_col)
    )
    firm_win = _slice_by_date(firm, start_date, end_date)
    mkt = (
        market_returns
        if market_returns is not None
        else compute_market_returns(prices, market_index_col)
    )
    mkt_win = _slice_by_date(mkt, start_date, end_date)
    return _ols_alpha_beta(firm_win, mkt_win, min_obs=min_obs)


def compute_car(
    prices: pd.DataFrame,
    ticker: str,
    announce_date: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    window_start: int,
    window_end: int,
    alpha: float,
    beta: float,
    use_midquote: bool = False,
    market_returns: pd.Series | None = None,
    firm_series: pd.Series | None = None,
) -> float | None:
    """Cumulative Abnormal Return over ``[window_start, window_end]``.

    ``CAR = sum_t AR_t`` with ``AR_t = R_firm_t - (alpha + beta * R_market_t)``.
    When ``use_midquote=True``, mid-quote-to-mid-quote returns are used in place
    of close-to-close returns (for both firm and market).

    Returns ``None`` if the window contains no usable observations. Windows
    truncated by missing data (e.g. delistings) sum the available days.
    """
    ret_col = Col.RET_MIDQUOTE if use_midquote else Col.RET
    start_date, end_date = get_window_dates(announce_date, window_start, window_end, calendar)
    panel = _ensure_midquote_panel(prices, use_midquote, firm_series, market_returns)
    firm = firm_series if firm_series is not None else _extract_firm_series(panel, ticker, ret_col)
    firm_win = _slice_by_date(firm, start_date, end_date)
    mkt = market_returns if market_returns is not None else compute_market_returns(panel, ret_col)
    mkt_win = _slice_by_date(mkt, start_date, end_date)
    return _car_from_aligned(firm_win, mkt_win, alpha, beta, window_start, window_end, ticker)


def compute_bhar(
    prices: pd.DataFrame,
    ticker: str,
    announce_date: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    window_start: int,
    window_end: int,
    benchmark_returns: pd.Series | None = None,
    use_midquote: bool = False,
    firm_series: pd.Series | None = None,
) -> float | None:
    """Buy-and-Hold Abnormal Return over ``[window_start, window_end]``.

    ``BHAR = prod(1+R_firm) - prod(1+R_bench)``. The benchmark defaults to the
    cross-sectional equal-weight market return.

    .. warning::
        BHAR is right-skewed and cross-correlated (Mitchell & Stafford 2000).
        Use it for reporting only; use CAR for inference.
    """
    ret_col = Col.RET_MIDQUOTE if use_midquote else Col.RET
    start_date, end_date = get_window_dates(announce_date, window_start, window_end, calendar)
    panel = _ensure_midquote_panel(prices, use_midquote, firm_series, benchmark_returns)
    firm = firm_series if firm_series is not None else _extract_firm_series(panel, ticker, ret_col)
    firm_win = _slice_by_date(firm, start_date, end_date)
    bench = (
        benchmark_returns
        if benchmark_returns is not None
        else compute_market_returns(panel, ret_col)
    )
    bench_win = _slice_by_date(bench, start_date, end_date)
    return _bhar_from_aligned(firm_win, bench_win)


# ─── Segment reconstruction for duplicated panels ───────────────────────────


def _prepare_event_panel(
    prices: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int] | None]:
    """Return ``(panel, event_to_seg)`` suitable for per-event extraction.

    * Clean panel (no duplicate ``(ticker, trading_date)``): returned unchanged
      with ``event_to_seg=None`` (events are located by ticker).
    * Duplicated panel (synthetic fixture): rows are split into per-event
      segments by detecting contiguous runs (a new segment starts whenever the
      ticker changes or the trading date steps backwards). Segments are aligned
      to the events table in row order and verified by ticker and by the
      announcement falling inside the segment's date span. On success
      ``event_to_seg[i]`` maps event *i* to its segment id. If verification
      fails the panel is de-duplicated (keeping the first row per
      ticker-date) and returned as a clean panel.
    """
    dup_mask = prices.duplicated(subset=[Col.TICKER, Col.TRADING_DATE], keep=False)
    if not dup_mask.any():
        return prices, None

    df = prices.reset_index(drop=True).copy()
    prev_tkr = df[Col.TICKER].shift(1)
    prev_dt = df[Col.TRADING_DATE].shift(1)
    # First row: prev_tkr is NA -> (tkr != NA) is True, so it starts segment 0.
    new_seg = (df[Col.TICKER] != prev_tkr) | (df[Col.TRADING_DATE] < prev_dt)
    df[_SEG_ID] = new_seg.cumsum().astype(int) - 1

    n_seg = int(df[_SEG_ID].max()) + 1
    event_to_seg: list[int] | None = None

    if n_seg == len(events):
        seg_info = df.groupby(_SEG_ID).agg(
            tkr=(Col.TICKER, "first"),
            lo=(Col.TRADING_DATE, "min"),
            hi=(Col.TRADING_DATE, "max"),
        )
        ev = events.reset_index(drop=True)
        seg_tkr = seg_info["tkr"].to_numpy()
        seg_lo = pd.DatetimeIndex(seg_info["lo"].to_numpy()).normalize()
        seg_hi = pd.DatetimeIndex(seg_info["hi"].to_numpy()).normalize()
        ann = pd.DatetimeIndex(pd.to_datetime(ev[Col.ANNOUNCE_DATE]).to_numpy()).normalize()
        tkr_ok = seg_tkr == ev[Col.TICKER].to_numpy()
        span_ok = (seg_lo <= ann) & (ann <= seg_hi)
        if bool(tkr_ok.all()) and bool(span_ok.all()):
            event_to_seg = list(range(n_seg))
            logger.debug("Reconstructed %d per-event price segments from duplicated panel.", n_seg)

    if event_to_seg is None:
        logger.warning(
            "Could not align %d price segments to %d events; "
            "de-duplicating (ticker, trading_date) keeping first occurrence.",
            n_seg,
            len(events),
        )
        df = prices.drop_duplicates(
            subset=[Col.TICKER, Col.TRADING_DATE], keep="first"
        ).reset_index(drop=True)
        return df, None

    return df, event_to_seg


# ─── Master: all events x all windows ───────────────────────────────────────


def compute_all_event_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    windows: list[tuple[str, int, int]],
    use_midquote: bool = True,
    estimation_start: int = -255,
    estimation_end: int = -46,
) -> pd.DataFrame:
    """Compute CAR and BHAR for all events across all event windows.

    Parameters
    ----------
    events:
        Earnings events with at least ``ticker``, ``announce_date``.
    prices:
        Daily prices panel (``ticker``, ``trading_date``, ``px_close``,
        ``px_bid``, ``px_ask``, ``ret``, ...). Duplicated panels emitted by the
        synthetic generator are reconstructed into per-event segments
        automatically.
    windows:
        Sequence of ``(name, start_offset, end_offset)`` tuples. Drift windows
        must exclude day 0 (e.g. ``(1, 20)``); day 0 belongs only to the
        announcement-reaction window.
    use_midquote:
        If True, also compute ``car_midquote_{name}`` for each window using
        mid-quote-to-mid-quote returns.
    estimation_start, estimation_end:
        Trading-day offsets of the market-model estimation window relative to
        the announcement trading day.

    Returns
    -------
    pd.DataFrame
        The events table augmented with ``alpha``, ``beta`` and, per window,
        ``car_{name}``, ``bhar_{name}`` and (if requested)
        ``car_midquote_{name}``. Insufficient data yields NaN.
    """
    calendar = build_trading_calendar(prices)
    n_cal = len(calendar)
    prepared, event_to_seg = _prepare_event_panel(prices, events)

    if use_midquote:
        prepared = _ensure_ret_midquote(prepared)

    # Precompute the market proxy once (equal-weight cross-sectional mean/day).
    mkt_close = compute_market_returns(prepared, Col.RET)
    mkt_mid = compute_market_returns(prepared, Col.RET_MIDQUOTE) if use_midquote else None

    # Group prepared panel into per-event frames indexed by trading date.
    group_key = _SEG_ID if event_to_seg is not None else Col.TICKER
    group_frames: dict[int | str, pd.DataFrame] = {}
    for key, g in prepared.groupby(group_key, sort=False):
        group_frames[key] = g.set_index(Col.TRADING_DATE).sort_index()

    out = events.reset_index(drop=True).copy()
    n_events = len(out)
    tickers = out[Col.TICKER].to_numpy()

    # Vectorized event-time → calendar-date bounds.
    # base_pos[i] = calendar index of the announcement trading day (snap forward).
    ann = pd.DatetimeIndex(pd.to_datetime(out[Col.ANNOUNCE_DATE])).normalize()
    base_pos = np.clip(calendar.searchsorted(ann, side="left"), 0, n_cal - 1)

    est_sd = calendar[np.clip(base_pos + estimation_start, 0, n_cal - 1)]
    est_ed = calendar[np.clip(base_pos + estimation_end, 0, n_cal - 1)]
    win_sd: dict[str, np.ndarray] = {}
    win_ed: dict[str, np.ndarray] = {}
    for name, ws, we in windows:
        win_sd[name] = calendar[np.clip(base_pos + ws, 0, n_cal - 1)]
        win_ed[name] = calendar[np.clip(base_pos + we, 0, n_cal - 1)]

    # Result containers (filled once at the end — no per-row pandas writes).
    alpha_arr = np.full(n_events, np.nan)
    beta_arr = np.full(n_events, np.nan)
    car_arr: dict[str, np.ndarray] = {n: np.full(n_events, np.nan) for n, _, _ in windows}
    bhar_arr: dict[str, np.ndarray] = {n: np.full(n_events, np.nan) for n, _, _ in windows}
    car_mq_arr: dict[str, np.ndarray] | None = (
        {n: np.full(n_events, np.nan) for n, _, _ in windows} if use_midquote else None
    )

    n_skipped = 0
    for idx in range(n_events):
        key: int | str = event_to_seg[idx] if event_to_seg is not None else tickers[idx]
        gf = group_frames.get(key)
        if gf is None:
            n_skipped += 1
            continue
        firm_close = gf[Col.RET]

        esd, eed = est_sd[idx], est_ed[idx]
        ab = _ols_alpha_beta(_slice_loc(firm_close, esd, eed), _slice_loc(mkt_close, esd, eed))
        if ab is None:
            n_skipped += 1
            continue
        alpha, beta = ab
        alpha_arr[idx] = alpha
        beta_arr[idx] = beta

        ticker = tickers[idx]
        for name, ws, we in windows:
            sd, ed = win_sd[name][idx], win_ed[name][idx]
            f_win = _slice_loc(firm_close, sd, ed)
            m_win = _slice_loc(mkt_close, sd, ed)

            car = _car_from_aligned(f_win, m_win, alpha, beta, ws, we, ticker)
            if car is not None:
                car_arr[name][idx] = car

            bhar = _bhar_from_aligned(f_win, m_win)
            if bhar is not None:
                bhar_arr[name][idx] = bhar

            if use_midquote:
                fm_win = _slice_loc(gf[Col.RET_MIDQUOTE], sd, ed)
                mm_win = _slice_loc(mkt_mid, sd, ed)
                car_mq = _car_from_aligned(fm_win, mm_win, alpha, beta, ws, we, ticker)
                if car_mq is not None:
                    assert car_mq_arr is not None
                    car_mq_arr[name][idx] = car_mq

    out[Col.ALPHA] = alpha_arr
    out[Col.BETA] = beta_arr
    for name, _ws, _we in windows:
        out[f"car_{name}"] = car_arr[name]
        out[f"bhar_{name}"] = bhar_arr[name]
        if use_midquote:
            assert car_mq_arr is not None
            out[f"car_midquote_{name}"] = car_mq_arr[name]

    if n_skipped:
        logger.info(
            "compute_all_event_returns: skipped %d of %d events with insufficient data",
            n_skipped,
            n_events,
        )
    return out
