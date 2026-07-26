"""
Bloomberg data ingest layer for the PEAD asymmetry pipeline.

This module pulls raw data from a Bloomberg terminal via BQL/BLPAPI (using the
``pdblp`` and ``xbbg`` Python wrappers). Because a live Bloomberg terminal is
not available in CI, every ``fetch_*`` function has a companion ``mock_*``
function that returns deterministic synthetic data (see
:mod:`pead.synthetic`) so the rest of the pipeline can be developed and tested
end-to-end without a terminal.

Design rules
------------
* ``xbbg`` and ``pdblp`` are imported **lazily** inside the functions that need
  them, so this module imports cleanly in an environment without Bloomberg.
* Every ``fetch_*`` function accepts an optional ``conn`` (a live
  ``pdblp.BCon`` connection).  When ``conn is None`` we attempt to create one;
  if ``pdblp`` is not installed we raise a clear ``RuntimeError`` telling the
  caller to use the mock functions instead.
* All output DataFrames use the canonical column names from
  :class:`pead.schema.Col` and are validated against
  :class:`pead.schema.ExpectedSchema`.

Reference: Livnat & Mendenhall (2006), Bernard & Thomas (1989/1990).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from pead.schema import Col, ExpectedSchema, validate_dataframe
from pead.synthetic import (
    generate_synthetic_earnings_events,
    generate_synthetic_fundamentals,
    generate_synthetic_prices,
)

logger = logging.getLogger(__name__)

# ─── Bloomberg ticker helpers ───────────────────────────────────────────────


def _to_bbg_ticker(ticker: str) -> str:
    """Normalize a ticker to the Bloomberg ``"ROOT SECTOR Equity"`` convention.

    Examples
    --------
    >>> _to_bbg_ticker("AAPL")
    'AAPL US Equity'
    >>> _to_bbg_ticker("AAPL US Equity")
    'AAPL US Equity'
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError(f"Invalid ticker: {ticker!r}")
    if " " in ticker.strip():
        return ticker.strip()
    return f"{ticker.strip()} US Equity"


def _bbg_tickers(tickers: list[str]) -> list[str]:
    """Normalize a list of tickers to Bloomberg convention."""
    return [_to_bbg_ticker(t) for t in tickers]


def _relabel_to_requested(df: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Relabel synthetic tickers to a caller-supplied ticker list (1:1).

    The synthetic generator produces tickers of the form ``"TKR000 US Equity"``.
    When a caller asks for a concrete universe we relabel the synthetic firms
    one-to-one onto the requested names so downstream code sees the expected
    tickers.  If ``requested`` is empty the frame is returned unchanged.
    """
    if not requested or Col.TICKER not in df.columns:
        return df
    synth = list(pd.unique(df[Col.TICKER]))
    n_requested = len(requested)
    if len(synth) <= n_requested:
        mapping = dict(zip(synth, requested[: len(synth)]))
    else:
        # More synthetic firms than requested: cycle the requested names.
        mapping = {s: requested[i % n_requested] for i, s in enumerate(synth)}
    df = df.copy()
    df[Col.TICKER] = df[Col.TICKER].map(mapping).fillna(df[Col.TICKER])
    return df


# ─── Connection handling ────────────────────────────────────────────────────


def _get_connection(conn: Any) -> Any:
    """Return ``conn`` or start a new ``pdblp.BCon`` session.

    Raises
    ------
    RuntimeError
        If ``conn is None`` and ``pdblp`` is not installed.
    """
    if conn is not None:
        return conn
    try:
        import pdblp  # lazy: only needed when actually talking to Bloomberg
    except ImportError as exc:  # pragma: no cover - exercised only without pdblp
        raise RuntimeError(
            "No Bloomberg connection was provided (conn=None) and the 'pdblp' "
            "package is not installed. Either pass an open pdblp.BCon via "
            "'conn=', install the Bloomberg extras "
            "('pip install pead-asymmetry[bloomberg]'), or use the mock_* "
            "functions for offline testing."
        ) from exc
    connection = pdblp.BCon(timeout=5000)
    connection.start()
    logger.info("Started new pdblp.BCon Bloomberg session")
    return connection


# ─── 1. Earnings estimates / actuals (BEST + EE) ────────────────────────────


def fetch_earnings_estimates(
    tickers: list[str],
    start_date: str,
    end_date: str,
    conn=None,
) -> pd.DataFrame:
    """Fetch analyst EPS estimates and actuals from Bloomberg BEST / EE.

    Pulls the I/B/E/S-equivalent consensus from Bloomberg's BEST dataset and
    the reported actual from the EE (Earnings & Estimates) dataset, then
    aligns them to the canonical earnings-event schema.

    Bloomberg fields used
    ---------------------
    * ``BEST_EPS_EST_PX_ANNS_DT`` — best-estimate EPS announcement date
      (event date for the consensus snapshot).
    * ``EE_API`` — earnings announcement date (event calendar).
    * ``EE_EPS_ACT`` — actual reported EPS (EE actual).
    * ``BEST_EPS_EST_MED`` — median analyst EPS estimate (SUE3 input).
    * ``BEST_EPS_EST_MEAN`` — mean analyst EPS estimate.
    * ``BEST_EPS_NEST`` — number of analysts in the consensus.

    Returns
    -------
    DataFrame with columns: ``ticker, announce_date, fiscal_quarter, fq_end,
    actual_eps, medest_eps, meanest_eps, n_analysts``.
    """
    con = _get_connection(conn)
    bbg = _bbg_tickers(tickers)

    fields = [
        "BEST_EPS_EST_PX_ANNS_DT",  # announcement date for the consensus
        "EE_API",  # earnings announcement date
        "EE_EPS_ACT",  # actual EPS
        "BEST_EPS_EST_MED",  # median estimate
        "BEST_EPS_EST_MEAN",  # mean estimate
        "BEST_EPS_NEST",  # number of estimates
    ]

    # Bulk reference query: one row per fiscal-period event per ticker.
    # pdblp exposes bulk data via ``bulk_ref``; xbbg exposes it via ``bds``.
    raw = con.bulk_ref(bbg, fields)

    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        announce = r.get("EE_API") or r.get("BEST_EPS_EST_PX_ANNS_DT")
        if pd.isna(announce):
            continue
        announce_ts = pd.Timestamp(announce)
        if announce_ts < pd.Timestamp(start_date) or announce_ts > pd.Timestamp(end_date):
            continue
        fq_end = _infer_fiscal_quarter_end(announce_ts)
        rows.append(
            {
                Col.TICKER: r.get("ticker"),
                Col.ANNOUNCE_DATE: announce_ts,
                Col.FISCAL_QUARTER: f"{fq_end.year}Q{fq_end.quarter}",
                Col.FISCAL_QUARTER_END: fq_end,
                Col.ACTUAL_EPS: float(r.get("EE_EPS_ACT", np.nan)),
                Col.MEDEST_EPS: float(r.get("BEST_EPS_EST_MED", np.nan)),
                Col.MEANEST_EPS: float(r.get("BEST_EPS_EST_MEAN", np.nan)),
                Col.N_ANALYSTS: int(r.get("BEST_EPS_NEST", 0))
                if pd.notna(r.get("BEST_EPS_NEST"))
                else 0,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out[Col.ANNOUNCE_DATE] = pd.to_datetime(out[Col.ANNOUNCE_DATE])
        out[Col.FISCAL_QUARTER_END] = pd.to_datetime(out[Col.FISCAL_QUARTER_END])

    validate_dataframe(
        out,
        ExpectedSchema.EARNINGS_EVENTS,
        name="fetch_earnings_estimates",
        strict_dates=[Col.ANNOUNCE_DATE],
    )
    return out


def mock_earnings_estimates(
    tickers: list[str],
    start_date: str,
    end_date: str,
    conn=None,  # noqa: ARG001 - kept for API parity with fetch_*
    seed: int = 42,
) -> pd.DataFrame:
    """Return deterministic synthetic earnings estimates (no Bloomberg needed).

    Mirrors :func:`fetch_earnings_estimates` but sources data from
    :func:`pead.synthetic.generate_synthetic_earnings_events`, relabels the
    synthetic firms onto ``tickers``, and filters to ``[start_date, end_date]``.
    """
    n = max(1, len(tickers)) if tickers else 50
    df = generate_synthetic_earnings_events(n_tickers=n, seed=seed)
    df = _relabel_to_requested(df, tickers)
    df = df[
        (df[Col.ANNOUNCE_DATE] >= pd.Timestamp(start_date))
        & (df[Col.ANNOUNCE_DATE] <= pd.Timestamp(end_date))
    ]
    df = df.reset_index(drop=True)

    validate_dataframe(
        df,
        ExpectedSchema.EARNINGS_EVENTS,
        name="mock_earnings_estimates",
        strict_dates=[Col.ANNOUNCE_DATE],
    )
    return df


# ─── 2. Quarterly fundamentals (FA_DATA) ────────────────────────────────────


def fetch_fundamentals(
    tickers: list[str],
    start_date: str,
    end_date: str,
    conn=None,
) -> pd.DataFrame:
    """Fetch quarterly fundamentals from Bloomberg ``FA_DATA``.

    Maps Bloomberg's ``FA_*`` (financial answers) fields to the Compustat
    equivalents used by the PEAD literature.

    Bloomberg fields used
    ---------------------
    * ``FA_AMP_EPS`` — primary / as-reported quarterly EPS (Compustat ``epspxq``).
    * ``FA_DILUTED_EPS`` — diluted quarterly EPS (Compustat ``epsfxq``).
    * ``FA_SPL_ITEM`` — special items (Compustat ``spiq``).
    * ``FA_BP_BAS_EV`` — basic shares outstanding, period end (Compustat ``cshoq``).
    * ``FA_BP_DIL_EV`` — diluted shares outstanding (Compustat ``cshfdq``).
    * ``PX_LAST`` — unadjusted price at fiscal-quarter end (Compustat ``prccq``).
    * ``CUR_RPT_DATE`` — most recent report date (Compustat ``rdq``).
    * ``EQY_DVD_SH_ADJ`` — cumulative split adjustment factor (CRSP ``cfacshr``).

    Returns
    -------
    DataFrame with columns: ``ticker, fq_end, fiscal_quarter, eps_primary,
    eps_diluted, special_items, shares_basic, shares_diluted, price_qe,
    report_date, adj_factor``.
    """
    con = _get_connection(conn)
    bbg = _bbg_tickers(tickers)

    fa_fields = [
        "FA_AMP_EPS",
        "FA_DILUTED_EPS",
        "FA_SPL_ITEM",
        "FA_BP_BAS_EV",
        "FA_BP_DIL_EV",
        "PX_LAST",
        "CUR_RPT_DATE",
        "EQY_DVD_SH_ADJ",
    ]

    raw = con.bulk_ref(bbg, fa_fields)

    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        fq_end = pd.Timestamp(r.get("FA_FISCAL_QUARTER_END") or r.get("date"))
        if pd.isna(fq_end):
            continue
        if fq_end < pd.Timestamp(start_date) or fq_end > pd.Timestamp(end_date):
            continue
        rows.append(
            {
                Col.TICKER: r.get("ticker"),
                Col.FISCAL_QUARTER_END: fq_end,
                Col.FISCAL_QUARTER: f"{fq_end.year}Q{fq_end.quarter}",
                Col.EPS_PRIMARY: float(r.get("FA_AMP_EPS", np.nan)),
                Col.EPS_DILUTED: float(r.get("FA_DILUTED_EPS", np.nan)),
                Col.SPECIAL_ITEMS: float(r.get("FA_SPL_ITEM", np.nan)),
                Col.SHARES_BASIC: float(r.get("FA_BP_BAS_EV", np.nan)),
                Col.SHARES_DILUTED: float(r.get("FA_BP_DIL_EV", np.nan)),
                Col.PRICE_QUARTER_END: float(r.get("PX_LAST", np.nan)),
                Col.REPORT_DATE: pd.to_datetime(r.get("CUR_RPT_DATE")),
                Col.ADJ_FACTOR: float(r.get("EQY_DVD_SH_ADJ", 1.0) or 1.0),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        for c in (Col.FISCAL_QUARTER_END, Col.REPORT_DATE):
            out[c] = pd.to_datetime(out[c])

    validate_dataframe(
        out,
        ExpectedSchema.FUNDAMENTALS_Q,
        name="fetch_fundamentals",
        strict_dates=[Col.FISCAL_QUARTER_END],
    )
    return out


def mock_fundamentals(
    tickers: list[str],
    start_date: str,
    end_date: str,
    conn=None,  # noqa: ARG001 - kept for API parity
    seed: int = 42,
) -> pd.DataFrame:
    """Return deterministic synthetic quarterly fundamentals (no Bloomberg).

    Builds on :func:`mock_earnings_estimates` and enriches with the
    Compustat-equivalent fields from
    :func:`pead.synthetic.generate_synthetic_fundamentals`.
    """
    events = mock_earnings_estimates(tickers, start_date, end_date, seed=seed)
    out = generate_synthetic_fundamentals(events, seed=seed)
    out = out.reset_index(drop=True)

    validate_dataframe(
        out,
        ExpectedSchema.FUNDAMENTALS_Q,
        name="mock_fundamentals",
        strict_dates=[Col.FISCAL_QUARTER_END, Col.REPORT_DATE],
    )
    return out


# ─── 3. Daily prices + bid/ask ──────────────────────────────────────────────


def fetch_daily_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    conn=None,
) -> pd.DataFrame:
    """Fetch daily OHLCV + bid/ask from Bloomberg.

    Bloomberg fields used
    ---------------------
    * ``PX_OPEN`` — open price.
    * ``PX_HIGH`` — high price.
    * ``PX_LOW`` — low price.
    * ``PX_LAST`` — close price (mapped to ``px_close``).
    * ``PX_BID`` — bid price at close.
    * ``PX_ASK`` — ask price at close.
    * ``PX_VOLUME`` — share volume.
    * ``DAY_TO_DAY_TOT_RETURN_GROSS_DVDS`` — gross total return incl. dividends
      (mapped to ``ret``); this is the correct return series for PEAD so the
      short side is not distorted by dividend dates.

    The mid-quote ``px_midquote`` is computed as ``(bid + ask) / 2`` here so the
    downstream market model can optionally use mid-quotes to purge bid-ask
    bounce (per the event-study config ``use_mid_quote``).

    Returns
    -------
    DataFrame with columns: ``ticker, trading_date, px_open, px_high, px_low,
    px_close, px_bid, px_ask, px_midquote, volume, ret``.
    """
    con = _get_connection(conn)
    bbg = _bbg_tickers(tickers)

    fields = [
        "PX_OPEN",
        "PX_HIGH",
        "PX_LOW",
        "PX_LAST",
        "PX_BID",
        "PX_ASK",
        "PX_VOLUME",
        "DAY_TO_DAY_TOT_RETURN_GROSS_DVDS",
    ]

    raw = con.bdh(bbg, fields, start_date, end_date)

    frames: list[pd.DataFrame] = []
    for tkr in bbg:
        sub = raw.xs(tkr, level=0, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw
        if sub.empty:
            continue
        df = pd.DataFrame(
            {
                Col.TICKER: tkr,
                Col.TRADING_DATE: pd.to_datetime(sub.index),
                Col.PX_OPEN: sub.get("PX_OPEN").to_numpy(),
                Col.PX_HIGH: sub.get("PX_HIGH").to_numpy(),
                Col.PX_LOW: sub.get("PX_LOW").to_numpy(),
                Col.PX_CLOSE: sub.get("PX_LAST").to_numpy(),
                Col.PX_BID: sub.get("PX_BID").to_numpy(),
                Col.PX_ASK: sub.get("PX_ASK").to_numpy(),
                Col.VOLUME: sub.get("PX_VOLUME").to_numpy(),
                Col.RET: sub.get("DAY_TO_DAY_TOT_RETURN_GROSS_DVDS").to_numpy(),
            }
        )
        frames.append(df)

    if not frames:
        out = pd.DataFrame(columns=[Col.TICKER, Col.TRADING_DATE])
    else:
        out = pd.concat(frames, ignore_index=True)

    out[Col.PX_MIDQUOTE] = (out[Col.PX_BID] + out[Col.PX_ASK]) / 2
    if not out.empty:
        out[Col.TRADING_DATE] = pd.to_datetime(out[Col.TRADING_DATE])

    validate_dataframe(
        out,
        ExpectedSchema.DAILY_PRICES,
        name="fetch_daily_prices",
        strict_dates=[Col.TRADING_DATE],
    )
    return out


def mock_daily_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    conn=None,  # noqa: ARG001 - kept for API parity
    seed: int = 42,
) -> pd.DataFrame:
    """Return deterministic synthetic daily prices with bid/ask (no Bloomberg).

    Sources from :func:`pead.synthetic.generate_synthetic_prices`, which injects
    a known PEAD asymmetry (misses drift ~2x beats) so the pipeline can be
    validated end-to-end.
    """
    events = mock_earnings_estimates(tickers, start_date, end_date, seed=seed)
    out = generate_synthetic_prices(events, seed=seed)
    out = out.reset_index(drop=True)

    # Guarantee the mid-quote invariant (bid+ask)/2 even if upstream changes.
    out[Col.PX_MIDQUOTE] = (out[Col.PX_BID] + out[Col.PX_ASK]) / 2
    out[Col.TRADING_DATE] = pd.to_datetime(out[Col.TRADING_DATE])

    validate_dataframe(
        out,
        ExpectedSchema.DAILY_PRICES,
        name="mock_daily_prices",
        strict_dates=[Col.TRADING_DATE],
    )
    return out


# ─── 4. Delisting events ────────────────────────────────────────────────────


def fetch_delisting_events(
    tickers: list[str],
    start_date: str,
    end_date: str,
    conn=None,
) -> pd.DataFrame:
    """Fetch delisting events to handle survivorship bias on the short side.

    This is **critical** for PEAD asymmetry: distressed firms (which tend to be
    large misses) are the ones that delist. Ignoring delisting returns biases
    the asymmetry ratio downward because the worst short-side outcomes are
    silently dropped.

    Bloomberg fields used
    ---------------------
    * ``EQY_DELIST_DATE`` — delisting / suspension date.
    * ``EQY_DELIST_RETURN`` — last available total return at delisting
      (analogous to CRSP ``dlret``).

    Returns
    -------
    DataFrame with columns: ``ticker, delisting_date, delisting_return``.
    """
    con = _get_connection(conn)
    bbg = _bbg_tickers(tickers)

    fields = ["EQY_DELIST_DATE", "EQY_DELIST_RETURN"]
    raw = con.ref(bbg, fields)

    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        ddate = r.get("EQY_DELIST_DATE")
        if pd.isna(ddate):
            continue
        ddate_ts = pd.Timestamp(ddate)
        if ddate_ts < pd.Timestamp(start_date) or ddate_ts > pd.Timestamp(end_date):
            continue
        rows.append(
            {
                Col.TICKER: r.get("ticker"),
                Col.DELISTING_DATE: ddate_ts,
                Col.DELISTING_RETURN: float(r.get("EQY_DELIST_RETURN", np.nan)),
            }
        )

    out = pd.DataFrame(rows, columns=[Col.TICKER, Col.DELISTING_DATE, Col.DELISTING_RETURN])
    if not out.empty:
        out[Col.DELISTING_DATE] = pd.to_datetime(out[Col.DELISTING_DATE])
    return out


def mock_delisting_events(
    tickers: list[str],
    start_date: str,
    end_date: str,
    conn=None,  # noqa: ARG001 - kept for API parity
    seed: int = 42,
    delist_fraction: float = 0.05,
) -> pd.DataFrame:
    """Return deterministic synthetic delisting events (no Bloomberg).

    A small fraction (``delist_fraction``) of firms are randomly assigned a
    delisting date within ``[start_date, end_date]`` with a (negative) delisting
    return, mirroring CRSP's delisting-return distribution. Distressed firms
    dominate this set, which is exactly the survivorship bias the short side
    needs corrected.
    """
    rng = np.random.default_rng(seed + 99)
    bbg = _bbg_tickers(tickers) if tickers else []
    names = bbg or [f"TKR{i:03d} US Equity" for i in range(50)]

    n_delist = max(1, int(round(len(names) * delist_fraction)))
    chosen = rng.choice(names, size=n_delist, replace=False)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    span = max(1, (end_ts - start_ts).days)

    rows: list[dict[str, Any]] = []
    for name in chosen:
        ddate = start_ts + pd.Timedelta(days=int(rng.integers(0, span)))
        # Delisting returns are typically negative (mean ~ -30%, heavy left tail).
        dret = float(rng.normal(-0.30, 0.35))
        rows.append(
            {
                Col.TICKER: name,
                Col.DELISTING_DATE: ddate,
                Col.DELISTING_RETURN: dret,
            }
        )

    out = pd.DataFrame(rows, columns=[Col.TICKER, Col.DELISTING_DATE, Col.DELISTING_RETURN])
    if not out.empty:
        out[Col.DELISTING_DATE] = pd.to_datetime(out[Col.DELISTING_DATE])
    return out


# ─── Livnat-Mendenhall (2006) data-quality filters ─────────────────────────


def apply_lm_filters(
    df: pd.DataFrame,
    min_price: float = 1.0,
    min_mcap_millions: float = 5.0,
    max_date_diff_days: int = 1,
) -> pd.DataFrame:
    """Apply the Livnat & Mendenhall (2006) data-quality screens.

    Filters
    -------
    1. ``price_qe > min_price`` — drop penny stocks (default $1).
    2. Market cap ``shares_basic * price_qe > min_mcap_millions`` — drop micro-caps
       (default $5M).  ``shares_basic`` is in millions of shares, so the product
       is in millions of USD.
    3. ``|report_date - announce_date| <= max_date_diff_days`` — keep only events
       where the Compustat report date and the I/B/E/S announce date agree
       (default ±1 day).  This avoids mismatching the surprise to the wrong
       earnings release.

    Any filter whose required columns are absent is silently skipped, so this
    function can be applied to either the merged earnings+fundamentals table or
    to the fundamentals table alone.
    """
    out = df.copy()
    n0 = len(out)

    # Filter 1: minimum price at quarter end.
    if Col.PRICE_QUARTER_END in out.columns:
        out = out[out[Col.PRICE_QUARTER_END].fillna(0.0) > min_price]

    # Filter 2: minimum market capitalization (shares in millions × price).
    if Col.SHARES_BASIC in out.columns and Col.PRICE_QUARTER_END in out.columns:
        mcap = out[Col.SHARES_BASIC].fillna(0.0) * out[Col.PRICE_QUARTER_END].fillna(0.0)
        out = out[mcap > min_mcap_millions]

    # Filter 3: announce vs. report date agreement.
    if Col.ANNOUNCE_DATE in out.columns and Col.REPORT_DATE in out.columns:
        ad = pd.to_datetime(out[Col.ANNOUNCE_DATE])
        rd = pd.to_datetime(out[Col.REPORT_DATE])
        diff_days = (rd - ad).dt.days.abs()
        out = out[diff_days <= max_date_diff_days]

    n1 = len(out)
    logger.info("apply_lm_filters: kept %d of %d rows (%d dropped)", n1, n0, n0 - n1)
    return out.reset_index(drop=True)


# ─── Internal helpers ───────────────────────────────────────────────────────


def _infer_fiscal_quarter_end(announce_date: pd.Timestamp) -> pd.Timestamp:
    """Approximate the fiscal-quarter end ~45 days before an announce date.

    Used only to populate the ``fq_end`` / ``fiscal_quarter`` columns when the
    BEST/EE response does not carry an explicit period end.
    """
    fq_end = announce_date - pd.Timedelta(days=45)
    return fq_end - pd.offsets.QuarterEnd(0)
