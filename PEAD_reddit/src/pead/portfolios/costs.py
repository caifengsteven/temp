"""
Transaction cost model for the PEAD portfolio pipeline.

Three layers of cost are applied:
    1. Commission      — per-side fixed cost (default 5 bps).
    2. Slippage        — base (5 bps) + Amihud illiquidity scaling. The price
                         impact of a trade scales with how illiquid the name is.
    3. Borrow cost     — short-side only, accrued over the holding period
                         (default 50 bps/year floor).

Transaction costs are ROUND-TRIP (buy at entry, sell at exit), so the gross
per-side cost is doubled. Borrow cost is NOT doubled — it is an accrual that
applies only to the short leg for the number of days the position is held.

References:
    Amihud (2002), "Illiquidity and Stock Returns", Journal of Financial Markets.
    Chordia, Goyal, Sadka, Sadka, Shivakumar (2009), "Liquidity and the
        Post-Earnings-Announcement Drift", Journal of Financial Research.
    Korczak, Korczak, Pacelli (2013) — CTP + transaction costs eliminate PEAD alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pead.schema import Col, TRADING_DAYS_PER_YEAR


# ─── Internal helpers ───────────────────────────────────────────────────────


def _ensure_dates(df: pd.DataFrame, col: str) -> None:
    """In-place conversion of ``df[col]`` to datetime64[ns]."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])


def _bps_to_frac(bps: float) -> float:
    """Convert basis points to a fraction (e.g. 5 bps -> 0.0005)."""
    return bps / 10_000.0


# ─── Public API ─────────────────────────────────────────────────────────────


def compute_amihud_illiquidity(
    prices: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Compute the Amihud (2002) illiquidity measure.

    For each (ticker, trading day):

        ILLIQ_t = |R_t| / DVOL_t

    where DVOL is the dollar trading volume ``px_close * volume``. We then
    average over the trailing ``window`` days (default 252, i.e. one trading
    year) to obtain a stable per-ticker illiquidity score.

    Args:
        prices: Long-format prices with ``ticker``, ``trading_date``,
                ``px_close``, ``volume``, ``ret``.
        window: Trailing window in trading days over which to average.

    Returns:
        DataFrame with columns ``[ticker, amihud]`` (one row per ticker).
        ``amihud`` is the mean of ``|R_t| / DVOL_t`` over the window. Higher
        values mean more illiquid.
    """
    for key in (Col.TICKER, Col.TRADING_DATE, Col.PX_CLOSE, Col.VOLUME, Col.RET):
        if key not in prices.columns:
            raise ValueError(f"prices missing required column {key!r}.")

    df = prices[[Col.TICKER, Col.TRADING_DATE, Col.PX_CLOSE, Col.VOLUME, Col.RET]].copy()
    _ensure_dates(df, Col.TRADING_DATE)
    df = df.sort_values([Col.TICKER, Col.TRADING_DATE]).reset_index(drop=True)

    # Dollar volume. Guard against zero / negative volume to avoid inf.
    dvol = (df[Col.PX_CLOSE].astype(float) * df[Col.VOLUME].astype(float)).to_numpy()
    dvol = np.where(dvol > 0, dvol, np.nan)

    df["_abs_ret"] = np.abs(df[Col.RET].astype(float).to_numpy())
    df["_illiq_daily"] = df["_abs_ret"].to_numpy() / dvol

    # transform on a single-column groupby passes a Series, not a DataFrame.
    df["_illiq_roll"] = df.groupby(Col.TICKER, observed=True)["_illiq_daily"].transform(
        lambda s: s.astype(float).rolling(window, min_periods=1).mean()
    )

    # Use the latest observed (rolling) illiquidity per ticker.
    latest = (
        df.dropna(subset=["_illiq_roll"])
        .sort_values(Col.TRADING_DATE)
        .groupby(Col.TICKER, observed=True)
        .tail(1)[[Col.TICKER, "_illiq_roll"]]
        .rename(columns={"_illiq_roll": Col.AMIHUD_ILLIQUIDITY})
        .reset_index(drop=True)
    )
    return latest


def apply_transaction_costs(
    portfolio_ret: pd.DataFrame,
    portfolio_assignments: pd.DataFrame,
    prices: pd.DataFrame,
    commission_bps: float = 5.0,
    slippage_base_bps: float = 5.0,
    slippage_amihud_scale: float = 1.0,
    borrow_cost_annual_bps: float = 50.0,
    holding_days: int = 20,
) -> pd.DataFrame:
    """Apply transaction costs to portfolio returns.

    Per-side gross cost (in fraction):

        cost_per_side_i = commission
                         + slippage_base
                         + slippage_amihud_scale * amihud_i

    Amihud is a *raw* illiquidity measure (|R|/DVOL, with DVOL in dollars),
    which is typically tiny in absolute terms (e.g. 1e-7 for liquid names,
    1e-4 for illiquid). To make ``slippage_amihud_scale`` interpretable as
    "additional bps of slippage per unit of *relative* illiquidity", we
    normalize each name's Amihud by the cross-sectional median before scaling.
    Thus scale=1.0 adds ~slippage_base extra bps for a name at median
    illiquidity, and proportionally more for less-liquid names.

    Round-trip multiplier:
        - Longs : one buy + one sell  -> 2 * cost_per_side
        - Shorts: one sell + one buy  -> 2 * cost_per_side  PLUS
                  borrow = borrow_rate_annual * (holding_days / 360)

    The total round-trip cost is subtracted from the per-event raw return
    before averaging up to the portfolio level. Borrow is ONLY applied to the
    short side (the bottom decile).

    Args:
        portfolio_ret:          Long-format portfolio returns with columns
                                ``[decile, event_window, port_ret_gross,
                                n_holdings]`` (output of
                                :func:`pead.portfolios.sort.compute_portfolio_returns`).
        portfolio_assignments:  Event-level assignments (output of
                                :func:`pead.portfolios.sort.assign_portfolio_deciles`).
        prices:                 Long-format daily prices (for Amihud).
        commission_bps:         Commission per side, in bps.
        slippage_base_bps:      Base slippage per side, in bps.
        slippage_amihud_scale:  Multiplier on (median-normalized) Amihud.
        borrow_cost_annual_bps: Annual borrow cost floor (shorts only).
        holding_days:           Holding period in days, for borrow accrual.

    Returns:
        ``portfolio_ret`` with an added ``port_ret_net`` column. ``port_ret_net``
        is ``port_ret_gross`` minus the value- (or equal-) weighted round-trip
        transaction cost for that decile.
    """
    for key in (Col.DECILE, "event_window", Col.PORTFOLIO_RET_GROSS, Col.N_HOLDINGS):
        if key not in portfolio_ret.columns:
            raise ValueError(f"portfolio_ret missing column {key!r}.")
    for key in (Col.TICKER, Col.ANNOUNCE_DATE, Col.DECILE):
        if key not in portfolio_assignments.columns:
            raise ValueError(f"portfolio_assignments missing column {key!r}.")

    # ── Per-ticker Amihud illiquidity ────────────────────────────────────
    amihud = compute_amihud_illiquidity(prices)

    # ── Attach Amihud to each event and normalize cross-sectionally ──────
    assign = portfolio_assignments[[Col.TICKER, Col.ANNOUNCE_DATE, Col.DECILE]].copy()
    if Col.MARKET_CAP in portfolio_assignments.columns:
        assign[Col.MARKET_CAP] = portfolio_assignments[Col.MARKET_CAP].astype(float)
    assign = assign.merge(amihud, on=Col.TICKER, how="left")
    assign[Col.AMIHUD_ILLIQUIDITY] = assign[Col.AMIHUD_ILLIQUIDITY].astype(float)

    # Median-normalize Amihud so scale is interpretable in bps.
    median_amihud = float(np.nanmedian(assign[Col.AMIHUD_ILLIQUIDITY].to_numpy()))
    if not np.isfinite(median_amihud) or median_amihud <= 0:
        median_amihud = 1.0
    assign["_amihud_norm"] = assign[Col.AMIHUD_ILLIQUIDITY].to_numpy() / median_amihud
    # Cap at a sane upper bound to prevent pathological names dominating.
    assign["_amihud_norm"] = np.nan_to_num(
        assign["_amihud_norm"].to_numpy(), nan=1.0, posinf=10.0, neginf=1.0
    )
    assign["_amihud_norm"] = np.clip(assign["_amihud_norm"].to_numpy(), 0.0, 10.0)

    # ── Per-event round-trip cost (fraction) ─────────────────────────────
    comm = _bps_to_frac(commission_bps)
    slip_base = _bps_to_frac(slippage_base_bps)
    slip_per_side = slip_base + slippage_amihud_scale * comm * assign["_amihud_norm"]
    cost_per_side = comm + slip_per_side
    round_trip = 2.0 * cost_per_side  # buy + sell for both legs

    # Borrow accrues only on the short side over the holding period.
    borrow_rate_annual = _bps_to_frac(borrow_cost_annual_bps)
    borrow = borrow_rate_annual * (holding_days / 360.0)

    assign["_tcost_frac"] = round_trip.to_numpy()
    # Determine the extreme deciles to mark the short leg.
    decile_vals = pd.Series(assign[Col.DECILE].dropna().astype(int).unique())
    if decile_vals.empty:
        raise ValueError("portfolio_assignments has no decile assignments.")
    bottom_decile = int(decile_vals.min())
    short_mask = assign[Col.DECILE].astype(float).to_numpy() == float(bottom_decile)
    assign.loc[short_mask, "_tcost_frac"] = (
        assign.loc[short_mask, "_tcost_frac"].to_numpy() + borrow
    )

    # ── Aggregate t-costs up to the decile level ─────────────────────────
    # Weighting mirrors compute_portfolio_returns: value-weight by market cap
    # if available, else equal-weight.
    has_mcap = Col.MARKET_CAP in assign.columns
    if has_mcap:
        w = assign[Col.MARKET_CAP].astype(float).to_numpy()
        w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    else:
        w = np.ones(len(assign), dtype=float)
    assign["_w"] = w

    def _weighted_tcost(g: pd.DataFrame) -> float:
        tc = g["_tcost_frac"].astype(float).to_numpy()
        wg = g["_w"].astype(float).to_numpy()
        mask = np.isfinite(tc) & np.isfinite(wg) & (wg > 0)
        if mask.sum() == 0:
            return np.nan
        wm = wg[mask] / wg[mask].sum()
        return float(np.sum(wm * tc[mask]))

    tcost_by_decile = (
        assign.groupby(Col.DECILE, observed=True)
        .apply(_weighted_tcost, include_groups=False)
        .reset_index()
        .rename(columns={0: "_tcost_decile"})
    )
    tcost_by_decile[Col.DECILE] = tcost_by_decile[Col.DECILE].astype(int)

    # ── Subtract t-costs from gross portfolio returns ────────────────────
    out = portfolio_ret.copy()
    out[Col.DECILE] = out[Col.DECILE].astype(int)
    out = out.merge(tcost_by_decile, on=Col.DECILE, how="left")
    out[Col.PORTFOLIO_RET_NET] = out[Col.PORTFOLIO_RET_GROSS] - out["_tcost_decile"]
    out = out.drop(columns=["_tcost_decile"])
    return out
