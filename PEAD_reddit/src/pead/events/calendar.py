"""
Trading-day calendar utilities for the PEAD event study.

The event study is conducted in *event time* (trading-day offsets relative to
the announcement day, day 0). These helpers translate between event-time
offsets and calendar dates using the trading calendar embedded in the daily
prices panel.

The calendar is derived empirically from the prices DataFrame (the union of all
observed trading dates) so the module works without an external holiday
calendar — useful both for the synthetic generator and for Bloomberg data that
already encodes exchange holidays by omission.

References
----------
MacKinlay (1997), "Event Studies in Economics and Finance," JEL 35(1).
"""

from __future__ import annotations

import logging

import pandas as pd

from pead.schema import Col

logger = logging.getLogger(__name__)


def build_trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract sorted unique trading dates from the prices DataFrame.

    The calendar is the union of every ``trading_date`` observed across all
    tickers. It is sorted ascending so that positional offsets correspond to
    trading-day (not calendar-day) steps.

    Parameters
    ----------
    prices:
        Daily prices panel with at least the ``trading_date`` column.

    Returns
    -------
    pd.DatetimeIndex
        Sorted unique trading dates (tz-naive, midnight-normalized).
    """
    if Col.TRADING_DATE not in prices.columns:
        raise ValueError(f"prices must contain column {Col.TRADING_DATE!r}")
    dates = pd.DatetimeIndex(pd.unique(prices[Col.TRADING_DATE].values)).sort_values()
    return dates.normalize()


def snap_to_trading_day(
    date: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    direction: str = "forward",
) -> pd.Timestamp:
    """Snap a date to a trading day on ``calendar``.

    If ``date`` is already a trading day it is returned unchanged.

    Parameters
    ----------
    date:
        The date to snap.
    calendar:
        Sorted trading calendar.
    direction:
        ``"forward"`` → next trading day if not a trading day
        (i.e. first trading day >= ``date``).
        ``"backward"`` → previous trading day
        (i.e. last trading day <= ``date``).

    Returns
    -------
    pd.Timestamp
        The snapped trading day (normalized to midnight).
    """
    if direction not in ("forward", "backward"):
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
    if len(calendar) == 0:
        raise ValueError("calendar is empty")

    date = pd.Timestamp(date).normalize()
    # searchsorted with side='left' gives the insertion index; calendar[pos] is the
    # first element >= date (when side='left' and dates are unique).
    pos = calendar.searchsorted(date, side="left")

    # Exact match -> already a trading day.
    if pos < len(calendar) and calendar[pos].normalize() == date:
        return date

    if direction == "forward":
        if pos < len(calendar):
            return pd.Timestamp(calendar[pos]).normalize()
        logger.warning(
            "Date %s is on/after the last calendar day %s; returning last trading day",
            date,
            calendar[-1],
        )
        return pd.Timestamp(calendar[-1]).normalize()
    # backward
    if pos > 0:
        return pd.Timestamp(calendar[pos - 1]).normalize()
    logger.warning(
        "Date %s is before the first calendar day %s; returning first trading day",
        date,
        calendar[0],
    )
    return pd.Timestamp(calendar[0]).normalize()


def trading_day_offset(
    announce_date: pd.Timestamp,
    offset: int,
    calendar: pd.DatetimeIndex,
) -> pd.Timestamp:
    """Return the trading date ``offset`` trading days from ``announce_date``.

    Day 0 is the announcement trading day. If ``announce_date`` is not itself a
    trading day it is first snapped *forward* to the next trading day (the
    convention used for earnings announcements, which is also how the synthetic
    generator injects post-announcement drift).

    Out-of-range offsets are clamped to the nearest available trading day so
    that callers near the calendar boundary receive a valid date; the
    return-computation layer is responsible for detecting missing firm data
    (e.g. delistings) inside the resulting window.

    Parameters
    ----------
    announce_date:
        Announcement date (need not be a trading day).
    offset:
        Trading-day offset. 0 → announcement trading day, +1 → next, −1 → prev.
    calendar:
        Sorted trading calendar.

    Returns
    -------
    pd.Timestamp
        The target trading day.
    """
    base = snap_to_trading_day(announce_date, calendar, direction="forward")
    base_idx = calendar.get_loc(base)
    # get_loc on a unique sorted DatetimeIndex returns a scalar int.
    base_idx = int(base_idx)
    target_idx = base_idx + int(offset)
    target_idx = max(0, min(len(calendar) - 1, target_idx))
    return pd.Timestamp(calendar[target_idx]).normalize()


def get_window_dates(
    announce_date: pd.Timestamp,
    start_offset: int,
    end_offset: int,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the ``(start_date, end_date)`` trading dates for an event window.

    Both bounds are inclusive trading days, expressed as offsets from the
    announcement trading day (day 0).

    Parameters
    ----------
    announce_date:
        Announcement date.
    start_offset, end_offset:
        Trading-day offsets relative to the announcement trading day.
        ``start_offset`` must be <= ``end_offset``.
    calendar:
        Sorted trading calendar.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        ``(start_date, end_date)``.
    """
    if start_offset > end_offset:
        raise ValueError(f"start_offset ({start_offset}) > end_offset ({end_offset})")
    start_date = trading_day_offset(announce_date, start_offset, calendar)
    end_date = trading_day_offset(announce_date, end_offset, calendar)
    return start_date, end_date
