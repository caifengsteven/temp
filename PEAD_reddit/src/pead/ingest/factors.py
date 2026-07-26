"""
Factor data loading for the PEAD asymmetry pipeline.

Provides the Fama-French 5-factor + momentum + risk-free return series used to
risk-adjust event and portfolio returns.  Three sources are attempted in order:

1. A live Bloomberg terminal (via ``pdblp``/``xbbg``) if a connection is given.
2. Kenneth French's public data library (downloaded and parsed on the fly).
3. The deterministic synthetic factor generator
   (:func:`pead.synthetic.generate_synthetic_factors`) as an offline fallback.

Design rules mirror :mod:`pead.ingest.bloomberg`: Bloomberg imports are lazy,
all output DataFrames conform to the canonical schema in
:class:`pead.schema.Col`, and everything is validated.
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Any

import pandas as pd

from pead.schema import Col, validate_dataframe
from pead.synthetic import generate_synthetic_factors

logger = logging.getLogger(__name__)

# Canonical column order for the factor table.
FACTOR_COLUMNS: list[str] = [
    Col.CALENDAR_DATE,
    Col.MKT_RF,
    Col.SMB,
    Col.HML,
    Col.RMW,
    Col.CMA,
    Col.MOM,
    Col.RF,
]

# Kenneth French data library — monthly FF5 factors (2x3 breakpoints).
FF5_MONTHLY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_monthly_CSV.zip"
)
# Monthly momentum factor (used for the ``mom`` column).
MOM_MONTHLY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"  # daily; we resample to monthly below
)
# A monthly momentum archive (cleaner match for monthly FF5).
MOM_MONTHLY_CSV_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
)

# Bloomberg-equivalent custom factor tickers (optional path).
BBG_FACTOR_TICKERS = {
    Col.MKT_RF: "FF_MKT_RF Index",
    Col.SMB: "FF_SMB Index",
    Col.HML: "FF_HML Index",
    Col.RMW: "FF_RMW Index",
    Col.CMA: "FF_CMA Index",
    Col.MOM: "FF_MOM Index",
    Col.RF: "FF_RF Index",
}


# ─── Public API ─────────────────────────────────────────────────────────────


def fetch_ff5_factors(
    start_date: str,
    end_date: str,
    conn: Any = None,
) -> pd.DataFrame:
    """Fetch FF5 + momentum + risk-free factor returns.

    Resolution order:

    1. **Bloomberg** — if ``conn`` is provided (or ``pdblp`` is importable) we
       pull the ``FF_*`` custom indices.  This is the fastest path on a desk
       with a terminal.
    2. **Kenneth French data library** — download and parse the public monthly
       CSVs.  Used in CI / laptops.
    3. **Synthetic fallback** — :func:`pead.synthetic.generate_synthetic_factors`.
       Used when the network is unavailable so the pipeline never hard-fails.

    Returns
    -------
    DataFrame with columns: ``calendar_date, mkt_rf, smb, hml, rmw, cma, mom,
    rf`` (monthly, in decimal returns) sorted ascending by date.
    """
    # Try Bloomberg first.
    if conn is not None or _pdblp_available():
        try:
            return _fetch_factors_bloomberg(start_date, end_date, conn)
        except Exception as exc:  # pragma: no cover - needs Bloomberg
            logger.warning("Bloomberg factor fetch failed (%s); falling back", exc)

    # Try Kenneth French data library.
    try:
        ff5 = _load_ff5_monthly(FF5_MONTHLY_URL)
        mom = _load_momentum_monthly(MOM_MONTHLY_CSV_URL)
        out = _merge_ff5_and_momentum(ff5, mom, start_date, end_date)
        logger.info("Loaded FF5 + momentum from Kenneth French data library")
        return out
    except Exception as exc:
        logger.warning("Kenneth French download failed (%s); using synthetic", exc)

    # Final fallback: synthetic.
    out = generate_synthetic_factors(start=start_date, end=end_date)
    out = out[FACTOR_COLUMNS]
    out[Col.CALENDAR_DATE] = pd.to_datetime(out[Col.CALENDAR_DATE])
    _validate_factors(out, name="fetch_ff5_factors (synthetic)")
    return out


def load_factors_from_parquet(path: str) -> pd.DataFrame:
    """Load pre-downloaded factor returns from a parquet file.

    The parquet file is expected to contain (at least) the canonical factor
    columns defined by :data:`FACTOR_COLUMNS`.  ``calendar_date`` is coerced to
    ``datetime64[ns]`` and the frame is returned sorted ascending by date.

    Raises
    ------
    ValueError
        If any required factor column is missing.
    """
    df = pd.read_parquet(path)
    if Col.CALENDAR_DATE not in df.columns:
        # Allow files indexed by date as a convenience.
        if df.index.name == Col.CALENDAR_DATE:
            df = df.reset_index()
        else:
            raise ValueError(f"Factor parquet '{path}' has no '{Col.CALENDAR_DATE}' column")

    df[Col.CALENDAR_DATE] = pd.to_datetime(df[Col.CALENDAR_DATE])
    df = df.sort_values(Col.CALENDAR_DATE).reset_index(drop=True)

    # Keep canonical columns if present; preserve any extras at the end.
    present = [c for c in FACTOR_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in FACTOR_COLUMNS]
    df = df[present + extras]

    _validate_factors(df, name=f"load_factors_from_parquet('{path}')")
    return df


def save_factors_to_parquet(df: pd.DataFrame, path: str) -> None:
    """Persist a factor DataFrame to parquet (round-trip companion to load).

    Convenience wrapper that ensures the canonical column order and datetime
    dtype before writing.
    """
    out = df.copy()
    if Col.CALENDAR_DATE in out.columns:
        out[Col.CALENDAR_DATE] = pd.to_datetime(out[Col.CALENDAR_DATE])
        out = out.sort_values(Col.CALENDAR_DATE).reset_index(drop=True)
    # Reorder to canonical, keep extras.
    present = [c for c in FACTOR_COLUMNS if c in out.columns]
    extras = [c for c in out.columns if c not in FACTOR_COLUMNS]
    out = out[present + extras]
    out.to_parquet(path, index=False)
    logger.info("Wrote %d factor rows to %s", len(out), path)


def mock_factors(
    start_date: str = "2014-06-01",
    end_date: str = "2025-12-31",
    conn: Any = None,  # noqa: ARG001 - API parity with fetch_ff5_factors
    seed: int = 42,
) -> pd.DataFrame:
    """Return deterministic synthetic factor returns (no network / Bloomberg).

    Thin wrapper around
    :func:`pead.synthetic.generate_synthetic_factors` with schema validation.
    """
    out = generate_synthetic_factors(start=start_date, end=end_date, seed=seed)
    out = out[FACTOR_COLUMNS]
    out[Col.CALENDAR_DATE] = pd.to_datetime(out[Col.CALENDAR_DATE])
    _validate_factors(out, name="mock_factors")
    return out


# ─── Bloomberg path ─────────────────────────────────────────────────────────


def _pdblp_available() -> bool:
    """Return True if ``pdblp`` is importable (does not start a session)."""
    try:
        import pdblp  # noqa: F401 - lazy availability probe
    except ImportError:
        return False
    return True


def _fetch_factors_bloomberg(
    start_date: str,
    end_date: str,
    conn: Any,
) -> pd.DataFrame:
    """Pull the ``FF_*`` custom indices from Bloomberg via pdblp/xbbg.

    Each factor is a total-return index series; we convert index levels to
    simple monthly returns.  ``conn`` may be ``None`` here only if ``pdblp`` is
    importable — we start a session in that case.
    """
    import pdblp  # lazy: only needed on a Bloomberg-equipped host

    con = conn
    if con is None:
        con = pdblp.BCon(timeout=5000)
        con.start()

    tickers = list(BBG_FACTOR_TICKERS.values())
    field = "PX_LAST"
    raw = con.bdh(tickers, field, start_date, end_date)

    # Flatten the MultiIndex columns into [date, factor, level].
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw.index = pd.to_datetime(raw.index)

    # Convert index levels to monthly returns (last-to-last).
    monthly = raw.resample("ME").last()
    rets = monthly.pct_change()

    out = pd.DataFrame({Col.CALENDAR_DATE: rets.index})
    for col, tkr in BBG_FACTOR_TICKERS.items():
        out[col] = rets[tkr].to_numpy() if tkr in rets.columns else pd.NA

    out = out.dropna(subset=[Col.CALENDAR_DATE]).reset_index(drop=True)
    out = out[FACTOR_COLUMNS]
    _validate_factors(out, name="_fetch_factors_bloomberg")
    return out


# ─── Kenneth French parsing ─────────────────────────────────────────────────


def _read_zipped_csv_from_url(url: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Download a zipped CSV from Kenneth French's library and return raw text.

    Returns the *raw* concatenated text of every CSV inside the archive so the
    caller can pick the right table via the skip-footer logic.
    """
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - trusted URL
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV inside archive at {url}")
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh, encoding=encoding)


def _load_ff5_monthly(url: str) -> pd.DataFrame:
    """Parse the monthly FF5 + RF CSV from Kenneth French's library.

    The FF5 archive contains one CSV whose header row starts after a few
    descriptor lines.  Values are in **percent**; we convert to decimals.
    """
    raw = _read_zipped_csv_from_url(url)
    # The first column is the YYYYMM period; the rest are the factor columns.
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = raw["date"].astype(str).str.strip()
    # Keep only rows whose first column parses as a 6-digit YYYYMM.
    raw = raw[raw["date"].str.fullmatch(r"\d{6}")].copy()
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m") + pd.offsets.MonthEnd(0)

    rename = {
        "Mkt-RF": Col.MKT_RF,
        "SMB": Col.SMB,
        "HML": Col.HML,
        "RMW": Col.RMW,
        "CMA": Col.CMA,
        "RF": Col.RF,
    }
    raw = raw.rename(columns=rename)
    factor_cols = [c for c in rename.values() if c in raw.columns]
    for c in factor_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce") / 100.0
    return raw[["date", *factor_cols]].dropna().reset_index(drop=True)


def _load_momentum_monthly(url: str) -> pd.DataFrame:
    """Parse the monthly momentum factor (Mom) from Kenneth French's library.

    The momentum archive contains both a monthly and a daily table; we use the
    monthly table.  Values are in **percent**; we convert to decimals.
    """
    raw = _read_zipped_csv_from_url(url)
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = raw["date"].astype(str).str.strip()
    raw = raw[raw["date"].str.fullmatch(r"\d{6}")].copy()
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m") + pd.offsets.MonthEnd(0)

    # Momentum column is usually named "Mom   " (trailing spaces) — match loosely.
    mom_col = next(
        (c for c in raw.columns if c != "date" and c.strip().lower().startswith("mom")),
        None,
    )
    if mom_col is None and raw.shape[1] >= 2:
        mom_col = raw.columns[1]
    raw[Col.MOM] = pd.to_numeric(raw[mom_col], errors="coerce") / 100.0
    return raw[["date", Col.MOM]].dropna().reset_index(drop=True)


def _merge_ff5_and_momentum(
    ff5: pd.DataFrame,
    mom: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Left-join FF5 with momentum on month-end date and clip to the window."""
    out = ff5.merge(mom, on="date", how="left")
    out[Col.MOM] = out[Col.MOM].fillna(0.0)
    out = out.rename(columns={"date": Col.CALENDAR_DATE})

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    out = out[(out[Col.CALENDAR_DATE] >= start_ts) & (out[Col.CALENDAR_DATE] <= end_ts)]
    out = out.sort_values(Col.CALENDAR_DATE).reset_index(drop=True)

    # Canonical column order.
    out = out[FACTOR_COLUMNS]
    _validate_factors(out, name="_merge_ff5_and_momentum")
    return out


# ─── Validation ─────────────────────────────────────────────────────────────


def _validate_factors(df: pd.DataFrame, name: str = "factors") -> None:
    """Validate a factor DataFrame against the canonical schema."""
    validate_dataframe(
        df,
        FACTOR_COLUMNS,
        name=name,
        strict_dates=[Col.CALENDAR_DATE],
    )
