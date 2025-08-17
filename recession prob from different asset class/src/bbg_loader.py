from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from xbbg import blp
except Exception as e:
    blp = None


def require_xbbg():
    if blp is None:
        raise ImportError("xbbg is not installed or Bloomberg API not available. Please install 'xbbg' and ensure Bloomberg Terminal is running.")


def bdh_series(ticker: str, fld: str = "PX_LAST", start: str = "1990-01-01", end: Optional[str] = None, per: str = "M") -> pd.Series:
    require_xbbg()
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    df = blp.bdh(tickers=ticker, flds=fld, start_date=start, end_date=end, Per=per, Fill="P")
    # xbbg returns multiindex columns (fld, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        s = df[(fld, ticker)].copy()
    else:
        s = df.iloc[:, 0].copy()
    s = s.rename(ticker)
    s.index = pd.to_datetime(s.index)
    s = s.resample("M").last()
    return s.astype(float)


def load_bbg_series(bbg_cfg: Dict, start_year: int = 1990) -> Dict[str, pd.Series]:
    require_xbbg()
    start = f"{start_year}-01-01"
    out: Dict[str, pd.Series] = {}
    for key, ticker in bbg_cfg.get("tickers", {}).items():
        fld = bbg_cfg.get("fields", {}).get(key, "PX_LAST")
        try:
            out[key] = bdh_series(ticker, fld=fld, start=start, per="M")
        except Exception:
            # tolerate missing series
            pass
    return out

