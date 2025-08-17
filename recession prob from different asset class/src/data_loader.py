import os
from typing import Dict, Optional
import pandas as pd
import numpy as np
from fredapi import Fred
import yaml

from .bbg_loader import load_bbg_series


def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_fred() -> Optional[Fred]:
    key = os.environ.get("FRED_API_KEY")
    try:
        fred = Fred(api_key=key) if key else Fred()
        # simple ping
        _ = fred.get_series("USREC").tail(1)
        return fred
    except Exception:
        return None


def fred_series(fred: Fred, code: str, freq: str = "M") -> pd.Series:
    s = fred.get_series(code)
    s = pd.Series(s, name=code)
    s.index = pd.to_datetime(s.index)
    if freq == "M":
        s = s.resample("M").last()
    return s.astype(float)


def read_local_csv(path: str) -> Optional[pd.Series]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Date" in df.columns:
        idx = pd.to_datetime(df["Date"])  # tolerate many formats
    else:
        # try index
        df.index = pd.to_datetime(df.index)
        idx = df.index
    # assume value column
    val_col = "value" if "value" in df.columns else df.columns[-1]
    s = pd.Series(df[val_col].values, index=idx, name=os.path.splitext(os.path.basename(path))[0])
    s = s.resample("M").last()
    return s.astype(float)


def load_all_series(cfg: Dict) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}

    # 1) Local CSV overrides
    for key, path in cfg.get("local_csv", {}).items():
        s = read_local_csv(path)
        if s is not None:
            out[key] = s

    # 2) Bloomberg (if configured and available)
    try:
        bbg_cfg = cfg.get("bloomberg", {})
        bbg_data = load_bbg_series(bbg_cfg, start_year=cfg.get("start_year", 1990))
        for k, s in bbg_data.items():
            out.setdefault(k, s)
    except Exception:
        pass

    # 3) FRED for remaining series (and always for NBER label)
    fred = get_fred()
    for key, code in cfg.get("fred", {}).items():
        if key in out:
            continue
        try:
            if fred is None:
                raise RuntimeError("FRED unavailable")
            out[key] = fred_series(fred, code, freq="M")
        except Exception:
            pass

    return out


def align_monthly(data: Dict[str, pd.Series], start_year: int = 1990) -> pd.DataFrame:
    df = pd.concat(data.values(), axis=1)
    df.columns = list(data.keys())
    df = df.sort_index().resample("M").last()
    df = df[df.index >= pd.Timestamp(start_year, 1, 1)]
    return df

