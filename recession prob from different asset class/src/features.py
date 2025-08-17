from typing import Dict
import pandas as pd
import numpy as np


def pct_change_months(s: pd.Series, m: int) -> pd.Series:
    return s.pct_change(m)


def zscore_expanding(s: pd.Series, winsor: float = 0.01) -> pd.Series:
    x = s.copy()
    lo = x.quantile(winsor)
    hi = x.quantile(1 - winsor)
    x = x.clip(lo, hi)
    mu = x.expanding().mean()
    sd = x.expanding().std().replace(0, np.nan)
    return (x - mu) / sd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    f = {}
    # Yield curve slopes
    if {"treasury_10y", "treasury_3m"}.issubset(df.columns):
        f["yc_10y_3m"] = df["treasury_10y"] - df["treasury_3m"]
    if {"treasury_10y", "treasury_2y"}.issubset(df.columns):
        f["yc_10y_2y"] = df["treasury_10y"] - df["treasury_2y"]

    # Equity momentum and vol gap
    if "spx_price" in df.columns:
        f["spx_ret_6m"] = pct_change_months(df["spx_price"], 6)
    if "vix" in df.columns and "spx_price" in df.columns:
        rv = df["spx_price"].pct_change().rolling(21).std() * (252 ** 0.5)
        rv_m = rv.resample("M").last()
        vix_ann = df["vix"]
        f["vix_rv_gap"] = vix_ann - rv_m

    # FX
    if "dxy" in df.columns:
        f["dxy_mom_3m"] = pct_change_months(df["dxy"], 3)

    # Commodities
    if set(["copper", "gold"]).issubset(df.columns):
        f["copper_gold"] = df["copper"] / df["gold"]
        f["copper_gold_mom_3m"] = pct_change_months(f["copper_gold"], 3)
    if "wti" in df.columns:
        f["wti_mom_3m"] = pct_change_months(df["wti"], 3)

    # Credit (optional if provided)
    if "ig_oas" in df.columns:
        f["ig_oas"] = df["ig_oas"]
    if "hy_oas" in df.columns:
        f["hy_oas"] = df["hy_oas"]
    if set(["hy_oas", "ig_oas"]).issubset(df.columns):
        f["hy_ig_spread"] = df["hy_oas"] - df["ig_oas"]

    # Vol indices
    if "move" in df.columns:
        f["move"] = df["move"]

    F = pd.DataFrame(f).sort_index()
    # Standardize with expanding z-score
    Fz = F.apply(zscore_expanding)
    return Fz

