from typing import List
import pandas as pd


def make_labels(nber_series: pd.Series, horizons: List[int], mode: str = "any") -> pd.DataFrame:
    """
    nber_series: monthly 0/1 indicator indexed by month-end
    mode: 'any' -> 1 if any month in next H months is recession
          'atH' -> 1 if recession at t+H
    Returns DataFrame with columns like 'y_3', 'y_6', ...
    """
    nber = nber_series.reindex(nber_series.index).fillna(0).astype(int)
    y = {}
    for H in horizons:
        if mode == "atH":
            y[f"y_{H}"] = nber.shift(-H)
        else:
            # any month in t+1..t+H
            any_fwd = nber.rolling(window=H, min_periods=1).max().shift(-H)
            y[f"y_{H}"] = any_fwd
    Y = pd.DataFrame(y)
    return Y

