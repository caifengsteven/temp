import numpy as np
import pandas as pd


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def attention_series(posteriors: pd.DataFrame,
                     k1: float = 1.0,
                     a_max: float = 1.0) -> pd.Series:
    """
    Compute attention a_t based on posterior entropy (higher uncertainty -> more attention).
    a_t = min( a_max, k1 * H(p_t) / H_max ), where H_max = log(K).
    """
    K = posteriors.shape[1]
    H_max = np.log(K)
    vals = []
    for _, row in posteriors.iterrows():
        p = row.values
        H = entropy(p)
        a = k1 * (H / (H_max + 1e-12))
        vals.append(min(a, a_max))
    return pd.Series(vals, index=posteriors.index, name='attention')

