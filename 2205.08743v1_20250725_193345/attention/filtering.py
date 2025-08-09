import numpy as np
import pandas as pd


def temper_posteriors(posteriors: pd.DataFrame, attention: pd.Series, k2: float = 1.0) -> pd.DataFrame:
    """
    Temper posteriors using alpha_t = 1 + k2 * a_t, then renormalize:
        p̂_t(i) ∝ p_t(i) ** alpha_t
    alpha_t >= 1 sharpens the distribution (lower effective noise).
    """
    assert posteriors.index.equals(attention.index)
    tempered = []
    for t in posteriors.index:
        p = posteriors.loc[t].values
        alpha = 1.0 + k2 * float(attention.loc[t])
        p2 = np.power(np.clip(p, 1e-12, 1.0), alpha)
        p2 = p2 / p2.sum()
        tempered.append(p2)
    return pd.DataFrame(tempered, index=posteriors.index, columns=posteriors.columns)

