from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


def evaluate_probs(P: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = P.index.intersection(Y.index)
    for col in P.columns:
        H = int(col.split("_")[1])
        y = Y.loc[idx, f"y_{H}"].astype(int)
        p = P.loc[idx, col].clip(1e-6, 1 - 1e-6)
        try:
            auc = roc_auc_score(y, p)
        except ValueError:
            auc = float("nan")
        brier = brier_score_loss(y, p)
        rows.append({"horizon": H, "auc": auc, "brier": brier})
    return pd.DataFrame(rows).set_index("horizon").sort_index()

