from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss


class MinimalEnsemble:
    def __init__(self, horizons: List[int]):
        self.horizons = horizons
        self.models = {H: {} for H in horizons}

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, start_train: str = None):
        Xy = X.join(Y, how="inner").dropna()
        if start_train:
            Xy = Xy[Xy.index >= pd.to_datetime(start_train)]
        for H in self.horizons:
            y = Xy[f"y_{H}"].astype(int).values
            Xmat = Xy[X.columns].values
            # Logistic with L2
            logi = LogisticRegression(max_iter=500, C=1.0, penalty="l2")
            # GBM
            gbm = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.03)
            # Calibrated versions
            logi_cal = CalibratedClassifierCV(logi, method="isotonic", cv=3)
            gbm_cal = CalibratedClassifierCV(gbm, method="isotonic", cv=3)
            logi_cal.fit(Xmat, y)
            gbm_cal.fit(Xmat, y)
            # Equal initial weights
            self.models[H] = {
                "logi": logi_cal,
                "gbm": gbm_cal,
                "weights": {"logi": 0.5, "gbm": 0.5},
                "features": list(X.columns),
            }
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = {}
        for H in self.horizons:
            m = self.models[H]
            XH = X[m["features"]].reindex(X.index).fillna(0.0).values
            p_logi = m["logi"].predict_proba(XH)[:, 1]
            p_gbm = m["gbm"].predict_proba(XH)[:, 1]
            w = m["weights"]
            preds[f"p_{H}"] = w["logi"] * p_logi + w["gbm"] * p_gbm
        return pd.DataFrame(preds, index=X.index)

    def update_weights_by_recent_performance(self, X: pd.DataFrame, Y: pd.DataFrame, window: int = 36):
        # Simple dynamic weighting based on recent Brier scores
        Xy = X.join(Y, how="inner").dropna()
        Xy = Xy.tail(window)
        for H in self.horizons:
            y = Xy[f"y_{H}"].astype(int).values
            m = self.models[H]
            XH = Xy[m["features"]].values
            p_logi = m["logi"].predict_proba(XH)[:, 1]
            p_gbm = m["gbm"].predict_proba(XH)[:, 1]
            b_logi = brier_score_loss(y, p_logi)
            b_gbm = brier_score_loss(y, p_gbm)
            # Convert to weights inversely proportional to loss
            eps = 1e-6
            s_logi = 1.0 / (b_logi + eps)
            s_gbm = 1.0 / (b_gbm + eps)
            z = s_logi + s_gbm
            self.models[H]["weights"] = {"logi": s_logi / z, "gbm": s_gbm / z}
        return self

