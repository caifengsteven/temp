from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.isotonic import IsotonicRegression


class OnlineLogit:
    """Online logistic regression via SGD with log-loss.
    We'll calibrate with an online isotonic fit on a rolling window.
    """

    def __init__(self, alpha: float = 0.0005):
        self.clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, max_iter=1, tol=None, learning_rate="optimal")
        self.isofit = None
        self.classes_ = np.array([0, 1])

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.partial_fit(X, y, classes=self.classes_)
        return self

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        p = self.clf.predict_proba(X)[:, 1]
        return p

    def fit_calibration(self, p_raw: np.ndarray, y: np.ndarray):
        # Fit isotonic on latest window
        self.isofit = IsotonicRegression(out_of_bounds="clip")
        self.isofit.fit(p_raw, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba_raw(X)
        if self.isofit is not None:
            p = self.isofit.transform(p)
        return p


class DynamicEnsemble:
    def __init__(self, horizons: List[int], calibr_window: int = 60):
        self.horizons = horizons
        self.models = {H: OnlineLogit() for H in horizons}
        self.feature_names: List[str] = []
        self.calibr_window = calibr_window

    def fit_online(self, X: pd.DataFrame, Y: pd.DataFrame, warmup: int = 120):
        Xy = X.join(Y, how="inner").dropna().copy()
        self.feature_names = list(X.columns)
        # Expanding online fit
        for i in range(len(Xy)):
            xi = Xy.iloc[[i]][self.feature_names].values
            for H in self.horizons:
                yi = Xy.iloc[[i]][f"y_{H}"].astype(int).values
                if i < warmup:
                    # accumulate without calibration
                    self.models[H].partial_fit(xi, yi)
                else:
                    # predict, then update
                    self.models[H].partial_fit(xi, yi)
                    # periodic calibration on recent window
                    if i % 3 == 0:
                        win = Xy.iloc[max(0, i - self.calibr_window + 1): i + 1]
                        Xw = win[self.feature_names].values
                        yw = win[f"y_{H}"].astype(int).values
                        p_raw = self.models[H].predict_proba_raw(Xw)
                        self.models[H].fit_calibration(p_raw, yw)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = {}
        Xmat = X[self.feature_names].reindex(X.index).fillna(0.0).values
        for H in self.horizons:
            preds[f"p_{H}"] = self.models[H].predict_proba(Xmat)
        return pd.DataFrame(preds, index=X.index)

