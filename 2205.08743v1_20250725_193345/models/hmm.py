from dataclasses import dataclass
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class HMMConfig:
    n_regimes: int = 3
    cov_type: str = 'full'
    n_iter: int = 200
    random_state: int = 42


class RegimeHMM:
    def __init__(self, cfg: HMMConfig):
        self.cfg = cfg
        self.model = GaussianHMM(n_components=cfg.n_regimes,
                                 covariance_type=cfg.cov_type,
                                 n_iter=cfg.n_iter,
                                 random_state=cfg.random_state)
        self.fitted = False
        self.means_ = None
        self.vars_ = None
        self.trans_ = None

    def fit_on_agg(self, returns: pd.DataFrame):
        # Fit univariate HMM on equal-weight average return
        r = returns.mean(axis=1).values.reshape(-1, 1)
        self.model.fit(r)
        self.fitted = True
        self.means_ = self.model.means_.flatten()
        self.vars_ = np.array([np.diag(cov)[0] if cov.ndim == 2 else cov for cov in self.model.covars_])
        self.trans_ = self.model.transmat_

    def filter_posteriors(self, returns: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("HMM not fitted")
        r = returns.mean(axis=1).values.reshape(-1, 1)
        # Compute posterior (gamma) per step using score_samples
        logprob, posteriors = self.model.score_samples(r)
        gam = pd.DataFrame(posteriors, index=returns.index, columns=[f"regime_{i}" for i in range(self.cfg.n_regimes)])
        return gam

    @staticmethod
    def per_regime_stats(returns: pd.DataFrame, regime_probs: pd.DataFrame):
        # Weighted estimates of mean and covariance for each regime
        T, N = returns.shape
        K = regime_probs.shape[1]
        mu_list, cov_list = [], []
        X = returns.values
        for k in range(K):
            w = regime_probs.iloc[:, k].values.reshape(-1, 1)
            w /= (w.sum() + 1e-12)
            mu_k = (w * X).sum(axis=0)
            Xc = X - mu_k
            cov_k = (w * Xc).T @ Xc  # weighted second moment
            cov_k = (cov_k + cov_k.T) / 2.0
            mu_list.append(mu_k)
            cov_list.append(cov_k)
        return np.array(mu_list), np.array(cov_list)

