import numpy as np


def mv_weights(mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float = 5.0, ridge: float = 1e-6) -> np.ndarray:
    """
    Long/short mean–variance with budget constraint sum(w)=1 (weights can be negative).
    Solves: maximize mu' w - (risk_aversion/2) w' Sigma w, subject to 1' w = 1.
    Closed form with Lagrange multiplier:
      w = (1/risk_aversion) * inv(Sigma) * (mu - lambda * 1),
      where lambda chosen so that 1' w = 1.
    """
    N = mu.shape[0]
    Sigma_reg = Sigma + ridge * np.eye(N)
    try:
        inv = np.linalg.inv(Sigma_reg)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sigma_reg)
    one = np.ones(N)
    ra = max(risk_aversion, 1e-6)
    inv_mu = inv @ mu
    inv_1 = inv @ one
    num = one @ inv_mu - ra  # 1' inv Σ μ - ra
    den = one @ inv_1        # 1' inv Σ 1
    lam = num / (den + 1e-12)
    w = (inv_mu - lam * inv_1) / ra
    # If numerical issues produce NaNs, fall back to equal weight
    if not np.all(np.isfinite(w)):
        w = np.ones(N) / N
    return w

