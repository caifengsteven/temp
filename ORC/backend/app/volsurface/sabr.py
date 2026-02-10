"""
SABR model implementation with Hagan et al. (2002) approximation.
Auto-calibration to market bid/ask vol quotes.
Pure Python Nelder-Mead optimizer (no scipy dependency).
"""
from __future__ import annotations
import math
import numpy as np
from typing import List
from .models import SABRParams, VolQuote
import datetime


def _nelder_mead_3d(func, x0, max_iter=3000, tol=1e-10):
    """Minimal Nelder-Mead for 3 parameters."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] += 0.1 * max(abs(pt[i]), 0.01)
        simplex.append(pt)
    fvals = [func(s) for s in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if abs(fvals[-1] - fvals[0]) < tol:
            break

        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        worst = simplex[-1]

        # Reflect
        xr = [2 * centroid[j] - worst[j] for j in range(n)]
        fr = func(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[0]:
            xe = [3 * centroid[j] - 2 * worst[j] for j in range(n)]
            fe = func(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        else:
            xc = [0.5 * (centroid[j] + worst[j]) for j in range(n)]
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [0.5 * (simplex[0][j] + simplex[i][j]) for j in range(n)]
                    fvals[i] = func(simplex[i])

    return simplex[0]


def sabr_vol(F: float, K: float, T: float, alpha: float, beta: float,
             rho: float, nu: float) -> float:
    """Hagan SABR implied vol approximation."""
    if T <= 0 or alpha <= 0:
        return alpha

    eps = 1e-10
    if abs(F - K) < eps:
        # ATM formula
        FK_beta = F ** (1 - beta)
        term1 = alpha / FK_beta
        term2 = ((1 - beta) ** 2 / 24.0) * (alpha ** 2) / (FK_beta ** 2)
        term3 = 0.25 * rho * beta * nu * alpha / FK_beta
        term4 = (2 - 3 * rho ** 2) * nu ** 2 / 24.0
        return term1 * (1 + (term2 + term3 + term4) * T)

    FK = F * K
    FK_beta_half = FK ** ((1 - beta) / 2.0)
    log_FK = math.log(F / K)

    z = (nu / alpha) * FK_beta_half * log_FK
    x_z = math.log((math.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    if abs(x_z) < eps:
        x_z = eps

    prefix = alpha / (FK_beta_half * (1 + (1 - beta) ** 2 / 24.0 * log_FK ** 2
                                       + (1 - beta) ** 4 / 1920.0 * log_FK ** 4))

    corr1 = ((1 - beta) ** 2 / 24.0) * alpha ** 2 / (FK ** (1 - beta))
    corr2 = 0.25 * rho * beta * nu * alpha / FK_beta_half
    corr3 = (2 - 3 * rho ** 2) * nu ** 2 / 24.0

    return prefix * (z / x_z) * (1 + (corr1 + corr2 + corr3) * T)


def calibrate_sabr_slice(
    quotes: List[VolQuote],
    forward: float,
    T: float,
    beta: float = 0.5,
    expiry: datetime.date = None,
) -> SABRParams:
    """Calibrate SABR params (alpha, rho, nu) to a set of market vol quotes for one expiry."""
    strikes = np.array([q.strike for q in quotes])
    market_vols = np.array([q.market_vol for q in quotes])

    # Weights: tighter bid-ask = higher weight
    weights = np.ones(len(quotes))
    for i, q in enumerate(quotes):
        if q.bid_vol is not None and q.ask_vol is not None:
            spread = q.ask_vol - q.bid_vol
            if spread > 0:
                weights[i] = 1.0 / spread

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        model_vols = np.array([
            sabr_vol(forward, k, T, alpha, beta, rho, nu) for k in strikes
        ])
        return float(np.sum(weights * (model_vols - market_vols) ** 2))

    # Initial guess: alpha ~ ATM vol * F^(beta-1)
    atm_vol = market_vols[np.argmin(np.abs(strikes - forward))]
    alpha0 = atm_vol * forward ** (1 - beta)

    alpha, rho, nu = _nelder_mead_3d(objective, [alpha0, -0.3, 0.4], max_iter=5000)
    rho = np.clip(rho, -0.999, 0.999)
    nu = max(nu, 0.001)
    alpha = max(alpha, 0.001)

    # Compute fit error (RMSE)
    model_vols = np.array([sabr_vol(forward, k, T, alpha, beta, rho, nu) for k in strikes])
    rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))

    return SABRParams(
        alpha=round(float(alpha), 6),
        beta=beta,
        rho=round(float(rho), 6),
        nu=round(float(nu), 6),
        expiry=expiry or datetime.date.today(),
        forward=forward,
        fit_error=round(rmse, 8),
    )

