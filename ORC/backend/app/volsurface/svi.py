"""
SVI (Stochastic Volatility Inspired) model by Gatheral (2004).
Raw SVI parameterization: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
where w = total implied variance = sigma_bs^2 * T, k = log(K/F)
Pure Python optimization (no scipy dependency).
"""
from __future__ import annotations
import math
import random
import numpy as np
from typing import List
from .models import SVIParams, VolQuote
import datetime


def _nelder_mead_5d(func, x0, max_iter=3000, tol=1e-10):
    """Minimal Nelder-Mead for N parameters."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] += 0.05 * max(abs(pt[i]), 0.01)
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
        xr = [2 * centroid[j] - worst[j] for j in range(n)]
        fr = func(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[0]:
            xe = [3 * centroid[j] - 2 * worst[j] for j in range(n)]
            fe = func(xe)
            simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
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


def svi_total_variance(k: float, a: float, b: float, rho: float,
                       m: float, sigma: float) -> float:
    """Raw SVI total variance w(k)."""
    return a + b * (rho * (k - m) + math.sqrt((k - m) ** 2 + sigma ** 2))


def svi_implied_vol(K: float, F: float, T: float, a: float, b: float,
                    rho: float, m: float, sigma: float) -> float:
    """Convert SVI total variance to Black-Scholes implied vol."""
    k = math.log(K / F)
    w = svi_total_variance(k, a, b, rho, m, sigma)
    if w <= 0 or T <= 0:
        return 0.01
    return math.sqrt(max(w / T, 1e-8))


def calibrate_svi_slice(
    quotes: List[VolQuote],
    forward: float,
    T: float,
    expiry: datetime.date = None,
) -> SVIParams:
    """Calibrate SVI parameters to market vol quotes for one expiry."""
    strikes = np.array([q.strike for q in quotes])
    market_vols = np.array([q.market_vol for q in quotes])
    log_moneyness = np.log(strikes / forward)
    market_total_var = market_vols ** 2 * T

    weights = np.ones(len(quotes))
    for i, q in enumerate(quotes):
        if q.bid_vol is not None and q.ask_vol is not None:
            spread = q.ask_vol - q.bid_vol
            if spread > 0:
                weights[i] = 1.0 / spread

    def objective(params):
        a, b, rho, m, sigma = params
        if b < 0 or sigma < 0 or abs(rho) >= 1:
            return 1e10
        model_w = np.array([
            svi_total_variance(k, a, b, rho, m, sigma) for k in log_moneyness
        ])
        if np.any(model_w < 0):
            return 1e10
        return np.sum(weights * (model_w - market_total_var) ** 2)

    # Simple random search + Nelder-Mead for global optimization
    atm_var = float(market_total_var[np.argmin(np.abs(strikes - forward))])
    bounds = [
        (atm_var * 0.1, atm_var * 3),   # a
        (0.001, 2.0),                     # b
        (-0.99, 0.99),                    # rho
        (-0.5, 0.5),                      # m
        (0.01, 1.0),                      # sigma
    ]

    rng = random.Random(42)
    best_x = None
    best_f = float('inf')
    # Random search for good starting point
    for _ in range(500):
        x = [rng.uniform(lo, hi) for lo, hi in bounds]
        f = objective(x)
        if f < best_f:
            best_f = f
            best_x = x

    # Refine with Nelder-Mead
    a, b, rho, m, sigma = _nelder_mead_5d(objective, best_x, max_iter=3000)

    # Compute fit error
    model_vols = np.array([
        svi_implied_vol(K, forward, T, a, b, rho, m, sigma) for K in strikes
    ])
    rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))

    return SVIParams(
        a=round(float(a), 6),
        b=round(float(b), 6),
        rho=round(float(rho), 6),
        m=round(float(m), 6),
        sigma=round(float(sigma), 6),
        expiry=expiry or datetime.date.today(),
        fit_error=round(rmse, 8),
    )

