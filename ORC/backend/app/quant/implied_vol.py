"""
Implied volatility solver using Brent's method (pure Python).
Falls back to py_vollib for fast rational approximation when available.
"""
from __future__ import annotations
import math
from .models import OptionType
from . import black_scholes as bs


def _brentq(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Pure Python Brent's method root finder."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Root not bracketed")
    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        m = 0.5 * (c - b)
        if abs(m) <= tol or abs(fb) < tol:
            return b
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if abs(a - c) < tol:
                p = 2.0 * m * s
                q_ = 1.0 - s
            else:
                q_ = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q_ * (q_ - r) - (b - a) * (r - 1.0))
                q_ = (q_ - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q_ = -q_
            else:
                p = -p
            if 2.0 * p < min(3.0 * m * q_ - abs(tol * q_), abs(e * q_)):
                e, d = d, p / q_
            else:
                d = e = m
        else:
            d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = f(b)
    return b


def implied_vol_brent(
    market_price: float,
    S: float, K: float, r: float, q: float, T: float,
    option_type: OptionType,
    vol_low: float = 0.001,
    vol_high: float = 5.0,
    tol: float = 1e-10,
) -> float:
    """Solve for implied vol using Brent's method on the BS pricing function."""
    if T <= 0:
        raise ValueError("Cannot compute IV for expired option")

    intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0) if option_type == OptionType.CALL \
        else max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)

    if market_price <= intrinsic + tol:
        raise ValueError(f"Market price {market_price:.4f} at or below intrinsic {intrinsic:.4f}")

    def objective(sigma):
        return bs.bs_price(S, K, r, q, sigma, T, option_type) - market_price

    try:
        return _brentq(objective, vol_low, vol_high, tol=tol, max_iter=200)
    except ValueError:
        raise ValueError(
            f"Could not find IV in [{vol_low}, {vol_high}] for price={market_price:.4f}, "
            f"S={S}, K={K}, T={T:.4f}"
        )


def implied_vol_newton(
    market_price: float,
    S: float, K: float, r: float, q: float, T: float,
    option_type: OptionType,
    initial_guess: float = 0.25,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson IV solver using vega as the derivative."""
    sigma = initial_guess
    for _ in range(max_iter):
        price = bs.bs_price(S, K, r, q, sigma, T, option_type)
        vega_100 = bs.bs_vega(S, K, r, q, sigma, T)  # per 1% move
        vega = vega_100 * 100.0  # convert to per 100% = per unit sigma

        if abs(vega) < 1e-15:
            break

        diff = market_price - price
        if abs(diff) < tol:
            return sigma

        sigma += diff / vega
        sigma = max(sigma, 0.001)
        sigma = min(sigma, 5.0)

    return sigma


def implied_vol(
    market_price: float,
    S: float, K: float, r: float, q: float, T: float,
    option_type: OptionType,
) -> float:
    """Best-effort IV solver: tries Newton first, falls back to Brent."""
    try:
        iv = implied_vol_newton(market_price, S, K, r, q, T, option_type)
        # Verify
        reprice = bs.bs_price(S, K, r, q, iv, T, option_type)
        if abs(reprice - market_price) < 1e-6:
            return iv
    except Exception:
        pass

    return implied_vol_brent(market_price, S, K, r, q, T, option_type)

