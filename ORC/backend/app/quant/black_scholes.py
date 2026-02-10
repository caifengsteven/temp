"""
Analytical Black-Scholes pricing engine with full Greeks.
Supports European calls and puts with continuous dividends.
Uses pure Python math (no scipy) for Python 3.14 compatibility.
"""
from __future__ import annotations
import math
from .models import OptionType

# Pure Python normal distribution using math.erf
def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return _d1(S, K, r, q, sigma, T) - sigma * math.sqrt(T)


def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float,
             option_type: OptionType) -> float:
    """Black-Scholes option price."""
    if T <= 0:
        if option_type == OptionType.CALL:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = _d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == OptionType.CALL:
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_delta(S: float, K: float, r: float, q: float, sigma: float, T: float,
             option_type: OptionType) -> float:
    if T <= 0:
        if option_type == OptionType.CALL:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = _d1(S, K, r, q, sigma, T)
    if option_type == OptionType.CALL:
        return math.exp(-q * T) * _norm_cdf(d1)
    return -math.exp(-q * T) * _norm_cdf(-d1)


def bs_gamma(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, r, q, sigma, T)
    return math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """Vega per 1% vol move."""
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, r, q, sigma, T)
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T) * 0.01


def bs_theta(S: float, K: float, r: float, q: float, sigma: float, T: float,
             option_type: OptionType) -> float:
    """Theta per calendar day (negative = time decay)."""
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    sqrtT = math.sqrt(T)

    term1 = -S * math.exp(-q * T) * _norm_pdf(d1) * sigma / (2.0 * sqrtT)
    if option_type == OptionType.CALL:
        term2 = -r * K * math.exp(-r * T) * _norm_cdf(d2)
        term3 = q * S * math.exp(-q * T) * _norm_cdf(d1)
    else:
        term2 = r * K * math.exp(-r * T) * _norm_cdf(-d2)
        term3 = -q * S * math.exp(-q * T) * _norm_cdf(-d1)

    return (term1 + term2 + term3) / 365.0


def bs_rho(S: float, K: float, r: float, q: float, sigma: float, T: float,
           option_type: OptionType) -> float:
    """Rho per 1% rate move."""
    if T <= 0:
        return 0.0
    d2 = _d2(S, K, r, q, sigma, T)
    if option_type == OptionType.CALL:
        return K * T * math.exp(-r * T) * _norm_cdf(d2) * 0.01
    return -K * T * math.exp(-r * T) * _norm_cdf(-d2) * 0.01


def bs_vanna(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """Vanna = d(delta)/d(sigma) = d(vega)/d(S)."""
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    return -math.exp(-q * T) * _norm_pdf(d1) * d2 / sigma


def bs_volga(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """Volga (Vomma) = d(vega)/d(sigma) = d²V/dσ²."""
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    vega_1pct = bs_vega(S, K, r, q, sigma, T)
    return vega_1pct * d1 * d2 / sigma

