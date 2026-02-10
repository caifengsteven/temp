"""
Main pricing engine - computes full Greeks for option contracts.
Uses analytical Black-Scholes for Europeans and binomial tree for Americans.
"""
from __future__ import annotations
import math
import datetime
import numpy as np
from .models import (
    OptionContract, MarketData, Greeks, GreeksUSD,
    OptionType, ExerciseStyle,
)
from . import black_scholes as bs


def year_fraction(d1: datetime.date, d2: datetime.date) -> float:
    """ACT/365 day count."""
    return (d2 - d1).days / 365.0


def price_european(contract: OptionContract, market: MarketData) -> Greeks:
    """Price a European option analytically."""
    S = market.spot
    K = contract.strike
    r = market.rate
    q = market.dividend_yield
    sigma = market.volatility
    T = year_fraction(market.valuation_date, contract.expiry)

    price = bs.bs_price(S, K, r, q, sigma, T, contract.option_type)
    delta = bs.bs_delta(S, K, r, q, sigma, T, contract.option_type)
    gamma = bs.bs_gamma(S, K, r, q, sigma, T)
    vega = bs.bs_vega(S, K, r, q, sigma, T)
    theta = bs.bs_theta(S, K, r, q, sigma, T, contract.option_type)
    rho = bs.bs_rho(S, K, r, q, sigma, T, contract.option_type)
    vanna = bs.bs_vanna(S, K, r, q, sigma, T)
    volga = bs.bs_volga(S, K, r, q, sigma, T)

    # Charm: numerical via finite difference on delta
    dt = 1.0 / 365.0
    if T > dt:
        delta_tomorrow = bs.bs_delta(S, K, r, q, sigma, T - dt, contract.option_type)
        charm = (delta_tomorrow - delta) / dt / 365.0
    else:
        charm = 0.0

    return Greeks(
        price=round(price, 6),
        delta=round(delta, 6),
        gamma=round(gamma, 6),
        vega=round(vega, 6),
        theta=round(theta, 6),
        rho=round(rho, 6),
        vanna=round(vanna, 6),
        volga=round(volga, 6),
        charm=round(charm, 6),
        iv=sigma,
    )


def price_american_binomial(contract: OptionContract, market: MarketData,
                            steps: int = 200) -> Greeks:
    """Price an American option using Cox-Ross-Rubinstein binomial tree."""
    S = market.spot
    K = contract.strike
    r = market.rate
    q = market.dividend_yield
    sigma = market.volatility
    T = year_fraction(market.valuation_date, contract.expiry)

    if T <= 0:
        intrinsic = max(S - K, 0) if contract.option_type == OptionType.CALL else max(K - S, 0)
        return Greeks(price=intrinsic, delta=0, gamma=0, vega=0, theta=0,
                      rho=0, vanna=0, volga=0, charm=0, iv=sigma)

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    # Build price tree at expiry
    prices = np.zeros(steps + 1)
    for i in range(steps + 1):
        prices[i] = S * (u ** (steps - i)) * (d ** i)

    # Option values at expiry
    if contract.option_type == OptionType.CALL:
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)

    # Backward induction with early exercise
    for j in range(steps - 1, -1, -1):
        for i in range(j + 1):
            spot_ji = S * (u ** (j - i)) * (d ** i)
            hold = disc * (p * values[i] + (1 - p) * values[i + 1])
            if contract.option_type == OptionType.CALL:
                exercise = max(spot_ji - K, 0.0)
            else:
                exercise = max(K - spot_ji, 0.0)
            values[i] = max(hold, exercise)

    price = values[0]

    # Numerical Greeks via bumping
    bump_s = S * 0.01
    bump_v = 0.001
    bump_r = 0.0001

    def _price_bump(**kwargs):
        m = MarketData(**{**market.model_dump(), **kwargs})
        return price_american_binomial(
            contract, m, steps=steps // 2  # fewer steps for bumps
        ).price

    p_up = _price_bump(spot=S + bump_s)
    p_dn = _price_bump(spot=S - bump_s)
    delta = (p_up - p_dn) / (2 * bump_s)
    gamma = (p_up - 2 * price + p_dn) / (bump_s ** 2)

    pv_up = _price_bump(volatility=sigma + bump_v)
    pv_dn = _price_bump(volatility=max(sigma - bump_v, 0.001))
    vega = (pv_up - pv_dn) / (2 * bump_v) * 0.01

    pr_up = _price_bump(rate=r + bump_r)
    pr_dn = _price_bump(rate=r - bump_r)
    rho = (pr_up - pr_dn) / (2 * bump_r) * 0.01

    # Theta: reprice with 1 day less
    tomorrow = market.valuation_date + datetime.timedelta(days=1)
    p_t = _price_bump(valuation_date=tomorrow)
    theta = (p_t - price) / 1.0  # per day

    return Greeks(
        price=round(price, 6), delta=round(delta, 6), gamma=round(gamma, 6),
        vega=round(vega, 6), theta=round(theta, 6), rho=round(rho, 6),
        vanna=0.0, volga=0.0, charm=0.0, iv=sigma,
    )

