"""
Volatility surface builder - calibrates model per expiry slice
and constructs the full surface with interpolation.
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Dict
from collections import defaultdict
import datetime

from .models import VolQuote, VolSurfaceData, CalibrationRequest, SABRParams, SVIParams
from .sabr import calibrate_sabr_slice, sabr_vol
from .svi import calibrate_svi_slice, svi_implied_vol


def _year_fraction(d1: datetime.date, d2: datetime.date) -> float:
    return (d2 - d1).days / 365.0


def build_surface(request: CalibrationRequest) -> VolSurfaceData:
    """Build a full vol surface from market quotes using the chosen model."""
    # Group quotes by expiry
    by_expiry: Dict[datetime.date, List[VolQuote]] = defaultdict(list)
    for q in request.quotes:
        by_expiry[q.expiry].append(q)

    expiries_sorted = sorted(by_expiry.keys())
    all_strikes = sorted(set(q.strike for q in request.quotes))

    # Build a finer strike grid for smooth surface
    k_min = min(all_strikes) * 0.9
    k_max = max(all_strikes) * 1.1
    strike_grid = np.linspace(k_min, k_max, 50).tolist()

    surface_vols: List[List[float]] = []
    params_list: List[Dict] = []
    fit_errors: List[float] = []
    today = datetime.date.today()

    for exp in expiries_sorted:
        quotes = by_expiry[exp]
        T = _year_fraction(today, exp)
        if T <= 0:
            continue

        forward = request.spot * math.exp((request.rate - request.dividend_yield) * T)

        if request.model == "sabr":
            params = calibrate_sabr_slice(
                quotes, forward, T, beta=request.beta, expiry=exp
            )
            # Generate vols on the strike grid
            vols_row = []
            for K in strike_grid:
                try:
                    v = sabr_vol(forward, K, T, params.alpha, params.beta,
                                 params.rho, params.nu)
                    vols_row.append(round(max(v, 0.001), 6))
                except Exception:
                    vols_row.append(0.0)
            params_list.append(params.model_dump())
            fit_errors.append(params.fit_error)

        elif request.model == "svi":
            params = calibrate_svi_slice(quotes, forward, T, expiry=exp)
            vols_row = []
            for K in strike_grid:
                try:
                    v = svi_implied_vol(K, forward, T, params.a, params.b,
                                        params.rho, params.m, params.sigma)
                    vols_row.append(round(max(v, 0.001), 6))
                except Exception:
                    vols_row.append(0.0)
            params_list.append(params.model_dump())
            fit_errors.append(params.fit_error)
        else:
            raise ValueError(f"Unknown model: {request.model}")

        surface_vols.append(vols_row)

    return VolSurfaceData(
        strikes=[round(k, 2) for k in strike_grid],
        expiries=[exp.isoformat() for exp in expiries_sorted if _year_fraction(today, exp) > 0],
        vols=surface_vols,
        model_type=request.model,
        params=params_list,
        fit_errors=fit_errors,
    )


def interpolate_vol(surface: VolSurfaceData, strike: float, expiry_str: str) -> float:
    """Interpolate vol from a fitted surface at arbitrary strike/expiry."""
    strikes = np.array(surface.strikes)
    if expiry_str in surface.expiries:
        idx = surface.expiries.index(expiry_str)
        vols = np.array(surface.vols[idx])
        return float(np.interp(strike, strikes, vols))
    # Linear interpolation between expiry slices
    # (simplified - would use flat vol between nearest slices)
    if len(surface.expiries) > 0:
        idx = 0  # fallback to nearest
        vols = np.array(surface.vols[idx])
        return float(np.interp(strike, strikes, vols))
    return 0.20  # fallback

