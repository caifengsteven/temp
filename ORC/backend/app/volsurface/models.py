"""Data models for volatility surface."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import datetime


class VolQuote(BaseModel):
    """A single market vol quote (or price) for calibration."""
    strike: float
    expiry: datetime.date
    bid_vol: Optional[float] = None
    ask_vol: Optional[float] = None
    mid_vol: Optional[float] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    option_type: str = "call"

    @property
    def market_vol(self) -> float:
        if self.mid_vol is not None:
            return self.mid_vol
        if self.bid_vol is not None and self.ask_vol is not None:
            return (self.bid_vol + self.ask_vol) / 2.0
        return 0.0


class SABRParams(BaseModel):
    """SABR model parameters for a single expiry slice."""
    alpha: float = Field(description="Initial vol level")
    beta: float = Field(default=0.5, description="CEV exponent [0,1]")
    rho: float = Field(description="Correlation [-1,1]")
    nu: float = Field(description="Vol of vol")
    expiry: datetime.date
    forward: float
    fit_error: float = 0.0


class SVIParams(BaseModel):
    """SVI (Stochastic Volatility Inspired) raw parameters."""
    a: float = Field(description="Overall variance level")
    b: float = Field(description="Slope of the wings")
    rho: float = Field(description="Rotation [-1,1]")
    m: float = Field(description="Translation")
    sigma: float = Field(description="ATM curvature")
    expiry: datetime.date
    fit_error: float = 0.0


class VolSurfaceData(BaseModel):
    """Full vol surface data for frontend visualization."""
    strikes: List[float]
    expiries: List[str]
    vols: List[List[float]]  # [expiry_idx][strike_idx]
    model_type: str
    params: List[Dict]
    fit_errors: List[float]


class CalibrationRequest(BaseModel):
    symbol: str
    spot: float
    rate: float = 0.05
    dividend_yield: float = 0.0
    quotes: List[VolQuote]
    model: str = Field(default="sabr", description="sabr or svi")
    beta: float = Field(default=0.5, description="SABR beta (fixed)")

