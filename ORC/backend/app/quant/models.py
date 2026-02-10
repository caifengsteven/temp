"""Data models for the quant library."""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
import datetime


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class OptionContract(BaseModel):
    """Represents a single option contract."""
    symbol: str = Field(..., description="Underlying symbol e.g. AAPL")
    option_type: OptionType
    strike: float = Field(..., gt=0)
    expiry: datetime.date
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    multiplier: float = Field(default=100.0, description="Contract multiplier")


class MarketData(BaseModel):
    """Market data snapshot for pricing."""
    spot: float = Field(..., gt=0, description="Underlying spot price")
    rate: float = Field(default=0.05, description="Risk-free rate (annualized)")
    dividend_yield: float = Field(default=0.0, description="Continuous dividend yield")
    volatility: float = Field(..., gt=0, le=5.0, description="Implied volatility")
    valuation_date: datetime.date = Field(default_factory=datetime.date.today)


class Greeks(BaseModel):
    """Full Greeks output from pricing."""
    price: float = Field(description="Option theoretical price")
    delta: float = Field(description="dV/dS")
    gamma: float = Field(description="d²V/dS²")
    vega: float = Field(description="dV/dσ (per 1% vol move)")
    theta: float = Field(description="dV/dt (per day)")
    rho: float = Field(description="dV/dr (per 1% rate move)")
    vanna: float = Field(description="d²V/(dS·dσ)")
    volga: float = Field(description="d²V/dσ² (vomma)")
    charm: float = Field(description="d²V/(dS·dt) - delta decay")
    iv: float = Field(description="Implied volatility used")


class GreeksUSD(BaseModel):
    """Greeks expressed in USD for portfolio management (per position)."""
    price_usd: float
    delta_usd: float
    gamma_usd: float
    vega_usd: float
    theta_usd: float
    rho_usd: float
    vanna_usd: float
    volga_usd: float
    charm_usd: float
    position_qty: int
    multiplier: float


class PricingRequest(BaseModel):
    contract: OptionContract
    market: MarketData


class PricingResponse(BaseModel):
    greeks: Greeks
    greeks_usd: Optional[GreeksUSD] = None

