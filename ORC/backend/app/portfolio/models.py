"""Data models for portfolio management."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import datetime


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class Position(BaseModel):
    """A single option position in the portfolio."""
    position_id: str
    symbol: str
    option_type: str  # "call" or "put"
    strike: float
    expiry: datetime.date
    quantity: int = Field(description="Signed qty: +ve = long, -ve = short")
    avg_price: float = Field(description="Average fill price per contract")
    multiplier: float = Field(default=100.0)
    exercise_style: str = Field(default="european")


class PositionGreeks(BaseModel):
    """Position with computed Greeks in USD."""
    position: Position
    # Per-unit Greeks (from pricing engine)
    theo_price: float = 0.0
    iv: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0
    # USD Greeks (quantity * multiplier * greek)
    delta_usd: float = 0.0
    gamma_usd: float = 0.0
    vega_usd: float = 0.0
    theta_usd: float = 0.0
    rho_usd: float = 0.0
    vanna_usd: float = 0.0
    volga_usd: float = 0.0
    charm_usd: float = 0.0
    # P&L
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


class PortfolioSummary(BaseModel):
    """Aggregated portfolio Greeks."""
    total_delta_usd: float = 0.0
    total_gamma_usd: float = 0.0
    total_vega_usd: float = 0.0
    total_theta_usd: float = 0.0
    total_rho_usd: float = 0.0
    total_vanna_usd: float = 0.0
    total_volga_usd: float = 0.0
    total_charm_usd: float = 0.0
    total_market_value: float = 0.0
    total_unrealized_pnl: float = 0.0
    position_count: int = 0


class PortfolioResponse(BaseModel):
    positions: List[PositionGreeks]
    summary: PortfolioSummary
    by_underlying: dict = Field(default_factory=dict)
    by_expiry: dict = Field(default_factory=dict)

