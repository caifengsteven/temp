"""
Stock and ADR data models for the HSTECH estimation system.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from decimal import Decimal


class Stock(BaseModel):
    """Represents a stock in the HSTECH index."""
    
    symbol: str = Field(..., description="Hong Kong stock symbol (e.g., '0700.HK')")
    name: str = Field(..., description="Company name")
    weight: float = Field(..., ge=0, le=1, description="Weight in HSTECH index")
    sector: str = Field(..., description="Sector classification")
    market_cap: Optional[float] = Field(None, description="Market capitalization in HKD")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v.endswith('.HK'):
            raise ValueError('Hong Kong symbols must end with .HK')
        return v
    
    @validator('weight')
    def validate_weight(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Weight must be between 0 and 1')
        return v


class ADRMapping(BaseModel):
    """Represents the mapping between a Hong Kong stock and its US ADR."""
    
    hk_symbol: str = Field(..., description="Hong Kong stock symbol")
    us_symbol: str = Field(..., description="US ADR symbol")
    conversion_ratio: float = Field(..., gt=0, description="Number of HK shares per ADR")
    currency_base: str = Field(default="HKD", description="Base currency")
    currency_quote: str = Field(default="USD", description="Quote currency")
    
    @validator('hk_symbol')
    def validate_hk_symbol(cls, v):
        if not v.endswith('.HK'):
            raise ValueError('Hong Kong symbols must end with .HK')
        return v


class PriceData(BaseModel):
    """Represents price data for a stock at a specific time."""
    
    symbol: str = Field(..., description="Stock symbol")
    price: Decimal = Field(..., gt=0, description="Stock price")
    currency: str = Field(..., description="Price currency")
    timestamp: datetime = Field(..., description="Price timestamp")
    volume: Optional[int] = Field(None, ge=0, description="Trading volume")
    source: str = Field(..., description="Data source")
    
    class Config:
        json_encoders = {
            Decimal: float,
            datetime: lambda v: v.isoformat()
        }


class CurrencyRate(BaseModel):
    """Represents a currency exchange rate."""
    
    base_currency: str = Field(..., description="Base currency code")
    quote_currency: str = Field(..., description="Quote currency code")
    rate: Decimal = Field(..., gt=0, description="Exchange rate")
    timestamp: datetime = Field(..., description="Rate timestamp")
    source: str = Field(..., description="Data source")
    
    class Config:
        json_encoders = {
            Decimal: float,
            datetime: lambda v: v.isoformat()
        }


class IndexData(BaseModel):
    """Represents HSTECH index data."""
    
    value: Decimal = Field(..., gt=0, description="Index value")
    timestamp: datetime = Field(..., description="Index timestamp")
    change: Optional[Decimal] = Field(None, description="Change from previous close")
    change_percent: Optional[Decimal] = Field(None, description="Percentage change")
    volume: Optional[int] = Field(None, ge=0, description="Total volume")
    
    class Config:
        json_encoders = {
            Decimal: float,
            datetime: lambda v: v.isoformat()
        }


class EstimationResult(BaseModel):
    """Represents the result of an HSTECH index estimation."""
    
    estimated_value: Decimal = Field(..., gt=0, description="Estimated index value")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    timestamp: datetime = Field(..., description="Estimation timestamp")
    method_weights: Dict[str, float] = Field(..., description="Weights used for different methods")
    component_contributions: Dict[str, float] = Field(..., description="Individual component contributions")
    
    class Config:
        json_encoders = {
            Decimal: float,
            datetime: lambda v: v.isoformat()
        }
