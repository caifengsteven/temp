"""
Configuration models for the HSTECH estimation system.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
import yaml
from pathlib import Path


class APIKeys(BaseModel):
    """API keys configuration."""

    bloomberg_api_key: Optional[str] = None
    bloomberg_cloud_url: Optional[str] = None
    alpha_vantage: Optional[str] = None  # Kept for fallback
    yahoo_finance: Optional[str] = None  # Kept for fallback


class DataSources(BaseModel):
    """Data sources configuration."""

    primary_data_source: str = Field(default="bloomberg")
    fallback_data_source: str = Field(default="alpha_vantage")
    price_update_frequency: int = Field(default=60, ge=1)
    currency_update_frequency: int = Field(default=300, ge=1)


class MarketHours(BaseModel):
    """Market hours configuration."""
    
    open: str = Field(..., description="Market open time in HH:MM format")
    close: str = Field(..., description="Market close time in HH:MM format")
    
    @validator('open', 'close')
    def validate_time_format(cls, v):
        try:
            hours, minutes = map(int, v.split(':'))
            if not (0 <= hours <= 23 and 0 <= minutes <= 59):
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError('Time must be in HH:MM format')
        return v


class EstimationWeights(BaseModel):
    """Weights for different estimation methods."""
    
    adr_based: float = Field(default=0.6, ge=0, le=1)
    covariance_based: float = Field(default=0.25, ge=0, le=1)
    market_indicators: float = Field(default=0.15, ge=0, le=1)
    
    @validator('market_indicators')
    def validate_weights_sum(cls, v, values):
        total = v + values.get('adr_based', 0) + values.get('covariance_based', 0)
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError('Estimation weights must sum to 1.0')
        return v


class EstimationConfig(BaseModel):
    """Estimation algorithm configuration."""
    
    lookback_days: int = Field(default=252, ge=1)
    min_correlation_threshold: float = Field(default=0.3, ge=0, le=1)
    weights: EstimationWeights = Field(default_factory=EstimationWeights)
    indicators: List[str] = Field(default=["PDD", "KWEB"])


class CurrencyConfig(BaseModel):
    """Currency configuration."""
    
    base: str = Field(default="HKD")
    quote: str = Field(default="USD")
    conversion_cache_minutes: int = Field(default=5, ge=1)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO")
    file: str = Field(default="logs/hstech_estimation.log")
    max_file_size: str = Field(default="10MB")
    backup_count: int = Field(default=5, ge=1)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    type: str = Field(default="sqlite")
    path: str = Field(default="data/hstech_estimation.db")


class BacktestingConfig(BaseModel):
    """Backtesting configuration."""
    
    start_date: str = Field(default="2023-01-01")
    end_date: str = Field(default="2024-08-18")
    metrics: List[str] = Field(default=["mse", "mae", "correlation", "directional_accuracy"])


class Config(BaseModel):
    """Main configuration class."""
    
    api_keys: APIKeys = Field(default_factory=APIKeys)
    data_sources: DataSources = Field(default_factory=DataSources)
    market_hours: Dict[str, MarketHours] = Field(default_factory=dict)
    estimation: EstimationConfig = Field(default_factory=EstimationConfig)
    currency: CurrencyConfig = Field(default_factory=CurrencyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    
    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        """Load configuration from YAML file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
