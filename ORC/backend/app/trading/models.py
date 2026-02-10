"""Data models for trading / order management."""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List
import datetime
import uuid


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class OrderStatus(str, Enum):
    PENDING = "pending"
    NEW = "new"           # Acknowledged by exchange
    PARTIAL = "partial"   # Partially filled
    FILLED = "filled"     # Fully filled
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class InstrumentType(str, Enum):
    OPTION = "option"
    STOCK = "stock"
    FUTURE = "future"


class OrderRequest(BaseModel):
    """New order request - what the trader submits."""
    symbol: str
    instrument_type: InstrumentType = InstrumentType.OPTION
    option_type: Optional[str] = None  # "call" or "put"
    strike: Optional[float] = None
    expiry: Optional[datetime.date] = None
    side: OrderSide
    quantity: int = Field(gt=0)
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    time_in_force: str = Field(default="day")


class Order(BaseModel):
    """Full order with state tracking (like a FIX execution report)."""
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    cl_ord_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str
    instrument_type: InstrumentType
    option_type: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[datetime.date] = None
    side: OrderSide
    quantity: int
    filled_quantity: int = 0
    remaining_quantity: int = 0
    order_type: OrderType
    limit_price: Optional[float] = None
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    reject_reason: Optional[str] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    exchange_order_id: Optional[str] = None

    def model_post_init(self, __context):
        self.remaining_quantity = self.quantity - self.filled_quantity


class Fill(BaseModel):
    """A single fill / execution."""
    fill_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    order_id: str
    price: float
    quantity: int
    side: OrderSide
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)


class OrderBook(BaseModel):
    """Simulated order book level."""
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int


class CancelRequest(BaseModel):
    order_id: str
    cl_ord_id: Optional[str] = None

