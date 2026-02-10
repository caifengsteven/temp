"""
Order Management System (OMS) - central order routing and management.
Handles order lifecycle: validation -> submission -> fill tracking -> position update.
Integrates with ExchangeSimulator and PortfolioManager.
"""
from __future__ import annotations
import datetime
from typing import List, Dict, Optional, Callable
from .models import (
    Order, Fill, OrderRequest, OrderSide, OrderType, OrderStatus,
    CancelRequest, InstrumentType,
)
from .exchange_simulator import ExchangeSimulator
from ..portfolio.models import Position
from ..portfolio.manager import PortfolioManager


class OrderManagementSystem:
    """Central OMS managing order lifecycle and routing to exchange."""

    def __init__(self, portfolio_manager: PortfolioManager,
                 exchange: Optional[ExchangeSimulator] = None):
        self._portfolio = portfolio_manager
        self._exchange = exchange or ExchangeSimulator()
        self._orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []
        self._event_handlers: List[Callable] = []

    def on_event(self, handler: Callable) -> None:
        """Register an event handler for order updates."""
        self._event_handlers.append(handler)

    def _emit(self, event_type: str, data: dict) -> None:
        for handler in self._event_handlers:
            try:
                handler(event_type, data)
            except Exception:
                pass

    def submit_order(self, request: OrderRequest) -> Order:
        """Validate and submit a new order."""
        # Validation
        if request.order_type == OrderType.LIMIT and request.limit_price is None:
            order = Order(
                symbol=request.symbol,
                instrument_type=request.instrument_type,
                option_type=request.option_type,
                strike=request.strike,
                expiry=request.expiry,
                side=request.side,
                quantity=request.quantity,
                order_type=request.order_type,
                limit_price=request.limit_price,
                status=OrderStatus.REJECTED,
                reject_reason="Limit price required for limit orders",
            )
            self._orders[order.order_id] = order
            self._emit("order_rejected", {"order": order.model_dump()})
            return order

        # Create order
        order = Order(
            symbol=request.symbol,
            instrument_type=request.instrument_type,
            option_type=request.option_type,
            strike=request.strike,
            expiry=request.expiry,
            side=request.side,
            quantity=request.quantity,
            order_type=request.order_type,
            limit_price=request.limit_price,
        )
        self._orders[order.order_id] = order
        self._emit("order_new", {"order": order.model_dump()})

        # Route to exchange
        order, fills = self._exchange.submit_order(order)
        self._orders[order.order_id] = order

        # Process fills
        for fill in fills:
            self._fills.append(fill)
            self._process_fill(order, fill)
            self._emit("fill", {"order": order.model_dump(), "fill": fill.model_dump()})

        self._emit("order_update", {"order": order.model_dump()})
        return order

    def cancel_order(self, request: CancelRequest) -> Optional[Order]:
        order = self._orders.get(request.order_id)
        if not order:
            return None
        order = self._exchange.cancel_order(order)
        self._orders[order.order_id] = order
        self._emit("order_cancelled", {"order": order.model_dump()})
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_all_orders(self) -> List[Order]:
        return list(self._orders.values())

    def get_active_orders(self) -> List[Order]:
        active = {OrderStatus.PENDING, OrderStatus.NEW, OrderStatus.PARTIAL}
        return [o for o in self._orders.values() if o.status in active]

    def get_fills(self, order_id: Optional[str] = None) -> List[Fill]:
        if order_id:
            return [f for f in self._fills if f.order_id == order_id]
        return self._fills

    def _process_fill(self, order: Order, fill: Fill) -> None:
        """Convert a fill into a portfolio position update."""
        qty = fill.quantity if order.side == OrderSide.BUY else -fill.quantity
        pos_id = f"{order.symbol}_{order.option_type}_{order.strike}_{order.expiry}"

        position = Position(
            position_id=pos_id,
            symbol=order.symbol,
            option_type=order.option_type or "call",
            strike=order.strike or 0,
            expiry=order.expiry or datetime.date.today(),
            quantity=qty,
            avg_price=fill.price,
        )
        self._portfolio.add_position(position)

