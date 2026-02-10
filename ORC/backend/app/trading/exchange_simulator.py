"""
Exchange Simulator - simulates order matching and fills.
Acts as a simple matching engine for testing the OMS without a real exchange.
Generates realistic fills with simulated slippage and partial fills.
"""
from __future__ import annotations
import random
import datetime
from typing import List, Optional, Callable
from .models import (
    Order, Fill, OrderRequest, OrderSide, OrderType, OrderStatus,
    OrderBook, CancelRequest,
)


class ExchangeSimulator:
    """Simulates an exchange with basic order matching logic."""

    def __init__(self, fill_probability: float = 0.85, partial_fill_prob: float = 0.2):
        self._fill_prob = fill_probability
        self._partial_prob = partial_fill_prob
        self._simulated_books: dict = {}  # symbol -> OrderBook
        self._exchange_seq = 0

    def set_order_book(self, symbol: str, book: OrderBook) -> None:
        self._simulated_books[symbol] = book

    def generate_book(self, mid_price: float, spread: float = 0.05,
                      size: int = 100) -> OrderBook:
        half = spread / 2
        return OrderBook(
            bid_price=round(mid_price - half, 2),
            bid_size=size + random.randint(-20, 20),
            ask_price=round(mid_price + half, 2),
            ask_size=size + random.randint(-20, 20),
        )

    def _next_exchange_id(self) -> str:
        self._exchange_seq += 1
        return f"EX{self._exchange_seq:06d}"

    def submit_order(self, order: Order) -> tuple[Order, List[Fill]]:
        """Simulate order submission and potential fill."""
        order.exchange_order_id = self._next_exchange_id()
        order.status = OrderStatus.NEW
        order.updated_at = datetime.datetime.now()

        fills: List[Fill] = []
        book = self._simulated_books.get(order.symbol)

        if order.order_type == OrderType.MARKET:
            # Market orders always fill (in simulator)
            fill_price = self._get_fill_price(order, book)
            fills = self._generate_fills(order, fill_price)

        elif order.order_type == OrderType.LIMIT:
            fill_price = self._check_limit_fill(order, book)
            if fill_price is not None:
                fills = self._generate_fills(order, fill_price)

        elif order.order_type == OrderType.IOC:
            fill_price = self._check_limit_fill(order, book)
            if fill_price is not None:
                fills = self._generate_fills(order, fill_price)
            if order.remaining_quantity > 0:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.datetime.now()

        # Apply fills to order
        for fill in fills:
            order.filled_quantity += fill.quantity
            order.remaining_quantity = order.quantity - order.filled_quantity

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL

        if fills:
            total_value = sum(f.price * f.quantity for f in fills)
            order.avg_fill_price = round(total_value / sum(f.quantity for f in fills), 4)

        order.updated_at = datetime.datetime.now()
        return order, fills

    def cancel_order(self, order: Order) -> Order:
        if order.status in (OrderStatus.PENDING, OrderStatus.NEW, OrderStatus.PARTIAL):
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.datetime.now()
        return order

    def _get_fill_price(self, order: Order, book: Optional[OrderBook]) -> float:
        """Determine fill price with simulated slippage."""
        if book:
            base = book.ask_price if order.side == OrderSide.BUY else book.bid_price
        elif order.limit_price:
            base = order.limit_price
        else:
            base = 1.0  # fallback

        # Add small random slippage
        slippage = random.uniform(-0.01, 0.02) if order.side == OrderSide.BUY \
            else random.uniform(-0.02, 0.01)
        return round(base + slippage, 4)

    def _check_limit_fill(self, order: Order, book: Optional[OrderBook]) -> Optional[float]:
        if not order.limit_price:
            return None
        if not book:
            # Random fill based on probability
            if random.random() < self._fill_prob:
                return order.limit_price
            return None

        if order.side == OrderSide.BUY and order.limit_price >= book.ask_price:
            return book.ask_price
        elif order.side == OrderSide.SELL and order.limit_price <= book.bid_price:
            return book.bid_price
        return None

    def _generate_fills(self, order: Order, price: float) -> List[Fill]:
        """Generate fills, potentially splitting into partial fills."""
        remaining = order.remaining_quantity
        if remaining <= 0:
            return []

        fills = []
        if random.random() < self._partial_prob and remaining > 1:
            # Partial fill
            fill_qty = random.randint(1, remaining - 1)
        else:
            fill_qty = remaining

        fills.append(Fill(
            order_id=order.order_id,
            price=price,
            quantity=fill_qty,
            side=order.side,
        ))
        return fills

