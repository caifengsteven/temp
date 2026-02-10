"""
Portfolio Manager - manages positions and computes aggregated Greeks in USD.
Similar to ORC Trader's portfolio view with full Greek decomposition.
"""
from __future__ import annotations
import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from .models import Position, PositionGreeks, PortfolioSummary, PortfolioResponse
from ..quant.models import OptionContract, MarketData, OptionType, ExerciseStyle
from ..quant.pricing_engine import price_european, price_american_binomial


class PortfolioManager:
    """In-memory portfolio manager with real-time Greek computation."""

    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._market_data: Dict[str, MarketData] = {}  # keyed by symbol

    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        if position.position_id in self._positions:
            existing = self._positions[position.position_id]
            # Average in the new fill
            total_qty = existing.quantity + position.quantity
            if total_qty != 0:
                avg = (existing.avg_price * existing.quantity +
                       position.avg_price * position.quantity) / total_qty
                existing.quantity = total_qty
                existing.avg_price = avg
            else:
                del self._positions[position.position_id]
                return
        else:
            self._positions[position.position_id] = position.model_copy()

    def remove_position(self, position_id: str) -> bool:
        if position_id in self._positions:
            del self._positions[position_id]
            return True
        return False

    def update_market_data(self, symbol: str, market: MarketData) -> None:
        self._market_data[symbol] = market

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def _compute_position_greeks(self, pos: Position) -> PositionGreeks:
        """Compute Greeks for a single position."""
        market = self._market_data.get(pos.symbol)
        if not market:
            return PositionGreeks(position=pos)

        contract = OptionContract(
            symbol=pos.symbol,
            option_type=OptionType(pos.option_type),
            strike=pos.strike,
            expiry=pos.expiry,
            exercise_style=ExerciseStyle(pos.exercise_style),
            multiplier=pos.multiplier,
        )

        if contract.exercise_style == ExerciseStyle.EUROPEAN:
            greeks = price_european(contract, market)
        else:
            greeks = price_american_binomial(contract, market)

        qty = pos.quantity
        mult = pos.multiplier
        market_value = greeks.price * qty * mult
        cost_basis = pos.avg_price * qty * mult

        return PositionGreeks(
            position=pos,
            theo_price=greeks.price,
            iv=greeks.iv,
            delta=greeks.delta,
            gamma=greeks.gamma,
            vega=greeks.vega,
            theta=greeks.theta,
            rho=greeks.rho,
            vanna=greeks.vanna,
            volga=greeks.volga,
            charm=greeks.charm,
            # USD Greeks: qty * multiplier * per-unit-greek * spot (for delta/gamma)
            delta_usd=round(greeks.delta * qty * mult * market.spot, 2),
            gamma_usd=round(greeks.gamma * qty * mult * market.spot * market.spot * 0.01, 2),
            vega_usd=round(greeks.vega * qty * mult, 2),
            theta_usd=round(greeks.theta * qty * mult, 2),
            rho_usd=round(greeks.rho * qty * mult, 2),
            vanna_usd=round(greeks.vanna * qty * mult, 2),
            volga_usd=round(greeks.volga * qty * mult, 2),
            charm_usd=round(greeks.charm * qty * mult, 2),
            market_value=round(market_value, 2),
            unrealized_pnl=round(market_value - cost_basis, 2),
        )

    def compute_portfolio(self) -> PortfolioResponse:
        """Compute full portfolio with aggregated Greeks."""
        position_greeks = [self._compute_position_greeks(p) for p in self._positions.values()]

        summary = PortfolioSummary(
            total_delta_usd=round(sum(p.delta_usd for p in position_greeks), 2),
            total_gamma_usd=round(sum(p.gamma_usd for p in position_greeks), 2),
            total_vega_usd=round(sum(p.vega_usd for p in position_greeks), 2),
            total_theta_usd=round(sum(p.theta_usd for p in position_greeks), 2),
            total_rho_usd=round(sum(p.rho_usd for p in position_greeks), 2),
            total_vanna_usd=round(sum(p.vanna_usd for p in position_greeks), 2),
            total_volga_usd=round(sum(p.volga_usd for p in position_greeks), 2),
            total_charm_usd=round(sum(p.charm_usd for p in position_greeks), 2),
            total_market_value=round(sum(p.market_value for p in position_greeks), 2),
            total_unrealized_pnl=round(sum(p.unrealized_pnl for p in position_greeks), 2),
            position_count=len(position_greeks),
        )

        # Group by underlying
        by_underlying = defaultdict(lambda: defaultdict(float))
        for pg in position_greeks:
            sym = pg.position.symbol
            by_underlying[sym]["delta_usd"] += pg.delta_usd
            by_underlying[sym]["gamma_usd"] += pg.gamma_usd
            by_underlying[sym]["vega_usd"] += pg.vega_usd
            by_underlying[sym]["theta_usd"] += pg.theta_usd

        # Group by expiry
        by_expiry = defaultdict(lambda: defaultdict(float))
        for pg in position_greeks:
            exp = pg.position.expiry.isoformat()
            by_expiry[exp]["delta_usd"] += pg.delta_usd
            by_expiry[exp]["gamma_usd"] += pg.gamma_usd
            by_expiry[exp]["vega_usd"] += pg.vega_usd
            by_expiry[exp]["theta_usd"] += pg.theta_usd

        return PortfolioResponse(
            positions=position_greeks,
            summary=summary,
            by_underlying=dict(by_underlying),
            by_expiry=dict(by_expiry),
        )

