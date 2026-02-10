"""
ORC Trading Platform - FastAPI Application
Main entry point with REST and WebSocket endpoints.
"""
from __future__ import annotations
import json
import asyncio
from typing import List, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .quant.models import PricingRequest, PricingResponse, OptionType, ExerciseStyle
from .quant.pricing_engine import price_european, price_american_binomial
from .quant.implied_vol import implied_vol
from .portfolio.models import Position, PortfolioResponse
from .portfolio.manager import PortfolioManager
from .volsurface.models import CalibrationRequest, VolSurfaceData
from .volsurface.surface_builder import build_surface
from .trading.models import OrderRequest, Order, CancelRequest, Fill
from .trading.oms import OrderManagementSystem
from .trading.exchange_simulator import ExchangeSimulator
from .sample_data import load_sample_data

# Global instances
portfolio_mgr = PortfolioManager()
exchange_sim = ExchangeSimulator()
oms = OrderManagementSystem(portfolio_mgr, exchange_sim)
ws_clients: Set[WebSocket] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_sample_data(portfolio_mgr, exchange_sim)
    yield


app = FastAPI(
    title="ORC Trading Platform",
    description="Options trading platform with quant library, portfolio management, "
                "volatility surface fitting, and order management.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket for real-time updates ──
async def broadcast(event_type: str, data: dict):
    msg = json.dumps({"type": event_type, "data": data}, default=str)
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    ws_clients -= dead


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            data = await ws.receive_text()
            # Client can request portfolio refresh
            if data == "refresh_portfolio":
                portfolio = portfolio_mgr.compute_portfolio()
                await ws.send_text(json.dumps({
                    "type": "portfolio_update",
                    "data": portfolio.model_dump()
                }, default=str))
    except WebSocketDisconnect:
        ws_clients.discard(ws)


# ── Pricing endpoints ──
@app.post("/api/pricing/price", response_model=PricingResponse)
def price_option(req: PricingRequest):
    if req.contract.exercise_style == ExerciseStyle.EUROPEAN:
        greeks = price_european(req.contract, req.market)
    else:
        greeks = price_american_binomial(req.contract, req.market)
    return PricingResponse(greeks=greeks)


@app.post("/api/pricing/implied-vol")
def compute_implied_vol(
    market_price: float, spot: float, strike: float,
    rate: float, dividend_yield: float, T: float, option_type: str
):
    iv = implied_vol(market_price, spot, strike, rate, dividend_yield, T, OptionType(option_type))
    return {"implied_volatility": round(iv, 6)}


# ── Portfolio endpoints ──
@app.get("/api/portfolio", response_model=PortfolioResponse)
def get_portfolio():
    return portfolio_mgr.compute_portfolio()


@app.post("/api/portfolio/position")
def add_position(position: Position):
    portfolio_mgr.add_position(position)
    return {"status": "ok", "position_id": position.position_id}


@app.delete("/api/portfolio/position/{position_id}")
def remove_position(position_id: str):
    if portfolio_mgr.remove_position(position_id):
        return {"status": "removed"}
    raise HTTPException(404, "Position not found")


@app.post("/api/portfolio/market-data")
def update_market_data(symbol: str, spot: float, rate: float = 0.05,
                       dividend_yield: float = 0.0, volatility: float = 0.25):
    from .quant.models import MarketData
    md = MarketData(spot=spot, rate=rate, dividend_yield=dividend_yield, volatility=volatility)
    portfolio_mgr.update_market_data(symbol, md)
    return {"status": "ok"}


# ── Vol Surface endpoints ──
@app.post("/api/volsurface/calibrate", response_model=VolSurfaceData)
def calibrate_vol_surface(req: CalibrationRequest):
    return build_surface(req)


# ── Trading / Order endpoints ──
@app.post("/api/orders/submit", response_model=Order)
def submit_order(req: OrderRequest):
    order = oms.submit_order(req)
    return order


@app.post("/api/orders/cancel")
def cancel_order(req: CancelRequest):
    order = oms.cancel_order(req)
    if order:
        return order
    raise HTTPException(404, "Order not found")


@app.get("/api/orders", response_model=List[Order])
def get_orders():
    return oms.get_all_orders()


@app.get("/api/orders/active", response_model=List[Order])
def get_active_orders():
    return oms.get_active_orders()


@app.get("/api/orders/{order_id}", response_model=Order)
def get_order(order_id: str):
    order = oms.get_order(order_id)
    if order:
        return order
    raise HTTPException(404, "Order not found")


@app.get("/api/orders/{order_id}/fills", response_model=List[Fill])
def get_order_fills(order_id: str):
    return oms.get_fills(order_id)


@app.get("/api/fills", response_model=List[Fill])
def get_all_fills():
    return oms.get_fills()

