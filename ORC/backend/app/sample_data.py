"""
Sample data loader - populates the platform with realistic option positions
and market data for demonstration.
"""
from __future__ import annotations
import datetime
from .portfolio.models import Position
from .portfolio.manager import PortfolioManager
from .quant.models import MarketData
from .trading.exchange_simulator import ExchangeSimulator


def load_sample_data(portfolio: PortfolioManager, exchange: ExchangeSimulator):
    """Load sample positions and market data for demo."""
    today = datetime.date.today()

    # Market data for underlyings
    market_data = {
        "AAPL": MarketData(spot=185.0, rate=0.053, dividend_yield=0.005, volatility=0.28,
                           valuation_date=today),
        "SPY": MarketData(spot=502.0, rate=0.053, dividend_yield=0.013, volatility=0.16,
                          valuation_date=today),
        "TSLA": MarketData(spot=245.0, rate=0.053, dividend_yield=0.0, volatility=0.55,
                           valuation_date=today),
        "NVDA": MarketData(spot=720.0, rate=0.053, dividend_yield=0.0004, volatility=0.45,
                           valuation_date=today),
    }

    for symbol, md in market_data.items():
        portfolio.update_market_data(symbol, md)

    # Sample positions - a realistic options portfolio
    exp_near = today + datetime.timedelta(days=30)
    exp_mid = today + datetime.timedelta(days=60)
    exp_far = today + datetime.timedelta(days=120)

    positions = [
        # AAPL bull call spread
        Position(position_id="AAPL_C185_near", symbol="AAPL", option_type="call",
                 strike=185.0, expiry=exp_near, quantity=10, avg_price=5.20),
        Position(position_id="AAPL_C195_near", symbol="AAPL", option_type="call",
                 strike=195.0, expiry=exp_near, quantity=-10, avg_price=1.80),
        # AAPL protective put
        Position(position_id="AAPL_P175_mid", symbol="AAPL", option_type="put",
                 strike=175.0, expiry=exp_mid, quantity=5, avg_price=3.10),

        # SPY iron condor
        Position(position_id="SPY_P480_mid", symbol="SPY", option_type="put",
                 strike=480.0, expiry=exp_mid, quantity=-20, avg_price=4.50),
        Position(position_id="SPY_P470_mid", symbol="SPY", option_type="put",
                 strike=470.0, expiry=exp_mid, quantity=20, avg_price=2.80),
        Position(position_id="SPY_C520_mid", symbol="SPY", option_type="call",
                 strike=520.0, expiry=exp_mid, quantity=-20, avg_price=5.10),
        Position(position_id="SPY_C530_mid", symbol="SPY", option_type="call",
                 strike=530.0, expiry=exp_mid, quantity=20, avg_price=2.90),

        # TSLA straddle
        Position(position_id="TSLA_C245_far", symbol="TSLA", option_type="call",
                 strike=245.0, expiry=exp_far, quantity=5, avg_price=22.50),
        Position(position_id="TSLA_P245_far", symbol="TSLA", option_type="put",
                 strike=245.0, expiry=exp_far, quantity=5, avg_price=21.80),

        # NVDA directional calls
        Position(position_id="NVDA_C750_far", symbol="NVDA", option_type="call",
                 strike=750.0, expiry=exp_far, quantity=3, avg_price=45.00),
        Position(position_id="NVDA_C800_far", symbol="NVDA", option_type="call",
                 strike=800.0, expiry=exp_far, quantity=-3, avg_price=28.00),
    ]

    for pos in positions:
        portfolio.add_position(pos)

    # Set up exchange simulator order books
    from .trading.models import OrderBook
    books = {
        "AAPL": exchange.generate_book(5.20, spread=0.10),
        "SPY": exchange.generate_book(4.50, spread=0.05),
        "TSLA": exchange.generate_book(22.50, spread=0.50),
        "NVDA": exchange.generate_book(45.00, spread=1.00),
    }
    for sym, book in books.items():
        exchange.set_order_book(sym, book)

    print(f"[ORC] Loaded {len(positions)} sample positions across "
          f"{len(market_data)} underlyings")

