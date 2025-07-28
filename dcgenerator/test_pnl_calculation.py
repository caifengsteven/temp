#!/usr/bin/env python3
"""
Test P&L calculation to verify the logic works correctly.
"""

import math

class SimpleTradingStrategy:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.trade_count = 0
    
    def buy_signal(self, price):
        if self.position == 0 and self.cash > 1.0 and price > 0:
            cash_to_use = self.cash * 0.95
            self.position = cash_to_use / price
            self.cash -= cash_to_use
            self.trade_count += 1
            print(f"BUY: Price={price:.2f}, Cash_used={cash_to_use:.2f}, Position={self.position:.4f}, Remaining_cash={self.cash:.2f}")
    
    def sell_signal(self, price):
        if self.position > 0 and price > 0:
            sale_value = self.position * price
            self.cash += sale_value
            self.position = 0
            self.trade_count += 1
            print(f"SELL: Price={price:.2f}, Sale_value={sale_value:.2f}, Total_cash={self.cash:.2f}")
    
    def get_current_value(self, current_price):
        if not math.isfinite(current_price) or current_price <= 0:
            return self.cash
        return self.cash + self.position * current_price
    
    def get_pnl(self, current_price):
        return self.get_current_value(current_price) - self.initial_capital
    
    def get_total_return(self, current_price):
        if self.initial_capital <= 0:
            return 0.0
        return (self.get_current_value(current_price) - self.initial_capital) / self.initial_capital * 100.0
    
    def print_status(self, current_price):
        print(f"Status: Cash={self.cash:.2f}, Position={self.position:.4f}, Current_price={current_price:.2f}")
        print(f"        Total_value={self.get_current_value(current_price):.2f}, PnL={self.get_pnl(current_price):.2f}, Return={self.get_total_return(current_price):.2f}%")

def test_trading_strategy():
    print("=== Testing Trading Strategy P&L Calculation ===")
    
    # Test with simple price data
    test_prices = [100.0, 101.0, 102.0, 101.5, 100.5, 99.0, 100.5, 102.0]
    
    strategy = SimpleTradingStrategy(100000.0)
    
    print("Initial status:")
    strategy.print_status(test_prices[0])
    
    print("\nSimulating trades:")
    
    # Buy at price 100
    strategy.buy_signal(test_prices[0])
    strategy.print_status(test_prices[0])
    
    # Check value at different prices
    for i, price in enumerate(test_prices[1:], 1):
        print(f"\nPrice update to {price}:")
        strategy.print_status(price)
        
        # Sell at price 102
        if price >= 102.0:
            strategy.sell_signal(price)
            strategy.print_status(price)
            break
    
    print("\n=== Final Results ===")
    final_price = test_prices[-1]
    print(f"Final price: {final_price}")
    print(f"Final PnL: ${strategy.get_pnl(final_price):.2f}")
    print(f"Final Return: {strategy.get_total_return(final_price):.2f}%")
    print(f"Total trades: {strategy.trade_count}")

def test_edge_cases():
    print("\n=== Testing Edge Cases ===")
    
    strategy = SimpleTradingStrategy(100000.0)
    
    # Test with zero price
    print("Testing with zero price:")
    strategy.print_status(0.0)
    
    # Test with very large price
    print("Testing with large price:")
    strategy.print_status(1000000.0)
    
    # Test with NaN price
    print("Testing with NaN price:")
    strategy.print_status(float('nan'))
    
    # Test normal trading scenario
    print("\nTesting normal scenario:")
    strategy.buy_signal(100.0)
    strategy.print_status(100.0)
    strategy.print_status(105.0)  # 5% gain
    strategy.sell_signal(105.0)
    strategy.print_status(105.0)

if __name__ == "__main__":
    test_trading_strategy()
    test_edge_cases()
