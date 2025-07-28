#!/usr/bin/env python3
"""
Simple test of US futures database with DC analysis
"""

import sqlite3
import os
import numpy as np

class DCGenerator:
    def __init__(self, threshold):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.highest_price = 0.0
        self.lowest_price = 0.0
        self.is_upturn = True
        self.initialized = False
    
    def process_price(self, price):
        if price <= 0:
            return None
            
        if not self.initialized:
            self.highest_price = price
            self.lowest_price = price
            self.is_upturn = True
            self.initialized = True
            return None
        
        event = self._check_for_events(price)
        self._update_state(price, event)
        return event
    
    def _check_for_events(self, current_price):
        if self.is_upturn:
            if current_price <= self.highest_price * (1.0 - self.threshold):
                return "END_UPTURN"
        else:
            if current_price >= self.lowest_price * (1.0 + self.threshold):
                return "END_DOWNTURN"
        return None
    
    def _update_state(self, current_price, event):
        if event == "END_UPTURN":
            self.is_upturn = False
            self.lowest_price = current_price
        elif event == "END_DOWNTURN":
            self.is_upturn = True
            self.highest_price = current_price
        else:
            if self.is_upturn and current_price > self.highest_price:
                self.highest_price = current_price
            elif not self.is_upturn and current_price < self.lowest_price:
                self.lowest_price = current_price

class SimpleDCStrategy:
    def __init__(self, capital):
        self.initial_capital = capital
        self.current_cash = capital
        self.shares_held = 0
        self.trade_count = 0
    
    def on_dc_event(self, event, price):
        if event == "END_DOWNTURN" and self.current_cash > 0 and price > 0:
            self.shares_held = self.current_cash / price
            self.current_cash = 0
            self.trade_count += 1
        elif event == "END_UPTURN" and self.shares_held > 0 and price > 0:
            self.current_cash = self.shares_held * price
            self.shares_held = 0
            self.trade_count += 1
    
    def get_current_value(self, current_price):
        return self.current_cash + (self.shares_held * current_price)
    
    def get_total_return(self, current_price):
        return (self.get_current_value(current_price) - self.initial_capital) / self.initial_capital * 100.0

def test_futures_symbol(symbol):
    print(f"\n=== Testing Futures Symbol: {symbol} ===")
    
    db_path = "F:\\database\\us futures 1mins\\us_fut_1min.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get data for this symbol
        cursor.execute("SELECT close FROM futures_data WHERE symbol = ? ORDER BY datetime LIMIT 50000", (symbol,))
        rows = cursor.fetchall()
        
        if not rows:
            print(f"No data found for symbol: {symbol}")
            return
        
        prices = [row[0] for row in rows if row[0] > 0]
        
        if len(prices) < 100:
            print(f"Insufficient data: {len(prices)} points")
            return
        
        print(f"Loaded {len(prices)} price points")
        print(f"Price range: ${min(prices):.2f} to ${max(prices):.2f}")
        print(f"Price range %: {((max(prices) - min(prices)) / min(prices) * 100):.1f}%")
        
        # Test DC strategies
        thresholds = [0.005, 0.01, 0.02, 0.03, 0.05]  # 0.5% to 5%
        
        print(f"\n=== Simple DC Strategy Results ===")
        print(f"{'Threshold':<12} {'Trades':<8} {'Return %':<10} {'DC Events':<10}")
        print("-" * 45)
        
        for threshold in thresholds:
            dc_gen = DCGenerator(threshold)
            strategy = SimpleDCStrategy(100000.0)
            
            dc_events = 0
            for price in prices:
                event = dc_gen.process_price(price)
                if event:
                    strategy.on_dc_event(event, price)
                    dc_events += 1
            
            final_return = strategy.get_total_return(prices[-1])
            
            print(f"{threshold*100:8.1f}%     {strategy.trade_count:<8} {final_return:<9.2f}% {dc_events:<10}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("=== US Futures DC Testing (Python) ===")
    
    db_path = "F:\\database\\us futures 1mins\\us_fut_1min.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get available symbols
        cursor.execute("SELECT DISTINCT symbol FROM futures_data ORDER BY symbol LIMIT 20")
        symbols = cursor.fetchall()
        
        print(f"Available futures symbols ({len(symbols)}):")
        for symbol in symbols:
            print(f"  {symbol[0]}")
        
        conn.close()
        
        # Test first few symbols
        test_symbols = [symbol[0] for symbol in symbols[:5]]
        
        for symbol in test_symbols:
            test_futures_symbol(symbol)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
