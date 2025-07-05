import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Set
import random
from collections import defaultdict
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class State:
    """Represents the state of the AMM at a given point in time."""
    
    def __init__(self, x: float, y: float, x0: float, y0: float, c: float):
        """
        Initialize a state for the AMM.
        
        Args:
            x: Amount of asset X (cash)
            y: Amount of asset Y (expiring token)
            x0: Reference amount of X
            y0: Reference amount of Y
            c: Curve parameter
        """
        self.x = x
        self.y = y
        self.x0 = x0
        self.y0 = y0
        self.c = c
        
    def spot_price(self) -> float:
        """Calculate the current spot price of Y in terms of X."""
        return self.x0 * self.c * self.y0 / (self.y * (self.x0 / self.x)**c)
    
    def evaluate_curve(self, x_val: float) -> float:
        """
        Evaluate the state curve at a given x value.
        
        Args:
            x_val: The x value to evaluate
            
        Returns:
            The corresponding y value on the curve
        """
        return self.y0 * (x_val / self.x0)**(-self.c)
    
    def get_y_for_x_delta(self, delta_x: float) -> float:
        """
        Calculate how much Y would be returned for a given amount of X.
        
        Args:
            delta_x: Amount of X to add
            
        Returns:
            Amount of Y to remove
        """
        new_x = self.x + delta_x
        new_y = self.evaluate_curve(new_x)
        return self.y - new_y
    
    def get_x_for_y_delta(self, delta_y: float) -> float:
        """
        Calculate how much X would be returned for a given amount of Y.
        
        Args:
            delta_y: Amount of Y to add
            
        Returns:
            Amount of X to remove
        """
        new_y = self.y + delta_y
        # Solve for new_x: new_y = y0 * (new_x / x0)^(-c)
        # => (new_x / x0)^(-c) = new_y / y0
        # => (new_x / x0) = (new_y / y0)^(-1/c)
        # => new_x = x0 * (new_y / y0)^(-1/c)
        new_x = self.x0 * (new_y / self.y0)**(-1/self.c)
        return self.x - new_x
    
    def __str__(self) -> str:
        return f"State(x={self.x:.4f}, y={self.y:.4f}, price={self.spot_price():.4f})"


class LiquidityProvider:
    """Represents a liquidity provider in the AMM."""
    
    def __init__(self, id: str, initial_x: float, initial_y: float, 
                 price_threshold: float, c_value: float):
        """
        Initialize a liquidity provider.
        
        Args:
            id: Unique identifier
            initial_x: Initial amount of X provided
            initial_y: Initial amount of Y provided
            price_threshold: Minimum price at which LP will provide liquidity
            c_value: LP's desired curve parameter
        """
        self.id = id
        self.x = initial_x
        self.y = initial_y
        self.x_frozen = 0.0
        self.y_frozen = 0.0
        self.price_threshold = price_threshold
        self.c_value = c_value
        self.share = 0.0
        self.active = True
        self.risk_participation = 0.0
    
    def total_x(self) -> float:
        """Return the total amount of X owned by this LP."""
        return self.x + self.x_frozen
    
    def total_y(self) -> float:
        """Return the total amount of Y owned by this LP."""
        return self.y + self.y_frozen
    
    def freeze_liquidity(self, x_to_freeze: float, y_to_freeze: float):
        """
        Freeze a portion of liquidity.
        
        Args:
            x_to_freeze: Amount of X to freeze
            y_to_freeze: Amount of Y to freeze
        """
        assert x_to_freeze <= self.x and y_to_freeze <= self.y
        self.x -= x_to_freeze
        self.y -= y_to_freeze
        self.x_frozen += x_to_freeze
        self.y_frozen += y_to_freeze
        
    def unfreeze_liquidity(self, x_to_unfreeze: float, y_to_unfreeze: float):
        """
        Unfreeze a portion of liquidity.
        
        Args:
            x_to_unfreeze: Amount of X to unfreeze
            y_to_unfreeze: Amount of Y to unfreeze
        """
        assert x_to_unfreeze <= self.x_frozen and y_to_unfreeze <= self.y_frozen
        self.x += x_to_unfreeze
        self.y += y_to_unfreeze
        self.x_frozen -= x_to_unfreeze
        self.y_frozen -= y_to_unfreeze
    
    def __str__(self) -> str:
        status = "Active" if self.active else "Inactive"
        return (f"LP {self.id}: {status}, Share: {self.share:.4f}, "
                f"X: {self.x:.4f}, Y: {self.y:.4f}, "
                f"X frozen: {self.x_frozen:.4f}, Y frozen: {self.y_frozen:.4f}, "
                f"Min price: {self.price_threshold:.4f}")


class Order:
    """Represents an order in the order book."""
    
    def __init__(self, id: str, address: str, price: float, is_buy: bool):
        """
        Initialize an order.
        
        Args:
            id: Unique order identifier
            address: Address of the order submitter
            price: Order price (buy or sell)
            is_buy: True if buy order, False if sell order
        """
        self.id = id
        self.address = address
        self.price = price
        self.is_buy = is_buy
    
    def __str__(self) -> str:
        order_type = "Buy" if self.is_buy else "Sell"
        return f"{order_type} Order {self.id}: {self.address} @ {self.price:.4f}"


class ExpiringAssetAMM:
    """Automated Market Maker for expiring assets."""
    
    def __init__(self, a: float = 0.5, b: float = 2.0):
        """
        Initialize the AMM.
        
        Args:
            a: Lower bound for c parameter
            b: Upper bound for c parameter
        """
        self.state = None
        self.liquidity_providers: Dict[str, LiquidityProvider] = {}
        self.active_lps: Set[str] = set()
        self.inactive_lps: Set[str] = set()
        self.buy_orders: List[Order] = []
        self.sell_orders: List[Order] = []
        self.a = a
        self.b = b
        self.transaction_fee = 0.003  # 0.3% fee
        self.current_time = 0
        self.epochs = []
        self.expiry_time = 100
        self.retrieval_time = 95
        self.history = []
        self.trade_history = []
    
    def initialize(self, initial_x: float, initial_y: float, c: float):
        """
        Initialize the AMM with some liquidity.
        
        Args:
            initial_x: Initial amount of X
            initial_y: Initial amount of Y
            c: Initial curve parameter
        """
        self.state = State(initial_x, initial_y, initial_x, initial_y, c)
        self.record_state()
    
    def add_liquidity_provider(self, lp: LiquidityProvider):
        """
        Add a liquidity provider to the AMM.
        
        Args:
            lp: The liquidity provider to add
        """
        if self.state is None:
            raise ValueError("AMM must be initialized before adding liquidity providers")
        
        # Calculate the share of the new LP
        total_x = sum(lp.x for lp in self.liquidity_providers.values()) + self.state.x
        lp.share = lp.x / total_x
        
        # Adjust shares of existing LPs
        for existing_lp in self.liquidity_providers.values():
            existing_lp.share *= (1 - lp.share)
        
        # Add the LP to the AMM
        self.liquidity_providers[lp.id] = lp
        self.active_lps.add(lp.id)
        
        # Update the AMM state
        self.state.x += lp.x
        self.state.y += lp.y
        
        # Update reference values
        self.state.x0 = self.state.x
        self.state.y0 = self.state.y
        
        self.record_state()
    
    def remove_liquidity(self, lp_id: str):
        """
        Remove a liquidity provider from the AMM.
        
        Args:
            lp_id: ID of the liquidity provider to remove
        """
        if lp_id not in self.liquidity_providers:
            raise ValueError(f"LP {lp_id} not found")
        
        lp = self.liquidity_providers[lp_id]
        
        # Calculate the amounts to return to the LP
        x_to_return = lp.share * self.state.x
        y_to_return = lp.share * self.state.y
        
        # Update the AMM state
        self.state.x -= x_to_return
        self.state.y -= y_to_return
        
        # Update reference values
        self.state.x0 = self.state.x
        self.state.y0 = self.state.y
        
        # Remove the LP
        if lp_id in self.active_lps:
            self.active_lps.remove(lp_id)
        if lp_id in self.inactive_lps:
            self.inactive_lps.remove(lp_id)
        del self.liquidity_providers[lp_id]
        
        # Adjust shares of remaining LPs
        if self.active_lps:
            total_share = sum(self.liquidity_providers[id].share for id in self.active_lps)
            for active_lp_id in self.active_lps:
                self.liquidity_providers[active_lp_id].share /= total_share
        
        self.record_state()
        
        return x_to_return, y_to_return
    
    def liquidate(self, lp_id: str):
        """
        Liquidate a liquidity provider's position.
        
        Args:
            lp_id: ID of the liquidity provider to liquidate
        """
        if lp_id not in self.liquidity_providers:
            raise ValueError(f"LP {lp_id} not found")
        
        lp = self.liquidity_providers[lp_id]
        
        # Calculate the amounts to return to the LP
        x_share = lp.share * self.state.x
        y_share = lp.share * self.state.y
        
        # Sell the Y tokens to the AMM
        x_from_sale = 0.0
        if y_share > 0 and self.active_lps:
            x_from_sale = self.calculate_x_for_y(y_share)
            # Execute the sale
            self.state.x -= x_from_sale
            self.state.y += y_share
        
        # Return X to the LP
        x_to_return = x_share + x_from_sale
        
        # Update the AMM state
        self.state.x -= x_share
        self.state.y -= y_share
        
        # Update reference values
        self.state.x0 = self.state.x
        self.state.y0 = self.state.y
        
        # Remove the LP
        if lp_id in self.active_lps:
            self.active_lps.remove(lp_id)
        if lp_id in self.inactive_lps:
            self.inactive_lps.remove(lp_id)
        del self.liquidity_providers[lp_id]
        
        # Adjust shares of remaining LPs
        if self.active_lps:
            total_share = sum(self.liquidity_providers[id].share for id in self.active_lps)
            for active_lp_id in self.active_lps:
                self.liquidity_providers[active_lp_id].share /= total_share
        
        self.record_state()
        
        return x_to_return, 0.0
    
    def calculate_x_for_y(self, delta_y: float) -> float:
        """
        Calculate how much X would be returned for a given amount of Y.
        
        Args:
            delta_y: Amount of Y to add
            
        Returns:
            Amount of X to remove
        """
        if self.state is None:
            raise ValueError("AMM not initialized")
        
        return self.state.get_x_for_y_delta(delta_y)
    
    def calculate_y_for_x(self, delta_x: float) -> float:
        """
        Calculate how much Y would be returned for a given amount of X.
        
        Args:
            delta_x: Amount of X to add
            
        Returns:
            Amount of Y to remove
        """
        if self.state is None:
            raise ValueError("AMM not initialized")
        
        return self.state.get_y_for_x_delta(delta_x)
    
    def buy_tokens(self, address: str, x_amount: float) -> float:
        """
        Buy tokens with cash.
        
        Args:
            address: Address of the buyer
            x_amount: Amount of X to spend
            
        Returns:
            Amount of Y received
        """
        if self.state is None or not self.active_lps:
            # If AMM isn't initialized or no active LPs, use order book
            self.add_buy_order(Order(f"buy_{len(self.buy_orders)}", address, 
                                     self.state.spot_price() if self.state else float('inf'), True))
            return 0.0
        
        # Calculate fee
        fee = x_amount * self.transaction_fee
        x_amount_after_fee = x_amount - fee
        
        # Calculate Y to return
        y_to_return = self.calculate_y_for_x(x_amount_after_fee)
        
        # Check if we have enough Y
        if y_to_return > self.state.y:
            # Not enough Y, add to order book
            self.add_buy_order(Order(f"buy_{len(self.buy_orders)}", address, 
                                     x_amount / y_to_return, True))
            return 0.0
        
        # Update state
        self.state.x += x_amount_after_fee
        self.state.y -= y_to_return
        
        # Distribute fee to active LPs
        for lp_id in self.active_lps:
            lp = self.liquidity_providers[lp_id]
            lp.risk_participation += fee * lp.share
        
        # Check if any LPs need to be frozen
        self.check_and_update_lp_status()
        
        # Update reference values
        self.state.x0 = self.state.x
        self.state.y0 = self.state.y
        
        self.record_state()
        self.record_trade("buy", address, x_amount, y_to_return)
        
        return y_to_return
    
    def sell_tokens(self, address: str, y_amount: float) -> float:
        """
        Sell tokens for cash.
        
        Args:
            address: Address of the seller
            y_amount: Amount of Y to sell
            
        Returns:
            Amount of X received
        """
        if self.state is None or not self.active_lps:
            # If AMM isn't initialized or no active LPs, use order book
            self.add_sell_order(Order(f"sell_{len(self.sell_orders)}", address, 
                                      self.state.spot_price() if self.state else 0.0, False))
            return 0.0
        
        # Calculate X to return
        x_to_return = self.calculate_x_for_y(y_amount)
        
        # Calculate fee
        fee = x_to_return * self.transaction_fee
        x_to_return_after_fee = x_to_return - fee
        
        # Check if we have enough X
        if x_to_return_after_fee > self.state.x:
            # Not enough X, add to order book
            self.add_sell_order(Order(f"sell_{len(self.sell_orders)}", address, 
                                      x_to_return / y_amount, False))
            return 0.0
        
        # Update state
        self.state.x -= x_to_return_after_fee
        self.state.y += y_amount
        
        # Distribute fee to active LPs
        for lp_id in self.active_lps:
            lp = self.liquidity_providers[lp_id]
            lp.risk_participation += fee * lp.share
        
        # Check if any LPs need to be frozen
        self.check_and_update_lp_status()
        
        # Update reference values
        self.state.x0 = self.state.x
        self.state.y0 = self.state.y
        
        self.record_state()
        self.record_trade("sell", address, y_amount, x_to_return_after_fee)
        
        return x_to_return_after_fee
    
    def check_and_update_lp_status(self):
        """Check and update the active/inactive status of all LPs."""
        current_price = self.state.spot_price()
        
        # Check if any active LPs need to be frozen
        freeze_lps = []
        for lp_id in self.active_lps:
            lp = self.liquidity_providers[lp_id]
            if current_price < lp.price_threshold:
                freeze_lps.append(lp_id)
        
        # Freeze LPs
        for lp_id in freeze_lps:
            self.freeze_lp(lp_id)
        
        # Check if any inactive LPs can be unfrozen
        unfreeze_lps = []
        for lp_id in self.inactive_lps:
            lp = self.liquidity_providers[lp_id]
            if current_price >= lp.price_threshold:
                unfreeze_lps.append(lp_id)
        
        # Unfreeze LPs
        for lp_id in unfreeze_lps:
            self.unfreeze_lp(lp_id)
    
    def freeze_lp(self, lp_id: str):
        """
        Freeze a liquidity provider.
        
        Args:
            lp_id: ID of the LP to freeze
        """
        if lp_id not in self.active_lps:
            return
        
        lp = self.liquidity_providers[lp_id]
        
        # Calculate amounts to freeze
        x_to_freeze = lp.share * self.state.x
        y_to_freeze = lp.share * self.state.y
        
        # Update LP's state
        lp.freeze_liquidity(x_to_freeze, y_to_freeze)
        lp.active = False
        
        # Update AMM state
        self.state.x -= x_to_freeze
        self.state.y -= y_to_freeze
        
        # Update sets
        self.active_lps.remove(lp_id)
        self.inactive_lps.add(lp_id)
        
        # Adjust shares of remaining active LPs
        if self.active_lps:
            total_share = sum(self.liquidity_providers[id].share for id in self.active_lps)
            for active_lp_id in self.active_lps:
                self.liquidity_providers[active_lp_id].share /= total_share
    
    def unfreeze_lp(self, lp_id: str):
        """
        Unfreeze a liquidity provider.
        
        Args:
            lp_id: ID of the LP to unfreeze
        """
        if lp_id not in self.inactive_lps:
            return
        
        lp = self.liquidity_providers[lp_id]
        
        # Calculate the ratio of x to y in the AMM
        if self.state.y > 0:
            ratio = self.state.x / self.state.y
        else:
            ratio = 1.0  # Default if y is 0
        
        # Determine how much to unfreeze based on ratio
        if lp.x_frozen / lp.y_frozen >= ratio:
            # More X relative to Y than the AMM ratio
            y_to_unfreeze = lp.y_frozen
            x_to_unfreeze = y_to_unfreeze * ratio
        else:
            # More Y relative to X than the AMM ratio
            x_to_unfreeze = lp.x_frozen
            y_to_unfreeze = x_to_unfreeze / ratio
        
        # Update LP's state
        lp.unfreeze_liquidity(x_to_unfreeze, y_to_unfreeze)
        lp.active = True
        
        # Calculate new share
        total_x_after = self.state.x + x_to_unfreeze
        lp.share = x_to_unfreeze / total_x_after
        
        # Adjust shares of existing active LPs
        for active_lp_id in self.active_lps:
            active_lp = self.liquidity_providers[active_lp_id]
            active_lp.share *= (1 - lp.share)
        
        # Update AMM state
        self.state.x += x_to_unfreeze
        self.state.y += y_to_unfreeze
        
        # Update sets
        self.inactive_lps.remove(lp_id)
        self.active_lps.add(lp_id)
    
    def add_buy_order(self, order: Order):
        """
        Add a buy order to the order book.
        
        Args:
            order: The buy order to add
        """
        self.buy_orders.append(order)
        self.buy_orders.sort(key=lambda o: o.price, reverse=True)
        self.match_orders()
    
    def add_sell_order(self, order: Order):
        """
        Add a sell order to the order book.
        
        Args:
            order: The sell order to add
        """
        self.sell_orders.append(order)
        self.sell_orders.sort(key=lambda o: o.price)
        self.match_orders()
    
    def match_orders(self):
        """Match buy and sell orders if possible."""
        while self.buy_orders and self.sell_orders:
            best_buy = self.buy_orders[0]
            best_sell = self.sell_orders[0]
            
            if best_buy.price >= best_sell.price:
                # Execute the trade at the buy price
                self.record_trade("match", best_buy.address, best_buy.price, 1.0)
                self.buy_orders.pop(0)
                self.sell_orders.pop(0)
            else:
                break
    
    def adjust_curve(self):
        """Adjust the curve based on LP preferences at epoch boundaries."""
        if not self.active_lps:
            return
        
        # Calculate the new curve parameter
        c_values = []
        shares = []
        
        for lp_id in self.active_lps:
            lp = self.liquidity_providers[lp_id]
            c_values.append(lp.c_value)
            shares.append(lp.share)
        
        # Use weighted geometric mean for aggregation
        c_new = np.exp(np.sum([np.log(c) * s for c, s in zip(c_values, shares)]))
        
        # Ensure c is within bounds
        c_new = max(min(c_new, self.b), self.a)
        
        # Update the AMM curve
        self.state.c = c_new
        
        # Update reference values
        self.state.x0 = self.state.x
        self.state.y0 = self.state.y
        
        self.record_state()
    
    def run_market_clearing_auction(self):
        """Run the market clearing auction at retrieval time."""
        if not self.buy_orders:
            return
        
        # Sort buy orders by price (highest first)
        self.buy_orders.sort(key=lambda o: o.price, reverse=True)
        
        # Calculate total Y available across all LPs
        total_y = sum(lp.y + lp.y_frozen for lp in self.liquidity_providers.values())
        
        # Execute the auction while we have buy orders and Y tokens
        while self.buy_orders and total_y >= 1.0:
            # Take the highest bid
            best_buy = self.buy_orders.pop(0)
            
            # If there's another bid, use second price auction
            second_price = best_buy.price
            if self.buy_orders:
                second_price = self.buy_orders[0].price
            
            # Distribute Y tokens to the buyer and X to LPs based on risk participation
            self.execute_auction_trade(best_buy.address, second_price)
            total_y -= 1.0
    
    def execute_auction_trade(self, buyer_address: str, price: float):
        """
        Execute a trade in the auction.
        
        Args:
            buyer_address: Address of the buyer
            price: Price at which to execute the trade
        """
        # Calculate total risk participation
        total_risk = sum(lp.risk_participation for lp in self.liquidity_providers.values())
        
        # If no risk participation, use shares instead
        if total_risk == 0:
            total_risk = 1.0
            for lp in self.liquidity_providers.values():
                lp.risk_participation = lp.share
        
        # Distribute X from the buyer to LPs based on risk participation
        for lp_id, lp in self.liquidity_providers.items():
            lp_share = lp.risk_participation / total_risk
            x_to_lp = price * lp_share
            
            # Update LP state
            if lp_id in self.active_lps:
                # Active LP
                if lp.y >= lp_share:
                    # LP has enough Y
                    lp.y -= lp_share
                    lp.x += x_to_lp
                else:
                    # LP doesn't have enough Y, use frozen Y if available
                    y_from_active = lp.y
                    y_needed = lp_share - y_from_active
                    
                    lp.y = 0
                    if lp.y_frozen >= y_needed:
                        lp.y_frozen -= y_needed
                    else:
                        # Not enough Y even with frozen, adjust allocation
                        y_needed = lp.y_frozen
                        lp.y_frozen = 0
                    
                    lp.x += x_to_lp * (y_from_active + y_needed) / lp_share
            else:
                # Inactive LP
                if lp.y_frozen >= lp_share:
                    # LP has enough frozen Y
                    lp.y_frozen -= lp_share
                    lp.x_frozen += x_to_lp
                else:
                    # Not enough Y, adjust allocation
                    y_given = lp.y_frozen
                    lp.y_frozen = 0
                    lp.x_frozen += x_to_lp * y_given / lp_share
        
        # Record the trade
        self.record_trade("auction", buyer_address, price, 1.0)
    
    def advance_time(self, time_delta: float = 1.0):
        """
        Advance the simulation time.
        
        Args:
            time_delta: Amount of time to advance
        """
        self.current_time += time_delta
        
        # Check if we've hit an epoch boundary
        if self.current_time in self.epochs:
            self.adjust_curve()
        
        # Check if we've hit the retrieval time
        if self.current_time >= self.retrieval_time and self.current_time - time_delta < self.retrieval_time:
            self.run_market_clearing_auction()
        
        # Check if tokens have expired
        if self.current_time >= self.expiry_time:
            # Tokens have expired, they're now worthless
            for lp in self.liquidity_providers.values():
                lp.y = 0
                lp.y_frozen = 0
            
            if self.state:
                self.state.y = 0
    
    def set_epochs(self, epochs: List[float]):
        """
        Set the epoch boundaries.
        
        Args:
            epochs: List of epoch boundary times
        """
        self.epochs = sorted(epochs)
    
    def record_state(self):
        """Record the current state for history."""
        if self.state is None:
            return
        
        self.history.append({
            'time': self.current_time,
            'x': self.state.x,
            'y': self.state.y,
            'price': self.state.spot_price(),
            'c': self.state.c,
            'active_lps': len(self.active_lps),
            'inactive_lps': len(self.inactive_lps)
        })
    
    def record_trade(self, trade_type: str, address: str, amount_in: float, amount_out: float):
        """
        Record a trade for history.
        
        Args:
            trade_type: Type of trade (buy, sell, match, auction)
            address: Address of the trader
            amount_in: Amount put into the trade
            amount_out: Amount received from the trade
        """
        self.trade_history.append({
            'time': self.current_time,
            'type': trade_type,
            'address': address,
            'amount_in': amount_in,
            'amount_out': amount_out,
            'price': amount_in / amount_out if amount_out > 0 else 0
        })
    
    def get_history_df(self) -> pd.DataFrame:
        """Get the history as a DataFrame."""
        return pd.DataFrame(self.history)
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get the trade history as a DataFrame."""
        return pd.DataFrame(self.trade_history)


class Producer:
    """A token producer who creates and sells expiring tokens."""
    
    def __init__(self, id: str, production_capacity: float, min_price: float,
                 urgency_profile: str = 'normal'):
        """
        Initialize a producer.
        
        Args:
            id: Unique identifier
            production_capacity: Maximum number of tokens that can be produced
            min_price: Minimum price at which producer will sell tokens
            urgency_profile: 'normal', 'urgent', or 'last_minute'
        """
        self.id = id
        self.production_capacity = production_capacity
        self.min_price = min_price
        self.urgency_profile = urgency_profile
        self.tokens_produced = 0
        self.cash_earned = 0
    
    def decide_action(self, amm: ExpiringAssetAMM) -> Optional[str]:
        """
        Decide what action to take based on current market conditions.
        
        Args:
            amm: The AMM
            
        Returns:
            Action to take ('sell', 'order', None)
        """
        if self.tokens_produced >= self.production_capacity:
            return None  # No more production capacity
        
        current_price = amm.state.spot_price() if amm.state else 0
        time_to_expiry = amm.expiry_time - amm.current_time
        
        # Check if production is still possible
        if time_to_expiry <= 0:
            return None  # Can't produce after expiry
        
        # Adjust pricing based on urgency profile and time to expiry
        if self.urgency_profile == 'normal':
            # Normal producers sell at market price if it's above their minimum
            if current_price >= self.min_price and amm.active_lps:
                return 'sell'
        elif self.urgency_profile == 'urgent':
            # Urgent producers become more willing to sell as expiry approaches
            adjusted_min_price = max(self.min_price * (0.5 + 0.5 * time_to_expiry / amm.expiry_time), 
                                    self.min_price * 0.5)
            if current_price >= adjusted_min_price and amm.active_lps:
                return 'sell'
        elif self.urgency_profile == 'last_minute':
            # Last-minute producers panic as expiry approaches
            if time_to_expiry < amm.expiry_time * 0.2:
                adjusted_min_price = max(self.min_price * (time_to_expiry / (amm.expiry_time * 0.2)), 
                                        self.min_price * 0.1)
                if current_price >= adjusted_min_price and amm.active_lps:
                    return 'sell'
        
        # If price is below minimum but close to retrieval time, place an order
        if time_to_expiry < amm.expiry_time * 0.1:
            return 'order'
        
        return None
    
    def execute_action(self, action: str, amm: ExpiringAssetAMM) -> None:
        """
        Execute the decided action.
        
        Args:
            action: Action to execute ('sell', 'order', None)
            amm: The AMM
        """
        if action == 'sell':
            # Sell one token to the AMM
            x_received = amm.sell_tokens(self.id, 1.0)
            self.tokens_produced += 1
            self.cash_earned += x_received
        elif action == 'order':
            # Place a sell order
            order = Order(f"sell_{self.id}_{amm.current_time}", self.id, self.min_price, False)
            amm.add_sell_order(order)
            self.tokens_produced += 1


class Consumer:
    """A token consumer who buys expiring tokens."""
    
    def __init__(self, id: str, budget: float, max_price: float,
                 urgency_profile: str = 'normal'):
        """
        Initialize a consumer.
        
        Args:
            id: Unique identifier
            budget: Maximum amount of cash available
            max_price: Maximum price willing to pay for a token
            urgency_profile: 'normal', 'bargain_hunter', or 'high_flyer'
        """
        self.id = id
        self.budget = budget
        self.max_price = max_price
        self.urgency_profile = urgency_profile
        self.tokens_purchased = 0
        self.cash_spent = 0
    
    def decide_action(self, amm: ExpiringAssetAMM) -> Optional[str]:
        """
        Decide what action to take based on current market conditions.
        
        Args:
            amm: The AMM
            
        Returns:
            Action to take ('buy', 'order', None)
        """
        if self.cash_spent >= self.budget:
            return None  # No more budget
        
        current_price = amm.state.spot_price() if amm.state else float('inf')
        time_to_expiry = amm.expiry_time - amm.current_time
        
        # Check if consumption is still possible
        if time_to_expiry <= 0:
            return None  # Can't buy after expiry
        
        # Adjust pricing based on urgency profile and time to expiry
        if self.urgency_profile == 'normal':
            # Normal consumers buy at market price if it's below their maximum
            if current_price <= self.max_price and amm.active_lps and amm.state.y > 0:
                return 'buy'
        elif self.urgency_profile == 'bargain_hunter':
            # Bargain hunters wait for lower prices as expiry approaches
            if time_to_expiry < amm.expiry_time * 0.2:
                # Only interested in last-minute deals
                adjusted_max_price = self.max_price * (0.5 * time_to_expiry / (amm.expiry_time * 0.2) + 0.5)
                if current_price <= adjusted_max_price and amm.active_lps and amm.state.y > 0:
                    return 'buy'
                return 'order'  # Place a low bid
        elif self.urgency_profile == 'high_flyer':
            # High flyers become more willing to pay as expiry approaches
            adjusted_max_price = self.max_price * (2 - time_to_expiry / amm.expiry_time)
            if current_price <= adjusted_max_price and amm.active_lps and amm.state.y > 0:
                return 'buy'
            elif time_to_expiry < amm.expiry_time * 0.3:
                return 'order'  # Place a high bid
        
        # If not buying directly, consider placing an order as retrieval time approaches
        if time_to_expiry < amm.expiry_time * 0.15:
            return 'order'
        
        return None
    
    def execute_action(self, action: str, amm: ExpiringAssetAMM) -> None:
        """
        Execute the decided action.
        
        Args:
            action: Action to execute ('buy', 'order', None)
            amm: The AMM
        """
        remaining_budget = self.budget - self.cash_spent
        
        if action == 'buy':
            # Buy a token from the AMM
            current_price = amm.state.spot_price()
            if current_price <= remaining_budget:
                y_received = amm.buy_tokens(self.id, current_price)
                if y_received > 0:
                    self.tokens_purchased += y_received
                    self.cash_spent += current_price
        elif action == 'order':
            # Place a buy order
            if self.urgency_profile == 'bargain_hunter':
                # Bargain hunters bid low
                bid_price = self.max_price * 0.7
            elif self.urgency_profile == 'high_flyer':
                # High flyers bid high
                time_to_expiry = amm.expiry_time - amm.current_time
                bid_price = self.max_price * (2 - time_to_expiry / amm.expiry_time)
            else:
                # Normal consumers bid their max price
                bid_price = self.max_price
            
            # Ensure bid is within budget
            bid_price = min(bid_price, remaining_budget)
            
            # Place the order
            order = Order(f"buy_{self.id}_{amm.current_time}", self.id, bid_price, True)
            amm.add_buy_order(order)


def run_simulation(num_producers=10, num_consumers=20, num_lps=5, 
                  duration=100, epochs=None, expiry_time=100, retrieval_time=95):
    """
    Run a simulation of the expiring asset AMM.
    
    Args:
        num_producers: Number of producers
        num_consumers: Number of consumers
        num_lps: Number of liquidity providers
        duration: Duration of the simulation
        epochs: List of epoch boundaries
        expiry_time: Time at which tokens expire
        retrieval_time: Time at which market clearing auction happens
        
    Returns:
        The AMM after simulation
    """
    # Initialize AMM
    amm = ExpiringAssetAMM()
    amm.expiry_time = expiry_time
    amm.retrieval_time = retrieval_time
    
    # Set epochs if provided
    if epochs:
        amm.set_epochs(epochs)
    else:
        # Default epochs at 20% intervals
        amm.set_epochs([i * duration / 5 for i in range(1, 5)])
    
    # Initialize AMM with some liquidity
    initial_x = 1000.0
    initial_y = 100.0
    initial_c = 1.0
    amm.initialize(initial_x, initial_y, initial_c)
    
    # Create liquidity providers
    lps = []
    for i in range(num_lps):
        c_value = np.random.uniform(0.8, 1.2)
        min_price = np.random.uniform(5.0, 15.0)
        initial_x_lp = np.random.uniform(100.0, 500.0)
        initial_y_lp = initial_x_lp / amm.state.spot_price()
        
        lp = LiquidityProvider(f"lp_{i}", initial_x_lp, initial_y_lp, min_price, c_value)
        lps.append(lp)
        amm.add_liquidity_provider(lp)
    
    # Create producers
    producers = []
    for i in range(num_producers):
        profile = np.random.choice(['normal', 'urgent', 'last_minute'], p=[0.6, 0.3, 0.1])
        capacity = np.random.uniform(5.0, 20.0)
        min_price = np.random.uniform(5.0, 15.0)
        
        producer = Producer(f"producer_{i}", capacity, min_price, profile)
        producers.append(producer)
    
    # Create consumers
    consumers = []
    for i in range(num_consumers):
        profile = np.random.choice(['normal', 'bargain_hunter', 'high_flyer'], p=[0.6, 0.2, 0.2])
        budget = np.random.uniform(50.0, 200.0)
        max_price = np.random.uniform(10.0, 30.0)
        
        consumer = Consumer(f"consumer_{i}", budget, max_price, profile)
        consumers.append(consumer)
    
    # Run the simulation
    for t in tqdm(range(duration)):
        # Random order for agents to act
        random.shuffle(producers)
        random.shuffle(consumers)
        
        # Producers act
        for producer in producers:
            action = producer.decide_action(amm)
            if action:
                producer.execute_action(action, amm)
        
        # Consumers act
        for consumer in consumers:
            action = consumer.decide_action(amm)
            if action:
                consumer.execute_action(action, amm)
        
        # LPs may choose to remove liquidity or liquidate
        if t in amm.epochs:
            for lp in list(amm.liquidity_providers.values()):
                # 5% chance of removing liquidity at epoch boundary
                if random.random() < 0.05:
                    amm.remove_liquidity(lp.id)
        
        # Advance time
        amm.advance_time()
    
    return amm, producers, consumers, lps


def analyze_results(amm, producers, consumers, lps):
    """
    Analyze the results of the simulation.
    
    Args:
        amm: The AMM after simulation
        producers: List of producers
        consumers: List of consumers
        lps: List of liquidity providers
    """
    # Get history as DataFrames
    history_df = amm.get_history_df()
    trade_df = amm.get_trade_history_df()
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Price over time
    plt.subplot(2, 2, 1)
    plt.plot(history_df['time'], history_df['price'])
    plt.axvline(x=amm.retrieval_time, color='r', linestyle='--', label='Retrieval Time')
    plt.axvline(x=amm.expiry_time, color='g', linestyle='--', label='Expiry Time')
    for epoch in amm.epochs:
        plt.axvline(x=epoch, color='k', linestyle=':', alpha=0.5)
    plt.title('Price Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    # Plot 2: Liquidity (X and Y) over time
    plt.subplot(2, 2, 2)
    plt.plot(history_df['time'], history_df['x'], label='X (Cash)')
    plt.plot(history_df['time'], history_df['y'], label='Y (Tokens)')
    plt.axvline(x=amm.retrieval_time, color='r', linestyle='--', label='Retrieval Time')
    plt.axvline(x=amm.expiry_time, color='g', linestyle='--', label='Expiry Time')
    plt.title('Liquidity Over Time')
    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.legend()
    
    # Plot 3: Number of active vs inactive LPs
    plt.subplot(2, 2, 3)
    plt.plot(history_df['time'], history_df['active_lps'], label='Active LPs')
    plt.plot(history_df['time'], history_df['inactive_lps'], label='Inactive LPs')
    plt.axvline(x=amm.retrieval_time, color='r', linestyle='--', label='Retrieval Time')
    plt.axvline(x=amm.expiry_time, color='g', linestyle='--', label='Expiry Time')
    plt.title('Active vs Inactive LPs')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot 4: Curve parameter c over time
    plt.subplot(2, 2, 4)
    plt.plot(history_df['time'], history_df['c'])
    plt.axvline(x=amm.retrieval_time, color='r', linestyle='--', label='Retrieval Time')
    plt.axvline(x=amm.expiry_time, color='g', linestyle='--', label='Expiry Time')
    plt.title('Curve Parameter c Over Time')
    plt.xlabel('Time')
    plt.ylabel('c')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('amm_metrics.png')
    plt.close()
    
    # Plot trade analysis
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Trade volume over time
    plt.subplot(2, 2, 1)
    trade_counts = trade_df.groupby('time').size()
    plt.bar(trade_counts.index, trade_counts.values)
    plt.axvline(x=amm.retrieval_time, color='r', linestyle='--', label='Retrieval Time')
    plt.axvline(x=amm.expiry_time, color='g', linestyle='--', label='Expiry Time')
    plt.title('Trade Volume Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Trades')
    plt.legend()
    
    # Plot 2: Trade types
    plt.subplot(2, 2, 2)
    trade_types = trade_df['type'].value_counts()
    plt.pie(trade_types.values, labels=trade_types.index, autopct='%1.1f%%')
    plt.title('Trade Types')
    
    # Plot 3: Trade prices over time
    plt.subplot(2, 2, 3)
    for trade_type in trade_df['type'].unique():
        type_df = trade_df[trade_df['type'] == trade_type]
        plt.scatter(type_df['time'], type_df['price'], label=trade_type, alpha=0.7)
    plt.axvline(x=amm.retrieval_time, color='r', linestyle='--', label='Retrieval Time')
    plt.axvline(x=amm.expiry_time, color='g', linestyle='--', label='Expiry Time')
    plt.title('Trade Prices by Type')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    # Plot 4: Cumulative trading volume
    plt.subplot(2, 2, 4)
    cumulative_volume = trade_df.groupby('time').size().cumsum()
    plt.plot(cumulative_volume.index, cumulative_volume.values)
    plt.axvline(x=amm.retrieval_time, color='r', linestyle='--', label='Retrieval Time')
    plt.axvline(x=amm.expiry_time, color='g', linestyle='--', label='Expiry Time')
    plt.title('Cumulative Trading Volume')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Number of Trades')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('trade_analysis.png')
    plt.close()
    
    # Analyze market participants
    # Producer analysis
    producer_df = pd.DataFrame([{
        'id': p.id,
        'profile': p.urgency_profile,
        'capacity': p.production_capacity,
        'tokens_produced': p.tokens_produced,
        'cash_earned': p.cash_earned,
        'min_price': p.min_price,
        'utilization': p.tokens_produced / p.production_capacity if p.production_capacity > 0 else 0,
        'avg_price': p.cash_earned / p.tokens_produced if p.tokens_produced > 0 else 0
    } for p in producers])
    
    # Consumer analysis
    consumer_df = pd.DataFrame([{
        'id': c.id,
        'profile': c.urgency_profile,
        'budget': c.budget,
        'tokens_purchased': c.tokens_purchased,
        'cash_spent': c.cash_spent,
        'max_price': c.max_price,
        'budget_utilization': c.cash_spent / c.budget if c.budget > 0 else 0,
        'avg_price': c.cash_spent / c.tokens_purchased if c.tokens_purchased > 0 else 0
    } for c in consumers])
    
    # LP analysis
    lp_df = pd.DataFrame([{
        'id': lp.id,
        'initial_x': lp.x + lp.x_frozen,
        'initial_y': lp.y + lp.y_frozen,
        'final_x': lp.x + lp.x_frozen,
        'final_y': lp.y + lp.y_frozen,
        'active': lp.active,
        'price_threshold': lp.price_threshold,
        'c_value': lp.c_value,
        'risk_participation': lp.risk_participation
    } for lp in amm.liquidity_providers.values()])
    
    # Plot participant analysis
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Producer profiles
    plt.subplot(3, 2, 1)
    producer_profiles = producer_df['profile'].value_counts()
    plt.pie(producer_profiles.values, labels=producer_profiles.index, autopct='%1.1f%%')
    plt.title('Producer Profiles')
    
    # Plot 2: Consumer profiles
    plt.subplot(3, 2, 2)
    consumer_profiles = consumer_df['profile'].value_counts()
    plt.pie(consumer_profiles.values, labels=consumer_profiles.index, autopct='%1.1f%%')
    plt.title('Consumer Profiles')
    
    # Plot 3: Producer capacity utilization by profile
    plt.subplot(3, 2, 3)
    sns.boxplot(x='profile', y='utilization', data=producer_df)
    plt.title('Producer Capacity Utilization by Profile')
    plt.ylim(0, 1)
    
    # Plot 4: Consumer budget utilization by profile
    plt.subplot(3, 2, 4)
    sns.boxplot(x='profile', y='budget_utilization', data=consumer_df)
    plt.title('Consumer Budget Utilization by Profile')
    plt.ylim(0, 1)
    
    # Plot 5: Average price paid/received by producer profile
    plt.subplot(3, 2, 5)
    sns.boxplot(x='profile', y='avg_price', data=producer_df)
    plt.title('Average Price Received by Producer Profile')
    
    # Plot 6: Average price paid by consumer profile
    plt.subplot(3, 2, 6)
    sns.boxplot(x='profile', y='avg_price', data=consumer_df)
    plt.title('Average Price Paid by Consumer Profile')
    
    plt.tight_layout()
    plt.savefig('participant_analysis.png')
    plt.close()
    
    # LP analysis plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: LP risk participation vs initial investment
    plt.subplot(2, 2, 1)
    plt.scatter(lp_df['initial_x'], lp_df['risk_participation'])
    plt.title('LP Risk Participation vs Initial Investment')
    plt.xlabel('Initial X Investment')
    plt.ylabel('Risk Participation')
    
    # Plot 2: LP final position (active vs inactive)
    plt.subplot(2, 2, 2)
    active_lps = lp_df[lp_df['active']]
    inactive_lps = lp_df[~lp_df['active']]
    plt.scatter(active_lps['final_x'], active_lps['final_y'], label='Active', color='green')
    plt.scatter(inactive_lps['final_x'], inactive_lps['final_y'], label='Inactive', color='red')
    plt.title('LP Final Positions')
    plt.xlabel('Final X')
    plt.ylabel('Final Y')
    plt.legend()
    
    # Plot 3: LP price thresholds vs c values
    plt.subplot(2, 2, 3)
    plt.scatter(lp_df['price_threshold'], lp_df['c_value'])
    plt.title('LP Price Thresholds vs c Values')
    plt.xlabel('Price Threshold')
    plt.ylabel('c Value')
    
    # Plot 4: LP X-Y change
    plt.subplot(2, 2, 4)
    plt.scatter(lp_df['initial_x'], lp_df['initial_y'], label='Initial', alpha=0.5)
    plt.scatter(lp_df['final_x'], lp_df['final_y'], label='Final')
    for i, row in lp_df.iterrows():
        plt.arrow(row['initial_x'], row['initial_y'], 
                 row['final_x'] - row['initial_x'], row['final_y'] - row['initial_y'],
                 alpha=0.3, width=0.5)
    plt.title('LP Position Changes')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('lp_analysis.png')
    plt.close()
    
    # Print summary statistics
    print("=== Simulation Summary ===")
    print(f"Duration: {amm.current_time} time units")
    print(f"Final Price: {history_df['price'].iloc[-1]:.4f}")
    print(f"Total Trades: {len(trade_df)}")
    print(f"AMM Trades: {len(trade_df[trade_df['type'].isin(['buy', 'sell'])])}")
    print(f"Order Book Matches: {len(trade_df[trade_df['type'] == 'match'])}")
    print(f"Auction Trades: {len(trade_df[trade_df['type'] == 'auction'])}")
    print("\n=== Market Clearing ===")
    print(f"Tokens Produced: {producer_df['tokens_produced'].sum()}")
    print(f"Tokens Purchased: {consumer_df['tokens_purchased'].sum()}")
    print(f"Market Clearing Rate: {consumer_df['tokens_purchased'].sum() / producer_df['tokens_produced'].sum():.2%}")
    print("\n=== Participants ===")
    print(f"Producers: {len(producers)}")
    print(f"Consumers: {len(consumers)}")
    print(f"Liquidity Providers: {len(lp_df)}")
    print(f"Active LPs at End: {len(lp_df[lp_df['active']])}")
    
    # Return the dataframes for further analysis
    return {
        'history': history_df,
        'trades': trade_df,
        'producers': producer_df,
        'consumers': consumer_df,
        'lps': lp_df
    }


# Run a simulation
amm, producers, consumers, lps = run_simulation(
    num_producers=20,
    num_consumers=30,
    num_lps=8,
    duration=100,
    expiry_time=100,
    retrieval_time=95
)

# Analyze the results
results = analyze_results(amm, producers, consumers, lps)