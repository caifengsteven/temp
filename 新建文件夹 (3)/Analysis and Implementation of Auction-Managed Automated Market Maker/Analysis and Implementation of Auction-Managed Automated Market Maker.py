import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import random
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class AMM:
    """Base class for Automated Market Makers"""
    
    def __init__(self, x_reserve, y_reserve, name="Generic AMM"):
        """
        Initialize an AMM with x and y reserves
        x_reserve: amount of token X (risky asset)
        y_reserve: amount of token Y (stable asset/numeraire)
        """
        self.x_reserve = x_reserve
        self.y_reserve = y_reserve
        self.k = x_reserve * y_reserve  # Constant product invariant
        self.liquidity = np.sqrt(self.k)  # Liquidity measure as in the paper
        self.name = name
        
        # History tracking
        self.history = {
            'time': [],
            'x_reserve': [],
            'y_reserve': [],
            'liquidity': [],
            'spot_price': [],
            'external_price': [],
            'mispricing': [],
            'fees_collected': [],
            'arb_profits': [],
            'lp_profits': [],
            'lp_value': []
        }
        self.total_fees = 0
        self.total_arb_profits = 0
        self.total_lp_profits = 0
        
    def spot_price(self):
        """Get the current spot price (y/x)"""
        return self.y_reserve / self.x_reserve
    
    def get_value(self, external_price):
        """Get the total value of the pool in terms of the numeraire"""
        return self.y_reserve + external_price * self.x_reserve
    
    def get_liquidity(self):
        """Get the current liquidity measure"""
        return self.liquidity
    
    def add_liquidity(self, x_amount, y_amount):
        """Add liquidity to the pool"""
        # Check that the added liquidity matches the current price ratio
        if abs(y_amount / x_amount - self.spot_price()) > 1e-10:
            raise ValueError("Added liquidity must match current price ratio")
        
        # Update reserves
        self.x_reserve += x_amount
        self.y_reserve += y_amount
        
        # Update invariant and liquidity
        self.k = self.x_reserve * self.y_reserve
        self.liquidity = np.sqrt(self.k)
        
        return True
    
    def remove_liquidity(self, liquidity_fraction, withdrawal_fee=0):
        """Remove a fraction of liquidity from the pool"""
        if liquidity_fraction <= 0 or liquidity_fraction > 1:
            raise ValueError("Liquidity fraction must be between 0 and 1")
        
        # Calculate amounts to remove (proportional to current reserves)
        x_remove = self.x_reserve * liquidity_fraction
        y_remove = self.y_reserve * liquidity_fraction
        
        # Apply withdrawal fee if specified
        if withdrawal_fee > 0:
            x_remove *= (1 - withdrawal_fee)
            y_remove *= (1 - withdrawal_fee)
        
        # Update reserves
        self.x_reserve -= x_remove
        self.y_reserve -= y_remove
        
        # Update invariant and liquidity
        self.k = self.x_reserve * self.y_reserve
        self.liquidity = np.sqrt(self.k)
        
        return x_remove, y_remove
    
    def get_output_amount(self, input_amount, input_is_x, fee=0):
        """
        Calculate the output amount for a swap with the given input
        input_amount: amount of input token
        input_is_x: True if input token is X, False if input token is Y
        fee: fee rate (as a decimal)
        """
        if input_amount <= 0:
            return 0
        
        # Apply fee to input amount
        input_amount_after_fee = input_amount * (1 - fee)
        
        if input_is_x:
            # Calculate new Y reserve after swap
            new_y_reserve = self.k / (self.x_reserve + input_amount_after_fee)
            # Output is the difference in Y reserves
            output_amount = self.y_reserve - new_y_reserve
        else:
            # Calculate new X reserve after swap
            new_x_reserve = self.k / (self.y_reserve + input_amount_after_fee)
            # Output is the difference in X reserves
            output_amount = self.x_reserve - new_x_reserve
        
        return output_amount
    
    def swap(self, input_amount, input_is_x, fee=0):
        """
        Execute a swap with the given input
        input_amount: amount of input token
        input_is_x: True if input token is X, False if input token is Y
        fee: fee rate (as a decimal)
        """
        # Apply fee to input amount
        input_amount_after_fee = input_amount * (1 - fee)
        fee_amount = input_amount * fee
        
        # Calculate output amount
        output_amount = self.get_output_amount(input_amount, input_is_x, fee)
        
        # Update reserves
        if input_is_x:
            self.x_reserve += input_amount_after_fee
            self.y_reserve -= output_amount
            # Track fees collected (in terms of token X)
            self.total_fees += fee_amount
        else:
            self.y_reserve += input_amount_after_fee
            self.x_reserve -= output_amount
            # Track fees collected (in terms of token Y)
            self.total_fees += fee_amount
        
        # Update invariant (should remain constant except for rounding errors)
        self.k = self.x_reserve * self.y_reserve
        
        return output_amount, fee_amount
    
    def record_state(self, time, external_price, fees_collected=0, arb_profits=0, lp_profits=0):
        """Record the current state of the AMM for later analysis"""
        self.history['time'].append(time)
        self.history['x_reserve'].append(self.x_reserve)
        self.history['y_reserve'].append(self.y_reserve)
        self.history['liquidity'].append(self.liquidity)
        self.history['spot_price'].append(self.spot_price())
        self.history['external_price'].append(external_price)
        self.history['mispricing'].append(np.log(self.spot_price() / external_price))
        self.history['fees_collected'].append(fees_collected)
        self.history['arb_profits'].append(arb_profits)
        self.history['lp_profits'].append(lp_profits)
        self.history['lp_value'].append(self.get_value(external_price))
    
    def get_history_df(self):
        """Get the history as a pandas DataFrame"""
        return pd.DataFrame(self.history)

class FixedFeeAMM(AMM):
    """Fixed-fee AMM implementation"""
    
    def __init__(self, x_reserve, y_reserve, fee=0.003, name="Fixed-fee AMM"):
        """
        Initialize a fixed-fee AMM
        x_reserve: amount of token X (risky asset)
        y_reserve: amount of token Y (stable asset/numeraire)
        fee: fixed fee rate (as a decimal)
        """
        super().__init__(x_reserve, y_reserve, name)
        self.fee = fee
    
    def execute_swap(self, input_amount, input_is_x):
        """Execute a swap with the fixed fee"""
        return self.swap(input_amount, input_is_x, self.fee)
    
    def handle_arbitrage(self, external_price):
        """
        Handle arbitrage against an external market
        external_price: the price of token X in terms of token Y on an external market
        
        Returns the profit made by arbitrageurs and the fees collected
        """
        spot_price = self.spot_price()
        
        # Calculate mispricing (in log space)
        mispricing = np.log(spot_price / external_price)
        
        # No arbitrage if within fee threshold
        if abs(mispricing) <= np.log(1 + self.fee):
            return 0, 0
        
        arb_profit = 0
        fees_collected = 0
        
        # If AMM price is too high, buy X from external market and sell to AMM
        if spot_price > external_price * (1 + self.fee):
            # Calculate optimal trade size to bring price to external price
            # after considering the fee
            new_y_reserve = np.sqrt(self.k * external_price / (1 - self.fee))
            delta_y = new_y_reserve - self.y_reserve
            
            if delta_y > 0:
                # Execute the swap
                delta_x = self.get_output_amount(delta_y, False, self.fee)
                output, fee = self.swap(delta_y, False, self.fee)
                
                # Calculate arbitrage profit
                # Cost of X on external market = delta_x * external_price
                # Revenue from selling to AMM = delta_y
                arb_profit = delta_y - delta_x * external_price
                fees_collected = fee
        
        # If AMM price is too low, buy X from AMM and sell to external market
        elif spot_price < external_price / (1 + self.fee):
            # Calculate optimal trade size to bring price to external price
            # after considering the fee
            new_x_reserve = np.sqrt(self.k / (external_price * (1 - self.fee)))
            delta_x = new_x_reserve - self.x_reserve
            
            if delta_x > 0:
                # Execute the swap
                delta_y = self.get_output_amount(delta_x, True, self.fee)
                output, fee = self.swap(delta_x, True, self.fee)
                
                # Calculate arbitrage profit
                # Cost of Y to buy from AMM = delta_y
                # Revenue from selling X on external market = delta_x * external_price
                arb_profit = delta_x * external_price - delta_y
                fees_collected = fee
        
        self.total_arb_profits += arb_profit
        return arb_profit, fees_collected

class AuctionManagedAMM(AMM):
    """Auction-managed AMM implementation"""
    
    def __init__(self, x_reserve, y_reserve, max_fee=0.01, 
                 withdrawal_fee=0.0001238, name="Auction-managed AMM"):
        """
        Initialize an auction-managed AMM
        x_reserve: amount of token X (risky asset)
        y_reserve: amount of token Y (stable asset/numeraire)
        max_fee: maximum fee rate allowed (as a decimal)
        withdrawal_fee: fee charged when LPs withdraw liquidity
        """
        super().__init__(x_reserve, y_reserve, name)
        self.max_fee = max_fee
        self.withdrawal_fee = withdrawal_fee
        self.current_fee = max_fee / 2  # Start with middle fee
        self.manager = None
        self.rent = 0
        
    def remove_liquidity(self, liquidity_fraction):
        """Override to apply the withdrawal fee"""
        return super().remove_liquidity(liquidity_fraction, self.withdrawal_fee)
    
    def set_manager(self, manager, rent):
        """Set a new pool manager and rent"""
        self.manager = manager
        self.rent = rent
    
    def set_fee(self, fee):
        """Set a new fee rate (up to max_fee)"""
        if fee < 0 or fee > self.max_fee:
            raise ValueError(f"Fee must be between 0 and {self.max_fee}")
        self.current_fee = fee
    
    def execute_swap(self, input_amount, input_is_x, from_manager=False):
        """
        Execute a swap
        from_manager: if True, no fee is charged (manager can trade fee-free)
        """
        fee = 0 if from_manager else self.current_fee
        return self.swap(input_amount, input_is_x, fee)
    
    def handle_arbitrage(self, external_price):
        """
        Handle arbitrage against an external market
        
        Returns:
        - manager_profit: profit captured by the manager
        - arb_excess: profit leaked to external arbitrageurs
        - manager_fees: fees collected by the manager from noise traders
        """
        spot_price = self.spot_price()
        
        # Calculate mispricing (in log space)
        mispricing = np.log(spot_price / external_price)
        
        manager_profit = 0
        arb_excess = 0
        
        # Calculate no-trade region boundaries
        no_trade_lower = np.log(1 / (1 + self.current_fee))
        no_trade_upper = np.log(1 + self.current_fee)
        
        # If within no-trade region, no arbitrage
        if no_trade_lower <= mispricing <= no_trade_upper:
            return 0, 0, 0
        
        # If outside no-trade region but manager can capture all arbitrage
        # First, external arbitrageurs correct price to the boundary of the no-trade region
        if mispricing > no_trade_upper:
            # AMM price is too high, arbs buy X from external and sell to AMM
            # They correct until the price is at the upper boundary
            boundary_price = external_price * (1 + self.current_fee)
            
            # Calculate reserves after external arb
            new_x_reserve = self.x_reserve
            new_y_reserve = boundary_price * new_x_reserve
            
            # Calculate how much value leaked to external arbitrageurs
            value_before = self.y_reserve + external_price * self.x_reserve
            value_after = new_y_reserve + external_price * new_x_reserve
            arb_excess = value_before - value_after
            
            # Now manager corrects the rest of the way with zero fee
            # They trade until the AMM price equals the external price
            final_x_reserve = np.sqrt(self.k / external_price)
            final_y_reserve = external_price * final_x_reserve
            
            # Calculate manager's profit
            manager_value_before = new_y_reserve + external_price * new_x_reserve
            manager_value_after = final_y_reserve + external_price * final_x_reserve
            manager_profit = manager_value_before - manager_value_after
            
            # Update reserves to final state
            self.x_reserve = final_x_reserve
            self.y_reserve = final_y_reserve
            
        elif mispricing < no_trade_lower:
            # AMM price is too low, arbs buy X from AMM and sell to external
            # They correct until the price is at the lower boundary
            boundary_price = external_price / (1 + self.current_fee)
            
            # Calculate reserves after external arb
            new_x_reserve = self.x_reserve
            new_y_reserve = boundary_price * new_x_reserve
            
            # Calculate how much value leaked to external arbitrageurs
            value_before = self.y_reserve + external_price * self.x_reserve
            value_after = new_y_reserve + external_price * new_x_reserve
            arb_excess = value_before - value_after
            
            # Now manager corrects the rest of the way with zero fee
            # They trade until the AMM price equals the external price
            final_x_reserve = np.sqrt(self.k / external_price)
            final_y_reserve = external_price * final_x_reserve
            
            # Calculate manager's profit
            manager_value_before = new_y_reserve + external_price * new_x_reserve
            manager_value_after = final_y_reserve + external_price * final_x_reserve
            manager_profit = manager_value_before - manager_value_after
            
            # Update reserves to final state
            self.x_reserve = final_x_reserve
            self.y_reserve = final_y_reserve
            
        self.total_arb_profits += (manager_profit + arb_excess)
        return manager_profit, arb_excess, 0

class Manager:
    """Pool manager for auction-managed AMM"""
    
    def __init__(self, name, fee_strategy="optimal"):
        """
        Initialize a pool manager
        name: name of the manager
        fee_strategy: strategy for setting fees ("optimal", "max", "min", or "dynamic")
        """
        self.name = name
        self.fee_strategy = fee_strategy
        self.profit = 0
        self.rent_paid = 0
        self.fees_collected = 0
        self.arb_profits = 0
        
        # For fee optimization
        self.fee_history = []
        self.volume_history = []
        self.last_optimization = 0
        
    def calculate_bid(self, amm, price_volatility, volume_estimate, optimization_period=10):
        """
        Calculate the bid amount for the auction
        amm: the AMM pool
        price_volatility: volatility of the external price
        volume_estimate: estimated trading volume
        optimization_period: how often to recalculate optimal fee
        
        Returns the bid amount (rent per block)
        """
        # Estimate value that can be extracted from arbitrage
        arb_value = self._estimate_arb_value(amm, price_volatility)
        
        # Estimate value from fees
        fee_value = self._estimate_fee_value(amm, volume_estimate)
        
        # Combine for total expected value
        total_value = arb_value + fee_value
        
        # Bid slightly less than total value to ensure profit
        bid = total_value * 0.9
        
        return max(0, bid)
    
    def _estimate_arb_value(self, amm, price_volatility):
        """Estimate value from arbitrage opportunities"""
        # Simplified model: value ~ pool_value * volatility^2
        # This is based on the formula in the paper's Example 2
        pool_value = amm.get_value(amm.spot_price())
        return pool_value * price_volatility**2 / 8
    
    def _estimate_fee_value(self, amm, volume_estimate):
        """Estimate value from fees collected"""
        # Simplified: expected fee revenue = volume * optimal_fee
        optimal_fee = self.determine_optimal_fee(amm, volume_estimate)
        return volume_estimate * optimal_fee
    
    def determine_optimal_fee(self, amm, volume_estimate):
        """Determine the optimal fee based on strategy"""
        if self.fee_strategy == "max":
            return amm.max_fee
        elif self.fee_strategy == "min":
            return 0.0001  # Very small fee
        elif self.fee_strategy == "dynamic":
            # Dynamic strategy based on recent market conditions
            if len(self.fee_history) > 10:
                # Use recent fee and volume data to optimize
                volumes = np.array(self.volume_history[-10:])
                fees = np.array(self.fee_history[-10:])
                
                # Find fee with highest revenue
                revenues = volumes * fees
                best_idx = np.argmax(revenues)
                return fees[best_idx]
            else:
                # Not enough history, use mid-range fee
                return amm.max_fee / 2
        else:  # "optimal"
            # Implement a simple demand model for noise traders
            # Assuming demand = c * exp(-price_elasticity * fee)
            # Optimal fee = 1/price_elasticity
            
            # We'll use a fixed price elasticity for simplicity
            price_elasticity = 200  # Higher elasticity -> lower optimal fee
            return min(1/price_elasticity, amm.max_fee)
    
    def set_fee_for_block(self, amm, external_price, block_number, volume_estimate):
        """
        Set the fee for the current block
        amm: the AMM pool
        external_price: current external price
        block_number: current block number
        volume_estimate: estimated trading volume
        """
        # Only recalculate fee periodically to avoid excessive computation
        if block_number - self.last_optimization >= 10:
            optimal_fee = self.determine_optimal_fee(amm, volume_estimate)
            amm.set_fee(optimal_fee)
            self.last_optimization = block_number
            
            # Record for optimization
            self.fee_history.append(optimal_fee)
            self.volume_history.append(volume_estimate)
    
    def handle_block(self, amm, external_price, block_number, volume_estimate):
        """
        Handle operations for a block
        amm: the AMM pool
        external_price: current external price
        block_number: current block number
        volume_estimate: estimated trading volume
        
        Returns manager's profit for the block
        """
        # Set fee for the block
        self.set_fee_for_block(amm, external_price, block_number, volume_estimate)
        
        # Capture arbitrage opportunities
        manager_profit, arb_excess, manager_fees = amm.handle_arbitrage(external_price)
        
        # Process noise trader swaps and collect fees
        if volume_estimate > 0:
            # Simulate noise traders (random buys and sells)
            buy_volume = volume_estimate * random.uniform(0.4, 0.6)
            sell_volume = volume_estimate - buy_volume
            
            # Buy X with Y
            output_x, fee_y = amm.execute_swap(buy_volume, False)
            # Sell X for Y
            output_y, fee_x = amm.execute_swap(sell_volume / external_price, True)
            
            # Convert fee_x to Y-terms using external price
            manager_fees = fee_y + fee_x * external_price
        
        # Pay rent
        self.rent_paid += amm.rent
        
        # Update profit
        block_profit = manager_profit + manager_fees - amm.rent
        self.profit += block_profit
        self.arb_profits += manager_profit
        self.fees_collected += manager_fees
        
        return block_profit

class MarketSimulator:
    """Simulator for testing AMM strategies"""
    
    def __init__(self, initial_price=100, volatility=0.01, mean_block_time=12):
        """
        Initialize a market simulator
        initial_price: starting price of the risky asset
        volatility: price volatility per time unit
        mean_block_time: average time between blocks
        """
        self.current_price = initial_price
        self.volatility = volatility
        self.mean_block_time = mean_block_time
        self.time = 0
        self.block_number = 0
        
        # Track price history
        self.price_history = [initial_price]
        self.time_history = [0]
        
        # Stats for noise traders
        self.base_volume = 10000  # Base daily volume
        self.price_elasticity = 200  # Elasticity of volume to fees
        
    def generate_next_block(self):
        """Generate the next block"""
        # Time until next block (exponential distribution)
        block_time = np.random.exponential(self.mean_block_time)
        self.time += block_time
        self.block_number += 1
        
        # Update price (geometric Brownian motion)
        drift = 0  # Assuming zero drift for simplicity
        diffusion = self.volatility * np.sqrt(block_time) * np.random.normal()
        self.current_price *= np.exp(drift + diffusion)
        
        # Record price
        self.price_history.append(self.current_price)
        self.time_history.append(self.time)
        
        return block_time
    
    def estimate_volume(self, fee):
        """Estimate trading volume based on fee"""
        # Model: volume = base_volume * exp(-price_elasticity * fee)
        daily_volume = self.base_volume * np.exp(-self.price_elasticity * fee)
        # Convert to per-block volume
        block_volume = daily_volume * self.mean_block_time / (60 * 60 * 24)
        return block_volume
    
    def run_auction(self, amm, managers, blocks=1000):
        """
        Run an auction for pool manager
        amm: the AMM pool
        managers: list of Manager objects
        blocks: number of blocks to simulate
        """
        # Initialize variables
        current_manager = None
        manager_profits = {manager.name: [] for manager in managers}
        lp_profits = []
        
        # Initial liquidity value
        initial_value = amm.get_value(self.current_price)
        
        # Simulate blocks
        for block in tqdm(range(blocks)):
            # Generate next block
            block_time = self.generate_next_block()
            
            # Run auction for pool manager (every 10 blocks for simplicity)
            if block % 10 == 0 or current_manager is None:
                # Get bids from all managers
                bids = []
                for manager in managers:
                    # Volume estimate for the fee they would set
                    optimal_fee = manager.determine_optimal_fee(amm, 0)
                    volume_estimate = self.estimate_volume(optimal_fee)
                    
                    # Calculate bid
                    bid_amount = manager.calculate_bid(amm, self.volatility, volume_estimate)
                    bids.append((manager, bid_amount))
                
                # Select highest bidder
                highest_bid = max(bids, key=lambda x: x[1])
                current_manager, rent = highest_bid
                
                # Set manager and rent
                amm.set_manager(current_manager, rent)
            
            # Manager handles the block
            if current_manager:
                # Estimate volume based on current fee
                volume_estimate = self.estimate_volume(amm.current_fee)
                
                # Manager handles block
                profit = current_manager.handle_block(amm, self.current_price, 
                                                     self.block_number, volume_estimate)
                
                # Record profits
                manager_profits[current_manager.name].append(profit)
            
            # Record LP profits (change in value + rent - initial value)
            current_value = amm.get_value(self.current_price)
            lp_profit = current_value - initial_value + (amm.rent * block)
            lp_profits.append(lp_profit)
            
            # Record state
            amm.record_state(self.time, self.current_price)
        
        return manager_profits, lp_profits
    
    def run_fixed_fee_amm(self, amm, blocks=1000):
        """
        Simulate a fixed-fee AMM
        amm: the fixed-fee AMM pool
        blocks: number of blocks to simulate
        """
        # Initial liquidity value
        initial_value = amm.get_value(self.current_price)
        lp_profits = []
        fees_collected = []
        arb_profits = []
        
        # Simulate blocks
        for block in tqdm(range(blocks)):
            # Generate next block
            block_time = self.generate_next_block()
            
            # Handle arbitrage
            arb_profit, fee = amm.handle_arbitrage(self.current_price)
            arb_profits.append(arb_profit)
            
            # Process noise trader swaps
            volume_estimate = self.estimate_volume(amm.fee)
            
            if volume_estimate > 0:
                # Simulate noise traders (random buys and sells)
                buy_volume = volume_estimate * random.uniform(0.4, 0.6)
                sell_volume = volume_estimate - buy_volume
                
                # Buy X with Y
                output_x, fee_y = amm.swap(buy_volume, False, amm.fee)
                # Sell X for Y
                output_y, fee_x = amm.swap(sell_volume / self.current_price, True, amm.fee)
                
                # Convert fee_x to Y-terms using external price
                fees = fee_y + fee_x * self.current_price
                fees_collected.append(fees)
            else:
                fees_collected.append(0)
            
            # Record LP profits (change in value + fees - arb losses)
            current_value = amm.get_value(self.current_price)
            lp_profit = current_value - initial_value + sum(fees_collected) - sum(arb_profits)
            lp_profits.append(lp_profit)
            
            # Record state
            amm.record_state(self.time, self.current_price, 
                            fees_collected=fees_collected[-1], 
                            arb_profits=arb_profits[-1], 
                            lp_profits=lp_profit)
        
        return lp_profits, fees_collected, arb_profits

def run_simulation():
    """Run the full simulation comparing fixed-fee and auction-managed AMMs"""
    print("Initializing simulation...")
    
    # Initialize simulation parameters
    initial_price = 100
    initial_liquidity = 10000  # sqrt(x*y)
    volatility = 0.02
    blocks = 1000
    
    # Calculate initial reserves
    initial_x = np.sqrt(initial_liquidity**2 / initial_price)
    initial_y = initial_x * initial_price
    
    # Initialize AMMs
    ff_amm_low = FixedFeeAMM(initial_x, initial_y, fee=0.001, name="Fixed-fee AMM (0.1%)")
    ff_amm_med = FixedFeeAMM(initial_x, initial_y, fee=0.003, name="Fixed-fee AMM (0.3%)")
    ff_amm_high = FixedFeeAMM(initial_x, initial_y, fee=0.01, name="Fixed-fee AMM (1%)")
    
    am_amm = AuctionManagedAMM(initial_x, initial_y, max_fee=0.01, name="Auction-managed AMM")
    
    # Initialize market simulator
    simulator = MarketSimulator(initial_price=initial_price, volatility=volatility)
    
    # Initialize managers with different strategies
    managers = [
        Manager("Optimal Fee Manager", fee_strategy="optimal"),
        Manager("Max Fee Manager", fee_strategy="max"),
        Manager("Min Fee Manager", fee_strategy="min"),
        Manager("Dynamic Fee Manager", fee_strategy="dynamic")
    ]
    
    # Run simulations
    print("Running fixed-fee AMM (0.1%) simulation...")
    ff_low_profits, ff_low_fees, ff_low_arbs = simulator.run_fixed_fee_amm(ff_amm_low, blocks)
    
    # Reset simulator
    simulator = MarketSimulator(initial_price=initial_price, volatility=volatility)
    print("Running fixed-fee AMM (0.3%) simulation...")
    ff_med_profits, ff_med_fees, ff_med_arbs = simulator.run_fixed_fee_amm(ff_amm_med, blocks)
    
    # Reset simulator
    simulator = MarketSimulator(initial_price=initial_price, volatility=volatility)
    print("Running fixed-fee AMM (1%) simulation...")
    ff_high_profits, ff_high_fees, ff_high_arbs = simulator.run_fixed_fee_amm(ff_amm_high, blocks)
    
    # Reset simulator
    simulator = MarketSimulator(initial_price=initial_price, volatility=volatility)
    print("Running auction-managed AMM simulation...")
    am_manager_profits, am_lp_profits = simulator.run_auction(am_amm, managers, blocks)
    
    # Analyze results
    print("\nAnalysis of Results:")
    print("-------------------")
    
    # Fixed-fee AMM results
    print(f"Fixed-fee AMM (0.1%):")
    print(f"  Final LP profit: {ff_low_profits[-1]:.2f}")
    print(f"  Total fees collected: {sum(ff_low_fees):.2f}")
    print(f"  Total arb profits: {sum(ff_low_arbs):.2f}")
    
    print(f"Fixed-fee AMM (0.3%):")
    print(f"  Final LP profit: {ff_med_profits[-1]:.2f}")
    print(f"  Total fees collected: {sum(ff_med_fees):.2f}")
    print(f"  Total arb profits: {sum(ff_med_arbs):.2f}")
    
    print(f"Fixed-fee AMM (1%):")
    print(f"  Final LP profit: {ff_high_profits[-1]:.2f}")
    print(f"  Total fees collected: {sum(ff_high_fees):.2f}")
    print(f"  Total arb profits: {sum(ff_high_arbs):.2f}")
    
    # Auction-managed AMM results
    print(f"Auction-managed AMM:")
    print(f"  Final LP profit: {am_lp_profits[-1]:.2f}")
    
    for manager_name, profits in am_manager_profits.items():
        if profits:
            print(f"  {manager_name} final profit: {sum(profits):.2f}")
    
    # Find the most profitable manager
    best_manager = max(managers, key=lambda m: m.profit)
    print(f"  Best manager: {best_manager.name}")
    print(f"    Total profit: {best_manager.profit:.2f}")
    print(f"    Fees collected: {best_manager.fees_collected:.2f}")
    print(f"    Arb profits: {best_manager.arb_profits:.2f}")
    print(f"    Rent paid: {best_manager.rent_paid:.2f}")
    
    # Plot results
    plot_results(
        ff_amm_low, ff_amm_med, ff_amm_high, am_amm,
        ff_low_profits, ff_med_profits, ff_high_profits, am_lp_profits,
        simulator, am_manager_profits
    )
    
    return {
        'ff_low': ff_amm_low,
        'ff_med': ff_amm_med,
        'ff_high': ff_amm_high,
        'am_amm': am_amm,
        'simulator': simulator,
        'managers': managers,
        'ff_low_profits': ff_low_profits,
        'ff_med_profits': ff_med_profits,
        'ff_high_profits': ff_high_profits,
        'am_lp_profits': am_lp_profits,
        'am_manager_profits': am_manager_profits
    }

def plot_results(ff_amm_low, ff_amm_med, ff_amm_high, am_amm, 
                ff_low_profits, ff_med_profits, ff_high_profits, am_lp_profits,
                simulator, am_manager_profits):
    """Plot the simulation results"""
    # Convert to DataFrames for easier plotting
    ff_low_df = ff_amm_low.get_history_df()
    ff_med_df = ff_amm_med.get_history_df()
    ff_high_df = ff_amm_high.get_history_df()
    am_df = am_amm.get_history_df()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: External price
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(simulator.time_history, simulator.price_history, label='External Price')
    ax1.set_title('External Price Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: LP profits
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(ff_low_profits, label=ff_amm_low.name)
    ax2.plot(ff_med_profits, label=ff_amm_med.name)
    ax2.plot(ff_high_profits, label=ff_amm_high.name)
    ax2.plot(am_lp_profits, label=am_amm.name)
    ax2.set_title('LP Profits Over Time')
    ax2.set_xlabel('Block')
    ax2.set_ylabel('Profit')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Mispricing
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(ff_low_df['time'], ff_low_df['mispricing'], label=ff_amm_low.name)
    ax3.plot(ff_med_df['time'], ff_med_df['mispricing'], label=ff_amm_med.name)
    ax3.plot(ff_high_df['time'], ff_high_df['mispricing'], label=ff_amm_high.name)
    ax3.plot(am_df['time'], am_df['mispricing'], label=am_amm.name)
    ax3.set_title('Mispricing Over Time (log scale)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Log Mispricing')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Fees collected and arb profits for fixed-fee AMMs
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(np.cumsum(ff_low_df['fees_collected']), label=f'{ff_amm_low.name} Fees')
    ax4.plot(np.cumsum(ff_med_df['fees_collected']), label=f'{ff_amm_med.name} Fees')
    ax4.plot(np.cumsum(ff_high_df['fees_collected']), label=f'{ff_amm_high.name} Fees')
    ax4.plot(np.cumsum(ff_low_df['arb_profits']), label=f'{ff_amm_low.name} Arb', linestyle='--')
    ax4.plot(np.cumsum(ff_med_df['arb_profits']), label=f'{ff_amm_med.name} Arb', linestyle='--')
    ax4.plot(np.cumsum(ff_high_df['arb_profits']), label=f'{ff_amm_high.name} Arb', linestyle='--')
    ax4.set_title('Cumulative Fees and Arb Profits')
    ax4.set_xlabel('Block')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True)
    
    # Plot 5: Manager profits
    ax5 = fig.add_subplot(3, 2, 5)
    for manager_name, profits in am_manager_profits.items():
        if profits:
            ax5.plot(np.cumsum(profits), label=f'{manager_name}')
    ax5.set_title('Cumulative Manager Profits')
    ax5.set_xlabel('Block')
    ax5.set_ylabel('Profit')
    ax5.legend()
    ax5.grid(True)
    
    # Plot 6: Liquidity comparison
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(ff_low_df['time'], ff_low_df['liquidity'], label=ff_amm_low.name)
    ax6.plot(ff_med_df['time'], ff_med_df['liquidity'], label=ff_amm_med.name)
    ax6.plot(ff_high_df['time'], ff_high_df['liquidity'], label=ff_amm_high.name)
    ax6.plot(am_df['time'], am_df['liquidity'], label=am_amm.name)
    ax6.set_title('Liquidity Over Time')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Liquidity')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('amm_simulation_results.png')
    plt.show()

# Run the simulation
results = run_simulation()