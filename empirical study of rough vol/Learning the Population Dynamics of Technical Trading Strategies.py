import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TechnicalTradingStrategy:
    """
    Class implementing the expert-based online learning algorithm for technical trading
    strategies as described in Murphy and Gebbie's paper.
    """
    
    def __init__(self, n1_params=[4, 8, 16, 32], n2_params=[8, 16, 32, 64], 
                 transaction_costs=True, cost_per_day=0.001):
        """
        Initialize the strategy with parameters for expert generation.
        
        Parameters:
        -----------
        n1_params : list
            Short-term look-back parameters
        n2_params : list
            Long-term look-back parameters
        transaction_costs: bool
            Whether to include transaction costs
        cost_per_day: float
            Transaction cost per day (in basis points)
        """
        self.n1_params = n1_params
        self.n2_params = n2_params
        self.transaction_costs = transaction_costs
        self.cost_per_day = cost_per_day
        
        # Initialize strategy lists and dictionaries
        self.strategies = []
        self.strategy_info = {}
        
        # Initialize strategy results
        self.expert_weights = None
        self.expert_wealth = None
        self.portfolio_weights = None
        self.portfolio_wealth = None
        self.pnl = None
        
        # Add technical trading strategies
        self._add_strategies()
        
    def _add_strategies(self):
        """Add all technical trading strategies to the strategy list"""
        
        # Simple strategies with one parameter
        self.strategies.extend([
            ('RSI', self._rsi_rule, 1), 
            ('PROC', self._proc_rule, 1), 
            ('MOM', self._momentum_rule, 1),
            ('ACC', self._acceleration_rule, 1),
            ('BOLL', self._bollinger_rule, 1),
            ('Williams %R', self._williams_r_rule, 1),
            ('SAR', self._sar_rule, 1)
        ])
        
        # Strategies with two parameters
        self.strategies.extend([
            ('EMA X-over', self._ema_crossover_rule, 2),
            ('Moving Ave X-over', self._ma_crossover_rule, 2),
            ('MACD', self._macd_rule, 2),
            ('Ichimoku Kijun Sen', self._ichimoku_rule, 2),
            ('Fast Stochastic', self._fast_stochastic_rule, 2),
            ('Slow Stochastic', self._slow_stochastic_rule, 2),
            ('MARSI', self._marsi_rule, 2)
        ])
        
        # Portfolio algorithms adapted for zero-cost portfolios
        self.strategies.extend([
            ('Online Z-BCRP', self._zero_cost_bcrp, 1),
            ('Online Anti-Z-BCRP', self._zero_cost_anti_bcrp, 1),
            ('Online Z-Anticor', self._zero_cost_anticor, 1)
        ])
    
    def generate_experts(self, universe_size):
        """
        Generate the full set of experts based on strategies, parameters, and clusters.
        
        Parameters:
        -----------
        universe_size : int
            Number of stocks in the trading universe
        
        Returns:
        --------
        expert_count : int
            Total number of experts generated
        """
        # We'll consider 4 object clusters as specified in the paper:
        # 1. All stocks (trivial cluster)
        # 2-4. Three major sector clusters (Resources, Industrials, Financials)
        # For simplicity in simulation, we'll divide stocks equally into the 3 sectors
        
        self.clusters = {
            'All': list(range(universe_size)),
            'Resources': list(range(universe_size//3)),
            'Industrials': list(range(universe_size//3, 2*universe_size//3)),
            'Financials': list(range(2*universe_size//3, universe_size))
        }
        
        # Calculate the total number of experts
        expert_count = 0
        for strategy_name, _, param_count in self.strategies:
            if param_count == 1:
                # For strategies with one parameter
                expert_count += len(self.clusters) * len(self.n1_params)
            else:
                # For strategies with two parameters
                for n1 in self.n1_params:
                    for n2 in self.n2_params:
                        if n1 < n2:  # Only include when n1 < n2
                            expert_count += len(self.clusters)
        
        self.expert_count = expert_count
        self.universe_size = universe_size
        
        print(f"Generated {expert_count} experts from {len(self.strategies)} strategies")
        
        return expert_count

    def initialize_wealth(self, T):
        """
        Initialize the wealth arrays for experts and portfolio
        
        Parameters:
        -----------
        T : int
            Number of time periods
        """
        self.expert_wealth = np.ones((self.expert_count, T))
        self.expert_weights = np.zeros((self.expert_count, self.universe_size + 1, T))
        self.portfolio_wealth = np.ones(T)
        self.portfolio_weights = np.zeros((self.universe_size + 1, T))
        self.pnl = np.zeros(T)

    def run_strategy(self, price_data, volumes=None):
        """
        Run the trading strategy on the provided price data
        
        Parameters:
        -----------
        price_data : ndarray
            Array of shape (T, universe_size) with closing prices
        volumes : ndarray, optional
            Array of shape (T, universe_size) with trading volumes
            
        Returns:
        --------
        portfolio_wealth : ndarray
            Cumulative wealth of the portfolio over time
        pnl : ndarray
            Profit and loss (returns) over time
        """
        T, universe_size = price_data.shape
        assert universe_size == self.universe_size, "Price data dimensions must match universe size"
        
        # Initialize wealth arrays
        self.initialize_wealth(T)
        
        # Calculate returns
        returns = np.zeros_like(price_data)
        returns[1:] = price_data[1:] / price_data[:-1] - 1
        
        # For each time period
        for t in tqdm(range(1, T), desc="Running strategy"):
            # Skip very early periods where we don't have enough data for lookback
            if t < max(self.n2_params):
                continue
                
            # 1. Generate expert signals and transform to weights
            expert_idx = 0
            for strategy_name, strategy_func, param_count in self.strategies:
                for cluster_name, cluster_stocks in self.clusters.items():
                    if param_count == 1:
                        for n1 in self.n1_params:
                            if t >= n1:  # Check if enough data is available
                                # Generate signals for this expert
                                signals = np.zeros(universe_size)
                                for stock_idx in cluster_stocks:
                                    # Call strategy function to get signal for this stock
                                    signal = strategy_func(price_data[:t+1, stock_idx], n1)
                                    signals[stock_idx] = signal
                                
                                # Transform signals to weights
                                weights = self._transform_signals_to_weights(signals, price_data[:t+1])
                                
                                # Store weights
                                self.expert_weights[expert_idx, :, t] = weights
                                
                                # Update expert wealth
                                stock_returns = returns[t]
                                exp_return = np.sum(weights[:-1] * stock_returns)
                                self.expert_wealth[expert_idx, t] = self.expert_wealth[expert_idx, t-1] * (1 + exp_return)
                                
                            expert_idx += 1
                    else:  # param_count == 2
                        for n1 in self.n1_params:
                            for n2 in self.n2_params:
                                if n1 < n2 and t >= n2:  # Check if n1 < n2 and enough data is available
                                    # Generate signals for this expert
                                    signals = np.zeros(universe_size)
                                    for stock_idx in cluster_stocks:
                                        # Call strategy function to get signal for this stock
                                        signal = strategy_func(price_data[:t+1, stock_idx], n1, n2)
                                        signals[stock_idx] = signal
                                    
                                    # Transform signals to weights
                                    weights = self._transform_signals_to_weights(signals, price_data[:t+1])
                                    
                                    # Store weights
                                    self.expert_weights[expert_idx, :, t] = weights
                                    
                                    # Update expert wealth
                                    stock_returns = returns[t]
                                    exp_return = np.sum(weights[:-1] * stock_returns)
                                    self.expert_wealth[expert_idx, t] = self.expert_wealth[expert_idx, t-1] * (1 + exp_return)
                                    
                                    expert_idx += 1
            
            # 2. Update expert mixtures (weights)
            # Normalize expert wealth to get mixture weights
            expert_mixture = self.expert_wealth[:, t]
            expert_mixture = expert_mixture / np.sum(expert_mixture)
            
            # 3. Calculate portfolio weights as weighted average of expert weights
            for stock_idx in range(self.universe_size + 1):
                self.portfolio_weights[stock_idx, t] = np.sum(expert_mixture * self.expert_weights[:, stock_idx, t])
            
            # Normalize portfolio weights to ensure zero-cost and unit leverage
            self._normalize_portfolio_weights(t)
            
            # 4. Calculate portfolio return and wealth
            stock_returns = returns[t]
            portfolio_return = np.sum(self.portfolio_weights[:-1, t] * stock_returns)
            
            # Apply transaction costs if enabled
            if self.transaction_costs and t > 1:
                turnover = np.sum(np.abs(self.portfolio_weights[:-1, t] - self.portfolio_weights[:-1, t-1]))
                cost = turnover * self.cost_per_day
                portfolio_return -= cost
            
            self.portfolio_wealth[t] = self.portfolio_wealth[t-1] * (1 + portfolio_return)
            self.pnl[t] = portfolio_return
        
        return self.portfolio_wealth, self.pnl
    
    def _normalize_portfolio_weights(self, t):
        """
        Normalize portfolio weights to ensure zero-cost and unit leverage
        
        Parameters:
        -----------
        t : int
            Current time period
        """
        # Ensure zero-cost: sum of weights = 0
        offset = np.sum(self.portfolio_weights[:-1, t]) / (self.universe_size + 1)
        self.portfolio_weights[:, t] -= offset
        
        # Ensure unit leverage: sum of absolute weights = 1
        leverage = np.sum(np.abs(self.portfolio_weights[:, t]))
        if leverage > 0:
            self.portfolio_weights[:, t] /= leverage
    
    def _transform_signals_to_weights(self, signals, price_data):
        """
        Transform signals (-1, 0, 1) into portfolio weights as described in the paper.
        
        Parameters:
        -----------
        signals : ndarray
            Array of trading signals (-1, 0, 1) for each stock
        price_data : ndarray
            Historical price data up to current time
            
        Returns:
        --------
        weights : ndarray
            Portfolio weights including the risk-free asset
        """
        n_stocks = len(signals)
        weights = np.zeros(n_stocks + 1)  # +1 for risk-free asset
        
        # Case 1: All signals are hold (0)
        if np.all(signals == 0):
            return weights
        
        # Calculate 90-day volatility for weighting
        lookback = min(90, price_data.shape[0] - 1)
        returns = price_data[-lookback:] / price_data[-lookback-1:-1] - 1
        volatility = np.std(returns, axis=0)
        
        # Case 2: All signals are non-negative (0 or 1)
        if np.all(signals >= 0):
            # Get stocks with buy signals
            buy_indices = np.where(signals > 0)[0]
            if len(buy_indices) > 0:
                # Weight proportional to volatility
                vol_buy = volatility[buy_indices]
                vol_sum = np.sum(vol_buy)
                if vol_sum > 0:
                    for i, idx in enumerate(buy_indices):
                        weights[idx] = 0.5 * vol_buy[i] / vol_sum
                else:
                    # Equal weight if volatility is zero
                    for idx in buy_indices:
                        weights[idx] = 0.5 / len(buy_indices)
                
                # Short the risk-free asset
                weights[-1] = -0.5
            
        # Case 3: All signals are non-positive (0 or -1)
        elif np.all(signals <= 0):
            # Get stocks with sell signals
            sell_indices = np.where(signals < 0)[0]
            if len(sell_indices) > 0:
                # Weight proportional to volatility
                vol_sell = volatility[sell_indices]
                vol_sum = np.sum(vol_sell)
                if vol_sum > 0:
                    for i, idx in enumerate(sell_indices):
                        weights[idx] = -0.5 * vol_sell[i] / vol_sum
                else:
                    # Equal weight if volatility is zero
                    for idx in sell_indices:
                        weights[idx] = -0.5 / len(sell_indices)
                
                # Long the risk-free asset
                weights[-1] = 0.5
                
        # Case 4: Combination of buy, sell, and hold
        else:
            # Get stocks with buy and sell signals
            buy_indices = np.where(signals > 0)[0]
            sell_indices = np.where(signals < 0)[0]
            
            # Process buy signals
            if len(buy_indices) > 0:
                vol_buy = volatility[buy_indices]
                vol_sum = np.sum(vol_buy)
                if vol_sum > 0:
                    for i, idx in enumerate(buy_indices):
                        weights[idx] = 0.5 * vol_buy[i] / vol_sum
                else:
                    for idx in buy_indices:
                        weights[idx] = 0.5 / len(buy_indices)
            
            # Process sell signals
            if len(sell_indices) > 0:
                vol_sell = volatility[sell_indices]
                vol_sum = np.sum(vol_sell)
                if vol_sum > 0:
                    for i, idx in enumerate(sell_indices):
                        weights[idx] = -0.5 * vol_sell[i] / vol_sum
                else:
                    for idx in sell_indices:
                        weights[idx] = -0.5 / len(sell_indices)
            
            # Adjust risk-free asset to make the portfolio self-financing
            weights[-1] = -np.sum(weights[:-1])
        
        return weights
    
    # Technical trading strategy implementations
    def _rsi_rule(self, prices, lookback):
        """RSI trading rule"""
        # Need at least lookback+1 data points
        if len(prices) <= lookback+1:
            return 0
            
        # Calculate returns
        returns = prices[1:] / prices[:-1] - 1
        
        # Separate up and down moves
        up_moves = np.zeros_like(returns)
        down_moves = np.zeros_like(returns)
        
        up_moves[returns > 0] = returns[returns > 0]
        down_moves[returns < 0] = -returns[returns < 0]  # Convert to positive values
        
        # Calculate RSI
        avg_up = np.mean(up_moves[-lookback:])
        avg_down = np.mean(down_moves[-lookback:])
        
        if avg_down == 0:
            rsi = 100
        else:
            rs = avg_up / avg_down
            rsi = 100 - (100 / (1 + rs))
        
        # Get current and previous RSI
        curr_rsi = rsi
        
        # Generate trading signal
        if curr_rsi < 30:
            return 1  # Buy signal
        elif curr_rsi > 70:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _proc_rule(self, prices, lookback):
        """Price Rate of Change trading rule"""
        if len(prices) <= lookback:
            return 0
            
        # Calculate Price Rate of Change
        proc = 100 * (prices[-1] - prices[-lookback-1]) / prices[-lookback-1]
        
        # Generate trading signal
        if proc > 0:
            return 1  # Buy signal
        elif proc < 0:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _momentum_rule(self, prices, lookback):
        """Momentum trading rule"""
        if len(prices) <= lookback:
            return 0
            
        # Calculate momentum
        momentum = prices[-1] - prices[-lookback-1]
        
        # Calculate EMA of momentum
        ema_period = max(2, lookback // 3)
        weights = np.exp(np.linspace(-1., 0., ema_period))
        weights /= weights.sum()
        
        # Calculate previous momentum values
        momentum_values = np.zeros(ema_period)
        for i in range(ema_period):
            if len(prices) > lookback + i + 1:
                momentum_values[i] = prices[-i-2] - prices[-i-lookback-2]
        
        # Calculate EMA of momentum
        ema_momentum = np.sum(weights * momentum_values)
        
        # Generate trading signal
        if momentum > ema_momentum:
            return 1  # Buy signal
        elif momentum < ema_momentum:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _acceleration_rule(self, prices, lookback):
        """Acceleration trading rule"""
        if len(prices) <= lookback + 1:
            return 0
            
        # Calculate current and previous momentum
        current_momentum = prices[-1] - prices[-lookback-1]
        prev_momentum = prices[-2] - prices[-lookback-2]
        
        # Calculate acceleration
        acceleration = current_momentum - prev_momentum
        
        # Generate trading signal
        if acceleration > 0:
            return 1  # Buy signal
        elif acceleration < 0:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _bollinger_rule(self, prices, lookback):
        """Bollinger Bands trading rule"""
        if len(prices) <= lookback:
            return 0
            
        # Calculate SMA
        sma = np.mean(prices[-lookback:])
        
        # Calculate standard deviation
        std = np.std(prices[-lookback:])
        
        # Calculate Bollinger Bands
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        # Generate trading signal
        if prices[-1] < lower_band:
            return 1  # Buy signal
        elif prices[-1] > upper_band:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _williams_r_rule(self, prices, lookback):
        """Williams %R trading rule"""
        if len(prices) <= lookback:
            return 0
            
        # Calculate highest high and lowest low
        highest_high = np.max(prices[-lookback:])
        lowest_low = np.min(prices[-lookback:])
        
        # Calculate Williams %R
        if highest_high - lowest_low == 0:
            williams_r = -50  # Neutral when no range
        else:
            williams_r = -100 * (highest_high - prices[-1]) / (highest_high - lowest_low)
        
        # Generate trading signal
        if williams_r > -20:
            return -1  # Sell signal (overbought)
        elif williams_r < -80:
            return 1  # Buy signal (oversold)
        else:
            return 0  # Hold
    
    def _sar_rule(self, prices, lookback):
        """Parabolic SAR trading rule"""
        if len(prices) <= lookback:
            return 0
            
        # Simple implementation of Parabolic SAR
        # In practice, this would be more complex with AF and EP
        
        # Use a simple trend detection for this simplified implementation
        trend = np.mean(prices[-lookback//2:]) > np.mean(prices[-lookback:-lookback//2])
        
        if trend:
            return 1  # Buy signal
        else:
            return -1  # Sell signal
    
    def _ema_crossover_rule(self, prices, short_lookback, long_lookback):
        """Exponential Moving Average Crossover rule"""
        if len(prices) <= long_lookback:
            return 0
            
        # Calculate EMAs
        alpha_short = 2 / (short_lookback + 1)
        alpha_long = 2 / (long_lookback + 1)
        
        ema_short = np.zeros(2)
        ema_long = np.zeros(2)
        
        # Initialize with SMA
        ema_short[0] = np.mean(prices[-long_lookback-1:-long_lookback+short_lookback-1])
        ema_long[0] = np.mean(prices[-long_lookback-1:-1])
        
        # Calculate current EMAs
        for i in range(-long_lookback, 0):
            if i == -long_lookback:
                continue
                
            ema_short[1] = prices[i] * alpha_short + ema_short[0] * (1 - alpha_short)
            ema_long[1] = prices[i] * alpha_long + ema_long[0] * (1 - alpha_long)
            
            ema_short[0] = ema_short[1]
            ema_long[0] = ema_long[1]
        
        # Generate trading signal
        if ema_short[1] > ema_long[1]:
            return 1  # Buy signal
        elif ema_short[1] < ema_long[1]:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _ma_crossover_rule(self, prices, short_lookback, long_lookback):
        """Moving Average Crossover rule"""
        if len(prices) <= long_lookback:
            return 0
            
        # Calculate SMAs
        sma_short = np.mean(prices[-short_lookback:])
        sma_long = np.mean(prices[-long_lookback:])
        
        # Generate trading signal
        if sma_short > sma_long:
            return 1  # Buy signal
        elif sma_short < sma_long:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _macd_rule(self, prices, short_lookback, long_lookback):
        """MACD trading rule"""
        if len(prices) <= long_lookback:
            return 0
            
        # Calculate EMAs
        alpha_short = 2 / (short_lookback + 1)
        alpha_long = 2 / (long_lookback + 1)
        
        # Calculate short EMA
        short_ema = prices[-long_lookback]
        for i in range(-long_lookback+1, 0):
            short_ema = prices[i] * alpha_short + short_ema * (1 - alpha_short)
        
        # Calculate long EMA
        long_ema = prices[-long_lookback]
        for i in range(-long_lookback+1, 0):
            long_ema = prices[i] * alpha_long + long_ema * (1 - alpha_long)
        
        # Calculate MACD line
        macd_line = short_ema - long_ema
        
        # Calculate Signal line (9-period EMA of MACD)
        signal_period = 9
        alpha_signal = 2 / (signal_period + 1)
        
        # For simplicity, calculate previous MACD values
        prev_macd_values = np.zeros(signal_period)
        for i in range(signal_period):
            if len(prices) > long_lookback + i + 1:
                short_ema_prev = prices[-long_lookback-i-1]
                long_ema_prev = prices[-long_lookback-i-1]
                
                for j in range(-long_lookback-i, -i):
                    short_ema_prev = prices[j] * alpha_short + short_ema_prev * (1 - alpha_short)
                    long_ema_prev = prices[j] * alpha_long + long_ema_prev * (1 - alpha_long)
                
                prev_macd_values[i] = short_ema_prev - long_ema_prev
        
        signal_line = np.mean(prev_macd_values)  # Simplified for brevity
        
        # Generate trading signal
        if macd_line > signal_line:
            return 1  # Buy signal
        elif macd_line < signal_line:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _ichimoku_rule(self, prices, short_lookback, long_lookback):
        """Ichimoku Kijun Sen trading rule"""
        if len(prices) <= long_lookback:
            return 0
            
        # Calculate Kijun Sen (Base Line)
        highest_high = np.max(prices[-long_lookback:])
        lowest_low = np.min(prices[-long_lookback:])
        kijun_sen = (highest_high + lowest_low) / 2
        
        # Generate trading signal based on price crossing Kijun Sen
        if prices[-2] <= kijun_sen and prices[-1] > kijun_sen:
            return 1  # Buy signal
        elif prices[-2] >= kijun_sen and prices[-1] < kijun_sen:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _fast_stochastic_rule(self, prices, short_lookback, long_lookback):
        """Fast Stochastic trading rule"""
        if len(prices) <= long_lookback:
            return 0
            
        # Calculate %K
        highest_high = np.max(prices[-short_lookback:])
        lowest_low = np.min(prices[-short_lookback:])
        
        if highest_high == lowest_low:
            percent_k = 50  # Neutral when no range
        else:
            percent_k = 100 * (prices[-1] - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (3-period SMA of %K)
        k_values = np.zeros(3)
        for i in range(3):
            if len(prices) > short_lookback + i + 1:
                hh = np.max(prices[-short_lookback-i-1:-i-1])
                ll = np.min(prices[-short_lookback-i-1:-i-1])
                if hh == ll:
                    k_values[i] = 50
                else:
                    k_values[i] = 100 * (prices[-i-2] - ll) / (hh - ll)
        
        percent_d = np.mean(k_values)
        
        # Generate trading signal
        if percent_k > percent_d:
            return 1  # Buy signal
        elif percent_k < percent_d:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _slow_stochastic_rule(self, prices, short_lookback, long_lookback):
        """Slow Stochastic trading rule"""
        if len(prices) <= long_lookback:
            return 0
            
        # Calculate %K (from Fast Stochastic)
        k_values = np.zeros(3)
        for i in range(3):
            if len(prices) > short_lookback + i + 1:
                hh = np.max(prices[-short_lookback-i-1:-i-1])
                ll = np.min(prices[-short_lookback-i-1:-i-1])
                if hh == ll:
                    k_values[i] = 50
                else:
                    k_values[i] = 100 * (prices[-i-2] - ll) / (hh - ll)
        
        # %K for Slow Stochastic is the %D of Fast Stochastic
        slow_k = np.mean(k_values)
        
        # Calculate %D for Slow Stochastic (3-period SMA of Slow %K)
        # For simplicity, calculate from previous values
        slow_d = slow_k  # Simplified for brevity
        
        # Generate trading signal
        if slow_k > slow_d:
            return 1  # Buy signal
        elif slow_k < slow_d:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _marsi_rule(self, prices, lookback, ma_period):
        """Moving Average RSI trading rule"""
        if len(prices) <= lookback + ma_period:
            return 0
            
        # Calculate RSI values
        rsi_values = np.zeros(ma_period)
        for i in range(ma_period):
            # Calculate subset of prices
            subset = prices[-lookback-ma_period+i:-ma_period+i] if i < ma_period-1 else prices[-lookback:]
            
            # Calculate returns
            subset_returns = subset[1:] / subset[:-1] - 1
            
            # Separate up and down moves
            up_moves = subset_returns.copy()
            up_moves[up_moves < 0] = 0
            
            down_moves = -subset_returns.copy()
            down_moves[down_moves < 0] = 0
            
            # Calculate RSI
            avg_up = np.mean(up_moves)
            avg_down = np.mean(down_moves)
            
            if avg_down == 0:
                rsi_values[i] = 100
            else:
                rs = avg_up / avg_down
                rsi_values[i] = 100 - (100 / (1 + rs))
        
        # Calculate MARSI (MA of RSI)
        marsi = np.mean(rsi_values)
        
        # Generate trading signal
        if marsi < 30:
            return 1  # Buy signal (oversold)
        elif marsi > 70:
            return -1  # Sell signal (overbought)
        else:
            return 0  # Hold
    
    def _zero_cost_bcrp(self, prices, lookback):
        """Zero-Cost Best Constant Rebalanced Portfolio (trend following)"""
        if len(prices) <= lookback:
            return 0
            
        # Calculate returns
        returns = prices[-lookback:] / prices[-lookback-1:-1] - 1
        
        # Calculate mean returns and covariance
        mean_returns = np.mean(returns)
        
        # Generate simple signal based on mean return
        if mean_returns > 0:
            return 1  # Buy signal
        elif mean_returns < 0:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _zero_cost_anti_bcrp(self, prices, lookback):
        """Zero-Cost Anti-BCRP (contrarian)"""
        if len(prices) <= lookback:
            return 0
            
        # Calculate returns
        returns = prices[-lookback:] / prices[-lookback-1:-1] - 1
        
        # Calculate mean returns and covariance
        mean_returns = np.mean(returns)
        
        # Generate opposite signal to BCRP
        if mean_returns < 0:
            return 1  # Buy signal
        elif mean_returns > 0:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def _zero_cost_anticor(self, prices, lookback):
        """Zero-Cost Anti-Correlation (contrarian mean reversion)"""
        if len(prices) <= 2 * lookback:
            return 0
            
        # Calculate returns for two windows
        window1 = prices[-2*lookback:-lookback]
        window2 = prices[-lookback:]
        
        returns1 = window1[1:] / window1[:-1] - 1
        returns2 = window2[1:] / window2[:-1] - 1
        
        # Calculate mean returns
        mean1 = np.mean(returns1)
        mean2 = np.mean(returns2)
        
        # Check if recent returns < previous returns (mean reversion)
        if mean2 < mean1:
            return 1  # Buy signal
        elif mean2 > mean1:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def plot_results(self, title="Technical Trading Strategy Performance"):
        """
        Plot the results of the trading strategy
        
        Parameters:
        -----------
        title : str
            Title for the plot
        """
        if self.portfolio_wealth is None:
            print("Run the strategy first by calling run_strategy()")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot portfolio wealth
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_wealth, label='Portfolio Wealth')
        plt.title(title)
        plt.ylabel('Wealth')
        plt.grid(True)
        plt.legend()
        
        # Plot cumulative PnL
        plt.subplot(2, 1, 2)
        plt.plot(np.cumsum(self.pnl), label='Cumulative P&L')
        plt.xlabel('Time')
        plt.ylabel('Cumulative P&L')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_expert_wealth(self, top_n=5):
        """
        Plot the wealth of the top N experts
        
        Parameters:
        -----------
        top_n : int
            Number of top experts to display
        """
        if self.expert_wealth is None:
            print("Run the strategy first by calling run_strategy()")
            return
        
        final_wealth = self.expert_wealth[:, -1]
        top_experts = np.argsort(final_wealth)[-top_n:]
        
        plt.figure(figsize=(15, 10))
        
        # Plot each top expert's wealth
        for i, expert_idx in enumerate(top_experts):
            plt.plot(self.expert_wealth[expert_idx], label=f'Expert {expert_idx}')
        
        plt.title(f'Wealth of Top {top_n} Experts')
        plt.xlabel('Time')
        plt.ylabel('Expert Wealth')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def plot_strategy_wealth(self):
        """
        Plot the average wealth of each strategy type
        """
        if self.expert_wealth is None:
            print("Run the strategy first by calling run_strategy()")
            return
        
        # Group experts by strategy
        strategy_wealth = {}
        expert_idx = 0
        
        for strategy_name, _, param_count in self.strategies:
            strategy_experts = []
            
            for cluster_name in self.clusters:
                if param_count == 1:
                    for _ in self.n1_params:
                        strategy_experts.append(expert_idx)
                        expert_idx += 1
                else:  # param_count == 2
                    for n1 in self.n1_params:
                        for n2 in self.n2_params:
                            if n1 < n2:
                                strategy_experts.append(expert_idx)
                                expert_idx += 1
            
            if strategy_experts:
                strategy_wealth[strategy_name] = np.mean(self.expert_wealth[strategy_experts], axis=0)
        
        # Plot average wealth for each strategy
        plt.figure(figsize=(15, 10))
        
        for strategy_name, wealth in strategy_wealth.items():
            plt.plot(wealth, label=strategy_name)
        
        plt.title('Average Wealth by Strategy')
        plt.xlabel('Time')
        plt.ylabel('Average Wealth')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def test_statistical_arbitrage(self, n_periods=400, constrained_mean=True):
        """
        Test for statistical arbitrage using the Jarrow et al. method
        
        Parameters:
        -----------
        n_periods : int
            Number of periods to use for the test
        constrained_mean : bool
            Whether to use the constrained mean model (CM) or unconstrained mean model (UM)
            
        Returns:
        --------
        is_arb : bool
            Whether the strategy is a statistical arbitrage
        min_t : float
            Min-t statistic
        p_value : float
            p-value for the Min-t statistic
        """
        if self.pnl is None or len(self.pnl) < n_periods:
            print("Run the strategy first with sufficient data")
            return False, 0, 1.0
        
        # Use the last n_periods of PnL
        incremental_profits = self.pnl[-n_periods:]
        
        # For the CM model (θ = 0), fit parameters μ, σ, λ
        # For the UM model, fit parameters μ, σ, λ, θ
        
        # Maximum likelihood estimation
        if constrained_mean:
            # CM model
            def neg_log_likelihood(params):
                mu, sigma, lambda_ = params
                
                if sigma <= 0:
                    return 1e10  # Penalty for invalid parameters
                
                ll = 0
                for i in range(len(incremental_profits)):
                    t = i + 1
                    variance = sigma**2 * (t**(2*lambda_))
                    ll += -0.5 * np.log(2 * np.pi * variance) - (incremental_profits[i] - mu)**2 / (2 * variance)
                
                return -ll
            
            # Initial parameter guess
            initial_params = [np.mean(incremental_profits), np.std(incremental_profits), -0.1]
            
            # Parameter bounds
            bounds = [(None, None), (1e-10, None), (None, None)]
            
            # Optimize
            result = minimize(neg_log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
            
            mu, sigma, lambda_ = result.x
            theta = 0  # Constrained
            
            # Calculate standard errors from Hessian
            hessian = result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') else result.hess_inv
            std_errors = np.sqrt(np.diag(hessian))
            
            # Calculate t-statistics
            t_mu = mu / std_errors[0]
            t_lambda = -lambda_ / std_errors[2]
            
            # Calculate Min-t statistic
            min_t = min(t_mu, t_lambda)
            
            # Empirical p-value calculation using Monte Carlo
            p_value = self._monte_carlo_p_value(min_t, mu, sigma, lambda_, theta, n_periods, constrained_mean)
            
            # Check if statistically significant (reject null at 5% level)
            is_arb = p_value < 0.05
            
            # Calculate probability of loss
            prob_loss = self._probability_of_loss(mu, sigma, lambda_, theta, n_periods)
            
            return is_arb, min_t, p_value, prob_loss
        else:
            # TODO: Implement UM model
            print("UM model not implemented yet")
            return False, 0, 1.0, None
    
    def _monte_carlo_p_value(self, min_t, mu, sigma, lambda_, theta, n_periods, constrained_mean, n_sims=5000):
        """
        Calculate p-value for Min-t statistic using Monte Carlo simulation
        
        Parameters:
        -----------
        min_t : float
            Observed Min-t statistic
        mu, sigma, lambda_, theta : float
            Model parameters
        n_periods : int
            Number of periods in simulation
        constrained_mean : bool
            Whether to use the CM model
        n_sims : int
            Number of simulations
            
        Returns:
        --------
        p_value : float
            Empirical p-value
        """
        # For null hypothesis, set mu=0 and lambda=0
        min_t_values = np.zeros(n_sims)
        
        for i in range(n_sims):
            # Simulate process under null
            incremental_profits = np.random.normal(0, sigma, n_periods)
            
            # Fit model
            if constrained_mean:
                def neg_log_likelihood(params):
                    mu_, sigma_, lambda_ = params
                    
                    if sigma_ <= 0:
                        return 1e10
                    
                    ll = 0
                    for j in range(len(incremental_profits)):
                        t = j + 1
                        variance = sigma_**2 * (t**(2*lambda_))
                        ll += -0.5 * np.log(2 * np.pi * variance) - (incremental_profits[j] - mu_)**2 / (2 * variance)
                    
                    return -ll
                
                initial_params = [np.mean(incremental_profits), np.std(incremental_profits), -0.1]
                bounds = [(None, None), (1e-10, None), (None, None)]
                
                try:
                    result = minimize(neg_log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
                    
                    mu_, sigma_, lambda_ = result.x
                    
                    # Calculate standard errors
                    hessian = result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') else result.hess_inv
                    std_errors = np.sqrt(np.diag(hessian))
                    
                    # Calculate t-statistics
                    t_mu = mu_ / std_errors[0]
                    t_lambda = -lambda_ / std_errors[2]
                    
                    # Calculate Min-t statistic
                    min_t_values[i] = min(t_mu, t_lambda)
                except:
                    # If optimization fails, use a conservative value
                    min_t_values[i] = -999
            
        # Calculate empirical p-value
        p_value = np.mean(min_t_values >= min_t)
        
        return p_value
    
    def _probability_of_loss(self, mu, sigma, lambda_, theta, n_periods, time_points=25):
        """
        Calculate probability of loss over time
        
        Parameters:
        -----------
        mu, sigma, lambda_, theta : float
            Model parameters
        n_periods : int
            Number of periods to project
        time_points : int
            Number of time points to evaluate
            
        Returns:
        --------
        prob_loss : ndarray
            Probability of loss for each time point
        """
        time_points = min(time_points, n_periods)
        periods = np.linspace(1, n_periods, time_points).astype(int)
        prob_loss = np.zeros(time_points)
        
        for i, n in enumerate(periods):
            # Calculate mean and variance
            mu_n = mu * n
            var_n = sigma**2 * np.sum(np.arange(1, n+1)**(2*lambda_))
            
            # Calculate probability of loss
            if var_n > 0:
                prob_loss[i] = norm.cdf(-mu_n / np.sqrt(var_n))
            else:
                prob_loss[i] = 0.5  # Default to 50% when variance is 0
        
        return prob_loss


def generate_simulated_data(n_stocks=15, n_days=1000, mean_reversion=0.01, noise_level=0.01):
    """
    Generate simulated price data for testing the strategy
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks to simulate
    n_days : int
        Number of days to simulate
    mean_reversion : float
        Mean reversion parameter
    noise_level : float
        Noise level (standard deviation)
        
    Returns:
    --------
    prices : ndarray
        Simulated price data of shape (n_days, n_stocks)
    """
    # Generate correlated stock data
    prices = np.zeros((n_days, n_stocks))
    
    # Initialize with random starting prices
    prices[0] = np.random.uniform(50, 200, n_stocks)
    
    # Create a valid correlation matrix using factor model approach
    # This ensures it's positive definite
    num_factors = 3  # Number of latent factors
    
    # Factor loadings
    factor_loadings = np.random.normal(0, 1, (n_stocks, num_factors))
    
    # Factor-based correlation matrix (ensures positive definiteness)
    corr_matrix = np.dot(factor_loadings, factor_loadings.T)
    
    # Add diagonal to ensure it's a proper correlation matrix
    diag_vals = np.diag(corr_matrix)
    for i in range(n_stocks):
        for j in range(n_stocks):
            corr_matrix[i, j] = corr_matrix[i, j] / np.sqrt(diag_vals[i] * diag_vals[j])
    
    # Add some sector-based correlation structure
    for i in range(n_stocks):
        sector_i = i // 5  # 3 sectors of 5 stocks each
        for j in range(i+1, n_stocks):
            sector_j = j // 5
            
            # Higher correlation within the same sector
            if sector_i == sector_j:
                corr_matrix[i, j] = 0.7 * corr_matrix[i, j] + 0.3  # Increase correlation
                corr_matrix[j, i] = corr_matrix[i, j]
    
    # Ensure diagonal is 1.0
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Make sure it's positive definite by adding a small value to the diagonal if needed
    while True:
        try:
            L = np.linalg.cholesky(corr_matrix)
            break
        except np.linalg.LinAlgError:
            corr_matrix = corr_matrix * 0.99
            np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate price paths
    for t in range(1, n_days):
        # Generate correlated noise
        z = np.random.normal(0, noise_level, n_stocks)
        correlated_noise = np.dot(L, z)
        
        # Add mean reversion term
        mean_levels = prices[0]  # Mean reversion levels
        mean_reversion_term = mean_reversion * (mean_levels - prices[t-1])
        
        # Compute returns
        returns = mean_reversion_term + correlated_noise
        
        # Update prices
        prices[t] = prices[t-1] * (1 + returns)
    
    return prices


def estimate_pbo(strategy, n_simulations=30, backtest_length=60):
    """
    Estimate the probability of backtest overfitting using CSCV
    
    Parameters:
    -----------
    strategy : TechnicalTradingStrategy
        Trading strategy
    n_simulations : int
        Number of simulations to run
    backtest_length : int
        Length of each backtest
        
    Returns:
    --------
    pbo : float
        Estimated probability of backtest overfitting
    """
    # Generate simulated data for testing
    total_length = n_simulations * backtest_length
    prices = generate_simulated_data(n_stocks=15, n_days=total_length)
    
    # Array to store performance (Sharpe ratio) for each simulation
    is_performances = np.zeros(n_simulations)
    oos_performances = np.zeros(n_simulations)
    
    # Run simulations
    for i in range(n_simulations):
        # Extract data for this simulation
        start_idx = i * backtest_length
        end_idx = (i + 1) * backtest_length
        
        # Mid-point for in-sample/out-of-sample split
        mid_idx = start_idx + backtest_length // 2
        
        # In-sample period
        is_data = prices[start_idx:mid_idx]
        
        # Out-of-sample period
        oos_data = prices[mid_idx:end_idx]
        
        # Run strategy on in-sample data
        strategy.generate_experts(is_data.shape[1])
        _, is_pnl = strategy.run_strategy(is_data)
        
        # Calculate in-sample Sharpe ratio
        is_sharpe = np.mean(is_pnl) / (np.std(is_pnl) + 1e-10) * np.sqrt(252)
        is_performances[i] = is_sharpe
        
        # Run strategy on out-of-sample data (using parameters learned in-sample)
        _, oos_pnl = strategy.run_strategy(oos_data)
        
        # Calculate out-of-sample Sharpe ratio
        oos_sharpe = np.mean(oos_pnl) / (np.std(oos_pnl) + 1e-10) * np.sqrt(252)
        oos_performances[i] = oos_sharpe
    
    # Calculate probability of backtest overfitting
    # PBO = Probability(OOS Rank < IS Rank)
    count = 0
    total = 0
    
    for i in range(n_simulations):
        for j in range(n_simulations):
            if i != j:
                # Count cases where IS rank is better but OOS rank is worse
                if (is_performances[i] > is_performances[j]) and (oos_performances[i] < oos_performances[j]):
                    count += 1
                total += 1
    
    pbo = count / total
    
    return pbo


# Example usage
if __name__ == "__main__":
    # Generate simulated data
    print("Generating simulated data...")
    prices = generate_simulated_data(n_stocks=15, n_days=1000)
    
    # Initialize strategy
    strategy = TechnicalTradingStrategy(
        n1_params=[4, 8, 16, 32], 
        n2_params=[8, 16, 32, 64],
        transaction_costs=False
    )
    
    # Generate experts
    strategy.generate_experts(prices.shape[1])
    
    # Run strategy
    print("Running strategy without transaction costs...")
    portfolio_wealth, pnl = strategy.run_strategy(prices)
    
    # Plot results
    strategy.plot_results("Technical Trading Strategy (No Transaction Costs)")
    
    # Plot expert wealth
    strategy.plot_expert_wealth(top_n=5)
    
    # Plot strategy wealth
    strategy.plot_strategy_wealth()
    
    # Test for statistical arbitrage
    print("Testing for statistical arbitrage...")
    is_arb, min_t, p_value, prob_loss = strategy.test_statistical_arbitrage(n_periods=400)
    
    print(f"Statistical arbitrage test result (no transaction costs):")
    print(f"Min-t statistic: {min_t:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Is statistical arbitrage: {is_arb}")
    
    # Plot probability of loss
    plt.figure(figsize=(10, 6))
    plt.plot(prob_loss)
    plt.title('Probability of Loss Over Time')
    plt.xlabel('Time Period')
    plt.ylabel('Probability of Loss')
    plt.grid(True)
    plt.show()
    
    # Run strategy with transaction costs
    strategy_with_costs = TechnicalTradingStrategy(
        n1_params=[4, 8, 16, 32], 
        n2_params=[8, 16, 32, 64],
        transaction_costs=True,
        cost_per_day=0.0009  # 9 bps per day
    )
    
    strategy_with_costs.generate_experts(prices.shape[1])
    
    print("Running strategy with transaction costs...")
    portfolio_wealth_with_costs, pnl_with_costs = strategy_with_costs.run_strategy(prices)
    
    # Plot results with transaction costs
    strategy_with_costs.plot_results("Technical Trading Strategy (With Transaction Costs)")
    
    # Test for statistical arbitrage with transaction costs
    print("Testing for statistical arbitrage with transaction costs...")
    is_arb_with_costs, min_t_with_costs, p_value_with_costs, prob_loss_with_costs = strategy_with_costs.test_statistical_arbitrage(n_periods=400)
    
    print(f"Statistical arbitrage test result (with transaction costs):")
    print(f"Min-t statistic: {min_t_with_costs:.4f}")
    print(f"p-value: {p_value_with_costs:.4f}")
    print(f"Is statistical arbitrage: {is_arb_with_costs}")
    
    # Plot probability of loss with transaction costs
    plt.figure(figsize=(10, 6))
    plt.plot(prob_loss_with_costs)
    plt.title('Probability of Loss Over Time (With Transaction Costs)')
    plt.xlabel('Time Period')
    plt.ylabel('Probability of Loss')
    plt.grid(True)
    plt.show()
    
    # Estimate probability of backtest overfitting
    print("Estimating probability of backtest overfitting...")
    pbo = estimate_pbo(strategy, n_simulations=10, backtest_length=60)  # Reduced to 10 simulations for speed
    print(f"Probability of backtest overfitting: {pbo:.4f}")