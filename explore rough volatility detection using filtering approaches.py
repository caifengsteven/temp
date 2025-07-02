import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma, loggamma
from scipy.stats import poisson
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RoughVolatilityModel:
    """
    Class to simulate and estimate rough volatility models using a filtering approach.
    """
    
    def __init__(self, T=1, N=480, b=8000):
        """
        Initialize the rough volatility model.
        
        Parameters:
        -----------
        T : float
            Time horizon in days
        N : int
            Number of discrete time steps per day
        b : float
            Base intensity parameter
        """
        self.T = T
        self.N = N
        self.dt = 1/N
        self.time_grid = np.linspace(0, T, int(T*N) + 1)
        self.b = b
        
    def _calculate_coefficients(self, H, J, xi_0, xi_J):
        """
        Calculate coefficients and mean-reversion speeds for fBM approximation.
        """
        # Calculate the constant c_H
        c_H = np.sqrt(np.pi * H * (2*H - 1) / 
                      ((2 - 2*H) * (H + 0.5)**2 * np.sin(np.pi * (H - 0.5))))
        
        # Create geometric partition for xi
        r = (xi_J / xi_0)**(1/J)
        xi = xi_0 * r**np.arange(J+1)
        
        # Calculate coefficients and mean-reversion speeds
        c = np.zeros(J)
        kappa = np.zeros(J)
        
        for j in range(J):
            # Define the measure μ(dx)
            def mu(x):
                return c_H * x**(-H-0.5) / gamma(0.5-H)
            
            # Compute c_j and kappa_j using numerical integration
            c[j] = c_H * (xi[j+1]**(0.5-H) - xi[j]**(0.5-H)) / ((0.5-H) * gamma(0.5-H))
            
            # Compute kappa_j (mean-reversion speed)
            def integrand(x):
                return x * mu(x)
            
            # Use midpoint rule for simplicity
            x_vals = np.linspace(xi[j], xi[j+1], 100)
            kappa[j] = np.sum(integrand(x_vals)) * (xi[j+1] - xi[j]) / 100 / c[j]
        
        return c, kappa
    
    def estimate_hurst_direct(self, log_returns, method='variance'):
        """
        Estimate Hurst parameter directly from log returns using simple methods.
        
        Parameters:
        -----------
        log_returns : numpy.ndarray
            Log returns
        method : str
            Method to use for estimation ('variance' or 'dma')
            
        Returns:
        --------
        H : float
            Estimated Hurst parameter
        """
        if method == 'variance':
            # Variance method based on aggregation
            H_estimates = []
            max_tau = min(20, len(log_returns) // 4)  # Don't use too large tau
            
            for tau in range(2, max_tau):
                # Aggregate returns
                agg_returns = np.array([np.sum(log_returns[i:i+tau]) for i in range(0, len(log_returns) - tau, tau)])
                
                # Calculate variance ratio
                var_ratio = np.var(agg_returns) / (tau * np.var(log_returns))
                
                # Estimate H
                H_est = 0.5 + np.log(var_ratio) / (2 * np.log(tau))
                H_estimates.append(min(max(H_est, 0.01), 0.49))
            
            # Use median to be robust to outliers
            H = np.median(H_estimates)
            
        elif method == 'dma':
            # Detrended Moving Average method
            H_estimates = []
            max_window = min(20, len(log_returns) // 4)
            
            for window in range(5, max_window):
                # Calculate moving average
                ma = np.convolve(log_returns, np.ones(window)/window, mode='valid')
                
                # Calculate DMA
                dma = log_returns[window-1:window-1+len(ma)] - ma
                
                # Calculate variance of DMA for different scales
                var_dma = np.var(dma)
                
                # Estimate H
                H_est = 0.5 - np.log(var_dma) / (2 * np.log(window))
                H_estimates.append(min(max(H_est, 0.01), 0.49))
            
            # Use median to be robust to outliers
            H = np.median(H_estimates)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return H

class MarketSimulator:
    """
    Class to simulate market data with rough volatility dynamics
    """
    
    def __init__(self, start_date='2023-01-01', num_days=30, intraday_points=78):
        """
        Initialize the market simulator.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        num_days : int
            Number of days to simulate
        intraday_points : int
            Number of intraday points per day
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.num_days = num_days
        self.intraday_points = intraday_points
        
        # Market parameters
        self.price_drift = 0.0002  # Daily drift term
        self.price_vol = 0.015  # Base volatility (increased for better signal)
        self.vol_mean_reversion = 0.1  # Speed of mean reversion for volatility
        self.vol_vol = 0.5  # Volatility of volatility (increased for better signal)
        
    def simulate_market_with_rough_volatility(self, H_values=None, seed=None):
        """
        Simulate market data with rough volatility dynamics.
        
        Parameters:
        -----------
        H_values : list, optional
            List of Hurst parameters for different regimes
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        intraday_data : pandas.DataFrame
            Simulated intraday price data
        daily_data : pandas.DataFrame
            Simulated daily price data
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Define Hurst parameter regimes if not provided
        if H_values is None:
            H_values = [0.1, 0.3, 0.1, 0.4, 0.2]  # Different volatility regimes
            
        # Extend H_values if needed
        while len(H_values) < self.num_days:
            H_values = H_values + H_values
        
        H_values = H_values[:self.num_days]
        
        # Create date range for daily data
        dates = [self.start_date + timedelta(days=i) for i in range(self.num_days)]
        
        # Filter out weekends
        dates = [date for date in dates if date.weekday() < 5]
        num_trading_days = len(dates)
        
        # Initialize price and volatility arrays
        daily_open = np.zeros(num_trading_days)
        daily_high = np.zeros(num_trading_days)
        daily_low = np.zeros(num_trading_days)
        daily_close = np.zeros(num_trading_days)
        daily_volume = np.zeros(num_trading_days)
        daily_h = np.zeros(num_trading_days)
        
        # Initialize intraday data storage
        intraday_times = []
        intraday_prices = []
        intraday_volumes = []
        intraday_h = []
        
        # Set initial price
        price = 100.0
        daily_open[0] = price
        
        # Simulate each trading day
        for i, date in enumerate(dates):
            # Get Hurst parameter for this day
            day_idx = min(i, len(H_values)-1)
            H = H_values[day_idx]
            daily_h[i] = H
            
            # For rough volatility (low H), use higher volatility
            vol_factor = 1.0 + (0.5 - H) * 4.0  # More impact for lower H
            day_vol = self.price_vol * vol_factor
            
            # Simulate intraday prices
            day_prices = np.zeros(self.intraday_points)
            day_prices[0] = price
            
            # Track high and low
            day_high = price
            day_low = price
            
            # Generate intraday market times
            market_open = datetime.combine(date.date(), datetime.strptime("9:30", "%H:%M").time())
            
            # Simulate a fractal Brownian motion with Hurst parameter H
            # We'll use a simple method based on cumulative sum of normal variables
            dX = np.zeros(self.intraday_points)
            
            # First point is just normal
            dX[0] = np.random.normal(0, 1)
            
            # For the rest, we'll create some dependency based on H
            for j in range(1, self.intraday_points):
                # For low H (rough), correlation with past is negative
                # For high H (smooth), correlation with past is positive
                memory_factor = 2*H - 1  # -0.8 for H=0.1, 0.6 for H=0.8
                
                # Weight for past values, decaying with distance
                weights = np.power(np.arange(1, min(j+1, 10), 1, dtype=float), memory_factor-1)
                weights = weights / np.sum(weights)
                
                # Calculate the weighted sum of past values
                weighted_past = np.sum(weights * dX[j-len(weights):j])
                
                # Generate current value with dependency on past
                if H < 0.5:  # Rough - negative correlation with past
                    dX[j] = np.random.normal(-0.5 * weighted_past, 1)
                else:  # Smooth - positive correlation with past
                    dX[j] = np.random.normal(0.5 * weighted_past, 1)
            
            # Scale to desired volatility and convert to returns
            returns = day_vol * dX / np.sqrt(self.intraday_points)
            
            # Add drift
            returns += self.price_drift / self.intraday_points
            
            # Convert to prices
            for j in range(1, self.intraday_points):
                price = price * (1 + returns[j])
                day_prices[j] = price
                
                # Update high and low
                day_high = max(day_high, price)
                day_low = min(day_low, price)
                
                # Calculate time for this point
                minutes_offset = j * (390 / self.intraday_points)
                current_time = market_open + timedelta(minutes=minutes_offset)
                
                # Store intraday data
                intraday_times.append(current_time)
                intraday_prices.append(price)
                intraday_h.append(H)
                
                # Generate random volume
                volume = np.random.exponential(1000000)
                intraday_volumes.append(volume)
            
            # Store daily OHLC
            daily_open[i] = day_prices[0]
            daily_high[i] = day_high
            daily_low[i] = day_low
            daily_close[i] = day_prices[-1]
            
            # Generate daily volume (sum of intraday volumes)
            daily_volume[i] = sum(intraday_volumes[-self.intraday_points:])
            
            # Update price for next day
            price = day_prices[-1]
        
        # Create daily DataFrame
        daily_data = pd.DataFrame({
            'Open': daily_open,
            'High': daily_high,
            'Low': daily_low,
            'Close': daily_close,
            'Volume': daily_volume,
            'True_H': daily_h
        }, index=dates)
        
        # Create intraday DataFrame
        intraday_data = pd.DataFrame({
            'Open': intraday_prices,
            'High': intraday_prices,
            'Low': intraday_prices,
            'Close': intraday_prices,
            'Volume': intraday_volumes,
            'True_H': intraday_h
        }, index=intraday_times)
        
        return intraday_data, daily_data, daily_h  # Return daily_h explicitly

class RoughVolatilityStrategy:
    """
    Trading strategy based on rough volatility detection
    """
    
    def __init__(self, lookback_days=10, intraday_freq='5min', rolling_window=5, 
                 entry_threshold=0.8, exit_threshold=0.3, max_positions=1):
        """
        Initialize the trading strategy.
        
        Parameters:
        -----------
        lookback_days : int
            Number of days to look back for training the model
        intraday_freq : str
            Frequency of intraday data
        rolling_window : int
            Window size for rolling volatility normalization
        entry_threshold : float
            Threshold for entry signals (in standard deviations)
        exit_threshold : float
            Threshold for exit signals (in standard deviations)
        max_positions : int
            Maximum number of positions allowed (1 for long-only, 2 for long-short)
        """
        self.lookback_days = lookback_days
        self.intraday_freq = intraday_freq
        self.rolling_window = rolling_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_positions = max_positions
        
        # Extract number of discrete steps per day
        if intraday_freq.endswith('min'):
            mins = int(intraday_freq[:-3])
            self.N = int(390 / mins)  # 390 minutes in a trading day (6.5 hours)
        else:
            raise ValueError("Frequency must be specified in minutes (e.g., '5min')")
        
        # Initialize rough volatility model
        self.model = RoughVolatilityModel(T=1, N=self.N, b=8000)
        
        # Portfolio state
        self.positions = 0  # -1 (short), 0 (flat), 1 (long)
        self.position_prices = []
        self.trades = []
        
    def estimate_hurst_parameter(self, intraday_data, current_date):
        """
        Estimate the Hurst parameter from intraday data.
        
        Parameters:
        -----------
        intraday_data : pandas.DataFrame
            Intraday data
        current_date : datetime
            Current date
            
        Returns:
        --------
        H_estimate : float
            Estimated Hurst parameter
        """
        # Get data for the lookback period
        start_date = current_date - timedelta(days=self.lookback_days)
        
        lookback_data = intraday_data[intraday_data.index >= start_date]
        lookback_data = lookback_data[lookback_data.index < current_date]
        
        # Convert to log returns
        log_returns = np.log(lookback_data['Close'] / lookback_data['Close'].shift(1)).dropna()
        
        if len(log_returns) < 10:  # Need enough data points
            return 0.25  # Default to mid-point if not enough data
        
        # Use direct estimation methods (simpler and more robust)
        try:
            H_estimate = self.model.estimate_hurst_direct(log_returns.values, method='variance')
        except:
            # Fallback to a simple estimation
            H_estimate = 0.25  # Default to mid-point
        
        return H_estimate
    
    def calculate_realized_volatility(self, intraday_data, current_date):
        """
        Calculate realized volatility from intraday data.
        
        Parameters:
        -----------
        intraday_data : pandas.DataFrame
            Intraday data
        current_date : datetime
            Current date
            
        Returns:
        --------
        realized_vol : float
            Realized volatility
        """
        # Get data for the current day
        day_data = intraday_data[intraday_data.index.date == current_date.date()]
        
        if len(day_data) < 2:
            return None
        
        # Calculate log returns
        log_returns = np.log(day_data['Close'] / day_data['Close'].shift(1)).dropna()
        
        # Calculate realized volatility (sum of squared returns)
        realized_vol = np.sqrt(np.sum(log_returns**2))
        
        return realized_vol
    
    def generate_signals(self, intraday_data, daily_data, test_start_date, test_end_date):
        """
        Generate trading signals based on rough volatility detection.
        
        Parameters:
        -----------
        intraday_data : pandas.DataFrame
            Intraday data
        daily_data : pandas.DataFrame
            Daily data
        test_start_date : datetime
            Start date for testing period
        test_end_date : datetime
            End date for testing period
            
        Returns:
        --------
        signals : pandas.DataFrame
            DataFrame with trading signals
        """
        # Initialize results
        dates = []
        h_values = []
        true_h_values = []  # Store true H if available
        realized_vols = []
        signals = []
        positions = []
        
        # Initialize volatility history
        vol_history = []
        
        # Current position
        current_position = 0
        
        # Get all trading days in the period
        all_dates = daily_data.index
        test_dates = all_dates[(all_dates >= test_start_date) & (all_dates <= test_end_date)]
        
        # Iterate through each trading day
        for current_date in tqdm(test_dates, desc="Generating Signals"):
            # Estimate Hurst parameter
            H_estimate = self.estimate_hurst_parameter(intraday_data, current_date)
            
            # Get true H if available
            true_H = daily_data.loc[current_date, 'True_H'] if 'True_H' in daily_data.columns else None
            
            # Calculate realized volatility
            realized_vol = self.calculate_realized_volatility(intraday_data, current_date)
            
            if realized_vol is None:
                # Use daily data if intraday not available
                if current_date > all_dates[0]:
                    prev_close = daily_data.loc[current_date - timedelta(days=1):current_date - timedelta(days=1), 'Close'].iloc[0]
                    today_close = daily_data.loc[current_date, 'Close']
                    realized_vol = abs(np.log(today_close / prev_close))
                else:
                    realized_vol = 0.01  # Default value
            
            # Add to history
            vol_history.append(realized_vol)
            
            # Generate signal based on H and realized volatility
            signal = 0  # 0 = no signal, 1 = buy, -1 = sell
            
            if len(vol_history) >= self.rolling_window:
                # Normalize recent volatility relative to history
                recent_vol_mean = np.mean(vol_history[-self.rolling_window:])
                recent_vol_std = np.std(vol_history[-self.rolling_window:])
                
                # Volatility ratio (current vs expected)
                vol_z_score = (realized_vol - recent_vol_mean) / recent_vol_std if recent_vol_std > 0 else 0
                
                # Roughness factor (lower H = rougher volatility)
                # Transform H to a scale where smaller H gives larger values
                roughness_factor = 0.5 - H_estimate  # 0.5 is non-rough (Brownian), lower is rougher
                
                # Strategy logic: When volatility is high and rough, sell/short
                # When volatility is low and less rough, buy/long
                if current_position <= 0 and vol_z_score < -self.entry_threshold and roughness_factor < 0.2:
                    # Low volatility, smoother process - BUY signal
                    signal = 1
                elif current_position >= 0 and vol_z_score > self.entry_threshold and roughness_factor > 0.3:
                    # High volatility, rougher process - SELL signal
                    signal = -1
                elif (current_position > 0 and vol_z_score > self.exit_threshold) or \
                     (current_position < 0 and vol_z_score < -self.exit_threshold):
                    # Exit positions when volatility reverts
                    signal = 0
                    
                # If no signal generated yet, consider using actual position changes
                if signal == 0 and current_position != 0:
                    # Bias towards keeping positions (reduce noise)
                    if current_position > 0 and (vol_z_score > 1.0 or roughness_factor > 0.4):
                        signal = -1  # Close long position
                    elif current_position < 0 and (vol_z_score < -1.0 or roughness_factor < 0.1):
                        signal = 1  # Close short position
            
            # Update position based on signal and max_positions constraint
            if signal == 1 and current_position < self.max_positions:
                current_position += 1
            elif signal == -1 and current_position > -self.max_positions:
                current_position -= 1
            elif signal == 0 and current_position != 0:
                current_position = 0
            
            # Store results
            dates.append(current_date)
            h_values.append(H_estimate)
            if true_H is not None:
                true_h_values.append(true_H)
            realized_vols.append(realized_vol)
            signals.append(signal)
            positions.append(current_position)
        
        # Create results DataFrame
        results_dict = {
            'Date': dates,
            'Hurst_Parameter': h_values,
            'Realized_Volatility': realized_vols,
            'Signal': signals,
            'Position': positions
        }
        
        # Add true H if available
        if true_h_values:
            results_dict['Estimated_True_H'] = true_h_values
            
        results = pd.DataFrame(results_dict)
        results.set_index('Date', inplace=True)
        
        return results
    
    def backtest_strategy(self, signals, daily_data):
        """
        Backtest the trading strategy.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with trading signals
        daily_data : pandas.DataFrame
            Daily data
            
        Returns:
        --------
        performance : pandas.DataFrame
            DataFrame with backtest performance
        """
        # Check if signals DataFrame is empty
        if signals.empty:
            print("\nNo signals generated. This could be due to insufficient data for estimation.")
            # Create a default performance DataFrame with minimal information
            performance = pd.DataFrame({
                'Returns': [0],
                'Strategy_Returns': [0],
                'Cum_Market_Returns': [0],
                'Cum_Strategy_Returns': [0]
            }, index=[daily_data.index[0]])
            return performance
        
        # Merge signals with price data
        # Add suffixes to avoid column name conflicts
        backtest_data = daily_data.join(signals, how='inner', lsuffix='_daily', rsuffix='_signals')
        
        # Check if the merged DataFrame is empty
        if backtest_data.empty:
            print("\nNo matching dates between signals and price data.")
            # Create a default performance DataFrame
            performance = pd.DataFrame({
                'Returns': [0],
                'Strategy_Returns': [0],
                'Cum_Market_Returns': [0],
                'Cum_Strategy_Returns': [0]
            }, index=[daily_data.index[0]])
            return performance
        
        # Initialize portfolio columns
        backtest_data['Returns'] = backtest_data['Close'].pct_change()
        backtest_data['Strategy_Returns'] = backtest_data['Position'] * backtest_data['Returns'].shift(-1)
        
        # Replace NaN with 0 for the first row
        backtest_data['Returns'].iloc[0] = 0
        backtest_data['Strategy_Returns'].iloc[-1] = 0  # Last row has no next day return
        
        # Calculate cumulative returns
        backtest_data['Cum_Market_Returns'] = (1 + backtest_data['Returns']).cumprod() - 1
        backtest_data['Cum_Strategy_Returns'] = (1 + backtest_data['Strategy_Returns']).cumprod() - 1
        
        # Calculate performance metrics
        total_days = len(backtest_data)
        
        # Return metrics
        total_return = backtest_data['Cum_Strategy_Returns'].iloc[-1]
        annual_return = (1 + total_return)**(252/total_days) - 1
        
        # Risk metrics
        daily_vol = backtest_data['Strategy_Returns'].std()
        annual_vol = daily_vol * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown analysis
        backtest_data['Peak'] = backtest_data['Cum_Strategy_Returns'].cummax()
        backtest_data['Drawdown'] = backtest_data['Peak'] - backtest_data['Cum_Strategy_Returns']
        max_drawdown = backtest_data['Drawdown'].max()
        
        # Trading metrics
        trades = np.sum(backtest_data['Position'] != backtest_data['Position'].shift(1))
        
        # Print summary
        print("\nStrategy Performance Summary:")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annual_return:.2%}")
        print(f"Annualized Volatility: {annual_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Number of Trades: {trades}")
        
        # Plot performance
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(backtest_data.index, backtest_data['Cum_Market_Returns'], 'b-', label='Market')
        plt.plot(backtest_data.index, backtest_data['Cum_Strategy_Returns'], 'g-', label='Strategy')
        plt.fill_between(backtest_data.index, 0, backtest_data['Drawdown'], color='red', alpha=0.3)
        plt.legend()
        plt.title('Cumulative Returns')
        
        plt.subplot(3, 1, 2)
        plt.plot(backtest_data.index, backtest_data['Hurst_Parameter'], 'r-', label='Estimated H')
        if 'True_H_daily' in backtest_data.columns:
            plt.plot(backtest_data.index, backtest_data['True_H_daily'], 'b--', label='True H')
            plt.legend()
        elif 'Estimated_True_H' in backtest_data.columns:
            plt.plot(backtest_data.index, backtest_data['Estimated_True_H'], 'b--', label='True H')
            plt.legend()
        plt.title('Hurst Parameter')
        plt.ylabel('H')
        
        plt.subplot(3, 1, 3)
        plt.plot(backtest_data.index, backtest_data['Realized_Volatility'], 'k-')
        plt.title('Realized Volatility')
        
        plt.tight_layout()
        plt.savefig('rough_volatility_strategy_performance.png')
        
        # Create a second plot showing trading signals
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(backtest_data.index, backtest_data['Close'], 'b-', label='Price')
        
        # Buy signals
        buy_signals = backtest_data[backtest_data['Signal'] == 1].index
        if len(buy_signals) > 0:
            plt.scatter(buy_signals, backtest_data.loc[buy_signals, 'Close'], 
                        marker='^', color='g', s=100, label='Buy')
        
        # Sell signals
        sell_signals = backtest_data[backtest_data['Signal'] == -1].index
        if len(sell_signals) > 0:
            plt.scatter(sell_signals, backtest_data.loc[sell_signals, 'Close'], 
                        marker='v', color='r', s=100, label='Sell')
        
        plt.legend()
        plt.title('Price and Trading Signals')
        
        plt.subplot(2, 1, 2)
        plt.plot(backtest_data.index, backtest_data['Position'], 'b-')
        plt.title('Position')
        plt.axhline(y=0, color='k', linestyle='--')
        
        plt.tight_layout()
        plt.savefig('trading_signals.png')
        
        return backtest_data
    
    def run_strategy_with_simulated_data(self, start_date='2023-01-01', num_days=60, lookback=10):
        """
        Run the strategy on simulated data.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        num_days : int
            Number of days to simulate
        lookback : int
            Number of days for lookback period
            
        Returns:
        --------
        performance : pandas.DataFrame
            DataFrame with backtest performance
        """
        print(f"Running rough volatility strategy on simulated data from {start_date} for {num_days} days")
        
        # Create market simulator
        simulator = MarketSimulator(start_date=start_date, num_days=num_days, intraday_points=self.N)
        
        # Generate Hurst parameter sequence with market regimes
        # Low H = rough volatility (high volatility regime)
        # High H = smooth volatility (low volatility regime)
        H_regimes = [
            # Start with medium roughness
            [0.3, 0.3, 0.3, 0.3, 0.3],
            # Transition to rough regime (e.g., market stress)
            [0.25, 0.2, 0.15, 0.1, 0.1],
            # Sustained rough period
            [0.1, 0.1, 0.1, 0.1, 0.1],
            # Gradual recovery
            [0.15, 0.2, 0.25, 0.3, 0.35],
            # Calm market
            [0.4, 0.4, 0.4, 0.4, 0.4],
            # New market stress
            [0.35, 0.3, 0.25, 0.2, 0.15],
        ]
        
        # Flatten the regime list
        H_values = [H for regime in H_regimes for H in regime]
        
        # Simulate market data
        intraday_data, daily_data, daily_h = simulator.simulate_market_with_rough_volatility(H_values=H_values, seed=42)
        
        # Define test period (skip lookback period)
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        test_start = start_datetime + timedelta(days=lookback)
        test_end = start_datetime + timedelta(days=num_days)
        
        # Generate signals
        signals = self.generate_signals(intraday_data, daily_data, test_start, test_end)
        
        # Backtest strategy
        performance = self.backtest_strategy(signals, daily_data)
        
        # Create a second plot showing the true H values vs time
        plt.figure(figsize=(12, 6))
        
        # Plot estimated H values
        if not signals.empty and 'Hurst_Parameter' in signals.columns:
            plt.plot(signals.index, signals['Hurst_Parameter'], 'r--', label='Estimated H')
            
        # Plot true H values from daily_data
        plt.plot(daily_data.index, daily_data['True_H'], 'b-', label='True H')
        
        plt.title('True vs Estimated Hurst Parameter')
        plt.legend()
        plt.savefig('true_vs_estimated_H.png')
        
        return performance, intraday_data, daily_data, H_values

# Usage example
if __name__ == "__main__":
    # Create strategy with parameters optimized for simulated data
    strategy = RoughVolatilityStrategy(
        lookback_days=5,
        intraday_freq='5min',
        rolling_window=5,
        entry_threshold=0.8,  # Lower threshold to generate more signals
        exit_threshold=0.3,   # Lower threshold to exit positions more quickly
        max_positions=1
    )
    
    # Run on simulated data
    performance, intraday_data, daily_data, true_H_values = strategy.run_strategy_with_simulated_data(
        start_date='2023-01-01',
        num_days=60,  # Increase number of days for better statistics
        lookback=5
    )
    
    # Print correlation between true and estimated H if we have enough data
    if not performance.empty:
        # Check different possible column names for true H
        true_h_col = None
        if 'True_H_daily' in performance.columns:
            true_h_col = 'True_H_daily'
        elif 'Estimated_True_H' in performance.columns:
            true_h_col = 'Estimated_True_H'
            
        if true_h_col and 'Hurst_Parameter' in performance.columns:
            correlation = performance['Hurst_Parameter'].corr(performance[true_h_col])
            print(f"\nCorrelation between true and estimated H: {correlation:.4f}")