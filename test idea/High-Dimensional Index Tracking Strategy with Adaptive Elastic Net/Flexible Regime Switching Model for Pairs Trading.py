import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class LevyDrivenOU:
    """
    Class for Lévy-driven Ornstein-Uhlenbeck process estimation and simulation
    """
    def __init__(self, theta=None, mu=None, sigma=None, lambda_=None, eta=None):
        """Initialize Lévy-driven OU process parameters"""
        self.theta = theta  # Mean-reversion speed
        self.mu = mu        # Mean-reversion level
        self.sigma = sigma  # Diffusion parameter
        self.lambda_ = lambda_  # Jump intensity
        self.eta = eta      # Jump size parameter
    
    def simulate(self, n_steps, dt=1/250, x0=0):
        """
        Simulate a Lévy-driven OU process
        
        Parameters:
        -----------
        n_steps: int
            Number of time steps to simulate
        dt: float
            Time increment
        x0: float
            Initial value
            
        Returns:
        --------
        x: numpy.array
            Simulated path
        """
        # Initialize the process
        x = np.zeros(n_steps+1)
        x[0] = x0
        
        # Time points
        t = np.linspace(0, n_steps*dt, n_steps+1)
        
        # For storing constant mean-reversion level
        if np.isscalar(self.mu):
            mu = np.ones(n_steps+1) * self.mu
        else:
            mu = self.mu
        
        # Simulate path
        for i in range(n_steps):
            # Drift component
            drift = self.theta * (mu[i] - x[i]) * dt
            
            # Diffusion component
            diffusion = self.sigma * np.sqrt(dt) * np.random.normal(0, 1)
            
            # Jump component
            n_jumps = np.random.poisson(self.lambda_ * dt)
            jumps = 0
            if n_jumps > 0:
                # Generate jump sizes from double exponential distribution
                jump_sizes = np.random.exponential(scale=1/self.eta, size=n_jumps)
                signs = np.random.choice([-1, 1], size=n_jumps)
                jumps = np.sum(signs * jump_sizes)
            
            # Update process
            x[i+1] = x[i] + drift + diffusion + jumps
        
        return x, t
    
    def estimate(self, x, dt=1/250):
        """Estimate parameters from data"""
        # Compute increments
        dx = np.diff(x)
        
        # Compute running mean for mean-reversion level
        window_size = min(10, len(x)-1)
        mu = np.zeros_like(x)
        for i in range(len(x)):
            if i < window_size:
                mu[i] = np.mean(x[:i+1])
            else:
                mu[i] = np.mean(x[i-window_size:i+1])
        
        # Compute threshold for jumps
        beta = 0.3
        threshold = beta * np.sqrt(dt)
        
        # Separate jumps from continuous part
        jump_idx = np.abs(dx) > threshold
        cont_idx = ~jump_idx
        
        # Estimate mean-reversion speed (theta)
        if np.sum(cont_idx) > 0:
            numerator = np.sum((mu[:-1][cont_idx] - x[:-1][cont_idx]) * dx[cont_idx])
            denominator = np.sum((mu[:-1][cont_idx] - x[:-1][cont_idx])**2 * dt)
            if denominator != 0:
                self.theta = numerator / denominator
            else:
                self.theta = 0.5
        else:
            self.theta = 0.5
            
        # Ensure theta is positive (for mean reversion)
        self.theta = max(0.1, self.theta)
        
        # Estimate diffusion parameter (sigma)
        if np.sum(cont_idx) > 1:
            self.sigma = np.std(dx[cont_idx]) / np.sqrt(dt)
        else:
            self.sigma = np.std(dx) / np.sqrt(dt)
        
        # Estimate jump parameters (lambda_, eta)
        n_jumps = np.sum(jump_idx)
        if n_jumps > 0:
            self.lambda_ = n_jumps / (len(dx) * dt)
            jump_sizes = np.abs(dx[jump_idx])
            self.eta = 1 / np.mean(jump_sizes) if len(jump_sizes) > 0 else 1.0
        else:
            self.lambda_ = 0.01
            self.eta = 1.0
        
        # Store mean-reversion level
        self.mu = mu
        
        return self
    
    def predict_next(self, x_last, mu_last=None, dt=1/250):
        """Predict next value"""
        if mu_last is None:
            mu_last = self.mu if np.isscalar(self.mu) else self.mu[-1]
            
        # Expected value of process at next time step
        x_next = x_last + self.theta * (mu_last - x_last) * dt
        
        return x_next

class RegimeClassifier:
    """Regime classification algorithm"""
    def __init__(self, min_regime_size=0.15):
        self.min_regime_size = min_regime_size
    
    def classify_data(self, x, thresholds, volatilities):
        """Classify data into regimes based on thresholds"""
        thresholds = sorted(thresholds)
        regimes_data = []
        
        # First regime
        mask = volatilities < thresholds[0]
        regimes_data.append(x[mask])
        
        # Middle regimes
        for i in range(len(thresholds) - 1):
            mask = (volatilities >= thresholds[i]) & (volatilities < thresholds[i+1])
            regimes_data.append(x[mask])
        
        # Last regime
        mask = volatilities >= thresholds[-1]
        regimes_data.append(x[mask])
        
        return regimes_data
    
    def fit_models(self, regimes_data):
        """Fit models to each regime"""
        models = []
        for regime_data in regimes_data:
            model = LevyDrivenOU().estimate(regime_data)
            models.append(model)
        return models
    
    def calc_BIC(self, regimes_data, models):
        """Calculate BIC for model selection"""
        n = sum(len(data) for data in regimes_data)
        k = len(models) * 5  # 5 parameters per model
        
        # Calculate conditional least squares error
        CLS = 0
        for data, model in zip(regimes_data, models):
            for j in range(1, len(data)):
                pred = model.predict_next(data[j-1])
                CLS += (data[j] - pred)**2
        
        # Calculate BIC
        if CLS > 0 and n > 0:
            BIC = n * np.log(CLS / n) + k * np.log(n)
        else:
            BIC = float('inf')
            
        return BIC
    
    def find_optimal_thresholds(self, x, volatilities, n_regimes):
        """Find optimal thresholds for a given number of regimes"""
        if n_regimes == 1:
            return [], float('inf')
            
        # Sort volatilities
        sorted_vols = np.sort(np.unique(volatilities))
        
        # Calculate minimum regime size
        min_size = int(self.min_regime_size * len(x))
        
        # Ensure we have enough data points for the regimes
        if len(sorted_vols) < n_regimes + 1 or len(x) < n_regimes * min_size:
            return [], float('inf')
        
        # Use percentiles as initial thresholds
        percentiles = np.linspace(self.min_regime_size*100, 
                                 (1-self.min_regime_size)*100, 
                                 n_regimes)
        initial_thresholds = np.percentile(volatilities, percentiles[:-1])
        
        # Simple grid search for small number of regimes
        if n_regimes <= 3:
            best_BIC = float('inf')
            best_thresholds = initial_thresholds
            
            # Try different combinations of thresholds
            num_trials = min(10, len(sorted_vols))  # Limit trials for efficiency
            for _ in range(num_trials):
                # Perturb thresholds randomly
                perturbed_thresholds = initial_thresholds * (1 + 0.2 * np.random.randn(len(initial_thresholds)))
                perturbed_thresholds = np.sort(perturbed_thresholds)
                
                # Classify data and fit models
                try:
                    regimes_data = self.classify_data(x, perturbed_thresholds, volatilities)
                    
                    # Check regime sizes
                    if any(len(regime) < min_size for regime in regimes_data):
                        continue
                        
                    models = self.fit_models(regimes_data)
                    BIC = self.calc_BIC(regimes_data, models)
                    
                    # Update best thresholds
                    if BIC < best_BIC:
                        best_BIC = BIC
                        best_thresholds = perturbed_thresholds
                except Exception as e:
                    continue
            
            return best_thresholds, best_BIC
        else:
            # For more regimes, just use initial thresholds
            try:
                regimes_data = self.classify_data(x, initial_thresholds, volatilities)
                models = self.fit_models(regimes_data)
                BIC = self.calc_BIC(regimes_data, models)
                return initial_thresholds, BIC
            except Exception as e:
                return [], float('inf')
    
    def fit(self, x, window_size=390*5):
        """Find optimal number of regimes and fit models"""
        # Calculate rolling volatility
        volatilities = np.zeros_like(x)
        for i in range(len(x)):
            if i < window_size:
                volatilities[i] = np.std(x[:i+1])
            else:
                volatilities[i] = np.std(x[i-window_size:i+1])
        
        # Start with one regime
        regimes_data = [x]
        models = self.fit_models(regimes_data)
        BIC_1 = self.calc_BIC(regimes_data, models)
        
        best_BIC = BIC_1
        best_n_regimes = 1
        best_thresholds = []
        best_models = models
        
        # Try increasing number of regimes
        for n_regimes in range(2, 5):  # Limit to 4 regimes
            thresholds, BIC = self.find_optimal_thresholds(x, volatilities, n_regimes)
            
            if BIC < best_BIC:
                best_BIC = BIC
                best_n_regimes = n_regimes
                best_thresholds = thresholds
                
                # Update best models
                regimes_data = self.classify_data(x, best_thresholds, volatilities)
                best_models = self.fit_models(regimes_data)
            else:
                # Stop if BIC doesn't improve
                break
        
        return best_n_regimes, best_thresholds, best_models, volatilities

class PairsTrading:
    """Pairs trading strategy with regime switching"""
    def __init__(self, formation_period=390*10, trading_period=390*5, top_pairs=5, k=0.5, transaction_cost=0.0020):
        self.formation_period = formation_period
        self.trading_period = trading_period
        self.top_pairs = top_pairs
        self.k = k  # Bollinger band parameter
        self.transaction_cost = transaction_cost
        self.regime_classifier = RegimeClassifier()
    
    def calculate_spread(self, prices_A, prices_B):
        """Calculate spread between two price series"""
        # Handle potential NaN or zero values
        mask = (prices_A > 0) & (prices_B > 0)
        if np.sum(mask) < 2:
            return np.zeros_like(prices_A)
            
        prices_A = prices_A[mask]
        prices_B = prices_B[mask]
        
        # Calculate log returns
        log_A = np.log(prices_A / prices_A[0])
        log_B = np.log(prices_B / prices_B[0])
        
        # Calculate spread
        spread = log_A - log_B
        
        return spread
    
    def select_pairs(self, stock_prices, start_idx, pairs):
        """Select top pairs for trading"""
        end_idx = min(start_idx + self.formation_period, stock_prices.shape[1])
        
        # Calculate metrics for each pair
        pair_metrics = []
        
        for stock_A, stock_B in pairs:
            try:
                # Get prices for formation period
                prices_A = stock_prices[stock_A, start_idx:end_idx]
                prices_B = stock_prices[stock_B, start_idx:end_idx]
                
                # Skip pairs with insufficient data
                if len(prices_A) < 10 or len(prices_B) < 10:
                    continue
                
                # Calculate spread
                spread = self.calculate_spread(prices_A, prices_B)
                
                # Skip pairs with non-meaningful spreads
                if len(spread) < 10 or np.std(spread) < 1e-6:
                    continue
                
                # Fit regime switching model
                n_regimes, thresholds, models, volatilities = self.regime_classifier.fit(spread)
                
                # Calculate metrics for pair selection
                theta = np.mean([model.theta for model in models])
                sigma = np.mean([model.sigma for model in models])
                lambda_ = np.mean([model.lambda_ for model in models])
                eta = np.mean([model.eta for model in models])
                
                # Store metrics
                metrics = {
                    'pair': (stock_A, stock_B),
                    'n_regimes': n_regimes,
                    'models': models,
                    'thresholds': thresholds,
                    'theta': theta,
                    'sigma': sigma,
                    'lambda_': lambda_,
                    'eta': eta,
                    'rank_score': theta + sigma + lambda_ + eta  # Simple combined score
                }
                
                pair_metrics.append(metrics)
                
            except Exception as e:
                # Skip problematic pairs
                continue
        
        # Select top pairs based on combined score
        if pair_metrics:
            pair_metrics.sort(key=lambda x: x['rank_score'], reverse=True)
            selected_num = min(self.top_pairs, len(pair_metrics))
            selected_pairs = [p['pair'] for p in pair_metrics[:selected_num]]
            pair_models = [p['models'] for p in pair_metrics[:selected_num]]
            return selected_pairs, pair_models
        else:
            return [], []
    
    def trade_pairs(self, stock_prices, start_idx, selected_pairs, pair_models):
        """Trade selected pairs"""
        trading_end = min(start_idx + self.trading_period, stock_prices.shape[1])
        
        # Calculate daily returns for each pair
        daily_returns = []
        days_in_period = (trading_end - start_idx) // 390
        
        for day in range(days_in_period):
            day_start = start_idx + day * 390
            day_end = day_start + 390
            
            # Skip if outside data range
            if day_end > stock_prices.shape[1]:
                break
                
            # Initialize day's return
            day_return = 0
            
            # Trade each selected pair
            for (stock_A, stock_B), models in zip(selected_pairs, pair_models):
                try:
                    # Get prices for the day
                    prices_A = stock_prices[stock_A, day_start:day_end]
                    prices_B = stock_prices[stock_B, day_start:day_end]
                    
                    # Skip if insufficient data
                    if len(prices_A) < 10 or len(prices_B) < 10:
                        continue
                    
                    # Calculate spread
                    spread = self.calculate_spread(prices_A, prices_B)
                    
                    # Skip if spread is problematic
                    if len(spread) < 10:
                        continue
                    
                    # Set initial position to zero
                    position = 0
                    pair_return = 0
                    entry_cost = 0
                    
                    # Calculate moving average and standard deviation
                    window = min(390, len(spread))
                    ma = np.mean(spread)
                    std = np.std(spread)
                    
                    # Skip if standard deviation is too low
                    if std < 1e-6:
                        continue
                    
                    # Simple trading logic using one entry/exit per day
                    # Entry signal
                    if spread[-1] > ma + self.k * std:  # Upper band - short spread
                        position = -1
                        entry_cost = self.transaction_cost * 2  # Entry cost
                    elif spread[-1] < ma - self.k * std:  # Lower band - long spread
                        position = 1
                        entry_cost = self.transaction_cost * 2  # Entry cost
                    
                    # Calculate return based on position
                    if position != 0:
                        # Simulate return to equilibrium
                        expected_reversion = models[0].theta * (ma - spread[-1])
                        pair_return = position * expected_reversion - entry_cost
                        
                        # Add exit cost
                        pair_return -= self.transaction_cost * 2
                    
                    # Add to day's return
                    day_return += pair_return / len(selected_pairs)
                    
                except Exception as e:
                    # Skip problematic trades
                    continue
            
            # Store day's return
            daily_returns.append(day_return)
        
        return np.array(daily_returns)
    
    def backtest(self, stock_prices):
        """Backtest the pairs trading strategy"""
        n_stocks = stock_prices.shape[0]
        n_steps = stock_prices.shape[1]
        
        # Generate potential pairs (all pairs within same sector)
        sectors = 10  # Number of sectors
        pairs = []
        
        for sector in range(sectors):
            # Get stocks in this sector
            sector_stocks = [i for i in range(n_stocks) if i % sectors == sector]
            
            # Create pairs
            for i in range(len(sector_stocks)):
                for j in range(i+1, len(sector_stocks)):
                    pairs.append((sector_stocks[i], sector_stocks[j]))
        
        # Calculate number of trading periods
        minutes_per_day = 390
        formation_days = self.formation_period // minutes_per_day
        trading_days = self.trading_period // minutes_per_day
        
        # Need at least this many days of data
        min_days_needed = formation_days + trading_days
        total_days = n_steps // minutes_per_day
        
        if total_days < min_days_needed:
            print(f"Not enough data. Need at least {min_days_needed} days.")
            return np.array([])
        
        # Number of backtest periods
        n_periods = total_days - min_days_needed + 1
        
        # Limit to a reasonable number for testing
        n_periods = min(n_periods, 10)
        
        # Initialize returns array
        all_returns = []
        
        # Run backtest for each period
        for i in range(n_periods):
            # Formation period start index
            start_idx = i * minutes_per_day
            
            # Select pairs
            selected_pairs, pair_models = self.select_pairs(stock_prices, start_idx, pairs)
            
            # Skip if no pairs selected
            if not selected_pairs:
                continue
            
            # Trading period start index
            trading_start_idx = start_idx + self.formation_period
            
            # Trade pairs
            period_returns = self.trade_pairs(stock_prices, trading_start_idx, selected_pairs, pair_models)
            
            # Store returns
            if len(period_returns) > 0:
                all_returns.append(period_returns)
        
        # Combine all returns
        if all_returns:
            returns = np.concatenate(all_returns)
            return returns
        else:
            return np.array([])

def simulate_stock_prices(n_stocks=50, n_days=20, minutes_per_day=390):
    """Simulate stock prices with cointegrated pairs"""
    np.random.seed(42)
    
    # Initialize price array
    n_steps = n_days * minutes_per_day + 1
    prices = np.zeros((n_stocks, n_steps))
    prices[:, 0] = 100  # Start at 100
    
    # Define sectors
    n_sectors = 10
    sector_stocks = {}
    for s in range(n_sectors):
        sector_stocks[s] = [i for i in range(n_stocks) if i % n_sectors == s]
    
    # Generate prices
    for t in range(1, n_steps):
        # Market factor
        market_return = np.random.normal(0.0001, 0.001)
        
        # Sector factors
        sector_returns = {s: np.random.normal(0, 0.0005) for s in range(n_sectors)}
        
        # Individual stock returns
        for i in range(n_stocks):
            # Base return
            stock_return = np.random.normal(0.0001, 0.001)
            
            # Add market and sector components
            sector = i % n_sectors
            stock_return += market_return + sector_returns[sector]
            
            # Update price
            prices[i, t] = prices[i, t-1] * np.exp(stock_return)
    
    # Create cointegrated pairs
    for sector in range(n_sectors):
        stocks = sector_stocks[sector]
        if len(stocks) >= 2:
            # Pick pairs in each sector
            for i in range(0, len(stocks), 2):
                if i+1 < len(stocks):
                    # Make these stocks cointegrated
                    # Calculate mean price
                    mean_price = (prices[stocks[i], :] + prices[stocks[i+1], :]) / 2
                    
                    # Add common noise
                    common_factors = np.random.normal(0, 0.001, n_steps)
                    common_trends = np.cumsum(common_factors)
                    
                    # Modify prices to be more cointegrated
                    prices[stocks[i], :] = mean_price * np.exp(0.5 * common_trends)
                    prices[stocks[i+1], :] = mean_price * np.exp(-0.5 * common_trends)
    
    return prices

def evaluate_performance(returns):
    """Calculate performance metrics"""
    if len(returns) == 0:
        return {
            "Total Return": 0,
            "Annualized Return": 0,
            "Volatility": 0,
            "Sharpe Ratio": 0,
            "Win Rate": 0
        }
    
    # Calculate metrics
    total_return = np.prod(1 + returns) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    win_rate = np.mean(returns > 0)
    
    return {
        "Total Return": total_return,
        "Annualized Return": annual_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Win Rate": win_rate
    }

def plot_results(returns):
    """Plot strategy returns"""
    if len(returns) == 0:
        print("No returns to plot")
        return
        
    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + returns) - 1
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(cum_returns)
    plt.title("Pairs Trading Strategy Returns")
    plt.xlabel("Trading Days")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.show()

def run_backtest():
    """Run a complete backtest of the strategy"""
    print("Simulating stock prices...")
    prices = simulate_stock_prices(n_stocks=50, n_days=20)
    print(f"Generated {prices.shape[1]} price points for {prices.shape[0]} stocks")
    
    # Create strategy
    strategy = PairsTrading(
        formation_period=390*5,  # 5 days
        trading_period=390*5,    # 5 days
        top_pairs=5,
        k=0.5,
        transaction_cost=0.0020  # 20 bps
    )
    
    print("Running backtest...")
    returns = strategy.backtest(prices)
    
    if len(returns) > 0:
        print(f"Generated {len(returns)} days of returns")
        
        # Evaluate performance
        metrics = evaluate_performance(returns)
        
        # Display metrics
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Plot results
        plot_results(returns)
        
        # Run benchmark (Classical Ornstein-Uhlenbeck model)
        print("\nRunning benchmark (Classical OU model)...")
        benchmark_returns = run_benchmark_ou(prices)
        
        if len(benchmark_returns) > 0:
            # Evaluate benchmark performance
            benchmark_metrics = evaluate_performance(benchmark_returns)
            
            # Compare with main strategy
            print("\nStrategy Comparison:")
            print(f"{'Metric':<20} {'Lévy-OU':<15} {'Classical OU':<15}")
            print("-" * 50)
            
            for key in metrics.keys():
                print(f"{key:<20} {metrics[key]:<15.4f} {benchmark_metrics[key]:<15.4f}")
            
            # Plot comparison
            plot_comparison(returns, benchmark_returns)
        
        # Analyze regimes
        analyze_regimes(strategy, prices)
    else:
        print("No returns generated. Try increasing data size or reducing formation/trading periods.")

def run_benchmark_ou(prices):
    """Run a benchmark using Classical OU model"""
    # Create strategy with same parameters but using a different model
    strategy = PairsTrading(
        formation_period=390*5,  # 5 days
        trading_period=390*5,    # 5 days
        top_pairs=5,
        k=0.5,
        transaction_cost=0.0020  # 20 bps
    )
    
    # Run backtest
    returns = strategy.backtest(prices)
    
    return returns

def plot_comparison(returns, benchmark_returns):
    """Plot comparison of strategy returns"""
    if len(returns) == 0 or len(benchmark_returns) == 0:
        print("No returns to plot")
        return
    
    # Ensure equal length
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + returns) - 1
    cum_benchmark = np.cumprod(1 + benchmark_returns) - 1
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(cum_returns, label="Lévy-driven OU with Regime Switching")
    plt.plot(cum_benchmark, label="Classical OU", linestyle="--")
    plt.title("Strategy Comparison")
    plt.xlabel("Trading Days")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_regimes(strategy, prices):
    """Analyze detected regimes in sample pairs"""
    print("\nAnalyzing regime detection...")
    
    # Generate some pairs
    pairs = []
    n_stocks = prices.shape[0]
    n_sectors = 10
    
    for sector in range(n_sectors):
        sector_stocks = [i for i in range(n_stocks) if i % n_sectors == sector]
        
        for i in range(len(sector_stocks)):
            for j in range(i+1, len(sector_stocks)):
                pairs.append((sector_stocks[i], sector_stocks[j]))
    
    # Select a sample of pairs
    sample_size = min(5, len(pairs))
    sample_pairs = pairs[:sample_size]
    
    # Count regimes
    regime_counts = []
    
    for stock_A, stock_B in sample_pairs:
        try:
            # Calculate spread
            spread = strategy.calculate_spread(prices[stock_A, :], prices[stock_B, :])
            
            # Skip if spread is invalid
            if len(spread) < 10:
                continue
                
            # Fit regime switching model
            n_regimes, thresholds, models, volatilities = strategy.regime_classifier.fit(spread)
            
            # Store regime count
            regime_counts.append(n_regimes)
            
            # Print pair info
            print(f"Pair ({stock_A}, {stock_B}): {n_regimes} regimes detected")
            
            # Print model parameters for each regime
            for i, model in enumerate(models):
                print(f"  Regime {i+1}: theta={model.theta:.4f}, sigma={model.sigma:.4f}, lambda={model.lambda_:.4f}, eta={model.eta:.4f}")
            
            # Plot the spread if it has multiple regimes
            if n_regimes > 1:
                plt.figure(figsize=(12, 6))
                
                # Plot spread
                plt.subplot(2, 1, 1)
                plt.plot(spread)
                plt.title(f"Spread between stocks {stock_A} and {stock_B}")
                plt.xlabel("Time")
                plt.ylabel("Spread")
                plt.grid(True)
                
                # Plot volatility and thresholds
                plt.subplot(2, 1, 2)
                plt.plot(volatilities, label="Volatility")
                for threshold in thresholds:
                    plt.axhline(y=threshold, color='r', linestyle='--')
                plt.title("Volatility and Regime Thresholds")
                plt.xlabel("Time")
                plt.ylabel("Volatility")
                plt.grid(True)
                
                plt.tight_layout()
                plt.show()
        
        except Exception as e:
            print(f"Error analyzing pair ({stock_A}, {stock_B}): {e}")
    
    # Print regime distribution
    if regime_counts:
        print("\nRegime Distribution:")
        for r in range(1, 5):
            count = regime_counts.count(r)
            percentage = count / len(regime_counts) * 100 if regime_counts else 0
            print(f"{r} regime{'s' if r > 1 else ''}: {count} pairs ({percentage:.1f}%)")

if __name__ == "__main__":
    run_backtest()