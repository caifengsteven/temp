import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from scipy.special import gamma
from scipy import optimize
import time
from tqdm import tqdm
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')

class EnsembleProperties:
    """
    Implementation of the ensemble trading model described in the paper:
    "Ensemble properties of high-frequency data and intraday trading rules"
    by F. Baldovin, F. Camana, M. Caporin, M. Caraglio, and A.L. Stella
    """
    
    def __init__(self, Dm=0.35, Da=1.31, alpha=3.29, beta_m=2.5e-3, beta_a=7.5e-5, t_star=20):
        """
        Initialize the model with the parameters.
        
        Parameters:
        -----------
        Dm : float
            Morning scaling exponent
        Da : float
            Afternoon scaling exponent
        alpha : float
            Form parameter of the Student's t-distribution
        beta_m : float
            Morning scale parameter
        beta_a : float
            Afternoon scale parameter
        t_star : int
            Time index for the transition from morning to afternoon model
        """
        self.Dm = Dm
        self.Da = Da
        self.alpha = alpha
        self.beta_m = beta_m
        self.beta_a = beta_a
        self.t_star = t_star
        
    def calculate_a_t(self, t):
        """
        Calculate a_t coefficients using equation (13) in the paper.
        
        Parameters:
        -----------
        t : int
            Time index
            
        Returns:
        --------
        float : a_t coefficient
        """
        D_t = self.Dm if t <= self.t_star else self.Da
        return np.sqrt(t**(2*D_t) - (t-1)**(2*D_t))
    
    def calculate_beta_t(self, t):
        """
        Calculate beta_t using equation (16) in the paper.
        
        Parameters:
        -----------
        t : int
            Time index
            
        Returns:
        --------
        float : beta_t scale parameter
        """
        return self.beta_m if t <= self.t_star else self.beta_a
    
    def lambda_function(self, tau, t_star=None):
        """
        Calculate lambda function using equation (19) in the paper.
        
        Parameters:
        -----------
        tau : int
            Aggregation time
        t_star : int, optional
            Transition time point, defaults to self.t_star
            
        Returns:
        --------
        float : lambda value
        """
        if t_star is None:
            t_star = self.t_star
            
        if tau <= t_star:
            return self.beta_m * (tau ** self.Dm)
        else:
            term1 = (self.beta_m ** 2) * (t_star ** (2 * self.Dm))
            term2 = (self.beta_a ** 2) * (tau ** (2 * self.Da) - t_star ** (2 * self.Da))
            return np.sqrt(term1 + term2)
    
    def generate_returns(self, n_days=1000, n_periods=36):
        """
        Generate returns based on the model.
        
        Parameters:
        -----------
        n_days : int
            Number of days to simulate
        n_periods : int
            Number of periods per day
            
        Returns:
        --------
        numpy.ndarray : Array of simulated returns with shape (n_days, n_periods)
        """
        returns = np.zeros((n_days, n_periods))
        
        for day in range(n_days):
            # Sample the volatility from inverse-gamma distribution
            sigma = np.random.gamma(shape=self.alpha/2, scale=1)
            sigma = 1.0 / np.sqrt(sigma)
            
            # Generate the returns using equation (11) in the paper
            for t in range(1, n_periods + 1):
                a_t = self.calculate_a_t(t)
                beta_t = self.calculate_beta_t(t)
                returns[day, t-1] = np.random.normal(0, sigma * a_t * beta_t)
                
        return returns
    
    def calculate_aggregated_returns(self, returns):
        """
        Calculate aggregated returns for all possible horizons.
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Array of returns with shape (n_days, n_periods)
            
        Returns:
        --------
        dict : Dictionary with aggregated returns for each horizon
        """
        n_days, n_periods = returns.shape
        agg_returns = {}
        
        for tau in range(1, n_periods + 1):
            agg_ret = np.zeros((n_days, n_periods - tau + 1))
            for i in range(n_periods - tau + 1):
                agg_ret[:, i] = np.sum(returns[:, i:i+tau], axis=1)
            agg_returns[tau] = agg_ret
            
        return agg_returns
    
    def student_t_pdf(self, x, alpha, scale=1.0):
        """
        Calculate Student's t-distribution PDF with given parameters.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Points at which to evaluate the PDF
        alpha : float
            Form parameter
        scale : float, optional
            Scale parameter, defaults to 1.0
            
        Returns:
        --------
        numpy.ndarray : PDF values
        """
        coef = gamma((alpha+1)/2) / (np.sqrt(np.pi) * gamma(alpha/2))
        return coef * (1 + (x/scale)**2)**(-(alpha+1)/2) / scale
    
    def student_t_cdf(self, x, alpha, scale=1.0):
        """
        Calculate Student's t-distribution CDF with given parameters.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Points at which to evaluate the CDF
        alpha : float
            Form parameter
        scale : float, optional
            Scale parameter, defaults to 1.0
            
        Returns:
        --------
        numpy.ndarray : CDF values
        """
        # Use scipy's t distribution with appropriate parameters
        df = alpha
        return t_dist.cdf(x/scale, df)
    
    def calculate_quantile(self, q, tau, t_star=None):
        """
        Calculate quantiles of the aggregated returns distribution.
        
        Parameters:
        -----------
        q : float
            Quantile level (between 0 and 1)
        tau : int
            Aggregation time
        t_star : int, optional
            Transition time point, defaults to self.t_star
            
        Returns:
        --------
        float : Quantile value
        """
        scale = self.lambda_function(tau, t_star)
        # Use scipy's t distribution with appropriate parameters
        df = self.alpha
        return t_dist.ppf(q, df) * scale
    
    def conditional_pdf(self, r, tau, past_returns, tp):
        """
        Calculate conditional PDF of future returns given past returns.
        
        Parameters:
        -----------
        r : float or numpy.ndarray
            Return values at which to evaluate the PDF
        tau : int
            Time horizon for the future returns
        past_returns : numpy.ndarray
            Array of past returns
        tp : int
            Number of past returns to condition on
            
        Returns:
        --------
        float or numpy.ndarray : Conditional PDF values
        """
        if tp == 0:
            # No conditioning, use simple PDF
            scale = self.lambda_function(tau)
            return self.student_t_pdf(r, self.alpha, scale)
        
        # Use equation (28) in the paper for conditional PDF
        numerator_term = 0
        for t in range(tp):
            a_t = self.calculate_a_t(t+1)
            beta_t = self.calculate_beta_t(t+1)
            numerator_term += (past_returns[t] / (a_t * beta_t))**2
            
        denominator_term = 0
        for t in range(tp, tp+tau):
            a_t = self.calculate_a_t(t+1)
            beta_t = self.calculate_beta_t(t+1)
            denominator_term += (a_t * beta_t)**2
            
        # Calculate the conditional PDF based on equation (28)
        coef = gamma((self.alpha + tp + 1)/2) / (np.sqrt(np.pi) * gamma((self.alpha + tp)/2))
        
        if np.isscalar(r):
            z = r / np.sqrt(denominator_term) * np.sqrt(1 + numerator_term)
            pdf_value = coef * (1 + z**2)**(-(self.alpha + tp + 1)/2) / np.sqrt(denominator_term)
            return pdf_value
        else:
            z = r / np.sqrt(denominator_term) * np.sqrt(1 + numerator_term)
            pdf_values = coef * (1 + z**2)**(-(self.alpha + tp + 1)/2) / np.sqrt(denominator_term)
            return pdf_values
    
    def conditional_quantile(self, q, tau, past_returns, tp):
        """
        Calculate conditional quantiles given past returns.
        
        Parameters:
        -----------
        q : float
            Quantile level (between 0 and 1)
        tau : int
            Time horizon for the future returns
        past_returns : numpy.ndarray
            Array of past returns
        tp : int
            Number of past returns to condition on
            
        Returns:
        --------
        float : Conditional quantile value
        """
        if tp == 0:
            # No conditioning, use simple quantile
            return self.calculate_quantile(q, tau)
        
        # Calculate terms for the transformation
        numerator_term = 0
        for t in range(tp):
            a_t = self.calculate_a_t(t+1)
            beta_t = self.calculate_beta_t(t+1)
            numerator_term += (past_returns[t] / (a_t * beta_t))**2
            
        denominator_term = 0
        for t in range(tp, tp+tau):
            a_t = self.calculate_a_t(t+1)
            beta_t = self.calculate_beta_t(t+1)
            denominator_term += (a_t * beta_t)**2
            
        # Use scipy's t distribution with appropriate parameters
        df = self.alpha + tp
        z = t_dist.ppf(q, df)
        
        # Transform back using equation (31)
        return z * np.sqrt(denominator_term) / np.sqrt(1 + numerator_term)
    
    def price_barriers(self, initial_price, q, returns=None, tp=0):
        """
        Calculate price barriers for trading.
        
        Parameters:
        -----------
        initial_price : float
            Initial price at the start of the trading day
        q : float
            Quantile level for the barriers (between 0 and 0.5)
        returns : numpy.ndarray, optional
            Array of past returns if conditioning is used
        tp : int, optional
            Number of past returns to condition on, defaults to 0
            
        Returns:
        --------
        tuple : (lower_barriers, upper_barriers) arrays with price levels
        """
        n_periods = 36  # Total periods in a day
        lower_barriers = np.zeros(n_periods)
        upper_barriers = np.zeros(n_periods)
        
        if tp > 0 and returns is None:
            raise ValueError("Past returns must be provided when tp > 0")
        
        for t in range(tp+1, n_periods+1):
            tau = t
            
            if tp > 0:
                # Use conditional quantiles
                past_ret = returns[:tp]
                lower_q = self.conditional_quantile(q, tau-tp, past_ret, tp)
                upper_q = self.conditional_quantile(1-q, tau-tp, past_ret, tp)
            else:
                # Use unconditional quantiles
                lower_q = self.calculate_quantile(q, tau)
                upper_q = self.calculate_quantile(1-q, tau)
            
            # Calculate price barriers
            lower_barriers[t-1] = initial_price * np.exp(lower_q)
            upper_barriers[t-1] = initial_price * np.exp(upper_q)
            
        return lower_barriers, upper_barriers

    def simulate_prices(self, initial_price, returns):
        """
        Convert returns to price series.
        
        Parameters:
        -----------
        initial_price : float
            Initial price at the start of the trading day
        returns : numpy.ndarray
            Array of returns
            
        Returns:
        --------
        numpy.ndarray : Price series
        """
        n_days, n_periods = returns.shape
        prices = np.zeros((n_days, n_periods+1))
        prices[:, 0] = initial_price
        
        for t in range(n_periods):
            prices[:, t+1] = prices[:, t] * np.exp(returns[:, t])
            
        return prices


class GJR_GARCH_Model:
    """
    Implements the GJR-GARCH model used as a benchmark in the paper.
    """
    
    def __init__(self, n_periods=36):
        """
        Initialize the GJR-GARCH model.
        
        Parameters:
        -----------
        n_periods : int
            Number of periods per day
        """
        self.n_periods = n_periods
        self.periodic_component = np.ones(n_periods)
        self.omega = 0.01
        self.alpha = 0.05
        self.gamma = 0.1
        self.beta = 0.8
        
    def fit(self, returns):
        """
        Fit the GJR-GARCH model to the returns data.
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Array of returns with shape (n_days, n_periods)
            
        Returns:
        --------
        bool : True if fitting was successful
        """
        n_days, n_periods = returns.shape
        
        # Estimate the periodic component
        for t in range(n_periods):
            self.periodic_component[t] = np.std(returns[:, t])**2
            
        # Normalize the periodic component
        self.periodic_component = self.periodic_component / np.mean(self.periodic_component)
        
        # Deseasonalize the returns
        deseasonalized_returns = returns.copy()
        for t in range(n_periods):
            deseasonalized_returns[:, t] = returns[:, t] / np.sqrt(self.periodic_component[t])
            
        # Flatten the returns to a single time series
        flat_returns = deseasonalized_returns.flatten()
        
        try:
            # Fit a GJR-GARCH(1,1,1) model
            model = arch_model(flat_returns, p=1, o=1, q=1, vol='GARCH', dist='normal')
            results = model.fit(disp='off', show_warning=False)
            
            # Extract parameters
            self.omega = results.params['omega']
            self.alpha = results.params['alpha[1]']
            self.gamma = results.params['gamma[1]']
            self.beta = results.params['beta[1]']
            
            return True
        except:
            print("GJR-GARCH model fitting failed. Using default parameters.")
            return False
    
    def simulate_variance(self, n_days=1, n_ahead=36):
        """
        Simulate the variance process for n_days.
        
        Parameters:
        -----------
        n_days : int
            Number of days to simulate
        n_ahead : int
            Number of periods ahead to forecast
            
        Returns:
        --------
        numpy.ndarray : Simulated variance with shape (n_days, n_ahead)
        """
        variance = np.zeros((n_days, n_ahead))
        
        # Start with unconditional variance
        h0 = self.omega / (1 - self.alpha - self.gamma / 2 - self.beta)
        
        for day in range(n_days):
            # Simulate the variance process
            h = h0
            for t in range(n_ahead):
                # Generate a random shock
                z = np.random.normal(0, 1)
                r = np.sqrt(h) * z
                
                # Update the variance for the next period
                asymmetric = 1 if r < 0 else 0
                h = self.omega + (self.alpha + self.gamma * asymmetric) * r**2 + self.beta * h
                
                # Store the variance
                variance[day, t] = h * self.periodic_component[t]
        
        return variance
    
    def quantile_forecast(self, q, initial_price, n_ahead=36):
        """
        Forecast quantiles for the price distribution.
        
        Parameters:
        -----------
        q : float
            Quantile level (between 0 and 1)
        initial_price : float
            Initial price at the start of the trading day
        n_ahead : int
            Number of periods ahead to forecast
            
        Returns:
        --------
        tuple : (lower_quantiles, upper_quantiles) arrays with shape (n_ahead)
        """
        # Simulate the variance process
        variance = self.simulate_variance(1, n_ahead)[0]  # Just one day
        
        # Calculate the quantiles for a normal distribution
        from scipy.stats import norm
        z_q = norm.ppf(q)
        z_1_q = norm.ppf(1 - q)
        
        # Calculate the return quantiles
        lower_return_quantiles = np.zeros(n_ahead)
        upper_return_quantiles = np.zeros(n_ahead)
        
        # Calculate cumulative returns for each time horizon
        for t in range(n_ahead):
            # For each t, we need to calculate the distribution of the cumulative return
            # from time 0 to time t
            if t == 0:
                # Single period return distribution
                vol = np.sqrt(variance[0])
                lower_return_quantiles[0] = z_q * vol
                upper_return_quantiles[0] = z_1_q * vol
            else:
                # For multi-period returns, we need to simulate paths
                n_sims = 10000
                cum_returns = np.zeros(n_sims)
                
                for sim in range(n_sims):
                    # Simulate a path of returns
                    path_returns = np.zeros(t+1)
                    
                    for i in range(t+1):
                        vol = np.sqrt(variance[i])
                        path_returns[i] = np.random.normal(0, vol)
                    
                    # Calculate the cumulative return
                    cum_returns[sim] = np.sum(path_returns)
                
                # Calculate the quantiles of the cumulative returns
                lower_return_quantiles[t] = np.percentile(cum_returns, q*100)
                upper_return_quantiles[t] = np.percentile(cum_returns, (1-q)*100)
        
        # Convert to price quantiles
        lower_price_quantiles = initial_price * np.exp(lower_return_quantiles)
        upper_price_quantiles = initial_price * np.exp(upper_return_quantiles)
        
        return lower_price_quantiles, upper_price_quantiles
    
    def price_barriers(self, initial_price, q, returns=None, tp=0):
        """
        Calculate price barriers for trading.
        
        Parameters:
        -----------
        initial_price : float
            Initial price at the start of the trading day
        q : float
            Quantile level for the barriers (between 0 and 0.5)
        returns : numpy.ndarray, optional
            Array of past returns if conditioning is used (not used in GJR-GARCH)
        tp : int, optional
            Number of past returns to condition on (not used in GJR-GARCH)
            
        Returns:
        --------
        tuple : (lower_barriers, upper_barriers) arrays with price levels
        """
        # The GJR-GARCH model doesn't use conditioning in the same way
        # as the ensemble model, so we ignore tp and returns
        lower_barriers, upper_barriers = self.quantile_forecast(q, initial_price)
        
        return lower_barriers, upper_barriers


class TradingStrategy:
    """
    Implements the trading strategy described in the paper.
    """
    
    def __init__(self, model, quantile=0.1, tp=0, initial_capital=1000000):
        """
        Initialize the trading strategy.
        
        Parameters:
        -----------
        model : EnsembleProperties or GJR_GARCH_Model
            The model used for density forecasts
        quantile : float
            Quantile level for the trading barriers (0 < quantile < 0.5)
        tp : int
            Number of past returns to condition on
        initial_capital : float
            Initial capital for the trading simulation
        """
        self.model = model
        self.quantile = quantile
        self.tp = tp
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.trades = []
        self.equity_curve = [initial_capital]
        
    def reset(self):
        """Reset the strategy to initial state."""
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
    def execute_trading_day(self, day_prices, day_returns):
        """
        Execute trading for a single day.
        
        Parameters:
        -----------
        day_prices : numpy.ndarray
            Array of prices for the day
        day_returns : numpy.ndarray
            Array of returns for the day
            
        Returns:
        --------
        float : Daily profit/loss
        """
        initial_price = day_prices[0]
        
        # Calculate the price barriers based on the quantile
        if self.tp > 0 and len(day_returns) >= self.tp:
            lower_barriers, upper_barriers = self.model.price_barriers(
                initial_price, self.quantile, returns=day_returns, tp=self.tp)
        else:
            lower_barriers, upper_barriers = self.model.price_barriers(
                initial_price, self.quantile)
            
        # Start from position = 0 at the beginning of each day
        self.position = 0
        self.entry_price = 0
        day_pnl = 0
        
        # Trading logic
        for t in range(self.tp+1, len(day_prices)-1):
            price = day_prices[t]
            prev_price = day_prices[t-1]
            
            # Check for trading signals
            if self.position == 0:
                # Buy signal: price breaks above upper barrier
                if price > upper_barriers[t-1] and prev_price < upper_barriers[t-2]:
                    self.position = 1
                    self.entry_price = price
                    trade_size = 0.9 * self.capital / price
                    
                # Sell signal: price breaks below lower barrier
                elif price < lower_barriers[t-1] and prev_price > lower_barriers[t-2]:
                    self.position = -1
                    self.entry_price = price
                    trade_size = 0.9 * self.capital / price
                    
            elif self.position == 1:  # Long position
                # Close signal: price breaks below upper barrier
                if price < upper_barriers[t-1] and prev_price > upper_barriers[t-2]:
                    pnl = (price - self.entry_price) * trade_size
                    self.capital += pnl
                    day_pnl += pnl
                    self.equity_curve.append(self.capital)
                    
                    # Record the trade
                    self.trades.append({
                        'type': 'long',
                        'entry_price': self.entry_price,
                        'exit_price': price,
                        'profit': pnl,
                        'profit_bp': pnl / (trade_size * self.entry_price) * 10000  # basis points
                    })
                    
                    self.position = 0
                    self.entry_price = 0
                    
            elif self.position == -1:  # Short position
                # Close signal: price breaks above lower barrier
                if price > lower_barriers[t-1] and prev_price < lower_barriers[t-2]:
                    pnl = (self.entry_price - price) * trade_size
                    self.capital += pnl
                    day_pnl += pnl
                    self.equity_curve.append(self.capital)
                    
                    # Record the trade
                    self.trades.append({
                        'type': 'short',
                        'entry_price': self.entry_price,
                        'exit_price': price,
                        'profit': pnl,
                        'profit_bp': pnl / (trade_size * self.entry_price) * 10000  # basis points
                    })
                    
                    self.position = 0
                    self.entry_price = 0
                    
        # Close any open position at the end of the day
        if self.position != 0:
            price = day_prices[-1]
            
            if self.position == 1:  # Long position
                pnl = (price - self.entry_price) * trade_size
            else:  # Short position
                pnl = (self.entry_price - price) * trade_size
                
            self.capital += pnl
            day_pnl += pnl
            self.equity_curve.append(self.capital)
            
            # Record the trade
            self.trades.append({
                'type': 'long' if self.position == 1 else 'short',
                'entry_price': self.entry_price,
                'exit_price': price,
                'profit': pnl,
                'profit_bp': pnl / (trade_size * self.entry_price) * 10000,  # basis points
                'forced_close': True
            })
            
            self.position = 0
            self.entry_price = 0
            
        return day_pnl
    
    def backtest(self, prices, returns):
        """
        Backtest the trading strategy.
        
        Parameters:
        -----------
        prices : numpy.ndarray
            Array of prices with shape (n_days, n_periods+1)
        returns : numpy.ndarray
            Array of returns with shape (n_days, n_periods)
            
        Returns:
        --------
        dict : Performance metrics
        """
        self.reset()
        n_days = prices.shape[0]
        daily_pnl = np.zeros(n_days)
        
        for day in tqdm(range(n_days), desc="Backtesting"):
            day_prices = prices[day]
            day_returns = returns[day]
            daily_pnl[day] = self.execute_trading_day(day_prices, day_returns)
            
        # Calculate performance metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        trade_returns = np.array([trade['profit_bp'] for trade in self.trades])
        
        long_trades = [t for t in self.trades if t['type'] == 'long']
        short_trades = [t for t in self.trades if t['type'] == 'short']
        
        long_returns = np.array([t['profit_bp'] for t in long_trades]) if long_trades else np.array([])
        short_returns = np.array([t['profit_bp'] for t in short_trades]) if short_trades else np.array([])
        
        win_rate = np.mean(trade_returns > 0) if len(trade_returns) > 0 else 0
        
        metrics = {
            'total_return': total_return * 100,  # in percent
            'annual_return': total_return * 252 / n_days * 100,  # in percent, assuming 252 trading days per year
            'sharpe_ratio': (np.mean(daily_pnl) / np.std(daily_pnl)) * np.sqrt(252) if np.std(daily_pnl) > 0 else 0,
            'num_trades': len(self.trades),
            'win_rate': win_rate * 100,  # in percent
            'avg_trade_return_bp': np.mean(trade_returns) if len(trade_returns) > 0 else 0,
            'avg_long_return_bp': np.mean(long_returns) if len(long_returns) > 0 else 0,
            'avg_short_return_bp': np.mean(short_returns) if len(short_returns) > 0 else 0,
            'equity_curve': np.array(self.equity_curve),
            'trades': self.trades
        }
        
        return metrics


def simulate_sp500_data(n_days=1000, n_periods=36, initial_price=2000, 
                        vol=0.002, mean_reversion=0.7, small_correlation=0.05):
    """
    Simulate S&P 500-like price data with small correlations.
    
    Parameters:
    -----------
    n_days : int
        Number of days to simulate
    n_periods : int
        Number of periods per day
    initial_price : float
        Initial price level
    vol : float
        Base volatility level
    mean_reversion : float
        Mean reversion parameter for volatility clustering
    small_correlation : float
        Small correlation between consecutive returns
    
    Returns:
    --------
    tuple : (prices, returns) arrays
    """
    prices = np.zeros((n_days, n_periods+1))
    returns = np.zeros((n_days, n_periods))
    
    # Initialize prices with initial_price
    prices[:, 0] = initial_price
    
    # Volatility pattern: decreasing in morning, increasing in afternoon
    vol_pattern = np.zeros(n_periods)
    t_star = 20
    for t in range(n_periods):
        if t < t_star:
            # Morning: decreasing volatility
            vol_pattern[t] = 1.2 - 0.5 * (t / t_star)
        else:
            # Afternoon: increasing volatility
            vol_pattern[t] = 0.7 + 0.5 * ((t - t_star) / (n_periods - t_star))
    
    # Simulate returns with small correlation and volatility clustering
    for day in range(n_days):
        # Start with a random volatility level for the day
        day_vol = np.random.gamma(shape=2.0, scale=0.5) * vol
        
        for t in range(n_periods):
            # Add small correlation with previous return
            if t > 0:
                corr_term = small_correlation * returns[day, t-1]
            else:
                corr_term = 0
            
            # Generate the return with small correlation and volatility clustering
            returns[day, t] = corr_term + day_vol * vol_pattern[t] * np.random.normal(0, 1)
            
            # Mean reversion in volatility
            day_vol = mean_reversion * day_vol + (1 - mean_reversion) * vol * (1 + 0.2 * np.random.normal(0, 1))
            
            # Update price
            prices[day, t+1] = prices[day, t] * np.exp(returns[day, t])
    
    return prices, returns


def evaluate_model_performance(ensemble_model, gjr_model, prices, returns, quantiles=[0.05, 0.1, 0.25]):
    """
    Evaluate and compare the performance of the ensemble model and GJR-GARCH model.
    
    Parameters:
    -----------
    ensemble_model : EnsembleProperties
        The ensemble model
    gjr_model : GJR_GARCH_Model
        The GJR-GARCH model
    prices : numpy.ndarray
        Array of prices
    returns : numpy.ndarray
        Array of returns
    quantiles : list
        List of quantile levels to test
        
    Returns:
    --------
    dict : Performance metrics for both models
    """
    results = {}
    
    # Test unconditioned trading
    for q in quantiles:
        print(f"\nTesting unconditioned trading with quantile {q}...")
        # Ensemble model
        ensemble_strategy = TradingStrategy(ensemble_model, quantile=q, tp=0)
        ensemble_metrics = ensemble_strategy.backtest(prices, returns)
        
        # GJR-GARCH model
        gjr_strategy = TradingStrategy(gjr_model, quantile=q, tp=0)
        gjr_metrics = gjr_strategy.backtest(prices, returns)
        
        results[f'Unconditioned_Q{int(q*100)}'] = {
            'Ensemble': ensemble_metrics,
            'GJR-GARCH': gjr_metrics
        }
    
    # Test conditioned trading with 3 points
    for q in quantiles:
        print(f"\nTesting conditioned trading with quantile {q}...")
        # Ensemble model
        ensemble_strategy = TradingStrategy(ensemble_model, quantile=q, tp=3)
        ensemble_metrics = ensemble_strategy.backtest(prices, returns)
        
        results[f'Conditioned3_Q{int(q*100)}'] = {
            'Ensemble': ensemble_metrics
        }
    
    return results


def plot_results(results):
    """
    Plot the results of the model evaluation.
    
    Parameters:
    -----------
    results : dict
        Performance metrics from evaluate_model_performance
    """
    # Plot average profit per trade
    plt.figure(figsize=(12, 8))
    
    # Extract data for the plot
    quantiles = [5, 10, 25]
    uncond_ensemble = [results[f'Unconditioned_Q{q}']['Ensemble']['avg_trade_return_bp'] for q in quantiles]
    cond_ensemble = [results[f'Conditioned3_Q{q}']['Ensemble']['avg_trade_return_bp'] for q in quantiles]
    uncond_gjr = [results[f'Unconditioned_Q{q}']['GJR-GARCH']['avg_trade_return_bp'] for q in quantiles]
    
    width = 0.25
    x = np.arange(len(quantiles))
    
    plt.bar(x - width, uncond_ensemble, width, label='Unconditioned Ensemble')
    plt.bar(x, cond_ensemble, width, label='Conditioned Ensemble (3 points)')
    plt.bar(x + width, uncond_gjr, width, label='GJR-GARCH')
    
    plt.xlabel('Quantile Level (%)')
    plt.ylabel('Average Profit per Trade (basis points)')
    plt.title('Trading Strategy Performance Comparison')
    plt.xticks(x, quantiles)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('profit_per_trade.png')
    plt.show()
    
    # Plot equity curves for 25% quantile
    plt.figure(figsize=(12, 8))
    
    ensemble_equity = results['Unconditioned_Q25']['Ensemble']['equity_curve']
    cond_ensemble_equity = results['Conditioned3_Q25']['Ensemble']['equity_curve']
    gjr_equity = results['Unconditioned_Q25']['GJR-GARCH']['equity_curve']
    
    plt.plot(ensemble_equity, label='Unconditioned Ensemble')
    plt.plot(cond_ensemble_equity, label='Conditioned Ensemble (3 points)')
    plt.plot(gjr_equity, label='GJR-GARCH')
    
    plt.xlabel('Trade Number')
    plt.ylabel('Equity')
    plt.title('Equity Curves (25% Quantile)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('equity_curves.png')
    plt.show()
    
    # Plot number of trades
    plt.figure(figsize=(12, 8))
    
    uncond_ensemble_trades = [results[f'Unconditioned_Q{q}']['Ensemble']['num_trades'] for q in quantiles]
    cond_ensemble_trades = [results[f'Conditioned3_Q{q}']['Ensemble']['num_trades'] for q in quantiles]
    uncond_gjr_trades = [results[f'Unconditioned_Q{q}']['GJR-GARCH']['num_trades'] for q in quantiles]
    
    plt.bar(x - width, uncond_ensemble_trades, width, label='Unconditioned Ensemble')
    plt.bar(x, cond_ensemble_trades, width, label='Conditioned Ensemble (3 points)')
    plt.bar(x + width, uncond_gjr_trades, width, label='GJR-GARCH')
    
    plt.xlabel('Quantile Level (%)')
    plt.ylabel('Number of Trades')
    plt.title('Number of Trades Comparison')
    plt.xticks(x, quantiles)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('number_of_trades.png')
    plt.show()
    
    # Print summary statistics
    print("Summary Statistics:")
    print("==================")
    
    for q in [25, 10, 5]:
        print(f"\nQuantile {q}%:")
        print("-" * 15)
        
        print("Unconditioned Ensemble:")
        print(f"  Annual Return: {results[f'Unconditioned_Q{q}']['Ensemble']['annual_return']:.2f}%")
        print(f"  Sharpe Ratio: {results[f'Unconditioned_Q{q}']['Ensemble']['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {results[f'Unconditioned_Q{q}']['Ensemble']['win_rate']:.2f}%")
        print(f"  Average Trade Return: {results[f'Unconditioned_Q{q}']['Ensemble']['avg_trade_return_bp']:.2f} bp")
        print(f"  Number of Trades: {results[f'Unconditioned_Q{q}']['Ensemble']['num_trades']}")
        
        print("\nConditioned Ensemble (3 points):")
        print(f"  Annual Return: {results[f'Conditioned3_Q{q}']['Ensemble']['annual_return']:.2f}%")
        print(f"  Sharpe Ratio: {results[f'Conditioned3_Q{q}']['Ensemble']['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {results[f'Conditioned3_Q{q}']['Ensemble']['win_rate']:.2f}%")
        print(f"  Average Trade Return: {results[f'Conditioned3_Q{q}']['Ensemble']['avg_trade_return_bp']:.2f} bp")
        print(f"  Number of Trades: {results[f'Conditioned3_Q{q}']['Ensemble']['num_trades']}")
        
        print("\nGJR-GARCH:")
        print(f"  Annual Return: {results[f'Unconditioned_Q{q}']['GJR-GARCH']['annual_return']:.2f}%")
        print(f"  Sharpe Ratio: {results[f'Unconditioned_Q{q}']['GJR-GARCH']['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {results[f'Unconditioned_Q{q}']['GJR-GARCH']['win_rate']:.2f}%")
        print(f"  Average Trade Return: {results[f'Unconditioned_Q{q}']['GJR-GARCH']['avg_trade_return_bp']:.2f} bp")
        print(f"  Number of Trades: {results[f'Unconditioned_Q{q}']['GJR-GARCH']['num_trades']}")
        
    # Save the summary to a file
    with open('trading_results_summary.txt', 'w') as f:
        f.write("Summary Statistics:\n")
        f.write("==================\n")
        
        for q in [25, 10, 5]:
            f.write(f"\nQuantile {q}%:\n")
            f.write("-" * 15 + "\n")
            
            f.write("Unconditioned Ensemble:\n")
            f.write(f"  Annual Return: {results[f'Unconditioned_Q{q}']['Ensemble']['annual_return']:.2f}%\n")
            f.write(f"  Sharpe Ratio: {results[f'Unconditioned_Q{q}']['Ensemble']['sharpe_ratio']:.2f}\n")
            f.write(f"  Win Rate: {results[f'Unconditioned_Q{q}']['Ensemble']['win_rate']:.2f}%\n")
            f.write(f"  Average Trade Return: {results[f'Unconditioned_Q{q}']['Ensemble']['avg_trade_return_bp']:.2f} bp\n")
            f.write(f"  Number of Trades: {results[f'Unconditioned_Q{q}']['Ensemble']['num_trades']}\n")
            
            f.write("\nConditioned Ensemble (3 points):\n")
            f.write(f"  Annual Return: {results[f'Conditioned3_Q{q}']['Ensemble']['annual_return']:.2f}%\n")
            f.write(f"  Sharpe Ratio: {results[f'Conditioned3_Q{q}']['Ensemble']['sharpe_ratio']:.2f}\n")
            f.write(f"  Win Rate: {results[f'Conditioned3_Q{q}']['Ensemble']['win_rate']:.2f}%\n")
            f.write(f"  Average Trade Return: {results[f'Conditioned3_Q{q}']['Ensemble']['avg_trade_return_bp']:.2f} bp\n")
            f.write(f"  Number of Trades: {results[f'Conditioned3_Q{q}']['Ensemble']['num_trades']}\n")
            
            f.write("\nGJR-GARCH:\n")
            f.write(f"  Annual Return: {results[f'Unconditioned_Q{q}']['GJR-GARCH']['annual_return']:.2f}%\n")
            f.write(f"  Sharpe Ratio: {results[f'Unconditioned_Q{q}']['GJR-GARCH']['sharpe_ratio']:.2f}\n")
            f.write(f"  Win Rate: {results[f'Unconditioned_Q{q}']['GJR-GARCH']['win_rate']:.2f}%\n")
            f.write(f"  Average Trade Return: {results[f'Unconditioned_Q{q}']['GJR-GARCH']['avg_trade_return_bp']:.2f} bp\n")
            f.write(f"  Number of Trades: {results[f'Unconditioned_Q{q}']['GJR-GARCH']['num_trades']}\n")


def main():
    """Main function to run the simulation and analysis."""
    
    print("Simulating S&P 500 price data...")
    prices, returns = simulate_sp500_data(n_days=500, n_periods=36, initial_price=2000)
    
    print("Initializing models...")
    # Initialize the ensemble model with parameters from the paper
    ensemble_model = EnsembleProperties(
        Dm=0.35, Da=1.31, alpha=3.29, beta_m=2.5e-3, beta_a=7.5e-5, t_star=20
    )
    
    # Initialize and fit the GJR-GARCH model
    gjr_model = GJR_GARCH_Model(n_periods=36)
    gjr_model.fit(returns)
    
    print("Evaluating model performance...")
    results = evaluate_model_performance(ensemble_model, gjr_model, prices, returns)
    
    print("Plotting results...")
    plot_results(results)


if __name__ == "__main__":
    main()