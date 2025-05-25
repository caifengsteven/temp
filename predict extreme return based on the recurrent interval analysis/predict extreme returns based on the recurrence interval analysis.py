import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdblp
import datetime as dt
from scipy import stats
from scipy.special import gamma, gammainc
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

class ExtremeReturnPredictor:
    """
    Predicts extreme returns using recurrence interval analysis as described in
    Jiang et al. (2018), "Short term prediction of extreme returns based on
    the recurrence interval analysis".
    """
    
    def __init__(self, return_type='negative', threshold_method='quantile', threshold_value=0.99):
        """
        Initialize the predictor
        
        Parameters:
        -----------
        return_type : str
            Type of returns to analyze: 'negative', 'positive', or 'absolute'
        threshold_method : str
            Method to determine extreme thresholds: 'quantile' or 'evt'
            'quantile' uses a fixed quantile of returns
            'evt' uses extreme value theory to determine the threshold
        threshold_value : float
            If threshold_method='quantile', this is the quantile value (e.g., 0.95, 0.99)
            If threshold_method='evt', this is not used
        """
        self.return_type = return_type
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        
        # Parameters to be estimated
        self.threshold = None
        self.extreme_indices = None
        self.recurrence_intervals = None
        self.distribution_params = None
        self.hazard_threshold = None
        
        # Performance metrics
        self.in_sample_metrics = None
        self.out_of_sample_metrics = None
    
    def fit(self, returns, theta=0.5, delta_t=1):
        """
        Fit the model to historical return data
        
        Parameters:
        -----------
        returns : pd.Series or np.array
            Daily returns to analyze
        theta : float, optional
            Weight parameter for balancing missing events and false alarms (0 to 1)
        delta_t : int, optional
            Time horizon for prediction (e.g., 1 day ahead)
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Convert returns to the desired type
        processed_returns = self._process_returns(returns)
        
        # Identify extreme returns
        self.extreme_indices = self._identify_extremes(processed_returns)
        
        # Calculate recurrence intervals
        self.recurrence_intervals = np.diff(self.extreme_indices)
        
        # Fit distribution to recurrence intervals
        self.distribution_params = self._fit_distribution()
        
        # Find optimal hazard threshold that maximizes usefulness
        self.hazard_threshold = self._optimize_hazard_threshold(processed_returns, theta, delta_t)
        
        # Calculate in-sample performance metrics
        self.in_sample_metrics = self._evaluate_predictions(processed_returns, self.extreme_indices, delta_t)
        
        return self
    
    def predict(self, new_returns, last_extreme_index=None, delta_t=1):
        """
        Predict extreme returns in new data
        
        Parameters:
        -----------
        new_returns : pd.Series or np.array
            New returns to predict extremes for
        last_extreme_index : int, optional
            Index of the last extreme event in the training data
            If None, the last extreme from training is used
        delta_t : int, optional
            Time horizon for prediction
            
        Returns:
        --------
        predictions : np.array
            Binary array where 1 indicates predicted extreme return
        """
        # Convert returns to the desired type
        processed_returns = self._process_returns(new_returns)
        
        # Identify actual extremes in new data (for evaluation)
        extreme_indices = self._identify_extremes(processed_returns)
        
        # Initialize prediction array
        predictions = np.zeros(len(processed_returns), dtype=int)
        
        # Initialize time since last extreme
        if last_extreme_index is None:
            t = 0  # If no information about previous extremes
        else:
            t = 1  # Start with 1 day elapsed
        
        # For each day, calculate hazard probability and make prediction
        for i in range(len(processed_returns)):
            # Calculate hazard probability
            hazard_prob = self._hazard_probability(t, delta_t)
            
            # Make prediction
            if hazard_prob > self.hazard_threshold:
                predictions[i] = 1
            
            # Update time since last extreme
            if i in extreme_indices:
                t = 1  # Reset counter after extreme
            else:
                t += 1  # Increment counter
        
        # Calculate out-of-sample performance metrics
        self.out_of_sample_metrics = self._evaluate_predictions(processed_returns, extreme_indices, delta_t)
        
        return predictions
    
    def _process_returns(self, returns):
        """Process returns according to the specified return type"""
        if self.return_type == 'negative':
            return -np.array(returns)  # Negative returns become positive
        elif self.return_type == 'positive':
            return np.array(returns)
        elif self.return_type == 'absolute':
            return np.abs(returns)
        else:
            raise ValueError(f"Unknown return type: {self.return_type}")
    
    def _identify_extremes(self, processed_returns):
        """Identify extreme returns based on the threshold method"""
        if self.threshold_method == 'quantile':
            self.threshold = np.quantile(processed_returns, self.threshold_value)
            return np.where(processed_returns >= self.threshold)[0]
        
        elif self.threshold_method == 'evt':
            # Implement extreme value theory threshold determination
            # Using the KS statistic minimization as described in the paper
            sorted_returns = np.sort(processed_returns)
            n = len(sorted_returns)
            
            # Try different thresholds and compute KS statistic
            ks_stats = []
            thresholds = []
            
            for i in range(n // 10, n - 10):  # Search in middle 80% of data
                threshold = sorted_returns[i]
                excess = sorted_returns[i:] - threshold
                
                # Estimate shape parameter using Hill estimator
                k = len(excess)
                log_excess = np.log(excess + 1e-10)  # Avoid log(0)
                gamma = 1 / np.mean(log_excess - log_excess.min())
                
                # Compute empirical CDF
                ecdf = np.arange(1, k + 1) / k
                
                # Compute theoretical CDF (GPD)
                tcdf = 1 - (1 + gamma * excess / excess.mean()) ** (-1/gamma)
                
                # Compute KS statistic
                ks_stat = np.max(np.abs(ecdf - tcdf))
                ks_stats.append(ks_stat)
                thresholds.append(threshold)
            
            # Find threshold with minimum KS statistic
            min_idx = np.argmin(ks_stats)
            self.threshold = thresholds[min_idx]
            return np.where(processed_returns >= self.threshold)[0]
        
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
    
    def _fit_distribution(self):
        """Fit q-exponential distribution to recurrence intervals"""
        # Calculate mean recurrence interval
        mean_recurrence = np.mean(self.recurrence_intervals)
        
        # Define likelihood function (negative log-likelihood to minimize)
        def neg_log_likelihood(q):
            if q <= 1 or q >= 1.5:
                return np.inf
            
            # Calculate lambda based on q and mean recurrence interval
            lambda_val = 1 / (mean_recurrence * (3 - 2*q))
            
            # Calculate log-likelihood
            n = len(self.recurrence_intervals)
            log_lik = n * np.log(lambda_val * (2 - q))
            log_lik -= (1 / (q - 1)) * np.sum(np.log(1 + (q - 1) * lambda_val * self.recurrence_intervals))
            
            return -log_lik
        
        # Find q that maximizes likelihood
        result = minimize_scalar(neg_log_likelihood, bounds=(1.001, 1.499), method='bounded')
        q = result.x
        
        # Calculate lambda based on q and mean recurrence interval
        lambda_val = 1 / (mean_recurrence * (3 - 2*q))
        
        return {'q': q, 'lambda': lambda_val}
    
    def _hazard_probability(self, t, delta_t=1):
        """
        Calculate hazard probability W(t|delta_t)
        
        The probability that an extreme event will occur within delta_t time units
        after t time units have passed since the last extreme event.
        """
        q = self.distribution_params['q']
        lambda_val = self.distribution_params['lambda']
        
        # q-exponential hazard probability (equation 10 in the paper)
        hazard_prob = 1 - ((1 + (q - 1) * lambda_val * (t + delta_t)) / 
                           (1 + (q - 1) * lambda_val * t)) ** (1 - 1/(q-1))
        
        return hazard_prob
    
    def _optimize_hazard_threshold(self, returns, theta=0.5, delta_t=1):
        """
        Find optimal hazard threshold that maximizes usefulness
        
        Parameters:
        -----------
        returns : np.array
            Processed returns
        theta : float
            Weight parameter for balancing Type I and Type II errors
        delta_t : int
            Time horizon for prediction
            
        Returns:
        --------
        optimal_threshold : float
            Optimal hazard threshold
        """
        # Create actual extreme event indicator
        actual_extremes = np.zeros(len(returns), dtype=int)
        actual_extremes[self.extreme_indices] = 1
        
        # Initialize variables for tracking best threshold
        best_usefulness = -np.inf
        best_threshold = 0
        
        # Calculate time elapsed since last extreme for each day
        elapsed_times = np.zeros(len(returns), dtype=int)
        last_extreme = -1000  # Start with a large negative value
        
        for i in range(len(returns)):
            if i in self.extreme_indices:
                last_extreme = i
                elapsed_times[i] = 0
            else:
                elapsed_times[i] = i - last_extreme
        
        # Calculate hazard probabilities for each day
        hazard_probs = np.array([self._hazard_probability(t, delta_t) for t in elapsed_times])
        
        # Try different thresholds
        thresholds = np.linspace(0, 1, 101)  # 0 to 1 in steps of 0.01
        
        for threshold in thresholds:
            # Make predictions
            predictions = (hazard_probs > threshold).astype(int)
            
            # Calculate performance metrics
            # We need to handle the prediction lag correctly
            # A prediction at time t is for an extreme within the next delta_t days
            shifted_predictions = np.zeros_like(predictions)
            for i in range(len(predictions) - delta_t):
                if predictions[i] == 1:
                    shifted_predictions[i:i+delta_t+1] = 1
            
            # Calculate Type I and Type II errors
            n11 = np.sum((actual_extremes == 1) & (shifted_predictions == 1))  # Correct predictions of extremes
            n01 = np.sum((actual_extremes == 1) & (shifted_predictions == 0))  # Missed events
            n10 = np.sum((actual_extremes == 0) & (shifted_predictions == 1))  # False alarms
            n00 = np.sum((actual_extremes == 0) & (shifted_predictions == 0))  # Correct predictions of non-extremes
            
            # Correct prediction rate D
            D = n11 / (n01 + n11) if (n01 + n11) > 0 else 0
            
            # False alarm rate A
            A = n10 / (n00 + n10) if (n00 + n10) > 0 else 0
            
            # Calculate usefulness (equation 14 in the paper)
            L = theta * (1 - D) + (1 - theta) * A
            U = min(theta, 1 - theta) - L
            
            # Update best threshold
            if U > best_usefulness:
                best_usefulness = U
                best_threshold = threshold
        
        return best_threshold
    
    def _evaluate_predictions(self, returns, extreme_indices, delta_t=1):
        """
        Evaluate prediction performance
        
        Parameters:
        -----------
        returns : np.array
            Processed returns
        extreme_indices : np.array
            Indices of extreme events
        delta_t : int
            Time horizon for prediction
            
        Returns:
        --------
        metrics : dict
            Dictionary of performance metrics
        """
        # Create actual extreme event indicator
        actual_extremes = np.zeros(len(returns), dtype=int)
        actual_extremes[extreme_indices] = 1
        
        # Calculate time elapsed since last extreme for each day
        elapsed_times = np.zeros(len(returns), dtype=int)
        last_extreme = -1000  # Start with a large negative value
        
        for i in range(len(returns)):
            if i in extreme_indices:
                last_extreme = i
                elapsed_times[i] = 0
            else:
                elapsed_times[i] = i - last_extreme
        
        # Calculate hazard probabilities for each day
        hazard_probs = np.array([self._hazard_probability(t, delta_t) for t in elapsed_times])
        
        # Make predictions
        predictions = (hazard_probs > self.hazard_threshold).astype(int)
        
        # Calculate performance metrics
        # We need to handle the prediction lag correctly
        # A prediction at time t is for an extreme within the next delta_t days
        shifted_predictions = np.zeros_like(predictions)
        for i in range(len(predictions) - delta_t):
            if predictions[i] == 1:
                shifted_predictions[i:i+delta_t+1] = 1
        
        # Calculate Type I and Type II errors
        n11 = np.sum((actual_extremes == 1) & (shifted_predictions == 1))  # Correct predictions of extremes
        n01 = np.sum((actual_extremes == 1) & (shifted_predictions == 0))  # Missed events
        n10 = np.sum((actual_extremes == 0) & (shifted_predictions == 1))  # False alarms
        n00 = np.sum((actual_extremes == 0) & (shifted_predictions == 0))  # Correct predictions of non-extremes
        
        # Correct prediction rate D
        D = n11 / (n01 + n11) if (n01 + n11) > 0 else 0
        
        # False alarm rate A
        A = n10 / (n00 + n10) if (n00 + n10) > 0 else 0
        
        # Calculate Hanssen-Kuiper skill score (KSS)
        KSS = D - A
        
        # Calculate usefulness
        L = 0.5 * (1 - D) + 0.5 * A  # Assuming theta = 0.5
        U = min(0.5, 0.5) - L
        
        return {
            'correct_prediction_rate': D,
            'false_alarm_rate': A,
            'KSS': KSS,
            'usefulness': U,
            'n11': n11,
            'n01': n01,
            'n10': n10,
            'n00': n00
        }
    
    def plot_recurrence_interval_distribution(self):
        """Plot the distribution of recurrence intervals"""
        if self.recurrence_intervals is None:
            raise ValueError("Model has not been fitted yet")
        
        plt.figure(figsize=(10, 6))
        
        # Empirical distribution
        counts, bins = np.histogram(self.recurrence_intervals, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.scatter(bin_centers, counts, label='Empirical', alpha=0.7)
        
        # Fitted q-exponential distribution
        q = self.distribution_params['q']
        lambda_val = self.distribution_params['lambda']
        x = np.linspace(0, max(self.recurrence_intervals), 1000)
        y = (2 - q) * lambda_val * (1 + (q - 1) * lambda_val * x) ** (-1/(q-1))
        plt.plot(x, y, 'r-', label=f'q-exponential (q={q:.3f}, λ={lambda_val:.4f})')
        
        plt.xlabel('Recurrence Interval')
        plt.ylabel('Probability Density')
        plt.title('Recurrence Interval Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_hazard_function(self, t_max=100):
        """Plot the hazard function W(t|1)"""
        if self.distribution_params is None:
            raise ValueError("Model has not been fitted yet")
        
        t_values = np.arange(1, t_max + 1)
        hazard_probs = np.array([self._hazard_probability(t, 1) for t in t_values])
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_values, hazard_probs, 'b-', linewidth=2)
        plt.axhline(y=self.hazard_threshold, color='r', linestyle='--', 
                   label=f'Hazard Threshold = {self.hazard_threshold:.3f}')
        plt.xlabel('Time Since Last Extreme (days)')
        plt.ylabel('Hazard Probability W(t|1)')
        plt.title('Hazard Function')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_roc_curve(self, returns, extreme_indices, delta_t=1):
        """Plot ROC curve for the predictions"""
        # Create actual extreme event indicator
        actual_extremes = np.zeros(len(returns), dtype=int)
        actual_extremes[extreme_indices] = 1
        
        # Calculate time elapsed since last extreme for each day
        elapsed_times = np.zeros(len(returns), dtype=int)
        last_extreme = -1000  # Start with a large negative value
        
        for i in range(len(returns)):
            if i in extreme_indices:
                last_extreme = i
                elapsed_times[i] = 0
            else:
                elapsed_times[i] = i - last_extreme
        
        # Calculate hazard probabilities for each day
        hazard_probs = np.array([self._hazard_probability(t, delta_t) for t in elapsed_times])
        
        # Handle the prediction lag correctly
        # A prediction at time t is for an extreme within the next delta_t days
        shifted_actuals = np.zeros_like(actual_extremes)
        for i in range(len(actual_extremes) - delta_t):
            if np.any(actual_extremes[i:i+delta_t+1] == 1):
                shifted_actuals[i] = 1
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(shifted_actuals, hazard_probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
        return roc_auc

def get_data_from_bloomberg(ticker, start_date, end_date):
    """
    Retrieve data from Bloomberg
    
    Parameters:
    -----------
    ticker : str
        Bloomberg ticker (e.g., 'SPX Index')
    start_date : str
        Start date in 'YYYYMMDD' format
    end_date : str
        End date in 'YYYYMMDD' format
        
    Returns:
    --------
    returns : pd.Series
        Daily returns
    """
    try:
        # Connect to Bloomberg
        print(f"Connecting to Bloomberg to retrieve {ticker} data...")
        conn = pdblp.BCon()
        conn.start()
        
        # Retrieve price data
        df = conn.bdh(ticker, ['PX_LAST'], start_date, end_date)
        
        # Calculate returns
        returns = df['PX_LAST'].pct_change().dropna()
        
        print(f"Retrieved {len(returns)} daily returns from {returns.index[0]} to {returns.index[-1]}")
        return returns
    
    except Exception as e:
        print(f"Error retrieving data from Bloomberg: {str(e)}")
        print("Using simulated data instead")
        
        # Generate simulated returns with realistic properties
        np.random.seed(42)
        n_days = 5000
        
        # Parameters for a realistic return series
        mu = 0.0005  # Daily drift (about 12% annual)
        sigma = 0.01  # Daily volatility (about 16% annual)
        
        # Generate normal returns with occasional jumps
        returns = np.random.normal(mu, sigma, n_days)
        
        # Add occasional jumps (crashes and rallies)
        n_jumps = 50
        jump_idx = np.random.choice(range(n_days), n_jumps, replace=False)
        jump_sizes = np.random.normal(-0.02, 0.05, n_jumps)  # Negative mean for more crashes
        returns[jump_idx] += jump_sizes
        
        # Create date index
        end_date_dt = dt.datetime.now()
        start_date_dt = end_date_dt - dt.timedelta(days=n_days)
        date_range = pd.date_range(start=start_date_dt, end=end_date_dt, periods=n_days)
        
        return pd.Series(returns, index=date_range)

def backtest_strategy(ticker='SPX Index', threshold_value=0.99, return_type='negative'):
    """
    Backtest the extreme return prediction strategy
    
    Parameters:
    -----------
    ticker : str
        Bloomberg ticker
    threshold_value : float
        Quantile threshold for extreme returns
    return_type : str
        Type of returns to analyze: 'negative', 'positive', or 'absolute'
        
    Returns:
    --------
    results : dict
        Dictionary of backtest results
    """
    # Get data
    returns = get_data_from_bloomberg(ticker, '20000101', '20201231')
    
    # Split into in-sample and out-of-sample periods
    # Use the first 80% as in-sample
    split_idx = int(len(returns) * 0.8)
    in_sample_returns = returns.iloc[:split_idx]
    out_sample_returns = returns.iloc[split_idx:]
    
    print(f"In-sample period: {in_sample_returns.index[0]} to {in_sample_returns.index[-1]}")
    print(f"Out-of-sample period: {out_sample_returns.index[0]} to {out_sample_returns.index[-1]}")
    
    # Create and fit the predictor
    predictor = ExtremeReturnPredictor(
        return_type=return_type,
        threshold_method='quantile',
        threshold_value=threshold_value
    )
    
    predictor.fit(in_sample_returns)
    
    # Make out-of-sample predictions
    predictions = predictor.predict(out_sample_returns)
    
    # Display performance metrics
    print("\nIn-sample performance:")
    print(f"Correct prediction rate: {predictor.in_sample_metrics['correct_prediction_rate']:.4f}")
    print(f"False alarm rate: {predictor.in_sample_metrics['false_alarm_rate']:.4f}")
    print(f"KSS score: {predictor.in_sample_metrics['KSS']:.4f}")
    print(f"Usefulness: {predictor.in_sample_metrics['usefulness']:.4f}")
    
    print("\nOut-of-sample performance:")
    print(f"Correct prediction rate: {predictor.out_of_sample_metrics['correct_prediction_rate']:.4f}")
    print(f"False alarm rate: {predictor.out_of_sample_metrics['false_alarm_rate']:.4f}")
    print(f"KSS score: {predictor.out_of_sample_metrics['KSS']:.4f}")
    print(f"Usefulness: {predictor.out_of_sample_metrics['usefulness']:.4f}")
    
    # Plot recurrence interval distribution
    predictor.plot_recurrence_interval_distribution()
    
    # Plot hazard function
    predictor.plot_hazard_function()
    
    # Plot ROC curve for out-of-sample predictions
    processed_returns = predictor._process_returns(out_sample_returns)
    extreme_indices = predictor._identify_extremes(processed_returns)
    roc_auc = predictor.plot_roc_curve(processed_returns, extreme_indices)
    
    # Simulate trading strategy
    portfolio_value = 100.0  # Initial portfolio value
    position = 0  # 0: no position, -1: short position
    values = [portfolio_value]
    equity_curve = []
    
    for i in range(len(out_sample_returns)):
        # If we have a warning signal and no position, go short
        if predictions[i] == 1 and position == 0:
            position = -1
        
        # If we have an extreme event, close any position
        if processed_returns[i] >= predictor.threshold and position == -1:
            position = 0
        
        # Update portfolio value
        if position == -1:
            portfolio_value *= (1 - out_sample_returns.iloc[i])  # Short position profits from price decreases
        
        values.append(portfolio_value)
        equity_curve.append((out_sample_returns.index[i], portfolio_value))
    
    # Calculate strategy performance
    returns_df = pd.DataFrame({
        'Strategy': np.diff(np.log(values)),
        'Buy&Hold': out_sample_returns.values
    })
    
    strategy_return = (values[-1] / values[0]) - 1
    strategy_annual_return = (1 + strategy_return) ** (252 / len(out_sample_returns)) - 1
    strategy_volatility = returns_df['Strategy'].std() * np.sqrt(252)
    strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility > 0 else 0
    
    buy_hold_return = (1 + out_sample_returns).prod() - 1
    buy_hold_annual_return = (1 + buy_hold_return) ** (252 / len(out_sample_returns)) - 1
    buy_hold_volatility = returns_df['Buy&Hold'].std() * np.sqrt(252)
    buy_hold_sharpe = buy_hold_annual_return / buy_hold_volatility if buy_hold_volatility > 0 else 0
    
    # Plot equity curve
    equity_dates = [date for date, _ in equity_curve]
    equity_values = [value for _, value in equity_curve]
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_dates, equity_values, label='Strategy')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Equity Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Return results
    results = {
        'in_sample_metrics': predictor.in_sample_metrics,
        'out_of_sample_metrics': predictor.out_of_sample_metrics,
        'strategy_return': strategy_return,
        'strategy_annual_return': strategy_annual_return,
        'strategy_volatility': strategy_volatility,
        'strategy_sharpe': strategy_sharpe,
        'buy_hold_return': buy_hold_return,
        'buy_hold_annual_return': buy_hold_annual_return,
        'buy_hold_volatility': buy_hold_volatility,
        'buy_hold_sharpe': buy_hold_sharpe,
        'roc_auc': roc_auc
    }
    
    print("\nTrading Strategy Performance:")
    print(f"Strategy Total Return: {strategy_return:.2%}")
    print(f"Strategy Annual Return: {strategy_annual_return:.2%}")
    print(f"Strategy Annual Volatility: {strategy_volatility:.2%}")
    print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
    print(f"Buy & Hold Total Return: {buy_hold_return:.2%}")
    print(f"Buy & Hold Annual Return: {buy_hold_annual_return:.2%}")
    print(f"Buy & Hold Annual Volatility: {buy_hold_volatility:.2%}")
    print(f"Buy & Hold Sharpe Ratio: {buy_hold_sharpe:.2f}")
    
    return results

def backtest_improved_strategy(ticker='SPX Index', threshold_value=0.99, return_type='negative'):
    """
    Backtest an improved version of the extreme return prediction strategy
    
    Parameters:
    -----------
    ticker : str
        Bloomberg ticker
    threshold_value : float
        Quantile threshold for extreme returns
    return_type : str
        Type of returns to analyze: 'negative', 'positive', or 'absolute'
        
    Returns:
    --------
    results : dict
        Dictionary of backtest results
    """
    # Get data
    returns = get_data_from_bloomberg(ticker, '20000101', '20201231')
    
    # Split into in-sample and out-of-sample periods
    split_idx = int(len(returns) * 0.8)
    in_sample_returns = returns.iloc[:split_idx]
    out_sample_returns = returns.iloc[split_idx:]
    
    print(f"In-sample period: {in_sample_returns.index[0]} to {in_sample_returns.index[-1]}")
    print(f"Out-of-sample period: {out_sample_returns.index[0]} to {out_sample_returns.index[-1]}")
    
    # Create and fit the predictor
    predictor = ExtremeReturnPredictor(
        return_type=return_type,
        threshold_method='quantile',
        threshold_value=threshold_value
    )
    
    predictor.fit(in_sample_returns)
    
    # Get processed returns and actual extreme events for analysis
    processed_returns = predictor._process_returns(out_sample_returns)
    extreme_indices = predictor._identify_extremes(processed_returns)
    
    # Calculate time elapsed since last extreme for each day
    elapsed_times = np.zeros(len(processed_returns), dtype=int)
    last_extreme = -1000  # Start with a large negative value
    
    for i in range(len(processed_returns)):
        if i in extreme_indices:
            last_extreme = i
            elapsed_times[i] = 0
        else:
            elapsed_times[i] = i - last_extreme
    
    # Calculate hazard probabilities for each day
    hazard_probs = np.array([predictor._hazard_probability(t, 1) for t in elapsed_times])
    
    # Create a DataFrame for the backtest
    backtest_df = pd.DataFrame({
        'Returns': out_sample_returns.values,
        'ProcessedReturns': processed_returns,
        'HazardProb': hazard_probs,
        'IsExtreme': np.isin(np.arange(len(processed_returns)), extreme_indices).astype(int)
    })
    
    # Add signal column (1 = warning signal)
    backtest_df['Signal'] = (backtest_df['HazardProb'] > predictor.hazard_threshold).astype(int)
    
    # Add lagged signals and returns for analysis
    for i in range(1, 6):  # Add 5 lags
        backtest_df[f'Signal_lag{i}'] = backtest_df['Signal'].shift(i)
        backtest_df[f'Returns_lag{i}'] = backtest_df['Returns'].shift(i)
    
    # Drop NaN rows
    backtest_df = backtest_df.dropna()
    
    # Analyze returns after signals
    signal_returns = backtest_df.loc[backtest_df['Signal'] == 1, 'Returns'].mean()
    no_signal_returns = backtest_df.loc[backtest_df['Signal'] == 0, 'Returns'].mean()
    
    print(f"\nAverage return after signal: {signal_returns:.4%}")
    print(f"Average return with no signal: {no_signal_returns:.4%}")
    
    # Check if signals correctly predict extreme events
    correct_signals = np.mean(backtest_df.loc[backtest_df['Signal'] == 1, 'IsExtreme'])
    print(f"Proportion of signals followed by extreme events: {correct_signals:.4f}")
    
    # ------------------- IMPROVED STRATEGY IMPLEMENTATION -------------------
    
    # Strategy 1: Basic signal-based strategy with stop-loss and take-profit
    backtest_df['Strategy1_Position'] = 0
    backtest_df['Strategy1_Value'] = 100.0  # Start with $100
    transaction_cost = 0.001  # 10 bps per trade
    
    position = 0
    entry_price = 0
    stop_loss = 0.05  # 5% stop loss
    take_profit = 0.03  # 3% take profit
    
    for i in range(1, len(backtest_df)):
        prev_value = backtest_df.iloc[i-1]['Strategy1_Value']
        
        # Check for exit conditions if we have a position
        if position != 0:
            # Calculate current P&L
            if position == -1:  # Short position
                pnl = (entry_price - (entry_price * (1 + backtest_df.iloc[i]['Returns']))) / entry_price
            else:  # Long position
                pnl = ((entry_price * (1 + backtest_df.iloc[i]['Returns'])) - entry_price) / entry_price
            
            # Check stop loss/take profit
            if pnl <= -stop_loss or pnl >= take_profit:
                # Close position
                if position == -1:
                    backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy1_Value')] = \
                        prev_value * (1 - backtest_df.iloc[i]['Returns'] - transaction_cost)
                else:
                    backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy1_Value')] = \
                        prev_value * (1 + backtest_df.iloc[i]['Returns'] - transaction_cost)
                position = 0
                continue
        
        # No position - check for entry signal
        if position == 0:
            # Strong signal for short (high hazard probability)
            if backtest_df.iloc[i]['HazardProb'] > predictor.hazard_threshold * 1.2:
                position = -1
                entry_price = 100  # Arbitrary price level for calculating P&L
                # Enter short position
                backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy1_Position')] = -1
                backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy1_Value')] = \
                    prev_value * (1 - transaction_cost)  # Apply transaction cost
            # Low hazard probability might indicate upside potential
            elif backtest_df.iloc[i]['HazardProb'] < predictor.hazard_threshold * 0.5:
                position = 1
                entry_price = 100
                # Enter long position
                backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy1_Position')] = 1
                backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy1_Value')] = \
                    prev_value * (1 - transaction_cost)  # Apply transaction cost
            else:
                # No position change
                backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy1_Position')] = 0
                backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy1_Value')] = prev_value
        
        # Update position value
        if position == -1:  # Short position
            backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy1_Value')] = \
                prev_value * (1 - backtest_df.iloc[i]['Returns'])
        elif position == 1:  # Long position
            backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy1_Value')] = \
                prev_value * (1 + backtest_df.iloc[i]['Returns'])
    
    # Strategy 2: Position sizing based on hazard probability
    backtest_df['Strategy2_Position'] = 0
    backtest_df['Strategy2_Value'] = 100.0
    
    position = 0
    max_position = -1.0  # Maximum short position size
    
    for i in range(1, len(backtest_df)):
        prev_value = backtest_df.iloc[i-1]['Strategy2_Value']
        hazard_prob = backtest_df.iloc[i]['HazardProb']
        
        # Scale position size based on hazard probability
        if hazard_prob > predictor.hazard_threshold:
            # Calculate position size between 0 and max_position
            position_size = max_position * min(1, (hazard_prob - predictor.hazard_threshold) / 
                                             (1 - predictor.hazard_threshold))
            
            # If position changes, apply transaction cost
            if position != position_size:
                transaction_cost_amount = abs(position - position_size) * transaction_cost
                backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy2_Value')] = \
                    prev_value * (1 - transaction_cost_amount)
                position = position_size
            else:
                backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy2_Value')] = prev_value
        else:
            # No signal - reduce position if we have one
            if position != 0:
                transaction_cost_amount = abs(position) * transaction_cost
                backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy2_Value')] = \
                    prev_value * (1 - transaction_cost_amount)
                position = 0
            else:
                backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy2_Value')] = prev_value
        
        # Record position
        backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy2_Position')] = position
        
        # Update position value based on return
        if position != 0:
            backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy2_Value')] = \
                backtest_df.iloc[i]['Strategy2_Value'] * (1 + position * backtest_df.iloc[i]['Returns'])
    
    # Strategy 3: Combined approach with volatility adjustment
    backtest_df['Strategy3_Position'] = 0
    backtest_df['Strategy3_Value'] = 100.0
    
    # Calculate rolling volatility
    backtest_df['Rolling_Vol'] = backtest_df['Returns'].rolling(20).std() * np.sqrt(252)
    backtest_df['Vol_Scaled_Position'] = 0
    
    position = 0
    max_position = -1.0
    vol_target = 0.15  # Target annualized volatility
    
    for i in range(20, len(backtest_df)):  # Start after we have enough data for volatility
        prev_value = backtest_df.iloc[i-1]['Strategy3_Value']
        hazard_prob = backtest_df.iloc[i]['HazardProb']
        current_vol = backtest_df.iloc[i]['Rolling_Vol']
        
        # Calculate base position size based on hazard probability
        if hazard_prob > predictor.hazard_threshold:
            # Calculate position size between 0 and max_position
            base_position_size = max_position * min(1, (hazard_prob - predictor.hazard_threshold) / 
                                                  (1 - predictor.hazard_threshold))
            
            # Scale position based on volatility
            if current_vol > 0:
                vol_scaled_position = base_position_size * (vol_target / current_vol)
                # Limit position size for risk management
                position_size = max(min(vol_scaled_position, max_position), -1)
            else:
                position_size = base_position_size
        else:
            position_size = 0
        
        # If position changes, apply transaction cost
        if position != position_size:
            transaction_cost_amount = abs(position - position_size) * transaction_cost
            backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy3_Value')] = \
                prev_value * (1 - transaction_cost_amount)
            position = position_size
        else:
            backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy3_Value')] = prev_value
        
        # Record position
        backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy3_Position')] = position
        backtest_df.iloc[i, backtest_df.columns.get_loc('Vol_Scaled_Position')] = position_size
        
        # Update position value based on return
        if position != 0:
            backtest_df.iloc[i, backtest_df.columns.get_loc('Strategy3_Value')] = \
                backtest_df.iloc[i]['Strategy3_Value'] * (1 + position * backtest_df.iloc[i]['Returns'])
    
    # Calculate buy and hold performance for comparison
    backtest_df['BuyHold_Value'] = 100 * (1 + backtest_df['Returns']).cumprod()
    
    # Calculate performance metrics
    strategies = ['Strategy1_Value', 'Strategy2_Value', 'Strategy3_Value', 'BuyHold_Value']
    metrics = {}
    
    for strat in strategies:
        # Skip first value (initial value)
        values = backtest_df[strat].values[1:]
        returns = np.diff(np.log(values))
        
        # Calculate metrics
        total_return = (values[-1] / values[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Store metrics
        metrics[strat] = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
    
    # Print performance comparison
    print("\nStrategy Performance Comparison:")
    print(f"{'Strategy':<15} {'Total Return':<15} {'Annual Return':<15} {'Volatility':<15} {'Sharpe':<10} {'Max DD':<10}")
    print("-" * 80)
    
    for strat, stats in metrics.items():
        strat_name = strat.split('_')[0]
        print(f"{strat_name:<15} {stats['Total Return']:14.2%} {stats['Annual Return']:14.2%} "
              f"{stats['Annual Volatility']:14.2%} {stats['Sharpe Ratio']:9.2f} {stats['Max Drawdown']:9.2%}")
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    for strat in strategies:
        plt.plot(backtest_df.index, backtest_df[strat], label=strat.split('_')[0])
    
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Strategy Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot positions over time for Strategy 3
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(backtest_df.index, backtest_df['Returns'], 'b-', alpha=0.5)
    plt.scatter(backtest_df.index[backtest_df['IsExtreme'] == 1], 
                backtest_df.loc[backtest_df['IsExtreme'] == 1, 'Returns'],
                color='r', marker='o', label='Extreme Returns')
    plt.ylabel('Returns')
    plt.title('Returns and Extreme Events')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(backtest_df.index, backtest_df['HazardProb'], 'g-', label='Hazard Probability')
    plt.axhline(y=predictor.hazard_threshold, color='r', linestyle='--', 
                label=f'Threshold={predictor.hazard_threshold:.3f}')
    plt.ylabel('Probability')
    plt.title('Hazard Probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(backtest_df.index, backtest_df['Strategy3_Position'], 'b-', label='Position Size')
    plt.ylabel('Position')
    plt.title('Strategy 3 Position')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate hit ratio (accuracy of signals in predicting direction)
    correct_direction = np.sum((backtest_df['Signal'] == 1) & (backtest_df['Returns'] < 0)) + \
                       np.sum((backtest_df['Signal'] == 0) & (backtest_df['Returns'] >= 0))
    hit_ratio = correct_direction / len(backtest_df)
    
    print(f"\nSignal hit ratio (direction prediction accuracy): {hit_ratio:.4f}")
    
    # Return detailed results
    return {
        'metrics': metrics,
        'backtest_df': backtest_df,
        'predictor': predictor
    }

def analyze_hazard_probability_effectiveness(results):
    """
    Analyze the relationship between hazard probability and subsequent returns
    
    Parameters:
    -----------
    results : dict
        Results from backtest_improved_strategy
        
    Returns:
    --------
    None
    """
    backtest_df = results['backtest_df']
    predictor = results['predictor']
    
    # Create bins of hazard probability
    bins = np.linspace(0, 1, 11)  # 10 equal bins from 0 to 1
    backtest_df['HazardBin'] = pd.cut(backtest_df['HazardProb'], bins, labels=False)
    
    # Calculate average return for each bin
    bin_returns = backtest_df.groupby('HazardBin')['Returns'].mean()
    bin_counts = backtest_df.groupby('HazardBin').size()
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate percentage of extreme events in each bin
    bin_extremes = backtest_df.groupby('HazardBin')['IsExtreme'].mean()
    
    # Plot relationship
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: Average returns vs hazard probability
    ax1.bar(bin_centers, bin_returns, width=0.08, alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.axvline(x=predictor.hazard_threshold, color='g', linestyle='--', 
                label=f'Threshold={predictor.hazard_threshold:.3f}')
    ax1.set_ylabel('Average Return')
    ax1.set_title('Average Return by Hazard Probability')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Count of observations in each bin
    ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Number of Observations by Hazard Probability')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Percentage of extreme events in each bin
    ax3.bar(bin_centers, bin_extremes, width=0.08, alpha=0.7)
    ax3.set_xlabel('Hazard Probability')
    ax3.set_ylabel('Extreme Event Rate')
    ax3.set_title('Percentage of Extreme Events by Hazard Probability')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation between hazard probability and future returns
    correlation = backtest_df['HazardProb'].corr(backtest_df['Returns'])
    print(f"Correlation between hazard probability and next-day return: {correlation:.4f}")
    
    # Calculate correlation between hazard probability and extreme events
    extreme_corr = backtest_df['HazardProb'].corr(backtest_df['IsExtreme'])
    print(f"Correlation between hazard probability and extreme events: {extreme_corr:.4f}")
    
    # Look at returns after highest hazard probabilities
    high_hazard = backtest_df[backtest_df['HazardProb'] > 0.9]
    if len(high_hazard) > 0:
        avg_return = high_hazard['Returns'].mean()
        extreme_rate = high_hazard['IsExtreme'].mean()
        print(f"\nAverage return after very high hazard probability (>0.9): {avg_return:.4%}")
        print(f"Extreme event rate after very high hazard probability: {extreme_rate:.4f}")
    
    # Create a forward-looking measure: maximum drawdown in next N days
    n_days = 5
    backtest_df['Forward_Max_Drawdown'] = 0.0
    
    for i in range(len(backtest_df) - n_days):
        future_returns = backtest_df['Returns'].iloc[i+1:i+n_days+1]
        cumulative_returns = (1 + future_returns).cumprod() - 1
        max_drawdown = min(0, np.min(cumulative_returns))
        backtest_df.iloc[i, backtest_df.columns.get_loc('Forward_Max_Drawdown')] = max_drawdown
    
    # Analyze relationship between hazard probability and future drawdowns
    drawdown_corr = backtest_df['HazardProb'].corr(backtest_df['Forward_Max_Drawdown'])
    print(f"Correlation between hazard probability and {n_days}-day forward max drawdown: {drawdown_corr:.4f}")
    
    # Plot hazard probability vs forward drawdown
    plt.figure(figsize=(10, 6))
    plt.scatter(backtest_df['HazardProb'], backtest_df['Forward_Max_Drawdown'], alpha=0.3)
    plt.xlabel('Hazard Probability')
    plt.ylabel(f'{n_days}-Day Forward Max Drawdown')
    plt.title('Hazard Probability vs. Future Drawdowns')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate optimal hazard threshold for predicting drawdowns
    thresholds = np.linspace(0, 1, 101)
    best_score = -np.inf
    best_threshold = 0
    
    for threshold in thresholds:
        # Predict significant drawdowns (e.g., > 2%)
        predictions = (backtest_df['HazardProb'] > threshold).astype(int)
        actuals = (backtest_df['Forward_Max_Drawdown'] < -0.02).astype(int)
        
        # Calculate KSS score
        if sum(actuals) > 0 and sum(predictions) > 0:
            tp = sum((predictions == 1) & (actuals == 1))
            fp = sum((predictions == 1) & (actuals == 0))
            tn = sum((predictions == 0) & (actuals == 0))
            fn = sum((predictions == 0) & (actuals == 1))
            
            if tp + fn > 0 and tn + fp > 0:
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                kss = sensitivity + specificity - 1
                
                if kss > best_score:
                    best_score = kss
                    best_threshold = threshold
    
    print(f"\nOptimal hazard threshold for predicting significant drawdowns: {best_threshold:.4f}")
    print(f"KSS score at optimal threshold: {best_score:.4f}")
    
    return None

def run_multi_period_analysis():
    """Run analysis on multiple historical crisis periods as described in the paper"""
    # Define time periods from the paper
    periods = [
        ('Wall Street Crash', '1885-01-01', '1929-01-01', '1929-01-01', '1932-12-31'),
        ('Oil Crisis', '1885-01-01', '1973-01-01', '1973-01-01', '1975-12-31'),
        ('Black Monday', '1885-01-01', '1987-01-01', '1987-01-01', '1989-12-31'),
        ('Dot-com Bubble', '1885-01-01', '2000-01-01', '2000-01-01', '2003-12-31'),
        ('Financial Crisis', '1885-01-01', '2007-01-01', '2007-01-01', '2009-12-31'),
        ('European Debt Crisis', '1885-01-01', '2011-01-01', '2011-01-01', '2015-12-31')
    ]
    
    # Define different return types and threshold values to test
    return_types = ['negative', 'positive', 'absolute']
    threshold_values = [0.95, 0.975, 0.99]
    
    # Get DJIA data (or use simulated data if Bloomberg is not available)
    try:
        full_returns = get_data_from_bloomberg('INDU Index', '18850101', '20151231')
    except:
        print("Using simulated data for DJIA")
        np.random.seed(42)
        dates = pd.date_range(start='1885-01-01', end='2015-12-31', freq='B')
        returns = pd.Series(np.random.normal(0.0002, 0.01, len(dates)), index=dates)
        
        # Add crashes and bubbles
        crash_periods = [
            ('1929-10-24', '1932-07-08', -0.03),  # Great Depression
            ('1973-01-01', '1974-12-31', -0.015),  # Oil Crisis
            ('1987-10-19', '1987-10-19', -0.22),  # Black Monday
            ('2000-03-10', '2002-10-09', -0.02),  # Dot-com Bubble
            ('2007-10-09', '2009-03-09', -0.025),  # Financial Crisis
            ('2011-07-01', '2011-10-03', -0.018)   # European Debt Crisis
        ]
        
        for start, end, drift in crash_periods:
            mask = (returns.index >= start) & (returns.index <= end)
            crash_length = mask.sum()
            if crash_length > 0:
                returns.loc[mask] = np.random.normal(drift, 0.02, crash_length)
        
        full_returns = returns
    
    # Create results table
    results = {}
    
    # Analyze each period
    for period_name, in_start, in_end, out_start, out_end in periods:
        print(f"\n{'-'*80}\nAnalyzing {period_name}")
        
        # Filter data for this period
        in_sample_returns = full_returns[(full_returns.index >= in_start) & (full_returns.index < in_end)]
        out_sample_returns = full_returns[(full_returns.index >= out_start) & (full_returns.index <= out_end)]
        
        print(f"In-sample: {in_sample_returns.index[0]} to {in_sample_returns.index[-1]} ({len(in_sample_returns)} observations)")
        print(f"Out-of-sample: {out_sample_returns.index[0]} to {out_sample_returns.index[-1]} ({len(out_sample_returns)} observations)")
        
        period_results = {}
        
        # Test different combinations
        for return_type in return_types:
            for threshold in threshold_values:
                key = f"{return_type}_{threshold}"
                print(f"\nTesting {return_type} returns with {threshold*100}% threshold")
                
                # Create and fit predictor
                predictor = ExtremeReturnPredictor(
                    return_type=return_type,
                    threshold_method='quantile',
                    threshold_value=threshold
                )
                
                try:
                    predictor.fit(in_sample_returns)
                    predictions = predictor.predict(out_sample_returns)
                    
                    # Store metrics
                    period_results[key] = {
                        'in_sample': predictor.in_sample_metrics,
                        'out_of_sample': predictor.out_of_sample_metrics,
                        'q': predictor.distribution_params['q'],
                        'lambda': predictor.distribution_params['lambda'],
                        'threshold': predictor.threshold,
                        'hazard_threshold': predictor.hazard_threshold
                    }
                    
                    print(f"In-sample KSS: {predictor.in_sample_metrics['KSS']:.4f}")
                    print(f"Out-of-sample KSS: {predictor.out_of_sample_metrics['KSS']:.4f}")
                    
                except Exception as e:
                    print(f"Error in analysis: {str(e)}")
                    period_results[key] = {'error': str(e)}
        
        results[period_name] = period_results
    
    # Create summary table
    summary = pd.DataFrame(
        columns=['Period', 'Return Type', 'Threshold', 
                 'In-Sample D', 'In-Sample A', 'In-Sample KSS', 'In-Sample U',
                 'Out-of-Sample D', 'Out-of-Sample A', 'Out-of-Sample KSS', 'Out-of-Sample U']
    )
    
    row = 0
    for period, period_results in results.items():
        for key, metrics in period_results.items():
            if 'error' in metrics:
                continue
                
            return_type, threshold = key.split('_')
            threshold = float(threshold)
            
            summary.loc[row] = [
                period, 
                return_type, 
                threshold,
                metrics['in_sample']['correct_prediction_rate'],
                metrics['in_sample']['false_alarm_rate'],
                metrics['in_sample']['KSS'],
                metrics['in_sample']['usefulness'],
                metrics['out_of_sample']['correct_prediction_rate'],
                metrics['out_of_sample']['false_alarm_rate'],
                metrics['out_of_sample']['KSS'],
                metrics['out_of_sample']['usefulness']
            ]
            row += 1
    
    # Display summary
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 180)
    print("\nSummary of Results:")
    print(summary)
    
    # Find best combinations
    best_in_sample = summary.loc[summary['In-Sample KSS'].idxmax()]
    best_out_sample = summary.loc[summary['Out-of-Sample KSS'].idxmax()]
    
    print("\nBest In-Sample Performance:")
    print(best_in_sample)
    
    print("\nBest Out-of-Sample Performance:")
    print(best_out_sample)
    
    return results, summary

if __name__ == "__main__":
    # Run individual backtest with the original strategy
    print("Running original backtest for S&P 500 with negative returns and 99% threshold...")
    backtest_strategy(ticker='SPX Index', threshold_value=0.99, return_type='negative')
    
    # Run improved backtest with better risk management and position sizing
    print("\nRunning improved backtest for S&P 500 with negative returns and 99% threshold...")
    results = backtest_improved_strategy(ticker='SPX Index', threshold_value=0.99, return_type='negative')
    
    # Analyze the effectiveness of hazard probabilities in predicting returns
    print("\nAnalyzing effectiveness of hazard probabilities...")
    analyze_hazard_probability_effectiveness(results)
    
    # Uncomment to run comprehensive analysis of all periods
    # print("\nRunning comprehensive analysis of all historical periods...")
    # results, summary = run_multi_period_analysis()