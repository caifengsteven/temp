import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

class UIPStrategy:
    """
    Implements a trading strategy based on the Uncovered Interest Rate Parity (UIP) relationship
    with adjustments for risk premiums, expectational errors, and time-varying parameters.
    """
    
    def __init__(self, window_size=60, decay_factor=0.95, risk_aversion=1.0):
        """
        Initialize the UIP strategy
        
        Parameters:
        -----------
        window_size: int
            Rolling window size for parameter estimation
        decay_factor: float
            Weight decay factor for past observations (0 < decay_factor <= 1)
        risk_aversion: float
            Risk aversion parameter for position sizing
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.risk_aversion = risk_aversion
        
    def calculate_weights(self):
        """Calculate exponentially decaying weights for observations"""
        weights = np.array([self.decay_factor ** i for i in range(self.window_size)])
        return weights / weights.sum()
    
    def estimate_beta(self, y, x, weights=None):
        """
        Estimate the beta coefficient using weighted least squares
        
        Parameters:
        -----------
        y: array-like
            Exchange rate changes
        x: array-like
            Interest rate differentials
        weights: array-like, optional
            Observation weights
            
        Returns:
        --------
        float
            Estimated beta coefficient
        """
        if weights is None:
            weights = np.ones(len(y))
            
        # Add constant to x
        X = np.column_stack((np.ones(len(x)), x))
        
        # Weighted least squares
        W = np.diag(weights)
        XtW = X.T @ W
        beta = np.linalg.inv(XtW @ X) @ XtW @ y
        
        return beta[1]  # Return the slope coefficient
    
    def execute_basic_strategy(self, dates, exchange_rates, interest_differentials):
        """
        Execute basic UIP strategy without proxy variables
        
        Parameters:
        -----------
        dates: array-like
            Dates corresponding to the data
        exchange_rates: array-like
            Exchange rates (currency per USD)
        interest_differentials: array-like
            Interest rate differentials (foreign - US)
            
        Returns:
        --------
        DataFrame
            Trading results
        """
        # Calculate exchange rate changes
        exchange_rate_changes = np.diff(np.log(exchange_rates))
        
        # Initialize results
        results = pd.DataFrame({
            'exchange_rate': exchange_rates[1:],
            'exchange_rate_change': exchange_rate_changes,
            'interest_differential': interest_differentials[:-1],
            'position': np.zeros(len(exchange_rate_changes)),
            'return': np.zeros(len(exchange_rate_changes)),
            'beta': np.zeros(len(exchange_rate_changes)),
            'predicted_change': np.zeros(len(exchange_rate_changes)),
            'uip_deviation': np.zeros(len(exchange_rate_changes))
        }, index=dates[1:])
        
        # Calculate time-varying beta estimates
        weights = self.calculate_weights()
        
        for t in range(self.window_size, len(results)):
            # Get rolling window of data
            y_window = results['exchange_rate_change'].iloc[t-self.window_size:t].values
            x_window = results['interest_differential'].iloc[t-self.window_size:t].values
            
            # Estimate beta
            beta = self.estimate_beta(y_window, x_window, weights)
            results.loc[results.index[t], 'beta'] = beta
            
            # Calculate predicted exchange rate change based on UIP
            predicted_change = results['interest_differential'].iloc[t-1]
            results.loc[results.index[t], 'predicted_change'] = predicted_change
            
            # Calculate UIP deviation
            uip_deviation = predicted_change - beta * predicted_change
            results.loc[results.index[t], 'uip_deviation'] = uip_deviation
            
            # Determine position: long foreign currency if predicted change is positive
            # (i.e., foreign currency is expected to appreciate)
            position_size = np.sign(predicted_change + uip_deviation) / self.risk_aversion
            results.loc[results.index[t], 'position'] = position_size
            
            # Calculate return: position * actual exchange rate change
            # (positive return means profit from correctly predicting currency movement)
            results.loc[results.index[t], 'return'] = position_size * results['exchange_rate_change'].iloc[t]
        
        # Calculate cumulative returns
        results['cumulative_return'] = (1 + results['return']).cumprod() - 1
        
        return results.iloc[self.window_size:]
    
    def execute_enhanced_strategy(self, dates, exchange_rates, interest_differentials, risk_proxies, error_proxies):
        """
        Execute enhanced UIP strategy with proxy variables
        
        Parameters:
        -----------
        dates: array-like
            Dates corresponding to the data
        exchange_rates: array-like
            Exchange rates (currency per USD)
        interest_differentials: array-like
            Interest rate differentials (foreign - US)
        risk_proxies: DataFrame
            Proxy variables for risk premiums
        error_proxies: DataFrame
            Proxy variables for expectational errors
            
        Returns:
        --------
        DataFrame
            Trading results
        """
        # Calculate exchange rate changes
        exchange_rate_changes = np.diff(np.log(exchange_rates))
        
        # Initialize results
        results = pd.DataFrame({
            'exchange_rate': exchange_rates[1:],
            'exchange_rate_change': exchange_rate_changes,
            'interest_differential': interest_differentials[:-1],
            'position': np.zeros(len(exchange_rate_changes)),
            'return': np.zeros(len(exchange_rate_changes)),
            'beta': np.zeros(len(exchange_rate_changes)),
            'adjusted_prediction': np.zeros(len(exchange_rate_changes))
        }, index=dates[1:])
        
        # Ensure proxy variables align with exchange rate changes
        risk_proxies = risk_proxies.iloc[:len(results)]
        error_proxies = error_proxies.iloc[:len(results)]
        
        # Calculate time-varying parameters
        weights = self.calculate_weights()
        
        for t in range(self.window_size, len(results)):
            # Get rolling window of data
            y_window = results['exchange_rate_change'].iloc[t-self.window_size:t].values
            x_window = results['interest_differential'].iloc[t-self.window_size:t].values
            
            # Get current risk and error proxies
            current_risk_proxies = risk_proxies.iloc[t-1].values
            current_error_proxies = error_proxies.iloc[t].values if t < len(error_proxies) else error_proxies.iloc[-1].values
            
            # Estimate beta using basic relationship
            beta = self.estimate_beta(y_window, x_window, weights)
            results.loc[results.index[t], 'beta'] = beta
            
            # Calculate basic UIP prediction
            uip_prediction = results['interest_differential'].iloc[t-1]
            
            # Adjust prediction using proxy variables
            # This is a simplified approach - in practice we would estimate time-varying
            # coefficients for each proxy variable using DLM-DMA as in the paper
            risk_adjustment = np.mean(current_risk_proxies) * 0.01  # Simple adjustment factor
            error_adjustment = np.mean(current_error_proxies) * 0.01  # Simple adjustment factor
            
            adjusted_prediction = beta * uip_prediction + risk_adjustment + error_adjustment
            results.loc[results.index[t], 'adjusted_prediction'] = adjusted_prediction
            
            # Determine position: long foreign currency if adjusted prediction is positive
            position_size = np.sign(adjusted_prediction) / self.risk_aversion
            results.loc[results.index[t], 'position'] = position_size
            
            # Calculate return
            results.loc[results.index[t], 'return'] = position_size * results['exchange_rate_change'].iloc[t]
        
        # Calculate cumulative returns
        results['cumulative_return'] = (1 + results['return']).cumprod() - 1
        
        return results.iloc[self.window_size:]
    
    def execute_dynamic_model_averaging_strategy(self, dates, exchange_rates, interest_differentials, 
                                                proxy_variables, alpha=0.95):
        """
        Execute UIP strategy with dynamic model averaging of proxy variables
        
        Parameters:
        -----------
        dates: array-like
            Dates corresponding to the data
        exchange_rates: array-like
            Exchange rates (currency per USD)
        interest_differentials: array-like
            Interest rate differentials (foreign - US)
        proxy_variables: DataFrame
            All proxy variables for CIP deviations, risk premiums, and expectational errors
        alpha: float
            Forgetting factor for model probabilities (0 < alpha <= 1)
            
        Returns:
        --------
        DataFrame
            Trading results
        """
        # Calculate exchange rate changes
        exchange_rate_changes = np.diff(np.log(exchange_rates))
        
        # Initialize results
        results = pd.DataFrame({
            'exchange_rate': exchange_rates[1:],
            'exchange_rate_change': exchange_rate_changes,
            'interest_differential': interest_differentials[:-1],
            'position': np.zeros(len(exchange_rate_changes)),
            'return': np.zeros(len(exchange_rate_changes)),
            'beta': np.zeros(len(exchange_rate_changes)),
            'dma_prediction': np.zeros(len(exchange_rate_changes))
        }, index=dates[1:])
        
        # Ensure proxy variables align with exchange rate changes
        proxy_variables = proxy_variables.iloc[:len(results)]
        
        # Number of proxy variables
        num_proxies = min(5, proxy_variables.shape[1])  # Limit to 5 for computational simplicity
        
        # Generate all possible models (simplified for computational reasons)
        # In practice, we would consider all 2^n models as in the paper
        models = []
        for i in range(num_proxies + 1):
            if i == 0:
                # Base model with only interest rate differential
                models.append([])
            else:
                # Models with one proxy variable
                models.append([i-1])
        
        # Initialize model probabilities
        model_probs = np.ones(len(models)) / len(models)
        
        # Calculate time-varying parameters
        weights = self.calculate_weights()
        
        for t in range(self.window_size, len(results)):
            # Get rolling window of data
            y_window = results['exchange_rate_change'].iloc[t-self.window_size:t].values
            x_window = results['interest_differential'].iloc[t-self.window_size:t].values
            
            # Estimate beta using basic relationship
            beta = self.estimate_beta(y_window, x_window, weights)
            results.loc[results.index[t], 'beta'] = beta
            
            # Calculate basic UIP prediction
            uip_prediction = results['interest_differential'].iloc[t-1]
            
            # Get current proxy variables
            if t < len(proxy_variables):
                current_proxies = proxy_variables.iloc[t-1].values[:num_proxies]
            else:
                current_proxies = proxy_variables.iloc[-1].values[:num_proxies]
            
            # Calculate prediction for each model
            model_predictions = []
            model_errors = []
            
            for model_idx, model in enumerate(models):
                if len(model) == 0:
                    # Base model: just beta * interest_differential
                    prediction = beta * uip_prediction
                else:
                    # Model with proxy variables
                    proxy_adjustment = np.sum([current_proxies[idx] * 0.01 for idx in model])
                    prediction = beta * uip_prediction + proxy_adjustment
                
                model_predictions.append(prediction)
                
                # Calculate recent prediction error for this model
                if t > self.window_size:
                    prev_prediction = beta * results['interest_differential'].iloc[t-2]
                    if len(model) > 0:
                        if t-2 < len(proxy_variables):
                            prev_proxies = proxy_variables.iloc[t-2].values[:num_proxies]
                        else:
                            prev_proxies = proxy_variables.iloc[-1].values[:num_proxies]
                        prev_proxy_adjustment = np.sum([prev_proxies[idx] * 0.01 for idx in model])
                        prev_prediction += prev_proxy_adjustment
                    
                    error = (results['exchange_rate_change'].iloc[t-1] - prev_prediction) ** 2
                    model_errors.append(error)
                else:
                    model_errors.append(0.01)  # Initial error
            
            # Update model probabilities using prediction errors
            if t > self.window_size:
                # Convert errors to likelihoods
                likelihoods = np.exp(-0.5 * np.array(model_errors))
                
                # Update probabilities with forgetting factor
                model_probs = model_probs ** alpha
                model_probs = model_probs * likelihoods
                model_probs = model_probs / np.sum(model_probs)
            
            # Calculate model-averaged prediction
            dma_prediction = np.sum(np.array(model_predictions) * model_probs)
            results.loc[results.index[t], 'dma_prediction'] = dma_prediction
            
            # Determine position
            position_size = np.sign(dma_prediction) / self.risk_aversion
            results.loc[results.index[t], 'position'] = position_size
            
            # Calculate return
            results.loc[results.index[t], 'return'] = position_size * results['exchange_rate_change'].iloc[t]
        
        # Calculate cumulative returns
        results['cumulative_return'] = (1 + results['return']).cumprod() - 1
        
        return results.iloc[self.window_size:]
    
    def calculate_performance_metrics(self, results):
        """
        Calculate performance metrics for the strategy
        
        Parameters:
        -----------
        results: DataFrame
            Trading results
            
        Returns:
        --------
        dict
            Performance metrics
        """
        returns = results['return']
        
        # Basic metrics
        total_return = results['cumulative_return'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Average win/loss
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        
        # Profit factor
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Beta analysis
        avg_beta = results['beta'].mean()
        beta_std = results['beta'].std()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_beta': avg_beta,
            'beta_std': beta_std
        }

def generate_currency_data(num_days=1000, interest_rate_vol=0.001, 
                          exchange_rate_vol=0.008, beta_true=0.5, seed=42):
    """
    Generate simulated currency data with a specified UIP relationship
    
    Parameters:
    -----------
    num_days: int
        Number of days to generate
    interest_rate_vol: float
        Volatility of interest rate differentials
    exchange_rate_vol: float
        Base volatility of exchange rate changes
    beta_true: float
        True beta coefficient in the UIP relationship
    seed: int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (dates, exchange_rates, interest_differentials, risk_proxies, error_proxies, proxy_variables)
    """
    np.random.seed(seed)
    
    # Generate dates
    base_date = datetime(2020, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(num_days)]
    
    # Generate interest rate differentials
    interest_differentials = np.zeros(num_days)
    interest_differentials[0] = np.random.normal(0, interest_rate_vol)
    
    for i in range(1, num_days):
        # Interest rates are persistent
        interest_differentials[i] = 0.98 * interest_differentials[i-1] + np.random.normal(0, interest_rate_vol)
    
    # Generate CIP deviations
    cip_deviations = np.zeros(num_days)
    for i in range(1, num_days):
        # CIP deviations show persistence
        cip_deviations[i] = 0.9 * cip_deviations[i-1] + 0.1 * np.random.normal(0, 0.0005)
        
        # Introduce larger CIP deviations during "crisis" periods
        if 300 <= i < 350 or 700 <= i < 750:
            cip_deviations[i] += np.random.normal(0, 0.002)
    
    # Generate risk premium factors (8 macro variables)
    num_risk_factors = 8
    risk_factors = np.zeros((num_days, num_risk_factors))
    
    for j in range(num_risk_factors):
        # Different persistence for different factors
        persistence = 0.8 + 0.15 * j / num_risk_factors
        vol = 0.001 + 0.002 * j / num_risk_factors
        
        risk_factors[0, j] = np.random.normal(0, vol)
        for i in range(1, num_days):
            risk_factors[i, j] = persistence * risk_factors[i-1, j] + np.random.normal(0, vol)
    
    # Generate expectational error factors (5 variables)
    num_error_factors = 5
    error_factors = np.zeros((num_days, num_error_factors))
    
    for j in range(num_error_factors):
        vol = 0.002 + 0.003 * j / num_error_factors
        for i in range(num_days):
            error_factors[i, j] = np.random.normal(0, vol)
    
    # Create risk premium based on risk factors (time-varying effects)
    risk_premium = np.zeros(num_days)
    risk_weights = np.zeros((num_days, num_risk_factors))
    
    for i in range(num_days):
        # Weights shift over time, especially during "regime changes"
        if i == 0:
            risk_weights[i] = np.random.uniform(0, 0.2, num_risk_factors)
        else:
            # Normal weight evolution
            weight_change = np.random.normal(0, 0.001, num_risk_factors)
            
            # Larger changes during regime shifts
            if i % 200 == 0:
                weight_change = np.random.normal(0, 0.05, num_risk_factors)
            
            risk_weights[i] = risk_weights[i-1] + weight_change
        
        # Normalize weights to sum to 1
        risk_weights[i] = np.abs(risk_weights[i])
        risk_weights[i] = risk_weights[i] / np.sum(risk_weights[i])
        
        # Calculate risk premium
        risk_premium[i] = np.sum(risk_weights[i] * risk_factors[i])
    
    # Create expectational errors based on error factors (with regime changes)
    expectational_error = np.zeros(num_days)
    error_weights = np.zeros((num_days, num_error_factors))
    
    for i in range(num_days):
        if i == 0:
            error_weights[i] = np.random.uniform(0, 0.2, num_error_factors)
        else:
            # Normal weight evolution
            weight_change = np.random.normal(0, 0.001, num_error_factors)
            
            # Larger changes during regime shifts
            if i % 200 == 0:
                weight_change = np.random.normal(0, 0.05, num_error_factors)
            
            error_weights[i] = error_weights[i-1] + weight_change
        
        # Normalize weights
        error_weights[i] = np.abs(error_weights[i])
        error_weights[i] = error_weights[i] / np.sum(error_weights[i])
        
        # Calculate expectational error
        expectational_error[i] = np.sum(error_weights[i] * error_factors[i])
    
    # Generate exchange rate changes based on UIP relationship with time-varying beta
    time_varying_beta = np.zeros(num_days)
    exchange_rate_changes = np.zeros(num_days)
    
    for i in range(1, num_days):
        # Beta evolves over time
        if i == 1:
            time_varying_beta[i] = beta_true
        else:
            # Gradual evolution with occasional jumps
            if i % 200 == 0:
                time_varying_beta[i] = np.random.uniform(-0.5, 1.5)
            else:
                time_varying_beta[i] = 0.98 * time_varying_beta[i-1] + 0.02 * np.random.normal(beta_true, 0.1)
        
        # Exchange rate change based on UIP relationship plus deviations
        exchange_rate_changes[i] = (time_varying_beta[i] * interest_differentials[i-1] + 
                                   cip_deviations[i] + risk_premium[i] + expectational_error[i] + 
                                   np.random.normal(0, exchange_rate_vol))
    
    # Cumulative exchange rate levels (starting at 1.0)
    exchange_rates = np.exp(np.cumsum(np.insert(exchange_rate_changes[1:], 0, 0)))
    
    # Create DataFrames for proxy variables
    risk_proxies = pd.DataFrame(risk_factors, index=dates, 
                              columns=[f'risk_factor_{i+1}' for i in range(num_risk_factors)])
    
    error_proxies = pd.DataFrame(error_factors, index=dates, 
                               columns=[f'error_factor_{i+1}' for i in range(num_error_factors)])
    
    # Combine all proxy variables
    proxy_variables = pd.DataFrame(index=dates)
    proxy_variables['cip_deviation'] = cip_deviations
    
    for i in range(num_risk_factors):
        proxy_variables[f'risk_factor_{i+1}'] = risk_factors[:, i]
    
    for i in range(num_error_factors):
        proxy_variables[f'error_factor_{i+1}'] = error_factors[:, i]
    
    # Add true beta for reference
    proxy_variables['true_beta'] = time_varying_beta
    
    return dates, exchange_rates, interest_differentials, risk_proxies, error_proxies, proxy_variables

def run_uip_strategy_comparison():
    """
    Run and compare different UIP strategy implementations
    """
    # Generate simulated data
    print("Generating simulated currency data...")
    dates, exchange_rates, interest_differentials, risk_proxies, error_proxies, proxy_variables = \
        generate_currency_data(num_days=1000, beta_true=0.5)
    
    # Initialize strategy
    uip_strategy = UIPStrategy(window_size=60, decay_factor=0.95)
    
    # Run basic strategy
    print("Running basic UIP strategy...")
    basic_results = uip_strategy.execute_basic_strategy(dates, exchange_rates, interest_differentials)
    basic_metrics = uip_strategy.calculate_performance_metrics(basic_results)
    
    # Run enhanced strategy
    print("Running enhanced UIP strategy with proxy variables...")
    enhanced_results = uip_strategy.execute_enhanced_strategy(
        dates, exchange_rates, interest_differentials, risk_proxies, error_proxies)
    enhanced_metrics = uip_strategy.calculate_performance_metrics(enhanced_results)
    
    # Run DMA strategy
    print("Running UIP strategy with dynamic model averaging...")
    dma_results = uip_strategy.execute_dynamic_model_averaging_strategy(
        dates, exchange_rates, interest_differentials, proxy_variables)
    dma_metrics = uip_strategy.calculate_performance_metrics(dma_results)
    
    # Print performance metrics
    print("\n=== PERFORMANCE COMPARISON ===")
    print("\nBasic UIP Strategy:")
    print(f"Total Return: {basic_metrics['total_return']:.2%}")
    print(f"Annual Return: {basic_metrics['annual_return']:.2%}")
    print(f"Volatility: {basic_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {basic_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {basic_metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {basic_metrics['win_rate']:.2%}")
    print(f"Profit Factor: {basic_metrics['profit_factor']:.2f}")
    print(f"Average Beta: {basic_metrics['avg_beta']:.2f}")
    print(f"Beta Standard Deviation: {basic_metrics['beta_std']:.2f}")
    
    print("\nEnhanced UIP Strategy:")
    print(f"Total Return: {enhanced_metrics['total_return']:.2%}")
    print(f"Annual Return: {enhanced_metrics['annual_return']:.2%}")
    print(f"Volatility: {enhanced_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {enhanced_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {enhanced_metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {enhanced_metrics['win_rate']:.2%}")
    print(f"Profit Factor: {enhanced_metrics['profit_factor']:.2f}")
    print(f"Average Beta: {enhanced_metrics['avg_beta']:.2f}")
    print(f"Beta Standard Deviation: {enhanced_metrics['beta_std']:.2f}")
    
    print("\nDynamic Model Averaging Strategy:")
    print(f"Total Return: {dma_metrics['total_return']:.2%}")
    print(f"Annual Return: {dma_metrics['annual_return']:.2%}")
    print(f"Volatility: {dma_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {dma_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {dma_metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {dma_metrics['win_rate']:.2%}")
    print(f"Profit Factor: {dma_metrics['profit_factor']:.2f}")
    print(f"Average Beta: {dma_metrics['avg_beta']:.2f}")
    print(f"Beta Standard Deviation: {dma_metrics['beta_std']:.2f}")
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Cumulative Returns Comparison
    plt.subplot(3, 1, 1)
    plt.plot(basic_results.index, basic_results['cumulative_return'], label='Basic UIP Strategy')
    plt.plot(enhanced_results.index, enhanced_results['cumulative_return'], label='Enhanced UIP Strategy')
    plt.plot(dma_results.index, dma_results['cumulative_return'], label='DMA UIP Strategy')
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Time-varying Beta Estimates
    plt.subplot(3, 1, 2)
    plt.plot(basic_results.index, basic_results['beta'], label='Estimated Beta (Basic)')
    plt.plot(dma_results.index, dma_results['beta'], label='Estimated Beta (DMA)', alpha=0.7)
    
    # Align proxy_variables index with results index
    aligned_true_beta = proxy_variables.loc[proxy_variables.index.isin(basic_results.index), 'true_beta']
    if len(aligned_true_beta) > 0:
        plt.plot(aligned_true_beta.index, aligned_true_beta.values, label='True Beta', linestyle='--', color='black')
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='UIP Theoretical Value (β=1)')
    plt.title('Time-varying Beta Estimates')
    plt.xlabel('Date')
    plt.ylabel('Beta')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Strategy Positions
    plt.subplot(3, 1, 3)
    plt.plot(basic_results.index, basic_results['position'], label='Basic Strategy Positions', alpha=0.5)
    plt.plot(dma_results.index, dma_results['position'], label='DMA Strategy Positions')
    plt.title('Trading Positions')
    plt.xlabel('Date')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional plots
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Interest Rate Differential
    plt.subplot(3, 1, 1)
    plt.plot(dates[:-1], interest_differentials[:-1])
    plt.title('Interest Rate Differential')
    plt.xlabel('Date')
    plt.ylabel('Differential')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: CIP Deviations
    plt.subplot(3, 1, 2)
    plt.plot(proxy_variables.index, proxy_variables['cip_deviation'])
    plt.title('CIP Deviations')
    plt.xlabel('Date')
    plt.ylabel('Deviation')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Exchange Rate
    plt.subplot(3, 1, 3)
    plt.plot(dates, exchange_rates)
    plt.title('Exchange Rate')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return basic_results, enhanced_results, dma_results, proxy_variables

def investigate_beta_distribution(results, proxy_variables):
    """
    Investigate the distribution of beta estimates and their relationship to proxy variables
    """
    # Create a DataFrame for analysis with just the beta values first
    analysis_df = pd.DataFrame({
        'beta': results['beta'],
        'interest_differential': results['interest_differential'],
        'exchange_rate_change': results['exchange_rate_change']
    })
    
    # Select proxy variables data that matches the results timeframe
    # First make sure proxy_variables has the same index type as results
    if isinstance(results.index, pd.DatetimeIndex) and isinstance(proxy_variables.index, pd.DatetimeIndex):
        # Get subset of proxy_variables that matches results index
        matching_indices = proxy_variables.index.intersection(results.index)
        if len(matching_indices) > 0:
            proxy_subset = proxy_variables.loc[matching_indices]
            
            # Add proxy variables to analysis_df
            analysis_df['true_beta'] = proxy_subset['true_beta']
            analysis_df['cip_deviation'] = proxy_subset['cip_deviation']
            
            # Add some risk and error factors
            for i in range(1, 4):
                if f'risk_factor_{i}' in proxy_subset.columns:
                    analysis_df[f'risk_factor_{i}'] = proxy_subset[f'risk_factor_{i}']
                if f'error_factor_{i}' in proxy_subset.columns:
                    analysis_df[f'error_factor_{i}'] = proxy_subset[f'error_factor_{i}']
        else:
            # If no matching indices, use dummy values
            print("Warning: No matching indices between results and proxy_variables. Using dummy values.")
            analysis_df['true_beta'] = 0.5  # Use the true beta value from simulation
            analysis_df['cip_deviation'] = 0.0
            
            # Add some risk and error factors
            for i in range(1, 4):
                analysis_df[f'risk_factor_{i}'] = 0.0
                analysis_df[f'error_factor_{i}'] = 0.0
    else:
        # Different index types, use positional alignment
        print("Warning: Different index types. Attempting positional alignment.")
        
        # Start index for proxy_variables (corresponding to first entry in results)
        start_idx = 0
        if hasattr(results.index, 'min') and hasattr(proxy_variables.index, 'min'):
            if results.index.min() > proxy_variables.index.min():
                # Find the approximate position in proxy_variables that corresponds to start of results
                start_idx = (results.index.min() - proxy_variables.index.min()).days
                start_idx = max(0, min(start_idx, len(proxy_variables) - len(results)))
        
        # Extract subset of proxy_variables
        end_idx = min(start_idx + len(results), len(proxy_variables))
        proxy_subset = proxy_variables.iloc[start_idx:end_idx]
        
        if len(proxy_subset) >= len(results):
            # Add proxy variables to analysis_df
            analysis_df['true_beta'] = proxy_subset['true_beta'].values[:len(results)]
            analysis_df['cip_deviation'] = proxy_subset['cip_deviation'].values[:len(results)]
            
            # Add some risk and error factors
            for i in range(1, 4):
                if f'risk_factor_{i}' in proxy_subset.columns:
                    analysis_df[f'risk_factor_{i}'] = proxy_subset[f'risk_factor_{i}'].values[:len(results)]
                if f'error_factor_{i}' in proxy_subset.columns:
                    analysis_df[f'error_factor_{i}'] = proxy_subset[f'error_factor_{i}'].values[:len(results)]
        else:
            # Not enough data
            print("Warning: Not enough proxy data. Using dummy values.")
            analysis_df['true_beta'] = 0.5
            analysis_df['cip_deviation'] = 0.0
            
            # Add some risk and error factors
            for i in range(1, 4):
                analysis_df[f'risk_factor_{i}'] = 0.0
                analysis_df[f'error_factor_{i}'] = 0.0
    
    # Plot beta distribution
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Beta Distribution
    plt.subplot(3, 2, 1)
    sns.histplot(analysis_df['beta'], kde=True)
    plt.axvline(x=1.0, color='r', linestyle='--', label='UIP Theoretical Value (β=1)')
    plt.axvline(x=analysis_df['beta'].mean(), color='g', linestyle='--', 
                label=f'Mean Estimated Beta ({analysis_df["beta"].mean():.2f})')
    plt.title('Distribution of Beta Estimates')
    plt.xlabel('Beta')
    plt.legend()
    
    # Plot 2: True Beta vs Estimated Beta
    plt.subplot(3, 2, 2)
    plt.scatter(analysis_df['true_beta'], analysis_df['beta'], alpha=0.5)
    true_beta_min = analysis_df['true_beta'].min()
    true_beta_max = analysis_df['true_beta'].max()
    plt.plot([true_beta_min, true_beta_max], [true_beta_min, true_beta_max], 'r--', label='Perfect Estimation')
    plt.title('True Beta vs Estimated Beta')
    plt.xlabel('True Beta')
    plt.ylabel('Estimated Beta')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Beta vs Interest Rate Differential
    plt.subplot(3, 2, 3)
    plt.scatter(analysis_df['interest_differential'], analysis_df['beta'], alpha=0.5)
    plt.title('Beta vs Interest Rate Differential')
    plt.xlabel('Interest Rate Differential')
    plt.ylabel('Beta')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Beta vs CIP Deviation
    plt.subplot(3, 2, 4)
    plt.scatter(analysis_df['cip_deviation'], analysis_df['beta'], alpha=0.5)
    plt.title('Beta vs CIP Deviation')
    plt.xlabel('CIP Deviation')
    plt.ylabel('Beta')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Beta vs Risk Factor 1
    plt.subplot(3, 2, 5)
    if 'risk_factor_1' in analysis_df.columns:
        plt.scatter(analysis_df['risk_factor_1'], analysis_df['beta'], alpha=0.5)
        plt.title('Beta vs Risk Factor 1')
        plt.xlabel('Risk Factor 1')
        plt.ylabel('Beta')
    else:
        plt.text(0.5, 0.5, 'Risk Factor 1 data not available', ha='center', va='center')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Beta vs Error Factor 1
    plt.subplot(3, 2, 6)
    if 'error_factor_1' in analysis_df.columns:
        plt.scatter(analysis_df['error_factor_1'], analysis_df['beta'], alpha=0.5)
        plt.title('Beta vs Error Factor 1')
        plt.xlabel('Error Factor 1')
        plt.ylabel('Beta')
    else:
        plt.text(0.5, 0.5, 'Error Factor 1 data not available', ha='center', va='center')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate correlations
    corr_matrix = analysis_df.corr()
    
    print("\n=== CORRELATIONS WITH BETA ===")
    for col in corr_matrix.columns:
        if col != 'beta':
            print(f"{col}: {corr_matrix.loc['beta', col]:.3f}")
    
    # Calculate mean absolute deviation of beta from 1 (UIP theoretical value)
    mad_from_1 = abs(analysis_df['beta'] - 1).mean()
    
    # Calculate mean absolute deviation of beta from true beta
    if 'true_beta' in analysis_df.columns:
        mad_from_true = abs(analysis_df['beta'] - analysis_df['true_beta']).mean()
        print(f"\nMean Absolute Deviation of Beta from 1: {mad_from_1:.3f}")
        print(f"Mean Absolute Deviation of Beta from True Beta: {mad_from_true:.3f}")
    else:
        print(f"\nMean Absolute Deviation of Beta from 1: {mad_from_1:.3f}")
        print("True beta data not available for comparison.")
    
    return analysis_df

# Run the strategies
basic_results, enhanced_results, dma_results, proxy_variables = run_uip_strategy_comparison()

# Investigate beta distributions
analysis_df = investigate_beta_distribution(dma_results, proxy_variables)