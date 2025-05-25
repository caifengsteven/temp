import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ShortingLeveragedETFPairs:
    def __init__(self, leverage_ratio=3):
        """
        Initialize the ShortingLeveragedETFPairs class
        
        Parameters:
        leverage_ratio (int): The leverage ratio of the ETFs (default: 3 for triple-leveraged ETFs)
        """
        self.leverage_ratio = leverage_ratio
        np.random.seed(42)  # For reproducibility
    
    def simulate_benchmark_index(self, n_days=1000, daily_drift=0.0003, 
                                volatility=0.01, autocorr=-0.05,
                                start_date='2010-01-01'):
        """
        Simulate a benchmark index with specified properties
        
        Parameters:
        n_days (int): Number of days to simulate
        daily_drift (float): Daily expected return
        volatility (float): Daily volatility
        autocorr (float): First-order autocorrelation
        start_date (str): Start date for the simulation
        
        Returns:
        pandas.DataFrame: Simulated benchmark index prices and returns
        """
        # Generate dates
        dates = pd.date_range(start=start_date, periods=n_days)
        
        # Initialize arrays
        returns = np.zeros(n_days)
        
        # Generate first return
        returns[0] = np.random.normal(daily_drift, volatility)
        
        # Generate subsequent returns with autocorrelation
        for i in range(1, n_days):
            # AR(1) process: r_t = μ + ρ(r_{t-1} - μ) + ε_t
            returns[i] = daily_drift + autocorr * (returns[i-1] - daily_drift) + np.random.normal(0, volatility * np.sqrt(1 - autocorr**2))
        
        # Convert returns to prices
        prices = 100 * np.cumprod(1 + returns)
        
        # Create DataFrame
        df = pd.DataFrame({'Price': prices, 'Return': returns}, index=dates)
        
        return df
    
    def simulate_mean_reverting_benchmark(self, n_days=1000, volatility=0.01,
                                         mean_reversion_strength=0.1, mean_level=100,
                                         start_date='2010-01-01'):
        """
        Simulate a mean-reverting benchmark index
        
        Parameters:
        n_days (int): Number of days to simulate
        volatility (float): Daily volatility
        mean_reversion_strength (float): Strength of mean reversion
        mean_level (float): Mean level to revert to
        start_date (str): Start date for the simulation
        
        Returns:
        pandas.DataFrame: Simulated benchmark index prices and returns
        """
        # Generate dates
        dates = pd.date_range(start=start_date, periods=n_days)
        
        # Initialize arrays
        prices = np.zeros(n_days)
        returns = np.zeros(n_days)
        
        # Initial price
        prices[0] = mean_level
        
        # Generate prices using Ornstein-Uhlenbeck process
        for i in range(1, n_days):
            # Ornstein-Uhlenbeck: dX_t = θ(μ - X_t)dt + σdW_t
            prices[i] = prices[i-1] + mean_reversion_strength * (mean_level - prices[i-1]) + volatility * np.random.normal(0, 1)
            returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
        
        # First return is 0
        returns[0] = 0
        
        # Create DataFrame
        df = pd.DataFrame({'Price': prices, 'Return': returns}, index=dates)
        
        return df
    
    def simulate_leveraged_etfs(self, benchmark_df):
        """
        Simulate leveraged ETFs based on a benchmark index
        
        Parameters:
        benchmark_df (pandas.DataFrame): DataFrame with benchmark index prices and returns
        
        Returns:
        pandas.DataFrame: Dataframe with benchmark, LETF, and IETF prices and returns
        """
        # Extract benchmark returns
        benchmark_returns = benchmark_df['Return'].values
        
        # Initialize arrays
        n_days = len(benchmark_returns)
        letf_returns = np.zeros(n_days)
        ietf_returns = np.zeros(n_days)
        
        # Generate daily returns for leveraged ETFs
        for i in range(n_days):
            letf_returns[i] = self.leverage_ratio * benchmark_returns[i]
            ietf_returns[i] = -self.leverage_ratio * benchmark_returns[i]
        
        # Convert returns to prices
        letf_prices = 100 * np.cumprod(1 + letf_returns)
        ietf_prices = 100 * np.cumprod(1 + ietf_returns)
        
        # Create DataFrame
        df = benchmark_df.copy()
        df['LETF_Price'] = letf_prices
        df['LETF_Return'] = letf_returns
        df['IETF_Price'] = ietf_prices
        df['IETF_Return'] = ietf_returns
        
        return df
    
    def backtest_monthly_shorting_strategy(self, etf_df):
        """
        Backtest the strategy of shorting both LETF and IETF on a monthly basis
        
        Parameters:
        etf_df (pandas.DataFrame): DataFrame with benchmark, LETF, and IETF prices and returns
        
        Returns:
        pandas.DataFrame: DataFrame with monthly strategy returns
        """
        # Resample to monthly frequency for end-of-month values
        monthly_df = etf_df.resample('M').last()
        
        # Calculate monthly returns
        monthly_returns = pd.DataFrame(index=monthly_df.index)
        monthly_returns['Benchmark'] = monthly_df['Price'].pct_change()
        monthly_returns['LETF'] = monthly_df['LETF_Price'].pct_change()
        monthly_returns['IETF'] = monthly_df['IETF_Price'].pct_change()
        
        # Calculate strategy returns (short equal amounts of LETF and IETF)
        monthly_returns['Strategy'] = -0.5 * monthly_returns['LETF'] - 0.5 * monthly_returns['IETF']
        
        # Calculate cumulative returns
        monthly_returns['Benchmark_Cum'] = (1 + monthly_returns['Benchmark']).cumprod() - 1
        monthly_returns['Strategy_Cum'] = (1 + monthly_returns['Strategy']).cumprod() - 1
        
        return monthly_returns
    
    def calculate_autocorrelations(self, returns, max_lag=20):
        """
        Calculate autocorrelations of returns
        
        Parameters:
        returns (numpy.array): Array of returns
        max_lag (int): Maximum lag to calculate
        
        Returns:
        numpy.array: Array of autocorrelations
        """
        return acf(returns, nlags=max_lag)[1:]  # Skip lag 0 (which is always 1)
    
    def calculate_sum_ar(self, autocorrs, n=21):
        """
        Calculate weighted sum of autocorrelations as per Equation 9 in the paper
        
        Parameters:
        autocorrs (numpy.array): Array of autocorrelations
        n (int): Number of trading days in a month
        
        Returns:
        float: Weighted sum of autocorrelations
        """
        weights = [(n - i) / (n - 1) for i in range(1, len(autocorrs) + 1)]
        return np.sum(autocorrs * weights[:len(autocorrs)])
    
    def calculate_expected_return(self, autocorrs, volatility, n=21):
        """
        Calculate expected return of shorting strategy as per Equation 10 in the paper
        
        Parameters:
        autocorrs (numpy.array): Array of autocorrelations
        volatility (float): Daily volatility of benchmark index
        n (int): Number of trading days in a month
        
        Returns:
        float: Expected monthly return of shorting strategy
        """
        sum_ar = self.calculate_sum_ar(autocorrs, n)
        return -self.leverage_ratio**2 * (n - 1) * volatility**2 * sum_ar
    
    def calculate_sum_cross_products(self, returns):
        """
        Calculate sum of cross products of daily returns within a month
        
        Parameters:
        returns (numpy.array): Array of daily returns
        
        Returns:
        float: Sum of cross products
        """
        n = len(returns)
        sum_cp = 0
        
        for i in range(n):
            for j in range(i+1, n):
                sum_cp += returns[i] * returns[j]
        
        return sum_cp
    
    def analyze_strategy_performance(self, etf_df, monthly_returns):
        """
        Analyze the performance of the shorting strategy
        
        Parameters:
        etf_df (pandas.DataFrame): DataFrame with daily benchmark, LETF, and IETF prices and returns
        monthly_returns (pandas.DataFrame): DataFrame with monthly strategy returns
        
        Returns:
        None (prints analysis results and produces plots)
        """
        # Calculate daily statistics
        daily_returns = etf_df['Return'].values
        volatility = np.std(daily_returns)
        autocorrs = self.calculate_autocorrelations(daily_returns)
        sum_ar = self.calculate_sum_ar(autocorrs)
        expected_monthly_return = self.calculate_expected_return(autocorrs, volatility)
        
        # Monthly statistics
        mean_monthly_return = monthly_returns['Strategy'].mean()
        std_monthly_return = monthly_returns['Strategy'].std()
        sharpe_ratio = np.sqrt(12) * mean_monthly_return / std_monthly_return
        
        # Print results
        print("=== Benchmark Index Analysis ===")
        print(f"Daily Volatility: {volatility:.4f}")
        print(f"First-order Autocorrelation: {autocorrs[0]:.4f}")
        print(f"SUM_AR: {sum_ar:.4f}")
        print(f"Expected Monthly Return of Shorting Strategy: {expected_monthly_return:.4f} ({expected_monthly_return*100:.2f}%)")
        
        print("\n=== Shorting Strategy Performance ===")
        print(f"Mean Monthly Return: {mean_monthly_return:.4f} ({mean_monthly_return*100:.2f}%)")
        print(f"Standard Deviation of Monthly Returns: {std_monthly_return:.4f}")
        print(f"Annualized Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Min Monthly Return: {monthly_returns['Strategy'].min():.4f} ({monthly_returns['Strategy'].min()*100:.2f}%)")
        print(f"Max Monthly Return: {monthly_returns['Strategy'].max():.4f} ({monthly_returns['Strategy'].max()*100:.2f}%)")
        
        # Calculate monthly sum of cross products
        monthly_sum_cp = []
        monthly_vol = []
        
        for month_end in monthly_returns.index:
            # Get daily returns for this month
            month_start = month_end.replace(day=1)
            month_data = etf_df.loc[month_start:month_end]
            
            # Calculate sum of cross products
            sum_cp = self.calculate_sum_cross_products(month_data['Return'].values)
            monthly_sum_cp.append(sum_cp)
            
            # Calculate monthly volatility
            monthly_vol.append(np.std(month_data['Return'].values))
        
        monthly_returns['SUM_CP'] = monthly_sum_cp
        monthly_returns['VOL'] = monthly_vol
        
        # Run regressions as in Exhibit 5 of the paper
        # Panel A: Strategy Return vs SUM_CP
        X = sm.add_constant(monthly_returns['SUM_CP'])
        model = sm.OLS(monthly_returns['Strategy'], X)
        results = model.fit()
        print("\n=== Regression: Strategy Return vs SUM_CP ===")
        print(f"Constant: {results.params[0]:.4f} (t-stat: {results.tvalues[0]:.2f})")
        print(f"SUM_CP Coefficient: {results.params[1]:.4f} (t-stat: {results.tvalues[1]:.2f})")
        print(f"R-squared: {results.rsquared:.4f}")
        
        # Panel B: Strategy Return vs VOL
        X = sm.add_constant(monthly_returns['VOL'])
        model = sm.OLS(monthly_returns['Strategy'], X)
        results = model.fit()
        print("\n=== Regression: Strategy Return vs VOL ===")
        print(f"Constant: {results.params[0]:.4f} (t-stat: {results.tvalues[0]:.2f})")
        print(f"VOL Coefficient: {results.params[1]:.4f} (t-stat: {results.tvalues[1]:.2f})")
        print(f"R-squared: {results.rsquared:.4f}")
        
        # Panel C: Strategy Return vs SUM_CP and VOL
        X = sm.add_constant(pd.DataFrame({'SUM_CP': monthly_returns['SUM_CP'], 
                                         'VOL': monthly_returns['VOL']}))
        model = sm.OLS(monthly_returns['Strategy'], X)
        results = model.fit()
        print("\n=== Regression: Strategy Return vs SUM_CP and VOL ===")
        print(f"Constant: {results.params[0]:.4f} (t-stat: {results.tvalues[0]:.2f})")
        print(f"SUM_CP Coefficient: {results.params[1]:.4f} (t-stat: {results.tvalues[1]:.2f})")
        print(f"VOL Coefficient: {results.params[2]:.4f} (t-stat: {results.tvalues[2]:.2f})")
        print(f"R-squared: {results.rsquared:.4f}")
        
        # Create lagged VOL for forecasting
        monthly_returns['VOL_Lag'] = monthly_returns['VOL'].shift(1)
        monthly_returns['SUM_CP_Lag'] = monthly_returns['SUM_CP'].shift(1)
        
        # Split the sample based on previous month's volatility
        median_vol = monthly_returns['VOL_Lag'].median()
        low_vol = monthly_returns[monthly_returns['VOL_Lag'] <= median_vol]['Strategy']
        high_vol = monthly_returns[monthly_returns['VOL_Lag'] > median_vol]['Strategy']
        
        print("\n=== Strategy Performance Conditional on Previous Month's Volatility ===")
        print("Low Volatility (VOL_Lag <= Median):")
        print(f"  Mean Return: {low_vol.mean():.4f} ({low_vol.mean()*100:.2f}%)")
        print(f"  Std Dev: {low_vol.std():.4f}")
        print(f"  Sharpe Ratio: {np.sqrt(12) * low_vol.mean() / low_vol.std():.4f}")
        
        print("High Volatility (VOL_Lag > Median):")
        print(f"  Mean Return: {high_vol.mean():.4f} ({high_vol.mean()*100:.2f}%)")
        print(f"  Std Dev: {high_vol.std():.4f}")
        print(f"  Sharpe Ratio: {np.sqrt(12) * high_vol.mean() / high_vol.std():.4f}")
        
        # Panel D: Strategy Return vs Lagged SUM_CP and VOL
        X = sm.add_constant(pd.DataFrame({'SUM_CP_Lag': monthly_returns['SUM_CP_Lag'].dropna(), 
                                         'VOL_Lag': monthly_returns['VOL_Lag'].dropna()}))
        model = sm.OLS(monthly_returns['Strategy'].iloc[1:], X)
        results = model.fit()
        print("\n=== Regression: Strategy Return vs Lagged SUM_CP and VOL ===")
        print(f"Constant: {results.params[0]:.4f} (t-stat: {results.tvalues[0]:.2f})")
        print(f"SUM_CP_Lag Coefficient: {results.params[1]:.4f} (t-stat: {results.tvalues[1]:.2f})")
        print(f"VOL_Lag Coefficient: {results.params[2]:.4f} (t-stat: {results.tvalues[2]:.2f})")
        print(f"R-squared: {results.rsquared:.4f}")
        
        # Plot autocorrelations
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, len(autocorrs) + 1), autocorrs)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelations of Benchmark Index Returns')
        plt.show()
        
        # Plot prices
        plt.figure(figsize=(12, 6))
        plt.plot(etf_df.index, etf_df['Price'], label='Benchmark')
        plt.plot(etf_df.index, etf_df['LETF_Price'], label=f'{self.leverage_ratio}x Leveraged ETF')
        plt.plot(etf_df.index, etf_df['IETF_Price'], label=f'{self.leverage_ratio}x Inverse Leveraged ETF')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Price Comparison')
        plt.legend()
        plt.show()
        
        # Plot strategy returns vs benchmark returns
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_returns.index, monthly_returns['Benchmark_Cum'], label='Benchmark')
        plt.plot(monthly_returns.index, monthly_returns['Strategy_Cum'], label='Shorting Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns Comparison')
        plt.legend()
        plt.show()
        
        # Scatter plot of strategy returns vs SUM_CP
        plt.figure(figsize=(10, 6))
        plt.scatter(monthly_returns['SUM_CP'], monthly_returns['Strategy'])
        plt.xlabel('Sum of Cross Products')
        plt.ylabel('Strategy Return')
        plt.title('Strategy Return vs Sum of Cross Products')
        plt.grid(True)
        plt.show()
        
        # Scatter plot of strategy returns vs VOL
        plt.figure(figsize=(10, 6))
        plt.scatter(monthly_returns['VOL'], monthly_returns['Strategy'])
        plt.xlabel('Volatility')
        plt.ylabel('Strategy Return')
        plt.title('Strategy Return vs Volatility')
        plt.grid(True)
        plt.show()
    
    def run_simulation(self, mean_reverting=True, n_days=5000, volatility=0.015,
                      autocorr=-0.08, mean_reversion_strength=0.1,
                      plot_title_prefix=""):
        """
        Run a full simulation of the shorting leveraged ETF pairs strategy
        
        Parameters:
        mean_reverting (bool): Whether to use mean-reverting process or AR(1) process
        n_days (int): Number of days to simulate
        volatility (float): Daily volatility
        autocorr (float): Autocorrelation (for AR(1) process)
        mean_reversion_strength (float): Mean reversion strength (for mean-reverting process)
        plot_title_prefix (str): Prefix for plot titles
        
        Returns:
        tuple: Tuple of (etf_df, monthly_returns)
        """
        # Simulate benchmark index
        if mean_reverting:
            benchmark_df = self.simulate_mean_reverting_benchmark(
                n_days=n_days, 
                volatility=volatility,
                mean_reversion_strength=mean_reversion_strength
            )
        else:
            benchmark_df = self.simulate_benchmark_index(
                n_days=n_days,
                volatility=volatility,
                autocorr=autocorr
            )
        
        # Simulate leveraged ETFs
        etf_df = self.simulate_leveraged_etfs(benchmark_df)
        
        # Backtest strategy
        monthly_returns = self.backtest_monthly_shorting_strategy(etf_df)
        
        # Analyze performance
        print(f"=== {plot_title_prefix} Simulation Results ===\n")
        self.analyze_strategy_performance(etf_df, monthly_returns)
        
        return etf_df, monthly_returns
    
    def create_exhibit1_example(self):
        """
        Recreate Exhibit 1 from the paper with the six hypothetical scenarios
        
        Returns:
        pandas.DataFrame: DataFrame with the six scenarios
        """
        # Create the example cases from Exhibit 1
        cases = []
        
        # Case 1: Two consecutive positive daily returns (10% and 9.09%)
        rb_t1 = 0.10
        rb_t2 = 0.0909
        cases.append([rb_t1, rb_t2])
        
        # Case 2: Positive return followed by negative return (30% and -7.69%)
        rb_t1 = 0.30
        rb_t2 = -0.0769
        cases.append([rb_t1, rb_t2])
        
        # Case 3: Two consecutive negative daily returns (-10% and -11.11%)
        rb_t1 = -0.10
        rb_t2 = -0.1111
        cases.append([rb_t1, rb_t2])
        
        # Case 4: Negative return followed by positive return (-30% and 14.29%)
        rb_t1 = -0.30
        rb_t2 = 0.1429
        cases.append([rb_t1, rb_t2])
        
        # Case 5: Equal magnitude opposite returns (5% and -4.76%)
        rb_t1 = 0.05
        rb_t2 = -0.0476
        cases.append([rb_t1, rb_t2])
        
        # Case 6: Equal magnitude opposite returns with larger values (20% and -16.67%)
        rb_t1 = 0.20
        rb_t2 = -0.1667
        cases.append([rb_t1, rb_t2])
        
        # Calculate all metrics
        results = []
        
        for i, (rb_t1, rb_t2) in enumerate(cases):
            # Benchmark two-day return
            rb_2day = (1 + rb_t1) * (1 + rb_t2) - 1
            
            # LETF daily and two-day returns (using leverage = 3)
            rl_t1 = 3 * rb_t1
            rl_t2 = 3 * rb_t2
            rl_2day = (1 + rl_t1) * (1 + rl_t2) - 1
            
            # IETF daily and two-day returns (using leverage = -3)
            ri_t1 = -3 * rb_t1
            ri_t2 = -3 * rb_t2
            ri_2day = (1 + ri_t1) * (1 + ri_t2) - 1
            
            # Shorting strategy return
            rs_2day = -0.5 * rl_2day - 0.5 * ri_2day
            
            # Product of daily returns
            rb_product = rb_t1 * rb_t2
            
            results.append({
                'Case': i + 1,
                'rb_t1': rb_t1,
                'rb_t2': rb_t2,
                'rb_2day': rb_2day,
                'rl_2day': rl_2day,
                'ri_2day': ri_2day,
                'rs_2day': rs_2day,
                'rb_product': rb_product
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Format for display
        df['rb_t1'] = df['rb_t1'].map('{:.2%}'.format)
        df['rb_t2'] = df['rb_t2'].map('{:.2%}'.format)
        df['rb_2day'] = df['rb_2day'].map('{:.2%}'.format)
        df['rl_2day'] = df['rl_2day'].map('{:.2%}'.format)
        df['ri_2day'] = df['ri_2day'].map('{:.2%}'.format)
        df['rs_2day'] = df['rs_2day'].map('{:.2%}'.format)
        df['rb_product'] = df['rb_product'].map('{:.4%}'.format)
        
        return df

# Create the ShortingLeveragedETFPairs instance
strategy = ShortingLeveragedETFPairs(leverage_ratio=3)

# Recreate Exhibit 1 from the paper
exhibit1 = strategy.create_exhibit1_example()
print("Exhibit 1: Hypothetical Two-Day Performance of Shorting Triple-Leveraged ETF Pair")
print(exhibit1)
print("\n")

# Run simulation with mean-reverting benchmark
print("Simulating Mean-Reverting Market...")
etf_df_mr, monthly_returns_mr = strategy.run_simulation(
    mean_reverting=True,
    n_days=2500,
    volatility=0.015,
    mean_reversion_strength=0.1,
    plot_title_prefix="Mean-Reverting Market"
)

# Run simulation with trending benchmark (low auto-correlation)
print("\nSimulating Low Mean-Reversion Market...")
etf_df_tr, monthly_returns_tr = strategy.run_simulation(
    mean_reverting=False,
    n_days=2500,
    volatility=0.015,
    autocorr=-0.02,  # Low negative autocorrelation
    plot_title_prefix="Low Mean-Reversion Market"
)

# Run simulation with high volatility
print("\nSimulating High Volatility Market...")
etf_df_hv, monthly_returns_hv = strategy.run_simulation(
    mean_reverting=True,
    n_days=2500,
    volatility=0.025,  # Higher volatility
    mean_reversion_strength=0.1,
    plot_title_prefix="High Volatility Market"
)

# Compare the performance across different market conditions
monthly_returns_mr['Strategy_MR'] = monthly_returns_mr['Strategy']
monthly_returns_tr['Strategy_LMR'] = monthly_returns_tr['Strategy']
monthly_returns_hv['Strategy_HV'] = monthly_returns_hv['Strategy']

# Align indexes for fair comparison
common_idx = monthly_returns_mr.index.intersection(monthly_returns_tr.index).intersection(monthly_returns_hv.index)

comparison_df = pd.DataFrame({
    'Mean-Reverting': monthly_returns_mr.loc[common_idx, 'Strategy'],
    'Low Mean-Reversion': monthly_returns_tr.loc[common_idx, 'Strategy'],
    'High Volatility': monthly_returns_hv.loc[common_idx, 'Strategy']
})

# Calculate performance metrics
comparison_stats = pd.DataFrame({
    'Mean Monthly Return': comparison_df.mean(),
    'Std Dev': comparison_df.std(),
    'Sharpe Ratio': np.sqrt(12) * comparison_df.mean() / comparison_df.std(),
    'Min Return': comparison_df.min(),
    'Max Return': comparison_df.max()
})

print("\n=== Comparison of Strategy Performance Across Market Conditions ===")
print(comparison_stats)

# Plot cumulative returns comparison
plt.figure(figsize=(12, 6))
((1 + comparison_df).cumprod() - 1).plot()
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns of Shorting Strategy Across Market Conditions')
plt.grid(True)
plt.legend()
plt.show()

# Run a conditional strategy based on previous month's volatility
print("\n=== Conditional Strategy Based on Previous Month's Volatility ===")

# Combine dataframes and create strategy signal
combined_df = pd.DataFrame({
    'VOL_MR': monthly_returns_mr['VOL'],
    'VOL_LMR': monthly_returns_tr['VOL'],
    'VOL_HV': monthly_returns_hv['VOL'],
    'Strategy_MR': monthly_returns_mr['Strategy'],
    'Strategy_LMR': monthly_returns_tr['Strategy'],
    'Strategy_HV': monthly_returns_hv['Strategy']
})

# Create lagged volatility
combined_df['VOL_MR_Lag'] = combined_df['VOL_MR'].shift(1)
combined_df['VOL_LMR_Lag'] = combined_df['VOL_LMR'].shift(1)
combined_df['VOL_HV_Lag'] = combined_df['VOL_HV'].shift(1)

# Create conditional strategy
# Only short the LETF/IETF pair when previous month's volatility is above median
for market in ['MR', 'LMR', 'HV']:
    median_vol = combined_df[f'VOL_{market}_Lag'].median()
    combined_df[f'Conditional_{market}'] = np.where(
        combined_df[f'VOL_{market}_Lag'] > median_vol,
        combined_df[f'Strategy_{market}'],
        0  # No position when volatility is low
    )

# Drop NaN values due to lagging
combined_df = combined_df.dropna()

# Calculate performance metrics for conditional strategies
conditional_stats = pd.DataFrame({
    'Mean Monthly Return': [
        combined_df['Strategy_MR'].mean(),
        combined_df['Conditional_MR'].mean(),
        combined_df['Strategy_LMR'].mean(),
        combined_df['Conditional_LMR'].mean(),
        combined_df['Strategy_HV'].mean(),
        combined_df['Conditional_HV'].mean()
    ],
    'Std Dev': [
        combined_df['Strategy_MR'].std(),
        # Adjust std dev calculation for conditional strategies (only active months)
        combined_df['Conditional_MR'][combined_df['Conditional_MR'] != 0].std() if sum(combined_df['Conditional_MR'] != 0) > 1 else np.nan,
        combined_df['Strategy_LMR'].std(),
        combined_df['Conditional_LMR'][combined_df['Conditional_LMR'] != 0].std() if sum(combined_df['Conditional_LMR'] != 0) > 1 else np.nan,
        combined_df['Strategy_HV'].std(),
        combined_df['Conditional_HV'][combined_df['Conditional_HV'] != 0].std() if sum(combined_df['Conditional_HV'] != 0) > 1 else np.nan
    ]
})

# Calculate Sharpe ratio
conditional_stats['Sharpe Ratio'] = np.sqrt(12) * conditional_stats['Mean Monthly Return'] / conditional_stats['Std Dev']

# Scale by activity rate (percentage of months the strategy is active)
conditional_stats['Activity Rate'] = [
    1.0,  # Always active
    sum(combined_df['Conditional_MR'] != 0) / len(combined_df),
    1.0,  # Always active
    sum(combined_df['Conditional_LMR'] != 0) / len(combined_df),
    1.0,  # Always active
    sum(combined_df['Conditional_HV'] != 0) / len(combined_df)
]

# Calculate annualized returns
conditional_stats['Annualized Return'] = (1 + conditional_stats['Mean Monthly Return'])**12 - 1

conditional_stats.index = [
    'Mean-Reverting (Always)',
    'Mean-Reverting (Conditional)',
    'Low Mean-Reversion (Always)',
    'Low Mean-Reversion (Conditional)',
    'High Volatility (Always)',
    'High Volatility (Conditional)'
]

print(conditional_stats)

# Plot cumulative returns of conditional strategies
plt.figure(figsize=(12, 6))

# Calculate cumulative returns
for col in ['Strategy_MR', 'Conditional_MR', 'Strategy_HV', 'Conditional_HV']:
    # For conditional strategies, replace 0 with NaN for plotting
    if 'Conditional' in col:
        series = combined_df[col].copy()
        series[series == 0] = 0  # Replace 0 with zero return (no change)
        cumulative = (1 + series).cumprod() - 1
    else:
        cumulative = (1 + combined_df[col]).cumprod() - 1
    
    # Plot with appropriate label
    if col == 'Strategy_MR':
        label = 'Mean-Reverting (Always)'
    elif col == 'Conditional_MR':
        label = 'Mean-Reverting (Conditional)'
    elif col == 'Strategy_HV':
        label = 'High Volatility (Always)'
    elif col == 'Conditional_HV':
        label = 'High Volatility (Conditional)'
    else:
        label = col
    
    plt.plot(combined_df.index, cumulative, label=label)

plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns of Conditional Shorting Strategies')
plt.grid(True)
plt.legend()
plt.show()