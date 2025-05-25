import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CrossSectionalDispersionStrategy:
    """
    Implementation of the Cross-Sectional Dispersion (CSD) Strategy
    based on the paper "Cross-sectional dispersion and expected returns"
    by Thanos Verousis and Nikolaos Voukelatos
    """
    
    def __init__(self, tickers=None, start_date=None, end_date=None, bloomberg_conn=None):
        """
        Initialize the CSD strategy.
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers to use in the strategy
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format
        bloomberg_conn : object
            Bloomberg connection object
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.bloomberg_conn = bloomberg_conn
        self.market_index = "SPX Index"  # CRSP Value-Weighted Index (approximated with S&P 500)
        self.risk_free_rate = "US0003M Index"  # 3-month T-bill rate
        
        # Data containers
        self.stock_returns = None
        self.market_returns = None
        self.rf_returns = None
        self.csd_series = None
        self.delta_csd_series = None
        self.stock_betas = {}
        self.quintile_portfolios = {}
        self.spread_portfolio_returns = None
        
    def fetch_data(self):
        """
        Fetch daily price data from Bloomberg for stocks and market index.
        """
        print("Fetching data from Bloomberg...")
        
        if self.bloomberg_conn is None:
            raise ValueError("Bloomberg connection not initialized.")
            
        # Fetch daily adjusted prices for stocks
        stock_data = self.bloomberg_conn.bdh(
            self.tickers, 
            'PX_LAST', 
            self.start_date, 
            self.end_date
        )
        
        # Fetch daily adjusted prices for market index
        market_data = self.bloomberg_conn.bdh(
            self.market_index, 
            'PX_LAST', 
            self.start_date, 
            self.end_date
        )
        
        # Fetch risk-free rate
        rf_data = self.bloomberg_conn.bdh(
            self.risk_free_rate, 
            'PX_LAST', 
            self.start_date, 
            self.end_date
        )
        
        # Convert prices to returns
        self.stock_returns = stock_data.pct_change().dropna()
        self.market_returns = market_data.pct_change().dropna()
        self.rf_returns = rf_data / 252 / 100  # Convert annual percentage to daily decimal
        
        # Align dates
        common_dates = self.stock_returns.index.intersection(self.market_returns.index)
        common_dates = common_dates.intersection(self.rf_returns.index)
        
        self.stock_returns = self.stock_returns.loc[common_dates]
        self.market_returns = self.market_returns.loc[common_dates]
        self.rf_returns = self.rf_returns.loc[common_dates]
        
        print(f"Data fetched for {len(self.stock_returns)} days and {len(self.tickers)} stocks.")
    
    def calculate_csd(self):
        """
        Calculate the cross-sectional dispersion of stock returns.
        
        CSD_t = ∑|r_i,t - r_mkt,t| / (N-1)
        """
        print("Calculating cross-sectional dispersion...")
        
        # Calculate cross-sectional dispersion for each day
        csd = pd.Series(index=self.stock_returns.index)
        
        for date in self.stock_returns.index:
            # Get stock returns for the day
            daily_returns = self.stock_returns.loc[date]
            
            # Get market return for the day
            market_return = self.market_returns.loc[date].values[0]
            
            # Calculate absolute deviations from market return
            abs_deviations = np.abs(daily_returns - market_return)
            
            # Calculate CSD
            n_stocks = len(daily_returns.dropna())
            if n_stocks > 1:  # Ensure we have at least 2 stocks
                csd[date] = abs_deviations.sum() / (n_stocks - 1)
            else:
                csd[date] = np.nan
        
        self.csd_series = csd.dropna()
        
        # Calculate first differences of CSD
        self.delta_csd_series = self.csd_series.diff().dropna()
        
        print("CSD calculated.")
        
    def estimate_betas(self, window=21):
        """
        Estimate dispersion betas for each stock using a rolling window.
        
        Parameters:
        -----------
        window : int
            Number of days to use for beta estimation (default: 21, approx. 1 month)
        """
        print("Estimating dispersion betas...")
        
        # Ensure we have delta CSD
        if self.delta_csd_series is None:
            self.calculate_csd()
        
        # Create a DataFrame to store betas
        dates = self.delta_csd_series.index[window:]
        beta_df = pd.DataFrame(index=dates, columns=self.tickers)
        
        # Estimate betas for each stock
        for ticker in self.tickers:
            # Get excess returns for the stock
            stock_excess_returns = self.stock_returns[ticker] - self.rf_returns.values.flatten()
            
            # Get market excess returns
            market_excess_returns = self.market_returns.values.flatten() - self.rf_returns.values.flatten()
            
            # For each date, estimate beta using the previous window days
            for i, date in enumerate(dates):
                # Get window indices
                start_idx = i
                end_idx = i + window
                
                # Get data for the window
                y = stock_excess_returns.iloc[start_idx:end_idx].values
                X = np.column_stack([
                    np.ones(window),
                    market_excess_returns[start_idx:end_idx],
                    self.delta_csd_series.iloc[start_idx:end_idx].values
                ])
                
                # Check for NaNs
                if np.isnan(y).any() or np.isnan(X).any():
                    beta_df.loc[date, ticker] = np.nan
                    continue
                
                # Ensure we have at least 15 valid observations
                if len(y) < 15:
                    beta_df.loc[date, ticker] = np.nan
                    continue
                
                # Estimate regression
                try:
                    model = sm.OLS(y, X)
                    results = model.fit()
                    
                    # Store the dispersion beta (3rd coefficient)
                    beta_df.loc[date, ticker] = results.params[2]
                except:
                    beta_df.loc[date, ticker] = np.nan
        
        self.stock_betas = beta_df
        print("Betas estimated.")
        
    def form_portfolios(self, rebalance_freq='M'):
        """
        Form quintile portfolios based on dispersion betas.
        
        Parameters:
        -----------
        rebalance_freq : str
            Rebalancing frequency: 'M' for monthly, 'W' for weekly
        """
        print("Forming portfolios...")
        
        # Resample dates based on rebalance frequency
        if rebalance_freq == 'M':
            rebalance_dates = pd.date_range(
                self.stock_betas.index[0], 
                self.stock_betas.index[-1], 
                freq='MS'
            )
        elif rebalance_freq == 'W':
            rebalance_dates = pd.date_range(
                self.stock_betas.index[0], 
                self.stock_betas.index[-1], 
                freq='W-MON'
            )
        else:
            raise ValueError("rebalance_freq must be 'M' or 'W'")
        
        # Adjust to actual trading days
        rebalance_dates = [d for d in rebalance_dates if d in self.stock_betas.index]
        
        # Initialize portfolio returns
        portfolio_returns = {
            'Q1': pd.Series(index=self.stock_returns.index),  # Lowest beta
            'Q2': pd.Series(index=self.stock_returns.index),
            'Q3': pd.Series(index=self.stock_returns.index),
            'Q4': pd.Series(index=self.stock_returns.index),
            'Q5': pd.Series(index=self.stock_returns.index),  # Highest beta
            'N': pd.Series(index=self.stock_returns.index),   # Negative beta
            'P': pd.Series(index=self.stock_returns.index)    # Positive beta
        }
        
        # For each rebalance date
        for i, rebalance_date in enumerate(rebalance_dates[:-1]):
            # Get next rebalance date
            next_rebalance_date = rebalance_dates[i+1]
            
            # Get betas on rebalance date
            betas = self.stock_betas.loc[rebalance_date].dropna()
            
            if len(betas) < 5:  # Ensure we have enough stocks
                continue
            
            # Sort stocks into quintiles based on betas
            sorted_betas = betas.sort_values()
            n_stocks = len(sorted_betas)
            quintile_size = n_stocks // 5
            
            # Assign stocks to quintiles
            quintiles = {}
            for q in range(1, 6):
                if q < 5:
                    start_idx = (q-1) * quintile_size
                    end_idx = q * quintile_size
                    quintiles[f'Q{q}'] = sorted_betas.index[start_idx:end_idx].tolist()
                else:
                    start_idx = (q-1) * quintile_size
                    quintiles[f'Q{q}'] = sorted_betas.index[start_idx:].tolist()
            
            # Assign stocks to N (negative beta) and P (positive beta) portfolios
            quintiles['N'] = sorted_betas[sorted_betas < 0].index.tolist()
            quintiles['P'] = sorted_betas[sorted_betas >= 0].index.tolist()
            
            # Get portfolio returns for the holding period
            holding_period = self.stock_returns.loc[rebalance_date:next_rebalance_date].index
            for date in holding_period:
                if date == rebalance_date:
                    continue
                
                # Get daily returns
                daily_returns = self.stock_returns.loc[date]
                
                # Calculate value-weighted returns for each quintile
                for portfolio, stocks in quintiles.items():
                    if not stocks:  # Skip if no stocks in portfolio
                        continue
                    
                    # Get returns for the stocks in the portfolio
                    port_returns = daily_returns[stocks].dropna()
                    
                    if len(port_returns) == 0:
                        continue
                    
                    # Equal-weighted portfolio for simplicity
                    # In the paper they use value-weighted, but we'd need market caps
                    portfolio_returns[portfolio][date] = port_returns.mean()
        
        # Store portfolio returns
        self.quintile_portfolios = portfolio_returns
        
        # Calculate spread portfolio returns (Q1-Q5 and N-P)
        self.spread_portfolio_returns = {
            '1-5': portfolio_returns['Q1'] - portfolio_returns['Q5'],
            'N-P': portfolio_returns['N'] - portfolio_returns['P']
        }
        
        print("Portfolios formed.")
        
    def calculate_performance(self, monthly=True):
        """
        Calculate performance metrics for the portfolios.
        
        Parameters:
        -----------
        monthly : bool
            Whether to report monthly (True) or daily (False) returns
        """
        print("Calculating performance metrics...")
        
        # Convert to monthly if required
        if monthly:
            portfolio_returns = {}
            for name, returns in self.quintile_portfolios.items():
                portfolio_returns[name] = returns.resample('M').apply(
                    lambda x: (1 + x).prod() - 1
                )
            
            spread_returns = {}
            for name, returns in self.spread_portfolio_returns.items():
                spread_returns[name] = returns.resample('M').apply(
                    lambda x: (1 + x).prod() - 1
                )
            
            # Convert market and risk-free to monthly
            market_monthly = self.market_returns.resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            rf_monthly = self.rf_returns.resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
        else:
            portfolio_returns = self.quintile_portfolios
            spread_returns = self.spread_portfolio_returns
            market_monthly = self.market_returns
            rf_monthly = self.rf_returns
        
        # Calculate performance metrics
        results = {}
        
        # Quintile portfolios
        for name, returns in portfolio_returns.items():
            clean_returns = returns.dropna()
            results[name] = {
                'Mean Return': clean_returns.mean(),
                'Std Dev': clean_returns.std(),
                'Sharpe Ratio': (clean_returns.mean() - rf_monthly.mean().values[0]) / clean_returns.std(),
                'Observations': len(clean_returns)
            }
        
        # Spread portfolios
        for name, returns in spread_returns.items():
            clean_returns = returns.dropna()
            
            # Calculate Fama-French-Carhart alpha
            if len(clean_returns) > 0:
                y = clean_returns.values
                X = sm.add_constant(np.column_stack([
                    market_monthly.loc[clean_returns.index].values.flatten() - rf_monthly.loc[clean_returns.index].values.flatten(),
                    # Here we would add SMB, HML, and MOM factors if available
                ]))
                
                try:
                    model = sm.OLS(y, X)
                    results_reg = model.fit()
                    alpha = results_reg.params[0]
                    alpha_tstat = results_reg.tvalues[0]
                except:
                    alpha = np.nan
                    alpha_tstat = np.nan
            else:
                alpha = np.nan
                alpha_tstat = np.nan
            
            results[name] = {
                'Mean Return': clean_returns.mean(),
                'Std Dev': clean_returns.std(),
                't-stat': (clean_returns.mean() / (clean_returns.std() / np.sqrt(len(clean_returns)))) if len(clean_returns) > 0 else np.nan,
                'Sharpe Ratio': (clean_returns.mean() - rf_monthly.mean().values[0]) / clean_returns.std() if len(clean_returns) > 0 else np.nan,
                'Alpha': alpha,
                'Alpha t-stat': alpha_tstat,
                'Observations': len(clean_returns)
            }
        
        return pd.DataFrame(results).T
    
    def plot_results(self):
        """
        Plot the cumulative returns of the portfolios.
        """
        # Create quintile portfolio cumulative returns
        cumulative_returns = {}
        for name, returns in self.quintile_portfolios.items():
            cumulative_returns[name] = (1 + returns.dropna()).cumprod()
        
        # Create spread portfolio cumulative returns
        for name, returns in self.spread_portfolio_returns.items():
            cumulative_returns[name] = (1 + returns.dropna()).cumprod()
        
        # Create market cumulative return
        cumulative_returns['Market'] = (1 + self.market_returns.dropna()).cumprod()
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Plot quintile portfolios
        for i in range(1, 6):
            plt.plot(cumulative_returns[f'Q{i}'], label=f'Quintile {i}')
        
        # Plot spread portfolios with thicker lines
        plt.plot(cumulative_returns['1-5'], label='1-5 Spread', linewidth=2.5, color='black')
        plt.plot(cumulative_returns['N-P'], label='N-P Spread', linewidth=2.5, color='darkred')
        
        # Plot market
        plt.plot(cumulative_returns['Market'], label='Market', linewidth=2.5, linestyle='--', color='blue')
        
        plt.title('Cumulative Returns of CSD-Based Portfolios')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig('csd_strategy_returns.png')
        plt.show()
        
    def run_strategy(self):
        """
        Run the full CSD strategy pipeline.
        """
        # Fetch data
        self.fetch_data()
        
        # Calculate CSD
        self.calculate_csd()
        
        # Estimate betas
        self.estimate_betas()
        
        # Form portfolios
        self.form_portfolios()
        
        # Calculate performance
        performance = self.calculate_performance()
        print("\nPortfolio Performance:")
        print(performance)
        
        # Print average pre-formation betas for each quintile
        self.print_quintile_betas()
        
        # Plot results
        self.plot_results()
        
        return performance
    
    def print_quintile_betas(self):
        """
        Print the average pre-formation betas for each quintile portfolio.
        """
        rebalance_dates = pd.date_range(
            self.stock_betas.index[0], 
            self.stock_betas.index[-1], 
            freq='MS'
        )
        rebalance_dates = [d for d in rebalance_dates if d in self.stock_betas.index]
        
        pre_formation_betas = {
            'Q1': [],
            'Q2': [],
            'Q3': [],
            'Q4': [],
            'Q5': [],
            'N': [],
            'P': []
        }
        
        for rebalance_date in rebalance_dates:
            betas = self.stock_betas.loc[rebalance_date].dropna()
            if len(betas) < 5:
                continue
            
            sorted_betas = betas.sort_values()
            n_stocks = len(sorted_betas)
            quintile_size = n_stocks // 5
            
            # Calculate average beta for each quintile
            for q in range(1, 6):
                if q < 5:
                    start_idx = (q-1) * quintile_size
                    end_idx = q * quintile_size
                    quintile_betas = sorted_betas.iloc[start_idx:end_idx]
                else:
                    start_idx = (q-1) * quintile_size
                    quintile_betas = sorted_betas.iloc[start_idx:]
                
                pre_formation_betas[f'Q{q}'].append(quintile_betas.mean())
            
            # Calculate average beta for N and P portfolios
            pre_formation_betas['N'].append(sorted_betas[sorted_betas < 0].mean())
            pre_formation_betas['P'].append(sorted_betas[sorted_betas >= 0].mean())
        
        # Calculate the average across all rebalance dates
        for portfolio, betas in pre_formation_betas.items():
            avg_beta = np.mean(betas) if betas else np.nan
            print(f"Average pre-formation beta for {portfolio}: {avg_beta:.4f}")


class ImprovedSimulatedCrossSectionalDispersionStrategy(CrossSectionalDispersionStrategy):
    """
    An improved version of the simulated strategy with more realistic relationships
    between dispersion betas and returns as described in the paper.
    """
    
    def fetch_data(self):
        """
        Generate simulated price data that reflects the relationship described in the paper:
        Stocks with higher dispersion betas have lower expected returns.
        """
        print("Generating simulated data with negative dispersion risk premium...")
        
        # Parse dates
        start_date = datetime.strptime(self.start_date, '%Y%m%d')
        end_date = datetime.strptime(self.end_date, '%Y%m%d')
        
        # Generate date range
        business_days = pd.bdate_range(start=start_date, end=end_date)
        num_days = len(business_days)
        
        # Number of stocks
        num_stocks = len(self.tickers)
        
        # Simulate true dispersion betas for each stock
        np.random.seed(42)
        # These are the "true" dispersion betas for each stock
        true_dispersion_betas = np.random.uniform(-7, 7, num_stocks)
        
        # Risk-free rate (constant for simplicity)
        rf_daily = 0.02 / 252  # 2% annual rate
        
        # Create risk-free series
        rf_returns = pd.DataFrame(
            np.ones(num_days) * rf_daily,
            index=business_days,
            columns=[self.risk_free_rate]
        )
        
        # Simulate cross-sectional dispersion with realistic AR(1) dynamics
        csd = np.zeros(num_days)
        csd[0] = 0.01  # Initial value
        csd_ar_param = 0.7  # Autocorrelation parameter
        csd_vol = 0.002  # Volatility of innovations
        
        for t in range(1, num_days):
            csd[t] = csd_ar_param * csd[t-1] + np.random.normal(0, csd_vol)
            if csd[t] < 0.005:  # Ensure CSD remains positive
                csd[t] = 0.005
        
        # First differences in CSD
        delta_csd = np.diff(csd, prepend=csd[0])
        
        # Simulate market returns with realistic properties
        market_mu = 0.0005  # Mean daily return
        market_sigma = 0.01  # Daily volatility
        market_returns_values = np.random.normal(market_mu, market_sigma, num_days)
        
        # Store market returns
        market_returns = pd.DataFrame(
            market_returns_values,
            index=business_days,
            columns=[self.market_index]
        )
        
        # Initialize stock returns matrix
        stock_returns_values = np.zeros((num_days, num_stocks))
        
        # Dispersion risk premium (negative as per the paper)
        dispersion_risk_premium = -0.01  # Monthly premium
        
        # Create a negative relationship between dispersion beta and expected returns
        # Stocks with higher (more positive) dispersion betas should have lower expected returns
        for i in range(num_stocks):
            # Baseline expected return 
            baseline_return = market_mu
            
            # Adjust expected return based on dispersion beta
            # Higher beta -> lower expected return (negative premium)
            expected_return = baseline_return + (dispersion_risk_premium/22) * true_dispersion_betas[i]
            
            # Market beta (uncorrelated with dispersion beta)
            market_beta = np.random.uniform(0.7, 1.3)
            
            # Idiosyncratic volatility
            idio_vol = np.random.uniform(0.01, 0.03)
            
            # Generate stock returns with these properties
            for t in range(num_days):
                # Systematic component from market
                systematic_return = market_beta * market_returns_values[t]
                
                # Component from dispersion risk
                dispersion_component = true_dispersion_betas[i] * delta_csd[t]
                
                # Idiosyncratic component
                idiosyncratic_return = np.random.normal(0, idio_vol)
                
                # Total return combining components
                # The expected contribution from dispersion is already in the expected return
                stock_returns_values[t, i] = expected_return + systematic_return + dispersion_component + idiosyncratic_return
        
        # Store in dataframes
        stock_returns = pd.DataFrame(
            stock_returns_values,
            index=business_days,
            columns=self.tickers
        )
        
        # Store data
        self.stock_returns = stock_returns
        self.market_returns = market_returns
        self.rf_returns = rf_returns
        
        # Store the "true" dispersion betas for later comparison
        self.true_dispersion_betas = pd.Series(true_dispersion_betas, index=self.tickers)
        
        # Also store the CSD and delta_CSD series for reference
        self.csd_series = pd.Series(csd, index=business_days)
        self.delta_csd_series = pd.Series(delta_csd, index=business_days)
        
        print(f"Simulated data generated for {len(self.stock_returns)} days and {len(self.tickers)} stocks.")
        print(f"Data designed with a negative dispersion risk premium of {dispersion_risk_premium*100:.2f}% per month.")


def main():
    # Use simulated data for S&P 500 constituents
    # Generate 100 stock symbols
    tickers = [f"STOCK_{i}" for i in range(1, 101)]
    
    # Run with improved simulated data that reflects the paper's findings
    strategy = ImprovedSimulatedCrossSectionalDispersionStrategy(
        tickers=tickers,
        start_date="20180101",
        end_date="20230101"
    )
    
    # Run the strategy
    strategy.run_strategy()


if __name__ == "__main__":
    main()