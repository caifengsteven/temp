import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import logging
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the Bloomberg API
try:
    import blpapi
    BLOOMBERG_AVAILABLE = True
except ImportError:
    logger.warning("Bloomberg API (blpapi) not available. Will use fallback data.")
    BLOOMBERG_AVAILABLE = False

class PortfolioStrategies:
    """Class implementing the portfolio strategies from the paper."""
    
    def __init__(self):
        """Initialize the portfolio strategies class."""
        self.data = None
        self.tickers = None
        self.index_ticker = None
        self.mean_target = None
        self.kappa_m = None
        self.l = None
    
    def load_data_from_bloomberg(self, 
                            tickers: List[str], 
                            index_ticker: str,
                            start_date: dt.datetime,
                            end_date: dt.datetime) -> pd.DataFrame:
        """
        Load data from Bloomberg for the given tickers and index.
        
        Args:
            tickers: List of asset tickers
            index_ticker: Ticker for the benchmark index
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with asset and index returns
        """
        if not BLOOMBERG_AVAILABLE:
            logger.error("Bloomberg API not available. Cannot fetch data.")
            return None
        
        logger.info(f"Connecting to Bloomberg to fetch data for {len(tickers)} assets and index {index_ticker}...")
        
        # Store the tickers
        self.tickers = tickers
        self.index_ticker = index_ticker
        
        try:
            # Initialize Bloomberg session
            session_options = blpapi.SessionOptions()
            session_options.setServerHost("localhost")
            session_options.setServerPort(8194)
            session = blpapi.Session(session_options)
            
            # Start session
            if not session.start():
                logger.error("Failed to start Bloomberg session")
                return None
                
            # Open reference data service
            if not session.openService("//blp/refdata"):
                logger.error("Failed to open //blp/refdata service")
                session.stop()
                return None
                
            refdata_service = session.getService("//blp/refdata")
            
            # Create request for historical data
            request = refdata_service.createRequest("HistoricalDataRequest")
            
            # Add all tickers including the index
            all_tickers = tickers + [index_ticker]
            for ticker in all_tickers:
                request.getElement("securities").appendValue(ticker)
            
            # Add fields
            request.getElement("fields").appendValue("PX_LAST")
            
            # Set date range
            request.set("periodicitySelection", "MONTHLY")
            request.set("startDate", start_date.strftime("%Y%m%d"))
            request.set("endDate", end_date.strftime("%Y%m%d"))
            
            # Send request
            logger.info("Sending request to Bloomberg...")
            session.sendRequest(request)
            
            # Process response
            data_dict = {}
            
            while True:
                event = session.nextEvent(500)  # Timeout in milliseconds
                
                for msg in event:
                    if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                        security_data = msg.getElement("securityData")
                        ticker = security_data.getElementAsString("security")
                        logger.info(f"Processing data for {ticker}")
                        
                        field_data = security_data.getElement("fieldData")
                        
                        dates = []
                        prices = []
                        
                        for i in range(field_data.numValues()):
                            field_value = field_data.getValue(i)
                            
                            # Fixed date handling - check the type before calling .date()
                            date_element = field_value.getElementAsDatetime("date")
                            if isinstance(date_element, dt.datetime):
                                date = date_element.date()
                            else:
                                date = date_element  # already a date object
                            
                            price = field_value.getElementAsFloat("PX_LAST")
                            
                            dates.append(date)
                            prices.append(price)
                        
                        data_dict[ticker] = pd.Series(prices, index=dates)
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            # Close the session
            session.stop()
            
            if not data_dict:
                logger.error("No data retrieved from Bloomberg")
                return None
            
            # Convert to DataFrame
            logger.info("Creating DataFrame from Bloomberg data")
            data_df = pd.DataFrame(data_dict)
            
            # Calculate returns
            logger.info("Calculating returns")
            returns_df = data_df.pct_change().dropna()
            
            # Separate index and asset returns
            asset_returns = returns_df[tickers]
            index_returns = returns_df[index_ticker]
            
            # Store the data
            self.data = {
                'asset_returns': asset_returns,
                'index_returns': index_returns
            }
            
            logger.info(f"Successfully loaded {len(returns_df)} monthly returns for {len(tickers)} assets and index.")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching Bloomberg data: {e}")
            # Try to close the session if it exists
            if 'session' in locals():
                try:
                    session.stop()
                except:
                    pass
            return None
    
    def generate_synthetic_data(self, 
                              tickers: List[str], 
                              index_ticker: str,
                              start_date: dt.datetime,
                              end_date: dt.datetime,
                              seed: int = 42) -> Dict:
        """
        Generate synthetic return data for backtesting when Bloomberg is unavailable.
        
        Args:
            tickers: List of asset tickers
            index_ticker: Ticker for the benchmark index
            start_date: Start date for synthetic data
            end_date: End date for synthetic data
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with asset and index returns
        """
        np.random.seed(seed)
        
        # Store the tickers
        self.tickers = tickers
        self.index_ticker = index_ticker
        
        # Generate dates (monthly frequency)
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Generate index returns (normal distribution)
        index_mean = 0.008  # ~10% annually
        index_vol = 0.045   # ~15% annually
        index_returns = pd.Series(np.random.normal(index_mean, index_vol, len(dates)), index=dates, name=index_ticker)
        
        # Generate asset returns (market model with idiosyncratic returns)
        asset_returns_data = {}
        
        for ticker in tickers:
            # Generate random beta between 0.7 and 1.5
            beta = np.random.uniform(0.7, 1.5)
            
            # Generate random alpha between -0.002 and 0.004 monthly
            alpha = np.random.uniform(-0.002, 0.004)
            
            # Generate idiosyncratic volatility between 0.03 and 0.08 monthly
            idio_vol = np.random.uniform(0.03, 0.08)
            
            # Generate returns using market model: r_i = alpha + beta * r_m + epsilon
            epsilon = np.random.normal(0, idio_vol, len(dates))
            asset_return = alpha + beta * index_returns.values + epsilon
            
            asset_returns_data[ticker] = asset_return
        
        # Create asset returns DataFrame
        asset_returns = pd.DataFrame(asset_returns_data, index=dates)
        
        # Store the data
        self.data = {
            'asset_returns': asset_returns,
            'index_returns': index_returns
        }
        
        logger.info(f"Generated {len(dates)} months of synthetic return data for {len(tickers)} assets and index.")
        
        return self.data
    
    def load_data_from_csv(self, 
                          asset_returns_file: str,
                          index_returns_file: str = None) -> Dict:
        """
        Load data from CSV files.
        
        Args:
            asset_returns_file: Path to CSV file with asset returns
            index_returns_file: Path to CSV file with index returns (if separate)
            
        Returns:
            Dictionary with asset and index returns
        """
        # Load asset returns
        asset_returns = pd.read_csv(asset_returns_file, index_col=0, parse_dates=True)
        
        # Get tickers from columns
        self.tickers = asset_returns.columns.tolist()
        
        # Load index returns
        if index_returns_file:
            index_returns = pd.read_csv(index_returns_file, index_col=0, parse_dates=True)
            index_returns = index_returns.squeeze()  # Convert DataFrame to Series
            self.index_ticker = index_returns.name
        else:
            # If no separate index file, assume the last column is the index
            self.index_ticker = self.tickers[-1]
            index_returns = asset_returns[self.index_ticker]
            self.tickers = self.tickers[:-1]
            asset_returns = asset_returns[self.tickers]
        
        # Store the data
        self.data = {
            'asset_returns': asset_returns,
            'index_returns': index_returns
        }
        
        logger.info(f"Successfully loaded {len(asset_returns)} returns for {len(self.tickers)} assets and index.")
        
        return self.data
    
    def set_mean_target(self, mean_target: float) -> None:
        """
        Set the target mean return for the portfolios.
        
        Args:
            mean_target: Target mean return (monthly)
        """
        self.mean_target = mean_target
    
    def calculate_parameters(self, 
                            returns: pd.DataFrame, 
                            index_returns: pd.Series) -> Tuple:
        """
        Calculate necessary parameters for portfolio construction.
        
        Args:
            returns: DataFrame with asset returns
            index_returns: Series with index returns
            
        Returns:
            Tuple with mu, Sigma, alpha0, alpha1, alpha2, l, and kappa_m
        """
        # Calculate moments
        mu = returns.mean().values
        Sigma = returns.cov().values
        
        # Calculate covariance with the index
        cov_with_index = returns.apply(lambda x: x.cov(index_returns)).values
        
        # Index volatility
        sigma_b_squared = index_returns.var()
        
        # Calculate alpha0, alpha1, alpha2
        Sigma_inv = np.linalg.inv(Sigma)
        ones = np.ones(len(mu))
        
        # Alpha0: GMVP weights
        denominator = ones.T @ Sigma_inv @ ones
        alpha0 = (Sigma_inv @ ones) / denominator
        
        # Alpha1: Arbitrage portfolio weights
        B = Sigma_inv - (Sigma_inv @ ones.reshape(-1, 1) @ ones.reshape(1, -1) @ Sigma_inv) / denominator
        alpha1 = B @ mu
        
        # Alpha2: Index-tracking arbitrage portfolio
        c = sigma_b_squared * cov_with_index / sigma_b_squared  # This is just cov_with_index
        alpha2 = B @ c
        
        # Calculate eta0, eta1, eta2
        eta0 = mu @ alpha0
        eta1 = mu @ alpha1
        eta2 = mu @ alpha2
        
        # Calculate l
        l = eta2 / eta1
        
        # Calculate minimum feasible mean
        eta2_pos = max(eta2, 0)
        m_min = eta0 + eta2_pos
        
        # Ensure target mean is feasible
        if self.mean_target is None:
            self.mean_target = 1.5 * m_min
            logger.info(f"Setting target mean to {self.mean_target:.4f} (1.5 * m_min)")
        elif self.mean_target <= m_min:
            logger.warning(f"Target mean {self.mean_target} is below minimum {m_min}. Adjusting to 1.1*m_min.")
            self.mean_target = 1.1 * m_min
        
        # Calculate kappa_m
        self.kappa_m = eta1 / (self.mean_target - eta0)
        self.l = l
        
        return mu, Sigma, alpha0, alpha1, alpha2, l, self.kappa_m
    
    def construct_portfolio(self, 
                          epsilon: float, 
                          returns: pd.DataFrame, 
                          index_returns: pd.Series) -> np.ndarray:
        """
        Construct portfolio with the given epsilon.
        
        Args:
            epsilon: The epsilon parameter (0 for MVEP, 1 for MTEP)
            returns: DataFrame with asset returns
            index_returns: Series with index returns
            
        Returns:
            Array of portfolio weights
        """
        mu, Sigma, alpha0, alpha1, alpha2, l, kappa_m = self.calculate_parameters(returns, index_returns)
        
        # Construct portfolio
        weights = alpha0 + (1/kappa_m - epsilon*l)*alpha1 + epsilon*alpha2
        
        return weights
    
    def calculate_implicit_value(self, 
                              sample_size: int, 
                              returns: pd.DataFrame, 
                              index_returns: pd.Series) -> float:
        """
        Calculate the implicit value of index tracking.
        
        Args:
            sample_size: The sample size T used for estimation
            returns: DataFrame with asset returns
            index_returns: Series with index returns
            
        Returns:
            The implicit value metric
        """
        mu, Sigma, alpha0, alpha1, alpha2, l, kappa_m = self.calculate_parameters(returns, index_returns)
        
        # Number of assets
        d = len(mu)
        
        # Calculate sigma0_squared, sigma1_squared, sigma2_squared
        sigma0_squared = alpha0 @ Sigma @ alpha0
        sigma1_squared = alpha1 @ Sigma @ alpha1
        sigma2_squared = alpha2 @ Sigma @ alpha2
        
        # Index volatility
        sigma_b_squared = index_returns.var()
        
        # Calculate beta
        cov_with_index = returns.apply(lambda x: x.cov(index_returns)).values
        beta = cov_with_index / sigma_b_squared
        
        # Calculate eta0, eta1, eta2
        eta0 = mu @ alpha0
        eta1 = mu @ alpha1
        eta2 = mu @ alpha2
        
        # Calculate d constants
        T = sample_size
        d1 = ((T - 1)**2) / ((T - d)*(T - d - 1)*(T - d - 3))
        d2 = (T - 1) / (T - d - 1)
        d3 = (T - d + 1) / (T - d - 1) * d1
        d4 = (d - 1) / (T - d - 1)
        
        # Calculate f(d, T, sigma1_squared)
        f = d1 * (sigma1_squared + (T - 2) / T)
        
        # Calculate g(d, T, sigma1_squared)
        g = sigma1_squared * (d3 + d2) + f * (d - 1)
        
        # Calculate lambda
        lambda_term = sigma_b_squared * (1 - 2 * beta @ alpha0) - sigma2_squared
        
        # Calculate F1(m) and F2(m)
        m = self.mean_target
        F1 = 1 - (2 - eta2 / (m - eta0)) * d2
        F2 = eta2 * (2*m - 2*eta0 - eta2) / (eta1**2)
        
        # Calculate the implicit value
        implicit_value = eta2 * F1 + (kappa_m / 2) * (g * F2 - (d4 * lambda_term + sigma2_squared))
        
        # Calculate normalized implicit value
        expected_utility_mvep = mu @ (alpha0 + (1/kappa_m)*alpha1) - (kappa_m/2) * ((alpha0 + (1/kappa_m)*alpha1) @ Sigma @ (alpha0 + (1/kappa_m)*alpha1))
        normalized_implicit_value = implicit_value / abs(expected_utility_mvep)
        
        return normalized_implicit_value
    
    def backtest(self, 
               window_size: int = 60, 
               rebalance_freq: int = 1) -> pd.DataFrame:
        """
        Backtest the MVEP and MTEP strategies.
        
        Args:
            window_size: Size of the rolling window in months
            rebalance_freq: Rebalancing frequency in months
            
        Returns:
            DataFrame with backtest results
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data_from_bloomberg or load_data_from_csv first.")
            return None
        
        asset_returns = self.data['asset_returns']
        index_returns = self.data['index_returns']
        
        # Ensure index alignment
        asset_returns = asset_returns.loc[asset_returns.index.isin(index_returns.index)]
        index_returns = index_returns.loc[index_returns.index.isin(asset_returns.index)]
        
        # Initialize results
        results = {
            'date': [],
            'mvep_weights': [],
            'mtep_weights': [],
            'mvep_return': [],
            'mtep_return': [],
            'diff_return': [],
            'index_return': [],
            'implicit_value': []
        }
        
        # Backtest
        dates = asset_returns.index[window_size:]
        
        for i in range(0, len(dates), rebalance_freq):
            if i >= len(asset_returns) - window_size:
                break
                
            current_date = dates[i]
            logger.info(f"Processing date: {current_date}")
            
            # Get training data
            end_idx = i + window_size
            train_returns = asset_returns.iloc[i:end_idx]
            train_index_returns = index_returns.iloc[i:end_idx]
            
            # Get test data (next month)
            if end_idx < len(asset_returns):
                test_returns = asset_returns.iloc[end_idx]
                test_index_return = index_returns.iloc[end_idx]
            else:
                logger.warning(f"End of data reached at index {i}. Stopping backtest.")
                break
            
            # Construct portfolios
            try:
                mvep_weights = self.construct_portfolio(0, train_returns, train_index_returns)
                mtep_weights = self.construct_portfolio(1, train_returns, train_index_returns)
                
                # Calculate portfolio returns
                mvep_return = np.sum(mvep_weights * test_returns)
                mtep_return = np.sum(mtep_weights * test_returns)
                diff_return = mtep_return - mvep_return
                
                # Calculate implicit value
                implicit_value = self.calculate_implicit_value(window_size, train_returns, train_index_returns)
                
                # Store results
                results['date'].append(current_date)
                results['mvep_weights'].append(mvep_weights)
                results['mtep_weights'].append(mtep_weights)
                results['mvep_return'].append(mvep_return)
                results['mtep_return'].append(mtep_return)
                results['diff_return'].append(diff_return)
                results['index_return'].append(test_index_return)
                results['implicit_value'].append(implicit_value)
                
            except Exception as e:
                logger.error(f"Error in backtest for date {current_date}: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        logger.info(f"Backtest completed with {len(results_df)} periods")
        
        return results_df
    
    def calculate_performance_metrics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics for the backtest results.
        
        Args:
            results_df: DataFrame with backtest results
            
        Returns:
            DataFrame with performance metrics
        """
        if len(results_df) == 0:
            logger.error("No backtest results to analyze")
            return pd.DataFrame()
            
        # Calculate annualized metrics
        metrics = {}
        
        # Mean annual return
        metrics['Mean'] = {
            'MVEP': results_df['mvep_return'].mean() * 12 * 100,
            'MTEP': results_df['mtep_return'].mean() * 12 * 100,
            'DIFF': results_df['diff_return'].mean() * 12 * 100,
            'INDEX': results_df['index_return'].mean() * 12 * 100
        }
        
        # Annualized volatility
        metrics['Std'] = {
            'MVEP': results_df['mvep_return'].std() * np.sqrt(12) * 100,
            'MTEP': results_df['mtep_return'].std() * np.sqrt(12) * 100,
            'DIFF': results_df['diff_return'].std() * np.sqrt(12) * 100,
            'INDEX': results_df['index_return'].std() * np.sqrt(12) * 100
        }
        
        # Sharpe ratio (assuming 0% risk-free rate)
        metrics['SR'] = {
            'MVEP': metrics['Mean']['MVEP'] / metrics['Std']['MVEP'] if metrics['Std']['MVEP'] > 0 else 0,
            'MTEP': metrics['Mean']['MTEP'] / metrics['Std']['MTEP'] if metrics['Std']['MTEP'] > 0 else 0,
            'DIFF': metrics['Mean']['DIFF'] / metrics['Std']['DIFF'] if metrics['Std']['DIFF'] > 0 else 0,
            'INDEX': metrics['Mean']['INDEX'] / metrics['Std']['INDEX'] if metrics['Std']['INDEX'] > 0 else 0
        }
        
        # Certainty equivalent return
        kappa = self.kappa_m * 12  # Annualized risk aversion
        metrics['CEQ'] = {
            'MVEP': metrics['Mean']['MVEP'] - (kappa/2) * (metrics['Std']['MVEP']/100)**2,
            'MTEP': metrics['Mean']['MTEP'] - (kappa/2) * (metrics['Std']['MTEP']/100)**2,
            'DIFF': metrics['Mean']['DIFF'] - (kappa/2) * (metrics['Std']['DIFF']/100)**2,
            'INDEX': metrics['Mean']['INDEX'] - (kappa/2) * (metrics['Std']['INDEX']/100)**2
        }
        
        # Calculate turnover
        turnover = {'MVEP': 0, 'MTEP': 0}
        
        for i in range(1, len(results_df)):
            prev_mvep = results_df['mvep_weights'].iloc[i-1]
            prev_mtep = results_df['mtep_weights'].iloc[i-1]
            
            # Adjust weights based on returns
            prev_mvep_returns = prev_mvep * (1 + results_df['mvep_return'].iloc[i-1])
            prev_mtep_returns = prev_mtep * (1 + results_df['mtep_return'].iloc[i-1])
            
            # Normalize
            prev_mvep_adjusted = prev_mvep_returns / np.sum(prev_mvep_returns)
            prev_mtep_adjusted = prev_mtep_returns / np.sum(prev_mtep_returns)
            
            # Current weights
            curr_mvep = results_df['mvep_weights'].iloc[i]
            curr_mtep = results_df['mtep_weights'].iloc[i]
            
            # Calculate turnover
            turnover['MVEP'] += np.sum(np.abs(curr_mvep - prev_mvep_adjusted))
            turnover['MTEP'] += np.sum(np.abs(curr_mtep - prev_mtep_adjusted))
        
        # Normalize by number of periods
        if len(results_df) > 1:
            metrics['TO'] = {
                'MVEP': turnover['MVEP'] / (len(results_df) - 1) * 100,
                'MTEP': turnover['MTEP'] / (len(results_df) - 1) * 100,
                'DIFF': (turnover['MTEP'] - turnover['MVEP']) / (len(results_df) - 1) * 100,
                'INDEX': 0  # Index has 0 turnover
            }
        else:
            metrics['TO'] = {'MVEP': 0, 'MTEP': 0, 'DIFF': 0, 'INDEX': 0}
        
        # Calculate average implicit value
        avg_implicit_value = results_df['implicit_value'].mean()
        
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Add average implicit value
        metrics_df.loc['IV', :] = avg_implicit_value
        
        return metrics_df
    
    def plot_results(self, results_df: pd.DataFrame) -> None:
        """
        Plot backtest results.
        
        Args:
            results_df: DataFrame with backtest results
        """
        if len(results_df) == 0:
            logger.error("No results to plot")
            return
            
        plt.figure(figsize=(12, 10))
        
        # Plot cumulative returns
        plt.subplot(3, 1, 1)
        cum_mvep = (1 + results_df['mvep_return']).cumprod()
        cum_mtep = (1 + results_df['mtep_return']).cumprod()
        cum_index = (1 + results_df['index_return']).cumprod()
        
        plt.plot(results_df['date'], cum_mvep, label='MVEP', color='blue')
        plt.plot(results_df['date'], cum_mtep, label='MTEP', color='green')
        plt.plot(results_df['date'], cum_index, label='Index', color='black', linestyle='--')
        
        plt.title('Cumulative Portfolio Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # Plot return difference (MTEP - MVEP)
        plt.subplot(3, 1, 2)
        plt.bar(results_df['date'], results_df['diff_return'] * 100, color=['green' if x > 0 else 'red' for x in results_df['diff_return']])
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.title('Monthly Return Difference (MTEP - MVEP)')
        plt.xlabel('Date')
        plt.ylabel('Return Difference (%)')
        plt.grid(True)
        
        # Plot implicit value
        plt.subplot(3, 1, 3)
        plt.plot(results_df['date'], results_df['implicit_value'], color='red')
        plt.axhline(y=0, color='black', linestyle='--')
        
        plt.title('Implicit Value of Index-Tracking')
        plt.xlabel('Date')
        plt.ylabel('Implicit Value')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.show()
        
        # Plot weight distribution over time
        plt.figure(figsize=(12, 10))
        
        # Convert weights to DataFrame for easier plotting
        mvep_weights_df = pd.DataFrame([w for w in results_df['mvep_weights']], index=results_df['date'])
        mtep_weights_df = pd.DataFrame([w for w in results_df['mtep_weights']], index=results_df['date'])
        
        # Plot MVEP weights
        plt.subplot(2, 1, 1)
        plt.stackplot(mvep_weights_df.index, mvep_weights_df.T, 
                      labels=[f'Asset {i+1}' for i in range(mvep_weights_df.shape[1])],
                      alpha=0.7)
        
        plt.title('MVEP Portfolio Weights Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.ylim(0, 1)
        plt.grid(True)
        
        # Plot MTEP weights
        plt.subplot(2, 1, 2)
        plt.stackplot(mtep_weights_df.index, mtep_weights_df.T, 
                      labels=[f'Asset {i+1}' for i in range(mtep_weights_df.shape[1])],
                      alpha=0.7)
        
        plt.title('MTEP Portfolio Weights Over Time')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.ylim(0, 1)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('portfolio_weights.png')
        plt.show()


def main():
    """Main function to run the strategies."""
    # Initialize the portfolio strategies
    portfolio = PortfolioStrategies()
    
    # Define tickers and index
    # You can replace these with your own tickers
    tickers = [
        "AAPL US Equity",
        "MSFT US Equity",
        "AMZN US Equity",
        "GOOGL US Equity",
        "META US Equity",
        "TSLA US Equity",
        "BRK/B US Equity",
        "JNJ US Equity",
        "UNH US Equity",
        "XOM US Equity"
    ]
    index_ticker = "SPX Index"
    
    # Define date range
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2025, 4, 1)
    
    # Set mean target (monthly)
    monthly_mean_target = 0.01  # 1% per month, approximately 12% annually
    portfolio.set_mean_target(monthly_mean_target)
    
    # Try to load data from Bloomberg
    data = None
    if BLOOMBERG_AVAILABLE:
        try:
            data = portfolio.load_data_from_bloomberg(tickers, index_ticker, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to load Bloomberg data: {e}")
            data = None
    
    # If Bloomberg failed or is not available, try to load from CSV
    if data is None:
        if os.path.exists("asset_returns.csv") and os.path.exists("index_returns.csv"):
            logger.info("Loading data from CSV files")
            data = portfolio.load_data_from_csv("asset_returns.csv", "index_returns.csv")
        else:
            # Generate synthetic data as last resort
            logger.info("Generating synthetic data for testing")
            data = portfolio.generate_synthetic_data(tickers, index_ticker, start_date, end_date)
    
    # Backtest the strategies
    logger.info("Starting backtest...")
    results = portfolio.backtest(window_size=60, rebalance_freq=1)
    
    if results is not None and not results.empty:
        # Calculate performance metrics
        logger.info("Calculating performance metrics...")
        metrics = portfolio.calculate_performance_metrics(results)
        
        # Print metrics
        print("\nPerformance Metrics:")
        print(metrics)
        
        # Plot results
        logger.info("Plotting results...")
        portfolio.plot_results(results)
        
        # Save results
        results.to_csv("backtest_results.csv")
        metrics.to_csv("performance_metrics.csv")
        
        logger.info("Analysis complete. Results saved to CSV files and plot.")
    else:
        logger.error("Backtest failed or produced no results.")


if __name__ == "__main__":
    main()