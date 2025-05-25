import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import blpapi
import logging
from scipy import stats
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BloombergDataFetcher:
    """Class to handle Bloomberg data retrieval"""
    
    def __init__(self, host="localhost", port=8194):
        self.host = host
        self.port = port
        self.session = None
    
    def start_session(self):
        """Start a Bloomberg API session"""
        logger.info("Starting Bloomberg API session...")
        
        # Create session options
        session_options = blpapi.SessionOptions()
        session_options.setServerHost(self.host)
        session_options.setServerPort(self.port)
        
        # Create and start session
        self.session = blpapi.Session(session_options)
        if not self.session.start():
            logger.error("Failed to start session.")
            return False
        
        logger.info("Session started successfully.")
        
        # Open reference data service
        if not self.session.openService("//blp/refdata"):
            logger.error("Failed to open //blp/refdata service.")
            return False
            
        logger.info("Reference data service opened successfully.")
        return True
    
    def stop_session(self):
        """Stop the Bloomberg API session"""
        if self.session:
            self.session.stop()
            logger.info("Session stopped.")
    
    def get_historical_data(self, securities, fields, start_date, end_date, period="DAILY"):
        """
        Get historical data from Bloomberg.
        
        Parameters:
        -----------
        securities : list
            List of securities to retrieve data for
        fields : list
            List of fields to retrieve
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        period : str
            Periodicity of data ("DAILY", "WEEKLY", "MONTHLY", etc.)
            
        Returns:
        --------
        dict
            Dictionary of pandas DataFrames with historical data for each security
        """
        if not self.session:
            logger.error("No active session. Please start a session first.")
            return {}
        
        ref_data_service = self.session.getService("//blp/refdata")
        request = ref_data_service.createRequest("HistoricalDataRequest")
        
        # Add securities
        for security in securities:
            request.append("securities", security)
        
        # Add fields
        for field in fields:
            request.append("fields", field)
        
        # Set dates
        request.set("startDate", start_date.replace('-', ''))
        request.set("endDate", end_date.replace('-', ''))
        
        # Set periodicity
        request.set("periodicitySelection", period)
        
        logger.info(f"Sending request for {len(securities)} securities from {start_date} to {end_date}")
        self.session.sendRequest(request)
        
        # Process response
        data_by_security = {}
        
        end_reached = False
        while not end_reached:
            event = self.session.nextEvent(500)
            
            if event.eventType() in [blpapi.Event.PARTIAL_RESPONSE, blpapi.Event.RESPONSE]:
                for msg in event:
                    security_data = msg.getElement("securityData")
                    ticker = security_data.getElementAsString("security")
                    
                    # Check for any field errors
                    if security_data.hasElement("securityError"):
                        logger.warning(f"Security error for {ticker}: {security_data.getElement('securityError')}")
                        continue
                    
                    # Process field data
                    field_data = security_data.getElement("fieldData")
                    
                    # Extract data into a list of dictionaries
                    data_list = []
                    for i in range(field_data.numValues()):
                        point = field_data.getValue(i)
                        data_point = {'date': point.getElementAsDatetime("date")}
                        
                        # Add all requested fields
                        for field in fields:
                            if point.hasElement(field):
                                field_element = point.getElement(field)
                                if field_element.datatype() == blpapi.DataType.FLOAT64:
                                    data_point[field] = field_element.getValueAsFloat()
                                elif field_element.datatype() == blpapi.DataType.INT64:
                                    data_point[field] = field_element.getValueAsInteger()
                                else:
                                    data_point[field] = field_element.getValueAsString()
                            else:
                                data_point[field] = np.nan
                        
                        data_list.append(data_point)
                    
                    # Convert to DataFrame
                    if data_list:
                        df = pd.DataFrame(data_list)
                        df.set_index('date', inplace=True)
                        df.sort_index(inplace=True)
                        data_by_security[ticker] = df
                    else:
                        logger.warning(f"No data found for {ticker}")
            
            if event.eventType() == blpapi.Event.RESPONSE:
                end_reached = True
        
        return data_by_security


class CROStrategy:
    """
    Implementation of Chronological Return Order strategy based on the research paper.
    
    The strategy sorts stocks based on the correlation between historical returns and 
    time passed since those returns, with the expectation that stocks with higher
    returns in the distant past and lower returns in the recent past (high CRO)
    will outperform those with the opposite pattern (low CRO).
    """
    
    def __init__(self, universe=None, start_date=None, end_date=None):
        """
        Initialize the CRO strategy.
        
        Parameters:
        -----------
        universe : list
            List of securities to include in the strategy
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        """
        self.universe = universe or []
        self.start_date = start_date or (datetime.datetime.now() - datetime.timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.datetime.now().strftime('%Y-%m-%d')
        
        self.data = {}
        self.monthly_data = {}
        self.crom_values = {}
        self.croa_values = {}
        self.portfolios = {}
        self.performance = {}
        
        # Initialize positions dictionary
        self.positions = {
            'CROM': {},
            'CROA': {}
        }
        
        self.bloomberg = BloombergDataFetcher()
    
    def fetch_data(self):
        """Fetch required historical data from Bloomberg"""
        logger.info("Fetching historical data from Bloomberg...")
        
        if not self.bloomberg.start_session():
            logger.error("Failed to start Bloomberg session.")
            return False
        
        try:
            # Adjust start date to ensure we have enough data for lookback periods
            adjusted_start_date = (pd.to_datetime(self.start_date) - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Fetch daily price data
            self.data = self.bloomberg.get_historical_data(
                securities=self.universe,
                fields=["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "VOLUME"],
                start_date=adjusted_start_date,
                end_date=self.end_date,
                period="DAILY"
            )
            
            # Calculate returns for daily data
            for ticker, df in self.data.items():
                self.data[ticker]['return'] = df['PX_LAST'].pct_change()
            
            # Fetch monthly price data
            self.monthly_data = self.bloomberg.get_historical_data(
                securities=self.universe,
                fields=["PX_LAST"],
                start_date=adjusted_start_date,
                end_date=self.end_date,
                period="MONTHLY"
            )
            
            # Calculate returns for monthly data
            for ticker, df in self.monthly_data.items():
                self.monthly_data[ticker]['return'] = df['PX_LAST'].pct_change()
            
            logger.info(f"Successfully fetched data for {len(self.data)} securities.")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return False
            
        finally:
            self.bloomberg.stop_session()
    
    def calculate_crom(self, date_str, window=21):
        """
        Calculate CRO-Monthly for each security.
        
        Parameters:
        -----------
        date_str : str
            Calculation date in format 'YYYY-MM-DD'
        window : int
            Number of days to look back (default: 21 trading days, ~1 month)
            
        Returns:
        --------
        dict
            Dictionary of CROM values for each security
        """
        date = pd.to_datetime(date_str)
        crom_values = {}
        
        for ticker, df in self.data.items():
            # Ensure date is in the DataFrame index
            if date not in df.index:
                # Find closest previous date
                mask = df.index <= date
                if mask.any():
                    date = df.index[mask][-1]
                else:
                    continue
            
            # Get data for the window
            end_loc = df.index.get_loc(date)
            start_loc = max(0, end_loc - window + 1)
            
            if end_loc <= start_loc:
                continue  # Not enough data
                
            window_data = df.iloc[start_loc:end_loc+1]
            if len(window_data) < 10:  # Require at least 10 data points
                continue
                
            # Calculate time sequence (days from the end date)
            time_seq = np.arange(len(window_data))[::-1]  # Reverse so recent days have lower values
            
            # Calculate correlation between returns and time
            returns = window_data['return'].values
            
            # Check if we have valid data
            if not np.isfinite(returns).all() or len(returns) < 2:
                continue
                
            # Calculate correlation
            corr, _ = stats.pearsonr(returns, time_seq)
            crom_values[ticker] = corr
        
        return crom_values
    
    def calculate_croa(self, date_str, window=12):
        """
        Calculate CRO-Annual for each security.
        
        Parameters:
        -----------
        date_str : str
            Calculation date in format 'YYYY-MM-DD'
        window : int
            Number of months to look back (default: 12 months, 1 year)
            
        Returns:
        --------
        dict
            Dictionary of CROA values for each security
        """
        date = pd.to_datetime(date_str)
        croa_values = {}
        
        for ticker, df in self.monthly_data.items():
            # Ensure we're working with the closest month end
            mask = df.index <= date
            if not mask.any():
                continue
                
            calc_date = df.index[mask][-1]
            
            # Get data for the window
            end_loc = df.index.get_loc(calc_date)
            start_loc = max(0, end_loc - window + 1)
            
            if end_loc <= start_loc:
                continue  # Not enough data
                
            window_data = df.iloc[start_loc:end_loc+1]
            if len(window_data) < 6:  # Require at least 6 months of data
                continue
                
            # Calculate time sequence (months from the end date)
            time_seq = np.arange(len(window_data))[::-1]  # Reverse so recent months have lower values
            
            # Calculate correlation between returns and time
            returns = window_data['return'].values
            
            # Check if we have valid data
            if not np.isfinite(returns).all() or len(returns) < 2:
                continue
                
            # Calculate correlation
            corr, _ = stats.pearsonr(returns, time_seq)
            croa_values[ticker] = corr
        
        return croa_values
    
    def run_backtest(self, rebalance_freq='monthly', num_portfolios=10, lookback_days=21, lookback_months=12):
        """
        Run a backtest of the CRO strategy.
        
        Parameters:
        -----------
        rebalance_freq : str
            Rebalance frequency ('monthly' or 'quarterly')
        num_portfolios : int
            Number of portfolios to form (default: 10 deciles)
        lookback_days : int
            Number of days to look back for CROM (default: 21 trading days)
        lookback_months : int
            Number of months to look back for CROA (default: 12 months)
            
        Returns:
        --------
        dict
            Dictionary of backtest results
        """
        logger.info(f"Running backtest with {rebalance_freq} rebalancing...")
        
        # Determine rebalance dates
        if rebalance_freq == 'monthly':
            rebalance_dates = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq='MS'  # Month Start
            )
        elif rebalance_freq == 'quarterly':
            rebalance_dates = pd.date_range(
                start=self.start_date,
                end=self.end_date,
                freq='QS'  # Quarter Start
            )
        else:
            raise ValueError(f"Invalid rebalance frequency: {rebalance_freq}")
        
        # Initialize portfolios
        self.portfolios = {
            'CROM': {f'P{i+1}': pd.DataFrame() for i in range(num_portfolios)},
            'CROA': {f'P{i+1}': pd.DataFrame() for i in range(num_portfolios)}
        }
        
        # Run backtest
        for i, date in enumerate(tqdm(rebalance_dates, desc="Backtesting")):
            date_str = date.strftime('%Y-%m-%d')
            
            # Calculate CROM values
            self.crom_values[date_str] = self.calculate_crom(date_str, window=lookback_days)
            
            # Calculate CROA values
            self.croa_values[date_str] = self.calculate_croa(date_str, window=lookback_months)
            
            # Form portfolios based on CROM
            self._form_portfolios(date_str, 'CROM', num_portfolios)
            
            # Form portfolios based on CROA
            self._form_portfolios(date_str, 'CROA', num_portfolios)
        
        # Calculate portfolio returns
        self._calculate_portfolio_returns('CROM', num_portfolios)
        self._calculate_portfolio_returns('CROA', num_portfolios)
        
        # Calculate strategy performance
        self._calculate_performance(num_portfolios)
        
        return self.performance
    
    def _form_portfolios(self, date_str, cro_type, num_portfolios):
        """
        Form portfolios based on CRO values.
        
        Parameters:
        -----------
        date_str : str
            Rebalance date
        cro_type : str
            Type of CRO to use ('CROM' or 'CROA')
        num_portfolios : int
            Number of portfolios to form
        """
        date = pd.to_datetime(date_str)
        
        # Get CRO values
        if cro_type == 'CROM':
            cro_values = self.crom_values.get(date_str, {})
        else:
            cro_values = self.croa_values.get(date_str, {})
        
        if not cro_values:
            return
        
        # Convert to DataFrame for sorting
        cro_df = pd.DataFrame(list(cro_values.items()), columns=['ticker', 'cro'])
        
        # Sort by CRO value
        cro_df = cro_df.sort_values('cro')
        
        # Calculate portfolio breakpoints
        if len(cro_df) < num_portfolios:
            logger.warning(f"Not enough stocks ({len(cro_df)}) to form {num_portfolios} portfolios on {date_str}.")
            return
            
        portfolio_size = len(cro_df) // num_portfolios
        portfolio_assignments = {}
        
        for i in range(num_portfolios):
            start_idx = i * portfolio_size
            end_idx = (i + 1) * portfolio_size if i < num_portfolios - 1 else len(cro_df)
            
            portfolio_stocks = cro_df.iloc[start_idx:end_idx]
            portfolio_name = f'P{i+1}'
            
            for _, row in portfolio_stocks.iterrows():
                portfolio_assignments[row['ticker']] = portfolio_name
        
        # Store positions for this date
        self.positions[cro_type][date_str] = portfolio_assignments
    
    def _calculate_portfolio_returns(self, cro_type, num_portfolios):
        """
        Calculate returns for each portfolio.
        
        Parameters:
        -----------
        cro_type : str
            Type of CRO to use ('CROM' or 'CROA')
        num_portfolios : int
            Number of portfolios
        """
        # Get positions for this CRO type
        positions = self.positions[cro_type]
        
        if not positions:
            return
        
        # Create a DataFrame for each portfolio's returns
        portfolio_returns = {f'P{i+1}': [] for i in range(num_portfolios)}
        
        # Sort dates
        dates = sorted(positions.keys())
        
        # Calculate returns for each rebalance period
        for i in range(len(dates) - 1):
            start_date = pd.to_datetime(dates[i])
            end_date = pd.to_datetime(dates[i+1])
            
            # Get stocks in each portfolio
            portfolio_stocks = {}
            for ticker, portfolio in positions[dates[i]].items():
                if portfolio not in portfolio_stocks:
                    portfolio_stocks[portfolio] = []
                portfolio_stocks[portfolio].append(ticker)
            
            # Calculate returns for each portfolio
            for portfolio, tickers in portfolio_stocks.items():
                portfolio_return = 0
                weight_sum = 0
                
                for ticker in tickers:
                    if ticker not in self.data:
                        continue
                    
                    # Get stock data
                    stock_data = self.data[ticker]
                    
                    # Calculate return for this stock during this period
                    mask = (stock_data.index >= start_date) & (stock_data.index < end_date)
                    period_data = stock_data[mask]
                    
                    if len(period_data) < 2:
                        continue
                    
                    stock_return = (period_data['PX_LAST'].iloc[-1] / period_data['PX_LAST'].iloc[0]) - 1
                    
                    # Use equal weighting for now
                    weight = 1 / len(tickers)
                    portfolio_return += stock_return * weight
                    weight_sum += weight
                
                # Normalize returns by weight sum
                if weight_sum > 0:
                    portfolio_return /= weight_sum
                else:
                    portfolio_return = np.nan
                
                # Store returns
                portfolio_returns[portfolio].append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'return': portfolio_return
                })
        
        # Convert to DataFrames
        for portfolio, returns in portfolio_returns.items():
            if returns:
                df = pd.DataFrame(returns)
                df.set_index('end_date', inplace=True)
                self.portfolios[cro_type][portfolio] = df
    
    def _calculate_performance(self, num_portfolios):
        """
        Calculate performance metrics for portfolios.
        
        Parameters:
        -----------
        num_portfolios : int
            Number of portfolios
        """
        # Initialize performance dictionary
        self.performance = {
            'CROM': {
                'portfolios': {},
                'long_short': {}
            },
            'CROA': {
                'portfolios': {},
                'long_short': {}
            }
        }
        
        # Calculate performance for each CRO type
        for cro_type in ['CROM', 'CROA']:
            # Calculate cumulative returns for each portfolio
            cumulative_returns = {}
            
            for portfolio, df in self.portfolios[cro_type].items():
                if not df.empty:
                    # Calculate portfolio performance metrics
                    returns = df['return'].dropna()
                    
                    if len(returns) > 0:
                        cumulative_return = (1 + returns).prod() - 1
                        annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
                        volatility = returns.std() * np.sqrt(252)
                        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                        max_drawdown = self._calculate_max_drawdown(returns)
                        
                        self.performance[cro_type]['portfolios'][portfolio] = {
                            'cumulative_return': cumulative_return,
                            'annualized_return': annualized_return,
                            'volatility': volatility,
                            'sharpe_ratio': sharpe_ratio,
                            'max_drawdown': max_drawdown
                        }
                        
                        cumulative_returns[portfolio] = (1 + returns).cumprod()
            
            # Calculate long-short portfolio (high CRO - low CRO)
            if f'P{num_portfolios}' in self.portfolios[cro_type] and 'P1' in self.portfolios[cro_type]:
                high_cro = self.portfolios[cro_type][f'P{num_portfolios}']
                low_cro = self.portfolios[cro_type]['P1']
                
                if not high_cro.empty and not low_cro.empty:
                    # Align dates
                    common_dates = sorted(set(high_cro.index) & set(low_cro.index))
                    
                    if common_dates:
                        high_returns = high_cro.loc[common_dates, 'return']
                        low_returns = low_cro.loc[common_dates, 'return']
                        
                        # Calculate long-short returns
                        long_short_returns = high_returns - low_returns
                        
                        cumulative_return = (1 + long_short_returns).prod() - 1
                        annualized_return = (1 + cumulative_return) ** (252 / len(long_short_returns)) - 1
                        volatility = long_short_returns.std() * np.sqrt(252)
                        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                        max_drawdown = self._calculate_max_drawdown(long_short_returns)
                        
                        self.performance[cro_type]['long_short'] = {
                            'cumulative_return': cumulative_return,
                            'annualized_return': annualized_return,
                            'volatility': volatility,
                            'sharpe_ratio': sharpe_ratio,
                            'max_drawdown': max_drawdown,
                            'returns': long_short_returns
                        }
                        
                        # Store cumulative returns for plotting
                        self.performance[cro_type]['cumulative_returns'] = cumulative_returns
    
    def _calculate_max_drawdown(self, returns):
        """
        Calculate maximum drawdown from a series of returns.
        
        Parameters:
        -----------
        returns : pandas.Series
            Series of returns
            
        Returns:
        --------
        float
            Maximum drawdown
        """
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative / peak) - 1
        return drawdown.min()
    
    def plot_results(self):
        """Plot the results of the backtest"""
        # Create a figure with 2 rows (one for CROM, one for CROA)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot results for each CRO type
        for i, cro_type in enumerate(['CROM', 'CROA']):
            # Skip if no performance data
            if cro_type not in self.performance or not self.performance[cro_type]['portfolios']:
                continue
                
            # Plot cumulative returns for each portfolio
            if 'cumulative_returns' in self.performance[cro_type]:
                for portfolio, returns in self.performance[cro_type]['cumulative_returns'].items():
                    # Only plot P1 (lowest CRO), P5 (middle), and P10 (highest CRO) for clarity
                    if portfolio in ['P1', f'P{len(self.performance[cro_type]["portfolios"])//2}', f'P{len(self.performance[cro_type]["portfolios"])}']:
                        label = 'Low CRO' if portfolio == 'P1' else 'High CRO' if portfolio == f'P{len(self.performance[cro_type]["portfolios"])}' else 'Mid CRO'
                        axes[i, 0].plot(returns.index, returns, label=f'{label} ({portfolio})')
                
                axes[i, 0].set_title(f'{cro_type} Portfolio Cumulative Returns')
                axes[i, 0].set_ylabel('Cumulative Return')
                axes[i, 0].legend()
                axes[i, 0].grid(True)
            
            # Plot long-short returns
            if 'long_short' in self.performance[cro_type] and 'returns' in self.performance[cro_type]['long_short']:
                long_short_returns = self.performance[cro_type]['long_short']['returns']
                cumulative_long_short = (1 + long_short_returns).cumprod()
                
                axes[i, 1].plot(cumulative_long_short.index, cumulative_long_short, label='High CRO - Low CRO', color='green')
                axes[i, 1].set_title(f'{cro_type} Long-Short Portfolio Cumulative Returns')
                axes[i, 1].set_ylabel('Cumulative Return')
                axes[i, 1].legend()
                axes[i, 1].grid(True)
                
                # Add key metrics as text annotation
                metrics = self.performance[cro_type]['long_short']
                textstr = '\n'.join([
                    f'Annualized Return: {metrics["annualized_return"]:.2%}',
                    f'Sharpe Ratio: {metrics["sharpe_ratio"]:.2f}',
                    f'Max Drawdown: {metrics["max_drawdown"]:.2%}'
                ])
                
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                axes[i, 1].text(0.05, 0.95, textstr, transform=axes[i, 1].transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()
        
        # Create a bar chart of returns by portfolio decile
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for i, cro_type in enumerate(['CROM', 'CROA']):
            if cro_type not in self.performance or not self.performance[cro_type]['portfolios']:
                continue
                
            # Extract annualized returns for each portfolio
            portfolios = sorted(self.performance[cro_type]['portfolios'].keys(), 
                              key=lambda x: int(x[1:]))
            returns = [self.performance[cro_type]['portfolios'][p]['annualized_return'] for p in portfolios]
            
            # Create bar chart
            axes[i].bar(portfolios, [r * 100 for r in returns])
            axes[i].set_title(f'{cro_type} Portfolio Annualized Returns (%)')
            axes[i].set_xlabel('Portfolio (Low to High CRO)')
            axes[i].set_ylabel('Annualized Return (%)')
            axes[i].grid(True, axis='y')
            
            # Add trend line
            axes[i].plot(portfolios, [r * 100 for r in returns], 'r-', alpha=0.7)
            
            # Add text showing the high-low spread
            if len(returns) > 1:
                spread = returns[-1] - returns[0]
                axes[i].text(0.5, 0.95, f'High-Low Spread: {spread:.2%}',
                          transform=axes[i].transAxes, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Print a summary of the backtest results"""
        for cro_type in ['CROM', 'CROA']:
            if cro_type not in self.performance:
                continue
                
            print(f"\n{'-'*50}")
            print(f"{cro_type} Strategy Results")
            print(f"{'-'*50}")
            
            # Print portfolio performance
            print("\nPortfolio Performance:")
            portfolio_data = []
            
            for portfolio in sorted(self.performance[cro_type]['portfolios'].keys(), 
                                 key=lambda x: int(x[1:])):
                metrics = self.performance[cro_type]['portfolios'][portfolio]
                portfolio_data.append({
                    'Portfolio': portfolio,
                    'Ann. Return': f"{metrics['annualized_return']:.2%}",
                    'Volatility': f"{metrics['volatility']:.2%}",
                    'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                    'Max DD': f"{metrics['max_drawdown']:.2%}"
                })
            
            # Convert to DataFrame for prettier printing
            portfolio_df = pd.DataFrame(portfolio_data)
            print(portfolio_df.to_string(index=False))
            
            # Print long-short performance
            if 'long_short' in self.performance[cro_type]:
                print("\nLong-Short (High CRO - Low CRO) Performance:")
                metrics = self.performance[cro_type]['long_short']
                print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
                print(f"Annualized Return: {metrics['annualized_return']:.2%}")
                print(f"Volatility: {metrics['volatility']:.2%}")
                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
                
                # Print monthly return statistics
                if 'returns' in metrics:
                    monthly_returns = metrics['returns']
                    print("\nMonthly Return Statistics:")
                    print(f"Mean: {monthly_returns.mean():.2%}")
                    print(f"Median: {monthly_returns.median():.2%}")
                    print(f"Std Dev: {monthly_returns.std():.2%}")
                    print(f"Min: {monthly_returns.min():.2%}")
                    print(f"Max: {monthly_returns.max():.2%}")
                    print(f"% Positive: {(monthly_returns > 0).mean():.2%}")


def main():
    """Main function to run the CRO strategy"""
    # Define universe (for example, S&P 500 stocks)
    # For testing purposes, we'll use a smaller subset of major indices
    universe = [
        "SPX Index",      # S&P 500
        "INDU Index",     # Dow Jones Industrial Average
        "NDX Index",      # NASDAQ 100
        "RTY Index",      # Russell 2000
        "SX5E Index",     # EURO STOXX 50
        "UKX Index",      # FTSE 100
        "DAX Index",      # German DAX
        "CAC Index",      # French CAC 40
        "SMI Index",      # Swiss Market Index
        "NKY Index",      # Nikkei 225
        "HSI Index",      # Hang Seng
        "AS51 Index",     # Australian ASX 200
        "IBOV Index",     # Brazilian Bovespa
        "KOSPI Index",    # Korean KOSPI
        "SENSEX Index",   # Indian SENSEX
        "SHCOMP Index",   # Shanghai Composite
        "000300 CH Index" # CSI 300
    ]
    
    # Alternative test with individual stocks:
    """
    universe = [
        "AAPL US Equity", "MSFT US Equity", "AMZN US Equity", "GOOGL US Equity",
        "META US Equity", "NVDA US Equity", "TSLA US Equity", "BRK/B US Equity",
        "UNH US Equity", "JNJ US Equity", "JPM US Equity", "V US Equity",
        "PG US Equity", "XOM US Equity", "HD US Equity", "CVX US Equity",
        "MA US Equity", "BAC US Equity", "ABBV US Equity", "PFE US Equity"
    ]
    """
    
    # Create strategy instance
    strategy = CROStrategy(
        universe=universe,
        start_date='2015-01-01',
        end_date='2023-12-31'
    )
    
    # Fetch data
    if not strategy.fetch_data():
        logger.error("Failed to fetch data. Exiting.")
        return
    
    # Run backtest
    strategy.run_backtest(
        rebalance_freq='monthly',
        num_portfolios=5,  # Use quintiles for indices
        lookback_days=21,
        lookback_months=12
    )
    
    # Print summary
    strategy.print_summary()
    
    # Plot results
    strategy.plot_results()


if __name__ == "__main__":
    main()
    