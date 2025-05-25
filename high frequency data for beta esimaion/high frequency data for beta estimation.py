import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdblp
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class MarketNeutralStrategy:
    """
    Implementation of a market neutral strategy based on Lee et al. (2019)
    "Targeting Market Neutrality"
    """
    
    def __init__(self, tickers, market_ticker='SPY', start_date=None, end_date=None, intraday_frequency='10min'):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers
        market_ticker : str
            Market index ticker
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        intraday_frequency : str
            Frequency of intraday data ('10min', '30min', '60min', or 'daily')
        """
        self.tickers = tickers
        self.market_ticker = market_ticker
        
        if start_date is None:
            self.start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        else:
            self.start_date = start_date.replace('-', '')
            
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y%m%d')
        else:
            self.end_date = end_date.replace('-', '')
            
        self.intraday_frequency = intraday_frequency
        
        # Determine optimal estimation window based on frequency
        if intraday_frequency == '10min':
            self.estimation_window = 4  # 4 weeks
        elif intraday_frequency == '30min':
            self.estimation_window = 12  # 12 weeks
        elif intraday_frequency == '60min':
            self.estimation_window = 16  # 16 weeks
        else:  # daily
            self.estimation_window = 32  # 32 weeks
            
        # Bloomberg connection
        self.conn = None
        
        # Data storage
        self.intraday_data = {}
        self.beta_forecasts = pd.DataFrame()
        self.portfolio_betas = pd.DataFrame()
        self.portfolio_returns = pd.DataFrame()
        
    def connect_to_bloomberg(self):
        """Connect to Bloomberg"""
        print("Connecting to Bloomberg...")
        try:
            self.conn = pdblp.BCon(timeout=60000)  # 60-second timeout
            self.conn.start()
            print("Connected to Bloomberg")
            return True
        except Exception as e:
            print(f"Failed to connect to Bloomberg: {e}")
            return False
    
    def get_bloomberg_data_with_retries(self, tickers, fields, start_date, end_date, max_retries=3):
        """
        Get data from Bloomberg with retries on timeout
        
        Parameters:
        -----------
        tickers : list
            Bloomberg tickers
        fields : list
            Fields to retrieve
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str
            End date in format 'YYYYMMDD'
        max_retries : int
            Maximum number of retry attempts
            
        Returns:
        --------
        data : DataFrame
            Retrieved data or None if failed
        """
        for attempt in range(max_retries):
            try:
                data = self.conn.bdh(tickers, fields, start_date, end_date)
                return data
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if "Timeout" in str(e) and attempt < max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                else:
                    return None
        return None
    
    def get_intraday_data(self):
        """
        Get intraday data for all tickers from Bloomberg
        """
        if self.conn is None:
            if not self.connect_to_bloomberg():
                return False
                
        print(f"Getting {self.intraday_frequency} intraday data...")
        
        # Determine Bloomberg interval code
        if self.intraday_frequency == '10min':
            interval = 10
        elif self.intraday_frequency == '30min':
            interval = 30
        elif self.intraday_frequency == '60min':
            interval = 60
        else:  # daily
            # For daily data, use bdh instead of bdit
            all_tickers = self.tickers + [self.market_ticker]
            daily_data = self.get_bloomberg_data_with_retries(
                all_tickers,
                ['PX_OPEN', 'PX_LAST'],
                self.start_date,
                self.end_date
            )
            
            if daily_data is None:
                print("Failed to retrieve daily data")
                return False
                
            # Process daily data
            for ticker in all_tickers:
                if ticker in daily_data.columns.levels[0]:
                    # Calculate returns
                    opens = daily_data[ticker]['PX_OPEN']
                    closes = daily_data[ticker]['PX_LAST']
                    
                    # Calculate open-to-open returns except for the last interval
                    returns = opens.pct_change()
                    
                    # Store data
                    self.intraday_data[ticker] = pd.DataFrame({
                        'open': opens,
                        'close': closes,
                        'return': returns
                    })
                    
            print(f"Retrieved daily data for {len(self.intraday_data)} tickers")
            return True
        
        # For intraday data
        # Adjust dates to ensure we get enough historical data for beta calculation
        adjusted_start_date = (datetime.strptime(self.start_date, '%Y%m%d') - 
                              timedelta(weeks=self.estimation_window)).strftime('%Y%m%d')
        
        # Process all tickers including market
        all_tickers = self.tickers + [self.market_ticker]
        
        for ticker in all_tickers:
            try:
                print(f"Getting intraday data for {ticker}...")
                
                # Get intraday data
                intraday_data = self.conn.bdit(
                    ticker,
                    ['TRADE'],
                    adjusted_start_date,
                    self.end_date,
                    interval=interval
                )
                
                if intraday_data is None or intraday_data.empty:
                    print(f"No intraday data for {ticker}")
                    continue
                
                # Resample to ensure regular intervals
                intraday_data = intraday_data.resample(self.intraday_frequency).last()
                
                # Calculate returns
                intraday_data['return'] = intraday_data['TRADE'].pct_change()
                
                # Store data
                self.intraday_data[ticker] = intraday_data
                
            except Exception as e:
                print(f"Error getting intraday data for {ticker}: {e}")
        
        print(f"Retrieved intraday data for {len(self.intraday_data)} tickers")
        return len(self.intraday_data) > 0
    
    def compute_realized_betas(self, start_date, end_date):
        """
        Compute realized betas for all stocks over the specified period
        
        Parameters:
        -----------
        start_date : datetime
            Start date
        end_date : datetime
            End date
            
        Returns:
        --------
        betas : Series
            Series of realized betas for each stock
        """
        betas = {}
        
        # Get market returns
        if self.market_ticker not in self.intraday_data:
            print(f"Market ticker {self.market_ticker} data not available")
            return None
            
        market_data = self.intraday_data[self.market_ticker]
        market_returns = market_data[(market_data.index >= start_date) & 
                                    (market_data.index <= end_date)]['return'].dropna()
        
        if len(market_returns) < 5:  # Require at least 5 observations
            print(f"Insufficient market data between {start_date} and {end_date}")
            return None
        
        # Compute realized betas for each stock
        for ticker in self.tickers:
            if ticker not in self.intraday_data:
                continue
                
            stock_data = self.intraday_data[ticker]
            stock_returns = stock_data[(stock_data.index >= start_date) & 
                                      (stock_data.index <= end_date)]['return'].dropna()
            
            # Align stock and market returns
            common_idx = market_returns.index.intersection(stock_returns.index)
            if len(common_idx) < 5:  # Require at least 5 observations
                print(f"Insufficient aligned data for {ticker}")
                continue
                
            stock_returns = stock_returns[common_idx]
            market_returns_aligned = market_returns[common_idx]
            
            # Compute realized beta using formula from equation (5) in the paper
            covariance = (stock_returns * market_returns_aligned).sum()
            market_variance = (market_returns_aligned ** 2).sum()
            
            if market_variance > 0:
                beta = covariance / market_variance
            else:
                beta = np.nan
                
            betas[ticker] = beta
        
        return pd.Series(betas)
    
    def forecast_betas(self):
        """
        Generate beta forecasts for all stocks on a weekly basis
        """
        print("Generating beta forecasts...")
        
        # Determine week boundaries
        if self.intraday_frequency == 'daily':
            # For daily data, use calendar weeks
            all_dates = sorted(list(set().union(*[data.index for data in self.intraday_data.values()])))
            weeks = pd.to_datetime(all_dates).to_period('W').unique()
        else:
            # For intraday data, use trading days grouped into weeks
            # Get trading days
            trading_days = sorted(list(set([date.date() for data in self.intraday_data.values() 
                                          for date in data.index])))
            
            # Group into 5-day trading weeks
            weeks = []
            for i in range(0, len(trading_days), 5):
                if i + 5 <= len(trading_days):
                    weeks.append((trading_days[i], trading_days[i+4]))
                    
        # Generate forecasts for each week
        forecasts = []
        
        for i, week in enumerate(weeks[:-1]):  # Skip the last week as we need it for evaluation
            if self.intraday_frequency == 'daily':
                week_start = week.start_time
                week_end = week.end_time
                next_week_start = weeks[i+1].start_time
                next_week_end = weeks[i+1].end_time
            else:
                week_start = datetime.combine(week[0], datetime.min.time())
                week_end = datetime.combine(week[1], datetime.max.time())
                next_week_start = datetime.combine(weeks[i+1][0], datetime.min.time())
                next_week_end = datetime.combine(weeks[i+1][1], datetime.max.time())
            
            # Calculate estimation window start date
            est_start = week_start - timedelta(weeks=self.estimation_window)
            
            # Compute realized betas for estimation window
            realized_betas = self.compute_realized_betas(est_start, week_end)
            
            if realized_betas is not None and not realized_betas.empty:
                # Store forecast
                forecast = pd.DataFrame({
                    'beta_forecast': realized_betas,
                    'forecast_date': week_end.strftime('%Y-%m-%d'),
                    'next_week_start': next_week_start.strftime('%Y-%m-%d'),
                    'next_week_end': next_week_end.strftime('%Y-%m-%d')
                })
                
                forecasts.append(forecast)
        
        if not forecasts:
            print("No beta forecasts generated")
            return False
        
        # Combine all forecasts
        self.beta_forecasts = pd.concat(forecasts, ignore_index=True)
        print(f"Generated {len(self.beta_forecasts)} beta forecasts")
        
        return True
    
    def construct_market_neutral_portfolios(self, n_long=14, n_short=14, n_portfolios=100):
        """
        Construct market neutral portfolios
        
        Parameters:
        -----------
        n_long : int
            Number of stocks in long portfolio
        n_short : int
            Number of stocks in short portfolio
        n_portfolios : int
            Number of random portfolios to generate
            
        Returns:
        --------
        portfolio_betas : DataFrame
            Realized portfolio betas
        """
        print(f"Constructing {n_portfolios} market neutral portfolios...")
        
        # Get unique forecast dates
        forecast_dates = self.beta_forecasts['forecast_date'].unique()
        
        # Initialize storage for portfolio betas
        portfolio_betas = {}
        portfolio_returns = {}
        
        for i in range(n_portfolios):
            portfolio_id = f"Portfolio_{i+1}"
            week_betas = []
            week_returns = []
            
            for date in forecast_dates:
                # Get forecasts for this date
                date_forecasts = self.beta_forecasts[self.beta_forecasts['forecast_date'] == date]
                
                # Get tickers with valid forecasts
                valid_tickers = date_forecasts.dropna(subset=['beta_forecast']).index
                
                if len(valid_tickers) < n_long + n_short:
                    continue
                
                # Randomly select long and short stocks
                np.random.seed(i + int(pd.to_datetime(date).timestamp()))  # Ensure reproducibility but different for each portfolio
                long_tickers = np.random.choice(valid_tickers, n_long, replace=False)
                
                # Select short tickers from the remaining
                remaining_tickers = [t for t in valid_tickers if t not in long_tickers]
                short_tickers = np.random.choice(remaining_tickers, n_short, replace=False)
                
                # Calculate forecasted portfolio beta
                long_betas = date_forecasts.loc[long_tickers, 'beta_forecast'].values
                short_betas = date_forecasts.loc[short_tickers, 'beta_forecast'].values
                
                # Average beta for long and short sides
                long_beta = np.mean(long_betas)
                short_beta = np.mean(short_betas)
                
                # Forecasted portfolio beta before hedge
                portfolio_beta_forecast = long_beta - short_beta
                
                # Calculate hedge position in market to achieve beta neutrality
                hedge_position = -portfolio_beta_forecast
                
                # Get next week dates
                next_week_start = date_forecasts['next_week_start'].iloc[0]
                next_week_end = date_forecasts['next_week_end'].iloc[0]
                
                # Calculate realized portfolio beta for the next week
                next_week_start_dt = pd.to_datetime(next_week_start)
                next_week_end_dt = pd.to_datetime(next_week_end)
                
                # Get market returns for next week
                if self.market_ticker not in self.intraday_data:
                    continue
                    
                market_data = self.intraday_data[self.market_ticker]
                market_returns = market_data[(market_data.index >= next_week_start_dt) & 
                                           (market_data.index <= next_week_end_dt)]['return'].dropna()
                
                if len(market_returns) < 5:  # Require at least 5 observations
                    continue
                
                # Calculate realized portfolio returns
                portfolio_returns_series = pd.Series(0, index=market_returns.index)
                
                # Add long positions (1/n_long weight for each)
                for ticker in long_tickers:
                    if ticker not in self.intraday_data:
                        continue
                        
                    stock_data = self.intraday_data[ticker]
                    stock_returns = stock_data[(stock_data.index >= next_week_start_dt) & 
                                            (stock_data.index <= next_week_end_dt)]['return'].dropna()
                    
                    # Align with market returns
                    common_idx = market_returns.index.intersection(stock_returns.index)
                    if len(common_idx) < 5:
                        continue
                        
                    portfolio_returns_series[common_idx] += stock_returns[common_idx] / n_long
                
                # Add short positions (-1/n_short weight for each)
                for ticker in short_tickers:
                    if ticker not in self.intraday_data:
                        continue
                        
                    stock_data = self.intraday_data[ticker]
                    stock_returns = stock_data[(stock_data.index >= next_week_start_dt) & 
                                            (stock_data.index <= next_week_end_dt)]['return'].dropna()
                    
                    # Align with market returns
                    common_idx = market_returns.index.intersection(stock_returns.index)
                    if len(common_idx) < 5:
                        continue
                        
                    portfolio_returns_series[common_idx] -= stock_returns[common_idx] / n_short
                
                # Add market hedge position
                portfolio_returns_series += hedge_position * market_returns
                
                # Calculate realized beta for the portfolio
                # Align portfolio and market returns
                common_idx = market_returns.index.intersection(portfolio_returns_series.index)
                if len(common_idx) < 5:
                    continue
                    
                portfolio_returns_aligned = portfolio_returns_series[common_idx]
                market_returns_aligned = market_returns[common_idx]
                
                # Compute realized beta
                covariance = (portfolio_returns_aligned * market_returns_aligned).sum()
                market_variance = (market_returns_aligned ** 2).sum()
                
                if market_variance > 0:
                    realized_beta = covariance / market_variance
                else:
                    realized_beta = np.nan
                
                # Store results
                week_betas.append({
                    'date': date,
                    'portfolio_beta_forecast': portfolio_beta_forecast,
                    'hedge_position': hedge_position,
                    'realized_beta': realized_beta
                })
                
                # Store weekly return
                week_returns.append({
                    'date': date,
                    'return': portfolio_returns_aligned.sum()  # Weekly return
                })
            
            # Store portfolio results
            if week_betas:
                portfolio_betas[portfolio_id] = pd.DataFrame(week_betas)
                portfolio_returns[portfolio_id] = pd.DataFrame(week_returns)
        
        # Combine results
        self.portfolio_betas = portfolio_betas
        self.portfolio_returns = portfolio_returns
        
        print(f"Constructed {len(portfolio_betas)} market neutral portfolios")
        return portfolio_betas
    
    def analyze_market_neutrality(self):
        """
        Analyze the market neutrality of the portfolios
        
        Returns:
        --------
        statistics : dict
            Dictionary of market neutrality statistics
        """
        if not self.portfolio_betas:
            print("No portfolio data available")
            return None
        
        # Calculate statistics for each portfolio
        statistics = {}
        for portfolio_id, data in self.portfolio_betas.items():
            realized_betas = data['realized_beta'].dropna()
            
            # Skip if too few observations
            if len(realized_betas) < 10:
                continue
                
            # Calculate statistics
            statistics[portfolio_id] = {
                'mean': realized_betas.mean(),
                'std': realized_betas.std(),
                'max': realized_betas.max(),
                'min': realized_betas.min(),
                'range': realized_betas.max() - realized_betas.min(),
                'mae': np.abs(realized_betas).mean(),
                'mse': (realized_betas ** 2).mean(),
                'n_weeks': len(realized_betas)
            }
        
        # Compute aggregate statistics
        if not statistics:
            print("No statistics calculated")
            return None
            
        # Convert to DataFrame for easier analysis
        stats_df = pd.DataFrame(statistics).T
        
        # Calculate percentiles
        percentiles = {
            '10th': stats_df.quantile(0.1),
            '50th': stats_df.quantile(0.5),
            '90th': stats_df.quantile(0.9)
        }
        
        print("\nMarket Neutrality Statistics:")
        print("-----------------------------")
        for percentile, values in percentiles.items():
            print(f"\n{percentile} Percentile:")
            print(f"  Mean Beta: {values['mean']:.4f}")
            print(f"  Standard Deviation: {values['std']:.4f}")
            print(f"  Maximum Beta: {values['max']:.4f}")
            print(f"  Minimum Beta: {values['min']:.4f}")
            print(f"  Range (MAX-MIN): {values['range']:.4f}")
            print(f"  Mean Absolute Error: {values['mae']:.4f}")
            print(f"  Mean Squared Error: {values['mse']:.4f}")
        
        return percentiles
    
    def plot_realized_betas(self, sample_size=5):
        """
        Plot realized betas for a sample of portfolios
        
        Parameters:
        -----------
        sample_size : int
            Number of portfolios to plot
        """
        if not self.portfolio_betas:
            print("No portfolio data available")
            return
            
        # Select a sample of portfolios
        portfolio_ids = list(self.portfolio_betas.keys())
        if len(portfolio_ids) > sample_size:
            np.random.seed(42)
            sample_portfolios = np.random.choice(portfolio_ids, sample_size, replace=False)
        else:
            sample_portfolios = portfolio_ids
        
        plt.figure(figsize=(12, 8))
        
        for portfolio_id in sample_portfolios:
            data = self.portfolio_betas[portfolio_id]
            plt.plot(pd.to_datetime(data['date']), data['realized_beta'], 
                     label=f"{portfolio_id} (MAE: {np.abs(data['realized_beta']).mean():.4f})")
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f'Realized Betas of Market Neutral Portfolios ({self.intraday_frequency} data)')
        plt.xlabel('Date')
        plt.ylabel('Realized Beta')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_portfolio_returns(self, sample_size=5):
        """
        Plot cumulative returns for a sample of portfolios
        
        Parameters:
        -----------
        sample_size : int
            Number of portfolios to plot
        """
        if not self.portfolio_returns:
            print("No portfolio return data available")
            return
            
        # Select a sample of portfolios
        portfolio_ids = list(self.portfolio_returns.keys())
        if len(portfolio_ids) > sample_size:
            np.random.seed(42)
            sample_portfolios = np.random.choice(portfolio_ids, sample_size, replace=False)
        else:
            sample_portfolios = portfolio_ids
        
        plt.figure(figsize=(12, 8))
        
        for portfolio_id in sample_portfolios:
            data = self.portfolio_returns[portfolio_id]
            
            # Calculate cumulative return
            data = data.sort_values('date')
            data['cumulative_return'] = (1 + data['return']).cumprod() - 1
            
            plt.plot(pd.to_datetime(data['date']), data['cumulative_return'], 
                     label=f"{portfolio_id} (Final: {data['cumulative_return'].iloc[-1]:.2%})")
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f'Cumulative Returns of Market Neutral Portfolios ({self.intraday_frequency} data)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_synthetic_data(self, n_weeks=52):
        """Generate synthetic data for testing when Bloomberg is unavailable"""
        print("Generating synthetic data...")
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=n_weeks)
        
        # Generate trading days (Monday to Friday)
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = all_days[all_days.dayofweek < 5]
        
        # Generate intraday timestamps
        if self.intraday_frequency == '10min':
            freq = '10T'
        elif self.intraday_frequency == '30min':
            freq = '30T'
        elif self.intraday_frequency == '60min':
            freq = '60T'
        else:  # daily
            # For daily data, just use trading days
            timestamps = trading_days
            for ticker in self.tickers + [self.market_ticker]:
                # Generate random prices
                np.random.seed(hash(ticker) % 10000)
                
                # Start with a random price between 50 and 200
                initial_price = np.random.uniform(50, 200)
                
                # Generate daily returns with some autocorrelation
                daily_returns = np.random.normal(0.0005, 0.015, len(timestamps))
                
                # Add a market component for stock returns (not for market itself)
                if ticker != self.market_ticker:
                    # Random beta between 0.5 and 1.5
                    beta = np.random.uniform(0.5, 1.5)
                    
                    # Generate market returns
                    market_returns = np.random.normal(0.0005, 0.01, len(timestamps))
                    
                    # Add market component to stock returns
                    daily_returns += beta * market_returns
                
                # Convert returns to prices
                prices = initial_price * np.cumprod(1 + daily_returns)
                
                # Store data
                self.intraday_data[ticker] = pd.DataFrame({
                    'open': prices,
                    'close': prices,
                    'return': pd.Series(daily_returns, index=timestamps)
                }, index=timestamps)
            
            return True
        
        # For intraday data, generate timestamps for each trading day
        intraday_timestamps = []
        for day in trading_days:
            # Trading hours: 9:30 AM to 4:00 PM
            day_timestamps = pd.date_range(
                start=day.replace(hour=9, minute=30),
                end=day.replace(hour=16, minute=0),
                freq=freq
            )
            intraday_timestamps.extend(day_timestamps)
        
        timestamps = pd.DatetimeIndex(intraday_timestamps)
        
        # Generate synthetic price data for each ticker
        for ticker in self.tickers + [self.market_ticker]:
            # Generate random prices
            np.random.seed(hash(ticker) % 10000)
            
            # Start with a random price between 50 and 200
            initial_price = np.random.uniform(50, 200)
            
            # Generate returns with some autocorrelation
            returns = np.random.normal(0.00005, 0.002, len(timestamps))
            
            # Add a market component for stock returns (not for market itself)
            if ticker != self.market_ticker:
                # Random beta between 0.5 and 1.5
                beta = np.random.uniform(0.5, 1.5)
                
                # Generate market returns
                market_returns = np.random.normal(0.00005, 0.0015, len(timestamps))
                
                # Add market component to stock returns
                returns += beta * market_returns
            
            # Convert returns to prices
            prices = initial_price * np.cumprod(1 + returns)
            
            # Store data
            self.intraday_data[ticker] = pd.DataFrame({
                'TRADE': prices,
                'return': pd.Series(returns, index=timestamps)
            }, index=timestamps)
        
        print("Synthetic data generated successfully")
        return True
        
    def run(self, use_synthetic=False):
        """
        Run the market neutral strategy
        
        Parameters:
        -----------
        use_synthetic : bool
            Whether to use synthetic data instead of Bloomberg data
            
        Returns:
        --------
        statistics : dict
            Dictionary of market neutrality statistics
        """
        if use_synthetic:
            success = self.generate_synthetic_data()
        else:
            success = self.get_intraday_data()
            
        if not success:
            print("Failed to prepare data for strategy")
            return None
            
        # Forecast betas
        if not self.forecast_betas():
            print("Failed to forecast betas")
            return None
            
        # Construct market neutral portfolios
        self.construct_market_neutral_portfolios()
        
        # Analyze market neutrality
        statistics = self.analyze_market_neutrality()
        
        # Plot results
        self.plot_realized_betas()
        self.plot_portfolio_returns()
        
        return statistics


# Example usage
if __name__ == "__main__":
    # DJIA tickers (28 instead of 29 because Visa was excluded in the paper)
    djia_tickers = [
        'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD',
        'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
        'NKE', 'PFE', 'PG', 'T', 'TRV', 'UNH', 'UTX', 'VZ', 'WMT', 'XOM'
    ]
    
    # Create strategy
    strategy = MarketNeutralStrategy(
        tickers=djia_tickers,
        market_ticker='SPY',
        start_date='2020-01-01',
        end_date='2021-12-31',
        intraday_frequency='10min'  # Options: '10min', '30min', '60min', 'daily'
    )
    
    # Run strategy
    try:
        # First try with real data
        results = strategy.run(use_synthetic=False)
        
        if results is None:
            print("Strategy run with real data failed")
            print("Falling back to synthetic data...")
            # Fall back to synthetic data
            results = strategy.run(use_synthetic=True)
            
    except Exception as e:
        print(f"Error with real data: {e}")
        print("Falling back to synthetic data...")
        # Fall back to synthetic data
        results = strategy.run(use_synthetic=True)
    
    if results:
        print("\nStrategy run completed successfully")