import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Try to import Bloomberg API
try:
    import pdblp
    BLOOMBERG_AVAILABLE = True
    print("Bloomberg API available!")
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("Bloomberg API not available. Will use simulated data for testing.")

# Set plotting style that works with older matplotlib versions
# Check available styles and use a compatible one
available_styles = plt.style.available
if 'seaborn-whitegrid' in available_styles:
    plt.style.use('seaborn-whitegrid')
elif 'ggplot' in available_styles:
    plt.style.use('ggplot')

# Set a color palette that works with older seaborn
try:
    sns.set_palette("Set1")
except:
    pass  # Ignore if it fails

class InsiderHorizonStrategy:
    """
    Implements a trading strategy based on insider investment horizon as outlined
    in Akbas, Jiang, and Koch (2020).
    """
    def __init__(self, start_date='2010-01-01', end_date=None, 
                 lookback_years=10, min_years_traded=4, 
                 use_bloomberg=False):
        """
        Initialize the insider horizon strategy.
        
        Parameters:
        -----------
        start_date : str
            Start date for the analysis period
        end_date : str
            End date for the analysis period. Default is today.
        lookback_years : int
            Number of years to look back for calculating insider horizon
        min_years_traded : int
            Minimum number of years an insider must have traded to be included
        use_bloomberg : bool
            Whether to use Bloomberg data or simulated data
        """
        self.start_date = pd.to_datetime(start_date)
        if end_date is None:
            self.end_date = pd.to_datetime(dt.datetime.today())
        else:
            self.end_date = pd.to_datetime(end_date)
            
        self.lookback_years = lookback_years
        self.min_years_traded = min_years_traded
        self.use_bloomberg = use_bloomberg and BLOOMBERG_AVAILABLE
        
        # Initialize data containers
        self.insider_data = None
        self.stock_data = None
        self.portfolio_performance = None
        
        # Initialize Bloomberg connection if available
        if self.use_bloomberg:
            self.bbg = pdblp.BCon(debug=False, port=8194)
            self.bbg.start()
            print("Connected to Bloomberg.")
        
    def fetch_insider_data(self, tickers=None):
        """
        Fetch insider trading data either from Bloomberg or from simulated data.
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers to analyze
            
        Returns:
        --------
        pd.DataFrame
            Insider trading data
        """
        if self.use_bloomberg and tickers is not None:
            print(f"Fetching insider trading data from Bloomberg for {len(tickers)} tickers...")
            
            # Convert dates to Bloomberg format
            start_str = (self.start_date - pd.DateOffset(years=self.lookback_years)).strftime('%Y%m%d')
            end_str = self.end_date.strftime('%Y%m%d')
            
            insider_data = []
            
            # Progress tracking
            total = len(tickers)
            for i, ticker in enumerate(tickers):
                if i % 10 == 0:
                    print(f"Processing {i+1}/{total} tickers...")
                
                try:
                    # Fetch insider filings data for the ticker
                    insider_df = self.bbg.ref(ticker, "INSIDER_TRANSACTION_DATA")
                    
                    # Check if data is available
                    if insider_df is not None and not insider_df.empty:
                        insider_df['ticker'] = ticker
                        insider_data.append(insider_df)
                except Exception as e:
                    print(f"Error fetching insider data for {ticker}: {e}")
            
            # Combine all insider data
            if len(insider_data) > 0:
                self.insider_data = pd.concat(insider_data, ignore_index=True)
                print(f"Fetched insider data for {len(insider_data)} tickers.")
            else:
                print("No insider data available. Using simulated data instead.")
                self.use_bloomberg = False
        
        # If Bloomberg is not available or no data was fetched, use simulated data
        if not self.use_bloomberg or self.insider_data is None or self.insider_data.empty:
            print("Generating simulated insider trading data...")
            self.insider_data = self._generate_simulated_insider_data(tickers)
        
        # Process the insider data
        self._process_insider_data()
        
        return self.insider_data
    
    def _generate_simulated_insider_data(self, tickers=None):
        """
        Generate simulated insider trading data for testing purposes.
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers to simulate data for
            
        Returns:
        --------
        pd.DataFrame
            Simulated insider trading data
        """
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB', 'TSLA', 
                       'JPM', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'NFLX']
        
        # Create date range starting from lookback_years before start_date
        full_date_range = pd.date_range(
            start=(self.start_date - pd.DateOffset(years=self.lookback_years)),
            end=self.end_date,
            freq='D'
        )
        
        # Generate simulated data
        np.random.seed(42)  # For reproducibility
        
        # Create a list to hold insider trading data
        data = []
        
        # For each ticker
        for ticker in tickers:
            # Number of insiders per ticker (random between 2-10)
            num_insiders = np.random.randint(2, 11)
            
            for insider_id in range(1, num_insiders + 1):
                # Generate insider name
                insider_name = f"Insider_{ticker}_{insider_id}"
                
                # Determine if this is a long-horizon or short-horizon insider
                # 50% probability of being a long-horizon insider (only buys or only sells)
                is_long_horizon = np.random.random() < 0.5
                
                # For LH insiders, determine if they only buy or only sell
                if is_long_horizon:
                    only_buys = np.random.random() < 0.5
                
                # Number of trades (random between 5-30)
                num_trades = np.random.randint(5, 31)
                
                # Generate trading dates (random subset of full_date_range)
                trade_dates = np.random.choice(full_date_range, size=num_trades, replace=False)
                trade_dates = sorted(trade_dates)
                
                for trade_date in trade_dates:
                    # If LH insider, their trade direction is determined by their type
                    if is_long_horizon:
                        is_purchase = only_buys
                    # If SH insider, they switch between buying and selling
                    else:
                        is_purchase = np.random.random() < 0.5
                    
                    # Generate number of shares (random between 100-10000)
                    shares = np.random.randint(100, 10001)
                    
                    # Generate price (random between $10-$1000)
                    price = np.random.uniform(10, 1000)
                    
                    # Add to our data list
                    data.append({
                        'ticker': ticker,
                        'trade_date': trade_date,
                        'insider_name': insider_name,
                        'insider_id': f"{ticker}_{insider_id}",
                        'is_purchase': is_purchase,
                        'is_sale': not is_purchase,
                        'shares': shares,
                        'price': price,
                        'transaction_type': 'P' if is_purchase else 'S',
                        'position': np.random.choice(['CEO', 'CFO', 'COO', 'Director', 'Other'], p=[0.1, 0.1, 0.1, 0.3, 0.4])
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Add calculated fields
        df['value'] = df['shares'] * df['price']
        
        return df
    
    def _process_insider_data(self):
        """
        Process the raw insider data to prepare it for analysis.
        """
        # Ensure trade_date is a datetime
        if 'trade_date' in self.insider_data.columns:
            self.insider_data['trade_date'] = pd.to_datetime(self.insider_data['trade_date'])
        else:
            # Try to find the correct date column
            date_columns = [col for col in self.insider_data.columns if 'date' in col.lower()]
            if date_columns:
                self.insider_data['trade_date'] = pd.to_datetime(self.insider_data[date_columns[0]])
            else:
                raise ValueError("Cannot find trade date column in insider data")
        
        # Standardize transaction_type if needed
        if 'transaction_type' not in self.insider_data.columns and 'is_purchase' in self.insider_data.columns:
            self.insider_data['transaction_type'] = self.insider_data['is_purchase'].apply(
                lambda x: 'P' if x else 'S')
        
        # Ensure we have all required columns
        required_cols = ['ticker', 'trade_date', 'insider_id', 'transaction_type', 'shares', 'price']
        for col in required_cols:
            if col not in self.insider_data.columns:
                raise ValueError(f"Required column '{col}' not found in insider data")
        
        # Convert shares to positive numbers (sales will be negative later)
        self.insider_data['shares'] = self.insider_data['shares'].abs()
        
        # Calculate transaction value
        if 'value' not in self.insider_data.columns:
            self.insider_data['value'] = self.insider_data['shares'] * self.insider_data['price']
        
        # Make shares negative for sales
        self.insider_data.loc[self.insider_data['transaction_type'] == 'S', 'shares'] *= -1
        
        # Sort by date
        self.insider_data = self.insider_data.sort_values(['ticker', 'insider_id', 'trade_date'])
        
        print(f"Processed insider data with {len(self.insider_data)} transactions.")
    
    def calculate_insider_horizons(self, as_of_date=None):
        """
        Calculate insider investment horizons for each insider as of a specific date.
        
        Parameters:
        -----------
        as_of_date : datetime or str
            Date to calculate horizons as of. If None, uses the most recent date.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with insider horizons
        """
        if self.insider_data is None:
            raise ValueError("No insider data available. Call fetch_insider_data() first.")
        
        if as_of_date is None:
            as_of_date = self.insider_data['trade_date'].max()
        else:
            as_of_date = pd.to_datetime(as_of_date)
        
        print(f"Calculating insider horizons as of {as_of_date}...")
        
        # Filter data to include only transactions before as_of_date
        # and within the lookback period
        lookback_start = as_of_date - pd.DateOffset(years=self.lookback_years)
        filtered_data = self.insider_data[
            (self.insider_data['trade_date'] >= lookback_start) & 
            (self.insider_data['trade_date'] <= as_of_date)
        ].copy()
        
        # Add year column
        filtered_data['year'] = filtered_data['trade_date'].dt.year
        
        # Calculate annual net order flow for each insider
        annual_flows = []
        
        for (ticker, insider_id, year), group in filtered_data.groupby(['ticker', 'insider_id', 'year']):
            purchases = group[group['transaction_type'] == 'P']['shares'].sum()
            sales = abs(group[group['transaction_type'] == 'S']['shares'].sum())
            
            # Calculate insider order flow
            iof = (purchases - sales) / (purchases + sales) if (purchases + sales) > 0 else 0
            
            annual_flows.append({
                'ticker': ticker,
                'insider_id': insider_id,
                'year': year,
                'purchases': purchases,
                'sales': sales,
                'iof': iof
            })
        
        # Convert to DataFrame
        annual_flows_df = pd.DataFrame(annual_flows)
        
        # Calculate the average annual net order flow for each insider
        insider_horizons = []
        
        for (ticker, insider_id), group in annual_flows_df.groupby(['ticker', 'insider_id']):
            # Only include insiders who traded in at least min_years_traded years
            if len(group) >= self.min_years_traded:
                avg_iof = group['iof'].mean()
                
                # Following the paper's formula, horizon = -1 * |avg_iof|
                # This makes HOR range from -1 (long horizon) to 0 (short horizon)
                hor = -1 * abs(avg_iof)
                
                # Determine if long-horizon (only buys or only sells)
                is_long_horizon = abs(avg_iof) > 0.999
                
                insider_horizons.append({
                    'ticker': ticker,
                    'insider_id': insider_id,
                    'horizon': hor,
                    'avg_iof': avg_iof,
                    'num_years_traded': len(group),
                    'is_long_horizon': is_long_horizon,
                    'is_short_horizon': not is_long_horizon,
                    'horizon_type': 'Long' if is_long_horizon else 'Short',
                    'as_of_date': as_of_date
                })
        
        # Convert to DataFrame
        horizons_df = pd.DataFrame(insider_horizons)
        
        # Print some summary statistics
        if not horizons_df.empty:
            lh_pct = horizons_df['is_long_horizon'].mean() * 100
            print(f"Calculated horizons for {len(horizons_df)} insiders.")
            print(f"Long-horizon insiders: {lh_pct:.1f}%")
            print(f"Short-horizon insiders: {(100-lh_pct):.1f}%")
        else:
            print("No insider horizons calculated. Check data or parameters.")
        
        return horizons_df
    
    def fetch_stock_data(self, tickers):
        """
        Fetch stock price data either from Bloomberg or from simulated data.
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers to fetch data for
            
        Returns:
        --------
        pd.DataFrame
            Stock price data
        """
        if self.use_bloomberg:
            print(f"Fetching stock price data from Bloomberg for {len(tickers)} tickers...")
            
            # Convert dates to Bloomberg format
            start_str = self.start_date.strftime('%Y%m%d')
            end_str = self.end_date.strftime('%Y%m%d')
            
            try:
                # Fetch daily price data
                fields = ['PX_LAST', 'VOLUME']
                stock_data = self.bbg.bdh(tickers, fields, start_str, end_str)
                
                if stock_data is not None and not stock_data.empty:
                    # Reshape and process the data
                    stock_data = stock_data.reset_index()
                    stock_data = stock_data.melt(id_vars='date', var_name=['ticker', 'field'], value_name='value')
                    stock_data = stock_data.pivot_table(index=['date', 'ticker'], columns='field', values='value').reset_index()
                    
                    # Rename columns
                    stock_data.columns.name = None
                    stock_data = stock_data.rename(columns={'PX_LAST': 'close', 'VOLUME': 'volume'})
                    
                    self.stock_data = stock_data
                    print(f"Fetched stock data for {len(tickers)} tickers.")
                else:
                    print("No stock data available from Bloomberg. Using simulated data instead.")
                    self.use_bloomberg = False
            except Exception as e:
                print(f"Error fetching stock data from Bloomberg: {e}")
                self.use_bloomberg = False
        
        # If Bloomberg is not available or no data was fetched, use simulated data
        if not self.use_bloomberg or self.stock_data is None or self.stock_data.empty:
            print("Generating simulated stock price data...")
            self.stock_data = self._generate_simulated_stock_data(tickers)
        
        return self.stock_data
    
    def _generate_simulated_stock_data(self, tickers):
        """
        Generate simulated stock price data for testing purposes.
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers to simulate data for
            
        Returns:
        --------
        pd.DataFrame
            Simulated stock price data
        """
        # Create date range for daily data
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        
        # Generate simulated data
        np.random.seed(42)  # For reproducibility
        
        # List to store data
        data = []
        
        for ticker in tickers:
            # Initial price (random between $10-$500)
            initial_price = np.random.uniform(10, 500)
            
            # Volatility (random between 0.1-0.5)
            volatility = np.random.uniform(0.1, 0.5)
            
            # Daily drift (random between -0.001 and 0.002)
            drift = np.random.uniform(-0.001, 0.002)
            
            # Generate log returns (random walk with drift)
            log_returns = np.random.normal(drift, volatility / np.sqrt(252), len(date_range))
            
            # Calculate prices
            prices = initial_price * np.exp(np.cumsum(log_returns))
            
            # Generate volumes (random between 100,000 and 10,000,000)
            volumes = np.random.randint(100000, 10000001, len(date_range))
            
            # Add to data list
            for i, date in enumerate(date_range):
                data.append({
                    'date': date,
                    'ticker': ticker,
                    'close': prices[i],
                    'volume': volumes[i]
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Calculate returns
        df = df.sort_values(['ticker', 'date'])
        df['prev_close'] = df.groupby('ticker')['close'].shift(1)
        df['return'] = df['close'] / df['prev_close'] - 1
        
        return df
    
    def calculate_trading_signals(self, horizons_df, recent_trades_window=30):
        """
        Calculate trading signals based on insider horizons and recent trades.
        
        Parameters:
        -----------
        horizons_df : pd.DataFrame
            DataFrame with insider horizons
        recent_trades_window : int
            Number of days to look back for recent trades
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with trading signals
        """
        if self.insider_data is None:
            raise ValueError("No insider data available. Call fetch_insider_data() first.")
        
        print(f"Calculating trading signals using a {recent_trades_window}-day window for recent trades...")
        
        # Get the most recent date in our data
        if 'as_of_date' in horizons_df.columns:
            as_of_date = horizons_df['as_of_date'].iloc[0]
        else:
            as_of_date = self.insider_data['trade_date'].max()
        
        # Filter insider trades to only include recent ones
        recent_cutoff = as_of_date - pd.DateOffset(days=recent_trades_window)
        recent_trades = self.insider_data[
            (self.insider_data['trade_date'] >= recent_cutoff) & 
            (self.insider_data['trade_date'] <= as_of_date)
        ].copy()
        
        # Merge with horizons data to add horizon information
        recent_trades = recent_trades.merge(
            horizons_df[['ticker', 'insider_id', 'horizon', 'horizon_type', 'avg_iof']], 
            on=['ticker', 'insider_id'], 
            how='left'
        )
        
        # For each ticker, calculate aggregate insider trading strength and signals
        signals = []
        
        for ticker, group in recent_trades.groupby('ticker'):
            # Skip if no horizon data for this ticker
            if group['horizon'].isna().all():
                continue
                
            # Calculate insider trading strength for each horizon type
            # STR = (Shares Purchased - Shares Sold) / Total Volume
            total_volume = self.stock_data[
                (self.stock_data['ticker'] == ticker) & 
                (self.stock_data['date'] >= recent_cutoff) & 
                (self.stock_data['date'] <= as_of_date)
            ]['volume'].sum()
            
            if total_volume == 0:
                continue  # Skip if no volume data
                
            # All insiders
            all_net_shares = group['shares'].sum()
            all_str = all_net_shares / total_volume if total_volume > 0 else 0
                
            # Long-horizon insiders
            lh_group = group[group['horizon_type'] == 'Long']
            lh_net_shares = lh_group['shares'].sum()
            lh_str = lh_net_shares / total_volume if total_volume > 0 else 0
                
            # Short-horizon insiders
            sh_group = group[group['horizon_type'] == 'Short']
            sh_net_shares = sh_group['shares'].sum()
            sh_str = sh_net_shares / total_volume if total_volume > 0 else 0
            
            # Calculate unexpected trades - focus on unexpected LH insider trades
            # An LH insider trade is unexpected if it's different from their typical pattern
            unexpected_lh_trades = False
            has_unexpected_lh_buy = False
            has_unexpected_lh_sell = False
            
            if not lh_group.empty:
                for _, insider in lh_group.groupby('insider_id'):
                    # Get the average IOF (whether insider typically buys or sells)
                    avg_iof = insider['avg_iof'].iloc[0] if 'avg_iof' in insider.columns else None
                    
                    # Check if there's any unexpected trade
                    if avg_iof is not None:
                        recent_purchase = (insider['transaction_type'] == 'P').any()
                        recent_sale = (insider['transaction_type'] == 'S').any()
                        
                        # Unexpected buy: LH insider who typically sells (avg_iof < 0) is now buying
                        if avg_iof < -0.9 and recent_purchase:
                            unexpected_lh_trades = True
                            has_unexpected_lh_buy = True
                            
                        # Unexpected sell: LH insider who typically buys (avg_iof > 0) is now selling
                        if avg_iof > 0.9 and recent_sale:
                            unexpected_lh_trades = True
                            has_unexpected_lh_sell = True
            
            # Generate trading signals based on the paper's findings:
            # 1. SH insider trades are more informative than LH insider trades
            # 2. Unexpected LH insider trades are even more informative
            
            # Default signal is neutral
            signal = 0
            signal_type = 'Neutral'
            signal_strength = 0
            
            # Check for unexpected LH insider trades (highest priority)
            if unexpected_lh_trades:
                if has_unexpected_lh_buy and not has_unexpected_lh_sell:
                    signal = 2  # Strong buy
                    signal_type = 'Strong Buy - Unexpected LH Buy'
                    signal_strength = 2
                elif has_unexpected_lh_sell and not has_unexpected_lh_buy:
                    signal = -2  # Strong sell
                    signal_type = 'Strong Sell - Unexpected LH Sell'
                    signal_strength = -2
                else:
                    # If there are both unexpected buys and sells, signal is mixed
                    signal = 0
                    signal_type = 'Mixed - Unexpected LH Trades'
                    signal_strength = 0
            
            # Check for SH insider trades if no unexpected LH insider signals
            elif not sh_group.empty and sh_str != 0:
                # Using the SH insider trading strength to determine signal
                if sh_str > 0.0001:  # Positive threshold for buys
                    signal = 1  # Buy
                    signal_type = 'Buy - SH Insider Buying'
                    signal_strength = min(sh_str * 10000, 1)  # Scale signal strength
                elif sh_str < -0.0001:  # Negative threshold for sells
                    signal = -1  # Sell
                    signal_type = 'Sell - SH Insider Selling'
                    signal_strength = max(sh_str * 10000, -1)  # Scale signal strength
            
            # Otherwise, check LH insider trades (lowest priority)
            elif not lh_group.empty and lh_str != 0:
                # Using the LH insider trading strength with reduced weight
                if lh_str > 0.0002:  # Higher threshold for LH buys
                    signal = 0.5  # Weak buy
                    signal_type = 'Weak Buy - LH Insider Buying'
                    signal_strength = min(lh_str * 5000, 0.5)  # Reduced weight
                elif lh_str < -0.0002:  # Higher threshold for LH sells
                    signal = -0.5  # Weak sell
                    signal_type = 'Weak Sell - LH Insider Selling'
                    signal_strength = max(lh_str * 5000, -0.5)  # Reduced weight
            
            signals.append({
                'ticker': ticker,
                'as_of_date': as_of_date,
                'signal': signal,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'all_str': all_str,
                'lh_str': lh_str,
                'sh_str': sh_str,
                'all_net_shares': all_net_shares,
                'lh_net_shares': lh_net_shares,
                'sh_net_shares': sh_net_shares,
                'unexpected_lh_trades': unexpected_lh_trades,
                'recent_trades_count': len(group)
            })
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)
        
        if not signals_df.empty:
            print(f"Generated trading signals for {len(signals_df)} tickers.")
            
            # Print signal distribution
            signal_counts = signals_df['signal_type'].value_counts()
            print("\nSignal distribution:")
            for signal_type, count in signal_counts.items():
                print(f"{signal_type}: {count} tickers")
        else:
            print("No trading signals generated. Check data or parameters.")
        
        return signals_df
    
    def backtest_strategy(self, signals_df, portfolio_size=10, holding_period=20,
                          include_transaction_costs=True, transaction_cost_pct=0.001):
        """
        Backtest the insider horizon trading strategy.
        
        Parameters:
        -----------
        signals_df : pd.DataFrame
            DataFrame with trading signals
        portfolio_size : int
            Number of stocks to include in the portfolio
        holding_period : int
            Number of days to hold each position
        include_transaction_costs : bool
            Whether to include transaction costs in the backtest
        transaction_cost_pct : float
            Transaction cost as a percentage of position value
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio performance
        """
        if self.stock_data is None:
            raise ValueError("No stock data available. Call fetch_stock_data() first.")
        
        if signals_df.empty:
            raise ValueError("No trading signals available. Call calculate_trading_signals() first.")
        
        print(f"Backtesting strategy with portfolio size {portfolio_size} and holding period {holding_period} days...")
        
        # Get the signal date
        signal_date = signals_df['as_of_date'].iloc[0]
        
        # Select the top N long and short signals
        long_signals = signals_df[signals_df['signal'] > 0].sort_values('signal_strength', ascending=False).head(portfolio_size)
        short_signals = signals_df[signals_df['signal'] < 0].sort_values('signal_strength', ascending=True).head(portfolio_size)
        
        # Get performance data for the signal tickers
        performance_data = []
        
        # Start date for backtest is the day after signal date
        start_date = signal_date + pd.DateOffset(days=1)
        end_date = start_date + pd.DateOffset(days=holding_period)
        
        # Get future stock data for the holding period
        future_stock_data = self.stock_data[
            (self.stock_data['date'] > signal_date) & 
            (self.stock_data['date'] <= end_date)
        ].copy()
        
        # Process long positions
        for _, signal in long_signals.iterrows():
            ticker_data = future_stock_data[future_stock_data['ticker'] == signal['ticker']].copy()
            
            if ticker_data.empty:
                continue
                
            # Calculate returns
            ticker_data = ticker_data.sort_values('date')
            ticker_data['position'] = 'LONG'
            ticker_data['signal_type'] = signal['signal_type']
            ticker_data['signal_strength'] = signal['signal_strength']
            
            # Calculate cumulative return
            initial_price = ticker_data['close'].iloc[0]
            ticker_data['cumulative_return'] = ticker_data['close'] / initial_price - 1
            
            # Add position weight (equal weight for now)
            ticker_data['position_weight'] = 1.0 / portfolio_size if len(long_signals) > 0 else 0
            
            # Add to performance data
            performance_data.append(ticker_data)
        
        # Process short positions
        for _, signal in short_signals.iterrows():
            ticker_data = future_stock_data[future_stock_data['ticker'] == signal['ticker']].copy()
            
            if ticker_data.empty:
                continue
                
            # Calculate returns (negative for short positions)
            ticker_data = ticker_data.sort_values('date')
            ticker_data['position'] = 'SHORT'
            ticker_data['signal_type'] = signal['signal_type']
            ticker_data['signal_strength'] = signal['signal_strength']
            
            # Calculate cumulative return (negative for short positions)
            initial_price = ticker_data['close'].iloc[0]
            ticker_data['cumulative_return'] = -1 * (ticker_data['close'] / initial_price - 1)
            
            # Add position weight (equal weight for now)
            ticker_data['position_weight'] = 1.0 / portfolio_size if len(short_signals) > 0 else 0
            
            # Add to performance data
            performance_data.append(ticker_data)
        
        # Combine performance data
        if performance_data:
            portfolio_df = pd.concat(performance_data)
            
            # Calculate weighted returns
            portfolio_df['weighted_return'] = portfolio_df['cumulative_return'] * portfolio_df['position_weight']
            
            # Aggregate portfolio performance by date
            portfolio_performance = portfolio_df.groupby('date').agg({
                'weighted_return': 'sum',
                'position_weight': 'sum'
            }).reset_index()
            
            # Create benchmark returns (equal-weighted average of all tickers)
            benchmark_data = future_stock_data.copy()
            benchmark_data = benchmark_data.sort_values(['ticker', 'date'])
            benchmark_data['return'] = benchmark_data.groupby('ticker')['close'].pct_change()
            benchmark_performance = benchmark_data.groupby('date')['return'].mean().reset_index()
            benchmark_performance.columns = ['date', 'benchmark_return']
            
            # Merge benchmark data
            portfolio_performance = portfolio_performance.merge(benchmark_performance, on='date', how='left')
            
            # Calculate cumulative returns
            portfolio_performance['cumulative_portfolio_return'] = (1 + portfolio_performance['weighted_return']).cumprod() - 1
            portfolio_performance['cumulative_benchmark_return'] = (1 + portfolio_performance['benchmark_return']).cumprod() - 1
            
            # Calculate excess return
            portfolio_performance['excess_return'] = portfolio_performance['weighted_return'] - portfolio_performance['benchmark_return']
            portfolio_performance['cumulative_excess_return'] = portfolio_performance['cumulative_portfolio_return'] - portfolio_performance['cumulative_benchmark_return']
            
            # Include transaction costs if specified
            if include_transaction_costs:
                # Apply transaction costs at the beginning (entry) and end (exit)
                # This is a simplification - in reality costs would be applied to each ticker separately
                entry_cost = portfolio_performance['position_weight'].iloc[0] * transaction_cost_pct
                exit_cost = portfolio_performance['position_weight'].iloc[-1] * transaction_cost_pct
                
                # Adjust the cumulative return
                portfolio_performance['cumulative_portfolio_return_after_costs'] = portfolio_performance['cumulative_portfolio_return'] - entry_cost
                portfolio_performance.loc[portfolio_performance.index[-1], 'cumulative_portfolio_return_after_costs'] -= exit_cost
                
                # Adjust excess return after costs
                portfolio_performance['cumulative_excess_return_after_costs'] = portfolio_performance['cumulative_portfolio_return_after_costs'] - portfolio_performance['cumulative_benchmark_return']
            
            # Calculate performance statistics
            final_portfolio_return = portfolio_performance['cumulative_portfolio_return'].iloc[-1]
            final_benchmark_return = portfolio_performance['cumulative_benchmark_return'].iloc[-1]
            final_excess_return = portfolio_performance['cumulative_excess_return'].iloc[-1]
            
            annualized_return = (1 + final_portfolio_return) ** (252 / holding_period) - 1
            annualized_benchmark = (1 + final_benchmark_return) ** (252 / holding_period) - 1
            annualized_excess = (1 + final_excess_return) ** (252 / holding_period) - 1
            
            # Calculate volatility and Sharpe ratio
            daily_returns = portfolio_performance['weighted_return'].std() * np.sqrt(252)
            sharpe_ratio = annualized_return / daily_returns if daily_returns > 0 else 0
            
            print("\nBacktest Results:")
            print(f"Holding Period Return: {final_portfolio_return:.2%}")
            print(f"Benchmark Return: {final_benchmark_return:.2%}")
            print(f"Excess Return: {final_excess_return:.2%}")
            print(f"Annualized Return: {annualized_return:.2%}")
            print(f"Annualized Benchmark: {annualized_benchmark:.2%}")
            print(f"Annualized Excess Return: {annualized_excess:.2%}")
            print(f"Annualized Volatility: {daily_returns:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            
            if include_transaction_costs:
                final_return_after_costs = portfolio_performance['cumulative_portfolio_return_after_costs'].iloc[-1]
                final_excess_after_costs = portfolio_performance['cumulative_excess_return_after_costs'].iloc[-1]
                annualized_return_after_costs = (1 + final_return_after_costs) ** (252 / holding_period) - 1
                
                print("\nAfter Transaction Costs:")
                print(f"Holding Period Return: {final_return_after_costs:.2%}")
                print(f"Excess Return: {final_excess_after_costs:.2%}")
                print(f"Annualized Return: {annualized_return_after_costs:.2%}")
            
            # Store portfolio performance
            self.portfolio_performance = portfolio_performance
            
            return portfolio_performance
        else:
            print("No portfolio data available for backtest.")
            return None
    
    def plot_performance(self):
        """
        Plot the performance of the strategy.
        """
        if self.portfolio_performance is None:
            raise ValueError("No portfolio performance available. Call backtest_strategy() first.")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot cumulative returns
        self.portfolio_performance.set_index('date', inplace=True)
        self.portfolio_performance['cumulative_portfolio_return'].plot(
            ax=ax1, label='Portfolio', color='blue', linewidth=2)
        self.portfolio_performance['cumulative_benchmark_return'].plot(
            ax=ax1, label='Benchmark', color='red', linewidth=2, linestyle='--')
        
        if 'cumulative_portfolio_return_after_costs' in self.portfolio_performance.columns:
            self.portfolio_performance['cumulative_portfolio_return_after_costs'].plot(
                ax=ax1, label='Portfolio (After Costs)', color='green', linewidth=2, linestyle='-.')
        
        ax1.set_title('Cumulative Returns', fontsize=14)
        ax1.set_ylabel('Return', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True)
        
        # Plot excess returns
        self.portfolio_performance['cumulative_excess_return'].plot(
            ax=ax2, label='Excess Return', color='purple', linewidth=2)
        
        if 'cumulative_excess_return_after_costs' in self.portfolio_performance.columns:
            self.portfolio_performance['cumulative_excess_return_after_costs'].plot(
                ax=ax2, label='Excess Return (After Costs)', color='orange', linewidth=2, linestyle='-.')
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Cumulative Excess Returns', fontsize=14)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Excess Return', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('insider_horizon_performance.png')
        print("Performance plot saved as 'insider_horizon_performance.png'")
        plt.close()
        
        # Reset index for future use
        self.portfolio_performance.reset_index(inplace=True)
    
    def analyze_performance_factors(self, signals_df):
        """
        Analyze what factors contributed to the strategy's performance.
        
        Parameters:
        -----------
        signals_df : pd.DataFrame
            DataFrame with trading signals
        """
        if self.portfolio_performance is None:
            raise ValueError("No portfolio performance available. Call backtest_strategy() first.")
        
        # Get the tickers in our portfolio
        portfolio_tickers = self.portfolio_performance['ticker'].unique()
        
        # Filter signals to include only portfolio tickers
        portfolio_signals = signals_df[signals_df['ticker'].isin(portfolio_tickers)].copy()
        
        if portfolio_signals.empty:
            print("No signal data for portfolio tickers.")
            return
        
        # Analyze performance by signal type
        performance_by_signal = []
        
        for ticker in portfolio_tickers:
            # Get the signal info
            signal_info = portfolio_signals[portfolio_signals['ticker'] == ticker]
            
            if signal_info.empty:
                continue
                
            # Get the ticker's performance
            ticker_perf = self.portfolio_performance[self.portfolio_performance['ticker'] == ticker].copy()
            
            if ticker_perf.empty:
                continue
                
            # Get final return for the ticker
            final_return = ticker_perf['cumulative_return'].iloc[-1]
            
            # Add to performance by signal
            performance_by_signal.append({
                'ticker': ticker,
                'signal': signal_info['signal'].iloc[0],
                'signal_type': signal_info['signal_type'].iloc[0],
                'signal_strength': signal_info['signal_strength'].iloc[0],
                'position': ticker_perf['position'].iloc[0],
                'return': final_return,
                'unexpected_lh_trades': signal_info['unexpected_lh_trades'].iloc[0]
            })
        
        # Convert to DataFrame
        perf_by_signal_df = pd.DataFrame(performance_by_signal)
        
        if not perf_by_signal_df.empty:
            # Group by position (LONG/SHORT)
            print("\nPerformance by Position Type:")
            position_perf = perf_by_signal_df.groupby('position')['return'].agg(['mean', 'count']).reset_index()
            print(position_perf)
            
            # Group by signal type
            print("\nPerformance by Signal Type:")
            signal_perf = perf_by_signal_df.groupby('signal_type')['return'].agg(['mean', 'count']).reset_index()
            print(signal_perf)
            
            # Analyze unexpected LH trades vs regular trades
            print("\nUnexpected LH Trades vs Regular Trades:")
            unexpected_perf = perf_by_signal_df.groupby('unexpected_lh_trades')['return'].agg(['mean', 'count']).reset_index()
            unexpected_perf['unexpected_lh_trades'] = unexpected_perf['unexpected_lh_trades'].map({True: 'Unexpected', False: 'Regular'})
            print(unexpected_perf)
            
            # Correlation between signal strength and returns
            correlation = perf_by_signal_df['signal_strength'].corr(perf_by_signal_df['return'])
            print(f"\nCorrelation between Signal Strength and Returns: {correlation:.4f}")
            
            # Save performance by signal type to CSV
            perf_by_signal_df.to_csv('insider_performance_by_signal.csv', index=False)
            print("Performance by signal saved to 'insider_performance_by_signal.csv'")
        else:
            print("No performance data available by signal type.")
    
    def run_full_analysis(self, tickers=None, portfolio_size=10, holding_period=20):
        """
        Run a full analysis including fetching data, calculating signals, and backtesting.
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers to analyze
        portfolio_size : int
            Number of stocks to include in the portfolio
        holding_period : int
            Number of days to hold each position
            
        Returns:
        --------
        tuple
            (insider_horizons, trading_signals, portfolio_performance)
        """
        # Fetch insider data
        self.fetch_insider_data(tickers)
        
        # Calculate insider horizons
        horizons_df = self.calculate_insider_horizons()
        
        # Get unique tickers from insider data
        if tickers is None:
            tickers = self.insider_data['ticker'].unique().tolist()
        
        # Fetch stock data
        self.fetch_stock_data(tickers)
        
        # Calculate trading signals
        signals_df = self.calculate_trading_signals(horizons_df)
        
        # Backtest strategy
        self.backtest_strategy(signals_df, portfolio_size=portfolio_size, holding_period=holding_period)
        
        # Plot performance
        if self.portfolio_performance is not None:
            self.plot_performance()
            self.analyze_performance_factors(signals_df)
        
        return horizons_df, signals_df, self.portfolio_performance


# Example usage
if __name__ == "__main__":
    # Define list of tickers to analyze (for Bloomberg)
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 
               'JPM', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'NFLX']
    
    # Set analysis parameters
    start_date = '2022-01-01'  # Start date for backtest
    lookback_years = 10        # Lookback period for calculating insider horizon
    portfolio_size = 5         # Number of stocks in portfolio
    holding_period = 20        # Number of days to hold positions
    
    # Create strategy instance
    strategy = InsiderHorizonStrategy(
        start_date=start_date,
        lookback_years=lookback_years,
        use_bloomberg=True    # Set to True to use Bloomberg API if available
    )
    
    # Run full analysis
    horizons, signals, performance = strategy.run_full_analysis(
        tickers=tickers,
        portfolio_size=portfolio_size,
        holding_period=holding_period
    )