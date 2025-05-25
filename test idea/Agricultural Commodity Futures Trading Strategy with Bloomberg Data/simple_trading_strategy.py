import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import blpapi
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class BloombergDataManager:
    def __init__(self):
        """Initialize Bloomberg connection"""
        self.session = None
        self.connected = False
        
    def connect(self):
        """Establish connection to Bloomberg API"""
        try:
            self.session = blpapi.Session()
            if not self.session.start():
                print("❌ Failed to start Bloomberg session.")
                return False
            
            if not self.session.openService("//blp/refdata"):
                print("❌ Failed to open //blp/refdata service.")
                return False
            
            self.connected = True
            print("✅ Successfully connected to Bloomberg")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Bloomberg: {e}")
            print("Please ensure Bloomberg Terminal is running and you have proper permissions.")
            return False
    
    def check_connection(self):
        """Check if Bloomberg connection is active"""
        if not self.connected:
            return self.connect()
        return True
    
    def get_historical_data(self, tickers, start_date, end_date, fields=['PX_LAST']):
        """
        Retrieve historical data from Bloomberg
        
        Args:
            tickers: List of Bloomberg tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fields: Bloomberg fields to retrieve
            
        Returns:
            DataFrame with historical data
        """
        if not self.check_connection():
            print("❌ Bloomberg connection not available. Cannot retrieve data.")
            return None
        
        try:
            print(f"📊 Retrieving historical data for {len(tickers)} tickers ({start_date} to {end_date})...")
            
            # Get the reference data service
            refDataService = self.session.getService("//blp/refdata")
            
            # Create a request for historical data
            request = refDataService.createRequest("HistoricalDataRequest")
            
            # Set the securities
            for ticker in tickers:
                request.append("securities", ticker)
            
            # Set the fields
            for field in fields:
                request.append("fields", field)
            
            # Set the date range
            request.set("startDate", start_date.replace("-", ""))
            request.set("endDate", end_date.replace("-", ""))
            
            # Send the request
            self.session.sendRequest(request)
            
            # Process the response
            data_dict = {}
            dates = set()
            
            while True:
                event = self.session.nextEvent(500)
                for msg in event:
                    if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                        securityData = msg.getElement("securityData")
                        security = securityData.getElementAsString("security")
                        fieldData = securityData.getElement("fieldData")
                        
                        for i in range(fieldData.numValues()):
                            row = fieldData.getValue(i)
                            date = row.getElementAsDatetime("date").strftime("%Y-%m-%d")
                            dates.add(date)
                            
                            for field in fields:
                                if row.hasElement(field):
                                    value = row.getElementAsFloat(field)
                                    if security not in data_dict:
                                        data_dict[security] = {}
                                    if date not in data_dict[security]:
                                        data_dict[security][date] = {}
                                    data_dict[security][date][field] = value
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            # Convert to DataFrame
            if not data_dict:
                print("⚠️ No data retrieved from Bloomberg")
                return None
            
            # Create a multi-index DataFrame
            dates_list = sorted(list(dates))
            index = pd.DatetimeIndex(dates_list)
            
            if len(fields) == 1:
                # Single field - create a simple DataFrame
                data = pd.DataFrame(index=index)
                for security in data_dict:
                    values = []
                    for date in dates_list:
                        if date in data_dict[security] and fields[0] in data_dict[security][date]:
                            values.append(data_dict[security][date][fields[0]])
                        else:
                            values.append(np.nan)
                    data[security] = values
            else:
                # Multiple fields - create a multi-level column DataFrame
                columns = pd.MultiIndex.from_product([tickers, fields])
                data = pd.DataFrame(index=index, columns=columns)
                
                for security in data_dict:
                    for date in dates_list:
                        if date in data_dict[security]:
                            for field in fields:
                                if field in data_dict[security][date]:
                                    data.loc[date, (security, field)] = data_dict[security][date][field]
            
            print(f"✅ Successfully retrieved data with shape: {data.shape}")
            return data
            
        except Exception as e:
            print(f"❌ Error retrieving data from Bloomberg: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def close(self):
        """Close Bloomberg connection"""
        if self.connected and self.session is not None:
            try:
                self.session.stop()
                self.connected = False
                print("✅ Bloomberg connection closed")
            except Exception as e:
                print(f"❌ Error closing Bloomberg connection: {e}")

class SimpleMovingAverageCrossoverStrategy:
    def __init__(self, data, short_window=20, long_window=50):
        """
        Initialize Simple Moving Average Crossover Strategy
        
        Args:
            data: DataFrame with price data
            short_window: Short moving average window
            long_window: Long moving average window
        """
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signals = None
        self.positions = None
        self.portfolio = None
        
    def generate_signals(self):
        """
        Generate trading signals based on moving average crossover
        
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=self.data.index)
        
        for ticker in self.data.columns:
            # Create signal series
            signals[f'{ticker}_signal'] = 0.0
            
            # Create short and long moving averages
            signals[f'{ticker}_short_mavg'] = self.data[ticker].rolling(window=self.short_window, min_periods=1).mean()
            signals[f'{ticker}_long_mavg'] = self.data[ticker].rolling(window=self.long_window, min_periods=1).mean()
            
            # Create signals
            signals[f'{ticker}_signal'][self.short_window:] = np.where(
                signals[f'{ticker}_short_mavg'][self.short_window:] > signals[f'{ticker}_long_mavg'][self.short_window:], 1.0, 0.0)
            
            # Generate trading orders
            signals[f'{ticker}_positions'] = signals[f'{ticker}_signal'].diff()
        
        self.signals = signals
        return signals
    
    def backtest_strategy(self, initial_capital=100000.0):
        """
        Backtest the trading strategy
        
        Args:
            initial_capital: Initial capital
            
        Returns:
            DataFrame with portfolio performance
        """
        if self.signals is None:
            self.generate_signals()
        
        # Create portfolio DataFrame
        portfolio = pd.DataFrame(index=self.signals.index)
        portfolio['positions'] = 0.0
        portfolio['cash'] = initial_capital
        portfolio['total'] = initial_capital
        portfolio['returns'] = 0.0
        
        # Iterate through each ticker
        for ticker in self.data.columns:
            # Calculate positions and holdings
            position_size = initial_capital / len(self.data.columns) / self.data[ticker].iloc[0]
            portfolio[f'{ticker}_position'] = self.signals[f'{ticker}_signal'] * position_size
            portfolio[f'{ticker}_holdings'] = portfolio[f'{ticker}_position'] * self.data[ticker]
            
            # Update portfolio value
            portfolio['positions'] += portfolio[f'{ticker}_holdings']
        
        # Calculate total portfolio value and returns
        portfolio['total'] = portfolio['positions'] + portfolio['cash']
        portfolio['returns'] = portfolio['total'].pct_change()
        
        self.portfolio = portfolio
        return portfolio
    
    def plot_strategy_performance(self, ticker=None):
        """
        Plot strategy performance
        
        Args:
            ticker: Specific ticker to plot (if None, plot overall portfolio)
        """
        if self.signals is None or self.portfolio is None:
            self.backtest_strategy()
        
        if ticker is None:
            # Plot overall portfolio performance
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot portfolio value
            ax1.plot(self.portfolio.index, self.portfolio['total'], label='Portfolio Value')
            ax1.set_title('Portfolio Value')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True)
            
            # Plot portfolio returns
            ax2.plot(self.portfolio.index, self.portfolio['returns'], label='Portfolio Returns')
            ax2.set_title('Portfolio Returns')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Returns')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('portfolio_performance.png')
            print(f"✅ Portfolio performance plot saved to portfolio_performance.png")
        else:
            # Plot specific ticker performance
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot price and moving averages
            ax1.plot(self.data.index, self.data[ticker], label=f'{ticker} Price')
            ax1.plot(self.signals.index, self.signals[f'{ticker}_short_mavg'], label=f'{self.short_window} Day MA')
            ax1.plot(self.signals.index, self.signals[f'{ticker}_long_mavg'], label=f'{self.long_window} Day MA')
            
            # Plot buy signals
            ax1.plot(self.signals.loc[self.signals[f'{ticker}_positions'] == 1.0].index, 
                    self.data[ticker][self.signals[f'{ticker}_positions'] == 1.0],
                    '^', markersize=10, color='g', label='Buy')
            
            # Plot sell signals
            ax1.plot(self.signals.loc[self.signals[f'{ticker}_positions'] == -1.0].index, 
                    self.data[ticker][self.signals[f'{ticker}_positions'] == -1.0],
                    'v', markersize=10, color='r', label='Sell')
            
            ax1.set_title(f'{ticker} Price and Moving Averages')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            
            # Plot holdings
            ax2.plot(self.portfolio.index, self.portfolio[f'{ticker}_holdings'], label=f'{ticker} Holdings')
            ax2.set_title(f'{ticker} Holdings')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Value')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{ticker}_performance.png')
            print(f"✅ {ticker} performance plot saved to {ticker}_performance.png")
    
    def get_performance_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if self.portfolio is None:
            self.backtest_strategy()
        
        # Calculate metrics
        total_return = (self.portfolio['total'].iloc[-1] / self.portfolio['total'].iloc[0]) - 1
        annual_return = total_return / (len(self.portfolio) / 252)
        volatility = self.portfolio['returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        portfolio_value = self.portfolio['total']
        drawdown = portfolio_value / portfolio_value.cummax() - 1
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        trades = 0
        winning_trades = 0
        
        for ticker in self.data.columns:
            positions = self.signals[f'{ticker}_positions']
            trades += (positions != 0).sum()
            
            # Calculate profit for each trade
            trade_profit = []
            in_position = False
            entry_price = 0
            
            for i in range(len(positions)):
                if positions.iloc[i] == 1.0:  # Buy signal
                    in_position = True
                    entry_price = self.data[ticker].iloc[i]
                elif positions.iloc[i] == -1.0 and in_position:  # Sell signal
                    exit_price = self.data[ticker].iloc[i]
                    profit = exit_price - entry_price
                    trade_profit.append(profit)
                    in_position = False
            
            winning_trades += sum(1 for profit in trade_profit if profit > 0)
        
        win_rate = winning_trades / trades if trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': trades
        }

def run_simple_trading_strategy():
    """
    Run a simple trading strategy on agricultural commodity futures
    """
    # Initialize Bloomberg connection
    bloomberg = BloombergDataManager()
    bloomberg.connect()
    
    # Define agricultural commodity futures tickers with proper Bloomberg syntax
    tickers = [
        'SB1 Comdty',       # ICE eleventh sugar
        'KC1 Comdty',       # ICE coffee
        'CC1 Comdty'        # ICE cocoa
    ]
    
    # Define date range
    start_date = '2023-01-01'
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Retrieve data from Bloomberg
    data = bloomberg.get_historical_data(tickers, start_date, end_date)
    
    # Close Bloomberg connection
    bloomberg.close()
    
    if data is not None and not data.empty:
        # Save raw data
        data.to_csv('agricultural_futures_data.csv')
        print(f"✅ Data saved to agricultural_futures_data.csv")
        
        # Display sample data
        print("\nSample data:")
        print(data.head())
        
        # Initialize and run strategy
        strategy = SimpleMovingAverageCrossoverStrategy(data, short_window=20, long_window=50)
        signals = strategy.generate_signals()
        portfolio = strategy.backtest_strategy(initial_capital=100000.0)
        
        # Plot strategy performance
        strategy.plot_strategy_performance()
        
        # Plot individual ticker performance
        for ticker in data.columns:
            strategy.plot_strategy_performance(ticker)
        
        # Calculate and display performance metrics
        metrics = strategy.get_performance_metrics()
        
        print("\n===== Strategy Performance Metrics =====")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Total Trades: {metrics['trades']}")
        
        return {
            'data': data,
            'signals': signals,
            'portfolio': portfolio,
            'metrics': metrics
        }
    else:
        print("❌ Failed to retrieve data. Exiting.")
        return None

if __name__ == "__main__":
    print("Starting Simple Trading Strategy...")
    try:
        results = run_simple_trading_strategy()
        print("Strategy execution completed successfully.")
    except Exception as e:
        import traceback
        print(f"Error executing strategy: {e}")
        traceback.print_exc()
