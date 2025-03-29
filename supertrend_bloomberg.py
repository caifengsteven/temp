import pandas as pd
import numpy as np
import time
import logging
import os
from datetime import datetime, timedelta
import schedule
import pdblp  # Python wrapper for Bloomberg API

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_alerts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class SupertrendStrategy:
    def __init__(self, instruments_file, atr_period=10, multiplier=3):
        """
        Initialize the Supertrend strategy.
        
        Args:
            instruments_file (str): Path to file containing list of instruments
            atr_period (int): Period for ATR calculation
            multiplier (int): Multiplier for Supertrend calculation
        """
        self.instruments_file = instruments_file
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.instruments = self.load_instruments()
        self.bloomberg_conn = None
        self.last_alerts = {}  # To avoid duplicate alerts
        
    def load_instruments(self):
        """Load instruments from file."""
        try:
            with open(self.instruments_file, 'r') as f:
                instruments = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(instruments)} instruments from {self.instruments_file}")
            return instruments
        except Exception as e:
            logger.error(f"Error loading instruments file: {e}")
            return []
            
    def connect_to_bloomberg(self):
        """Connect to Bloomberg API."""
        try:
            self.bloomberg_conn = pdblp.BCon(timeout=5000)
            self.bloomberg_conn.start()
            logger.info("Successfully connected to Bloomberg")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Bloomberg: {e}")
            return False
            
    def get_intraday_data(self, ticker, interval=1, lookback=30):
        """
        Get intraday data for a given ticker.
        
        Args:
            ticker (str): Bloomberg ticker
            interval (int): Bar interval in minutes
            lookback (int): How many minutes to look back
            
        Returns:
            DataFrame: OHLC data
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback)
            
            # Format the start and end time strings
            start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Request data from Bloomberg
            data = self.bloomberg_conn.bdib(
                ticker, 
                event_type="TRADE",
                interval=interval, 
                start_datetime=start_str, 
                end_datetime=end_str
            )
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
                
            # Rename columns to standard OHLC format
            data = data.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            logger.info(f"Retrieved {len(data)} bars for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting intraday data for {ticker}: {e}")
            return None
    
    def calculate_supertrend(self, data):
        """
        Calculate Supertrend indicator.
        
        Args:
            data (DataFrame): OHLC data
            
        Returns:
            DataFrame: Original data with Supertrend values added
        """
        if data is None or len(data) < self.atr_period:
            return None
            
        # Make a copy to avoid SettingWithCopyWarning
        df = data.copy()
        
        # Calculate True Range (TR)
        df['TR'] = np.maximum(
            np.maximum(
                df['High'] - df['Low'],
                abs(df['High'] - df['Close'].shift(1))
            ),
            abs(df['Low'] - df['Close'].shift(1))
        )
        
        # Calculate Average True Range (ATR)
        df['ATR'] = df['TR'].rolling(self.atr_period).mean()
        
        # Calculate basic upper and lower bands
        df['basic_upper'] = (df['High'] + df['Low']) / 2 + (self.multiplier * df['ATR'])
        df['basic_lower'] = (df['High'] + df['Low']) / 2 - (self.multiplier * df['ATR'])
        
        # Initialize Supertrend columns
        df['ST_upper'] = np.nan
        df['ST_lower'] = np.nan
        df['Supertrend'] = np.nan
        df['Supertrend_direction'] = np.nan
        
        # Calculate Supertrend
        for i in range(self.atr_period, len(df)):
            # Calculate final upper band
            if df['basic_upper'].iloc[i] < df['ST_upper'].iloc[i-1] or df['Close'].iloc[i-1] > df['ST_upper'].iloc[i-1]:
                df.loc[df.index[i], 'ST_upper'] = df['basic_upper'].iloc[i]
            else:
                df.loc[df.index[i], 'ST_upper'] = df['ST_upper'].iloc[i-1]
                
            # Calculate final lower band
            if df['basic_lower'].iloc[i] > df['ST_lower'].iloc[i-1] or df['Close'].iloc[i-1] < df['ST_lower'].iloc[i-1]:
                df.loc[df.index[i], 'ST_lower'] = df['basic_lower'].iloc[i]
            else:
                df.loc[df.index[i], 'ST_lower'] = df['ST_lower'].iloc[i-1]
                
            # Determine Supertrend value and direction
            if df['Close'].iloc[i] > df['ST_upper'].iloc[i-1]:
                df.loc[df.index[i], 'Supertrend'] = df['ST_lower'].iloc[i]
                df.loc[df.index[i], 'Supertrend_direction'] = 1  # Uptrend
            elif df['Close'].iloc[i] < df['ST_lower'].iloc[i-1]:
                df.loc[df.index[i], 'Supertrend'] = df['ST_upper'].iloc[i]
                df.loc[df.index[i], 'Supertrend_direction'] = -1  # Downtrend
            else:
                df.loc[df.index[i], 'Supertrend'] = df['Supertrend'].iloc[i-1]
                df.loc[df.index[i], 'Supertrend_direction'] = df['Supertrend_direction'].iloc[i-1]
                
        return df
        
    def check_signals(self, ticker, data):
        """
        Check for buy/sell signals based on Supertrend.
        
        Args:
            ticker (str): Instrument ticker
            data (DataFrame): OHLC data with Supertrend values
            
        Returns:
            str: Alert message if signal detected, None otherwise
        """
        if data is None or data.empty or 'Supertrend_direction' not in data.columns:
            return None
            
        # Get the last two rows to check for crossovers
        if len(data) < 2:
            return None
            
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Check if we've already generated an alert for this signal
        signal_key = f"{ticker}_{current.name}"  # Use timestamp as part of the key
        
        # Buy signal: Price closes above Supertrend (crossing from below)
        if current['Supertrend_direction'] == 1 and previous['Supertrend_direction'] == -1:
            message = f"BUY SIGNAL for {ticker}: Price closed above Supertrend at {current['Close']}"
            if signal_key not in self.last_alerts:
                self.last_alerts[signal_key] = message
                return message
                
        # Sell signal: Price closes below Supertrend (crossing from above)
        elif current['Supertrend_direction'] == -1 and previous['Supertrend_direction'] == 1:
            message = f"SELL SIGNAL for {ticker}: Price closed below Supertrend at {current['Close']}"
            if signal_key not in self.last_alerts:
                self.last_alerts[signal_key] = message
                return message
                
        return None
        
    def process_instrument(self, ticker):
        """Process a single instrument."""
        try:
            # Get 30-minute data
            data_30min = self.get_intraday_data(ticker, interval=30, lookback=30*5)  # Get more bars for calculation
            
            if data_30min is None or len(data_30min) < self.atr_period:
                logger.warning(f"Insufficient data for {ticker} to calculate Supertrend")
                return
                
            # Calculate Supertrend on 30-minute data
            st_data = self.calculate_supertrend(data_30min)
            if st_data is None:
                return
                
            # Get latest 1-minute bar to check against Supertrend
            latest_1min = self.get_intraday_data(ticker, interval=1, lookback=1)
            if latest_1min is None or latest_1min.empty:
                return
                
            # Compare the close of the 1-minute bar with the Supertrend level
            current_close = latest_1min.iloc[-1]['Close']
            current_time = latest_1min.index[-1]
            
            # Get latest Supertrend values
            latest_st = st_data.iloc[-1]
            
            # Prepare a ticker data structure for signal checking
            signal_data = pd.DataFrame({
                'Open': [latest_st['Open'], latest_1min.iloc[-1]['Open']],
                'High': [latest_st['High'], latest_1min.iloc[-1]['High']],
                'Low': [latest_st['Low'], latest_1min.iloc[-1]['Low']],
                'Close': [latest_st['Close'], latest_1min.iloc[-1]['Close']],
                'Supertrend': [latest_st['Supertrend'], latest_st['Supertrend']],
                'Supertrend_direction': [
                    latest_st['Supertrend_direction'], 
                    1 if current_close > latest_st['Supertrend'] else -1
                ]
            }, index=[latest_st.name, current_time])
            
            # Check for signals
            alert = self.check_signals(ticker, signal_data)
            if alert:
                logger.info(alert)
                # Here you could add code to send SMS, email, or other notifications
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
    
    def run_scan(self):
        """Run the scan for all instruments."""
        logger.info("Starting market scan...")
        
        if self.bloomberg_conn is None or not self.bloomberg_conn._session:
            success = self.connect_to_bloomberg()
            if not success:
                logger.error("Cannot run scan without Bloomberg connection")
                return
                
        for ticker in self.instruments:
            logger.info(f"Processing {ticker}")
            self.process_instrument(ticker)
            
        logger.info("Scan completed")
            
    def start(self):
        """Start the strategy with scheduled runs."""
        # Connect to Bloomberg
        if not self.connect_to_bloomberg():
            logger.error("Failed to start strategy due to Bloomberg connection failure")
            return False
            
        # Run immediately once
        self.run_scan()
        
        # Schedule to run every 10 minutes
        schedule.every(10).minutes.do(self.run_scan)
        
        logger.info("Strategy started, running every 10 minutes")
        
        # Keep the script running
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            if self.bloomberg_conn:
                self.bloomberg_conn.stop()
                logger.info("Bloomberg connection closed")
                
        return True


if __name__ == "__main__":
    # Path to file containing list of instruments (one ticker per line)
    instruments_file = "instrument.txt"
    
    # Create and start the strategy
    strategy = SupertrendStrategy(
        instruments_file=instruments_file,
        atr_period=14,  # Typical ATR period
        multiplier=3    # Supertrend multiplier
    )
    
    strategy.start()