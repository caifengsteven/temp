import pandas as pd
import numpy as np
import logging
import pdblp
import datetime
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("daily_supertrend_alerts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class DailySupertrendScanner:
    def __init__(self, instruments_file, atr_period=10, multiplier=3):
        """
        Initialize the Daily Supertrend Scanner.
        
        Args:
            instruments_file (str): Path to file containing list of instruments
            atr_period (int): Period for ATR calculation
            multiplier (int): Multiplier for Supertrend calculation
        """
        self.instruments_file = instruments_file
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.instruments = []
        self.bloomberg_conn = None
        
    def load_instruments(self):
        """Load instruments from file."""
        try:
            with open(self.instruments_file, 'r') as f:
                self.instruments = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.instruments)} instruments from {self.instruments_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading instruments file: {e}")
            return False
            
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
            
    def get_daily_data(self, ticker, lookback_days=100):
        """
        Get daily market data for a given ticker.
        
        Args:
            ticker (str): Bloomberg ticker
            lookback_days (int): How many days to look back
            
        Returns:
            DataFrame: OHLC data
        """
        try:
            end_date = datetime.datetime.now().strftime('%Y%m%d')
            start_date = (datetime.datetime.now() - datetime.timedelta(days=lookback_days)).strftime('%Y%m%d')
            
            # Request data from Bloomberg
            data = self.bloomberg_conn.bdh(
                tickers=ticker,
                flds=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'VOLUME'],
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
                
            # Flatten multi-level column index if needed
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[1] for col in data.columns]
                
            # Rename columns to standard OHLC format
            data = data.rename(columns={
                'PX_OPEN': 'Open',
                'PX_HIGH': 'High',
                'PX_LOW': 'Low',
                'PX_LAST': 'Close',
                'VOLUME': 'Volume'
            })
            
            logger.info(f"Retrieved {len(data)} days of data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting daily data for {ticker}: {e}")
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
            if i == self.atr_period:
                df.loc[df.index[i], 'ST_upper'] = df['basic_upper'].iloc[i]
                df.loc[df.index[i], 'ST_lower'] = df['basic_lower'].iloc[i]
                df.loc[df.index[i], 'Supertrend'] = df['basic_upper'].iloc[i]
                df.loc[df.index[i], 'Supertrend_direction'] = -1
                continue
                
            # Calculate upper band
            if df['basic_upper'].iloc[i] < df['ST_upper'].iloc[i-1] or df['Close'].iloc[i-1] > df['ST_upper'].iloc[i-1]:
                df.loc[df.index[i], 'ST_upper'] = df['basic_upper'].iloc[i]
            else:
                df.loc[df.index[i], 'ST_upper'] = df['ST_upper'].iloc[i-1]
                
            # Calculate lower band
            if df['basic_lower'].iloc[i] > df['ST_lower'].iloc[i-1] or df['Close'].iloc[i-1] < df['ST_lower'].iloc[i-1]:
                df.loc[df.index[i], 'ST_lower'] = df['basic_lower'].iloc[i]
            else:
                df.loc[df.index[i], 'ST_lower'] = df['ST_lower'].iloc[i-1]
                
            # Determine Supertrend value and direction
            if df['ST_upper'].iloc[i-1] == df['Supertrend'].iloc[i-1] and df['Close'].iloc[i] > df['ST_upper'].iloc[i]:
                # Crossing above upper band
                df.loc[df.index[i], 'Supertrend'] = df['ST_lower'].iloc[i]
                df.loc[df.index[i], 'Supertrend_direction'] = 1
            elif df['ST_lower'].iloc[i-1] == df['Supertrend'].iloc[i-1] and df['Close'].iloc[i] < df['ST_lower'].iloc[i]:
                # Crossing below lower band
                df.loc[df.index[i], 'Supertrend'] = df['ST_upper'].iloc[i]
                df.loc[df.index[i], 'Supertrend_direction'] = -1
            else:
                # No crossing
                if df['Close'].iloc[i] > df['Supertrend'].iloc[i-1]:
                    df.loc[df.index[i], 'Supertrend'] = df['ST_lower'].iloc[i]
                    df.loc[df.index[i], 'Supertrend_direction'] = 1
                else:
                    df.loc[df.index[i], 'Supertrend'] = df['ST_upper'].iloc[i]
                    df.loc[df.index[i], 'Supertrend_direction'] = -1
        
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
        if data is None or len(data) < 2:
            return None
            
        # Get the last two rows to check for crossovers
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Buy signal: Direction changed from -1 to 1 (price crossed above Supertrend)
        if current['Supertrend_direction'] == 1 and previous['Supertrend_direction'] == -1:
            return f"BUY SIGNAL for {ticker}: Price closed above Supertrend at {current['Close']} on {current.name.date()}"
            
        # Sell signal: Direction changed from 1 to -1 (price crossed below Supertrend)
        elif current['Supertrend_direction'] == -1 and previous['Supertrend_direction'] == 1:
            return f"SELL SIGNAL for {ticker}: Price closed below Supertrend at {current['Close']} on {current.name.date()}"
            
        return None
        
    def process_instrument(self, ticker):
        """Process a single instrument."""
        try:
            # Get daily data
            data = self.get_daily_data(ticker)
            if data is None or len(data) < self.atr_period:
                logger.warning(f"Insufficient data for {ticker} to calculate Supertrend")
                return
                
            # Calculate Supertrend
            st_data = self.calculate_supertrend(data)
            if st_data is None:
                return
                
            # Check for signals
            alert = self.check_signals(ticker, st_data)
            if alert:
                logger.info(alert)
                print(alert)  # Print to console as well
                
            # Additional information about current status
            current = st_data.iloc[-1]
            trend_direction = "UPTREND" if current['Supertrend_direction'] == 1 else "DOWNTREND"
            logger.info(f"{ticker} is in {trend_direction}. Close: {current['Close']}, Supertrend: {current['Supertrend']:.2f}")
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
    
    def run_scan(self):
        """Run the scan for all instruments."""
        logger.info("Starting daily Supertrend scan...")
        
        # Load instruments
        if not self.load_instruments() or not self.instruments:
            logger.error("No instruments to scan")
            return False
            
        # Connect to Bloomberg
        if not self.connect_to_bloomberg():
            logger.error("Cannot run scan without Bloomberg connection")
            return False
            
        # Process each instrument
        for ticker in self.instruments:
            logger.info(f"Processing {ticker}")
            self.process_instrument(ticker)
            
        logger.info("Scan completed")
        
        # Close Bloomberg connection
        if self.bloomberg_conn:
            self.bloomberg_conn.stop()
            logger.info("Bloomberg connection closed")
            
        return True


if __name__ == "__main__":
    # Get instruments file from command line or use default
    instruments_file = sys.argv[1] if len(sys.argv) > 1 else "instrument.txt"
    
    if not os.path.exists(instruments_file):
        print(f"Error: Instruments file '{instruments_file}' not found.")
        sys.exit(1)
    
    # Create scanner and run
    scanner = DailySupertrendScanner(
        instruments_file=instruments_file,
        atr_period=14,  # Typical ATR period
        multiplier=3    # Supertrend multiplier
    )
    
    scanner.run_scan()