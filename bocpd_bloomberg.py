"""
Real-time Change-Point Detection System with Interactive Visualization

This application connects to Bloomberg, monitors stocks at 1-minute intervals,
uses Score-Driven Bayesian Online Change-Point Detection (SD-BOCPD) to
identify regime changes, and provides real-time visualization with highlighted
abnormal points.
"""

import os
import sys
import time
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import blpapi
from threading import Thread, Lock
import queue
import logging
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import the SD-BOCPD package
from ScoreDrivenBOCPD.sd_bocpd import SDBocpd, Hazard
from ScoreDrivenBOCPD.prob_model import GaussianModel


"""
How to Use This Code
Prerequisites:
Bloomberg Desktop API installed and running
blpapi Python package installed
SD-BOCPD package installed (from the files you provided)
Required Python packages: numpy, pandas, matplotlib, tkinter
Create an Instrument List File:
Create a text file with one Bloomberg ticker per line, for example:

Copy
AAPL US Equity
MSFT US Equity
AMZN US Equity
GOOGL US Equity
Run the Program:

Copy
python bloomberg_cpt_visualization.py instruments.txt
Features and Functionality
1. Interactive Real-Time Visualization
This application provides a comprehensive GUI with:

Price Chart with Highlighted Change Points:
The main chart displays 1-minute price data as a line chart
Detected change points are highlighted with red triangles
Vertical red dashed lines mark the times of regime changes
Volume Chart:
A synchronized volume chart below the price chart
Helps correlate price changes with trading activity
Instrument Selection Panel:
A list of all monitored instruments
Click on any instrument to view its chart and change points
Change Point Information Panel:
Lists recent change points with timestamps
Shows the probability/confidence of each change point
2. Bloomberg Data Integration
Connects to Bloomberg API to fetch real-time and historical data
Retrieves 1-minute OHLCV (Open, High, Low, Close, Volume) bars
Synchronizes data collection to minute boundaries
3. Advanced Change Point Detection
Uses the SD-BOCPD algorithm to detect regime changes
Processes data in real-time as it arrives
Automatically marks data points where abnormal patterns are detected
4. Technical Features
Multi-threaded Architecture:
Separate threads for data collection, processing, and UI
Ensures responsive interface during data processing
Data Management:
Maintains historical data for each instrument
Efficiently stores and retrieves data for visualization
Real-time Updates:
Charts update every second to show latest data
Change point information updates automatically
Customization Options
You can modify these parameters to adjust the change point detection:

In the ChangePointDetector initialization:
python

Run

Copy
detector = ChangePointDetector(
    window_size=120,     # Number of data points to consider
    hazard_rate=0.01,    # Prior probability of change point (1% per minute)
    model_type=1,        # 1 for MBOC model (most sophisticated)
    score_type=0.5       # 0.5 for robust score function
)
Chart Display Options:
Adjust the chart size, layout, and appearance in the _create_widgets method
Modify the chart update interval by changing the 1000 ms value in _timer_update_chart
Data Collection Frequency:
The application is set to collect data every minute
For testing or special cases, you could modify the collection interval in _collect_data
How It Works
When you start monitoring:
The app connects to Bloomberg and retrieves 2 hours of historical data
It initializes the change point detectors with this data
It displays the historical data in the chart
Every minute:
A new 1-minute bar is fetched from Bloomberg for each instrument
The data is added to the time series for that instrument
The SD-BOCPD algorithm processes the new data point
When a change point is detected:
The application marks the data point as a change point
It updates the chart with a highlighted marker
It logs the event and updates the change point information panel
The chart updates automatically every second to show the latest data and any newly detected change points.
This application provides a complete solution for real-time monitoring of multiple stocks, with sophisticated change point detection and interactive visualization that clearly highlights abnormal points in the time series.

"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bloomberg_cpt_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BloombergCPTDetector")

class BloombergDataFetcher:
    """Class to handle Bloomberg API connection and data retrieval"""
    
    def __init__(self):
        self.session = None
        self.refDataService = None
        self.is_connected = False
        self.lock = Lock()
        
    def connect(self):
        """Connect to Bloomberg API"""
        try:
            # Initialize session
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost("localhost")
            sessionOptions.setServerPort(8194)
            
            self.session = blpapi.Session(sessionOptions)
            if not self.session.start():
                logger.error("Failed to start Bloomberg API session.")
                return False
            
            if not self.session.openService("//blp/refdata"):
                logger.error("Failed to open //blp/refdata service")
                return False
            
            self.refDataService = self.session.getService("//blp/refdata")
            self.is_connected = True
            logger.info("Connected to Bloomberg API")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Bloomberg: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Bloomberg API"""
        with self.lock:
            if self.session:
                self.session.stop()
                self.is_connected = False
                logger.info("Disconnected from Bloomberg API")
    
    def get_historical_bars(self, security, start_datetime, end_datetime, interval=1):
        """
        Retrieve historical bar data for a security
        
        Parameters:
        security (str): Bloomberg security identifier (e.g., 'AAPL US Equity')
        start_datetime (datetime): Start time
        end_datetime (datetime): End time
        interval (int): Bar interval in minutes (default 1)
        
        Returns:
        DataFrame: Bar data with columns for time, open, high, low, close, volume
        """
        with self.lock:
            if not self.is_connected:
                logger.error("Not connected to Bloomberg API")
                return None
            
            try:
                # Format datetime strings for Bloomberg
                start_str = start_datetime.strftime("%Y-%m-%dT%H:%M:%S")
                end_str = end_datetime.strftime("%Y-%m-%dT%H:%M:%S")
                
                # Create request
                request = self.refDataService.createRequest("IntradayBarRequest")
                request.set("security", security)
                request.set("eventType", "TRADE")
                request.set("interval", interval)  # 1-minute bars
                request.set("startDateTime", start_str)
                request.set("endDateTime", end_str)
                
                logger.debug(f"Requesting bar data for {security} from {start_str} to {end_str}")
                
                # Send request
                self.session.sendRequest(request)
                
                # Process response
                data = []
                while True:
                    event = self.session.nextEvent(500)  # Timeout in milliseconds
                    
                    for msg in event:
                        if msg.messageType() == blpapi.Name("IntradayBarResponse"):
                            barData = msg.getElement("barData")
                            barTickData = barData.getElement("barTickData")
                            
                            for bar in barTickData.values():
                                time_value = bar.getElementAsDatetime("time")
                                open_value = bar.getElementAsFloat("open")
                                high_value = bar.getElementAsFloat("high")
                                low_value = bar.getElementAsFloat("low")
                                close_value = bar.getElementAsFloat("close")
                                volume_value = bar.getElementAsInteger("volume")
                                
                                data.append((time_value, open_value, high_value, 
                                            low_value, close_value, volume_value))
                    
                    if event.eventType() == blpapi.Event.RESPONSE:
                        break
                
                if not data:
                    logger.warning(f"No data retrieved for {security}")
                    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
                
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
                return df
                
            except Exception as e:
                logger.error(f"Error fetching bar data for {security}: {e}")
                return None


class ChangePointDetector:
    """Class to detect change points in time series data using SD-BOCPD"""
    
    def __init__(self, window_size=120, hazard_rate=0.01, model_type=1, score_type=0.5):
        """
        Initialize the change point detector
        
        Parameters:
        window_size (int): Number of historical points to keep
        hazard_rate (float): Prior probability of change at each time point
        model_type (int): 0 for BOCPD, 0.5 for MBO, 1 for MBOC
        score_type (float): 0 for Gaussian score, 0.5 for absolute error score
        """
        self.window_size = window_size
        self.hazard_rate = hazard_rate
        self.model_type = model_type
        self.score_type = score_type
        self.change_points = {}  # Store change points for each security
        self.data_history = {}   # Store data history for each security
        self.timestamps = {}     # Store timestamps for each security
        
    def initialize_security(self, security, initial_data, timestamps):
        """Initialize tracking for a new security with historical data"""
        if len(initial_data) < 10:
            logger.warning(f"Not enough initial data for {security}")
            return False
            
        # Store the initial data
        self.data_history[security] = list(initial_data)
        self.timestamps[security] = list(timestamps)
        
        # Initialize change points list
        self.change_points[security] = []
        
        logger.info(f"Initialized change point detection for {security} with {len(initial_data)} data points")
        return True
    
    def update(self, security, new_point, timestamp):
        """
        Process a new data point and check for change points
        
        Parameters:
        security (str): Security identifier
        new_point (float): New data point to analyze
        timestamp (datetime): Timestamp of the new data point
        
        Returns:
        bool: True if a change point was detected, False otherwise
        tuple: (index of change point, probability) if detected, (None, None) otherwise
        """
        if security not in self.data_history:
            logger.warning(f"Security {security} not initialized")
            return False, (None, None)
            
        # Add the new point to history
        self.data_history[security].append(new_point)
        self.timestamps[security].append(timestamp)
        
        # Trim history to window size
        if len(self.data_history[security]) > self.window_size:
            self.data_history[security] = self.data_history[security][-self.window_size:]
            self.timestamps[security] = self.timestamps[security][-self.window_size:]
        
        # Check if we have enough data
        if len(self.data_history[security]) < 30:
            logger.warning(f"Not enough data for {security}")
            return False, (None, None)
            
        # Set up SD-BOCPD model
        data = self.data_history[security]
        T = len(data)
        
        bocpd_model = SDBocpd(T=T, d=self.score_type, q=self.model_type)
        hazard_model = Hazard(T=T, hazard=self.hazard_rate)
        
        # Initialize the probability model
        mean0 = np.mean(data[:10])
        var0 = np.var(data[:10])
        if var0 == 0:  # Avoid division by zero
            var0 = 0.001
        varx = var0
        init_cor = 0.3 if self.model_type > 0 else 0
        init_theta = [0, 0.1, 0.9, 0.2, 1]
        
        prob_model = GaussianModel(
            mean0=mean0,
            var0=var0,
            varx=varx,
            init_theta=init_theta,
            q=self.model_type,
            init_cor=init_cor
        )
        
        # Run change point detection
        # Disable plotting for real-time usage
        run_length_matrix, cp_list = bocpd_model.bocpd(
            data=data,
            model=prob_model,
            hazard=hazard_model,
            true_cps=[],
            plot=False
        )
        
        # Check if the most recent change point is near the end
        if len(cp_list) > 1:  # Skip the initial point
            most_recent_cp = max(cp_list[1:])  # Skip the initial change point
            end_offset = T - most_recent_cp
            change_detected = (end_offset <= 3)  # Change detected in last 3 points
            
            if change_detected:
                # Compute the actual index in our data array
                data_idx = len(data) - end_offset - 1
                if data_idx >= 0 and data_idx < len(self.data_history[security]):
                    # Get the change point probability from the run length matrix
                    # The probability is the value at run_length=0 for the given time point
                    cp_prob = run_length_matrix[most_recent_cp, 0]
                    
                    # Store the change point
                    self.change_points[security].append(data_idx)
                    logger.info(f"Change point detected for {security} at index {data_idx}, probability: {cp_prob:.4f}")
                    
                    return True, (data_idx, cp_prob)
        
        return False, (None, None)


class DataPoint:
    """Class to store data points for visualization"""
    def __init__(self, timestamp, open_price, high_price, low_price, close_price, volume, is_change_point=False, cp_probability=None):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
        self.volume = volume
        self.is_change_point = is_change_point
        self.cp_probability = cp_probability


class StockData:
    """Class to store stock data for a single instrument"""
    def __init__(self, symbol, name=None):
        self.symbol = symbol
        self.name = name if name else symbol
        self.data_points = []
        self.lock = Lock()
    
    def add_point(self, data_point):
        """Add a new data point"""
        with self.lock:
            self.data_points.append(data_point)
            
            # Keep only the most recent points for efficient display
            if len(self.data_points) > 200:
                self.data_points = self.data_points[-200:]
    
    def get_data_arrays(self):
        """Get data arrays for plotting"""
        with self.lock:
            timestamps = [dp.timestamp for dp in self.data_points]
            opens = [dp.open for dp in self.data_points]
            highs = [dp.high for dp in self.data_points]
            lows = [dp.low for dp in self.data_points]
            closes = [dp.close for dp in self.data_points]
            volumes = [dp.volume for dp in self.data_points]
            
            # Find change points
            cp_indices = [i for i, dp in enumerate(self.data_points) if dp.is_change_point]
            cp_times = [self.data_points[i].timestamp for i in cp_indices]
            cp_prices = [self.data_points[i].close for i in cp_indices]
            cp_probs = [self.data_points[i].cp_probability for i in cp_indices]
            
            return {
                'timestamps': timestamps,
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'volumes': volumes,
                'cp_times': cp_times,
                'cp_prices': cp_prices,
                'cp_probs': cp_probs
            }


class ChartApp:
    """GUI application for displaying stock charts with change points"""
    def __init__(self, root, instruments):
        self.root = root
        self.root.title("Bloomberg Change Point Detection")
        self.root.geometry("1400x800")
        
        self.instruments = instruments
        self.stock_data = {symbol: StockData(symbol) for symbol in instruments}
        self.current_symbol = instruments[0] if instruments else None
        
        self.bloomberg = BloombergDataFetcher()
        self.detector = ChangePointDetector(window_size=120, hazard_rate=0.01, model_type=1, score_type=0.5)
        
        self.data_queue = queue.Queue()
        self.running = False
        self.threads = []
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)
        
        # Left panel for instrument selection and controls
        left_panel = ttk.Frame(main_frame, width=200)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)
        
        # Right panel for charts
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Instrument selector
        ttk.Label(left_panel, text="Instruments:").pack(anchor="w", pady=(0, 5))
        
        self.instrument_listbox = tk.Listbox(left_panel, height=15)
        self.instrument_listbox.pack(fill="x")
        for symbol in self.instruments:
            self.instrument_listbox.insert(tk.END, symbol)
        self.instrument_listbox.bind('<<ListboxSelect>>', self._on_instrument_select)
        
        # Control buttons
        ttk.Button(left_panel, text="Start Monitoring", command=self.start_monitoring).pack(fill="x", pady=(20, 5))
        ttk.Button(left_panel, text="Stop Monitoring", command=self.stop_monitoring).pack(fill="x", pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left_panel, textvariable=self.status_var).pack(pady=10)
        
        # Change point info
        ttk.Label(left_panel, text="Recent Change Points:").pack(anchor="w", pady=(20, 5))
        self.cp_text = tk.Text(left_panel, height=15, width=30)
        self.cp_text.pack(fill="x")
        
        # Chart frame
        self.chart_frame = ttk.Frame(right_panel)
        self.chart_frame.pack(fill="both", expand=True)
        
        # Create chart
        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # Price chart
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)  # Volume chart
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.chart_frame)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Initial plot
        self._update_chart()
    
    def _on_instrument_select(self, event):
        """Handle instrument selection"""
        selection = self.instrument_listbox.curselection()
        if selection:
            index = selection[0]
            self.current_symbol = self.instruments[index]
            self._update_chart()
            self._update_cp_text()
    
    def _update_chart(self):
        """Update the chart with current data"""
        if not self.current_symbol:
            return
            
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Get data
        data = self.stock_data[self.current_symbol].get_data_arrays()
        
        if not data['timestamps']:
            self.ax1.set_title(f"No data for {self.current_symbol}")
            self.canvas.draw()
            return
        
        # Plot price chart
        self.ax1.plot(data['timestamps'], data['closes'], 'b-', label='Close Price')
        
        # Highlight change points
        if data['cp_times']:
            self.ax1.scatter(
                data['cp_times'], 
                data['cp_prices'], 
                c='red', 
                s=80, 
                marker='^', 
                label='Change Points'
            )
            
            # Add vertical lines at change points
            for cp_time in data['cp_times']:
                self.ax1.axvline(x=cp_time, color='r', linestyle='--', alpha=0.3)
        
        # Plot volume chart
        self.ax2.bar(data['timestamps'], data['volumes'], color='gray', alpha=0.5)
        
        # Format chart
        self.ax1.set_title(f"{self.current_symbol} - 1-Minute Chart with Change Points")
        self.ax1.set_ylabel('Price')
        self.ax1.grid(True)
        self.ax1.legend()
        
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Volume')
        self.ax2.grid(True)
        
        # Format dates
        self.fig.autofmt_xdate()
        date_format = DateFormatter('%H:%M')
        self.ax1.xaxis.set_major_formatter(date_format)
        
        # Tight layout
        self.fig.tight_layout()
        
        # Draw
        self.canvas.draw()
    
    def _update_cp_text(self):
        """Update the change points text area"""
        if not self.current_symbol:
            return
            
        # Clear text
        self.cp_text.delete(1.0, tk.END)
        
        # Get data
        data = self.stock_data[self.current_symbol].get_data_arrays()
        
        if not data['cp_times']:
            self.cp_text.insert(tk.END, "No change points detected")
            return
        
        # Display recent change points
        self.cp_text.insert(tk.END, f"Change points for {self.current_symbol}:\n\n")
        
        for i, (cp_time, cp_prob) in enumerate(zip(data['cp_times'], data['cp_probs'])):
            time_str = cp_time.strftime('%H:%M:%S')
            prob_str = f"{cp_prob:.4f}" if cp_prob is not None else "N/A"
            self.cp_text.insert(tk.END, f"{i+1}. {time_str} (prob: {prob_str})\n")
    
    def start_monitoring(self):
        """Start monitoring instruments"""
        if self.running:
            return
            
        # Connect to Bloomberg
        if not self.bloomberg.connect():
            self.status_var.set("Failed to connect to Bloomberg")
            return
        
        # Initialize with historical data
        self.status_var.set("Initializing with historical data...")
        self.root.update()
        
        self._initialize_detectors()
        
        # Start data processing thread
        self.running = True
        processor_thread = Thread(target=self._process_data)
        processor_thread.daemon = True
        processor_thread.start()
        self.threads.append(processor_thread)
        
        # Start data collection thread
        collector_thread = Thread(target=self._collect_data)
        collector_thread.daemon = True
        collector_thread.start()
        self.threads.append(collector_thread)
        
        # Start chart update timer
        self.update_timer = self.root.after(1000, self._timer_update_chart)
        
        self.status_var.set(f"Monitoring {len(self.instruments)} instruments")
        logger.info(f"Started monitoring {len(self.instruments)} instruments")
    
    def stop_monitoring(self):
        """Stop monitoring instruments"""
        if not self.running:
            return
            
        self.running = False
        self.bloomberg.disconnect()
        
        # Stop update timer
        if hasattr(self, 'update_timer'):
            self.root.after_cancel(self.update_timer)
        
        # Wait for threads to terminate
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        self.threads = []
        self.status_var.set("Monitoring stopped")
        logger.info("Stopped monitoring")
    
    def _timer_update_chart(self):
        """Timer callback to update chart"""
        if self.running:
            self._update_chart()
            self._update_cp_text()
            self.update_timer = self.root.after(1000, self._timer_update_chart)
    
    def _initialize_detectors(self):
        """Initialize change point detectors with historical data"""
        now = dt.datetime.now()
        # Get data from the last 2 hours
        start_time = now - dt.timedelta(hours=2)
        
        for instrument in self.instruments:
            try:
                # Get historical 1-minute bars
                data = self.bloomberg.get_historical_bars(
                    security=instrument,
                    start_datetime=start_time,
                    end_datetime=now,
                    interval=1
                )
                
                if data is not None and not data.empty:
                    # Add historical data to stock data for display
                    for _, row in data.iterrows():
                        data_point = DataPoint(
                            timestamp=row['time'],
                            open_price=row['open'],
                            high_price=row['high'],
                            low_price=row['low'],
                            close_price=row['close'],
                            volume=row['volume'],
                            is_change_point=False
                        )
                        self.stock_data[instrument].add_point(data_point)
                    
                    # Use close prices for change point detection
                    close_prices = data['close'].values
                    timestamps = data['time'].values
                    self.detector.initialize_security(instrument, close_prices, timestamps)
                else:
                    logger.warning(f"Could not initialize {instrument} - no data")
            except Exception as e:
                logger.error(f"Error initializing {instrument}: {e}")
    
    def _collect_data(self):
        """Thread function to collect data every minute"""
        while self.running:
            # Wait until the start of the next minute
            next_minute = dt.datetime.now().replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
            wait_seconds = (next_minute - dt.datetime.now()).total_seconds()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            
            # Get the current time and the time 1 minute ago
            now = dt.datetime.now()
            one_min_ago = now - dt.timedelta(minutes=1)
            
            # Get data for each instrument
            for instrument in self.instruments:
                try:
                    # Get 1-minute bar
                    data = self.bloomberg.get_historical_bars(
                        security=instrument,
                        start_datetime=one_min_ago,
                        end_datetime=now,
                        interval=1
                    )
                    
                    if data is not None and not data.empty:
                        # Use the most recent row
                        latest = data.iloc[-1]
                        
                        # Put data in queue for processing
                        self.data_queue.put((
                            instrument, 
                            latest['time'], 
                            latest['open'], 
                            latest['high'], 
                            latest['low'], 
                            latest['close'], 
                            latest['volume']
                        ))
                        logger.debug(f"Collected data for {instrument}")
                    else:
                        logger.warning(f"No data collected for {instrument}")
                except Exception as e:
                    logger.error(f"Error collecting data for {instrument}: {e}")
    
    def _process_data(self):
        """Thread function to process data and detect change points"""
        while self.running:
            try:
                # Get data from queue with timeout
                instrument, timestamp, open_price, high_price, low_price, close_price, volume = self.data_queue.get(timeout=1)
                
                # Create data point
                data_point = DataPoint(
                    timestamp=timestamp,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    volume=volume,
                    is_change_point=False
                )
                
                # Update the detector with the new data point
                change_detected, (cp_idx, cp_prob) = self.detector.update(instrument, close_price, timestamp)
                
                # Handle change point detection
                if change_detected and cp_idx is not None:
                    # Mark this point as a change point
                    data_point.is_change_point = True
                    data_point.cp_probability = cp_prob
                    
                    # Log the detection
                    logger.warning(f"ALERT: Change point detected for {instrument} at {timestamp}, price: {close_price}")
                
                # Add data point to stock data
                self.stock_data[instrument].add_point(data_point)
                
                # Mark task as done
                self.data_queue.task_done()
            except queue.Empty:
                # Queue timeout, just continue
                pass
            except Exception as e:
                logger.error(f"Error processing data: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_monitoring()
        self.root.destroy()


def load_instruments(file_path):
    """Load instruments from file"""
    try:
        with open(file_path, 'r') as f:
            instruments = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(instruments)} instruments from {file_path}")
        return instruments
    except Exception as e:
        logger.error(f"Error loading instruments: {e}")
        return []


def main():
    """Main function"""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python bloomberg_cpt_visualization.py <instrument_file>")
        return
        
    instrument_file = sys.argv[1]
    instruments = load_instruments(instrument_file)
    
    if not instruments:
        print("No instruments loaded. Please check your instrument file.")
        return
    
    # Create and start the GUI application
    root = tk.Tk()
    app = ChartApp(root, instruments)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()