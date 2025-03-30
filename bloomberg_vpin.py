import sys
import time
import threading
import queue
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import blpapi
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, 
                             QTabWidget, QFileDialog, QTableWidget, QTableWidgetItem,
                             QSplitter, QMessageBox, QTextEdit, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

"""

Key Features and Implementation Details

1. Bloomberg API Connection
The program uses the blpapi library to connect to Bloomberg and fetch real-time trade data:

Connects to the local Bloomberg terminal (localhost:8194)
Uses the Intraday Tick Request to get trade-by-trade data
Handles Bloomberg API session management and error handling

2. Instrument Management

Reads a list of instruments from a text or CSV file
Creates a separate tab with chart for each instrument
Tracks VPIN calculation for each instrument independently

3. VPIN Calculation

Implements volume-synchronized bucket analysis
Classifies trades as buy or sell initiated using the tick rule
Calculates VPIN as the average absolute difference between consecutive buckets
Configurable bucket size and number of buckets

4. Real-time Visualization

Uses matplotlib integrated with PyQt5 for interactive charts
Shows VPIN values over time with a configurable threshold line
Auto-scales to show recent data points
Updates every minute with new VPIN calculations

5. Alert System
Monitors VPIN values against a user-defined threshold
Generates pop-up alerts when VPIN exceeds the threshold
Logs all alerts with timestamps in the application log

6. User Interface

Clean, tab-based interface with a chart for each instrument
Configuration panel for Bloomberg connection, securities file, and VPIN parameters
Start/stop controls for monitoring
Log area for tracking application status and VPIN values
How to Use the Program
Start the application and connect to Bloomberg using the "Connect" button
Load securities by selecting a text file with Bloomberg tickers (one per line)
Configure VPIN parameters:
Bucket Size: Volume size for each VPIN bucket (default: 100,000)
Num Buckets: Number of buckets to use in VPIN calculation (default: 50)
VPIN Threshold: Alert threshold level (default: 0.5)
Start monitoring with the "Start Monitoring" button
View VPIN charts in the tabs, one for each security
Receive alerts when VPIN exceeds the threshold
Notes and Limitations
This implementation uses the tick rule for trade classification (buy/sell). In practice, you might want to use more sophisticated methods if available through Bloomberg.
The application fetches data every minute. For high-frequency trading instruments, you might want to adjust this interval.
VPIN calculation depends on having sufficient trading volume. For less liquid securities, you might need to adjust the bucket size.
You need a valid Bloomberg terminal subscription with API access to run this program.
For production use, you might want to add features like:
Saving VPIN history to a database
Email or SMS alerts
More advanced trade classification methods
Statistical analysis of VPIN values and threshold crossings

"""

class BloombergDataFetcher:
    """Class to handle Bloomberg API connection and data retrieval"""
    
    def __init__(self):
        self.session = None
        self.refDataService = None
        self.is_connected = False
        
    def connect(self):
        """Connect to Bloomberg API"""
        try:
            # Initialize session
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost("localhost")
            sessionOptions.setServerPort(8194)
            
            self.session = blpapi.Session(sessionOptions)
            if not self.session.start():
                print("Failed to start session.")
                return False
            
            if not self.session.openService("//blp/refdata"):
                print("Failed to open //blp/refdata service")
                return False
            
            self.refDataService = self.session.getService("//blp/refdata")
            self.is_connected = True
            print("Connected to Bloomberg API")
            return True
            
        except Exception as e:
            print(f"Error connecting to Bloomberg: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Bloomberg API"""
        if self.session:
            self.session.stop()
            self.is_connected = False
            print("Disconnected from Bloomberg API")
    
    def get_intraday_tick_data(self, security, start_time, end_time):
        """
        Retrieve intraday tick data for a security within the specified time range
        
        Parameters:
        security (str): Bloomberg security identifier
        start_time (datetime): Start of the time range
        end_time (datetime): End of the time range
        
        Returns:
        DataFrame: Tick data with columns for time, price, size, and side
        """
        if not self.is_connected:
            print("Not connected to Bloomberg API")
            return None
        
        try:
            # Create request
            request = self.refDataService.createRequest("IntradayTickRequest")
            request.set("security", security)
            request.set("eventType", "TRADE")  # Only interested in trades
            request.set("startDateTime", start_time)
            request.set("endDateTime", end_time)
            request.set("includeConditionCodes", True)  # To help identify trade side
            
            print(f"Requesting tick data for {security} from {start_time} to {end_time}")
            
            # Send request
            self.session.sendRequest(request)
            
            # Process response
            data = []
            while True:
                event = self.session.nextEvent(500)  # Timeout in milliseconds
                
                for msg in event:
                    if msg.messageType() == blpapi.Name("IntradayTickResponse"):
                        for tickData in msg.getElement("tickData").getElement("tickData"):
                            time_value = tickData.getElementAsDatetime("time")
                            price = tickData.getElementAsFloat("value")
                            size = tickData.getElementAsInteger("size")
                            
                            # Try to determine trade side (buy/sell)
                            # Note: This is a simplification. In practice, you'd use condition codes
                            # or other methods to determine the trade side more accurately
                            side = None
                            if tickData.hasElement("type"):
                                type_value = tickData.getElementAsString("type")
                                if type_value == "TRADE_NORMAL":
                                    side = "buy"  # Default to buy for normal trades
                                    
                                    # Check condition codes for sell indicator
                                    if tickData.hasElement("conditionCodes"):
                                        codes = tickData.getElementAsString("conditionCodes")
                                        if "S" in codes or "Z" in codes:  # Common sell indicators
                                            side = "sell"
                            
                            data.append((time_value, price, size, side))
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            if not data:
                return pd.DataFrame(columns=["time", "price", "size", "side"])
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=["time", "price", "size", "side"])
            
            # Fill missing sides using tick rule
            last_price = None
            for i, row in df.iterrows():
                if row["side"] is None:
                    if last_price is not None:
                        if row["price"] > last_price:
                            df.at[i, "side"] = "buy"
                        elif row["price"] < last_price:
                            df.at[i, "side"] = "sell"
                        else:
                            # If price unchanged, keep previous side or default to buy
                            if i > 0:
                                df.at[i, "side"] = df.at[i-1, "side"]
                            else:
                                df.at[i, "side"] = "buy"
                    else:
                        df.at[i, "side"] = "buy"  # Default for first trade
                
                last_price = row["price"]
            
            return df
            
        except Exception as e:
            print(f"Error fetching intraday tick data: {e}")
            return None


class VPINCalculator:
    """Class to calculate VPIN from tick data"""
    
    def __init__(self, bucket_size=100000, num_buckets=50):
        """
        Initialize VPIN calculator
        
        Parameters:
        bucket_size (int): Volume size for each bucket
        num_buckets (int): Number of buckets to use for VPIN calculation
        """
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets
        self.buckets = []
        self.current_bucket_buy_volume = 0
        self.current_bucket_total_volume = 0
        
    def reset(self, bucket_size=None, num_buckets=None):
        """Reset VPIN calculator and optionally change parameters"""
        if bucket_size is not None:
            self.bucket_size = bucket_size
        if num_buckets is not None:
            self.num_buckets = num_buckets
        
        self.buckets = []
        self.current_bucket_buy_volume = 0
        self.current_bucket_total_volume = 0
    
    def add_trades(self, tick_df):
        """
        Process trade ticks and update VPIN calculation
        
        Parameters:
        tick_df (DataFrame): Tick data with columns for time, price, size, and side
        
        Returns:
        float: Current VPIN value if enough buckets, otherwise None
        """
        if tick_df is None or tick_df.empty:
            return None
        
        vpin = None
        
        for _, row in tick_df.iterrows():
            size = row["size"]
            is_buy = row["side"] == "buy"
            
            # Add to current bucket
            if is_buy:
                self.current_bucket_buy_volume += size
            self.current_bucket_total_volume += size
            
            # Check if bucket is full
            if self.current_bucket_total_volume >= self.bucket_size:
                # Calculate buy volume fraction
                buy_fraction = self.current_bucket_buy_volume / self.current_bucket_total_volume if self.current_bucket_total_volume > 0 else 0.5
                
                # Add to buckets
                self.buckets.append(buy_fraction)
                
                # Keep only necessary number of buckets
                while len(self.buckets) > self.num_buckets:
                    self.buckets.pop(0)
                
                # Reset current bucket
                self.current_bucket_buy_volume = 0
                self.current_bucket_total_volume = 0
                
                # Calculate VPIN if we have enough buckets
                if len(self.buckets) == self.num_buckets:
                    vpin = self.calculate_vpin()
        
        return vpin
    
    def calculate_vpin(self):
        """
        Calculate VPIN based on current buckets
        
        Returns:
        float: VPIN value
        """
        if len(self.buckets) < 2:
            return 0.0
        
        # Calculate sum of absolute differences between consecutive buckets
        abs_diff_sum = sum(abs(self.buckets[i] - self.buckets[i+1]) for i in range(len(self.buckets)-1))
        
        # VPIN is the average of absolute differences
        vpin = abs_diff_sum / (len(self.buckets) - 1)
        
        return vpin


class VPINSignals(QObject):
    """Signal handler for VPIN alerts"""
    vpin_alert = pyqtSignal(str, float)  # Security, VPIN value


class VPINChart(FigureCanvas):
    """Custom matplotlib chart for VPIN visualization"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(VPINChart, self).__init__(fig)
        
        self.setParent(parent)
        
        self.x_data = []  # Time data
        self.y_data = []  # VPIN data
        self.threshold_value = 0.5
        
        # Configure chart
        self.axes.set_title('VPIN Over Time')
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('VPIN')
        self.axes.grid(True)
        
        self.threshold_line = self.axes.axhline(y=self.threshold_value, color='r', linestyle='--', label='Threshold')
        self.vpin_line, = self.axes.plot([], [], 'b-', label='VPIN')
        
        self.axes.legend()
        fig.tight_layout()
    
    def update_data(self, timestamp, vpin_value):
        """Add a new data point to the chart"""
        self.x_data.append(timestamp)
        self.y_data.append(vpin_value)
        
        # Keep only the last 60 points for better visualization
        if len(self.x_data) > 60:
            self.x_data.pop(0)
            self.y_data.pop(0)
        
        self.vpin_line.set_data(self.x_data, self.y_data)
        
        # Update x and y limits
        if self.x_data:
            self.axes.set_xlim(min(self.x_data), max(self.x_data))
            
            # Set y limits with some padding
            y_min = max(0, min(min(self.y_data) - 0.1, self.threshold_value - 0.1))
            y_max = max(max(self.y_data) + 0.1, self.threshold_value + 0.1, 1.0)
            self.axes.set_ylim(y_min, y_max)
        
        self.draw()
    
    def set_threshold(self, threshold):
        """Update the threshold line"""
        self.threshold_value = threshold
        self.threshold_line.set_ydata([threshold, threshold])
        self.draw()


class VPINMonitorApp(QMainWindow):
    """Main application for VPIN monitoring"""
    
    def __init__(self):
        super().__init__()
        
        self.bloomberg = BloombergDataFetcher()
        self.vpin_calculators = {}  # Dictionary of security -> VPINCalculator
        self.vpin_values = {}  # Dictionary of security -> latest VPIN value
        self.vpin_signals = VPINSignals()
        self.vpin_signals.vpin_alert.connect(self.handle_vpin_alert)
        
        self.securities = []
        self.update_interval = 60  # Default update interval in seconds
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_vpin_data)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Bloomberg VPIN Monitor')
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Control panel
        control_panel = QWidget()
        control_layout = QGridLayout(control_panel)
        
        # Bloomberg connection
        control_layout.addWidget(QLabel("Bloomberg API:"), 0, 0)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_to_bloomberg)
        control_layout.addWidget(self.connect_btn, 0, 1)
        
        self.status_label = QLabel("Disconnected")
        control_layout.addWidget(self.status_label, 0, 2)
        
        # Security list
        control_layout.addWidget(QLabel("Securities File:"), 1, 0)
        self.file_path = QLineEdit()
        control_layout.addWidget(self.file_path, 1, 1)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_securities_file)
        control_layout.addWidget(self.browse_btn, 1, 2)
        
        self.load_btn = QPushButton("Load Securities")
        self.load_btn.clicked.connect(self.load_securities)
        control_layout.addWidget(self.load_btn, 1, 3)
        
        # VPIN parameters
        control_layout.addWidget(QLabel("Bucket Size:"), 2, 0)
        self.bucket_size_spin = QSpinBox()
        self.bucket_size_spin.setRange(1000, 1000000)
        self.bucket_size_spin.setSingleStep(1000)
        self.bucket_size_spin.setValue(100000)
        control_layout.addWidget(self.bucket_size_spin, 2, 1)
        
        control_layout.addWidget(QLabel("Num Buckets:"), 2, 2)
        self.num_buckets_spin = QSpinBox()
        self.num_buckets_spin.setRange(10, 100)
        self.num_buckets_spin.setValue(50)
        control_layout.addWidget(self.num_buckets_spin, 2, 3)
        
        control_layout.addWidget(QLabel("VPIN Threshold:"), 3, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.valueChanged.connect(self.update_thresholds)
        control_layout.addWidget(self.threshold_spin, 3, 1)
        
        # Control buttons
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn, 3, 2)
        
        self.stop_btn = QPushButton("Stop Monitoring")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn, 3, 3)
        
        # Add control panel to main layout
        main_layout.addWidget(control_panel)
        
        # Tab widget for security charts
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Log area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        main_layout.addWidget(self.log_text)
        
        # Set main widget
        self.setCentralWidget(main_widget)
        
        self.log("VPIN Monitor application started")
    
    def log(self, message):
        """Add a message to the log area"""
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def connect_to_bloomberg(self):
        """Connect to Bloomberg API"""
        if not self.bloomberg.is_connected:
            if self.bloomberg.connect():
                self.status_label.setText("Connected")
                self.connect_btn.setText("Disconnect")
                self.start_btn.setEnabled(True)
                self.log("Connected to Bloomberg API")
            else:
                self.log("Failed to connect to Bloomberg API")
        else:
            self.bloomberg.disconnect()
            self.status_label.setText("Disconnected")
            self.connect_btn.setText("Connect")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.update_timer.stop()
            self.log("Disconnected from Bloomberg API")
    
    def browse_securities_file(self):
        """Open file dialog to select securities list file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Securities List File", "", 
                                                 "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)", 
                                                 options=options)
        if file_name:
            self.file_path.setText(file_name)
    
    def load_securities(self):
        """Load securities from file"""
        file_path = self.file_path.text()
        if not file_path:
            self.log("Please select a securities file")
            return
        
        try:
            with open(file_path, 'r') as f:
                securities = [line.strip() for line in f if line.strip()]
            
            self.securities = securities
            self.log(f"Loaded {len(securities)} securities from file")
            
            # Clear existing tabs
            while self.tabs.count() > 0:
                self.tabs.removeTab(0)
            
            # Create tabs for each security
            for security in securities:
                # Create tab with chart
                tab = QWidget()
                layout = QVBoxLayout(tab)
                
                chart = VPINChart(tab)
                chart.set_threshold(self.threshold_spin.value())
                layout.addWidget(chart)
                
                self.tabs.addTab(tab, security)
                
                # Initialize VPIN calculator for this security
                self.vpin_calculators[security] = VPINCalculator(
                    bucket_size=self.bucket_size_spin.value(),
                    num_buckets=self.num_buckets_spin.value()
                )
                self.vpin_values[security] = 0.0
            
            if self.bloomberg.is_connected:
                self.start_btn.setEnabled(True)
            
        except Exception as e:
            self.log(f"Error loading securities: {e}")
    
    def update_thresholds(self):
        """Update threshold lines on all charts"""
        threshold = self.threshold_spin.value()
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            chart = tab.findChild(VPINChart)
            if chart:
                chart.set_threshold(threshold)
    
    def start_monitoring(self):
        """Start VPIN monitoring"""
        if not self.securities:
            self.log("No securities loaded. Please load securities first.")
            return
        
        # Reset VPIN calculators with current parameters
        bucket_size = self.bucket_size_spin.value()
        num_buckets = self.num_buckets_spin.value()
        
        for security in self.securities:
            self.vpin_calculators[security].reset(bucket_size, num_buckets)
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log("Started VPIN monitoring")
        
        # Start update timer
        self.update_timer.start(self.update_interval * 1000)
        
        # Immediate first update
        QTimer.singleShot(100, self.update_vpin_data)
    
    def stop_monitoring(self):
        """Stop VPIN monitoring"""
        self.update_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log("Stopped VPIN monitoring")
    
    def update_vpin_data(self):
        """Update VPIN data for all securities"""
        if not self.bloomberg.is_connected:
            self.log("Bloomberg API not connected")
            return
        
        now = dt.datetime.now()
        one_minute_ago = now - dt.timedelta(minutes=1)
        
        for security in self.securities:
            try:
                # Get tick data for last minute
                tick_data = self.bloomberg.get_intraday_tick_data(security, one_minute_ago, now)
                
                if tick_data is not None and not tick_data.empty:
                    # Calculate VPIN
                    vpin = self.vpin_calculators[security].add_trades(tick_data)
                    
                    if vpin is not None:
                        self.vpin_values[security] = vpin
                        
                        # Update chart
                        idx = self.securities.index(security)
                        tab = self.tabs.widget(idx)
                        chart = tab.findChild(VPINChart)
                        if chart:
                            chart.update_data(now, vpin)
                        
                        self.log(f"{security} VPIN: {vpin:.4f}")
                        
                        # Check for threshold crossing
                        threshold = self.threshold_spin.value()
                        if vpin >= threshold:
                            self.vpin_signals.vpin_alert.emit(security, vpin)
                else:
                    self.log(f"No tick data for {security}")
                
            except Exception as e:
                self.log(f"Error updating VPIN for {security}: {e}")
    
    def handle_vpin_alert(self, security, vpin):
        """Handle VPIN threshold alert"""
        threshold = self.threshold_spin.value()
        message = f"VPIN ALERT: {security} VPIN value {vpin:.4f} exceeds threshold {threshold:.4f}"
        self.log(message)
        
        # Create alert dialog
        alert = QMessageBox()
        alert.setIcon(QMessageBox.Warning)
        alert.setText("VPIN Alert")
        alert.setInformativeText(message)
        alert.setWindowTitle("VPIN Threshold Alert")
        alert.setStandardButtons(QMessageBox.Ok)
        
        # Show alert non-blocking
        alert.show()
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.update_timer.stop()
        if self.bloomberg.is_connected:
            self.bloomberg.disconnect()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VPINMonitorApp()
    window.show()
    sys.exit(app.exec_())