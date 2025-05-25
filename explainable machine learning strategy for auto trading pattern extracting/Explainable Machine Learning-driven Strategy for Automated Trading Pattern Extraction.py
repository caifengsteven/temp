import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import datetime as dt
import pytz
import logging
from collections import defaultdict, deque
import time
import random
from typing import Dict, List, Tuple, Optional

# Try to import Bloomberg API, but provide fallback for testing
try:
    import blpapi
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    logging.warning("Bloomberg API not available. Will use simulation mode.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BloombergDataFetcher:
    """
    Class to fetch market data from Bloomberg
    """
    
    def __init__(self, securities: List[str]):
        """
        Initialize Bloomberg data connection
        
        Args:
            securities: List of Bloomberg security identifiers
        """
        self.securities = securities
        self.session = None
        self.refdata_service = None
        self.market_data_service = None
        self.subscriptions = {}
        self.debug_mode = False  # Set to True to see all fields
        
    def start_session(self):
        """
        Start a Bloomberg session and open services
        """
        if not BLOOMBERG_AVAILABLE:
            logger.warning("Bloomberg API not available. Using simulation mode.")
            return True
            
        # Start a session
        session_options = blpapi.SessionOptions()
        session_options.setServerHost("localhost")
        session_options.setServerPort(8194)
        
        logger.info("Connecting to Bloomberg...")
        self.session = blpapi.Session(session_options)
        
        if not self.session.start():
            logger.error("Failed to start Bloomberg session")
            return False
        
        # Open services
        if not self.session.openService("//blp/refdata"):
            logger.error("Failed to open //blp/refdata service")
            return False
        
        if not self.session.openService("//blp/mktdata"):
            logger.error("Failed to open //blp/mktdata service")
            return False
        
        self.refdata_service = self.session.getService("//blp/refdata")
        self.market_data_service = self.session.getService("//blp/mktdata")
        
        logger.info("Bloomberg session started successfully")
        return True
    
    def get_historical_data(self, start_date: dt.datetime, end_date: dt.datetime, fields: List[str]) -> pd.DataFrame:
        """
        Fetch historical data from Bloomberg
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            fields: List of Bloomberg fields to retrieve
            
        Returns:
            DataFrame containing historical data
        """
        if not BLOOMBERG_AVAILABLE:
            # Return simulated data
            return self._simulate_historical_data(start_date, end_date)
            
        request = self.refdata_service.createRequest("HistoricalDataRequest")
        
        for security in self.securities:
            request.append("securities", security)
        
        for field in fields:
            request.append("fields", field)
            
        request.set("startDate", start_date.strftime("%Y%m%d"))
        request.set("endDate", end_date.strftime("%Y%m%d"))
        request.set("periodicitySelection", "TICK")
        
        logger.info(f"Sending historical data request for {self.securities}")
        self.session.sendRequest(request)
        
        data_points = []
        
        # Process response
        done = False
        while not done:
            ev = self.session.nextEvent(500)
            for msg in ev:
                if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                    security_data = msg.getElement("securityData")
                    security_name = security_data.getElementAsString("security")
                    field_data = security_data.getElement("fieldData")
                    
                    for i in range(field_data.numValues()):
                        field_values = field_data.getValue(i)
                        data_point = {"security": security_name}
                        
                        for field in fields:
                            if field_values.hasElement(field):
                                data_point[field] = field_values.getElementValue(field)
                        
                        time_element = field_values.getElement("date")
                        data_point["timestamp"] = dt.datetime.combine(
                            dt.date(time_element.year(), time_element.month(), time_element.day()),
                            dt.time(time_element.hours(), time_element.minutes(), time_element.seconds())
                        )
                        
                        data_points.append(data_point)
            
            if ev.eventType() == blpapi.Event.RESPONSE:
                done = True
        
        return pd.DataFrame(data_points)
    
    def _simulate_historical_data(self, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
        """
        Generate simulated historical data for testing
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with simulated data
        """
        # Generate a random walk for price
        days = (end_date - start_date).days + 1
        data_points = []
        
        current_price = 4000.0  # Starting price for ES
        timestamp = start_date
        
        while timestamp <= end_date:
            # Generate multiple ticks per day
            num_ticks = random.randint(100, 500)
            
            for _ in range(num_ticks):
                # Random price movement
                price_change = random.normalvariate(0, 0.5)
                current_price += price_change
                
                # Random volume
                volume = max(1, int(random.normalvariate(100, 50)))
                
                # Create data point
                data_point = {
                    "security": self.securities[0],
                    "timestamp": timestamp,
                    "LAST_PRICE": current_price,
                    "VOLUME": volume,
                    "BID": current_price - 0.25,
                    "ASK": current_price + 0.25,
                    "BID_SIZE": max(1, int(random.normalvariate(50, 20))),
                    "ASK_SIZE": max(1, int(random.normalvariate(50, 20)))
                }
                
                data_points.append(data_point)
                
                # Increment tick time
                timestamp += dt.timedelta(seconds=random.randint(1, 5))
            
            # Move to next day
            timestamp = dt.datetime.combine(
                timestamp.date() + dt.timedelta(days=1),
                dt.time(8, 30)  # Start of trading day
            )
        
        return pd.DataFrame(data_points)
    
    def subscribe_market_data(self, fields: List[str]):
        """
        Subscribe to real-time market data
        
        Args:
            fields: List of fields to subscribe to
        """
        if not BLOOMBERG_AVAILABLE:
            logger.info(f"Simulation mode: Subscribed to market data for {self.securities}")
            return
            
        # Create subscription list
        subscriptions = blpapi.SubscriptionList()
        
        for security in self.securities:
            # Correctly add the subscription with correlation ID
            correlation_id = blpapi.CorrelationId(security)
            subscriptions.add(
                security,
                fields,
                "",  # options string
                correlation_id
            )
        
        # Subscribe using the subscription list
        self.session.subscribe(subscriptions)
        logger.info(f"Subscribed to market data for {self.securities}")
    
    def receive_market_data(self, timeout: int = 500) -> Optional[dict]:
        """
        Receive market data updates
        
        Args:
            timeout: Timeout in milliseconds
            
        Returns:
            Dictionary with market data or None if no data received
        """
        if not BLOOMBERG_AVAILABLE:
            # Simulate market data
            return self._simulate_market_data()
            
        event = self.session.nextEvent(timeout)
        
        if event.eventType() == blpapi.Event.SUBSCRIPTION_DATA:
            for msg in event:
                correlation_id = msg.correlationIds()[0].value()
                
                data_point = {"security": correlation_id}
                
                # Get all fields from the message
                element = msg.asElement()
                for i in range(element.numElements()):
                    field = element.getElement(i)
                    field_name = field.name()
                    
                    # Handle different data types safely
                    try:
                        # Get the value directly - most values are already primitive types
                        value = field.getValue()
                        
                        # Store the value directly
                        data_point[field_name] = value
                    except Exception as e:
                        if self.debug_mode:
                            logger.warning(f"Error processing field {field_name}: {e}")
                        # If we couldn't get the value, try using null
                        data_point[field_name] = None
                
                # Make sure we at least have LAST_PRICE, BID, ASK
                required_fields = ['LAST_PRICE', 'BID', 'ASK', 'VOLUME']
                if any(field not in data_point or data_point[field] is None for field in required_fields):
                    # Skip this update if missing critical fields
                    continue
                
                return data_point
        
        return None
    
    def _simulate_market_data(self) -> dict:
        """
        Generate simulated market data for testing
        
        Returns:
            Dictionary with simulated market data
        """
        # Keep track of last price between calls
        if not hasattr(self, '_last_price'):
            self._last_price = 4000.0
        
        # Random price movement
        price_change = random.normalvariate(0, 0.25)
        current_price = self._last_price + price_change
        self._last_price = current_price
        
        # Random volume
        volume = max(1, int(random.normalvariate(50, 20)))
        
        # Create data point
        data_point = {
            "security": self.securities[0],
            "LAST_PRICE": current_price,
            "VOLUME": volume,
            "BID": current_price - 0.25,
            "ASK": current_price + 0.25,
            "BID_SIZE": max(1, int(random.normalvariate(50, 20))),
            "ASK_SIZE": max(1, int(random.normalvariate(50, 20)))
        }
        
        # Simulate network delay
        time.sleep(0.05)
        
        return data_point
    
    def close_session(self):
        """
        Close the Bloomberg session
        """
        if BLOOMBERG_AVAILABLE and self.session:
            self.session.stop()
        logger.info("Bloomberg session closed")


class VolumeProfileBuilder:
    """
    Build volume profiles for tick data
    """
    
    def __init__(self, tick_range: int = 7):
        """
        Initialize the volume profile builder
        
        Args:
            tick_range: Range configuration in ticks
        """
        self.tick_range = tick_range
        self.active_buffers = {}  # Price -> buffer
        self.completed_profiles = []
    
    def process_tick(self, price: float, volume: int, bid_volume: int, ask_volume: int, bid_trades: int, ask_trades: int) -> Optional[dict]:
        """
        Process a new tick
        
        Args:
            price: Price of the tick
            volume: Total volume traded
            bid_volume: Volume traded at bid
            ask_volume: Volume traded at ask 
            bid_trades: Number of trades at bid
            ask_trades: Number of trades at ask
            
        Returns:
            Completed volume profile if one is created, None otherwise
        """
        # Start a new buffer at this price if none exists
        if price not in self.active_buffers:
            self.active_buffers[price] = {
                'start_price': price,
                'min_price': price,
                'max_price': price,
                'volumes': defaultdict(int),  # Price -> volume
                'bid_volumes': defaultdict(int),  # Price -> bid volume
                'ask_volumes': defaultdict(int),  # Price -> ask volume
                'bid_trades': defaultdict(int),  # Price -> bid trades
                'ask_trades': defaultdict(int),  # Price -> ask trades
                'start_time': dt.datetime.now(pytz.UTC)
            }
        
        # Update all active buffers
        completed_profile = None
        
        for start_price, buffer in list(self.active_buffers.items()):
            # Update buffer with the new tick
            buffer['volumes'][price] += volume
            buffer['bid_volumes'][price] += bid_volume
            buffer['ask_volumes'][price] += ask_volume
            buffer['bid_trades'][price] += bid_trades
            buffer['ask_trades'][price] += ask_trades
            
            # Update min/max price
            buffer['min_price'] = min(buffer['min_price'], price)
            buffer['max_price'] = max(buffer['max_price'], price)
            
            # Check if buffer is complete (range is reached)
            price_range = buffer['max_price'] - buffer['min_price']
            if price_range >= self.tick_range:
                # Buffer is complete - check if it's a volume-centered profile
                volume_profile = dict(buffer['volumes'])
                
                # Find price with maximum volume
                if volume_profile:
                    poc_price = max(volume_profile.items(), key=lambda x: x[1])[0]
                    
                    # Check if PoC is in the center
                    price_range = buffer['max_price'] - buffer['min_price']
                    center_distance = abs(poc_price - (buffer['min_price'] + price_range / 2))
                    
                    if center_distance <= price_range * 0.25:  # PoC is near center (within 25% of range)
                        buffer['poc_price'] = poc_price
                        buffer['end_time'] = dt.datetime.now(pytz.UTC)
                        
                        # This is a VCRB - return it
                        completed_profile = buffer.copy()
                        self.completed_profiles.append(completed_profile)
                        logger.info(f"VCRB created with PoC at {poc_price:.2f}, range: {price_range:.2f}")
                
                # Remove the completed buffer
                del self.active_buffers[start_price]
        
        return completed_profile

    def extract_features(self, profile: dict) -> dict:
        """
        Extract features from a volume profile as described in the paper
        
        Args:
            profile: Volume profile data
            
        Returns:
            Dictionary of features
        """
        features = {}
        poc_price = profile['poc_price']
        
        # Pattern features (as per Table 1 in the paper)
        # Calculate upper and lower volumes/trades
        upper_bid_volume = sum(profile['bid_volumes'][p] for p in profile['bid_volumes'] 
                               if p > poc_price and p <= poc_price + 5)
        lower_bid_volume = sum(profile['bid_volumes'][p] for p in profile['bid_volumes'] 
                              if p < poc_price and p >= poc_price - 5)
        
        upper_ask_volume = sum(profile['ask_volumes'][p] for p in profile['ask_volumes'] 
                               if p > poc_price and p <= poc_price + 5)
        lower_ask_volume = sum(profile['ask_volumes'][p] for p in profile['ask_volumes'] 
                              if p < poc_price and p >= poc_price - 5)
        
        upper_bid_trades = sum(profile['bid_trades'][p] for p in profile['bid_trades'] 
                              if p > poc_price and p <= poc_price + 5)
        lower_bid_trades = sum(profile['bid_trades'][p] for p in profile['bid_trades'] 
                              if p < poc_price and p >= poc_price - 5)
        
        upper_ask_trades = sum(profile['ask_trades'][p] for p in profile['ask_trades'] 
                              if p > poc_price and p <= poc_price + 5)
        lower_ask_trades = sum(profile['ask_trades'][p] for p in profile['ask_trades'] 
                              if p < poc_price and p >= poc_price - 5)
        
        # P0: Upper/lower bid volumes ratio
        features['P0'] = upper_bid_volume / max(1, lower_bid_volume)
        
        # P1: Upper/lower ask volumes ratio
        features['P1'] = upper_ask_volume / max(1, lower_ask_volume)
        
        # P2: Upper/lower ask trades ratio
        features['P2'] = upper_ask_trades / max(1, lower_ask_trades)
        
        # P3: Upper/lower bid trades ratio
        features['P3'] = upper_bid_trades / max(1, lower_bid_trades)
        
        # P4: Average upper bid trade size
        features['P4'] = upper_bid_volume / max(1, upper_bid_trades)
        
        # P5: Average upper ask trade size
        features['P5'] = upper_ask_volume / max(1, upper_ask_trades)
        
        # P6: Average lower bid trade size
        features['P6'] = lower_bid_volume / max(1, lower_bid_trades)
        
        # P7: Average lower ask trade size
        features['P7'] = lower_ask_volume / max(1, lower_ask_trades)
        
        # P8: PoC bid volume / upper bid volumes
        features['P8'] = profile['bid_volumes'][poc_price] / max(1, upper_bid_volume)
        
        # P9: PoC ask volume / upper ask volumes
        features['P9'] = profile['ask_volumes'][poc_price] / max(1, upper_ask_volume)
        
        # P10: PoC bid volume / lower bid volumes
        features['P10'] = profile['bid_volumes'][poc_price] / max(1, lower_bid_volume)
        
        # P11: PoC ask volume / lower ask volumes
        features['P11'] = profile['ask_volumes'][poc_price] / max(1, lower_ask_volume)
        
        # P12: Bid/ask volumes ratios for PoC and neighbors
        for t in range(-1, 2):
            price = poc_price + t
            if price in profile['bid_volumes'] and price in profile['ask_volumes']:
                features[f'P12_{t}'] = profile['bid_volumes'][price] / max(1, profile['ask_volumes'][price])
            else:
                features[f'P12_{t}'] = 1.0
        
        # P13: Bid/ask trades ratios for PoC and neighbors
        for t in range(-1, 2):
            price = poc_price + t
            if price in profile['bid_trades'] and price in profile['ask_trades']:
                features[f'P13_{t}'] = profile['bid_trades'][price] / max(1, profile['ask_trades'][price])
            else:
                features[f'P13_{t}'] = 1.0
        
        # P14: Side (above or below)
        last_price = list(profile['volumes'].keys())[-1]
        features['P14'] = 1 if last_price > poc_price else 0
        
        # No Market Shift features as we don't have historical context in this implementation
        # You would need to keep track of the last 237 and 21 ticks to compute these
        
        return features


class TradingStrategy:
    """
    Trading strategy based on VCRB method from the paper
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the trading strategy
        
        Args:
            model_path: Path to a saved CatBoost model, or None to create a new one
        """
        if model_path:
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
        else:
            # Initialize with default parameters (as mentioned in the paper)
            self.model = CatBoostClassifier(
                iterations=1000,
                depth=6,
                learning_rate=0.01,
                loss_function='Logloss',
                random_seed=42,
                verbose=False
            )
        
        self.training_data = []
        self.labels = []
        
        # Initialize with some dummy training data since we don't have any yet
        # This allows us to make predictions from the start
        self._initialize_dummy_data()
    
    def _initialize_dummy_data(self):
        """
        Initialize with dummy data so we can make predictions immediately
        """
        # Create some dummy features
        for _ in range(100):
            features = {
                'P0': random.uniform(0.5, 2.0),
                'P1': random.uniform(0.5, 2.0),
                'P2': random.uniform(0.5, 2.0),
                'P3': random.uniform(0.5, 2.0),
                'P4': random.uniform(20, 100),
                'P5': random.uniform(20, 100),
                'P6': random.uniform(20, 100),
                'P7': random.uniform(20, 100),
                'P8': random.uniform(0.1, 0.5),
                'P9': random.uniform(0.1, 0.5),
                'P10': random.uniform(0.1, 0.5),
                'P11': random.uniform(0.1, 0.5),
                'P12_-1': random.uniform(0.8, 1.2),
                'P12_0': random.uniform(0.8, 1.2),
                'P12_1': random.uniform(0.8, 1.2),
                'P13_-1': random.uniform(0.8, 1.2),
                'P13_0': random.uniform(0.8, 1.2),
                'P13_1': random.uniform(0.8, 1.2),
                'P14': random.choice([0, 1])
            }
            
            self.training_data.append(features)
            self.labels.append(random.choice([0, 1]))
        
        # Train the model on dummy data
        self.train()
    
    def predict(self, features: dict) -> Tuple[float, bool]:
        """
        Make a prediction based on features
        
        Args:
            features: Dictionary of features
            
        Returns:
            Tuple of (probability of reversal, recommendation to enter a trade)
        """
        features_df = pd.DataFrame([features])
        
        # Fill NAs with 0
        features_df = features_df.fillna(0)
        
        # Make prediction
        try:
            proba = self.model.predict_proba(features_df)[0][1]
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            # Return conservative values
            return 0.0, False
        
        # Per paper, we need precision above 24.1% (theoretical) or 33.3% (conservative)
        # Use the conservative threshold
        should_trade = proba > 0.333
        
        return proba, should_trade
    
    def add_training_sample(self, features: dict, label: bool):
        """
        Add a sample to the training data
        
        Args:
            features: Dictionary of features
            label: True for reversal, False for crossing
        """
        self.training_data.append(features)
        self.labels.append(1 if label else 0)
    
    def train(self):
        """
        Train the model on collected data
        """
        if not self.training_data:
            logger.warning("No training data available")
            return
        
        X = pd.DataFrame(self.training_data)
        y = np.array(self.labels)
        
        logger.info(f"Training model on {len(X)} samples (positive: {sum(y)}, negative: {len(y) - sum(y)})")
        
        # Fill NAs with 0
        X = X.fillna(0)
        
        try:
            self.model.fit(X, y)
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def save_model(self, path: str):
        """
        Save the model to a file
        
        Args:
            path: Path to save the model
        """
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")


class VCRBTrader:
    """
    Main class for VCRB-based trading
    """
    
    def __init__(self, security: str, tick_range: int = 7, model_path: Optional[str] = None):
        """
        Initialize the VCRB trader
        
        Args:
            security: Bloomberg security identifier
            tick_range: Range configuration in ticks
            model_path: Path to a saved model or None to create a new one
        """
        self.security = security
        self.bloomberg = BloombergDataFetcher([security])
        self.profile_builder = VolumeProfileBuilder(tick_range=tick_range)
        self.strategy = TradingStrategy(model_path=model_path)
        
        # Trading state
        self.position = 0  # Current position (0 = flat, 1 = long, -1 = short)
        self.entry_price = 0.0  # Entry price for current position
        self.profit_loss = 0.0  # Total P&L
        self.trades = []  # List of completed trades
        
        # Price tracking for labeling
        self.poc_tracking = {}  # POC price -> {tracking data}
        
        # Parameters from the paper
        self.reversal_ticks = 15  # Ticks required for reversal
        self.crossing_ticks = 3   # Ticks required for crossing
        self.take_profit = 15      # Take profit in ticks
        self.stop_loss = 3         # Stop loss in ticks
    
    def start(self):
        """
        Start the trading system
        """
        if not self.bloomberg.start_session():
            logger.error("Failed to start Bloomberg session")
            return
        
        # Subscribe to market data
        self.bloomberg.subscribe_market_data([
            "LAST_PRICE", "VOLUME", "BID", "ASK", "BID_SIZE", "ASK_SIZE"
        ])
        
        last_price = None
        
        try:
            logger.info("Trading system started. Press Ctrl+C to stop.")
            
            while True:
                # Get market data
                data = self.bloomberg.receive_market_data()
                
                if data and "LAST_PRICE" in data:
                    price = data["LAST_PRICE"]
                    volume = data.get("VOLUME", 0)
                    
                    # Determine bid/ask volumes based on price movement
                    bid_volume = 0
                    ask_volume = 0
                    
                    if last_price is not None:
                        if price >= last_price:
                            # Price went up or stayed the same - assume ask volume
                            ask_volume = volume
                        else:
                            # Price went down - assume bid volume
                            bid_volume = volume
                    else:
                        # First tick - assume equal distribution
                        bid_volume = volume / 2
                        ask_volume = volume / 2
                    
                    # Process the tick
                    profile = self.profile_builder.process_tick(
                        price=price,
                        volume=volume,
                        bid_volume=bid_volume,
                        ask_volume=ask_volume,
                        bid_trades=1 if bid_volume > 0 else 0,
                        ask_trades=1 if ask_volume > 0 else 0
                    )
                    
                    # If a complete profile was created
                    if profile:
                        # Extract features
                        features = self.profile_builder.extract_features(profile)
                        
                        # Make prediction
                        proba, should_trade = self.strategy.predict(features)
                        logger.info(f"Prediction for POC {profile['poc_price']:.2f}: {proba:.4f}, Trade: {should_trade}")
                        
                        if should_trade:
                            # Determine direction (only trade reversals)
                            # Long if price is below POC, short if above
                            poc_price = profile['poc_price']
                            
                            if price < poc_price:
                                # Go long - expecting a reversal up
                                self._enter_trade(price, 1, poc_price)
                            elif price > poc_price:
                                # Go short - expecting a reversal down
                                self._enter_trade(price, -1, poc_price)
                        
                        # Start tracking this POC for labeling
                        self._start_tracking_poc(profile['poc_price'], price)
                    
                    # Update POC tracking
                    self._update_poc_tracking(price)
                    
                    # Check for position management (stop loss, take profit)
                    self._manage_positions(price)
                    
                    last_price = price
                
                # Every 200 samples, retrain the model
                if len(self.strategy.training_data) % 200 == 0 and len(self.strategy.training_data) > 0:
                    self.strategy.train()
                
                time.sleep(0.05)  # Avoid CPU spinning
                
        except KeyboardInterrupt:
            logger.info("Trading system stopped by user")
        finally:
            self.bloomberg.close_session()
            self._print_trading_summary()
    
    def _enter_trade(self, price: float, direction: int, poc_price: float):
        """
        Enter a new trade
        
        Args:
            price: Current price
            direction: 1 for long, -1 for short
            poc_price: Point of Control price
        """
        if self.position != 0:
            logger.info(f"Already in position, not entering new trade")
            return
        
        self.position = direction
        self.entry_price = price
        
        direction_text = "LONG" if direction == 1 else "SHORT"
        logger.info(f"ENTER {direction_text} at {price:.2f}, POC: {poc_price:.2f}")
    
    def _exit_trade(self, price: float, reason: str):
        """
        Exit the current trade
        
        Args:
            price: Current price
            reason: Reason for exiting
        """
        if self.position == 0:
            return
        
        # Calculate P&L
        pnl = (price - self.entry_price) * self.position
        self.profit_loss += pnl
        
        # Record the trade
        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': price,
            'direction': self.position,
            'pnl': pnl,
            'reason': reason
        })
        
        direction_text = "LONG" if self.position == 1 else "SHORT"
        logger.info(f"EXIT {direction_text} at {price:.2f}, P&L: {pnl:.2f}, Reason: {reason}")
        
        self.position = 0
        self.entry_price = 0
    
    def _manage_positions(self, current_price: float):
        """
        Manage existing positions (check for take profit or stop loss)
        
        Args:
            current_price: Current market price
        """
        if self.position == 0:
            return
        
        # Calculate unrealized P&L in ticks
        pnl_ticks = (current_price - self.entry_price) * self.position
        
        # Check for take profit
        if pnl_ticks >= self.take_profit:
            self._exit_trade(current_price, "Take Profit")
        
        # Check for stop loss
        elif pnl_ticks <= -self.stop_loss:
            self._exit_trade(current_price, "Stop Loss")
    
    def _start_tracking_poc(self, poc_price: float, current_price: float):
        """
        Start tracking a POC for labeling
        
        Args:
            poc_price: Point of Control price
            current_price: Current market price
        """
        self.poc_tracking[poc_price] = {
            'start_price': current_price,
            'max_distance': 0,
            'crossed': False,
            'reversed': False,
            'label_determined': False
        }
    
    def _update_poc_tracking(self, current_price: float):
        """
        Update tracking for all POCs
        
        Args:
            current_price: Current market price
        """
        for poc_price, tracking in list(self.poc_tracking.items()):
            if tracking['label_determined']:
                continue
            
            # Calculate distance to POC
            distance = abs(current_price - poc_price)
            
            # Update max distance
            tracking['max_distance'] = max(tracking['max_distance'], distance)
            
            # Check if we've hit the POC
            if (tracking['start_price'] < poc_price and current_price >= poc_price) or \
               (tracking['start_price'] > poc_price and current_price <= poc_price):
                
                # We've hit the POC, now track what happens next
                if not tracking['crossed'] and not tracking['reversed']:
                    # Record that we've hit the POC
                    tracking['hit_price'] = current_price
                
                # Check for crossing (3 ticks beyond POC)
                beyond_distance = abs(current_price - poc_price)
                if beyond_distance >= self.crossing_ticks:
                    tracking['crossed'] = True
                    tracking['label_determined'] = True
                    
                    # Add negative sample (crossing = 0)
                    features = self._get_features_for_poc(poc_price)
                    if features:
                        self.strategy.add_training_sample(features, False)
                        logger.info(f"Added CROSSING sample for POC {poc_price:.2f}")
            
            # Check for reversal (15 ticks away from POC)
            elif tracking['max_distance'] >= self.reversal_ticks:
                tracking['reversed'] = True
                tracking['label_determined'] = True
                
                # Add positive sample (reversal = 1)
                features = self._get_features_for_poc(poc_price)
                if features:
                    self.strategy.add_training_sample(features, True)
                    logger.info(f"Added REVERSAL sample for POC {poc_price:.2f}")
            
            # Remove completed tracking
            if tracking['label_determined']:
                del self.poc_tracking[poc_price]
    
    def _get_features_for_poc(self, poc_price: float) -> Optional[dict]:
        """
        Find the features for a specific POC price
        
        Args:
            poc_price: Point of Control price
            
        Returns:
            Dictionary of features or None if not found
        """
        # Find the profile with this POC
        for profile in self.profile_builder.completed_profiles:
            if profile['poc_price'] == poc_price:
                return self.profile_builder.extract_features(profile)
        
        return None
    
    def _print_trading_summary(self):
        """
        Print a summary of trading performance
        """
        num_trades = len(self.trades)
        if num_trades == 0:
            logger.info("No trades executed")
            return
        
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        total_pnl = self.profit_loss
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        logger.info("\n=== Trading Summary ===")
        logger.info(f"Total P&L: {total_pnl:.2f}")
        logger.info(f"Number of Trades: {num_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Average Win: {avg_win:.2f}")
        logger.info(f"Average Loss: {avg_loss:.2f}")
        logger.info("=====================")


# Main execution
if __name__ == "__main__":
    # Choose one of these futures contracts (per the paper)
    # ES = S&P E-mini Futures, B6 = British Pound Futures
    security = "ESM5 Index"  # Sample ticker for S&P E-mini Futures (June 2025)
    
    # Initialize and start the trading system
    trader = VCRBTrader(security=security, tick_range=7)  # Range 7 had best results in the paper
    trader.start()