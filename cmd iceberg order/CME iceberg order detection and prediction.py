import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from enum import Enum
import time
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm
from lifelines import KaplanMeierFitter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# For Bloomberg data access
try:
    import pdblp
    from pdblp import BCon
    HAS_BLOOMBERG = True
    print("Bloomberg API available.")
except ImportError:
    HAS_BLOOMBERG = False
    print("Bloomberg API not available.")

class OrderAction(Enum):
    LIMIT = "Limit"        # New order
    MODIFY = "Modify"      # Update order
    DELETE = "Delete"      # Remove order
    TRADE = "Trade"        # Trade event

class OrderSide(Enum):
    BUY = "B"
    SELL = "S"

class OrderState(Enum):
    UNKNOWN = "unknown"
    START = "start"
    INITIAL_TRADE = "initial_trade"
    FIRST_TRANCHE = "first_tranche"
    FIRST_TRANCHE_TRADED = "first_tranche_traded"
    NEXT_TRANCHE = "next_tranche"
    NEXT_TRANCHE_TRADED = "next_tranche_traded"
    ORDINARY = "ordinary"
    COMPLETE = "complete"
    CANCELLED = "cancelled"

class IcebergState(Enum):
    ACTIVE = "active"
    COMPLETE = "complete"
    CANCELLED = "cancelled"

class Tranche:
    """
    Represents a single tranche of an iceberg order
    """
    def __init__(self, order_id, timestamp, side=None, price=None):
        self.order_id = order_id
        self.timestamp = timestamp
        self.side = side
        self.price = price
        self.actions = []
        self.volume = 0
        self.resting_volume = 0
        self.traded_volume = 0
        self.deleted_volume = 0
        self.parent_iceberg = None
        self.children = []  # For synthetic icebergs

    def add_action(self, action, side, volume, timestamp, price=None):
        """
        Add an action to this tranche
        
        Parameters:
        -----------
        action : str
            The action type (LIMIT, MODIFY, DELETE, TRADE)
        side : str
            The side (B or S)
        volume : int
            The volume of the action
        timestamp : datetime
            The timestamp of the action
        price : float, optional
            The price of the action
        """
        # Store action details
        action_data = {
            'action': action,
            'side': side,
            'volume': volume,
            'timestamp': timestamp,
            'price': price if price is not None else self.price
        }
        self.actions.append(action_data)
        
        # Update side and price if not already set
        if self.side is None:
            self.side = side
        if self.price is None and price is not None:
            self.price = price
        
        # Update volumes based on action type
        if action == OrderAction.LIMIT.value:
            self.resting_volume = volume
        elif action == OrderAction.MODIFY.value:
            self.resting_volume = volume
        elif action == OrderAction.DELETE.value:
            self.deleted_volume = self.resting_volume
            self.resting_volume = 0
        elif action == OrderAction.TRADE.value:
            self.traded_volume += volume
            self.resting_volume -= volume
            if self.resting_volume < 0:
                self.resting_volume = 0

    def get_total_volume(self):
        return self.traded_volume + self.deleted_volume

    def get_last_action(self):
        return self.actions[-1] if self.actions else None

    def __str__(self):
        return (f"Tranche(order_id={self.order_id}, volume={self.get_total_volume()}, "
                f"side={self.side}, price={self.price}, "
                f"resting={self.resting_volume}, traded={self.traded_volume}, deleted={self.deleted_volume})")

class Iceberg:
    """
    Base class for iceberg orders
    """
    def __init__(self, order_id, timestamp, side=None, price=None):
        self.order_id = order_id
        self.timestamp = timestamp
        self.tranches = []
        self.state = IcebergState.ACTIVE
        self.peak_size = None
        self.total_volume = 0
        self.side = side
        self.price = price

    def add_tranche(self, tranche):
        """
        Add a tranche to this iceberg
        """
        tranche.parent_iceberg = self
        self.tranches.append(tranche)
        
        # Update side and price if not already set
        if self.side is None and tranche.side is not None:
            self.side = tranche.side
        if self.price is None and tranche.price is not None:
            self.price = tranche.price
            
        self.update_total_volume()

    def update_total_volume(self):
        """
        Update the total volume of this iceberg
        """
        self.total_volume = sum(t.get_total_volume() for t in self.tranches)

    def get_total_volume(self):
        return self.total_volume

    def get_num_tranches(self):
        return len(self.tranches)

    def determine_peak_size(self):
        """
        Determine the peak size based on the observed tranches
        """
        if not self.tranches:
            return
            
        # Look at the first tranche
        first_tranche = self.tranches[0]
        if not first_tranche.actions:
            return
            
        # For simplicity, use the initial volume as the peak size
        limit_action = next((a for a in first_tranche.actions if a.get('action') == OrderAction.LIMIT.value), None)
        if limit_action:
            self.peak_size = limit_action.get('volume')
        else:
            # If no limit action, use the most common volume
            volumes = [a.get('volume') for t in self.tranches for a in t.actions if a.get('action') != OrderAction.TRADE.value]
            if volumes:
                self.peak_size = max(set(volumes), key=volumes.count)

    def __str__(self):
        return (f"Iceberg(order_id={self.order_id}, state={self.state.name}, peak_size={self.peak_size}, "
                f"total_volume={self.total_volume}, num_tranches={len(self.tranches)})")

class NativeIceberg(Iceberg):
    """
    Represents a native iceberg order managed by the exchange
    """
    def __init__(self, order_id, timestamp, side=None, price=None):
        super().__init__(order_id, timestamp, side, price)
        self.is_native = True

class SyntheticIceberg(Iceberg):
    """
    Represents a synthetic iceberg order managed by an ISV
    """
    def __init__(self, order_id, timestamp, side=None, price=None):
        super().__init__(order_id, timestamp, side, price)
        self.is_native = False
        self.chains = []  # List of possible tranche chains

class NativeIcebergDetector:
    """
    Detector for native iceberg orders managed by the exchange
    """
    def __init__(self):
        self.orders = {}  # Maps order_id to Order objects
        self.current_tranches = {}  # Maps order_id to current Tranche
        self.icebergs = []  # List of detected icebergs
        
    def process_message(self, message):
        """
        Process a single LOB message to detect native icebergs
        """
        try:
            order_id = message.get('order_id')
            affected_id = message.get('affected_id')
            action = message.get('action')
            side = message.get('side')
            price = message.get('price')
            volume = message.get('volume')
            timestamp = message.get('timestamp')
            
            # Process messages affecting existing orders
            if affected_id and affected_id in self.current_tranches:
                resting_tranche = self.current_tranches[affected_id]
                
                # This is likely a trade message
                if action == OrderAction.TRADE.value:
                    # Add the trade to the tranche
                    resting_tranche.add_action(action, side, volume, timestamp, price)
                    
                    # Check if this is potentially an iceberg
                    if volume > resting_tranche.resting_volume:
                        # This is likely an iceberg trade
                        iceberg = next((i for i in self.icebergs if i.order_id == affected_id), None)
                        if iceberg is None:
                            # Create new iceberg
                            iceberg = NativeIceberg(affected_id, timestamp, resting_tranche.side, resting_tranche.price)
                            iceberg.add_tranche(resting_tranche)
                            self.icebergs.append(iceberg)
            
            # Process new or existing order
            if order_id:
                # If this is a new order, create tranche
                if action == OrderAction.LIMIT.value:
                    # Create new tranche if needed
                    if order_id not in self.current_tranches:
                        tranche = Tranche(order_id, timestamp, side, price)
                        self.current_tranches[order_id] = tranche
                    
                    # Add action to tranche
                    tranche = self.current_tranches[order_id]
                    tranche.add_action(action, side, volume, timestamp, price)
                    
                elif action == OrderAction.MODIFY.value:
                    # Update existing tranche
                    if order_id in self.current_tranches:
                        tranche = self.current_tranches[order_id]
                        tranche.add_action(action, side, volume, timestamp, price)
                        
                        # If resting volume was 0 and now > 0, this might be a new iceberg tranche
                        if tranche.resting_volume == 0 and volume > 0:
                            # Check if this order is already associated with an iceberg
                            iceberg = next((i for i in self.icebergs if i.order_id == order_id), None)
                            if iceberg is not None:
                                # Create a new tranche for this iceberg
                                new_tranche = Tranche(order_id, timestamp, side, price)
                                new_tranche.add_action(action, side, volume, timestamp, price)
                                iceberg.add_tranche(new_tranche)
                                self.current_tranches[order_id] = new_tranche
                
                elif action == OrderAction.DELETE.value:
                    # Delete tranche
                    if order_id in self.current_tranches:
                        tranche = self.current_tranches[order_id]
                        tranche.add_action(action, side, volume, timestamp, price)
                        
                        # Check if this order is associated with an iceberg
                        iceberg = next((i for i in self.icebergs if i.order_id == order_id), None)
                        if iceberg is not None:
                            # Update iceberg state
                            if tranche.traded_volume > 0:
                                iceberg.state = IcebergState.COMPLETE
                            else:
                                iceberg.state = IcebergState.CANCELLED
                            
                            # Update total volume
                            iceberg.update_total_volume()
                            
                        # Remove from tracking
                        del self.current_tranches[order_id]
                
                elif action == OrderAction.TRADE.value:
                    # If the order is already tracked, update it
                    if order_id in self.current_tranches:
                        tranche = self.current_tranches[order_id]
                        tranche.add_action(action, side, volume, timestamp, price)
        except Exception as e:
            print(f"Error processing message in NativeIcebergDetector: {e}")
            
    def get_detected_icebergs(self):
        """
        Return the list of detected icebergs
        """
        # Determine peak size for each iceberg
        for iceberg in self.icebergs:
            iceberg.determine_peak_size()
            
        return self.icebergs

class SyntheticIcebergDetector:
    """
    Detector for synthetic iceberg orders managed by ISVs
    """
    def __init__(self, refill_time_threshold=0.3):
        self.refill_time_threshold = refill_time_threshold  # seconds
        self.tranches = {}  # Maps order_id to Tranche
        self.listening_tranches = {}  # Maps (side, price, volume) to list of tranches
        self.icebergs = []  # List of detected icebergs
        
    def process_message(self, message):
        """
        Process a single LOB message to detect synthetic icebergs
        """
        try:
            order_id = message.get('order_id')
            affected_id = message.get('affected_id')
            action = message.get('action')
            side = message.get('side')
            price = message.get('price')
            volume = message.get('volume')
            timestamp = message.get('timestamp')
            
            # Process messages affecting existing orders
            if affected_id and affected_id in self.tranches:
                tranche = self.tranches[affected_id]
                
                # This is likely a trade message
                if action == OrderAction.TRADE.value:
                    # Add the trade to the tranche
                    tranche.add_action(action, side, volume, timestamp, price)
            
            # Process order actions
            if order_id:
                # New limit order - could be a refill for an iceberg
                if action == OrderAction.LIMIT.value:
                    # Create tranche for the new order
                    tranche = Tranche(order_id, timestamp, side, price)
                    tranche.add_action(action, side, volume, timestamp, price)
                    self.tranches[order_id] = tranche
                    
                    # Check if this matches any listening tranches
                    key = (side, price, volume)
                    if key in self.listening_tranches and self.listening_tranches[key]:
                        matching_tranches = []
                        
                        # Find all listening tranches that match within the time threshold
                        for listening_tranche in list(self.listening_tranches[key]):
                            try:
                                last_action = listening_tranche.get_last_action()
                                if last_action:
                                    time_diff = (timestamp - last_action.get('timestamp')).total_seconds()
                                    if time_diff <= self.refill_time_threshold:
                                        matching_tranches.append(listening_tranche)
                            except Exception as e:
                                print(f"Error checking time difference: {e}")
                        
                        if matching_tranches:
                            # Find the most recent matching tranche
                            matching_tranche = max(matching_tranches, key=lambda t: t.get_last_action().get('timestamp'))
                            
                            # Check if this tranche is already part of an iceberg
                            iceberg = next((i for i in self.icebergs if any(t.order_id == matching_tranche.order_id for t in i.tranches)), None)
                            
                            if iceberg:
                                # Add this tranche to the existing iceberg
                                iceberg.add_tranche(tranche)
                            else:
                                # Create a new iceberg with both tranches
                                iceberg = SyntheticIceberg(matching_tranche.order_id, matching_tranche.timestamp, 
                                                           matching_tranche.side, matching_tranche.price)
                                iceberg.add_tranche(matching_tranche)
                                iceberg.add_tranche(tranche)
                                self.icebergs.append(iceberg)
                            
                            # Remove the listening tranches
                            self.listening_tranches[key] = [t for t in self.listening_tranches[key] if t not in matching_tranches]
                    
                elif action == OrderAction.MODIFY.value:
                    # Update existing tranche
                    if order_id in self.tranches:
                        tranche = self.tranches[order_id]
                        tranche.add_action(action, side, volume, timestamp, price)
                
                elif action == OrderAction.DELETE.value:
                    # Delete tranche and start listening for refill
                    if order_id in self.tranches:
                        tranche = self.tranches[order_id]
                        tranche.add_action(action, side, volume, timestamp, price)
                        
                        # Start listening for refill
                        key = (tranche.side, tranche.price, tranche.resting_volume)
                        if key not in self.listening_tranches:
                            self.listening_tranches[key] = []
                        self.listening_tranches[key].append(tranche)
                        
                        # Check if this tranche is part of an iceberg
                        iceberg = next((i for i in self.icebergs if any(t.order_id == order_id for t in i.tranches)), None)
                        if iceberg:
                            # Update iceberg state if this was the last tranche
                            if tranche.traded_volume > 0:
                                # If the tranche was traded, mark as complete (unless we see a refill)
                                iceberg.state = IcebergState.COMPLETE
                            else:
                                # If the tranche was cancelled, mark the iceberg as cancelled
                                iceberg.state = IcebergState.CANCELLED
                
                elif action == OrderAction.TRADE.value:
                    # Update existing tranche
                    if order_id in self.tranches:
                        tranche = self.tranches[order_id]
                        tranche.add_action(action, side, volume, timestamp, price)
        except Exception as e:
            print(f"Error processing message in SyntheticIcebergDetector: {e}")
            
    def get_detected_icebergs(self):
        """
        Return the list of detected icebergs
        """
        # Determine peak size for each iceberg
        for iceberg in self.icebergs:
            iceberg.determine_peak_size()
            
        return self.icebergs

class IcebergPredictor:
    """
    Predicts the total size of iceberg orders based on observed peak size
    """
    def __init__(self, min_tranches=2, min_icebergs=1):
        self.min_tranches = min_tranches  # Minimum number of tranches per iceberg to include in training
        self.min_icebergs = min_icebergs  # Minimum number of icebergs per peak size to include in model
        self.native_models = {}  # Kaplan-Meier models for native icebergs
        self.synthetic_models = {}  # Kaplan-Meier models for synthetic icebergs
        
    def fit(self, icebergs):
        """
        Fit the prediction model based on observed icebergs
        """
        try:
            # Separate native and synthetic icebergs
            native_icebergs = [iceberg for iceberg in icebergs if isinstance(iceberg, NativeIceberg)]
            synthetic_icebergs = [iceberg for iceberg in icebergs if isinstance(iceberg, SyntheticIceberg)]
            
            print(f"Fitting model with {len(native_icebergs)} native and {len(synthetic_icebergs)} synthetic icebergs")
            
            # Process native icebergs
            self._fit_native_icebergs(native_icebergs)
            
            # Process synthetic icebergs
            self._fit_synthetic_icebergs(synthetic_icebergs)
        except Exception as e:
            print(f"Error fitting iceberg predictor: {e}")
        
    def _fit_native_icebergs(self, icebergs):
        """
        Fit the prediction model for native icebergs
        """
        try:
            # Filter icebergs based on minimum number of tranches
            filtered_icebergs = [iceberg for iceberg in icebergs 
                              if len(iceberg.tranches) >= self.min_tranches and iceberg.peak_size is not None]
            
            # Group icebergs by peak size
            peak_to_icebergs = defaultdict(list)
            for iceberg in filtered_icebergs:
                peak_to_icebergs[iceberg.peak_size].append(iceberg)
                
            # For each peak size with sufficient icebergs, fit a Kaplan-Meier estimator
            for peak_size, peak_icebergs in peak_to_icebergs.items():
                if len(peak_icebergs) >= self.min_icebergs:
                    # Create event and censoring data
                    total_volumes = []
                    event_observed = []
                    
                    for iceberg in peak_icebergs:
                        total_volumes.append(iceberg.get_total_volume())
                        event_observed.append(iceberg.state == IcebergState.COMPLETE)
                    
                    # Fit Kaplan-Meier model
                    try:
                        kmf = KaplanMeierFitter()
                        kmf.fit(total_volumes, event_observed)
                        
                        # Store the model
                        self.native_models[peak_size] = kmf
                    except Exception as e:
                        print(f"Error fitting KM model for native iceberg peak size {peak_size}: {e}")
                        continue
        except Exception as e:
            print(f"Error fitting native icebergs: {e}")
                
    def _fit_synthetic_icebergs(self, icebergs):
        """
        Fit the prediction model for synthetic icebergs
        """
        try:
            # Filter icebergs based on minimum number of tranches
            filtered_icebergs = [iceberg for iceberg in icebergs 
                              if len(iceberg.tranches) >= self.min_tranches and iceberg.peak_size is not None]
            
            # Group icebergs by peak size
            peak_to_icebergs = defaultdict(list)
            for iceberg in filtered_icebergs:
                peak_to_icebergs[iceberg.peak_size].append(iceberg)
                
            # For each peak size with sufficient icebergs, fit a Kaplan-Meier estimator
            for peak_size, peak_icebergs in peak_to_icebergs.items():
                if len(peak_icebergs) >= self.min_icebergs:
                    # Create event and censoring data
                    total_volumes = []
                    event_observed = []
                    weights = []
                    
                    for iceberg in peak_icebergs:
                        # For synthetic icebergs, we need to handle multiple chains
                        total_volume = iceberg.get_total_volume()
                        total_volumes.append(total_volume)
                        event_observed.append(iceberg.state == IcebergState.COMPLETE)
                        weights.append(1.0)  # Weight each iceberg equally for simplicity
                    
                    # Fit Kaplan-Meier model
                    try:
                        kmf = KaplanMeierFitter()
                        kmf.fit(total_volumes, event_observed, weights=weights)
                        
                        # Store the model
                        self.synthetic_models[peak_size] = kmf
                    except Exception as e:
                        print(f"Error fitting KM model for synthetic iceberg peak size {peak_size}: {e}")
                        continue
        except Exception as e:
            print(f"Error fitting synthetic icebergs: {e}")
                
    def predict(self, iceberg, prediction_type='mode'):
        """
        Predict the total size of an iceberg order
        """
        try:
            # Check if we have a model for this peak size
            if iceberg.peak_size is None:
                return None
                
            model_dict = self.native_models if isinstance(iceberg, NativeIceberg) else self.synthetic_models
            if iceberg.peak_size not in model_dict:
                return None
                
            # Get the model
            model = model_dict[iceberg.peak_size]
            
            # Get the accumulated volume so far
            accumulated_volume = iceberg.get_total_volume()
            
            # Make prediction based on the requested type
            if prediction_type == 'mean':
                # Calculate conditional mean: E[X | X > v]
                survival = model.survival_function_.iloc[:, 0]
                volumes = model.survival_function_.index
                valid_volumes = volumes[volumes > accumulated_volume]
                
                if len(valid_volumes) == 0:
                    return accumulated_volume
                    
                # Calculate probabilities
                probabilities = []
                for i in range(len(valid_volumes)):
                    if i == 0:
                        probabilities.append(survival[valid_volumes[i]])
                    else:
                        probabilities.append(survival[valid_volumes[i-1]] - survival[valid_volumes[i]])
                
                # Normalize probabilities
                prob_sum = sum(probabilities)
                if prob_sum == 0:
                    return accumulated_volume
                probabilities = [p / prob_sum for p in probabilities]
                
                # Calculate weighted mean
                prediction = sum(v * p for v, p in zip(valid_volumes, probabilities))
                
                return max(int(round(prediction)), accumulated_volume)
                
            elif prediction_type == 'median':
                # Calculate conditional median
                survival = model.survival_function_.iloc[:, 0]
                volumes = model.survival_function_.index
                valid_volumes = volumes[volumes > accumulated_volume]
                
                if len(valid_volumes) == 0:
                    return accumulated_volume
                    
                # Find the median (where survival = 0.5)
                target_survival = 0.5
                for v in valid_volumes:
                    if survival[v] <= target_survival:
                        return max(int(v), accumulated_volume)
                        
                # If no value found, return the largest volume
                return max(int(valid_volumes[-1]), accumulated_volume) if len(valid_volumes) > 0 else accumulated_volume
                
            elif prediction_type == 'mode':
                # Calculate the most likely total volume
                survival = model.survival_function_.iloc[:, 0]
                volumes = model.survival_function_.index
                valid_volumes = volumes[volumes > accumulated_volume]
                
                if len(valid_volumes) == 0:
                    return accumulated_volume
                    
                # Calculate probability density
                probabilities = []
                for i in range(len(valid_volumes)):
                    if i == 0:
                        probabilities.append(survival[valid_volumes[i]])
                    else:
                        probabilities.append(survival[valid_volumes[i-1]] - survival[valid_volumes[i]])
                
                # Find the mode
                if not probabilities:
                    return accumulated_volume
                max_prob_idx = probabilities.index(max(probabilities))
                prediction = valid_volumes[max_prob_idx]
                
                return max(int(prediction), accumulated_volume)
            else:
                raise ValueError(f"Unknown prediction type: {prediction_type}")
        except Exception as e:
            print(f"Error making prediction: {e}")
            return accumulated_volume
            
    def evaluate(self, icebergs, prediction_type='mode'):
        """
        Evaluate the performance of the prediction model
        """
        try:
            # Separate native and synthetic icebergs
            native_icebergs = [iceberg for iceberg in icebergs if isinstance(iceberg, NativeIceberg)]
            synthetic_icebergs = [iceberg for iceberg in icebergs if isinstance(iceberg, SyntheticIceberg)]
            
            # Evaluate both types
            native_metrics = self._evaluate_icebergs(native_icebergs, prediction_type, "native")
            synthetic_metrics = self._evaluate_icebergs(synthetic_icebergs, prediction_type, "synthetic")
            
            # Combine metrics
            native_count = native_metrics.get("num_predictions", 0)
            synthetic_count = synthetic_metrics.get("num_predictions", 0)
            total_count = native_count + synthetic_count
            
            # Compute weighted average for overall metrics
            overall_metrics = {}
            if total_count > 0:
                for key in ["accuracy", "precision", "recall", "f1", "mae"]:
                    overall_metrics[key] = (
                        (native_metrics.get(key, 0) * native_count + 
                         synthetic_metrics.get(key, 0) * synthetic_count) 
                        / total_count
                    )
            else:
                for key in ["accuracy", "precision", "recall", "f1", "mae"]:
                    overall_metrics[key] = 0
                    
            overall_metrics["num_predictions"] = total_count
            
            return {
                "native": native_metrics,
                "synthetic": synthetic_metrics,
                "overall": overall_metrics
            }
        except Exception as e:
            print(f"Error evaluating model: {e}")
            # Return empty metrics
            empty_metrics = {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "mae": 0,
                "confusion_matrix": [[0, 0], [0, 0]],
                "num_predictions": 0
            }
            return {
                "native": empty_metrics,
                "synthetic": empty_metrics,
                "overall": empty_metrics
            }
        
    def _evaluate_icebergs(self, icebergs, prediction_type, iceberg_type):
        """
        Evaluate the performance of the prediction model on a set of icebergs
        """
        try:
            # Initialize evaluation metrics
            y_true = []
            y_pred = []
            residuals = []
            num_predictions = 0
            
            # For each iceberg, evaluate predictions after each tranche
            for iceberg in icebergs:
                # Skip icebergs with unknown peak size
                if iceberg.peak_size is None:
                    continue
                    
                # Skip icebergs with peak sizes not in our model
                model_dict = self.native_models if iceberg_type == "native" else self.synthetic_models
                if iceberg.peak_size not in model_dict:
                    continue
                    
                # Skip icebergs that don't meet minimum tranches requirement
                if len(iceberg.tranches) < self.min_tranches:
                    continue
                    
                # Get the total volume
                total_volume = iceberg.get_total_volume()
                    
                # Evaluate prediction after each tranche (except the last)
                accumulated_volume = 0
                for i in range(len(iceberg.tranches) - 1):
                    # Update accumulated volume
                    tranche = iceberg.tranches[i]
                    accumulated_volume += tranche.get_total_volume()
                    
                    # Make a temporary copy of the iceberg with just the first i+1 tranches
                    temp_iceberg = NativeIceberg(iceberg.order_id, iceberg.timestamp) if iceberg_type == "native" else SyntheticIceberg(iceberg.order_id, iceberg.timestamp)
                    temp_iceberg.peak_size = iceberg.peak_size
                    for j in range(i+1):
                        temp_iceberg.add_tranche(iceberg.tranches[j])
                        
                    # Predict total volume
                    prediction = self.predict(temp_iceberg, prediction_type)
                    if prediction is not None:
                        # Evaluate classification: are there more tranches to come?
                        more_tranches = (i < len(iceberg.tranches) - 2)  # True if not the penultimate tranche
                        has_hidden = (prediction > accumulated_volume)
                        
                        y_true.append(more_tranches)
                        y_pred.append(has_hidden)
                        
                        # Evaluate regression: how close is the prediction to actual total?
                        residuals.append(abs(prediction - total_volume))
                        
                        num_predictions += 1
                        
            # Calculate metrics
            metrics = {
                "num_predictions": num_predictions
            }
            
            # Calculate accuracy
            if len(y_true) > 0:
                metrics["accuracy"] = sum(1 for y, yp in zip(y_true, y_pred) if y == yp) / len(y_true)
            else:
                metrics["accuracy"] = 0
                
            # Calculate precision
            if sum(y_pred) > 0:
                metrics["precision"] = sum(1 for y, yp in zip(y_true, y_pred) if y and yp) / sum(y_pred)
            else:
                metrics["precision"] = 0
                
            # Calculate recall
            if sum(y_true) > 0:
                metrics["recall"] = sum(1 for y, yp in zip(y_true, y_pred) if y and yp) / sum(y_true)
            else:
                metrics["recall"] = 0
                
            # Calculate F1
            if metrics["precision"] + metrics["recall"] > 0:
                metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
            else:
                metrics["f1"] = 0
                
            # Calculate MAE
            if residuals:
                metrics["mae"] = sum(residuals) / len(residuals)
            else:
                metrics["mae"] = 0
                
            # Calculate confusion matrix
            metrics["confusion_matrix"] = [
                [sum(1 for y, yp in zip(y_true, y_pred) if not y and not yp), 
                 sum(1 for y, yp in zip(y_true, y_pred) if not y and yp)],
                [sum(1 for y, yp in zip(y_true, y_pred) if y and not yp), 
                 sum(1 for y, yp in zip(y_true, y_pred) if y and yp)]
            ]
            
            return metrics
        except Exception as e:
            print(f"Error in _evaluate_icebergs for {iceberg_type} icebergs: {e}")
            return {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "mae": 0,
                "confusion_matrix": [[0, 0], [0, 0]],
                "num_predictions": 0
            }

def generate_sample_mbo_data():
    """
    Generate sample Market-By-Order data
    """
    print("Generating sample MBO data...")
    
    # Sample data with native and synthetic icebergs
    start_time = pd.Timestamp('2023-01-01 09:30:00')
    
    messages = []
    
    # Native iceberg example
    # Initial order placement
    messages.append({
        'timestamp': start_time,
        'order_id': '1001',
        'side': 'S',
        'action': OrderAction.LIMIT.value,
        'price': 100.00,
        'volume': 10,
        'affected_id': None
    })
    
    # First trade (partial)
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=5),
        'order_id': '2001',
        'side': 'B',
        'action': OrderAction.TRADE.value,
        'price': 100.00,
        'volume': 8,
        'affected_id': '1001'
    })
    
    # Modify after trade
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=5.01),
        'order_id': '1001',
        'side': 'S',
        'action': OrderAction.MODIFY.value,
        'price': 100.00,
        'volume': 7,
        'affected_id': None
    })
    
    # Second trade
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=10),
        'order_id': '2002',
        'side': 'B',
        'action': OrderAction.TRADE.value,
        'price': 100.00,
        'volume': 3,
        'affected_id': '1001'
    })
    
    # Modify after trade
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=10.01),
        'order_id': '1001',
        'side': 'S',
        'action': OrderAction.MODIFY.value,
        'price': 100.00,
        'volume': 4,
        'affected_id': None
    })
    
    # Third trade
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=15),
        'order_id': '2003',
        'side': 'B',
        'action': OrderAction.TRADE.value,
        'price': 100.00,
        'volume': 4,
        'affected_id': '1001'
    })
    
    # Delete order
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=15.01),
        'order_id': '1001',
        'side': 'S',
        'action': OrderAction.DELETE.value,
        'price': 100.00,
        'volume': 0,
        'affected_id': None
    })
    
    # Synthetic iceberg example
    # First tranche
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=20),
        'order_id': '3001',
        'side': 'B',
        'action': OrderAction.LIMIT.value,
        'price': 99.95,
        'volume': 5,
        'affected_id': None
    })
    
    # Trade first tranche
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=25),
        'order_id': '4001',
        'side': 'S',
        'action': OrderAction.TRADE.value,
        'price': 99.95,
        'volume': 5,
        'affected_id': '3001'
    })
    
    # Delete first tranche
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=25.01),
        'order_id': '3001',
        'side': 'B',
        'action': OrderAction.DELETE.value,
        'price': 99.95,
        'volume': 0,
        'affected_id': None
    })
    
    # Second tranche
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=25.2),  # Within refill threshold
        'order_id': '3002',
        'side': 'B',
        'action': OrderAction.LIMIT.value,
        'price': 99.95,
        'volume': 5,
        'affected_id': None
    })
    
    # Trade second tranche
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=30),
        'order_id': '4002',
        'side': 'S',
        'action': OrderAction.TRADE.value,
        'price': 99.95,
        'volume': 5,
        'affected_id': '3002'
    })
    
    # Delete second tranche
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=30.01),
        'order_id': '3002',
        'side': 'B',
        'action': OrderAction.DELETE.value,
        'price': 99.95,
        'volume': 0,
        'affected_id': None
    })
    
    # Third tranche
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=30.2),  # Within refill threshold
        'order_id': '3003',
        'side': 'B',
        'action': OrderAction.LIMIT.value,
        'price': 99.95,
        'volume': 5,
        'affected_id': None
    })
    
    # Trade third tranche
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=35),
        'order_id': '4003',
        'side': 'S',
        'action': OrderAction.TRADE.value,
        'price': 99.95,
        'volume': 5,
        'affected_id': '3003'
    })
    
    # Delete third tranche
    messages.append({
        'timestamp': start_time + pd.Timedelta(seconds=35.01),
        'order_id': '3003',
        'side': 'B',
        'action': OrderAction.DELETE.value,
        'price': 99.95,
        'volume': 0,
        'affected_id': None
    })
    
    # Generate more examples to have sufficient data for training
    for i in range(10):
        # Native icebergs
        base_time = start_time + pd.Timedelta(minutes=i+1)
        order_id = f'5{i:03d}'
        
        # Initial placement
        messages.append({
            'timestamp': base_time,
            'order_id': order_id,
            'side': 'S' if i % 2 == 0 else 'B',
            'action': OrderAction.LIMIT.value,
            'price': 100.00 - (0.05 * (i % 10)),
            'volume': 10,
            'affected_id': None
        })
        
        # Trades and refreshes
        for j in range(3):  # 3 tranches
            trade_id = f'6{i:03d}{j}'
            trade_time = base_time + pd.Timedelta(seconds=5*(j+1))
            
            messages.append({
                'timestamp': trade_time,
                'order_id': trade_id,
                'side': 'B' if i % 2 == 0 else 'S',
                'action': OrderAction.TRADE.value,
                'price': 100.00 - (0.05 * (i % 10)),
                'volume': 10,
                'affected_id': order_id
            })
            
            if j < 2:  # Not the last tranche
                messages.append({
                    'timestamp': trade_time + pd.Timedelta(milliseconds=10),
                    'order_id': order_id,
                    'side': 'S' if i % 2 == 0 else 'B',
                    'action': OrderAction.MODIFY.value,
                    'price': 100.00 - (0.05 * (i % 10)),
                    'volume': 10,
                    'affected_id': None
                })
            else:  # Last tranche
                messages.append({
                    'timestamp': trade_time + pd.Timedelta(milliseconds=10),
                    'order_id': order_id,
                    'side': 'S' if i % 2 == 0 else 'B',
                    'action': OrderAction.DELETE.value,
                    'price': 100.00 - (0.05 * (i % 10)),
                    'volume': 0,
                    'affected_id': None
                })
    
    # Add synthetic icebergs
    for i in range(10):
        base_time = start_time + pd.Timedelta(minutes=i+2)
        
        # 3 tranches
        for j in range(3):
            order_id = f'7{i:03d}{j}'
            trade_id = f'8{i:03d}{j}'
            tranche_time = base_time + pd.Timedelta(seconds=j*10)
            
            # Place order
            messages.append({
                'timestamp': tranche_time,
                'order_id': order_id,
                'side': 'B' if i % 2 == 0 else 'S',
                'action': OrderAction.LIMIT.value,
                'price': 99.90 + (0.05 * (i % 10)),
                'volume': 5,
                'affected_id': None
            })
            
            # Trade
            messages.append({
                'timestamp': tranche_time + pd.Timedelta(seconds=2),
                'order_id': trade_id,
                'side': 'S' if i % 2 == 0 else 'B',
                'action': OrderAction.TRADE.value,
                'price': 99.90 + (0.05 * (i % 10)),
                'volume': 5,
                'affected_id': order_id
            })
            
            # Delete
            messages.append({
                'timestamp': tranche_time + pd.Timedelta(seconds=2.01),
                'order_id': order_id,
                'side': 'B' if i % 2 == 0 else 'S',
                'action': OrderAction.DELETE.value,
                'price': 99.90 + (0.05 * (i % 10)),
                'volume': 0,
                'affected_id': None
            })
    
    # Sort messages by timestamp
    messages.sort(key=lambda m: m['timestamp'])
    
    return messages

def run_iceberg_detection_and_prediction(messages, training_ratio=0.7):
    """
    Run the full iceberg detection and prediction pipeline
    """
    try:
        # Split messages into training and testing sets
        split_idx = int(len(messages) * training_ratio)
        train_messages = messages[:split_idx]
        test_messages = messages[split_idx:]
        
        print(f"Split {len(messages)} messages into {len(train_messages)} training and {len(test_messages)} testing")
        
        # Detect icebergs in training data
        native_detector = NativeIcebergDetector()
        synthetic_detector = SyntheticIcebergDetector(refill_time_threshold=0.3)
        
        print("Detecting icebergs in training data...")
        for message in tqdm(train_messages):
            native_detector.process_message(message)
            synthetic_detector.process_message(message)
        
        train_native_icebergs = native_detector.get_detected_icebergs()
        train_synthetic_icebergs = synthetic_detector.get_detected_icebergs()
        
        print(f"Detected {len(train_native_icebergs)} native and {len(train_synthetic_icebergs)} synthetic icebergs in training data")
        
        # Fit predictor on training icebergs
        predictor = IcebergPredictor(min_tranches=2, min_icebergs=1)  # Lower min_icebergs for sample data
        predictor.fit(train_native_icebergs + train_synthetic_icebergs)
        
        # Detect icebergs in testing data
        native_detector_test = NativeIcebergDetector()
        synthetic_detector_test = SyntheticIcebergDetector(refill_time_threshold=0.3)
        
        print("Detecting icebergs in testing data...")
        for message in tqdm(test_messages):
            native_detector_test.process_message(message)
            synthetic_detector_test.process_message(message)
        
        test_native_icebergs = native_detector_test.get_detected_icebergs()
        test_synthetic_icebergs = synthetic_detector_test.get_detected_icebergs()
        
        print(f"Detected {len(test_native_icebergs)} native and {len(test_synthetic_icebergs)} synthetic icebergs in testing data")
        
        # Evaluate predictions
        print("Evaluating predictions...")
        evaluation_metrics = predictor.evaluate(test_native_icebergs + test_synthetic_icebergs, prediction_type='mode')
        
        # Compute iceberg statistics
        train_stats = compute_iceberg_statistics(train_native_icebergs, train_synthetic_icebergs)
        test_stats = compute_iceberg_statistics(test_native_icebergs, test_synthetic_icebergs)
        
        # Prepare results
        results = {
            'training': {
                'native_icebergs': train_native_icebergs,
                'synthetic_icebergs': train_synthetic_icebergs,
                'stats': train_stats
            },
            'testing': {
                'native_icebergs': test_native_icebergs,
                'synthetic_icebergs': test_synthetic_icebergs,
                'stats': test_stats
            },
            'evaluation': evaluation_metrics
        }
        
        return results
    except Exception as e:
        print(f"Error in run_iceberg_detection_and_prediction: {e}")
        # Return empty results
        empty_stats = {
            'counts': {
                'native': {'total': 0, 'complete': 0, 'cancelled': 0},
                'synthetic': {'total': 0, 'complete': 0, 'cancelled': 0}
            },
            'volumes': {'native': 0, 'synthetic': 0},
            'distributions': {
                'native': {'tranches': [], 'peak_sizes': [], 'total_sizes': []},
                'synthetic': {'tranches': [], 'peak_sizes': [], 'total_sizes': []}
            }
        }
        empty_metrics = {
            "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "mae": 0,
            "confusion_matrix": [[0, 0], [0, 0]], "num_predictions": 0
        }
        return {
            'training': {
                'native_icebergs': [],
                'synthetic_icebergs': [],
                'stats': empty_stats
            },
            'testing': {
                'native_icebergs': [],
                'synthetic_icebergs': [],
                'stats': empty_stats
            },
            'evaluation': {
                "native": empty_metrics,
                "synthetic": empty_metrics,
                "overall": empty_metrics
            }
        }

def compute_iceberg_statistics(native_icebergs, synthetic_icebergs):
    """
    Compute statistics about the detected icebergs
    """
    try:
        # Count total number of icebergs
        total_native = len(native_icebergs)
        total_synthetic = len(synthetic_icebergs)
        
        # Count completed vs. cancelled icebergs
        native_complete = sum(1 for i in native_icebergs if i.state == IcebergState.COMPLETE)
        native_cancelled = sum(1 for i in native_icebergs if i.state == IcebergState.CANCELLED)
        synthetic_complete = sum(1 for i in synthetic_icebergs if i.state == IcebergState.COMPLETE)
        synthetic_cancelled = sum(1 for i in synthetic_icebergs if i.state == IcebergState.CANCELLED)
        
        # Calculate total volumes
        native_total_volume = sum(i.get_total_volume() for i in native_icebergs)
        synthetic_total_volume = sum(i.get_total_volume() for i in synthetic_icebergs)
        
        # Calculate distribution of number of tranches
        native_tranche_counts = [len(i.tranches) for i in native_icebergs]
        synthetic_tranche_counts = [len(i.tranches) for i in synthetic_icebergs]
        
        # Calculate distribution of peak sizes
        native_peak_sizes = [i.peak_size for i in native_icebergs if i.peak_size is not None]
        synthetic_peak_sizes = [i.peak_size for i in synthetic_icebergs if i.peak_size is not None]
        
        # Calculate distribution of total sizes
        native_total_sizes = [i.get_total_volume() for i in native_icebergs]
        synthetic_total_sizes = [i.get_total_volume() for i in synthetic_icebergs]
        
        # Return statistics
        return {
            'counts': {
                'native': {
                    'total': total_native,
                    'complete': native_complete,
                    'cancelled': native_cancelled
                },
                'synthetic': {
                    'total': total_synthetic,
                    'complete': synthetic_complete,
                    'cancelled': synthetic_cancelled
                }
            },
            'volumes': {
                'native': native_total_volume,
                'synthetic': synthetic_total_volume
            },
            'distributions': {
                'native': {
                    'tranches': native_tranche_counts,
                    'peak_sizes': native_peak_sizes,
                    'total_sizes': native_total_sizes
                },
                'synthetic': {
                    'tranches': synthetic_tranche_counts,
                    'peak_sizes': synthetic_peak_sizes,
                    'total_sizes': synthetic_total_sizes
                }
            }
        }
    except Exception as e:
        print(f"Error computing iceberg statistics: {e}")
        return {
            'counts': {
                'native': {'total': 0, 'complete': 0, 'cancelled': 0},
                'synthetic': {'total': 0, 'complete': 0, 'cancelled': 0}
            },
            'volumes': {'native': 0, 'synthetic': 0},
            'distributions': {
                'native': {'tranches': [], 'peak_sizes': [], 'total_sizes': []},
                'synthetic': {'tranches': [], 'peak_sizes': [], 'total_sizes': []}
            }
        }

def plot_results(results):
    """
    Plot results of the iceberg detection and prediction
    """
    try:
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot iceberg completion status distribution
        native_complete = results['training']['stats']['counts']['native']['complete']
        native_cancelled = results['training']['stats']['counts']['native']['cancelled']
        synthetic_complete = results['training']['stats']['counts']['synthetic']['complete']
        synthetic_cancelled = results['training']['stats']['counts']['synthetic']['cancelled']
        
        axs[0, 0].bar(['Native Complete', 'Native Cancelled', 'Synthetic Complete', 'Synthetic Cancelled'],
                   [native_complete, native_cancelled, synthetic_complete, synthetic_cancelled])
        axs[0, 0].set_title('Iceberg Completion Status Distribution')
        axs[0, 0].set_ylabel('Count')
        for i, v in enumerate([native_complete, native_cancelled, synthetic_complete, synthetic_cancelled]):
            axs[0, 0].text(i, v + 0.1, str(v), ha='center')
        
        # Plot tranches per iceberg
        native_tranches = results['training']['stats']['distributions']['native']['tranches']
        synthetic_tranches = results['training']['stats']['distributions']['synthetic']['tranches']
        
        if native_tranches and synthetic_tranches:
            max_bins = max(5, min(20, max(max(native_tranches, default=0), max(synthetic_tranches, default=0))))
            if max_bins > 0:
                axs[0, 1].hist([native_tranches, synthetic_tranches], bins=max_bins, label=['Native', 'Synthetic'])
                axs[0, 1].set_title('Number of Tranches per Iceberg')
                axs[0, 1].set_xlabel('Number of Tranches')
                axs[0, 1].set_ylabel('Frequency')
                axs[0, 1].legend()
            else:
                axs[0, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                axs[0, 1].set_title('Number of Tranches per Iceberg')
        else:
            axs[0, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axs[0, 1].set_title('Number of Tranches per Iceberg')
        
        # Plot peak size distribution
        native_peaks = results['training']['stats']['distributions']['native']['peak_sizes']
        synthetic_peaks = results['training']['stats']['distributions']['synthetic']['peak_sizes']
        
        if native_peaks and synthetic_peaks:
            max_bins = max(5, min(20, max(max(native_peaks, default=0), max(synthetic_peaks, default=0))))
            if max_bins > 0:
                axs[1, 0].hist([native_peaks, synthetic_peaks], bins=max_bins, label=['Native', 'Synthetic'])
                axs[1, 0].set_title('Peak Size Distribution')
                axs[1, 0].set_xlabel('Peak Size')
                axs[1, 0].set_ylabel('Frequency')
                axs[1, 0].legend()
            else:
                axs[1, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                axs[1, 0].set_title('Peak Size Distribution')
        else:
            axs[1, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axs[1, 0].set_title('Peak Size Distribution')
        
        # Plot total size distribution
        native_totals = results['training']['stats']['distributions']['native']['total_sizes']
        synthetic_totals = results['training']['stats']['distributions']['synthetic']['total_sizes']
        
        if native_totals and synthetic_totals:
            max_bins = max(5, min(20, max(max(native_totals, default=0), max(synthetic_totals, default=0))))
            if max_bins > 0:
                axs[1, 1].hist([native_totals, synthetic_totals], bins=max_bins, label=['Native', 'Synthetic'])
                axs[1, 1].set_title('Total Size Distribution')
                axs[1, 1].set_xlabel('Total Size')
                axs[1, 1].set_ylabel('Frequency')
                axs[1, 1].legend()
            else:
                axs[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                axs[1, 1].set_title('Total Size Distribution')
        else:
            axs[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axs[1, 1].set_title('Total Size Distribution')
        
        # Plot confusion matrix for native icebergs
        if 'native' in results['evaluation'] and results['evaluation']['native'].get('num_predictions', 0) > 0:
            cm = results['evaluation']['native']['confusion_matrix']
            try:
                axs[2, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                axs[2, 0].set_title('Confusion Matrix - Native Icebergs')
                axs[2, 0].set_xlabel('Predicted')
                axs[2, 0].set_ylabel('Actual')
                axs[2, 0].set_xticks([0, 1])
                axs[2, 0].set_yticks([0, 1])
                axs[2, 0].set_xticklabels(['No Hidden', 'Hidden'])
                axs[2, 0].set_yticklabels(['No Hidden', 'Hidden'])
                
                # Show counts in confusion matrix
                thresh = np.max(cm) / 2. if np.max(cm) > 0 else 0
                for i in range(min(2, len(cm))):
                    for j in range(min(2, len(cm[i]))):
                        axs[2, 0].text(j, i, format(cm[i][j], 'd'),
                                    ha="center", va="center",
                                    color="white" if cm[i][j] > thresh else "black")
            except Exception as e:
                print(f"Error plotting native iceberg confusion matrix: {e}")
                axs[2, 0].text(0.5, 0.5, 'Error plotting confusion matrix', ha='center', va='center')
                axs[2, 0].set_title('Confusion Matrix - Native Icebergs')
        else:
            axs[2, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axs[2, 0].set_title('Confusion Matrix - Native Icebergs')
        
        # Plot confusion matrix for synthetic icebergs
        if 'synthetic' in results['evaluation'] and results['evaluation']['synthetic'].get('num_predictions', 0) > 0:
            cm = results['evaluation']['synthetic']['confusion_matrix']
            try:
                axs[2, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                axs[2, 1].set_title('Confusion Matrix - Synthetic Icebergs')
                axs[2, 1].set_xlabel('Predicted')
                axs[2, 1].set_ylabel('Actual')
                axs[2, 1].set_xticks([0, 1])
                axs[2, 1].set_yticks([0, 1])
                axs[2, 1].set_xticklabels(['No Hidden', 'Hidden'])
                axs[2, 1].set_yticklabels(['No Hidden', 'Hidden'])
                
                # Show counts in confusion matrix
                thresh = np.max(cm) / 2. if np.max(cm) > 0 else 0
                for i in range(min(2, len(cm))):
                    for j in range(min(2, len(cm[i]))):
                        axs[2, 1].text(j, i, format(cm[i][j], 'd'),
                                    ha="center", va="center",
                                    color="white" if cm[i][j] > thresh else "black")
            except Exception as e:
                print(f"Error plotting synthetic iceberg confusion matrix: {e}")
                axs[2, 1].text(0.5, 0.5, 'Error plotting confusion matrix', ha='center', va='center')
                axs[2, 1].set_title('Confusion Matrix - Synthetic Icebergs')
        else:
            axs[2, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axs[2, 1].set_title('Confusion Matrix - Synthetic Icebergs')
        
        plt.tight_layout()
        plt.savefig('iceberg_detection_results.png')
        print("Saved results plot to 'iceberg_detection_results.png'")
        plt.show()
        
        # Print evaluation metrics
        print("\nEvaluation Metrics:")
        if 'overall' in results['evaluation']:
            print(f"Overall Accuracy: {results['evaluation']['overall']['accuracy']:.4f}")
            print(f"Overall Precision: {results['evaluation']['overall']['precision']:.4f}")
            print(f"Overall Recall: {results['evaluation']['overall']['recall']:.4f}")
            print(f"Overall F1 Score: {results['evaluation']['overall']['f1']:.4f}")
            print(f"Overall MAE: {results['evaluation']['overall']['mae']:.4f}")
            print(f"Number of Predictions: {results['evaluation']['overall']['num_predictions']}")
        
        if 'native' in results['evaluation'] and results['evaluation']['native'].get('num_predictions', 0) > 0:
            print("\nNative Icebergs:")
            print(f"  Accuracy: {results['evaluation']['native']['accuracy']:.4f}")
            print(f"  Precision: {results['evaluation']['native']['precision']:.4f}")
            print(f"  Recall: {results['evaluation']['native']['recall']:.4f}")
            print(f"  F1 Score: {results['evaluation']['native']['f1']:.4f}")
            print(f"  MAE: {results['evaluation']['native']['mae']:.4f}")
        
        if 'synthetic' in results['evaluation'] and results['evaluation']['synthetic'].get('num_predictions', 0) > 0:
            print("\nSynthetic Icebergs:")
            print(f"  Accuracy: {results['evaluation']['synthetic']['accuracy']:.4f}")
            print(f"  Precision: {results['evaluation']['synthetic']['precision']:.4f}")
            print(f"  Recall: {results['evaluation']['synthetic']['recall']:.4f}")
            print(f"  F1 Score: {results['evaluation']['synthetic']['f1']:.4f}")
            print(f"  MAE: {results['evaluation']['synthetic']['mae']:.4f}")
    except Exception as e:
        print(f"Error plotting results: {e}")

def main():
    """
    Main function to run the iceberg detection and prediction pipeline
    """
    try:
        # Generate sample data
        messages = generate_sample_mbo_data()
        
        # Run detection and prediction
        results = run_iceberg_detection_and_prediction(messages)
        
        # Plot results
        plot_results(results)
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()