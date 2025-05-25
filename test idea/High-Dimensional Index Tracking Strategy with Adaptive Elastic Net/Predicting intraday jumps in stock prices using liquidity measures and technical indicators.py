import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class CustomSMOTE:
    """
    A simplified implementation of SMOTE (Synthetic Minority Over-sampling Technique)
    """
    def __init__(self, random_state=None):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit_resample(self, X, y):
        # Find the minority and majority classes
        unique_classes, class_counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_classes = [c for c in unique_classes if c != majority_class]
        
        # Set target size to be the same as majority class
        target_size = max(class_counts)
        
        X_resampled = []
        y_resampled = []
        
        # Process each class
        for cls in unique_classes:
            X_cls = X[y == cls]
            y_cls = y[y == cls]
            
            # If this is a minority class, oversample
            if cls in minority_classes:
                n_samples = len(X_cls)
                n_to_generate = target_size - n_samples
                
                if n_to_generate > 0:
                    # Generate synthetic samples
                    for _ in range(n_to_generate):
                        # Randomly select a sample
                        idx = np.random.randint(0, n_samples)
                        sample = X_cls[idx]
                        
                        # Find 5 nearest neighbors
                        # (simplified by using random neighbors)
                        neighbor_idx = np.random.randint(0, n_samples)
                        while neighbor_idx == idx:
                            neighbor_idx = np.random.randint(0, n_samples)
                        
                        neighbor = X_cls[neighbor_idx]
                        
                        # Generate synthetic sample
                        alpha = np.random.random()
                        synthetic_sample = sample + alpha * (neighbor - sample)
                        
                        X_resampled.append(synthetic_sample)
                        y_resampled.append(cls)
            
            # Always include original samples
            X_resampled.extend(X_cls)
            y_resampled.extend(y_cls)
        
        return np.array(X_resampled), np.array(y_resampled)

class RandomUnderSampler:
    """
    A simple implementation of random undersampling
    """
    def __init__(self, sampling_strategy=None, random_state=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit_resample(self, X, y):
        # Find unique classes and their counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Determine target size for each class
        if self.sampling_strategy is None:
            # If no strategy provided, use the size of the minority class
            target_size = min(class_counts)
        else:
            # If strategy is a dictionary, use specified sizes
            target_size = {cls: self.sampling_strategy.get(cls, count) 
                          for cls, count in zip(unique_classes, class_counts)}
        
        X_resampled = []
        y_resampled = []
        
        # Process each class
        for cls in unique_classes:
            indices = np.where(y == cls)[0]
            n_samples = len(indices)
            
            # Determine how many samples to keep
            if isinstance(target_size, dict):
                n_to_keep = min(n_samples, target_size.get(cls, n_samples))
            else:
                n_to_keep = min(n_samples, target_size)
            
            # Randomly select samples to keep
            indices_to_keep = np.random.choice(indices, size=n_to_keep, replace=False)
            
            X_resampled.extend(X[indices_to_keep])
            y_resampled.extend(y[indices_to_keep])
        
        return np.array(X_resampled), np.array(y_resampled)

class StockJumpPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.features = None
        self.selected_features = None
        
    def generate_simulated_data(self, n_days=100, intervals_per_day=48, n_stocks=20):
        """
        Generate simulated stock data
        
        Parameters:
        -----------
        n_days : int
            Number of trading days
        intervals_per_day : int
            Number of 5-minute intervals per day
        n_stocks : int
            Number of stocks
            
        Returns:
        --------
        data : pandas DataFrame
            Simulated stock data
        """
        print("Generating simulated stock data...")
        
        # Initialize variables
        n_intervals = n_days * intervals_per_day
        data = []
        
        for stock_id in range(n_stocks):
            # Baseline price starts at a random value between 50 and 200
            base_price = np.random.uniform(50, 200)
            # Daily volatility (annualized 20-30%)
            daily_volatility = np.random.uniform(0.20, 0.30) / np.sqrt(252)
            # Intraday volatility pattern (U-shape)
            intraday_vol_pattern = np.concatenate([
                np.linspace(1.5, 1.0, intervals_per_day // 4),
                np.linspace(1.0, 1.0, intervals_per_day // 2),
                np.linspace(1.0, 1.3, intervals_per_day // 4)
            ])
            
            # Generate price path
            prices = [base_price]
            for day in range(n_days):
                for interval in range(intervals_per_day):
                    idx = day * intervals_per_day + interval
                    # Adjust volatility for time of day
                    interval_vol = daily_volatility * intraday_vol_pattern[interval]
                    
                    # Introduce occasional jumps (about 5% probability)
                    jump = 0
                    jump_direction = 0
                    if np.random.random() < 0.05:
                        # Jump size as percentage of price
                        jump_size = np.random.uniform(0.005, 0.03)
                        # Determine jump direction
                        jump_direction = 1 if np.random.random() < 0.5 else -1
                        jump = prices[-1] * jump_size * jump_direction
                    
                    # Regular price movement + potential jump
                    price_change = np.random.normal(0, interval_vol) * prices[-1] + jump
                    new_price = prices[-1] + price_change
                    prices.append(new_price)
                    
                    # Compute high and low prices within the interval
                    high_price = new_price * (1 + np.random.uniform(0, 0.005))
                    low_price = new_price * (1 - np.random.uniform(0, 0.005))
                    
                    # Generate volume (higher on jumps)
                    volume_factor = 1.5 if abs(jump) > 0 else 1.0
                    volume = np.random.gamma(shape=2.0, scale=1000 * volume_factor)
                    
                    # Generate number of trades
                    n_trades = int(np.random.gamma(shape=2.0, scale=50 * volume_factor))
                    
                    # Generate best bid and ask prices
                    bid_price = new_price * (1 - np.random.uniform(0.0005, 0.002))
                    ask_price = new_price * (1 + np.random.uniform(0.0005, 0.002))
                    
                    # Generate best bid and ask volumes
                    bid_volume = np.random.gamma(shape=2.0, scale=500)
                    ask_volume = np.random.gamma(shape=2.0, scale=500)
                    
                    # Imbalance in order book often precedes jumps
                    if np.random.random() < 0.7 and abs(jump) > 0:
                        if jump_direction > 0:  # Upward jump
                            bid_volume *= 1.5  # More buying pressure
                            ask_volume *= 0.8  # Less selling pressure
                        else:  # Downward jump
                            bid_volume *= 0.8  # Less buying pressure
                            ask_volume *= 1.5  # More selling pressure
                    
                    # Build record
                    record = {
                        'stock_id': stock_id,
                        'day': day,
                        'interval': interval,
                        'timestamp': day * intervals_per_day + interval,
                        'open_price': prices[-2] if idx > 0 else prices[-1],
                        'high_price': high_price,
                        'low_price': low_price,
                        'close_price': new_price,
                        'volume': volume,
                        'n_trades': n_trades,
                        'bid_price_1': bid_price,
                        'ask_price_1': ask_price,
                        'bid_volume_1': bid_volume,
                        'ask_volume_1': ask_volume,
                        'jump': 1 if abs(jump) > 0 else 0,
                        'jump_direction': jump_direction
                    }
                    
                    data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Sort by stock, day, and interval
        df = df.sort_values(['stock_id', 'day', 'interval']).reset_index(drop=True)
        
        print(f"Generated data for {n_stocks} stocks across {n_days} days with {intervals_per_day} intervals per day")
        print(f"Jump frequency: {df['jump'].mean():.2%}")
        
        return df
    
    def compute_liquidity_measures(self, data):
        """
        Compute liquidity measures as described in the paper
        
        Parameters:
        -----------
        data : pandas DataFrame
            Raw stock data
            
        Returns:
        --------
        liq_measures : pandas DataFrame
            Computed liquidity measures
        """
        print("Computing liquidity measures...")
        
        # Group by stock_id to compute measures for each stock separately
        grouped = data.groupby('stock_id')
        
        all_liq_measures = []
        
        for stock_id, stock_data in grouped:
            # Pre-compute mid prices
            stock_data['mid_price'] = (stock_data['bid_price_1'] + stock_data['ask_price_1']) / 2
            
            # 1. Return (rt)
            stock_data['return'] = stock_data['close_price'].pct_change()
            
            # 2. Cumulative return (Rt)
            # For each day, compute intraday cumulative return
            stock_data['cum_return'] = stock_data.groupby('day')['close_price'].transform(
                lambda x: np.log(x / x.iloc[0])
            )
            
            # 3. Number of trades (kt)
            # Already in the data as n_trades
            
            # 4. Trading volume (vt)
            # Already in the data as volume
            
            # 5. Trading size (st) - average volume per trade
            stock_data['trade_size'] = stock_data['volume'] / stock_data['n_trades'].replace(0, 1)
            
            # 6. Order imbalance (oit)
            # We don't have signed trades, so we'll approximate by price movement
            stock_data['price_change'] = stock_data['close_price'].diff()
            stock_data['order_imbalance'] = stock_data.apply(
                lambda x: 2 * x['volume'] * np.sign(x['price_change']) / x['volume'] if x['volume'] > 0 else 0, 
                axis=1
            )
            
            # 7. Depth imbalance (dit)
            stock_data['depth_imbalance'] = 2 * (stock_data['bid_volume_1'] - stock_data['ask_volume_1']) / (stock_data['bid_volume_1'] + stock_data['ask_volume_1'])
            
            # 8. Quoted spread (qst)
            stock_data['quoted_spread'] = 2 * (stock_data['ask_price_1'] - stock_data['bid_price_1']) / (stock_data['ask_price_1'] + stock_data['bid_price_1'])
            
            # 9. Effective spread (est)
            # For simplicity, we'll use a proxy based on the mid price and close price
            stock_data['effective_spread'] = 2 * abs(stock_data['close_price'] - stock_data['mid_price']) / stock_data['mid_price']
            
            # 10. Realized volatility (rvt)
            # Using squared returns within each interval as proxy
            stock_data['realized_vol'] = stock_data['return'] ** 2
            
            all_liq_measures.append(stock_data)
        
        # Combine all stocks back together
        liq_measures = pd.concat(all_liq_measures).sort_values(['stock_id', 'day', 'interval']).reset_index(drop=True)
        
        # Fill NaN values
        liq_measures = liq_measures.fillna(0)
        
        print("Liquidity measures computed successfully!")
        return liq_measures
    
    def compute_technical_indicators(self, data, q_values=[5, 10, 20, 30]):
        """
        Compute technical indicators as described in the paper
        
        Parameters:
        -----------
        data : pandas DataFrame
            Stock data with liquidity measures
        q_values : list
            List of lagged intervals to use
            
        Returns:
        --------
        tech_indicators : pandas DataFrame
            Data with added technical indicators
        """
        print("Computing technical indicators...")
        
        # Group by stock_id to compute indicators for each stock separately
        grouped = data.groupby('stock_id')
        
        all_tech_indicators = []
        
        for stock_id, stock_data in grouped:
            # Make a copy to avoid SettingWithCopyWarning
            df = stock_data.copy()
            
            # For each lag value q
            for q in q_values:
                # 1. Price rate of change (PROC)
                df[f'PROC_{q}'] = df['close_price'].pct_change(q)
                
                # 2. Volume rate of change (VROC)
                df[f'VROC_{q}'] = df['volume'].pct_change(q)
                
                # 3. Moving average of price (MA)
                df[f'MA_{q}'] = df['close_price'].rolling(window=q).mean()
                
                # 4. Exponential moving average of price (EMA)
                df[f'EMA_{q}'] = df['close_price'].ewm(span=q, adjust=False).mean()
                
                # 5. Bias to MA (BIAS)
                df[f'BIAS_{q}'] = (df['close_price'] - df[f'MA_{q}']) / df[f'MA_{q}']
                
                # 6. Bias to EMA (EBIAS)
                df[f'EBIAS_{q}'] = (df['close_price'] - df[f'EMA_{q}']) / df[f'EMA_{q}']
                
                # 7. Price oscillator to MA (OSCP)
                df[f'OSCP_{q}'] = df[f'MA_{q}'].pct_change()
                
                # 8. Price oscillator to EMA (EOSCP)
                df[f'EOSCP_{q}'] = df[f'EMA_{q}'].pct_change()
                
                # 9. Fast stochastic %K (fK)
                highest_high = df['high_price'].rolling(window=q).max()
                lowest_low = df['low_price'].rolling(window=q).min()
                df[f'fK_{q}'] = 100 * (df['close_price'] - lowest_low) / (highest_high - lowest_low + 1e-10)
                
                # 10. Fast stochastic %D (fD)
                df[f'fD_{q}'] = df[f'fK_{q}'].rolling(window=3).mean()
                
                # 11. Slow stochastic %D (sD)
                df[f'sD_{q}'] = df[f'fD_{q}'].rolling(window=3).mean()
                
                # 12. Commodity channel index (CCI)
                typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
                tp_ma = typical_price.rolling(window=q).mean()
                tp_mean_dev = typical_price.rolling(window=q).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
                df[f'CCI_{q}'] = (typical_price - tp_ma) / (0.015 * tp_mean_dev + 1e-10)
            
            # 13. Accumulation/Distribution oscillator (ADO)
            close_open_diff = df['close_price'] - df['open_price']
            high_low_diff = df['high_price'] - df['low_price']
            df['ADO'] = ((df['high_price'] - df['open_price']) - (df['close_price'] - df['low_price'])) / (2 * high_low_diff + 1e-10)
            
            # 14. True range (TR)
            df['prev_close'] = df['close_price'].shift(1)
            df['TR'] = df.apply(
                lambda x: max(
                    x['high_price'] - x['low_price'],
                    abs(x['high_price'] - x['prev_close']),
                    abs(x['low_price'] - x['prev_close'])
                ) if not pd.isna(x['prev_close']) else x['high_price'] - x['low_price'],
                axis=1
            )
            
            # 15. Price and volume trend (PVT)
            df['PVT'] = df['close_price'].pct_change() * df['volume']
            
            # 16. On balance volume (OBV)
            df['OBV'] = 0
            price_diff = df['close_price'].diff()
            df.loc[price_diff > 0, 'OBV'] = df['volume']
            df.loc[price_diff < 0, 'OBV'] = -df['volume']
            df['OBV'] = df['OBV'].cumsum()
            
            # 17. Negative volume index (NVI)
            df['NVI'] = 100.0
            vol_diff = df['volume'].diff()
            for i in range(1, len(df)):
                if vol_diff.iloc[i] < 0:
                    df.iloc[i, df.columns.get_loc('NVI')] = df.iloc[i-1, df.columns.get_loc('NVI')] * (1 + df.iloc[i, df.columns.get_loc('return')])
                else:
                    df.iloc[i, df.columns.get_loc('NVI')] = df.iloc[i-1, df.columns.get_loc('NVI')]
            
            # 18. Positive volume index (PVI)
            df['PVI'] = 100.0
            for i in range(1, len(df)):
                if vol_diff.iloc[i] > 0:
                    df.iloc[i, df.columns.get_loc('PVI')] = df.iloc[i-1, df.columns.get_loc('PVI')] * (1 + df.iloc[i, df.columns.get_loc('return')])
                else:
                    df.iloc[i, df.columns.get_loc('PVI')] = df.iloc[i-1, df.columns.get_loc('PVI')]
                    
            all_tech_indicators.append(df)
        
        # Combine all stocks back together
        tech_indicators = pd.concat(all_tech_indicators).sort_values(['stock_id', 'day', 'interval']).reset_index(drop=True)
        
        # Fill NaN values
        tech_indicators = tech_indicators.fillna(0)
        
        print("Technical indicators computed successfully!")
        return tech_indicators
    
    def prepare_features(self, data, look_back=12):
        """
        Prepare features for model training by aggregating previous intervals
        
        Parameters:
        -----------
        data : pandas DataFrame
            Stock data with liquidity measures and technical indicators
        look_back : int
            Number of previous intervals to include
            
        Returns:
        --------
        features : pandas DataFrame
            Prepared features for model training
        """
        print(f"Preparing features with look_back={look_back}...")
        
        # List of liquidity measures and technical indicators
        liquidity_measures = [
            'return', 'cum_return', 'n_trades', 'volume', 'trade_size',
            'order_imbalance', 'depth_imbalance', 'quoted_spread',
            'effective_spread', 'realized_vol'
        ]
        
        # Get list of technical indicator columns
        tech_indicators = [col for col in data.columns if any(col.startswith(x) for x in 
                          ['PROC_', 'VROC_', 'MA_', 'EMA_', 'BIAS_', 'EBIAS_', 'OSCP_', 'EOSCP_', 
                           'fK_', 'fD_', 'sD_', 'CCI_']) or col in ['ADO', 'TR', 'PVT', 'OBV', 'NVI', 'PVI']]
        
        # Group by stock_id
        grouped = data.groupby('stock_id')
        
        all_features = []
        
        for stock_id, stock_data in grouped:
            # Make a copy to avoid SettingWithCopyWarning
            df = stock_data.copy()
            
            # For each liquidity measure, include previous intervals
            for measure in liquidity_measures:
                for lag in range(1, look_back + 1):
                    df[f'{measure}_lag_{lag}'] = df[measure].shift(lag)
                
                # Also compute average over the past look_back intervals
                df[f'{measure}_avg'] = df[measure].rolling(window=look_back).mean().shift(1)
            
            # For technical indicators, we already have different lagged versions
            # so we don't need to create shifts, but we do need to ensure they're
            # properly aligned (for prediction, we use technical indicators 
            # calculated at the end of the previous interval)
            for indicator in tech_indicators:
                df[f'{indicator}_prev'] = df[indicator].shift(1)
            
            # Drop rows with NaN values due to lags
            df = df.iloc[look_back:].reset_index(drop=True)
            
            all_features.append(df)
        
        # Combine all stocks back together
        features = pd.concat(all_features).sort_values(['stock_id', 'day', 'interval']).reset_index(drop=True)
        
        # Select feature columns for prediction
        # These are the lagged liquidity measures and technical indicators
        liquidity_features = [col for col in features.columns if any(col.endswith(f'lag_{i}') for i in range(1, look_back + 1))
                            or any(col.endswith('_avg') for col in features.columns)]
        tech_features = [col for col in features.columns if col.endswith('_prev')]
        
        self.features = liquidity_features + tech_features
        
        print(f"Feature preparation complete! Total features: {len(self.features)}")
        return features
    
    def select_features(self, X, y, n_features=20):
        """
        Select the most informative features using mutual information
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target vector
        n_features : int
            Number of features to select
            
        Returns:
        --------
        selected_features : list
            List of selected feature indices
        """
        print(f"Selecting top {n_features} features...")
        
        # Calculate mutual information between features and target
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Get indices of features with highest mutual information
        selected_features = np.argsort(mi_scores)[-n_features:]
        
        print("Feature selection complete!")
        return selected_features
    
    def balance_classes(self, X, y, binary=True):
        """
        Balance classes using custom implementations of SMOTE and random undersampling
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target vector
        binary : bool
            Whether the classification is binary or multi-class
            
        Returns:
        --------
        X_balanced : numpy array
            Balanced feature matrix
        y_balanced : numpy array
            Balanced target vector
        """
        print("Balancing classes...")
        
        if binary:
            # Use random undersampling for binary classification
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X, y)
        else:
            # For multi-class (upward jump, downward jump, no jump)
            # First separate jumping and non-jumping instances
            jump_mask = (y != 0)
            X_jumps, y_jumps = X[jump_mask], y[jump_mask]
            X_no_jumps, y_no_jumps = X[~jump_mask], y[~jump_mask]
            
            # Balance upward and downward jumps using SMOTE
            smote = CustomSMOTE(random_state=42)
            X_jumps_balanced, y_jumps_balanced = smote.fit_resample(X_jumps, y_jumps)
            
            # Randomly undersample non-jumping instances
            n_jumps = len(y_jumps_balanced)
            rus = RandomUnderSampler(sampling_strategy={0: n_jumps}, random_state=42)
            X_no_jumps_balanced, y_no_jumps_balanced = rus.fit_resample(X_no_jumps, y_no_jumps)
            
            # Combine balanced jumping and non-jumping instances
            X_balanced = np.vstack([X_jumps_balanced, X_no_jumps_balanced])
            y_balanced = np.hstack([y_jumps_balanced, y_no_jumps_balanced])
            
            # Shuffle the data
            indices = np.arange(len(y_balanced))
            np.random.shuffle(indices)
            X_balanced = X_balanced[indices]
            y_balanced = y_balanced[indices]
        
        print(f"Class balancing complete! New data shape: {X_balanced.shape}")
        return X_balanced, y_balanced
    
    def train_models(self, X, y, model_type='rf', binary=True):
        """
        Train prediction models
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target vector
        model_type : str
            Type of model to train ('rf', 'svm', 'mlp', 'knn')
        binary : bool
            Whether the classification is binary or multi-class
            
        Returns:
        --------
        model : trained model
        """
        print(f"Training {model_type} model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select features based on mutual information
        self.selected_features = self.select_features(X_scaled, y, n_features=min(50, X.shape[1]))
        X_selected = X_scaled[:, self.selected_features]
        
        # Balance classes
        X_balanced, y_balanced = self.balance_classes(X_selected, y, binary=binary)
        
        # Train model
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            model = SVC(kernel='rbf', C=1.0, gamma=10, probability=True, random_state=42)
        elif model_type == 'mlp':
            model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        elif model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=30)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_balanced, y_balanced)
        
        print("Model training complete!")
        self.model = model
        return model
    
    def evaluate_model(self, X, y, binary=True):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target vector
        binary : bool
            Whether the classification is binary or multi-class
            
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        print("Evaluating model performance...")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Select features based on mutual information
        X_selected = X_scaled[:, self.selected_features]
        
        # Predict
        y_pred = self.model.predict(X_selected)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        
        if binary:
            # For binary classification
            f1 = f1_score(y, y_pred)
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(y, y_pred)
            }
        else:
            # For multi-class classification
            f1_upward = f1_score(y, y_pred, labels=[1], average=None)[0] if 1 in y else 0
            f1_downward = f1_score(y, y_pred, labels=[-1], average=None)[0] if -1 in y else 0
            
            metrics = {
                'accuracy': accuracy,
                'f1_upward': f1_upward,
                'f1_downward': f1_downward,
                'confusion_matrix': confusion_matrix(y, y_pred, normalize='true')
            }
        
        print("Model evaluation complete!")
        return metrics
    
    def plot_jump_distribution(self, data):
        """
        Plot the distribution of jumps across intervals
        
        Parameters:
        -----------
        data : pandas DataFrame
            Stock data
        """
        # Count jumps by interval
        jump_counts = data.groupby('interval')['jump'].sum().reset_index()
        
        # Count upward and downward jumps
        upward_jumps = data[(data['jump'] == 1) & (data['jump_direction'] == 1)].groupby('interval').size().reset_index(name='upward')
        downward_jumps = data[(data['jump'] == 1) & (data['jump_direction'] == -1)].groupby('interval').size().reset_index(name='downward')
        
        # Merge the counts
        jump_counts = jump_counts.merge(upward_jumps, on='interval', how='left').merge(downward_jumps, on='interval', how='left')
        jump_counts = jump_counts.fillna(0)
        
        # Plot the distribution
        plt.figure(figsize=(14, 6))
        
        # Plot total jumps
        plt.bar(jump_counts['interval'], jump_counts['jump'], alpha=0.5, label='Total Jumps')
        
        # Plot upward and downward jumps
        plt.bar(jump_counts['interval'], jump_counts['upward'], alpha=0.7, label='Upward Jumps')
        plt.bar(jump_counts['interval'], -jump_counts['downward'], alpha=0.7, label='Downward Jumps')
        
        plt.xlabel('Interval')
        plt.ylabel('Number of Jumps')
        plt.title('Distribution of Jumps Across Intervals')
        plt.legend()
        plt.xticks(np.arange(0, 48, 4))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names=None):
        """
        Plot feature importance for random forest model
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        """
        if not isinstance(self.model, RandomForestClassifier):
            print("Feature importance plot is only available for Random Forest models.")
            return
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.selected_features))]
        else:
            feature_names = [feature_names[i] for i in self.selected_features]
        
        # Get feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot the top 20 features
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.barh(range(min(20, len(indices))), importances[indices][:20], align='center')
        plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.show()
    
    def run_jump_prediction(self, binary=True, model_type='rf'):
        """
        Run the full jump prediction pipeline
        
        Parameters:
        -----------
        binary : bool
            Whether to run binary (jump/no-jump) or multi-class (upward/downward/no-jump) classification
        model_type : str
            Type of model to train ('rf', 'svm', 'mlp', 'knn')
            
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        # Generate simulated data
        data = self.generate_simulated_data()
        
        # Compute liquidity measures
        data = self.compute_liquidity_measures(data)
        
        # Compute technical indicators
        data = self.compute_technical_indicators(data)
        
        # Plot jump distribution
        self.plot_jump_distribution(data)
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Prepare target variable
        if binary:
            y = features['jump'].values
        else:
            # Create multi-class target: 1 for upward jump, -1 for downward jump, 0 for no jump
            y = features['jump'].values * features['jump_direction'].values
        
        # Get feature matrix
        X = features[self.features].values
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        self.train_models(X_train, y_train, model_type=model_type, binary=binary)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test, binary=binary)
        
        # Plot feature importance for random forest
        if model_type == 'rf':
            self.plot_feature_importance(feature_names=self.features)
        
        return metrics

# Run binary classification
print("\n=== BINARY CLASSIFICATION (JUMP/NO-JUMP) ===\n")
predictor = StockJumpPredictor()
metrics_binary = predictor.run_jump_prediction(binary=True, model_type='rf')
print("\nBinary Classification Metrics:")
print(f"Accuracy: {metrics_binary['accuracy']:.4f}")
print(f"F1 Score: {metrics_binary['f1_score']:.4f}")
print("\nConfusion Matrix:")
print(metrics_binary['confusion_matrix'])

# Run multi-class classification
print("\n=== MULTI-CLASS CLASSIFICATION (UPWARD/DOWNWARD/NO-JUMP) ===\n")
predictor = StockJumpPredictor()
metrics_multiclass = predictor.run_jump_prediction(binary=False, model_type='rf')
print("\nMulti-class Classification Metrics:")
print(f"Accuracy: {metrics_multiclass['accuracy']:.4f}")
print(f"F1 Score (Upward Jumps): {metrics_multiclass['f1_upward']:.4f}")
print(f"F1 Score (Downward Jumps): {metrics_multiclass['f1_downward']:.4f}")
print("\nConfusion Matrix:")
print(metrics_multiclass['confusion_matrix'])

# Compare different models for binary classification
print("\n=== MODEL COMPARISON FOR BINARY CLASSIFICATION ===\n")
model_types = ['rf', 'svm', 'mlp', 'knn']
binary_results = {}

for model_type in model_types:
    print(f"\nTraining {model_type} model...")
    predictor = StockJumpPredictor()
    binary_results[model_type] = predictor.run_jump_prediction(binary=True, model_type=model_type)

# Plot model comparison
accuracies = [binary_results[model]['accuracy'] for model in model_types]
f1_scores = [binary_results[model]['f1_score'] for model in model_types]

plt.figure(figsize=(10, 6))
x = np.arange(len(model_types))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy')
plt.bar(x + width/2, f1_scores, width, label='F1 Score')

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Comparison for Binary Jump Prediction')
plt.xticks(x, model_types)
plt.legend()
plt.tight_layout()
plt.show()