import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import math
import random
from tqdm import tqdm

# For ANFIS implementation, we'll use the skfuzzy package
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Set random seed for reproducibility
np.random.seed(42)

class SimplifiedANFIS:
    """
    A simplified implementation of ANFIS (Adaptive Network-Based Fuzzy Inference System)
    for cryptocurrency price prediction as described in the paper
    """
    
    def __init__(self, n_mfs=3, learning_rate=0.01, epochs=100, clustering='fcm'):
        """
        Initialize the ANFIS model
        
        Parameters:
        -----------
        n_mfs : int
            Number of membership functions
        learning_rate : float
            Learning rate for the backpropagation algorithm
        epochs : int
            Number of training epochs
        clustering : str
            Clustering method ('fcm', 'subtractive', 'grid')
        """
        self.n_mfs = n_mfs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.clustering = clustering
        self.mf_params = None
        self.consequent_params = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def _create_membership_functions(self, X):
        """
        Create membership functions for input variables
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        
        Returns:
        --------
        list
            List of membership functions
        """
        n_features = X.shape[1]
        mfs = []
        
        for i in range(n_features):
            feature_range = np.linspace(np.min(X[:, i]), np.max(X[:, i]), self.n_mfs)
            feature_mfs = []
            
            for j in range(self.n_mfs):
                # Using Gaussian membership function (gaussmf) as mentioned in the paper
                sigma = (feature_range[1] - feature_range[0]) / 2
                c = feature_range[j]
                
                if j == 0:
                    c = feature_range[j]
                elif j == self.n_mfs - 1:
                    c = feature_range[j]
                else:
                    c = (feature_range[j-1] + feature_range[j]) / 2
                
                feature_mfs.append((c, sigma))
            
            mfs.append(feature_mfs)
        
        return mfs
    
    def _fcm_clustering(self, X, n_clusters):
        """
        Fuzzy C-Means clustering
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        n_clusters : int
            Number of clusters
        
        Returns:
        --------
        tuple
            Cluster centers and membership matrix
        """
        # Initialize centroids randomly
        centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
        
        # Initialize membership matrix
        m = 2.0  # fuzziness parameter
        U = np.zeros((X.shape[0], n_clusters))
        
        for _ in range(100):  # max iterations
            # Update membership matrix
            for i in range(X.shape[0]):
                distances = np.linalg.norm(X[i] - centroids, axis=1)
                if np.any(distances == 0):
                    U[i] = np.zeros(n_clusters)
                    U[i, np.argmin(distances)] = 1.0
                else:
                    U[i] = 1.0 / np.sum([(distances[i] / distances[j]) ** (2 / (m - 1)) 
                                        for j in range(n_clusters)])
            
            # Update centroids
            old_centroids = centroids.copy()
            for j in range(n_clusters):
                weights = U[:, j] ** m
                centroids[j] = np.sum(X * weights.reshape(-1, 1), axis=0) / np.sum(weights)
            
            # Check convergence
            if np.linalg.norm(centroids - old_centroids) < 1e-6:
                break
        
        return centroids, U
    
    def _subtractive_clustering(self, X, radius=0.5):
        """
        Subtractive clustering
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        radius : float
            Radius of influence of each cluster center
        
        Returns:
        --------
        numpy.ndarray
            Cluster centers
        """
        # Normalize data
        X_norm = self.scaler.fit_transform(X)
        
        # Initialize potential for each data point
        potential = np.ones(X_norm.shape[0])
        centroids = []
        
        while True:
            # Find the point with highest potential
            max_potential_idx = np.argmax(potential)
            max_potential = potential[max_potential_idx]
            
            # Stop if potential is too low
            if max_potential < 0.15:
                break
            
            # Add point to centroids
            centroids.append(X_norm[max_potential_idx])
            
            # Update potentials
            for i in range(X_norm.shape[0]):
                distance = np.linalg.norm(X_norm[i] - X_norm[max_potential_idx])
                potential[i] -= max_potential * np.exp(-(distance / (radius/2)) ** 2)
            
            # Stop if we have enough clusters
            if len(centroids) >= self.n_mfs:
                break
        
        # Transform centroids back to original scale
        centroids = self.scaler.inverse_transform(np.array(centroids))
        
        return centroids
    
    def _grid_partition(self, X):
        """
        Grid partition
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data
        
        Returns:
        --------
        numpy.ndarray
            Grid points
        """
        n_features = X.shape[1]
        grid_points = []
        
        for i in range(n_features):
            feature_min = np.min(X[:, i])
            feature_max = np.max(X[:, i])
            grid_points.append(np.linspace(feature_min, feature_max, self.n_mfs))
        
        return np.array(grid_points)
    
    def _gaussian_membership(self, x, c, sigma):
        """
        Gaussian membership function
        
        Parameters:
        -----------
        x : float
            Input value
        c : float
            Center of the Gaussian function
        sigma : float
            Width of the Gaussian function
        
        Returns:
        --------
        float
            Membership value
        """
        return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
    
    def fit(self, X, y):
        """
        Train the ANFIS model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        y : numpy.ndarray
            Target values
        
        Returns:
        --------
        self
        """
        # Normalize the data
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Initialize membership functions
        if self.clustering == 'fcm':
            # Use Fuzzy C-Means for clustering
            centroids, _ = self._fcm_clustering(X_scaled, self.n_mfs)
            # Derive membership function parameters from clusters
            self.mf_params = []
            for i in range(X.shape[1]):
                feature_mfs = []
                for j in range(self.n_mfs):
                    c = centroids[j, i]
                    # Estimate sigma based on data spread
                    sigma = 0.2 / self.n_mfs
                    feature_mfs.append((c, sigma))
                self.mf_params.append(feature_mfs)
        
        elif self.clustering == 'subtractive':
            # Use Subtractive clustering
            centroids = self._subtractive_clustering(X_scaled)
            # Derive membership function parameters from clusters
            self.mf_params = []
            for i in range(X.shape[1]):
                feature_mfs = []
                for j in range(min(self.n_mfs, len(centroids))):
                    c = centroids[j, i]
                    # Estimate sigma based on data spread
                    sigma = 0.2 / self.n_mfs
                    feature_mfs.append((c, sigma))
                self.mf_params.append(feature_mfs)
        
        else:  # grid partition
            # Use grid partition
            self.mf_params = self._create_membership_functions(X_scaled)
        
        # Initialize consequent parameters randomly
        n_rules = self.n_mfs ** X.shape[1]
        n_consequent_params = X.shape[1] + 1  # Linear consequent: a1*x1 + a2*x2 + ... + b
        self.consequent_params = np.random.normal(0, 0.1, (n_rules, n_consequent_params))
        
        # Train the model using backpropagation
        for epoch in range(self.epochs):
            total_error = 0
            
            for i in range(X_scaled.shape[0]):
                # Forward pass
                x = X_scaled[i]
                target = y_scaled[i]
                
                # Calculate membership values for each input
                membership_values = []
                for j in range(X.shape[1]):
                    feature_membership = []
                    for k in range(self.n_mfs):
                        c, sigma = self.mf_params[j][k]
                        membership = self._gaussian_membership(x[j], c, sigma)
                        feature_membership.append(membership)
                    membership_values.append(feature_membership)
                
                # Calculate rule firing strengths
                firing_strengths = []
                rule_idx = 0
                
                # For a simple case with 2 inputs and 3 MFs, we'd have 9 rules
                for idx_combinations in np.ndindex(*[self.n_mfs] * X.shape[1]):
                    # Calculate the AND of membership values (product t-norm)
                    firing_strength = 1.0
                    for j, idx in enumerate(idx_combinations):
                        firing_strength *= membership_values[j][idx]
                    
                    firing_strengths.append(firing_strength)
                    rule_idx += 1
                
                # Normalize firing strengths
                sum_firing = sum(firing_strengths)
                if sum_firing > 0:
                    normalized_firing = [fs / sum_firing for fs in firing_strengths]
                else:
                    normalized_firing = [1.0 / len(firing_strengths)] * len(firing_strengths)
                
                # Calculate consequent values
                consequent_values = []
                for j in range(len(firing_strengths)):
                    # Linear combination: a1*x1 + a2*x2 + ... + b
                    consequent = self.consequent_params[j, -1]  # bias term
                    for k in range(X.shape[1]):
                        consequent += self.consequent_params[j, k] * x[k]
                    
                    consequent_values.append(consequent)
                
                # Calculate output
                output = sum(n * c for n, c in zip(normalized_firing, consequent_values))
                
                # Calculate error
                error = target - output
                total_error += error ** 2
                
                # Backward pass - update consequent parameters
                for j in range(len(firing_strengths)):
                    # Update bias term
                    self.consequent_params[j, -1] += self.learning_rate * error * normalized_firing[j]
                    
                    # Update weights
                    for k in range(X.shape[1]):
                        self.consequent_params[j, k] += self.learning_rate * error * normalized_firing[j] * x[k]
            
            # Print epoch error
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Error: {total_error / X_scaled.shape[0]:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained ANFIS model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        # Normalize the data
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        
        for i in range(X_scaled.shape[0]):
            x = X_scaled[i]
            
            # Calculate membership values for each input
            membership_values = []
            for j in range(X.shape[1]):
                feature_membership = []
                for k in range(min(self.n_mfs, len(self.mf_params[j]))):
                    c, sigma = self.mf_params[j][k]
                    membership = self._gaussian_membership(x[j], c, sigma)
                    feature_membership.append(membership)
                membership_values.append(feature_membership)
            
            # Calculate rule firing strengths
            firing_strengths = []
            rule_idx = 0
            
            # For a simple case with 2 inputs and 3 MFs, we'd have 9 rules
            for idx_combinations in np.ndindex(*[self.n_mfs] * X.shape[1]):
                # Calculate the AND of membership values (product t-norm)
                firing_strength = 1.0
                for j, idx in enumerate(idx_combinations):
                    if idx < len(membership_values[j]):
                        firing_strength *= membership_values[j][idx]
                
                firing_strengths.append(firing_strength)
                rule_idx += 1
            
            # Normalize firing strengths
            sum_firing = sum(firing_strengths)
            if sum_firing > 0:
                normalized_firing = [fs / sum_firing for fs in firing_strengths]
            else:
                normalized_firing = [1.0 / len(firing_strengths)] * len(firing_strengths)
            
            # Calculate consequent values
            consequent_values = []
            for j in range(len(firing_strengths)):
                # Linear combination: a1*x1 + a2*x2 + ... + b
                if j < self.consequent_params.shape[0]:
                    consequent = self.consequent_params[j, -1]  # bias term
                    for k in range(X.shape[1]):
                        consequent += self.consequent_params[j, k] * x[k]
                    
                    consequent_values.append(consequent)
            
            # Calculate output
            if consequent_values:
                output = sum(n * c for n, c in zip(normalized_firing[:len(consequent_values)], consequent_values))
                predictions.append(output)
            else:
                predictions.append(0)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        return predictions

def generate_crypto_data(days=1000, volatility=0.05, trend=0.001):
    """
    Generate synthetic cryptocurrency price data
    
    Parameters:
    -----------
    days : int
        Number of days to generate
    volatility : float
        Daily volatility
    trend : float
        Daily trend factor
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with generated cryptocurrency data
    """
    # Start date
    start_date = datetime.datetime(2020, 1, 1)
    
    # Generate dates
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    
    # Generate BTC price data
    btc_price = 10000  # Starting price
    btc_prices = [btc_price]
    
    for i in range(1, days):
        # Random daily return with trend and volatility
        daily_return = np.random.normal(trend, volatility)
        btc_price = btc_price * (1 + daily_return)
        btc_prices.append(btc_price)
    
    # Generate ETH price data (correlated with BTC but with own dynamics)
    eth_price = 200  # Starting price
    eth_prices = [eth_price]
    
    for i in range(1, days):
        # Correlation with BTC return + own dynamics
        btc_return = (btc_prices[i] / btc_prices[i-1]) - 1
        eth_return = 0.7 * btc_return + 0.3 * np.random.normal(trend*1.2, volatility*1.2)
        eth_price = eth_price * (1 + eth_return)
        eth_prices.append(eth_price)
    
    # Generate BTC dominance data
    btc_dominance = 60  # Starting dominance percentage
    btc_dominances = [btc_dominance]
    
    for i in range(1, days):
        # BTC dominance changes based on price action
        btc_return = (btc_prices[i] / btc_prices[i-1]) - 1
        eth_return = (eth_prices[i] / eth_prices[i-1]) - 1
        
        # If BTC outperforms ETH, dominance increases
        dominance_change = 0.2 * (btc_return - eth_return) + np.random.normal(0, 0.002)
        btc_dominance = btc_dominance * (1 + dominance_change)
        # Keep dominance within reasonable bounds
        btc_dominance = max(30, min(90, btc_dominance))
        btc_dominances.append(btc_dominance)
    
    # Generate ETH dominance data
    eth_dominance = 15  # Starting dominance percentage
    eth_dominances = [eth_dominance]
    
    for i in range(1, days):
        # ETH dominance changes based on price action and inversely to BTC dominance
        eth_return = (eth_prices[i] / eth_prices[i-1]) - 1
        btc_return = (btc_prices[i] / btc_prices[i-1]) - 1
        btc_dominance_change = (btc_dominances[i] / btc_dominances[i-1]) - 1
        
        # If ETH outperforms BTC or BTC dominance decreases, ETH dominance increases
        dominance_change = 0.1 * (eth_return - btc_return) - 0.3 * btc_dominance_change + np.random.normal(0, 0.002)
        eth_dominance = eth_dominance * (1 + dominance_change)
        # Keep dominance within reasonable bounds
        eth_dominance = max(5, min(40, eth_dominance))
        eth_dominances.append(eth_dominance)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'BTC': btc_prices,
        'ETH': eth_prices,
        'BTC.D': btc_dominances,
        'ETH.D': eth_dominances
    })
    
    return df

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using RMSE and RMSRE as in the paper
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Root Mean Square Relative Error (RMSRE)
    rmsre = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))
    
    return {
        'RMSE': rmse,
        'RMSRE': rmsre
    }

def test_strategy(df, window_size=30, test_ratio=0.1):
    """
    Test the ANFIS strategy on cryptocurrency data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with cryptocurrency data
    window_size : int
        Number of days to use for training
    test_ratio : float
        Ratio of data to use for testing
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics and predictions
    """
    # Split data into training and testing
    train_size = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Training data size: {len(train_df)}")
    print(f"Testing data size: {len(test_df)}")
    
    # Prepare data for ANFIS
    # For BTC prediction
    X_btc_train = np.array(train_df[['BTC', 'BTC.D']].values[:-1])
    y_btc_train = np.array(train_df['BTC'].values[1:])
    
    X_btc_test = np.array(test_df[['BTC', 'BTC.D']].values[:-1])
    y_btc_test = np.array(test_df['BTC'].values[1:])
    
    # For ETH prediction
    X_eth_train = np.array(train_df[['ETH', 'ETH.D']].values[:-1])
    y_eth_train = np.array(train_df['ETH'].values[1:])
    
    X_eth_test = np.array(test_df[['ETH', 'ETH.D']].values[:-1])
    y_eth_test = np.array(test_df['ETH'].values[1:])
    
    # For BTC.D prediction
    X_btcd_train = np.array(train_df['BTC.D'].values[:-1]).reshape(-1, 1)
    y_btcd_train = np.array(train_df['BTC.D'].values[1:])
    
    X_btcd_test = np.array(test_df['BTC.D'].values[:-1]).reshape(-1, 1)
    y_btcd_test = np.array(test_df['BTC.D'].values[1:])
    
    # For ETH.D prediction
    X_ethd_train = np.array(train_df['ETH.D'].values[:-1]).reshape(-1, 1)
    y_ethd_train = np.array(train_df['ETH.D'].values[1:])
    
    X_ethd_test = np.array(test_df['ETH.D'].values[:-1]).reshape(-1, 1)
    y_ethd_test = np.array(test_df['ETH.D'].values[1:])
    
    # Train ANFIS models
    print("Training BTC model...")
    anfis_btc = SimplifiedANFIS(n_mfs=3, learning_rate=0.01, epochs=50, clustering='fcm')
    anfis_btc.fit(X_btc_train, y_btc_train)
    
    print("Training ETH model...")
    anfis_eth = SimplifiedANFIS(n_mfs=3, learning_rate=0.01, epochs=50, clustering='fcm')
    anfis_eth.fit(X_eth_train, y_eth_train)
    
    print("Training BTC.D model...")
    anfis_btcd = SimplifiedANFIS(n_mfs=3, learning_rate=0.01, epochs=50, clustering='fcm')
    anfis_btcd.fit(X_btcd_train, y_btcd_train)
    
    print("Training ETH.D model...")
    anfis_ethd = SimplifiedANFIS(n_mfs=3, learning_rate=0.01, epochs=50, clustering='fcm')
    anfis_ethd.fit(X_ethd_train, y_ethd_train)
    
    # Make predictions on test data
    y_btc_pred = anfis_btc.predict(X_btc_test)
    y_eth_pred = anfis_eth.predict(X_eth_test)
    y_btcd_pred = anfis_btcd.predict(X_btcd_test)
    y_ethd_pred = anfis_ethd.predict(X_ethd_test)
    
    # Evaluate model performance
    btc_metrics = evaluate_model(y_btc_test, y_btc_pred)
    eth_metrics = evaluate_model(y_eth_test, y_eth_pred)
    btcd_metrics = evaluate_model(y_btcd_test, y_btcd_pred)
    ethd_metrics = evaluate_model(y_ethd_test, y_ethd_pred)
    
    print("BTC Metrics:", btc_metrics)
    print("ETH Metrics:", eth_metrics)
    print("BTC.D Metrics:", btcd_metrics)
    print("ETH.D Metrics:", ethd_metrics)
    
    # Predict 7 days ahead as described in the paper
    btc_7_days = []
    eth_7_days = []
    btcd_7_days = []
    ethd_7_days = []
    
    # Start with the last day in the test set
    current_btc = test_df['BTC'].values[-1]
    current_eth = test_df['ETH'].values[-1]
    current_btcd = test_df['BTC.D'].values[-1]
    current_ethd = test_df['ETH.D'].values[-1]
    
    # True values for comparison (if available)
    true_btc_7_days = []
    true_eth_7_days = []
    
    # Predict 7 days ahead
    for i in range(7):
        # Make predictions
        next_btcd = anfis_btcd.predict(np.array([[current_btcd]]))[0]
        next_ethd = anfis_ethd.predict(np.array([[current_ethd]]))[0]
        
        next_btc = anfis_btc.predict(np.array([[current_btc, current_btcd]]))[0]
        next_eth = anfis_eth.predict(np.array([[current_eth, current_ethd]]))[0]
        
        # Store predictions
        btc_7_days.append(next_btc)
        eth_7_days.append(next_eth)
        btcd_7_days.append(next_btcd)
        ethd_7_days.append(next_ethd)
        
        # Update current values for next prediction
        current_btc = next_btc
        current_eth = next_eth
        current_btcd = next_btcd
        current_ethd = next_ethd
    
    return {
        'BTC Metrics': btc_metrics,
        'ETH Metrics': eth_metrics,
        'BTC.D Metrics': btcd_metrics,
        'ETH.D Metrics': ethd_metrics,
        'BTC 7-day Forecast': btc_7_days,
        'ETH 7-day Forecast': eth_7_days,
        'BTC.D 7-day Forecast': btcd_7_days,
        'ETH.D 7-day Forecast': ethd_7_days,
        'BTC Predictions': y_btc_pred,
        'ETH Predictions': y_eth_pred,
        'BTC.D Predictions': y_btcd_pred,
        'ETH.D Predictions': y_ethd_pred,
        'BTC True': y_btc_test,
        'ETH True': y_eth_test,
        'BTC.D True': y_btcd_test,
        'ETH.D True': y_ethd_test
    }

def compare_models():
    """
    Compare different ANFIS configurations and neural network models
    as mentioned in the paper
    """
    # Generate data
    df = generate_crypto_data(days=500, volatility=0.05, trend=0.001)
    
    # Split data into training and testing
    train_size = int(len(df) * 0.9)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Prepare data for BTC prediction
    X_btc_train_single = np.array(train_df['BTC'].values[:-1]).reshape(-1, 1)
    y_btc_train = np.array(train_df['BTC'].values[1:])
    
    X_btc_test_single = np.array(test_df['BTC'].values[:-1]).reshape(-1, 1)
    y_btc_test = np.array(test_df['BTC'].values[1:])
    
    X_btc_train_multi = np.array(train_df[['BTC', 'BTC.D']].values[:-1])
    X_btc_test_multi = np.array(test_df[['BTC', 'BTC.D']].values[:-1])
    
    # Compare different ANFIS configurations
    anfis_configs = [
        {'name': 'ANFIS Grid + Hybrid (BTC only)', 'clustering': 'grid', 'training': 'hybrid', 'multi_input': False},
        {'name': 'ANFIS Grid + Backprop (BTC only)', 'clustering': 'grid', 'training': 'backprop', 'multi_input': False},
        {'name': 'ANFIS Sub + Hybrid (BTC only)', 'clustering': 'subtractive', 'training': 'hybrid', 'multi_input': False},
        {'name': 'ANFIS Sub + Backprop (BTC only)', 'clustering': 'subtractive', 'training': 'backprop', 'multi_input': False},
        {'name': 'ANFIS FCM + Hybrid (BTC only)', 'clustering': 'fcm', 'training': 'hybrid', 'multi_input': False},
        {'name': 'ANFIS FCM + Backprop (BTC only)', 'clustering': 'fcm', 'training': 'backprop', 'multi_input': False},
        {'name': 'ANFIS Grid + Hybrid (BTC + BTC.D)', 'clustering': 'grid', 'training': 'hybrid', 'multi_input': True},
        {'name': 'ANFIS Grid + Backprop (BTC + BTC.D)', 'clustering': 'grid', 'training': 'backprop', 'multi_input': True},
        {'name': 'ANFIS Sub + Hybrid (BTC + BTC.D)', 'clustering': 'subtractive', 'training': 'hybrid', 'multi_input': True},
        {'name': 'ANFIS Sub + Backprop (BTC + BTC.D)', 'clustering': 'subtractive', 'training': 'backprop', 'multi_input': True},
        {'name': 'ANFIS FCM + Hybrid (BTC + BTC.D)', 'clustering': 'fcm', 'training': 'hybrid', 'multi_input': True},
        {'name': 'ANFIS FCM + Backprop (BTC + BTC.D)', 'clustering': 'fcm', 'training': 'backprop', 'multi_input': True}
    ]
    
    results = []
    
    for config in anfis_configs:
        print(f"\nTraining {config['name']}...")
        
        # Choose input data based on configuration
        if config['multi_input']:
            X_train = X_btc_train_multi
            X_test = X_btc_test_multi
        else:
            X_train = X_btc_train_single
            X_test = X_btc_test_single
        
        # Create and train ANFIS model
        epochs = 20  # Reduced for demonstration
        learning_rate = 0.01
        
        if config['training'] == 'hybrid':
            # Hybrid algorithm modifies learning rate during training
            learning_rate = 0.02
        
        anfis = SimplifiedANFIS(
            n_mfs=3, 
            learning_rate=learning_rate, 
            epochs=epochs, 
            clustering=config['clustering']
        )
        
        anfis.fit(X_train, y_btc_train)
        
        # Make predictions
        y_train_pred = anfis.predict(X_train)
        y_test_pred = anfis.predict(X_test)
        
        # Evaluate model
        train_metrics = evaluate_model(y_btc_train, y_train_pred)
        test_metrics = evaluate_model(y_btc_test, y_test_pred)
        
        results.append({
            'Name': config['name'],
            'Train RMSE': train_metrics['RMSE'],
            'Test RMSE': test_metrics['RMSE'],
            'Train RMSRE': train_metrics['RMSRE'],
            'Test RMSRE': test_metrics['RMSRE']
        })
    
    # Convert to DataFrame for easier comparison
    results_df = pd.DataFrame(results)
    print("\nComparison of different ANFIS configurations:")
    print(results_df)
    
    # Find the best configuration
    best_config = results_df.loc[results_df['Test RMSE'].idxmin()]
    print("\nBest configuration:")
    print(best_config)
    
    return results_df

def plot_results(results):
    """
    Plot the results of the ANFIS strategy
    
    Parameters:
    -----------
    results : dict
        Dictionary with evaluation metrics and predictions
    """
    # Plot BTC predictions vs true values
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(results['BTC True'], label='True')
    plt.plot(results['BTC Predictions'], label='Predicted')
    plt.title('BTC Price: True vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot ETH predictions vs true values
    plt.subplot(2, 2, 2)
    plt.plot(results['ETH True'], label='True')
    plt.plot(results['ETH Predictions'], label='Predicted')
    plt.title('ETH Price: True vs Predicted')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot BTC 7-day forecast
    plt.subplot(2, 2, 3)
    plt.plot(range(len(results['BTC True'])), results['BTC True'], label='True')
    plt.plot(range(len(results['BTC Predictions']), len(results['BTC Predictions']) + 7), 
             results['BTC 7-day Forecast'], label='7-day Forecast', linestyle='--', marker='o')
    plt.title('BTC 7-day Forecast')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot ETH 7-day forecast
    plt.subplot(2, 2, 4)
    plt.plot(range(len(results['ETH True'])), results['ETH True'], label='True')
    plt.plot(range(len(results['ETH Predictions']), len(results['ETH Predictions']) + 7), 
             results['ETH 7-day Forecast'], label='7-day Forecast', linestyle='--', marker='o')
    plt.title('ETH 7-day Forecast')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot prediction errors
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    errors_btc = results['BTC True'] - results['BTC Predictions']
    plt.plot(errors_btc)
    plt.title('BTC Prediction Error')
    plt.xlabel('Days')
    plt.ylabel('Error ($)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    errors_eth = results['ETH True'] - results['ETH Predictions']
    plt.plot(errors_eth)
    plt.title('ETH Prediction Error')
    plt.xlabel('Days')
    plt.ylabel('Error ($)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.hist(errors_btc, bins=20)
    plt.title('BTC Error Distribution')
    plt.xlabel('Error ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.hist(errors_eth, bins=20)
    plt.title('ETH Error Distribution')
    plt.xlabel('Error ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def backtest_trading_strategy(df, results, initial_capital=10000, commission=0.001):
    """
    Backtest a simple trading strategy based on ANFIS predictions
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with cryptocurrency data
    results : dict
        Dictionary with evaluation metrics and predictions
    initial_capital : float
        Initial capital to invest
    commission : float
        Trading commission as a percentage
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with backtest results
    """
    # Extract test data
    train_size = int(len(df) * 0.9)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    
    # Create a DataFrame for backtest results
    backtest = pd.DataFrame(index=range(len(test_df)))
    backtest['Date'] = test_df['Date']
    backtest['BTC_Price'] = test_df['BTC']
    backtest['ETH_Price'] = test_df['ETH']
    
    # Add predictions to backtest DataFrame
    backtest['BTC_Pred'] = np.nan
    backtest['ETH_Pred'] = np.nan
    backtest.loc[:len(results['BTC Predictions'])-1, 'BTC_Pred'] = results['BTC Predictions']
    backtest.loc[:len(results['ETH Predictions'])-1, 'ETH_Pred'] = results['ETH Predictions']
    
    # Calculate predicted returns
    backtest['BTC_Pred_Return'] = backtest['BTC_Pred'].pct_change()
    backtest['ETH_Pred_Return'] = backtest['ETH_Pred'].pct_change()
    
    # Initialize portfolio metrics
    backtest['BTC_Position'] = 0
    backtest['ETH_Position'] = 0
    backtest['Cash'] = initial_capital
    backtest['Portfolio_Value'] = initial_capital
    
    # Define trading strategy
    # Buy if predicted return is positive, sell if negative
    for i in range(1, len(backtest)):
        if pd.isna(backtest.loc[i, 'BTC_Pred_Return']) or pd.isna(backtest.loc[i, 'ETH_Pred_Return']):
            # No prediction available, hold current position
            backtest.loc[i, 'BTC_Position'] = backtest.loc[i-1, 'BTC_Position']
            backtest.loc[i, 'ETH_Position'] = backtest.loc[i-1, 'ETH_Position']
            backtest.loc[i, 'Cash'] = backtest.loc[i-1, 'Cash']
        else:
            # Previous positions
            prev_btc_pos = backtest.loc[i-1, 'BTC_Position']
            prev_eth_pos = backtest.loc[i-1, 'ETH_Position']
            
            # Current cash
            cash = backtest.loc[i-1, 'Cash']
            
            # Decide on BTC position
            if backtest.loc[i, 'BTC_Pred_Return'] > 0.01:  # Buy signal with 1% threshold
                if prev_btc_pos == 0:  # Not already in position
                    # Allocate 40% of portfolio to BTC
                    portfolio_value = backtest.loc[i-1, 'Portfolio_Value']
                    allocation = 0.4 * portfolio_value
                    
                    if allocation <= cash:  # Check if enough cash
                        # Buy BTC
                        btc_to_buy = allocation / backtest.loc[i, 'BTC_Price']
                        # Apply commission
                        btc_to_buy *= (1 - commission)
                        
                        backtest.loc[i, 'BTC_Position'] = btc_to_buy
                        backtest.loc[i, 'Cash'] = cash - allocation
                    else:
                        # Not enough cash, hold position
                        backtest.loc[i, 'BTC_Position'] = prev_btc_pos
                        backtest.loc[i, 'Cash'] = cash
                else:
                    # Already in position, hold
                    backtest.loc[i, 'BTC_Position'] = prev_btc_pos
                    backtest.loc[i, 'Cash'] = cash
            elif backtest.loc[i, 'BTC_Pred_Return'] < -0.01:  # Sell signal with 1% threshold
                if prev_btc_pos > 0:  # In position
                    # Sell BTC
                    cash_from_sale = prev_btc_pos * backtest.loc[i, 'BTC_Price']
                    # Apply commission
                    cash_from_sale *= (1 - commission)
                    
                    backtest.loc[i, 'BTC_Position'] = 0
                    backtest.loc[i, 'Cash'] = cash + cash_from_sale
                else:
                    # Not in position, hold
                    backtest.loc[i, 'BTC_Position'] = prev_btc_pos
                    backtest.loc[i, 'Cash'] = cash
            else:
                # No clear signal, hold position
                backtest.loc[i, 'BTC_Position'] = prev_btc_pos
                backtest.loc[i, 'Cash'] = cash
            
            # Decide on ETH position
            cash = backtest.loc[i, 'Cash']  # Updated cash after BTC decision
            
            if backtest.loc[i, 'ETH_Pred_Return'] > 0.01:  # Buy signal with 1% threshold
                if prev_eth_pos == 0:  # Not already in position
                    # Allocate 40% of portfolio to ETH
                    portfolio_value = backtest.loc[i-1, 'Portfolio_Value']
                    allocation = 0.4 * portfolio_value
                    
                    if allocation <= cash:  # Check if enough cash
                        # Buy ETH
                        eth_to_buy = allocation / backtest.loc[i, 'ETH_Price']
                        # Apply commission
                        eth_to_buy *= (1 - commission)
                        
                        backtest.loc[i, 'ETH_Position'] = eth_to_buy
                        backtest.loc[i, 'Cash'] = cash - allocation
                    else:
                        # Not enough cash, hold position
                        backtest.loc[i, 'ETH_Position'] = prev_eth_pos
                        backtest.loc[i, 'Cash'] = cash
                else:
                    # Already in position, hold
                    backtest.loc[i, 'ETH_Position'] = prev_eth_pos
                    backtest.loc[i, 'Cash'] = cash
            elif backtest.loc[i, 'ETH_Pred_Return'] < -0.01:  # Sell signal with 1% threshold
                if prev_eth_pos > 0:  # In position
                    # Sell ETH
                    cash_from_sale = prev_eth_pos * backtest.loc[i, 'ETH_Price']
                    # Apply commission
                    cash_from_sale *= (1 - commission)
                    
                    backtest.loc[i, 'ETH_Position'] = 0
                    backtest.loc[i, 'Cash'] = cash + cash_from_sale
                else:
                    # Not in position, hold
                    backtest.loc[i, 'ETH_Position'] = prev_eth_pos
                    backtest.loc[i, 'Cash'] = cash
            else:
                # No clear signal, hold position
                backtest.loc[i, 'ETH_Position'] = prev_eth_pos
                backtest.loc[i, 'Cash'] = cash
        
        # Calculate portfolio value
        btc_value = backtest.loc[i, 'BTC_Position'] * backtest.loc[i, 'BTC_Price']
        eth_value = backtest.loc[i, 'ETH_Position'] * backtest.loc[i, 'ETH_Price']
        backtest.loc[i, 'Portfolio_Value'] = backtest.loc[i, 'Cash'] + btc_value + eth_value
    
    # Calculate returns
    backtest['Daily_Return'] = backtest['Portfolio_Value'].pct_change()
    
    # Calculate cumulative returns
    backtest['Cumulative_Return'] = (1 + backtest['Daily_Return']).cumprod() - 1
    
    # Calculate BTC buy and hold returns
    backtest['BTC_Hold_Value'] = initial_capital * backtest['BTC_Price'] / backtest['BTC_Price'].iloc[0]
    backtest['BTC_Hold_Return'] = backtest['BTC_Hold_Value'].pct_change()
    backtest['BTC_Hold_Cumulative'] = (1 + backtest['BTC_Hold_Return']).cumprod() - 1
    
    # Calculate ETH buy and hold returns
    backtest['ETH_Hold_Value'] = initial_capital * backtest['ETH_Price'] / backtest['ETH_Price'].iloc[0]
    backtest['ETH_Hold_Return'] = backtest['ETH_Hold_Value'].pct_change()
    backtest['ETH_Hold_Cumulative'] = (1 + backtest['ETH_Hold_Return']).cumprod() - 1
    
    # Calculate portfolio statistics
    portfolio_stats = {}
    portfolio_stats['Initial Capital'] = initial_capital
    portfolio_stats['Final Portfolio Value'] = backtest['Portfolio_Value'].iloc[-1]
    portfolio_stats['Total Return'] = (backtest['Portfolio_Value'].iloc[-1] / initial_capital) - 1
    portfolio_stats['Annualized Return'] = (1 + portfolio_stats['Total Return']) ** (252 / len(backtest)) - 1
    portfolio_stats['Volatility'] = backtest['Daily_Return'].std() * np.sqrt(252)
    portfolio_stats['Sharpe Ratio'] = portfolio_stats['Annualized Return'] / portfolio_stats['Volatility']
    portfolio_stats['Max Drawdown'] = (backtest['Portfolio_Value'] / backtest['Portfolio_Value'].cummax() - 1).min()
    
    # Calculate BTC buy and hold statistics
    btc_stats = {}
    btc_stats['Initial Capital'] = initial_capital
    btc_stats['Final Portfolio Value'] = backtest['BTC_Hold_Value'].iloc[-1]
    btc_stats['Total Return'] = (backtest['BTC_Hold_Value'].iloc[-1] / initial_capital) - 1
    btc_stats['Annualized Return'] = (1 + btc_stats['Total Return']) ** (252 / len(backtest)) - 1
    btc_stats['Volatility'] = backtest['BTC_Hold_Return'].std() * np.sqrt(252)
    btc_stats['Sharpe Ratio'] = btc_stats['Annualized Return'] / btc_stats['Volatility']
    btc_stats['Max Drawdown'] = (backtest['BTC_Hold_Value'] / backtest['BTC_Hold_Value'].cummax() - 1).min()
    
    # Calculate ETH buy and hold statistics
    eth_stats = {}
    eth_stats['Initial Capital'] = initial_capital
    eth_stats['Final Portfolio Value'] = backtest['ETH_Hold_Value'].iloc[-1]
    eth_stats['Total Return'] = (backtest['ETH_Hold_Value'].iloc[-1] / initial_capital) - 1
    eth_stats['Annualized Return'] = (1 + eth_stats['Total Return']) ** (252 / len(backtest)) - 1
    eth_stats['Volatility'] = backtest['ETH_Hold_Return'].std() * np.sqrt(252)
    eth_stats['Sharpe Ratio'] = eth_stats['Annualized Return'] / eth_stats['Volatility']
    eth_stats['Max Drawdown'] = (backtest['ETH_Hold_Value'] / backtest['ETH_Hold_Value'].cummax() - 1).min()
    
    print("Portfolio Statistics:")
    for key, value in portfolio_stats.items():
        print(f"{key}: {value:.4f}")
    
    print("\nBTC Buy and Hold Statistics:")
    for key, value in btc_stats.items():
        print(f"{key}: {value:.4f}")
    
    print("\nETH Buy and Hold Statistics:")
    for key, value in eth_stats.items():
        print(f"{key}: {value:.4f}")
    
    # Plot portfolio value and buy and hold values
    plt.figure(figsize=(12, 6))
    plt.plot(backtest['Date'], backtest['Portfolio_Value'], label='ANFIS Strategy')
    plt.plot(backtest['Date'], backtest['BTC_Hold_Value'], label='BTC Buy and Hold')
    plt.plot(backtest['Date'], backtest['ETH_Hold_Value'], label='ETH Buy and Hold')
    plt.title('Portfolio Value vs Buy and Hold')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(backtest['Date'], backtest['Cumulative_Return'], label='ANFIS Strategy')
    plt.plot(backtest['Date'], backtest['BTC_Hold_Cumulative'], label='BTC Buy and Hold')
    plt.plot(backtest['Date'], backtest['ETH_Hold_Cumulative'], label='ETH Buy and Hold')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return backtest, portfolio_stats, btc_stats, eth_stats

# Main execution
if __name__ == "__main__":
    # Generate synthetic cryptocurrency data
    print("Generating synthetic cryptocurrency data...")
    df = generate_crypto_data(days=500, volatility=0.05, trend=0.001)
    
    # Display the first few rows of the generated data
    print("\nGenerated Data Sample:")
    print(df.head())
    
    # Compare different ANFIS configurations
    print("\nComparing different ANFIS configurations...")
    compare_results = compare_models()
    
    # Test the ANFIS strategy
    print("\nTesting ANFIS strategy...")
    results = test_strategy(df)
    
    # Plot results
    print("\nPlotting results...")
    plot_results(results)
    
    # Backtest trading strategy
    print("\nBacktesting trading strategy...")
    backtest, portfolio_stats, btc_stats, eth_stats = backtest_trading_strategy(df, results)
    
    print("\nBacktest complete!")