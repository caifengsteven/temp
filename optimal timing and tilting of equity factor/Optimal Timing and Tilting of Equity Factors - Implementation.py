import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
import warnings
import datetime as dt
import time
from sklearn.decomposition import PCA
import os
warnings.filterwarnings('ignore')

#########################################################################
# SIMULATION MODE - For when Bloomberg isn't available
#########################################################################

class SimulatedMarketDataGenerator:
    """Generates simulated data when Bloomberg is unavailable"""
    
    def __init__(self, start_date="2000-01-01", end_date="2016-12-31", n_factors=20):
        """
        Initialize the data generator
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        n_factors : int
            Number of factors to generate
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.n_factors = n_factors
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
        
        # For reproducibility
        np.random.seed(42)
    
    def generate_factor_returns(self):
        """
        Generate simulated factor returns
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with simulated factor returns
        """
        factor_names = [
            'CFY', 'DY', 'BTM', 'EY', 'PROF',           # Value factors
            'MOM12', 'STR', 'LTR',                       # Momentum factors
            'AT', 'DLTD', 'DSO', 'AG', 'CP', 'PMA',      # Quality factors
            'LEV', 'ROA', 'STC', 'STI', 'ACC', 'Size'    # More quality factors and size
        ]
        
        # Ensure we have enough factor names
        if len(factor_names) < self.n_factors:
            factor_names.extend([f'Factor{i}' for i in range(len(factor_names)+1, self.n_factors+1)])
        
        # Use only the required number of factors
        factor_names = factor_names[:self.n_factors]
        
        # Create a simple, guaranteed positive definite correlation matrix
        # Start with identity matrix (diagonal of 1s)
        corr_matrix = np.eye(self.n_factors)
        
        # Add some correlation between factors (conservative values)
        # Value factors correlated with each other (indices 0-4)
        value_indices = [0, 1, 2, 3, 4]
        for i in value_indices:
            for j in value_indices:
                if i != j and i < self.n_factors and j < self.n_factors:
                    corr_matrix[i, j] = 0.5  # Conservative correlation
        
        # Momentum factors correlated with each other (indices 5-7)
        momentum_indices = [5, 6, 7]
        for i in momentum_indices:
            for j in momentum_indices:
                if i != j and i < self.n_factors and j < self.n_factors:
                    corr_matrix[i, j] = 0.4  # Conservative correlation
                    
        # Quality factors correlated with each other (indices 8-18)
        quality_indices = list(range(8, 19))
        for i in quality_indices:
            for j in quality_indices:
                if i != j and i < self.n_factors and j < self.n_factors:
                    corr_matrix[i, j] = 0.3  # Conservative correlation
        
        # Size factor (index 19) slightly negative correlated with value factors
        if 19 < self.n_factors:
            for i in value_indices:
                if i < self.n_factors:
                    corr_matrix[19, i] = corr_matrix[i, 19] = -0.2
        
        # Generate monthly factor returns
        n_periods = len(self.date_range)
        
        # Set up factor characteristics
        annualized_returns = np.array([
            9.69, 5.63, 3.58, 8.33, 7.70,  # Value
            12.05, 1.94, 3.20,             # Momentum
            4.55, 4.96, 7.28, 5.96, 4.09, 3.99,  # Quality
            3.75, 5.07, 5.28, 2.48, 0.29, 2.97   # More quality + Size
        ][:self.n_factors]) / 100  # Convert to decimal
        
        annualized_vols = np.array([
            12.61, 14.08, 11.66, 11.35, 6.66,  # Value
            20.21, 14.56, 12.72,               # Momentum
            5.25, 7.22, 9.02, 10.04, 8.19, 8.63,  # Quality
            13.75, 7.12, 11.84, 5.68, 5.70, 13.56  # More quality + Size
        ][:self.n_factors]) / 100  # Convert to decimal
        
        # Convert to monthly
        monthly_returns = annualized_returns / 12
        monthly_vols = annualized_vols / np.sqrt(12)
        
        # Method 1: Simple independent returns + correlation
        factor_returns = np.zeros((n_periods, self.n_factors))
        
        # First generate independent returns
        for i in range(self.n_factors):
            factor_returns[:, i] = np.random.normal(monthly_returns[i], monthly_vols[i], n_periods)
        
        # Add correlation patterns
        # This is a simplified approach that doesn't require Cholesky decomposition
        for t in range(n_periods):
            # Value factors tend to move together
            common_value_shock = np.random.normal(0, 0.01)
            for i in value_indices:
                if i < self.n_factors:
                    factor_returns[t, i] += common_value_shock
            
            # Momentum factors tend to move together but opposite to value
            common_momentum_shock = np.random.normal(0, 0.015)
            for i in momentum_indices:
                if i < self.n_factors:
                    factor_returns[t, i] += common_momentum_shock
            
            # Quality factors have their own pattern
            common_quality_shock = np.random.normal(0, 0.008)
            for i in quality_indices:
                if i < self.n_factors:
                    factor_returns[t, i] += common_quality_shock
            
            # Size is often negatively correlated with value
            if 19 < self.n_factors:
                factor_returns[t, 19] -= common_value_shock * 0.5
        
        # Create DataFrame
        returns_df = pd.DataFrame(
            factor_returns, 
            index=self.date_range, 
            columns=factor_names
        )
        
        return returns_df
    
    def generate_fundamental_predictors(self):
        """
        Generate simulated fundamental predictor variables
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with simulated fundamental predictors
        """
        n_periods = len(self.date_range)
        predictor_names = ['dp', 'dy', 'ep', 'bm', 'svar', 'tbl', 'lty', 'ltr', 'dfy', 'infl', 'tms']
        n_predictors = len(predictor_names)
        
        # Generate basic time series with autocorrelation
        predictor_data = np.zeros((n_periods, n_predictors))
        
        for i in range(n_predictors):
            # Start with random values
            series = np.random.normal(0, 1, n_periods)
            
            # Add autocorrelation
            for t in range(1, n_periods):
                series[t] = 0.7 * series[t-1] + 0.3 * series[t]
            
            # Add trend and seasonality for some predictors
            if i < 4:  # Add trend to first few predictors
                trend = np.linspace(0, 1, n_periods)
                series += trend
            
            if i % 3 == 0:  # Add seasonality to some predictors
                seasonality = 0.5 * np.sin(np.linspace(0, 4 * np.pi, n_periods))
                series += seasonality
            
            predictor_data[:, i] = series
        
        # Create DataFrame
        predictors_df = pd.DataFrame(
            predictor_data,
            index=self.date_range,
            columns=predictor_names
        )
        
        return predictors_df
    
    def generate_technical_indicators(self, factor_returns):
        """
        Generate simulated technical indicators based on factor returns
        
        Parameters:
        -----------
        factor_returns : pandas.DataFrame
            DataFrame with factor returns
            
        Returns:
        --------
        dict
            Dictionary with technical indicators for each factor
        """
        technical_indicators = {}
        
        for factor in factor_returns.columns:
            # Create DataFrame for technical indicators
            factor_indicators = pd.DataFrame(index=factor_returns.index)
            returns = factor_returns[factor]
            
            # Generate momentum indicators (MOMm)
            for m in [1, 3, 6, 9, 12]:
                if len(returns) > m:
                    # MOMm = 1 if Pt > Pt-m, 0 otherwise
                    momentum = (returns > returns.shift(m)).astype(int)
                    # Add some randomness
                    random_flip = (np.random.rand(len(momentum)) > 0.8).astype(int)
                    momentum = momentum.astype(int) ^ random_flip
                    factor_indicators[f'MOM{m}'] = momentum
            
            # Generate moving average indicators (MAs-l)
            for s in [1, 2, 3]:
                for l in [9, 12]:
                    if len(returns) > l:
                        # Use shorter series if needed
                        valid_len = min(len(returns), len(returns) - (l - s))
                        if valid_len <= 0:
                            continue
                            
                        # Calculate moving averages
                        if valid_len > s:
                            short_ma = returns.rolling(window=s).mean()
                        else:
                            short_ma = returns
                            
                        if valid_len > l:
                            long_ma = returns.rolling(window=l).mean()
                        else:
                            long_ma = returns.expanding().mean()
                        
                        # MA_s-l = 1 if short_ma > long_ma, 0 otherwise
                        ma_indicator = (short_ma > long_ma).astype(int)
                        # Add some randomness
                        random_flip = (np.random.rand(len(ma_indicator)) > 0.85).astype(int)
                        ma_indicator = ma_indicator.astype(int) ^ random_flip
                        factor_indicators[f'MA{s}-{l}'] = ma_indicator
            
            technical_indicators[factor] = factor_indicators
        
        return technical_indicators
    
    def generate_factor_characteristics(self, factor_returns):
        """
        Generate simulated factor characteristics for tilting
        
        Parameters:
        -----------
        factor_returns : pandas.DataFrame
            DataFrame with factor returns
            
        Returns:
        --------
        dict
            Dictionary with factor characteristics
        """
        characteristics = {}
        
        # 1. Factor Momentum
        momentum = factor_returns.shift(1)
        characteristics['momentum'] = momentum
        
        # 2. Factor Volatility
        volatility = factor_returns.rolling(window=3).std().fillna(method='bfill')
        characteristics['volatility'] = volatility
        
        # 3. Factor Valuation (higher = more expensive)
        valuation = factor_returns.rolling(window=12).mean().fillna(method='bfill')
        # Add some randomness
        random = pd.DataFrame(
            np.random.normal(0, 0.02, size=valuation.shape),
            index=valuation.index,
            columns=valuation.columns
        )
        valuation = valuation + random
        characteristics['valuation'] = valuation
        
        # 4. Factor Spread
        spread = factor_returns.rolling(window=6).std().fillna(method='bfill')
        # Add some randomness
        random = pd.DataFrame(
            np.random.normal(0, 0.005, size=spread.shape),
            index=spread.index,
            columns=spread.columns
        )
        spread = spread + random
        characteristics['spread'] = spread
        
        # 5. Factor Crowding (long and short)
        # Higher values indicate more crowding
        crowding_long = factor_returns.ewm(span=3).mean() / factor_returns.ewm(span=3).std()
        crowding_short = factor_returns.ewm(span=6).mean() / factor_returns.ewm(span=6).std()
        
        # Replace any NaN or infinite values
        crowding_long = crowding_long.replace([np.inf, -np.inf], np.nan).fillna(0)
        crowding_short = crowding_short.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Add some randomness and trends
        time_trend = np.linspace(0, 1, len(factor_returns))
        for col in crowding_long.columns:
            # More recent crowding tends to be higher (increasing time trend)
            crowding_long[col] = crowding_long[col] + 0.5 * time_trend + np.random.normal(0, 0.2, len(crowding_long))
            crowding_short[col] = crowding_short[col] + 0.3 * time_trend + np.random.normal(0, 0.15, len(crowding_short))
        
        characteristics['crowding_long'] = crowding_long
        characteristics['crowding_short'] = crowding_short
        
        return characteristics

#########################################################################
# Parametric Portfolio Policies for Factor Timing and Tilting
#########################################################################

class ParametricPortfolioPolicy:
    """Implements parametric portfolio policies for factor timing and tilting"""
    
    def __init__(self, factor_returns, lookback_window=60, risk_aversion=5, 
                 tracking_error_target=0.025, transaction_cost=0.002):
        """
        Initialize the parametric portfolio policy
        
        Parameters:
        -----------
        factor_returns : pandas.DataFrame
            DataFrame with factor returns
        lookback_window : int
            Length of lookback window for estimation
        risk_aversion : float
            Risk aversion parameter
        tracking_error_target : float
            Target tracking error (annualized)
        transaction_cost : float
            Transaction cost for 100% turnover
        """
        self.factor_returns = factor_returns
        self.lookback_window = lookback_window
        self.risk_aversion = risk_aversion
        self.tracking_error_target = tracking_error_target
        self.transaction_cost = transaction_cost
        self.num_factors = factor_returns.shape[1]
        
        # Equal-weight benchmark
        self.benchmark_weights = np.ones(self.num_factors) / self.num_factors
        
        # For tracking results
        self.theta_timing = None
        self.phi_tilting = None
        self.weights_history = {'timing': {}, 'tilting': {}}
        self.returns_history = {'timing': pd.Series(), 'tilting': pd.Series(), 'benchmark': pd.Series()}
    
    def factor_timing(self, time_series_predictors):
        """
        Implement factor timing using parametric portfolio policy
        
        Parameters:
        -----------
        time_series_predictors : tuple
            (fundamental_pca, technical_pca) - PCA factors for timing
            
        Returns:
        --------
        pandas.Series
            Series with portfolio weights
        """
        fundamental_pca, technical_pca = time_series_predictors
        dates = self.factor_returns.index
        
        # We need at least lookback_window periods of data for estimation
        if len(dates) <= self.lookback_window:
            print("Not enough data for factor timing")
            return pd.Series(self.benchmark_weights, index=self.factor_returns.columns)
        
        # Start with expanding window, then rolling
        first_estimation_date = dates[self.lookback_window]
        
        for i, current_date in enumerate(dates[self.lookback_window:], self.lookback_window):
            if i % 10 == 0:  # Only print every 10th date to reduce output
                print(f"Factor timing optimization for {current_date}...")
            
            # Get historical data for estimation
            if i < 2*self.lookback_window:
                # Use expanding window initially
                hist_start_idx = 0
            else:
                # Use rolling window
                hist_start_idx = i - self.lookback_window
                
            hist_end_idx = i
            
            # Check if we have PCA factors for this period
            if fundamental_pca is None or technical_pca is None:
                # If no predictors, use equal weight
                self.weights_history['timing'][current_date] = pd.Series(
                    self.benchmark_weights, index=self.factor_returns.columns
                )
                continue
                
            # Get historical returns
            hist_returns = self.factor_returns.iloc[hist_start_idx:hist_end_idx]
            
            # Get PCA factors for this period
            hist_fun_pca = fundamental_pca.loc[fundamental_pca.index <= current_date]
            
            if hist_fun_pca.empty:
                # If no predictors, use equal weight
                self.weights_history['timing'][current_date] = pd.Series(
                    self.benchmark_weights, index=self.factor_returns.columns
                )
                continue
            
            # Align data
            aligned_returns = hist_returns
            aligned_fun_pca = hist_fun_pca.reindex(aligned_returns.index).fillna(0)
            
            # Get the technical PCA for each factor
            hist_tech_pca = {}
            for factor in self.factor_returns.columns:
                if factor in technical_pca:
                    factor_tech = technical_pca[factor].loc[technical_pca[factor].index <= current_date]
                    hist_tech_pca[factor] = factor_tech.reindex(aligned_returns.index).fillna(0)
                else:
                    # If no technical PCA for this factor, use zeros
                    hist_tech_pca[factor] = pd.DataFrame(0, index=aligned_returns.index, columns=['TECH1'])
            
            try:
                # Estimate PPP parameters (theta) - separately for each predictor
                theta_fun = self._estimate_timing_theta(aligned_returns, aligned_fun_pca['FUN1'])
                
                theta_tech = {}
                for factor in self.factor_returns.columns:
                    theta_tech[factor] = self._estimate_timing_theta(
                        aligned_returns[factor], hist_tech_pca[factor]['TECH1']
                    )
                
                # Store the estimated parameters
                self.theta_timing = {'fundamental': theta_fun, 'technical': theta_tech}
                
                # Get current values of the predictors
                current_fun = aligned_fun_pca.iloc[-1]['FUN1'] if not aligned_fun_pca.empty else 0
                current_tech = {factor: hist_tech_pca[factor].iloc[-1]['TECH1'] if not hist_tech_pca[factor].empty else 0 
                               for factor in self.factor_returns.columns}
                
                # Compute factor weights based on current predictor values
                weights = self.benchmark_weights.copy()
                
                for j, factor in enumerate(self.factor_returns.columns):
                    # Apply factor timing
                    fund_timing = theta_fun[j] * current_fun if j < len(theta_fun) else 0
                    tech_timing = theta_tech[factor] * current_tech[factor] if factor in theta_tech else 0
                    
                    weights[j] += fund_timing + tech_timing
                
                # Normalize weights to sum to 1
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                
                # Apply tracking error constraint
                weights = self._apply_tracking_error_constraint(weights, hist_returns)
            except Exception as e:
                print(f"Error in factor timing optimization: {e}")
                weights = self.benchmark_weights.copy()
            
            # Store weights
            self.weights_history['timing'][current_date] = pd.Series(
                weights, index=self.factor_returns.columns
            )
        
        # Return the latest weights
        if self.weights_history['timing']:
            latest_date = max(self.weights_history['timing'].keys())
            return self.weights_history['timing'][latest_date]
        else:
            return pd.Series(self.benchmark_weights, index=self.factor_returns.columns)
    
    def _estimate_timing_theta(self, returns, predictor):
        """
        Estimate theta parameters for factor timing
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Historical factor returns
        predictor : pandas.Series
            Historical predictor values
            
        Returns:
        --------
        numpy.ndarray
            Estimated theta parameters
        """
        if predictor.isna().all() or len(predictor) == 0:
            return np.zeros(returns.shape[1] if isinstance(returns, pd.DataFrame) else 1)
        
        # Create augmented factor returns
        if isinstance(returns, pd.Series):
            # If returns is a Series (single factor)
            augmented_returns = returns * predictor
            
            # Estimate theta using mean-variance utility
            expected_return = np.nanmean(augmented_returns)
            variance = np.nanvar(augmented_returns)
            
            if variance > 0:
                theta = expected_return / (self.risk_aversion * variance)
            else:
                theta = 0
                
            return theta
            
        else:
            # If returns is a DataFrame (multiple factors)
            augmented_returns = returns.multiply(predictor, axis=0)
            
            # Estimate theta using mean-variance utility
            expected_returns = np.nanmean(augmented_returns, axis=0)
            
            # Handle NaN values
            augmented_returns_filled = augmented_returns.fillna(0)
            cov_matrix = np.cov(augmented_returns_filled.T)
            
            # Handle potentially singular covariance matrix
            if np.linalg.matrix_rank(cov_matrix) < cov_matrix.shape[0]:
                # Add a small amount to the diagonal
                cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
            
            try:
                # Solve for theta
                theta = np.linalg.solve(self.risk_aversion * cov_matrix, expected_returns)
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in theta estimation")
                theta = np.zeros(returns.shape[1])
            
            return theta
    
    def factor_tilting(self, factor_characteristics):
        """
        Implement factor tilting using parametric portfolio policy
        
        Parameters:
        -----------
        factor_characteristics : dict
            Dictionary with factor characteristics
            
        Returns:
        --------
        pandas.Series
            Series with portfolio weights
        """
        dates = self.factor_returns.index
        
        # We need at least lookback_window periods of data for estimation
        if len(dates) <= self.lookback_window:
            print("Not enough data for factor tilting")
            return pd.Series(self.benchmark_weights, index=self.factor_returns.columns)
        
        # Standardize characteristics
        standardized_characteristics = self._standardize_characteristics(factor_characteristics)
        
        # Start with expanding window, then rolling
        first_estimation_date = dates[self.lookback_window]
        
        for i, current_date in enumerate(dates[self.lookback_window:], self.lookback_window):
            if i % 10 == 0:  # Only print every 10th date to reduce output
                print(f"Factor tilting optimization for {current_date}...")
            
            # Get historical data for estimation
            if i < 2*self.lookback_window:
                # Use expanding window initially
                hist_start_idx = 0
            else:
                # Use rolling window
                hist_start_idx = i - self.lookback_window
                
            hist_end_idx = i
            
            # Check if we have characteristics for this period
            if not standardized_characteristics:
                # If no characteristics, use equal weight
                self.weights_history['tilting'][current_date] = pd.Series(
                    self.benchmark_weights, index=self.factor_returns.columns
                )
                continue
                
            # Get historical returns
            hist_returns = self.factor_returns.iloc[hist_start_idx:hist_end_idx]
            
            # Get historical characteristics
            hist_characteristics = {}
            for name, characteristic in standardized_characteristics.items():
                if current_date in characteristic.index:
                    end_date_idx = list(characteristic.index).index(current_date)
                    start_date_idx = max(0, end_date_idx - (hist_end_idx - hist_start_idx) + 1)
                    hist_characteristics[name] = characteristic.iloc[start_date_idx:end_date_idx+1]
                else:
                    # Skip if no data for this date
                    continue
            
            if not hist_characteristics:
                # If no characteristics, use equal weight
                self.weights_history['tilting'][current_date] = pd.Series(
                    self.benchmark_weights, index=self.factor_returns.columns
                )
                continue
            
            try:
                # Estimate PPP parameters (phi) - separately for each characteristic
                phi = {}
                for name, characteristic in hist_characteristics.items():
                    # Align data
                    aligned_data = pd.concat([hist_returns, characteristic], axis=1)
                    aligned_data = aligned_data.dropna()
                    
                    if len(aligned_data) < 10:  # Skip if too little data
                        continue
                        
                    aligned_returns = aligned_data[hist_returns.columns]
                    aligned_characteristic = aligned_data[characteristic.columns]
                    
                    # Estimate phi using Brandt et al. (2009) approach
                    phi[name] = self._estimate_tilting_phi_simple(aligned_returns, aligned_characteristic)
                
                # Store the estimated parameters
                self.phi_tilting = phi
                
                # Get current values of the characteristics
                current_characteristics = {}
                for name, characteristic in standardized_characteristics.items():
                    if current_date in characteristic.index:
                        current_characteristics[name] = characteristic.loc[current_date]
                    else:
                        current_characteristics[name] = pd.Series(0, index=self.factor_returns.columns)
                
                # Compute factor weights based on current characteristic values
                weights = self.benchmark_weights.copy()
                
                for factor in self.factor_returns.columns:
                    # Start with benchmark weight
                    factor_idx = list(self.factor_returns.columns).index(factor)
                    factor_weight = weights[factor_idx]
                    
                    # Apply characteristic tilts
                    for name, phi_value in phi.items():
                        if name in current_characteristics:
                            factor_char = current_characteristics[name][factor]
                            factor_weight += phi_value * factor_char
                    
                    weights[factor_idx] = factor_weight
                
                # Normalize weights to sum to 1
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = self.benchmark_weights.copy()
                
                # Apply tracking error constraint
                weights = self._apply_tracking_error_constraint(weights, hist_returns)
            except Exception as e:
                print(f"Error in factor tilting optimization: {e}")
                weights = self.benchmark_weights.copy()
            
            # Store weights
            self.weights_history['tilting'][current_date] = pd.Series(
                weights, index=self.factor_returns.columns
            )
        
        # Return the latest weights
        if self.weights_history['tilting']:
            latest_date = max(self.weights_history['tilting'].keys())
            return self.weights_history['tilting'][latest_date]
        else:
            return pd.Series(self.benchmark_weights, index=self.factor_returns.columns)
    
    def _standardize_characteristics(self, factor_characteristics):
        """
        Standardize factor characteristics cross-sectionally
        
        Parameters:
        -----------
        factor_characteristics : dict
            Dictionary with factor characteristics
            
        Returns:
        --------
        dict
            Dictionary with standardized factor characteristics
        """
        standardized = {}
        
        for name, characteristic in factor_characteristics.items():
            # Standardize cross-sectionally at each point in time
            mean = characteristic.mean(axis=1)
            std = characteristic.std(axis=1).replace(0, 1)  # Replace zero std to avoid division by zero
            standardized[name] = characteristic.sub(mean, axis=0).div(std, axis=0)
        
        return standardized
    
    def _estimate_tilting_phi_simple(self, returns, characteristics):
        """
        Simplified estimate of phi parameter for factor tilting
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Historical factor returns
        characteristics : pandas.DataFrame
            Historical factor characteristics
            
        Returns:
        --------
        float
            Estimated phi parameter
        """
        # Calculate correlation between characteristics and future returns
        correlations = []
        for t in range(len(returns) - 1):
            current_chars = characteristics.iloc[t]
            next_returns = returns.iloc[t + 1]
            
            # Calculate correlation between characteristics and next period returns
            if not current_chars.isna().all() and not next_returns.isna().all():
                # Only include non-NaN values
                valid_idx = ~current_chars.isna() & ~next_returns.isna()
                if valid_idx.sum() > 2:  # Need at least 3 data points for correlation
                    corr = np.corrcoef(current_chars[valid_idx], next_returns[valid_idx])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        # If we have correlations, use the mean correlation as phi
        if correlations:
            phi = np.mean(correlations)
            # Scale phi to be more conservative
            phi *= 2.0  # Adjust based on desired aggressiveness
            return phi
        else:
            return 0.0
    
    def _estimate_tilting_phi(self, returns, characteristics):
        """
        Estimate phi parameter for factor tilting
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            Historical factor returns
        characteristics : pandas.DataFrame
            Historical factor characteristics
            
        Returns:
        --------
        float
            Estimated phi parameter
        """
        if characteristics.isna().all().all() or len(characteristics) == 0:
            return 0
        
        # Instead of using optimization, we'll use a simpler approach based on correlation
        # between characteristics and next-period returns
        phi_values = []
        
        # Loop through the time series
        for t in range(len(returns) - 1):
            char_t = characteristics.iloc[t]
            ret_t1 = returns.iloc[t + 1]
            
            # Calculate the portfolio return for different candidate phi values
            phi_candidates = np.linspace(-1, 1, 21)  # Test 21 phi values from -1 to 1
            best_utility = -np.inf
            best_phi = 0
            
            for phi in phi_candidates:
                # Adjust weights based on characteristics
                weights = self.benchmark_weights.copy()
                for i in range(len(weights)):
                    for col_idx in range(characteristics.shape[1]):
                        if i < len(char_t) and not np.isnan(char_t.iloc[col_idx]):
                            weights[i] += phi * char_t.iloc[col_idx]
                
                # Normalize weights to sum to 1
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                
                # Calculate portfolio return
                portfolio_return = np.sum(weights * ret_t1)
                
                # Calculate variance (simplified with just a single period)
                variance = 0.01  # Assume constant variance for simplicity
                
                # Calculate utility
                utility = portfolio_return - 0.5 * self.risk_aversion * variance
                
                if utility > best_utility:
                    best_utility = utility
                    best_phi = phi
            
            phi_values.append(best_phi)
        
        # Return average phi
        if phi_values:
            return np.mean(phi_values)
        else:
            return 0
    
    def _apply_tracking_error_constraint(self, weights, hist_returns):
        """
        Apply tracking error constraint to weights
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Portfolio weights
        hist_returns : pandas.DataFrame
            Historical factor returns
            
        Returns:
        --------
        numpy.ndarray
            Constrained portfolio weights
        """
        # Calculate active weights
        active_weights = weights - self.benchmark_weights
        
        # Calculate expected tracking error (annualized)
        if len(hist_returns) > 1:
            try:
                # Handle NaN values
                hist_returns_filled = hist_returns.fillna(0)
                cov_matrix = np.cov(hist_returns_filled.T)
                tracking_error = np.sqrt(np.dot(active_weights, np.dot(cov_matrix, active_weights))) * np.sqrt(12)
                
                # Scale active weights if tracking error exceeds target
                if tracking_error > self.tracking_error_target and tracking_error > 0:
                    scaling_factor = self.tracking_error_target / tracking_error
                    scaled_active_weights = active_weights * scaling_factor
                    
                    # Apply scaled active weights to benchmark
                    weights = self.benchmark_weights + scaled_active_weights
                    
                    # Ensure weights sum to 1 (can diverge slightly due to numerical precision)
                    if np.sum(weights) > 0:
                        weights = weights / np.sum(weights)
            except Exception as e:
                print(f"Error applying tracking error constraint: {e}")
        
        return weights
    
    def compute_strategy_returns(self, transaction_costs=True):
        """
        Compute returns for the factor timing and tilting strategies
        
        Parameters:
        -----------
        transaction_costs : bool
            Whether to apply transaction costs
            
        Returns:
        --------
        dict
            Dictionary with strategy returns
        """
        # Get all dates with weights
        timing_dates = sorted(self.weights_history['timing'].keys())
        tilting_dates = sorted(self.weights_history['tilting'].keys())
        
        # Compute benchmark returns
        benchmark_returns = pd.Series(index=self.factor_returns.index[self.lookback_window+1:])
        
        for date in benchmark_returns.index:
            # Get factor returns for the next period
            next_date_idx = list(self.factor_returns.index).index(date) + 1
            
            if next_date_idx >= len(self.factor_returns.index):
                continue
                
            next_date = self.factor_returns.index[next_date_idx]
            next_returns = self.factor_returns.loc[next_date]
            
            # Calculate benchmark return
            benchmark_return = np.sum(self.benchmark_weights * next_returns)
            benchmark_returns[date] = benchmark_return
        
        # Compute timing returns
        timing_returns = pd.Series(index=self.factor_returns.index[self.lookback_window+1:])
        prev_weights = None
        
        for date in timing_dates:
            # Get index of current date
            try:
                date_idx = list(self.factor_returns.index).index(date)
            except ValueError:
                # Skip if date not found in factor returns
                continue
                
            # Get factor returns for the next period
            if date_idx + 1 >= len(self.factor_returns.index):
                continue
                
            next_date = self.factor_returns.index[date_idx + 1]
            next_returns = self.factor_returns.loc[next_date]
            
            # Get current weights
            weights = self.weights_history['timing'][date].values
            
            # Calculate transaction costs
            tc = 0
            if transaction_costs and prev_weights is not None:
                weight_change = np.abs(weights - prev_weights).sum()
                tc = weight_change * self.transaction_cost / 2  # One-way cost
            
            prev_weights = weights
            
            # Calculate strategy return
            strategy_return = np.sum(weights * next_returns) - tc
            timing_returns[date] = strategy_return
        
        # Compute tilting returns
        tilting_returns = pd.Series(index=self.factor_returns.index[self.lookback_window+1:])
        prev_weights = None
        
        for date in tilting_dates:
            # Get index of current date
            try:
                date_idx = list(self.factor_returns.index).index(date)
            except ValueError:
                # Skip if date not found in factor returns
                continue
                
            # Get factor returns for the next period
            if date_idx + 1 >= len(self.factor_returns.index):
                continue
                
            next_date = self.factor_returns.index[date_idx + 1]
            next_returns = self.factor_returns.loc[next_date]
            
            # Get current weights
            weights = self.weights_history['tilting'][date].values
            
            # Calculate transaction costs
            tc = 0
            if transaction_costs and prev_weights is not None:
                weight_change = np.abs(weights - prev_weights).sum()
                tc = weight_change * self.transaction_cost / 2  # One-way cost
            
            prev_weights = weights
            
            # Calculate strategy return
            strategy_return = np.sum(weights * next_returns) - tc
            tilting_returns[date] = strategy_return
        
        # Store returns
        self.returns_history = {
            'timing': timing_returns,
            'tilting': tilting_returns,
            'benchmark': benchmark_returns
        }
        
        return self.returns_history
    
    def compute_performance_stats(self):
        """
        Compute performance statistics for the strategies
        
        Returns:
        --------
        dict
            Dictionary with performance statistics
        """
        if not self.returns_history['benchmark'].any():
            return None
        
        stats = {}
        
        for strategy in ['benchmark', 'timing', 'tilting']:
            returns = self.returns_history[strategy].dropna()
            
            if len(returns) == 0:
                continue
                
            # Annualized return
            ann_return = returns.mean() * 12
            
            # Annualized volatility
            ann_vol = returns.std() * np.sqrt(12)
            
            # Sharpe ratio
            sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0
            
            # Maximum drawdown
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max - 1)
            max_drawdown = drawdown.min()
            
            # Tracking error and information ratio for active strategies
            if strategy != 'benchmark':
                aligned_returns = returns.align(self.returns_history['benchmark'], join='inner')[0]
                aligned_benchmark = self.returns_history['benchmark'].align(returns, join='inner')[0]
                
                tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(12)
                information_ratio = ((aligned_returns - aligned_benchmark).mean() * 12) / tracking_error if tracking_error > 0 else 0
                
                # t-statistic for information ratio
                t_stat = information_ratio * np.sqrt(len(aligned_returns))
                
                # Turnover
                turnover = 0
                prev_weights = None
                
                for date in sorted(self.weights_history[strategy].keys()):
                    weights = self.weights_history[strategy][date]
                    
                    if prev_weights is not None:
                        turnover += np.abs(weights - prev_weights).sum() / 2  # One-way turnover
                    
                    prev_weights = weights
                
                # Annualize turnover
                ann_turnover = turnover / (len(returns) / 12)
                
                stats[strategy] = {
                    'annualized_return': ann_return,
                    'annualized_volatility': ann_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'maximum_drawdown': max_drawdown,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    't_statistic': t_stat,
                    'annualized_turnover': ann_turnover
                }
            else:
                stats[strategy] = {
                    'annualized_return': ann_return,
                    'annualized_volatility': ann_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'maximum_drawdown': max_drawdown
                }
        
        return stats
    
    def apply_smoothing_methods(self, smoothing_methods):
        """
        Apply smoothing methods to factor allocations
        
        Parameters:
        -----------
        smoothing_methods : list
            List of smoothing methods to apply ('constraints', 'black_litterman', 'tc_penalty')
            
        Returns:
        --------
        dict
            Dictionary with smoothed strategy returns
        """
        smoothed_weights = {'timing': {}, 'tilting': {}}
        
        for strategy in ['timing', 'tilting']:
            dates = sorted(self.weights_history[strategy].keys())
            prev_weights = None
            
            for date in dates:
                weights = self.weights_history[strategy][date].copy()
                
                # Apply constraints (long-only, 10% cap)
                if 'constraints' in smoothing_methods:
                    weights = np.maximum(weights, 0)  # Long-only
                    weights = np.minimum(weights, 0.1)  # 10% cap
                    
                    # Re-normalize
                    if np.sum(weights) > 0:
                        weights = weights / np.sum(weights)
                
                # Apply Black-Litterman shrinkage
                if 'black_litterman' in smoothing_methods:
                    if prev_weights is not None:
                        # Use previous weights as anchor (prior)
                        tau = 0.5  # Shrinkage intensity
                        weights = tau * weights + (1 - tau) * prev_weights
                    else:
                        # Use benchmark weights as anchor if no previous weights
                        tau = 0.5
                        weights = tau * weights + (1 - tau) * self.benchmark_weights
                
                # Apply transaction cost penalty
                if 'tc_penalty' in smoothing_methods and prev_weights is not None:
                    # Penalize weight changes
                    penalty = 0.3  # Penalty strength
                    weight_change = weights - prev_weights
                    
                    # Shrink changes based on penalty
                    adjusted_change = weight_change * (1 - penalty)
                    weights = prev_weights + adjusted_change
                    
                    # Re-normalize
                    if np.sum(weights) > 0:
                        weights = weights / np.sum(weights)
                
                # Store smoothed weights
                smoothed_weights[strategy][date] = pd.Series(weights, index=self.factor_returns.columns)
                prev_weights = weights
        
        # Save original weights
        original_weights = self.weights_history.copy()
        
        # Replace with smoothed weights
        self.weights_history = smoothed_weights
        
        # Compute returns with smoothed weights
        smoothed_returns = self.compute_strategy_returns()
        
        # Restore original weights
        self.weights_history = original_weights
        
        return smoothed_returns

#########################################################################
# Main Function
#########################################################################

def main():
    # Parameters
    start_date = "20000101"  # January 1, 2000
    end_date = "20161231"    # December 31, 2016
    lookback_window = 60     # 5 years of monthly data for estimation
    risk_aversion = 5
    tracking_error_target = 0.025  # 2.5% annualized tracking error
    transaction_cost = 0.002  # 20 bps for factor allocation turnover
    
    # Create output directory for saving results
    output_dir = "factor_timing_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting Equity Factor Timing and Tilting Strategy...")
    print(f"Period: {start_date} to {end_date}")
    
    # Set simulation mode to True since Bloomberg isn't working correctly
    simulation_mode = True
    
    # Convert dates to simulation-friendly format
    start_date_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    end_date_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
    
    try:
        # Create simulator
        simulator = SimulatedMarketDataGenerator(start_date_fmt, end_date_fmt)
        
        # 1. Generate simulated factor returns
        print("Generating simulated factor returns...")
        factor_returns = simulator.generate_factor_returns()
        
        print("\nFactor Returns Summary:")
        print(factor_returns.describe())
        
        # Save factor returns
        factor_returns.to_csv(os.path.join(output_dir, "factor_returns.csv"))
        
        # 2. Generate fundamental predictors for factor timing
        print("Generating fundamental predictors...")
        fundamental_predictors = simulator.generate_fundamental_predictors()
        
        # Standardize predictors
        standardized_predictors = fundamental_predictors.copy()
        for col in standardized_predictors.columns:
            rolling_mean = standardized_predictors[col].rolling(window=12, min_periods=1).mean()
            rolling_std = standardized_predictors[col].rolling(window=12, min_periods=1).std()
            standardized_predictors[col] = (standardized_predictors[col] - rolling_mean) / rolling_std.replace(0, 1)
            standardized_predictors[col] = standardized_predictors[col].clip(-5, 5)
        
        # Run PCA on fundamental predictors
        pca = PCA(n_components=1)
        pca_fundamental = pd.DataFrame(
            pca.fit_transform(standardized_predictors.fillna(0)),
            index=standardized_predictors.index,
            columns=['FUN1']
        )
        print(f"Explained variance by FUN1: {pca.explained_variance_ratio_[0]:.2f}")
        
        # 3. Generate technical indicators for factor timing
        print("Generating technical indicators...")
        technical_indicators = simulator.generate_technical_indicators(factor_returns)
        
        # Run PCA on technical indicators for each factor
        pca_technical = {}
        for factor, indicators in technical_indicators.items():
            if indicators.empty:
                continue
                
            # Fill NaN values with 0
            indicators_filled = indicators.fillna(0)
            
            # Run PCA if we have enough data
            if indicators_filled.shape[1] > 0:
                pca = PCA(n_components=1)
                pca_result = pca.fit_transform(indicators_filled)
                pca_technical[factor] = pd.DataFrame(pca_result, index=indicators.index, columns=['TECH1'])
        
        # 4. Generate factor characteristics for tilting
        print("Generating factor characteristics...")
        factor_characteristics = simulator.generate_factor_characteristics(factor_returns)
        
        # 5. Implement parametric portfolio policies
        print("Implementing parametric portfolio policies...")
        policy = ParametricPortfolioPolicy(
            factor_returns, 
            lookback_window=lookback_window,
            risk_aversion=risk_aversion,
            tracking_error_target=tracking_error_target,
            transaction_cost=transaction_cost
        )
        
        # 5a. Factor timing
        timing_weights = policy.factor_timing((pca_fundamental, pca_technical))
        print("\nFactor timing weights:")
        print(timing_weights)
        
        # Save timing weights
        timing_weights.to_csv(os.path.join(output_dir, "timing_weights.csv"))
        
        # 5b. Factor tilting
        tilting_weights = policy.factor_tilting(factor_characteristics)
        print("\nFactor tilting weights:")
        print(tilting_weights)
        
        # Save tilting weights
        tilting_weights.to_csv(os.path.join(output_dir, "tilting_weights.csv"))
        
        # 6. Compute strategy returns
        strategy_returns = policy.compute_strategy_returns()
        
        # Save strategy returns
        pd.DataFrame(strategy_returns).to_csv(os.path.join(output_dir, "strategy_returns.csv"))
        
        # 7. Compute performance statistics
        performance_stats = policy.compute_performance_stats()
        
        if performance_stats:
            print("\nPerformance Statistics:")
            print("-" * 50)
            
            for strategy, stats in performance_stats.items():
                print(f"\n{strategy.capitalize()} Strategy:")
                for metric, value in stats.items():
                    print(f"  {metric}: {value:.4f}")
            
            # Save performance stats
            with open(os.path.join(output_dir, "performance_stats.txt"), "w") as f:
                for strategy, stats in performance_stats.items():
                    f.write(f"\n{strategy.capitalize()} Strategy:\n")
                    for metric, value in stats.items():
                        f.write(f"  {metric}: {value:.4f}\n")
        
        # 8. Apply smoothing methods
        print("\nApplying smoothing methods...")
        smoothing_methods = ['constraints', 'black_litterman', 'tc_penalty']
        smoothed_returns = policy.apply_smoothing_methods(smoothing_methods)
        
        # Save smoothed returns
        pd.DataFrame(smoothed_returns).to_csv(os.path.join(output_dir, "smoothed_returns.csv"))
        
        # Save original weights history
        original_weights = policy.weights_history
        
        # Replace with smoothed weights to compute statistics
        policy.weights_history = {'timing': policy.weights_history['timing'], 'tilting': policy.weights_history['tilting']}
        policy.returns_history = smoothed_returns
        
        smoothed_stats = policy.compute_performance_stats()
        
        if smoothed_stats:
            print("\nPerformance Statistics with Smoothing:")
            print("-" * 50)
            
            for strategy, stats in smoothed_stats.items():
                if strategy == 'benchmark':
                    continue
                    
                print(f"\n{strategy.capitalize()} Strategy (Smoothed):")
                for metric, value in stats.items():
                    print(f"  {metric}: {value:.4f}")
            
            # Save smoothed performance stats
            with open(os.path.join(output_dir, "smoothed_performance_stats.txt"), "w") as f:
                for strategy, stats in smoothed_stats.items():
                    if strategy == 'benchmark':
                        continue
                    f.write(f"\n{strategy.capitalize()} Strategy (Smoothed):\n")
                    for metric, value in stats.items():
                        f.write(f"  {metric}: {value:.4f}\n")
        
        # 9. Plot results
        print("\nGenerating plots...")
        plot_results(strategy_returns, performance_stats, factor_returns, output_dir)
        
        print(f"\nResults saved to {output_dir} directory")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

def plot_results(strategy_returns, performance_stats, factor_returns, output_dir):
    """Plot strategy results"""
    try:
        # Plot cumulative returns
        plt.figure(figsize=(12, 8))
        
        for strategy, returns in strategy_returns.items():
            if len(returns) > 0:
                cum_returns = (1 + returns.fillna(0)).cumprod()
                plt.plot(cum_returns.index, cum_returns, label=strategy.capitalize())
        
        plt.title('Cumulative Strategy Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'strategy_returns.png'))
        
        # Plot factor correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = factor_returns.corr()
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Factor Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'factor_correlations.png'))
        
        # Plot performance metrics
        plt.figure(figsize=(12, 8))
        metrics = ['annualized_return', 'sharpe_ratio']
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, 2, i+1)
            
            strategy_names = []
            metric_values = []
            
            for strategy, stats in performance_stats.items():
                if metric in stats:
                    strategy_names.append(strategy.capitalize())
                    metric_values.append(stats[metric])
            
            plt.bar(strategy_names, metric_values)
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.ylabel('Value')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
        
    except Exception as e:
        print(f"Error in plot_results: {e}")

if __name__ == "__main__":
    main()