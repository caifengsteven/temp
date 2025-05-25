#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SLOPE: Sparse Index Clones via the Sorted L1-Norm

Implementation based on the paper "Sparse index clones via the sorted 1-Norm"
by Philipp J. Kremer, Damian Brzyski, Małgorzata Bogdan, and Sandra Paterlini
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import cvxpy as cp
import logging
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import seaborn as sns
from scipy.stats import norm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the Bloomberg API if available
try:
    import blpapi
    BLOOMBERG_AVAILABLE = True
except ImportError:
    logger.warning("Bloomberg API (blpapi) not available. Will use fallback or synthetic data.")
    BLOOMBERG_AVAILABLE = False


class SLOPE:
    """
    Sorted L1 Penalized Estimator (SLOPE) for index tracking.
    
    This class implements the SLOPE methodology for creating sparse index clones
    as described in Kremer et al. (2022).
    """
    
    def __init__(self, long_only: bool = True, bounds: Optional[Tuple[float, float]] = None):
        """
        Initialize the SLOPE model.
        
        Args:
            long_only: If True, constrains the weights to be non-negative
            bounds: Optional tuple (lower_bound, upper_bound) for weight constraints
        """
        self.long_only = long_only
        self.bounds = bounds or (0.0, None) if long_only else (None, None)
        self.weights = None
        self.lambda_seq = None
        self.tracking_error = None
        self.groups = None
        self.partial_correlations = None
    
    def _generate_lambda_sequence(self, n_assets: int, alpha: float = 0.05, q: float = 0.1) -> np.ndarray:
        """
        Generate a decreasing sequence of lambda parameters for SLOPE.
        
        Args:
            n_assets: Number of assets
            alpha: Scaling parameter for the sequence
            q: Parameter that regulates how fast the sequence decreases
            
        Returns:
            Array of lambda values in decreasing order
        """
        # Following Bogdan et al. (2013) approach
        # λi = α * Φ^-1(1 - qi), where qi = i * q/(2K)
        lambda_seq = np.zeros(n_assets)
        for i in range(n_assets):
            qi = (i + 1) * q / (2 * n_assets)
            lambda_seq[i] = alpha * norm.ppf(1 - qi)
        
        # Ensure the sequence is non-increasing
        lambda_seq = np.sort(lambda_seq)[::-1]
        
        return lambda_seq
    
    def fit(self, 
            Y: np.ndarray, 
            R: np.ndarray, 
            lambda_seq: Optional[np.ndarray] = None, 
            alpha: float = 0.05, 
            q: float = 0.1) -> 'SLOPE':
        """
        Fit the SLOPE model to track the benchmark returns.
        
        Args:
            Y: Vector of benchmark returns (T x 1)
            R: Matrix of constituent returns (T x K)
            lambda_seq: Sequence of lambda parameters (if None, generated automatically)
            alpha: Scaling parameter for the lambda sequence (used if lambda_seq is None)
            q: Parameter that regulates how fast the sequence decreases (used if lambda_seq is None)
            
        Returns:
            Self with fitted weights
        """
        T, K = R.shape
        
        # Generate lambda sequence if not provided
        if lambda_seq is None:
            lambda_seq = self._generate_lambda_sequence(K, alpha, q)
        self.lambda_seq = lambda_seq
        
        # Instead of directly implementing SLOPE, which is complex in CVXPY,
        # we'll use a simpler approach for LASSO and approximate SLOPE behavior
        if np.all(lambda_seq == lambda_seq[0]):
            # LASSO case: all lambda values are the same
            w = cp.Variable(K)
            objective = cp.Minimize(0.5 * cp.sum_squares(Y - R @ w) + lambda_seq[0] * cp.norm(w, 1))
            
            # Set constraints
            constraints = [cp.sum(w) == 1]  # Budget constraint
            
            if self.long_only:
                constraints.append(w >= 0)
            elif self.bounds[0] is not None or self.bounds[1] is not None:
                # Apply specified bounds
                if self.bounds[0] is not None:
                    constraints.append(w >= self.bounds[0])
                if self.bounds[1] is not None:
                    constraints.append(w <= self.bounds[1])
            
            # Solve the problem
            problem = cp.Problem(objective, constraints)
            try:
                problem.solve(solver=cp.ECOS)
            except cp.error.SolverError as e:
                logger.error(f"Solver error: {e}")
                return self
                
        else:
            # For SLOPE, use an alternating optimization approach
            # Initialize weights using LASSO
            w_lasso = cp.Variable(K)
            objective_lasso = cp.Minimize(0.5 * cp.sum_squares(Y - R @ w_lasso) + lambda_seq[0] * cp.norm(w_lasso, 1))
            
            constraints_lasso = [cp.sum(w_lasso) == 1]
            if self.long_only:
                constraints_lasso.append(w_lasso >= 0)
                
            problem_lasso = cp.Problem(objective_lasso, constraints_lasso)
            problem_lasso.solve(solver=cp.ECOS)
            
            # Initialize weights
            w_current = w_lasso.value
            
            # Use iterative soft thresholding for SLOPE
            max_iter = 100
            tol = 1e-6
            
            for iter_idx in range(max_iter):
                # Sort weights by magnitude
                sorted_indices = np.argsort(np.abs(w_current))[::-1]
                
                # Assign lambda values by the sorted order
                lambda_ordered = np.zeros(K)
                for i, idx in enumerate(sorted_indices):
                    lambda_ordered[idx] = lambda_seq[i]
                
                # Solve weighted LASSO with the ordered lambda values
                w = cp.Variable(K)
                objective = cp.Minimize(0.5 * cp.sum_squares(Y - R @ w) + cp.sum(cp.multiply(lambda_ordered, cp.abs(w))))
                
                constraints = [cp.sum(w) == 1]
                if self.long_only:
                    constraints.append(w >= 0)
                    
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.ECOS)
                
                # Check convergence
                if np.linalg.norm(w.value - w_current) < tol:
                    break
                    
                w_current = w.value
        
        # Store results
        if problem.status != cp.OPTIMAL:
            logger.warning(f"Problem status: {problem.status}")
            if problem.status == cp.INFEASIBLE:
                logger.error("Problem is infeasible")
                return self
        
        self.weights = w.value
        self.tracking_error = np.sqrt(np.mean((Y - R @ self.weights) ** 2))
        
        # Identify groups and calculate partial correlations
        self._identify_groups_and_partial_correlations(Y, R)
        
        return self
    
    def _identify_groups_and_partial_correlations(self, Y: np.ndarray, R: np.ndarray) -> None:
        """
        Identify groups of assets with the same weights and calculate partial correlations.
        
        Args:
            Y: Vector of benchmark returns
            R: Matrix of constituent returns
        """
        # Round weights to avoid floating point comparison issues
        rounded_weights = np.round(self.weights, 6)
        
        # Identify unique weights and their groups
        unique_weights = np.unique(rounded_weights)
        groups = {}
        
        for weight in unique_weights:
            if weight != 0:  # Ignore zero weights
                indices = np.where(rounded_weights == weight)[0]
                groups[weight] = indices.tolist()
        
        self.groups = groups
        
        # Calculate partial correlations for each asset
        partial_correlations = {}
        
        for group_weight, asset_indices in groups.items():
            for asset_idx in asset_indices:
                # Remove current asset and assets with the same weight from R
                other_assets_in_group = [idx for idx in asset_indices if idx != asset_idx]
                assets_to_remove = [asset_idx] + other_assets_in_group
                
                # Create mask for retained assets
                mask = np.ones(R.shape[1], dtype=bool)
                mask[assets_to_remove] = False
                
                if np.any(mask):  # Check if we have any assets left
                    # Solve for weights using retained assets
                    R_subset = R[:, mask]
                    
                    # Calculate residuals: Y - R_subset * w_subset
                    try:
                        w_subset, *_ = np.linalg.lstsq(R_subset, Y, rcond=None)
                        residuals = Y - R_subset @ w_subset
                        
                        # Calculate correlation between current asset and residuals
                        asset_returns = R[:, asset_idx]
                        partial_corr = np.corrcoef(asset_returns, residuals)[0, 1]
                        partial_correlations[asset_idx] = partial_corr
                    except np.linalg.LinAlgError:
                        partial_correlations[asset_idx] = 0
                else:
                    # If no assets are left after removal, correlation is 1
                    partial_correlations[asset_idx] = 1
        
        self.partial_correlations = partial_correlations
    
    def select_groups(self, percentile: float = 75) -> np.ndarray:
        """
        Select groups based on their median partial correlation with the index.
        
        Args:
            percentile: Percentile threshold for group selection
            
        Returns:
            Array of weights with only selected groups active
        """
        if self.groups is None or self.partial_correlations is None:
            logger.error("No groups or partial correlations available. Run fit() first.")
            return None
        
        # Calculate median partial correlation for each group
        group_median_pc = {}
        for group_weight, asset_indices in self.groups.items():
            pcs = [self.partial_correlations.get(idx, 0) for idx in asset_indices]
            group_median_pc[group_weight] = np.median(pcs)
        
        # Determine threshold for selection
        threshold = np.percentile(list(group_median_pc.values()), percentile)
        
        # Create new weights with only selected groups
        selected_weights = np.zeros_like(self.weights)
        
        for group_weight, median_pc in group_median_pc.items():
            if median_pc >= threshold:
                asset_indices = self.groups[group_weight]
                selected_weights[asset_indices] = group_weight
        
        # Rescale to ensure sum equals 1
        if np.sum(selected_weights) > 0:
            selected_weights = selected_weights / np.sum(selected_weights)
        
        return selected_weights


class IndicesReplicator:
    """
    Class for replicating indices using SLOPE and other methods.
    """
    
    def __init__(self):
        """Initialize the replicator."""
        self.data = None
        self.tickers = None
        self.index_returns = None
        self.constituent_returns = None
        self.weights = None
        
    def fetch_from_bloomberg(self, 
                           index_ticker: str, 
                           start_date: dt.datetime,
                           end_date: dt.datetime) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fetch index and constituent returns data from Bloomberg.
        
        Args:
            index_ticker: Bloomberg ticker for the index
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Tuple of (index_returns, constituent_returns, tickers)
        """
        if not BLOOMBERG_AVAILABLE:
            logger.error("Bloomberg API not available. Cannot fetch data.")
            return None, None, None
        
        logger.info(f"Connecting to Bloomberg to fetch data for {index_ticker}")
        
        try:
            # Initialize Bloomberg session
            session_options = blpapi.SessionOptions()
            session_options.setServerHost("localhost")
            session_options.setServerPort(8194)
            session = blpapi.Session(session_options)
            
            # Start session
            if not session.start():
                logger.error("Failed to start Bloomberg session")
                return None, None, None
                
            # Open reference data service
            if not session.openService("//blp/refdata"):
                logger.error("Failed to open //blp/refdata service")
                session.stop()
                return None, None, None
                
            refdata_service = session.getService("//blp/refdata")
            
            # Step 1: Get index members
            logger.info(f"Retrieving constituents for {index_ticker}")
            
            request = refdata_service.createRequest("ReferenceDataRequest")
            request.append("securities", index_ticker)
            request.append("fields", "INDX_MEMBERS")
            
            session.sendRequest(request)
            
            constituents = []
            
            while True:
                event = session.nextEvent(500)
                
                for msg in event:
                    if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                        security_data = msg.getElement("securityData")
                        
                        if security_data.hasElement("fieldData"):
                            field_data = security_data.getElement("fieldData")
                            
                            if field_data.hasElement("INDX_MEMBERS"):
                                members = field_data.getElement("INDX_MEMBERS")
                                
                                for i in range(members.numValues()):
                                    member = members.getValue(i)
                                    ticker = member.getElementAsString("Member Ticker and Exchange Code")
                                    constituents.append(ticker)
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            if not constituents:
                logger.error(f"No constituents found for {index_ticker}")
                session.stop()
                return None, None, None
            
            logger.info(f"Found {len(constituents)} constituents for {index_ticker}")
            
            # Step 2: Get historical price data for index
            logger.info(f"Fetching historical data for {index_ticker}")
            
            index_prices = {}
            
            request = refdata_service.createRequest("HistoricalDataRequest")
            request.getElement("securities").appendValue(index_ticker)
            request.getElement("fields").appendValue("PX_LAST")
            request.set("periodicitySelection", "DAILY")
            request.set("startDate", start_date.strftime("%Y%m%d"))
            request.set("endDate", end_date.strftime("%Y%m%d"))
            
            session.sendRequest(request)
            
            while True:
                event = session.nextEvent(500)
                
                for msg in event:
                    if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                        security_data = msg.getElement("securityData")
                        field_data = security_data.getElement("fieldData")
                        
                        for i in range(field_data.numValues()):
                            field_value = field_data.getValue(i)
                            date = field_value.getElementAsDatetime("date").date()
                            price = field_value.getElementAsFloat("PX_LAST")
                            index_prices[date] = price
                
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
            
            # Step 3: Get historical price data for constituents
            logger.info(f"Fetching historical data for {len(constituents)} constituents")
            
            constituent_prices = {ticker: {} for ticker in constituents}
            
            for ticker in tqdm(constituents, desc="Fetching constituent data"):
                request = refdata_service.createRequest("HistoricalDataRequest")
                request.getElement("securities").appendValue(ticker)
                request.getElement("fields").appendValue("PX_LAST")
                request.set("periodicitySelection", "DAILY")
                request.set("startDate", start_date.strftime("%Y%m%d"))
                request.set("endDate", end_date.strftime("%Y%m%d"))
                
                session.sendRequest(request)
                
                try:
                    while True:
                        event = session.nextEvent(500)
                        
                        for msg in event:
                            if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                                security_data = msg.getElement("securityData")
                                
                                # Check for security errors
                                if security_data.hasElement("securityError"):
                                    break
                                
                                field_data = security_data.getElement("fieldData")
                                
                                for i in range(field_data.numValues()):
                                    field_value = field_data.getValue(i)
                                    date = field_value.getElementAsDatetime("date").date()
                                    price = field_value.getElementAsFloat("PX_LAST")
                                    constituent_prices[ticker][date] = price
                        
                        if event.eventType() == blpapi.Event.RESPONSE:
                            break
                except Exception as e:
                    logger.warning(f"Error fetching data for {ticker}: {e}")
                    continue
            
            # Step 4: Convert prices to returns
            # Create a DataFrame with all price data
            all_dates = sorted(set().union(*[set(d.keys()) for d in [index_prices] + list(constituent_prices.values())]))
            
            # Create a DataFrame for the index
            index_df = pd.DataFrame(index=all_dates)
            index_df['index_price'] = pd.Series(index_prices)
            index_df['index_return'] = index_df['index_price'].pct_change()
            
            # Create a DataFrame for constituents
            constituent_df = pd.DataFrame(index=all_dates)
            
            valid_tickers = []
            
            for ticker in constituents:
                prices = constituent_prices[ticker]
                if prices:  # Only include tickers with data
                    constituent_df[f'{ticker}_price'] = pd.Series(prices)
                    constituent_df[f'{ticker}_return'] = constituent_df[f'{ticker}_price'].pct_change()
                    valid_tickers.append(ticker)
            
            # Remove first row (NaN returns) and drop price columns
            index_df = index_df.iloc[1:]
            constituent_df = constituent_df.iloc[1:]
            
            index_returns = index_df['index_return'].values
            
            # Extract return columns
            return_cols = [col for col in constituent_df.columns if col.endswith('_return')]
            constituent_returns = constituent_df[return_cols].values
            
            # Clean tickers (remove _return suffix)
            clean_tickers = [col.replace('_return', '') for col in return_cols]
            
            # Close the session
            session.stop()
            
            # Remove rows with NaN values
            valid_rows = ~np.isnan(index_returns) & ~np.isnan(constituent_returns).any(axis=1)
            
            index_returns = index_returns[valid_rows]
            constituent_returns = constituent_returns[valid_rows]
            
            logger.info(f"Successfully fetched {len(index_returns)} days of data for {len(valid_tickers)} constituents")
            
            return index_returns, constituent_returns, clean_tickers
            
        except Exception as e:
            logger.error(f"Error fetching Bloomberg data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Try to close the session if it exists
            if 'session' in locals():
                try:
                    session.stop()
                except:
                    pass
                    
            return None, None, None
        
    def generate_synthetic_data(self, 
                             n_stocks: int = 100, 
                             n_days: int = 500, 
                             n_factors: int = 3, 
                             group_structure: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate synthetic data for testing.
        
        Args:
            n_stocks: Number of stocks
            n_days: Number of days
            n_factors: Number of risk factors
            group_structure: If True, create groups of stocks with similar factor exposures
            
        Returns:
            Tuple of (index_returns, constituent_returns, tickers)
        """
        logger.info(f"Generating synthetic data with {n_stocks} stocks over {n_days} days")
        
        # Generate synthetic tickers
        tickers = [f"STOCK{i+1}" for i in range(n_stocks)]
        
        # Step 1: Generate factor returns
        np.random.seed(42)
        F = np.random.normal(0, 0.01, size=(n_days, n_factors))
        
        # Step 2: Generate loadings matrix
        if group_structure:
            # Create groups with similar factor exposures
            n_groups = min(n_factors, 3)  # Limit to 3 groups for simplicity
            stocks_per_group = n_stocks // n_groups
            
            B = np.zeros((n_factors, n_stocks))
            
            for g in range(n_groups):
                # Create a distinctive loading pattern for this group
                group_loading = np.random.uniform(0.5, 1.5, size=n_factors)
                
                # Add some noise to each stock in the group
                start_idx = g * stocks_per_group
                end_idx = (g + 1) * stocks_per_group if g < n_groups - 1 else n_stocks
                
                for i in range(start_idx, end_idx):
                    B[:, i] = group_loading + np.random.normal(0, 0.05, size=n_factors)
        else:
            # Random factor loadings for each stock
            B = np.random.normal(0, 1, size=(n_factors, n_stocks))
        
        # Step 3: Generate stock returns with error terms
        epsilon = np.random.normal(0, 0.02, size=(n_days, n_stocks))
        R = F @ B + epsilon
        
        # Step 4: Generate index returns as weighted sum of constituent returns
        # Use equal weights for simplicity
        w_true = np.ones(n_stocks) / n_stocks
        Y = R @ w_true + np.random.normal(0, 0.001, size=n_days)  # Add small tracking error
        
        logger.info(f"Generated synthetic data with shape R: {R.shape}, Y: {Y.shape}")
        
        return Y, R, tickers
    
    def rolling_window_backtest(self, 
                              index_returns: np.ndarray, 
                              constituent_returns: np.ndarray,
                              tickers: List[str],
                              window_size: int = 252,
                              rebalance_freq: int = 21,
                              methods: List[str] = ['SLOPE', 'SLOPE-SLC', 'LASSO']):
        """
        Perform a rolling window backtest of different replication methods.
        
        Args:
            index_returns: Vector of index returns
            constituent_returns: Matrix of constituent returns
            tickers: List of ticker symbols
            window_size: Size of the rolling window
            rebalance_freq: Rebalancing frequency in days
            methods: List of methods to test
            
        Returns:
            Dictionary of replication results
        """
        if len(index_returns) <= window_size:
            logger.error(f"Not enough data for backtest. Need more than {window_size} days.")
            return None
        
        T = len(index_returns)
        K = constituent_returns.shape[1]
        
        # Initialize results
        results = {
            method: {
                'weights': np.zeros((T - window_size, K)),
                'oos_returns': np.zeros(T - window_size),
                'tracking_errors': np.zeros(T - window_size),
                'active_positions': np.zeros(T - window_size, dtype=int),
                'turnover': np.zeros(T - window_size)
            } for method in methods
        }
        
        # Prepare alpha values for lambda sequence
        alphas = {
            'SLOPE': 0.05,
            'SLOPE-SLC': 0.05,
            'LASSO': 0.05
        }
        
        # For tracking the last rebalance day
        last_rebalance = -1
        
        # Rolling window backtest
        for t in tqdm(range(window_size, T), desc="Rolling window backtest"):
            # Check if we need to rebalance
            if t - last_rebalance >= rebalance_freq:
                last_rebalance = t
                
                # Training data
                Y_train = index_returns[t - window_size:t]
                R_train = constituent_returns[t - window_size:t]
                
                # Standardize returns
                scaler = StandardScaler()
                R_train_scaled = scaler.fit_transform(R_train)
                
                # Fit models
                for method in methods:
                    if method == 'SLOPE' or method == 'SLOPE-SLC':
                        # Fit SLOPE model
                        model = SLOPE(long_only=True)
                        model.fit(Y_train, R_train_scaled, alpha=alphas[method])
                        
                        if method == 'SLOPE-SLC':
                            # Apply group selection
                            weights = model.select_groups(percentile=75)
                        else:
                            weights = model.weights
                    
                    elif method == 'LASSO':
                        # Special case for LASSO: use SLOPE with constant lambda sequence
                        lambda_seq = np.ones(K) * alphas['LASSO']
                        model = SLOPE(long_only=True)
                        model.fit(Y_train, R_train_scaled, lambda_seq=lambda_seq)
                        weights = model.weights
                    
                    # Apply threshold for practical implementation
                    weights[np.abs(weights) < 0.0005] = 0
                    
                    # Normalize to ensure weights sum to 1
                    if np.sum(weights) > 0:
                        weights = weights / np.sum(weights)
                    
                    # Store weights
                    results[method]['weights'][t - window_size] = weights
                    
                    # Calculate turnover (for days after the first rebalance)
                    if t > window_size:
                        prev_weights = results[method]['weights'][t - window_size - 1]
                        results[method]['turnover'][t - window_size] = np.sum(np.abs(weights - prev_weights))
                    
                    # Count active positions
                    results[method]['active_positions'][t - window_size] = np.sum(weights > 0)
            
            else:
                # If not rebalancing, use the previous weights
                for method in methods:
                    results[method]['weights'][t - window_size] = results[method]['weights'][t - window_size - 1]
                    
                    # Calculate turnover (always 0 for non-rebalancing days)
                    if t > window_size:
                        results[method]['turnover'][t - window_size] = 0
                    
                    # Count active positions (same as previous day)
                    results[method]['active_positions'][t - window_size] = results[method]['active_positions'][t - window_size - 1]
            
            # Calculate out-of-sample returns and tracking errors for all methods
            R_oos = constituent_returns[t]
            Y_oos = index_returns[t]
            
            for method in methods:
                weights = results[method]['weights'][t - window_size]
                clone_return = np.dot(R_oos, weights)
                results[method]['oos_returns'][t - window_size] = clone_return
                results[method]['tracking_errors'][t - window_size] = Y_oos - clone_return
        
        # Calculate summary statistics
        summary = {}
        
        for method in methods:
            oos_returns = results[method]['oos_returns']
            tracking_errors = results[method]['tracking_errors']
            active_positions = results[method]['active_positions']
            turnover = results[method]['turnover']
            
            summary[method] = {
                'Annualized Tracking Error Volatility (%)': np.std(tracking_errors) * np.sqrt(252) * 100,
                'Annualized Tracking Error (%)': np.mean(tracking_errors) * 252 * 100,
                'Information Ratio': np.mean(tracking_errors) / np.std(tracking_errors) * np.sqrt(252),
                'Average Active Positions': np.mean(active_positions),
                'Average Turnover (%)': np.mean(turnover) * 100,
                'Correlation': np.corrcoef(oos_returns, index_returns[window_size:])[0, 1],
                'Max Drawdown (%)': self._calculate_max_drawdown(oos_returns) * 100
            }
        
        return results, summary
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate the maximum drawdown from a series of returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown as a decimal
        """
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdowns
        drawdowns = (cum_returns - running_max) / running_max
        
        return np.min(drawdowns)
    
    def plot_results(self, results: Dict, index_returns: np.ndarray, window_size: int, methods: List[str]):
        """
        Plot the results of the backtest.
        
        Args:
            results: Results dictionary from rolling_window_backtest
            index_returns: Vector of index returns
            window_size: Size of the rolling window used in backtest
            methods: List of methods to plot
        """
        # Create cumulative return series
        cum_index = np.cumprod(1 + index_returns[window_size:])
        
        plt.figure(figsize=(12, 8))
        
        # Plot cumulative index returns
        plt.plot(cum_index, label='Index', linewidth=2, color='black')
        
        # Plot cumulative clone returns for each method
        for method in methods:
            clone_returns = results[method]['oos_returns']
            cum_clone = np.cumprod(1 + clone_returns)
            plt.plot(cum_clone, label=method, linewidth=1.5)
        
        plt.title('Cumulative Returns')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('cumulative_returns.png')
        plt.close()
        
        # Plot tracking errors
        plt.figure(figsize=(12, 8))
        
        for method in methods:
            tracking_errors = results[method]['tracking_errors']
            plt.plot(tracking_errors, label=method, alpha=0.7)
        
        plt.title('Daily Tracking Errors')
        plt.xlabel('Days')
        plt.ylabel('Tracking Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('tracking_errors.png')
        plt.close()
        
        # Plot active positions
        plt.figure(figsize=(12, 8))
        
        for method in methods:
            active_positions = results[method]['active_positions']
            plt.plot(active_positions, label=method)
        
        plt.title('Number of Active Positions')
        plt.xlabel('Days')
        plt.ylabel('Number of Positions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('active_positions.png')
        plt.close()
        
        # Plot turnover
        plt.figure(figsize=(12, 8))
        
        for method in methods:
            turnover = results[method]['turnover']
            plt.plot(turnover, label=method)
        
        plt.title('Portfolio Turnover')
        plt.xlabel('Days')
        plt.ylabel('Turnover')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('turnover.png')
        plt.close()
        
        # Plot partial correlations histogram for the last SLOPE run
        for method in methods:
            if method == 'SLOPE' and hasattr(results[method], 'partial_correlations'):
                plt.figure(figsize=(10, 6))
                
                partial_correlations = list(results[method]['partial_correlations'].values())
                plt.hist(partial_correlations, bins=20, alpha=0.7)
                
                plt.title('Distribution of Partial Correlations with Index')
                plt.xlabel('Partial Correlation')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.savefig('partial_correlations.png')
                plt.close()


def main():
    """Main function to run the SLOPE index tracking."""
    # Initialize the replicator
    replicator = IndicesReplicator()
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'bloomberg':
        # Use Bloomberg data
        if BLOOMBERG_AVAILABLE:
            logger.info("Using Bloomberg data")
            
            # Define index and date range
            index_ticker = sys.argv[2] if len(sys.argv) > 2 else "SPX Index"
            
            # Default to 5 years of data
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=5*365)
            
            # Fetch data
            index_returns, constituent_returns, tickers = replicator.fetch_from_bloomberg(
                index_ticker, start_date, end_date)
            
            if index_returns is None:
                logger.error("Failed to fetch Bloomberg data. Using synthetic data instead.")
                index_returns, constituent_returns, tickers = replicator.generate_synthetic_data()
        else:
            logger.warning("Bloomberg API not available. Using synthetic data instead.")
            index_returns, constituent_returns, tickers = replicator.generate_synthetic_data()
    else:
        # Use synthetic data
        logger.info("Using synthetic data")
        index_returns, constituent_returns, tickers = replicator.generate_synthetic_data(
            n_stocks=100, n_days=1000, group_structure=True)
    
    # Run the backtest
    methods = ['SLOPE', 'SLOPE-SLC', 'LASSO']
    results, summary = replicator.rolling_window_backtest(
        index_returns, constituent_returns, tickers, 
        window_size=252, rebalance_freq=21, methods=methods
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("===================")
    
    for method in methods:
        print(f"\n{method}:")
        for stat, value in summary[method].items():
            print(f"  {stat}: {value:.4f}")
    
    # Plot results
    replicator.plot_results(results, index_returns, window_size=252, methods=methods)
    
    logger.info("Analysis completed successfully.")


if __name__ == "__main__":
    main()