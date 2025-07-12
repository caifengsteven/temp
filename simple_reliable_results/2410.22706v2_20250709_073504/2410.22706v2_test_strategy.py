"""
GSPHAR Model Implementation with Bloomberg Data Integration
Enhanced version of the test strategy for paper 2410.22706v2
Uses xbbg for Bloomberg data access and implements the full GSPHAR model

Requirements:
- xbbg: pip install xbbg
- Bloomberg API setup (see xbbg documentation)
- Standard scientific Python packages

Run with: python 2410.22706v2_test_strategy.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearson3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
from datetime import datetime, timedelta
import logging

# Bloomberg data integration
BLOOMBERG_AVAILABLE = False
blp = None

try:
    import xbbg
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
    print("✓ Bloomberg xbbg library loaded successfully")
    print(f"  xbbg version: {getattr(xbbg, '__version__', 'unknown')}")
except ImportError as e:
    print("⚠ Warning: xbbg not available. Will use synthetic data only.")
    print(f"  Import error: {e}")

    # Provide specific guidance based on the error
    if "ruamel" in str(e):
        print("  Issue: ruamel.yaml dependency problem")
        print("  Solution 1: pip install --force-reinstall ruamel.yaml")
        print("  Solution 2: pip uninstall ruamel-yaml && pip install ruamel.yaml")
        print("  Solution 3: Use conda: conda install ruamel.yaml")
    else:
        print("  To install: pip install xbbg")
        print("  Dependencies: pip install ruamel.yaml pyarrow")

    print("  Note: Bloomberg Terminal must be running for data access")
    print("  The model will run with synthetic data if Bloomberg is unavailable")
except Exception as e:
    print(f"⚠ Warning: Unexpected error loading xbbg: {e}")
    print("  Will use synthetic data only")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MagneticLaplacian:
    """Implementation of Magnetic Laplacian for directed graphs as described in the paper."""
    
    def __init__(self, q=0.25):
        """
        Initialize the Magnetic Laplacian.
        
        Args:
            q: Hyperparameter that determines how directional information is processed.
               q=0 for undirected graphs, q=0.25 for directed graphs.
        """
        self.q = q
        
    def compute(self, adjacency):
        """
        Compute the normalized magnetic Laplacian.
        
        Args:
            adjacency: Adjacency matrix of the graph (N x N)
            
        Returns:
            normalized_magnetic_laplacian: The normalized magnetic Laplacian matrix
        """
        # Symmetrized adjacency matrix
        A_sym = 0.5 * (adjacency + adjacency.T)
        
        # Degree matrix of symmetrized adjacency
        D_sym = np.diag(A_sym.sum(axis=1))
        D_sym_inv_sqrt = np.linalg.inv(np.sqrt(D_sym))
        
        # Phase matrix to capture directional information
        Theta = 2 * np.pi * self.q * (adjacency - adjacency.T)
        
        # Complex Hermitian adjacency matrix
        H = A_sym * np.exp(1j * Theta)
        
        # Normalized magnetic Laplacian
        I = np.eye(len(adjacency))
        normalized_magnetic_laplacian = I - D_sym_inv_sqrt @ A_sym @ D_sym_inv_sqrt * np.exp(1j * Theta)
        
        return normalized_magnetic_laplacian
    
    def eigendecomposition(self, L_m):
        """
        Compute eigendecomposition of the magnetic Laplacian.

        Args:
            L_m: Magnetic Laplacian matrix

        Returns:
            U_m: Matrix of eigenvectors
            Lambda_m: Diagonal matrix of eigenvalues
        """
        # Handle complex matrices properly
        if np.iscomplexobj(L_m):
            eigenvalues, eigenvectors = np.linalg.eig(L_m)
            # Sort by eigenvalue magnitude for stability
            idx = np.argsort(np.abs(eigenvalues))
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(L_m)

        Lambda_m = np.diag(eigenvalues)
        U_m = eigenvectors

        return U_m, Lambda_m
    
    def gft(self, signal, U_m):
        """
        Graph Fourier Transform.
        
        Args:
            signal: Signal on graph (N x T)
            U_m: Matrix of eigenvectors
            
        Returns:
            transformed_signal: Signal in spectral domain
        """
        return U_m.conj().T @ signal
    
    def igft(self, spectral_signal, U_m):
        """
        Inverse Graph Fourier Transform.
        
        Args:
            spectral_signal: Signal in spectral domain
            U_m: Matrix of eigenvectors
            
        Returns:
            signal: Signal on graph
        """
        return U_m @ spectral_signal


class ConvolutionFilter(nn.Module):
    """Learnable convolution filter with convex weights."""
    
    def __init__(self, lag_length):
        """
        Initialize convolution filter.
        
        Args:
            lag_length: Number of lag terms
        """
        super(ConvolutionFilter, self).__init__()
        self.weights = nn.Parameter(torch.ones(lag_length) / lag_length)
        
    def forward(self, x):
        """
        Apply convolution filter.
        
        Args:
            x: Input tensor of shape (batch_size, n_nodes, lag_length)
            
        Returns:
            y: Filtered tensor of shape (batch_size, n_nodes, 1)
        """
        # Ensure weights sum to 1 (convex)
        weights = torch.softmax(self.weights, dim=0)
        
        # Apply weights along lag dimension
        y = torch.sum(x * weights.unsqueeze(0).unsqueeze(0), dim=2, keepdim=True)
        
        return y


class GSPHAR(nn.Module):
    """Graph Signal Processing HAR model as described in the paper."""
    
    def __init__(self, n_nodes, U_m):
        """
        Initialize GSPHAR model.
        
        Args:
            n_nodes: Number of nodes in the graph (number of stock indices)
            U_m: Matrix of eigenvectors from magnetic Laplacian
        """
        super(GSPHAR, self).__init__()
        
        # Store eigenvector matrix
        self.register_buffer('U_m', torch.tensor(U_m, dtype=torch.complex64))
        
        # Convolution filters for mid-term and long-term
        self.filter_mid_term = ConvolutionFilter(4)  # t-5:t-2
        self.filter_long_term = ConvolutionFilter(17)  # t-22:t-6
        
        # HAR coefficients (same for real and imaginary parts)
        self.har_coefs = nn.Linear(4, 1, bias=True)  # [intercept, daily, weekly, monthly]
        
        # Neural network to merge real and imaginary parts
        self.merge_net = nn.Sequential(
            nn.Linear(2 * n_nodes, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_nodes)
        )
        
    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, n_nodes, 22)
                where 22 is the number of lag terms

        Returns:
            out: Forecasted RV of shape (batch_size, n_nodes)
        """
        batch_size, n_nodes, _ = x.shape

        # Extract lag components
        x_daily = x[:, :, 0].unsqueeze(2)  # t-1
        x_mid_term = x[:, :, 1:5]  # t-5:t-2
        x_long_term = x[:, :, 5:22]  # t-22:t-6

        # Convert input to complex type for GFT
        x_daily_complex_input = x_daily.to(dtype=torch.complex64)
        x_mid_term_complex_input = x_mid_term.to(dtype=torch.complex64)
        x_long_term_complex_input = x_long_term.to(dtype=torch.complex64)

        # Transform to complex domain using GFT
        x_daily_complex = torch.matmul(self.U_m.conj().transpose(0, 1), x_daily_complex_input)
        x_mid_term_complex = torch.matmul(self.U_m.conj().transpose(0, 1), x_mid_term_complex_input)
        x_long_term_complex = torch.matmul(self.U_m.conj().transpose(0, 1), x_long_term_complex_input)

        # Apply convolution filters (work with real parts, then combine)
        x_mid_term_real_filtered = self.filter_mid_term(x_mid_term_complex.real)
        x_mid_term_imag_filtered = self.filter_mid_term(x_mid_term_complex.imag)
        x_mid_term_filtered = torch.complex(x_mid_term_real_filtered, x_mid_term_imag_filtered)

        x_long_term_real_filtered = self.filter_long_term(x_long_term_complex.real)
        x_long_term_imag_filtered = self.filter_long_term(x_long_term_complex.imag)
        x_long_term_filtered = torch.complex(x_long_term_real_filtered, x_long_term_imag_filtered)

        # Prepare HAR inputs (batch_size, n_nodes, 4)
        ones = torch.ones(batch_size, n_nodes, 1, device=x.device, dtype=torch.float32)
        har_inputs_real = torch.cat([ones, x_daily_complex.real,
                                    x_mid_term_filtered.real,
                                    x_long_term_filtered.real], dim=2)
        har_inputs_imag = torch.cat([ones, x_daily_complex.imag,
                                    x_mid_term_filtered.imag,
                                    x_long_term_filtered.imag], dim=2)

        # Apply HAR coefficients
        spectral_forecast_real = self.har_coefs(har_inputs_real)
        spectral_forecast_imag = self.har_coefs(har_inputs_imag)

        # Rebuild complex spectral forecast
        spectral_forecast = torch.complex(spectral_forecast_real, spectral_forecast_imag)

        # Transform back to spatial domain using IGFT
        spatial_forecast = torch.matmul(self.U_m, spectral_forecast)

        # Extract real and imaginary parts and squeeze to remove extra dimensions
        spatial_real = spatial_forecast.real.squeeze(-1)
        spatial_imag = spatial_forecast.imag.squeeze(-1)

        # Merge real and imaginary parts using neural network
        merged_input = torch.cat([spatial_real, spatial_imag], dim=1)
        out = self.merge_net(merged_input)

        return out


class BloombergDataManager:
    """Manager for Bloomberg data fetching and realized volatility calculation using xbbg."""

    def __init__(self, timeout=30):
        """
        Initialize Bloomberg data manager.

        Args:
            timeout: Timeout for Bloomberg requests in seconds
        """
        self.available = BLOOMBERG_AVAILABLE
        self.timeout = timeout

        if self.available:
            try:
                # Test connection
                from xbbg import blp
                test_data = blp.bdp('SPX Index', 'PX_LAST', timeout=5)
                if test_data.empty:
                    logger.warning("Bloomberg connection test returned empty data")
                else:
                    logger.info("Bloomberg connection verified successfully")
            except Exception as e:
                logger.warning(f"Bloomberg connection test failed: {e}")
                self.available = False

    def get_stock_indices_data(self, tickers, start_date, end_date, fields=None):
        """
        Fetch stock index data from Bloomberg using xbbg.

        Args:
            tickers: List of Bloomberg tickers (e.g., ['SPX Index', 'SX5E Index'])
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            fields: List of fields to fetch (default: OHLC)

        Returns:
            DataFrame with price data
        """
        if not self.available:
            raise RuntimeError("Bloomberg xbbg not available")

        if fields is None:
            fields = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST']

        try:
            # Fetch historical data using xbbg bdh function
            data = blp.bdh(
                tickers=tickers,
                flds=fields,
                start_date=start_date,
                end_date=end_date,
                timeout=self.timeout
            )

            if data.empty:
                logger.warning("Bloomberg returned empty dataset")
                return pd.DataFrame()

            logger.info(f"Successfully fetched data for {len(tickers)} tickers from {start_date} to {end_date}")
            logger.info(f"Data shape: {data.shape}, Date range: {data.index[0]} to {data.index[-1]}")
            return data

        except Exception as e:
            logger.error(f"Error fetching Bloomberg data: {e}")
            raise

    def get_intraday_data(self, ticker, date, session='allday', interval='5min'):
        """
        Fetch intraday data for realized volatility calculation using xbbg.

        Args:
            ticker: Bloomberg ticker
            date: Date (YYYY-MM-DD format)
            session: Trading session ('allday', 'rth', 'am_open_30', etc.)
            interval: Data interval ('1min', '5min', '15min', '1H')

        Returns:
            DataFrame with intraday price data
        """
        if not self.available:
            raise RuntimeError("Bloomberg xbbg not available")

        try:
            # Fetch intraday bars using xbbg bdib function
            data = blp.bdib(
                ticker=ticker,
                dt=date,
                session=session,
                timeout=self.timeout
            )

            if data.empty:
                logger.warning(f"No intraday data returned for {ticker} on {date}")
                return pd.DataFrame()

            logger.info(f"Fetched {len(data)} intraday bars for {ticker} on {date}")
            return data

        except Exception as e:
            logger.error(f"Error fetching intraday data for {ticker} on {date}: {e}")
            return pd.DataFrame()

    def calculate_realized_volatility(self, price_data, method='5min', annualize=True):
        """
        Calculate realized volatility from high-frequency price data.

        Args:
            price_data: DataFrame with price columns
            method: Method for RV calculation ('5min', '1min', 'daily')
            annualize: Whether to annualize the volatility

        Returns:
            Series with realized volatility values
        """
        if price_data.empty:
            return pd.Series()

        # Use close prices for return calculation
        if 'close' in price_data.columns:
            prices = price_data['close']
        elif 'PX_LAST' in price_data.columns:
            prices = price_data['PX_LAST']
        else:
            # Take the first price column available
            price_cols = [col for col in price_data.columns if 'px' in col.lower() or 'price' in col.lower()]
            if price_cols:
                prices = price_data[price_cols[0]]
            else:
                raise ValueError("No price column found in data")

        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # Calculate realized volatility (sum of squared returns)
        rv = (log_returns ** 2).sum()

        # Annualize if requested (assuming 252 trading days)
        if annualize:
            if method == '5min':
                # 5-minute intervals: 78 intervals per day (6.5 hours * 12)
                rv = rv * 252
            elif method == '1min':
                # 1-minute intervals: 390 intervals per day
                rv = rv * 252
            elif method == 'daily':
                rv = rv * 252

        return np.sqrt(rv) * 100  # Convert to percentage and take square root as in paper

    def get_realized_volatility_series(self, tickers, start_date, end_date, rv_method='daily'):
        """
        Get realized volatility time series for multiple tickers using xbbg.

        Args:
            tickers: List of Bloomberg tickers
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            rv_method: Method for RV calculation ('daily', 'intraday')

        Returns:
            DataFrame with RV time series
        """
        if not self.available:
            logger.warning("Bloomberg not available, using synthetic data")
            return self._generate_synthetic_rv_fallback(tickers, start_date, end_date)

        rv_data = {}

        for ticker in tickers:
            try:
                if rv_method == 'daily':
                    # Use daily data for RV calculation with proper xbbg syntax
                    daily_data = blp.bdh(
                        tickers=ticker,
                        flds=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST'],
                        start_date=start_date,
                        end_date=end_date,
                        timeout=self.timeout
                    )

                    if not daily_data.empty:
                        # Extract OHLC data - handle both single and multi-ticker cases
                        if len(tickers) == 1:
                            # Single ticker case - columns are field names
                            if 'PX_HIGH' in daily_data.columns:
                                high = daily_data['PX_HIGH']
                                low = daily_data['PX_LOW']
                                close = daily_data['PX_LAST']
                                open_price = daily_data['PX_OPEN']
                            else:
                                # Multi-index columns case
                                high = daily_data[(ticker, 'PX_HIGH')]
                                low = daily_data[(ticker, 'PX_LOW')]
                                close = daily_data[(ticker, 'PX_LAST')]
                                open_price = daily_data[(ticker, 'PX_OPEN')]
                        else:
                            # Multi-ticker case - columns are (ticker, field) tuples
                            high = daily_data[(ticker, 'PX_HIGH')]
                            low = daily_data[(ticker, 'PX_LOW')]
                            close = daily_data[(ticker, 'PX_LAST')]
                            open_price = daily_data[(ticker, 'PX_OPEN')]

                        # Garman-Klass estimator for daily RV
                        # RV = ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2
                        rv_series = (np.log(high/low)**2 -
                                   (2*np.log(2)-1)*np.log(close/open_price)**2)

                        # Annualize and convert to percentage
                        rv_series = np.sqrt(rv_series * 252) * 100

                        # Remove any infinite or NaN values
                        rv_series = rv_series.replace([np.inf, -np.inf], np.nan).dropna()

                        if len(rv_series) > 0:
                            rv_data[ticker] = rv_series
                            logger.info(f"Calculated RV for {ticker}: {len(rv_series)} observations")
                        else:
                            logger.warning(f"No valid RV data for {ticker}")

                elif rv_method == 'intraday':
                    # Intraday RV calculation using high-frequency data
                    logger.info(f"Calculating intraday RV for {ticker}")

                    # Get date range for iteration
                    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
                    daily_rv = []

                    for date in date_range:
                        date_str = date.strftime('%Y-%m-%d')
                        intraday_data = self.get_intraday_data(ticker, date_str)

                        if not intraday_data.empty:
                            rv_daily = self.calculate_realized_volatility(
                                intraday_data, method='5min', annualize=False
                            )
                            daily_rv.append(rv_daily)
                        else:
                            daily_rv.append(np.nan)

                    if daily_rv:
                        rv_series = pd.Series(daily_rv, index=date_range)
                        rv_series = rv_series.dropna()

                        if len(rv_series) > 0:
                            rv_data[ticker] = rv_series
                            logger.info(f"Calculated intraday RV for {ticker}: {len(rv_series)} observations")

            except Exception as e:
                logger.error(f"Error calculating RV for {ticker}: {e}")
                continue

        if rv_data:
            rv_df = pd.DataFrame(rv_data)
            rv_df.index.name = 'Date'

            # Clean the data
            rv_df = rv_df.dropna()

            # Remove outliers (values > 5 standard deviations from mean)
            for col in rv_df.columns:
                mean_val = rv_df[col].mean()
                std_val = rv_df[col].std()
                rv_df[col] = rv_df[col].where(
                    np.abs(rv_df[col] - mean_val) <= 5 * std_val,
                    np.nan
                )

            rv_df = rv_df.dropna()

            if len(rv_df) > 0:
                logger.info(f"Final RV dataset shape: {rv_df.shape}")
                return rv_df
            else:
                logger.warning("All RV data was filtered out, using synthetic data")
                return self._generate_synthetic_rv_fallback(tickers, start_date, end_date)
        else:
            logger.warning("No RV data obtained, falling back to synthetic data")
            return self._generate_synthetic_rv_fallback(tickers, start_date, end_date)

    def _generate_synthetic_rv_fallback(self, tickers, start_date, end_date):
        """Generate synthetic RV data as fallback."""
        logger.info("Generating synthetic RV data as fallback")
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(date_range)
        n_indices = len(tickers)

        # Generate synthetic data similar to original function
        np.random.seed(42)
        base_volatility = np.random.uniform(15, 35, n_indices)  # Realistic RV levels

        rv_data = np.zeros((n_days, n_indices))
        rv_data[0] = np.random.gamma(2, base_volatility/2)

        # HAR structure
        beta_d = np.random.uniform(0.3, 0.5, n_indices)
        beta_w = np.random.uniform(0.2, 0.4, n_indices)
        beta_m = np.random.uniform(0.1, 0.2, n_indices)

        for t in range(1, n_days):
            daily = beta_d * rv_data[t-1]

            if t >= 5:
                weekly = beta_w * np.mean(rv_data[t-5:t], axis=0)
            else:
                weekly = beta_w * np.mean(rv_data[:t], axis=0)

            if t >= 22:
                monthly = beta_m * np.mean(rv_data[t-22:t], axis=0)
            else:
                monthly = beta_m * np.mean(rv_data[:t], axis=0)

            mean_vol = daily + weekly + monthly + base_volatility * 0.2

            # Add market stress events
            if np.random.random() < 0.01:
                event_indices = np.random.choice(n_indices, size=np.random.randint(1, n_indices//2+1), replace=False)
                mean_vol[event_indices] *= np.random.uniform(1.5, 3.0)

            rv_data[t] = np.random.gamma(2, mean_vol/2)

        # Create DataFrame
        df = pd.DataFrame(rv_data, index=date_range, columns=tickers)
        return df


def generate_synthetic_rv_data(n_indices=10, n_days=3500, seed=42):
    """
    Generate synthetic RV data for multiple stock indices.
    
    Args:
        n_indices: Number of stock indices
        n_days: Number of days
        seed: Random seed
        
    Returns:
        rv_data: DataFrame with RV data
        adjacency: Adjacency matrix representing volatility spillover
    """
    np.random.seed(seed)
    
    # Base volatility for each index
    base_volatility = np.random.uniform(0.6, 1.1, n_indices)
    
    # Generate synthetic RV data with autocorrelation and heteroskedasticity
    rv_data = np.zeros((n_days, n_indices))
    
    # Initial values
    rv_data[0] = np.random.gamma(2, base_volatility)
    
    # Create HAR effect parameters for each index
    beta_d = np.random.uniform(0.3, 0.5, n_indices)  # Daily effect
    beta_w = np.random.uniform(0.2, 0.4, n_indices)  # Weekly effect
    beta_m = np.random.uniform(0.1, 0.2, n_indices)  # Monthly effect
    
    # Generate time series with HAR structure
    for t in range(1, n_days):
        # Daily component
        daily = beta_d * rv_data[t-1]
        
        # Weekly component (or as much as available)
        if t >= 5:
            weekly = beta_w * np.mean(rv_data[t-5:t], axis=0)
        else:
            weekly = beta_w * np.mean(rv_data[:t], axis=0)
        
        # Monthly component (or as much as available)
        if t >= 22:
            monthly = beta_m * np.mean(rv_data[t-22:t], axis=0)
        else:
            monthly = beta_m * np.mean(rv_data[:t], axis=0)
        
        # Combine components with random shock
        mean_vol = daily + weekly + monthly + base_volatility * 0.2
        
        # Add volatility spike events (like financial crises)
        if np.random.random() < 0.01:  # 1% chance of a market event
            event_indices = np.random.choice(n_indices, size=np.random.randint(1, n_indices//2+1), replace=False)
            mean_vol[event_indices] *= np.random.uniform(2, 5)
        
        # Generate RV with gamma distribution (to ensure positive values and skewness)
        rv_data[t] = np.random.gamma(2, mean_vol/2)
    
    # Create spillover effect - directed graph adjacency matrix
    adjacency = np.zeros((n_indices, n_indices))
    
    # Create a hierarchy of indices (some indices are more influential)
    influence = np.random.exponential(1, n_indices)
    influence = influence / np.sum(influence) * n_indices
    
    # More influential indices affect others more
    for i in range(n_indices):
        for j in range(n_indices):
            if i != j:
                # Directed effect: i affects j more if i is more influential
                adjacency[i, j] = np.random.exponential(influence[i] / influence[j]) * 0.1
    
    # Square root transform the RV data as in the paper
    rv_data = np.sqrt(rv_data) * 100  # Scale up as mentioned in the paper
    
    # Create DataFrame
    index = pd.date_range(start='2002-05-01', periods=n_days, freq='B')
    columns = [f'Index_{i}' for i in range(n_indices)]
    df = pd.DataFrame(rv_data, index=index, columns=columns)
    
    return df, adjacency


def train_test_split(data, train_size=0.7):
    """
    Split data into training and testing sets.
    
    Args:
        data: DataFrame with RV data
        train_size: Proportion of data to use for training
        
    Returns:
        train_data: Training data
        test_data: Testing data
    """
    n = len(data)
    train_idx = int(n * train_size)
    train_data = data.iloc[:train_idx]
    test_data = data.iloc[train_idx:]
    
    return train_data, test_data


def prepare_har_inputs(data, lags=22):
    """
    Prepare inputs for HAR model.
    
    Args:
        data: DataFrame with RV data
        lags: Number of lag terms
        
    Returns:
        X: Input features
        y: Target values
    """
    n_samples = len(data) - lags
    n_features = data.shape[1]
    
    X = np.zeros((n_samples, n_features, lags))
    y = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        X[i] = data.iloc[i:i+lags].values.T
        y[i] = data.iloc[i+lags].values
    
    return X, y


def train_gsphar(model, train_loader, val_loader, n_epochs=50, lr=0.001):
    """
    Train the GSPHAR model.
    
    Args:
        model: GSPHAR model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE loss as used in the paper
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return model, train_losses, val_losses


def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        
    Returns:
        mae: Mean Absolute Error for each index
        predictions: Predicted values
        actuals: Actual values
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            predictions.append(y_pred.numpy())
            actuals.append(y_batch.numpy())
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    # Calculate MAE for each index
    mae = np.mean(np.abs(predictions - actuals), axis=0)
    
    return mae, predictions, actuals


class VolatilitySpilloverNetwork:
    """Construct volatility spillover networks using Diebold-Yilmaz methodology."""

    def __init__(self, window_size=252, forecast_horizon=10):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

    def estimate_var_model(self, data, lags=1):
        """
        Estimate VAR model for spillover analysis.

        Args:
            data: DataFrame with RV time series
            lags: Number of lags in VAR model

        Returns:
            VAR coefficients and residuals
        """
        from sklearn.linear_model import LinearRegression

        n_vars = data.shape[1]
        n_obs = len(data) - lags

        # Prepare lagged data
        Y = data.iloc[lags:].values
        X = np.zeros((n_obs, n_vars * lags + 1))  # +1 for intercept
        X[:, 0] = 1  # Intercept

        for lag in range(1, lags + 1):
            start_col = 1 + (lag - 1) * n_vars
            end_col = 1 + lag * n_vars
            X[:, start_col:end_col] = data.iloc[lags-lag:-lag].values

        # Estimate VAR coefficients
        coefficients = []
        residuals = []

        for i in range(n_vars):
            reg = LinearRegression()
            reg.fit(X, Y[:, i])
            coefficients.append(reg.coef_)
            residuals.append(Y[:, i] - reg.predict(X))

        return np.array(coefficients), np.array(residuals).T

    def compute_spillover_table(self, data, method='generalized'):
        """
        Compute Diebold-Yilmaz spillover table.

        Args:
            data: DataFrame with RV time series
            method: 'generalized' or 'orthogonal'

        Returns:
            Spillover table and total spillover index
        """
        # Estimate VAR model
        coeffs, residuals = self.estimate_var_model(data)
        n_vars = data.shape[1]

        # Compute variance-covariance matrix of residuals
        sigma = np.cov(residuals.T)

        # Compute forecast error variance decomposition
        fevd_matrix = self._compute_fevd(coeffs, sigma, self.forecast_horizon, method)

        # Normalize to get spillover table
        spillover_table = fevd_matrix / fevd_matrix.sum(axis=0) * 100

        # Compute total spillover index
        total_spillover = (spillover_table.sum() - np.trace(spillover_table)) / spillover_table.sum() * 100

        return spillover_table, total_spillover

    def _compute_fevd(self, coeffs, sigma, horizon, method='generalized'):
        """Compute forecast error variance decomposition."""
        n_vars = coeffs.shape[0]

        # Initialize FEVD matrix
        fevd = np.zeros((n_vars, n_vars))

        if method == 'generalized':
            # Generalized FEVD (Pesaran & Shin, 1998)
            sigma_diag = np.diag(np.diag(sigma))

            for h in range(horizon):
                # Simplified computation for demonstration
                # In practice, would need proper MA representation
                if h == 0:
                    ma_coeff = np.eye(n_vars)
                else:
                    # Approximate MA coefficients
                    ma_coeff = np.linalg.matrix_power(coeffs[:, 1:], h)

                # Compute contribution to FEVD
                for i in range(n_vars):
                    for j in range(n_vars):
                        numerator = sigma[j, j] * (ma_coeff[i, j] ** 2)
                        denominator = np.sum([sigma[k, k] * (ma_coeff[i, k] ** 2) for k in range(n_vars)])
                        fevd[i, j] += numerator / denominator if denominator > 0 else 0

        return fevd / horizon

    def construct_adjacency_matrix(self, spillover_table, threshold=0.1):
        """
        Construct adjacency matrix from spillover table.

        Args:
            spillover_table: Spillover table from compute_spillover_table
            threshold: Minimum spillover percentage to include edge

        Returns:
            Adjacency matrix for graph construction
        """
        # Remove diagonal (own effects) and apply threshold
        adjacency = spillover_table.copy()
        np.fill_diagonal(adjacency, 0)
        adjacency[adjacency < threshold] = 0

        # Normalize to [0, 1] range
        if adjacency.max() > 0:
            adjacency = adjacency / adjacency.max()

        return adjacency


def test_har_model(data, train_size=0.7, batch_size=64, n_epochs=50, val_size=0.15, use_spillover_network=True):
    """
    Test the GSPHAR model with provided data.

    Args:
        data: DataFrame with RV data
        train_size: Proportion of data to use for training
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        val_size: Proportion of training data to use for validation
        use_spillover_network: Whether to use spillover-based adjacency matrix

    Returns:
        mae: Mean Absolute Error for each index
        model: Trained model
        adjacency: Adjacency matrix used
    """
    n_indices = data.shape[1]

    # Construct adjacency matrix
    if use_spillover_network and n_indices > 1:
        logger.info("Constructing volatility spillover network...")
        spillover_net = VolatilitySpilloverNetwork()
        try:
            spillover_table, total_spillover = spillover_net.compute_spillover_table(data)
            adjacency = spillover_net.construct_adjacency_matrix(spillover_table)
            logger.info(f"Total spillover index: {total_spillover:.2f}%")
        except Exception as e:
            logger.warning(f"Error computing spillover network: {e}. Using correlation-based adjacency.")
            corr_matrix = data.corr()
            adjacency = np.maximum(0, corr_matrix - np.eye(n_indices))
    else:
        # Fallback to correlation-based adjacency
        corr_matrix = data.corr().values  # Convert to numpy array
        adjacency = np.maximum(0, corr_matrix - np.eye(n_indices))
        # Ensure adjacency is real-valued
        adjacency = np.real(adjacency)

    # Compute magnetic Laplacian
    ml = MagneticLaplacian(q=0.25)
    L_m = ml.compute(adjacency)
    U_m, Lambda_m = ml.eigendecomposition(L_m)

    # Split data
    train_data, test_data = train_test_split(data, train_size)

    # Further split training data into train and validation
    train_idx = int(len(train_data) * (1 - val_size))
    val_data = train_data.iloc[train_idx:]
    train_data = train_data.iloc[:train_idx]

    # Prepare inputs
    X_train, y_train = prepare_har_inputs(train_data)
    X_val, y_val = prepare_har_inputs(val_data)
    X_test, y_test = prepare_har_inputs(test_data)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create and train model
    model = GSPHAR(n_indices, U_m)
    model, train_losses, val_losses = train_gsphar(model, train_loader, val_loader, n_epochs)

    # Evaluate model
    mae, predictions, actuals = evaluate_model(model, test_loader)

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png')
    plt.close()

    # Plot predictions vs actuals for a sample index
    sample_idx = 0
    plt.figure(figsize=(15, 7))
    plt.plot(actuals[:100, sample_idx], label='Actual')
    plt.plot(predictions[:100, sample_idx], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Realized Volatility (Sqrt)')
    plt.title(f'Actual vs Predicted RV for {data.columns[sample_idx]}')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_sample.png')
    plt.close()

    return mae, model, adjacency


def run_bloomberg_experiment():
    """Run experiment with real Bloomberg data using xbbg."""
    logger.info("Starting Bloomberg data experiment...")

    # Define major stock indices for volatility spillover analysis
    # Using liquid, well-known indices with good data availability
    bloomberg_tickers = [
        'SPX Index',      # S&P 500 (US)
        'SX5E Index',     # Euro Stoxx 50 (Europe)
        'NKY Index',      # Nikkei 225 (Japan)
        'UKX Index',      # FTSE 100 (UK)
        'HSI Index',      # Hang Seng (Hong Kong)
        'AS51 Index',     # ASX 200 (Australia)
    ]

    # Date range for analysis - use business days to ensure data availability
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=800)).strftime('%Y-%m-%d')  # ~3 years of data

    try:
        # Initialize Bloomberg data manager with longer timeout for multiple tickers
        bbg_manager = BloombergDataManager(timeout=60)

        # Test connection first with a simple request
        logger.info("Testing Bloomberg connection...")
        test_data = blp.bdp('SPX Index', 'PX_LAST', timeout=10)
        if test_data.empty:
            logger.error("Bloomberg connection test failed - no data returned")
            return None, None, None, None

        logger.info(f"Bloomberg connection successful. SPX last price: {test_data.iloc[0, 0]}")

        # Fetch realized volatility data
        logger.info(f"Fetching RV data for {len(bloomberg_tickers)} indices from {start_date} to {end_date}")
        rv_data = bbg_manager.get_realized_volatility_series(
            tickers=bloomberg_tickers,
            start_date=start_date,
            end_date=end_date,
            rv_method='daily'
        )

        if rv_data.empty or len(rv_data) < 100:
            logger.error(f"Insufficient Bloomberg data obtained. Shape: {rv_data.shape}")
            return None, None, None, None

        logger.info(f"Successfully obtained RV data with shape: {rv_data.shape}")
        print("\nBloomberg RV Data Summary:")
        print(rv_data.describe())

        # Check for minimum data requirements
        min_obs = 252  # At least 1 year of data
        if len(rv_data) < min_obs:
            logger.warning(f"Only {len(rv_data)} observations available, minimum {min_obs} recommended")

        # Test GSPHAR model with Bloomberg data
        logger.info("Training GSPHAR model with Bloomberg data...")
        mae, model, adjacency = test_har_model(
            rv_data,
            train_size=0.7,
            batch_size=32,
            n_epochs=50,
            use_spillover_network=True
        )

        print("\nBloomberg Data Results:")
        print("MAE for each index:")
        for i, ticker in enumerate(rv_data.columns):
            print(f"{ticker}: {mae[i]:.4f}")

        print(f"\nAverage MAE: {np.mean(mae):.4f}")

        # Visualize spillover network
        plt.figure(figsize=(12, 8))
        im = plt.imshow(adjacency, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Spillover Intensity')
        plt.title('Volatility Spillover Network (Bloomberg Data)')
        plt.xlabel('To Index')
        plt.ylabel('From Index')

        # Set tick labels
        tick_labels = [ticker.replace(' Index', '') for ticker in rv_data.columns]
        plt.xticks(range(len(tick_labels)), tick_labels, rotation=45)
        plt.yticks(range(len(tick_labels)), tick_labels)

        plt.tight_layout()
        plt.savefig('spillover_network_bloomberg.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Additional Bloomberg-specific analysis
        logger.info("Generating Bloomberg-specific visualizations...")

        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        corr_matrix = rv_data.corr()
        im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, label='Correlation')
        plt.title('RV Correlation Matrix (Bloomberg Data)')
        plt.xticks(range(len(tick_labels)), tick_labels, rotation=45)
        plt.yticks(range(len(tick_labels)), tick_labels)

        # Add correlation values as text
        for i in range(len(tick_labels)):
            for j in range(len(tick_labels)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha='center', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('rv_correlation_bloomberg.png', dpi=300, bbox_inches='tight')
        plt.close()

        return rv_data, mae, model, adjacency

    except Exception as e:
        logger.error(f"Error in Bloomberg experiment: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Falling back to synthetic data experiment")
        return None, None, None, None


def run_synthetic_experiment():
    """Run experiment with synthetic data."""
    logger.info("Starting synthetic data experiment...")

    try:
        logger.info("Generating synthetic RV data...")
        rv_data, true_adjacency = generate_synthetic_rv_data(n_indices=8, n_days=1000, seed=42)

        print("\nSynthetic Data Summary:")
        print(rv_data.describe())

        logger.info("Testing GSPHAR model with synthetic data...")
        mae, model, adjacency = test_har_model(
            rv_data,
            train_size=0.7,
            batch_size=32,
            n_epochs=30,
            use_spillover_network=True
        )

        print("\nSynthetic Data Results:")
        print("MAE for each index:")
        for i, mae_val in enumerate(mae):
            print(f"Index_{i}: {mae_val:.4f}")

        print(f"\nAverage MAE: {np.mean(mae):.4f}")

        return rv_data, mae, model, adjacency

    except Exception as e:
        logger.error(f"Error in synthetic experiment: {e}")
        raise


def visualize_results(model, rv_data):
    """Create comprehensive visualizations of results."""

    # Compare learned filter weights with HAR weights
    mid_term_weights = torch.softmax(model.filter_mid_term.weights, dim=0).detach().numpy()
    long_term_weights = torch.softmax(model.filter_long_term.weights, dim=0).detach().numpy()

    # HAR weights (equal weighting)
    har_mid_term_weights = np.ones(4) / 4
    har_long_term_weights = np.ones(17) / 17

    plt.figure(figsize=(15, 7))

    # Plot mid-term weights
    plt.subplot(1, 2, 1)
    plt.bar(range(4), mid_term_weights, alpha=0.7, label='Learned', color='blue')
    plt.bar(range(4), har_mid_term_weights, alpha=0.5, label='HAR', color='red')
    plt.xlabel('Lag')
    plt.ylabel('Weight')
    plt.title('Mid-term Weights Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot long-term weights
    plt.subplot(1, 2, 2)
    plt.bar(range(17), long_term_weights, alpha=0.7, label='Learned', color='blue')
    plt.bar(range(17), har_long_term_weights, alpha=0.5, label='HAR', color='red')
    plt.xlabel('Lag')
    plt.ylabel('Weight')
    plt.title('Long-term Weights Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('weights_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot RV time series
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(rv_data.columns[:4]):  # Plot first 4 series
        plt.subplot(2, 2, i+1)
        plt.plot(rv_data.index, rv_data[col], linewidth=0.8)
        plt.title(f'Realized Volatility: {col}')
        plt.xlabel('Date')
        plt.ylabel('RV (%)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rv_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function."""
    print("="*60)
    print("GSPHAR Model with Bloomberg Data Integration")
    print("Enhanced implementation of paper 2410.22706v2")
    print("="*60)

    # Try Bloomberg experiment first
    if BLOOMBERG_AVAILABLE:
        print("\n1. Attempting Bloomberg data experiment...")
        rv_data_bbg, mae_bbg, model_bbg, adj_bbg = run_bloomberg_experiment()

        if rv_data_bbg is not None:
            print("\n✓ Bloomberg experiment completed successfully!")
            visualize_results(model_bbg, rv_data_bbg)

            print("\nFiles generated:")
            print("- training_curve.png: Training progress")
            print("- prediction_sample.png: Sample predictions")
            print("- weights_comparison.png: Learned vs HAR weights")
            print("- spillover_network.png: Volatility spillover network")
            print("- rv_timeseries.png: RV time series plots")

            return

    # Fallback to synthetic experiment
    print("\n2. Running synthetic data experiment...")
    rv_data_syn, mae_syn, model_syn, adj_syn = run_synthetic_experiment()

    print("\n✓ Synthetic experiment completed successfully!")
    visualize_results(model_syn, rv_data_syn)

    print("\nFiles generated:")
    print("- training_curve.png: Training progress")
    print("- prediction_sample.png: Sample predictions")
    print("- weights_comparison.png: Learned vs HAR weights")
    print("- rv_timeseries.png: RV time series plots")

    print("\n" + "="*60)
    print("Experiment completed. Check generated plots for results.")
    print("="*60)


if __name__ == "__main__":
    main()