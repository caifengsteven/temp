"""
Enhanced DGNN Implementation with Bloomberg Data Integration
Based on paper 2410.23275v1 - Dynamic Graph Neural Networks for Margin Call Forecasting
Enhanced with real market data via xbbg Bloomberg API

Features:
- Real-time OIS rate fetching from Bloomberg
- Market-calibrated CIR process
- Enhanced contract pricing with yield curves
- Real volatility surfaces integration

Requirements:
- xbbg: pip install xbbg
- Bloomberg Terminal or BPIPE connection required

Run with: python 2410.23275v1_test_strategy.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import List, Tuple, Dict, Optional
import pandas as pd
import datetime as dt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Try to import torch_geometric, fallback to simplified implementation if not available
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv
    TORCH_GEOMETRIC_AVAILABLE = True
    print("PyTorch Geometric loaded successfully")
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. Using simplified GCN implementation.")

    # Simplified GCN implementation
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index):
            # Simplified: just apply linear transformation
            # In a full implementation, this would include graph convolution
            return self.linear(x)

# Bloomberg data integration
try:
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
    print("Bloomberg xbbg module loaded successfully")
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("Warning: xbbg not available. Using simulated data only.")
    print("To install: pip install xbbg")

    # Create mock blp module for fallback
    class MockBlp:
        def bdh(self, *args, **kwargs):
            return pd.DataFrame()
        def bdp(self, *args, **kwargs):
            return pd.DataFrame()
    blp = MockBlp()

@dataclass
class MarketDataConfig:
    """Configuration for Bloomberg market data fetching"""
    # OIS curve tickers for different currencies
    ois_tickers: Dict[str, List[str]] = None
    # Volatility surface tickers
    vol_tickers: Dict[str, str] = None
    # Reference rate tickers
    reference_rates: Dict[str, str] = None

    def __post_init__(self):
        if self.ois_tickers is None:
            self.ois_tickers = {
                'USD': ['USSO1Z Curncy', 'USSO2Z Curncy', 'USSO3Z Curncy', 'USSOA Curncy',
                       'USSOB Curncy', 'USSOC Curncy', 'USSOD Curncy', 'USSOE Curncy',
                       'USSOF Curncy', 'USSOG Curncy', 'USSOH Curncy', 'USSOI Curncy',
                       'USSOJ Curncy', 'USSO1 Curncy', 'USSO2 Curncy', 'USSO3 Curncy',
                       'USSO4 Curncy', 'USSO5 Curncy', 'USSO7 Curncy', 'USSO10 Curncy'],
                'EUR': ['EUSWO1Z Curncy', 'EUSWO2Z Curncy', 'EUSWO3Z Curncy', 'EUSWOA Curncy',
                       'EUSWOB Curncy', 'EUSWOC Curncy', 'EUSWOD Curncy', 'EUSWOE Curncy',
                       'EUSWOF Curncy', 'EUSWOG Curncy', 'EUSWOH Curncy', 'EUSWOI Curncy',
                       'EUSWOJ Curncy', 'EUSWO1 Curncy', 'EUSWO2 Curncy', 'EUSWO3 Curncy',
                       'EUSWO4 Curncy', 'EUSWO5 Curncy', 'EUSWO7 Curncy', 'EUSWO10 Curncy']
            }

        if self.vol_tickers is None:
            self.vol_tickers = {
                'USD': 'USSV1Y Curncy',  # 1Y USD OIS volatility
                'EUR': 'EUSV1Y Curncy'   # 1Y EUR OIS volatility
            }

        if self.reference_rates is None:
            self.reference_rates = {
                'USD': 'FEDL01 Index',   # Fed Funds Effective Rate
                'EUR': 'EONIA Index',    # EONIA Rate
                'SOFR': 'SOFR Index'     # SOFR Rate
            }

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class BloombergDataFetcher:
    """Bloomberg data fetching and processing module"""

    def __init__(self, config: MarketDataConfig = None):
        """
        Initialize Bloomberg data fetcher

        Args:
            config: Market data configuration
        """
        self.config = config or MarketDataConfig()
        self.cache = {}  # Simple caching mechanism

    def fetch_ois_curve(self, currency: str = 'USD', date: str = None) -> pd.DataFrame:
        """
        Fetch OIS curve data from Bloomberg

        Args:
            currency: Currency code (USD, EUR, etc.)
            date: Date string in YYYY-MM-DD format, defaults to latest

        Returns:
            DataFrame with tenors and rates
        """
        if not BLOOMBERG_AVAILABLE:
            return self._simulate_ois_curve(currency)

        try:
            tickers = self.config.ois_tickers.get(currency, self.config.ois_tickers['USD'])

            if date is None:
                date = dt.datetime.now().strftime('%Y-%m-%d')

            # Fetch data from Bloomberg using correct API
            data = blp.bdh(tickers, 'PX_LAST', start_date=date, end_date=date)

            if data.empty:
                print(f"No Bloomberg data available for {date}, using simulated data")
                return self._simulate_ois_curve(currency)

            # Process and clean data
            rates_data = []
            tenor_mapping = {
                'O1Z': '1D', 'O2Z': '2D', 'O3Z': '3D', 'OA': '1W', 'OB': '2W', 'OC': '3W', 'OD': '1M',
                'OE': '2M', 'OF': '3M', 'OG': '4M', 'OH': '5M', 'OI': '6M', 'OJ': '7M', 'O1': '1Y',
                'O2': '2Y', 'O3': '3Y', 'O4': '4Y', 'O5': '5Y', 'O7': '7Y', 'O10': '10Y'
            }

            for ticker in tickers:
                if ticker in data.columns:
                    rate_value = data[ticker].iloc[-1] if not data[ticker].empty else None
                    if rate_value is not None and not np.isnan(rate_value):
                        # Extract tenor from ticker
                        for key, tenor in tenor_mapping.items():
                            if key in ticker:
                                rates_data.append({'tenor': tenor, 'rate': rate_value / 100})  # Convert to decimal
                                break

            if not rates_data:
                print(f"No valid Bloomberg OIS data found for {currency}, using simulated data")
                return self._simulate_ois_curve(currency)

            df = pd.DataFrame(rates_data)
            df['days'] = df['tenor'].apply(self._tenor_to_days)
            df = df.sort_values('days').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"Error fetching Bloomberg OIS data: {e}")
            return self._simulate_ois_curve(currency)

    def fetch_reference_rate(self, currency: str = 'USD', days: int = 30) -> pd.Series:
        """
        Fetch reference rate time series

        Args:
            currency: Currency code
            days: Number of historical days to fetch

        Returns:
            Time series of reference rates
        """
        if not BLOOMBERG_AVAILABLE:
            return self._simulate_reference_rate(days)

        try:
            ticker = self.config.reference_rates.get(currency, self.config.reference_rates['USD'])
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=days)

            data = blp.bdh(ticker, 'PX_LAST',
                         start_date=start_date.strftime('%Y-%m-%d'),
                         end_date=end_date.strftime('%Y-%m-%d'))

            if data.empty or ticker not in data.columns:
                print(f"No Bloomberg reference rate data for {currency}, using simulated data")
                return self._simulate_reference_rate(days)

            rates = data[ticker].dropna() / 100  # Convert to decimal
            return rates

        except Exception as e:
            print(f"Error fetching Bloomberg reference rate: {e}")
            return self._simulate_reference_rate(days)

    def fetch_volatility_surface(self, currency: str = 'USD') -> float:
        """
        Fetch implied volatility for OIS options

        Args:
            currency: Currency code

        Returns:
            Implied volatility (annualized)
        """
        if not BLOOMBERG_AVAILABLE:
            return 0.15  # Default 15% volatility

        try:
            ticker = self.config.vol_tickers.get(currency, self.config.vol_tickers['USD'])
            data = blp.bdp(ticker, 'PX_LAST')

            if data.empty or ticker not in data.index:
                print(f"No Bloomberg volatility data for {currency}, using default")
                return 0.15

            vol = data.loc[ticker, 'PX_LAST'] / 100  # Convert to decimal
            return vol if not np.isnan(vol) else 0.15

        except Exception as e:
            print(f"Error fetching Bloomberg volatility: {e}")
            return 0.15

    def _simulate_ois_curve(self, currency: str) -> pd.DataFrame:
        """Simulate OIS curve when Bloomberg data is not available"""
        base_rate = 0.04 if currency == 'USD' else 0.02
        tenors = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '10Y']
        days = [1, 7, 30, 90, 180, 365, 730, 1095, 1825, 3650]

        # Simple upward sloping curve with some noise
        rates = []
        for i, d in enumerate(days):
            rate = base_rate + (d / 3650) * 0.02 + np.random.normal(0, 0.002)
            rates.append(max(rate, 0.001))  # Floor at 0.1%

        return pd.DataFrame({
            'tenor': tenors,
            'days': days,
            'rate': rates
        })

    def _simulate_reference_rate(self, days: int) -> pd.Series:
        """Simulate reference rate time series"""
        dates = pd.date_range(end=dt.datetime.now(), periods=days, freq='D')
        # Simple random walk around 4%
        rates = [0.04]
        for _ in range(days - 1):
            change = np.random.normal(0, 0.001)
            new_rate = max(rates[-1] + change, 0.001)
            rates.append(new_rate)

        return pd.Series(rates, index=dates)

    def _tenor_to_days(self, tenor: str) -> int:
        """Convert tenor string to days"""
        tenor_map = {
            '1D': 1, '2D': 2, '3D': 3, '1W': 7, '2W': 14, '3W': 21,
            '1M': 30, '2M': 60, '3M': 90, '4M': 120, '5M': 150, '6M': 180, '7M': 210,
            '1Y': 365, '2Y': 730, '3Y': 1095, '4Y': 1460, '5Y': 1825, '7Y': 2555, '10Y': 3650
        }
        return tenor_map.get(tenor, 365)

class EnhancedCIRProcess:
    """Enhanced Cox-Ingersoll-Ross process with Bloomberg data calibration"""

    def __init__(self, k: float = None, theta: float = None, sigma: float = None, r0: float = None,
                 data_fetcher: BloombergDataFetcher = None, currency: str = 'USD'):
        """
        Initialize CIR process with optional market data calibration

        Args:
            k: Mean reversion speed (auto-calibrated if None)
            theta: Long-term mean (auto-calibrated if None)
            sigma: Volatility (auto-calibrated if None)
            r0: Initial interest rate (fetched from market if None)
            data_fetcher: Bloomberg data fetcher instance
            currency: Currency for market data
        """
        self.data_fetcher = data_fetcher or BloombergDataFetcher()
        self.currency = currency

        # Calibrate parameters from market data if not provided
        if any(param is None for param in [k, theta, sigma, r0]):
            self._calibrate_from_market()

        # Set parameters (use calibrated or provided values)
        self.k = k if k is not None else getattr(self, 'k', 0.6)
        self.theta = theta if theta is not None else getattr(self, 'theta', 0.04)
        self.sigma = sigma if sigma is not None else getattr(self, 'sigma', 0.14)
        self.r0 = r0 if r0 is not None else getattr(self, 'r0', 0.04)

        # Check Feller condition
        if 2 * self.k * self.theta < self.sigma ** 2:
            print("Warning: Feller condition not satisfied. Process may reach zero.")
            # Adjust sigma to satisfy Feller condition
            self.sigma = np.sqrt(2 * self.k * self.theta * 0.9)
            print(f"Adjusted sigma to {self.sigma:.4f} to satisfy Feller condition")

    def _calibrate_from_market(self):
        """Calibrate CIR parameters from market data"""
        try:
            print(f"Calibrating CIR parameters from {self.currency} market data...")

            # Fetch historical reference rates for calibration
            historical_rates = self.data_fetcher.fetch_reference_rate(self.currency, days=252)  # 1 year

            if len(historical_rates) < 10:
                print("Insufficient market data for calibration, using default parameters")
                self._set_default_parameters()
                return

            # Convert to numpy array and handle missing values
            rates = historical_rates.dropna().values

            if len(rates) < 10:
                print("Insufficient valid market data for calibration, using default parameters")
                self._set_default_parameters()
                return

            # Simple calibration using method of moments
            self.r0 = rates[-1]  # Current rate
            self.theta = np.mean(rates)  # Long-term mean

            # Estimate mean reversion and volatility from time series
            dt = 1/252  # Daily data, assuming 252 trading days per year
            rate_changes = np.diff(rates)

            # Mean reversion speed (simplified estimation)
            # E[dr] = k(theta - r)dt, so k ≈ -E[dr/r]/dt when r ≈ theta
            mean_rate = np.mean(rates[:-1])
            mean_change = np.mean(rate_changes)
            if mean_rate > 0:
                self.k = max(-mean_change / (mean_rate * dt), 0.1)  # Ensure positive k
            else:
                self.k = 0.6  # Default

            # Volatility estimation
            # Var[dr] ≈ sigma^2 * r * dt, so sigma ≈ sqrt(Var[dr]/(r*dt))
            var_changes = np.var(rate_changes)
            if mean_rate > 0 and dt > 0:
                self.sigma = np.sqrt(var_changes / (mean_rate * dt))
            else:
                self.sigma = 0.14  # Default

            # Apply reasonable bounds
            self.k = np.clip(self.k, 0.1, 5.0)
            self.theta = np.clip(self.theta, 0.001, 0.15)
            self.sigma = np.clip(self.sigma, 0.05, 0.5)

            print(f"Calibrated parameters: k={self.k:.4f}, theta={self.theta:.4f}, sigma={self.sigma:.4f}, r0={self.r0:.4f}")

        except Exception as e:
            print(f"Error in market calibration: {e}")
            self._set_default_parameters()

    def _set_default_parameters(self):
        """Set default CIR parameters"""
        defaults = {
            'USD': {'k': 0.6, 'theta': 0.04, 'sigma': 0.14, 'r0': 0.04},
            'EUR': {'k': 0.5, 'theta': 0.02, 'sigma': 0.12, 'r0': 0.02}
        }

        params = defaults.get(self.currency, defaults['USD'])
        self.k = params['k']
        self.theta = params['theta']
        self.sigma = params['sigma']
        self.r0 = params['r0']

        print(f"Using default {self.currency} parameters: k={self.k}, theta={self.theta}, sigma={self.sigma}, r0={self.r0}")

    def simulate(self, T: int, dt: float = 1/365) -> np.ndarray:
        """
        Simulate CIR process for T periods with time step dt

        Args:
            T: Number of time steps
            dt: Time step size (default: daily)

        Returns:
            Array of simulated interest rates
        """
        n_steps = T  # T is already the number of days
        rates = np.zeros(n_steps + 1)
        rates[0] = self.r0

        for i in range(1, n_steps + 1):
            # Use exact simulation method for CIR process
            c = (self.sigma ** 2) * (1 - np.exp(-self.k * dt)) / (4 * self.k)
            d = (4 * self.theta * self.k) / (self.sigma ** 2)
            nc = rates[i-1] * (np.exp(-self.k * dt) / c)

            # Sample from non-central chi-squared distribution
            try:
                rates[i] = c * stats.ncx2.rvs(d, nc)
            except:
                # Fallback to Euler scheme if sampling fails
                dW = np.random.normal(0, np.sqrt(dt))
                dr = self.k * (self.theta - rates[i-1]) * dt + self.sigma * np.sqrt(max(rates[i-1], 0)) * dW
                rates[i] = max(rates[i-1] + dr, 0.0001)  # Floor at 1bp

        return rates

    def get_yield_curve(self, tenors: List[str] = None) -> pd.DataFrame:
        """
        Get current yield curve from market data

        Args:
            tenors: List of tenor strings, defaults to standard tenors

        Returns:
            DataFrame with tenors, days, and rates
        """
        if tenors is None:
            tenors = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '10Y']

        return self.data_fetcher.fetch_ois_curve(self.currency)

class EnhancedFinancialNetwork:
    """Enhanced dynamic financial network with real market data integration"""

    def __init__(self, n_nodes: int, hub_ratio: float = 0.3,
                 gamma: float = 3.0, eta: float = -4.0,
                 theta: float = 20.0, beta: float = 5.0,
                 data_fetcher: BloombergDataFetcher = None,
                 currency: str = 'USD'):
        """
        Initialize enhanced financial network

        Args:
            n_nodes: Number of nodes (financial entities)
            hub_ratio: Proportion of nodes that are hubs
            gamma, eta, theta, beta: Parameters for contract generation intensity
            data_fetcher: Bloomberg data fetcher instance
            currency: Currency for market data
        """
        self.n_nodes = n_nodes
        self.n_hubs = int(n_nodes * hub_ratio)
        self.gamma = gamma
        self.eta = eta
        self.theta = theta
        self.beta = beta
        self.currency = currency

        # Bloomberg data integration
        self.data_fetcher = data_fetcher or BloombergDataFetcher()
        self.yield_curve = None
        self.volatility = None
        self._update_market_data()

        # Initialize node types (hubs = 1, privates = -1)
        self.node_types = np.ones(n_nodes) * -1
        self.node_types[:self.n_hubs] = 1

        # Contract storage
        self.contracts = {}  # Will store active contracts
        self.contract_counter = 0

    def _update_market_data(self):
        """Update market data from Bloomberg"""
        try:
            self.yield_curve = self.data_fetcher.fetch_ois_curve(self.currency)
            self.volatility = self.data_fetcher.fetch_volatility_surface(self.currency)
            print(f"Updated market data for {self.currency}")
        except Exception as e:
            print(f"Error updating market data: {e}")
            # Use fallback data
            self.yield_curve = self.data_fetcher._simulate_ois_curve(self.currency)
            self.volatility = 0.15
        
    def g_function(self, x_i: float, x_j: float) -> float:
        """
        Compute g function for node features as defined in the paper

        Args:
            x_i, x_j: Node features

        Returns:
            g function value
        """
        return -x_i * x_j + abs(x_i - x_j) + (x_i + x_j) / 3

    def contract_intensity(self, r: float, i: int, j: int) -> float:
        """
        Compute stochastic intensity for contract generation

        Args:
            r: Current interest rate
            i, j: Node indices

        Returns:
            Intensity value
        """
        x_i, x_j = self.node_types[i], self.node_types[j]
        g_val = self.g_function(x_i, x_j)
        return self.gamma * np.exp(self.eta + (self.theta + self.beta * g_val) * r)

    def _interpolate_rate(self, maturity_days: int) -> float:
        """
        Interpolate rate from yield curve for given maturity

        Args:
            maturity_days: Maturity in days

        Returns:
            Interpolated rate
        """
        if self.yield_curve is None or len(self.yield_curve) == 0:
            return 0.04  # Default rate

        curve_days = self.yield_curve['days'].values
        curve_rates = self.yield_curve['rate'].values

        # Linear interpolation
        if maturity_days <= curve_days[0]:
            return curve_rates[0]
        elif maturity_days >= curve_days[-1]:
            return curve_rates[-1]
        else:
            return np.interp(maturity_days, curve_days, curve_rates)

    def _calculate_ois_fair_rate(self, maturity_years: float, current_rate: float) -> float:
        """
        Calculate fair OIS rate using yield curve

        Args:
            maturity_years: Contract maturity in years
            current_rate: Current overnight rate

        Returns:
            Fair OIS rate
        """
        maturity_days = int(maturity_years * 365)

        # Get rate from yield curve
        fair_rate = self._interpolate_rate(maturity_days)

        # Add small spread based on volatility and maturity
        spread = self.volatility * np.sqrt(maturity_years) * 0.1  # 10% of vol-adjusted spread

        return fair_rate + spread

    def _calculate_contract_value(self, contract: Dict, current_rate: float,
                                current_time: float) -> float:
        """
        Calculate mark-to-market value of OIS contract using market data

        Args:
            contract: Contract dictionary
            current_rate: Current overnight rate
            current_time: Current time

        Returns:
            Contract value
        """
        time_to_maturity = contract['maturity'] - current_time

        if time_to_maturity <= 0:
            return 0.0

        # Get current fair rate for remaining maturity
        current_fair_rate = self._calculate_ois_fair_rate(time_to_maturity, current_rate)

        # Calculate present value of rate difference
        # Simplified: PV = (fair_rate - contract_rate) * maturity * discount_factor
        discount_rate = self._interpolate_rate(int(time_to_maturity * 365))
        discount_factor = np.exp(-discount_rate * time_to_maturity)

        rate_diff = current_fair_rate - contract['rate']
        pv = rate_diff * time_to_maturity * contract['principal'] * discount_factor

        return pv
    
    def simulate_contracts(self, rates: np.ndarray, T: int, dt: float = 1/365) -> List[Dict]:
        """
        Enhanced contract simulation with market-based pricing

        Args:
            rates: Array of interest rates
            T: Time horizon
            dt: Time step

        Returns:
            List of contract events
        """
        n_steps = len(rates)
        time_grid = np.arange(0, n_steps) * dt
        contracts_history = []
        active_contracts = {}

        print(f"Simulating {n_steps} time steps with enhanced pricing...")

        # For each time step
        for t_idx, t in enumerate(time_grid):
            # Update market data periodically (e.g., monthly to reduce API calls)
            if t_idx % 30 == 0 and t_idx > 0:  # Update monthly, skip first iteration
                self._update_market_data()

            # Process contract maturities
            expired = []
            for c_id, contract in active_contracts.items():
                if t >= contract['maturity']:
                    expired.append(c_id)

            for c_id in expired:
                del active_contracts[c_id]

            # Generate new contracts
            current_rate = rates[t_idx]
            for i in range(self.n_nodes):
                for j in range(i+1, self.n_nodes):  # Upper triangle only
                    # Compute intensity and probability of contract
                    intensity = self.contract_intensity(current_rate, i, j)
                    p_contract = 1 - np.exp(-intensity * dt)

                    # Generate contract with probability p_contract
                    if np.random.random() < p_contract:
                        # Create standardized 1-year OIS contract
                        self.contract_counter += 1
                        contract_id = f"OIS_{self.contract_counter}"

                        # Use market-based fair rate calculation
                        maturity_years = 1.0  # 1-year contract
                        fair_rate = self._calculate_ois_fair_rate(maturity_years, current_rate)
                        maturity = t + maturity_years

                        # Randomly assign fixed rate payer/receiver
                        if np.random.random() < 0.5:
                            delta_i, delta_j = 1, -1  # i is receiver, j is payer
                        else:
                            delta_i, delta_j = -1, 1  # i is payer, j is receiver

                        # Create contract with enhanced features
                        contract = {
                            'id': contract_id,
                            'start_time': t,
                            'maturity': maturity,
                            'nodes': (i, j),
                            'rate': fair_rate,
                            'delta': (delta_i, delta_j),
                            'principal': 1.0,
                            'currency': self.currency,
                            'contract_type': 'OIS',
                            'notional': 1000000,  # $1M notional
                            'day_count': 'ACT/360'
                        }

                        # Store contract
                        contracts_history.append(contract)
                        active_contracts[contract_id] = contract
            
            # Calculate mark-to-market values and variation margins using enhanced pricing
            if t_idx > 0:
                for node in range(self.n_nodes):
                    # Calculate net value of all contracts for this node using market pricing
                    net_value = 0
                    for c_id, contract in active_contracts.items():
                        i, j = contract['nodes']
                        if i == node or j == node:
                            # Get node's position in contract (0 or 1)
                            idx = 0 if i == node else 1
                            delta = contract['delta'][idx]

                            # Enhanced mark-to-market calculation using market data
                            contract_value = self._calculate_contract_value(contract, current_rate, t)
                            net_value += delta * contract_value

                    # Store node's net value
                    if t_idx not in self.contracts:
                        self.contracts[t_idx] = {'net_values': np.zeros(self.n_nodes),
                                               'active_contracts': len(active_contracts)}
                    self.contracts[t_idx]['net_values'][node] = net_value

        # Calculate variation margins with enhanced methodology
        for t_idx in range(1, n_steps):
            if t_idx in self.contracts and t_idx-1 in self.contracts:
                prev_values = self.contracts[t_idx-1]['net_values']
                curr_values = self.contracts[t_idx]['net_values']

                # Enhanced margin calculation with proper discounting
                overnight_rate = rates[t_idx]
                overnight_factor = 1 + overnight_rate * dt

                # Variation margin = change in portfolio value + accrued interest
                margins = curr_values - prev_values * overnight_factor

                # Apply minimum transfer amount (threshold)
                threshold = 10000  # $10k minimum transfer
                margins = np.where(np.abs(margins) < threshold, 0, margins)

                self.contracts[t_idx]['margins'] = margins
                self.contracts[t_idx]['overnight_rate'] = overnight_rate

        print(f"Generated {len(contracts_history)} contracts over {n_steps} time steps")
        return contracts_history

    def get_adjacency_matrix(self, t_idx: int) -> np.ndarray:
        """Get adjacency matrix at time t_idx with enhanced contract tracking"""
        adj_matrix = np.zeros((self.n_nodes, self.n_nodes))

        # Check stored contracts for active ones at time t_idx
        current_time = t_idx * (1/365)

        # Look through all stored contract data to find active contracts
        for time_key, time_data in self.contracts.items():
            if isinstance(time_key, int) and time_key <= t_idx:
                # This is a time step, not a contract
                continue

        # Alternative: track active contracts separately
        if hasattr(self, '_active_contracts_by_time'):
            active_contracts = self._active_contracts_by_time.get(t_idx, {})
            for contract in active_contracts.values():
                i, j = contract['nodes']
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

        return adj_matrix

    def get_variation_margins(self, t_idx: int) -> np.ndarray:
        """Get variation margins at time t_idx"""
        if t_idx in self.contracts and 'margins' in self.contracts[t_idx]:
            return self.contracts[t_idx]['margins']
        return np.zeros(self.n_nodes)

    def get_portfolio_statistics(self, t_idx: int) -> Dict:
        """Get portfolio statistics at time t_idx"""
        stats = {
            'total_notional': 0,
            'num_contracts': 0,
            'avg_maturity': 0,
            'net_values': np.zeros(self.n_nodes),
            'gross_values': np.zeros(self.n_nodes)
        }

        if t_idx in self.contracts:
            stats['net_values'] = self.contracts[t_idx].get('net_values', np.zeros(self.n_nodes))
            stats['num_contracts'] = self.contracts[t_idx].get('active_contracts', 0)

        return stats

class GCLSTM(nn.Module):
    """Graph Convolutional LSTM as described in the paper"""
    
    def __init__(self, in_channels: int, out_channels: int, K: int = 2):
        """
        Initialize GC-LSTM
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            K: Number of GCN layers
        """
        super(GCLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # GCN layers for each LSTM gate
        self.gcn_i = GCNConv(out_channels, out_channels)
        self.gcn_f = GCNConv(out_channels, out_channels)
        self.gcn_c = GCNConv(out_channels, out_channels)
        self.gcn_o = GCNConv(out_channels, out_channels)
        
        # Linear transformations for input
        self.W_i = nn.Linear(in_channels, out_channels)
        self.W_f = nn.Linear(in_channels, out_channels)
        self.W_c = nn.Linear(in_channels, out_channels)
        self.W_o = nn.Linear(in_channels, out_channels)
        
        # Biases
        self.b_i = nn.Parameter(torch.zeros(out_channels))
        self.b_f = nn.Parameter(torch.zeros(out_channels))
        self.b_c = nn.Parameter(torch.zeros(out_channels))
        self.b_o = nn.Parameter(torch.zeros(out_channels))
        
        # Cell and hidden states
        self.c = None
        self.h = None
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GC-LSTM
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph edges [2, num_edges]
            
        Returns:
            Updated node embeddings [num_nodes, out_channels]
        """
        batch_size = x.size(0)
        
        # Initialize states if needed
        if self.h is None or self.c is None:
            self.h = torch.zeros(batch_size, self.out_channels, device=x.device)
            self.c = torch.zeros(batch_size, self.out_channels, device=x.device)
        
        # Input gate
        i = torch.sigmoid(
            self.W_i(x) + self.gcn_i(self.h, edge_index) + self.b_i
        )
        
        # Forget gate
        f = torch.sigmoid(
            self.W_f(x) + self.gcn_f(self.h, edge_index) + self.b_f
        )
        
        # Cell state
        c_tilde = torch.tanh(
            self.W_c(x) + self.gcn_c(self.h, edge_index) + self.b_c
        )
        self.c = f * self.c + i * c_tilde
        
        # Output gate
        o = torch.sigmoid(
            self.W_o(x) + self.gcn_o(self.h, edge_index) + self.b_o
        )
        
        # Hidden state
        self.h = o * torch.tanh(self.c)
        
        return self.h
    
    def reset_states(self):
        """Reset cell and hidden states"""
        self.c = None
        self.h = None

class PricingModule(nn.Module):
    """Pricing module for contract valuation"""
    
    def __init__(self, contract_features: int = 6, hidden_size: int = 15, node_embedding_size: int = 15):
        """
        Initialize pricing module
        
        Args:
            contract_features: Number of features per contract
            hidden_size: Hidden size for LSTM
            node_embedding_size: Size of node embeddings from GNN
        """
        super(PricingModule, self).__init__()
        self.contract_features = contract_features
        self.hidden_size = hidden_size
        self.node_embedding_size = node_embedding_size
        
        # LSTM for processing contract features
        self.lstm = nn.LSTM(contract_features, hidden_size, batch_first=True)
        
        # FFNNs for predictions
        combined_size = hidden_size + 1 + node_embedding_size  # LSTM output + rate + node embedding
        
        # FFNN1 for next contract features
        self.ffnn1 = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, contract_features)
        )
        
        # FFNN2 for variation margin prediction
        self.ffnn2 = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, contract_matrix: torch.Tensor, future_rate: torch.Tensor, 
                node_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of pricing module
        
        Args:
            contract_matrix: Matrix of contract features [batch_size, seq_len, num_contracts, contract_features]
            future_rate: Future interest rate [batch_size]
            node_embedding: Node embeddings from GNN [batch_size, node_embedding_size]
            
        Returns:
            Tuple of (predicted_contract_features, predicted_variation_margin)
        """
        batch_size, seq_len, num_contracts, _ = contract_matrix.shape
        
        # Reshape for LSTM
        contract_matrix_reshaped = contract_matrix.view(batch_size * num_contracts, seq_len, self.contract_features)
        
        # Skip empty contracts
        non_zero_mask = torch.sum(torch.abs(contract_matrix_reshaped), dim=(1, 2)) > 0
        contract_matrix_filtered = contract_matrix_reshaped[non_zero_mask]
        
        if contract_matrix_filtered.size(0) == 0:
            # No valid contracts
            return (
                torch.zeros(batch_size, self.contract_features, device=contract_matrix.device),
                torch.zeros(batch_size, 1, device=contract_matrix.device)
            )
        
        # Process contracts with LSTM
        lstm_out, _ = self.lstm(contract_matrix_filtered)
        lstm_last = lstm_out[:, -1, :]  # Take last output
        
        # Expand future rate and node embedding for each contract
        expanded_rate = future_rate.unsqueeze(1).repeat(1, num_contracts).view(-1)[non_zero_mask].unsqueeze(1)
        expanded_embedding = node_embedding.unsqueeze(1).repeat(1, num_contracts, 1).view(batch_size * num_contracts, -1)[non_zero_mask]
        
        # Concatenate
        combined = torch.cat([lstm_last, expanded_rate, expanded_embedding], dim=1)
        
        # Get predictions
        contract_pred = self.ffnn1(combined)
        margin_pred = self.ffnn2(combined)
        
        # Sum up margins for all contracts
        margin_pred_full = torch.zeros(batch_size * num_contracts, 1, device=margin_pred.device)
        margin_pred_full[non_zero_mask] = margin_pred
        margin_pred_reshaped = margin_pred_full.view(batch_size, num_contracts, 1)
        total_margin = torch.sum(margin_pred_reshaped, dim=1)
        
        return contract_pred, total_margin

class DGNN(nn.Module):
    """Dynamic Graph Neural Network for conditional forecasting"""
    
    def __init__(self, node_features: int, contract_features: int, node_embedding_size: int = 15, 
                 hidden_size: int = 15, max_contracts: int = 20):
        """
        Initialize DGNN
        
        Args:
            node_features: Number of node features
            contract_features: Number of contract features
            node_embedding_size: Size of node embeddings
            hidden_size: Hidden size for LSTM
            max_contracts: Maximum number of contracts per node
        """
        super(DGNN, self).__init__()
        
        # GNN module
        self.gnn_module = GCLSTM(node_features, node_embedding_size)
        
        # Pricing module
        self.pricing_module = PricingModule(
            contract_features=contract_features,
            hidden_size=hidden_size,
            node_embedding_size=node_embedding_size
        )
        
        self.max_contracts = max_contracts
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                contract_matrix: torch.Tensor, future_rate: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DGNN
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph edges [2, num_edges]
            contract_matrix: Matrix of contract features [batch_size, seq_len, num_contracts, contract_features]
            future_rate: Future interest rate [batch_size]
            
        Returns:
            Predicted variation margins [batch_size, 1]
        """
        # Get node embeddings from GNN module
        node_embedding = self.gnn_module(x, edge_index)
        
        # Get predictions from pricing module
        _, margin_pred = self.pricing_module(contract_matrix, future_rate, node_embedding)
        
        return margin_pred
    
    def multi_step_forecast(self, x: torch.Tensor, edge_index: torch.Tensor, 
                          contract_matrix: torch.Tensor, future_rates: torch.Tensor, 
                          steps: int) -> torch.Tensor:
        """
        Multi-step ahead forecasting
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph edges [2, num_edges]
            contract_matrix: Initial contract matrix [batch_size, seq_len, num_contracts, contract_features]
            future_rates: Future interest rates for all steps [batch_size, steps]
            steps: Number of steps to forecast
            
        Returns:
            Predicted variation margins for all steps [batch_size, steps, 1]
        """
        batch_size = contract_matrix.size(0)
        predictions = []
        
        current_contracts = contract_matrix.clone()
        
        for step in range(steps):
            # Get current step's rate
            current_rate = future_rates[:, step]
            
            # Get node embeddings
            node_embedding = self.gnn_module(x, edge_index)
            
            # Get predictions from pricing module
            next_contracts, margin_pred = self.pricing_module(
                current_contracts, current_rate, node_embedding
            )
            
            # Store prediction
            predictions.append(margin_pred)
            
            # Update contract matrix for next step
            # Shift contract matrix one step forward and add new predicted contracts
            current_contracts = torch.roll(current_contracts, -1, dims=1)
            current_contracts[:, -1, :, :] = next_contracts.unsqueeze(1)
        
        return torch.stack(predictions, dim=1)

def create_enhanced_contract_matrix(contracts: Dict, node_idx: int, seq_len: int,
                                  max_contracts: int, contract_features: int = 7) -> torch.Tensor:
    """
    Create contract matrix for a specific node
    
    Args:
        contracts: Dictionary of contracts
        node_idx: Node index
        seq_len: Sequence length (lookback window)
        max_contracts: Maximum number of contracts to consider
        contract_features: Number of features per contract
    
    Returns:
        Contract matrix [1, seq_len, max_contracts, contract_features]
    """
    contract_matrix = torch.zeros(1, seq_len, max_contracts, contract_features)
    
    for t in range(seq_len):
        t_idx = t
        contract_count = 0
        
        if t_idx not in contracts:
            continue
            
        for contract_id, contract in contracts.items():
            if isinstance(contract_id, str) and contract_id.isdigit():
                if contract['start_time'] <= t_idx * (1/365) and contract['maturity'] > t_idx * (1/365):
                    i, j = contract['nodes']
                    if i == node_idx or j == node_idx:
                        # Node's position in contract (0 or 1)
                        idx = 0 if i == node_idx else 1
                        delta = contract['delta'][idx]
                        
                        # Contract features: [time_to_maturity, rate, principal, delta]
                        time_to_maturity = (contract['maturity'] - t_idx * (1/365)) * 365  # Convert to days
                        
                        # Enhanced contract features
                        contract_feature = [
                            time_to_maturity / 365,  # Normalize to years
                            contract['rate'],
                            1.0,  # Bond price at t0
                            1.0,  # Bond price at t
                            1.0,  # B(t0)
                            1.0,  # B(t)
                            delta,
                            contract.get('notional', 1000000) / 1000000  # Normalized notional
                        ]
                        
                        if contract_count < max_contracts:
                            contract_matrix[0, t, contract_count, :len(contract_feature)] = torch.tensor(contract_feature)
                            contract_count += 1
                        else:
                            break
    
    return contract_matrix

def test_enhanced_model():
    """Test the enhanced DGNN model with Bloomberg data integration"""

    # Parameters
    n_nodes = 5
    train_days = 365 * 1  # 1 year (reduced for faster testing with real data)
    test_days = 90        # 3 months
    total_days = train_days + test_days
    lookback = 5          # 5 days lookback
    forecast_horizon = 5  # 5 days ahead prediction
    currency = 'USD'      # Currency for market data

    print("=== Enhanced DGNN Model Test with Bloomberg Data ===")
    print(f"Currency: {currency}")
    print(f"Training period: {train_days} days")
    print(f"Test period: {test_days} days")
    print(f"Bloomberg available: {BLOOMBERG_AVAILABLE}")

    # Initialize Bloomberg data fetcher
    data_fetcher = BloombergDataFetcher()

    print("\nInitializing enhanced CIR process with market calibration...")
    # Enhanced CIR process with market calibration
    cir = EnhancedCIRProcess(data_fetcher=data_fetcher, currency=currency)
    rates = cir.simulate(total_days)

    print(f"Simulated {len(rates)} interest rate points")
    print(f"Rate range: {np.min(rates):.4f} - {np.max(rates):.4f}")

    print("\nInitializing enhanced financial network...")
    # Enhanced financial network with market data
    network = EnhancedFinancialNetwork(
        n_nodes=n_nodes,
        data_fetcher=data_fetcher,
        currency=currency
    )
    contracts_history = network.simulate_contracts(rates, total_days)
    
    print("\nPreparing enhanced training data...")
    # Prepare training data
    train_rates = rates[:train_days]
    test_rates = rates[train_days:]

    # Convert to PyTorch tensors
    node_features = torch.tensor(network.node_types.reshape(-1, 1), dtype=torch.float, requires_grad=False)

    # Display market data summary
    if network.yield_curve is not None:
        print(f"\nYield curve data points: {len(network.yield_curve)}")
        print(f"Yield curve range: {network.yield_curve['rate'].min():.4f} - {network.yield_curve['rate'].max():.4f}")
    print(f"Market volatility: {network.volatility:.4f}")

    # Create enhanced model
    model = DGNN(
        node_features=1,
        contract_features=7,  # Increased for enhanced contract features
        node_embedding_size=15,
        hidden_size=15,
        max_contracts=10
    )

    print(f"\nModel architecture:")
    print(f"- Node features: 1")
    print(f"- Contract features: 7")
    print(f"- Node embedding size: 15")
    print(f"- Max contracts per node: 10")
    
    # Enhanced training parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    num_epochs = 30  # Reduced for faster testing with real data
    batch_size = 8   # Reduced batch size

    print(f"\nTraining enhanced model...")
    print(f"- Epochs: {num_epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: 0.001")

    # Training loop with enhanced monitoring
    train_losses = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batches = 0
        
        # Reset GNN states
        model.gnn_module.reset_states()
        
        # Sample random starting points for sequences
        start_indices = np.random.randint(lookback, train_days - forecast_horizon - 1, size=batch_size)
        
        for start_idx in start_indices:
            # For each node
            for node_idx in range(n_nodes):
                # Create enhanced contract matrix
                contract_matrix = create_enhanced_contract_matrix(
                    network.contracts, node_idx, lookback, max_contracts=10, contract_features=7
                )
                
                # Get adjacency matrix and create edge index
                adj_matrix = network.get_adjacency_matrix(start_idx)
                edge_index = torch.tensor(np.array(np.where(adj_matrix > 0)), dtype=torch.long)
                
                # Get future rates
                future_rates = torch.tensor(rates[start_idx+1:start_idx+1+forecast_horizon],
                                          dtype=torch.float, requires_grad=False).unsqueeze(0)

                # Get target margins
                target_margins = []
                for i in range(1, forecast_horizon+1):
                    t_idx = start_idx + i
                    margin = network.get_variation_margins(t_idx)[node_idx]
                    target_margins.append(margin)

                target_margins = torch.tensor(target_margins, dtype=torch.float,
                                            requires_grad=False).unsqueeze(0).unsqueeze(-1)
                
                # Forward pass
                predicted_margins = model.multi_step_forecast(
                    node_features, edge_index, contract_matrix, future_rates, forecast_horizon
                )
                
                # Compute loss
                loss = F.mse_loss(predicted_margins, target_margins)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        avg_loss = epoch_loss / batches
        train_losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_enhanced_dgnn_model.pth')

    print(f"\nTraining completed. Best loss: {best_loss:.6f}")
    print("Testing enhanced model...")

    # Load best model for testing
    model.load_state_dict(torch.load('best_enhanced_dgnn_model.pth'))
    model.eval()
    test_losses = []
    
    # Test on each node
    for node_idx in range(n_nodes):
        # Sample test points
        test_start_idx = train_days + lookback
        test_end_idx = total_days - forecast_horizon - 1
        
        # Take 5 random test points
        test_indices = np.linspace(test_start_idx, test_end_idx, 5, dtype=int)
        
        node_predictions = []
        node_targets = []
        node_benchmarks = []
        
        for start_idx in test_indices:
            # Create enhanced contract matrix
            contract_matrix = create_enhanced_contract_matrix(
                network.contracts, node_idx, lookback, max_contracts=10, contract_features=7
            )
            
            # Get adjacency matrix and create edge index
            adj_matrix = network.get_adjacency_matrix(start_idx)
            edge_index = torch.tensor(np.array(np.where(adj_matrix > 0)), dtype=torch.long)
            
            # Get future rates
            future_rates = torch.tensor(rates[start_idx+1:start_idx+1+forecast_horizon], dtype=torch.float).unsqueeze(0)
            
            # Get target margins
            target_margins = []
            for i in range(1, forecast_horizon+1):
                t_idx = start_idx + i
                margin = network.get_variation_margins(t_idx)[node_idx]
                target_margins.append(margin)
            
            target_margins = torch.tensor(target_margins, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
            
            # Forward pass
            with torch.no_grad():
                predicted_margins = model.multi_step_forecast(
                    node_features, edge_index, contract_matrix, future_rates, forecast_horizon
                )
            
            # Compute loss
            loss = F.mse_loss(predicted_margins, target_margins)
            test_losses.append(loss.item())
            
            # Store predictions and targets for plotting
            node_predictions.append(predicted_margins.squeeze().numpy())
            node_targets.append(target_margins.squeeze().numpy())
            
            # Simple benchmark (theoretical best predictor would be more complex)
            benchmark = np.zeros(forecast_horizon)
            node_benchmarks.append(benchmark)
        
        # Plot results for this node
        plt.figure(figsize=(12, 6))
        plt.title(f"Node {node_idx} - {forecast_horizon}-steps ahead forecast")
        
        for i, start_idx in enumerate(test_indices):
            plt.subplot(len(test_indices), 1, i+1)
            time_points = np.arange(start_idx+1, start_idx+1+forecast_horizon)
            plt.plot(time_points, node_targets[i], 'b-', label='Actual')
            plt.plot(time_points, node_predictions[i], 'r--', label='Predicted')
            plt.plot(time_points, node_benchmarks[i], 'g-.', label='Benchmark')
            plt.ylabel(f'Margin at t={start_idx}')
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"enhanced_node_{node_idx}_forecast.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nTesting Results:")
    print(f"- Average test loss: {np.mean(test_losses):.6f}")
    print(f"- Number of test samples: {len(test_losses)}")

    # Plot enhanced training loss
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses)
    plt.title('Enhanced DGNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)

    # Plot rate evolution
    plt.subplot(2, 1, 2)
    plt.plot(rates[:100])  # First 100 days
    plt.title('Simulated Interest Rate Evolution (First 100 Days)')
    plt.xlabel('Days')
    plt.ylabel('Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("enhanced_training_results.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nEnhanced testing complete. Results saved as PNG files.")
    print("Files generated:")
    print("- enhanced_node_*_forecast.png: Individual node forecasts")
    print("- enhanced_training_results.png: Training loss and rate evolution")
    print("- best_enhanced_dgnn_model.pth: Best model weights")

def test_original_model():
    """Test the original DGNN model for comparison"""
    print("\n=== Running Original Model for Comparison ===")

    # Use original classes for comparison
    n_nodes = 5
    train_days = 365 * 1
    test_days = 90
    total_days = train_days + test_days

    # Original CIR process
    from scipy.stats import ncx2

    class OriginalCIRProcess:
        def __init__(self, k=0.6, theta=0.04, sigma=0.14, r0=0.04):
            self.k, self.theta, self.sigma, self.r0 = k, theta, sigma, r0

        def simulate(self, T, dt=1/365):
            n_steps = int(T / dt)
            rates = np.zeros(n_steps + 1)
            rates[0] = self.r0
            for i in range(1, n_steps + 1):
                c = (self.sigma ** 2) * (1 - np.exp(-self.k * dt)) / (4 * self.k)
                d = (4 * self.theta * self.k) / (self.sigma ** 2)
                nc = rates[i-1] * (np.exp(-self.k * dt) / c)
                rates[i] = c * ncx2.rvs(d, nc)
            return rates

    cir_orig = OriginalCIRProcess()
    rates_orig = cir_orig.simulate(total_days)

    print(f"Original model rate range: {np.min(rates_orig):.4f} - {np.max(rates_orig):.4f}")
    print("Original model testing completed for comparison.")

def main():
    """Main function to run enhanced DGNN tests"""
    print("Enhanced DGNN Implementation with Bloomberg Data Integration")
    print("=" * 60)

    try:
        # Test enhanced model
        test_enhanced_model()

        # Optionally test original model for comparison
        test_original_model()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")

        if BLOOMBERG_AVAILABLE:
            print("✓ Bloomberg data integration active")
        else:
            print("⚠ Bloomberg data not available - using simulated data")

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()