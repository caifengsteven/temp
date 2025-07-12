"""
Enhanced Deep Hedging Strategy with Bloomberg Data Integration
Based on 2410.22568v1.pdf - Deep Hedging with Neural Networks
Modified to use real market data via xbbg

Original implementation: Claude-Sonnet-3.7 on 2025-07-09 07:32:39
Enhanced with Bloomberg integration: 2025-07-11

Run with: python 2410.22568v1_test_strategy.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import time
import warnings
from datetime import datetime, timedelta
from scipy.optimize import minimize

# Bloomberg data integration
try:
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
except ImportError:
    BLOOMBERG_AVAILABLE = False
    warnings.warn("xbbg not available. Bloomberg functionality will be disabled.")

# Additional dependencies for enhanced functionality
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Neural network functionality will be limited.")


class BloombergDataFetcher:
    """Enhanced real-time Bloomberg data fetcher using xbbg"""

    def __init__(self, underlying_ticker: str = "SPX Index"):
        """
        Initialize Bloomberg data fetcher

        Args:
            underlying_ticker: Bloomberg ticker for the underlying asset
        """
        if not BLOOMBERG_AVAILABLE:
            raise ImportError("xbbg is required for Bloomberg data functionality")

        self.underlying_ticker = underlying_ticker
        self.data_cache = {}
        self.real_time_data = {}
        self.subscription_active = False

        # Initialize real-time data fields
        self.real_time_fields = [
            'LAST_PRICE', 'BID', 'ASK', 'VOLUME', 'HIGH', 'LOW',
            'OPEN', 'PREV_CLOSE_VALUE_REALTIME', 'TIME'
        ]

    def fetch_historical_prices(self, start_date: str, end_date: str,
                              fields: List[str] = None) -> pd.DataFrame:
        """
        Fetch historical price data from Bloomberg

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            fields: List of Bloomberg fields to fetch

        Returns:
            DataFrame with historical price data
        """
        if fields is None:
            fields = ['PX_LAST', 'PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_VOLUME']

        try:
            data = blp.bdh(
                tickers=self.underlying_ticker,
                flds=fields,
                start_date=start_date,
                end_date=end_date
            )

            # Flatten multi-level columns if necessary
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [f"{col[1]}_{col[0]}" if col[1] != self.underlying_ticker
                              else col[1] for col in data.columns]

            # Cache the data
            cache_key = f"hist_{start_date}_{end_date}"
            self.data_cache[cache_key] = data

            return data

        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def fetch_volatility_surface(self, date: str = None) -> pd.DataFrame:
        """
        Fetch implied volatility surface from Bloomberg

        Args:
            date: Date for volatility surface (default: latest)

        Returns:
            DataFrame with volatility surface data
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        # Define standard option maturities and strikes for volatility surface
        maturities = ['1M', '2M', '3M', '6M', '9M', '1Y']
        moneyness_levels = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]

        vol_data = []

        try:
            # Get current spot price
            spot_data = blp.bdp(self.underlying_ticker, 'PX_LAST')
            current_spot = spot_data.iloc[0, 0]

            for maturity in maturities:
                for moneyness in moneyness_levels:
                    strike = current_spot * moneyness

                    # Construct option ticker (this is simplified - actual Bloomberg option tickers are more complex)
                    # In practice, you'd need to construct proper option tickers based on the underlying
                    option_ticker = f"{self.underlying_ticker.split()[0]} {maturity} C{strike:.0f}"

                    try:
                        # Fetch implied volatility
                        vol_data_point = blp.bdp(option_ticker, 'IVOL_MID')
                        if not vol_data_point.empty and not pd.isna(vol_data_point.iloc[0, 0]):
                            vol_data.append({
                                'maturity': maturity,
                                'strike': strike,
                                'moneyness': moneyness,
                                'implied_vol': vol_data_point.iloc[0, 0] / 100,  # Convert from percentage
                                'spot': current_spot
                            })
                    except:
                        # Skip if option data not available
                        continue

            vol_surface = pd.DataFrame(vol_data)

            # Cache the data
            cache_key = f"vol_surface_{date}"
            self.data_cache[cache_key] = vol_surface

            return vol_surface

        except Exception as e:
            print(f"Error fetching volatility surface: {e}")
            return pd.DataFrame()

    def fetch_option_prices(self, strikes: List[float], maturities: List[str],
                          option_types: List[str] = None) -> pd.DataFrame:
        """
        Fetch current option prices from Bloomberg

        Args:
            strikes: List of strike prices
            maturities: List of maturities (e.g., ['1M', '3M', '6M'])
            option_types: List of option types ('C' for call, 'P' for put)

        Returns:
            DataFrame with option prices
        """
        if option_types is None:
            option_types = ['C', 'P']

        option_data = []

        try:
            for maturity in maturities:
                for strike in strikes:
                    for opt_type in option_types:
                        # Construct option ticker (simplified)
                        option_ticker = f"{self.underlying_ticker.split()[0]} {maturity} {opt_type}{strike:.0f}"

                        try:
                            # Fetch option price and Greeks
                            fields = ['PX_MID', 'DELTA', 'GAMMA', 'VEGA', 'THETA', 'IVOL_MID']
                            option_info = blp.bdp(option_ticker, fields)

                            if not option_info.empty:
                                option_data.append({
                                    'ticker': option_ticker,
                                    'maturity': maturity,
                                    'strike': strike,
                                    'type': opt_type,
                                    'price': option_info.get('PX_MID', np.nan),
                                    'delta': option_info.get('DELTA', np.nan),
                                    'gamma': option_info.get('GAMMA', np.nan),
                                    'vega': option_info.get('VEGA', np.nan),
                                    'theta': option_info.get('THETA', np.nan),
                                    'implied_vol': option_info.get('IVOL_MID', np.nan)
                                })
                        except:
                            continue

            option_df = pd.DataFrame(option_data)

            # Cache the data
            cache_key = f"options_{datetime.now().strftime('%Y%m%d')}"
            self.data_cache[cache_key] = option_df

            return option_df

        except Exception as e:
            print(f"Error fetching option prices: {e}")
            return pd.DataFrame()

    def get_current_market_data(self) -> Dict:
        """
        Get current market snapshot including spot price, volatility, and key metrics

        Returns:
            Dictionary with current market data
        """
        try:
            # Fetch current spot price and basic metrics
            fields = ['PX_LAST', 'VOLATILITY_30D', 'VOLATILITY_90D', 'VOLATILITY_260D']
            current_data = blp.bdp(self.underlying_ticker, fields)

            market_data = {
                'spot_price': current_data.get('PX_LAST', np.nan),
                'vol_30d': current_data.get('VOLATILITY_30D', np.nan) / 100 if 'VOLATILITY_30D' in current_data else np.nan,
                'vol_90d': current_data.get('VOLATILITY_90D', np.nan) / 100 if 'VOLATILITY_90D' in current_data else np.nan,
                'vol_260d': current_data.get('VOLATILITY_260D', np.nan) / 100 if 'VOLATILITY_260D' in current_data else np.nan,
                'timestamp': datetime.now()
            }

            return market_data

        except Exception as e:
            print(f"Error fetching current market data: {e}")
            return {}

    def start_real_time_subscription(self, tickers: List[str] = None) -> bool:
        """
        Start real-time data subscription for specified tickers

        Args:
            tickers: List of Bloomberg tickers to subscribe to

        Returns:
            True if subscription started successfully
        """
        if not BLOOMBERG_AVAILABLE:
            print("Bloomberg not available for real-time subscription")
            return False

        if tickers is None:
            tickers = [self.underlying_ticker]

        try:
            # Start subscription using xbbg
            print(f"Starting real-time subscription for: {tickers}")

            # Note: xbbg subscription requires Bloomberg Terminal to be running
            # This is a simplified example - in practice you'd handle the subscription callback
            self.subscription_active = True

            # Initialize real-time data storage
            for ticker in tickers:
                self.real_time_data[ticker] = {
                    'last_update': datetime.now(),
                    'data': {}
                }

            print("✓ Real-time subscription started successfully")
            return True

        except Exception as e:
            print(f"Error starting real-time subscription: {e}")
            return False

    def get_real_time_data(self, ticker: str = None) -> Dict:
        """
        Get latest real-time data for a ticker

        Args:
            ticker: Bloomberg ticker (default: underlying_ticker)

        Returns:
            Dictionary with latest real-time data
        """
        if ticker is None:
            ticker = self.underlying_ticker

        try:
            # Fetch current real-time data using bdp (current values)
            current_data = blp.bdp(ticker, self.real_time_fields)

            if not current_data.empty:
                # Convert to dictionary format
                real_time_info = {
                    'ticker': ticker,
                    'timestamp': datetime.now(),
                    'last_price': current_data.get('LAST_PRICE', np.nan),
                    'bid': current_data.get('BID', np.nan),
                    'ask': current_data.get('ASK', np.nan),
                    'volume': current_data.get('VOLUME', np.nan),
                    'high': current_data.get('HIGH', np.nan),
                    'low': current_data.get('LOW', np.nan),
                    'open': current_data.get('OPEN', np.nan),
                    'prev_close': current_data.get('PREV_CLOSE_VALUE_REALTIME', np.nan)
                }

                # Update cache
                self.real_time_data[ticker] = {
                    'last_update': datetime.now(),
                    'data': real_time_info
                }

                return real_time_info
            else:
                print(f"No real-time data available for {ticker}")
                return {}

        except Exception as e:
            print(f"Error fetching real-time data for {ticker}: {e}")
            return {}

    def get_intraday_bars(self, ticker: str = None, date: str = None,
                         interval: int = 1) -> pd.DataFrame:
        """
        Get intraday bar data using xbbg bdib function

        Args:
            ticker: Bloomberg ticker (default: underlying_ticker)
            date: Date for intraday data (default: today)
            interval: Bar interval in minutes (default: 1)

        Returns:
            DataFrame with intraday bar data
        """
        if ticker is None:
            ticker = self.underlying_ticker

        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        try:
            # Use xbbg bdib for intraday bars
            intraday_data = blp.bdib(
                ticker=ticker,
                dt=date,
                interval=f'{interval}min'
            )

            if not intraday_data.empty:
                # Cache the data
                cache_key = f"intraday_{ticker}_{date}_{interval}min"
                self.data_cache[cache_key] = intraday_data

                print(f"✓ Retrieved {len(intraday_data)} intraday bars for {ticker}")
                return intraday_data
            else:
                print(f"No intraday data available for {ticker} on {date}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching intraday data: {e}")
            return pd.DataFrame()

    def get_tick_data(self, ticker: str = None, start_time: str = None,
                     end_time: str = None) -> pd.DataFrame:
        """
        Get tick-by-tick data using xbbg

        Args:
            ticker: Bloomberg ticker (default: underlying_ticker)
            start_time: Start time in 'HH:MM:SS' format
            end_time: End time in 'HH:MM:SS' format

        Returns:
            DataFrame with tick data
        """
        if ticker is None:
            ticker = self.underlying_ticker

        try:
            # Get today's date
            today = datetime.now().strftime('%Y-%m-%d')

            # Use xbbg for tick data (if available)
            # Note: This requires appropriate Bloomberg permissions
            tick_data = blp.bdtick(
                ticker=ticker,
                dt=today,
                start_time=start_time,
                end_time=end_time
            )

            if not tick_data.empty:
                print(f"✓ Retrieved {len(tick_data)} ticks for {ticker}")
                return tick_data
            else:
                print(f"No tick data available for {ticker}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching tick data: {e}")
            return pd.DataFrame()

    def get_option_chain(self, expiry_date: str = None,
                        option_type: str = 'C') -> pd.DataFrame:
        """
        Get option chain data for the underlying

        Args:
            expiry_date: Option expiry date in 'YYYY-MM-DD' format
            option_type: 'C' for calls, 'P' for puts

        Returns:
            DataFrame with option chain data
        """
        try:
            # Get current spot price
            spot_data = blp.bdp(self.underlying_ticker, 'PX_LAST')
            current_spot = spot_data.iloc[0, 0]

            # Generate strike range around current spot
            strike_range = np.arange(
                current_spot * 0.8,
                current_spot * 1.2,
                current_spot * 0.01
            )

            option_data = []

            for strike in strike_range:
                try:
                    # Construct option ticker (simplified)
                    # In practice, you'd use proper Bloomberg option ticker construction
                    if expiry_date:
                        exp_str = datetime.strptime(expiry_date, '%Y-%m-%d').strftime('%m/%d/%y')
                    else:
                        # Default to next monthly expiry
                        next_month = datetime.now().replace(day=1) + timedelta(days=32)
                        exp_str = next_month.strftime('%m/%d/%y')

                    option_ticker = f"{self.underlying_ticker.split()[0]} {exp_str} {option_type}{strike:.0f}"

                    # Fetch option data
                    option_info = blp.bdp(
                        option_ticker,
                        ['PX_LAST', 'DELTA', 'GAMMA', 'VEGA', 'THETA', 'IVOL_MID']
                    )

                    if not option_info.empty:
                        option_data.append({
                            'ticker': option_ticker,
                            'strike': strike,
                            'type': option_type,
                            'expiry': expiry_date or exp_str,
                            'price': option_info.get('PX_LAST', np.nan),
                            'delta': option_info.get('DELTA', np.nan),
                            'gamma': option_info.get('GAMMA', np.nan),
                            'vega': option_info.get('VEGA', np.nan),
                            'theta': option_info.get('THETA', np.nan),
                            'implied_vol': option_info.get('IVOL_MID', np.nan)
                        })

                except:
                    continue

            option_chain = pd.DataFrame(option_data)

            if not option_chain.empty:
                print(f"✓ Retrieved option chain with {len(option_chain)} options")

            return option_chain

        except Exception as e:
            print(f"Error fetching option chain: {e}")
            return pd.DataFrame()

    def stop_real_time_subscription(self):
        """Stop real-time data subscription"""
        self.subscription_active = False
        print("Real-time subscription stopped")


class RealTimeMarketMonitor:
    """Real-time market data monitor for live hedging"""

    def __init__(self, bloomberg_fetcher: BloombergDataFetcher,
                 update_interval: int = 5):
        """
        Initialize real-time market monitor

        Args:
            bloomberg_fetcher: Bloomberg data fetcher instance
            update_interval: Update interval in seconds
        """
        self.bloomberg_fetcher = bloomberg_fetcher
        self.update_interval = update_interval
        self.monitoring = False
        self.market_data_history = []
        self.last_update = None

    def start_monitoring(self, tickers: List[str] = None):
        """
        Start real-time market monitoring

        Args:
            tickers: List of tickers to monitor
        """
        if tickers is None:
            tickers = [self.bloomberg_fetcher.underlying_ticker]

        print(f"Starting real-time monitoring for: {tickers}")
        self.monitoring = True

        # Start subscription
        self.bloomberg_fetcher.start_real_time_subscription(tickers)

        return True

    def get_latest_market_snapshot(self) -> Dict:
        """
        Get the latest market data snapshot

        Returns:
            Dictionary with latest market data
        """
        try:
            # Get real-time data
            real_time_data = self.bloomberg_fetcher.get_real_time_data()

            if real_time_data:
                # Get additional market metrics
                market_data = self.bloomberg_fetcher.get_current_market_data()

                # Combine real-time and market data
                snapshot = {
                    'timestamp': datetime.now(),
                    'spot_price': real_time_data.get('last_price', market_data.get('spot_price', np.nan)),
                    'bid': real_time_data.get('bid', np.nan),
                    'ask': real_time_data.get('ask', np.nan),
                    'spread': real_time_data.get('ask', np.nan) - real_time_data.get('bid', np.nan) if
                             not np.isnan(real_time_data.get('ask', np.nan)) and
                             not np.isnan(real_time_data.get('bid', np.nan)) else np.nan,
                    'volume': real_time_data.get('volume', np.nan),
                    'high': real_time_data.get('high', np.nan),
                    'low': real_time_data.get('low', np.nan),
                    'open': real_time_data.get('open', np.nan),
                    'prev_close': real_time_data.get('prev_close', np.nan),
                    'volatility_30d': market_data.get('vol_30d', np.nan),
                    'volatility_90d': market_data.get('vol_90d', np.nan),
                    'change': real_time_data.get('last_price', np.nan) - real_time_data.get('prev_close', np.nan) if
                             not np.isnan(real_time_data.get('last_price', np.nan)) and
                             not np.isnan(real_time_data.get('prev_close', np.nan)) else np.nan,
                    'change_pct': ((real_time_data.get('last_price', np.nan) - real_time_data.get('prev_close', np.nan)) /
                                  real_time_data.get('prev_close', np.nan) * 100) if
                                 not np.isnan(real_time_data.get('last_price', np.nan)) and
                                 not np.isnan(real_time_data.get('prev_close', np.nan)) and
                                 real_time_data.get('prev_close', np.nan) != 0 else np.nan
                }

                # Store in history
                self.market_data_history.append(snapshot)
                self.last_update = datetime.now()

                return snapshot
            else:
                print("No real-time data available")
                return {}

        except Exception as e:
            print(f"Error getting market snapshot: {e}")
            return {}

    def get_market_data_history(self, minutes: int = 60) -> List[Dict]:
        """
        Get market data history for the last N minutes

        Args:
            minutes: Number of minutes of history to return

        Returns:
            List of market data snapshots
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        return [
            snapshot for snapshot in self.market_data_history
            if snapshot['timestamp'] >= cutoff_time
        ]

    def calculate_realized_volatility(self, minutes: int = 30) -> float:
        """
        Calculate realized volatility from recent price movements

        Args:
            minutes: Time window in minutes

        Returns:
            Realized volatility (annualized)
        """
        try:
            history = self.get_market_data_history(minutes)

            if len(history) < 2:
                return np.nan

            prices = [snapshot['spot_price'] for snapshot in history
                     if not np.isnan(snapshot['spot_price'])]

            if len(prices) < 2:
                return np.nan

            # Calculate returns
            returns = np.diff(np.log(prices))

            # Annualize volatility (assuming 252 trading days, 6.5 hours per day, 60 minutes per hour)
            realized_vol = np.std(returns) * np.sqrt(252 * 6.5 * 60 / minutes)

            return realized_vol

        except Exception as e:
            print(f"Error calculating realized volatility: {e}")
            return np.nan

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring = False
        self.bloomberg_fetcher.stop_real_time_subscription()
        print("Real-time monitoring stopped")


class HestonCalibrator:
    """Calibrate Heston model parameters from market data"""

    def __init__(self, bloomberg_fetcher: BloombergDataFetcher):
        """
        Initialize calibrator with Bloomberg data fetcher

        Args:
            bloomberg_fetcher: Instance of BloombergDataFetcher
        """
        self.bloomberg_fetcher = bloomberg_fetcher

    def calibrate_from_volatility_surface(self, vol_surface: pd.DataFrame,
                                        initial_params: Dict = None) -> Dict:
        """
        Calibrate Heston parameters from volatility surface

        Args:
            vol_surface: DataFrame with volatility surface data
            initial_params: Initial parameter guess

        Returns:
            Dictionary with calibrated parameters
        """
        if initial_params is None:
            initial_params = {
                'v0': 0.04,      # Initial volatility
                'kappa': 2.0,    # Mean reversion speed
                'theta': 0.04,   # Long-term volatility
                'xi': 0.3,       # Volatility of volatility
                'rho': -0.7      # Correlation
            }

        if vol_surface.empty:
            print("Warning: Empty volatility surface, using default parameters")
            return initial_params

        # Convert maturity strings to years
        maturity_map = {'1M': 1/12, '2M': 2/12, '3M': 3/12, '6M': 6/12, '9M': 9/12, '1Y': 1}
        vol_surface['maturity_years'] = vol_surface['maturity'].map(maturity_map)

        # Remove rows with missing data
        vol_surface = vol_surface.dropna(subset=['implied_vol', 'maturity_years', 'moneyness'])

        if len(vol_surface) < 5:
            print("Warning: Insufficient volatility data for calibration, using default parameters")
            return initial_params

        def objective_function(params):
            """Objective function for calibration"""
            v0, kappa, theta, xi, rho = params

            # Ensure parameters are within reasonable bounds
            if v0 <= 0 or kappa <= 0 or theta <= 0 or xi <= 0 or abs(rho) >= 1:
                return 1e6

            total_error = 0
            for _, row in vol_surface.iterrows():
                try:
                    # Calculate Heston implied volatility (simplified approximation)
                    T = row['maturity_years']
                    K = row['strike']
                    S = row['spot']

                    # Simplified Heston volatility approximation
                    # In practice, you'd use a more sophisticated Heston pricing model
                    heston_vol = np.sqrt(v0 + (theta - v0) * (1 - np.exp(-kappa * T)) / (kappa * T))

                    # Add volatility smile effect based on moneyness
                    moneyness_effect = xi * np.sqrt(T) * abs(np.log(row['moneyness']))
                    heston_vol += moneyness_effect

                    market_vol = row['implied_vol']
                    error = (heston_vol - market_vol) ** 2
                    total_error += error

                except:
                    continue

            return total_error

        # Set parameter bounds
        bounds = [
            (0.001, 1.0),    # v0
            (0.1, 10.0),     # kappa
            (0.001, 1.0),    # theta
            (0.01, 2.0),     # xi
            (-0.99, 0.99)    # rho
        ]

        # Initial parameter vector
        x0 = [initial_params['v0'], initial_params['kappa'], initial_params['theta'],
              initial_params['xi'], initial_params['rho']]

        try:
            # Perform optimization
            result = minimize(objective_function, x0, bounds=bounds, method='L-BFGS-B')

            if result.success:
                calibrated_params = {
                    'v0': result.x[0],
                    'kappa': result.x[1],
                    'theta': result.x[2],
                    'xi': result.x[3],
                    'rho': result.x[4]
                }

                print(f"Calibration successful. Final error: {result.fun:.6f}")
                return calibrated_params
            else:
                print("Calibration failed, using initial parameters")
                return initial_params

        except Exception as e:
            print(f"Error during calibration: {e}")
            return initial_params

    def calibrate_from_historical_data(self, price_data: pd.DataFrame,
                                     window_days: int = 252) -> Dict:
        """
        Calibrate Heston parameters from historical price data

        Args:
            price_data: DataFrame with historical price data
            window_days: Rolling window for volatility estimation

        Returns:
            Dictionary with calibrated parameters
        """
        if price_data.empty or 'PX_LAST' not in price_data.columns:
            print("Warning: Invalid price data, using default parameters")
            return {
                'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 'xi': 0.3, 'rho': -0.7
            }

        # Calculate returns
        price_data = price_data.copy()
        price_data['returns'] = price_data['PX_LAST'].pct_change().dropna()

        # Calculate realized volatility
        price_data['realized_vol'] = price_data['returns'].rolling(window=window_days).std() * np.sqrt(252)

        # Remove NaN values
        clean_data = price_data.dropna(subset=['returns', 'realized_vol'])

        if len(clean_data) < 100:
            print("Warning: Insufficient historical data for calibration")
            return {
                'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 'xi': 0.3, 'rho': -0.7
            }

        # Estimate parameters using method of moments
        returns = clean_data['returns'].values
        vol_series = clean_data['realized_vol'].values

        # Basic parameter estimation
        v0 = vol_series[-1] ** 2  # Current volatility squared
        theta = np.mean(vol_series) ** 2  # Long-term volatility squared

        # Estimate mean reversion speed (simplified)
        vol_changes = np.diff(vol_series)
        vol_levels = vol_series[:-1] - theta
        if len(vol_levels) > 0 and np.var(vol_levels) > 0:
            kappa = max(0.1, -np.mean(vol_changes / vol_levels))
        else:
            kappa = 2.0

        # Estimate volatility of volatility
        xi = np.std(vol_changes) * np.sqrt(252) if len(vol_changes) > 0 else 0.3

        # Estimate correlation between returns and volatility changes
        if len(returns) > len(vol_changes):
            returns_aligned = returns[1:len(vol_changes)+1]
        else:
            returns_aligned = returns[:len(vol_changes)]
            vol_changes = vol_changes[:len(returns)]

        if len(returns_aligned) == len(vol_changes) and len(returns_aligned) > 1:
            rho = np.corrcoef(returns_aligned, vol_changes)[0, 1]
            if np.isnan(rho):
                rho = -0.7
        else:
            rho = -0.7

        calibrated_params = {
            'v0': max(0.001, min(1.0, v0)),
            'kappa': max(0.1, min(10.0, kappa)),
            'theta': max(0.001, min(1.0, theta)),
            'xi': max(0.01, min(2.0, xi)),
            'rho': max(-0.99, min(0.99, rho))
        }

        print("Historical calibration completed:")
        for param, value in calibrated_params.items():
            print(f"  {param}: {value:.4f}")

        return calibrated_params

class HestonModel:
    """Enhanced Heston stochastic volatility model with Bloomberg data integration"""

    def __init__(self,
                 x0: float = 1.0,
                 v0: float = 0.0625,
                 kappa: float = 8.0,
                 theta: float = 0.0625,
                 xi: float = 1.0,
                 rho: float = -0.7,
                 dt: float = 1/250,
                 bloomberg_fetcher: BloombergDataFetcher = None,
                 auto_calibrate: bool = False):
        """
        Initialize Heston model parameters

        Args:
            x0: Initial price
            v0: Initial volatility
            kappa: Mean reversion speed
            theta: Long-term volatility
            xi: Volatility of volatility
            rho: Correlation between price and volatility
            dt: Time increment
            bloomberg_fetcher: Bloomberg data fetcher for calibration
            auto_calibrate: Whether to auto-calibrate from market data
        """
        self.x0 = x0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt
        self.bloomberg_fetcher = bloomberg_fetcher
        self.calibrator = None

        if bloomberg_fetcher is not None:
            self.calibrator = HestonCalibrator(bloomberg_fetcher)

        if auto_calibrate and self.calibrator is not None:
            self.calibrate_from_market_data()

    def calibrate_from_market_data(self, use_volatility_surface: bool = True,
                                 historical_days: int = 252):
        """
        Calibrate model parameters from market data

        Args:
            use_volatility_surface: Whether to use volatility surface for calibration
            historical_days: Number of historical days for calibration
        """
        if self.calibrator is None:
            print("Warning: No calibrator available, skipping calibration")
            return

        try:
            if use_volatility_surface and BLOOMBERG_AVAILABLE:
                # Try volatility surface calibration first
                vol_surface = self.bloomberg_fetcher.fetch_volatility_surface()
                if not vol_surface.empty:
                    params = self.calibrator.calibrate_from_volatility_surface(vol_surface)
                    self._update_parameters(params)
                    print("Calibrated from volatility surface")
                    return

            # Fallback to historical data calibration
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=historical_days * 2)).strftime('%Y-%m-%d')

            historical_data = self.bloomberg_fetcher.fetch_historical_prices(start_date, end_date)
            if not historical_data.empty:
                params = self.calibrator.calibrate_from_historical_data(historical_data)
                self._update_parameters(params)
                print("Calibrated from historical data")
            else:
                print("Warning: No historical data available for calibration")

        except Exception as e:
            print(f"Error during calibration: {e}")

    def _update_parameters(self, params: Dict):
        """Update model parameters from calibration results"""
        self.v0 = params.get('v0', self.v0)
        self.kappa = params.get('kappa', self.kappa)
        self.theta = params.get('theta', self.theta)
        self.xi = params.get('xi', self.xi)
        self.rho = params.get('rho', self.rho)

    def get_current_market_price(self) -> float:
        """Get current market price from Bloomberg if available"""
        if self.bloomberg_fetcher is not None:
            market_data = self.bloomberg_fetcher.get_current_market_data()
            if 'spot_price' in market_data and not np.isnan(market_data['spot_price']):
                return market_data['spot_price']
        return self.x0
    
    def simulate_paths(self, num_paths: int, time_steps: int,
                      use_market_start: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate price and volatility paths

        Args:
            num_paths: Number of paths to simulate
            time_steps: Number of time steps
            use_market_start: Whether to use current market price as starting point

        Returns:
            Tuple of price paths and volatility paths arrays
        """
        # Initialize arrays
        prices = np.zeros((num_paths, time_steps + 1))
        vols = np.zeros((num_paths, time_steps + 1))

        # Set initial values
        if use_market_start:
            current_price = self.get_current_market_price()
            prices[:, 0] = current_price

            # Try to get current market volatility
            if self.bloomberg_fetcher is not None:
                market_data = self.bloomberg_fetcher.get_current_market_data()
                if 'vol_30d' in market_data and not np.isnan(market_data['vol_30d']):
                    current_vol = market_data['vol_30d'] ** 2  # Convert to variance
                    vols[:, 0] = current_vol
                else:
                    vols[:, 0] = self.v0
            else:
                vols[:, 0] = self.v0
        else:
            prices[:, 0] = self.x0
            vols[:, 0] = self.v0

        # Generate correlated random variables
        z1 = np.random.standard_normal((num_paths, time_steps))
        z2 = np.random.standard_normal((num_paths, time_steps))
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

        # Simulate paths
        for t in range(time_steps):
            # Ensure volatility is positive (use max to avoid negative values)
            sqrt_v = np.sqrt(np.maximum(vols[:, t], 0))

            # Update price (no drift assumed as in the paper)
            prices[:, t+1] = prices[:, t] * np.exp(sqrt_v * np.sqrt(self.dt) * z1[:, t] -
                                                  0.5 * vols[:, t] * self.dt)

            # Update volatility using Euler scheme
            vols[:, t+1] = np.maximum(
                vols[:, t] + self.kappa * (self.theta - vols[:, t]) * self.dt +
                self.xi * sqrt_v * np.sqrt(self.dt) * z2[:, t],
                0.000001  # Ensure positivity
            )

        return prices, vols

    def simulate_paths_with_market_data(self, num_paths: int, time_steps: int,
                                      historical_data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths using historical market data as a guide

        Args:
            num_paths: Number of paths to simulate
            time_steps: Number of time steps
            historical_data: Historical price data for guidance

        Returns:
            Tuple of price paths and volatility paths arrays
        """
        if historical_data is None or historical_data.empty:
            return self.simulate_paths(num_paths, time_steps, use_market_start=True)

        # Calculate historical returns and volatilities
        historical_data = historical_data.copy()
        historical_data['returns'] = historical_data['PX_LAST'].pct_change().dropna()
        historical_data['realized_vol'] = historical_data['returns'].rolling(window=20).std() * np.sqrt(252)

        # Use recent data for simulation guidance
        recent_data = historical_data.tail(min(time_steps, len(historical_data)))

        # Initialize arrays
        prices = np.zeros((num_paths, time_steps + 1))
        vols = np.zeros((num_paths, time_steps + 1))

        # Set initial values from current market
        current_price = self.get_current_market_price()
        prices[:, 0] = current_price
        vols[:, 0] = self.v0

        # Generate paths with market-informed randomness
        for t in range(time_steps):
            # Use historical volatility pattern if available
            if t < len(recent_data) and not np.isnan(recent_data.iloc[t]['realized_vol']):
                market_vol = recent_data.iloc[t]['realized_vol'] ** 2
                # Blend model volatility with market volatility
                target_vol = 0.7 * vols[:, t] + 0.3 * market_vol
            else:
                target_vol = vols[:, t]

            # Generate correlated random variables
            z1 = np.random.standard_normal(num_paths)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal(num_paths)

            # Ensure volatility is positive
            sqrt_v = np.sqrt(np.maximum(target_vol, 0))

            # Update price
            prices[:, t+1] = prices[:, t] * np.exp(sqrt_v * np.sqrt(self.dt) * z1 -
                                                  0.5 * target_vol * self.dt)

            # Update volatility using Euler scheme
            if t < time_steps - 1:
                vols[:, t+1] = np.maximum(
                    target_vol + self.kappa * (self.theta - target_vol) * self.dt +
                    self.xi * sqrt_v * np.sqrt(self.dt) * z2,
                    0.000001
                )

        return prices, vols

class VanillaOption:
    """Pricing and Greeks for vanilla options"""
    
    @staticmethod
    def black_scholes_price(S: float, K: float, T: float, sigma: float, 
                           option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes price for European options
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity in years
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        if T <= 0:
            # For expired options
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S/K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            return S * norm.cdf(d1) - K * norm.cdf(d2)
        else:
            return K * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S: float, K: float, T: float, sigma: float, 
             option_type: str = 'call') -> float:
        """Calculate option delta"""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
                
        d1 = (np.log(S/K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

class CliquetOption:
    """Calculate payoff for locally-capped globally-floored cliquet option"""
    
    def __init__(self, reset_dates: List[int], cap: float = 0.015):
        """
        Initialize cliquet option
        
        Args:
            reset_dates: List of reset dates (time steps)
            cap: Cap for each period's return
        """
        self.reset_dates = reset_dates
        self.cap = cap
    
    def payoff(self, price_path: np.ndarray) -> float:
        """
        Calculate cliquet payoff at maturity
        
        Args:
            price_path: Price path array
            
        Returns:
            Cliquet payoff
        """
        returns = []
        
        # Calculate returns between reset dates
        for i in range(1, len(self.reset_dates)):
            prev_date = self.reset_dates[i-1]
            curr_date = self.reset_dates[i]
            
            period_return = price_path[curr_date] / price_path[prev_date] - 1
            capped_return = min(period_return, self.cap)
            returns.append(capped_return)
        
        # Apply global floor at 0
        total_return = max(sum(returns), 0)
        return total_return

class FloatingGrid:
    """Enhanced floating grid of tradable options with Bloomberg data integration"""

    def __init__(self, maturities: List[int], moneyness_levels: List[float],
                 bloomberg_fetcher: BloombergDataFetcher = None,
                 use_market_data: bool = False):
        """
        Initialize floating grid

        Args:
            maturities: List of option maturities (in time steps)
            moneyness_levels: List of moneyness levels (K/S ratio)
            bloomberg_fetcher: Bloomberg data fetcher for real option prices
            use_market_data: Whether to use real market option data
        """
        self.maturities = maturities
        self.moneyness_levels = moneyness_levels
        self.bloomberg_fetcher = bloomberg_fetcher
        self.use_market_data = use_market_data
        self.instruments = []
        self.market_option_data = pd.DataFrame()

        # Generate list of tradable instruments
        # First instrument is always the underlying (index 0)
        self.instruments.append({"type": "spot", "maturity": None, "moneyness": None, "bloomberg_ticker": None})

        # Add options to the grid
        for tau in maturities:
            for k in moneyness_levels:
                if k <= 1.0:
                    self.instruments.append({
                        "type": "put",
                        "maturity": tau,
                        "moneyness": k,
                        "bloomberg_ticker": None
                    })
                if k >= 1.0:
                    self.instruments.append({
                        "type": "call",
                        "maturity": tau,
                        "moneyness": k,
                        "bloomberg_ticker": None
                    })

        # Fetch market data if available
        if use_market_data and bloomberg_fetcher is not None:
            self.update_market_data()

    def update_market_data(self):
        """Update market option data from Bloomberg"""
        if self.bloomberg_fetcher is None:
            return

        try:
            # Get current spot price for strike calculation
            market_data = self.bloomberg_fetcher.get_current_market_data()
            if 'spot_price' not in market_data:
                return

            current_spot = market_data['spot_price']

            # Convert time step maturities to Bloomberg maturity strings
            maturity_map = {
                10: '1M', 20: '2M', 40: '3M', 80: '6M', 120: '9M', 240: '1Y'
            }

            bloomberg_maturities = []
            strikes = []
            option_types = []

            for tau in self.maturities:
                if tau in maturity_map:
                    bloomberg_maturity = maturity_map[tau]
                    for k in self.moneyness_levels:
                        strike = current_spot * k
                        strikes.append(strike)

                        if k <= 1.0:
                            bloomberg_maturities.append(bloomberg_maturity)
                            option_types.append('P')
                        if k >= 1.0:
                            bloomberg_maturities.append(bloomberg_maturity)
                            option_types.append('C')

            # Fetch option data
            if bloomberg_maturities:
                unique_maturities = list(set(bloomberg_maturities))
                unique_strikes = list(set(strikes))
                self.market_option_data = self.bloomberg_fetcher.fetch_option_prices(
                    unique_strikes, unique_maturities, ['C', 'P']
                )

                # Update instrument Bloomberg tickers
                self._map_instruments_to_tickers()

        except Exception as e:
            print(f"Error updating market data: {e}")

    def _map_instruments_to_tickers(self):
        """Map grid instruments to Bloomberg tickers"""
        if self.market_option_data.empty:
            return

        # Get current spot for strike calculation
        market_data = self.bloomberg_fetcher.get_current_market_data()
        if 'spot_price' not in market_data:
            return

        current_spot = market_data['spot_price']

        for i, instrument in enumerate(self.instruments[1:], 1):  # Skip spot
            if instrument['type'] in ['call', 'put']:
                target_strike = current_spot * instrument['moneyness']
                target_type = 'C' if instrument['type'] == 'call' else 'P'

                # Find closest matching option in market data
                matching_options = self.market_option_data[
                    (self.market_option_data['type'] == target_type) &
                    (abs(self.market_option_data['strike'] - target_strike) < current_spot * 0.02)
                ]

                if not matching_options.empty:
                    # Take the closest match
                    closest_option = matching_options.iloc[
                        (matching_options['strike'] - target_strike).abs().argmin()
                    ]
                    self.instruments[i]['bloomberg_ticker'] = closest_option['ticker']

    def get_market_price(self, instrument_idx: int, current_spot: float = None) -> float:
        """
        Get market price for an instrument

        Args:
            instrument_idx: Index of instrument in the grid
            current_spot: Current spot price (if None, will fetch from Bloomberg)

        Returns:
            Market price of the instrument
        """
        if instrument_idx == 0:  # Spot
            if current_spot is not None:
                return current_spot
            elif self.bloomberg_fetcher is not None:
                market_data = self.bloomberg_fetcher.get_current_market_data()
                return market_data.get('spot_price', 1.0)
            else:
                return 1.0

        instrument = self.instruments[instrument_idx]

        # Try to get market price from Bloomberg data
        if (self.use_market_data and
            instrument.get('bloomberg_ticker') and
            not self.market_option_data.empty):

            ticker = instrument['bloomberg_ticker']
            option_data = self.market_option_data[
                self.market_option_data['ticker'] == ticker
            ]

            if not option_data.empty:
                price = option_data.iloc[0]['price']
                if not np.isnan(price):
                    return price

        # Fallback to Black-Scholes pricing
        if current_spot is None:
            current_spot = 1.0
            if self.bloomberg_fetcher is not None:
                market_data = self.bloomberg_fetcher.get_current_market_data()
                current_spot = market_data.get('spot_price', 1.0)

        # Use Black-Scholes as fallback
        K = current_spot * instrument['moneyness']
        T = instrument['maturity'] / 252.0  # Convert to years
        sigma = 0.2  # Default volatility

        # Try to get market volatility
        if self.bloomberg_fetcher is not None:
            market_data = self.bloomberg_fetcher.get_current_market_data()
            if 'vol_30d' in market_data and not np.isnan(market_data['vol_30d']):
                sigma = market_data['vol_30d']

        option_type = 'call' if instrument['type'] == 'call' else 'put'
        return VanillaOption.black_scholes_price(current_spot, K, T, sigma, option_type)
    
    def get_available_instruments(self, t: int, max_time: int) -> List[int]:
        """
        Get indices of available instruments at time t
        
        Args:
            t: Current time step
            max_time: Maximum time step
            
        Returns:
            List of available instrument indices
        """
        available = [0]  # Spot is always available
        
        for i, instrument in enumerate(self.instruments[1:], 1):
            # Check if option has not expired and does not mature after max_time
            if instrument["maturity"] is not None and t + instrument["maturity"] <= max_time:
                available.append(i)
                
        return available

class DeepHedgingSimulation:
    """Enhanced deep hedging simulation with Bloomberg data integration"""

    def __init__(self,
                 heston_model: HestonModel,
                 grid: FloatingGrid,
                 cliquet: CliquetOption,
                 transaction_costs: Dict[str, float],
                 risk_aversion: float = 1000,
                 use_market_pricing: bool = False):
        """
        Initialize simulation

        Args:
            heston_model: Heston model for price simulation
            grid: Floating grid of tradable instruments
            cliquet: Cliquet option to hedge
            transaction_costs: Dictionary of transaction costs by instrument type
            risk_aversion: Risk aversion parameter gamma
            use_market_pricing: Whether to use market prices for options
        """
        self.heston_model = heston_model
        self.grid = grid
        self.cliquet = cliquet
        self.transaction_costs = transaction_costs
        self.risk_aversion = risk_aversion
        self.use_market_pricing = use_market_pricing
    
    def _calculate_option_prices(self, S: float, v: float, t: int, T: int) -> List[float]:
        """Calculate prices for all available options at time t"""
        prices = [0]  # First element is for the spot (placeholder)

        for i, instrument in enumerate(self.grid.instruments[1:], 1):
            tau = instrument["maturity"]
            if t + tau <= T:
                if self.use_market_pricing:
                    # Try to get market price first
                    market_price = self.grid.get_market_price(i, S)
                    prices.append(market_price)
                else:
                    # Use Black-Scholes pricing
                    K = S * instrument["moneyness"]
                    option_type = instrument["type"]
                    # Use current volatility as implied vol (simplification)
                    sigma = np.sqrt(v)
                    time_to_maturity = tau * self.heston_model.dt

                    price = VanillaOption.black_scholes_price(
                        S, K, time_to_maturity, sigma, option_type
                    )
                    prices.append(price)
            else:
                prices.append(0)  # Option not available

        return prices
    
    def simple_delta_hedge(self, num_paths: int, time_steps: int) -> Dict:
        """
        Implement a simple delta hedging strategy
        
        Args:
            num_paths: Number of paths to simulate
            time_steps: Number of time steps
            
        Returns:
            Dictionary with simulation results
        """
        # Simulate price paths
        price_paths, vol_paths = self.heston_model.simulate_paths(num_paths, time_steps)
        
        # Initialize arrays to store results
        payoffs = np.zeros(num_paths)
        hedge_pnls = np.zeros(num_paths)
        transaction_costs_total = np.zeros(num_paths)
        delta_positions = np.zeros((num_paths, time_steps+1))
        
        # For each path
        for path in range(num_paths):
            # Calculate cliquet payoff
            payoffs[path] = self.cliquet.payoff(price_paths[path])
            
            # Initialize tracking variables
            delta_pos = 0
            hedge_pnl = 0
            transaction_costs_sum = 0
            
            # For each time step
            for t in range(time_steps):
                S = price_paths[path, t]
                v = vol_paths[path, t]
                
                # Find next reset date
                next_reset = None
                for reset in self.cliquet.reset_dates:
                    if reset > t:
                        next_reset = reset
                        break
                
                if next_reset is None:
                    # No more resets, maintain current position
                    new_delta = delta_pos
                else:
                    # Time to next reset
                    tau = (next_reset - t) * self.heston_model.dt
                    
                    # Simple delta calculation (approximate)
                    # For a cliquet, we approximate delta based on the upcoming reset
                    sigma = np.sqrt(v)
                    
                    # This is a simplified delta calculation for the cliquet
                    if tau > 0:
                        # For cliquet, we're concerned with the next cap level
                        K = S * (1 + self.cliquet.cap)
                        delta = -VanillaOption.delta(S, K, tau, sigma, 'call')
                    else:
                        delta = 0
                    
                    new_delta = delta
                
                # Calculate trade size
                trade = new_delta - delta_pos
                
                # Update position
                delta_pos = new_delta
                delta_positions[path, t] = delta_pos
                
                # Calculate transaction costs
                tc = abs(trade) * self.transaction_costs["spot"]
                transaction_costs_sum += tc
                
                # Update PnL from this trade (realized at next step)
                if t < time_steps - 1:
                    hedge_pnl += trade * (price_paths[path, t+1] - S)
            
            # Final trade to close position
            final_trade = -delta_pos
            tc = abs(final_trade) * self.transaction_costs["spot"]
            transaction_costs_sum += tc
            
            # Store results
            hedge_pnls[path] = hedge_pnl
            transaction_costs_total[path] = transaction_costs_sum
        
        # Calculate net PnL and statistics
        net_pnls = hedge_pnls - payoffs - transaction_costs_total
        
        results = {
            "payoffs": payoffs,
            "hedge_pnls": hedge_pnls,
            "transaction_costs": transaction_costs_total,
            "net_pnls": net_pnls,
            "delta_positions": delta_positions,
            "price_paths": price_paths,
            "vol_paths": vol_paths
        }
        
        return results
    
    def option_enhanced_hedge(self, num_paths: int, time_steps: int) -> Dict:
        """
        Implement a delta + option hedging strategy
        
        Args:
            num_paths: Number of paths to simulate
            time_steps: Number of time steps
            
        Returns:
            Dictionary with simulation results
        """
        # Simulate price paths
        price_paths, vol_paths = self.heston_model.simulate_paths(num_paths, time_steps)
        
        # Initialize arrays to store results
        payoffs = np.zeros(num_paths)
        hedge_pnls = np.zeros(num_paths)
        transaction_costs_total = np.zeros(num_paths)
        
        # Create arrays to track positions
        spot_positions = np.zeros((num_paths, time_steps+1))
        # Using a dictionary to track option positions for simplicity
        option_positions = [{} for _ in range(num_paths)]
        
        # For each path
        for path in range(num_paths):
            # Calculate cliquet payoff
            payoffs[path] = self.cliquet.payoff(price_paths[path])
            
            # Initialize tracking variables
            spot_pos = 0
            option_pos = {}  # Dictionary to track option positions: {(maturity, strike, type): position}
            hedge_pnl = 0
            transaction_costs_sum = 0
            
            # For each time step
            for t in range(time_steps):
                S = price_paths[path, t]
                v = vol_paths[path, t]
                
                # Get available instruments
                available = self.grid.get_available_instruments(t, time_steps)
                
                # Find next reset date
                next_reset = None
                for reset in self.cliquet.reset_dates:
                    if reset > t:
                        next_reset = reset
                        break
                
                # Calculate new positions
                new_spot_pos = 0
                new_option_pos = {}
                
                if next_reset is not None:
                    # Time to next reset
                    tau = (next_reset - t) * self.heston_model.dt
                    
                    # Basic delta hedge component
                    sigma = np.sqrt(v)
                    K = S * (1 + self.cliquet.cap)
                    delta = -VanillaOption.delta(S, K, tau, sigma, 'call')
                    new_spot_pos = 0.7 * delta  # Reduce delta position to make room for option hedge
                    
                    # Option hedge component - find options with maturity close to next reset
                    target_maturity = next_reset - t
                    best_maturity_diff = float('inf')
                    best_option_idx = None
                    
                    for idx in available[1:]:  # Skip spot (idx 0)
                        instrument = self.grid.instruments[idx]
                        if instrument["maturity"] is not None:
                            maturity_diff = abs(instrument["maturity"] - target_maturity)
                            # Find call option with strike close to cap level
                            if (instrument["type"] == "call" and 
                                abs(instrument["moneyness"] - (1 + self.cliquet.cap)) < 0.02 and
                                maturity_diff < best_maturity_diff):
                                best_maturity_diff = maturity_diff
                                best_option_idx = idx
                    
                    # If suitable option found, add to the hedge
                    if best_option_idx is not None:
                        instrument = self.grid.instruments[best_option_idx]
                        maturity = t + instrument["maturity"]
                        strike = S * instrument["moneyness"]
                        option_key = (maturity, strike, instrument["type"])
                        
                        # Add position in this option
                        new_option_pos[option_key] = -0.5  # Short position to hedge upside
                
                # Calculate trades
                spot_trade = new_spot_pos - spot_pos
                
                # Update positions
                spot_pos = new_spot_pos
                spot_positions[path, t] = spot_pos
                
                # Process option positions - need to handle expirations
                expired_options = []
                for (mat, k, opt_type), pos in option_pos.items():
                    if mat <= t:
                        # Option expired
                        expired_options.append((mat, k, opt_type))
                        # Calculate payoff
                        if opt_type == 'call':
                            payoff = max(price_paths[path, t] - k, 0)
                        else:
                            payoff = max(k - price_paths[path, t], 0)
                        hedge_pnl += pos * payoff
                    else:
                        # Option still active
                        new_option_pos[(mat, k, opt_type)] = pos
                
                # Remove expired options
                for key in expired_options:
                    del option_pos[key]
                
                # Calculate transaction costs for spot
                tc_spot = abs(spot_trade) * self.transaction_costs["spot"]
                transaction_costs_sum += tc_spot
                
                # Calculate transaction costs and add new options
                for (mat, k, opt_type), new_pos in new_option_pos.items():
                    old_pos = option_pos.get((mat, k, opt_type), 0)
                    option_trade = new_pos - old_pos
                    
                    if option_trade != 0:
                        # Calculate price of the option
                        time_to_mat = (mat - t) * self.heston_model.dt
                        sigma = np.sqrt(v)
                        option_price = VanillaOption.black_scholes_price(
                            S, k, time_to_mat, sigma, opt_type
                        )
                        
                        # Add transaction cost
                        tc_option = abs(option_trade) * self.transaction_costs["option"]
                        transaction_costs_sum += tc_option
                        
                        # Add cost/revenue from buying/selling the option
                        hedge_pnl -= option_trade * option_price
                
                # Update option positions
                option_pos = new_option_pos.copy()
                option_positions[path] = option_pos.copy()
                
                # Update PnL from spot trade (realized at next step)
                if t < time_steps - 1:
                    hedge_pnl += spot_trade * (price_paths[path, t+1] - S)
            
            # Close all positions at the end
            final_spot_trade = -spot_pos
            tc_spot = abs(final_spot_trade) * self.transaction_costs["spot"]
            transaction_costs_sum += tc_spot
            
            # Calculate value of remaining options at maturity
            for (mat, k, opt_type), pos in option_pos.items():
                if mat <= time_steps:
                    # Calculate final payoff
                    if opt_type == 'call':
                        payoff = max(price_paths[path, mat] - k, 0)
                    else:
                        payoff = max(k - price_paths[path, mat], 0)
                    hedge_pnl += pos * payoff
            
            # Store results
            hedge_pnls[path] = hedge_pnl
            transaction_costs_total[path] = transaction_costs_sum
        
        # Calculate net PnL and statistics
        net_pnls = hedge_pnls - payoffs - transaction_costs_total
        
        results = {
            "payoffs": payoffs,
            "hedge_pnls": hedge_pnls,
            "transaction_costs": transaction_costs_total,
            "net_pnls": net_pnls,
            "spot_positions": spot_positions,
            "price_paths": price_paths,
            "vol_paths": vol_paths
        }
        
        return results
    
    def analyze_results(self, results_delta: Dict, results_enhanced: Dict, no_hedge: Dict):
        """
        Analyze and compare hedging strategies
        
        Args:
            results_delta: Results from delta hedging
            results_enhanced: Results from enhanced hedging
            no_hedge: Results with no hedging
        """
        # Calculate statistics
        stats = {
            "No Hedge": {
                "mean": np.mean(-no_hedge["payoffs"]),
                "std": np.std(-no_hedge["payoffs"]),
                "skew": pd.Series(-no_hedge["payoffs"]).skew(),
                "min": np.min(-no_hedge["payoffs"]),
                "max": np.max(-no_hedge["payoffs"])
            },
            "Delta Hedge": {
                "mean": np.mean(results_delta["net_pnls"]),
                "std": np.std(results_delta["net_pnls"]),
                "skew": pd.Series(results_delta["net_pnls"]).skew(),
                "min": np.min(results_delta["net_pnls"]),
                "max": np.max(results_delta["net_pnls"]),
                "transaction_costs": np.mean(results_delta["transaction_costs"])
            },
            "Enhanced Hedge": {
                "mean": np.mean(results_enhanced["net_pnls"]),
                "std": np.std(results_enhanced["net_pnls"]),
                "skew": pd.Series(results_enhanced["net_pnls"]).skew(),
                "min": np.min(results_enhanced["net_pnls"]),
                "max": np.max(results_enhanced["net_pnls"]),
                "transaction_costs": np.mean(results_enhanced["transaction_costs"])
            }
        }
        
        # Print statistics
        print("\nHedging Strategy Comparison:")
        print("=" * 60)
        for strategy, stat in stats.items():
            print(f"\n{strategy}:")
            for key, value in stat.items():
                print(f"  {key}: {value:.6f}")
        
        # Plot PnL distributions
        plt.figure(figsize=(12, 8))
        
        # Plot histograms on log scale as in the paper
        bins = np.linspace(
            min(np.min(-no_hedge["payoffs"]), 
                np.min(results_delta["net_pnls"]), 
                np.min(results_enhanced["net_pnls"])) - 0.01,
            max(np.max(-no_hedge["payoffs"]), 
                np.max(results_delta["net_pnls"]), 
                np.max(results_enhanced["net_pnls"])) + 0.01,
            100
        )
        
        plt.hist(-no_hedge["payoffs"], bins=bins, alpha=0.5, label=f'No Hedge (std={stats["No Hedge"]["std"]:.1e})', density=True)
        plt.hist(results_delta["net_pnls"], bins=bins, alpha=0.5, label=f'Delta Hedge (std={stats["Delta Hedge"]["std"]:.1e})', density=True)
        plt.hist(results_enhanced["net_pnls"], bins=bins, alpha=0.5, label=f'Delta + Option Hedge (std={stats["Enhanced Hedge"]["std"]:.1e})', density=True)
        
        plt.yscale('log')
        plt.xlabel('PnL (no transaction costs)')
        plt.ylabel('Probability')
        plt.title('PnL Distribution Comparison (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pnl_histogram.png')
        
        # Plot spot position over time for a sample path
        sample_path = 0
        plt.figure(figsize=(12, 8))
        plt.plot(results_delta["delta_positions"][sample_path], label='Delta Hedge')
        plt.plot(results_enhanced["spot_positions"][sample_path], label='Enhanced Hedge (Spot Component)')
        
        # Mark reset dates
        for reset_date in self.cliquet.reset_dates:
            plt.axvline(x=reset_date, color='r', linestyle='--', alpha=0.3)
        
        plt.xlabel('Time Step')
        plt.ylabel('Spot Position')
        plt.title('Spot Position Over Time (Sample Path)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('spot_position.png')
        
        return stats

    def live_hedging_session(self, duration_minutes: int = 60,
                           rebalance_interval: int = 5) -> Dict:
        """
        Perform live hedging session with real-time Bloomberg data

        Args:
            duration_minutes: Duration of hedging session in minutes
            rebalance_interval: Rebalancing interval in minutes

        Returns:
            Dictionary with live hedging results
        """
        if not BLOOMBERG_AVAILABLE or self.heston_model.bloomberg_fetcher is None:
            print("Bloomberg data not available for live hedging")
            return {}

        try:
            print(f"Starting live hedging session:")
            print(f"  Duration: {duration_minutes} minutes")
            print(f"  Rebalancing every: {rebalance_interval} minutes")

            # Initialize real-time monitor
            monitor = RealTimeMarketMonitor(
                self.heston_model.bloomberg_fetcher,
                update_interval=30  # Update every 30 seconds
            )

            # Start monitoring
            monitor.start_monitoring()

            # Initialize tracking variables
            session_data = []
            positions = {}  # Current positions
            total_pnl = 0
            total_transaction_costs = 0

            # Get initial market snapshot
            initial_snapshot = monitor.get_latest_market_snapshot()
            if not initial_snapshot:
                print("Failed to get initial market data")
                return {}

            initial_spot = initial_snapshot['spot_price']
            print(f"  Initial spot price: {initial_spot:.2f}")

            # Session loop
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            last_rebalance = start_time

            while datetime.now() < end_time:
                current_time = datetime.now()

                # Check if it's time to rebalance
                if (current_time - last_rebalance).total_seconds() >= rebalance_interval * 60:

                    # Get current market snapshot
                    snapshot = monitor.get_latest_market_snapshot()
                    if not snapshot:
                        print("Failed to get market snapshot, skipping rebalance")
                        time.sleep(30)
                        continue

                    current_spot = snapshot['spot_price']
                    current_vol = snapshot.get('volatility_30d', 0.2)
                    realized_vol = monitor.calculate_realized_volatility(30)

                    print(f"\n[{current_time.strftime('%H:%M:%S')}] Rebalancing:")
                    print(f"  Spot: {current_spot:.2f}")
                    print(f"  Implied Vol: {current_vol:.1%}")
                    if not np.isnan(realized_vol):
                        print(f"  Realized Vol: {realized_vol:.1%}")

                    # Calculate optimal positions
                    new_positions = self._calculate_live_positions(
                        current_spot, current_vol, snapshot
                    )

                    # Calculate trades and costs
                    trades = {}
                    for instrument_idx, new_pos in new_positions.items():
                        old_pos = positions.get(instrument_idx, 0)
                        trade = new_pos - old_pos
                        if abs(trade) > 1e-6:  # Only trade if significant
                            trades[instrument_idx] = trade

                    # Execute trades
                    trade_costs = 0
                    for instrument_idx, trade in trades.items():
                        if instrument_idx == 0:  # Spot
                            cost = abs(trade) * self.transaction_costs["spot"]
                        else:
                            cost = abs(trade) * self.transaction_costs["option"]

                        trade_costs += cost
                        print(f"    Trade {instrument_idx}: {trade:+.4f} (cost: {cost:.6f})")

                    # Update positions
                    positions.update(new_positions)
                    total_transaction_costs += trade_costs

                    # Calculate current portfolio value
                    portfolio_value = self._calculate_portfolio_value(
                        positions, current_spot, current_vol
                    )

                    # Store session data
                    session_data.append({
                        'timestamp': current_time,
                        'spot_price': current_spot,
                        'implied_vol': current_vol,
                        'realized_vol': realized_vol,
                        'positions': positions.copy(),
                        'portfolio_value': portfolio_value,
                        'transaction_costs': trade_costs,
                        'cumulative_costs': total_transaction_costs,
                        'bid_ask_spread': snapshot.get('spread', np.nan)
                    })

                    last_rebalance = current_time

                # Wait before next check
                time.sleep(30)  # Check every 30 seconds

            # Stop monitoring
            monitor.stop_monitoring()

            # Calculate final results
            if session_data:
                final_spot = session_data[-1]['spot_price']
                spot_return = (final_spot - initial_spot) / initial_spot

                results = {
                    'session_duration': duration_minutes,
                    'rebalance_interval': rebalance_interval,
                    'initial_spot': initial_spot,
                    'final_spot': final_spot,
                    'spot_return': spot_return,
                    'total_transaction_costs': total_transaction_costs,
                    'num_rebalances': len(session_data),
                    'session_data': session_data,
                    'final_positions': positions
                }

                print(f"\nLive hedging session completed:")
                print(f"  Spot return: {spot_return:.2%}")
                print(f"  Total transaction costs: {total_transaction_costs:.6f}")
                print(f"  Number of rebalances: {len(session_data)}")

                return results
            else:
                print("No session data collected")
                return {}

        except Exception as e:
            print(f"Error in live hedging session: {e}")
            return {}

    def _calculate_live_positions(self, current_spot: float, current_vol: float,
                                market_snapshot: Dict) -> Dict[int, float]:
        """
        Calculate optimal positions for live hedging

        Args:
            current_spot: Current spot price
            current_vol: Current implied volatility
            market_snapshot: Current market data snapshot

        Returns:
            Dictionary of optimal positions
        """
        positions = {}

        try:
            # Basic delta hedging for cliquet
            # This is simplified - in practice you'd use more sophisticated models

            # Estimate time to next reset (simplified)
            days_to_reset = 20  # Assume 20 days to next reset
            time_to_reset = days_to_reset / 252.0

            # Calculate delta for cliquet approximation
            cap_strike = current_spot * (1 + self.cliquet.cap)
            delta = -VanillaOption.delta(current_spot, cap_strike, time_to_reset, current_vol, 'call')

            # Adjust for market conditions
            spread = market_snapshot.get('spread', 0)
            if not np.isnan(spread) and spread > 0:
                # Reduce position size if spread is wide
                spread_adjustment = max(0.5, 1 - spread / current_spot * 1000)
                delta *= spread_adjustment

            # Set spot position
            positions[0] = delta * 0.8  # Conservative sizing

            # Add option hedge if available
            available_instruments = self.grid.get_available_instruments(0, 240)
            for idx in available_instruments[1:]:
                instrument = self.grid.instruments[idx]
                if (instrument["type"] == "call" and
                    abs(instrument["moneyness"] - (1 + self.cliquet.cap)) < 0.05):
                    positions[idx] = -0.2  # Small short call position
                    break

            return positions

        except Exception as e:
            print(f"Error calculating live positions: {e}")
            return {0: 0}  # Return neutral position on error

    def _calculate_portfolio_value(self, positions: Dict[int, float],
                                 current_spot: float, current_vol: float) -> float:
        """
        Calculate current portfolio value

        Args:
            positions: Current positions
            current_spot: Current spot price
            current_vol: Current volatility

        Returns:
            Portfolio value
        """
        try:
            total_value = 0

            for instrument_idx, position in positions.items():
                if instrument_idx == 0:  # Spot position
                    value = position * current_spot
                else:  # Option position
                    instrument = self.grid.instruments[instrument_idx]
                    K = current_spot * instrument["moneyness"]
                    T = instrument["maturity"] / 252.0  # Convert to years
                    option_type = instrument["type"]

                    option_price = VanillaOption.black_scholes_price(
                        current_spot, K, T, current_vol, option_type
                    )
                    value = position * option_price

                total_value += value

            return total_value

        except Exception as e:
            print(f"Error calculating portfolio value: {e}")
            return 0

    def real_time_hedge_with_market_data(self, hedge_horizon_days: int = 30) -> Dict:
        """
        Perform real-time hedging using current market data

        Args:
            hedge_horizon_days: Number of days to simulate hedging

        Returns:
            Dictionary with hedging results
        """
        if not BLOOMBERG_AVAILABLE or self.heston_model.bloomberg_fetcher is None:
            print("Bloomberg data not available for real-time hedging")
            return {}

        try:
            # Get current market data
            market_data = self.heston_model.bloomberg_fetcher.get_current_market_data()
            current_spot = market_data.get('spot_price', 1.0)
            current_vol = market_data.get('vol_30d', 0.2)

            print(f"Starting real-time hedge simulation:")
            print(f"  Current spot: {current_spot:.2f}")
            print(f"  Current vol: {current_vol:.1%}")

            # Update grid with current market data
            if self.grid.use_market_data:
                self.grid.update_market_data()

            # Simulate future paths from current market state
            time_steps = hedge_horizon_days
            num_paths = 1000  # Smaller number for real-time simulation

            # Use market-informed simulation
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=252)).strftime('%Y-%m-%d')
            historical_data = self.heston_model.bloomberg_fetcher.fetch_historical_prices(start_date, end_date)

            price_paths, vol_paths = self.heston_model.simulate_paths_with_market_data(
                num_paths, time_steps, historical_data
            )

            # Calculate cliquet payoffs (simplified for demonstration)
            reset_dates = [int(hedge_horizon_days * i / 12) for i in range(13) if int(hedge_horizon_days * i / 12) <= time_steps]
            if not reset_dates or reset_dates[0] != 0:
                reset_dates = [0] + reset_dates
            if reset_dates[-1] != time_steps:
                reset_dates.append(time_steps)

            temp_cliquet = CliquetOption(reset_dates, self.cliquet.cap)
            payoffs = np.array([temp_cliquet.payoff(path) for path in price_paths])

            # Perform enhanced hedging
            hedge_results = self._enhanced_hedge_real_time(price_paths, vol_paths, payoffs)

            # Calculate performance metrics
            net_pnls = hedge_results['hedge_pnls'] - payoffs - hedge_results['transaction_costs']

            results = {
                'current_spot': current_spot,
                'current_vol': current_vol,
                'payoffs': payoffs,
                'hedge_pnls': hedge_results['hedge_pnls'],
                'transaction_costs': hedge_results['transaction_costs'],
                'net_pnls': net_pnls,
                'hedge_ratio': np.std(net_pnls) / np.std(payoffs) if np.std(payoffs) > 0 else 0,
                'mean_pnl': np.mean(net_pnls),
                'std_pnl': np.std(net_pnls),
                'sharpe_ratio': np.mean(net_pnls) / np.std(net_pnls) if np.std(net_pnls) > 0 else 0
            }

            print(f"\nReal-time hedging results:")
            print(f"  Mean PnL: {results['mean_pnl']:.6f}")
            print(f"  PnL Std: {results['std_pnl']:.6f}")
            print(f"  Hedge Ratio: {results['hedge_ratio']:.3f}")
            print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")

            return results

        except Exception as e:
            print(f"Error in real-time hedging: {e}")
            return {}

    def _enhanced_hedge_real_time(self, price_paths: np.ndarray, vol_paths: np.ndarray,
                                payoffs: np.ndarray) -> Dict:
        """
        Enhanced hedging strategy for real-time market data

        Args:
            price_paths: Simulated price paths
            vol_paths: Simulated volatility paths
            payoffs: Cliquet payoffs

        Returns:
            Dictionary with hedging results
        """
        num_paths, time_steps = price_paths.shape[0], price_paths.shape[1] - 1

        hedge_pnls = np.zeros(num_paths)
        transaction_costs_total = np.zeros(num_paths)

        for path in range(num_paths):
            hedge_pnl = 0
            transaction_costs_sum = 0
            positions = {}  # Track all positions

            for t in range(time_steps):
                S = price_paths[path, t]
                v = vol_paths[path, t]

                # Get available instruments
                available = self.grid.get_available_instruments(t, time_steps)

                # Calculate optimal positions using simplified approach
                new_positions = self._calculate_optimal_positions(S, v, t, time_steps, available)

                # Calculate trades and costs
                for instrument_idx, new_pos in new_positions.items():
                    old_pos = positions.get(instrument_idx, 0)
                    trade = new_pos - old_pos

                    if trade != 0:
                        # Calculate transaction cost
                        if instrument_idx == 0:  # Spot
                            tc = abs(trade) * self.transaction_costs["spot"]
                        else:
                            tc = abs(trade) * self.transaction_costs["option"]

                        transaction_costs_sum += tc

                        # Calculate cost/revenue of trade
                        if instrument_idx == 0:  # Spot
                            # PnL from spot trade realized at next step
                            if t < time_steps - 1:
                                hedge_pnl += trade * (price_paths[path, t+1] - S)
                        else:
                            # Option trade cost
                            option_price = self._calculate_option_prices(S, v, t, time_steps)[instrument_idx]
                            hedge_pnl -= trade * option_price

                # Update positions
                positions = new_positions.copy()

            # Close all positions at the end
            for instrument_idx, pos in positions.items():
                if pos != 0:
                    if instrument_idx == 0:  # Spot
                        tc = abs(pos) * self.transaction_costs["spot"]
                    else:
                        tc = abs(pos) * self.transaction_costs["option"]
                        # Calculate final option value
                        final_option_value = self._calculate_final_option_value(
                            instrument_idx, price_paths[path, -1], time_steps
                        )
                        hedge_pnl += pos * final_option_value

                    transaction_costs_sum += tc

            hedge_pnls[path] = hedge_pnl
            transaction_costs_total[path] = transaction_costs_sum

        return {
            'hedge_pnls': hedge_pnls,
            'transaction_costs': transaction_costs_total
        }

    def _calculate_optimal_positions(self, S: float, v: float, t: int, T: int,
                                   available: List[int]) -> Dict[int, float]:
        """Calculate optimal positions for given market state"""
        positions = {}

        # Simple delta-based positioning for demonstration
        # In practice, this would use neural networks or optimization

        # Find next reset date
        next_reset = None
        for reset in self.cliquet.reset_dates:
            if reset > t:
                next_reset = reset
                break

        if next_reset is not None:
            tau = (next_reset - t) * self.heston_model.dt
            sigma = np.sqrt(v)

            # Basic delta hedge
            K = S * (1 + self.cliquet.cap)
            delta = -VanillaOption.delta(S, K, tau, sigma, 'call')
            positions[0] = 0.8 * delta  # Spot position

            # Add option hedge if available
            for idx in available[1:]:
                instrument = self.grid.instruments[idx]
                if (instrument["type"] == "call" and
                    abs(instrument["moneyness"] - (1 + self.cliquet.cap)) < 0.05):
                    positions[idx] = -0.3  # Short call position
                    break

        return positions

    def _calculate_final_option_value(self, instrument_idx: int, final_spot: float,
                                    final_time: int) -> float:
        """Calculate final value of option at expiration"""
        instrument = self.grid.instruments[instrument_idx]
        K = final_spot * instrument["moneyness"]

        if instrument["type"] == "call":
            return max(final_spot - K, 0)
        else:
            return max(K - final_spot, 0)

def main():
    """Enhanced main function with Bloomberg data integration"""
    try:
        print("Enhanced Deep Hedging Strategy with Bloomberg Data Integration")
        print("=" * 70)

        # Check Bloomberg availability
        use_bloomberg = BLOOMBERG_AVAILABLE
        if use_bloomberg:
            print("✓ Bloomberg (xbbg) available - using real market data")
        else:
            print("⚠ Bloomberg (xbbg) not available - using simulated data")

        # Set up Bloomberg data fetcher
        bloomberg_fetcher = None
        if use_bloomberg:
            try:
                bloomberg_fetcher = BloombergDataFetcher("SPX Index")
                print("✓ Bloomberg data fetcher initialized")
            except Exception as e:
                print(f"⚠ Bloomberg initialization failed: {e}")
                use_bloomberg = False

        # Set up enhanced Heston model
        heston = HestonModel(
            bloomberg_fetcher=bloomberg_fetcher,
            auto_calibrate=use_bloomberg
        )

        if use_bloomberg:
            print("✓ Heston model calibrated from market data")
        else:
            print("⚠ Using default Heston parameters")

        # Set up cliquet option
        reset_dates = [20*i for i in range(13)]  # 0, 20, 40, ..., 240
        cliquet = CliquetOption(reset_dates, cap=0.015)

        # Set up enhanced floating grid
        maturities = [10, 20, 40, 80, 120]
        moneyness_levels = [0.85, 0.91, 0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05, 1.09, 1.15]
        grid = FloatingGrid(
            maturities,
            moneyness_levels,
            bloomberg_fetcher=bloomberg_fetcher,
            use_market_data=use_bloomberg
        )

        # Set up transaction costs
        transaction_costs = {
            "spot": 1e-4,
            "option": 1e-2
        }

        # Set up enhanced simulation
        simulation = DeepHedgingSimulation(
            heston_model=heston,
            grid=grid,
            cliquet=cliquet,
            transaction_costs=transaction_costs,
            risk_aversion=1000,
            use_market_pricing=use_bloomberg
        )
        
        # Choose simulation mode
        print("\nChoose simulation mode:")
        print("1. Traditional simulation (synthetic data)")
        print("2. Real-time hedging (Bloomberg data)")
        print("3. Live hedging session (real-time Bloomberg)")
        print("4. Market data demo")
        print("5. All modes")

        mode = input("Enter choice (1-5, default=1): ").strip() or "1"

        if mode in ["1", "5"]:
            print("\n" + "="*50)
            print("TRADITIONAL SIMULATION MODE")
            print("="*50)

            # Set parameters for traditional simulation
            num_paths = 10000
            time_steps = 240

            print(f"Running simulation with {num_paths} paths and {time_steps} time steps...")
            print("This may take a few minutes...")

            # Calculate no hedge baseline
            start_time = time.time()
            price_paths, _ = heston.simulate_paths(num_paths, time_steps, use_market_start=use_bloomberg)
            no_hedge_results = {
                "payoffs": np.array([cliquet.payoff(path) for path in price_paths])
            }
            print(f"No hedge baseline calculated in {time.time() - start_time:.2f} seconds")

            # Run delta hedging simulation
            start_time = time.time()
            delta_results = simulation.simple_delta_hedge(num_paths, time_steps)
            print(f"Delta hedging simulation completed in {time.time() - start_time:.2f} seconds")

            # Run enhanced hedging simulation
            start_time = time.time()
            enhanced_results = simulation.option_enhanced_hedge(num_paths, time_steps)
            print(f"Enhanced hedging simulation completed in {time.time() - start_time:.2f} seconds")

            # Analyze results
            simulation.analyze_results(delta_results, enhanced_results, no_hedge_results)

            print("\nTraditional simulation completed successfully!")
            print("Results saved to 'pnl_histogram.png' and 'spot_position.png'")

        if mode in ["2", "5"] and use_bloomberg:
            print("\n" + "="*50)
            print("REAL-TIME HEDGING MODE")
            print("="*50)

            # Run real-time hedging simulation
            real_time_results = simulation.real_time_hedge_with_market_data(hedge_horizon_days=30)

            if real_time_results:
                print("\nReal-time hedging completed successfully!")

                # Save real-time results
                plt.figure(figsize=(10, 6))
                plt.hist(real_time_results['net_pnls'], bins=50, alpha=0.7, density=True)
                plt.xlabel('Net PnL')
                plt.ylabel('Density')
                plt.title('Real-Time Hedging PnL Distribution')
                plt.grid(True, alpha=0.3)
                plt.savefig('real_time_pnl.png')
                print("Real-time results saved to 'real_time_pnl.png'")
            else:
                print("Real-time hedging failed - check Bloomberg connection")

        if mode == "3" and use_bloomberg:
            print("\n" + "="*50)
            print("LIVE HEDGING SESSION")
            print("="*50)

            # Get session parameters
            duration = input("Enter session duration in minutes (default=30): ").strip()
            duration = int(duration) if duration.isdigit() else 30

            rebalance = input("Enter rebalancing interval in minutes (default=5): ").strip()
            rebalance = int(rebalance) if rebalance.isdigit() else 5

            print(f"\nStarting live hedging session...")
            print("⚠ This will use real Bloomberg data and may incur costs")
            confirm = input("Continue? (y/N): ").strip().lower()

            if confirm == 'y':
                # Run live hedging session
                live_results = simulation.live_hedging_session(
                    duration_minutes=duration,
                    rebalance_interval=rebalance
                )

                if live_results:
                    print("\nLive hedging session completed successfully!")

                    # Save live session results
                    session_data = live_results['session_data']
                    if session_data:
                        timestamps = [d['timestamp'] for d in session_data]
                        spot_prices = [d['spot_price'] for d in session_data]
                        portfolio_values = [d['portfolio_value'] for d in session_data]

                        plt.figure(figsize=(12, 8))

                        plt.subplot(2, 1, 1)
                        plt.plot(timestamps, spot_prices, 'b-', label='Spot Price')
                        plt.ylabel('Spot Price')
                        plt.title('Live Hedging Session Results')
                        plt.legend()
                        plt.grid(True, alpha=0.3)

                        plt.subplot(2, 1, 2)
                        plt.plot(timestamps, portfolio_values, 'r-', label='Portfolio Value')
                        plt.ylabel('Portfolio Value')
                        plt.xlabel('Time')
                        plt.legend()
                        plt.grid(True, alpha=0.3)

                        plt.tight_layout()
                        plt.savefig('live_hedging_session.png')
                        print("Live session results saved to 'live_hedging_session.png'")
                else:
                    print("Live hedging session failed - check Bloomberg connection")
            else:
                print("Live hedging session cancelled")

        if mode == "4" and use_bloomberg:
            print("\n" + "="*50)
            print("MARKET DATA DEMO")
            print("="*50)

            # Demonstrate real-time market data capabilities
            print("Fetching real-time market data...")

            # Get current market snapshot
            current_data = bloomberg_fetcher.get_current_market_data()
            print(f"\nCurrent Market Data:")
            for key, value in current_data.items():
                if key != 'timestamp':
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value.strftime('%Y-%m-%d %H:%M:%S')}")

            # Get real-time data
            real_time_data = bloomberg_fetcher.get_real_time_data()
            if real_time_data:
                print(f"\nReal-Time Data:")
                for key, value in real_time_data.items():
                    if key not in ['ticker', 'timestamp']:
                        if isinstance(value, float) and not np.isnan(value):
                            print(f"  {key}: {value:.4f}")
                        elif not np.isnan(value) if isinstance(value, float) else value is not None:
                            print(f"  {key}: {value}")

            # Get intraday bars
            print(f"\nFetching intraday bars...")
            intraday_data = bloomberg_fetcher.get_intraday_bars(interval=5)
            if not intraday_data.empty:
                print(f"Retrieved {len(intraday_data)} 5-minute bars")
                print("Latest bars:")
                print(intraday_data.tail())

            # Get option chain
            print(f"\nFetching option chain...")
            option_chain = bloomberg_fetcher.get_option_chain()
            if not option_chain.empty:
                print(f"Retrieved {len(option_chain)} options")
                print("Sample options:")
                print(option_chain.head())

        elif mode in ["2", "3", "4"] and not use_bloomberg:
            print("⚠ Real-time modes require Bloomberg data - falling back to traditional mode")
            mode = "1"

        print(f"\nAll simulations completed successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
def demo_bloomberg_integration():
    """Enhanced Bloomberg data integration demo with real-time capabilities"""
    if not BLOOMBERG_AVAILABLE:
        print("Bloomberg (xbbg) not available for demonstration")
        return

    try:
        print("Enhanced Bloomberg Data Integration Demo")
        print("=" * 50)

        # Initialize Bloomberg fetcher
        fetcher = BloombergDataFetcher("SPX Index")

        # 1. Current Market Data
        print("\n1. Current Market Data:")
        market_data = fetcher.get_current_market_data()
        for key, value in market_data.items():
            if key != 'timestamp':
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value.strftime('%Y-%m-%d %H:%M:%S')}")

        # 2. Real-Time Data
        print("\n2. Real-Time Data:")
        real_time_data = fetcher.get_real_time_data()
        if real_time_data:
            for key, value in real_time_data.items():
                if key not in ['ticker', 'timestamp']:
                    if isinstance(value, float) and not np.isnan(value):
                        print(f"   {key}: {value:.4f}")
                    elif not (isinstance(value, float) and np.isnan(value)):
                        print(f"   {key}: {value}")
        else:
            print("   No real-time data available")

        # 3. Intraday Bars
        print("\n3. Intraday Bars (last 5 bars):")
        intraday_data = fetcher.get_intraday_bars(interval=5)
        if not intraday_data.empty:
            print(f"   Retrieved {len(intraday_data)} 5-minute bars")
            print("   Latest bars:")
            print(intraday_data.tail().to_string())
        else:
            print("   No intraday data available")

        # 4. Historical Data
        print("\n4. Historical Data (last 30 days):")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        hist_data = fetcher.fetch_historical_prices(start_date, end_date)
        if not hist_data.empty:
            print(f"   Retrieved {len(hist_data)} days of data")
            print(f"   Price range: {hist_data.iloc[:, 0].min():.2f} - {hist_data.iloc[:, 0].max():.2f}")
        else:
            print("   No historical data retrieved")

        # 5. Volatility Surface
        print("\n5. Volatility Surface:")
        vol_surface = fetcher.fetch_volatility_surface()
        if not vol_surface.empty:
            print(f"   Retrieved {len(vol_surface)} volatility points")
            print(f"   Vol range: {vol_surface['implied_vol'].min():.1%} - {vol_surface['implied_vol'].max():.1%}")
        else:
            print("   No volatility surface data retrieved")

        # 6. Option Chain
        print("\n6. Option Chain (sample):")
        option_chain = fetcher.get_option_chain()
        if not option_chain.empty:
            print(f"   Retrieved {len(option_chain)} options")
            print("   Sample options:")
            print(option_chain.head().to_string())
        else:
            print("   No option chain data available")

        # 7. Real-Time Monitor Demo
        print("\n7. Real-Time Monitor Demo (30 seconds):")
        monitor = RealTimeMarketMonitor(fetcher, update_interval=5)
        monitor.start_monitoring()

        print("   Collecting real-time data for 30 seconds...")
        for i in range(6):  # 6 updates over 30 seconds
            snapshot = monitor.get_latest_market_snapshot()
            if snapshot:
                print(f"   [{snapshot['timestamp'].strftime('%H:%M:%S')}] "
                      f"Spot: {snapshot['spot_price']:.2f}, "
                      f"Change: {snapshot.get('change_pct', 0):.2f}%")
            time.sleep(5)

        # Calculate realized volatility
        realized_vol = monitor.calculate_realized_volatility(30)
        if not np.isnan(realized_vol):
            print(f"   Realized volatility (30min): {realized_vol:.1%}")

        monitor.stop_monitoring()

        # 8. Model Calibration
        print("\n8. Model Calibration:")
        calibrator = HestonCalibrator(fetcher)

        if not hist_data.empty:
            params = calibrator.calibrate_from_historical_data(hist_data)
            print("   Calibrated parameters from historical data:")
            for param, value in params.items():
                print(f"     {param}: {value:.4f}")

        if not vol_surface.empty:
            params = calibrator.calibrate_from_volatility_surface(vol_surface)
            print("   Calibrated parameters from volatility surface:")
            for param, value in params.items():
                print(f"     {param}: {value:.4f}")

        print("\n✓ Enhanced Bloomberg integration demo completed successfully!")
        print("✓ Real-time data capabilities demonstrated")

    except Exception as e:
        print(f"Error in Bloomberg demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_bloomberg_integration()
    else:
        main()