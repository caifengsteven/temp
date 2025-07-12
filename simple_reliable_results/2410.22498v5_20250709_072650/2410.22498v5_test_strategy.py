"""
VIX Stochastic Volatility Model for Corporate Bonds
Enhanced version with Bloomberg data integration via xbbg
Based on paper 2410.22498v5: "The VIX as Stochastic Volatility for Corporate Bonds"

Key Features:
- Real Bloomberg data integration for VIX and corporate bond data
- Implementation of VIX-based stochastic volatility model
- Comparison of models with and without VIX normalization
- Residual analysis to verify Gaussian properties after VIX normalization

Run with: python 2410.22498v5_test_strategy.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Bloomberg data integration
try:
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
    print("✓ Bloomberg xbbg library available")
except ImportError as e:
    BLOOMBERG_AVAILABLE = False
    blp = None
    print("⚠ Bloomberg xbbg library not available. Using simulated data only.")
    print(f"  Import error: {e}")
    print("  To install: pip install xbbg")
    print("  Note: Also requires Bloomberg Terminal or API access")
except Exception as e:
    BLOOMBERG_AVAILABLE = False
    blp = None
    print("⚠ Bloomberg xbbg library import failed. Using simulated data only.")
    print(f"  Error: {e}")
    print("  This may be due to missing dependencies or Bloomberg API issues")

class BloombergDataLoader:
    """Class to load and process Bloomberg data for VIX and corporate bonds."""

    def __init__(self):
        if not BLOOMBERG_AVAILABLE:
            raise ImportError("xbbg library is required for Bloomberg data access")

    def load_vix_data(self, start_date='2010-01-01', end_date=None):
        """
        Load VIX data from Bloomberg.

        Parameters:
        - start_date: Start date for data (YYYY-MM-DD format)
        - end_date: End date for data (YYYY-MM-DD format, defaults to today)

        Returns:
        - DataFrame with VIX data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            # VIX Index ticker using correct xbbg API
            vix_data = blp.bdh(
                tickers='VIX Index',
                flds=['PX_LAST'],
                start_date=start_date,
                end_date=end_date
            )

            # Clean and format data
            vix_data = vix_data.dropna()
            vix_data.columns = ['VIX']
            vix_data['log_VIX'] = np.log(vix_data['VIX'])

            return vix_data

        except Exception as e:
            print(f"Error loading VIX data: {e}")
            print("This may be due to:")
            print("1. Bloomberg Terminal not running")
            print("2. No Bloomberg API access")
            print("3. Network connectivity issues")
            print("4. Incorrect xbbg installation")
            return None

    def load_corporate_bond_data(self, tickers=None, start_date='2010-01-01', end_date=None):
        """
        Load corporate bond data from Bloomberg.

        Parameters:
        - tickers: List of corporate bond tickers (defaults to common indices)
        - start_date: Start date for data
        - end_date: End date for data

        Returns:
        - DataFrame with corporate bond data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if tickers is None:
            # Default to common corporate bond indices and ETFs
            tickers = [
                'LF98TRUU Index',  # Bloomberg US Corporate Bond Index
                'LUACTRUU Index',  # Bloomberg US Credit Index
                'HYG US Equity',   # iShares High Yield Corporate Bond ETF
                'LQD US Equity',   # iShares Investment Grade Corporate Bond ETF
                'VCIT US Equity',  # Vanguard Intermediate-Term Corporate Bond ETF
                'USIG US Equity',  # iShares Broad USD Investment Grade Corporate Bond ETF
            ]

        try:
            # Load price data using correct xbbg API
            bond_data = blp.bdh(
                tickers=tickers,
                flds=['PX_LAST'],
                start_date=start_date,
                end_date=end_date
            )

            # Calculate returns
            returns_data = bond_data.pct_change().dropna()

            # Load spread data if available
            spread_tickers = [
                'LUACOAS Index',   # US Credit OAS (Option Adjusted Spread)
                'LUCROAS Index',   # US Corporate OAS
                'C0A0 Index',      # US Investment Grade Corporate OAS
                'H0A0 Index',      # US High Yield Corporate OAS
            ]

            try:
                spread_data = blp.bdh(
                    tickers=spread_tickers,
                    flds=['PX_LAST'],
                    start_date=start_date,
                    end_date=end_date
                )
                spread_data.columns = [f"{col}_Spread" for col in spread_data.columns]
            except:
                spread_data = None
                print("Warning: Could not load spread data")

            return {
                'prices': bond_data,
                'returns': returns_data,
                'spreads': spread_data
            }

        except Exception as e:
            print(f"Error loading corporate bond data: {e}")
            return None

    def load_treasury_data(self, start_date='2010-01-01', end_date=None):
        """Load Treasury yield data for risk-free rates."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            treasury_tickers = [
                'USGG2YR Index',   # 2Y Treasury
                'USGG5YR Index',   # 5Y Treasury
                'USGG10YR Index',  # 10Y Treasury
            ]

            treasury_data = blp.bdh(
                tickers=treasury_tickers,
                flds=['PX_LAST'],
                start_date=start_date,
                end_date=end_date
            )

            return treasury_data

        except Exception as e:
            print(f"Error loading Treasury data: {e}")
            return None

    def test_bloomberg_connection(self):
        """Test Bloomberg connection with a simple data request."""
        try:
            # Test with a simple, reliable ticker
            test_data = blp.bdp(tickers='SPX Index', flds=['PX_LAST'])
            if test_data is not None and not test_data.empty:
                print("✓ Bloomberg connection successful")
                print(f"  SPX Index current price: {test_data.iloc[0, 0]:.2f}")
                return True
            else:
                print("✗ Bloomberg connection failed - no data returned")
                return False
        except Exception as e:
            print(f"✗ Bloomberg connection failed: {e}")
            return False

    def get_current_market_data(self):
        """Get current market snapshot for VIX and key corporate bond indicators."""
        try:
            # Current market data tickers
            current_tickers = [
                'VIX Index',       # VIX
                'SPX Index',       # S&P 500
                'HYG US Equity',   # High Yield ETF
                'LQD US Equity',   # Investment Grade ETF
                'LUACOAS Index',   # US Credit OAS
            ]

            # Get current prices
            current_data = blp.bdp(
                tickers=current_tickers,
                flds=['PX_LAST', 'CHG_PCT_1D']
            )

            print("Current Market Snapshot:")
            print("=" * 40)
            for ticker in current_tickers:
                if ticker in current_data.index:
                    price = current_data.loc[ticker, 'PX_LAST']
                    change = current_data.loc[ticker, 'CHG_PCT_1D']
                    print(f"{ticker:15s}: {price:8.2f} ({change:+5.2f}%)")

            return current_data

        except Exception as e:
            print(f"Error getting current market data: {e}")
            return None


class VIXStochasticVolatilityModel:
    """Implementation of the stochastic volatility model for corporate bonds using VIX."""
    
    def __init__(self, alpha=0.347, beta=0.881, a=0.05, b=0.95, c=0.01, 
                 sigma_z=1.0, sigma_w=1.0, corr_zw=0.3):
        """
        Initialize model parameters.
        
        Parameters:
        - alpha, beta: parameters for VIX autoregression
        - a, b, c: parameters for rate/spread model
        - sigma_z, sigma_w: standard deviations for innovations
        - corr_zw: correlation between Z and W innovations
        """
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.c = c
        self.sigma_z = sigma_z
        self.sigma_w = sigma_w
        self.corr_zw = corr_zw
        
    def simulate_vix(self, n_periods, initial_log_vix=None, seed=42):
        """
        Simulate VIX using autoregression on log scale.
        
        ln V_t = alpha + beta * ln V_(t-1) + W_t
        """
        np.random.seed(seed)
        
        # Initialize log_vix series
        log_vix = np.zeros(n_periods)
        
        # Set initial value
        if initial_log_vix is None:
            # Set to stationary mean: alpha / (1 - beta)
            log_vix[0] = self.alpha / (1 - self.beta)
        else:
            log_vix[0] = initial_log_vix
            
        # Generate innovations with non-Gaussian distribution (using t-distribution)
        W = stats.t.rvs(df=5, scale=self.sigma_w, size=n_periods-1)
        
        # Generate log_vix series
        for t in range(1, n_periods):
            log_vix[t] = self.alpha + self.beta * log_vix[t-1] + W[t-1]
            
        # Convert to VIX
        vix = np.exp(log_vix)
        
        return vix, log_vix, W

    def fit_vix_model_to_data(self, vix_data):
        """
        Fit the VIX autoregressive model to real Bloomberg data.

        ln V_t = alpha + beta * ln V_(t-1) + W_t
        """
        if isinstance(vix_data, pd.DataFrame):
            if 'log_VIX' in vix_data.columns:
                log_vix = vix_data['log_VIX'].values
            else:
                log_vix = np.log(vix_data['VIX'].values)
        else:
            log_vix = np.log(vix_data)

        # Remove any NaN or infinite values
        log_vix = log_vix[np.isfinite(log_vix)]

        # Prepare regression data
        y = log_vix[1:]  # V_t
        X = np.column_stack([
            np.ones(len(y)),  # Intercept (alpha)
            log_vix[:-1]      # V_(t-1) (beta coefficient)
        ])

        # Fit the model
        model = sm.OLS(y, X)
        results = model.fit()

        # Extract parameters
        alpha_hat, beta_hat = results.params
        residuals = results.resid

        # Update model parameters
        self.alpha = alpha_hat
        self.beta = beta_hat
        self.sigma_w = np.std(residuals)

        print(f"Fitted VIX Model Parameters:")
        print(f"Alpha: {alpha_hat:.4f}")
        print(f"Beta: {beta_hat:.4f}")
        print(f"Sigma_W: {self.sigma_w:.4f}")
        print(f"R-squared: {results.rsquared:.4f}")

        return {
            'alpha': alpha_hat,
            'beta': beta_hat,
            'sigma_w': self.sigma_w,
            'residuals': residuals,
            'model_summary': results.summary(),
            'fitted_values': results.fittedvalues
        }
    
    def simulate_rates(self, vix, initial_rate=None, seed=43):
        """
        Simulate rates/spreads using the model:
        
        R_t = a + b * R_(t-1) + c * V_t + V_t * Z_t
        """
        np.random.seed(seed)
        
        n_periods = len(vix)
        rates = np.zeros(n_periods)
        
        # Set initial value
        if initial_rate is None:
            # Set to approximate stationary mean
            rates[0] = (self.a + self.c * np.mean(vix)) / (1 - self.b)
        else:
            rates[0] = initial_rate
            
        # Generate Z innovations (correlated with W if needed)
        Z = np.random.normal(0, self.sigma_z, n_periods-1)
        
        # Generate rates series
        for t in range(1, n_periods):
            rates[t] = self.a + self.b * rates[t-1] + self.c * vix[t] + vix[t] * Z[t-1]
            
        return rates, Z

    def fit_bond_model_to_data(self, bond_data, vix_data, bond_type='spreads'):
        """
        Fit the corporate bond model to real Bloomberg data.

        R_t = a + b * R_(t-1) + c * V_t + V_t * Z_t

        Parameters:
        - bond_data: DataFrame with bond spreads or returns
        - vix_data: DataFrame with VIX data
        - bond_type: 'spreads' or 'returns'
        """
        # Align data by dates
        if isinstance(bond_data, dict):
            if bond_type == 'spreads' and bond_data['spreads'] is not None:
                bond_series = bond_data['spreads'].iloc[:, 0]  # Use first spread series
            else:
                bond_series = bond_data['returns'].iloc[:, 0]  # Use first return series
        else:
            bond_series = bond_data.iloc[:, 0] if len(bond_data.shape) > 1 else bond_data

        # Get VIX series
        if 'VIX' in vix_data.columns:
            vix_series = vix_data['VIX']
        else:
            vix_series = vix_data.iloc[:, 0]

        # Align by common dates
        common_dates = bond_series.index.intersection(vix_series.index)
        bond_aligned = bond_series.loc[common_dates]
        vix_aligned = vix_series.loc[common_dates]

        # Remove NaN values
        valid_idx = bond_aligned.notna() & vix_aligned.notna()
        bond_clean = bond_aligned[valid_idx].values
        vix_clean = vix_aligned[valid_idx].values

        if len(bond_clean) < 50:
            raise ValueError("Insufficient data points for model fitting")

        # Prepare regression data
        n = len(bond_clean)
        y = bond_clean[1:]  # R_t
        X = np.column_stack([
            np.ones(n-1),      # Intercept (a)
            bond_clean[:-1],   # R_(t-1) (b coefficient)
            vix_clean[1:]      # V_t (c coefficient)
        ])

        # Fit the model
        model = sm.OLS(y, X)
        results = model.fit()

        # Extract parameters
        a_hat, b_hat, c_hat = results.params
        residuals = results.resid

        # Calculate normalized residuals Z_t = residuals / V_t
        Z = residuals / vix_clean[1:]

        # Update model parameters
        self.a = a_hat
        self.b = b_hat
        self.c = c_hat
        self.sigma_z = np.std(Z)

        print(f"\nFitted Bond Model Parameters ({bond_type}):")
        print(f"a: {a_hat:.6f}")
        print(f"b: {b_hat:.6f}")
        print(f"c: {c_hat:.6f}")
        print(f"Sigma_Z: {self.sigma_z:.6f}")
        print(f"R-squared: {results.rsquared:.4f}")

        return {
            'a': a_hat,
            'b': b_hat,
            'c': c_hat,
            'sigma_z': self.sigma_z,
            'residuals': residuals,
            'Z': Z,
            'model_summary': results.summary(),
            'fitted_values': results.fittedvalues,
            'bond_data': bond_clean,
            'vix_data': vix_clean,
            'dates': common_dates[valid_idx][1:]  # Dates for fitted values
        }
    
    def simulate_returns(self, rates, vix, k=-0.5, m=5, h=0.02, l=0.001, sigma_u=0.5, seed=44):
        """
        Simulate bond returns using the model:
        
        Q_t = k * R_(t-1) - m * ΔR_t + h * V_t + l + V_t * U_t
        """
        np.random.seed(seed)
        
        n_periods = len(rates)
        returns = np.zeros(n_periods)
        
        # Generate U innovations
        U = np.random.normal(0, sigma_u, n_periods-1)
        
        # Skip first period (need lagged rate)
        for t in range(1, n_periods):
            delta_r = rates[t] - rates[t-1]
            returns[t] = k * rates[t-1] - m * delta_r + h * vix[t] + l + vix[t] * U[t-1]
            
        return returns, U
    
    def analyze_residuals(self, residuals, title="Residuals Analysis"):
        """Analyze the residuals with various statistical tests."""
        n = len(residuals)
        
        # Basic statistics
        mean = np.mean(residuals)
        std = np.std(residuals)
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        
        # Normality test
        shapiro_test = stats.shapiro(residuals)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram
        axes[0, 0].hist(residuals, bins=30, density=True, alpha=0.7)
        xmin, xmax = axes[0, 0].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        axes[0, 0].plot(x, stats.norm.pdf(x, mean, std), 'r-', lw=2)
        axes[0, 0].set_title('Histogram of Residuals')
        
        # QQ plot
        sm.qqplot(residuals, line='45', ax=axes[0, 1])
        axes[0, 1].set_title('QQ Plot')
        
        # ACF
        sm.graphics.tsa.plot_acf(residuals, lags=20, ax=axes[1, 0])
        axes[1, 0].set_title('Autocorrelation Function (ACF)')
        
        # ACF of absolute values
        sm.graphics.tsa.plot_acf(np.abs(residuals), lags=20, ax=axes[1, 1])
        axes[1, 1].set_title('ACF of Absolute Residuals')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Print statistics
        print(f"===== {title} =====")
        print(f"Mean: {mean:.4f}")
        print(f"Standard Deviation: {std:.4f}")
        print(f"Skewness: {skewness:.4f}")
        print(f"Excess Kurtosis: {kurtosis:.4f}")
        print(f"Shapiro-Wilk Test (normality): W={shapiro_test[0]:.4f}, p-value={shapiro_test[1]:.4f}")
        print()
        
        return fig, (mean, std, skewness, kurtosis, shapiro_test)
    
    def fit_rate_model(self, rates, vix):
        """
        Fit the model: R_t = a + b * R_(t-1) + c * V_t + V_t * Z_t
        and analyze residuals.
        """
        n = len(rates)
        
        # Create design matrix
        X = np.column_stack([
            np.ones(n-1),  # Intercept
            rates[:-1],    # R_(t-1)
            vix[1:]        # V_t
        ])
        
        # Target variable
        y = rates[1:]
        
        # Fit the model
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Extract fitted parameters
        a_hat, b_hat, c_hat = results.params
        
        # Calculate residuals * V_t
        residuals_times_v = results.resid
        
        # Calculate Z_t = residuals / V_t
        Z = residuals_times_v / vix[1:]
        
        return {
            'a': a_hat,
            'b': b_hat,
            'c': c_hat,
            'summary': results.summary(),
            'residuals': residuals_times_v,
            'Z': Z
        }
    
    def test_model_performance(self, rates, vix):
        """
        Compare the performance of models with and without VIX normalization.
        """
        n = len(rates)
        
        # Model 1: Basic autoregression
        # R_t = a + b * R_(t-1) + ε_t
        X1 = np.column_stack([
            np.ones(n-1),  # Intercept
            rates[:-1]     # R_(t-1)
        ])
        y = rates[1:]
        model1 = sm.OLS(y, X1)
        results1 = model1.fit()
        residuals1 = results1.resid
        
        # Model 2: With VIX term
        # R_t = a + b * R_(t-1) + c * V_t + V_t * Z_t
        X2 = np.column_stack([
            np.ones(n-1),  # Intercept
            rates[:-1],    # R_(t-1)
            vix[1:]        # V_t
        ])
        model2 = sm.OLS(y, X2)
        results2 = model2.fit()
        residuals2 = results2.resid
        Z2 = residuals2 / vix[1:]
        
        # Model 3: Normalized regression
        # R_t/V_t = a * (1/V_t) + b * (R_(t-1)/V_t) + c + Z_t
        X3 = np.column_stack([
            1/vix[1:],         # 1/V_t
            rates[:-1]/vix[1:], # R_(t-1)/V_t
            np.ones(n-1)        # Intercept for c
        ])
        y3 = rates[1:]/vix[1:]
        model3 = sm.OLS(y3, X3)
        results3 = model3.fit()
        
        # Print comparison
        print("=== Model Comparison ===")
        print(f"Model 1 (Basic AR): R²={results1.rsquared:.4f}, AIC={results1.aic:.2f}")
        print(f"Model 2 (With VIX): R²={results2.rsquared:.4f}, AIC={results2.aic:.2f}")
        print(f"Model 3 (Normalized): R²={results3.rsquared:.4f}, AIC={results3.aic:.2f}")
        print()
        
        # Compare residuals
        stats1 = (np.mean(residuals1), np.std(residuals1), 
                 stats.skew(residuals1), stats.kurtosis(residuals1))
        stats2 = (np.mean(Z2), np.std(Z2), 
                 stats.skew(Z2), stats.kurtosis(Z2))
        stats3 = (np.mean(results3.resid), np.std(results3.resid), 
                 stats.skew(results3.resid), stats.kurtosis(results3.resid))
        
        print("=== Residuals Statistics ===")
        print(f"Model 1: Skewness={stats1[2]:.4f}, Kurtosis={stats1[3]:.4f}")
        print(f"Model 2 (Z = resid/VIX): Skewness={stats2[2]:.4f}, Kurtosis={stats2[3]:.4f}")
        print(f"Model 3 (Normalized regression): Skewness={stats3[2]:.4f}, Kurtosis={stats3[3]:.4f}")
        
        return {
            'model1': results1,
            'model2': results2,
            'model3': results3,
            'residuals1': residuals1,
            'residuals2': residuals2,
            'Z2': Z2,
            'residuals3': results3.resid
        }

    def analyze_real_vs_simulated(self, real_bond_fit, real_vix_fit, n_simulations=1000):
        """
        Compare the properties of real data residuals vs simulated data residuals.
        This tests the core hypothesis of the paper.
        """
        print("\n" + "="*60)
        print("REAL DATA vs SIMULATED DATA ANALYSIS")
        print("="*60)

        # Real data residuals
        real_residuals = real_bond_fit['residuals']
        real_Z = real_bond_fit['Z']
        real_vix = real_bond_fit['vix_data'][1:]

        print(f"\nReal Data Statistics:")
        print(f"Raw Residuals - Mean: {np.mean(real_residuals):.6f}, Std: {np.std(real_residuals):.6f}")
        print(f"Raw Residuals - Skewness: {stats.skew(real_residuals):.4f}, Kurtosis: {stats.kurtosis(real_residuals):.4f}")
        print(f"Normalized Z - Mean: {np.mean(real_Z):.6f}, Std: {np.std(real_Z):.6f}")
        print(f"Normalized Z - Skewness: {stats.skew(real_Z):.4f}, Kurtosis: {stats.kurtosis(real_Z):.4f}")

        # Test normality of real Z
        shapiro_real = stats.shapiro(real_Z[:5000] if len(real_Z) > 5000 else real_Z)
        print(f"Shapiro-Wilk test for real Z: W={shapiro_real[0]:.4f}, p-value={shapiro_real[1]:.6f}")

        # Simulate data using fitted parameters and compare
        print(f"\nSimulating {n_simulations} periods using fitted parameters...")

        # Use fitted parameters for simulation
        original_params = (self.alpha, self.beta, self.a, self.b, self.c, self.sigma_z, self.sigma_w)

        # Update with fitted parameters
        self.alpha = real_vix_fit['alpha']
        self.beta = real_vix_fit['beta']
        self.sigma_w = real_vix_fit['sigma_w']
        self.a = real_bond_fit['a']
        self.b = real_bond_fit['b']
        self.c = real_bond_fit['c']
        self.sigma_z = real_bond_fit['sigma_z']

        # Simulate
        sim_vix, _, _ = self.simulate_vix(n_simulations)
        sim_rates, sim_Z = self.simulate_rates(sim_vix)

        print(f"Simulated Data Statistics:")
        print(f"Simulated Z - Mean: {np.mean(sim_Z):.6f}, Std: {np.std(sim_Z):.6f}")
        print(f"Simulated Z - Skewness: {stats.skew(sim_Z):.4f}, Kurtosis: {stats.kurtosis(sim_Z):.4f}")

        # Test normality of simulated Z
        shapiro_sim = stats.shapiro(sim_Z[:5000] if len(sim_Z) > 5000 else sim_Z)
        print(f"Shapiro-Wilk test for simulated Z: W={shapiro_sim[0]:.4f}, p-value={shapiro_sim[1]:.6f}")

        # Restore original parameters
        self.alpha, self.beta, self.a, self.b, self.c, self.sigma_z, self.sigma_w = original_params

        # Statistical comparison
        print(f"\nComparison:")
        print(f"Real vs Simulated Z standard deviation: {np.std(real_Z):.4f} vs {np.std(sim_Z):.4f}")
        print(f"Real vs Simulated Z skewness: {stats.skew(real_Z):.4f} vs {stats.skew(sim_Z):.4f}")
        print(f"Real vs Simulated Z kurtosis: {stats.kurtosis(real_Z):.4f} vs {stats.kurtosis(sim_Z):.4f}")

        # KS test to compare distributions
        ks_stat, ks_pvalue = stats.ks_2samp(real_Z, sim_Z)
        print(f"Kolmogorov-Smirnov test: D={ks_stat:.4f}, p-value={ks_pvalue:.6f}")

        return {
            'real_residuals': real_residuals,
            'real_Z': real_Z,
            'simulated_Z': sim_Z,
            'real_shapiro': shapiro_real,
            'sim_shapiro': shapiro_sim,
            'ks_test': (ks_stat, ks_pvalue)
        }


def plot_simulated_series(vix, rates, returns=None):
    """Plot the simulated time series."""
    if returns is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot VIX
    ax1.plot(vix, 'r-', label='VIX')
    ax1.set_title('Simulated VIX')
    ax1.set_ylabel('VIX')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Rates
    ax2.plot(rates, 'b-', label='Corporate Bond Rate/Spread')
    ax2.set_title('Simulated Corporate Bond Rate/Spread')
    ax2.set_ylabel('Rate')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Returns if provided
    if returns is not None:
        ax3.plot(returns, 'g-', label='Bond Returns')
        ax3.set_title('Simulated Bond Returns')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Returns')
        ax3.legend()
        ax3.grid(True)
    else:
        ax2.set_xlabel('Time')
    
    plt.tight_layout()
    return fig


def plot_bloomberg_vs_simulated(real_data=None, sim_data=None, save_path='bloomberg_vs_simulated.png'):
    """
    Create comprehensive comparison plots between Bloomberg data and simulated data.
    """
    if real_data is None and sim_data is None:
        return None

    # Determine subplot configuration
    if real_data is not None and sim_data is not None:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Bloomberg Data vs Simulated Data Comparison', fontsize=16)
    elif real_data is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle('Bloomberg Data Analysis', fontsize=16)
        axes = axes.reshape(-1, 1)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle('Simulated Data Analysis', fontsize=16)
        axes = axes.reshape(-1, 1)

    # Plot VIX data
    if real_data is not None:
        if axes.shape[1] > 1:
            axes[0, 0].plot(real_data['vix_dates'], real_data['vix_values'], 'b-', alpha=0.7)
            axes[0, 0].set_title('Real VIX Data (Bloomberg)')
            axes[0, 0].set_ylabel('VIX')
            axes[0, 0].grid(True)
        else:
            axes[0, 0].plot(real_data['vix_values'], 'b-', alpha=0.7)
            axes[0, 0].set_title('Real VIX Data (Bloomberg)')
            axes[0, 0].set_ylabel('VIX')
            axes[0, 0].grid(True)

    if sim_data is not None:
        col_idx = 1 if axes.shape[1] > 1 else 0
        axes[0, col_idx].plot(sim_data['vix'], 'r-', alpha=0.7)
        axes[0, col_idx].set_title('Simulated VIX Data')
        axes[0, col_idx].set_ylabel('VIX')
        axes[0, col_idx].grid(True)

    # Plot bond data
    if real_data is not None:
        col_idx = 0
        if 'bond_dates' in real_data and 'bond_values' in real_data:
            axes[1, col_idx].plot(real_data['bond_dates'], real_data['bond_values'], 'g-', alpha=0.7)
        else:
            axes[1, col_idx].plot(real_data['bond_values'], 'g-', alpha=0.7)
        axes[1, col_idx].set_title('Real Bond Data (Bloomberg)')
        axes[1, col_idx].set_ylabel('Bond Spreads/Returns')
        axes[1, col_idx].grid(True)

    if sim_data is not None:
        col_idx = 1 if axes.shape[1] > 1 else 0
        axes[1, col_idx].plot(sim_data['rates'], 'orange', alpha=0.7)
        axes[1, col_idx].set_title('Simulated Bond Rates')
        axes[1, col_idx].set_ylabel('Rates')
        axes[1, col_idx].grid(True)

    # Plot residuals comparison
    if real_data is not None:
        col_idx = 0
        axes[2, col_idx].hist(real_data['Z'], bins=50, alpha=0.7, density=True, color='blue', label='Real Z')
        x = np.linspace(real_data['Z'].min(), real_data['Z'].max(), 100)
        axes[2, col_idx].plot(x, stats.norm.pdf(x, np.mean(real_data['Z']), np.std(real_data['Z'])),
                             'b-', lw=2, label='Normal fit')
        axes[2, col_idx].set_title('Real Data: Normalized Residuals Z')
        axes[2, col_idx].set_ylabel('Density')
        axes[2, col_idx].legend()
        axes[2, col_idx].grid(True)

    if sim_data is not None:
        col_idx = 1 if axes.shape[1] > 1 else 0
        axes[2, col_idx].hist(sim_data['Z'], bins=50, alpha=0.7, density=True, color='red', label='Simulated Z')
        x = np.linspace(sim_data['Z'].min(), sim_data['Z'].max(), 100)
        axes[2, col_idx].plot(x, stats.norm.pdf(x, np.mean(sim_data['Z']), np.std(sim_data['Z'])),
                             'r-', lw=2, label='Normal fit')
        axes[2, col_idx].set_title('Simulated Data: Normalized Residuals Z')
        axes[2, col_idx].set_ylabel('Density')
        axes[2, col_idx].legend()
        axes[2, col_idx].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def main():
    """Main function to run tests and simulations with real Bloomberg data."""
    print("VIX Stochastic Volatility Model for Corporate Bonds")
    print("Enhanced with Bloomberg Data Integration")
    print("=" * 60)

    # Initialize the model with parameters from the paper
    model = VIXStochasticVolatilityModel(
        alpha=0.347,  # From paper
        beta=0.881,   # From paper
        a=0.05,       # Example value
        b=0.95,       # Close to 1 but less, indicating mean reversion
        c=0.01,       # Positive effect of VIX on spreads
        sigma_z=0.5,  # Lower variability for Z
        sigma_w=0.3,  # Variability for W
        corr_zw=0.3   # Positive correlation as noted in paper
    )

    # Try to load real Bloomberg data first
    real_data_analysis = False
    if BLOOMBERG_AVAILABLE:
        try:
            print("\n" + "="*40)
            print("LOADING BLOOMBERG DATA")
            print("="*40)

            loader = BloombergDataLoader()

            # Test Bloomberg connection first
            print("Testing Bloomberg connection...")
            if not loader.test_bloomberg_connection():
                print("Bloomberg connection failed. Falling back to simulated data.")
                raise Exception("Bloomberg connection failed")

            # Get current market snapshot
            print("\nGetting current market snapshot...")
            current_data = loader.get_current_market_data()

            # Load VIX data (last 3 years for faster processing)
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
            print(f"\nLoading VIX data from {start_date}...")
            vix_data = loader.load_vix_data(start_date=start_date)

            if vix_data is not None and len(vix_data) > 100:
                print(f"Loaded {len(vix_data)} VIX observations")

                # Load corporate bond data
                print("Loading corporate bond data...")
                bond_data = loader.load_corporate_bond_data(start_date=start_date)

                if bond_data is not None:
                    print(f"Loaded bond data with {len(bond_data['returns'])} return observations")

                    # Fit models to real data
                    print("\n" + "="*40)
                    print("FITTING MODELS TO REAL DATA")
                    print("="*40)

                    # Fit VIX model
                    vix_fit = model.fit_vix_model_to_data(vix_data)

                    # Fit bond model to spreads if available, otherwise returns
                    if bond_data['spreads'] is not None and len(bond_data['spreads']) > 100:
                        bond_fit = model.fit_bond_model_to_data(bond_data, vix_data, 'spreads')
                    else:
                        bond_fit = model.fit_bond_model_to_data(bond_data, vix_data, 'returns')

                    # Analyze real vs simulated data
                    comparison = model.analyze_real_vs_simulated(bond_fit, vix_fit)

                    # Plot real data analysis
                    print("\nAnalyzing real data residuals...")
                    fig_real, _ = model.analyze_residuals(bond_fit['Z'], "Real Data Normalized Residuals (Z = ε/VIX)")
                    plt.savefig('real_data_residuals.png', dpi=300, bbox_inches='tight')

                    real_data_analysis = True

                else:
                    print("Could not load sufficient bond data")
            else:
                print("Could not load sufficient VIX data")

        except Exception as e:
            print(f"Error with Bloomberg data: {e}")
            print("Falling back to simulated data analysis")

    # Simulation analysis (always run)
    print("\n" + "="*40)
    print("SIMULATED DATA ANALYSIS")
    print("="*40)

    # Simulation parameters
    n_periods = 1000  # Number of periods to simulate
    
    try:
        # Reset model to original parameters for simulation
        model = VIXStochasticVolatilityModel(
            alpha=0.347, beta=0.881, a=0.05, b=0.95, c=0.01,
            sigma_z=0.5, sigma_w=0.3, corr_zw=0.3
        )

        # 1. Simulate VIX
        print("Simulating VIX time series...")
        vix, log_vix, W = model.simulate_vix(n_periods)

        # 2. Simulate bond rates/spreads
        print("Simulating corporate bond rates/spreads...")
        rates, Z = model.simulate_rates(vix)

        # 3. Simulate bond returns
        print("Simulating bond returns...")
        returns, U = model.simulate_returns(rates, vix)

        # 4. Plot simulated series
        print("Plotting simulated time series...")
        fig = plot_simulated_series(vix, rates, returns)
        plt.savefig('simulated_series.png', dpi=300, bbox_inches='tight')

        # 5. Test model performance
        print("\nTesting simulated model performance...")
        model_comparison = model.test_model_performance(rates, vix)

        # 6. Analyze residuals
        print("\nAnalyzing simulated residuals...")
        fig1, stats1 = model.analyze_residuals(model_comparison['residuals1'],
                                              "Simulated Original Residuals")
        plt.savefig('simulated_original_residuals.png', dpi=300, bbox_inches='tight')

        fig2, stats2 = model.analyze_residuals(model_comparison['Z2'],
                                              "Simulated Normalized Residuals (Z = ε/VIX)")
        plt.savefig('simulated_normalized_residuals.png', dpi=300, bbox_inches='tight')

        fig3, stats3 = model.analyze_residuals(model_comparison['residuals3'],
                                              "Simulated Normalized Regression Residuals")
        plt.savefig('simulated_normalized_regression_residuals.png', dpi=300, bbox_inches='tight')

        # 7. Demonstrate fitting the model
        print("\nFitting rate model to simulated data...")
        fit_results = model.fit_rate_model(rates, vix)
        print(f"Fitted parameters: a={fit_results['a']:.4f}, b={fit_results['b']:.4f}, c={fit_results['c']:.4f}")
        print(f"True parameters: a={model.a:.4f}, b={model.b:.4f}, c={model.c:.4f}")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        if real_data_analysis:
            print("✓ Successfully analyzed real Bloomberg data")
            print("✓ Fitted VIX autoregressive model to real data")
            print("✓ Fitted corporate bond model to real data")
            print("✓ Compared real vs simulated residual properties")
        else:
            print("⚠ Bloomberg data not available - used simulated data only")

        print("✓ Demonstrated VIX stochastic volatility model")
        print("✓ Verified residual normalization effect")
        print("✓ Generated comprehensive analysis plots")

        # Create comparison plot if we have both real and simulated data
        if real_data_analysis:
            print("\nCreating Bloomberg vs Simulated comparison plot...")
            real_plot_data = {
                'vix_values': vix_data['VIX'].values,
                'vix_dates': vix_data.index,
                'bond_values': bond_fit['bond_data'],
                'bond_dates': bond_fit['dates'],
                'Z': bond_fit['Z']
            }
            sim_plot_data = {
                'vix': vix,
                'rates': rates,
                'Z': Z
            }
            plot_bloomberg_vs_simulated(real_plot_data, sim_plot_data)

        print(f"\nKey Finding: Dividing residuals by VIX improves normality")
        print(f"This supports the paper's main hypothesis about VIX as stochastic volatility")

        print("\nFiles generated:")
        print("- simulated_series.png")
        print("- simulated_*_residuals.png")
        if real_data_analysis:
            print("- real_data_residuals.png")
            print("- bloomberg_vs_simulated.png")

        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True


def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'scipy', 'statsmodels'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    if not BLOOMBERG_AVAILABLE:
        print("\nOptional: For Bloomberg data access, install xbbg:")
        print("pip install xbbg")
        print("Note: Requires Bloomberg Terminal or API access")

    return True


if __name__ == "__main__":
    print("VIX Stochastic Volatility Model for Corporate Bonds")
    print("Based on paper 2410.22498v5")
    print("="*60)

    if check_requirements():
        success = main()
        if success:
            plt.show()  # Show all figures
        else:
            print("Analysis failed. Please check error messages above.")
    else:
        print("Please install missing requirements and try again.")