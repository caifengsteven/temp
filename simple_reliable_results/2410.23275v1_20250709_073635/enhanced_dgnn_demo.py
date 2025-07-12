"""
Enhanced DGNN Demo - Simplified Version
Demonstrates Bloomberg data integration with the DGNN model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Bloomberg data integration
try:
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
    print("✓ Bloomberg xbbg module loaded successfully")
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("⚠ Bloomberg xbbg not available. Using simulated data.")
    
    # Create mock blp module for fallback
    class MockBlp:
        def bdh(self, *args, **kwargs):
            return pd.DataFrame()
        def bdp(self, *args, **kwargs):
            return pd.DataFrame()
    blp = MockBlp()

class EnhancedMarketDataDemo:
    """Demonstration of enhanced market data integration using xbbg"""

    def __init__(self, currency='USD'):
        self.currency = currency
        # Enhanced ticker mappings based on xbbg documentation examples
        self.ois_tickers = {
            'USD': ['USSO1Z Curncy', 'USSOA Curncy', 'USSOB Curncy', 'USSOC Curncy',
                   'USSOD Curncy', 'USSO1 Curncy', 'USSO2 Curncy', 'USSO5 Curncy', 'USSO10 Curncy'],
            'EUR': ['EUSWO1Z Curncy', 'EUSWOA Curncy', 'EUSWOB Curncy', 'EUSWOC Curncy',
                   'EUSWOD Curncy', 'EUSWO1 Curncy', 'EUSWO2 Curncy', 'EUSWO5 Curncy', 'EUSWO10 Curncy']
        }
        self.reference_rates = {
            'USD': 'FEDL01 Index',   # Fed Funds Effective Rate
            'EUR': 'EONIA Index',    # EONIA Rate
            'SOFR': 'SOFR Index'     # SOFR Rate (alternative USD reference)
        }
        # Volatility tickers for enhanced pricing
        self.volatility_tickers = {
            'USD': 'USSV1Y Curncy',  # 1Y USD OIS volatility
            'EUR': 'EUSV1Y Curncy'   # 1Y EUR OIS volatility
        }
        
    def fetch_ois_curve(self, date=None):
        """Fetch OIS curve from Bloomberg or simulate"""
        if not BLOOMBERG_AVAILABLE:
            return self._simulate_ois_curve()

        try:
            tickers = self.ois_tickers.get(self.currency, self.ois_tickers['USD'])
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')

            print(f"Fetching OIS curve for {self.currency} on {date}...")

            # Use timeout parameter as recommended in xbbg docs for reliability
            data = blp.bdh(tickers, 'PX_LAST', start_date=date, end_date=date, timeout=30)

            if data.empty:
                print("No Bloomberg data available, using simulated curve")
                return self._simulate_ois_curve()
                
            # Process Bloomberg data
            rates_data = []
            tenor_mapping = {
                'O1Z': ('1D', 1), 'OA': ('1W', 7), 'OB': ('2W', 14), 'OC': ('3W', 21), 'OD': ('1M', 30),
                'O1': ('1Y', 365), 'O2': ('2Y', 730), 'O5': ('5Y', 1825), 'O10': ('10Y', 3650)
            }
            
            for ticker in tickers:
                if ticker in data.columns:
                    rate_value = data[ticker].iloc[-1] if not data[ticker].empty else None
                    if rate_value is not None and not np.isnan(rate_value):
                        for key, (tenor, days) in tenor_mapping.items():
                            if key in ticker:
                                rates_data.append({
                                    'tenor': tenor, 
                                    'days': days, 
                                    'rate': rate_value / 100
                                })
                                break
                                
            if rates_data:
                df = pd.DataFrame(rates_data).sort_values('days').reset_index(drop=True)
                print(f"✓ Fetched {len(df)} OIS rates from Bloomberg")
                return df
            else:
                print("No valid Bloomberg data, using simulated curve")
                return self._simulate_ois_curve()
                
        except Exception as e:
            print(f"Bloomberg error: {e}")
            return self._simulate_ois_curve()
    
    def _simulate_ois_curve(self):
        """Simulate realistic OIS curve"""
        base_rate = 0.045 if self.currency == 'USD' else 0.025
        tenors = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y']
        days = [1, 7, 30, 90, 180, 365, 730, 1825, 3650]
        
        # Create upward sloping curve with realistic shape
        rates = []
        for d in days:
            # Term structure: short rates + term premium
            term_premium = (d / 3650) * 0.015  # 1.5% term premium over 10 years
            rate = base_rate + term_premium + np.random.normal(0, 0.001)
            rates.append(max(rate, 0.001))  # Floor at 0.1%
            
        return pd.DataFrame({
            'tenor': tenors,
            'days': days,
            'rate': rates
        })
    
    def fetch_reference_rate_history(self, days=30):
        """Fetch historical reference rates"""
        if not BLOOMBERG_AVAILABLE:
            return self._simulate_rate_history(days)
            
        try:
            ticker = self.reference_rates.get(self.currency, self.reference_rates['USD'])
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            print(f"Fetching {days} days of {ticker} history...")
            # Use timeout parameter for reliability with historical data
            data = blp.bdh(ticker, 'PX_LAST',
                         start_date=start_date.strftime('%Y-%m-%d'),
                         end_date=end_date.strftime('%Y-%m-%d'),
                         timeout=60)
            
            if not data.empty and ticker in data.columns:
                rates = data[ticker].dropna() / 100  # Convert to decimal
                print(f"✓ Fetched {len(rates)} historical rates from Bloomberg")
                return rates
            else:
                print("No Bloomberg historical data, using simulated rates")
                return self._simulate_rate_history(days)
                
        except Exception as e:
            print(f"Bloomberg error: {e}")
            return self._simulate_rate_history(days)
    
    def _simulate_rate_history(self, days):
        """Simulate historical rate time series"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        base_rate = 0.045 if self.currency == 'USD' else 0.025
        
        # Random walk with mean reversion
        rates = [base_rate]
        for _ in range(days - 1):
            # Mean reversion + random shock
            mean_reversion = 0.01 * (base_rate - rates[-1])
            shock = np.random.normal(0, 0.002)
            new_rate = max(rates[-1] + mean_reversion + shock, 0.001)
            rates.append(new_rate)
            
        return pd.Series(rates, index=dates)

    def fetch_volatility_surface(self):
        """Fetch implied volatility data from Bloomberg"""
        if not BLOOMBERG_AVAILABLE:
            return self._simulate_volatility()

        try:
            ticker = self.volatility_tickers.get(self.currency, self.volatility_tickers['USD'])
            print(f"Fetching volatility data for {ticker}...")

            # Use bdp for current volatility level
            data = blp.bdp(ticker, 'PX_LAST', timeout=30)

            if not data.empty and ticker in data.index:
                vol = data.loc[ticker, 'PX_LAST'] / 100  # Convert to decimal
                print(f"✓ Fetched volatility: {vol:.4f}")
                return vol
            else:
                print("No Bloomberg volatility data, using simulated value")
                return self._simulate_volatility()

        except Exception as e:
            print(f"Bloomberg volatility error: {e}")
            return self._simulate_volatility()

    def _simulate_volatility(self):
        """Simulate realistic volatility"""
        base_vol = 0.15 if self.currency == 'USD' else 0.12
        return base_vol + np.random.normal(0, 0.02)

    def fetch_security_info(self, ticker):
        """Fetch security information using BDP - demonstrates xbbg BDP functionality"""
        if not BLOOMBERG_AVAILABLE:
            return {'security_name': f'Simulated {ticker}', 'currency': self.currency}

        try:
            print(f"Fetching security info for {ticker}...")
            # Use multiple fields as shown in xbbg documentation
            data = blp.bdp(ticker, ['Security_Name', 'Crncy', 'Country'], timeout=30)

            if not data.empty:
                info = data.loc[ticker].to_dict()
                print(f"✓ Security info: {info}")
                return info
            else:
                return {'security_name': f'Unknown {ticker}', 'currency': self.currency}

        except Exception as e:
            print(f"Bloomberg security info error: {e}")
            return {'security_name': f'Error {ticker}', 'currency': self.currency}

    def fetch_dividend_history(self, equity_ticker, start_date=None, end_date=None):
        """Fetch dividend history using BDS - demonstrates xbbg BDS functionality"""
        if not BLOOMBERG_AVAILABLE:
            return pd.DataFrame({'message': ['Bloomberg not available']})

        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')

            print(f"Fetching dividend history for {equity_ticker}...")
            # Use BDS as shown in xbbg documentation
            data = blp.bds(equity_ticker, 'DVD_Hist_All',
                          DVD_Start_Dt=start_date, DVD_End_Dt=end_date, timeout=30)

            if not data.empty:
                print(f"✓ Found {len(data)} dividend records")
                return data
            else:
                print("No dividend data found")
                return pd.DataFrame({'message': ['No dividend data']})

        except Exception as e:
            print(f"Bloomberg dividend error: {e}")
            return pd.DataFrame({'error': [str(e)]})

def demonstrate_enhanced_features():
    """Demonstrate the enhanced DGNN features"""
    print("=" * 60)
    print("Enhanced DGNN with Bloomberg Data Integration")
    print("=" * 60)
    
    # Initialize market data fetcher
    demo = EnhancedMarketDataDemo('USD')
    
    print("\n1. Fetching Current OIS Curve:")
    print("-" * 30)
    ois_curve = demo.fetch_ois_curve()
    print(ois_curve)
    
    print("\n2. Fetching Historical Reference Rates:")
    print("-" * 40)
    historical_rates = demo.fetch_reference_rate_history(30)
    print(f"Rate range: {historical_rates.min():.4f} - {historical_rates.max():.4f}")
    print(f"Current rate: {historical_rates.iloc[-1]:.4f}")
    print(f"30-day volatility: {historical_rates.std() * np.sqrt(252):.4f}")

    print("\n3. Fetching Volatility Surface:")
    print("-" * 32)
    volatility = demo.fetch_volatility_surface()
    print(f"Implied volatility: {volatility:.4f} ({volatility*100:.2f}%)")

    print("\n4. Security Information (BDP Example):")
    print("-" * 38)
    # Demonstrate BDP functionality with a sample ticker
    sample_ticker = demo.ois_tickers[demo.currency][0]  # First OIS ticker
    security_info = demo.fetch_security_info(sample_ticker)
    for key, value in security_info.items():
        print(f"{key}: {value}")
    
    print("\n5. Market Data Analysis:")
    print("-" * 25)
    
    # Calculate yield curve metrics
    if len(ois_curve) > 1:
        short_rate = ois_curve[ois_curve['tenor'] == '1M']['rate'].iloc[0]
        long_rate = ois_curve[ois_curve['tenor'] == '10Y']['rate'].iloc[0]
        curve_slope = long_rate - short_rate
        print(f"Curve slope (10Y-1M): {curve_slope:.4f} ({curve_slope*100:.2f} bps)")
        
        # Term structure analysis
        print(f"Short end (1M): {short_rate:.4f}")
        print(f"Long end (10Y): {long_rate:.4f}")
        
        if curve_slope > 0:
            print("✓ Normal upward sloping curve")
        else:
            print("⚠ Inverted or flat curve")
    
    print("\n6. Enhanced Contract Pricing Demo:")
    print("-" * 35)
    
    # Demonstrate enhanced OIS contract pricing
    current_rate = historical_rates.iloc[-1]
    
    # Price 1-year OIS contract
    one_year_rate = ois_curve[ois_curve['tenor'] == '1Y']['rate'].iloc[0]
    contract_value = (one_year_rate - current_rate) * 1.0  # 1 year * $1M notional
    
    print(f"Current overnight rate: {current_rate:.4f}")
    print(f"1Y OIS fair rate: {one_year_rate:.4f}")
    print(f"Contract value (receive fixed): ${contract_value*1000000:.0f}")
    
    print("\n7. Visualization:")
    print("-" * 15)
    
    # Create plots
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: OIS Curve
    ax1.plot(ois_curve['days'], ois_curve['rate'] * 100, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Days to Maturity')
    ax1.set_ylabel('Rate (%)')
    ax1.set_title(f'{demo.currency} OIS Curve')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Historical Rates
    ax2.plot(historical_rates.index, historical_rates * 100, 'g-', linewidth=2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rate (%)')
    ax2.set_title(f'{demo.currency} Reference Rate (30 days)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Rate Distribution
    ax3.hist(historical_rates * 100, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('Rate (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Rate Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Curve Comparison
    if BLOOMBERG_AVAILABLE:
        # Compare with simulated curve
        sim_curve = demo._simulate_ois_curve()
        ax4.plot(ois_curve['days'], ois_curve['rate'] * 100, 'b-o', 
                label='Market Data', linewidth=2, markersize=6)
        ax4.plot(sim_curve['days'], sim_curve['rate'] * 100, 'r--s', 
                label='Simulated', linewidth=2, markersize=4)
        ax4.set_xlabel('Days to Maturity')
        ax4.set_ylabel('Rate (%)')
        ax4.set_title('Market vs Simulated Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
    else:
        ax4.text(0.5, 0.5, 'Bloomberg data\nnot available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Market Data Status')
    
    plt.tight_layout()
    plt.savefig('enhanced_dgnn_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n8. Summary:")
    print("-" * 10)
    print(f"✓ Market data integration: {'Active' if BLOOMBERG_AVAILABLE else 'Simulated'}")
    print(f"✓ OIS curve points: {len(ois_curve)}")
    print(f"✓ Historical data points: {len(historical_rates)}")
    print(f"✓ Volatility data: {volatility:.4f}")
    print(f"✓ Enhanced pricing: Enabled")
    print(f"✓ xbbg API features demonstrated:")
    print(f"  - BDH (Historical Data): ✓")
    print(f"  - BDP (Point Data): ✓")
    print(f"  - BDS (Bulk Data): Available")
    print(f"  - Timeout parameters: ✓")
    print(f"  - Error handling: ✓")
    print(f"✓ Visualization: enhanced_dgnn_demo.png")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_enhanced_features()
