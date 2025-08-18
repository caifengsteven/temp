"""
Bloomberg-specific implementation for RKHS Discount Curve Trading
This version uses proper Bloomberg API calls with xbbg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import xbbg
    from xbbg import blp
    BLOOMBERG_AVAILABLE = True
    print("Bloomberg xbbg library available")
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("Bloomberg xbbg not available")

class BloombergDiscountCurveTrading:
    """
    Bloomberg-specific implementation of discount curve trading strategy
    """
    
    def __init__(self):
        self.treasury_tickers = {
            '2Y': 'USGG2YR Index',    # 2-Year Treasury Yield
            '5Y': 'USGG5YR Index',    # 5-Year Treasury Yield  
            '10Y': 'USGG10YR Index',  # 10-Year Treasury Yield
            '30Y': 'USGG30YR Index'   # 30-Year Treasury Yield
        }
        
        self.bond_tickers = {
            '2Y': 'GT2 Govt',         # 2-Year Treasury Bond
            '5Y': 'GT5 Govt',         # 5-Year Treasury Bond
            '10Y': 'GT10 Govt',       # 10-Year Treasury Bond
            '30Y': 'GT30 Govt'        # 30-Year Treasury Bond
        }
        
    def fetch_treasury_yields(self, start_date, end_date):
        """
        Fetch Treasury yield data from Bloomberg
        """
        if not BLOOMBERG_AVAILABLE:
            raise ImportError("Bloomberg xbbg not available")
        
        try:
            # Fetch yield data
            tickers = list(self.treasury_tickers.values())
            
            # Use proper xbbg function
            data = blp.bdh(
                tickers=tickers,
                flds='PX_LAST',
                start_date=start_date,
                end_date=end_date
            )
            
            # Rename columns for easier access
            column_mapping = {v: k for k, v in self.treasury_tickers.items()}
            data = data.rename(columns=column_mapping)
            
            return data.dropna()
            
        except Exception as e:
            print(f"Error fetching Bloomberg data: {e}")
            raise
    
    def fetch_bond_prices(self, start_date, end_date):
        """
        Fetch Treasury bond price data from Bloomberg
        """
        if not BLOOMBERG_AVAILABLE:
            raise ImportError("Bloomberg xbbg not available")
        
        try:
            tickers = list(self.bond_tickers.values())
            
            data = blp.bdh(
                tickers=tickers,
                flds='PX_LAST',
                start_date=start_date,
                end_date=end_date
            )
            
            # Rename columns
            column_mapping = {v: k for k, v in self.bond_tickers.items()}
            data = data.rename(columns=column_mapping)
            
            return data.dropna()
            
        except Exception as e:
            print(f"Error fetching Bloomberg bond data: {e}")
            raise
    
    def fetch_real_time_data(self):
        """
        Fetch real-time Treasury data for live trading
        """
        if not BLOOMBERG_AVAILABLE:
            raise ImportError("Bloomberg xbbg not available")
        
        try:
            tickers = list(self.treasury_tickers.values())
            
            # Get real-time data
            data = blp.bdp(
                tickers=tickers,
                flds=['PX_LAST', 'CHG_NET_1D', 'YLD_YTM_MID']
            )
            
            return data
            
        except Exception as e:
            print(f"Error fetching real-time data: {e}")
            raise
    
    def calculate_discount_curve_features(self, yield_data):
        """
        Calculate discount curve features from yield data
        """
        features = pd.DataFrame(index=yield_data.index)
        
        # Level: Average yield across the curve
        features['level'] = yield_data.mean(axis=1)
        
        # Slope: Long-term minus short-term yields
        if '30Y' in yield_data.columns and '2Y' in yield_data.columns:
            features['slope'] = yield_data['30Y'] - yield_data['2Y']
        
        # Curvature: Butterfly spread
        if all(col in yield_data.columns for col in ['2Y', '10Y', '30Y']):
            features['curvature'] = 2 * yield_data['10Y'] - yield_data['2Y'] - yield_data['30Y']
        
        # Calculate rolling statistics for mean reversion
        window = 20
        for feature in ['level', 'slope', 'curvature']:
            if feature in features.columns:
                features[f'{feature}_ma'] = features[feature].rolling(window).mean()
                features[f'{feature}_std'] = features[feature].rolling(window).std()
                features[f'{feature}_zscore'] = (
                    (features[feature] - features[f'{feature}_ma']) / 
                    features[f'{feature}_std']
                )
        
        return features
    
    def generate_trading_signals(self, features_data, threshold=1.5):
        """
        Generate trading signals based on yield curve analysis
        """
        signals = pd.DataFrame(index=features_data.index)
        
        # Mean reversion signals based on z-scores
        for feature in ['level', 'slope', 'curvature']:
            zscore_col = f'{feature}_zscore'
            if zscore_col in features_data.columns:
                # Generate signal when z-score exceeds threshold
                condition = features_data[zscore_col].abs() > threshold
                signals[f'{feature}_signal'] = np.where(
                    condition,
                    -np.sign(features_data[zscore_col]),  # Mean reversion
                    0
                )
        
        # Combine signals for different tenors
        # 2Y: Sensitive to level and Fed policy
        signals['2Y_position'] = (
            0.6 * signals.get('level_signal', 0) + 
            0.4 * signals.get('slope_signal', 0)
        )
        
        # 5Y: Balanced exposure
        signals['5Y_position'] = (
            0.4 * signals.get('level_signal', 0) + 
            0.3 * signals.get('slope_signal', 0) +
            0.3 * signals.get('curvature_signal', 0)
        )
        
        # 10Y: Sensitive to long-term expectations
        signals['10Y_position'] = (
            0.3 * signals.get('level_signal', 0) + 
            0.4 * signals.get('slope_signal', 0) +
            0.3 * signals.get('curvature_signal', 0)
        )
        
        # 30Y: Duration play
        signals['30Y_position'] = (
            0.2 * signals.get('level_signal', 0) + 
            0.5 * signals.get('slope_signal', 0) +
            0.3 * signals.get('curvature_signal', 0)
        )
        
        return signals
    
    def backtest_strategy(self, bond_prices, signals, transaction_cost=0.002):
        """
        Backtest the trading strategy using bond price data
        """
        # Calculate bond returns
        returns = bond_prices.pct_change().dropna()
        
        # Align signals with returns
        common_dates = signals.index.intersection(returns.index)
        signals_aligned = signals.loc[common_dates]
        returns_aligned = returns.loc[common_dates]
        
        # Calculate strategy returns for each tenor
        strategy_returns = pd.DataFrame(index=common_dates)
        
        for tenor in ['2Y', '5Y', '10Y', '30Y']:
            position_col = f'{tenor}_position'
            
            if position_col in signals_aligned.columns and tenor in returns_aligned.columns:
                # Lag positions by 1 day
                positions = signals_aligned[position_col].shift(1).fillna(0)
                
                # Calculate gross returns
                gross_returns = positions * returns_aligned[tenor]
                
                # Apply transaction costs
                position_changes = positions.diff().abs().fillna(0)
                costs = position_changes * transaction_cost
                net_returns = gross_returns - costs
                
                strategy_returns[f'{tenor}_strategy'] = net_returns
        
        # Portfolio returns (equal weight across tenors)
        portfolio_returns = strategy_returns.mean(axis=1)
        
        # Calculate performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        performance_metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}"
        }
        
        return portfolio_returns, performance_metrics, strategy_returns
    
    def run_live_strategy(self):
        """
        Run the strategy with live Bloomberg data
        """
        print("Fetching live Treasury data...")
        
        try:
            # Get current market data
            live_data = self.fetch_real_time_data()
            print("Current Treasury Yields:")
            print(live_data)
            
            # Get historical data for context
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # 2 months of data
            
            historical_yields = self.fetch_treasury_yields(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # Calculate features
            features = self.calculate_discount_curve_features(historical_yields)
            
            # Generate current signals
            current_signals = self.generate_trading_signals(features)
            
            # Get latest signals
            latest_signals = current_signals.iloc[-1]
            
            print("\nCurrent Trading Signals:")
            for tenor in ['2Y', '5Y', '10Y', '30Y']:
                position_col = f'{tenor}_position'
                if position_col in latest_signals:
                    signal = latest_signals[position_col]
                    direction = "LONG" if signal > 0 else "SHORT" if signal < 0 else "NEUTRAL"
                    print(f"{tenor} Treasury: {direction} (Signal: {signal:.2f})")
            
            return latest_signals, features
            
        except Exception as e:
            print(f"Error in live strategy: {e}")
            return None, None

def main_bloomberg():
    """
    Main function for Bloomberg implementation
    """
    if not BLOOMBERG_AVAILABLE:
        print("Bloomberg xbbg not available. Please install with: pip install xbbg")
        return
    
    strategy = BloombergDiscountCurveTrading()
    
    try:
        # Historical backtest
        print("=== Historical Backtest ===")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        # Fetch historical data
        yield_data = strategy.fetch_treasury_yields(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        bond_data = strategy.fetch_bond_prices(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Calculate features and signals
        features = strategy.calculate_discount_curve_features(yield_data)
        signals = strategy.generate_trading_signals(features)
        
        # Backtest
        portfolio_returns, performance, strategy_returns = strategy.backtest_strategy(
            bond_data, signals
        )
        
        print("\nBacktest Results:")
        for metric, value in performance.items():
            print(f"{metric}: {value}")
        
        # Live strategy
        print("\n=== Live Strategy ===")
        live_signals, live_features = strategy.run_live_strategy()
        
    except Exception as e:
        print(f"Error running Bloomberg strategy: {e}")
        print("Make sure Bloomberg Terminal is running and you have proper permissions")

if __name__ == "__main__":
    main_bloomberg()
