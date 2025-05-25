import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import logging
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import time
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImpliedVolatilityTrader:
    """
    Trading strategy based on forecasting implied volatility using Genetic Programming model
    as described in the paper 'Applying Dynamic Training-Subset Selection Methods Using Genetic
    Programming for Forecasting Implied Volatility'
    """
    
    def __init__(self, start_date=None, end_date=None, 
                 risk_free_rate=0.02, trade_threshold=0.02,
                 num_simulated_options=200,
                 holding_period_days=30,
                 reversion_factor=0.5,
                 transaction_cost_pct=0.01):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        start_date : datetime.date
            Start date for simulated data
        end_date : datetime.date
            End date for simulated data
        risk_free_rate : float
            Risk-free interest rate
        trade_threshold : float
            Threshold for volatility difference to trigger trades
        num_simulated_options : int
            Number of options to simulate
        holding_period_days : int
            Number of days to hold options before exiting
        reversion_factor : float
            Fraction of the way volatility is assumed to revert toward true volatility
        transaction_cost_pct : float
            Transaction costs as percentage of option price
        """
        self.start_date = start_date or dt.date.today() - dt.timedelta(days=30)
        self.end_date = end_date or dt.date.today() 
        self.risk_free_rate = risk_free_rate
        self.trade_threshold = trade_threshold
        self.num_simulated_options = num_simulated_options
        self.holding_period_days = holding_period_days
        self.holding_period = holding_period_days / 365.0  # Convert to years
        self.reversion_factor = reversion_factor
        self.transaction_cost_pct = transaction_cost_pct
        self.data = None
        self.trades = []
        self.index_price = 4500.0  # Simulated S&P500 index level
        
    def generate_simulated_options_data(self):
        """Generate simulated options data instead of fetching from Bloomberg"""
        logger.info(f"Generating simulated options data with {self.num_simulated_options} options...")
        
        # Current date for reference
        current_date = dt.datetime.now()
        
        # Parameters for simulation
        min_days_to_expiry = 10
        max_days_to_expiry = 365
        min_moneyness = 0.8    # S/K ratio for deep OTM
        max_moneyness = 1.2    # S/K ratio for deep ITM
        
        # Base volatility and volatility smile/skew parameters
        base_volatility = 0.20  # 20% annual volatility
        skew_factor = 0.15      # How strong the volatility skew is
        term_structure_factor = 0.05  # How strong the term structure is
        
        # Noise level to add to volatilities
        noise_level = 0.05
        
        # Lists to collect option data
        option_data = []
        
        # Generate random options
        for i in range(self.num_simulated_options):
            # Generate random time to expiry (in days)
            dte = random.randint(min_days_to_expiry, max_days_to_expiry)
            expiry_date = current_date + dt.timedelta(days=dte)
            time_to_maturity = dte / 365.0  # In years
            
            # Generate random moneyness
            moneyness = random.uniform(min_moneyness, max_moneyness)
            
            # Calculate strike price based on moneyness
            strike_price = self.index_price / moneyness
            
            # Generate option ticker (simulated)
            ticker = f"SPX_{expiry_date.strftime('%y%m%d')}C{int(strike_price)}"
            
            # Calculate implied volatility with smile/skew effect
            # Further OTM puts (lower strikes) have higher IV (volatility skew)
            # Longer dated options may have different IV (term structure)
            distance_from_atm = abs(1.0 - moneyness)
            skew_adjustment = skew_factor * distance_from_atm * (-1 if moneyness < 1 else 0.5)
            term_adjustment = term_structure_factor * (max(0, time_to_maturity - 0.25))
            
            # True implied volatility with smile and term structure
            true_iv = base_volatility + skew_adjustment + term_adjustment
            
            # Add small random noise to create market inefficiencies
            market_iv = true_iv * (1 + random.uniform(-noise_level, noise_level))
            market_iv = max(0.05, market_iv)  # Ensure reasonable lower bound
            
            # Calculate Black-Scholes option price
            option_price = self.black_scholes_call(
                self.index_price, strike_price, time_to_maturity, 
                self.risk_free_rate, market_iv
            )
            
            # Create option record
            option_data.append({
                'ticker': ticker,
                'strike': strike_price,
                'expiry': expiry_date,
                'time_to_maturity': time_to_maturity,
                'price': option_price,
                'index_price': self.index_price,
                'true_market_iv': market_iv  # Store the true IV for validation
            })
        
        # Convert to DataFrame
        self.data = pd.DataFrame(option_data)
        
        # Filter according to paper's criteria
        # Remove deep ITM and OTM options
        self.data = self.data[
            (self.data['index_price'] / self.data['strike'] >= 0.8) & 
            (self.data['index_price'] / self.data['strike'] <= 1.2)
        ]
        
        # Remove options with less than 10 days to expiry
        self.data = self.data[self.data['time_to_maturity'] >= 0.027]  # ~10 days
        
        # Ensure call prices satisfy arbitrage constraints
        min_call_value = np.maximum(
            0, 
            self.data['index_price'] - self.data['strike'] * np.exp(-self.risk_free_rate * self.data['time_to_maturity'])
        )
        self.data = self.data[self.data['price'] >= min_call_value]
        
        logger.info(f"Generated {len(self.data)} valid option records after filtering")
        
        return self.data
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """
        Calculate Black-Scholes call option price
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to maturity in years
        r : float
            Risk-free interest rate
        sigma : float
            Volatility of the underlying asset
            
        Returns:
        --------
        float:
            Call option price
        """
        if T <= 0 or sigma <= 0:
            # For expired options or invalid volatility, return intrinsic value
            return max(0, S - K)
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def calculate_vega(self, S, K, T, r, sigma):
        """
        Calculate option Vega - sensitivity to volatility changes
        
        Parameters:
        -----------
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to maturity in years
        r : float
            Risk-free interest rate
        sigma : float
            Volatility of the underlying asset
            
        Returns:
        --------
        float:
            Option Vega (change in price for 1 percentage point change in volatility)
        """
        if T <= 0 or sigma <= 0:
            return 0
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        return vega / 100  # Convert to change for 1 percentage point move in volatility

    def calculate_black_scholes_implied_volatility(self, call_price, S, K, T, r):
        """
        Calculate Black-Scholes implied volatility using numerical methods
        
        Parameters:
        -----------
        call_price : float
            Call option price
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to maturity in years
        r : float
            Risk-free interest rate
            
        Returns:
        --------
        float:
            Implied volatility
        """
        def black_scholes_call(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        def objective(sigma):
            return abs(black_scholes_call(sigma) - call_price)
        
        # Check if option price satisfies arbitrage constraints
        intrinsic_value = max(0, S - K * np.exp(-r * T))
        if call_price < intrinsic_value:
            logger.warning(f"Option price {call_price} violates arbitrage constraints (intrinsic value: {intrinsic_value})")
            return np.nan
        
        # Solve for implied volatility
        result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
        
        if result.success:
            return result.x
        else:
            logger.warning(f"Failed to find implied volatility for option with parameters: S={S}, K={K}, T={T}, r={r}, price={call_price}")
            return np.nan

    def calculate_genetic_programming_implied_volatility(self, call_price, S, K, T):
        """
        Calculate implied volatility using the best genetic programming model (MGAR)
        from the paper: GP = (C/K) + (S/K) * [(S/K) - τ]
        
        Parameters:
        -----------
        call_price : float
            Call option price
        S : float
            Current price of the underlying asset
        K : float
            Strike price of the option
        T : float
            Time to maturity in years
            
        Returns:
        --------
        float:
            Implied volatility estimate from genetic programming
        """
        C_K = call_price / K  # Call price / Strike price
        S_K = S / K  # Index price / Strike price
        
        # Calculate GP-based implied volatility using the MGAR formula from the paper
        gp_iv = C_K + S_K * (S_K - T)
        
        # Ensure non-negative implied volatility
        gp_iv = max(0.001, gp_iv)
        
        return gp_iv

    def prepare_trading_signals(self):
        """Calculate model vs market implied volatility and generate trading signals"""
        if self.data is None or len(self.data) == 0:
            logger.warning("No data available for signal generation")
            return None
        
        # Create a copy to avoid modifying original data
        data = self.data.copy()
        
        # Calculate BS implied volatility (market view)
        # For our simulated data, we already know the true IV, but we'll recalculate
        # as if we were working with real market data
        data['bs_iv'] = data.apply(
            lambda row: self.calculate_black_scholes_implied_volatility(
                row['price'], row['index_price'], row['strike'], 
                row['time_to_maturity'], self.risk_free_rate
            ), 
            axis=1
        )
        
        # Calculate GP-based implied volatility (model view)
        data['gp_iv'] = data.apply(
            lambda row: self.calculate_genetic_programming_implied_volatility(
                row['price'], row['index_price'], row['strike'], row['time_to_maturity']
            ),
            axis=1
        )
        
        # Calculate volatility difference and signal
        data['iv_diff'] = data['gp_iv'] - data['bs_iv']
        data['trade_signal'] = 0
        
        # Generate signals based on volatility differences
        # If GP predicts higher volatility than market -> Buy option (market underpricing volatility)
        # If GP predicts lower volatility than market -> Sell option (market overpricing volatility)
        data.loc[data['iv_diff'] > self.trade_threshold, 'trade_signal'] = 1  # Buy signal
        data.loc[data['iv_diff'] < -self.trade_threshold, 'trade_signal'] = -1  # Sell signal
        
        # Filter out options with NaN values
        data = data.dropna(subset=['bs_iv', 'gp_iv'])
        
        logger.info(f"Generated trading signals for {len(data)} options")
        logger.info(f"Buy signals: {len(data[data['trade_signal'] == 1])}")
        logger.info(f"Sell signals: {len(data[data['trade_signal'] == -1])}")
        
        return data

    def execute_trades(self, signals_data):
        """Simulate trade execution based on signals"""
        if signals_data is None or len(signals_data) == 0:
            logger.warning("No signals data available for trade execution")
            return
        
        # Get current time for trade timestamp
        trade_time = dt.datetime.now()
        
        # Process buy signals
        buy_signals = signals_data[signals_data['trade_signal'] == 1]
        if len(buy_signals) > 0:
            logger.info(f"Executing {len(buy_signals)} buy trades")
            for _, row in buy_signals.iterrows():
                trade = {
                    'time': trade_time,
                    'ticker': row['ticker'],
                    'action': 'BUY',
                    'price': row['price'],
                    'strike': row['strike'],
                    'index_price': row['index_price'],
                    'expiry': row['expiry'],
                    'time_to_maturity': row['time_to_maturity'],
                    'market_iv': row['bs_iv'],
                    'model_iv': row['gp_iv'],
                    'true_iv': row['true_market_iv'],
                    'iv_diff': row['iv_diff']
                }
                self.trades.append(trade)
        
        # Process sell signals
        sell_signals = signals_data[signals_data['trade_signal'] == -1]
        if len(sell_signals) > 0:
            logger.info(f"Executing {len(sell_signals)} sell trades")
            for _, row in sell_signals.iterrows():
                trade = {
                    'time': trade_time,
                    'ticker': row['ticker'],
                    'action': 'SELL',
                    'price': row['price'],
                    'strike': row['strike'],
                    'index_price': row['index_price'],
                    'expiry': row['expiry'],
                    'time_to_maturity': row['time_to_maturity'],
                    'market_iv': row['bs_iv'],
                    'model_iv': row['gp_iv'],
                    'true_iv': row['true_market_iv'],
                    'iv_diff': row['iv_diff']
                }
                self.trades.append(trade)
        
        logger.info(f"Total trades executed: {len(self.trades)}")

    def calculate_pnl(self):
        """Calculate PnL for executed trades"""
        if not self.trades:
            logger.warning("No trades to calculate PnL")
            return None
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate PnL for each trade
        pnl_results = []
        
        for _, trade in trades_df.iterrows():
            # Extract trade details
            action = trade['action']
            initial_price = trade['price']
            strike = trade['strike']
            index_price = trade['index_price']
            initial_ttm = trade['time_to_maturity']
            market_iv = trade['market_iv']
            true_iv = trade['true_iv']
            
            # Calculate exit time to maturity
            exit_ttm = max(0, initial_ttm - self.holding_period)
            
            # Assume volatility moves toward the true volatility by the reversion factor
            exit_iv = market_iv + self.reversion_factor * (true_iv - market_iv)
            
            # Calculate exit option price
            exit_price = self.black_scholes_call(
                index_price, strike, exit_ttm, self.risk_free_rate, exit_iv
            )
            
            # Calculate transaction costs
            transaction_cost = initial_price * self.transaction_cost_pct
            
            # Calculate PnL based on trade direction
            if action == 'BUY':
                pnl = exit_price - initial_price - transaction_cost
            else:  # SELL
                pnl = initial_price - exit_price - transaction_cost
                
            # Calculate percentage return
            pct_return = (pnl / initial_price) * 100
            
            # Calculate option vega at entry (sensitivity to volatility changes)
            vega = self.calculate_vega(
                index_price, strike, initial_ttm, self.risk_free_rate, market_iv
            )
            
            # Determine if the trade was a correct prediction
            correct_prediction = (
                (action == 'BUY' and true_iv > market_iv) or 
                (action == 'SELL' and true_iv < market_iv)
            )
            
            # Calculate theo PnL based on vega exposure
            theo_volatility_change = exit_iv - market_iv
            theo_pnl = vega * 100 * theo_volatility_change  # Vega is per 1% change
            
            # Store results
            pnl_results.append({
                'ticker': trade['ticker'],
                'action': action,
                'initial_price': initial_price,
                'exit_price': exit_price,
                'holding_period_years': min(self.holding_period, initial_ttm),
                'initial_iv': market_iv,
                'exit_iv': exit_iv,
                'true_iv': true_iv,
                'iv_diff_at_entry': trade['iv_diff'],
                'iv_change': exit_iv - market_iv,
                'vega': vega,
                'transaction_cost': transaction_cost,
                'pnl': pnl,
                'theo_pnl': theo_pnl,
                'pct_return': pct_return,
                'correct_prediction': correct_prediction
            })
        
        # Convert to DataFrame
        pnl_df = pd.DataFrame(pnl_results)
        
        # Calculate summary statistics
        total_pnl = pnl_df['pnl'].sum()
        avg_pnl_per_trade = pnl_df['pnl'].mean()
        avg_pct_return = pnl_df['pct_return'].mean()
        win_rate = (pnl_df['pnl'] > 0).mean()
        profitable_trades = pnl_df[pnl_df['pnl'] > 0]
        losing_trades = pnl_df[pnl_df['pnl'] < 0]
        
        avg_win = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Calculate profit factor
        gross_profit = profitable_trades['pnl'].sum() if len(profitable_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate prediction accuracy
        prediction_accuracy = pnl_df['correct_prediction'].mean()
        
        # Print summary
        logger.info("\n--- PNL ANALYSIS ---")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Average PnL per trade: ${avg_pnl_per_trade:.2f}")
        logger.info(f"Average percentage return: {avg_pct_return:.2f}%")
        logger.info(f"Win rate: {win_rate:.2%}")
        logger.info(f"Average winning trade: ${avg_win:.2f}")
        logger.info(f"Average losing trade: ${avg_loss:.2f}")
        logger.info(f"Profit factor: {profit_factor:.2f}")
        logger.info(f"Prediction accuracy: {prediction_accuracy:.2%}")
        
        # Calculate PnL by trade direction
        buy_trades = pnl_df[pnl_df['action'] == 'BUY']
        sell_trades = pnl_df[pnl_df['action'] == 'SELL']
        
        buy_pnl = buy_trades['pnl'].sum() if len(buy_trades) > 0 else 0
        sell_pnl = sell_trades['pnl'].sum() if len(sell_trades) > 0 else 0
        
        buy_win_rate = (buy_trades['pnl'] > 0).mean() if len(buy_trades) > 0 else 0
        sell_win_rate = (sell_trades['pnl'] > 0).mean() if len(sell_trades) > 0 else 0
        
        logger.info(f"Buy trades PnL: ${buy_pnl:.2f} (Win rate: {buy_win_rate:.2%})")
        logger.info(f"Sell trades PnL: ${sell_pnl:.2f} (Win rate: {sell_win_rate:.2%})")
        
        # Plot PnL analysis
        self.plot_pnl_analysis(pnl_df)
        
        return pnl_df

    def run_strategy(self):
        """Run the full trading strategy pipeline"""
        logger.info("Starting implied volatility trading strategy")
        
        # Generate simulated options data
        self.generate_simulated_options_data()
        
        if self.data is not None and len(self.data) > 0:
            # Generate trading signals
            signals = self.prepare_trading_signals()
            
            # Execute trades based on signals
            if signals is not None and len(signals) > 0:
                self.execute_trades(signals)
                
                # Create trade summary report
                self.create_trade_report()
                
                # Plot volatility comparisons
                self.plot_volatility_comparison(signals)
                
                # Calculate PnL
                pnl_df = self.calculate_pnl()
                
                # Analyze strategy performance
                self.analyze_performance()
            else:
                logger.warning("No valid trading signals generated")
        else:
            logger.warning("No valid options data retrieved")
        
        logger.info("Strategy execution completed")
    
    def create_trade_report(self):
        """Create a summary report of executed trades"""
        if not self.trades:
            logger.warning("No trades to report")
            return
        
        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate some summary statistics
        buy_count = len(trades_df[trades_df['action'] == 'BUY'])
        sell_count = len(trades_df[trades_df['action'] == 'SELL'])
        avg_iv_diff_buy = trades_df[trades_df['action'] == 'BUY']['iv_diff'].mean()
        avg_iv_diff_sell = trades_df[trades_df['action'] == 'SELL']['iv_diff'].mean()
        
        # Print summary
        logger.info("\n--- TRADE REPORT ---")
        logger.info(f"Total trades: {len(trades_df)}")
        logger.info(f"Buy trades: {buy_count}")
        logger.info(f"Sell trades: {sell_count}")
        logger.info(f"Average IV difference for buy trades: {avg_iv_diff_buy:.4f}")
        logger.info(f"Average IV difference for sell trades: {avg_iv_diff_sell:.4f}")
        
        # Save trades to CSV
        report_filename = f"implied_volatility_trades_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(report_filename, index=False)
        logger.info(f"Trade details saved to {report_filename}")

    def analyze_performance(self):
        """Analyze the performance of the trading strategy"""
        if not self.trades:
            logger.warning("No trades to analyze")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate prediction accuracy - how often our model is closer to true IV than BS IV
        # For buy signals (GP IV > BS IV), we're right if True IV > BS IV
        # For sell signals (GP IV < BS IV), we're right if True IV < BS IV
        
        trades_df['correct_prediction'] = (
            ((trades_df['action'] == 'BUY') & (trades_df['true_iv'] > trades_df['market_iv'])) | 
            ((trades_df['action'] == 'SELL') & (trades_df['true_iv'] < trades_df['market_iv']))
        )
        
        # Overall accuracy
        overall_accuracy = trades_df['correct_prediction'].mean() if len(trades_df) > 0 else 0
        
        # For buy signals
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        if len(buy_trades) > 0:
            buy_accuracy = buy_trades['correct_prediction'].mean()
            
            # Average profit potential (approximated by IV difference)
            avg_buy_potential = (buy_trades['true_iv'] - buy_trades['market_iv']).mean()
        else:
            buy_accuracy = None
            avg_buy_potential = None
        
        # For sell signals
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        if len(sell_trades) > 0:
            sell_accuracy = sell_trades['correct_prediction'].mean()
            
            # Average profit potential (approximated by IV difference)
            avg_sell_potential = (sell_trades['market_iv'] - sell_trades['true_iv']).mean()
        else:
            sell_accuracy = None
            avg_sell_potential = None
        
        # Print analysis
        logger.info("\n--- PERFORMANCE ANALYSIS ---")
        logger.info(f"Overall prediction accuracy: {overall_accuracy:.2%}")
        
        if buy_accuracy is not None:
            logger.info(f"Buy signal accuracy: {buy_accuracy:.2%}")
            logger.info(f"Average buy potential (IV diff): {avg_buy_potential:.4f}")
        
        if sell_accuracy is not None:
            logger.info(f"Sell signal accuracy: {sell_accuracy:.2%}")
            logger.info(f"Average sell potential (IV diff): {avg_sell_potential:.4f}")
        
        # Plot analysis
        self.plot_performance_analysis(trades_df)
    
    def plot_performance_analysis(self, trades_df):
        """Create plots to visualize trading performance"""
        if len(trades_df) == 0:
            logger.warning("No trades available for performance analysis")
            return
        
        # Check for required columns
        required_columns = ['strike', 'true_iv', 'market_iv', 'model_iv']
        missing_columns = [col for col in required_columns if col not in trades_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for analysis: {missing_columns}")
            return
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Accuracy by moneyness (S/K ratio)
        # Make sure both index_price and strike are available
        if 'index_price' in trades_df.columns and 'strike' in trades_df.columns:
            trades_df['moneyness'] = trades_df['index_price'] / trades_df['strike']
            trades_df['correct_prediction'] = (
                ((trades_df['action'] == 'BUY') & (trades_df['true_iv'] > trades_df['market_iv'])) | 
                ((trades_df['action'] == 'SELL') & (trades_df['true_iv'] < trades_df['market_iv']))
            )
            
            # Create moneyness bins
            trades_df['moneyness_bin'] = pd.cut(
                trades_df['moneyness'], 
                bins=[0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.2],
                labels=['0.8-0.9', '0.9-0.95', '0.95-0.98', '0.98-1.02', '1.02-1.05', '1.05-1.1', '1.1-1.2']
            )
            
            # Group by moneyness bin and calculate accuracy
            moneyness_accuracy = trades_df.groupby('moneyness_bin')['correct_prediction'].mean()
            moneyness_counts = trades_df.groupby('moneyness_bin').size()
            
            ax = axes[0, 0]
            moneyness_accuracy.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Prediction Accuracy by Moneyness')
            ax.set_xlabel('Moneyness (S/K)')
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            # Add trade counts as text on bars
            for i, v in enumerate(moneyness_accuracy):
                ax.text(i, v + 0.05, f"n={moneyness_counts.iloc[i]}", ha='center')
        else:
            ax = axes[0, 0]
            ax.text(0.5, 0.5, "Moneyness analysis not available\nMissing required data", 
                    ha='center', va='center')
            ax.set_title('Prediction Accuracy by Moneyness')
        
        # 2. Accuracy by time to maturity
        if 'time_to_maturity' in trades_df.columns:
            trades_df['dte'] = trades_df['time_to_maturity'] * 365
            trades_df['dte_bin'] = pd.cut(
                trades_df['dte'], 
                bins=[0, 30, 60, 90, 180, 365],
                labels=['0-30', '30-60', '60-90', '90-180', '180-365']
            )
            
            # Ensure 'correct_prediction' is present
            if 'correct_prediction' not in trades_df.columns:
                trades_df['correct_prediction'] = (
                    ((trades_df['action'] == 'BUY') & (trades_df['true_iv'] > trades_df['market_iv'])) | 
                    ((trades_df['action'] == 'SELL') & (trades_df['true_iv'] < trades_df['market_iv']))
                )
            
            # Group by DTE bin and calculate accuracy
            dte_accuracy = trades_df.groupby('dte_bin')['correct_prediction'].mean()
            dte_counts = trades_df.groupby('dte_bin').size()
            
            ax = axes[0, 1]
            dte_accuracy.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title('Prediction Accuracy by Days to Expiry')
            ax.set_xlabel('Days to Expiry')
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            # Add trade counts as text on bars
            for i, v in enumerate(dte_accuracy):
                ax.text(i, v + 0.05, f"n={dte_counts.iloc[i]}", ha='center')
        else:
            ax = axes[0, 1]
            ax.text(0.5, 0.5, "Time to maturity analysis not available\nMissing required data", 
                    ha='center', va='center')
            ax.set_title('Prediction Accuracy by Days to Expiry')
        
        # 3. Compare GP IV to True IV and BS IV
        ax = axes[1, 0]
        
        # Create scatter plots
        ax.scatter(trades_df['true_iv'], trades_df['market_iv'], 
                  label='BS IV vs True IV', color='blue', alpha=0.5)
        ax.scatter(trades_df['true_iv'], trades_df['model_iv'], 
                  label='GP IV vs True IV', color='red', alpha=0.5)
        
        # Add reference line
        min_iv = min(trades_df['true_iv'].min(), trades_df['market_iv'].min(), trades_df['model_iv'].min())
        max_iv = max(trades_df['true_iv'].max(), trades_df['market_iv'].max(), trades_df['model_iv'].max())
        ax.plot([min_iv, max_iv], [min_iv, max_iv], 'k--', alpha=0.7)
        
        ax.set_title('Model Comparison to True Volatility')
        ax.set_xlabel('True Implied Volatility')
        ax.set_ylabel('Model Implied Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Error distribution (Model vs BS)
        ax = axes[1, 1]
        
        # Calculate errors
        trades_df['bs_error'] = abs(trades_df['market_iv'] - trades_df['true_iv'])
        trades_df['gp_error'] = abs(trades_df['model_iv'] - trades_df['true_iv'])
        
        # Plot error histograms
        ax.hist(trades_df['bs_error'], bins=20, alpha=0.5, label='BS IV Error', color='blue')
        ax.hist(trades_df['gp_error'], bins=20, alpha=0.5, label='GP IV Error', color='red')
        
        ax.set_title('Error Distribution')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        avg_bs_error = trades_df['bs_error'].mean()
        avg_gp_error = trades_df['gp_error'].mean()
        ax.text(0.7, 0.8, f"Avg BS Error: {avg_bs_error:.4f}\nAvg GP Error: {avg_gp_error:.4f}",
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add a main title
        plt.suptitle('Trading Performance Analysis', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        plot_filename = f"performance_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300)
        logger.info(f"Performance analysis plot saved to {plot_filename}")
        
        # Show the plot
        plt.show()

    def plot_pnl_analysis(self, pnl_df):
        """Create plots to visualize PnL performance"""
        if len(pnl_df) == 0:
            logger.warning("No PnL data available for analysis")
            return
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. PnL distribution
        ax = axes[0, 0]
        ax.hist(pnl_df['pnl'], bins=30, color='skyblue', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_title('PnL Distribution')
        ax.set_xlabel('PnL ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        mean_pnl = pnl_df['pnl'].mean()
        median_pnl = pnl_df['pnl'].median()
        win_rate = (pnl_df['pnl'] > 0).mean()
        ax.text(0.05, 0.95, f"Mean: ${mean_pnl:.2f}\nMedian: ${median_pnl:.2f}\nWin Rate: {win_rate:.2%}",
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7),
                verticalalignment='top')
        
        # 2. PnL by trade direction
        ax = axes[0, 1]
        direction_groups = pnl_df.groupby('action')['pnl'].agg(['mean', 'sum', 'count'])
        colors = ['green' if x > 0 else 'red' for x in direction_groups['sum']]
        direction_groups['sum'].plot(kind='bar', ax=ax, color=colors)
        ax.set_title('Total PnL by Trade Direction')
        ax.set_xlabel('Trade Direction')
        ax.set_ylabel('Total PnL ($)')
        ax.grid(True, alpha=0.3)
        
        # Add count and mean to the bars
        for i, v in enumerate(direction_groups['sum']):
            count = direction_groups['count'].iloc[i]
            mean = direction_groups['mean'].iloc[i]
            ax.text(i, v + (5 if v >= 0 else -15), f"n={count}\nAvg=${mean:.2f}", ha='center')
        
        # 3. PnL vs Volatility Change
        ax = axes[1, 0]
        ax.scatter(pnl_df['iv_change'], pnl_df['pnl'], 
                  c=pnl_df['action'].map({'BUY': 'blue', 'SELL': 'red'}), alpha=0.6)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax.set_title('PnL vs Implied Volatility Change')
        ax.set_xlabel('IV Change (Exit - Entry)')
        ax.set_ylabel('PnL ($)')
        ax.grid(True, alpha=0.3)
        
        # Add a best fit line
        if len(pnl_df) > 1:
            z = np.polyfit(pnl_df['iv_change'], pnl_df['pnl'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(pnl_df['iv_change'].min(), pnl_df['iv_change'].max(), 100)
            ax.plot(x_range, p(x_range), "k--", alpha=0.8)
            
            # Add correlation coefficient
            correlation = np.corrcoef(pnl_df['iv_change'], pnl_df['pnl'])[0, 1]
            ax.text(0.05, 0.95, f"Correlation: {correlation:.2f}",
                    transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7),
                    verticalalignment='top')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Buy'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Sell')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # 4. Accuracy vs PnL
        ax = axes[1, 1]
        
        # Create a new column for accuracy grouping
        pnl_df['correct_group'] = pnl_df['correct_prediction'].map({True: 'Correct Prediction', False: 'Incorrect Prediction'})
        
        # Calculate average PnL by accuracy and trade direction
        grouped_data = pnl_df.groupby(['correct_group', 'action'])['pnl'].mean().unstack()
        
        # Plot grouped bar chart
        grouped_data.plot(kind='bar', ax=ax, color=['green', 'orange'])
        ax.set_title('Average PnL by Prediction Accuracy and Direction')
        ax.set_xlabel('Prediction Accuracy')
        ax.set_ylabel('Average PnL ($)')
        ax.grid(True, alpha=0.3)
        
        # Add trade counts
        counts = pnl_df.groupby(['correct_group', 'action']).size().unstack()
        for i, (_, row) in enumerate(grouped_data.iterrows()):
            for j, v in enumerate(row):
                if not np.isnan(v):
                    action = grouped_data.columns[j]
                    count = counts.iloc[i, j] if i < len(counts) and j < len(counts.columns) else 0
                    ax.text(i - 0.15 + j*0.3, v + (5 if v >= 0 else -15), f"n={count}", ha='center')
        
        # Add a main title
        plt.suptitle('PnL Analysis', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        plot_filename = f"pnl_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300)
        logger.info(f"PnL analysis plot saved to {plot_filename}")
        
        # Show the plot
        plt.show()

    def plot_volatility_comparison(self, signals_data):
        """
        Create plots comparing market implied volatility vs. GP model implied volatility
        
        Parameters:
        -----------
        signals_data : DataFrame
            DataFrame containing volatility and signal data
        """
        if signals_data is None or len(signals_data) == 0:
            logger.warning("No data available for plotting")
            return
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot of BS IV vs GP IV
        ax = axes[0, 0]
        ax.scatter(signals_data['bs_iv'], signals_data['gp_iv'], alpha=0.6)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2)  # Add a 45-degree line
        ax.set_xlabel('Black-Scholes Implied Volatility')
        ax.set_ylabel('GP Model Implied Volatility')
        ax.set_title('Market vs Model Implied Volatility')
        ax.grid(True, alpha=0.3)
        
        # 2. Histogram of volatility differences
        ax = axes[0, 1]
        ax.hist(signals_data['iv_diff'], bins=30, alpha=0.7)
        ax.axvline(x=self.trade_threshold, color='g', linestyle='--', linewidth=2, label='Buy Threshold')
        ax.axvline(x=-self.trade_threshold, color='r', linestyle='--', linewidth=2, label='Sell Threshold')
        ax.set_xlabel('Volatility Difference (GP - BS)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Volatility Differences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. IV difference vs Strike (Volatility Smile/Skew)
        ax = axes[1, 0]
        ax.scatter(signals_data['strike'], signals_data['iv_diff'], alpha=0.6)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Volatility Difference (GP - BS)')
        ax.set_title('Volatility Difference vs Strike Price')
        ax.grid(True, alpha=0.3)
        
        # 4. IV difference vs Time to Maturity (Term Structure)
        ax = axes[1, 1]
        ax.scatter(signals_data['time_to_maturity'] * 365, signals_data['iv_diff'], alpha=0.6)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax.set_xlabel('Time to Maturity (Days)')
        ax.set_ylabel('Volatility Difference (GP - BS)')
        ax.set_title('Volatility Difference vs Time to Maturity')
        ax.grid(True, alpha=0.3)
        
        # Add a main title
        plt.suptitle('Implied Volatility Analysis for S&P500 Options', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the figure
        plot_filename = f"volatility_analysis_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300)
        logger.info(f"Volatility analysis plot saved to {plot_filename}")
        
        # Show the plot
        plt.show()


# Main function to run the strategy
def main():
    # Create and run the strategy with simulated data
    strategy = ImpliedVolatilityTrader(
        start_date=dt.date.today() - dt.timedelta(days=30),
        end_date=dt.date.today() + dt.timedelta(days=90),
        risk_free_rate=0.02,  # 2% risk-free rate
        trade_threshold=0.02,  # 2% difference in implied volatility triggers a trade
        num_simulated_options=500,  # Number of options to simulate
        holding_period_days=30,  # Hold options for 30 days
        reversion_factor=0.5,  # Assume volatility reverts 50% toward true value
        transaction_cost_pct=0.01  # 1% transaction costs
    )
    
    strategy.run_strategy()

if __name__ == "__main__":
    main()