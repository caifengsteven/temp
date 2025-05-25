import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdblp
import datetime as dt
import time
import os
import warnings
warnings.filterwarnings('ignore')

class BasisMomentumStrategy:
    """
    Implementation of the Basis-Momentum strategy from Boons and Porras Prado (2019)
    """
    
    def __init__(self, commodities=None, lookback_period=12):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        commodities : list
            List of Bloomberg tickers for commodity futures
        lookback_period : int
            Lookback period in months for calculating momentum (default: 12)
        """
        self.commodities = commodities if commodities else []
        self.lookback_period = lookback_period
        self.bloomberg = None
        self.data = {}
        self.signals = {}
        self.portfolios = {}
        self.performance = {}
        
    def connect_to_bloomberg(self, timeout=10000):
        """
        Connect to Bloomberg API
        
        Parameters:
        -----------
        timeout : int
            Timeout in milliseconds for Bloomberg API requests
            
        Returns:
        --------
        bool
            True if connection is successful, False otherwise
        """
        try:
            print("Connecting to Bloomberg...")
            self.bloomberg = pdblp.BCon(timeout=timeout)
            self.bloomberg.start()
            print("Connected to Bloomberg successfully")
            return True
        except Exception as e:
            print(f"Failed to connect to Bloomberg: {e}")
            return False
    
    def disconnect_from_bloomberg(self):
        """Disconnect from Bloomberg API"""
        if self.bloomberg:
            try:
                self.bloomberg.stop()
                print("Disconnected from Bloomberg")
            except:
                print("Error while disconnecting from Bloomberg")
    
    def get_historical_data(self, start_date, end_date):
        """
        Retrieve historical futures data from Bloomberg
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format
            
        Returns:
        --------
        dict
            Dictionary with historical data for each commodity
        """
        if not self.bloomberg:
            print("Not connected to Bloomberg")
            return None
        
        if not self.commodities:
            print("No commodities specified")
            return None
        
        print(f"Retrieving historical data from {start_date} to {end_date}...")
        
        # For each commodity, get active contracts data
        for commodity in self.commodities:
            print(f"Getting data for {commodity}...")
            
            try:
                # Create proper Bloomberg ticker for generic futures
                first_nearby = f"{commodity}1 Comdty"  # 1st nearby contract
                second_nearby = f"{commodity}2 Comdty"  # 2nd nearby contract
                
                print(f"  Requesting {first_nearby} and {second_nearby}...")
                
                # Get price data
                futures_data = self.bloomberg.bdh(
                    [first_nearby, second_nearby],
                    ["PX_LAST"],
                    start_date,
                    end_date
                )
                
                if futures_data is not None and not futures_data.empty:
                    print(f"  Data received successfully")
                    
                    # Extract and organize the data
                    self.data[commodity] = {
                        'first_nearby': {
                            'price': futures_data[(first_nearby, 'PX_LAST')]
                        },
                        'second_nearby': {
                            'price': futures_data[(second_nearby, 'PX_LAST')]
                        }
                    }
                    
                    # Convert index to datetime for easier handling
                    self.data[commodity]['first_nearby']['price'].index = pd.to_datetime(self.data[commodity]['first_nearby']['price'].index)
                    self.data[commodity]['second_nearby']['price'].index = pd.to_datetime(self.data[commodity]['second_nearby']['price'].index)
                    
                    print(f"  Got {len(self.data[commodity]['first_nearby']['price'])} data points")
                
                else:
                    print(f"  Warning: No data retrieved for {commodity}")
                    
            except Exception as e:
                print(f"  Error retrieving data for {commodity}: {e}")
        
        # Check if we retrieved any data
        if not any(self.data):
            print("Failed to retrieve any commodity data. Using synthetic data instead.")
            self.generate_synthetic_data(start_date, end_date, len(self.commodities))
            
        return self.data
    
    def calculate_returns(self):
        """
        Calculate monthly returns for each commodity
        
        Returns:
        --------
        dict
            Dictionary with monthly returns for each commodity
        """
        if not self.data:
            print("No data available")
            return None
        
        for commodity in self.data:
            # Get monthly prices (end of month)
            first_nearby_price = self.data[commodity]['first_nearby']['price'].resample('M').last()
            second_nearby_price = self.data[commodity]['second_nearby']['price'].resample('M').last()
            
            # Calculate monthly returns
            self.data[commodity]['returns'] = {
                'first_nearby': first_nearby_price.pct_change(),
                'second_nearby': second_nearby_price.pct_change(),
                'spreading': first_nearby_price.pct_change() - second_nearby_price.pct_change()
            }
            
            # Calculate basis (as defined in the paper)
            self.data[commodity]['basis'] = second_nearby_price / first_nearby_price - 1
        
        return self.data
    
    def calculate_signals(self):
        """
        Calculate basis-momentum signals
        
        Returns:
        --------
        dict
            Dictionary with basis-momentum signals for each commodity
        """
        if not self.data:
            print("No data available")
            return None
        
        for commodity in self.data:
            if 'returns' not in self.data[commodity]:
                print(f"No returns calculated for {commodity}")
                continue
                
            # Get monthly returns
            r1 = self.data[commodity]['returns']['first_nearby']
            r2 = self.data[commodity]['returns']['second_nearby']
            
            # Calculate momentum for first and second nearby
            m1 = r1.rolling(window=self.lookback_period).apply(lambda x: np.prod(1 + x) - 1)
            m2 = r2.rolling(window=self.lookback_period).apply(lambda x: np.prod(1 + x) - 1)
            
            # Calculate basis-momentum (difference between momentum in first and second nearby)
            bm = m1 - m2
            
            # Calculate traditional momentum
            momentum = m1
            
            # Store signals
            self.signals[commodity] = {
                'basis': self.data[commodity]['basis'],
                'momentum': momentum,
                'basis_momentum': bm
            }
        
        return self.signals
    
    def form_portfolios(self, n_top=4, n_bottom=4):
        """
        Form portfolios based on basis-momentum signals
        
        Parameters:
        -----------
        n_top : int
            Number of commodities in top portfolio (High4)
        n_bottom : int
            Number of commodities in bottom portfolio (Low4)
            
        Returns:
        --------
        dict
            Dictionary with portfolio assignments for each month
        """
        if not self.signals:
            print("No signals calculated")
            return None
        
        # Get all dates
        all_dates = set()
        for commodity in self.signals:
            all_dates.update(self.signals[commodity]['basis_momentum'].index)
        all_dates = sorted(all_dates)
        
        # Initialize portfolios
        portfolios = {
            'basis_momentum': {date: {'High4': [], 'Low4': [], 'Mid': []} for date in all_dates},
            'basis': {date: {'High4': [], 'Low4': [], 'Mid': []} for date in all_dates},
            'momentum': {date: {'High4': [], 'Low4': [], 'Mid': []} for date in all_dates}
        }
        
        # For each date, rank commodities and form portfolios
        for date in all_dates:
            # Get signals for this date
            signals_at_date = {
                'basis_momentum': {},
                'basis': {},
                'momentum': {}
            }
            
            for commodity in self.signals:
                try:
                    bm = self.signals[commodity]['basis_momentum'].loc[date]
                    b = self.signals[commodity]['basis'].loc[date]
                    m = self.signals[commodity]['momentum'].loc[date]
                    
                    if not np.isnan(bm):
                        signals_at_date['basis_momentum'][commodity] = bm
                    if not np.isnan(b):
                        signals_at_date['basis'][commodity] = b
                    if not np.isnan(m):
                        signals_at_date['momentum'][commodity] = m
                except:
                    continue
            
            # Form portfolios for each signal type
            for signal_type in signals_at_date:
                if len(signals_at_date[signal_type]) >= n_top + n_bottom:
                    # Sort commodities by signal
                    sorted_commodities = sorted(
                        signals_at_date[signal_type].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Assign to portfolios
                    portfolios[signal_type][date]['High4'] = [x[0] for x in sorted_commodities[:n_top]]
                    portfolios[signal_type][date]['Low4'] = [x[0] for x in sorted_commodities[-n_bottom:]]
                    portfolios[signal_type][date]['Mid'] = [
                        x[0] for x in sorted_commodities[n_top:-n_bottom]
                    ]
        
        self.portfolios = portfolios
        return portfolios
    
    def calculate_portfolio_returns(self):
        """
        Calculate portfolio returns
        
        Returns:
        --------
        dict
            Dictionary with portfolio returns
        """
        if not self.portfolios:
            print("No portfolios formed")
            return None
        
        # Initialize performance
        performance = {
            signal_type: {
                portfolio: {
                    'nearby': pd.Series(dtype=float),
                    'spreading': pd.Series(dtype=float)
                }
                for portfolio in ['High4', 'Mid', 'Low4', 'High4-Low4']
            }
            for signal_type in self.portfolios
        }
        
        # For each date, calculate portfolio returns
        for signal_type in self.portfolios:
            sorted_dates = sorted(self.portfolios[signal_type].keys())
            
            for i in range(len(sorted_dates) - 1):
                current_date = sorted_dates[i]
                next_date = sorted_dates[i + 1]
                
                # Skip if we don't have enough commodities
                if (not self.portfolios[signal_type][current_date]['High4'] or
                    not self.portfolios[signal_type][current_date]['Low4']):
                    continue
                
                # Calculate returns for each portfolio
                for portfolio in ['High4', 'Mid', 'Low4']:
                    nearby_returns = []
                    spreading_returns = []
                    
                    for commodity in self.portfolios[signal_type][current_date][portfolio]:
                        try:
                            # Get returns for next month
                            r1 = self.data[commodity]['returns']['first_nearby'].loc[next_date]
                            spreading = self.data[commodity]['returns']['spreading'].loc[next_date]
                            
                            if not np.isnan(r1):
                                nearby_returns.append(r1)
                            if not np.isnan(spreading):
                                spreading_returns.append(spreading)
                        except:
                            continue
                    
                    # Calculate equal-weighted portfolio returns
                    if nearby_returns:
                        performance[signal_type][portfolio]['nearby'][next_date] = np.mean(nearby_returns)
                    if spreading_returns:
                        performance[signal_type][portfolio]['spreading'][next_date] = np.mean(spreading_returns)
                
                # Calculate High4-Low4 returns
                for return_type in ['nearby', 'spreading']:
                    if (next_date in performance[signal_type]['High4'][return_type].index and 
                        next_date in performance[signal_type]['Low4'][return_type].index):
                        high_return = performance[signal_type]['High4'][return_type][next_date]
                        low_return = performance[signal_type]['Low4'][return_type][next_date]
                        performance[signal_type]['High4-Low4'][return_type][next_date] = high_return - low_return
        
        self.performance = performance
        return performance
    
    def summarize_performance(self):
        """
        Summarize portfolio performance
        
        Returns:
        --------
        dict
            Dictionary with performance statistics
        """
        if not self.performance:
            print("No performance data available")
            return None
        
        summary = {}
        
        for signal_type in self.performance:
            summary[signal_type] = {}
            
            for portfolio in self.performance[signal_type]:
                summary[signal_type][portfolio] = {}
                
                for return_type in self.performance[signal_type][portfolio]:
                    returns = self.performance[signal_type][portfolio][return_type]
                    
                    if len(returns) > 0:
                        # Calculate annualized statistics
                        avg_return = returns.mean() * 12 * 100  # Annualized percentage
                        volatility = returns.std() * np.sqrt(12) * 100  # Annualized percentage
                        sharpe = avg_return / volatility if volatility > 0 else 0
                        t_stat = avg_return / (volatility / np.sqrt(len(returns))) if volatility > 0 else 0
                        
                        summary[signal_type][portfolio][return_type] = {
                            'avg_return': avg_return,
                            'volatility': volatility,
                            'sharpe': sharpe,
                            't_stat': t_stat
                        }
        
        return summary
    
    def plot_performance(self, save_path=None):
        """
        Plot cumulative returns of the basis-momentum strategy
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.performance:
            print("No performance data available")
            return
        
        # Create figure
        plt.figure(figsize=(15, 12))
        
        # Plot nearby returns
        plt.subplot(2, 1, 1)
        for signal_type in ['basis_momentum', 'basis', 'momentum']:
            returns = self.performance[signal_type]['High4-Low4']['nearby']
            if len(returns) > 0:
                cumulative_returns = (1 + returns).cumprod()
                plt.plot(cumulative_returns.index, cumulative_returns, 
                         label=f"{signal_type} (Sharpe: {self.summarize_performance()[signal_type]['High4-Low4']['nearby']['sharpe']:.2f})")
        
        plt.title('Cumulative Nearby Returns of High4-Low4 Portfolios', fontsize=14)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        
        # Plot spreading returns
        plt.subplot(2, 1, 2)
        for signal_type in ['basis_momentum', 'basis', 'momentum']:
            returns = self.performance[signal_type]['High4-Low4']['spreading']
            if len(returns) > 0:
                cumulative_returns = (1 + returns).cumprod()
                plt.plot(cumulative_returns.index, cumulative_returns, 
                         label=f"{signal_type} (Sharpe: {self.summarize_performance()[signal_type]['High4-Low4']['spreading']['sharpe']:.2f})")
        
        plt.title('Cumulative Spreading Returns of High4-Low4 Portfolios', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def run_strategy(self, start_date, end_date, lookback_date=None):
        """
        Run the basis-momentum strategy from start to end date
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format
        lookback_date : str, optional
            Lookback date for calculating initial signals (default: start_date minus lookback_period months)
            
        Returns:
        --------
        dict
            Dictionary with portfolio performance
        """
        # Set lookback date if not provided
        if lookback_date is None:
            start_dt = dt.datetime.strptime(start_date, "%Y%m%d")
            lookback_dt = start_dt - dt.timedelta(days=int(self.lookback_period * 30.5))
            lookback_date = lookback_dt.strftime("%Y%m%d")
        
        # Get historical data
        self.get_historical_data(lookback_date, end_date)
        
        # Calculate returns
        self.calculate_returns()
        
        # Calculate signals
        self.calculate_signals()
        
        # Form portfolios
        self.form_portfolios()
        
        # Calculate portfolio returns
        self.calculate_portfolio_returns()
        
        # Summarize performance
        summary = self.summarize_performance()
        
        # Print performance summary
        print("\nPerformance Summary for High4-Low4 Portfolios:")
        print("------------------------------------------------")
        for signal_type in summary:
            if 'High4-Low4' in summary[signal_type] and 'nearby' in summary[signal_type]['High4-Low4']:
                print(f"\n{signal_type.upper()}:")
                
                # Nearby returns
                nearby = summary[signal_type]['High4-Low4']['nearby']
                print(f"  Nearby Returns: {nearby['avg_return']:.2f}% (t-stat: {nearby['t_stat']:.2f})")
                print(f"  Sharpe Ratio: {nearby['sharpe']:.2f}")
                
                # Spreading returns
                spreading = summary[signal_type]['High4-Low4']['spreading']
                print(f"  Spreading Returns: {spreading['avg_return']:.2f}% (t-stat: {spreading['t_stat']:.2f})")
                print(f"  Sharpe Ratio: {spreading['sharpe']:.2f}")
        
        return summary
    
    def generate_synthetic_data(self, start_date, end_date, n_commodities=21, seed=42):
        """
        Generate synthetic data for testing when Bloomberg is not available
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format
        n_commodities : int
            Number of commodities to generate
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary with synthetic data
        """
        np.random.seed(seed)
        
        print("Generating synthetic commodity data...")
        
        # Create date range
        start_dt = dt.datetime.strptime(start_date, "%Y%m%d")
        end_dt = dt.datetime.strptime(end_date, "%Y%m%d")
        
        # Create daily date range
        dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
        
        # Create commodity tickers if not provided
        if not self.commodities or len(self.commodities) < n_commodities:
            print("Using synthetic commodity identifiers")
            self.commodities = [f"COM{i+1}" for i in range(n_commodities)]
        else:
            # Use the existing commodity list but limit to n_commodities
            self.commodities = self.commodities[:n_commodities]
        
        # Initialize data dictionary
        self.data = {}
        
        # For each commodity, generate price series
        for i, commodity in enumerate(self.commodities):
            # Generate parameters
            # Different seeds for different commodities
            np.random.seed(seed + i)
            
            # Create more realistic parameters based on the paper's findings
            mu = np.random.normal(0.05, 0.02)  # Annual drift
            sigma = np.random.uniform(0.2, 0.4)  # Annual volatility
            mean_reversion = np.random.uniform(0.1, 0.9)  # Mean reversion strength
            
            # Add sector correlation by grouping commodities
            sector = i % 5  # 5 sectors
            sector_drift = np.random.normal(0, 0.01) * (sector + 1)
            
            # Daily parameters
            daily_mu = (mu + sector_drift) / 252
            daily_sigma = sigma / np.sqrt(252)
            
            # Generate log prices
            n_days = len(dates)
            log_prices = np.zeros(n_days)
            log_prices[0] = np.log(100 * (1 + 0.1 * sector))  # Start at sector-dependent value
            
            # Add commodity-specific trend 
            trend = np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.15) / 252
            
            for t in range(1, n_days):
                # Add mean reversion
                reversion = mean_reversion * (log_prices[0] - log_prices[t-1])
                # Add trend
                log_prices[t] = log_prices[t-1] + daily_mu + reversion + trend + daily_sigma * np.random.normal()
            
            # Convert to prices
            prices = np.exp(log_prices)
            
            # Generate first nearby prices
            first_nearby_prices = pd.Series(prices, index=dates)
            
            # Generate basis as mean-reverting process with occasional spikes
            # This better captures the term structure dynamics described in the paper
            basis = np.zeros(n_days)
            basis[0] = np.random.normal(0, 0.02)
            basis_mean = np.random.uniform(-0.03, 0.03)  # Different mean basis for each commodity
            basis_reversion = np.random.uniform(0.05, 0.2)
            basis_vol = np.random.uniform(0.005, 0.02)
            
            for t in range(1, n_days):
                basis[t] = basis[t-1] + basis_reversion * (basis_mean - basis[t-1]) + basis_vol * np.random.normal()
                
                # Occasional spikes (contango/backwardation)
                if np.random.random() < 0.005:  # 0.5% chance of spike
                    basis[t] += np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.10)
            
            # Generate second nearby prices using basis
            second_nearby_prices = first_nearby_prices * (1 + basis)
            
            # Store in data dictionary
            self.data[commodity] = {
                'first_nearby': {
                    'price': first_nearby_prices,
                },
                'second_nearby': {
                    'price': second_nearby_prices,
                }
            }
        
        print(f"Generated synthetic data for {len(self.commodities)} commodities from {dates[0].date()} to {dates[-1].date()}")
        return self.data


def main():
    # Create output directory
    output_dir = "basis_momentum_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define commodity tickers (for Bloomberg)
    # Using major commodities with proper Bloomberg generic ticker format
    commodities = [
        "CL",   # Crude Oil WTI
        "XB",   # RBOB Gasoline 
        "NG",   # Natural Gas
        "HO",   # Heating Oil
        "GC",   # Gold
        "SI",   # Silver
        "HG",   # Copper
        "PL",   # Platinum
        "PA",   # Palladium
        "C ",   # Corn (note the space)
        "W ",   # Wheat (note the space)
        "S ",   # Soybeans (note the space)
        "SM",   # Soybean Meal
        "BO",   # Soybean Oil
        "KC",   # Coffee
        "SB",   # Sugar
        "CT",   # Cotton
        "CC",   # Cocoa
        "LC",   # Live Cattle
        "LH",   # Lean Hogs
        "FC"    # Feeder Cattle
    ]
    
    # Initialize strategy
    strategy = BasisMomentumStrategy(commodities=commodities)
    
    # Set date range
    end_date = dt.datetime.now().strftime("%Y%m%d")
    # Go back 5 years for testing
    start_date = (dt.datetime.now() - dt.timedelta(days=5*365)).strftime("%Y%m%d")
    # Add lookback period for calculating initial signals
    lookback_date = (dt.datetime.now() - dt.timedelta(days=(5+1)*365)).strftime("%Y%m%d")
    
    print(f"Analysis period: {start_date} to {end_date}")
    print(f"Lookback period for initial signals: {lookback_date}")
    
    # Try to connect to Bloomberg
    use_synthetic_data = False
    if not strategy.connect_to_bloomberg():
        print("Using synthetic data since Bloomberg connection failed")
        use_synthetic_data = True
    
    try:
        # Run strategy
        if use_synthetic_data:
            # Generate synthetic data
            strategy.generate_synthetic_data(lookback_date, end_date)
        else:
            # Get data from Bloomberg
            strategy.get_historical_data(lookback_date, end_date)
            
            # Check if we got any data, if not use synthetic
            if not strategy.data:
                print("No data retrieved from Bloomberg. Using synthetic data.")
                strategy.generate_synthetic_data(lookback_date, end_date)
        
        # Calculate returns
        strategy.calculate_returns()
        
        # Calculate signals
        strategy.calculate_signals()
        
        # Form portfolios
        strategy.form_portfolios()
        
        # Calculate portfolio returns
        strategy.calculate_portfolio_returns()
        
        # Summarize performance
        summary = strategy.summarize_performance()
        
        # Print detailed information about the portfolios and returns
        print("\nPortfolio Composition Analysis:")
        print("------------------------------")
        
        # Count the number of months for each commodity in high and low portfolios
        commodity_counts = {commodity: {'High4': 0, 'Low4': 0} for commodity in strategy.commodities}
        
        # Get sorted dates for basis-momentum
        sorted_dates = sorted(strategy.portfolios['basis_momentum'].keys())
        
        if sorted_dates:
            for date in sorted_dates:
                for commodity in strategy.portfolios['basis_momentum'][date]['High4']:
                    commodity_counts[commodity]['High4'] += 1
                for commodity in strategy.portfolios['basis_momentum'][date]['Low4']:
                    commodity_counts[commodity]['Low4'] += 1
            
            # Print top 5 commodities by frequency in High4 and Low4
            high_top5 = sorted(commodity_counts.items(), key=lambda x: x[1]['High4'], reverse=True)[:5]
            low_top5 = sorted(commodity_counts.items(), key=lambda x: x[1]['Low4'], reverse=True)[:5]
            
            print("\nTop 5 commodities in High4 basis-momentum portfolio:")
            for commodity, counts in high_top5:
                if counts['High4'] > 0:
                    print(f"  {commodity}: {counts['High4']} months ({counts['High4']/len(sorted_dates)*100:.1f}%)")
            
            print("\nTop 5 commodities in Low4 basis-momentum portfolio:")
            for commodity, counts in low_top5:
                if counts['Low4'] > 0:
                    print(f"  {commodity}: {counts['Low4']} months ({counts['Low4']/len(sorted_dates)*100:.1f}%)")
        
        # Plot performance
        strategy.plot_performance(save_path=os.path.join(output_dir, "performance.png"))
        
        # Save detailed results
        # Convert summary to DataFrame
        summary_data = []
        for signal_type in summary:
            for portfolio in summary[signal_type]:
                for return_type in summary[signal_type][portfolio]:
                    stats = summary[signal_type][portfolio][return_type]
                    summary_data.append({
                        'Signal': signal_type,
                        'Portfolio': portfolio,
                        'Return_Type': return_type,
                        'Avg_Return': stats['avg_return'],
                        'Volatility': stats['volatility'],
                        'Sharpe': stats['sharpe'],
                        'T_Stat': stats['t_stat']
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, "strategy_summary.csv"), index=False)
        
        # Save monthly returns
        for signal_type in strategy.performance:
            # Nearby returns
            nearby_returns = pd.DataFrame({
                portfolio: strategy.performance[signal_type][portfolio]['nearby']
                for portfolio in ['High4', 'Mid', 'Low4', 'High4-Low4']
            })
            nearby_returns.to_csv(os.path.join(output_dir, f"{signal_type}_nearby_returns.csv"))
            
            # Spreading returns
            spreading_returns = pd.DataFrame({
                portfolio: strategy.performance[signal_type][portfolio]['spreading']
                for portfolio in ['High4', 'Mid', 'Low4', 'High4-Low4']
            })
            spreading_returns.to_csv(os.path.join(output_dir, f"{signal_type}_spreading_returns.csv"))
        
        print(f"\nResults saved to {output_dir}")
        
    finally:
        # Disconnect from Bloomberg
        if not use_synthetic_data:
            strategy.disconnect_from_bloomberg()

if __name__ == "__main__":
    main()