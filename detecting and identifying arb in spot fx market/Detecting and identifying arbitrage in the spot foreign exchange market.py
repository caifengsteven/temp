import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdblp
from scipy import linalg
import datetime as dt
import time
import os
import warnings
warnings.filterwarnings('ignore')

class TriangularArbitrageDetector:
    """
    Class to detect and identify triangular arbitrage opportunities in the FX market
    based on the paper by Cui et al. (2020)
    """
    
    def __init__(self, currencies=None, transaction_cost=0.0002):
        """
        Initialize the detector with a list of currencies and transaction cost
        
        Parameters:
        -----------
        currencies : list
            List of currency codes (e.g., ['USD', 'EUR', 'GBP', ...])
        transaction_cost : float
            Transaction cost as a fraction (e.g., 0.0002 for 2 basis points)
        """
        self.currencies = currencies if currencies else []
        self.transaction_cost = transaction_cost
        self.bloomberg = None
        self.rate_matrix = None
        self.bid_matrix = None
        self.ask_matrix = None
        
    def connect_to_bloomberg(self, timeout=5000):
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
    
    def set_currencies(self, currencies):
        """
        Set the currencies to monitor
        
        Parameters:
        -----------
        currencies : list
            List of currency codes (e.g., ['USD', 'EUR', 'GBP', ...])
        """
        self.currencies = currencies
    
    def get_fx_rates_from_bloomberg(self, reference_currency='USD'):
        """
        Get FX rates from Bloomberg using ref() method
        
        Parameters:
        -----------
        reference_currency : str
            Reference currency for FX rates
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with bid and ask rates for all currency pairs
        """
        if not self.bloomberg:
            print("Not connected to Bloomberg")
            return None
        
        if not self.currencies:
            print("No currencies specified")
            return None
        
        # Create currency pairs
        pairs = []
        for base in self.currencies:
            for quote in self.currencies:
                if base != quote:
                    pairs.append(f"{base}{quote} Curncy")
        
        try:
            # Get bid and ask prices using ref method
            bid_data = self.bloomberg.ref(pairs, "PX_BID")
            ask_data = self.bloomberg.ref(pairs, "PX_ASK")
            
            # Combine bid and ask data
            fx_data = pd.DataFrame(index=pairs, columns=["PX_BID", "PX_ASK"])
            
            if bid_data is not None and not bid_data.empty:
                for ticker in bid_data.index:
                    fx_data.loc[ticker, "PX_BID"] = bid_data.loc[ticker][0]
                    
            if ask_data is not None and not ask_data.empty:
                for ticker in ask_data.index:
                    fx_data.loc[ticker, "PX_ASK"] = ask_data.loc[ticker][0]
            
            return fx_data.dropna()
            
        except Exception as e:
            print(f"Error retrieving FX rates: {e}")
            return None
    
    def construct_rate_matrices(self, fx_data):
        """
        Construct bid and ask matrices from FX data
        
        Parameters:
        -----------
        fx_data : pd.DataFrame
            DataFrame with bid and ask rates for all currency pairs
            
        Returns:
        --------
        tuple
            (bid_matrix, ask_matrix)
        """
        n = len(self.currencies)
        bid_matrix = np.ones((n, n))
        ask_matrix = np.ones((n, n))
        
        for i, base in enumerate(self.currencies):
            for j, quote in enumerate(self.currencies):
                if base == quote:
                    continue
                
                pair = f"{base}{quote} Curncy"
                if pair in fx_data.index:
                    bid_matrix[i, j] = fx_data.loc[pair, "PX_BID"]
                    ask_matrix[i, j] = fx_data.loc[pair, "PX_ASK"]
                else:
                    # Use reciprocal if direct rate is not available
                    inverse_pair = f"{quote}{base} Curncy"
                    if inverse_pair in fx_data.index:
                        bid_matrix[i, j] = 1 / fx_data.loc[inverse_pair, "PX_ASK"]
                        ask_matrix[i, j] = 1 / fx_data.loc[inverse_pair, "PX_BID"]
        
        # Ensure diagonal elements are 1
        np.fill_diagonal(bid_matrix, 1)
        np.fill_diagonal(ask_matrix, 1)
        
        self.bid_matrix = bid_matrix
        self.ask_matrix = ask_matrix
        
        # Use mid price for rate matrix
        self.rate_matrix = (bid_matrix + ask_matrix) / 2
        
        return bid_matrix, ask_matrix
    
    def compute_max_eigenvalue(self, matrix=None):
        """
        Compute the maximum eigenvalue of a matrix
        
        Parameters:
        -----------
        matrix : numpy.ndarray, optional
            The matrix to compute the eigenvalue for.
            If None, use self.rate_matrix
            
        Returns:
        --------
        float
            Maximum eigenvalue
        """
        if matrix is None:
            matrix = self.rate_matrix
            
        if matrix is None:
            print("No rate matrix available")
            return None
        
        # Compute eigenvalues
        eigenvalues = linalg.eigvals(matrix)
        
        # Return the eigenvalue with the largest real part
        max_eigenvalue = max(eigenvalues, key=lambda x: abs(x.real))
        return max_eigenvalue.real
    
    def detect_arbitrage_eigenvalue(self, epsilon=None):
        """
        Detect arbitrage opportunities using the eigenvalue method
        
        Parameters:
        -----------
        epsilon : float, optional
            Epsilon threshold for arbitrage detection.
            If None, use transaction_cost
            
        Returns:
        --------
        tuple
            (arbitrage_exists, lambda_max, threshold)
        """
        if self.rate_matrix is None:
            print("No rate matrix available")
            return False, None, None
        
        # Set epsilon to transaction cost if not provided
        if epsilon is None:
            epsilon = self.transaction_cost
            
        n = len(self.currencies)
        lambda_max = self.compute_max_eigenvalue()
        threshold = n * np.exp(7/3 * epsilon)
        
        # As per Theorem 10 in the paper
        arbitrage_exists = lambda_max > threshold
        
        return arbitrage_exists, lambda_max, threshold
    
    def detect_triangular_arbitrage_computational(self):
        """
        Detect triangular arbitrage opportunities using the computational method
        
        Returns:
        --------
        list
            List of arbitrage opportunities with details
        """
        if self.bid_matrix is None or self.ask_matrix is None:
            print("No bid/ask matrices available")
            return []
        
        n = len(self.currencies)
        opportunities = []
        
        # Modified bid and ask matrices as per Appendix A.1
        A_prime = np.minimum(self.ask_matrix, 1/np.transpose(self.bid_matrix))
        B_prime = np.maximum(self.bid_matrix, 1/np.transpose(self.ask_matrix))
        
        # Check all possible triplets
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                    
                for k in range(n):
                    if i == k or j == k:
                        continue
                    
                    # Check buy path: i->j->k->i
                    buy_path_profit = 1/(A_prime[i,j] * A_prime[j,k] * A_prime[k,i])
                    buy_path_profit_with_cost = buy_path_profit * (1 - self.transaction_cost)**3
                    
                    # Check sell path: i->j->k->i
                    sell_path_profit = B_prime[i,j] * B_prime[j,k] * B_prime[k,i]
                    sell_path_profit_with_cost = sell_path_profit * (1 - self.transaction_cost)**3
                    
                    # Check if arbitrage exists
                    if buy_path_profit_with_cost > 1:
                        opportunities.append({
                            'type': 'buy',
                            'path': [self.currencies[i], self.currencies[j], self.currencies[k], self.currencies[i]],
                            'profit': buy_path_profit,
                            'profit_with_cost': buy_path_profit_with_cost,
                            'profit_percentage': (buy_path_profit_with_cost - 1) * 100
                        })
                    
                    if sell_path_profit_with_cost > 1:
                        opportunities.append({
                            'type': 'sell',
                            'path': [self.currencies[i], self.currencies[j], self.currencies[k], self.currencies[i]],
                            'profit': sell_path_profit,
                            'profit_with_cost': sell_path_profit_with_cost,
                            'profit_percentage': (sell_path_profit_with_cost - 1) * 100
                        })
        
        # Sort opportunities by profit
        opportunities.sort(key=lambda x: x['profit_with_cost'], reverse=True)
        
        return opportunities
    
    def monitor_real_time(self, interval=60, max_iterations=None):
        """
        Monitor FX market for arbitrage opportunities in real time
        
        Parameters:
        -----------
        interval : int
            Interval between checks in seconds
        max_iterations : int, optional
            Maximum number of iterations to run
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with historical arbitrage opportunities
        """
        if not self.bloomberg:
            print("Not connected to Bloomberg")
            return None
        
        if not self.currencies:
            print("No currencies specified")
            return None
        
        results = []
        iterations = 0
        
        try:
            while True:
                # Check if max_iterations is reached
                if max_iterations is not None and iterations >= max_iterations:
                    break
                
                # Get current timestamp
                timestamp = dt.datetime.now()
                
                # Get FX rates
                fx_data = self.get_fx_rates_from_bloomberg()
                
                if fx_data is not None and not fx_data.empty:
                    # Construct rate matrices
                    self.construct_rate_matrices(fx_data)
                    
                    # Detect arbitrage using eigenvalue method
                    arb_exists, lambda_max, threshold = self.detect_arbitrage_eigenvalue()
                    
                    # If arbitrage exists, identify opportunities
                    arb_opportunities = []
                    if arb_exists:
                        arb_opportunities = self.detect_triangular_arbitrage_computational()
                    
                    # Record results
                    results.append({
                        'timestamp': timestamp,
                        'lambda_max': lambda_max,
                        'threshold': threshold,
                        'arbitrage_exists': arb_exists,
                        'opportunities': arb_opportunities
                    })
                    
                    # Print current status
                    print(f"[{timestamp}] λ_max = {lambda_max:.6f}, threshold = {threshold:.6f}")
                    
                    if arb_exists:
                        print(f"  Arbitrage detected! {len(arb_opportunities)} opportunities found.")
                        for i, opp in enumerate(arb_opportunities[:3]):  # Show top 3
                            path_str = " -> ".join(opp['path'])
                            print(f"  {i+1}. {path_str}: {opp['profit_percentage']:.4f}% profit")
                    else:
                        print("  No arbitrage opportunities detected.")
                else:
                    print(f"[{timestamp}] Could not retrieve FX data")
                
                # Increment iterations
                iterations += 1
                
                # Wait for next check
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        
        # Convert results to DataFrame
        if results:
            results_df = pd.DataFrame([
                {
                    'timestamp': r['timestamp'],
                    'lambda_max': r['lambda_max'],
                    'threshold': r['threshold'],
                    'arbitrage_exists': r['arbitrage_exists'],
                    'opportunities_count': len(r['opportunities']),
                    'max_profit_percentage': max([o['profit_percentage'] for o in r['opportunities']], default=0)
                }
                for r in results
            ])
        else:
            results_df = pd.DataFrame()
        
        return results_df, results
    
    def visualize_results(self, results_df, save_path=None):
        """
        Visualize results from monitoring
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with historical arbitrage opportunities
        save_path : str, optional
            Path to save the plot
        """
        if results_df.empty:
            print("No results to visualize")
            return
            
        plt.figure(figsize=(14, 10))
        
        # Plot 1: λ_max and threshold
        plt.subplot(2, 1, 1)
        plt.plot(results_df['timestamp'], results_df['lambda_max'], 'b-', label='λ_max')
        plt.plot(results_df['timestamp'], results_df['threshold'], 'r--', label='Threshold')
        plt.title('Maximum Eigenvalue and Threshold')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Maximum profit percentage
        plt.subplot(2, 1, 2)
        plt.bar(results_df['timestamp'], results_df['max_profit_percentage'], 
                color='g', alpha=0.7, label='Max Profit %')
        plt.title('Maximum Arbitrage Profit Percentage')
        plt.xlabel('Time')
        plt.ylabel('Profit %')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def run_historical_test(self, start_date, end_date, interval='1 min'):
        """
        Run a historical test using Bloomberg historical data
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format
        interval : str
            Interval for historical data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with historical arbitrage opportunities
        """
        if not self.bloomberg:
            print("Not connected to Bloomberg")
            return None, None
        
        if not self.currencies:
            print("No currencies specified")
            return None, None
        
        # Create currency pairs
        pairs = []
        for base in self.currencies:
            for quote in self.currencies:
                if base != quote:
                    pairs.append(f"{base}{quote} Curncy")
        
        try:
            # Get historical data using bdh method instead of bdib
            print(f"Fetching historical data from {start_date} to {end_date}...")
            historical_data = {}
            
            for pair in pairs:
                try:
                    # Get bid data
                    bid_data = self.bloomberg.bdh(
                        pair,
                        'PX_BID',
                        start_date,
                        end_date
                    )
                    
                    # Get ask data
                    ask_data = self.bloomberg.bdh(
                        pair,
                        'PX_ASK',
                        start_date,
                        end_date
                    )
                    
                    if bid_data is not None and not bid_data.empty and ask_data is not None and not ask_data.empty:
                        # Combine bid and ask data
                        combined_data = pd.DataFrame(index=bid_data.index)
                        combined_data['px_bid'] = bid_data[pair]['PX_BID']
                        combined_data['px_ask'] = ask_data[pair]['PX_ASK']
                        
                        historical_data[pair] = combined_data
                except Exception as e:
                    print(f"Error retrieving data for {pair}: {e}")
            
            if not historical_data:
                print("No historical data retrieved")
                return None, None
            
            # Process each date
            results = []
            unique_dates = set()
            
            for pair, data in historical_data.items():
                unique_dates.update(data.index)
            
            unique_dates = sorted(list(unique_dates))
            
            for date in unique_dates:
                # Get data for this date
                fx_data = pd.DataFrame(index=pairs, columns=["PX_BID", "PX_ASK"])
                
                for pair in pairs:
                    if pair in historical_data and date in historical_data[pair].index:
                        fx_data.loc[pair, "PX_BID"] = historical_data[pair].loc[date, "px_bid"]
                        fx_data.loc[pair, "PX_ASK"] = historical_data[pair].loc[date, "px_ask"]
                
                # Drop rows with missing data
                fx_data = fx_data.dropna()
                
                if not fx_data.empty:
                    # Construct rate matrices
                    self.construct_rate_matrices(fx_data)
                    
                    # Detect arbitrage using eigenvalue method
                    arb_exists, lambda_max, threshold = self.detect_arbitrage_eigenvalue()
                    
                    # If arbitrage exists, identify opportunities
                    arb_opportunities = []
                    if arb_exists:
                        arb_opportunities = self.detect_triangular_arbitrage_computational()
                    
                    # Record results
                    results.append({
                        'timestamp': date,
                        'lambda_max': lambda_max,
                        'threshold': threshold,
                        'arbitrage_exists': arb_exists,
                        'opportunities': arb_opportunities
                    })
            
            # Convert results to DataFrame
            if results:
                results_df = pd.DataFrame([
                    {
                        'timestamp': r['timestamp'],
                        'lambda_max': r['lambda_max'],
                        'threshold': r['threshold'],
                        'arbitrage_exists': r['arbitrage_exists'],
                        'opportunities_count': len(r['opportunities']),
                        'max_profit_percentage': max([o['profit_percentage'] for o in r['opportunities']], default=0)
                    }
                    for r in results
                ])
            else:
                results_df = pd.DataFrame()
            
            return results_df, results
            
        except Exception as e:
            print(f"Error running historical test: {e}")
            return None, None

    def create_synthetic_data(self):
        """
        Create synthetic data for testing when Bloomberg data is not available
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with synthetic bid and ask rates
        """
        n = len(self.currencies)
        
        # Generate base exchange rates (all against USD)
        base_rates = {
            'EUR': 1.1, 'GBP': 1.3, 'JPY': 0.009, 'CHF': 1.05, 
            'AUD': 0.7, 'CAD': 0.75, 'NZD': 0.65, 'SEK': 0.1, 'NOK': 0.095
        }
        
        # Initialize DataFrame
        fx_data = pd.DataFrame(index=[], columns=["PX_BID", "PX_ASK"])
        
        # Generate cross rates
        for i, base in enumerate(self.currencies):
            for j, quote in enumerate(self.currencies):
                if base != quote:
                    pair = f"{base}{quote} Curncy"
                    
                    # Calculate the mid rate
                    if base == 'USD':
                        mid_rate = 1.0 / base_rates[quote]
                    elif quote == 'USD':
                        mid_rate = base_rates[base]
                    else:
                        mid_rate = base_rates[base] / base_rates[quote]
                    
                    # Add small random noise to create bid-ask spread
                    spread = mid_rate * 0.0005  # 0.5 bp spread
                    
                    # Set bid and ask
                    fx_data.loc[pair, "PX_BID"] = mid_rate - spread/2
                    fx_data.loc[pair, "PX_ASK"] = mid_rate + spread/2
                    
                    # Introduce arbitrage opportunity in some cases (5% chance)
                    if np.random.random() < 0.05:
                        # Increase bid or decrease ask to create arbitrage
                        if np.random.random() < 0.5:
                            fx_data.loc[pair, "PX_BID"] *= (1 + 0.001)  # Increase bid by 0.1%
                        else:
                            fx_data.loc[pair, "PX_ASK"] *= (1 - 0.001)  # Decrease ask by 0.1%
        
        return fx_data


def main():
    # Create output directory
    output_dir = "arbitrage_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector with G10 currencies
    g10_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'NZD', 'SEK', 'NOK']
    detector = TriangularArbitrageDetector(currencies=g10_currencies, transaction_cost=0.0002)
    
    # Connect to Bloomberg
    if not detector.connect_to_bloomberg():
        print("Failed to connect to Bloomberg. Exiting.")
        return
    
    try:
        # Option 1: Run real-time monitoring with Bloomberg data
        print("\nStarting real-time monitoring...")
        results_df, results = detector.monitor_real_time(interval=60, max_iterations=5)
        
        if results_df is not None and not results_df.empty:
            # Save results
            results_df.to_csv(os.path.join(output_dir, "real_time_results.csv"), index=False)
            
            # Visualize results
            detector.visualize_results(results_df, save_path=os.path.join(output_dir, "real_time_plot.png"))
        
        # Option 2: Demonstrate with synthetic data if Bloomberg data is not available or insufficient
        if results_df is None or results_df.empty:
            print("\nUsing synthetic data for demonstration...")
            
            # Generate synthetic results
            synthetic_results = []
            
            for i in range(10):
                timestamp = dt.datetime.now() - dt.timedelta(minutes=i*15)
                
                # Generate synthetic FX data
                fx_data = detector.create_synthetic_data()
                
                # Construct rate matrices
                detector.construct_rate_matrices(fx_data)
                
                # Detect arbitrage using eigenvalue method
                arb_exists, lambda_max, threshold = detector.detect_arbitrage_eigenvalue()
                
                # If arbitrage exists, identify opportunities
                arb_opportunities = []
                if arb_exists:
                    arb_opportunities = detector.detect_triangular_arbitrage_computational()
                
                # Record results
                synthetic_results.append({
                    'timestamp': timestamp,
                    'lambda_max': lambda_max,
                    'threshold': threshold,
                    'arbitrage_exists': arb_exists,
                    'opportunities': arb_opportunities
                })
            
            # Convert synthetic results to DataFrame
            synthetic_df = pd.DataFrame([
                {
                    'timestamp': r['timestamp'],
                    'lambda_max': r['lambda_max'],
                    'threshold': r['threshold'],
                    'arbitrage_exists': r['arbitrage_exists'],
                    'opportunities_count': len(r['opportunities']),
                    'max_profit_percentage': max([o['profit_percentage'] for o in r['opportunities']], default=0)
                }
                for r in synthetic_results
            ])
            
            # Save synthetic results
            synthetic_df.to_csv(os.path.join(output_dir, "synthetic_results.csv"), index=False)
            
            # Visualize synthetic results
            detector.visualize_results(synthetic_df, save_path=os.path.join(output_dir, "synthetic_plot.png"))
            
            # Print summary of synthetic results
            print("\nSynthetic data summary:")
            print(f"Total periods analyzed: {len(synthetic_df)}")
            print(f"Periods with arbitrage: {synthetic_df['arbitrage_exists'].sum()}")
            
            if synthetic_df['arbitrage_exists'].sum() > 0:
                print(f"Maximum potential profit: {synthetic_df['max_profit_percentage'].max():.4f}%")
                print(f"Average potential profit when arbitrage exists: {synthetic_df.loc[synthetic_df['arbitrage_exists'], 'max_profit_percentage'].mean():.4f}%")
            
            # Print some sample arbitrage opportunities
            for i, r in enumerate(synthetic_results):
                if r['arbitrage_exists'] and r['opportunities']:
                    print(f"\nArbitrage opportunities at {r['timestamp']}:")
                    for j, opp in enumerate(r['opportunities'][:3]):  # Show top 3
                        path_str = " -> ".join(opp['path'])
                        print(f"  {j+1}. {path_str}: {opp['profit_percentage']:.4f}% profit")
                    break
        
    finally:
        # Disconnect from Bloomberg
        detector.disconnect_from_bloomberg()
    
    print("\nAnalysis complete. Results saved to", output_dir)

if __name__ == "__main__":
    main()