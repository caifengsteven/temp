import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings
warnings.filterwarnings('ignore')

# Strategy Parameters
LOOKBACK_WINDOW = 63      # Quarter (~63 trading days) to calculate exponential beta
BETA_HALFLIFE = 63        # Half-life for exponential weighting
BETA_THRESHOLD = 0.1      # Bottom 10% of constituents by beta
VOLUME_ZSCORE = 3         # 3 standard deviations for volume spike
HOLDING_PERIOD = 40       # Hold positions for 40 days
TRANSACTION_COST = 0.0017  # 17 bps transaction cost (with leverage)
BORROWING_COST = 0.0010   # 10 bps borrowing cost for 40 days

class ETFOutsiderStrategy:
    def __init__(self, etf_data_path=None, constituent_data_path=None):
        """
        Initialize strategy with data paths or load data directly if provided
        
        In a real implementation, these would be loaded from Bloomberg API
        """
        self.etf_data = None
        self.constituent_data = None
        self.constituents_by_date = {}
        self.results = {}
        
        if etf_data_path:
            self.load_etf_data(etf_data_path)
        if constituent_data_path:
            self.load_constituent_data(constituent_data_path)
    
    def load_bloomberg_data(self, etf_symbols, start_date, end_date):
        """
        Load Bloomberg data for ETFs and constituents
        
        In a real implementation, you would use pdblp or similar package:
        
        import pdblp
        con = pdblp.BCon(timeout=5000)
        con.start()
        
        # Get ETF data
        etf_data = con.bdh(etf_symbols, 
                          ['PX_LAST', 'PX_VOLUME'], 
                          start_date, 
                          end_date)
        
        # Get ETF constituent data (requires special Bloomberg requests)
        # This is more complex and would require using MBRS<GO> function or similar
        """
        print("This would fetch ETF and constituent data from Bloomberg in a real implementation")
        print("Proceeding with sample data for demonstration purposes")
    
    def load_etf_data(self, etf_data_path):
        """Load ETF price and volume data"""
        self.etf_data = pd.read_csv(etf_data_path, index_col='Date', parse_dates=True)
    
    def load_constituent_data(self, constituent_data_path):
        """Load constituent price data and membership information"""
        self.constituent_data = pd.read_csv(constituent_data_path, parse_dates=['Date'])
    
    def calculate_volume_spikes(self, etf_symbol, lookback_days=252):
        """
        Identify days with abnormal trading volume (z-score >= threshold)
        """
        if self.etf_data is None:
            raise ValueError("ETF data not loaded. Please load data first.")
        
        # Extract volume data for the specific ETF
        volume = self.etf_data[f'{etf_symbol}_Volume'].copy()
        
        # Calculate rolling mean and standard deviation
        volume_mean = volume.rolling(window=lookback_days).mean()
        volume_std = volume.rolling(window=lookback_days).std()
        
        # Calculate z-score
        volume_zscore = (volume - volume_mean) / volume_std
        
        # Identify spike days (z-score >= threshold and negative returns)
        price = self.etf_data[f'{etf_symbol}_Close']
        returns = price.pct_change()
        
        spike_days = (volume_zscore >= VOLUME_ZSCORE) & (returns < 0)
        spike_dates = volume_zscore[spike_days].index.tolist()
        
        return spike_dates, volume_zscore
    
    def calculate_exponential_betas(self, etf_symbol, stocks, dates):
        """
        Calculate exponentially weighted betas of stocks to the ETF
        """
        if self.constituent_data is None:
            raise ValueError("Constituent data not loaded. Please load data first.")
        
        # Prepare data for beta calculation
        etf_returns = self.etf_data[f'{etf_symbol}_Close'].pct_change()
        stock_data = self.constituent_data[self.constituent_data['Ticker'].isin(stocks)]
        
        betas = {}
        
        for date in dates:
            end_date = date
            start_date = end_date - pd.Timedelta(days=LOOKBACK_WINDOW*1.5)  # Add buffer for trading days
            
            # Filter data for the lookback period
            period_etf_returns = etf_returns.loc[start_date:end_date].dropna()
            
            if len(period_etf_returns) < 30:  # Need enough data points
                continue
                
            date_betas = {}
            
            for stock in stocks:
                stock_prices = stock_data[stock_data['Ticker'] == stock]['Close']
                stock_prices = stock_prices.set_index(stock_data[stock_data['Ticker'] == stock]['Date'])
                stock_prices = stock_prices.loc[start_date:end_date]
                
                if len(stock_prices) < 30:  # Need enough data points
                    continue
                    
                stock_returns = stock_prices.pct_change().dropna()
                
                # Align the data
                aligned_data = pd.concat([period_etf_returns, stock_returns], axis=1).dropna()
                
                if len(aligned_data) < 30:
                    continue
                
                # Calculate exponentially weighted beta
                weights = np.exp(-np.arange(len(aligned_data)) * np.log(2) / BETA_HALFLIFE)
                weights = weights / weights.sum()
                
                X = sm.add_constant(aligned_data.iloc[:, 0])
                model = sm.WLS(aligned_data.iloc[:, 1], X, weights=weights).fit()
                beta = model.params[1]
                
                date_betas[stock] = beta
            
            betas[date] = date_betas
        
        return betas
    
    def identify_outsiders(self, etf_symbol, spike_date, betas):
        """
        Identify outsider stocks (low beta to ETF) on spike days
        """
        if spike_date not in betas:
            return []
        
        date_betas = betas[spike_date]
        
        # Sort stocks by beta
        sorted_stocks = sorted(date_betas.items(), key=lambda x: x[1])
        
        # Select bottom 10% of stocks by beta
        num_outsiders = max(int(len(sorted_stocks) * BETA_THRESHOLD), 1)
        outsiders = [stock for stock, beta in sorted_stocks[:num_outsiders]]
        
        return outsiders
    
    def backtest_strategy(self, etf_symbol, start_date='2010-01-01', end_date='2017-12-29'):
        """
        Backtest the outsider stock strategy
        """
        # Identify volume spike days
        spike_dates, volume_zscores = self.calculate_volume_spikes(etf_symbol)
        spike_dates = [date for date in spike_dates if date >= pd.Timestamp(start_date) and date <= pd.Timestamp(end_date)]
        
        if not spike_dates:
            print(f"No volume spikes found for {etf_symbol} in the specified period")
            return None
        
        # Get constituents for each spike date
        constituents = self.get_constituents(etf_symbol, spike_dates)
        
        # Calculate betas for constituents
        betas = self.calculate_exponential_betas(etf_symbol, constituents, spike_dates)
        
        # Initialize results storage
        results = {
            'event_dates': [],
            'outsiders': [],
            'cumulative_returns': []
        }
        
        # Process each spike event
        for spike_date in spike_dates:
            if spike_date not in betas:
                continue
                
            # Identify outsiders
            outsiders = self.identify_outsiders(etf_symbol, spike_date, betas)
            
            if not outsiders:
                continue
            
            # Calculate post-event returns
            outsider_returns = self.calculate_post_event_returns(outsiders, spike_date, HOLDING_PERIOD)
            etf_returns = self.calculate_post_event_returns([etf_symbol], spike_date, HOLDING_PERIOD)
            
            if outsider_returns is None or etf_returns is None:
                continue
            
            # Calculate alpha (outsider returns - ETF returns)
            # Note: In practice, we would lever to ETF beta of 1.0 as in the paper
            alpha = outsider_returns - etf_returns
            
            # Apply transaction and borrowing costs
            alpha_after_costs = alpha - TRANSACTION_COST - BORROWING_COST
            
            results['event_dates'].append(spike_date)
            results['outsiders'].append(outsiders)
            results['cumulative_returns'].append(alpha_after_costs)
        
        # Save results
        self.results[etf_symbol] = results
        
        return results
    
    def get_constituents(self, etf_symbol, dates):
        """
        Get unique constituents across all relevant dates
        
        In practice, this would query Bloomberg for historical constituents
        """
        # For demonstration, return placeholder list
        # In real implementation, get actual ETF constituents for each date
        all_constituents = set()
        for date in dates:
            # This is where you'd get the constituents for the ETF as of date
            # For example: constituents = get_bloomberg_constituents(etf_symbol, date)
            if date in self.constituents_by_date:
                all_constituents.update(self.constituents_by_date[date])
            
        return list(all_constituents)
    
    def calculate_post_event_returns(self, tickers, event_date, holding_period):
        """
        Calculate cumulative returns for the specified tickers over the holding period
        """
        end_date = event_date + pd.Timedelta(days=holding_period*1.5)  # Add buffer for trading days
        
        if isinstance(tickers[0], str) and tickers[0].startswith('XL'):  # It's an ETF
            # Get ETF returns
            try:
                prices = self.etf_data[f'{tickers[0]}_Close'].loc[event_date:end_date]
                if len(prices) < holding_period / 2:  # Need enough data points
                    return None
                
                # Get first trading day on or after event date
                start_idx = prices.index.get_indexer([event_date], method='pad')[0]
                end_idx = min(start_idx + holding_period, len(prices) - 1)
                
                if end_idx <= start_idx:
                    return None
                
                cumulative_return = prices.iloc[end_idx] / prices.iloc[start_idx] - 1
                return cumulative_return
            except:
                return None
        else:
            # Get stock returns (equal-weighted portfolio)
            stock_returns = []
            for ticker in tickers:
                try:
                    stock_data = self.constituent_data[self.constituent_data['Ticker'] == ticker]
                    stock_prices = stock_data.set_index('Date')['Close'].loc[event_date:end_date]
                    
                    if len(stock_prices) < holding_period / 2:  # Need enough data points
                        continue
                    
                    # Get first trading day on or after event date
                    start_idx = stock_prices.index.get_indexer([event_date], method='pad')[0]
                    end_idx = min(start_idx + holding_period, len(stock_prices) - 1)
                    
                    if end_idx <= start_idx:
                        continue
                    
                    stock_return = stock_prices.iloc[end_idx] / stock_prices.iloc[start_idx] - 1
                    stock_returns.append(stock_return)
                except:
                    continue
            
            if not stock_returns:
                return None
            
            # Equal-weighted portfolio return
            cumulative_return = np.mean(stock_returns)
            return cumulative_return
    
    def plot_results(self, etf_symbol=None):
        """
        Plot the cumulative alpha results
        """
        if etf_symbol:
            if etf_symbol not in self.results:
                print(f"No results found for {etf_symbol}. Run backtest_strategy first.")
                return
            
            results = self.results[etf_symbol]
            
            plt.figure(figsize=(12, 6))
            plt.title(f"Cumulative Alpha for {etf_symbol} Outsider Strategy")
            plt.xlabel("Days After Volume Spike")
            plt.ylabel("Cumulative Alpha (%)")
            
            # Plot individual events
            for i, returns in enumerate(results['cumulative_returns']):
                plt.plot(returns * 100, alpha=0.3, color='gray')
            
            # Plot average
            avg_returns = np.mean([r for r in results['cumulative_returns']], axis=0)
            plt.plot(avg_returns * 100, linewidth=2, color='blue', label='Average Alpha')
            
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            # Plot average across all ETFs
            avg_returns_by_etf = {}
            
            for etf in self.results:
                if not self.results[etf]['cumulative_returns']:
                    continue
                
                avg_returns = np.mean([r for r in self.results[etf]['cumulative_returns']], axis=0)
                avg_returns_by_etf[etf] = avg_returns
            
            if not avg_returns_by_etf:
                print("No results to plot. Run backtest_strategy first.")
                return
            
            plt.figure(figsize=(14, 7))
            plt.title("Average Cumulative Alpha by ETF")
            plt.xlabel("Days After Volume Spike")
            plt.ylabel("Cumulative Alpha (%)")
            
            for etf, returns in avg_returns_by_etf.items():
                plt.plot(returns * 100, linewidth=2, label=etf)
            
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

# Function to run a complete case study
def run_case_study(etf_symbol, event_date, real_data=False):
    """
    Run a detailed case study for a specific ETF and event date
    """
    if real_data:
        # This would use the full strategy with real data
        pass
    else:
        # Use the pharmaceutical case study from the paper
        if etf_symbol == "XLV" and event_date == "2015-09-28":
            print("\nCase Study: Pharmaceuticals & Healthcare ETF (XLV) - September 2015")
            print("----------------------------------------------------------------")
            print("Period: September 18-28, 2015")
            print("Event: Hillary Clinton's tweet about pharma price gouging + Valeant subpoena")
            print("ETF Performance: XLV dropped 10.7% vs. S&P 500 drop of 5.4%")
            print("\nOutsider Stocks (Low Beta to XLV):")
            print("1. DaVita (DVA) - Beta: 0.57 - Kidney dialysis services")
            print("2. Patterson Companies (PDCO) - Beta: 0.59 - Dental products & vet supplies")
            print("3. Baxter International (BAX) - Beta: 0.66 - Medical equipment")
            print("4. AmerisourceBergen (ABC) - Beta: 0.67 - Medical distribution")
            print("5. LabCorp (LH) - Beta: 0.68 - Clinical laboratory tests")
            print("\nThese companies were in healthcare but not directly exposed to drug pricing concerns.")
            print("Yet they all sold off with the ETF due to indiscriminate ETF selling.")
            print("\nStrategy Results:")
            print("- Equal-weighted portfolio of these 5 stocks returned -8.3% during the selloff")
            print("- This was worse than their ETF-beta implied return of -6.1%")
            print("- Over the next 40 days, they outperformed XLV by +4.2% (after costs)")
        
        elif etf_symbol == "XLF" and event_date == "2016-02-11":
            print("\nCase Study: Financial Sector ETF (XLF) - February 2016")
            print("----------------------------------------------------------------")
            print("Period: February 4-11, 2016")
            print("Event: Fed Chair Janet Yellen indicated no rush to raise rates")
            print("ETF Performance: XLF dropped 6.6% vs. S&P 500 drop of 4.4%")
            print("\nOutsider Stocks (Low Beta to XLF):")
            print("- Seven REITs plus American Express (AXP)")
            print("- Unlike banks, REITs and AXP should have benefited from lower rates")
            print("\nThese companies were in financials but not negatively exposed to lower rates.")
            print("Yet they all sold off with the ETF due to indiscriminate ETF selling.")
            print("\nStrategy Results:")
            print("- Equal-weighted portfolio of these 8 stocks returned -8.5% during the selloff")
            print("- Over the next 40 days, they outperformed XLF by +20.0% (after costs)")
        
        else:
            print(f"No case study available for {etf_symbol} on {event_date}")

# Example of running the strategy with sample data
def run_sample_backtest():
    """
    Demonstrate how the strategy would be run with real data
    """
    strategy = ETFOutsiderStrategy()
    
    # In real implementation, load Bloomberg data here
    # strategy.load_bloomberg_data(['SPY', 'XLV', 'XLF', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLK', 'XLU', 'IJR'],
    #                             '2010-01-01', '2017-12-29')
    
    # For now, show the case studies from the paper
    print("\nETF Outsider Strategy - Based on 'The Revenge of the Stock Pickers'")
    print("=================================================================")
    print("This strategy identifies 'outsider' stocks that are unfairly dragged down during")
    print("high-volume ETF selloffs and buys them as they revert to fundamentals.")
    
    run_case_study("XLV", "2015-09-28")
    run_case_study("XLF", "2016-02-11")
    
    print("\nKey Findings from Full Backtest (2010-2017):")
    print("- Average alpha of 200-300 bps over 40 days (after costs)")
    print("- Strategy works best for sector ETFs rather than broad market ETFs")
    print("- Approximately 30 opportunities per year across ETFs studied")
    print("- Total of 240 volume spike events in the study period")
    
    print("\nTo implement in real-time with Bloomberg data:")
    print("1. Monitor ETF volumes for spikes (Z-score >= 3)")
    print("2. Calculate exponentially weighted betas for ETF constituents")
    print("3. When a high-volume selloff occurs, buy the bottom 10% beta stocks")
    print("4. Hold for up to 40 days, monitoring fundamentals")
    print("5. Lever to ETF beta = 1.0 to calculate proper alpha")

if __name__ == "__main__":
    run_sample_backtest()