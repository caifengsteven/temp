import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import blpapi
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# Set up plotting styles - using a more compatible style
try:
    plt.style.use('seaborn-whitegrid')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # If no seaborn style is available, use default

# Set figure size and DPI
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

class BloombergDataFetcher:
    """
    Class for fetching data from Bloomberg
    """
    def __init__(self):
        self.session_options = blpapi.SessionOptions()
        self.session_options.setServerHost('localhost')
        self.session_options.setServerPort(8194)
    
    def get_sp500_constituents(self, date=None):
        """
        Get S&P 500 constituents as of a given date
        
        Parameters:
        -----------
        date : datetime.date, optional
            Date for which to get constituents. If None, uses current date.
            
        Returns:
        --------
        list
            List of S&P 500 constituent tickers
        """
        session = blpapi.Session(self.session_options)
        constituents = []
        
        try:
            if not session.start():
                raise ConnectionError("Failed to start Bloomberg API session")
            
            if not session.openService('//blp/refdata'):
                raise ConnectionError("Failed to open //blp/refdata service")
            
            refDataService = session.getService('//blp/refdata')
            request = refDataService.createRequest('ReferenceDataRequest')
            
            request.append('securities', 'SPX Index')
            request.append('fields', 'INDX_MEMBERS')
            
            if date is not None:
                overrides = request.getElement('overrides')
                override = overrides.appendElement()
                override.setElement('fieldId', 'REFERENCE_DATE')
                override.setElement('value', date.strftime('%Y%m%d'))
            
            session.sendRequest(request)
            
            done = False
            while not done:
                event = session.nextEvent(500)
                
                if event.eventType() in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
                    for msg in event:
                        if msg.hasElement('securityData') and msg.getElement('securityData').hasElement('fieldData'):
                            field_data = msg.getElement('securityData').getElement('fieldData')
                            if field_data.hasElement('INDX_MEMBERS'):
                                members = field_data.getElement('INDX_MEMBERS')
                                for i in range(members.numValues()):
                                    member = members.getValue(i)
                                    if member.hasElement('Member Ticker and Exchange Code'):
                                        constituents.append(member.getElementAsString('Member Ticker and Exchange Code'))
                    
                    if event.eventType() == blpapi.Event.RESPONSE:
                        done = True
            
            return constituents
            
        except Exception as e:
            print(f"Bloomberg API error when getting index members: {e}")
            return []
        finally:
            session.stop()
    
    def get_historical_prices(self, securities, fields, start_date, end_date):
        """
        Get historical price data for a list of securities
        
        Parameters:
        -----------
        securities : list
            List of security identifiers
        fields : list
            List of fields to retrieve
        start_date : datetime.date
            Start date for data
        end_date : datetime.date
            End date for data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with price/return data
        """
        session = blpapi.Session(self.session_options)
        
        try:
            if not session.start():
                raise ConnectionError("Failed to start Bloomberg API session")
            
            if not session.openService('//blp/refdata'):
                raise ConnectionError("Failed to open //blp/refdata service")
            
            refDataService = session.getService('//blp/refdata')
            request = refDataService.createRequest('HistoricalDataRequest')
            
            for security in securities:
                request.append('securities', security)
            
            for field in fields:
                request.append('fields', field)
            
            request.set('startDate', start_date.strftime('%Y%m%d'))
            request.set('endDate', end_date.strftime('%Y%m%d'))
            
            session.sendRequest(request)
            
            data_dict = {security: {} for security in securities}
            
            done = False
            while not done:
                event = session.nextEvent(500)
                
                if event.eventType() in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
                    for msg in event:
                        security_data = msg.getElement('securityData')
                        ticker = security_data.getElementAsString('security')
                        
                        if security_data.hasElement('fieldData'):
                            field_data = security_data.getElement('fieldData')
                            
                            for i in range(field_data.numValues()):
                                date_element = field_data.getValue(i).getElement('date')
                                date = date_element.getValue()
                                
                                for field in fields:
                                    if field_data.getValue(i).hasElement(field):
                                        field_value = field_data.getValue(i).getElement(field).getValue()
                                        
                                        if date not in data_dict[ticker]:
                                            data_dict[ticker][date] = {}
                                        
                                        data_dict[ticker][date][field] = field_value
                    
                    if event.eventType() == blpapi.Event.RESPONSE:
                        done = True
            
            # Convert to DataFrame
            data_frames = []
            
            for ticker, dates in data_dict.items():
                if dates:  # Check if we have data for this ticker
                    ticker_df = pd.DataFrame.from_dict(dates, orient='index')
                    ticker_df['ticker'] = ticker
                    data_frames.append(ticker_df)
            
            if data_frames:
                combined_df = pd.concat(data_frames, axis=0)
                combined_df.index = pd.to_datetime(combined_df.index)
                combined_df.sort_index(inplace=True)
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Bloomberg API error: {e}")
            return pd.DataFrame()
        finally:
            session.stop()
    
    def get_market_cap(self, securities, date):
        """
        Get market capitalization for a list of securities as of a specific date
        
        Parameters:
        -----------
        securities : list
            List of security identifiers
        date : datetime.date
            Date for which to get market cap
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with market cap data
        """
        session = blpapi.Session(self.session_options)
        
        try:
            if not session.start():
                raise ConnectionError("Failed to start Bloomberg API session")
            
            if not session.openService('//blp/refdata'):
                raise ConnectionError("Failed to open //blp/refdata service")
            
            refDataService = session.getService('//blp/refdata')
            request = refDataService.createRequest('ReferenceDataRequest')
            
            for security in securities:
                request.append('securities', security)
            
            request.append('fields', 'CUR_MKT_CAP')
            
            # Set the date
            overrides = request.getElement('overrides')
            override = overrides.appendElement()
            override.setElement('fieldId', 'REFERENCE_DATE')
            override.setElement('value', date.strftime('%Y%m%d'))
            
            session.sendRequest(request)
            
            market_caps = {}
            
            done = False
            while not done:
                event = session.nextEvent(500)
                
                if event.eventType() in [blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE]:
                    for msg in event:
                        securities_data = msg.getElement('securityData')
                        
                        for i in range(securities_data.numValues()):
                            security_data = securities_data.getValue(i)
                            ticker = security_data.getElementAsString('security')
                            
                            if security_data.hasElement('fieldData') and security_data.getElement('fieldData').hasElement('CUR_MKT_CAP'):
                                market_cap = security_data.getElement('fieldData').getElement('CUR_MKT_CAP').getValue()
                                market_caps[ticker] = market_cap
                    
                    if event.eventType() == blpapi.Event.RESPONSE:
                        done = True
            
            return pd.DataFrame(market_caps.items(), columns=['ticker', 'market_cap']).set_index('ticker')
                
        except Exception as e:
            print(f"Bloomberg API error: {e}")
            return pd.DataFrame()
        finally:
            session.stop()


class PortfolioAnalyzer:
    """
    Class for analyzing equal-weighted and market cap-weighted portfolios
    """
    def __init__(self, start_date, end_date, rebalance_freq='M', transaction_cost=0.0015):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        start_date : datetime.date
            Start date for analysis
        end_date : datetime.date
            End date for analysis
        rebalance_freq : str, optional
            Rebalancing frequency: 'M' for monthly, 'Q' for quarterly, 'Y' for yearly
        transaction_cost : float, optional
            Transaction cost per trade (default: 15 basis points)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.data_fetcher = BloombergDataFetcher()
        
        # Data containers
        self.constituents_history = {}
        self.price_data = None
        self.market_cap_data = {}
        self.portfolio_returns = None
        self.weights_history = {}
        
    def fetch_data(self):
        """
        Fetch required data from Bloomberg
        """
        print("Fetching S&P 500 constituent history...")
        
        # Generate dates for constituent checks (monthly)
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        
        for date in dates:
            try:
                constituents = self.data_fetcher.get_sp500_constituents(date)
                if constituents:
                    self.constituents_history[date.date()] = constituents
                    print(f"Retrieved {len(constituents)} constituents for {date.date()}")
                else:
                    print(f"No constituents found for {date.date()}")
            except Exception as e:
                print(f"Error retrieving constituents for {date.date()}: {e}")
                
        if not self.constituents_history:
            print("Warning: No constituent data was retrieved.")
            return False
        
        # Get unique constituents across all dates
        unique_constituents = set()
        for constituents in self.constituents_history.values():
            unique_constituents.update(constituents)
        
        print(f"Total unique constituents: {len(unique_constituents)}")
        
        if not unique_constituents:
            print("Warning: No unique constituents found.")
            return False
        
        # Fetch price data for all constituents
        print("Fetching price data...")
        try:
            self.price_data = self.data_fetcher.get_historical_prices(
                list(unique_constituents),
                ['PX_LAST', 'DIVIDEND_YIELD'],
                self.start_date,
                self.end_date
            )
            
            if self.price_data.empty:
                print("Warning: No price data retrieved.")
                return False
                
            print(f"Retrieved price data for {self.price_data['ticker'].nunique()} tickers")
        except Exception as e:
            print(f"Error fetching price data: {e}")
            return False
        
        # Fetch market cap data for rebalancing dates
        print("Fetching market cap data...")
        
        for date in dates:
            constituents = self.constituents_history.get(date.date(), [])
            if constituents:
                try:
                    market_caps = self.data_fetcher.get_market_cap(constituents, date)
                    if not market_caps.empty:
                        self.market_cap_data[date.date()] = market_caps
                        print(f"Retrieved market cap data for {date.date()}: {len(market_caps)} stocks")
                    else:
                        print(f"No market cap data for {date.date()}")
                except Exception as e:
                    print(f"Error fetching market cap data for {date.date()}: {e}")
            
        if not self.market_cap_data:
            print("Warning: No market cap data was retrieved.")
            return False
            
        return True
    
    def construct_portfolios(self):
        """
        Construct and backtest equal-weighted and market cap-weighted portfolios
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio returns
        """
        if self.price_data is None or not self.constituents_history or not self.market_cap_data:
            print("No data available. Run fetch_data() first.")
            return None
        
        # Rebalancing dates (dates with both constituents and market cap data)
        common_dates = sorted(set(self.constituents_history.keys()) & set(self.market_cap_data.keys()))
        
        if not common_dates:
            print("No common dates found with both constituent and market cap data.")
            return None
            
        rebalance_dates = common_dates
        
        # Initialize portfolios
        portfolios = {
            'equal_weight': {'weights': {}, 'value': 1.0, 'history': []},
            'market_cap': {'weights': {}, 'value': 1.0, 'history': []}
        }
        
        # Prepare price data
        price_pivot = self.price_data.pivot(columns='ticker', values='PX_LAST')
        returns = price_pivot.pct_change().fillna(0)
        
        # Track portfolio evolution over time
        for i, date in enumerate(rebalance_dates):
            constituents = self.constituents_history[date]
            
            # Equal-weighted portfolio
            equal_weights = {}
            for ticker in constituents:
                equal_weights[ticker] = 1.0 / len(constituents)
            
            # Market cap-weighted portfolio
            market_caps = self.market_cap_data[date]
            total_market_cap = market_caps['market_cap'].sum() if 'market_cap' in market_caps.columns else 0
            market_cap_weights = {}
            
            if total_market_cap > 0:
                for ticker in constituents:
                    if ticker in market_caps.index and 'market_cap' in market_caps.columns:
                        market_cap_weights[ticker] = market_caps.loc[ticker, 'market_cap'] / total_market_cap
                    else:
                        market_cap_weights[ticker] = 0
            else:
                # If no market cap data, use equal weights as fallback
                for ticker in constituents:
                    market_cap_weights[ticker] = 1.0 / len(constituents)
            
            # Normalize weights to ensure they sum to 1
            equal_weights_sum = sum(equal_weights.values())
            market_cap_weights_sum = sum(market_cap_weights.values())
            
            if equal_weights_sum > 0:
                for ticker in equal_weights:
                    equal_weights[ticker] /= equal_weights_sum
            
            if market_cap_weights_sum > 0:
                for ticker in market_cap_weights:
                    market_cap_weights[ticker] /= market_cap_weights_sum
            
            # Store weights
            portfolios['equal_weight']['weights'] = equal_weights
            portfolios['market_cap']['weights'] = market_cap_weights
            
            # Store in weights history
            self.weights_history[(date, 'equal_weight')] = equal_weights
            self.weights_history[(date, 'market_cap')] = market_cap_weights
            
            # Calculate portfolio returns between this rebalance date and the next
            if i < len(rebalance_dates) - 1:
                next_date = rebalance_dates[i + 1]
                
                # Get dates between current and next rebalance date
                mask = (returns.index >= pd.Timestamp(date)) & (returns.index < pd.Timestamp(next_date))
                period_returns = returns.loc[mask]
                
                # Calculate portfolio returns
                for portfolio_type in portfolios:
                    weights = portfolios[portfolio_type]['weights']
                    
                    # Calculate daily returns
                    daily_returns = []
                    
                    for day, day_returns in period_returns.iterrows():
                        # Calculate weighted return for the day
                        day_return = 0
                        for ticker, weight in weights.items():
                            if ticker in day_returns.index and not pd.isna(day_returns[ticker]):
                                day_return += weight * day_returns[ticker]
                        
                        daily_returns.append((day, day_return))
                    
                    # Update portfolio value
                    for day, day_return in daily_returns:
                        portfolios[portfolio_type]['value'] *= (1 + day_return)
                        portfolios[portfolio_type]['history'].append((day, portfolios[portfolio_type]['value']))
                
                # Apply transaction costs at rebalancing (simplified approach)
                for portfolio_type in portfolios:
                    if i > 0:  # Skip first rebalance (initial portfolio construction)
                        old_weights = self.weights_history.get((rebalance_dates[i-1], portfolio_type), {})
                        new_weights = portfolios[portfolio_type]['weights']
                        
                        # Calculate turnover
                        turnover = 0
                        for ticker in set(old_weights.keys()) | set(new_weights.keys()):
                            old_weight = old_weights.get(ticker, 0)
                            new_weight = new_weights.get(ticker, 0)
                            turnover += abs(new_weight - old_weight)
                        
                        # Apply transaction costs
                        cost = turnover * self.transaction_cost
                        portfolios[portfolio_type]['value'] *= (1 - cost)
        
        # Convert portfolio histories to DataFrames
        if not portfolios['equal_weight']['history'] or not portfolios['market_cap']['history']:
            print("Warning: No portfolio history generated.")
            return None
            
        portfolio_values = pd.DataFrame({
            'equal_weight': [x[1] for x in portfolios['equal_weight']['history']],
            'market_cap': [x[1] for x in portfolios['market_cap']['history']]
        }, index=[pd.Timestamp(x[0]) for x in portfolios['equal_weight']['history']])
        
        # Calculate returns
        self.portfolio_returns = portfolio_values.pct_change().fillna(0)
        
        return self.portfolio_returns
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for the portfolios
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with performance metrics
        """
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            print("No portfolio returns available. Run construct_portfolios() first.")
            return None
        
        # Calculate cumulative returns
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        
        # Calculate metrics
        metrics = {}
        
        for portfolio in ['equal_weight', 'market_cap']:
            # Annualized return
            total_return = cumulative_returns[portfolio].iloc[-1] / cumulative_returns[portfolio].iloc[0] - 1
            years = (self.portfolio_returns.index[-1] - self.portfolio_returns.index[0]).days / 365.25
            ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Volatility
            ann_vol = self.portfolio_returns[portfolio].std() * np.sqrt(252)  # Assuming daily returns
            
            # Sharpe ratio (using 0% risk-free rate for simplicity)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            # Maximum drawdown
            rolling_max = cumulative_returns[portfolio].cummax()
            drawdowns = (cumulative_returns[portfolio] / rolling_max) - 1
            max_drawdown = drawdowns.min()
            
            # Store metrics
            metrics[portfolio] = {
                'ann_return': ann_return,
                'ann_vol': ann_vol,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown
            }
        
        return pd.DataFrame(metrics).T
    
    def plot_portfolio_performance(self):
        """
        Plot portfolio performance
        """
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            print("No portfolio returns available. Run construct_portfolios() first.")
            return
        
        # Calculate cumulative returns
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        cumulative_returns.plot()
        plt.title('Cumulative Returns')
        plt.ylabel('Cumulative Return')
        plt.xlabel('Date')
        plt.legend(['Equal-Weighted', 'Market Cap-Weighted'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cumulative_returns.png')
        plt.close()
        
        # Plot relative performance
        plt.figure(figsize=(12, 6))
        relative_perf = cumulative_returns['equal_weight'] / cumulative_returns['market_cap']
        relative_perf.plot()
        plt.title('Equal-Weighted Portfolio Relative to Market Cap-Weighted')
        plt.ylabel('Relative Performance')
        plt.xlabel('Date')
        plt.axhline(y=1, color='r', linestyle='--')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('relative_performance.png')
        plt.close()
        
        # Plot rolling 1-year returns
        plt.figure(figsize=(12, 6))
        rolling_returns = self.portfolio_returns.rolling(252).apply(lambda x: (1 + x).prod() - 1)
        rolling_relative = (1 + rolling_returns['equal_weight']) / (1 + rolling_returns['market_cap']) - 1
        rolling_relative.plot()
        plt.title('Rolling 1-Year Equal-Weight Relative Returns')
        plt.ylabel('Relative Return')
        plt.xlabel('Date')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('rolling_relative_returns.png')
        plt.close()
    
    def calculate_stochastic_components(self):
        """
        Calculate the components of the stochastic portfolio theory decomposition
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with decomposition components
        """
        if self.portfolio_returns is None or not self.weights_history:
            print("No data available. Run construct_portfolios() first.")
            return None
        
        # Get rebalancing dates
        rebalance_dates = sorted(set(date for date, _ in self.weights_history.keys()))
        
        # Initialize containers for components
        components = {
            'portfolio_gen_function': [],
            'excess_growth_rate': [],
            'leakage': [],
            'relative_return': []
        }
        
        # Track actual relative returns for comparison
        actual_relative_returns = []
        
        # For each rebalance period
        for i in range(len(rebalance_dates) - 1):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]
            
            # Get market cap weights
            mc_weights_current = self.weights_history.get((current_date, 'market_cap'), {})
            mc_weights_next = self.weights_history.get((next_date, 'market_cap'), {})
            
            # Calculate portfolio generating function (ln(S(μ)))
            # For equal weight: S(μ) = (μ₁ × μ₂ × ... × μₙ)^(1/n)
            log_S_current = 0
            log_S_next = 0
            
            n_current = len(mc_weights_current)
            n_next = len(mc_weights_next)
            
            if n_current > 0:
                # Calculate log(S(μ)) for current weights
                log_S_current = (1/n_current) * sum(np.log(max(weight, 1e-10)) for weight in mc_weights_current.values())
            
            if n_next > 0:
                # Calculate log(S(μ)) for next weights
                log_S_next = (1/n_next) * sum(np.log(max(weight, 1e-10)) for weight in mc_weights_next.values())
            
            # Change in portfolio generating function
            delta_log_S = log_S_next - log_S_current
            
            # Calculate excess growth rate (γ*)
            # γ* = (1/2) × [Σ(πᵢ × σᵢ²) - Σ(πᵢ × πⱼ × σᵢⱼ)]
            
            # Get equal weight portfolio weights
            eq_weights = self.weights_history.get((current_date, 'equal_weight'), {})
            
            # Get prices for the period around current date
            if self.price_data is not None and not self.price_data.empty:
                # Convert current_date to Timestamp for comparison
                current_ts = pd.Timestamp(current_date)
                start_date_3m = current_ts - pd.DateOffset(months=3)
                mask = ((self.price_data.index >= start_date_3m) & 
                        (self.price_data.index <= current_ts))
                
                if mask.any():
                    prices_period = self.price_data.loc[mask]
                    
                    # Calculate returns
                    price_pivot = prices_period.pivot(columns='ticker', values='PX_LAST')
                    returns = price_pivot.pct_change().dropna()
                    
                    # Get returns for constituents we have weights for
                    common_tickers = set(eq_weights.keys()) & set(returns.columns)
                    
                    if common_tickers:
                        returns_subset = returns[list(common_tickers)]
                        
                        # Reorganize weights to match returns columns
                        eq_weights_array = np.array([eq_weights.get(ticker, 0) for ticker in returns_subset.columns])
                        
                        # Calculate covariance matrix
                        cov_matrix = returns_subset.cov().values
                        
                        # Calculate excess growth rate
                        weighted_variances = np.sum(eq_weights_array * np.diag(cov_matrix))
                        portfolio_variance = np.dot(eq_weights_array, np.dot(cov_matrix, eq_weights_array))
                        
                        excess_growth = 0.5 * (weighted_variances - portfolio_variance)
                    else:
                        excess_growth = 0
                else:
                    excess_growth = 0
            else:
                excess_growth = 0
            
            # Calculate leakage
            # This is complex and requires tracking of rank changes
            # We'll use a simplified approach
            
            mc_tickers_current = set(mc_weights_current.keys())
            mc_tickers_next = set(mc_weights_next.keys())
            
            # Stocks that exited
            exited_tickers = mc_tickers_current - mc_tickers_next
            # Stocks that entered
            entered_tickers = mc_tickers_next - mc_tickers_current
            
            # Calculate leakage impact
            leakage = 0
            
            # Impact of stocks exiting (negative impact on equal weight relative performance)
            for ticker in exited_tickers:
                mc_weight = mc_weights_current.get(ticker, 0)
                eq_weight = eq_weights.get(ticker, 0)
                leakage -= (eq_weight - mc_weight)  # Simplified approach
            
            # Impact of stocks entering (positive impact on equal weight relative performance)
            for ticker in entered_tickers:
                mc_weight_next = mc_weights_next.get(ticker, 0)
                eq_weight_next = 1.0 / n_next if n_next > 0 else 0
                leakage += (eq_weight_next - mc_weight_next)  # Simplified approach
            
            # Scale leakage by a factor (this is a simplification)
            leakage *= 0.05
            
            # Store components
            components['portfolio_gen_function'].append((next_date, delta_log_S))
            components['excess_growth_rate'].append((next_date, excess_growth))
            components['leakage'].append((next_date, leakage))
            
            # Calculate theoretical relative return
            relative_return = delta_log_S + excess_growth + leakage
            components['relative_return'].append((next_date, relative_return))
            
            # Calculate actual relative return for the period
            mask = ((self.portfolio_returns.index >= pd.Timestamp(current_date)) & 
                    (self.portfolio_returns.index < pd.Timestamp(next_date)))
            
            if mask.any():
                period_returns = self.portfolio_returns.loc[mask]
                eq_period_return = (1 + period_returns['equal_weight']).prod() - 1
                mc_period_return = (1 + period_returns['market_cap']).prod() - 1
                actual_relative = (1 + eq_period_return) / (1 + mc_period_return) - 1
                actual_relative_returns.append((next_date, actual_relative))
        
        # Convert to DataFrames
        components_df = {}
        for component, values in components.items():
            if values:
                components_df[component] = pd.Series([x[1] for x in values], index=[pd.Timestamp(x[0]) for x in values])
            else:
                components_df[component] = pd.Series(dtype=float)
        
        if actual_relative_returns:
            actual_relative_df = pd.Series([x[1] for x in actual_relative_returns], 
                                          index=[pd.Timestamp(x[0]) for x in actual_relative_returns])
            components_df['actual_relative'] = actual_relative_df
        
        return pd.DataFrame(components_df)
    
    def plot_stochastic_components(self, components_df):
        """
        Plot stochastic portfolio theory components
        
        Parameters:
        -----------
        components_df : pd.DataFrame
            DataFrame with decomposition components
        """
        if components_df is None or components_df.empty:
            print("No components data available.")
            return
            
        # Plot portfolio generating function
        plt.figure(figsize=(12, 6))
        components_df['portfolio_gen_function'].cumsum().plot()
        plt.title('Cumulative Change in Portfolio Generating Function')
        plt.ylabel('Cumulative Change')
        plt.xlabel('Date')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('portfolio_generating_function.png')
        plt.close()
        
        # Plot excess growth rate
        plt.figure(figsize=(12, 6))
        components_df['excess_growth_rate'].plot()
        plt.title('Excess Growth Rate')
        plt.ylabel('Growth Rate')
        plt.xlabel('Date')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('excess_growth_rate.png')
        plt.close()
        
        # Plot relative return decomposition
        plt.figure(figsize=(10, 8))
        
        # Cumulative components
        cumulative_components = pd.DataFrame({
            'Portfolio Gen. Function': components_df['portfolio_gen_function'].cumsum(),
            'Excess Growth Rate': components_df['excess_growth_rate'].cumsum(),
            'Leakage': components_df['leakage'].cumsum(),
            'Total Theoretical': components_df['relative_return'].cumsum()
        })
        
        if 'actual_relative' in components_df.columns:
            cumulative_components['Actual Relative'] = components_df['actual_relative'].cumsum()
        
        cumulative_components.plot()
        plt.title('Cumulative Decomposition of Relative Returns')
        plt.ylabel('Cumulative Contribution')
        plt.xlabel('Date')
        plt.legend(['Portfolio Gen. Function', 'Excess Growth Rate', 'Leakage', 
                   'Total Theoretical', 'Actual Relative'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cumulative_decomposition.png')
        plt.close()
        
        # Plot rolling 12-month contributions if we have enough data
        if len(components_df) >= 12:
            plt.figure(figsize=(12, 8))
            
            # Rolling 12-month sum of components
            rolling_components = pd.DataFrame({
                'Portfolio Gen. Function': components_df['portfolio_gen_function'].rolling(12).sum(),
                'Excess Growth Rate': components_df['excess_growth_rate'].rolling(12).sum(),
                'Leakage': components_df['leakage'].rolling(12).sum(),
                'Total Theoretical': components_df['relative_return'].rolling(12).sum()
            })
            
            if 'actual_relative' in components_df.columns:
                rolling_components['Actual Relative'] = components_df['actual_relative'].rolling(12).sum()
            
            rolling_components.plot()
            plt.title('Rolling 12-Month Decomposition of Relative Returns')
            plt.ylabel('12-Month Contribution')
            plt.xlabel('Date')
            plt.legend(['Portfolio Gen. Function', 'Excess Growth Rate', 'Leakage', 
                      'Total Theoretical', 'Actual Relative'])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('rolling_decomposition.png')
            plt.close()
    
    def implement_dynamic_strategy(self, components_df, lookback_window=36):
        """
        Implement the dynamic switching strategy proposed in the paper
        
        Parameters:
        -----------
        components_df : pd.DataFrame
            DataFrame with decomposition components
        lookback_window : int, optional
            Number of months to use for regression model fitting
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with strategy returns
        """
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            print("No portfolio returns available. Run construct_portfolios() first.")
            return None
            
        if components_df is None or components_df.empty:
            print("No components data available. Run calculate_stochastic_components() first.")
            return None
        
        # Initialize strategy container
        dynamic_strategy = pd.Series(index=self.portfolio_returns.index, dtype=float)
        
        # Rebalancing dates (monthly)
        rebalance_dates = pd.date_range(start=self.portfolio_returns.index[0], 
                                        end=self.portfolio_returns.index[-1], 
                                        freq='MS')
        
        # Current allocation (0 = market cap, 1 = equal weight)
        current_allocation = 1  # Start with equal weight
        
        # For each rebalance date
        for i, date in enumerate(rebalance_dates):
            # Skip if we don't have enough lookback data
            if i < lookback_window:
                # For initial period, use equal weight
                mask = ((self.portfolio_returns.index >= date) & 
                        (self.portfolio_returns.index < (date + pd.DateOffset(months=1))))
                
                if mask.any():
                    dynamic_strategy.loc[mask] = self.portfolio_returns.loc[mask, 'equal_weight']
                continue
            
            # Find closest date in components_df
            closest_idx = components_df.index.get_indexer([date], method='pad')[0]
            
            if closest_idx < 0 or closest_idx < lookback_window:
                # Not enough data
                continue
                
            # Get training data
            X_train = components_df.iloc[closest_idx-lookback_window:closest_idx][['portfolio_gen_function', 'excess_growth_rate', 'leakage']]
            if 'actual_relative' in components_df.columns:
                y_train = components_df.iloc[closest_idx-lookback_window:closest_idx]['actual_relative']
            else:
                y_train = components_df.iloc[closest_idx-lookback_window:closest_idx]['relative_return']
            
            # Skip if we don't have complete data
            if X_train.isna().any().any() or y_train.isna().any():
                continue
            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict next month's relative return
            next_features = components_df.iloc[closest_idx][['portfolio_gen_function', 'excess_growth_rate', 'leakage']]
            
            # Skip if we have NaN values
            if pd.isna(next_features).any():
                continue
                
            predicted_relative = model.predict(next_features.values.reshape(1, -1))[0]
            
            # Decide on allocation
            # Switch only if predicted outperformance is greater than transaction costs
            if predicted_relative > self.transaction_cost and current_allocation == 0:
                # Switch to equal weight
                current_allocation = 1
            elif predicted_relative < -self.transaction_cost and current_allocation == 1:
                # Switch to market cap
                current_allocation = 0
            
            # Apply allocation for the next month
            next_month = date + pd.DateOffset(months=1)
            mask = ((self.portfolio_returns.index >= date) & 
                    (self.portfolio_returns.index < next_month))
            
            if mask.any():
                if current_allocation == 1:
                    # Equal weight
                    dynamic_strategy.loc[mask] = self.portfolio_returns.loc[mask, 'equal_weight']
                else:
                    # Market cap
                    dynamic_strategy.loc[mask] = self.portfolio_returns.loc[mask, 'market_cap']
        
        # Fill initial period with equal weight returns
        mask = dynamic_strategy.isna()
        dynamic_strategy.loc[mask] = self.portfolio_returns.loc[mask, 'equal_weight']
        
        # Add dynamic strategy to portfolio returns
        self.portfolio_returns['dynamic'] = dynamic_strategy
        
        return self.portfolio_returns
    
    def evaluate_strategy(self):
        """
        Evaluate all strategies (equal weight, market cap, dynamic)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with performance metrics
        """
        if 'dynamic' not in self.portfolio_returns.columns:
            print("Dynamic strategy not implemented. Run implement_dynamic_strategy() first.")
            return None
        
        # Calculate cumulative returns
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        
        # Calculate metrics
        metrics = {}
        
        for portfolio in ['equal_weight', 'market_cap', 'dynamic']:
            # Annualized return
            total_return = cumulative_returns[portfolio].iloc[-1] / cumulative_returns[portfolio].iloc[0] - 1
            years = (self.portfolio_returns.index[-1] - self.portfolio_returns.index[0]).days / 365.25
            ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Volatility
            ann_vol = self.portfolio_returns[portfolio].std() * np.sqrt(252)  # Assuming daily returns
            
            # Sharpe ratio (using 0% risk-free rate for simplicity)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            # Information ratio (vs. market cap)
            if portfolio != 'market_cap':
                excess_returns = self.portfolio_returns[portfolio] - self.portfolio_returns['market_cap']
                tracking_error = excess_returns.std() * np.sqrt(252)
                ir = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            else:
                ir = float('nan')
            
            # Maximum drawdown
            rolling_max = cumulative_returns[portfolio].cummax()
            drawdowns = (cumulative_returns[portfolio] / rolling_max) - 1
            max_drawdown = drawdowns.min()
            
            # Maximum relative drawdown (vs. market cap)
            if portfolio != 'market_cap':
                relative_cumulative = cumulative_returns[portfolio] / cumulative_returns['market_cap']
                relative_max = relative_cumulative.cummax()
                relative_drawdowns = (relative_cumulative / relative_max) - 1
                rel_max_drawdown = relative_drawdowns.min()
            else:
                rel_max_drawdown = float('nan')
            
            # Store metrics
            metrics[portfolio] = {
                'ann_return': ann_return,
                'ann_vol': ann_vol,
                'sharpe': sharpe,
                'info_ratio': ir,
                'max_drawdown': max_drawdown,
                'max_rel_drawdown': rel_max_drawdown
            }
        
        return pd.DataFrame(metrics).T
    
    def plot_strategy_performance(self):
        """
        Plot performance of all strategies
        """
        if 'dynamic' not in self.portfolio_returns.columns:
            print("Dynamic strategy not implemented. Run implement_dynamic_strategy() first.")
            return
        
        # Calculate cumulative returns
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        cumulative_returns.plot()
        plt.title('Cumulative Returns')
        plt.ylabel('Cumulative Return')
        plt.xlabel('Date')
        plt.legend(['Equal-Weighted', 'Market Cap-Weighted', 'Dynamic Strategy'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('strategy_cumulative_returns.png')
        plt.close()
        
        # Plot relative performance to market cap
        plt.figure(figsize=(12, 6))
        relative_eq = cumulative_returns['equal_weight'] / cumulative_returns['market_cap']
        relative_dynamic = cumulative_returns['dynamic'] / cumulative_returns['market_cap']
        
        relative_perf = pd.DataFrame({
            'Equal-Weighted': relative_eq,
            'Dynamic Strategy': relative_dynamic
        })
        
        relative_perf.plot()
        plt.title('Relative Performance to Market Cap-Weighted Portfolio')
        plt.ylabel('Relative Performance')
        plt.xlabel('Date')
        plt.axhline(y=1, color='r', linestyle='--')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('strategy_relative_performance.png')
        plt.close()
        
        # Plot drawdowns
        plt.figure(figsize=(12, 6))
        
        # Calculate drawdowns
        drawdowns = {}
        for portfolio in ['equal_weight', 'market_cap', 'dynamic']:
            rolling_max = cumulative_returns[portfolio].cummax()
            drawdowns[portfolio] = (cumulative_returns[portfolio] / rolling_max) - 1
        
        drawdowns_df = pd.DataFrame(drawdowns)
        drawdowns_df.plot()
        plt.title('Portfolio Drawdowns')
        plt.ylabel('Drawdown')
        plt.xlabel('Date')
        plt.legend(['Equal-Weighted', 'Market Cap-Weighted', 'Dynamic Strategy'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('strategy_drawdowns.png')
        plt.close()
        
        # Plot relative drawdowns
        plt.figure(figsize=(12, 6))
        
        # Calculate relative drawdowns
        rel_drawdowns = {}
        for portfolio in ['equal_weight', 'dynamic']:
            relative_cum = cumulative_returns[portfolio] / cumulative_returns['market_cap']
            relative_max = relative_cum.cummax()
            rel_drawdowns[portfolio] = (relative_cum / relative_max) - 1
        
        rel_drawdowns_df = pd.DataFrame(rel_drawdowns)
        rel_drawdowns_df.plot()
        plt.title('Relative Drawdowns to Market Cap-Weighted Portfolio')
        plt.ylabel('Relative Drawdown')
        plt.xlabel('Date')
        plt.legend(['Equal-Weighted', 'Dynamic Strategy'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('strategy_relative_drawdowns.png')
        plt.close()
        
        # Plot allocation over time
        plt.figure(figsize=(12, 6))
        allocation = (self.portfolio_returns['dynamic'] == self.portfolio_returns['equal_weight']).astype(int)
        allocation = allocation.resample('M').mean()  # Monthly resampling for clearer view
        allocation.plot(drawstyle='steps-post')
        plt.title('Dynamic Strategy Allocation')
        plt.ylabel('Allocation (1 = Equal Weight, 0 = Market Cap)')
        plt.xlabel('Date')
        plt.yticks([0, 1], ['Market Cap', 'Equal Weight'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('strategy_allocation.png')
        plt.close()


def main():
    print("Analyzing Equal Weight vs. Market Cap Weight Portfolio Performance")
    
    try:
        # Set date range - using a recent period to cover the underperformance since 2016
        start_date = dt.date(2013, 1, 1)
        end_date = dt.date(2023, 12, 31)
        
        print(f"Analysis period: {start_date} to {end_date}")
        
        # Create analyzer
        analyzer = PortfolioAnalyzer(
            start_date=start_date,
            end_date=end_date,
            rebalance_freq='M',
            transaction_cost=0.0015  # 15 basis points
        )
        
        # Fetch data
        success = analyzer.fetch_data()
        
        if not success:
            print("Data fetch unsuccessful. Check Bloomberg connection or try with a different date range.")
            return
        
        # Construct portfolios
        portfolio_returns = analyzer.construct_portfolios()
        
        if portfolio_returns is None:
            print("Portfolio construction failed. Check the data.")
            return
        
        # Calculate performance metrics
        metrics = analyzer.calculate_performance_metrics()
        print("\nPortfolio Performance Metrics:")
        print(metrics)
        
        # Plot portfolio performance
        analyzer.plot_portfolio_performance()
        
        # Calculate stochastic components
        components = analyzer.calculate_stochastic_components()
        
        if components is not None:
            # Plot stochastic components
            analyzer.plot_stochastic_components(components)
            
            # Implement dynamic strategy
            analyzer.implement_dynamic_strategy(components)
            
            # Evaluate strategy
            strategy_metrics = analyzer.evaluate_strategy()
            print("\nStrategy Performance Metrics:")
            print(strategy_metrics)
            
            # Plot strategy performance
            analyzer.plot_strategy_performance()
            
            print("\nAnalysis complete. Results saved as PNG files in the current directory.")
        else:
            print("Stochastic component calculation failed. Check the data.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()