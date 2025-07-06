import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define S&P 500 sector ETFs
sector_etfs = {
    'XLE': 'Energy',
    'XLB': 'Materials',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLV': 'Health Care',
    'XLF': 'Financials',
    'XLK': 'Information Technology',
    'XLC': 'Communication Services',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate'
}

class SectorRotationStrategy:
    def __init__(self, start_date='2010-01-01', end_date='2023-01-01', data_frequency='monthly'):
        """
        Initialize the sector rotation strategy.
        
        Parameters:
        -----------
        start_date : str
            Start date for data collection
        end_date : str
            End date for data collection
        data_frequency : str
            Frequency of data ('monthly', 'quarterly', etc.)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_frequency = data_frequency
        self.sector_data = None
        self.factor_data = None
        self.fundamental_data = None
        self.model = None
        
    def get_data(self, use_simulated=True):
        """
        Get sector price data and fundamental data.
        
        Parameters:
        -----------
        use_simulated : bool
            If True, use simulated data instead of downloading real data
        """
        if use_simulated:
            # Generate simulated price data for sectors
            print("Generating simulated sector price data...")
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
            
            # Create an empty DataFrame with dates as index
            df = pd.DataFrame(index=dates)
            
            # Generate price series for each sector
            for ticker, sector_name in sector_etfs.items():
                # Base price
                base_price = 100
                
                # Sector-specific parameters (mean and volatility)
                if sector_name == 'Information Technology':
                    mean_return = 0.00045  # Higher mean return
                    volatility = 0.015
                elif sector_name == 'Energy':
                    mean_return = 0.0002
                    volatility = 0.018  # Higher volatility
                elif sector_name == 'Utilities':
                    mean_return = 0.0002
                    volatility = 0.008  # Lower volatility
                else:
                    mean_return = 0.0003
                    volatility = 0.012
                
                # Generate random returns with mean and volatility
                returns = np.random.normal(mean_return, volatility, len(dates))
                
                # Apply occasional sector-specific shocks
                if sector_name == 'Energy':
                    # Oil price shock
                    shock_idx = len(dates) // 3
                    returns[shock_idx:shock_idx+20] -= 0.02
                elif sector_name == 'Financials':
                    # Financial crisis
                    shock_idx = len(dates) // 4
                    returns[shock_idx:shock_idx+30] -= 0.015
                
                # Convert returns to price series
                prices = base_price * (1 + returns).cumprod()
                
                # Add to DataFrame
                df[ticker] = prices
            
            # Sample at desired frequency
            if self.data_frequency == 'monthly':
                df = df.resample('M').last()
            elif self.data_frequency == 'quarterly':
                df = df.resample('Q').last()
            
            self.sector_data = df
            
            # Generate simulated fundamental data
            self.generate_simulated_fundamentals()
        else:
            # Download real data using yfinance
            print("Downloading sector ETF data...")
            self.sector_data = yf.download(
                tickers=list(sector_etfs.keys()),
                start=self.start_date,
                end=self.end_date,
                interval='1d'
            )['Adj Close']
            
            # Resample to desired frequency
            if self.data_frequency == 'monthly':
                self.sector_data = self.sector_data.resample('M').last()
            elif self.data_frequency == 'quarterly':
                self.sector_data = self.sector_data.resample('Q').last()
            
            # We would need to fetch fundamental data from a financial API
            # This is a placeholder for that functionality
            print("Downloading fundamental data is not implemented. Using simulated data instead.")
            self.generate_simulated_fundamentals()
        
        # Calculate returns
        self.returns = self.sector_data.pct_change().dropna()
        
        print(f"Data collected for {len(self.sector_data)} time periods.")
    
    def generate_simulated_fundamentals(self):
        """Generate simulated fundamental data for each sector."""
        print("Generating simulated fundamental data...")
        
        # Use quarterly frequency for fundamentals
        if self.data_frequency == 'monthly':
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='Q')
        else:
            dates = self.sector_data.index
        
        # Create a DataFrame for fundamental data
        fundamentals = {}
        
        # For each sector, generate fundamental metrics
        for ticker, sector_name in sector_etfs.items():
            sector_fundamentals = pd.DataFrame(index=dates)
            
            # Set sector-specific base values
            if sector_name == 'Information Technology':
                pe_base = 25
                pb_base = 5
                ev_ebitda_base = 15
                dividend_yield_base = 1
                gross_margin_base = 45
                profit_margin_base = 20
                roe_base = 25
            elif sector_name == 'Energy':
                pe_base = 15
                pb_base = 1.5
                ev_ebitda_base = 8
                dividend_yield_base = 4
                gross_margin_base = 20
                profit_margin_base = 10
                roe_base = 12
            elif sector_name == 'Utilities':
                pe_base = 18
                pb_base = 2
                ev_ebitda_base = 10
                dividend_yield_base = 3.5
                gross_margin_base = 40
                profit_margin_base = 15
                roe_base = 10
            elif sector_name == 'Financials':
                pe_base = 12
                pb_base = 1.2
                ev_ebitda_base = 9
                dividend_yield_base = 3
                gross_margin_base = 35
                profit_margin_base = 18
                roe_base = 15
            elif sector_name == 'Real Estate':
                pe_base = 20
                pb_base = 2.5
                ev_ebitda_base = 18
                dividend_yield_base = 3.2
                gross_margin_base = 30
                profit_margin_base = 25
                roe_base = 8
            else:
                pe_base = 17
                pb_base = 2.5
                ev_ebitda_base = 12
                dividend_yield_base = 2
                gross_margin_base = 30
                profit_margin_base = 15
                roe_base = 15
            
            # Generate time series with trends and noise
            noise_level = 0.15  # 15% noise
            trend_level = 0.02  # 2% annual trend
            
            # Time component for trends
            t = np.linspace(0, len(dates)/4, len(dates))  # Assuming quarterly data, so 4 points per year
            
            # Generate fundamental metrics with trends and noise
            sector_fundamentals['PE'] = pe_base * (1 + trend_level * t) * (1 + np.random.normal(0, noise_level, len(dates)))
            sector_fundamentals['PB'] = pb_base * (1 + trend_level * t) * (1 + np.random.normal(0, noise_level, len(dates)))
            sector_fundamentals['EV_EBITDA'] = ev_ebitda_base * (1 + trend_level * t/2) * (1 + np.random.normal(0, noise_level, len(dates)))
            sector_fundamentals['DividendYield'] = dividend_yield_base * (1 - trend_level * t/3) * (1 + np.random.normal(0, noise_level/2, len(dates)))
            sector_fundamentals['GrossMargin'] = gross_margin_base * (1 + trend_level * t/4) * (1 + np.random.normal(0, noise_level/3, len(dates)))
            sector_fundamentals['ProfitMargin'] = profit_margin_base * (1 + trend_level * t/3) * (1 + np.random.normal(0, noise_level, len(dates)))
            sector_fundamentals['ROE'] = roe_base * (1 + trend_level * t/2) * (1 + np.random.normal(0, noise_level, len(dates)))
            
            # Add sector-specific events
            if sector_name == 'Energy':
                # Oil price shock
                shock_idx = len(dates) // 3
                sector_fundamentals.iloc[shock_idx:shock_idx+4, :] *= 0.8  # Decrease metrics by 20%
            elif sector_name == 'Information Technology':
                # Tech boom
                shock_idx = len(dates) // 2
                sector_fundamentals.iloc[shock_idx:shock_idx+6, :] *= 1.2  # Increase metrics by 20%
            
            fundamentals[ticker] = sector_fundamentals
        
        # Combine all sectors into a multi-level DataFrame
        self.fundamental_data = pd.concat(fundamentals, axis=1)
    
    def calculate_momentum_factors(self, lookback_periods=[1, 2, 3, 6, 7, 8, 9, 12]):
        """
        Calculate momentum factors for different lookback periods.
        
        Parameters:
        -----------
        lookback_periods : list
            List of lookback periods in months
        """
        print("Calculating momentum factors...")
        
        # Convert lookback periods to number of observations
        if self.data_frequency == 'monthly':
            lookback_obs = lookback_periods
        elif self.data_frequency == 'quarterly':
            lookback_obs = [max(1, int(p/3)) for p in lookback_periods]
        
        # Initialize factor data
        self.momentum_factors = {}
        
        # Calculate momentum for each lookback period
        for period in lookback_obs:
            # Calculate cumulative return over the period
            cum_returns = self.sector_data.pct_change(period).shift(1)
            
            # Skip most recent 10% of the period to avoid short-term reversion
            skip_obs = max(1, int(0.1 * period))
            short_term_returns = self.sector_data.pct_change(skip_obs).shift(1)
            
            # Calculate momentum factor as past period return minus short-term return
            momentum = cum_returns - short_term_returns
            
            # Store the factor
            self.momentum_factors[f'MOM_{period}'] = momentum
        
        print(f"Calculated momentum factors for {len(lookback_obs)} lookback periods.")
    
    def calculate_reversion_factors(self, lookback_days=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]):
        """
        Calculate short-term reversion factors for different lookback periods.
        
        Parameters:
        -----------
        lookback_days : list
            List of lookback periods in days
        """
        print("Calculating reversion factors...")
        
        # Convert lookback days to number of observations
        if self.data_frequency == 'monthly':
            lookback_obs = [max(1, int(d/21)) for d in lookback_days]  # Approximately 21 trading days per month
        elif self.data_frequency == 'quarterly':
            lookback_obs = [max(1, int(d/63)) for d in lookback_days]  # Approximately 63 trading days per quarter
        
        # Initialize factor data
        self.reversion_factors = {}
        
        # Calculate reversion for each lookback period
        for days, obs in zip(lookback_days, lookback_obs):
            # Calculate negative of cumulative return over the period
            reversion = -self.sector_data.pct_change(obs).shift(1)
            
            # Store the factor
            self.reversion_factors[f'REV_{days}D'] = reversion
        
        print(f"Calculated reversion factors for {len(lookback_days)} lookback periods.")
    
    def backtest_factor(self, factor_data, factor_name):
        """
        Backtest a factor by ranking sectors and taking long/short positions.
        
        Parameters:
        -----------
        factor_data : pd.DataFrame
            Factor data for all sectors
        factor_name : str
            Name of the factor
            
        Returns:
        --------
        factor_returns : pd.Series
            Returns of the factor strategy
        """
        factor_returns = []
        
        # Iterate through each time period
        for date in factor_data.index:
            if date in self.returns.index:
                # Get factor values for this date
                factor_values = factor_data.loc[date]
                
                # Rank sectors based on factor values
                rankings = factor_values.rank(ascending=False)
                
                # Take long positions in top 2 sectors
                long_sectors = rankings[rankings <= 2].index
                
                # Take short positions in bottom 2 sectors
                short_sectors = rankings[rankings >= len(rankings) - 1].index
                
                # Calculate next period returns
                next_returns = self.returns.loc[date]
                
                # Calculate strategy return
                long_return = next_returns[long_sectors].mean()
                short_return = -next_returns[short_sectors].mean()
                strategy_return = (long_return + short_return) / 2
                
                factor_returns.append(strategy_return)
            
        # Convert to Series
        factor_returns = pd.Series(factor_returns, index=factor_data.index[factor_data.index.isin(self.returns.index)])
        
        # Calculate statistics
        total_return = factor_returns.sum()
        sharpe_ratio = factor_returns.mean() / factor_returns.std() * np.sqrt(12)  # Annualized
        
        print(f"{factor_name} Factor - Total Return: {total_return:.4f}, Sharpe Ratio: {sharpe_ratio:.4f}")
        
        return factor_returns
    
    def evaluate_factors(self):
        """Evaluate all momentum and reversion factors."""
        print("\nEvaluating momentum factors...")
        
        # Evaluate momentum factors
        momentum_returns = {}
        for factor_name, factor_data in self.momentum_factors.items():
            momentum_returns[factor_name] = self.backtest_factor(factor_data, factor_name)
        
        # Find best momentum factor
        best_momentum = max(momentum_returns.items(), key=lambda x: x[1].sum())
        print(f"Best momentum factor: {best_momentum[0]}")
        
        print("\nEvaluating reversion factors...")
        
        # Evaluate reversion factors
        reversion_returns = {}
        for factor_name, factor_data in self.reversion_factors.items():
            reversion_returns[factor_name] = self.backtest_factor(factor_data, factor_name)
        
        # Find best reversion factor
        best_reversion = max(reversion_returns.items(), key=lambda x: x[1].sum())
        print(f"Best reversion factor: {best_reversion[0]}")
        
        # Plot factor returns
        plt.figure(figsize=(14, 7))
        
        plt.subplot(1, 2, 1)
        for name, returns in momentum_returns.items():
            plt.plot(returns.cumsum(), label=name)
        plt.title('Momentum Factors Cumulative Returns')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        for name, returns in reversion_returns.items():
            plt.plot(returns.cumsum(), label=name)
        plt.title('Reversion Factors Cumulative Returns')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return best_momentum[0], best_reversion[0]
    
    def prepare_fundamental_data(self):
        """Prepare fundamental data for machine learning."""
        print("\nPreparing fundamental data for prediction...")
        
        # Flatten multi-level index of fundamental data
        fund_data = self.fundamental_data.copy()
        fund_data.columns = [f"{col[0]}_{col[1]}" for col in fund_data.columns]
        
        # Normalize data cross-sectionally for each time period
        normalized_data = []
        
        for date in fund_data.index:
            # Get data for this date
            date_data = fund_data.loc[date]
            
            # Reshape to have sectors as rows and metrics as columns
            date_data = date_data.unstack().reset_index()
            date_data.columns = ['Metric', 'Sector', 'Value']
            date_data = date_data.pivot(index='Sector', columns='Metric', values='Value')
            
            # Normalize each metric
            for col in date_data.columns:
                date_data[col] = (date_data[col] - date_data[col].mean()) / date_data[col].std()
            
            # Add date and next period returns
            date_data['Date'] = date
            
            if date in self.returns.index:
                next_returns = self.returns.loc[date]
                for sector in date_data.index:
                    date_data.loc[sector, 'NextReturn'] = next_returns.get(sector, np.nan)
            
            normalized_data.append(date_data)
        
        # Combine all periods
        combined_data = pd.concat(normalized_data)
        combined_data = combined_data.dropna(subset=['NextReturn'])
        
        # Create target variable (1 for positive return, 0 for negative)
        combined_data['Target'] = (combined_data['NextReturn'] > 0).astype(int)
        
        # Drop unnecessary columns
        X = combined_data.drop(['Date', 'NextReturn', 'Target'], axis=1)
        y = combined_data['Target']
        
        return X, y
    
    def train_neural_network(self, X, y):
        """
        Train a neural network model for sector prediction.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
            
        Returns:
        --------
        model : MLPClassifier
            Trained neural network model
        """
        print("\nTraining neural network model...")
        
        # Split data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        # Hyperparameter tuning
        best_score = 0
        best_params = {}
        
        for hidden_layer_size in [(5,), (8,), (11,)]:
            for alpha in [0.001, 0.01, 0.1, 0.5, 1.0]:
                # Create and train model
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_size,
                    alpha=alpha,
                    solver='lbfgs',  # Limited-memory BFGS (quasi-Newton method)
                    activation='relu',
                    max_iter=1000,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                score = model.score(X_val, y_val)
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'hidden_layer_sizes': hidden_layer_size,
                        'alpha': alpha
                    }
        
        print(f"Best parameters: {best_params}")
        print(f"Best validation score: {best_score:.4f}")
        
        # Train final model with best parameters
        final_model = MLPClassifier(
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            alpha=best_params['alpha'],
            solver='lbfgs',
            activation='relu',
            max_iter=1000,
            random_state=42
        )
        
        final_model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_score = final_model.score(X_test, y_test)
        print(f"Test score: {test_score:.4f}")
        
        # Detailed evaluation
        y_pred = final_model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        self.model = final_model
        return final_model
    
    def backtest_model_strategy(self, X, y):
        """
        Backtest a trading strategy based on the trained model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
            
        Returns:
        --------
        strategy_returns : pd.Series
            Returns of the model-based strategy
        """
        print("\nBacktesting model-based strategy...")
        
        # Get prediction probabilities
        probs = self.model.predict_proba(X)[:, 1]  # Probability of positive return
        
        # Create a DataFrame with sectors and probabilities
        pred_df = pd.DataFrame({
            'Sector': X.index,
            'Probability': probs
        })
        
        # Group by date and get sector rankings
        strategy_returns = []
        dates = []
        
        # Use the index of X to group by date
        for date in X.index.get_level_values(0).unique():
            if date in self.returns.index:
                # Get sectors and probabilities for this date
                date_preds = pred_df[pred_df.index.get_level_values(0) == date]
                
                # Rank sectors by probability
                rankings = date_preds['Probability'].rank(ascending=False)
                
                # Take long positions in top 3 sectors
                long_sectors = rankings[rankings <= 3].index.get_level_values(1)
                
                # Take short positions in bottom 3 sectors
                short_sectors = rankings[rankings >= len(rankings) - 2].index.get_level_values(1)
                
                # Calculate next period returns
                next_returns = self.returns.loc[date]
                
                # Calculate strategy return
                long_return = next_returns[long_sectors].mean()
                short_return = -next_returns[short_sectors].mean()
                strategy_return = (long_return + short_return) / 2
                
                strategy_returns.append(strategy_return)
                dates.append(date)
        
        # Convert to Series
        strategy_returns = pd.Series(strategy_returns, index=dates)
        
        # Calculate statistics
        total_return = strategy_returns.sum()
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(12)  # Annualized
        
        print(f"Model Strategy - Total Return: {total_return:.4f}, Sharpe Ratio: {sharpe_ratio:.4f}")
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(strategy_returns.cumsum(), label='Model Strategy')
        
        # Add benchmark (equal-weighted sectors)
        benchmark_returns = self.returns.mean(axis=1)
        benchmark_returns = benchmark_returns[benchmark_returns.index.isin(strategy_returns.index)]
        plt.plot(benchmark_returns.cumsum(), label='Equal-Weighted Benchmark')
        
        plt.title('Model Strategy Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return strategy_returns
    
    def visualize_fundamental_analysis(self):
        """Visualize fundamental metrics for each sector."""
        print("\nVisualizing fundamental analysis...")
        
        # Select a few key metrics to visualize
        metrics = ['PE', 'PB', 'EV_EBITDA', 'DividendYield', 'GrossMargin', 'ROE']
        
        # Create a figure
        plt.figure(figsize=(15, 20))
        
        for i, metric in enumerate(metrics):
            plt.subplot(len(metrics), 1, i+1)
            
            # Extract data for this metric
            metric_data = self.fundamental_data.xs(metric, axis=1, level=1)
            
            # Plot each sector
            for sector in metric_data.columns:
                plt.plot(metric_data.index, metric_data[sector], label=sector_etfs[sector])
            
            plt.title(f'{metric} Ratio')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.grid(True)
            
            # Add legend to the first plot only
            if i == 0:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show()
    
    def run_strategy(self):
        """Run the complete sector rotation strategy."""
        # Get data
        self.get_data(use_simulated=True)
        
        # Calculate factors
        self.calculate_momentum_factors()
        self.calculate_reversion_factors()
        
        # Evaluate factors
        best_momentum, best_reversion = self.evaluate_factors()
        
        # Prepare fundamental data
        X, y = self.prepare_fundamental_data()
        
        # Train neural network
        self.train_neural_network(X, y)
        
        # Backtest model strategy
        self.backtest_model_strategy(X, y)
        
        # Visualize fundamental analysis
        self.visualize_fundamental_analysis()
        
        # Return the best factors
        return best_momentum, best_reversion


# Run the strategy
if __name__ == "__main__":
    # Create and run strategy
    strategy = SectorRotationStrategy(start_date='2010-01-01', end_date='2023-01-01', data_frequency='monthly')
    best_momentum, best_reversion = strategy.run_strategy()
    
    # Compare factor strategies and model strategy
    # This would be additional analysis comparing the performance of different approaches