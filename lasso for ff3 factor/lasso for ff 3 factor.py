import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV
import warnings
import requests
import io
from zipfile import ZipFile
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class IndustryReturnPredictor:
    def __init__(self, start_date='1960-01', end_date=None, use_synthetic=False):
        """Initialize the industry return predictor."""
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.to_datetime('today')
        self.use_synthetic = use_synthetic
        
        # Load data
        if use_synthetic:
            self.excess_returns = self.generate_synthetic_data()
            print(f"Using synthetic data with shape: {self.excess_returns.shape}")
        else:
            try:
                self.excess_returns = self.load_french_data()
                print(f"Successfully loaded Kenneth French data with shape: {self.excess_returns.shape}")
            except Exception as e:
                print(f"Error loading Kenneth French data: {e}")
                print("Falling back to synthetic data...")
                self.excess_returns = self.generate_synthetic_data()
                print(f"Using synthetic data with shape: {self.excess_returns.shape}")
        
        # Initialize models dictionary
        self.models = {}
        self.forecasts = {}
    
    def load_french_data(self):
        """Load and process Kenneth French's 30 Industry Portfolios data."""
        print("Loading Kenneth French's 30 Industry Portfolios data...")
        
        # Load industry returns
        industry_returns = self.download_industry_returns()
        print(f"Downloaded industry returns with shape: {industry_returns.shape}")
        
        # Load risk-free rate
        rf = self.download_risk_free_rate()
        print(f"Downloaded risk-free rate with shape: {rf.shape}")
        
        # Align dates and calculate excess returns
        excess_returns = self.calculate_excess_returns(industry_returns, rf)
        return excess_returns
    
    def download_industry_returns(self):
        """Download and process Kenneth French's 30 Industry Portfolios returns."""
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_CSV.zip"
        
        try:
            # Download the zip file
            response = requests.get(url)
            with ZipFile(io.BytesIO(response.content)) as zip_file:
                # Find the correct file in the zip
                for file_name in zip_file.namelist():
                    if '30_Industry_Portfolios.CSV' in file_name:
                        with zip_file.open(file_name) as csv_file:
                            # Read the content as text
                            content = csv_file.read().decode('utf-8')
                            
                            # Split into lines
                            lines = content.strip().split('\n')
                            
                            # Find where the data starts
                            start_line = None
                            for i, line in enumerate(lines):
                                if line.strip().startswith(('19', '20')) and ',' in line:  # Year starts with 19xx or 20xx
                                    start_line = i
                                    break
                            
                            if start_line is None:
                                raise ValueError("Could not identify the start of data in the industry returns file")
                            
                            # Extract column names from the header line (which should be right before the data)
                            header_line = lines[start_line - 1] if start_line > 0 else None
                            if header_line:
                                column_names = [col.strip() for col in header_line.split(',')[1:] if col.strip()]
                            else:
                                # Default column names if header can't be found
                                column_names = [f'Industry_{i+1}' for i in range(30)]
                            
                            # Parse the data
                            dates = []
                            data = []
                            
                            for line in lines[start_line:]:
                                if not line.strip() or line.startswith(';') or 'Copyright' in line:
                                    continue
                                
                                parts = line.split(',')
                                if len(parts) < 2:
                                    continue
                                
                                try:
                                    # Parse the date
                                    date_str = parts[0].strip()
                                    if len(date_str) == 6 and date_str.isdigit():  # Format: YYYYMM
                                        year = int(date_str[:4])
                                        month = int(date_str[4:])
                                        date = pd.Timestamp(year=year, month=month, day=1)
                                        
                                        # Parse return values
                                        values = []
                                        for val_str in parts[1:]:
                                            try:
                                                val = float(val_str.strip()) / 100.0  # Convert from percent to decimal
                                                values.append(val)
                                            except ValueError:
                                                values.append(np.nan)
                                        
                                        # Add to lists if we have the right number of values
                                        if len(values) >= len(column_names):
                                            dates.append(date)
                                            data.append(values[:len(column_names)])  # Take only the number of columns we need
                                except Exception as e:
                                    print(f"Error parsing line: {line[:50]}... - {str(e)}")
                            
                            # Create DataFrame
                            if not dates or not data:
                                raise ValueError("No valid data parsed from industry returns file")
                            
                            df = pd.DataFrame(data, index=dates, columns=column_names)
                            
                            # Handle duplicate dates if they exist
                            if df.index.duplicated().any():
                                print(f"Found {df.index.duplicated().sum()} duplicate dates in industry returns. Keeping first occurrences.")
                                df = df[~df.index.duplicated(keep='first')]
                            
                            return df
        
        except Exception as e:
            print(f"Error downloading industry returns: {e}")
            raise
    
    def download_risk_free_rate(self):
        """Download and process Kenneth French's Fama-French factors to get risk-free rate."""
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
        
        try:
            # Download the zip file
            response = requests.get(url)
            with ZipFile(io.BytesIO(response.content)) as zip_file:
                # Find the correct file in the zip
                for file_name in zip_file.namelist():
                    if 'F-F_Research_Data_Factors.CSV' in file_name:
                        with zip_file.open(file_name) as csv_file:
                            # Read the content as text
                            content = csv_file.read().decode('utf-8')
                            
                            # Split into lines
                            lines = content.strip().split('\n')
                            
                            # Find where the data starts
                            start_line = None
                            for i, line in enumerate(lines):
                                if line.strip().startswith(('19', '20')) and ',' in line:  # Year starts with 19xx or 20xx
                                    start_line = i
                                    break
                            
                            if start_line is None:
                                raise ValueError("Could not identify the start of data in the factors file")
                            
                            # Parse the data
                            dates = []
                            rf_values = []
                            
                            for line in lines[start_line:]:
                                if not line.strip() or line.startswith(';') or 'Copyright' in line:
                                    continue
                                
                                parts = line.split(',')
                                if len(parts) < 2:
                                    continue
                                
                                try:
                                    # Parse the date
                                    date_str = parts[0].strip()
                                    if len(date_str) == 6 and date_str.isdigit():  # Format: YYYYMM
                                        year = int(date_str[:4])
                                        month = int(date_str[4:])
                                        date = pd.Timestamp(year=year, month=month, day=1)
                                        
                                        # Parse RF value
                                        rf_str = parts[1].strip()
                                        rf = float(rf_str) / 100.0  # Convert from percent to decimal
                                        
                                        dates.append(date)
                                        rf_values.append(rf)
                                except Exception as e:
                                    print(f"Error parsing line: {line[:50]}... - {str(e)}")
                            
                            # Create Series
                            if not dates or not rf_values:
                                raise ValueError("No valid data parsed from factors file")
                            
                            series = pd.Series(rf_values, index=dates, name='RF')
                            
                            # Handle duplicate dates if they exist
                            if series.index.duplicated().any():
                                print(f"Found {series.index.duplicated().sum()} duplicate dates in risk-free rate. Keeping first occurrences.")
                                series = series[~series.index.duplicated(keep='first')]
                            
                            return series
        
        except Exception as e:
            print(f"Error downloading risk-free rate: {e}")
            raise
    
    def calculate_excess_returns(self, industry_returns, rf):
        """Calculate excess returns by aligning industry returns with risk-free rate."""
        print("Calculating excess returns...")
        
        # Filter date range
        industry_returns = industry_returns[
            (industry_returns.index >= self.start_date) & 
            (industry_returns.index <= self.end_date)
        ]
        
        rf = rf[
            (rf.index >= self.start_date) & 
            (rf.index <= self.end_date)
        ]
        
        print(f"After date filtering: Industry returns shape: {industry_returns.shape}, RF shape: {rf.shape}")
        
        # Align dates (find the intersection)
        common_dates = industry_returns.index.intersection(rf.index)
        
        if len(common_dates) == 0:
            raise ValueError("No common dates between industry returns and risk-free rate")
        
        print(f"Found {len(common_dates)} common dates")
        
        # Filter to common dates
        industry_returns = industry_returns.loc[common_dates]
        rf = rf.loc[common_dates]
        
        # Calculate excess returns
        excess_returns = pd.DataFrame(index=common_dates)
        
        # Process column by column to avoid broadcasting issues
        for col in industry_returns.columns:
            excess_returns[col] = industry_returns[col] - rf
        
        print(f"Final excess returns shape: {excess_returns.shape}")
        return excess_returns
    
    def generate_synthetic_data(self):
        """Generate synthetic industry return data."""
        print("Generating synthetic industry return data...")
        
        # Generate date range
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        
        # Generate 30 synthetic industries (similar to Kenneth French's 30 Industry Portfolios)
        n_industries = 30
        industry_names = [
            'Food', 'Beer', 'Smoke', 'Games', 'Books', 'Hshld', 'Clths', 'Hlth', 'Chems', 'Txtls',
            'Cnstr', 'Steel', 'FabPr', 'ElcEq', 'Autos', 'Carry', 'Mines', 'Coal', 'Oil', 'Util',
            'Telcm', 'Servs', 'BusEq', 'Paper', 'Trans', 'Whlsl', 'Rtail', 'Meals', 'Fin', 'Other'
        ]
        
        # Create synthetic excess returns with realistic correlations and volatilities
        np.random.seed(42)
        
        # Generate market factor
        T = len(dates)
        market = np.random.normal(0.005, 0.045, T)  # Market factor: 6% annual mean, 15% annual vol
        
        # Generate industry-specific factors
        industry_betas = np.random.uniform(0.5, 1.5, n_industries)  # Industry betas
        industry_vols = np.random.uniform(0.02, 0.05, n_industries)  # Industry-specific vol
        
        # Create correlated returns
        excess_returns = np.zeros((T, n_industries))
        for i in range(n_industries):
            # Each industry = beta * market + specific factor
            industry_specific = np.random.normal(0, industry_vols[i], T)
            excess_returns[:, i] = industry_betas[i] * market + industry_specific
        
        # Create DataFrame with proper dates and industry names
        return pd.DataFrame(excess_returns, index=dates, columns=industry_names)
    
    def compute_descriptive_stats(self):
        """Compute descriptive statistics for the excess returns."""
        stats = pd.DataFrame({
            'Ann Mean': self.excess_returns.mean() * 12 * 100,  # Annualized mean in %
            'Ann Vol': self.excess_returns.std() * np.sqrt(12) * 100,  # Annualized volatility in %
            'Min': self.excess_returns.min() * 100,  # Minimum monthly return in %
            'Max': self.excess_returns.max() * 100,  # Maximum monthly return in %
        })
        
        # Compute annualized Sharpe ratios
        stats['Ann Sharpe Ratio'] = stats['Ann Mean'] / stats['Ann Vol']
        
        return stats
    
    def fit_ols_post_lasso(self, y, X, use_cv=True, cv=5):
        """Fit a LASSO model to select variables, then refit with OLS."""
        # Fit LASSO to select variables
        if use_cv:
            lasso_cv = LassoCV(cv=cv, random_state=42, max_iter=10000)
            lasso_cv.fit(X, y)
            alpha = lasso_cv.alpha_
        else:
            alpha = 0.01  # Default value if not using CV
        
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        lasso.fit(X, y)
        
        # Get selected features
        selected_features = np.where(lasso.coef_ != 0)[0]
        
        if len(selected_features) == 0:
            # If no features are selected, return empty results
            return [], [], None
        
        # Refit with OLS using selected features
        X_selected = X[:, selected_features]
        X_selected_with_const = sm.add_constant(X_selected)
        ols_model = sm.OLS(y, X_selected_with_const).fit()
        
        # Return feature indices and OLS coefficients
        return selected_features, ols_model.params[1:], ols_model
    
    def fit_predictive_models(self, train_data, method='ols_post_lasso'):
        """Fit predictive models for each industry using lagged returns."""
        # Prepare lagged predictors
        X = train_data.values[:-1]  # All but the last observation
        
        models_dict = {}
        selected_features_dict = {}
        
        for i, industry in enumerate(train_data.columns):
            # Target variable is next period's return
            y = train_data[industry].values[1:]  # All but the first observation
            
            if method == 'prevailing_mean':
                # Just use historical mean
                models_dict[industry] = {'mean': y.mean()}
                selected_features_dict[industry] = []
                
            elif method == 'ols_post_lasso':
                # Fit OLS post LASSO
                selected, coef, model = self.fit_ols_post_lasso(y, X)
                
                if len(selected) == 0:
                    # If no features were selected, just use the mean
                    models_dict[industry] = {'mean': y.mean(), 'model': None}
                else:
                    # Store selected features and coefficients
                    models_dict[industry] = {
                        'intercept': model.params[0] if model else y.mean(),
                        'coef': coef,
                        'model': model,
                        'selected': selected
                    }
                selected_features_dict[industry] = [(train_data.columns[j], coef[i]) 
                                                  for i, j in enumerate(selected)]
            
        return models_dict, selected_features_dict
    
    def predict(self, models_dict, last_returns, method='ols_post_lasso'):
        """Generate predictions for each industry based on the fitted models."""
        predictions = {}
        
        for industry in models_dict:
            if method == 'prevailing_mean':
                predictions[industry] = models_dict[industry]['mean']
                
            elif method == 'ols_post_lasso':
                if 'selected' not in models_dict[industry] or len(models_dict[industry].get('selected', [])) == 0:
                    # If no features were selected, use the mean
                    predictions[industry] = models_dict[industry]['mean'] if 'mean' in models_dict[industry] else 0
                else:
                    # Get selected features
                    selected = models_dict[industry]['selected']
                    X_selected = last_returns.values[selected]
                    
                    # Generate prediction
                    intercept = models_dict[industry]['intercept']
                    coef = models_dict[industry]['coef']
                    predictions[industry] = intercept + np.dot(X_selected, coef)
        
        return pd.Series(predictions)
    
    def generate_out_of_sample_forecasts(self, method='ols_post_lasso', start_year=1970):
        """Generate out-of-sample forecasts using the specified method."""
        print(f"Generating out-of-sample forecasts using {method} from {start_year}...")
        
        # Set the start date for forecasting
        forecast_start = pd.Timestamp(f"{start_year}-01-01")
        
        # Filter data
        excess_returns = self.excess_returns.copy()
        
        # Initialize results container
        forecasts = pd.DataFrame(index=excess_returns.index, columns=excess_returns.columns)
        
        # Iterate through each month
        for t, current_date in enumerate(excess_returns.index):
            if current_date < forecast_start:
                continue
                
            # Define the training window (expanding window)
            train_data = excess_returns.loc[:current_date]
            
            # Fit models using training data
            models_dict, _ = self.fit_predictive_models(train_data, method=method)
            
            # Last observed returns
            last_returns = train_data.iloc[-1]
            
            # Generate predictions for next month
            preds = self.predict(models_dict, last_returns, method=method)
            
            # Store forecasts for the next month
            next_dates = excess_returns.index[excess_returns.index > current_date]
            if len(next_dates) > 0:
                next_date = next_dates[0]
                forecasts.loc[next_date] = preds
                
            # Print progress every 5 years
            if current_date.month == 12 and current_date.year % 5 == 0:
                print(f"  Generated forecasts up to {current_date.strftime('%Y-%m')}")
        
        # Remove rows with all NaNs
        forecasts = forecasts.dropna(how='all')
        
        # Store forecasts
        self.forecasts[method] = forecasts
        
        return forecasts
    
    def construct_industry_rotation_portfolio(self, forecasts, n_top=6, n_bottom=6):
        """Construct a long-short industry rotation portfolio based on forecasts."""
        # Initialize portfolio returns
        portfolio_returns = pd.Series(index=forecasts.index, dtype=float)
        
        # Get actual returns
        actual_returns = self.excess_returns.copy()
        
        # For each date in forecasts
        for date in forecasts.index:
            # Skip if this is the last date (no future returns)
            if date >= actual_returns.index[-1]:
                continue
            
            # Get forecasts for current date
            current_forecasts = forecasts.loc[date].dropna()
            
            if len(current_forecasts) < n_top + n_bottom:
                continue
            
            # Sort industries by forecasted returns
            sorted_industries = current_forecasts.sort_values(ascending=False)
            
            # Select top and bottom industries
            top_industries = sorted_industries.index[:n_top]
            bottom_industries = sorted_industries.index[-n_bottom:]
            
            # Find the next date in actual returns
            next_dates = actual_returns.index[actual_returns.index > date]
            if len(next_dates) == 0:
                continue
                
            next_date = next_dates[0]
            
            # Calculate portfolio return (equal-weighted long-short)
            long_return = actual_returns.loc[next_date, top_industries].mean()
            short_return = actual_returns.loc[next_date, bottom_industries].mean()
            portfolio_return = long_return - short_return
            
            # Store portfolio return
            portfolio_returns[next_date] = portfolio_return
        
        return portfolio_returns
    
    def evaluate_portfolio(self, portfolio_returns):
        """Evaluate the performance of a portfolio."""
        # Remove NaN values
        portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) == 0:
            return {"error": "No valid portfolio returns"}
        
        # Calculate performance metrics
        ann_factor = 12  # Monthly to annual conversion
        
        mean_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        
        # Annualized metrics
        ann_return = mean_return * ann_factor
        ann_vol = volatility * np.sqrt(ann_factor)
        
        # Sharpe ratio (assuming zero risk-free rate since we're using excess returns)
        sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Downside risk
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_risk = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else 0
        sortino_ratio = ann_return / downside_risk if downside_risk > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        max_return = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / max_return - 1)
        max_drawdown = drawdown.min()
        
        # Compile metrics
        perf_metrics = {
            "Ann. Mean": ann_return * 100,  # in percent
            "Ann. Vol": ann_vol * 100,  # in percent
            "Ann. Sharpe Ratio": sharpe_ratio,
            "Ann. Downside Risk": downside_risk * 100,  # in percent
            "Ann. Sortino Ratio": sortino_ratio,
            "Max Drawdown": max_drawdown * 100,  # in percent
        }
        
        return perf_metrics
    
    def plot_portfolio_performance(self, portfolio_returns, title="Industry Rotation Portfolio", method_name="LASSO"):
        """Plot the cumulative performance of the portfolio."""
        # Remove NaN values
        portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) == 0:
            print("No valid portfolio returns to plot")
            return None
        
        # Calculate cumulative returns
        portfolio_cum_return = (1 + portfolio_returns).cumprod()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot portfolio performance
        ax.plot(portfolio_returns.index, portfolio_cum_return, 
                label=f"{method_name} Industry Rotation", linewidth=2)
        
        # Add title and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cumulative Return", fontsize=12)
        
        # Add legend and grid
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def run_full_analysis(self, methods=['ols_post_lasso', 'prevailing_mean'], start_year=1970, n_top=6, n_bottom=6):
        """Run a complete analysis pipeline."""
        results = {}
        
        # Compute descriptive statistics
        print("\nComputing descriptive statistics...")
        stats = self.compute_descriptive_stats()
        print(stats.round(2))
        results['descriptive_stats'] = stats
        
        # Generate forecasts and construct portfolios for each method
        for method in methods:
            print(f"\nAnalyzing method: {method}")
            
            # Generate out-of-sample forecasts
            forecasts = self.generate_out_of_sample_forecasts(method=method, start_year=start_year)
            
            # Construct industry rotation portfolio
            portfolio_returns = self.construct_industry_rotation_portfolio(
                forecasts, n_top=n_top, n_bottom=n_bottom)
            
            # Evaluate portfolio performance
            perf_metrics = self.evaluate_portfolio(portfolio_returns)
            
            # Print performance metrics
            print(f"\nPerformance Metrics for {method}:")
            for metric, value in perf_metrics.items():
                print(f"  {metric}: {value:.2f}")
            
            # Plot portfolio performance
            fig = self.plot_portfolio_performance(portfolio_returns, method_name=method)
            
            # Store results
            results[method] = {
                'forecasts': forecasts,
                'portfolio_returns': portfolio_returns,
                'performance_metrics': perf_metrics,
                'figure': fig
            }
            
            # Save figure
            fig_filename = f"industry_rotation_{method}.png"
            fig.savefig(fig_filename)
            print(f"Figure saved as {fig_filename}")
        
        # Compare methods
        if len(methods) > 1:
            print("\nComparing Methods:")
            comparison = pd.DataFrame({method: results[method]['performance_metrics'] 
                                      for method in methods})
            print(comparison.round(2))
            results['comparison'] = comparison
        
        return results

# Example usage
if __name__ == "__main__":
    # Create predictor with real data (falling back to synthetic if needed)
    predictor = IndustryReturnPredictor(start_date='1960-01', end_date='2022-12-31', use_synthetic=False)
    
    # Run the full analysis
    results = predictor.run_full_analysis(
        methods=['ols_post_lasso', 'prevailing_mean'], 
        start_year=1970,
        n_top=6, 
        n_bottom=6
    )
    
    # Show the plots
    plt.show()
    