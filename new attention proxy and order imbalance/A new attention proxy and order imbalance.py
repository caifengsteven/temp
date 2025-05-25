import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm
import seaborn as sns
from datetime import datetime, timedelta

np.random.seed(42)  # For reproducibility

# Set plot style - use a style available in older matplotlib versions
try:
    plt.style.use('seaborn-darkgrid')  # Try this style for older versions
except:
    try:
        plt.style.use('ggplot')  # Backup style
    except:
        pass  # Just use default style if none of these work

# Set a colorblind-friendly palette
sns.set_palette("colorblind")

# Simulation parameters
num_stocks = 200
num_days = 500
start_date = datetime(2019, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(num_days)]
trading_dates = [date for date in dates if date.weekday() < 5]  # Only weekdays

# Create DataFrame for market data
def create_market_data(num_stocks, trading_dates):
    all_data = []
    
    for stock_id in range(1, num_stocks + 1):
        # Generate basic stock properties
        market_cap = np.random.lognormal(10, 1.5)  # Market capitalization
        book_value = market_cap * np.random.uniform(0.5, 1.5)  # Book value
        initial_price = np.random.uniform(10, 100)  # Initial price
        
        # Fundamental volatility and drift parameters
        vol = np.random.uniform(0.01, 0.03)  # Daily volatility
        drift = np.random.uniform(0.0001, 0.0005)  # Daily drift
        
        # Simulate price movement
        prices = [initial_price]
        returns = []
        
        for i in range(1, len(trading_dates)):
            # Basic random walk
            daily_return = np.random.normal(drift, vol)
            # Add some autocorrelation for realism
            if i > 1 and np.random.random() < 0.3:  # 30% chance of momentum
                daily_return += 0.2 * returns[-1]
            
            returns.append(daily_return)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # Ensure first return is defined
        returns.insert(0, 0)
        
        # Generate trading volume with some relationship to volatility and price
        base_volume = market_cap / initial_price * 0.02  # 2% daily turnover on average
        volumes = []
        
        for i in range(len(trading_dates)):
            # Volume increases with volatility and has autocorrelation
            if i > 0:
                vol_factor = 1 + 5 * abs(returns[i])  # More volume during big price moves
                auto_factor = 0.6 + 0.4 * (volumes[-1] / base_volume)  # Some autocorrelation
                daily_volume = base_volume * vol_factor * auto_factor * np.random.lognormal(0, 0.3)
            else:
                daily_volume = base_volume * np.random.lognormal(0, 0.3)
            
            volumes.append(daily_volume)
        
        # Calculate turnover
        turnover = [vol / market_cap for vol in volumes]
        
        # Calculate book-to-market ratio
        bm_ratio = book_value / (prices[-1] * base_volume / turnover[-1])
        
        # Simulate spread as a function of price and volume
        spreads = [np.random.uniform(0.01, 0.05) * p / np.sqrt(v) for p, v in zip(prices, volumes)]
        
        # Create DataFrame for this stock
        stock_df = pd.DataFrame({
            'stock_id': stock_id,
            'date': trading_dates,
            'price': prices,
            'return': returns,
            'volume': volumes,
            'turnover': turnover,
            'spread': spreads,
            'market_cap': market_cap,
            'book_value': book_value,
            'bm_ratio': bm_ratio
        })
        
        all_data.append(stock_df)
    
    # Combine all stocks
    market_data = pd.concat(all_data, ignore_index=True)
    return market_data

def simulate_attention_proxies(market_data):
    """
    Simulate attention proxies for each stock over time.
    
    1. GB1: Logarithmic value of daily reading quantity of posts
    2. Ab_volume: Abnormal trading volume
    3. Abs_ab_ret: Absolute value of abnormal return
    4. Ab_News1: Abnormal reading quantity of news
    """
    # Group by stock
    grouped = market_data.groupby('stock_id')
    
    # Lists to store the data
    all_attention_data = []
    
    for stock_id, stock_df in grouped:
        stock_df = stock_df.sort_values('date').reset_index(drop=True)
        
        # Base metrics for attention calculated from market data
        abs_returns = np.abs(stock_df['return'])
        log_volume = np.log(stock_df['volume'])
        
        # Generate GB1 - Base it on returns and volume with noise
        # GB1 should be correlated with volume and return but lead them slightly
        gb1_base = 0.7 * log_volume + 1.5 * abs_returns
        gb1_noise = np.random.normal(0, 0.5, len(stock_df))
        gb1_trend = np.linspace(0, 0.5, len(stock_df))  # Increasing trend of social media usage
        
        # Add some autocorrelation in GB1
        gb1 = [gb1_base[0] + gb1_noise[0] + gb1_trend[0]]
        for i in range(1, len(stock_df)):
            # Previous day's GB1 has some influence
            gb1_val = 0.6 * gb1_base[i] + 0.2 * gb1[-1] + gb1_noise[i] + gb1_trend[i]
            gb1.append(gb1_val)
        
        gb1 = np.array(gb1)
        
        # Generate News1 - Less frequent than GB1 but correlated
        news_freq = np.random.binomial(1, 0.3, len(stock_df))  # News occurs 30% of days
        news1 = np.where(news_freq == 1, gb1 * 0.5 + np.random.normal(0, 0.8, len(stock_df)), 0)
        
        # Calculate abnormal GB1 (Ab_GB1)
        ab_gb1 = []
        for i in range(len(stock_df)):
            if i < 10:
                # For first 10 days, just use deviation from average
                ab_gb1.append(gb1[i] - np.mean(gb1[:i+1]))
            else:
                # Abnormal GB1 according to the formula in the paper
                baseline = np.median(gb1[i-10:i])
                ab_gb1.append(gb1[i] - baseline)
        
        # Calculate abnormal volume (Ab_volume)
        ab_volume = []
        volume_window = 244  # As in the paper
        
        for i in range(len(stock_df)):
            if i < volume_window:
                # For the first window, use deviation from available data
                if i == 0:
                    ab_volume.append(0)
                else:
                    ab_volume.append(log_volume[i] - np.mean(log_volume[:i]))
            else:
                # Abnormal volume according to the formula in the paper
                ab_volume.append(log_volume[i] - np.mean(log_volume[i-volume_window:i]))
        
        # Calculate absolute abnormal return (Abs_ab_ret)
        rm = np.mean(stock_df['return'])
        abs_ab_ret = np.abs(stock_df['return'] - rm)
        
        # Calculate abnormal News1 (Ab_News1)
        ab_news1 = []
        for i in range(len(stock_df)):
            if i < 10:
                # For first 10 days, just use deviation from average
                news_vals = [n for j, n in enumerate(news1[:i+1]) if news_freq[j] == 1]
                if news_vals:
                    ab_news1.append(news1[i] - np.mean(news_vals))
                else:
                    ab_news1.append(0)
            else:
                # News baseline based on median of last 10 days with news
                news_vals = [n for j, n in enumerate(news1[i-10:i]) if news_freq[j] == 1]
                if news_vals and news_freq[i] == 1:
                    ab_news1.append(news1[i] - np.median(news_vals))
                else:
                    ab_news1.append(0)
        
        # Create DataFrame for this stock
        attention_df = pd.DataFrame({
            'stock_id': stock_id,
            'date': stock_df['date'],
            'GB1': gb1,
            'Ab_GB1': ab_gb1,
            'Ab_volume': ab_volume,
            'Abs_ab_ret': abs_ab_ret,
            'News1': news1,
            'Ab_News1': ab_news1,
            'News_freq': news_freq
        })
        
        all_attention_data.append(attention_df)
    
    # Combine all stocks
    attention_data = pd.concat(all_attention_data, ignore_index=True)
    return attention_data

def simulate_investor_trading(data):
    """
    Simulate trading behavior for different investor types:
    - 5 categories of individual investors (Ind1-Ind5)
    - 6 categories of professional institutional investors (Pro1-Pro6)
    - Ordinary institutional investors (OI)
    """
    # Group by stock and date
    grouped = data.groupby(['stock_id', 'date'])
    
    # Lists to store the trading data
    all_trading_data = []
    
    for (stock_id, date), group in tqdm(grouped, desc="Simulating investor trading"):
        # Get attention values and market data for this stock-date
        ab_gb1 = group['Ab_GB1'].values[0]
        price = group['price'].values[0]
        market_cap = group['market_cap'].values[0]
        bm_ratio = group['bm_ratio'].values[0]
        spread = group['spread'].values[0]
        turnover = group['turnover'].values[0]
        volume = group['volume'].values[0]
        
        # Base volumes for different investor types (based on description in the paper)
        # Individual investors with smaller capital have higher trading frequency
        ind_base_volumes = {
            'Ind1': volume * 0.35,  # Smallest individual investors
            'Ind2': volume * 0.25,
            'Ind3': volume * 0.15,
            'Ind4': volume * 0.05,
            'Ind5': volume * 0.05   # Largest individual investors
        }
        
        pro_base_volumes = {
            'Pro1': volume * 0.05,  # Investment Funds
            'Pro2': volume * 0.02,  # QFII
            'Pro3': volume * 0.01,  # Insurance Funds
            'Pro4': volume * 0.02,  # Self-Brokerage
            'Pro5': volume * 0.01,  # Asset Management
            'Pro6': volume * 0.01   # Social Insurance Funds
        }
        
        oi_base_volume = volume * 0.03  # Ordinary institutional investors
        
        # Define how different investors react to attention (based on the paper's findings)
        # Individual investors with small capital are more likely to be net buyers with attention
        # Professional investors are more likely to be net sellers with attention
        
        ind_attention_bias = {
            'Ind1': 1.5,    # Strong positive bias (buy with attention)
            'Ind2': 1.7,
            'Ind3': 1.0,
            'Ind4': 0.4,
            'Ind5': -0.2    # Slight negative bias (sell with attention)
        }
        
        pro_attention_bias = {
            'Pro1': -4.2,   # Strong negative bias (sell with attention)
            'Pro2': -3.1,
            'Pro3': -5.1,
            'Pro4': -1.8,
            'Pro5': -1.5,
            'Pro6': -8.7
        }
        
        oi_attention_bias = -1.8  # Negative bias for ordinary institutional investors
        
        # Trading results
        trading_results = {
            'stock_id': stock_id,
            'date': date
        }
        
        # Generate buy/sell volumes for each investor type based on their biases
        for investor, bias in ind_attention_bias.items():
            base_vol = ind_base_volumes[investor]
            # Attention effect (scaled by ab_gb1 and bias)
            attention_effect = bias * ab_gb1 / 100
            
            # Apply other effects based on paper findings
            cap_effect = -0.3 * np.log(market_cap) / 100  # Small cap preference
            bm_effect = 0.8 * bm_ratio / 100  # High B/M preference
            spread_effect = 0.2 * spread / 100  # High spread preference  
            turnover_effect = -0.4 * turnover / 100  # Low turnover preference
            
            # Total effects
            total_effect = attention_effect + cap_effect + bm_effect + spread_effect + turnover_effect
            
            # Convert to order imbalance (-1 to 1)
            imbalance = np.tanh(total_effect)
            
            # Apply some noise
            imbalance += np.random.normal(0, 0.1)
            imbalance = np.clip(imbalance, -0.9, 0.9)  # Ensure we don't get extreme imbalances
            
            # Calculate buy and sell volumes
            total_vol = base_vol * (1 + abs(imbalance) * 0.2)  # More attention = more trading
            if imbalance > 0:
                buy_vol = total_vol * (1 + imbalance) / 2
                sell_vol = total_vol * (1 - imbalance) / 2
            else:
                buy_vol = total_vol * (1 + imbalance) / 2
                sell_vol = total_vol * (1 - imbalance) / 2
            
            # Add to results
            trading_results[f'{investor}_buy'] = buy_vol
            trading_results[f'{investor}_sell'] = sell_vol
            trading_results[f'{investor}_im'] = (buy_vol - sell_vol) / (buy_vol + sell_vol)
        
        # Repeat for professional investors
        for investor, bias in pro_attention_bias.items():
            base_vol = pro_base_volumes[investor]
            # Attention effect
            attention_effect = bias * ab_gb1 / 100
            
            # Professional investors have different behaviors (based on the paper)
            cap_effect = 0.1 * np.log(market_cap) / 100  # Very slight large cap preference
            bm_effect = -0.5 * bm_ratio / 100  # Slight low B/M preference
            spread_effect = -0.2 * spread / 100  # Slight low spread preference  
            turnover_effect = 1.0 * turnover / 100  # High turnover preference
            
            # Total effects
            total_effect = attention_effect + cap_effect + bm_effect + spread_effect + turnover_effect
            
            # Convert to order imbalance (-1 to 1)
            imbalance = np.tanh(total_effect)
            
            # Apply some noise
            imbalance += np.random.normal(0, 0.1)
            imbalance = np.clip(imbalance, -0.9, 0.9)
            
            # Calculate buy and sell volumes
            total_vol = base_vol * (1 + abs(imbalance) * 0.2)
            if imbalance > 0:
                buy_vol = total_vol * (1 + imbalance) / 2
                sell_vol = total_vol * (1 - imbalance) / 2
            else:
                buy_vol = total_vol * (1 + imbalance) / 2
                sell_vol = total_vol * (1 - imbalance) / 2
            
            # Add to results
            trading_results[f'{investor}_buy'] = buy_vol
            trading_results[f'{investor}_sell'] = sell_vol
            trading_results[f'{investor}_im'] = (buy_vol - sell_vol) / (buy_vol + sell_vol)
        
        # Ordinary institutional investors
        base_vol = oi_base_volume
        attention_effect = oi_attention_bias * ab_gb1 / 100
        
        # OI investors behaviors
        cap_effect = 0.2 * np.log(market_cap) / 100
        bm_effect = 0.5 * bm_ratio / 100
        spread_effect = 0.0 * spread / 100
        turnover_effect = -0.2 * turnover / 100
        
        # Total effects
        total_effect = attention_effect + cap_effect + bm_effect + spread_effect + turnover_effect
        
        # Convert to order imbalance (-1 to 1)
        imbalance = np.tanh(total_effect)
        
        # Apply some noise
        imbalance += np.random.normal(0, 0.1)
        imbalance = np.clip(imbalance, -0.9, 0.9)
        
        # Calculate buy and sell volumes
        total_vol = base_vol * (1 + abs(imbalance) * 0.2)
        if imbalance > 0:
            buy_vol = total_vol * (1 + imbalance) / 2
            sell_vol = total_vol * (1 - imbalance) / 2
        else:
            buy_vol = total_vol * (1 + imbalance) / 2
            sell_vol = total_vol * (1 - imbalance) / 2
        
        # Add to results
        trading_results[f'OI_buy'] = buy_vol
        trading_results[f'OI_sell'] = sell_vol
        trading_results[f'OI_im'] = (buy_vol - sell_vol) / (buy_vol + sell_vol)
        
        all_trading_data.append(trading_results)
    
    # Convert to DataFrame
    trading_data = pd.DataFrame(all_trading_data)
    return trading_data

def analyze_attention_proxies(data):
    """
    Analyze the correlation between different attention proxies
    """
    attention_proxies = ['Ab_GB1', 'Ab_volume', 'Abs_ab_ret', 'Ab_News1']
    
    # Calculate correlations
    corr_matrix = data[attention_proxies].corr()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Between Attention Proxies', fontsize=16)
    plt.tight_layout()
    plt.savefig("correlation_attention_proxies.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix

def run_var_analysis(data):
    """
    Run a VAR analysis to study the lead-lag relationship between attention proxies
    """
    # Prepare data
    attention_proxies = ['Ab_GB1', 'Ab_volume', 'Abs_ab_ret', 'Ab_News1']
    
    # Group by stock to run VAR for each stock
    grouped = data.groupby('stock_id')
    
    # Store results
    var_results = {proxy: [] for proxy in attention_proxies}
    
    for stock_id, stock_data in grouped:
        # Sort by date
        stock_data = stock_data.sort_values('date')
        
        # Extract attention proxies
        proxy_data = stock_data[attention_proxies]
        
        # Lag the data
        lagged_data = proxy_data.shift(1)
        lagged_data.columns = [f"{col}_lag" for col in lagged_data.columns]
        
        # Combine current and lagged data
        combined_data = pd.concat([proxy_data, lagged_data], axis=1)
        combined_data = combined_data.dropna()
        
        # Run regressions for each proxy
        for proxy in attention_proxies:
            X = combined_data[[f"{col}_lag" for col in attention_proxies]]
            X = sm.add_constant(X)
            y = combined_data[proxy]
            
            # Run regression
            model = sm.OLS(y, X).fit()
            
            # Store coefficients
            var_results[proxy].append(model.params.values)
    
    # Average coefficients across stocks
    avg_coefficients = {}
    std_coefficients = {}
    
    for proxy in attention_proxies:
        avg_coefficients[proxy] = np.mean(var_results[proxy], axis=0)
        std_coefficients[proxy] = np.std(var_results[proxy], axis=0) / np.sqrt(len(var_results[proxy]))
    
    # Create VAR coefficient table
    var_table = pd.DataFrame(index=['constant'] + [f"{proxy}_lag" for proxy in attention_proxies])
    
    for proxy in attention_proxies:
        var_table[proxy] = avg_coefficients[proxy]
    
    # Create figure with coefficients
    plt.figure(figsize=(12, 10))
    
    # Plot the coefficients as a heatmap
    values = var_table.values[1:, :]  # Skip the constant
    sns.heatmap(values, annot=True, fmt=".3f", cmap='coolwarm', center=0,
                xticklabels=attention_proxies, yticklabels=[f"{proxy}_lag" for proxy in attention_proxies])
    
    plt.title('VAR Coefficients for Attention Proxies', fontsize=16)
    plt.xlabel('Dependent Variables', fontsize=14)
    plt.ylabel('Lagged Independent Variables', fontsize=14)
    plt.tight_layout()
    plt.savefig("var_coefficients.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return var_table

def analyze_attention_returns(data):
    """
    Analyze the impact of attention proxies on current and future returns
    
    Fixed version to handle varying coefficient shapes
    """
    # Standardize attention proxies
    standardized_data = data.copy()
    
    attention_proxies = ['Ab_GB1', 'Ab_volume', 'Abs_ab_ret', 'Ab_News1']
    scaler = StandardScaler()
    
    for proxy in attention_proxies:
        standardized_data[f"{proxy}_std"] = scaler.fit_transform(data[proxy].values.reshape(-1, 1))
    
    # Calculate abnormal returns
    # First compute average return per date
    daily_avg_returns = data.groupby('date')['return'].mean().reset_index()
    daily_avg_returns.columns = ['date', 'avg_return']
    
    # Merge with original data
    standardized_data = pd.merge(standardized_data, daily_avg_returns, on='date')
    
    # Calculate abnormal return
    standardized_data['abnormal_return'] = standardized_data['return'] - standardized_data['avg_return']
    
    # Calculate future abnormal returns
    for i in range(1, 5):
        standardized_data[f'abnormal_return_t+{i}'] = standardized_data.groupby('stock_id')['abnormal_return'].shift(-i)
    
    # Run Fama-MacBeth regressions
    fama_macbeth_results = []
    
    for day in range(5):  # t to t+4
        if day == 0:
            target = 'abnormal_return'
        else:
            target = f'abnormal_return_t+{day}'
        
        # Group by date
        grouped = standardized_data.groupby('date')
        
        daily_results = []
        
        for date, day_data in grouped:
            # Run cross-sectional regression
            X = day_data[[f"{proxy}_std" for proxy in attention_proxies]]
            X = sm.add_constant(X)
            y = day_data[target]
            
            # Skip dates with too few data points
            if len(day_data) < 20 or X.isnull().any().any() or y.isnull().any():
                continue
            
            try:
                model = sm.OLS(y, X).fit()
                # Store coefficients with explicit shape to ensure consistency
                # Store exactly 5 coefficients (constant + 4 proxies)
                coeffs = model.params
                if len(coeffs) == 5:  # Check if we have the expected number of coefficients
                    daily_results.append(coeffs.values)
            except:
                continue
        
        # Skip days with no results
        if not daily_results:
            print(f"Warning: No valid regression results for day {day}")
            fama_macbeth_results.append({
                'day': day,
                'coefficients': np.zeros(5),
                'std_errors': np.zeros(5),
                't_stats': np.zeros(5)
            })
            continue
        
        # Convert to numpy array with explicit shape checking
        daily_results_array = np.array(daily_results)
        
        # Calculate average coefficients across days
        avg_coefficients = np.mean(daily_results_array, axis=0)
        std_coefficients = np.std(daily_results_array, axis=0) / np.sqrt(len(daily_results_array))
        
        # Calculate t-statistics
        t_stats = avg_coefficients / std_coefficients
        
        # Store results
        fama_macbeth_results.append({
            'day': day,
            'coefficients': avg_coefficients,
            'std_errors': std_coefficients,
            't_stats': t_stats
        })
    
    # Create table of results
    fm_table = pd.DataFrame(index=['constant'] + [f"{proxy}_std" for proxy in attention_proxies])
    
    for i, result in enumerate(fama_macbeth_results):
        if i == 0:
            col_name = 'return_t'
        else:
            col_name = f'return_t+{i}'
        
        fm_table[col_name] = result['coefficients']
    
    # Plot the coefficients over time
    plt.figure(figsize=(12, 10))
    
    # Ensure we have proper data to plot
    valid_days = [result['day'] for result in fama_macbeth_results if not np.isnan(result['coefficients']).any()]
    
    for i in range(1, 5):  # Skip constant
        proxy = attention_proxies[i-1]
        coefs = []
        for result in fama_macbeth_results:
            if i < len(result['coefficients']):
                coefs.append(result['coefficients'][i])
            else:
                coefs.append(np.nan)  # Use NaN for missing coefficients
                
        valid_coefs = [c for c, d in zip(coefs, valid_days) if not np.isnan(c)]
        if valid_coefs:
            plt.plot(range(5), coefs, marker='o', linewidth=2, label=proxy)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Day', fontsize=14)
    plt.ylabel('Coefficient', fontsize=14)
    plt.title('Impact of Attention Proxies on Returns', fontsize=16)
    plt.xticks(range(5), ['t', 't+1', 't+2', 't+3', 't+4'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("attention_returns_impact.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fm_table

def analyze_attention_investor_behavior(data):
    """
    Analyze the relationship between attention and order imbalance for different investor types
    """
    investor_types = ['Ind1', 'Ind2', 'Ind3', 'Ind4', 'Ind5', 
                      'Pro1', 'Pro2', 'Pro3', 'Pro4', 'Pro5', 'Pro6', 'OI']
    
    # Regression results
    regression_results = {}
    
    # Control variables
    controls = ['market_cap', 'bm_ratio', 'spread', 'turnover']
    
    # Prepare data
    regression_data = data.copy()
    regression_data['log_market_cap'] = np.log(regression_data['market_cap'] + 1)
    
    # Run regression for each investor type
    for investor in investor_types:
        # Set up the model
        y = regression_data[f'{investor}_im']
        X = regression_data[['Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']]
        X = sm.add_constant(X)
        
        # Skip any rows with NaN
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 10:
            print(f"Warning: Not enough data for investor {investor}")
            regression_results[investor] = {
                'coefficients': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                'std_errors': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                't_stats': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                'p_values': pd.Series([1]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                'r_squared': 0
            }
            continue
        
        # Run fixed effects regression
        try:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            
            # Store results
            regression_results[investor] = {
                'coefficients': model.params,
                'std_errors': model.HC3_se,
                't_stats': model.params / model.HC3_se,
                'p_values': model.pvalues,
                'r_squared': model.rsquared
            }
        except Exception as e:
            print(f"Error in regression for investor {investor}: {e}")
            regression_results[investor] = {
                'coefficients': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                'std_errors': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                't_stats': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                'p_values': pd.Series([1]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                'r_squared': 0
            }
    
    # Create table of coefficients
    coef_table = pd.DataFrame(index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover'])
    r_squared = {}
    
    for investor in investor_types:
        coef_table[investor] = regression_results[investor]['coefficients']
        r_squared[investor] = regression_results[investor]['r_squared']
    
    # Plot coefficients of Ab_GB1 for different investor types
    plt.figure(figsize=(14, 8))
    
    individual = investor_types[:5]
    professional = investor_types[5:11]
    other = investor_types[11:]
    
    # Coefficients for Ab_GB1
    ind_coefs = [regression_results[inv]['coefficients']['Ab_GB1'] for inv in individual]
    pro_coefs = [regression_results[inv]['coefficients']['Ab_GB1'] for inv in professional]
    oi_coefs = [regression_results[inv]['coefficients']['Ab_GB1'] for inv in other]
    
    # Standard errors for Ab_GB1
    ind_se = [regression_results[inv]['std_errors']['Ab_GB1'] for inv in individual]
    pro_se = [regression_results[inv]['std_errors']['Ab_GB1'] for inv in professional]
    oi_se = [regression_results[inv]['std_errors']['Ab_GB1'] for inv in other]
    
    # Plotting
    plt.bar(range(len(individual)), ind_coefs, yerr=ind_se, capsize=5, color='blue', label='Individual Investors')
    plt.bar(range(len(individual), len(individual) + len(professional)), pro_coefs, yerr=pro_se, capsize=5, color='red', 
            label='Professional Investors')
    plt.bar(range(len(individual) + len(professional), len(individual) + len(professional) + len(other)), oi_coefs, yerr=oi_se, 
            capsize=5, color='green', label='Other Institutional')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xticks(range(len(investor_types)), investor_types, rotation=45)
    plt.xlabel('Investor Type', fontsize=14)
    plt.ylabel('Coefficient of Ab_GB1', fontsize=14)
    plt.title('Impact of Attention on Order Imbalance by Investor Type', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("attention_investor_behavior.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return coef_table, r_squared

def implement_trading_strategy(data, lookback_period=10, holding_period=5, rebalance_freq=1):
    """
    Implement a trading strategy based on the attention proxy and investor behavior.
    
    Strategy:
    1. Rank stocks by Ab_GB1
    2. Take long positions in stocks with high attention (expecting short-term price pressure)
    3. Take short positions in stocks with high attention after holding period (expecting reversal)
    
    Parameters:
    -----------
    data : DataFrame
        The full dataset with market data, attention proxies, and investor trading
    lookback_period : int
        Number of days to look back for calculating baselines
    holding_period : int
        Number of days to hold positions
    rebalance_freq : int
        How often to rebalance the portfolio (in days)
        
    Returns:
    --------
    DataFrame with strategy performance
    """
    # Sort data by date and stock_id
    sorted_data = data.sort_values(['date', 'stock_id']).reset_index(drop=True)
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(sorted_data['date']):
        sorted_data['date'] = pd.to_datetime(sorted_data['date'])
    
    # Get unique dates
    unique_dates = sorted_data['date'].unique()
    unique_dates = pd.to_datetime(unique_dates)
    
    # Initialize portfolio and performance tracking
    portfolio = pd.DataFrame(index=unique_dates, columns=['long_stocks', 'short_stocks', 'long_return', 'short_return', 
                                                          'total_return', 'cumulative_return'])
    portfolio['long_stocks'] = np.nan
    portfolio['short_stocks'] = np.nan
    portfolio['long_return'] = 0.0
    portfolio['short_return'] = 0.0
    portfolio['total_return'] = 0.0
    portfolio['cumulative_return'] = 1.0
    
    # Initialize positions dictionary to track holdings
    long_positions = {}  # {date_entered: {stock_id: entry_price}}
    short_positions = {}  # {date_entered: {stock_id: entry_price}}
    
    # Trading days when we rebalance
    trading_days = list(range(0, len(unique_dates), rebalance_freq))
    
    for day_idx in tqdm(range(lookback_period, len(unique_dates)), desc="Running trading strategy"):
        current_date = unique_dates[day_idx]
        
        # Get data for current date
        current_day_data = sorted_data[sorted_data['date'] == current_date]
        
        # Enter new positions on rebalance days
        if day_idx in trading_days:
            # Rank stocks by Ab_GB1
            current_day_data = current_day_data.sort_values('Ab_GB1', ascending=False)
            
            # Select top 10% for long positions
            top_stocks = current_day_data.head(int(len(current_day_data) * 0.1))
            
            # Enter long positions
            long_stocks = {}
            for _, stock in top_stocks.iterrows():
                stock_id = stock['stock_id']
                price = stock['price']
                long_stocks[stock_id] = price
            
            long_positions[current_date] = long_stocks
            portfolio.at[current_date, 'long_stocks'] = len(long_stocks)
            
            # Enter short positions from past high attention stocks (after holding period)
            if day_idx >= lookback_period + holding_period and day_idx - holding_period in trading_days:
                past_date = unique_dates[day_idx - holding_period]
                
                # Get high attention stocks from past date
                past_day_data = sorted_data[sorted_data['date'] == past_date]
                past_day_data = past_day_data.sort_values('Ab_GB1', ascending=False)
                bottom_stocks = past_day_data.head(int(len(past_day_data) * 0.1))
                
                # Enter short positions
                short_stocks = {}
                for _, stock in bottom_stocks.iterrows():
                    stock_id = stock['stock_id']
                    price = current_day_data[current_day_data['stock_id'] == stock_id]['price'].values
                    if len(price) > 0:
                        short_stocks[stock_id] = price[0]
                
                short_positions[current_date] = short_stocks
                portfolio.at[current_date, 'short_stocks'] = len(short_stocks)
        
        # Calculate returns from existing positions
        # Long returns
        long_return = 0
        long_count = 0
        
        for entry_date, positions in list(long_positions.items()):
            # Check if it's time to close positions
            # Calculate days held using pandas Timedeltas
            days_held = (current_date - entry_date).days
            
            if days_held >= holding_period:
                # Close positions and calculate returns
                for stock_id, entry_price in positions.items():
                    current_price = current_day_data[current_day_data['stock_id'] == stock_id]['price'].values
                    if len(current_price) > 0:
                        long_return += (current_price[0] / entry_price - 1)
                        long_count += 1
                
                # Remove closed positions
                del long_positions[entry_date]
            
        # Short returns
        short_return = 0
        short_count = 0
        
        for entry_date, positions in list(short_positions.items()):
            # Check if it's time to close positions
            # Calculate days held using pandas Timedeltas
            days_held = (current_date - entry_date).days
            
            if days_held >= holding_period:
                # Close positions and calculate returns
                for stock_id, entry_price in positions.items():
                    current_price = current_day_data[current_day_data['stock_id'] == stock_id]['price'].values
                    if len(current_price) > 0:
                        short_return += (entry_price / current_price[0] - 1)  # Short return
                        short_count += 1
                
                # Remove closed positions
                del short_positions[entry_date]
        
        # Record returns
        if long_count > 0:
            portfolio.at[current_date, 'long_return'] = long_return / long_count
        
        if short_count > 0:
            portfolio.at[current_date, 'short_return'] = short_return / short_count
        
        # Calculate total return (equal weighting of long and short)
        if long_count > 0 and short_count > 0:
            portfolio.at[current_date, 'total_return'] = (portfolio.at[current_date, 'long_return'] + 
                                                           portfolio.at[current_date, 'short_return']) / 2
        elif long_count > 0:
            portfolio.at[current_date, 'total_return'] = portfolio.at[current_date, 'long_return']
        elif short_count > 0:
            portfolio.at[current_date, 'total_return'] = portfolio.at[current_date, 'short_return']
    
    # Calculate cumulative returns
    portfolio['cumulative_return'] = (1 + portfolio['total_return']).cumprod()
    
    # Calculate strategy metrics
    days_per_year = 252
    years = len(unique_dates) / days_per_year
    
    total_return = portfolio['cumulative_return'].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1
    annualized_volatility = portfolio['total_return'].std() * np.sqrt(days_per_year)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Calculate drawdowns
    portfolio['peak'] = portfolio['cumulative_return'].cummax()
    portfolio['drawdown'] = (portfolio['cumulative_return'] - portfolio['peak']) / portfolio['peak']
    max_drawdown = portfolio['drawdown'].min()
    
    # Plot strategy performance
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio.index, portfolio['cumulative_return'], label='Strategy', linewidth=2)
    
    # Plot drawdowns
    plt.fill_between(portfolio.index, 1, portfolio['cumulative_return'] / portfolio['peak'], 
                     color='red', alpha=0.3, label='Drawdowns')
    
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.title('Attention-Based Trading Strategy Performance', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add metrics to the plot
    metrics_text = (f"Annualized Return: {annualized_return:.2%}\n"
                    f"Annualized Volatility: {annualized_volatility:.2%}\n"
                    f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                    f"Max Drawdown: {max_drawdown:.2%}")
    
    plt.figtext(0.15, 0.15, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("trading_strategy_performance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot long and short performance separately
    plt.figure(figsize=(14, 7))
    
    cumulative_long = (1 + portfolio['long_return'].fillna(0)).cumprod()
    cumulative_short = (1 + portfolio['short_return'].fillna(0)).cumprod()
    
    plt.plot(portfolio.index, cumulative_long, label='Long Strategy', linewidth=2, color='green')
    plt.plot(portfolio.index, cumulative_short, label='Short Strategy', linewidth=2, color='red')
    plt.plot(portfolio.index, portfolio['cumulative_return'], label='Combined Strategy', linewidth=2, color='blue')
    
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.title('Long vs. Short Strategy Performance', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("long_vs_short_performance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate monthly returns
    portfolio['year_month'] = portfolio.index.map(lambda x: (x.year, x.month))
    monthly_returns = portfolio.groupby('year_month')['total_return'].sum().reset_index()
    
    # Plot monthly returns
    plt.figure(figsize=(14, 7))
    
    plt.bar(range(len(monthly_returns)), monthly_returns['total_return'], color='blue')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Monthly Return', fontsize=14)
    plt.title('Monthly Strategy Returns', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("monthly_returns.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return strategy performance data
    strategy_metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }
    
    return portfolio, strategy_metrics

def short_sale_constraint_analysis(data):
    """
    Analyze the effect of short sale constraints on the relationship between
    attention and order imbalance.
    
    Split the sample into two groups:
    1. Stocks that can be sold short
    2. Stocks that cannot be sold short
    """
    # Make a copy of the data to avoid modifying the original
    analysis_data = data.copy()
    
    # Add log_market_cap column if it doesn't exist
    if 'log_market_cap' not in analysis_data.columns:
        analysis_data['log_market_cap'] = np.log(analysis_data['market_cap'] + 1)
        
    # Randomly assign stocks to short-sellable and non-short-sellable groups
    np.random.seed(42)
    stock_ids = analysis_data['stock_id'].unique()
    short_sellable = np.random.choice(stock_ids, size=int(len(stock_ids) * 0.5), replace=False)
    
    # Create a flag for short-sellable stocks
    analysis_data['short_sellable'] = analysis_data['stock_id'].isin(short_sellable)
    
    # Split the data
    short_sell_data = analysis_data[analysis_data['short_sellable']]
    no_short_sell_data = analysis_data[~analysis_data['short_sellable']]
    
    # List of individual investors
    individual_investors = ['Ind1', 'Ind2', 'Ind3', 'Ind4', 'Ind5']
    
    # Run regressions for each group
    results = {}
    
    for group_name, group_data in [('short_sell', short_sell_data), ('no_short_sell', no_short_sell_data)]:
        group_results = {}
        
        for investor in individual_investors:
            # Prepare data for regression
            y = group_data[f'{investor}_im']
            X = group_data[['Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']]
            X = sm.add_constant(X)
            
            # Skip any rows with NaN
            valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 10:
                print(f"Warning: Not enough data for investor {investor} in {group_name} group")
                group_results[investor] = {
                    'coefficients': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                    'std_errors': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                    't_stats': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                    'p_values': pd.Series([1]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                    'r_squared': 0
                }
                continue
            
            # Run regression
            try:
                model = sm.OLS(y, X).fit(cov_type='HC3')
                
                # Store results
                group_results[investor] = {
                    'coefficients': model.params,
                    'std_errors': model.HC3_se,
                    't_stats': model.params / model.HC3_se,
                    'p_values': model.pvalues,
                    'r_squared': model.rsquared
                }
            except Exception as e:
                print(f"Error in regression for investor {investor} in {group_name} group: {e}")
                group_results[investor] = {
                    'coefficients': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                    'std_errors': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                    't_stats': pd.Series([0]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                    'p_values': pd.Series([1]*6, index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover']),
                    'r_squared': 0
                }
        
        results[group_name] = group_results
    
    # Create comparison table
    comparison_table = pd.DataFrame(index=['constant', 'Ab_GB1', 'log_market_cap', 'bm_ratio', 'spread', 'turnover'])
    
    for investor in individual_investors:
        if investor in results['short_sell'] and investor in results['no_short_sell']:
            comparison_table[f"{investor}_short"] = results['short_sell'][investor]['coefficients']
            comparison_table[f"{investor}_no_short"] = results['no_short_sell'][investor]['coefficients']
    
    # Calculate differences and Z-statistics for Ab_GB1 coefficient
    diff_results = {}
    
    for investor in individual_investors:
        if investor in results['short_sell'] and investor in results['no_short_sell']:
            if 'Ab_GB1' in results['short_sell'][investor]['coefficients'] and 'Ab_GB1' in results['no_short_sell'][investor]['coefficients']:
                b1 = results['short_sell'][investor]['coefficients']['Ab_GB1']
                b2 = results['no_short_sell'][investor]['coefficients']['Ab_GB1']
                se1 = results['short_sell'][investor]['std_errors']['Ab_GB1']
                se2 = results['no_short_sell'][investor]['std_errors']['Ab_GB1']
                
                diff = b1 - b2
                z_stat = diff / np.sqrt(se1**2 + se2**2) if (se1**2 + se2**2) > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                diff_results[investor] = {
                    'diff': diff,
                    'z_stat': z_stat,
                    'p_value': p_value
                }
            else:
                diff_results[investor] = {
                    'diff': 0,
                    'z_stat': 0,
                    'p_value': 1
                }
        else:
            diff_results[investor] = {
                'diff': 0,
                'z_stat': 0,
                'p_value': 1
            }
    
    # Plot the differences
    plt.figure(figsize=(12, 8))
    
    # Plot Ab_GB1 coefficients for both groups
    x = np.arange(len(individual_investors))
    width = 0.35
    
    short_coefs = []
    no_short_coefs = []
    short_errors = []
    no_short_errors = []
    
    for inv in individual_investors:
        if inv in results['short_sell'] and 'Ab_GB1' in results['short_sell'][inv]['coefficients']:
            short_coefs.append(results['short_sell'][inv]['coefficients']['Ab_GB1'])
            short_errors.append(results['short_sell'][inv]['std_errors']['Ab_GB1'])
        else:
            short_coefs.append(0)
            short_errors.append(0)
            
        if inv in results['no_short_sell'] and 'Ab_GB1' in results['no_short_sell'][inv]['coefficients']:
            no_short_coefs.append(results['no_short_sell'][inv]['coefficients']['Ab_GB1'])
            no_short_errors.append(results['no_short_sell'][inv]['std_errors']['Ab_GB1'])
        else:
            no_short_coefs.append(0)
            no_short_errors.append(0)
    
    plt.bar(x - width/2, short_coefs, width, yerr=short_errors, label='Short Sellable', color='blue', capsize=5)
    plt.bar(x + width/2, no_short_coefs, width, yerr=no_short_errors, label='Not Short Sellable', color='red', capsize=5)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Investor Type', fontsize=14)
    plt.ylabel('Coefficient of Ab_GB1', fontsize=14)
    plt.title('Impact of Short Selling Constraints on Attention Effect', fontsize=16)
    plt.xticks(x, individual_investors)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add significance stars
    for i, inv in enumerate(individual_investors):
        if inv in diff_results:
            p_val = diff_results[inv]['p_value']
            y_pos = max(short_coefs[i], no_short_coefs[i]) + max(short_errors[i], no_short_errors[i]) + 0.1
            
            if p_val < 0.01:
                plt.text(i, y_pos, '***', ha='center', fontsize=14)
            elif p_val < 0.05:
                plt.text(i, y_pos, '**', ha='center', fontsize=14)
            elif p_val < 0.1:
                plt.text(i, y_pos, '*', ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig("short_sale_constraint_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_table, diff_results

if __name__ == "__main__":
    print("Simulating market data...")
    market_data = create_market_data(num_stocks, trading_dates)
    
    print("Simulating attention proxies...")
    attention_data = simulate_attention_proxies(market_data)
    
    print("Merging market and attention data...")
    data = pd.merge(market_data, attention_data, on=['stock_id', 'date'])
    
    print("Simulating investor trading...")
    trading_data = simulate_investor_trading(data)
    
    print("Merging all data...")
    full_data = pd.merge(data, trading_data, on=['stock_id', 'date'])
    
    print("\nAnalyzing attention proxies...")
    corr_matrix = analyze_attention_proxies(full_data)
    print(corr_matrix)
    
    print("\nRunning VAR analysis...")
    var_table = run_var_analysis(full_data)
    print(var_table)
    
    print("\nAnalyzing impact of attention on returns...")
    fm_table = analyze_attention_returns(full_data)
    print(fm_table)
    
    print("\nAnalyzing impact of attention on investor behavior...")
    coef_table, r_squared = analyze_attention_investor_behavior(full_data)
    print(coef_table)
    
    print("\nImplementing trading strategy...")
    strategy_results, strategy_metrics = implement_trading_strategy(full_data)
    
    print("\nStrategy Metrics:")
    for metric, value in strategy_metrics.items():
        if metric in ['annualized_return', 'annualized_volatility', 'max_drawdown']:
            print(f"{metric}: {value:.2%}")
        else:
            print(f"{metric}: {value:.4f}")
    
    print("\nAnalyzing short sale constraints...")
    comparison_table, diff_results = short_sale_constraint_analysis(full_data)
    print(comparison_table)
    
    for investor, result in diff_results.items():
        print(f"{investor}: Diff = {result['diff']:.4f}, Z-stat = {result['z_stat']:.4f}, p-value = {result['p_value']:.4f}")
    
    print("\nAnalysis complete! All charts have been saved.")