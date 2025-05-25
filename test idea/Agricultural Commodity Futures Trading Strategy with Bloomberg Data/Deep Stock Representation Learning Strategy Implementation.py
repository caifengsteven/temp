import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import datetime as dt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setting up device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to generate simulated price data
def generate_simulated_data(n_stocks=10, n_days=1000, start_date="2015-01-01"):
    """Generate simulated OHLC data for multiple stocks"""
    start_date = pd.to_datetime(start_date)
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')
    
    # Create random sectors for stocks (to create correlation structure)
    sectors = np.random.randint(0, 5, n_stocks)
    
    # Initialize market movement (common to all stocks)
    market_movement = np.zeros(n_days)
    market_movement[0] = 100
    
    # Generate random walk with momentum for market
    for i in range(1, n_days):
        momentum = 0.1 * (market_movement[i-1] - 100)  # Mean reversion to 100
        market_movement[i] = market_movement[i-1] * (1 + np.random.normal(0, 0.01) - momentum / 100)
    
    # Generate sector movements
    sector_movements = {}
    for sector in np.unique(sectors):
        sector_movement = np.zeros(n_days)
        sector_movement[0] = 100
        
        for i in range(1, n_days):
            # Sector follows market with own random component
            market_influence = 0.3 * ((market_movement[i] / market_movement[i-1]) - 1)
            sector_movement[i] = sector_movement[i-1] * (1 + market_influence + np.random.normal(0, 0.015))
        
        sector_movements[sector] = sector_movement
    
    # Generate stock data
    stock_data = {}
    for i in range(n_stocks):
        ticker = f"STOCK_{i+1}"
        sector = sectors[i]
        
        # Start price with random value between 50 and 200
        price = np.random.uniform(50, 200)
        
        # Generate prices with correlation to sector and market
        prices = np.zeros(n_days)
        prices[0] = price
        
        for j in range(1, n_days):
            # Stock follows its sector with own random component
            sector_influence = 0.5 * ((sector_movements[sector][j] / sector_movements[sector][j-1]) - 1)
            market_influence = 0.2 * ((market_movement[j] / market_movement[j-1]) - 1)
            
            # Add some mean reversion
            mean_reversion = 0.05 * (price - prices[j-1]) / price
            
            # Add some random noise specific to the stock
            stock_specific = np.random.normal(0, 0.02)
            
            # Combine all factors
            daily_return = sector_influence + market_influence + mean_reversion + stock_specific
            prices[j] = prices[j-1] * (1 + daily_return)
        
        # Generate OHLC data from prices
        ohlc_data = pd.DataFrame(index=dates)
        
        # Close price is the main simulated price
        ohlc_data['Close'] = prices
        
        # Generate realistic Open, High, Low from Close
        for j in range(n_days):
            if j > 0:
                # Open price is close to previous day's close with some gap
                open_price = prices[j-1] * (1 + np.random.normal(0, 0.005))
            else:
                open_price = prices[j] * (1 - np.random.uniform(0, 0.01))
            
            # Daily range is typically 1-2%
            daily_range = prices[j] * np.random.uniform(0.01, 0.03)
            
            # High and low depend on whether day is up or down
            if prices[j] > open_price:  # Up day
                high_price = prices[j] + np.random.uniform(0, daily_range * 0.3)
                low_price = open_price - np.random.uniform(0, daily_range * 0.7)
            else:  # Down day
                high_price = open_price + np.random.uniform(0, daily_range * 0.3)
                low_price = prices[j] - np.random.uniform(0, daily_range * 0.7)
            
            ohlc_data.loc[dates[j], 'Open'] = open_price
            ohlc_data.loc[dates[j], 'High'] = high_price
            ohlc_data.loc[dates[j], 'Low'] = low_price
        
        # Add volume (correlated with price movement)
        daily_returns = ohlc_data['Close'].pct_change().fillna(0)
        base_volume = np.random.uniform(1000000, 5000000)
        ohlc_data['Volume'] = base_volume * (1 + 2 * np.abs(daily_returns))
        
        stock_data[ticker] = ohlc_data
    
    # Create benchmark index
    benchmark = pd.DataFrame(index=dates)
    benchmark['Close'] = market_movement
    benchmark['Open'] = market_movement * (1 + np.random.normal(0, 0.001, n_days))
    benchmark['High'] = benchmark[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, n_days))
    benchmark['Low'] = benchmark[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, n_days))
    benchmark['Volume'] = np.random.uniform(10000000, 50000000, n_days)
    
    return stock_data, benchmark, sectors

class CandlestickImageGenerator:
    """Class to generate candlestick chart images from OHLC data"""
    
    def __init__(self, output_dir="candlestick_images", img_size=(224, 224)):
        self.output_dir = output_dir
        self.img_size = img_size
        os.makedirs(output_dir, exist_ok=True)
    
    def _create_candlestick_chart(self, ohlc_data, ticker, window_start=0, window_size=20):
        """Create a single candlestick chart image"""
        # Extract the window of data we want
        window_data = ohlc_data.iloc[window_start:window_start+window_size].copy()
        
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        
        # Use mplfinance to create the candlestick chart
        mpf.plot(window_data, type='candle', style='yahoo', 
                 title='', ylabel='', 
                 volume=False, show_nontrading=False,
                 figsize=(10, 6), xrotation=0, tight_layout=True,
                 axisoff=True, returnfig=True)
        
        # Save the chart to a file
        filename = f"{self.output_dir}/{ticker}_{window_start}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        
        return filename
    
    def generate_images_for_ticker(self, ohlc_data, ticker, stride=10):
        """Generate candlestick chart images for a ticker with a sliding window"""
        window_size = 20
        num_windows = len(ohlc_data) - window_size + 1
        
        filenames = []
        for i in range(0, num_windows, stride):
            try:
                filename = self._create_candlestick_chart(ohlc_data, ticker, i, window_size)
                filenames.append(filename)
            except Exception as e:
                print(f"Error generating image for {ticker} at window {i}: {e}")
        
        return filenames

class StockImageDataset(Dataset):
    """Dataset for loading stock candlestick images"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image of the same size as a fallback
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image

# Simplified CNN for feature extraction
class SimpleEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(SimpleEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv_layers = nn.Sequential(
            # First block: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fifth block: 14x14 -> 7x7
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layer to get latent representation
        self.fc = nn.Linear(512 * 7 * 7, latent_dim)
        
    def forward(self, x):
        # Apply convolution layers
        x = self.conv_layers(x)  # Output: [batch_size, 512, 7, 7]
        
        # Flatten the output
        x = x.view(x.size(0), -1)  # Output: [batch_size, 512*7*7]
        
        # Apply fully connected layer to get latent representation
        x = self.fc(x)  # Output: [batch_size, latent_dim]
        
        return x

# Decoder to reconstruct the image
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(SimpleDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Fully connected layer to expand from latent dimension
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
        
        # Transpose convolution layers
        self.deconv_layers = nn.Sequential(
            # First block: 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Second block: 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Third block: 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth block: 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth block: 112x112 -> 224x224
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Final output between 0 and 1
        )
        
    def forward(self, x):
        # Apply fully connected layer
        x = self.fc(x)  # Output: [batch_size, 512*7*7]
        
        # Reshape to [batch_size, 512, 7, 7]
        x = x.view(x.size(0), 512, 7, 7)
        
        # Apply transpose convolution layers
        x = self.deconv_layers(x)  # Output: [batch_size, 3, 224, 224]
        
        return x

class SimpleCAE(nn.Module):
    """Simplified Convolutional Autoencoder"""
    
    def __init__(self, latent_dim=512):
        super(SimpleCAE, self).__init__()
        self.encoder = SimpleEncoder(latent_dim)
        self.decoder = SimpleDecoder(latent_dim)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input to get feature representation"""
        return self.encoder(x)

def train_cae(model, train_loader, num_epochs=2, learning_rate=0.001):
    """Train the Convolutional Autoencoder"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # For demonstration, we'll use a very short training regime
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs = data.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print statistics every few batches
            if (i+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/5:.4f}')
                running_loss = 0.0
                
    print('Finished Training')
    return model

def extract_features(model, data_loader):
    """Extract features from trained model"""
    model.eval()
    features = []
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = data.to(device)
            encoded = model.encode(inputs)
            features.append(encoded.cpu().numpy())
            if (i+1) % 5 == 0:
                print(f'Extracting features: batch {i+1}/{len(data_loader)}')
    
    return np.vstack(features) if features else np.array([])

def modularity_clustering(similarity_matrix, threshold=0.5):
    """
    Perform clustering using modularity optimization
    
    Parameters:
    similarity_matrix: NxN matrix of similarities between stocks
    threshold: Threshold for similarity to create an edge
    
    Returns:
    clusters: List of stock clusters
    """
    # Create a binary adjacency matrix based on threshold
    adj_matrix = (similarity_matrix >= threshold).astype(int)
    
    # Convert to a NetworkX graph
    G = nx.from_numpy_array(adj_matrix)
    
    # Run community detection using greedy modularity optimization
    partition = nx.community.greedy_modularity_communities(G)
    
    # Convert partition to list of clusters
    clusters = [list(cluster) for cluster in partition]
    
    # If no clusters found, create one cluster with all stocks
    if not clusters:
        clusters = [list(range(similarity_matrix.shape[0]))]
    
    return clusters

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio for a given return series"""
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns)

def select_portfolio(stock_data, clusters, tickers, lookback=20):
    """
    Select stocks for portfolio based on Sharpe ratio
    
    Parameters:
    stock_data: Dictionary of DataFrames with price data for each ticker
    clusters: List of clusters, where each cluster is a list of stock indices
    tickers: List of stock tickers
    lookback: Number of days to use for Sharpe ratio calculation
    
    Returns:
    selected_stocks: List of selected tickers for portfolio
    """
    selected_stocks = []
    
    for cluster in clusters:
        # Get stocks in this cluster
        cluster_tickers = [tickers[idx] for idx in cluster]
        
        best_sharpe = -np.inf
        best_ticker = None
        
        # Find stock with highest Sharpe ratio in cluster
        for ticker in cluster_tickers:
            if ticker in stock_data:
                df = stock_data[ticker]
                if len(df) >= lookback:
                    # Calculate daily returns
                    returns = df['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        # Use last lookback days
                        recent_returns = returns.iloc[-lookback:]
                        sharpe = calculate_sharpe_ratio(recent_returns)
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_ticker = ticker
        
        if best_ticker:
            selected_stocks.append(best_ticker)
    
    return selected_stocks

def backtest_strategy(stock_data, window_size=20, stride=10, hold_period=10, 
                      num_stocks=5, initial_capital=100000.0):
    """
    Backtest the deep stock representation learning strategy
    
    Parameters:
    stock_data: Dictionary of DataFrames with price data for each ticker
    window_size: Size of window for feature extraction (days)
    stride: Stride for rolling windows (days)
    hold_period: How long to hold the portfolio (days)
    num_stocks: Number of stocks to include in portfolio
    initial_capital: Initial investment capital
    
    Returns:
    results: Dictionary containing backtest results
    """
    tickers = list(stock_data.keys())
    
    # Get common date range from the first stock (all stocks have same dates in simulation)
    dates = stock_data[tickers[0]].index
    
    # Initialize results tracking
    portfolio_values = []
    holdings = {}
    cash = initial_capital
    
    # Initialize CAE model - simplified version
    model = SimpleCAE(latent_dim=512).to(device)
    
    # Training data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Trading days with stride
    trading_days = list(range(0, len(dates) - window_size - hold_period, stride))
    
    for day_idx in trading_days:
        print(f"\nProcessing trading day {day_idx}/{trading_days[-1]} - Date: {dates[day_idx].strftime('%Y-%m-%d')}")
        
        # Generate candlestick images for all stocks in the current window
        image_generator = CandlestickImageGenerator()
        all_images = []
        valid_tickers = []
        
        for ticker in tickers:
            try:
                # Extract window of data
                window_data = stock_data[ticker].iloc[day_idx:day_idx+window_size]
                
                # Generate candlestick image
                images = image_generator.generate_images_for_ticker(window_data, ticker, stride=20)
                if images:
                    all_images.extend(images)
                    valid_tickers.append(ticker)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
        
        if not all_images:
            print("No valid images generated, skipping this window")
            continue
        
        print(f"Generated {len(all_images)} candlestick images for {len(valid_tickers)} stocks")
        
        # Create dataset and dataloader for all images
        dataset = StockImageDataset(all_images, transform=transform)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Train CAE model on all images (simplified training for demonstration)
        model = train_cae(model, dataloader, num_epochs=2, learning_rate=0.001)
        
        # Extract features for all stocks
        features = extract_features(model, dataloader)
        
        if len(features) == 0:
            print("No features extracted, skipping this window")
            continue
            
        # Calculate similarity matrix
        # Use cosine similarity (1 - cosine distance)
        similarity_matrix = 1 - pairwise_distances(features, metric='cosine')
        
        # For stability, set diagonal to 1 (each stock is fully similar to itself)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Make sure the similarity matrix dimensions match the number of valid tickers
        if len(valid_tickers) != similarity_matrix.shape[0]:
            valid_tickers = valid_tickers[:similarity_matrix.shape[0]]
            
        # Perform clustering using modularity optimization
        clusters = modularity_clustering(similarity_matrix, threshold=0.5)
        
        print(f"Found {len(clusters)} clusters")
        
        # Select portfolio based on clusters and Sharpe ratio
        selected_stocks = select_portfolio(stock_data, clusters, valid_tickers)
        
        print(f"Selected stocks: {selected_stocks}")
        
        # If we need a specific number of stocks, adjust the selection
        if len(selected_stocks) > num_stocks:
            # Keep top num_stocks by Sharpe ratio
            stock_sharpes = {}
            for ticker in selected_stocks:
                returns = stock_data[ticker]['Close'].pct_change().dropna()
                if len(returns) > 0:
                    recent_returns = returns.iloc[-window_size:]
                    sharpe = calculate_sharpe_ratio(recent_returns)
                    stock_sharpes[ticker] = sharpe
            
            selected_stocks = sorted(stock_sharpes.keys(), key=lambda x: stock_sharpes[x], reverse=True)[:num_stocks]
            print(f"Reduced to top {num_stocks} stocks: {selected_stocks}")
        
        # Liquidate current holdings
        for ticker, shares in holdings.items():
            price = stock_data[ticker]['Close'].iloc[day_idx + window_size]
            cash += shares * price
        
        holdings = {}
        
        # Allocate equal capital to each selected stock
        if selected_stocks:
            capital_per_stock = cash / len(selected_stocks)
            
            for ticker in selected_stocks:
                price = stock_data[ticker]['Close'].iloc[day_idx + window_size]
                shares = capital_per_stock / price
                holdings[ticker] = shares
                cash -= shares * price
        
        # Track portfolio value over the holding period
        for offset in range(hold_period):
            if day_idx + window_size + offset < len(dates):
                eval_date = dates[day_idx + window_size + offset]
                port_value = cash
                
                for ticker, shares in holdings.items():
                    price = stock_data[ticker]['Close'].iloc[day_idx + window_size + offset]
                    port_value += shares * price
                
                portfolio_values.append((eval_date, port_value))
    
    # Calculate performance metrics
    if not portfolio_values:
        return {"error": "No valid trading periods"}
    
    portfolio_values.sort(key=lambda x: x[0])  # Sort by date
    dates, values = zip(*portfolio_values)
    
    # Calculate daily returns
    daily_returns = [(values[i] / values[i-1]) - 1 for i in range(1, len(values))]
    daily_returns_series = pd.Series(daily_returns, index=dates[1:])
    
    # Total return
    total_return = (values[-1] / values[0] - 1) * 100
    
    # Sharpe ratio (annualized)
    daily_sharpe = calculate_sharpe_ratio(daily_returns_series) * np.sqrt(252)
    
    # Max drawdown
    peak = values[0]
    max_drawdown = 0
    
    for value in values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    max_drawdown *= 100
    
    # Monthly and yearly returns
    returns_df = pd.DataFrame({'return': daily_returns_series})
    monthly_returns = returns_df.resample('M').apply(lambda x: (1 + x).prod() - 1)
    yearly_returns = returns_df.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    
    # Mean returns (annualized)
    daily_mean_ret = np.mean(daily_returns_series) * 100 * 252
    monthly_mean_ret = np.mean(monthly_returns['return']) * 100 * 12 if not monthly_returns.empty else 0
    yearly_mean_ret = np.mean(yearly_returns['return']) * 100 if not yearly_returns.empty else 0
    
    # Win years
    win_years = (yearly_returns['return'] > 0).mean() * 100 if not yearly_returns.empty else 0
    
    # Create results dictionary
    results = {
        'portfolio_values': pd.Series(values, index=dates),
        'total_return': total_return,
        'daily_sharpe': daily_sharpe,
        'max_drawdown': max_drawdown,
        'daily_mean_ret': daily_mean_ret,
        'monthly_mean_ret': monthly_mean_ret,
        'yearly_mean_ret': yearly_mean_ret,
        'win_years': win_years
    }
    
    return results

def generate_report(results, benchmark_results=None):
    """Generate performance report and charts"""
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print("\nPortfolio Performance Report")
    print("===========================")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Daily Sharpe Ratio: {results['daily_sharpe']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Annualized Daily Mean Return: {results['daily_mean_ret']:.2f}%")
    print(f"Annualized Monthly Mean Return: {results['monthly_mean_ret']:.2f}%")
    print(f"Yearly Mean Return: {results['yearly_mean_ret']:.2f}%")
    print(f"Win Years: {results['win_years']:.2f}%")
    
    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(results['portfolio_values'].index, results['portfolio_values'].values, label='Portfolio Value')
    
    if benchmark_results is not None:
        # Normalize benchmark to same starting value
        initial_value_ratio = results['portfolio_values'].iloc[0] / benchmark_results['portfolio_values'].iloc[0]
        benchmark_normalized = benchmark_results['portfolio_values'] * initial_value_ratio
        
        # Reindex to align dates
        common_dates = results['portfolio_values'].index.intersection(benchmark_normalized.index)
        if len(common_dates) > 0:
            portfolio_aligned = results['portfolio_values'].loc[common_dates]
            benchmark_aligned = benchmark_normalized.loc[common_dates]
            
            plt.plot(benchmark_aligned.index, benchmark_aligned.values, label='Benchmark')
    
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    print("Performance chart saved as 'portfolio_performance.png'")
    
    # Compare with benchmark if available
    if benchmark_results is not None:
        print("\nComparative Performance")
        print("======================")
        print(f"Portfolio Total Return: {results['total_return']:.2f}%")
        print(f"Benchmark Total Return: {benchmark_results['total_return']:.2f}%")
        print(f"Outperformance: {results['total_return'] - benchmark_results['total_return']:.2f}%")
        
        print(f"Portfolio Sharpe: {results['daily_sharpe']:.2f}")
        print(f"Benchmark Sharpe: {benchmark_results['daily_sharpe']:.2f}")
        
        print(f"Portfolio Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Benchmark Max Drawdown: {benchmark_results['max_drawdown']:.2f}%")


def main():
    # Generate simulated data for 20 stocks over 500 days
    print("Generating simulated stock data...")
    n_stocks = 20
    n_days = 500
    stock_data, benchmark_data, sectors = generate_simulated_data(n_stocks=n_stocks, n_days=n_days)
    
    print(f"Generated data for {n_stocks} stocks over {n_days} days")
    
    # Print sector information
    sector_counts = {}
    for i, sector in enumerate(sectors):
        ticker = f"STOCK_{i+1}"
        if sector not in sector_counts:
            sector_counts[sector] = []
        sector_counts[sector].append(ticker)
    
    print("\nStock sectors:")
    for sector, tickers in sector_counts.items():
        print(f"Sector {sector}: {', '.join(tickers)}")
    
    # Calculate benchmark performance metrics
    benchmark_returns = benchmark_data['Close'].pct_change().dropna()
    benchmark_monthly_returns = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    benchmark_yearly_returns = benchmark_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    
    benchmark_results = {
        'portfolio_values': benchmark_data['Close'],
        'total_return': (benchmark_data['Close'].iloc[-1] / benchmark_data['Close'].iloc[0] - 1) * 100,
        'daily_sharpe': calculate_sharpe_ratio(benchmark_returns) * np.sqrt(252),
        'max_drawdown': (benchmark_data['Close'].cummax() - benchmark_data['Close']).max() / benchmark_data['Close'].cummax().max() * 100,
        'daily_mean_ret': np.mean(benchmark_returns) * 100 * 252,
        'monthly_mean_ret': np.mean(benchmark_monthly_returns) * 100 * 12,
        'yearly_mean_ret': np.mean(benchmark_yearly_returns) * 100,
        'win_years': (benchmark_yearly_returns > 0).mean() * 100
    }
    
    # Run backtest with smaller window and fewer days for faster execution
    print("\nRunning backtest...")
    results = backtest_strategy(
        stock_data, 
        window_size=20,      # 20-day window 
        stride=20,           # 20-day stride (for faster execution)
        hold_period=10,      # 10-day holding period
        num_stocks=5,        # 5 stocks in portfolio
        initial_capital=100000.0
    )
    
    # Generate report
    generate_report(results, benchmark_results)
    
    # Clean up temporary image files
    try:
        import shutil
        shutil.rmtree('candlestick_images', ignore_errors=True)
        print("\nCleaned up temporary files")
    except:
        pass

if __name__ == "__main__":
    main()