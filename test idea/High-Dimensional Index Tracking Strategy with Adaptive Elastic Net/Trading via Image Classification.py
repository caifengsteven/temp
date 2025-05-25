import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def simulate_stock_data(ticker, start_date, end_date, initial_price=100, drift=0.0001, volatility=0.01, 
                        jump_probability=0.05, jump_mean=0, jump_std=0.03):
    """
    Simulate daily stock data
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (used only for identification)
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    initial_price : float
        Initial stock price
    drift : float
        Daily drift (average return)
    volatility : float
        Daily volatility
    jump_probability : float
        Probability of a price jump on any given day
    jump_mean : float
        Mean of jump size
    jump_std : float
        Standard deviation of jump size
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with simulated stock data
    """
    # Create date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    # Initialize price and other series
    n_days = len(dates)
    prices = np.zeros(n_days)
    prices[0] = initial_price
    
    # Simulate daily Open, High, Low, Close values
    open_prices = np.zeros(n_days)
    high_prices = np.zeros(n_days)
    low_prices = np.zeros(n_days)
    close_prices = np.zeros(n_days)
    volumes = np.zeros(n_days)
    
    open_prices[0] = initial_price
    high_prices[0] = initial_price * (1 + np.random.normal(0, volatility/2))
    low_prices[0] = initial_price * (1 - np.random.normal(0, volatility/2))
    close_prices[0] = initial_price * (1 + np.random.normal(0, volatility))
    volumes[0] = np.random.normal(1000000, 200000)
    
    # Simulate price path
    for i in range(1, n_days):
        # Random price change
        daily_return = np.random.normal(drift, volatility)
        
        # Add jumps occasionally
        if np.random.random() < jump_probability:
            daily_return += np.random.normal(jump_mean, jump_std)
        
        # Calculate price
        open_prices[i] = close_prices[i-1]
        close_prices[i] = open_prices[i] * (1 + daily_return)
        
        # Calculate high and low
        daily_volatility = volatility * open_prices[i]
        high_prices[i] = max(open_prices[i], close_prices[i]) + np.random.uniform(0, daily_volatility)
        low_prices[i] = min(open_prices[i], close_prices[i]) - np.random.uniform(0, daily_volatility)
        
        # Generate volume (positively correlated with price movement)
        base_volume = 1000000
        volume_volatility = abs(daily_return) * 10 * base_volume
        volumes[i] = base_volume + np.random.normal(0, volume_volatility)
        volumes[i] = max(volumes[i], 100000)  # Ensure positive volume
    
    # Create DataFrame
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes,
        'Ticker': ticker
    }, index=dates)
    
    return data

def simulate_market_data(tickers, start_date, end_date, market_drift=0.0001, market_vol=0.01, sector_vol=0.005):
    """
    Simulate market data for multiple stocks with correlation
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    market_drift : float
        Market drift (average return)
    market_vol : float
        Market volatility
    sector_vol : float
        Sector-specific volatility
        
    Returns:
    --------
    dict
        Dictionary of DataFrames with simulated stock data
    """
    # Create date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = pd.date_range(start=start, end=end, freq='B')  # Business days
    n_days = len(dates)
    
    # Simulate market factor
    market_returns = np.random.normal(market_drift, market_vol, n_days)
    
    # Group tickers into sectors
    n_sectors = 3  # Example: 3 sectors
    sectors = {}
    for i, ticker in enumerate(tickers):
        sector_id = i % n_sectors
        if sector_id not in sectors:
            sectors[sector_id] = []
        sectors[sector_id].append(ticker)
    
    # Simulate sector factors
    sector_returns = {}
    for sector_id in sectors:
        sector_returns[sector_id] = np.random.normal(0, sector_vol, n_days)
    
    # Simulate stock data
    market_data = {}
    
    for sector_id, sector_tickers in sectors.items():
        for ticker in sector_tickers:
            # Individual stock parameters
            initial_price = np.random.uniform(50, 200)
            stock_vol = np.random.uniform(0.01, 0.02)
            stock_drift = np.random.uniform(-0.0005, 0.001)
            
            # Initialize price series
            open_prices = np.zeros(n_days)
            high_prices = np.zeros(n_days)
            low_prices = np.zeros(n_days)
            close_prices = np.zeros(n_days)
            volumes = np.zeros(n_days)
            
            # Set initial values
            open_prices[0] = initial_price
            close_prices[0] = initial_price * (1 + np.random.normal(0, stock_vol))
            high_prices[0] = max(open_prices[0], close_prices[0]) * (1 + abs(np.random.normal(0, stock_vol/2)))
            low_prices[0] = min(open_prices[0], close_prices[0]) * (1 - abs(np.random.normal(0, stock_vol/2)))
            volumes[0] = np.random.normal(1000000, 200000)
            
            # Simulate price path with market and sector correlation
            for i in range(1, n_days):
                # Stock-specific return
                stock_specific = np.random.normal(stock_drift, stock_vol)
                
                # Combine with market and sector returns
                daily_return = market_returns[i] + sector_returns[sector_id][i] + stock_specific
                
                # Calculate prices
                open_prices[i] = close_prices[i-1]
                close_prices[i] = open_prices[i] * (1 + daily_return)
                
                # Calculate high and low
                daily_range = abs(daily_return) * open_prices[i] * np.random.uniform(1, 2)
                high_prices[i] = max(open_prices[i], close_prices[i]) + np.random.uniform(0, daily_range/2)
                low_prices[i] = min(open_prices[i], close_prices[i]) - np.random.uniform(0, daily_range/2)
                
                # Generate volume (positively correlated with price movement)
                base_volume = 1000000
                volume_volatility = abs(daily_return) * 10 * base_volume
                volumes[i] = base_volume + np.random.normal(0, volume_volatility)
                volumes[i] = max(volumes[i], 100000)  # Ensure positive volume
            
            # Create DataFrame
            market_data[ticker] = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volumes,
                'Ticker': ticker
            }, index=dates)
    
    return market_data

def generate_bb_signal(data):
    """
    Generate Bollinger Bands (BB) crossing signal
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock price data
        
    Returns:
    --------
    pandas.Series
        Series with BB crossing signals (1 for buy, 0 for no signal)
    """
    # Calculate Bollinger Bands
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['STDDEV'] = data['Close'].rolling(window=20).std()
    data['UpperBand'] = data['MA20'] + (data['STDDEV'] * 2)
    data['LowerBand'] = data['MA20'] - (data['STDDEV'] * 2)
    
    # Generate BB crossing signal
    data['PrevClose'] = data['Close'].shift(1)
    data['PrevLowerBand'] = data['LowerBand'].shift(1)
    
    # Buy signal: Close crosses above the lower band
    data['BB_Signal'] = 0
    data.loc[(data['PrevClose'] < data['PrevLowerBand']) & 
             (data['Close'] > data['LowerBand']), 'BB_Signal'] = 1
    
    return data['BB_Signal']

def generate_macd_signal(data):
    """
    Generate Moving Average Convergence Divergence (MACD) crossing signal
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock price data
        
    Returns:
    --------
    pandas.Series
        Series with MACD crossing signals (1 for buy, 0 for no signal)
    """
    # Calculate MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Generate MACD crossing signal
    data['PrevMACD'] = data['MACD'].shift(1)
    data['PrevSignal'] = data['Signal_Line'].shift(1)
    
    # Buy signal: MACD crosses above the signal line
    data['MACD_Signal'] = 0
    data.loc[(data['PrevMACD'] < data['PrevSignal']) & 
             (data['MACD'] > data['Signal_Line']), 'MACD_Signal'] = 1
    
    return data['MACD_Signal']

def generate_rsi_signal(data):
    """
    Generate Relative Strength Index (RSI) crossing signal
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock price data
        
    Returns:
    --------
    pandas.Series
        Series with RSI crossing signals (1 for buy, 0 for no signal)
    """
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Calculate up and down moves
    up = delta.copy()
    up[up < 0] = 0
    down = -delta.copy()
    down[down < 0] = 0
    
    # Calculate RSI
    avg_up = up.rolling(window=14).mean()
    avg_down = down.rolling(window=14).mean()
    
    rs = avg_up / avg_down
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Generate RSI crossing signal
    data['PrevRSI'] = data['RSI'].shift(1)
    
    # Buy signal: RSI crosses above 30 (oversold)
    data['RSI_Signal'] = 0
    data.loc[(data['PrevRSI'] < 30) & (data['RSI'] > 30), 'RSI_Signal'] = 1
    
    return data['RSI_Signal']

def create_candlestick_image(data, window_size, idx, image_size=(30, 30), image_type='candlestick'):
    """
    Create a candlestick image for the specified window of data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock price data
    window_size : int
        Size of the window to include in the image
    idx : int
        Index of the end of the window
    image_size : tuple
        Size of the output image (width, height)
    image_type : str
        Type of image to create ('candlestick', 'line', 'width_varying', 'prev_close', 'volume')
        
    Returns:
    --------
    numpy.ndarray
        Grayscale image array
    """
    # Create a figure without display
    fig = Figure(figsize=(5, 5), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # Get window data
    window_data = data.iloc[idx-window_size+1:idx+1]
    
    # Different visualization types
    if image_type == 'line':
        # Line plot with Close data only
        ax.plot(range(len(window_data)), window_data['Close'], color='black')
    
    elif image_type == 'width_varying':
        # Candlestick with varying width to indicate time direction
        for i, (date, row) in enumerate(window_data.iterrows()):
            width = 0.2 + 0.5 * i / window_size  # Width increases with time
            if row['Open'] > row['Close']:
                color = 'black'
            else:
                color = 'white'
            
            # Draw rectangle for body
            rect = plt.Rectangle((i-width/2, min(row['Open'], row['Close'])), 
                                width, abs(row['Open'] - row['Close']),
                                color=color, alpha=1, edgecolor='black')
            ax.add_patch(rect)
            
            # Draw whiskers
            ax.plot([i, i], [row['Low'], min(row['Open'], row['Close'])], color='black')
            ax.plot([i, i], [max(row['Open'], row['Close']), row['High']], color='black')
    
    elif image_type == 'prev_close':
        # Candlestick with previous Close
        for i, (date, row) in enumerate(window_data.iterrows()):
            width = 0.6
            if row['Open'] > row['Close']:
                color = 'black'
            else:
                color = 'white'
            
            # Draw rectangle for body
            rect = plt.Rectangle((i-width/2, min(row['Open'], row['Close'])), 
                                width, abs(row['Open'] - row['Close']),
                                color=color, alpha=1, edgecolor='black')
            ax.add_patch(rect)
            
            # Draw whiskers
            ax.plot([i, i], [row['Low'], min(row['Open'], row['Close'])], color='black')
            ax.plot([i, i], [max(row['Open'], row['Close']), row['High']], color='black')
            
            # Draw previous Close
            if i > 0:
                prev_close = window_data.iloc[i-1]['Close']
                ax.plot([i-width/2, i+width/2], [prev_close, prev_close], color='darkgrey', linewidth=1)
    
    elif image_type == 'volume':
        # Candlestick with width varying by volume
        max_volume = window_data['Volume'].max()
        for i, (date, row) in enumerate(window_data.iterrows()):
            width = 0.2 + 0.6 * row['Volume'] / max_volume if max_volume > 0 else 0.6  # Width based on volume
            if row['Open'] > row['Close']:
                color = 'black'
            else:
                color = 'white'
            
            # Draw rectangle for body
            rect = plt.Rectangle((i-width/2, min(row['Open'], row['Close'])), 
                                width, abs(row['Open'] - row['Close']),
                                color=color, alpha=1, edgecolor='black')
            ax.add_patch(rect)
            
            # Draw whiskers
            ax.plot([i, i], [row['Low'], min(row['Open'], row['Close'])], color='black')
            ax.plot([i, i], [max(row['Open'], row['Close']), row['High']], color='black')
    
    else:  # Default: standard candlestick
        # Candlestick chart with OHLC data
        for i, (date, row) in enumerate(window_data.iterrows()):
            width = 0.6
            if row['Open'] > row['Close']:
                color = 'black'
            else:
                color = 'white'
            
            # Draw rectangle for body
            rect = plt.Rectangle((i-width/2, min(row['Open'], row['Close'])), 
                                width, abs(row['Open'] - row['Close']),
                                color=color, alpha=1, edgecolor='black')
            ax.add_patch(rect)
            
            # Draw whiskers
            ax.plot([i, i], [row['Low'], min(row['Open'], row['Close'])], color='black')
            ax.plot([i, i], [max(row['Open'], row['Close']), row['High']], color='black')
    
    # Remove axes, ticks, labels
    ax.axis('off')
    
    # Set tight layout
    fig.tight_layout(pad=0)
    
    # Convert figure to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, pad_inches=0, bbox_inches='tight')
    buf.seek(0)
    
    # Open image and convert to grayscale
    img = Image.open(buf).convert('L')
    
    # Resize image
    img = img.resize(image_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    plt.close(fig)
    
    return img_array

def create_dataset(data, signal_name, window_size, image_size=(30, 30), image_type='candlestick', sample_size=None):
    """
    Create a dataset of candlestick images and labels
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock price data
    signal_name : str
        Name of the signal column in data
    window_size : int
        Size of the window to include in the image
    image_size : tuple
        Size of the output image
    image_type : str
        Type of image to create
    sample_size : int or None
        Number of samples to take for each class (if None, use all)
        
    Returns:
    --------
    tuple
        (X, y) where X is a list of images and y is a list of labels
    """
    # Remove NaN values
    data = data.dropna()
    
    # Initialize lists for images and labels
    X = []
    y = []
    
    # Get indices where signal is 1 (buy)
    buy_indices = data[data[signal_name] == 1].index
    
    # Get indices where signal is 0 (no buy)
    no_buy_indices = data[data[signal_name] == 0].index
    
    # If sample_size is provided, sample from buy and no_buy indices
    if sample_size is not None:
        if len(buy_indices) > sample_size:
            buy_indices = np.random.choice(buy_indices, sample_size, replace=False)
        
        if len(no_buy_indices) > sample_size:
            no_buy_indices = np.random.choice(no_buy_indices, sample_size, replace=False)
    
    # Create images for buy signals
    for idx in tqdm(buy_indices, desc=f"Creating buy images ({image_type})"):
        # Find position of index in data
        pos = data.index.get_loc(idx)
        
        # Skip if not enough data for the window
        if pos < window_size - 1:
            continue
        
        # Create image
        img = create_candlestick_image(data, window_size, pos, image_size, image_type)
        
        # Add to dataset
        X.append(img)
        y.append(1)
    
    # Create images for no buy signals
    for idx in tqdm(no_buy_indices, desc=f"Creating no-buy images ({image_type})"):
        # Find position of index in data
        pos = data.index.get_loc(idx)
        
        # Skip if not enough data for the window
        if pos < window_size - 1:
            continue
        
        # Create image
        img = create_candlestick_image(data, window_size, pos, image_size, image_type)
        
        # Add to dataset
        X.append(img)
        y.append(0)
    
    return np.array(X), np.array(y)

# PyTorch CNN Model
class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        
        # Get dimensions from input shape
        self.height, self.width = input_shape
        
        # Define layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Calculate size after convolutions and pooling
        conv_output_size = (self.height // 4) * (self.width // 4) * 32
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Reshape input if it's not in the right shape
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
            
        # Convolutional layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x

def train_cnn(model, train_loader, val_loader, epochs=10, lr=0.001):
    """
    Train a CNN model
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model to train
    train_loader : torch.utils.data.DataLoader
        Training data loader
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    epochs : int
        Number of epochs to train
    lr : float
        Learning rate
        
    Returns:
    --------
    tuple
        (trained model, training history)
    """
    # Move model to device
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, history

def predict_cnn(model, loader):
    """
    Make predictions using a CNN model
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained PyTorch model
    loader : torch.utils.data.DataLoader
        Data loader
        
    Returns:
    --------
    tuple
        (predictions, targets)
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            
            # Make predictions
            outputs = model(inputs)
            preds = (outputs > 0.5).float().cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.numpy())
    
    return np.array(all_preds), np.array(all_targets)

def train_and_evaluate_models(X_train, y_train, X_test, y_test, image_size):
    """
    Train and evaluate multiple machine learning models
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training images
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Test images
    y_test : numpy.ndarray
        Test labels
    image_size : tuple
        Size of input images
        
    Returns:
    --------
    tuple
        (models, results) where models is a dictionary of trained models
        and results is a dictionary of evaluation metrics
    """
    # Flatten images for traditional ML models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Standardize data for ML models
    scaler = StandardScaler()
    X_train_flat_scaled = scaler.fit_transform(X_train_flat)
    X_test_flat_scaled = scaler.transform(X_test_flat)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'KNN': KNeighborsClassifier(),
        'Linear SVM': LinearSVC(max_iter=10000),
        'RBF SVM': SVC(gamma='scale'),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Extra Trees': ExtraTreesClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging': BaggingClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_flat_scaled, y_train)
        
        y_pred = model.predict(X_test_flat_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        trained_models[name] = model
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Prepare data for PyTorch CNN
    X_train_tensor = torch.tensor(X_train.reshape(-1, 1, image_size[0], image_size[1]), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.reshape(-1, 1, image_size[0], image_size[1]), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Split train data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train and evaluate CNN
    print("Training CNN...")
    cnn = CNN(image_size)
    cnn, history = train_cnn(cnn, train_loader, val_loader, epochs=10, lr=0.001)
    
    # Evaluate CNN
    y_pred, y_true = predict_cnn(cnn, test_loader)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    results['CNN'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    trained_models['CNN'] = cnn
    
    print(f"CNN - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Create and evaluate voting classifier
    print("Training Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', models['Logistic Regression']),
            ('rf', models['Random Forest']),
            ('gb', models['Gradient Boosting'])
        ],
        voting='hard'
    )
    
    voting_clf.fit(X_train_flat_scaled, y_train)
    y_pred = voting_clf.predict(X_test_flat_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results['Voting Classifier'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    trained_models['Voting Classifier'] = voting_clf
    
    print(f"Voting Classifier - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return trained_models, results

def plot_results(results, metric='accuracy', title=None):
    """
    Plot results for different models
    
    Parameters:
    -----------
    results : dict
        Dictionary of evaluation metrics for different models
    metric : str
        Metric to plot ('accuracy', 'precision', 'recall', or 'f1')
    title : str or None
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Sort models by metric
    sorted_models = sorted(results.keys(), key=lambda x: results[x][metric], reverse=True)
    
    # Get metric values
    metric_values = [results[model][metric] for model in sorted_models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal bars
    ax.barh(sorted_models, metric_values, color='skyblue')
    
    # Add values to bars
    for i, v in enumerate(metric_values):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    # Set labels and title
    ax.set_xlabel(metric.capitalize())
    ax.set_ylabel('Model')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Model Comparison by {metric.capitalize()}')
    
    # Set x-axis limits
    ax.set_xlim(0, 1.1)
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig

def simulate_trading(data, model, window_size, image_size=(30, 30), image_type='candlestick', is_cnn=False):
    """
    Simulate trading using the trained model
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock price data
    model : object
        Trained model with predict method
    window_size : int
        Size of the window to include in the image
    image_size : tuple
        Size of the output image
    image_type : str
        Type of image to create
    is_cnn : bool
        Whether the model is a CNN
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with simulation results
    """
    # Initialize results DataFrame
    results = data.copy()
    results['Signal'] = 0
    results['Position'] = 0
    results['Returns'] = 0
    results['Strategy_Returns'] = 0
    
    # Remove NaN values
    results = results.dropna()
    
    # Calculate daily returns
    results['Returns'] = results['Close'].pct_change()
    
    # Loop through data
    for i in range(window_size, len(results)):
        # Create image for current window
        img = create_candlestick_image(results, window_size, i, image_size, image_type)
        
        # Make prediction
        if is_cnn:
            # Prepare image for CNN
            img_tensor = torch.tensor(img.reshape(1, 1, image_size[0], image_size[1]), dtype=torch.float32).to(device)
            
            # Set model to evaluation mode
            model.eval()
            
            # Make prediction
            with torch.no_grad():
                output = model(img_tensor)
                signal = (output > 0.5).cpu().numpy()[0][0]
        else:
            # Flatten image for traditional ML models
            img_flat = img.reshape(1, -1)
            
            # Scale the features
            scaler = StandardScaler()
            img_flat_scaled = scaler.fit_transform(img_flat)
            
            # Predict
            signal = model.predict(img_flat_scaled)[0]
        
        # Store signal
        results.iloc[i, results.columns.get_loc('Signal')] = signal
        
        # Update position (0 or 1)
        if signal == 1:
            results.iloc[i, results.columns.get_loc('Position')] = 1
    
    # Calculate strategy returns (position from previous day affects today's return)
    results['Strategy_Returns'] = results['Position'].shift(1) * results['Returns']
    results['Strategy_Returns'] = results['Strategy_Returns'].fillna(0)
    
    # Calculate cumulative returns
    results['Cumulative_Returns'] = (1 + results['Returns']).cumprod() - 1
    results['Cumulative_Strategy_Returns'] = (1 + results['Strategy_Returns']).cumprod() - 1
    
    return results

def plot_training_history(history):
    """
    Plot training history for CNN
    
    Parameters:
    -----------
    history : dict
        Dictionary with training history
        
    Returns:
    --------
    tuple
        (accuracy figure, loss figure)
    """
    # Plot accuracy
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(history['train_acc'], label='Train Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(history['train_loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    return fig1, fig2

def plot_example_patterns(data, signal_name, window_size, image_size=(30, 30), image_type='candlestick'):
    """
    Plot example patterns for the specified signal
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock price data
    signal_name : str
        Name of the signal column in data
    window_size : int
        Size of the window to include in the image
    image_size : tuple
        Size of the output image
    image_type : str
        Type of image to create
        
    Returns:
    --------
    tuple
        (positive example figure, negative example figure)
    """
    # Find positive signal example
    positive_indices = data[data[signal_name] == 1].index
    
    if len(positive_indices) > 0:
        pos_idx = positive_indices[0]
        pos_pos = data.index.get_loc(pos_idx)
        
        # Skip if not enough data for the window
        if pos_pos < window_size - 1:
            positive_indices = positive_indices[1:]
            if len(positive_indices) > 0:
                pos_idx = positive_indices[0]
                pos_pos = data.index.get_loc(pos_idx)
            else:
                pos_pos = None
        
        if pos_pos is not None and pos_pos >= window_size - 1:
            # Create positive example image
            fig_pos, axs_pos = plt.subplots(1, 2, figsize=(12, 5))
            
            # Get window data
            window_data = data.iloc[pos_pos-window_size+1:pos_pos+1]
            
            # Plot original candlestick chart
            for i, (date, row) in enumerate(window_data.iterrows()):
                width = 0.6
                if row['Open'] > row['Close']:
                    color = 'black'
                else:
                    color = 'white'
                
                # Draw rectangle for body
                rect = plt.Rectangle((i-width/2, min(row['Open'], row['Close'])), 
                                     width, abs(row['Open'] - row['Close']),
                                     color=color, alpha=1, edgecolor='black')
                axs_pos[0].add_patch(rect)
                
                # Draw whiskers
                axs_pos[0].plot([i, i], [row['Low'], min(row['Open'], row['Close'])], color='black')
                axs_pos[0].plot([i, i], [max(row['Open'], row['Close']), row['High']], color='black')
            
            # Add signal annotation
            axs_pos[0].set_title(f'Positive {signal_name} Example')
            
            # Plot the transformed image
            img = create_candlestick_image(data, window_size, pos_pos, image_size, image_type)
            axs_pos[1].imshow(img, cmap='gray')
            axs_pos[1].set_title(f'Transformed Image ({image_type})')
            axs_pos[1].axis('off')
            
            fig_pos.tight_layout()
        else:
            fig_pos = None
    else:
        fig_pos = None
    
    # Find negative signal example
    negative_indices = data[data[signal_name] == 0].index
    
    if len(negative_indices) > 0:
        neg_idx = negative_indices[0]
        neg_pos = data.index.get_loc(neg_idx)
        
        # Skip if not enough data for the window
        if neg_pos < window_size - 1:
            negative_indices = negative_indices[1:]
            if len(negative_indices) > 0:
                neg_idx = negative_indices[0]
                neg_pos = data.index.get_loc(neg_idx)
            else:
                neg_pos = None
        
        if neg_pos is not None and neg_pos >= window_size - 1:
            # Create negative example image
            fig_neg, axs_neg = plt.subplots(1, 2, figsize=(12, 5))
            
            # Get window data
            window_data = data.iloc[neg_pos-window_size+1:neg_pos+1]
            
            # Plot original candlestick chart
            for i, (date, row) in enumerate(window_data.iterrows()):
                width = 0.6
                if row['Open'] > row['Close']:
                    color = 'black'
                else:
                    color = 'white'
                
                # Draw rectangle for body
                rect = plt.Rectangle((i-width/2, min(row['Open'], row['Close'])), 
                                     width, abs(row['Open'] - row['Close']),
                                     color=color, alpha=1, edgecolor='black')
                axs_neg[0].add_patch(rect)
                
                # Draw whiskers
                axs_neg[0].plot([i, i], [row['Low'], min(row['Open'], row['Close'])], color='black')
                axs_neg[0].plot([i, i], [max(row['Open'], row['Close']), row['High']], color='black')
            
            # Add signal annotation
            axs_neg[0].set_title(f'Negative {signal_name} Example')
            
            # Plot the transformed image
            img = create_candlestick_image(data, window_size, neg_pos, image_size, image_type)
            axs_neg[1].imshow(img, cmap='gray')
            axs_neg[1].set_title(f'Transformed Image ({image_type})')
            axs_neg[1].axis('off')
            
            fig_neg.tight_layout()
        else:
            fig_neg = None
    else:
        fig_neg = None
    
    return fig_pos, fig_neg

def main():
    """Main function to run the entire pipeline"""
    
    # Simulate market data
    print("Simulating market data...")
    sp500_tickers = [f'SIM{i:03d}' for i in range(20)]  # 20 simulated tickers
    
    # Parameters
    start_date = '2010-01-01'
    end_date = '2017-12-31'
    test_start_date = '2018-01-01'
    test_end_date = '2018-12-31'
    image_size = (30, 30)
    sample_size = 50  # Number of samples per class per ticker
    
    # Image types to test
    image_types = ['candlestick', 'line', 'width_varying', 'prev_close', 'volume']
    
    # Signal types and their corresponding window sizes
    signals = {
        'BB_Signal': 20,
        'MACD_Signal': 26,
        'RSI_Signal': 27
    }
    
    # Simulate market data for training period
    all_data = simulate_market_data(sp500_tickers, start_date, end_date)
    
    # Generate signals for each stock
    for ticker, data in all_data.items():
        data['BB_Signal'] = generate_bb_signal(data)
        data['MACD_Signal'] = generate_macd_signal(data)
        data['RSI_Signal'] = generate_rsi_signal(data)
        
        # Count signals
        for signal_name in signals.keys():
            num_signals = data[signal_name].sum()
            print(f"{ticker} - {signal_name}: {num_signals} signals")
    
    # Plot an example of each signal type
    for signal_name, window_size in signals.items():
        for ticker, data in all_data.items():
            fig_pos, fig_neg = plot_example_patterns(data, signal_name, window_size)
            if fig_pos is not None and fig_neg is not None:
                fig_pos.savefig(f'example_{signal_name}_positive.png')
                fig_neg.savefig(f'example_{signal_name}_negative.png')
                plt.close(fig_pos)
                plt.close(fig_neg)
                break
    
    # Let's test different image types with BB signal as example
    signal_name = 'BB_Signal'
    window_size = signals[signal_name]
    
    image_type_results = {}
    
    for image_type in image_types:
        print(f"\nTesting {image_type} images with {signal_name}...")
        
        # Create dataset
        X_all = []
        y_all = []
        
        for ticker, data in all_data.items():
            X, y = create_dataset(data, signal_name, window_size, image_size, image_type, sample_size)
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
        
        if len(X_all) > 0:
            X_all = np.vstack(X_all)
            y_all = np.concatenate(y_all)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
            
            # Train and evaluate models
            trained_models, results = train_and_evaluate_models(X_train, y_train, X_test, y_test, image_size)
            
            # Store results
            image_type_results[image_type] = results
            
            # Plot results
            fig = plot_results(results, 'accuracy', f'Model Accuracy - {image_type} Images')
            plt.savefig(f'accuracy_{image_type}.png')
            plt.close(fig)
        else:
            print(f"No data available for {image_type} images")
    
    # Compare image types
    print("\nComparing image types...")
    
    # Get average accuracy for each image type
    avg_accuracy = {}
    for image_type, results in image_type_results.items():
        avg_accuracy[image_type] = np.mean([results[model]['accuracy'] for model in results])
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(avg_accuracy.keys(), avg_accuracy.values(), color='skyblue')
    plt.title('Average Accuracy by Image Type')
    plt.xlabel('Image Type')
    plt.ylabel('Average Accuracy')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('image_type_comparison.png')
    plt.close()
    
    # Test different signals with the best image type
    best_image_type = max(avg_accuracy, key=avg_accuracy.get)
    print(f"\nTesting different signals with {best_image_type} images...")
    
    signal_results = {}
    
    for signal_name, window_size in signals.items():
        print(f"\nTesting {signal_name}...")
        
        # Create dataset
        X_all = []
        y_all = []
        
        for ticker, data in all_data.items():
            X, y = create_dataset(data, signal_name, window_size, image_size, best_image_type, sample_size)
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
        
        if len(X_all) > 0:
            X_all = np.vstack(X_all)
            y_all = np.concatenate(y_all)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
            
            # Train and evaluate models
            trained_models, results = train_and_evaluate_models(X_train, y_train, X_test, y_test, image_size)
            
            # Store results
            signal_results[signal_name] = results
            
            # Plot results
            fig = plot_results(results, 'accuracy', f'Model Accuracy - {signal_name}')
            plt.savefig(f'accuracy_{signal_name}.png')
            plt.close(fig)
        else:
            print(f"No data available for {signal_name}")
    
    # Compare signals
    print("\nComparing signals...")
    
    # Get average accuracy for each signal
    avg_accuracy = {}
    for signal_name, results in signal_results.items():
        avg_accuracy[signal_name] = np.mean([results[model]['accuracy'] for model in results])
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(avg_accuracy.keys(), avg_accuracy.values(), color='skyblue')
    plt.title('Average Accuracy by Signal Type')
    plt.xlabel('Signal Type')
    plt.ylabel('Average Accuracy')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('signal_comparison.png')
    plt.close()
    
    # Test on out-of-sample data (2018)
    print("\nTesting on out-of-sample data (2018)...")
    
    # Get best signal and image type
    best_signal = max(avg_accuracy, key=avg_accuracy.get)
    window_size = signals[best_signal]
    
    # Get best model for the best signal
    best_model_name = max(signal_results[best_signal], key=lambda x: signal_results[best_signal][x]['accuracy'])
    
    print(f"Best signal: {best_signal}")
    print(f"Best image type: {best_image_type}")
    print(f"Best model: {best_model_name}")
    
    # Train final model
    X_all = []
    y_all = []
    
    for ticker, data in all_data.items():
        X, y = create_dataset(data, best_signal, window_size, image_size, best_image_type, sample_size)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
    
    if len(X_all) > 0:
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)
        
        # Check if best model is CNN
        is_cnn = best_model_name == 'CNN'
        
        if is_cnn:
            # Prepare data for CNN
            X_tensor = torch.tensor(X_all.reshape(-1, 1, image_size[0], image_size[1]), dtype=torch.float32)
            y_tensor = torch.tensor(y_all, dtype=torch.float32)
            
            # Create dataset and loader
            dataset = TensorDataset(X_tensor, y_tensor)
            batch_size = 32
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Split for training and validation
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Create and train CNN
            final_model = CNN(image_size)
            final_model, history = train_cnn(final_model, train_loader, val_loader, epochs=10, lr=0.001)
            
            # Plot training history
            fig_acc, fig_loss = plot_training_history(history)
            fig_acc.savefig('cnn_accuracy.png')
            fig_loss.savefig('cnn_loss.png')
            plt.close(fig_acc)
            plt.close(fig_loss)
        else:
            # Flatten images for traditional ML models
            X_flat = X_all.reshape(X_all.shape[0], -1)
            
            # Standardize data
            scaler = StandardScaler()
            X_flat_scaled = scaler.fit_transform(X_flat)
            
            # Create and train model
            if best_model_name == 'Logistic Regression':
                final_model = LogisticRegression(max_iter=1000)
            elif best_model_name == 'Naive Bayes':
                final_model = GaussianNB()
            elif best_model_name == 'LDA':
                final_model = LinearDiscriminantAnalysis()
            elif best_model_name == 'QDA':
                final_model = QuadraticDiscriminantAnalysis()
            elif best_model_name == 'KNN':
                final_model = KNeighborsClassifier()
            elif best_model_name == 'Linear SVM':
                final_model = LinearSVC(max_iter=10000)
            elif best_model_name == 'RBF SVM':
                final_model = SVC(gamma='scale')
            elif best_model_name == 'Decision Tree':
                final_model = DecisionTreeClassifier()
            elif best_model_name == 'Random Forest':
                final_model = RandomForestClassifier()
            elif best_model_name == 'Extra Trees':
                final_model = ExtraTreesClassifier()
            elif best_model_name == 'AdaBoost':
                final_model = AdaBoostClassifier()
            elif best_model_name == 'Bagging':
                final_model = BaggingClassifier()
            elif best_model_name == 'Gradient Boosting':
                final_model = GradientBoostingClassifier()
            elif best_model_name == 'Voting Classifier':
                final_model = VotingClassifier(
                    estimators=[
                        ('lr', LogisticRegression(max_iter=1000)),
                        ('rf', RandomForestClassifier()),
                        ('gb', GradientBoostingClassifier())
                    ],
                    voting='hard'
                )
            
            # Train model
            final_model.fit(X_flat_scaled, y_all)
        
        # Simulate test data for 2018
        test_data = simulate_market_data(sp500_tickers, test_start_date, test_end_date)
        
        # Generate signals and simulate trading
        test_results = {}
        
        for ticker, data in test_data.items():
            # Generate signal
            if best_signal == 'BB_Signal':
                data[best_signal] = generate_bb_signal(data)
            elif best_signal == 'MACD_Signal':
                data[best_signal] = generate_macd_signal(data)
            elif best_signal == 'RSI_Signal':
                data[best_signal] = generate_rsi_signal(data)
            
            # Simulate trading
            sim_results = simulate_trading(data, final_model, window_size, image_size, best_image_type, is_cnn)
            
            # Store results
            test_results[ticker] = sim_results
        
        # Plot trading results
        plt.figure(figsize=(12, 8))
        
        for ticker, results in test_results.items():
            # Only plot a few tickers for clarity
            if sp500_tickers.index(ticker) < 5:
                plt.plot(results.index, results['Cumulative_Strategy_Returns'], label=f'{ticker} Strategy')
                plt.plot(results.index, results['Cumulative_Returns'], linestyle='--', alpha=0.5, label=f'{ticker} Buy & Hold')
        
        plt.title('Trading Simulation Results (2018)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('trading_simulation.png')
        plt.close()
        
        # Calculate performance metrics
        print("\nTrading Simulation Performance:")
        
        all_strategy_returns = []
        all_bh_returns = []
        
        for ticker, results in test_results.items():
            # Calculate annualized return
            days = (results.index[-1] - results.index[0]).days
            annual_factor = 252 / days if days > 0 else 1
            
            # Strategy metrics
            strategy_return = results['Cumulative_Strategy_Returns'].iloc[-1]
            strategy_annual_return = (1 + strategy_return) ** annual_factor - 1
            strategy_volatility = results['Strategy_Returns'].std() * np.sqrt(252)
            strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility != 0 else 0
            
            # Buy & Hold metrics
            bh_return = results['Cumulative_Returns'].iloc[-1]
            bh_annual_return = (1 + bh_return) ** annual_factor - 1
            bh_volatility = results['Returns'].std() * np.sqrt(252)
            bh_sharpe = bh_annual_return / bh_volatility if bh_volatility != 0 else 0
            
            all_strategy_returns.append(strategy_return)
            all_bh_returns.append(bh_return)
            
            print(f"\n{ticker}:")
            print(f"Strategy: Return = {strategy_return:.4f}, Annual Return = {strategy_annual_return:.4f}, "
                  f"Volatility = {strategy_volatility:.4f}, Sharpe = {strategy_sharpe:.4f}")
            print(f"Buy & Hold: Return = {bh_return:.4f}, Annual Return = {bh_annual_return:.4f}, "
                  f"Volatility = {bh_volatility:.4f}, Sharpe = {bh_sharpe:.4f}")
        
        # Calculate average performance
        avg_strategy_return = np.mean(all_strategy_returns)
        avg_bh_return = np.mean(all_bh_returns)
        
        print(f"\nAverage Strategy Return: {avg_strategy_return:.4f}")
        print(f"Average Buy & Hold Return: {avg_bh_return:.4f}")
        print(f"Outperformance: {avg_strategy_return - avg_bh_return:.4f}")
    else:
        print("No data available for training the final model")

if __name__ == "__main__":
    main()