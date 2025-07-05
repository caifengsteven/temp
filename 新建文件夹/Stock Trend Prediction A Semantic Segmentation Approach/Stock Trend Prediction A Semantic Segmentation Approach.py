import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, 
    BatchNormalization, Concatenate, Activation
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to generate simulated stock price data
def generate_simulated_data(num_stocks=5, days=2000, volatility=0.01):
    """
    Generate simulated stock price data for multiple stocks.
    
    Parameters:
    -----------
    num_stocks : int
        Number of stocks to simulate
    days : int
        Number of days to simulate
    volatility : float
        Daily volatility of stock prices
    
    Returns:
    --------
    data : dict
        Dictionary containing simulated price data for each stock
    """
    data = {}
    
    for i in range(num_stocks):
        # Initial price
        price = 100
        
        # Arrays to store prices
        open_prices = []
        high_prices = []
        low_prices = []
        close_prices = []
        
        for _ in range(days):
            # Daily return (random)
            daily_return = np.random.normal(0, volatility)
            
            # Generate random intraday volatility
            intraday_vol = volatility * np.random.uniform(0.5, 1.5)
            
            # Open price (slightly different from previous close)
            open_price = price * (1 + np.random.normal(0, volatility/2))
            
            # Close price
            close_price = open_price * (1 + daily_return)
            
            # High and low prices
            price_range = abs(open_price - close_price) + open_price * intraday_vol
            if open_price > close_price:
                high_price = open_price + price_range * np.random.uniform(0, 0.8)
                low_price = close_price - price_range * np.random.uniform(0, 0.2)
            else:
                high_price = close_price + price_range * np.random.uniform(0, 0.2)
                low_price = open_price - price_range * np.random.uniform(0, 0.8)
            
            # Ensure high is highest and low is lowest
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Store prices
            open_prices.append(open_price)
            high_prices.append(high_price)
            low_prices.append(low_price)
            close_prices.append(close_price)
            
            # Update price for next day
            price = close_price
        
        # Create DataFrame
        dates = pd.date_range(start='2010-01-01', periods=days)
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices
        }, index=dates)
        
        data[f'Stock_{i+1}'] = df
    
    return data

# Function to fetch real stock data using yfinance
def fetch_real_data(tickers, start_date='2010-01-01', end_date='2023-01-01'):
    """
    Fetch real stock price data using yfinance.
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    start_date : str
        Start date for data retrieval
    end_date : str
        End date for data retrieval
    
    Returns:
    --------
    data : dict
        Dictionary containing price data for each stock
    """
    data = {}
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                data[ticker] = df[['Open', 'High', 'Low', 'Close']]
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return data

# Function to preprocess data and create price frames and labels
def create_price_frames(data, input_days=20, output_days=20, stride=1):
    """
    Create price frames and corresponding label frames from stock data.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing price data for each stock
    input_days : int
        Number of days in each input price frame
    output_days : int
        Number of days in each output label frame
    stride : int
        Stride between consecutive frames
    
    Returns:
    --------
    X : numpy.ndarray
        Array of input price frames
    y : numpy.ndarray
        Array of output label frames
    """
    X = []
    y = []
    
    for stock, df in data.items():
        # Make sure we have enough data
        if len(df) < input_days + output_days:
            continue
        
        # Iterate through the dataframe with stride
        for i in range(0, len(df) - input_days - output_days + 1, stride):
            # Extract input price frame
            input_frame = df.iloc[i:i+input_days][['Open', 'High', 'Low', 'Close']].values
            
            # Extract output price frame
            output_frame = df.iloc[i+input_days:i+input_days+output_days][['Open', 'High', 'Low', 'Close']].values
            
            # Min-max scaling for input frame
            min_val = np.min(input_frame)
            max_val = np.max(input_frame)
            input_frame_scaled = (input_frame - min_val) / (max_val - min_val)
            
            # Create label frame (1 if price increased, 0 otherwise)
            label_frame = np.zeros_like(output_frame, dtype=np.int32)
            # Compare each output price with the last day of input
            last_day_prices = input_frame[-1]
            for j in range(output_days):
                for k in range(4):  # 4 price types (OHLC)
                    label_frame[j, k] = 1 if output_frame[j, k] > last_day_prices[k] else 0
            
            X.append(input_frame_scaled)
            y.append(label_frame)
    
    return np.array(X), np.array(y)

# ASPP block implementation
def ASPP_block(input_tensor, filters):
    """
    Implement Atrous Spatial Pyramid Pooling (ASPP) block.
    
    Parameters:
    -----------
    input_tensor : tf.Tensor
        Input tensor
    filters : int
        Number of filters for convolutions
    
    Returns:
    --------
    output : tf.Tensor
        Output tensor after ASPP
    """
    # Parallel atrous convolutions with different rates
    conv_1x1 = Conv2D(filters, (1, 1), padding='same', use_bias=False)(input_tensor)
    conv_3x3_rate1 = Conv2D(filters, (3, 3), padding='same', dilation_rate=(1, 1), use_bias=False)(input_tensor)
    conv_3x3_rate2 = Conv2D(filters, (3, 3), padding='same', dilation_rate=(2, 2), use_bias=False)(input_tensor)
    conv_3x3_rate3 = Conv2D(filters, (3, 3), padding='same', dilation_rate=(3, 3), use_bias=False)(input_tensor)
    
    # Concatenate all branches
    concat = Concatenate()([conv_1x1, conv_3x3_rate1, conv_3x3_rate2, conv_3x3_rate3])
    
    # Apply 1x1 convolution to reduce channels
    output = Conv2D(filters, (1, 1), padding='same', use_bias=False)(concat)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    
    return output

# Build the proposed semantic segmentation model
def build_semantic_segmentation_model(input_shape, output_shape, num_frames=1):
    """
    Build the proposed semantic segmentation model with parallel encoders.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of a single input frame (height, width, channels)
    output_shape : tuple
        Shape of a single output frame (height, width, channels)
    num_frames : int
        Number of input frames
    
    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras model
    """
    # Create parallel encoders for each input frame
    inputs = []
    encoder_outputs = []
    
    for i in range(num_frames):
        # Input for this frame
        input_frame = Input(shape=input_shape)
        inputs.append(input_frame)
        
        # Encoder
        # Stage 1
        conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_frame)
        conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
        aspp1 = ASPP_block(conv1, 32)
        pool1 = MaxPooling2D(pool_size=(2, 2))(aspp1)
        
        # Stage 2
        conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
        conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
        aspp2 = ASPP_block(conv2, 64)
        pool2 = MaxPooling2D(pool_size=(2, 2))(aspp2)
        
        # Stage 3
        conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
        conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
        aspp3 = ASPP_block(conv3, 128)
        
        # Store encoder outputs
        encoder_outputs.append([aspp1, aspp2, aspp3])
    
    # Concatenate features from parallel encoders for each scale
    fused_features = []
    
    for scale in range(3):
        scale_features = [encoder_outputs[i][scale] for i in range(num_frames)]
        if len(scale_features) == 1:
            fused = scale_features[0]
        else:
            fused = Concatenate()(scale_features)
            fused = Conv2D(encoder_outputs[0][scale].shape[-1], (1, 1), padding='same')(fused)
            fused = BatchNormalization()(fused)
            fused = Activation('relu')(fused)
        fused_features.append(fused)
    
    # Decoder
    up1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(fused_features[2])
    concat1 = Concatenate()([up1, fused_features[1]])
    conv6 = Conv2D(64, (3, 3), padding='same', activation='relu')(concat1)
    conv6 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv6)
    
    up2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv6)
    concat2 = Concatenate()([up2, fused_features[0]])
    conv7 = Conv2D(32, (3, 3), padding='same', activation='relu')(concat2)
    conv7 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv7)
    
    # Output layer
    output = Conv2D(output_shape[-1], (1, 1), activation='sigmoid')(conv7)
    
    # Create and compile model
    model = Model(inputs=inputs if num_frames > 1 else inputs[0], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    return model

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test, num_frames=1, days_to_evaluate=[1, 5, 10, 20]):
    """
    Evaluate model performance for specific days in the output horizon.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    X_test : numpy.ndarray
        Test input data
    y_test : numpy.ndarray
        Test labels
    num_frames : int
        Number of input frames
    days_to_evaluate : list
        List of days to evaluate individually
    
    Returns:
    --------
    results : dict
        Dictionary containing evaluation metrics
    """
    # Prepare inputs for prediction
    if num_frames > 1:
        # Split X_test into multiple inputs
        model_inputs = [X_test[:, i] for i in range(num_frames)]
    else:
        model_inputs = X_test
    
    # Make predictions
    y_pred = model.predict(model_inputs)
    
    # Binary predictions (threshold = 0.5)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Initialize results dictionary
    results = {
        'total': {},
        'days': {}
    }
    
    # Evaluate total performance
    y_test_flat = y_test.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    y_pred_binary_flat = y_pred_binary.reshape(-1)
    
    results['total']['accuracy'] = accuracy_score(y_test_flat, y_pred_binary_flat)
    results['total']['auc'] = roc_auc_score(y_test_flat, y_pred_flat)
    results['total']['precision'] = precision_score(y_test_flat, y_pred_binary_flat)
    results['total']['recall'] = recall_score(y_test_flat, y_pred_binary_flat)
    results['total']['f1'] = f1_score(y_test_flat, y_pred_binary_flat)
    
    # Evaluate performance for specific days
    for day in days_to_evaluate:
        if day <= y_test.shape[1]:  # Check if day is within range
            day_idx = day - 1  # Convert to 0-based index
            
            y_test_day = y_test[:, day_idx].reshape(-1)
            y_pred_day = y_pred[:, day_idx].reshape(-1)
            y_pred_binary_day = y_pred_binary[:, day_idx].reshape(-1)
            
            results['days'][day] = {
                'accuracy': accuracy_score(y_test_day, y_pred_binary_day),
                'auc': roc_auc_score(y_test_day, y_pred_day),
                'precision': precision_score(y_test_day, y_pred_binary_day),
                'recall': recall_score(y_test_day, y_pred_binary_day),
                'f1': f1_score(y_test_day, y_pred_binary_day)
            }
    
    return results

# Trading strategy implementation
def implement_trading_strategy(model, data, stock, input_days=20, output_days=20, num_frames=1, 
                              initial_capital=10000, commission=0.001):
    """
    Implement a trading strategy based on model predictions.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    data : dict
        Dictionary containing price data
    stock : str
        Stock to trade
    input_days : int
        Number of days in input frame
    output_days : int
        Number of days in output frame
    num_frames : int
        Number of input frames
    initial_capital : float
        Initial capital for trading
    commission : float
        Trading commission as a fraction of transaction value
    
    Returns:
    --------
    results : dict
        Dictionary containing trading results
    """
    df = data[stock].copy()
    
    # Initialize trading variables
    cash = initial_capital
    shares = 0
    positions = []
    equity = []
    
    # Generate predictions for each day in test period
    for i in range(len(df) - input_days - output_days):
        # Extract input frame
        input_frame = df.iloc[i:i+input_days][['Open', 'High', 'Low', 'Close']].values
        
        # Min-max scaling
        min_val = np.min(input_frame)
        max_val = np.max(input_frame)
        input_frame_scaled = (input_frame - min_val) / (max_val - min_val)
        
        # Reshape for model input
        X = np.array([input_frame_scaled])
        
        # Make prediction
        if num_frames > 1:
            # In this simple example, we're using the same frame multiple times
            # In practice, you'd use consecutive frames
            model_inputs = [X] * num_frames
            prediction = model.predict(model_inputs)[0]
        else:
            prediction = model.predict(X)[0]
        
        # Extract prediction for next day's close price
        next_day_close_pred = prediction[0, 3]  # First day, close price
        
        # Trading logic
        current_price = df.iloc[i+input_days]['Close']
        next_price = df.iloc[i+input_days+1]['Close'] if i+input_days+1 < len(df) else current_price
        
        # Buy signal: predicted uptrend and not already holding shares
        if next_day_close_pred > 0.5 and shares == 0:
            # Calculate how many shares to buy
            shares_to_buy = int(cash / (current_price * (1 + commission)))
            
            if shares_to_buy > 0:
                # Execute buy
                cash -= shares_to_buy * current_price * (1 + commission)
                shares = shares_to_buy
                positions.append(('buy', i+input_days, current_price, shares))
        
        # Sell signal: predicted downtrend and holding shares
        elif next_day_close_pred <= 0.5 and shares > 0:
            # Execute sell
            cash += shares * current_price * (1 - commission)
            positions.append(('sell', i+input_days, current_price, shares))
            shares = 0
        
        # Calculate equity
        equity.append(cash + shares * next_price)
    
    # Final portfolio value
    final_value = cash + shares * df.iloc[-1]['Close']
    
    # Calculate returns
    returns = (final_value / initial_capital - 1) * 100
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    equity_array = np.array(equity)
    daily_returns = equity_array[1:] / equity_array[:-1] - 1
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    max_drawdown = np.max(drawdown) * 100
    
    # Results
    results = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'returns': returns,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'positions': positions,
        'equity_curve': equity
    }
    
    return results

# Function to plot equity curve
def plot_equity_curve(results, title):
    """
    Plot equity curve from trading results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing trading results
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(results['equity_curve'])
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.show()

# Main function to run the entire pipeline
def main():
    print("Stock Trend Prediction: A Semantic Segmentation Approach")
    print("-------------------------------------------------------")
    
    # Parameters
    use_real_data = False
    input_days = 20
    output_days = 20
    num_frames = 1  # Number of consecutive frames to use
    
    # Generate or fetch data
    if use_real_data:
        print("Fetching real stock data...")
        tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
        data = fetch_real_data(tickers)
    else:
        print("Generating simulated stock data...")
        data = generate_simulated_data(num_stocks=5, days=2000)
    
    # Create price frames and labels
    print("Creating price frames and labels...")
    X, y = create_price_frames(data, input_days=input_days, output_days=output_days)
    
    # Reshape X to (samples, height, width, channels)
    X = X.reshape(-1, X.shape[1], X.shape[2], 1)
    y = y.reshape(-1, y.shape[1], y.shape[2], 1)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Build and train model
    print("Building and training model...")
    input_shape = (input_days, 4, 1)
    output_shape = (output_days, 4, 1)
    
    model = build_semantic_segmentation_model(input_shape, output_shape, num_frames)
    
    # If using multiple frames, prepare inputs accordingly
    if num_frames > 1:
        # For simplicity, using the same frame multiple times
        # In practice, you'd use consecutive frames
        train_inputs = [X_train] * num_frames
        test_inputs = [X_test] * num_frames
    else:
        train_inputs = X_train
        test_inputs = X_test
    
    # Train model
    history = model.fit(
        train_inputs, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test, num_frames, days_to_evaluate=[1, 5, 10, 20])
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print("------------------")
    print(f"Total Accuracy: {results['total']['accuracy']:.4f}")
    print(f"Total AUC: {results['total']['auc']:.4f}")
    print(f"Total F1 Score: {results['total']['f1']:.4f}")
    
    for day, metrics in results['days'].items():
        print(f"\nDay {day}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
    # Implement trading strategy
    print("\nImplementing trading strategy...")
    stock_to_trade = list(data.keys())[0]  # First stock
    
    trading_results = implement_trading_strategy(
        model, data, stock_to_trade, 
        input_days=input_days, output_days=output_days, 
        num_frames=num_frames
    )
    
    # Print trading results
    print("\nTrading Results:")
    print("--------------")
    print(f"Initial Capital: ${trading_results['initial_capital']:.2f}")
    print(f"Final Value: ${trading_results['final_value']:.2f}")
    print(f"Returns: {trading_results['returns']:.2f}%")
    print(f"Sharpe Ratio: {trading_results['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {trading_results['max_drawdown']:.2f}%")
    print(f"Number of Trades: {len(trading_results['positions'])}")
    
    # Plot equity curve
    plot_equity_curve(trading_results, f"Equity Curve - {stock_to_trade}")
    
    # Compare with buy and hold strategy
    print("\nComparison with Buy and Hold Strategy:")
    print("-------------------------------------")
    
    stock_df = data[stock_to_trade]
    buy_hold_return = (stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[input_days] - 1) * 100
    
    print(f"Trading Strategy Return: {trading_results['returns']:.2f}%")
    print(f"Buy and Hold Return: {buy_hold_return:.2f}%")
    
    # Visualize example predictions
    print("\nVisualizing example predictions...")
    
    # Get a sample from test set
    sample_idx = np.random.randint(0, len(X_test))
    
    if num_frames > 1:
        sample_inputs = [X_test[sample_idx:sample_idx+1]] * num_frames
    else:
        sample_inputs = X_test[sample_idx:sample_idx+1]
    
    sample_pred = model.predict(sample_inputs)[0]
    sample_true = y_test[sample_idx]
    
    # Reshape for visualization
    sample_input = X_test[sample_idx].reshape(input_days, 4)
    sample_pred = sample_pred.reshape(output_days, 4)
    sample_true = sample_true.reshape(output_days, 4)
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot input
    im0 = axes[0].imshow(sample_input, aspect='auto', cmap='Blues')
    axes[0].set_title('Input Price Frame')
    axes[0].set_xlabel('Price Type (O, H, L, C)')
    axes[0].set_ylabel('Day')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot true output
    im1 = axes[1].imshow(sample_true, aspect='auto', cmap='RdYlGn')
    axes[1].set_title('True Price Trend')
    axes[1].set_xlabel('Price Type (O, H, L, C)')
    axes[1].set_ylabel('Day')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot predicted output
    im2 = axes[2].imshow(sample_pred, aspect='auto', cmap='RdYlGn')
    axes[2].set_title('Predicted Price Trend')
    axes[2].set_xlabel('Price Type (O, H, L, C)')
    axes[2].set_ylabel('Day')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()